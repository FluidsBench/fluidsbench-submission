"""AirfRANS participant orchestration. Scientific reductions remain in the pinned evaluator.

The downloadable support, native meshes, predictions, runtime and evaluator are
bound to each campaign. Completed cases are installed atomically and immutable.
"""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from reference.hosting import support_url

import numpy as np

from reference.native_participant import (
    SAFE_ID,
    WorkflowError,
    campaign_lock,
    digest,
    file_tree,
    load_json,
    write_json,
)
from reference.scoring_support import load_support_release, sha256_file
from reference.airfrans.profile_truth import resolve_case_set_truth, validate_release

ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = ROOT / "benchmark-specs/airfrans"
SPEC_PATH = SPEC_ROOT / "submission-spec.json"
PROFILE_PATH = SPEC_ROOT / "velocity-profiles-v1.json"
ARCHIVES = {
    "support-metadata.tar.gz": "native",
    "native-tables.tar": "native",
    "profile-truth.tar.gz": "profiles",
}
PREDICTION_KEYS = {"velocity", "pressure", "internal_points", "airfoil_points"}


def official_split(split_id):
    spec = load_json(SPEC_PATH)
    if (
        spec.get("intake_branch") != "main"
        or not spec["scoring_support"].get("submissions_open")
        or spec["scoring_support"]["status"] != "official"
    ):
        raise WorkflowError("AirfRANS official intake is not open")
    choices = {s["id"]: s for s in spec["splits"]}
    if split_id not in choices:
        raise WorkflowError(
            f"choose an enabled official split: {list(choices)}; Reynolds remains closed"
        )
    declared = choices[split_id]
    path = SPEC_ROOT / declared["index_file"]
    if sha256_file(path) != declared["sha256"]:
        raise WorkflowError("official split checksum differs")
    split = load_json(path)
    if (
        len(split["case_ids"]) != declared["case_count"]
        or len(set(split["case_ids"])) != declared["case_count"]
    ):
        raise WorkflowError("official split membership differs")
    return split


def bindings():
    spec = load_json(SPEC_PATH)
    return {
        "native": spec["scoring_support"]["manifest_sha256"],
        "profiles": spec["profile_definition"]["profile_ground_truth"][
            "manifest_sha256"
        ],
    }


def archive_plan():
    release = load_json(SPEC_ROOT / "official-release-binding.json")
    assets = {
        a["url"].rsplit("/", 1)[-1]: a
        for a in release["assets"]
        if a["url"].rsplit("/", 1)[-1] in ARCHIVES
    }
    if set(assets) != set(ARCHIVES):
        raise WorkflowError("approved archive descriptors are incomplete")
    return {name: {**entry, "url": support_url(entry["url"])} for name, entry in assets.items()}


def check_file(path, descriptor):
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != descriptor["byte_size"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise WorkflowError(f"file size or SHA-256 differs: {path}")


def unpack(archive, destination):
    """Only ordinary relative files; no links, devices, traversal or overwrites."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise WorkflowError(f"unsafe archive member: {member.name}")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise WorkflowError("archive member escapes destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out)
            else:
                raise WorkflowError(f"unsupported archive member: {member.name}")


def verify_support(root):
    root = Path(root).resolve() / "installed"
    receipt = load_json(root / "support-receipt.json")
    if (
        receipt.get("schema") != "fluidsbench-airfrans-installed-support-v1"
        or receipt.get("bindings") != bindings()
    ):
        raise WorkflowError("installed support belongs to a different dev contract")
    for kind, expected in bindings().items():
        if sha256_file(root / kind / "manifest.json") != expected:
            raise WorkflowError(f"{kind} manifest checksum differs")
        if file_tree(root / kind) != receipt["files"][kind]:
            raise WorkflowError(
                f"installed {kind} support changed; fetch into a new directory"
            )
    return receipt


def fetch_support(destination, *, offline=False, archive_dir=None, dry_run=False):
    plan = archive_plan()
    if dry_run:
        return {
            "total_bytes": sum(a["byte_size"] for a in plan.values()),
            "archives": plan,
        }
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(root):
        if (root / "installed/support-receipt.json").exists():
            verify_support(root)
            return {"status": "verified_cached_support", "support_root": str(root)}
        if (root / "installed").exists():
            raise WorkflowError(
                "incomplete support installation; choose a new destination"
            )
        archives = root / "archives"
        archives.mkdir(exist_ok=True)
        for name, descriptor in plan.items():
            target = archives / name
            if target.exists():
                check_file(target, descriptor)
                continue
            source = Path(archive_dir) / name if archive_dir else None
            partial = archives / (name + ".part")
            partial.unlink(missing_ok=True)
            try:
                if source and source.is_file():
                    check_file(source, descriptor)
                    shutil.copyfile(source, partial)
                elif offline:
                    raise WorkflowError(f"offline support archive missing: {name}")
                else:
                    request = urllib.request.Request(
                        support_url(descriptor["url"]),
                        headers={"User-Agent": "FluidsBench-AirfRANS"},
                    )
                    with urllib.request.urlopen(
                        request, timeout=120
                    ) as src, partial.open("xb") as out:
                        shutil.copyfileobj(src, out)
                check_file(partial, descriptor)
                partial.rename(target)
            finally:
                partial.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as temporary:
            stage = Path(temporary)
            for name, kind in ARCHIVES.items():
                unpack(archives / name, stage / kind)
            for kind, expected in bindings().items():
                if sha256_file(stage / kind / "manifest.json") != expected:
                    raise WorkflowError(f"downloaded {kind} manifest differs from dev")
            from scripts.publish_airfrans_official_release import verify_native

            native = verify_native(stage / "native")
            approved = load_json(
                SPEC_ROOT
                / "releases/airfrans-evaluation-v1/publication-validation.json"
            )
            if native != approved["native_support"]:
                raise WorkflowError(
                    "native support differs from the approved validation receipt"
                )
            validate_release(
                release_root=stage / "profiles",
                profile_definition_path=PROFILE_PATH,
                repository_root=ROOT,
            )
            receipt = {
                "schema": "fluidsbench-airfrans-installed-support-v1",
                "bindings": bindings(),
                "files": {
                    kind: file_tree(stage / kind) for kind in ("native", "profiles")
                },
            }
            write_json(stage / "support-receipt.json", receipt)
            stage.rename(root / "installed")
    return {"status": "installed_verified_support", "support_root": str(root)}


def load_entry(root):
    value = load_json(Path(root) / "entry.json")
    required = {"schema", "submission_id", "split_id"}
    if not required <= set(value) or set(value) - required - {
        "force_prediction_source"
    }:
        raise WorkflowError(
            "entry has missing or unknown keys; custom case lists and force values are forbidden"
        )
    if (
        value["schema"] != "fluidsbench-airfrans-entry-v1"
        or not isinstance(value["submission_id"], str)
        or not SAFE_ID.fullmatch(value["submission_id"])
    ):
        raise WorkflowError("invalid AirfRANS entry schema or submission ID")
    if value.get("force_prediction_source", "field_integrated") != "field_integrated":
        raise WorkflowError("only field_integrated forces are accepted")
    official_split(value["split_id"])
    return value


def validate_prediction(path, counts):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != PREDICTION_KEYS:
            raise WorkflowError(
                f"{path}: require exactly velocity, pressure, internal_points and airfoil_points; direct forces are forbidden"
            )
        arrays = {k: np.asarray(data[k]) for k in PREDICTION_KEYS}
    n, m = counts
    shapes = {
        "velocity": {(n, 2)},
        "pressure": {(n,), (n, 1)},
        "internal_points": {(n, 2), (n, 3)},
        "airfoil_points": {(m, 2), (m, 3)},
    }
    for name, value in arrays.items():
        if (
            value.shape not in shapes[name]
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise WorkflowError(f"{path}: invalid shape or non-finite {name}")
    return arrays


def mesh_paths(dataset_root, case_id):
    directory = Path(dataset_root) / case_id
    return {
        "internal_vtu": directory / f"{case_id}_internal.vtu",
        "aerofoil_vtp": directory / f"{case_id}_aerofoil.vtp",
    }


def evaluator_identity():
    paths = [
        *ROOT.glob("reference/*.py"),
        *ROOT.glob("reference/airfrans/*.py"),
        ROOT / "scripts/airfrans.py",
        ROOT / "scripts/derive_airfrans_predicted_fields.py",
        ROOT / "scripts/finalize_airfrans_metrics.py",
        ROOT / "scripts/assemble_airfrans_schema_v3_candidate.py",
        ROOT / "scripts/validate_submission.py",
        ROOT / "requirements-airfrans-evaluator.txt",
        ROOT / "requirements-airfrans-participant.txt",
        *ROOT.glob("examples/airfrans-profile-extraction/*.py"),
        *ROOT.glob("schemas/v3/*.json"),
        *ROOT.glob("benchmark-specs/airfrans/*.json"),
    ]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(set(paths))}


def extractor():
    path = ROOT / "examples/airfrans-profile-extraction/extract.py"
    module_spec = importlib.util.spec_from_file_location(
        "airfrans_participant_extractor", path
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def check_runtime():
    return extractor().validate_runtime(load_json(PROFILE_PATH), ROOT)


def preflight(entry_root, dataset_root, support_root):
    entry_root, dataset_root, support_root = [
        Path(p).expanduser().resolve() for p in (entry_root, dataset_root, support_root)
    ]
    entry = load_entry(entry_root)
    split = official_split(entry["split_id"])
    verify_support(support_root)
    support_root = support_root / "installed"
    predictions = entry_root / "predictions"
    expected = {f"{cid}.npz" for cid in split["case_ids"]}
    if not predictions.is_dir() or {p.name for p in predictions.iterdir()} != expected:
        raise WorkflowError(
            "prediction directory must contain exactly one NPZ per official test case"
        )
    release = load_support_release(
        support_root / "native/manifest.json", split["case_set_id"]
    )
    truth_paths = resolve_case_set_truth(
        support_root / "profiles", split["case_set_id"]
    )
    inputs = {}
    for cid, truth_path in zip(split["case_ids"], truth_paths, strict=True):
        truth = load_json(truth_path)
        if [c["case_id"] for c in truth["cases"]] != [cid]:
            raise WorkflowError("profile truth case order differs")
        meshes = mesh_paths(dataset_root, cid)
        native_hashes = truth["provenance"]["input_hashes"]
        for key, path in meshes.items():
            if (
                not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != native_hashes[key]
            ):
                raise WorkflowError(
                    f"native mesh missing or differs from published source: {path}"
                )
        counts = {
            s["support_id"]: s["entity_count"]
            for s in release.cases[cid]["support_instances"]
        }
        prediction = predictions / f"{cid}.npz"
        if prediction.is_symlink():
            raise WorkflowError("predictions cannot be symlinks")
        validate_prediction(
            prediction,
            (counts["two-dimensional-domain-native"], counts["airfoil-curve-native"]),
        )
        inputs[cid] = {
            "prediction_sha256": sha256_file(prediction),
            "meshes": native_hashes,
            "truth_sha256": sha256_file(truth_path),
        }
    return {
        "schema": "fluidsbench-airfrans-campaign-v1",
        "entry": entry,
        "case_ids": split["case_ids"],
        "case_set_id": split["case_set_id"],
        "bindings": bindings(),
        "inputs": inputs,
        "evaluator": evaluator_identity(),
    }


def evaluate_case(case_id, prediction_path, dataset_root, truth_path, output):
    """One native case in a short-lived worker; full meshes never accumulate."""
    import airfrans as af
    from scripts.derive_airfrans_predicted_fields import (
        derive_case,
        load_packed_prediction,
    )

    module = extractor()
    versions = module.validate_runtime(load_json(PROFILE_PATH), ROOT)
    prediction = load_packed_prediction(prediction_path)
    simulation = af.Simulation(root=str(dataset_root), name=case_id)
    output.mkdir(parents=True, exist_ok=True)
    case, summary = derive_case(simulation, case_id, prediction, output / "data")
    series, sampling = module.extract_profiles(
        simulation,
        load_json(PROFILE_PATH),
        reference=False,
        velocity=prediction["velocity"],
    )
    truth = load_json(truth_path)
    if sampling != truth["provenance"]["sampling"]["stations"]:
        raise WorkflowError(
            "predicted profile sampling differs from the published native geometry"
        )
    write_json(output / "case.json", case)
    write_json(output / "summary.json", summary)
    write_json(
        output / "profiles.json",
        {
            "schema_version": "1.0",
            "provenance": {"mode": "model_prediction", "versions": versions},
            "cases": [{"case_id": case_id, "series": series}],
        },
    )


def check_case_inputs(cid, campaign, entry_root, dataset_root):
    expected = campaign["inputs"][cid]
    prediction = Path(entry_root) / "predictions" / f"{cid}.npz"
    if sha256_file(prediction) != expected["prediction_sha256"]:
        raise WorkflowError(f"prediction changed during evaluation: {cid}")
    for name, path in mesh_paths(dataset_root, cid).items():
        if sha256_file(path) != expected["meshes"][name]:
            raise WorkflowError(f"native mesh changed during evaluation: {cid}")


def verify_case(path, campaign, cid):
    receipt = load_json(path / "checkpoint.json")
    if (
        receipt.get("campaign_sha256") != digest(campaign)
        or receipt.get("input") != campaign["inputs"][cid]
    ):
        raise WorkflowError(f"case checkpoint belongs to different inputs: {cid}")
    if file_tree(path, exclude=("checkpoint.json",)) != receipt["files"]:
        raise WorkflowError(
            f"saved case files changed: {cid}; use a new output directory"
        )
    return receipt


def reduce_campaign(output, campaign, support_root):
    from reference.airfrans_profiles import score_airfrans_velocity_profiles
    from scripts.finalize_airfrans_metrics import finalize

    definition = load_json(PROFILE_PATH)
    truth_paths = resolve_case_set_truth(
        support_root / "profiles", campaign["case_set_id"]
    )
    with tempfile.TemporaryDirectory(prefix=".reduce-", dir=output) as temporary:
        stage = Path(temporary)
        predicted, truth = [], []
        cases = []
        for cid, truth_path in zip(campaign["case_ids"], truth_paths, strict=True):
            if sha256_file(truth_path) != campaign["inputs"][cid]["truth_sha256"]:
                raise WorkflowError(f"profile truth changed during evaluation: {cid}")
            case_root = output / "cases" / cid
            verify_case(case_root, campaign, cid)
            case = load_json(case_root / "case.json")
            for support in case["files"]:
                source = case_root / support["file"]
                target = stage / "predicted-data" / cid / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                support["file"] = target.relative_to(stage).as_posix()
            cases.append(case)
            predicted.append(load_json(case_root / "profiles.json"))
            truth.append(load_json(truth_path))
            (stage / "profiles").mkdir(exist_ok=True)
            shutil.copyfile(
                case_root / "profiles.json", stage / "profiles" / f"{cid}.json"
            )
        support_manifest = load_json(support_root / "native/manifest.json")
        manifest = {
            "kind": "scored_predictions",
            "submission_id": campaign["entry"]["submission_id"],
            "dataset_id": "airfrans",
            "split_id": campaign["entry"]["split_id"],
            "case_set_id": campaign["case_set_id"],
            "support_release_id": support_manifest["release_id"],
            "support_manifest_sha256": campaign["bindings"]["native"],
            "case_count": len(cases),
            "cases": cases,
        }
        write_json(stage / "prediction-manifest.json", manifest)
        profile_score = score_airfrans_velocity_profiles(
            truth,
            predicted,
            station_ids=[s["id"] for s in definition["stations"]],
            quantity_ids=[q["id"] for q in definition["quantities"]],
            panel_id=definition["metric_binding"]["panel_id"],
            expected_case_ids=campaign["case_ids"],
            allow_partial_case_coverage=False,
        )
        profile_score["split_id"] = campaign["entry"]["split_id"]
        write_json(stage / "profile-score.json", profile_score)
        metrics = finalize(
            support_manifest_path=support_root / "native/manifest.json",
            case_set_id=campaign["case_set_id"],
            prediction_manifest_path=stage / "prediction-manifest.json",
            submission_id=campaign["entry"]["submission_id"],
            split_id=campaign["entry"]["split_id"],
            profile_score_path=stage / "profile-score.json",
        )
        write_json(stage / "metrics-cases.json", metrics)
        shutil.copyfile(
            support_root / "native/manifest.json", stage / "support-manifest.json"
        )
        # Retain truth for standalone reporting without a network connection.
        write_json(
            stage / "profile-truth.json",
            {"cases": [c for document in truth for c in document["cases"]]},
        )
        write_json(
            stage / "workflow.json",
            {
                "schema": "fluidsbench-airfrans-evaluation-v1",
                "campaign": campaign,
                "files": file_tree(stage),
                "case_checkpoints": {
                    cid: sha256_file(output / "cases" / cid / "checkpoint.json")
                    for cid in campaign["case_ids"]
                },
            },
        )
        stage.rename(output / "evaluation")


def verify_evaluation(root):
    root = Path(root).resolve()
    receipt = load_json(root / "workflow.json")
    campaign = receipt["campaign"]
    split = official_split(campaign["entry"]["split_id"])
    if (
        receipt.get("schema") != "fluidsbench-airfrans-evaluation-v1"
        or campaign["case_ids"] != split["case_ids"]
        or campaign["bindings"] != bindings()
        or campaign["evaluator"] != evaluator_identity()
    ):
        raise WorkflowError(
            "evaluation belongs to a different evaluator or dev contract"
        )
    if file_tree(root, exclude=("workflow.json",)) != receipt["files"]:
        raise WorkflowError("evaluation files changed; rerun in a new output directory")
    if load_json(root.parent / "campaign.json") != campaign:
        raise WorkflowError("campaign receipt differs")
    for cid in campaign["case_ids"]:
        case_root = root.parent / "cases" / cid
        verify_case(case_root, campaign, cid)
        if (
            sha256_file(case_root / "checkpoint.json")
            != receipt["case_checkpoints"][cid]
        ):
            raise WorkflowError("case checkpoint changed after reduction")
    return receipt


def evaluate_entry(entry_root, dataset_root, support_root, output, *, resume=False):
    entry_root, dataset_root, support_root, output = [
        Path(p).expanduser().resolve()
        for p in (entry_root, dataset_root, support_root, output)
    ]
    campaign = preflight(entry_root, dataset_root, support_root)
    campaign["runtime"] = check_runtime()
    support_root = support_root / "installed"
    truth_paths = dict(
        zip(
            campaign["case_ids"],
            resolve_case_set_truth(support_root / "profiles", campaign["case_set_id"]),
            strict=True,
        )
    )
    with campaign_lock(output):
        identity = output / "campaign.json"
        if identity.exists():
            if not resume or load_json(identity) != campaign:
                raise WorkflowError(
                    "use --resume with unchanged inputs, runtime and evaluator, or choose a new output directory"
                )
        else:
            (output / ".campaign.json").unlink(missing_ok=True)
            if any(p.name != ".lock" for p in output.iterdir()):
                raise WorkflowError("output is not an empty campaign directory")
            write_json(output / ".campaign.json", campaign)
            (output / ".campaign.json").rename(identity)
        if (output / "evaluation").exists():
            verify_evaluation(output / "evaluation")
            return {
                "status": "verified_completed_evaluation",
                "evaluation_root": str(output / "evaluation"),
            }
        (output / "cases").mkdir(exist_ok=True)
        for index, cid in enumerate(campaign["case_ids"], 1):
            case_root = output / "cases" / cid
            check_case_inputs(cid, campaign, entry_root, dataset_root)
            if case_root.exists():
                verify_case(case_root, campaign, cid)
                print(f"[{index}/{len(campaign['case_ids'])}] reused {cid}", flush=True)
                continue
            with tempfile.TemporaryDirectory(prefix=".case-", dir=output) as temporary:
                stage = Path(temporary)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/airfrans.py"),
                        "_evaluate-case",
                        "--case-id",
                        cid,
                        "--prediction",
                        str(entry_root / "predictions" / f"{cid}.npz"),
                        "--dataset-root",
                        str(dataset_root),
                        "--truth",
                        str(truth_paths[cid]),
                        "--output",
                        str(stage),
                    ],
                    check=True,
                )
                check_case_inputs(cid, campaign, entry_root, dataset_root)
                write_json(
                    stage / "checkpoint.json",
                    {
                        "campaign_sha256": digest(campaign),
                        "input": campaign["inputs"][cid],
                        "files": file_tree(stage),
                    },
                )
                stage.rename(case_root)
            print(f"[{index}/{len(campaign['case_ids'])}] evaluated {cid}", flush=True)
        expected_predictions = {f"{cid}.npz" for cid in campaign["case_ids"]}
        if {
            p.name for p in (entry_root / "predictions").iterdir()
        } != expected_predictions:
            raise WorkflowError("prediction inventory changed during evaluation")
        for cid in campaign["case_ids"]:
            check_case_inputs(cid, campaign, entry_root, dataset_root)
        reduce_campaign(output, campaign, support_root)
        verify_evaluation(output / "evaluation")
    return {"status": "evaluated", "evaluation_root": str(output / "evaluation")}


def verify_package(directory):
    from scripts import validate_submission as validator

    directory = Path(directory).resolve()
    file_tree(directory)  # Reject links before the schema reader follows paths.
    submission = load_json(directory / "submission.json")
    if submission.get("dataset_id") != "airfrans":
        raise WorkflowError("package is not AirfRANS")
    if directory.name != submission.get("submission_id"):
        raise WorkflowError("package directory must be named for its submission_id")
    official_split(submission.get("split_id"))
    manifest = validator.manifest_with_benchmark_contract(
        validator.load_json(validator.MANIFEST_PATH)
    )
    errors, stats = validator.validate_submission_file(
        directory / "submission.json",
        manifest,
        contributor_stage=True,
        require_registered_path=False,
    )
    if errors:
        raise WorkflowError("package validation failed:\n" + "\n".join(errors))
    return {"status": "valid_package", "counts": stats, "output": str(directory)}


def package(
    evaluation_root, config_path, discretization_cases, output, checkpoints=None
):
    from scripts.assemble_airfrans_schema_v3_candidate import assemble, find_blockers

    root, output = Path(evaluation_root).resolve(), Path(output).resolve()
    campaign = verify_evaluation(root)["campaign"]
    config = load_json(config_path)
    required = {
        "submission_id",
        "split_id",
        "result_revision",
        "participant",
        "reproducibility",
        "spatial_discretization",
    }
    if not required <= set(config) or set(config) - required - {"_comment"}:
        raise WorkflowError("package config keys differ from the participant template")
    if any(
        config[key] != campaign["entry"][key] for key in ("submission_id", "split_id")
    ):
        raise WorkflowError("package configuration must match the evaluated entry")
    blockers = find_blockers(config)
    if blockers:
        raise WorkflowError(
            "complete the participant metadata:\n" + "\n".join(blockers)
        )
    methods = config["participant"]["methodology"]
    if methods.get("record_kind") != "submitter_reported":
        raise WorkflowError(
            "a submission requires actual submitter_reported methodology"
        )
    declared = {c["id"]: c["sha256"] for c in methods["checkpoints"]}
    for key, path in (checkpoints or {}).items():
        if key not in declared or sha256_file(Path(path)) != declared[key]:
            raise WorkflowError(f"checkpoint bytes do not match declared digest: {key}")
    spec = load_json(SPEC_PATH)
    config["case_set_id"] = campaign["case_set_id"]
    config["release_bindings"] = {
        "scoring_support": spec["scoring_support"],
        "profile_ground_truth": spec["profile_definition"]["profile_ground_truth"],
    }
    spatial = copy.deepcopy(config["spatial_discretization"])
    if set(spatial) != {"training", "inference"}:
        raise WorkflowError(
            "spatial_discretization must contain the actual training and inference descriptions"
        )
    spatial.update(
        {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
            "schema_version": "1.0",
            "submission_id": config["submission_id"],
            "dataset_id": "airfrans",
            "split_id": config["split_id"],
            "scoring_support_release_id": spec["scoring_support"]["release_id"],
            "scoring_support_manifest_sha256": bindings()["native"],
            "case_manifest": {
                "format": "jsonl",
                "file": "discretization/cases.jsonl",
                "case_count": len(campaign["case_ids"]),
                "sha256": "",
            },
        }
    )
    rows = [
        json.loads(line)
        for line in Path(discretization_cases).read_text().splitlines()
        if line.strip()
    ]
    if [row["case_id"] for row in rows] != campaign["case_ids"]:
        raise WorkflowError(
            "discretization records must cover the exact ordered official split"
        )
    for row in rows:
        if any(
            row.get(key) != spatial[key]
            for key in ("submission_id", "dataset_id", "split_id")
        ):
            raise WorkflowError("discretization record identity differs")
    if output.exists():
        raise WorkflowError("package output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".package-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / output.name
        assemble(
            config=config,
            scoring_support_manifest_path=root / "support-manifest.json",
            metrics_cases_path=root / "metrics-cases.json",
            predicted_profiles_dir=root / "profiles",
            profile_score_path=root / "profile-score.json",
            output_dir=stage,
            spec_path=SPEC_PATH,
            intake=True,
            discretization_documents=(spatial, rows),
        )
        verify_package(stage)
        stage.rename(output)
    return {
        "status": "validated_package",
        "output": str(output),
    }


def report(evaluation_root, case_id, output):
    root = Path(evaluation_root).resolve()
    campaign = verify_evaluation(root)["campaign"]
    if case_id not in campaign["case_ids"]:
        raise WorkflowError("case is not in the evaluated split")
    truth = next(
        c
        for c in load_json(root / "profile-truth.json")["cases"]
        if c["case_id"] == case_id
    )
    prediction = load_json(root / "profiles" / f"{case_id}.json")["cases"][0]
    rows = [
        "<!doctype html><meta charset='utf-8'><title>AirfRANS evaluation</title>",
        "<style>body{font:16px system-ui;max-width:1000px;margin:2rem}td,th{padding:.3rem;text-align:left}svg{width:100%;max-width:640px;border:1px solid #ccc}</style>",
        f"<h1>AirfRANS / {html.escape(case_id)}</h1><p>Evaluation. Blue: truth. Orange: prediction.</p>",
        "<h2>Complete-split metrics</h2><table>",
    ]
    for key, value in load_json(root / "metrics-cases.json")["metric_values"].items():
        rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        )
    rows.append("</table><h2>Case profiles</h2>")
    truths = {(s["station_id"], s["quantity_id"]): s for s in truth["series"]}
    for series in prediction["series"]:
        key = (series["station_id"], series["quantity_id"])
        x, y, t = series["coordinate"], series["prediction"], truths[key]["prediction"]
        lo, hi = min(*y, *t), max(*y, *t)
        rows.append(
            f"<h3>{html.escape(' / '.join(key))}</h3><svg viewBox='0 0 640 240' role='img' aria-label='Truth and prediction'>"
        )
        for values, color in ((t, "#1671b8"), (y, "#d96b00")):
            points = " ".join(
                f"{20+(a-min(x))/(max(x)-min(x) or 1)*600:.3f},{220-(b-lo)/(hi-lo or 1)*200:.3f}"
                for a, b in zip(x, values, strict=True)
            )
            rows.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='1.5' points='{points}'/>"
            )
        rows.append("</svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write("\n".join(rows))
    return {
        "status": "report_written",
        "output": str(output),
        "profile_count": len(prediction["series"]),
    }
