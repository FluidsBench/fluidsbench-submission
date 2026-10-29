"""Participant orchestration around the existing closed DrivAerML evaluator.

Official splits and field-integrated forces only. Case checkpoints are published
atomically and bound to the evaluator, contracts and complete prediction bytes.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reference.drivaerml import dev_profiles as profiles
from reference.drivaerml.prediction_chunks import load_prediction_chunk_manifest
from reference.drivaerml.source import load_native_source_pin
from scripts.assemble_drivaerml_schema_v3_candidate import load_json

ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = ROOT / "benchmark-specs/drivaerml"
SPEC = SPEC_ROOT / "submission-spec.json"
NATIVE_PIN = SPEC_ROOT / "proposal/native-source-pin.json"
SCOPES = ("surface_only", "surface_and_volume")
ENTRY_SCHEMA = "fluidsbench-drivaerml-entry-v1"
CHECKPOINT_SCHEMA = "fluidsbench-drivaerml-case-checkpoint-v2"


class WorkflowError(ValueError):
    """An entry cannot be evaluated or resumed reproducibly."""


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def official_split(split_id):
    spec = load_json(SPEC)
    matches = [s for s in spec["splits"] if s["id"] == split_id]
    if len(matches) != 1:
        raise WorkflowError(
            f"split_id must be one of the official splits: {[s['id'] for s in spec['splits']]}"
        )
    item = matches[0]
    path = SPEC_ROOT / item["index_file"]
    if profiles.sha256(path) != item["sha256"]:
        raise WorkflowError("official split SHA-256 differs")
    return load_json(path)


def load_entry(entry_root):
    entry = load_json(Path(entry_root) / "entry.json")
    required = {"schema", "submission_id", "split_id", "prediction_scope"}
    if not required <= set(entry) or set(entry) - required - {
        "force_prediction_source"
    }:
        raise WorkflowError(
            "entry.json has missing or unknown keys; custom case lists and split overrides are forbidden"
        )
    if entry["schema"] != ENTRY_SCHEMA or entry["prediction_scope"] not in SCOPES:
        raise WorkflowError("entry schema or prediction_scope differs")
    if not isinstance(entry["submission_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{0,159}", entry["submission_id"]
    ):
        raise WorkflowError("submission_id must be a safe lowercase identifier")
    if entry.get("force_prediction_source", "field_integrated") != "field_integrated":
        raise WorkflowError(
            "only field_integrated forces are accepted; direct coefficients are forbidden"
        )
    official_split(entry["split_id"])
    return entry


def support_root(path=None):
    return (
        Path(
            path
            or os.environ.get("FLUIDSBENCH_DRIVAERML_TRUTH_ROOT")
            or Path.home() / ".cache/fluidsbench/drivaerml" / profiles.INDEX_SHA256
        )
        .expanduser()
        .resolve()
    )


def fetch_support(split_id, destination=None, *, offline=False):
    """Idempotently retain only the published chunks needed by this split."""
    cases = official_split(split_id)["case_ids"]
    root = support_root(destination)
    index = profiles.retained_json(
        "index.json", profiles.INDEX_SHA256, root=root, offline=offline
    )
    locations = {row["case_id"]: row for row in index["case_locations"]}
    chunks = {}
    for case_id in cases:
        if case_id not in locations:
            raise WorkflowError(f"profile support omits {case_id}")
        row = locations[case_id]
        chunks[row["chunk_path"]] = row["chunk_sha256"]
    for path, sha in sorted(chunks.items()):
        profiles.retained_json(path, sha, root=root, offline=offline)
    return {
        "root": str(root),
        "case_count": len(cases),
        "chunk_count": len(chunks),
        "index_sha256": profiles.INDEX_SHA256,
    }


def native_files(case, scope):
    files = [case.boundary, case.surface_cell_area]
    if scope == "surface_and_volume":
        files.extend(case.volume_parts)
    return files


def fetch_data(split_id, destination, scope, *, dry_run=False):
    if scope not in SCOPES:
        raise WorkflowError("invalid prediction scope")
    split = official_split(split_id)
    pin = load_native_source_pin(NATIVE_PIN)
    files = [
        item for cid in split["case_ids"] for item in native_files(pin.case(cid), scope)
    ]
    # A local plan works on an offline machine and never starts a large download.
    plan = {
        "repository": pin.repository_id,
        "revision": pin.repository_revision,
        "prediction_scope": scope,
        "case_count": len(split["case_ids"]),
        "total_bytes": sum(f.size_bytes for f in files),
        "files": [f.path.as_posix() for f in files],
    }
    if not dry_run:
        hf = shutil.which("hf")
        if hf is None:
            raise WorkflowError(
                "hf is unavailable; install requirements-drivaerml-participant.txt"
            )
        subprocess.run(
            [
                hf,
                "download",
                pin.repository_id,
                *plan["files"],
                "--type",
                "dataset",
                "--revision",
                pin.repository_revision,
                "--local-dir",
                str(Path(destination).expanduser().resolve()),
            ],
            check=True,
        )
    return plan


def prediction_manifests(
    entry_root, entry, case_id, native_case, *, maximum_chunk_rows=1_000_000
):
    case_root = Path(entry_root) / "cases" / case_id
    expected = {"surface"} | (
        {"volume"} if entry["prediction_scope"] == "surface_and_volume" else set()
    )
    if not case_root.is_dir() or {p.name for p in case_root.iterdir()} != expected:
        raise WorkflowError(
            f"{case_id}: expected only {sorted(expected)} prediction directories; direct force files are forbidden"
        )
    result = {}
    for domain in sorted(expected):
        manifest = load_prediction_chunk_manifest(case_root / domain / "manifest.json")
        if (
            manifest.case_id != case_id
            or manifest.support_id != domain + "_native_cells"
        ):
            raise WorkflowError(
                f"{case_id}/{domain}: manifest case or support identity differs"
            )
        if (
            domain == "surface"
            and manifest.total_row_count != native_case.surface_cell_area.element_count
        ):
            raise WorkflowError(
                f"{case_id}: surface row count differs from native support"
            )
        if any(c.row_count > maximum_chunk_rows for c in manifest.chunks):
            raise WorkflowError(
                f"{case_id}/{domain}: prediction chunk exceeds the row limit"
            )
        result[domain] = manifest
    return result


def preflight(
    entry_root,
    dataset_root,
    truth_root=None,
    *,
    offline=False,
    maximum_chunk_rows=1_000_000,
):
    entry_root, dataset_root = Path(entry_root), Path(dataset_root)
    entry = load_entry(entry_root)
    split = official_split(entry["split_id"])
    cases_root = entry_root / "cases"
    if not cases_root.is_dir() or {p.name for p in cases_root.iterdir()} != set(
        split["case_ids"]
    ):
        raise WorkflowError(
            "prediction cases must exactly match the selected official test split"
        )
    pin = load_native_source_pin(NATIVE_PIN)
    count = 0
    for cid in split["case_ids"]:
        native = pin.case(cid)
        prediction_manifests(
            entry_root, entry, cid, native, maximum_chunk_rows=maximum_chunk_rows
        )
        for item in native_files(native, entry["prediction_scope"]):
            path = dataset_root / item.path
            if not path.is_file() or path.stat().st_size != item.size_bytes:
                raise WorkflowError(
                    f"missing or wrong-sized native file: {path}; run fetch-data"
                )
            count += 1
    support = fetch_support(entry["split_id"], truth_root, offline=offline)
    return {
        "status": "preflight_passed",
        "entry": entry,
        "case_count": len(split["case_ids"]),
        "native_file_count": count,
        "profile_support": support,
        "prediction_arrays": "checked by the evaluator before each case completes",
    }


def implementation_identity():
    paths = list((ROOT / "reference").rglob("*.py"))
    paths += list((ROOT / "scripts").glob("*.py"))
    paths += list(SPEC_ROOT.glob("*.json"))
    paths += list((SPEC_ROOT / "splits").glob("*.json"))
    paths += [NATIVE_PIN]
    return digest({str(p.relative_to(ROOT)): profiles.sha256(p) for p in sorted(paths)})


def check_runtime():
    import numpy as np

    try:
        import vtk
    except ImportError as error:
        raise WorkflowError(
            "install the pinned participant requirements or use the container"
        ) from error
    if (
        platform.system(),
        platform.python_version(),
        np.__version__,
        vtk.vtkVersion.GetVTKVersion(),
    ) != ("Linux", "3.12.13", "2.2.6", "9.5.2"):
        raise WorkflowError(
            "evaluation requires Linux, Python 3.12.13, NumPy 2.2.6 and VTK 9.5.2; use Dockerfile.drivaerml"
        )


def case_inputs(entry, manifests, implementation, maximum_chunk_rows):
    return {
        "entry": entry,
        "implementation": implementation,
        "truth_index": profiles.INDEX_SHA256,
        "maximum_chunk_rows": maximum_chunk_rows,
        "predictions": {
            key: {"manifest": m.sha256, "chunks": [c.sha256 for c in m.chunks]}
            for key, m in manifests.items()
        },
    }


def verify_prediction_bytes(manifests):
    for manifest in manifests.values():
        if profiles.sha256(manifest.path) != manifest.sha256:
            raise WorkflowError("prediction manifest changed")
        for chunk in manifest.chunks:
            if profiles.sha256(chunk.path) != chunk.sha256:
                raise WorkflowError(f"prediction chunk SHA-256 differs: {chunk.path}")


def verify_checkpoint(directory, inputs):
    receipt = load_json(directory / "checkpoint.json")
    if receipt.get("schema") != CHECKPOINT_SCHEMA or receipt.get("inputs") != inputs:
        raise WorkflowError(
            f"{directory.name}: checkpoint inputs or evaluator changed; choose a new output directory"
        )
    expected = {"core.json", "profiles.json"}
    if set(receipt.get("outputs", {})) != expected:
        raise WorkflowError("checkpoint output coverage differs")
    for name in expected:
        if profiles.sha256(directory / name) != receipt["outputs"][name]:
            raise WorkflowError(f"checkpoint output changed: {directory / name}")


def run_case(
    entry,
    case_id,
    manifests,
    native,
    dataset_root,
    truth_root,
    staging,
    maximum_chunk_rows,
):
    args = [
        sys.executable,
        str(ROOT / "scripts/evaluate_drivaerml_candidate_case.py"),
        "--case-id",
        case_id,
        "--native-source-pin",
        str(NATIVE_PIN),
        "--dataset-root",
        str(dataset_root),
        "--surface-area-npy",
        str(dataset_root / native.surface_cell_area.path),
        "--prediction-scope",
        entry["prediction_scope"],
        "--surface-prediction-manifest",
        str(manifests["surface"].path),
        "--output",
        str(staging / "core.json"),
        "--profile-output",
        str(staging / "profiles.json"),
        "--maximum-prediction-chunk-rows",
        str(maximum_chunk_rows),
    ]
    if entry["prediction_scope"] == "surface_and_volume":
        args += [
            "--multipart",
            "--volume-prediction-manifest",
            str(manifests["volume"].path),
        ]
    env = {**os.environ, "FLUIDSBENCH_DRIVAERML_TRUTH_ROOT": str(truth_root)}
    subprocess.run(args, check=True, env=env)


def reduce_cases(entry, completed, staging, truth_root):
    # Existing reducer expects run_N.json in separate core/profile directories.
    for domain, filename in [("core", "core.json"), ("profiles", "profiles.json")]:
        folder = staging / domain
        folder.mkdir()
        for cid, path in completed.items():
            shutil.copyfile(path / filename, folder / f"{cid}.json")
    args = [
        sys.executable,
        str(ROOT / "scripts/score_drivaerml_candidate_dataset.py"),
        "--split-id",
        entry["split_id"],
        "--submission-id",
        entry["submission_id"],
        "--core-evidence-dir",
        str(staging / "core"),
        "--diagnostic-evidence-dir",
        str(staging / "profiles"),
        "--output",
        str(staging / "candidate-dataset-evidence.json"),
        "--regional-diagnostics-output",
        str(staging / "regional-diagnostics.json"),
        "--schema-v3-case-metrics-output",
        str(staging / "metrics/cases.json"),
        "--schema-v3-profile-output-dir",
        str(staging / "package-profiles"),
    ]
    subprocess.run(
        args,
        check=True,
        env={**os.environ, "FLUIDSBENCH_DRIVAERML_TRUTH_ROOT": str(truth_root)},
    )


def evaluate_entry(
    entry_root,
    dataset_root,
    output,
    truth_root=None,
    *,
    resume=False,
    offline=False,
    maximum_chunk_rows=1_000_000,
):
    check_runtime()
    entry_root, dataset_root, output = [
        Path(p).expanduser().resolve() for p in (entry_root, dataset_root, output)
    ]
    if output.exists() and any(output.iterdir()) and not resume:
        raise WorkflowError(
            "output is not empty; use --resume or choose a new output directory"
        )
    if maximum_chunk_rows < 1 or maximum_chunk_rows > 1_000_000:
        raise WorkflowError("maximum_chunk_rows must be between 1 and 1000000")
    checked = preflight(
        entry_root,
        dataset_root,
        truth_root,
        offline=offline,
        maximum_chunk_rows=maximum_chunk_rows,
    )
    entry = checked["entry"]
    truth_root = Path(checked["profile_support"]["root"])
    pin = load_native_source_pin(NATIVE_PIN)
    implementation = implementation_identity()
    cases = official_split(entry["split_id"])["case_ids"]
    output.mkdir(parents=True, exist_ok=True)
    case_root = output / "cases"
    case_root.mkdir(exist_ok=True)
    if {p.name for p in case_root.iterdir()} - set(cases):
        raise WorkflowError(
            "checkpoint directory contains cases outside the official split"
        )
    completed, reused = {}, 0
    for index, cid in enumerate(cases, 1):
        native = pin.case(cid)
        manifests = prediction_manifests(
            entry_root, entry, cid, native, maximum_chunk_rows=maximum_chunk_rows
        )
        inputs = case_inputs(entry, manifests, implementation, maximum_chunk_rows)
        destination = case_root / cid
        if destination.exists():
            verify_checkpoint(destination, inputs)
            verify_prediction_bytes(manifests)
            reused += 1
            print(
                f"[{index}/{len(cases)}] {cid}: verified checkpoint",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(cases)}] {cid}: evaluating", file=sys.stderr, flush=True
            )
            # Publish the core/profile pair together; interruption cannot leave
            # a half-written case that requires manual deletion to retry.
            with tempfile.TemporaryDirectory(prefix=".case-", dir=output) as temporary:
                staging = Path(temporary) / cid
                staging.mkdir()
                run_case(
                    entry,
                    cid,
                    manifests,
                    native,
                    dataset_root,
                    truth_root,
                    staging,
                    maximum_chunk_rows,
                )
                verify_prediction_bytes(manifests)
                if implementation_identity() != implementation:
                    raise WorkflowError("evaluator changed during evaluation")
                write_json(
                    staging / "checkpoint.json",
                    {
                        "schema": CHECKPOINT_SCHEMA,
                        "inputs": inputs,
                        "outputs": {
                            name: profiles.sha256(staging / name)
                            for name in ["core.json", "profiles.json"]
                        },
                    },
                )
                staging.rename(destination)
        completed[cid] = destination
    final = output / "evaluation"
    if final.exists():
        retained = load_json(final / "workflow.json")
        if (
            retained.get("entry") != entry
            or retained.get("implementation") != implementation
        ):
            raise WorkflowError("completed evaluation identity differs")
        expected = retained.get("outputs", {})
        actual = {
            p.relative_to(final).as_posix(): profiles.sha256(p)
            for p in final.rglob("*")
            if p.is_file() and p.name != "workflow.json"
        }
        if actual != expected:
            raise WorkflowError("completed evaluation files changed")
    else:
        with tempfile.TemporaryDirectory(prefix=".reduce-", dir=output) as temporary:
            staging = Path(temporary) / "evaluation"
            staging.mkdir()
            reduce_cases(entry, completed, staging, truth_root)
            files = {
                p.relative_to(staging).as_posix(): profiles.sha256(p)
                for p in staging.rglob("*")
                if p.is_file()
            }
            write_json(
                staging / "workflow.json",
                {"entry": entry, "implementation": implementation, "outputs": files},
            )
            staging.rename(final)
    return {
        "status": "complete_dev_evaluation",
        "case_count": len(cases),
        "reused_cases": reused,
        "output": str(final),
    }


def verify_package(directory, truth_root=None):
    from reference.drivaerml.pre_release import (
        build_registry_entry,
        package_tree_binding,
    )

    directory = Path(directory).expanduser().resolve()
    if load_json(ROOT / "benchmark-specs/drivaerml/submission-spec.json").get("status") == "official":
        from scripts.validate_submission import validate_submission_file

        prior_truth_root = os.environ.get("FLUIDSBENCH_DRIVAERML_TRUTH_ROOT")
        try:
            if truth_root:
                os.environ["FLUIDSBENCH_DRIVAERML_TRUTH_ROOT"] = str(support_root(truth_root))
            errors, _ = validate_submission_file(
                directory / "submission.json",
                contributor_stage=True,
                require_registered_path=False,
            )
            if errors:
                raise WorkflowError("; ".join(errors))
        finally:
            if prior_truth_root is None:
                os.environ.pop("FLUIDSBENCH_DRIVAERML_TRUTH_ROOT", None)
            else:
                os.environ["FLUIDSBENCH_DRIVAERML_TRUTH_ROOT"] = prior_truth_root
        return {"status": "valid_official_package", "output": str(directory)}
    # Run the same semantic candidate checks used by dev intake. Independently
    # enforce the genuine-inference/package-tree checks in a conventional
    # temporary layout, allowing verification before copying into a Git fork.
    package_tree_binding(directory)
    submission_id = load_json(directory / "submission.json").get("submission_id")
    if not isinstance(submission_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{0,159}", submission_id
    ):
        raise WorkflowError("package submission_id is invalid")
    with tempfile.TemporaryDirectory(prefix="drivaerml-verify-") as temporary:
        repository = Path(temporary)
        copied = repository / "submissions/drivaerml" / submission_id
        shutil.copytree(directory, copied)
        build_registry_entry(copied / "submission.json", repository_root=repository)
    args = [
        sys.executable,
        str(ROOT / "scripts/validate_submission.py"),
        "--candidate-dry-run",
        str(directory),
    ]
    env = dict(os.environ)
    if truth_root:
        env["FLUIDSBENCH_DRIVAERML_TRUTH_ROOT"] = str(support_root(truth_root))
    subprocess.run(args, check=True, env=env)
    return {"status": "valid_dev_package", "output": str(directory)}


def package(
    evaluation_root, config, discretization_cases, output, checkpoints, truth_root=None
):
    from scripts.assemble_drivaerml_schema_v3_candidate import (
        assemble_package,
        DEFAULT_SPECIFICATION,
    )

    evaluation_root, output = Path(evaluation_root).resolve(), Path(output).resolve()
    workflow = load_json(evaluation_root / "workflow.json")
    config_value = load_json(config)
    entry = workflow["entry"]
    official_split(entry["split_id"])
    if (
        config_value.get("split_id") != entry["split_id"]
        or config_value.get("prediction_scope", "surface_and_volume")
        != entry["prediction_scope"]
        or config_value.get("participant", {}).get("submission_id")
        != entry["submission_id"]
    ):
        raise WorkflowError(
            "package config must match the evaluated submission_id, split and scope"
        )
    if workflow["implementation"] != implementation_identity():
        raise WorkflowError(
            "evaluation was produced with a different evaluator or contract"
        )
    for name, sha in workflow["outputs"].items():
        path = evaluation_root / name
        if (
            not path.resolve().is_relative_to(evaluation_root)
            or profiles.sha256(path) != sha
        ):
            raise WorkflowError("evaluation output changed")
    if output.exists():
        raise WorkflowError("package output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".package-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / output.name
        assemble_package(
            config_path=Path(config),
            specification_path=DEFAULT_SPECIFICATION,
            case_metrics_path=evaluation_root / "metrics/cases.json",
            profiles_path=evaluation_root / "package-profiles",
            discretization_cases_path=Path(discretization_cases),
            output_path=staging,
            regional_diagnostics_path=evaluation_root / "regional-diagnostics.json",
            checkpoint_paths=checkpoints,
        )
        if load_json(DEFAULT_SPECIFICATION).get("status") == "official":
            from scripts.finalize_official_native_package import finalize

            finalize(staging, DEFAULT_SPECIFICATION)
        verify_package(staging, truth_root)
        staging.rename(output)
    return {
        "status": (
            "validated_official_package"
            if load_json(DEFAULT_SPECIFICATION).get("status") == "official"
            else "validated_dev_package"
        ),
        "output": str(output),
    }


def report(evaluation_root, case_id, output, truth_root=None):
    """Render a self-contained case report, preserving gaps in every curve."""
    evidence = load_json(Path(evaluation_root) / "candidate-dataset-evidence.json")
    cases = [c for c in evidence["cases"] if c["case_id"] == case_id]
    if len(cases) != 1:
        raise WorkflowError("case is absent from evaluated split")
    root = support_root(truth_root)
    index = profiles.retained_json(
        "index.json", profiles.INDEX_SHA256, root=root, offline=True
    )
    location = next(x for x in index["case_locations"] if x["case_id"] == case_id)
    truth = profiles.retained_json(
        location["chunk_path"], location["chunk_sha256"], root=root, offline=True
    )["cases"][location["case_offset"]]
    truth_series = {(s["family_id"], s["station_id"]): s for s in truth["series"]}
    rows = [
        "<!doctype html><meta charset='utf-8'><title>DrivAerML case report</title>",
        "<style>body{font:16px system-ui;margin:2rem;max-width:1000px}svg{border:1px solid #ccc;width:100%;max-width:640px}section{break-inside:avoid}td,th{padding:.35rem;text-align:left}</style>",
        f"<h1>DrivAerML {html.escape(case_id)}</h1><p>Development result. Blue: truth. Orange: prediction. Relative profiles are report-only.</p>",
        "<h2>Complete-split metrics</h2><table>",
    ]
    for key, value in evidence["metric_values"].items():
        rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        )
    rows.append("</table><h2>Case profiles</h2>")
    for series in cases[0]["profile_series"]:
        source = truth_series[(series["family_id"], series["station_id"])]
        if source["representation"] == "shared_alias":
            continue
        x, y, predicted = source["coordinate"], source["value"], series["prediction"]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(*y, *predicted), max(*y, *predicted)
        scale = lambda a, lo, hi, extent: (a - lo) / (hi - lo or 1.0) * extent
        rows.append(
            f"<section><h3>{html.escape(series['family_id'] + ' / ' + series['station_id'])}</h3><p>{html.escape(source['coordinate_id'])} ({html.escape(source['coordinate_unit'])})</p><svg viewBox='0 0 640 240' role='img' aria-label='Truth and prediction profile'>"
        )
        for values, color in [(y, "#1671b8"), (predicted, "#d96b00")]:
            for segment in source["segments"]:
                points = " ".join(
                    f"{20+scale(x[i],xmin,xmax,600):.3f},{220-scale(values[i],ymin,ymax,200):.3f}"
                    for i in range(
                        segment["emitted_index_start"], segment["emitted_index_stop"]
                    )
                )
                rows.append(
                    f"<polyline fill='none' stroke='{color}' stroke-width='1.5' points='{points}'/>"
                )
        rows.append(
            f"</svg><p>Coordinate: {xmin:.4g} to {xmax:.4g}; value: {ymin:.4g} to {ymax:.4g}</p></section>"
        )
    with Path(output).open("x") as handle:
        handle.write("\n".join(rows))
    return {"status": "report_written", "output": str(output)}
