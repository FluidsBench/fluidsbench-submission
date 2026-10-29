"""Participant orchestration for the candidate DrivAerNet++ pressure benchmark.

An entry directory holds ``entry.json`` and one ``cases/<case_id>/prediction.npz``
per official_test case.  Evaluation is resumable: each case is scored once,
published atomically with a checkpoint bound to the evaluator implementation,
the prepared support, and the exact prediction bytes, and reused on restart.
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import html
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .contract import (
    DATASET_ID,
    DATASET_NAME,
    DISTRIBUTION_PATH,
    RELEASE_DIR,
    ROOT,
    SOURCE_IDENTITY_PATH,
    SOURCE_IDENTITY_SCHEMA,
    SPEC_ROOT,
    SUBMISSION_ID_RE,
    SUPPORT_ID,
    ContractError,
    digest,
    official_split,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .dataset_scorer import score_candidate_dataset
from .distribution import cache_root, fetch_support, load_case_support, retained_record
from .evaluator import PREDICTION_FILE, load_prediction, score_case

ENTRY_SCHEMA = "fluidsbench-drivaernetplusplus-entry-v1"
CAMPAIGN_SCHEMA = "fluidsbench-drivaernetplusplus-campaign-v1"
CHECKPOINT_SCHEMA = "fluidsbench-drivaernetplusplus-case-checkpoint-v1"
WORKFLOW_SCHEMA = "fluidsbench-drivaernetplusplus-evaluation-v1"
RUNTIME = ("Linux", "3.12.13", "2.2.6")
DEFAULT_WORKERS = 8


class WorkflowError(ContractError):
    """An entry cannot be evaluated, resumed, or packaged reproducibly."""


def check_runtime() -> dict[str, str]:
    import numpy as np

    observed = (platform.system(), platform.python_version(), np.__version__)
    if observed != RUNTIME:
        raise WorkflowError(
            f"evaluation requires Linux, Python {RUNTIME[1]} and NumPy {RUNTIME[2]} "
            f"(found {observed}); install requirements-drivaernetplusplus-participant.txt "
            "or use Dockerfile.drivaernetplusplus"
        )
    return {"system": observed[0], "python": observed[1], "numpy": observed[2]}


def implementation_identity() -> str:
    """Hash every file that can change a DrivAerNet++ score or package."""

    paths = sorted((ROOT / "reference" / "drivaernetplusplus").glob("*.py"))
    paths += [
        ROOT / "reference" / "scores.py",
        ROOT / "scripts" / "drivaernetplusplus.py",
        ROOT / "scripts" / "assemble_drivaernetplusplus_schema_v3_candidate.py",
        SPEC_ROOT / "submission-spec.json",
        SPEC_ROOT / "methodology-contract.json",
        DISTRIBUTION_PATH,
        RELEASE_DIR / "manifest.json",
    ]
    paths += sorted((SPEC_ROOT / "splits").glob("*.json"))
    return digest({path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths})


def load_entry(entry_root: Path) -> dict[str, Any]:
    entry = read_json(Path(entry_root) / "entry.json", "entry.json")
    if not isinstance(entry, dict) or set(entry) != {"schema", "submission_id", "split_id"}:
        raise WorkflowError(
            "entry.json must contain exactly schema, submission_id and split_id; custom case lists are forbidden"
        )
    if entry["schema"] != ENTRY_SCHEMA:
        raise WorkflowError(f"entry.json schema must be {ENTRY_SCHEMA!r}")
    if not isinstance(entry["submission_id"], str) or not SUBMISSION_ID_RE.fullmatch(entry["submission_id"]) or (
        "." in entry["submission_id"] or "_" in entry["submission_id"]
    ):
        raise WorkflowError("submission_id must use lowercase letters, digits and hyphens (3-80 characters)")
    if not 3 <= len(entry["submission_id"]) <= 80:
        raise WorkflowError("submission_id must be 3-80 characters long")
    official_split(entry["split_id"])
    return entry


# ---------------------------------------------------------------------------
# Public data and prepared support
# ---------------------------------------------------------------------------


def source_identity() -> dict[str, Any]:
    identity = read_json(SOURCE_IDENTITY_PATH, "public source identity")
    if identity.get("schema") != SOURCE_IDENTITY_SCHEMA or identity.get("dataset_id") != DATASET_ID:
        raise WorkflowError("public source identity differs")
    return identity


def verify_data(dataset_root: Path, split_id: str, *, check_hashes: bool = True, workers: int = DEFAULT_WORKERS) -> dict[str, Any]:
    """Confirm the locally extracted public files are the pinned release bytes.

    The Dataverse release is a set of family ZIP archives behind Globus, so
    FluidsBench cannot download single files for you.  Extract
    ``SurfacePressureVTK`` and point ``--dataset-root`` at it; files named
    ``<case_id>.vtk`` and ``DrivAer_<case_id>.vtk`` are both accepted.
    """

    root = Path(dataset_root).expanduser().resolve()
    expected = {row["case_id"]: row for row in source_identity()["cases"]}
    cases = official_split(split_id)["case_ids"]
    located, problems = {}, []
    for case_id in cases:
        candidates = [root / f"{case_id}.vtk", root / f"DrivAer_{case_id}.vtk"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            problems.append(f"{case_id}: missing")
        elif path.stat().st_size != expected[case_id]["size_bytes"]:
            problems.append(f"{case_id}: size {path.stat().st_size} != {expected[case_id]['size_bytes']}")
        else:
            located[case_id] = path
    if check_hashes and located:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            digests = dict(zip(located, pool.map(sha256_file, located.values())))
        problems += [f"{c}: SHA-256 differs" for c, d in digests.items() if d != expected[c]["sha256"]]
    if problems:
        raise WorkflowError(f"{len(problems)} public file problem(s): {problems[:5]}")
    return {
        "status": "public_files_verified",
        "dataset_root": str(root),
        "case_count": len(cases),
        "checked": "size_and_sha256" if check_hashes else "size_only",
    }


# ---------------------------------------------------------------------------
# Entry validation and resumable evaluation
# ---------------------------------------------------------------------------


def prediction_path(entry_root: Path, case_id: str) -> Path:
    return Path(entry_root) / "cases" / case_id / PREDICTION_FILE


def validate_entry(entry_root: Path, *, check_arrays: bool = True) -> dict[str, Any]:
    entry_root = Path(entry_root).expanduser().resolve()
    entry = load_entry(entry_root)
    cases = official_split(entry["split_id"])["case_ids"]
    cases_root = entry_root / "cases"
    present = {p.name for p in cases_root.iterdir()} if cases_root.is_dir() else set()
    if present != set(cases):
        raise WorkflowError(
            "prediction cases must exactly match the official split: "
            f"missing={sorted(set(cases) - present)[:5]} unexpected={sorted(present - set(cases))[:5]}"
        )
    for case_id in cases:
        files = {p.name for p in (cases_root / case_id).iterdir()}
        if files != {PREDICTION_FILE}:
            raise WorkflowError(f"{case_id}: expected exactly {PREDICTION_FILE}, found {sorted(files)}")
        if check_arrays:
            record, _ = retained_record(case_id)
            load_prediction(prediction_path(entry_root, case_id), record["native"]["point_count"])
    return {"status": "entry_valid", "entry": entry, "case_count": len(cases)}


def _campaign(entry: dict[str, Any], runtime: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "entry": entry,
        "implementation": implementation_identity(),
        "runtime": runtime,
    }


def _case_inputs(campaign_digest: str, entry_root: Path, case_id: str) -> dict[str, Any]:
    _, record_bytes = retained_record(case_id)
    return {
        "campaign": campaign_digest,
        "case_support_sha256": sha256_bytes(record_bytes),
        "prediction_sha256": sha256_file(prediction_path(entry_root, case_id)),
    }


def _verify_checkpoint(directory: Path, inputs: dict[str, Any]) -> None:
    receipt = read_json(directory / "checkpoint.json", f"{directory.name} checkpoint")
    if receipt.get("schema") != CHECKPOINT_SCHEMA or receipt.get("inputs") != inputs:
        raise WorkflowError(
            f"{directory.name}: checkpoint inputs changed (prediction, support or evaluator); "
            "choose a new output directory"
        )
    if receipt.get("outputs") != {"evidence.json": sha256_file(directory / "evidence.json")}:
        raise WorkflowError(f"{directory.name}: checkpointed evidence changed")


def _evaluate_one(case_id: str, entry_root: str, output: str, support_root: str, inputs: dict[str, Any]) -> str:
    destination = Path(output) / "cases" / case_id
    support = load_case_support(case_id, destination=support_root)
    evidence = score_case(support, prediction_path(Path(entry_root), case_id))
    if evidence["prediction"]["sha256"] != inputs["prediction_sha256"]:
        raise WorkflowError(f"{case_id}: prediction changed during evaluation")
    with tempfile.TemporaryDirectory(prefix=f".case-{case_id}-", dir=Path(output) / "cases") as temporary:
        staging = Path(temporary) / case_id
        staging.mkdir()
        evidence_sha = write_json(staging / "evidence.json", evidence)
        write_json(
            staging / "checkpoint.json",
            {"schema": CHECKPOINT_SCHEMA, "inputs": inputs, "outputs": {"evidence.json": evidence_sha}},
        )
        staging.rename(destination)
    return f"{case_id}: rel_l2 {evidence['metric_values']['surface_pressure_rel_l2']:.3f}%"


def evaluate_entry(
    entry_root: Path,
    output: Path,
    *,
    support_root: Path | None = None,
    resume: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    runtime = check_runtime()
    entry_root, output = Path(entry_root).expanduser().resolve(), Path(output).expanduser().resolve()
    checked = validate_entry(entry_root, check_arrays=False)
    entry = checked["entry"]
    cases = official_split(entry["split_id"])["case_ids"]
    root = cache_root(support_root)
    fetch_support(cases, destination=root, offline=True, dry_run=True)
    if output.exists() and any(output.iterdir()) and not resume:
        raise WorkflowError("output is not empty; pass --resume or choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowError(f"another evaluation is using {output}") from error
        campaign = _campaign(entry, runtime)
        campaign_path = output / "campaign.json"
        if campaign_path.exists():
            if read_json(campaign_path) != campaign:
                raise WorkflowError(
                    "this output was started with a different entry, evaluator, or runtime; "
                    "choose a new output directory"
                )
        else:
            write_json(campaign_path, campaign)
        campaign_digest = digest(campaign)
        (output / "cases").mkdir(exist_ok=True)
        stray = {p.name for p in (output / "cases").iterdir() if not p.name.startswith(".")} - set(cases)
        if stray:
            raise WorkflowError(f"checkpoint directory contains cases outside the split: {sorted(stray)[:5]}")
        pending, reused = {}, 0
        for case_id in cases:
            inputs = _case_inputs(campaign_digest, entry_root, case_id)
            if (output / "cases" / case_id).exists():
                _verify_checkpoint(output / "cases" / case_id, inputs)
                reused += 1
            else:
                pending[case_id] = inputs
        print(f"{reused} verified checkpoint(s) reused; evaluating {len(pending)} case(s)", file=sys.stderr, flush=True)
        failures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_evaluate_one, c, str(entry_root), str(output), str(root), pending[c]): c for c in pending
            }
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    print(f"[{number}/{len(pending)}] {future.result()}", file=sys.stderr, flush=True)
                except Exception as error:  # keep completed checkpoints, report all failures
                    failures.append(f"{futures[future]}: {error}")
        if failures:
            raise WorkflowError(
                f"{len(failures)} case(s) failed; completed cases are checkpointed, fix and rerun with --resume: "
                + "; ".join(failures[:5])
            )
        if implementation_identity() != campaign["implementation"]:
            raise WorkflowError("evaluator files changed during evaluation")
        return _reduce(entry, cases, output, campaign)


def _reduce(entry: dict[str, Any], cases: Sequence[str], output: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    final = output / "evaluation"
    if final.exists():
        verify_evaluation(final)
        return {"status": "complete_dev_evaluation", "case_count": len(cases), "output": str(final), "reused": True}
    with tempfile.TemporaryDirectory(prefix=".reduce-", dir=output) as temporary:
        staging = Path(temporary) / "evaluation"
        (staging / "case-evidence").mkdir(parents=True)
        for case_id in cases:
            shutil.copyfile(output / "cases" / case_id / "evidence.json", staging / "case-evidence" / f"{case_id}.json")
        evidence = score_candidate_dataset(
            split_id=entry["split_id"],
            case_evidence_directory=staging / "case-evidence",
            submission_id=entry["submission_id"],
        )
        write_json(staging / "dataset-evidence.json", evidence)
        outputs = {
            p.relative_to(staging).as_posix(): sha256_file(p) for p in sorted(staging.rglob("*")) if p.is_file()
        }
        write_json(staging / "workflow.json", {"schema": WORKFLOW_SCHEMA, "campaign": campaign, "outputs": outputs})
        staging.rename(final)
    metrics = read_json(final / "dataset-evidence.json")["metric_values"]
    return {"status": "complete_dev_evaluation", "case_count": len(cases), "output": str(final), "metric_values": metrics}


def verify_evaluation(evaluation_root: Path) -> dict[str, Any]:
    evaluation_root = Path(evaluation_root).expanduser().resolve()
    workflow = read_json(evaluation_root / "workflow.json", "workflow.json")
    if workflow.get("schema") != WORKFLOW_SCHEMA:
        raise WorkflowError("evaluation workflow schema differs")
    if workflow["campaign"].get("implementation") != implementation_identity():
        raise WorkflowError("evaluation was produced with a different evaluator or contract")
    actual = {
        p.relative_to(evaluation_root).as_posix(): sha256_file(p)
        for p in sorted(evaluation_root.rglob("*"))
        if p.is_file() and p.name != "workflow.json"
    }
    if actual != workflow.get("outputs"):
        raise WorkflowError("completed evaluation files changed")
    return workflow


# ---------------------------------------------------------------------------
# Discretization template, packaging, and reports
# ---------------------------------------------------------------------------


def discretization_template(split_id: str, output: Path) -> dict[str, Any]:
    """Write per-case discretization records for native-point inference.

    The default describes a model that consumed and predicted every native
    point directly (identity mapping).  Edit inputs/direct_outputs if your model
    used a different representation; the mapping counts must stay native.
    """

    import json

    cases = official_split(split_id)["case_ids"]
    lines = []
    for case_id in cases:
        record, _ = retained_record(case_id)
        n = record["native"]["point_count"]
        counts = [{"entity": "points", "count": n}]
        fraction = [{"entity": "points", "fraction": 1.0}]
        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "inference": {
                        "inputs": [
                            {
                                "id": "surface-native-input",
                                "entity_counts": counts,
                                "native_entity_counts": counts,
                                "native_fractions": fraction,
                                "domain": {"kind": "full_dataset_domain"},
                            }
                        ],
                        "direct_outputs": [
                            {
                                "id": "surface-native-output",
                                "entity_counts": counts,
                                "native_entity_counts": counts,
                                "native_fractions": fraction,
                                "domain": {"kind": "full_dataset_domain"},
                            }
                        ],
                        "mappings": [
                            {
                                "support_id": SUPPORT_ID,
                                "source_output_id": "surface-native-output",
                                "support_count": n,
                                "scored_count": n,
                                "unmapped_count": 0,
                                "extrapolated_count": 0,
                                "final_coverage_fraction": 1.0,
                            }
                        ],
                    },
                },
                separators=(",", ":"),
            )
        )
    output = Path(output)
    if output.exists():
        raise WorkflowError(f"{output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return {"status": "template_written", "output": str(output), "case_count": len(cases)}


def package(
    evaluation_root: Path,
    config: Path,
    discretization_cases: Path,
    output: Path,
    checkpoints: dict[str, Path],
) -> dict[str, Any]:
    from scripts.assemble_drivaernetplusplus_schema_v3_candidate import assemble_package

    check_runtime()
    evaluation_root, output = Path(evaluation_root).expanduser().resolve(), Path(output).expanduser().resolve()
    workflow = verify_evaluation(evaluation_root)
    if output.exists():
        raise WorkflowError("package output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        result = assemble_package(
            config_path=Path(config).expanduser().resolve(),
            evaluation_root=evaluation_root,
            entry=workflow["campaign"]["entry"],
            discretization_cases_path=Path(discretization_cases).expanduser().resolve(),
            checkpoint_paths=checkpoints,
            output_path=staging,
        )
        specification_path = ROOT / "benchmark-specs/drivaernetplusplus/submission-spec.json"
        if read_json(specification_path).get("status") == "official":
            from scripts.finalize_official_native_package import finalize
            finalize(staging, specification_path)
        verify_package(staging)
        staging.rename(output)
    result.update({"status": "validated_package", "output": str(output)})
    return result


def verify_package(directory: Path) -> dict[str, Any]:
    import subprocess

    from .pre_release import build_registry_entry, package_tree_binding

    directory = Path(directory).expanduser().resolve()
    if read_json(ROOT / "benchmark-specs/drivaernetplusplus/submission-spec.json").get("status") == "official":
        from scripts.validate_submission import validate_submission_file
        errors, _ = validate_submission_file(directory / "submission.json", contributor_stage=True, require_registered_path=False)
        if errors:
            raise WorkflowError("; ".join(errors))
        return {"status": "valid_official_package", "output": str(directory)}
    package_tree_binding(directory)
    submission_id = read_json(directory / "submission.json").get("submission_id")
    if not isinstance(submission_id, str) or not SUBMISSION_ID_RE.fullmatch(submission_id):
        raise WorkflowError("package submission_id is invalid")
    # Same exact checks dev intake applies, in a conventional temporary layout,
    # so a package can be verified before it is copied into a fork.
    with tempfile.TemporaryDirectory(prefix="drivaernetplusplus-verify-") as temporary:
        copied = Path(temporary) / "submissions" / DATASET_ID / submission_id
        shutil.copytree(directory, copied)
        build_registry_entry(copied / "submission.json", repository_root=Path(temporary))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), "--candidate-dry-run", str(directory)],
        check=True,
    )
    return {"status": "valid_dev_package", "output": str(directory)}


def report(evaluation_root: Path, case_id: str, output: Path) -> dict[str, Any]:
    """Self-contained HTML: split metrics, case metrics, and the three Cp profiles."""

    evaluation_root = Path(evaluation_root).expanduser().resolve()
    verify_evaluation(evaluation_root)
    dataset = read_json(evaluation_root / "dataset-evidence.json")
    path = evaluation_root / "case-evidence" / f"{case_id}.json"
    if not path.is_file():
        raise WorkflowError(f"{case_id} is not part of the evaluated split")
    case = read_json(path)
    rows = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{DATASET_NAME} {html.escape(case_id)}</title>",
        "<style>body{font:16px system-ui;margin:2rem;max-width:1000px}svg{border:1px solid #ccc;width:100%;max-width:680px}"
        "td,th{padding:.3rem .6rem;text-align:left}</style>",
        f"<h1>{DATASET_NAME} {html.escape(case_id)}</h1>",
        "<p>Development result, not an official score. Blue: reference Cp. Orange: prediction. Profiles are report-only.</p>",
        "<h2>Complete-split metrics</h2><table>",
    ]
    rows += [f"<tr><th>{html.escape(k)}</th><td>{v:.6g}</td></tr>" for k, v in dataset["metric_values"].items()]
    rows.append("</table><h2>This case</h2><table>")
    rows += [f"<tr><th>{html.escape(k)}</th><td>{v:.6g}</td></tr>" for k, v in case["metric_values"].items()]
    rows.append("</table>")
    for profile in case["profiles"]:
        x, truth, prediction = profile["coordinate"], profile["truth_cp"], profile["prediction_cp"]
        x0, x1 = min(x), max(x)
        y0, y1 = min(*truth, *prediction), max(*truth, *prediction)

        def scale(v, lo, hi, extent):
            return (v - lo) / ((hi - lo) or 1.0) * extent

        rows.append(f"<h3>{html.escape(profile['station_id'])} ({html.escape(profile['coordinate_id'])})</h3>")
        rows.append("<svg viewBox='0 0 680 260' role='img' aria-label='Reference and predicted Cp'>")
        for values, colour in ((truth, "#1671b8"), (prediction, "#d96b00")):
            # Cp axis inverted, as is conventional: suction plotted upward.
            points = " ".join(
                f"{20 + scale(a, x0, x1, 640):.2f},{20 + scale(b, y0, y1, 220):.2f}" for a, b in zip(x, values)
            )
            rows.append(f"<polyline fill='none' stroke='{colour}' stroke-width='1.4' points='{points}'/>")
        rows.append(f"</svg><p>coordinate {x0:.3g} to {x1:.3g}; Cp {y0:.3g} to {y1:.3g} (axis inverted)</p>")
    output = Path(output)
    with output.open("x") as handle:
        handle.write("\n".join(rows))
    return {"status": "report_written", "output": str(output)}
