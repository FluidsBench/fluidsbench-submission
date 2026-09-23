"""One explicit native-to-SI export stage; never alter relative statistics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ID = "hiliftaeroml-dimensional-export-si-v1"
CONTRACT_PATH = Path("benchmark-specs/hiliftaeroml/dimensional-export-si-v1.json")
CONTRACT_SHA256 = "a25f25b24228b72b0798704a155745ea5aab40db928e3e27a01904ec868acc59"
CORRECTION_PATH = Path("benchmark-specs/hiliftaeroml/dimensional-export-correction-v1.json")
# Derived from the code-pinned originals; historical receipts remain immutable.
CORRECTION_SHA256 = "19c0cfff3c2a26e099a50639a59f48db89f102c4006abed874fa5b37c6d5979b"
PRESSURE_TO_PA = 14.5939029372 / 0.0254
VELOCITY_TO_MPS = 0.0254
METRIC_FACTORS = {
    f"{field}_{reduction}": VELOCITY_TO_MPS if field == "volume_velocity" else PRESSURE_TO_PA
    for field in ("surface_pressure", "surface_wall_shear", "volume_pressure", "volume_velocity")
    for reduction in ("mae", "rmse")
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_binding() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "stage": "fluidsbench_package_export",
        "input_basis": "solver_native_dimensional",
        "output_basis": "SI",
        "application_count": 1,
    }


def validate_contract(root: Path = ROOT) -> None:
    path = root / CONTRACT_PATH
    if digest(path) != CONTRACT_SHA256:
        raise ValueError("HiLiftAeroML dimensional export contract hash differs")
    if json.loads(path.read_text())["metric_factors"] != METRIC_FACTORS:
        raise ValueError("HiLiftAeroML dimensional conversion factors differ")


def native_error_to_si(metric_id: str, value: float) -> float:
    """Accept only the eight dimensional metrics from the native assembler input."""
    if metric_id not in METRIC_FACTORS:
        raise ValueError(f"{metric_id}: not a dimensional HiLiftAeroML field error")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{metric_id}: expected a finite nonnegative native error")
    converted = value * METRIC_FACTORS[metric_id]
    if not math.isfinite(converted):
        raise ValueError(f"{metric_id}: SI conversion overflow")
    return converted


def corrected_documents(
    submission: dict[str, Any], evidence: dict[str, Any], cases: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive a correction without mutating inputs; refuse a second conversion."""
    if submission.get("dataset_id") != "hiliftaeroml" or evidence.get("dataset_id") != "hiliftaeroml":
        raise ValueError("SI correction is only defined for HiLiftAeroML")
    if "dimensional_unit_conversion" in evidence:
        raise ValueError("dimensional unit conversion is already recorded")
    if submission["metric_values"] != evidence["metric_values"]:
        raise ValueError("source submission/evidence metrics differ")
    if cases.get("metric_values") != submission["metric_values"]:
        raise ValueError("source case aggregate/submission metrics differ")
    result, proof, case_document = copy.deepcopy((submission, evidence, cases))
    reductions = {metric: [] for metric in METRIC_FACTORS}
    for case in case_document["cases"]:
        seen = set()
        for support in case["supports"]:
            values = support["metric_values"]
            for metric in METRIC_FACTORS.keys() & values.keys():
                if metric in seen:
                    raise ValueError(f"{case['case_id']}: duplicate {metric}")
                seen.add(metric)
                values[metric] = native_error_to_si(metric, values[metric])
                reductions[metric].append(values[metric])
        if seen != METRIC_FACTORS.keys():
            raise ValueError(f"{case['case_id']}: incomplete dimensional metric inventory")
    if not case_document["cases"]:
        raise ValueError("cannot correct an empty case set")
    for metric, values in reductions.items():
        corrected = math.fsum(values) / len(values)
        if not math.isclose(corrected, native_error_to_si(metric, result["metric_values"][metric]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{metric}: source aggregate is not the equal-case mean")
        result["metric_values"][metric] = corrected
        proof["metric_values"][metric] = corrected
        case_document["metric_values"][metric] = corrected
    proof["dimensional_unit_conversion"] = export_binding()
    proof["notes"] = (proof.get("notes", "") + " Dimensional field MAE/RMSE is converted from solver-native units to SI by "
                      + CONTRACT_ID + "; this export stage is separate from the earlier frozen native-evaluator binding. "
                      "Predictions, relative statistics, profiles, coefficients and composite scores are unchanged.")
    return result, proof, case_document


def corrected_registration_view(
    root: Path, path: Path, submission: dict[str, Any], original_binding: dict[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate a pinned derived package while retaining its original receipt.

    Original metadata/evidence remain byte-identical to their existing code-pinned
    registration. Only the separately pinned correction may replace the package.
    The ordinary validator also rebuilds and checks the corrected whole ZIP.
    """
    validate_contract(root)
    registry_path = root / CORRECTION_PATH
    if digest(registry_path) != CORRECTION_SHA256:
        raise ValueError("HiLiftAeroML SI correction registry hash differs")
    registry = json.loads(registry_path.read_text())
    if registry["export_binding"] != export_binding():
        raise ValueError("HiLiftAeroML SI correction version differs")
    entry = registry["records"][original_binding["submission_id"]]
    corrected = entry["corrected"]
    if digest(path) != corrected["submission_json_sha256"] or json.loads(path.read_text()) != submission:
        raise ValueError("corrected submission differs from the registered bytes")
    if digest(path.parent / "evaluation-evidence.json") != corrected["evaluation_evidence_sha256"]:
        raise ValueError("corrected evaluation evidence differs")
    if digest(path.parent / "metrics/cases.json") != corrected["case_metrics_sha256"]:
        raise ValueError("corrected case metrics differ")
    original = entry["source"]
    source_submission = root / original["submission_file"]
    source_evidence = root / original["evidence_file"]
    if digest(source_submission) != original_binding["submission_json_sha256"]:
        raise ValueError("original submission identity differs")
    if digest(source_evidence) != original["evaluation_evidence_sha256"]:
        raise ValueError("original evaluation evidence identity differs")
    binding = copy.deepcopy(original_binding)
    binding["submission_json_sha256"] = corrected["submission_json_sha256"]
    binding["deterministic_archive"] = corrected["deterministic_archive"]
    binding["dimensional_export_version"] = CONTRACT_ID
    return source_submission, source_evidence, binding
