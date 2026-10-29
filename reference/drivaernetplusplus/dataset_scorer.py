"""Reduce per-case DrivAerNet++ evidence into one split-level result.

Field metrics are calculated per case and macro-averaged (every design counts
equally, per the dataset's ``case_aggregation``).  The equal-point R^2 is
pooled over every native point of every case by an exact parallel-variance
combination.  The overall score comes from :mod:`reference.scores`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from reference.scores import composite_overall_score

from .contract import (
    CASE_SET_ID,
    DATASET_ID,
    DATASET_VERSION,
    EVALUATION_REFERENCE_VERSION,
    RELEASE_ID,
    SUPPORT_ID,
    ContractError,
    load_specification,
    official_split,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .distribution import manifest_sha256, retained_record
from .evaluator import CASE_EVIDENCE_SCHEMA
from .native import PROFILE_STATIONS

DATASET_EVIDENCE_SCHEMA = "fluidsbench-drivaernetplusplus-candidate-dataset-evaluation-v1"
DATASET_EVIDENCE_STATUS = "non_ranked_development_evidence_not_official_submission"
MACRO_METRICS = (
    "surface_pressure_rel_l2",
    "surface_pressure_equal_entity_rel_l2",
    "surface_pressure_rel_l1",
)


class DatasetScorerError(ContractError):
    """Raised when a split cannot be reduced exactly."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatasetScorerError(f"{label} must be a finite number")
    return float(value)


def check_case_evidence(evidence: Mapping[str, Any], case_id: str) -> None:
    """Recompute every reported case metric from its own sufficient statistics."""

    record, record_bytes = retained_record(case_id)
    support = evidence.get("support", {})
    if (
        evidence.get("schema") != CASE_EVIDENCE_SCHEMA
        or evidence.get("dataset_id") != DATASET_ID
        or evidence.get("release_id") != RELEASE_ID
        or evidence.get("case_id") != case_id
        or support.get("support_id") != SUPPORT_ID
        or support.get("entity_count") != record["native"]["point_count"]
        or support.get("case_support_sha256") != sha256_bytes(record_bytes)
        or evidence.get("prediction", {}).get("entity_count") != record["native"]["point_count"]
    ):
        raise DatasetScorerError(f"{case_id}: evidence identity differs from the frozen support")
    values = evidence.get("metric_values", {})
    stats = evidence.get("metric_sufficient_statistics", {})
    for metric in ("surface_pressure_rel_l2", "surface_pressure_equal_entity_rel_l2"):
        item = stats.get(metric, {})
        numerator = _finite(item.get("numerator"), f"{case_id}.{metric}.numerator")
        denominator = _finite(item.get("denominator"), f"{case_id}.{metric}.denominator")
        if numerator < 0 or denominator <= 0:
            raise DatasetScorerError(f"{case_id}: invalid {metric} statistics")
        if not math.isclose(
            _finite(values.get(metric), metric), 100.0 * math.sqrt(numerator / denominator), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise DatasetScorerError(f"{case_id}: {metric} differs from its statistics")
    l1 = evidence.get("relative_l1_statistics", {})
    if not math.isclose(
        _finite(values.get("surface_pressure_rel_l1"), "rel_l1"),
        100.0 * _finite(l1.get("numerator"), "l1") / _finite(l1.get("denominator"), "l1"),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise DatasetScorerError(f"{case_id}: surface_pressure_rel_l1 differs from its statistics")
    r2 = evidence.get("r2_statistics", {})
    if r2.get("count") != record["native"]["point_count"] or _finite(r2.get("truth_centred_sum_squares"), "r2") <= 0:
        raise DatasetScorerError(f"{case_id}: invalid R2 statistics")
    profiles = evidence.get("profiles")
    if not isinstance(profiles, list) or [p.get("station_id") for p in profiles] != list(PROFILE_STATIONS):
        raise DatasetScorerError(f"{case_id}: profile stations differ from the contract")
    for profile in profiles:
        expected = record["profile_stations"][profile["station_id"]]["point_count"]
        if any(len(profile.get(key, [])) != expected for key in ("coordinate", "truth_cp", "prediction_cp")):
            raise DatasetScorerError(f"{case_id}: profile {profile['station_id']} length differs")


def pooled_r2(cases: Sequence[Mapping[str, Any]]) -> float:
    count, mean, centred, residual = 0, 0.0, 0.0, 0.0
    for case in cases:
        stats = case["r2_statistics"]
        n_b, mean_b = int(stats["count"]), float(stats["truth_mean"])
        total = count + n_b
        delta = mean_b - mean
        centred += float(stats["truth_centred_sum_squares"]) + delta * delta * count * n_b / total
        mean += delta * n_b / total
        count = total
        residual += float(stats["sum_squared_error"])
    return 1.0 - residual / centred


def score_candidate_dataset(
    *,
    split_id: str,
    case_evidence_directory: Path,
    submission_id: str,
) -> dict[str, Any]:
    split = official_split(split_id)
    case_ids = split["case_ids"]
    directory = Path(case_evidence_directory)
    present = {path.stem for path in directory.glob("*.json")}
    if present != set(case_ids):
        raise DatasetScorerError(
            "case evidence must cover exactly the split: "
            f"missing={sorted(set(case_ids) - present)[:5]}, unexpected={sorted(present - set(case_ids))[:5]}"
        )
    cases, bindings = [], []
    for case_id in case_ids:
        path = directory / f"{case_id}.json"
        evidence = read_json(path, f"{case_id} evidence")
        check_case_evidence(evidence, case_id)
        cases.append(evidence)
        bindings.append({"case_id": case_id, "sha256": sha256_file(path)})
    metric_values: dict[str, float] = {
        metric: math.fsum(case["metric_values"][metric] for case in cases) / len(cases) for metric in MACRO_METRICS
    }
    metric_values["surface_pressure_r2"] = pooled_r2(cases)
    spec = load_specification()
    metric_values["overall_score"] = composite_overall_score(metric_values, spec["overall_score_composite"])
    ordered = {metric["id"]: metric_values[metric["id"]] for metric in spec["metrics"]}
    if set(ordered) != set(metric_values):
        raise DatasetScorerError("specification metrics differ from the evaluator's metric set")
    declaration = next(item for item in spec["splits"] if item["id"] == split_id)
    return {
        "schema": DATASET_EVIDENCE_SCHEMA,
        "status": DATASET_EVIDENCE_STATUS,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "evaluation_reference_version": EVALUATION_REFERENCE_VERSION,
        "scoring_support_release_id": RELEASE_ID,
        "scoring_support_manifest_sha256": manifest_sha256(),
        "submission_id": submission_id,
        "split_id": split_id,
        "case_set_id": split.get("case_set_id", CASE_SET_ID),
        "split_sha256": declaration["sha256"],
        "case_count": len(case_ids),
        "case_evidence": bindings,
        "metric_values": ordered,
    }
