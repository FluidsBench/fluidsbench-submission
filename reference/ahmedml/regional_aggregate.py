"""Build and validate zero-weight AhmedML regional diagnostic aggregates.

The native evaluator emits additive error sums for three mutually exclusive
and exhaustive volume regions.  This module reduces those case reports while
keeping the official field metrics unchanged.  It intentionally accepts only
the evaluator-owned evidence format; participant-authored regional values are
never trusted as scoring inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from .contract import REGION_DEFINITION_SHA256


AGGREGATE_REGIONAL_REPORT_SCHEMA = "ahmedml-regional-diagnostics-aggregate-v1"
REGIONAL_DEFINITION_ID = "ahmedml-native-regions-v1-candidate"
REGIONAL_SUPPORT_ID = "ahmedml-volume-three-geometric-regions-v1"
REGION_IDS = ("near_body", "wake", "farfield")
FIELD_DEFINITIONS = {
    "volume_pressure": ("pMean", "m2 s-2"),
    "volume_velocity": ("UMean", "m s-1"),
}
WEIGHTINGS = ("uniform", "physical")
_STATISTICS = ("absolute_error", "squared_error", "squared_truth", "total_weight")


class AhmedMLRegionalAggregateError(ValueError):
    """Raised when AhmedML regional evidence is incomplete or inconsistent."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AhmedMLRegionalAggregateError(f"{label} must be an object")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AhmedMLRegionalAggregateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or result < 0.0:
        qualifier = "positive finite" if positive else "non-negative finite"
        raise AhmedMLRegionalAggregateError(f"{label} must be {qualifier}")
    return result


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AhmedMLRegionalAggregateError(f"{label} must be an integer")
    if (positive and value < 1) or (not positive and value < 0):
        raise AhmedMLRegionalAggregateError(f"{label} is out of range")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2e-10, abs_tol=2e-12)


def _metric(statistics: Mapping[str, float]) -> dict[str, float]:
    squared_error = statistics["squared_error"]
    squared_truth = statistics["squared_truth"]
    total_weight = statistics["total_weight"]
    if squared_truth <= 0.0 or total_weight <= 0.0:
        raise AhmedMLRegionalAggregateError("regional denominator must be positive")
    return {
        "absolute_error": statistics["absolute_error"],
        "squared_error": squared_error,
        "squared_truth": squared_truth,
        "total_weight": total_weight,
        "relative_l2_percent": 100.0 * math.sqrt(squared_error / squared_truth),
        "mae": statistics["absolute_error"] / total_weight,
        "rmse": math.sqrt(squared_error / total_weight),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AhmedMLRegionalAggregateError("cannot reduce an empty distribution")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    blend = position - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def _distribution(values: Sequence[float]) -> dict[str, float | str]:
    return {
        "minimum": min(values),
        "median": float(median(values)),
        "p90": _percentile(values, 0.9),
        "maximum": max(values),
        "method": "linear_order_statistics_over_complete_cases",
    }


def _case_regional(
    document: Mapping[str, Any],
    *,
    case_id: str,
) -> Mapping[str, Any]:
    if document.get("case_id") != case_id:
        raise AhmedMLRegionalAggregateError(f"{case_id} evidence identity differs")
    regional = _mapping(
        document.get("report_only_regional_diagnostics"),
        f"{case_id}.report_only_regional_diagnostics",
    )
    if (
        regional.get("definition_sha256") != REGION_DEFINITION_SHA256
        or regional.get("ranking_effect") != "none"
    ):
        raise AhmedMLRegionalAggregateError(f"{case_id} regional contract differs")
    return regional


def build_aggregate_regional_diagnostics(
    *,
    case_ids: Sequence[str],
    case_evidence: Mapping[str, Mapping[str, Any]],
    split_id: str,
) -> dict[str, Any]:
    """Reduce complete evaluator evidence into a dashboard-ready report."""

    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise AhmedMLRegionalAggregateError("case_ids must be non-empty and unique")
    if set(case_evidence) != set(case_ids):
        raise AhmedMLRegionalAggregateError("case evidence membership differs")

    regional_by_case = {
        case_id: _case_regional(case_evidence[case_id], case_id=case_id)
        for case_id in case_ids
    }
    field_reports: dict[str, Any] = {}
    for field_id, (quantity, unit) in FIELD_DEFINITIONS.items():
        region_accumulators: dict[str, dict[str, Any]] = {}
        support_entity_count = 0
        support_physical_weight = 0.0
        for region_id in REGION_IDS:
            region_accumulators[region_id] = {
                "entity_count": 0,
                "physical_weight": 0.0,
                "sums": {
                    weighting: {name: 0.0 for name in _STATISTICS}
                    for weighting in WEIGHTINGS
                },
                "case_metrics": {
                    weighting: {name: [] for name in ("relative_l2_percent", "mae", "rmse")}
                    for weighting in WEIGHTINGS
                },
            }

        for case_id in case_ids:
            field = _mapping(regional_by_case[case_id].get(field_id), f"{case_id}.{field_id}")
            case_entity_count = 0
            case_physical_weight = 0.0
            reconstructed = {
                weighting: {name: 0.0 for name in _STATISTICS}
                for weighting in WEIGHTINGS
            }
            for region_id in REGION_IDS:
                region = _mapping(field.get(region_id), f"{case_id}.{field_id}.{region_id}")
                entity_count = _integer(
                    region.get("entity_count"),
                    f"{case_id}.{field_id}.{region_id}.entity_count",
                    positive=True,
                )
                physical_weight = _finite(
                    region.get("physical_weight_sum"),
                    f"{case_id}.{field_id}.{region_id}.physical_weight_sum",
                    positive=True,
                )
                case_entity_count += entity_count
                case_physical_weight += physical_weight
                accumulator = region_accumulators[region_id]
                accumulator["entity_count"] += entity_count
                accumulator["physical_weight"] += physical_weight
                for weighting in WEIGHTINGS:
                    metrics = _mapping(
                        region.get(weighting),
                        f"{case_id}.{field_id}.{region_id}.{weighting}",
                    )
                    values = {
                        name: _finite(
                            metrics.get(name),
                            f"{case_id}.{field_id}.{region_id}.{weighting}.{name}",
                            positive=name in {"squared_truth", "total_weight"},
                        )
                        for name in _STATISTICS
                    }
                    calculated = _metric(values)
                    for metric_id in ("relative_l2_percent", "mae", "rmse"):
                        declared = _finite(
                            metrics.get(metric_id),
                            f"{case_id}.{field_id}.{region_id}.{weighting}.{metric_id}",
                        )
                        if not _close(declared, calculated[metric_id]):
                            raise AhmedMLRegionalAggregateError(
                                f"{case_id}.{field_id}.{region_id}.{weighting}.{metric_id} differs from sums"
                            )
                        accumulator["case_metrics"][weighting][metric_id].append(declared)
                    for name, value in values.items():
                        accumulator["sums"][weighting][name] += value
                        reconstructed[weighting][name] += value

            global_fields = _mapping(
                case_evidence[case_id].get("field_statistics"),
                f"{case_id}.field_statistics",
            )
            global_field = _mapping(global_fields.get(field_id), f"{case_id}.field_statistics.{field_id}")
            if _integer(global_field.get("entity_count"), f"{case_id}.{field_id}.entity_count", positive=True) != case_entity_count:
                raise AhmedMLRegionalAggregateError(f"{case_id}.{field_id} regional entity counts do not reconstruct")
            for weighting in WEIGHTINGS:
                global_weighting = _mapping(global_field.get(weighting), f"{case_id}.{field_id}.{weighting}")
                for name in _STATISTICS:
                    expected = _finite(
                        global_weighting.get(name),
                        f"{case_id}.{field_id}.{weighting}.{name}",
                        positive=name in {"squared_truth", "total_weight"},
                    )
                    if not _close(reconstructed[weighting][name], expected):
                        raise AhmedMLRegionalAggregateError(
                            f"{case_id}.{field_id}.{weighting}.{name} does not reconstruct the global field"
                        )
            support_entity_count += case_entity_count
            support_physical_weight += case_physical_weight

        total_squared_error = {
            weighting: math.fsum(
                region_accumulators[region_id]["sums"][weighting]["squared_error"]
                for region_id in REGION_IDS
            )
            for weighting in WEIGHTINGS
        }
        regions: list[dict[str, Any]] = []
        for code, region_id in enumerate(REGION_IDS):
            accumulator = region_accumulators[region_id]
            region: dict[str, Any] = {
                "region_id": region_id,
                "code": code,
                "entity_count": accumulator["entity_count"],
                "entity_fraction": accumulator["entity_count"] / support_entity_count,
                "physical_weight": accumulator["physical_weight"],
                "physical_weight_fraction": accumulator["physical_weight"] / support_physical_weight,
            }
            for weighting in WEIGHTINGS:
                pooled = _metric(accumulator["sums"][weighting])
                pooled["fraction_of_support_squared_error"] = (
                    pooled["squared_error"] / total_squared_error[weighting]
                )
                case_metrics = accumulator["case_metrics"][weighting]
                region["equal_entity" if weighting == "uniform" else "physical"] = {
                    "pooled": pooled,
                    "macro_case_mean": {
                        metric_id: math.fsum(values) / len(values)
                        for metric_id, values in case_metrics.items()
                    },
                    "case_distribution": {
                        metric_id: _distribution(values)
                        for metric_id, values in case_metrics.items()
                    },
                }
            regions.append(region)
        field_reports[field_id] = {
            "case_count": len(case_ids),
            "entity_count": support_entity_count,
            "physical_weight": support_physical_weight,
            "primary_weighting": "equal_entity",
            "quantity": quantity,
            "unit": unit,
            "regions": regions,
            "validation": {
                "all_case_reports_reconstructed_global_sums": True,
                "three_regions_mutually_exclusive_exhaustive": True,
                "macro_is_equal_case_mean": True,
                "pooled_is_report_only_not_official_case_reduction": True,
            },
        }

    result = {
        "schema": AGGREGATE_REGIONAL_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete_report_only",
        "definition_id": REGIONAL_DEFINITION_ID,
        "contract_sha256": REGION_DEFINITION_SHA256,
        "dataset_id": "ahmedml",
        "split_id": split_id,
        "prediction_scope": "surface_and_volume",
        "case_count": len(case_ids),
        "case_ids": list(case_ids),
        "scoring": {
            "official_metric_inputs_changed": False,
            "official_score_changed": False,
            "role": "report_only",
            "weight": 0.0,
        },
        "validation": {
            "all_case_reports_strictly_validated": True,
            "all_regional_sums_reconstruct_unchanged_global_field_sums": True,
            "exact_case_order_and_membership": True,
            "regional_values_consumed_by_official_score": False,
        },
        "supports": {
            REGIONAL_SUPPORT_ID: {
                "definition_sha256": REGION_DEFINITION_SHA256,
                "definition": {
                    "definition_id": REGIONAL_DEFINITION_ID,
                    "support_id": "ahmedml-volume-native-cells-v1",
                    "guide": "ahmed_volume",
                    "coordinate_frame": "native AhmedML body frame in metres",
                    "partition_properties": "mutually_exclusive_and_exhaustive",
                    "regions_in_code_order": [
                        {
                            "region_id": "near_body",
                            "code": 0,
                            "predicate": "-1.25*L <= x <= 0; |y| <= 0.75*W; 0 <= z <= z_surface_min + 2*H",
                        },
                        {
                            "region_id": "wake",
                            "code": 1,
                            "predicate": "0 < x <= 2*L; |y| <= W; 0 <= z <= z_surface_min + 2*H",
                        },
                        {
                            "region_id": "farfield",
                            "code": 2,
                            "predicate": "all remaining native volume cells",
                        },
                    ],
                    "scoring_role": "report_only_zero_weight",
                    "scoring_weight": 0.0,
                    "semantic_limit": "Geometric diagnostic zones only; the wake box is not a streamline- or topology-derived wake label.",
                },
                "fields": field_reports,
            }
        },
    }
    validate_aggregate_regional_diagnostics(
        result,
        expected_case_ids=case_ids,
        expected_split_id=split_id,
    )
    return result


def validate_aggregate_regional_diagnostics(
    report: Mapping[str, Any],
    *,
    expected_case_ids: Sequence[str],
    expected_split_id: str,
) -> None:
    """Fail closed on a published AhmedML regional aggregate."""

    expected_header = {
        "schema": AGGREGATE_REGIONAL_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete_report_only",
        "definition_id": REGIONAL_DEFINITION_ID,
        "contract_sha256": REGION_DEFINITION_SHA256,
        "dataset_id": "ahmedml",
        "split_id": expected_split_id,
        "prediction_scope": "surface_and_volume",
        "case_count": len(expected_case_ids),
    }
    for key, expected in expected_header.items():
        if report.get(key) != expected:
            raise AhmedMLRegionalAggregateError(f"regional report {key} differs")
    if report.get("case_ids") != list(expected_case_ids):
        raise AhmedMLRegionalAggregateError("regional report case order differs")
    if report.get("scoring") != {
        "official_metric_inputs_changed": False,
        "official_score_changed": False,
        "role": "report_only",
        "weight": 0.0,
    }:
        raise AhmedMLRegionalAggregateError("regional scoring declaration differs")
    validation = _mapping(report.get("validation"), "validation")
    if dict(validation) != {
        "all_case_reports_strictly_validated": True,
        "all_regional_sums_reconstruct_unchanged_global_field_sums": True,
        "exact_case_order_and_membership": True,
        "regional_values_consumed_by_official_score": False,
    }:
        raise AhmedMLRegionalAggregateError("regional validation declaration differs")
    support = _mapping(_mapping(report.get("supports"), "supports").get(REGIONAL_SUPPORT_ID), REGIONAL_SUPPORT_ID)
    if support.get("definition_sha256") != REGION_DEFINITION_SHA256:
        raise AhmedMLRegionalAggregateError("regional support definition digest differs")
    fields = _mapping(support.get("fields"), "regional fields")
    if set(fields) != set(FIELD_DEFINITIONS):
        raise AhmedMLRegionalAggregateError("regional field set differs")
    for field_id in FIELD_DEFINITIONS:
        field = _mapping(fields[field_id], field_id)
        regions = field.get("regions")
        if not isinstance(regions, list) or [item.get("region_id") for item in regions if isinstance(item, Mapping)] != list(REGION_IDS):
            raise AhmedMLRegionalAggregateError(f"{field_id} region order differs")
        if field.get("case_count") != len(expected_case_ids) or field.get("primary_weighting") != "equal_entity":
            raise AhmedMLRegionalAggregateError(f"{field_id} header differs")
        for fraction_key in ("entity_fraction", "physical_weight_fraction"):
            total = math.fsum(
                _finite(region.get(fraction_key), f"{field_id}.{fraction_key}")
                for region in regions
            )
            if not _close(total, 1.0):
                raise AhmedMLRegionalAggregateError(f"{field_id} {fraction_key} does not reconstruct")
        for weighting_key in ("equal_entity", "physical"):
            error_fraction = math.fsum(
                _finite(
                    _mapping(_mapping(region.get(weighting_key), weighting_key).get("pooled"), "pooled").get("fraction_of_support_squared_error"),
                    f"{field_id}.{weighting_key}.fraction_of_support_squared_error",
                )
                for region in regions
            )
            if not _close(error_fraction, 1.0):
                raise AhmedMLRegionalAggregateError(f"{field_id} {weighting_key} error fractions do not reconstruct")


__all__ = [
    "AGGREGATE_REGIONAL_REPORT_SCHEMA",
    "AhmedMLRegionalAggregateError",
    "REGIONAL_DEFINITION_ID",
    "REGIONAL_SUPPORT_ID",
    "build_aggregate_regional_diagnostics",
    "validate_aggregate_regional_diagnostics",
]
