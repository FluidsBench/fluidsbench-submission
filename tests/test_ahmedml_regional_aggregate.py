from __future__ import annotations

import math

import pytest

from reference.ahmedml.contract import REGION_DEFINITION_SHA256
from reference.ahmedml.regional_aggregate import (
    AhmedMLRegionalAggregateError,
    build_aggregate_regional_diagnostics,
    validate_aggregate_regional_diagnostics,
)


def _metrics(*, absolute: float, squared: float, truth: float, weight: float) -> dict[str, float]:
    return {
        "absolute_error": absolute,
        "squared_error": squared,
        "squared_truth": truth,
        "total_weight": weight,
        "relative_l2_percent": 100.0 * math.sqrt(squared / truth),
        "mae": absolute / weight,
        "rmse": math.sqrt(squared / weight),
    }


def _case(case_id: str, scale: float = 1.0) -> dict[str, object]:
    region_counts = (2, 3, 5)
    physical_weights = (0.25, 0.75, 9.0)
    regional: dict[str, object] = {
        "definition_sha256": REGION_DEFINITION_SHA256,
        "ranking_effect": "none",
    }
    fields: dict[str, object] = {}
    for field_index, field_id in enumerate(("volume_pressure", "volume_velocity"), start=1):
        regions: dict[str, object] = {}
        uniform_sums = {name: 0.0 for name in ("absolute_error", "squared_error", "squared_truth", "total_weight")}
        physical_sums = dict(uniform_sums)
        for region_index, (region_id, count, physical_weight) in enumerate(
            zip(("near_body", "wake", "farfield"), region_counts, physical_weights, strict=True),
            start=1,
        ):
            squared = scale * field_index * region_index
            truth = scale * 100.0 * field_index * region_index
            absolute = scale * 0.5 * field_index * region_index
            uniform = _metrics(
                absolute=absolute,
                squared=squared,
                truth=truth,
                weight=float(count),
            )
            physical = _metrics(
                absolute=absolute / 10.0,
                squared=squared / 100.0,
                truth=truth / 100.0,
                weight=physical_weight,
            )
            regions[region_id] = {
                "code": region_index - 1,
                "entity_count": count,
                "physical_weight_sum": physical_weight,
                "uniform": uniform,
                "physical": physical,
            }
            for name in uniform_sums:
                uniform_sums[name] += uniform[name]
                physical_sums[name] += physical[name]
        fields[field_id] = regions
        regional[field_id] = regions
        fields[field_id] = {
            "entity_count": sum(region_counts),
            "uniform": {**uniform_sums, **_metrics(**{
                "absolute": uniform_sums["absolute_error"],
                "squared": uniform_sums["squared_error"],
                "truth": uniform_sums["squared_truth"],
                "weight": uniform_sums["total_weight"],
            })},
            "physical": {**physical_sums, **_metrics(**{
                "absolute": physical_sums["absolute_error"],
                "squared": physical_sums["squared_error"],
                "truth": physical_sums["squared_truth"],
                "weight": physical_sums["total_weight"],
            })},
        }
    return {
        "case_id": case_id,
        "field_statistics": fields,
        "report_only_regional_diagnostics": regional,
    }


def test_builds_exact_report_only_aggregate() -> None:
    case_ids = ["run_1", "run_2"]
    report = build_aggregate_regional_diagnostics(
        case_ids=case_ids,
        case_evidence={"run_1": _case("run_1"), "run_2": _case("run_2", 2.0)},
        split_id="full",
    )
    assert report["case_ids"] == case_ids
    assert report["scoring"]["weight"] == 0.0
    field = report["supports"]["ahmedml-volume-three-geometric-regions-v1"]["fields"]["volume_pressure"]
    assert [region["region_id"] for region in field["regions"]] == [
        "near_body",
        "wake",
        "farfield",
    ]
    assert sum(region["entity_fraction"] for region in field["regions"]) == pytest.approx(1.0)


def test_validator_rejects_error_fraction_tampering() -> None:
    case_ids = ["run_1"]
    report = build_aggregate_regional_diagnostics(
        case_ids=case_ids,
        case_evidence={"run_1": _case("run_1")},
        split_id="full",
    )
    report["supports"]["ahmedml-volume-three-geometric-regions-v1"]["fields"]["volume_velocity"]["regions"][0]["equal_entity"]["pooled"]["fraction_of_support_squared_error"] += 0.1
    with pytest.raises(AhmedMLRegionalAggregateError, match="error fractions"):
        validate_aggregate_regional_diagnostics(
            report,
            expected_case_ids=case_ids,
            expected_split_id="full",
        )
