from __future__ import annotations

import math

import pytest

from reference.ahmedml.contract import REGION_DEFINITION_SHA256
from reference.ahmedml.regional_aggregate import (
    SURFACE_REGIONAL_SUPPORT_ID,
    SURFACE_REGION_IDS,
    VOLUME_REGIONAL_SUPPORT_ID,
    VOLUME_REGION_IDS,
    AhmedMLRegionalAggregateError,
    build_aggregate_regional_diagnostics,
    validate_aggregate_regional_diagnostics,
)


def _metrics(
    *, absolute: float, squared: float, truth: float, weight: float
) -> dict[str, float]:
    return {
        "absolute_error": absolute,
        "squared_error": squared,
        "squared_truth": truth,
        "total_weight": weight,
        "relative_l2_percent": 100.0 * math.sqrt(squared / truth),
        "mae": absolute / weight,
        "rmse": math.sqrt(squared / weight),
    }


def _field(
    *, field_index: int, region_ids: tuple[str, ...], scale: float
) -> tuple[dict[str, object], dict[str, object]]:
    region_counts = tuple(range(2, 2 + len(region_ids)))
    physical_weights = tuple(0.25 * (index + 1) for index in range(len(region_ids)))
    regions: dict[str, object] = {}
    uniform_sums = {
        name: 0.0
        for name in ("absolute_error", "squared_error", "squared_truth", "total_weight")
    }
    physical_sums = dict(uniform_sums)
    for region_index, (region_id, count, physical_weight) in enumerate(
        zip(region_ids, region_counts, physical_weights, strict=True), start=1
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
    global_field = {
        "entity_count": sum(region_counts),
        "uniform": {
            **uniform_sums,
            **_metrics(
                absolute=uniform_sums["absolute_error"],
                squared=uniform_sums["squared_error"],
                truth=uniform_sums["squared_truth"],
                weight=uniform_sums["total_weight"],
            ),
        },
        "physical": {
            **physical_sums,
            **_metrics(
                absolute=physical_sums["absolute_error"],
                squared=physical_sums["squared_error"],
                truth=physical_sums["squared_truth"],
                weight=physical_sums["total_weight"],
            ),
        },
    }
    return regions, global_field


def _case(case_id: str, scale: float = 1.0) -> dict[str, object]:
    regional: dict[str, object] = {
        "contract_sha256": REGION_DEFINITION_SHA256,
        "definition_id": "ahmedml-native-regions-v2-candidate",
        "ranking_effect": "none",
    }
    fields: dict[str, object] = {}
    definitions = (
        ("surface_pressure", SURFACE_REGION_IDS),
        ("surface_wall_shear", SURFACE_REGION_IDS),
        ("volume_pressure", VOLUME_REGION_IDS),
        ("volume_velocity", VOLUME_REGION_IDS),
    )
    for field_index, (field_id, region_ids) in enumerate(definitions, start=1):
        regions, global_field = _field(
            field_index=field_index, region_ids=region_ids, scale=scale
        )
        regional[field_id] = regions
        fields[field_id] = global_field
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
    assert set(report["supports"]) == {
        SURFACE_REGIONAL_SUPPORT_ID,
        VOLUME_REGIONAL_SUPPORT_ID,
    }
    surface = report["supports"][SURFACE_REGIONAL_SUPPORT_ID]["fields"][
        "surface_pressure"
    ]
    assert surface["primary_weighting"] == "physical"
    assert [region["region_id"] for region in surface["regions"]] == list(
        SURFACE_REGION_IDS
    )
    volume = report["supports"][VOLUME_REGIONAL_SUPPORT_ID]["fields"][
        "volume_pressure"
    ]
    assert volume["primary_weighting"] == "equal_entity"
    assert [region["region_id"] for region in volume["regions"]] == list(
        VOLUME_REGION_IDS
    )
    assert sum(region["entity_fraction"] for region in volume["regions"]) == pytest.approx(
        1.0
    )


def test_validator_rejects_surface_error_fraction_tampering() -> None:
    case_ids = ["run_1"]
    report = build_aggregate_regional_diagnostics(
        case_ids=case_ids,
        case_evidence={"run_1": _case("run_1")},
        split_id="full",
    )
    report["supports"][SURFACE_REGIONAL_SUPPORT_ID]["fields"][
        "surface_wall_shear"
    ]["regions"][0]["physical"]["pooled"][
        "fraction_of_support_squared_error"
    ] += 0.1
    with pytest.raises(AhmedMLRegionalAggregateError, match="error fractions"):
        validate_aggregate_regional_diagnostics(
            report,
            expected_case_ids=case_ids,
            expected_split_id="full",
        )


def test_builder_rejects_missing_surface_region() -> None:
    evidence = _case("run_1")
    del evidence["report_only_regional_diagnostics"]["surface_pressure"][
        "upward_facing"
    ]
    with pytest.raises(AhmedMLRegionalAggregateError, match="membership differs"):
        build_aggregate_regional_diagnostics(
            case_ids=["run_1"],
            case_evidence={"run_1": evidence},
            split_id="full",
        )
