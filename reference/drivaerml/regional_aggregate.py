"""Case envelopes and split aggregation for zero-weight regional diagnostics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from .regional_diagnostics import (
    REGION_DEFINITIONS,
    SURFACE_REGION_DEFINITION,
    VOLUME_REGION_DEFINITION,
    RegionalDiagnosticError,
    canonical_json_bytes,
    validate_reconstruction,
    validate_regional_report,
)

REGIONAL_DIAGNOSTICS_CONTRACT_SHA256 = (
    "2bfd372817989112642056e4c76cfb418dbdcee445c57ee20ca37ee9ca158583"
)

CASE_REGIONAL_ENVELOPE_SCHEMA = "drivaerml-regional-case-envelope-v2"
AGGREGATE_REGIONAL_REPORT_SCHEMA = "drivaerml-regional-aggregate-v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_RE = re.compile(r"run_[1-9][0-9]*\Z")
PREDICTION_SCOPE_FULL = "surface_and_volume"
PREDICTION_SCOPE_SURFACE_ONLY = "surface_only"
PREDICTION_SCOPES = frozenset(
    {PREDICTION_SCOPE_FULL, PREDICTION_SCOPE_SURFACE_ONLY}
)
_REQUIRED_FIELDS = {
    SURFACE_REGION_DEFINITION.definition_id: {
        "surface_pressure",
        "surface_wall_shear",
    },
    VOLUME_REGION_DEFINITION.definition_id: {
        "volume_pressure",
        "volume_velocity",
    },
}
_SCOPE_DEFINITION_IDS = {
    PREDICTION_SCOPE_FULL: tuple(_REQUIRED_FIELDS),
    PREDICTION_SCOPE_SURFACE_ONLY: (SURFACE_REGION_DEFINITION.definition_id,),
}
_SCORING_BOUNDARY = {
    "role": "report_only",
    "weight": 0.0,
    "official_metric_inputs_changed": False,
    "official_score_changed": False,
}


class RegionalAggregateError(ValueError):
    """Raised when retained regional cases cannot form one exact report."""


def _finite(value: object, label: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RegionalAggregateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RegionalAggregateError(f"{label} must be a finite number")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RegionalAggregateError("regional mean requires at least one value")
    return math.fsum(values) / len(values)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RegionalAggregateError("regional quantile requires at least one value")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _distribution(values: Sequence[float]) -> dict[str, object]:
    return {
        "method": "linear_order_statistics_over_complete_cases",
        "minimum": min(values),
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "maximum": max(values),
    }


def build_case_regional_envelope(
    *,
    case_id: str,
    case_report: Mapping[str, object],
    prediction_scope: str,
    volume_geometry_audit: Mapping[str, object] | None,
    official_additive_sums: Mapping[str, object],
) -> dict[str, object]:
    """Bind one strict case report to its immutable contract and geometry audit."""

    envelope: dict[str, object] = {
        "schema": CASE_REGIONAL_ENVELOPE_SCHEMA,
        "schema_version": 2,
        "status": "complete_report_only",
        "contract_sha256": REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
        "scoring": dict(_SCORING_BOUNDARY),
        "prediction_scope": prediction_scope,
        "case_report": dict(case_report),
        "volume_geometry_audit": (
            dict(volume_geometry_audit)
            if volume_geometry_audit is not None
            else None
        ),
    }
    validate_case_regional_envelope(
        envelope,
        expected_case_id=case_id,
        expected_additive_sums=official_additive_sums,
    )
    return envelope


def validate_case_regional_envelope(
    envelope: Mapping[str, object],
    *,
    expected_case_id: str | None = None,
    expected_additive_sums: Mapping[str, object] | None = None,
) -> None:
    """Fail closed on a missing, malformed, unbound, or partial case report."""

    if not isinstance(envelope, Mapping) or set(envelope) != {
        "schema",
        "schema_version",
        "status",
        "contract_sha256",
        "scoring",
        "prediction_scope",
        "case_report",
        "volume_geometry_audit",
    }:
        raise RegionalAggregateError("regional case envelope keys differ")
    if (
        envelope.get("schema") != CASE_REGIONAL_ENVELOPE_SCHEMA
        or envelope.get("schema_version") != 2
        or envelope.get("status") != "complete_report_only"
        or envelope.get("contract_sha256")
        != REGIONAL_DIAGNOSTICS_CONTRACT_SHA256
        or envelope.get("scoring") != _SCORING_BOUNDARY
    ):
        raise RegionalAggregateError("regional case envelope identity differs")
    prediction_scope = envelope.get("prediction_scope")
    if prediction_scope not in PREDICTION_SCOPES:
        raise RegionalAggregateError("regional case prediction scope differs")
    definition_ids = _SCOPE_DEFINITION_IDS[str(prediction_scope)]
    case_report = envelope.get("case_report")
    if not isinstance(case_report, Mapping):
        raise RegionalAggregateError("regional case report is absent")
    try:
        validate_regional_report(
            case_report,
            expected_case_id=expected_case_id,
            required_definition_ids=definition_ids,
        )
    except RegionalDiagnosticError as error:
        raise RegionalAggregateError(str(error)) from error
    case_id = case_report.get("case_id")
    if not isinstance(case_id, str) or _CASE_RE.fullmatch(case_id) is None:
        raise RegionalAggregateError("regional case ID is invalid")
    for definition_id in definition_ids:
        expected_fields = _REQUIRED_FIELDS[definition_id]
        support = case_report["supports"][definition_id]
        if set(support["fields"]) != expected_fields:
            raise RegionalAggregateError(
                f"{case_id} regional field membership differs for {definition_id}"
            )
    if expected_additive_sums is not None:
        expected_fields = set().union(
            *(_REQUIRED_FIELDS[definition_id] for definition_id in definition_ids)
        )
        if not isinstance(expected_additive_sums, Mapping) or set(
            expected_additive_sums
        ) != expected_fields:
            raise RegionalAggregateError("official additive field membership differs")
        try:
            for definition_id in definition_ids:
                field_ids = _REQUIRED_FIELDS[definition_id]
                support = case_report["supports"][definition_id]
                for field_id in field_ids:
                    sums = expected_additive_sums[field_id]
                    if not isinstance(sums, Mapping) or "uniform" not in sums:
                        raise RegionalAggregateError(
                            f"official additive sums are absent for {field_id}"
                        )
                    physical = sums.get("physical", sums["uniform"])
                    validate_reconstruction(
                        support["fields"][field_id]["statistics"],
                        expected_uniform=sums["uniform"],
                        expected_physical=physical,
                    )
        except RegionalDiagnosticError as error:
            raise RegionalAggregateError(
                "regional sums differ from official scored additive sums"
            ) from error
    audit = envelope.get("volume_geometry_audit")
    if prediction_scope == PREDICTION_SCOPE_SURFACE_ONLY:
        if audit is not None:
            raise RegionalAggregateError(
                "surface-only regional report must not contain a volume geometry audit"
            )
        canonical_json_bytes(envelope)
        return
    if not isinstance(audit, Mapping):
        raise RegionalAggregateError("regional volume geometry audit is absent")
    entity_count = audit.get("entity_count")
    volume_assignment = case_report["supports"][
        VOLUME_REGION_DEFINITION.definition_id
    ]["assignment"]
    if (
        not isinstance(entity_count, int)
        or isinstance(entity_count, bool)
        or entity_count != volume_assignment.get("entity_count")
        or audit.get("all_cells_assigned_exactly_once") is not True
    ):
        raise RegionalAggregateError("regional volume geometry coverage differs")
    coordinate_identity = audit.get("coordinate_identity_sha256")
    if (
        not isinstance(coordinate_identity, str)
        or _SHA256_RE.fullmatch(coordinate_identity) is None
    ):
        raise RegionalAggregateError("regional volume coordinate identity is invalid")
    if audit.get("assignment_identity_sha256") != volume_assignment.get("sha256"):
        raise RegionalAggregateError("regional volume assignment identity differs")
    labels = audit.get("region_labels")
    counts = audit.get("region_entity_count")
    if (
        labels != list(VOLUME_REGION_DEFINITION.region_ids)
        or not isinstance(counts, Mapping)
        or set(counts) != set(VOLUME_REGION_DEFINITION.region_ids)
        or any(
            not isinstance(counts[label], int)
            or isinstance(counts[label], bool)
            or counts[label] < 1
            for label in labels
        )
        or sum(counts.values()) != entity_count
    ):
        raise RegionalAggregateError("regional volume region counts differ")
    volume_fields = case_report["supports"][VOLUME_REGION_DEFINITION.definition_id][
        "fields"
    ]
    for field in volume_fields.values():
        field_counts = {
            row["region_id"]: row["entity_count"]
            for row in field["statistics"]["regions"]
        }
        if field_counts != dict(counts):
            raise RegionalAggregateError(
                "regional volume geometry and field counts differ"
            )
    canonical_json_bytes(envelope)


def _component_pooled(
    rows: Sequence[Mapping[str, object]],
    *,
    weighting: str,
    total_weight: float,
) -> dict[str, object]:
    components = list(rows[0][weighting]["component_squared_error"])
    if any(
        list(row[weighting]["component_squared_error"]) != components
        or list(row[weighting]["component_squared_truth"]) != components
        for row in rows
    ):
        raise RegionalAggregateError("regional component labels/order differ")
    error = {
        component: math.fsum(
            float(row[weighting]["component_squared_error"][component])
            for row in rows
        )
        for component in components
    }
    truth = {
        component: math.fsum(
            float(row[weighting]["component_squared_truth"][component])
            for row in rows
        )
        for component in components
    }
    total_error = math.fsum(error.values())
    return {
        "squared_error": error,
        "squared_truth": truth,
        "relative_l2_percent": {
            component: (
                100.0 * math.sqrt(error[component] / truth[component])
                if truth[component] > 0.0
                else None
            )
            for component in components
        },
        "rmse": {
            component: math.sqrt(error[component] / total_weight)
            for component in components
        },
        "fraction_of_region_squared_error": {
            component: error[component] / total_error if total_error > 0.0 else None
            for component in components
        },
    }


def _aggregate_weighting(
    rows: Sequence[Mapping[str, object]],
    *,
    weighting: str,
    support_squared_error: float,
) -> dict[str, object]:
    values = [row[weighting] for row in rows]
    if any(not isinstance(value, Mapping) for value in values):
        raise RegionalAggregateError("regional weighting row is invalid")
    total_weight = (
        float(sum(int(row["entity_count"]) for row in rows))
        if weighting == "equal_entity"
        else math.fsum(float(row["physical_weight"]) for row in rows)
    )
    absolute_error = math.fsum(float(value["absolute_error"]) for value in values)
    squared_error = math.fsum(float(value["squared_error"]) for value in values)
    squared_truth = math.fsum(float(value["squared_truth"]) for value in values)
    metric_names = ("relative_l2_percent", "mae", "rmse")
    macro: dict[str, float] = {}
    distributions: dict[str, object] = {}
    for name in metric_names:
        case_values = [
            float(value[name])
            for value in values
            if _finite(value.get(name), f"regional {weighting}.{name}", nullable=True)
            is not None
        ]
        if len(case_values) != len(rows):
            raise RegionalAggregateError(
                f"regional {weighting}.{name} is undefined for at least one case"
            )
        macro[name] = _mean(case_values)
        distributions[name] = _distribution(case_values)
    return {
        "pooled": {
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "squared_truth": squared_truth,
            "total_weight": total_weight,
            "relative_l2_percent": 100.0 * math.sqrt(squared_error / squared_truth),
            "mae": absolute_error / total_weight,
            "rmse": math.sqrt(squared_error / total_weight),
            "fraction_of_support_squared_error": (
                squared_error / support_squared_error
                if support_squared_error > 0.0
                else None
            ),
            "components": _component_pooled(
                rows, weighting=weighting, total_weight=total_weight
            ),
        },
        "macro_case_mean": macro,
        "case_distribution": distributions,
    }


def _aggregate_velocity(
    rows: Sequence[Mapping[str, object]], *, truth_squared_norm: float
) -> dict[str, object]:
    velocity = [row.get("velocity") for row in rows]
    if any(not isinstance(value, Mapping) for value in velocity):
        raise RegionalAggregateError("volume velocity region lacks diagnostics")
    magnitude = [value["speed_magnitude"] for value in velocity]
    direction = [value["direction"] for value in velocity]
    entity_count = sum(int(row["entity_count"]) for row in rows)
    magnitude_absolute = math.fsum(float(row["absolute_error"]) for row in magnitude)
    magnitude_squared = math.fsum(float(row["squared_error"]) for row in magnitude)
    direction_count = sum(int(row["defined_entity_count"]) for row in direction)
    cosine_sum = math.fsum(float(row["cosine_sum"]) for row in direction)
    angle_sum = math.fsum(float(row["angular_error_sum_degrees"]) for row in direction)
    thresholds = {float(row["minimum_speed_m_per_s"]) for row in direction}
    if len(thresholds) != 1:
        raise RegionalAggregateError("velocity direction thresholds differ across cases")
    return {
        "speed_magnitude": {
            "absolute_error": magnitude_absolute,
            "squared_error": magnitude_squared,
            "relative_l2_percent_with_vector_truth_norm": (
                100.0 * math.sqrt(magnitude_squared / truth_squared_norm)
                if truth_squared_norm > 0.0
                else None
            ),
            "mae": magnitude_absolute / entity_count,
            "rmse": math.sqrt(magnitude_squared / entity_count),
        },
        "direction": {
            "minimum_speed_m_per_s": thresholds.pop(),
            "defined_entity_count": direction_count,
            "defined_entity_fraction": direction_count / entity_count,
            "cosine_sum": cosine_sum,
            "angular_error_sum_degrees": angle_sum,
            "mean_cosine_similarity": (
                cosine_sum / direction_count if direction_count else None
            ),
            "mean_angular_error_degrees": (
                angle_sum / direction_count if direction_count else None
            ),
        },
    }


def _aggregate_field(
    case_fields: Sequence[Mapping[str, object]],
    *,
    definition_id: str,
) -> dict[str, object]:
    metadata = {
        key: case_fields[0][key]
        for key in ("quantity", "unit", "primary_weighting")
    }
    if any(
        any(field.get(key) != value for key, value in metadata.items())
        for field in case_fields
    ):
        raise RegionalAggregateError("regional field metadata differs across cases")
    statistics = [field["statistics"] for field in case_fields]
    definition = REGION_DEFINITIONS[definition_id]
    regions: list[dict[str, object]] = []
    support_error = {
        weighting: math.fsum(
            float(row["statistics"]["reconstruction"][f"reconstructed_{weighting}"]["squared_error"])
            for row in case_fields
        )
        for weighting in ("uniform", "physical")
    }
    total_entities = sum(int(row["entity_count"]) for row in statistics)
    total_physical_weight = math.fsum(
        float(row["reconstruction"]["reconstructed_physical"]["total_weight"])
        for row in statistics
    )
    for index, region_id in enumerate(definition.region_ids):
        rows = [row["regions"][index] for row in statistics]
        if any(row.get("region_id") != region_id for row in rows):
            raise RegionalAggregateError("regional field region order differs")
        entity_count = sum(int(row["entity_count"]) for row in rows)
        physical_weight = math.fsum(float(row["physical_weight"]) for row in rows)
        equal = _aggregate_weighting(
            rows,
            weighting="equal_entity",
            support_squared_error=support_error["uniform"],
        )
        physical = _aggregate_weighting(
            rows,
            weighting="physical",
            support_squared_error=support_error["physical"],
        )
        region: dict[str, object] = {
            "region_id": region_id,
            "entity_count": entity_count,
            "entity_fraction": entity_count / total_entities,
            "physical_weight": physical_weight,
            "physical_weight_fraction": physical_weight / total_physical_weight,
            "equal_entity": equal,
            "physical": physical,
        }
        if any("velocity" in row for row in rows):
            if not all("velocity" in row for row in rows):
                raise RegionalAggregateError("velocity diagnostic coverage differs")
            region["velocity"] = _aggregate_velocity(
                rows,
                truth_squared_norm=float(
                    equal["pooled"]["squared_truth"]
                ),
            )
        regions.append(region)
    return {
        **metadata,
        "case_count": len(case_fields),
        "entity_count": total_entities,
        "physical_weight": total_physical_weight,
        "regions": regions,
        "validation": {
            "all_case_reports_reconstructed_global_sums": True,
            "four_regions_mutually_exclusive_exhaustive": True,
            "macro_is_equal_case_mean": True,
            "pooled_is_report_only_not_official_case_reduction": True,
        },
    }


def aggregate_regional_diagnostics(
    case_documents: Sequence[Mapping[str, object]],
    *,
    case_ids: Sequence[str],
) -> dict[str, object]:
    """Aggregate exact ordered cases without touching any official score input."""

    expected = tuple(case_ids)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(not isinstance(case_id, str) or _CASE_RE.fullmatch(case_id) is None for case_id in expected)
        or len(case_documents) != len(expected)
    ):
        raise RegionalAggregateError("regional aggregate case set is invalid")
    by_id: dict[str, Mapping[str, object]] = {}
    for document in case_documents:
        case_id = document.get("case_id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise RegionalAggregateError("regional cases must have unique IDs")
        by_id[case_id] = document
    if set(by_id) != set(expected):
        raise RegionalAggregateError("regional case membership differs")
    envelopes: list[Mapping[str, object]] = []
    for case_id in expected:
        core = by_id[case_id].get("core")
        if not isinstance(core, Mapping):
            raise RegionalAggregateError(f"{case_id} core result is absent")
        envelope = core.get("report_only_regional_diagnostics")
        if not isinstance(envelope, Mapping):
            raise RegionalAggregateError(f"{case_id} regional report is absent")
        validate_case_regional_envelope(
            envelope,
            expected_case_id=case_id,
            expected_additive_sums=core.get("additive_sums"),
        )
        envelopes.append(envelope)
    prediction_scopes = {envelope.get("prediction_scope") for envelope in envelopes}
    if len(prediction_scopes) != 1:
        raise RegionalAggregateError("regional cases use different prediction scopes")
    prediction_scope = str(prediction_scopes.pop())
    definition_ids = _SCOPE_DEFINITION_IDS[prediction_scope]
    reports = [envelope["case_report"] for envelope in envelopes]
    supports: dict[str, object] = {}
    for definition_id in definition_ids:
        required_fields = _REQUIRED_FIELDS[definition_id]
        definition = REGION_DEFINITIONS[definition_id]
        case_supports = [report["supports"][definition_id] for report in reports]
        fields = {
            field_id: _aggregate_field(
                [support["fields"][field_id] for support in case_supports],
                definition_id=definition_id,
            )
            for field_id in sorted(required_fields)
        }
        supports[definition_id] = {
            "definition": definition.to_json(),
            "definition_sha256": definition.sha256,
            "assignments": [
                {
                    "case_id": case_id,
                    "entity_count": support["assignment"]["entity_count"],
                    "sha256": support["assignment"]["sha256"],
                }
                for case_id, support in zip(expected, case_supports, strict=True)
            ],
            "fields": fields,
        }
    aggregate: dict[str, object] = {
        "schema": AGGREGATE_REGIONAL_REPORT_SCHEMA,
        "schema_version": 2,
        "status": "complete_report_only",
        "dataset_id": "drivaerml",
        "prediction_scope": prediction_scope,
        "contract_sha256": REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
        "scoring": dict(_SCORING_BOUNDARY),
        "case_count": len(expected),
        "case_ids": list(expected),
        "supports": supports,
        "validation": {
            "exact_case_order_and_membership": True,
            "all_case_reports_strictly_validated": True,
            "all_regional_sums_reconstruct_unchanged_global_field_sums": True,
            "regional_values_consumed_by_official_score": False,
        },
    }
    validate_aggregate_regional_diagnostics(aggregate, expected_case_ids=expected)
    return aggregate


def validate_aggregate_regional_diagnostics(
    report: Mapping[str, object], *, expected_case_ids: Sequence[str] | None = None
) -> None:
    """Strictly validate a packaged split-level regional report offline."""

    if not isinstance(report, Mapping) or set(report) != {
        "schema",
        "schema_version",
        "status",
        "dataset_id",
        "prediction_scope",
        "contract_sha256",
        "scoring",
        "case_count",
        "case_ids",
        "supports",
        "validation",
    }:
        raise RegionalAggregateError("regional aggregate keys differ")
    if (
        report.get("schema") != AGGREGATE_REGIONAL_REPORT_SCHEMA
        or report.get("schema_version") != 2
        or report.get("status") != "complete_report_only"
        or report.get("dataset_id") != "drivaerml"
        or report.get("contract_sha256")
        != REGIONAL_DIAGNOSTICS_CONTRACT_SHA256
        or report.get("scoring") != _SCORING_BOUNDARY
    ):
        raise RegionalAggregateError("regional aggregate identity differs")
    prediction_scope = report.get("prediction_scope")
    if prediction_scope not in PREDICTION_SCOPES:
        raise RegionalAggregateError("regional aggregate prediction scope differs")
    definition_ids = _SCOPE_DEFINITION_IDS[str(prediction_scope)]
    case_ids = report.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or len(case_ids) != len(set(case_ids))
        or report.get("case_count") != len(case_ids)
        or any(not isinstance(case_id, str) or _CASE_RE.fullmatch(case_id) is None for case_id in case_ids)
        or (expected_case_ids is not None and list(expected_case_ids) != case_ids)
    ):
        raise RegionalAggregateError("regional aggregate case coverage differs")
    supports = report.get("supports")
    if not isinstance(supports, Mapping) or set(supports) != set(definition_ids):
        raise RegionalAggregateError("regional aggregate support membership differs")
    for definition_id in definition_ids:
        expected_fields = _REQUIRED_FIELDS[definition_id]
        definition = REGION_DEFINITIONS[definition_id]
        support = supports[definition_id]
        if not isinstance(support, Mapping) or set(support) != {
            "definition",
            "definition_sha256",
            "assignments",
            "fields",
        }:
            raise RegionalAggregateError("regional aggregate support keys differ")
        if (
            support.get("definition") != definition.to_json()
            or support.get("definition_sha256") != definition.sha256
            or not isinstance(support.get("fields"), Mapping)
            or set(support["fields"]) != expected_fields
        ):
            raise RegionalAggregateError("regional aggregate support definition differs")
        assignments = support.get("assignments")
        if (
            not isinstance(assignments, list)
            or [row.get("case_id") for row in assignments if isinstance(row, Mapping)]
            != case_ids
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"case_id", "entity_count", "sha256"}
                or not isinstance(row.get("entity_count"), int)
                or row["entity_count"] < 1
                or not isinstance(row.get("sha256"), str)
                or _SHA256_RE.fullmatch(row["sha256"]) is None
                for row in assignments
            )
        ):
            raise RegionalAggregateError("regional aggregate assignments differ")
        for field in support["fields"].values():
            if not isinstance(field, Mapping) or not isinstance(field.get("regions"), list):
                raise RegionalAggregateError("regional aggregate field is invalid")
            if [row.get("region_id") for row in field["regions"]] != list(
                definition.region_ids
            ):
                raise RegionalAggregateError("regional aggregate region order differs")
            for row in field["regions"]:
                for weighting in ("equal_entity", "physical"):
                    pooled = row.get(weighting, {}).get("pooled")
                    macro = row.get(weighting, {}).get("macro_case_mean")
                    if not isinstance(pooled, Mapping) or not isinstance(macro, Mapping):
                        raise RegionalAggregateError("regional aggregate metrics are absent")
                    for name in ("relative_l2_percent", "mae", "rmse"):
                        _finite(pooled.get(name), f"aggregate pooled {name}")
                        _finite(macro.get(name), f"aggregate macro {name}")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or validation != {
        "exact_case_order_and_membership": True,
        "all_case_reports_strictly_validated": True,
        "all_regional_sums_reconstruct_unchanged_global_field_sums": True,
        "regional_values_consumed_by_official_score": False,
    }:
        raise RegionalAggregateError("regional aggregate validation is incomplete")
    canonical_json_bytes(report)


__all__ = [
    "AGGREGATE_REGIONAL_REPORT_SCHEMA",
    "CASE_REGIONAL_ENVELOPE_SCHEMA",
    "RegionalAggregateError",
    "aggregate_regional_diagnostics",
    "build_case_regional_envelope",
    "validate_aggregate_regional_diagnostics",
    "validate_case_regional_envelope",
]
