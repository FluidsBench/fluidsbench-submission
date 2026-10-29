"""Strict envelope validation for zero-weight HiLiftAeroML region reports.

The numerical producer lives with the dataset evaluator because it consumes
the same streaming sufficient statistics as official field scoring.  This
repository-side validator freezes the portable aggregate envelope and keeps
it separate from the older DrivAerML regional contract.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence


REGIONAL_DIAGNOSTICS_CONTRACT_SHA256 = (
    "8cf926d06706b8cc7fd58d821f395bf8cb6565ef1f3aabe6be18e0a279101da4"
)
REGIONAL_DEFINITION_ID = "hiliftaeroml-native-geometric-regions-v1"
AGGREGATE_REGIONAL_REPORT_SCHEMA = (
    "hiliftaeroml-regional-diagnostics-aggregate-v2"
)
SURFACE_SUPPORT_ID = "surface_native_points"
VOLUME_SUPPORT_ID = "volume_native_valid_points"
SURFACE_REGION_IDS = (
    "nacelle_installation_envelope_proxy",
    "inboard_high_lift_envelope_proxy",
    "outboard_high_lift_envelope_proxy",
    "fuselage_tail_and_remaining",
)
VOLUME_REGION_IDS = (
    "near_airframe_sdf_band",
    "aft_airframe_wake_envelope_proxy",
    "near_aircraft_flow_envelope",
    "farfield_and_remaining",
)
SURFACE_FIELDS = frozenset(
    {
        "pressure",
        "tau_wall",
        "tau_wall_x",
        "tau_wall_y",
        "tau_wall_z",
        "tau_wall_magnitude",
    }
)
VOLUME_FIELDS = frozenset(
    {
        "pressure",
        "velocity",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "velocity_magnitude",
    }
)
_SAFE_FIELD = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_REQUIRED_TOP_LEVEL = {
    "schema",
    "contract_sha256",
    "dataset_id",
    "split_id",
    "prediction_scope",
    "case_count",
    "surface",
    "volume",
    "reconstruction",
}
_OPTIONAL_TOP_LEVEL = {"schema_version", "status", "definition_id", "case_ids", "scoring"}
_FIELD_KEYS = {"global", "regions", "pooled", "macro", "case_distribution"}
_REGION_KEYS = {
    "region_id",
    "entity_fraction",
    "weight_fraction",
    "squared_error_fraction",
    "whole_support_normalized_rmse_percent",
    "relative_l2_percent",
    "r2",
    "r2_status",
    "mae",
    "rmse",
    "case_macro",
    "case_distribution",
}
_CASE_MACRO_METRICS = {
    "whole_support_normalized_rmse_percent",
    "relative_l2_percent",
    "r2",
    "mae",
    "rmse",
}
_CASE_DISTRIBUTION_METRICS = {
    "whole_support_normalized_rmse_percent",
    "relative_l2_percent",
    "r2",
}


class HiLiftRegionalAggregateError(ValueError):
    """Raised when a HiLift regional report is partial or contract-incompatible."""


def _finite_or_none(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HiLiftRegionalAggregateError(f"{label} must be finite or null")
    result = float(value)
    if not math.isfinite(result):
        raise HiLiftRegionalAggregateError(f"{label} must be finite or null")
    return result


def _finite_json_tree(value: object, label: str) -> None:
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int | float):
        _finite_or_none(value, label)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise HiLiftRegionalAggregateError(f"{label} has an invalid key")
            _finite_json_tree(child, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _finite_json_tree(child, f"{label}[{index}]")
        return
    raise HiLiftRegionalAggregateError(f"{label} is not canonical JSON data")


def _fraction(value: object, label: str) -> float | None:
    result = _finite_or_none(value, label)
    if result is not None and not 0.0 <= result <= 1.0:
        raise HiLiftRegionalAggregateError(f"{label} must lie in [0, 1] or be null")
    return result


def _defined_case_count(value: object, label: str, *, case_count: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= case_count
    ):
        raise HiLiftRegionalAggregateError(
            f"{label} must be an integer in [0, {case_count}]"
        )
    return value


def _case_macro(
    value: object,
    label: str,
    *,
    case_count: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _CASE_MACRO_METRICS:
        raise HiLiftRegionalAggregateError(f"{label} metric inventory differs")
    for metric_id, raw in value.items():
        metric_label = f"{label}.{metric_id}"
        if not isinstance(raw, Mapping) or set(raw) != {
            "value",
            "defined_case_count",
        }:
            raise HiLiftRegionalAggregateError(f"{metric_label} keys differ")
        count = _defined_case_count(
            raw["defined_case_count"],
            f"{metric_label}.defined_case_count",
            case_count=case_count,
        )
        observed = _finite_or_none(raw["value"], f"{metric_label}.value")
        if (count == 0) != (observed is None):
            raise HiLiftRegionalAggregateError(
                f"{metric_label} value/count nullability differs"
            )
        if (
            observed is not None
            and metric_id != "r2"
            and observed < 0.0
        ):
            raise HiLiftRegionalAggregateError(
                f"{metric_label}.value must be non-negative"
            )


def _case_distribution(
    value: object,
    label: str,
    *,
    case_count: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _CASE_DISTRIBUTION_METRICS:
        raise HiLiftRegionalAggregateError(f"{label} metric inventory differs")
    for metric_id, raw in value.items():
        metric_label = f"{label}.{metric_id}"
        required = {
            "defined_case_count",
            "minimum",
            "median",
            "p90",
            "maximum",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise HiLiftRegionalAggregateError(f"{metric_label} keys differ")
        count = _defined_case_count(
            raw["defined_case_count"],
            f"{metric_label}.defined_case_count",
            case_count=case_count,
        )
        summary = [
            _finite_or_none(raw[key], f"{metric_label}.{key}")
            for key in ("minimum", "median", "p90", "maximum")
        ]
        if count == 0:
            if any(item is not None for item in summary):
                raise HiLiftRegionalAggregateError(
                    f"{metric_label} empty distribution must be null"
                )
            continue
        if any(item is None for item in summary):
            raise HiLiftRegionalAggregateError(
                f"{metric_label} populated distribution must be finite"
            )
        numeric = [float(item) for item in summary if item is not None]
        if metric_id != "r2" and any(item < 0.0 for item in numeric):
            raise HiLiftRegionalAggregateError(
                f"{metric_label} values must be non-negative"
            )
        if numeric != sorted(numeric):
            raise HiLiftRegionalAggregateError(
                f"{metric_label} quantiles are not ordered"
            )


def _support(
    value: object,
    *,
    label: str,
    expected_support_id: str,
    expected_regions: tuple[str, ...],
    allowed_fields: frozenset[str],
    required_primary_fields: frozenset[str],
    case_count: int,
) -> int:
    if not isinstance(value, Mapping) or set(value) != {
        "support_id",
        "region_order",
        "fields",
    }:
        raise HiLiftRegionalAggregateError(f"{label} keys differ from the contract")
    if value.get("support_id") != expected_support_id:
        raise HiLiftRegionalAggregateError(f"{label}.support_id differs")
    if value.get("region_order") != list(expected_regions):
        raise HiLiftRegionalAggregateError(f"{label}.region_order differs")
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise HiLiftRegionalAggregateError(f"{label}.fields must be an object")
    field_ids = set(fields)
    if not required_primary_fields.issubset(field_ids) or not field_ids.issubset(
        allowed_fields
    ):
        raise HiLiftRegionalAggregateError(
            f"{label}.fields differ: observed={sorted(field_ids)}"
        )
    checked_regions = 0
    for field_id, field in fields.items():
        if not isinstance(field_id, str) or _SAFE_FIELD.fullmatch(field_id) is None:
            raise HiLiftRegionalAggregateError(f"{label} has an invalid field ID")
        field_label = f"{label}.fields.{field_id}"
        if not isinstance(field, Mapping) or set(field) != _FIELD_KEYS:
            raise HiLiftRegionalAggregateError(f"{field_label} keys differ")
        for view in ("global", "pooled", "macro", "case_distribution"):
            if not isinstance(field[view], Mapping):
                raise HiLiftRegionalAggregateError(f"{field_label}.{view} must be an object")
            _finite_json_tree(field[view], f"{field_label}.{view}")
        regions = field["regions"]
        if not isinstance(regions, list) or len(regions) != len(expected_regions):
            raise HiLiftRegionalAggregateError(f"{field_label}.regions count differs")
        if [row.get("region_id") for row in regions if isinstance(row, Mapping)] != list(
            expected_regions
        ):
            raise HiLiftRegionalAggregateError(f"{field_label}.regions order differs")
        fractions: dict[str, list[float]] = {
            "entity_fraction": [],
            "weight_fraction": [],
            "squared_error_fraction": [],
        }
        whole_support_normalized: list[float] = []
        for row in regions:
            if not isinstance(row, Mapping) or set(row) != _REGION_KEYS:
                raise HiLiftRegionalAggregateError(
                    f"{field_label} region keys differ"
                )
            for name in fractions:
                observed = _fraction(row[name], f"{field_label}.{row['region_id']}.{name}")
                if observed is not None:
                    fractions[name].append(observed)
            for name in (
                "whole_support_normalized_rmse_percent",
                "relative_l2_percent",
                "mae",
                "rmse",
            ):
                observed = _finite_or_none(
                    row[name], f"{field_label}.{row['region_id']}.{name}"
                )
                if observed is not None and observed < 0.0:
                    raise HiLiftRegionalAggregateError(
                        f"{field_label}.{row['region_id']}.{name} must be non-negative"
                    )
                if (
                    name == "whole_support_normalized_rmse_percent"
                    and observed is not None
                ):
                    whole_support_normalized.append(observed)
            r2 = _finite_or_none(
                row["r2"], f"{field_label}.{row['region_id']}.r2"
            )
            r2_status = row["r2_status"]
            if r2_status not in {"ok", "empty", "zero_target_variance"}:
                raise HiLiftRegionalAggregateError(
                    f"{field_label}.{row['region_id']}.r2_status differs"
                )
            if (r2 is None) != (r2_status != "ok"):
                raise HiLiftRegionalAggregateError(
                    f"{field_label}.{row['region_id']} r2 value/status differs"
                )
            _case_macro(
                row["case_macro"],
                f"{field_label}.{row['region_id']}.case_macro",
                case_count=case_count,
            )
            _case_distribution(
                row["case_distribution"],
                f"{field_label}.{row['region_id']}.case_distribution",
                case_count=case_count,
            )
            checked_regions += 1
        for fraction_id, values in fractions.items():
            if len(values) == len(expected_regions) and not math.isclose(
                math.fsum(values), 1.0, rel_tol=5.0e-12, abs_tol=1.0e-12
            ):
                raise HiLiftRegionalAggregateError(
                    f"{field_label}.{fraction_id} does not sum to one"
                )
        global_relative_l2 = _finite_or_none(
            field["global"].get("relative_l2_percent"),
            f"{field_label}.global.relative_l2_percent",
        )
        if (
            global_relative_l2 is not None
            and len(fractions["weight_fraction"]) == len(expected_regions)
            and len(whole_support_normalized) == len(expected_regions)
        ):
            reconstructed_relative_l2 = math.sqrt(
                math.fsum(
                    fraction * value * value
                    for fraction, value in zip(
                        fractions["weight_fraction"],
                        whole_support_normalized,
                        strict=True,
                    )
                )
            )
            if not math.isclose(
                reconstructed_relative_l2,
                global_relative_l2,
                rel_tol=5.0e-12,
                abs_tol=1.0e-12,
            ):
                raise HiLiftRegionalAggregateError(
                    f"{field_label} whole-support-normalized regional RMSE "
                    "does not reconstruct global relative L2"
                )
    return checked_regions


def validate_aggregate_regional_diagnostics(
    report: Mapping[str, object],
    *,
    expected_case_ids: Sequence[str],
    expected_split_id: str | None = None,
) -> None:
    """Validate one complete, report-only HiLift regional aggregate."""

    if not isinstance(report, Mapping):
        raise HiLiftRegionalAggregateError("regional report must be an object")
    keys = set(report)
    if not _REQUIRED_TOP_LEVEL.issubset(keys) or not keys.issubset(
        _REQUIRED_TOP_LEVEL | _OPTIONAL_TOP_LEVEL
    ):
        raise HiLiftRegionalAggregateError("regional report top-level keys differ")
    if (
        report.get("schema") != AGGREGATE_REGIONAL_REPORT_SCHEMA
        or report.get("contract_sha256") != REGIONAL_DIAGNOSTICS_CONTRACT_SHA256
        or report.get("dataset_id") != "hiliftaeroml"
        or report.get("prediction_scope") != "surface_and_volume"
    ):
        raise HiLiftRegionalAggregateError("regional report identity differs")
    if "schema_version" in report and report.get("schema_version") != 2:
        raise HiLiftRegionalAggregateError("regional report schema_version differs")
    if "status" in report and report.get("status") != "complete_report_only":
        raise HiLiftRegionalAggregateError("regional report status differs")
    if "definition_id" in report and report.get("definition_id") != REGIONAL_DEFINITION_ID:
        raise HiLiftRegionalAggregateError("regional report definition_id differs")
    if expected_split_id is not None and report.get("split_id") != expected_split_id:
        raise HiLiftRegionalAggregateError("regional report split_id differs")
    if (
        not expected_case_ids
        or len(expected_case_ids) != len(set(expected_case_ids))
        or report.get("case_count") != len(expected_case_ids)
    ):
        raise HiLiftRegionalAggregateError("regional report case count differs")
    if "case_ids" in report and report.get("case_ids") != list(expected_case_ids):
        raise HiLiftRegionalAggregateError("regional report case order differs")
    if "scoring" in report and report.get("scoring") != {
        "role": "report_only",
        "weight": 0.0,
        "official_metric_inputs_changed": False,
        "official_score_changed": False,
    }:
        raise HiLiftRegionalAggregateError("regional report scoring boundary differs")

    checked_regions = _support(
        report["surface"],
        label="surface",
        expected_support_id=SURFACE_SUPPORT_ID,
        expected_regions=SURFACE_REGION_IDS,
        allowed_fields=SURFACE_FIELDS,
        required_primary_fields=frozenset({"pressure", "tau_wall"}),
        case_count=len(expected_case_ids),
    )
    checked_regions += _support(
        report["volume"],
        label="volume",
        expected_support_id=VOLUME_SUPPORT_ID,
        expected_regions=VOLUME_REGION_IDS,
        allowed_fields=VOLUME_FIELDS,
        required_primary_fields=frozenset({"pressure", "velocity"}),
        case_count=len(expected_case_ids),
    )
    reconstruction = report["reconstruction"]
    expected_reconstruction_keys = {
        "status",
        "relative_tolerance",
        "absolute_tolerance",
        "checked_case_count",
        "checked_field_count",
    }
    if not isinstance(reconstruction, Mapping) or set(reconstruction) != expected_reconstruction_keys:
        raise HiLiftRegionalAggregateError("regional reconstruction keys differ")
    if (
        reconstruction.get("status") != "pass"
        or reconstruction.get("relative_tolerance") != 5.0e-12
        or reconstruction.get("absolute_tolerance") != 1.0e-12
        or reconstruction.get("checked_case_count") != len(expected_case_ids)
        or not isinstance(reconstruction.get("checked_field_count"), int)
        or isinstance(reconstruction.get("checked_field_count"), bool)
        or reconstruction["checked_field_count"] < 4
        or checked_regions < 16
    ):
        raise HiLiftRegionalAggregateError("regional reconstruction evidence differs")
    _finite_json_tree(report, "regional report")
