#!/usr/bin/env python3
"""Build a native FluidsBench DrivAerML replay from retained Transolver output.

The input is a complete, already-evaluated Transolver result.  It is used only
as a local build input: the generated package is expressed entirely in the
FluidsBench DrivAerML contract, current native-source pin, and current profile
support identities.  No source-archive identifier is copied into the package.

The retained result contains complete field sufficient statistics, force
coefficients, and materialized profile predictions, but not the transient
sparse raw-cell gather buffers.  The candidate evaluator therefore uses the
explicit retained-inference diagnostic representation rather than fabricating
those unavailable buffers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.coordinate_identity import (  # noqa: E402
    CoordinateIdentityError,
    coordinate_array_identity_sha256,
)
from reference.drivaerml.dataset_scorer import (  # noqa: E402
    CONSTANT_SERIES_SUPPORT_INDEX_SHA256,
    OFFICIAL_IMMUTABLE_PINS,
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_CONTRACT_SHA256,
    RELATIVE_PROFILE_FORMAT,
    RELATIVE_PROFILE_SERIES_PER_CASE,
    RELATIVE_SERIES_SUPPORT_INDEX_SHA256,
    RETAINED_INFERENCE_DIAGNOSTIC_CASE_SCHEMA,
    RETAINED_INFERENCE_DIAGNOSTIC_CASE_STATUS,
    _constant_series_support_index,
    _load_contract,
    _relative_profile_expected_keys,
    _relative_series_support_index,
    evaluate_candidate_dataset,
    schema_v3_relative_profile_chunks_candidate_adapter,
    write_candidate_dataset_evidence,
    write_regional_diagnostics,
    write_schema_v3_profile_chunks_candidate,
)
from reference.scores import (  # noqa: E402
    composite_component_group_scores,
    composite_overall_score,
)


# This is an input-only identity check.  The generated package does not retain
# this source schema name or any source-package provenance fields.
SOURCE_RESULT_SCHEMA = "autocfd5-aiml-drivaerml-result-v1"
SOURCE_CASE_SCHEMA = "autocfd5-aiml-drivaerml-case-result-v1"
SOURCE_CASE_SCHEMA_VERSION = 1
SOURCE_CORE_SCHEMA = "autocfd5-aiml-drivaerml-case-evaluation-v2"
SOURCE_CORE_SCHEMA_VERSION = 2
SOURCE_PROFILE_SCHEMA = "autocfd5-aiml-profile-case-result-v1"
SOURCE_PROFILE_SCHEMA_VERSION = 1

NATIVE_CORE_SCHEMA = "drivaerml-candidate-case-evaluation-v2"
NATIVE_CORE_STATUS = "candidate_evaluator_evidence_not_official_submission"
NATIVE_PACKAGE_SCHEMA_VERSION = "1.0"
DEFAULT_GENERATED_AT = "2026-09-09T00:00:00Z"


class NativeReplayError(ValueError):
    """Raised when retained output cannot prove a native FluidsBench replay."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeReplayError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> Any:
    raise NativeReplayError(f"JSON contains forbidden non-finite token {token}")


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except NativeReplayError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeReplayError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise NativeReplayError(f"{label} {path} must contain a JSON object")
    return value


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeReplayError(f"value is not finite canonical JSON: {error}") from error


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise NativeReplayError(f"{label} keys must be {sorted(keys)}; observed {actual}")
    return value


def require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeReplayError(f"{label} must be a non-empty string")
    return value


def require_finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeReplayError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise NativeReplayError(f"{label} must be finite{' and nonnegative' if nonnegative else ''}")
    return result


def require_same_float(actual: object, expected: float, label: str) -> float:
    result = require_finite(actual, label)
    if not math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise NativeReplayError(f"{label} differs from its retained reduction")
    return result


def _source_case_ids(result: Mapping[str, Any]) -> tuple[str, ...]:
    split = result.get("split")
    if not isinstance(split, dict):
        raise NativeReplayError("retained result has no split record")
    values = split.get("test_case_ids")
    if not isinstance(values, list) or not values:
        raise NativeReplayError("retained result split test_case_ids must be non-empty")
    case_ids = tuple(require_string(value, "retained result case ID") for value in values)
    if len(case_ids) != len(set(case_ids)):
        raise NativeReplayError("retained result split contains duplicate case IDs")
    if split.get("test_case_count") != len(case_ids):
        raise NativeReplayError("retained result split test_case_count differs from IDs")
    return case_ids


def _segments_arc_length(series: Mapping[str, Any], *, label: str) -> tuple[int, float]:
    segments = series.get("segments")
    if not isinstance(segments, list) or not segments:
        raise NativeReplayError(f"{label} needs at least one explicit support segment")
    lengths: list[float] = []
    for position, raw in enumerate(segments):
        if not isinstance(raw, dict):
            raise NativeReplayError(f"{label} segment {position} must be an object")
        start = require_finite(raw.get("coordinate_start"), f"{label} segment start")
        stop = require_finite(raw.get("coordinate_stop"), f"{label} segment stop")
        # An isolated retained sample is represented as a zero-length segment.
        # It contributes no arc to the reduction but must not be discarded: its
        # presence is part of the exact support/gap topology.
        if stop < start:
            raise NativeReplayError(f"{label} segment {position} is descending")
        lengths.append(stop - start)
    return len(segments), math.fsum(lengths)


def _support_identity(
    *,
    case_id: str,
    family_id: str,
    station_id: str,
    representation: str,
    constant_support: Mapping[tuple[str, str, str], Mapping[str, object]],
    relative_support: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> Mapping[str, object]:
    if family_id in {"drivaerml-autocfd5-constant-v1", "drivaerml_cp_constant_v1"}:
        if representation != "materialized":
            raise NativeReplayError(
                f"{case_id}/{family_id}/{station_id} constant profile must be materialized"
            )
        identity = constant_support.get((case_id, family_id, station_id))
    else:
        identity = relative_support.get((case_id, family_id, station_id))
    if identity is None or identity.get("representation") != representation:
        raise NativeReplayError(
            f"{case_id}/{family_id}/{station_id} has no matching FluidsBench support"
        )
    return identity


def convert_namespaced_profiles(
    *,
    case_id: str,
    profiles: Mapping[str, Any],
    constant_support: Mapping[tuple[str, str, str], Mapping[str, object]],
    relative_support: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rebind exact retained profile arrays to FluidsBench support identities."""

    raw_series = profiles.get("series")
    if not isinstance(raw_series, list) or len(raw_series) != RELATIVE_PROFILE_SERIES_PER_CASE:
        raise NativeReplayError(
            f"{case_id} must retain exactly {RELATIVE_PROFILE_SERIES_PER_CASE} profile series"
        )
    expected_keys = tuple(_relative_profile_expected_keys())
    expected_key_set = set(expected_keys)
    converted: list[dict[str, object]] = []
    observed: list[tuple[str, str, str, str, str]] = []
    constant_cp_supports: dict[str, str] = {}
    deferred_aliases: list[tuple[dict[str, object], str]] = []
    for position, raw_item in enumerate(raw_series):
        if not isinstance(raw_item, dict):
            raise NativeReplayError(f"{case_id} profile series {position} must be an object")
        panel_id = require_string(raw_item.get("panel_id"), f"{case_id} profile panel")
        family_id = require_string(raw_item.get("family_id"), f"{case_id} profile family")
        station_id = require_string(raw_item.get("station_id"), f"{case_id} profile station")
        quantity_id = require_string(raw_item.get("quantity_id"), f"{case_id} profile quantity")
        representation = require_string(
            raw_item.get("representation"), f"{case_id} profile representation"
        )
        key = (panel_id, family_id, station_id, quantity_id, representation)
        observed.append(key)
        if key not in expected_key_set:
            raise NativeReplayError(f"{case_id} has an undeclared profile series {key}")
        identity = _support_identity(
            case_id=case_id,
            family_id=family_id,
            station_id=station_id,
            representation=representation,
            constant_support=constant_support,
            relative_support=relative_support,
        )
        placement_mode = "relative" if family_id in {
            "drivaerml-velocity-relative-v3",
            "drivaerml_cp_relative_v1",
        } else "constant"
        scoring_role = "report_only" if placement_mode == "relative" else "inherits_parent_candidate"
        result: dict[str, object] = {
            "panel_id": panel_id,
            "family_id": family_id,
            "placement_mode": placement_mode,
            "station_id": station_id,
            "quantity_id": quantity_id,
            "scoring_role": scoring_role,
            "representation": representation,
            "placement_receipt_identity_sha256": identity[
                "placement_receipt_identity_sha256"
            ],
        }
        if representation == "materialized":
            coordinates = raw_item.get("coordinate")
            predictions = raw_item.get("prediction")
            if (
                not isinstance(coordinates, list)
                or not isinstance(predictions, list)
                or len(coordinates) < 2
                or len(coordinates) != len(predictions)
            ):
                raise NativeReplayError(
                    f"{case_id}/{family_id}/{station_id} has malformed retained arrays"
                )
            try:
                coordinate_identity = coordinate_array_identity_sha256(coordinates)
            except CoordinateIdentityError as error:
                raise NativeReplayError(
                    f"{case_id}/{family_id}/{station_id} has invalid coordinates: {error}"
                ) from error
            for value_index, value in enumerate(predictions):
                require_finite(
                    value,
                    f"{case_id}/{family_id}/{station_id} prediction {value_index}",
                )
            if (
                len(coordinates) != identity.get("coordinate_count")
                or coordinate_identity != identity.get("coordinate_identity_sha256")
            ):
                raise NativeReplayError(
                    f"{case_id}/{family_id}/{station_id} coordinates do not match FluidsBench support"
                )
            coordinate_id = require_string(
                raw_item.get("coordinate_id"),
                f"{case_id}/{family_id}/{station_id} coordinate_id",
            )
            coordinate_unit = require_string(
                raw_item.get("coordinate_unit"),
                f"{case_id}/{family_id}/{station_id} coordinate_unit",
            )
            result.update(
                {
                    "coordinate_id": coordinate_id,
                    "coordinate_unit": coordinate_unit,
                    "coordinate": coordinates,
                    "prediction": predictions,
                    "support_identity_sha256": identity["support_identity_sha256"],
                }
            )
            if (
                panel_id == "pressure_profiles"
                and family_id == "drivaerml_cp_constant_v1"
                and station_id in {"upperbody_centerline", "underbody_centerline"}
            ):
                constant_cp_supports[station_id] = str(identity["support_identity_sha256"])
        elif representation == "shared_alias":
            if station_id not in {"upperbody_centerline", "underbody_centerline"}:
                raise NativeReplayError(
                    f"{case_id}/{family_id}/{station_id} is not a supported shared alias"
                )
            deferred_aliases.append((result, station_id))
        else:
            raise NativeReplayError(
                f"{case_id}/{family_id}/{station_id} representation is unsupported"
            )
        converted.append(result)
    if tuple(observed) != expected_keys:
        raise NativeReplayError(f"{case_id} retained profile ordering differs from FluidsBench")
    shared_ids = {
        "upperbody_centerline": "drivaerml-cp-upperbody-centerline-y0-v1",
        "underbody_centerline": "drivaerml-cp-underbody-centerline-y0-v1",
    }
    for alias, station_id in deferred_aliases:
        support_identity = constant_cp_supports.get(station_id)
        if support_identity is None:
            raise NativeReplayError(
                f"{case_id} shared Cp alias {station_id} has no constant support"
            )
        alias["shared_support_ref"] = {
            "shared_support_id": shared_ids[station_id],
            "canonical_family_id": "drivaerml_cp_constant_v1",
            "canonical_station_id": station_id,
            "canonical_support_identity_sha256": support_identity,
        }
    return converted


def convert_core_document(
    source_core: Mapping[str, Any], *, native_source_pin_sha256: str
) -> dict[str, object]:
    result = copy.deepcopy(dict(source_core))
    if (
        result.get("schema") != SOURCE_CORE_SCHEMA
        or result.get("schema_version") != SOURCE_CORE_SCHEMA_VERSION
        or result.get("status") != "complete_submitter_evaluation"
        or result.get("official_submission") is not False
    ):
        raise NativeReplayError("retained core evidence has an unexpected identity")
    source = result.get("source")
    if not isinstance(source, dict):
        raise NativeReplayError("retained core evidence has no source record")
    result["schema"] = NATIVE_CORE_SCHEMA
    result["schema_version"] = 2
    result["status"] = NATIVE_CORE_STATUS
    source["native_source_pin_sha256"] = native_source_pin_sha256
    return result


def _block_rmse(rows: object, *, count: int, label: str) -> list[float]:
    if not isinstance(rows, list) or len(rows) != count:
        raise NativeReplayError(f"{label} must contain exactly {count} retained blocks")
    result: list[float] = []
    for position, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3:
            raise NativeReplayError(f"{label} block {position} must be a three-value tuple")
        mean_squared_error = require_finite(row[2], f"{label} block {position} MSE", nonnegative=True)
        result.append(math.sqrt(mean_squared_error))
    return result


def build_retained_diagnostic_document(
    *,
    case_id: str,
    source_profiles: Mapping[str, Any],
    native_core: Mapping[str, Any],
    contract: Any,
) -> dict[str, object]:
    """Create transparent candidate diagnostic evidence from retained reductions."""

    statistics = source_profiles.get("metric_statistics")
    if not isinstance(statistics, dict):
        raise NativeReplayError(f"{case_id} retained profiles have no metric statistics")
    velocity_rmse = _block_rmse(
        statistics.get("velocity_profile_r2_blocks"),
        count=len(contract.profile_velocity_line_ids),
        label=f"{case_id} retained velocity profile",
    )
    cp_rmse = _block_rmse(
        statistics.get("cp_cut_r2_blocks"),
        count=len(contract.profile_cp_cut_ids),
        label=f"{case_id} retained Cp cut",
    )
    require_same_float(
        statistics.get("velocity_profile_uinf_rmse"),
        math.fsum(velocity_rmse) / len(velocity_rmse),
        f"{case_id} retained velocity profile mean",
    )
    require_same_float(
        statistics.get("cp_cut_rmse"),
        math.fsum(cp_rmse) / len(cp_rmse),
        f"{case_id} retained Cp cut mean",
    )
    station_series = {
        str(series.get("station_id")): series
        for series in source_profiles.get("series", [])
        if isinstance(series, dict)
        and series.get("family_id") == "drivaerml-autocfd5-constant-v1"
    }
    pressure_series = {
        str(series.get("station_id")): series
        for series in source_profiles.get("series", [])
        if isinstance(series, dict) and series.get("family_id") == "drivaerml_cp_constant_v1"
    }
    if set(station_series) != set(contract.profile_velocity_station_ids):
        raise NativeReplayError(f"{case_id} retained velocity stations differ from FluidsBench")
    if set(pressure_series) != set(contract.profile_cp_cut_ids):
        raise NativeReplayError(f"{case_id} retained Cp stations differ from FluidsBench")
    line_rows = []
    by_profile_id: dict[str, dict[str, object]] = {}
    for profile_id, station_id, sample_count, rmse in zip(
        contract.profile_velocity_line_ids,
        contract.profile_velocity_station_ids,
        contract.profile_velocity_line_sample_counts,
        velocity_rmse,
        strict=True,
    ):
        _segment_count, arc_length = _segments_arc_length(
            station_series[station_id], label=f"{case_id}/{station_id}"
        )
        row = {
            "profile_id": profile_id,
            "sample_count": sample_count,
            "arc_length_m": arc_length,
            "rmse": rmse,
        }
        line_rows.append(row)
        by_profile_id[profile_id] = row
    experimental_rows = [
        copy.deepcopy(by_profile_id[profile_id])
        for profile_id in contract.profile_experimental_velocity_line_ids
    ]
    experimental_mean = math.fsum(row["rmse"] for row in experimental_rows) / len(
        experimental_rows
    )
    require_same_float(
        statistics.get("velocity_profile_experimental_subset_uinf_rmse"),
        experimental_mean,
        f"{case_id} retained experimental velocity profile mean",
    )
    cut_rows = []
    for cut_id, rmse in zip(contract.profile_cp_cut_ids, cp_rmse, strict=True):
        segment_count, arc_length = _segments_arc_length(
            pressure_series[cut_id], label=f"{case_id}/{cut_id}"
        )
        cut_rows.append(
            {
                "cut_id": cut_id,
                "segment_count": segment_count,
                "arc_length_m": arc_length,
                "rmse": rmse,
            }
        )
    prediction_inputs = native_core.get("prediction_inputs")
    if not isinstance(prediction_inputs, dict):
        raise NativeReplayError(f"{case_id} retained core has no prediction inputs")
    compact_inputs: dict[str, object] = {}
    for support_id in ("surface_native_cells", "volume_native_cells"):
        raw = prediction_inputs.get(support_id)
        if not isinstance(raw, dict):
            raise NativeReplayError(f"{case_id} has no {support_id} prediction inputs")
        compact_inputs[support_id] = {
            "manifest_sha256": raw.get("manifest_sha256"),
            "chunk_sha256": raw.get("chunk_sha256"),
        }
    false_claims = {
        "scoring_contract_active": False,
        "official_submission": False,
        "owner_scientific_approval": False,
        "profile_resolution_convergence": False,
        "three_real_model_ordering": False,
        "independent_participant_dry_run": False,
        "all_case_chunk_partition_invariance": False,
    }
    return {
        "schema": RETAINED_INFERENCE_DIAGNOSTIC_CASE_SCHEMA,
        "schema_version": 1,
        "status": RETAINED_INFERENCE_DIAGNOSTIC_CASE_STATUS,
        "case_id": case_id,
        "official_submission": False,
        "prediction_scope": "surface_and_volume",
        "prediction_inputs": compact_inputs,
        "profile_support": {
            "native_source_pin_sha256": contract.native_source_pin_sha256,
            "profile_definition_sha256": contract.profile_sha256,
            "constant_series_support_index_sha256": CONSTANT_SERIES_SUPPORT_INDEX_SHA256,
            "relative_series_support_index_sha256": RELATIVE_SERIES_SUPPORT_INDEX_SHA256,
            "profile_series_per_case": RELATIVE_PROFILE_SERIES_PER_CASE,
            "coordinates_rebound_without_resampling": True,
        },
        "metrics": {
            "cp_cut_rmse": {
                "metric_id": "cp_cut_rmse",
                "ranked_value_available": True,
                "required_cut_count": len(contract.profile_cp_cut_ids),
                "unavailable_reasons": [],
                "case_equal_cut_mean_rmse": math.fsum(cp_rmse) / len(cp_rmse),
                "aggregation": "equal_case_equal_cut_macro_average",
                "weighting": "native_cut_intersection_segment_length",
                "support_status": "retained_native_support_rebind",
                "discrete_cp_probe_fallback_used": False,
                "cut_rmse": cut_rows,
            },
            "velocity_profile_uinf_rmse": {
                "metric_id": "velocity_profile_uinf_rmse",
                "ranked_value_available": True,
                "required_line_count": len(contract.profile_velocity_line_ids),
                "required_sample_count": contract.profile_velocity_sample_count,
                "unavailable_reasons": [],
                "line_rmse": line_rows,
                "case_equal_line_mean_rmse": math.fsum(velocity_rmse)
                / len(velocity_rmse),
                "quantity": "magnitude(UMeanTrim)/Uinf",
                "Uinf_m_per_s": contract.force_constants[
                    "freestream_velocity_m_per_s"
                ],
                "arc_rule": "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging",
                "aggregation": "equal_case_equal_line_macro_average",
                "weighting": "trapezoidal_arc_length_within_line",
            },
            "velocity_profile_experimental_subset_uinf_rmse": {
                "metric_id": "velocity_profile_experimental_subset_uinf_rmse",
                "value_available": True,
                "required_line_count": len(contract.profile_experimental_velocity_line_ids),
                "required_profile_ids": list(contract.profile_experimental_velocity_line_ids),
                "unavailable_reasons": [],
                "line_rmse": experimental_rows,
                "case_equal_experimental_line_mean_rmse": experimental_mean,
                "quantity": "magnitude(UMeanTrim)/Uinf",
                "Uinf_m_per_s": contract.force_constants[
                    "freestream_velocity_m_per_s"
                ],
                "arc_rule": "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging",
                "aggregation": "equal_case_equal_experimental_line_macro_average",
                "weighting": "trapezoidal_arc_length_within_line",
            },
        },
        "claims": false_claims,
    }


def _check_source_result(
    result: Mapping[str, Any], *, contract: Any, case_ids: tuple[str, ...]
) -> None:
    if result.get("schema") != SOURCE_RESULT_SCHEMA or result.get("schema_version") != 1:
        raise NativeReplayError("retained result schema/version is not recognized")
    split = result.get("split")
    if not isinstance(split, dict):
        raise NativeReplayError("retained result has no split metadata")
    if (
        split.get("split_id") != contract.split_id
        or split.get("case_set_id") != contract.case_set_id
        or tuple(split.get("test_case_ids", ())) != case_ids
        or case_ids != contract.case_ids
    ):
        raise NativeReplayError("retained result split does not equal the FluidsBench split")
    inputs = result.get("inputs")
    if not isinstance(inputs, dict):
        raise NativeReplayError("retained result has no input identities")
    if inputs.get("dataset_revision") != contract.native_source_pin.repository_revision:
        raise NativeReplayError("retained result dataset revision differs from FluidsBench")


def _macro_retained_core_metrics(
    core_metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Macro-average every non-spatial field metric retained in the core docs."""

    if not core_metric_rows:
        raise NativeReplayError("no retained core metric rows were supplied")
    metric_ids = set(core_metric_rows[0])
    if not metric_ids:
        raise NativeReplayError("retained core metrics are empty")
    if any(set(row) != metric_ids for row in core_metric_rows):
        raise NativeReplayError("retained core metric namespaces differ between cases")
    return {
        metric_id: math.fsum(
            require_finite(row[metric_id], f"retained core {metric_id}")
            for row in core_metric_rows
        )
        / len(core_metric_rows)
        for metric_id in sorted(metric_ids)
    }


def _published_metric_ids(specification_path: Path) -> tuple[str, ...]:
    specification = load_json(specification_path, label="FluidsBench submission specification")
    records = specification.get("metrics")
    if not isinstance(records, list) or not records:
        raise NativeReplayError("FluidsBench submission specification has no metrics")
    metric_ids = tuple(
        require_string(
            record.get("id") if isinstance(record, dict) else None,
            "FluidsBench metric ID",
        )
        for record in records
    )
    if len(metric_ids) != len(set(metric_ids)):
        raise NativeReplayError("FluidsBench submission specification repeats a metric ID")
    return metric_ids


def _scores_from_native_specification(
    metrics: Mapping[str, float], specification_path: Path
) -> dict[str, float]:
    specification = load_json(specification_path, label="FluidsBench submission specification")
    composite = specification.get("overall_score_composite")
    component_groups = specification.get("component_score_groups")
    if not isinstance(composite, dict) or not isinstance(component_groups, dict):
        raise NativeReplayError("FluidsBench submission specification has no score contract")
    try:
        scores = composite_component_group_scores(metrics, composite, component_groups)
        scores["overall_score"] = composite_overall_score(metrics, composite)
    except (KeyError, TypeError, ValueError) as error:
        raise NativeReplayError(f"cannot evaluate native FluidsBench scores: {error}") from error
    return scores


def _verify_reduction_matches_source(
    *,
    evaluation: Mapping[str, Any],
    source_result: Mapping[str, Any],
    retained_core_metrics: Mapping[str, float],
    submission_specification: Path,
) -> dict[str, float]:
    source_values = source_result.get("metric_values")
    reduced_values = evaluation.get("metric_values")
    if not isinstance(source_values, dict) or not isinstance(reduced_values, dict):
        raise NativeReplayError("retained or reduced metric values are absent")
    reduced_numeric: dict[str, float] = {}
    for metric_id, value in reduced_values.items():
        if value is None:
            raise NativeReplayError(f"native replay left {metric_id} unavailable")
        reduced_numeric[metric_id] = require_finite(value, f"native reduction {metric_id}")
        # The current native evaluator adds equal-entity diagnostics that were
        # not printed by the retained evaluator version.  Every shared metric
        # is still required to agree, including all ranked field, force, and
        # profile reductions.
        if metric_id in source_values:
            require_same_float(source_values[metric_id], reduced_numeric[metric_id], metric_id)
    numeric_source = {
        key: require_finite(value, f"retained result metric {key}")
        for key, value in source_values.items()
    }
    required_overlap = {
        "surface_pressure_rel_l2",
        "surface_wall_shear_rel_l2",
        "volume_velocity_rel_l2",
        "volume_pressure_rel_l2",
        "field_integrated_cd_rmse",
        "field_integrated_cl_rmse",
        "field_integrated_cmpitch_rmse",
        "field_integrated_clf_rmse",
        "field_integrated_clr_rmse",
        "field_integrated_lift_closure_max_abs",
        "velocity_profile_uinf_rmse",
        "velocity_profile_experimental_subset_uinf_rmse",
        "cp_cut_rmse",
    }
    missing_source = sorted(required_overlap - set(source_values))
    missing_reduction = sorted(required_overlap - set(reduced_values))
    if missing_source or missing_reduction:
        raise NativeReplayError(
            "native replay metric coverage is incomplete: "
            + ", ".join(
                [
                    *(f"retained:{metric_id}" for metric_id in missing_source),
                    *(f"reduced:{metric_id}" for metric_id in missing_reduction),
                ]
            )
        )
    native_metrics = dict(numeric_source)
    native_metrics.update(retained_core_metrics)
    native_metrics.update(reduced_numeric)
    scores = _scores_from_native_specification(native_metrics, submission_specification)
    for metric_id in ("overall_score", "field_score", "force_score", "diagnostic_score"):
        require_same_float(
            numeric_source[metric_id], float(scores[metric_id]), metric_id
        )
        # Preserve the retained evaluator's canonical printed scalar after the
        # native specification independently verifies it.  This avoids a
        # harmless binary64 reduction-order change becoming a visible rewrite.
        native_metrics[metric_id] = numeric_source[metric_id]
    return native_metrics


def _current_revision() -> str:
    try:
        return (
            subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            .stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise NativeReplayError(f"cannot resolve the FluidsBench evaluator revision: {error}") from error


def _build_public_package(
    *,
    staging: Path,
    submission_id: str,
    generated_at: str,
    source_metrics: Mapping[str, float],
    contract: Any,
    regional_diagnostics: Path | None,
) -> None:
    profile_index = staging / "profiles" / "index.json"
    profile_sha256 = sha256_file(profile_index)
    revision = _current_revision()
    evaluation_evidence: dict[str, object] = {
        "schema_version": NATIVE_PACKAGE_SCHEMA_VERSION,
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": contract.split_id,
        "reference_version": "drivaerml-evaluator-v3-candidate",
        "code_revision": revision,
        "generated_at": generated_at,
        "status": "pre_release_reference",
        "metric_values": dict(source_metrics),
        "profile_index_sha256": profile_sha256,
        "notes": (
            "Complete Transolver field evaluation expressed with the native "
            "FluidsBench DrivAerML evaluator and support identities. This is a "
            "closed pre-release reference, not an official result or ranking claim."
        ),
        "command": (
            "FluidsBench DrivAerML retained-inference replay with the native "
            "full-split evaluator and profile support."
        ),
    }
    regional_sha256: str | None = None
    if regional_diagnostics is not None:
        destination = staging / "regional-diagnostics.json"
        shutil.copyfile(regional_diagnostics, destination)
        regional_sha256 = sha256_file(destination)
        evaluation_evidence["regional_diagnostics_sha256"] = regional_sha256
    evidence_sha256 = write_json(staging / "evaluation-evidence.json", evaluation_evidence)
    split_name = "Full" if contract.split_id == "full" else contract.split_id
    submission: dict[str, object] = {
        "$schema": "https://fluidsbench.org/schemas/v1/submission.schema.json",
        "schema_version": NATIVE_PACKAGE_SCHEMA_VERSION,
        "submission_id": submission_id,
        "model": "Transolver",
        "model_type": "Transformer",
        "model_types": ["Transformer"],
        "training_regime": "other",
        "target_data_used": "other",
        "external_pretraining": False,
        "pretraining_data": [],
        "dataset": "DrivAerML",
        "dataset_id": "drivaerml",
        "dataset_version": "drivaerml-native-v3-candidate",
        "split": split_name,
        "split_id": contract.split_id,
        "case_set_id": contract.case_set_id,
        "split_sha256": contract.split_sha256,
        "parameter_count_millions": None,
        "submitter_name": "FluidsBench maintainers",
        "institution": "Pre-release validation reference",
        "paper_url": "",
        "code_url": "",
        "submitted_at": generated_at[:10],
        "evaluation": {
            "reference_version": "drivaerml-evaluator-v3-candidate",
            "code_revision": revision,
            "command": evaluation_evidence["command"],
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": evidence_sha256,
        },
        "approval": {
            "status": "prototype",
            "note": (
                "Real complete-field Transolver inference retained as a closed "
                "FluidsBench pre-release reference; it is not an official "
                "benchmark submission or ranking claim."
            ),
        },
        "metric_values": dict(source_metrics),
        "profile_data": {
            "format": RELATIVE_PROFILE_FORMAT,
            "index_file": "profiles/index.json",
            "case_count": len(contract.case_ids),
            "case_set_id": contract.case_set_id,
        },
        "note": (
            "Native FluidsBench representation of the complete full-split "
            "Transolver inference, including all current constant and relative "
            "profile series. It is pending the closed DrivAerML candidate-release gates."
        ),
        "training_regime_explanation": (
            "Training configuration is not represented in this retained evaluation package."
        ),
        "record_type": "pre_release_reference",
    }
    if regional_sha256 is not None:
        submission["regional_diagnostics"] = {
            "format": "drivaerml-regional-aggregate-v2",
            "file": "regional-diagnostics.json",
            "sha256": regional_sha256,
            "contract_sha256": "2bfd372817989112642056e4c76cfb418dbdcee445c57ee20ca37ee9ca158583",
            "case_count": len(contract.case_ids),
            "role": "report_only",
            "weight": 0.0,
            "official_score_changed": False,
        }
    write_json(staging / "submission.json", submission)


def build_replay(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.retained_evaluation_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise NativeReplayError("--retained-evaluation-dir must be a directory")
    source_result = load_json(source_root / "result.json", label="retained result")
    contract, _truth = _load_contract(
        submission_specification=args.submission_specification,
        split_id=args.split_id,
        force_truth_csv=args.force_truth,
        immutable_pins=OFFICIAL_IMMUTABLE_PINS,
    )
    case_ids = _source_case_ids(source_result)
    _check_source_result(source_result, contract=contract, case_ids=case_ids)
    source_cases = source_root / ".work" / "cases"
    if not source_cases.is_dir():
        raise NativeReplayError("retained evaluation has no .work/cases directory")
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise NativeReplayError(f"refusing to overwrite existing output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    constant_support = _constant_series_support_index()
    relative_support = _relative_series_support_index()
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        core_directory = staging_parent / "evaluator-evidence" / "core"
        diagnostic_directory = staging_parent / "evaluator-evidence" / "diagnostics"
        namespaced_by_case: dict[str, Sequence[Mapping[str, object]]] = {}
        retained_core_metric_rows: list[Mapping[str, Any]] = []
        for case_id in case_ids:
            source_case = load_json(source_cases / f"{case_id}.json", label="retained case")
            if (
                source_case.get("schema") != SOURCE_CASE_SCHEMA
                or source_case.get("schema_version") != SOURCE_CASE_SCHEMA_VERSION
                or source_case.get("status") != "complete"
                or source_case.get("case_id") != case_id
            ):
                raise NativeReplayError(f"retained case {case_id} schema/status mismatch")
            source_core = source_case.get("core")
            source_profiles = source_case.get("profiles")
            if not isinstance(source_core, dict) or not isinstance(source_profiles, dict):
                raise NativeReplayError(f"retained case {case_id} has incomplete core/profile evidence")
            if (
                source_profiles.get("schema") != SOURCE_PROFILE_SCHEMA
                or source_profiles.get("schema_version") != SOURCE_PROFILE_SCHEMA_VERSION
                or source_profiles.get("case_id") != case_id
            ):
                raise NativeReplayError(f"retained profile evidence {case_id} schema mismatch")
            native_core = convert_core_document(
                source_core, native_source_pin_sha256=contract.native_source_pin_sha256
            )
            native_core_metrics = native_core.get("metric_values")
            if not isinstance(native_core_metrics, Mapping):
                raise NativeReplayError(f"retained core {case_id} has no metric values")
            retained_core_metric_rows.append(native_core_metrics)
            namespaced_by_case[case_id] = convert_namespaced_profiles(
                case_id=case_id,
                profiles=source_profiles,
                constant_support=constant_support,
                relative_support=relative_support,
            )
            native_diagnostic = build_retained_diagnostic_document(
                case_id=case_id,
                source_profiles=source_profiles,
                native_core=native_core,
                contract=contract,
            )
            write_json(core_directory / f"{case_id}.json", native_core)
            write_json(diagnostic_directory / f"{case_id}.json", native_diagnostic)

        evaluation = evaluate_candidate_dataset(
            submission_specification=args.submission_specification,
            split_id=args.split_id,
            force_truth_csv=args.force_truth,
            core_case_evidence=tuple(core_directory / f"{case_id}.json" for case_id in case_ids),
            diagnostic_case_evidence=tuple(
                diagnostic_directory / f"{case_id}.json" for case_id in case_ids
            ),
        )
        evaluator_evidence = evaluation.to_json()
        native_metrics = _verify_reduction_matches_source(
            evaluation=evaluator_evidence,
            source_result=source_result,
            retained_core_metrics=_macro_retained_core_metrics(retained_core_metric_rows),
            submission_specification=args.submission_specification,
        )
        published_metric_ids = _published_metric_ids(args.submission_specification)
        missing_published_metrics = [
            metric_id for metric_id in published_metric_ids if metric_id not in native_metrics
        ]
        if missing_published_metrics:
            raise NativeReplayError(
                "native replay cannot populate declared FluidsBench metrics: "
                + ", ".join(missing_published_metrics)
            )
        source_metrics = {
            metric_id: native_metrics[metric_id] for metric_id in published_metric_ids
        }
        write_candidate_dataset_evidence(
            evaluation, staging_parent / "evaluator-evidence" / "candidate-dataset-evaluation.json"
        )
        if evaluation.regional_diagnostics is not None:
            write_regional_diagnostics(
                evaluation,
                staging_parent / "evaluator-evidence" / "regional-diagnostics.json",
            )
        profile_package = schema_v3_relative_profile_chunks_candidate_adapter(
            evaluation,
            submission_id=args.submission_id,
            namespaced_series_by_case=namespaced_by_case,
            contract_sha256=RELATIVE_PROFILE_CONTRACT_SHA256,
            cases_per_chunk=args.profile_cases_per_chunk,
        )
        write_schema_v3_profile_chunks_candidate(profile_package, staging_parent / "profiles")
        regional_input = (
            args.regional_diagnostics.expanduser().resolve()
            if args.regional_diagnostics is not None
            else None
        )
        if regional_input is not None and not regional_input.is_file():
            raise NativeReplayError("--regional-diagnostics must be a regular file")
        _build_public_package(
            staging=staging_parent,
            submission_id=args.submission_id,
            generated_at=args.generated_at,
            source_metrics=source_metrics,
            contract=contract,
            regional_diagnostics=regional_input,
        )
        os.replace(staging_parent, output)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "submission_id": args.submission_id,
        "case_count": len(case_ids),
        "profile_series_per_case": RELATIVE_PROFILE_SERIES_PER_CASE,
        "profile_contract_id": RELATIVE_PROFILE_CONTRACT_ID,
        "profile_contract_sha256": RELATIVE_PROFILE_CONTRACT_SHA256,
        "overall_score": source_metrics["overall_score"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-evaluation-dir", type=Path, required=True)
    parser.add_argument("--force-truth", type=Path, required=True)
    parser.add_argument(
        "--submission-specification",
        type=Path,
        default=ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json",
    )
    parser.add_argument("--split-id", default="full")
    parser.add_argument("--submission-id", default="drivaerml-transolver")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regional-diagnostics", type=Path)
    parser.add_argument("--generated-at", default=DEFAULT_GENERATED_AT)
    parser.add_argument("--profile-cases-per-chunk", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profile_cases_per_chunk < 1:
        raise NativeReplayError("--profile-cases-per-chunk must be positive")
    try:
        result = build_replay(args)
    except NativeReplayError as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
