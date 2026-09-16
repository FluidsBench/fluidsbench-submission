#!/usr/bin/env python3
"""Aggregate candidate DrivAerML AutoCFD5 Cp case-support receipts.

Complete mode is deliberately strict: exactly one JSON/CSV pair is required
for each of the 484 immutable native-source-pin cases.  ``--pilot-case`` is
the only partial mode and its outputs are permanently labelled incomplete and
non-public.  This tool validates candidate evidence; it does not assert owner
visual sign-off or activate scoring support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    CP_PROBE_COUNT,
    U_INF_M_PER_S,
    AutoCFD5Error,
    CpProbeMappingEvidence,
    load_autocfd5_definition,
    validate_cp_mapping_evidence,
)
from reference.drivaerml.source import (  # noqa: E402
    NativeCaseRecord,
    NativeSourceError,
    load_native_source_pin,
)
from scripts.build_drivaerml_cp_case_support import (  # noqa: E402
    CSV_SCHEMA,
    EVIDENCE_SCHEMA,
    EXPECTED_AUTOCFD5_PROFILE_SHA256,
    CpCaseSupportError,
    _algorithm_payload,
    _csv_bytes,
)


DEFAULT_NATIVE_SOURCE_PIN = (
    ROOT / "benchmark-specs" / "drivaerml" / "proposal" / "native-source-pin.json"
)
DEFAULT_AUTOCFD5_PROFILE = (
    ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
)
OFFICIAL_NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
OFFICIAL_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
OFFICIAL_UNAVAILABLE_RUNS = frozenset(
    {167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473}
)
OFFICIAL_CASE_IDS = tuple(
    f"run_{run_number}"
    for run_number in range(1, 501)
    if run_number not in OFFICIAL_UNAVAILABLE_RUNS
)
OFFICIAL_REGISTRY_SHA256 = {
    "cp_nominal_registry": (
        "2c1e216ef5693b26d43b9b4f55586ca35516224af0ad874987e5902c67f08e6f"
    ),
    "cp_panel_membership": (
        "6e288077b8b87e0c5d8d77c1050b07e4b5dd155a8e4fad8815ce4b0581995538"
    ),
    "cp_component_registry": (
        "b4ef9270f2633f2bd3d271e18de23f93e8da3398d67fbc34e77271fb37cbce66"
    ),
    "velocity_lines": (
        "6eb1528034e27a75ab1949551d58c8a161331bf6d343ca7fb4324f0800ca4d12"
    ),
    "velocity_scoring_grid": (
        "5f1bcf84a633aa6bfdd776764c3295d5d624ef0b6c3649f67de446281ae5ba97"
    ),
}
PINNED_DEPENDENCIES = {"numpy": "2.2.6", "vtk": "9.5.2"}

AGGREGATE_SCHEMA = "drivaerml-autocfd5-cp-support-aggregate-candidate-v1"
STL_INVENTORY_SCHEMA = "drivaerml-autocfd5-cp-stl-source-inventory-candidate-v1"
RECEIPT_STATUS = "candidate_not_owner_approved_not_active_scoring_support"

_CASE_RE = re.compile(r"run_([1-9][0-9]*)")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")
_MAPPING_INVALID_REASONS = frozenset(
    {
        "declared_component_absent",
        "declared_cut_has_no_finite_nondegenerate_intersection",
        "declared_component_has_no_finite_nondegenerate_triangle",
        "nominal_displacement_exceeds_10pct_wheelbase",
        "no_native_polygon_bounds_candidate_within_2mm",
        "no_finite_nondegenerate_native_polygon",
        "native_bridge_distance_exceeds_2mm",
        "native_bridge_normal_agreement_below_cos30",
    }
)
_REVIEW_FLAGS = frozenset(
    {
        "nominal_displacement_gt_7p5pct_wheelbase",
        "native_bridge_distance_gt_0p5mm",
    }
)
_ALLOWED_PRESSURE_VTK_TYPES = frozenset({"float", "double"})


class CpSupportAggregateError(ValueError):
    """Raised when Cp receipt coverage or content is not exact."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise CpSupportAggregateError(f"cannot hash input file: {path}") from error
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CpSupportAggregateError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CpSupportAggregateError(
                    f"{label} contains forbidden non-finite token {value}"
                )
            ),
        )
    except CpSupportAggregateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CpSupportAggregateError(f"cannot read valid {label}: {path}") from error
    if not isinstance(value, dict):
        raise CpSupportAggregateError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CpSupportAggregateError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result))
        unexpected = sorted(set(result) - expected_set)
        raise CpSupportAggregateError(
            f"{label} keys differ from schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return result


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise CpSupportAggregateError(f"{label} must be {qualifier}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CpSupportAggregateError(f"{label} must be Boolean")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CpSupportAggregateError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CpSupportAggregateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CpSupportAggregateError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise CpSupportAggregateError(f"{label} must be non-negative")
    return result


def _optional_finite(
    value: object, label: str, *, nonnegative: bool = False
) -> float | None:
    if value is None:
        return None
    return _finite(value, label, nonnegative=nonnegative)


def _point(value: object, label: str, *, optional: bool = False) -> tuple[float, ...] | None:
    if optional and value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise CpSupportAggregateError(f"{label} must contain exactly three values")
    return tuple(_finite(item, label) for item in value)


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise CpSupportAggregateError(f"{label} must be a lowercase SHA-256")
    return result


def _case_number(case_id: str) -> int:
    match = _CASE_RE.fullmatch(case_id)
    if match is None:
        raise CpSupportAggregateError(f"invalid canonical case ID {case_id!r}")
    return int(match.group(1))


def _ordered_cases(case_ids: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(_string(value, label) for value in case_ids)
    if not result:
        raise CpSupportAggregateError(f"{label} cannot be empty")
    for case_id in result:
        _case_number(case_id)
    if len(result) != len(set(result)):
        raise CpSupportAggregateError(f"{label} must be unique")
    if tuple(sorted(result, key=_case_number)) != result:
        raise CpSupportAggregateError(f"{label} must use increasing run-number order")
    return result


def _counter(value: object, label: str) -> dict[str, int]:
    result = _mapping(value, label)
    parsed: dict[str, int] = {}
    for key, count in result.items():
        name = _string(key, f"{label} key")
        parsed[name] = _integer(count, f"{label}.{name}", minimum=1)
    return parsed


def _range(value: object, label: str) -> dict[str, float] | None:
    if value is None:
        return None
    result = _exact_keys(value, {"minimum", "maximum"}, label)
    minimum = _finite(result["minimum"], f"{label}.minimum")
    maximum = _finite(result["maximum"], f"{label}.maximum")
    if minimum > maximum:
        raise CpSupportAggregateError(f"{label} minimum exceeds maximum")
    return {"minimum": minimum, "maximum": maximum}


def _expected_range(values: Sequence[float]) -> dict[str, float] | None:
    return {"minimum": min(values), "maximum": max(values)} if values else None


def _assert_no_absolute_paths(value: object, label: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
            raise CpSupportAggregateError(f"{label} contains an absolute path")


def _load_definition(profile_path: Path) -> tuple[Any, str, dict[str, str]]:
    profile_sha256 = sha256_file(profile_path)
    if profile_sha256 != EXPECTED_AUTOCFD5_PROFILE_SHA256:
        raise CpSupportAggregateError(
            "AutoCFD5 profile does not match the exact candidate v8 SHA-256"
        )
    try:
        definition = load_autocfd5_definition(profile_path)
    except AutoCFD5Error as error:
        raise CpSupportAggregateError("AutoCFD5 v8 registries are invalid") from error
    if sha256_file(profile_path) != profile_sha256:
        raise CpSupportAggregateError(
            "AutoCFD5 profile changed while its registries were validated"
        )
    registry_sha256 = dict(definition.source_sha256)
    if registry_sha256 != OFFICIAL_REGISTRY_SHA256:
        raise CpSupportAggregateError("AutoCFD5 registry SHA-256 identities are not exact")
    if (
        len(definition.cp_probes) != CP_PROBE_COUNT
        or len(definition.cp_component_rules) != CP_PROBE_COUNT
    ):
        raise CpSupportAggregateError("AutoCFD5 v8 must define exactly 209 probes and rules")
    return definition, profile_sha256, registry_sha256


def _load_pin(
    path: Path,
    *,
    expected_cases: tuple[str, ...],
    expected_pin_sha256: str | None,
) -> tuple[Any, str]:
    digest = sha256_file(path)
    if expected_pin_sha256 is not None and digest != expected_pin_sha256:
        raise CpSupportAggregateError(
            "native-source pin SHA-256 mismatch: "
            f"expected {expected_pin_sha256}, got {digest}"
        )
    try:
        pin = load_native_source_pin(path)
    except NativeSourceError as error:
        raise CpSupportAggregateError("native-source pin is invalid") from error
    if sha256_file(path) != digest:
        raise CpSupportAggregateError("native-source pin changed during validation")
    if (
        pin.repository_id != "neashton/drivaerml"
        or pin.repository_revision != OFFICIAL_REVISION
    ):
        raise CpSupportAggregateError("native-source repository identity is not frozen")
    actual_cases = tuple(case.case_id for case in pin.cases)
    if actual_cases != expected_cases:
        raise CpSupportAggregateError(
            "native-source pin case order/set differs from the requested official scope"
        )
    return pin, digest


_ROW_KEYS = {
    "case_id",
    "autocfd_probe_id",
    "nominal_point_m",
    "drivaerml_component",
    "projection_mode",
    "cut_axis",
    "cut_value_m",
    "owner_review_status",
    "mapping_valid",
    "mapping_reason",
    "mapped_stl_point_m",
    "raw_stl_triangle_id",
    "stl_unit_normal",
    "nominal_displacement_m",
    "native_closest_point_m",
    "raw_vtk_polygon_id",
    "native_polygon_unit_normal",
    "bridge_distance_m",
    "bridge_abs_normal_dot",
    "component_facet_count",
    "projection_candidate_count",
    "native_bounds_candidate_count",
    "native_distance_pass_count",
    "native_normal_pass_count",
    "review_flags",
    "truth_valid",
    "truth_reason",
    "pMeanTrim_m2_per_s2",
    "truth_Cp",
    "support_valid",
}


def _same_optional_float(actual: object, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    return math.isfinite(float(actual)) and float(actual) == float(expected)


def _validate_rows(
    value: object,
    *,
    case_id: str,
    definition: Any,
    stl_sha256: str,
    boundary_sha256: str,
    stl_facet_count: int,
    solid_facet_counts: Mapping[str, int],
    boundary_polygon_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(value, list) or len(value) != CP_PROBE_COUNT:
        raise CpSupportAggregateError(f"{case_id} must retain exactly 209 Cp rows")
    expected_ids = tuple(probe.autocfd_probe_id for probe in definition.cp_probes)
    rule_by_id = {
        rule.autocfd_probe_id: rule for rule in definition.cp_component_rules
    }
    pressure_values: list[float] = []
    cp_values: list[float] = []
    evidence_rows: list[CpProbeMappingEvidence] = []
    mapping_reasons: Counter[str] = Counter()
    truth_reasons: Counter[str] = Counter()
    review_flags: Counter[str] = Counter()
    owner_statuses: Counter[str] = Counter()
    mapping_valid_count = 0
    truth_valid_count = 0
    support_valid_count = 0
    parsed_rows: list[dict[str, Any]] = []

    for position, (raw_row, expected_probe_id) in enumerate(
        zip(value, expected_ids, strict=True)
    ):
        label = f"{case_id} row {position}"
        row = dict(_exact_keys(raw_row, _ROW_KEYS, label))
        if row["case_id"] != case_id:
            raise CpSupportAggregateError(f"{label} has the wrong case_id")
        probe_id = _integer(row["autocfd_probe_id"], f"{label}.autocfd_probe_id", minimum=1)
        if probe_id != expected_probe_id:
            raise CpSupportAggregateError(
                f"{case_id} rows must use exact ordered unique v8 probe IDs"
            )
        probe = definition.cp_probes[position]
        rule = rule_by_id[probe_id]
        nominal = _point(row["nominal_point_m"], f"{label}.nominal_point_m")
        if nominal != tuple(float(item) for item in probe.point_m):
            raise CpSupportAggregateError(f"{label} differs from nominal v8 coordinates")
        if (
            row["drivaerml_component"] != rule.drivaerml_component
            or row["projection_mode"] != rule.projection_mode
            or row["cut_axis"] != rule.cut_axis
            or not _same_optional_float(row["cut_value_m"], rule.cut_value_m)
            or row["owner_review_status"] != rule.owner_review_status
        ):
            raise CpSupportAggregateError(f"{label} differs from its v8 mapping rule")
        owner_statuses[rule.owner_review_status] += 1

        mapping_valid = _boolean(row["mapping_valid"], f"{label}.mapping_valid")
        mapping_reason = _string(
            row["mapping_reason"], f"{label}.mapping_reason", allow_empty=True
        )
        if mapping_valid:
            if mapping_reason:
                raise CpSupportAggregateError(f"{label} valid mapping has a reason")
            mapping_valid_count += 1
        else:
            if mapping_reason not in _MAPPING_INVALID_REASONS:
                raise CpSupportAggregateError(f"{label} has an unknown invalid reason")
            mapping_reasons[mapping_reason] += 1

        mapped_point = _point(
            row["mapped_stl_point_m"], f"{label}.mapped_stl_point_m", optional=True
        )
        stl_id = _optional_integer(row["raw_stl_triangle_id"], f"{label}.raw_stl_triangle_id")
        stl_normal = _point(
            row["stl_unit_normal"], f"{label}.stl_unit_normal", optional=True
        )
        displacement = _optional_finite(
            row["nominal_displacement_m"],
            f"{label}.nominal_displacement_m",
            nonnegative=True,
        )
        native_point = _point(
            row["native_closest_point_m"],
            f"{label}.native_closest_point_m",
            optional=True,
        )
        polygon_id = _optional_integer(
            row["raw_vtk_polygon_id"], f"{label}.raw_vtk_polygon_id"
        )
        native_normal = _point(
            row["native_polygon_unit_normal"],
            f"{label}.native_polygon_unit_normal",
            optional=True,
        )
        bridge_distance = _optional_finite(
            row["bridge_distance_m"], f"{label}.bridge_distance_m", nonnegative=True
        )
        normal_dot = _optional_finite(
            row["bridge_abs_normal_dot"],
            f"{label}.bridge_abs_normal_dot",
            nonnegative=True,
        )
        if stl_id is not None and stl_id >= stl_facet_count:
            raise CpSupportAggregateError(f"{label} raw STL triangle ID is out of range")
        if polygon_id is not None and polygon_id >= boundary_polygon_count:
            raise CpSupportAggregateError(f"{label} raw VTK polygon ID is out of range")
        if normal_dot is not None and normal_dot > 1.0:
            raise CpSupportAggregateError(f"{label} normal agreement exceeds one")
        for normal, normal_label in (
            (stl_normal, "STL"),
            (native_normal, "native polygon"),
        ):
            if normal is not None and not math.isclose(
                math.sqrt(sum(component * component for component in normal)),
                1.0,
                rel_tol=0.0,
                abs_tol=2.0e-12,
            ):
                raise CpSupportAggregateError(f"{label} {normal_label} normal is not unit")

        component_count = _integer(
            row["component_facet_count"], f"{label}.component_facet_count"
        )
        projection_count = _integer(
            row["projection_candidate_count"], f"{label}.projection_candidate_count"
        )
        bounds_count = _integer(
            row["native_bounds_candidate_count"],
            f"{label}.native_bounds_candidate_count",
        )
        distance_count = _integer(
            row["native_distance_pass_count"], f"{label}.native_distance_pass_count"
        )
        normal_count = _integer(
            row["native_normal_pass_count"], f"{label}.native_normal_pass_count"
        )
        if component_count != solid_facet_counts.get(rule.drivaerml_component, 0):
            raise CpSupportAggregateError(
                f"{label} component facet count differs from the named STL inventory"
            )
        if component_count > stl_facet_count or projection_count > component_count:
            raise CpSupportAggregateError(f"{label} STL candidate counts do not close")
        if not 0 <= normal_count <= distance_count <= bounds_count:
            raise CpSupportAggregateError(f"{label} native candidate counts do not close")

        flags_value = row["review_flags"]
        if (
            not isinstance(flags_value, list)
            or any(not isinstance(flag, str) or flag not in _REVIEW_FLAGS for flag in flags_value)
            or len(flags_value) != len(set(flags_value))
        ):
            raise CpSupportAggregateError(f"{label} review_flags are invalid")
        review_flags.update(flags_value)

        truth_valid = _boolean(row["truth_valid"], f"{label}.truth_valid")
        truth_reason = _string(
            row["truth_reason"], f"{label}.truth_reason", allow_empty=True
        )
        pressure = _optional_finite(
            row["pMeanTrim_m2_per_s2"], f"{label}.pMeanTrim_m2_per_s2"
        )
        truth_cp = _optional_finite(row["truth_Cp"], f"{label}.truth_Cp")
        support_valid = _boolean(row["support_valid"], f"{label}.support_valid")
        if support_valid != (mapping_valid and truth_valid):
            raise CpSupportAggregateError(f"{label} support_valid is inconsistent")
        if truth_valid:
            if not mapping_valid or truth_reason or pressure is None or truth_cp is None:
                raise CpSupportAggregateError(f"{label} valid truth fields are incomplete")
            expected_cp = 2.0 * pressure / (U_INF_M_PER_S * U_INF_M_PER_S)
            if not math.isclose(truth_cp, expected_cp, rel_tol=0.0, abs_tol=1.0e-15):
                raise CpSupportAggregateError(f"{label} truth Cp does not match pMeanTrim")
            pressure_values.append(pressure)
            cp_values.append(truth_cp)
            truth_valid_count += 1
        else:
            expected_reason = (
                "nonfinite_native_pMeanTrim" if mapping_valid else "mapping_invalid"
            )
            if truth_reason != expected_reason or pressure is not None or truth_cp is not None:
                raise CpSupportAggregateError(f"{label} invalid truth fields are inconsistent")
            truth_reasons[truth_reason] += 1
        support_valid_count += int(support_valid)

        try:
            evidence_rows.append(
                CpProbeMappingEvidence(
                    case_id=case_id,
                    autocfd_probe_id=probe_id,
                    valid=mapping_valid,
                    reason=mapping_reason,
                    mapped_point_m=mapped_point,
                    raw_stl_triangle_id=stl_id,
                    raw_vtk_polygon_id=polygon_id,
                    nominal_displacement_m=displacement,
                    bridge_distance_m=bridge_distance,
                    bridge_abs_normal_dot=normal_dot,
                    source_sha256=(stl_sha256, boundary_sha256),
                )
            )
        except AutoCFD5Error as error:
            raise CpSupportAggregateError(
                f"{label} mapping fields violate the evidence schema"
            ) from error
        parsed_rows.append(row)

    try:
        validate_cp_mapping_evidence(
            evidence_rows,
            definition,
            expected_case_ids=(case_id,),
            require_all_valid=False,
        )
    except AutoCFD5Error as error:
        raise CpSupportAggregateError(f"{case_id} mapping evidence fails v8 gates") from error

    summary = {
        "row_count": CP_PROBE_COUNT,
        "mapping_valid_count": mapping_valid_count,
        "mapping_invalid_count": CP_PROBE_COUNT - mapping_valid_count,
        "mapping_invalid_reason_counts": dict(sorted(mapping_reasons.items())),
        "truth_valid_count": truth_valid_count,
        "truth_invalid_count": CP_PROBE_COUNT - truth_valid_count,
        "truth_invalid_reason_counts": dict(sorted(truth_reasons.items())),
        "support_valid_count": support_valid_count,
        "owner_review_status_counts": dict(sorted(owner_statuses.items())),
        "review_flag_counts": dict(sorted(review_flags.items())),
        "pMeanTrim_range_m2_per_s2": _expected_range(pressure_values),
        "truth_Cp_range": _expected_range(cp_values),
    }
    return parsed_rows, summary


def _validate_case_receipt(
    *,
    json_path: Path,
    csv_path: Path,
    pinned_case: NativeCaseRecord,
    definition: Any,
    profile_sha256: str,
    registry_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = pinned_case.case_id
    json_sha256 = sha256_file(json_path)
    csv_sha256 = sha256_file(csv_path)
    try:
        json_size_bytes = json_path.stat().st_size
    except OSError as error:
        raise CpSupportAggregateError(f"cannot stat {case_id} receipt JSON") from error
    receipt = _read_json(json_path, f"{case_id} Cp receipt JSON")
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "case_id",
            "status",
            "owner_visual_signoff_claimed",
            "definition",
            "sources",
            "dependencies",
            "algorithm",
            "summary",
            "artifacts",
            "rows",
        },
        f"{case_id} receipt",
    )
    if receipt["schema"] != EVIDENCE_SCHEMA or receipt["schema_version"] != "1.0":
        raise CpSupportAggregateError(f"{case_id} receipt schema is not exact")
    if receipt["case_id"] != case_id or receipt["status"] != RECEIPT_STATUS:
        raise CpSupportAggregateError(f"{case_id} receipt identity/status is not exact")
    if _boolean(
        receipt["owner_visual_signoff_claimed"],
        f"{case_id}.owner_visual_signoff_claimed",
    ):
        raise CpSupportAggregateError(f"{case_id} receipt improperly claims owner signoff")

    definition_value = _exact_keys(
        receipt["definition"],
        {"profile_sha256", "registry_sha256", "probe_count", "rule_count"},
        f"{case_id} definition",
    )
    probe_count = _integer(
        definition_value["probe_count"], f"{case_id} definition probe_count", minimum=1
    )
    rule_count = _integer(
        definition_value["rule_count"], f"{case_id} definition rule_count", minimum=1
    )
    if (
        definition_value["profile_sha256"] != profile_sha256
        or dict(_mapping(definition_value["registry_sha256"], "registry SHA-256"))
        != dict(registry_sha256)
        or probe_count != CP_PROBE_COUNT
        or rule_count != CP_PROBE_COUNT
    ):
        raise CpSupportAggregateError(f"{case_id} receipt v8 definition is not exact")

    sources = _exact_keys(receipt["sources"], {"stl", "boundary"}, f"{case_id} sources")
    stl = _exact_keys(
        sources["stl"],
        {"file", "sha256", "size_bytes", "solid_count", "facet_count", "solid_facet_counts"},
        f"{case_id} STL source",
    )
    expected_stl_file = f"drivaer_{pinned_case.run_number}.stl"
    if stl["file"] != expected_stl_file:
        raise CpSupportAggregateError(f"{case_id} receipt has the wrong named STL")
    stl_sha256 = _sha256(stl["sha256"], f"{case_id} STL SHA-256")
    stl_size = _integer(stl["size_bytes"], f"{case_id} STL size", minimum=1)
    stl_facet_count = _integer(stl["facet_count"], f"{case_id} STL facet count", minimum=1)
    solid_counts_raw = _mapping(stl["solid_facet_counts"], f"{case_id} solid facet counts")
    solid_counts: dict[str, int] = {}
    for name, count in solid_counts_raw.items():
        solid_counts[_string(name, f"{case_id} STL solid name")] = _integer(
            count, f"{case_id} STL solid {name} facet count", minimum=1
        )
    if (
        _integer(stl["solid_count"], f"{case_id} STL solid count", minimum=1)
        != len(solid_counts)
        or sum(solid_counts.values()) != stl_facet_count
    ):
        raise CpSupportAggregateError(f"{case_id} STL inventory does not close")

    boundary = _exact_keys(
        sources["boundary"],
        {
            "file",
            "sha256",
            "size_bytes",
            "point_count",
            "polygon_count",
            "pMeanTrim_tuple_count",
            "pMeanTrim_component_count",
            "pMeanTrim_vtk_data_type",
        },
        f"{case_id} boundary source",
    )
    expected_boundary_file = f"boundary_{pinned_case.run_number}.vtp"
    if boundary["file"] != expected_boundary_file:
        raise CpSupportAggregateError(f"{case_id} receipt has the wrong named boundary")
    boundary_sha256 = _sha256(boundary["sha256"], f"{case_id} boundary SHA-256")
    boundary_size = _integer(boundary["size_bytes"], f"{case_id} boundary size", minimum=1)
    if (
        boundary_sha256 != pinned_case.boundary.sha256
        or boundary_size != pinned_case.boundary.size_bytes
    ):
        raise CpSupportAggregateError(f"{case_id} boundary identity differs from the pin")
    point_count = _integer(boundary["point_count"], f"{case_id} point count", minimum=1)
    polygon_count = _integer(
        boundary["polygon_count"], f"{case_id} polygon count", minimum=1
    )
    if polygon_count != pinned_case.surface_cell_area.element_count:
        raise CpSupportAggregateError(
            f"{case_id} polygon count differs from the pinned native area support"
        )
    tuple_count = _integer(
        boundary["pMeanTrim_tuple_count"], f"{case_id} pMeanTrim tuple count", minimum=1
    )
    component_count = _integer(
        boundary["pMeanTrim_component_count"],
        f"{case_id} pMeanTrim component count",
        minimum=1,
    )
    pressure_vtk_type = _string(
        boundary["pMeanTrim_vtk_data_type"], f"{case_id} pMeanTrim VTK data type"
    )
    if (
        tuple_count != polygon_count
        or component_count != 1
        or pressure_vtk_type not in _ALLOWED_PRESSURE_VTK_TYPES
    ):
        raise CpSupportAggregateError(f"{case_id} pMeanTrim native CellData metadata is not exact")

    dependencies = _exact_keys(
        receipt["dependencies"], PINNED_DEPENDENCIES, f"{case_id} dependencies"
    )
    if dict(dependencies) != PINNED_DEPENDENCIES:
        raise CpSupportAggregateError(f"{case_id} dependency versions are not pinned")
    algorithm = _mapping(receipt["algorithm"], f"{case_id} algorithm")
    stl_algorithm = _mapping(algorithm.get("stl"), f"{case_id} STL algorithm")
    chunk_facets = _integer(
        stl_algorithm.get("stream_chunk_facets"),
        f"{case_id} stream_chunk_facets",
        minimum=1,
    )
    if dict(algorithm) != _algorithm_payload(stl_chunk_facets=chunk_facets):
        raise CpSupportAggregateError(f"{case_id} algorithm constants are not exact")

    rows, expected_summary = _validate_rows(
        receipt["rows"],
        case_id=case_id,
        definition=definition,
        stl_sha256=stl_sha256,
        boundary_sha256=boundary_sha256,
        stl_facet_count=stl_facet_count,
        solid_facet_counts=solid_counts,
        boundary_polygon_count=polygon_count,
    )
    summary = _exact_keys(receipt["summary"], expected_summary, f"{case_id} summary")
    # Parse the range and counter subdocuments before exact comparison so bad
    # types fail with a focused message rather than Python equality quirks.
    for key in (
        "mapping_invalid_reason_counts",
        "truth_invalid_reason_counts",
        "owner_review_status_counts",
        "review_flag_counts",
    ):
        _counter(summary[key], f"{case_id} summary.{key}")
    for key in (
        "row_count",
        "mapping_valid_count",
        "mapping_invalid_count",
        "truth_valid_count",
        "truth_invalid_count",
        "support_valid_count",
    ):
        _integer(summary[key], f"{case_id} summary.{key}")
    _range(summary["pMeanTrim_range_m2_per_s2"], f"{case_id} pMeanTrim range")
    _range(summary["truth_Cp_range"], f"{case_id} Cp range")
    if dict(summary) != expected_summary:
        raise CpSupportAggregateError(f"{case_id} summary does not match its 209 rows")

    artifacts = _exact_keys(
        receipt["artifacts"],
        {"mapping_csv_schema", "mapping_csv_row_count", "mapping_csv_sha256"},
        f"{case_id} artifacts",
    )
    csv_row_count = _integer(
        artifacts["mapping_csv_row_count"], f"{case_id} CSV row count", minimum=1
    )
    if (
        artifacts["mapping_csv_schema"] != CSV_SCHEMA
        or csv_row_count != CP_PROBE_COUNT
        or _sha256(artifacts["mapping_csv_sha256"], f"{case_id} CSV SHA-256")
        != csv_sha256
    ):
        raise CpSupportAggregateError(f"{case_id} CSV artifact declaration is not exact")
    try:
        expected_csv = _csv_bytes(rows)
        actual_csv = csv_path.read_bytes()
    except (OSError, UnicodeError, TypeError, ValueError, KeyError, CpCaseSupportError) as error:
        raise CpSupportAggregateError(f"cannot validate {case_id} mapping CSV") from error
    if actual_csv != expected_csv:
        raise CpSupportAggregateError(
            f"{case_id} CSV schema/order/rows differ from its JSON receipt"
        )
    if hashlib.sha256(actual_csv).hexdigest() != csv_sha256:
        raise CpSupportAggregateError(f"{case_id} mapping CSV changed during validation")
    if sha256_file(json_path) != json_sha256:
        raise CpSupportAggregateError(f"{case_id} receipt JSON changed during validation")

    case_record = {
        "case_id": case_id,
        "receipt_json_sha256": json_sha256,
        "receipt_json_size_bytes": json_size_bytes,
        "mapping_csv_sha256": csv_sha256,
        "mapping_csv_size_bytes": len(actual_csv),
        "stl": {
            "file": expected_stl_file,
            "sha256": stl_sha256,
            "size_bytes": stl_size,
            "facet_count": stl_facet_count,
            "solid_count": len(solid_counts),
            "solid_facet_counts": dict(sorted(solid_counts.items())),
        },
        "boundary": {
            "file": pinned_case.boundary.path.as_posix(),
            "sha256": boundary_sha256,
            "size_bytes": boundary_size,
            "point_count": point_count,
            "polygon_count": polygon_count,
            "pMeanTrim_tuple_count": polygon_count,
            "pMeanTrim_component_count": 1,
            "pMeanTrim_vtk_data_type": pressure_vtk_type,
        },
        "validity": expected_summary,
    }
    inventory_record = {
        "case_id": case_id,
        "file": expected_stl_file,
        "sha256": stl_sha256,
        "size_bytes": stl_size,
        "facet_count": stl_facet_count,
        "solid_count": len(solid_counts),
        "solid_facet_counts": dict(sorted(solid_counts.items())),
        "source_receipt_json_sha256": json_sha256,
    }
    return case_record, inventory_record


def aggregate_cp_support_receipts(
    *,
    native_source_pin_path: Path,
    profile_path: Path,
    receipt_pairs: Sequence[tuple[Path, Path]],
    pilot_case_ids: Sequence[str] | None = None,
    official_case_ids: Sequence[str] = OFFICIAL_CASE_IDS,
    expected_pin_sha256: str | None = OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return validated aggregate evidence and a separate STL inventory."""

    official_cases = _ordered_cases(official_case_ids, "official case IDs")
    pin, pin_sha256 = _load_pin(
        Path(native_source_pin_path),
        expected_cases=official_cases,
        expected_pin_sha256=expected_pin_sha256,
    )
    definition, profile_sha256, registry_sha256 = _load_definition(Path(profile_path))
    pilot = pilot_case_ids is not None
    if pilot:
        selected_cases = _ordered_cases(pilot_case_ids or (), "pilot case IDs")
        if not set(selected_cases).issubset(official_cases):
            raise CpSupportAggregateError("pilot cases must be official pinned cases")
        if selected_cases == official_cases:
            raise CpSupportAggregateError(
                "pilot mode must be a strict subset; omit --pilot-case for complete mode"
            )
    else:
        selected_cases = official_cases

    pinned_by_case = {case.case_id: case for case in pin.cases}
    records_by_case: dict[str, dict[str, Any]] = {}
    inventories_by_case: dict[str, dict[str, Any]] = {}
    for pair in receipt_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise CpSupportAggregateError("each receipt input must be one JSON/CSV pair")
        json_path, csv_path = (Path(pair[0]), Path(pair[1]))
        if json_path.resolve() == csv_path.resolve():
            raise CpSupportAggregateError("receipt JSON and CSV paths must differ")
        receipt = _read_json(json_path, "Cp receipt JSON identity")
        case_id = _string(receipt.get("case_id"), "Cp receipt case_id")
        if case_id in records_by_case:
            raise CpSupportAggregateError(f"duplicate Cp receipt pair for {case_id}")
        if case_id not in pinned_by_case:
            raise CpSupportAggregateError(f"unexpected Cp receipt case {case_id}")
        record, inventory = _validate_case_receipt(
            json_path=json_path,
            csv_path=csv_path,
            pinned_case=pinned_by_case[case_id],
            definition=definition,
            profile_sha256=profile_sha256,
            registry_sha256=registry_sha256,
        )
        records_by_case[case_id] = record
        inventories_by_case[case_id] = inventory

    if set(records_by_case) != set(selected_cases):
        missing = sorted(set(selected_cases) - set(records_by_case), key=_case_number)
        unexpected = sorted(set(records_by_case) - set(selected_cases), key=_case_number)
        raise CpSupportAggregateError(
            "Cp receipt pairs must cover the requested case set exactly once "
            f"(missing={missing}, unexpected={unexpected})"
        )
    records = [records_by_case[case_id] for case_id in selected_cases]
    inventory_records = [inventories_by_case[case_id] for case_id in selected_cases]

    owner_statuses: Counter[str] = Counter()
    mapping_reasons: Counter[str] = Counter()
    truth_reasons: Counter[str] = Counter()
    review_flags: Counter[str] = Counter()
    pressure_vtk_types: Counter[str] = Counter()
    for record in records:
        validity = record["validity"]
        owner_statuses.update(validity["owner_review_status_counts"])
        mapping_reasons.update(validity["mapping_invalid_reason_counts"])
        truth_reasons.update(validity["truth_invalid_reason_counts"])
        review_flags.update(validity["review_flag_counts"])
        pressure_vtk_types[record["boundary"]["pMeanTrim_vtk_data_type"]] += 1

    common_source = {
        "dataset": {
            "provider": "Hugging Face Hub",
            "repo_id": pin.repository_id,
            "revision": pin.repository_revision,
        },
        "native_source_pin": {
            "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
            "sha256": pin_sha256,
        },
        "autocfd5": {
            "profile_sha256": profile_sha256,
            "registry_sha256": registry_sha256,
            "probe_count": CP_PROBE_COUNT,
        },
    }
    inventory: dict[str, Any] = {
        "schema": STL_INVENTORY_SCHEMA,
        "mode": "partial_pilot" if pilot else "complete",
        "status": (
            "incomplete_non_public_pilot"
            if pilot
            else "complete_candidate_inventory_pending_owner_visual_review"
        ),
        "complete": not pilot,
        "public_evidence_eligible": not pilot,
        "public_scoring_support_eligible": False,
        "owner_visual_signoff_claimed": False,
        "separate_from_native_source_pin": True,
        "case_count": len(inventory_records),
        "official_case_count": len(official_cases),
        "omitted_official_case_count": len(official_cases) - len(inventory_records),
        "identity_contract": (
            "strict_named_ASCII_drivaer_N.stl_SHA256_size_global_raw_facet_order"
        ),
        "source": common_source,
        "aggregate": {
            "size_bytes": sum(record["size_bytes"] for record in inventory_records),
            "facet_count": sum(record["facet_count"] for record in inventory_records),
            "solid_count": sum(record["solid_count"] for record in inventory_records),
        },
        "cases": inventory_records,
    }
    if pilot:
        inventory["pilot_warning"] = (
            "incomplete subset; not a public source-inventory or scoring-support artifact"
        )
    inventory_sha256 = hashlib.sha256(_canonical_json_bytes(inventory)).hexdigest()

    aggregate: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "mode": "partial_pilot" if pilot else "complete",
        "status": (
            "incomplete_non_public_pilot"
            if pilot
            else "complete_all_official_cases_candidate_evidence_pending_owner_visual_review"
        ),
        "complete": not pilot,
        "public_evidence_eligible": not pilot,
        "public_scoring_support_eligible": False,
        "owner_visual_signoff_claimed": False,
        "activation_status": "does_not_activate_scoring_contract",
        "case_count": len(records),
        "official_case_count": len(official_cases),
        "omitted_official_case_count": len(official_cases) - len(records),
        "case_order": "native_source_pin_increasing_run_number",
        "source": common_source,
        "dependencies": dict(PINNED_DEPENDENCIES),
        "candidate_stl_inventory": {
            "schema": STL_INVENTORY_SCHEMA,
            "sha256": inventory_sha256,
            "separate_from_native_source_pin": True,
        },
        "aggregate": {
            "probe_row_count": CP_PROBE_COUNT * len(records),
            "mapping_valid_count": sum(
                record["validity"]["mapping_valid_count"] for record in records
            ),
            "mapping_invalid_count": sum(
                record["validity"]["mapping_invalid_count"] for record in records
            ),
            "mapping_invalid_reason_counts": dict(sorted(mapping_reasons.items())),
            "truth_valid_count": sum(
                record["validity"]["truth_valid_count"] for record in records
            ),
            "truth_invalid_count": sum(
                record["validity"]["truth_invalid_count"] for record in records
            ),
            "truth_invalid_reason_counts": dict(sorted(truth_reasons.items())),
            "support_valid_count": sum(
                record["validity"]["support_valid_count"] for record in records
            ),
            "owner_review_status_counts": dict(sorted(owner_statuses.items())),
            "review_flag_counts": dict(sorted(review_flags.items())),
            "pMeanTrim_vtk_data_type_counts": dict(
                sorted(pressure_vtk_types.items())
            ),
            "receipt_json_size_bytes": sum(
                record["receipt_json_size_bytes"] for record in records
            ),
            "mapping_csv_size_bytes": sum(
                record["mapping_csv_size_bytes"] for record in records
            ),
            "stl_size_bytes": sum(record["stl"]["size_bytes"] for record in records),
            "stl_facet_count": sum(record["stl"]["facet_count"] for record in records),
            "boundary_size_bytes": sum(
                record["boundary"]["size_bytes"] for record in records
            ),
            "boundary_polygon_count": sum(
                record["boundary"]["polygon_count"] for record in records
            ),
        },
        "cases": records,
    }
    if pilot:
        aggregate["pilot_warning"] = (
            "incomplete subset; not public evidence and not scoring-support activation"
        )
    _assert_no_absolute_paths(inventory)
    _assert_no_absolute_paths(aggregate)
    return aggregate, inventory


def write_evidence(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write deterministic compact path-free JSON evidence."""

    _assert_no_absolute_paths(value)
    payload = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source-pin", type=Path, default=DEFAULT_NATIVE_SOURCE_PIN)
    parser.add_argument("--profile", type=Path, default=DEFAULT_AUTOCFD5_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stl-inventory-output", type=Path, required=True)
    parser.add_argument(
        "--pilot-case",
        action="append",
        default=None,
        help="explicit official case ID; repeat only for an incomplete non-public pilot",
    )
    parser.add_argument(
        "--receipt",
        nargs=2,
        action="append",
        required=True,
        metavar=("JSON", "CSV"),
        help="one candidate per-case JSON and its compact mapping CSV; repeat per case",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.resolve() == args.stl_inventory_output.resolve():
        raise CpSupportAggregateError("aggregate and STL inventory output paths must differ")
    protected_inputs = {
        args.native_source_pin.resolve(),
        args.profile.resolve(),
        *(
            Path(path).resolve()
            for pair in args.receipt
            for path in pair
        ),
    }
    if (
        args.output.resolve() in protected_inputs
        or args.stl_inventory_output.resolve() in protected_inputs
    ):
        raise CpSupportAggregateError("output paths must not overwrite any validated input")
    aggregate, inventory = aggregate_cp_support_receipts(
        native_source_pin_path=args.native_source_pin,
        profile_path=args.profile,
        receipt_pairs=tuple((Path(pair[0]), Path(pair[1])) for pair in args.receipt),
        pilot_case_ids=args.pilot_case,
    )
    write_evidence(args.stl_inventory_output, inventory)
    write_evidence(args.output, aggregate)
    print(
        json.dumps(
            {
                "case_count": aggregate["case_count"],
                "complete": aggregate["complete"],
                "mode": aggregate["mode"],
                "owner_visual_signoff_claimed": False,
                "status": aggregate["status"],
                "output": str(args.output),
                "stl_inventory_output": str(args.stl_inventory_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
