#!/usr/bin/env python3
"""Aggregate strict DrivAerML native-surface force replay receipts.

The per-case evaluator in :mod:`audit_drivaerml_force_case` performs the
expensive VTK replay.  This tool verifies exactly one resulting receipt for
every pinned public case, cross-checks each receipt against the immutable
native source pin and authoritative aggregate truth table, and emits compact
deterministic public evidence.  Local input paths and nondeterministic runtime
durations are deliberately excluded from that evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE_SOURCE_PIN = (
    ROOT / "benchmark-specs" / "drivaerml" / "proposal" / "native-source-pin.json"
)

OFFICIAL_NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
OFFICIAL_TRUTH_SHA256 = (
    "4e9e003da38ccdcacad359451079888361eae221d3c8dad7fd5682250d257865"
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

COEFFICIENT_IDS = ("Cd", "Cl", "CmPitch", "Clf", "Clr", "Cs")
CSV_COLUMNS = ("run", "cd", "cl", "clf", "clr", "cs")
COEFFICIENT_ABSOLUTE_TOLERANCE = 1.0e-6
CHUNK_INVARIANCE_ABSOLUTE_TOLERANCE = 2.0e-12
SURFACE_AREA_RELATIVE_TOLERANCE = 6.0e-8
SOURCE_LIFT_CLOSURE_ABSOLUTE_TOLERANCE = 1.0e-7
SOURCE_LIFT_CLOSURE_BINARY_COMPARISON_GUARD = 1.0e-15
RECONSTRUCTED_LIFT_CLOSURE_ABSOLUTE_TOLERANCE = 2.0e-12

_CASE_RE = re.compile(r"run_([1-9][0-9]*)")
_RUN_RE = re.compile(r"[1-9][0-9]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")


class ForceAggregateError(ValueError):
    """Raised when force replay evidence is incomplete or inconsistent."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForceAggregateError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except ForceAggregateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ForceAggregateError(f"cannot read valid {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ForceAggregateError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForceAggregateError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result))
        unexpected = sorted(set(result) - expected_set)
        raise ForceAggregateError(
            f"{label} keys differ from the receipt schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ForceAggregateError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ForceAggregateError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ForceAggregateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ForceAggregateError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise ForceAggregateError(f"{label} must be non-negative")
    return result


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise ForceAggregateError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _case_number(case_id: str) -> int:
    match = _CASE_RE.fullmatch(case_id)
    if match is None:
        raise ForceAggregateError(f"invalid canonical case ID {case_id!r}")
    return int(match.group(1))


def _expected_cases(case_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_string(case_id, "expected case ID") for case_id in case_ids)
    if not result:
        raise ForceAggregateError("expected case set cannot be empty")
    for case_id in result:
        _case_number(case_id)
    if len(result) != len(set(result)):
        raise ForceAggregateError("expected case IDs must be unique")
    if tuple(sorted(result, key=_case_number)) != result:
        raise ForceAggregateError("expected case IDs must use increasing run-number order")
    return result


def _relative_public_path(value: object, label: str) -> str:
    result = _string(value, label).replace("\\", "/")
    if result.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(result) or ".." in result.split("/"):
        raise ForceAggregateError(f"{label} must be a relative public-release path")
    return result


def _path_has_suffix(value: object, expected_suffix: str, label: str) -> None:
    actual = _string(value, label).replace("\\", "/").rstrip("/")
    expected = expected_suffix.replace("\\", "/").strip("/")
    if actual != expected and not actual.endswith("/" + expected):
        raise ForceAggregateError(f"{label} does not identify pinned path {expected!r}")


def _same_float(actual: float, expected: float, *, atol: float = 1.0e-15) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol)


def _finite_vector(value: object, count: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ForceAggregateError(f"{label} must contain exactly {count} values")
    return tuple(_finite(item, label) for item in value)


def _validate_native_pin(
    path: Path,
    *,
    expected_case_ids: tuple[str, ...],
    expected_pin_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], str]:
    pin_sha256 = sha256_file(path)
    if expected_pin_sha256 is not None and pin_sha256 != expected_pin_sha256:
        raise ForceAggregateError(
            "native source pin SHA-256 mismatch: "
            f"expected {expected_pin_sha256}, got {pin_sha256}"
        )
    pin = _read_json(path, "native source pin")
    if pin.get("schema") != "drivaerml-fluidsbench-public-native-source-pin-v1":
        raise ForceAggregateError("unexpected native source pin schema")
    if pin.get("schema_version") != 1:
        raise ForceAggregateError("unexpected native source pin schema_version")

    repository = _exact_keys(
        pin.get("repository"),
        {"provider", "repo_id", "repo_type", "revision"},
        "native source pin repository",
    )
    if (
        repository["provider"] != "Hugging Face Hub"
        or repository["repo_id"] != "neashton/drivaerml"
        or repository["repo_type"] != "dataset"
        or repository["revision"] != OFFICIAL_REVISION
    ):
        raise ForceAggregateError("native source pin repository identity is not frozen")

    cases = pin.get("cases")
    if not isinstance(cases, list):
        raise ForceAggregateError("native source pin cases must be a list")
    if len(cases) != len(expected_case_ids):
        raise ForceAggregateError("native source pin case count differs from expected set")
    by_case: dict[str, Mapping[str, Any]] = {}
    ordered_ids: list[str] = []
    for position, raw_case in enumerate(cases):
        case = _mapping(raw_case, f"native source pin case {position}")
        case_id = _string(case.get("case_id"), "native source pin case_id")
        if case_id in by_case:
            raise ForceAggregateError(f"duplicate native source pin case {case_id}")
        run_number = _integer(case.get("run_number"), f"{case_id} run_number", minimum=1)
        if run_number != _case_number(case_id):
            raise ForceAggregateError(f"{case_id} run_number does not match case ID")
        boundary = _mapping(case.get("boundary"), f"{case_id} boundary pin")
        boundary_path = _relative_public_path(
            boundary.get("path"), f"{case_id} boundary path"
        )
        if boundary_path != f"{case_id}/boundary_{run_number}.vtp":
            raise ForceAggregateError(f"{case_id} boundary path is not canonical")
        boundary_sha = _sha256(
            boundary.get("lfs_sha256"), f"{case_id} boundary SHA-256"
        )
        _integer(boundary.get("size_bytes"), f"{case_id} boundary size", minimum=1)

        area = _mapping(case.get("surface_cell_area"), f"{case_id} area pin")
        area_path = _relative_public_path(area.get("path"), f"{case_id} area path")
        if area_path != f"{case_id}/boundary_cell_area_{run_number}.npy":
            raise ForceAggregateError(f"{case_id} surface-area path is not canonical")
        _sha256(area.get("lfs_sha256"), f"{case_id} area SHA-256")
        if area.get("source_boundary_sha256") != boundary_sha:
            raise ForceAggregateError(f"{case_id} area source boundary hash mismatch")
        if area.get("dtype") != "<f4":
            raise ForceAggregateError(f"{case_id} area dtype must be little-endian float32")
        entity_count = _integer(
            area.get("element_count"), f"{case_id} area element_count", minimum=1
        )
        area_size = _integer(area.get("size_bytes"), f"{case_id} area size", minimum=1)
        if area_size != 128 + 4 * entity_count:
            raise ForceAggregateError(f"{case_id} NPY area size is inconsistent with count")
        by_case[case_id] = case
        ordered_ids.append(case_id)
    if tuple(ordered_ids) != expected_case_ids:
        raise ForceAggregateError("native source pin case IDs/order differ from expected set")

    case_scope = _mapping(pin.get("case_scope"), "native source pin case_scope")
    if case_scope.get("case_count") != len(expected_case_ids):
        raise ForceAggregateError("native source pin case_scope count mismatch")
    if expected_case_ids == OFFICIAL_CASE_IDS:
        if (
            case_scope.get("run_number_min") != 1
            or case_scope.get("run_number_max") != 500
            or case_scope.get("unavailable_or_held_back_run_numbers")
            != sorted(OFFICIAL_UNAVAILABLE_RUNS)
        ):
            raise ForceAggregateError("official 484-case scope is not exact")
        totals = _mapping(pin.get("totals"), "native source pin totals")
        for key in (
            "boundary_file_count",
            "logical_volume_count",
            "surface_cell_area_file_count",
        ):
            if totals.get(key) != len(OFFICIAL_CASE_IDS):
                raise ForceAggregateError(f"native source pin total {key!r} is inconsistent")

    supports = _mapping(
        pin.get("authoritative_support_files"),
        "native source pin authoritative_support_files",
    )
    truth_binding = _mapping(
        supports.get("force_mom_constref_all"),
        "native source pin force_mom_constref_all",
    )
    if _relative_public_path(
        truth_binding.get("path"), "authoritative truth public path"
    ) != "force_mom_constref_all.csv":
        raise ForceAggregateError("authoritative truth path is not frozen")
    truth_sha = _sha256(
        truth_binding.get("sha256"), "authoritative truth pinned SHA-256"
    )
    if expected_case_ids == OFFICIAL_CASE_IDS and truth_sha != OFFICIAL_TRUTH_SHA256:
        raise ForceAggregateError("official authoritative truth SHA-256 is not frozen")
    _integer(
        truth_binding.get("size_bytes"), "authoritative truth pinned size", minimum=1
    )
    return pin, by_case, pin_sha256


def _load_authoritative_truth(
    path: Path,
    *,
    expected_case_ids: tuple[str, ...],
    truth_binding: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], str]:
    truth_sha256 = sha256_file(path)
    expected_sha = _sha256(
        truth_binding.get("sha256"), "authoritative truth pinned SHA-256"
    )
    if truth_sha256 != expected_sha:
        raise ForceAggregateError(
            "authoritative truth SHA-256 mismatch: "
            f"expected {expected_sha}, got {truth_sha256}"
        )
    expected_size = _integer(
        truth_binding.get("size_bytes"), "authoritative truth pinned size", minimum=1
    )
    if path.stat().st_size != expected_size:
        raise ForceAggregateError("authoritative truth size differs from source pin")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise ForceAggregateError(
                    f"authoritative truth header must be exactly {CSV_COLUMNS}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ForceAggregateError("cannot read authoritative truth CSV") from error
    if len(rows) != len(expected_case_ids):
        raise ForceAggregateError("authoritative truth row count differs from case set")

    expected_set = set(expected_case_ids)
    result: dict[str, dict[str, float]] = {}
    for row_number, row in enumerate(rows, start=2):
        if set(row) != set(CSV_COLUMNS) or any(row[key] is None for key in CSV_COLUMNS):
            raise ForceAggregateError(
                f"authoritative truth row {row_number} does not match the exact CSV schema"
            )
        run_text = row["run"]
        if _RUN_RE.fullmatch(run_text) is None:
            raise ForceAggregateError(
                f"authoritative truth row {row_number} run must be a base-10 integer"
            )
        case_id = f"run_{int(run_text)}"
        if case_id in result:
            raise ForceAggregateError(f"duplicate authoritative truth case {case_id}")
        if case_id not in expected_set:
            raise ForceAggregateError(f"unexpected authoritative truth case {case_id}")
        values: dict[str, float] = {}
        csv_coefficient_ids = (
            COEFFICIENT_IDS[:2] + COEFFICIENT_IDS[3:5] + ("Cs",)
        )
        for column, coefficient_id in zip(CSV_COLUMNS[1:], csv_coefficient_ids):
            try:
                parsed = float(row[column])
            except (TypeError, ValueError) as error:
                raise ForceAggregateError(
                    f"authoritative truth {case_id} column {column!r} is not numeric"
                ) from error
            if not math.isfinite(parsed):
                raise ForceAggregateError(
                    f"authoritative truth {case_id} column {column!r} is non-finite"
                )
            values[coefficient_id] = parsed
        values["CmPitch"] = (values["Clf"] - values["Clr"]) / 2.0
        result[case_id] = values
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result), key=_case_number)
        raise ForceAggregateError(f"authoritative truth is missing cases {missing}")
    return result, truth_sha256


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    pinned_case: Mapping[str, Any],
    authoritative_truth: Mapping[str, float],
) -> dict[str, Any]:
    if receipt.get("schema") != "drivaerml-native-surface-force-case-audit-v1":
        raise ForceAggregateError("unexpected force case receipt schema")
    if receipt.get("status") != "passed_candidate_evaluator_case_audit":
        raise ForceAggregateError("force case receipt does not have passing status")
    case_id = _string(receipt.get("case_id"), "receipt case_id")
    run_number = _case_number(case_id)
    if pinned_case.get("case_id") != case_id:
        raise ForceAggregateError(f"receipt case {case_id} does not match source pin")

    source = _exact_keys(
        receipt.get("source"), {"boundary", "surface_areas", "truth"}, f"{case_id} source"
    )
    boundary_source = _exact_keys(
        source["boundary"], {"path", "sha256"}, f"{case_id} boundary source"
    )
    area_source = _exact_keys(
        source["surface_areas"],
        {"path", "sha256", "role"},
        f"{case_id} surface-area source",
    )
    truth_source = _exact_keys(
        source["truth"], {"path", "sha256"}, f"{case_id} mirror truth source"
    )
    boundary_pin = _mapping(pinned_case.get("boundary"), f"{case_id} boundary pin")
    area_pin = _mapping(
        pinned_case.get("surface_cell_area"), f"{case_id} surface-area pin"
    )
    _path_has_suffix(
        boundary_source["path"], boundary_pin["path"], f"{case_id} boundary source path"
    )
    _path_has_suffix(
        area_source["path"], area_pin["path"], f"{case_id} surface-area source path"
    )
    _path_has_suffix(
        truth_source["path"],
        f"{case_id}/force_mom_constref_{run_number}.csv",
        f"{case_id} mirror truth source path",
    )
    if boundary_source["sha256"] != boundary_pin.get("lfs_sha256"):
        raise ForceAggregateError(f"{case_id} receipt boundary SHA-256 mismatch")
    if area_source["sha256"] != area_pin.get("lfs_sha256"):
        raise ForceAggregateError(f"{case_id} receipt surface-area SHA-256 mismatch")
    if area_source["role"] != "fixed_input_not_regenerated":
        raise ForceAggregateError(f"{case_id} did not treat surface areas as fixed input")
    mirror_truth_sha256 = _sha256(
        truth_source["sha256"], f"{case_id} mirror truth SHA-256"
    )

    if receipt.get("raw_cell_order") != "unchanged zero-based VTK CellData tuple order":
        raise ForceAggregateError(f"{case_id} raw native cell order is not preserved")
    units = _exact_keys(
        receipt.get("units"),
        {"coordinates", "surface_areas", "pMeanTrim", "wallShearStressMeanTrim"},
        f"{case_id} units",
    )
    if dict(units) != {
        "coordinates": "m",
        "surface_areas": "m^2",
        "pMeanTrim": "m^2/s^2",
        "wallShearStressMeanTrim": "m^2/s^2",
    }:
        raise ForceAggregateError(f"{case_id} field units differ from the contract")

    expected_entities = _integer(
        area_pin.get("element_count"), f"{case_id} pinned polygon count", minimum=1
    )
    mesh = _exact_keys(
        receipt.get("mesh_and_fields"),
        {
            "vtk_version",
            "point_count",
            "polygon_count",
            "available_point_arrays",
            "available_cell_arrays",
            "pressure_dtype",
            "wall_shear_dtype",
        },
        f"{case_id} mesh_and_fields",
    )
    if _integer(mesh["polygon_count"], f"{case_id} polygon_count", minimum=1) != expected_entities:
        raise ForceAggregateError(f"{case_id} receipt polygon count differs from source pin")
    _integer(mesh["point_count"], f"{case_id} point_count", minimum=1)
    if mesh["pressure_dtype"] != "float32" or mesh["wall_shear_dtype"] != "float32":
        raise ForceAggregateError(f"{case_id} native surface field dtype is not float32")
    cell_arrays = mesh["available_cell_arrays"]
    if not isinstance(cell_arrays, list) or not {
        "pMeanTrim",
        "wallShearStressMeanTrim",
    }.issubset(cell_arrays):
        raise ForceAggregateError(f"{case_id} required native CellData arrays are missing")
    if not isinstance(mesh["available_point_arrays"], list):
        raise ForceAggregateError(f"{case_id} available_point_arrays must be a list")
    vtk_version = _string(mesh["vtk_version"], f"{case_id} VTK version")

    coefficients = _exact_keys(
        receipt.get("coefficients"),
        {
            *COEFFICIENT_IDS,
            "entity_count",
            "force_n",
            "lift_closure_abs",
            "moment_about_forces_cor_n_m",
        },
        f"{case_id} coefficients",
    )
    if _integer(
        coefficients["entity_count"], f"{case_id} coefficient entity_count", minimum=1
    ) != expected_entities:
        raise ForceAggregateError(f"{case_id} coefficient entity count mismatch")
    coefficient_values = {
        key: _finite(coefficients[key], f"{case_id} coefficient {key}")
        for key in COEFFICIENT_IDS
    }
    _finite_vector(coefficients["force_n"], 3, f"{case_id} force_n")
    _finite_vector(
        coefficients["moment_about_forces_cor_n_m"],
        3,
        f"{case_id} moment_about_forces_cor_n_m",
    )
    reconstructed_closure = abs(
        coefficient_values["Cl"]
        - coefficient_values["Clf"]
        - coefficient_values["Clr"]
    )
    reported_closure = _finite(
        coefficients["lift_closure_abs"],
        f"{case_id} reported lift closure",
        nonnegative=True,
    )
    if not _same_float(reported_closure, reconstructed_closure):
        raise ForceAggregateError(f"{case_id} reported lift closure is inconsistent")
    if reconstructed_closure > RECONSTRUCTED_LIFT_CLOSURE_ABSOLUTE_TOLERANCE:
        raise ForceAggregateError(f"{case_id} reconstructed lift closure exceeds tolerance")
    axle_pitch = (coefficient_values["Clf"] - coefficient_values["Clr"]) / 2.0
    if not _same_float(coefficient_values["CmPitch"], axle_pitch):
        raise ForceAggregateError(f"{case_id} CmPitch/axle-load identity is inconsistent")

    mirror_truth = _exact_keys(
        receipt.get("truth"),
        {"cd", "cl", "clf", "clr", "cmpitch", "cs"},
        f"{case_id} mirror truth",
    )
    mirror_canonical = {
        "Cd": _finite(mirror_truth["cd"], f"{case_id} mirror Cd"),
        "Cl": _finite(mirror_truth["cl"], f"{case_id} mirror Cl"),
        "CmPitch": _finite(mirror_truth["cmpitch"], f"{case_id} mirror CmPitch"),
        "Clf": _finite(mirror_truth["clf"], f"{case_id} mirror Clf"),
        "Clr": _finite(mirror_truth["clr"], f"{case_id} mirror Clr"),
        "Cs": _finite(mirror_truth["cs"], f"{case_id} mirror Cs"),
    }
    for key in COEFFICIENT_IDS:
        if mirror_canonical[key] != authoritative_truth[key]:
            raise ForceAggregateError(
                f"{case_id} mirror truth {key} differs from authoritative aggregate"
            )
    expected_mirror_pitch = (mirror_canonical["Clf"] - mirror_canonical["Clr"]) / 2.0
    if mirror_canonical["CmPitch"] != expected_mirror_pitch:
        raise ForceAggregateError(f"{case_id} mirror CmPitch is not derived from Clf/Clr")

    coefficient_tolerance = _finite(
        receipt.get("coefficient_absolute_tolerance"),
        f"{case_id} coefficient tolerance",
        nonnegative=True,
    )
    if coefficient_tolerance != COEFFICIENT_ABSOLUTE_TOLERANCE:
        raise ForceAggregateError(f"{case_id} coefficient tolerance is not frozen")
    reported_differences = _exact_keys(
        receipt.get("absolute_truth_difference"),
        COEFFICIENT_IDS,
        f"{case_id} absolute_truth_difference",
    )
    differences: dict[str, float] = {}
    for key in COEFFICIENT_IDS:
        difference = abs(coefficient_values[key] - authoritative_truth[key])
        reported = _finite(
            reported_differences[key],
            f"{case_id} reported {key} truth difference",
            nonnegative=True,
        )
        if not _same_float(reported, difference):
            raise ForceAggregateError(f"{case_id} reported {key} difference is inconsistent")
        if difference > coefficient_tolerance:
            raise ForceAggregateError(f"{case_id} {key} difference exceeds tolerance")
        differences[key] = difference

    area = _exact_keys(
        receipt.get("area_audit"),
        {
            "entity_count",
            "calculated_sum_m2",
            "published_sum_m2",
            "maximum_absolute_difference_m2",
            "maximum_relative_difference",
        },
        f"{case_id} area_audit",
    )
    if (
        _integer(area["entity_count"], f"{case_id} area entity_count", minimum=1)
        != expected_entities
    ):
        raise ForceAggregateError(f"{case_id} area entity count mismatch")
    calculated_area_sum = _finite(
        area["calculated_sum_m2"], f"{case_id} calculated area sum"
    )
    published_area_sum = _finite(
        area["published_sum_m2"], f"{case_id} published area sum"
    )
    if calculated_area_sum <= 0.0 or published_area_sum <= 0.0:
        raise ForceAggregateError(f"{case_id} area sums must be positive")
    maximum_area_absolute = _finite(
        area["maximum_absolute_difference_m2"],
        f"{case_id} maximum area absolute difference",
        nonnegative=True,
    )
    maximum_area_relative = _finite(
        area["maximum_relative_difference"],
        f"{case_id} maximum area relative difference",
        nonnegative=True,
    )
    if maximum_area_relative > SURFACE_AREA_RELATIVE_TOLERANCE + 1.0e-15:
        raise ForceAggregateError(f"{case_id} surface-area audit exceeds tolerance")
    if abs(calculated_area_sum - published_area_sum) > (
        SURFACE_AREA_RELATIVE_TOLERANCE * calculated_area_sum + 1.0e-15
    ):
        raise ForceAggregateError(f"{case_id} aggregate surface-area sums disagree")

    chunk = _exact_keys(
        receipt.get("chunk_invariance"),
        {
            "chunk_polygons",
            "absolute_difference",
            "maximum_absolute_difference",
            "tolerance",
        },
        f"{case_id} chunk_invariance",
    )
    chunk_sizes = chunk["chunk_polygons"]
    if (
        not isinstance(chunk_sizes, list)
        or len(chunk_sizes) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in chunk_sizes
        )
        or chunk_sizes[0] == chunk_sizes[1]
    ):
        raise ForceAggregateError(f"{case_id} chunk sizes must be two distinct positives")
    chunk_tolerance = _finite(
        chunk["tolerance"], f"{case_id} chunk tolerance", nonnegative=True
    )
    if chunk_tolerance != CHUNK_INVARIANCE_ABSOLUTE_TOLERANCE:
        raise ForceAggregateError(f"{case_id} chunk invariance tolerance is not frozen")
    chunk_reported = _exact_keys(
        chunk["absolute_difference"], COEFFICIENT_IDS, f"{case_id} chunk differences"
    )
    chunk_differences = {
        key: _finite(
            chunk_reported[key], f"{case_id} {key} chunk difference", nonnegative=True
        )
        for key in COEFFICIENT_IDS
    }
    maximum_chunk = max(chunk_differences.values())
    reported_maximum_chunk = _finite(
        chunk["maximum_absolute_difference"],
        f"{case_id} maximum chunk difference",
        nonnegative=True,
    )
    if not _same_float(reported_maximum_chunk, maximum_chunk, atol=1.0e-18):
        raise ForceAggregateError(f"{case_id} maximum chunk difference is inconsistent")
    if maximum_chunk > chunk_tolerance:
        raise ForceAggregateError(f"{case_id} chunk invariance exceeds tolerance")

    runtime = _exact_keys(
        receipt.get("runtime"),
        {"python", "numpy", "vtk", "elapsed_seconds"},
        f"{case_id} runtime",
    )
    python_version = _string(runtime["python"], f"{case_id} Python version")
    numpy_version = _string(runtime["numpy"], f"{case_id} NumPy version")
    runtime_vtk = _string(runtime["vtk"], f"{case_id} runtime VTK version")
    if runtime_vtk != vtk_version:
        raise ForceAggregateError(f"{case_id} runtime/mesh VTK versions differ")
    _finite(runtime["elapsed_seconds"], f"{case_id} elapsed_seconds", nonnegative=True)

    return {
        "case_id": case_id,
        "receipt_sha256": receipt_sha256,
        "mirror_truth_sha256": mirror_truth_sha256,
        "differences": differences,
        "truth_closure": abs(
            authoritative_truth["Cl"]
            - authoritative_truth["Clf"]
            - authoritative_truth["Clr"]
        ),
        "reconstructed_closure": reconstructed_closure,
        "area": {
            "entity_count": expected_entities,
            "calculated_sum_m2": calculated_area_sum,
            "published_sum_m2": published_area_sum,
            "maximum_absolute_difference_m2": maximum_area_absolute,
            "maximum_relative_difference": maximum_area_relative,
        },
        "chunk_sizes": tuple(chunk_sizes),
        "chunk_differences": chunk_differences,
        "maximum_chunk_difference": maximum_chunk,
        "versions": {
            "python": python_version,
            "numpy": numpy_version,
            "vtk": runtime_vtk,
        },
    }


def _difference_summary(
    values: Sequence[tuple[str, float]],
) -> dict[str, float | str]:
    if not values:
        raise ForceAggregateError("cannot summarize an empty value sequence")
    maximum_case, maximum = max(values, key=lambda item: item[1])
    count = len(values)
    return {
        "case_id_at_maximum": maximum_case,
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": math.fsum(value for _, value in values) / count,
        "root_mean_square_difference": math.sqrt(
            math.fsum(value * value for _, value in values) / count
        ),
    }


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
            raise ForceAggregateError(f"{label} contains a local absolute path")


def aggregate_force_replay(
    *,
    native_source_pin_path: Path,
    authoritative_truth_path: Path,
    receipt_paths: Sequence[Path],
    expected_case_ids: Sequence[str] = OFFICIAL_CASE_IDS,
    expected_pin_sha256: str | None = OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
) -> dict[str, Any]:
    """Validate a complete case set and return deterministic aggregate evidence.

    Production callers use the defaults, which hard-bind the exact 484-case
    source pin.  Tests may supply a smaller ordered case set and set
    ``expected_pin_sha256=None``; the CLI intentionally exposes neither
    override.
    """

    cases = _expected_cases(expected_case_ids)
    pin, pinned_cases, pin_sha256 = _validate_native_pin(
        Path(native_source_pin_path),
        expected_case_ids=cases,
        expected_pin_sha256=expected_pin_sha256,
    )
    supports = _mapping(
        pin["authoritative_support_files"], "native source pin support files"
    )
    truth_binding = _mapping(
        supports["force_mom_constref_all"], "authoritative truth binding"
    )
    truth, truth_sha256 = _load_authoritative_truth(
        Path(authoritative_truth_path),
        expected_case_ids=cases,
        truth_binding=truth_binding,
    )

    receipts_by_case: dict[str, dict[str, Any]] = {}
    for receipt_path in receipt_paths:
        path = Path(receipt_path)
        receipt_sha256 = sha256_file(path)
        receipt = _read_json(path, "force case receipt")
        case_id = _string(receipt.get("case_id"), "receipt case_id")
        if case_id in receipts_by_case:
            raise ForceAggregateError(f"duplicate force receipt for {case_id}")
        if case_id not in pinned_cases:
            raise ForceAggregateError(f"unexpected force receipt case {case_id}")
        receipts_by_case[case_id] = _validate_receipt(
            receipt,
            receipt_sha256=receipt_sha256,
            pinned_case=pinned_cases[case_id],
            authoritative_truth=truth[case_id],
        )
    if set(receipts_by_case) != set(cases):
        missing = sorted(set(cases) - set(receipts_by_case), key=_case_number)
        unexpected = sorted(set(receipts_by_case) - set(cases), key=_case_number)
        raise ForceAggregateError(
            "force receipts must cover the expected case set exactly once "
            f"(missing={missing}, unexpected={unexpected})"
        )
    receipts = [receipts_by_case[case_id] for case_id in cases]

    truth_closure_values = [
        (row["case_id"], row["truth_closure"]) for row in receipts
    ]
    maximum_truth_closure = max(value for _, value in truth_closure_values)
    if maximum_truth_closure > SOURCE_LIFT_CLOSURE_ABSOLUTE_TOLERANCE and not math.isclose(
        maximum_truth_closure,
        SOURCE_LIFT_CLOSURE_ABSOLUTE_TOLERANCE,
        rel_tol=0.0,
        abs_tol=SOURCE_LIFT_CLOSURE_BINARY_COMPARISON_GUARD,
    ):
        raise ForceAggregateError("authoritative truth lift closure exceeds CSV tolerance")

    coefficient_summaries = {
        key: _difference_summary(
            [(row["case_id"], row["differences"][key]) for row in receipts]
        )
        for key in COEFFICIENT_IDS
    }
    reconstructed_closure_summary = _difference_summary(
        [(row["case_id"], row["reconstructed_closure"]) for row in receipts]
    )
    truth_closure_summary = _difference_summary(truth_closure_values)
    chunk_by_coefficient = {
        key: _difference_summary(
            [(row["case_id"], row["chunk_differences"][key]) for row in receipts]
        )
        for key in COEFFICIENT_IDS
    }
    chunk_overall = _difference_summary(
        [(row["case_id"], row["maximum_chunk_difference"]) for row in receipts]
    )
    maximum_area_absolute_row = max(
        receipts, key=lambda row: row["area"]["maximum_absolute_difference_m2"]
    )
    maximum_area_relative_row = max(
        receipts, key=lambda row: row["area"]["maximum_relative_difference"]
    )

    repository = _mapping(pin["repository"], "native source pin repository")
    evidence: dict[str, Any] = {
        "schema": "drivaerml-native-surface-force-replay-aggregate-v1",
        "status": (
            "passed_candidate_evaluator_all_484_case_force_replay"
            if cases == OFFICIAL_CASE_IDS
            else "passed_requested_case_set_force_replay"
        ),
        "case_count": len(cases),
        "case_order": "native_source_pin_increasing_run_number",
        "source": {
            "dataset": {
                "provider": repository["provider"],
                "repo_id": repository["repo_id"],
                "revision": repository["revision"],
            },
            "native_source_pin": {
                "schema": pin["schema"],
                "sha256": pin_sha256,
            },
            "authoritative_truth": {
                "path_at_pinned_dataset_revision": truth_binding["path"],
                "sha256": truth_sha256,
                "exact_header": list(CSV_COLUMNS),
            },
        },
        "tolerances": {
            "coefficient_absolute": COEFFICIENT_ABSOLUTE_TOLERANCE,
            "chunk_invariance_absolute": CHUNK_INVARIANCE_ABSOLUTE_TOLERANCE,
            "reconstructed_lift_closure_absolute": RECONSTRUCTED_LIFT_CLOSURE_ABSOLUTE_TOLERANCE,
            "source_lift_closure_absolute_csv_rounding": SOURCE_LIFT_CLOSURE_ABSOLUTE_TOLERANCE,
            "source_lift_closure_binary_comparison_guard_absolute": SOURCE_LIFT_CLOSURE_BINARY_COMPARISON_GUARD,
            "surface_area_relative": SURFACE_AREA_RELATIVE_TOLERANCE,
        },
        "coefficient_absolute_difference": coefficient_summaries,
        "lift_closure": {
            "equation": "Cl-(Clf+Clr)",
            "reconstructed": reconstructed_closure_summary,
            "authoritative_and_mirror_truth": truth_closure_summary,
        },
        "mirror_truth_validation": {
            "status": "exact_match_to_authoritative_aggregate",
            "case_count": len(cases),
            "coefficient_ids": list(COEFFICIENT_IDS),
        },
        "surface_area_audit": {
            "role": "fixed_input_not_regenerated",
            "association": "native_boundary_polygon_raw_CellData_order",
            "entity_count": sum(row["area"]["entity_count"] for row in receipts),
            "calculated_sum_m2": math.fsum(
                row["area"]["calculated_sum_m2"] for row in receipts
            ),
            "published_sum_m2": math.fsum(
                row["area"]["published_sum_m2"] for row in receipts
            ),
            "absolute_total_sum_difference_m2": abs(
                math.fsum(row["area"]["calculated_sum_m2"] for row in receipts)
                - math.fsum(row["area"]["published_sum_m2"] for row in receipts)
            ),
            "maximum_polygon_absolute_difference_m2": maximum_area_absolute_row["area"][
                "maximum_absolute_difference_m2"
            ],
            "maximum_polygon_absolute_difference_case_id": maximum_area_absolute_row[
                "case_id"
            ],
            "maximum_polygon_relative_difference": maximum_area_relative_row["area"][
                "maximum_relative_difference"
            ],
            "maximum_polygon_relative_difference_case_id": maximum_area_relative_row[
                "case_id"
            ],
        },
        "chunk_invariance": {
            "by_coefficient": chunk_by_coefficient,
            "overall": chunk_overall,
            "chunk_polygon_pairs": [
                list(pair)
                for pair in sorted({row["chunk_sizes"] for row in receipts})
            ],
        },
        "evaluator_versions": {
            key: sorted({row["versions"][key] for row in receipts})
            for key in ("python", "numpy", "vtk")
        },
        "case_receipts": [
            {
                "case_id": row["case_id"],
                "receipt_sha256": row["receipt_sha256"],
                "mirror_truth_sha256": row["mirror_truth_sha256"],
            }
            for row in receipts
        ],
    }
    _assert_no_absolute_paths(evidence)
    return evidence


def write_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    """Write stable, human-reviewable JSON with one terminal newline."""

    _assert_no_absolute_paths(evidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-source-pin",
        type=Path,
        default=DEFAULT_NATIVE_SOURCE_PIN,
    )
    parser.add_argument("--authoritative-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "receipts",
        type=Path,
        nargs="+",
        help="exactly one audit_drivaerml_force_case.py JSON receipt per public case",
    )
    args = parser.parse_args()
    evidence = aggregate_force_replay(
        native_source_pin_path=args.native_source_pin,
        authoritative_truth_path=args.authoritative_truth,
        receipt_paths=args.receipts,
    )
    write_evidence(args.output, evidence)
    print(
        json.dumps(
            {
                "case_count": evidence["case_count"],
                "status": evidence["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
