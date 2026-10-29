#!/usr/bin/env python3
"""Aggregate strict DrivAerML velocity-cell assignment case evidence.

Complete mode is the default and requires exactly one receipt directory for
all 484 immutable public cases.  Partial aggregation is possible only through
one or more explicit ``--pilot-case`` arguments and is permanently labelled
non-public.  Every compact assignment row is reconstructed and checked; case
summaries and artifact hashes are never trusted without replay.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    VelocityCellAssignmentEvidence,
    load_autocfd5_definition,
)
from reference.drivaerml.source import load_native_source_pin  # noqa: E402
from reference.drivaerml.velocity_assignments import (  # noqa: E402
    CELL_EVALUATION_FAILURE_REASON_PREFIX,
    KERNEL_ID,
    NO_CLOSURE_CELL_REASON,
    OWNER_INVALID_REASONS,
    POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES,
    POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES,
    POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE,
    QUERY_CACHE_KEY_ID,
    TOLERANCE_REPLAY_M,
    assignment_evidence_sha256,
    candidate_kernel_settings,
    generate_definition_velocity_samples,
)
from scripts.generate_drivaerml_velocity_assignments import (  # noqa: E402
    ARTIFACT_SCHEMA,
    EXPECTED_AUTOCFD5_PROFILE_SHA256,
    EXPECTED_DATASET_REPOSITORY_ID,
    EXPECTED_DATASET_REVISION,
    EXPECTED_SAMPLE_COUNTS,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    RESOLUTIONS,
)


MANIFEST_SCHEMA = "drivaerml-velocity-cell-assignments-all-case-candidate-v1"
DEFAULT_NATIVE_SOURCE_PIN = (
    ROOT / "benchmark-specs" / "drivaerml" / "proposal" / "native-source-pin.json"
)
DEFAULT_AUTOCFD5_PROFILE = (
    ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
)
OFFICIAL_NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
OFFICIAL_UNAVAILABLE_RUNS = frozenset(
    {167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473}
)
OFFICIAL_CASE_IDS = tuple(
    f"run_{run_number}"
    for run_number in range(1, 501)
    if run_number not in OFFICIAL_UNAVAILABLE_RUNS
)
PINNED_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.2.6",
    "vtk": "9.5.2",
    "vtk_source": "vtk version 9.5.2",
}
PINNED_KERNEL_SETTINGS = candidate_kernel_settings()
PINNED_KERNEL_SETTINGS_SHA256 = (
    "6cbd2b2fb56fc782fd9e9990bd43f2bad7fd040b355f128555ecf101b369fa1b"
)
FALSE_CASE_CLAIMS = {
    "resolution_convergence": False,
    "ranked_result_invariance": False,
    "model_ordering": False,
    "owner_validity_mask_complete": False,
    "owner_scientific_signoff": False,
    "scoring_contract_active": False,
    "official_submission": False,
}
MANIFEST_FALSE_CLAIMS = {
    **FALSE_CASE_CLAIMS,
    "three_real_model_ordering": False,
    "all_case_tolerance_replay_complete": False,
    "independent_participant_dry_run": False,
}
ROW_FIELDS = [
    "profile_id",
    "sample_index",
    "point_m",
    "distance_m",
    "valid",
    "reason",
    "raw_vtk_cell_id",
    "candidate_count",
]
_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")
_QUERY_CACHE_KEY = struct.Struct(">dddd")


class VelocityAssignmentAggregateError(ValueError):
    """Raised when case mapping evidence is incomplete or inconsistent."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as source:
            while block := source.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise VelocityAssignmentAggregateError(
            f"cannot hash required evidence file: {error}"
        ) from error
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VelocityAssignmentAggregateError(
                f"JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise VelocityAssignmentAggregateError(f"{label} cannot be a symbolic link")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except VelocityAssignmentAggregateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VelocityAssignmentAggregateError(
            f"cannot read valid {label} JSON: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise VelocityAssignmentAggregateError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VelocityAssignmentAggregateError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result))
        unexpected = sorted(set(result) - expected_set)
        raise VelocityAssignmentAggregateError(
            f"{label} keys differ from schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return result


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise VelocityAssignmentAggregateError(f"{label} must be {qualifier}")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise VelocityAssignmentAggregateError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VelocityAssignmentAggregateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VelocityAssignmentAggregateError(f"{label} must be finite")
    return result


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise VelocityAssignmentAggregateError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return result


def _case_number(case_id: str) -> int:
    match = _CASE_RE.fullmatch(case_id)
    if match is None:
        raise VelocityAssignmentAggregateError(
            f"invalid canonical case ID {case_id!r}"
        )
    return int(match.group(1))


def _ordered_case_ids(case_ids: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(_string(case_id, label) for case_id in case_ids)
    if not result:
        raise VelocityAssignmentAggregateError(f"{label} cannot be empty")
    for case_id in result:
        _case_number(case_id)
    if len(result) != len(set(result)):
        raise VelocityAssignmentAggregateError(f"{label} must be unique")
    return tuple(sorted(result, key=_case_number))


def _basename(value: object, label: str, expected: str | None = None) -> str:
    result = _string(value, label)
    if (
        result.startswith("/")
        or _WINDOWS_ABSOLUTE_RE.match(result)
        or "/" in result
        or "\\" in result
        or result in {".", ".."}
    ):
        raise VelocityAssignmentAggregateError(f"{label} must be a path-free basename")
    if expected is not None and result != expected:
        raise VelocityAssignmentAggregateError(
            f"{label} must equal {expected!r}, got {result!r}"
        )
    return result


def _same_float(actual: object, expected: float, label: str) -> float:
    value = _finite(actual, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=0.0):
        raise VelocityAssignmentAggregateError(
            f"{label} must equal {expected}, got {value}"
        )
    return value


def _evaluation_failure_ids(
    reason: str, *, cell_count: int, label: str
) -> tuple[int, ...] | None:
    """Parse the explicit conservative-invalid cell-evaluation reason."""

    if not reason.startswith(CELL_EVALUATION_FAILURE_REASON_PREFIX):
        return None
    suffix = reason[len(CELL_EVALUATION_FAILURE_REASON_PREFIX) :]
    if not suffix:
        raise VelocityAssignmentAggregateError(
            f"{label} must record at least one failed raw cell ID"
        )
    tokens = suffix.split(",")
    if any(
        not token.isascii()
        or not token.isdecimal()
        or token != str(int(token))
        for token in tokens
    ):
        raise VelocityAssignmentAggregateError(
            f"{label} contains a non-canonical raw cell ID"
        )
    ids = tuple(int(token) for token in tokens)
    if ids != tuple(sorted(set(ids))):
        raise VelocityAssignmentAggregateError(
            f"{label} raw cell IDs must be unique and increasing"
        )
    if any(raw_id >= cell_count for raw_id in ids):
        raise VelocityAssignmentAggregateError(
            f"{label} contains an out-of-range raw cell ID"
        )
    return ids


def _validate_profile(path: Path) -> tuple[Any, dict[str, object]]:
    digest = sha256_file(path)
    if digest != EXPECTED_AUTOCFD5_PROFILE_SHA256:
        raise VelocityAssignmentAggregateError(
            "AutoCFD5 profile SHA-256 is not the exact v8 candidate identity"
        )
    try:
        definition = load_autocfd5_definition(path)
    except ValueError as error:
        raise VelocityAssignmentAggregateError(
            f"AutoCFD5 v8 registry validation failed: {error}"
        ) from error
    if sha256_file(path) != digest:
        raise VelocityAssignmentAggregateError(
            "AutoCFD5 profile changed during registry validation"
        )
    binding: dict[str, object] = {
        "profile_sha256": digest,
        "source_registry_sha256": {
            name: source_hash for name, source_hash in definition.source_sha256
        },
        "line_count": 16,
        "fixed_10mm_sample_count": EXPECTED_SAMPLE_COUNTS[10],
    }
    return definition, binding


def _validate_pin(
    path: Path,
    *,
    official_case_ids: tuple[str, ...],
    expected_pin_sha256: str | None,
) -> tuple[Any, dict[str, Any], str]:
    digest = sha256_file(path)
    if expected_pin_sha256 is not None and digest != expected_pin_sha256:
        raise VelocityAssignmentAggregateError(
            "native-source pin SHA-256 mismatch: "
            f"expected {expected_pin_sha256}, got {digest}"
        )
    try:
        pin = load_native_source_pin(path)
    except ValueError as error:
        raise VelocityAssignmentAggregateError(
            f"native-source pin validation failed: {error}"
        ) from error
    if sha256_file(path) != digest:
        raise VelocityAssignmentAggregateError(
            "native-source pin changed during validation"
        )
    if (
        pin.repository_id != EXPECTED_DATASET_REPOSITORY_ID
        or pin.repository_revision != EXPECTED_DATASET_REVISION
    ):
        raise VelocityAssignmentAggregateError(
            "native-source repository identity is not exact"
        )
    actual_case_ids = tuple(case.case_id for case in pin.cases)
    if actual_case_ids != official_case_ids:
        raise VelocityAssignmentAggregateError(
            "native-source pin case order/set differs from the configured public scope"
        )
    by_case = {case.case_id: case for case in pin.cases}
    return pin, by_case, digest


def _expected_samples_by_resolution(definition: Any) -> dict[int, tuple[Any, ...]]:
    result: dict[int, tuple[Any, ...]] = {}
    for spacing_mm, spacing_m, _ in RESOLUTIONS:
        samples = (
            tuple(definition.velocity_samples)
            if spacing_mm == 10
            else generate_definition_velocity_samples(definition, spacing_m)
        )
        if len(samples) != EXPECTED_SAMPLE_COUNTS[spacing_mm]:
            raise VelocityAssignmentAggregateError(
                f"exact {spacing_mm} mm registry grid count is inconsistent"
            )
        result[spacing_mm] = tuple(samples)
    return result


def _expected_query_cache_audit(
    expected_samples: Mapping[int, Sequence[Any]],
) -> dict[str, int | str]:
    """Derive exact cross-resolution cache counts from official sample grids."""

    keys: set[bytes] = set()
    total_rows = 0
    for spacing_mm, _, _ in RESOLUTIONS:
        samples = expected_samples.get(spacing_mm)
        if samples is None or len(samples) != EXPECTED_SAMPLE_COUNTS[spacing_mm]:
            raise VelocityAssignmentAggregateError(
                f"exact {spacing_mm} mm registry samples are unavailable for "
                "query-cache validation"
            )
        for sample in samples:
            point = sample.point_m
            keys.add(
                _QUERY_CACHE_KEY.pack(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(POINT_IN_CELL_CLOSURE_TOLERANCE_M),
                )
            )
        total_rows += len(samples)
    unique_query_keys = len(keys)
    if unique_query_keys < 1 or unique_query_keys > total_rows:
        raise VelocityAssignmentAggregateError(
            "derived containing-cell query-cache counts are invalid"
        )
    return {
        "key_id": QUERY_CACHE_KEY_ID,
        "total_rows": total_rows,
        "unique_query_keys": unique_query_keys,
        "cache_hits": total_rows - unique_query_keys,
    }


def _validate_registry_binding(
    value: object, expected: dict[str, object], label: str
) -> dict[str, object]:
    binding = _exact_keys(
        value,
        {
            "profile_sha256",
            "source_registry_sha256",
            "line_count",
            "fixed_10mm_sample_count",
        },
        label,
    )
    if dict(binding) != expected:
        raise VelocityAssignmentAggregateError(
            f"{label} differs from the exact AutoCFD5 v8 registry binding"
        )
    return dict(binding)


def _validate_kernel(value: object, label: str) -> dict[str, object]:
    kernel = _exact_keys(
        value, {"kernel_id", "settings_sha256", "versions", "settings"}, label
    )
    if kernel["kernel_id"] != KERNEL_ID:
        raise VelocityAssignmentAggregateError(f"{label} kernel_id is not exact")
    if (
        _sha256(kernel["settings_sha256"], f"{label} settings SHA-256")
        != PINNED_KERNEL_SETTINGS_SHA256
    ):
        raise VelocityAssignmentAggregateError(
            f"{label} kernel settings hash is not pinned"
        )
    if kernel["settings"] != PINNED_KERNEL_SETTINGS:
        raise VelocityAssignmentAggregateError(
            f"{label} kernel settings are not exact"
        )
    versions = _exact_keys(kernel["versions"], PINNED_VERSIONS, f"{label} versions")
    if dict(versions) != PINNED_VERSIONS:
        raise VelocityAssignmentAggregateError(
            f"{label} dependency versions are not pinned"
        )
    return {
        "kernel_id": KERNEL_ID,
        "settings_sha256": PINNED_KERNEL_SETTINGS_SHA256,
        "versions": dict(versions),
        "settings": PINNED_KERNEL_SETTINGS,
    }


def _validate_source_binding(
    value: object,
    *,
    case_id: str,
    pinned_case: Any,
    pin_sha256: str,
) -> dict[str, object]:
    binding = _exact_keys(
        value,
        {
            "pin_sha256",
            "repository_id",
            "repository_revision",
            "case_id",
            "logical_size_bytes",
            "ordered_verified_segments",
            "verification",
        },
        f"{case_id} native_source_binding",
    )
    if _sha256(binding["pin_sha256"], f"{case_id} pin SHA-256") != pin_sha256:
        raise VelocityAssignmentAggregateError(
            f"{case_id} receipt is bound to a different native-source pin"
        )
    if (
        binding["repository_id"] != EXPECTED_DATASET_REPOSITORY_ID
        or binding["repository_revision"] != EXPECTED_DATASET_REVISION
        or binding["case_id"] != case_id
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} native-source identity is not exact"
        )
    logical_size = _integer(
        binding["logical_size_bytes"], f"{case_id} logical size", minimum=1
    )
    if logical_size != pinned_case.volume_total_size_bytes:
        raise VelocityAssignmentAggregateError(
            f"{case_id} logical size differs from the pin"
        )
    raw_segments = binding["ordered_verified_segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) != len(
        pinned_case.volume_parts
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} does not bind every ordered source segment"
        )
    segments: list[dict[str, int | str]] = []
    offset = 0
    for part_index, (raw, part) in enumerate(
        zip(raw_segments, pinned_case.volume_parts, strict=True)
    ):
        segment = _exact_keys(
            raw,
            {"part_index", "byte_offset", "size_bytes", "sha256"},
            f"{case_id} source segment {part_index}",
        )
        if (
            _integer(segment["part_index"], "part_index") != part_index
            or _integer(segment["byte_offset"], "byte_offset") != offset
            or _integer(segment["size_bytes"], "size_bytes", minimum=1)
            != part.size_bytes
            or _sha256(segment["sha256"], "segment SHA-256") != part.sha256
        ):
            raise VelocityAssignmentAggregateError(
                f"{case_id} source segment {part_index} differs from the pin"
            )
        segments.append(
            {
                "part_index": part_index,
                "byte_offset": offset,
                "size_bytes": part.size_bytes,
                "sha256": part.sha256,
            }
        )
        offset += part.size_bytes
    verification = _exact_keys(
        binding["verification"],
        {"method", "timing", "vtk_input", "post_vtk_fstat"},
        f"{case_id} verification",
    )
    expected_verification = {
        "method": "exact_ordered_segment_size_and_sha256",
        "timing": "completed_before_vtk_geometry_reader",
        "vtk_input": "retained_verified_file_descriptor",
        "post_vtk_fstat": "unchanged",
    }
    if dict(verification) != expected_verification:
        raise VelocityAssignmentAggregateError(
            f"{case_id} source verification method/timing is not exact"
        )
    return {
        "pin_sha256": pin_sha256,
        "repository_id": EXPECTED_DATASET_REPOSITORY_ID,
        "repository_revision": EXPECTED_DATASET_REVISION,
        "case_id": case_id,
        "logical_size_bytes": logical_size,
        "ordered_verified_segments": segments,
        "verification": expected_verification,
    }


def _validate_reader_audit(value: object, case_id: str) -> dict[str, object]:
    audit = _exact_keys(
        value,
        {
            "disabled_point_array_count",
            "disabled_point_arrays",
            "disabled_cell_array_count",
            "disabled_cell_arrays",
        },
        f"{case_id} reader audit",
    )
    result: dict[str, object] = {}
    for association in ("point", "cell"):
        arrays = audit[f"disabled_{association}_arrays"]
        if (
            not isinstance(arrays, list)
            or any(not isinstance(name, str) or not name for name in arrays)
            or len(arrays) != len(set(arrays))
        ):
            raise VelocityAssignmentAggregateError(
                f"{case_id} disabled {association} arrays must be unique names"
            )
        count = _integer(
            audit[f"disabled_{association}_array_count"],
            f"{case_id} disabled {association} array count",
        )
        if count != len(arrays):
            raise VelocityAssignmentAggregateError(
                f"{case_id} disabled {association} array count is inconsistent"
            )
        result[f"disabled_{association}_array_count"] = count
        result[f"disabled_{association}_arrays"] = list(arrays)
    if not {"pMeanTrim", "UMeanTrim"}.issubset(
        set(result["disabled_cell_arrays"])
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} geometry reader did not disable required native CellData"
        )
    return result


def _validate_geometry(value: object, case_id: str) -> dict[str, object]:
    geometry = _exact_keys(
        value,
        {
            "dataset_type",
            "version",
            "byte_order",
            "header_type",
            "compressor",
            "piece_count",
            "declared_point_count",
            "declared_cell_count",
            "vtk_loaded_point_count",
            "vtk_loaded_cell_count",
            "reader_audit",
            "association",
            "raw_vtk_cell_id",
            "remeshing",
            "reordering",
        },
        f"{case_id} geometry",
    )
    if (
        geometry["dataset_type"] != "UnstructuredGrid"
        or geometry["byte_order"] != "LittleEndian"
        or geometry["header_type"] != "UInt64"
        or geometry["piece_count"] != 1
        or geometry["association"] != "native_volume_CellData"
        or geometry["raw_vtk_cell_id"] != "zero_based_native_GetCell_index"
        or geometry["remeshing"] is not False
        or geometry["reordering"] is not False
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} geometry semantics are not exact"
        )
    _string(geometry["version"], f"{case_id} VTK XML version")
    if geometry["compressor"] is not None:
        _string(geometry["compressor"], f"{case_id} VTK XML compressor")
    point_count = _integer(
        geometry["declared_point_count"], f"{case_id} point count", minimum=1
    )
    cell_count = _integer(
        geometry["declared_cell_count"], f"{case_id} cell count", minimum=1
    )
    if (
        geometry["vtk_loaded_point_count"] != point_count
        or geometry["vtk_loaded_cell_count"] != cell_count
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} declared and VTK-loaded geometry counts differ"
        )
    return {
        **dict(geometry),
        "declared_point_count": point_count,
        "declared_cell_count": cell_count,
        "vtk_loaded_point_count": point_count,
        "vtk_loaded_cell_count": cell_count,
        "reader_audit": _validate_reader_audit(geometry["reader_audit"], case_id),
    }


def _validate_false_claims(value: object, label: str) -> dict[str, bool]:
    claims = _exact_keys(value, FALSE_CASE_CLAIMS, label)
    if dict(claims) != FALSE_CASE_CLAIMS:
        raise VelocityAssignmentAggregateError(
            f"{label} must retain every candidate limitation as false"
        )
    return dict(FALSE_CASE_CLAIMS)


def _validate_histogram(
    value: object, expected: Mapping[str, int], label: str
) -> dict[str, int]:
    histogram = _mapping(value, label)
    result: dict[str, int] = {}
    for key, count in histogram.items():
        _string(key, f"{label} key")
        result[key] = _integer(count, f"{label}[{key!r}]")
    if result != dict(expected):
        raise VelocityAssignmentAggregateError(f"{label} is inconsistent with rows")
    return result


def _validate_artifact(
    path: Path,
    *,
    expected_name: str,
    case_id: str,
    spacing_mm: int,
    spacing_m: float,
    expected_samples: tuple[Any, ...],
    expected_source_hashes: tuple[str, ...],
    registry_binding: dict[str, object],
    source_binding: dict[str, object],
    geometry: dict[str, object],
    kernel: dict[str, object],
    summary: Mapping[str, Any],
) -> dict[str, object]:
    _basename(path.name, "artifact filename", expected_name)
    artifact_size = path.stat().st_size
    artifact_sha256 = sha256_file(path)
    if (
        _integer(summary["size_bytes"], "artifact size", minimum=1)
        != artifact_size
        or _sha256(summary["sha256"], "artifact SHA-256") != artifact_sha256
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm artifact bytes differ from its receipt"
        )
    artifact = _read_json(path, f"{case_id} {spacing_mm} mm artifact")
    _exact_keys(
        artifact,
        {
            "schema",
            "schema_version",
            "status",
            "case_id",
            "resolution",
            "constant_evidence_fields",
            "row_fields",
            "rows",
            "coverage",
            "assignment_evidence_sha256",
            "units",
            "registries",
            "native_source_binding",
            "geometry",
            "kernel",
            "claims",
        },
        f"{case_id} {spacing_mm} mm artifact",
    )
    if (
        artifact["schema"] != ARTIFACT_SCHEMA
        or artifact["schema_version"] != 1
        or artifact["status"] != RECEIPT_STATUS
        or artifact["case_id"] != case_id
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm artifact identity is not exact"
        )
    resolution = _exact_keys(
        artifact["resolution"],
        {
            "nominal_spacing_mm",
            "nominal_spacing_m",
            "sample_source",
            "line_count",
            "sample_count",
        },
        f"{case_id} {spacing_mm} mm resolution",
    )
    expected_source = (
        "expanded_autocfd5_v8_10mm_registry"
        if spacing_mm == 10
        else "endpoint_inclusive_equal_arc_from_v8_line_registry"
    )
    if (
        resolution["nominal_spacing_mm"] != spacing_mm
        or resolution["nominal_spacing_m"] != spacing_m
        or resolution["sample_source"] != expected_source
        or resolution["line_count"] != 16
        or resolution["sample_count"] != len(expected_samples)
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm resolution declaration is not exact"
        )
    constants = _exact_keys(
        artifact["constant_evidence_fields"],
        {"geometric_tolerance_m", "source_sha256"},
        f"{case_id} {spacing_mm} mm constant evidence",
    )
    _same_float(
        constants["geometric_tolerance_m"],
        POINT_IN_CELL_CLOSURE_TOLERANCE_M,
        "geometric_tolerance_m",
    )
    if constants["source_sha256"] != list(expected_source_hashes):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm source hashes are not pinned"
        )
    if artifact["row_fields"] != ROW_FIELDS:
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm row field order is not exact"
        )
    if artifact["units"] != {"point_m": "m", "distance_m": "m"}:
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm units are not exact"
        )
    if (
        artifact["registries"] != registry_binding
        or artifact["native_source_binding"] != source_binding
        or artifact["geometry"] != geometry
        or artifact["kernel"] != kernel
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm artifact bindings differ from its receipt"
        )
    _validate_false_claims(
        artifact["claims"], f"{case_id} {spacing_mm} mm claims"
    )

    rows = artifact["rows"]
    if not isinstance(rows, list) or len(rows) != len(expected_samples):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm rows omit required samples"
        )
    evidence: list[VelocityCellAssignmentEvidence] = []
    valid_count = 0
    invalid_count = 0
    tie_count = 0
    candidate_count_sum = 0
    invalid_reasons: collections.Counter[str] = collections.Counter()
    candidate_histogram: collections.Counter[int] = collections.Counter()
    line_counts: collections.Counter[str] = collections.Counter()
    raw_ids: list[int] = []
    seen: set[tuple[str, int]] = set()
    allowed_invalid_reasons = {NO_CLOSURE_CELL_REASON, *OWNER_INVALID_REASONS}
    cell_count = int(geometry["declared_cell_count"])
    for position, (raw_row, sample) in enumerate(
        zip(rows, expected_samples, strict=True)
    ):
        if not isinstance(raw_row, list) or len(raw_row) != len(ROW_FIELDS):
            raise VelocityAssignmentAggregateError(
                f"{case_id} {spacing_mm} mm row {position} has the wrong shape"
            )
        profile_id = _string(raw_row[0], "profile_id")
        sample_index = _integer(raw_row[1], "sample_index")
        key = (profile_id, sample_index)
        if key in seen:
            raise VelocityAssignmentAggregateError(
                f"{case_id} {spacing_mm} mm rows contain duplicate key {key}"
            )
        seen.add(key)
        if key != (sample.profile_id, sample.sample_index):
            raise VelocityAssignmentAggregateError(
                f"{case_id} {spacing_mm} mm row {position} is out of registry order"
            )
        point = raw_row[2]
        if (
            not isinstance(point, list)
            or len(point) != 3
            or tuple(_finite(item, "point_m") for item in point) != sample.point_m
        ):
            raise VelocityAssignmentAggregateError(
                f"{case_id} {spacing_mm} mm row {position} point differs from grid"
            )
        distance = _finite(raw_row[3], "distance_m")
        if distance != sample.distance_m:
            raise VelocityAssignmentAggregateError(
                f"{case_id} {spacing_mm} mm row {position} distance differs from grid"
            )
        valid = raw_row[4]
        if not isinstance(valid, bool):
            raise VelocityAssignmentAggregateError("valid must be Boolean")
        reason = _string(raw_row[5], "reason", allow_empty=True)
        raw_cell_id = raw_row[6]
        candidate_count = _integer(raw_row[7], "candidate_count")
        if candidate_count > cell_count:
            raise VelocityAssignmentAggregateError(
                f"{case_id} {spacing_mm} mm candidate_count exceeds cell count"
            )
        if valid:
            if (
                reason != ""
                or not isinstance(raw_cell_id, int)
                or isinstance(raw_cell_id, bool)
                or raw_cell_id < 0
                or raw_cell_id >= cell_count
                or candidate_count < 1
            ):
                raise VelocityAssignmentAggregateError(
                    f"{case_id} {spacing_mm} mm valid row is inconsistent"
                )
            valid_count += 1
            raw_ids.append(raw_cell_id)
        else:
            evaluation_failure_ids = _evaluation_failure_ids(
                reason,
                cell_count=cell_count,
                label=f"{case_id} {spacing_mm} mm row {position} reason",
            )
            if (
                reason not in allowed_invalid_reasons
                and evaluation_failure_ids is None
            ) or (
                raw_cell_id is not None
                or (reason == NO_CLOSURE_CELL_REASON and candidate_count != 0)
            ):
                raise VelocityAssignmentAggregateError(
                    f"{case_id} {spacing_mm} mm invalid row is inconsistent"
                )
            if (
                evaluation_failure_ids is not None
                and candidate_count > cell_count - len(evaluation_failure_ids)
            ):
                raise VelocityAssignmentAggregateError(
                    f"{case_id} {spacing_mm} mm invalid row candidate/failure "
                    "counts exceed the native cell count"
                )
            invalid_count += 1
            invalid_reasons[reason] += 1
        line_counts[profile_id] += 1
        candidate_histogram[candidate_count] += 1
        candidate_count_sum += candidate_count
        tie_count += candidate_count > 1
        evidence.append(
            VelocityCellAssignmentEvidence(
                case_id=case_id,
                profile_id=profile_id,
                sample_index=sample_index,
                point_m=tuple(point),
                distance_m=distance,
                valid=valid,
                reason=reason,
                raw_vtk_cell_id=raw_cell_id,
                candidate_count=candidate_count,
                geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                source_sha256=expected_source_hashes,
            )
        )

    evidence_hash = assignment_evidence_sha256(evidence)
    if (
        _sha256(
            artifact["assignment_evidence_sha256"], "assignment evidence SHA-256"
        )
        != evidence_hash
        or _sha256(
            summary["assignment_evidence_sha256"],
            "receipt assignment evidence SHA-256",
        )
        != evidence_hash
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm assignment evidence hash is inconsistent"
        )
    expected_line_counts = {
        profile_id: count for profile_id, count in sorted(line_counts.items())
    }
    expected_reason_counts = {
        reason: count for reason, count in sorted(invalid_reasons.items())
    }
    expected_candidate_histogram = {
        str(count): frequency
        for count, frequency in sorted(candidate_histogram.items())
    }
    coverage = _exact_keys(
        artifact["coverage"],
        {
            "expected_sample_count",
            "actual_sample_count",
            "unique_line_sample_key_count",
            "complete_duplicate_free_no_omissions",
            "line_sample_counts",
            "valid_count",
            "invalid_count",
            "invalid_reason_counts",
            "candidate_count_histogram",
            "selected_raw_vtk_cell_id_min",
            "selected_raw_vtk_cell_id_max",
        },
        f"{case_id} {spacing_mm} mm coverage",
    )
    if (
        coverage["expected_sample_count"] != len(expected_samples)
        or coverage["actual_sample_count"] != len(expected_samples)
        or coverage["unique_line_sample_key_count"] != len(expected_samples)
        or coverage["complete_duplicate_free_no_omissions"] is not True
        or coverage["valid_count"] != valid_count
        or coverage["invalid_count"] != invalid_count
        or coverage["selected_raw_vtk_cell_id_min"]
        != (min(raw_ids) if raw_ids else None)
        or coverage["selected_raw_vtk_cell_id_max"]
        != (max(raw_ids) if raw_ids else None)
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm coverage summary is inconsistent"
        )
    _validate_histogram(
        coverage["line_sample_counts"],
        expected_line_counts,
        f"{case_id} {spacing_mm} mm line counts",
    )
    _validate_histogram(
        coverage["invalid_reason_counts"],
        expected_reason_counts,
        f"{case_id} {spacing_mm} mm invalid reasons",
    )
    _validate_histogram(
        coverage["candidate_count_histogram"],
        expected_candidate_histogram,
        f"{case_id} {spacing_mm} mm candidate histogram",
    )
    if (
        summary["nominal_spacing_mm"] != spacing_mm
        or summary["nominal_spacing_m"] != spacing_m
        or summary["line_count"] != 16
        or summary["sample_count"] != len(expected_samples)
        or summary["valid_count"] != valid_count
        or summary["invalid_count"] != invalid_count
        or summary["invalid_reason_counts"] != expected_reason_counts
        or summary["complete_duplicate_free_no_omissions"] is not True
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} {spacing_mm} mm receipt summary is inconsistent"
        )
    return {
        "nominal_spacing_mm": spacing_mm,
        "nominal_spacing_m": spacing_m,
        "artifact_name": expected_name,
        "artifact_size_bytes": artifact_size,
        "artifact_sha256": artifact_sha256,
        "assignment_evidence_sha256": evidence_hash,
        "line_count": 16,
        "sample_count": len(expected_samples),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_reason_counts": expected_reason_counts,
        "tie_assignment_count": tie_count,
        "candidate_count_sum": candidate_count_sum,
        "selected_raw_vtk_cell_id_min": min(raw_ids) if raw_ids else None,
        "selected_raw_vtk_cell_id_max": max(raw_ids) if raw_ids else None,
        "complete_duplicate_free_no_omissions": True,
    }


def _validate_polyhedron_runtime_audits(
    geometry_cache_value: object,
    evaluation_value: object,
    *,
    case_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    cache = _exact_keys(
        geometry_cache_value,
        {
            "policy",
            "maximum_entries",
            "maximum_emitted_triangles",
            "current_entries",
            "current_emitted_triangles",
            "peak_entries",
            "peak_emitted_triangles",
            "cache_hits",
            "cache_misses",
            "evictions",
            "oversized_entry_bypasses",
            "fail_closed_preparations",
            "vtk_objects_cached",
        },
        f"{case_id} polyhedron geometry-cache audit",
    )
    maximum_entries = _integer(
        cache["maximum_entries"], "maximum_entries", minimum=1
    )
    maximum_triangles = _integer(
        cache["maximum_emitted_triangles"],
        "maximum_emitted_triangles",
        minimum=1,
    )
    current_entries = _integer(
        cache["current_entries"], "current_entries", minimum=0
    )
    current_triangles = _integer(
        cache["current_emitted_triangles"],
        "current_emitted_triangles",
        minimum=0,
    )
    peak_entries = _integer(cache["peak_entries"], "peak_entries", minimum=0)
    peak_triangles = _integer(
        cache["peak_emitted_triangles"],
        "peak_emitted_triangles",
        minimum=0,
    )
    cache_hits = _integer(cache["cache_hits"], "cache_hits", minimum=0)
    cache_misses = _integer(cache["cache_misses"], "cache_misses", minimum=0)
    evictions = _integer(cache["evictions"], "evictions", minimum=0)
    oversized_bypasses = _integer(
        cache["oversized_entry_bypasses"],
        "oversized_entry_bypasses",
        minimum=0,
    )
    preparation_failures = _integer(
        cache["fail_closed_preparations"],
        "fail_closed_preparations",
        minimum=0,
    )
    if (
        cache["policy"] != "deterministic_least_recently_used"
        or cache["vtk_objects_cached"] is not False
        or maximum_entries != POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES
        or maximum_triangles != POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES
        or current_entries > maximum_entries
        or peak_entries < current_entries
        or peak_entries > maximum_entries
        or current_triangles > maximum_triangles
        or peak_triangles < current_triangles
        or peak_triangles > maximum_triangles
        or current_entries
        != cache_misses - oversized_bypasses - evictions
        or preparation_failures > cache_misses
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} polyhedron geometry-cache audit is inconsistent"
        )
    cache_audit = {
        "policy": "deterministic_least_recently_used",
        "maximum_entries": maximum_entries,
        "maximum_emitted_triangles": maximum_triangles,
        "current_entries": current_entries,
        "current_emitted_triangles": current_triangles,
        "peak_entries": peak_entries,
        "peak_emitted_triangles": peak_triangles,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "evictions": evictions,
        "oversized_entry_bypasses": oversized_bypasses,
        "fail_closed_preparations": preparation_failures,
        "vtk_objects_cached": False,
    }

    evaluation = _exact_keys(
        evaluation_value,
        {
            "scope",
            "broad_phase_polyhedron_visit_count",
            "boundary_count",
            "inside_count",
            "outside_count",
            "ambiguous_count",
            "winding_classified_count",
            "minimum_winding_classification_margin_steradian",
            "classification_absolute_tolerance_steradian",
        },
        f"{case_id} polyhedron evaluation audit",
    )
    visit_count = _integer(
        evaluation["broad_phase_polyhedron_visit_count"],
        "broad_phase_polyhedron_visit_count",
        minimum=0,
    )
    boundary_count = _integer(
        evaluation["boundary_count"], "boundary_count", minimum=0
    )
    inside_count = _integer(
        evaluation["inside_count"], "inside_count", minimum=0
    )
    outside_count = _integer(
        evaluation["outside_count"], "outside_count", minimum=0
    )
    ambiguous_count = _integer(
        evaluation["ambiguous_count"], "ambiguous_count", minimum=0
    )
    classified_count = _integer(
        evaluation["winding_classified_count"],
        "winding_classified_count",
        minimum=0,
    )
    _same_float(
        evaluation["classification_absolute_tolerance_steradian"],
        POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE,
        "classification_absolute_tolerance_steradian",
    )
    margin_value = evaluation[
        "minimum_winding_classification_margin_steradian"
    ]
    if margin_value is None:
        minimum_margin = None
    elif isinstance(margin_value, (int, float)) and not isinstance(
        margin_value, bool
    ):
        minimum_margin = float(margin_value)
    else:
        raise VelocityAssignmentAggregateError(
            f"{case_id} polyhedron classification margin must be numeric or null"
        )
    if (
        evaluation["scope"]
        != "broad_phase_polyhedron_visits_for_uncached_exact_xyz_tolerance_queries"
        or visit_count
        != boundary_count + inside_count + outside_count + ambiguous_count
        or visit_count != cache_hits + cache_misses
        or classified_count != inside_count + outside_count
        or ambiguous_count < preparation_failures
        or (classified_count == 0) != (minimum_margin is None)
        or (
            minimum_margin is not None
            and (
                not math.isfinite(minimum_margin)
                or minimum_margin < 0.0
                or minimum_margin
                > POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
            )
        )
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} polyhedron evaluation audit is inconsistent"
        )
    evaluation_audit = {
        "scope": evaluation["scope"],
        "broad_phase_polyhedron_visit_count": visit_count,
        "boundary_count": boundary_count,
        "inside_count": inside_count,
        "outside_count": outside_count,
        "ambiguous_count": ambiguous_count,
        "winding_classified_count": classified_count,
        "minimum_winding_classification_margin_steradian": minimum_margin,
        "classification_absolute_tolerance_steradian": (
            POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
        ),
    }
    return cache_audit, evaluation_audit


def _validate_case_receipt(
    case_directory: Path,
    *,
    case_id: str,
    pinned_case: Any,
    pin_sha256: str,
    registry_binding: dict[str, object],
    expected_samples: dict[int, tuple[Any, ...]],
    expected_query_cache_audit: Mapping[str, int | str],
    allow_missing_query_cache_audit: bool,
) -> dict[str, object]:
    if case_directory.is_symlink() or not case_directory.is_dir():
        raise VelocityAssignmentAggregateError(
            f"{case_id} receipt directory must be a real directory"
        )
    receipt_path = case_directory / "receipt.json"
    receipt_sha256 = sha256_file(receipt_path)
    receipt = _read_json(receipt_path, f"{case_id} receipt")
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "status",
            "case_id",
            "registries",
            "native_source_binding",
            "geometry",
            "kernel",
            "execution",
            "artifacts",
            "coverage",
            "claims",
        },
        f"{case_id} receipt",
    )
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["schema_version"] != 1
        or receipt["status"] != RECEIPT_STATUS
        or receipt["case_id"] != case_id
    ):
        raise VelocityAssignmentAggregateError(f"{case_id} receipt identity is not exact")
    registries = _validate_registry_binding(
        receipt["registries"], registry_binding, f"{case_id} registries"
    )
    source = _validate_source_binding(
        receipt["native_source_binding"],
        case_id=case_id,
        pinned_case=pinned_case,
        pin_sha256=pin_sha256,
    )
    geometry = _validate_geometry(receipt["geometry"], case_id)
    kernel = _validate_kernel(receipt["kernel"], f"{case_id} kernel")
    _validate_false_claims(receipt["claims"], f"{case_id} claims")
    raw_execution = _mapping(receipt["execution"], f"{case_id} execution")
    query_cache_present = "containing_cell_query_cache" in raw_execution
    if not query_cache_present and not allow_missing_query_cache_audit:
        raise VelocityAssignmentAggregateError(
            f"{case_id} complete-mode receipt is missing the containing-cell "
            "query-cache audit"
        )
    execution_keys = {
        "io_chunk_bytes",
        "validation_chunk_cells",
        "resolution_order_mm",
        "geometric_tolerance_m",
        "polyhedron_geometry_cache",
        "polyhedron_evaluation",
    }
    if query_cache_present:
        execution_keys.add("containing_cell_query_cache")
    execution = _exact_keys(
        receipt["execution"], execution_keys, f"{case_id} execution"
    )
    _integer(execution["io_chunk_bytes"], "io_chunk_bytes", minimum=1)
    _integer(execution["validation_chunk_cells"], "validation_chunk_cells", minimum=1)
    if execution["resolution_order_mm"] != [1, 2, 5, 10]:
        raise VelocityAssignmentAggregateError(
            f"{case_id} resolution order is not exact"
        )
    _same_float(
        execution["geometric_tolerance_m"],
        POINT_IN_CELL_CLOSURE_TOLERANCE_M,
        "geometric_tolerance_m",
    )
    query_cache_audit: dict[str, object] | None = None
    if query_cache_present:
        audit = _exact_keys(
            execution["containing_cell_query_cache"],
            {"enabled", "key_id", "total_rows", "unique_query_keys", "cache_hits"},
            f"{case_id} containing-cell query-cache audit",
        )
        total_rows = _integer(audit["total_rows"], "total_rows", minimum=1)
        unique_keys = _integer(
            audit["unique_query_keys"], "unique_query_keys", minimum=1
        )
        cache_hits = _integer(audit["cache_hits"], "cache_hits", minimum=0)
        if (
            audit["enabled"] is not True
            or audit["key_id"] != QUERY_CACHE_KEY_ID
            or total_rows != expected_query_cache_audit["total_rows"]
            or unique_keys != expected_query_cache_audit["unique_query_keys"]
            or cache_hits != expected_query_cache_audit["cache_hits"]
        ):
            raise VelocityAssignmentAggregateError(
                f"{case_id} containing-cell query-cache audit differs from the "
                "exact registry-derived query keys"
            )
        query_cache_audit = {
            "enabled": True,
            "key_id": QUERY_CACHE_KEY_ID,
            "total_rows": total_rows,
            "unique_query_keys": unique_keys,
            "cache_hits": cache_hits,
        }
    polyhedron_cache_audit, polyhedron_evaluation_audit = (
        _validate_polyhedron_runtime_audits(
            execution["polyhedron_geometry_cache"],
            execution["polyhedron_evaluation"],
            case_id=case_id,
        )
    )
    coverage = _exact_keys(
        receipt["coverage"],
        {
            "resolution_count",
            "all_sixteen_lines_each_resolution",
            "all_samples_explicit_no_silent_omissions",
            "expected_sample_counts",
        },
        f"{case_id} receipt coverage",
    )
    exact_counts = {
        str(key): value for key, value in sorted(EXPECTED_SAMPLE_COUNTS.items())
    }
    if (
        coverage["resolution_count"] != 4
        or coverage["all_sixteen_lines_each_resolution"] is not True
        or coverage["all_samples_explicit_no_silent_omissions"] is not True
        or coverage["expected_sample_counts"] != exact_counts
    ):
        raise VelocityAssignmentAggregateError(
            f"{case_id} receipt coverage declaration is not exact"
        )
    raw_summaries = receipt["artifacts"]
    if not isinstance(raw_summaries, list) or len(raw_summaries) != 4:
        raise VelocityAssignmentAggregateError(
            f"{case_id} receipt must contain four artifact summaries"
        )
    expected_artifact_names = {
        f"velocity-cell-mapping-{label}.json" for _, _, label in RESOLUTIONS
    }
    actual_mapping_names = {
        path.name for path in case_directory.glob("velocity-cell-mapping-*.json")
    }
    if actual_mapping_names != expected_artifact_names:
        raise VelocityAssignmentAggregateError(
            f"{case_id} mapping artifact file set is not exact"
        )
    resolution_rows: list[dict[str, object]] = []
    for summary, (spacing_mm, spacing_m, label) in zip(
        raw_summaries, RESOLUTIONS, strict=True
    ):
        summary_mapping = _exact_keys(
            summary,
            {
                "nominal_spacing_mm",
                "nominal_spacing_m",
                "artifact",
                "size_bytes",
                "sha256",
                "assignment_evidence_sha256",
                "line_count",
                "sample_count",
                "valid_count",
                "invalid_count",
                "invalid_reason_counts",
                "complete_duplicate_free_no_omissions",
            },
            f"{case_id} {spacing_mm} mm receipt summary",
        )
        artifact_name = f"velocity-cell-mapping-{label}.json"
        _basename(summary_mapping["artifact"], "artifact", artifact_name)
        resolution_rows.append(
            _validate_artifact(
                case_directory / artifact_name,
                expected_name=artifact_name,
                case_id=case_id,
                spacing_mm=spacing_mm,
                spacing_m=spacing_m,
                expected_samples=expected_samples[spacing_mm],
                expected_source_hashes=tuple(
                    part.sha256 for part in pinned_case.volume_parts
                ),
                registry_binding=registries,
                source_binding=source,
                geometry=geometry,
                kernel=kernel,
                summary=summary_mapping,
            )
        )
    return {
        "case_id": case_id,
        "receipt_sha256": receipt_sha256,
        "geometry": {
            "point_count": geometry["declared_point_count"],
            "cell_count": geometry["declared_cell_count"],
            "ordered_verified_segment_count": len(pinned_case.volume_parts),
        },
        "containing_cell_query_cache": query_cache_audit,
        "polyhedron_geometry_cache": polyhedron_cache_audit,
        "polyhedron_evaluation": polyhedron_evaluation_audit,
        "resolutions": resolution_rows,
    }


def _discover_case_directories(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise VelocityAssignmentAggregateError(
            "receipt root must be an existing directory"
        )
    result: dict[str, Path] = {}
    for receipt_path in root.rglob("receipt.json"):
        parent = receipt_path.parent
        if parent.parent.resolve() != root.resolve():
            raise VelocityAssignmentAggregateError(
                "receipt.json files must be direct children of run_N directories"
            )
        case_id = parent.name
        _case_number(case_id)
        if case_id in result:
            raise VelocityAssignmentAggregateError(
                f"duplicate receipt directory for {case_id}"
            )
        result[case_id] = parent
    return result


def aggregate_velocity_assignments(
    *,
    receipts_root: Path | str,
    native_source_pin: Path | str = DEFAULT_NATIVE_SOURCE_PIN,
    autocfd5_profile: Path | str = DEFAULT_AUTOCFD5_PROFILE,
    pilot_case_ids: Sequence[str] = (),
    official_case_ids: Sequence[str] = OFFICIAL_CASE_IDS,
    expected_pin_sha256: str | None = OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
) -> dict[str, object]:
    """Validate case artifacts and return a compact path-free manifest."""

    configured_full_cases = _ordered_case_ids(
        tuple(official_case_ids), "configured public case IDs"
    )
    pilot = _ordered_case_ids(tuple(pilot_case_ids), "pilot case IDs") if pilot_case_ids else ()
    if not pilot and configured_full_cases != OFFICIAL_CASE_IDS:
        raise VelocityAssignmentAggregateError(
            "complete mode requires the exact official 484-case scope; "
            "use pilot_case_ids for any reduced or custom scope"
        )
    if not pilot and expected_pin_sha256 != OFFICIAL_NATIVE_SOURCE_PIN_SHA256:
        raise VelocityAssignmentAggregateError(
            "complete mode requires the immutable official native-source pin hash"
        )
    if pilot and any(case_id not in configured_full_cases for case_id in pilot):
        raise VelocityAssignmentAggregateError(
            "pilot cases must belong to the configured public case set"
        )
    expected_cases = pilot or configured_full_cases
    mode = "explicit_non_public_pilot" if pilot else "complete_484_case_default"

    definition, registry_binding = _validate_profile(
        Path(autocfd5_profile).expanduser().resolve()
    )
    _, pinned_cases, pin_sha256 = _validate_pin(
        Path(native_source_pin).expanduser().resolve(),
        official_case_ids=configured_full_cases,
        expected_pin_sha256=expected_pin_sha256,
    )
    expected_samples = _expected_samples_by_resolution(definition)
    expected_query_cache_audit = _expected_query_cache_audit(expected_samples)
    discovered = _discover_case_directories(
        Path(receipts_root).expanduser().resolve()
    )
    if set(discovered) != set(expected_cases):
        missing = sorted(set(expected_cases) - set(discovered), key=_case_number)
        unexpected = sorted(set(discovered) - set(expected_cases), key=_case_number)
        required = "explicit pilot" if pilot else "all 484 public"
        raise VelocityAssignmentAggregateError(
            f"{required} receipt directories are not exact "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    cases: list[dict[str, object]] = []
    totals_by_resolution: dict[int, dict[str, int]] = {
        spacing_mm: {
            "case_count": 0,
            "sample_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "tie_assignment_count": 0,
            "candidate_count_sum": 0,
        }
        for spacing_mm, _, _ in RESOLUTIONS
    }
    total_points = 0
    total_cells = 0
    total_segments = 0
    cache_audit_receipt_count = 0
    cache_audit_total_rows = 0
    cache_audit_unique_query_keys = 0
    cache_audit_hits = 0
    polyhedron_cache_totals: collections.Counter[str] = collections.Counter()
    polyhedron_cache_peak_entries_max = 0
    polyhedron_cache_peak_triangles_max = 0
    polyhedron_evaluation_totals: collections.Counter[str] = collections.Counter()
    polyhedron_minimum_winding_margin: float | None = None
    all_invalid_reasons: collections.Counter[str] = collections.Counter()
    for case_id in expected_cases:
        case = _validate_case_receipt(
            discovered[case_id],
            case_id=case_id,
            pinned_case=pinned_cases[case_id],
            pin_sha256=pin_sha256,
            registry_binding=registry_binding,
            expected_samples=expected_samples,
            expected_query_cache_audit=expected_query_cache_audit,
            allow_missing_query_cache_audit=bool(pilot),
        )
        cases.append(case)
        geometry = case["geometry"]
        total_points += int(geometry["point_count"])
        total_cells += int(geometry["cell_count"])
        total_segments += int(geometry["ordered_verified_segment_count"])
        cache_audit = case["containing_cell_query_cache"]
        if cache_audit is not None:
            cache_audit_receipt_count += 1
            cache_audit_total_rows += int(cache_audit["total_rows"])
            cache_audit_unique_query_keys += int(cache_audit["unique_query_keys"])
            cache_audit_hits += int(cache_audit["cache_hits"])
        polyhedron_cache = case["polyhedron_geometry_cache"]
        for key in (
            "current_entries",
            "current_emitted_triangles",
            "cache_hits",
            "cache_misses",
            "evictions",
            "oversized_entry_bypasses",
            "fail_closed_preparations",
        ):
            polyhedron_cache_totals[key] += int(polyhedron_cache[key])
        polyhedron_cache_peak_entries_max = max(
            polyhedron_cache_peak_entries_max,
            int(polyhedron_cache["peak_entries"]),
        )
        polyhedron_cache_peak_triangles_max = max(
            polyhedron_cache_peak_triangles_max,
            int(polyhedron_cache["peak_emitted_triangles"]),
        )
        polyhedron_evaluation = case["polyhedron_evaluation"]
        for key in (
            "broad_phase_polyhedron_visit_count",
            "boundary_count",
            "inside_count",
            "outside_count",
            "ambiguous_count",
            "winding_classified_count",
        ):
            polyhedron_evaluation_totals[key] += int(polyhedron_evaluation[key])
        case_margin = polyhedron_evaluation[
            "minimum_winding_classification_margin_steradian"
        ]
        if case_margin is not None and (
            polyhedron_minimum_winding_margin is None
            or float(case_margin) < polyhedron_minimum_winding_margin
        ):
            polyhedron_minimum_winding_margin = float(case_margin)
        for resolution in case["resolutions"]:
            spacing_mm = int(resolution["nominal_spacing_mm"])
            totals = totals_by_resolution[spacing_mm]
            totals["case_count"] += 1
            for key in (
                "sample_count",
                "valid_count",
                "invalid_count",
                "tie_assignment_count",
                "candidate_count_sum",
            ):
                totals[key] += int(resolution[key])
            all_invalid_reasons.update(resolution["invalid_reason_counts"])

    complete_mode = not pilot
    status = (
        "complete_484_case_candidate_mapping_manifest_not_activation_evidence"
        if complete_mode
        else "incomplete_explicit_pilot_not_public_or_activation_evidence"
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "case_scope": {
            "case_count": len(cases),
            "complete_484_public_case_set": complete_mode,
            "explicit_pilot_case_ids": list(pilot),
        },
        "source_bindings": {
            "native_source_pin_sha256": pin_sha256,
            "repository_id": EXPECTED_DATASET_REPOSITORY_ID,
            "repository_revision": EXPECTED_DATASET_REVISION,
            "autocfd5": registry_binding,
        },
        "kernel": {
            "kernel_id": KERNEL_ID,
            "settings_sha256": PINNED_KERNEL_SETTINGS_SHA256,
            "versions": PINNED_VERSIONS,
            "settings": PINNED_KERNEL_SETTINGS,
        },
        "tolerance_evidence": {
            "primary_geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
            "kernel_required_replay_m": list(TOLERANCE_REPLAY_M),
            "requested_case_primary_tolerance_rows_complete": True,
            "all_484_primary_tolerance_rows_complete": complete_mode,
            "half_one_two_micrometre_replay_complete": False,
            "non_face_assignment_invariance_claim": False,
            "ranked_result_invariance_claim": False,
        },
        "cases": cases,
        "totals": {
            "case_count": len(cases),
            "geometry_point_count_sum": total_points,
            "geometry_cell_count_sum": total_cells,
            "ordered_verified_segment_count": total_segments,
            "containing_cell_query_cache": {
                "key_id": QUERY_CACHE_KEY_ID,
                "expected_per_case": {
                    "total_rows": expected_query_cache_audit["total_rows"],
                    "unique_query_keys": expected_query_cache_audit[
                        "unique_query_keys"
                    ],
                    "cache_hits": expected_query_cache_audit["cache_hits"],
                },
                "audited_receipt_count": cache_audit_receipt_count,
                "missing_pre_cache_pilot_receipt_count": (
                    len(cases) - cache_audit_receipt_count
                ),
                "total_rows": cache_audit_total_rows,
                "unique_query_keys_sum": cache_audit_unique_query_keys,
                "cache_hits": cache_audit_hits,
            },
            "polyhedron_geometry_cache": {
                "audited_receipt_count": len(cases),
                "policy": "deterministic_least_recently_used",
                "maximum_entries_per_case": (
                    POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES
                ),
                "maximum_emitted_triangles_per_case": (
                    POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES
                ),
                "peak_entries_max": polyhedron_cache_peak_entries_max,
                "peak_emitted_triangles_max": (
                    polyhedron_cache_peak_triangles_max
                ),
                **dict(polyhedron_cache_totals),
            },
            "polyhedron_evaluation": {
                "audited_receipt_count": len(cases),
                "classification_absolute_tolerance_steradian": (
                    POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
                ),
                **dict(polyhedron_evaluation_totals),
                "minimum_winding_classification_margin_steradian": (
                    polyhedron_minimum_winding_margin
                ),
            },
            "explicit_assignment_row_count": sum(
                values["sample_count"] for values in totals_by_resolution.values()
            ),
            "invalid_reason_counts": {
                reason: count
                for reason, count in sorted(all_invalid_reasons.items())
            },
            "by_resolution_mm": {
                str(spacing_mm): totals_by_resolution[spacing_mm]
                for spacing_mm, _, _ in RESOLUTIONS
            },
        },
        "claims": MANIFEST_FALSE_CLAIMS,
    }


def write_manifest(path: Path | str, evidence: dict[str, object]) -> dict[str, object]:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise VelocityAssignmentAggregateError("manifest output must use .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly aggregate all 484 DrivAerML velocity assignment receipts. "
            "Use --pilot-case explicitly for non-public partial evidence."
        )
    )
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument(
        "--native-source-pin", type=Path, default=DEFAULT_NATIVE_SOURCE_PIN
    )
    parser.add_argument(
        "--autocfd5-profile", type=Path, default=DEFAULT_AUTOCFD5_PROFILE
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-case",
        action="append",
        default=[],
        help="explicit run_N to aggregate as incomplete non-public pilot evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        evidence = aggregate_velocity_assignments(
            receipts_root=args.receipts_root,
            native_source_pin=args.native_source_pin,
            autocfd5_profile=args.autocfd5_profile,
            pilot_case_ids=args.pilot_case,
            official_case_ids=OFFICIAL_CASE_IDS,
            expected_pin_sha256=OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
        )
        output = write_manifest(args.output, evidence)
    except VelocityAssignmentAggregateError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "case_count": evidence["case_scope"]["case_count"],
                "manifest_size_bytes": output["size_bytes"],
                "manifest_sha256": output["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
