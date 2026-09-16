#!/usr/bin/env python3
"""Aggregate strict equal-native-cell DrivAerML volume-audit evidence.

The all-case mode covers every case in the immutable native-source pin.  The
pilot mode requires an explicit, ordered subset.  Both modes validate exactly
one unit weight per native volume cell; geometric cell-volume weights are not
part of this scoring contract.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.retained_file import (  # noqa: E402
    RetainedFileError,
    RetainedVerifiedFile,
)
from reference.drivaerml.source import (  # noqa: E402
    NativeCaseRecord,
    NativeSourceError,
    NativeSourcePin,
    load_native_source_pin,
)


DEFAULT_NATIVE_SOURCE_PIN = (
    ROOT / "benchmark-specs" / "drivaerml" / "proposal" / "native-source-pin.json"
)
CASE_AUDIT_SCHEMA = "drivaerml-native-volume-case-audit-v2"
EQUAL_CELL_PILOT_AGGREGATE_SCHEMA = (
    "drivaerml-native-volume-equal-cell-pilot-aggregate-v2"
)
EQUAL_CELL_ALL_CASE_AUDIT_SCHEMA = (
    "drivaerml-native-volume-equal-cell-all-case-audit-v2"
)
REFERENCE_CHUNK_CELLS = 1_000_003
COMPARISON_CHUNK_CELLS = 777_779
MAX_ADDITIVE_RELATIVE_DIFFERENCE = 2.0e-12
MAX_METRIC_ABSOLUTE_DIFFERENCE = 2.0e-12
ZERO_PREDICTION_ROLE = "all_zero_invariance_fixture_not_a_published_baseline"

_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")


class NativeVolumeAuditAggregateError(ValueError):
    """Raised when case-audit evidence is incomplete or inconsistent."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeVolumeAuditAggregateError(
                f"JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _read_json_with_sha256(
    path: Path,
    label: str,
) -> tuple[dict[str, Any], str]:
    """Parse and hash exactly one retained byte read of a JSON receipt."""

    unresolved = Path(path).expanduser()
    try:
        with RetainedVerifiedFile.open(unresolved, label=label) as retained:
            payload = retained.handle.read()
            digest = hashlib.sha256(payload).hexdigest()
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
            retained.assert_unchanged(
                context="while its bytes were read and parsed"
            )
    except NativeVolumeAuditAggregateError:
        raise
    except RetainedFileError as error:
        raise NativeVolumeAuditAggregateError(str(error)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeVolumeAuditAggregateError(
            f"cannot read valid {label} JSON: {unresolved}"
        ) from error
    if not isinstance(value, dict):
        raise NativeVolumeAuditAggregateError(f"{label} must be a JSON object")
    return value, digest


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeVolumeAuditAggregateError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object,
    expected: Iterable[str],
    label: str,
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result))
        unexpected = sorted(set(result) - expected_set)
        raise NativeVolumeAuditAggregateError(
            f"{label} keys differ from schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeVolumeAuditAggregateError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise NativeVolumeAuditAggregateError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NativeVolumeAuditAggregateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NativeVolumeAuditAggregateError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise NativeVolumeAuditAggregateError(f"{label} must be non-negative")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise NativeVolumeAuditAggregateError(f"{label} must be strictly positive")
    return result


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise NativeVolumeAuditAggregateError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return result


def _case_number(case_id: str) -> int:
    match = _CASE_RE.fullmatch(case_id)
    if match is None:
        raise NativeVolumeAuditAggregateError(
            f"invalid canonical DrivAerML case ID {case_id!r}"
        )
    return int(match.group(1))


def _relative_public_path(value: object, label: str) -> str:
    result = _string(value, label).replace("\\", "/")
    if (
        result.startswith("/")
        or _WINDOWS_ABSOLUTE_RE.match(result)
        or any(part in {"", ".", ".."} for part in result.split("/"))
    ):
        raise NativeVolumeAuditAggregateError(
            f"{label} must be a normalized relative public path"
        )
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2.0e-12, abs_tol=1.0e-30)


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
            raise NativeVolumeAuditAggregateError(
                f"{label} contains an absolute path"
            )


def _selected_cases(
    pin: NativeSourcePin,
    selected_case_ids: Sequence[str] | None,
) -> tuple[NativeCaseRecord, ...]:
    if selected_case_ids is None:
        return pin.cases
    if not selected_case_ids:
        raise NativeVolumeAuditAggregateError("selected case IDs cannot be empty")
    identifiers = tuple(
        _string(case_id, "selected case ID") for case_id in selected_case_ids
    )
    if len(identifiers) != len(set(identifiers)):
        raise NativeVolumeAuditAggregateError("selected case IDs must be unique")
    for case_id in identifiers:
        _case_number(case_id)
    pin_order = {case.case_id: index for index, case in enumerate(pin.cases)}
    unknown = sorted(set(identifiers) - set(pin_order), key=_case_number)
    if unknown:
        raise NativeVolumeAuditAggregateError(
            f"selected cases are absent from the native-source pin: {unknown}"
        )
    if tuple(sorted(identifiers, key=pin_order.__getitem__)) != identifiers:
        raise NativeVolumeAuditAggregateError(
            "selected cases must follow native-source pin order"
        )
    return tuple(pin.case(case_id) for case_id in identifiers)


def _source_segments(
    value: object,
    *,
    case: NativeCaseRecord,
) -> list[dict[str, int | str]]:
    if not isinstance(value, list) or len(value) != len(case.volume_parts):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} must verify every pinned monolithic segment"
        )
    result: list[dict[str, int | str]] = []
    expected_offset = 0
    physical_path: str | None = None
    for part_index, (raw, part) in enumerate(
        zip(value, case.volume_parts, strict=True)
    ):
        row = _exact_keys(
            raw,
            {"label", "path", "file_offset", "size_bytes", "sha256"},
            f"{case.case_id} verified segment {part_index}",
        )
        if row["label"] != f"{case.case_id}:monolithic-segment:{part_index}":
            raise NativeVolumeAuditAggregateError(
                f"{case.case_id} verified segment label/order is not exact"
            )
        path = _string(row["path"], f"{case.case_id} verified physical path")
        if physical_path is None:
            physical_path = path
        elif path != physical_path:
            raise NativeVolumeAuditAggregateError(
                f"{case.case_id} verified segments do not use one monolithic source"
            )
        offset = _integer(
            row["file_offset"],
            f"{case.case_id} verified segment {part_index} offset",
        )
        size = _integer(
            row["size_bytes"],
            f"{case.case_id} verified segment {part_index} size",
            minimum=1,
        )
        digest = _sha256(
            row["sha256"],
            f"{case.case_id} verified segment {part_index} SHA-256",
        )
        if (
            offset != expected_offset
            or size != part.size_bytes
            or digest != part.sha256
        ):
            raise NativeVolumeAuditAggregateError(
                f"{case.case_id} verified segment {part_index} is not pinned"
            )
        result.append(
            {"part_index": part_index, "size_bytes": size, "sha256": digest}
        )
        expected_offset += size
    return result


def _validate_public_source(
    value: object,
    *,
    pin: NativeSourcePin,
    case: NativeCaseRecord,
) -> dict[str, object]:
    source = _exact_keys(
        value,
        {
            "repository_id",
            "immutable_revision",
            "logical_path",
            "logical_size_bytes",
            "multipart_part_count",
            "verified_monolithic_segments",
        },
        f"{case.case_id} public_source",
    )
    if (
        source["repository_id"] != pin.repository_id
        or source["immutable_revision"] != pin.repository_revision
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} public repository identity is not pinned"
        )
    logical_path = _relative_public_path(
        source["logical_path"], f"{case.case_id} logical path"
    )
    if logical_path != case.volume_logical_path.as_posix():
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} logical volume path is not pinned"
        )
    logical_size = _integer(
        source["logical_size_bytes"],
        f"{case.case_id} logical volume size",
        minimum=1,
    )
    part_count = _integer(
        source["multipart_part_count"],
        f"{case.case_id} multipart part count",
        minimum=1,
    )
    if logical_size != case.volume_total_size_bytes or part_count != len(
        case.volume_parts
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} logical size/part count is not pinned"
        )
    return {
        "repository_id": pin.repository_id,
        "immutable_revision": pin.repository_revision,
        "logical_path": logical_path,
        "logical_size_bytes": logical_size,
        "multipart_part_count": part_count,
        "ordered_verified_segments": _source_segments(
            source["verified_monolithic_segments"], case=case
        ),
    }


def _validate_piece(value: object, case_id: str) -> dict[str, int]:
    piece = _exact_keys(
        value,
        {
            "piece_index",
            "number_of_points",
            "number_of_cells",
            "number_of_verts",
            "number_of_lines",
            "number_of_strips",
            "number_of_polys",
        },
        f"{case_id} VTK Piece",
    )
    normalized = {
        name: _integer(
            piece[name],
            f"{case_id} VTK Piece {name}",
            minimum=1 if name in {"number_of_points", "number_of_cells"} else 0,
        )
        for name in piece
    }
    if normalized["piece_index"] != 0:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} native volume must use VTK Piece index zero"
        )
    if any(
        normalized[name] != 0
        for name in (
            "number_of_verts",
            "number_of_lines",
            "number_of_strips",
            "number_of_polys",
        )
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} UnstructuredGrid Piece has invalid PolyData counts"
        )
    return normalized


def _validate_data_array_index(
    value: object,
    *,
    case_id: str,
    position: int,
    logical_size_bytes: int,
) -> dict[str, Any]:
    row = _exact_keys(
        value,
        {
            "array_index",
            "piece_index",
            "association",
            "name",
            "vtk_type",
            "number_of_components",
            "format",
            "opening_tag_start",
            "encoded_start",
            "encoded_end",
            "closing_tag_end",
        },
        f"{case_id} VTK DataArray index {position}",
    )
    if _integer(row["array_index"], f"{case_id} array_index") != position:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} VTK DataArray indices are not in raw XML order"
        )
    piece_index = _integer(
        row["piece_index"], f"{case_id} DataArray piece_index", minimum=-1
    )
    association = _string(row["association"], f"{case_id} DataArray association")
    name = row["name"]
    if name is not None and (not isinstance(name, str) or not name):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} DataArray name must be null or a non-empty string"
        )
    vtk_type = _string(row["vtk_type"], f"{case_id} DataArray VTK type")
    components = _integer(
        row["number_of_components"],
        f"{case_id} DataArray components",
        minimum=1,
    )
    if row["format"] != "binary":
        raise NativeVolumeAuditAggregateError(
            f"{case_id} native DataArrays must use inline binary format"
        )
    offsets = [
        _integer(row[key], f"{case_id} DataArray {key}")
        for key in (
            "opening_tag_start",
            "encoded_start",
            "encoded_end",
            "closing_tag_end",
        )
    ]
    if not (
        offsets[0] < offsets[1] <= offsets[2] < offsets[3] <= logical_size_bytes
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} VTK DataArray byte offsets are invalid"
        )
    return {
        "array_index": position,
        "piece_index": piece_index,
        "association": association,
        "name": name,
        "vtk_type": vtk_type,
        "number_of_components": components,
    }


def _validate_vtk(
    value: object,
    *,
    case_id: str,
    logical_size_bytes: int,
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    vtk = _exact_keys(
        value,
        {
            "dataset_type",
            "version",
            "byte_order",
            "header_type",
            "compressor",
            "piece",
            "data_arrays_in_raw_xml_order",
        },
        f"{case_id} vtk",
    )
    if (
        vtk["dataset_type"] != "UnstructuredGrid"
        or vtk["byte_order"] != "LittleEndian"
        or vtk["header_type"] != "UInt64"
        or vtk["compressor"] is not None
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} VTK container metadata is not the candidate contract"
        )
    version = vtk["version"]
    if version is not None and (not isinstance(version, str) or not version):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} VTK version must be null or a non-empty string"
        )
    piece = _validate_piece(vtk["piece"], case_id)
    raw_arrays = vtk["data_arrays_in_raw_xml_order"]
    if not isinstance(raw_arrays, list) or not raw_arrays:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} VTK DataArray index cannot be empty"
        )
    arrays = [
        _validate_data_array_index(
            row,
            case_id=case_id,
            position=position,
            logical_size_bytes=logical_size_bytes,
        )
        for position, row in enumerate(raw_arrays)
    ]
    return (
        {
            "dataset_type": "UnstructuredGrid",
            "version": version,
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
            "compressor": None,
            "piece": piece,
            "data_array_count": len(arrays),
        },
        arrays,
    )


def _finite_vector(
    value: object,
    *,
    length: int,
    label: str,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise NativeVolumeAuditAggregateError(
            f"{label} must contain {length} finite values"
        )
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate_field_audit(
    value: object,
    *,
    case_id: str,
    field_name: str,
    components: int,
    units: str,
    cell_count: int,
    indexed_arrays: list[dict[str, Any]],
) -> dict[str, object]:
    audit = _exact_keys(
        value,
        {
            "name",
            "association",
            "vtk_type",
            "number_of_components",
            "tuple_count",
            "scalar_count",
            "decoded_payload_bytes",
            "payload_sha256",
            "finite",
            "minimum_by_component",
            "maximum_by_component",
            "units",
            "raw_id_start",
            "raw_id_stop",
        },
        f"{case_id} {field_name} audit",
    )
    if (
        audit["name"] != field_name
        or audit["association"] != "CellData"
        or audit["vtk_type"] != "Float32"
        or audit["number_of_components"] != components
        or audit["units"] != units
        or audit["finite"] is not True
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} association/type/components/units/finite are invalid"
        )
    tuple_count = _integer(
        audit["tuple_count"], f"{case_id} {field_name} tuple_count", minimum=1
    )
    scalar_count = _integer(
        audit["scalar_count"], f"{case_id} {field_name} scalar_count", minimum=1
    )
    payload_bytes = _integer(
        audit["decoded_payload_bytes"],
        f"{case_id} {field_name} decoded payload bytes",
        minimum=1,
    )
    if (
        tuple_count != cell_count
        or scalar_count != cell_count * components
        or payload_bytes != scalar_count * 4
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} tuple/component/payload counts are inconsistent"
        )
    if audit["raw_id_start"] != 0 or audit["raw_id_stop"] != cell_count:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} raw-ID coverage is incomplete"
        )
    minimum = _finite_vector(
        audit["minimum_by_component"],
        length=components,
        label=f"{case_id} {field_name} minimum",
    )
    maximum = _finite_vector(
        audit["maximum_by_component"],
        length=components,
        label=f"{case_id} {field_name} maximum",
    )
    if any(left > right for left, right in zip(minimum, maximum, strict=True)):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} component range is inverted"
        )
    indexed = [
        row
        for row in indexed_arrays
        if row["association"] == "CellData" and row["name"] == field_name
    ]
    if len(indexed) != 1 or (
        indexed[0]["piece_index"] != 0
        or indexed[0]["vtk_type"] != "Float32"
        or indexed[0]["number_of_components"] != components
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} VTK index binding is not exact"
        )
    return {
        "association": "CellData",
        "vtk_type": "Float32",
        "number_of_components": components,
        "tuple_count": tuple_count,
        "scalar_count": scalar_count,
        "decoded_payload_bytes": payload_bytes,
        "payload_sha256": _sha256(
            audit["payload_sha256"], f"{case_id} {field_name} payload SHA-256"
        ),
        "finite": True,
        "minimum_by_component": minimum,
        "maximum_by_component": maximum,
        "units": units,
        "raw_id_start": 0,
        "raw_id_stop": cell_count,
    }


def _validate_metric_summary(
    value: object,
    *,
    sums: Mapping[str, float | int],
    label: str,
) -> dict[str, float]:
    metric = _exact_keys(
        value,
        {"relative_l2_percent", "mae", "rmse"},
        label,
    )
    result = {
        name: _finite(metric[name], f"{label} {name}", nonnegative=True)
        for name in metric
    }
    squared_truth = float(sums["squared_truth"])
    if squared_truth <= 0.0:
        raise NativeVolumeAuditAggregateError(
            f"{label} ground-truth norm must be positive"
        )
    expected = {
        "relative_l2_percent": 100.0
        * math.sqrt(float(sums["squared_error"]) / squared_truth),
        "mae": float(sums["absolute_error"]) / float(sums["total_weight"]),
        "rmse": math.sqrt(
            float(sums["squared_error"]) / float(sums["total_weight"])
        ),
    }
    if any(not _close(result[name], expected[name]) for name in expected):
        raise NativeVolumeAuditAggregateError(
            f"{label} metrics do not match their additive sums"
        )
    if not _close(result["relative_l2_percent"], 100.0):
        raise NativeVolumeAuditAggregateError(
            f"{label} is not the declared all-zero prediction fixture"
        )
    return result


def _validate_additive_sums(
    value: object,
    *,
    cell_count: int,
    expected_total_weight: float,
    label: str,
) -> dict[str, float | int]:
    sums = _exact_keys(
        value,
        {
            "absolute_error",
            "squared_error",
            "squared_truth",
            "entity_count",
            "total_weight",
        },
        label,
    )
    result: dict[str, float | int] = {
        name: _finite(sums[name], f"{label} {name}", nonnegative=True)
        for name in ("absolute_error", "squared_error", "squared_truth")
    }
    result["entity_count"] = _integer(
        sums["entity_count"], f"{label} entity_count", minimum=1
    )
    result["total_weight"] = _positive(sums["total_weight"], f"{label} total_weight")
    if result["entity_count"] != cell_count or not _close(
        float(result["total_weight"]), expected_total_weight
    ):
        raise NativeVolumeAuditAggregateError(
            f"{label} entity coverage or total weight is inconsistent"
        )
    if not _close(
        float(result["squared_error"]), float(result["squared_truth"])
    ):
        raise NativeVolumeAuditAggregateError(
            f"{label} is not based on an all-zero prediction fixture"
        )
    return result


def _validate_metric_pass(
    value: object,
    *,
    case_id: str,
    field_name: str,
    components: int,
    chunk_cells: int,
    cell_count: int,
    payload_sha256: str,
    payload_bytes: int,
) -> dict[str, object]:
    metric_pass = _exact_keys(
        value,
        {
            "field_name",
            "chunk_entities",
            "source_payload_sha256",
            "source_payload_bytes",
            "entity_count",
            "component_count",
            "metrics",
            "additive_sums",
        },
        f"{case_id} {field_name} metric pass",
    )
    if (
        metric_pass["field_name"] != field_name
        or metric_pass["chunk_entities"] != chunk_cells
        or metric_pass["source_payload_sha256"] != payload_sha256
        or metric_pass["source_payload_bytes"] != payload_bytes
        or metric_pass["entity_count"] != cell_count
        or metric_pass["component_count"] != components
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} metric pass source/count/partition binding is invalid"
        )
    sums = _validate_additive_sums(
        metric_pass["additive_sums"],
        cell_count=cell_count,
        expected_total_weight=float(cell_count),
        label=f"{case_id} {field_name} equal-native-cell sums",
    )
    normalized_metrics = _validate_metric_summary(
        metric_pass["metrics"],
        sums=sums,
        label=f"{case_id} {field_name} equal-native-cell metrics",
    )
    return {
        "chunk_entities": chunk_cells,
        "metrics": normalized_metrics,
        "additive_sums": sums,
    }


def _relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), sys.float_info.min)
    return abs(left - right) / scale


def _validate_invariance(
    value: object,
    *,
    case_id: str,
    field_name: str,
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    relative_tolerance: float,
    metric_tolerance: float,
) -> tuple[float, float]:
    invariance = _exact_keys(
        value,
        {
            "same_source_payload_sha256",
            "same_complete_entity_coverage",
            "equal_native_cell_additive_sums",
            "maximum_additive_relative_difference",
            "maximum_metric_absolute_difference",
        },
        f"{case_id} {field_name} invariance",
    )
    if (
        invariance["same_source_payload_sha256"] is not True
        or invariance["same_complete_entity_coverage"] is not True
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} invariance lacks identical source/coverage"
        )
    recorded = _exact_keys(
        invariance["equal_native_cell_additive_sums"],
        {"absolute_error", "squared_error", "squared_truth", "total_weight"},
        f"{case_id} {field_name} equal-native-cell invariance additive sums",
    )
    maximum_relative = 0.0
    for name in ("absolute_error", "squared_error", "squared_truth", "total_weight"):
        difference = _exact_keys(
            recorded[name],
            {"absolute_difference", "relative_difference"},
            f"{case_id} {field_name} equal-native-cell {name} difference",
        )
        left = float(reference["additive_sums"][name])
        right = float(comparison["additive_sums"][name])
        absolute = abs(left - right)
        relative = _relative_difference(left, right)
        recorded_absolute = _finite(
            difference["absolute_difference"],
            f"{case_id} recorded absolute difference",
            nonnegative=True,
        )
        recorded_relative = _finite(
            difference["relative_difference"],
            f"{case_id} recorded relative difference",
            nonnegative=True,
        )
        if not _close(recorded_absolute, absolute) or not _close(
            recorded_relative, relative
        ):
            raise NativeVolumeAuditAggregateError(
                f"{case_id} {field_name} recorded additive invariance is incorrect"
            )
        maximum_relative = max(maximum_relative, relative)
    maximum_metric = max(
        abs(float(reference["metrics"][name]) - float(comparison["metrics"][name]))
        for name in ("relative_l2_percent", "mae", "rmse")
    )
    reported_relative = _finite(
        invariance["maximum_additive_relative_difference"],
        f"{case_id} maximum additive relative difference",
        nonnegative=True,
    )
    reported_metric = _finite(
        invariance["maximum_metric_absolute_difference"],
        f"{case_id} maximum metric absolute difference",
        nonnegative=True,
    )
    if not _close(reported_relative, maximum_relative) or not _close(
        reported_metric, maximum_metric
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} maximum invariance values are incorrect"
        )
    if maximum_relative > relative_tolerance or maximum_metric > metric_tolerance:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} exceeds frozen chunk-invariance thresholds"
        )
    return maximum_relative, maximum_metric


def _validate_metric_fixture(
    value: object,
    *,
    case_id: str,
    field_name: str,
    components: int,
    field_audit: Mapping[str, Any],
    cell_count: int,
    relative_tolerance: float,
    metric_tolerance: float,
) -> tuple[float, float]:
    fixture = _exact_keys(
        value,
        {"prediction_role", "reference_partition", "comparison_partition", "invariance"},
        f"{case_id} {field_name} metric fixture",
    )
    if fixture["prediction_role"] != ZERO_PREDICTION_ROLE:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} {field_name} must label zeros as an invariance fixture"
        )
    if cell_count <= max(REFERENCE_CHUNK_CELLS, COMPARISON_CHUNK_CELLS):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} cell count is too small to prove both chunk partitions"
        )
    reference = _validate_metric_pass(
        fixture["reference_partition"],
        case_id=case_id,
        field_name=field_name,
        components=components,
        chunk_cells=REFERENCE_CHUNK_CELLS,
        cell_count=cell_count,
        payload_sha256=field_audit["payload_sha256"],
        payload_bytes=field_audit["decoded_payload_bytes"],
    )
    comparison = _validate_metric_pass(
        fixture["comparison_partition"],
        case_id=case_id,
        field_name=field_name,
        components=components,
        chunk_cells=COMPARISON_CHUNK_CELLS,
        cell_count=cell_count,
        payload_sha256=field_audit["payload_sha256"],
        payload_bytes=field_audit["decoded_payload_bytes"],
    )
    return _validate_invariance(
        fixture["invariance"],
        case_id=case_id,
        field_name=field_name,
        reference=reference,
        comparison=comparison,
        relative_tolerance=relative_tolerance,
        metric_tolerance=metric_tolerance,
    )


def _validate_equal_cell_weighting(
    value: object,
    *,
    case_id: str,
    cell_count: int,
) -> dict[str, object]:
    weighting = _exact_keys(
        value,
        {
            "weighting",
            "entity_count",
            "total_weight",
            "geometric_cell_volume_weights_used",
        },
        f"{case_id} volume_weighting",
    )
    entity_count = _integer(
        weighting["entity_count"],
        f"{case_id} volume weighting entity_count",
        minimum=1,
    )
    total_weight = _finite(
        weighting["total_weight"],
        f"{case_id} volume weighting total_weight",
        nonnegative=True,
    )
    if (
        weighting["weighting"] != "one_per_native_cell"
        or entity_count != cell_count
        or not isinstance(weighting["total_weight"], float)
        or total_weight != float(cell_count)
        or weighting["geometric_cell_volume_weights_used"] is not False
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} volume weighting must declare exactly one weight per native cell"
        )
    return {
        "weighting": "one_per_native_cell",
        "entity_count": cell_count,
        "total_weight": float(cell_count),
        "geometric_cell_volume_weights_used": False,
    }


def _validate_runtime(value: object, case_id: str) -> None:
    runtime = _exact_keys(
        value,
        {
            "python",
            "numpy",
            "segment_verification_seconds",
            "xml_index_seconds",
            "field_audit_and_equal_cell_metric_seconds",
            "total_seconds",
        },
        f"{case_id} runtime",
    )
    _string(runtime["python"], f"{case_id} runtime python")
    _string(runtime["numpy"], f"{case_id} runtime numpy")
    for name in (
        "segment_verification_seconds",
        "xml_index_seconds",
        "field_audit_and_equal_cell_metric_seconds",
        "total_seconds",
    ):
        _finite(runtime[name], f"{case_id} runtime {name}", nonnegative=True)


def _validate_case_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    pin: NativeSourcePin,
    case: NativeCaseRecord,
) -> dict[str, object]:
    _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "case_id",
            "public_source",
            "vtk",
            "required_cell_data",
            "units",
            "raw_cell_order",
            "coverage",
            "volume_weighting",
            "metrics",
            "chunk_invariance_tolerances",
            "runtime",
        },
        f"{case.case_id} case-audit receipt",
    )
    if (
        receipt["schema"] != CASE_AUDIT_SCHEMA
        or receipt["status"] != "passed_candidate_evaluator_case_audit"
        or receipt["case_id"] != case.case_id
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} case-audit schema/status/case binding is invalid"
        )
    source = _validate_public_source(receipt["public_source"], pin=pin, case=case)
    vtk, indexed_arrays = _validate_vtk(
        receipt["vtk"],
        case_id=case.case_id,
        logical_size_bytes=case.volume_total_size_bytes,
    )
    cell_count = int(vtk["piece"]["number_of_cells"])
    volume_weighting = _validate_equal_cell_weighting(
        receipt["volume_weighting"],
        case_id=case.case_id,
        cell_count=cell_count,
    )
    if receipt["units"] != {
        "coordinates": "m",
        "pMeanTrim": "m^2/s^2",
        "UMeanTrim": "m/s",
    }:
        raise NativeVolumeAuditAggregateError(f"{case.case_id} units are not exact")
    if receipt["raw_cell_order"] != (
        "zero-based VTK Piece CellData tuple order, unchanged"
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} raw-cell-order declaration is not exact"
        )
    coverage = _exact_keys(
        receipt["coverage"],
        {"expected_raw_id_interval", "validation"},
        f"{case.case_id} coverage",
    )
    if coverage["expected_raw_id_interval"] != [0, cell_count] or coverage[
        "validation"
    ] != "each metric pass finalized exact gap-free duplicate-free coverage":
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} raw-ID coverage evidence is incomplete"
        )
    fields_raw = _exact_keys(
        receipt["required_cell_data"],
        {"pMeanTrim", "UMeanTrim"},
        f"{case.case_id} required_cell_data",
    )
    requirements = {
        "pMeanTrim": (1, "m^2/s^2"),
        "UMeanTrim": (3, "m/s"),
    }
    fields = {
        name: _validate_field_audit(
            fields_raw[name],
            case_id=case.case_id,
            field_name=name,
            components=components,
            units=units,
            cell_count=cell_count,
            indexed_arrays=indexed_arrays,
        )
        for name, (components, units) in requirements.items()
    }
    tolerances = _exact_keys(
        receipt["chunk_invariance_tolerances"],
        {
            "maximum_additive_relative_difference",
            "maximum_metric_absolute_difference",
        },
        f"{case.case_id} chunk-invariance tolerances",
    )
    relative_tolerance = _positive(
        tolerances["maximum_additive_relative_difference"],
        f"{case.case_id} additive invariance tolerance",
    )
    metric_tolerance = _positive(
        tolerances["maximum_metric_absolute_difference"],
        f"{case.case_id} metric invariance tolerance",
    )
    if (
        relative_tolerance != MAX_ADDITIVE_RELATIVE_DIFFERENCE
        or metric_tolerance != MAX_METRIC_ABSOLUTE_DIFFERENCE
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} chunk-invariance thresholds are not frozen"
        )
    metrics = _exact_keys(
        receipt["metrics"],
        {"pMeanTrim", "UMeanTrim"},
        f"{case.case_id} metrics",
    )
    invariance = {
        name: _validate_metric_fixture(
            metrics[name],
            case_id=case.case_id,
            field_name=name,
            components=requirements[name][0],
            field_audit=fields[name],
            cell_count=cell_count,
            relative_tolerance=relative_tolerance,
            metric_tolerance=metric_tolerance,
        )
        for name in requirements
    }
    _validate_runtime(receipt["runtime"], case.case_id)
    return {
        "case_id": case.case_id,
        "receipt_sha256": receipt_sha256,
        "source": source,
        "vtk": vtk,
        "required_cell_data": fields,
        "volume_weighting": volume_weighting,
        "chunk_invariance": {
            "prediction_role": ZERO_PREDICTION_ROLE,
            "not_a_physics_null_baseline": True,
            "reference_chunk_cells": REFERENCE_CHUNK_CELLS,
            "comparison_chunk_cells": COMPARISON_CHUNK_CELLS,
            "maximum_additive_relative_difference": max(
                values[0] for values in invariance.values()
            ),
            "maximum_metric_absolute_difference": max(
                values[1] for values in invariance.values()
            ),
            "thresholds": {
                "maximum_additive_relative_difference": relative_tolerance,
                "maximum_metric_absolute_difference": metric_tolerance,
            },
        },
    }


def _aggregate_native_volume_equal_cell(
    *,
    native_source_pin_path: Path,
    receipt_paths: Sequence[Path],
    selected_case_ids: Sequence[str] | None,
    require_complete_scope: bool,
) -> dict[str, object]:
    """Validate path-free one-weight-per-native-cell volume evidence."""

    pin_path = Path(native_source_pin_path)
    pin_sha256 = sha256_file(pin_path)
    try:
        pin = load_native_source_pin(pin_path)
    except NativeSourceError as error:
        raise NativeVolumeAuditAggregateError(str(error)) from error
    if sha256_file(pin_path) != pin_sha256:
        raise NativeVolumeAuditAggregateError(
            "native-source pin changed while being loaded"
        )
    selected = _selected_cases(pin, selected_case_ids)
    complete_scope = len(selected) == len(pin.cases)
    if require_complete_scope and not complete_scope:
        raise NativeVolumeAuditAggregateError(
            "equal-cell all-case audit must cover the complete native-source pin"
        )
    if not require_complete_scope and complete_scope:
        raise NativeVolumeAuditAggregateError(
            "equal-cell pilot mode must remain an incomplete selected subset"
        )
    selected_ids = tuple(case.case_id for case in selected)

    receipts: dict[str, dict[str, object]] = {}
    for raw_path in receipt_paths:
        path = Path(raw_path)
        receipt, digest = _read_json_with_sha256(
            path, "native-volume equal-cell receipt"
        )
        case_id = _string(receipt.get("case_id"), "case-audit case_id")
        if case_id in receipts:
            raise NativeVolumeAuditAggregateError(
                f"duplicate native-volume audit receipt for {case_id}"
            )
        if case_id not in selected_ids:
            raise NativeVolumeAuditAggregateError(
                f"unexpected native-volume audit receipt for {case_id}"
            )
        receipts[case_id] = _validate_case_receipt(
            receipt,
            receipt_sha256=digest,
            pin=pin,
            case=pin.case(case_id),
        )
    if set(receipts) != set(selected_ids):
        missing = [case_id for case_id in selected_ids if case_id not in receipts]
        raise NativeVolumeAuditAggregateError(
            "native-volume equal-cell receipts must cover selected cases "
            f"exactly once (missing={missing})"
        )

    records = [receipts[case_id] for case_id in selected_ids]
    two_part = sum(row["source"]["multipart_part_count"] == 2 for row in records)
    three_part = sum(row["source"]["multipart_part_count"] == 3 for row in records)
    if two_part + three_part != len(records):
        raise NativeVolumeAuditAggregateError(
            "selected cases contain an unsupported multipart count"
        )
    evidence: dict[str, object] = {
        "schema": (
            EQUAL_CELL_ALL_CASE_AUDIT_SCHEMA
            if require_complete_scope
            else EQUAL_CELL_PILOT_AGGREGATE_SCHEMA
        ),
        "schema_version": 2,
        "status": (
            "passed_all_case_equal_native_cell_audit"
            if require_complete_scope
            else "passed_incomplete_equal_native_cell_pilot"
        ),
        "activation_status": "does_not_activate_scoring_contract",
        "scope_note": (
            "all native cases were audited for transport, fields, raw-cell "
            "coverage, and chunk invariance using one weight per native cell"
            if require_complete_scope
            else "selected-case implementation evidence only; all-case native-source "
            "pin coverage is not claimed"
        ),
        "scope": {
            "selected_case_count": len(records),
            "native_source_pin_case_count": len(pin.cases),
            "complete_native_source_pin_scope": complete_scope,
            "case_order": "native_source_pin_order",
        },
        "source": {
            "native_source_pin_sha256": pin_sha256,
            "repository_id": pin.repository_id,
            "immutable_revision": pin.repository_revision,
        },
        "fixture_semantics": {
            "prediction_role": ZERO_PREDICTION_ROLE,
            "purpose": "native_transport_fields_coverage_and_chunk_invariance",
            "physics_null_baseline": False,
            "model_quality_claim": False,
            "volume_weighting": "one_per_native_cell",
            "geometric_cell_volume_weights_used": False,
        },
        "totals": {
            "case_count": len(records),
            "two_part_case_count": two_part,
            "three_part_case_count": three_part,
            "verified_segment_count": sum(
                row["source"]["multipart_part_count"] for row in records
            ),
            "logical_volume_size_bytes": sum(
                row["source"]["logical_size_bytes"] for row in records
            ),
            "native_point_count": sum(
                row["vtk"]["piece"]["number_of_points"] for row in records
            ),
            "native_cell_count": sum(
                row["vtk"]["piece"]["number_of_cells"] for row in records
            ),
            "decoded_payload_bytes": {
                name: sum(
                    row["required_cell_data"][name]["decoded_payload_bytes"]
                    for row in records
                )
                for name in ("pMeanTrim", "UMeanTrim")
            },
            "field_ranges": {
                name: _aggregate_field_ranges(records, name)
                for name in ("pMeanTrim", "UMeanTrim")
            },
            "maximum_additive_relative_difference": max(
                row["chunk_invariance"]["maximum_additive_relative_difference"]
                for row in records
            ),
            "maximum_metric_absolute_difference": max(
                row["chunk_invariance"]["maximum_metric_absolute_difference"]
                for row in records
            ),
        },
        "cases": records,
    }
    _assert_no_absolute_paths(evidence)
    return evidence


def aggregate_native_volume_equal_cell_pilot(
    *,
    native_source_pin_path: Path,
    receipt_paths: Sequence[Path],
    selected_case_ids: Sequence[str],
) -> dict[str, object]:
    """Validate an explicitly incomplete equal-cell selected-case pilot."""

    return _aggregate_native_volume_equal_cell(
        native_source_pin_path=native_source_pin_path,
        receipt_paths=receipt_paths,
        selected_case_ids=selected_case_ids,
        require_complete_scope=False,
    )


def aggregate_native_volume_equal_cell_all_case_audit(
    *,
    native_source_pin_path: Path,
    receipt_paths: Sequence[Path],
) -> dict[str, object]:
    """Validate complete all-pin equal-cell native-volume audit evidence."""

    return _aggregate_native_volume_equal_cell(
        native_source_pin_path=native_source_pin_path,
        receipt_paths=receipt_paths,
        selected_case_ids=None,
        require_complete_scope=True,
    )


def _aggregate_field_ranges(
    records: list[dict[str, object]],
    field_name: str,
) -> dict[str, list[float]]:
    component_count = len(
        records[0]["required_cell_data"][field_name]["minimum_by_component"]
    )
    return {
        "minimum_by_component": [
            min(
                float(
                    row["required_cell_data"][field_name][
                        "minimum_by_component"
                    ][component]
                )
                for row in records
            )
            for component in range(component_count)
        ],
        "maximum_by_component": [
            max(
                float(
                    row["required_cell_data"][field_name][
                        "maximum_by_component"
                    ][component]
                )
                for row in records
            )
            for component in range(component_count)
        ],
    }


def write_evidence(path: Path, evidence: Mapping[str, object]) -> None:
    """Atomically write stable compact path-free aggregate JSON."""

    _assert_no_absolute_paths(evidence)
    destination = Path(path)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-source-pin",
        type=Path,
        default=DEFAULT_NATIVE_SOURCE_PIN,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--equal-cell-pilot",
        action="store_true",
        help="publish an explicitly incomplete equal-cell selected-case pilot",
    )
    mode.add_argument(
        "--equal-cell-all-case-audit",
        action="store_true",
        help="publish a complete all-pin equal-native-cell audit",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help="explicit selected case in native-source pin order; repeat as needed",
    )
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        if args.equal_cell_pilot:
            if not args.case:
                raise NativeVolumeAuditAggregateError(
                    "--equal-cell-pilot requires explicit --case selections"
                )
            evidence = aggregate_native_volume_equal_cell_pilot(
                native_source_pin_path=args.native_source_pin,
                receipt_paths=args.receipts,
                selected_case_ids=args.case,
            )
        elif args.equal_cell_all_case_audit:
            if args.case:
                raise NativeVolumeAuditAggregateError(
                    "--equal-cell-all-case-audit does not accept --case"
                )
            evidence = aggregate_native_volume_equal_cell_all_case_audit(
                native_source_pin_path=args.native_source_pin,
                receipt_paths=args.receipts,
            )
        write_evidence(args.output, evidence)
    except NativeVolumeAuditAggregateError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "case_count": evidence["totals"]["case_count"],
                "status": evidence["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_AUDIT_SCHEMA",
    "COMPARISON_CHUNK_CELLS",
    "EQUAL_CELL_ALL_CASE_AUDIT_SCHEMA",
    "EQUAL_CELL_PILOT_AGGREGATE_SCHEMA",
    "MAX_ADDITIVE_RELATIVE_DIFFERENCE",
    "MAX_METRIC_ABSOLUTE_DIFFERENCE",
    "NativeVolumeAuditAggregateError",
    "REFERENCE_CHUNK_CELLS",
    "ZERO_PREDICTION_ROLE",
    "aggregate_native_volume_equal_cell_all_case_audit",
    "aggregate_native_volume_equal_cell_pilot",
    "main",
    "sha256_file",
    "write_evidence",
]
