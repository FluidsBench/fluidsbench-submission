#!/usr/bin/env python3
"""Aggregate strict DrivAerML native-volume case-audit evidence.

The default scope is every case in the immutable native-source pin.  Explicit
``--case`` arguments permit a bounded pilot, but receipts and the supplied
volume-weight aggregate must still cover that selected set exactly once.
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

from reference.drivaerml.source import (  # noqa: E402
    NativeCaseRecord,
    NativeSourceError,
    NativeSourcePin,
    load_native_source_pin,
)


DEFAULT_NATIVE_SOURCE_PIN = (
    ROOT / "benchmark-specs" / "drivaerml" / "proposal" / "native-source-pin.json"
)
CASE_AUDIT_SCHEMA = "drivaerml-native-volume-case-audit-v1"
WEIGHT_AGGREGATE_SCHEMA = "drivaerml-volume-cell-weight-receipt-aggregate-v1"
AGGREGATE_SCHEMA = "drivaerml-native-volume-audit-aggregate-v1"
PRIMARY_PILOT_AGGREGATE_SCHEMA = (
    "drivaerml-native-volume-equal-cell-primary-pilot-aggregate-v1"
)
PRIMARY_ALL_CASE_AUDIT_SCHEMA = (
    "drivaerml-native-volume-equal-cell-primary-all-case-audit-v1"
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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except NativeVolumeAuditAggregateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeVolumeAuditAggregateError(
            f"cannot read valid {label} JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise NativeVolumeAuditAggregateError(f"{label} must be a JSON object")
    return value


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


def _safe_basename(value: object, label: str, *, suffix: str) -> str:
    result = _relative_public_path(value, label)
    if "/" in result or not result.endswith(suffix):
        raise NativeVolumeAuditAggregateError(
            f"{label} must be a basename ending in {suffix}"
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
    volume_sum_m3: float,
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
    additive = _exact_keys(
        metric_pass["additive_sums"],
        {"uniform", "physical"},
        f"{case_id} {field_name} additive sums",
    )
    sums = {
        "uniform": _validate_additive_sums(
            additive["uniform"],
            cell_count=cell_count,
            expected_total_weight=float(cell_count),
            label=f"{case_id} {field_name} uniform sums",
        ),
        "physical": _validate_additive_sums(
            additive["physical"],
            cell_count=cell_count,
            expected_total_weight=volume_sum_m3,
            label=f"{case_id} {field_name} physical sums",
        ),
    }
    metrics = _exact_keys(
        metric_pass["metrics"],
        {"uniform", "physical"},
        f"{case_id} {field_name} metric values",
    )
    normalized_metrics = {
        weighting: _validate_metric_summary(
            metrics[weighting],
            sums=sums[weighting],
            label=f"{case_id} {field_name} {weighting} metrics",
        )
        for weighting in ("uniform", "physical")
    }
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
            "additive_sums",
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
        invariance["additive_sums"],
        {"uniform", "physical"},
        f"{case_id} {field_name} invariance additive sums",
    )
    maximum_relative = 0.0
    for weighting in ("uniform", "physical"):
        rows = _exact_keys(
            recorded[weighting],
            {"absolute_error", "squared_error", "squared_truth", "total_weight"},
            f"{case_id} {field_name} {weighting} invariance",
        )
        for name in ("absolute_error", "squared_error", "squared_truth", "total_weight"):
            difference = _exact_keys(
                rows[name],
                {"absolute_difference", "relative_difference"},
                f"{case_id} {field_name} {weighting} {name} difference",
            )
            left = float(reference["additive_sums"][weighting][name])
            right = float(comparison["additive_sums"][weighting][name])
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
        abs(
            float(reference["metrics"][weighting][name])
            - float(comparison["metrics"][weighting][name])
        )
        for weighting in ("uniform", "physical")
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
    volume_sum_m3: float,
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
        volume_sum_m3=volume_sum_m3,
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
        volume_sum_m3=volume_sum_m3,
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


def _validate_weight_source_binding(
    value: object,
    *,
    pin: NativeSourcePin,
    pin_sha256: str,
    case: NativeCaseRecord,
) -> None:
    binding = _exact_keys(
        value,
        {"pin", "case_id", "logical_volume", "verification"},
        f"{case.case_id} weight native_source_binding",
    )
    pin_binding = _exact_keys(
        binding["pin"],
        {"sha256", "repository_id", "repository_revision"},
        f"{case.case_id} weight pin binding",
    )
    if (
        pin_binding["sha256"] != pin_sha256
        or pin_binding["repository_id"] != pin.repository_id
        or pin_binding["repository_revision"] != pin.repository_revision
        or binding["case_id"] != case.case_id
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} weight source pin/case binding is inconsistent"
        )
    logical = _exact_keys(
        binding["logical_volume"],
        {"path", "size_bytes", "ordered_verified_segments"},
        f"{case.case_id} weight logical volume",
    )
    if (
        logical["path"] != case.volume_logical_path.as_posix()
        or logical["size_bytes"] != case.volume_total_size_bytes
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} weight logical volume is not pinned"
        )
    segments = logical["ordered_verified_segments"]
    if not isinstance(segments, list) or len(segments) != len(case.volume_parts):
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} weight source segments are incomplete"
        )
    for index, (raw, part) in enumerate(zip(segments, case.volume_parts, strict=True)):
        row = _exact_keys(
            raw,
            {"part_index", "size_bytes", "sha256"},
            f"{case.case_id} weight source segment {index}",
        )
        verified_index = _integer(
            row["part_index"], f"{case.case_id} weight segment {index} index"
        )
        verified_size = _integer(
            row["size_bytes"],
            f"{case.case_id} weight segment {index} size",
            minimum=1,
        )
        verified_sha256 = _sha256(
            row["sha256"], f"{case.case_id} weight segment {index} SHA-256"
        )
        if (
            verified_index != index
            or verified_size != part.size_bytes
            or verified_sha256 != part.sha256
        ):
            raise NativeVolumeAuditAggregateError(
                f"{case.case_id} weight source segment {index} is not pinned"
            )
    if dict(_mapping(binding["verification"], "weight verification")) != {
        "method": "exact_ordered_segment_size_and_sha256",
        "timing": "completed_before_vtk_geometry_reader",
        "vtk_input": "retained_verified_file_descriptor",
        "post_vtk_fstat": "unchanged",
    }:
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} weight source verification is not exact"
        )


def _validate_weight_output(
    value: object,
    *,
    case_id: str,
) -> dict[str, object]:
    output = _exact_keys(
        value,
        {
            "file",
            "dtype",
            "shape",
            "cell_count",
            "size_bytes",
            "sha256",
            "volume_sum_m3",
            "volume_min_m3",
            "volume_max_m3",
        },
        f"{case_id} weight output",
    )
    count = _integer(output["cell_count"], f"{case_id} weight cell_count", minimum=1)
    if output["dtype"] != "<f8" or output["shape"] != [count]:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} weight dtype/shape/count is invalid"
        )
    size = _integer(output["size_bytes"], f"{case_id} weight size", minimum=1)
    if size != 128 + 8 * count:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} weight NPY size/count is invalid"
        )
    total = _positive(output["volume_sum_m3"], f"{case_id} weight sum")
    minimum = _positive(output["volume_min_m3"], f"{case_id} weight minimum")
    maximum = _positive(output["volume_max_m3"], f"{case_id} weight maximum")
    if minimum > maximum or not minimum * count <= total <= maximum * count:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} weight sum/minimum/maximum is inconsistent"
        )
    return {
        "file": _safe_basename(
            output["file"], f"{case_id} weight file", suffix=".npy"
        ),
        "dtype": "<f8",
        "cell_count": count,
        "size_bytes": size,
        "sha256": _sha256(output["sha256"], f"{case_id} weight SHA-256"),
        "volume_sum_m3": total,
        "volume_min_m3": minimum,
        "volume_max_m3": maximum,
    }


def _load_weight_aggregate(
    path: Path,
    *,
    pin: NativeSourcePin,
    pin_sha256: str,
    selected_cases: tuple[NativeCaseRecord, ...],
) -> tuple[str, dict[str, dict[str, object]]]:
    digest = sha256_file(path)
    document = _read_json(path, "volume-weight aggregate evidence")
    _assert_no_absolute_paths(document, "volume-weight aggregate")
    required_root = {
        "schema",
        "mode",
        "status",
        "complete",
        "public_evidence_eligible",
        "activation_status",
        "case_count",
        "official_case_count",
        "omitted_official_case_count",
        "case_order",
        "source",
        "dependencies",
        "algorithm",
        "aggregate",
        "cases",
    }
    if document.get("mode") == "partial_pilot":
        required_root.add("pilot_warning")
    _exact_keys(document, required_root, "volume-weight aggregate")
    if (
        document["schema"] != WEIGHT_AGGREGATE_SCHEMA
        or document["activation_status"] != "does_not_activate_scoring_contract"
        or document["case_order"] != "native_source_pin_increasing_run_number"
    ):
        raise NativeVolumeAuditAggregateError(
            "volume-weight aggregate schema/status/order is invalid"
        )
    source = _exact_keys(
        document["source"],
        {"dataset", "native_source_pin", "assembled_vtu_identity"},
        "volume-weight aggregate source",
    )
    dataset = _mapping(source["dataset"], "weight dataset")
    source_pin = _mapping(source["native_source_pin"], "weight source pin")
    if (
        dataset.get("provider") != "Hugging Face Hub"
        or dataset.get("repo_id") != pin.repository_id
        or dataset.get("revision") != pin.repository_revision
        or source_pin.get("schema")
        != "drivaerml-fluidsbench-public-native-source-pin-v1"
        or source_pin.get("sha256") != pin_sha256
        or source["assembled_vtu_identity"]
        != "exact_ordered_part_segment_size_and_sha256_verified_before_vtk"
    ):
        raise NativeVolumeAuditAggregateError(
            "volume-weight aggregate is bound to a different native source"
        )
    selected_ids = tuple(case.case_id for case in selected_cases)
    all_pin_ids = tuple(case.case_id for case in pin.cases)
    full_scope = selected_ids == all_pin_ids
    expected_mode = "complete" if full_scope else "partial_pilot"
    expected_status = (
        "complete_all_official_cases_candidate_evidence"
        if full_scope
        else "incomplete_non_public_pilot"
    )
    if (
        document["mode"] != expected_mode
        or document["status"] != expected_status
        or document["complete"] is not full_scope
        or document["public_evidence_eligible"] is not full_scope
        or document["case_count"] != len(selected_ids)
        or document["official_case_count"] != len(pin.cases)
        or document["omitted_official_case_count"]
        != len(pin.cases) - len(selected_ids)
    ):
        raise NativeVolumeAuditAggregateError(
            "volume-weight aggregate scope/status flags are inconsistent"
        )
    if not full_scope and not isinstance(document.get("pilot_warning"), str):
        raise NativeVolumeAuditAggregateError(
            "partial volume-weight aggregate lacks its pilot warning"
        )
    raw_cases = document["cases"]
    if not isinstance(raw_cases, list):
        raise NativeVolumeAuditAggregateError("volume-weight cases must be an array")
    by_case: dict[str, dict[str, object]] = {}
    for raw in raw_cases:
        row = _exact_keys(
            raw,
            {
                "case_id",
                "receipt_sha256",
                "native_source_binding",
                "output",
                "reader_audit",
            },
            "volume-weight case",
        )
        case_id = _string(row["case_id"], "volume-weight case_id")
        if case_id in by_case:
            raise NativeVolumeAuditAggregateError(
                f"duplicate volume-weight case {case_id}"
            )
        try:
            case = pin.case(case_id)
        except NativeSourceError as error:
            raise NativeVolumeAuditAggregateError(str(error)) from error
        _sha256(row["receipt_sha256"], f"{case_id} weight receipt SHA-256")
        _validate_weight_source_binding(
            row["native_source_binding"],
            pin=pin,
            pin_sha256=pin_sha256,
            case=case,
        )
        by_case[case_id] = _validate_weight_output(row["output"], case_id=case_id)
    if tuple(by_case) != selected_ids:
        raise NativeVolumeAuditAggregateError(
            "volume-weight aggregate cases must exactly match selected audit cases"
        )
    aggregate = _exact_keys(
        document["aggregate"],
        {
            "cell_count",
            "source_vtu_size_bytes",
            "output_size_bytes",
            "volume_sum_m3",
            "volume_min_m3",
            "volume_max_m3",
            "all_outputs_dtype",
            "all_outputs_one_dimensional",
            "all_values_strictly_positive_finite",
        },
        "volume-weight aggregate totals",
    )
    if (
        aggregate["all_outputs_dtype"] != "<f8"
        or aggregate["all_outputs_one_dimensional"] is not True
        or aggregate["all_values_strictly_positive_finite"] is not True
    ):
        raise NativeVolumeAuditAggregateError(
            "volume-weight aggregate output guarantees are invalid"
        )
    expected = {
        "cell_count": sum(int(row["cell_count"]) for row in by_case.values()),
        "source_vtu_size_bytes": sum(
            case.volume_total_size_bytes for case in selected_cases
        ),
        "output_size_bytes": sum(int(row["size_bytes"]) for row in by_case.values()),
        "volume_sum_m3": math.fsum(
            float(row["volume_sum_m3"]) for row in by_case.values()
        ),
        "volume_min_m3": min(
            float(row["volume_min_m3"]) for row in by_case.values()
        ),
        "volume_max_m3": max(
            float(row["volume_max_m3"]) for row in by_case.values()
        ),
    }
    for name, expected_value in expected.items():
        actual = aggregate.get(name)
        if isinstance(expected_value, float):
            if not isinstance(actual, (int, float)) or not _close(
                float(actual), expected_value
            ):
                raise NativeVolumeAuditAggregateError(
                    f"volume-weight aggregate {name} is inconsistent"
                )
        elif actual != expected_value:
            raise NativeVolumeAuditAggregateError(
                f"volume-weight aggregate {name} is inconsistent"
            )
    return digest, by_case


def _validate_case_weight_audit(
    value: object,
    *,
    case_id: str,
    expected: Mapping[str, object],
) -> dict[str, object]:
    audit = _exact_keys(
        value,
        {
            "path",
            "sha256",
            "dtype",
            "cell_count",
            "sum_m3",
            "minimum_m3",
            "maximum_m3",
            "role",
            "status",
        },
        f"{case_id} volume_weights",
    )
    if (
        audit["role"] != "fixed_same_order_cell_volume_secondary_weights"
        or audit["status"] != "audited"
        or audit["dtype"] != "<f8"
    ):
        raise NativeVolumeAuditAggregateError(
            f"{case_id} did not audit the fixed v2 volume weights"
        )
    path_name = Path(_string(audit["path"], f"{case_id} weight path")).name
    comparisons = {
        "file": path_name,
        "sha256": _sha256(audit["sha256"], f"{case_id} audited weight SHA-256"),
        "cell_count": _integer(
            audit["cell_count"], f"{case_id} audited weight count", minimum=1
        ),
        "volume_sum_m3": _positive(audit["sum_m3"], f"{case_id} audited weight sum"),
        "volume_min_m3": _positive(
            audit["minimum_m3"], f"{case_id} audited weight minimum"
        ),
        "volume_max_m3": _positive(
            audit["maximum_m3"], f"{case_id} audited weight maximum"
        ),
    }
    for name, actual in comparisons.items():
        expected_value = expected[name]
        if isinstance(actual, float):
            matches = _close(actual, float(expected_value))
        else:
            matches = actual == expected_value
        if not matches:
            raise NativeVolumeAuditAggregateError(
                f"{case_id} audited weight {name} differs from v2 weight evidence"
            )
    return {
        "file": expected["file"],
        "dtype": "<f8",
        "cell_count": expected["cell_count"],
        "size_bytes": expected["size_bytes"],
        "sha256": expected["sha256"],
        "volume_sum_m3": expected["volume_sum_m3"],
        "volume_min_m3": expected["volume_min_m3"],
        "volume_max_m3": expected["volume_max_m3"],
        "status": "audited_against_v2_volume_weight_evidence",
    }


def _validate_equal_cell_primary_weight_audit(
    value: object,
    *,
    case_id: str,
    cell_count: int,
) -> dict[str, object]:
    audit = _exact_keys(
        value,
        {"role", "status"},
        f"{case_id} equal-cell pilot weights",
    )
    if audit != {
        "role": "unit_weights_for_equal_native_cell_primary_pilot",
        "status": "physical_secondary_not_exercised",
    }:
        raise NativeVolumeAuditAggregateError(
            f"{case_id} equal-cell pilot weight declaration is not exact"
        )
    return {
        "weighting": "one_per_native_cell",
        "cell_count": cell_count,
        "volume_sum_m3": float(cell_count),
        "status": "equal_cell_primary_only_physical_secondary_not_exercised",
    }


def _validate_runtime(value: object, case_id: str) -> None:
    runtime = _exact_keys(
        value,
        {
            "python",
            "numpy",
            "segment_verification_seconds",
            "xml_index_seconds",
            "volume_weight_audit_seconds",
            "field_audit_and_dual_metric_seconds",
            "total_seconds",
        },
        f"{case_id} runtime",
    )
    _string(runtime["python"], f"{case_id} runtime python")
    _string(runtime["numpy"], f"{case_id} runtime numpy")
    for name in (
        "segment_verification_seconds",
        "xml_index_seconds",
        "volume_weight_audit_seconds",
        "field_audit_and_dual_metric_seconds",
        "total_seconds",
    ):
        _finite(runtime[name], f"{case_id} runtime {name}", nonnegative=True)


def _validate_case_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    pin: NativeSourcePin,
    case: NativeCaseRecord,
    expected_weight: Mapping[str, object] | None,
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
            "volume_weights",
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
    if expected_weight is None:
        weights = _validate_equal_cell_primary_weight_audit(
            receipt["volume_weights"],
            case_id=case.case_id,
            cell_count=cell_count,
        )
    else:
        weights = _validate_case_weight_audit(
            receipt["volume_weights"],
            case_id=case.case_id,
            expected=expected_weight,
        )
    if weights["cell_count"] != cell_count:
        raise NativeVolumeAuditAggregateError(
            f"{case.case_id} VTK and volume-weight cell counts differ"
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
            volume_sum_m3=float(weights["volume_sum_m3"]),
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
        "volume_weights": weights,
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


def _aggregate_native_volume_primary(
    *,
    native_source_pin_path: Path,
    receipt_paths: Sequence[Path],
    selected_case_ids: Sequence[str] | None,
    require_complete_scope: bool,
) -> dict[str, object]:
    """Validate path-free equal-cell native-volume evidence without weights.

    This deliberately cannot represent a completed physical-volume secondary
    audit. It records native-source, CellData, raw-order, coverage, and
    chunk-invariance evidence while the independent deterministic volume-weight
    gate remains unresolved.
    """

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
            "equal-cell primary all-case audit must cover the complete native-source pin"
        )
    if not require_complete_scope and complete_scope:
        raise NativeVolumeAuditAggregateError(
            "equal-cell primary pilot mode must remain an incomplete selected subset"
        )
    selected_ids = tuple(case.case_id for case in selected)

    receipts: dict[str, dict[str, object]] = {}
    for raw_path in receipt_paths:
        path = Path(raw_path)
        digest = sha256_file(path)
        receipt = _read_json(path, "native-volume equal-cell primary receipt")
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
            expected_weight=None,
        )
    if set(receipts) != set(selected_ids):
        missing = [case_id for case_id in selected_ids if case_id not in receipts]
        raise NativeVolumeAuditAggregateError(
            "native-volume equal-cell primary receipts must cover selected cases "
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
            PRIMARY_ALL_CASE_AUDIT_SCHEMA
            if require_complete_scope
            else PRIMARY_PILOT_AGGREGATE_SCHEMA
        ),
        "schema_version": 1,
        "status": (
            "passed_all_case_equal_cell_primary_audit"
            if require_complete_scope
            else "passed_incomplete_equal_cell_primary_pilot"
        ),
        "activation_status": "does_not_activate_scoring_contract",
        "public_scoring_support_eligible": False,
        (
            "audit_warning"
            if require_complete_scope
            else "pilot_warning"
        ): (
            "all native cases were audited for equal-cell primary transport, "
            "fields, coverage, and chunk invariance; deterministic physical "
            "volume weights and the secondary weighting are not claimed"
            if require_complete_scope
            else "selected-case implementation evidence only; deterministic "
            "physical volume weights and all-case volume coverage are not claimed"
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
            "volume_weighting_exercised": "equal_native_cell_primary_only",
            "physical_volume_secondary_exercised": False,
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


def aggregate_native_volume_primary_pilot(
    *,
    native_source_pin_path: Path,
    receipt_paths: Sequence[Path],
    selected_case_ids: Sequence[str],
) -> dict[str, object]:
    """Validate an explicitly incomplete equal-cell selected-case pilot."""

    return _aggregate_native_volume_primary(
        native_source_pin_path=native_source_pin_path,
        receipt_paths=receipt_paths,
        selected_case_ids=selected_case_ids,
        require_complete_scope=False,
    )


def aggregate_native_volume_primary_all_case_audit(
    *,
    native_source_pin_path: Path,
    receipt_paths: Sequence[Path],
) -> dict[str, object]:
    """Validate complete all-pin equal-cell native-volume audit evidence."""

    return _aggregate_native_volume_primary(
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


def aggregate_native_volume_audits(
    *,
    native_source_pin_path: Path,
    volume_weight_aggregate_path: Path,
    receipt_paths: Sequence[Path],
    selected_case_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate exact selected-case coverage and return path-free evidence."""

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
    selected_ids = tuple(case.case_id for case in selected)
    weight_aggregate_sha256, weight_cases = _load_weight_aggregate(
        Path(volume_weight_aggregate_path),
        pin=pin,
        pin_sha256=pin_sha256,
        selected_cases=selected,
    )

    receipts: dict[str, dict[str, object]] = {}
    for raw_path in receipt_paths:
        path = Path(raw_path)
        digest = sha256_file(path)
        receipt = _read_json(path, "native-volume case-audit receipt")
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
            expected_weight=weight_cases[case_id],
        )
    if set(receipts) != set(selected_ids):
        missing = [case_id for case_id in selected_ids if case_id not in receipts]
        extra = [case_id for case_id in receipts if case_id not in selected_ids]
        raise NativeVolumeAuditAggregateError(
            "native-volume audit receipts must cover selected cases exactly once "
            f"(missing={missing}, extra={extra})"
        )
    records = [receipts[case_id] for case_id in selected_ids]
    two_part = sum(row["source"]["multipart_part_count"] == 2 for row in records)
    three_part = sum(row["source"]["multipart_part_count"] == 3 for row in records)
    evidence: dict[str, object] = {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": 1,
        "status": "passed_candidate_native_volume_audit_aggregate",
        "activation_status": "does_not_activate_scoring_contract",
        "scope": {
            "selected_case_count": len(records),
            "native_source_pin_case_count": len(pin.cases),
            "complete_native_source_pin_scope": len(records) == len(pin.cases),
            "case_order": "native_source_pin_order",
        },
        "source": {
            "native_source_pin_sha256": pin_sha256,
            "repository_id": pin.repository_id,
            "immutable_revision": pin.repository_revision,
            "volume_weight_aggregate_sha256": weight_aggregate_sha256,
        },
        "fixture_semantics": {
            "prediction_role": ZERO_PREDICTION_ROLE,
            "purpose": "transport_and_chunk_invariance_validation_only",
            "physics_null_baseline": False,
            "model_quality_claim": False,
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
            "volume_weight_size_bytes": sum(
                row["volume_weights"]["size_bytes"] for row in records
            ),
            "volume_sum_m3": math.fsum(
                row["volume_weights"]["volume_sum_m3"] for row in records
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
                row["chunk_invariance"][
                    "maximum_additive_relative_difference"
                ]
                for row in records
            ),
            "maximum_metric_absolute_difference": max(
                row["chunk_invariance"]["maximum_metric_absolute_difference"]
                for row in records
            ),
        },
        "cases": records,
    }
    if two_part + three_part != len(records):
        raise NativeVolumeAuditAggregateError(
            "selected cases contain an unsupported multipart count"
        )
    _assert_no_absolute_paths(evidence)
    return evidence


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
    mode.add_argument("--volume-weight-aggregate", type=Path)
    mode.add_argument(
        "--primary-only-pilot",
        action="store_true",
        help="publish an explicitly incomplete equal-cell selected-case pilot",
    )
    mode.add_argument(
        "--primary-only-all-case-audit",
        action="store_true",
        help="publish complete all-pin equal-cell audit without physical weights",
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
        if args.primary_only_pilot:
            if not args.case:
                raise NativeVolumeAuditAggregateError(
                    "--primary-only-pilot requires explicit --case selections"
                )
            evidence = aggregate_native_volume_primary_pilot(
                native_source_pin_path=args.native_source_pin,
                receipt_paths=args.receipts,
                selected_case_ids=args.case,
            )
        elif args.primary_only_all_case_audit:
            if args.case:
                raise NativeVolumeAuditAggregateError(
                    "--primary-only-all-case-audit does not accept --case"
                )
            evidence = aggregate_native_volume_primary_all_case_audit(
                native_source_pin_path=args.native_source_pin,
                receipt_paths=args.receipts,
            )
        else:
            evidence = aggregate_native_volume_audits(
                native_source_pin_path=args.native_source_pin,
                volume_weight_aggregate_path=args.volume_weight_aggregate,
                receipt_paths=args.receipts,
                selected_case_ids=args.case,
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
    "AGGREGATE_SCHEMA",
    "PRIMARY_PILOT_AGGREGATE_SCHEMA",
    "PRIMARY_ALL_CASE_AUDIT_SCHEMA",
    "CASE_AUDIT_SCHEMA",
    "COMPARISON_CHUNK_CELLS",
    "MAX_ADDITIVE_RELATIVE_DIFFERENCE",
    "MAX_METRIC_ABSOLUTE_DIFFERENCE",
    "NativeVolumeAuditAggregateError",
    "REFERENCE_CHUNK_CELLS",
    "WEIGHT_AGGREGATE_SCHEMA",
    "ZERO_PREDICTION_ROLE",
    "aggregate_native_volume_audits",
    "aggregate_native_volume_primary_pilot",
    "aggregate_native_volume_primary_all_case_audit",
    "main",
    "sha256_file",
    "write_evidence",
]
