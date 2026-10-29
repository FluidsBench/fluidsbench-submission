#!/usr/bin/env python3
"""Export compact, source-bound DrivAerML native profile truth.

The exporter has two deliberately separate phases.  ``case`` is safe for a
484-task scheduler array: each task verifies one official case and atomically
writes ``cases/run_N.json``.  ``assemble`` accepts only the exact complete
official case set and deterministically writes website chunks, the all-case
index, eight thin submission-split indexes, provenance, and a release receipt.

This is a truth publication utility.  It does not alter the scoring contract,
activate relative profiles, open submissions, or approve any release gate.
"""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    U_INF_M_PER_S,
    cp_from_kinematic_pressure,
    velocity_magnitude_ratio,
)
from reference.drivaerml.coordinate_identity import (  # noqa: E402
    coordinate_array_identity_sha256,
)


DATASET_ID = "drivaerml"
DATASET_REPOSITORY = "neashton/drivaerml"
DATASET_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
CONSTANT_PROFILE_SHA256 = (
    "df22bc807b62f925c32659d681ac44064e6acf46449038b8431b1e9139aba1e8"
)
RUN419_SELECTED_VALUES_SHA256 = (
    "a1cd9c5bad71b720e6434fbb821aa480fc2f7555516375329bfd02ced43752d0"
)

CASE_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-case-v3"
CHUNK_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-chunk-v3"
INDEX_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-index-v3"
SPLIT_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-split-index-v3"
PROVENANCE_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-provenance-v3"
RELEASE_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-release-v3"
SCHEMA_VERSION = "3.0"

CONSTANT_VELOCITY_FAMILY = "drivaerml-autocfd5-constant-v1"
RELATIVE_VELOCITY_FAMILY = "drivaerml-velocity-relative-v3"
CONSTANT_CP_FAMILY = "drivaerml_cp_constant_v1"
RELATIVE_CP_FAMILY = "drivaerml_cp_relative_v1"
VELOCITY_STATIONS = (
    "V1", "V2", "V3", "V4", "V5", "V6",
    "U1", "U2", "U3", "U4", "U5", "U6",
    "L1", "R1", "R2", "R3",
)
VELOCITY_SAMPLE_COUNTS = {
    **{station: 201 for station in VELOCITY_STATIONS[:6]},
    **{station: 301 for station in VELOCITY_STATIONS[6:12]},
    "L1": 651,
    "R1": 31,
    "R2": 31,
    "R3": 31,
}
CONSTANT_CP_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
    "sidewall_z_0_15",
    "front_left_wheelhouse_y_neg_0_6",
)
RELATIVE_CP_ALIAS_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
)
RELATIVE_CP_MOVING_STATIONS = (
    "sidewall_front_wheelhouse_relative",
    "front_left_wheelhouse_relative",
)
EXPECTED_SPLITS = (
    "full",
    "medium",
    "scarce",
    "super_scarce",
    "geometry",
    "high_drag",
    "low_drag",
    "rear_separation",
)
MAPPING_ROW_FIELDS = [
    "profile_id",
    "sample_index",
    "point_m",
    "distance_m",
    "valid",
    "reason",
    "raw_vtk_cell_id",
    "candidate_count",
]
RELATIVE_COORDINATE_FIELDS = [
    "family_id",
    "placement_mode",
    "profile_id",
    "sample_index",
    "point_count",
    "line_fraction",
    "distance_m",
    "x_m",
    "y_m",
    "z_m",
]
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
CASE_ID_RE = re.compile(r"run_([1-9][0-9]*)\Z")
VALUE_ARRAY_DOMAIN = b"fluidsbench-drivaerml-native-value-array-v1\x00"
INTEGER_ARRAY_DOMAIN = b"fluidsbench-drivaerml-native-id-array-v1\x00"
SERIES_IDENTITY_SCHEMA_V1 = "fluidsbench-drivaerml-native-series-identity-v1"
SERIES_IDENTITY_SCHEMA_V2 = "fluidsbench-drivaerml-native-series-identity-v2"
CP_DISPLAY_COORDINATE_ID = "streamwise_x_m"
CP_DISPLAY_COORDINATE_UNIT = "m"
CP_DISPLAY_COORDINATE_DEFINITION = (
    "retained_plane_intersection_segment_endpoint_midpoint_x"
)
TRUTH_SOURCE = {
    "source_kind": "native_cfd",
    "analytical_dummy": False,
    "native_quantity_source": "pinned_drivaerml_cell_data",
}
EXPECTED_ALL484_COVERAGE = {
    CONSTANT_VELOCITY_FAMILY: {
        "materialized_series_count": 484 * 16,
        "shared_alias_series_count": 0,
        "sample_count": 1_766_227,
        "unsupported_sample_count": 51_677,
        "display_coordinate_sample_count": 0,
    },
    RELATIVE_VELOCITY_FAMILY: {
        "materialized_series_count": 484 * 16,
        "shared_alias_series_count": 0,
        "sample_count": 1_779_592,
        "unsupported_sample_count": 38_312,
        "display_coordinate_sample_count": 0,
    },
    CONSTANT_CP_FAMILY: {
        "materialized_series_count": 484 * 4,
        "shared_alias_series_count": 0,
        "sample_count": 2_768_745,
        "unsupported_sample_count": 0,
        "display_coordinate_sample_count": 2_768_745,
    },
    RELATIVE_CP_FAMILY: {
        "materialized_series_count": 484 * 2,
        "shared_alias_series_count": 484 * 2,
        "sample_count": 739_775,
        "unsupported_sample_count": 0,
        "display_coordinate_sample_count": 739_775,
    },
}

RELATIVE_PLACEMENT_MANIFEST = (
    "velocity_support_v3/production_campaign_v1/aggregate/"
    "relative-velocity-v3-production-all484-inputs-v1.json"
)
RELATIVE_MAPPING_MANIFEST = (
    "velocity_mapping_v3/aggregate/"
    "relative-velocity-v3-mapping-all484-v1.json"
)
RELATIVE_CP_MANIFEST = (
    "cp_support/campaign_v3/aggregate_v3/all484/"
    "relative-cp-native-support-manifest-v3.json"
)
CONSTANT_CP_AGGREGATE = (
    "aggregate/continuous-cp-cut-all484-aggregate-candidate-v8.json"
)


class ExportError(ValueError):
    """Raised when input evidence or output state differs from the contract."""


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExportError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def load_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ExportError(f"cannot read {label}: {error}") from error
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExportError(f"{label} contains non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ExportError(f"{label} must contain one JSON object")
    return value, payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise ExportError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ExportError(f"{label} must be a lowercase SHA-256")
    if value == "0" * 64:
        raise ExportError(f"{label} cannot be all zeroes")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExportError(f"{label} must be finite")
    return result


def positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExportError(f"{label} must be a positive integer")
    return value


def case_id(value: object, label: str = "case_id") -> str:
    if not isinstance(value, str) or CASE_ID_RE.fullmatch(value) is None:
        raise ExportError(f"{label} must be run_N")
    return value


def binary64_array_identity_sha256(values: Iterable[object]) -> str:
    raw = list(values)
    encoded = bytearray(VALUE_ARRAY_DOMAIN)
    encoded.extend(struct.pack(">Q", len(raw)))
    for index, item in enumerate(raw):
        value = finite(item, f"value[{index}]")
        if value == 0.0:
            value = 0.0
        encoded.extend(struct.pack(">d", value))
    return sha256_bytes(bytes(encoded))


def integer_array_identity_sha256(values: Iterable[object]) -> str:
    raw = list(values)
    encoded = bytearray(INTEGER_ARRAY_DOMAIN)
    encoded.extend(struct.pack(">Q", len(raw)))
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ExportError(f"integer value[{index}] must be non-negative")
        encoded.extend(struct.pack(">Q", item))
    return sha256_bytes(bytes(encoded))


def series_identity_sha256(series: Mapping[str, object]) -> str:
    """Bind a series through compact descriptors and component identities."""

    representation = series.get("representation")
    common = {
        key: series[key]
        for key in (
            "panel_id",
            "family_id",
            "placement_mode",
            "station_id",
            "quantity_id",
            "quantity",
            "units",
            "scoring_role",
            "representation",
            "placement_receipt_identity_sha256",
        )
    }
    projection: dict[str, object] = {"schema": SERIES_IDENTITY_SCHEMA_V1, **common}
    if representation == "materialized":
        sample_index = series.get("sample_index")
        raw_ids = series.get("raw_native_cell_id")
        coordinate = series.get("coordinate")
        values = series.get("value")
        if not all(isinstance(item, list) for item in (sample_index, raw_ids, coordinate, values)):
            raise ExportError("materialized series arrays must be lists")
        count = len(sample_index)  # type: ignore[arg-type]
        if not (len(raw_ids) == len(coordinate) == len(values) == count):  # type: ignore[arg-type]
            raise ExportError("materialized series arrays must be aligned")
        projection.update(
            {
                "support_identity_sha256": series["support_identity_sha256"],
                "coordinate_id": series["coordinate_id"],
                "coordinate_unit": series["coordinate_unit"],
                "sample_count": count,
                "sample_index_identity_sha256": integer_array_identity_sha256(sample_index),  # type: ignore[arg-type]
                "raw_native_cell_id_identity_sha256": integer_array_identity_sha256(raw_ids),  # type: ignore[arg-type]
                "coordinate_identity_sha256": series["coordinate_identity_sha256"],
                "value_identity_sha256": series["value_identity_sha256"],
                "segments": series["segments"],
                "unsupported_samples": series["unsupported_samples"],
            }
        )
        display_keys = {
            "display_coordinate_id",
            "display_coordinate_unit",
            "display_coordinate",
            "display_coordinate_identity_sha256",
        }
        present_display_keys = display_keys.intersection(series)
        if present_display_keys:
            if present_display_keys != display_keys:
                raise ExportError("materialized display coordinate fields must be complete")
            display_coordinate = series["display_coordinate"]
            if not isinstance(display_coordinate, list) or len(display_coordinate) != count:
                raise ExportError("materialized display coordinate must be aligned")
            projection["schema"] = SERIES_IDENTITY_SCHEMA_V2
            projection.update(
                {
                    "display_coordinate_id": series["display_coordinate_id"],
                    "display_coordinate_unit": series["display_coordinate_unit"],
                    "display_coordinate_identity_sha256": series[
                        "display_coordinate_identity_sha256"
                    ],
                }
            )
    elif representation == "shared_alias":
        shared_support_ref = series.get("shared_support_ref")
        if not isinstance(shared_support_ref, Mapping) or set(shared_support_ref) != {
            "canonical_family_id",
            "canonical_station_id",
            "canonical_support_identity_sha256",
            "shared_support_id",
        }:
            raise ExportError("shared alias support reference shape differs")
        digest(
            shared_support_ref.get("canonical_support_identity_sha256"),
            "shared alias canonical support identity",
        )
        projection["shared_support_ref"] = dict(shared_support_ref)
    else:
        raise ExportError("series representation differs")
    return sha256_bytes(canonical_json_bytes(projection))


def document_identity_sha256(document: Mapping[str, object], field: str) -> str:
    body = dict(document)
    identity = body.pop(field, None)
    if not isinstance(identity, dict):
        raise ExportError(f"{field} must be an identity object")
    expected = digest(identity.get("sha256"), f"{field}.sha256")
    actual = sha256_bytes(canonical_json_bytes(body))
    if actual != expected:
        raise ExportError(f"{field} does not replay")
    return expected


def identity_bound_document(body: Mapping[str, object], field: str) -> dict[str, object]:
    result = dict(body)
    result[field] = {
        "algorithm": "sha256",
        "scope": f"canonical_json_body_without_{field}",
        "sha256": sha256_bytes(canonical_json_bytes(result)),
    }
    return result


def write_canonical(path: Path, value: object, *, check: bool) -> dict[str, object]:
    payload = canonical_json_bytes(value)
    expected = sha256_bytes(payload)
    if check:
        try:
            current = path.read_bytes()
        except OSError as error:
            raise ExportError(f"cannot check {path}: {error}") from error
        if current != payload:
            raise ExportError(f"{path} differs from deterministic replay")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return {"path": path.name, "sha256": expected, "size_bytes": len(payload)}


def stable_stat(path: Path) -> dict[str, int]:
    try:
        value = path.stat()
    except OSError as error:
        raise ExportError(f"cannot stat {path}: {error}") from error
    if not path.is_file():
        raise ExportError(f"{path} is not a regular file")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def selected_array_evidence_sha256(raw_ids: np.ndarray, values: np.ndarray) -> str:
    """Replay SparseNativeField.audit_record selected-values identity."""

    ids = np.asarray(raw_ids, dtype="<i8")
    array = np.asarray(values, dtype="<f8")
    evidence = hashlib.sha256()
    evidence.update(
        json.dumps(
            {"ids_shape": ids.shape, "values_shape": array.shape},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    evidence.update(ids.tobytes(order="C"))
    evidence.update(array.tobytes(order="C"))
    return evidence.hexdigest()


def compact_segments(
    sample_indices: Sequence[int],
    coordinates: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    if len(sample_indices) != len(coordinates) or not sample_indices:
        raise ExportError("segment inputs must be non-empty and aligned")
    if labels is None:
        labels = ("supported",) * len(sample_indices)
    if len(labels) != len(sample_indices):
        raise ExportError("segment labels must align with samples")
    ranges: list[dict[str, object]] = []
    start = 0
    for position in range(1, len(sample_indices) + 1):
        boundary = position == len(sample_indices)
        if not boundary:
            boundary = (
                sample_indices[position] != sample_indices[position - 1] + 1
                or labels[position] != labels[position - 1]
            )
        if boundary:
            ranges.append(
                {
                    "segment_id": labels[start],
                    "emitted_index_start": start,
                    "emitted_index_stop": position,
                    "sample_index_start": sample_indices[start],
                    "sample_index_stop": sample_indices[position - 1] + 1,
                    "coordinate_start": coordinates[start],
                    "coordinate_stop": coordinates[position - 1],
                }
            )
            start = position
    return ranges


def _xml_attributes(raw: bytes, label: str) -> tuple[str, dict[str, str], bool]:
    if raw.startswith(b"<?") or raw.startswith(b"<!"):
        return "special", {}, True
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ExportError(f"{label} XML tag is not ASCII") from error
    match = re.fullmatch(r"<\s*(/?)\s*([A-Za-z0-9_:.-]+)(.*?)>", text, re.S)
    if match is None:
        raise ExportError(f"cannot parse {label} XML tag")
    closing = bool(match.group(1))
    name = match.group(2)
    tail = match.group(3)
    attributes: dict[str, str] = {}
    for item in re.finditer(r"([A-Za-z0-9_:.-]+)\s*=\s*(['\"])(.*?)\2", tail, re.S):
        key = item.group(1)
        if key in attributes:
            raise ExportError(f"duplicate {label} XML attribute {key}")
        attributes[key] = item.group(3)
    return (f"/{name}" if closing else name), attributes, tail.rstrip().endswith("/")


def _next_xml_tag(stream: BinaryIO, *, max_scan_bytes: int = 16 * 1024 * 1024) -> bytes:
    scanned = 0
    while True:
        byte = stream.read(1)
        if not byte:
            raise ExportError("unexpected end of VTK XML while locating a tag")
        scanned += 1
        if byte == b"<":
            break
        if scanned > max_scan_bytes:
            raise ExportError("VTK XML markup gap exceeds the safety limit")
    raw = bytearray(b"<")
    quote: int | None = None
    while True:
        byte = stream.read(1)
        if not byte:
            raise ExportError("VTK XML tag is truncated")
        raw.extend(byte)
        value = byte[0]
        if quote is None and value in (34, 39):
            quote = value
        elif quote == value:
            quote = None
        elif quote is None and value == 62:
            return bytes(raw)
        if len(raw) > 1024 * 1024:
            raise ExportError("VTK XML tag exceeds the safety limit")


def _seek_binary_pattern(
    stream: BinaryIO,
    pattern: bytes,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> bool:
    """Seek to a literal pattern without byte-wise scanning large base64 arrays."""

    if not pattern:
        raise ExportError("binary seek pattern cannot be empty")
    overlap = b""
    while block := stream.read(chunk_bytes):
        block_start = stream.tell() - len(block)
        combined = overlap + block
        found = combined.find(pattern)
        if found >= 0:
            stream.seek(block_start - len(overlap) + found)
            return True
        overlap = combined[-(len(pattern) - 1) :] if len(pattern) > 1 else b""
    return False


@dataclass(frozen=True)
class DirectInlineArray:
    byte_order: str
    header_type: str
    vtk_type: str
    name: str
    components: int
    tuple_count: int
    declared_payload_bytes: int
    encoded_start: int
    total_encoded_chars: int
    closing_tag_end: int


def locate_uncompressed_cell_field(
    stream: BinaryIO,
    *,
    field_name: str,
    expected_components: int,
    expected_tuple_count: int | None,
    expected_dataset_type: str = "UnstructuredGrid",
    piece_tuple_attribute: str = "NumberOfCells",
) -> DirectInlineArray:
    """Locate an inline field while seeking over every preceding base64 body."""

    if not stream.seekable() or not stream.readable():
        raise ExportError("native VTK stream must be seekable and readable")
    stream.seek(0)
    byte_order: str | None = None
    header_type: str | None = None
    compressor: str | None = None
    association: str | None = None
    piece_cells: int | None = None
    while True:
        tag = _next_xml_tag(stream)
        name, attributes, self_closing = _xml_attributes(tag, "VTK")
        if name == "VTKFile":
            byte_order = attributes.get("byte_order")
            header_type = attributes.get("header_type")
            compressor = attributes.get("compressor")
            if attributes.get("type") != expected_dataset_type:
                raise ExportError(
                    f"native VTK dataset must be {expected_dataset_type}"
                )
            if byte_order not in {"LittleEndian", "BigEndian"} or header_type != "UInt64":
                raise ExportError("native VTK byte/header order differs")
            if compressor is not None:
                raise ExportError("native VTK inline arrays must be uncompressed")
        elif name == "Piece":
            piece_cells = int(attributes.get(piece_tuple_attribute, "-1"))
            if piece_cells < 1 or (
                expected_tuple_count is not None
                and piece_cells != expected_tuple_count
            ):
                raise ExportError("native VTK Piece tuple count differs")
            # Some public PolyData files contain a malformed byte-count header
            # on a preceding connectivity array.  Base64 cannot contain '<',
            # so find CellData markup in large binary bodies without trusting
            # unrelated array byte counts; the selected field's own header is
            # still parsed and verified below.
            if expected_dataset_type == "PolyData":
                if not _seek_binary_pattern(stream, b"<CellData"):
                    raise ExportError("native PolyData CellData is missing")
                cell_tag = _next_xml_tag(stream)
                cell_name, _, _ = _xml_attributes(cell_tag, "CellData")
                if cell_name != "CellData":
                    raise ExportError("native PolyData CellData markup differs")
                while True:
                    if not _seek_binary_pattern(stream, b"<"):
                        raise ExportError(
                            f"native PolyData CellData field {field_name} is missing"
                        )
                    data_tag = _next_xml_tag(stream, max_scan_bytes=4096)
                    data_name, data_attributes, data_self_closing = _xml_attributes(
                        data_tag, "PolyData CellData"
                    )
                    if data_name == "/CellData":
                        raise ExportError(
                            f"native PolyData CellData field {field_name} is missing"
                        )
                    if data_name != "DataArray" or data_self_closing:
                        continue
                    if data_attributes.get("Name") != field_name:
                        continue
                    if data_attributes.get("format") != "binary":
                        raise ExportError("native VTK DataArray is not inline binary")
                    while True:
                        byte = stream.read(1)
                        if not byte:
                            raise ExportError("native VTK DataArray body is missing")
                        if byte not in b" \t\r\n":
                            stream.seek(-1, os.SEEK_CUR)
                            break
                    encoded_start = stream.tell()
                    encoded_header = stream.read(12)
                    try:
                        header = base64.b64decode(encoded_header, validate=True)
                    except ValueError as error:
                        raise ExportError(
                            "native VTK UInt64 base64 header differs"
                        ) from error
                    if len(header) != 9:
                        raise ExportError("native VTK UInt64 header length differs")
                    prefix = "<" if byte_order == "LittleEndian" else ">"
                    declared_payload_bytes = struct.unpack(prefix + "Q", header[:8])[0]
                    total_encoded_chars = 4 * (
                        (8 + declared_payload_bytes + 2) // 3
                    )
                    stream.seek(encoded_start + total_encoded_chars)
                    closing = _next_xml_tag(stream, max_scan_bytes=4096)
                    closing_name, _, _ = _xml_attributes(
                        closing, "DataArray closing"
                    )
                    if closing_name != "/DataArray":
                        raise ExportError(
                            "native VTK DataArray payload boundary differs"
                        )
                    result = DirectInlineArray(
                        byte_order=byte_order or "",
                        header_type=header_type or "",
                        vtk_type=data_attributes.get("type", ""),
                        name=data_attributes.get("Name", ""),
                        components=int(
                            data_attributes.get("NumberOfComponents", "1")
                        ),
                        tuple_count=piece_cells,
                        declared_payload_bytes=declared_payload_bytes,
                        encoded_start=encoded_start,
                        total_encoded_chars=total_encoded_chars,
                        closing_tag_end=stream.tell(),
                    )
                    if (
                        result.vtk_type != "Float32"
                        or result.components != expected_components
                        or result.tuple_count < 1
                        or (
                            expected_tuple_count is not None
                            and result.tuple_count != expected_tuple_count
                        )
                        or result.declared_payload_bytes
                        != result.tuple_count * expected_components * 4
                    ):
                        raise ExportError(f"native {field_name} declaration differs")
                    return result
        elif name in {"PointData", "CellData", "FieldData"}:
            association = name
        elif name in {"/PointData", "/CellData", "/FieldData"}:
            association = None
        elif name == "DataArray" and not self_closing:
            if byte_order is None or header_type is None:
                raise ExportError("DataArray precedes VTK source declaration")
            if attributes.get("format") != "binary":
                raise ExportError("native VTK DataArray is not inline binary")
            while True:
                byte = stream.read(1)
                if not byte:
                    raise ExportError("native VTK DataArray body is missing")
                if byte not in b" \t\r\n":
                    stream.seek(-1, os.SEEK_CUR)
                    break
            encoded_start = stream.tell()
            encoded_header = stream.read(12)
            try:
                header = base64.b64decode(encoded_header, validate=True)
            except ValueError as error:
                raise ExportError("native VTK UInt64 base64 header differs") from error
            # VTK encodes UInt64 header plus payload as one continuous base64
            # stream.  Twelve encoded characters therefore decode nine bytes:
            # the eight-byte length and the first payload byte.
            if len(header) != 9:
                raise ExportError("native VTK UInt64 header length differs")
            prefix = "<" if byte_order == "LittleEndian" else ">"
            declared_payload_bytes = struct.unpack(prefix + "Q", header[:8])[0]
            total_encoded_chars = 4 * ((8 + declared_payload_bytes + 2) // 3)
            stream.seek(encoded_start + total_encoded_chars)
            closing = _next_xml_tag(stream, max_scan_bytes=4096)
            closing_name, _, _ = _xml_attributes(closing, "DataArray closing")
            if closing_name != "/DataArray":
                raise ExportError("native VTK DataArray payload boundary differs")
            result = DirectInlineArray(
                byte_order=byte_order,
                header_type=header_type,
                vtk_type=attributes.get("type", ""),
                name=attributes.get("Name", ""),
                components=int(attributes.get("NumberOfComponents", "1")),
                tuple_count=piece_cells if association == "CellData" and piece_cells else -1,
                declared_payload_bytes=declared_payload_bytes,
                encoded_start=encoded_start,
                total_encoded_chars=total_encoded_chars,
                closing_tag_end=stream.tell(),
            )
            if association == "CellData" and result.name == field_name:
                if (
                    result.vtk_type != "Float32"
                    or result.components != expected_components
                    or result.tuple_count < 1
                    or (
                        expected_tuple_count is not None
                        and result.tuple_count != expected_tuple_count
                    )
                    or result.declared_payload_bytes
                    != result.tuple_count * expected_components * 4
                ):
                    raise ExportError(f"native {field_name} declaration differs")
                return result


def direct_sparse_float32(
    stream: BinaryIO,
    array: DirectInlineArray,
    raw_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Gather exact tuples by direct seeks into an unwrapped base64 payload."""

    ids = np.unique(np.asarray(raw_ids, dtype=np.int64))
    if ids.ndim != 1 or len(ids) < 1 or np.any(ids < 0) or np.any(ids >= array.tuple_count):
        raise ExportError("mapped raw native IDs exceed the native field")
    if array.components < 1 or array.vtk_type != "Float32":
        raise ExportError("direct sparse gather requires a Float32 array")
    prefix = "<" if array.byte_order == "LittleEndian" else ">"
    tuple_bytes = 4 * array.components
    values = np.empty((len(ids), array.components), dtype=np.float64)
    for position, raw_id in enumerate(ids):
        raw_offset = 8 + tuple_bytes * int(raw_id)
        encoded_offset = 4 * (raw_offset // 3)
        decoded_offset = raw_offset % 3
        stream.seek(array.encoded_start + encoded_offset)
        encoded_count = 4 * ((decoded_offset + tuple_bytes + 2) // 3)
        encoded = stream.read(encoded_count)
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ExportError(f"raw native cell {raw_id} base64 tuple differs") from error
        payload = payload[decoded_offset : decoded_offset + tuple_bytes]
        if len(payload) != tuple_bytes:
            raise ExportError(f"raw native cell {raw_id} tuple length differs")
        values[position] = struct.unpack(prefix + "f" * array.components, payload)
    if not np.all(np.isfinite(values)):
        raise ExportError("selected native UMeanTrim values contain non-finite values")
    return ids, values


def direct_sparse_float32x3(
    stream: BinaryIO,
    array: DirectInlineArray,
    raw_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if array.components != 3:
        raise ExportError("direct sparse Float32x3 gather has wrong component count")
    return direct_sparse_float32(stream, array, raw_ids)


def direct_sparse_float32x1(
    stream: BinaryIO,
    array: DirectInlineArray,
    raw_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if array.components != 1:
        raise ExportError("direct sparse Float32 scalar gather has wrong component count")
    ids, values = direct_sparse_float32(stream, array, raw_ids)
    return ids, values[:, 0]


@dataclass(frozen=True)
class InputContext:
    native_source_pin: Path
    dataset_root: Path
    constant_velocity_receipts_root: Path
    relative_producer_root: Path
    constant_cp_campaign_root: Path
    native_volume_audit: Path
    output_dir: Path
    evaluator_git_revision: str
    official_case_ids: tuple[str, ...]
    pin_sha256: str
    pin_by_case: Mapping[str, Mapping[str, object]]
    native_audit_by_case: Mapping[str, Mapping[str, object]]
    relative_placement_by_case: Mapping[str, Mapping[str, object]]
    relative_mapping_by_case: Mapping[str, Mapping[str, object]]
    relative_cp_by_case: Mapping[str, Mapping[str, object]]
    constant_cp_by_case: Mapping[str, Mapping[str, object]]
    global_bindings: Mapping[str, Mapping[str, object]]
    split_indexes: Mapping[str, Mapping[str, object]]
    split_bindings: Mapping[str, Mapping[str, object]]


def _case_rows(
    document: Mapping[str, object], label: str, official: Sequence[str]
) -> dict[str, Mapping[str, object]]:
    rows = document.get("cases")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ExportError(f"{label} cases must be an object array")
    observed = [row.get("case_id") for row in rows]
    if observed != list(official):
        raise ExportError(f"{label} does not cover the exact ordered official case set")
    return {str(row["case_id"]): row for row in rows}


def _exact_case_directories(root: Path, official: Sequence[str], label: str) -> None:
    if not root.is_dir():
        raise ExportError(f"{label} root is not a directory: {root}")
    observed = {
        path.name
        for path in root.glob("run_*")
        if path.is_dir() and CASE_ID_RE.fullmatch(path.name)
    }
    expected = set(official)
    if observed != expected:
        missing = sorted(expected - observed, key=lambda item: int(item[4:]))
        extra = sorted(observed - expected, key=lambda item: int(item[4:]))
        raise ExportError(f"{label} case directories differ; missing={missing} extra={extra}")


def _file_binding(path: Path, logical_path: str) -> dict[str, object]:
    if not path.is_file():
        raise ExportError(f"required input is missing: {path}")
    return {
        "path": logical_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _parse_split_arguments(
    values: Sequence[str], official: Sequence[str]
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    paths: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ExportError("--split-index must be SPLIT_ID=PATH")
        split_id, raw_path = raw.split("=", 1)
        if split_id in paths or split_id not in EXPECTED_SPLITS or not raw_path:
            raise ExportError(f"invalid or duplicate split binding {raw!r}")
        paths[split_id] = Path(raw_path).expanduser().resolve()
    if set(paths) != set(EXPECTED_SPLITS):
        raise ExportError(
            f"assemble requires exactly these split IDs: {list(EXPECTED_SPLITS)}"
        )
    documents: dict[str, Mapping[str, object]] = {}
    bindings: dict[str, Mapping[str, object]] = {}
    official_set = set(official)
    for split_id in EXPECTED_SPLITS:
        path = paths[split_id]
        document, payload = load_json(path, f"{split_id} split index")
        split_cases = document.get("case_ids")
        if (
            document.get("schema_version") != "1.0"
            or document.get("dataset_id") != DATASET_ID
            or document.get("split_id") != split_id
            or document.get("case_id_status") != "official"
            or not isinstance(split_cases, list)
            or not all(isinstance(item, str) for item in split_cases)
            or len(split_cases) != len(set(split_cases))
            or document.get("case_count") != len(split_cases)
            or not set(split_cases).issubset(official_set)
        ):
            raise ExportError(f"{split_id} split index declaration differs")
        documents[split_id] = document
        bindings[split_id] = {
            "path": path.name,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    return documents, bindings


def _git_revision(explicit: str | None) -> str:
    if explicit is None:
        try:
            explicit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ExportError("cannot resolve evaluator Git revision") from error
    if GIT_COMMIT_RE.fullmatch(explicit) is None:
        raise ExportError("evaluator Git revision must be a full lowercase commit ID")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{explicit}^{{commit}}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExportError("evaluator Git revision does not resolve locally") from error
    return explicit


def load_context(args: argparse.Namespace) -> InputContext:
    pin_path = args.native_source_pin.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    constant_velocity_root = args.constant_velocity_receipts_root.expanduser().resolve()
    relative_root = args.relative_producer_root.expanduser().resolve()
    constant_cp_root = args.constant_cp_campaign_root.expanduser().resolve()
    native_audit_path = args.native_volume_audit.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    pin, pin_payload = load_json(pin_path, "native source pin")
    pin_sha = sha256_bytes(pin_payload)
    cases = pin.get("cases")
    repository = pin.get("repository")
    if (
        pin_sha != NATIVE_SOURCE_PIN_SHA256
        or pin.get("schema") != "drivaerml-fluidsbench-public-native-source-pin-v1"
        or pin.get("schema_version") != 1
        or not isinstance(repository, dict)
        or repository.get("repo_id") != DATASET_REPOSITORY
        or repository.get("revision") != DATASET_REVISION
        or not isinstance(cases, list)
        or not all(isinstance(row, dict) for row in cases)
    ):
        raise ExportError("native source pin identity or declaration differs")
    official = tuple(case_id(row.get("case_id")) for row in cases)
    if len(official) != 484 or len(set(official)) != 484:
        raise ExportError("native source pin must contain exactly 484 unique cases")
    pin_by_case = {str(row["case_id"]): row for row in cases}

    native_audit, native_audit_payload = load_json(
        native_audit_path, "retained native volume audit"
    )
    if (
        native_audit.get("schema")
        != "drivaerml-native-volume-equal-cell-primary-all-case-audit-v1"
        or native_audit.get("schema_version") != 1
        or native_audit.get("status") != "passed_all_case_equal_cell_primary_audit"
        or native_audit.get("source")
        != {
            "repository_id": DATASET_REPOSITORY,
            "immutable_revision": DATASET_REVISION,
            "native_source_pin_sha256": pin_sha,
        }
    ):
        raise ExportError("retained native volume audit declaration differs")
    native_by_case = _case_rows(native_audit, "native volume audit", official)

    placement_path = relative_root / RELATIVE_PLACEMENT_MANIFEST
    mapping_path = relative_root / RELATIVE_MAPPING_MANIFEST
    relative_cp_path = relative_root / RELATIVE_CP_MANIFEST
    constant_cp_path = constant_cp_root / CONSTANT_CP_AGGREGATE
    placement, placement_payload = load_json(placement_path, "relative velocity placement manifest")
    mapping, mapping_payload = load_json(mapping_path, "relative velocity mapping manifest")
    relative_cp, relative_cp_payload = load_json(relative_cp_path, "relative Cp manifest")
    constant_cp, constant_cp_payload = load_json(constant_cp_path, "constant Cp aggregate")

    if (
        placement.get("schema")
        != "drivaerml-relative-velocity-v3-production-input-manifest-v1"
        or placement.get("schema_version") != 1
        or placement.get("case_count") != 484
        or placement.get("case_ids") != list(official)
        or placement.get("dataset_id") != DATASET_ID
        or placement.get("public_dataset_revision") != DATASET_REVISION
        or placement.get("family_id") != RELATIVE_VELOCITY_FAMILY
    ):
        raise ExportError("relative velocity placement manifest differs")
    if (
        mapping.get("schema") != "drivaerml-velocity-relative-v3-mapping-aggregate-v1"
        or mapping.get("schema_version") != 1
        or mapping.get("official_case_count") != 484
        or mapping.get("included_case_count") != 484
        or mapping.get("complete_official_case_coverage") is not True
        or mapping.get("dataset_id") != DATASET_ID
        or mapping.get("public_dataset_revision") != DATASET_REVISION
        or mapping.get("family_id") != RELATIVE_VELOCITY_FAMILY
    ):
        raise ExportError("relative velocity mapping manifest differs")
    if (
        relative_cp.get("schema") != "drivaerml-relative-cp-native-support-manifest-v3"
        or relative_cp.get("schema_version") != 3
        or relative_cp.get("case_count") != 484
        or relative_cp.get("all_official_cases_generated_and_replayed") is not True
        or relative_cp.get("dataset_id") != DATASET_ID
        or relative_cp.get("public_dataset_revision") != DATASET_REVISION
        or relative_cp.get("family_id") != RELATIVE_CP_FAMILY
        or relative_cp.get("moving_station_count_per_case") != 2
        or relative_cp.get("shared_constant_centerline_alias_count_per_case") != 2
    ):
        raise ExportError("relative Cp manifest differs or is incomplete")
    document_identity_sha256(relative_cp, "document_identity")
    if (
        constant_cp.get("schema")
        != "drivaerml-continuous-cp-cut-all484-aggregate-candidate-v8"
        or constant_cp.get("case_count") != 484
        or constant_cp.get("case_order") != "native_source_pin_order"
        or constant_cp.get("dataset_id") != DATASET_ID
        or constant_cp.get("native_source_pin_sha256") != pin_sha
    ):
        raise ExportError("constant Cp aggregate differs")
    document_identity_sha256(constant_cp, "aggregate_identity")

    placement_by_case = _case_rows(placement, "relative velocity placement", official)
    mapping_by_case = _case_rows(mapping, "relative velocity mapping", official)
    relative_cp_by_case = _case_rows(relative_cp, "relative Cp manifest", official)
    constant_cp_by_case = _case_rows(constant_cp, "constant Cp aggregate", official)

    _exact_case_directories(constant_velocity_root, official, "constant velocity")
    _exact_case_directories(
        relative_root / "velocity_support_v3/production_campaign_v1/cases",
        official,
        "relative velocity placement",
    )
    _exact_case_directories(
        relative_root / "velocity_mapping_v3/cases",
        official,
        "relative velocity mapping",
    )
    _exact_case_directories(
        relative_root / "cp_support/campaign_v3/native_support_v3/cases",
        official,
        "relative Cp",
    )
    _exact_case_directories(constant_cp_root / "cases", official, "constant Cp")

    split_indexes: dict[str, Mapping[str, object]] = {}
    split_bindings: dict[str, Mapping[str, object]] = {}
    if args.mode in {"assemble", "all"}:
        split_indexes, split_bindings = _parse_split_arguments(args.split_index, official)

    global_bindings = {
        "native_source_pin": {
            "path": pin_path.name,
            "sha256": pin_sha,
            "size_bytes": len(pin_payload),
        },
        "native_volume_audit": {
            "path": native_audit_path.name,
            "sha256": sha256_bytes(native_audit_payload),
            "size_bytes": len(native_audit_payload),
        },
        "relative_velocity_placement_manifest": {
            "path": RELATIVE_PLACEMENT_MANIFEST,
            "sha256": sha256_bytes(placement_payload),
            "size_bytes": len(placement_payload),
        },
        "relative_velocity_mapping_manifest": {
            "path": RELATIVE_MAPPING_MANIFEST,
            "sha256": sha256_bytes(mapping_payload),
            "size_bytes": len(mapping_payload),
        },
        "relative_cp_manifest": {
            "path": RELATIVE_CP_MANIFEST,
            "sha256": sha256_bytes(relative_cp_payload),
            "size_bytes": len(relative_cp_payload),
        },
        "constant_cp_aggregate": {
            "path": CONSTANT_CP_AGGREGATE,
            "sha256": sha256_bytes(constant_cp_payload),
            "size_bytes": len(constant_cp_payload),
        },
        "exporter_source": {
            "path": "scripts/export_drivaerml_native_profile_truth.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "size_bytes": Path(__file__).resolve().stat().st_size,
        },
    }
    return InputContext(
        native_source_pin=pin_path,
        dataset_root=dataset_root,
        constant_velocity_receipts_root=constant_velocity_root,
        relative_producer_root=relative_root,
        constant_cp_campaign_root=constant_cp_root,
        native_volume_audit=native_audit_path,
        output_dir=output_dir,
        evaluator_git_revision=_git_revision(args.evaluator_git_revision),
        official_case_ids=official,
        pin_sha256=pin_sha,
        pin_by_case=pin_by_case,
        native_audit_by_case=native_by_case,
        relative_placement_by_case=placement_by_case,
        relative_mapping_by_case=mapping_by_case,
        relative_cp_by_case=relative_cp_by_case,
        constant_cp_by_case=constant_cp_by_case,
        global_bindings=global_bindings,
        split_indexes=split_indexes,
        split_bindings=split_bindings,
    )


def _source_parts(pin_case: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    volume = pin_case.get("volume")
    if not isinstance(volume, dict) or not isinstance(volume.get("parts"), list):
        raise ExportError("native source pin volume declaration differs")
    output: list[dict[str, object]] = []
    segments: list[dict[str, object]] = []
    offset = 0
    for expected_index, raw in enumerate(volume["parts"]):
        if not isinstance(raw, dict):
            raise ExportError("native source part must be an object")
        index = raw.get("part_index")
        size = raw.get("size_bytes")
        sha = raw.get("lfs_sha256")
        path = raw.get("path")
        if (
            index != expected_index
            or not isinstance(size, int)
            or size < 1
            or not isinstance(path, str)
        ):
            raise ExportError("native source part declaration differs")
        sha = digest(sha, "native source part")
        output.append(
            {"part_index": index, "path": path, "size_bytes": size, "sha256": sha}
        )
        segments.append(
            {
                "byte_offset": offset,
                "part_index": index,
                "size_bytes": size,
                "sha256": sha,
            }
        )
        offset += size
    if volume.get("part_count") != len(output) or volume.get("total_size_bytes") != offset:
        raise ExportError("native source part totals differ")
    return output, segments


def _verify_native_binding(
    binding: object,
    *,
    case: str,
    pin_case: Mapping[str, object],
    pin_sha: str,
    label: str,
) -> None:
    if not isinstance(binding, dict):
        raise ExportError(f"{label} native source binding must be an object")
    _, segments = _source_parts(pin_case)
    volume = pin_case["volume"]
    if (
        binding.get("case_id") != case
        or binding.get("pin_sha256") != pin_sha
        or binding.get("repository_id") != DATASET_REPOSITORY
        or binding.get("repository_revision") != DATASET_REVISION
        or binding.get("logical_size_bytes") != volume["total_size_bytes"]
        or binding.get("ordered_verified_segments") != segments
    ):
        raise ExportError(f"{label} native source binding differs")


@dataclass(frozen=True)
class VerifiedVelocity:
    family_id: str
    placement_mode: str
    mapping_sha256: str
    mapping_receipt_sha256: str
    placement_receipt_sha256: str
    support_identity_by_station: Mapping[str, str]
    rows_by_station: Mapping[str, tuple[Mapping[str, object], ...]]
    input_bindings: Mapping[str, object]


def _expected_mapping_keys() -> list[tuple[str, int]]:
    return [
        (station, index)
        for station in VELOCITY_STATIONS
        for index in range(VELOCITY_SAMPLE_COUNTS[station])
    ]


def _assignment_evidence_identity(
    case: str,
    raw_rows: Sequence[Sequence[object]],
    source_sha256: Sequence[str],
) -> str:
    rows = [
        {
            "case_id": case,
            "profile_id": raw[0],
            "sample_index": raw[1],
            "point_m": raw[2],
            "distance_m": raw[3],
            "valid": raw[4],
            "reason": raw[5],
            "raw_vtk_cell_id": raw[6],
            "candidate_count": raw[7],
            "geometric_tolerance_m": 1e-6,
            "source_sha256": list(source_sha256),
        }
        for raw in raw_rows
    ]
    return sha256_bytes(canonical_json_bytes(rows, newline=False))


def _validate_mapping_rows(
    raw_rows: object,
    *,
    case: str,
    native_cell_count: int,
    coordinate_rows: Mapping[str, Sequence[Mapping[str, object]]] | None,
) -> tuple[dict[str, tuple[Mapping[str, object], ...]], dict[str, object]]:
    if not isinstance(raw_rows, list) or len(raw_rows) != 3756:
        raise ExportError(f"{case} velocity mapping must contain 3,756 rows")
    expected_keys = _expected_mapping_keys()
    observed_keys: list[tuple[str, int]] = []
    by_station: dict[str, list[Mapping[str, object]]] = {
        station: [] for station in VELOCITY_STATIONS
    }
    valid_count = 0
    invalid_count = 0
    invalid_reasons: collections.Counter[str] = collections.Counter()
    candidate_histogram: collections.Counter[str] = collections.Counter()
    valid_raw_ids: list[int] = []
    for position, raw in enumerate(raw_rows):
        if not isinstance(raw, list) or len(raw) != 8:
            raise ExportError(f"{case} velocity mapping row {position} shape differs")
        station, sample, point, distance, valid, reason, raw_id, candidates = raw
        if (
            station not in by_station
            or isinstance(sample, bool)
            or not isinstance(sample, int)
            or not isinstance(point, list)
            or len(point) != 3
            or not all(math.isfinite(finite(value, "velocity point")) for value in point)
            or not math.isfinite(finite(distance, "velocity distance"))
            or not isinstance(valid, bool)
            or isinstance(candidates, bool)
            or not isinstance(candidates, int)
            or candidates < 0
        ):
            raise ExportError(f"{case} velocity mapping row {position} differs")
        observed_keys.append((station, sample))
        coordinate_row = None if coordinate_rows is None else coordinate_rows[station][sample]
        if coordinate_row is not None and (
            point != coordinate_row["point_m"] or distance != coordinate_row["distance_m"]
        ):
            raise ExportError(f"{case} relative velocity row {position} differs from CSV")
        if valid:
            if (
                reason != ""
                or isinstance(raw_id, bool)
                or not isinstance(raw_id, int)
                or raw_id < 0
                or raw_id >= native_cell_count
                or candidates < 1
            ):
                raise ExportError(f"{case} valid velocity row {position} differs")
            valid_count += 1
            valid_raw_ids.append(raw_id)
        else:
            if not isinstance(reason, str) or not reason or raw_id is not None:
                raise ExportError(f"{case} invalid velocity row {position} differs")
            invalid_count += 1
            invalid_reasons[reason] += 1
        candidate_histogram[str(candidates)] += 1
        by_station[station].append(
            {
                "sample_index": sample,
                "point_m": point,
                "distance_m": distance,
                "line_fraction": None
                if coordinate_row is None
                else coordinate_row["line_fraction"],
                "valid": valid,
                "reason": reason,
                "raw_native_cell_id": raw_id,
                "candidate_count": candidates,
            }
        )
    if observed_keys != expected_keys:
        raise ExportError(f"{case} velocity station/sample order differs")
    return (
        {station: tuple(by_station[station]) for station in VELOCITY_STATIONS},
        {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "invalid_reason_counts": dict(invalid_reasons),
            "candidate_count_histogram": dict(candidate_histogram),
            "selected_raw_vtk_cell_id_min": min(valid_raw_ids),
            "selected_raw_vtk_cell_id_max": max(valid_raw_ids),
        },
    )


def verify_constant_velocity(
    context: InputContext, case: str, native_cell_count: int
) -> VerifiedVelocity:
    root = context.constant_velocity_receipts_root / case
    mapping_path = root / "velocity-cell-mapping-10mm.json"
    receipt_path = root / "receipt.json"
    mapping, mapping_payload = load_json(mapping_path, f"{case} constant velocity mapping")
    receipt, receipt_payload = load_json(receipt_path, f"{case} constant velocity receipt")
    mapping_sha = sha256_bytes(mapping_payload)
    receipt_sha = sha256_bytes(receipt_payload)
    artifacts = receipt.get("artifacts")
    ten_mm = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("nominal_spacing_mm") == 10
    ] if isinstance(artifacts, list) else []
    resolution = mapping.get("resolution")
    coverage = mapping.get("coverage")
    registries = mapping.get("registries")
    if (
        mapping.get("schema") != "drivaerml-velocity-cell-mapping-candidate-v1"
        or mapping.get("schema_version") != 1
        or mapping.get("case_id") != case
        or mapping.get("status")
        != "candidate_complete_geometry_mapping_not_activation_evidence"
        or mapping.get("row_fields") != MAPPING_ROW_FIELDS
        or not isinstance(resolution, dict)
        or resolution.get("sample_count") != 3756
        or resolution.get("line_count") != 16
        or resolution.get("nominal_spacing_mm") != 10
        or not isinstance(coverage, dict)
        or coverage.get("complete_duplicate_free_no_omissions") is not True
        or not isinstance(registries, dict)
        or registries.get("profile_sha256") != CONSTANT_PROFILE_SHA256
        or receipt.get("schema")
        != "drivaerml-velocity-cell-assignments-case-candidate-v2"
        or receipt.get("schema_version") != 2
        or receipt.get("case_id") != case
        or len(ten_mm) != 1
        or ten_mm[0].get("artifact") != mapping_path.name
        or ten_mm[0].get("sha256") != mapping_sha
        or ten_mm[0].get("size_bytes") != len(mapping_payload)
    ):
        raise ExportError(f"{case} constant velocity declaration differs")
    pin_case = context.pin_by_case[case]
    _verify_native_binding(
        mapping.get("native_source_binding"),
        case=case,
        pin_case=pin_case,
        pin_sha=context.pin_sha256,
        label="constant velocity mapping",
    )
    _verify_native_binding(
        receipt.get("native_source_binding"),
        case=case,
        pin_case=pin_case,
        pin_sha=context.pin_sha256,
        label="constant velocity receipt",
    )
    by_station, counts = _validate_mapping_rows(
        mapping.get("rows"),
        case=case,
        native_cell_count=native_cell_count,
        coordinate_rows=None,
    )
    for key in (
        "valid_count",
        "invalid_count",
        "invalid_reason_counts",
        "candidate_count_histogram",
        "selected_raw_vtk_cell_id_min",
        "selected_raw_vtk_cell_id_max",
    ):
        if coverage.get(key) != counts[key]:
            raise ExportError(f"{case} constant velocity coverage {key} differs")
    for key in ("valid_count", "invalid_count", "invalid_reason_counts"):
        if ten_mm[0].get(key) != counts[key]:
            raise ExportError(f"{case} constant velocity receipt {key} differs")
    source_parts, _ = _source_parts(pin_case)
    source_hashes = [str(item["sha256"]) for item in source_parts]
    assignment_sha = _assignment_evidence_identity(
        case, mapping["rows"], source_hashes  # type: ignore[arg-type]
    )
    if (
        assignment_sha != mapping.get("assignment_evidence_sha256")
        or assignment_sha != ten_mm[0].get("assignment_evidence_sha256")
    ):
        raise ExportError(f"{case} constant velocity assignment identity differs")

    support: dict[str, str] = {}
    for station in VELOCITY_STATIONS:
        identity_rows = [
            {
                "sample_index": row["sample_index"],
                "distance_m": row["distance_m"],
                "valid": row["valid"],
                "raw_vtk_cell_id": row["raw_native_cell_id"],
            }
            for row in by_station[station]
        ]
        support[station] = sha256_bytes(
            canonical_json_bytes(
                {
                    # Historical domain name is retained so run_419 remains
                    # byte-compatible with its evaluator-produced fixture.
                    "schema": "drivaerml-run419-constant-velocity-support-identity-v1",
                    "case_id": case,
                    "family_id": CONSTANT_VELOCITY_FAMILY,
                    "station_id": f"autocfd5_{station.lower()}",
                    "mapping_sha256": mapping_sha,
                    "ordered_rows": identity_rows,
                }
            )
        )
    return VerifiedVelocity(
        family_id=CONSTANT_VELOCITY_FAMILY,
        placement_mode="constant",
        mapping_sha256=mapping_sha,
        mapping_receipt_sha256=receipt_sha,
        placement_receipt_sha256=receipt_sha,
        support_identity_by_station=support,
        rows_by_station=by_station,
        input_bindings={
            "mapping": {"path": f"{case}/velocity-cell-mapping-10mm.json", "sha256": mapping_sha, "size_bytes": len(mapping_payload)},
            "receipt": {"path": f"{case}/receipt.json", "sha256": receipt_sha, "size_bytes": len(receipt_payload)},
            "profile_registry_sha256": CONSTANT_PROFILE_SHA256,
        },
    )


def _relative_coordinate_rows(
    path: Path,
    *,
    case: str,
) -> tuple[dict[str, tuple[Mapping[str, object], ...]], dict[str, str]]:
    rows: dict[str, list[Mapping[str, object]]] = {
        station: [] for station in VELOCITY_STATIONS
    }
    hashers = {station: hashlib.sha256() for station in VELOCITY_STATIONS}
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ExportError(f"cannot read {case} relative coordinate CSV: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RELATIVE_COORDINATE_FIELDS:
            raise ExportError(f"{case} relative coordinate CSV fields differ")
        for position, raw in enumerate(reader):
            station = raw["profile_id"]
            if (
                station not in rows
                or raw["family_id"] != RELATIVE_VELOCITY_FAMILY
                or raw["placement_mode"] != "relative"
            ):
                raise ExportError(f"{case} relative coordinate row {position} namespace differs")
            sample = int(raw["sample_index"])
            point_count = int(raw["point_count"])
            line_fraction = finite(float(raw["line_fraction"]), "line fraction")
            distance = finite(float(raw["distance_m"]), "relative distance")
            point = [finite(float(raw[key]), f"relative {key}") for key in ("x_m", "y_m", "z_m")]
            if sample != len(rows[station]) or point_count != VELOCITY_SAMPLE_COUNTS[station]:
                raise ExportError(f"{case} relative coordinate row {position} order differs")
            hashers[station].update(struct.pack(">ddd", *point))
            rows[station].append(
                {
                    "sample_index": sample,
                    "point_count": point_count,
                    "line_fraction": line_fraction,
                    "distance_m": distance,
                    "point_m": point,
                }
            )
    for station in VELOCITY_STATIONS:
        station_rows = rows[station]
        if (
            len(station_rows) != VELOCITY_SAMPLE_COUNTS[station]
            or station_rows[0]["line_fraction"] != 0.0
            or station_rows[-1]["line_fraction"] != 1.0
            or any(
                right["line_fraction"] <= left["line_fraction"]
                for left, right in zip(station_rows, station_rows[1:])
            )
        ):
            raise ExportError(f"{case}/{station} relative coordinate grid differs")
    return (
        {station: tuple(rows[station]) for station in VELOCITY_STATIONS},
        {station: hashers[station].hexdigest() for station in VELOCITY_STATIONS},
    )


def verify_relative_velocity(
    context: InputContext, case: str, native_cell_count: int
) -> VerifiedVelocity:
    placement_row = context.relative_placement_by_case[case]
    mapping_row = context.relative_mapping_by_case[case]
    placement_root = (
        context.relative_producer_root
        / "velocity_support_v3/production_campaign_v1/cases"
        / case
    )
    mapping_root = context.relative_producer_root / "velocity_mapping_v3/cases" / case
    coordinate_path = placement_root / f"{case}-relative-v3-velocity-10mm-coordinates.csv"
    placement_path = placement_root / f"{case}-relative-v3-velocity-receipt.json"
    mapping_path = mapping_root / "velocity-relative-v3-cell-mapping-10mm.json"
    mapping_receipt_path = mapping_root / "receipt.json"
    coordinate_sha = sha256_file(coordinate_path)
    placement, placement_payload = load_json(placement_path, f"{case} relative placement receipt")
    mapping, mapping_payload = load_json(mapping_path, f"{case} relative velocity mapping")
    mapping_receipt, mapping_receipt_payload = load_json(
        mapping_receipt_path, f"{case} relative velocity mapping receipt"
    )
    placement_sha = sha256_bytes(placement_payload)
    mapping_sha = sha256_bytes(mapping_payload)
    mapping_receipt_sha = sha256_bytes(mapping_receipt_payload)
    aggregate_ten = [
        item
        for item in mapping_row.get("artifacts", [])
        if isinstance(item, dict) and item.get("nominal_spacing_mm") == 10
    ]
    receipt_ten = [
        item
        for item in mapping_receipt.get("artifacts", [])
        if isinstance(item, dict) and item.get("nominal_spacing_mm") == 10
    ]
    if (
        placement_row.get("coordinates_sha256") != coordinate_sha
        or placement_row.get("receipt_sha256") != placement_sha
        or mapping_row.get("coordinate_csv_sha256") != coordinate_sha
        or mapping_row.get("placement_receipt_sha256") != placement_sha
        or mapping_row.get("receipt_sha256") != mapping_receipt_sha
        or len(aggregate_ten) != 1
        or aggregate_ten[0].get("sha256") != mapping_sha
        or len(receipt_ten) != 1
        or receipt_ten[0].get("sha256") != mapping_sha
        or receipt_ten[0].get("assignment_evidence_sha256")
        != aggregate_ten[0].get("assignment_evidence_sha256")
    ):
        raise ExportError(f"{case} relative velocity byte bindings differ")
    coordinate_rows, replayed_support = _relative_coordinate_rows(coordinate_path, case=case)
    profiles = placement.get("profiles")
    if (
        placement.get("schema") != "drivaerml-relative-velocity-v3-case-receipt-v1"
        or placement.get("case_id") != case
        or placement.get("family_id") != RELATIVE_VELOCITY_FAMILY
        or placement.get("placement_mode") != "relative"
        or placement.get("profile_count") != 16
        or not isinstance(profiles, list)
        or [item.get("profile_id") for item in profiles if isinstance(item, dict)]
        != list(VELOCITY_STATIONS)
    ):
        raise ExportError(f"{case} relative velocity placement declaration differs")
    support: dict[str, str] = {}
    for profile in profiles:
        station = str(profile["profile_id"])
        support[station] = digest(
            profile.get("coordinates_binary64_be_sha256"),
            f"{case}/{station} relative velocity support",
        )
        if support[station] != replayed_support[station]:
            raise ExportError(f"{case}/{station} relative support identity does not replay")
    resolution = mapping.get("resolution")
    coverage = mapping.get("coverage")
    relative_support = mapping.get("relative_placement_support")
    if (
        mapping.get("schema") != "drivaerml-velocity-relative-v3-cell-mapping-v1"
        or mapping.get("schema_version") != 1
        or mapping.get("case_id") != case
        or mapping.get("dataset_id") != DATASET_ID
        or mapping.get("family_id") != RELATIVE_VELOCITY_FAMILY
        or mapping.get("placement_mode") != "relative"
        or mapping.get("row_fields") != MAPPING_ROW_FIELDS
        or mapping.get("assignment_evidence_sha256")
        != aggregate_ten[0].get("assignment_evidence_sha256")
        or not isinstance(resolution, dict)
        or resolution.get("sample_count") != 3756
        or resolution.get("line_count") != 16
        or resolution.get("nominal_spacing_mm") != 10
        or resolution.get("sample_namespace") != "fixed_normalized_arc_index"
        or not isinstance(coverage, dict)
        or coverage.get("complete_duplicate_free_no_omissions") is not True
        or not isinstance(relative_support, dict)
        or relative_support.get("coordinate_csv_sha256") != coordinate_sha
        or relative_support.get("placement_receipt_sha256") != placement_sha
        or mapping_receipt.get("schema")
        != "drivaerml-velocity-relative-v3-cell-mapping-case-v1"
        or mapping_receipt.get("schema_version") != 1
        or mapping_receipt.get("case_id") != case
        or mapping_receipt.get("dataset_id") != DATASET_ID
        or mapping_receipt.get("public_dataset_revision") != DATASET_REVISION
    ):
        raise ExportError(f"{case} relative velocity mapping declaration differs")
    pin_case = context.pin_by_case[case]
    _verify_native_binding(
        mapping.get("native_source_binding"),
        case=case,
        pin_case=pin_case,
        pin_sha=context.pin_sha256,
        label="relative velocity mapping",
    )
    _verify_native_binding(
        mapping_receipt.get("native_source_binding"),
        case=case,
        pin_case=pin_case,
        pin_sha=context.pin_sha256,
        label="relative velocity receipt",
    )
    by_station, counts = _validate_mapping_rows(
        mapping.get("rows"),
        case=case,
        native_cell_count=native_cell_count,
        coordinate_rows=coordinate_rows,
    )
    for key in ("valid_count", "invalid_count"):
        if coverage.get(key) != counts[key] or receipt_ten[0].get(key) != counts[key]:
            raise ExportError(f"{case} relative velocity {key} differs")
    if coverage.get("invalid_reason_counts") != counts["invalid_reason_counts"]:
        raise ExportError(f"{case} relative velocity invalid reasons differ")
    source_parts, _ = _source_parts(pin_case)
    assignment_sha = _assignment_evidence_identity(
        case,
        mapping["rows"],  # type: ignore[arg-type]
        [str(item["sha256"]) for item in source_parts],
    )
    if assignment_sha != mapping.get("assignment_evidence_sha256"):
        raise ExportError(f"{case} relative velocity assignment identity differs")
    return VerifiedVelocity(
        family_id=RELATIVE_VELOCITY_FAMILY,
        placement_mode="relative",
        mapping_sha256=mapping_sha,
        mapping_receipt_sha256=mapping_receipt_sha,
        placement_receipt_sha256=placement_sha,
        support_identity_by_station=support,
        rows_by_station=by_station,
        input_bindings={
            "coordinate_csv": {"path": f"velocity_support_v3/production_campaign_v1/cases/{case}/{coordinate_path.name}", "sha256": coordinate_sha, "size_bytes": coordinate_path.stat().st_size},
            "placement_receipt": {"path": f"velocity_support_v3/production_campaign_v1/cases/{case}/{placement_path.name}", "sha256": placement_sha, "size_bytes": len(placement_payload)},
            "mapping": {"path": f"velocity_mapping_v3/cases/{case}/{mapping_path.name}", "sha256": mapping_sha, "size_bytes": len(mapping_payload)},
            "mapping_receipt": {"path": f"velocity_mapping_v3/cases/{case}/{mapping_receipt_path.name}", "sha256": mapping_receipt_sha, "size_bytes": len(mapping_receipt_payload)},
        },
    )


@dataclass(frozen=True)
class VerifiedCp:
    constant_cuts: tuple[Mapping[str, object], ...]
    relative_aliases: tuple[Mapping[str, object], ...]
    relative_moving_cuts: tuple[Mapping[str, object], ...]
    constant_receipt_identity_sha256: str
    relative_placement_identity_sha256: str
    input_bindings: Mapping[str, object]


def cp_display_coordinate_x(row: Mapping[str, object], label: str) -> float:
    """Return raw streamwise x at the retained cut-segment midpoint.

    Both producer families retain the exact plane-intersection segment
    endpoints represented by each row.  Their midpoint is therefore on the
    precise cut segment used to construct the scoring arc-length support.
    """

    endpoints: list[list[float]] = []
    for field in ("endpoint_start_m", "endpoint_end_m"):
        raw = row.get(field)
        if not isinstance(raw, list) or len(raw) != 3:
            raise ExportError(f"{label} {field} must contain three coordinates")
        endpoints.append(
            [finite(component, f"{label} {field}[{axis}]") for axis, component in enumerate(raw)]
        )
    midpoint_x = 0.5 * (endpoints[0][0] + endpoints[1][0])
    retained_length = finite(row.get("length_m"), f"{label} length_m")
    replayed_length = math.sqrt(
        sum((right - left) ** 2 for left, right in zip(*endpoints))
    )
    if not math.isclose(retained_length, replayed_length, rel_tol=1e-12, abs_tol=1e-15):
        raise ExportError(f"{label} retained segment length does not replay endpoints")
    return finite(midpoint_x, f"{label} display streamwise x")


def _verify_cp_rows(
    rows: object,
    *,
    case: str,
    station: str,
    coordinate_key: str,
) -> None:
    if not isinstance(rows, list) or len(rows) < 2:
        raise ExportError(f"{case}/{station} Cp rows differ")
    previous_ordinal = -1
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ExportError(f"{case}/{station} Cp row {position} differs")
        ordinal = row.get("ordinal")
        raw_id = row.get("raw_polygon_id")
        pressure = finite(row.get("pMeanTrim"), "pMeanTrim")
        coordinate = finite(row.get(coordinate_key), coordinate_key)
        cp_display_coordinate_x(row, f"{case}/{station} Cp row {position}")
        expected_cp = float(cp_from_kinematic_pressure(pressure))
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != previous_ordinal + 1
            or isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or raw_id < 0
            or coordinate < 0.0
            or row.get("Cp_from_pMeanTrim") != expected_cp
            or finite(row.get("Cp_consistency_absolute_error"), "Cp consistency error")
            > finite(row.get("Cp_consistency_tolerance"), "Cp consistency tolerance")
        ):
            raise ExportError(f"{case}/{station} Cp row {position} truth differs")
        previous_ordinal = ordinal


def verify_cp(context: InputContext, case: str) -> VerifiedCp:
    pin_case = context.pin_by_case[case]
    constant_row = context.constant_cp_by_case[case]
    constant_root = context.constant_cp_campaign_root / "cases" / case
    constant_support_path = constant_root / "case-support.json"
    constant_receipt_path = constant_root / "receipt.json"
    constant_support, constant_support_payload = load_json(
        constant_support_path, f"{case} constant Cp support"
    )
    constant_receipt, constant_receipt_payload = load_json(
        constant_receipt_path, f"{case} constant Cp receipt"
    )
    constant_support_sha = sha256_bytes(constant_support_payload)
    constant_receipt_sha = sha256_bytes(constant_receipt_payload)
    outputs = {
        item.get("path"): item
        for item in constant_row.get("outputs", [])
        if isinstance(item, dict)
    }
    if (
        constant_row.get("receipt_file_sha256") != constant_receipt_sha
        or constant_row.get("receipt_size_bytes") != len(constant_receipt_payload)
        or not isinstance(outputs.get("case-support.json"), dict)
        or outputs["case-support.json"].get("sha256") != constant_support_sha
        or outputs["case-support.json"].get("size_bytes") != len(constant_support_payload)
        or constant_receipt.get("schema")
        != "drivaerml-continuous-cp-cut-case-artifacts-candidate-v3"
        or constant_receipt.get("case_id") != case
        or constant_receipt.get("dataset_id") != DATASET_ID
        or constant_receipt.get("status")
        != "candidate_not_owner_approved_not_evaluator_activation"
    ):
        raise ExportError(f"{case} constant Cp aggregate/receipt binding differs")
    constant_receipt_identity = document_identity_sha256(
        constant_receipt, "receipt_identity"
    )
    native_source = constant_receipt.get("native_source")
    repository = native_source.get("repository") if isinstance(native_source, dict) else None
    if (
        not isinstance(native_source, dict)
        or native_source.get("pin_sha256") != context.pin_sha256
        or native_source.get("case_record") != pin_case
        or not isinstance(repository, dict)
        or repository.get("repo_id") != DATASET_REPOSITORY
        or repository.get("revision") != DATASET_REVISION
    ):
        raise ExportError(f"{case} constant Cp native source binding differs")
    envelope_identities = constant_support.get("identities")
    support = constant_support.get("support")
    if (
        constant_support.get("envelope_schema")
        != "drivaerml-continuous-cp-cut-case-support-candidate-v2"
        or not isinstance(envelope_identities, dict)
        or not isinstance(support, dict)
        or support.get("case_id") != case
        or support.get("freestream_velocity_m_per_s") != U_INF_M_PER_S
    ):
        raise ExportError(f"{case} constant Cp support declaration differs")
    constant_case_identity = digest(
        envelope_identities.get("case_support_sha256"), f"{case} constant Cp support"
    )
    if (
        constant_case_identity != constant_row.get("case_support_sha256")
        or constant_receipt.get("case_support", {}).get("case_support_sha256")
        != constant_case_identity
        or constant_receipt.get("case_support", {}).get("json_sha256")
        != constant_support_sha
    ):
        raise ExportError(f"{case} constant Cp support identity differs")
    cuts = support.get("cuts")
    if (
        not isinstance(cuts, list)
        or len(cuts) != 4
        or [
            cut.get("definition", {}).get("cut_id")
            for cut in cuts
            if isinstance(cut, dict)
        ]
        != list(CONSTANT_CP_STATIONS)
    ):
        raise ExportError(f"{case} constant Cp station set differs")
    for cut in cuts:
        station = cut["definition"]["cut_id"]
        _verify_cp_rows(
            cut.get("segments"),
            case=case,
            station=station,
            coordinate_key="arc_length_end_m",
        )

    relative_row = context.relative_cp_by_case[case]
    relative_root = (
        context.relative_producer_root
        / "cp_support/campaign_v3/native_support_v3/cases"
        / case
    )
    relative_support_path = relative_root / "relative-case-support.json"
    relative_receipt_path = relative_root / "receipt.json"
    placement_path = relative_root / "placement-receipt.json"
    relative_support, relative_support_payload = load_json(
        relative_support_path, f"{case} relative Cp support"
    )
    relative_receipt, relative_receipt_payload = load_json(
        relative_receipt_path, f"{case} relative Cp receipt"
    )
    placement, placement_payload = load_json(placement_path, f"{case} relative Cp placement")
    relative_support_sha = sha256_bytes(relative_support_payload)
    relative_receipt_sha = sha256_bytes(relative_receipt_payload)
    placement_file_sha = sha256_bytes(placement_payload)
    artifacts = relative_receipt.get("artifacts")
    if (
        relative_row.get("relative_case_receipt_sha256") != relative_receipt_sha
        or relative_receipt.get("schema")
        != "drivaerml-relative-cp-case-artifact-receipt-v1"
        or relative_receipt.get("schema_version") != 1
        or relative_receipt.get("case_id") != case
        or relative_receipt.get("dataset_revision") != DATASET_REVISION
        or relative_receipt.get("family_id") != RELATIVE_CP_FAMILY
        or not isinstance(artifacts, dict)
        or artifacts.get("relative-case-support.json", {}).get("sha256")
        != relative_support_sha
        or artifacts.get("placement-receipt.json", {}).get("sha256")
        != placement_file_sha
    ):
        raise ExportError(f"{case} relative Cp aggregate/receipt binding differs")
    document_identity_sha256(relative_receipt, "document_identity")
    if (
        placement.get("schema") != "drivaerml-relative-cp-placement-receipt-v1"
        or placement.get("schema_version") != 1
        or placement.get("case_id") != case
        or placement.get("dataset_id") != DATASET_ID
        or placement.get("family_id") != RELATIVE_CP_FAMILY
        or placement.get("placement_mode") != "relative"
    ):
        raise ExportError(f"{case} relative Cp placement declaration differs")
    relative_placement_identity = document_identity_sha256(
        placement, "receipt_identity"
    )
    relative_identities = relative_support.get("identities")
    relative_body = relative_support.get("support")
    if (
        relative_support.get("envelope_schema")
        != "drivaerml-relative-cp-case-support-envelope-v1"
        or not isinstance(relative_identities, dict)
        or not isinstance(relative_body, dict)
        or relative_body.get("schema") != "drivaerml-relative-cp-case-support-v1"
        or relative_body.get("schema_version") != 1
        or relative_body.get("case_id") != case
        or relative_body.get("family_id") != RELATIVE_CP_FAMILY
        or relative_body.get("freestream_velocity_m_per_s") != U_INF_M_PER_S
    ):
        raise ExportError(f"{case} relative Cp support declaration differs")
    relative_case_identity = digest(
        relative_identities.get("case_support_sha256"), f"{case} relative Cp support"
    )
    if (
        relative_case_identity != sha256_bytes(canonical_json_bytes(relative_body))
        or relative_case_identity != relative_row.get("relative_case_support_sha256")
        or relative_receipt.get("support_identities", {}).get("case_support_sha256")
        != relative_case_identity
        or relative_receipt.get("support_identities", {}).get("json_sha256")
        != relative_support_sha
        or relative_receipt.get("source_inputs", {})
        .get("constant_centerline_authority", {})
        .get("case_support_sha256")
        != constant_case_identity
    ):
        raise ExportError(f"{case} relative Cp support identity differs")
    aliases = relative_body.get("centerline_aliases")
    moving = relative_body.get("moving_cuts")
    if (
        not isinstance(aliases, list)
        or not isinstance(moving, list)
        or [row.get("station_id") for row in aliases if isinstance(row, dict)]
        != list(RELATIVE_CP_ALIAS_STATIONS)
        or [row.get("station_id") for row in moving if isinstance(row, dict)]
        != list(RELATIVE_CP_MOVING_STATIONS)
    ):
        raise ExportError(f"{case} relative Cp station set differs")
    constant_cut_support = {
        cut["definition"]["cut_id"]: sha256_bytes(canonical_json_bytes(cut))
        for cut in cuts
    }
    for alias in aliases:
        station = alias["station_id"]
        if (
            alias.get("kind") != "shared_alias"
            or alias.get("canonical_family_id") != CONSTANT_CP_FAMILY
            or alias.get("canonical_station_id") != station
            or alias.get("canonical_cut_support_sha256")
            != constant_cut_support[station]
        ):
            raise ExportError(f"{case}/{station} relative Cp alias differs")
    for cut in moving:
        station = cut["station_id"]
        cut_body = dict(cut)
        support_identity = digest(
            cut_body.pop("support_identity_sha256", None),
            f"{case}/{station} relative Cp support",
        )
        if support_identity != sha256_bytes(canonical_json_bytes(cut_body)):
            raise ExportError(f"{case}/{station} relative Cp support identity differs")
        _verify_cp_rows(
            cut.get("rows"),
            case=case,
            station=station,
            coordinate_key="interval_arc_end_m",
        )
    return VerifiedCp(
        constant_cuts=tuple(cuts),
        relative_aliases=tuple(aliases),
        relative_moving_cuts=tuple(moving),
        constant_receipt_identity_sha256=constant_receipt_identity,
        relative_placement_identity_sha256=relative_placement_identity,
        input_bindings={
            "constant_support": {"path": f"cases/{case}/{constant_support_path.name}", "sha256": constant_support_sha, "size_bytes": len(constant_support_payload)},
            "constant_receipt": {"path": f"cases/{case}/{constant_receipt_path.name}", "sha256": constant_receipt_sha, "size_bytes": len(constant_receipt_payload), "receipt_identity_sha256": constant_receipt_identity},
            "relative_support": {"path": f"cp_support/campaign_v3/native_support_v3/cases/{case}/{relative_support_path.name}", "sha256": relative_support_sha, "size_bytes": len(relative_support_payload)},
            "relative_receipt": {"path": f"cp_support/campaign_v3/native_support_v3/cases/{case}/{relative_receipt_path.name}", "sha256": relative_receipt_sha, "size_bytes": len(relative_receipt_payload)},
            "relative_placement_receipt": {"path": f"cp_support/campaign_v3/native_support_v3/cases/{case}/{placement_path.name}", "sha256": placement_file_sha, "size_bytes": len(placement_payload), "receipt_identity_sha256": relative_placement_identity},
        },
    )


def materialized_series(
    *,
    panel_id: str,
    family_id: str,
    placement_mode: str,
    station_id: str,
    quantity_id: str,
    quantity: str,
    units: str,
    scoring_role: str,
    support_identity_sha256: str,
    placement_receipt_identity_sha256: str,
    coordinate_id: str,
    coordinate_unit: str,
    sample_index: Sequence[int],
    raw_native_cell_id: Sequence[int],
    coordinate: Sequence[float],
    value: Sequence[float],
    segments: Sequence[Mapping[str, object]],
    unsupported_samples: Sequence[Mapping[str, object]],
    display_coordinate_id: str | None = None,
    display_coordinate_unit: str | None = None,
    display_coordinate: Sequence[float] | None = None,
) -> dict[str, object]:
    samples = list(sample_index)
    raw_ids = list(raw_native_cell_id)
    coordinates = [finite(item, "series coordinate") for item in coordinate]
    values = [finite(item, "series value") for item in value]
    if not (len(samples) == len(raw_ids) == len(coordinates) == len(values)) or not samples:
        raise ExportError("materialized series arrays must be non-empty and aligned")
    if len(set(samples)) != len(samples) or any(
        right <= left for left, right in zip(samples, samples[1:])
    ):
        raise ExportError("materialized sample indexes must be strictly increasing")
    for item in unsupported_samples:
        if set(item) != {"sample_index", "coordinate", "reason"}:
            raise ExportError("unsupported sample shape differs")
        finite(item.get("coordinate"), "unsupported coordinate")
        if (
            isinstance(item.get("sample_index"), bool)
            or not isinstance(item.get("sample_index"), int)
            or not isinstance(item.get("reason"), str)
            or not item.get("reason")
        ):
            raise ExportError("unsupported sample content differs")
    result: dict[str, object] = {
        "panel_id": panel_id,
        "family_id": family_id,
        "placement_mode": placement_mode,
        "station_id": station_id,
        "quantity_id": quantity_id,
        "quantity": quantity,
        "units": units,
        "scoring_role": scoring_role,
        "representation": "materialized",
        "support_identity_sha256": digest(
            support_identity_sha256, "series support identity"
        ),
        "placement_receipt_identity_sha256": digest(
            placement_receipt_identity_sha256, "series placement receipt"
        ),
        "coordinate_id": coordinate_id,
        "coordinate_unit": coordinate_unit,
        "coordinate_identity_sha256": coordinate_array_identity_sha256(coordinates),
        "value_identity_sha256": binary64_array_identity_sha256(values),
        "sample_index": samples,
        "raw_native_cell_id": raw_ids,
        "coordinate": coordinates,
        "value": values,
        "segments": [dict(item) for item in segments],
        "unsupported_samples": [dict(item) for item in unsupported_samples],
    }
    display_inputs = (
        display_coordinate_id,
        display_coordinate_unit,
        display_coordinate,
    )
    if any(item is not None for item in display_inputs):
        if (
            not isinstance(display_coordinate_id, str)
            or not display_coordinate_id
            or not isinstance(display_coordinate_unit, str)
            or not display_coordinate_unit
            or display_coordinate is None
        ):
            raise ExportError("display coordinate id, unit, and array must be supplied together")
        display_coordinates = [
            finite(item, "series display coordinate") for item in display_coordinate
        ]
        if len(display_coordinates) != len(samples):
            raise ExportError("display coordinate must align with every materialized sample")
        result.update(
            {
                "display_coordinate_id": display_coordinate_id,
                "display_coordinate_unit": display_coordinate_unit,
                "display_coordinate_identity_sha256": coordinate_array_identity_sha256(
                    display_coordinates
                ),
                "display_coordinate": display_coordinates,
            }
        )
    result["series_identity_sha256"] = series_identity_sha256(result)
    return result


def alias_series(
    *,
    station_id: str,
    placement_receipt_identity_sha256: str,
    shared_support_ref: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "canonical_family_id",
        "canonical_station_id",
        "canonical_support_identity_sha256",
        "shared_support_id",
    }
    if set(shared_support_ref) != expected:
        raise ExportError("shared_support_ref shape differs")
    result: dict[str, object] = {
        "panel_id": "pressure_profiles",
        "family_id": RELATIVE_CP_FAMILY,
        "placement_mode": "relative",
        "station_id": station_id,
        "quantity_id": "cp",
        "quantity": "pressure_coefficient",
        "units": "1",
        "scoring_role": "report_only",
        "representation": "shared_alias",
        "placement_receipt_identity_sha256": digest(
            placement_receipt_identity_sha256, "alias placement receipt"
        ),
        "shared_support_ref": dict(shared_support_ref),
    }
    result["series_identity_sha256"] = series_identity_sha256(result)
    return result


def _velocity_series(
    verified: VerifiedVelocity,
    sparse_ids: np.ndarray,
    sparse_values: np.ndarray,
) -> list[dict[str, object]]:
    position_by_id = {int(raw): index for index, raw in enumerate(sparse_ids)}
    output: list[dict[str, object]] = []
    for station in VELOCITY_STATIONS:
        rows = verified.rows_by_station[station]
        valid_rows = [row for row in rows if row["valid"]]
        invalid_rows = [row for row in rows if not row["valid"]]
        samples = [int(row["sample_index"]) for row in valid_rows]
        raw_ids = [int(row["raw_native_cell_id"]) for row in valid_rows]
        if verified.placement_mode == "constant":
            coordinates = [float(row["distance_m"]) for row in valid_rows]
            unsupported = [
                {
                    "sample_index": int(row["sample_index"]),
                    "coordinate": float(row["distance_m"]),
                    "reason": str(row["reason"]),
                }
                for row in invalid_rows
            ]
            coordinate_id = "distance_m"
            coordinate_unit = "m"
            station_id = f"autocfd5_{station.lower()}"
            scoring_role = "inherits_parent_candidate"
        else:
            coordinates = [float(row["line_fraction"]) for row in valid_rows]
            unsupported = [
                {
                    "sample_index": int(row["sample_index"]),
                    "coordinate": float(row["line_fraction"]),
                    "reason": str(row["reason"]),
                }
                for row in invalid_rows
            ]
            coordinate_id = "normalized_arc_length"
            coordinate_unit = "1"
            station_id = station
            scoring_role = "report_only"
        vectors = np.asarray(
            [sparse_values[position_by_id[raw]] for raw in raw_ids], dtype=np.float64
        )
        ratios = velocity_magnitude_ratio(vectors).tolist()
        output.append(
            materialized_series(
                panel_id="velocity_profiles",
                family_id=verified.family_id,
                placement_mode=verified.placement_mode,
                station_id=station_id,
                quantity_id="velocity_ratio",
                quantity="velocity_magnitude_ratio",
                units="1",
                scoring_role=scoring_role,
                support_identity_sha256=verified.support_identity_by_station[station],
                placement_receipt_identity_sha256=verified.placement_receipt_sha256,
                coordinate_id=coordinate_id,
                coordinate_unit=coordinate_unit,
                sample_index=samples,
                raw_native_cell_id=raw_ids,
                coordinate=coordinates,
                value=ratios,
                segments=compact_segments(samples, coordinates),
                unsupported_samples=unsupported,
            )
        )
    return output


def _native_cp_values(
    rows: Sequence[Mapping[str, object]],
    position_by_id: Mapping[int, int],
    native_pressure: np.ndarray,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_id = int(row["raw_polygon_id"])
        try:
            pressure = float(native_pressure[position_by_id[raw_id]])
        except KeyError as error:
            raise ExportError(f"native boundary selection omitted polygon {raw_id}") from error
        values.append(float(cp_from_kinematic_pressure(pressure)))
    return values


def _constant_cp_series(
    cp: VerifiedCp,
    sparse_ids: np.ndarray,
    sparse_pressure: np.ndarray,
) -> list[dict[str, object]]:
    position_by_id = {int(raw): index for index, raw in enumerate(sparse_ids)}
    output: list[dict[str, object]] = []
    for cut in cp.constant_cuts:
        station = str(cut["definition"]["cut_id"])
        rows = cut["segments"]
        samples = [int(row["ordinal"]) for row in rows]
        coordinates = [float(row["arc_length_end_m"]) for row in rows]
        display_coordinates = [
            cp_display_coordinate_x(row, f"{station} constant Cp row {position}")
            for position, row in enumerate(rows)
        ]
        raw_ids = [int(row["raw_polygon_id"]) for row in rows]
        values = _native_cp_values(rows, position_by_id, sparse_pressure)
        output.append(
            materialized_series(
                panel_id="pressure_profiles",
                family_id=CONSTANT_CP_FAMILY,
                placement_mode="constant",
                station_id=station,
                quantity_id="cp",
                quantity="pressure_coefficient",
                units="1",
                scoring_role="inherits_parent_candidate",
                support_identity_sha256=sha256_bytes(canonical_json_bytes(cut)),
                placement_receipt_identity_sha256=cp.constant_receipt_identity_sha256,
                coordinate_id="arc_length_m",
                coordinate_unit="m",
                sample_index=samples,
                raw_native_cell_id=raw_ids,
                coordinate=coordinates,
                value=values,
                segments=compact_segments(
                    samples, coordinates, labels=["native_cut_path"] * len(samples)
                ),
                unsupported_samples=[],
                display_coordinate_id=CP_DISPLAY_COORDINATE_ID,
                display_coordinate_unit=CP_DISPLAY_COORDINATE_UNIT,
                display_coordinate=display_coordinates,
            )
        )
    return output


def _relative_cp_series(
    cp: VerifiedCp,
    sparse_ids: np.ndarray,
    sparse_pressure: np.ndarray,
) -> list[dict[str, object]]:
    position_by_id = {int(raw): index for index, raw in enumerate(sparse_ids)}
    output: list[dict[str, object]] = []
    for alias in cp.relative_aliases:
        output.append(
            alias_series(
                station_id=str(alias["station_id"]),
                placement_receipt_identity_sha256=cp.relative_placement_identity_sha256,
                shared_support_ref={
                    "canonical_family_id": alias["canonical_family_id"],
                    "canonical_station_id": alias["canonical_station_id"],
                    "canonical_support_identity_sha256": alias[
                        "canonical_cut_support_sha256"
                    ],
                    "shared_support_id": alias["shared_support_id"],
                },
            )
        )
    for cut in cp.relative_moving_cuts:
        rows = cut["rows"]
        station = str(cut["station_id"])
        samples = [int(row["ordinal"]) for row in rows]
        coordinates = [float(row["interval_arc_end_m"]) for row in rows]
        display_coordinates = [
            cp_display_coordinate_x(row, f"{station} relative Cp row {position}")
            for position, row in enumerate(rows)
        ]
        raw_ids = [int(row["raw_polygon_id"]) for row in rows]
        values = _native_cp_values(rows, position_by_id, sparse_pressure)
        output.append(
            materialized_series(
                panel_id="pressure_profiles",
                family_id=RELATIVE_CP_FAMILY,
                placement_mode="relative",
                station_id=station,
                quantity_id="cp",
                quantity="pressure_coefficient",
                units="1",
                scoring_role="report_only",
                support_identity_sha256=str(cut["support_identity_sha256"]),
                placement_receipt_identity_sha256=cp.relative_placement_identity_sha256,
                coordinate_id="arc_length_m",
                coordinate_unit="m",
                sample_index=samples,
                raw_native_cell_id=raw_ids,
                coordinate=coordinates,
                value=values,
                segments=compact_segments(
                    samples,
                    coordinates,
                    labels=[str(row["interval_id"]) for row in rows],
                ),
                unsupported_samples=[],
                display_coordinate_id=CP_DISPLAY_COORDINATE_ID,
                display_coordinate_unit=CP_DISPLAY_COORDINATE_UNIT,
                display_coordinate=display_coordinates,
            )
        )
    return output


def export_case(context: InputContext, case: str) -> dict[str, object]:
    if case not in context.pin_by_case:
        raise ExportError(f"{case} is not an official pinned case")
    pin_case = context.pin_by_case[case]
    native_audit = context.native_audit_by_case[case]
    piece = native_audit.get("vtk", {}).get("piece", {})
    field_audit = native_audit.get("required_cell_data", {}).get("UMeanTrim", {})
    native_cell_count = positive_integer(piece.get("number_of_cells"), "native cell count")
    if (
        field_audit.get("association") != "CellData"
        or field_audit.get("vtk_type") != "Float32"
        or field_audit.get("number_of_components") != 3
        or field_audit.get("tuple_count") != native_cell_count
        or field_audit.get("decoded_payload_bytes") != native_cell_count * 12
        or field_audit.get("finite") is not True
    ):
        raise ExportError(f"{case} retained native UMeanTrim audit differs")
    source_parts, _ = _source_parts(pin_case)
    audit_source = native_audit.get("source")
    expected_audit_parts = [
        {
            "part_index": item["part_index"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in source_parts
    ]
    volume = pin_case["volume"]
    if (
        not isinstance(audit_source, dict)
        or audit_source.get("repository_id") != DATASET_REPOSITORY
        or audit_source.get("immutable_revision") != DATASET_REVISION
        or audit_source.get("logical_path") != volume["logical_path_after_assembly"]
        or audit_source.get("logical_size_bytes") != volume["total_size_bytes"]
        or audit_source.get("ordered_verified_segments") != expected_audit_parts
    ):
        raise ExportError(f"{case} native audit source binding differs")

    constant_velocity = verify_constant_velocity(context, case, native_cell_count)
    relative_velocity = verify_relative_velocity(context, case, native_cell_count)
    cp = verify_cp(context, case)
    velocity_raw_ids = sorted(
        {
            int(row["raw_native_cell_id"])
            for verified in (constant_velocity, relative_velocity)
            for station in VELOCITY_STATIONS
            for row in verified.rows_by_station[station]
            if row["valid"]
        }
    )
    native_path = context.dataset_root / str(volume["logical_path_after_assembly"])
    volume_before = stable_stat(native_path)
    if volume_before["size_bytes"] != volume["total_size_bytes"]:
        raise ExportError(f"{case} assembled native volume size differs")
    try:
        stream = native_path.open("rb", buffering=0)
    except OSError as error:
        raise ExportError(f"cannot open {case} native volume: {error}") from error
    with stream:
        volume_inline = locate_uncompressed_cell_field(
            stream,
            field_name="UMeanTrim",
            expected_components=3,
            expected_tuple_count=native_cell_count,
        )
        if volume_inline.declared_payload_bytes != field_audit["decoded_payload_bytes"]:
            raise ExportError(f"{case} parsed UMeanTrim payload size differs")
        sparse_ids, sparse_values = direct_sparse_float32x3(
            stream, volume_inline, velocity_raw_ids
        )
    volume_after = stable_stat(native_path)
    if volume_before != volume_after:
        raise ExportError(f"{case} native volume stat changed during sparse gather")
    volume_selected_sha = selected_array_evidence_sha256(sparse_ids, sparse_values)
    if case == "run_419" and volume_selected_sha != RUN419_SELECTED_VALUES_SHA256:
        raise ExportError("run_419 native selected-values identity differs from evaluator evidence")

    cp_rows = [
        row
        for cut in (*cp.constant_cuts, *cp.relative_moving_cuts)
        for row in (cut["segments"] if "segments" in cut else cut["rows"])
    ]
    boundary_raw_ids = sorted({int(row["raw_polygon_id"]) for row in cp_rows})
    boundary = pin_case.get("boundary")
    if not isinstance(boundary, dict):
        raise ExportError(f"{case} pinned native boundary declaration differs")
    boundary_logical_path = boundary.get("path")
    boundary_size = boundary.get("size_bytes")
    boundary_pin_sha = boundary.get("lfs_sha256")
    boundary_blob_sha1 = boundary.get("git_blob_sha1")
    if (
        not isinstance(boundary_logical_path, str)
        or not boundary_logical_path
        or Path(boundary_logical_path).is_absolute()
        or ".." in Path(boundary_logical_path).parts
        or isinstance(boundary_size, bool)
        or not isinstance(boundary_size, int)
        or boundary_size < 1
        or not isinstance(boundary_pin_sha, str)
        or SHA256_RE.fullmatch(boundary_pin_sha) is None
        or boundary_pin_sha == "0" * 64
        or not isinstance(boundary_blob_sha1, str)
        or SHA1_RE.fullmatch(boundary_blob_sha1) is None
        or boundary_blob_sha1 == "0" * 40
    ):
        raise ExportError(f"{case} pinned native boundary declaration differs")
    boundary_path = context.dataset_root / boundary_logical_path
    boundary_before = stable_stat(boundary_path)
    if boundary_before["size_bytes"] != boundary_size:
        raise ExportError(f"{case} native boundary size differs from the immutable pin")
    boundary_source_sha = sha256_file(boundary_path)
    boundary_after_hash = stable_stat(boundary_path)
    if boundary_before != boundary_after_hash:
        raise ExportError(f"{case} native boundary stat changed during full-file hashing")
    if boundary_source_sha != boundary_pin_sha:
        raise ExportError(f"{case} native boundary SHA-256 differs from the immutable pin")
    try:
        boundary_stream = boundary_path.open("rb", buffering=0)
    except OSError as error:
        raise ExportError(f"cannot open {case} native boundary: {error}") from error
    with boundary_stream:
        boundary_inline = locate_uncompressed_cell_field(
            boundary_stream,
            field_name="pMeanTrim",
            expected_components=1,
            expected_tuple_count=None,
            expected_dataset_type="PolyData",
            piece_tuple_attribute="NumberOfPolys",
        )
        boundary_sparse_ids, boundary_sparse_pressure = direct_sparse_float32x1(
            boundary_stream, boundary_inline, boundary_raw_ids
        )
    boundary_after_gather = stable_stat(boundary_path)
    if boundary_before != boundary_after_gather:
        raise ExportError(f"{case} native boundary stat changed during sparse gather")
    boundary_position_by_id = {
        int(raw): index for index, raw in enumerate(boundary_sparse_ids)
    }
    for position, row in enumerate(cp_rows):
        raw_id = int(row["raw_polygon_id"])
        producer_pressure = finite(row.get("pMeanTrim"), "producer pMeanTrim")
        native_pressure = float(
            boundary_sparse_pressure[boundary_position_by_id[raw_id]]
        )
        if producer_pressure != native_pressure:
            raise ExportError(
                f"{case} Cp producer row {position} pMeanTrim differs from "
                f"native boundary polygon {raw_id}"
            )
    boundary_selected_sha = selected_array_evidence_sha256(
        boundary_sparse_ids, boundary_sparse_pressure
    )

    series: list[dict[str, object]] = []
    series.extend(_velocity_series(constant_velocity, sparse_ids, sparse_values))
    series.extend(_velocity_series(relative_velocity, sparse_ids, sparse_values))
    series.extend(_constant_cp_series(cp, boundary_sparse_ids, boundary_sparse_pressure))
    series.extend(_relative_cp_series(cp, boundary_sparse_ids, boundary_sparse_pressure))
    keys = [(item["family_id"], item["station_id"]) for item in series]
    if len(series) != 40 or len(set(keys)) != 40:
        raise ExportError(f"{case} output must contain exactly 40 unique series")
    body = {
        "schema": CASE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "case_id": case,
        "series_count": 40,
        "truth_source": dict(TRUTH_SOURCE),
        "relative_scoring_activated": False,
        "native_volume": {
            "logical_path": volume["logical_path_after_assembly"],
            "logical_size_bytes": volume["total_size_bytes"],
            "source_parts": source_parts,
            "pre_post_source_stat_unchanged": True,
            "field_name": "UMeanTrim",
            "association": "CellData",
            "vtk_type": "Float32",
            "number_of_components": 3,
            "tuple_count": native_cell_count,
            "declared_payload_bytes": volume_inline.declared_payload_bytes,
            "retained_full_payload_sha256": digest(
                field_audit.get("payload_sha256"), f"{case} UMeanTrim payload"
            ),
            "full_payload_verification": (
                "retained_all484_native_volume_audit_exact_source_binding"
            ),
            "selected_unique_raw_cell_id_count": len(sparse_ids),
            "selected_values_sha256": volume_selected_sha,
        },
        "native_boundary": {
            "logical_path": boundary_logical_path,
            "logical_size_bytes": boundary_size,
            "source_sha256": boundary_source_sha,
            "pin_git_blob_sha1": boundary_blob_sha1,
            "full_file_verification": "sha256_recomputed_against_native_source_pin",
            "pre_post_source_stat_unchanged": True,
            "field_name": "pMeanTrim",
            "association": "CellData",
            "vtk_type": "Float32",
            "number_of_components": 1,
            "tuple_count": boundary_inline.tuple_count,
            "declared_payload_bytes": boundary_inline.declared_payload_bytes,
            "referenced_producer_row_count": len(cp_rows),
            "selected_unique_raw_polygon_id_count": len(boundary_sparse_ids),
            "selected_values_sha256": boundary_selected_sha,
            "producer_value_crosscheck": "exact_float_equality_all_referenced_rows",
        },
        "cp_display_coordinate": {
            "coordinate_id": CP_DISPLAY_COORDINATE_ID,
            "coordinate_unit": CP_DISPLAY_COORDINATE_UNIT,
            "definition": CP_DISPLAY_COORDINATE_DEFINITION,
            "source": "retained_cp_plane_intersection_segment_endpoints",
            "native_coordinate_frame": "raw_vtp_coordinates_metres",
            "transformation": "none_no_shift_normalization_sort_or_resampling",
            "materialized_row_count": len(cp_rows),
            "endpoint_geometry_verification": (
                "three_finite_binary64_components_and_replayed_segment_length_all_rows"
            ),
            "purpose": "display_only_arc_length_remains_scoring_coordinate",
        },
        "input_bindings": {
            "constant_velocity": constant_velocity.input_bindings,
            "relative_velocity": relative_velocity.input_bindings,
            "cp": cp.input_bindings,
        },
        "generator": {
            "evaluator_git_revision": context.evaluator_git_revision,
            "exporter_source": context.global_bindings["exporter_source"],
        },
        "series": series,
    }
    return identity_bound_document(body, "case_identity")


def _expected_series_keys() -> list[tuple[str, str]]:
    return [
        *[
            (CONSTANT_VELOCITY_FAMILY, f"autocfd5_{station.lower()}")
            for station in VELOCITY_STATIONS
        ],
        *[(RELATIVE_VELOCITY_FAMILY, station) for station in VELOCITY_STATIONS],
        *[(CONSTANT_CP_FAMILY, station) for station in CONSTANT_CP_STATIONS],
        *[(RELATIVE_CP_FAMILY, station) for station in RELATIVE_CP_ALIAS_STATIONS],
        *[(RELATIVE_CP_FAMILY, station) for station in RELATIVE_CP_MOVING_STATIONS],
    ]


def _load_case_artifact(
    context: InputContext, expected_case: str
) -> tuple[dict[str, object], dict[str, object]]:
    path = context.output_dir / "cases" / f"{expected_case}.json"
    document, payload = load_json(path, f"{expected_case} exported case")
    if payload != canonical_json_bytes(document):
        raise ExportError(f"{expected_case} case artifact is not canonical JSON")
    if (
        document.get("schema") != CASE_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("dataset_id") != DATASET_ID
        or document.get("dataset_revision") != DATASET_REVISION
        or document.get("case_id") != expected_case
        or document.get("series_count") != 40
        or document.get("truth_source") != TRUTH_SOURCE
        or document.get("relative_scoring_activated") is not False
        or document.get("generator")
        != {
            "evaluator_git_revision": context.evaluator_git_revision,
            "exporter_source": context.global_bindings["exporter_source"],
        }
    ):
        raise ExportError(f"{expected_case} case artifact declaration differs")
    document_identity_sha256(document, "case_identity")
    pin_boundary = context.pin_by_case[expected_case].get("boundary")
    native_boundary = document.get("native_boundary")
    cp_display = document.get("cp_display_coordinate")
    if (
        not isinstance(pin_boundary, dict)
        or not isinstance(native_boundary, dict)
        or not isinstance(cp_display, dict)
    ):
        raise ExportError(f"{expected_case} native boundary binding is missing")
    boundary_tuple_count = positive_integer(
        native_boundary.get("tuple_count"), f"{expected_case} boundary tuple count"
    )
    boundary_selected_count = positive_integer(
        native_boundary.get("selected_unique_raw_polygon_id_count"),
        f"{expected_case} boundary selected count",
    )
    boundary_producer_rows = positive_integer(
        native_boundary.get("referenced_producer_row_count"),
        f"{expected_case} boundary producer row count",
    )
    if (
        native_boundary.get("logical_path") != pin_boundary.get("path")
        or native_boundary.get("logical_size_bytes") != pin_boundary.get("size_bytes")
        or native_boundary.get("source_sha256") != pin_boundary.get("lfs_sha256")
        or native_boundary.get("pin_git_blob_sha1")
        != pin_boundary.get("git_blob_sha1")
        or native_boundary.get("full_file_verification")
        != "sha256_recomputed_against_native_source_pin"
        or native_boundary.get("pre_post_source_stat_unchanged") is not True
        or native_boundary.get("field_name") != "pMeanTrim"
        or native_boundary.get("association") != "CellData"
        or native_boundary.get("vtk_type") != "Float32"
        or native_boundary.get("number_of_components") != 1
        or native_boundary.get("declared_payload_bytes") != boundary_tuple_count * 4
        or boundary_selected_count > boundary_tuple_count
        or boundary_producer_rows < boundary_selected_count
        or native_boundary.get("producer_value_crosscheck")
        != "exact_float_equality_all_referenced_rows"
    ):
        raise ExportError(f"{expected_case} native boundary binding differs")
    if cp_display != {
        "coordinate_id": CP_DISPLAY_COORDINATE_ID,
        "coordinate_unit": CP_DISPLAY_COORDINATE_UNIT,
        "definition": CP_DISPLAY_COORDINATE_DEFINITION,
        "source": "retained_cp_plane_intersection_segment_endpoints",
        "native_coordinate_frame": "raw_vtp_coordinates_metres",
        "transformation": "none_no_shift_normalization_sort_or_resampling",
        "materialized_row_count": boundary_producer_rows,
        "endpoint_geometry_verification": (
            "three_finite_binary64_components_and_replayed_segment_length_all_rows"
        ),
        "purpose": "display_only_arc_length_remains_scoring_coordinate",
    }:
        raise ExportError(f"{expected_case} Cp display coordinate declaration differs")
    digest(
        native_boundary.get("selected_values_sha256"),
        f"{expected_case} native boundary selected values",
    )
    series = document.get("series")
    if not isinstance(series, list) or len(series) != 40:
        raise ExportError(f"{expected_case} case artifact series differ")
    if not all(isinstance(item, dict) for item in series):
        raise ExportError(f"{expected_case} series must be objects")
    keys = [(item.get("family_id"), item.get("station_id")) for item in series]
    if keys != _expected_series_keys():
        raise ExportError(f"{expected_case} case artifact series order differs")
    expected_representations = ["materialized"] * 36 + [
        "shared_alias",
        "shared_alias",
        "materialized",
        "materialized",
    ]
    for item, expected_representation in zip(series, expected_representations):
        if item.get("representation") != expected_representation:
            raise ExportError(
                f"{expected_case}/{item.get('family_id')}/{item.get('station_id')} "
                "representation differs"
            )
        common_fields = {
            "panel_id",
            "family_id",
            "placement_mode",
            "station_id",
            "quantity_id",
            "quantity",
            "units",
            "scoring_role",
            "representation",
            "placement_receipt_identity_sha256",
            "series_identity_sha256",
        }
        if expected_representation == "shared_alias":
            expected_fields = common_fields | {"shared_support_ref"}
        else:
            expected_fields = common_fields | {
                "support_identity_sha256",
                "coordinate_id",
                "coordinate_unit",
                "coordinate_identity_sha256",
                "value_identity_sha256",
                "sample_index",
                "raw_native_cell_id",
                "coordinate",
                "value",
                "segments",
                "unsupported_samples",
            }
            if item.get("quantity_id") == "cp":
                expected_fields |= {
                    "display_coordinate_id",
                    "display_coordinate_unit",
                    "display_coordinate_identity_sha256",
                    "display_coordinate",
                }
        if set(item) != expected_fields:
            raise ExportError(
                f"{expected_case}/{item.get('family_id')}/{item.get('station_id')} "
                "series fields differ from the native-v3 contract"
            )
        supplied = item.get("series_identity_sha256")
        body = dict(item)
        body.pop("series_identity_sha256", None)
        if supplied != series_identity_sha256(body):
            raise ExportError(
                f"{expected_case}/{item.get('family_id')}/{item.get('station_id')} identity differs"
            )
        if item.get("representation") == "materialized":
            coordinates = item.get("coordinate")
            values = item.get("value")
            if (
                not isinstance(coordinates, list)
                or not isinstance(values, list)
                or item.get("coordinate_identity_sha256")
                != coordinate_array_identity_sha256(coordinates)
                or item.get("value_identity_sha256")
                != binary64_array_identity_sha256(values)
            ):
                raise ExportError(
                    f"{expected_case}/{item.get('family_id')}/{item.get('station_id')} component identity differs"
                )
            display_fields = {
                "display_coordinate_id",
                "display_coordinate_unit",
                "display_coordinate_identity_sha256",
                "display_coordinate",
            }
            present_display_fields = display_fields.intersection(item)
            is_cp = item.get("quantity_id") == "cp"
            if is_cp:
                display_coordinates = item.get("display_coordinate")
                if (
                    present_display_fields != display_fields
                    or item.get("coordinate_id") != "arc_length_m"
                    or item.get("coordinate_unit") != "m"
                    or item.get("display_coordinate_id") != CP_DISPLAY_COORDINATE_ID
                    or item.get("display_coordinate_unit") != CP_DISPLAY_COORDINATE_UNIT
                    or not isinstance(display_coordinates, list)
                    or len(display_coordinates) != len(coordinates)
                    or item.get("display_coordinate_identity_sha256")
                    != coordinate_array_identity_sha256(display_coordinates)
                ):
                    raise ExportError(
                        f"{expected_case}/{item.get('family_id')}/{item.get('station_id')} "
                        "Cp display coordinate differs"
                    )
            elif present_display_fields:
                raise ExportError(
                    f"{expected_case}/{item.get('family_id')}/{item.get('station_id')} "
                    "non-Cp series must not contain a display coordinate"
                )
    verified_cp = verify_cp(context, expected_case)
    expected_display: dict[tuple[str, str], tuple[list[float], str, str]] = {}
    for cut in verified_cp.constant_cuts:
        station = str(cut["definition"]["cut_id"])
        rows = cut["segments"]
        expected_display[(CONSTANT_CP_FAMILY, station)] = (
            [
                cp_display_coordinate_x(
                    row, f"{expected_case}/{station} retained constant Cp row {position}"
                )
                for position, row in enumerate(rows)
            ],
            sha256_bytes(canonical_json_bytes(cut)),
            verified_cp.constant_receipt_identity_sha256,
        )
    for cut in verified_cp.relative_moving_cuts:
        station = str(cut["station_id"])
        rows = cut["rows"]
        expected_display[(RELATIVE_CP_FAMILY, station)] = (
            [
                cp_display_coordinate_x(
                    row, f"{expected_case}/{station} retained relative Cp row {position}"
                )
                for position, row in enumerate(rows)
            ],
            str(cut["support_identity_sha256"]),
            verified_cp.relative_placement_identity_sha256,
        )
    cp_materialized = {
        (str(item["family_id"]), str(item["station_id"])): item
        for item in series
        if item.get("representation") == "materialized"
        and item.get("quantity_id") == "cp"
    }
    if set(cp_materialized) != set(expected_display):
        raise ExportError(f"{expected_case} materialized Cp display series set differs")
    for key, (coordinates, support_identity, placement_identity) in expected_display.items():
        item = cp_materialized[key]
        if (
            item.get("display_coordinate") != coordinates
            or item.get("display_coordinate_identity_sha256")
            != coordinate_array_identity_sha256(coordinates)
            or item.get("support_identity_sha256") != support_identity
            or item.get("placement_receipt_identity_sha256") != placement_identity
        ):
            raise ExportError(
                f"{expected_case}/{key[0]}/{key[1]} display coordinate does not "
                "replay retained producer segment midpoints"
            )
    expected_aliases = {
        str(alias["station_id"]): {
            "canonical_family_id": alias["canonical_family_id"],
            "canonical_station_id": alias["canonical_station_id"],
            "canonical_support_identity_sha256": alias[
                "canonical_cut_support_sha256"
            ],
            "shared_support_id": alias["shared_support_id"],
        }
        for alias in verified_cp.relative_aliases
    }
    actual_aliases = {
        str(item["station_id"]): item
        for item in series
        if item.get("representation") == "shared_alias"
    }
    if set(actual_aliases) != set(expected_aliases):
        raise ExportError(f"{expected_case} shared Cp alias station set differs")
    for station, expected_reference in expected_aliases.items():
        item = actual_aliases[station]
        if (
            item.get("family_id") != RELATIVE_CP_FAMILY
            or item.get("placement_mode") != "relative"
            or item.get("quantity_id") != "cp"
            or item.get("shared_support_ref") != expected_reference
            or item.get("placement_receipt_identity_sha256")
            != verified_cp.relative_placement_identity_sha256
        ):
            raise ExportError(
                f"{expected_case}/{station} shared Cp alias does not replay retained producer support"
            )
    return document, {
        "path": f"cases/{expected_case}.json",
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _relative_binding(path: str, binding: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path,
        "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
    }


def _all484_coverage_summary(
    case_documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    families = {
        family: {
            "materialized_series_count": 0,
            "shared_alias_series_count": 0,
            "sample_count": 0,
            "unsupported_sample_count": 0,
            "segment_count": 0,
            "display_coordinate_sample_count": 0,
        }
        for family in EXPECTED_ALL484_COVERAGE
    }
    for document in case_documents:
        series = document.get("series")
        if not isinstance(series, list):
            raise ExportError("case series are missing while deriving coverage")
        for item in series:
            if not isinstance(item, dict) or item.get("family_id") not in families:
                raise ExportError("case series family differs while deriving coverage")
            family = families[str(item["family_id"])]
            representation = item.get("representation")
            if representation == "shared_alias":
                family["shared_alias_series_count"] += 1
                continue
            if representation != "materialized":
                raise ExportError("case series representation differs while deriving coverage")
            samples = item.get("sample_index")
            unsupported = item.get("unsupported_samples")
            segments = item.get("segments")
            if not all(isinstance(value, list) for value in (samples, unsupported, segments)):
                raise ExportError("materialized coverage arrays differ")
            family["materialized_series_count"] += 1
            family["sample_count"] += len(samples)
            family["unsupported_sample_count"] += len(unsupported)
            family["segment_count"] += len(segments)
            display_coordinates = item.get("display_coordinate")
            if item.get("quantity_id") == "cp":
                if (
                    not isinstance(display_coordinates, list)
                    or len(display_coordinates) != len(samples)
                    or item.get("display_coordinate_id") != CP_DISPLAY_COORDINATE_ID
                    or item.get("display_coordinate_unit") != CP_DISPLAY_COORDINATE_UNIT
                    or item.get("display_coordinate_identity_sha256")
                    != coordinate_array_identity_sha256(display_coordinates)
                ):
                    raise ExportError("materialized Cp display coordinate coverage differs")
                family["display_coordinate_sample_count"] += len(display_coordinates)
            elif display_coordinates is not None:
                raise ExportError("velocity series unexpectedly has a display coordinate")
    for family_id, expected in EXPECTED_ALL484_COVERAGE.items():
        observed = families[family_id]
        for field, expected_value in expected.items():
            if observed[field] != expected_value:
                raise ExportError(
                    f"all-484 {family_id} {field} differs: "
                    f"observed={observed[field]} expected={expected_value}"
                )
        observed["requested_sample_count"] = (
            observed["sample_count"] + observed["unsupported_sample_count"]
        )
    totals = {
        field: sum(int(family[field]) for family in families.values())
        for field in (
            "materialized_series_count",
            "shared_alias_series_count",
            "sample_count",
            "unsupported_sample_count",
            "requested_sample_count",
            "segment_count",
            "display_coordinate_sample_count",
        )
    }
    return {
        "derivation": "recomputed_from_all_484_case_series_arrays",
        "producer_expected_counts_verified": True,
        "families": families,
        "totals": totals,
    }


def assemble(context: InputContext, *, cases_per_chunk: int, check: bool) -> dict[str, object]:
    case_documents: list[dict[str, object]] = []
    case_bindings: list[dict[str, object]] = []
    for expected_case in context.official_case_ids:
        document, binding = _load_case_artifact(context, expected_case)
        case_documents.append(document)
        case_bindings.append(binding)
    coverage_summary = _all484_coverage_summary(case_documents)

    chunk_bindings: list[dict[str, object]] = []
    case_locations: list[dict[str, object]] = []
    chunk_by_case: dict[str, Mapping[str, object]] = {}
    for chunk_number, start in enumerate(range(0, len(case_documents), cases_per_chunk)):
        selected = case_documents[start : start + cases_per_chunk]
        selected_ids = [str(item["case_id"]) for item in selected]
        chunk_id = f"chunk-{chunk_number:03d}"
        chunk = identity_bound_document(
            {
                "schema": CHUNK_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "chunk_id": chunk_id,
                "case_count": len(selected),
                "case_ids": selected_ids,
                "series_per_case": 40,
                "series_count": 40 * len(selected),
                "truth_source": dict(TRUTH_SOURCE),
                "cases": selected,
            },
            "chunk_identity",
        )
        output_path = context.output_dir / "chunks" / f"{chunk_id}.json"
        raw_binding = write_canonical(output_path, chunk, check=check)
        binding = {
            "chunk_id": chunk_id,
            "path": f"chunks/{chunk_id}.json",
            "sha256": raw_binding["sha256"],
            "size_bytes": raw_binding["size_bytes"],
            "case_count": len(selected),
            "case_ids": selected_ids,
            "series_count": 40 * len(selected),
        }
        chunk_bindings.append(binding)
        for offset, selected_case in enumerate(selected_ids):
            location = {
                "case_id": selected_case,
                "chunk_id": chunk_id,
                "chunk_path": binding["path"],
                "chunk_sha256": binding["sha256"],
                "case_offset": offset,
            }
            case_locations.append(location)
            chunk_by_case[selected_case] = binding

    input_inventory = [
        {"case_id": document["case_id"], "input_bindings": document["input_bindings"]}
        for document in case_documents
    ]
    native_volume_inventory = [
        {
            "case_id": document["case_id"],
            "logical_path": document["native_volume"]["logical_path"],
            "logical_size_bytes": document["native_volume"]["logical_size_bytes"],
            "retained_full_payload_sha256": document["native_volume"][
                "retained_full_payload_sha256"
            ],
            "selected_unique_raw_cell_id_count": document["native_volume"][
                "selected_unique_raw_cell_id_count"
            ],
            "selected_values_sha256": document["native_volume"][
                "selected_values_sha256"
            ],
        }
        for document in case_documents
    ]
    native_boundary_inventory = [
        {
            "case_id": document["case_id"],
            "logical_path": document["native_boundary"]["logical_path"],
            "logical_size_bytes": document["native_boundary"]["logical_size_bytes"],
            "source_sha256": document["native_boundary"]["source_sha256"],
            "tuple_count": document["native_boundary"]["tuple_count"],
            "referenced_producer_row_count": document["native_boundary"][
                "referenced_producer_row_count"
            ],
            "selected_unique_raw_polygon_id_count": document["native_boundary"][
                "selected_unique_raw_polygon_id_count"
            ],
            "selected_values_sha256": document["native_boundary"][
                "selected_values_sha256"
            ],
        }
        for document in case_documents
    ]
    cp_display_coordinate_inventory = [
        {
            "case_id": document["case_id"],
            "series": [
                {
                    "family_id": item["family_id"],
                    "station_id": item["station_id"],
                    "sample_count": len(item["display_coordinate"]),
                    "display_coordinate_identity_sha256": item[
                        "display_coordinate_identity_sha256"
                    ],
                    "series_identity_sha256": item["series_identity_sha256"],
                }
                for item in document["series"]
                if item.get("representation") == "materialized"
                and item.get("quantity_id") == "cp"
            ],
        }
        for document in case_documents
    ]
    provenance = identity_bound_document(
        {
            "schema": PROVENANCE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "evaluator_git_revision": context.evaluator_git_revision,
            "truth_source": dict(TRUTH_SOURCE),
            "relative_scoring_activated": False,
            "input_bindings": dict(context.global_bindings),
            "split_source_bindings": dict(context.split_bindings),
            "case_artifact_count": 484,
            "case_artifact_inventory_sha256": sha256_bytes(
                canonical_json_bytes(case_bindings)
            ),
            "case_input_binding_inventory_sha256": sha256_bytes(
                canonical_json_bytes(input_inventory)
            ),
            "native_volume_selected_values_inventory_sha256": sha256_bytes(
                canonical_json_bytes(native_volume_inventory)
            ),
            "native_boundary_selected_values_inventory_sha256": sha256_bytes(
                canonical_json_bytes(native_boundary_inventory)
            ),
            "cp_display_coordinate_inventory_sha256": sha256_bytes(
                canonical_json_bytes(cp_display_coordinate_inventory)
            ),
            "coverage_summary": coverage_summary,
            "run_419_selected_values_sha256": RUN419_SELECTED_VALUES_SHA256,
            "truth_definitions": {
                "velocity_ratio": "numpy.linalg.norm(Float32 UMeanTrim cast exactly to binary64, axis=-1) / 38.889",
                "cp": "2.0 * binary64(directly selected native boundary Float32 pMeanTrim) / (38.889 * 38.889)",
                "cp_display_coordinate": (
                    "binary64 midpoint x = 0.5 * (endpoint_start_m[0] + "
                    "endpoint_end_m[0]) from each retained producer plane-intersection "
                    "segment, in source order and raw VTP metres; display only, with no "
                    "shift, normalization, sorting, interpolation, or resampling"
                ),
                "cp_scoring_coordinate": (
                    "the unchanged retained arc_length_m array and its existing identity"
                ),
                "sampling": "zeroth-order native CellData/raw polygon CellData; every boundary file is fully SHA-256 verified against the immutable pin; producer pMeanTrim is exact-cross-checked but is not the published value source; no interpolation, snapping, remeshing, or gap filling",
            },
            "identity_encodings": {
                "coordinate": (
                    "ASCII domain fluidsbench-drivaerml-coordinate-array-v1 plus NUL, "
                    "uint64_be count, finite IEEE-754 binary64_be values, signed zero normalized"
                ),
                "display_coordinate": (
                    "the same platform-independent coordinate-array encoding as coordinate; "
                    "its SHA-256 is bound into native-series-identity-v2"
                ),
                "value": (
                    "ASCII domain fluidsbench-drivaerml-native-value-array-v1 plus NUL, "
                    "uint64_be count, finite IEEE-754 binary64_be values, signed zero normalized"
                ),
                "native_id": (
                    "ASCII domain fluidsbench-drivaerml-native-id-array-v1 plus NUL, "
                    "uint64_be count, non-negative uint64_be IDs"
                ),
                "series": (
                    "SHA-256 of UTF-8 sorted-key compact JSON plus LF; velocity and aliases "
                    "retain native-series-identity-v1, while materialized Cp uses "
                    "native-series-identity-v2 to additionally bind display coordinate "
                    "id, unit, and identity"
                ),
            },
        },
        "provenance_identity",
    )
    provenance_path = context.output_dir / "provenance.json"
    provenance_raw = write_canonical(provenance_path, provenance, check=check)
    provenance_binding = _relative_binding("provenance.json", provenance_raw)

    split_output_bindings: list[dict[str, object]] = []
    for split_id in EXPECTED_SPLITS:
        source = context.split_indexes[split_id]
        split_cases = source["case_ids"]
        chunk_refs: list[dict[str, object]] = []
        for binding in chunk_bindings:
            included = [item for item in split_cases if item in set(binding["case_ids"])]
            if included:
                chunk_refs.append(
                    {
                        "chunk_id": binding["chunk_id"],
                        "path": binding["path"],
                        "sha256": binding["sha256"],
                        "case_ids": included,
                    }
                )
        split_document = identity_bound_document(
            {
                "schema": SPLIT_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "split_id": split_id,
                "case_set_id": source["case_set_id"],
                "case_id_status": "official",
                "case_count": len(split_cases),
                "case_ids": split_cases,
                "series_per_case": 40,
                "series_count": 40 * len(split_cases),
                "truth_source": dict(TRUTH_SOURCE),
                "source_split_index": context.split_bindings[split_id],
                "chunk_refs": chunk_refs,
            },
            "split_identity",
        )
        relative_path = f"splits/{split_id}.json"
        raw = write_canonical(
            context.output_dir / relative_path, split_document, check=check
        )
        split_output_bindings.append(
            {
                "split_id": split_id,
                **_relative_binding(relative_path, raw),
                "case_count": len(split_cases),
            }
        )

    index = identity_bound_document(
        {
            "schema": INDEX_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "case_count": 484,
            "case_ids": list(context.official_case_ids),
            "cases_per_chunk": cases_per_chunk,
            "series_per_case": 40,
            "series_count": 484 * 40,
            "truth_source": dict(TRUTH_SOURCE),
            "relative_scoring_activated": False,
            "profile_set": {
                CONSTANT_VELOCITY_FAMILY: 16,
                RELATIVE_VELOCITY_FAMILY: 16,
                CONSTANT_CP_FAMILY: 4,
                RELATIVE_CP_FAMILY: 4,
            },
            "cp_display_coordinate": {
                "coordinate_id": CP_DISPLAY_COORDINATE_ID,
                "coordinate_unit": CP_DISPLAY_COORDINATE_UNIT,
                "definition": CP_DISPLAY_COORDINATE_DEFINITION,
                "default_website_axis": True,
                "display_only": True,
                "scoring_coordinate_unchanged": "arc_length_m",
            },
            "coverage_summary": coverage_summary,
            "chunks": chunk_bindings,
            "case_locations": case_locations,
            "splits": split_output_bindings,
            "provenance": provenance_binding,
            "release_receipt_path": "release-receipt.json",
        },
        "index_identity",
    )
    index_raw = write_canonical(context.output_dir / "index.json", index, check=check)
    index_binding = _relative_binding("index.json", index_raw)

    case_artifact_bytes = sum(int(item["size_bytes"]) for item in case_bindings)
    chunk_artifact_bytes = sum(int(item["size_bytes"]) for item in chunk_bindings)
    split_artifact_bytes = sum(
        int(item["size_bytes"]) for item in split_output_bindings
    )
    artifact_size_summary = {
        "scope": "browser_publication_artifacts_bound_by_this_receipt_excluding_the_receipt_itself",
        "public_artifact_count": (
            len(chunk_bindings) + len(split_output_bindings) + 2
        ),
        "public_total_size_bytes": (
            chunk_artifact_bytes
            + split_artifact_bytes
            + int(provenance_binding["size_bytes"])
            + int(index_binding["size_bytes"])
        ),
        "staging_case_artifacts": {
            "published": False,
            "count": len(case_bindings),
            "total_size_bytes": case_artifact_bytes,
            "inventory_bound_in_provenance": True,
        },
        "public_chunk_artifact_count": len(chunk_bindings),
        "public_chunk_artifact_total_size_bytes": chunk_artifact_bytes,
        "chunk_sizes": [
            {"chunk_id": item["chunk_id"], "size_bytes": item["size_bytes"]}
            for item in chunk_bindings
        ],
        "public_split_artifact_count": len(split_output_bindings),
        "public_split_artifact_total_size_bytes": split_artifact_bytes,
        "public_provenance_size_bytes": provenance_binding["size_bytes"],
        "public_index_size_bytes": index_binding["size_bytes"],
    }

    release = identity_bound_document(
        {
            "schema": RELEASE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "evaluator_git_revision": context.evaluator_git_revision,
            "status": "complete_native_truth_export_not_scoring_activation",
            "truth_source": dict(TRUTH_SOURCE),
            "relative_scoring_activated": False,
            "submissions_opened": False,
            "owner_approval_complete": False,
            "coverage_summary": coverage_summary,
            "artifact_size_summary": artifact_size_summary,
            "artifact_bindings": {
                "index": index_binding,
                "provenance": provenance_binding,
                "chunks": chunk_bindings,
                "splits": split_output_bindings,
            },
        },
        "release_identity",
    )
    release_raw = write_canonical(
        context.output_dir / "release-receipt.json", release, check=check
    )
    return {
        "mode": "assemble",
        "case_count": 484,
        "chunk_count": len(chunk_bindings),
        "index": index_binding,
        "provenance": provenance_binding,
        "release_receipt": _relative_binding("release-receipt.json", release_raw),
        "relative_scoring_activated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("case", "assemble", "all"), required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--constant-velocity-receipts-root", type=Path, required=True)
    parser.add_argument("--relative-producer-root", type=Path, required=True)
    parser.add_argument("--constant-cp-campaign-root", type=Path, required=True)
    parser.add_argument("--native-volume-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-git-revision")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--case-id")
    target.add_argument("--case-index", type=int)
    parser.add_argument(
        "--split-index",
        action="append",
        default=[],
        metavar="SPLIT_ID=PATH",
        help="repeat once for each of the eight official submission splits",
    )
    parser.add_argument("--cases-per-chunk", type=int, default=8)
    parser.add_argument("--check", action="store_true")
    values = parser.parse_args(argv)
    if values.mode == "case" and values.case_id is None and values.case_index is None:
        parser.error("case mode requires --case-id or --case-index")
    if values.mode != "case" and (values.case_id is not None or values.case_index is not None):
        parser.error("--case-id/--case-index are valid only in case mode")
    if values.cases_per_chunk < 1:
        parser.error("--cases-per-chunk must be positive")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        context = load_context(args)
        if args.mode == "case":
            if args.case_index is not None:
                if args.case_index < 0 or args.case_index >= len(context.official_case_ids):
                    raise ExportError("--case-index must be in [0, 483]")
                selected = context.official_case_ids[args.case_index]
            else:
                selected = case_id(args.case_id)
            document = export_case(context, selected)
            binding = write_canonical(
                context.output_dir / "cases" / f"{selected}.json",
                document,
                check=args.check,
            )
            print(
                json.dumps(
                    {
                        "mode": "case",
                        "case_id": selected,
                        "artifact": binding,
                        "selected_values_sha256": document["native_volume"][
                            "selected_values_sha256"
                        ],
                        "relative_scoring_activated": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.mode == "all":
            for selected in context.official_case_ids:
                document = export_case(context, selected)
                write_canonical(
                    context.output_dir / "cases" / f"{selected}.json",
                    document,
                    check=args.check,
                )
        result = assemble(
            context, cases_per_chunk=args.cases_per_chunk, check=args.check
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (ExportError, OSError, ValueError) as error:
        raise SystemExit(f"native profile truth export failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
