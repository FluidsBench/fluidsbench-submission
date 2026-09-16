"""Strict private-support serialization and scoring for compact profiles.

The compact participant directory contains one prediction-only NPZ per case.
Coordinates, topology, validity masks, weights, and reference values remain in
an explicitly supplied evaluator-owned support release.  This module validates
both sides of that boundary before it serializes or scores any prediction.

The release format is deliberately a local, inactive candidate.  Supplying a
release path and its manifest digest does not publish or activate the format.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reference.hiliftaeroml.compact_profiles import (
    COMPACT_PROFILE_FORMAT,
    CP_FIXED_POINT_SCALE,
    CP_POINTS_PER_GRAPH,
    PREDICTION_ARRAYS,
    SUPPORT_ARRAYS,
    VELOCITY_ROW_COUNT,
    VELOCITY_STORAGE_ENCODING,
    VELOCITY_STATIONS,
    CompactProfileError,
    compact_case_metadata,
    encode_native_predictions,
    load_compact_prediction_npz,
    load_compact_support_npz,
    score_compact_profiles,
    write_compact_prediction_npz,
    write_compact_support_npz,
)
from reference.hiliftaeroml.native_profiles import (
    CP_SOURCE_ARRAYS,
    SAFE_CASE_ID,
    VELOCITY_SOURCE_ARRAYS,
    NativeProfileError,
    canonical_json_bytes,
    load_json,
    sha256_file,
    validate_cp_source,
    validate_velocity_source,
)


ROOT = Path(__file__).resolve().parents[2]
COMPACT_PROFILE_CONTRACT_PATH = (
    ROOT / "benchmark-specs" / "hiliftaeroml" / "native-profile-format-v2.json"
)
COMPACT_PROFILE_CONTRACT_ID = (
    "hiliftaeroml-compact-profile-predictions-v2-candidate"
)
COMPACT_PROFILE_CONTRACT_SHA256 = (
    "b1fd29b2cb6c1c84c694ddffc5d85f6cd68fd175b79c3695a443aa78bf962c2f"
)
COMPACT_PROFILE_CHUNK_SCHEMA = (
    "hiliftaeroml-compact-profile-chunk-v2-candidate"
)
COMPACT_PROFILE_SCHEMA_VERSION = "2.0"
COMPACT_PROFILE_INDEX_SCHEMA_VERSION = "1.0"

COMPACT_SUPPORT_RELEASE_ID = "hiliftaeroml-compact-profile-support-v2-candidate"
COMPACT_SUPPORT_RELEASE_FORMAT = (
    "fluidsbench-hiliftaeroml-compact-support-release-v2-candidate"
)
COMPACT_SUPPORT_RELEASE_STATUS = "complete_candidate_not_published"
COMPACT_SUPPORT_RELEASE_USAGE = "maintainer_local_candidate_dry_run_only"
COMPACT_SUPPORT_MANIFEST_SCHEMA = (
    "hiliftaeroml-compact-profile-support-manifest-v2-candidate"
)
COMPACT_SUPPORT_INDEX_SCHEMA = (
    "hiliftaeroml-compact-profile-support-index-v2-candidate"
)
COMPACT_SUPPORT_CASE_SCHEMA = (
    "hiliftaeroml-compact-profile-support-case-v2-candidate"
)

SOURCE_ARTIFACT_SHA256_KEYS = (
    "cp_profile_metrics",
    "cp_cut_values",
    "velocity_profile_metrics",
    "velocity_profiles",
)

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,159}$")
_SAFE_SUBMISSION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_MAX_CANONICAL_JSON_BYTES = 8 << 20
_MAX_NATIVE_METADATA_JSON_BYTES = 16 << 20
_MAX_NPY_HEADER_BYTES = 16 << 10
_MAX_NATIVE_PROFILE_NPZ_BYTES = 128 << 20
_MAX_NATIVE_PROFILE_UNCOMPRESSED_BYTES = 256 << 20
_MAX_COMPACT_SUPPORT_NPZ_BYTES = 128 << 20
_MAX_COMPACT_SUPPORT_UNCOMPRESSED_BYTES = 128 << 20
_MAX_COMPACT_PREDICTION_NPZ_BYTES = 15_000_000
_MAX_COMPACT_CP_POINTS_PER_CASE = 2_000_000

_INACTIVE_ACTIVATION = {
    "owner_approval_complete": False,
    "published": False,
    "submissions_opened": False,
    "candidate_validation_may_change_activation": False,
}

_SURFACE_METADATA_KEYS = {
    "support_identity_sha256",
    "prediction_order_sha256",
    "physical_graph_count",
    "retained_branch_count",
    "retained_point_count",
    "maximum_points_per_physical_graph",
    "quantization_scale",
    "quantization_dtype",
    "delta_dtype",
    "prediction_array",
}
_VELOCITY_METADATA_KEYS = {
    "support_identity_sha256",
    "prediction_order_sha256",
    "station_order",
    "station_count",
    "row_count",
    "valid_row_count",
    "invalid_row_count",
    "prediction_dtype",
    "prediction_array",
    "storage_dtype",
    "storage_encoding",
    "stored_byte_count",
}


class CompactProfileEvaluationError(ValueError):
    """Raised when compact support or participant bytes are not trustworthy."""


@dataclass(frozen=True)
class CompactSupportRelease:
    """Validated handle to one exact case set in a private support release."""

    release_root: Path
    release_id: str
    manifest_sha256: str
    source_profile_truth_release_id: str
    source_profile_truth_manifest_sha256: str
    manifest: dict[str, Any]
    index: dict[str, Any]
    case_set_id: str
    case_ids: tuple[str, ...]
    case_records: dict[str, dict[str, Any]]


def _fail(message: str) -> None:
    raise CompactProfileEvaluationError(message)


def _require_exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        _fail(f"{label} keys differ (observed={observed})")
    return value


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _bounded_int(
    value: Any, label: str, *, minimum: int, maximum: int
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        _fail(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        _fail(f"{label} must equal integer {expected}")
    return value


def _exact_boolean_mapping(
    value: Any, expected: Mapping[str, bool], label: str
) -> None:
    mapping = _require_exact_keys(value, set(expected), label)
    for key, expected_value in expected.items():
        if type(mapping.get(key)) is not bool or mapping.get(key) is not expected_value:
            _fail(f"{label}.{key} must be boolean {expected_value}")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        _fail(f"{label} is invalid")
    return value


def _case_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        _fail(f"{label} must be a sequence of case IDs")
    result = tuple(values)
    if not result or len(result) != len(set(result)):
        _fail(f"{label} must be non-empty and unique")
    for case_id in result:
        if not isinstance(case_id, str) or SAFE_CASE_ID.fullmatch(case_id) is None:
            _fail(f"{label} contains invalid case ID {case_id!r}")
    return result


def _require_contract() -> None:
    try:
        observed = sha256_file(COMPACT_PROFILE_CONTRACT_PATH)
    except NativeProfileError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    if observed != COMPACT_PROFILE_CONTRACT_SHA256:
        _fail("compact profile contract SHA-256 changed")


def _root_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label} must be a regular non-symlink directory: {path}")
    return path.resolve()


def _bounded_regular_file(path: Path, *, maximum: int, label: str) -> int:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise CompactProfileEvaluationError(f"cannot inspect {label}: {error}") from error
    if size < 1 or size > maximum:
        _fail(f"{label} byte size {size} exceeds its bounded input contract")
    return size


def _read_exact(handle: Any, count: int, label: str) -> bytes:
    payload = handle.read(count)
    if len(payload) != count:
        _fail(f"{label} is truncated")
    return payload


def _read_npy_header(
    handle: Any, *, label: str
) -> tuple[np.dtype[Any], tuple[int, ...], bool, int]:
    if _read_exact(handle, 6, label) != b"\x93NUMPY":
        _fail(f"{label} has invalid NPY magic")
    version = tuple(_read_exact(handle, 2, label))
    if version == (1, 0):
        length_size = 2
        encoding = "latin1"
    elif version == (2, 0):
        length_size = 4
        encoding = "latin1"
    elif version == (3, 0):
        length_size = 4
        encoding = "utf-8"
    else:
        _fail(f"{label} uses unsupported NPY version {version}")
    header_length = int.from_bytes(
        _read_exact(handle, length_size, label), "little", signed=False
    )
    if header_length < 1 or header_length > _MAX_NPY_HEADER_BYTES:
        _fail(f"{label} NPY header exceeds the bounded header contract")
    header_bytes = _read_exact(handle, header_length, label)
    try:
        header_text = header_bytes.decode(encoding)
        header = ast.literal_eval(header_text.strip())
    except (UnicodeError, SyntaxError, ValueError, MemoryError, RecursionError) as error:
        raise CompactProfileEvaluationError(
            f"{label} has an invalid NPY header: {error}"
        ) from error
    mapping = _require_exact_keys(
        header, {"descr", "fortran_order", "shape"}, f"{label} NPY header"
    )
    if type(mapping.get("fortran_order")) is not bool:
        _fail(f"{label} NPY fortran_order must be boolean")
    shape = mapping.get("shape")
    if not isinstance(shape, tuple) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        _fail(f"{label} NPY shape must be a tuple of non-negative integers")
    try:
        dtype = np.dtype(mapping.get("descr"))
    except (TypeError, ValueError, MemoryError) as error:
        raise CompactProfileEvaluationError(
            f"{label} has an invalid NPY dtype: {error}"
        ) from error
    if dtype.hasobject:
        _fail(f"{label} NPY dtype must not contain objects")
    header_size = 6 + 2 + length_size + header_length
    return dtype, shape, mapping["fortran_order"], header_size


def _zip_directory_preflight(
    path: Path, *, expected_member_count: int, label: str
) -> None:
    size = path.stat().st_size
    tail_size = min(size, 65_557)
    try:
        with path.open("rb") as handle:
            handle.seek(size - tail_size)
            tail = handle.read(tail_size)
    except OSError as error:
        raise CompactProfileEvaluationError(f"cannot inspect {label}: {error}") from error
    signature = b"PK\x05\x06"
    offset = tail.rfind(signature)
    if offset < 0 or len(tail) - offset < 22:
        _fail(f"{label} has no bounded ZIP end record")
    try:
        (
            _signature,
            disk_number,
            directory_disk,
            disk_entries,
            total_entries,
            directory_size,
            directory_offset,
            comment_length,
        ) = struct.unpack("<4s4H2LH", tail[offset : offset + 22])
    except struct.error as error:
        raise CompactProfileEvaluationError(
            f"{label} ZIP end record is invalid: {error}"
        ) from error
    absolute_offset = size - tail_size + offset
    if (
        disk_number != 0
        or directory_disk != 0
        or disk_entries != expected_member_count
        or total_entries != expected_member_count
        or directory_size > expected_member_count * 65_536
        or directory_offset + directory_size != absolute_offset
        or absolute_offset + 22 + comment_length != size
    ):
        _fail(f"{label} ZIP directory or member count exceeds its contract")


def _array_data_size(
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    *,
    maximum: int,
    label: str,
) -> int:
    count = 1
    for dimension in shape:
        if dimension and count > maximum // dimension:
            _fail(f"{label} declared shape exceeds its bounded array contract")
        count *= dimension
    if dtype.itemsize and count > maximum // dtype.itemsize:
        _fail(f"{label} declared dtype/shape exceeds its bounded array contract")
    return count * dtype.itemsize


def _preflight_npz(
    path: Path,
    *,
    order: Sequence[str],
    label: str,
    maximum_archive_bytes: int,
    maximum_uncompressed_bytes: int,
    expected_arrays: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]] | None = None,
    require_deflate: bool = False,
) -> None:
    _bounded_regular_file(path, maximum=maximum_archive_bytes, label=label)
    _zip_directory_preflight(
        path, expected_member_count=len(order), label=label
    )
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            expected_names = [f"{name}.npy" for name in order]
            if [info.filename for info in infos] != expected_names:
                _fail(f"{label} array inventory or order differs")
            total_uncompressed = 0
            for name, info in zip(order, infos, strict=True):
                member_label = f"{label}/{name}.npy"
                if info.is_dir() or info.flag_bits & 0x1:
                    _fail(f"{member_label} must be a regular unencrypted member")
                allowed = {zipfile.ZIP_DEFLATED} if require_deflate else {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }
                if info.compress_type not in allowed:
                    _fail(f"{member_label} compression method differs")
                if info.file_size < 1 or info.file_size > maximum_uncompressed_bytes:
                    _fail(f"{member_label} uncompressed size exceeds its contract")
                total_uncompressed += info.file_size
                if total_uncompressed > maximum_uncompressed_bytes:
                    _fail(f"{label} uncompressed size exceeds its contract")
                with archive.open(info, mode="r") as member:
                    dtype, shape, fortran_order, header_size = _read_npy_header(
                        member, label=member_label
                    )
                maximum_data = maximum_uncompressed_bytes - header_size
                data_size = _array_data_size(
                    dtype, shape, maximum=maximum_data, label=member_label
                )
                if header_size + data_size != info.file_size:
                    _fail(f"{member_label} shape does not match member byte size")
                if expected_arrays is not None:
                    expected_dtype, expected_shape = expected_arrays[name]
                    if (
                        dtype != expected_dtype
                        or shape != expected_shape
                        or fortran_order is not False
                    ):
                        _fail(f"{member_label} dtype, shape, or order differs")
    except CompactProfileEvaluationError:
        raise
    except (
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        MemoryError,
    ) as error:
        raise CompactProfileEvaluationError(f"cannot preflight {label}: {error}") from error


def _support_array_contract(
    metadata: Mapping[str, Any], *, label: str
) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    outer = _require_exact_keys(
        metadata, {"surface_cp", "volume_velocity"}, label
    )
    surface = _require_exact_keys(
        outer.get("surface_cp"), _SURFACE_METADATA_KEYS, f"{label}.surface_cp"
    )
    velocity = _require_exact_keys(
        outer.get("volume_velocity"),
        _VELOCITY_METADATA_KEYS,
        f"{label}.volume_velocity",
    )
    for key in ("support_identity_sha256", "prediction_order_sha256"):
        _require_sha(surface.get(key), f"{label}.surface_cp.{key}")
        _require_sha(velocity.get(key), f"{label}.volume_velocity.{key}")
    point_count = _bounded_int(
        surface.get("retained_point_count"),
        f"{label}.surface_cp.retained_point_count",
        minimum=2,
        maximum=_MAX_COMPACT_CP_POINTS_PER_CASE,
    )
    branch_count = _bounded_int(
        surface.get("retained_branch_count"),
        f"{label}.surface_cp.retained_branch_count",
        minimum=1,
        maximum=point_count // 2,
    )
    graph_count = _bounded_int(
        surface.get("physical_graph_count"),
        f"{label}.surface_cp.physical_graph_count",
        minimum=1,
        maximum=branch_count,
    )
    if point_count > graph_count * CP_POINTS_PER_GRAPH:
        _fail(f"{label}.surface_cp counts exceed the per-graph point contract")
    _exact_int(
        surface.get("maximum_points_per_physical_graph"),
        CP_POINTS_PER_GRAPH,
        f"{label}.surface_cp.maximum_points_per_physical_graph",
    )
    _exact_int(
        surface.get("quantization_scale"),
        CP_FIXED_POINT_SCALE,
        f"{label}.surface_cp.quantization_scale",
    )
    expected_surface_strings = {
        "quantization_dtype": "int16",
        "delta_dtype": "int16",
        "prediction_array": "cp_q_delta",
    }
    for key, expected in expected_surface_strings.items():
        if type(surface.get(key)) is not str or surface.get(key) != expected:
            _fail(f"{label}.surface_cp.{key} differs")

    if velocity.get("station_order") != list(VELOCITY_STATIONS) or not all(
        type(item) is str for item in velocity.get("station_order", [])
    ):
        _fail(f"{label}.volume_velocity.station_order differs")
    _exact_int(
        velocity.get("station_count"),
        len(VELOCITY_STATIONS),
        f"{label}.volume_velocity.station_count",
    )
    _exact_int(
        velocity.get("row_count"),
        VELOCITY_ROW_COUNT,
        f"{label}.volume_velocity.row_count",
    )
    valid_count = _bounded_int(
        velocity.get("valid_row_count"),
        f"{label}.volume_velocity.valid_row_count",
        minimum=1,
        maximum=VELOCITY_ROW_COUNT,
    )
    invalid_count = _bounded_int(
        velocity.get("invalid_row_count"),
        f"{label}.volume_velocity.invalid_row_count",
        minimum=0,
        maximum=VELOCITY_ROW_COUNT - 1,
    )
    if valid_count + invalid_count != VELOCITY_ROW_COUNT:
        _fail(f"{label}.volume_velocity valid/invalid counts differ")
    expected_velocity_strings = {
        "prediction_dtype": "float32",
        "prediction_array": "velocity_speed_over_u_inf",
        "storage_dtype": "uint8",
        "storage_encoding": VELOCITY_STORAGE_ENCODING,
    }
    for key, expected in expected_velocity_strings.items():
        if type(velocity.get(key)) is not str or velocity.get(key) != expected:
            _fail(f"{label}.volume_velocity.{key} differs")
    _exact_int(
        velocity.get("stored_byte_count"),
        valid_count * np.dtype(np.float32).itemsize,
        f"{label}.volume_velocity.stored_byte_count",
    )

    branch_shape = (branch_count,)
    return {
        "cp_xyz_in": (np.dtype(np.float64), (point_count, 3)),
        "cp_arc_length_in": (np.dtype(np.float64), (point_count,)),
        "cp_truth": (np.dtype(np.float64), (point_count,)),
        "cp_branch_point_offsets": (np.dtype(np.int64), (branch_count + 1,)),
        "cp_branch_row_code": (np.dtype(np.uint8), branch_shape),
        "cp_branch_graph_component_code": (np.dtype(np.int64), branch_shape),
        "cp_branch_component_code": (np.dtype(np.uint8), branch_shape),
        "cp_branch_plane_piece_code": (np.dtype(np.uint8), branch_shape),
        "cp_branch_side_code": (np.dtype(np.uint8), branch_shape),
        "cp_branch_topology_patch_code": (np.dtype(np.uint8), branch_shape),
        "cp_source_branch_index": (np.dtype(np.int64), branch_shape),
        "velocity_requested_xyz_in": (
            np.dtype(np.float64),
            (VELOCITY_ROW_COUNT, 3),
        ),
        "velocity_valid_mask": (np.dtype(np.bool_), (VELOCITY_ROW_COUNT,)),
        "velocity_station_names": (
            np.dtype("<U3"),
            (len(VELOCITY_STATIONS),),
        ),
        "velocity_station_row_offsets": (
            np.dtype(np.int64),
            (len(VELOCITY_STATIONS) + 1,),
        ),
        "velocity_line_length_weights_in": (
            np.dtype(np.float64),
            (VELOCITY_ROW_COUNT,),
        ),
        "velocity_truth_speed_over_uinf": (
            np.dtype(np.float64),
            (VELOCITY_ROW_COUNT,),
        ),
    }


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        _fail(f"{label} must remain inside its root")
    resolved_root = root.resolve()
    unresolved = resolved_root / candidate
    resolved = unresolved.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        _fail(f"{label} resolves outside its root")
    cursor = resolved_root
    for part in candidate.parts:
        cursor /= part
        if cursor.is_symlink():
            _fail(f"{label} traverses a symlink")
    return resolved


def _tree_inventory(root: Path, label: str) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise CompactProfileEvaluationError(
                f"cannot inventory {label} {directory}: {error}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                _fail(f"{label} contains a symlink: {relative}")
            if entry.is_dir(follow_symlinks=False):
                directories.add(relative)
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                files.add(relative)
            else:
                _fail(f"{label} contains a non-regular entry: {relative}")

    visit(root)
    return files, directories


def _set_private_release_modes(root: Path) -> None:
    files, directories = _tree_inventory(root, "compact support release")
    try:
        os.chmod(root, 0o700)
        for relative in sorted(directories):
            os.chmod(root / relative, 0o700)
        for relative in sorted(files):
            os.chmod(root / relative, 0o600)
    except OSError as error:
        raise CompactProfileEvaluationError(
            f"cannot make compact support release private: {error}"
        ) from error


def _require_private_release_modes(
    root: Path, files: set[str], directories: set[str]
) -> None:
    expected_owner = os.geteuid()
    entries = [(root, 0o700)] + [
        (root / relative, 0o700) for relative in sorted(directories)
    ] + [(root / relative, 0o600) for relative in sorted(files)]
    for path, expected_mode in entries:
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as error:
            raise CompactProfileEvaluationError(
                f"cannot inspect compact support release mode {path}: {error}"
            ) from error
        if (
            metadata.st_uid != expected_owner
            or stat.S_IMODE(metadata.st_mode) != expected_mode
        ):
            _fail(
                "compact support release ownership/mode differs from the "
                f"private contract: {path}"
            )


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _bounded_regular_file(
        path, maximum=_MAX_CANONICAL_JSON_BYTES, label=label
    )
    try:
        value, digest = load_json(path, label=label)
        canonical_digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except NativeProfileError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    except (MemoryError, RecursionError) as error:
        raise CompactProfileEvaluationError(
            f"cannot safely load {label}: {error}"
        ) from error
    if digest != canonical_digest:
        _fail(f"{label} is not canonical JSON")
    return value, digest


def _write_json_exclusive(path: Path, value: Any, label: str) -> str:
    try:
        payload = canonical_json_bytes(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except NativeProfileError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    except FileExistsError as error:
        raise CompactProfileEvaluationError(
            f"refusing to overwrite {label}: {path}"
        ) from error
    except OSError as error:
        raise CompactProfileEvaluationError(
            f"cannot write {label} {path}: {error}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    return {
        "file": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _load_descriptor(
    root: Path,
    descriptor: Any,
    *,
    label: str,
    expected_file: str | None = None,
) -> tuple[Path, str]:
    value = _require_exact_keys(
        descriptor, {"file", "sha256", "byte_size"}, f"{label} descriptor"
    )
    if expected_file is not None and value.get("file") != expected_file:
        _fail(f"{label} file differs")
    path = _safe_child(root, value.get("file"), f"{label}.file")
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} must be a regular non-symlink file")
    expected_sha = _require_sha(value.get("sha256"), f"{label}.sha256")
    expected_size = _positive_int(value.get("byte_size"), f"{label}.byte_size")
    try:
        before = path.stat()
        observed_sha = sha256_file(path)
        after = path.stat()
    except (OSError, NativeProfileError) as error:
        raise CompactProfileEvaluationError(f"cannot inspect {label}: {error}") from error
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after):
        _fail(f"{label} changed while it was read")
    if before.st_size != expected_size or observed_sha != expected_sha:
        _fail(f"{label} live bytes differ")
    return path, observed_sha


def _source_hashes(value: Any, label: str) -> dict[str, str]:
    mapping = _require_exact_keys(value, set(SOURCE_ARTIFACT_SHA256_KEYS), label)
    return {
        key: _require_sha(mapping.get(key), f"{label}.{key}")
        for key in SOURCE_ARTIFACT_SHA256_KEYS
    }


def _validate_case_record_identity(
    record: Mapping[str, Any], *, case_id: str
) -> None:
    _require_exact_keys(
        record,
        {
            "schema",
            "schema_version",
            "status",
            "usage",
            "dataset_id",
            "release_id",
            "format",
            "profile_contract_id",
            "profile_contract_sha256",
            "case_id",
            "source_artifact_sha256",
            "support",
            "surface_cp",
            "volume_velocity",
        },
        f"{case_id} compact support case record",
    )
    if (
        record.get("schema") != COMPACT_SUPPORT_CASE_SCHEMA
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != 2
        or record.get("status") != COMPACT_SUPPORT_RELEASE_STATUS
        or record.get("usage") != COMPACT_SUPPORT_RELEASE_USAGE
        or record.get("dataset_id") != "hiliftaeroml"
        or record.get("release_id") != COMPACT_SUPPORT_RELEASE_ID
        or record.get("format") != COMPACT_SUPPORT_RELEASE_FORMAT
        or record.get("profile_contract_id") != COMPACT_PROFILE_CONTRACT_ID
        or record.get("profile_contract_sha256")
        != COMPACT_PROFILE_CONTRACT_SHA256
        or record.get("case_id") != case_id
    ):
        _fail(f"{case_id} compact support case identity differs")
    _source_hashes(
        record.get("source_artifact_sha256"),
        f"{case_id} source_artifact_sha256",
    )
    support = _require_exact_keys(
        record.get("support"),
        {
            "format",
            "file",
            "sha256",
            "byte_size",
            "array_order",
            "ownership",
            "participant_visible",
            "contains_truth",
        },
        f"{case_id} support artifact",
    )
    if (
        support.get("format") != "numpy-npz-v1"
        or support.get("file")
        != f"artifacts/{case_id}/compact-profile-support.npz"
        or support.get("array_order") != list(SUPPORT_ARRAYS)
        or support.get("ownership") != "evaluator"
        or type(support.get("participant_visible")) is not bool
        or support.get("participant_visible") is not False
        or type(support.get("contains_truth")) is not bool
        or support.get("contains_truth") is not True
    ):
        _fail(f"{case_id} compact support artifact boundary differs")
    _require_sha(support.get("sha256"), f"{case_id} support.sha256")
    _positive_int(support.get("byte_size"), f"{case_id} support.byte_size")
    if not isinstance(record.get("surface_cp"), Mapping) or not isinstance(
        record.get("volume_velocity"), Mapping
    ):
        _fail(f"{case_id} compact support metadata is absent")
    _support_array_contract(
        {
            "surface_cp": record["surface_cp"],
            "volume_velocity": record["volume_velocity"],
        },
        label=f"{case_id} compact support metadata",
    )


def _load_support_record(
    release_root: Path, record: Mapping[str, Any], case_id: str
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    descriptor = record["support"]
    path = _safe_child(
        release_root,
        descriptor["file"],
        f"{case_id} compact support artifact",
    )
    expected_arrays = _support_array_contract(
        {
            "surface_cp": record["surface_cp"],
            "volume_velocity": record["volume_velocity"],
        },
        label=f"{case_id} compact support metadata",
    )
    _preflight_npz(
        path,
        order=SUPPORT_ARRAYS,
        label=f"{case_id} compact support artifact",
        maximum_archive_bytes=_MAX_COMPACT_SUPPORT_NPZ_BYTES,
        maximum_uncompressed_bytes=_MAX_COMPACT_SUPPORT_UNCOMPRESSED_BYTES,
        expected_arrays=expected_arrays,
        require_deflate=True,
    )
    try:
        support, observed_sha = load_compact_support_npz(path)
    except CompactProfileError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    if (
        observed_sha != descriptor["sha256"]
        or path.stat().st_size != descriptor["byte_size"]
    ):
        _fail(f"{case_id} compact support artifact live bytes differ")
    metadata = compact_case_metadata(support)
    if (
        record["surface_cp"] != metadata["surface_cp"]
        or record["volume_velocity"] != metadata["volume_velocity"]
    ):
        _fail(f"{case_id} compact support metadata differs from live support")
    return support, _source_hashes(
        record["source_artifact_sha256"],
        f"{case_id} source_artifact_sha256",
    )


def _load_support_case(
    release: CompactSupportRelease, case_id: str
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if case_id not in release.case_ids:
        _fail(f"{case_id} is outside the selected compact support case set")
    return _load_support_record(
        release.release_root, release.case_records[case_id], case_id
    )


def open_compact_support_release(
    *,
    release_root: Path,
    expected_manifest_sha256: str,
    expected_case_ids: Sequence[str],
    case_set_id: str,
) -> CompactSupportRelease:
    """Open and fully validate one exact private candidate support case set."""

    _require_contract()
    expected_cases = _case_ids(expected_case_ids, "selected compact support case IDs")
    selected_case_set = _safe_id(case_set_id, "selected compact support case-set ID")
    expected_manifest = _require_sha(
        expected_manifest_sha256, "compact support manifest digest"
    )
    root = _root_directory(release_root, "compact support release")
    manifest, manifest_sha = _load_canonical_json(
        _safe_child(root, "manifest.json", "compact support manifest"),
        "compact support manifest",
    )
    if manifest_sha != expected_manifest:
        _fail("compact support manifest differs from its explicit binding")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "schema_version",
            "status",
            "usage",
            "dataset_id",
            "release_id",
            "format",
            "profile_contract_id",
            "profile_contract_sha256",
            "source_profile_truth_release_id",
            "source_profile_truth_manifest_sha256",
            "case_count",
            "case_set_count",
            "activation",
            "privacy_boundary",
            "index",
        },
        "compact support manifest",
    )
    privacy = {
        "evaluator_owned": True,
        "participant_visible": False,
        "contains_truth": True,
        "copied_into_participant_package": False,
    }
    if (
        manifest.get("schema") != COMPACT_SUPPORT_MANIFEST_SCHEMA
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
        or manifest.get("status") != COMPACT_SUPPORT_RELEASE_STATUS
        or manifest.get("usage") != COMPACT_SUPPORT_RELEASE_USAGE
        or manifest.get("dataset_id") != "hiliftaeroml"
        or manifest.get("release_id") != COMPACT_SUPPORT_RELEASE_ID
        or manifest.get("format") != COMPACT_SUPPORT_RELEASE_FORMAT
        or manifest.get("profile_contract_id") != COMPACT_PROFILE_CONTRACT_ID
        or manifest.get("profile_contract_sha256")
        != COMPACT_PROFILE_CONTRACT_SHA256
    ):
        _fail("compact support manifest identity or privacy boundary differs")
    _exact_boolean_mapping(
        manifest.get("privacy_boundary"), privacy, "compact support privacy boundary"
    )
    _exact_boolean_mapping(
        manifest.get("activation"), _INACTIVE_ACTIVATION, "compact support activation"
    )
    source_truth_release_id = _safe_id(
        manifest.get("source_profile_truth_release_id"),
        "source profile-truth release ID",
    )
    source_truth_manifest_sha = _require_sha(
        manifest.get("source_profile_truth_manifest_sha256"),
        "source profile-truth manifest SHA-256",
    )
    case_count = _positive_int(manifest.get("case_count"), "manifest case_count")
    case_set_count = _positive_int(
        manifest.get("case_set_count"), "manifest case_set_count"
    )
    index_path, index_sha = _load_descriptor(
        root,
        manifest.get("index"),
        label="compact support index",
        expected_file="index.json",
    )
    index, loaded_index_sha = _load_canonical_json(
        index_path, "compact support index"
    )
    if loaded_index_sha != index_sha:
        _fail("compact support index changed while it was loaded")
    _require_exact_keys(
        index,
        {
            "schema",
            "schema_version",
            "status",
            "usage",
            "dataset_id",
            "release_id",
            "format",
            "profile_contract_id",
            "profile_contract_sha256",
            "source_profile_truth_release_id",
            "source_profile_truth_manifest_sha256",
            "case_count",
            "case_ids",
            "case_sets",
            "case_records",
            "activation",
        },
        "compact support index",
    )
    indexed_cases_raw = index.get("case_ids")
    if not isinstance(indexed_cases_raw, list):
        _fail("compact support index case_ids must be a list")
    indexed_cases = _case_ids(indexed_cases_raw, "compact support index case IDs")
    if (
        index.get("schema") != COMPACT_SUPPORT_INDEX_SCHEMA
        or type(index.get("schema_version")) is not int
        or index.get("schema_version") != 2
        or index.get("status") != COMPACT_SUPPORT_RELEASE_STATUS
        or index.get("usage") != COMPACT_SUPPORT_RELEASE_USAGE
        or index.get("dataset_id") != "hiliftaeroml"
        or index.get("release_id") != COMPACT_SUPPORT_RELEASE_ID
        or index.get("format") != COMPACT_SUPPORT_RELEASE_FORMAT
        or index.get("profile_contract_id") != COMPACT_PROFILE_CONTRACT_ID
        or index.get("profile_contract_sha256")
        != COMPACT_PROFILE_CONTRACT_SHA256
        or index.get("source_profile_truth_release_id")
        != source_truth_release_id
        or index.get("source_profile_truth_manifest_sha256")
        != source_truth_manifest_sha
        or case_count != len(indexed_cases)
    ):
        _fail("compact support index identity or coverage differs")
    _exact_int(
        index.get("case_count"), len(indexed_cases), "compact support index case_count"
    )
    _exact_boolean_mapping(
        index.get("activation"), _INACTIVE_ACTIVATION, "compact support index activation"
    )

    case_sets = index.get("case_sets")
    if not isinstance(case_sets, list) or len(case_sets) != case_set_count:
        _fail("compact support case-set inventory differs")
    seen_case_sets: set[str] = set()
    selected_matches: list[Mapping[str, Any]] = []
    covered: set[str] = set()
    for number, raw in enumerate(case_sets):
        descriptor = _require_exact_keys(
            raw,
            {"case_set_id", "case_count", "case_ids"},
            f"compact support case set {number}",
        )
        identifier = _safe_id(
            descriptor.get("case_set_id"), f"compact support case set {number} ID"
        )
        if identifier in seen_case_sets:
            _fail("compact support case-set IDs are not unique")
        seen_case_sets.add(identifier)
        raw_case_ids = descriptor.get("case_ids")
        if not isinstance(raw_case_ids, list):
            _fail(f"compact support case set {identifier} case_ids must be a list")
        members = _case_ids(raw_case_ids, f"compact support case set {identifier}")
        _exact_int(
            descriptor.get("case_count"),
            len(members),
            f"compact support case set {identifier} count",
        )
        if not set(members).issubset(indexed_cases):
            _fail(f"compact support case set {identifier} has an unknown case")
        covered.update(members)
        if identifier == selected_case_set:
            selected_matches.append(descriptor)
    if covered != set(indexed_cases):
        _fail("compact support case sets do not cover the master case inventory")
    if len(selected_matches) != 1 or selected_matches[0].get("case_ids") != list(
        expected_cases
    ):
        _fail("compact support selected case-set binding differs")

    record_descriptors = _require_exact_keys(
        index.get("case_records"), set(indexed_cases), "compact support case records"
    )
    expected_files = {"manifest.json", "index.json"}
    expected_directories = {"cases", "artifacts"}
    records: dict[str, dict[str, Any]] = {}
    for case_id in indexed_cases:
        expected_record_file = f"cases/{case_id}.json"
        expected_support_file = f"artifacts/{case_id}/compact-profile-support.npz"
        expected_files.update({expected_record_file, expected_support_file})
        expected_directories.add(f"artifacts/{case_id}")
        record_path, record_sha = _load_descriptor(
            root,
            record_descriptors[case_id],
            label=f"{case_id} compact support case record",
            expected_file=expected_record_file,
        )
        record, loaded_record_sha = _load_canonical_json(
            record_path, f"{case_id} compact support case record"
        )
        if loaded_record_sha != record_sha:
            _fail(f"{case_id} compact support case record changed while loaded")
        _validate_case_record_identity(record, case_id=case_id)
        _load_descriptor(
            root,
            {
                key: record["support"][key]
                for key in ("file", "sha256", "byte_size")
            },
            label=f"{case_id} compact support artifact",
            expected_file=expected_support_file,
        )
        records[case_id] = dict(record)

    observed_files, observed_directories = _tree_inventory(
        root, "compact support release"
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        _fail("compact support release file or directory inventory differs")
    _require_private_release_modes(root, observed_files, observed_directories)

    release = CompactSupportRelease(
        release_root=root,
        release_id=COMPACT_SUPPORT_RELEASE_ID,
        manifest_sha256=manifest_sha,
        source_profile_truth_release_id=source_truth_release_id,
        source_profile_truth_manifest_sha256=source_truth_manifest_sha,
        manifest=dict(manifest),
        index=dict(index),
        case_set_id=selected_case_set,
        case_ids=expected_cases,
        case_records=records,
    )
    # Validate canonical NPZ bytes and every metadata binding while streaming
    # one case at a time, so opening a release has a strict guarantee without
    # retaining the entire private truth corpus in memory.
    for case_id in indexed_cases:
        _load_support_record(root, records[case_id], case_id)
    return release


def write_compact_support_release(
    *,
    release_root: Path,
    case_ids: Sequence[str],
    case_sets: Mapping[str, Sequence[str]],
    supports: Mapping[str, Mapping[str, np.ndarray]],
    source_artifact_sha256: Mapping[str, Mapping[str, str]],
    source_profile_truth_release_id: str,
    source_profile_truth_manifest_sha256: str,
) -> str:
    """Materialize canonical private support bytes and return manifest SHA-256.

    This maintainer-facing helper writes only the inactive candidate release.
    It never writes evaluator-owned arrays into a participant directory.
    """

    _require_contract()
    cases = _case_ids(case_ids, "compact support release case IDs")
    source_truth_release_id = _safe_id(
        source_profile_truth_release_id, "source profile-truth release ID"
    )
    source_truth_manifest_sha = _require_sha(
        source_profile_truth_manifest_sha256,
        "source profile-truth manifest SHA-256",
    )
    if set(supports) != set(cases) or set(source_artifact_sha256) != set(cases):
        _fail("compact support release case mapping differs")
    if not isinstance(case_sets, Mapping) or not case_sets:
        _fail("compact support release case_sets must be a non-empty mapping")
    normalized_case_sets: list[dict[str, Any]] = []
    covered: set[str] = set()
    for identifier in sorted(case_sets):
        safe_identifier = _safe_id(identifier, "compact support case-set ID")
        members = _case_ids(
            case_sets[identifier], f"compact support case set {safe_identifier}"
        )
        if not set(members).issubset(cases):
            _fail(f"compact support case set {safe_identifier} has an unknown case")
        covered.update(members)
        normalized_case_sets.append(
            {
                "case_set_id": safe_identifier,
                "case_count": len(members),
                "case_ids": list(members),
            }
        )
    if covered != set(cases):
        _fail("compact support case sets do not cover the release cases")
    if release_root.exists() or release_root.is_symlink():
        _fail(f"compact support release output already exists: {release_root}")
    final_root = release_root
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{final_root.name}.staging-", dir=final_root.parent
        )
    )
    release_root = staging
    try:
        record_descriptors: dict[str, dict[str, Any]] = {}
        for case_id in cases:
            source_hashes = _source_hashes(
                source_artifact_sha256[case_id],
                f"{case_id} source_artifact_sha256",
            )
            support_path = (
                release_root
                / "artifacts"
                / case_id
                / "compact-profile-support.npz"
            )
            try:
                support_sha = write_compact_support_npz(
                    support_path, supports[case_id]
                )
            except CompactProfileError as error:
                raise CompactProfileEvaluationError(str(error)) from error
            metadata = compact_case_metadata(supports[case_id])
            record = {
                "schema": COMPACT_SUPPORT_CASE_SCHEMA,
                "schema_version": 2,
                "status": COMPACT_SUPPORT_RELEASE_STATUS,
                "usage": COMPACT_SUPPORT_RELEASE_USAGE,
                "dataset_id": "hiliftaeroml",
                "release_id": COMPACT_SUPPORT_RELEASE_ID,
                "format": COMPACT_SUPPORT_RELEASE_FORMAT,
                "profile_contract_id": COMPACT_PROFILE_CONTRACT_ID,
                "profile_contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
                "case_id": case_id,
                "source_artifact_sha256": source_hashes,
                "support": {
                    "format": "numpy-npz-v1",
                    "file": support_path.relative_to(release_root).as_posix(),
                    "sha256": support_sha,
                    "byte_size": support_path.stat().st_size,
                    "array_order": list(SUPPORT_ARRAYS),
                    "ownership": "evaluator",
                    "participant_visible": False,
                    "contains_truth": True,
                },
                "surface_cp": metadata["surface_cp"],
                "volume_velocity": metadata["volume_velocity"],
            }
            record_path = release_root / "cases" / f"{case_id}.json"
            _write_json_exclusive(
                record_path, record, f"{case_id} compact support case record"
            )
            record_descriptors[case_id] = _descriptor(record_path, release_root)
        index = {
            "schema": COMPACT_SUPPORT_INDEX_SCHEMA,
            "schema_version": 2,
            "status": COMPACT_SUPPORT_RELEASE_STATUS,
            "usage": COMPACT_SUPPORT_RELEASE_USAGE,
            "dataset_id": "hiliftaeroml",
            "release_id": COMPACT_SUPPORT_RELEASE_ID,
            "format": COMPACT_SUPPORT_RELEASE_FORMAT,
            "profile_contract_id": COMPACT_PROFILE_CONTRACT_ID,
            "profile_contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
            "source_profile_truth_release_id": source_truth_release_id,
            "source_profile_truth_manifest_sha256": source_truth_manifest_sha,
            "case_count": len(cases),
            "case_ids": list(cases),
            "case_sets": normalized_case_sets,
            "case_records": record_descriptors,
            "activation": dict(_INACTIVE_ACTIVATION),
        }
        index_path = release_root / "index.json"
        _write_json_exclusive(index_path, index, "compact support index")
        manifest = {
            "schema": COMPACT_SUPPORT_MANIFEST_SCHEMA,
            "schema_version": 2,
            "status": COMPACT_SUPPORT_RELEASE_STATUS,
            "usage": COMPACT_SUPPORT_RELEASE_USAGE,
            "dataset_id": "hiliftaeroml",
            "release_id": COMPACT_SUPPORT_RELEASE_ID,
            "format": COMPACT_SUPPORT_RELEASE_FORMAT,
            "profile_contract_id": COMPACT_PROFILE_CONTRACT_ID,
            "profile_contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
            "source_profile_truth_release_id": source_truth_release_id,
            "source_profile_truth_manifest_sha256": source_truth_manifest_sha,
            "case_count": len(cases),
            "case_set_count": len(normalized_case_sets),
            "activation": dict(_INACTIVE_ACTIVATION),
            "privacy_boundary": {
                "evaluator_owned": True,
                "participant_visible": False,
                "contains_truth": True,
                "copied_into_participant_package": False,
            },
            "index": _descriptor(index_path, release_root),
        }
        manifest_sha = _write_json_exclusive(
            release_root / "manifest.json", manifest, "compact support manifest"
        )
        _set_private_release_modes(release_root)
        # Re-open the first declared case set through the public validator before
        # handing a digest to callers.
        first = normalized_case_sets[0]
        open_compact_support_release(
            release_root=release_root,
            expected_manifest_sha256=manifest_sha,
            expected_case_ids=first["case_ids"],
            case_set_id=first["case_set_id"],
        )
        if final_root.exists() or final_root.is_symlink():
            _fail(f"compact support release output already exists: {final_root}")
        os.rename(release_root, final_root)
        staging = None
        return manifest_sha
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_builder_inputs(
    *,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    case_ids: Sequence[str],
    cases_per_chunk: int,
    expected_case_artifact_sha256: Mapping[str, Mapping[str, str]],
) -> tuple[str, str, str, tuple[str, ...], dict[str, dict[str, str]]]:
    if not isinstance(submission_id, str) or _SAFE_SUBMISSION_ID.fullmatch(
        submission_id
    ) is None:
        _fail("compact profile submission_id is invalid")
    safe_split = _safe_id(split_id, "compact profile split_id")
    safe_case_set = _safe_id(case_set_id, "compact profile case_set_id")
    cases = _case_ids(case_ids, "compact profile case IDs")
    if (
        not isinstance(cases_per_chunk, int)
        or isinstance(cases_per_chunk, bool)
        or cases_per_chunk < 1
    ):
        _fail("cases_per_chunk must be a positive integer")
    if not isinstance(expected_case_artifact_sha256, Mapping) or set(
        expected_case_artifact_sha256
    ) != set(cases):
        _fail("receipt-bound compact profile case inventory differs")
    hashes = {
        case_id: _source_hashes(
            expected_case_artifact_sha256[case_id],
            f"{case_id} receipt-bound artifact SHA-256",
        )
        for case_id in cases
    }
    return submission_id, safe_split, safe_case_set, cases, hashes


def _opened_support_release(
    *,
    support_release_root: Path,
    support_manifest_sha256: str,
    case_ids: tuple[str, ...],
    case_set_id: str,
    opened_support_release: CompactSupportRelease | None,
) -> CompactSupportRelease:
    """Return one exact support handle, opening it only when necessary.

    An assembler may already have fully validated the immutable evaluator-owned
    release for this exact case set.  Reusing that in-process handle avoids
    streaming the same private support bytes again while preserving every
    release, digest, case-set, and ordered-case binding.
    """

    if opened_support_release is None:
        return open_compact_support_release(
            release_root=support_release_root,
            expected_manifest_sha256=support_manifest_sha256,
            expected_case_ids=case_ids,
            case_set_id=case_set_id,
        )
    if not isinstance(opened_support_release, CompactSupportRelease):
        _fail("opened compact support release handle is invalid")
    if (
        opened_support_release.release_root.resolve()
        != support_release_root.resolve()
        or opened_support_release.manifest_sha256 != support_manifest_sha256
        or opened_support_release.case_set_id != case_set_id
        or opened_support_release.case_ids != case_ids
    ):
        _fail("opened compact support release differs from the requested binding")
    return opened_support_release


def build_compact_profile_directory(
    *,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    case_ids: Sequence[str],
    support_release_root: Path,
    support_manifest_sha256: str,
    outputs_root: Path | None,
    profiles_root: Path,
    cases_per_chunk: int = 10,
    expected_case_artifact_sha256: Mapping[str, Mapping[str, str]],
    surface_outputs_root: Path | None = None,
    volume_outputs_root: Path | None = None,
    opened_support_release: CompactSupportRelease | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Build a deterministic prediction-only compact participant directory."""

    (
        checked_submission,
        checked_split,
        checked_case_set,
        cases,
        expected_hashes,
    ) = _validate_builder_inputs(
        submission_id=submission_id,
        split_id=split_id,
        case_set_id=case_set_id,
        case_ids=case_ids,
        cases_per_chunk=cases_per_chunk,
        expected_case_artifact_sha256=expected_case_artifact_sha256,
    )
    release = _opened_support_release(
        support_release_root=support_release_root,
        support_manifest_sha256=support_manifest_sha256,
        case_ids=cases,
        case_set_id=checked_case_set,
        opened_support_release=opened_support_release,
    )
    surface_root = surface_outputs_root or outputs_root
    volume_root = volume_outputs_root or outputs_root
    if surface_root is None or volume_root is None:
        _fail("surface and volume native-output roots are both required")
    surface_root = _root_directory(surface_root, "surface native-output root")
    volume_root = _root_directory(volume_root, "volume native-output root")
    if profiles_root.exists() or profiles_root.is_symlink():
        _fail(f"compact profile output already exists: {profiles_root}")
    final_profiles_root = profiles_root
    final_profiles_root.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{final_profiles_root.name}.staging-",
            dir=final_profiles_root.parent,
        )
    )
    profiles_root = staging
    try:
        chunks: list[dict[str, Any]] = []
        metrics: dict[str, dict[str, float]] = {}
        for chunk_number, start in enumerate(
            range(0, len(cases), cases_per_chunk)
        ):
            selected = list(cases[start : start + cases_per_chunk])
            entries: list[dict[str, Any]] = []
            for case_id in selected:
                support, _materialization_source_hashes = _load_support_case(
                    release, case_id
                )
                # These release hashes preserve the exact native artifacts used
                # to materialize evaluator-owned geometry and truth. They are
                # provenance, not a prediction authorization: several
                # surrogates can legitimately predict the same physical case.
                # The current surrogate is independently authenticated below
                # against its own receipts, then aligned to the support.
                cp_metrics_path = _safe_child(
                    surface_root,
                    f"{case_id}/surface_submission_stream/cp_profile_metrics.json",
                    f"{case_id} Cp metrics",
                )
                cp_npz_path = _safe_child(
                    surface_root,
                    f"{case_id}/surface_submission_stream/cp_cut_values.npz",
                    f"{case_id} Cp native NPZ",
                )
                velocity_metrics_path = _safe_child(
                    volume_root,
                    (
                        f"{case_id}/volume_submission_stream/"
                        "velocity_profile_metrics.json"
                    ),
                    f"{case_id} velocity metrics",
                )
                velocity_npz_path = _safe_child(
                    volume_root,
                    f"{case_id}/volume_submission_stream/velocity_profiles.npz",
                    f"{case_id} velocity native NPZ",
                )
                _bounded_regular_file(
                    cp_metrics_path,
                    maximum=_MAX_NATIVE_METADATA_JSON_BYTES,
                    label=f"{case_id} Cp metrics",
                )
                _bounded_regular_file(
                    velocity_metrics_path,
                    maximum=_MAX_NATIVE_METADATA_JSON_BYTES,
                    label=f"{case_id} velocity metrics",
                )
                _preflight_npz(
                    cp_npz_path,
                    order=CP_SOURCE_ARRAYS,
                    label=f"{case_id} Cp native NPZ",
                    maximum_archive_bytes=_MAX_NATIVE_PROFILE_NPZ_BYTES,
                    maximum_uncompressed_bytes=(
                        _MAX_NATIVE_PROFILE_UNCOMPRESSED_BYTES
                    ),
                )
                _preflight_npz(
                    velocity_npz_path,
                    order=VELOCITY_SOURCE_ARRAYS,
                    label=f"{case_id} velocity native NPZ",
                    maximum_archive_bytes=_MAX_NATIVE_PROFILE_NPZ_BYTES,
                    maximum_uncompressed_bytes=(
                        _MAX_NATIVE_PROFILE_UNCOMPRESSED_BYTES
                    ),
                )
                try:
                    cp_arrays, _ = validate_cp_source(
                        metrics_path=cp_metrics_path,
                        npz_path=cp_npz_path,
                        expected_metrics_sha256=expected_hashes[case_id][
                            "cp_profile_metrics"
                        ],
                        expected_npz_sha256=expected_hashes[case_id][
                            "cp_cut_values"
                        ],
                    )
                    velocity_arrays, _ = validate_velocity_source(
                        case_id=case_id,
                        metrics_path=velocity_metrics_path,
                        npz_path=velocity_npz_path,
                        expected_metrics_sha256=expected_hashes[case_id][
                            "velocity_profile_metrics"
                        ],
                        expected_npz_sha256=expected_hashes[case_id][
                            "velocity_profiles"
                        ],
                    )
                    artifact = encode_native_predictions(
                        support=support,
                        cp_native=cp_arrays,
                        velocity_native=velocity_arrays,
                    )
                except (NativeProfileError, CompactProfileError) as error:
                    raise CompactProfileEvaluationError(str(error)) from error
                metadata = compact_case_metadata(support)
                artifact_path = (
                    profiles_root
                    / "artifacts"
                    / case_id
                    / "compact-profile-predictions.npz"
                )
                try:
                    artifact_sha = write_compact_prediction_npz(
                        artifact_path,
                        artifact,
                        support=support,
                        metadata=metadata,
                    )
                    case_metrics = score_compact_profiles(
                        artifact, support=support, metadata=metadata
                    )
                except CompactProfileError as error:
                    raise CompactProfileEvaluationError(str(error)) from error
                entries.append(
                    {
                        "case_id": case_id,
                        "artifact": {
                            "format": "numpy-npz-v1",
                            "file": artifact_path.relative_to(
                                profiles_root
                            ).as_posix(),
                            "sha256": artifact_sha,
                            "byte_size": artifact_path.stat().st_size,
                            "array_order": list(PREDICTION_ARRAYS),
                            "ownership": "participant",
                            "content": "predictions_only",
                        },
                        "surface_cp": metadata["surface_cp"],
                        "volume_velocity": metadata["volume_velocity"],
                    }
                )
                metrics[case_id] = case_metrics
            chunk = {
                "schema": COMPACT_PROFILE_CHUNK_SCHEMA,
                "schema_version": COMPACT_PROFILE_SCHEMA_VERSION,
                "format": COMPACT_PROFILE_FORMAT,
                "contract_id": COMPACT_PROFILE_CONTRACT_ID,
                "contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
                "submission_id": checked_submission,
                "dataset_id": "hiliftaeroml",
                "split_id": checked_split,
                "case_set_id": checked_case_set,
                "cases": entries,
            }
            filename = f"chunk-{chunk_number:03d}.json"
            digest = _write_json_exclusive(
                profiles_root / filename, chunk, f"compact profile {filename}"
            )
            chunks.append(
                {"file": filename, "case_ids": selected, "sha256": digest}
            )
        index = {
            "schema_version": COMPACT_PROFILE_INDEX_SCHEMA_VERSION,
            "format": COMPACT_PROFILE_FORMAT,
            "contract_id": COMPACT_PROFILE_CONTRACT_ID,
            "contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
            "submission_id": checked_submission,
            "dataset_id": "hiliftaeroml",
            "split_id": checked_split,
            "case_set_id": checked_case_set,
            "case_count": len(cases),
            "case_id_status": "official",
            "evaluator_support_release_id": release.release_id,
            "evaluator_support_manifest_sha256": release.manifest_sha256,
            "chunks": chunks,
        }
        index_sha = _write_json_exclusive(
            profiles_root / "index.json", index, "compact profile index"
        )
        if final_profiles_root.exists() or final_profiles_root.is_symlink():
            _fail(f"compact profile output already exists: {final_profiles_root}")
        os.rename(profiles_root, final_profiles_root)
        staging = None
        return index_sha, metrics
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise


def _artifact_descriptor(value: Any, *, case_id: str) -> Mapping[str, Any]:
    descriptor = _require_exact_keys(
        value,
        {
            "format",
            "file",
            "sha256",
            "byte_size",
            "array_order",
            "ownership",
            "content",
        },
        f"{case_id} compact prediction artifact",
    )
    if (
        descriptor.get("format") != "numpy-npz-v1"
        or descriptor.get("file")
        != f"artifacts/{case_id}/compact-profile-predictions.npz"
        or descriptor.get("array_order") != list(PREDICTION_ARRAYS)
        or descriptor.get("ownership") != "participant"
        or descriptor.get("content") != "predictions_only"
    ):
        _fail(f"{case_id} compact prediction artifact boundary differs")
    _require_sha(descriptor.get("sha256"), f"{case_id} prediction SHA-256")
    _positive_int(descriptor.get("byte_size"), f"{case_id} prediction byte size")
    return descriptor


def score_compact_profile_directory(
    *,
    profiles_root: Path,
    support_release_root: Path,
    support_manifest_sha256: str,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    expected_case_ids: Sequence[str],
    opened_support_release: CompactSupportRelease | None = None,
) -> dict[str, dict[str, float]]:
    """Strictly validate and score one compact participant profile directory."""

    (
        checked_submission,
        checked_split,
        checked_case_set,
        cases,
        _,
    ) = _validate_builder_inputs(
        submission_id=submission_id,
        split_id=split_id,
        case_set_id=case_set_id,
        case_ids=expected_case_ids,
        cases_per_chunk=1,
        expected_case_artifact_sha256={
            case_id: {key: "0" * 64 for key in SOURCE_ARTIFACT_SHA256_KEYS}
            for case_id in expected_case_ids
        },
    )
    release = _opened_support_release(
        support_release_root=support_release_root,
        support_manifest_sha256=support_manifest_sha256,
        case_ids=cases,
        case_set_id=checked_case_set,
        opened_support_release=opened_support_release,
    )
    root = _root_directory(profiles_root, "compact profile directory")
    index, _ = _load_canonical_json(
        _safe_child(root, "index.json", "compact profile index"),
        "compact profile index",
    )
    _require_exact_keys(
        index,
        {
            "schema_version",
            "format",
            "contract_id",
            "contract_sha256",
            "submission_id",
            "dataset_id",
            "split_id",
            "case_set_id",
            "case_count",
            "case_id_status",
            "evaluator_support_release_id",
            "evaluator_support_manifest_sha256",
            "chunks",
        },
        "compact profile index",
    )
    if (
        index.get("schema_version") != COMPACT_PROFILE_INDEX_SCHEMA_VERSION
        or index.get("format") != COMPACT_PROFILE_FORMAT
        or index.get("contract_id") != COMPACT_PROFILE_CONTRACT_ID
        or index.get("contract_sha256") != COMPACT_PROFILE_CONTRACT_SHA256
        or index.get("submission_id") != checked_submission
        or index.get("dataset_id") != "hiliftaeroml"
        or index.get("split_id") != checked_split
        or index.get("case_set_id") != checked_case_set
        or index.get("case_id_status") != "official"
        or index.get("evaluator_support_release_id") != release.release_id
        or index.get("evaluator_support_manifest_sha256")
        != release.manifest_sha256
    ):
        _fail("compact profile index identity or support binding differs")
    _exact_int(index.get("case_count"), len(cases), "compact profile index case_count")
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        _fail("compact profile index chunks must be a non-empty list")

    expected_files = {"index.json"}
    expected_directories = {"artifacts"}
    ordered_cases: list[str] = []
    scores: dict[str, dict[str, float]] = {}
    for number, raw_descriptor in enumerate(chunks):
        descriptor = _require_exact_keys(
            raw_descriptor,
            {"file", "case_ids", "sha256"},
            f"compact profile chunk {number}",
        )
        expected_filename = f"chunk-{number:03d}.json"
        if descriptor.get("file") != expected_filename:
            _fail("compact profile chunk order or filename differs")
        _require_sha(descriptor.get("sha256"), f"{expected_filename} SHA-256")
        declared_cases = descriptor.get("case_ids")
        if not isinstance(declared_cases, list):
            _fail(f"{expected_filename} case_ids must be a list")
        chunk_cases = _case_ids(
            declared_cases, f"{expected_filename} declared case IDs"
        )
        chunk_path = _safe_child(root, expected_filename, expected_filename)
        chunk, chunk_sha = _load_canonical_json(
            chunk_path, f"compact profile {expected_filename}"
        )
        if chunk_sha != descriptor["sha256"]:
            _fail(f"{expected_filename} SHA-256 differs from index")
        _require_exact_keys(
            chunk,
            {
                "schema",
                "schema_version",
                "format",
                "contract_id",
                "contract_sha256",
                "submission_id",
                "dataset_id",
                "split_id",
                "case_set_id",
                "cases",
            },
            f"compact profile {expected_filename}",
        )
        if (
            chunk.get("schema") != COMPACT_PROFILE_CHUNK_SCHEMA
            or chunk.get("schema_version") != COMPACT_PROFILE_SCHEMA_VERSION
            or chunk.get("format") != COMPACT_PROFILE_FORMAT
            or chunk.get("contract_id") != COMPACT_PROFILE_CONTRACT_ID
            or chunk.get("contract_sha256") != COMPACT_PROFILE_CONTRACT_SHA256
            or chunk.get("submission_id") != checked_submission
            or chunk.get("dataset_id") != "hiliftaeroml"
            or chunk.get("split_id") != checked_split
            or chunk.get("case_set_id") != checked_case_set
        ):
            _fail(f"compact profile {expected_filename} identity differs")
        entries = chunk.get("cases")
        if not isinstance(entries, list) or not entries:
            _fail(f"compact profile {expected_filename} has no cases")
        entry_ids: list[str] = []
        for entry in entries:
            value = _require_exact_keys(
                entry,
                {"case_id", "artifact", "surface_cp", "volume_velocity"},
                f"{expected_filename} compact profile case",
            )
            case_id = value.get("case_id")
            if not isinstance(case_id, str) or SAFE_CASE_ID.fullmatch(case_id) is None:
                _fail(f"{expected_filename} contains an invalid case ID")
            if case_id in scores:
                _fail(f"compact profile case {case_id} is duplicated")
            entry_ids.append(case_id)
            support, _ = _load_support_case(release, case_id)
            metadata = {
                "surface_cp": value.get("surface_cp"),
                "volume_velocity": value.get("volume_velocity"),
            }
            _support_array_contract(
                metadata, label=f"{case_id} compact participant metadata"
            )
            if metadata != compact_case_metadata(support):
                _fail(f"{case_id} compact metadata differs from evaluator support")
            artifact_descriptor = _artifact_descriptor(
                value.get("artifact"), case_id=case_id
            )
            artifact_path = _safe_child(
                root,
                artifact_descriptor["file"],
                f"{case_id} compact prediction artifact",
            )
            if not artifact_path.is_file() or artifact_path.is_symlink():
                _fail(f"{case_id} compact prediction artifact is not regular")
            if artifact_path.stat().st_size != artifact_descriptor["byte_size"]:
                _fail(f"{case_id} compact prediction artifact byte size differs")
            prediction_contract = {
                "cp_q_delta": (
                    np.dtype(np.int16),
                    (len(support["cp_truth"]),),
                ),
                "velocity_speed_over_u_inf": (
                    np.dtype(np.uint8),
                    (
                        int(np.count_nonzero(support["velocity_valid_mask"]))
                        * np.dtype(np.float32).itemsize,
                    ),
                ),
            }
            _preflight_npz(
                artifact_path,
                order=PREDICTION_ARRAYS,
                label=f"{case_id} compact prediction artifact",
                maximum_archive_bytes=_MAX_COMPACT_PREDICTION_NPZ_BYTES,
                maximum_uncompressed_bytes=_MAX_COMPACT_PREDICTION_NPZ_BYTES,
                expected_arrays=prediction_contract,
                require_deflate=True,
            )
            try:
                artifact, artifact_sha = load_compact_prediction_npz(
                    artifact_path, support=support, metadata=metadata
                )
                case_scores = score_compact_profiles(
                    artifact, support=support, metadata=metadata
                )
            except CompactProfileError as error:
                raise CompactProfileEvaluationError(str(error)) from error
            if artifact_sha != artifact_descriptor["sha256"]:
                _fail(f"{case_id} compact prediction artifact SHA-256 differs")
            scores[case_id] = case_scores
            expected_files.add(artifact_descriptor["file"])
            expected_directories.add(f"artifacts/{case_id}")
        if entry_ids != list(chunk_cases):
            _fail(f"{expected_filename} case order differs from index")
        ordered_cases.extend(entry_ids)
        expected_files.add(expected_filename)
    if tuple(ordered_cases) != cases:
        _fail("compact profile case coverage or global order differs")
    observed_files, observed_directories = _tree_inventory(
        root, "compact profile directory"
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        _fail("compact profile file or directory inventory differs")
    return scores


__all__ = [
    "COMPACT_PROFILE_CHUNK_SCHEMA",
    "COMPACT_PROFILE_CONTRACT_ID",
    "COMPACT_PROFILE_CONTRACT_PATH",
    "COMPACT_PROFILE_CONTRACT_SHA256",
    "COMPACT_PROFILE_FORMAT",
    "COMPACT_PROFILE_INDEX_SCHEMA_VERSION",
    "COMPACT_PROFILE_SCHEMA_VERSION",
    "COMPACT_SUPPORT_CASE_SCHEMA",
    "COMPACT_SUPPORT_INDEX_SCHEMA",
    "COMPACT_SUPPORT_MANIFEST_SCHEMA",
    "COMPACT_SUPPORT_RELEASE_FORMAT",
    "COMPACT_SUPPORT_RELEASE_ID",
    "COMPACT_SUPPORT_RELEASE_STATUS",
    "COMPACT_SUPPORT_RELEASE_USAGE",
    "SOURCE_ARTIFACT_SHA256_KEYS",
    "CompactProfileEvaluationError",
    "CompactSupportRelease",
    "build_compact_profile_directory",
    "open_compact_support_release",
    "score_compact_profile_directory",
    "write_compact_support_release",
]
