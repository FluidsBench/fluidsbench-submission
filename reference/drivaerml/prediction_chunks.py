"""Candidate local NPZ transport for native DrivAerML predictions.

This is an evaluator input format, not an official FluidsBench prediction
artifact.  One JSON manifest describes exactly one ``case_id`` and one native
``CellData`` support.  Its shape is intentionally small and closed::

    {
      "format": "drivaerml-native-prediction-chunks-candidate",
      "format_version": 1,
      "artifact_role": "local_evaluator_input_not_official_submission_artifact",
      "case_id": "run_1",
      "support_id": "volume_native_cells",
      "association": "CellData",
      "total_row_count": 147449586,
      "field_components": {"pMeanTrim": 1, "UMeanTrim": 3},
      "chunks": [
        {
          "chunk_index": 0,
          "file": "chunks/chunk-00000.npz",
          "sha256": "...",
          "row_count": 1000000,
          "raw_cell_id_start": 0,
          "raw_cell_id_stop": 1000000
        }
      ]
    }

Every NPZ contains ``raw_cell_id`` plus exactly the two fields prescribed for
its support.  Raw IDs are signed int64 and each file covers the half-open
interval declared in the manifest.  Manifest intervals must form ``[0, N)``
in source order, so a fully consumed iterator proves gap-free, duplicate-free
coverage without retaining IDs from earlier chunks.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import struct
import zipfile
from dataclasses import dataclass, field
from math import prod
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO, Iterator, Mapping

import numpy as np


CANDIDATE_FORMAT = "drivaerml-native-prediction-chunks-candidate"
CANDIDATE_FORMAT_VERSION = 1
CANDIDATE_ARTIFACT_ROLE = (
    "local_evaluator_input_not_official_submission_artifact"
)
RAW_CELL_ID_FIELD = "raw_cell_id"
DEFAULT_HASH_CHUNK_BYTES = 1024 * 1024
DEFAULT_VALIDATION_BLOCK_ROWS = 1024 * 1024
DEFAULT_MAX_NPY_HEADER_BYTES = 4096
DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES = 64 * 1024
DEFAULT_MAX_MANIFEST_BYTES = 16 * 1024 * 1024

_NPY_MAGIC = b"\x93NUMPY"
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP_EOCD_BYTES = 22
_ZIP_MAX_COMMENT_BYTES = 65535
_ZIP_MAX_CANDIDATE_MEMBER_COUNT = 16
_ALLOWED_ZIP_COMPRESSION = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID_RE = re.compile(r"run_[1-9][0-9]*\Z")

_SUPPORT_FIELD_COMPONENTS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        "surface_native_cells": MappingProxyType(
            {
                "pMeanTrim": 1,
                "wallShearStressMeanTrim": 3,
            }
        ),
        "volume_native_cells": MappingProxyType(
            {
                "pMeanTrim": 1,
                "UMeanTrim": 3,
            }
        ),
    }
)

_MANIFEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "artifact_role",
        "case_id",
        "support_id",
        "association",
        "total_row_count",
        "field_components",
        "chunks",
    }
)
_CHUNK_KEYS = frozenset(
    {
        "chunk_index",
        "file",
        "sha256",
        "row_count",
        "raw_cell_id_start",
        "raw_cell_id_stop",
    }
)


class PredictionChunkError(ValueError):
    """Raised when candidate prediction chunks violate their local contract."""


@dataclass(frozen=True)
class PredictionChunkDescriptor:
    """Pinned identity and raw-cell interval for one NPZ file."""

    chunk_index: int
    relative_file: PurePosixPath
    path: Path
    sha256: str
    row_count: int
    raw_cell_id_start: int
    raw_cell_id_stop: int


@dataclass(frozen=True)
class PredictionChunkManifest:
    """Validated one-case, one-support candidate manifest."""

    path: Path
    sha256: str
    case_id: str
    support_id: str
    association: str
    total_row_count: int
    field_components: Mapping[str, int]
    chunks: tuple[PredictionChunkDescriptor, ...]


@dataclass(frozen=True)
class PredictionChunk:
    """One verified native prediction chunk held in memory."""

    case_id: str
    support_id: str
    descriptor: PredictionChunkDescriptor
    raw_cell_id: np.ndarray
    fields: Mapping[str, np.ndarray]

    def field(self, name: str) -> np.ndarray:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise PredictionChunkError(
                f"chunk {self.descriptor.chunk_index} has no field {name!r}"
            ) from exc


@dataclass(frozen=True)
class PredictionChunkValidation:
    """Compact evidence after every chunk in a manifest has been consumed."""

    manifest_path: Path
    manifest_sha256: str
    case_id: str
    support_id: str
    total_row_count: int
    chunk_count: int
    chunk_sha256: tuple[str, ...]
    complete_gap_free_duplicate_free_coverage: bool = field(default=True)


@dataclass(frozen=True)
class _NpyHeader:
    """Allocation-free metadata parsed from one bounded NPY header."""

    dtype: np.dtype
    shape: tuple[int, ...]
    fortran_order: bool
    header_and_preamble_bytes: int


@dataclass(frozen=True)
class _OpenFileSnapshot:
    """Mutation-sensitive identity for one retained participant input."""

    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_handle(cls, source: BinaryIO, *, context: str) -> "_OpenFileSnapshot":
        try:
            value = os.fstat(source.fileno())
        except OSError as exc:
            raise PredictionChunkError(f"cannot fstat {context}: {exc}") from exc
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )


def _open_snapshot(source: BinaryIO, *, context: str) -> _OpenFileSnapshot:
    snapshot = _OpenFileSnapshot.from_handle(source, context=context)
    if not stat.S_ISREG(snapshot.mode):
        raise PredictionChunkError(f"{context} must be a regular file")
    return snapshot


def _assert_snapshot(
    source: BinaryIO,
    snapshot: _OpenFileSnapshot,
    *,
    context: str,
) -> None:
    if _OpenFileSnapshot.from_handle(source, context=context) != snapshot:
        raise PredictionChunkError(
            f"{context} changed while it was hashed and parsed; device, inode, "
            "mode, size, mtime, or ctime differs from the retained pre-read snapshot"
        )


def support_field_components(support_id: str) -> Mapping[str, int]:
    """Return the immutable candidate field/component contract for a support."""

    try:
        return _SUPPORT_FIELD_COMPONENTS[support_id]
    except KeyError as exc:
        expected = ", ".join(sorted(_SUPPORT_FIELD_COMPONENTS))
        raise PredictionChunkError(
            f"unknown support_id {support_id!r}; expected one of {expected}"
        ) from exc


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PredictionChunkError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _closed_keys(
    value: object,
    *,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PredictionChunkError(f"{context} must be a JSON object")
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        raise PredictionChunkError(f"{context}: {'; '.join(details)}")
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PredictionChunkError(f"{context} must be a positive integer")
    if value > np.iinfo(np.int64).max:
        raise PredictionChunkError(f"{context} exceeds the signed int64 range")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PredictionChunkError(f"{context} must be a non-negative integer")
    if value > np.iinfo(np.int64).max:
        raise PredictionChunkError(f"{context} exceeds the signed int64 range")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PredictionChunkError(f"{context} must be a non-empty string")
    return value


def _relative_npz_path(value: object, context: str) -> PurePosixPath:
    text = _string(value, context)
    if "\\" in text:
        raise PredictionChunkError(f"{context} must use POSIX path separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or text != path.as_posix()
    ):
        raise PredictionChunkError(
            f"{context} must be a normalized relative path without traversal"
        )
    if path.suffix != ".npz":
        raise PredictionChunkError(f"{context} must name an .npz file")
    return path


def _resolve_npz(root: Path, relative: PurePosixPath, context: str) -> Path:
    unresolved = root.joinpath(*relative.parts)
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise PredictionChunkError(f"{context} does not exist: {unresolved}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PredictionChunkError(f"{context} escapes the manifest directory") from exc
    if not resolved.is_file():
        raise PredictionChunkError(f"{context} is not a regular file")
    return resolved


def _field_component_mapping(
    value: object,
    support_id: str,
) -> Mapping[str, int]:
    if not isinstance(value, dict):
        raise PredictionChunkError("field_components must be a JSON object")
    expected = support_field_components(support_id)
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise PredictionChunkError(f"field_components: {'; '.join(details)}")
    parsed: dict[str, int] = {}
    for name, components in expected.items():
        actual = value[name]
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != components
        ):
            raise PredictionChunkError(
                f"field_components.{name} must equal {components}"
            )
        parsed[name] = components
    return MappingProxyType(parsed)


def load_prediction_chunk_manifest(
    path: str | Path,
) -> PredictionChunkManifest:
    """Load and structurally validate one candidate chunk manifest.

    Paths are resolved at load time and cannot leave the manifest directory,
    including through symlinks.  File bytes and NPZ arrays are verified lazily
    by :func:`iter_prediction_chunks`, one chunk at a time.
    """

    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open("rb", buffering=0) as source:
            snapshot = _open_snapshot(source, context="prediction chunk manifest")
            if snapshot.size_bytes > DEFAULT_MAX_MANIFEST_BYTES:
                raise PredictionChunkError(
                    "prediction chunk manifest exceeds the "
                    f"{DEFAULT_MAX_MANIFEST_BYTES}-byte limit"
                )
            encoded = source.read(DEFAULT_MAX_MANIFEST_BYTES + 1)
            if len(encoded) != snapshot.size_bytes:
                raise PredictionChunkError(
                    "prediction chunk manifest size changed while it was read"
                )
            manifest_sha256 = hashlib.sha256(encoded).hexdigest()
            _assert_snapshot(
                source,
                snapshot,
                context="prediction chunk manifest",
            )
        document = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except PredictionChunkError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionChunkError(
            f"cannot load prediction chunk manifest {manifest_path}: {exc}"
        ) from exc

    manifest = _closed_keys(
        document,
        expected=_MANIFEST_KEYS,
        context="prediction chunk manifest",
    )
    if manifest["format"] != CANDIDATE_FORMAT:
        raise PredictionChunkError(
            f"format must be {CANDIDATE_FORMAT!r}; this loader does not read "
            "official prediction artifacts"
        )
    if (
        isinstance(manifest["format_version"], bool)
        or not isinstance(manifest["format_version"], int)
        or manifest["format_version"] != CANDIDATE_FORMAT_VERSION
    ):
        raise PredictionChunkError(
            f"format_version must equal {CANDIDATE_FORMAT_VERSION}"
        )
    if manifest["artifact_role"] != CANDIDATE_ARTIFACT_ROLE:
        raise PredictionChunkError(
            f"artifact_role must be {CANDIDATE_ARTIFACT_ROLE!r}"
        )
    case_id = _string(manifest["case_id"], "case_id")
    if _CASE_ID_RE.fullmatch(case_id) is None:
        raise PredictionChunkError("case_id must have the form run_<positive integer>")
    support_id = _string(manifest["support_id"], "support_id")
    expected_fields = _field_component_mapping(
        manifest["field_components"], support_id
    )
    if manifest["association"] != "CellData":
        raise PredictionChunkError("association must equal 'CellData'")
    total_rows = _positive_integer(manifest["total_row_count"], "total_row_count")

    chunk_values = manifest["chunks"]
    if not isinstance(chunk_values, list) or not chunk_values:
        raise PredictionChunkError("chunks must be a non-empty JSON array")
    root = manifest_path.parent.resolve()
    descriptors: list[PredictionChunkDescriptor] = []
    seen_paths: set[Path] = set()
    expected_start = 0
    for position, chunk_value in enumerate(chunk_values):
        context = f"chunks[{position}]"
        chunk = _closed_keys(
            chunk_value,
            expected=_CHUNK_KEYS,
            context=context,
        )
        chunk_index = _nonnegative_integer(
            chunk["chunk_index"], f"{context}.chunk_index"
        )
        if chunk_index != position:
            raise PredictionChunkError(
                f"{context}.chunk_index must equal its manifest position {position}"
            )
        relative = _relative_npz_path(chunk["file"], f"{context}.file")
        resolved = _resolve_npz(root, relative, f"{context}.file")
        if resolved in seen_paths:
            raise PredictionChunkError(
                f"{context}.file resolves to a file already used by another chunk"
            )
        seen_paths.add(resolved)
        digest = _string(chunk["sha256"], f"{context}.sha256")
        if _SHA256_RE.fullmatch(digest) is None:
            raise PredictionChunkError(
                f"{context}.sha256 must be a lowercase SHA-256 digest"
            )
        row_count = _positive_integer(chunk["row_count"], f"{context}.row_count")
        start = _nonnegative_integer(
            chunk["raw_cell_id_start"], f"{context}.raw_cell_id_start"
        )
        stop = _positive_integer(
            chunk["raw_cell_id_stop"], f"{context}.raw_cell_id_stop"
        )
        if start != expected_start:
            relation = "gap" if start > expected_start else "overlap or duplicate"
            raise PredictionChunkError(
                f"{context} creates a raw-cell coverage {relation}: "
                f"expected start {expected_start}, found {start}"
            )
        if stop <= start:
            raise PredictionChunkError(
                f"{context}.raw_cell_id_stop must exceed raw_cell_id_start"
            )
        if stop - start != row_count:
            raise PredictionChunkError(
                f"{context}.row_count does not equal its raw-cell interval length"
            )
        descriptors.append(
            PredictionChunkDescriptor(
                chunk_index=chunk_index,
                relative_file=relative,
                path=resolved,
                sha256=digest,
                row_count=row_count,
                raw_cell_id_start=start,
                raw_cell_id_stop=stop,
            )
        )
        expected_start = stop

    if expected_start != total_rows:
        raise PredictionChunkError(
            f"chunk coverage stops at raw cell {expected_start}, "
            f"but total_row_count is {total_rows}"
        )
    return PredictionChunkManifest(
        path=manifest_path,
        sha256=manifest_sha256,
        case_id=case_id,
        support_id=support_id,
        association="CellData",
        total_row_count=total_rows,
        field_components=expected_fields,
        chunks=tuple(descriptors),
    )


def _sha256_open_file(source: BinaryIO, chunk_size: int) -> str:
    if chunk_size < 1:
        raise PredictionChunkError("hash_chunk_bytes must be positive")
    source.seek(0)
    digest = hashlib.sha256()
    while True:
        block = source.read(chunk_size)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _read_exact_member_bytes(
    source: BinaryIO,
    count: int,
    *,
    context: str,
) -> bytes:
    value = source.read(count)
    if len(value) != count:
        raise PredictionChunkError(
            f"{context} ends before its declared NPY header is complete"
        )
    return value


def _read_bounded_npy_header(
    source: BinaryIO,
    *,
    context: str,
    max_header_bytes: int,
) -> _NpyHeader:
    """Parse a simple NPY header without asking NumPy to allocate an array."""

    magic = _read_exact_member_bytes(source, len(_NPY_MAGIC), context=context)
    if magic != _NPY_MAGIC:
        raise PredictionChunkError(f"{context} does not start with NPY magic")
    version_bytes = _read_exact_member_bytes(source, 2, context=context)
    version = (version_bytes[0], version_bytes[1])
    if version == (1, 0):
        length_size = 2
        encoding = "latin1"
        length_format = "<H"
    elif version in {(2, 0), (3, 0)}:
        length_size = 4
        encoding = "latin1" if version == (2, 0) else "utf-8"
        length_format = "<I"
    else:
        raise PredictionChunkError(
            f"{context} uses unsupported NPY version {version[0]}.{version[1]}"
        )
    length_bytes = _read_exact_member_bytes(
        source,
        length_size,
        context=context,
    )
    header_length = struct.unpack(length_format, length_bytes)[0]
    if header_length < 1 or header_length > max_header_bytes:
        raise PredictionChunkError(
            f"{context} NPY header length {header_length} exceeds the "
            f"{max_header_bytes}-byte limit"
        )
    encoded_header = _read_exact_member_bytes(
        source,
        header_length,
        context=context,
    )
    if not encoded_header.endswith(b"\n"):
        raise PredictionChunkError(f"{context} NPY header must end with a newline")
    try:
        header_text = encoded_header.decode(encoding)
        header = ast.literal_eval(header_text.strip())
    except (
        UnicodeDecodeError,
        SyntaxError,
        ValueError,
        TypeError,
        MemoryError,
        RecursionError,
        OverflowError,
    ) as exc:
        raise PredictionChunkError(
            f"{context} contains an invalid bounded NPY header: {exc}"
        ) from exc
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise PredictionChunkError(
            f"{context} NPY header must contain exactly descr, fortran_order, "
            "and shape"
        )
    if not isinstance(header["fortran_order"], bool):
        raise PredictionChunkError(
            f"{context} NPY fortran_order must be a boolean"
        )
    if header["fortran_order"]:
        raise PredictionChunkError(
            f"{context} must use C-order, not Fortran-order, array storage"
        )
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in shape
    ):
        raise PredictionChunkError(
            f"{context} NPY shape must be a tuple of non-negative integers"
        )
    descriptor = header["descr"]
    if not isinstance(descriptor, str):
        raise PredictionChunkError(
            f"{context} must use a simple scalar NPY dtype descriptor"
        )
    try:
        dtype = np.dtype(descriptor)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise PredictionChunkError(
            f"{context} contains an invalid NPY dtype descriptor: {exc}"
        ) from exc
    if dtype.fields is not None or dtype.subdtype is not None or dtype.hasobject:
        raise PredictionChunkError(
            f"{context} must use a simple scalar dtype; object dtype is forbidden"
        )
    return _NpyHeader(
        dtype=dtype,
        shape=shape,
        fortran_order=False,
        header_and_preamble_bytes=(
            len(_NPY_MAGIC) + 2 + length_size + header_length
        ),
    )


def _expected_member_shape(
    name: str,
    *,
    descriptor: PredictionChunkDescriptor,
    field_components: Mapping[str, int],
) -> tuple[int, ...]:
    if name == RAW_CELL_ID_FIELD:
        return (descriptor.row_count,)
    components = field_components[name]
    if components == 1:
        return (descriptor.row_count,)
    return (descriptor.row_count, components)


def _validate_preflight_dtype(
    dtype: np.dtype,
    *,
    name: str,
    context: str,
) -> None:
    if name == RAW_CELL_ID_FIELD:
        if dtype.kind != "i" or dtype.itemsize != 8:
            raise PredictionChunkError(
                f"{context} must have signed int64 dtype"
            )
        return
    if dtype.kind != "f" or dtype.itemsize not in {4, 8}:
        raise PredictionChunkError(
            f"{context} must use Float32 or Float64, found {dtype}"
        )


def _validate_local_zip_header(
    source: BinaryIO,
    member: zipfile.ZipInfo,
    *,
    context: str,
) -> None:
    """Reject dangerous local-header flags hidden by benign central metadata."""

    source.seek(member.header_offset)
    fixed_header = source.read(30)
    if len(fixed_header) != 30 or fixed_header[:4] != b"PK\x03\x04":
        raise PredictionChunkError(
            f"{context} member {member.filename!r} has an invalid local ZIP header"
        )
    local_flags = struct.unpack_from("<H", fixed_header, 6)[0]
    local_compression = struct.unpack_from("<H", fixed_header, 8)[0]
    if local_flags != member.flag_bits:
        raise PredictionChunkError(
            f"{context} member {member.filename!r} has inconsistent local and "
            "central ZIP flags"
        )
    if local_flags & 0x1:
        raise PredictionChunkError(f"{context} contains encrypted ZIP members")
    if local_flags & 0x8:
        raise PredictionChunkError(
            f"{context} contains ZIP data-descriptor members"
        )
    if local_compression != member.compress_type:
        raise PredictionChunkError(
            f"{context} member {member.filename!r} has inconsistent local and "
            "central ZIP compression methods"
        )


def _preflight_zip_directory(
    source: BinaryIO,
    *,
    expected_member_count: int,
    context: str,
    max_central_directory_bytes: int,
) -> None:
    """Bound central-directory parsing before constructing ``ZipFile``."""

    if max_central_directory_bytes < 1:
        raise PredictionChunkError(
            "max_npz_central_directory_bytes must be positive"
        )
    source.seek(0, 2)
    archive_size = source.tell()
    if archive_size < _ZIP_EOCD_BYTES:
        raise PredictionChunkError(f"{context} is too short to be an NPZ archive")
    tail_size = min(
        archive_size,
        _ZIP_EOCD_BYTES + _ZIP_MAX_COMMENT_BYTES,
    )
    source.seek(archive_size - tail_size)
    tail = source.read(tail_size)
    if len(tail) != tail_size:
        raise PredictionChunkError(f"{context} could not be read completely")

    search_stop = len(tail)
    eocd_offset = -1
    while search_stop >= len(_ZIP_EOCD_SIGNATURE):
        candidate = tail.rfind(_ZIP_EOCD_SIGNATURE, 0, search_stop)
        if candidate < 0:
            break
        if candidate + _ZIP_EOCD_BYTES <= len(tail):
            comment_length = struct.unpack_from("<H", tail, candidate + 20)[0]
            if candidate + _ZIP_EOCD_BYTES + comment_length == len(tail):
                eocd_offset = candidate
                break
        search_stop = candidate
    if eocd_offset < 0:
        raise PredictionChunkError(
            f"{context} has no valid bounded ZIP end-of-central-directory record"
        )

    (
        disk_number,
        directory_disk,
        entries_on_disk,
        total_entries,
        directory_size,
        directory_offset,
    ) = struct.unpack_from("<4H2L", tail, eocd_offset + 4)
    if disk_number != 0 or directory_disk != 0 or entries_on_disk != total_entries:
        raise PredictionChunkError(f"{context} must be a single-disk ZIP archive")
    if total_entries in {0xFFFF} or directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        raise PredictionChunkError(
            f"{context} may use ZIP64 members but not a ZIP64 central directory"
        )
    if total_entries < 1 or total_entries > max(
        expected_member_count,
        _ZIP_MAX_CANDIDATE_MEMBER_COUNT,
    ):
        raise PredictionChunkError(
            f"{context} declares {total_entries} ZIP members, exceeding the "
            f"bounded candidate archive limit of "
            f"{max(expected_member_count, _ZIP_MAX_CANDIDATE_MEMBER_COUNT)}"
        )
    if directory_size > max_central_directory_bytes:
        raise PredictionChunkError(
            f"{context} central directory declares {directory_size} bytes, "
            f"exceeding the {max_central_directory_bytes}-byte limit"
        )
    absolute_eocd_offset = archive_size - tail_size + eocd_offset
    if directory_offset + directory_size != absolute_eocd_offset:
        raise PredictionChunkError(
            f"{context} central-directory extent is inconsistent with its "
            "end record"
        )


def _validate_npz_members(
    source: BinaryIO,
    *,
    descriptor: PredictionChunkDescriptor,
    field_components: Mapping[str, int],
    context: str,
    max_header_bytes: int,
    max_uncompressed_bytes: int,
    max_central_directory_bytes: int,
) -> None:
    if max_header_bytes < 1:
        raise PredictionChunkError("max_npy_header_bytes must be positive")
    if max_uncompressed_bytes < 1:
        raise PredictionChunkError(
            "max_npz_uncompressed_bytes must be positive"
        )
    expected_fields = frozenset(
        {RAW_CELL_ID_FIELD, *field_components.keys()}
    )
    _preflight_zip_directory(
        source,
        expected_member_count=len(expected_fields),
        context=context,
        max_central_directory_bytes=max_central_directory_bytes,
    )
    source.seek(0)
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if len(member_names) != len(set(member_names)):
                raise PredictionChunkError(f"{context} has duplicate ZIP members")
            for member in members:
                if (
                    member.is_dir()
                    or "/" in member.filename
                    or "\\" in member.filename
                    or PurePosixPath(member.filename).name != member.filename
                ):
                    raise PredictionChunkError(
                        f"{context} contains a non-flat ZIP member "
                        f"{member.filename!r}"
                    )
                if member.flag_bits & 0x1:
                    raise PredictionChunkError(
                        f"{context} contains encrypted ZIP members"
                    )
                if member.flag_bits & 0x8:
                    raise PredictionChunkError(
                        f"{context} contains ZIP data-descriptor members"
                    )
                if member.compress_type not in _ALLOWED_ZIP_COMPRESSION:
                    raise PredictionChunkError(
                        f"{context} member {member.filename!r} uses unsupported "
                        f"ZIP compression method {member.compress_type}"
                    )
                _validate_local_zip_header(source, member, context=context)
                if (
                    member.compress_type == zipfile.ZIP_STORED
                    and member.compress_size != member.file_size
                ):
                    raise PredictionChunkError(
                        f"{context} stored member {member.filename!r} has "
                        "inconsistent compressed and uncompressed sizes"
                    )
            expected_members = {f"{name}.npy" for name in expected_fields}
            actual_members = set(member_names)
            if actual_members != expected_members:
                unknown = sorted(actual_members - expected_members)
                missing = sorted(expected_members - actual_members)
                details: list[str] = []
                if missing:
                    details.append(f"missing arrays {missing}")
                if unknown:
                    details.append(f"unknown arrays {unknown}")
                raise PredictionChunkError(f"{context}: {'; '.join(details)}")

            total_uncompressed = sum(member.file_size for member in members)
            if total_uncompressed > max_uncompressed_bytes:
                raise PredictionChunkError(
                    f"{context} declares {total_uncompressed} uncompressed bytes, "
                    f"exceeding the {max_uncompressed_bytes}-byte per-chunk limit"
                )

            for member in members:
                name = member.filename[:-4]
                expected_shape = _expected_member_shape(
                    name,
                    descriptor=descriptor,
                    field_components=field_components,
                )
                maximum_itemsize = 8
                maximum_payload_bytes = (
                    prod(expected_shape) * maximum_itemsize
                )
                maximum_member_bytes = (
                    12 + max_header_bytes + maximum_payload_bytes
                )
                if member.file_size > maximum_member_bytes:
                    raise PredictionChunkError(
                        f"{context} member {member.filename!r} declares "
                        f"{member.file_size} uncompressed bytes, exceeding its "
                        f"shape-derived {maximum_member_bytes}-byte bound"
                    )
                member_context = f"{context} array {name!r}"
                with archive.open(member, mode="r") as member_source:
                    header = _read_bounded_npy_header(
                        member_source,
                        context=member_context,
                        max_header_bytes=max_header_bytes,
                    )
                _validate_preflight_dtype(
                    header.dtype,
                    name=name,
                    context=member_context,
                )
                if header.shape != expected_shape:
                    raise PredictionChunkError(
                        f"{member_context} must have shape {expected_shape}, "
                        f"found {header.shape}"
                    )
                payload_bytes = (
                    prod(header.shape) * header.dtype.itemsize
                )
                expected_member_bytes = (
                    header.header_and_preamble_bytes + payload_bytes
                )
                if member.file_size != expected_member_bytes:
                    raise PredictionChunkError(
                        f"{member_context} has uncompressed size "
                        f"{member.file_size}, but its NPY header requires exactly "
                        f"{expected_member_bytes} bytes"
                    )
    except PredictionChunkError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PredictionChunkError(f"{context} is not a valid NPZ archive: {exc}") from exc


def _validate_raw_cell_ids(
    values: np.ndarray,
    descriptor: PredictionChunkDescriptor,
    *,
    validation_block_rows: int,
) -> np.ndarray:
    context = f"chunk {descriptor.chunk_index} raw_cell_id"
    if values.dtype.kind != "i" or values.dtype.itemsize != 8:
        raise PredictionChunkError(f"{context} must have signed int64 dtype")
    if values.ndim != 1 or values.shape != (descriptor.row_count,):
        raise PredictionChunkError(
            f"{context} must have shape ({descriptor.row_count},), "
            f"found {values.shape}"
        )
    for offset in range(0, descriptor.row_count, validation_block_rows):
        stop = min(offset + validation_block_rows, descriptor.row_count)
        expected = np.arange(
            descriptor.raw_cell_id_start + offset,
            descriptor.raw_cell_id_start + stop,
            dtype=np.int64,
        )
        if not np.array_equal(values[offset:stop], expected):
            raise PredictionChunkError(
                f"{context} is not the exact contiguous interval "
                f"[{descriptor.raw_cell_id_start}, {descriptor.raw_cell_id_stop})"
            )
    values.setflags(write=False)
    return values


def _validate_field(
    values: np.ndarray,
    *,
    field_name: str,
    components: int,
    row_count: int,
    chunk_index: int,
    validation_block_rows: int,
) -> np.ndarray:
    context = f"chunk {chunk_index} field {field_name!r}"
    if values.dtype.hasobject:
        raise PredictionChunkError(f"{context} must not use object dtype")
    if values.dtype.kind != "f" or values.dtype.itemsize not in {4, 8}:
        raise PredictionChunkError(
            f"{context} must use Float32 or Float64, found {values.dtype}"
        )
    expected_shape = (row_count,) if components == 1 else (row_count, components)
    if values.shape != expected_shape:
        raise PredictionChunkError(
            f"{context} must have shape {expected_shape}, found {values.shape}"
        )
    for offset in range(0, row_count, validation_block_rows):
        stop = min(offset + validation_block_rows, row_count)
        if not np.all(np.isfinite(values[offset:stop])):
            raise PredictionChunkError(f"{context} contains non-finite values")
    values.setflags(write=False)
    return values


def _load_prediction_chunk(
    manifest: PredictionChunkManifest,
    descriptor: PredictionChunkDescriptor,
    *,
    hash_chunk_bytes: int,
    validation_block_rows: int,
    max_npy_header_bytes: int,
    max_npz_uncompressed_bytes: int,
    max_npz_central_directory_bytes: int,
) -> PredictionChunk:
    if validation_block_rows < 1:
        raise PredictionChunkError("validation_block_rows must be positive")
    expected_names = frozenset(
        {RAW_CELL_ID_FIELD, *manifest.field_components.keys()}
    )
    context = f"chunk {descriptor.chunk_index} file {descriptor.relative_file.as_posix()!r}"
    try:
        with descriptor.path.open("rb", buffering=0) as source:
            snapshot = _open_snapshot(source, context=context)
            actual_digest = _sha256_open_file(source, hash_chunk_bytes)
            if actual_digest != descriptor.sha256:
                raise PredictionChunkError(
                    f"{context} SHA-256 mismatch: expected {descriptor.sha256}, "
                    f"found {actual_digest}"
                )
            _assert_snapshot(source, snapshot, context=context)
            _validate_npz_members(
                source,
                descriptor=descriptor,
                field_components=manifest.field_components,
                context=context,
                max_header_bytes=max_npy_header_bytes,
                max_uncompressed_bytes=max_npz_uncompressed_bytes,
                max_central_directory_bytes=max_npz_central_directory_bytes,
            )
            _assert_snapshot(source, snapshot, context=context)
            source.seek(0)
            try:
                with np.load(source, allow_pickle=False) as archive:
                    archive_names = frozenset(archive.files)
                    if archive_names != expected_names:
                        raise PredictionChunkError(
                            f"{context} contains unknown or missing NPZ fields"
                        )
                    raw_cell_id = _validate_raw_cell_ids(
                        archive[RAW_CELL_ID_FIELD],
                        descriptor,
                        validation_block_rows=validation_block_rows,
                    )
                    fields = {
                        name: _validate_field(
                            archive[name],
                            field_name=name,
                            components=components,
                            row_count=descriptor.row_count,
                            chunk_index=descriptor.chunk_index,
                            validation_block_rows=validation_block_rows,
                        )
                        for name, components in manifest.field_components.items()
                    }
                _assert_snapshot(source, snapshot, context=context)
            except PredictionChunkError:
                raise
            except (ValueError, zipfile.BadZipFile, EOFError) as exc:
                message = str(exc)
                if "Object arrays cannot be loaded" in message:
                    raise PredictionChunkError(
                        f"{context} contains object dtype; pickle loading is forbidden"
                    ) from exc
                raise PredictionChunkError(
                    f"{context} contains an invalid NPY array: {exc}"
                ) from exc
    except PredictionChunkError:
        raise
    except OSError as exc:
        raise PredictionChunkError(f"cannot read {context}: {exc}") from exc
    return PredictionChunk(
        case_id=manifest.case_id,
        support_id=manifest.support_id,
        descriptor=descriptor,
        raw_cell_id=raw_cell_id,
        fields=MappingProxyType(fields),
    )


def iter_prediction_chunks(
    manifest: PredictionChunkManifest | str | Path,
    *,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
    max_npy_header_bytes: int = DEFAULT_MAX_NPY_HEADER_BYTES,
    max_npz_uncompressed_bytes: int = DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES,
    max_npz_central_directory_bytes: int = (
        DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES
    ),
) -> Iterator[PredictionChunk]:
    """Yield fully verified chunks in contiguous raw-cell order.

    The iterator opens, hashes, loads, and closes one NPZ before yielding it.
    It releases its reference to that chunk before opening the next file.
    Callers that retain yielded arrays can of course consume additional memory.
    """

    parsed = (
        manifest
        if isinstance(manifest, PredictionChunkManifest)
        else load_prediction_chunk_manifest(manifest)
    )
    for descriptor in parsed.chunks:
        chunk = _load_prediction_chunk(
            parsed,
            descriptor,
            hash_chunk_bytes=hash_chunk_bytes,
            validation_block_rows=validation_block_rows,
            max_npy_header_bytes=max_npy_header_bytes,
            max_npz_uncompressed_bytes=max_npz_uncompressed_bytes,
            max_npz_central_directory_bytes=max_npz_central_directory_bytes,
        )
        yield chunk
        del chunk


def validate_prediction_chunks(
    manifest: PredictionChunkManifest | str | Path,
    *,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
    max_npy_header_bytes: int = DEFAULT_MAX_NPY_HEADER_BYTES,
    max_npz_uncompressed_bytes: int = DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES,
    max_npz_central_directory_bytes: int = (
        DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES
    ),
) -> PredictionChunkValidation:
    """Exhaust a candidate manifest and return compact validation evidence."""

    parsed = (
        manifest
        if isinstance(manifest, PredictionChunkManifest)
        else load_prediction_chunk_manifest(manifest)
    )
    rows = 0
    chunks = 0
    digests: list[str] = []
    for chunk in iter_prediction_chunks(
        parsed,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
        max_npy_header_bytes=max_npy_header_bytes,
        max_npz_uncompressed_bytes=max_npz_uncompressed_bytes,
        max_npz_central_directory_bytes=max_npz_central_directory_bytes,
    ):
        rows += chunk.descriptor.row_count
        chunks += 1
        digests.append(chunk.descriptor.sha256)
        del chunk
    if rows != parsed.total_row_count or chunks != len(parsed.chunks):
        raise PredictionChunkError("internal error while validating chunk coverage")
    return PredictionChunkValidation(
        manifest_path=parsed.path,
        manifest_sha256=parsed.sha256,
        case_id=parsed.case_id,
        support_id=parsed.support_id,
        total_row_count=rows,
        chunk_count=chunks,
        chunk_sha256=tuple(digests),
    )


__all__ = [
    "CANDIDATE_ARTIFACT_ROLE",
    "CANDIDATE_FORMAT",
    "CANDIDATE_FORMAT_VERSION",
    "DEFAULT_HASH_CHUNK_BYTES",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_NPY_HEADER_BYTES",
    "DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES",
    "DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES",
    "DEFAULT_VALIDATION_BLOCK_ROWS",
    "PredictionChunk",
    "PredictionChunkDescriptor",
    "PredictionChunkError",
    "PredictionChunkManifest",
    "PredictionChunkValidation",
    "RAW_CELL_ID_FIELD",
    "iter_prediction_chunks",
    "load_prediction_chunk_manifest",
    "support_field_components",
    "validate_prediction_chunks",
]
