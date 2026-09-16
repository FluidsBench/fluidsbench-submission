"""Pinned native-source transport and inline-binary VTK XML support.

This module deliberately does not depend on VTK.  It covers the byte-level
portion of the DrivAerML contract:

* validate the immutable native-source pin and resolve its repository paths;
* expose multipart volumes, or a proven monolithic reconstruction, as one
  seekable stream without copying the logical file;
* verify every transport part/segment by exact size and SHA-256; and
* index and decode inline-binary VTK XML arrays with bounded buffers.

The module does not interpret mesh topology.  In particular, it preserves the
order in which ``Piece`` and ``DataArray`` elements occur in the source and
reports the declared native tuple counts to downstream loaders.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import hashlib
import html
import io
import json
import os
import re
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence


DEFAULT_IO_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_XML_TAG_BYTES = 1024 * 1024

_PIN_SCHEMA = "drivaerml-fluidsbench-public-native-source-pin-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_TAG_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*")
_ATTRIBUTE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.:-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"
)
_XML_WHITESPACE = b" \t\r\n"

_VTK_SCALAR_WIDTHS = MappingProxyType(
    {
        "Int8": 1,
        "UInt8": 1,
        "Int16": 2,
        "UInt16": 2,
        "Int32": 4,
        "UInt32": 4,
        "Int64": 8,
        "UInt64": 8,
        "Float32": 4,
        "Float64": 8,
    }
)


class NativeSourceError(ValueError):
    """Base exception for invalid DrivAerML native-source metadata."""


class NativeSourceIntegrityError(NativeSourceError):
    """Raised when materialized bytes do not match their pinned identity."""


class VTKXMLIndexError(NativeSourceError):
    """Raised when a VTK XML stream cannot be indexed safely."""


class InlineBinaryDecodeError(NativeSourceError):
    """Raised when an inline-binary payload violates its VTK declaration."""


@dataclass(frozen=True)
class PinnedFile:
    """Identity of one whole file in the immutable dataset release."""

    path: PurePosixPath
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PinnedVolumePart(PinnedFile):
    """One position in an ordered multipart volume transport."""

    part_index: int


@dataclass(frozen=True)
class PinnedSurfaceArea(PinnedFile):
    """Pinned surface-area array and its native-boundary binding."""

    dtype: str
    element_count: int
    source_boundary_sha256: str


@dataclass(frozen=True)
class NativeCaseRecord:
    """Validated native-source identities for one DrivAerML case."""

    case_id: str
    run_number: int
    boundary: PinnedFile
    surface_cell_area: PinnedSurfaceArea
    volume_logical_path: PurePosixPath
    volume_parts: tuple[PinnedVolumePart, ...]
    volume_total_size_bytes: int


@dataclass(frozen=True)
class NativeSourcePin:
    """Validated native-source pin, indexed by stable case ID."""

    source_path: Path
    repository_id: str
    repository_revision: str
    cases: tuple[NativeCaseRecord, ...]
    _by_case_id: Mapping[str, NativeCaseRecord] = field(
        repr=False, compare=False
    )

    def case(self, case_id: str) -> NativeCaseRecord:
        """Return one case or raise a contract error with a useful message."""

        try:
            return self._by_case_id[case_id]
        except KeyError as exc:
            raise NativeSourceError(f"unknown native-source case_id {case_id!r}") from exc

    def resolve(self, case_id: str, dataset_root: str | Path) -> "ResolvedNativeCase":
        """Resolve a pinned case beneath a materialized dataset root."""

        record = self.case(case_id)
        root = Path(dataset_root).expanduser().resolve()
        return ResolvedNativeCase(
            record=record,
            dataset_root=root,
            boundary_path=_resolve_pinned_path(root, record.boundary.path),
            surface_cell_area_path=_resolve_pinned_path(
                root, record.surface_cell_area.path
            ),
            volume_part_paths=tuple(
                _resolve_pinned_path(root, part.path)
                for part in record.volume_parts
            ),
        )


@dataclass(frozen=True)
class ResolvedNativeCase:
    """Filesystem paths corresponding to a validated native case record."""

    record: NativeCaseRecord
    dataset_root: Path
    boundary_path: Path
    surface_cell_area_path: Path
    volume_part_paths: tuple[Path, ...]

    def multipart_segments(self) -> tuple["FileSegment", ...]:
        """Return ordered whole-file segments for the native part files."""

        return tuple(
            FileSegment(
                label=f"{self.record.case_id}:part:{part.part_index}",
                path=path,
                file_offset=0,
                size_bytes=part.size_bytes,
                sha256=part.sha256,
                require_whole_file=True,
            )
            for part, path in zip(
                self.record.volume_parts, self.volume_part_paths, strict=True
            )
        )

    def monolithic_segments(
        self, monolithic_path: str | Path | None = None
    ) -> tuple["FileSegment", ...]:
        """Return equivalent segment ranges over a reconstructed volume file."""

        path = (
            Path(monolithic_path).expanduser().resolve()
            if monolithic_path is not None
            else _resolve_pinned_path(
                self.dataset_root, self.record.volume_logical_path
            )
        )
        segments: list[FileSegment] = []
        offset = 0
        for part in self.record.volume_parts:
            segments.append(
                FileSegment(
                    label=f"{self.record.case_id}:monolithic-segment:{part.part_index}",
                    path=path,
                    file_offset=offset,
                    size_bytes=part.size_bytes,
                    sha256=part.sha256,
                    require_whole_file=False,
                )
            )
            offset += part.size_bytes
        return tuple(segments)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise NativeSourceError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise NativeSourceError(f"{context} must be a JSON array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeSourceError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeSourceError(f"{context} must be an integer >= {minimum}")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if _SHA256_RE.fullmatch(digest) is None:
        raise NativeSourceError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _relative_path(value: object, context: str) -> PurePosixPath:
    text = _string(value, context)
    if "\\" in text:
        raise NativeSourceError(f"{context} must use POSIX path separators")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise NativeSourceError(f"{context} must be a safe relative POSIX path")
    return path


def _pinned_file(value: object, context: str) -> PinnedFile:
    record = _mapping(value, context)
    return PinnedFile(
        path=_relative_path(record.get("path"), f"{context}.path"),
        size_bytes=_integer(
            record.get("size_bytes"), f"{context}.size_bytes", minimum=1
        ),
        sha256=_sha256(record.get("lfs_sha256"), f"{context}.lfs_sha256"),
    )


def _surface_area(value: object, context: str) -> PinnedSurfaceArea:
    record = _mapping(value, context)
    dtype = _string(record.get("dtype"), f"{context}.dtype")
    count = _integer(
        record.get("element_count"), f"{context}.element_count", minimum=1
    )
    size = _integer(record.get("size_bytes"), f"{context}.size_bytes", minimum=1)
    if dtype == "<f4" and size != 128 + 4 * count:
        raise NativeSourceError(
            f"{context}.size_bytes is inconsistent with its <f4 NPY element count"
        )
    return PinnedSurfaceArea(
        path=_relative_path(record.get("path"), f"{context}.path"),
        size_bytes=size,
        sha256=_sha256(record.get("lfs_sha256"), f"{context}.lfs_sha256"),
        dtype=dtype,
        element_count=count,
        source_boundary_sha256=_sha256(
            record.get("source_boundary_sha256"),
            f"{context}.source_boundary_sha256",
        ),
    )


def load_native_source_pin(path: str | Path) -> NativeSourcePin:
    """Load and strictly validate a DrivAerML native-source pin.

    Validation is metadata-only: this function does not touch the large files.
    Use :func:`open_verified_multipart` or :func:`open_verified_monolithic` to
    prove materialized bytes before parsing them.
    """

    source_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeSourceError(f"cannot load native-source pin {source_path}: {exc}") from exc

    root = _mapping(document, "native-source pin")
    if root.get("schema") != _PIN_SCHEMA or root.get("schema_version") != 1:
        raise NativeSourceError(
            f"native-source pin must declare schema {_PIN_SCHEMA!r}, version 1"
        )

    repository = _mapping(root.get("repository"), "repository")
    repository_id = _string(repository.get("repo_id"), "repository.repo_id")
    revision = _string(repository.get("revision"), "repository.revision")
    if _REVISION_RE.fullmatch(revision) is None:
        raise NativeSourceError(
            "repository.revision must be a lowercase 40-character Git commit"
        )

    scope = _mapping(root.get("case_scope"), "case_scope")
    declared_case_count = _integer(
        scope.get("case_count"), "case_scope.case_count", minimum=1
    )
    run_min = _integer(
        scope.get("run_number_min"), "case_scope.run_number_min", minimum=1
    )
    run_max = _integer(
        scope.get("run_number_max"), "case_scope.run_number_max", minimum=run_min
    )
    unavailable_values = _sequence(
        scope.get("unavailable_or_held_back_run_numbers"),
        "case_scope.unavailable_or_held_back_run_numbers",
    )
    unavailable = tuple(
        _integer(value, f"case_scope.unavailable_or_held_back_run_numbers[{i}]", minimum=run_min)
        for i, value in enumerate(unavailable_values)
    )
    if len(unavailable) != len(set(unavailable)) or tuple(sorted(unavailable)) != unavailable:
        raise NativeSourceError(
            "case_scope.unavailable_or_held_back_run_numbers must be unique and sorted"
        )
    if any(value > run_max for value in unavailable):
        raise NativeSourceError("held-back run number lies outside the declared run range")

    case_values = _sequence(root.get("cases"), "cases")
    if len(case_values) != declared_case_count:
        raise NativeSourceError(
            f"case_scope.case_count declares {declared_case_count}, "
            f"found {len(case_values)} records"
        )

    cases: list[NativeCaseRecord] = []
    seen_ids: set[str] = set()
    seen_runs: set[int] = set()
    for case_offset, value in enumerate(case_values):
        context = f"cases[{case_offset}]"
        case = _mapping(value, context)
        case_id = _string(case.get("case_id"), f"{context}.case_id")
        run_number = _integer(
            case.get("run_number"), f"{context}.run_number", minimum=run_min
        )
        if run_number > run_max:
            raise NativeSourceError(f"{context}.run_number lies outside case_scope")
        if case_id != f"run_{run_number}":
            raise NativeSourceError(
                f"{context}.case_id must equal run_<run_number>; found {case_id!r}"
            )
        if case_id in seen_ids or run_number in seen_runs:
            raise NativeSourceError(f"duplicate native-source case {case_id!r}")
        if cases and run_number <= cases[-1].run_number:
            raise NativeSourceError("native-source case records must be in run-number order")
        seen_ids.add(case_id)
        seen_runs.add(run_number)

        boundary = _pinned_file(case.get("boundary"), f"{context}.boundary")
        expected_boundary = PurePosixPath(
            f"run_{run_number}/boundary_{run_number}.vtp"
        )
        if boundary.path != expected_boundary:
            raise NativeSourceError(
                f"{context}.boundary.path must be {expected_boundary.as_posix()!r}"
            )

        area = _surface_area(
            case.get("surface_cell_area"), f"{context}.surface_cell_area"
        )
        expected_area = PurePosixPath(
            f"run_{run_number}/boundary_cell_area_{run_number}.npy"
        )
        if area.path != expected_area:
            raise NativeSourceError(
                f"{context}.surface_cell_area.path must be {expected_area.as_posix()!r}"
            )
        if area.source_boundary_sha256 != boundary.sha256:
            raise NativeSourceError(
                f"{context}.surface_cell_area is not bound to the pinned boundary hash"
            )

        volume = _mapping(case.get("volume"), f"{context}.volume")
        logical_path = _relative_path(
            volume.get("logical_path_after_assembly"),
            f"{context}.volume.logical_path_after_assembly",
        )
        expected_logical = PurePosixPath(f"run_{run_number}/volume_{run_number}.vtu")
        if logical_path != expected_logical:
            raise NativeSourceError(
                f"{context}.volume logical path must be {expected_logical.as_posix()!r}"
            )
        part_count = _integer(
            volume.get("part_count"), f"{context}.volume.part_count", minimum=2
        )
        if part_count not in {2, 3}:
            raise NativeSourceError(
                f"{context}.volume.part_count must be two or three"
            )
        part_values = _sequence(volume.get("parts"), f"{context}.volume.parts")
        if len(part_values) != part_count:
            raise NativeSourceError(
                f"{context}.volume.part_count does not match its parts array"
            )
        parts: list[PinnedVolumePart] = []
        for expected_index, part_value in enumerate(part_values):
            part_context = f"{context}.volume.parts[{expected_index}]"
            part_record = _mapping(part_value, part_context)
            part_index = _integer(
                part_record.get("part_index"),
                f"{part_context}.part_index",
                minimum=0,
            )
            if part_index != expected_index:
                raise NativeSourceError(
                    f"{part_context}.part_index must be {expected_index} "
                    "to prove concatenation order"
                )
            part_path = _relative_path(
                part_record.get("path"), f"{part_context}.path"
            )
            expected_part_path = PurePosixPath(
                f"{logical_path.as_posix()}.{part_index:02d}.part"
            )
            if part_path != expected_part_path:
                raise NativeSourceError(
                    f"{part_context}.path must be {expected_part_path.as_posix()!r}"
                )
            parts.append(
                PinnedVolumePart(
                    path=part_path,
                    size_bytes=_integer(
                        part_record.get("size_bytes"),
                        f"{part_context}.size_bytes",
                        minimum=1,
                    ),
                    sha256=_sha256(
                        part_record.get("lfs_sha256"),
                        f"{part_context}.lfs_sha256",
                    ),
                    part_index=part_index,
                )
            )
        total_size = _integer(
            volume.get("total_size_bytes"),
            f"{context}.volume.total_size_bytes",
            minimum=1,
        )
        actual_total = sum(part.size_bytes for part in parts)
        if total_size != actual_total:
            raise NativeSourceError(
                f"{context}.volume.total_size_bytes is {total_size}, expected {actual_total}"
            )

        cases.append(
            NativeCaseRecord(
                case_id=case_id,
                run_number=run_number,
                boundary=boundary,
                surface_cell_area=area,
                volume_logical_path=logical_path,
                volume_parts=tuple(parts),
                volume_total_size_bytes=total_size,
            )
        )

    complete_range = set(range(run_min, run_max + 1))
    if seen_runs | set(unavailable) != complete_range or seen_runs & set(unavailable):
        raise NativeSourceError(
            "available and held-back run numbers must exactly partition case_scope"
        )

    totals = _mapping(root.get("totals"), "totals")
    expected_totals = {
        "boundary_file_count": len(cases),
        "boundary_bytes": sum(case.boundary.size_bytes for case in cases),
        "surface_cell_area_file_count": len(cases),
        "surface_cell_area_bytes": sum(
            case.surface_cell_area.size_bytes for case in cases
        ),
        "logical_volume_count": len(cases),
        "volume_part_file_count": sum(len(case.volume_parts) for case in cases),
        "reconstructed_volume_bytes": sum(
            case.volume_total_size_bytes for case in cases
        ),
    }
    for key, expected in expected_totals.items():
        declared = _integer(totals.get(key), f"totals.{key}", minimum=1)
        if declared != expected:
            raise NativeSourceError(
                f"totals.{key} declares {declared}, expected {expected}"
            )

    case_tuple = tuple(cases)
    return NativeSourcePin(
        source_path=source_path,
        repository_id=repository_id,
        repository_revision=revision,
        cases=case_tuple,
        _by_case_id=MappingProxyType({case.case_id: case for case in case_tuple}),
    )


def _resolve_pinned_path(root: Path, relative_path: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative_path.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NativeSourceError(
            f"pinned path {relative_path.as_posix()!r} escapes dataset root {root}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class FileSegment:
    """One byte range contributing to a logical concatenated stream."""

    label: str
    path: Path
    file_offset: int
    size_bytes: int
    sha256: str
    require_whole_file: bool = False

    def __post_init__(self) -> None:
        if self.file_offset < 0 or self.size_bytes <= 0:
            raise NativeSourceError(f"invalid byte range for segment {self.label!r}")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise NativeSourceError(
                f"segment {self.label!r} has an invalid SHA-256 identity"
            )


@dataclass(frozen=True)
class SegmentVerification:
    """Evidence produced while verifying one file segment."""

    label: str
    path: Path
    file_offset: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _OpenFileSnapshot:
    """Identity and mutation-sensitive metadata for one retained file."""

    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_handle(cls, handle: BinaryIO) -> "_OpenFileSnapshot":
        try:
            value = os.fstat(handle.fileno())
        except OSError as exc:
            raise NativeSourceIntegrityError(
                f"cannot fstat retained native-source descriptor: {exc}"
            ) from exc
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )


@dataclass
class _RetainedOpenFile:
    """An open source object and the metadata frozen before verification."""

    path: Path
    handle: BinaryIO
    snapshot: _OpenFileSnapshot

    def assert_unchanged(self, *, context: str) -> None:
        current = _OpenFileSnapshot.from_handle(self.handle)
        if current != self.snapshot:
            raise NativeSourceIntegrityError(
                f"retained native-source file {self.path} changed {context}; "
                "device, inode, mode, size, mtime, or ctime differs from the "
                "pre-verification fstat snapshot"
            )


def _open_retained_files(
    segments: Sequence[FileSegment],
) -> tuple[tuple[_RetainedOpenFile, ...], tuple[int, ...]]:
    """Open every unique caller path once and retain those exact objects."""

    retained: list[_RetainedOpenFile] = []
    by_path: dict[Path, int] = {}
    segment_file_indices: list[int] = []
    try:
        for segment in segments:
            file_index = by_path.get(segment.path)
            if file_index is None:
                try:
                    handle = segment.path.open("rb", buffering=0)
                except OSError as exc:
                    raise NativeSourceIntegrityError(
                        f"cannot open segment file {segment.path}: {exc}"
                    ) from exc
                snapshot = _OpenFileSnapshot.from_handle(handle)
                if not stat.S_ISREG(snapshot.mode):
                    handle.close()
                    raise NativeSourceIntegrityError(
                        f"native-source segment is not a regular file: {segment.path}"
                    )
                file_index = len(retained)
                retained.append(
                    _RetainedOpenFile(
                        path=segment.path,
                        handle=handle,
                        snapshot=snapshot,
                    )
                )
                by_path[segment.path] = file_index
            segment_file_indices.append(file_index)
    except Exception:
        for source in retained:
            source.handle.close()
        raise
    return tuple(retained), tuple(segment_file_indices)


def _verify_retained_segments(
    segments: Sequence[FileSegment],
    retained_files: Sequence[_RetainedOpenFile],
    segment_file_indices: Sequence[int],
    *,
    chunk_size: int,
    exact_single_file_size: int | None = None,
) -> tuple[SegmentVerification, ...]:
    """Hash ranges from retained descriptors and reject concurrent mutation."""

    if exact_single_file_size is not None:
        if len(retained_files) != 1:
            raise NativeSourceIntegrityError(
                "exact physical-size verification requires one retained file"
            )
        physical_size = retained_files[0].snapshot.size_bytes
        if physical_size != exact_single_file_size:
            raise NativeSourceIntegrityError(
                f"reconstructed volume {retained_files[0].path} has "
                f"{physical_size} bytes; expected {exact_single_file_size}"
            )

    results: list[SegmentVerification] = []
    for segment, file_index in zip(
        segments, segment_file_indices, strict=True
    ):
        source = retained_files[file_index]
        physical_size = source.snapshot.size_bytes
        required_end = segment.file_offset + segment.size_bytes
        if segment.require_whole_file and (
            segment.file_offset != 0 or physical_size != segment.size_bytes
        ):
            raise NativeSourceIntegrityError(
                f"segment {segment.label!r} expects a whole file of "
                f"{segment.size_bytes} bytes; found {physical_size}"
            )
        if not segment.require_whole_file and physical_size < required_end:
            raise NativeSourceIntegrityError(
                f"segment {segment.label!r} ends at byte {required_end}, "
                f"but {segment.path} has only {physical_size} bytes"
            )

        source.assert_unchanged(context="before hashing")
        digest = hashlib.sha256()
        remaining = segment.size_bytes
        try:
            source.handle.seek(segment.file_offset)
            while remaining:
                block = source.handle.read(min(chunk_size, remaining))
                if not block:
                    raise NativeSourceIntegrityError(
                        f"unexpected EOF while hashing segment {segment.label!r}"
                    )
                digest.update(block)
                remaining -= len(block)
        except NativeSourceIntegrityError:
            raise
        except OSError as exc:
            raise NativeSourceIntegrityError(
                f"cannot read segment {segment.label!r}: {exc}"
            ) from exc
        source.assert_unchanged(context="while hashing")
        actual = digest.hexdigest()
        if actual != segment.sha256:
            raise NativeSourceIntegrityError(
                f"segment {segment.label!r} SHA-256 mismatch: "
                f"expected {segment.sha256}, found {actual}"
            )
        results.append(
            SegmentVerification(
                label=segment.label,
                path=segment.path,
                file_offset=segment.file_offset,
                size_bytes=segment.size_bytes,
                sha256=actual,
            )
        )
    for source in retained_files:
        source.assert_unchanged(context="during ordered-segment verification")
    return tuple(results)


def verify_file_segments(
    segments: Sequence[FileSegment], *, chunk_size: int = DEFAULT_IO_CHUNK_BYTES
) -> tuple[SegmentVerification, ...]:
    """Verify exact range size and SHA-256 for each ordered segment.

    Memory use is bounded by ``chunk_size``.  Whole-file part segments also
    require the physical file size to equal the pinned part size.
    """

    if not segments:
        raise NativeSourceError("at least one file segment is required")
    if chunk_size <= 0:
        raise NativeSourceError("chunk_size must be positive")

    retained_files, segment_file_indices = _open_retained_files(segments)
    try:
        return _verify_retained_segments(
            segments,
            retained_files,
            segment_file_indices,
            chunk_size=chunk_size,
        )
    finally:
        for source in retained_files:
            source.handle.close()


class SegmentedReader(io.RawIOBase):
    """Seekable read-only view over ordered file ranges.

    Every physical file is opened once when the reader is constructed and that
    exact descriptor is retained until close.  ``readinto`` crosses part
    boundaries directly into the caller's buffer and checks mutation-sensitive
    ``fstat`` metadata before and after each read.  As with a normal file,
    callers should pass a finite size to ``read`` when operating on large
    native volumes.
    """

    def __init__(
        self,
        segments: Sequence[FileSegment],
        *,
        verification: Sequence[SegmentVerification] = (),
        _retained_files: Sequence[_RetainedOpenFile] | None = None,
        _segment_file_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not segments:
            raise NativeSourceError("SegmentedReader requires at least one segment")
        self._segments = tuple(segments)
        if (_retained_files is None) != (_segment_file_indices is None):
            raise NativeSourceError(
                "retained files and segment-file indices must be supplied together"
            )
        if _retained_files is None:
            retained_files, segment_file_indices = _open_retained_files(
                self._segments
            )
        else:
            retained_files = tuple(_retained_files)
            segment_file_indices = tuple(_segment_file_indices or ())
        if len(segment_file_indices) != len(self._segments):
            for source in retained_files:
                source.handle.close()
            raise NativeSourceError(
                "one retained-file index is required for every segment"
            )
        if any(
            file_index < 0 or file_index >= len(retained_files)
            for file_index in segment_file_indices
        ):
            for source in retained_files:
                source.handle.close()
            raise NativeSourceError("invalid retained-file index")
        self._retained_files = tuple(retained_files)
        self._segment_file_indices = tuple(segment_file_indices)
        self._starts: tuple[int, ...]
        starts: list[int] = []
        position = 0
        for segment in self._segments:
            starts.append(position)
            position += segment.size_bytes
        self._starts = tuple(starts)
        self._length = position
        self._position = 0
        self.verification = tuple(verification)

    @property
    def segments(self) -> tuple[FileSegment, ...]:
        return self._segments

    @property
    def logical_size_bytes(self) -> int:
        return self._length

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._checkClosed()
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._position + offset
        elif whence == os.SEEK_END:
            target = self._length + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self._position = target
        return target

    def assert_unchanged(self, *, context: str = "after verification") -> None:
        """Reject identity, size, mtime, or ctime changes on retained sources."""

        self._checkClosed()
        for source in self._retained_files:
            source.assert_unchanged(context=context)

    def single_file_descriptor_path(self) -> Path:
        """Return a safe path to the one retained inode for in-process VTK.

        This is intentionally limited to a single physical source (the
        monolithic reconstruction case).  Linux ``/proc/self/fd`` is preferred;
        ``/dev/fd`` is accepted only after its target's device and inode match
        the retained descriptor.  The caller must keep this reader open and
        call :meth:`assert_unchanged` after the external reader finishes.
        """

        self.assert_unchanged(context="before exposing its descriptor path")
        if len(self._retained_files) != 1:
            raise NativeSourceIntegrityError(
                "a single descriptor path requires exactly one retained file"
            )
        source = self._retained_files[0]
        descriptor = source.handle.fileno()
        for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = directory / str(descriptor)
            try:
                target = os.stat(candidate)
            except OSError:
                continue
            if (
                int(target.st_dev) == source.snapshot.device
                and int(target.st_ino) == source.snapshot.inode
            ):
                return candidate
        raise NativeSourceIntegrityError(
            "no safe descriptor filesystem exposes the retained native-source "
            "file; expected /proc/self/fd or /dev/fd"
        )

    def readinto(self, buffer: object) -> int:
        self._checkClosed()
        view = memoryview(buffer).cast("B")
        if not view or self._position >= self._length:
            return 0
        self.assert_unchanged(context="before a logical-stream read")
        requested = min(len(view), self._length - self._position)
        written = 0
        while written < requested:
            segment_index = bisect.bisect_right(self._starts, self._position) - 1
            if segment_index < 0:
                segment_index = 0
            segment = self._segments[segment_index]
            segment_logical_start = self._starts[segment_index]
            within_segment = self._position - segment_logical_start
            available = segment.size_bytes - within_segment
            take = min(requested - written, available)
            handle = self._retained_files[
                self._segment_file_indices[segment_index]
            ].handle
            handle.seek(segment.file_offset + within_segment)
            target = view[written : written + take]
            consumed = 0
            while consumed < take:
                count = handle.readinto(target[consumed:])
                if not count:
                    raise NativeSourceIntegrityError(
                        f"unexpected EOF while reading segment {segment.label!r}"
                    )
                consumed += count
            written += take
            self._position += take
        self.assert_unchanged(context="during a logical-stream read")
        return written

    def close(self) -> None:
        for source in self._retained_files:
            source.handle.close()
        self._retained_files = ()
        self._segment_file_indices = ()
        super().close()


def _open_verified_segment_reader(
    segments: Sequence[FileSegment],
    *,
    verification_chunk_size: int,
    exact_single_file_size: int | None = None,
) -> SegmentedReader:
    """Open once, verify through retained descriptors, and return them."""

    if not segments:
        raise NativeSourceError("at least one file segment is required")
    if verification_chunk_size <= 0:
        raise NativeSourceError("chunk_size must be positive")
    retained_files, segment_file_indices = _open_retained_files(segments)
    try:
        verification = _verify_retained_segments(
            segments,
            retained_files,
            segment_file_indices,
            chunk_size=verification_chunk_size,
            exact_single_file_size=exact_single_file_size,
        )
        return SegmentedReader(
            segments,
            verification=verification,
            _retained_files=retained_files,
            _segment_file_indices=segment_file_indices,
        )
    except Exception:
        for source in retained_files:
            source.handle.close()
        raise


def open_verified_multipart(
    resolved_case: ResolvedNativeCase,
    *,
    verification_chunk_size: int = DEFAULT_IO_CHUNK_BYTES,
) -> SegmentedReader:
    """Verify and open the case's ordered two- or three-part volume."""

    segments = resolved_case.multipart_segments()
    return _open_verified_segment_reader(
        segments,
        verification_chunk_size=verification_chunk_size,
    )


def open_verified_monolithic(
    resolved_case: ResolvedNativeCase,
    monolithic_path: str | Path | None = None,
    *,
    verification_chunk_size: int = DEFAULT_IO_CHUNK_BYTES,
) -> SegmentedReader:
    """Verify every pinned part range in a monolithic reconstruction.

    Equality of every range's size and SHA-256 proves that this view is byte
    equivalent to concatenating the native parts in their pinned order.
    """

    segments = resolved_case.monolithic_segments(monolithic_path)
    return _open_verified_segment_reader(
        segments,
        verification_chunk_size=verification_chunk_size,
        exact_single_file_size=resolved_case.record.volume_total_size_bytes,
    )


@dataclass(frozen=True)
class VTKPieceIndex:
    """Declared native tuple counts for one VTK XML Piece."""

    piece_index: int
    number_of_points: int
    number_of_cells: int
    number_of_verts: int = 0
    number_of_lines: int = 0
    number_of_strips: int = 0
    number_of_polys: int = 0


@dataclass(frozen=True)
class VTKDataArrayIndex:
    """Byte offsets and metadata for one inline-binary DataArray."""

    array_index: int
    piece_index: int
    association: str
    name: str | None
    vtk_type: str
    number_of_components: int
    format: str
    opening_tag_start: int
    encoded_start: int
    encoded_end: int
    closing_tag_end: int


@dataclass(frozen=True)
class VTKXMLIndex:
    """Bounded-memory index of inline-binary arrays in one VTK XML file."""

    dataset_type: str
    version: str | None
    byte_order: str
    header_type: str
    compressor: str | None
    pieces: tuple[VTKPieceIndex, ...]
    data_arrays: tuple[VTKDataArrayIndex, ...]
    source_size_bytes: int

    def arrays_for(
        self,
        *,
        association: str | None = None,
        name: str | None = None,
        piece_index: int | None = None,
    ) -> tuple[VTKDataArrayIndex, ...]:
        """Select indexed arrays without changing their source order."""

        return tuple(
            array
            for array in self.data_arrays
            if (association is None or array.association == association)
            and (name is None or array.name == name)
            and (piece_index is None or array.piece_index == piece_index)
        )


@dataclass
class _OpenElement:
    name: str
    piece_index: int | None
    binary_array: dict[str, object] | None = None


def _local_name(name: str) -> str:
    return name.rsplit(":", 1)[-1]


def _markup_tags(
    stream: BinaryIO,
    *,
    chunk_size: int,
    max_tag_bytes: int,
) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(start, end, markup)`` without retaining non-markup text."""

    if chunk_size <= 0 or max_tag_bytes <= 0:
        raise VTKXMLIndexError("XML scan buffer limits must be positive")
    absolute = 0
    pending: bytearray | None = None
    pending_start = 0
    mode = "unknown"
    quote: int | None = None

    while True:
        block = stream.read(chunk_size)
        if not block:
            break
        cursor = 0
        while cursor < len(block):
            if pending is None:
                marker = block.find(b"<", cursor)
                if marker < 0:
                    break
                pending = bytearray()
                pending_start = absolute + marker
                mode = "unknown"
                quote = None
                cursor = marker

            byte = block[cursor]
            pending.append(byte)
            cursor += 1
            if len(pending) > max_tag_bytes:
                raise VTKXMLIndexError(
                    f"XML markup at byte {pending_start} exceeds max_tag_bytes"
                )

            complete = False
            if mode == "unknown":
                raw = bytes(pending)
                special_prefixes = (b"<!--", b"<![CDATA[", b"<?")
                if raw.startswith(b"<!--"):
                    mode = "comment"
                elif raw.startswith(b"<![CDATA["):
                    mode = "cdata"
                elif raw.startswith(b"<?"):
                    mode = "processing"
                elif any(prefix.startswith(raw) for prefix in special_prefixes):
                    continue
                elif len(raw) >= 2:
                    mode = "normal"

            if mode == "comment":
                complete = pending.endswith(b"-->")
            elif mode == "cdata":
                complete = pending.endswith(b"]]>")
            elif mode == "processing":
                complete = pending.endswith(b"?>")
            elif mode == "normal":
                if quote is not None:
                    if byte == quote:
                        quote = None
                elif byte in (ord('"'), ord("'")):
                    quote = byte
                elif byte == ord(">"):
                    complete = True

            if complete:
                end = absolute + cursor
                yield pending_start, end, bytes(pending)
                pending = None
        absolute += len(block)

    if pending is not None:
        raise VTKXMLIndexError(
            f"unterminated XML markup beginning at byte {pending_start}"
        )


def _parse_markup(raw: bytes) -> tuple[str, bool, bool, dict[str, str]] | None:
    if raw.startswith((b"<?", b"<!--", b"<![CDATA[", b"<!DOCTYPE")):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VTKXMLIndexError("non-UTF-8 bytes in XML markup") from exc
    if not text.startswith("<") or not text.endswith(">"):
        raise VTKXMLIndexError("invalid XML markup boundary")
    body = text[1:-1].strip()
    closing = body.startswith("/")
    if closing:
        body = body[1:].lstrip()
    self_closing = not closing and body.endswith("/")
    if self_closing:
        body = body[:-1].rstrip()
    match = _TAG_NAME_RE.match(body)
    if match is None:
        raise VTKXMLIndexError(f"cannot parse XML element name from {text[:120]!r}")
    name = _local_name(match.group(0))
    remainder = body[match.end() :]
    attributes: dict[str, str] = {}
    consumed = [False] * len(remainder)
    for attribute in _ATTRIBUTE_RE.finditer(remainder):
        key = _local_name(attribute.group(1))
        if key in attributes:
            raise VTKXMLIndexError(f"duplicate XML attribute {key!r}")
        value = attribute.group(2) if attribute.group(2) is not None else attribute.group(3)
        attributes[key] = html.unescape(value)
        for index in range(attribute.start(), attribute.end()):
            consumed[index] = True
    unexplained = "".join(
        character
        for index, character in enumerate(remainder)
        if not consumed[index] and not character.isspace()
    )
    if unexplained:
        raise VTKXMLIndexError(
            f"unsupported text in <{name}> markup: {unexplained[:80]!r}"
        )
    if closing and (attributes or self_closing):
        raise VTKXMLIndexError(f"invalid closing tag </{name}>")
    return name, closing, self_closing, attributes


def _xml_count(attributes: Mapping[str, str], key: str, context: str) -> int:
    raw = attributes.get(key)
    if raw is None or not raw.isdigit():
        raise VTKXMLIndexError(f"{context} must declare non-negative {key}")
    return int(raw)


def _xml_optional_count(attributes: Mapping[str, str], key: str) -> int:
    raw = attributes.get(key)
    if raw is None:
        return 0
    if not raw.isdigit():
        raise VTKXMLIndexError(f"Piece {key} must be a non-negative integer")
    return int(raw)


def _piece_index(
    attributes: Mapping[str, str], piece_index: int
) -> VTKPieceIndex:
    number_of_points = _xml_count(attributes, "NumberOfPoints", "Piece")
    if "NumberOfCells" in attributes:
        return VTKPieceIndex(
            piece_index=piece_index,
            number_of_points=number_of_points,
            number_of_cells=_xml_count(attributes, "NumberOfCells", "Piece"),
        )

    polydata_keys = (
        "NumberOfVerts",
        "NumberOfLines",
        "NumberOfStrips",
        "NumberOfPolys",
    )
    if not any(key in attributes for key in polydata_keys):
        raise VTKXMLIndexError(
            "Piece must declare NumberOfCells or PolyData cell counts"
        )
    verts, lines, strips, polys = (
        _xml_optional_count(attributes, key) for key in polydata_keys
    )
    return VTKPieceIndex(
        piece_index=piece_index,
        number_of_points=number_of_points,
        number_of_cells=verts + lines + strips + polys,
        number_of_verts=verts,
        number_of_lines=lines,
        number_of_strips=strips,
        number_of_polys=polys,
    )


def index_inline_binary_vtk_xml(
    stream: BinaryIO,
    *,
    scan_chunk_size: int = DEFAULT_IO_CHUNK_BYTES,
    max_tag_bytes: int = DEFAULT_MAX_XML_TAG_BYTES,
    require_uint64_header: bool = True,
) -> VTKXMLIndex:
    """Index inline-binary VTK XML arrays without reading by line.

    Only XML markup is retained.  Base64 bodies are skipped with byte searches,
    so memory use is independent of array size and tags may cross both scan and
    multipart boundaries.  The stream position is restored before returning.
    """

    if not stream.seekable() or not stream.readable():
        raise VTKXMLIndexError("VTK XML indexing requires a seekable readable stream")
    original_position = stream.tell()
    try:
        stream.seek(0, os.SEEK_END)
        source_size = stream.tell()
        stream.seek(0)
        stack: list[_OpenElement] = []
        pieces: list[VTKPieceIndex] = []
        arrays: list[VTKDataArrayIndex] = []
        vtk_attributes: dict[str, str] | None = None

        for tag_start, tag_end, raw in _markup_tags(
            stream, chunk_size=scan_chunk_size, max_tag_bytes=max_tag_bytes
        ):
            parsed = _parse_markup(raw)
            if parsed is None:
                continue
            name, closing, self_closing, attributes = parsed
            if closing:
                if not stack or stack[-1].name != name:
                    expected = stack[-1].name if stack else None
                    raise VTKXMLIndexError(
                        f"unexpected closing tag </{name}> at byte {tag_start}; "
                        f"expected {expected!r}"
                    )
                element = stack.pop()
                if name == "DataArray" and element.binary_array is not None:
                    record = element.binary_array
                    arrays.append(
                        VTKDataArrayIndex(
                            array_index=len(arrays),
                            piece_index=int(record["piece_index"]),
                            association=str(record["association"]),
                            name=(
                                None
                                if record["name"] is None
                                else str(record["name"])
                            ),
                            vtk_type=str(record["vtk_type"]),
                            number_of_components=int(record["number_of_components"]),
                            format="binary",
                            opening_tag_start=int(record["opening_tag_start"]),
                            encoded_start=int(record["encoded_start"]),
                            encoded_end=tag_start,
                            closing_tag_end=tag_end,
                        )
                    )
                continue

            parent_piece = next(
                (
                    element.piece_index
                    for element in reversed(stack)
                    if element.piece_index is not None
                ),
                None,
            )
            binary_record: dict[str, object] | None = None
            element_piece = parent_piece

            if name == "VTKFile":
                if vtk_attributes is not None:
                    raise VTKXMLIndexError("VTK XML stream contains multiple VTKFile roots")
                vtk_attributes = attributes
            elif name == "Piece":
                element_piece = len(pieces)
                pieces.append(_piece_index(attributes, element_piece))
            elif name == "DataArray":
                data_format = attributes.get("format", "ascii").lower()
                if data_format == "binary":
                    if self_closing:
                        raise VTKXMLIndexError(
                            "inline-binary DataArray cannot be self-closing"
                        )
                    association = next(
                        (
                            element.name
                            for element in reversed(stack)
                            if element.name
                            in {
                                "Points",
                                "Cells",
                                "Verts",
                                "Lines",
                                "Strips",
                                "Polys",
                                "PointData",
                                "CellData",
                                "FieldData",
                            }
                        ),
                        "Unassociated",
                    )
                    if element_piece is None and association != "FieldData":
                        raise VTKXMLIndexError(
                            "inline-binary DataArray occurs outside a Piece or FieldData"
                        )
                    vtk_type = attributes.get("type")
                    if not vtk_type:
                        raise VTKXMLIndexError("binary DataArray has no VTK type")
                    components_text = attributes.get("NumberOfComponents", "1")
                    if not components_text.isdigit() or int(components_text) <= 0:
                        raise VTKXMLIndexError(
                            "DataArray NumberOfComponents must be a positive integer"
                        )
                    binary_record = {
                        # Dataset-level FieldData is not associated with a Piece.
                        # Use -1 as an explicit sentinel; downstream native field
                        # selection is by CellData and can never confuse the two.
                        "piece_index": -1 if element_piece is None else element_piece,
                        "association": association,
                        "name": attributes.get("Name"),
                        "vtk_type": vtk_type,
                        "number_of_components": int(components_text),
                        "opening_tag_start": tag_start,
                        "encoded_start": tag_end,
                    }

            if not self_closing:
                stack.append(
                    _OpenElement(
                        name=name,
                        piece_index=element_piece,
                        binary_array=binary_record,
                    )
                )

        if stack:
            raise VTKXMLIndexError(f"unclosed XML element <{stack[-1].name}>")
        if vtk_attributes is None:
            raise VTKXMLIndexError("VTK XML stream has no VTKFile root")
        dataset_type = vtk_attributes.get("type")
        byte_order = vtk_attributes.get("byte_order")
        header_type = vtk_attributes.get("header_type", "UInt32")
        if not dataset_type:
            raise VTKXMLIndexError("VTKFile has no dataset type")
        if byte_order not in {"LittleEndian", "BigEndian"}:
            raise VTKXMLIndexError(
                "VTKFile byte_order must be LittleEndian or BigEndian"
            )
        if require_uint64_header and header_type != "UInt64":
            raise VTKXMLIndexError(
                f"native VTK transport requires header_type UInt64, found {header_type!r}"
            )
        if not pieces:
            raise VTKXMLIndexError("VTK XML stream contains no Piece elements")
        return VTKXMLIndex(
            dataset_type=dataset_type,
            version=vtk_attributes.get("version"),
            byte_order=byte_order,
            header_type=header_type,
            compressor=vtk_attributes.get("compressor"),
            pieces=tuple(pieces),
            data_arrays=tuple(arrays),
            source_size_bytes=source_size,
        )
    finally:
        stream.seek(original_position)


@dataclass(frozen=True)
class InlineBinaryPayloadSummary:
    """Exact byte and tuple evidence for one decoded DataArray payload."""

    array_index: int
    declared_payload_bytes: int
    decoded_payload_bytes: int
    payload_sha256: str
    scalar_count: int
    tuple_count: int
    number_of_components: int
    vtk_type: str


def _decoded_base64_blocks(
    stream: BinaryIO,
    *,
    start: int,
    end: int,
    encoded_chunk_size: int,
) -> Iterator[bytes]:
    if encoded_chunk_size <= 0:
        raise InlineBinaryDecodeError("encoded_chunk_size must be positive")
    if start < 0 or end < start:
        raise InlineBinaryDecodeError("invalid encoded payload byte range")
    stream.seek(start)
    remaining = end - start
    pending = bytearray()
    while remaining:
        raw = stream.read(min(encoded_chunk_size, remaining))
        if not raw:
            raise InlineBinaryDecodeError("unexpected EOF in inline-binary base64 text")
        remaining -= len(raw)
        pending.extend(raw.translate(None, _XML_WHITESPACE))
        complete_size = len(pending) - (len(pending) % 4)
        if not complete_size:
            continue
        complete = bytes(pending[:complete_size])
        del pending[:complete_size]
        cursor = 0
        while cursor < len(complete):
            padding = complete.find(b"=", cursor)
            if padding < 0:
                member_end = len(complete)
            else:
                member_end = ((padding // 4) + 1) * 4
            member = complete[cursor:member_end]
            try:
                decoded = base64.b64decode(member, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InlineBinaryDecodeError(
                    "invalid base64 in inline-binary DataArray"
                ) from exc
            if decoded:
                yield decoded
            cursor = member_end
    if pending:
        raise InlineBinaryDecodeError(
            "inline-binary base64 text does not contain complete quartets"
        )


def _expected_tuple_count(
    vtk_index: VTKXMLIndex, array: VTKDataArrayIndex
) -> int | None:
    if array.piece_index < 0 and array.association == "FieldData":
        return None
    try:
        piece = vtk_index.pieces[array.piece_index]
    except IndexError as exc:
        raise InlineBinaryDecodeError(
            f"DataArray references missing Piece {array.piece_index}"
        ) from exc
    if piece.piece_index != array.piece_index:
        raise InlineBinaryDecodeError("VTK Piece index is not contiguous")
    if array.association in {"Points", "PointData"}:
        return piece.number_of_points
    if array.association == "CellData":
        return piece.number_of_cells
    if array.association == "Cells" and array.name in {
        "offsets",
        "types",
        "faceoffsets",
    }:
        return piece.number_of_cells
    polydata_counts = {
        "Verts": piece.number_of_verts,
        "Lines": piece.number_of_lines,
        "Strips": piece.number_of_strips,
        "Polys": piece.number_of_polys,
    }
    if array.association in polydata_counts and array.name == "offsets":
        return polydata_counts[array.association]
    return None


def stream_inline_binary_payload(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    sink: BinaryIO | Callable[[bytes], object] | None = None,
    *,
    encoded_chunk_size: int = DEFAULT_IO_CHUNK_BYTES,
) -> InlineBinaryPayloadSummary:
    """Decode one UInt64-header inline payload with bounded memory.

    Decoded payload chunks are passed to ``sink`` (a ``write`` object or
    callable).  A ``None`` sink validates and hashes without retaining data.
    The function accepts both a jointly encoded ``header + payload`` and the
    independently padded base64 members emitted by some VTK writers.
    """

    if vtk_index.compressor:
        raise InlineBinaryDecodeError(
            f"compressed inline payloads are unsupported: {vtk_index.compressor!r}"
        )
    if vtk_index.header_type != "UInt64":
        raise InlineBinaryDecodeError(
            f"expected UInt64 VTK binary header, found {vtk_index.header_type!r}"
        )
    if vtk_index.byte_order == "LittleEndian":
        header_struct = struct.Struct("<Q")
    elif vtk_index.byte_order == "BigEndian":
        header_struct = struct.Struct(">Q")
    else:
        raise InlineBinaryDecodeError(
            f"unsupported VTK byte order {vtk_index.byte_order!r}"
        )
    width = _VTK_SCALAR_WIDTHS.get(array.vtk_type)
    if width is None:
        raise InlineBinaryDecodeError(
            f"unsupported VTK scalar type {array.vtk_type!r}"
        )

    if sink is None:
        emit: Callable[[bytes], object] | None = None
    elif callable(sink) and not hasattr(sink, "write"):
        emit = sink
    else:
        writer = getattr(sink, "write", None)
        if writer is None or not callable(writer):
            raise TypeError("sink must be None, callable, or a binary write object")
        emit = writer

    header = bytearray()
    declared_bytes: int | None = None
    decoded_bytes = 0
    digest = hashlib.sha256()
    for decoded in _decoded_base64_blocks(
        stream,
        start=array.encoded_start,
        end=array.encoded_end,
        encoded_chunk_size=encoded_chunk_size,
    ):
        cursor = 0
        if declared_bytes is None:
            needed = header_struct.size - len(header)
            take = min(needed, len(decoded))
            header.extend(decoded[:take])
            cursor = take
            if len(header) == header_struct.size:
                declared_bytes = header_struct.unpack(header)[0]
        if cursor == len(decoded):
            continue
        if declared_bytes is None:
            raise InlineBinaryDecodeError("internal error while decoding VTK header")
        payload = decoded[cursor:]
        if decoded_bytes + len(payload) > declared_bytes:
            raise InlineBinaryDecodeError(
                f"DataArray {array.array_index} contains more than its declared "
                f"{declared_bytes} payload bytes"
            )
        decoded_bytes += len(payload)
        digest.update(payload)
        if emit is not None:
            result = emit(payload)
            if isinstance(result, int) and result != len(payload):
                raise InlineBinaryDecodeError(
                    f"sink accepted {result} of {len(payload)} decoded bytes"
                )

    if declared_bytes is None:
        raise InlineBinaryDecodeError(
            f"DataArray {array.array_index} does not contain a complete UInt64 header"
        )
    if decoded_bytes != declared_bytes:
        raise InlineBinaryDecodeError(
            f"DataArray {array.array_index} declares {declared_bytes} payload bytes, "
            f"decoded {decoded_bytes}"
        )
    tuple_width = width * array.number_of_components
    if declared_bytes % tuple_width:
        raise InlineBinaryDecodeError(
            f"DataArray {array.array_index} payload size {declared_bytes} is not "
            f"divisible by its {tuple_width}-byte tuple width"
        )
    scalar_count = declared_bytes // width
    tuple_count = declared_bytes // tuple_width
    expected_tuples = _expected_tuple_count(vtk_index, array)
    if expected_tuples is not None and tuple_count != expected_tuples:
        raise InlineBinaryDecodeError(
            f"DataArray {array.array_index} has {tuple_count} tuples for "
            f"{array.association}; Piece {array.piece_index} declares {expected_tuples}"
        )
    return InlineBinaryPayloadSummary(
        array_index=array.array_index,
        declared_payload_bytes=declared_bytes,
        decoded_payload_bytes=decoded_bytes,
        payload_sha256=digest.hexdigest(),
        scalar_count=scalar_count,
        tuple_count=tuple_count,
        number_of_components=array.number_of_components,
        vtk_type=array.vtk_type,
    )


__all__ = [
    "DEFAULT_IO_CHUNK_BYTES",
    "DEFAULT_MAX_XML_TAG_BYTES",
    "FileSegment",
    "InlineBinaryDecodeError",
    "InlineBinaryPayloadSummary",
    "NativeCaseRecord",
    "NativeSourceError",
    "NativeSourceIntegrityError",
    "NativeSourcePin",
    "PinnedFile",
    "PinnedSurfaceArea",
    "PinnedVolumePart",
    "ResolvedNativeCase",
    "SegmentVerification",
    "SegmentedReader",
    "VTKDataArrayIndex",
    "VTKPieceIndex",
    "VTKXMLIndex",
    "VTKXMLIndexError",
    "index_inline_binary_vtk_xml",
    "load_native_source_pin",
    "open_verified_monolithic",
    "open_verified_multipart",
    "stream_inline_binary_payload",
    "verify_file_segments",
]
