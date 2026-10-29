"""Bounded-memory native volume-cell regions for report-only diagnostics.

Regional membership is derived only from the verified native VTU.  Participant
files never provide coordinates or region labels.  The six geometry/topology
arrays needed by the pinned DrivAerML volumes are decoded to temporary raw
files.  Only Points remains memory-mapped; every topology interval is mapped,
copied into an owned block, and closed before VTK sees it.  Native cells are
then assembled in bounded blocks and their centres are calculated by VTK
9.5.2's own ``vtkCellCenters::ComputeCellCenters`` implementation.

Using VTK itself is important.  The pinned volumes contain tetrahedra,
hexahedra, wedges, pyramids, and legacy-XML polyhedra.  Several centres are not
arithmetic vertex means: wedges use the literal parametric coordinate
``0.333333``, pyramids use ``(0.4, 0.4, 0.2)``, and polyhedra use VTK's signed
volume centroid reconstructed from the retained face stream.
"""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import vtk
from vtk.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)

from .regional_diagnostics import (
    VOLUME_REGION_DEFINITION,
    RegionalDiagnosticError,
    RegionAssignmentHasher,
    volume_region_codes,
)
from .source import (
    InlineBinaryDecodeError,
    InlineBinaryPayloadSummary,
    VTKDataArrayIndex,
    VTKXMLIndex,
    stream_inline_binary_payload,
)

VOLUME_REGION_LABELS = VOLUME_REGION_DEFINITION.region_ids
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14
VTK_POLYHEDRON = 42
_FIXED_ARITY = {
    VTK_TETRA: 4,
    VTK_HEXAHEDRON: 8,
    VTK_WEDGE: 6,
    VTK_PYRAMID: 5,
}
_SUPPORTED_TYPES = (*_FIXED_ARITY, VTK_POLYHEDRON)
_DTYPE_CODES = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
}
_COORDINATE_DOMAIN = b"autocfd5-volume-cell-centres-f64le-v1\0"
_MAX_GEOMETRY_BLOCK_CELLS = 1_000_000


class VolumeRegionGeometryError(ValueError):
    """Raised when native VTU geometry cannot define exact report regions."""


@dataclass(frozen=True)
class VolumeRegionSupport:
    """One immutable native-order region code per volume cell plus provenance."""

    codes: np.ndarray
    audit: dict[str, object]


def _dtype(vtk_type: str, byte_order: str) -> np.dtype:
    try:
        code = _DTYPE_CODES[vtk_type]
    except KeyError as error:
        raise VolumeRegionGeometryError(
            f"unsupported native geometry VTK type {vtk_type!r}"
        ) from error
    if code.endswith("1"):
        return np.dtype(code)
    if byte_order == "LittleEndian":
        return np.dtype("<" + code)
    if byte_order == "BigEndian":
        return np.dtype(">" + code)
    raise VolumeRegionGeometryError(f"unsupported VTK byte order {byte_order!r}")


def _unique_array(
    vtk_index: VTKXMLIndex,
    *,
    association: str,
    name: str | None = None,
    required: bool = True,
) -> VTKDataArrayIndex | None:
    rows = vtk_index.arrays_for(association=association, name=name)
    if not rows and not required:
        return None
    if len(rows) != 1:
        label = association if name is None else f"{association}/{name}"
        raise VolumeRegionGeometryError(
            f"native volume must contain exactly one inline-binary {label} array"
        )
    return rows[0]


class _RawSpoolSink:
    """Write one decoded payload to a temporary raw file."""

    def __init__(self, handle: BinaryIO, *, expected_bytes: int | None) -> None:
        self.handle = handle
        self.expected_bytes = expected_bytes
        self.cursor = 0

    def _inspect(self, payload: bytes) -> None:
        del payload

    def write(self, payload: bytes) -> int:
        stop = self.cursor + len(payload)
        if self.expected_bytes is not None and stop > self.expected_bytes:
            raise VolumeRegionGeometryError(
                "native geometry payload exceeds its indexed tuple count"
            )
        self._inspect(payload)
        written = self.handle.write(payload)
        if written != len(payload):
            raise VolumeRegionGeometryError(
                f"temporary geometry file accepted {written} of {len(payload)} bytes"
            )
        self.cursor = stop
        return len(payload)

    def finish(self) -> None:
        if self.expected_bytes is not None and self.cursor != self.expected_bytes:
            raise VolumeRegionGeometryError(
                "native geometry payload is shorter than its indexed tuple count"
            )
        self.handle.flush()


class _PointSpoolSink(_RawSpoolSink):
    """Spool Float32[3] points while auditing every coordinate tuple."""

    def __init__(
        self,
        handle: BinaryIO,
        *,
        expected_bytes: int,
        dtype: np.dtype,
    ) -> None:
        super().__init__(handle, expected_bytes=expected_bytes)
        self.dtype = dtype
        self.tuple_bytes = dtype.itemsize * 3
        self.pending = bytearray()
        self.tuple_count = 0
        self.minimum = np.full(3, math.inf, dtype=np.float64)
        self.maximum = np.full(3, -math.inf, dtype=np.float64)

    def _inspect(self, payload: bytes) -> None:
        self.pending.extend(payload)
        complete_bytes = len(self.pending) - len(self.pending) % self.tuple_bytes
        if not complete_bytes:
            return
        complete = bytes(self.pending[:complete_bytes])
        del self.pending[:complete_bytes]
        values = np.frombuffer(complete, dtype=self.dtype).reshape(-1, 3)
        if not np.all(np.isfinite(values)):
            raise VolumeRegionGeometryError("native volume Points contain non-finite values")
        self.minimum = np.minimum(self.minimum, np.min(values, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(values, axis=0))
        self.tuple_count += values.shape[0]

    def finish(self) -> None:
        super().finish()
        if self.pending:
            raise VolumeRegionGeometryError("native volume Points end within a tuple")


class _TypeSpoolSink(_RawSpoolSink):
    """Spool UInt8 cell types and fail closed on unfamiliar topology."""

    def __init__(self, handle: BinaryIO, *, expected_bytes: int) -> None:
        super().__init__(handle, expected_bytes=expected_bytes)
        self.type_counts = {cell_type: 0 for cell_type in _SUPPORTED_TYPES}

    def _inspect(self, payload: bytes) -> None:
        values = np.frombuffer(payload, dtype=np.uint8)
        supported = np.isin(values, _SUPPORTED_TYPES)
        if not np.all(supported):
            unexpected = np.unique(values[~supported]).tolist()
            raise VolumeRegionGeometryError(
                f"native volume contains unsupported VTK cell types: {unexpected}"
            )
        for cell_type in _SUPPORTED_TYPES:
            self.type_counts[cell_type] += int(np.count_nonzero(values == cell_type))


def _spool_array(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    *,
    tuple_count: int | None,
    path: Path,
    encoded_chunk_size: int,
    sink_kind: str,
) -> tuple[InlineBinaryPayloadSummary, _RawSpoolSink]:
    dtype = _dtype(array.vtk_type, vtk_index.byte_order)
    expected_bytes = (
        None if tuple_count is None else tuple_count * array.number_of_components * dtype.itemsize
    )
    try:
        with path.open("w+b") as handle:
            if sink_kind == "points":
                if expected_bytes is None:
                    raise AssertionError("Points require a declared tuple count")
                sink: _RawSpoolSink = _PointSpoolSink(
                    handle,
                    expected_bytes=expected_bytes,
                    dtype=dtype,
                )
            elif sink_kind == "types":
                if expected_bytes is None:
                    raise AssertionError("types require a declared tuple count")
                sink = _TypeSpoolSink(handle, expected_bytes=expected_bytes)
            elif sink_kind == "raw":
                sink = _RawSpoolSink(handle, expected_bytes=expected_bytes)
            else:
                raise AssertionError(f"unknown geometry spool kind {sink_kind!r}")
            summary = stream_inline_binary_payload(
                stream,
                vtk_index,
                array,
                sink,
                encoded_chunk_size=encoded_chunk_size,
            )
            sink.finish()
    except InlineBinaryDecodeError as error:
        raise VolumeRegionGeometryError(str(error)) from error
    except OSError as error:
        raise VolumeRegionGeometryError(
            f"cannot spool native {sink_kind} payload to temporary storage"
        ) from error
    if tuple_count is not None and summary.tuple_count != tuple_count:
        raise VolumeRegionGeometryError(
            f"native {sink_kind} tuple count differs from the indexed VTK Piece"
        )
    return summary, sink


def classify_volume_centres(centres_m: object) -> np.ndarray:
    """Return frozen first-match four-region codes for finite ``[cell,3]`` centres."""

    try:
        return volume_region_codes(centres_m)
    except RegionalDiagnosticError as error:
        raise VolumeRegionGeometryError(str(error)) from error


def _native_points_file(
    source: np.memmap,
    *,
    path: Path,
    block_points: int,
) -> tuple[np.memmap, bool]:
    """Return native-endian Float32 points, converting out-of-place if needed."""

    if source.dtype.isnative:
        return source, False
    try:
        target = np.memmap(
            path,
            dtype=np.float32,
            mode="w+",
            shape=source.shape,
            order="C",
        )
        for first in range(0, source.shape[0], block_points):
            last = min(first + block_points, source.shape[0])
            target[first:last] = source[first:last]
        target.flush()
        del target
        return (
            np.memmap(
                path,
                dtype=np.float32,
                mode="r",
                shape=source.shape,
                order="C",
            ),
            True,
        )
    except (MemoryError, OSError, ValueError) as error:
        raise VolumeRegionGeometryError(
            "cannot convert native Points to host byte order in bounded storage"
        ) from error


def _vtk_array(values: np.ndarray, *, array_type: int | None = None) -> object:
    contiguous = np.ascontiguousarray(values)
    return numpy_to_vtk(
        contiguous,
        deep=False,
        array_type=array_type,
    ), contiguous


def _vtk_id_array(values: np.ndarray) -> tuple[object, np.ndarray]:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return numpy_to_vtkIdTypeArray(contiguous, deep=False), contiguous


@dataclass
class _ReductionState:
    assignment_hasher: RegionAssignmentHasher
    coordinate_digest: object
    coordinate_float64_digest: object
    coordinate_float32_digest: object
    minimum: np.ndarray
    maximum: np.ndarray
    face_cursor: int = 0
    face_count: int = 0


def _validate_block_offsets(
    local_types: np.ndarray,
    global_offsets: np.ndarray,
    *,
    previous_offset: int,
    raw_cell_start: int,
) -> np.ndarray:
    local_offsets = np.empty(local_types.size + 1, dtype=np.int64)
    local_offsets[0] = 0
    local_offsets[1:] = global_offsets - previous_offset
    arities = np.diff(local_offsets)
    if np.any(arities <= 0):
        mismatch = int(np.flatnonzero(arities <= 0)[0])
        raise VolumeRegionGeometryError(
            f"native cell offsets are not strictly increasing at raw_cell_id "
            f"{raw_cell_start + mismatch}"
        )
    for cell_type, expected_arity in _FIXED_ARITY.items():
        rows = local_types == cell_type
        if np.any(arities[rows] != expected_arity):
            local = int(np.flatnonzero(rows & (arities != expected_arity))[0])
            raise VolumeRegionGeometryError(
                "native cell offset differs from its VTK cell type arity at "
                f"raw_cell_id {raw_cell_start + local}: found {int(arities[local])}, "
                f"expected {expected_arity}"
            )
    poly_rows = local_types == VTK_POLYHEDRON
    if np.any(arities[poly_rows] < 4):
        local = int(np.flatnonzero(poly_rows & (arities < 4))[0])
        raise VolumeRegionGeometryError(
            f"native polyhedron has fewer than four unique points at raw_cell_id "
            f"{raw_cell_start + local}"
        )
    return local_offsets


def _copy_memmap_slice(
    path: Path,
    *,
    dtype: np.dtype,
    scalar_start: int,
    scalar_count: int,
    output_dtype: np.dtype,
) -> np.ndarray:
    """Map one raw scalar interval, copy it, and close the mapping immediately."""

    if scalar_start < 0 or scalar_count < 0:
        raise VolumeRegionGeometryError("native topology slice bounds cannot be negative")
    if scalar_count == 0:
        return np.empty(0, dtype=output_dtype)
    mapping: np.memmap | None = None
    try:
        mapping = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=scalar_start * dtype.itemsize,
            shape=(scalar_count,),
            order="C",
        )
        return np.array(mapping, dtype=output_dtype, copy=True, order="C")
    except (MemoryError, OSError, ValueError) as error:
        raise VolumeRegionGeometryError(
            f"cannot map bounded native topology slice from {path.name}"
        ) from error
    finally:
        _close_memmap(mapping)


def _legacy_face_locations(
    local_types: np.ndarray,
    global_face_offsets: np.ndarray | None,
    faces_path: Path | None,
    faces_dtype: np.dtype | None,
    faces_scalar_count: int,
    state: _ReductionState,
    *,
    raw_cell_start: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    poly_rows = np.flatnonzero(local_types == VTK_POLYHEDRON)
    if global_face_offsets is None:
        if poly_rows.size:
            raise VolumeRegionGeometryError(
                "native polyhedra require legacy faces and faceoffsets arrays"
            )
        return None, None
    non_poly = local_types != VTK_POLYHEDRON
    if np.any(global_face_offsets[non_poly] != -1):
        local = int(np.flatnonzero(non_poly & (global_face_offsets != -1))[0])
        raise VolumeRegionGeometryError(
            f"non-polyhedron has a face stream at raw_cell_id {raw_cell_start + local}"
        )
    if not poly_rows.size:
        return None, None
    if faces_path is None or faces_dtype is None:
        raise VolumeRegionGeometryError("native polyhedra require a faces array")
    ends = global_face_offsets[poly_rows]
    starts = np.empty_like(ends)
    starts[0] = state.face_cursor
    if ends.size > 1:
        starts[1:] = ends[:-1]
    if np.any(ends <= starts) or int(ends[-1]) > faces_scalar_count:
        local_poly = int(np.flatnonzero((ends <= starts) | (ends > faces_scalar_count))[0])
        raise VolumeRegionGeometryError(
            "invalid legacy polyhedron faceoffset at raw_cell_id "
            f"{raw_cell_start + int(poly_rows[local_poly])}"
        )
    face_locations = np.full(local_types.size, -1, dtype=np.int64)
    face_locations[poly_rows] = starts - state.face_cursor
    local_faces = _copy_memmap_slice(
        faces_path,
        dtype=faces_dtype,
        scalar_start=state.face_cursor,
        scalar_count=int(ends[-1]) - state.face_cursor,
        output_dtype=np.dtype(np.int64),
    )

    # Every stream has at least [nfaces, npts, p0, p1, p2] per face.
    stream_lengths = ends - starts
    headings = local_faces[starts - state.face_cursor]
    if np.any(headings < 4) or np.any(stream_lengths < 1 + 4 * headings):
        bad = (headings < 4) | (stream_lengths < 1 + 4 * headings)
        local_poly = int(np.flatnonzero(bad)[0])
        raise VolumeRegionGeometryError(
            "malformed legacy polyhedron face stream at raw_cell_id "
            f"{raw_cell_start + int(poly_rows[local_poly])}"
        )
    for local_poly, (start, end, number_of_faces) in enumerate(
        zip(starts, ends, headings, strict=True)
    ):
        cursor = int(start - state.face_cursor + 1)
        local_end = int(end - state.face_cursor)
        for _ in range(int(number_of_faces)):
            if cursor >= local_end:
                raise VolumeRegionGeometryError(
                    "truncated legacy polyhedron face stream at raw_cell_id "
                    f"{raw_cell_start + int(poly_rows[local_poly])}"
                )
            number_of_points = int(local_faces[cursor])
            if number_of_points < 3 or cursor + 1 + number_of_points > local_end:
                raise VolumeRegionGeometryError(
                    "invalid legacy polyhedron face at raw_cell_id "
                    f"{raw_cell_start + int(poly_rows[local_poly])}"
                )
            cursor += 1 + number_of_points
        if cursor != local_end:
            raise VolumeRegionGeometryError(
                "legacy polyhedron face count does not consume its faceoffset at "
                f"raw_cell_id {raw_cell_start + int(poly_rows[local_poly])}"
            )
    state.face_cursor = int(ends[-1])
    state.face_count += int(np.sum(headings, dtype=np.int64))
    return face_locations, local_faces


def _compute_vtk_centres(
    points: np.memmap,
    local_types: np.ndarray,
    local_offsets: np.ndarray,
    local_connectivity: np.ndarray,
    face_locations: np.ndarray | None,
    local_faces: np.ndarray | None,
) -> np.ndarray:
    """Construct one bounded native-order block and invoke VTK itself."""

    vtk_points_data, points_owner = _vtk_array(points)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_data)
    vtk_offsets, offsets_owner = _vtk_id_array(local_offsets)
    vtk_connectivity, connectivity_owner = _vtk_id_array(local_connectivity)
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetData(vtk_offsets, vtk_connectivity)
    vtk_types, types_owner = _vtk_array(
        local_types,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    owners: list[object] = [
        points_owner,
        offsets_owner,
        connectivity_owner,
        types_owner,
    ]
    if face_locations is None:
        grid.SetCells(vtk_types, vtk_cells)
    else:
        if local_faces is None:
            raise AssertionError("face locations require a face stream")
        vtk_face_locations, face_locations_owner = _vtk_id_array(face_locations)
        vtk_faces, faces_owner = _vtk_id_array(local_faces)
        owners.extend((face_locations_owner, faces_owner))
        # VTK 9.5.2 retains this legacy overload specifically to convert the
        # XML 0.1 faces/faceoffsets representation used by the pinned source.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            grid.SetCells(
                vtk_types,
                vtk_cells,
                vtk_face_locations,
                vtk_faces,
            )
        modern_faces = grid.GetPolyhedronFaces()
        if modern_faces is None:
            raise VolumeRegionGeometryError("VTK rejected the polyhedron face stream")
        face_point_ids = np.asarray(
            vtk_to_numpy(modern_faces.GetConnectivityArray()),
            dtype=np.int64,
        )
        if np.any(face_point_ids < 0) or np.any(face_point_ids >= points.shape[0]):
            raise VolumeRegionGeometryError(
                "native polyhedron face stream contains an invalid point ID"
            )

    centres_array = vtk.vtkDoubleArray()
    centres_array.SetNumberOfComponents(3)
    centres_array.SetNumberOfTuples(local_types.size)
    vtk.vtkCellCenters.ComputeCellCenters(grid, centres_array)
    centres = np.array(vtk_to_numpy(centres_array), dtype=np.float64, copy=True)
    del owners
    if centres.shape != (local_types.size, 3) or not np.all(np.isfinite(centres)):
        raise VolumeRegionGeometryError("VTK calculated invalid native volume cell centres")
    return centres


def _reduce_geometry_blocks(
    *,
    points: np.memmap,
    types_path: Path,
    types_dtype: np.dtype,
    connectivity_path: Path,
    connectivity_dtype: np.dtype,
    connectivity_scalar_count: int,
    offsets_path: Path,
    offsets_dtype: np.dtype,
    faces_path: Path | None,
    faces_dtype: np.dtype | None,
    faces_scalar_count: int,
    face_offsets_path: Path | None,
    face_offsets_dtype: np.dtype | None,
    cell_count: int,
    point_count: int,
    block_cells: int,
    codes: np.ndarray,
) -> tuple[_ReductionState, int]:
    if vtk.vtkVersion.GetVTKVersion() != "9.5.2":
        raise VolumeRegionGeometryError(
            "regional volume centres require the pinned VTK 9.5.2 runtime"
        )
    state = _ReductionState(
        assignment_hasher=RegionAssignmentHasher(VOLUME_REGION_DEFINITION, cell_count),
        coordinate_digest=hashlib.sha256(_COORDINATE_DOMAIN),
        coordinate_float64_digest=hashlib.sha256(),
        coordinate_float32_digest=hashlib.sha256(),
        minimum=np.full(3, math.inf, dtype=np.float64),
        maximum=np.full(3, -math.inf, dtype=np.float64),
    )
    previous_offset = 0
    for first in range(0, cell_count, block_cells):
        last = min(first + block_cells, cell_count)
        count = last - first
        local_types = _copy_memmap_slice(
            types_path,
            dtype=types_dtype,
            scalar_start=first,
            scalar_count=count,
            output_dtype=np.dtype(np.uint8),
        )
        global_offsets = _copy_memmap_slice(
            offsets_path,
            dtype=offsets_dtype,
            scalar_start=first,
            scalar_count=count,
            output_dtype=np.dtype(np.int64),
        )
        local_offsets = _validate_block_offsets(
            local_types,
            global_offsets,
            previous_offset=previous_offset,
            raw_cell_start=first,
        )
        stop = int(global_offsets[-1])
        if stop > connectivity_scalar_count:
            raise VolumeRegionGeometryError(
                f"native cell offset exceeds connectivity at raw_cell_id {last - 1}"
            )
        local_connectivity = _copy_memmap_slice(
            connectivity_path,
            dtype=connectivity_dtype,
            scalar_start=previous_offset,
            scalar_count=stop - previous_offset,
            output_dtype=np.dtype(np.int64),
        )
        if np.any(local_connectivity < 0) or np.any(local_connectivity >= point_count):
            raise VolumeRegionGeometryError("native connectivity contains an invalid point ID")
        if face_offsets_path is None:
            global_face_offsets = None
        else:
            if face_offsets_dtype is None:
                raise AssertionError("faceoffsets path requires a dtype")
            global_face_offsets = _copy_memmap_slice(
                face_offsets_path,
                dtype=face_offsets_dtype,
                scalar_start=first,
                scalar_count=count,
                output_dtype=np.dtype(np.int64),
            )
        face_locations, local_faces = _legacy_face_locations(
            local_types,
            global_face_offsets,
            faces_path,
            faces_dtype,
            faces_scalar_count,
            state,
            raw_cell_start=first,
        )
        centres = _compute_vtk_centres(
            points,
            local_types,
            local_offsets,
            local_connectivity,
            face_locations,
            local_faces,
        )
        block_codes = classify_volume_centres(centres)
        codes[first:last] = block_codes
        state.assignment_hasher.add_chunk(first, block_codes)
        canonical64 = np.asarray(centres, dtype="<f8", order="C")
        canonical64_bytes = memoryview(canonical64).cast("B")
        state.coordinate_digest.update(canonical64_bytes)
        state.coordinate_float64_digest.update(canonical64_bytes)
        canonical32 = np.asarray(centres, dtype="<f4", order="C")
        state.coordinate_float32_digest.update(memoryview(canonical32).cast("B"))
        state.minimum = np.minimum(state.minimum, np.min(centres, axis=0))
        state.maximum = np.maximum(state.maximum, np.max(centres, axis=0))
        previous_offset = stop
    if previous_offset != connectivity_scalar_count:
        raise VolumeRegionGeometryError(
            "final native cell offset differs from connectivity scalar count"
        )
    if faces_path is not None and state.face_cursor != faces_scalar_count:
        raise VolumeRegionGeometryError("final native faceoffset differs from faces scalar count")
    return state, previous_offset


def _payload_audit(summary: InlineBinaryPayloadSummary) -> dict[str, object]:
    return {
        "vtk_array_index": summary.array_index,
        "vtk_type": summary.vtk_type,
        "number_of_components": summary.number_of_components,
        "tuple_count": summary.tuple_count,
        "scalar_count": summary.scalar_count,
        "decoded_payload_bytes": summary.decoded_payload_bytes,
        "payload_sha256": summary.payload_sha256,
    }


def _close_memmap(values: np.memmap | None) -> None:
    if values is None:
        return
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


def build_volume_region_support(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    *,
    encoded_chunk_size: int,
    calculation_block_cells: int = 1_000_000,
) -> VolumeRegionSupport:
    """Build trusted native-order four-region codes with bounded memory."""

    if not isinstance(vtk_index, VTKXMLIndex):
        raise VolumeRegionGeometryError("vtk_index must be a validated VTKXMLIndex")
    if vtk_index.dataset_type != "UnstructuredGrid" or len(vtk_index.pieces) != 1:
        raise VolumeRegionGeometryError(
            "regional volume support requires one VTK UnstructuredGrid Piece"
        )
    if (
        not isinstance(encoded_chunk_size, int)
        or isinstance(encoded_chunk_size, bool)
        or encoded_chunk_size < 1
        or not isinstance(calculation_block_cells, int)
        or isinstance(calculation_block_cells, bool)
        or calculation_block_cells < 1
    ):
        raise VolumeRegionGeometryError("volume region chunk limits must be positive")
    effective_block_cells = min(
        calculation_block_cells,
        _MAX_GEOMETRY_BLOCK_CELLS,
    )
    piece = vtk_index.pieces[0]
    point_count = piece.number_of_points
    cell_count = piece.number_of_cells
    if point_count < 1 or cell_count < 1:
        raise VolumeRegionGeometryError("native volume geometry cannot be empty")

    points_array = _unique_array(vtk_index, association="Points")
    connectivity_array = _unique_array(
        vtk_index,
        association="Cells",
        name="connectivity",
    )
    offsets_array = _unique_array(vtk_index, association="Cells", name="offsets")
    types_array = _unique_array(vtk_index, association="Cells", name="types")
    faces_array = _unique_array(
        vtk_index,
        association="Cells",
        name="faces",
        required=False,
    )
    face_offsets_array = _unique_array(
        vtk_index,
        association="Cells",
        name="faceoffsets",
        required=False,
    )
    if not all(
        isinstance(row, VTKDataArrayIndex)
        for row in (points_array, connectivity_array, offsets_array, types_array)
    ):
        raise AssertionError("required native geometry array resolution failed")
    assert isinstance(points_array, VTKDataArrayIndex)
    assert isinstance(connectivity_array, VTKDataArrayIndex)
    assert isinstance(offsets_array, VTKDataArrayIndex)
    assert isinstance(types_array, VTKDataArrayIndex)
    if points_array.vtk_type != "Float32" or points_array.number_of_components != 3:
        raise VolumeRegionGeometryError("native volume Points must use Float32[3]")
    for name, array in (
        ("connectivity", connectivity_array),
        ("offsets", offsets_array),
    ):
        if array.number_of_components != 1 or array.vtk_type not in {"Int32", "Int64"}:
            raise VolumeRegionGeometryError(f"native cell {name} must use scalar Int32 or Int64")
    if types_array.vtk_type != "UInt8" or types_array.number_of_components != 1:
        raise VolumeRegionGeometryError("native cell types must use scalar UInt8")
    if (faces_array is None) != (face_offsets_array is None):
        raise VolumeRegionGeometryError(
            "native legacy faces and faceoffsets arrays must appear together"
        )
    for name, array in (("faces", faces_array), ("faceoffsets", face_offsets_array)):
        if array is not None and (
            array.number_of_components != 1 or array.vtk_type not in {"Int32", "Int64"}
        ):
            raise VolumeRegionGeometryError(f"native legacy {name} must use scalar Int32 or Int64")

    codes = np.empty(cell_count, dtype=np.uint8)
    summaries: dict[str, InlineBinaryPayloadSummary] = {}
    points_sink: _RawSpoolSink | None = None
    types_sink: _RawSpoolSink | None = None
    converted_points = False
    try:
        with tempfile.TemporaryDirectory(prefix="autocfd5-volume-regions-") as directory:
            root = Path(directory)
            arrays_to_spool: list[tuple[str, VTKDataArrayIndex, int | None, str]] = [
                ("points", points_array, point_count, "points"),
                ("types", types_array, cell_count, "types"),
                ("connectivity", connectivity_array, None, "raw"),
                ("offsets", offsets_array, cell_count, "raw"),
            ]
            if faces_array is not None and face_offsets_array is not None:
                arrays_to_spool.extend(
                    (
                        ("faces", faces_array, None, "raw"),
                        ("faceoffsets", face_offsets_array, cell_count, "raw"),
                    )
                )
            for name, array, tuple_count, sink_kind in arrays_to_spool:
                summary, sink = _spool_array(
                    stream,
                    vtk_index,
                    array,
                    tuple_count=tuple_count,
                    path=root / f"{name}.raw",
                    encoded_chunk_size=encoded_chunk_size,
                    sink_kind=sink_kind,
                )
                summaries[name] = summary
                if name == "points":
                    points_sink = sink
                elif name == "types":
                    types_sink = sink
            if not isinstance(points_sink, _PointSpoolSink):
                raise AssertionError("point spool sink has the wrong type")
            if not isinstance(types_sink, _TypeSpoolSink):
                raise AssertionError("cell-type spool sink has the wrong type")
            if points_sink.tuple_count != point_count:
                raise VolumeRegionGeometryError(
                    "point audit count differs from the indexed VTK Piece"
                )
            if sum(types_sink.type_counts.values()) != cell_count:
                raise VolumeRegionGeometryError(
                    "cell-type audit count differs from the indexed VTK Piece"
                )
            has_polyhedra = types_sink.type_counts[VTK_POLYHEDRON] > 0
            if has_polyhedra and "faces" not in summaries:
                raise VolumeRegionGeometryError(
                    "native polyhedra require legacy faces and faceoffsets arrays"
                )
            if not has_polyhedra and "faces" in summaries:
                raise VolumeRegionGeometryError(
                    "legacy face arrays are present but no polyhedron cells exist"
                )

            maps: list[np.memmap] = []
            try:
                source_points = np.memmap(
                    root / "points.raw",
                    dtype=_dtype(points_array.vtk_type, vtk_index.byte_order),
                    mode="r",
                    shape=(point_count, 3),
                    order="C",
                )
                maps.append(source_points)
                points, converted_points = _native_points_file(
                    source_points,
                    path=root / "points-native.raw",
                    block_points=effective_block_cells,
                )
                if points is not source_points:
                    _close_memmap(source_points)
                    maps.pop()
                    maps.append(points)
                reduction, final_offset = _reduce_geometry_blocks(
                    points=points,
                    types_path=root / "types.raw",
                    types_dtype=np.dtype(np.uint8),
                    connectivity_path=root / "connectivity.raw",
                    connectivity_dtype=_dtype(connectivity_array.vtk_type, vtk_index.byte_order),
                    connectivity_scalar_count=summaries["connectivity"].scalar_count,
                    offsets_path=root / "offsets.raw",
                    offsets_dtype=_dtype(offsets_array.vtk_type, vtk_index.byte_order),
                    faces_path=(root / "faces.raw" if faces_array is not None else None),
                    faces_dtype=(
                        _dtype(faces_array.vtk_type, vtk_index.byte_order)
                        if faces_array is not None
                        else None
                    ),
                    faces_scalar_count=(
                        summaries["faces"].scalar_count if faces_array is not None else 0
                    ),
                    face_offsets_path=(
                        root / "faceoffsets.raw" if face_offsets_array is not None else None
                    ),
                    face_offsets_dtype=(
                        _dtype(face_offsets_array.vtk_type, vtk_index.byte_order)
                        if face_offsets_array is not None
                        else None
                    ),
                    cell_count=cell_count,
                    point_count=point_count,
                    block_cells=effective_block_cells,
                    codes=codes,
                )
            finally:
                for values in reversed(maps):
                    _close_memmap(values)
    except VolumeRegionGeometryError:
        raise
    except (MemoryError, OSError, ValueError, RuntimeError) as error:
        raise VolumeRegionGeometryError(
            "cannot construct bounded native volume region support"
        ) from error

    counts = np.bincount(codes, minlength=len(VOLUME_REGION_LABELS))
    if counts.shape != (len(VOLUME_REGION_LABELS),) or int(np.sum(counts)) != cell_count:
        raise VolumeRegionGeometryError(
            "native region assignment does not cover every cell exactly once"
        )
    assignment = reduction.assignment_hasher.finalize()
    codes.setflags(write=False)
    topology_validation: dict[str, object] = {
        "connectivity_scalar_count": summaries["connectivity"].scalar_count,
        "final_offset": final_offset,
        "offsets_match_fixed_type_arity": True,
        "polyhedron_variable_arity_validated": True,
        "point_ids_in_range": True,
    }
    if "faces" in summaries:
        topology_validation.update(
            {
                "faces_scalar_count": summaries["faces"].scalar_count,
                "final_faceoffset": reduction.face_cursor,
                "polyhedron_face_count": reduction.face_count,
                "legacy_faces_faceoffsets_validated": True,
            }
        )
    supported_arity: dict[str, object] = {
        str(cell_type): arity for cell_type, arity in _FIXED_ARITY.items()
    }
    supported_arity[str(VTK_POLYHEDRON)] = "variable_from_connectivity_offsets"
    bounded_storage = {
        "points": "temporary_raw_file_read_only_memmap",
        "types": "per_block_memmap_copy_then_close",
        "connectivity": "per_block_memmap_copy_then_close",
        "offsets": "per_block_memmap_copy_then_close",
        "faces": ("per_block_memmap_copy_then_close" if "faces" in summaries else "absent"),
        "faceoffsets": (
            "per_block_memmap_copy_then_close" if "faceoffsets" in summaries else "absent"
        ),
        "region_codes": "returned_uint8_array",
    }
    return VolumeRegionSupport(
        codes=codes,
        audit={
            "method": "vtk_9_5_2_GetParametricCenter_EvaluateLocation_all_native_cell_types",
            "vtk_version": vtk.vtkVersion.GetVTKVersion(),
            "vtk_smp_backend": vtk.vtkSMPTools.GetBackend(),
            "coordinate_frame": "raw_native_VTU_coordinates",
            "coordinate_units": "m",
            "cell_order": "native_GetCell_raw_cell_id_order",
            "supported_vtk_cell_type_arity": supported_arity,
            "vtk_cell_type_counts": {
                str(cell_type): types_sink.type_counts[cell_type] for cell_type in _SUPPORTED_TYPES
            },
            "entity_count": cell_count,
            "region_labels": list(VOLUME_REGION_LABELS),
            "region_entity_count": {
                label: int(count) for label, count in zip(VOLUME_REGION_LABELS, counts, strict=True)
            },
            "region_assignment": assignment,
            "assignment_identity_sha256": assignment["sha256"],
            "coordinate_identity_sha256": reduction.coordinate_digest.hexdigest(),
            "coordinate_float64_le_sha256": (reduction.coordinate_float64_digest.hexdigest()),
            "coordinate_float32_le_sha256": (reduction.coordinate_float32_digest.hexdigest()),
            "minimum_xyz_m": reduction.minimum.tolist(),
            "maximum_xyz_m": reduction.maximum.tolist(),
            "geometry_payloads": {
                name: _payload_audit(summary) for name, summary in summaries.items()
            },
            "topology_validation": topology_validation,
            "bounded_storage": bounded_storage,
            "host_byte_order": sys.byteorder,
            "points_host_byte_order_conversion": converted_points,
            "requested_calculation_block_cells": calculation_block_cells,
            "calculation_block_cells": effective_block_cells,
            "maximum_geometry_block_cells": _MAX_GEOMETRY_BLOCK_CELLS,
            "all_cells_assigned_exactly_once": True,
        },
    )


__all__ = [
    "VOLUME_REGION_LABELS",
    "VolumeRegionGeometryError",
    "VolumeRegionSupport",
    "build_volume_region_support",
    "classify_volume_centres",
]
