"""Candidate geometric producer for DrivAerML AutoCFD5 Cp mappings.

This module implements the geometry rules recorded in the closed candidate
contract.  It does not confer owner visual approval and must not be described
as active scoring support.  In particular, every returned row retains an
explicit validity and reason and can be converted to
``CpProbeMappingEvidence`` for the contract-level coverage validator.

The case STL is parsed as ASCII in one pass.  Facets retain their zero-based
global order across named ``solid`` blocks.  VTK is optional and is used only
by ``Vtk95VtpPolygonLocator`` to read the native VTP and discover polygon
bounds candidates.  Final closest-point, normal, gate, and tie calculations
are performed here in binary64.  That optional adapter deliberately requires
VTK 9.5.2; changing the dependency is a new candidate algorithm, not an
in-place change to frozen evidence.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Protocol, Sequence

import numpy as np

from reference.drivaerml.autocfd5 import (
    NATIVE_BRIDGE_MAX_DISTANCE_M,
    NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT,
    NOMINAL_PROBE_DISPLACEMENT_MAX_M,
    AutoCFD5Error,
    CpComponentRule,
    CpProbeDefinition,
    CpProbeMappingEvidence,
)


STL_DISTANCE_TIE_M = 1.0e-6
CUT_VERTEX_PLANE_TOLERANCE_M = 1.0e-9
NATIVE_DISTANCE_TIE_M = 1.0e-6
NATIVE_NORMAL_TIE = 1.0e-12
UNNORMALIZED_AREA_VECTOR_MIN_M2 = 1.0e-15
NOMINAL_PROBE_DISPLACEMENT_REVIEW_M = 0.2089635
NATIVE_BRIDGE_DISTANCE_REVIEW_M = 5.0e-4
REQUIRED_VTK_VERSION = "9.5.2"


class CpMappingError(AutoCFD5Error):
    """Raised for malformed sources or an invalid mapping invocation."""


@dataclass(frozen=True)
class RetainedFileSnapshot:
    """Mutation-sensitive identity of one already-open regular file."""

    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_handle(cls, handle: BinaryIO, *, label: str) -> "RetainedFileSnapshot":
        try:
            value = os.fstat(handle.fileno())
        except OSError as error:
            raise CpMappingError(
                f"cannot fstat retained {label} descriptor: {error}"
            ) from error
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )


@dataclass
class RetainedVerifiedRegularFile:
    """Retain one source inode from hashing through every parser pass.

    Only the supplied pathname is opened, and it is opened exactly once.  Code
    which can only accept a filename receives a verified ``/proc/self/fd`` or
    ``/dev/fd`` alias to this descriptor.  The mutation-sensitive ``fstat``
    identity is checked before and after all reads.  Consequently, replacing
    the pathname cannot redirect a later STL or VTK pass to different bytes.
    """

    source_path: Path
    label: str
    handle: BinaryIO
    snapshot: RetainedFileSnapshot
    _descriptor_path: Path | None = None
    _final_verification_complete: bool = False

    @classmethod
    def open(
        cls, source_path: str | Path, *, label: str
    ) -> "RetainedVerifiedRegularFile":
        path = Path(source_path)
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as error:
            raise CpMappingError(f"cannot open {label} {path}: {error}") from error
        try:
            try:
                value = os.fstat(descriptor)
            except OSError as error:
                raise CpMappingError(
                    f"cannot fstat retained {label} descriptor: {error}"
                ) from error
            snapshot = RetainedFileSnapshot(
                device=int(value.st_dev),
                inode=int(value.st_ino),
                mode=int(value.st_mode),
                size_bytes=int(value.st_size),
                mtime_ns=int(value.st_mtime_ns),
                ctime_ns=int(value.st_ctime_ns),
            )
            if not stat.S_ISREG(snapshot.mode):
                raise CpMappingError(f"{label} is not a regular file: {path}")
            handle = os.fdopen(descriptor, "rb", buffering=0)
            descriptor = None
            return cls(
                source_path=path,
                label=label,
                handle=handle,
                snapshot=snapshot,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            raise

    def __enter__(self) -> "RetainedVerifiedRegularFile":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.handle.close()

    def assert_unchanged(self, *, context: str) -> None:
        current = RetainedFileSnapshot.from_handle(self.handle, label=self.label)
        try:
            pathname = os.stat(self.source_path)
            pathname_identity = (int(pathname.st_dev), int(pathname.st_ino))
        except OSError:
            pathname_identity = None
        if current != self.snapshot or pathname_identity != (
            self.snapshot.device,
            self.snapshot.inode,
        ):
            raise CpMappingError(
                f"retained {self.label} {self.source_path} changed {context}; "
                "the pathname no longer names the retained inode, or device, "
                "inode, mode, size, mtime, or ctime differs from the pre-hash "
                "fstat snapshot"
            )

    def sha256(self, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
        """Hash this exact open inode without resolving its pathname again."""

        if (
            not isinstance(chunk_bytes, int)
            or isinstance(chunk_bytes, bool)
            or chunk_bytes < 1
        ):
            raise CpMappingError("chunk_bytes must be a positive integer")
        self.assert_unchanged(context="before hashing")
        digest = hashlib.sha256()
        try:
            self.handle.seek(0)
            while block := self.handle.read(chunk_bytes):
                digest.update(block)
        except OSError as error:
            raise CpMappingError(
                f"cannot hash retained {self.label} descriptor: {error}"
            ) from error
        self.assert_unchanged(context="while hashing")
        return digest.hexdigest()

    def descriptor_path(self) -> Path:
        """Return a verified filename alias for this exact retained inode."""

        self.assert_unchanged(context="before exposing its descriptor path")
        if self._descriptor_path is not None:
            return self._descriptor_path
        descriptor = self.handle.fileno()
        for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = directory / str(descriptor)
            try:
                target = os.stat(candidate)
            except OSError:
                continue
            if (
                int(target.st_dev) == self.snapshot.device
                and int(target.st_ino) == self.snapshot.inode
                and stat.S_ISREG(int(target.st_mode))
            ):
                self._descriptor_path = candidate
                return candidate
        raise CpMappingError(
            f"no safe descriptor filesystem exposes the retained {self.label}; "
            "expected /proc/self/fd or /dev/fd"
        )

    def finalize_verification(self) -> None:
        """Prove the retained identity is unchanged after every consumer."""

        self.assert_unchanged(context="while all hashing and parser passes ran")
        self._final_verification_complete = True

    def verification_receipt(self, *, consumers: Sequence[str]) -> dict[str, object]:
        """Return a path-free declaration only after final verification passes."""

        if not self._final_verification_complete:
            raise CpMappingError(
                f"retained {self.label} has not completed post-read verification"
            )
        self.assert_unchanged(context="after final verification")
        if self._descriptor_path is None:
            raise CpMappingError(
                f"retained {self.label} was not exposed through a descriptor filesystem"
            )
        if not consumers or any(
            not isinstance(value, str) or not value for value in consumers
        ):
            raise CpMappingError("verification consumers must be non-empty strings")
        return {
            "pathname_open": "single_open_retained_through_all_reads",
            "file_type": "regular_file_verified_by_fstat",
            "hash_input": "retained_verified_file_descriptor",
            "parser_inputs": list(consumers),
            "descriptor_transport": "verified_procfs_or_devfs_fd_alias_same_device_inode",
            "identity_fields": [
                "device",
                "inode",
                "mode",
                "size_bytes",
                "mtime_ns",
                "ctime_ns",
            ],
            "post_read_fstat": "unchanged",
        }


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 of one source without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _point3(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise CpMappingError(f"{label} must contain exactly three finite coordinates")
    return array


def _vertices(value: object, label: str, *, exact_count: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise CpMappingError(f"{label} must have shape (N, 3)")
    if exact_count is not None and array.shape[0] != exact_count:
        raise CpMappingError(f"{label} must have shape ({exact_count}, 3)")
    if array.shape[0] < 3 or not np.all(np.isfinite(array)):
        raise CpMappingError(f"{label} must contain at least three finite vertices")
    return array


def _unit_triangle_normal(triangle: object) -> np.ndarray | None:
    vertices = _vertices(triangle, "triangle", exact_count=3)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    magnitude = float(np.linalg.norm(normal))
    if not math.isfinite(magnitude) or magnitude <= UNNORMALIZED_AREA_VECTOR_MIN_M2:
        return None
    return normal / magnitude


def ordered_polygon_unit_normal(vertices: object) -> np.ndarray | None:
    """Recompute the native polygon normal from released connectivity order."""

    points = _vertices(vertices, "polygon vertices")
    area_vector = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    magnitude = float(np.linalg.norm(area_vector))
    if not math.isfinite(magnitude) or magnitude <= UNNORMALIZED_AREA_VECTOR_MIN_M2:
        return None
    return area_vector / magnitude


def _closest_points_on_segments(
    point: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    directions = ends - starts
    denominators = np.einsum("ij,ij->i", directions, directions)
    numerators = np.einsum("ij,ij->i", point[None, :] - starts, directions)
    parameters = np.zeros_like(numerators)
    nonzero = denominators > 0.0
    parameters[nonzero] = numerators[nonzero] / denominators[nonzero]
    np.clip(parameters, 0.0, 1.0, out=parameters)
    return starts + parameters[:, None] * directions


def _closest_points_on_triangles(
    point: object, triangles: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return closest points, distances, unit normals, and valid triangle mask."""

    query = _point3(point, "query point")
    cells = np.asarray(triangles, dtype=np.float64)
    if cells.ndim != 3 or cells.shape[1:] != (3, 3):
        raise CpMappingError("triangles must have shape (N, 3, 3)")
    if not np.all(np.isfinite(cells)):
        raise CpMappingError("triangles contain non-finite coordinates")

    count = cells.shape[0]
    closest = np.full((count, 3), np.nan, dtype=np.float64)
    distances = np.full(count, math.inf, dtype=np.float64)
    unit_normals = np.full((count, 3), np.nan, dtype=np.float64)
    edge0 = cells[:, 1] - cells[:, 0]
    edge1 = cells[:, 2] - cells[:, 0]
    normals = np.cross(edge0, edge1)
    magnitudes = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(magnitudes) & (
        magnitudes > UNNORMALIZED_AREA_VECTOR_MIN_M2
    )
    if not np.any(valid):
        return closest, distances, unit_normals, valid

    indices = np.flatnonzero(valid)
    selected = cells[indices]
    e0 = edge0[indices]
    e1 = edge1[indices]
    n = normals[indices]
    n2 = np.einsum("ij,ij->i", n, n)
    unit_normals[indices] = n / np.sqrt(n2)[:, None]

    from_v0 = query[None, :] - selected[:, 0]
    signed_scale = np.einsum("ij,ij->i", from_v0, n) / n2
    projected = query[None, :] - signed_scale[:, None] * n
    projected_delta = projected - selected[:, 0]
    dot00 = np.einsum("ij,ij->i", e0, e0)
    dot01 = np.einsum("ij,ij->i", e0, e1)
    dot11 = np.einsum("ij,ij->i", e1, e1)
    dot20 = np.einsum("ij,ij->i", projected_delta, e0)
    dot21 = np.einsum("ij,ij->i", projected_delta, e1)
    denominator = dot00 * dot11 - dot01 * dot01
    bary1 = (dot11 * dot20 - dot01 * dot21) / denominator
    bary2 = (dot00 * dot21 - dot01 * dot20) / denominator
    inside = (bary1 >= 0.0) & (bary2 >= 0.0) & (bary1 + bary2 <= 1.0)

    edge_candidates = np.stack(
        [
            _closest_points_on_segments(query, selected[:, 0], selected[:, 1]),
            _closest_points_on_segments(query, selected[:, 1], selected[:, 2]),
            _closest_points_on_segments(query, selected[:, 2], selected[:, 0]),
        ],
        axis=1,
    )
    edge_distance2 = np.sum(
        (edge_candidates - query[None, None, :]) ** 2, axis=2
    )
    edge_choice = np.argmin(edge_distance2, axis=1)
    row = np.arange(len(indices))
    selected_closest = edge_candidates[row, edge_choice]
    selected_closest[inside] = projected[inside]
    closest[indices] = selected_closest
    distances[indices] = np.linalg.norm(selected_closest - query[None, :], axis=1)
    return closest, distances, unit_normals, valid


def closest_point_on_triangle(point: object, triangle: object) -> np.ndarray | None:
    """Return the closest point on one finite, nondegenerate closed triangle."""

    cells = _vertices(triangle, "triangle", exact_count=3)[None, :, :]
    closest, _, _, valid = _closest_points_on_triangles(point, cells)
    return closest[0].copy() if bool(valid[0]) else None


def _closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return _closest_points_on_segments(point, start[None, :], end[None, :])[0]


def closest_point_on_triangle_cut(
    point: object,
    triangle: object,
    *,
    cut_axis: str,
    cut_value_m: float,
    vertex_tolerance_m: float = CUT_VERTEX_PLANE_TOLERANCE_M,
) -> np.ndarray | None:
    """Return the closest point on a triangle's exact declared plane cut.

    Vertices within the declared tolerance are classified as lying on the
    plane.  A fully coplanar triangle contributes its complete closed area;
    all other intersections contribute their closed segment or point.
    """

    query = _point3(point, "query point")
    vertices = _vertices(triangle, "triangle", exact_count=3)
    if _unit_triangle_normal(vertices) is None:
        return None
    if cut_axis not in {"x", "y", "z"}:
        raise CpMappingError("cut_axis must be x, y, or z")
    if not math.isfinite(float(cut_value_m)):
        raise CpMappingError("cut_value_m must be finite")
    if not math.isfinite(float(vertex_tolerance_m)) or vertex_tolerance_m < 0.0:
        raise CpMappingError("vertex_tolerance_m must be finite and non-negative")
    axis = {"x": 0, "y": 1, "z": 2}[cut_axis]
    signed = vertices[:, axis] - float(cut_value_m)
    classification = np.sign(signed).astype(np.int8)
    classification[np.abs(signed) <= vertex_tolerance_m] = 0
    if np.all(classification == 0):
        return closest_point_on_triangle(query, vertices)

    intersections: list[np.ndarray] = [
        vertices[index].copy()
        for index in range(3)
        if classification[index] == 0
    ]
    for first, second in ((0, 1), (1, 2), (2, 0)):
        if classification[first] * classification[second] < 0:
            parameter = signed[first] / (signed[first] - signed[second])
            intersections.append(
                vertices[first] + parameter * (vertices[second] - vertices[first])
            )

    unique: list[np.ndarray] = []
    for candidate in intersections:
        if not any(np.array_equal(candidate, existing) for existing in unique):
            unique.append(candidate)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]

    # A plane/closed-triangle intersection is a segment.  Classification at
    # tolerance can produce duplicate or near-duplicate endpoints; the
    # farthest pair spans the complete closed intersection deterministically.
    best_pair = (0, 1)
    best_distance2 = -1.0
    for first in range(len(unique)):
        for second in range(first + 1, len(unique)):
            distance2 = float(np.sum((unique[first] - unique[second]) ** 2))
            if distance2 > best_distance2:
                best_distance2 = distance2
                best_pair = (first, second)
    return _closest_point_on_segment(
        query, unique[best_pair[0]], unique[best_pair[1]]
    )


def closest_point_on_polygon(point: object, vertices: object) -> np.ndarray | None:
    """Return the deterministic closest point on a native VTK polygon.

    VTK polygon cells are evaluated as the ordered triangle fan rooted at the
    first released connectivity vertex.  This retains the native ordering and
    is exact for the planar convex polygons represented by ``VTK_POLYGON``.
    """

    query = _point3(point, "query point")
    polygon = _vertices(vertices, "polygon vertices")
    if ordered_polygon_unit_normal(polygon) is None:
        return None
    triangles = np.stack(
        [
            np.stack((polygon[0], polygon[index], polygon[index + 1]))
            for index in range(1, len(polygon) - 1)
        ]
    )
    closest, distances, _, valid = _closest_points_on_triangles(query, triangles)
    if not np.any(valid):
        return None
    eligible = np.flatnonzero(valid)
    best_distance = float(np.min(distances[eligible]))
    # Fan triangles share no raw IDs.  Their order is nevertheless fixed; use
    # the earliest fan triangle for an exact-distance tie.
    tied = eligible[distances[eligible] <= best_distance]
    return closest[int(tied[0])].copy()


@dataclass(frozen=True)
class STLTriangleChunk:
    """One bounded-memory block of native ASCII STL facets."""

    solid_name: str
    raw_facet_ids: np.ndarray
    vertices_m: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.solid_name, str) or not self.solid_name:
            raise CpMappingError("solid_name must be non-empty")
        ids = np.asarray(self.raw_facet_ids)
        vertices = np.asarray(self.vertices_m)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise CpMappingError("raw_facet_ids must be a one-dimensional integer array")
        if vertices.shape != (len(ids), 3, 3):
            raise CpMappingError("vertices_m must have shape (N, 3, 3)")
        if not np.all(np.isfinite(vertices)):
            raise CpMappingError("STL chunk contains non-finite vertices")


class AsciiStlTriangleStream:
    """Single-use, bounded-memory ASCII STL parser preserving raw facet IDs."""

    def __init__(self, path: str | Path, *, chunk_facets: int = 16_384) -> None:
        self.path = Path(path)
        if not isinstance(chunk_facets, int) or isinstance(chunk_facets, bool) or chunk_facets < 1:
            raise CpMappingError("chunk_facets must be a positive integer")
        self.chunk_facets = chunk_facets
        self.solid_names: tuple[str, ...] | None = None
        self.facet_count: int | None = None
        self.sha256: str | None = None
        self._consumed = False

    def __iter__(self) -> Iterator[STLTriangleChunk]:
        if self._consumed:
            raise CpMappingError("AsciiStlTriangleStream is single-use")
        self._consumed = True
        digest = hashlib.sha256()
        seen: list[str] = []
        current_solid: str | None = None
        facet_vertices: list[list[float]] | None = None
        raw_facet_id = 0
        chunk_ids: list[int] = []
        chunk_vertices: list[list[list[float]]] = []

        def make_chunk() -> STLTriangleChunk:
            assert current_solid is not None
            result = STLTriangleChunk(
                solid_name=current_solid,
                raw_facet_ids=np.asarray(chunk_ids, dtype=np.int64),
                vertices_m=np.asarray(chunk_vertices, dtype=np.float64),
            )
            chunk_ids.clear()
            chunk_vertices.clear()
            return result

        try:
            with self.path.open("rb", buffering=16 * 1024 * 1024) as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    digest.update(raw_line)
                    try:
                        stripped = raw_line.decode("ascii", errors="strict").strip()
                    except UnicodeDecodeError as error:
                        raise CpMappingError(
                            f"{self.path} is not a strict ASCII STL"
                        ) from error
                    if not stripped:
                        continue
                    if stripped.startswith("solid "):
                        if current_solid is not None or facet_vertices is not None:
                            raise CpMappingError(f"nested solid at line {line_number}")
                        name = stripped[6:].strip()
                        if not name or name in seen:
                            raise CpMappingError(
                                f"missing or duplicate solid name at line {line_number}"
                            )
                        current_solid = name
                        seen.append(name)
                    elif stripped.startswith("facet "):
                        if current_solid is None or facet_vertices is not None:
                            raise CpMappingError(f"orphan or nested facet at line {line_number}")
                        facet_vertices = []
                    elif stripped.startswith("vertex "):
                        if facet_vertices is None:
                            raise CpMappingError(f"orphan vertex at line {line_number}")
                        fields = stripped.split()
                        if len(fields) != 4:
                            raise CpMappingError(f"invalid vertex at line {line_number}")
                        try:
                            vertex = [float(value) for value in fields[1:]]
                        except ValueError as error:
                            raise CpMappingError(
                                f"invalid vertex at line {line_number}"
                            ) from error
                        if not all(math.isfinite(value) for value in vertex):
                            raise CpMappingError(f"non-finite vertex at line {line_number}")
                        facet_vertices.append(vertex)
                    elif stripped == "endfacet":
                        if facet_vertices is None or len(facet_vertices) != 3:
                            raise CpMappingError(
                                f"facet does not contain three vertices at line {line_number}"
                            )
                        chunk_ids.append(raw_facet_id)
                        chunk_vertices.append(facet_vertices)
                        raw_facet_id += 1
                        facet_vertices = None
                        if len(chunk_ids) >= self.chunk_facets:
                            yield make_chunk()
                    elif stripped.startswith("endsolid"):
                        if current_solid is None or facet_vertices is not None:
                            raise CpMappingError(f"orphan endsolid at line {line_number}")
                        declared = stripped[8:].strip()
                        if declared and declared != current_solid:
                            raise CpMappingError(
                                f"endsolid name differs at line {line_number}"
                            )
                        if chunk_ids:
                            yield make_chunk()
                        current_solid = None
                    # ``outer loop`` and ``endloop`` carry no identity or
                    # geometry beyond the strictly parsed vertex records.
        except OSError as error:
            raise CpMappingError(f"cannot read ASCII STL {self.path}") from error

        if current_solid is not None or facet_vertices is not None:
            raise CpMappingError("unterminated ASCII STL solid or facet")
        if chunk_ids:
            raise CpMappingError("internal STL parser chunk crossed a solid boundary")
        if not seen or raw_facet_id == 0:
            raise CpMappingError("ASCII STL contains no named facets")
        self.solid_names = tuple(seen)
        self.facet_count = raw_facet_id
        self.sha256 = digest.hexdigest()


@dataclass(frozen=True)
class StlProjectionResult:
    """Candidate component-constrained case-STL projection for one probe."""

    autocfd_probe_id: int
    nominal_point_m: tuple[float, float, float]
    drivaerml_component: str
    projection_mode: str
    cut_axis: str | None
    cut_value_m: float | None
    owner_review_status: str
    valid: bool
    reason: str
    mapped_point_m: tuple[float, float, float] | None
    raw_stl_triangle_id: int | None
    stl_unit_normal: tuple[float, float, float] | None
    nominal_displacement_m: float | None
    component_facet_count: int
    projection_candidate_count: int
    review_flags: tuple[str, ...]


@dataclass
class _ProjectionAccumulator:
    minimum_distance_m: float = math.inf
    selected_distance_m: float = math.inf
    selected_raw_id: int | None = None
    selected_point: np.ndarray | None = None
    selected_normal: np.ndarray | None = None
    component_facet_count: int = 0
    projection_candidate_count: int = 0

    def consider(
        self,
        raw_ids: np.ndarray,
        points: np.ndarray,
        distances: np.ndarray,
        normals: np.ndarray,
    ) -> None:
        if len(raw_ids) == 0:
            return
        finite = (
            np.isfinite(distances)
            & np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(normals), axis=1)
        )
        raw_ids = raw_ids[finite]
        points = points[finite]
        distances = distances[finite]
        normals = normals[finite]
        self.projection_candidate_count += len(raw_ids)
        if len(raw_ids) == 0:
            return
        new_minimum = min(self.minimum_distance_m, float(np.min(distances)))
        choices: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        if (
            self.selected_raw_id is not None
            and self.selected_distance_m <= new_minimum + STL_DISTANCE_TIE_M
        ):
            assert self.selected_point is not None and self.selected_normal is not None
            choices.append(
                (
                    self.selected_raw_id,
                    self.selected_distance_m,
                    self.selected_point,
                    self.selected_normal,
                )
            )
        tied = np.flatnonzero(distances <= new_minimum + STL_DISTANCE_TIE_M)
        if len(tied):
            local = int(tied[np.argmin(raw_ids[tied])])
            choices.append(
                (
                    int(raw_ids[local]),
                    float(distances[local]),
                    points[local].copy(),
                    normals[local].copy(),
                )
            )
        if choices:
            selected = min(choices, key=lambda row: row[0])
            (
                self.selected_raw_id,
                self.selected_distance_m,
                self.selected_point,
                self.selected_normal,
            ) = selected
        self.minimum_distance_m = new_minimum


def _validate_probe_rules(
    probes: Sequence[CpProbeDefinition], rules: Sequence[CpComponentRule]
) -> tuple[tuple[CpProbeDefinition, CpComponentRule], ...]:
    if not probes:
        raise CpMappingError("at least one Cp probe is required")
    probe_ids = [probe.autocfd_probe_id for probe in probes]
    rule_ids = [rule.autocfd_probe_id for rule in rules]
    if len(probe_ids) != len(set(probe_ids)) or len(rule_ids) != len(set(rule_ids)):
        raise CpMappingError("Cp probes and rules must contain unique IDs")
    if set(probe_ids) != set(rule_ids):
        raise CpMappingError("Cp rules must cover the requested probes exactly once")
    by_id = {rule.autocfd_probe_id: rule for rule in rules}
    return tuple((probe, by_id[probe.autocfd_probe_id]) for probe in probes)


def _chunk_candidates(
    chunk: STLTriangleChunk,
    probe: CpProbeDefinition,
    rule: CpComponentRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    point = np.asarray(probe.point_m, dtype=np.float64)
    triangles = np.asarray(chunk.vertices_m, dtype=np.float64)
    closest, distances, normals, nondegenerate = _closest_points_on_triangles(
        point, triangles
    )
    nondegenerate_count = int(np.count_nonzero(nondegenerate))
    if rule.projection_mode == "component_closest_3d":
        selected = np.flatnonzero(nondegenerate)
        return (
            chunk.raw_facet_ids[selected],
            closest[selected],
            distances[selected],
            normals[selected],
            nondegenerate_count,
        )

    assert rule.projection_mode == "cut_plane_closest"
    assert rule.cut_axis is not None and rule.cut_value_m is not None
    axis = {"x": 0, "y": 1, "z": 2}[rule.cut_axis]
    signed = triangles[:, :, axis] - rule.cut_value_m
    classification = np.sign(signed).astype(np.int8)
    classification[np.abs(signed) <= CUT_VERTEX_PLANE_TOLERANCE_M] = 0
    intersects = nondegenerate & (
        np.all(classification == 0, axis=1)
        | (
            (np.min(classification, axis=1) <= 0)
            & (np.max(classification, axis=1) >= 0)
        )
    )
    ids: list[int] = []
    mapped: list[np.ndarray] = []
    mapped_distances: list[float] = []
    mapped_normals: list[np.ndarray] = []
    for index in np.flatnonzero(intersects):
        candidate = closest_point_on_triangle_cut(
            point,
            triangles[index],
            cut_axis=rule.cut_axis,
            cut_value_m=rule.cut_value_m,
        )
        if candidate is None:
            continue
        ids.append(int(chunk.raw_facet_ids[index]))
        mapped.append(candidate)
        mapped_distances.append(float(np.linalg.norm(candidate - point)))
        mapped_normals.append(normals[index])
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(mapped, dtype=np.float64).reshape((-1, 3)),
        np.asarray(mapped_distances, dtype=np.float64),
        np.asarray(mapped_normals, dtype=np.float64).reshape((-1, 3)),
        nondegenerate_count,
    )


def project_probes_to_case_stl(
    stl_path: str | Path,
    probes: Sequence[CpProbeDefinition],
    rules: Sequence[CpComponentRule],
    *,
    chunk_facets: int = 16_384,
) -> tuple[StlProjectionResult, ...]:
    """Project every requested probe in one bounded-memory ASCII STL pass."""

    pairs = _validate_probe_rules(probes, rules)
    by_component: dict[str, list[int]] = {}
    for index, (_, rule) in enumerate(pairs):
        by_component.setdefault(rule.drivaerml_component, []).append(index)
    accumulators = [_ProjectionAccumulator() for _ in pairs]
    stream = AsciiStlTriangleStream(stl_path, chunk_facets=chunk_facets)
    for chunk in stream:
        positions = by_component.get(chunk.solid_name, ())
        for position in positions:
            probe, rule = pairs[position]
            accumulator = accumulators[position]
            accumulator.component_facet_count += len(chunk.raw_facet_ids)
            ids, mapped, distances, normals, _ = _chunk_candidates(
                chunk, probe, rule
            )
            accumulator.consider(ids, mapped, distances, normals)

    results: list[StlProjectionResult] = []
    for (probe, rule), accumulator in zip(pairs, accumulators):
        reason = ""
        valid = True
        if accumulator.component_facet_count == 0:
            valid = False
            reason = "declared_component_absent"
        elif accumulator.projection_candidate_count == 0:
            valid = False
            reason = (
                "declared_cut_has_no_finite_nondegenerate_intersection"
                if rule.projection_mode == "cut_plane_closest"
                else "declared_component_has_no_finite_nondegenerate_triangle"
            )
        mapped = accumulator.selected_point
        normal = accumulator.selected_normal
        displacement = (
            float(np.linalg.norm(mapped - np.asarray(probe.point_m, dtype=np.float64)))
            if mapped is not None
            else None
        )
        flags: list[str] = []
        if displacement is not None and displacement > NOMINAL_PROBE_DISPLACEMENT_REVIEW_M:
            flags.append("nominal_displacement_gt_7p5pct_wheelbase")
        if displacement is not None and displacement > NOMINAL_PROBE_DISPLACEMENT_MAX_M:
            valid = False
            reason = "nominal_displacement_exceeds_10pct_wheelbase"
        results.append(
            StlProjectionResult(
                autocfd_probe_id=probe.autocfd_probe_id,
                nominal_point_m=tuple(float(value) for value in probe.point_m),
                drivaerml_component=rule.drivaerml_component,
                projection_mode=rule.projection_mode,
                cut_axis=rule.cut_axis,
                cut_value_m=rule.cut_value_m,
                owner_review_status=rule.owner_review_status,
                valid=valid,
                reason=reason,
                mapped_point_m=(
                    tuple(float(value) for value in mapped) if mapped is not None else None
                ),
                raw_stl_triangle_id=accumulator.selected_raw_id,
                stl_unit_normal=(
                    tuple(float(value) for value in normal) if normal is not None else None
                ),
                nominal_displacement_m=displacement,
                component_facet_count=accumulator.component_facet_count,
                projection_candidate_count=accumulator.projection_candidate_count,
                review_flags=tuple(flags),
            )
        )
    return tuple(results)


class NativePolygonLocator(Protocol):
    """Candidate-bounds and raw-connectivity interface used by the bridge."""

    @property
    def polygon_count(self) -> int: ...

    def candidate_polygon_ids(
        self, point_m: Sequence[float], radius_m: float
    ) -> Sequence[int]: ...

    def polygon_vertices(self, raw_polygon_id: int) -> np.ndarray: ...


class ArrayPolygonLocator:
    """Pure-array locator used by golden tests and small deterministic cases."""

    def __init__(self, polygons: Iterable[object]) -> None:
        self._polygons = tuple(
            _vertices(vertices, "polygon vertices").copy() for vertices in polygons
        )
        if not self._polygons:
            raise CpMappingError("at least one native polygon is required")
        self._minimum = np.asarray(
            [np.min(vertices, axis=0) for vertices in self._polygons]
        )
        self._maximum = np.asarray(
            [np.max(vertices, axis=0) for vertices in self._polygons]
        )

    @property
    def polygon_count(self) -> int:
        return len(self._polygons)

    def candidate_polygon_ids(
        self, point_m: Sequence[float], radius_m: float
    ) -> tuple[int, ...]:
        point = _point3(point_m, "candidate point")
        if not math.isfinite(radius_m) or radius_m < 0.0:
            raise CpMappingError("candidate radius must be finite and non-negative")
        lower = point - radius_m
        upper = point + radius_m
        mask = np.all(self._maximum >= lower, axis=1) & np.all(
            self._minimum <= upper, axis=1
        )
        return tuple(int(value) for value in np.flatnonzero(mask))

    def polygon_vertices(self, raw_polygon_id: int) -> np.ndarray:
        if not isinstance(raw_polygon_id, int) or isinstance(raw_polygon_id, bool):
            raise CpMappingError("raw polygon ID must be an integer")
        if not 0 <= raw_polygon_id < self.polygon_count:
            raise CpMappingError("raw polygon ID is outside the native support")
        return self._polygons[raw_polygon_id].copy()


class Vtk95VtpPolygonLocator:
    """Optional exact-VTK adapter used only for VTP bounds candidates."""

    def __init__(self, path: str | Path) -> None:
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy
            from vtkmodules.vtkCommonCore import vtkIdList, vtkVersion
            from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
            from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        except ImportError as error:  # pragma: no cover - depends on optional VTK
            raise CpMappingError(
                f"VTP mapping requires optional VTK {REQUIRED_VTK_VERSION}"
            ) from error
        version = vtkVersion.GetVTKVersion()
        if version != REQUIRED_VTK_VERSION:  # pragma: no cover - version-specific
            raise CpMappingError(
                f"VTP mapping requires VTK {REQUIRED_VTK_VERSION}, found {version}"
            )

        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(Path(path)))
        reader.UpdateInformation()
        for selection in (
            reader.GetPointDataArraySelection(),
            reader.GetCellDataArraySelection(),
        ):
            for index in range(selection.GetNumberOfArrays()):
                selection.DisableArray(selection.GetArrayName(index))
        reader.Update()
        if reader.GetErrorCode():
            raise CpMappingError(f"VTK could not read native VTP {path}")
        polydata = reader.GetOutput()
        if (
            polydata.GetNumberOfCells() != polydata.GetNumberOfPolys()
            or polydata.GetNumberOfVerts()
            or polydata.GetNumberOfLines()
            or polydata.GetNumberOfStrips()
        ):
            raise CpMappingError(
                "native VTP must contain polygons only so raw polygon IDs equal cell IDs"
            )
        if polydata.GetNumberOfPolys() < 1:
            raise CpMappingError("native VTP contains no polygons")
        points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()))
        offsets = np.asarray(vtk_to_numpy(polydata.GetPolys().GetOffsetsArray()))
        connectivity = np.asarray(
            vtk_to_numpy(polydata.GetPolys().GetConnectivityArray())
        )
        if offsets.shape != (polydata.GetNumberOfPolys() + 1,):
            raise CpMappingError("native VTP polygon offsets are malformed")
        if not np.all(np.isfinite(points)):
            raise CpMappingError("native VTP contains non-finite points")
        locator = vtkStaticCellLocator()
        locator.SetDataSet(polydata)
        locator.BuildLocator()

        self.path = Path(path)
        self._reader = reader
        self._polydata = polydata
        self._locator = locator
        self._vtk_id_list_type = vtkIdList
        self._points = points
        self._offsets = offsets
        self._connectivity = connectivity

    @property
    def polygon_count(self) -> int:
        return len(self._offsets) - 1

    def candidate_polygon_ids(
        self, point_m: Sequence[float], radius_m: float
    ) -> tuple[int, ...]:
        point = _point3(point_m, "candidate point")
        if not math.isfinite(radius_m) or radius_m < 0.0:
            raise CpMappingError("candidate radius must be finite and non-negative")
        bounds = (
            float(point[0] - radius_m),
            float(point[0] + radius_m),
            float(point[1] - radius_m),
            float(point[1] + radius_m),
            float(point[2] - radius_m),
            float(point[2] + radius_m),
        )
        ids = self._vtk_id_list_type()
        self._locator.FindCellsWithinBounds(bounds, ids)
        return tuple(sorted({int(ids.GetId(index)) for index in range(ids.GetNumberOfIds())}))

    def polygon_vertices(self, raw_polygon_id: int) -> np.ndarray:
        if not isinstance(raw_polygon_id, int) or isinstance(raw_polygon_id, bool):
            raise CpMappingError("raw polygon ID must be an integer")
        if not 0 <= raw_polygon_id < self.polygon_count:
            raise CpMappingError("raw polygon ID is outside the native support")
        begin = int(self._offsets[raw_polygon_id])
        end = int(self._offsets[raw_polygon_id + 1])
        return np.asarray(
            self._points[self._connectivity[begin:end]], dtype=np.float64
        )


@dataclass(frozen=True)
class CpMappingRecord:
    """Rich candidate result with a lossless contract-evidence conversion."""

    case_id: str
    autocfd_probe_id: int
    nominal_point_m: tuple[float, float, float]
    drivaerml_component: str
    projection_mode: str
    cut_axis: str | None
    cut_value_m: float | None
    owner_review_status: str
    valid: bool
    reason: str
    mapped_point_m: tuple[float, float, float] | None
    raw_stl_triangle_id: int | None
    stl_unit_normal: tuple[float, float, float] | None
    nominal_displacement_m: float | None
    native_closest_point_m: tuple[float, float, float] | None
    raw_vtk_polygon_id: int | None
    native_polygon_unit_normal: tuple[float, float, float] | None
    bridge_distance_m: float | None
    bridge_abs_normal_dot: float | None
    native_bounds_candidate_count: int
    native_distance_pass_count: int
    native_normal_pass_count: int
    component_facet_count: int
    projection_candidate_count: int
    review_flags: tuple[str, ...]
    source_sha256: tuple[str, ...]

    def to_evidence(self) -> CpProbeMappingEvidence:
        """Return the subset consumed by the contract-wide coverage validator."""

        return CpProbeMappingEvidence(
            case_id=self.case_id,
            autocfd_probe_id=self.autocfd_probe_id,
            valid=self.valid,
            reason=self.reason,
            mapped_point_m=self.mapped_point_m,
            raw_stl_triangle_id=self.raw_stl_triangle_id,
            raw_vtk_polygon_id=self.raw_vtk_polygon_id,
            nominal_displacement_m=self.nominal_displacement_m,
            bridge_distance_m=self.bridge_distance_m,
            bridge_abs_normal_dot=self.bridge_abs_normal_dot,
            source_sha256=self.source_sha256,
        )


@dataclass(frozen=True)
class _NativeCandidate:
    raw_polygon_id: int
    closest_point: np.ndarray = field(compare=False)
    unit_normal: np.ndarray = field(compare=False)
    distance_m: float
    abs_normal_dot: float


def _mapping_from_projection(
    case_id: str,
    projection: StlProjectionResult,
    *,
    valid: bool,
    reason: str,
    source_sha256: tuple[str, ...],
    native_closest_point: np.ndarray | None = None,
    raw_polygon_id: int | None = None,
    native_unit_normal: np.ndarray | None = None,
    bridge_distance_m: float | None = None,
    bridge_abs_normal_dot: float | None = None,
    bounds_count: int = 0,
    distance_count: int = 0,
    normal_count: int = 0,
    extra_flags: tuple[str, ...] = (),
) -> CpMappingRecord:
    record = CpMappingRecord(
        case_id=case_id,
        autocfd_probe_id=projection.autocfd_probe_id,
        nominal_point_m=projection.nominal_point_m,
        drivaerml_component=projection.drivaerml_component,
        projection_mode=projection.projection_mode,
        cut_axis=projection.cut_axis,
        cut_value_m=projection.cut_value_m,
        owner_review_status=projection.owner_review_status,
        valid=valid,
        reason=reason,
        mapped_point_m=projection.mapped_point_m,
        raw_stl_triangle_id=projection.raw_stl_triangle_id,
        stl_unit_normal=projection.stl_unit_normal,
        nominal_displacement_m=projection.nominal_displacement_m,
        native_closest_point_m=(
            tuple(float(value) for value in native_closest_point)
            if native_closest_point is not None
            else None
        ),
        raw_vtk_polygon_id=raw_polygon_id,
        native_polygon_unit_normal=(
            tuple(float(value) for value in native_unit_normal)
            if native_unit_normal is not None
            else None
        ),
        bridge_distance_m=bridge_distance_m,
        bridge_abs_normal_dot=bridge_abs_normal_dot,
        native_bounds_candidate_count=bounds_count,
        native_distance_pass_count=distance_count,
        native_normal_pass_count=normal_count,
        component_facet_count=projection.component_facet_count,
        projection_candidate_count=projection.projection_candidate_count,
        review_flags=tuple(dict.fromkeys((*projection.review_flags, *extra_flags))),
        source_sha256=source_sha256,
    )
    # Validate compatibility immediately, including valid/invalid field rules
    # and SHA syntax, rather than deferring failures to a later aggregation.
    record.to_evidence()
    return record


def bridge_projection_to_native(
    case_id: str,
    projection: StlProjectionResult,
    locator: NativePolygonLocator,
    *,
    source_sha256: tuple[str, ...],
) -> CpMappingRecord:
    """Bridge one STL result to native polygons with all fixed gates and ties."""

    if not isinstance(case_id, str) or not case_id:
        raise CpMappingError("case_id must be non-empty")
    if not projection.valid:
        return _mapping_from_projection(
            case_id,
            projection,
            valid=False,
            reason=projection.reason,
            source_sha256=source_sha256,
        )
    assert projection.mapped_point_m is not None
    assert projection.stl_unit_normal is not None
    point = np.asarray(projection.mapped_point_m, dtype=np.float64)
    source_normal = np.asarray(projection.stl_unit_normal, dtype=np.float64)
    raw_candidates = tuple(
        sorted(
            set(
                int(value)
                for value in locator.candidate_polygon_ids(
                    projection.mapped_point_m, NATIVE_BRIDGE_MAX_DISTANCE_M
                )
            )
        )
    )
    if any(value < 0 or value >= locator.polygon_count for value in raw_candidates):
        raise CpMappingError("native locator returned an out-of-range raw polygon ID")
    if not raw_candidates:
        return _mapping_from_projection(
            case_id,
            projection,
            valid=False,
            reason="no_native_polygon_bounds_candidate_within_2mm",
            source_sha256=source_sha256,
        )

    finite_nondegenerate = 0
    distance_pass: list[_NativeCandidate] = []
    for raw_id in raw_candidates:
        vertices = locator.polygon_vertices(raw_id)
        native_normal = ordered_polygon_unit_normal(vertices)
        closest = closest_point_on_polygon(point, vertices)
        if native_normal is None or closest is None:
            continue
        finite_nondegenerate += 1
        distance = float(np.linalg.norm(closest - point))
        if not math.isfinite(distance) or distance > NATIVE_BRIDGE_MAX_DISTANCE_M:
            continue
        agreement = min(1.0, float(abs(np.dot(source_normal, native_normal))))
        if not math.isfinite(agreement):
            continue
        distance_pass.append(
            _NativeCandidate(raw_id, closest, native_normal, distance, agreement)
        )
    if finite_nondegenerate == 0:
        return _mapping_from_projection(
            case_id,
            projection,
            valid=False,
            reason="no_finite_nondegenerate_native_polygon",
            source_sha256=source_sha256,
            bounds_count=len(raw_candidates),
        )
    if not distance_pass:
        return _mapping_from_projection(
            case_id,
            projection,
            valid=False,
            reason="native_bridge_distance_exceeds_2mm",
            source_sha256=source_sha256,
            bounds_count=len(raw_candidates),
        )

    normal_pass = [
        candidate
        for candidate in distance_pass
        if candidate.abs_normal_dot >= NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT
    ]
    if not normal_pass:
        return _mapping_from_projection(
            case_id,
            projection,
            valid=False,
            reason="native_bridge_normal_agreement_below_cos30",
            source_sha256=source_sha256,
            bounds_count=len(raw_candidates),
            distance_count=len(distance_pass),
        )

    minimum_distance = min(candidate.distance_m for candidate in normal_pass)
    distance_tie = [
        candidate
        for candidate in normal_pass
        if candidate.distance_m <= minimum_distance + NATIVE_DISTANCE_TIE_M
    ]
    maximum_agreement = max(candidate.abs_normal_dot for candidate in distance_tie)
    normal_tie = [
        candidate
        for candidate in distance_tie
        if candidate.abs_normal_dot >= maximum_agreement - NATIVE_NORMAL_TIE
    ]
    selected = min(normal_tie, key=lambda candidate: candidate.raw_polygon_id)
    flags: list[str] = []
    if selected.distance_m > NATIVE_BRIDGE_DISTANCE_REVIEW_M:
        flags.append("native_bridge_distance_gt_0p5mm")
    return _mapping_from_projection(
        case_id,
        projection,
        valid=True,
        reason="",
        source_sha256=source_sha256,
        native_closest_point=selected.closest_point,
        raw_polygon_id=selected.raw_polygon_id,
        native_unit_normal=selected.unit_normal,
        bridge_distance_m=selected.distance_m,
        bridge_abs_normal_dot=selected.abs_normal_dot,
        bounds_count=len(raw_candidates),
        distance_count=len(distance_pass),
        normal_count=len(normal_pass),
        extra_flags=tuple(flags),
    )


def map_cp_probes_for_case(
    case_id: str,
    stl_path: str | Path,
    locator: NativePolygonLocator,
    probes: Sequence[CpProbeDefinition],
    rules: Sequence[CpComponentRule],
    *,
    source_sha256: tuple[str, ...],
    chunk_facets: int = 16_384,
) -> tuple[CpMappingRecord, ...]:
    """Produce one retained candidate mapping row per requested probe."""

    projections = project_probes_to_case_stl(
        stl_path, probes, rules, chunk_facets=chunk_facets
    )
    return tuple(
        bridge_projection_to_native(
            case_id, projection, locator, source_sha256=source_sha256
        )
        for projection in projections
    )
