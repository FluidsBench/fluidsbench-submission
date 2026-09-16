"""DrivAerML native-polygon geometry and field-derived force integration.

The implementation follows the OpenFOAM v2212
``primitiveMeshTools::makeFaceCentresAndAreas`` convention frozen in the
DrivAerML candidate contract.  It operates on NumPy views of the native VTP
polygon connectivity and never triangulates, reorders, or drops a polygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


U_INF_M_PER_S = 38.889
RHO_INF_KG_PER_M3 = 1.0
REFERENCE_AREA_M2 = 2.17
REFERENCE_LENGTH_M = 2.78618
CENTRE_OF_ROTATION_M = np.asarray([1.40009, 0.0, -0.3176], dtype=np.float64)


class DrivAerSurfaceForceError(ValueError):
    """Raised when native surface geometry or fields violate the contract."""


@dataclass(frozen=True)
class SurfaceGeometryChunk:
    """OpenFOAM face geometry for one half-open raw polygon-ID interval."""

    raw_id_start: int
    raw_id_stop: int
    centres_m: np.ndarray
    oriented_area_vectors_m2: np.ndarray
    areas_m2: np.ndarray


@dataclass(frozen=True)
class ForceMomentSums:
    """Dimensional force and moment sums before coefficient normalization."""

    raw_id_start: int
    raw_id_stop: int
    force_n: tuple[float, float, float]
    moment_n_m: tuple[float, float, float]


def _numeric_array(value: object, *, label: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise DrivAerSurfaceForceError(f"{label} must be numeric") from error
    if array.ndim != ndim:
        raise DrivAerSurfaceForceError(f"{label} must be {ndim}-dimensional")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise DrivAerSurfaceForceError(f"{label} must be numeric")
    return array


def validate_polygon_topology(
    points_m: object,
    connectivity: object,
    offsets: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate native polygon-only topology without changing its order."""

    points = _numeric_array(points_m, label="points_m", ndim=2)
    if points.shape[1:] != (3,) or points.shape[0] < 3:
        raise DrivAerSurfaceForceError("points_m must have shape [point, 3]")
    if not np.all(np.isfinite(points)):
        raise DrivAerSurfaceForceError("points_m must contain only finite values")
    points = np.asarray(points, dtype=np.float64)

    conn = _numeric_array(connectivity, label="connectivity", ndim=1)
    ends = _numeric_array(offsets, label="offsets", ndim=1)
    if conn.dtype.kind not in {"i", "u"} or ends.dtype.kind not in {"i", "u"}:
        raise DrivAerSurfaceForceError("connectivity and offsets must use integer dtypes")
    if ends.size < 2:
        raise DrivAerSurfaceForceError("offsets must contain n_polygons + 1 values")
    conn = np.asarray(conn, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if ends[0] != 0 or ends[-1] != conn.size or np.any(np.diff(ends) < 3):
        raise DrivAerSurfaceForceError(
            "offsets must start at zero, end at connectivity size, and define polygons"
        )
    if conn.size == 0 or np.any(conn < 0) or np.any(conn >= points.shape[0]):
        raise DrivAerSurfaceForceError("connectivity contains an invalid point ID")
    return points, conn, ends


def surface_geometry_chunk(
    points_m: object,
    connectivity: object,
    offsets: object,
    raw_id_start: int,
    raw_id_stop: int,
) -> SurfaceGeometryChunk:
    """Calculate oriented area vectors and OpenFOAM centres for a raw-ID range."""

    points, conn, ends = validate_polygon_topology(points_m, connectivity, offsets)
    return surface_geometry_chunk_validated(
        points, conn, ends, raw_id_start, raw_id_stop
    )


def surface_geometry_chunk_validated(
    points_m: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    raw_id_start: int,
    raw_id_stop: int,
) -> SurfaceGeometryChunk:
    """Calculate a chunk after one prior :func:`validate_polygon_topology` call.

    This entry point avoids rescanning multi-million-entity topology for every
    bounded-memory chunk.  Callers must pass the exact arrays returned by
    :func:`validate_polygon_topology`.
    """

    points = points_m
    conn = connectivity
    ends = offsets
    if (
        not isinstance(points, np.ndarray)
        or points.dtype != np.dtype(np.float64)
        or points.ndim != 2
        or points.shape[1:] != (3,)
        or not isinstance(conn, np.ndarray)
        or conn.dtype != np.dtype(np.int64)
        or conn.ndim != 1
        or not isinstance(ends, np.ndarray)
        or ends.dtype != np.dtype(np.int64)
        or ends.ndim != 1
    ):
        raise DrivAerSurfaceForceError(
            "validated geometry arrays must come from validate_polygon_topology"
        )
    n_polygons = ends.size - 1
    if (
        not isinstance(raw_id_start, int)
        or isinstance(raw_id_start, bool)
        or not isinstance(raw_id_stop, int)
        or isinstance(raw_id_stop, bool)
        or raw_id_start < 0
        or raw_id_stop <= raw_id_start
        or raw_id_stop > n_polygons
    ):
        raise DrivAerSurfaceForceError("invalid half-open raw polygon-ID interval")

    starts = ends[raw_id_start:raw_id_stop]
    counts = ends[raw_id_start + 1 : raw_id_stop + 1] - starts
    size = raw_id_stop - raw_id_start
    maximum_arity = int(counts.max())

    vertex_mean = np.zeros((size, 3), dtype=np.float64)
    for local_index in range(maximum_arity):
        mask = counts > local_index
        vertex_mean[mask] += points[conn[starts[mask] + local_index]]
    vertex_mean /= counts[:, None]

    twice_area = np.zeros((size, 3), dtype=np.float64)
    centre_numerator = np.zeros((size, 3), dtype=np.float64)
    centre_weight = np.zeros(size, dtype=np.float64)
    for local_index in range(maximum_arity):
        mask = counts > local_index
        local_starts = starts[mask]
        next_index = (local_index + 1) % counts[mask]
        point = points[conn[local_starts + local_index]]
        point_next = points[conn[local_starts + next_index]]
        relative = point - vertex_mean[mask]
        relative_next = point_next - vertex_mean[mask]
        twice_area[mask] += np.cross(relative, relative_next)

        # OpenFOAM makeFaceCentresAndAreas uses magnitudes of the triangles
        # formed by each edge and the arithmetic vertex mean.
        triangle_normal = np.cross(point_next - point, vertex_mean[mask] - point)
        triangle_twice_area = np.linalg.norm(triangle_normal, axis=1)
        centre_weight[mask] += triangle_twice_area
        centre_numerator[mask] += triangle_twice_area[:, None] * (
            point + point_next + vertex_mean[mask]
        )

    oriented_area = 0.5 * twice_area
    areas = np.linalg.norm(oriented_area, axis=1)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise DrivAerSurfaceForceError(
            "every native polygon must have one positive finite area"
        )
    if not np.all(np.isfinite(centre_weight)) or np.any(centre_weight <= 0.0):
        raise DrivAerSurfaceForceError("OpenFOAM face-centre weight is invalid")

    centres = centre_numerator / (3.0 * centre_weight[:, None])
    triangle_mask = counts == 3
    centres[triangle_mask] = vertex_mean[triangle_mask]
    if not np.all(np.isfinite(centres)) or not np.all(np.isfinite(oriented_area)):
        raise DrivAerSurfaceForceError("calculated face geometry is non-finite")

    return SurfaceGeometryChunk(
        raw_id_start=raw_id_start,
        raw_id_stop=raw_id_stop,
        centres_m=centres,
        oriented_area_vectors_m2=oriented_area,
        areas_m2=areas,
    )


def audit_fixed_surface_areas(
    calculated_areas_m2: object,
    published_areas_m2: object,
    *,
    rtol: float = 6.0e-8,
    atol_m2: float = 0.0,
) -> dict[str, float | int]:
    """Audit fixed public float32 areas without regenerating or redefining them."""

    calculated = np.asarray(calculated_areas_m2, dtype=np.float64)
    published_input = np.asarray(published_areas_m2)
    if calculated.ndim != 1 or published_input.ndim != 1:
        raise DrivAerSurfaceForceError("surface area arrays must be one-dimensional")
    if calculated.shape != published_input.shape:
        raise DrivAerSurfaceForceError("surface area tuple count differs from the VTP")
    if published_input.dtype != np.dtype("<f4"):
        raise DrivAerSurfaceForceError(
            "published surface areas must use little-endian float32"
        )
    published = published_input.astype(np.float64)
    if (
        not np.all(np.isfinite(calculated))
        or not np.all(np.isfinite(published))
        or np.any(calculated <= 0.0)
        or np.any(published <= 0.0)
    ):
        raise DrivAerSurfaceForceError("surface areas must be positive and finite")
    difference = np.abs(calculated - published)
    tolerance = atol_m2 + rtol * np.abs(calculated)
    if np.any(difference > tolerance):
        index = int(np.argmax(difference - tolerance))
        raise DrivAerSurfaceForceError(
            f"published surface area differs from native polygon {index}"
        )
    return {
        "entity_count": int(calculated.size),
        "calculated_sum_m2": float(np.sum(calculated, dtype=np.float64)),
        "published_sum_m2": float(np.sum(published, dtype=np.float64)),
        "maximum_absolute_difference_m2": float(difference.max(initial=0.0)),
        "maximum_relative_difference": float(
            np.max(difference / calculated, initial=0.0)
        ),
    }


def force_moment_chunk(
    geometry: SurfaceGeometryChunk,
    pressure_m2_per_s2: object,
    wall_shear_m2_per_s2: object,
    *,
    centre_of_rotation_m: object = CENTRE_OF_ROTATION_M,
    rho_inf_kg_per_m3: float = RHO_INF_KG_PER_M3,
) -> ForceMomentSums:
    """Accumulate pressure-plus-wall-shear body force for one geometry chunk."""

    count = geometry.raw_id_stop - geometry.raw_id_start
    pressure = np.asarray(pressure_m2_per_s2, dtype=np.float64)
    shear = np.asarray(wall_shear_m2_per_s2, dtype=np.float64)
    centre = np.asarray(centre_of_rotation_m, dtype=np.float64)
    if pressure.shape != (count,) or shear.shape != (count, 3):
        raise DrivAerSurfaceForceError(
            "pressure/shear tuple and component counts must match the geometry chunk"
        )
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise DrivAerSurfaceForceError("centre_of_rotation_m must be a finite 3-vector")
    if (
        not np.all(np.isfinite(pressure))
        or not np.all(np.isfinite(shear))
        or not math.isfinite(rho_inf_kg_per_m3)
        or rho_inf_kg_per_m3 <= 0.0
    ):
        raise DrivAerSurfaceForceError("surface fields and density must be finite")

    face_force = rho_inf_kg_per_m3 * (
        pressure[:, None] * geometry.oriented_area_vectors_m2
        - shear * geometry.areas_m2[:, None]
    )
    face_moment = np.cross(geometry.centres_m - centre, face_force)
    force = np.sum(face_force, axis=0, dtype=np.longdouble)
    moment = np.sum(face_moment, axis=0, dtype=np.longdouble)
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(moment)):
        raise DrivAerSurfaceForceError("force integration became non-finite")
    return ForceMomentSums(
        raw_id_start=geometry.raw_id_start,
        raw_id_stop=geometry.raw_id_stop,
        force_n=tuple(float(value) for value in force),
        moment_n_m=tuple(float(value) for value in moment),
    )


def _validate_complete_force_chunks(
    chunks: Iterable[ForceMomentSums], expected_entity_count: int
) -> list[ForceMomentSums]:
    ordered = sorted(chunks, key=lambda item: item.raw_id_start)
    if not ordered:
        raise DrivAerSurfaceForceError("at least one force chunk is required")
    cursor = 0
    for chunk in ordered:
        if chunk.raw_id_start != cursor or chunk.raw_id_stop <= chunk.raw_id_start:
            raise DrivAerSurfaceForceError(
                "force chunks must cover every raw polygon exactly once"
            )
        cursor = chunk.raw_id_stop
    if cursor != expected_entity_count:
        raise DrivAerSurfaceForceError(
            "force chunks must cover every raw polygon exactly once"
        )
    return ordered


def finalize_force_coefficients(
    chunks: Iterable[ForceMomentSums],
    *,
    expected_entity_count: int,
    u_inf_m_per_s: float = U_INF_M_PER_S,
    rho_inf_kg_per_m3: float = RHO_INF_KG_PER_M3,
    reference_area_m2: float = REFERENCE_AREA_M2,
    reference_length_m: float = REFERENCE_LENGTH_M,
) -> dict[str, float | int | list[float]]:
    """Merge complete raw-ID coverage and derive the frozen coefficients."""

    if not isinstance(expected_entity_count, int) or expected_entity_count < 1:
        raise DrivAerSurfaceForceError("expected_entity_count must be positive")
    constants = (
        u_inf_m_per_s,
        rho_inf_kg_per_m3,
        reference_area_m2,
        reference_length_m,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in constants):
        raise DrivAerSurfaceForceError("force normalization constants must be positive")
    ordered = _validate_complete_force_chunks(chunks, expected_entity_count)
    force = np.asarray(
        [math.fsum(chunk.force_n[axis] for chunk in ordered) for axis in range(3)],
        dtype=np.float64,
    )
    moment = np.asarray(
        [math.fsum(chunk.moment_n_m[axis] for chunk in ordered) for axis in range(3)],
        dtype=np.float64,
    )
    q_area = 0.5 * rho_inf_kg_per_m3 * u_inf_m_per_s**2 * reference_area_m2
    cd = float(force[0] / q_area)
    cl = float(force[2] / q_area)
    cs = float(force[1] / q_area)
    cm_pitch = float(moment[1] / (q_area * reference_length_m))
    clf = cl / 2.0 + cm_pitch
    clr = cl / 2.0 - cm_pitch
    return {
        "entity_count": expected_entity_count,
        "force_n": force.tolist(),
        "moment_about_forces_cor_n_m": moment.tolist(),
        "Cd": cd,
        "Cl": cl,
        "Cs": cs,
        "CmPitch": cm_pitch,
        "Clf": clf,
        "Clr": clr,
        "lift_closure_abs": abs(cl - (clf + clr)),
    }


def iter_raw_id_ranges(entity_count: int, chunk_entities: int) -> Iterator[tuple[int, int]]:
    """Yield deterministic ascending half-open raw-ID ranges."""

    if entity_count < 1 or chunk_entities < 1:
        raise DrivAerSurfaceForceError("entity_count and chunk_entities must be positive")
    for start in range(0, entity_count, chunk_entities):
        yield start, min(entity_count, start + chunk_entities)
