"""Candidate native-cell assignments for DrivAerML velocity diagnostics.

This module implements the proposed AutoCFD5 containing-cell semantics without
changing the native ``vtkUnstructuredGrid``.  Raw cell IDs are the zero-based
indices accepted by ``vtkUnstructuredGrid.GetCell``.  A
``vtkCellTreeLocator`` supplies a broad phase using an explicitly expanded
point bound and exact native-cell axis-aligned-bounds intersection.  Fixed
native cell types are evaluated with ``vtkGenericCell.EvaluatePosition``.
Native ``vtkPolyhedron`` cells instead use a deterministic triangulated-surface
kernel: Euclidean point-to-triangle distance supplies the absolute closure and
a signed solid-angle winding sum supplies containment.  This avoids
``vtkPolyhedron::IsInside`` in VTK 9.5.2, whose private random sequence is seeded
from wall-clock time.  All closure candidates are retained for the count and
the smallest raw cell ID is selected.

The implementation remains a *candidate*: its settings and evidence are
designed for the required 0.5/1/2 micrometre replay, but this module does not
claim that the all-case replay, owner validity mask, or scientific approval is
complete.  VTK is optional at import time. Candidate evidence generation
requires exactly Python 3.12.13, NumPy 2.2.6, and VTK 9.5.2; low-level static
settings remain inspectable without that runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .autocfd5 import (
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    AutoCFD5Definition,
    VelocityCellAssignmentEvidence,
    VelocityLineDefinition,
    VelocitySampleDefinition,
)

try:  # VTK is an optional, separately pinned evaluator dependency.
    import vtk  # type: ignore[import-not-found]
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import-not-found]
except ImportError as error:  # pragma: no cover - exercised in the base environment.
    vtk = None
    vtk_to_numpy = None
    _VTK_IMPORT_ERROR: ImportError | None = error
else:
    _VTK_IMPORT_ERROR = None


REQUIRED_PYTHON_VERSION = "3.12.13"
REQUIRED_NUMPY_VERSION = "2.2.6"
REQUIRED_VTK_VERSION = "9.5.2"
REQUIRED_VTK_SOURCE_VERSION = "vtk version 9.5.2"
KERNEL_ID = "drivaerml-native-containing-cell-candidate-v7"
REPLAY_SCHEMA = "drivaerml-velocity-cell-tolerance-replay-candidate-v1"
DEFAULT_VALIDATION_CHUNK_SIZE = 1_000_000
TOLERANCE_REPLAY_M = (0.5e-6, 1.0e-6, 2.0e-6)
QUERY_CACHE_KEY_ID = "ieee754-binary64-big-endian-exact-xyz-tolerance-v1"
POLYHEDRON_CLOSURE_ID = "triangulated-signed-solid-angle-v1"
POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE = 1.0e-3
POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES = 8_192
POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES = 131_072
POLYHEDRON_UPSTREAM_BASIS_COMMIT = (
    "33c54a2c6b829031e4f013bf5cb7e45daef299d9"
)
_QUERY_CACHE_KEY = struct.Struct(">dddd")
_CACHE_MISS = object()

# Integer values are stable VTK public constants and keep static settings
# inspectable even when the optional VTK wheel is absent.
SUPPORTED_CELL_TYPES = (
    (10, "VTK_TETRA"),
    (12, "VTK_HEXAHEDRON"),
    (13, "VTK_WEDGE"),
    (14, "VTK_PYRAMID"),
    (42, "VTK_POLYHEDRON"),
)
_SUPPORTED_CELL_TYPE_IDS = frozenset(code for code, _ in SUPPORTED_CELL_TYPES)
_FIXED_POINT_COUNTS = {10: 4, 12: 8, 13: 6, 14: 5}
OWNER_INVALID_REASONS = frozenset(
    {"inside_morphed_solid", "outside_released_fluid_domain"}
)
NO_CLOSURE_CELL_REASON = "no_native_cell_within_closure_tolerance"
CELL_EVALUATION_FAILURE_REASON_PREFIX = (
    "native_cell_closure_evaluation_failed_for_broad_phase_cells:"
)
# Compatibility alias for callers of candidate-v6 and earlier Python APIs.  New
# candidate-v7 artifacts use the generalized value above because failures can
# arise from fixed-cell EvaluatePosition or deterministic polyhedron geometry.
EVALUATE_POSITION_FAILURE_REASON_PREFIX = CELL_EVALUATION_FAILURE_REASON_PREFIX
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class VelocityAssignmentError(ValueError):
    """Raised when the candidate geometry kernel cannot produce safe evidence."""


def _vtk_version() -> str | None:
    if vtk is None:
        return None
    return str(vtk.vtkVersion.GetVTKVersion())


def vtk_available() -> bool:
    """Return whether the exact optional VTK dependency is importable."""

    return _vtk_version() == REQUIRED_VTK_VERSION


def _require_vtk() -> None:
    if vtk is None or vtk_to_numpy is None:
        detail = f": {_VTK_IMPORT_ERROR}" if _VTK_IMPORT_ERROR is not None else ""
        raise VelocityAssignmentError(
            "DrivAerML velocity assignments require optional "
            f"VTK {REQUIRED_VTK_VERSION}{detail}"
        )
    actual = _vtk_version()
    if actual != REQUIRED_VTK_VERSION:
        raise VelocityAssignmentError(
            f"DrivAerML velocity assignments require VTK {REQUIRED_VTK_VERSION}, "
            f"got {actual}"
        )
    actual_source = str(vtk.vtkVersion.GetVTKSourceVersion())
    if actual_source != REQUIRED_VTK_SOURCE_VERSION:
        raise VelocityAssignmentError(
            "DrivAerML velocity assignments require VTK source identity "
            f"{REQUIRED_VTK_SOURCE_VERSION!r}, got {actual_source!r}"
        )


def _require_frozen_runtime() -> None:
    """Reject candidate evidence generation outside the aggregate runtime."""

    actual_python = platform.python_version()
    if actual_python != REQUIRED_PYTHON_VERSION:
        raise VelocityAssignmentError(
            "DrivAerML velocity assignment generation requires Python "
            f"{REQUIRED_PYTHON_VERSION}, got {actual_python}"
        )
    actual_numpy = np.__version__
    if actual_numpy != REQUIRED_NUMPY_VERSION:
        raise VelocityAssignmentError(
            "DrivAerML velocity assignment generation requires NumPy "
            f"{REQUIRED_NUMPY_VERSION}, got {actual_numpy}"
        )
    _require_vtk()


def _positive_finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VelocityAssignmentError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise VelocityAssignmentError(f"{label} must be finite and positive")
    return result


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise VelocityAssignmentError(f"{label} must be a positive integer")
    return value


def _source_hashes(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise VelocityAssignmentError(
            "source_sha256 must be a non-empty tuple of lowercase SHA-256 digests"
        )
    if any(
        not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
        for item in value
    ):
        raise VelocityAssignmentError(
            "source_sha256 must contain lowercase SHA-256 digests"
        )
    return value


def candidate_kernel_settings() -> dict[str, object]:
    """Return the complete proposed geometry and tie-break settings."""

    return {
        "kernel_id": KERNEL_ID,
        "status": "candidate_pending_all_case_replay_and_owner_approval",
        "required_dependency": {"vtk": REQUIRED_VTK_VERSION},
        "input": {
            "vtk_data_object": "vtkUnstructuredGrid",
            "association": "native_CellData",
            "mesh_transform": "none",
            "raw_vtk_cell_id": "zero_based_native_GetCell_index",
            "mutation_policy": (
                "reject_changed_native_grid_or_geometry_array_VTK_MTime_"
                "between_kernel_construction_and_assignment"
            ),
            "supported_cell_types": [
                {"vtk_type_id": code, "vtk_name": name}
                for code, name in SUPPORTED_CELL_TYPES
            ],
        },
        "candidate_discovery": {
            "class": "vtkCellTreeLocator",
            "method": "FindCellsWithinBounds",
            "query": (
                "exact_native_cell_axis_aligned_bounds_intersection_with_"
                "point_bounds_expanded_by_absolute_tolerance"
            ),
            "query_bounds_order": ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"],
            "deduplicate_raw_cell_ids": True,
            "settings": {
                "automatic": True,
                "number_of_cells_per_node": 8,
                "number_of_build_buckets": 6,
                "cache_cell_bounds": True,
            },
        },
        "closure": {
            "cell_wrapper": "vtkGenericCell",
            "cell_wrapper_lifecycle": {
                "fixed_native_cell_types": (
                    "fresh_vtkGenericCell_per_raw_cell_evaluation"
                ),
                "native_vtk_polyhedron": (
                    "fresh_vtkGenericCell_on_first_uncached_raw_cell_visit_only"
                ),
            },
            "dispatch": "native_vtk_cell_type",
            "fixed_native_cell_types": {
                "vtk_type_ids": [10, 12, 13, 14],
                "method": "vtkGenericCell.EvaluatePosition",
                "acceptance": (
                    "inside_return_code_1_with_dist2_exactly_zero_or_"
                    "outside_return_code_0_with_sqrt(dist2)<=absolute_tolerance_m"
                ),
                "mutable_output_initialization": (
                    "quiet_NaN_for_all_floating_outputs_and_zero_for_sub_id_"
                    "before_every_cell"
                ),
                "failure_conditions": [
                    "return_code_minus_1",
                    "non_finite_numeric_output",
                    "negative_squared_distance",
                    "inside_return_with_nonzero_squared_distance",
                ],
            },
            "native_vtk_polyhedron": {
                "vtk_type_id": 42,
                "closure_id": POLYHEDRON_CLOSURE_ID,
                "randomness": "none",
                "vtk_9_5_2_IsInside_called": False,
                "surface": (
                    "evaluator_only_native_face_tessellation_with_native_grid_"
                    "and_raw_cell_order_unchanged"
                ),
                "face_order": "native_GetFace_index",
                "triangulation": (
                    "native_triangles_unchanged_otherwise_"
                    "vtkPolygon.TriangulateLocalIds_index_0"
                ),
                "topology_validation": [
                    "at_least_four_faces",
                    "at_least_three_distinct_points_per_face",
                    "all_native_cell_points_used",
                    "unique_faces",
                    "each_undirected_edge_occurs_exactly_twice_with_opposite_directions",
                    "single_connected_face_shell",
                    "finite_coordinates",
                    "complete_non_degenerate_orientation_preserving_triangulation",
                    (
                        "emitted_triangles_cover_each_oriented_face_boundary_"
                        "edge_once_and_pair_each_internal_edge_oppositely"
                    ),
                ],
                "boundary_distance": (
                    "minimum_Euclidean_point_to_emitted_triangle_distance_in_metres"
                ),
                "containment": "absolute_signed_solid_angle_winding_sum",
                "solid_angle_formula": "Van_Oosterom_Strackee",
                "summation": "math.fsum_in_native_face_then_triangle_order",
                "inside_target_steradian": 4.0 * math.pi,
                "outside_target_steradian": 0.0,
                "classification_absolute_tolerance_steradian": (
                    POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
                ),
                "classification_tolerance_status": (
                    "candidate_scientific_choice_pending_owner_approval"
                ),
                "ambiguous_action": "fail_closed_and_record_raw_cell_id",
                "upstream_algorithm_basis": {
                    "repository": "https://gitlab.kitware.com/vtk/vtk",
                    "commit": POLYHEDRON_UPSTREAM_BASIS_COMMIT,
                    "file": "Common/DataModel/vtkPolyhedron.cxx",
                    "function": "vtkPolyhedron::IsInside",
                },
                "deviations_from_upstream_basis": [
                    "Python_math_atan2_and_math_fsum_in_pinned_triangle_order",
                    "fail_closed_topology_validation_and_native_face_triangulation",
                    "absolute_1e-6_m_Euclidean_boundary_closure_before_winding",
                    "ambiguous_topology_distance_or_winding_state_fails_closed",
                ],
                "work_bound": {
                    "first_uncached_raw_cell_visit": (
                        "one_native_face_topology_validation_and_triangulation"
                    ),
                    "each_query": (
                        "ordered_triangle_distances_until_boundary_acceptance_"
                        "otherwise_all_triangle_distances_and_solid_angles"
                    ),
                },
                "geometry_cache": {
                    "key": "raw_vtk_cell_id",
                    "value": (
                        "immutable_Python_binary64_emitted_triangles_or_"
                        "None_for_fail_closed_preparation"
                    ),
                    "vtk_objects_cached": False,
                    "population": "lazy_on_first_broad_phase_visit",
                    "eviction": "deterministic_least_recently_used",
                    "maximum_entries": POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES,
                    "maximum_emitted_triangles": (
                        POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES
                    ),
                    "entry_bound": (
                        "minimum_of_maximum_entries_distinct_broad_phase_native_"
                        "polyhedron_raw_cell_ids_visited_during_kernel_lifetime_"
                        "and_native_cell_count"
                    ),
                    "generator_practical_bound": (
                        "distinct_broad_phase_native_polyhedron_raw_cell_ids_"
                        "visited_by_the_fixed_four_resolution_sample_set"
                    ),
                    "oversized_single_entry_action": (
                        "evaluate_normally_without_caching_when_emitted_triangle_"
                        "count_exceeds_maximum_emitted_triangles"
                    ),
                },
            },
            "distance": "Euclidean_closest_point_distance_in_metres",
            "failure_action": (
                "retain_the_sample_as_invalid_and_record_sorted_failed_raw_cell_ids"
            ),
        },
        "selection": {
            "enumeration": "all_distinct_cells_passing_closure_predicate",
            "candidate_count": "number_of_distinct_cells_passing_closure_predicate",
            "tie_break": "smallest_raw_vtk_cell_id",
        },
        "invalidity": {
            "owner_reasons": sorted(OWNER_INVALID_REASONS),
            "no_closure_cell_reason": NO_CLOSURE_CELL_REASON,
            "snapping": False,
            "extrapolation": False,
            "owner_reason_overrides_geometric_selection": True,
        },
        "primary_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
        "required_tolerance_replay_m": list(TOLERANCE_REPLAY_M),
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def candidate_kernel_receipt() -> dict[str, object]:
    """Return runtime versions plus a deterministic hash of static settings."""

    _require_frozen_runtime()
    settings = candidate_kernel_settings()
    return {
        "kernel_id": KERNEL_ID,
        "settings_sha256": hashlib.sha256(_canonical_json(settings)).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "vtk": _vtk_version(),
            "vtk_source": str(vtk.vtkVersion.GetVTKSourceVersion()),
        },
        "settings": settings,
    }


def generate_velocity_line_samples(
    line: VelocityLineDefinition,
    nominal_spacing_m: float,
) -> tuple[VelocitySampleDefinition, ...]:
    """Generate an endpoint-inclusive equal-arc grid at nominal spacing.

    The nearest integer number of segments is used and the actual spacing is
    therefore ``endpoint distance / segment_count``.  This exactly reproduces
    the proposed 1, 2, 5, and 10 mm grids, including the three oblique lines.
    """

    if not isinstance(line, VelocityLineDefinition):
        raise VelocityAssignmentError("line must be a VelocityLineDefinition")
    spacing = _positive_finite(nominal_spacing_m, "nominal_spacing_m")
    length = math.dist(line.start_m, line.end_m)
    if not math.isfinite(length) or length <= 0.0:
        raise VelocityAssignmentError("velocity line endpoint distance must be positive")
    segment_count = max(1, int(math.floor(length / spacing + 0.5)))
    point_count = segment_count + 1
    result: list[VelocitySampleDefinition] = []
    for sample_index in range(point_count):
        fraction = sample_index / segment_count
        if sample_index == 0:
            point = line.start_m
        elif sample_index == segment_count:
            point = line.end_m
        else:
            point = tuple(
                start + fraction * (end - start)
                for start, end in zip(line.start_m, line.end_m)
            )
        result.append(
            VelocitySampleDefinition(
                profile_id=line.profile_id,
                sample_index=sample_index,
                point_count=point_count,
                line_fraction=fraction,
                distance_m=fraction * length,
                point_m=point,
            )
        )
    return tuple(result)


def generate_definition_velocity_samples(
    definition: AutoCFD5Definition,
    nominal_spacing_m: float,
) -> tuple[VelocitySampleDefinition, ...]:
    """Generate a complete arbitrary-spacing grid in registry line order."""

    if not isinstance(definition, AutoCFD5Definition):
        raise VelocityAssignmentError("definition must be an AutoCFD5Definition")
    spacing = _positive_finite(nominal_spacing_m, "nominal_spacing_m")
    return tuple(
        sample
        for line in definition.velocity_lines
        for sample in generate_velocity_line_samples(line, spacing)
    )


def _validate_native_grid(grid: Any, *, chunk_size: int) -> tuple[int, int]:
    _require_vtk()
    chunk_size = _positive_integer(chunk_size, "validation_chunk_size")
    if grid is None or not hasattr(grid, "IsA") or not grid.IsA("vtkUnstructuredGrid"):
        raise VelocityAssignmentError("input must be a native vtkUnstructuredGrid")
    cell_count = int(grid.GetNumberOfCells())
    point_count = int(grid.GetNumberOfPoints())
    if cell_count < 1:
        raise VelocityAssignmentError("native volume geometry contains no cells")
    if point_count < 1 or grid.GetPoints() is None:
        raise VelocityAssignmentError("native volume geometry contains no points")

    point_data = grid.GetPoints().GetData()
    if point_data is None:
        raise VelocityAssignmentError("native point coordinates are unavailable")
    coordinates = np.asarray(vtk_to_numpy(point_data))
    if coordinates.shape != (point_count, 3):
        raise VelocityAssignmentError("native point coordinates must have shape [N, 3]")
    for start in range(0, point_count, chunk_size):
        if not np.all(np.isfinite(coordinates[start : start + chunk_size])):
            raise VelocityAssignmentError("native geometry contains non-finite coordinates")

    type_array = grid.GetCellTypesArray()
    if type_array is None:
        raise VelocityAssignmentError("native cell-type array is unavailable")
    cell_types = np.asarray(vtk_to_numpy(type_array))
    if cell_types.shape != (cell_count,):
        raise VelocityAssignmentError("native cell-type array has the wrong length")
    for start in range(0, cell_count, chunk_size):
        chunk = cell_types[start : start + chunk_size]
        supported = np.isin(chunk, tuple(_SUPPORTED_CELL_TYPE_IDS))
        if not np.all(supported):
            offset = int(np.flatnonzero(~supported)[0])
            raw_id = start + offset
            raise VelocityAssignmentError(
                "unsupported native cell type "
                f"{int(chunk[offset])} at raw VTK cell ID {raw_id}"
            )

    cells = grid.GetCells()
    if cells is None:
        raise VelocityAssignmentError("native cell connectivity is unavailable")
    offsets_array = cells.GetOffsetsArray()
    connectivity_array = cells.GetConnectivityArray()
    if offsets_array is None or connectivity_array is None:
        raise VelocityAssignmentError("native cell connectivity arrays are unavailable")
    offsets = np.asarray(vtk_to_numpy(offsets_array))
    connectivity = np.asarray(vtk_to_numpy(connectivity_array))
    if (
        offsets.shape != (cell_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != connectivity.size
    ):
        raise VelocityAssignmentError("native cell offsets are malformed")
    point_counts = np.diff(offsets)
    if np.any(point_counts < 4):
        raw_id = int(np.flatnonzero(point_counts < 4)[0])
        raise VelocityAssignmentError(
            f"native cell {raw_id} has fewer than four connectivity points"
        )
    for cell_type, expected_count in _FIXED_POINT_COUNTS.items():
        wrong = (cell_types == cell_type) & (point_counts != expected_count)
        if np.any(wrong):
            raw_id = int(np.flatnonzero(wrong)[0])
            raise VelocityAssignmentError(
                f"native cell {raw_id} has the wrong point count for type {cell_type}"
            )
    if np.any((cell_types == 42) & (point_counts < 4)):
        raw_id = int(np.flatnonzero((cell_types == 42) & (point_counts < 4))[0])
        raise VelocityAssignmentError(f"native polyhedron {raw_id} is malformed")
    for start in range(0, connectivity.size, chunk_size):
        chunk = connectivity[start : start + chunk_size]
        invalid = (chunk < 0) | (chunk >= point_count)
        if np.any(invalid):
            raise VelocityAssignmentError("native cell connectivity has an invalid point ID")
    return point_count, cell_count


def _native_grid_mtime_binding(grid: Any) -> tuple[int | None, ...]:
    """Bind VTK modification times for geometry objects used by the kernel."""

    points = grid.GetPoints()
    cells = grid.GetCells()
    polyhedron_faces = grid.GetPolyhedronFaces()
    polyhedron_face_locations = grid.GetPolyhedronFaceLocations()
    objects = [
        grid,
        points,
        points.GetData() if points is not None else None,
        grid.GetCellTypesArray(),
        cells,
        cells.GetOffsetsArray() if cells is not None else None,
        cells.GetConnectivityArray() if cells is not None else None,
        polyhedron_faces,
        (
            polyhedron_faces.GetOffsetsArray()
            if polyhedron_faces is not None
            else None
        ),
        (
            polyhedron_faces.GetConnectivityArray()
            if polyhedron_faces is not None
            else None
        ),
        polyhedron_face_locations,
        (
            polyhedron_face_locations.GetOffsetsArray()
            if polyhedron_face_locations is not None
            else None
        ),
        (
            polyhedron_face_locations.GetConnectivityArray()
            if polyhedron_face_locations is not None
            else None
        ),
    ]
    return tuple(
        None if item is None else int(item.GetMTime()) for item in objects
    )


def _sample_key(sample: VelocitySampleDefinition) -> tuple[str, int]:
    return sample.profile_id, sample.sample_index


def _validated_samples(
    samples: Sequence[VelocitySampleDefinition],
) -> tuple[VelocitySampleDefinition, ...]:
    result = tuple(samples)
    if not result:
        raise VelocityAssignmentError("velocity samples cannot be empty")
    if any(not isinstance(sample, VelocitySampleDefinition) for sample in result):
        raise VelocityAssignmentError(
            "velocity samples must contain VelocitySampleDefinition records"
        )
    keys = [_sample_key(sample) for sample in result]
    if len(keys) != len(set(keys)):
        raise VelocityAssignmentError("velocity samples contain duplicate line/sample IDs")
    return result


def _validated_invalid_reasons(
    samples: tuple[VelocitySampleDefinition, ...],
    invalid_reasons: Mapping[tuple[str, int], str] | None,
) -> dict[tuple[str, int], str]:
    if invalid_reasons is None:
        return {}
    if not isinstance(invalid_reasons, Mapping):
        raise VelocityAssignmentError("invalid_reasons must be a mapping")
    result: dict[tuple[str, int], str] = {}
    sample_keys = {_sample_key(sample) for sample in samples}
    for key, reason in invalid_reasons.items():
        if key not in sample_keys:
            raise VelocityAssignmentError(
                f"owner invalidity mask contains an unexpected sample key: {key!r}"
            )
        if reason not in OWNER_INVALID_REASONS:
            raise VelocityAssignmentError(
                f"unsupported owner invalidity reason for {key!r}: {reason!r}"
            )
        result[key] = reason
    return result


_Point3 = tuple[float, float, float]
_Triangle3 = tuple[_Point3, _Point3, _Point3]


@dataclass(frozen=True)
class _PolyhedronClosureEvaluation:
    closure: bool | None
    category: str
    accepted_classification_margin_steradian: float | None = None


def _subtract(a: _Point3, b: _Point3) -> _Point3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _dot(a: _Point3, b: _Point3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: _Point3, b: _Point3) -> _Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _point_segment_distance_squared(x: _Point3, a: _Point3, b: _Point3) -> float:
    ab = _subtract(b, a)
    ax = _subtract(x, a)
    denominator = _dot(ab, ab)
    if denominator <= 0.0:
        return _dot(ax, ax)
    fraction = max(0.0, min(1.0, _dot(ax, ab) / denominator))
    closest = (
        a[0] + fraction * ab[0],
        a[1] + fraction * ab[1],
        a[2] + fraction * ab[2],
    )
    delta = _subtract(x, closest)
    return _dot(delta, delta)


def _point_triangle_distance_squared(x: _Point3, triangle: _Triangle3) -> float:
    """Return the deterministic Euclidean distance to one emitted triangle."""

    p1, p2, p3 = triangle
    e1 = _subtract(p2, p1)
    e2 = _subtract(p3, p1)
    normal = _cross(e1, e2)
    normal_squared = _dot(normal, normal)
    if normal_squared <= 0.0:
        return min(
            _point_segment_distance_squared(x, p1, p2),
            _point_segment_distance_squared(x, p2, p3),
            _point_segment_distance_squared(x, p3, p1),
        )

    inverse_normal_length = 1.0 / math.sqrt(normal_squared)
    p1_to_x = _subtract(x, p1)
    plane_distance = _dot(p1_to_x, normal) * inverse_normal_length
    projected = (
        x[0] - plane_distance * normal[0] * inverse_normal_length,
        x[1] - plane_distance * normal[1] * inverse_normal_length,
        x[2] - plane_distance * normal[2] * inverse_normal_length,
    )

    v0 = _subtract(p3, p1)
    v1 = _subtract(p2, p1)
    v2 = _subtract(projected, p1)
    dot00 = _dot(v0, v0)
    dot01 = _dot(v0, v1)
    dot02 = _dot(v0, v2)
    dot11 = _dot(v1, v1)
    dot12 = _dot(v1, v2)
    denominator = dot00 * dot11 - dot01 * dot01
    if denominator > 0.0:
        inverse_denominator = 1.0 / denominator
        u = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
        v = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            return plane_distance * plane_distance

    return min(
        _point_segment_distance_squared(x, p1, p2),
        _point_segment_distance_squared(x, p2, p3),
        _point_segment_distance_squared(x, p3, p1),
    )


def _triangle_solid_angle(x: _Point3, triangle: _Triangle3) -> float:
    """Return the Van Oosterom--Strackee signed triangle solid angle."""

    a = _subtract(triangle[0], x)
    b = _subtract(triangle[1], x)
    c = _subtract(triangle[2], x)
    length_a = math.sqrt(_dot(a, a))
    length_b = math.sqrt(_dot(b, b))
    length_c = math.sqrt(_dot(c, c))
    determinant = _dot(a, _cross(b, c))
    denominator = (
        length_a * length_b * length_c
        + _dot(a, b) * length_c
        + _dot(a, c) * length_b
        + _dot(b, c) * length_a
    )
    return 2.0 * math.atan2(determinant, denominator)


def _newell_normal(points: tuple[_Point3, ...]) -> _Point3:
    components = [0.0, 0.0, 0.0]
    previous = points[-1]
    for current in points:
        components[0] += (previous[1] - current[1]) * (
            previous[2] + current[2]
        )
        components[1] += (previous[2] - current[2]) * (
            previous[0] + current[0]
        )
        components[2] += (previous[0] - current[0]) * (
            previous[1] + current[1]
        )
        previous = current
    return components[0], components[1], components[2]


def _prepare_polyhedron_triangles(cell: Any) -> tuple[_Triangle3, ...] | None:
    """Validate and triangulate one native polyhedron without changing the grid."""

    polyhedron = vtk.vtkPolyhedron.SafeDownCast(cell.GetRepresentativeCell())
    if polyhedron is None or int(polyhedron.GetNumberOfFaces()) < 4:
        return None
    cell_point_ids = tuple(
        int(polyhedron.GetPointIds().GetId(index))
        for index in range(int(polyhedron.GetNumberOfPoints()))
    )
    if len(cell_point_ids) < 4 or len(cell_point_ids) != len(set(cell_point_ids)):
        return None
    cell_point_id_set = set(cell_point_ids)

    triangles: list[_Triangle3] = []
    used_point_ids: set[int] = set()
    face_keys: set[tuple[int, ...]] = set()
    coordinate_by_point_id: dict[int, _Point3] = {}
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    face_count = int(polyhedron.GetNumberOfFaces())
    for face_index in range(face_count):
        face = polyhedron.GetFace(face_index)
        if face is None:
            return None
        face_point_count = int(face.GetNumberOfPoints())
        if face_point_count < 3:
            return None
        point_ids = tuple(
            int(face.GetPointIds().GetId(index))
            for index in range(face_point_count)
        )
        if (
            len(point_ids) != len(set(point_ids))
            or not set(point_ids).issubset(cell_point_id_set)
        ):
            return None
        face_key = tuple(sorted(point_ids))
        if face_key in face_keys:
            return None
        face_keys.add(face_key)
        used_point_ids.update(point_ids)

        face_points = tuple(
            tuple(float(value) for value in face.GetPoints().GetPoint(index))
            for index in range(face_point_count)
        )
        if any(
            not all(math.isfinite(value) for value in point)
            for point in face_points
        ):
            return None
        for point_id, point in zip(point_ids, face_points):
            existing = coordinate_by_point_id.setdefault(point_id, point)
            if existing != point:
                return None
        for edge_index, start in enumerate(point_ids):
            end = point_ids[(edge_index + 1) % face_point_count]
            if start == end:
                return None
            key = (min(start, end), max(start, end))
            direction = 1 if start < end else -1
            edge_occurrences.setdefault(key, []).append((face_index, direction))

        if face_point_count == 3:
            local_triangle_ids = (0, 1, 2)
        else:
            triangle_ids = vtk.vtkIdList()
            if int(face.TriangulateLocalIds(0, triangle_ids)) != 1:
                return None
            local_triangle_ids = tuple(
                int(triangle_ids.GetId(index))
                for index in range(int(triangle_ids.GetNumberOfIds()))
            )
            if len(local_triangle_ids) != 3 * (face_point_count - 2):
                return None
        if any(
            local_id < 0 or local_id >= face_point_count
            for local_id in local_triangle_ids
        ):
            return None

        face_normal = _newell_normal(face_points)
        if not math.isfinite(_dot(face_normal, face_normal)) or _dot(
            face_normal, face_normal
        ) <= 0.0:
            return None
        local_triangle_keys: set[tuple[int, int, int]] = set()
        local_triangle_edges: dict[
            tuple[int, int], list[tuple[int, int]]
        ] = {}
        for start in range(0, len(local_triangle_ids), 3):
            ids = local_triangle_ids[start : start + 3]
            if len(set(ids)) != 3:
                return None
            triangle_key = tuple(sorted(ids))
            if triangle_key in local_triangle_keys:
                return None
            local_triangle_keys.add(triangle_key)
            triangle = (
                face_points[ids[0]],
                face_points[ids[1]],
                face_points[ids[2]],
            )
            triangle_normal = _cross(
                _subtract(triangle[1], triangle[0]),
                _subtract(triangle[2], triangle[0]),
            )
            orientation = _dot(triangle_normal, face_normal)
            if not math.isfinite(orientation) or orientation <= 0.0:
                return None
            for edge_start, edge_end in (
                (ids[0], ids[1]),
                (ids[1], ids[2]),
                (ids[2], ids[0]),
            ):
                edge_key = (
                    min(edge_start, edge_end),
                    max(edge_start, edge_end),
                )
                local_triangle_edges.setdefault(edge_key, []).append(
                    (edge_start, edge_end)
                )
            triangles.append(triangle)

        boundary_edges = {
            (min(index, (index + 1) % face_point_count),
             max(index, (index + 1) % face_point_count)): (
                index,
                (index + 1) % face_point_count,
            )
            for index in range(face_point_count)
        }
        if not set(boundary_edges).issubset(local_triangle_edges):
            return None
        for edge_key, directed_occurrences in local_triangle_edges.items():
            boundary_direction = boundary_edges.get(edge_key)
            if boundary_direction is not None:
                if directed_occurrences != [boundary_direction]:
                    return None
            elif (
                len(directed_occurrences) != 2
                or directed_occurrences[0]
                != tuple(reversed(directed_occurrences[1]))
            ):
                return None

    if used_point_ids != cell_point_id_set or not triangles:
        return None
    face_adjacency = [set() for _ in range(face_count)]
    for occurrences in edge_occurrences.values():
        if len(occurrences) != 2 or occurrences[0][1] + occurrences[1][1] != 0:
            return None
        first_face = occurrences[0][0]
        second_face = occurrences[1][0]
        if first_face == second_face:
            return None
        face_adjacency[first_face].add(second_face)
        face_adjacency[second_face].add(first_face)
    connected = {0}
    pending = [0]
    while pending:
        face_index = pending.pop()
        for neighbor in face_adjacency[face_index]:
            if neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)
    if len(connected) != face_count:
        return None
    return tuple(triangles)


def _classify_polyhedron_solid_angle_detail(
    triangles: tuple[_Triangle3, ...], point_m: _Point3
) -> _PolyhedronClosureEvaluation:
    absolute_total = abs(
        math.fsum(
            _triangle_solid_angle(point_m, triangle) for triangle in triangles
        )
    )
    if not math.isfinite(absolute_total):
        return _PolyhedronClosureEvaluation(None, "ambiguous")
    inside_error = abs(absolute_total - 4.0 * math.pi)
    if inside_error <= POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE:
        return _PolyhedronClosureEvaluation(
            True,
            "inside",
            max(
                0.0,
                POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE - inside_error,
            ),
        )
    if absolute_total <= POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE:
        return _PolyhedronClosureEvaluation(
            False,
            "outside",
            max(
                0.0,
                POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE - absolute_total,
            ),
        )
    return _PolyhedronClosureEvaluation(None, "ambiguous")


def _classify_polyhedron_solid_angle(
    triangles: tuple[_Triangle3, ...], point_m: _Point3
) -> bool | None:
    """Compatibility wrapper returning only the candidate-v7 classification."""

    return _classify_polyhedron_solid_angle_detail(triangles, point_m).closure


def _evaluate_prepared_polyhedron_closure(
    triangles: tuple[_Triangle3, ...],
    point_m: _Point3,
    tolerance_squared_m2: float,
) -> _PolyhedronClosureEvaluation:
    if not triangles:
        return _PolyhedronClosureEvaluation(None, "ambiguous")
    for triangle in triangles:
        distance_squared = _point_triangle_distance_squared(point_m, triangle)
        if not math.isfinite(distance_squared) or distance_squared < 0.0:
            return _PolyhedronClosureEvaluation(None, "ambiguous")
        if distance_squared <= tolerance_squared_m2:
            return _PolyhedronClosureEvaluation(True, "boundary")
    return _classify_polyhedron_solid_angle_detail(triangles, point_m)


def _evaluate_polyhedron_closure(
    cell: Any, point_m: _Point3, tolerance_squared_m2: float
) -> bool | None:
    triangles = _prepare_polyhedron_triangles(cell)
    if triangles is None:
        return None
    return _evaluate_prepared_polyhedron_closure(
        triangles, point_m, tolerance_squared_m2
    ).closure


class NativeContainingCellKernel:
    """Deterministic candidate locator over an unchanged native volume grid."""

    def __init__(
        self,
        grid: Any,
        *,
        validation_chunk_size: int = DEFAULT_VALIDATION_CHUNK_SIZE,
        query_cache_enabled: bool = True,
    ) -> None:
        if not isinstance(query_cache_enabled, bool):
            raise VelocityAssignmentError("query_cache_enabled must be a boolean")
        _, self.cell_count = _validate_native_grid(
            grid, chunk_size=validation_chunk_size
        )
        self.grid = grid
        self.locator = vtk.vtkCellTreeLocator()
        self.locator.SetAutomatic(True)
        self.locator.SetNumberOfCellsPerNode(8)
        self.locator.SetNumberOfBuckets(6)
        self.locator.SetCacheCellBounds(True)
        self.locator.SetDataSet(grid)
        self.locator.BuildLocator()
        if self.locator.GetDataSet() is not grid:
            raise VelocityAssignmentError("VTK locator did not retain the native grid")
        if (
            not bool(self.locator.GetAutomatic())
            or int(self.locator.GetNumberOfCellsPerNode()) != 8
            or int(self.locator.GetNumberOfBuckets()) != 6
            or not bool(self.locator.GetCacheCellBounds())
        ):
            raise VelocityAssignmentError(
                "VTK cell-tree locator did not retain the pinned build settings"
            )
        # Assignment is deliberately serial.  Reuse the broad-phase ID list and
        # point-count-specific output buffers, but never reuse vtkGenericCell:
        # real vtkPolyhedron materialization retains internal state that is not
        # reliably cleared when one wrapper is overwritten with another raw
        # native cell.
        self._broad_ids = vtk.vtkIdList()
        self._evaluation_scratch: dict[
            int,
            tuple[
                list[float],
                Any,
                list[float],
                Any,
                list[float],
            ],
        ] = {}
        # VTK face materialization and polygon triangulation are substantially
        # more expensive than the small distance/winding calculation.  Profile
        # samples are spatially ordered and repeatedly touch the same native
        # cells, so retain immutable emitted triangles lazily by raw cell ID.
        # A cached None is also intentional: malformed geometry must fail the
        # same way on every query without repeating expensive VTK work.
        self._polyhedron_triangle_cache: OrderedDict[
            int, tuple[_Triangle3, ...] | None
        ] = OrderedDict()
        self._polyhedron_cache_max_entries = (
            POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES
        )
        self._polyhedron_cache_max_triangles = (
            POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES
        )
        self._polyhedron_cache_current_triangles = 0
        self._polyhedron_cache_peak_entries = 0
        self._polyhedron_cache_peak_triangles = 0
        self._polyhedron_cache_hits = 0
        self._polyhedron_cache_misses = 0
        self._polyhedron_cache_evictions = 0
        self._polyhedron_cache_oversized_bypasses = 0
        self._polyhedron_preparation_failures = 0
        self._polyhedron_evaluation_counts = {
            "boundary": 0,
            "inside": 0,
            "outside": 0,
            "ambiguous": 0,
        }
        self._polyhedron_minimum_classification_margin: float | None = None
        self._native_grid_mtime = _native_grid_mtime_binding(grid)
        self.query_cache_enabled = query_cache_enabled
        self._query_cache: dict[
            bytes, tuple[tuple[int, ...], tuple[int, ...]]
        ] = {}
        self._query_keys_seen: set[bytes] = set()
        self._query_total_rows = 0
        self._query_evaluation_count = 0

    @staticmethod
    def _exact_query_key(
        point_m: tuple[float, float, float], tolerance_m: float
    ) -> bytes:
        """Encode the exact binary64 XYZ/tolerance values without rounding."""

        return _QUERY_CACHE_KEY.pack(
            float(point_m[0]),
            float(point_m[1]),
            float(point_m[2]),
            float(tolerance_m),
        )

    def query_cache_audit(self) -> dict[str, object]:
        """Return deterministic counters without exposing local cache contents."""

        cache_hits = self._query_total_rows - self._query_evaluation_count
        if cache_hits < 0:  # Defensive invariant; unreachable through public calls.
            raise VelocityAssignmentError("containing-cell query counters are invalid")
        return {
            "enabled": self.query_cache_enabled,
            "key_id": QUERY_CACHE_KEY_ID,
            "total_rows": self._query_total_rows,
            "unique_query_keys": len(self._query_keys_seen),
            "cache_hits": cache_hits,
        }

    def polyhedron_geometry_cache_audit(self) -> dict[str, object]:
        """Return bounded-cache counters without exposing cached geometry."""

        return {
            "policy": "deterministic_least_recently_used",
            "maximum_entries": self._polyhedron_cache_max_entries,
            "maximum_emitted_triangles": self._polyhedron_cache_max_triangles,
            "current_entries": len(self._polyhedron_triangle_cache),
            "current_emitted_triangles": (
                self._polyhedron_cache_current_triangles
            ),
            "peak_entries": self._polyhedron_cache_peak_entries,
            "peak_emitted_triangles": self._polyhedron_cache_peak_triangles,
            "cache_hits": self._polyhedron_cache_hits,
            "cache_misses": self._polyhedron_cache_misses,
            "evictions": self._polyhedron_cache_evictions,
            "oversized_entry_bypasses": (
                self._polyhedron_cache_oversized_bypasses
            ),
            "fail_closed_preparations": self._polyhedron_preparation_failures,
            "vtk_objects_cached": False,
        }

    def polyhedron_evaluation_audit(self) -> dict[str, object]:
        """Return non-output-changing deterministic polyhedron query counters."""

        counts = self._polyhedron_evaluation_counts
        classified_count = counts["inside"] + counts["outside"]
        minimum_margin = self._polyhedron_minimum_classification_margin
        if (classified_count == 0) != (minimum_margin is None):
            raise VelocityAssignmentError(
                "polyhedron classification-margin counters are inconsistent"
            )
        return {
            "scope": (
                "broad_phase_polyhedron_visits_for_uncached_exact_xyz_"
                "tolerance_queries"
            ),
            "broad_phase_polyhedron_visit_count": sum(counts.values()),
            "boundary_count": counts["boundary"],
            "inside_count": counts["inside"],
            "outside_count": counts["outside"],
            "ambiguous_count": counts["ambiguous"],
            "winding_classified_count": classified_count,
            "minimum_winding_classification_margin_steradian": minimum_margin,
            "classification_absolute_tolerance_steradian": (
                POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
            ),
        }

    def _record_polyhedron_evaluation(
        self, evaluation: _PolyhedronClosureEvaluation
    ) -> None:
        expected_closure = {
            "boundary": True,
            "inside": True,
            "outside": False,
            "ambiguous": None,
        }
        if (
            evaluation.category not in expected_closure
            or evaluation.closure is not expected_closure[evaluation.category]
        ):
            raise VelocityAssignmentError(
                "polyhedron closure evaluation returned an invalid category"
            )
        margin = evaluation.accepted_classification_margin_steradian
        if evaluation.category in {"inside", "outside"}:
            if (
                margin is None
                or not math.isfinite(margin)
                or margin < 0.0
                or margin > POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
            ):
                raise VelocityAssignmentError(
                    "polyhedron closure evaluation returned an invalid margin"
                )
            current = self._polyhedron_minimum_classification_margin
            if current is None or margin < current:
                self._polyhedron_minimum_classification_margin = margin
        elif margin is not None:
            raise VelocityAssignmentError(
                "polyhedron non-winding evaluation unexpectedly returned a margin"
            )
        self._polyhedron_evaluation_counts[evaluation.category] += 1

    def _cache_polyhedron_triangles(
        self, raw_id: int, triangles: tuple[_Triangle3, ...] | None
    ) -> None:
        triangle_count = 0 if triangles is None else len(triangles)
        if triangle_count > self._polyhedron_cache_max_triangles:
            self._polyhedron_cache_oversized_bypasses += 1
            return
        previous = self._polyhedron_triangle_cache.pop(raw_id, _CACHE_MISS)
        if previous is not _CACHE_MISS and previous is not None:
            self._polyhedron_cache_current_triangles -= len(previous)
        self._polyhedron_triangle_cache[raw_id] = triangles
        self._polyhedron_cache_current_triangles += triangle_count
        while (
            len(self._polyhedron_triangle_cache) > self._polyhedron_cache_max_entries
            or self._polyhedron_cache_current_triangles
            > self._polyhedron_cache_max_triangles
        ):
            _, evicted = self._polyhedron_triangle_cache.popitem(last=False)
            if evicted is not None:
                self._polyhedron_cache_current_triangles -= len(evicted)
            self._polyhedron_cache_evictions += 1
        self._polyhedron_cache_peak_entries = max(
            self._polyhedron_cache_peak_entries,
            len(self._polyhedron_triangle_cache),
        )
        self._polyhedron_cache_peak_triangles = max(
            self._polyhedron_cache_peak_triangles,
            self._polyhedron_cache_current_triangles,
        )

    def _cached_polyhedron_triangles(
        self, raw_id: int
    ) -> tuple[_Triangle3, ...] | None | object:
        triangles = self._polyhedron_triangle_cache.get(raw_id, _CACHE_MISS)
        if triangles is _CACHE_MISS:
            return _CACHE_MISS
        self._polyhedron_triangle_cache.move_to_end(raw_id)
        self._polyhedron_cache_hits += 1
        return triangles

    def _assert_native_grid_unchanged(self) -> None:
        if _native_grid_mtime_binding(self.grid) != self._native_grid_mtime:
            raise VelocityAssignmentError(
                "native volume geometry changed after containing-cell kernel "
                "construction"
            )

    def _query_closure_candidates(
        self,
        point_m: tuple[float, float, float],
        tolerance_m: float,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        key = self._exact_query_key(point_m, tolerance_m)
        self._query_total_rows += 1
        self._query_keys_seen.add(key)
        if self.query_cache_enabled:
            cached = self._query_cache.get(key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                return cached  # type: ignore[return-value]
        self._query_evaluation_count += 1
        result = self._closure_candidates(point_m, tolerance_m)
        if self.query_cache_enabled:
            self._query_cache[key] = result
        return result

    def _closure_candidates(
        self,
        point_m: tuple[float, float, float],
        tolerance_m: float,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        bounds = [
            point_m[0] - tolerance_m,
            point_m[0] + tolerance_m,
            point_m[1] - tolerance_m,
            point_m[1] + tolerance_m,
            point_m[2] - tolerance_m,
            point_m[2] + tolerance_m,
        ]
        broad_ids = self._broad_ids
        broad_ids.Reset()
        self.locator.FindCellsWithinBounds(bounds, broad_ids)
        raw_ids = sorted(
            {int(broad_ids.GetId(index)) for index in range(broad_ids.GetNumberOfIds())}
        )
        accepted: list[int] = []
        evaluation_failures: list[int] = []
        tolerance_squared = tolerance_m * tolerance_m
        for raw_id in raw_ids:
            if raw_id < 0 or raw_id >= self.cell_count:
                raise VelocityAssignmentError(
                    f"VTK locator returned invalid raw cell ID {raw_id}"
                )
            cached_triangles = self._cached_polyhedron_triangles(raw_id)
            if cached_triangles is not _CACHE_MISS:
                triangles = cached_triangles
                if triangles is None:
                    self._record_polyhedron_evaluation(
                        _PolyhedronClosureEvaluation(None, "ambiguous")
                    )
                    evaluation_failures.append(raw_id)
                else:
                    polyhedron_evaluation = _evaluate_prepared_polyhedron_closure(
                        triangles,
                        point_m,
                        tolerance_squared,
                    )
                    self._record_polyhedron_evaluation(polyhedron_evaluation)
                    if polyhedron_evaluation.closure is None:
                        evaluation_failures.append(raw_id)
                    elif polyhedron_evaluation.closure:
                        accepted.append(raw_id)
                continue

            cell = vtk.vtkGenericCell()
            self.grid.GetCell(raw_id, cell)
            cell_type = int(cell.GetCellType())
            if cell_type not in _SUPPORTED_CELL_TYPE_IDS:
                raise VelocityAssignmentError(
                    f"unsupported native cell type {cell_type} at raw VTK cell ID {raw_id}"
                )
            if cell_type == 42:
                self._polyhedron_cache_misses += 1
                triangles = _prepare_polyhedron_triangles(cell)
                if triangles is None:
                    self._polyhedron_preparation_failures += 1
                self._cache_polyhedron_triangles(raw_id, triangles)
                if triangles is None:
                    self._record_polyhedron_evaluation(
                        _PolyhedronClosureEvaluation(None, "ambiguous")
                    )
                    evaluation_failures.append(raw_id)
                else:
                    polyhedron_evaluation = _evaluate_prepared_polyhedron_closure(
                        triangles,
                        point_m,
                        tolerance_squared,
                    )
                    self._record_polyhedron_evaluation(polyhedron_evaluation)
                    if polyhedron_evaluation.closure is None:
                        evaluation_failures.append(raw_id)
                    elif polyhedron_evaluation.closure:
                        accepted.append(raw_id)
                continue
            cell_point_count = int(cell.GetNumberOfPoints())
            scratch = self._evaluation_scratch.get(cell_point_count)
            if scratch is None:
                scratch = (
                    [0.0, 0.0, 0.0],
                    vtk.reference(0),
                    [0.0, 0.0, 0.0],
                    vtk.reference(0.0),
                    [0.0] * cell_point_count,
                )
                self._evaluation_scratch[cell_point_count] = scratch
            closest, sub_id, parametric, distance_squared, weights = scratch
            # Poison every floating output before reuse.  Real native cell
            # implementations do not all overwrite every output on every
            # return path.  NaN makes an unwritten output fail closed instead of
            # disguising it as a valid zero-distance closure candidate.
            closest[0] = closest[1] = closest[2] = math.nan
            sub_id.set(0)
            parametric[0] = parametric[1] = parametric[2] = math.nan
            distance_squared.set(math.nan)
            for weight_index in range(cell_point_count):
                weights[weight_index] = math.nan
            status = int(
                cell.EvaluatePosition(
                    point_m,
                    closest,
                    sub_id,
                    parametric,
                    distance_squared,
                    weights,
                )
            )
            if status == -1:
                # VTK documents -1 as a numerical evaluation failure.  This can
                # occur for a malformed/degenerate broad-phase cell even when
                # the query has another usable closure candidate.  Silently
                # dropping that cell could alter both containment and the
                # smallest-raw-ID tie break, so the caller retains the complete
                # sample as explicitly invalid and records every failed ID.
                evaluation_failures.append(raw_id)
                continue
            if status not in {0, 1}:
                raise VelocityAssignmentError(
                    "vtkGenericCell.EvaluatePosition returned undocumented code "
                    f"{status} for raw VTK cell ID {raw_id}"
                )
            distance2 = float(distance_squared)
            numeric_outputs = (*closest, *parametric, *weights, distance2)
            if (
                not all(math.isfinite(float(value)) for value in numeric_outputs)
                or distance2 < 0.0
                or (status == 1 and distance2 != 0.0)
            ):
                # Some vtkPolyhedron failure paths return status 0 while leaving
                # vtkCellLocator's negative distance sentinel in place.  Treat
                # malformed numeric outputs exactly like the documented -1
                # return: retain the raw ID and conservatively invalidate the
                # complete sample instead of aborting or accepting the cell.
                evaluation_failures.append(raw_id)
                continue
            if status == 1 or distance2 <= tolerance_squared:
                accepted.append(raw_id)
        return tuple(accepted), tuple(evaluation_failures)

    def assign(
        self,
        samples: Sequence[VelocitySampleDefinition],
        *,
        case_id: str,
        source_sha256: tuple[str, ...],
        tolerance_m: float = POINT_IN_CELL_CLOSURE_TOLERANCE_M,
        invalid_reasons: Mapping[tuple[str, int], str] | None = None,
    ) -> tuple[VelocityCellAssignmentEvidence, ...]:
        """Assign samples in caller order and retain every invalid row."""

        self._assert_native_grid_unchanged()
        if not isinstance(case_id, str) or not case_id.strip():
            raise VelocityAssignmentError("case_id must be a non-empty string")
        tolerance = _positive_finite(tolerance_m, "tolerance_m")
        hashes = _source_hashes(source_sha256)
        records = _validated_samples(samples)
        owner_invalid = _validated_invalid_reasons(records, invalid_reasons)
        result: list[VelocityCellAssignmentEvidence] = []
        for sample in records:
            candidates, evaluation_failures = self._query_closure_candidates(
                sample.point_m, tolerance
            )
            key = _sample_key(sample)
            owner_reason = owner_invalid.get(key)
            if owner_reason is not None:
                valid = False
                reason = owner_reason
                raw_cell_id = None
            elif evaluation_failures:
                valid = False
                reason = CELL_EVALUATION_FAILURE_REASON_PREFIX + ",".join(
                    str(raw_id) for raw_id in evaluation_failures
                )
                raw_cell_id = None
            elif candidates:
                valid = True
                reason = ""
                raw_cell_id = candidates[0]
            else:
                valid = False
                reason = NO_CLOSURE_CELL_REASON
                raw_cell_id = None
            result.append(
                VelocityCellAssignmentEvidence(
                    case_id=case_id,
                    profile_id=sample.profile_id,
                    sample_index=sample.sample_index,
                    point_m=sample.point_m,
                    distance_m=sample.distance_m,
                    valid=valid,
                    reason=reason,
                    raw_vtk_cell_id=raw_cell_id,
                    candidate_count=len(candidates),
                    geometric_tolerance_m=tolerance,
                    source_sha256=hashes,
                )
            )
        return tuple(result)


def _assignment_dict(record: VelocityCellAssignmentEvidence) -> dict[str, object]:
    return {
        "case_id": record.case_id,
        "profile_id": record.profile_id,
        "sample_index": record.sample_index,
        "point_m": list(record.point_m),
        "distance_m": record.distance_m,
        "valid": record.valid,
        "reason": record.reason,
        "raw_vtk_cell_id": record.raw_vtk_cell_id,
        "candidate_count": record.candidate_count,
        "geometric_tolerance_m": record.geometric_tolerance_m,
        "source_sha256": list(record.source_sha256),
    }


def assignment_evidence_sha256(
    assignments: Sequence[VelocityCellAssignmentEvidence],
) -> str:
    """Hash assignment rows in their explicit evidence order."""

    records = tuple(assignments)
    if any(not isinstance(item, VelocityCellAssignmentEvidence) for item in records):
        raise VelocityAssignmentError("assignment evidence contains the wrong record type")
    return hashlib.sha256(
        _canonical_json([_assignment_dict(item) for item in records])
    ).hexdigest()


@dataclass(frozen=True)
class VelocityToleranceReplay:
    """Three deterministic assignment sets for tolerance-sensitivity review."""

    case_id: str
    source_sha256: tuple[str, ...]
    tolerances_m: tuple[float, ...]
    assignments_by_tolerance: tuple[
        tuple[VelocityCellAssignmentEvidence, ...], ...
    ]

    def as_dict(self, *, include_assignments: bool = True) -> dict[str, object]:
        receipt = candidate_kernel_receipt()
        rows: list[dict[str, object]] = []
        for tolerance, assignments in zip(
            self.tolerances_m, self.assignments_by_tolerance
        ):
            item: dict[str, object] = {
                "tolerance_m": tolerance,
                "assignment_count": len(assignments),
                "valid_count": sum(record.valid for record in assignments),
                "invalid_count": sum(not record.valid for record in assignments),
                "candidate_count_sum": sum(
                    record.candidate_count for record in assignments
                ),
                "multiple_candidate_count": sum(
                    record.candidate_count > 1 for record in assignments
                ),
                "assignment_sha256": assignment_evidence_sha256(assignments),
            }
            if include_assignments:
                item["assignments"] = [
                    _assignment_dict(record) for record in assignments
                ]
            rows.append(item)
        return {
            "schema": REPLAY_SCHEMA,
            "status": "candidate_not_frozen",
            "case_id": self.case_id,
            "source_sha256": list(self.source_sha256),
            "kernel": receipt,
            "tolerances": rows,
        }


def replay_assignment_tolerances(
    kernel: NativeContainingCellKernel,
    samples: Sequence[VelocitySampleDefinition],
    *,
    case_id: str,
    source_sha256: tuple[str, ...],
    invalid_reasons: Mapping[tuple[str, int], str] | None = None,
    tolerances_m: Sequence[float] = TOLERANCE_REPLAY_M,
) -> VelocityToleranceReplay:
    """Run deterministic assignment evidence at 0.5, 1, and 2 micrometres."""

    if not isinstance(kernel, NativeContainingCellKernel):
        raise VelocityAssignmentError("kernel must be a NativeContainingCellKernel")
    records = _validated_samples(samples)
    tolerances = tuple(
        _positive_finite(value, "tolerances_m") for value in tolerances_m
    )
    if not tolerances:
        raise VelocityAssignmentError("tolerances_m cannot be empty")
    if len(tolerances) != len(set(tolerances)):
        raise VelocityAssignmentError("tolerances_m must be unique")
    hashes = _source_hashes(source_sha256)
    assignments = tuple(
        kernel.assign(
            records,
            case_id=case_id,
            source_sha256=hashes,
            tolerance_m=tolerance,
            invalid_reasons=invalid_reasons,
        )
        for tolerance in tolerances
    )
    return VelocityToleranceReplay(
        case_id=case_id,
        source_sha256=hashes,
        tolerances_m=tolerances,
        assignments_by_tolerance=assignments,
    )


__all__ = [
    "CELL_EVALUATION_FAILURE_REASON_PREFIX",
    "DEFAULT_VALIDATION_CHUNK_SIZE",
    "EVALUATE_POSITION_FAILURE_REASON_PREFIX",
    "KERNEL_ID",
    "NO_CLOSURE_CELL_REASON",
    "NativeContainingCellKernel",
    "OWNER_INVALID_REASONS",
    "POLYHEDRON_CLOSURE_ID",
    "POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES",
    "POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES",
    "POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE",
    "POLYHEDRON_UPSTREAM_BASIS_COMMIT",
    "QUERY_CACHE_KEY_ID",
    "REPLAY_SCHEMA",
    "REQUIRED_VTK_VERSION",
    "SUPPORTED_CELL_TYPES",
    "TOLERANCE_REPLAY_M",
    "VelocityAssignmentError",
    "VelocityToleranceReplay",
    "assignment_evidence_sha256",
    "candidate_kernel_receipt",
    "candidate_kernel_settings",
    "generate_definition_velocity_samples",
    "generate_velocity_line_samples",
    "replay_assignment_tolerances",
    "vtk_available",
]
