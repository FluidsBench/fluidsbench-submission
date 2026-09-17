"""AirfRANS native-mesh geometry: point-dual area/length weighting.

FluidsBench's AirfRANS support publishes predictions as PointData on the
native 2D internal mesh and the native airfoil boundary curve, but the only
geometric measure the pinned ``airfrans`` library computes is per-cell
(``Simulation.internal``/``Simulation.airfoil`` are annotated with cell-level
``Area``/``Length`` by ``pyvista.compute_cell_sizes`` inside
``Simulation.reset``). This module converts those cell measures into a
physical per-point weight via equal-share (row-sum) mass lumping: each cell's
measure is split evenly across its own vertices and summed at every point.

This is the first use of point-dual weighting in FluidsBench (every other
dataset publishes cell-associated fields with cell-based weights instead), so
there is no existing utility to reuse. Equal-share lumping is used because it
is the simplest scheme with the property this module's own tests verify: the
sum of all point weights over a mesh equals the sum of the source cell
measure exactly, so weighted quantities. Nothing is double-counted or
dropped between the cell and point representations.
"""

from __future__ import annotations

import numpy as np


class AirfRANSGeometryError(ValueError):
    """Raised when a native mesh cannot be reduced to point-dual weights."""


def _cell_point_ids(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (cell_index, point_id) incidence pairs for every mesh cell."""

    cell_ids: list[np.ndarray] = []
    point_ids: list[np.ndarray] = []
    for cell_index in range(mesh.n_cells):
        points = mesh.get_cell(cell_index).point_ids
        if not points:
            raise AirfRANSGeometryError(f"cell {cell_index} has no vertices")
        cell_ids.append(np.full(len(points), cell_index, dtype=np.int64))
        point_ids.append(np.asarray(points, dtype=np.int64))
    return np.concatenate(cell_ids), np.concatenate(point_ids)


def equal_share_dual_weights(mesh, measure_array_name: str) -> np.ndarray:
    """Return one physical weight per mesh point via equal-share lumping.

    Each cell's ``cell_data[measure_array_name]`` value (an Area for the 2D
    internal domain, a Length for the 1D airfoil curve, both produced by
    ``pyvista.compute_cell_sizes``) is divided equally among that cell's own
    vertices; a point's weight is the sum of every incident cell's share.

    Raises ``AirfRANSGeometryError`` if the measure is missing, non-finite,
    non-positive, or if any resulting point weight is non-positive (an
    isolated point with no incident cells, which should not occur on a
    native AirfRANS mesh).
    """

    if measure_array_name not in mesh.cell_data:
        raise AirfRANSGeometryError(
            f"mesh is missing cell_data[{measure_array_name!r}]; call "
            "pyvista's compute_cell_sizes first"
        )
    measure = np.asarray(mesh.cell_data[measure_array_name], dtype=np.float64)
    if measure.shape != (mesh.n_cells,):
        raise AirfRANSGeometryError(
            f"{measure_array_name} must have one value per cell"
        )
    if not np.all(np.isfinite(measure)) or np.any(measure <= 0):
        raise AirfRANSGeometryError(
            f"{measure_array_name} must be finite and strictly positive for every cell"
        )

    cell_ids, point_ids = _cell_point_ids(mesh)
    vertex_counts_per_cell = np.bincount(cell_ids, minlength=mesh.n_cells)
    share_per_cell = measure / vertex_counts_per_cell
    share_per_incidence = share_per_cell[cell_ids]

    weights = np.zeros(mesh.n_points, dtype=np.float64)
    np.add.at(weights, point_ids, share_per_incidence)

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise AirfRANSGeometryError(
            "every mesh point must receive a finite, strictly positive dual weight "
            "(an isolated point with no incident cells would violate this)"
        )
    return weights


def assert_dual_weight_conservation(
    mesh,
    measure_array_name: str,
    weights: np.ndarray,
    *,
    rtol: float = 1e-9,
) -> None:
    """Assert that point-dual weights sum to the total source cell measure."""

    total_measure = float(np.sum(mesh.cell_data[measure_array_name]))
    total_weight = float(np.sum(weights))
    if not np.isclose(total_weight, total_measure, rtol=rtol, atol=0.0):
        raise AirfRANSGeometryError(
            "point-dual weights do not conserve the total cell measure: "
            f"cells sum to {total_measure!r}, points sum to {total_weight!r}"
        )
