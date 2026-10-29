from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from reference.airfrans.geometry import (
    AirfRANSGeometryError,
    assert_dual_weight_conservation,
    equal_share_dual_weights,
)


@dataclass
class _FakeCell:
    point_ids: list[int]


class _FakeMesh:
    """Minimal duck-typed stand-in for a pyvista mesh.

    ``reference.airfrans.geometry`` never imports pyvista itself -- it only
    calls ``n_cells``/``n_points``/``cell_data``/``get_cell(i).point_ids`` --
    so these tests exercise the real lumping math without needing PyVista or
    VTK installed, keeping this test in the repository's light, dependency-
    free unit-test tier (see requirements-airfrans-evaluator.txt).
    """

    def __init__(self, n_points: int, cells: list[list[int]], cell_data: dict[str, np.ndarray]):
        self.n_points = n_points
        self.n_cells = len(cells)
        self._cells = cells
        self.cell_data = cell_data

    def get_cell(self, index: int) -> _FakeCell:
        return _FakeCell(point_ids=self._cells[index])


class EqualShareDualWeightsTests(unittest.TestCase):
    def test_two_triangles_conserve_total_area(self) -> None:
        # A unit square split into two triangles sharing the diagonal.
        # Triangle areas: 0.5 and 0.5 -> total area 1.0.
        mesh = _FakeMesh(
            n_points=4,
            cells=[[0, 1, 2], [0, 2, 3]],
            cell_data={"Area": np.array([0.5, 0.5])},
        )
        weights = equal_share_dual_weights(mesh, "Area")
        self.assertEqual(weights.shape, (4,))
        # point 0 and point 2 are shared by both triangles: 0.5/3 + 0.5/3 each.
        self.assertAlmostEqual(weights[0], 0.5 / 3 + 0.5 / 3)
        self.assertAlmostEqual(weights[2], 0.5 / 3 + 0.5 / 3)
        # points 1 and 3 belong to exactly one triangle each.
        self.assertAlmostEqual(weights[1], 0.5 / 3)
        self.assertAlmostEqual(weights[3], 0.5 / 3)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        assert_dual_weight_conservation(mesh, "Area", weights)

    def test_closed_quadrilateral_curve_conserves_perimeter(self) -> None:
        # A closed 4-segment polyline (unit square boundary), segment length 1 each.
        mesh = _FakeMesh(
            n_points=4,
            cells=[[0, 1], [1, 2], [2, 3], [3, 0]],
            cell_data={"Length": np.array([1.0, 1.0, 1.0, 1.0])},
        )
        weights = equal_share_dual_weights(mesh, "Length")
        # Every point touches exactly two unit-length segments -> weight 1.0 each.
        np.testing.assert_allclose(weights, np.full(4, 1.0))
        self.assertAlmostEqual(float(weights.sum()), 4.0)
        assert_dual_weight_conservation(mesh, "Length", weights)

    def test_unequal_valence_is_weighted_by_incident_cells(self) -> None:
        # Three triangles fanned around a shared center point (point 0).
        mesh = _FakeMesh(
            n_points=4,
            cells=[[0, 1, 2], [0, 2, 3], [0, 3, 1]],
            cell_data={"Area": np.array([3.0, 6.0, 9.0])},
        )
        weights = equal_share_dual_weights(mesh, "Area")
        # Center point collects a third of every incident triangle.
        self.assertAlmostEqual(weights[0], 3.0 / 3 + 6.0 / 3 + 9.0 / 3)
        self.assertAlmostEqual(float(weights.sum()), 18.0)

    def test_missing_measure_array_raises(self) -> None:
        mesh = _FakeMesh(n_points=3, cells=[[0, 1, 2]], cell_data={})
        with self.assertRaises(AirfRANSGeometryError):
            equal_share_dual_weights(mesh, "Area")

    def test_non_positive_measure_raises(self) -> None:
        mesh = _FakeMesh(n_points=3, cells=[[0, 1, 2]], cell_data={"Area": np.array([0.0])})
        with self.assertRaises(AirfRANSGeometryError):
            equal_share_dual_weights(mesh, "Area")

    def test_conservation_check_detects_corruption(self) -> None:
        mesh = _FakeMesh(
            n_points=3,
            cells=[[0, 1, 2]],
            cell_data={"Area": np.array([1.0])},
        )
        weights = equal_share_dual_weights(mesh, "Area")
        corrupted = weights.copy()
        corrupted[0] *= 2.0
        with self.assertRaises(AirfRANSGeometryError):
            assert_dual_weight_conservation(mesh, "Area", corrupted)


if __name__ == "__main__":
    unittest.main()
