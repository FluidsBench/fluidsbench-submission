from __future__ import annotations

import math
import unittest

import numpy as np

from reference.drivaerml.surface_forces import (
    DrivAerSurfaceForceError,
    audit_fixed_surface_areas,
    finalize_force_coefficients,
    force_moment_chunk,
    surface_geometry_chunk,
)


class DrivAerMLSurfaceForceTests(unittest.TestCase):
    def test_openfoam_area_and_centres_for_triangle_and_skew_quad(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.2, 2.0, 0.0],
                [-0.1, 2.0, 0.0],
            ]
        )
        connectivity = np.asarray([0, 1, 2, 2, 1, 3, 4, 5])
        offsets = np.asarray([0, 3, 8])
        geometry = surface_geometry_chunk(points, connectivity, offsets, 0, 2)
        np.testing.assert_allclose(geometry.centres_m[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
        np.testing.assert_allclose(geometry.oriented_area_vectors_m2[0], [0.0, 0.0, 1.0])
        self.assertGreater(geometry.areas_m2[1], 0.0)
        self.assertAlmostEqual(geometry.centres_m[1, 2], 0.0)

    def test_closed_cube_uniform_pressure_has_zero_force_and_chunk_invariance(self) -> None:
        points = np.asarray(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            dtype=np.float64,
        )
        # Six outward-oriented quad faces.
        faces = [
            [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        connectivity = np.asarray([value for face in faces for value in face])
        offsets = np.arange(0, 25, 4)
        pressure = np.full(6, 17.0)
        shear = np.zeros((6, 3))

        whole_geometry = surface_geometry_chunk(points, connectivity, offsets, 0, 6)
        whole = force_moment_chunk(whole_geometry, pressure, shear)
        whole_result = finalize_force_coefficients([whole], expected_entity_count=6)
        np.testing.assert_allclose(whole_result["force_n"], [0.0, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(
            whole_result["moment_about_forces_cor_n_m"], [0.0, 0.0, 0.0], atol=1e-14
        )

        chunks = []
        for start, stop in ((0, 1), (1, 4), (4, 6)):
            geometry = surface_geometry_chunk(points, connectivity, offsets, start, stop)
            chunks.append(force_moment_chunk(geometry, pressure[start:stop], shear[start:stop]))
        chunked_result = finalize_force_coefficients(reversed(chunks), expected_entity_count=6)
        for key in ("Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr"):
            self.assertTrue(math.isclose(whole_result[key], chunked_result[key], abs_tol=1e-14))
        self.assertEqual(chunked_result["lift_closure_abs"], 0.0)

    def test_fixed_area_audit_uses_published_float32_as_input(self) -> None:
        calculated = np.asarray([1.0, math.pi, 1.0e-6], dtype=np.float64)
        published = calculated.astype("<f4")
        receipt = audit_fixed_surface_areas(calculated, published)
        self.assertEqual(receipt["entity_count"], 3)
        with self.assertRaisesRegex(DrivAerSurfaceForceError, "little-endian float32"):
            audit_fixed_surface_areas(calculated, calculated)
        bad = published.copy()
        bad[1] *= 1.1
        with self.assertRaisesRegex(DrivAerSurfaceForceError, "differs"):
            audit_fixed_surface_areas(calculated, bad)

    def test_topology_fields_and_coverage_fail_closed(self) -> None:
        points = np.eye(3)
        with self.assertRaisesRegex(DrivAerSurfaceForceError, "define polygons"):
            surface_geometry_chunk(points, [0, 1], [0, 2], 0, 1)
        geometry = surface_geometry_chunk(points, [0, 1, 2], [0, 3], 0, 1)
        with self.assertRaisesRegex(DrivAerSurfaceForceError, "tuple"):
            force_moment_chunk(geometry, [1.0, 2.0], np.zeros((1, 3)))
        valid = force_moment_chunk(geometry, [1.0], np.zeros((1, 3)))
        with self.assertRaisesRegex(DrivAerSurfaceForceError, "cover every"):
            finalize_force_coefficients([valid], expected_entity_count=2)


if __name__ == "__main__":
    unittest.main()
