from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from reference.drivaerml import velocity_assignments as velocity_assignments_module
from reference.drivaerml.autocfd5 import (
    VelocityCellAssignmentEvidence,
    VelocitySampleDefinition,
    load_autocfd5_definition,
)
from reference.drivaerml.velocity_assignments import (
    CELL_EVALUATION_FAILURE_REASON_PREFIX,
    KERNEL_ID,
    NO_CLOSURE_CELL_REASON,
    QUERY_CACHE_KEY_ID,
    REQUIRED_NUMPY_VERSION,
    REQUIRED_PYTHON_VERSION,
    REQUIRED_VTK_VERSION,
    TOLERANCE_REPLAY_M,
    NativeContainingCellKernel,
    VelocityAssignmentError,
    _require_frozen_runtime,
    assignment_evidence_sha256,
    candidate_kernel_receipt,
    candidate_kernel_settings,
    generate_definition_velocity_samples,
    replay_assignment_tolerances,
    vtk_available,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
SOURCE_HASHES = ("a" * 64, "b" * 64)
VTK_READY = vtk_available()
SYNTHETIC_ASSIGNMENT_SHA256 = (
    "437f69f14282929a7a7ba2388f62ee3266a2611f3c25d7900752c9326333b404"
)
FULL_GRID_SHA256 = "46ffdde32e4892562d7e80e41a5727d50db22420efde26f17aa0db363b43a9f0"
KERNEL_SETTINGS_SHA256 = (
    "6cbd2b2fb56fc782fd9e9990bd43f2bad7fd040b355f128555ecf101b369fa1b"
)

if VTK_READY:
    import vtk


def _sample_payload(samples: tuple[VelocitySampleDefinition, ...]) -> bytes:
    rows = [
        {
            "profile_id": row.profile_id,
            "sample_index": row.sample_index,
            "point_count": row.point_count,
            "line_fraction": row.line_fraction,
            "distance_m": row.distance_m,
            "point_m": list(row.point_m),
        }
        for row in samples
    ]
    return json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class DrivAerMLVelocityGridDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_autocfd5_definition(PROFILE_PATH)

    def test_full_definition_generates_all_four_nested_grids(self) -> None:
        expected_counts = {
            0.001: 37_416,
            0.002: 18_716,
            0.005: 7_496,
            0.010: 3_756,
        }
        generated = {
            spacing: generate_definition_velocity_samples(self.definition, spacing)
            for spacing in expected_counts
        }
        self.assertEqual(
            {spacing: len(rows) for spacing, rows in generated.items()},
            expected_counts,
        )
        one_mm_by_line: dict[str, list[VelocitySampleDefinition]] = {}
        for row in generated[0.001]:
            one_mm_by_line.setdefault(row.profile_id, []).append(row)
        for spacing, stride in ((0.002, 2), (0.005, 5), (0.010, 10)):
            coarse_by_line: dict[str, list[VelocitySampleDefinition]] = {}
            for row in generated[spacing]:
                coarse_by_line.setdefault(row.profile_id, []).append(row)
            for profile_id, coarse in coarse_by_line.items():
                reference = one_mm_by_line[profile_id][::stride]
                self.assertEqual(len(coarse), len(reference))
                np.testing.assert_allclose(
                    [row.point_m for row in coarse],
                    [row.point_m for row in reference],
                    rtol=0.0,
                    atol=2.0e-15,
                )
                np.testing.assert_allclose(
                    [row.distance_m for row in coarse],
                    [row.distance_m for row in reference],
                    rtol=0.0,
                    atol=2.0e-15,
                )

        fixed = self.definition.velocity_samples
        ten_mm = generated[0.010]
        self.assertEqual(
            [(row.profile_id, row.sample_index, row.point_count) for row in ten_mm],
            [(row.profile_id, row.sample_index, row.point_count) for row in fixed],
        )
        np.testing.assert_allclose(
            [row.point_m for row in ten_mm],
            [row.point_m for row in fixed],
            rtol=0.0,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            [row.distance_m for row in ten_mm],
            [row.distance_m for row in fixed],
            rtol=0.0,
            atol=2.0e-11,
        )
        self.assertEqual(
            hashlib.sha256(_sample_payload(ten_mm)).hexdigest(),
            FULL_GRID_SHA256,
        )

    def test_grid_generation_rejects_invalid_spacing(self) -> None:
        for value in (0.0, -0.001, math.nan, math.inf, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    VelocityAssignmentError, "finite and positive|numeric"
                ):
                    generate_definition_velocity_samples(self.definition, value)

    def test_candidate_settings_are_explicitly_not_frozen(self) -> None:
        settings = candidate_kernel_settings()
        self.assertEqual(settings["kernel_id"], KERNEL_ID)
        self.assertTrue(KERNEL_ID.endswith("candidate-v7"))
        self.assertIn("candidate_pending", settings["status"])
        self.assertEqual(settings["required_dependency"], {"vtk": "9.5.2"})
        self.assertEqual(
            settings["candidate_discovery"],
            {
                "class": "vtkCellTreeLocator",
                "method": "FindCellsWithinBounds",
                "query": (
                    "exact_native_cell_axis_aligned_bounds_intersection_with_"
                    "point_bounds_expanded_by_absolute_tolerance"
                ),
                "query_bounds_order": [
                    "xmin",
                    "xmax",
                    "ymin",
                    "ymax",
                    "zmin",
                    "zmax",
                ],
                "deduplicate_raw_cell_ids": True,
                "settings": {
                    "automatic": True,
                    "number_of_cells_per_node": 8,
                    "number_of_build_buckets": 6,
                    "cache_cell_bounds": True,
                },
            },
        )
        self.assertEqual(
            settings["selection"]["candidate_count"],
            "number_of_distinct_cells_passing_closure_predicate",
        )
        closure = settings["closure"]
        self.assertEqual(closure["dispatch"], "native_vtk_cell_type")
        self.assertEqual(
            closure["fixed_native_cell_types"]["vtk_type_ids"],
            [10, 12, 13, 14],
        )
        polyhedron = closure["native_vtk_polyhedron"]
        self.assertEqual(
            polyhedron["closure_id"], "triangulated-signed-solid-angle-v1"
        )
        self.assertEqual(polyhedron["randomness"], "none")
        self.assertFalse(polyhedron["vtk_9_5_2_IsInside_called"])
        self.assertEqual(polyhedron["vtk_type_id"], 42)
        self.assertEqual(
            polyhedron["ambiguous_action"],
            "fail_closed_and_record_raw_cell_id",
        )
        self.assertEqual(
            polyhedron["classification_absolute_tolerance_steradian"],
            1.0e-3,
        )
        self.assertEqual(
            polyhedron["classification_tolerance_status"],
            "candidate_scientific_choice_pending_owner_approval",
        )
        self.assertEqual(
            polyhedron["upstream_algorithm_basis"],
            {
                "repository": "https://gitlab.kitware.com/vtk/vtk",
                "commit": "33c54a2c6b829031e4f013bf5cb7e45daef299d9",
                "file": "Common/DataModel/vtkPolyhedron.cxx",
                "function": "vtkPolyhedron::IsInside",
            },
        )
        self.assertEqual(len(polyhedron["deviations_from_upstream_basis"]), 4)
        self.assertIn(
            "all_triangle_distances_and_solid_angles",
            polyhedron["work_bound"]["each_query"],
        )
        geometry_cache = polyhedron["geometry_cache"]
        self.assertEqual(geometry_cache["key"], "raw_vtk_cell_id")
        self.assertFalse(geometry_cache["vtk_objects_cached"])
        self.assertIn("None_for_fail_closed", geometry_cache["value"])
        self.assertIn("distinct_broad_phase", geometry_cache["entry_bound"])
        self.assertEqual(geometry_cache["maximum_entries"], 8_192)
        self.assertEqual(
            geometry_cache["maximum_emitted_triangles"], 131_072
        )
        self.assertEqual(
            geometry_cache["eviction"], "deterministic_least_recently_used"
        )
        self.assertEqual(
            settings["required_tolerance_replay_m"], list(TOLERANCE_REPLAY_M)
        )
        self.assertFalse(settings["invalidity"]["snapping"])
        self.assertFalse(settings["invalidity"]["extrapolation"])


@unittest.skipUnless(VTK_READY, f"optional VTK {REQUIRED_VTK_VERSION} is unavailable")
class DrivAerMLVelocityContainingCellTests(unittest.TestCase):
    @staticmethod
    def _append_points(
        points: object, coordinates: list[tuple[float, float, float]]
    ) -> list[int]:
        return [points.InsertNextPoint(*coordinate) for coordinate in coordinates]

    @classmethod
    def mixed_grid(cls) -> object:
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.Allocate(5)

        tetra = cls._append_points(
            points,
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        grid.InsertNextCell(vtk.VTK_TETRA, len(tetra), tetra)

        hexahedron = cls._append_points(
            points,
            [
                (2, 0, 0),
                (3, 0, 0),
                (3, 1, 0),
                (2, 1, 0),
                (2, 0, 1),
                (3, 0, 1),
                (3, 1, 1),
                (2, 1, 1),
            ],
        )
        grid.InsertNextCell(vtk.VTK_HEXAHEDRON, len(hexahedron), hexahedron)

        wedge = cls._append_points(
            points,
            [
                (4, 0, 0),
                (5, 0, 0),
                (4, 1, 0),
                (4, 0, 1),
                (5, 0, 1),
                (4, 1, 1),
            ],
        )
        wedge_order = [wedge[index] for index in (0, 2, 1, 3, 5, 4)]
        grid.InsertNextCell(vtk.VTK_WEDGE, len(wedge_order), wedge_order)

        pyramid = cls._append_points(
            points,
            [(6, 0, 0), (7, 0, 0), (7, 1, 0), (6, 1, 0), (6.5, 0.5, 1)],
        )
        grid.InsertNextCell(vtk.VTK_PYRAMID, len(pyramid), pyramid)

        polyhedron = cls._append_points(
            points,
            [
                (8, 0, 0),
                (9, 0, 0),
                (9, 1, 0),
                (8, 1, 0),
                (8, 0, 1),
                (9, 0, 1),
                (9, 1, 1),
                (8, 1, 1),
            ],
        )
        faces = vtk.vtkCellArray()
        for local_face in (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ):
            face = [polyhedron[index] for index in local_face]
            faces.InsertNextCell(len(face), face)
        grid.InsertNextCell(
            vtk.VTK_POLYHEDRON,
            len(polyhedron),
            polyhedron,
            faces,
        )
        return grid

    @staticmethod
    def two_hex_grid() -> object:
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for point in (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
        ):
            points.InsertNextPoint(*point)
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, [0, 1, 2, 3, 4, 5, 6, 7])
        grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, [1, 8, 9, 2, 5, 10, 11, 6])
        return grid

    @classmethod
    def concave_l_prism_grid(
        cls, *, reverse_face_index: int | None = None
    ) -> object:
        footprint = ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        point_ids = cls._append_points(
            points,
            [
                (x, y, z)
                for z in (0, 1)
                for x, y in footprint
            ],
        )
        local_faces = [
            tuple(reversed(range(6))),
            tuple(range(6, 12)),
            *(
                (index, (index + 1) % 6, (index + 1) % 6 + 6, index + 6)
                for index in range(6)
            ),
        ]
        if reverse_face_index is not None:
            local_faces[reverse_face_index] = tuple(
                reversed(local_faces[reverse_face_index])
            )
        faces = vtk.vtkCellArray()
        for local_face in local_faces:
            face = [point_ids[index] for index in local_face]
            faces.InsertNextCell(len(face), face)
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.InsertNextCell(
            vtk.VTK_POLYHEDRON,
            len(point_ids),
            point_ids,
            faces,
        )
        return grid

    @staticmethod
    def sample(
        profile_id: str,
        sample_index: int,
        point_m: tuple[float, float, float],
    ) -> VelocitySampleDefinition:
        return VelocitySampleDefinition(
            profile_id=profile_id,
            sample_index=sample_index,
            point_count=2,
            line_fraction=0.0,
            distance_m=float(sample_index),
            point_m=point_m,
        )

    def test_all_five_native_cell_types_and_golden_evidence(self) -> None:
        grid = self.mixed_grid()
        cell_types_before = [grid.GetCellType(index) for index in range(5)]
        points_before = np.asarray(
            [grid.GetPoint(index) for index in range(grid.GetNumberOfPoints())]
        )
        samples = (
            self.sample("tetra", 0, (0.1, 0.1, 0.1)),
            self.sample("hex", 1, (2.5, 0.5, 0.5)),
            self.sample("wedge", 2, (4.2, 0.2, 0.5)),
            self.sample("pyramid", 3, (6.5, 0.5, 0.2)),
            self.sample("polyhedron", 4, (8.5, 0.5, 0.5)),
        )
        assignments = NativeContainingCellKernel(grid).assign(
            samples,
            case_id="synthetic_mixed",
            source_sha256=SOURCE_HASHES,
        )
        self.assertTrue(
            all(isinstance(row, VelocityCellAssignmentEvidence) for row in assignments)
        )
        self.assertEqual([row.raw_vtk_cell_id for row in assignments], list(range(5)))
        self.assertEqual([row.candidate_count for row in assignments], [1] * 5)
        self.assertTrue(all(row.valid and row.reason == "" for row in assignments))
        self.assertEqual(assignment_evidence_sha256(assignments), SYNTHETIC_ASSIGNMENT_SHA256)
        self.assertEqual(
            [grid.GetCellType(index) for index in range(5)], cell_types_before
        )
        np.testing.assert_array_equal(
            [grid.GetPoint(index) for index in range(grid.GetNumberOfPoints())],
            points_before,
        )
        self.assertEqual(grid.GetCellData().GetNumberOfArrays(), 0)

    def test_deterministic_concave_polyhedron_closure(self) -> None:
        kernel = NativeContainingCellKernel(self.concave_l_prism_grid())
        assignments = kernel.assign(
            (
                self.sample("inside", 0, (0.5, 1.5, 0.5)),
                self.sample("aabb_notch", 0, (1.5, 1.5, 0.5)),
                self.sample("half_micrometre", 0, (1.0 + 0.5e-6, 1.5, 0.5)),
                self.sample("two_micrometres", 0, (1.0 + 2.0e-6, 1.5, 0.5)),
            ),
            case_id="synthetic_concave_l_prism",
            source_sha256=SOURCE_HASHES,
        )
        self.assertTrue(assignments[0].valid)
        self.assertEqual(assignments[0].raw_vtk_cell_id, 0)
        self.assertEqual(assignments[0].candidate_count, 1)
        self.assertFalse(assignments[1].valid)
        self.assertEqual(assignments[1].reason, NO_CLOSURE_CELL_REASON)
        self.assertEqual(assignments[1].candidate_count, 0)
        self.assertTrue(assignments[2].valid)
        self.assertEqual(assignments[2].raw_vtk_cell_id, 0)
        self.assertEqual(assignments[2].candidate_count, 1)
        self.assertFalse(assignments[3].valid)
        self.assertEqual(assignments[3].reason, NO_CLOSURE_CELL_REASON)
        self.assertEqual(assignments[3].candidate_count, 0)
        evaluation_audit = kernel.polyhedron_evaluation_audit()
        self.assertEqual(evaluation_audit["broad_phase_polyhedron_visit_count"], 4)
        self.assertEqual(evaluation_audit["boundary_count"], 1)
        self.assertEqual(evaluation_audit["inside_count"], 1)
        self.assertEqual(evaluation_audit["outside_count"], 2)
        self.assertEqual(evaluation_audit["ambiguous_count"], 0)
        self.assertEqual(evaluation_audit["winding_classified_count"], 3)
        self.assertGreaterEqual(
            evaluation_audit[
                "minimum_winding_classification_margin_steradian"
            ],
            0.0,
        )
        self.assertLessEqual(
            evaluation_audit[
                "minimum_winding_classification_margin_steradian"
            ],
            1.0e-3,
        )

    def test_polyhedron_triangulation_is_cached_without_vtk_objects(self) -> None:
        kernel = NativeContainingCellKernel(
            self.concave_l_prism_grid(), query_cache_enabled=False
        )
        with patch.object(
            velocity_assignments_module,
            "_prepare_polyhedron_triangles",
            wraps=velocity_assignments_module._prepare_polyhedron_triangles,
        ) as prepare, patch.object(
            velocity_assignments_module.vtk,
            "vtkGenericCell",
            wraps=vtk.vtkGenericCell,
        ) as cell_factory:
            assignments = kernel.assign(
                (
                    self.sample("inside_first", 0, (0.5, 1.5, 0.5)),
                    self.sample("inside_second", 0, (0.5, 1.4, 0.5)),
                ),
                case_id="synthetic_polyhedron_geometry_cache",
                source_sha256=SOURCE_HASHES,
            )
        self.assertTrue(all(row.valid for row in assignments))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(cell_factory.call_count, 1)
        self.assertEqual(set(kernel._polyhedron_triangle_cache), {0})
        triangles = kernel._polyhedron_triangle_cache[0]
        self.assertIsInstance(triangles, tuple)
        self.assertTrue(all(isinstance(triangle, tuple) for triangle in triangles))
        triangle_count = len(triangles)
        self.assertEqual(
            kernel.polyhedron_geometry_cache_audit(),
            {
                "policy": "deterministic_least_recently_used",
                "maximum_entries": 8_192,
                "maximum_emitted_triangles": 131_072,
                "current_entries": 1,
                "current_emitted_triangles": triangle_count,
                "peak_entries": 1,
                "peak_emitted_triangles": triangle_count,
                "cache_hits": 1,
                "cache_misses": 1,
                "evictions": 0,
                "oversized_entry_bypasses": 0,
                "fail_closed_preparations": 0,
                "vtk_objects_cached": False,
            },
        )

    def test_misoriented_polyhedron_face_fails_closed(self) -> None:
        kernel = NativeContainingCellKernel(
            self.concave_l_prism_grid(reverse_face_index=2),
            query_cache_enabled=False,
        )
        with patch.object(
            velocity_assignments_module,
            "_prepare_polyhedron_triangles",
            wraps=velocity_assignments_module._prepare_polyhedron_triangles,
        ) as prepare:
            assignments = kernel.assign(
                (
                    self.sample("inside_first", 0, (0.5, 1.5, 0.5)),
                    self.sample("inside_second", 0, (0.5, 1.4, 0.5)),
                ),
                case_id="synthetic_misoriented_polyhedron",
                source_sha256=SOURCE_HASHES,
            )
        self.assertEqual(prepare.call_count, 1)
        self.assertIsNone(kernel._polyhedron_triangle_cache[0])
        self.assertEqual(
            kernel.polyhedron_geometry_cache_audit()["fail_closed_preparations"],
            1,
        )
        self.assertEqual(
            kernel.polyhedron_geometry_cache_audit()["cache_hits"], 1
        )
        self.assertEqual(
            kernel.polyhedron_evaluation_audit()["ambiguous_count"], 2
        )
        for assignment in assignments:
            self.assertFalse(assignment.valid)
            self.assertEqual(assignment.raw_vtk_cell_id, None)
            self.assertEqual(assignment.candidate_count, 0)
            self.assertEqual(
                assignment.reason,
                CELL_EVALUATION_FAILURE_REASON_PREFIX + "0",
            )

    def test_polyhedron_geometry_cache_uses_bounded_lru_eviction(self) -> None:
        kernel = NativeContainingCellKernel(self.concave_l_prism_grid())
        kernel._polyhedron_cache_max_entries = 2
        triangle = (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),)
        kernel._cache_polyhedron_triangles(10, triangle)
        kernel._cache_polyhedron_triangles(20, None)
        self.assertEqual(kernel._cached_polyhedron_triangles(10), triangle)
        kernel._cache_polyhedron_triangles(30, triangle)
        self.assertEqual(list(kernel._polyhedron_triangle_cache), [10, 30])
        self.assertIs(
            kernel._cached_polyhedron_triangles(20),
            velocity_assignments_module._CACHE_MISS,
        )
        audit = kernel.polyhedron_geometry_cache_audit()
        self.assertEqual(audit["maximum_entries"], 2)
        self.assertEqual(audit["current_entries"], 2)
        self.assertEqual(audit["current_emitted_triangles"], 2)
        self.assertEqual(audit["peak_entries"], 2)
        self.assertEqual(audit["peak_emitted_triangles"], 2)
        self.assertEqual(audit["cache_hits"], 1)
        self.assertEqual(audit["evictions"], 1)

    def test_polyhedron_geometry_cache_bypasses_oversized_entry(self) -> None:
        kernel = NativeContainingCellKernel(self.concave_l_prism_grid())
        kernel._polyhedron_cache_max_triangles = 1
        triangle = (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        kernel._cache_polyhedron_triangles(10, triangle)
        self.assertEqual(kernel._polyhedron_triangle_cache, {})
        audit = kernel.polyhedron_geometry_cache_audit()
        self.assertEqual(audit["current_entries"], 0)
        self.assertEqual(audit["current_emitted_triangles"], 0)
        self.assertEqual(audit["oversized_entry_bypasses"], 1)

    def test_grid_modification_is_rejected_before_cached_assignment(self) -> None:
        grid = self.concave_l_prism_grid()
        kernel = NativeContainingCellKernel(grid)
        grid.GetPoints().SetPoint(0, -0.25, 0.0, 0.0)
        grid.GetPoints().Modified()
        with self.assertRaisesRegex(
            VelocityAssignmentError,
            "geometry changed after containing-cell kernel construction",
        ):
            kernel.assign(
                (self.sample("inside", 0, (0.5, 1.5, 0.5)),),
                case_id="synthetic_modified_grid",
                source_sha256=SOURCE_HASHES,
            )

    def test_ambiguous_polyhedron_winding_fails_closed(self) -> None:
        kernel = NativeContainingCellKernel(self.concave_l_prism_grid())
        with patch.object(
            velocity_assignments_module,
            "_classify_polyhedron_solid_angle_detail",
            return_value=velocity_assignments_module._PolyhedronClosureEvaluation(
                None, "ambiguous"
            ),
        ):
            assignment = kernel.assign(
                (self.sample("inside", 0, (0.5, 1.5, 0.5)),),
                case_id="synthetic_ambiguous_polyhedron",
                source_sha256=SOURCE_HASHES,
            )[0]
        self.assertFalse(assignment.valid)
        self.assertEqual(assignment.raw_vtk_cell_id, None)
        self.assertEqual(assignment.candidate_count, 0)
        self.assertEqual(
            assignment.reason,
            CELL_EVALUATION_FAILURE_REASON_PREFIX + "0",
        )

    def test_serial_queries_reuse_buffers_but_not_generic_cell_state(self) -> None:
        kernel = NativeContainingCellKernel(
            self.mixed_grid(), query_cache_enabled=False
        )
        broad_ids_identity = id(kernel._broad_ids)
        samples = (
            self.sample("tetra_first", 0, (0.1, 0.1, 0.1)),
            self.sample("tetra_second", 0, (0.2, 0.2, 0.2)),
            self.sample("hex", 0, (2.5, 0.5, 0.5)),
            self.sample("wedge", 0, (4.2, 0.2, 0.5)),
            self.sample("pyramid", 0, (6.5, 0.5, 0.2)),
            self.sample("polyhedron", 0, (8.5, 0.5, 0.5)),
        )
        with patch.object(
            velocity_assignments_module.vtk,
            "vtkGenericCell",
            wraps=vtk.vtkGenericCell,
        ) as cell_factory:
            assignments = kernel.assign(
                samples,
                case_id="synthetic_scratch_reuse",
                source_sha256=SOURCE_HASHES,
            )
        self.assertTrue(all(row.valid for row in assignments))
        self.assertEqual(id(kernel._broad_ids), broad_ids_identity)
        self.assertEqual(cell_factory.call_count, len(samples))
        self.assertEqual(set(kernel._evaluation_scratch), {4, 5, 6, 8})
        for point_count, scratch in kernel._evaluation_scratch.items():
            self.assertEqual(len(scratch[-1]), point_count)

    def test_reused_outputs_are_poisoned_to_detect_partial_vtk_writes(self) -> None:
        kernel = NativeContainingCellKernel(
            self.two_hex_grid(), query_cache_enabled=False
        )
        kernel._closure_candidates((0.5, 0.5, 0.5), 1.0e-6)
        closest, sub_id, parametric, distance_squared, weights = (
            kernel._evaluation_scratch[8]
        )
        closest[:] = [math.nan, math.inf, -math.inf]
        sub_id.set(23)
        parametric[:] = [-math.inf, math.nan, math.inf]
        distance_squared.set(-7.0)
        weights[:] = [math.nan] * 8

        class OneCellLocator:
            @staticmethod
            def FindCellsWithinBounds(_bounds: object, ids: object) -> None:
                ids.InsertNextId(0)

        class NoOutputCell:
            observed: tuple[object, ...] | None = None

            @staticmethod
            def GetCellType() -> int:
                return vtk.VTK_HEXAHEDRON

            @staticmethod
            def GetNumberOfPoints() -> int:
                return 8

            def EvaluatePosition(
                self,
                _point: object,
                output_closest: list[float],
                output_sub_id: object,
                output_parametric: list[float],
                output_distance_squared: object,
                output_weights: list[float],
            ) -> int:
                self.observed = (
                    tuple(output_closest),
                    output_sub_id.get(),
                    tuple(output_parametric),
                    output_distance_squared.get(),
                    tuple(output_weights),
                )
                return 1

        class OneCellGrid:
            @staticmethod
            def GetCell(_raw_id: int, _cell: object) -> None:
                return None

        cell = NoOutputCell()
        kernel.locator = OneCellLocator()
        kernel.grid = OneCellGrid()
        with patch.object(
            velocity_assignments_module.vtk,
            "vtkGenericCell",
            return_value=cell,
        ):
            candidates, failures = kernel._closure_candidates(
                (0.5, 0.5, 0.5), 1.0e-6
            )

        self.assertEqual(candidates, ())
        self.assertEqual(failures, (0,))
        self.assertEqual(cell.observed[1], 0)
        for values in (cell.observed[0], cell.observed[2], cell.observed[4]):
            self.assertTrue(all(math.isnan(value) for value in values))
        self.assertTrue(math.isnan(cell.observed[3]))

    def test_malformed_evaluate_position_outputs_are_retained_as_failures(self) -> None:
        kernel = NativeContainingCellKernel(
            self.two_hex_grid(), query_cache_enabled=False
        )

        class OneCellLocator:
            @staticmethod
            def FindCellsWithinBounds(_bounds: object, ids: object) -> None:
                ids.InsertNextId(0)

        class OneCellGrid:
            @staticmethod
            def GetCell(_raw_id: int, _cell: object) -> None:
                return None

        class MalformedOutputCell:
            def __init__(self, mode: str) -> None:
                self.mode = mode

            @staticmethod
            def GetCellType() -> int:
                return vtk.VTK_HEXAHEDRON

            @staticmethod
            def GetNumberOfPoints() -> int:
                return 8

            def EvaluatePosition(
                self,
                _point: object,
                output_closest: list[float],
                _output_sub_id: object,
                output_parametric: list[float],
                output_distance_squared: object,
                output_weights: list[float],
            ) -> int:
                output_closest[:] = [0.0, 0.0, 0.0]
                output_parametric[:] = [0.5, 0.5, 0.5]
                output_weights[:] = [0.125] * 8
                if self.mode == "negative_distance":
                    output_distance_squared.set(-1.0)
                elif self.mode == "nonfinite_output":
                    output_distance_squared.set(0.0)
                    output_closest[0] = math.nan
                elif self.mode == "inside_nonzero_distance":
                    output_distance_squared.set(1.0e-20)
                    return 1
                else:  # pragma: no cover - closed by the subtest inputs.
                    raise AssertionError(self.mode)
                return 0

        kernel.locator = OneCellLocator()
        kernel.grid = OneCellGrid()
        for mode in (
            "negative_distance",
            "nonfinite_output",
            "inside_nonzero_distance",
        ):
            with self.subTest(mode=mode):
                with patch.object(
                    velocity_assignments_module.vtk,
                    "vtkGenericCell",
                    return_value=MalformedOutputCell(mode),
                ):
                    candidates, failures = kernel._closure_candidates(
                        (0.5, 0.5, 0.5), 1.0e-6
                    )
                self.assertEqual(candidates, ())
                self.assertEqual(failures, (0,))

    def test_cell_tree_broad_phase_uses_exact_native_cell_aabb_candidates(self) -> None:
        kernel = NativeContainingCellKernel(self.mixed_grid())
        assignment = kernel.assign(
            (self.sample("tetra_only", 0, (0.1, 0.1, 0.1)),),
            case_id="synthetic_exact_aabb",
            source_sha256=SOURCE_HASHES,
        )[0]
        self.assertTrue(kernel.locator.IsA("vtkCellTreeLocator"))
        self.assertTrue(kernel.locator.GetAutomatic())
        self.assertEqual(kernel.locator.GetNumberOfCellsPerNode(), 8)
        self.assertEqual(kernel.locator.GetNumberOfBuckets(), 6)
        self.assertTrue(kernel.locator.GetCacheCellBounds())
        self.assertEqual(
            [
                kernel._broad_ids.GetId(index)
                for index in range(kernel._broad_ids.GetNumberOfIds())
            ],
            [0],
        )
        self.assertTrue(assignment.valid)
        self.assertEqual(assignment.raw_vtk_cell_id, 0)
        self.assertEqual(assignment.candidate_count, 1)
        self.assertEqual(set(kernel._evaluation_scratch), {4})

    def test_shared_face_enumerates_both_and_selects_smallest_raw_id(self) -> None:
        kernel = NativeContainingCellKernel(self.two_hex_grid())
        assignments = kernel.assign(
            (self.sample("face", 0, (1.0, 0.5, 0.5)),),
            case_id="synthetic_face",
            source_sha256=SOURCE_HASHES,
        )
        self.assertEqual(assignments[0].candidate_count, 2)
        self.assertEqual(assignments[0].raw_vtk_cell_id, 0)

    def test_input_order_is_preserved(self) -> None:
        kernel = NativeContainingCellKernel(self.two_hex_grid())
        samples = (
            self.sample("second_cell_first", 7, (1.5, 0.5, 0.5)),
            self.sample("first_cell_second", 3, (0.5, 0.5, 0.5)),
        )
        assignments = kernel.assign(
            samples,
            case_id="synthetic_order",
            source_sha256=SOURCE_HASHES,
        )
        self.assertEqual(
            [(row.profile_id, row.sample_index) for row in assignments],
            [("second_cell_first", 7), ("first_cell_second", 3)],
        )
        self.assertEqual([row.raw_vtk_cell_id for row in assignments], [1, 0])

    def test_cached_and_uncached_assignments_and_hashes_are_identical(self) -> None:
        samples = (
            self.sample("inside_first", 0, (0.5, 0.5, 0.5)),
            self.sample("inside_duplicate", 0, (0.5, 0.5, 0.5)),
            self.sample("outside_first", 0, (-2.0, 0.5, 0.5)),
            self.sample("outside_duplicate", 0, (-2.0, 0.5, 0.5)),
        )
        cached_kernel = NativeContainingCellKernel(
            self.two_hex_grid(), query_cache_enabled=True
        )
        uncached_kernel = NativeContainingCellKernel(
            self.two_hex_grid(), query_cache_enabled=False
        )
        cached = cached_kernel.assign(
            samples,
            case_id="synthetic_cached",
            source_sha256=SOURCE_HASHES,
        )
        uncached = uncached_kernel.assign(
            samples,
            case_id="synthetic_cached",
            source_sha256=SOURCE_HASHES,
        )
        self.assertEqual(cached, uncached)
        self.assertEqual(
            assignment_evidence_sha256(cached),
            assignment_evidence_sha256(uncached),
        )
        self.assertEqual(
            cached_kernel.query_cache_audit(),
            {
                "enabled": True,
                "key_id": QUERY_CACHE_KEY_ID,
                "total_rows": 4,
                "unique_query_keys": 2,
                "cache_hits": 2,
            },
        )
        self.assertEqual(
            uncached_kernel.query_cache_audit(),
            {
                "enabled": False,
                "key_id": QUERY_CACHE_KEY_ID,
                "total_rows": 4,
                "unique_query_keys": 2,
                "cache_hits": 0,
            },
        )

    def test_cache_retains_failure_rows_and_uses_exact_xyz_tolerance_keys(self) -> None:
        exact = (0.5, 0.5, 0.5)
        adjacent = (math.nextafter(0.5, math.inf), 0.5, 0.5)
        samples = (
            self.sample("exact_first", 0, exact),
            self.sample("exact_duplicate", 0, exact),
            self.sample("adjacent_binary64", 0, adjacent),
        )
        cached_kernel = NativeContainingCellKernel(self.two_hex_grid())
        with patch.object(
            cached_kernel,
            "_closure_candidates",
            return_value=((0,), (1,)),
        ) as cached_query:
            cached = cached_kernel.assign(
                samples,
                case_id="synthetic_failure_cache",
                source_sha256=SOURCE_HASHES,
            )
            cached_kernel.assign(
                (self.sample("different_tolerance", 0, exact),),
                case_id="synthetic_failure_cache",
                source_sha256=SOURCE_HASHES,
                tolerance_m=2.0e-6,
            )
        self.assertEqual(cached_query.call_count, 3)
        self.assertTrue(all(not row.valid for row in cached))
        self.assertTrue(all(row.raw_vtk_cell_id is None for row in cached))
        self.assertTrue(all(row.candidate_count == 1 for row in cached))
        self.assertTrue(
            all(
                row.reason == CELL_EVALUATION_FAILURE_REASON_PREFIX + "1"
                for row in cached
            )
        )
        self.assertEqual(
            cached_kernel.query_cache_audit(),
            {
                "enabled": True,
                "key_id": QUERY_CACHE_KEY_ID,
                "total_rows": 4,
                "unique_query_keys": 3,
                "cache_hits": 1,
            },
        )

        uncached_kernel = NativeContainingCellKernel(
            self.two_hex_grid(), query_cache_enabled=False
        )
        with patch.object(
            uncached_kernel,
            "_closure_candidates",
            return_value=((0,), (1,)),
        ) as uncached_query:
            uncached = uncached_kernel.assign(
                samples,
                case_id="synthetic_failure_cache",
                source_sha256=SOURCE_HASHES,
            )
        self.assertEqual(uncached_query.call_count, 3)
        self.assertEqual(cached, uncached)
        self.assertEqual(
            assignment_evidence_sha256(cached),
            assignment_evidence_sha256(uncached),
        )

    def test_invalid_rows_are_retained_with_owner_reasons(self) -> None:
        kernel = NativeContainingCellKernel(self.two_hex_grid())
        samples = (
            self.sample("outside", 0, (-2.0, 0.5, 0.5)),
            self.sample("solid", 0, (4.0, 0.5, 0.5)),
            self.sample("masked_inside", 0, (0.5, 0.5, 0.5)),
            self.sample("unclassified", 0, (5.0, 0.5, 0.5)),
        )
        invalid_reasons = {
            ("outside", 0): "outside_released_fluid_domain",
            ("solid", 0): "inside_morphed_solid",
            ("masked_inside", 0): "inside_morphed_solid",
        }
        assignments = kernel.assign(
            samples,
            case_id="synthetic_invalid",
            source_sha256=SOURCE_HASHES,
            invalid_reasons=invalid_reasons,
        )
        self.assertEqual(
            [row.reason for row in assignments],
            [
                "outside_released_fluid_domain",
                "inside_morphed_solid",
                "inside_morphed_solid",
                NO_CLOSURE_CELL_REASON,
            ],
        )
        self.assertEqual([row.candidate_count for row in assignments], [0, 0, 1, 0])
        self.assertTrue(all(not row.valid for row in assignments))
        self.assertTrue(all(row.raw_vtk_cell_id is None for row in assignments))

    def test_evaluate_position_failure_is_retained_as_invalid(self) -> None:
        kernel = NativeContainingCellKernel(self.two_hex_grid())
        sample = self.sample("evaluation_failure", 0, (0.5, 0.5, 0.5))
        with patch.object(
            kernel,
            "_closure_candidates",
            return_value=((0,), (1,)),
        ):
            assignment = kernel.assign(
                (sample,),
                case_id="synthetic_evaluation_failure",
                source_sha256=SOURCE_HASHES,
            )[0]
        self.assertFalse(assignment.valid)
        self.assertIsNone(assignment.raw_vtk_cell_id)
        self.assertEqual(assignment.candidate_count, 1)
        self.assertEqual(
            assignment.reason,
            CELL_EVALUATION_FAILURE_REASON_PREFIX + "1",
        )

    def test_half_one_two_micrometre_replay_is_tolerance_sensitive(self) -> None:
        kernel = NativeContainingCellKernel(self.two_hex_grid())
        # 1.5 micrometres from the x=0 face: accepted only at 2 micrometres.
        sample = self.sample("tolerance", 0, (-1.5e-6, 0.5, 0.5))
        replay = replay_assignment_tolerances(
            kernel,
            (sample,),
            case_id="synthetic_tolerance",
            source_sha256=SOURCE_HASHES,
        )
        self.assertEqual(replay.tolerances_m, TOLERANCE_REPLAY_M)
        self.assertEqual(
            [rows[0].valid for rows in replay.assignments_by_tolerance],
            [False, False, True],
        )
        self.assertEqual(
            [rows[0].raw_vtk_cell_id for rows in replay.assignments_by_tolerance],
            [None, None, 0],
        )
        self.assertEqual(
            [rows[0].candidate_count for rows in replay.assignments_by_tolerance],
            [0, 0, 1],
        )
        self.assertTrue(
            all(
                isinstance(rows[0], VelocityCellAssignmentEvidence)
                for rows in replay.assignments_by_tolerance
            )
        )
        receipt = replay.as_dict()
        self.assertEqual(receipt, replay.as_dict())
        self.assertEqual(receipt["status"], "candidate_not_frozen")
        self.assertEqual(receipt["kernel"]["versions"]["vtk"], REQUIRED_VTK_VERSION)
        self.assertEqual(
            [row["valid_count"] for row in receipt["tolerances"]], [0, 0, 1]
        )

    def test_unsupported_and_nonfinite_geometry_fail_closed(self) -> None:
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for point in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)):
            points.InsertNextPoint(*point)
        unsupported = vtk.vtkUnstructuredGrid()
        unsupported.SetPoints(points)
        unsupported.InsertNextCell(vtk.VTK_QUAD, 4, [0, 1, 2, 3])
        with self.assertRaisesRegex(VelocityAssignmentError, "unsupported native cell type"):
            NativeContainingCellKernel(unsupported)

        nonfinite_points = vtk.vtkPoints()
        nonfinite_points.SetDataTypeToDouble()
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, math.nan)):
            nonfinite_points.InsertNextPoint(*point)
        nonfinite = vtk.vtkUnstructuredGrid()
        nonfinite.SetPoints(nonfinite_points)
        nonfinite.InsertNextCell(vtk.VTK_TETRA, 4, [0, 1, 2, 3])
        with self.assertRaisesRegex(VelocityAssignmentError, "non-finite coordinates"):
            NativeContainingCellKernel(nonfinite)

    def test_duplicate_samples_and_bad_owner_mask_fail_closed(self) -> None:
        kernel = NativeContainingCellKernel(self.two_hex_grid())
        sample = self.sample("duplicate", 0, (0.5, 0.5, 0.5))
        with self.assertRaisesRegex(VelocityAssignmentError, "duplicate"):
            kernel.assign(
                (sample, sample),
                case_id="synthetic",
                source_sha256=SOURCE_HASHES,
            )
        with self.assertRaisesRegex(VelocityAssignmentError, "unexpected sample key"):
            kernel.assign(
                (sample,),
                case_id="synthetic",
                source_sha256=SOURCE_HASHES,
                invalid_reasons={("missing", 0): "inside_morphed_solid"},
            )
        with self.assertRaisesRegex(VelocityAssignmentError, "unsupported owner"):
            kernel.assign(
                (sample,),
                case_id="synthetic",
                source_sha256=SOURCE_HASHES,
                invalid_reasons={("duplicate", 0): "guessed_reason"},
            )

    def test_runtime_receipt_pins_exact_vtk_and_settings_hash(self) -> None:
        receipt = candidate_kernel_receipt()
        self.assertEqual(receipt["versions"]["vtk"], REQUIRED_VTK_VERSION)
        self.assertEqual(receipt["versions"]["vtk_source"], "vtk version 9.5.2")
        self.assertEqual(receipt["settings"], candidate_kernel_settings())
        self.assertEqual(receipt["settings_sha256"], KERNEL_SETTINGS_SHA256)


class DrivAerMLVelocityAssignmentOptionalDependencyTests(unittest.TestCase):
    def test_python_and_numpy_mismatches_fail_before_vtk(self) -> None:
        cases = (
            ("python", "Python 3.12.13, got 3.12.12"),
            ("numpy", "NumPy 2.2.6, got 2.2.5"),
        )
        for mismatch, expected in cases:
            with self.subTest(mismatch=mismatch), patch.object(
                velocity_assignments_module.platform,
                "python_version",
                return_value=(
                    "3.12.12" if mismatch == "python" else REQUIRED_PYTHON_VERSION
                ),
            ), patch.object(
                velocity_assignments_module.np,
                "__version__",
                "2.2.5" if mismatch == "numpy" else REQUIRED_NUMPY_VERSION,
            ), patch(
                "reference.drivaerml.velocity_assignments._require_vtk"
            ) as vtk_gate:
                with self.assertRaisesRegex(VelocityAssignmentError, expected):
                    _require_frozen_runtime()
                vtk_gate.assert_not_called()

    def test_exact_python_and_numpy_reach_vtk_gate(self) -> None:
        with patch.object(
            velocity_assignments_module.platform,
            "python_version",
            return_value=REQUIRED_PYTHON_VERSION,
        ), patch.object(
            velocity_assignments_module.np,
            "__version__",
            REQUIRED_NUMPY_VERSION,
        ), patch(
            "reference.drivaerml.velocity_assignments._require_vtk"
        ) as vtk_gate:
            _require_frozen_runtime()
        vtk_gate.assert_called_once_with()

    def test_vtk_source_identity_mismatch_fails_closed(self) -> None:
        fake_vtk = Mock()
        fake_vtk.vtkVersion.GetVTKVersion.return_value = REQUIRED_VTK_VERSION
        fake_vtk.vtkVersion.GetVTKSourceVersion.return_value = (
            "vtk version 9.5.2-custom"
        )
        with patch.object(
            velocity_assignments_module, "vtk", fake_vtk
        ), patch.object(
            velocity_assignments_module, "vtk_to_numpy", object()
        ), self.assertRaisesRegex(VelocityAssignmentError, "VTK source identity"):
            velocity_assignments_module._require_vtk()

    def test_optional_dependency_probe_is_boolean(self) -> None:
        self.assertIsInstance(vtk_available(), bool)


if __name__ == "__main__":
    unittest.main()
