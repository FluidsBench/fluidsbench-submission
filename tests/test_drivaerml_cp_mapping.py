from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reference.drivaerml.autocfd5 import (
    CpComponentRule,
    CpProbeDefinition,
    CpProbeMappingEvidence,
)
from reference.drivaerml.cp_mapping import (
    ArrayPolygonLocator,
    AsciiStlTriangleStream,
    CpMappingError,
    RetainedVerifiedRegularFile,
    StlProjectionResult,
    Vtk95VtpPolygonLocator,
    bridge_projection_to_native,
    closest_point_on_triangle_cut,
    map_cp_probes_for_case,
    project_probes_to_case_stl,
)


SOURCE_HASHES = ("0" * 64, "1" * 64)


def _stl(solid_triangles: list[tuple[str, list[list[list[float]]]]]) -> str:
    lines: list[str] = []
    for solid, triangles in solid_triangles:
        lines.append(f"solid {solid}")
        for triangle in triangles:
            lines.extend(("  facet normal 9 9 9", "    outer loop"))
            for vertex in triangle:
                lines.append("      vertex " + " ".join(str(value) for value in vertex))
            lines.extend(("    endloop", "  endfacet"))
        lines.append(f"endsolid {solid}")
    return "\n".join(lines) + "\n"


def _probe(probe_id: int, point: tuple[float, float, float]) -> CpProbeDefinition:
    return CpProbeDefinition(
        autocfd_probe_id=probe_id,
        station_id=f"station_{probe_id}",
        ansa_pid=600,
        station_point_index=probe_id,
        point_m=point,
    )


def _rule(
    probe_id: int,
    component: str,
    mode: str = "component_closest_3d",
    *,
    axis: str | None = None,
    value: float | None = None,
) -> CpComponentRule:
    return CpComponentRule(
        autocfd_probe_id=probe_id,
        drivaerml_component=component,
        projection_mode=mode,
        cut_axis=axis,
        cut_value_m=value,
        owner_review_status="pending_visual_signoff",
    )


def _projection(
    *,
    valid: bool = True,
    reason: str = "",
    mapped: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] | None = (0.0, 0.0, 1.0),
) -> StlProjectionResult:
    return StlProjectionResult(
        autocfd_probe_id=1,
        nominal_point_m=(0.0, 0.0, 0.0),
        drivaerml_component="Body",
        projection_mode="component_closest_3d",
        cut_axis=None,
        cut_value_m=None,
        owner_review_status="pending_visual_signoff",
        valid=valid,
        reason=reason,
        mapped_point_m=mapped,
        raw_stl_triangle_id=4 if mapped is not None else None,
        stl_unit_normal=normal,
        nominal_displacement_m=0.0 if mapped is not None else None,
        component_facet_count=8,
        projection_candidate_count=8,
        review_flags=(),
    )


class DrivAerMLCpMappingTests(unittest.TestCase):
    def test_retained_regular_file_survives_path_replacement_and_detects_mutation(self) -> None:
        payload = b"retained source identity\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.dat"
            path.write_bytes(payload)
            with RetainedVerifiedRegularFile.open(
                path, label="test source"
            ) as retained:
                self.assertEqual(retained.sha256(), hashlib.sha256(payload).hexdigest())
                descriptor_path = retained.descriptor_path()
                self.assertIn(
                    descriptor_path.parent,
                    {Path("/proc/self/fd"), Path("/dev/fd")},
                )
                path.replace(Path(directory) / "old-source.dat")
                path.write_bytes(b"replacement pathname\n")
                self.assertEqual(descriptor_path.read_bytes(), payload)

            stable = Path(directory) / "stable.dat"
            stable.write_bytes(payload)
            with RetainedVerifiedRegularFile.open(
                stable, label="stable source"
            ) as retained:
                retained.sha256()
                retained.descriptor_path()
                retained.finalize_verification()
                receipt = retained.verification_receipt(consumers=("test_parser",))
                self.assertEqual(receipt["post_read_fstat"], "unchanged")
                self.assertEqual(receipt["parser_inputs"], ["test_parser"])

            mutable = Path(directory) / "mutable.dat"
            mutable.write_bytes(payload)
            with RetainedVerifiedRegularFile.open(
                mutable, label="mutable source"
            ) as retained:
                retained.sha256()
                retained.descriptor_path()
                mutable.write_bytes(payload + b"changed\n")
                with self.assertRaisesRegex(
                    CpMappingError, "retained mutable source .* changed"
                ):
                    retained.finalize_verification()

    def test_retained_source_rejects_non_regular_file_and_unfinalized_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_directory = Path(directory) / "not-a-file"
            source_directory.mkdir()
            with self.assertRaisesRegex(CpMappingError, "not a regular file"):
                RetainedVerifiedRegularFile.open(
                    source_directory, label="test source"
                )

            path = Path(directory) / "source.dat"
            path.write_bytes(b"source\n")
            with RetainedVerifiedRegularFile.open(path, label="test source") as retained:
                retained.descriptor_path()
                with self.assertRaisesRegex(
                    CpMappingError, "has not completed post-read verification"
                ):
                    retained.verification_receipt(consumers=("test_parser",))

    def test_ascii_stream_global_raw_order_component_restriction_and_stl_tie(self) -> None:
        other = [[-1, -1, 0.0009], [1, -1, 0.0009], [-1, 1, 0.0009]]
        wanted_earlier = [[-1, -1, 0.0], [1, -1, 0.0], [-1, 1, 0.0]]
        wanted_later = [
            [-1, -1, 0.0000005],
            [1, -1, 0.0000005],
            [-1, 1, 0.0000005],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.stl"
            path.write_text(
                _stl([("Other", [other]), ("Wanted", [wanted_earlier, wanted_later])]),
                encoding="ascii",
            )
            stream = AsciiStlTriangleStream(path, chunk_facets=1)
            chunks = list(stream)
            self.assertEqual([chunk.solid_name for chunk in chunks], ["Other", "Wanted", "Wanted"])
            self.assertEqual(
                [int(chunk.raw_facet_ids[0]) for chunk in chunks], [0, 1, 2]
            )
            self.assertEqual(stream.solid_names, ("Other", "Wanted"))
            self.assertEqual(stream.facet_count, 3)
            self.assertRegex(stream.sha256 or "", r"^[0-9a-f]{64}$")

            projection = project_probes_to_case_stl(
                path,
                [_probe(1, (0.0, 0.0, 0.001))],
                [_rule(1, "Wanted")],
                chunk_facets=1,
            )[0]
        self.assertTrue(projection.valid)
        # Facet 0 is geometrically closer but belongs to a forbidden solid.
        # Facets 1 and 2 differ by only 0.5 micrometre, so raw facet 1 wins.
        self.assertEqual(projection.raw_stl_triangle_id, 1)
        np.testing.assert_allclose(projection.mapped_point_m, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(projection.stl_unit_normal, [0.0, 0.0, 1.0])

    def test_cut_crossing_coplanar_and_one_nanometre_vertex_classification(self) -> None:
        crossing = np.asarray([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        point = closest_point_on_triangle_cut(
            (0.0, 0.25, 0.5), crossing, cut_axis="x", cut_value_m=0.0
        )
        np.testing.assert_allclose(point, [0.0, 0.25, 0.0], atol=1.0e-15)

        coplanar = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        point = closest_point_on_triangle_cut(
            (0.0, 0.25, 0.25), coplanar, cut_axis="x", cut_value_m=0.0
        )
        np.testing.assert_allclose(point, [0.0, 0.25, 0.25], atol=1.0e-15)

        near_plane_vertex = np.asarray(
            [[0.5e-9, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float
        )
        point = closest_point_on_triangle_cut(
            (0.0, 0.0, 0.1),
            near_plane_vertex,
            cut_axis="x",
            cut_value_m=0.0,
        )
        np.testing.assert_array_equal(point, near_plane_vertex[0])

    def test_all_rule_modes_and_invalid_component_are_retained_end_to_end(self) -> None:
        crossing = [[-1, 0, 0], [1, 0, 0], [0, 1, 0]]
        closest = [[-1, -1, 0], [1, -1, 0], [-1, 1, 0]]
        native = [[-2, -2, 0.001], [2, -2, 0.001], [0, 2, 0.001]]
        probes = (
            _probe(1, (0.0, 0.25, 0.1)),
            _probe(2, (0.0, 0.0, 0.1)),
            _probe(3, (0.0, 0.0, 0.1)),
        )
        rules = (
            _rule(1, "Cut", "cut_plane_closest", axis="x", value=0.0),
            _rule(2, "Closest"),
            _rule(3, "Absent"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.stl"
            path.write_text(
                _stl([("Cut", [crossing]), ("Closest", [closest])]),
                encoding="ascii",
            )
            records = map_cp_probes_for_case(
                "run_1",
                path,
                ArrayPolygonLocator([native]),
                probes,
                rules,
                source_sha256=SOURCE_HASHES,
                chunk_facets=1,
            )
        self.assertEqual(len(records), 3)
        self.assertTrue(records[0].valid)
        self.assertTrue(records[1].valid)
        self.assertFalse(records[2].valid)
        self.assertEqual(records[2].reason, "declared_component_absent")
        self.assertEqual(records[0].projection_mode, "cut_plane_closest")
        self.assertEqual(records[1].projection_mode, "component_closest_3d")
        for record in records:
            self.assertIsInstance(record.to_evidence(), CpProbeMappingEvidence)

    def test_native_distance_tie_then_normal_tie_then_smallest_raw_id(self) -> None:
        earlier_farther = [
            [-0.01, -0.01, 0.0010005],
            [0.01, -0.01, 0.0010005],
            [-0.01, 0.01, 0.0010005],
        ]
        later_nearer = [
            [-0.01, -0.01, 0.0010],
            [0.01, -0.01, 0.0010],
            [-0.01, 0.01, 0.0010],
        ]
        record = bridge_projection_to_native(
            "run_1",
            _projection(),
            ArrayPolygonLocator([earlier_farther, later_nearer]),
            source_sha256=SOURCE_HASHES,
        )
        self.assertTrue(record.valid)
        # Both distances are within the one-micrometre tie and normals tie, so
        # the smaller raw polygon ID wins even though it is farther away.
        self.assertEqual(record.raw_vtk_polygon_id, 0)
        self.assertAlmostEqual(record.bridge_distance_m or 0.0, 0.0010005)
        self.assertIn("native_bridge_distance_gt_0p5mm", record.review_flags)

    def test_native_distance_tie_prefers_higher_normal_agreement(self) -> None:
        angle = np.deg2rad(20.0)
        slope = np.tan(angle)
        tilted = [
            [-0.01, -0.01, 0.0010 - 0.01 * slope],
            [0.01, -0.01, 0.0010 - 0.01 * slope],
            [0.01, 0.01, 0.0010 + 0.01 * slope],
            [-0.01, 0.01, 0.0010 + 0.01 * slope],
        ]
        horizontal = [
            [-0.01, -0.01, 0.0009401],
            [0.01, -0.01, 0.0009401],
            [-0.01, 0.01, 0.0009401],
        ]
        record = bridge_projection_to_native(
            "run_1",
            _projection(),
            ArrayPolygonLocator([tilted, horizontal]),
            source_sha256=SOURCE_HASHES,
        )
        self.assertTrue(record.valid)
        self.assertEqual(record.raw_vtk_polygon_id, 1)
        self.assertAlmostEqual(record.bridge_abs_normal_dot or 0.0, 1.0)

    def test_native_distance_normal_and_degeneracy_failures_remain_rows(self) -> None:
        vertical = [
            [0.001, -0.001, -0.001],
            [0.001, 0.001, -0.001],
            [0.001, -0.001, 0.001],
        ]
        normal_failure = bridge_projection_to_native(
            "run_1",
            _projection(),
            ArrayPolygonLocator([vertical]),
            source_sha256=SOURCE_HASHES,
        )
        self.assertFalse(normal_failure.valid)
        self.assertEqual(
            normal_failure.reason, "native_bridge_normal_agreement_below_cos30"
        )

        corner = [
            [0.0015, 0.0015, 0.0],
            [0.0019, 0.0015, 0.0],
            [0.0015, 0.0019, 0.0],
        ]
        distance_failure = bridge_projection_to_native(
            "run_1",
            _projection(),
            ArrayPolygonLocator([corner]),
            source_sha256=SOURCE_HASHES,
        )
        self.assertFalse(distance_failure.valid)
        self.assertEqual(distance_failure.reason, "native_bridge_distance_exceeds_2mm")

        degenerate = [[0.0, 0.0, 0.001], [0.001, 0.0, 0.001], [0.002, 0.0, 0.001]]
        degenerate_failure = bridge_projection_to_native(
            "run_1",
            _projection(),
            ArrayPolygonLocator([degenerate]),
            source_sha256=SOURCE_HASHES,
        )
        self.assertFalse(degenerate_failure.valid)
        self.assertEqual(
            degenerate_failure.reason, "no_finite_nondegenerate_native_polygon"
        )

    def test_nominal_displacement_review_and_hard_gate(self) -> None:
        review_triangle = [[-1, -1, 0.22], [1, -1, 0.22], [-1, 1, 0.22]]
        invalid_triangle = [[-1, -1, 0.30], [1, -1, 0.30], [-1, 1, 0.30]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.stl"
            path.write_text(
                _stl([("Review", [review_triangle]), ("Invalid", [invalid_triangle])]),
                encoding="ascii",
            )
            results = project_probes_to_case_stl(
                path,
                [_probe(1, (0.0, 0.0, 0.0)), _probe(2, (0.0, 0.0, 0.0))],
                [_rule(1, "Review"), _rule(2, "Invalid")],
            )
        self.assertTrue(results[0].valid)
        self.assertIn(
            "nominal_displacement_gt_7p5pct_wheelbase", results[0].review_flags
        )
        self.assertFalse(results[1].valid)
        self.assertEqual(
            results[1].reason, "nominal_displacement_exceeds_10pct_wheelbase"
        )

    def test_degenerate_stl_and_malformed_ascii_fail_explicitly(self) -> None:
        degenerate = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "degenerate.stl"
            path.write_text(_stl([("Body", [degenerate])]), encoding="ascii")
            projection = project_probes_to_case_stl(
                path, [_probe(1, (0.0, 0.0, 0.0))], [_rule(1, "Body")]
            )[0]
            self.assertFalse(projection.valid)
            self.assertEqual(
                projection.reason,
                "declared_component_has_no_finite_nondegenerate_triangle",
            )

            malformed = Path(directory) / "malformed.stl"
            malformed.write_text(
                "solid Body\nfacet normal 0 0 1\nvertex 0 0 0\nendfacet\nendsolid Body\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(CpMappingError, "three vertices"):
                list(AsciiStlTriangleStream(malformed))


def _vtk_95_available() -> bool:
    if importlib.util.find_spec("vtkmodules") is None:
        return False
    from vtkmodules.vtkCommonCore import vtkVersion

    return vtkVersion.GetVTKVersion() == "9.5.2"


@unittest.skipUnless(_vtk_95_available(), "optional exact VTK 9.5.2 is unavailable")
class DrivAerMLCpMappingVtkTests(unittest.TestCase):
    def test_vtk_adapter_returns_native_raw_polygon_ids_and_connectivity(self) -> None:
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

        points = vtkPoints()
        for point in ((-1.0, -1.0, 0.001), (1.0, -1.0, 0.001), (-1.0, 1.0, 0.001)):
            points.InsertNextPoint(*point)
        polygons = vtkCellArray()
        polygons.InsertNextCell(3)
        for point_id in range(3):
            polygons.InsertCellPoint(point_id)
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.vtp"
            writer = vtkXMLPolyDataWriter()
            writer.SetFileName(str(path))
            writer.SetInputData(polydata)
            self.assertEqual(writer.Write(), 1)
            with RetainedVerifiedRegularFile.open(
                path, label="native boundary VTP"
            ) as retained:
                retained.sha256()
                descriptor_path = retained.descriptor_path()
                locator = Vtk95VtpPolygonLocator(descriptor_path)
                self.assertEqual(locator.polygon_count, 1)
                self.assertEqual(locator.candidate_polygon_ids((0, 0, 0), 0.002), (0,))
                np.testing.assert_allclose(
                    locator.polygon_vertices(0),
                    [[-1, -1, 0.001], [1, -1, 0.001], [-1, 1, 0.001]],
                )
                retained.finalize_verification()


if __name__ == "__main__":
    unittest.main()
