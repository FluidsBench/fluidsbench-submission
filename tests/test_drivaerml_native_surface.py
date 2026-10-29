from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import reference.drivaerml.native_surface as native_surface_module

from reference.drivaerml.native_surface import (
    REQUIRED_VTK_VERSION,
    DrivAerNativeSurfaceError,
    audit_fixed_surface_area_file,
    evaluate_native_surface_predictions,
    load_native_surface_vtp,
    sha256_file,
    vtk_available,
)
from reference.drivaerml.surface_forces import (
    finalize_force_coefficients,
    force_moment_chunk,
    surface_geometry_chunk_validated,
)


VTK_READY = vtk_available()
if VTK_READY:
    import vtk


class DrivAerMLNativeSurfaceOptionalDependencyTests(unittest.TestCase):
    def test_optional_dependency_probe_is_boolean(self) -> None:
        self.assertIsInstance(vtk_available(), bool)


@unittest.skipUnless(VTK_READY, f"optional VTK {REQUIRED_VTK_VERSION} is unavailable")
class DrivAerMLNativeSurfaceTests(unittest.TestCase):
    @staticmethod
    def make_surface(
        *,
        include_line: bool = False,
        include_shear: bool = True,
        shear_components: int = 3,
        nonfinite_pressure: bool = False,
    ) -> object:
        points = vtk.vtkPoints()
        for coordinate in (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (2, 0, 1),
            (2, 1, 1),
            (0, 1, 1),
        ):
            points.InsertNextPoint(*coordinate)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(4, [0, 1, 2, 3])
        polygons.InsertNextCell(4, [4, 5, 6, 7])
        polydata.SetPolys(polygons)
        if include_line:
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(2, [0, 1])
            polydata.SetLines(lines)

        cell_count = polydata.GetNumberOfCells()
        pressure = vtk.vtkFloatArray()
        pressure.SetName("pMeanTrim")
        pressure.SetNumberOfComponents(1)
        pressure.SetNumberOfTuples(cell_count)
        for index in range(cell_count):
            pressure.SetValue(index, float("nan") if nonfinite_pressure else 2.0 + index)
        polydata.GetCellData().AddArray(pressure)

        if include_shear:
            shear = vtk.vtkFloatArray()
            shear.SetName("wallShearStressMeanTrim")
            shear.SetNumberOfComponents(shear_components)
            shear.SetNumberOfTuples(cell_count)
            for cell_index in range(cell_count):
                for component in range(shear_components):
                    shear.SetComponent(
                        cell_index,
                        component,
                        0.1 * (cell_index + 1) * (component + 1),
                    )
            polydata.GetCellData().AddArray(shear)

        distractor = vtk.vtkDoubleArray()
        distractor.SetName("cell_array_that_must_not_load")
        distractor.SetNumberOfTuples(cell_count)
        distractor.Fill(9.0)
        polydata.GetCellData().AddArray(distractor)
        point_distractor = vtk.vtkDoubleArray()
        point_distractor.SetName("point_array_that_must_not_load")
        point_distractor.SetNumberOfTuples(points.GetNumberOfPoints())
        point_distractor.Fill(7.0)
        polydata.GetPointData().AddArray(point_distractor)
        return polydata

    @staticmethod
    def write_vtp(polydata: object, path: Path) -> None:
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(polydata)
        writer.SetDataModeToAppended()
        writer.SetCompressorTypeToNone()
        if writer.Write() != 1:
            raise AssertionError("synthetic VTP writer failed")

    def load_fixture(self, directory: Path) -> tuple[object, object]:
        directory.mkdir(parents=True, exist_ok=True)
        boundary = directory / "boundary_1.vtp"
        self.write_vtp(self.make_surface(), boundary)
        surface = load_native_surface_vtp(boundary)
        area_path = directory / "boundary_cell_area_1.npy"
        np.save(area_path, np.asarray([1.0, 2.0], dtype="<f4"), allow_pickle=False)
        areas = audit_fixed_surface_area_file(
            surface,
            area_path,
            expected_area_sha256=sha256_file(area_path),
            source_boundary_sha256=surface.boundary_sha256,
            validation_chunk_entities=1,
        )
        return surface, areas

    def test_loader_preserves_polygon_cell_data_order_and_area_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            surface, areas = self.load_fixture(Path(directory_name))
            self.assertEqual(surface.vtk_version, REQUIRED_VTK_VERSION)
            self.assertEqual(surface.polygon_count, 2)
            self.assertEqual(surface.point_count, 8)
            np.testing.assert_array_equal(surface.pressure_m2_per_s2, [2.0, 3.0])
            np.testing.assert_allclose(
                surface.wall_shear_m2_per_s2,
                [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
            )
            np.testing.assert_array_equal(surface.offsets, [0, 4, 8])
            np.testing.assert_array_equal(
                surface.connectivity, [0, 1, 2, 3, 4, 5, 6, 7]
            )
            self.assertIn("cell_array_that_must_not_load", surface.available_cell_arrays)
            self.assertIn("point_array_that_must_not_load", surface.available_point_arrays)
            self.assertEqual(surface.vtk_owner.GetPointData().GetNumberOfArrays(), 0)
            self.assertEqual(surface.vtk_owner.GetCellData().GetNumberOfArrays(), 2)
            self.assertEqual(areas.values_m2.dtype.str, "<f4")
            self.assertEqual(areas.entity_count, 2)
            self.assertEqual(areas.area_sum_m2, 3.0)
            self.assertEqual(
                areas.audit_record()["role"], "fixed_external_input_not_regenerated"
            )

    def test_path_replacement_cannot_redirect_vtk_after_boundary_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            boundary = directory / "boundary_1.vtp"
            replacement = directory / "replacement.vtp"
            self.write_vtp(self.make_surface(), boundary)
            original_sha256 = sha256_file(boundary)

            replacement_surface = self.make_surface()
            replacement_pressure = replacement_surface.GetCellData().GetArray(
                "pMeanTrim"
            )
            replacement_pressure.SetValue(0, 102.0)
            replacement_pressure.SetValue(1, 103.0)
            self.write_vtp(replacement_surface, replacement)
            replacement_sha256 = sha256_file(replacement)
            self.assertNotEqual(original_sha256, replacement_sha256)

            original_descriptor_path = (
                native_surface_module._RetainedSurfaceFile.descriptor_path
            )
            observed: dict[str, object] = {}

            def replace_path_after_hash(
                retained: object,
            ) -> Path:
                descriptor_path = original_descriptor_path(retained)
                observed["descriptor_path"] = descriptor_path
                observed["descriptor_identity"] = (
                    descriptor_path.stat().st_dev,
                    descriptor_path.stat().st_ino,
                )
                observed["original_identity"] = (
                    boundary.stat().st_dev,
                    boundary.stat().st_ino,
                )
                os.replace(replacement, boundary)
                return descriptor_path

            with mock.patch.object(
                native_surface_module._RetainedSurfaceFile,
                "descriptor_path",
                replace_path_after_hash,
            ), self.assertRaisesRegex(
                DrivAerNativeSurfaceError, "changed while VTK read"
            ):
                load_native_surface_vtp(boundary)

            descriptor_path = observed["descriptor_path"]
            self.assertIsInstance(descriptor_path, Path)
            self.assertIn(
                descriptor_path.parent,
                {Path("/proc/self/fd"), Path("/dev/fd")},
            )
            self.assertEqual(
                observed["descriptor_identity"], observed["original_identity"]
            )
            self.assertEqual(sha256_file(boundary), replacement_sha256)

    def test_in_place_mutation_during_vtk_pass_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            boundary = Path(directory_name) / "boundary_1.vtp"
            self.write_vtp(self.make_surface(), boundary)
            original_parser = (
                native_surface_module._load_native_surface_vtp_from_descriptor
            )

            def parse_then_mutate(*args: object, **kwargs: object) -> object:
                surface = original_parser(*args, **kwargs)
                with boundary.open("ab") as stream:
                    stream.write(b"changed-during-vtk")
                    stream.flush()
                    os.fsync(stream.fileno())
                return surface

            with mock.patch.object(
                native_surface_module,
                "_load_native_surface_vtp_from_descriptor",
                side_effect=parse_then_mutate,
            ), self.assertRaisesRegex(
                DrivAerNativeSurfaceError, "changed while VTK read"
            ):
                load_native_surface_vtp(boundary)

    def test_chunked_metrics_and_forces_match_whole_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            surface, areas = self.load_fixture(Path(directory_name))
            pressure_prediction = np.asarray([2.5, 2.75])
            shear_prediction = np.asarray([[0.2, 0.1, 0.4], [0.0, 0.5, 0.8]])
            raw_ids = np.arange(2, dtype=np.int64)
            whole = evaluate_native_surface_predictions(
                surface,
                areas,
                raw_cell_ids=raw_ids,
                pressure_prediction=pressure_prediction,
                wall_shear_prediction=shear_prediction,
                chunk_polygons=2,
            )
            chunked = evaluate_native_surface_predictions(
                surface,
                areas,
                raw_cell_ids=raw_ids,
                pressure_prediction=pressure_prediction,
                wall_shear_prediction=shear_prediction,
                chunk_polygons=1,
            )
            self.assertEqual(len(whole.metric_values), 12)
            self.assertNotEqual(
                whole.metric_values["surface_pressure_rel_l2"],
                whole.metric_values["surface_pressure_equal_entity_rel_l2"],
            )
            for metric_id, value in whole.metric_values.items():
                self.assertTrue(
                    math.isclose(value, chunked.metric_values[metric_id], rel_tol=2e-15)
                )
            for coefficient in ("Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr"):
                self.assertTrue(
                    math.isclose(
                        float(whole.force_coefficients[coefficient]),
                        float(chunked.force_coefficients[coefficient]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                )

            geometry = surface_geometry_chunk_validated(
                surface.points_m,
                surface.connectivity,
                surface.offsets,
                0,
                2,
            )
            direct = finalize_force_coefficients(
                [force_moment_chunk(geometry, pressure_prediction, shear_prediction)],
                expected_entity_count=2,
            )
            for coefficient in ("Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr"):
                self.assertAlmostEqual(
                    float(whole.force_coefficients[coefficient]),
                    float(direct[coefficient]),
                )
            self.assertEqual(
                whole.metric_sufficient_statistics["surface_pressure_rel_l2"][
                    "dataset_weighting"
                ],
                "surface_face_area",
            )
            self.assertEqual(
                whole.metric_sufficient_statistics[
                    "surface_pressure_equal_entity_rel_l2"
                ]["total_weight"],
                2.0,
            )

    def test_area_hash_source_dtype_count_and_values_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            boundary = directory / "boundary_1.vtp"
            self.write_vtp(self.make_surface(), boundary)
            surface = load_native_surface_vtp(boundary)

            def write(values: np.ndarray, name: str = "areas.npy") -> Path:
                path = directory / name
                np.save(path, values, allow_pickle=False)
                return path

            valid = write(np.asarray([1.0, 2.0], dtype="<f4"))
            with self.assertRaisesRegex(DrivAerNativeSurfaceError, "SHA-256 mismatch"):
                audit_fixed_surface_area_file(
                    surface,
                    valid,
                    expected_area_sha256="0" * 64,
                    source_boundary_sha256=surface.boundary_sha256,
                )
            with self.assertRaisesRegex(DrivAerNativeSurfaceError, "source boundary"):
                audit_fixed_surface_area_file(
                    surface,
                    valid,
                    expected_area_sha256=sha256_file(valid),
                    source_boundary_sha256="0" * 64,
                )
            for values, message, name in (
                (np.asarray([1.0, 2.0], dtype="<f8"), "little-endian", "f8.npy"),
                (np.asarray([1.0], dtype="<f4"), "count differs", "count.npy"),
                (np.asarray([1.0, 0.0], dtype="<f4"), "strictly positive", "zero.npy"),
                (np.asarray([1.0, np.nan], dtype="<f4"), "finite", "nan.npy"),
            ):
                with self.subTest(message=message):
                    path = write(values, name)
                    with self.assertRaisesRegex(DrivAerNativeSurfaceError, message):
                        audit_fixed_surface_area_file(
                            surface,
                            path,
                            expected_area_sha256=sha256_file(path),
                            source_boundary_sha256=surface.boundary_sha256,
                        )

    def test_topology_field_and_prediction_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for name, polydata, message in (
                ("line.vtp", self.make_surface(include_line=True), "polygons only"),
                ("missing.vtp", self.make_surface(include_shear=False), "missing required"),
                (
                    "components.vtp",
                    self.make_surface(shear_components=2),
                    "one 3-vector",
                ),
                (
                    "nonfinite.vtp",
                    self.make_surface(nonfinite_pressure=True),
                    "non-finite",
                ),
            ):
                with self.subTest(name=name):
                    path = directory / name
                    self.write_vtp(polydata, path)
                    with self.assertRaisesRegex(DrivAerNativeSurfaceError, message):
                        load_native_surface_vtp(path)

            surface, areas = self.load_fixture(directory / "valid")
            valid_pressure = np.asarray([2.0, 3.0])
            valid_shear = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
            invalid_calls = (
                (np.asarray([1, 0]), valid_pressure, valid_shear, "zero-based"),
                (np.asarray([0]), valid_pressure, valid_shear, "one integer ID"),
                (np.asarray([0, 1]), np.asarray([2.0]), valid_shear, "one numeric scalar"),
                (
                    np.asarray([0, 1]),
                    valid_pressure,
                    np.ones((2, 2)),
                    "one numeric 3-vector",
                ),
                (
                    np.asarray([0, 1]),
                    np.asarray([2.0, np.nan]),
                    valid_shear,
                    "finite",
                ),
            )
            for raw_ids, pressure, shear, message in invalid_calls:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(DrivAerNativeSurfaceError, message):
                        evaluate_native_surface_predictions(
                            surface,
                            areas,
                            raw_cell_ids=raw_ids,
                            pressure_prediction=pressure,
                            wall_shear_prediction=shear,
                            chunk_polygons=1,
                        )


if __name__ == "__main__":
    unittest.main()
