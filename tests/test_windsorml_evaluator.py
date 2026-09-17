"""Candidate WindsorML evaluator tests.

The fixtures are deliberately tiny and hand-checkable: every expected force and
error is recomputed here with explicit loops, so a vectorised mistake in the
evaluator cannot be masked by reusing the evaluator's own expression.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError:  # pragma: no cover - exercised only without VTK/NumPy
    np = None
    vtk = None

from reference.windsorml.contract import (  # noqa: E402
    FORCE_REFERENCE_AREA_M2,
    SourceFileIdentity,
    WindsorMLForceTruth,
    WindsorMLSourceCase,
)
from reference.windsorml.prediction_chunks import (  # noqa: E402
    CANDIDATE_ARTIFACT_ROLE,
    WINDSORML_CANDIDATE_FORMAT,
    WINDSORML_SURFACE_SUPPORT_ID,
    WINDSORML_VOLUME_SUPPORT_ID,
    PredictionChunkError,
    load_prediction_chunk_manifest,
)

CASE_ID = "run_0"  # WindsorML is 0-indexed; AhmedML's pattern would reject this.

SURFACE_NORMALS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
)
SURFACE_DUAL_AREA = (0.03, 0.05, 0.02, 0.04)
SURFACE_CP = (0.20, -0.30, 0.40, -0.10)
SURFACE_CFX = (0.011, 0.013, 0.017, 0.019)
SURFACE_CFY = (0.021, 0.023, 0.027, 0.029)
SURFACE_CFZ = (0.031, 0.033, 0.037, 0.039)

VOLUME_PRESSURE = (0.1, -0.2, 0.3)
VOLUME_UX = (0.8, 0.6, 1.1)
VOLUME_UY = (0.1, -0.1, 0.0)
VOLUME_UZ = (0.0, 0.2, -0.2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, relative: str) -> SourceFileIdentity:
    return SourceFileIdentity(PurePosixPath(relative), sha256(path), path.stat().st_size)


def vtk_array(name: str, values) -> object:
    result = numpy_to_vtk(np.asarray(values), deep=True)
    result.SetName(name)
    return result


def strip_vtk_information_keys(path: Path) -> None:
    """Match the public inline-binary files, which have no nested metadata."""

    payload = path.read_bytes()
    payload = re.sub(
        rb"\s*<InformationKey\b.*?</InformationKey>", b"", payload, flags=re.DOTALL
    )
    path.write_bytes(payload)


def expected_force_xyz() -> "np.ndarray":
    """Integrate the synthetic surface with explicit loops.

    C_vec = sum_i [ -cp_i * n_i + cf_i ] * dA_i, then divided by A_ref.
    """

    total = [0.0, 0.0, 0.0]
    shear = (SURFACE_CFX, SURFACE_CFY, SURFACE_CFZ)
    for index in range(len(SURFACE_CP)):
        area = SURFACE_DUAL_AREA[index]
        for axis in range(3):
            pressure_term = -SURFACE_CP[index] * SURFACE_NORMALS[index][axis]
            total[axis] += (pressure_term + shear[axis][index]) * area
    return np.asarray(total, dtype=np.float64)


def write_surface(path: Path, *, keep_information_keys: bool = False) -> None:
    """Write the synthetic boundary.

    ``keep_information_keys`` leaves the nested ``<InformationKey>`` children
    that VTK emits for arrays owning an information object. The real WindsorML
    boundary files keep them on ``Normals``; DrivAerML and AhmedML files do not,
    which is why their fixtures strip them. Keeping them exercises the indexer
    path where a DataArray's base64 body ends at its first child rather than at
    ``</DataArray>``.
    """
    return _write_surface(path, keep_information_keys=keep_information_keys)


def _write_surface(path: Path, *, keep_information_keys: bool = False) -> None:
    points = vtk.vtkPoints()
    for offset in range(4):
        points.InsertNextPoint(float(offset), 0.0, 0.0)
    cells = vtk.vtkCellArray()
    tetra = vtk.vtkTetra()
    for local in range(4):
        tetra.GetPointIds().SetId(local, local)
    cells.InsertNextCell(tetra)
    data = vtk.vtkUnstructuredGrid()
    data.SetPoints(points)
    data.SetCells(vtk.VTK_TETRA, cells)
    # WindsorML publishes its surface fields on POINTS, not cells.
    point_data = data.GetPointData()
    point_data.AddArray(vtk_array("Normals", np.asarray(SURFACE_NORMALS, dtype=np.float32)))
    point_data.AddArray(vtk_array("cpavg", np.asarray(SURFACE_CP, dtype=np.float32)))
    point_data.AddArray(vtk_array("cfxavg", np.asarray(SURFACE_CFX, dtype=np.float32)))
    point_data.AddArray(vtk_array("cfyavg", np.asarray(SURFACE_CFY, dtype=np.float32)))
    point_data.AddArray(vtk_array("cfzavg", np.asarray(SURFACE_CFZ, dtype=np.float32)))
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    writer.SetHeaderTypeToUInt64()
    assert writer.Write() == 1
    if not keep_information_keys:
        strip_vtk_information_keys(path)


def write_volume(path: Path) -> None:
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    for cell_index in range(3):
        origin = 3.0 * cell_index
        base = points.GetNumberOfPoints()
        for point in (
            (origin, 0.0, 0.0),
            (origin + 1.0, 0.0, 0.0),
            (origin, 1.0, 0.0),
            (origin, 0.0, 1.0),
        ):
            points.InsertNextPoint(*point)
        tetra = vtk.vtkTetra()
        for local in range(4):
            tetra.GetPointIds().SetId(local, base + local)
        cells.InsertNextCell(tetra)
    data = vtk.vtkUnstructuredGrid()
    data.SetPoints(points)
    data.SetCells(vtk.VTK_TETRA, cells)
    cell_data = data.GetCellData()
    cell_data.AddArray(vtk_array("velocityxavg", np.asarray(VOLUME_UX, dtype=np.float32)))
    cell_data.AddArray(vtk_array("velocityyavg", np.asarray(VOLUME_UY, dtype=np.float32)))
    cell_data.AddArray(vtk_array("velocityzavg", np.asarray(VOLUME_UZ, dtype=np.float32)))
    cell_data.AddArray(vtk_array("pressureavg", np.asarray(VOLUME_PRESSURE, dtype=np.float32)))
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    writer.SetHeaderTypeToUInt64()
    assert writer.Write() == 1
    strip_vtk_information_keys(path)


def prediction_manifest(
    root: Path,
    *,
    support_id: str,
    association: str,
    fields: dict,
    chunk_count: int = 1,
    case_id: str = CASE_ID,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    row_count = next(iter(fields.values())).shape[0]
    bounds = np.array_split(np.arange(row_count), chunk_count)
    chunks = []
    for index, rows in enumerate(bounds):
        chunk = root / f"chunk-{index:05d}.npz"
        np.savez(
            chunk,
            raw_cell_id=rows.astype(np.int64),
            **{name: values[rows] for name, values in fields.items()},
        )
        chunks.append(
            {
                "chunk_index": index,
                "file": chunk.name,
                "sha256": sha256(chunk),
                "row_count": int(rows.shape[0]),
                "raw_cell_id_start": int(rows[0]),
                "raw_cell_id_stop": int(rows[-1]) + 1,
            }
        )
    document = {
        "format": WINDSORML_CANDIDATE_FORMAT,
        "format_version": 1,
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "case_id": case_id,
        "support_id": support_id,
        "association": association,
        "total_row_count": row_count,
        "field_components": {name: 1 for name in fields},
        "chunks": chunks,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def surface_fields(scale: float = 1.0) -> dict:
    return {
        "cpavg": np.asarray(SURFACE_CP, dtype=np.float32) * scale,
        "cfxavg": np.asarray(SURFACE_CFX, dtype=np.float32) * scale,
        "cfyavg": np.asarray(SURFACE_CFY, dtype=np.float32) * scale,
        "cfzavg": np.asarray(SURFACE_CFZ, dtype=np.float32) * scale,
    }


def volume_fields() -> dict:
    return {
        "velocityxavg": np.asarray(VOLUME_UX, dtype=np.float32),
        "velocityyavg": np.asarray(VOLUME_UY, dtype=np.float32),
        "velocityzavg": np.asarray(VOLUME_UZ, dtype=np.float32),
        "pressureavg": np.asarray(VOLUME_PRESSURE, dtype=np.float32),
    }


def build_case(
    root: Path,
    *,
    published: WindsorMLForceTruth | None = None,
    keep_information_keys: bool = False,
):
    dataset = root / "dataset"
    (dataset / CASE_ID).mkdir(parents=True, exist_ok=True)
    boundary = dataset / CASE_ID / f"boundary_{CASE_ID.removeprefix('run_')}.vtu"
    volume = dataset / CASE_ID / f"volume_{CASE_ID.removeprefix('run_')}.vtu"
    dual = dataset / CASE_ID / f"boundary_dual_area_{CASE_ID.removeprefix('run_')}.npy"
    write_surface(boundary, keep_information_keys=keep_information_keys)
    write_volume(volume)
    np.save(dual, np.asarray(SURFACE_DUAL_AREA, dtype=np.float32), allow_pickle=False)

    geo = dataset / CASE_ID / "geo_parameters_0.csv"
    geo.write_text("frontal_area\n0.114\n", encoding="utf-8")
    forces = dataset / CASE_ID / "force_mom_0.csv"
    forces.write_text("cd, cs, cl, cmy\n0,0,0,0\n", encoding="utf-8")

    if published is None:
        integrated = expected_force_xyz() / FORCE_REFERENCE_AREA_M2
        published = WindsorMLForceTruth(
            cd=float(integrated[0]), cl=float(integrated[1]), cs=float(integrated[2]), cmy=0.0
        )
    case = WindsorMLSourceCase(
        case_id=CASE_ID,
        run_id=0,
        surface_entity_count=len(SURFACE_CP),
        volume_entity_count=len(VOLUME_PRESSURE),
        boundary=file_identity(boundary, f"{CASE_ID}/boundary_0.vtu"),
        surface_dual_area=file_identity(dual, f"{CASE_ID}/boundary_dual_area_0.npy"),
        volume=file_identity(volume, f"{CASE_ID}/volume_0.vtu"),
        geometry_parameters=file_identity(geo, f"{CASE_ID}/geo_parameters_0.csv"),
        force_coefficients=file_identity(forces, f"{CASE_ID}/force_mom_0.csv"),
        force_truth=published,
    )
    return case, dataset


@unittest.skipIf(np is None or vtk is None, "requires NumPy and VTK")
class WindsorMLEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _evaluate(self, *, chunk_count: int = 1, scale: float = 1.0, case=None, dataset=None):
        from reference.windsorml.evaluator import evaluate_candidate_case

        if case is None:
            case, dataset = build_case(self.root)
        surface = prediction_manifest(
            self.root / f"surface-{chunk_count}-{scale}",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="PointData",
            fields=surface_fields(scale),
            chunk_count=chunk_count,
        )
        volume = prediction_manifest(
            self.root / f"volume-{chunk_count}-{scale}",
            support_id=WINDSORML_VOLUME_SUPPORT_ID,
            association="CellData",
            fields=volume_fields(),
            chunk_count=1,
        )
        return evaluate_candidate_case(
            case=case,
            dataset_root=dataset,
            surface_manifest=surface,
            volume_manifest=volume,
        ).to_json()

    def test_perfect_prediction_scores_zero_error(self) -> None:
        evidence = self._evaluate()
        surface_metrics = evidence["surface"]["metrics"]
        for name, value in surface_metrics.items():
            self.assertAlmostEqual(value, 0.0, places=9, msg=f"{name} should be zero")
        volume_metrics = evidence["volume"]["metrics"]
        for name, value in volume_metrics.items():
            self.assertAlmostEqual(value, 0.0, places=9, msg=f"{name} should be zero")

    def test_integrated_forces_match_an_independent_hand_integration(self) -> None:
        evidence = self._evaluate()
        expected = expected_force_xyz() / FORCE_REFERENCE_AREA_M2
        truth = evidence["forces"]["truth_integrated"]
        self.assertAlmostEqual(truth["cd"], float(expected[0]), places=6)
        self.assertAlmostEqual(truth["cl"], float(expected[1]), places=6)
        self.assertAlmostEqual(truth["cs"], float(expected[2]), places=6)
        # A perfect prediction must integrate to the same coefficients, which is
        # only true because truth and prediction are integrated identically.
        self.assertEqual(truth, evidence["forces"]["prediction_integrated"])

    def test_lift_is_the_y_axis_not_the_z_axis(self) -> None:
        """Regression guard for the WindsorML axis convention.

        The synthetic surface is built so the y and z coefficients differ; if
        lift were ever remapped to +z this asserts loudly rather than silently
        reporting a plausible-looking number.
        """

        evidence = self._evaluate()
        expected = expected_force_xyz() / FORCE_REFERENCE_AREA_M2
        truth = evidence["forces"]["truth_integrated"]
        # The native arrays are Float32, so the integration carries float32
        # precision; 6 places is far tighter than any axis mix-up.
        self.assertAlmostEqual(truth["cl"], float(expected[1]), places=6)
        self.assertNotAlmostEqual(float(expected[1]), float(expected[2]), places=6)
        self.assertNotAlmostEqual(truth["cl"], float(expected[2]), places=6)
        self.assertEqual(evidence["forces"]["axes"]["lift"], "+y")

    def test_metrics_are_invariant_to_prediction_chunking(self) -> None:
        single = self._evaluate(chunk_count=1, scale=0.9)
        split = self._evaluate(chunk_count=4, scale=0.9)
        self.assertEqual(
            single["surface"]["metrics"], split["surface"]["metrics"]
        )
        self.assertEqual(
            single["forces"]["prediction_integrated"],
            split["forces"]["prediction_integrated"],
        )

    def test_imperfect_prediction_produces_positive_error(self) -> None:
        evidence = self._evaluate(scale=0.5)
        self.assertGreater(evidence["surface"]["metrics"]["surface_pressure_rel_l2"], 0.0)
        self.assertGreater(
            evidence["surface"]["metrics"]["surface_wall_shear_rel_l2"], 0.0
        )

    def test_nested_information_keys_do_not_corrupt_the_payload(self) -> None:
        """Regression guard for the inline-binary indexer.

        The real WindsorML boundary files keep VTK's nested ``<InformationKey>``
        children on ``Normals``. The indexer previously ran a DataArray's base64
        body all the way to ``</DataArray>``, swallowing that trailing XML and
        failing with "invalid base64" on every real case. Only real data caught
        it, so it is pinned here.
        """

        root = self.root / "infokeys"
        root.mkdir()
        case, dataset = build_case(root, keep_information_keys=True)
        evidence = self._evaluate(case=case, dataset=dataset)
        for name, value in evidence["surface"]["metrics"].items():
            self.assertAlmostEqual(value, 0.0, places=9, msg=name)

    def _profile_support(self, *, case_id: str = CASE_ID) -> dict:
        """Minimal two-family support over the synthetic case's native entities."""

        cp = np.asarray(SURFACE_CP, dtype=np.float64)
        ux = np.asarray(VOLUME_UX, dtype=np.float64)
        u_ref = 42.1
        point_ids = [0, 2, 3]
        cell_ids = [0, 2]
        cp_station = {
            "native_point_ids": point_ids,
            "coordinate": [0.0, 0.5, 1.0],
            "truth_cp": cp[point_ids].tolist(),
        }
        velocity_station = {
            "native_cell_ids": cell_ids,
            "coordinate": [0.0, 1.0],
            "truth_ux_over_uinf": (ux[cell_ids] / u_ref).tolist(),
        }
        return {
            "schema": "windsorml-profile-support-v2",
            "case_id": case_id,
            "run_id": 0,
            "sample_count": 3,
            "reference_velocity_m_s": u_ref,
            "body_height_m": 0.34342,
            "families": {
                "windsorml_cp_constant_v1": {"cp_centreline_upper": cp_station},
                "windsorml_cp_relative_v1": {
                    "cp_centreline_upper_relative": dict(cp_station)
                },
                "windsorml_velocity_constant_v1": {
                    "wake_vertical_x_0p05l": velocity_station
                },
                "windsorml_velocity_relative_v1": {
                    "wake_vertical_x_0p05l_relative": dict(velocity_station)
                },
            },
        }

    def test_profiles_are_derived_from_the_same_submitted_fields(self) -> None:
        from reference.windsorml.evaluator import evaluate_candidate_case

        case, dataset = build_case(self.root)
        surface = prediction_manifest(
            self.root / "surface-profiles",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="PointData",
            fields=surface_fields(),
        )
        volume = prediction_manifest(
            self.root / "volume-profiles",
            support_id=WINDSORML_VOLUME_SUPPORT_ID,
            association="CellData",
            fields=volume_fields(),
        )
        evidence = evaluate_candidate_case(
            case=case,
            dataset_root=dataset,
            surface_manifest=surface,
            volume_manifest=volume,
            profile_support=self._profile_support(),
        ).to_json()

        profiles = evidence["profiles"]
        self.assertIs(profiles["participant_profile_payload_accepted"], False)
        self.assertEqual(
            set(profiles["families"]),
            {
                "windsorml_cp_constant_v1",
                "windsorml_cp_relative_v1",
                "windsorml_velocity_constant_v1",
                "windsorml_velocity_relative_v1",
            },
        )
        for family, stations in profiles["families"].items():
            for station in stations:
                # A perfect prediction must reproduce the truth series exactly.
                for truth, prediction in zip(
                    station["truth"], station["prediction"], strict=True
                ):
                    self.assertAlmostEqual(truth, prediction, places=6, msg=family)

    def test_stale_profile_support_truth_is_rejected(self) -> None:
        """Support whose stored truth no longer matches the pinned source fails."""

        from reference.windsorml.evaluator import (
            WindsorMLCandidateEvaluatorError,
            evaluate_candidate_case,
        )

        case, dataset = build_case(self.root)
        support = self._profile_support()
        support["families"]["windsorml_cp_constant_v1"]["cp_centreline_upper"][
            "truth_cp"
        ] = [9.9, 9.9, 9.9]
        surface = prediction_manifest(
            self.root / "surface-stale",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="PointData",
            fields=surface_fields(),
        )
        with self.assertRaises(WindsorMLCandidateEvaluatorError) as caught:
            evaluate_candidate_case(
                case=case,
                dataset_root=dataset,
                surface_manifest=surface,
                profile_support=support,
            )
        self.assertIn("disagrees with the native field", str(caught.exception))

    def test_profile_support_for_another_case_is_rejected(self) -> None:
        from reference.windsorml.evaluator import (
            WindsorMLCandidateEvaluatorError,
            evaluate_candidate_case,
        )

        case, dataset = build_case(self.root)
        surface = prediction_manifest(
            self.root / "surface-wrongcase",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="PointData",
            fields=surface_fields(),
        )
        with self.assertRaises(WindsorMLCandidateEvaluatorError):
            evaluate_candidate_case(
                case=case,
                dataset_root=dataset,
                surface_manifest=surface,
                profile_support=self._profile_support(case_id="run_7"),
            )

    def test_surface_manifest_must_declare_point_data(self) -> None:
        path = prediction_manifest(
            self.root / "bad-association",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="CellData",
            fields=surface_fields(),
        )
        with self.assertRaises(PredictionChunkError) as caught:
            load_prediction_chunk_manifest(path)
        self.assertIn("PointData", str(caught.exception))

    def test_published_force_disagreement_beyond_tolerance_is_rejected(self) -> None:
        from reference.windsorml.evaluator import WindsorMLCandidateEvaluatorError

        integrated = expected_force_xyz() / FORCE_REFERENCE_AREA_M2
        drifted = WindsorMLForceTruth(
            cd=float(integrated[0]) * 1.5,
            cl=float(integrated[1]),
            cs=float(integrated[2]),
            cmy=0.0,
        )
        case, dataset = build_case(self.root, published=drifted)
        with self.assertRaises(WindsorMLCandidateEvaluatorError) as caught:
            self._evaluate(case=case, dataset=dataset)
        self.assertIn("replay differs", str(caught.exception))

    def test_run_zero_is_an_accepted_case_id(self) -> None:
        """WindsorML is 0-indexed; the shared pattern used to reject run_0."""

        path = prediction_manifest(
            self.root / "run-zero",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="PointData",
            fields=surface_fields(),
            case_id="run_0",
        )
        manifest = load_prediction_chunk_manifest(path)
        self.assertEqual(manifest.case_id, "run_0")

    def test_drivaerml_support_id_is_rejected_by_the_windsorml_loader(self) -> None:
        path = prediction_manifest(
            self.root / "foreign-support",
            support_id=WINDSORML_SURFACE_SUPPORT_ID,
            association="PointData",
            fields=surface_fields(),
        )
        document = json.loads(path.read_text())
        document["support_id"] = "volume_native_cells"
        document["format"] = WINDSORML_CANDIDATE_FORMAT
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(PredictionChunkError):
            load_prediction_chunk_manifest(path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
