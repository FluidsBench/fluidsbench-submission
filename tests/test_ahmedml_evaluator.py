from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np
import pytest

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk  # noqa: E402
except ModuleNotFoundError as error:
    raise unittest.SkipTest("VTK is required for AhmedML evaluator fixtures") from error

from reference.ahmedml.contract import (  # noqa: E402
    DATASET_VERSION,
    PROFILE_DEFINITION_SHA256,
    REGION_DEFINITION_SHA256,
    VOLUME_REGION_DEFINITION_SHA256,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
    AhmedMLSourceCase,
    AhmedMLSourceIdentity,
    SourceFileIdentity,
)
from reference.ahmedml.evaluator import evaluate_candidate_case  # noqa: E402
from reference.ahmedml.support import (  # noqa: E402
    CASE_SUPPORT_SCHEMA,
    CASE_SUPPORT_STATUS,
    SURFACE_STATIONS,
    VOLUME_STATIONS,
    load_case_support,
)
from reference.ahmedml.prediction_chunks import (  # noqa: E402
    AHMEDML_CANDIDATE_FORMAT,
    CANDIDATE_ARTIFACT_ROLE,
    PredictionChunkError,
    load_prediction_chunk_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, relative: str) -> SourceFileIdentity:
    return SourceFileIdentity(PurePosixPath(relative), sha256(path), path.stat().st_size)


def vtk_array(name: str, values: np.ndarray) -> object:
    result = numpy_to_vtk(np.asarray(values), deep=True)
    result.SetName(name)
    return result


def strip_vtk_information_keys(path: Path) -> None:
    """Match the public inline-binary files, which have no nested metadata."""

    payload = path.read_bytes()
    payload = re.sub(
        rb"\s*<InformationKey\b.*?</InformationKey>",
        b"",
        payload,
        flags=re.DOTALL,
    )
    path.write_bytes(payload)


def write_surface(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = vtk.vtkPoints()
    for point in (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ):
        points.InsertNextPoint(*point)
    polys = vtk.vtkCellArray()
    for ids in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (0, 1, 2)):
        triangle = vtk.vtkTriangle()
        for index, point_id in enumerate(ids):
            triangle.GetPointIds().SetId(index, point_id)
        polys.InsertNextCell(triangle)
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(polys)
    pressure = np.asarray((0.2, -0.3, 0.4, -0.1), dtype=np.float32)
    shear = np.asarray(
        (
            (0.01, 0.02, 0.03),
            (0.03, 0.01, 0.02),
            (0.02, 0.03, 0.01),
            (0.04, 0.02, 0.01),
        ),
        dtype=np.float32,
    )
    data.GetCellData().AddArray(vtk_array("pMean", pressure))
    data.GetCellData().AddArray(vtk_array("wallShearStressMean", shear))
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    writer.SetHeaderTypeToUInt64()
    assert writer.Write() == 1
    strip_vtk_information_keys(path)
    area_vectors = np.asarray(
        (
            (0.0, 0.0, -0.5),
            (0.0, -0.5, 0.0),
            (-0.5, 0.0, 0.0),
            (0.0, 0.0, 0.5),
        ),
        dtype=np.float64,
    )
    areas = np.linalg.norm(area_vectors, axis=1).astype(np.float32)
    return pressure, shear, area_vectors


def write_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    pressure = np.asarray((0.1, -0.2, 0.3), dtype=np.float32)
    velocity = np.asarray(
        ((0.8, 0.1, 0.0), (0.6, -0.1, 0.2), (1.1, 0.0, -0.2)),
        dtype=np.float32,
    )
    data.GetCellData().AddArray(vtk_array("pMean", pressure))
    data.GetCellData().AddArray(vtk_array("UMean", velocity))
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    writer.SetHeaderTypeToUInt64()
    assert writer.Write() == 1
    strip_vtk_information_keys(path)
    return pressure, velocity


def artifact(path: Path, array: np.ndarray) -> dict[str, object]:
    np.save(path, array, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "format": "npy",
    }


def source_entry(value: SourceFileIdentity) -> dict[str, object]:
    return {
        "path": value.path.as_posix(),
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
    }


def prediction_manifest(
    root: Path,
    *,
    support_id: str,
    fields: dict[str, np.ndarray],
) -> Path:
    root.mkdir()
    chunk = root / "chunk-00000.npz"
    row_count = next(iter(fields.values())).shape[0]
    if any(values.shape[0] != row_count for values in fields.values()):
        raise AssertionError("prediction fields must share one row count")
    np.savez(
        chunk,
        raw_cell_id=np.arange(row_count, dtype=np.int64),
        **fields,
    )
    document = {
        "format": AHMEDML_CANDIDATE_FORMAT,
        "format_version": 1,
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "case_id": "run_1",
        "support_id": support_id,
        "association": "CellData",
        "total_row_count": row_count,
        "field_components": {
            name: 1 if values.ndim == 1 else values.shape[1]
            for name, values in fields.items()
        },
        "chunks": [
            {
                "chunk_index": 0,
                "file": chunk.name,
                "sha256": sha256(chunk),
                "row_count": row_count,
                "raw_cell_id_start": 0,
                "raw_cell_id_stop": row_count,
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def make_fixture(root: Path) -> tuple[
    AhmedMLSourceIdentity,
    Path,
    Path,
    Path,
]:
    run = root / "run_1"
    run.mkdir()
    boundary = run / "boundary_1.vtp"
    volume = run / "volume_1.vtu"
    area_path = run / "boundary_cell_area_1.npy"
    geometry = run / "geo_parameters_1.csv"
    force = run / "force_mom_1.csv"
    surface_pressure, surface_shear, area_vectors = write_surface(boundary)
    volume_pressure, volume_velocity = write_volume(volume)
    areas = np.linalg.norm(area_vectors, axis=1).astype(np.float32)
    np.save(area_path, areas, allow_pickle=False)
    geometry.write_text("body-length\n1000\n", encoding="utf-8")
    force.write_text("cd,cl\n0,0\n", encoding="utf-8")
    case = AhmedMLSourceCase(
        case_id="run_1",
        run_id=1,
        surface_entity_count=4,
        volume_entity_count=3,
        boundary=file_identity(boundary, "run_1/boundary_1.vtp"),
        surface_cell_area=file_identity(area_path, "run_1/boundary_cell_area_1.npy"),
        volume=file_identity(volume, "run_1/volume_1.vtu"),
        geometry_parameters=file_identity(geometry, "run_1/geo_parameters_1.csv"),
        force_coefficients=file_identity(force, "run_1/force_mom_1.csv"),
    )
    identity = AhmedMLSourceIdentity(
        source_path=root / "identity.json",
        sha256=SOURCE_IDENTITY_SHA256,
        repository_id=REPOSITORY_ID,
        repository_revision=REPOSITORY_REVISION,
        cases=(case,),
        _by_case=MappingProxyType({"run_1": case}),
    )

    support_root = root / "support" / "run_1"
    support_root.mkdir(parents=True)
    area_vector_artifact = artifact(
        support_root / "surface_area_vector_m2.npy", area_vectors.astype("<f8")
    )
    volume_artifact = artifact(
        support_root / "volume_cell_volume_m3.npy",
        np.asarray((1.0 / 6.0,) * 3, dtype="<f4"),
    )
    region_artifact = artifact(
        support_root / "volume_region_code.npy", np.arange(3, dtype=np.uint8)
    )
    surface_ids = np.vstack(
        [np.arange(128, dtype=np.int64) % 4 for _ in SURFACE_STATIONS]
    )
    volume_ids = np.vstack(
        [np.arange(128, dtype=np.int64) % 3 for _ in VOLUME_STATIONS]
    )
    profiles = {
        "surface_raw_cell_id": surface_ids.astype("<i8"),
        "surface_coordinate": np.vstack(
            [np.linspace(0.0, 1.0, 128) for _ in SURFACE_STATIONS]
        ).astype("<f8"),
        "surface_target_xyz_m": np.zeros((3, 128, 3), dtype="<f8"),
        "surface_projected_xyz_m": np.zeros((3, 128, 3), dtype="<f8"),
        "surface_distance_m": np.zeros((3, 128), dtype="<f8"),
        "surface_truth_cp": (2.0 * surface_pressure[surface_ids]).astype("<f4"),
        "volume_raw_cell_id": volume_ids.astype("<i8"),
        "volume_coordinate": np.vstack(
            [
                np.linspace(0.0, 2.0, 128),
                np.linspace(0.0, 2.0, 128),
                np.linspace(0.0, 2.0, 128),
                np.linspace(-1.0, 1.0, 128),
            ]
        ).astype("<f8"),
        "volume_target_xyz_m": np.zeros((4, 128, 3), dtype="<f8"),
        "volume_projected_xyz_m": np.zeros((4, 128, 3), dtype="<f8"),
        "volume_distance_m": np.zeros((4, 128), dtype="<f8"),
        "volume_truth_ux_over_uinf": volume_velocity[volume_ids, 0].astype("<f4"),
    }
    profile_path = support_root / "profile_mapping_and_truth.npz"
    np.savez_compressed(profile_path, **profiles)
    profile_artifact = {
        "path": profile_path.name,
        "sha256": sha256(profile_path),
        "size_bytes": profile_path.stat().st_size,
        "format": "npz",
        "arrays": {
            name: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for name, value in profiles.items()
        },
    }
    manifest = {
        "schema": CASE_SUPPORT_SCHEMA,
        "schema_version": 1,
        "status": CASE_SUPPORT_STATUS,
        "official_submission_support": False,
        "case_id": "run_1",
        "run_id": 1,
        "dataset_revision": REPOSITORY_REVISION,
        "source": {
            "source_identity_sha256": SOURCE_IDENTITY_SHA256,
            "boundary": source_entry(case.boundary),
            "surface_cell_area": source_entry(case.surface_cell_area),
            "volume": source_entry(case.volume),
            "geometry_parameters": source_entry(case.geometry_parameters),
            "force_coefficients": source_entry(case.force_coefficients),
        },
        "definitions": {
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
            "regional_definition_sha256": VOLUME_REGION_DEFINITION_SHA256,
            "profile_sample_count": 128,
            "surface_station_ids": list(SURFACE_STATIONS),
            "volume_station_ids": list(VOLUME_STATIONS),
        },
        "entity_counts": {"surface": 4, "volume": 3},
        "geometry_parameters_mm_or_degrees": {"body-length": 1000.0},
        "artifacts": {
            "surface_area_vector": area_vector_artifact,
            "volume_cell_volume": volume_artifact,
            "volume_region_code": region_artifact,
            "profile_mapping_and_truth": profile_artifact,
        },
        "audits": {},
    }
    support_manifest = support_root / "case-support.json"
    support_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    surface_prediction = prediction_manifest(
        root / "surface-prediction",
        support_id="ahmedml_surface_native_cells",
        fields={
            "pMean": surface_pressure,
            "wallShearStressMean": surface_shear,
        },
    )
    volume_prediction = prediction_manifest(
        root / "volume-prediction",
        support_id="ahmedml_volume_native_cells",
        fields={"pMean": volume_pressure, "UMean": volume_velocity},
    )
    return identity, support_manifest, surface_prediction, volume_prediction


def test_ahmed_format_rejects_drivaer_support_id() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, _, surface, _ = make_fixture(root)
        document = json.loads(surface.read_text())
        document["support_id"] = "surface_native_cells"
        surface.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(PredictionChunkError, match="not valid for format"):
            load_prediction_chunk_manifest(surface)


def test_perfect_native_predictions_produce_zero_error_and_exact_profiles() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        identity, support_path, surface, volume = make_fixture(root)
        support = load_case_support(support_path, source_identity=identity)
        result = evaluate_candidate_case(
            case_id="run_1",
            dataset_root=root,
            source_identity=identity,
            case_support=support,
            surface_prediction_manifest=surface,
            volume_prediction_manifest=volume,
            maximum_prediction_chunk_rows=4,
            encoded_chunk_bytes=7,
            hash_chunk_bytes=11,
            validation_block_rows=2,
        ).to_json()
        assert result["official_submission"] is False
        assert result["leaderboard_eligible"] is False
        assert all(value == pytest.approx(0.0) for value in result["metric_values"].values())
        force_result = result["force_coefficients"]
        assert force_result["prediction_cd"] == pytest.approx(force_result["truth_cd"])
        assert force_result["prediction_cl"] == pytest.approx(force_result["truth_cl"])
        assert len(result["profiles"]["series"]) == 7
        for series in result["profiles"]["series"]:
            assert len(series["truth"]) == 128
            assert series["prediction"] == pytest.approx(series["truth"])
        diagnostics = result["report_only_regional_diagnostics"]
        assert diagnostics["ranking_effect"] == "none"
        assert diagnostics["contract_sha256"] == REGION_DEFINITION_SHA256
        assert set(diagnostics["surface_pressure"]) == {
            "streamwise_facing",
            "lateral_facing",
            "upward_facing",
            "downward_facing",
        }
        assert set(diagnostics["surface_wall_shear"]) == set(
            diagnostics["surface_pressure"]
        )
        assert set(diagnostics["volume_velocity"]) == {
            "near_body",
            "wake",
            "farfield",
        }
