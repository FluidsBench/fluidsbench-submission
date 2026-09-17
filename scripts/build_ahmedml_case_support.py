#!/usr/bin/env python3
"""Build exact native scoring/profile support for one AhmedML case.

This is an expensive, deterministic maintainer command.  It loads one native
surface and volume with VTK 9.5.2, preserves source cell order, and emits only
evaluator-owned support:

* oriented surface area vectors for field-derived force coefficients;
* Float32 native-cell volumes for the report-only physical volume weighting;
* mutually exclusive native-volume region codes; and
* frozen native cell IDs for all 3 Cp cuts and 4 velocity profiles, including
  128-point public truth series.

No model prediction is used while building this support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import vtk  # type: ignore[import-not-found]
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import-not-found]
except ImportError as error:  # pragma: no cover - command dependency check.
    raise SystemExit(f"VTK 9.5.2 is required: {error}") from error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    FORCE_ABSOLUTE_TOLERANCE,
    PROFILE_DEFINITION_SHA256,
    SOURCE_IDENTITY_SHA256,
    VOLUME_REGION_DEFINITION_SHA256,
    AhmedMLContractError,
    AhmedMLSourceCase,
    classify_force_replay,
    load_source_identity,
    sha256_file,
)


SCHEMA = "ahmedml-native-case-support-v1"
STATUS = "candidate_evaluator_support_not_official"
REQUIRED_VTK_VERSION = "9.5.2"
PROFILE_SAMPLE_COUNT = 128
SURFACE_STATIONS = (
    "upper_body_centerline",
    "underbody_centerline",
    "rear_slant_centerline",
)
VOLUME_STATIONS = (
    "wake_vertical_x_0p25_l",
    "wake_vertical_x_0p50_l",
    "wake_vertical_x_1p00_l",
    "wake_lateral_x_0p50_l_z_0p50_h",
)
GEOMETRY_COLUMNS = (
    "body-length",
    "body-height",
    "body-width",
    "front-arc-diameter",
    "slant-angle-length",
    "slant-angle-height",
    "slant-surface-length",
    "slant-angle-degrees",
)
Q_REF = 0.5
A_REF = 0.112032
FORCE_DENOMINATOR = Q_REF * A_REF
AREA_RELATIVE_TOLERANCE = 2.0e-6


class SupportBuildError(ValueError):
    """Raised when exact case support cannot be constructed."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _verify_source_file(path: Path, *, size_bytes: int, digest: str, label: str) -> None:
    try:
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError as error:
        raise SupportBuildError(f"cannot resolve {label} {path}: {error}") from error
    if size != size_bytes:
        raise SupportBuildError(
            f"{label} size is {size}, expected pinned size {size_bytes}"
        )
    actual = sha256_file(resolved)
    if actual != digest:
        raise SupportBuildError(
            f"{label} SHA-256 is {actual}, expected pinned digest {digest}"
        )


def _geometry(path: Path) -> dict[str, float]:
    try:
        with path.resolve(strict=True).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise SupportBuildError(f"cannot read geometry parameters: {error}") from error
    if len(rows) != 1 or reader.fieldnames is None:
        raise SupportBuildError("geometry parameter CSV must contain exactly one row")
    normalized = {key.strip(): value for key, value in rows[0].items() if key is not None}
    if set(normalized) != set(GEOMETRY_COLUMNS):
        raise SupportBuildError("geometry parameter columns differ from the contract")
    result: dict[str, float] = {}
    for key in GEOMETRY_COLUMNS:
        try:
            number = float(normalized[key])
        except (TypeError, ValueError) as error:
            raise SupportBuildError(f"geometry parameter {key!r} is not numeric") from error
        if not math.isfinite(number) or number <= 0.0:
            raise SupportBuildError(f"geometry parameter {key!r} must be positive")
        result[key] = number
    return result


def _force_truth(path: Path) -> tuple[float, float]:
    try:
        with path.resolve(strict=True).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise SupportBuildError(f"cannot read force coefficients: {error}") from error
    if len(rows) != 1:
        raise SupportBuildError("force coefficient CSV must contain exactly one row")
    normalized = {key.strip(): value for key, value in rows[0].items() if key is not None}
    if set(normalized) != {"cd", "cl"}:
        raise SupportBuildError("force coefficient CSV must contain cd and cl")
    try:
        cd, cl = float(normalized["cd"]), float(normalized["cl"])
    except (TypeError, ValueError) as error:
        raise SupportBuildError("force coefficient CSV values must be numeric") from error
    if not math.isfinite(cd) or not math.isfinite(cl):
        raise SupportBuildError("force coefficient CSV values must be finite")
    return cd, cl


def _configure_reader(reader: Any, path: Path, cell_arrays: tuple[str, ...]) -> Any:
    reader.SetFileName(str(path.resolve(strict=True)))
    reader.UpdateInformation()
    point_selection = reader.GetPointDataArraySelection()
    cell_selection = reader.GetCellDataArraySelection()
    point_selection.DisableAllArrays()
    cell_selection.DisableAllArrays()
    for name in cell_arrays:
        if not cell_selection.ArrayExists(name):
            raise SupportBuildError(f"native VTK file has no CellData array {name!r}")
        cell_selection.EnableArray(name)
    reader.Update()
    data = reader.GetOutput()
    if data is None:
        raise SupportBuildError("VTK reader produced no dataset")
    return data


def _cell_array(data: Any, name: str, components: int, count: int) -> np.ndarray:
    value = data.GetCellData().GetArray(name)
    if value is None:
        raise SupportBuildError(f"VTK output has no CellData array {name!r}")
    array = np.asarray(vtk_to_numpy(value))
    expected = (count,) if components == 1 else (count, components)
    if array.shape != expected or array.dtype != np.dtype("float32"):
        raise SupportBuildError(
            f"CellData {name!r} has {array.dtype} {array.shape}, expected float32 {expected}"
        )
    if not np.all(np.isfinite(array)):
        raise SupportBuildError(f"CellData {name!r} contains non-finite values")
    return array


def _surface_area_vectors(polydata: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float64)
    polygons = polydata.GetPolys()
    connectivity = np.asarray(
        vtk_to_numpy(polygons.GetConnectivityArray()), dtype=np.int64
    )
    offsets = np.asarray(vtk_to_numpy(polygons.GetOffsetsArray()), dtype=np.int64)
    count = polydata.GetNumberOfCells()
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(connectivity):
        raise SupportBuildError("native polygon offsets do not cover connectivity")
    sizes = np.diff(offsets)
    if np.any(sizes < 3):
        raise SupportBuildError("native boundary contains a polygon with fewer than 3 vertices")
    if np.any(connectivity < 0) or np.any(connectivity >= len(points)):
        raise SupportBuildError("native polygon connectivity references a missing point")
    successor = np.empty_like(connectivity)
    successor[:-1] = connectivity[1:]
    successor[-1] = connectivity[-1]
    successor[offsets[1:] - 1] = connectivity[offsets[:-1]]
    edge_cross = np.cross(points[connectivity], points[successor])
    vectors = 0.5 * np.add.reduceat(edge_cross, offsets[:-1], axis=0)
    magnitudes = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (count, 3) or np.any(~np.isfinite(vectors)):
        raise SupportBuildError("surface area-vector calculation failed")
    if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise SupportBuildError("surface polygon areas must be finite and positive")
    return vectors, magnitudes


def _closest_surface_mapping(
    polydata: Any,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    ids = np.empty(len(targets), dtype=np.int64)
    projected = np.empty((len(targets), 3), dtype=np.float64)
    distances = np.empty(len(targets), dtype=np.float64)
    for index, target in enumerate(targets):
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        distance2 = vtk.reference(0.0)
        locator.FindClosestPoint(
            tuple(float(item) for item in target),
            closest,
            cell_id,
            sub_id,
            distance2,
        )
        ids[index] = int(cell_id)
        projected[index] = closest
        distances[index] = math.sqrt(float(distance2))
    return ids, projected, distances


def _volume_mapping(
    grid: Any,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()
    ids = np.empty(len(targets), dtype=np.int64)
    projected = np.asarray(targets, dtype=np.float64).copy()
    distances = np.zeros(len(targets), dtype=np.float64)
    fallback_count = 0
    for index, target in enumerate(targets):
        query = tuple(float(item) for item in target)
        cell_id = int(locator.FindCell(query))
        if cell_id < 0:
            closest = [0.0, 0.0, 0.0]
            cell_reference = vtk.reference(0)
            sub_id = vtk.reference(0)
            distance2 = vtk.reference(0.0)
            locator.FindClosestPoint(
                query,
                closest,
                cell_reference,
                sub_id,
                distance2,
            )
            cell_id = int(cell_reference)
            projected[index] = closest
            distances[index] = math.sqrt(float(distance2))
            fallback_count += 1
        if cell_id < 0 or cell_id >= grid.GetNumberOfCells():
            raise SupportBuildError("volume profile mapping returned an invalid cell ID")
        ids[index] = cell_id
    return ids, projected, distances, fallback_count


def _surface_targets(
    geometry: dict[str, float], bounds: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    length = geometry["body-length"] / 1000.0
    slant_dx = geometry["slant-angle-length"] / 1000.0
    slant_dz = geometry["slant-angle-height"] / 1000.0
    z_max = float(bounds[5])
    unit = np.linspace(0.0, 1.0, PROFILE_SAMPLE_COUNT, dtype=np.float64)
    targets = np.empty((len(SURFACE_STATIONS), PROFILE_SAMPLE_COUNT, 3), dtype=np.float64)
    targets[0, :, 0] = -length + unit * (length - slant_dx)
    targets[0, :, 1] = 0.0
    targets[0, :, 2] = z_max + 0.05 * length
    targets[1, :, 0] = -length + unit * length
    targets[1, :, 1] = 0.0
    targets[1, :, 2] = 0.0
    targets[2, :, 0] = -slant_dx + unit * slant_dx
    targets[2, :, 1] = 0.0
    targets[2, :, 2] = z_max - unit * slant_dz
    return targets, np.broadcast_to(unit, (len(SURFACE_STATIONS), len(unit))).copy()


def _volume_targets(
    geometry: dict[str, float], surface_bounds: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    length = geometry["body-length"] / 1000.0
    height = geometry["body-height"] / 1000.0
    width = geometry["body-width"] / 1000.0
    z_min = float(surface_bounds[4])
    vertical = np.linspace(0.0, 2.0, PROFILE_SAMPLE_COUNT, dtype=np.float64)
    lateral = np.linspace(-1.0, 1.0, PROFILE_SAMPLE_COUNT, dtype=np.float64)
    targets = np.empty((len(VOLUME_STATIONS), PROFILE_SAMPLE_COUNT, 3), dtype=np.float64)
    for station, factor in enumerate((0.25, 0.50, 1.00)):
        targets[station, :, 0] = factor * length
        targets[station, :, 1] = 0.0
        targets[station, :, 2] = vertical * height
    targets[3, :, 0] = 0.50 * length
    targets[3, :, 1] = lateral * width
    targets[3, :, 2] = z_min + 0.50 * height
    coordinates = np.vstack((vertical, vertical, vertical, lateral))
    return targets, coordinates


def _save_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    np.save(path, array, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "format": "npy",
    }


def _save_profile_mapping(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    np.savez_compressed(path, **arrays)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "format": "npz",
        "arrays": {
            name: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for name, value in arrays.items()
        },
    }


def _volume_regions(
    centres: np.ndarray,
    geometry: dict[str, float],
    z_surface_min: float,
) -> np.ndarray:
    length = geometry["body-length"] / 1000.0
    height = geometry["body-height"] / 1000.0
    width = geometry["body-width"] / 1000.0
    x, y, z = centres[:, 0], centres[:, 1], centres[:, 2]
    codes = np.full(len(centres), 2, dtype=np.uint8)
    near_body = (
        (x >= -1.25 * length)
        & (x <= 0.0)
        & (np.abs(y) <= 0.75 * width)
        & (z >= 0.0)
        & (z <= z_surface_min + 2.0 * height)
    )
    wake = (
        (x > 0.0)
        & (x <= 2.0 * length)
        & (np.abs(y) <= width)
        & (z >= 0.0)
        & (z <= z_surface_min + 2.0 * height)
    )
    codes[near_body] = 0
    codes[wake] = 1
    return codes


def _source_summary(case: AhmedMLSourceCase) -> dict[str, Any]:
    def item(value: Any) -> dict[str, Any]:
        return {
            "path": value.path.as_posix(),
            "sha256": value.sha256,
            "size_bytes": value.size_bytes,
        }

    return {
        "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "boundary": item(case.boundary),
        "surface_cell_area": item(case.surface_cell_area),
        "volume": item(case.volume),
        "geometry_parameters": item(case.geometry_parameters),
        "force_coefficients": item(case.force_coefficients),
    }


def build_case_support(
    *,
    case_id: str,
    dataset_root: Path,
    output_root: Path,
    source_identity_path: Path,
) -> Path:
    if str(vtk.vtkVersion.GetVTKVersion()) != REQUIRED_VTK_VERSION:
        raise SupportBuildError(
            f"support generation requires VTK {REQUIRED_VTK_VERSION}, got "
            f"{vtk.vtkVersion.GetVTKVersion()}"
        )
    identity = load_source_identity(source_identity_path)
    case = identity.case(case_id)
    paths = {
        "boundary": case.boundary.resolve(dataset_root),
        "area": case.surface_cell_area.resolve(dataset_root),
        "volume": case.volume.resolve(dataset_root),
        "geometry": case.geometry_parameters.resolve(dataset_root),
        "force": case.force_coefficients.resolve(dataset_root),
    }
    for key, source in (
        ("boundary", case.boundary),
        ("surface area", case.surface_cell_area),
        ("volume", case.volume),
        ("geometry", case.geometry_parameters),
        ("force", case.force_coefficients),
    ):
        path_key = {
            "surface area": "area",
            "geometry": "geometry",
            "force": "force",
        }.get(key, key)
        _verify_source_file(
            paths[path_key],
            size_bytes=source.size_bytes,
            digest=source.sha256,
            label=key,
        )

    final = output_root.expanduser().resolve() / case_id
    if final.exists():
        raise SupportBuildError(f"output already exists: {final}")
    output_root.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{case_id}-partial-", dir=output_root))

    geometry = _geometry(paths["geometry"])
    published_cd, published_cl = _force_truth(paths["force"])

    surface_reader = vtk.vtkXMLPolyDataReader()
    surface = _configure_reader(
        surface_reader,
        paths["boundary"],
        ("pMean", "wallShearStressMean"),
    )
    if surface.GetNumberOfCells() != case.surface_entity_count:
        raise SupportBuildError("surface entity count differs from source identity")
    if surface.GetNumberOfVerts() or surface.GetNumberOfLines() or surface.GetNumberOfStrips():
        raise SupportBuildError("native boundary must contain polygons only")
    surface_pressure = _cell_array(surface, "pMean", 1, case.surface_entity_count)
    surface_shear = _cell_array(
        surface, "wallShearStressMean", 3, case.surface_entity_count
    )
    area_vectors, calculated_area = _surface_area_vectors(surface)
    published_area = np.load(paths["area"].resolve(strict=True), mmap_mode="r", allow_pickle=False)
    if published_area.dtype != np.dtype("float32") or published_area.shape != (
        case.surface_entity_count,
    ):
        raise SupportBuildError("published surface area array has the wrong dtype or shape")
    relative_area_error = np.abs(calculated_area - published_area) / calculated_area
    maximum_area_error = float(np.max(relative_area_error))
    if maximum_area_error > AREA_RELATIVE_TOLERANCE:
        raise SupportBuildError(
            f"surface area-vector magnitudes differ from published areas by {maximum_area_error}"
        )

    truth_pressure_force = np.sum(
        surface_pressure[:, None].astype(np.float64) * area_vectors,
        axis=0,
        dtype=np.float64,
    )
    truth_shear_force = np.sum(
        -surface_shear.astype(np.float64) * published_area[:, None],
        axis=0,
        dtype=np.float64,
    )
    truth_force = truth_pressure_force + truth_shear_force
    replay_cd = float(truth_force[0] / FORCE_DENOMINATOR)
    replay_cl = float(truth_force[2] / FORCE_DENOMINATOR)
    try:
        force_replay_exception = classify_force_replay(
            case_id=case_id,
            published_cd=published_cd,
            published_cl=published_cl,
            field_replay_cd=replay_cd,
            field_replay_cl=replay_cl,
        )
    except AhmedMLContractError as error:
        raise SupportBuildError(str(error)) from error

    surface_target, surface_coordinate = _surface_targets(geometry, surface.GetBounds())
    flat_surface_target = surface_target.reshape(-1, 3)
    surface_ids, surface_projected, surface_distance = _closest_surface_mapping(
        surface, flat_surface_target
    )
    surface_ids = surface_ids.reshape(len(SURFACE_STATIONS), PROFILE_SAMPLE_COUNT)
    surface_truth_cp = (2.0 * surface_pressure[surface_ids]).astype(np.float32)
    surface_bounds = tuple(float(item) for item in surface.GetBounds())
    area_artifact = _save_npy(
        partial / "surface_area_vector_m2.npy", area_vectors.astype("<f8", copy=False)
    )

    volume_reader = vtk.vtkXMLUnstructuredGridReader()
    volume = _configure_reader(volume_reader, paths["volume"], ("pMean", "UMean"))
    if volume.GetNumberOfCells() != case.volume_entity_count:
        raise SupportBuildError("volume entity count differs from source identity")
    volume_pressure = _cell_array(volume, "pMean", 1, case.volume_entity_count)
    volume_velocity = _cell_array(volume, "UMean", 3, case.volume_entity_count)
    volume_target, volume_coordinate = _volume_targets(geometry, surface_bounds)
    (
        volume_ids,
        volume_projected,
        volume_distance,
        volume_fallback_count,
    ) = _volume_mapping(volume, volume_target.reshape(-1, 3))
    volume_ids = volume_ids.reshape(len(VOLUME_STATIONS), PROFILE_SAMPLE_COUNT)
    volume_truth_ux = volume_velocity[volume_ids, 0].astype(np.float32)

    # Field arrays are no longer needed.  Removing them before geometry filters
    # materially lowers the peak memory of the 20M-cell cases.
    volume.GetCellData().Initialize()
    volume.GetPointData().Initialize()

    centre_filter = vtk.vtkCellCenters()
    centre_filter.SetInputData(volume)
    centre_filter.VertexCellsOff()
    centre_filter.Update()
    centres = np.asarray(
        vtk_to_numpy(centre_filter.GetOutput().GetPoints().GetData()),
        dtype=np.float64,
    )
    if centres.shape != (case.volume_entity_count, 3) or np.any(~np.isfinite(centres)):
        raise SupportBuildError("VTK cell centres do not cover the native volume")
    region_codes = _volume_regions(centres, geometry, surface_bounds[4])
    region_counts = np.bincount(region_codes, minlength=3)
    if np.any(region_counts == 0) or int(np.sum(region_counts)) != case.volume_entity_count:
        raise SupportBuildError("volume regions are not non-empty and exhaustive")
    region_artifact = _save_npy(partial / "volume_region_code.npy", region_codes)
    del centres, region_codes
    centre_filter = None

    size_filter = vtk.vtkCellSizeFilter()
    size_filter.SetInputData(volume)
    size_filter.ComputeLengthOff()
    size_filter.ComputeAreaOff()
    size_filter.ComputeVolumeOn()
    size_filter.SetVolumeArrayName("ahmedml_cell_volume_m3")
    size_filter.Update()
    volume_values = np.asarray(
        vtk_to_numpy(
            size_filter.GetOutput().GetCellData().GetArray("ahmedml_cell_volume_m3")
        ),
        dtype=np.float64,
    )
    if volume_values.shape != (case.volume_entity_count,):
        raise SupportBuildError("VTK cell-volume output has the wrong shape")
    if np.any(~np.isfinite(volume_values)) or np.any(volume_values <= 0.0):
        raise SupportBuildError("native cell volumes must be finite and positive")
    volume_sum_f64 = float(np.sum(volume_values, dtype=np.float64))
    volume_f32 = volume_values.astype("<f4")
    if np.any(volume_f32 <= 0.0):
        raise SupportBuildError("Float32 cell-volume support underflowed to zero")
    volume_artifact = _save_npy(partial / "volume_cell_volume_m3.npy", volume_f32)

    profile_arrays = {
        "surface_raw_cell_id": surface_ids.astype("<i8", copy=False),
        "surface_coordinate": surface_coordinate.astype("<f8", copy=False),
        "surface_target_xyz_m": surface_target.astype("<f8", copy=False),
        "surface_projected_xyz_m": surface_projected.reshape(
            len(SURFACE_STATIONS), PROFILE_SAMPLE_COUNT, 3
        ).astype("<f8", copy=False),
        "surface_distance_m": surface_distance.reshape(
            len(SURFACE_STATIONS), PROFILE_SAMPLE_COUNT
        ).astype("<f8", copy=False),
        "surface_truth_cp": surface_truth_cp.astype("<f4", copy=False),
        "volume_raw_cell_id": volume_ids.astype("<i8", copy=False),
        "volume_coordinate": volume_coordinate.astype("<f8", copy=False),
        "volume_target_xyz_m": volume_target.astype("<f8", copy=False),
        "volume_projected_xyz_m": volume_projected.reshape(
            len(VOLUME_STATIONS), PROFILE_SAMPLE_COUNT, 3
        ).astype("<f8", copy=False),
        "volume_distance_m": volume_distance.reshape(
            len(VOLUME_STATIONS), PROFILE_SAMPLE_COUNT
        ).astype("<f8", copy=False),
        "volume_truth_ux_over_uinf": volume_truth_ux.astype("<f4", copy=False),
    }
    profile_artifact = _save_profile_mapping(
        partial / "profile_mapping_and_truth.npz", profile_arrays
    )

    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": STATUS,
        "official_submission_support": False,
        "case_id": case.case_id,
        "run_id": case.run_id,
        "dataset_revision": identity.repository_revision,
        "source": _source_summary(case),
        "definitions": {
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
            "regional_definition_sha256": VOLUME_REGION_DEFINITION_SHA256,
            "profile_sample_count": PROFILE_SAMPLE_COUNT,
            "surface_station_ids": list(SURFACE_STATIONS),
            "volume_station_ids": list(VOLUME_STATIONS),
        },
        "entity_counts": {
            "surface": case.surface_entity_count,
            "volume": case.volume_entity_count,
        },
        "geometry_parameters_mm_or_degrees": geometry,
        "artifacts": {
            "surface_area_vector": area_artifact,
            "volume_cell_volume": volume_artifact,
            "volume_region_code": region_artifact,
            "profile_mapping_and_truth": profile_artifact,
        },
        "audits": {
            "vtk_version": REQUIRED_VTK_VERSION,
            "surface_bounds_m": list(surface_bounds),
            "volume_bounds_m": [float(item) for item in volume.GetBounds()],
            "surface_area": {
                "calculated_sum_m2": float(np.sum(calculated_area, dtype=np.float64)),
                "published_sum_m2": float(np.sum(published_area, dtype=np.float64)),
                "maximum_relative_magnitude_difference": maximum_area_error,
                "relative_tolerance": AREA_RELATIVE_TOLERANCE,
            },
            "force_truth": {
                "published_cd": published_cd,
                "published_cl": published_cl,
                "field_replay_cd": replay_cd,
                "field_replay_cl": replay_cl,
                "absolute_tolerance": FORCE_ABSOLUTE_TOLERANCE,
                "pressure_force_xyz": truth_pressure_force.tolist(),
                "wall_shear_force_xyz": truth_shear_force.tolist(),
                **(
                    {
                        "source_exception": {
                            "exception_id": force_replay_exception.exception_id,
                            "scope": "published_force_csv_replay_only",
                            "scoring_truth_source": "native_surface_field_integration",
                            "published_csv_used_for_scoring": False,
                            "reason": force_replay_exception.reason,
                        }
                    }
                    if force_replay_exception is not None
                    else {}
                ),
            },
            "volume_cell_volume": {
                "storage_dtype": "<f4",
                "sum_before_float32_m3": volume_sum_f64,
                "sum_after_float32_m3": float(np.sum(volume_f32, dtype=np.float64)),
                "minimum_m3": float(np.min(volume_f32)),
                "maximum_m3": float(np.max(volume_f32)),
            },
            "volume_regions": {
                "near_body": int(region_counts[0]),
                "wake": int(region_counts[1]),
                "farfield": int(region_counts[2]),
                "sum": int(np.sum(region_counts)),
            },
            "profile_mapping": {
                "surface_maximum_projection_distance_m": float(
                    np.max(surface_distance)
                ),
                "volume_nearest_cell_fallback_count": volume_fallback_count,
                "volume_maximum_projection_distance_m": float(
                    np.max(volume_distance)
                ),
                "runtime_geometric_remapping_forbidden": True,
            },
        },
    }
    manifest_sha = _write_json(partial / "case-support.json", manifest)
    os.rename(partial, final)
    print(
        json.dumps(
            {
                "case_id": case_id,
                "output": str(final),
                "case_support_sha256": manifest_sha,
                "surface_entities": case.surface_entity_count,
                "volume_entities": case.volume_entity_count,
                "region_counts": region_counts.tolist(),
                "volume_mapping_fallback_count": volume_fallback_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-identity",
        type=Path,
        default=(
            ROOT
            / "benchmark-specs"
            / "ahmedml"
            / "public-source-identity"
            / "ahmedml-public-source-identity-v1.json"
        ),
    )
    args = parser.parse_args()
    try:
        build_case_support(
            case_id=args.case_id,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            source_identity_path=args.source_identity,
        )
    except (AhmedMLContractError, SupportBuildError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
