"""Per-case DrivAerNet++ scoring support: deterministic arrays and their record.

Each official case contributes one ``support.npz`` and one
``case-support.json``.  The NPZ is written member by member with fixed ZIP
metadata, so its SHA-256 depends only on the array values and the NumPy
``.npy`` header format.  The JSON record binds the NPZ, every array inside it,
and the public source file, and is itself retained in Git.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import (
    CASE_SUPPORT_SCHEMA,
    DATASET_ID,
    RELEASE_ID,
    ContractError,
    SHA256_RE,
    sha256_bytes,
)
from .native import (
    DYNAMIC_HEAD,
    PROFILE_STATIONS,
    SUPPORT_RULES_VERSION,
    array_sha256,
    connected_components,
    extract_profile_stations,
    mass_lumped_dual_areas,
    oriented_point_normals,
    read_native_surface,
    sha256_file,
)

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
WEIGHT_ARRAY = "dual_area_m2"
TRUTH_ARRAY = "p"


def station_array_names(station_id: str) -> tuple[str, str]:
    return f"profile_{station_id}_index", f"profile_{station_id}_coordinate"


EXPECTED_ARRAYS = (
    WEIGHT_ARRAY,
    TRUTH_ARRAY,
    *(name for station in PROFILE_STATIONS for name in station_array_names(station)),
)
EXPECTED_DTYPES = {
    WEIGHT_ARRAY: "<f8",
    TRUTH_ARRAY: "<f4",
    **{
        name: dtype
        for station in PROFILE_STATIONS
        for name, dtype in zip(station_array_names(station), ("<i8", "<f8"))
    },
}


def deterministic_zip_info(name: str, *, compress: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize arrays as an NPZ whose bytes do not depend on wall-clock time."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(array), version=(1, 0), allow_pickle=False)
            archive.writestr(deterministic_zip_info(f"{name}.npy", compress=True), member.getvalue(), compresslevel=6)
    return buffer.getvalue()


def load_npz_bytes(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            return {name: np.array(loaded[name]) for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ContractError(f"cannot read {label}: {error}") from error


def build_case_support(case_id: str, vtk_path: Path) -> tuple[bytes, dict[str, Any]]:
    """Materialize one case from its public file; return NPZ bytes and record."""

    surface = read_native_surface(vtk_path)
    weights = mass_lumped_dual_areas(surface)
    labels = connected_components(surface)
    normals = oriented_point_normals(vtk_path, surface)
    stations = extract_profile_stations(surface, labels, normals)
    arrays: dict[str, np.ndarray] = {
        WEIGHT_ARRAY: weights.astype("<f8"),
        TRUTH_ARRAY: surface.pressure.astype("<f4"),
    }
    station_records = {}
    for station in stations:
        index_name, coordinate_name = station_array_names(station.station_id)
        arrays[index_name] = station.point_index.astype("<i8")
        arrays[coordinate_name] = station.coordinate.astype("<f8")
        station_records[station.station_id] = {
            "coordinate_id": station.coordinate_id,
            "point_count": int(len(station.point_index)),
            "coordinate_min": float(station.coordinate[0]),
            "coordinate_max": float(station.coordinate[-1]),
        }
    payload = npz_bytes(arrays)
    points = surface.points
    record = {
        "schema": CASE_SUPPORT_SCHEMA,
        "release_id": RELEASE_ID,
        "dataset_id": DATASET_ID,
        "case_id": case_id,
        "rules_version": SUPPORT_RULES_VERSION,
        "source": {
            "file_name": f"{case_id}.vtk",
            "alternate_file_name": f"DrivAer_{case_id}.vtk",
            "sha256": sha256_file(vtk_path),
            "size_bytes": Path(vtk_path).stat().st_size,
        },
        "native": {
            "association": "PointData",
            "point_count": surface.point_count,
            "polygon_count": surface.cell_count,
            "points_float32_sha256": array_sha256(points.astype("<f4")),
            "connected_component_point_counts": sorted(np.bincount(labels).tolist(), reverse=True),
            "bounding_box_m": {
                "min": [float(v) for v in points.min(axis=0)],
                "max": [float(v) for v in points.max(axis=0)],
            },
            "total_dual_area_m2": float(np.sum(weights)),
            "pressure_min": float(surface.pressure.min()),
            "pressure_max": float(surface.pressure.max()),
            "dynamic_head_m2_s2": DYNAMIC_HEAD,
        },
        "support_npz": {
            "member": f"{case_id}/support.npz",
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
            "arrays": {
                name: {
                    "dtype": arrays[name].dtype.str,
                    "shape": list(arrays[name].shape),
                    "sha256": array_sha256(arrays[name]),
                }
                for name in arrays
            },
        },
        "profile_stations": station_records,
    }
    return payload, record


def validate_case_support(record: Mapping[str, Any], arrays: Mapping[str, np.ndarray], case_id: str) -> None:
    """Fail closed unless the arrays are exactly those the record binds."""

    if (
        record.get("schema") != CASE_SUPPORT_SCHEMA
        or record.get("release_id") != RELEASE_ID
        or record.get("dataset_id") != DATASET_ID
        or record.get("case_id") != case_id
        or record.get("rules_version") != SUPPORT_RULES_VERSION
    ):
        raise ContractError(f"{case_id}: case-support identity differs")
    declared = record.get("support_npz", {}).get("arrays", {})
    if list(arrays) != list(EXPECTED_ARRAYS) or list(declared) != list(EXPECTED_ARRAYS):
        raise ContractError(f"{case_id}: support arrays differ from the contract")
    n = record.get("native", {}).get("point_count")
    for name, array in arrays.items():
        spec = declared[name]
        if (
            array.dtype.str != EXPECTED_DTYPES[name]
            or spec.get("dtype") != array.dtype.str
            or spec.get("shape") != list(array.shape)
            or spec.get("sha256") != array_sha256(array)
        ):
            raise ContractError(f"{case_id}: support array {name} differs from its record")
    weights, truth = arrays[WEIGHT_ARRAY], arrays[TRUTH_ARRAY]
    if weights.shape != (n,) or truth.shape != (n,) or not np.all(weights > 0) or not np.all(np.isfinite(truth)):
        raise ContractError(f"{case_id}: weights or reference pressure are invalid")
    for station in PROFILE_STATIONS:
        index_name, coordinate_name = station_array_names(station)
        index, coordinate = arrays[index_name], arrays[coordinate_name]
        if (
            index.ndim != 1
            or index.shape != coordinate.shape
            or len(index) < 2
            or index.min() < 0
            or index.max() >= n
            or np.any(np.diff(coordinate) <= 0)
        ):
            raise ContractError(f"{case_id}: profile station {station} is invalid")
    if not isinstance(record.get("support_npz", {}).get("sha256"), str) or not SHA256_RE.fullmatch(
        record["support_npz"]["sha256"]
    ):
        raise ContractError(f"{case_id}: support NPZ digest is invalid")
