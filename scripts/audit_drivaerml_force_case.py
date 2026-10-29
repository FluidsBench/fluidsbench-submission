#!/usr/bin/env python3
"""Replay one DrivAerML native-surface force case with chunk invariance.

This maintainer tool consumes the immutable boundary VTP and the already
published same-order surface-area array.  It audits the area array but never
regenerates or rewrites it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.surface_forces import (  # noqa: E402
    audit_fixed_surface_areas,
    finalize_force_coefficients,
    force_moment_chunk,
    iter_raw_id_ranges,
    surface_geometry_chunk_validated,
    validate_polygon_topology,
)


def sha256_file(path: Path, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def read_truth(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("per-case force truth must contain exactly one row")
    normalized = {key.lower(): value for key, value in rows[0].items()}
    required = {"cd", "cl", "clf", "clr", "cs"}
    if set(normalized) != required:
        raise ValueError("unexpected per-case force truth columns")
    values = {key: float(normalized[key]) for key in sorted(required)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("per-case force truth contains a non-finite value")
    return values


def load_native_surface(path: Path) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as error:  # pragma: no cover - depends on optional evaluator env
        raise RuntimeError(
            "VTK is required; use the pinned DrivAerML evaluator environment"
        ) from error

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    point_arrays = [
        reader.GetPointArrayName(index) for index in range(reader.GetNumberOfPointArrays())
    ]
    cell_arrays = [
        reader.GetCellArrayName(index) for index in range(reader.GetNumberOfCellArrays())
    ]
    for name in point_arrays:
        reader.SetPointArrayStatus(name, 0)
    for name in cell_arrays:
        reader.SetCellArrayStatus(name, 0)
    required = ("pMeanTrim", "wallShearStressMeanTrim")
    missing = sorted(set(required) - set(cell_arrays))
    if missing:
        raise ValueError(f"boundary VTP is missing required CellData arrays: {missing}")
    for name in required:
        reader.SetCellArrayStatus(name, 1)
    reader.Update()
    if reader.GetErrorCode():
        raise RuntimeError(f"VTK XML reader failed with code {reader.GetErrorCode()}")
    poly = reader.GetOutput()
    n_polygons = int(poly.GetNumberOfPolys())
    if (
        n_polygons < 1
        or poly.GetNumberOfCells() != n_polygons
        or poly.GetNumberOfVerts() != 0
        or poly.GetNumberOfLines() != 0
        or poly.GetNumberOfStrips() != 0
    ):
        raise ValueError("boundary VTP support must consist only of native polygons")

    points = vtk_to_numpy(poly.GetPoints().GetData())
    offsets = vtk_to_numpy(poly.GetPolys().GetOffsetsArray())
    connectivity = vtk_to_numpy(poly.GetPolys().GetConnectivityArray())
    pressure = vtk_to_numpy(poly.GetCellData().GetArray("pMeanTrim"))
    wall_shear = vtk_to_numpy(
        poly.GetCellData().GetArray("wallShearStressMeanTrim")
    )
    if pressure.shape != (n_polygons,) or wall_shear.shape != (n_polygons, 3):
        raise ValueError("surface CellData tuple/component counts are invalid")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(wall_shear)):
        raise ValueError("surface truth fields contain non-finite values")
    points, connectivity, offsets = validate_polygon_topology(
        points, connectivity, offsets
    )
    metadata = {
        "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        "point_count": int(points.shape[0]),
        "polygon_count": n_polygons,
        "available_point_arrays": point_arrays,
        "available_cell_arrays": cell_arrays,
        "pressure_dtype": str(pressure.dtype),
        "wall_shear_dtype": str(wall_shear.dtype),
    }
    return poly, points, connectivity, offsets, pressure, wall_shear, metadata


def replay(
    *,
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    pressure: np.ndarray,
    wall_shear: np.ndarray,
    published_areas: np.ndarray,
    chunk_polygons: int,
    audit_areas: bool,
) -> tuple[dict[str, object], dict[str, float | int]]:
    force_chunks = []
    area_receipts = []
    n_polygons = offsets.size - 1
    for start, stop in iter_raw_id_ranges(n_polygons, chunk_polygons):
        geometry = surface_geometry_chunk_validated(
            points, connectivity, offsets, start, stop
        )
        if audit_areas:
            area_receipts.append(
                audit_fixed_surface_areas(
                    geometry.areas_m2, published_areas[start:stop]
                )
            )
        force_chunks.append(
            force_moment_chunk(
                geometry,
                pressure[start:stop],
                wall_shear[start:stop],
            )
        )
    coefficients = finalize_force_coefficients(
        force_chunks, expected_entity_count=n_polygons
    )
    area_summary: dict[str, float | int] = {}
    if audit_areas:
        area_summary = {
            "entity_count": sum(int(row["entity_count"]) for row in area_receipts),
            "calculated_sum_m2": math.fsum(
                float(row["calculated_sum_m2"]) for row in area_receipts
            ),
            "published_sum_m2": math.fsum(
                float(row["published_sum_m2"]) for row in area_receipts
            ),
            "maximum_absolute_difference_m2": max(
                float(row["maximum_absolute_difference_m2"]) for row in area_receipts
            ),
            "maximum_relative_difference": max(
                float(row["maximum_relative_difference"]) for row in area_receipts
            ),
        }
    return coefficients, area_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--surface-areas", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-polygons", type=int, default=500_000)
    parser.add_argument("--comparison-chunk-polygons", type=int, default=333_333)
    parser.add_argument("--coefficient-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--chunk-invariance-tolerance", type=float, default=2.0e-12)
    args = parser.parse_args()
    if not args.case_id.startswith("run_"):
        raise ValueError("case ID must use canonical run_N form")
    started = time.time()

    poly, points, connectivity, offsets, pressure, wall_shear, mesh = load_native_surface(
        args.boundary
    )
    published_areas = np.load(args.surface_areas, mmap_mode="r", allow_pickle=False)
    if published_areas.shape != (offsets.size - 1,) or published_areas.dtype != np.dtype("<f4"):
        raise ValueError("published surface-area shape/dtype does not match native cells")

    primary, area_audit = replay(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure=pressure,
        wall_shear=wall_shear,
        published_areas=published_areas,
        chunk_polygons=args.chunk_polygons,
        audit_areas=True,
    )
    comparison, _ = replay(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure=pressure,
        wall_shear=wall_shear,
        published_areas=published_areas,
        chunk_polygons=args.comparison_chunk_polygons,
        audit_areas=False,
    )
    coefficient_keys = ("Cd", "Cl", "CmPitch", "Clf", "Clr", "Cs")
    chunk_differences = {
        key: abs(float(primary[key]) - float(comparison[key])) for key in coefficient_keys
    }
    maximum_chunk_difference = max(chunk_differences.values())
    if maximum_chunk_difference > args.chunk_invariance_tolerance:
        raise ValueError("full-case force result is not invariant to chunk partitioning")

    truth = read_truth(args.truth)
    truth_mapping = {"Cd": "cd", "Cl": "cl", "Clf": "clf", "Clr": "clr", "Cs": "cs"}
    truth_differences = {
        key: abs(float(primary[key]) - truth[truth_key])
        for key, truth_key in truth_mapping.items()
    }
    truth_cmpitch = (truth["clf"] - truth["clr"]) / 2.0
    truth_differences["CmPitch"] = abs(float(primary["CmPitch"]) - truth_cmpitch)
    passed = all(
        truth_differences[key] <= args.coefficient_tolerance
        for key in ("Cd", "Cl", "CmPitch", "Clf", "Clr", "Cs")
    )
    if not passed:
        raise ValueError("field-integrated coefficients exceed the frozen tolerance")

    receipt = {
        "schema": "drivaerml-native-surface-force-case-audit-v1",
        "status": "passed_candidate_evaluator_case_audit",
        "case_id": args.case_id,
        "source": {
            "boundary": {"path": str(args.boundary), "sha256": sha256_file(args.boundary)},
            "surface_areas": {
                "path": str(args.surface_areas),
                "sha256": sha256_file(args.surface_areas),
                "role": "fixed_input_not_regenerated",
            },
            "truth": {"path": str(args.truth), "sha256": sha256_file(args.truth)},
        },
        "mesh_and_fields": mesh,
        "units": {
            "coordinates": "m",
            "surface_areas": "m^2",
            "pMeanTrim": "m^2/s^2",
            "wallShearStressMeanTrim": "m^2/s^2",
        },
        "raw_cell_order": "unchanged zero-based VTK CellData tuple order",
        "area_audit": area_audit,
        "coefficients": primary,
        "truth": {**truth, "cmpitch": truth_cmpitch},
        "absolute_truth_difference": truth_differences,
        "coefficient_absolute_tolerance": args.coefficient_tolerance,
        "chunk_invariance": {
            "chunk_polygons": [args.chunk_polygons, args.comparison_chunk_polygons],
            "absolute_difference": chunk_differences,
            "maximum_absolute_difference": maximum_chunk_difference,
            "tolerance": args.chunk_invariance_tolerance,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "vtk": mesh["vtk_version"],
            "elapsed_seconds": time.time() - started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    # Keep the VTK-owned arrays alive until all NumPy views have been consumed.
    _ = poly


if __name__ == "__main__":
    main()
