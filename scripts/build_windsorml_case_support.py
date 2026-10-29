#!/usr/bin/env python3
"""Build one shard of the WindsorML per-case support record.

For each run this records the facts the FluidsBench contract must pin:

* SHA-256 and size of every per-case source file;
* native entity counts (boundary points, volume cells) read from VTU metadata
  only -- a full read of a 21 GB volume file needs >100 GB of RAM;
* the published force coefficients;
* the force coefficients re-integrated from the surface coefficient fields
  against the published dual-area weights.

The replay is a cross-check, not the truth source: WindsorML's published point
dual-area quadrature approximates the solver's exact cell integration, so it
agrees with the CSV to a fraction of a percent rather than to 1e-6. Collecting
the replay deltas across all 350 runs is what lets the contract freeze an
honest tolerance instead of a guessed one.

Axis convention (measured, not assumed): drag=+x, lift=+y, side=+z. The Windsor
body sits on a ground plane at y=0 and is laterally symmetric about z=0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import vtk


DATASET_ROOT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/data"
)
REPOSITORY_ID = "neashton/windsorml"
REPOSITORY_REVISION = "8a6ca32ae22c94f54df2186d1b0ccf9662a294c2"
A_REF = 0.112  # m^2, the constant reference area behind force_mom_*.csv

# Surface force axes: index into the (x, y, z) component order.
DRAG_AXIS, LIFT_AXIS, SIDE_AXIS = 0, 1, 2

SURFACE_FIELDS = ("cpavg", "cfxavg", "cfyavg", "cfzavg")
VOLUME_FIELDS = ("velocityxavg", "velocityyavg", "velocityzavg", "pressureavg")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path, chunk_bytes: int = 64 * 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def vtu_counts(path: Path) -> tuple[int, int, list[str], list[str]]:
    """Entity counts and array names without loading payloads."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    point_arrays = [reader.GetPointArrayName(i) for i in range(reader.GetNumberOfPointArrays())]
    cell_arrays = [reader.GetCellArrayName(i) for i in range(reader.GetNumberOfCellArrays())]
    # Counts live in the Piece header; parse them from the XML preamble.
    head = path.open("rb").read(4096).decode("utf-8", "replace")
    n_points = n_cells = -1
    marker = head.find("<Piece")
    if marker >= 0:
        piece = head[marker : head.find(">", marker)]
        for token in piece.split():
            if token.startswith("NumberOfPoints="):
                n_points = int(token.split('"')[1])
            elif token.startswith("NumberOfCells="):
                n_cells = int(token.split('"')[1])
    return n_points, n_cells, point_arrays, cell_arrays


def read_force_csv(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.open()))
    header = [h.strip() for h in rows[0]]
    return {k: float(v) for k, v in zip(header, rows[1], strict=True)}


def replay_forces(boundary: Path, dual_area: Path) -> dict[str, float]:
    """Re-integrate cd/cl/cs from the surface coefficient fields.

    C = sum_i (-cp_i * n_i + cf_i) * w_i / A_ref, with w_i the published
    barycentric dual area for native point i and n_i the outward unit normal.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(boundary))
    reader.Update()
    grid = reader.GetOutput()
    pd = grid.GetPointData()

    def array(name: str) -> np.ndarray:
        from vtk.util.numpy_support import vtk_to_numpy

        vtk_array = pd.GetArray(name)
        if vtk_array is None:
            raise RuntimeError(f"{boundary.name} is missing PointData array {name!r}")
        return vtk_to_numpy(vtk_array).astype(np.float64)

    normals = array("Normals")
    cp = array("cpavg")
    cf = np.stack([array(f"cf{axis}avg") for axis in "xyz"], axis=1)
    weights = np.load(dual_area).astype(np.float64)
    if weights.shape[0] != cp.shape[0]:
        raise RuntimeError(
            f"{dual_area.name} has {weights.shape[0]} weights for "
            f"{cp.shape[0]} native points"
        )

    contribution = (-cp[:, None] * normals + cf) * weights[:, None]
    total = contribution.sum(axis=0) / A_REF
    return {
        "cd": float(total[DRAG_AXIS]),
        "cl": float(total[LIFT_AXIS]),
        "cs": float(total[SIDE_AXIS]),
        "dual_area_sum": float(weights.sum()),
        "closure_residual": float(np.abs((weights[:, None] * normals).sum(axis=0)).max()),
    }


def build_case(run_id: int, *, skip_volume_hash: bool) -> dict:
    run = DATASET_ROOT / f"run_{run_id}"
    files = {
        "boundary": run / f"boundary_{run_id}.vtu",
        "volume": run / f"volume_{run_id}.vtu",
        "surface_dual_area": run / f"boundary_dual_area_{run_id}.npy",
        "geometry_parameters": run / f"geo_parameters_{run_id}.csv",
        "force_coefficients": run / f"force_mom_{run_id}.csv",
        "force_coefficients_varref": run / f"force_mom_varref_{run_id}.csv",
        "geometry_stl": run / f"windsor_{run_id}.stl",
    }
    record: dict = {"case_id": f"run_{run_id}", "run_id": run_id, "files": {}}

    for key, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
        if key == "volume" and skip_volume_hash:
            record["files"][key] = {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": None,
                "size_bytes": path.stat().st_size,
            }
            continue
        digest, size = sha256_file(path)
        record["files"][key] = {
            "path": str(path.relative_to(DATASET_ROOT)),
            "sha256": digest,
            "size_bytes": size,
        }

    s_pts, s_cells, s_point_arrays, s_cell_arrays = vtu_counts(files["boundary"])
    v_pts, v_cells, v_point_arrays, v_cell_arrays = vtu_counts(files["volume"])
    record["surface_entity_count"] = s_pts
    record["surface_cell_count"] = s_cells
    record["volume_entity_count"] = v_cells
    record["volume_point_count"] = v_pts
    record["surface_point_arrays"] = s_point_arrays
    record["surface_cell_arrays"] = s_cell_arrays
    record["volume_cell_arrays"] = v_cell_arrays

    missing_surface = [f for f in SURFACE_FIELDS if f not in s_point_arrays]
    missing_volume = [f for f in VOLUME_FIELDS if f not in v_cell_arrays]
    if missing_surface or missing_volume:
        raise RuntimeError(
            f"run_{run_id} missing contract fields: "
            f"surface={missing_surface} volume={missing_volume}"
        )

    published = read_force_csv(files["force_coefficients"])
    record["published_forces"] = published
    replay = replay_forces(files["boundary"], files["surface_dual_area"])
    record["replay_forces"] = replay
    record["replay_delta"] = {
        key: replay[key] - published[key] for key in ("cd", "cl", "cs")
    }
    record["replay_relative_delta"] = {
        key: (replay[key] - published[key]) / published[key]
        if published[key] != 0
        else None
        for key in ("cd", "cl", "cs")
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-volume-hash",
        action="store_true",
        help="Skip the 21 GB volume SHA-256 (fast structural pass).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_ids = [r for r in range(350) if r % args.num_shards == args.task_id]
    log(f"shard {args.task_id}/{args.num_shards}: {len(run_ids)} runs")

    records = []
    for index, run_id in enumerate(run_ids, start=1):
        started = time.time()
        records.append(build_case(run_id, skip_volume_hash=args.skip_volume_hash))
        log(f"  [{index}/{len(run_ids)}] run_{run_id} in {time.time() - started:.0f}s")

    out = args.out_dir / f"shard-{args.task_id:02d}.json"
    out.write_text(
        json.dumps(
            {
                "repository_id": REPOSITORY_ID,
                "repository_revision": REPOSITORY_REVISION,
                "a_ref": A_REF,
                "axis_convention": {"drag": "+x", "lift": "+y", "side": "+z"},
                "task_id": args.task_id,
                "num_shards": args.num_shards,
                "volume_hash_skipped": args.skip_volume_hash,
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
