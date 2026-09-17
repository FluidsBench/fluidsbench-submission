#!/usr/bin/env python3
"""Generate per-case WindsorML profile support, or sweep its resolution.

For each frozen station this resolves every sample point to the native entity
that carries its value, and records that entity ID alongside the ground-truth
value. The evaluator then reads a participant's prediction at the same native
IDs, so a profile can never become a second prediction path.

Resolution is by nearest native cell/point centre within a bounding box around
the station, not a VTK cell locator: a locator over ~291M cells costs far more
than the few hundred points actually needed.

``--sweep`` evaluates several sample counts on one case and reports how much the
profile changes between them, which is the evidence the sample count must be
frozen on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/data"
)
DEFINITION = REPO_ROOT / "benchmark-specs" / "windsorml" / "profile-definition-v1.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_volume_field(path: Path, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (cell centres, field values) loading only the requested array."""

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    for index in range(reader.GetNumberOfCellArrays()):
        name = reader.GetCellArrayName(index)
        reader.SetCellArrayStatus(name, 1 if name == field else 0)
    reader.Update()
    grid = reader.GetOutput()
    values = vtk_to_numpy(grid.GetCellData().GetArray(field)).astype(np.float32)
    centres_filter = vtk.vtkCellCenters()
    centres_filter.SetInputData(grid)
    centres_filter.Update()
    centres = vtk_to_numpy(centres_filter.GetOutput().GetPoints().GetData()).astype(
        np.float32
    )
    return centres, values


def read_surface(path: Path, field: str) -> tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float32)
    values = vtk_to_numpy(grid.GetPointData().GetArray(field)).astype(np.float32)
    return points, values


def station_points(station: dict, sample_count: int) -> np.ndarray:
    start = np.asarray(station["start_m"], dtype=np.float64)
    end = np.asarray(station["end_m"], dtype=np.float64)
    t = np.linspace(0.0, 1.0, sample_count)
    return start[None, :] + t[:, None] * (end - start)[None, :]


def resolve_nearest(
    centres: np.ndarray, targets: np.ndarray, *, margin: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centre index for each target, restricted to a local box.

    A brute-force pairwise distance is not viable: the box around a wake
    station holds millions of cells in this mesh, so the (candidates x targets)
    block runs to gigabytes. Restrict to the box, then index the survivors with
    a KD-tree.
    """

    lo = targets.min(axis=0) - margin
    hi = targets.max(axis=0) + margin
    inside = np.all((centres >= lo) & (centres <= hi), axis=1)
    candidate_ids = np.nonzero(inside)[0]
    if candidate_ids.size == 0:
        raise SystemExit(f"no native entities within {margin} m of the station")

    from scipy.spatial import cKDTree

    tree = cKDTree(centres[candidate_ids].astype(np.float64))
    distances, local_index = tree.query(targets, k=1, workers=-1)
    return candidate_ids[local_index], distances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Report profile sensitivity across 64/96/128/160 samples.",
    )
    parser.add_argument("--sample-count", type=int, default=128)
    args = parser.parse_args()

    definition = json.loads(DEFINITION.read_text())
    u_ref = definition["velocity_profiles"]["reference_velocity_m_s"]
    velocity_stations = definition["velocity_profiles"]["stations"]

    run = args.run
    log(f"run_{run}: loading volume velocityxavg")
    centres, ux = read_volume_field(DATA / f"run_{run}" / f"volume_{run}.vtu", "velocityxavg")
    log(f"  {len(ux):,} cells")

    counts = (64, 96, 128, 160) if args.sweep else (args.sample_count,)
    resolved: dict[int, dict[str, np.ndarray]] = {}
    for count in counts:
        resolved[count] = {}
        for station in velocity_stations:
            targets = station_points(station, count)
            ids, distances = resolve_nearest(centres, targets)
            values = ux[ids].astype(np.float64) / u_ref
            resolved[count][station["id"]] = values
            if not args.sweep:
                log(
                    f"  {station['id']}: {count} samples, "
                    f"max snap distance {distances.max() * 1000:.2f} mm, "
                    f"ux/uinf in [{values.min():.4f}, {values.max():.4f}]"
                )

    if args.sweep:
        print("\nresolution sweep -- each profile resampled onto the 64-sample")
        print("coordinate grid and compared to the coarsest, as max |delta ux/uinf|:")
        print(f"  {'station':<32} " + "".join(f"{c:>10}" for c in counts))
        for station in velocity_stations:
            row = f"  {station['id']:<32} "
            base_grid = np.linspace(0.0, 1.0, 64)
            reference = resolved[64][station["id"]]
            for count in counts:
                values = resolved[count][station["id"]]
                grid = np.linspace(0.0, 1.0, count)
                interpolated = np.interp(base_grid, grid, values)
                row += f"{np.abs(interpolated - reference).max():10.5f}"
            print(row)
        print("\n  (column 64 is the self-comparison and must be exactly 0)")
        return 0

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": f"run_{run}",
            "sample_count": args.sample_count,
            "reference_velocity_m_s": u_ref,
            "velocity_profiles": {
                sid: values.tolist() for sid, values in resolved[args.sample_count].items()
            },
        }
        out = args.out_dir / f"run_{run}.json"
        out.write_text(json.dumps(payload) + "\n")
        log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
