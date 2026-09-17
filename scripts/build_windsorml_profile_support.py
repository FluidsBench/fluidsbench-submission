#!/usr/bin/env python3
"""Generate per-case WindsorML profile support for both placement families.

Each frozen station is resolved to the native entities that carry its values,
and the entity IDs are stored alongside the ground truth. The evaluator reads a
participant's prediction at those same IDs, so a profile can never become a
second prediction path.

Two families are produced per diagnostic kind, following the DrivAerML
relative-diagnostics contract:

* ``constant`` -- fixed absolute coordinates, identical in every case. Scored.
* ``relative`` -- the vertical coordinate rescaled by that case's body height
  ``h_case``. Report-only at zero weight until reviewed.

Only the vertical coordinate is relativised: WindsorML's streamwise and lateral
extents are identical across all 350 published runs, so a constant x or z
station already lands on the same physical location everywhere. Relative eta
values are anchored so each relative station reduces exactly to its constant
counterpart on run_0.

Resolution is nearest-native-entity inside a local bounding box, indexed with a
KD-tree; a VTK cell locator over ~291M cells is far more expensive than the few
hundred points actually needed.
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
DEFINITION = REPO_ROOT / "benchmark-specs" / "windsorml" / "profile-definition-v2.json"

SLAB_TOLERANCE_M = 0.004
BASE_X = 0.48325
NOSE_X = -0.56075
BODY_LENGTH = 1.044


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    cp = vtk_to_numpy(grid.GetPointData().GetArray("cpavg")).astype(np.float64)
    return points, cp


def read_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    for index in range(reader.GetNumberOfCellArrays()):
        name = reader.GetCellArrayName(index)
        reader.SetCellArrayStatus(name, 1 if name == "velocityxavg" else 0)
    reader.Update()
    grid = reader.GetOutput()
    ux = vtk_to_numpy(grid.GetCellData().GetArray("velocityxavg")).astype(np.float32)
    centres_filter = vtk.vtkCellCenters()
    centres_filter.SetInputData(grid)
    centres_filter.Update()
    centres = vtk_to_numpy(centres_filter.GetOutput().GetPoints().GetData()).astype(
        np.float32
    )
    return centres, ux


def nearest_in_box(
    positions: np.ndarray, targets: np.ndarray, *, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    lo = targets.min(axis=0) - margin
    hi = targets.max(axis=0) + margin
    inside = np.all((positions >= lo) & (positions <= hi), axis=1)
    candidate_ids = np.nonzero(inside)[0]
    if candidate_ids.size == 0:
        raise RuntimeError("no native entities near the station")
    from scipy.spatial import cKDTree

    tree = cKDTree(positions[candidate_ids].astype(np.float64))
    distances, local = tree.query(targets, k=1, workers=-1)
    return candidate_ids[local], distances


def extreme_along_axis(
    points: np.ndarray,
    *,
    slab_mask: np.ndarray,
    bin_axis: int,
    bin_values: np.ndarray,
    extreme_axis: int,
    largest: bool,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """For each bin centre, the slab point with the extreme coordinate.

    Used for surface curves such as "the upper centreline", which is not a
    nearest-neighbour query: at a given x there are points on both the roof and
    the underbody, and the station wants the roof.
    """

    ids = np.full(bin_values.shape[0], -1, dtype=np.int64)
    candidate_ids = np.nonzero(slab_mask)[0]
    if candidate_ids.size == 0:
        raise RuntimeError("station slab selected no boundary points")
    bin_coord = points[candidate_ids, bin_axis]
    extreme_coord = points[candidate_ids, extreme_axis]
    order = np.argsort(bin_coord)
    bin_sorted = bin_coord[order]
    for index, value in enumerate(bin_values):
        left = np.searchsorted(bin_sorted, value - half_width, side="left")
        right = np.searchsorted(bin_sorted, value + half_width, side="right")
        if right <= left:
            continue
        window = order[left:right]
        local = np.argmax(extreme_coord[window]) if largest else np.argmin(
            extreme_coord[window]
        )
        ids[index] = candidate_ids[window[local]]
    resolved = ids >= 0
    if not resolved.all():
        # Fill unresolved samples from the nearest resolved neighbour so the
        # series stays a complete 128-point curve; record how many.
        valid_positions = np.nonzero(resolved)[0]
        if valid_positions.size == 0:
            raise RuntimeError("station resolved no samples at all")
        for index in np.nonzero(~resolved)[0]:
            nearest = valid_positions[np.argmin(np.abs(valid_positions - index))]
            ids[index] = ids[nearest]
    return ids, resolved


def build_case(run: int, sample_count: int) -> dict:
    definition = json.loads(DEFINITION.read_text())
    u_ref = definition["velocity_stations"]["reference_velocity_m_s"]
    eta_span = definition["relative_frame"]["vertical_span_eta"]
    eta_cut = definition["relative_frame"]["horizontal_cut_eta"]

    log(f"run_{run}: reading boundary")
    points, cp = read_surface(DATA / f"run_{run}" / f"boundary_{run}.vtu")
    h_case = float(points[:, 1].max())
    log(f"  {len(cp):,} points, h_case = {h_case:.5f} m")

    families: dict[str, dict] = {
        "windsorml_cp_constant_v1": {},
        "windsorml_cp_relative_v1": {},
        "windsorml_velocity_constant_v1": {},
        "windsorml_velocity_relative_v1": {},
    }
    diagnostics: dict[str, float] = {}

    centreline_slab = np.abs(points[:, 2]) < SLAB_TOLERANCE_M
    x_samples = np.linspace(NOSE_X, BASE_X, sample_count)
    half_bin = 0.5 * (BASE_X - NOSE_X) / (sample_count - 1) + SLAB_TOLERANCE_M

    # --- cp: upper centreline (shared geometry, two parameterisations) -------
    ids, resolved = extreme_along_axis(
        points,
        slab_mask=centreline_slab,
        bin_axis=0,
        bin_values=x_samples,
        extreme_axis=1,
        largest=True,
        half_width=half_bin,
    )
    families["windsorml_cp_constant_v1"]["cp_centreline_upper"] = {
        "native_point_ids": ids.tolist(),
        "coordinate": x_samples.tolist(),
        "truth_cp": cp[ids].tolist(),
    }
    families["windsorml_cp_relative_v1"]["cp_centreline_upper_relative"] = {
        "native_point_ids": ids.tolist(),
        "coordinate": ((x_samples - NOSE_X) / BODY_LENGTH).tolist(),
        "truth_cp": cp[ids].tolist(),
    }
    diagnostics["cp_centreline_upper_unresolved"] = int((~resolved).sum())

    # --- cp: base vertical --------------------------------------------------
    base_slab = centreline_slab & (points[:, 0] > BASE_X - 0.01)
    for family, station, y_values, coordinate in (
        (
            "windsorml_cp_constant_v1",
            "cp_base_vertical",
            np.linspace(0.0, 0.5, sample_count),
            None,
        ),
        (
            "windsorml_cp_relative_v1",
            "cp_base_vertical_relative",
            np.linspace(0.0, 1.0, sample_count) * h_case,
            np.linspace(0.0, 1.0, sample_count),
        ),
    ):
        targets = np.column_stack(
            [np.full(sample_count, BASE_X), y_values, np.zeros(sample_count)]
        )
        subset = np.nonzero(base_slab)[0]
        from scipy.spatial import cKDTree

        tree = cKDTree(points[subset])
        _, local = tree.query(targets, k=1, workers=-1)
        base_ids = subset[local]
        families[family][station] = {
            "native_point_ids": base_ids.tolist(),
            "coordinate": (coordinate if coordinate is not None else y_values).tolist(),
            "truth_cp": cp[base_ids].tolist(),
        }

    # --- cp: side horizontal cut -------------------------------------------
    for family, station, y_cut in (
        ("windsorml_cp_constant_v1", "cp_side_horizontal_y_0p194", 0.194),
        ("windsorml_cp_relative_v1", "cp_side_horizontal_relative", eta_cut * h_case),
    ):
        slab = np.abs(points[:, 1] - y_cut) < SLAB_TOLERANCE_M
        if not slab.any():
            raise RuntimeError(f"{station}: no points at y = {y_cut:.5f}")
        side_ids, side_resolved = extreme_along_axis(
            points,
            slab_mask=slab,
            bin_axis=0,
            bin_values=x_samples,
            extreme_axis=2,
            largest=True,
            half_width=half_bin,
        )
        families[family][station] = {
            "native_point_ids": side_ids.tolist(),
            "coordinate": x_samples.tolist(),
            "truth_cp": cp[side_ids].tolist(),
            "cut_height_m": y_cut,
        }
        diagnostics[f"{station}_unresolved"] = int((~side_resolved).sum())

    del points, cp

    # --- velocity -----------------------------------------------------------
    log(f"run_{run}: reading volume velocityxavg")
    centres, ux = read_volume(DATA / f"run_{run}" / f"volume_{run}.vtu")
    log(f"  {len(ux):,} cells")

    constant = {s["id"]: s for s in definition["velocity_stations"]["constant"]}
    for station in definition["velocity_stations"]["relative"]:
        source = constant[station["source_station_id"]]
        if station["varying"] == "eta":
            eta = np.linspace(eta_span[0], eta_span[1], sample_count)
            y_values = eta * h_case
            targets = np.column_stack(
                [
                    np.full(sample_count, station["x_m"]),
                    y_values,
                    np.full(sample_count, station["z_m"]),
                ]
            )
            coordinate = eta
        else:
            z_values = np.linspace(*station["interval_m"], sample_count)
            targets = np.column_stack(
                [
                    np.full(sample_count, station["x_m"]),
                    np.full(sample_count, station["eta"] * h_case),
                    z_values,
                ]
            )
            coordinate = z_values
        ids, distances = nearest_in_box(centres, targets, margin=0.02)
        families["windsorml_velocity_relative_v1"][station["id"]] = {
            "native_cell_ids": ids.tolist(),
            "coordinate": coordinate.tolist(),
            "truth_ux_over_uinf": (ux[ids].astype(np.float64) / u_ref).tolist(),
            "max_snap_distance_m": float(distances.max()),
        }

    for station in definition["velocity_stations"]["constant"]:
        if station["varying"] == "y":
            y_values = np.linspace(*station["interval_m"], sample_count)
            targets = np.column_stack(
                [
                    np.full(sample_count, station["x_m"]),
                    y_values,
                    np.full(sample_count, station["z_m"]),
                ]
            )
            coordinate = y_values
        else:
            z_values = np.linspace(*station["interval_m"], sample_count)
            targets = np.column_stack(
                [
                    np.full(sample_count, station["x_m"]),
                    np.full(sample_count, station["y_m"]),
                    z_values,
                ]
            )
            coordinate = z_values
        ids, distances = nearest_in_box(centres, targets, margin=0.02)
        families["windsorml_velocity_constant_v1"][station["id"]] = {
            "native_cell_ids": ids.tolist(),
            "coordinate": coordinate.tolist(),
            "truth_ux_over_uinf": (ux[ids].astype(np.float64) / u_ref).tolist(),
            "max_snap_distance_m": float(distances.max()),
        }

    return {
        "schema": "windsorml-profile-support-v2",
        "case_id": f"run_{run}",
        "run_id": run,
        "sample_count": sample_count,
        "reference_velocity_m_s": u_ref,
        "body_height_m": h_case,
        "relative_frame": {"eta_span": eta_span, "horizontal_cut_eta": eta_cut},
        "diagnostics": diagnostics,
        "families": families,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="comma list, or start-stop")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if "-" in args.runs and "," not in args.runs:
        start, stop = args.runs.split("-")
        runs = list(range(int(start), int(stop) + 1))
    else:
        runs = [int(token) for token in args.runs.split(",") if token]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        out = args.out_dir / f"run_{run}.json"
        if args.skip_existing and out.is_file():
            log(f"run_{run}: already present, skipping")
            continue
        started = time.time()
        payload = build_case(run, args.sample_count)
        out.write_text(json.dumps(payload) + "\n")
        log(f"run_{run}: wrote {out.name} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
