#!/usr/bin/env python3
"""Measure WindsorML wake structure to site velocity-profile stations from data.

AutoCFD Case 1 is the same 1/4-scale Windsor body (1044 x 389 x 289 mm,
A_ref 0.112 m^2) and measures 2D PIV on constant-x wake planes plus a symmetry
plane and a horizontal cut. Its published plane coordinates cannot be mapped
into the WindsorML frame without the AutoCFD origin, which is not stated on the
public case page -- so rather than guess it, this locates the equivalent
physics directly in the WindsorML data.

Reports, for one case:
  * the streamwise centreline recovery of velocityxavg behind the base, which
    fixes the recirculation length and therefore where wake planes should sit;
  * vertical velocity profiles at candidate stations;
  * the body height, so a horizontal cut can be expressed sensibly.

Cell centres are computed once and selected with NumPy rather than probing with
a VTK locator, which would have to index ~291M cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

DATA = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/data"
)
BASE_X = 0.48325  # rear face; identical for every published variant
NOSE_X = -0.56075
BODY_LENGTH = BASE_X - NOSE_X
HALF_WIDTH = 0.1945


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--tube-radius", type=float, default=0.006)
    args = parser.parse_args()

    boundary = pv.read(DATA / f"run_{args.run}" / f"boundary_{args.run}.vtu")
    body_top = float(np.asarray(boundary.points)[:, 1].max())
    print(f"run_{args.run}: body top y = {body_top:.5f} m, base x = {BASE_X}, "
          f"length = {BODY_LENGTH:.5f} m")

    # Load only velocityxavg: the file carries ten cell arrays and skipping the
    # other nine saves about 10 GB on a 291M-cell mesh.
    print("reading volume (velocityxavg only)...", flush=True)
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(DATA / f"run_{args.run}" / f"volume_{args.run}.vtu"))
    reader.UpdateInformation()
    for index in range(reader.GetNumberOfCellArrays()):
        name = reader.GetCellArrayName(index)
        reader.SetCellArrayStatus(name, 1 if name == "velocityxavg" else 0)
    reader.SetPointArrayStatus("", 0)
    reader.Update()
    volume = pv.wrap(reader.GetOutput())
    print(f"  {volume.n_cells:,} cells", flush=True)

    ux = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("velocityxavg")).astype(
        np.float32
    )
    centre_filter = vtk.vtkCellCenters()
    centre_filter.SetInputData(reader.GetOutput())
    centre_filter.Update()
    centres = vtk_to_numpy(centre_filter.GetOutput().GetPoints().GetData()).astype(
        np.float32
    )
    print("  cell centres computed", flush=True)

    # 1. Centreline recovery behind the base, at mid-body height.
    mid_y = 0.5 * body_top
    tube = (
        (np.abs(centres[:, 2]) < args.tube_radius)
        & (np.abs(centres[:, 1] - mid_y) < args.tube_radius)
        & (centres[:, 0] > BASE_X)
        & (centres[:, 0] < BASE_X + 1.2)
    )
    x = centres[tube, 0]
    u = ux[tube]
    order = np.argsort(x)
    x, u = x[order], u[order]
    edges = np.arange(BASE_X, BASE_X + 1.2, 0.01, dtype=np.float32)
    idx = np.digitize(x, edges)
    print(f"\ncentreline wake recovery at y={mid_y:.4f} (z=0), "
          f"{tube.sum():,} cells in tube:")
    print(f"  {'x-x_base (m)':>13} {'x/L':>7} {'Ux (m/s)':>10}")
    u_inf = None
    reattach = None
    previous = None
    for b in range(1, len(edges)):
        sel = idx == b
        if not sel.any():
            continue
        xb = float(edges[b - 1] - BASE_X + 0.005)
        ub = float(np.mean(u[sel]))
        if u_inf is None and xb > 1.0:
            u_inf = ub
        if previous is not None and previous < 0.0 <= ub and reattach is None:
            reattach = xb
        previous = ub
        if b % 4 == 1 or xb < 0.3:
            print(f"  {xb:13.3f} {xb / BODY_LENGTH:7.3f} {ub:10.3f}")
    if reattach is not None:
        print(f"\n  streamwise velocity recovers to zero at "
              f"x-x_base = {reattach:.3f} m ({reattach / BODY_LENGTH:.3f} L)")

    # 2. Vertical profiles at candidate wake stations.
    print("\nvertical profiles on the symmetry plane (z=0):")
    for fraction in (0.05, 0.10, 0.25, 0.50, 1.00):
        station = BASE_X + fraction * BODY_LENGTH
        sel = (
            (np.abs(centres[:, 2]) < args.tube_radius)
            & (np.abs(centres[:, 0] - station) < args.tube_radius)
            & (centres[:, 1] > 0.0)
            & (centres[:, 1] < 2.0 * body_top)
        )
        if not sel.any():
            print(f"  x/L={fraction:.2f}: no cells")
            continue
        y = centres[sel, 1]
        uu = ux[sel]
        o = np.argsort(y)
        y, uu = y[o], uu[o]
        print(f"  x-x_base={fraction * BODY_LENGTH:6.3f} m (x/L={fraction:.2f}): "
              f"{sel.sum():6,} cells, y in [{y.min():.3f}, {y.max():.3f}], "
              f"Ux in [{uu.min():7.3f}, {uu.max():7.3f}]")

    # 3. Free-stream reference.
    far = (
        (centres[:, 0] > BASE_X + 1.5)
        & (centres[:, 1] > 1.5 * body_top)
        & (np.abs(centres[:, 2]) < 0.05)
    )
    if far.any():
        print(f"\nfar-field Ux (x>base+1.5, y>1.5*h): {float(np.mean(ux[far])):.4f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
