#!/usr/bin/env python3
"""Template: pack a model's raw AirfRANS predictions into native-order .npz files.

This is the one model-specific step before the AirfRANS pipeline
(README.md, steps 1-4). Only ``load_raw_prediction`` depends on your model;
adapt it to however your inference script stores its outputs. Everything
else is generic.

Each output ``<case_id>.npz`` holds exactly the four arrays that
``scripts/airfrans.py validate-entry`` accepts: ``velocity`` (N, 2),
``pressure`` (N,), ``internal_points`` (N, 3) and ``airfoil_points`` (M, 3),
where N/M are the native ``_internal.vtu``/``_aerofoil.vtp`` point counts in
native point order. Any other array is rejected; airfoil pressure, wall shear
and forces are derived from these fields during evaluation.

Predictions must already be denormalized and in native point order. This
script checks that by comparing the model's coordinates to the native mesh
index by index and stops on any mismatch: it never interpolates, because
``discretization.json`` declares the domain mapping as ``identity``. A model
that predicts on a different point set needs its own explicit mapping, and
that mapping must then be declared in the package config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_FILE = ROOT / "benchmark-specs" / "airfrans" / "splits" / "full.json"


def load_raw_prediction(raw_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADAPT THIS: return (coords (N, 2), velocity (N, 2), pressure (N,)), denormalized.

    Default: ``<case_id>_preds.npz`` with ``coords`` (N, 2) and ``pred_fields``
    (N, 3) = (Ux, Uy, p), as written by the Transolver++ inference script.
    """
    with np.load(raw_dir / f"{case_id}_preds.npz") as data:
        coords = np.asarray(data["coords"], dtype=np.float64)
        fields = np.asarray(data["pred_fields"], dtype=np.float64)
    return coords, fields[:, :2], fields[:, 2]


def native_prediction_arrays(
    velocity: np.ndarray,
    pressure: np.ndarray,
    internal_points: np.ndarray,
    airfoil_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """The exact array set ``validate-entry`` accepts; nothing else may be added."""
    return {
        "velocity": velocity,
        "pressure": pressure,
        "internal_points": internal_points,
        "airfoil_points": airfoil_points,
    }


def pack_case(dataset_root: Path, raw_dir: Path, output_dir: Path, case_id: str, tolerance: float) -> None:
    import pyvista as pv

    case_dir = dataset_root / case_id
    internal_points = np.asarray(pv.read(case_dir / f"{case_id}_internal.vtu").points, dtype=np.float64)
    airfoil_points = np.asarray(pv.read(case_dir / f"{case_id}_aerofoil.vtp").points, dtype=np.float64)

    coords, velocity, pressure = load_raw_prediction(raw_dir, case_id)
    if coords.shape != (len(internal_points), 2):
        raise ValueError(
            f"{case_id}: {coords.shape[0]} predicted points vs {len(internal_points)} native points"
        )
    offset = float(np.max(np.abs(coords - internal_points[:, :2])))
    if offset > tolerance:
        raise ValueError(
            f"{case_id}: predictions are not in native point order (max coordinate offset "
            f"{offset:.3e} > {tolerance:.1e})"
        )
    if not (np.all(np.isfinite(velocity)) and np.all(np.isfinite(pressure))):
        raise ValueError(f"{case_id}: non-finite predicted values")

    np.savez_compressed(
        output_dir / f"{case_id}.npz",
        **native_prediction_arrays(velocity, pressure, internal_points, airfoil_points),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=1e-5,
        help="Max allowed |model coord - native coord|; covers float32 round-off.",
    )
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"output directory already exists, refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    case_ids = json.loads(args.split_file.read_text(encoding="utf-8"))["case_ids"]
    for index, case_id in enumerate(case_ids, start=1):
        pack_case(args.dataset_root, args.raw_predictions_dir, args.output, case_id, args.coordinate_tolerance)
        if index % 25 == 0 or index == len(case_ids):
            print(f"[{index}/{len(case_ids)}] packed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
