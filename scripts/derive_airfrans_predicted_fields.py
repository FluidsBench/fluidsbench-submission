#!/usr/bin/env python3
"""Derive per-case AirfRANS predicted quantities from packed native predictions (Layer 2).

This is the per-submission counterpart to
``scripts/build_airfrans_scoring_support.py``: it still needs the pinned
airfrans/pyvista/vtk stack (see requirements-airfrans-evaluator.txt), but
unlike Layer 1 it reads a participant's predicted velocity/pressure and must
be rerun for every new candidate submission, not just once per dataset.

For every case it:

1. Loads the native ``airfrans.Simulation`` (ground truth, unmodified) purely
   to get the native mesh objects (``sim.internal``, ``sim.airfoil``) and to
   cross-check that the packed prediction's point coordinates are the exact,
   unpermuted native mesh -- this is the "preserve native point order, no
   remeshing" requirement from ``benchmark-specs/airfrans/
   methodology-contract.json``, enforced here as a hard failure rather than a
   silent mis-scoring risk.
2. Injects the predicted velocity/pressure into ``sim.velocity``/
   ``sim.pressure`` (the same in-place substitution
   ``examples/airfrans-profile-extraction/extract.py`` and the pinned
   airfrans library's own ``Simulation.force``/``wallshearstress`` use).
3. Derives predicted curve pressure the same way Layer 1 derived ground-truth
   curve pressure: ``reorganize`` of the (now-predicted) domain pressure at
   surface-flagged points onto the airfoil's native point order -- not from
   any separately packed "airfoil_pressure" array, so a submission is scored
   on a single, internally consistent domain-field prediction rather than two
   independently produced signals. (Checked against this repo's own
   transolverpp-full packing: for case
   airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548 the two are bit-for-bit
   identical, so this does not discard any information here -- it just makes
   Layer 2 self-contained regardless of how a future submission's predictions
   were packed.)
4. Derives predicted wall-shear-stress (``Simulation.wallshearstress(
   over_airfoil=True)``) and predicted force/force-coefficients
   (``Simulation.force``/``force_coefficient``) -- exact re-integration of
   the submitted fields, never taken from an independently submitted value.

Output tables use the exact ``support_id``/``*_pred`` column names declared
in the Layer 1 scoring-support manifest's ``quantities[].components[]``, so
Layer 3's evaluator can read them with the existing generic
``reference.scoring_support.load_scored_predictions``/``align_predictions``
-- nothing prediction-specific needs to be reinvented there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SPLIT_FILE = ROOT / "benchmark-specs" / "airfrans" / "splits" / "full.json"

DOMAIN_SUPPORT_ID = "two-dimensional-domain-native"
CURVE_SUPPORT_ID = "airfoil-curve-native"
FORCE_SUPPORT_ID = "airfoil-force-coefficients-native"


class DeriveError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def write_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def load_packed_prediction(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise DeriveError(f"missing packed prediction file: {path}")
    with np.load(path) as payload:
        required = {"velocity", "pressure", "internal_points", "airfoil_points"}
        missing = required - set(payload.files)
        if missing:
            raise DeriveError(f"{path} is missing required arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in payload.files}


def derive_case(sim, case_id: str, prediction: dict[str, np.ndarray], data_dir: Path) -> tuple[dict, dict]:
    from airfrans.reorganize import reorganize

    n_domain = sim.internal.n_points
    n_curve = sim.airfoil.n_points

    velocity = np.asarray(prediction["velocity"], dtype=np.float64)
    pressure = np.asarray(prediction["pressure"], dtype=np.float64)
    if velocity.shape != (n_domain, 2):
        raise DeriveError(f"{case_id}: predicted velocity shape {velocity.shape} != ({n_domain}, 2)")
    if pressure.shape not in ((n_domain,), (n_domain, 1)):
        raise DeriveError(f"{case_id}: predicted pressure shape {pressure.shape} != ({n_domain},)")
    pressure = pressure.reshape(n_domain)
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(pressure)):
        raise DeriveError(f"{case_id}: predicted domain fields contain non-finite values")

    internal_points = np.asarray(prediction["internal_points"])[:, :2]
    airfoil_points = np.asarray(prediction["airfoil_points"])[:, :2]
    if not np.allclose(internal_points, sim.position, rtol=0.0, atol=1e-9):
        raise DeriveError(
            f"{case_id}: packed internal_points do not exactly match the native mesh "
            "point order (native-point-order requirement violated)"
        )
    if not np.allclose(airfoil_points, sim.airfoil_position, rtol=0.0, atol=1e-9):
        raise DeriveError(
            f"{case_id}: packed airfoil_points do not exactly match the native "
            "airfoil-curve point order (native-point-order requirement violated)"
        )

    # Inject predictions in place, exactly like extract.py's prediction path
    # and the pinned airfrans library's own force()/wallshearstress().
    sim.velocity = velocity
    sim.pressure = pressure[:, None]

    domain_ids = np.array([f"{i:07d}" for i in range(n_domain)])
    domain_path = data_dir / f"{case_id}-domain-predictions.npz"
    domain_sha256 = write_npz(
        domain_path,
        support_id=domain_ids,
        velocity_x_pred=velocity[:, 0].astype(np.float32),
        velocity_y_pred=velocity[:, 1].astype(np.float32),
        pressure_pred=pressure.astype(np.float32),
    )

    curve_pressure_pred = reorganize(
        sim.position[sim.surface], sim.airfoil_position, sim.pressure[sim.surface]
    )[:, 0]
    curve_wss_pred = sim.wallshearstress(over_airfoil=True)
    if curve_wss_pred.shape != (n_curve, 2):
        raise DeriveError(f"{case_id}: unexpected predicted WSS shape {curve_wss_pred.shape}")
    curve_ids = np.array([f"{i:07d}" for i in range(n_curve)])
    curve_path = data_dir / f"{case_id}-curve-predictions.npz"
    curve_sha256 = write_npz(
        curve_path,
        support_id=curve_ids,
        pressure_pred=curve_pressure_pred.astype(np.float32),
        wall_shear_stress_x_pred=curve_wss_pred[:, 0].astype(np.float32),
        wall_shear_stress_y_pred=curve_wss_pred[:, 1].astype(np.float32),
    )

    force, force_p, force_v = sim.force()
    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient()
    force_payload = {
        "columns": {
            "support_id": ["0000000"],
            "force_x_pred": [float(force[0])],
            "force_y_pred": [float(force[1])],
            "force_pressure_x_pred": [float(force_p[0])],
            "force_pressure_y_pred": [float(force_p[1])],
            "force_viscous_x_pred": [float(force_v[0])],
            "force_viscous_y_pred": [float(force_v[1])],
            "cd_pred": [float(cd)],
            "cdp_pred": [float(cdp)],
            "cdv_pred": [float(cdv)],
            "cl_pred": [float(cl)],
            "clp_pred": [float(clp)],
            "clv_pred": [float(clv)],
        }
    }
    force_path = data_dir / f"{case_id}-force-predictions.json"
    force_sha256 = write_json(force_path, force_payload)

    # File shape matches what reference.evaluate_predictions.evaluate_prediction_artifact
    # expects for a "kind": "scored_predictions" case declaration: support_id/file/
    # sha256/format/row_count, nothing AirfRANS-specific about the shape itself.
    manifest_case = {
        "case_id": case_id,
        "files": [
            {
                "support_id": DOMAIN_SUPPORT_ID,
                "file": f"data/{domain_path.name}",
                "sha256": domain_sha256,
                "format": "npz",
                "row_count": int(n_domain),
            },
            {
                "support_id": CURVE_SUPPORT_ID,
                "file": f"data/{curve_path.name}",
                "sha256": curve_sha256,
                "format": "npz",
                "row_count": int(n_curve),
            },
            {
                "support_id": FORCE_SUPPORT_ID,
                "file": f"data/{force_path.name}",
                "sha256": force_sha256,
                "format": "json",
                "row_count": 1,
            },
        ],
    }
    summary = {
        "case_id": case_id,
        "n_domain": int(n_domain),
        "n_curve": int(n_curve),
        "cd_pred": float(cd),
        "cl_pred": float(cl),
    }
    return manifest_case, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--scoring-support-manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    import airfrans as af

    if args.output.exists():
        raise DeriveError(f"output directory already exists, refusing to overwrite: {args.output}")

    support_manifest = json.loads(args.scoring_support_manifest.read_text(encoding="utf-8"))
    support_release_id = support_manifest["release_id"]
    support_manifest_sha256 = sha256_file(args.scoring_support_manifest)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    case_ids = split["case_ids"]
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    data_dir = args.output / "data"
    cases = []
    summaries = []
    t_start = time.time()
    for index, case_id in enumerate(case_ids):
        t0 = time.time()
        sim = af.Simulation(root=str(args.dataset_root), name=case_id)
        prediction = load_packed_prediction(args.predictions_dir / f"{case_id}.npz")
        manifest_case, summary = derive_case(sim, case_id, prediction, data_dir)
        cases.append(manifest_case)
        summaries.append(summary)
        print(
            f"[{index + 1}/{len(case_ids)}] {case_id} "
            f"cd_pred={summary['cd_pred']:.6f} cl_pred={summary['cl_pred']:.6f} "
            f"({time.time() - t0:.2f}s)",
            flush=True,
        )

    manifest = {
        "kind": "scored_predictions",
        "submission_id": args.submission_id,
        "dataset_id": "airfrans",
        "split_id": split["split_id"],
        "case_set_id": split["case_set_id"],
        "support_release_id": support_release_id,
        "support_manifest_sha256": support_manifest_sha256,
        "case_count": len(case_ids),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": cases,
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "build-summary.json",
        {
            "submission_id": args.submission_id,
            "case_count": len(case_ids),
            "elapsed_seconds": time.time() - t_start,
            "cases": summaries,
        },
    )
    print(f"\nWrote {len(case_ids)} cases to {args.output} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
