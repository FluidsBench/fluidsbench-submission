#!/usr/bin/env python3
"""Build the AirfRANS native ground-truth scoring-support release (Layer 1).

This is a ground-truth-only, ran-once builder: it never reads a participant
prediction. For every case in a declared AirfRANS split it opens the native
``airfrans.Simulation`` (unmodified, so every quantity read from it is the
official ground truth), computes the point-dual physical weights described
in ``reference/airfrans/geometry.py``, and writes a schema
(``schemas/scoring-support/v1``)-compliant manifest -> case-set index ->
chunk -> materialized-table tree, exactly like every other FluidsBench
dataset's scoring-support release. The AirfRANS-specific evaluator (Layer 3)
and every future AirfRANS submission load this release through the existing
generic loader in ``reference/scoring_support.py`` -- nothing here is
AirfRANS-only at read time, only at build time.

Three supports are published, matching ``benchmark-specs/airfrans/
submission-spec.json``:

* ``two-dimensional-domain-native`` (volume): native ``_internal.vtu``
  points, dual-cell-area weights, ground-truth velocity/pressure.
* ``airfoil-curve-native`` (line): native ``_aerofoil.vtp`` points,
  dual-segment-length weights, ground-truth pressure and wall-shear-stress
  (the latter derived by the pinned airfrans library's own gradient
  machinery, ``Simulation.wallshearstress``).
* ``airfoil-force-coefficients-native`` (scalar_case): one row per case,
  ground-truth force / force coefficients from ``Simulation.force`` and
  ``Simulation.force_coefficient`` (exact re-integration of the -- here,
  ground-truth -- surface prediction, never submitted independently).

This release remains a *candidate*: AirfRANS submissions stay closed
(``scoring_support.status: owner_review_required`` in submission-spec.json)
until the dataset owner completes the ``owner_decisions_required`` list.
Re-running this script is expected to reproduce byte-identical output for a
fixed airfrans/pyvista/vtk pin, since it reads nothing but the public
dataset and computes everything deterministically.
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

from reference.airfrans.geometry import (  # noqa: E402
    AirfRANSGeometryError,
    assert_dual_weight_conservation,
    equal_share_dual_weights,
)
from reference.weightings import evaluator_weighting  # noqa: E402

RELEASE_ID = "airfrans-native-ground-truth-v1-candidate"
DATASET_ID = "airfrans"
DATASET_VERSION = "prototype-1"
EVALUATION_REFERENCE_VERSION = "prototype-1"

DOMAIN_SUPPORT_ID = "two-dimensional-domain-native"
CURVE_SUPPORT_ID = "airfoil-curve-native"
FORCE_SUPPORT_ID = "airfoil-force-coefficients-native"

DEFAULT_SPEC = ROOT / "benchmark-specs" / "airfrans" / "submission-spec.json"
DEFAULT_SPLIT_FILE = ROOT / "benchmark-specs" / "airfrans" / "splits" / "full.json"


class BuildError(RuntimeError):
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


def _metric_binding(metric_id, quantity_id, reduction, dataset_weighting, aggregation, case_evidence):
    return {
        "metric_id": metric_id,
        "quantity_id": quantity_id,
        "reduction": reduction,
        "weighting": evaluator_weighting(dataset_weighting),
        "dataset_weighting": dataset_weighting,
        "aggregation": aggregation,
        "case_evidence": case_evidence,
    }


def build_manifest(case_count: int) -> dict:
    domain_support = {
        "id": DOMAIN_SUPPORT_ID,
        "domain": "volume",
        "location_definition": {
            "mode": "materialized_table",
            "format": "npz",
            "artifact_role": "ground_truth_table",
            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
            "coordinate_fields": ["x", "y"],
            "weight_rule": {
                "kind": "artifact_field",
                "artifact_role": "ground_truth_table",
                "field": "weight",
            },
            "ordering": "support_id_ascending",
        },
        "quantities": [
            {
                "id": "velocity",
                "unit": "m/s",
                "components": [
                    {
                        "id": "velocity_x",
                        "target_field": "velocity_x",
                        "target_association": "table_field",
                        "prediction_field": "velocity_x_pred",
                    },
                    {
                        "id": "velocity_y",
                        "target_field": "velocity_y",
                        "target_association": "table_field",
                        "prediction_field": "velocity_y_pred",
                    },
                ],
            },
            {
                "id": "pressure",
                "unit": "m^2/s^2",
                "components": [
                    {
                        "id": "pressure",
                        "target_field": "pressure",
                        "target_association": "table_field",
                        "prediction_field": "pressure_pred",
                    }
                ],
            },
        ],
        "metric_bindings": [
            _metric_binding(
                "flow_domain_velocity_rel_l2", "velocity", "relative_l2_percent",
                "interior_nodes_equal", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "flow_domain_velocity_physical_rel_l2", "velocity", "relative_l2_percent",
                "interior_point_dual_area", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "flow_domain_velocity_rel_l1", "velocity", "relative_l1_percent",
                "interior_nodes_equal", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "flow_domain_pressure_rel_l2", "pressure", "relative_l2_percent",
                "interior_nodes_equal", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "flow_domain_pressure_physical_rel_l2", "pressure", "relative_l2_percent",
                "interior_point_dual_area", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "flow_domain_pressure_rel_l1", "pressure", "relative_l1_percent",
                "interior_nodes_equal", "per_geometry_then_macro_average", "metric_value",
            ),
        ],
        "required_coverage": {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0},
        "extrapolation_policy": "forbidden",
        "notes": (
            "Physical weight is mass-lumped dual cell area (equal-share lumping of "
            "pyvista.compute_cell_sizes Area); the official primary metric for this "
            "support is equal-entity (interior_nodes_equal), the dual-area weighting "
            "is the declared secondary/diagnostic variant."
        ),
    }

    curve_support = {
        "id": CURVE_SUPPORT_ID,
        "domain": "line",
        "location_definition": {
            "mode": "materialized_table",
            "format": "npz",
            "artifact_role": "ground_truth_table",
            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
            "coordinate_fields": ["x", "y"],
            "weight_rule": {
                "kind": "artifact_field",
                "artifact_role": "ground_truth_table",
                "field": "weight",
            },
            "ordering": "support_id_ascending",
        },
        "quantities": [
            {
                "id": "pressure",
                "unit": "m^2/s^2",
                "components": [
                    {
                        "id": "pressure",
                        "target_field": "pressure",
                        "target_association": "table_field",
                        "prediction_field": "pressure_pred",
                    }
                ],
            },
            {
                "id": "wall_shear_stress",
                "unit": "m^2/s^2",
                "components": [
                    {
                        "id": "wall_shear_stress_x",
                        "target_field": "wall_shear_stress_x",
                        "target_association": "table_field",
                        "prediction_field": "wall_shear_stress_x_pred",
                    },
                    {
                        "id": "wall_shear_stress_y",
                        "target_field": "wall_shear_stress_y",
                        "target_association": "table_field",
                        "prediction_field": "wall_shear_stress_y_pred",
                    },
                ],
            },
        ],
        "metric_bindings": [
            _metric_binding(
                "surface_pressure_rel_l2", "pressure", "relative_l2_percent",
                "boundary_point_dual_length", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "surface_pressure_equal_entity_rel_l2", "pressure", "relative_l2_percent",
                "boundary_nodes_equal", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "surface_pressure_rel_l1", "pressure", "relative_l1_percent",
                "boundary_point_dual_length", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "surface_wall_shear_rel_l2", "wall_shear_stress", "relative_l2_percent",
                "boundary_point_dual_length", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "surface_wall_shear_equal_entity_rel_l2", "wall_shear_stress", "relative_l2_percent",
                "boundary_nodes_equal", "per_geometry_then_macro_average", "metric_value",
            ),
            _metric_binding(
                "surface_wall_shear_rel_l1", "wall_shear_stress", "relative_l1_percent",
                "boundary_point_dual_length", "per_geometry_then_macro_average", "metric_value",
            ),
        ],
        "required_coverage": {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0},
        "extrapolation_policy": "forbidden",
        "notes": (
            "Physical weight is mass-lumped dual segment length (equal-share lumping "
            "of pyvista.compute_cell_sizes Length); this is the declared PRIMARY "
            "weighting for surface/curve metrics, boundary_nodes_equal is secondary. "
            "wall_shear_stress is the full 2-component vector reorganized onto the "
            "native _aerofoil.vtp point order via airfrans.reorganize, matching "
            "Simulation.wallshearstress(over_airfoil=True); it is NOT reduced to a "
            "scalar magnitude, per the dataset's vector_rule (one weight multiplies "
            "the squared vector magnitude)."
        ),
    }

    force_support = {
        "id": FORCE_SUPPORT_ID,
        "domain": "scalar_case",
        "location_definition": {
            "mode": "materialized_table",
            "format": "json",
            "artifact_role": "ground_truth_table",
            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
            "coordinate_fields": ["case_ordinal"],
            "weight_rule": {"kind": "uniform"},
            "ordering": "support_id_ascending",
        },
        "quantities": [
            {
                "id": "drag_coefficient",
                "unit": "dimensionless",
                "components": [
                    {
                        "id": "cd",
                        "target_field": "cd",
                        "target_association": "table_field",
                        "prediction_field": "cd_pred",
                    }
                ],
            },
            {
                "id": "lift_coefficient",
                "unit": "dimensionless",
                "components": [
                    {
                        "id": "cl",
                        "target_field": "cl",
                        "target_association": "table_field",
                        "prediction_field": "cl_pred",
                    }
                ],
            },
        ],
        "metric_bindings": [
            _metric_binding(
                "cd_r2", "drag_coefficient", "r2", "cases_equal", "all_test_cases", "aggregate_only",
            ),
            _metric_binding(
                "cl_r2", "lift_coefficient", "r2", "cases_equal", "all_test_cases", "aggregate_only",
            ),
            _metric_binding(
                "c_drag_mae", "drag_coefficient", "mae", "cases_equal", "all_test_cases", "metric_value",
            ),
            _metric_binding(
                "c_lift_mae", "lift_coefficient", "mae", "cases_equal", "all_test_cases", "metric_value",
            ),
        ],
        "required_coverage": {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0},
        "extrapolation_policy": "forbidden",
        "notes": (
            "One row per case (entity_count=1); force and force coefficients are "
            "derived by exact re-integration over the airfoil curve "
            "(Simulation.force / Simulation.force_coefficient), never submitted "
            "independently. coordinate_fields is a placeholder (case_ordinal=0): "
            "this support has no spatial coordinates."
        ),
    }

    return {
        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "status": "candidate",
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "evaluation_reference_version": EVALUATION_REFERENCE_VERSION,
        "coordinate_frame": {
            "id": "airfrans-native-physical-v1",
            "axis_order": ["x", "y"],
            "handedness": "not_applicable",
            "length_unit": "m",
            "normalization": "none",
        },
        "supports": [domain_support, curve_support, force_support],
        "case_sets": [
            {
                "id": "standard",
                "case_count": case_count,
                "index_file": "case-sets/standard/index.json",
                "index_sha256": None,  # filled in after the index file is written
            }
        ],
        "notes": (
            "Candidate AirfRANS ground-truth scoring-support release, built entirely "
            "from the public dataset with no participant predictions involved. "
            "AirfRANS submissions remain closed (submission-spec.json "
            "scoring_support.status=owner_review_required) pending the "
            "owner_decisions_required checklist; this release is not yet an official "
            "FluidsBench scoring-support publication."
        ),
    }


def build_case(sim, case_id: str, data_dir: Path) -> tuple[dict, dict]:
    from airfrans.reorganize import reorganize

    internal = sim.internal
    domain_weights = equal_share_dual_weights(internal, "Area")
    assert_dual_weight_conservation(internal, "Area", domain_weights)
    n_domain = internal.n_points
    domain_ids = np.array([f"{i:07d}" for i in range(n_domain)])
    domain_xy = sim.position
    domain_path = data_dir / f"{case_id}-domain.npz"
    domain_sha256 = write_npz(
        domain_path,
        support_id=domain_ids,
        x=domain_xy[:, 0].astype(np.float32),
        y=domain_xy[:, 1].astype(np.float32),
        weight=domain_weights.astype(np.float32),
        velocity_x=sim.velocity[:, 0].astype(np.float32),
        velocity_y=sim.velocity[:, 1].astype(np.float32),
        pressure=sim.pressure[:, 0].astype(np.float32),
    )

    airfoil = sim.airfoil
    curve_weights = equal_share_dual_weights(airfoil, "Length")
    assert_dual_weight_conservation(airfoil, "Length", curve_weights)
    n_curve = airfoil.n_points
    curve_ids = np.array([f"{i:07d}" for i in range(n_curve)])
    curve_xy = sim.airfoil_position
    curve_pressure = reorganize(
        sim.position[sim.surface], sim.airfoil_position, sim.pressure[sim.surface]
    )[:, 0]
    curve_wss = sim.wallshearstress(over_airfoil=True)
    if curve_wss.shape != (n_curve, 2):
        raise BuildError(f"{case_id}: unexpected wall-shear-stress shape {curve_wss.shape}")
    curve_path = data_dir / f"{case_id}-curve.npz"
    curve_sha256 = write_npz(
        curve_path,
        support_id=curve_ids,
        x=curve_xy[:, 0].astype(np.float32),
        y=curve_xy[:, 1].astype(np.float32),
        weight=curve_weights.astype(np.float32),
        pressure=curve_pressure.astype(np.float32),
        wall_shear_stress_x=curve_wss[:, 0].astype(np.float32),
        wall_shear_stress_y=curve_wss[:, 1].astype(np.float32),
    )

    force, force_p, force_v = sim.force()
    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient()
    force_payload = {
        "columns": {
            "support_id": ["0000000"],
            "case_ordinal": [0],
            "force_x": [float(force[0])],
            "force_y": [float(force[1])],
            "force_pressure_x": [float(force_p[0])],
            "force_pressure_y": [float(force_p[1])],
            "force_viscous_x": [float(force_v[0])],
            "force_viscous_y": [float(force_v[1])],
            "cd": [float(cd)],
            "cdp": [float(cdp)],
            "cdv": [float(cdv)],
            "cl": [float(cl)],
            "clp": [float(clp)],
            "clv": [float(clv)],
        }
    }
    force_path = data_dir / f"{case_id}-forces.json"
    force_sha256 = write_json(force_path, force_payload)

    chunk_case = {
        "case_id": case_id,
        "support_instances": [
            {
                "support_id": DOMAIN_SUPPORT_ID,
                "entity_count": int(n_domain),
                "artifacts": [
                    {
                        "role": "ground_truth_table",
                        "path": f"data/{domain_path.name}",
                        "sha256": domain_sha256,
                        "format": "npz",
                    }
                ],
            },
            {
                "support_id": CURVE_SUPPORT_ID,
                "entity_count": int(n_curve),
                "artifacts": [
                    {
                        "role": "ground_truth_table",
                        "path": f"data/{curve_path.name}",
                        "sha256": curve_sha256,
                        "format": "npz",
                    }
                ],
            },
            {
                "support_id": FORCE_SUPPORT_ID,
                "entity_count": 1,
                "artifacts": [
                    {
                        "role": "ground_truth_table",
                        "path": f"data/{force_path.name}",
                        "sha256": force_sha256,
                        "format": "json",
                    }
                ],
            },
        ],
    }
    summary = {
        "case_id": case_id,
        "n_domain": int(n_domain),
        "n_curve": int(n_curve),
        "domain_area_total": float(np.sum(internal.cell_data["Area"])),
        "curve_length_total": float(np.sum(airfoil.cell_data["Length"])),
        "force_x": float(force[0]),
        "force_y": float(force[1]),
        "cd": float(cd),
        "cl": float(cl),
    }
    return chunk_case, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N cases (debugging)")
    args = parser.parse_args(argv)

    import airfrans as af

    if args.output.exists():
        raise BuildError(f"output directory already exists, refusing to overwrite: {args.output}")

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    case_ids = split["case_ids"]
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    data_dir = args.output / "case-sets" / "standard" / "data"
    chunk_cases = []
    summaries = []
    t_start = time.time()
    for index, case_id in enumerate(case_ids):
        t0 = time.time()
        sim = af.Simulation(root=str(args.dataset_root), name=case_id)
        try:
            chunk_case, summary = build_case(sim, case_id, data_dir)
        except AirfRANSGeometryError as error:
            raise BuildError(f"{case_id}: dual-weight geometry check failed: {error}") from error
        chunk_cases.append(chunk_case)
        summaries.append(summary)
        elapsed = time.time() - t0
        print(
            f"[{index + 1}/{len(case_ids)}] {case_id} "
            f"n_domain={summary['n_domain']} n_curve={summary['n_curve']} "
            f"cd={summary['cd']:.6f} cl={summary['cl']:.6f} ({elapsed:.2f}s)",
            flush=True,
        )

    chunk_path = args.output / "case-sets" / "standard" / "chunk-000.json"
    chunk_sha256 = write_json(
        chunk_path,
        {
            "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
            "schema_version": "1.0",
            "release_id": RELEASE_ID,
            "dataset_id": DATASET_ID,
            "case_set_id": "standard",
            "cases": chunk_cases,
        },
    )

    index_path = args.output / "case-sets" / "standard" / "index.json"
    index_sha256 = write_json(
        index_path,
        {
            "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
            "schema_version": "1.0",
            "release_id": RELEASE_ID,
            "dataset_id": DATASET_ID,
            "case_set_id": "standard",
            "case_count": len(case_ids),
            "chunks": [
                {
                    "file": "chunk-000.json",
                    "sha256": chunk_sha256,
                    "case_count": len(case_ids),
                    "case_ids": case_ids,
                }
            ],
        },
    )

    manifest = build_manifest(len(case_ids))
    manifest["case_sets"][0]["index_sha256"] = index_sha256
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)

    summary_path = args.output / "build-summary.json"
    write_json(
        summary_path,
        {
            "release_id": RELEASE_ID,
            "case_count": len(case_ids),
            "elapsed_seconds": time.time() - t_start,
            "cases": summaries,
        },
    )

    print(f"\nWrote {len(case_ids)} cases to {args.output} in {time.time() - t_start:.1f}s")
    print(f"manifest.json sha256-referenced index sha256: {index_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
