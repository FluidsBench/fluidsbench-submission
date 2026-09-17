#!/usr/bin/env python3
"""Regenerate the WindsorML submission spec from frozen decisions.

Replaces the prototype spec, whose splits were a single fabricated identifier
(``windsorml_default_test_0001``) and whose status was ``prototype_dummy_data``.

Four decisions are baked in here:

* **Scored case sets are the official manifest intersected with the published
  runs** (233 unique test cases). ``splits/*.json`` record both counts.
* **Forces use the constant reference area, 0.112 m^2**, and are scored by
  integrating truth and prediction identically; the published CSV is the audit
  anchor, not the scoring truth.
* **Volume metrics are equal-cell weighted for v1.** The physical
  (cell-volume-weighted) variants are removed rather than left unimplemented,
  because generating the sidecars costs roughly 400 GB and 1,000 CPU-hours.
* **Profile panels are declared but not scored in v1.** The prototype's four
  velocity stations were fabricated placeholders, so the component weights are
  renormalised over the field and force metrics that v1 can actually compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "benchmark-specs" / "windsorml"

DATASET_VERSION = "windsorml-native-v1-candidate"
EVALUATION_REFERENCE_VERSION = "windsorml-evaluator-v0.1-candidate"

# Candidate resolution; the stratified 64/96/128/160 sweep is still outstanding,
# so this is recorded as a candidate rather than frozen.
PROFILE_SAMPLE_COUNT = 128

# Prototype weights were field 0.50, force 0.25, diagnostics 0.25. Dropping the
# unscored diagnostics and renormalising over the remaining 0.75 preserves the
# field:force ratio exactly.
SCORED_COMPONENTS = (
    ("surface_pressure_rel_l2", 0.15, "bounded_error", 15.0),
    ("surface_wall_shear_rel_l2", 0.10, "bounded_error", 20.0),
    ("volume_velocity_rel_l2", 0.15, "bounded_error", 12.0),
    ("volume_pressure_rel_l2", 0.10, "bounded_error", 15.0),
    ("cd_r2", 0.15, "bounded_quality", None),
    ("cl_r2", 0.10, "bounded_quality", None),
)


def rel_l2(metric_id: str, weighting: str, aggregation: str = "per_geometry_then_macro_average") -> dict:
    return {
        "id": metric_id,
        "unit": "%",
        "direction": "lower",
        "kind": "error",
        "equation": "100\\,\\frac{\\sqrt{\\sum_i w_i(\\hat{y}_i-y_i)^2}}{\\sqrt{\\sum_i w_i y_i^2}}",
        "aggregation": aggregation,
        "weighting": weighting,
    }


def r2(metric_id: str) -> dict:
    return {
        "id": metric_id,
        "unit": "",
        "direction": "higher",
        "kind": "r2",
        "equation": "1-\\frac{\\sum_i(y_i-\\hat{y}_i)^2}{\\sum_i(y_i-\\bar{y})^2}",
        "aggregation": "all_test_cases",
        "weighting": "cases_equal",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SPEC_DIR / "submission-spec.json")
    args = parser.parse_args()

    index_path = SPEC_DIR / "splits" / "splits-index.json"
    if not index_path.is_file():
        raise SystemExit(f"run build_windsorml_splits.py first: {index_path} missing")
    splits = json.loads(index_path.read_text())

    total = sum(weight for _, weight, _, _ in SCORED_COMPONENTS)
    components = []
    for metric_id, weight, transform, cap in SCORED_COMPONENTS:
        entry = {
            "metric_id": metric_id,
            "weight": weight / total,
            "transform": transform,
        }
        if cap is not None:
            entry["cap"] = cap
        components.append(entry)

    document = {
        "schema_version": "1.2",
        "dataset_id": "windsorml",
        "dataset_name": "WindsorML",
        "dataset_version": DATASET_VERSION,
        "status": "candidate_native_support",
        "scoring_support": {
            "status": "owner_review_required",
            "submissions_open": False,
            "closed_reason": "Candidate support is under owner review.",
            "public_source": {
                "repository_id": "neashton/windsorml",
                "revision": "8a6ca32ae22c94f54df2186d1b0ccf9662a294c2",
                "identity_file": (
                    "public-source-identity/windsorml-public-source-identity-v1.json"
                ),
                "published_case_count": 350,
                "note": (
                    "The aggregate tables and the official split manifest enumerate "
                    "355 design variants, but run_350..run_354 have no per-run "
                    "payload in the public release and are therefore unscoreable."
                ),
            },
            "coverage_contract": {
                "dimensionality": "3D surface and 3D flow domain",
                "rule": "complete_original_public_release_entities",
                "inference_may_be_chunked": True,
                "chunk_metrics_must_use_additive_sufficient_statistics": True,
                "complete_case_and_entity_coverage_required": True,
                "full_prediction_artifact_required": False,
                "case_aggregation": "calculate_each_case_then_macro_average_cases_equally",
                "public_ground_truth": "Ground-truth fields are in the public dataset release.",
            },
            "public_supports": [
                {
                    "id": "windsorml_surface_native_points",
                    "public_file": "run_<case>/boundary_<case>.vtu",
                    "format": "VTK UnstructuredGrid boundary mesh",
                    "domain": "three_dimensional_body_surface",
                    "association": "PointData",
                    "entities": "all_native_boundary_points",
                    "arrays": ["cpavg", "cfxavg", "cfyavg", "cfzavg"],
                    "available_point_arrays": [
                        "cpavg", "cfxavg", "cfyavg", "cfzavg", "cpvar", "yplusavg", "Normals",
                    ],
                    "units_note": (
                        "cpavg and the cfxavg/cfyavg/cfzavg skin-friction components "
                        "are dimensionless coefficients."
                    ),
                    "physical_weight": "published_barycentric_dual_area",
                    "physical_weight_file": "run_<case>/boundary_dual_area_<case>.npy",
                    "physical_weight_note": (
                        "The published per-point dual areas are canonical. The dataset "
                        "README warns that recomputing polygon areas with VTK can drift "
                        "for non-planar quads, so they must not be regenerated."
                    ),
                    "verification": "confirmed_from_public_file",
                },
                {
                    "id": "windsorml_volume_native_cells",
                    "public_file": "run_<case>/volume_<case>.vtu",
                    "format": "VTK UnstructuredGrid",
                    "domain": "three_dimensional_flow_domain",
                    "association": "CellData",
                    "entities": "all_native_volume_cells",
                    "arrays": [
                        "velocityxavg", "velocityyavg", "velocityzavg", "pressureavg",
                    ],
                    "available_cell_arrays": [
                        "velocityxavg", "velocityyavg", "velocityzavg",
                        "reynoldsstressxx", "reynoldsstressxy", "reynoldsstressxz",
                        "reynoldsstressyy", "reynoldsstressyz", "reynoldsstresszz",
                        "pressureavg",
                    ],
                    "physical_weight": "equal_native_entity",
                    "physical_weight_note": (
                        "v1 scores the volume with equal-cell weighting. WindsorML "
                        "publishes no cell-volume sidecar and generating one for "
                        "~291M cells x 350 cases costs roughly 400 GB and 1,000 "
                        "CPU-hours, so the cell-volume-weighted variants are deferred."
                    ),
                    "verification": "confirmed_from_public_file",
                },
            ],
            "force_convention": {
                "reference_area_m2": 0.112,
                "reference_area_source": "constant_reference_area_force_mom_csv",
                "axes": {"drag": "+x", "lift": "+y", "side": "+z"},
                "axes_note": (
                    "The vertical axis is +y. The body sits on a ground plane at y=0 "
                    "spanning y in [0, 0.343] and is laterally symmetric about z=0. "
                    "Mapping lift to +z collapses cl from about 0.49 to about -0.01."
                ),
                "truth_source": "native_field_integration",
                "audit_reference": "published_force_mom_csv",
                "audit_note": (
                    "Truth and prediction are integrated identically so a perfect "
                    "prediction scores R^2 = 1; the published CSV bounds the truth "
                    "integration rather than defining it."
                ),
            },
            "relative_l2_policy": {
                "surface_or_curve_primary": "physical_measure_weighted",
                "surface_or_curve_secondary": "equal_native_entity",
                "flow_domain_primary": "equal_native_entity",
                "point_or_vertex_physical_weights": "published_barycentric_dual_area",
                "vector_rule": "one_entity_weight_multiplies_the_squared_vector_magnitude",
                "vector_assembly_note": (
                    "Vector components are published as independent scalar arrays. "
                    "Relative L2 recombines exactly from per-component squared sums; "
                    "Euclidean MAE and RMSE do not, and are reported per component."
                ),
                "chunk_rule": "sum_numerators_and_denominators_across_chunks_then_take_one_square_root",
            },
            "scored_case_set": {
                "status": "owner_approved",
                "approved_on": "2026-09-17",
                "rule": "official_split_manifest_intersected_with_published_runs",
                "official_unique_test_cases": 235,
                "scored_unique_test_cases": 233,
                "excluded_case_ids": ["run_352", "run_354"],
                "excluded_reason": (
                    "The official manifest assigns 355 identifiers, but run_350 "
                    "through run_354 have no per-run payload in the public release. "
                    "run_352 (high_drag) and run_354 (baseline and low_drag test "
                    "sets) are therefore unscoreable."
                ),
                "comparability_note": (
                    "Scores on these case sets are not comparable to any evaluation "
                    "run against the unreduced official manifest."
                ),
            },
            "owner_decisions_required": [
                "approve_component_weights_after_profiles_are_scored",
            ],
        },
        "evaluation_reference_version": EVALUATION_REFERENCE_VERSION,
        "default_field_reduction": "per_geometry_then_macro_average",
        "ranking": {
            "metric_id": "overall_score",
            "direction": "higher",
            "decimal_places": 1,
            "rounding": "decimal_half_up",
            "method": "competition",
        },
        "overall_score_composite": {
            "metric_id": "overall_score",
            "operation": "weighted_component_scores",
            "components": components,
            "tolerance": 1e-06,
        },
        "component_score_groups": {
            "operation": "normalized_weighted_component_scores",
            "groups": [
                {
                    "metric_id": "field_score",
                    "component_metric_ids": [
                        "surface_pressure_rel_l2",
                        "surface_wall_shear_rel_l2",
                        "volume_velocity_rel_l2",
                        "volume_pressure_rel_l2",
                    ],
                },
                {
                    "metric_id": "force_score",
                    "component_metric_ids": ["cd_r2", "cl_r2"],
                },
            ],
            "tolerance": 1e-06,
        },
        "metrics": [
            {
                "id": "overall_score",
                "unit": "",
                "direction": "higher",
                "kind": "score",
                "equation": "\\sum_k \\alpha_k S_k,\\quad \\sum_k \\alpha_k=1",
                "aggregation": "derived_score_equation",
                "weighting": "dataset_declared_component_weights",
            },
            {
                "id": "field_score",
                "unit": "",
                "direction": "higher",
                "kind": "score",
                "equation": "\\frac{\\sum_{j\\in F}\\alpha_j S_j}{\\sum_{j\\in F}\\alpha_j}",
                "aggregation": "derived_score_equation",
                "weighting": "dataset_declared_component_weights",
            },
            {
                "id": "force_score",
                "unit": "",
                "direction": "higher",
                "kind": "score",
                "equation": "\\frac{\\sum_{j\\in C}\\alpha_j S_j}{\\sum_{j\\in C}\\alpha_j}",
                "aggregation": "derived_score_equation",
                "weighting": "dataset_declared_component_weights",
            },
            rel_l2("surface_pressure_rel_l2", "surface_point_dual_area"),
            rel_l2("surface_pressure_equal_entity_rel_l2", "surface_entities_equal"),
            rel_l2("surface_wall_shear_rel_l2", "surface_point_dual_area"),
            rel_l2("surface_wall_shear_equal_entity_rel_l2", "surface_entities_equal"),
            rel_l2("volume_velocity_rel_l2", "volume_cells_equal"),
            rel_l2("volume_pressure_rel_l2", "volume_cells_equal"),
            r2("cd_r2"),
            r2("cl_r2"),
            {
                "id": "c_drag_mae",
                "unit": "",
                "direction": "lower",
                "kind": "error",
                "equation": "\\frac{1}{N}\\sum_i\\lvert\\hat{y}_i-y_i\\rvert",
                "aggregation": "all_test_cases",
                "weighting": "cases_equal",
            },
            {
                "id": "c_lift_mae",
                "unit": "",
                "direction": "lower",
                "kind": "error",
                "equation": "\\frac{1}{N}\\sum_i\\lvert\\hat{y}_i-y_i\\rvert",
                "aggregation": "all_test_cases",
                "weighting": "cases_equal",
            },
        ],
        "profile_panels": [
            {
                "id": "pressure_profiles",
                "required": True,
                "scored_in_this_version": False,
                "status": "stations_frozen_pending_per_case_support",
                "definition_file": "profile-definition-v1.json",
                "allow_unlisted_stations": False,
                "minimum_points": 2,
                "sample_count": PROFILE_SAMPLE_COUNT,
                "coordinate_order": "strictly_increasing",
                "station_ids": [
                    "surface_symmetry_centreline",
                    "surface_horizontal_cut_y_0p194",
                ],
                "quantity_ids": ["cp"],
                "source": "evaluator_derived_from_the_submitted_native_surface_field",
                "note": (
                    "Stations are aligned with AutoCFD Case 1, the same 1/4-scale "
                    "Windsor body. Scoring activates once the per-case profile "
                    "support is generated and hash-pinned."
                ),
            },
            {
                "id": "velocity_profiles",
                "required": True,
                "scored_in_this_version": False,
                "status": "stations_frozen_pending_per_case_support",
                "definition_file": "profile-definition-v1.json",
                "allow_unlisted_stations": False,
                "minimum_points": 2,
                "sample_count": PROFILE_SAMPLE_COUNT,
                "coordinate_order": "strictly_increasing",
                "station_ids": [
                    "wake_vertical_x_0p05l",
                    "wake_vertical_x_0p10l",
                    "wake_vertical_x_0p25l",
                    "wake_vertical_x_0p50l",
                    "wake_lateral_x_0p10l_y_0p194",
                ],
                "quantity_ids": ["ux_over_uinf"],
                "source": "evaluator_derived_from_the_submitted_native_volume_field",
                "note": (
                    "The prototype's four fabricated stations were removed. These five "
                    "are sited from the measured wake of run_0: two inside the "
                    "recirculation (Ux reaches -11.5 and -7.8 m/s), one just past "
                    "closure, one in the recovering wake, plus a spanwise cut."
                ),
            },
        ],
        "splits": splits,
    }

    args.out.write_text(json.dumps(document, indent=2) + "\n")
    weight_sum = sum(c["weight"] for c in components)
    print(f"wrote {args.out}")
    print(f"  splits: {len(splits)}  scored components: {len(components)}")
    print(f"  component weight sum: {weight_sum!r}")
    print(f"  total scored test cases: {sum(s['case_count'] for s in splits)} (with overlap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
