#!/usr/bin/env python3
"""Regenerate the WindsorML submission spec from frozen decisions.

Replaces the prototype spec, whose splits were a single fabricated identifier
(``windsorml_default_test_0001``) and whose status was ``prototype_dummy_data``.

Five decisions are baked in here:

* **Scored case sets are the official manifest intersected with the published
  runs** (233 unique test cases). ``splits/*.json`` record both counts.
* **Forces use the constant reference area, 0.112 m^2**, and are scored by
  integrating truth and prediction identically; the published CSV is the audit
  anchor, not the scoring truth.
* **Volume metrics are equal-cell weighted for v1.** The physical
  (cell-volume-weighted) variants are removed rather than left unimplemented,
  because generating the sidecars costs roughly 400 GB and 1,000 CPU-hours.
* **Profile stations are real and scored.** The prototype's four fabricated
  velocity stations are replaced by stations sited from the measured wake and
  aligned with AutoCFD Case 1; see ``profile-definition-v2.json``.
* **Every diagnostic kind is published in two placement families**, following
  the DrivAerML relative-diagnostics contract: a ``constant`` family at fixed
  absolute coordinates, which carries the weight, and a geometry-following
  ``relative`` family reported at zero weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "benchmark-specs" / "windsorml"

DATASET_VERSION = "windsorml-native-v1-candidate"
EVALUATION_REFERENCE_VERSION = "windsorml-evaluator-v0.1-candidate"

# Frozen after the stratified 64/96/128/160 sweep. See
# profile-definition-v2.json for why higher counts add nothing: sampling already
# sits at the local cell size.
PROFILE_SAMPLE_COUNT = 128
PROFILE_DEFINITION_FILE = "profile-definition-v2.json"
PROFILE_SUPPORT_MANIFEST_SHA256 = (
    "e660311ef82ec19347e91eeb1753c1f0af4c1bd65ba01491fb47ac960da51376"
)

# Field 0.50, force 0.25, diagnostics 0.25 -- the prototype's split, restored in
# full now that the profile diagnostics are computed from real frozen stations
# instead of fabricated placeholders. Only the constant-placement families carry
# weight; the relative families are reported at zero.
SCORED_COMPONENTS = (
    ("surface_pressure_rel_l2", 0.15, "bounded_error", 15.0),
    ("surface_wall_shear_rel_l2", 0.10, "bounded_error", 20.0),
    ("volume_velocity_rel_l2", 0.15, "bounded_error", 12.0),
    ("volume_pressure_rel_l2", 0.10, "bounded_error", 15.0),
    ("cd_r2", 0.15, "bounded_quality", None),
    ("cl_r2", 0.10, "bounded_quality", None),
    ("velocity_profile_r2", 0.15, "bounded_quality", None),
    ("cp_cut_r2", 0.10, "bounded_quality", None),
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


def profile_r2(metric_id: str, family_id: str, *, ranked: bool) -> dict:
    """A profile R2, pooled over every sample of every station in one family."""

    return {
        "id": metric_id,
        "unit": "",
        "direction": "higher",
        "kind": "r2",
        "equation": "1-\\frac{\\sum_i(y_i-\\hat{y}_i)^2}{\\sum_i(y_i-\\bar{y})^2}",
        "aggregation": "flatten_all_required_profile_samples",
        "weighting": "samples_equal",
        "profile_family_id": family_id,
        "scoring_role": "ranked_candidate" if ranked else "report_only",
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
            # validate_scoring_supports.py enforces a closed vocabulary here
            # (official / owner_review_required / prototype / retired). Support
            # completeness is stated in profile_support.status and
            # closed_reason rather than by inventing a new status value.
            "status": "owner_review_required",
            "submissions_open": False,
            "closed_reason": (
                "Every contract input is frozen and every support artefact is "
                "generated and hash-pinned; opening submissions is a release "
                "decision for the dataset owner."
            ),
            "profile_support": {
                "status": "generated_and_hash_pinned",
                "manifest_file": (
                    "profile-support/windsorml-profile-support-v3-manifest.json"
                ),
                "manifest_sha256": PROFILE_SUPPORT_MANIFEST_SHA256,
                "support_schema": "windsorml-profile-support-v3",
                "case_count": 233,
                "note": (
                    "One support file per scored test case, each pinning the native "
                    "entity IDs the profile series are read at. Every case reports "
                    "exactly 8 containment fallbacks: the eight vertical stations "
                    "each begin at y = 0 on the ground plane, outside the fluid mesh."
                ),
            },
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
            "placement_mode_policy": {
                "rule": "at_most_one_placement_mode_per_diagnostic_kind_may_have_nonzero_weight",
                "ranked_mode": "constant",
                "report_only_mode": "relative",
                "definition_file": PROFILE_DEFINITION_FILE,
            },
            "component_weights": {
                "status": "owner_approved",
                "approved_on": "2026-09-17",
                "split": {"field_score": 0.50, "force_score": 0.25, "diagnostic_score": 0.25},
                "rationale": (
                    "Identical to AhmedML and HiLiftAeroML component for component, "
                    "and to DrivAerML apart from its force group, which splits three "
                    "ways (cd 0.15, cl 0.05, c_pitch 0.05) because it scores pitching "
                    "moment. WindsorML publishes cmy, so a c_pitch_r2 term is "
                    "supported by the data, but it would need surface moment "
                    "integration and a pinned moment reference point that the dataset "
                    "does not publish. Not adopted."
                ),
            },
            # The validator requires a non-empty list while the status is
            # owner_review_required, and that is the honest state: both
            # technical decisions are approved and recorded above, so what
            # remains is the release decision itself.
            "owner_decisions_required": ["approve_release_and_open_submissions"],
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
                {
                    "metric_id": "diagnostic_score",
                    "component_metric_ids": ["velocity_profile_r2", "cp_cut_r2"],
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
            {
                "id": "diagnostic_score",
                "unit": "",
                "direction": "higher",
                "kind": "score",
                "equation": "\\frac{\\sum_{j\\in D}\\alpha_j S_j}{\\sum_{j\\in D}\\alpha_j}",
                "aggregation": "derived_score_equation",
                "weighting": "dataset_declared_component_weights",
            },
            r2("cd_r2"),
            r2("cl_r2"),
            profile_r2("velocity_profile_r2", "windsorml_velocity_constant_v1", ranked=True),
            profile_r2("cp_cut_r2", "windsorml_cp_constant_v1", ranked=True),
            profile_r2(
                "velocity_profile_relative_r2",
                "windsorml_velocity_relative_v1",
                ranked=False,
            ),
            profile_r2("cp_cut_relative_r2", "windsorml_cp_relative_v1", ranked=False),
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
                "scored_in_this_version": True,
                "status": "stations_resolution_and_support_frozen",
                "definition_file": PROFILE_DEFINITION_FILE,
                "allow_unlisted_stations": False,
                "minimum_points": 2,
                "sample_count": PROFILE_SAMPLE_COUNT,
                "coordinate_order": "strictly_increasing",
                "families": {
                    "constant": "windsorml_cp_constant_v1",
                    "relative": "windsorml_cp_relative_v1",
                },
                "ranked_family_id": "windsorml_cp_constant_v1",
                "station_ids": [
                    "cp_centreline_upper",
                    "cp_base_vertical",
                    "cp_side_horizontal_y_0p194",
                ],
                "quantity_ids": ["cp"],
                "source": "evaluator_derived_from_the_submitted_native_surface_field",
                "note": (
                    "Stations are aligned with AutoCFD Case 1, the same 1/4-scale "
                    "Windsor body. Published in a constant family (scored) and a "
                    "body-height-relative family (report only, zero weight)."
                ),
            },
            {
                "id": "velocity_profiles",
                "required": True,
                "scored_in_this_version": True,
                "status": "stations_resolution_and_support_frozen",
                "definition_file": PROFILE_DEFINITION_FILE,
                "allow_unlisted_stations": False,
                "minimum_points": 2,
                "sample_count": PROFILE_SAMPLE_COUNT,
                "coordinate_order": "strictly_increasing",
                "families": {
                    "constant": "windsorml_velocity_constant_v1",
                    "relative": "windsorml_velocity_relative_v1",
                },
                "ranked_family_id": "windsorml_velocity_constant_v1",
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
                    "recirculation (ux/uinf reaches -0.251 and -0.154), one just past "
                    "closure, one in the recovering wake, plus a spanwise cut. The "
                    "relative family rescales the vertical coordinate by each case's "
                    "body height, which varies from 0.316 m to 0.473 m."
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
