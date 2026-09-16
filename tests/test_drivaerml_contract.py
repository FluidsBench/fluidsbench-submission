from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from scripts.validate_submission import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "benchmark-specs" / "drivaerml"
REMOVED_PHYSICAL_VOLUME_METRIC_IDS = {
    "volume_velocity_physical_rel_l2",
    "volume_pressure_physical_rel_l2",
    "drivaerml_volume_velocity_physical_mae",
    "drivaerml_volume_velocity_physical_rmse",
    "drivaerml_volume_pressure_physical_mae",
    "drivaerml_volume_pressure_physical_rmse",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DrivAerMLContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specification = load_json(DATASET_ROOT / "submission-spec.json")
        cls.profile = load_json(
            DATASET_ROOT / cls.specification["profile_definition"]["file"]
        )

    def test_candidate_is_concrete_but_closed(self) -> None:
        self.assertEqual(
            self.specification["dataset_version"],
            "drivaerml-native-v3-candidate",
        )
        support = self.specification["scoring_support"]
        self.assertEqual(support["status"], "owner_review_required")
        self.assertFalse(support["submissions_open"])
        scientific_approval = support["scientific_approval"]
        self.assertEqual(scientific_approval["status"], "approved")
        self.assertEqual(scientific_approval["approved_by"], "neilashton")
        approval_binding = scientific_approval["approval_record"]
        approval_path = DATASET_ROOT / approval_binding["file"]
        self.assertTrue(approval_path.is_file())
        self.assertEqual(sha256_file(approval_path), approval_binding["sha256"])
        approval = load_json(approval_path)
        self.assertEqual(approval["status"], "approved")
        self.assertFalse(approval["non_activation"]["submissions_opened"])
        self.assertFalse(
            approval["non_activation"]["genuine_three_model_sensitivity_complete"]
        )
        self.assertTrue(support["owner_decisions_required"])
        self.assertFalse(
            support["coverage_contract"]["full_prediction_artifact_required"]
        )
        self.assertTrue(
            all(panel["required"] for panel in self.specification["profile_panels"])
        )
        participant_process = " ".join(support["participant_process"])
        self.assertIn("participant-authored scalar metrics", participant_process)
        self.assertIn(
            "complete velocity-profile and continuous-Cp-cut JSON chunks",
            participant_process,
        )
        self.assertIn("methodology disclosure", participant_process)
        self.assertIn("checkpoint publication remains optional", participant_process)
        self.assertIn("optional audits", participant_process)
        self.assertNotIn("complete-split scored-prediction artifact", participant_process)
        self.assertNotIn("recomputation receipt", support["closed_reason"])
        self.assertIn(
            "schema_v3_participant_result_binding", support["activation_gates"]
        )
        self.assertNotIn(
            "schema_v3_nonspatial_result_binding", support["activation_gates"]
        )
        self.assertEqual(
            support["source_release"]["revision"],
            "7a5c0948ce27be709b1116a3a190f806e7a8f79f",
        )
        self.assertEqual(
            self.specification["evaluation_reference_version"],
            "drivaerml-evaluator-v3-candidate",
        )
        self.assertEqual(
            support["dataset_evaluator_binding"]["evaluator_reference_version"],
            "drivaerml-evaluator-v3-candidate",
        )
        self.assertEqual(
            support["activation_gates"]["volume_field_weighting"],
            "complete_equal_native_cell_no_geometric_cell_volume_weights_required",
        )
        self.assertEqual(
            support["activation_gates"]["surface_area_weights"],
            "complete_public_all_484_pinned_order_count_hash_and_value_audit_passed",
        )
        self.assertFalse(
            any(
                "volume" in decision and "weight" in decision
                for decision in support["owner_decisions_required"]
            )
        )

    def test_candidate_evidence_manifest_is_hash_complete_and_nonactivating(self) -> None:
        support = self.specification["scoring_support"]
        self.assertEqual(
            support["candidate_evidence_manifest_file"], "evidence/manifest.json"
        )
        manifest = load_json(DATASET_ROOT / support["candidate_evidence_manifest_file"])
        self.assertEqual(
            manifest["status"], "candidate_evidence_not_scoring_support"
        )
        self.assertFalse(manifest["official_submission_scoring_enabled"])
        self.assertTrue(manifest["owner_scientific_approval"])
        approval_record = manifest["scientific_approval_record"]
        approval_path = DATASET_ROOT / "evidence" / approval_record["file"]
        self.assertEqual(sha256_file(approval_path), approval_record["sha256"])
        files = [item["file"] for item in manifest["artifacts"]]
        self.assertEqual(len(files), len(set(files)))
        for artifact in manifest["artifacts"]:
            with self.subTest(file=artifact["file"]):
                path = DATASET_ROOT / "evidence" / artifact["file"]
                self.assertTrue(path.is_file())
                self.assertEqual(artifact["sha256"], sha256_file(path))
                self.assertFalse(artifact["public_scoring_support_eligible"])

    def test_runtime_and_candidate_evidence_status_docs_are_truthful(self) -> None:
        guide = (DATASET_ROOT / "PARTICIPANT_GUIDE.md").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements-drivaerml-evaluator.txt").read_text(
            encoding="utf-8"
        )
        scientific_contract = (
            DATASET_ROOT / "proposal" / "SCIENTIFIC_CONTRACT.md"
        ).read_text(encoding="utf-8")

        for text in (guide, requirements):
            self.assertIn("Velocity-assignment generation rejects", text)
            self.assertIn("Force replay records", text)
            self.assertNotIn("generators reject every other", text)
        self.assertIn("completed candidate numerical evidence", scientific_contract)
        self.assertIn("completed candidate all-484 native-field", scientific_contract)
        self.assertNotIn(
            "pending all-case native-field replay", scientific_contract
        )
        self.assertNotIn(
            "chunk invariance is an activation claim", scientific_contract
        )

    def test_official_split_indexes_are_exact_owner_manifest_partitions(self) -> None:
        owner = load_json(DATASET_ROOT / "proposal" / "owner-published-splits.json")
        all_public_cases = set(owner["full_train"] + owner["full_val"] + owner["full_test"])
        self.assertEqual(len(all_public_cases), 484)
        for binding in self.specification["splits"]:
            split_id = binding["id"]
            split_path = DATASET_ROOT / binding["index_file"]
            split = load_json(split_path)
            with self.subTest(split_id=split_id):
                self.assertEqual(binding["sha256"], sha256_file(split_path))
                self.assertEqual(split["case_id_status"], "official")
                self.assertEqual(split["train_case_ids"], owner[f"{split_id}_train"])
                self.assertEqual(split["validation_case_ids"], owner[f"{split_id}_val"])
                self.assertEqual(split["case_ids"], owner[f"{split_id}_test"])
                combined = (
                    split["train_case_ids"]
                    + split["validation_case_ids"]
                    + split["case_ids"]
                )
                self.assertEqual(len(combined), len(set(combined)))
                self.assertTrue(set(combined).issubset(all_public_cases))
                if split_id not in {"medium", "scarce", "super_scarce"}:
                    self.assertEqual(set(combined), all_public_cases)
        for split_id in ("medium", "scarce", "super_scarce"):
            self.assertEqual(owner[f"{split_id}_val"], owner["full_val"])
            self.assertEqual(owner[f"{split_id}_test"], owner["full_test"])
            self.assertTrue(set(owner[f"{split_id}_train"]).issubset(owner["full_train"]))
        self.assertTrue(set(owner["super_scarce_train"]).issubset(owner["scarce_train"]))
        self.assertTrue(set(owner["scarce_train"]).issubset(owner["medium_train"]))

    def test_native_cell_support_uses_surface_areas_and_equal_volume_cells(self) -> None:
        supports = {
            item["id"]: item
            for item in self.specification["scoring_support"]["public_supports"]
        }
        surface = supports["surface_native_cells"]
        self.assertEqual(surface["association"], "CellData")
        self.assertEqual(surface["arrays"], ["pMeanTrim", "wallShearStressMeanTrim"])
        self.assertEqual(
            surface["physical_weight_manifest"]["sha256"],
            "1401c7e80bd86f3aa2d640289db9b088ce1e0825327e18eeb1ab2852de04323e",
        )
        area_release = surface["physical_weight_release"]
        self.assertEqual(area_release["status"], "public_complete")
        self.assertEqual(area_release["case_count"], 484)
        self.assertEqual(area_release["payload_file_count"], 484)
        self.assertEqual(area_release["native_polygon_count"], 4_159_517_910)
        self.assertEqual(area_release["payload_bytes"], 16_638_133_592)
        volume = supports["volume_native_cells"]
        self.assertEqual(volume["association"], "CellData")
        self.assertEqual(volume["arrays"], ["UMeanTrim", "pMeanTrim"])
        self.assertIn("optional .02.part", volume["transport_parts"])
        self.assertEqual(volume["weighting"], "one_per_native_cell")
        self.assertFalse(volume["geometric_cell_volume_weights_required"])
        self.assertNotIn("primary_weighting", volume)
        self.assertNotIn("secondary_weighting", volume)
        all_case_evidence = volume["candidate_primary_validation_evidence"]
        self.assertEqual(
            all_case_evidence["file"],
            "evidence/native-volume-equal-cell-primary-all484.json",
        )
        self.assertTrue(all_case_evidence["equal_native_cell_weighting_exercised"])
        self.assertTrue(all_case_evidence["complete_all_484_cases"])
        self.assertEqual(
            all_case_evidence["artifact_schema"],
            "drivaerml-native-volume-equal-cell-primary-all-case-audit-v1",
        )
        self.assertTrue(all_case_evidence["legacy_unit_weight_vocabulary"])
        self.assertEqual(
            all_case_evidence["current_tool_aggregate_schema"],
            "drivaerml-native-volume-equal-cell-all-case-audit-v2",
        )
        self.assertIn("explicit_normalization", all_case_evidence["normalization_status"])

        metrics = {item["id"]: item for item in self.specification["metrics"]}
        self.assertEqual(len(metrics), 36)
        self.assertTrue(REMOVED_PHYSICAL_VOLUME_METRIC_IDS.isdisjoint(metrics))
        self.assertEqual(metrics["surface_pressure_rel_l2"]["weighting"], "surface_face_area")
        self.assertEqual(
            metrics["surface_pressure_equal_entity_rel_l2"]["weighting"],
            "surface_entities_equal",
        )
        self.assertEqual(metrics["volume_velocity_rel_l2"]["weighting"], "volume_cells_equal")
        self.assertEqual(
            metrics["drivaerml_volume_velocity_equal_entity_rmse"]["weighting"],
            "volume_cells_equal",
        )

    def test_force_contract_uses_constant_reference_and_field_integration(self) -> None:
        support = self.specification["scoring_support"]
        force_truth = next(
            item
            for item in support["public_supports"]
            if item["id"] == "field_integrated_force_truth"
        )
        self.assertEqual(force_truth["authoritative_public_file"], "force_mom_constref_all.csv")
        self.assertEqual(
            force_truth["per_case_mirror"],
            "<case_id>/force_mom_constref_<run_number>.csv",
        )
        statistics = force_truth["prototype_r2_truth_statistics"]
        statistics_path = DATASET_ROOT / statistics["file"]
        self.assertEqual(statistics["sha256"], sha256_file(statistics_path))
        self.assertEqual(
            load_json(statistics_path)["source"]["sha256"], force_truth["sha256"]
        )
        force = support["force_integration"]
        self.assertEqual(force["reference_area_m2"], 2.17)
        self.assertEqual(force["reference_length_m"], 2.78618)
        self.assertEqual(force["centre_of_rotation_m"], [1.40009, 0.0, -0.3176])
        self.assertEqual(
            force["ranked_reduction"],
            "separate_equal_case_R2_for_Cd_Cl_and_CmPitch_with_RMSE_reported_as_diagnostics",
        )
        self.assertEqual(force["dependent_axle_loads"]["composite_weight"], 0.0)

    def test_composite_uses_fixed_field_caps_and_bounded_r2(self) -> None:
        composite = self.specification["overall_score_composite"]
        self.assertEqual(composite["status"], "active")
        self.assertEqual(composite["score_range"], [0.0, 100.0])
        self.assertEqual(len(composite["components"]), 9)
        self.assertTrue(
            math.isclose(
                sum(item["weight"] for item in composite["components"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(
            [item["transform"] for item in composite["components"]],
            ["bounded_error"] * 4 + ["bounded_quality"] * 5,
        )
        self.assertEqual(
            [item["cap"] for item in composite["components"][:4]],
            [15.0, 20.0, 12.0, 15.0],
        )
        self.assertTrue(
            all("cap" not in item for item in composite["components"][4:])
        )
        self.assertTrue(all("baseline_error" not in item for item in composite["components"]))
        self.assertEqual(
            [item["weight"] for item in composite["components"]],
            [0.15, 0.10, 0.15, 0.10, 0.15, 0.05, 0.05, 0.15, 0.10],
        )

        proposal_binding = self.specification["scoring_support"]["source_release"][
            "proposal_contract"
        ]
        proposal_path = DATASET_ROOT / proposal_binding["file"]
        self.assertEqual(proposal_binding["sha256"], sha256_file(proposal_path))
        proposal = load_json(proposal_path)
        proposal_composite = proposal["deferred_contracts"]["overall_composite"]
        self.assertNotIn("physics_null_predictions", proposal_composite)
        self.assertEqual(
            [
                proposal_composite["component_transforms"][metric]["cap_percent"]
                for metric in (
                    "native_surface_pressure_field",
                    "native_surface_wall_shear_field",
                    "native_volume_velocity_field",
                    "native_volume_pressure_field",
                )
            ],
            [15.0, 20.0, 12.0, 15.0],
        )
        self.assertTrue(
            all(
                proposal_composite["component_transforms"][metric]["kind"]
                == "bounded_R2"
                for metric in (
                    "field_integrated_Cd",
                    "field_integrated_Cl",
                    "field_integrated_CmPitch",
                    "autocfd_velocity_profiles",
                    "fluidsbench_cp_cuts",
                )
            )
        )

    def test_v9_diagnostic_profile_contract_is_fully_pinned(self) -> None:
        binding = self.specification["profile_definition"]
        profile_path = DATASET_ROOT / binding["file"]
        self.assertEqual(binding["sha256"], sha256_file(profile_path))
        self.assertEqual(self.profile["id"], "drivaerml-diagnostics-v9-candidate")
        self.assertNotIn("pressure_profiles", self.profile)
        pressure = self.profile["pressure_cuts"]
        self.assertEqual(pressure["definition_authority"], "FluidsBench")
        self.assertEqual(pressure["cut_count"], 4)
        self.assertEqual(pressure["ranked_metric_id"], "cp_cut_r2")
        self.assertEqual(pressure["report_only_metric_id"], "cp_cut_rmse")
        self.assertEqual(
            pressure["reduction"],
            "equal_case_equal_cut_global_R2_with_normalized_native_intersection_segment_length_support_per_cut",
        )
        self.assertEqual(
            [station["id"] for station in pressure["stations"]],
            [
                "upperbody_centerline",
                "underbody_centerline",
                "sidewall_z_0_15",
                "front_left_wheelhouse_y_neg_0_6",
            ],
        )
        velocity = self.profile["velocity_profiles"]
        self.assertEqual(velocity["definition_authority"], "AutoCFD5")
        self.assertEqual(velocity["ranked_metric_id"], "velocity_profile_r2")
        self.assertEqual(
            velocity["report_only_metric_id"], "velocity_profile_uinf_rmse"
        )
        self.assertEqual(velocity["line_count"], 16)
        self.assertEqual(velocity["sample_count_per_case"], 3756)
        self.assertEqual(
            [station["source_profile_id"] for station in velocity["stations"]],
            [
                "V1", "V2", "V3", "V4", "V5", "V6",
                "U1", "U2", "U3", "U4", "U5", "U6",
                "L1", "R1", "R2", "R3",
            ],
        )

    def test_velocity_validity_policy_is_owner_bound_and_fail_closed(self) -> None:
        support_binding = self.specification["scoring_support"][
            "profile_definition"
        ]
        policy = support_binding["velocity_validity_policy"]
        self.assertEqual(
            policy["id"], "drivaerml-autocfd5-velocity-validity-v1"
        )
        self.assertEqual(
            self.specification["profile_definition"][
                "velocity_validity_policy_id"
            ],
            policy["id"],
        )
        self.assertEqual(policy["authority"], "dataset_owner")
        self.assertFalse(policy["participant_may_modify"])
        self.assertEqual(
            policy["mask_binding"],
            {
                "status": "pending_immutable_owner_release_no_exclusions_approved",
                "file": None,
                "sha256": None,
            },
        )
        self.assertEqual(
            policy["allowed_owner_exclusion_reasons"],
            ["inside_morphed_solid", "outside_released_fluid_domain"],
        )
        scope = policy["mask_scope"]
        self.assertEqual(scope["master_row_count"], 484 * 37416)
        self.assertEqual(scope["ranked_row_count"], 484 * 3756)
        self.assertEqual(
            scope["study_grid_derivation"],
            {"1_mm": 1, "2_mm": 2, "5_mm": 5, "10_mm": 10},
        )
        self.assertIn("gap_bridging", policy["forbidden_operations"])
        self.assertIn("participant_defined_mask", policy["forbidden_operations"])
        polyhedron = policy["polyhedron_classification"]
        self.assertEqual(polyhedron["absolute_tolerance_steradian"], 1.0e-3)
        self.assertEqual(polyhedron["spatial_boundary_tolerance_m"], 1.0e-6)
        self.assertIn("pending", polyhedron["status"])

        proposal = load_json(DATASET_ROOT / "proposal" / "contract-proposal.json")
        velocity = proposal["deferred_contracts"]["profiles_and_cp_cuts"][
            "velocity_profiles"
        ]
        self.assertEqual(velocity["validity_policy_id"], policy["id"])
        self.assertEqual(
            velocity["owner_mask_status"], policy["mask_binding"]["status"]
        )
        self.assertIn(
            "unavailable", velocity["unresolved_mapping_failure_rule"]
        )
        metrics = {row["id"]: row for row in self.specification["metrics"]}
        self.assertIn(
            "zero_unresolved_nonexcluded_samples",
            metrics["velocity_profile_uinf_rmse"]["availability"],
        )
        self.assertIn(
            "zero_unresolved_nonexcluded_samples",
            metrics["velocity_profile_experimental_subset_uinf_rmse"][
                "availability"
            ],
        )

    def test_discrete_cp_probe_evidence_is_inactive_research_only(self) -> None:
        profile_support = self.specification["scoring_support"]["profile_definition"]
        self.assertNotIn("candidate_cp_validation_evidence", profile_support)
        manifest = load_json(DATASET_ROOT / "evidence" / "manifest.json")
        cp_artifacts = [
            artifact
            for artifact in manifest["artifacts"]
            if artifact["file"].startswith("cp-")
        ]
        self.assertTrue(cp_artifacts)
        for artifact in cp_artifacts:
            self.assertIn("inactive_discrete_cp_probe", artifact["role"])
            self.assertIn("no_submission_or_cp_cut_support_role", artifact["role"])
            self.assertFalse(artifact["public_scoring_support_eligible"])

    def test_leaderboard_and_prototypes_use_the_candidate_contract(self) -> None:
        manifest = load_json(ROOT / "leaderboard" / "manifest.json")
        dataset = next(item for item in manifest["datasets"] if item["slug"] == "drivaerml")
        self.assertEqual(dataset["scoring_support"], self.specification["scoring_support"])
        self.assertEqual(dataset["overall_score_composite"], self.specification["overall_score_composite"])
        self.assertEqual(
            [len(panel["stations"]) for panel in dataset["diagnostic_panels"]],
            [4, 16],
        )
        expected_metrics = {item["id"] for item in self.specification["metrics"]}
        self.assertEqual(len(expected_metrics), 36)
        self.assertTrue(
            REMOVED_PHYSICAL_VOLUME_METRIC_IDS.isdisjoint(expected_metrics)
        )
        self.assertEqual(dataset["submission_format"], "drivaerml_native_candidate_v3")
        diagnostic_group = next(
            group
            for group in dataset["component_score_groups"]["groups"]
            if group["metric_id"] == "diagnostic_score"
        )
        self.assertEqual(
            diagnostic_group["component_metric_ids"],
            ["velocity_profile_r2", "cp_cut_r2"],
        )
        manifest_metric_ids = {
            definition["id"] for definition in manifest["metric_definitions"]
        }
        self.assertTrue(
            {
                "drivaerml_volume_velocity_physical_mae",
                "drivaerml_volume_velocity_physical_rmse",
                "drivaerml_volume_pressure_physical_mae",
                "drivaerml_volume_pressure_physical_rmse",
            }.isdisjoint(manifest_metric_ids)
        )
        self.assertIn("volume_velocity_physical_rel_l2", manifest_metric_ids)
        self.assertIn("volume_pressure_physical_rel_l2", manifest_metric_ids)
        for submission_path in sorted((ROOT / "submissions" / "drivaerml").glob("*/submission.json")):
            submission = load_json(submission_path)
            with self.subTest(submission_id=submission["submission_id"]):
                self.assertEqual(submission["dataset_version"], self.specification["dataset_version"])
                self.assertEqual(set(submission["metric_values"]), expected_metrics)
                self.assertEqual(submission["approval"]["status"], "prototype")
                self.assertGreater(submission["metric_values"]["overall_score"], 0.0)
                self.assertLessEqual(submission["metric_values"]["overall_score"], 100.0)

    def test_leaderboard_catalog_and_fixture_population_are_drivaer_specific(self) -> None:
        manifest = load_json(ROOT / "leaderboard" / "manifest.json")
        dataset = next(
            item for item in manifest["datasets"] if item["slug"] == "drivaerml"
        )
        dimensional = {
            item["id"]: item
            for item in manifest["metric_catalog"]["dimensional_fields"]
        }
        coefficients = {
            item["id"]: item
            for item in manifest["metric_catalog"]["coefficient_errors"]
        }

        # Generic catalog entries remain unchanged for the other datasets.
        self.assertEqual(dimensional["surface_pressure"]["unit"], "Pa")
        self.assertEqual(dimensional["surface_wall_shear"]["unit"], "Pa")
        self.assertEqual(dimensional["volume_pressure"]["unit"], "Pa")
        self.assertEqual(coefficients["c_drag"]["statistic"], "mae")
        self.assertEqual(coefficients["c_lift"]["statistic"], "mae")
        self.assertEqual(coefficients["c_pitch"]["statistic"], "mae")

        expected_dimensional = [
            "drivaerml_surface_pmeantrim_native_area",
            "drivaerml_surface_wallshearstressmeantrim_native_area",
            "drivaerml_volume_umeantrim_equal_cell",
            "drivaerml_volume_pmeantrim_equal_cell",
        ]
        expected_coefficients = [
            "drivaerml_cd_equal_case_rmse",
            "drivaerml_cl_equal_case_rmse",
            "drivaerml_cmpitch_equal_case_rmse",
        ]
        self.assertEqual(
            dataset["metrics"]["dimensional_fields"], expected_dimensional
        )
        self.assertEqual(
            dataset["metrics"]["coefficient_errors"], expected_coefficients
        )
        for metric_id in expected_dimensional[:2]:
            self.assertEqual(dimensional[metric_id]["unit"], "m^2/s^2")
            self.assertEqual(
                dimensional[metric_id]["weighting"],
                "native_surface_polygon_area",
            )
        self.assertEqual(
            dimensional["drivaerml_volume_umeantrim_equal_cell"]["unit"],
            "m/s",
        )
        self.assertEqual(
            dimensional["drivaerml_volume_pmeantrim_equal_cell"]["unit"],
            "m^2/s^2",
        )
        for metric_id in expected_dimensional[2:]:
            self.assertEqual(
                dimensional[metric_id]["weighting"],
                "equal_native_volume_cell",
            )
        for metric_id in expected_coefficients:
            self.assertEqual(coefficients[metric_id]["statistic"], "rmse")
            self.assertEqual(
                coefficients[metric_id]["aggregation"], "all_test_cases"
            )
            self.assertEqual(coefficients[metric_id]["weighting"], "cases_equal")

        selected_ids = set(expected_dimensional + expected_coefficients)
        for other in manifest["datasets"]:
            if other["slug"] == "drivaerml":
                continue
            other_ids = {
                metric_id
                for group in other.get("metrics", {}).values()
                for metric_id in group
            }
            self.assertTrue(selected_ids.isdisjoint(other_ids), other["slug"])

        fixture_paths = sorted(
            (ROOT / "submissions" / "drivaerml").glob("*/submission.json")
        )
        fixtures = [load_json(path) for path in fixture_paths]
        population = dataset["submission_population"]
        self.assertEqual(dataset["submission_count"], len(fixtures))
        self.assertEqual(population["feed_row_count"], len(fixtures))
        self.assertEqual(
            population["prototype_ineligible_fixture_count"], len(fixtures)
        )
        self.assertEqual(population["eligible_submission_count"], 0)
        self.assertEqual(population["scientific_result_count"], 0)
        self.assertEqual(
            population["submission_count_semantics"],
            "all_current_feed_rows_are_structural_prototype_fixtures_not_eligible_submissions",
        )
        self.assertTrue(
            all(item["approval"]["status"] == "prototype" for item in fixtures)
        )
        self.assertEqual(
            dataset["updated_at"], max(item["submitted_at"] for item in fixtures)
        )
        self.assertEqual(
            population["updated_at_semantics"],
            "latest_submitted_at_date_among_prototype_fixture_rows_not_a_contract_or_scientific_evidence_update",
        )


if __name__ == "__main__":
    unittest.main()
