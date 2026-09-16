from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.validate_scoring_supports import sha256_file, validate_specification


ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "benchmark-specs"
MANIFEST_PATH = ROOT / "leaderboard" / "manifest.json"
REMOVED_DRIVAERML_PHYSICAL_VOLUME_METRIC_IDS = {
    "volume_velocity_physical_rel_l2",
    "volume_pressure_physical_rel_l2",
    "drivaerml_volume_velocity_physical_mae",
    "drivaerml_volume_velocity_physical_rmse",
    "drivaerml_volume_pressure_physical_mae",
    "drivaerml_volume_pressure_physical_rmse",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ScoringSupportSpecGateTests(unittest.TestCase):
    def specifications(self) -> list[dict]:
        return [
            load_json(path)
            for path in sorted(SPEC_ROOT.glob("*/submission-spec.json"))
        ]

    def make_official_synthetic_dataset(
        self,
        temporary_root: Path,
    ) -> tuple[Path, Path, Path]:
        temporary_spec_root = temporary_root / "benchmark-specs"
        dataset_directory = temporary_spec_root / "synthetic"
        dataset_directory.mkdir(parents=True)
        methodology_contract = {
            "$schema": "https://fluidsbench.org/schemas/methodology-contract-v1.schema.json",
            "format": "fluidsbench-methodology-contract-v1",
            "dataset_id": "synthetic",
            "allow_additional_predicted_fields": True,
            "required_predicted_fields": [
                {
                    "field_id": "synthetic-support.pressure",
                    "domain": "domain",
                    "component_count": 1,
                    "description": "Synthetic scalar pressure fixture.",
                }
            ],
        }
        (dataset_directory / "methodology-contract.json").write_text(
            json.dumps(methodology_contract, indent=2) + "\n",
            encoding="utf-8",
        )
        support_directory = (
            dataset_directory / "scoring-support" / "synthetic-support-v1"
        )
        shutil.copytree(
            ROOT / "examples" / "v3-template" / "support",
            support_directory,
        )
        approval = {
            "approved_by": "Synthetic dataset owner",
            "approved_at": "2026-07-27",
            "pull_request_url": "https://github.com/example/benchmark/pull/17",
        }
        manifest_path = support_directory / "manifest.json"
        manifest = load_json(manifest_path)
        manifest["status"] = "official"
        manifest["owner_approval"] = approval
        for support in manifest["supports"]:
            for binding in support["metric_bindings"]:
                binding["dataset_weighting"] = (
                    "entities_equal"
                    if binding["weighting"] == "uniform"
                    else "cell_volume"
                )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        split_directory = dataset_directory / "splits"
        split_directory.mkdir(parents=True)
        split_path = split_directory / "default.json"
        split_index = {
            "schema_version": "1.0",
            "dataset_id": "synthetic",
            "split_id": "default",
            "case_set_id": "standard",
            "split_label": "Default",
            "case_id_status": "official",
            "case_count": 2,
            "case_ids": ["case-001", "case-002"],
        }
        split_path.write_text(
            json.dumps(split_index, indent=2) + "\n",
            encoding="utf-8",
        )
        metrics = [
            {
                "id": binding["metric_id"],
                "aggregation": binding["aggregation"],
                "weighting": binding["dataset_weighting"],
            }
            for support in manifest["supports"]
            for binding in support["metric_bindings"]
        ]
        specification = {
            "schema_version": "1.1",
            "dataset_id": "synthetic",
            "dataset_name": "Synthetic",
            "dataset_version": "synthetic-1",
            "status": "official",
            "evaluation_reference_version": "synthetic-evaluator-v1",
            "metrics": metrics,
            "scoring_support": {
                "status": "official",
                "submissions_open": True,
                "release_id": "synthetic-support-v1",
                "manifest_file": (
                    "scoring-support/synthetic-support-v1/manifest.json"
                ),
                "manifest_url": (
                    "https://example.org/releases/"
                    "synthetic-support-v1/manifest.json"
                ),
                "manifest_sha256": sha256_file(manifest_path),
                "owner_approval": approval,
            },
            "splits": [
                {
                    "id": "default",
                    "label": "Default",
                    "index_file": "splits/default.json",
                    "case_count": 2,
                    "case_set_id": "standard",
                    "case_id_status": "official",
                    "sha256": sha256_file(split_path),
                }
            ],
        }
        specification_path = dataset_directory / "submission-spec.json"
        specification_path.write_text(
            json.dumps(specification, indent=2) + "\n",
            encoding="utf-8",
        )
        return specification_path, manifest_path, temporary_spec_root

    def write_manifest_and_update_pin(
        self,
        specification_path: Path,
        manifest_path: Path,
        manifest: dict,
    ) -> None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        specification = load_json(specification_path)
        specification["scoring_support"]["manifest_sha256"] = sha256_file(
            manifest_path
        )
        specification_path.write_text(
            json.dumps(specification, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_every_dataset_has_a_valid_independent_publication_gate(self) -> None:
        specifications = self.specifications()
        self.assertEqual(len(specifications), 9)
        for specification in specifications:
            dataset_id = specification["dataset_id"]
            with self.subTest(dataset_id=dataset_id):
                support = specification.get("scoring_support")
                self.assertIsInstance(support, dict)
                if support["status"] != "official":
                    self.assertIs(support["submissions_open"], False)
                    self.assertTrue(support["closed_reason"].strip())
                else:
                    self.assertIsInstance(support["submissions_open"], bool)
                    self.assertTrue(support["release_id"].strip())
                    self.assertTrue(support["manifest_file"].strip())
                    self.assertRegex(support["manifest_sha256"], r"^[a-f0-9]{64}$")
                    self.assertTrue(support["owner_approval"]["approved_by"].strip())

    def test_scoring_support_validator_passes_repository_state(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/validate_scoring_supports.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("for 9 datasets", result.stdout)

    def test_one_dataset_can_become_official_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            (
                specification_path,
                _manifest_path,
                temporary_spec_root,
            ) = self.make_official_synthetic_dataset(
                Path(temporary_directory),
            )

            errors = validate_specification(
                specification_path,
                spec_root=temporary_spec_root,
            )
            self.assertEqual(errors, [])

    def test_official_metric_contract_rejects_missing_extra_and_duplicate_bindings(
        self,
    ) -> None:
        for mutation in ("missing", "extra", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                (
                    specification_path,
                    manifest_path,
                    temporary_spec_root,
                ) = self.make_official_synthetic_dataset(Path(temporary))
                manifest = load_json(manifest_path)
                bindings = manifest["supports"][0]["metric_bindings"]
                if mutation == "missing":
                    removed_metric_id = bindings.pop()["metric_id"]
                elif mutation == "extra":
                    extra = copy.deepcopy(bindings[0])
                    extra["metric_id"] = "invented_rel_l1"
                    bindings.append(extra)
                else:
                    bindings.append(copy.deepcopy(bindings[0]))
                self.write_manifest_and_update_pin(
                    specification_path,
                    manifest_path,
                    manifest,
                )

                errors = validate_specification(
                    specification_path,
                    spec_root=temporary_spec_root,
                )
                joined = "\n".join(errors)
                if mutation == "duplicate":
                    self.assertIn("each metric may be bound only once", joined)
                else:
                    self.assertIn(
                        "metric bindings must exactly cover dataset field and "
                        "case-scalar metrics",
                        joined,
                    )
                if mutation == "missing":
                    self.assertIn(f"missing=['{removed_metric_id}']", joined)
                if mutation == "extra":
                    self.assertIn("binds unknown metric_id 'invented_rel_l1'", joined)
                    self.assertIn("unexpected=['invented_rel_l1']", joined)

    def test_official_metric_contract_rejects_binding_rule_mutations(self) -> None:
        mutations = {
            "aggregation": (
                lambda binding: binding.__setitem__(
                    "aggregation",
                    "flatten_all_aligned_field_values",
                ),
                "aggregation must match the dataset specification",
            ),
            "dataset_weighting": (
                lambda binding: binding.__setitem__(
                    "dataset_weighting",
                    "cases_equal",
                ),
                "dataset_weighting must match the dataset specification",
            ),
            "reduction": (
                lambda binding: binding.__setitem__("reduction", "mae"),
                "reduction must be 'relative_l1_percent'",
            ),
            "weighting": (
                lambda binding: binding.__setitem__("weighting", "uniform"),
                "weighting must be 'support_weights'",
            ),
            "case_evidence": (
                lambda binding: binding.__setitem__(
                    "case_evidence",
                    "aggregate_only",
                ),
                "case_evidence must be 'metric_value'",
            ),
        }
        for name, (mutate, expected_error) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                (
                    specification_path,
                    manifest_path,
                    temporary_spec_root,
                ) = self.make_official_synthetic_dataset(Path(temporary))
                manifest = load_json(manifest_path)
                mutate(manifest["supports"][0]["metric_bindings"][0])
                self.write_manifest_and_update_pin(
                    specification_path,
                    manifest_path,
                    manifest,
                )

                errors = validate_specification(
                    specification_path,
                    spec_root=temporary_spec_root,
                )
                self.assertIn(expected_error, "\n".join(errors))

    def test_ahmedml_uses_complete_public_release_entities(self) -> None:
        support = load_json(
            SPEC_ROOT / "ahmedml" / "submission-spec.json"
        )["scoring_support"]
        coverage = support["coverage_contract"]
        self.assertEqual(
            coverage["rule"],
            "complete_original_public_release_entities",
        )
        self.assertTrue(coverage["inference_may_be_chunked"])
        self.assertTrue(coverage["complete_case_and_entity_coverage_required"])
        self.assertFalse(coverage["full_prediction_artifact_required"])
        supports = {item["id"]: item for item in support["public_supports"]}
        self.assertEqual(supports["surface-native"]["association"], "CellData")
        self.assertEqual(
            supports["flow-domain-native"]["entities"],
            "all_native_volume_cells",
        )
        required_decisions = set(support["owner_decisions_required"])
        self.assertIn(
            "pin_exact_public_release_files_revisions_and_hashes",
            required_decisions,
        )
        self.assertIn(
            "publish_authoritative_physical_weights_and_stable_entity_ids",
            required_decisions,
        )

    def test_drivaerml_uses_only_equal_native_volume_cell_weighting(self) -> None:
        specification = load_json(
            SPEC_ROOT / "drivaerml" / "submission-spec.json"
        )
        support = specification["scoring_support"]
        self.assertEqual(support["status"], "owner_review_required")
        self.assertFalse(support["submissions_open"])
        self.assertTrue(support["closed_reason"].strip())
        self.assertEqual(
            specification["evaluation_reference_version"],
            "drivaerml-evaluator-v3-candidate",
        )
        self.assertEqual(
            support["dataset_evaluator_binding"]["evaluator_reference_version"],
            "drivaerml-evaluator-v3-candidate",
        )

        metrics = {item["id"]: item for item in specification["metrics"]}
        self.assertEqual(len(metrics), 36)
        self.assertTrue(
            REMOVED_DRIVAERML_PHYSICAL_VOLUME_METRIC_IDS.isdisjoint(metrics)
        )
        for metric_id in (
            "cd_r2",
            "cl_r2",
            "c_pitch_r2",
            "velocity_profile_r2",
            "cp_cut_r2",
        ):
            self.assertEqual(metrics[metric_id]["kind"], "r2")
        self.assertEqual(
            metrics["surface_pressure_rel_l2"]["weighting"],
            "surface_face_area",
        )
        self.assertEqual(
            metrics["surface_pressure_equal_entity_rel_l2"]["weighting"],
            "surface_entities_equal",
        )
        for metric_id in (
            "volume_velocity_rel_l2",
            "volume_pressure_rel_l2",
            "drivaerml_volume_velocity_equal_entity_mae",
            "drivaerml_volume_velocity_equal_entity_rmse",
            "drivaerml_volume_pressure_equal_entity_mae",
            "drivaerml_volume_pressure_equal_entity_rmse",
        ):
            self.assertEqual(metrics[metric_id]["weighting"], "volume_cells_equal")

        supports = {item["id"]: item for item in support["public_supports"]}
        volume = supports["volume_native_cells"]
        self.assertEqual(volume["weighting"], "one_per_native_cell")
        self.assertFalse(volume["geometric_cell_volume_weights_required"])
        self.assertNotIn("candidate_secondary_weight_status", volume)
        all_case = volume["candidate_primary_validation_evidence"]
        self.assertEqual(all_case["case_count"], 484)
        self.assertTrue(all_case["equal_native_cell_weighting_exercised"])
        self.assertTrue(all_case["complete_all_484_cases"])
        self.assertEqual(
            all_case["artifact_schema"],
            "drivaerml-native-volume-equal-cell-primary-all-case-audit-v1",
        )
        self.assertTrue(all_case["legacy_unit_weight_vocabulary"])
        self.assertEqual(
            all_case["current_tool_aggregate_schema"],
            "drivaerml-native-volume-equal-cell-all-case-audit-v2",
        )

        gates = support["activation_gates"]
        self.assertEqual(
            gates["volume_field_weighting"],
            "complete_equal_native_cell_no_geometric_cell_volume_weights_required",
        )
        self.assertFalse(
            any(
                "volume" in decision and "weight" in decision
                for decision in support["owner_decisions_required"]
            )
        )

    def test_spec_gates_propagate_to_leaderboard_manifest(self) -> None:
        manifest_datasets = {
            dataset["slug"]: dataset
            for dataset in load_json(MANIFEST_PATH)["datasets"]
        }
        specifications = self.specifications()
        self.assertEqual(
            set(manifest_datasets),
            {specification["dataset_id"] for specification in specifications},
        )
        for specification in specifications:
            dataset_id = specification["dataset_id"]
            with self.subTest(dataset_id=dataset_id):
                self.assertEqual(
                    manifest_datasets[dataset_id].get("scoring_support"),
                    specification["scoring_support"],
                )


if __name__ == "__main__":
    unittest.main()
