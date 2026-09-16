from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from reference.drivaerml.dataset_scorer import _relative_series_support_index


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "benchmark-specs" / "drivaerml" / "evidence"
CP_FIXTURE = EVIDENCE_ROOT / "transolver-run419-relative-cp-evaluator-smoke-v1.json"
VELOCITY_FIXTURE = (
    EVIDENCE_ROOT / "transolver-run419-relative-velocity-evaluator-smoke-v1.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DrivAerMLRelativeEvaluatorFixtureTests(unittest.TestCase):
    def test_real_run419_reports_are_hash_bound_and_match_retained_support(self) -> None:
        evidence_manifest = load_json(EVIDENCE_ROOT / "manifest.json")
        evidence = {
            artifact["file"]: artifact
            for artifact in evidence_manifest["artifacts"]
        }
        for fixture in (CP_FIXTURE, VELOCITY_FIXTURE):
            binding = evidence[fixture.name]
            self.assertEqual(sha256_file(fixture), binding["sha256"])
            self.assertEqual(binding["scope"], "run_419_real_checkpoint_single_case_smoke")
            self.assertFalse(binding["public_scoring_support_eligible"])
            self.assertIn("nonactivation", binding["role"])

        support = _relative_series_support_index()

        cp_report = load_json(CP_FIXTURE)
        self.assertEqual(
            cp_report["schema"], "drivaerml-transolver-cp-cut-evaluator-smoke-v1"
        )
        self.assertEqual(cp_report["case_id"], "run_419")
        self.assertEqual(
            cp_report["status"], "complete_candidate_report_only_not_submission"
        )
        self.assertFalse(cp_report["scope_limit"]["official_submission_artifact"])
        self.assertFalse(cp_report["scope_limit"]["ranked_scoring_or_activation"])
        self.assertFalse(
            cp_report["scope_limit"]["immutable_evaluator_revision_gate_complete"]
        )
        self.assertEqual(
            cp_report["relative_report_only_view"]["family_id"],
            "drivaerml_cp_relative_v1",
        )
        self.assertEqual(cp_report["relative_report_only_view"]["composite_weight"], 0.0)

        cp_manifest = (
            ROOT
            / "benchmark-specs"
            / "drivaerml"
            / "support"
            / "relative-v3"
            / "manifests"
            / "cp-native-support-all484-v3.json"
        )
        self.assertEqual(
            cp_report["support_replay"]["relative_all484_aggregate_sha256"],
            sha256_file(cp_manifest),
        )
        self.assertTrue(cp_report["support_replay"]["relative_alias_bindings_replayed"])

        relative_cp = cp_report["relative_report_only_view"]
        observed_cp = {
            item["station_id"]: (
                "shared_alias",
                item["canonical_cut_support_sha256"],
            )
            for item in relative_cp["aliases"]
        }
        observed_cp.update(
            {
                item["station_id"]: (
                    "materialized",
                    item["support_identity_sha256"],
                )
                for item in relative_cp["moving_cuts"]
            }
        )
        expected_cp = {
            station_id: identity
            for (case_id, family_id, station_id), identity in support.items()
            if case_id == "run_419" and family_id == "drivaerml_cp_relative_v1"
        }
        self.assertEqual(set(observed_cp), set(expected_cp))
        for station_id, (representation, support_sha256) in observed_cp.items():
            self.assertEqual(expected_cp[station_id]["representation"], representation)
            self.assertEqual(
                expected_cp[station_id]["support_identity_sha256"], support_sha256
            )

        velocity_report = load_json(VELOCITY_FIXTURE)
        self.assertEqual(
            velocity_report["schema"],
            "drivaerml-transolver-velocity-profile-reference-smoke-v1",
        )
        self.assertEqual(velocity_report["case_id"], "run_419")
        self.assertEqual(
            velocity_report["status"],
            "complete_candidate_report_only_not_contract_metric",
        )
        self.assertFalse(
            velocity_report["scope_limit"]["official_submission_artifact"]
        )
        self.assertFalse(
            velocity_report["scope_limit"]["relative_v3_scoring_support_active"]
        )
        self.assertTrue(
            velocity_report["scope_limit"][
                "all_prediction_chunks_exhausted_and_hash_verified"
            ]
        )

        relative_velocity = next(
            family
            for family in velocity_report["families"]
            if family["family_id"] == "drivaerml-velocity-relative-v3"
        )
        self.assertEqual(relative_velocity["placement_mode"], "relative")
        self.assertEqual(relative_velocity["profile_count"], 16)
        self.assertEqual(
            relative_velocity["support_identity"],
            {
                "invalid_count": 78,
                "mapping_file": "velocity-relative-v3-cell-mapping-10mm.json",
                "mapping_sha256": (
                    "7b61b6a444379cf7edb1cb0e07a7a6b202cf87084b7d55720e7c225cf591ce1e"
                ),
                "receipt_file": "receipt.json",
                "receipt_sha256": (
                    "f7df45705d896236c31c4ab7c13039c001ddd5d0845f599a6225599230729833"
                ),
                "valid_count": 3678,
            },
        )
        expected_velocity_stations = {
            station_id
            for (case_id, family_id, station_id) in support
            if case_id == "run_419"
            and family_id == "drivaerml-velocity-relative-v3"
        }
        profiles = relative_velocity["profiles"]
        self.assertEqual({profile["profile_id"] for profile in profiles}, expected_velocity_stations)
        for profile in profiles:
            # The exact report contains real model/reference values, but predates
            # the submission format's per-station identity fields.  Do not fill
            # those fields from the retained index and mislabel them as evaluator
            # output.
            self.assertNotIn("support_identity_sha256", profile)
            self.assertNotIn("placement_receipt_identity_sha256", profile)
            valid_series = profile["valid_series"]
            self.assertEqual(len(valid_series["distance_m"]), profile["valid_count"])
            self.assertEqual(
                len(valid_series["prediction_velocity_ratio"]), profile["valid_count"]
            )
            self.assertEqual(
                len(valid_series["truth_velocity_ratio"]), profile["valid_count"]
            )


if __name__ == "__main__":
    unittest.main()
