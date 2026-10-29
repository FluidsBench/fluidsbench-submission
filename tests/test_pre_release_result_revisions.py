"""Revision lifecycle tests use tiny synthetic packages, never scientific results."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from reference.drivaernetplusplus.pre_release import (
    GENUINE_INFERENCE_ATTESTATION,
    REGISTRY_SCHEMA,
    REGISTRY_STATUS,
    build_registry_entry,
)
from scripts import manage_leaderboard as manager
from scripts import validate_submission as validator


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class PreReleaseResultRevisions(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        for module in (validator, manager):
            self.stack.enter_context(patch.object(module, "ROOT", self.root))
        self.manifest = {
            "data_release": {"status": "prototype_dummy_data", "source_ref": "dev"},
            "datasets": [{"name": "DrivAerNet++", "slug": "drivaernetplusplus"}],
        }
        self.first = self.make_package(1)
        self.second = self.make_package(2)
        self.registry_path = self.root / validator.DRIVAERNETPLUSPLUS_PRE_RELEASE_REGISTRY_PATH
        self.registry = {
            "schema": REGISTRY_SCHEMA,
            "status": REGISTRY_STATUS,
            "dataset_id": "drivaernetplusplus",
            "activation": {
                "owner_approval_complete": False,
                "published": False,
                "submissions_opened": False,
                "registration_changes_activation": False,
            },
            "entries": [build_registry_entry(self.first[0], repository_root=self.root)],
        }
        write(self.registry_path, self.registry)

    def make_package(self, version):
        series = "synthetic-revision-test"
        identifier = f"{series}-v{version}"
        directory = self.root / "submissions/drivaernetplusplus" / identifier
        chunk_path = directory / "profiles/chunk-000.json"
        write(chunk_path, {"cases": [{"case_id": "synthetic", "series": [{}, {}, {}]}]})
        index_path = directory / "profiles/index.json"
        write(index_path, {"case_count": 1, "chunks": [{
            "file": chunk_path.name, "sha256": validator.sha256_file(chunk_path),
        }]})
        cases_path = directory / "discretization/cases.jsonl"
        cases_path.parent.mkdir(parents=True)
        cases_path.write_text('{"case_id":"synthetic"}\n')
        spatial_path = directory / "discretization.json"
        write(spatial_path, {"case_manifest": {
            "file": "discretization/cases.jsonl", "sha256": validator.sha256_file(cases_path),
        }})
        metrics_path = directory / "metrics/cases.json"
        write(metrics_path, {"case_count": 1})
        evidence_path = directory / "evaluation-evidence.json"
        write(evidence_path, {
            "status": "submitted_evaluation", "submission_id": identifier,
            "dataset_id": "drivaernetplusplus", "split_id": "official_test",
            "case_set_id": "official_test", "notes": GENUINE_INFERENCE_ATTESTATION,
            "profile_index_sha256": validator.sha256_file(index_path),
        })
        submission = {
            "schema_version": "3.0", "submission_id": identifier,
            "dataset": "DrivAerNet++", "dataset_id": "drivaernetplusplus",
            "dataset_version": "synthetic-v1", "split_id": "official_test",
            "split_sha256": "a" * 64, "case_set_id": "official_test",
            "prediction_scope": "surface_only", "submitter_name": "Synthetic test",
            "institution": "Test", "submitted_at": f"2026-10-{19 + version:02d}",
            "result_revision": {
                "series_id": series, "version": version,
                "supersedes": f"{series}-v{version - 1}" if version > 1 else None,
                "change_summary": "Synthetic metadata correction.",
            },
            "methodology": {"record_kind": "submitter_reported", "checkpoints": [{
                "id": "synthetic-model", "sha256": "b" * 64,
            }]},
            "scoring_support": {"status": "candidate"},
            "evaluation": {"evidence_file": evidence_path.name,
                           "evidence_sha256": validator.sha256_file(evidence_path)},
            "case_metrics": {"file": "metrics/cases.json", "case_count": 1,
                             "sha256": validator.sha256_file(metrics_path)},
            "spatial_discretization": {"file": spatial_path.name,
                                       "sha256": validator.sha256_file(spatial_path)},
            "profile_data": {"index_file": "profiles/index.json"},
            "metric_values": {"overall_score": 85.0},
        }
        path = directory / "submission.json"
        write(path, submission)
        return path, submission

    def errors(self, manifest=None):
        return validator.validate_result_revisions(
            [self.first, self.second], focus_paths={self.second[0]},
            manifest=self.manifest if manifest is None else manifest,
        )

    def test_registered_predecessor_accepts_unregistered_successor_without_approval(self):
        self.assertEqual(self.errors(), [])
        self.assertNotIn("approval", self.first[1])
        self.assertNotIn("approval", self.second[1])

    def test_registration_is_required_and_cannot_be_self_declared(self):
        self.registry["entries"] = []
        write(self.registry_path, self.registry)
        self.first[1]["record_type"] = "pre_release_reference"
        self.assertIn("predecessor must already be published", "\n".join(self.errors()))

    def test_production_official_and_unspecified_releases_still_reject(self):
        for release in ({"status": "official", "source_ref": "dev"},
                        {"status": "prototype_dummy_data", "source_ref": "main"},
                        {"status": "prototype_dummy_data", "source_ref": "master"}, {}):
            with self.subTest(release=release):
                self.assertIn("predecessor must already be published",
                              "\n".join(self.errors({"data_release": release})))
        self.assertTrue(validator.validate_result_revisions([self.first, self.second]))

    def test_any_changed_package_byte_invalidates_predecessor(self):
        for relative in ("profiles/chunk-000.json", "submission.json", "extra.json"):
            with self.subTest(file=relative):
                path = self.first[0].parent / relative
                original = path.read_bytes() if path.exists() else None
                path.write_bytes((original or b"{}") + b"\n")
                self.assertIn("predecessor must already be published", "\n".join(self.errors()))
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)

    def test_stale_registry_digest_is_rejected(self):
        self.registry["entries"][0]["submission_json_sha256"] = "0" * 64
        write(self.registry_path, self.registry)
        self.assertIn("predecessor must already be published", "\n".join(self.errors()))

    def test_later_revisions_require_the_immediately_preceding_registered_version(self):
        third = self.make_package(3)
        self.assertIn("predecessor must already be published", "\n".join(
            validator.validate_result_revisions(
                [self.first, self.second, third], manifest=self.manifest)))
        self.registry["entries"].append(build_registry_entry(self.second[0], repository_root=self.root))
        write(self.registry_path, self.registry)
        self.assertEqual(validator.validate_result_revisions(
            [self.first, self.second, third], manifest=self.manifest), [])

    def test_scope_and_chronology_checks_remain_enforced(self):
        for field, value, message in (
            ("institution", "Other team", "must have the same institution"),
            ("split_sha256", "0" * 64, "must have the same split_sha256"),
            ("submitted_at", "2026-10-19", "must not be earlier"),
        ):
            original = self.second[1][field]
            with self.subTest(field=field):
                self.second[1][field] = value
                self.assertIn(message, "\n".join(self.errors()))
            self.second[1][field] = original
        third = self.make_package(3)
        third[1]["result_revision"]["supersedes"] = self.first[1]["submission_id"]
        self.assertIn("must supersede 'synthetic-revision-test-v2'", "\n".join(
            validator.validate_result_revisions([self.first, third], manifest=self.manifest)))
        duplicate = (self.root / "duplicate/submission.json", copy.deepcopy(self.second[1]))
        self.assertIn("duplicates", "\n".join(validator.validate_result_revisions(
            [self.first, self.second, duplicate], manifest=self.manifest)))

    def test_validate_many_passes_release_context_for_contributor_intake(self):
        with patch.object(validator, "manifest_with_benchmark_contract", side_effect=lambda m: m), \
             patch.object(validator, "validate_submission_file", return_value=([], {"cases": 1, "series": 3})):
            errors, totals = validator.validate_many(
                [self.second[0]], contributor_stage=True, dev_intake=True, manifest=self.manifest,
            )
        self.assertEqual(errors, [])
        self.assertEqual(totals, {"submissions": 1, "cases": 1, "series": 3})

    def test_feed_retains_v1_until_v2_registration_then_keeps_both_in_history(self):
        def published():
            return manager.source_rows_by_dataset(self.manifest)

        before = manager.latest_rows_by_dataset(published())["DrivAerNet++"]
        self.assertEqual([r["submission_id"] for r in before], [self.first[1]["submission_id"]])
        self.registry["entries"].append(build_registry_entry(self.second[0], repository_root=self.root))
        write(self.registry_path, self.registry)
        rows = published()
        current = manager.latest_rows_by_dataset(rows)["DrivAerNet++"]
        self.assertEqual([r["submission_id"] for r in current], [self.second[1]["submission_id"]])
        self.assertTrue(all(r["record_type"] == "pre_release_reference" for r in rows["DrivAerNet++"]))
        for row in rows["DrivAerNet++"]:
            eligibility = manager.claim_eligibility("prototype_dummy_data", row)
            self.assertFalse(eligibility["academic_citation"])
            self.assertFalse(eligibility["promotion"])
        history = manager.revision_history_payload(self.manifest, rows)
        self.assertEqual((history["record_count"], history["series_count"]), (2, 1))
        self.assertFalse(history["records"][0]["result_revision"]["is_latest"])
        self.assertTrue(history["records"][1]["result_revision"]["is_latest"])


if __name__ == "__main__":
    unittest.main()
