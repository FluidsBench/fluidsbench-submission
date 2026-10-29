"""Cross-contract publication must retain history without ranking the same run twice."""
import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.display_replacements import REGISTRY, apply_display_replacements
from scripts.manage_leaderboard import annotate_result_revisions, latest_rows_by_dataset


class DisplayReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {"data_release": {"status": "prototype_dummy_data", "source_ref": "dev"}}
        common = dict(dataset_id="airfrans", split_id="aoa", split_sha256="split", case_set_id="aoa", model="Model", methodology={"checkpoint": "fixed"}, reproducibility={}, metric_values={"score": 40.61})
        self.source = dict(common, submission_id="model-old-v1", dataset_version="prototype-1", record_type="pre_release_reference", result_revision={"series_id": "model-old", "version": 1, "supersedes": None})
        self.target = dict(common, submission_id="model-new-v1", dataset_version="native-v1", approval={"status": "approved"}, result_revision={"series_id": "model-new", "version": 1, "supersedes": None})
        self.item = dict(source_submission_id=self.source["submission_id"], replacement_submission_id=self.target["submission_id"], reason="Same inference under the published contract.", review_url="https://github.com/example/repo/pull/1")
        for label, row in (("source", self.source), ("replacement", self.target)):
            directory = self.root / "submissions" / "airfrans" / row["submission_id"]
            directory.mkdir(parents=True)
            file = directory / "submission.json"
            file.write_text(json.dumps(row))
            self.item[f"{label}_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            for section, field, content in (("profile_data", "index_file", "chunks"), ("case_metrics", "file", "cases")):
                artifact = directory / f"{section}.json"
                artifact.write_text(json.dumps({content: ["same data"]}))
                row[section] = {field: artifact.relative_to(self.root).as_posix()}
        self.rows = {"AirfRANS": [self.source, self.target]}
        annotate_result_revisions(self.rows)
        self.write_registry()

    def write_registry(self, items=None):
        path = self.root / REGISTRY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": "1.0", "replacements": items if items is not None else [self.item]}))

    def apply(self):
        apply_display_replacements(self.root, self.manifest, self.rows)

    def test_only_replacement_current_both_preserved_in_history(self):
        self.apply()
        self.assertEqual([r["submission_id"] for r in latest_rows_by_dataset(self.rows)["AirfRANS"]], ["model-new-v1"])
        self.assertEqual(len(self.rows["AirfRANS"]), 2)
        self.assertTrue(self.source["result_revision"]["is_latest"])
        self.assertEqual(self.source["display_replacement"]["replaced_by"], "model-new-v1")
        self.assertEqual(self.target["display_replacement"]["replaces"], "model-old-v1")

    def test_unapproved_missing_changed_or_different_results_fail_closed(self):
        for change, message in (({"approval": {}}, "approved"), ({"metric_values": {"score": 99}}, "metric_values"), ({"split_sha256": "different"}, "split_sha256"), ({"dataset_version": "prototype-1"}, "result_revision")):
            with self.subTest(change=change):
                original = deepcopy(self.target)
                self.target.update(change)
                with self.assertRaisesRegex(ValueError, message): self.apply()
                self.assertNotIn("display_replacement", self.source)
                self.target.clear()
                self.target.update(original)
        self.rows["AirfRANS"].remove(self.target)
        with self.assertRaisesRegex(ValueError, "both published"): self.apply()

    def test_package_hash_and_profile_changes_fail_closed(self):
        self.item["replacement_sha256"] = "0" * 64
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "hash mismatch"): self.apply()
        file = self.root / "submissions/airfrans/model-new-v1/submission.json"
        self.item["replacement_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.write_registry()
        (self.root / self.target["profile_data"]["index_file"]).write_text('{"chunks": ["changed"]}')
        with self.assertRaisesRegex(ValueError, "profile_data"): self.apply()

    def test_duplicate_and_cycles_fail_without_hiding_anything(self):
        self.write_registry([self.item, self.item])
        with self.assertRaisesRegex(ValueError, "duplicate"): self.apply()
        self.assertNotIn("display_replacement", self.source)
        reverse = dict(self.item, source_submission_id="model-new-v1", replacement_submission_id="model-old-v1")
        self.write_registry([self.item, reverse])
        with self.assertRaisesRegex(ValueError, "chains and cycles"): self.apply()
        self.assertNotIn("display_replacement", self.source)

    def test_official_release_is_unaffected(self):
        self.manifest["data_release"]["status"] = "official"
        self.apply()
        self.assertEqual(len(latest_rows_by_dataset(self.rows)["AirfRANS"]), 2)

    def test_future_revision_keeps_replacement_link(self):
        successor = deepcopy(self.target)
        successor.update(submission_id="model-new-v2", result_revision={"series_id": "model-new", "version": 2, "supersedes": "model-new-v1"})
        self.rows["AirfRANS"].append(successor)
        annotate_result_revisions(self.rows)
        self.apply()
        self.assertEqual([r["submission_id"] for r in latest_rows_by_dataset(self.rows)["AirfRANS"]], ["model-new-v2"])
        self.assertEqual(successor["display_replacement"]["replaces"], "model-old-v1")
