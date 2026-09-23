"""Exercise real synthetic recomputation, never a real submission or badge."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reference.evaluate_predictions import evaluate_prediction_artifact
from scripts.compare_prediction_replay import ComparisonError, compare_replay, main

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class PredictionReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        shutil.copytree(ROOT / "examples/v3-template", self.package)
        self.output = self.root / "recomputed.json"
        self.document = evaluate_prediction_artifact(
            support_manifest_path=self.package / "support/manifest.json",
            case_set_id="standard",
            prediction_manifest_path=self.package / "predictions/manifest.json",
            submission_id="synthetic-open-model-v1",
            split_id="default",
        )
        write(self.output, self.document)

    def compare(self):
        return compare_replay(self.package, self.output)

    def test_complete_synthetic_replay_preserves_submission(self):
        before = {p.relative_to(self.package): p.read_bytes() for p in self.package.rglob("*") if p.is_file()}
        report = self.compare()
        self.assertEqual(report["status"], "match")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["scope"], "comparison_only_not_verification_or_approval")
        self.assertEqual(report["submitted_metric_ids"], report["replayed_metric_ids"])
        self.assertEqual({p.relative_to(self.package): p.read_bytes() for p in self.package.rglob("*") if p.is_file()}, before)

    def test_mismatches_never_pass(self):
        mutations = [
            lambda d: d["metric_values"].pop("pressure_mae"),
            lambda d: d["metric_values"].update(pressure_mae=0.9),
            lambda d: d.update(split_id="different-split"),
            lambda d: d.update(scoring_support_manifest_sha256="a" * 64),
            lambda d: d["cases"].reverse(),
            lambda d: d["cases"][0]["supports"][0]["metric_sufficient_statistics"]["pressure_equal_rel_l2"].update(numerator=3.0),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = copy.deepcopy(self.document)
                mutate(document)
                write(self.output, document)
                report = self.compare()
                self.assertEqual(report["status"], "mismatch")
                self.assertGreater(report["difference_count"], 0)

    def test_incomplete_or_invalid_evidence_is_rejected(self):
        mutations = [
            lambda d: d["cases"].pop(),
            lambda d: d["cases"].__setitem__(1, copy.deepcopy(d["cases"][0])),
            lambda d: d["cases"][0]["supports"][0].update(scored_count=2, coverage_fraction=2 / 3),
            lambda d: d["metric_values"].update(pressure_mae=float("nan")),
            lambda d: d["metric_values"].update(pressure_mae=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = copy.deepcopy(self.document)
                mutate(document)
                write(self.output, document)
                with self.assertRaises(ComparisonError):
                    self.compare()

    def test_submitted_hash_and_separate_output_are_required(self):
        with self.assertRaisesRegex(ComparisonError, "separate evaluator output"):
            compare_replay(self.package, self.package / "metrics/cases.json")
        with (self.package / "metrics/cases.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ComparisonError, "SHA-256"):
            self.compare()

    def test_all_metrics_required_and_separate_aggregates_must_agree(self):
        submission = json.loads((self.package / "submission.json").read_text())
        submission["metric_values"]["overall_score"] = 80.0
        write(self.package / "submission.json", submission)
        self.assertEqual(self.compare()["status"], "mismatch")
        aggregate = self.root / "aggregates.json"
        values = {**self.document["metric_values"], "overall_score": 80.0}
        write(aggregate, {"metric_values": values})
        self.assertEqual(compare_replay(self.package, self.output, aggregate)["status"], "match")
        write(aggregate, {"metric_values": values, "split_id": "wrong-split"})
        self.assertEqual(compare_replay(self.package, self.output, aggregate)["status"], "mismatch")
        values["pressure_mae"] = 0.7
        write(aggregate, {"metric_values": values})
        self.assertEqual(compare_replay(self.package, self.output, aggregate)["status"], "mismatch")

    def test_generation_time_and_tiny_roundoff_are_allowed(self):
        self.document["generated_at"] = "2026-09-23T12:00:00Z"
        self.document["metric_values"]["pressure_mae"] += 1e-13
        write(self.output, self.document)
        self.assertEqual(self.compare()["status"], "match")

    def test_cli_refuses_overwrite_and_source_mutation(self):
        report = self.root / "comparison.json"
        args = [str(self.package), "--recomputed-case-metrics", str(self.output), "--output", str(report)]
        self.assertEqual(main(args), 0)
        saved = report.read_bytes()
        self.assertEqual(main(args), 2)
        self.assertEqual(report.read_bytes(), saved)
        self.assertEqual(main(args[:-1] + [str(self.package / "comparison.json")]), 2)
        self.assertFalse((self.package / "comparison.json").exists())
        self.document["metric_values"]["pressure_mae"] += 1
        write(self.output, self.document)
        self.assertEqual(main(args[:-1] + [str(self.root / "mismatch.json")]), 1)

    def test_duplicate_json_keys_are_rejected(self):
        self.output.write_text('{"case_count": 2, "case_count": 1}')
        with self.assertRaisesRegex(ComparisonError, "duplicate JSON key"):
            self.compare()
