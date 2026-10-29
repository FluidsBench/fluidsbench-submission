"""Keep the unweighted display explanation aligned with published scorers."""

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_submission import manifest_with_benchmark_contract

from reference.evaluate_predictions import (
    relative_l2_from_sufficient_statistics,
    relative_l2_sufficient_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
METRIC_IDS = {
    "volume_velocity_rel_l2",
    "volume_pressure_rel_l2",
    "surface_pressure_equal_entity_rel_l2",
    "surface_wall_shear_equal_entity_rel_l2",
}


class DisplayEquationTests(unittest.TestCase):
    def test_unweighted_explanation_matches_every_dataset_contract(self):
        manifest = json.loads((ROOT / "leaderboard/manifest.json").read_text())
        definitions = {d["id"]: d for d in manifest["metric_definitions"]}
        for metric_id in METRIC_IDS:
            definition = definitions[metric_id]
            self.assertNotIn("w_i", definition["equation"])
            self.assertIn(r"\lVert", definition["equation"])
            self.assertIn("all components", definition["description"])
            self.assertIn("equal-case mean", definition["description"])
        matched = set()
        for path in (ROOT / "benchmark-specs").glob("*/submission-spec.json"):
            for metric in json.loads(path.read_text())["metrics"]:
                if metric["id"] not in METRIC_IDS:
                    continue
                matched.add(metric["id"])
                with self.subTest(dataset=path.parent.name, metric=metric["id"]):
                    self.assertTrue(metric["weighting"].endswith("_equal"))
                    self.assertEqual(metric["aggregation"], "per_geometry_then_macro_average")
        self.assertEqual(matched, METRIC_IDS)

    def test_feed_rebuild_preserves_the_unweighted_display_equations(self):
        manifest = json.loads((ROOT / "leaderboard/manifest.json").read_text())
        original = {d["id"]: d["equation"] for d in manifest["metric_definitions"]}
        rebuilt = manifest_with_benchmark_contract(manifest)
        for definition in rebuilt["metric_definitions"]:
            if definition["id"] in METRIC_IDS:
                self.assertEqual(definition["equation"], original[definition["id"]])
                self.assertNotIn("w_i", definition["equation"])
                self.assertIn(r"\lVert", definition["equation"])

    def test_uniform_scalar_and_vector_errors_ignore_physical_weights(self):
        for truth, prediction in [
            ([[1.0], [2.0]], [[2.0], [2.0]]),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[2.0, 2.0, 5.0], [4.0, 6.0, 3.0]]),
        ]:
            truth, prediction = np.array(truth), np.array(prediction)
            expected = 100 * np.linalg.norm(prediction - truth) / np.linalg.norm(truth)
            for weights in [np.array([1.0, 1.0]), np.array([2.0, 90.0])]:
                stats = relative_l2_sufficient_statistics(truth, prediction, weights, weighting="uniform")
                self.assertAlmostEqual(relative_l2_from_sufficient_statistics([stats]), expected)


if __name__ == "__main__":
    unittest.main()
