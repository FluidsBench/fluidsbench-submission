"""Shared metric IDs retain each dataset's authoritative mathematical rules."""
import json
from pathlib import Path
import unittest
from scripts import manage_leaderboard as manager

ROOT = Path(__file__).resolve().parents[1]


class DatasetMetricContractTests(unittest.TestCase):
    def test_generated_definitions_match_every_specification(self):
        manifest = json.loads((ROOT / 'leaderboard/manifest.json').read_text())
        generated, _, _, _ = manager.expected_outputs(manifest)
        for dataset in generated['datasets']:
            spec = json.loads((ROOT / 'benchmark-specs' / dataset['slug'] / 'submission-spec.json').read_text())
            expected = {metric['id']: {key: metric[key] for key in ('equation', 'aggregation', 'weighting') if key in metric}
                        for metric in spec['metrics']}
            self.assertEqual(dataset['metric_contracts'], expected, dataset['slug'])
        contracts = {dataset['slug']: dataset['metric_contracts'] for dataset in generated['datasets']}
        self.assertNotEqual(contracts['airfrans']['velocity_profile_r2'], contracts['drivaerml']['velocity_profile_r2'])
        self.assertIn('trapezoidal', contracts['drivaerml']['cp_cut_r2']['weighting'])
        self.assertIn('bounded', contracts['airfrans']['velocity_profile_r2']['aggregation'])
