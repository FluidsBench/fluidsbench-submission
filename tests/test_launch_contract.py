"""Launch behavior independent of any retained participant submission or Git history."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import manage_leaderboard as manager
from scripts.validate_scoring_supports import validate_specification

ROOT = Path(__file__).resolve().parents[1]


class LaunchContractTests(unittest.TestCase):
    def test_exactly_five_owner_approved_datasets_accept_submissions(self):
        opened = set()
        for path in (ROOT / 'benchmark-specs').glob('*/submission-spec.json'):
            spec = json.loads(path.read_text())
            support = spec['scoring_support']
            if support['submissions_open']:
                opened.add(spec['dataset_id'])
                self.assertEqual(spec['status'], 'official')
                self.assertEqual(support['status'], 'official')
                self.assertTrue(support['owner_approval']['approved_by'])
                self.assertEqual(validate_specification(path), [])
        self.assertEqual(opened, {'airfrans', 'ahmedml', 'windsorml', 'drivaerml', 'drivaernetplusplus'})
        airfrans = json.loads((ROOT / 'benchmark-specs/airfrans/submission-spec.json').read_text())
        self.assertEqual({s['id'] for s in airfrans['splits']}, {'full', 'scarce', 'aoa_extrapolation'})

    def test_unreleased_feed_does_not_publish_approved_packages(self):
        manifest = {'data_release': {'status': 'unreleased'}, 'datasets': [{'name': 'Example'}]}
        package = ROOT / 'submissions/example/synthetic-v1/submission.json'
        with patch.object(manager, 'submission_files', return_value=[package]):
            self.assertEqual(manager.source_rows_by_dataset(manifest), {'Example': []})
        row = {'approval': {'status': 'approved'}, 'maintainer_validation': {'status': 'validated'}}
        eligibility = manager.claim_eligibility('unreleased', row)
        self.assertFalse(eligibility['academic_citation'])
        self.assertFalse(eligibility['promotion'])
        self.assertTrue(manager.claim_eligibility('official', row)['academic_citation'])
