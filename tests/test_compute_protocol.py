"""Complete-case protocol, repetition pairing and historical GB200 evidence."""
import copy
import json
import unittest

from jsonschema import Draft202012Validator
from reference.methodology import methodology_errors
from tests.test_drivaerml_candidate_package_assembler import valid_methodology
from tests.test_drivaerml_methodology import ROOT, methodology_validator, submission_for


def protocol_method():
    method = valid_methodology()
    compute = method['inference_compute']
    compute.update(
        timing_protocol='fluidsbench-complete-case-v1',
        accelerator={'type': 'gpu', 'vendor': 'NVIDIA', 'model': 'GB200'},
        hardware_identity_basis='owner_confirmed', devices_per_job=1,
        max_concurrent_device_count=4,
        includes_preprocessing=True, includes_mapping=True,
        execution={'precision': 'fp32', 'batch_size': 1, 'batch_unit': 'chunk',
                   'warmup_cases': 1, 'software': 'Synthetic test runtime 1.0'},
        campaign_wall_time_seconds=20, aggregate_device_time_seconds=60,
        campaign_runs=[{'wall_time_seconds': 30, 'aggregate_device_time_seconds': 90},
                       {'wall_time_seconds': 10, 'aggregate_device_time_seconds': 35},
                       {'wall_time_seconds': 20, 'aggregate_device_time_seconds': 60}],
    )
    return method


class CompleteCaseProtocolTests(unittest.TestCase):
    def setUp(self):
        self.method = protocol_method()
        self.compute = self.method['inference_compute']

    def test_complete_case_protocol_is_valid_and_preserved_by_assemblers(self):
        from scripts import assemble_ahmedml_schema_v3_candidate as ahmed
        from scripts import assemble_drivaerml_schema_v3_candidate as drivaer
        from scripts import assemble_hiliftaeroml_schema_v3_candidate as hilift
        self.assertEqual(list(methodology_validator().iter_errors(self.method)), [])
        self.assertEqual(methodology_errors(submission_for(self.method), expected_case_count=2), [])
        before = copy.deepcopy(self.method)
        for assembler in [ahmed, drivaer, hilift]:
            assembler._require_methodology_schema(self.method)
        self.assertEqual(before, self.method)

    def test_synthetic_documented_example_is_valid(self):
        method = json.loads((ROOT / 'examples/compute-v1/methodology.example.json').read_text())
        self.assertEqual(list(methodology_validator().iter_errors(method)), [])
        self.assertEqual(methodology_errors(submission_for(method), expected_case_count=50), [])

    def test_extreme_or_nonfinite_repetition_times_fail_without_crashing(self):
        for value in [10 ** 1000, float('nan'), float('inf'), 0, True]:
            method = copy.deepcopy(self.method)
            method['inference_compute']['campaign_runs'][0]['wall_time_seconds'] = value
            self.assertIn('finite positive numbers', '\n'.join(methodology_errors(submission_for(method))))

    def test_declared_protocol_requires_recorded_conditions(self):
        for field in ['execution', 'accelerator', 'campaign_runs', 'hardware_identity_basis', 'devices_per_job']:
            method = copy.deepcopy(self.method)
            del method['inference_compute'][field]
            with self.subTest(field=field):
                self.assertTrue(list(methodology_validator().iter_errors(method)))
        for field, value in [('includes_mapping', False), ('includes_preprocessing', False),
                             ('timing_protocol', 'invented-v1')]:
            method = copy.deepcopy(self.method)
            method['inference_compute'][field] = value
            self.assertTrue(list(methodology_validator().iter_errors(method)))
        for field, value in [('precision', 'unknown'), ('warmup_cases', None), ('software', None),
                             ('batch_unit', None), ('batch_size', 0), ('warmup_cases', -1)]:
            method = copy.deepcopy(self.method)
            method['inference_compute']['execution'][field] = value
            self.assertTrue(list(methodology_validator().iter_errors(method)))

    def test_lower_median_selects_a_paired_run_with_stable_ties(self):
        self.compute['campaign_runs'].append({'wall_time_seconds': 20, 'aggregate_device_time_seconds': 75})
        self.assertEqual(methodology_errors(submission_for(self.method)), [])
        for key, value in [('campaign_wall_time_seconds', 10), ('aggregate_device_time_seconds', 35)]:
            method = copy.deepcopy(self.method)
            method['inference_compute'][key] = value
            self.assertIn('must match the same lower-median', '\n'.join(methodology_errors(submission_for(method))))

    def test_every_repetition_must_fit_device_capacity(self):
        self.compute['max_concurrent_device_count'] = 4.0
        self.compute['campaign_runs'][0]['aggregate_device_time_seconds'] = 121
        errors = '\n'.join(methodology_errors(submission_for(self.method)))
        self.assertIn('campaign_runs[0].aggregate_device_time_seconds cannot exceed', errors)

    def test_repetitions_do_not_multiply_the_official_case_count(self):
        self.compute['case_count'] = 6
        self.assertIn('official evaluation case count 2', '\n'.join(methodology_errors(submission_for(self.method), expected_case_count=2)))

    def test_legacy_can_retain_unknown_execution_without_claiming_protocol(self):
        self.compute['timing_protocol'] = 'legacy_reported'
        self.compute['execution'].update(precision='unknown', warmup_cases=None, software=None, batch_unit=None)
        self.compute['accelerator']['model'] = None
        self.compute['hardware_identity_basis'] = 'unknown'
        self.assertEqual(list(methodology_validator().iter_errors(self.method)), [])

    def test_hilift_real_compute_blocks_validate_without_recertifying_history(self):
        fixture = json.loads((ROOT / 'tests/fixtures/hilift-gb200-compute.json').read_text())
        defs = json.loads((ROOT / 'schemas/v3/submission.schema.json').read_text())['$defs']
        method = fixture['methodology']
        blocks = [('methodology_measured_inference_compute', method['inference_compute'])]
        blocks.extend(('methodology_training_compute', s['compute']) for s in method['training']['stages'])
        for definition, block in blocks:
            validator = Draft202012Validator({'$defs': defs, '$ref': '#/$defs/' + definition})
            self.assertEqual(list(validator.iter_errors(block)), [])
            self.assertEqual(block['accelerator']['model'], 'GB200')
            self.assertEqual(block['hardware_identity_basis'], 'owner_confirmed')
        self.assertEqual(method['inference_compute']['timing_protocol'], 'legacy_reported')
        self.assertNotIn('campaign_runs', method['inference_compute'])
        self.assertTrue(all(s['compute']['measurement_basis'] == 'estimated' for s in method['training']['stages']))


if __name__ == '__main__':
    unittest.main()
