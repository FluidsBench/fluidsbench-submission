"""Missing timing must never become a numeric comparison or official exception."""
import unittest

from reference.drivaerml.methodology import methodology_errors
from scripts.validate_submission import allows_unmeasured_drivaernetplusplus_timing
from tests.test_drivaerml_methodology import methodology_validator, submission_for
from tests.test_drivaerml_candidate_package_assembler import valid_methodology


class CandidateInferenceTimingTests(unittest.TestCase):
    def setUp(self):
        self.submission = {"dataset_id": "drivaernetplusplus", "scoring_support": {"status": "candidate"}}
        self.spec = {"status": "candidate_native_support"}
        self.manifest = {"data_release": {"status": "prototype_dummy_data", "source_ref": "dev"}}

    def allowed(self, **kwargs):
        return allows_unmeasured_drivaernetplusplus_timing(
            self.submission, self.spec, self.manifest,
            candidate_contract_validation=kwargs.get("candidate", True),
        )

    def test_dev_candidate_allows_explicit_missing_timing(self):
        self.assertTrue(self.allowed())
        self.assertFalse(self.allowed(candidate=False))

    def test_release_and_dataset_boundaries_fail_closed(self):
        for field, values in [("status", [None, "official", "approved"]),
                              ("source_ref", [None, "main", "master"])]:
            for value in values:
                with self.subTest(field=field, value=value):
                    original = self.manifest["data_release"][field]
                    self.manifest["data_release"][field] = value
                    self.assertFalse(self.allowed())
                    self.manifest["data_release"][field] = original
        for dataset in [None, "drivaerml", "ahmedml", "airfrans", "hiliftaeroml"]:
            self.submission["dataset_id"] = dataset
            self.assertFalse(self.allowed())

    def test_official_contract_or_support_is_rejected(self):
        for status in [None, "official", "approved", "open"]:
            self.spec["status"] = status
            self.assertFalse(self.allowed())
        self.spec["status"] = "candidate_native_support"
        for status in [None, "official", "approved"]:
            self.submission["scoring_support"]["status"] = status
            self.assertFalse(self.allowed())

    def test_default_semantics_still_require_measured_timing(self):
        method = valid_methodology()
        method["inference_compute"] = {"status": "not_measured", "reason": "Only a partial output-file interval is available."}
        submission = submission_for(method)
        self.assertEqual(list(methodology_validator().iter_errors(method)), [])
        self.assertTrue(any("must be measured" in e for e in methodology_errors(submission)))
        self.assertEqual(methodology_errors(submission, allow_unmeasured_inference=True), [])
        for status in ["estimated", "partial", None]:
            method["inference_compute"]["status"] = status
            self.assertTrue(any("must be measured" in e for e in methodology_errors(submission, allow_unmeasured_inference=True)))

    def test_missing_timing_requires_reason_and_cannot_keep_numeric_costs(self):
        method = valid_methodology()
        for inference in [{"status": "not_measured"},
                          {"status": "not_measured", "reason": ""},
                          {"status": "not_measured", "reason": "Partial interval only", "campaign_wall_time_seconds": 10}]:
            method["inference_compute"] = inference
            self.assertTrue(list(methodology_validator().iter_errors(method)))

    def test_format_examples_do_not_get_candidate_exception(self):
        method = valid_methodology()
        method["record_kind"] = "format_example"
        method["inference_compute"] = {"status": "not_measured", "reason": "Missing"}
        self.assertTrue(any("must be measured" in e for e in methodology_errors(submission_for(method), allow_unmeasured_inference=True)))


if __name__ == "__main__":
    unittest.main()
