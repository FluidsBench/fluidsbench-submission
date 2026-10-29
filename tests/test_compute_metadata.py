"""Shared compute contract: legacy compatibility, allocation and assembler paths."""
from __future__ import annotations

import copy
import json
import unittest

from reference.methodology import methodology_errors
from scripts import assemble_ahmedml_schema_v3_candidate as ahmed
from scripts import assemble_drivaerml_schema_v3_candidate as drivaer
from scripts import assemble_hiliftaeroml_schema_v3_candidate as hilift
from tests.test_drivaerml_candidate_package_assembler import valid_methodology
from tests.test_drivaerml_methodology import ROOT, methodology_validator, submission_for


class ComputeMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.method = valid_methodology()
        self.blocks = [self.method["training"]["stages"][0]["compute"],
                       self.method["inference_compute"]]

    def assert_valid(self) -> None:
        self.assertEqual(list(methodology_validator().iter_errors(self.method)), [])
        self.assertEqual(methodology_errors(submission_for(self.method), expected_case_count=2), [])

    def test_legacy_fields_remain_valid(self) -> None:
        self.assert_valid()
        for block in self.blocks:
            self.assertNotIn("accelerator", block)
            self.assertNotIn("devices_per_job", block)

    def test_gpu_identity_and_parallel_jobs_are_valid(self) -> None:
        for block in self.blocks:
            block.update(accelerator={"type": "gpu", "vendor": "NVIDIA", "model": "H200"},
                         devices_per_job=4, max_concurrent_device_count=40)
        self.assert_valid()

    def test_unknown_mixed_and_non_gpu_hardware(self) -> None:
        for identity in [
            {"type": "gpu", "vendor": "NVIDIA", "model": None},
            {"type": "gpu", "vendor": None, "model": None},
            {"type": "mixed", "vendor": "NVIDIA", "model": None},
            {"type": "mixed", "vendor": None, "model": None},
            {"type": "unknown", "vendor": None, "model": None},
            {"type": "cpu", "vendor": "AMD", "model": "EPYC 9654"},
            {"type": "tpu", "vendor": "Google", "model": "v5e"},
            {"type": "other", "vendor": None, "model": "Example device"},
        ]:
            with self.subTest(identity=identity):
                for block in self.blocks:
                    block.update(accelerator=identity, devices_per_job=None)
                self.assert_valid()

    def test_schema_rejects_invalid_or_contradictory_identity(self) -> None:
        for identity in [None, {}, {"type": "gpu"},
                         {"type": "cuda", "vendor": None, "model": None},
                         {"type": "gpu", "vendor": " ", "model": "H200"},
                         {"type": "gpu", "vendor": "NVIDIA", "model": ""},
                         {"type": "gpu", "vendor": "NVIDIA", "model": "x" * 161},
                         {"type": "mixed", "vendor": None, "model": "H200"},
                         {"type": "unknown", "vendor": "NVIDIA", "model": None},
                         {"type": "gpu", "vendor": None, "model": None, "count": 4}]:
            for block in self.blocks:
                with self.subTest(identity=identity, block=block.get("status", "training")):
                    block["accelerator"] = identity
                    self.assertTrue(list(methodology_validator().iter_errors(self.method)))
                    del block["accelerator"]

    def test_allocation_must_be_a_positive_integer_or_null(self) -> None:
        for value in [0, -1, 1.5, True, "4", {}, []]:
            for block in self.blocks:
                with self.subTest(value=value):
                    block["devices_per_job"] = value
                    self.assertTrue(list(methodology_validator().iter_errors(self.method)))
                    del block["devices_per_job"]

    def test_per_job_cannot_exceed_campaign_peak_for_either_scope(self) -> None:
        for block, label in zip(self.blocks, ["training.stages[0].compute", "inference_compute"]):
            with self.subTest(scope=label):
                block.update(devices_per_job=8, max_concurrent_device_count=4)
                errors = methodology_errors(submission_for(self.method))
                self.assertIn(f"methodology.{label}.devices_per_job cannot exceed max_concurrent_device_count", errors)
                block["devices_per_job"] = 4
                self.assert_valid()

    def test_capacity_and_complete_split_checks_still_apply(self) -> None:
        for block in self.blocks:
            block.update(accelerator={"type": "gpu", "vendor": "AMD", "model": "MI300X"}, devices_per_job=1)
        self.blocks[0]["aggregate_device_hours"] = 1e9
        self.blocks[1]["aggregate_device_time_seconds"] = 1e9
        self.blocks[1]["case_count"] = 1
        errors = "\n".join(methodology_errors(submission_for(self.method), expected_case_count=2))
        self.assertIn("aggregate_device_hours cannot exceed", errors)
        self.assertIn("aggregate_device_time_seconds cannot exceed", errors)
        self.assertIn("official evaluation case count 2", errors)

    def test_integer_json_numbers_cannot_bypass_allocation_check(self) -> None:
        for per_job, peak in [(8.0, 4), (8, 4.0), (8.0, 4.0)]:
            with self.subTest(per_job=per_job, peak=peak):
                self.blocks[1].update(devices_per_job=per_job, max_concurrent_device_count=peak)
                self.assertEqual(list(methodology_validator().iter_errors(self.method)), [])
                errors = methodology_errors(submission_for(self.method))
                self.assertIn("methodology.inference_compute.devices_per_job cannot exceed max_concurrent_device_count", errors)

    def test_hardware_does_not_make_unmeasured_inference_valid(self) -> None:
        self.method["inference_compute"] = {"status": "not_measured", "reason": "No measurement"}
        self.assertIn("must be measured", " ".join(methodology_errors(submission_for(self.method))))
        self.method["inference_compute"]["accelerator"] = {"type": "gpu", "vendor": "NVIDIA", "model": "H200"}
        self.assertTrue(list(methodology_validator().iter_errors(self.method)))

    def test_all_three_assemblers_accept_the_extended_common_schema(self) -> None:
        for block in self.blocks:
            block.update(accelerator={"type": "gpu", "vendor": "NVIDIA", "model": "H200"}, devices_per_job=1)
        before = copy.deepcopy(self.method)
        for assembler in (ahmed, drivaer, hilift):
            assembler._require_methodology_schema(self.method)
        self.assertEqual(self.method, before)

    def test_hilift_participant_path_preserves_metadata(self) -> None:
        config = json.loads((ROOT / "examples/hiliftaeroml-v3-candidate/transolver-full360-candidate-config.json").read_text())
        method = config["participant"]["methodology"]
        contract = json.loads((ROOT / "benchmark-specs/hiliftaeroml/methodology-contract.json").read_text())
        submission = hilift._participant_submission(config=config, case_count=360, methodology_contract=contract)
        self.assertEqual(submission["methodology"], method)
        self.assertEqual(method["inference_compute"]["accelerator"]["type"], "gpu")
        self.assertIsNone(method["inference_compute"]["accelerator"]["model"])

    def test_checked_in_candidate_methodologies_validate_without_guessed_models(self) -> None:
        for path in sorted((ROOT / "examples/hiliftaeroml-v3-candidate").rglob("*-candidate-config.json")):
            with self.subTest(config=path.name):
                method = json.loads(path.read_text())["participant"]["methodology"]
                self.assertEqual(list(methodology_validator().iter_errors(method)), [])
                for compute in [s["compute"] for s in method["training"]["stages"]] + [method["inference_compute"]]:
                    self.assertIsNone(compute["accelerator"]["model"])


if __name__ == "__main__":
    unittest.main()
