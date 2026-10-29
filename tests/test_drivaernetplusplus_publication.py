"""Retained full-release evidence stays bound to the reviewed closed candidate."""
import copy
import hashlib
import json
from pathlib import Path
import importlib.util
import unittest
from unittest import mock

from reference.drivaernetplusplus import contract, distribution

PATH = contract.SPEC_ROOT / "tools/validate_publication.py"
SPEC = importlib.util.spec_from_file_location("dnpp_publication", PATH)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


class PreparedPublication(unittest.TestCase):
    def setUp(self):
        # Reproduce the historical receipt against its archived inputs, not the
        # newer official contract. No historical Git commits are needed.
        root = Path(__file__).resolve().parent / "fixtures/drivaernetplusplus-publication-inputs"
        archived = root / "benchmark-specs/drivaernetplusplus/submission-spec.json"
        candidate = json.loads(archived.read_text())
        def hashes():
            return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*") if p.is_file()}
        for patcher in (mock.patch.object(contract, "load_specification", return_value=candidate),
                        mock.patch.object(publication, "input_hashes", side_effect=hashes)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_committed_full_validation_evidence(self):
        receipt = contract.read_json(publication.PREPARATION / "publication-validation.json")
        publication.check_receipt(receipt)
        self.assertEqual(receipt["native_point_count"], 516881608)
        self.assertEqual(receipt["profile_series_count"], 3462)
        self.assertEqual(receipt["pressure_extremes"], {"points": 768, "cases": 79, "masked": False})
        self.assertFalse(receipt["source_mesh_replay_performed"])
        self.assertFalse(receipt["model_inference_performed"])

    def test_replaced_or_missing_hosted_archive_rejected(self):
        hosted = contract.read_json(publication.PREPARATION / "hosted-assets.json")
        for key, value in (("digest", "sha256:" + "0" * 64), ("size", 0),
                           ("browser_download_url", "https://example.com/replaced.zip")):
            altered = copy.deepcopy(hosted)
            altered["assets"][0][key] = value
            with self.subTest(field=key), self.assertRaises(contract.ContractError):
                publication.check_hosted_assets(altered, distribution.load_distribution())
        hosted["assets"].pop()
        with self.assertRaises(contract.ContractError):
            publication.check_hosted_assets(hosted, distribution.load_distribution())

    def test_stale_receipt_cannot_pass(self):
        receipt = contract.read_json(publication.PREPARATION / "publication-validation.json")
        receipt["input_sha256"]["benchmark-specs/drivaernetplusplus/OWNER_APPROVAL.md"] = "0" * 64
        with self.assertRaisesRegex(contract.ContractError, "inputs changed"):
            publication.check_receipt(receipt)

    def test_preparation_does_not_accept_open_intake(self):
        spec = contract.load_specification()
        spec["scoring_support"]["submissions_open"] = True
        with mock.patch.object(contract, "load_specification", return_value=spec):
            with self.assertRaisesRegex(contract.ContractError, "stay closed"):
                publication.check_contract()


if __name__ == "__main__":
    unittest.main()
