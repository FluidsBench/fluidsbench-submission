"""Official truth/support publication has no participant activation effect."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reference.airfrans.profile_truth import ProfileTruthError, validate_release
from scripts.validate_airfrans_official_release import check

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "benchmark-specs/airfrans"
TRUTH = SPEC / "releases/airfrans-evaluation-v1/airfrans/profile-truth/airfrans-native-profile-truth-v1"


class AirfransOfficialReleaseTests(unittest.TestCase):
    def test_dev_intake_opens_without_production_publication_or_candidate_promotion(self):
        self.assertEqual(check()["enabled_split_ids"], ["full", "scarce", "aoa_extrapolation"])
        self.assertTrue(check()["intake_open"])
        self.assertFalse(check()["publicly_published"])
        active = json.loads((SPEC / "submission-spec.json").read_text())
        self.assertTrue(active["scoring_support"]["submissions_open"])
        self.assertEqual(active["status"], "official")
        self.assertEqual(active["intake_branch"], "main")
        truth_manifest = json.loads((TRUTH / "manifest.json").read_text())
        self.assertFalse(truth_manifest["activation"]["published"])

    def test_main_intake_preserves_the_same_approved_scientific_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = root / "benchmark-specs/airfrans"
            shutil.copytree(SPEC, copied)
            path = copied / "submission-spec.json"
            active = json.loads(path.read_text())
            active.update(publication_status="official_intake", intake_branch="main")
            path.write_text(json.dumps(active))
            self.assertTrue(check(root)["intake_open"])
            active["metrics"][0]["unit"] = "modified-unit"
            path.write_text(json.dumps(active))
            with self.assertRaises(ValueError):
                check(root)

    def test_missing_arrays_cannot_pass_full_truth_validation(self):
        with self.assertRaises((OSError, ProfileTruthError)):
            validate_release(release_root=TRUTH, profile_definition_path=SPEC / "velocity-profiles-v1.json",
                             repository_root=ROOT)

    def test_rehashed_manifest_cannot_claim_unapproved_official_truth(self):
        from reference.scoring_support import sha256_file
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "truth"
            shutil.copytree(TRUTH, copied)
            manifest = json.loads((copied / "manifest.json").read_text())
            manifest.pop("owner_approval")
            (copied / "manifest.json").write_text(json.dumps(manifest))
            receipt = json.loads((copied / "release-receipt.json").read_text())
            receipt["manifest_sha256"] = sha256_file(copied / "manifest.json")
            (copied / "release-receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ProfileTruthError, "owner approval"):
                validate_release(release_root=copied, profile_definition_path=SPEC / "velocity-profiles-v1.json",
                                 repository_root=ROOT, metadata_only=True)

    def test_modified_committed_truth_graph_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "truth"
            shutil.copytree(TRUTH, copied)
            index_path = copied / "index.json"
            index_path.write_text(index_path.read_text() + " ")
            with self.assertRaisesRegex(ProfileTruthError, "SHA-256 differs"):
                validate_release(release_root=copied, profile_definition_path=SPEC / "velocity-profiles-v1.json",
                                 repository_root=ROOT, metadata_only=True)
