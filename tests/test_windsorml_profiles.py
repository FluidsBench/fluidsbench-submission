"""Runtime enforcement of the published WindsorML profile support."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reference.windsorml import profiles  # noqa: E402
from reference.windsorml.evaluator import (  # noqa: E402
    WindsorMLCandidateEvaluatorError,
    evaluate_candidate_case,
)


class WindsorMLProfileSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.support_path = profiles.PROFILE_MANIFEST_PATH.parent / "run_0.json"
        self.support = json.loads(self.support_path.read_text())

    def test_all_233_frozen_case_supports_load(self):
        manifest = json.loads(profiles.PROFILE_MANIFEST_PATH.read_text())
        self.assertEqual(len(manifest["cases"]), 233)
        for entry in manifest["cases"]:
            with self.subTest(case_id=entry["case_id"]):
                support = profiles.load_profile_support(case_id=entry["case_id"])
                self.assertEqual(support.sha256, entry["sha256"])
                self.assertEqual(support.document["sample_count"], 128)

    def test_exact_file_copy_and_decoded_document_are_accepted(self):
        path = self.root / "copy.json"
        path.write_bytes(self.support_path.read_bytes())
        copied = profiles.load_profile_support(case_id="run_0", value=path)
        decoded = profiles.load_profile_support(case_id="run_0", value=self.support)
        self.assertEqual(copied, decoded)

    def test_modified_file_is_rejected_even_if_values_are_unchanged(self):
        path = self.root / "copy.json"
        path.write_bytes(self.support_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(profiles.WindsorMLProfileError, "SHA-256"):
            profiles.load_profile_support(case_id="run_0", value=path)

    def test_modified_in_memory_support_is_rejected(self):
        family = "windsorml_velocity_constant_v1"
        station = "wake_vertical_x_0p05l"
        mutations = {
            "different native ID": lambda d: d["families"][family][station][
                "native_cell_ids"
            ].__setitem__(0, 0),
            "omitted station": lambda d: d["families"][family].pop(station),
            "extra family": lambda d: d["families"].update(extra={}),
            "changed coordinate": lambda d: d["families"][family][station][
                "coordinate"
            ].__setitem__(0, -1),
            "different velocity scale": lambda d: d.update(reference_velocity_m_s=1.0),
            "different body height": lambda d: d.update(body_height_m=1.0),
            "wrong case": lambda d: d.update(case_id="run_7"),
            "NaN truth": lambda d: d["families"][family][station][
                "truth_ux_over_uinf"
            ].__setitem__(0, float("nan")),
            "boolean native ID": lambda d: d["families"][family][station][
                "native_cell_ids"
            ].__setitem__(0, True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                document = deepcopy(self.support)
                mutate(document)
                with self.assertRaises(profiles.WindsorMLProfileError):
                    profiles.load_profile_support(case_id="run_0", value=document)

    def test_manifest_and_definition_bytes_are_verified_on_every_load(self):
        (self.root / "run_0.json").write_bytes(self.support_path.read_bytes())
        for attribute in ("PROFILE_MANIFEST_PATH", "PROFILE_DEFINITION_PATH"):
            with self.subTest(path=attribute):
                path = self.root / attribute
                path.write_bytes(getattr(profiles, attribute).read_bytes())
                with patch.object(profiles, attribute, path):
                    profiles.load_profile_support(case_id="run_0")
                    path.write_bytes(path.read_bytes() + b"\n")
                    with self.assertRaisesRegex(
                        profiles.WindsorMLProfileError, "SHA-256"
                    ):
                        profiles.load_profile_support(case_id="run_0")

    def test_support_bytes_are_rechecked_after_a_successful_load(self):
        path = self.root / "copy.json"
        path.write_bytes(self.support_path.read_bytes())
        profiles.load_profile_support(case_id="run_0", value=path)
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(profiles.WindsorMLProfileError, "SHA-256"):
            profiles.load_profile_support(case_id="run_0", value=path)

    def test_unscored_case_cannot_supply_its_own_support(self):
        with self.assertRaisesRegex(profiles.WindsorMLProfileError, "does not bind"):
            profiles.load_profile_support(case_id="run_350", value=self.support)

    def test_evaluator_checks_support_before_opening_prediction_or_source_files(self):
        self.support["sample_count"] = 2
        with self.assertRaisesRegex(WindsorMLCandidateEvaluatorError, "hash-pinned"):
            evaluate_candidate_case(
                case=SimpleNamespace(case_id="run_0"),
                dataset_root=self.root / "not-needed",
                surface_manifest=self.root / "not-needed.json",
                profile_support=self.support,
            )


if __name__ == "__main__":
    unittest.main()
