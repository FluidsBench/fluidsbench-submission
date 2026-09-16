from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json"
EVIDENCE_PATH = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "evidence"
    / "force-replay-all484.json"
)
EXPECTED_SHA256 = "631cd02c3a4215b254489652c1d93dfd781ecdb9478ff1ab11f24294743e8a17"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DrivAerMLForceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_submission_contract_binds_exact_candidate_evidence(self) -> None:
        binding = self.spec["scoring_support"]["force_integration"][
            "candidate_validation_evidence"
        ]
        self.assertEqual(binding["file"], "evidence/force-replay-all484.json")
        self.assertEqual(binding["sha256"], EXPECTED_SHA256)
        self.assertTrue(binding["owner_scientific_approval"])
        self.assertEqual(_sha256(EVIDENCE_PATH), EXPECTED_SHA256)

    def test_all_case_surface_force_and_area_replay_is_complete_but_candidate(self) -> None:
        evidence = self.evidence
        self.assertEqual(evidence["case_count"], 484)
        self.assertEqual(len(evidence["case_receipts"]), 484)
        self.assertEqual(
            evidence["status"],
            "passed_candidate_evaluator_all_484_case_force_replay",
        )
        self.assertEqual(
            evidence["mirror_truth_validation"]["status"],
            "exact_match_to_authoritative_aggregate",
        )
        self.assertEqual(
            evidence["surface_area_audit"]["entity_count"], 4_159_517_910
        )
        self.assertLessEqual(
            evidence["coefficient_absolute_difference"]["Cd"][
                "maximum_absolute_difference"
            ],
            evidence["tolerances"]["coefficient_absolute"],
        )
        self.assertLessEqual(
            evidence["chunk_invariance"]["overall"][
                "maximum_absolute_difference"
            ],
            evidence["tolerances"]["chunk_invariance_absolute"],
        )
        self.assertLessEqual(
            evidence["lift_closure"]["authoritative_and_mirror_truth"][
                "maximum_absolute_difference"
            ],
            evidence["tolerances"][
                "source_lift_closure_absolute_csv_rounding"
            ]
            + evidence["tolerances"][
                "source_lift_closure_binary_comparison_guard_absolute"
            ],
        )


if __name__ == "__main__":
    unittest.main()
