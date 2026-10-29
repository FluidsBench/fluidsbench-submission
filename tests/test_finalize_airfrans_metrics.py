from __future__ import annotations

import copy
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finalize_airfrans_metrics import FinalizeError, apply_profile_score_and_composites


def _spec() -> dict:
    # Trimmed to two field components, one force component, and the profile
    # metric, so the expected composite numbers are easy to hand-verify.
    return {
        "overall_score_composite": {
            "metric_id": "overall_score",
            "operation": "weighted_component_scores",
            "components": [
                {
                    "metric_id": "surface_pressure_rel_l2",
                    "weight": 0.4,
                    "transform": "bounded_error",
                    "cap": 10.0,
                },
                {
                    "metric_id": "cd_r2",
                    "weight": 0.3,
                    "transform": "bounded_quality",
                },
                {
                    "metric_id": "velocity_profile_r2",
                    "weight": 0.3,
                    "transform": "bounded_quality",
                },
            ],
            "tolerance": 1e-6,
        },
        "component_score_groups": {
            "operation": "normalized_weighted_component_scores",
            "groups": [
                {
                    "metric_id": "field_score",
                    "component_metric_ids": ["surface_pressure_rel_l2"],
                },
                {
                    "metric_id": "force_score",
                    "component_metric_ids": ["cd_r2"],
                },
                {
                    "metric_id": "diagnostic_score",
                    "component_metric_ids": ["velocity_profile_r2"],
                },
            ],
            "tolerance": 1e-6,
        },
    }


def _case_metrics() -> dict:
    return {
        "case_count": 200,
        "metric_values": {
            "surface_pressure_rel_l2": 5.0,  # bounded_error: 100*(1-5/10) = 50
            "cd_r2": 0.5,  # bounded_quality: 100*0.5 = 50
        },
    }


def _profile_score() -> dict:
    return {
        "metric_id": "velocity_profile_r2",
        "case_coverage": "complete_official_split",
        "case_count": 200,
        "value": 0.8,  # bounded_quality: 100*0.8 = 80
    }


class ApplyProfileScoreAndCompositesTests(unittest.TestCase):
    def test_merges_profile_r2_and_computes_composites(self) -> None:
        case_metrics = copy.deepcopy(_case_metrics())
        result = apply_profile_score_and_composites(case_metrics, _profile_score(), _spec())

        self.assertIs(result, case_metrics)
        values = result["metric_values"]
        self.assertAlmostEqual(values["velocity_profile_r2"], 0.8)

        # overall = 0.4*50 + 0.3*50 + 0.3*80 = 20 + 15 + 24 = 59
        self.assertAlmostEqual(values["overall_score"], 59.0, places=6)
        # each group here has exactly one component, so the group score
        # equals that component's own transformed score.
        self.assertAlmostEqual(values["field_score"], 50.0, places=6)
        self.assertAlmostEqual(values["force_score"], 50.0, places=6)
        self.assertAlmostEqual(values["diagnostic_score"], 80.0, places=6)

    def test_rejects_wrong_metric_id(self) -> None:
        profile_score = dict(_profile_score(), metric_id="something_else")
        with self.assertRaises(FinalizeError):
            apply_profile_score_and_composites(_case_metrics(), profile_score, _spec())

    def test_rejects_incomplete_profile_coverage(self) -> None:
        profile_score = dict(_profile_score(), case_coverage="partial")
        with self.assertRaises(FinalizeError):
            apply_profile_score_and_composites(_case_metrics(), profile_score, _spec())

    def test_rejects_case_count_mismatch(self) -> None:
        profile_score = dict(_profile_score(), case_count=196)
        with self.assertRaises(FinalizeError):
            apply_profile_score_and_composites(_case_metrics(), profile_score, _spec())


if __name__ == "__main__":
    unittest.main()
