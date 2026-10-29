from __future__ import annotations

import copy
import unittest
from pathlib import Path

from reference.airfrans_profiles import (
    PROFILE_AGGREGATION,
    score_airfrans_velocity_profiles,
)
from reference.metrics import r2_score


STATIONS = [
    "upper_x_0_25c",
    "upper_x_0_50c",
    "upper_x_0_75c",
    "upper_x_0_95c",
]
QUANTITIES = ["velocity_x_ratio", "velocity_y_ratio"]


def profile_document(*, transverse_prediction: list[float] | None = None) -> dict:
    coordinate = [0.0, 0.05, 0.1]
    streamwise_truth = [0.0, 1.0, 2.0]
    transverse_truth = [0.0, 0.01, 0.02]
    return {
        "schema_version": "1.0",
        "cases": [
            {
                "case_id": "case-1",
                "series": [
                    {
                        "panel_id": "velocity_profiles",
                        "station_id": station_id,
                        "quantity_id": quantity_id,
                        "coordinate": coordinate,
                        "prediction": (
                            transverse_prediction
                            if quantity_id == "velocity_y_ratio" and transverse_prediction is not None
                            else (
                                transverse_truth
                                if quantity_id == "velocity_y_ratio"
                                else streamwise_truth
                            )
                        ),
                    }
                    for station_id in STATIONS
                    for quantity_id in QUANTITIES
                ],
            }
        ],
    }


class AirfransProfileScoringTests(unittest.TestCase):
    def score(self, truth: dict, prediction: dict) -> dict:
        return score_airfrans_velocity_profiles(
            [truth],
            [prediction],
            station_ids=STATIONS,
            quantity_ids=QUANTITIES,
        )

    def test_perfect_prediction_scores_one(self) -> None:
        truth = profile_document()
        result = self.score(truth, copy.deepcopy(truth))
        self.assertEqual(result["aggregation"], PROFILE_AGGREGATION)
        self.assertEqual(result["group_count"], 8)
        self.assertEqual(result["value"], 1.0)
        self.assertTrue(all(group["raw_r2"] == 1.0 for group in result["groups"]))

    def test_equal_group_score_prevents_streamwise_scale_from_hiding_transverse_failure(self) -> None:
        truth = profile_document()
        prediction = profile_document(transverse_prediction=[0.02, 0.01, 0.0])
        result = self.score(truth, prediction)

        truth_flat = [
            value
            for series in truth["cases"][0]["series"]
            for value in series["prediction"]
        ]
        prediction_flat = [
            value
            for series in prediction["cases"][0]["series"]
            for value in series["prediction"]
        ]
        self.assertGreater(r2_score(truth_flat, prediction_flat), 0.999)
        self.assertEqual(result["value"], 0.5)
        self.assertTrue(
            all(
                group["bounded_r2"] == (1.0 if group["quantity_id"] == "velocity_x_ratio" else 0.0)
                for group in result["groups"]
            )
        )

    def test_case_coverage_and_coordinates_must_match(self) -> None:
        truth = profile_document()
        prediction = copy.deepcopy(truth)
        prediction["cases"][0]["case_id"] = "different-case"
        with self.assertRaisesRegex(ValueError, "case coverage"):
            self.score(truth, prediction)

        prediction = copy.deepcopy(truth)
        prediction["cases"][0]["series"][0]["coordinate"][-1] = 0.2
        with self.assertRaisesRegex(ValueError, "coordinates do not align"):
            self.score(truth, prediction)

    def test_official_split_coverage_fails_closed_unless_explicitly_partial(self) -> None:
        truth = profile_document()
        prediction = copy.deepcopy(truth)
        with self.assertRaisesRegex(ValueError, "selected official split"):
            score_airfrans_velocity_profiles(
                [truth],
                [prediction],
                station_ids=STATIONS,
                quantity_ids=QUANTITIES,
                expected_case_ids=["case-1", "case-2"],
            )
        result = score_airfrans_velocity_profiles(
            [truth],
            [prediction],
            station_ids=STATIONS,
            quantity_ids=QUANTITIES,
            expected_case_ids=["case-1", "case-2"],
            allow_partial_case_coverage=True,
        )
        self.assertEqual(result["case_coverage"], "partial_calibration_only")
        self.assertEqual(result["expected_case_count"], 2)


if __name__ == "__main__":
    unittest.main()
