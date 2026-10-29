"""Known answers for the existing pressure reductions; no new scoring path."""

from __future__ import annotations

import unittest

import numpy as np

from reference.evaluate_predictions import (
    relative_l2_from_sufficient_statistics,
    relative_l2_sufficient_statistics,
)
from reference.metrics import aggregate_case_metrics, field_rrmse, relative_l1, relative_l2
from reference.scoring_support import ScoringSupportError


def statistics(truth, prediction, weights=None):
    truth = np.asarray(truth, dtype=float).reshape(-1, 1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1, 1)
    return relative_l2_sufficient_statistics(
        truth, prediction,
        np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=float),
        weighting="uniform" if weights is None else "support_weights",
    )


class PressureReferenceInvariants(unittest.TestCase):
    def test_kinematic_to_cp_scaling_preserves_relative_errors(self):
        # DrivAerNet++: the recorded kinematic dynamic head is 450 m²/s².
        truth, prediction = np.array([-450.0, 900.0]), np.array([-405.0, 810.0])
        for scale in (1.0, 1 / 450, 1.184):
            for metric in (relative_l1, relative_l2):
                self.assertAlmostEqual(metric(truth * scale, prediction * scale, [2, 1]), 10.0)
            self.assertAlmostEqual(relative_l2_from_sufficient_statistics([statistics(truth * scale, prediction * scale, [2, 1])]), 10.0)

    def test_prescribed_offset_changes_the_relative_denominator(self):
        truth = np.array([100010.0, 100020.0])
        prediction = np.array([100011.0, 100022.0])
        cp_truth, cp_prediction = (truth - 100000) / 10, (prediction - 100000) / 10
        self.assertAlmostEqual(relative_l2_from_sufficient_statistics([statistics(cp_truth, cp_prediction)]), 10.0)
        self.assertLess(relative_l2_from_sufficient_statistics([statistics(truth, prediction)]), 0.002)

    def test_model_pressure_bias_is_not_fitted_away(self):
        expected = 100 * np.sqrt(2 / 5)
        self.assertAlmostEqual(relative_l2_from_sufficient_statistics([statistics([1, 2], [2, 3])]), expected)

    def test_chunk_closure_and_equal_case_mean(self):
        chunks = [statistics([1], [2]), statistics([3], [3])]
        self.assertAlmostEqual(relative_l2_from_sufficient_statistics(chunks), 100 / np.sqrt(10))
        # Case scores 100% and 0%, independent of their relative field magnitudes.
        self.assertAlmostEqual(aggregate_case_metrics([([1], [2], None), ([100], [100], None)], relative_l2), 50.0)

    def test_zero_norm_is_invalid_and_tiny_positive_norm_is_not_floored(self):
        for prediction in ([0, 0], [1, 1]):
            with self.assertRaises(ScoringSupportError):
                relative_l2_from_sufficient_statistics([statistics([0, 0], prediction)])
        self.assertAlmostEqual(relative_l2_from_sufficient_statistics([statistics([1e-100], [2e-100])]), 100.0)

    def test_nonfinite_statistics_are_invalid(self):
        valid = statistics([1], [2])
        for key in ("numerator", "denominator"):
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ScoringSupportError):
                    relative_l2_from_sufficient_statistics([{**valid, key: value}])

    def test_rotor_field_rrmse_keeps_its_distinct_denominator(self):
        truth = np.array([[100.0, 200.0], [200.0, 400.0]])
        prediction = truth * 1.1
        self.assertAlmostEqual(field_rrmse(truth, prediction), np.sqrt(0.00625))
        self.assertAlmostEqual(relative_l2(truth[0], prediction[0]), 10.0)
        with self.assertRaises(ValueError):
            field_rrmse([[0, 0]], [[0, 0]])


if __name__ == "__main__":
    unittest.main()
