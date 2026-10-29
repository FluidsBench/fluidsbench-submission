from __future__ import annotations

import unittest

from reference.weightings import PHYSICAL_WEIGHTINGS, evaluator_weighting


class WeightingVocabularyTests(unittest.TestCase):
    def test_all_published_physical_weightings_use_support_weights(self) -> None:
        expected = {
            "surface_face_area",
            "cell_volume",
            "boundary_line_length",
            "interior_cell_area",
            "surface_point_dual_area",
            "volume_point_dual_volume",
            "boundary_point_dual_length",
            "interior_point_dual_area",
        }
        self.assertEqual(PHYSICAL_WEIGHTINGS, expected)
        for weighting in expected:
            with self.subTest(weighting=weighting):
                self.assertEqual(evaluator_weighting(weighting), "support_weights")

    def test_nonphysical_weighting_uses_uniform_evaluator_weights(self) -> None:
        self.assertEqual(evaluator_weighting("points_equal_within_case_cases_equal"), "uniform")


if __name__ == "__main__":
    unittest.main()
