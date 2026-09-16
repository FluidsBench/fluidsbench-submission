from __future__ import annotations

import math
import unittest

import numpy as np

from reference.drivaerml.accumulators import (
    DrivAerAccumulatorError,
    StreamingFieldAccumulator,
    field_chunk_statistics,
)


class DrivAerMLAccumulatorTests(unittest.TestCase):
    def assert_sums_close(self, left: object, right: object) -> None:
        for name in ("absolute_error", "squared_error", "squared_truth", "total_weight"):
            self.assertTrue(
                math.isclose(
                    getattr(left, name),
                    getattr(right, name),
                    rel_tol=2e-15,
                    abs_tol=1e-15,
                ),
                msg=f"{name}: {getattr(left, name)!r} != {getattr(right, name)!r}",
            )
        self.assertEqual(left.entity_count, right.entity_count)

    def test_full_case_matches_arbitrary_chunks_and_merge_order(self) -> None:
        rng = np.random.default_rng(20260820)
        count = 97
        truth = rng.normal(size=(count, 3)) + np.asarray([2.0, -1.0, 0.5])
        prediction = truth + rng.normal(scale=0.2, size=(count, 3))
        weights = rng.lognormal(mean=-1.0, sigma=0.7, size=count)
        raw_ids = np.arange(count, dtype=np.int64)

        whole = StreamingFieldAccumulator(count, component_count=3)
        whole.add_chunk(raw_ids, truth, prediction, weights)
        whole_result = whole.finalize()

        boundaries = [0, 1, 8, 9, 31, 56, 57, 80, 97]
        chunks = [
            field_chunk_statistics(
                raw_ids[start:stop],
                truth[start:stop],
                prediction[start:stop],
                weights[start:stop],
            )
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
        partitioned = StreamingFieldAccumulator(count)
        for index in [5, 0, 7, 3, 1, 6, 2, 4]:
            partitioned.add_statistics(chunks[index])
        partitioned_result = partitioned.finalize()

        for weighting in ("uniform", "physical"):
            self.assert_sums_close(
                getattr(whole_result, weighting),
                getattr(partitioned_result, weighting),
            )
            for metric in ("relative_l2_percent", "mae", "rmse"):
                self.assertTrue(
                    math.isclose(
                        getattr(getattr(whole_result, weighting), metric)(),
                        getattr(getattr(partitioned_result, weighting), metric)(),
                        rel_tol=2e-15,
                        abs_tol=1e-15,
                    )
                )

        first = StreamingFieldAccumulator(count, component_count=3)
        second = StreamingFieldAccumulator(count, component_count=3)
        for chunk in chunks[::2]:
            first.add_statistics(chunk)
        for chunk in reversed(chunks[1::2]):
            second.add_statistics(chunk)
        first.merge(second)
        merged_result = first.finalize()
        self.assert_sums_close(partitioned_result.uniform, merged_result.uniform)
        self.assert_sums_close(partitioned_result.physical, merged_result.physical)

    def test_scalar_uniform_and_physical_metrics_and_evidence(self) -> None:
        accumulator = StreamingFieldAccumulator(2, component_count=1)
        accumulator.add_chunk(
            np.asarray([0, 1]),
            np.asarray([1.0, 2.0]),
            np.asarray([2.0, 2.0]),
            np.asarray([1.0, 3.0]),
        )
        result = accumulator.finalize()
        self.assertAlmostEqual(result.uniform.relative_l2_percent(), 100.0 / math.sqrt(5.0))
        self.assertAlmostEqual(result.physical.relative_l2_percent(), 100.0 / math.sqrt(13.0))
        self.assertAlmostEqual(result.uniform.mae(), 0.5)
        self.assertAlmostEqual(result.physical.mae(), 0.25)
        self.assertAlmostEqual(result.uniform.rmse(), math.sqrt(0.5))
        self.assertAlmostEqual(result.physical.rmse(), 0.5)
        self.assertEqual(
            result.uniform.relative_l2_evidence(
                weighting="uniform",
                dataset_weighting="volume_cells_equal",
            ),
            {
                "reduction": "relative_l2_percent",
                "weighting": "uniform",
                "dataset_weighting": "volume_cells_equal",
                "numerator": 1.0,
                "denominator": 5.0,
                "entity_count": 2,
                "total_weight": 2.0,
            },
        )

    def test_vector_mae_and_rmse_use_per_entity_euclidean_norm(self) -> None:
        accumulator = StreamingFieldAccumulator(1, component_count=2)
        accumulator.add_chunk(
            np.asarray([0]),
            np.asarray([[1.0, 1.0]]),
            np.asarray([[4.0, 5.0]]),
            np.asarray([7.0]),
        )
        result = accumulator.finalize()
        for sums in (result.uniform, result.physical):
            self.assertEqual(sums.mae(), 5.0)
            self.assertEqual(sums.rmse(), 5.0)
            self.assertAlmostEqual(
                sums.relative_l2_percent(),
                100.0 * 5.0 / math.sqrt(2.0),
            )

    def test_duplicate_overlap_and_gap_rejection(self) -> None:
        with self.assertRaisesRegex(DrivAerAccumulatorError, "contiguous"):
            field_chunk_statistics(
                np.asarray([0, 1, 1]),
                np.ones(3),
                np.ones(3),
                np.ones(3),
            )

        overlap = StreamingFieldAccumulator(3)
        overlap.add_chunk(np.asarray([0, 1]), np.ones(2), np.ones(2), np.ones(2))
        overlap.add_chunk(np.asarray([1, 2]), np.ones(2), np.ones(2), np.ones(2))
        with self.assertRaisesRegex(DrivAerAccumulatorError, "duplicate or overlapping"):
            overlap.finalize()

        gap = StreamingFieldAccumulator(3)
        gap.add_chunk(np.asarray([0]), np.ones(1), np.ones(1), np.ones(1))
        gap.add_chunk(np.asarray([2]), np.ones(1), np.ones(1), np.ones(1))
        with self.assertRaisesRegex(DrivAerAccumulatorError, "raw-ID gap"):
            gap.finalize()

        missing_tail = StreamingFieldAccumulator(3)
        missing_tail.add_chunk(np.asarray([0, 1]), np.ones(2), np.ones(2), np.ones(2))
        with self.assertRaisesRegex(DrivAerAccumulatorError, r"\[2, 3\)"):
            missing_tail.finalize()

    def test_shape_finite_raw_id_and_weight_failures(self) -> None:
        valid_ids = np.asarray([0, 1])
        valid_truth = np.ones((2, 3))
        valid_prediction = np.zeros((2, 3))
        valid_weights = np.ones(2)

        invalid_calls = [
            (np.asarray([0.0, 1.0]), valid_truth, valid_prediction, valid_weights, "integer dtype"),
            (valid_ids, np.ones((2, 3, 1)), valid_prediction, valid_weights, "shape"),
            (valid_ids, valid_truth, np.zeros((2, 2)), valid_weights, "identical"),
            (valid_ids, np.ones(2), np.ones((2, 1)), valid_weights, "identical"),
            (np.asarray([0]), valid_truth, valid_prediction, valid_weights, "same entity count"),
            (valid_ids, np.asarray([[np.nan, 0.0, 0.0], [1.0, 1.0, 1.0]]), valid_prediction, valid_weights, "finite"),
            (valid_ids, valid_truth, np.asarray([[np.inf, 0.0, 0.0], [1.0, 1.0, 1.0]]), valid_weights, "finite"),
            (valid_ids, valid_truth, valid_prediction, np.ones((2, 1)), "one value"),
            (valid_ids, valid_truth, valid_prediction, np.asarray([1.0, 0.0]), "strictly positive"),
            (valid_ids, valid_truth, valid_prediction, np.asarray([1.0, -1.0]), "strictly positive"),
            (valid_ids, valid_truth, valid_prediction, np.asarray([1.0, np.nan]), "finite"),
        ]
        for raw_ids, truth, prediction, weights, message in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DrivAerAccumulatorError, message):
                    field_chunk_statistics(raw_ids, truth, prediction, weights)

        out_of_range = StreamingFieldAccumulator(2)
        with self.assertRaisesRegex(DrivAerAccumulatorError, "exceed"):
            out_of_range.add_chunk(
                np.asarray([1, 2]),
                np.ones(2),
                np.ones(2),
                np.ones(2),
            )

        component_mismatch = StreamingFieldAccumulator(2)
        component_mismatch.add_chunk(
            np.asarray([0]), np.ones((1, 2)), np.ones((1, 2)), np.ones(1)
        )
        with self.assertRaisesRegex(DrivAerAccumulatorError, "same field component"):
            component_mismatch.add_chunk(
                np.asarray([1]), np.ones((1, 3)), np.ones((1, 3)), np.ones(1)
            )

        with self.assertRaisesRegex(DrivAerAccumulatorError, "overflowed"):
            field_chunk_statistics(
                valid_ids,
                np.full(2, np.finfo(np.float64).max),
                np.zeros(2),
                valid_weights,
            )

    def test_zero_truth_relative_l2_and_invalid_construction_fail(self) -> None:
        accumulator = StreamingFieldAccumulator(1)
        accumulator.add_chunk(
            np.asarray([0]), np.zeros(1), np.ones(1), np.ones(1)
        )
        result = accumulator.finalize()
        with self.assertRaisesRegex(DrivAerAccumulatorError, "ground-truth norm is zero"):
            result.uniform.relative_l2_percent()
        with self.assertRaisesRegex(DrivAerAccumulatorError, "positive integer"):
            StreamingFieldAccumulator(0)
        with self.assertRaisesRegex(DrivAerAccumulatorError, "empty accumulator"):
            StreamingFieldAccumulator(1).finalize()


if __name__ == "__main__":
    unittest.main()
