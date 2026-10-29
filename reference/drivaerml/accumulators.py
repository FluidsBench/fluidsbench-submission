"""Chunk-safe DrivAerML field statistics in raw native-cell order.

The DrivAerML contract reduces every complete case before averaging cases.  A
case may be supplied in contiguous raw-cell chunks, in any chunk arrival order,
but the chunks must form the exact zero-based native-cell interval without gaps
or overlap.  Only additive binary64 sums are retained, so field arrays are not
kept after :meth:`StreamingFieldAccumulator.add_chunk` returns.

For vector fields, MAE and RMSE use the Euclidean norm of each entity's
component-wise error.  Relative L2 uses the corresponding squared Euclidean
norm.  This differs deliberately from flattening components into independent
scalar samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class DrivAerAccumulatorError(ValueError):
    """Raised when chunks or metric inputs violate the native-cell contract."""


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DrivAerAccumulatorError(f"{label} must be a positive integer")
    if value > np.iinfo(np.int64).max:
        raise DrivAerAccumulatorError(f"{label} exceeds the supported raw-ID range")
    return value


def _finite_non_negative(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise DrivAerAccumulatorError(f"{label} must be finite and non-negative")


@dataclass(frozen=True)
class AdditiveFieldSums:
    """Additive sums for one weighting of one complete or partial field."""

    absolute_error: float
    squared_error: float
    squared_truth: float
    entity_count: int
    total_weight: float

    def __post_init__(self) -> None:
        _finite_non_negative(self.absolute_error, "absolute_error")
        _finite_non_negative(self.squared_error, "squared_error")
        _finite_non_negative(self.squared_truth, "squared_truth")
        _positive_integer(self.entity_count, "entity_count")
        if not math.isfinite(self.total_weight) or self.total_weight <= 0.0:
            raise DrivAerAccumulatorError("total_weight must be finite and positive")

    def relative_l2_percent(self) -> float:
        """Return ``100 * sqrt(squared_error / squared_truth)``."""

        if self.squared_truth <= 0.0:
            raise DrivAerAccumulatorError(
                "relative L2 is undefined when the ground-truth norm is zero"
            )
        value = 100.0 * math.sqrt(self.squared_error / self.squared_truth)
        if not math.isfinite(value):
            raise DrivAerAccumulatorError("relative L2 result is non-finite")
        return value

    def mae(self) -> float:
        """Return the weighted mean of per-entity Euclidean error norms."""

        value = self.absolute_error / self.total_weight
        if not math.isfinite(value):
            raise DrivAerAccumulatorError("MAE result is non-finite")
        return value

    def rmse(self) -> float:
        """Return the weighted RMS of per-entity Euclidean error norms."""

        value = math.sqrt(self.squared_error / self.total_weight)
        if not math.isfinite(value):
            raise DrivAerAccumulatorError("RMSE result is non-finite")
        return value

    def relative_l2_evidence(
        self,
        *,
        weighting: str,
        dataset_weighting: str,
    ) -> dict[str, float | int | str]:
        """Return schema-v3 relative-L2 sufficient-statistic evidence."""

        if weighting not in {"uniform", "support_weights"}:
            raise DrivAerAccumulatorError(
                "weighting must be 'uniform' or 'support_weights'"
            )
        if not isinstance(dataset_weighting, str) or not dataset_weighting:
            raise DrivAerAccumulatorError("dataset_weighting must be a non-empty string")
        self.relative_l2_percent()
        return {
            "reduction": "relative_l2_percent",
            "weighting": weighting,
            "dataset_weighting": dataset_weighting,
            "numerator": self.squared_error,
            "denominator": self.squared_truth,
            "entity_count": self.entity_count,
            "total_weight": self.total_weight,
        }


@dataclass(frozen=True)
class FieldChunkStatistics:
    """Additive statistics and raw-ID interval for one contiguous chunk."""

    raw_id_start: int
    raw_id_stop: int
    component_count: int
    uniform: AdditiveFieldSums
    physical: AdditiveFieldSums

    def __post_init__(self) -> None:
        if (
            not isinstance(self.raw_id_start, int)
            or isinstance(self.raw_id_start, bool)
            or self.raw_id_start < 0
        ):
            raise DrivAerAccumulatorError("raw_id_start must be a non-negative integer")
        if (
            not isinstance(self.raw_id_stop, int)
            or isinstance(self.raw_id_stop, bool)
            or self.raw_id_stop <= self.raw_id_start
        ):
            raise DrivAerAccumulatorError("raw_id_stop must be greater than raw_id_start")
        _positive_integer(self.component_count, "component_count")
        interval_count = self.raw_id_stop - self.raw_id_start
        if self.uniform.entity_count != interval_count:
            raise DrivAerAccumulatorError(
                "uniform entity_count does not match the raw-ID interval"
            )
        if self.physical.entity_count != interval_count:
            raise DrivAerAccumulatorError(
                "physical entity_count does not match the raw-ID interval"
            )
        if self.uniform.total_weight != float(interval_count):
            raise DrivAerAccumulatorError(
                "uniform total_weight must equal the raw-ID interval length"
            )


@dataclass(frozen=True)
class FinalizedFieldStatistics:
    """Complete-case sums after deterministic raw-ID-ordered reduction."""

    entity_count: int
    component_count: int
    uniform: AdditiveFieldSums
    physical: AdditiveFieldSums

    def metric_values(self) -> dict[str, dict[str, float]]:
        """Return relative-L2, MAE, and RMSE for both prescribed weightings."""

        return {
            "uniform": {
                "relative_l2_percent": self.uniform.relative_l2_percent(),
                "mae": self.uniform.mae(),
                "rmse": self.uniform.rmse(),
            },
            "physical": {
                "relative_l2_percent": self.physical.relative_l2_percent(),
                "mae": self.physical.mae(),
                "rmse": self.physical.rmse(),
            },
        }


def _raw_id_interval(raw_ids: Any) -> tuple[np.ndarray, int, int]:
    values = np.asarray(raw_ids)
    if values.ndim != 1 or len(values) == 0:
        raise DrivAerAccumulatorError(
            "raw_ids must be a non-empty one-dimensional array"
        )
    if values.dtype.kind not in {"i", "u"}:
        raise DrivAerAccumulatorError("raw_ids must use an integer dtype")
    if values.dtype.kind == "i" and np.any(values < 0):
        raise DrivAerAccumulatorError("raw_ids cannot be negative")
    if np.any(values > np.iinfo(np.int64).max):
        raise DrivAerAccumulatorError("raw_ids exceed the supported int64 range")
    normalized = values.astype(np.int64, copy=False)
    start = int(normalized[0])
    stop = start + len(normalized)
    # ``expected_entity_count`` is itself limited to int64 max, so the largest
    # usable zero-based raw ID is max-1 and the exclusive stop is max.
    if stop > np.iinfo(np.int64).max:
        raise DrivAerAccumulatorError("raw_ids exceed the supported int64 range")
    expected = np.arange(start, stop, dtype=np.int64)
    if not np.array_equal(normalized, expected):
        raise DrivAerAccumulatorError(
            "raw_ids within a chunk must be strictly increasing and contiguous"
        )
    return normalized, start, stop


def _field_array(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DrivAerAccumulatorError(f"{label} must be numeric") from error
    if array.ndim not in {1, 2}:
        raise DrivAerAccumulatorError(
            f"{label} must have shape [entity] or [entity, component]"
        )
    if array.shape[0] == 0 or (array.ndim == 2 and array.shape[1] == 0):
        raise DrivAerAccumulatorError(f"{label} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise DrivAerAccumulatorError(f"{label} must contain only finite values")
    return array


def _physical_weights(value: Any, expected_count: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DrivAerAccumulatorError("physical_weights must be numeric") from error
    if weights.ndim != 1 or weights.shape != (expected_count,):
        raise DrivAerAccumulatorError(
            "physical_weights must contain one value per spatial entity"
        )
    if not np.all(np.isfinite(weights)):
        raise DrivAerAccumulatorError(
            "physical_weights must contain only finite values"
        )
    if np.any(weights <= 0.0):
        raise DrivAerAccumulatorError("physical_weights must be strictly positive")
    return weights


def _sum(values: np.ndarray, label: str) -> float:
    value = float(np.sum(values, dtype=np.float64))
    if not math.isfinite(value) or value < 0.0:
        raise DrivAerAccumulatorError(f"{label} overflowed or became non-finite")
    return value


def field_chunk_statistics(
    raw_ids: Any,
    truth: Any,
    prediction: Any,
    physical_weights: Any,
) -> FieldChunkStatistics:
    """Validate one chunk and calculate its uniform and physical sums.

    ``truth`` and ``prediction`` must have identical scalar ``[entity]`` or
    vector ``[entity, component]`` shapes.  Physical weights are required and
    must contain one strictly positive finite value per entity.
    """

    normalized_ids, raw_id_start, raw_id_stop = _raw_id_interval(raw_ids)
    truth_array = _field_array(truth, "truth")
    prediction_array = _field_array(prediction, "prediction")
    if truth_array.shape != prediction_array.shape:
        raise DrivAerAccumulatorError(
            "truth and prediction must have identical entity/component shapes"
        )
    if truth_array.ndim == 1:
        truth_array = truth_array[:, None]
        prediction_array = prediction_array[:, None]
    entity_count, component_count = truth_array.shape
    if len(normalized_ids) != entity_count:
        raise DrivAerAccumulatorError(
            "raw_ids, truth, and prediction must contain the same entity count"
        )
    weights = _physical_weights(physical_weights, entity_count)

    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction_array - truth_array
        squared_error_norm = np.einsum("ij,ij->i", error, error, optimize=False)
        squared_truth_norm = np.einsum(
            "ij,ij->i", truth_array, truth_array, optimize=False
        )
        absolute_error_norm = np.sqrt(squared_error_norm)
        physical_absolute_error = weights * absolute_error_norm
        physical_squared_error = weights * squared_error_norm
        physical_squared_truth = weights * squared_truth_norm

    contributions = (
        squared_error_norm,
        squared_truth_norm,
        absolute_error_norm,
        physical_absolute_error,
        physical_squared_error,
        physical_squared_truth,
    )
    if any(not np.all(np.isfinite(value)) for value in contributions):
        raise DrivAerAccumulatorError(
            "field arithmetic overflowed or produced non-finite contributions"
        )

    uniform = AdditiveFieldSums(
        absolute_error=_sum(absolute_error_norm, "uniform absolute_error"),
        squared_error=_sum(squared_error_norm, "uniform squared_error"),
        squared_truth=_sum(squared_truth_norm, "uniform squared_truth"),
        entity_count=entity_count,
        total_weight=float(entity_count),
    )
    physical = AdditiveFieldSums(
        absolute_error=_sum(physical_absolute_error, "physical absolute_error"),
        squared_error=_sum(physical_squared_error, "physical squared_error"),
        squared_truth=_sum(physical_squared_truth, "physical squared_truth"),
        entity_count=entity_count,
        total_weight=_sum(weights, "physical total_weight"),
    )
    return FieldChunkStatistics(
        raw_id_start=raw_id_start,
        raw_id_stop=raw_id_stop,
        component_count=component_count,
        uniform=uniform,
        physical=physical,
    )


def _merged_sums(
    chunks: list[FieldChunkStatistics],
    attribute: str,
) -> AdditiveFieldSums:
    values = [getattr(chunk, attribute) for chunk in chunks]
    try:
        absolute_error = math.fsum(item.absolute_error for item in values)
        squared_error = math.fsum(item.squared_error for item in values)
        squared_truth = math.fsum(item.squared_truth for item in values)
        total_weight = math.fsum(item.total_weight for item in values)
    except OverflowError as error:
        raise DrivAerAccumulatorError(
            f"{attribute} sufficient-statistic merge overflowed"
        ) from error
    return AdditiveFieldSums(
        absolute_error=absolute_error,
        squared_error=squared_error,
        squared_truth=squared_truth,
        entity_count=sum(item.entity_count for item in values),
        total_weight=total_weight,
    )


class StreamingFieldAccumulator:
    """Collect additive chunk statistics and finalize in canonical raw-ID order."""

    def __init__(
        self,
        expected_entity_count: int,
        *,
        component_count: int | None = None,
    ) -> None:
        self.expected_entity_count = _positive_integer(
            expected_entity_count, "expected_entity_count"
        )
        self.component_count = (
            None
            if component_count is None
            else _positive_integer(component_count, "component_count")
        )
        self._chunks: list[FieldChunkStatistics] = []

    @property
    def chunks(self) -> tuple[FieldChunkStatistics, ...]:
        """Return the retained additive summaries, never the field arrays."""

        return tuple(self._chunks)

    def add_chunk(
        self,
        raw_ids: Any,
        truth: Any,
        prediction: Any,
        physical_weights: Any,
    ) -> FieldChunkStatistics:
        """Validate and reduce one chunk, then discard its field arrays."""

        statistics = field_chunk_statistics(
            raw_ids,
            truth,
            prediction,
            physical_weights,
        )
        self.add_statistics(statistics)
        return statistics

    def add_statistics(self, statistics: FieldChunkStatistics) -> None:
        """Add precomputed additive statistics for later deterministic merging."""

        if not isinstance(statistics, FieldChunkStatistics):
            raise DrivAerAccumulatorError(
                "statistics must be a FieldChunkStatistics instance"
            )
        if statistics.raw_id_stop > self.expected_entity_count:
            raise DrivAerAccumulatorError(
                "chunk raw IDs exceed expected_entity_count"
            )
        if self.component_count is None:
            self.component_count = statistics.component_count
        elif statistics.component_count != self.component_count:
            raise DrivAerAccumulatorError(
                "all chunks must use the same field component count"
            )
        self._chunks.append(statistics)

    def merge(self, other: "StreamingFieldAccumulator") -> None:
        """Merge partial accumulators; finalization canonically sorts intervals."""

        if not isinstance(other, StreamingFieldAccumulator):
            raise DrivAerAccumulatorError(
                "other must be a StreamingFieldAccumulator"
            )
        if other is self:
            raise DrivAerAccumulatorError("an accumulator cannot be merged with itself")
        if other.expected_entity_count != self.expected_entity_count:
            raise DrivAerAccumulatorError(
                "merged accumulators must have the same expected_entity_count"
            )
        for statistics in other.chunks:
            self.add_statistics(statistics)

    def finalize(self) -> FinalizedFieldStatistics:
        """Validate exact coverage and combine sums in canonical raw-ID order."""

        if not self._chunks:
            raise DrivAerAccumulatorError("cannot finalize an empty accumulator")
        ordered = sorted(
            self._chunks,
            key=lambda chunk: (chunk.raw_id_start, chunk.raw_id_stop),
        )
        cursor = 0
        for chunk in ordered:
            if chunk.raw_id_start < cursor:
                raise DrivAerAccumulatorError(
                    "chunks contain duplicate or overlapping raw IDs"
                )
            if chunk.raw_id_start > cursor:
                raise DrivAerAccumulatorError(
                    f"chunks leave a raw-ID gap [{cursor}, {chunk.raw_id_start})"
                )
            cursor = chunk.raw_id_stop
        if cursor != self.expected_entity_count:
            raise DrivAerAccumulatorError(
                f"chunks leave a raw-ID gap [{cursor}, {self.expected_entity_count})"
            )
        if self.component_count is None:  # Defensive; non-empty chunks set it.
            raise DrivAerAccumulatorError("field component count is undefined")

        uniform = _merged_sums(ordered, "uniform")
        physical = _merged_sums(ordered, "physical")
        if uniform.entity_count != self.expected_entity_count:
            raise DrivAerAccumulatorError(
                "merged uniform entity_count differs from expected_entity_count"
            )
        if physical.entity_count != self.expected_entity_count:
            raise DrivAerAccumulatorError(
                "merged physical entity_count differs from expected_entity_count"
            )
        return FinalizedFieldStatistics(
            entity_count=self.expected_entity_count,
            component_count=self.component_count,
            uniform=uniform,
            physical=physical,
        )
