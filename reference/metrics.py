"""Array-level reference implementations of the FluidsBench metrics.

Participants remain responsible for loading and dimensionalising their own data.
These functions define only the numerical reductions once aligned arrays are ready.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike


def _arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    weights: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {truth.shape} and {prediction.shape}")
    if truth.size == 0:
        raise ValueError("metric arrays cannot be empty")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metric arrays must contain only finite values")

    if weights is None:
        weight = np.ones_like(truth, dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        try:
            weight = np.broadcast_to(weight, truth.shape)
        except ValueError as error:
            raise ValueError(f"weights cannot be broadcast to data shape {truth.shape}") from error
        if not np.all(np.isfinite(weight)):
            raise ValueError("weights must contain only finite values")
        if np.any(weight < 0):
            raise ValueError("weights cannot be negative")
    if float(np.sum(weight)) <= 0:
        raise ValueError("weights must have a positive sum")
    return truth, prediction, weight


def weighted_mae(y_true: ArrayLike, y_pred: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Return sum(w * abs(y_pred - y_true)) / sum(w)."""

    truth, prediction, weight = _arrays(y_true, y_pred, weights)
    return float(np.sum(weight * np.abs(prediction - truth)) / np.sum(weight))


def weighted_mse(y_true: ArrayLike, y_pred: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Return sum(w * (y_pred - y_true)^2) / sum(w)."""

    truth, prediction, weight = _arrays(y_true, y_pred, weights)
    return float(np.sum(weight * np.square(prediction - truth)) / np.sum(weight))


def weighted_rmse(y_true: ArrayLike, y_pred: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Return sqrt(sum(w * (y_pred - y_true)^2) / sum(w))."""

    truth, prediction, weight = _arrays(y_true, y_pred, weights)
    return float(np.sqrt(np.sum(weight * np.square(prediction - truth)) / np.sum(weight)))


def relative_l1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    percentage: bool = True,
) -> float:
    """Return sum(w * abs(error)) / sum(w * abs(y_true))."""

    truth, prediction, weight = _arrays(y_true, y_pred, weights)
    denominator = float(np.sum(weight * np.abs(truth)))
    if denominator == 0:
        raise ValueError("relative L1 is undefined when the weighted ground-truth norm is zero")
    result = float(np.sum(weight * np.abs(prediction - truth)) / denominator)
    return 100.0 * result if percentage else result


def relative_l2(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    percentage: bool = True,
) -> float:
    """Return sqrt(sum(w * error^2)) / sqrt(sum(w * y_true^2))."""

    truth, prediction, weight = _arrays(y_true, y_pred, weights)
    denominator_squared = float(np.sum(weight * np.square(truth)))
    if denominator_squared == 0:
        raise ValueError("relative L2 is undefined when the weighted ground-truth norm is zero")
    result = float(
        np.sqrt(np.sum(weight * np.square(prediction - truth))) / np.sqrt(denominator_squared)
    )
    return 100.0 * result if percentage else result


def r2_score(y_true: ArrayLike, y_pred: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Return weighted R2 using the weighted ground-truth mean.

    R2 is undefined for constant ground truth and raises ValueError rather than
    silently returning zero or NaN.
    """

    truth, prediction, weight = _arrays(y_true, y_pred, weights)
    mean = float(np.sum(weight * truth) / np.sum(weight))
    denominator = float(np.sum(weight * np.square(truth - mean)))
    if denominator == 0:
        raise ValueError("R2 is undefined for constant ground truth")
    numerator = float(np.sum(weight * np.square(prediction - truth)))
    return 1.0 - numerator / denominator


def field_rrmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return sqrt(mean_i((||error_i||_2^2 / N_i) / ||truth_i||_inf^2)).

    Axis zero indexes cases. Remaining dimensions contain a case's field values.
    All cases must use a common rectangular array for this convenience function.
    Use the documented equation directly for ragged fields.
    """

    truth, prediction, _ = _arrays(y_true, y_pred, None)
    if truth.ndim < 2:
        raise ValueError("field RRMSE expects an array with a leading case axis")
    flattened_truth = truth.reshape(truth.shape[0], -1)
    flattened_prediction = prediction.reshape(prediction.shape[0], -1)
    scale = np.max(np.abs(flattened_truth), axis=1)
    if np.any(scale == 0):
        raise ValueError("field RRMSE is undefined for a case with zero infinity norm")
    case_mse = np.mean(np.square(flattened_prediction - flattened_truth), axis=1)
    return float(np.sqrt(np.mean(case_mse / np.square(scale))))


def scalar_rrmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return sqrt(mean_i((y_pred_i - y_true_i)^2 / abs(y_true_i)^2))."""

    truth, prediction, _ = _arrays(y_true, y_pred, None)
    if np.any(truth == 0):
        raise ValueError("scalar RRMSE is undefined when a ground-truth value is zero")
    return float(np.sqrt(np.mean(np.square(prediction - truth) / np.square(np.abs(truth)))))


def aggregate_case_metrics(
    cases: Iterable[tuple[ArrayLike, ArrayLike, ArrayLike | None]],
    metric: Callable[[ArrayLike, ArrayLike, ArrayLike | None], float],
) -> float:
    """Macro-average a metric so every test geometry has equal weight."""

    values = [metric(y_true, y_pred, weights) for y_true, y_pred, weights in cases]
    if not values:
        raise ValueError("at least one test geometry is required")
    return float(np.mean(values))


def arc_length_rmse(
    coordinate: ArrayLike,
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Return trapezoid-integrated RMSE along a strictly ordered profile.

    This is the DrivAerML AutoCFD5 velocity-profile reduction. Every edge is
    weighted by its arc-length increment and the squared endpoint errors are
    integrated with the trapezoid rule.
    """

    distance = np.asarray(coordinate, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if distance.ndim != 1 or truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("profile coordinate, truth, and prediction must be one-dimensional")
    if len(distance) < 2 or truth.shape != distance.shape or prediction.shape != distance.shape:
        raise ValueError("profile arrays must have matching lengths of at least two")
    if not (
        np.all(np.isfinite(distance))
        and np.all(np.isfinite(truth))
        and np.all(np.isfinite(prediction))
    ):
        raise ValueError("profile arrays must contain only finite values")
    increments = np.diff(distance)
    if np.any(increments <= 0.0):
        raise ValueError("profile coordinates must be strictly increasing")
    squared_error = np.square(prediction - truth)
    integral = np.sum(increments * (squared_error[:-1] + squared_error[1:]) / 2.0)
    return float(np.sqrt(integral / np.sum(increments)))


def unique_probe_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return equal-probe RMSE for one case's unique diagnostic probes."""

    return weighted_rmse(y_true, y_pred)
