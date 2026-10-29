"""Reference metric implementations for FluidsBench submissions."""

from .metrics import (
    aggregate_case_metrics,
    field_rrmse,
    r2_score,
    relative_l1,
    relative_l2,
    scalar_rrmse,
    weighted_mae,
    weighted_mse,
    weighted_rmse,
)
from .scores import arithmetic_mean, legacy_aero_scores

__all__ = [
    "aggregate_case_metrics",
    "arithmetic_mean",
    "field_rrmse",
    "legacy_aero_scores",
    "r2_score",
    "relative_l1",
    "relative_l2",
    "scalar_rrmse",
    "weighted_mae",
    "weighted_mse",
    "weighted_rmse",
]
