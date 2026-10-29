"""Score one DrivAerNet++ case: native-order predictions against frozen support.

A prediction is ``cases/<case_id>/prediction.npz`` holding exactly one array,
``p``: float32 or float64, shape ``(N,)``, finite, in native PointData order of
``SurfacePressureVTK/<case_id>.vtk`` (kinematic gauge pressure, m^2/s^2).
Every native point is scored exactly once; nothing is interpolated by the
evaluator.  Profiles are read off the same predictions at frozen native
points, so they need no separate participant artifact.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    DATASET_ID,
    RELEASE_ID,
    SUPPORT_ID,
    ContractError,
    sha256_file,
)
from .distribution import CaseSupport
from .native import DYNAMIC_HEAD, PROFILE_STATIONS
from .support import TRUTH_ARRAY, WEIGHT_ARRAY, station_array_names

CASE_EVIDENCE_SCHEMA = "fluidsbench-drivaernetplusplus-case-evidence-v1"
PREDICTION_FILE = "prediction.npz"
PREDICTION_ARRAY = "p"
MAX_PREDICTION_BYTES = 64 * 1024**2
ACCEPTED_DTYPES = ("<f4", "<f8")


class PredictionError(ContractError):
    """Raised when a submitted prediction is incomplete or malformed."""


def load_prediction(path: Path, expected_count: int) -> np.ndarray:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PredictionError(f"missing prediction file: {path}")
    if path.stat().st_size > MAX_PREDICTION_BYTES:
        raise PredictionError(f"prediction file is larger than {MAX_PREDICTION_BYTES} bytes: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist()]
        with np.load(path, allow_pickle=False) as loaded:
            files = list(loaded.files)
            value = np.array(loaded[PREDICTION_ARRAY]) if files == [PREDICTION_ARRAY] else None
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise PredictionError(f"cannot read {path}: {error}") from error
    if names != [f"{PREDICTION_ARRAY}.npy"] or value is None:
        raise PredictionError(f"{path} must contain exactly one array named {PREDICTION_ARRAY!r}")
    if value.dtype.str not in ACCEPTED_DTYPES:
        raise PredictionError(f"{path}: p must be little-endian float32 or float64, not {value.dtype}")
    if value.shape != (expected_count,):
        raise PredictionError(
            f"{path}: p has shape {value.shape}; expected ({expected_count},), one value per native point"
        )
    if not np.all(np.isfinite(value)):
        raise PredictionError(f"{path}: p contains non-finite values")
    return value.astype(np.float64)


def _relative_l2(numerator: float, denominator: float) -> float:
    return 100.0 * math.sqrt(numerator / denominator)


def score_case(support: CaseSupport, prediction_path: Path) -> dict[str, Any]:
    """Return the exact per-case evidence record."""

    n = support.point_count
    prediction = load_prediction(prediction_path, n)
    truth = support.arrays[TRUTH_ARRAY].astype(np.float64)
    weights = support.arrays[WEIGHT_ARRAY]
    error = prediction - truth
    squared_error = error * error
    squared_truth = truth * truth
    weighted = {
        "numerator": float(np.sum(weights * squared_error)),
        "denominator": float(np.sum(weights * squared_truth)),
    }
    uniform = {
        "numerator": float(np.sum(squared_error)),
        "denominator": float(np.sum(squared_truth)),
    }
    l1_numerator = float(np.sum(weights * np.abs(error)))
    l1_denominator = float(np.sum(weights * np.abs(truth)))
    total_weight = float(np.sum(weights))
    if weighted["denominator"] <= 0 or uniform["denominator"] <= 0 or l1_denominator <= 0:
        raise PredictionError(f"{support.case_id}: reference pressure is identically zero")
    truth_mean = float(np.mean(truth))
    centred = truth - truth_mean
    profiles = []
    for station in PROFILE_STATIONS:
        index_name, coordinate_name = station_array_names(station)
        index = support.arrays[index_name]
        profiles.append(
            {
                "station_id": station,
                "coordinate_id": support.record["profile_stations"][station]["coordinate_id"],
                "coordinate": support.arrays[coordinate_name].tolist(),
                "truth_cp": (truth[index] / DYNAMIC_HEAD).tolist(),
                "prediction_cp": (prediction[index] / DYNAMIC_HEAD).tolist(),
            }
        )
    return {
        "schema": CASE_EVIDENCE_SCHEMA,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "case_id": support.case_id,
        "support": {
            "support_id": SUPPORT_ID,
            "entity_count": n,
            "case_support_sha256": support.record_sha256,
            "total_weight": total_weight,
        },
        "prediction": {
            "file": PREDICTION_FILE,
            "sha256": sha256_file(prediction_path),
            "entity_count": n,
        },
        "metric_values": {
            "surface_pressure_rel_l2": _relative_l2(weighted["numerator"], weighted["denominator"]),
            "surface_pressure_equal_entity_rel_l2": _relative_l2(uniform["numerator"], uniform["denominator"]),
            "surface_pressure_rel_l1": 100.0 * l1_numerator / l1_denominator,
        },
        "metric_sufficient_statistics": {
            "surface_pressure_rel_l2": {
                "reduction": "relative_l2_percent",
                "weighting": "support_weights",
                "dataset_weighting": "surface_point_dual_area",
                **weighted,
                "entity_count": n,
                "total_weight": total_weight,
            },
            "surface_pressure_equal_entity_rel_l2": {
                "reduction": "relative_l2_percent",
                "weighting": "uniform",
                "dataset_weighting": "surface_points_equal",
                **uniform,
                "entity_count": n,
                "total_weight": float(n),
            },
        },
        "relative_l1_statistics": {"numerator": l1_numerator, "denominator": l1_denominator},
        # Chan-combinable statistics for the flattened, equal-point R^2.
        "r2_statistics": {
            "count": n,
            "sum_squared_error": uniform["numerator"],
            "truth_mean": truth_mean,
            "truth_centred_sum_squares": float(np.sum(centred * centred)),
        },
        "profiles": profiles,
    }
