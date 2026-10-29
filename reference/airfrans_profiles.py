"""Reference reduction for the AirfRANS boundary-layer profile score."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .metrics import r2_score


PROFILE_PANEL_ID = "velocity_profiles"
PROFILE_METRIC_ID = "velocity_profile_r2"
PROFILE_AGGREGATION = "equal_station_equal_quantity_mean_bounded_r2_across_split"


def _cases_by_id(documents: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        cases = document.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"{label} profile document must contain a cases list")
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError(f"{label} profile cases must be objects")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{label} profile case_id must be a non-empty string")
            if case_id in result:
                raise ValueError(f"{label} contains duplicate case_id {case_id!r}")
            result[case_id] = case
    if not result:
        raise ValueError(f"{label} must contain at least one profile case")
    return result


def _series_by_identity(
    case: Mapping[str, Any],
    *,
    label: str,
    panel_id: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    series_entries = case.get("series")
    if not isinstance(series_entries, list):
        raise ValueError(f"{label} case must contain a series list")
    for series in series_entries:
        if not isinstance(series, Mapping):
            raise ValueError(f"{label} profile series must be objects")
        if series.get("panel_id") != panel_id:
            continue
        station_id = series.get("station_id")
        quantity_id = series.get("quantity_id")
        if not isinstance(station_id, str) or not isinstance(quantity_id, str):
            raise ValueError(f"{label} profile identities must be strings")
        identity = (station_id, quantity_id)
        if identity in result:
            raise ValueError(f"{label} contains duplicate profile series {identity!r}")
        result[identity] = series
    return result


def score_airfrans_velocity_profiles(
    ground_truth_documents: Iterable[Mapping[str, Any]],
    prediction_documents: Iterable[Mapping[str, Any]],
    *,
    station_ids: Sequence[str],
    quantity_ids: Sequence[str],
    panel_id: str = PROFILE_PANEL_ID,
    coordinate_tolerance: float = 1e-12,
    expected_case_ids: Sequence[str] | None = None,
    allow_partial_case_coverage: bool = False,
) -> dict[str, Any]:
    """Return one balanced score and its station/quantity R2 diagnostics.

    R2 is calculated independently for each station/quantity group after all
    official cases and samples for that group have been concatenated. Each raw
    group R2 is then clipped to [0, 1], and the bounded group values are averaged
    equally. Consequently, neither a velocity component nor a chord station can
    dominate merely because its values have a larger scale or a different mean.
    """

    if not station_ids or len(station_ids) != len(set(station_ids)):
        raise ValueError("station_ids must be a non-empty unique sequence")
    if not quantity_ids or len(quantity_ids) != len(set(quantity_ids)):
        raise ValueError("quantity_ids must be a non-empty unique sequence")
    if coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance cannot be negative")

    truth_cases = _cases_by_id(ground_truth_documents, label="ground truth")
    prediction_cases = _cases_by_id(prediction_documents, label="prediction")
    if set(truth_cases) != set(prediction_cases):
        missing = sorted(set(truth_cases) - set(prediction_cases))
        unexpected = sorted(set(prediction_cases) - set(truth_cases))
        raise ValueError(
            "prediction case coverage must exactly match ground truth; "
            f"missing={missing}, unexpected={unexpected}"
        )
    case_coverage = "provided_cases_only"
    expected_case_count: int | None = None
    if expected_case_ids is not None:
        expected_case_id_set = set(expected_case_ids)
        if not expected_case_ids or len(expected_case_ids) != len(expected_case_id_set):
            raise ValueError("expected_case_ids must be a non-empty unique sequence")
        observed_case_ids = set(truth_cases)
        unexpected = sorted(observed_case_ids - expected_case_id_set)
        missing = sorted(expected_case_id_set - observed_case_ids)
        if unexpected or (missing and not allow_partial_case_coverage):
            raise ValueError(
                "profile case coverage does not match the selected official split; "
                f"missing={missing}, unexpected={unexpected}"
            )
        expected_case_count = len(expected_case_ids)
        case_coverage = "complete_official_split" if not missing else "partial_calibration_only"

    expected_identities = {
        (station_id, quantity_id)
        for station_id in station_ids
        for quantity_id in quantity_ids
    }
    grouped_truth: dict[tuple[str, str], list[np.ndarray]] = {
        identity: [] for identity in expected_identities
    }
    grouped_prediction: dict[tuple[str, str], list[np.ndarray]] = {
        identity: [] for identity in expected_identities
    }

    for case_id in truth_cases:
        truth_series = _series_by_identity(
            truth_cases[case_id],
            label=f"ground truth case {case_id!r}",
            panel_id=panel_id,
        )
        prediction_series = _series_by_identity(
            prediction_cases[case_id],
            label=f"prediction case {case_id!r}",
            panel_id=panel_id,
        )
        for label, observed in (
            ("ground truth", set(truth_series)),
            ("prediction", set(prediction_series)),
        ):
            if observed != expected_identities:
                missing = sorted(expected_identities - observed)
                unexpected = sorted(observed - expected_identities)
                raise ValueError(
                    f"{label} case {case_id!r} profile coverage is incomplete; "
                    f"missing={missing}, unexpected={unexpected}"
                )

        for identity in expected_identities:
            truth_entry = truth_series[identity]
            prediction_entry = prediction_series[identity]
            truth_coordinate = np.asarray(truth_entry.get("coordinate"), dtype=np.float64)
            prediction_coordinate = np.asarray(
                prediction_entry.get("coordinate"), dtype=np.float64
            )
            truth_values = np.asarray(truth_entry.get("prediction"), dtype=np.float64)
            prediction_values = np.asarray(
                prediction_entry.get("prediction"), dtype=np.float64
            )
            if (
                truth_coordinate.ndim != 1
                or prediction_coordinate.ndim != 1
                or truth_values.ndim != 1
                or prediction_values.ndim != 1
                or truth_coordinate.size < 2
                or truth_values.shape != truth_coordinate.shape
                or prediction_values.shape != prediction_coordinate.shape
            ):
                raise ValueError(
                    f"case {case_id!r} profile {identity!r} arrays must be matching "
                    "one-dimensional arrays with at least two samples"
                )
            if not (
                np.all(np.isfinite(truth_coordinate))
                and np.all(np.isfinite(prediction_coordinate))
                and np.all(np.isfinite(truth_values))
                and np.all(np.isfinite(prediction_values))
            ):
                raise ValueError(f"case {case_id!r} profile {identity!r} contains non-finite values")
            if truth_coordinate.shape != prediction_coordinate.shape or not np.allclose(
                truth_coordinate,
                prediction_coordinate,
                rtol=0.0,
                atol=coordinate_tolerance,
            ):
                raise ValueError(
                    f"case {case_id!r} profile {identity!r} coordinates do not align"
                )
            grouped_truth[identity].append(truth_values)
            grouped_prediction[identity].append(prediction_values)

    group_results: list[dict[str, Any]] = []
    for station_id in station_ids:
        for quantity_id in quantity_ids:
            identity = (station_id, quantity_id)
            truth = np.concatenate(grouped_truth[identity])
            prediction = np.concatenate(grouped_prediction[identity])
            raw_r2 = r2_score(truth, prediction)
            bounded_r2 = max(0.0, min(1.0, raw_r2))
            group_results.append(
                {
                    "station_id": station_id,
                    "quantity_id": quantity_id,
                    "raw_r2": raw_r2,
                    "bounded_r2": bounded_r2,
                    "case_count": len(truth_cases),
                    "sample_count": int(truth.size),
                }
            )

    score = float(np.mean([group["bounded_r2"] for group in group_results]))
    return {
        "metric_id": PROFILE_METRIC_ID,
        "aggregation": PROFILE_AGGREGATION,
        "value": score,
        "case_count": len(truth_cases),
        "expected_case_count": expected_case_count,
        "case_coverage": case_coverage,
        "station_count": len(station_ids),
        "quantity_count": len(quantity_ids),
        "group_count": len(group_results),
        "groups": group_results,
    }
