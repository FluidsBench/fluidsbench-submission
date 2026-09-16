#!/usr/bin/env python3
"""Regenerate realistic, deterministic DrivAerML prototype leaderboard rows.

The checked-in DrivAerML rows are explicitly non-rankable teaching fixtures.
This generator makes their scalar values internally coherent and gives every
case all four continuous Cp cuts and all sixteen AutoCFD5 velocity lines on the
grids declared by the current candidate submission contract.  The smooth
profiles are analytical CFD-like surrogates; they are not claims about model
quality and are not substitutes for evaluation against the native VTK fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scores import (
    composite_component_group_scores,
    composite_overall_score,
)
from scripts.prototype_methodologies import build_prototype_methodology


DATASET_ROOT = ROOT / "benchmark-specs" / "drivaerml"
SUBMISSIONS_ROOT = ROOT / "submissions" / "drivaerml"
FORCE_R2_REFERENCE_PATH = DATASET_ROOT / "force-r2-truth-statistics.json"

U_INF_M_PER_S = 38.889
CASES_PER_CHUNK = 5
GENERATED_AT = "2026-08-21T00:00:00Z"
GENERATOR_REVISION = "drivaerml-bounded-r2-prototype-v2"
GENERATOR_COMMAND = "python3 scripts/generate_drivaerml_leaderboard_fixtures.py"
CASE_ID_PATTERN = re.compile(r"run_([1-9][0-9]*)\Z")

SUBMISSION_SPLITS = {
    "drivaerml-ab-upt": "full",
    "drivaerml-gino": "medium",
    "drivaerml-graph-u-net": "scarce",
    "drivaerml-lno": "super_scarce",
    "drivaerml-oformer": "high_drag",
    "drivaerml-pointnet": "low_drag",
    "drivaerml-transformer": "rear_separation",
    "drivaerml-transolver": "full",
    "drivaerml-upt": "medium",
    "dummy-drivaerml-meshoperator-v1": "geometry",
}

# Continuous native cuts do not yet have an immutable public support.  These
# dense, case-varying grids are therefore display surrogates in metres, not a
# claim that every native intersection will contain these exact samples.
PRESSURE_SUPPORTS = {
    "upperbody_centerline": (4.92, 198),
    "underbody_centerline": (4.58, 184),
    "sidewall_z_0_15": (4.72, 190),
    "front_left_wheelhouse_y_neg_0_6": (1.78, 144),
}

PROTOTYPE_NOTE = (
    "Deterministic analytical teaching fixture for the closed DrivAerML candidate. "
    "It contains smooth CFD-like predictions on all four continuous Cp-cut supports "
    "and the exact 3,756-point AutoCFD5 velocity grid for every official test case. "
    "The dimensional field errors retain the prototype calibration and the profile "
    "RMSE and R2 diagnostics are recomputed from the generated curves. Force R2 "
    "is derived from each prototype's existing RMSE and hash-bound official split "
    "truth statistics. The four score values use the published bounded-error and "
    "bounded-R2 transforms. This is not a native-field evaluation, approved result, "
    "ranking entry, or citable model claim."
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    if compact:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    else:
        encoded = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)
    path.write_text(encoded + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def linspace(start: float, end: float, count: int) -> list[float]:
    denominator = count - 1
    values = [round(start + (end - start) * index / denominator, 12) for index in range(count)]
    values[0] = start
    values[-1] = end
    return values


def gaussian(value: float, centre: float, width: float) -> float:
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def smooth_window(value: float, start: float, end: float, width: float) -> float:
    return 0.5 * (
        math.tanh((value - start) / width) - math.tanh((value - end) / width)
    )


def case_number(case_id: str) -> int:
    match = CASE_ID_PATTERN.fullmatch(case_id)
    if match is None:
        raise ValueError(f"unsupported DrivAerML case ID: {case_id!r}")
    return int(match.group(1))


def stable_phase(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return 2.0 * math.pi * fraction


def pressure_coordinates(station_id: str, run: int) -> list[float]:
    base_length, base_count = PRESSURE_SUPPORTS[station_id]
    phase = 0.071 * run + stable_phase(station_id)
    length = base_length * (1.0 + 0.028 * math.sin(phase) + 0.011 * math.cos(0.47 * phase))
    count = base_count + ((run + int(10.0 * phase)) % 9) - 4
    return linspace(0.0, round(length, 9), count)


def pressure_truth(station_id: str, coordinate: Sequence[float], run: int) -> list[float]:
    length = coordinate[-1]
    case_phase = 0.113 * run
    shape = 1.0 + 0.055 * math.sin(case_phase)
    shift = 0.012 * math.cos(0.73 * case_phase)
    values: list[float] = []
    for arc in coordinate:
        t = arc / length
        if station_id == "upperbody_centerline":
            cp = (
                -0.055
                + 1.02 * gaussian(t, 0.018 + shift, 0.027)
                - 0.34 * shape * gaussian(t, 0.16 + shift, 0.082)
                + 0.43 * gaussian(t, 0.335 - shift, 0.046)
                - 0.39 * shape * gaussian(t, 0.505, 0.135)
                - 0.22 * gaussian(t, 0.765 + shift, 0.074)
                + 0.10 * gaussian(t, 0.915, 0.040)
            )
        elif station_id == "underbody_centerline":
            cp = (
                -0.035
                + 0.58 * gaussian(t, 0.018, 0.033)
                - 0.31 * shape * smooth_window(t, 0.075, 0.79, 0.028)
                - 0.23 * gaussian(t, 0.205 + shift, 0.050)
                + 0.12 * gaussian(t, 0.405, 0.040)
                - 0.16 * gaussian(t, 0.685 - shift, 0.065)
                + 0.24 * gaussian(t, 0.895, 0.080)
            )
        elif station_id == "sidewall_z_0_15":
            cp = (
                -0.025
                + 0.83 * gaussian(t, 0.022, 0.036)
                - 0.43 * shape * gaussian(t, 0.205 + shift, 0.055)
                - 0.12 * smooth_window(t, 0.29, 0.72, 0.045)
                - 0.31 * gaussian(t, 0.735 - shift, 0.060)
                + 0.13 * gaussian(t, 0.91, 0.055)
            )
        else:
            cp = (
                0.055
                + 0.48 * shape * math.cos(2.0 * math.pi * (t - 0.055 + shift))
                - 0.19 * math.cos(4.0 * math.pi * (t + 0.025))
                - 0.17 * gaussian(t, 0.54, 0.075)
                + 0.11 * gaussian(t, 0.91, 0.045)
            )
        cp += 0.018 * math.sin(2.0 * math.pi * t + case_phase)
        values.append(round(cp, 7))
    return values


def velocity_truth(station_id: str, coordinate: Sequence[float], run: int) -> list[float]:
    phase = 0.097 * run
    geom = 1.0 + 0.075 * math.sin(phase)
    values: list[float] = []
    for distance in coordinate:
        if station_id.startswith("autocfd5_v"):
            line = int(station_id[-1])
            z = distance
            ground_layer = 1.0 - math.exp(-((z / (0.105 + 0.012 * math.sin(phase))) ** 1.55))
            if line == 1:
                velocity = ground_layer * (1.0 + 0.035 * gaussian(z, 0.52, 0.24))
            elif line == 2:
                velocity = ground_layer * (
                    1.0 - 0.24 * geom * gaussian(z, 0.43, 0.24) + 0.08 * gaussian(z, 1.03, 0.25)
                )
            else:
                deficit = {3: 0.58, 4: 0.49, 5: 0.41, 6: 0.33}[line] * geom
                centre = 0.46 + 0.035 * (line - 3) + 0.025 * math.cos(phase)
                width = 0.24 + 0.025 * (line - 3)
                velocity = ground_layer - deficit * gaussian(z, centre, width)
                velocity += 0.065 * gaussian(z, 1.10 + 0.03 * (line - 3), 0.22)
        elif station_id.startswith("autocfd5_u"):
            line = int(station_id[-1])
            y = distance - 1.5
            central = gaussian(y, 0.0, 0.61 + 0.025 * math.sin(phase))
            left_wheel = gaussian(y, -0.72, 0.12)
            right_wheel = gaussian(y, 0.72, 0.12)
            central_effect = {1: -0.18, 2: 0.11, 3: 0.17, 4: 0.13, 5: 0.05, 6: -0.16}[line]
            wheel_effect = {1: -0.08, 2: -0.16, 3: -0.22, 4: -0.15, 5: -0.20, 6: -0.18}[line]
            asymmetry = 0.025 * math.sin(phase + 1.2 * y) * gaussian(y, 0.0, 0.95)
            velocity = 1.0 + central_effect * geom * central
            velocity += wheel_effect * geom * (left_wheel + right_wheel) + asymmetry
        elif station_id == "autocfd5_l1":
            x = distance - 0.993
            velocity = 1.0
            velocity -= 0.16 * geom * gaussian(x, -0.18, 0.20)
            velocity += 0.15 * smooth_window(x, 0.20, 3.15, 0.10)
            velocity -= 0.12 * gaussian(x, 0.70, 0.17)
            velocity -= 0.15 * gaussian(x, 2.75, 0.20)
            velocity -= 0.44 * geom * gaussian(x, 3.85, 0.48)
            velocity += 0.035 * math.sin(phase + 1.7 * x) * gaussian(x, 3.65, 0.95)
        else:
            line = int(station_id[-1])
            delta = {1: 0.048, 2: 0.059, 3: 0.072}[line] * (1.0 + 0.08 * math.sin(phase))
            velocity = 1.0 - math.exp(-((distance / delta) ** 1.42))
            velocity += 0.035 * gaussian(distance, 0.16 + 0.018 * line, 0.055)
        values.append(round(max(0.0, velocity), 7))
    return values


def raw_profile_error(
    panel_id: str,
    station_id: str,
    coordinate: Sequence[float],
    run: int,
    submission_id: str,
) -> list[float]:
    model_phase = stable_phase(submission_id)
    case_phase = 0.137 * run + model_phase
    end = coordinate[-1]
    station_phase = stable_phase(station_id)
    values: list[float] = []
    for distance in coordinate:
        t = distance / end
        correlated = (
            0.48 * math.sin(2.0 * math.pi * t + case_phase)
            + 0.24 * math.sin(5.0 * math.pi * t + station_phase)
            + 0.18 * math.cos(9.0 * math.pi * t + 0.7 * model_phase)
        )
        local = gaussian(t, 0.22 + 0.08 * math.sin(station_phase), 0.105)
        local -= 0.72 * gaussian(t, 0.70 + 0.04 * math.cos(case_phase), 0.14)
        bias = 0.14 * math.sin(case_phase + station_phase)
        error = correlated + local + bias
        if panel_id == "velocity_profiles":
            if station_id.startswith("autocfd5_r"):
                error *= 1.0 - math.exp(-distance / 0.035)
            elif station_id.startswith("autocfd5_v"):
                error *= 0.30 + 0.70 * (1.0 - math.exp(-distance / 0.075))
            elif station_id == "autocfd5_l1":
                error += 0.36 * gaussian(t, 0.76, 0.11)
            else:
                error += 0.20 * math.cos(4.0 * math.pi * t + case_phase)
        values.append(error)
    return values


def trapezoidal_rmse(coordinate: Sequence[float], errors: Sequence[float]) -> float:
    integral = 0.0
    length = coordinate[-1] - coordinate[0]
    for left, right, error_left, error_right in zip(
        coordinate,
        coordinate[1:],
        errors,
        errors[1:],
    ):
        integral += (right - left) * (error_left**2 + error_right**2) / 2.0
    return math.sqrt(integral / length)


def trapezoidal_r2_statistics(
    coordinate: Sequence[float],
    truth: Sequence[float],
    prediction: Sequence[float],
) -> tuple[float, float, float]:
    """Return normalized truth sum, truth-square sum, and SSE for one profile.

    Normalizing each line or cut to unit support gives every case/profile block
    equal total weight while retaining trapezoidal arc-length weighting within
    the block.  The three returned values are additive across blocks and are
    sufficient to evaluate one global weighted R2 with a global truth mean.
    """

    if not (len(coordinate) == len(truth) == len(prediction)) or len(coordinate) < 2:
        raise ValueError("profile R2 inputs must have a common length of at least two")
    length = coordinate[-1] - coordinate[0]
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("profile R2 coordinate support must have positive length")
    truth_integral = 0.0
    truth_squared_integral = 0.0
    squared_error_integral = 0.0
    for left, right, truth_left, truth_right, pred_left, pred_right in zip(
        coordinate,
        coordinate[1:],
        truth,
        truth[1:],
        prediction,
        prediction[1:],
    ):
        width = right - left
        if width <= 0.0:
            raise ValueError("profile R2 coordinates must be strictly increasing")
        error_left = pred_left - truth_left
        error_right = pred_right - truth_right
        truth_integral += width * (truth_left + truth_right) / 2.0
        truth_squared_integral += width * (truth_left**2 + truth_right**2) / 2.0
        squared_error_integral += width * (error_left**2 + error_right**2) / 2.0
    return (
        truth_integral / length,
        truth_squared_integral / length,
        squared_error_integral / length,
    )


def r2_from_block_statistics(
    statistics: Sequence[tuple[float, float, float]],
) -> float:
    if not statistics:
        raise ValueError("profile R2 requires at least one case/profile block")
    truth_sum = math.fsum(item[0] for item in statistics)
    truth_squared_sum = math.fsum(item[1] for item in statistics)
    squared_error_sum = math.fsum(item[2] for item in statistics)
    truth_sst = truth_squared_sum - truth_sum**2 / len(statistics)
    if truth_sst <= 0.0:
        raise ValueError("profile R2 is undefined for constant weighted truth")
    return 1.0 - squared_error_sum / truth_sst


def force_r2_metrics(
    metrics: dict[str, float],
    split_id: str,
    split_sha256: str,
    force_reference: dict[str, Any],
) -> dict[str, float]:
    split = force_reference["splits"].get(split_id)
    if not isinstance(split, dict) or split.get("split_sha256") != split_sha256:
        raise ValueError(f"force R2 truth statistics do not match split {split_id!r}")
    result: dict[str, float] = {}
    for target, rmse_metric_id, r2_metric_id in (
        ("cd", "field_integrated_cd_rmse", "cd_r2"),
        ("cl", "field_integrated_cl_rmse", "cl_r2"),
        ("c_pitch", "field_integrated_cmpitch_rmse", "c_pitch_r2"),
    ):
        statistics = split["targets"][target]
        count = int(statistics["case_count"])
        truth_sst = float(statistics["truth_sst"])
        squared_error_sum = count * float(metrics[rmse_metric_id]) ** 2
        result[r2_metric_id] = 1.0 - squared_error_sum / truth_sst
    return result


def velocity_contract(specification: dict[str, Any]) -> tuple[list[str], dict[str, list[float]]]:
    panel = next(panel for panel in specification["profile_panels"] if panel["id"] == "velocity_profiles")
    station_ids = list(panel["station_ids"])
    coordinates = {
        station_id: linspace(
            panel["station_coordinate_intervals"][station_id][0],
            panel["station_coordinate_intervals"][station_id][1],
            panel["station_sample_counts"][station_id],
        )
        for station_id in station_ids
    }
    return station_ids, coordinates


def raw_macro_rmse(
    panel_id: str,
    station_ids: Sequence[str],
    case_ids: Sequence[str],
    submission_id: str,
    velocity_coordinates: dict[str, list[float]],
) -> float:
    case_means: list[float] = []
    for case_id in case_ids:
        run = case_number(case_id)
        line_errors: list[float] = []
        for station_id in station_ids:
            coordinate = (
                velocity_coordinates[station_id]
                if panel_id == "velocity_profiles"
                else pressure_coordinates(station_id, run)
            )
            line_errors.append(
                trapezoidal_rmse(
                    coordinate,
                    raw_profile_error(panel_id, station_id, coordinate, run, submission_id),
                )
            )
        case_means.append(sum(line_errors) / len(line_errors))
    return sum(case_means) / len(case_means)


def corrected_dimensional_metrics(metrics: dict[str, float]) -> None:
    pairs = (
        (
            "surface_pressure_rel_l2",
            "surface_pressure_equal_entity_rel_l2",
            "drivaerml_surface_pressure_area_mae",
            "drivaerml_surface_pressure_area_rmse",
            "drivaerml_surface_pressure_equal_entity_mae",
            "drivaerml_surface_pressure_equal_entity_rmse",
        ),
        (
            "surface_wall_shear_rel_l2",
            "surface_wall_shear_equal_entity_rel_l2",
            "drivaerml_surface_wall_shear_area_mae",
            "drivaerml_surface_wall_shear_area_rmse",
            "drivaerml_surface_wall_shear_equal_entity_mae",
            "drivaerml_surface_wall_shear_equal_entity_rmse",
        ),
    )
    for area_rel, equal_rel, area_mae, area_rmse, equal_mae, equal_rmse in pairs:
        ratio = metrics[equal_rel] / metrics[area_rel]
        metrics[equal_mae] = round(metrics[area_mae] * ratio, 7)
        metrics[equal_rmse] = round(metrics[area_rmse] * ratio, 7)


def target_profile_metrics(metrics: dict[str, float]) -> tuple[float, float]:
    cp_target = (
        2.0
        * metrics["drivaerml_surface_pressure_equal_entity_rmse"]
        / U_INF_M_PER_S**2
    )
    # Profile errors grow with the whole-volume dimensional error, but should
    # not turn a poor prototype into an obviously impossible oscillation.  The
    # rational saturation retains the model ordering and approaches a 0.35
    # U_inf diagnostic penalty for increasingly weak field predictions.
    unsaturated_velocity_error = (
        1.25
        * metrics["drivaerml_volume_velocity_equal_entity_rmse"]
        / U_INF_M_PER_S
    )
    velocity_target = 0.005 + unsaturated_velocity_error / (
        1.0 + unsaturated_velocity_error / 0.35
    )
    return cp_target, velocity_target


def build_case(
    case_id: str,
    submission_id: str,
    pressure_station_ids: Sequence[str],
    velocity_station_ids: Sequence[str],
    velocity_coordinates: dict[str, list[float]],
    cp_scale: float,
    velocity_scale: float,
) -> tuple[
    dict[str, Any],
    list[float],
    list[tuple[str, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
]:
    run = case_number(case_id)
    series: list[dict[str, Any]] = []
    cp_losses: list[float] = []
    velocity_losses: list[tuple[str, float]] = []
    cp_r2_statistics: list[tuple[float, float, float]] = []
    velocity_r2_statistics: list[tuple[float, float, float]] = []

    for station_id in pressure_station_ids:
        coordinate = pressure_coordinates(station_id, run)
        truth = pressure_truth(station_id, coordinate, run)
        raw_error = raw_profile_error(
            "pressure_profiles", station_id, coordinate, run, submission_id
        )
        prediction = [round(value + cp_scale * error, 7) for value, error in zip(truth, raw_error)]
        rounded_errors = [predicted - expected for predicted, expected in zip(prediction, truth)]
        cp_losses.append(trapezoidal_rmse(coordinate, rounded_errors))
        cp_r2_statistics.append(
            trapezoidal_r2_statistics(coordinate, truth, prediction)
        )
        series.append(
            {
                "panel_id": "pressure_profiles",
                "station_id": station_id,
                "quantity_id": "cp",
                "coordinate": coordinate,
                "prediction": prediction,
            }
        )

    for station_id in velocity_station_ids:
        coordinate = velocity_coordinates[station_id]
        truth = velocity_truth(station_id, coordinate, run)
        raw_error = raw_profile_error(
            "velocity_profiles", station_id, coordinate, run, submission_id
        )
        prediction = [
            round(value + velocity_scale * error, 7)
            for value, error in zip(truth, raw_error)
        ]
        rounded_errors = [predicted - expected for predicted, expected in zip(prediction, truth)]
        velocity_losses.append((station_id, trapezoidal_rmse(coordinate, rounded_errors)))
        velocity_r2_statistics.append(
            trapezoidal_r2_statistics(coordinate, truth, prediction)
        )
        series.append(
            {
                "panel_id": "velocity_profiles",
                "station_id": station_id,
                "quantity_id": "velocity_ratio",
                "coordinate": coordinate,
                "prediction": prediction,
            }
        )

    return (
        {"case_id": case_id, "series": series},
        cp_losses,
        velocity_losses,
        cp_r2_statistics,
        velocity_r2_statistics,
    )


def build_ground_truth_case(
    case_id: str,
    pressure_station_ids: Sequence[str],
    velocity_station_ids: Sequence[str],
    velocity_coordinates: dict[str, list[float]],
) -> dict[str, Any]:
    run = case_number(case_id)
    series: list[dict[str, Any]] = []
    for station_id in pressure_station_ids:
        coordinate = pressure_coordinates(station_id, run)
        series.append(
            {
                "panel_id": "pressure_profiles",
                "station_id": station_id,
                "quantity_id": "cp",
                "coordinate": coordinate,
                "value": pressure_truth(station_id, coordinate, run),
            }
        )
    for station_id in velocity_station_ids:
        coordinate = velocity_coordinates[station_id]
        series.append(
            {
                "panel_id": "velocity_profiles",
                "station_id": station_id,
                "quantity_id": "velocity_ratio",
                "coordinate": coordinate,
                "value": velocity_truth(station_id, coordinate, run),
            }
        )
    return {"case_id": case_id, "series": series}


def write_ground_truth_bundle(
    ground_truth_root: Path,
    specification: dict[str, Any],
) -> str:
    """Write contract-aligned DrivAerML prototype truth and return manifest SHA-256."""

    manifest_path = ground_truth_root / "manifest.json"
    manifest = load_json(manifest_path)
    dataset = next(
        item for item in manifest["datasets"] if item.get("id") == "drivaerml"
    )
    dataset_directory = ground_truth_root / "datasets" / "drivaerml"
    pressure_station_ids = list(PRESSURE_SUPPORTS)
    velocity_station_ids, velocity_coordinates = velocity_contract(specification)

    case_sets: dict[str, list[str]] = {}
    split_case_sets: dict[str, str] = {}
    for split_entry in specification["splits"]:
        split = load_json(DATASET_ROOT / split_entry["index_file"])
        case_set_id = split["case_set_id"]
        previous = case_sets.setdefault(case_set_id, split["case_ids"])
        if previous != split["case_ids"]:
            raise ValueError(
                f"DrivAerML splits sharing {case_set_id!r} disagree on test cases"
            )
        split_case_sets[split_entry["id"]] = case_set_id

    index_metadata: dict[str, dict[str, Any]] = {}
    for case_set_id, case_ids in case_sets.items():
        case_set_directory = dataset_directory / case_set_id
        case_set_directory.mkdir(parents=True, exist_ok=True)
        for stale_path in case_set_directory.glob("chunk-*.json"):
            stale_path.unlink()

        index_chunks: list[dict[str, Any]] = []
        for chunk_number, chunk_case_ids in enumerate(chunks(case_ids, CASES_PER_CHUNK)):
            chunk_path = case_set_directory / f"chunk-{chunk_number:03d}.json"
            write_json(
                chunk_path,
                {
                    "schema_version": "1.0",
                    "cases": [
                        build_ground_truth_case(
                            case_id,
                            pressure_station_ids,
                            velocity_station_ids,
                            velocity_coordinates,
                        )
                        for case_id in chunk_case_ids
                    ],
                },
                compact=True,
            )
            index_chunks.append(
                {
                    "file": chunk_path.name,
                    "case_ids": list(chunk_case_ids),
                    "sha256": sha256_file(chunk_path),
                }
            )

        index_path = case_set_directory / "index.json"
        write_json(
            index_path,
            {
                "schema_version": "1.0",
                "dataset_id": "drivaerml",
                "case_set_id": case_set_id,
                "case_count": len(case_ids),
                "case_id_status": "official",
                "chunks": index_chunks,
            },
        )
        index_metadata[case_set_id] = {
            "id": case_set_id,
            "index_file": (
                Path("datasets") / "drivaerml" / case_set_id / "index.json"
            ).as_posix(),
            "index_sha256": sha256_file(index_path),
            "case_count": len(case_ids),
            "case_id_status": "official",
        }

    existing_case_sets = {
        item["id"]: item for item in dataset.get("case_sets", [])
    }
    dataset["case_sets"] = [
        {
            **existing_case_sets.get(case_set_id, {}),
            **index_metadata[case_set_id],
        }
        for case_set_id in case_sets
    ]
    for split in dataset["splits"]:
        case_set_id = split_case_sets[split["id"]]
        split["case_set_id"] = case_set_id
        split["case_count"] = len(case_sets[case_set_id])
    manifest["data_release"]["generated_at"] = GENERATED_AT
    write_json(manifest_path, manifest)
    return sha256_file(manifest_path)


def regenerate_submission(
    submission_id: str,
    split_id: str,
    specification: dict[str, Any],
    diagnostics: dict[str, Any],
    force_reference: dict[str, Any],
) -> dict[str, float]:
    directory = SUBMISSIONS_ROOT / submission_id
    submission_path = directory / "submission.json"
    submission = load_json(submission_path)
    if submission.get("approval", {}).get("status") != "prototype":
        raise ValueError(f"refusing to rewrite non-prototype submission {submission_id}")
    if submission.get("submission_id") != submission_id or submission.get("split_id") != split_id:
        raise ValueError(f"submission identity mismatch for {submission_id}")

    split_entry = next(entry for entry in specification["splits"] if entry["id"] == split_id)
    split_path = DATASET_ROOT / split_entry["index_file"]
    split = load_json(split_path)
    case_ids = split["case_ids"]
    if split.get("case_id_status") != "official" or split.get("case_set_id") != submission.get("case_set_id"):
        raise ValueError(f"official split binding mismatch for {submission_id}")

    pressure_panel = next(panel for panel in specification["profile_panels"] if panel["id"] == "pressure_profiles")
    pressure_station_ids = list(pressure_panel["station_ids"])
    if set(pressure_station_ids) != set(PRESSURE_SUPPORTS):
        raise ValueError("pressure support constants do not match submission-spec.json")
    velocity_station_ids, velocity_coordinates = velocity_contract(specification)
    diagnostic_velocity_ids = [station["id"] for station in diagnostics["velocity_profiles"]["stations"]]
    if velocity_station_ids != diagnostic_velocity_ids:
        raise ValueError("velocity panel and diagnostics registry station order differ")
    experimental_ids = {
        station["id"]
        for station in diagnostics["velocity_profiles"]["stations"]
        if station["experimental_availability"] != "none"
    }

    metrics = dict(submission["metric_values"])
    corrected_dimensional_metrics(metrics)
    cp_target, velocity_target = target_profile_metrics(metrics)
    cp_raw = raw_macro_rmse(
        "pressure_profiles",
        pressure_station_ids,
        case_ids,
        submission_id,
        velocity_coordinates,
    )
    velocity_raw = raw_macro_rmse(
        "velocity_profiles",
        velocity_station_ids,
        case_ids,
        submission_id,
        velocity_coordinates,
    )
    cp_scale = cp_target / cp_raw
    velocity_scale = velocity_target / velocity_raw

    profiles_directory = directory / "profiles"
    profiles_directory.mkdir(exist_ok=True)
    for stale_path in profiles_directory.glob("chunk-*.json"):
        stale_path.unlink()

    index_chunks: list[dict[str, Any]] = []
    all_cp_case_means: list[float] = []
    all_velocity_case_means: list[float] = []
    experimental_case_means: list[float] = []
    all_cp_r2_statistics: list[tuple[float, float, float]] = []
    all_velocity_r2_statistics: list[tuple[float, float, float]] = []
    for chunk_number, chunk_case_ids in enumerate(chunks(case_ids, CASES_PER_CHUNK)):
        cases: list[dict[str, Any]] = []
        for case_id in chunk_case_ids:
            (
                case,
                cp_losses,
                velocity_losses,
                cp_r2_statistics,
                velocity_r2_statistics,
            ) = build_case(
                case_id,
                submission_id,
                pressure_station_ids,
                velocity_station_ids,
                velocity_coordinates,
                cp_scale,
                velocity_scale,
            )
            cases.append(case)
            all_cp_case_means.append(sum(cp_losses) / len(cp_losses))
            all_velocity_case_means.append(
                sum(loss for _, loss in velocity_losses) / len(velocity_losses)
            )
            experimental_losses = [
                loss for station_id, loss in velocity_losses if station_id in experimental_ids
            ]
            experimental_case_means.append(
                sum(experimental_losses) / len(experimental_losses)
            )
            all_cp_r2_statistics.extend(cp_r2_statistics)
            all_velocity_r2_statistics.extend(velocity_r2_statistics)

        chunk_path = profiles_directory / f"chunk-{chunk_number:03d}.json"
        write_json(chunk_path, {"schema_version": "1.0", "cases": cases}, compact=True)
        index_chunks.append(
            {
                "file": chunk_path.name,
                "case_ids": list(chunk_case_ids),
                "sha256": sha256_file(chunk_path),
            }
        )

    metrics["velocity_profile_uinf_rmse"] = round(
        sum(all_velocity_case_means) / len(all_velocity_case_means), 12
    )
    metrics["velocity_profile_experimental_subset_uinf_rmse"] = round(
        sum(experimental_case_means) / len(experimental_case_means), 12
    )
    metrics["cp_cut_rmse"] = round(sum(all_cp_case_means) / len(all_cp_case_means), 12)
    metrics["velocity_profile_r2"] = round(
        r2_from_block_statistics(all_velocity_r2_statistics), 12
    )
    metrics["cp_cut_r2"] = round(r2_from_block_statistics(all_cp_r2_statistics), 12)
    metrics.update(
        {
            metric_id: round(value, 12)
            for metric_id, value in force_r2_metrics(
                metrics,
                split_id,
                sha256_file(split_path),
                force_reference,
            ).items()
        }
    )
    metrics.update(
        {
            metric_id: round(value, 12)
            for metric_id, value in composite_component_group_scores(
                metrics,
                specification["overall_score_composite"],
                specification["component_score_groups"],
            ).items()
        }
    )
    metrics["overall_score"] = round(
        composite_overall_score(metrics, specification["overall_score_composite"]),
        12,
    )

    index_path = profiles_directory / "index.json"
    index = {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": split_id,
        "case_set_id": split["case_set_id"],
        "case_count": len(case_ids),
        "case_id_status": "official",
        "chunks": index_chunks,
    }
    write_json(index_path, index)

    evidence_path = directory / submission["evaluation"]["evidence_file"]
    evidence = load_json(evidence_path)
    evidence.update(
        {
            "code_revision": GENERATOR_REVISION,
            "generated_at": GENERATED_AT,
            "metric_values": metrics,
            "profile_index_sha256": sha256_file(index_path),
            "notes": PROTOTYPE_NOTE,
            "command": GENERATOR_COMMAND,
        }
    )
    write_json(evidence_path, evidence)

    submission["split_sha256"] = sha256_file(split_path)
    submission["metric_values"] = metrics
    submission["evaluation"].update(
        {
            "code_revision": GENERATOR_REVISION,
            "command": GENERATOR_COMMAND,
            "evidence_sha256": sha256_file(evidence_path),
        }
    )
    submission["approval"]["note"] = (
        "Realistic deterministic prototype fixture for the closed DrivAerML candidate; "
        "no metric or score is an approved model result."
    )
    submission["profile_data"].update(
        {
            "index_file": "profiles/index.json",
            "case_count": len(case_ids),
            "case_set_id": split["case_set_id"],
        }
    )
    submission["note"] = PROTOTYPE_NOTE
    methodology_contract = load_json(DATASET_ROOT / "methodology-contract.json")
    submission["methodology"] = build_prototype_methodology(
        submission,
        methodology_contract,
    )
    write_json(submission_path, submission)
    return {
        "overall_score": metrics["overall_score"],
        "force_score": metrics["force_score"],
        "diagnostic_score": metrics["diagnostic_score"],
        "cp_cut_rmse": metrics["cp_cut_rmse"],
        "velocity_profile_uinf_rmse": metrics["velocity_profile_uinf_rmse"],
        "velocity_profile_experimental_subset_uinf_rmse": metrics[
            "velocity_profile_experimental_subset_uinf_rmse"
        ],
    }


def validate_fixture_inventory() -> None:
    actual = {
        path.name
        for path in SUBMISSIONS_ROOT.iterdir()
        if path.is_dir() and (path / "submission.json").is_file()
    }
    expected = set(SUBMISSION_SPLITS)
    if actual != expected:
        raise ValueError(
            "SUBMISSION_SPLITS must exactly cover DrivAerML prototype rows; "
            f"missing={sorted(actual - expected)}, unexpected={sorted(expected - actual)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-id",
        action="append",
        choices=sorted(SUBMISSION_SPLITS),
        help="Regenerate only the named prototype row (repeatable); default: all rows.",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        help=(
            "Additionally replace the DrivAerML dataset below a website "
            "assets/data/profile-ground-truth directory."
        ),
    )
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Write only --ground-truth-root; do not rewrite submission fixtures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_fixture_inventory()
    specification = load_json(DATASET_ROOT / "submission-spec.json")
    diagnostics = load_json(DATASET_ROOT / specification["profile_definition"]["file"])
    force_reference = load_json(FORCE_R2_REFERENCE_PATH)
    if args.ground_truth_only and args.ground_truth_root is None:
        raise ValueError("--ground-truth-only requires --ground-truth-root")
    if not args.ground_truth_only:
        selected = args.submission_id or list(SUBMISSION_SPLITS)
        for submission_id in selected:
            metrics = regenerate_submission(
                submission_id,
                SUBMISSION_SPLITS[submission_id],
                specification,
                diagnostics,
                force_reference,
            )
            print(
                f"Regenerated {submission_id}: "
                f"Cp={metrics['cp_cut_rmse']:.5f}, "
                f"U={metrics['velocity_profile_uinf_rmse']:.5f}, "
                f"U-exp={metrics['velocity_profile_experimental_subset_uinf_rmse']:.5f}, "
                f"score={metrics['overall_score']:.2f}"
            )
    if args.ground_truth_root is not None:
        ground_truth_root = args.ground_truth_root.expanduser().resolve()
        manifest_sha256 = write_ground_truth_bundle(
            ground_truth_root,
            specification,
        )
        print(f"Regenerated DrivAerML profile ground truth: {manifest_sha256}")


if __name__ == "__main__":
    main()
