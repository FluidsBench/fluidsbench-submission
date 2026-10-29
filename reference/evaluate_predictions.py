"""Evaluate canonical predictions against benchmark-owned scoring support.

This command joins submitter predictions to fixed ground truth by stable
``support_id``. Missing, duplicate, or unknown IDs fail. Ground truth,
coordinates, and weights are loaded only from the benchmark support release
and are never interpolated or resampled.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reference.scoring_support import (
    ScoringSupport,
    ScoringSupportError,
    align_predictions,
    load_scored_predictions,
    load_scoring_support,
    load_support_release,
    sha256_file,
)


def _prepare_field_metric_inputs(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    *,
    weighting: str,
    allow_zero_total_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and flatten one field while retaining spatial weights.

    Each entity's spatial weight is repeated for every vector component.
    """

    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    spatial_weights = np.asarray(weights, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ScoringSupportError("truth and prediction must have identical [entity, component] shapes")
    if spatial_weights.shape != (truth.shape[0],):
        raise ScoringSupportError("weights must contain one value per scoring entity")
    if (
        not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(spatial_weights))
    ):
        raise ScoringSupportError("metric arrays must contain finite values")
    if np.any(spatial_weights < 0):
        raise ScoringSupportError("metric weights cannot be negative")
    if weighting == "uniform":
        spatial_weights = np.ones_like(spatial_weights)
    elif weighting != "support_weights":
        raise ScoringSupportError(f"unknown weighting {weighting!r}")
    if not allow_zero_total_weight and not float(np.sum(spatial_weights)) > 0:
        raise ScoringSupportError("metric weights must have positive total weight")

    repeated_weights = np.repeat(spatial_weights, truth.shape[1])
    flat_truth = truth.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    return flat_truth, flat_prediction, spatial_weights, repeated_weights


def relative_l2_sufficient_statistics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    *,
    weighting: str,
) -> dict[str, float | int]:
    """Return additive statistics for an exact relative-L2 calculation.

    ``entity_count`` and ``total_weight`` count spatial entities, not flattened
    vector components. The numerator and denominator do include every declared
    component, with the same spatial weight repeated for each component.
    """

    flat_truth, flat_prediction, spatial_weights, repeated_weights = (
        _prepare_field_metric_inputs(
            truth,
            prediction,
            weights,
            weighting=weighting,
            allow_zero_total_weight=True,
        )
    )
    error = flat_prediction - flat_truth
    numerator = float(np.sum(repeated_weights * np.square(error)))
    denominator = float(np.sum(repeated_weights * np.square(flat_truth)))
    return {
        "numerator": numerator,
        "denominator": denominator,
        "entity_count": int(len(spatial_weights)),
        "total_weight": float(np.sum(spatial_weights)),
    }


def relative_l2_from_sufficient_statistics(
    statistics: Sequence[Mapping[str, float | int]],
) -> float:
    """Combine one or more chunks and calculate relative L2 in percent."""

    if not statistics:
        raise ScoringSupportError("relative L2 statistics must contain at least one chunk")
    numerator = 0.0
    denominator = 0.0
    entity_count = 0
    total_weight = 0.0
    for chunk in statistics:
        chunk_numerator = chunk.get("numerator")
        chunk_denominator = chunk.get("denominator")
        chunk_entity_count = chunk.get("entity_count")
        chunk_total_weight = chunk.get("total_weight")
        if (
            not isinstance(chunk_numerator, (int, float))
            or isinstance(chunk_numerator, bool)
            or not math.isfinite(chunk_numerator)
            or chunk_numerator < 0
        ):
            raise ScoringSupportError("relative L2 numerator must be finite and non-negative")
        if (
            not isinstance(chunk_denominator, (int, float))
            or isinstance(chunk_denominator, bool)
            or not math.isfinite(chunk_denominator)
            or chunk_denominator < 0
        ):
            raise ScoringSupportError(
                "relative L2 chunk denominator must be finite and non-negative"
            )
        if (
            not isinstance(chunk_entity_count, int)
            or isinstance(chunk_entity_count, bool)
            or chunk_entity_count < 1
        ):
            raise ScoringSupportError("relative L2 entity_count must be a positive integer")
        if (
            not isinstance(chunk_total_weight, (int, float))
            or isinstance(chunk_total_weight, bool)
            or not math.isfinite(chunk_total_weight)
            or chunk_total_weight < 0
        ):
            raise ScoringSupportError(
                "relative L2 chunk total_weight must be finite and non-negative"
            )
        numerator += float(chunk_numerator)
        denominator += float(chunk_denominator)
        entity_count += chunk_entity_count
        total_weight += float(chunk_total_weight)
    if entity_count < 1 or not total_weight > 0:
        raise ScoringSupportError("relative L2 statistics describe an empty support")
    if not denominator > 0:
        raise ScoringSupportError("relative L2 ground-truth denominator is zero")
    value = 100.0 * math.sqrt(numerator / denominator)
    if not math.isfinite(value):
        raise ScoringSupportError("metric result is non-finite")
    return value


def field_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    *,
    reduction: str,
    weighting: str,
) -> float:
    """Calculate one case metric after flattening vector components."""

    if reduction == "relative_l2_percent":
        return relative_l2_from_sufficient_statistics(
            [
                relative_l2_sufficient_statistics(
                    truth,
                    prediction,
                    weights,
                    weighting=weighting,
                )
            ]
        )

    flat_truth, flat_prediction, _spatial_weights, repeated_weights = (
        _prepare_field_metric_inputs(
            truth,
            prediction,
            weights,
            weighting=weighting,
        )
    )
    error = flat_prediction - flat_truth
    if reduction == "relative_l1_percent":
        denominator = float(np.sum(repeated_weights * np.abs(flat_truth)))
        if not denominator > 0:
            raise ScoringSupportError("relative L1 ground-truth denominator is zero")
        value = 100.0 * float(np.sum(repeated_weights * np.abs(error))) / denominator
    elif reduction == "mae":
        value = float(np.sum(repeated_weights * np.abs(error))) / float(
            np.sum(repeated_weights)
        )
    elif reduction == "mse":
        value = float(np.sum(repeated_weights * np.square(error))) / float(
            np.sum(repeated_weights)
        )
    elif reduction == "rmse":
        value = math.sqrt(
            float(np.sum(repeated_weights * np.square(error)))
            / float(np.sum(repeated_weights))
        )
    elif reduction == "r2":
        weighted_mean = float(np.sum(repeated_weights * flat_truth)) / float(
            np.sum(repeated_weights)
        )
        denominator = float(
            np.sum(repeated_weights * np.square(flat_truth - weighted_mean))
        )
        if not denominator > 0:
            raise ScoringSupportError("R2 ground truth is constant")
        value = 1.0 - float(
            np.sum(repeated_weights * np.square(error))
        ) / denominator
    else:
        raise ScoringSupportError(f"unknown metric reduction {reduction!r}")
    if not math.isfinite(value):
        raise ScoringSupportError("metric result is non-finite")
    return value


def evaluate_support(
    support: ScoringSupport,
    aligned: Mapping[str, np.ndarray],
    definition: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    values: dict[str, float] = {}
    sufficient_statistics: dict[str, dict[str, Any]] = {}
    for binding in definition["metric_bindings"]:
        if binding["case_evidence"] != "metric_value":
            continue
        quantity_id = binding["quantity_id"]
        metric_id = binding["metric_id"]
        if binding["reduction"] == "relative_l2_percent":
            statistics = relative_l2_sufficient_statistics(
                support.targets[quantity_id],
                aligned[quantity_id],
                support.weights,
                weighting=binding["weighting"],
            )
            values[metric_id] = relative_l2_from_sufficient_statistics([statistics])
            sufficient_statistics[metric_id] = {
                "reduction": "relative_l2_percent",
                "weighting": binding["weighting"],
                "dataset_weighting": binding["dataset_weighting"],
                **statistics,
            }
        else:
            values[metric_id] = field_metric(
                support.targets[quantity_id],
                aligned[quantity_id],
                support.weights,
                reduction=binding["reduction"],
                weighting=binding["weighting"],
            )
    return values, sufficient_statistics


def aggregate_metric(
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    binding: Mapping[str, Any],
) -> float:
    """Aggregate one bound metric using its published cross-case rule."""

    aggregation = binding["aggregation"]
    reduction = binding["reduction"]
    weighting = binding["weighting"]
    if aggregation == "per_geometry_then_macro_average":
        values = [
            field_metric(
                truth,
                prediction,
                weights,
                reduction=reduction,
                weighting=weighting,
            )
            for truth, prediction, weights in payloads
        ]
        value = float(np.mean(values))
    elif aggregation in {"flatten_all_aligned_field_values", "all_test_cases"}:
        truth = np.concatenate([item[0] for item in payloads], axis=0)
        prediction = np.concatenate([item[1] for item in payloads], axis=0)
        weights = np.concatenate([item[2] for item in payloads], axis=0)
        value = field_metric(
            truth,
            prediction,
            weights,
            reduction=reduction,
            weighting=weighting,
        )
    elif aggregation == "benchmark_field_rrmse_across_cases":
        if (
            reduction != "dataset_reference"
            or binding.get("reference_rule", {}).get("id") != aggregation
        ):
            raise ScoringSupportError(
                "benchmark field RRMSE requires its matching dataset reference rule"
            )
        terms: list[float] = []
        for truth, prediction, _weights in payloads:
            flat_truth = np.asarray(truth, dtype=np.float64).reshape(-1)
            flat_prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
            denominator = float(np.max(np.abs(flat_truth))) ** 2
            if not denominator > 0:
                raise ScoringSupportError("benchmark field RRMSE denominator is zero")
            terms.append(
                float(np.mean(np.square(flat_prediction - flat_truth))) / denominator
            )
        value = math.sqrt(float(np.mean(terms)))
    elif aggregation == "benchmark_scalar_rrmse_across_cases":
        if (
            reduction != "dataset_reference"
            or binding.get("reference_rule", {}).get("id") != aggregation
        ):
            raise ScoringSupportError(
                "benchmark scalar RRMSE requires its matching dataset reference rule"
            )
        terms = []
        for truth, prediction, _weights in payloads:
            flat_truth = np.asarray(truth, dtype=np.float64).reshape(-1)
            flat_prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
            if np.any(flat_truth == 0):
                raise ScoringSupportError("benchmark scalar RRMSE denominator is zero")
            terms.extend(
                (
                    np.square(flat_prediction - flat_truth)
                    / np.square(flat_truth)
                ).tolist()
            )
        value = math.sqrt(float(np.mean(terms)))
    else:
        raise ScoringSupportError(f"unknown metric aggregation {aggregation!r}")
    if not math.isfinite(value):
        raise ScoringSupportError("aggregate metric result is non-finite")
    return value


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ScoringSupportError(f"cannot read JSON {path}: {error}") from error


def _safe_relative(base: Path, value: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise ScoringSupportError(f"prediction path escapes its manifest: {value!r}") from error
    return candidate


def evaluate_prediction_artifact(
    *,
    support_manifest_path: Path,
    case_set_id: str,
    prediction_manifest_path: Path,
    submission_id: str,
    split_id: str,
) -> dict[str, Any]:
    """Evaluate a complete scored-prediction artifact and macro-average cases."""

    support_manifest_path = support_manifest_path.resolve()
    prediction_manifest_path = prediction_manifest_path.resolve()
    release = load_support_release(support_manifest_path, case_set_id)
    prediction_manifest = _load_json(prediction_manifest_path)
    if prediction_manifest.get("kind") != "scored_predictions":
        raise ScoringSupportError("the metric evaluator requires kind=scored_predictions")
    expected_identity = {
        "submission_id": submission_id,
        "dataset_id": release.manifest["dataset_id"],
        "split_id": split_id,
        "case_set_id": case_set_id,
        "support_release_id": release.manifest["release_id"],
        "support_manifest_sha256": sha256_file(support_manifest_path),
    }
    for key, expected in expected_identity.items():
        if prediction_manifest.get(key) != expected:
            raise ScoringSupportError(
                f"prediction manifest {key} must equal {expected!r}"
            )

    expected_case_ids = list(release.cases)
    prediction_cases = prediction_manifest.get("cases", [])
    actual_case_ids = [
        case.get("case_id") for case in prediction_cases if isinstance(case, dict)
    ]
    if prediction_manifest.get("case_count") != len(prediction_cases):
        raise ScoringSupportError("prediction manifest case_count does not match its cases")
    if actual_case_ids != expected_case_ids:
        raise ScoringSupportError(
            "prediction artifact must cover every scoring case in exact index order"
        )

    aggregate_payloads: dict[
        str,
        tuple[Mapping[str, Any], list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ] = {}
    case_records: list[dict[str, Any]] = []
    for case_declaration in prediction_cases:
        case_id = case_declaration["case_id"]
        files = case_declaration.get("files", [])
        file_support_ids = [
            item.get("support_id") for item in files if isinstance(item, dict)
        ]
        if (
            len(file_support_ids) != len(files)
            or len(file_support_ids) != len(set(file_support_ids))
            or set(file_support_ids) != set(release.supports)
        ):
            raise ScoringSupportError(
                f"{case_id} prediction files must define every support exactly once"
            )
        support_records: list[dict[str, Any]] = []
        for file_declaration in files:
            support_name = file_declaration["support_id"]
            prediction_path = _safe_relative(
                prediction_manifest_path.parent, file_declaration["file"]
            )
            if not prediction_path.is_file():
                raise ScoringSupportError(f"missing prediction file: {prediction_path}")
            if sha256_file(prediction_path) != file_declaration["sha256"]:
                raise ScoringSupportError(
                    f"prediction file SHA-256 mismatch: {prediction_path}"
                )
            support = load_scoring_support(release, case_id, support_name)
            predictions = load_scored_predictions(
                prediction_path,
                file_declaration["format"],
                release.supports[support_name],
                case_id=case_id,
                support_name=support_name,
            )
            if file_declaration.get("row_count") != len(predictions.support_ids):
                raise ScoringSupportError(
                    f"{case_id}/{support_name} row_count does not match prediction data"
                )
            aligned = align_predictions(support, predictions)
            metric_values, metric_sufficient_statistics = evaluate_support(
                support, aligned, release.supports[support_name]
            )
            for binding in release.supports[support_name]["metric_bindings"]:
                quantity_id = binding["quantity_id"]
                metric_id = binding["metric_id"]
                aggregate_binding, payloads = aggregate_payloads.setdefault(
                    metric_id,
                    (binding, []),
                )
                if aggregate_binding != binding:
                    raise ScoringSupportError(
                        f"metric {metric_id!r} has inconsistent bindings"
                    )
                payloads.append(
                    (
                        support.targets[quantity_id],
                        aligned[quantity_id],
                        support.weights,
                    )
                )
            count = len(support.support_ids)
            support_records.append(
                {
                    "support_id": support_name,
                    "support_count": count,
                    "scored_count": count,
                    "coverage_fraction": 1.0,
                    "weight_coverage_fraction": 1.0,
                    "unmapped_count": 0,
                    "extrapolated_count": 0,
                    "metric_values": metric_values,
                    "metric_sufficient_statistics": metric_sufficient_statistics,
                }
            )
        case_records.append({"case_id": case_id, "supports": support_records})

    aggregate = {
        metric_id: aggregate_metric(payloads, binding)
        for metric_id, (binding, payloads) in aggregate_payloads.items()
    }
    return {
        "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": release.manifest["dataset_id"],
        "split_id": split_id,
        "case_set_id": case_set_id,
        "scoring_support_release_id": release.manifest["release_id"],
        "scoring_support_manifest_sha256": sha256_file(support_manifest_path),
        "case_count": len(case_records),
        "cases": case_records,
        "metric_values": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--case-set", required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate_prediction_artifact(
            support_manifest_path=args.support_manifest,
            case_set_id=args.case_set,
            prediction_manifest_path=args.prediction_manifest,
            submission_id=args.submission_id,
            split_id=args.split_id,
        )
    except ScoringSupportError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"{json.dumps(result, indent=2, ensure_ascii=True)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
