"""Reduce per-case WindsorML evidence into one declared test split's score.

Field metrics are macro-averaged over cases (each case counts equally, per the
dataset's ``case_aggregation``). Coefficients are pooled across the split and
reduced to R^2, because R^2 is only meaningful over a population.

The score transforms are not reimplemented here: they come from
:mod:`reference.scores`, so WindsorML cannot silently disagree with DrivAerML,
AhmedML, or HiLiftAeroML about what ``bounded_error`` means.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from reference.scores import composite_component_group_scores, composite_overall_score

from .contract import read_json


DATASET_EVIDENCE_SCHEMA = "windsorml-candidate-dataset-evaluation-v1"
DATASET_EVIDENCE_STATUS = "non_ranked_development_evidence_not_official_submission"

# Macro-averaged per-case field metrics, read from each case's evidence.
SURFACE_METRICS = (
    "surface_pressure_rel_l2",
    "surface_wall_shear_rel_l2",
)
VOLUME_METRICS = (
    "volume_velocity_rel_l2",
    "volume_pressure_rel_l2",
)

# Profile R^2 is pooled over every sample of every station in a family, then
# reduced once. The constant-placement families carry the ranked weight; the
# relative families are reported at zero weight so a reviewer can compare them
# without either being counted twice.
PROFILE_FAMILY_METRICS = {
    "windsorml_velocity_constant_v1": "velocity_profile_r2",
    "windsorml_cp_constant_v1": "cp_cut_r2",
    "windsorml_velocity_relative_v1": "velocity_profile_relative_r2",
    "windsorml_cp_relative_v1": "cp_cut_relative_r2",
}
RANKED_PROFILE_METRICS = ("velocity_profile_r2", "cp_cut_r2")


class WindsorMLDatasetScorerError(ValueError):
    """Raised when a WindsorML split cannot be scored exactly."""


@dataclass(frozen=True)
class CandidateDatasetEvaluation:
    """One split's reduced evidence, explicitly ineligible for ranking."""

    value: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        return dict(self.value)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WindsorMLDatasetScorerError(f"{label} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WindsorMLDatasetScorerError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise WindsorMLDatasetScorerError(f"{label} must be finite")
    return result


def _r2(truth: Sequence[float], prediction: Sequence[float], label: str) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if (
        truth_array.ndim != 1
        or prediction_array.shape != truth_array.shape
        or len(truth_array) < 2
        or np.any(~np.isfinite(truth_array))
        or np.any(~np.isfinite(prediction_array))
    ):
        raise WindsorMLDatasetScorerError(f"{label} arrays are invalid")
    residual = float(np.sum((prediction_array - truth_array) ** 2, dtype=np.float64))
    centred = truth_array - float(np.mean(truth_array, dtype=np.float64))
    denominator = float(np.sum(centred * centred, dtype=np.float64))
    if denominator <= 0.0:
        raise WindsorMLDatasetScorerError(f"{label} ground truth is constant")
    result = 1.0 - residual / denominator
    if not math.isfinite(result):
        raise WindsorMLDatasetScorerError(f"{label} R2 is non-finite")
    return result


def _mae(truth: Sequence[float], prediction: Sequence[float]) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction, dtype=np.float64)
                - np.asarray(truth, dtype=np.float64)
            ),
            dtype=np.float64,
        )
    )


def score_candidate_dataset(
    *,
    submission_specification: str | Path,
    split_id: str,
    case_evidence_directory: str | Path,
) -> CandidateDatasetEvaluation:
    """Reduce exact case evidence for one declared WindsorML test split."""

    spec_path = Path(submission_specification).expanduser().resolve()
    specification = read_json(spec_path, label="WindsorML submission specification")
    if specification.get("dataset_id") != "windsorml":
        raise WindsorMLDatasetScorerError("specification is not the WindsorML spec")

    splits = {entry["id"]: entry for entry in specification.get("splits", [])}
    if split_id not in splits:
        raise WindsorMLDatasetScorerError(
            f"unknown split {split_id!r}; expected one of {sorted(splits)}"
        )
    split_entry = splits[split_id]
    split_document = read_json(
        spec_path.parent / split_entry["index_file"], label=f"{split_id} split"
    )
    case_ids = list(split_document["case_ids"])
    if len(case_ids) != split_entry["case_count"]:
        raise WindsorMLDatasetScorerError(f"{split_id} case count differs from the spec")

    scored_profile_families = {
        family
        for family, metric in PROFILE_FAMILY_METRICS.items()
        if metric in {m["id"] for m in specification.get("metrics", [])}
    }

    root = Path(case_evidence_directory).expanduser().resolve()
    per_case: dict[str, Mapping[str, float]] = {}
    cd_truth: list[float] = []
    cd_prediction: list[float] = []
    cl_truth: list[float] = []
    cl_prediction: list[float] = []
    profile_truth: dict[str, list[float]] = {f: [] for f in PROFILE_FAMILY_METRICS}
    profile_prediction: dict[str, list[float]] = {f: [] for f in PROFILE_FAMILY_METRICS}

    for case_id in case_ids:
        path = root / f"{case_id}.json"
        if not path.is_file():
            raise WindsorMLDatasetScorerError(f"missing case evidence for {case_id}")
        evidence = read_json(path, label=f"{case_id} evidence")
        if evidence.get("case_id") != case_id:
            raise WindsorMLDatasetScorerError(f"{path} does not bind {case_id}")
        if evidence.get("official_submission_artifact") is not False:
            raise WindsorMLDatasetScorerError(
                f"{case_id} evidence is not marked candidate-only"
            )

        surface = _mapping(evidence.get("surface"), f"{case_id}.surface")
        surface_metrics = _mapping(surface.get("metrics"), f"{case_id}.surface.metrics")
        volume = evidence.get("volume")
        if volume is None:
            raise WindsorMLDatasetScorerError(
                f"{case_id} has no volume evidence; the split scores volume metrics"
            )
        volume_metrics = _mapping(
            _mapping(volume, f"{case_id}.volume").get("metrics"),
            f"{case_id}.volume.metrics",
        )

        values: dict[str, float] = {}
        for name in SURFACE_METRICS:
            values[name] = _finite(surface_metrics.get(name), f"{case_id}.{name}")
        for name in VOLUME_METRICS:
            values[name] = _finite(volume_metrics.get(name), f"{case_id}.{name}")
        per_case[case_id] = MappingProxyType(values)

        forces = _mapping(evidence.get("forces"), f"{case_id}.forces")
        truth = _mapping(forces.get("truth_integrated"), f"{case_id}.forces.truth")
        prediction = _mapping(
            forces.get("prediction_integrated"), f"{case_id}.forces.prediction"
        )
        cd_truth.append(_finite(truth.get("cd"), f"{case_id}.truth.cd"))
        cd_prediction.append(_finite(prediction.get("cd"), f"{case_id}.prediction.cd"))
        cl_truth.append(_finite(truth.get("cl"), f"{case_id}.truth.cl"))
        cl_prediction.append(_finite(prediction.get("cl"), f"{case_id}.prediction.cl"))

        if scored_profile_families:
            profiles = evidence.get("profiles")
            if profiles is None:
                raise WindsorMLDatasetScorerError(
                    f"{case_id} has no profile evidence but the spec scores profiles"
                )
            if _mapping(profiles, f"{case_id}.profiles").get(
                "participant_profile_payload_accepted"
            ) is not False:
                raise WindsorMLDatasetScorerError(
                    f"{case_id} profiles must be evaluator-derived, never submitted"
                )
            families = _mapping(profiles.get("families"), f"{case_id}.profiles.families")
            for family in PROFILE_FAMILY_METRICS:
                if family not in families:
                    raise WindsorMLDatasetScorerError(
                        f"{case_id} profile family {family!r} is missing"
                    )
                for station in families[family]:
                    item = _mapping(station, f"{case_id}.{family}")
                    truth_values = item.get("truth")
                    prediction_values = item.get("prediction")
                    if not isinstance(truth_values, list) or not isinstance(
                        prediction_values, list
                    ):
                        raise WindsorMLDatasetScorerError(
                            f"{case_id} {family} series are malformed"
                        )
                    if len(truth_values) != len(prediction_values):
                        raise WindsorMLDatasetScorerError(
                            f"{case_id} {family} truth and prediction differ in length"
                        )
                    profile_truth[family].extend(
                        _finite(v, "profile truth") for v in truth_values
                    )
                    profile_prediction[family].extend(
                        _finite(v, "profile prediction") for v in prediction_values
                    )

    metric_values: dict[str, float] = {}
    for name in (*SURFACE_METRICS, *VOLUME_METRICS):
        metric_values[name] = float(
            np.mean([per_case[c][name] for c in case_ids], dtype=np.float64)
        )
    metric_values["cd_r2"] = _r2(cd_truth, cd_prediction, "cd")
    metric_values["cl_r2"] = _r2(cl_truth, cl_prediction, "cl")
    metric_values["c_drag_mae"] = _mae(cd_truth, cd_prediction)
    metric_values["c_lift_mae"] = _mae(cl_truth, cl_prediction)

    for family, metric_id in PROFILE_FAMILY_METRICS.items():
        if not profile_truth[family]:
            continue
        metric_values[metric_id] = _r2(
            profile_truth[family], profile_prediction[family], metric_id
        )

    composite = specification["overall_score_composite"]
    overall = composite_overall_score(metric_values, composite)
    groups = composite_component_group_scores(
        metric_values, composite, specification["component_score_groups"]
    )
    metric_values["overall_score"] = overall
    metric_values.update(groups)

    return CandidateDatasetEvaluation(
        value=MappingProxyType(
            {
                "schema": DATASET_EVIDENCE_SCHEMA,
                "status": DATASET_EVIDENCE_STATUS,
                "official_submission_artifact": False,
                "dataset_id": "windsorml",
                "dataset_version": specification.get("dataset_version"),
                "split_id": split_id,
                "case_count": len(case_ids),
                "official_case_count": split_document.get("official_case_count"),
                "excluded_case_ids": list(split_document.get("excluded_case_ids") or []),
                "metric_values": dict(metric_values),
                "per_case": {c: dict(per_case[c]) for c in case_ids},
            }
        )
    )


__all__ = [
    "CandidateDatasetEvaluation",
    "DATASET_EVIDENCE_SCHEMA",
    "DATASET_EVIDENCE_STATUS",
    "SURFACE_METRICS",
    "VOLUME_METRICS",
    "WindsorMLDatasetScorerError",
    "score_candidate_dataset",
]
