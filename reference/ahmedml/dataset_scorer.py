"""Candidate dataset reductions for complete AhmedML case evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from reference.scores import (
    composite_component_group_scores,
    composite_overall_score,
)

from .contract import (
    DATASET_ID,
    PROFILE_DEFINITION_SHA256,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
)
from .evaluator import EVIDENCE_SCHEMA, EVIDENCE_STATUS
from .support import (
    SURFACE_COORDINATE_INTERVALS,
    SURFACE_STATIONS,
    VOLUME_COORDINATE_INTERVALS,
    VOLUME_STATIONS,
)


DATASET_EVIDENCE_SCHEMA = "ahmedml-candidate-dataset-evaluation-v1"
DATASET_EVIDENCE_STATUS = "non_ranked_development_evidence_not_official_submission"
PRIMARY_FIELD_METRICS = (
    "surface_pressure_rel_l2",
    "surface_wall_shear_rel_l2",
    "volume_velocity_rel_l2",
    "volume_pressure_rel_l2",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class AhmedMLDatasetScorerError(ValueError):
    """Raised when case evidence cannot be reduced unambiguously."""


@dataclass(frozen=True)
class CandidateDatasetEvaluation:
    value: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        return dict(self.value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AhmedMLDatasetScorerError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AhmedMLDatasetScorerError(
                    f"{label} contains forbidden non-finite token {token}"
                )
            ),
        )
    except AhmedMLDatasetScorerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AhmedMLDatasetScorerError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise AhmedMLDatasetScorerError(f"{label} must contain an object")
    return value, hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AhmedMLDatasetScorerError(f"{label} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AhmedMLDatasetScorerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AhmedMLDatasetScorerError(f"{label} must be finite")
    return result


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AhmedMLDatasetScorerError(f"{label} must be a lowercase SHA-256")
    return value


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
        raise AhmedMLDatasetScorerError(f"{label} arrays are invalid")
    residual = float(
        np.sum((prediction_array - truth_array) ** 2, dtype=np.float64)
    )
    centred = truth_array - float(np.mean(truth_array, dtype=np.float64))
    denominator = float(np.sum(centred * centred, dtype=np.float64))
    if denominator <= 0.0:
        raise AhmedMLDatasetScorerError(f"{label} ground truth is constant")
    result = 1.0 - residual / denominator
    if not math.isfinite(result):
        raise AhmedMLDatasetScorerError(f"{label} R2 is non-finite")
    return result


def _series(
    evidence: Mapping[str, Any],
    *,
    case_id: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    profiles = _mapping(evidence.get("profiles"), f"{case_id}.profiles")
    if (
        profiles.get("profile_definition_sha256") != PROFILE_DEFINITION_SHA256
        or profiles.get("participant_profile_payload_accepted") is not False
    ):
        raise AhmedMLDatasetScorerError(f"{case_id} profile contract differs")
    raw_series = profiles.get("series")
    if not isinstance(raw_series, list) or len(raw_series) != 7:
        raise AhmedMLDatasetScorerError(f"{case_id} must contain exactly seven series")
    expected = (
        *(
            ("pressure_profiles", station, "cp", interval)
            for station, interval in zip(
                SURFACE_STATIONS, SURFACE_COORDINATE_INTERVALS, strict=True
            )
        ),
        *(
            ("velocity_profiles", station, "ux_over_uinf", interval)
            for station, interval in zip(
                VOLUME_STATIONS, VOLUME_COORDINATE_INTERVALS, strict=True
            )
        ),
    )
    cp_truth: list[float] = []
    cp_prediction: list[float] = []
    velocity_truth: list[float] = []
    velocity_prediction: list[float] = []
    for index, (raw, identity) in enumerate(zip(raw_series, expected, strict=True)):
        item = _mapping(raw, f"{case_id}.profiles.series[{index}]")
        panel_id, station_id, quantity_id, coordinate_interval = identity
        if (
            item.get("panel_id") != panel_id
            or item.get("station_id") != station_id
            or item.get("quantity_id") != quantity_id
            or item.get("sample_count") != 128
            or item.get("source") != "evaluator_derived_from_complete_native_fields"
        ):
            raise AhmedMLDatasetScorerError(
                f"{case_id} profile series {index} identity differs"
            )
        coordinate = item.get("coordinate")
        truth = item.get("truth")
        prediction = item.get("prediction")
        if not all(isinstance(value, list) and len(value) == 128 for value in (coordinate, truth, prediction)):
            raise AhmedMLDatasetScorerError(
                f"{case_id} profile series {index} must contain 128 samples"
            )
        coordinate_values = [_finite(value, "profile coordinate") for value in coordinate]
        truth_values = [_finite(value, "profile truth") for value in truth]
        prediction_values = [_finite(value, "profile prediction") for value in prediction]
        if any(
            right <= left
            for left, right in zip(coordinate_values, coordinate_values[1:])
        ):
            raise AhmedMLDatasetScorerError("profile coordinates must increase strictly")
        expected_coordinate = np.linspace(
            coordinate_interval[0],
            coordinate_interval[1],
            128,
            dtype=np.float64,
        )
        if not np.allclose(
            np.asarray(coordinate_values, dtype=np.float64),
            expected_coordinate,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AhmedMLDatasetScorerError(
                f"{case_id} profile series {index} differs from its exact 128-point grid"
            )
        if panel_id == "pressure_profiles":
            cp_truth.extend(truth_values)
            cp_prediction.extend(prediction_values)
        else:
            velocity_truth.extend(truth_values)
            velocity_prediction.extend(prediction_values)
    return cp_truth, cp_prediction, velocity_truth, velocity_prediction


def score_candidate_dataset(
    *,
    submission_specification: str | Path,
    split_id: str,
    case_evidence_directory: str | Path,
) -> CandidateDatasetEvaluation:
    """Reduce exact case evidence for one declared AhmedML test split."""

    spec_path = Path(submission_specification).expanduser().resolve()
    specification, specification_sha = _read_json(
        spec_path, "AhmedML submission specification"
    )
    if (
        specification.get("dataset_id") != DATASET_ID
        or specification.get("status") != "candidate_native_support"
        or specification.get("evaluation_reference_version")
        != "ahmedml-evaluator-v0.1-candidate"
    ):
        raise AhmedMLDatasetScorerError("submission specification is not the candidate AhmedML contract")
    scoring_support = _mapping(
        specification.get("scoring_support"), "scoring_support"
    )
    if scoring_support.get("submissions_open") is not False:
        raise AhmedMLDatasetScorerError("candidate AhmedML submissions must remain closed")
    source = _mapping(scoring_support.get("dataset_source"), "dataset_source")
    if (
        source.get("revision") != REPOSITORY_REVISION
        or source.get("identity_sha256") != SOURCE_IDENTITY_SHA256
    ):
        raise AhmedMLDatasetScorerError("candidate source identity differs")
    declarations = specification.get("splits")
    if not isinstance(declarations, list):
        raise AhmedMLDatasetScorerError("submission specification has no splits")
    matches = [
        item
        for item in declarations
        if isinstance(item, Mapping) and item.get("id") == split_id
    ]
    if len(matches) != 1:
        raise AhmedMLDatasetScorerError(f"split {split_id!r} is not unique")
    declaration = matches[0]
    split_path = (spec_path.parent / str(declaration.get("index_file"))).resolve()
    if not split_path.is_relative_to(spec_path.parent):
        raise AhmedMLDatasetScorerError("split index escapes the specification directory")
    split, split_sha = _read_json(split_path, "AhmedML split index")
    if split_sha != _digest(declaration.get("sha256"), "split SHA-256"):
        raise AhmedMLDatasetScorerError("split index SHA-256 differs")
    case_ids = split.get("case_ids")
    if (
        split.get("dataset_id") != DATASET_ID
        or split.get("split_id") != split_id
        or not isinstance(case_ids, list)
        or not case_ids
        or len(case_ids) != len(set(case_ids))
        or declaration.get("case_count") != len(case_ids)
        or split.get("case_count") != len(case_ids)
    ):
        raise AhmedMLDatasetScorerError("split index is inconsistent")

    evidence_root = Path(case_evidence_directory).expanduser().resolve()
    case_documents: list[Mapping[str, Any]] = []
    evidence_inputs: list[dict[str, str]] = []
    field_values: dict[str, list[float]] = {}
    cd_truth: list[float] = []
    cd_prediction: list[float] = []
    cl_truth: list[float] = []
    cl_prediction: list[float] = []
    cp_truth: list[float] = []
    cp_prediction: list[float] = []
    velocity_truth: list[float] = []
    velocity_prediction: list[float] = []
    for case_id in case_ids:
        if not isinstance(case_id, str):
            raise AhmedMLDatasetScorerError("split case IDs must be strings")
        path = evidence_root / f"{case_id}.json"
        document, digest = _read_json(path, f"{case_id} evidence")
        if (
            document.get("schema") != EVIDENCE_SCHEMA
            or document.get("schema_version") != 1
            or document.get("status") != EVIDENCE_STATUS
            or document.get("official_submission") is not False
            or document.get("leaderboard_eligible") is not False
            or document.get("case_id") != case_id
        ):
            raise AhmedMLDatasetScorerError(f"{case_id} evidence header differs")
        case_source = _mapping(document.get("source"), f"{case_id}.source")
        if (
            case_source.get("repository_revision") != REPOSITORY_REVISION
            or case_source.get("source_identity_sha256") != SOURCE_IDENTITY_SHA256
        ):
            raise AhmedMLDatasetScorerError(f"{case_id} source identity differs")
        coverage = _mapping(document.get("coverage"), f"{case_id}.coverage")
        if (
            coverage.get("complete_case") is not True
            or coverage.get("gap_free_duplicate_free") is not True
        ):
            raise AhmedMLDatasetScorerError(f"{case_id} coverage is incomplete")
        metrics = _mapping(document.get("metric_values"), f"{case_id}.metric_values")
        for metric_id, value in metrics.items():
            field_values.setdefault(metric_id, []).append(
                _finite(value, f"{case_id}.{metric_id}")
            )
        force = _mapping(document.get("force_coefficients"), f"{case_id}.force")
        cd_truth.append(_finite(force.get("truth_cd"), f"{case_id}.truth_cd"))
        cd_prediction.append(
            _finite(force.get("prediction_cd"), f"{case_id}.prediction_cd")
        )
        cl_truth.append(_finite(force.get("truth_cl"), f"{case_id}.truth_cl"))
        cl_prediction.append(
            _finite(force.get("prediction_cl"), f"{case_id}.prediction_cl")
        )
        case_cp_truth, case_cp_prediction, case_u_truth, case_u_prediction = _series(
            document, case_id=case_id
        )
        cp_truth.extend(case_cp_truth)
        cp_prediction.extend(case_cp_prediction)
        velocity_truth.extend(case_u_truth)
        velocity_prediction.extend(case_u_prediction)
        evidence_inputs.append(
            {"case_id": case_id, "file": path.name, "sha256": digest}
        )
        case_documents.append(document)

    if any(len(values) != len(case_ids) for values in field_values.values()):
        raise AhmedMLDatasetScorerError("case field metric sets differ")
    metric_values = {
        metric_id: float(math.fsum(values) / len(values))
        for metric_id, values in sorted(field_values.items())
    }
    missing_primary = set(PRIMARY_FIELD_METRICS) - set(metric_values)
    if missing_primary:
        raise AhmedMLDatasetScorerError(
            f"case evidence lacks primary fields {sorted(missing_primary)}"
        )
    metric_values.update(
        {
            "cd_r2": _r2(cd_truth, cd_prediction, "Cd"),
            "cl_r2": _r2(cl_truth, cl_prediction, "Cl"),
            "c_drag_mae": float(
                np.mean(np.abs(np.asarray(cd_prediction) - np.asarray(cd_truth)))
            ),
            "c_lift_mae": float(
                np.mean(np.abs(np.asarray(cl_prediction) - np.asarray(cl_truth)))
            ),
            "cp_cut_r2": _r2(cp_truth, cp_prediction, "Cp profiles"),
            "velocity_profile_r2": _r2(
                velocity_truth, velocity_prediction, "velocity profiles"
            ),
        }
    )
    overall = _mapping(
        specification.get("overall_score_composite"), "overall_score_composite"
    )
    groups = _mapping(
        specification.get("component_score_groups"), "component_score_groups"
    )
    try:
        metric_values.update(
            composite_component_group_scores(metric_values, overall, groups)
        )
        metric_values["overall_score"] = composite_overall_score(
            metric_values, overall
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AhmedMLDatasetScorerError(
            f"cannot apply AhmedML composite score: {error}"
        ) from error

    result: dict[str, object] = {
        "schema": DATASET_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "status": DATASET_EVIDENCE_STATUS,
        "official_submission": False,
        "leaderboard_eligible": False,
        "dataset_id": DATASET_ID,
        "dataset_version": specification.get("dataset_version"),
        "split_id": split_id,
        "case_set_id": split.get("case_set_id"),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "contract": {
            "submission_specification_sha256": specification_sha,
            "split_index_sha256": split_sha,
            "source_identity_sha256": SOURCE_IDENTITY_SHA256,
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
        },
        "metric_values": metric_values,
        "reductions": {
            "spatial_fields": "calculate_each_case_then_macro_average_cases_equally",
            "forces": "global_R2_and_case_equal_MAE",
            "profiles": "flatten_all_required_evaluator_derived_samples_then_global_R2",
            "regional_diagnostics": "report_only_not_in_score",
        },
        "case_evidence": evidence_inputs,
    }
    return CandidateDatasetEvaluation(MappingProxyType(result))


__all__ = [
    "AhmedMLDatasetScorerError",
    "CandidateDatasetEvaluation",
    "DATASET_EVIDENCE_SCHEMA",
    "DATASET_EVIDENCE_STATUS",
    "score_candidate_dataset",
]
