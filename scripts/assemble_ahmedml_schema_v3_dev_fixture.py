#!/usr/bin/env python3
"""Assemble the non-ranked AhmedML schema-v3 development fixture.

The fixture predictions are deterministic transforms of public CFD truth.  A
legacy GeoTransolver checkpoint supplies error-scale calibration only and is
never executed.  The resulting package exists to exercise the exact evaluator,
profile, discretization, regional-diagnostic, and dashboard paths; it is not a
scientific model submission and cannot be ranked or promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    DATASET_VERSION,
    PROFILE_DEFINITION_SHA256,
    REGION_DEFINITION_SHA256,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
)
from reference.ahmedml.dataset_scorer import score_candidate_dataset  # noqa: E402
from reference.ahmedml.regional_aggregate import (  # noqa: E402
    AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DEFINITION_ID,
    build_aggregate_regional_diagnostics,
)
from reference.scoring_support import load_support_release, sha256_file  # noqa: E402


SUBMISSION_ID = "ahmedml-geotransolver-calibrated-dev-fixture-v1"
SERIES_ID = "ahmedml-geotransolver-calibrated-dev-fixture"
SUPPORT_RELEASE_ID = "ahmedml-native-all-splits-support-v1-candidate"
EVALUATOR_VERSION = "ahmedml-evaluator-v0.1-candidate"
CHECKPOINT_SHA256 = "37ee4418ef1ee069b5c086322c4c5ef9c35f1a59ea4755b470b7d6358c642ac5"
CHECKPOINT_SIZE = 80_893_878
CHECKPOINT_PARAMETER_COUNT = 20_166_852
PROFILE_FORMAT = "fluidsbench-profile-chunks-v1"


class AssemblyError(ValueError):
    """Raised when fixture inputs cannot form an exact, honest package."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssemblyError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} must contain an object")
    return value


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(value, sort_keys=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for value in values
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AssemblyError(f"{label} must be finite")
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{label} must be an object")
    return value


def _r2(truth: Sequence[float], prediction: Sequence[float]) -> float:
    if len(truth) != len(prediction) or len(truth) < 2:
        raise AssemblyError("R2 arrays differ or are too short")
    truth_mean = math.fsum(truth) / len(truth)
    numerator = math.fsum((left - right) ** 2 for left, right in zip(truth, prediction, strict=True))
    denominator = math.fsum((value - truth_mean) ** 2 for value in truth)
    if denominator <= 0.0:
        raise AssemblyError("R2 truth is constant")
    return 1.0 - numerator / denominator


def _case_profiles(evidence: Mapping[str, Any], case_id: str) -> tuple[list[dict[str, Any]], float, float]:
    profiles = _mapping(evidence.get("profiles"), f"{case_id}.profiles")
    if (
        profiles.get("profile_definition_sha256") != PROFILE_DEFINITION_SHA256
        or profiles.get("participant_profile_payload_accepted") is not False
    ):
        raise AssemblyError(f"{case_id} profile contract differs")
    raw_series = profiles.get("series")
    if not isinstance(raw_series, list) or len(raw_series) != 7:
        raise AssemblyError(f"{case_id} must contain seven evaluator-derived series")
    output: list[dict[str, Any]] = []
    cp_truth: list[float] = []
    cp_prediction: list[float] = []
    velocity_truth: list[float] = []
    velocity_prediction: list[float] = []
    for index, raw in enumerate(raw_series):
        series = _mapping(raw, f"{case_id}.profiles.series[{index}]")
        coordinate = series.get("coordinate")
        truth = series.get("truth")
        prediction = series.get("prediction")
        if (
            series.get("sample_count") != 128
            or series.get("source") != "evaluator_derived_from_complete_native_fields"
            or not all(isinstance(values, list) and len(values) == 128 for values in (coordinate, truth, prediction))
        ):
            raise AssemblyError(f"{case_id} profile series {index} is not exact 128-point evaluator output")
        coordinates = [_finite(value, "profile coordinate") for value in coordinate]
        truths = [_finite(value, "profile truth") for value in truth]
        predictions = [_finite(value, "profile prediction") for value in prediction]
        if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
            raise AssemblyError(f"{case_id} profile coordinates must increase")
        panel_id = str(series.get("panel_id"))
        output.append(
            {
                "panel_id": panel_id,
                "station_id": series.get("station_id"),
                "quantity_id": series.get("quantity_id"),
                "coordinate": coordinates,
                "prediction": predictions,
            }
        )
        if panel_id == "pressure_profiles":
            cp_truth.extend(truths)
            cp_prediction.extend(predictions)
        elif panel_id == "velocity_profiles":
            velocity_truth.extend(truths)
            velocity_prediction.extend(predictions)
        else:
            raise AssemblyError(f"{case_id} profile panel differs")
    return output, _r2(cp_truth, cp_prediction), _r2(velocity_truth, velocity_prediction)


def _support_counts(release: Any, case_id: str) -> dict[str, int]:
    case = release.cases.get(case_id)
    if not isinstance(case, Mapping):
        raise AssemblyError(f"scoring support lacks {case_id}")
    result: dict[str, int] = {}
    for instance in case.get("support_instances", []):
        if not isinstance(instance, Mapping):
            continue
        support_id = instance.get("support_id")
        count = instance.get("entity_count")
        if not isinstance(support_id, str) or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise AssemblyError(f"{case_id} has an invalid support instance")
        if support_id in result:
            raise AssemblyError(f"{case_id} repeats support {support_id}")
        result[support_id] = count
    if set(result) != set(release.supports):
        raise AssemblyError(f"{case_id} support set differs")
    return result


def _relative_l2_statistics(
    *,
    field: Mapping[str, Any],
    weighting: str,
    dataset_weighting: str,
    entity_count: int,
) -> dict[str, Any]:
    statistics = _mapping(field.get(weighting), f"field.{weighting}")
    return {
        "reduction": "relative_l2_percent",
        "weighting": "uniform" if weighting == "uniform" else "support_weights",
        "dataset_weighting": dataset_weighting,
        "numerator": _finite(statistics.get("squared_error"), "squared_error"),
        "denominator": _finite(statistics.get("squared_truth"), "squared_truth"),
        "entity_count": entity_count,
        "total_weight": _finite(statistics.get("total_weight"), "total_weight"),
    }


def _support_record(
    *,
    support_id: str,
    count: int,
    metric_values: Mapping[str, float],
    statistics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "support_id": support_id,
        "support_count": count,
        "scored_count": count,
        "coverage_fraction": 1.0,
        "weight_coverage_fraction": 1.0,
        "unmapped_count": 0,
        "extrapolated_count": 0,
        "metric_values": dict(metric_values),
        "metric_sufficient_statistics": dict(statistics),
    }


def _case_metrics(
    evidence: Mapping[str, Any],
    *,
    case_id: str,
    counts: Mapping[str, int],
    cp_r2: float,
    velocity_r2: float,
) -> dict[str, Any]:
    values = _mapping(evidence.get("metric_values"), f"{case_id}.metric_values")
    fields = _mapping(evidence.get("field_statistics"), f"{case_id}.field_statistics")
    force = _mapping(evidence.get("force_coefficients"), f"{case_id}.force_coefficients")

    surface_metrics = {
        metric_id: _finite(values.get(metric_id), f"{case_id}.{metric_id}")
        for metric_id in (
            "surface_pressure_rel_l2",
            "surface_pressure_equal_entity_rel_l2",
            "surface_wall_shear_rel_l2",
            "surface_wall_shear_equal_entity_rel_l2",
            "surface_pressure_rel_l1",
            "surface_wall_shear_rel_l1",
            "surface_pressure_mae",
            "surface_pressure_rmse",
            "surface_wall_shear_mae",
            "surface_wall_shear_rmse",
        )
    }
    volume_metrics = {
        metric_id: _finite(values.get(metric_id), f"{case_id}.{metric_id}")
        for metric_id in (
            "volume_pressure_rel_l2",
            "volume_pressure_physical_rel_l2",
            "volume_velocity_rel_l2",
            "volume_velocity_physical_rel_l2",
            "volume_pressure_rel_l1",
            "volume_velocity_rel_l1",
            "volume_pressure_mae",
            "volume_pressure_rmse",
            "volume_velocity_mae",
            "volume_velocity_rmse",
        )
    }
    surface_count = counts["ahmedml-surface-native-cells-v1"]
    volume_count = counts["ahmedml-volume-native-cells-v1"]
    surface_pressure = _mapping(fields.get("surface_pressure"), "surface_pressure")
    surface_shear = _mapping(fields.get("surface_wall_shear"), "surface_wall_shear")
    volume_pressure = _mapping(fields.get("volume_pressure"), "volume_pressure")
    volume_velocity = _mapping(fields.get("volume_velocity"), "volume_velocity")
    surface_statistics = {
        "surface_pressure_rel_l2": _relative_l2_statistics(
            field=surface_pressure,
            weighting="physical",
            dataset_weighting="surface_face_area",
            entity_count=surface_count,
        ),
        "surface_pressure_equal_entity_rel_l2": _relative_l2_statistics(
            field=surface_pressure,
            weighting="uniform",
            dataset_weighting="surface_entities_equal",
            entity_count=surface_count,
        ),
        "surface_wall_shear_rel_l2": _relative_l2_statistics(
            field=surface_shear,
            weighting="physical",
            dataset_weighting="surface_face_area",
            entity_count=surface_count,
        ),
        "surface_wall_shear_equal_entity_rel_l2": _relative_l2_statistics(
            field=surface_shear,
            weighting="uniform",
            dataset_weighting="surface_entities_equal",
            entity_count=surface_count,
        ),
    }
    volume_statistics = {
        "volume_pressure_rel_l2": _relative_l2_statistics(
            field=volume_pressure,
            weighting="uniform",
            dataset_weighting="volume_cells_equal",
            entity_count=volume_count,
        ),
        "volume_pressure_physical_rel_l2": _relative_l2_statistics(
            field=volume_pressure,
            weighting="physical",
            dataset_weighting="cell_volume",
            entity_count=volume_count,
        ),
        "volume_velocity_rel_l2": _relative_l2_statistics(
            field=volume_velocity,
            weighting="uniform",
            dataset_weighting="volume_cells_equal",
            entity_count=volume_count,
        ),
        "volume_velocity_physical_rel_l2": _relative_l2_statistics(
            field=volume_velocity,
            weighting="physical",
            dataset_weighting="cell_volume",
            entity_count=volume_count,
        ),
    }
    truth_cd = _finite(force.get("truth_cd"), f"{case_id}.truth_cd")
    prediction_cd = _finite(force.get("prediction_cd"), f"{case_id}.prediction_cd")
    truth_cl = _finite(force.get("truth_cl"), f"{case_id}.truth_cl")
    prediction_cl = _finite(force.get("prediction_cl"), f"{case_id}.prediction_cl")
    drag_mae = abs(prediction_cd - truth_cd)
    lift_mae = abs(prediction_cl - truth_cl)
    return {
        "case_id": case_id,
        "supports": [
            _support_record(
                support_id="ahmedml-surface-native-cells-v1",
                count=surface_count,
                metric_values=surface_metrics,
                statistics=surface_statistics,
            ),
            _support_record(
                support_id="ahmedml-volume-native-cells-v1",
                count=volume_count,
                metric_values=volume_metrics,
                statistics=volume_statistics,
            ),
            _support_record(
                support_id="ahmedml-force-coefficients-v1",
                count=counts["ahmedml-force-coefficients-v1"],
                metric_values={"c_drag_mae": drag_mae, "c_lift_mae": lift_mae},
                statistics={},
            ),
            _support_record(
                support_id="ahmedml-surface-cp-profiles-v1",
                count=counts["ahmedml-surface-cp-profiles-v1"],
                metric_values={},
                statistics={},
            ),
            _support_record(
                support_id="ahmedml-volume-velocity-profiles-v1",
                count=counts["ahmedml-volume-velocity-profiles-v1"],
                metric_values={},
                statistics={},
            ),
        ],
        "nonspatial_metric_values": {
            "c_drag_mae": drag_mae,
            "c_lift_mae": lift_mae,
            "cp_cut_r2": cp_r2,
            "velocity_profile_r2": velocity_r2,
        },
        "force_coefficients": {
            "truth_cd": truth_cd,
            "prediction_cd": prediction_cd,
            "truth_cl": truth_cl,
            "prediction_cl": prediction_cl,
        },
    }


def _count_summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "kind": "per_case",
        "minimum": min(values),
        "median": float(median(values)),
        "maximum": max(values),
    }


def _representation_summary(
    *,
    description: str,
    entity: str,
    counts: Sequence[int],
    case_record_id: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "used": True,
        "representation": description,
        "entity_counts": [{"entity": entity, "count": _count_summary(counts)}],
        "native_comparison": {
            "status": "reported",
            "native_entity_counts": [{"entity": entity, "count": _count_summary(counts)}],
            "fractions": [{"entity": entity, "fraction": {"kind": "fixed", "value": 1.0}}],
        },
        "sampling": {"kind": "none"},
        "domain": {"kind": "full_dataset_domain"},
        "connectivity": "native",
    }
    if case_record_id is not None:
        value["case_record_id"] = case_record_id
    return value


def _case_representation(identifier: str, entity: str, count: int) -> dict[str, Any]:
    return {
        "id": identifier,
        "entity_counts": [{"entity": entity, "count": count}],
        "native_entity_counts": [{"entity": entity, "count": count}],
        "native_fractions": [{"entity": entity, "fraction": 1.0}],
        "domain": {"kind": "full_dataset_domain"},
    }


def _mapping_summary(support_id: str, source_output_id: str, *, derived: bool) -> dict[str, Any]:
    method = (
        {
            "kind": "reference_rule",
            "rule_id": "ahmedml-evaluator-derived-support-v1",
            "rule_version": "candidate-v1",
        }
        if derived
        else {"kind": "identity"}
    )
    return {
        "support_id": support_id,
        "source_output_id": source_output_id,
        "method": method,
        "implementation": (
            "Dataset evaluator derives force or profile values from the same complete native prediction fields."
            if derived
            else "Predictions retain the exact native CellData entity order."
        ),
        "extrapolation_policy": "forbidden",
        "unmapped_fraction": 0.0,
        "extrapolated_fraction": 0.0,
        "final_coverage_fraction": 1.0,
    }


def _case_mapping(support_id: str, source_output_id: str, count: int) -> dict[str, Any]:
    return {
        "support_id": support_id,
        "source_output_id": source_output_id,
        "support_count": count,
        "scored_count": count,
        "unmapped_count": 0,
        "extrapolated_count": 0,
        "final_coverage_fraction": 1.0,
    }


def _methodology(checkpoint_size: int) -> dict[str, Any]:
    return {
        "format": "fluidsbench-method-v1",
        "record_kind": "prototype_fixture",
        "record_note": (
            "Non-ranked evaluator and dashboard development fixture. Complete public CFD truth is transformed deterministically to produce native-cell predictions. The retained GeoTransolver checkpoint is used only to calibrate four target error magnitudes; it was not executed, and this record is not model inference."
        ),
        "architecture": {
            "description": (
                "A deterministic signed multiplicative transform produces complete native CellData fields. A 20,166,852-parameter legacy GeoTransolver checkpoint is recorded only as a calibration reference and contributes no executed inference graph."
            ),
            "total_parameter_count": CHECKPOINT_PARAMETER_COUNT,
            "parameter_count_basis": "exact",
            "submitter_trainable_parameter_count": 0,
            "components": [
                {
                    "id": "legacy-geotransolver-calibration-reference",
                    "family": "GeoTransolver",
                    "role": "Non-executed calibration reference",
                    "description": "Retained checkpoint metadata supplied approximate field-error scales only; the checkpoint does not correspond to an official AhmedML split and was not run for this fixture.",
                    "parameter_count": CHECKPOINT_PARAMETER_COUNT,
                },
                {
                    "id": "deterministic-native-field-fixture",
                    "family": "Deterministic evaluator fixture",
                    "role": "Synthetic native-field generator",
                    "description": "Applies a case- and entity-index-dependent signed multiplicative perturbation directly to complete public CFD truth.",
                    "parameter_count": 0,
                },
            ],
            "key_hyperparameters": [
                {
                    "id": "fixture-transform",
                    "component_ids": ["deterministic-native-field-fixture"],
                    "name": "fixture transform",
                    "value": "truth*(1+target_ratio*sign(sin(pattern)))",
                    "description": "Deterministic transform used solely for integration testing.",
                },
                {
                    "id": "profile-sample-count",
                    "component_ids": ["deterministic-native-field-fixture"],
                    "name": "evaluator-owned samples per profile",
                    "value": 128,
                    "description": "Every Cp cut and velocity profile is sampled by the dataset evaluator at exactly 128 frozen support locations.",
                },
            ],
            "input_features": [
                {
                    "id": "public-native-cfd-truth",
                    "component_ids": ["deterministic-native-field-fixture"],
                    "name": "complete public CFD truth fields",
                    "domain": "domain",
                    "component_count": 8,
                    "description": "Surface pMean and wallShearStressMean plus volume pMean and UMean on all native cells; use of truth makes this explicitly ineligible for ranking.",
                },
                {
                    "id": "case-and-cell-index",
                    "component_ids": ["deterministic-native-field-fixture"],
                    "name": "case and native cell identity",
                    "domain": "case",
                    "component_count": 2,
                    "description": "Run ID and native raw-cell index seed the deterministic sign pattern.",
                },
            ],
            "predicted_fields": [
                {
                    "field_id": "surface-native.pMean",
                    "domain": "surface",
                    "component_count": 1,
                    "component_ids": ["deterministic-native-field-fixture"],
                    "production": "direct_model_output",
                    "description": "Synthetic complete-native-cell pressure fixture, not model inference.",
                },
                {
                    "field_id": "surface-native.wallShearStressMean",
                    "domain": "surface",
                    "component_count": 3,
                    "component_ids": ["deterministic-native-field-fixture"],
                    "production": "direct_model_output",
                    "description": "Synthetic complete-native-cell wall-shear fixture, not model inference.",
                },
                {
                    "field_id": "flow-domain-native.pMean",
                    "domain": "volume",
                    "component_count": 1,
                    "component_ids": ["deterministic-native-field-fixture"],
                    "production": "direct_model_output",
                    "description": "Synthetic complete-native-cell volume pressure fixture, not model inference.",
                },
                {
                    "field_id": "flow-domain-native.UMean",
                    "domain": "volume",
                    "component_count": 3,
                    "component_ids": ["deterministic-native-field-fixture"],
                    "production": "direct_model_output",
                    "description": "Synthetic complete-native-cell velocity fixture, not model inference.",
                },
            ],
        },
        "data_handling": {
            "normalization": "No training normalization is claimed. The fixture transform acts on native SI-valued CellData; Cp=2*pMean and Ux/Uinf=Ux because Uinf=1 m/s.",
            "preprocessing": "The evaluator validates immutable public source hashes, complete native entity order, surface areas, cell volumes, frozen profile mappings, evaluator-owned dominant-normal surface regions, and the retained three-zone volume partition.",
            "sampling": "No field subsampling. All surface and volume native cells are transformed and scored; evaluator-owned Cp and velocity diagnostics each contain exactly 128 points per station.",
        },
        "training": {
            "stages": [
                {
                    "id": "development-fixture-no-training",
                    "status": "prototype_not_recorded",
                    "component_ids": [
                        "legacy-geotransolver-calibration-reference",
                        "deterministic-native-field-fixture",
                    ],
                    "description": "No training is attributed to this fixture. The legacy checkpoint's exact official-split provenance is unavailable, and the synthetic transform has no trainable parameters.",
                }
            ]
        },
        "checkpoints": [
            {
                "id": "legacy-geotransolver-calibration-reference",
                "component_ids": ["legacy-geotransolver-calibration-reference"],
                "sha256": CHECKPOINT_SHA256,
                "digest_scope": "raw_loaded_file",
                "bytes_description": f"Raw {checkpoint_size}-byte model_best.pt retained checkpoint; hashed for provenance but not loaded or executed by fixture generation.",
                "role": "Calibration reference only; not the source of submitted prediction values.",
                "selection_rule": "Previously retained best checkpoint from a legacy configured test split; used only to select representative error scales before fixture generation.",
            }
        ],
        "inference_compute": {
            "status": "not_measured",
            "reason": "No model inference occurred. CPU fixture generation and exact evaluator runtime are implementation-test costs and must not be reported as surrogate inference performance.",
        },
    }


def assemble(
    *,
    dataset_evidence_path: Path,
    case_evidence_root: Path,
    prediction_root: Path,
    support_manifest_path: Path,
    profile_truth_manifest_path: Path,
    checkpoint_path: Path,
    output: Path,
    generated_at: str,
    submitted_at: str,
    profile_cases_per_chunk: int,
) -> Path:
    if output.exists():
        raise AssemblyError(f"output already exists: {output}")
    if profile_cases_per_chunk < 1 or profile_cases_per_chunk > 50:
        raise AssemblyError("profile_cases_per_chunk must lie in [1, 50]")
    try:
        parsed_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        datetime.fromisoformat(submitted_at)
    except ValueError as error:
        raise AssemblyError("generated/submitted timestamps are invalid") from error
    if parsed_time.tzinfo is None:
        raise AssemblyError("generated_at must include a timezone")

    dataset_evidence = _read_json(dataset_evidence_path, "dataset evidence")
    if (
        dataset_evidence.get("schema") != "ahmedml-candidate-dataset-evaluation-v1"
        or dataset_evidence.get("status")
        != "non_ranked_development_evidence_not_official_submission"
        or dataset_evidence.get("official_submission") is not False
        or dataset_evidence.get("leaderboard_eligible") is not False
        or dataset_evidence.get("dataset_id") != "ahmedml"
        or dataset_evidence.get("split_id") != "full"
    ):
        raise AssemblyError("dataset evidence is not the closed Full development evaluation")
    case_ids = dataset_evidence.get("case_ids")
    if not isinstance(case_ids, list) or len(case_ids) != 50 or len(set(case_ids)) != 50:
        raise AssemblyError("Full dataset evidence must contain 50 unique cases")
    split_path = ROOT / "benchmark-specs" / "ahmedml" / "splits" / "full.json"
    split = _read_json(split_path, "Full split")
    split_sha = sha256_file(split_path)
    if case_ids != split.get("case_ids") or split_sha != dataset_evidence.get("contract", {}).get("split_index_sha256"):
        raise AssemblyError("dataset evidence does not match the frozen Full split")

    recomputed = score_candidate_dataset(
        submission_specification=ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json",
        split_id="full",
        case_evidence_directory=case_evidence_root,
    ).to_json()
    if recomputed != dataset_evidence:
        raise AssemblyError("dataset evidence differs from a fresh case-evidence reduction")

    support_manifest_path = support_manifest_path.resolve()
    if support_manifest_path.name != "manifest.json":
        support_manifest_path = support_manifest_path / "manifest.json"
    support_manifest_sha = sha256_file(support_manifest_path)
    release = load_support_release(support_manifest_path, str(split["case_set_id"]))
    if (
        release.manifest.get("release_id") != SUPPORT_RELEASE_ID
        or release.manifest.get("dataset_id") != "ahmedml"
        or list(release.cases) != case_ids
    ):
        raise AssemblyError("scoring-support release identity or case order differs")

    profile_truth_manifest = _read_json(profile_truth_manifest_path, "profile truth manifest")
    profile_truth_release_id = _mapping(
        profile_truth_manifest.get("data_release"), "profile truth data_release"
    ).get("id")
    if not isinstance(profile_truth_release_id, str) or not profile_truth_release_id:
        raise AssemblyError("profile truth manifest has no release ID")
    profile_truth_manifest_sha = sha256_file(profile_truth_manifest_path)

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != CHECKPOINT_SHA256 or checkpoint_path.stat().st_size != CHECKPOINT_SIZE:
        raise AssemblyError("calibration checkpoint bytes differ")

    case_documents: dict[str, dict[str, Any]] = {}
    profile_cases: dict[str, list[dict[str, Any]]] = {}
    case_metric_records: list[dict[str, Any]] = []
    support_counts_by_case: dict[str, dict[str, int]] = {}
    for case_id in case_ids:
        evidence = _read_json(case_evidence_root / f"{case_id}.json", f"{case_id} evidence")
        provenance = _read_json(
            prediction_root / case_id / "fixture-provenance.json",
            f"{case_id} fixture provenance",
        )
        if (
            provenance.get("schema") != "ahmedml-geotransolver-calibrated-development-fixture-v1"
            or provenance.get("status") != "synthetic_non_ranked_not_model_inference"
            or provenance.get("actual_model_inference") is not False
            or provenance.get("official_submission") is not False
            or provenance.get("leaderboard_eligible") is not False
            or provenance.get("case_id") != case_id
            or provenance.get("calibration_checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
            or provenance.get("source", {}).get("source_identity_sha256") != SOURCE_IDENTITY_SHA256
        ):
            raise AssemblyError(f"{case_id} fixture provenance is incomplete or misleading")
        profiles, cp_r2, velocity_r2 = _case_profiles(evidence, case_id)
        counts = _support_counts(release, case_id)
        case_documents[case_id] = evidence
        profile_cases[case_id] = profiles
        support_counts_by_case[case_id] = counts
        case_metric_records.append(
            _case_metrics(
                evidence,
                case_id=case_id,
                counts=counts,
                cp_r2=cp_r2,
                velocity_r2=velocity_r2,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-partial-", dir=output.parent))
    try:
        profile_chunks: list[dict[str, Any]] = []
        for chunk_index, offset in enumerate(range(0, len(case_ids), profile_cases_per_chunk)):
            selected = case_ids[offset : offset + profile_cases_per_chunk]
            name = f"chunk-{chunk_index:03d}.json"
            digest = _write_json(
                staging / "profiles" / name,
                {
                    "schema_version": "1.0",
                    "cases": [
                        {"case_id": case_id, "series": profile_cases[case_id]}
                        for case_id in selected
                    ],
                },
            )
            profile_chunks.append({"file": name, "case_ids": selected, "sha256": digest})
        profile_index_sha = _write_json(
            staging / "profiles" / "index.json",
            {
                "schema_version": "1.0",
                "format": PROFILE_FORMAT,
                "submission_id": SUBMISSION_ID,
                "dataset_id": "ahmedml",
                "split_id": "full",
                "case_set_id": split["case_set_id"],
                "case_count": len(case_ids),
                "case_id_status": "official",
                "chunks": profile_chunks,
            },
        )

        metric_values = {
            key: _finite(value, f"metric_values.{key}")
            for key, value in _mapping(dataset_evidence.get("metric_values"), "metric_values").items()
        }
        case_metrics_sha = _write_json(
            staging / "metrics" / "cases.json",
            {
                "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
                "schema_version": "1.0",
                "submission_id": SUBMISSION_ID,
                "dataset_id": "ahmedml",
                "split_id": "full",
                "case_set_id": split["case_set_id"],
                "scoring_support_release_id": SUPPORT_RELEASE_ID,
                "scoring_support_manifest_sha256": support_manifest_sha,
                "case_count": len(case_ids),
                "cases": case_metric_records,
                "metric_values": metric_values,
                "generated_at": generated_at,
            },
        )

        surface_counts = [
            support_counts_by_case[case_id]["ahmedml-surface-native-cells-v1"]
            for case_id in case_ids
        ]
        volume_counts = [
            support_counts_by_case[case_id]["ahmedml-volume-native-cells-v1"]
            for case_id in case_ids
        ]
        summary_mappings = [
            _mapping_summary("ahmedml-surface-native-cells-v1", "surface-native-output", derived=False),
            _mapping_summary("ahmedml-volume-native-cells-v1", "volume-native-output", derived=False),
            _mapping_summary("ahmedml-force-coefficients-v1", "surface-native-output", derived=True),
            _mapping_summary("ahmedml-surface-cp-profiles-v1", "surface-native-output", derived=True),
            _mapping_summary("ahmedml-volume-velocity-profiles-v1", "volume-native-output", derived=True),
        ]
        case_discretizations: list[dict[str, Any]] = []
        for case_id in case_ids:
            counts = support_counts_by_case[case_id]
            surface_count = counts["ahmedml-surface-native-cells-v1"]
            volume_count = counts["ahmedml-volume-native-cells-v1"]
            case_discretizations.append(
                {
                    "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
                    "schema_version": "1.0",
                    "submission_id": SUBMISSION_ID,
                    "dataset_id": "ahmedml",
                    "split_id": "full",
                    "case_id": case_id,
                    "inference": {
                        "inputs": [
                            _case_representation("surface-native-input", "cells", surface_count),
                            _case_representation("volume-native-input", "cells", volume_count),
                        ],
                        "direct_outputs": [
                            _case_representation("surface-native-output", "cells", surface_count),
                            _case_representation("volume-native-output", "cells", volume_count),
                        ],
                        "mappings": [
                            _case_mapping("ahmedml-surface-native-cells-v1", "surface-native-output", surface_count),
                            _case_mapping("ahmedml-volume-native-cells-v1", "volume-native-output", volume_count),
                            _case_mapping("ahmedml-force-coefficients-v1", "surface-native-output", counts["ahmedml-force-coefficients-v1"]),
                            _case_mapping("ahmedml-surface-cp-profiles-v1", "surface-native-output", counts["ahmedml-surface-cp-profiles-v1"]),
                            _case_mapping("ahmedml-volume-velocity-profiles-v1", "volume-native-output", counts["ahmedml-volume-velocity-profiles-v1"]),
                        ],
                    },
                }
            )
        case_discretization_sha = _write_jsonl(
            staging / "discretization" / "cases.jsonl", case_discretizations
        )
        spatial = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
            "schema_version": "1.0",
            "submission_id": SUBMISSION_ID,
            "dataset_id": "ahmedml",
            "split_id": "full",
            "scoring_support_release_id": SUPPORT_RELEASE_ID,
            "scoring_support_manifest_sha256": support_manifest_sha,
            "training": {
                "surface_input": {"used": False, "notes": "No model training is attributed to this development fixture."},
                "surface_supervision": {"used": False, "notes": "No model training is attributed to this development fixture."},
                "volume_input": {"used": False, "notes": "No model training is attributed to this development fixture."},
                "volume_supervision": {"used": False, "notes": "No model training is attributed to this development fixture."},
            },
            "inference": {
                "geometry_dependency": "native_cfd_mesh",
                "surface_input": _representation_summary(
                    description="complete native surface CFD truth used by the synthetic fixture transform",
                    entity="cells",
                    counts=surface_counts,
                    case_record_id="surface-native-input",
                ),
                "volume_input": _representation_summary(
                    description="complete native volume CFD truth used by the synthetic fixture transform",
                    entity="cells",
                    counts=volume_counts,
                    case_record_id="volume-native-input",
                ),
                "direct_outputs": [
                    {
                        "id": "surface-native-output",
                        "domain": "surface",
                        "representation": _representation_summary(
                            description="complete native surface-cell synthetic prediction fields",
                            entity="cells",
                            counts=surface_counts,
                        ),
                        "queries_per_forward_pass": {"kind": "fixed", "value": 1},
                    },
                    {
                        "id": "volume-native-output",
                        "domain": "volume",
                        "representation": _representation_summary(
                            description="complete native volume-cell synthetic prediction fields",
                            entity="cells",
                            counts=volume_counts,
                        ),
                        "queries_per_forward_pass": {"kind": "fixed", "value": 1},
                    },
                ],
                "mappings": summary_mappings,
            },
            "case_manifest": {
                "format": "jsonl",
                "file": "discretization/cases.jsonl",
                "sha256": case_discretization_sha,
                "case_count": len(case_ids),
            },
            "notes": "Exact full-native-cell evaluator fixture. The direct-output wording describes generated fixture arrays, not neural-network inference.",
        }
        spatial_sha = _write_json(staging / "discretization.json", spatial)

        regional = build_aggregate_regional_diagnostics(
            case_ids=case_ids,
            case_evidence=case_documents,
            split_id="full",
        )
        regional_sha = _write_json(staging / "regional-diagnostics.json", regional)

        command = (
            "python scripts/assemble_ahmedml_schema_v3_dev_fixture.py "
            "--dataset-evidence <full-evidence.json> --case-evidence-directory <case-evidence> "
            "--prediction-directory <synthetic-native-fields> --support-release <candidate-support> "
            "--profile-ground-truth-manifest <dashboard-manifest> --checkpoint <model_best.pt> "
            "--output submissions/ahmedml/ahmedml-geotransolver-calibrated-dev-fixture-v1"
        )
        evidence = {
            "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
            "schema_version": "3.0",
            "submission_id": SUBMISSION_ID,
            "dataset_id": "ahmedml",
            "dataset_version": DATASET_VERSION,
            "split_id": "full",
            "split_sha256": split_sha,
            "case_set_id": split["case_set_id"],
            "prediction_scope": "surface_and_volume",
            "reference_version": EVALUATOR_VERSION,
            "command": command,
            "generated_at": generated_at,
            "status": "submitted_evaluation",
            "metric_values": metric_values,
            "profile_index_sha256": profile_index_sha,
            "profile_ground_truth_release_id": profile_truth_release_id,
            "profile_ground_truth_manifest_sha256": profile_truth_manifest_sha,
            "scoring_support_release_id": SUPPORT_RELEASE_ID,
            "scoring_support_manifest_sha256": support_manifest_sha,
            "discretization_sha256": spatial_sha,
            "case_metrics_sha256": case_metrics_sha,
            "regional_diagnostics_sha256": regional_sha,
            "notes": (
                "Non-ranked synthetic development fixture. Exact metrics, forces, profiles, and regional diagnostics were recomputed by the candidate dataset evaluator from complete native-cell fixture fields. The calibration checkpoint was not executed."
            ),
        }
        evidence_sha = _write_json(staging / "evaluation-evidence.json", evidence)

        submission = {
            "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
            "schema_version": "3.0",
            "submission_id": SUBMISSION_ID,
            "result_revision": {
                "series_id": SERIES_ID,
                "version": 1,
                "supersedes": None,
                "change_summary": "Initial non-ranked complete-native-cell AhmedML evaluator and dashboard development fixture.",
            },
            "model": "GeoTransolver-calibrated synthetic development fixture",
            "model_type": "Deterministic fixture",
            "model_types": ["Deterministic fixture", "GeoTransolver calibration reference"],
            "training_regime": "other",
            "training_regime_explanation": (
                "Development fixture only: no AhmedML model training is claimed. "
                "A retained GeoTransolver checkpoint supplies error-scale calibration "
                "but is not executed."
            ),
            "target_data_used": "other",
            "external_pretraining": False,
            "pretraining_data": [],
            "methodology": _methodology(checkpoint_path.stat().st_size),
            "submitter_name": "Neil Ashton",
            "institution": "NVIDIA",
            "paper_url": "",
            "submitted_at": submitted_at,
            "reproducibility": {
                "contract_version": "open-reproducibility-3.0",
                "access": "public",
                "public_test_data_use": "evaluation_only",
                "result_data_license_spdx": "CC-BY-SA-4.0",
            },
            "note": (
                "DEVELOPMENT FIXTURE ONLY: predictions are deterministic transforms of public CFD truth, not checkpoint inference. This row is permanently ineligible for ranking, citation, promotion, or treatment as an AhmedML submission."
            ),
            "dataset_id": "ahmedml",
            "prediction_scope": "surface_and_volume",
            "parameter_count_millions": CHECKPOINT_PARAMETER_COUNT / 1_000_000,
            "dataset": "AhmedML",
            "dataset_version": DATASET_VERSION,
            "split": "Full",
            "split_id": "full",
            "case_set_id": split["case_set_id"],
            "split_sha256": split_sha,
            "evaluation": {
                "reference_version": EVALUATOR_VERSION,
                "command": command,
                "evidence_file": "evaluation-evidence.json",
                "evidence_sha256": evidence_sha,
            },
            "scoring_support": {
                "status": "candidate",
                "release_id": SUPPORT_RELEASE_ID,
                "manifest_url": (
                    "https://github.com/neilashton/fluidsbench-submission/blob/dev/"
                    "benchmark-specs/ahmedml/scoring-support/"
                    f"{SUPPORT_RELEASE_ID}/manifest.json"
                ),
                "manifest_sha256": support_manifest_sha,
            },
            "spatial_discretization": {
                "format": "fluidsbench-discretization-v1",
                "file": "discretization.json",
                "sha256": spatial_sha,
            },
            "case_metrics": {
                "format": "fluidsbench-case-metrics-v1",
                "file": "metrics/cases.json",
                "sha256": case_metrics_sha,
                "case_count": len(case_ids),
            },
            "metric_values": metric_values,
            "profile_data": {
                "format": PROFILE_FORMAT,
                "index_file": "profiles/index.json",
                "case_count": len(case_ids),
                "case_set_id": split["case_set_id"],
                "profile_ground_truth_release_id": profile_truth_release_id,
                "profile_ground_truth_manifest_sha256": profile_truth_manifest_sha,
            },
            "regional_diagnostics": {
                "format": AGGREGATE_REGIONAL_REPORT_SCHEMA,
                "file": "regional-diagnostics.json",
                "sha256": regional_sha,
                "contract_sha256": REGION_DEFINITION_SHA256,
                "definition_id": REGIONAL_DEFINITION_ID,
                "case_count": len(case_ids),
                "role": "report_only",
                "weight": 0.0,
                "official_score_changed": False,
            },
        }
        _write_json(staging / "submission.json", submission)
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output / "submission.json")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-evidence", type=Path, required=True)
    parser.add_argument("--case-evidence-directory", type=Path, required=True)
    parser.add_argument("--prediction-directory", type=Path, required=True)
    parser.add_argument("--support-release", type=Path, required=True)
    parser.add_argument("--profile-ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--submitted-at", required=True)
    parser.add_argument("--profile-cases-per-chunk", type=int, default=10)
    args = parser.parse_args()
    try:
        assemble(
            dataset_evidence_path=args.dataset_evidence.expanduser().resolve(),
            case_evidence_root=args.case_evidence_directory.expanduser().resolve(),
            prediction_root=args.prediction_directory.expanduser().resolve(),
            support_manifest_path=args.support_release.expanduser().resolve(),
            profile_truth_manifest_path=args.profile_ground_truth_manifest.expanduser().resolve(),
            checkpoint_path=args.checkpoint.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            generated_at=args.generated_at,
            submitted_at=args.submitted_at,
            profile_cases_per_chunk=args.profile_cases_per_chunk,
        )
    except (AssemblyError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
