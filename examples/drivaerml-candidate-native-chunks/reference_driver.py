#!/usr/bin/env python3
"""Build a complete, synthetic DrivAerML-shaped schema-v3 package.

This deliberately does not read or emulate the numerical contents of a VTP or
VTU file.  It exercises the boundaries between ordered multipart transport,
raw-cell chunk accounting, additive field statistics, keyed prediction
packaging, the generic FluidsBench evaluator, and the public JSON schemas.

The real DrivAerML scoring support is still a closed candidate.  This driver
therefore uses a separate synthetic dataset/support namespace, writes an
explicitly ineligible dummy ``submission.json``, and makes no DrivAerML
activation, participant, approval, or model-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reference.drivaerml.accumulators import (  # noqa: E402
    FinalizedFieldStatistics,
    StreamingFieldAccumulator,
    field_chunk_statistics,
)
from reference.drivaerml.dataset_scorer import (  # noqa: E402
    validate_schema_v3_candidate_nonspatial_metrics,
)
from reference.evaluate_predictions import evaluate_prediction_artifact  # noqa: E402
from reference.scoring_support import sha256_file  # noqa: E402

try:  # noqa: E402
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as error:  # pragma: no cover - requirements.txt supplies it
    raise SystemExit(
        "jsonschema is required; install the repository requirements first"
    ) from error


RESULT_SERIES_ID = "synthetic-drivaerml-native-chunk-demo"
SUBMISSION_ID = f"{RESULT_SERIES_ID}-v1"
DATASET_ID = "synthetic-drivaerml-shaped"
DATASET_VERSION = "synthetic-native-chunk-demo-v1"
RELEASE_ID = "synthetic-drivaerml-shaped-support-v1"
CASE_SET_ID = "two-and-three-part-pilots"
SPLIT_ID = "synthetic-demo"
SURFACE_SUPPORT_ID = "surface_native_cells"
VOLUME_SUPPORT_ID = "volume_native_cells"
PROFILE_RELEASE_ID = "synthetic-drivaerml-shaped-profiles-v1"
REAL_DRIVAERML_DATASET_ID = "drivaerml"
REAL_DRIVAERML_CANDIDATE_STATUS = "candidate_scoring_contract"
REAL_DRIVAERML_SCORING_STATUS = "owner_review_required"
GENERATED_AT = "2026-08-20T00:00:00Z"
SUBMITTED_AT = "2026-08-20"
FORCE_METRIC_IDS = (
    "field_integrated_cd_rmse",
    "field_integrated_cl_rmse",
    "field_integrated_cmpitch_rmse",
    "field_integrated_clf_rmse",
    "field_integrated_clr_rmse",
    "field_integrated_lift_closure_max_abs",
)
DIAGNOSTIC_METRIC_IDS = (
    "velocity_profile_uinf_rmse",
    "velocity_profile_experimental_subset_uinf_rmse",
    "cp_cut_rmse",
)
GLOBAL_R2_METRIC_IDS = (
    "cd_r2",
    "cl_r2",
    "c_pitch_r2",
    "velocity_profile_r2",
    "cp_cut_r2",
)


@dataclass(frozen=True)
class DemoCase:
    """One small synthetic case with independent byte-part and cell chunks."""

    case_id: str
    part_count: int
    entity_count: int
    chunk_ranges: tuple[tuple[int, int], ...]
    processing_order: tuple[int, ...]
    case_offset: float


CASES = (
    DemoCase(
        case_id="run_1",
        part_count=2,
        entity_count=7,
        chunk_ranges=((0, 2), (2, 6), (6, 7)),
        processing_order=(2, 0, 1),
        case_offset=0.0,
    ),
    DemoCase(
        case_id="run_44",
        part_count=3,
        entity_count=8,
        chunk_ranges=((0, 1), (1, 4), (4, 6), (6, 8)),
        processing_order=(3, 1, 0, 2),
        case_offset=4.4,
    ),
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, indent=2, ensure_ascii=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schema_errors(schema_path: Path, value: object) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
        for error in errors
    ]


def _validate_schema(schema_relative_path: str, value_path: Path) -> None:
    errors = _schema_errors(
        REPOSITORY_ROOT / "schemas" / schema_relative_path,
        json.loads(value_path.read_text(encoding="utf-8")),
    )
    if errors:
        raise ValueError(
            f"{value_path} does not match {schema_relative_path}:\n"
            + "\n".join(errors)
        )


def _assert_closed_real_drivaerml_candidate(candidate_spec: object) -> None:
    """Prevent this dummy generator from crossing the real activation gate."""

    if not isinstance(candidate_spec, dict):
        raise ValueError("real DrivAerML candidate specification must be an object")
    scoring_state = candidate_spec.get("scoring_support")
    if not isinstance(scoring_state, dict):
        raise ValueError("real DrivAerML scoring-support state is missing")
    if (
        candidate_spec.get("dataset_id") != REAL_DRIVAERML_DATASET_ID
        or candidate_spec.get("status") != REAL_DRIVAERML_CANDIDATE_STATUS
        or scoring_state.get("status") != REAL_DRIVAERML_SCORING_STATUS
        or scoring_state.get("submissions_open") is not False
    ):
        raise ValueError(
            "the real DrivAerML candidate is no longer in the exact closed, "
            "owner-review-required state expected by this synthetic fixture"
        )
    if (
        DATASET_ID == REAL_DRIVAERML_DATASET_ID
        or not DATASET_ID.startswith("synthetic-")
        or not RELEASE_ID.startswith("synthetic-")
        or not SUBMISSION_ID.startswith("synthetic-")
    ):
        raise ValueError(
            "dummy package identifiers must stay in an unmistakably synthetic "
            "namespace distinct from the real DrivAerML dataset"
        )


def _case_source_arrays(case: DemoCase) -> dict[str, np.ndarray]:
    """Return deterministic surface and volume truth for one demo case."""

    raw_ids = np.arange(case.entity_count, dtype=np.int64)
    index = raw_ids.astype(np.float64)
    phase = case.case_offset
    pressure_truth = 8.0 + phase + 0.7 * index
    velocity_truth = np.column_stack(
        (
            30.0 + phase + 0.5 * index,
            -1.0 + 0.12 * index,
            0.5 - 0.07 * index,
        )
    )
    coordinates = np.column_stack(
        (0.25 * index, np.full(case.entity_count, phase), -0.1 * index)
    )
    surface_coordinates = np.column_stack(
        (0.18 * index, 0.03 * np.sin(index + phase), 0.04 * index)
    )
    surface_area = 0.55 + 0.08 * index + 0.01 * phase
    surface_pressure_truth = 7.5 + 0.25 * phase + 0.45 * index
    surface_wall_shear_truth = np.column_stack(
        (
            0.9 + 0.03 * phase + 0.04 * index,
            -0.12 + 0.015 * index,
            0.08 - 0.009 * index,
        )
    )
    return {
        "raw_ids": raw_ids,
        "pressure_truth": pressure_truth,
        "velocity_truth": velocity_truth,
        "coordinates": coordinates,
        "surface_coordinates": surface_coordinates,
        "surface_area": surface_area,
        "surface_pressure_truth": surface_pressure_truth,
        "surface_wall_shear_truth": surface_wall_shear_truth,
    }


def _synthetic_model_outputs(case: DemoCase, raw_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Produce model outputs separately from the reconstructed source payload."""

    index = raw_ids.astype(np.float64)
    phase = case.case_offset
    pressure_prediction = (
        8.0
        + phase
        + 0.7 * index
        + np.where(raw_ids % 2 == 0, 0.25, -0.4)
    )
    velocity_prediction = np.column_stack(
        (
            30.1 + phase + 0.51 * index,
            -1.0 + 0.12 * index + np.where(raw_ids % 2 == 0, -0.06, 0.04),
            0.47 - 0.065 * index,
        )
    )
    surface_pressure_prediction = (
        7.5
        + 0.25 * phase
        + 0.45 * index
        + np.where(raw_ids % 2 == 0, 0.18, -0.31)
    )
    surface_wall_shear_prediction = np.column_stack(
        (
            0.91 + 0.03 * phase + 0.041 * index,
            -0.13 + 0.016 * index,
            0.075 - 0.008 * index,
        )
    )
    return {
        "pressure_prediction": pressure_prediction,
        "velocity_prediction": velocity_prediction,
        "surface_pressure_prediction": surface_pressure_prediction,
        "surface_wall_shear_prediction": surface_wall_shear_prediction,
    }


def _transport_payload(case: DemoCase, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Create a pedagogical JSON byte stream; this is intentionally not VTU."""

    return {
        "warning": "synthetic JSON teaching payload; not VTK and not DrivAerML data",
        "case_id": case.case_id,
        "raw_cell_ids": arrays["raw_ids"].tolist(),
        "pMeanTrim_truth": arrays["pressure_truth"].tolist(),
        "UMeanTrim_truth": arrays["velocity_truth"].tolist(),
        "coordinates": arrays["coordinates"].tolist(),
    }


def _split_ordered_transport(
    output: Path,
    case: DemoCase,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split/rejoin arbitrary bytes, keeping byte parts distinct from cell chunks."""

    encoded = _canonical_bytes(payload)
    boundaries = [round(len(encoded) * index / case.part_count) for index in range(case.part_count + 1)]
    parts: list[dict[str, Any]] = []
    reconstructed_blocks: list[bytes] = []
    part_directory = output / "synthetic-transport" / case.case_id
    for part_index, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        block = encoded[start:stop]
        part_path = part_directory / f"payload.{part_index:02d}.part"
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(block)
        reconstructed_blocks.append(part_path.read_bytes())
        parts.append(
            {
                "part_index": part_index,
                "file": part_path.relative_to(output).as_posix(),
                "byte_size": len(block),
                "sha256": _sha256_bytes(block),
            }
        )
    reconstructed = b"".join(reconstructed_blocks)
    if reconstructed != encoded:
        raise ValueError(f"{case.case_id} ordered byte reconstruction changed the payload")
    decoded = json.loads(reconstructed.decode("utf-8"))
    if decoded != payload:
        raise ValueError(f"{case.case_id} reconstructed payload changed semantically")
    return decoded, {
        "part_count": case.part_count,
        "parts": parts,
        "logical_byte_size": len(encoded),
        "logical_sha256": _sha256_bytes(encoded),
        "ordered_byte_reconstruction_exact": True,
        "note": (
            "Part boundaries are arbitrary byte offsets and are independent of "
            "the raw-cell inference chunks below."
        ),
    }


def _compare_sums(
    chunked: FinalizedFieldStatistics,
    whole: FinalizedFieldStatistics,
) -> dict[str, Any]:
    absolute_differences: dict[str, float] = {}
    for weighting in ("uniform", "physical"):
        chunk_sums = getattr(chunked, weighting)
        whole_sums = getattr(whole, weighting)
        for field in (
            "absolute_error",
            "squared_error",
            "squared_truth",
            "total_weight",
        ):
            key = f"{weighting}.{field}"
            absolute_differences[key] = abs(
                float(getattr(chunk_sums, field)) - float(getattr(whole_sums, field))
            )
            if not math.isclose(
                float(getattr(chunk_sums, field)),
                float(getattr(whole_sums, field)),
                rel_tol=2e-15,
                abs_tol=1e-15,
            ):
                raise ValueError(f"chunked/full sufficient statistics differ for {key}")
        if chunk_sums.entity_count != whole_sums.entity_count:
            raise ValueError(f"chunked/full entity counts differ for {weighting}")
    return {
        "passed": True,
        "maximum_absolute_sufficient_statistic_difference": max(
            absolute_differences.values(), default=0.0
        ),
        "absolute_differences": absolute_differences,
    }


def _accumulate_field(
    case: DemoCase,
    arrays: dict[str, np.ndarray],
    *,
    truth_key: str,
    prediction_key: str,
    component_count: int,
    physical_weight_key: str | None = None,
) -> tuple[FinalizedFieldStatistics, dict[str, Any]]:
    raw_ids = arrays["raw_ids"]
    truth = arrays[truth_key]
    prediction = arrays[prediction_key]
    physical_weights = (
        np.ones(case.entity_count, dtype=np.float64)
        if physical_weight_key is None
        else np.asarray(arrays[physical_weight_key], dtype=np.float64)
    )

    chunk_statistics = [
        field_chunk_statistics(
            raw_ids[start:stop],
            truth[start:stop],
            prediction[start:stop],
            physical_weights[start:stop],
        )
        for start, stop in case.chunk_ranges
    ]
    chunked = StreamingFieldAccumulator(
        case.entity_count,
        component_count=component_count,
    )
    for chunk_index in case.processing_order:
        chunked.add_statistics(chunk_statistics[chunk_index])
    chunked_result = chunked.finalize()

    whole = StreamingFieldAccumulator(
        case.entity_count,
        component_count=component_count,
    )
    whole.add_chunk(raw_ids, truth, prediction, physical_weights)
    whole_result = whole.finalize()
    invariance = _compare_sums(chunked_result, whole_result)
    return chunked_result, invariance


def _support_ids(raw_ids: Iterable[int], *, domain: str) -> list[str]:
    return [f"raw-{domain}-cell-{int(raw_id):012d}" for raw_id in raw_ids]


def _metric_evidence(
    surface_pressure: FinalizedFieldStatistics,
    surface_wall_shear: FinalizedFieldStatistics,
    volume_pressure: FinalizedFieldStatistics,
    volume_velocity: FinalizedFieldStatistics,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        SURFACE_SUPPORT_ID: {
            "surface_pressure_rel_l2": (
                surface_pressure.physical.relative_l2_evidence(
                    weighting="support_weights",
                    dataset_weighting="surface_face_area",
                )
            ),
            "surface_pressure_equal_entity_rel_l2": (
                surface_pressure.uniform.relative_l2_evidence(
                    weighting="uniform",
                    dataset_weighting="surface_entities_equal",
                )
            ),
            "surface_wall_shear_rel_l2": (
                surface_wall_shear.physical.relative_l2_evidence(
                    weighting="support_weights",
                    dataset_weighting="surface_face_area",
                )
            ),
            "surface_wall_shear_equal_entity_rel_l2": (
                surface_wall_shear.uniform.relative_l2_evidence(
                    weighting="uniform",
                    dataset_weighting="surface_entities_equal",
                )
            ),
        },
        VOLUME_SUPPORT_ID: {
            "volume_pressure_rel_l2": (
                volume_pressure.uniform.relative_l2_evidence(
                    weighting="uniform",
                    dataset_weighting="volume_cells_equal",
                )
            ),
            "volume_velocity_rel_l2": (
                volume_velocity.uniform.relative_l2_evidence(
                    weighting="uniform",
                    dataset_weighting="volume_cells_equal",
                )
            ),
        },
    }


def _assert_evaluator_matches_chunks(
    case_metrics: dict[str, Any],
    expected: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    for case in case_metrics["cases"]:
        actual_supports = {
            support["support_id"]: support for support in case["supports"]
        }
        for support_id, expected_metrics in expected[case["case_id"]].items():
            actual = actual_supports[support_id]["metric_sufficient_statistics"]
            for metric_id, expected_statistics in expected_metrics.items():
                actual_statistics = actual[metric_id]
                for key in (
                    "reduction",
                    "weighting",
                    "dataset_weighting",
                    "entity_count",
                ):
                    if actual_statistics[key] != expected_statistics[key]:
                        raise ValueError(
                            f"{case['case_id']}/{support_id}/{metric_id}/{key} "
                            "differs between chunk accumulator and schema-v3 evaluator"
                        )
                for key in ("numerator", "denominator", "total_weight"):
                    if not math.isclose(
                        float(actual_statistics[key]),
                        float(expected_statistics[key]),
                        rel_tol=2e-15,
                        abs_tol=1e-15,
                    ):
                        raise ValueError(
                            f"{case['case_id']}/{support_id}/{metric_id}/{key} "
                            "differs between chunk accumulator and schema-v3 evaluator"
                        )


def _metric_binding(
    metric_id: str,
    quantity_id: str,
    reduction: str,
    weighting: str,
    dataset_weighting: str,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "metric_id": metric_id,
        "quantity_id": quantity_id,
        "reduction": reduction,
        "weighting": weighting,
        "dataset_weighting": dataset_weighting,
        "aggregation": "per_geometry_then_macro_average",
        "case_evidence": "metric_value",
    }
    if reduction in {"mae", "rmse"}:
        binding["reference_rule"] = {
            "id": "per_entity_euclidean_error",
            "version": "drivaerml-candidate-v1",
        }
    return binding


def _surface_support_definition() -> dict[str, Any]:
    pressure_components = [
        {
            "id": "pressure",
            "target_field": "pMeanTrim_truth",
            "target_association": "table_field",
            "prediction_field": "pMeanTrim_prediction",
        }
    ]
    wall_shear_components = [
        {
            "id": component,
            "target_field": f"wallShearStressMeanTrim_{component}_truth",
            "target_association": "table_field",
            "prediction_field": f"wallShearStressMeanTrim_{component}_prediction",
        }
        for component in ("x", "y", "z")
    ]
    bindings: list[dict[str, Any]] = []
    for prefix, quantity_id in (
        ("surface_pressure", "surface_pressure"),
        ("surface_wall_shear", "surface_wall_shear"),
    ):
        bindings.extend(
            [
                _metric_binding(
                    f"{prefix}_rel_l2",
                    quantity_id,
                    "relative_l2_percent",
                    "support_weights",
                    "surface_face_area",
                ),
                _metric_binding(
                    f"{prefix}_equal_entity_rel_l2",
                    quantity_id,
                    "relative_l2_percent",
                    "uniform",
                    "surface_entities_equal",
                ),
                _metric_binding(
                    f"drivaerml_{prefix}_area_mae",
                    quantity_id,
                    "mae",
                    "support_weights",
                    "surface_face_area",
                ),
                _metric_binding(
                    f"drivaerml_{prefix}_area_rmse",
                    quantity_id,
                    "rmse",
                    "support_weights",
                    "surface_face_area",
                ),
                _metric_binding(
                    f"drivaerml_{prefix}_equal_entity_mae",
                    quantity_id,
                    "mae",
                    "uniform",
                    "surface_entities_equal",
                ),
                _metric_binding(
                    f"drivaerml_{prefix}_equal_entity_rmse",
                    quantity_id,
                    "rmse",
                    "uniform",
                    "surface_entities_equal",
                ),
            ]
        )
    return {
        "id": SURFACE_SUPPORT_ID,
        "domain": "surface",
        "location_definition": {
            "mode": "materialized_table",
            "format": "json",
            "artifact_role": "ground_truth_table",
            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
            "coordinate_fields": ["x", "y", "z"],
            "weight_rule": {
                "kind": "artifact_field",
                "artifact_role": "ground_truth_table",
                "field": "surface_area",
            },
            "ordering": "support_id_ascending",
        },
        "quantities": [
            {
                "id": "surface_pressure",
                "unit": "m^2/s^2",
                "components": pressure_components,
            },
            {
                "id": "surface_wall_shear",
                "unit": "m^2/s^2",
                "components": wall_shear_components,
            },
        ],
        "metric_bindings": bindings,
        "required_coverage": {
            "count_fraction": 1.0,
            "weight_fraction": 1.0,
            "unmapped_count": 0,
        },
        "extrapolation_policy": "forbidden",
    }


def _volume_support_definition() -> dict[str, Any]:
    pressure_components = [
        {
            "id": "pressure",
            "target_field": "pMeanTrim_truth",
            "target_association": "table_field",
            "prediction_field": "pMeanTrim_prediction",
        }
    ]
    velocity_components = [
        {
            "id": component,
            "target_field": f"UMeanTrim_{component}_truth",
            "target_association": "table_field",
            "prediction_field": f"UMeanTrim_{component}_prediction",
        }
        for component in ("x", "y", "z")
    ]
    bindings: list[dict[str, Any]] = []
    for prefix, quantity_id in (
        ("volume_velocity", "volume_velocity"),
        ("volume_pressure", "volume_pressure"),
    ):
        bindings.extend(
            [
                _metric_binding(
                    f"{prefix}_rel_l2",
                    quantity_id,
                    "relative_l2_percent",
                    "uniform",
                    "volume_cells_equal",
                ),
                _metric_binding(
                    f"drivaerml_{prefix}_equal_entity_mae",
                    quantity_id,
                    "mae",
                    "uniform",
                    "volume_cells_equal",
                ),
                _metric_binding(
                    f"drivaerml_{prefix}_equal_entity_rmse",
                    quantity_id,
                    "rmse",
                    "uniform",
                    "volume_cells_equal",
                ),
            ]
        )
    return {
        "id": VOLUME_SUPPORT_ID,
        "domain": "volume",
        "location_definition": {
            "mode": "materialized_table",
            "format": "json",
            "artifact_role": "ground_truth_table",
            "support_id_rule": {"kind": "artifact_field", "field": "support_id"},
            "coordinate_fields": ["x", "y", "z"],
            "weight_rule": {"kind": "uniform"},
            "ordering": "support_id_ascending",
        },
        "quantities": [
            {
                "id": "volume_pressure",
                "unit": "m^2/s^2",
                "components": pressure_components,
            },
            {
                "id": "volume_velocity",
                "unit": "m/s",
                "components": velocity_components,
            },
        ],
        "metric_bindings": bindings,
        "required_coverage": {
            "count_fraction": 1.0,
            "weight_fraction": 1.0,
            "unmapped_count": 0,
        },
        "extrapolation_policy": "forbidden",
    }


def _field_metric_values(
    surface_pressure: FinalizedFieldStatistics,
    surface_wall_shear: FinalizedFieldStatistics,
    volume_pressure: FinalizedFieldStatistics,
    volume_velocity: FinalizedFieldStatistics,
) -> dict[str, dict[str, float]]:
    """Project chunk-safe statistics onto the exact candidate metric IDs."""

    return {
        SURFACE_SUPPORT_ID: {
            "surface_pressure_rel_l2": (
                surface_pressure.physical.relative_l2_percent()
            ),
            "surface_pressure_equal_entity_rel_l2": (
                surface_pressure.uniform.relative_l2_percent()
            ),
            "drivaerml_surface_pressure_area_mae": surface_pressure.physical.mae(),
            "drivaerml_surface_pressure_area_rmse": surface_pressure.physical.rmse(),
            "drivaerml_surface_pressure_equal_entity_mae": (
                surface_pressure.uniform.mae()
            ),
            "drivaerml_surface_pressure_equal_entity_rmse": (
                surface_pressure.uniform.rmse()
            ),
            "surface_wall_shear_rel_l2": (
                surface_wall_shear.physical.relative_l2_percent()
            ),
            "surface_wall_shear_equal_entity_rel_l2": (
                surface_wall_shear.uniform.relative_l2_percent()
            ),
            "drivaerml_surface_wall_shear_area_mae": (
                surface_wall_shear.physical.mae()
            ),
            "drivaerml_surface_wall_shear_area_rmse": (
                surface_wall_shear.physical.rmse()
            ),
            "drivaerml_surface_wall_shear_equal_entity_mae": (
                surface_wall_shear.uniform.mae()
            ),
            "drivaerml_surface_wall_shear_equal_entity_rmse": (
                surface_wall_shear.uniform.rmse()
            ),
        },
        VOLUME_SUPPORT_ID: {
            "volume_velocity_rel_l2": volume_velocity.uniform.relative_l2_percent(),
            "drivaerml_volume_velocity_equal_entity_mae": (
                volume_velocity.uniform.mae()
            ),
            "drivaerml_volume_velocity_equal_entity_rmse": (
                volume_velocity.uniform.rmse()
            ),
            "volume_pressure_rel_l2": volume_pressure.uniform.relative_l2_percent(),
            "drivaerml_volume_pressure_equal_entity_mae": (
                volume_pressure.uniform.mae()
            ),
            "drivaerml_volume_pressure_equal_entity_rmse": (
                volume_pressure.uniform.rmse()
            ),
        },
    }


def _apply_candidate_field_metrics(
    case_metrics: dict[str, Any],
    expected: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Use DrivAerML's per-entity vector norm for MAE and RMSE.

    The generic table evaluator deliberately has dataset-neutral flattened
    vector reductions. DrivAerML instead reduces each entity's Euclidean error
    norm. Relative-L2 evidence is cross-checked separately; this projection
    replaces all spatial values with the candidate accumulator's exact rule.
    """

    aggregate_values: dict[str, list[float]] = {}
    for case in case_metrics["cases"]:
        support_map = {
            support["support_id"]: support for support in case["supports"]
        }
        for support_id, expected_values in expected[case["case_id"]].items():
            actual_values = support_map[support_id]["metric_values"]
            if set(actual_values) != set(expected_values):
                raise ValueError(
                    f"{case['case_id']}/{support_id} metric IDs differ from "
                    "the DrivAerML candidate projection"
                )
            support_map[support_id]["metric_values"] = dict(expected_values)
            for metric_id, value in expected_values.items():
                aggregate_values.setdefault(metric_id, []).append(value)
    case_metrics["metric_values"] = {
        metric_id: math.fsum(values) / len(values)
        for metric_id, values in aggregate_values.items()
    }


def _line_rmse(
    coordinate: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Return arc-length-weighted RMSE on an ordered synthetic profile."""

    spacing = np.diff(coordinate)
    if len(spacing) == 0 or np.any(spacing <= 0.0):
        raise ValueError("synthetic profile coordinates must be strictly increasing")
    squared_error = np.square(prediction - truth)
    integral = float(
        np.sum(spacing * (squared_error[:-1] + squared_error[1:]) / 2.0)
    )
    return math.sqrt(integral / float(np.sum(spacing)))


def _line_r2_statistics(
    coordinate: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    """Return unit-support truth moments and SSE for one ordered profile."""

    spacing = np.diff(coordinate)
    if (
        len(spacing) == 0
        or np.any(spacing <= 0.0)
        or len(coordinate) != len(truth)
        or len(coordinate) != len(prediction)
    ):
        raise ValueError("synthetic profile R2 inputs are malformed")
    length = float(np.sum(spacing))
    squared_error = np.square(prediction - truth)
    return (
        float(np.sum(spacing * (truth[:-1] + truth[1:]) / 2.0)) / length,
        float(
            np.sum(spacing * (np.square(truth[:-1]) + np.square(truth[1:])) / 2.0)
        )
        / length,
        float(
            np.sum(
                spacing
                * (squared_error[:-1] + squared_error[1:])
                / 2.0
            )
        )
        / length,
    )


def _global_r2(statistics: list[tuple[float, float, float]]) -> float:
    """Return global R2 after giving every supplied profile equal weight."""

    if not statistics:
        raise ValueError("synthetic global R2 requires at least one profile")
    truth_sum = math.fsum(item[0] for item in statistics)
    truth_squared_sum = math.fsum(item[1] for item in statistics)
    squared_error_sum = math.fsum(item[2] for item in statistics)
    truth_sst = truth_squared_sum - truth_sum * truth_sum / len(statistics)
    if truth_sst <= 0.0:
        raise ValueError("synthetic global R2 truth is constant")
    return 1.0 - squared_error_sum / truth_sst


def _synthetic_profiles(
    candidate_spec: dict[str, Any],
    profile_registry: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    """Create all four Cp cuts and all sixteen AutoCFD5-shaped lines."""

    panels = {panel["id"]: panel for panel in candidate_spec["profile_panels"]}
    pressure_panel = panels["pressure_profiles"]
    velocity_panel = panels["velocity_profiles"]
    registry_stations = {
        station["id"]: station
        for station in profile_registry["velocity_profiles"]["stations"]
    }
    velocity_station_ids = velocity_panel["station_ids"]
    if list(registry_stations) != velocity_station_ids:
        raise ValueError(
            "candidate profile panel and DrivAerML diagnostic registry differ"
        )
    experimental_station_ids = {
        station_id
        for station_id, station in registry_stations.items()
        if station["experimental_availability"] != "none"
    }

    prediction_cases: list[dict[str, Any]] = []
    truth_cases: list[dict[str, Any]] = []
    diagnostic_values: dict[str, dict[str, float]] = {}
    cp_r2_statistics: list[tuple[float, float, float]] = []
    velocity_r2_statistics: list[tuple[float, float, float]] = []
    for case_index, case in enumerate(CASES):
        prediction_series: list[dict[str, Any]] = []
        truth_series: list[dict[str, Any]] = []
        cp_errors: list[float] = []
        velocity_errors: list[float] = []
        experimental_errors: list[float] = []

        for station_index, station_id in enumerate(pressure_panel["station_ids"]):
            coordinate = np.linspace(0.0, 1.0, 41, dtype=np.float64)
            truth = (
                -0.35
                + 0.04 * case_index
                + 0.025 * station_index
                + 0.16 * np.sin(math.pi * coordinate)
            )
            offset = 0.012 * (case_index + 1) * (1.0 + station_index / 10.0)
            prediction = truth + offset
            cp_errors.append(_line_rmse(coordinate, truth, prediction))
            cp_r2_statistics.append(
                _line_r2_statistics(coordinate, truth, prediction)
            )
            prediction_series.append(
                {
                    "panel_id": pressure_panel["id"],
                    "station_id": station_id,
                    "quantity_id": pressure_panel["quantity_ids"][0],
                    "coordinate": coordinate.tolist(),
                    "prediction": prediction.tolist(),
                }
            )
            truth_series.append(
                {
                    "panel_id": pressure_panel["id"],
                    "station_id": station_id,
                    "quantity_id": pressure_panel["quantity_ids"][0],
                    "coordinate": coordinate.tolist(),
                    "truth": truth.tolist(),
                }
            )

        for station_index, station_id in enumerate(velocity_station_ids):
            sample_count = velocity_panel["station_sample_counts"][station_id]
            coordinate_start, coordinate_stop = velocity_panel[
                "station_coordinate_intervals"
            ][station_id]
            coordinate = np.linspace(
                coordinate_start,
                coordinate_stop,
                sample_count,
                dtype=np.float64,
            )
            normalized_coordinate = coordinate / max(coordinate_stop, 1.0)
            truth = (
                0.72
                + 0.015 * case_index
                + 0.006 * station_index
                + 0.08 * np.cos(math.pi * normalized_coordinate)
            )
            offset = 0.004 * (case_index + 1) * (1.0 + station_index / 20.0)
            prediction = truth + offset
            error = _line_rmse(coordinate, truth, prediction)
            velocity_r2_statistics.append(
                _line_r2_statistics(coordinate, truth, prediction)
            )
            velocity_errors.append(error)
            if station_id in experimental_station_ids:
                experimental_errors.append(error)
            prediction_series.append(
                {
                    "panel_id": velocity_panel["id"],
                    "station_id": station_id,
                    "quantity_id": velocity_panel["quantity_ids"][0],
                    "coordinate": coordinate.tolist(),
                    "prediction": prediction.tolist(),
                }
            )
            truth_series.append(
                {
                    "panel_id": velocity_panel["id"],
                    "station_id": station_id,
                    "quantity_id": velocity_panel["quantity_ids"][0],
                    "coordinate": coordinate.tolist(),
                    "truth": truth.tolist(),
                }
            )

        prediction_cases.append(
            {"case_id": case.case_id, "series": prediction_series}
        )
        truth_cases.append({"case_id": case.case_id, "series": truth_series})
        diagnostic_values[case.case_id] = {
            "velocity_profile_uinf_rmse": (
                math.fsum(velocity_errors) / len(velocity_errors)
            ),
            "velocity_profile_experimental_subset_uinf_rmse": (
                math.fsum(experimental_errors) / len(experimental_errors)
            ),
            "cp_cut_rmse": math.fsum(cp_errors) / len(cp_errors),
        }
    return (
        prediction_cases,
        truth_cases,
        diagnostic_values,
        {
            "velocity_profile_r2": _global_r2(velocity_r2_statistics),
            "cp_cut_r2": _global_r2(cp_r2_statistics),
        },
    )


def _synthetic_force_truth(case_index: int) -> dict[str, float]:
    """Return the exact synthetic force truth for one case."""

    truth_cl = 0.16 + 0.01 * case_index
    truth_cm_pitch = 0.015 - 0.002 * case_index
    return {
        "cd": 0.24 + 0.005 * case_index,
        "cl": truth_cl,
        "cm_pitch": truth_cm_pitch,
        "clf": truth_cl / 2.0 + truth_cm_pitch,
        "clr": truth_cl / 2.0 - truth_cm_pitch,
    }


def _force_r2(truth: list[float], prediction: list[float]) -> float:
    """Return equal-case R2 for one synthetic force coefficient."""

    if len(truth) != len(prediction) or len(truth) < 2:
        raise ValueError("synthetic force R2 inputs are malformed")
    truth_mean = math.fsum(truth) / len(truth)
    truth_sst = math.fsum((value - truth_mean) ** 2 for value in truth)
    if truth_sst <= 0.0:
        raise ValueError("synthetic force R2 truth is constant")
    squared_error = math.fsum(
        (predicted - observed) ** 2
        for observed, predicted in zip(truth, prediction, strict=True)
    )
    return 1.0 - squared_error / truth_sst


def _synthetic_force_values(
    case_index: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return coherent force coefficients and their per-case errors."""

    truth = _synthetic_force_truth(case_index)

    predicted_cd = truth["cd"] + 0.01 * (case_index + 1)
    predicted_cl = truth["cl"] - 0.012 * (case_index + 1)
    predicted_cm_pitch = truth["cm_pitch"] + 0.003 * (case_index + 1)
    prediction = {
        "cd": predicted_cd,
        "cl": predicted_cl,
        "cm_pitch": predicted_cm_pitch,
        "clf": predicted_cl / 2.0 + predicted_cm_pitch,
        "clr": predicted_cl / 2.0 - predicted_cm_pitch,
    }
    force_metric_keys = dict(
        zip(
            FORCE_METRIC_IDS[:-1],
            ("cd", "cl", "cm_pitch", "clf", "clr"),
            strict=True,
        )
    )
    errors = {
        metric_id: abs(prediction[key] - truth[key])
        for metric_id, key in force_metric_keys.items()
    }
    errors["field_integrated_lift_closure_max_abs"] = abs(
        prediction["cl"] - prediction["clf"] - prediction["clr"]
    )
    return prediction, errors


def _augment_candidate_nonspatial_metrics(
    case_metrics: dict[str, Any],
    *,
    candidate_spec: dict[str, Any],
    diagnostic_values: dict[str, dict[str, float]],
    profile_r2_values: dict[str, float],
) -> dict[str, Any]:
    """Add exact force/profile fields and validate DrivAerML reductions."""

    per_metric: dict[str, list[float]] = {
        metric_id: [] for metric_id in (*FORCE_METRIC_IDS, *DIAGNOSTIC_METRIC_IDS)
    }
    force_truth: dict[str, list[float]] = {
        coefficient: [] for coefficient in ("cd", "cl", "cm_pitch")
    }
    force_prediction: dict[str, list[float]] = {
        coefficient: [] for coefficient in force_truth
    }
    for case_index, case in enumerate(case_metrics["cases"]):
        force_coefficients, force_values = _synthetic_force_values(case_index)
        truth_coefficients = _synthetic_force_truth(case_index)
        nonspatial = {
            **force_values,
            **diagnostic_values[case["case_id"]],
        }
        case["force_coefficients"] = force_coefficients
        case["nonspatial_metric_values"] = nonspatial
        for metric_id, value in nonspatial.items():
            per_metric[metric_id].append(value)
        for coefficient in force_truth:
            force_truth[coefficient].append(truth_coefficients[coefficient])
            force_prediction[coefficient].append(force_coefficients[coefficient])

    for metric_id in FORCE_METRIC_IDS[:-1]:
        values = per_metric[metric_id]
        case_metrics["metric_values"][metric_id] = math.sqrt(
            math.fsum(value * value for value in values) / len(values)
        )
    closure_values = per_metric["field_integrated_lift_closure_max_abs"]
    case_metrics["metric_values"]["field_integrated_lift_closure_max_abs"] = max(
        closure_values
    )
    for metric_id in DIAGNOSTIC_METRIC_IDS:
        values = per_metric[metric_id]
        case_metrics["metric_values"][metric_id] = math.fsum(values) / len(values)
    for metric_id, coefficient in (
        ("cd_r2", "cd"),
        ("cl_r2", "cl"),
        ("c_pitch_r2", "cm_pitch"),
    ):
        case_metrics["metric_values"][metric_id] = _force_r2(
            force_truth[coefficient],
            force_prediction[coefficient],
        )
    if set(profile_r2_values) != {"velocity_profile_r2", "cp_cut_r2"}:
        raise ValueError("synthetic profile R2 metric set is incomplete")
    case_metrics["metric_values"].update(profile_r2_values)

    candidate_non_score_ids = {
        metric["id"]
        for metric in candidate_spec["metrics"]
        if metric["kind"] != "score"
    }
    if set(case_metrics["metric_values"]) != candidate_non_score_ids:
        raise ValueError(
            "synthetic case metrics do not cover the exact DrivAerML candidate "
            "non-score metric set"
        )
    validation_copy = json.loads(json.dumps(case_metrics))
    validation_copy["dataset_id"] = REAL_DRIVAERML_DATASET_ID
    return validate_schema_v3_candidate_nonspatial_metrics(validation_copy)


def _per_case_count(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "kind": "per_case",
        "minimum": min(ordered),
        "median": median,
        "maximum": max(ordered),
    }


def _used_native_representation(
    entity_counts: list[int],
    *,
    domain: str,
) -> dict[str, Any]:
    if domain not in {"surface", "volume"}:
        raise ValueError("synthetic native representation domain is invalid")
    entity = "native_polygons" if domain == "surface" else "native_cells"
    return {
        "used": True,
        "representation": f"synthetic raw {entity.replace('_', '-')}-shaped table",
        "entity_counts": [
            {"entity": entity, "count": _per_case_count(entity_counts)}
        ],
        "native_comparison": {"status": "not_applicable"},
        "sampling": {"kind": "none"},
        "domain": {"kind": "full_dataset_domain"},
        "connectivity": "none",
        "notes": (
            "Tiny synthetic teaching fixture; no VTP or VTU mesh is represented. "
            + (
                "A positive synthetic area is attached to every polygon."
                if domain == "surface"
                else "Every synthetic volume cell has equal scoring weight."
            )
        ),
    }


def _synthetic_methodology() -> dict[str, Any]:
    """Describe the parameter-free teaching function without implying ML training."""

    component_id = "deterministic-teaching-function"
    return {
        "format": "fluidsbench-method-v1",
        "record_kind": "prototype_fixture",
        "record_note": (
            "Synthetic schema exercise only. The outputs come from a deterministic "
            "parameter-free teaching function, not a trained model or DrivAerML result."
        ),
        "architecture": {
            "description": (
                "A parameter-free deterministic function that emits synthetic "
                "surface and volume fields for package-validation examples."
            ),
            "total_parameter_count": 0,
            "parameter_count_basis": "exact",
            "submitter_trainable_parameter_count": 0,
            "components": [
                {
                    "id": component_id,
                    "family": "Deterministic analytic fixture",
                    "role": "Generate synthetic surface and volume predictions.",
                    "description": (
                        "Fixed arithmetic expressions exercise chunk accumulation; "
                        "there are no learned weights or hidden model layers."
                    ),
                    "parameter_count": 0,
                }
            ],
            "key_hyperparameters": [
                {
                    "id": "fixture-kind",
                    "component_ids": [component_id],
                    "name": "fixture_kind",
                    "value": "deterministic_synthetic",
                    "description": "Identifies this output as a teaching fixture.",
                }
            ],
            "input_features": [
                {
                    "id": "synthetic-native-coordinates",
                    "component_ids": [component_id],
                    "name": "synthetic_native_coordinates",
                    "domain": "mesh",
                    "component_count": 3,
                    "description": (
                        "Synthetic three-dimensional coordinates and raw entity IDs."
                    ),
                }
            ],
            "predicted_fields": [
                {
                    "field_id": "surface_native_cells.pMeanTrim",
                    "domain": "surface",
                    "component_count": 1,
                    "component_ids": [component_id],
                    "production": "direct_model_output",
                    "description": "Synthetic surface-pressure fixture values.",
                },
                {
                    "field_id": "surface_native_cells.wallShearStressMeanTrim",
                    "domain": "surface",
                    "component_count": 3,
                    "component_ids": [component_id],
                    "production": "direct_model_output",
                    "description": "Synthetic surface wall-shear fixture values.",
                },
                {
                    "field_id": "volume_native_cells.pMeanTrim",
                    "domain": "volume",
                    "component_count": 1,
                    "component_ids": [component_id],
                    "production": "direct_model_output",
                    "description": "Synthetic volume-pressure fixture values.",
                },
                {
                    "field_id": "volume_native_cells.UMeanTrim",
                    "domain": "volume",
                    "component_count": 3,
                    "component_ids": [component_id],
                    "production": "direct_model_output",
                    "description": "Synthetic volume-velocity fixture values.",
                },
            ],
        },
        "data_handling": {
            "normalization": "No normalization; fixed synthetic values are emitted.",
            "preprocessing": "Generate the tiny in-memory teaching arrays.",
            "sampling": "Use every synthetic entity in both teaching cases.",
        },
        "training": {
            "stages": [
                {
                    "id": "training-not-applicable",
                    "status": "prototype_not_recorded",
                    "component_ids": [component_id],
                    "description": (
                        "No training was performed because the fixture is a fixed "
                        "parameter-free function."
                    ),
                }
            ]
        },
        "checkpoints": [],
        "inference_compute": {
            "status": "not_measured",
            "reason": (
                "Timing a tiny deterministic schema fixture would not represent "
                "model inference performance."
            ),
        },
    }


def _write_submission_package(
    output: Path,
    *,
    support_manifest_path: Path,
    prediction_manifest_path: Path,
    case_metrics_path: Path,
    profile_cases: list[dict[str, Any]],
    profile_truth_cases: list[dict[str, Any]],
) -> tuple[dict[str, Path], Path]:
    """Write the remaining schema-v3 package without making real-data claims."""

    case_ids = [case.case_id for case in CASES]
    entity_counts = [case.entity_count for case in CASES]
    split_path = output / "synthetic-split.json"
    _write_json(
        split_path,
        {
            "status": "synthetic_ineligible_fixture",
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "case_ids": case_ids,
            "not_an_official_drivaerml_split": True,
        },
    )
    split_sha256 = sha256_file(split_path)

    profile_chunk_path = output / "profiles" / "chunk-000.json"
    _write_json(
        profile_chunk_path,
        {"schema_version": "1.0", "cases": profile_cases},
    )
    profile_index_path = output / "profiles" / "index.json"
    _write_json(
        profile_index_path,
        {
            "schema_version": "1.0",
            "submission_id": SUBMISSION_ID,
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "case_set_id": CASE_SET_ID,
            "case_count": len(CASES),
            "case_id_status": "prototype_generated",
            "chunks": [
                {
                    "file": profile_chunk_path.name,
                    "case_ids": case_ids,
                    "sha256": sha256_file(profile_chunk_path),
                }
            ],
        },
    )
    profile_ground_truth_data_path = output / "profiles" / "ground-truth.json"
    _write_json(
        profile_ground_truth_data_path,
        {
            "schema_version": "synthetic-profile-truth-v1",
            "dataset_id": DATASET_ID,
            "case_set_id": CASE_SET_ID,
            "cases": profile_truth_cases,
        },
    )
    profile_ground_truth_path = output / "profiles" / "ground-truth-manifest.json"
    _write_json(
        profile_ground_truth_path,
        {
            "release_id": PROFILE_RELEASE_ID,
            "status": "synthetic_ineligible_fixture",
            "case_ids": case_ids,
            "artifact": {
                "file": profile_ground_truth_data_path.name,
                "sha256": sha256_file(profile_ground_truth_data_path),
            },
            "note": (
                "Synthetic truth for exercising all four candidate Cp cuts and "
                "all sixteen candidate AutoCFD5 velocity-line payloads; no real "
                "DrivAerML profile truth is present."
            ),
        },
    )
    profile_ground_truth_sha256 = sha256_file(profile_ground_truth_path)

    case_record_path = output / "discretization" / "cases.jsonl"
    case_records = [
        {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
            "schema_version": "1.0",
            "submission_id": SUBMISSION_ID,
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "case_id": case.case_id,
            "inference": {
                "inputs": [],
                "direct_outputs": [
                    {
                        "id": "surface-native-polygon-output",
                        "entity_counts": [
                            {"entity": "native_polygons", "count": case.entity_count}
                        ],
                        "native_entity_counts": [
                            {"entity": "native_polygons", "count": case.entity_count}
                        ],
                        "native_fractions": [
                            {"entity": "native_polygons", "fraction": 1.0}
                        ],
                        "domain": {"kind": "full_dataset_domain"},
                    },
                    {
                        "id": "volume-native-cell-output",
                        "entity_counts": [
                            {"entity": "native_cells", "count": case.entity_count}
                        ],
                        "native_entity_counts": [
                            {"entity": "native_cells", "count": case.entity_count}
                        ],
                        "native_fractions": [
                            {"entity": "native_cells", "fraction": 1.0}
                        ],
                        "domain": {"kind": "full_dataset_domain"},
                    }
                ],
                "mappings": [
                    {
                        "support_id": SURFACE_SUPPORT_ID,
                        "source_output_id": "surface-native-polygon-output",
                        "support_count": case.entity_count,
                        "scored_count": case.entity_count,
                        "unmapped_count": 0,
                        "extrapolated_count": 0,
                        "final_coverage_fraction": 1.0,
                    },
                    {
                        "support_id": VOLUME_SUPPORT_ID,
                        "source_output_id": "volume-native-cell-output",
                        "support_count": case.entity_count,
                        "scored_count": case.entity_count,
                        "unmapped_count": 0,
                        "extrapolated_count": 0,
                        "final_coverage_fraction": 1.0,
                    }
                ],
            },
        }
        for case in CASES
    ]
    _write_jsonl(case_record_path, case_records)

    discretization_path = output / "discretization.json"
    surface_representation = _used_native_representation(
        entity_counts,
        domain="surface",
    )
    volume_representation = _used_native_representation(
        entity_counts,
        domain="volume",
    )
    _write_json(
        discretization_path,
        {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
            "schema_version": "1.0",
            "submission_id": SUBMISSION_ID,
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "scoring_support_release_id": RELEASE_ID,
            "scoring_support_manifest_sha256": sha256_file(support_manifest_path),
            "training": {
                "surface_input": {"used": False},
                "surface_supervision": {"used": False},
                "volume_input": {"used": False},
                "volume_supervision": {"used": False},
            },
            "inference": {
                "geometry_dependency": "other",
                "geometry_dependency_explanation": (
                    "Synthetic arrays generated internally; no real geometry was read."
                ),
                "surface_input": {"used": False},
                "volume_input": {"used": False},
                "direct_outputs": [
                    {
                        "id": "surface-native-polygon-output",
                        "domain": "surface",
                        "representation": surface_representation,
                        "queries_per_forward_pass": _per_case_count(entity_counts),
                    },
                    {
                        "id": "volume-native-cell-output",
                        "domain": "volume",
                        "representation": volume_representation,
                        "queries_per_forward_pass": _per_case_count(entity_counts),
                    }
                ],
                "mappings": [
                    {
                        "support_id": SURFACE_SUPPORT_ID,
                        "source_output_id": "surface-native-polygon-output",
                        "method": {"kind": "identity"},
                        "implementation": (
                            "examples/drivaerml-candidate-native-chunks/reference_driver.py"
                        ),
                        "extrapolation_policy": "forbidden",
                        "unmapped_fraction": 0.0,
                        "extrapolated_fraction": 0.0,
                        "final_coverage_fraction": 1.0,
                    },
                    {
                        "support_id": VOLUME_SUPPORT_ID,
                        "source_output_id": "volume-native-cell-output",
                        "method": {"kind": "identity"},
                        "implementation": (
                            "examples/drivaerml-candidate-native-chunks/reference_driver.py"
                        ),
                        "extrapolation_policy": "forbidden",
                        "unmapped_fraction": 0.0,
                        "extrapolated_fraction": 0.0,
                        "final_coverage_fraction": 1.0,
                    }
                ],
            },
            "case_manifest": {
                "format": "jsonl",
                "file": "discretization/cases.jsonl",
                "sha256": sha256_file(case_record_path),
                "case_count": len(CASES),
            },
            "notes": (
                "Synthetic and ineligible; the surface/volume declaration mirrors "
                "native DrivAerML-shaped outputs, but no native DrivAerML mesh was used."
            ),
        },
    )

    case_metrics = json.loads(case_metrics_path.read_text(encoding="utf-8"))
    evidence_path = output / "evaluation-evidence.json"
    _write_json(
        evidence_path,
        {
            "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
            "schema_version": "3.0",
            "submission_id": SUBMISSION_ID,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "split_id": SPLIT_ID,
            "split_sha256": split_sha256,
            "case_set_id": CASE_SET_ID,
            "reference_version": "synthetic-drivaerml-chunk-driver-v3",
            "command": (
                "python examples/drivaerml-candidate-native-chunks/"
                "reference_driver.py --output OUTPUT"
            ),
            "generated_at": GENERATED_AT,
            "status": "submitted_evaluation",
            "metric_values": case_metrics["metric_values"],
            "profile_index_sha256": sha256_file(profile_index_path),
            "profile_ground_truth_release_id": PROFILE_RELEASE_ID,
            "profile_ground_truth_manifest_sha256": profile_ground_truth_sha256,
            "scoring_support_release_id": RELEASE_ID,
            "scoring_support_manifest_sha256": sha256_file(support_manifest_path),
            "discretization_sha256": sha256_file(discretization_path),
            "case_metrics_sha256": sha256_file(case_metrics_path),
            "notes": (
                "Schema exercise only: synthetic, ineligible, unapproved, and not "
                "a DrivAerML participant evaluation."
            ),
        },
    )

    submission_path = output / "submission.json"
    _write_json(
        submission_path,
        {
            "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
            "schema_version": "3.0",
            "submission_id": SUBMISSION_ID,
            "result_revision": {
                "series_id": RESULT_SERIES_ID,
                "version": 1,
                "supersedes": None,
                "change_summary": "Initial synthetic teaching-bundle result.",
            },
            "model": "Deterministic Synthetic Teaching Function",
            "model_type": "Synthetic fixture",
            "model_types": ["Synthetic fixture"],
            "training_regime": "other",
            "training_regime_explanation": (
                "No model was trained; deterministic synthetic values exercise packaging only."
            ),
            "target_data_used": "none",
            "external_pretraining": False,
            "pretraining_data": [],
            "dataset": "Synthetic DrivAerML-shaped teaching fixture",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "split": "Synthetic two-case demonstration",
            "split_id": SPLIT_ID,
            "case_set_id": CASE_SET_ID,
            "split_sha256": split_sha256,
            "parameter_count_millions": 0.0,
            "methodology": _synthetic_methodology(),
            "submitter_name": "FluidsBench synthetic fixture",
            "institution": "No institution; generated test fixture",
            "paper_url": "",
            "submitted_at": SUBMITTED_AT,
            "evaluation": {
                "reference_version": "synthetic-drivaerml-chunk-driver-v3",
                "command": (
                    "python examples/drivaerml-candidate-native-chunks/"
                    "reference_driver.py --output OUTPUT"
                ),
                "evidence_file": "evaluation-evidence.json",
                "evidence_sha256": sha256_file(evidence_path),
            },
            "reproducibility": {
                "contract_version": "open-reproducibility-3.0",
                "access": "public",
                "public_test_data_use": "evaluation_only",
                "result_data_license_spdx": "CC-BY-4.0",
            },
            "scoring_support": {
                "status": "candidate",
                "release_id": RELEASE_ID,
                "manifest_url": (
                    "https://example.invalid/fluidsbench/synthetic-drivaerml-shaped/"
                    "support/manifest.json"
                ),
                "manifest_sha256": sha256_file(support_manifest_path),
            },
            "spatial_discretization": {
                "format": "fluidsbench-discretization-v1",
                "file": "discretization.json",
                "sha256": sha256_file(discretization_path),
            },
            "case_metrics": {
                "format": "fluidsbench-case-metrics-v1",
                "file": "metrics/cases.json",
                "sha256": sha256_file(case_metrics_path),
                "case_count": len(CASES),
            },
            "prediction_artifacts": [
                {
                    "artifact_id": "drivaerml-synthetic-complete-predictions-v1",
                    "kind": "scored_predictions",
                    "repository_url": (
                        "https://huggingface.co/datasets/example/"
                        "synthetic-drivaerml-shaped-predictions"
                    ),
                    "revision": "0" * 40,
                    "manifest_file": "predictions/manifest.json",
                    "manifest_sha256": sha256_file(prediction_manifest_path),
                    "format": "fluidsbench-prediction-artifact-v1",
                    "support_release_id": RELEASE_ID,
                    "support_manifest_sha256": sha256_file(support_manifest_path),
                    "split_id": SPLIT_ID,
                    "coverage": {
                        "kind": "complete_split",
                        "case_count": len(CASES),
                        "expected_case_count": len(CASES),
                    },
                    "license_spdx": "CC-BY-4.0",
                }
            ],
            "metric_values": case_metrics["metric_values"],
            "profile_data": {
                "format": "fluidsbench-profile-chunks-v1",
                "index_file": "profiles/index.json",
                "case_count": len(CASES),
                "case_set_id": CASE_SET_ID,
                "profile_ground_truth_release_id": PROFILE_RELEASE_ID,
                "profile_ground_truth_manifest_sha256": profile_ground_truth_sha256,
            },
            "note": (
                "INELIGIBLE SYNTHETIC FIXTURE. This is not DrivAerML data, an "
                "active DrivAerML contract, a model result, owner approval, or an "
                "independent participant dry run. Candidate status applies only "
                "to this isolated fictional support namespace."
            ),
        },
    )
    return (
        {
            "v1/profile-chunk.schema.json": profile_chunk_path,
            "v1/profile-index.schema.json": profile_index_path,
            "v3/discretization-case.schema.json": case_record_path,
            "v3/discretization.schema.json": discretization_path,
            "v3/evaluation-evidence.schema.json": evidence_path,
            "v3/submission.schema.json": submission_path,
        },
        submission_path,
    )


def run_demo(output: Path) -> dict[str, Any]:
    """Generate, evaluate, and schema-check the candidate dry-run slice."""

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    candidate_spec_path = (
        REPOSITORY_ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json"
    )
    candidate_spec = json.loads(candidate_spec_path.read_text(encoding="utf-8"))
    _assert_closed_real_drivaerml_candidate(candidate_spec)
    scoring_state = candidate_spec["scoring_support"]
    profile_registry_path = (
        REPOSITORY_ROOT
        / "benchmark-specs"
        / "drivaerml"
        / "drivaerml-diagnostics-v9.json"
    )
    profile_registry = json.loads(profile_registry_path.read_text(encoding="utf-8"))

    support_data_directory = output / "support" / "case-sets" / CASE_SET_ID / "data"
    prediction_data_directory = output / "predictions" / "data"
    case_instances: list[dict[str, Any]] = []
    prediction_cases: list[dict[str, Any]] = []
    case_receipts: list[dict[str, Any]] = []
    expected_statistics: dict[
        str, dict[str, dict[str, dict[str, Any]]]
    ] = {}
    expected_field_values: dict[str, dict[str, dict[str, float]]] = {}

    for case in CASES:
        generated = _case_source_arrays(case)
        decoded, transport_receipt = _split_ordered_transport(
            output,
            case,
            _transport_payload(case, generated),
        )
        arrays = {
            "raw_ids": np.asarray(decoded["raw_cell_ids"], dtype=np.int64),
            "pressure_truth": np.asarray(decoded["pMeanTrim_truth"], dtype=np.float64),
            "velocity_truth": np.asarray(decoded["UMeanTrim_truth"], dtype=np.float64),
            "coordinates": np.asarray(decoded["coordinates"], dtype=np.float64),
            "surface_coordinates": generated["surface_coordinates"],
            "surface_area": generated["surface_area"],
            "surface_pressure_truth": generated["surface_pressure_truth"],
            "surface_wall_shear_truth": generated["surface_wall_shear_truth"],
        }
        arrays.update(_synthetic_model_outputs(case, arrays["raw_ids"]))
        surface_pressure, surface_pressure_invariance = _accumulate_field(
            case,
            arrays,
            truth_key="surface_pressure_truth",
            prediction_key="surface_pressure_prediction",
            component_count=1,
            physical_weight_key="surface_area",
        )
        surface_wall_shear, surface_wall_shear_invariance = _accumulate_field(
            case,
            arrays,
            truth_key="surface_wall_shear_truth",
            prediction_key="surface_wall_shear_prediction",
            component_count=3,
            physical_weight_key="surface_area",
        )
        volume_pressure, volume_pressure_invariance = _accumulate_field(
            case,
            arrays,
            truth_key="pressure_truth",
            prediction_key="pressure_prediction",
            component_count=1,
        )
        volume_velocity, volume_velocity_invariance = _accumulate_field(
            case,
            arrays,
            truth_key="velocity_truth",
            prediction_key="velocity_prediction",
            component_count=3,
        )
        expected_statistics[case.case_id] = _metric_evidence(
            surface_pressure,
            surface_wall_shear,
            volume_pressure,
            volume_velocity,
        )
        expected_field_values[case.case_id] = _field_metric_values(
            surface_pressure,
            surface_wall_shear,
            volume_pressure,
            volume_velocity,
        )

        surface_ids = _support_ids(arrays["raw_ids"], domain="surface")
        volume_ids = _support_ids(arrays["raw_ids"], domain="volume")
        surface_coordinates = arrays["surface_coordinates"]
        volume_coordinates = arrays["coordinates"]
        surface_truth_table = {
            "columns": {
                "support_id": surface_ids,
                "x": surface_coordinates[:, 0].tolist(),
                "y": surface_coordinates[:, 1].tolist(),
                "z": surface_coordinates[:, 2].tolist(),
                "surface_area": arrays["surface_area"].tolist(),
                "pMeanTrim_truth": arrays["surface_pressure_truth"].tolist(),
                "wallShearStressMeanTrim_x_truth": arrays[
                    "surface_wall_shear_truth"
                ][:, 0].tolist(),
                "wallShearStressMeanTrim_y_truth": arrays[
                    "surface_wall_shear_truth"
                ][:, 1].tolist(),
                "wallShearStressMeanTrim_z_truth": arrays[
                    "surface_wall_shear_truth"
                ][:, 2].tolist(),
            }
        }
        surface_prediction_table = {
            "columns": {
                "support_id": surface_ids,
                "pMeanTrim_prediction": arrays[
                    "surface_pressure_prediction"
                ].tolist(),
                "wallShearStressMeanTrim_x_prediction": arrays[
                    "surface_wall_shear_prediction"
                ][:, 0].tolist(),
                "wallShearStressMeanTrim_y_prediction": arrays[
                    "surface_wall_shear_prediction"
                ][:, 1].tolist(),
                "wallShearStressMeanTrim_z_prediction": arrays[
                    "surface_wall_shear_prediction"
                ][:, 2].tolist(),
            }
        }
        volume_truth_table = {
            "columns": {
                "support_id": volume_ids,
                "x": volume_coordinates[:, 0].tolist(),
                "y": volume_coordinates[:, 1].tolist(),
                "z": volume_coordinates[:, 2].tolist(),
                "pMeanTrim_truth": arrays["pressure_truth"].tolist(),
                "UMeanTrim_x_truth": arrays["velocity_truth"][:, 0].tolist(),
                "UMeanTrim_y_truth": arrays["velocity_truth"][:, 1].tolist(),
                "UMeanTrim_z_truth": arrays["velocity_truth"][:, 2].tolist(),
            }
        }
        volume_prediction_table = {
            "columns": {
                "support_id": volume_ids,
                "pMeanTrim_prediction": arrays["pressure_prediction"].tolist(),
                "UMeanTrim_x_prediction": arrays["velocity_prediction"][:, 0].tolist(),
                "UMeanTrim_y_prediction": arrays["velocity_prediction"][:, 1].tolist(),
                "UMeanTrim_z_prediction": arrays["velocity_prediction"][:, 2].tolist(),
            }
        }
        surface_truth_path = (
            support_data_directory / f"{case.case_id}-surface-ground-truth.json"
        )
        volume_truth_path = (
            support_data_directory / f"{case.case_id}-volume-ground-truth.json"
        )
        surface_prediction_path = (
            prediction_data_directory / f"{case.case_id}-surface.json"
        )
        volume_prediction_path = (
            prediction_data_directory / f"{case.case_id}-volume.json"
        )
        _write_json(surface_truth_path, surface_truth_table)
        _write_json(volume_truth_path, volume_truth_table)
        _write_json(surface_prediction_path, surface_prediction_table)
        _write_json(volume_prediction_path, volume_prediction_table)
        case_instances.append(
            {
                "case_id": case.case_id,
                "support_instances": [
                    {
                        "support_id": SURFACE_SUPPORT_ID,
                        "entity_count": case.entity_count,
                        "artifacts": [
                            {
                                "role": "ground_truth_table",
                                "path": f"data/{surface_truth_path.name}",
                                "sha256": sha256_file(surface_truth_path),
                                "byte_size": surface_truth_path.stat().st_size,
                                "format": "json",
                            }
                        ],
                    },
                    {
                        "support_id": VOLUME_SUPPORT_ID,
                        "entity_count": case.entity_count,
                        "artifacts": [
                            {
                                "role": "ground_truth_table",
                                "path": f"data/{volume_truth_path.name}",
                                "sha256": sha256_file(volume_truth_path),
                                "byte_size": volume_truth_path.stat().st_size,
                                "format": "json",
                            }
                        ],
                    }
                ],
            }
        )
        prediction_cases.append(
            {
                "case_id": case.case_id,
                "files": [
                    {
                        "support_id": SURFACE_SUPPORT_ID,
                        "file": f"data/{surface_prediction_path.name}",
                        "sha256": sha256_file(surface_prediction_path),
                        "format": "json",
                        "row_count": case.entity_count,
                    },
                    {
                        "support_id": VOLUME_SUPPORT_ID,
                        "file": f"data/{volume_prediction_path.name}",
                        "sha256": sha256_file(volume_prediction_path),
                        "format": "json",
                        "row_count": case.entity_count,
                    }
                ],
            }
        )
        case_receipts.append(
            {
                "case_id": case.case_id,
                "synthetic": True,
                "support_counts": {
                    SURFACE_SUPPORT_ID: case.entity_count,
                    VOLUME_SUPPORT_ID: case.entity_count,
                },
                "transport": transport_receipt,
                "inference_chunks": {
                    "raw_id_half_open_ranges": [list(item) for item in case.chunk_ranges],
                    "processing_order_by_range_index": list(case.processing_order),
                    "complete_duplicate_free_coverage": True,
                    "independent_of_transport_part_boundaries": True,
                },
                "full_case_vs_chunked": {
                    "surface_pressure": surface_pressure_invariance,
                    "surface_wall_shear": surface_wall_shear_invariance,
                    "volume_pressure": volume_pressure_invariance,
                    "volume_velocity": volume_velocity_invariance,
                },
            }
        )

    support_case_directory = output / "support" / "case-sets" / CASE_SET_ID
    chunk_path = support_case_directory / "chunk-000.json"
    chunk = {
        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "dataset_id": DATASET_ID,
        "case_set_id": CASE_SET_ID,
        "cases": case_instances,
    }
    _write_json(chunk_path, chunk)
    index_path = support_case_directory / "index.json"
    index = {
        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "dataset_id": DATASET_ID,
        "case_set_id": CASE_SET_ID,
        "case_count": len(CASES),
        "chunks": [
            {
                "file": chunk_path.name,
                "sha256": sha256_file(chunk_path),
                "case_count": len(CASES),
                "case_ids": [case.case_id for case in CASES],
            }
        ],
    }
    _write_json(index_path, index)
    support_manifest_path = output / "support" / "manifest.json"
    support_manifest = {
        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "status": "candidate",
        "published_at": "2026-08-20T00:00:00Z",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "evaluation_reference_version": "synthetic-drivaerml-chunk-driver-v3",
        "coordinate_frame": {
            "id": "synthetic-cartesian-v1",
            "axis_order": ["x", "y", "z"],
            "handedness": "right",
            "length_unit": "m",
            "normalization": "none",
        },
        "supports": [
            _surface_support_definition(),
            _volume_support_definition(),
        ],
        "case_sets": [
            {
                "id": CASE_SET_ID,
                "case_count": len(CASES),
                "index_file": f"case-sets/{CASE_SET_ID}/index.json",
                "index_sha256": sha256_file(index_path),
            }
        ],
        "notes": (
            "Synthetic candidate support only. This is not DrivAerML data, active "
            "DrivAerML scoring support, a participant result, or owner approval."
        ),
    }
    _write_json(support_manifest_path, support_manifest)

    prediction_manifest_path = output / "predictions" / "manifest.json"
    prediction_manifest = {
        "$schema": "https://fluidsbench.org/schemas/v3/prediction-artifact.schema.json",
        "schema_version": "1.0",
        "format": "fluidsbench-prediction-artifact-v1",
        "artifact_id": "drivaerml-synthetic-complete-predictions-v1",
        "kind": "scored_predictions",
        "submission_id": SUBMISSION_ID,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "case_set_id": CASE_SET_ID,
        "support_release_id": RELEASE_ID,
        "support_manifest_sha256": sha256_file(support_manifest_path),
        "case_count": len(CASES),
        "cases": prediction_cases,
    }
    _write_json(prediction_manifest_path, prediction_manifest)

    case_metrics = evaluate_prediction_artifact(
        support_manifest_path=support_manifest_path,
        case_set_id=CASE_SET_ID,
        prediction_manifest_path=prediction_manifest_path,
        submission_id=SUBMISSION_ID,
        split_id=SPLIT_ID,
    )
    _assert_evaluator_matches_chunks(case_metrics, expected_statistics)
    _apply_candidate_field_metrics(case_metrics, expected_field_values)
    (
        profile_cases,
        profile_truth_cases,
        diagnostic_values,
        profile_r2_values,
    ) = _synthetic_profiles(candidate_spec, profile_registry)
    nonspatial_validation = _augment_candidate_nonspatial_metrics(
        case_metrics,
        candidate_spec=candidate_spec,
        diagnostic_values=diagnostic_values,
        profile_r2_values=profile_r2_values,
    )
    case_metrics_path = output / "metrics" / "cases.json"
    _write_json(case_metrics_path, case_metrics)

    package_schema_paths, submission_path = _write_submission_package(
        output,
        support_manifest_path=support_manifest_path,
        prediction_manifest_path=prediction_manifest_path,
        case_metrics_path=case_metrics_path,
        profile_cases=profile_cases,
        profile_truth_cases=profile_truth_cases,
    )

    schema_checks = [
        ("scoring-support/v1/manifest.schema.json", support_manifest_path),
        ("scoring-support/v1/case-index.schema.json", index_path),
        ("scoring-support/v1/case-chunk.schema.json", chunk_path),
        ("v3/prediction-artifact.schema.json", prediction_manifest_path),
        ("v3/case-metrics.schema.json", case_metrics_path),
    ]
    discretization_records_path = package_schema_paths.pop(
        "v3/discretization-case.schema.json"
    )
    schema_checks.extend(package_schema_paths.items())
    for schema_relative_path, value_path in schema_checks:
        _validate_schema(schema_relative_path, value_path)
    discretization_record_count = 0
    for line_number, line in enumerate(
        discretization_records_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        errors = _schema_errors(
            REPOSITORY_ROOT / "schemas" / "v3" / "discretization-case.schema.json",
            json.loads(line),
        )
        if errors:
            raise ValueError(
                f"{discretization_records_path}:{line_number} does not match "
                "v3/discretization-case.schema.json:\n" + "\n".join(errors)
            )
        discretization_record_count += 1
    if discretization_record_count != len(CASES):
        raise ValueError("synthetic discretization JSONL does not cover both cases")

    receipt = {
        "status": "candidate_demo_valid",
        "mode": "synthetic_candidate_dry_run",
        "claims": {
            "uses_real_drivaerml_data": False,
            "validates_native_vtp_or_vtu_parsing": False,
            "requires_geometric_volume_cell_weights": False,
            "includes_synthetic_surface_field_payloads": True,
            "includes_synthetic_force_coefficient_payloads": True,
            "includes_all_twenty_candidate_profile_shapes": True,
            "validates_real_surface_force_integration": False,
            "validates_official_autocfd5_or_cp_extraction": False,
            "schema_v3_dummy_submission_generated": True,
            "official_drivaerml_submission_generated": False,
            "eligible_for_leaderboard_or_scientific_claims": False,
            "owner_scientific_approval": False,
            "independent_participant_dry_run": False,
        },
        "official_submission_block": (
            "DrivAerML scoring support is owner_review_required and submissions_open "
            "is false; generated submission.json belongs only to a separate, "
            "fictional, ineligible synthetic namespace."
        ),
        "observed_candidate_contract": {
            "file": candidate_spec_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "dataset_id": candidate_spec["dataset_id"],
            "dataset_status": candidate_spec["status"],
            "scoring_support_status": scoring_state["status"],
            "submissions_open": scoring_state["submissions_open"],
        },
        "demonstrated": [
            "ordered two-part byte reconstruction for synthetic run_1",
            "ordered three-part byte reconstruction for synthetic run_44",
            "raw-cell IDs and complete duplicate-free inference chunk coverage",
            "additive sufficient statistics reduced once per complete case",
            "surface area-weighted and equal-polygon candidate field reductions",
            "volume equal-native-cell candidate field reductions",
            "full-case versus chunked invariance for all four candidate fields",
            "schema-v3 keyed surface and volume prediction artifact packaging",
            "coherent five-coefficient force payloads and dataset reductions",
            "all four Cp-cut and sixteen AutoCFD5 velocity-profile payload shapes",
            "generic evaluator generation of schema-v3 case metrics",
            "DrivAerML-specific per-entity vector MAE/RMSE projection",
            "complete schema-v3 dummy submission package construction",
            "public JSON Schema validation of generated artifacts",
        ],
        "cases": case_receipts,
        "metric_values": case_metrics["metric_values"],
        "candidate_nonspatial_validation": nonspatial_validation,
        "artifacts": {
            path.relative_to(output).as_posix(): {
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in (
                support_manifest_path,
                index_path,
                chunk_path,
                prediction_manifest_path,
                case_metrics_path,
                submission_path,
                output / "evaluation-evidence.json",
                output / "discretization.json",
                output / "discretization" / "cases.jsonl",
                output / "profiles" / "index.json",
                output / "profiles" / "chunk-000.json",
                output / "profiles" / "ground-truth-manifest.json",
                output / "profiles" / "ground-truth.json",
            )
        },
        "schema_checks": [item[0] for item in schema_checks]
        + ["v3/discretization-case.schema.json"],
    }
    receipt_path = output / "validation-receipt.json"
    _write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty directory for the generated synthetic dry-run bundle",
    )
    args = parser.parse_args()
    try:
        receipt = run_demo(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "CANDIDATE DEMO VALID: synthetic DrivAerML-shaped chunk/package dry run; "
        f"receipt={args.output.resolve() / 'validation-receipt.json'}"
    )
    print(receipt["official_submission_block"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
