from __future__ import annotations

import copy
import json
import math

import pytest

from scripts import assemble_hiliftaeroml_schema_v3_candidate as assembler
from scripts import validate_submission as validator


def _force_evidence() -> tuple[dict, dict]:
    coefficient_rows = [
        {
            "predicted_c_drag": 0.12,
            "truth_c_drag": 0.10,
            "predicted_c_lift": 1.10,
            "truth_c_lift": 1.00,
            "predicted_c_pitch": -0.12,
            "truth_c_pitch": -0.10,
        },
        {
            "predicted_c_drag": 0.25,
            "truth_c_drag": 0.30,
            "predicted_c_lift": 1.80,
            "truth_c_lift": 2.00,
            "predicted_c_pitch": -0.15,
            "truth_c_pitch": -0.20,
        },
    ]
    per_case = []
    error_rows = []
    for index, coefficients in enumerate(coefficient_rows):
        errors = {
            metric_id: abs(coefficients[prediction] - coefficients[truth])
            for metric_id, (truth, prediction) in (
                validator.HILIFT_FORCE_MAE_INPUTS.items()
            )
        }
        error_rows.append(errors)
        per_case.append(
            {
                "case_id": f"case-{index}",
                "supports": [
                    {
                        "support_id": validator.HILIFT_FORCE_SUPPORT_ID,
                        "support_count": 1,
                        "scored_count": 1,
                        "coverage_fraction": 1.0,
                        "weight_coverage_fraction": 1.0,
                        "unmapped_count": 0,
                        "extrapolated_count": 0,
                        "metric_values": dict(errors),
                        "metric_sufficient_statistics": {},
                    }
                ],
                "nonspatial_metric_values": dict(errors),
                "force_coefficients": coefficients,
            }
        )

    metric_values = {
        metric_id: math.fsum(row[metric_id] for row in error_rows)
        / len(error_rows)
        for metric_id in validator.HILIFT_FORCE_MAE_INPUTS
    }
    metric_values.update({"cd_r2": 0.855, "cl_r2": 0.9})
    return (
        {"dataset_id": "hiliftaeroml", "metric_values": metric_values},
        {"cases": per_case, "metric_values": dict(metric_values)},
    )


def _force_errors(submission: dict, case_metrics: dict) -> list[str]:
    errors: list[str] = []
    validator._validate_hiliftaeroml_force_metric_bindings(
        errors.append,
        submission=submission,
        case_metrics=case_metrics,
    )
    return errors


def test_assembler_exports_complete_force_recomputation_inputs() -> None:
    load = {
        "predicted_c_drag": 0.12,
        "truth_c_drag": 0.10,
        "predicted_c_lift": 1.10,
        "truth_c_lift": 1.00,
        "predicted_c_pitch": -0.12,
        "truth_c_pitch": -0.10,
    }
    assert assembler._force_coefficient_evidence(load) == load
    assert tuple(load) == assembler.HILIFT_FORCE_COEFFICIENT_FIELDS


def test_hilift_force_metrics_recompute_from_coefficient_evidence() -> None:
    submission, case_metrics = _force_evidence()
    assert _force_errors(submission, case_metrics) == []


def _validate_case_metrics(tmp_path, submission: dict, case_metrics: dict) -> list[str]:
    split_case_ids = [case["case_id"] for case in case_metrics["cases"]]
    support_manifest = {
        "supports": [
            {
                "id": validator.HILIFT_FORCE_SUPPORT_ID,
                "extrapolation_policy": "forbidden",
                "metric_bindings": [
                    {
                        "metric_id": metric_id,
                        "reduction": (
                            "mae"
                            if metric_id in validator.HILIFT_FORCE_MAE_INPUTS
                            else "r2"
                        ),
                        "aggregation": "all_test_cases",
                        "case_evidence": (
                            "metric_value"
                            if metric_id in validator.HILIFT_FORCE_MAE_INPUTS
                            else "aggregate_only"
                        ),
                    }
                    for metric_id in (
                        *validator.HILIFT_FORCE_MAE_INPUTS,
                        *validator.HILIFT_FORCE_R2_INPUTS,
                    )
                ],
            }
        ]
    }
    support_case_index = {
        "_loaded_cases": [
            {
                "case_id": case_id,
                "support_instances": [
                    {
                        "support_id": validator.HILIFT_FORCE_SUPPORT_ID,
                        "entity_count": 1,
                    }
                ],
            }
            for case_id in split_case_ids
        ]
    }
    submission.update(
        {
            "submission_id": "hilift-force-test",
            "split_id": "full",
            "case_set_id": "case-set-test",
            "scoring_support": {
                "release_id": "hilift-support-test",
                "manifest_sha256": "a" * 64,
            },
            "case_metrics": {
                "file": "metrics/cases.json",
                "sha256": "0" * 64,
                "case_count": len(split_case_ids),
            },
        }
    )
    case_metrics.update(
        {
            "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
            "schema_version": "1.0",
            "submission_id": submission["submission_id"],
            "dataset_id": submission["dataset_id"],
            "split_id": submission["split_id"],
            "case_set_id": submission["case_set_id"],
            "scoring_support_release_id": submission["scoring_support"]["release_id"],
            "scoring_support_manifest_sha256": submission["scoring_support"][
                "manifest_sha256"
            ],
            "case_count": len(split_case_ids),
        }
    )
    path = tmp_path / submission["case_metrics"]["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case_metrics, indent=2) + "\n", encoding="utf-8")
    submission["case_metrics"]["sha256"] = validator.sha256_file(path)
    errors: list[str] = []
    validator.validate_v3_case_metrics(
        errors.append,
        tmp_path,
        submission,
        split_case_ids,
        support_manifest,
        support_case_index,
    )
    return errors


def test_v3_case_metric_validation_invokes_force_recomputation(tmp_path) -> None:
    submission, case_metrics = _force_evidence()
    assert _validate_case_metrics(tmp_path, submission, case_metrics) == []


def test_v3_case_metric_validation_rejects_coordinated_aggregate_tamper(
    tmp_path,
) -> None:
    submission, case_metrics = _force_evidence()
    submission["metric_values"]["c_drag_mae"] += 0.1
    case_metrics["metric_values"]["c_drag_mae"] = submission["metric_values"][
        "c_drag_mae"
    ]
    errors = "\n".join(_validate_case_metrics(tmp_path, submission, case_metrics))
    assert "metric_values.c_drag_mae must equal the equal-case MAE reduction" in errors
    assert "metric_values.c_drag_mae must equal the independent" in errors


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda submission, cases: cases["cases"][0]["force_coefficients"].__setitem__(
                "predicted_c_drag", 0.13
            ),
            "abs(force_coefficients.predicted_c_drag - force_coefficients.truth_c_drag)",
        ),
        (
            lambda submission, cases: cases["cases"][1]["force_coefficients"].__setitem__(
                "truth_c_lift", 2.02
            ),
            "abs(force_coefficients.predicted_c_lift - force_coefficients.truth_c_lift)",
        ),
        (
            lambda submission, cases: submission["metric_values"].__setitem__(
                "c_pitch_mae", submission["metric_values"]["c_pitch_mae"] + 0.01
            ),
            "metric_values.c_pitch_mae must equal the independent",
        ),
        (
            lambda submission, cases: submission["metric_values"].__setitem__(
                "cd_r2", submission["metric_values"]["cd_r2"] - 0.01
            ),
            "metric_values.cd_r2 must equal the independent",
        ),
        (
            lambda submission, cases: cases["cases"][0][
                "nonspatial_metric_values"
            ].__setitem__("c_drag_mae", 0.5),
            "nonspatial_metric_values.c_drag_mae must equal abs",
        ),
        (
            lambda submission, cases: cases["cases"][0]["supports"][0][
                "metric_values"
            ].__setitem__("c_drag_mae", 0.5),
            "supports[aerodynamic-case-coefficients-v1].metric_values.c_drag_mae",
        ),
    ],
)
def test_hilift_force_tampering_is_rejected(mutation, expected_error: str) -> None:
    submission, case_metrics = _force_evidence()
    submission = copy.deepcopy(submission)
    case_metrics = copy.deepcopy(case_metrics)
    mutation(submission, case_metrics)
    assert expected_error in "\n".join(_force_errors(submission, case_metrics))


def test_hilift_force_coefficient_inventory_is_exact() -> None:
    submission, case_metrics = _force_evidence()
    del case_metrics["cases"][0]["force_coefficients"]["truth_c_pitch"]
    errors = "\n".join(_force_errors(submission, case_metrics))
    assert "force_coefficients must contain exactly" in errors
    assert "truth_c_pitch" in errors
