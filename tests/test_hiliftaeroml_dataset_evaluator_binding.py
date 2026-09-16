from __future__ import annotations

import copy
import json

import pytest

from scripts import assemble_hiliftaeroml_schema_v3_candidate as assembler
from scripts import validate_submission as validator


REFERENCE_VERSION = "hiliftaeroml-evaluator-v0.1-candidate"
EVALUATOR_REVISION = "e" * 40


def _dataset_spec() -> dict:
    return {
        "scoring_support": {
            "dataset_evaluator_binding": {
                "status": "frozen",
                "evaluator_reference_version": REFERENCE_VERSION,
                "evaluator_code_revision": EVALUATOR_REVISION,
            }
        }
    }


def _submission_and_evidence() -> tuple[dict, dict]:
    metric_values = {"overall_score": 0.5}
    submission = {
        "schema_version": "3.0",
        "submission_id": "hilift-evaluator-binding-test",
        "dataset_id": "hiliftaeroml",
        "dataset_version": "1.0",
        "split_id": "full",
        "split_sha256": "1" * 64,
        "case_set_id": "case-set-test",
        "evaluation": {
            "reference_version": REFERENCE_VERSION,
            "command": "python evaluate.py",
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": "0" * 64,
        },
        "profile_data": {
            "index_file": "profiles/index.json",
            "profile_ground_truth_release_id": "candidate-truth-v1",
            "profile_ground_truth_manifest_sha256": "2" * 64,
        },
        "scoring_support": {
            "release_id": "candidate-support-v1",
            "manifest_sha256": "3" * 64,
        },
        "spatial_discretization": {"sha256": "4" * 64},
        "case_metrics": {"sha256": "5" * 64},
        "metric_values": metric_values,
    }
    evidence = {
        "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
        "schema_version": "3.0",
        "submission_id": submission["submission_id"],
        "dataset_id": submission["dataset_id"],
        "dataset_version": submission["dataset_version"],
        "split_id": submission["split_id"],
        "split_sha256": submission["split_sha256"],
        "case_set_id": submission["case_set_id"],
        "reference_version": submission["evaluation"]["reference_version"],
        "dataset_evaluator_binding": {
            "status": "frozen",
            "reference_version": REFERENCE_VERSION,
            "code_revision": EVALUATOR_REVISION,
        },
        "command": submission["evaluation"]["command"],
        "generated_at": "2026-09-02T12:00:00Z",
        "status": "submitted_evaluation",
        "metric_values": metric_values,
        "profile_index_sha256": "6" * 64,
        "profile_ground_truth_release_id": submission["profile_data"][
            "profile_ground_truth_release_id"
        ],
        "profile_ground_truth_manifest_sha256": submission["profile_data"][
            "profile_ground_truth_manifest_sha256"
        ],
        "scoring_support_release_id": submission["scoring_support"]["release_id"],
        "scoring_support_manifest_sha256": submission["scoring_support"][
            "manifest_sha256"
        ],
        "discretization_sha256": submission["spatial_discretization"]["sha256"],
        "case_metrics_sha256": submission["case_metrics"]["sha256"],
    }
    return submission, evidence


def _validate(tmp_path, submission: dict, evidence: dict) -> list[str]:
    evidence_path = tmp_path / "evaluation-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    submission["evaluation"]["evidence_sha256"] = validator.sha256_file(evidence_path)
    errors: list[str] = []
    validator.validate_evaluation_evidence(
        errors.append,
        tmp_path,
        submission,
        _dataset_spec(),
    )
    return errors


def test_assembler_emits_distinct_frozen_dataset_evaluator_identity() -> None:
    assert assembler._dataset_evaluator_evidence(
        {
            "reference_version": REFERENCE_VERSION,
            "code_revision": EVALUATOR_REVISION,
        }
    ) == {
        "status": "frozen",
        "reference_version": REFERENCE_VERSION,
        "code_revision": EVALUATOR_REVISION,
    }


def test_dataset_evaluator_binding_is_schema_valid_and_matches_spec(tmp_path) -> None:
    submission, evidence = _submission_and_evidence()
    assert validator.schema_errors(
        evidence,
        "evaluation-evidence.schema.json",
        schema_version="v3",
    ) == []
    assert _validate(tmp_path, submission, evidence) == []


def test_hilift_evidence_requires_dataset_evaluator_binding(tmp_path) -> None:
    submission, evidence = _submission_and_evidence()
    del evidence["dataset_evaluator_binding"]
    errors = "\n".join(_validate(tmp_path, submission, evidence))
    assert "requires a structured dataset_evaluator_binding" in errors


@pytest.mark.parametrize("field", ["reference_version", "code_revision"])
def test_hilift_evidence_rejects_dataset_evaluator_tampering(
    tmp_path, field: str
) -> None:
    submission, evidence = _submission_and_evidence()
    tampered = copy.deepcopy(evidence)
    tampered["dataset_evaluator_binding"][field] = (
        "different-evaluator" if field == "reference_version" else "f" * 40
    )
    errors = "\n".join(_validate(tmp_path, submission, tampered))
    assert f"dataset_evaluator_binding.{field} must equal" in errors


def test_unrelated_v3_evidence_does_not_require_dataset_evaluator_binding(
    tmp_path,
) -> None:
    submission, evidence = _submission_and_evidence()
    submission["dataset_id"] = "example"
    evidence["dataset_id"] = "example"
    del evidence["dataset_evaluator_binding"]
    evidence_path = tmp_path / "evaluation-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    submission["evaluation"]["evidence_sha256"] = validator.sha256_file(evidence_path)
    errors: list[str] = []
    validator.validate_evaluation_evidence(
        errors.append,
        tmp_path,
        submission,
    )
    assert errors == []
