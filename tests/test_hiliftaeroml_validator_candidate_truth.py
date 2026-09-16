from __future__ import annotations

import copy
from pathlib import Path

from scripts import validate_submission as validator


def _submission() -> dict:
    return {
        "schema_version": "3.0",
        "submission_id": "hilift-candidate-test",
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "case_set_id": "caseset-test",
        "methodology": {"record_kind": "submitter_reported"},
        "evaluation": {},
        "reproducibility": {
            "contract_version": "open-reproducibility-3.0",
        },
        "profile_data": {
            "profile_ground_truth_release_id": "candidate-truth-v1",
            "profile_ground_truth_manifest_sha256": "a" * 64,
        },
        "metric_values": {
            "cp_cut_r2": 0.5,
            "velocity_profile_r2": 0.75,
        },
    }


def _case_metrics() -> dict:
    return {
        "cases": [
            {
                "case_id": "case-a",
                "nonspatial_metric_values": {
                    "cp_cut_r2": 0.25,
                    "velocity_profile_r2": 0.5,
                },
            },
            {
                "case_id": "case-b",
                "nonspatial_metric_values": {
                    "cp_cut_r2": 0.75,
                    "velocity_profile_r2": 1.0,
                },
            },
        ]
    }


def _scores() -> dict[str, dict[str, float]]:
    return {
        "case-a": {"cp_cut_r2": 0.25, "velocity_profile_r2": 0.5},
        "case-b": {"cp_cut_r2": 0.75, "velocity_profile_r2": 1.0},
    }


def test_hidden_profile_scores_bind_per_case_and_macro_aggregates() -> None:
    errors: list[str] = []
    validator._validate_hilift_native_profile_score_bindings(
        errors.append,
        submission=_submission(),
        case_metrics=_case_metrics(),
        recomputed_scores=_scores(),
    )
    assert errors == []


def test_hidden_profile_score_mismatches_are_rejected() -> None:
    submission = _submission()
    submission["metric_values"]["velocity_profile_r2"] = 0.7
    case_metrics = _case_metrics()
    case_metrics["cases"][0]["nonspatial_metric_values"]["cp_cut_r2"] = 0.2
    errors: list[str] = []
    validator._validate_hilift_native_profile_score_bindings(
        errors.append,
        submission=submission,
        case_metrics=case_metrics,
        recomputed_scores=_scores(),
    )
    joined = "\n".join(errors)
    assert "case-a nonspatial_metric_values.cp_cut_r2" in joined
    assert "metric_values.velocity_profile_r2" in joined


def test_candidate_truth_release_is_rejected_outside_candidate_mode() -> None:
    errors, totals = validator.validate_many(
        [],
        candidate_profile_truth_release=Path("/tmp/inactive-candidate-truth"),
    )
    assert errors == [
        "--candidate-profile-truth-release requires --candidate-dry-run"
    ]
    assert totals == {"submissions": 0, "cases": 0, "series": 0}


def test_candidate_compact_support_release_is_rejected_outside_candidate_mode() -> None:
    errors, totals = validator.validate_many(
        [],
        candidate_compact_profile_support_release=Path(
            "/tmp/inactive-candidate-compact-support"
        ),
    )
    assert errors == [
        "--candidate-compact-profile-support-release requires --candidate-dry-run"
    ]
    assert totals == {"submissions": 0, "cases": 0, "series": 0}


def test_candidate_compact_support_preopen_cache_is_exact_per_case_set(
    tmp_path: Path, monkeypatch
) -> None:
    support_root = tmp_path / "support"
    support_root.mkdir()
    manifest = {
        "datasets": [
            {
                "slug": "hiliftaeroml",
                "splits": [{"id": "full"}],
            }
        ]
    }
    submission = {
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "profile_data": {"format": validator.HILIFT_COMPACT_PROFILE_FORMAT},
    }
    specification = {
        "splits": [
            {
                "id": "full",
                "case_set_id": "caseset-compact-test",
                "index_file": "splits/full.json",
            }
        ],
        "compact_profile_definition": {
            "candidate_dry_run_evaluator_support": {
                "manifest_sha256": "c" * 64,
            }
        },
    }
    split_index = {
        "case_set_id": "caseset-compact-test",
        "case_ids": ["case-0001"],
    }
    calls: list[dict] = []
    handle = object()

    monkeypatch.setattr(validator, "load_submission_json", lambda _path: submission)
    monkeypatch.setattr(
        validator,
        "load_json",
        lambda path: specification if path.name == "submission-spec.json" else split_index,
    )

    def open_support(**kwargs):
        calls.append(kwargs)
        return handle

    monkeypatch.setattr(validator, "open_hilift_compact_support_release", open_support)
    cache: dict = {}
    first = validator._preopened_hilift_compact_support_for_submission(
        path=tmp_path / "submission.json",
        manifest=manifest,
        support_release_root=support_root,
        cache=cache,
    )
    second = validator._preopened_hilift_compact_support_for_submission(
        path=tmp_path / "submission.json",
        manifest=manifest,
        support_release_root=support_root,
        cache=cache,
    )
    assert first is handle
    assert second is handle
    assert calls == [
        {
            "release_root": support_root,
            "expected_manifest_sha256": "c" * 64,
            "expected_case_ids": ["case-0001"],
            "case_set_id": "caseset-compact-test",
        }
    ]


def test_candidate_reproducibility_uses_candidate_truth_declaration() -> None:
    submission = _submission()
    dataset_spec = {
        "status": "owner_review_required",
        "profile_definition": {
            "profile_ground_truth": {
                "status": "not_published",
                "release_id": None,
                "manifest_sha256": None,
            },
            "candidate_dry_run_profile_ground_truth": {
                "status": "complete_candidate_not_published",
                "release_id": "candidate-truth-v1",
                "manifest_sha256": "a" * 64,
            },
        },
    }
    manifest = {
        "data_release": {
            "profile_ground_truth": {
                "release_id": "unrelated-active-truth",
                "manifest_sha256": "b" * 64,
            }
        }
    }
    errors: list[str] = []
    validator.validate_open_reproducibility(
        errors.append,
        Path("/tmp/does-not-exist"),
        submission,
        {"status": "submitted_evaluation"},
        manifest,
        dataset_spec,
        {"index_file": "does-not-exist.json"},
        contributor_stage=False,
        candidate_dry_run=True,
    )
    assert not [error for error in errors if "profile_ground_truth" in error]


def test_unpublished_public_truth_closes_hilift_native_profile_intake(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    submission = {
        "submission_id": "hilift-candidate-test",
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "case_set_id": "caseset-test",
        "profile_data": {
            "format": validator.HILIFT_PROFILE_FORMAT,
            "index_file": "profiles/index.json",
            "case_count": 1,
            "case_set_id": "caseset-test",
        },
    }
    dataset_spec = {
        "profile_definition": {
            "profile_ground_truth": {
                "status": "not_published",
                "release_id": None,
                "manifest_sha256": None,
            },
            "candidate_dry_run_profile_ground_truth": {
                "release_id": "candidate-truth-v1",
                "manifest_sha256": "a" * 64,
            },
        }
    }
    split_entry = {"index_file": "splits/full.json"}

    public_errors: list[str] = []
    validator.validate_profiles(
        public_errors.append,
        tmp_path,
        submission,
        dataset_spec,
        split_entry,
    )
    assert "native profile intake is closed" in "\n".join(public_errors)

    candidate_errors: list[str] = []
    validator.validate_profiles(
        candidate_errors.append,
        tmp_path,
        submission,
        dataset_spec,
        split_entry,
        candidate_dry_run=True,
    )
    assert "explicit local candidate profile-truth release" in "\n".join(
        candidate_errors
    )


def test_compact_profile_intake_requires_the_explicit_bound_support_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    submission = {
        "submission_id": "hilift-compact-candidate-test",
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "case_set_id": "caseset-test",
        "profile_data": {
            "format": validator.HILIFT_COMPACT_PROFILE_FORMAT,
            "index_file": "profiles/index.json",
            "case_count": 1,
            "case_set_id": "caseset-test",
            "profile_ground_truth_release_id": "candidate-truth-v1",
            "profile_ground_truth_manifest_sha256": "a" * 64,
            "evaluator_support_release_id": "compact-support-v2-candidate",
            "evaluator_support_manifest_sha256": "b" * 64,
        },
    }
    dataset_spec = {
        "profile_definition": {
            "candidate_dry_run_profile_ground_truth": {
                "release_id": "candidate-truth-v1",
                "manifest_sha256": "a" * 64,
            }
        },
        "compact_profile_definition": {
            "status": "additive_candidate_not_bound",
            "contract_id": validator.HILIFT_COMPACT_PROFILE_CONTRACT_ID,
            "format": validator.HILIFT_COMPACT_PROFILE_FORMAT,
            "sha256": validator.HILIFT_COMPACT_PROFILE_CONTRACT_SHA256,
            "evaluator_support": {
                "status": "not_published",
                "release_id": None,
                "manifest_sha256": None,
            },
            "candidate_dry_run_evaluator_support": {
                "status": "complete_candidate_not_published",
                "usage": "maintainer_local_candidate_dry_run_only",
                "release_id": "compact-support-v2-candidate",
                "manifest_sha256": "b" * 64,
            },
        },
    }
    errors: list[str] = []
    validator.validate_profiles(
        errors.append,
        tmp_path,
        submission,
        dataset_spec,
        {"index_file": "splits/full.json"},
        candidate_dry_run=True,
    )
    joined = "\n".join(errors)
    assert "explicit local candidate compact evaluator-support release" in joined

    public_errors: list[str] = []
    validator.validate_profiles(
        public_errors.append,
        tmp_path,
        submission,
        dataset_spec,
        {"index_file": "splits/full.json"},
    )
    assert "compact profile intake is closed" in "\n".join(public_errors)


def test_compact_profile_implementation_provenance_is_exact_and_cross_bound() -> None:
    implementation = copy.deepcopy(
        validator.HILIFT_COMPACT_PROFILE_IMPLEMENTATION_BINDING
    )
    revision = "1" * 40
    submission = {
        "evaluation": {"reference_version": "hilift-evaluator-v1"},
        "profile_data": {
            "format": validator.HILIFT_COMPACT_PROFILE_FORMAT,
            "compact_profile_implementation_binding": copy.deepcopy(
                implementation
            ),
        },
    }
    evidence = {
        "dataset_evaluator_binding": {
            "status": "frozen",
            "reference_version": "hilift-evaluator-v1",
            "code_revision": revision,
        },
        "compact_profile_implementation_binding": copy.deepcopy(implementation),
    }
    dataset_spec = {
        "scoring_support": {
            "dataset_evaluator_binding": {
                "status": "frozen",
                "evaluator_reference_version": "hilift-evaluator-v1",
                "evaluator_code_revision": revision,
            }
        }
    }

    errors: list[str] = []
    validator._validate_hiliftaeroml_dataset_evaluator_binding(
        errors.append,
        submission=submission,
        evidence=evidence,
        dataset_spec=dataset_spec,
    )
    assert errors == []

    falsely_frozen = copy.deepcopy(evidence)
    falsely_frozen["compact_profile_implementation_binding"].update(
        {"status": "frozen", "code_revision": revision}
    )
    frozen_errors: list[str] = []
    validator._validate_hiliftaeroml_dataset_evaluator_binding(
        frozen_errors.append,
        submission=submission,
        evidence=falsely_frozen,
        dataset_spec=dataset_spec,
    )
    joined = "\n".join(frozen_errors)
    assert "exact unbound compact_profile_implementation_binding" in joined
    assert "must exactly match submission.json profile_data" in joined

    native_submission = copy.deepcopy(submission)
    native_submission["profile_data"]["format"] = validator.HILIFT_PROFILE_FORMAT
    native_errors: list[str] = []
    validator._validate_hiliftaeroml_dataset_evaluator_binding(
        native_errors.append,
        submission=native_submission,
        evidence=evidence,
        dataset_spec=dataset_spec,
    )
    assert native_errors == [
        "compact_profile_implementation_binding is permitted only for the "
        "HiLiftAeroML compact-v2 profile format"
    ]
