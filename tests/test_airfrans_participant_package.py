"""Repackage retained predictions as disposable test fixtures, never real results."""

import json
import shutil

import pytest

from reference.airfrans import participant as w


def retained_evaluation(tmp_path, split_id):
    from tests.airfrans_fixture import make_fixture
    source = make_fixture(tmp_path / 'synthetic-input', split_id)
    original = w.load_json(source / "submission.json")
    submission_id = f"participant-test-{split_id.replace('_', '-')}-v1"
    spec = w.load_json(w.SPEC_PATH)
    split = w.official_split(split_id)
    entry = {
        "schema": "fluidsbench-airfrans-entry-v1",
        "submission_id": submission_id,
        "split_id": split_id,
    }
    campaign = {
        "schema": "fluidsbench-airfrans-campaign-v1",
        "entry": entry,
        "case_ids": split["case_ids"],
        "case_set_id": split["case_set_id"],
        "inputs": {cid: {} for cid in split["case_ids"]},
        "bindings": w.bindings(),
        "evaluator": w.evaluator_identity(),
    }
    output = tmp_path / "work"
    root = output / "evaluation"
    w.write_json(output / "campaign.json", campaign)
    profile_index = w.load_json(source / "profiles/index.json")
    profiles = {}
    for chunk in profile_index["chunks"]:
        for case in w.load_json(source / "profiles" / chunk["file"])["cases"]:
            profiles[case["case_id"]] = case
    checkpoints = {}
    for cid in campaign["case_ids"]:
        # Only the packaging/recovery boundary is exercised here; native field
        # derivation and numerical reduction have separate synthetic tests.
        case_root = output / "cases" / cid
        w.write_json(case_root / "case.json", {"case_id": cid, "files": []})
        w.write_json(
            case_root / "checkpoint.json",
            {
                "campaign_sha256": w.digest(campaign),
                "input": {},
                "files": w.file_tree(case_root),
            },
        )
        checkpoints[cid] = w.sha256_file(case_root / "checkpoint.json")
        w.write_json(root / "profiles" / f"{cid}.json", {"cases": [profiles[cid]]})
    metrics = w.load_json(source / "metrics/cases.json")
    metrics.update(
        {
            "submission_id": submission_id,
            "split_id": split_id,
            "scoring_support_release_id": spec["scoring_support"]["release_id"],
            "scoring_support_manifest_sha256": w.bindings()["native"],
        }
    )
    if "dataset_version" in metrics:
        metrics["dataset_version"] = spec["dataset_version"]
    if "reference_version" in metrics:
        metrics["reference_version"] = spec["evaluation_reference_version"]
    w.write_json(root / "metrics-cases.json", metrics)
    w.write_json(
        root / "profile-score.json",
        {"value": metrics["metric_values"]["velocity_profile_r2"]},
    )
    shutil.copyfile(
        w.SPEC_ROOT / spec["scoring_support"]["manifest_file"],
        root / "support-manifest.json",
    )
    w.write_json(root / "profile-truth.json", {"cases": list(profiles.values())})
    w.write_json(
        root / "workflow.json",
        {
            "schema": "fluidsbench-airfrans-evaluation-v1",
            "campaign": campaign,
            "files": w.file_tree(root),
            "case_checkpoints": checkpoints,
        },
    )
    spatial = w.load_json(source / "discretization.json")
    config = {
        "submission_id": submission_id,
        "split_id": split_id,
        "result_revision": {
            "series_id": submission_id,
            "version": 1,
            "supersedes": None,
            "change_summary": "Disposable participant packaging test.",
        },
        "participant": {
            key: original[key]
            for key in (
                "model",
                "model_type",
                "model_types",
                "training_regime",
                "target_data_used",
                "external_pretraining",
                "pretraining_data",
                "methodology",
                "submitter_name",
                "institution",
                "paper_url",
            )
        },
        "reproducibility": original["reproducibility"],
        "spatial_discretization": {
            key: spatial[key] for key in ("training", "inference")
        },
    }
    config["participant"]["methodology"][
        "record_note"
    ] = "Disposable packaging test using retained records; not a new model result."
    w.write_json(tmp_path / "config.json", config)
    rows = [
        json.loads(line)
        for line in (source / "discretization/cases.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row.update(submission_id=submission_id, split_id=split_id)
    (tmp_path / "cases.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return root, config, rows


@pytest.mark.parametrize("split_id", ["full", "scarce", "aoa_extrapolation"])
def test_complete_current_dev_package(tmp_path, split_id):
    root, config, rows = retained_evaluation(tmp_path, split_id)
    target = tmp_path / config["submission_id"]
    result = w.package(root, tmp_path / "config.json", tmp_path / "cases.jsonl", target)
    assert result["status"] == "validated_package"
    submission = w.load_json(target / "submission.json")
    assert submission["scoring_support"]["status"] == "official"
    assert "approval" not in submission
    assert submission["dataset_version"] == "airfrans-native-v1"
    assert submission["evaluation"]["reference_version"] == "airfrans-scoring-v2"
    assert (
        "scripts/airfrans.py evaluate-entry"
        in w.load_json(target / "evaluation-evidence.json")["command"]
    )
    if split_id == "full":
        from scripts import validate_submission as validator

        manifest = validator.manifest_with_benchmark_contract(
            validator.load_json(validator.MANIFEST_PATH)
        )
        errors, _ = validator.validate_submission_file(
            target / "submission.json", manifest, contributor_stage=True
        )
        assert len(errors) == 1 and "must be stored at" in errors[0]
    assert (
        w.load_json(target / "discretization.json")["inference"]
        == config["spatial_discretization"]["inference"]
    )
    assert (
        w.report(root, rows[0]["case_id"], tmp_path / "report.html")["profile_count"]
        == 8
    )
    submission["metric_values"]["overall_score"] += 1
    w.write_json(target / "submission.json", submission)
    with pytest.raises(w.WorkflowError, match="package validation failed"):
        w.verify_package(target)


def test_invalid_metadata_never_leaves_partial_package(tmp_path):
    root, config, _ = retained_evaluation(tmp_path, "full")
    config["participant"]["methodology"]["architecture"]["total_parameter_count"] += 10
    w.write_json(tmp_path / "config.json", config)
    target = tmp_path / config["submission_id"]
    with pytest.raises((w.WorkflowError, RuntimeError)):
        w.package(root, tmp_path / "config.json", tmp_path / "cases.jsonl", target)
    assert not target.exists()


def test_discretization_and_checkpoint_must_match(tmp_path):
    root, config, rows = retained_evaluation(tmp_path, "full")
    checkpoint = tmp_path / "wrong-checkpoint"
    checkpoint.write_bytes(b"not the model")
    with pytest.raises(w.WorkflowError, match="checkpoint bytes"):
        w.package(
            root,
            tmp_path / "config.json",
            tmp_path / "cases.jsonl",
            tmp_path / "bad",
            {"unknown": checkpoint},
        )
    (tmp_path / "cases.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(w.WorkflowError, match="exact ordered"):
        w.package(
            root, tmp_path / "config.json", tmp_path / "cases.jsonl", tmp_path / "bad"
        )
