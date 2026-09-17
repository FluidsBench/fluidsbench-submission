from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reference.ahmedml.pre_release import (
    AhmedMLPreReleaseError,
    GENUINE_INFERENCE_ATTESTATION,
    REGISTRY_SCHEMA,
    REGISTRY_STATUS,
    build_registry_entry,
    package_tree_binding,
)
from scripts import manage_leaderboard
from scripts.register_ahmedml_pre_release import _registry
from scripts.validate_submission import (
    is_registered_ahmedml_pre_release_path,
    registered_ahmedml_pre_release_reference,
)


SUBMISSION_ID = "ahmedml-real-model-full-v1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_package(root: Path) -> tuple[Path, dict[str, object]]:
    directory = root / "submissions" / "ahmedml" / SUBMISSION_ID
    evidence = {
        "status": "submitted_evaluation",
        "submission_id": SUBMISSION_ID,
        "dataset_id": "ahmedml",
        "split_id": "full",
        "case_set_id": "full-test",
        "notes": GENUINE_INFERENCE_ATTESTATION,
    }
    write_json(directory / "evaluation-evidence.json", evidence)
    write_json(directory / "metrics" / "cases.json", {"cases": []})
    series = [
        {
            "panel_id": "pressure_profiles" if index < 3 else "velocity_profiles",
            "station_id": f"station-{index}",
            "quantity_id": "cp" if index < 3 else "ux_over_uinf",
            "coordinate": [0.0, 1.0],
            "prediction": [0.0, 0.0],
        }
        for index in range(7)
    ]
    chunk_path = directory / "profiles" / "chunk-000.json"
    write_json(
        chunk_path,
        {"schema_version": "1.0", "cases": [{"case_id": "run_4", "series": series}]},
    )
    write_json(
        directory / "profiles" / "index.json",
        {
            "case_count": 1,
            "chunks": [
                {
                    "file": "chunk-000.json",
                    "case_ids": ["run_4"],
                    "sha256": digest(chunk_path),
                }
            ],
        },
    )
    cases_path = directory / "discretization" / "cases.jsonl"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text('{"case_id":"run_4"}\n', encoding="utf-8")
    write_json(
        directory / "discretization.json",
        {
            "case_manifest": {
                "file": "discretization/cases.jsonl",
                "sha256": digest(cases_path),
            }
        },
    )
    write_json(directory / "regional-diagnostics.json", {"case_ids": ["run_4"]})
    submission: dict[str, object] = {
        "schema_version": "3.0",
        "submission_id": SUBMISSION_ID,
        "dataset": "AhmedML",
        "dataset_id": "ahmedml",
        "prediction_scope": "surface_and_volume",
        "split_id": "full",
        "case_set_id": "full-test",
        "scoring_support": {"status": "candidate"},
        "methodology": {
            "record_kind": "submitter_reported",
            "checkpoints": [{"id": "model", "sha256": "a" * 64}],
        },
        "evaluation": {
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": digest(directory / "evaluation-evidence.json"),
        },
        "case_metrics": {
            "file": "metrics/cases.json",
            "case_count": 1,
            "sha256": digest(directory / "metrics" / "cases.json"),
        },
        "profile_data": {"index_file": "profiles/index.json"},
        "spatial_discretization": {
            "file": "discretization.json",
            "sha256": digest(directory / "discretization.json"),
        },
        "regional_diagnostics": {
            "file": "regional-diagnostics.json",
            "sha256": digest(directory / "regional-diagnostics.json"),
        },
    }
    evidence["profile_index_sha256"] = digest(directory / "profiles" / "index.json")
    write_json(directory / "evaluation-evidence.json", evidence)
    submission["evaluation"]["evidence_sha256"] = digest(  # type: ignore[index]
        directory / "evaluation-evidence.json"
    )
    path = directory / "submission.json"
    write_json(path, submission)
    return path, submission


def write_registry(root: Path, entry: dict[str, object]) -> None:
    write_json(
        root / "benchmark-specs" / "ahmedml" / "pre-release-reference-registry.json",
        {
            "schema": REGISTRY_SCHEMA,
            "status": REGISTRY_STATUS,
            "dataset_id": "ahmedml",
            "activation": {
                "owner_approval_complete": False,
                "published": False,
                "submissions_opened": False,
                "registration_changes_activation": False,
            },
            "entries": [entry],
        },
    )


def test_exact_genuine_inference_package_is_registered_dev_only(tmp_path: Path) -> None:
    path, submission = make_package(tmp_path)
    entry = build_registry_entry(path, repository_root=tmp_path)
    write_registry(tmp_path, entry)
    manifest = {"data_release": {"status": "prototype_dummy_data"}}

    assert is_registered_ahmedml_pre_release_path(path, root=tmp_path)
    assert (
        registered_ahmedml_pre_release_reference(
            path, submission, manifest, root=tmp_path
        )
        == entry
    )
    eligibility = manage_leaderboard.claim_eligibility(
        "prototype_dummy_data",
        {"dataset_id": "ahmedml", "record_type": "pre_release_reference"},
    )
    assert eligibility["academic_citation"] is False
    assert eligibility["promotion"] is False
    assert "AhmedML" in eligibility["reason"]


def test_changed_package_byte_invalidates_registration(tmp_path: Path) -> None:
    path, submission = make_package(tmp_path)
    entry = build_registry_entry(path, repository_root=tmp_path)
    write_registry(tmp_path, entry)
    write_json(path.parent / "regional-diagnostics.json", {"changed": True})
    assert (
        registered_ahmedml_pre_release_reference(
            path,
            submission,
            {"data_release": {"status": "prototype_dummy_data"}},
            root=tmp_path,
        )
        is None
    )


def test_exact_registration_enters_only_the_prototype_source_feed(
    tmp_path: Path,
) -> None:
    path, _submission = make_package(tmp_path)
    write_registry(
        tmp_path,
        build_registry_entry(path, repository_root=tmp_path),
    )
    manifest = {
        "data_release": {"status": "prototype_dummy_data"},
        "datasets": [{"name": "AhmedML"}],
    }
    with (
        patch.object(manage_leaderboard, "ROOT", tmp_path),
        patch.object(manage_leaderboard, "submission_files", return_value=[path]),
    ):
        rows = manage_leaderboard.source_rows_by_dataset(manifest)["AhmedML"]
    assert len(rows) == 1
    assert rows[0]["submission_id"] == SUBMISSION_ID
    assert rows[0]["record_type"] == "pre_release_reference"
    assert rows[0].get("approval") is None


def test_registration_is_never_active_in_official_release(tmp_path: Path) -> None:
    path, submission = make_package(tmp_path)
    entry = build_registry_entry(path, repository_root=tmp_path)
    write_registry(tmp_path, entry)
    assert (
        registered_ahmedml_pre_release_reference(
            path,
            submission,
            {"data_release": {"status": "official"}},
            root=tmp_path,
        )
        is None
    )


def test_fixture_methodology_cannot_receive_genuine_binding(tmp_path: Path) -> None:
    path, submission = make_package(tmp_path)
    submission["methodology"]["record_kind"] = "prototype_fixture"  # type: ignore[index]
    write_json(path, submission)
    with pytest.raises(AhmedMLPreReleaseError, match="genuine-inference"):
        build_registry_entry(path, repository_root=tmp_path)


def test_approval_key_cannot_receive_pre_release_binding(tmp_path: Path) -> None:
    path, submission = make_package(tmp_path)
    submission["approval"] = None
    write_json(path, submission)
    with pytest.raises(AhmedMLPreReleaseError, match="unapproved"):
        build_registry_entry(path, repository_root=tmp_path)


def test_malformed_scoring_support_fails_closed(tmp_path: Path) -> None:
    path, submission = make_package(tmp_path)
    submission["scoring_support"] = []
    write_json(path, submission)
    with pytest.raises(AhmedMLPreReleaseError, match="genuine-inference"):
        build_registry_entry(path, repository_root=tmp_path)


def test_registry_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        "{\"schema\":\"ahmedml-pre-release-reference-registry-v1\","
        "\"status\":\"closed_candidate_dev_only\",\"dataset_id\":\"ahmedml\","
        "\"activation\":{},\"entries\":[],\"entries\":[]}\n",
        encoding="utf-8",
    )
    with pytest.raises(AhmedMLPreReleaseError, match="cannot read registry"):
        _registry(registry_path)


def test_logical_tree_rejects_symbolic_links(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    target = package / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    (package / "alias.json").symlink_to(target)
    with pytest.raises(AhmedMLPreReleaseError, match="symbolic link"):
        package_tree_binding(package)
