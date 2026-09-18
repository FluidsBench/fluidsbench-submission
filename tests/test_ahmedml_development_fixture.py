from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_submission import (
    AHMEDML_DEVELOPMENT_FIXTURE_ID,
    registered_ahmedml_development_fixture,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    directory = (
        root
        / "submissions"
        / "ahmedml"
        / AHMEDML_DEVELOPMENT_FIXTURE_ID
    )
    evidence_path = directory / "evaluation-evidence.json"
    evidence = {
        "status": "submitted_evaluation",
        "submission_id": AHMEDML_DEVELOPMENT_FIXTURE_ID,
        "dataset_id": "ahmedml",
        "notes": (
            "Non-ranked synthetic development fixture. The calibration "
            "checkpoint was not executed."
        ),
    }
    _write_json(evidence_path, evidence)
    submission: dict[str, object] = {
        "schema_version": "3.0",
        "submission_id": AHMEDML_DEVELOPMENT_FIXTURE_ID,
        "dataset_id": "ahmedml",
        "dataset": "AhmedML",
        "split_id": "full",
        "scoring_support": {"status": "candidate"},
        "methodology": {"record_kind": "prototype_fixture"},
        "note": "DEVELOPMENT FIXTURE ONLY: not model inference.",
        "evaluation": {
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": sha256_file(evidence_path),
        },
    }
    submission_path = directory / "submission.json"
    _write_json(submission_path, submission)
    manifest = {"data_release": {"status": "prototype_dummy_data"}}
    return submission_path, submission, manifest


def test_exact_development_fixture_is_registered_but_never_claim_eligible(
    tmp_path: Path,
) -> None:
    path, submission, manifest = _fixture(tmp_path)
    binding = registered_ahmedml_development_fixture(
        path,
        submission,
        manifest,
        root=tmp_path,
    )
    assert binding == {
        "record_type": "development_fixture",
        "submission_id": AHMEDML_DEVELOPMENT_FIXTURE_ID,
        "claim_eligibility": {
            "academic_citation": False,
            "promotion": False,
        },
    }


def test_development_fixture_registration_rejects_misleading_evidence(
    tmp_path: Path,
) -> None:
    path, submission, manifest = _fixture(tmp_path)
    evidence_path = path.parent / "evaluation-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["notes"] = "Ordinary model inference."
    _write_json(evidence_path, evidence)
    submission["evaluation"]["evidence_sha256"] = sha256_file(evidence_path)  # type: ignore[index]
    assert (
        registered_ahmedml_development_fixture(
            path,
            submission,
            manifest,
            root=tmp_path,
        )
        is None
    )


def test_development_fixture_registration_is_unavailable_in_official_release(
    tmp_path: Path,
) -> None:
    path, submission, manifest = _fixture(tmp_path)
    manifest["data_release"]["status"] = "official"  # type: ignore[index]
    assert (
        registered_ahmedml_development_fixture(
            path,
            submission,
            manifest,
            root=tmp_path,
        )
        is None
    )
