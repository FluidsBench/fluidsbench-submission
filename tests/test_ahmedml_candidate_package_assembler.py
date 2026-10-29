from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import assemble_ahmedml_schema_v3_candidate as assembler


ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION = ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json"
TEMPLATE = ROOT / "examples" / "ahmedml-v3-candidate" / "package-config.template.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_template_is_deliberately_unresolved_but_release_bound() -> None:
    config = assembler.load_json(TEMPLATE, label="template")
    blockers = assembler.unresolved_tokens(config)
    assert blockers
    assert any(item["path"] == "$.participant.submission_id" for item in blockers)
    bindings = config["release_bindings"]
    specification = load(SPECIFICATION)
    candidate = specification["scoring_support"]["candidate_manifest"]
    assert bindings["candidate_manifest"] == {
        key: candidate[key]
        for key in ("status", "release_id", "manifest_url", "manifest_sha256")
    }
    assert bindings["profile_definition"] == {
        key: specification["profile_definition"][key]
        for key in ("id", "file", "sha256")
    }


def test_all_eight_official_splits_resolve_exactly() -> None:
    specification = load(SPECIFICATION)
    assert len(specification["splits"]) == 8
    for declaration in specification["splits"]:
        observed, split, case_ids, digest = assembler._find_split(
            specification, SPECIFICATION, declaration["id"]
        )
        assert observed == declaration
        assert split["case_ids"] == case_ids
        assert len(case_ids) == declaration["case_count"]
        assert digest == declaration["sha256"]


def test_checkpoint_arguments_and_bytes_are_exact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"actual-checkpoint-bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    methodology = {
        "checkpoints": [{"id": "joint-model", "sha256": digest}]
    }
    paths = assembler._checkpoint_arguments([f"joint-model={checkpoint}"])
    assembler._verify_checkpoints(methodology, paths)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(assembler.PackageAssemblyError, match="bytes differ"):
        assembler._verify_checkpoints(methodology, paths)


def test_genuine_inference_attestation_is_mandatory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": assembler.CONFIG_SCHEMA,
                "split_id": "full",
                "inference_attestation": {
                    "actual_model_inference": False,
                    "evaluation_truth_generated_predictions": True,
                    "scope": "fixture",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(assembler.PackageAssemblyError, match="genuine complete"):
        assembler.assemble_package(
            config_path=config_path,
            specification_path=SPECIFICATION,
            case_evidence_root=tmp_path,
            discretization_cases_path=tmp_path / "cases.jsonl",
            profile_truth_manifest_path=tmp_path / "truth.json",
            checkpoint_paths={},
            output_path=tmp_path / "output",
        )


def test_duplicate_checkpoint_arguments_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(assembler.PackageAssemblyError, match="unique"):
        assembler._checkpoint_arguments(
            [f"model={tmp_path / 'one'}", f"model={tmp_path / 'two'}"]
        )
