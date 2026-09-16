from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_hiliftaeroml_split_indices import (
    CAMPAIGN_SCHEMA_ID,
    CASE_SET_SHA256_RULE,
    SPLIT_BINDINGS,
    SplitPreparationError,
    build_split_documents,
    canonical_json_bytes,
    ordered_case_digest,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-specs" / "hiliftaeroml"
SPECIFICATION = DATASET / "submission-spec.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_path(split_id: str) -> Path:
    return DATASET / "splits" / f"{split_id}.json"


def repository_campaign_fixture() -> dict:
    case_sets: dict[str, dict] = {}
    for binding in SPLIT_BINDINGS:
        document = load(split_path(binding.split_id))
        case_sets.setdefault(
            binding.case_set_id,
            {
                "case_set_id": binding.case_set_id,
                "case_count": binding.case_count,
                "case_set_sha256": binding.case_set_sha256,
                "ordered_case_ids": document["case_ids"],
            },
        )
    return {"schema_id": CAMPAIGN_SCHEMA_ID, "case_sets": list(case_sets.values())}


def test_all_fourteen_split_indices_are_exact_and_official() -> None:
    assert len(SPLIT_BINDINGS) == 14
    assert len({binding.case_set_id for binding in SPLIT_BINDINGS}) == 8
    for binding in SPLIT_BINDINGS:
        document = load(split_path(binding.split_id))
        case_ids = document["case_ids"]
        assert document == {
            "schema_version": "1.0",
            "dataset_id": "hiliftaeroml",
            "split_id": binding.split_id,
            "case_set_id": binding.case_set_id,
            "split_label": binding.split_label,
            "case_id_status": "official",
            "case_count": binding.case_count,
            "case_set_sha256": binding.case_set_sha256,
            "case_set_sha256_rule": CASE_SET_SHA256_RULE,
            "case_ids": case_ids,
        }
        assert len(case_ids) == len(set(case_ids)) == binding.case_count
        assert ordered_case_digest(case_ids) == binding.case_set_sha256


def test_training_size_variants_share_exact_evaluation_order() -> None:
    standard = [
        load(split_path(split_id))["case_ids"]
        for split_id in ("full", "medium", "scarce", "super_scarce")
    ]
    geometry = [
        load(split_path(split_id))["case_ids"]
        for split_id in (
            "geometry",
            "geometry_medium",
            "geometry_scarce",
            "geometry_super_scarce",
        )
    ]
    assert all(case_ids == standard[0] for case_ids in standard[1:])
    assert all(case_ids == geometry[0] for case_ids in geometry[1:])
    assert standard[0] != geometry[0]


def test_submission_specification_binds_every_exact_split_file() -> None:
    specification = load(SPECIFICATION)
    descriptors = specification["splits"]
    assert [descriptor["id"] for descriptor in descriptors] == [
        binding.split_id for binding in SPLIT_BINDINGS
    ]
    for binding, descriptor in zip(SPLIT_BINDINGS, descriptors):
        path = split_path(binding.split_id)
        assert descriptor == {
            "id": binding.split_id,
            "label": binding.split_label,
            "index_file": f"splits/{binding.split_id}.json",
            "case_count": binding.case_count,
            "case_set_id": binding.case_set_id,
            "case_id_status": "official",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_generator_reproduces_repository_bytes_deterministically() -> None:
    documents = build_split_documents(repository_campaign_fixture())
    assert set(documents) == {binding.split_id for binding in SPLIT_BINDINGS}
    for binding in SPLIT_BINDINGS:
        assert canonical_json_bytes(documents[binding.split_id]) == split_path(
            binding.split_id
        ).read_bytes()


def test_generator_fails_closed_on_case_order_drift() -> None:
    campaign = repository_campaign_fixture()
    mutated = copy.deepcopy(campaign)
    case_ids = mutated["case_sets"][0]["ordered_case_ids"]
    case_ids[0], case_ids[1] = case_ids[1], case_ids[0]
    with pytest.raises(SplitPreparationError, match="digest differs"):
        build_split_documents(mutated)
