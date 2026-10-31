"""Official publication must preserve evaluated package identities and native CFD."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import manage_leaderboard as manager
from scripts.profile_truth_publication import publication_errors

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    manifest = json.loads((ROOT / "leaderboard/manifest.json").read_text())
    binding = manifest["data_release"]["profile_ground_truth_publication"]
    publication = json.loads((ROOT / "benchmark-specs" / f"{binding['release_id']}.json").read_text())
    return manifest, publication


def test_official_release_preserves_evaluated_reference_fingerprints():
    manifest, publication = inputs()
    release = manifest["data_release"]
    release.update(id="local-official-publication-test", status="official", source_commit="a" * 40,
                   archive_url="https://example.test/archive/local-official-publication-test",
                   asset_base_url="https://example.test/local-official-publication-test/",
                   release_view_url="https://fluidsbench.org/releases/local-official-publication-test/")
    assert manager.release_contract_errors(manifest) == []
    assert release["profile_ground_truth"]["manifest_sha256"] == "3d3b0d7c8546925f6bace2113969eab62b0258edb5e4fdba99c0e65e29b02fdc"
    airfrans = next(d for d in publication["datasets"] if d["id"] == "airfrans")
    contract = json.loads((ROOT / "benchmark-specs/airfrans/submission-spec.json").read_text())
    assert airfrans["submission_profile_ground_truth"]["manifest_sha256"] == contract["profile_definition"]["profile_ground_truth"]["manifest_sha256"]
    assert {d["id"] for d in publication["datasets"]} == {"airfrans", "ahmedml", "windsorml", "drivaerml", "drivaernetplusplus"}


@pytest.mark.parametrize("change", ["prototype", "mutable_source", "missing_dataset", "closed_dataset", "wrong_support", "wrong_truth"])
def test_publication_rejects_semantic_changes_even_with_a_new_checksum(change):
    manifest, publication = inputs()
    if change == "prototype":
        publication["data_release"]["status"] = "prototype_dummy_data"
    elif change == "mutable_source":
        publication["data_release"]["source_ref"] = "main"
    elif change == "missing_dataset":
        publication["datasets"].pop()
    elif change == "closed_dataset":
        publication["datasets"][0]["id"] = "hiliftaeroml"
    elif change == "wrong_support":
        publication["datasets"][0]["scoring_support"]["manifest_sha256"] = "a" * 64
    else:
        publication["datasets"][0]["submission_profile_ground_truth"]["manifest_sha256"] = "a" * 64
    payload = json.dumps(publication).encode()
    binding = copy.deepcopy(manifest["data_release"]["profile_ground_truth_publication"])
    binding["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    assert publication_errors(payload, binding, manifest)
