"""Exact maintainer bindings for closed AirfRANS development references.

Registration permits candidate contract validation and an explicitly unranked
dev row. It does not activate the dataset or grant scientific approval.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from reference.ahmedml.pre_release import package_tree_binding, sha256_file

REGISTRY_PATH = "benchmark-specs/airfrans/pre-release-reference-registry.json"
ACTIVATION = {
    "owner_approval_complete": False,
    "published": False,
    "submissions_opened": False,
    "registration_changes_activation": False,
}


def historical_contract(*, root=None):
    """Load the hash-bound closed contract used by retained dev references."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    registry = json.loads((root / REGISTRY_PATH).read_text())
    binding = registry["contract_snapshot"]
    path = root / binding["file"]
    if path.is_symlink() or not path.resolve().is_relative_to(root / "benchmark-specs/airfrans"):
        raise ValueError("historical AirfRANS contract escapes its dataset")
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("historical AirfRANS contract digest differs")
    spec = json.loads(path.read_text())
    if (spec["scoring_support"]["status"] != "owner_review_required"
            or spec["scoring_support"]["submissions_open"] is not False
            or "owner_approval" in spec["scoring_support"]):
        raise ValueError("historical AirfRANS contract must remain closed")
    return spec


def registered_prototype_contract(path, submission, manifest, *, root=None):
    """Use the old contract only for exact retained prototype package bytes."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    try:
        relative = path.resolve().relative_to(root).as_posix()
        registry = json.loads((root / REGISTRY_PATH).read_text())
        entry = next(e for e in registry["prototype_packages"] if e["submission_path"] == relative)
        if (manifest.get("data_release", {}).get("status") != "prototype_dummy_data"
                or submission.get("schema_version") != "1.0"
                or submission.get("approval", {}).get("status") != "prototype"
                or json.loads(path.read_text()) != submission
                or package_tree_binding(path.parent) != entry["package_tree"]):
            return None
        return historical_contract(root=root)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def registered_airfrans_pre_release_reference(path, submission, manifest, *, root=None):
    """Recognize only a complete, byte-bound package in the prototype feed."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    try:
        relative = path.resolve().relative_to(root).as_posix()
        registry = json.loads((root / REGISTRY_PATH).read_text())
        spec = historical_contract(root=root)
        if (
            registry["schema"] != "airfrans-pre-release-reference-registry-v1"
            or registry["status"] != "closed_candidate_dev_only"
            or registry["activation"] != ACTIVATION
            or registry["dataset_id"] != "airfrans"
            or manifest.get("data_release", {}).get("status") != "prototype_dummy_data"
            or spec["status"] not in {"prototype_dummy_data", "owner_review_required", "candidate"}
            or spec["scoring_support"]["status"] != "owner_review_required"
            or spec["scoring_support"]["submissions_open"] is not False
            or "owner_approval" in spec["scoring_support"]
        ):
            return None
        entries = registry["entries"]
        if len({entry["submission_path"] for entry in entries}) != len(entries):
            return None
        entry = next((item for item in entries if item["submission_path"] == relative), None)
        if entry is None:
            return None
        if (
            relative != f"submissions/airfrans/{submission['submission_id']}/submission.json"
            or submission["dataset_id"] != "airfrans"
            or submission["schema_version"] != "3.0"
            or submission.get("approval") is not None
            or any((path.parent / name).exists() for name in (
                "maintainer-validation.json", "maintainer-replay.json", "prediction-artifact-checks.json"
            ))
            or submission["scoring_support"]["status"] != "candidate"
            or entry["record_type"] != "pre_release_reference"
            or entry["package_tree"] != package_tree_binding(path.parent)
            or json.loads(path.read_text()) != submission
        ):
            return None
        for key in ("submission_id", "split_id", "case_set_id"):
            if submission[key] != entry[key]:
                return None
        # The receipt and all retained release metadata are independently bound.
        for artifact in entry["artifacts"]:
            file = root / artifact["file"]
            if file.is_symlink() or not file.resolve().is_relative_to(root / "benchmark-specs/airfrans"):
                return None
            if sha256_file(file) != artifact["sha256"]:
                return None
        retained_files = {item["file"] for item in entry["artifacts"]}
        if not {entry["validation_receipt"], entry["profile_truth_manifest_file"]}.issubset(retained_files):
            return None
        truth_manifest_path = root / entry["profile_truth_manifest_file"]
        truth_manifest = json.loads(truth_manifest_path.read_text())
        if (
            sha256_file(truth_manifest_path) != entry["profile_ground_truth"]["manifest_sha256"]
            or truth_manifest["release_id"] != entry["profile_ground_truth"]["release_id"]
            or entry["profile_ground_truth"]["status"] != "candidate"
        ):
            return None
        receipt = json.loads((root / entry["validation_receipt"]).read_text())
        if (
            receipt["schema"] != "airfrans-pre-release-validation-v1"
            or receipt["activation"] != ACTIVATION
            or receipt["package_tree"] != entry["package_tree"]
            or receipt["profile_scores_reproduced"] is not True
            or receipt["release_graph_valid"] is not True
            or receipt["native_supports_loaded"] != 3 * submission["profile_data"]["case_count"]
            or receipt["case_count"] != submission["profile_data"]["case_count"]
            or receipt["scoring_support_manifest_sha256"] != submission["scoring_support"]["manifest_sha256"]
            or receipt["profile_ground_truth"] != entry["profile_ground_truth"]
        ):
            return None
        support = entry["candidate_manifest"]
        for key in ("status", "release_id", "manifest_url", "manifest_sha256"):
            if support[key] != submission["scoring_support"][key]:
                return None
        for key in ("release_id", "manifest_sha256"):
            if entry["profile_ground_truth"][key] != submission["profile_data"]["profile_ground_truth_" + key]:
                return None
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None
    return deepcopy(entry)


def candidate_validation_view(spec, manifest, binding, *, root=None):
    """Build a per-package candidate view; never mutate published contracts."""
    spec, manifest = historical_contract(root=root), deepcopy(manifest)
    spec["status"] = "owner_review_required"
    spec["scoring_support"]["candidate_manifest"] = deepcopy(binding["candidate_manifest"])
    manifest["data_release"]["profile_ground_truth"] = deepcopy(binding["profile_ground_truth"])
    return spec, manifest
