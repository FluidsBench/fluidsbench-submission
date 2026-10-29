#!/usr/bin/env python3
"""Verify a downloaded AirfRANS handover and register one unranked dev reference.

Run only after extracting and checking the contributor's SHA256SUMS. The
profile-truth release is built with export_airfrans_profile_truth.py. Native
support tables and profile truth remain external; their immutable metadata and
this verification receipt are retained in Git. No dataset activation occurs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.airfrans.pre_release import ACTIVATION, REGISTRY_PATH
from reference.airfrans.profile_truth import validate_release
from reference.airfrans_profiles import score_airfrans_velocity_profiles
from reference.ahmedml.pre_release import package_tree_binding, sha256_file
from reference.scoring_support import load_support_release, load_scoring_support


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def register(submission_id, handover, truth_release):
    package = ROOT / "submissions/airfrans" / submission_id
    submission_path = package / "submission.json"
    submission = read(submission_path)
    if submission["dataset_id"] != "airfrans" or submission.get("approval") is not None:
        raise ValueError("Only unapproved AirfRANS candidates may be registered")
    split = submission["split_id"]
    case_set = submission["case_set_id"]
    support_id = submission["scoring_support"]["release_id"]
    support_root = handover / "extracted" / support_id
    if sha256_file(support_root / "manifest.json") != submission["scoring_support"]["manifest_sha256"]:
        raise ValueError("Scoring-support manifest does not match the submitted results")
    support = load_support_release(support_root / "manifest.json", case_set)
    expected_cases = read(ROOT / f"benchmark-specs/airfrans/splits/{split}.json")["case_ids"]
    if list(support.cases) != expected_cases:
        raise ValueError("Scoring-support coverage differs from the official split")
    support_count = 0
    for case in support.cases:
        for name in support.supports:
            load_scoring_support(support, case, name)
            support_count += 1
    definition_path = ROOT / "benchmark-specs/airfrans/velocity-profiles-v1.json"
    truth_summary = validate_release(release_root=truth_release,
                                    profile_definition_path=definition_path,
                                    repository_root=ROOT)
    truth_manifest = read(truth_release / "manifest.json")
    truth_index = read(truth_release / "index.json")
    truth = [read(truth_release / truth_index["case_locations"][case]["truth_artifact"]["file"])
             for case in sorted(expected_cases)]
    definition = read(definition_path)
    index = read(package / "profiles/index.json")
    predictions = [read(package / "profiles" / entry["file"]) for entry in index["chunks"]]
    score = score_airfrans_velocity_profiles(
        truth, predictions,
        station_ids=[item["id"] for item in definition["stations"]],
        quantity_ids=[item["id"] for item in definition["quantities"]],
        panel_id=definition["metric_binding"]["panel_id"],
        expected_case_ids=expected_cases, allow_partial_case_coverage=False)
    score["split_id"] = split
    score_path = handover / "downloads/profile-scores" / f"{submission_id}-profile-score.json"
    if (json.dumps(score, indent=2) + "\n").encode() != score_path.read_bytes():
        raise ValueError("Recomputed profile scores differ from the handover")
    # All verification above is read-only. Retain only the validated metadata.
    snapshot = ROOT / "benchmark-specs/airfrans/pre-release" / submission_id
    for relative in ("manifest.json", "build-summary.json", f"case-sets/{case_set}/index.json",
                     f"case-sets/{case_set}/chunk-000.json"):
        dest = snapshot / "scoring-support" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(support_root / relative, dest)
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(truth_release / "manifest.json", snapshot / "profile-truth-manifest.json")
    shutil.copyfile(score_path, snapshot / "profile-score.json")
    binding = {"status": "candidate", "release_id": truth_manifest["release_id"],
               "manifest_sha256": truth_summary["manifest_sha256"]}
    evidence_path = package / submission["evaluation"]["evidence_file"]
    evidence = read(evidence_path)
    for key in ("release_id", "manifest_sha256"):
        submission["profile_data"]["profile_ground_truth_" + key] = binding[key]
        evidence["profile_ground_truth_" + key] = binding[key]
    evidence["notes"] = (
        "Closed AirfRANS pre-release reference. Native scoring-support files and profile truth "
        "were verified locally; profile scores reproduced byte-for-byte. The ground-truth "
        "binding identifies the candidate data release, not the extraction definition. "
        "No source-data re-extraction or full model-inference replay was performed by this "
        "registration. Submitted metrics and prediction values are unchanged. "
        "Registration does not grant owner approval or open submissions.")
    write(evidence_path, evidence)
    submission["evaluation"]["evidence_sha256"] = sha256_file(evidence_path)
    write(submission_path, submission)
    tree = package_tree_binding(package)
    receipt = {
        "schema": "airfrans-pre-release-validation-v1", "activation": ACTIVATION,
        "package_tree": tree, "case_count": len(expected_cases),
        "native_supports_loaded": support_count, "profile_scores_reproduced": True,
        "profile_score_sha256": sha256_file(score_path), "release_graph_valid": True,
        "scoring_support_manifest_sha256": sha256_file(support_root / "manifest.json"),
        "profile_ground_truth": binding, "source_replay_complete": False,
        "model_inference_replayed": False,
        "handover_url": "https://drive.google.com/drive/folders/14Tgs28bccp5HP-OXE8LVaIC-9RtFjqWC",
        "handover_checksums_sha256": sha256_file(handover / "downloads/SHA256SUMS"),
    }
    receipt_path = snapshot / "validation.json"
    write(receipt_path, receipt)
    candidate_manifest = dict(submission["scoring_support"])
    candidate_manifest["manifest_file"] = (snapshot / "scoring-support/manifest.json").relative_to(ROOT / "benchmark-specs/airfrans").as_posix()
    entry = {
        "submission_path": submission_path.relative_to(ROOT).as_posix(),
        "submission_id": submission_id, "split_id": split, "case_set_id": case_set,
        "record_type": "pre_release_reference", "package_tree": tree,
        "candidate_manifest": candidate_manifest, "profile_ground_truth": binding,
        "validation_receipt": receipt_path.relative_to(ROOT).as_posix(),
        "profile_truth_manifest_file": (snapshot / "profile-truth-manifest.json").relative_to(ROOT).as_posix(),
        "artifacts": [{"file": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)}
                      for p in sorted(snapshot.rglob("*")) if p.is_file()],
    }
    registry_path = ROOT / REGISTRY_PATH
    registry = read(registry_path) if registry_path.exists() else {
        "schema": "airfrans-pre-release-reference-registry-v1",
        "status": "closed_candidate_dev_only", "dataset_id": "airfrans",
        "activation": ACTIVATION, "entries": []}
    registry["entries"] = sorted([e for e in registry["entries"] if e["submission_id"] != submission_id] + [entry],
                                 key=lambda e: e["submission_id"])
    write(registry_path, registry)
    print(f"Registered {submission_id}: {len(expected_cases)} cases; unranked dev reference only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", required=True, choices=("transolverpp-full-v1", "transolverpp-aoa-v1"))
    parser.add_argument("--handover-root", required=True, type=Path)
    parser.add_argument("--profile-truth-release", required=True, type=Path)
    args = parser.parse_args()
    register(args.submission_id, args.handover_root, args.profile_truth_release)
