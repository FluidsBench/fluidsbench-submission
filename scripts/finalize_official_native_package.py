#!/usr/bin/env python3
"""Bind a native candidate evaluation package to its approved official support.

The candidate and official manifests must have identical support definitions
and case/artifact content. This changes only release identities and hashes;
it does not change predictions or recompute metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.scoring_support import load_support_release, sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def official_drivaerml_profiles(directory: Path, submission: dict) -> str:
    """Retain scored fixed and zero-weight relative profiles under official format."""
    index_path = directory / submission["profile_data"]["index_file"]
    index = read(index_path)
    if index.get("format") != "fluidsbench-drivaerml-relative-profile-chunks-v3-candidate":
        raise ValueError("DrivAerML package has an unexpected profile format")
    official_format = "fluidsbench-drivaerml-relative-profile-chunks-v3-official"
    index["format"] = official_format
    submission["profile_data"]["format"] = official_format
    return write(index_path, index)


def finalize(directory: Path, specification_path: Path) -> None:
    specification = read(specification_path)
    support = specification["scoring_support"]
    if (
        specification.get("status") != "official"
        or support.get("status") != "official"
        or support.get("submissions_open") is not True
    ):
        raise ValueError("official package binding requires an open official contract")
    candidate = support["candidate_manifest"]
    candidate_path = specification_path.parent / candidate["manifest_file"]
    official_path = specification_path.parent / support["manifest_file"]
    if (
        sha256_file(candidate_path) != candidate["manifest_sha256"]
        or sha256_file(official_path) != support["manifest_sha256"]
    ):
        raise ValueError("candidate or official support manifest hash changed")
    candidate_manifest = read(candidate_path)
    official_manifest = read(official_path)
    if candidate_manifest["supports"] != official_manifest["supports"]:
        raise ValueError("official support definitions differ from the evaluated candidate")
    for case_set in candidate_manifest["case_sets"]:
        source = load_support_release(candidate_path, case_set["id"])
        target = load_support_release(official_path, case_set["id"])
        if list(source.cases) != list(target.cases):
            raise ValueError("official support case coverage differs from candidate")
        for case_id in source.cases:
            if source.cases[case_id]["support_instances"] != target.cases[case_id]["support_instances"]:
                raise ValueError(f"official support artifacts differ for {case_id}")

    directory = directory.resolve()
    submission_path = directory / "submission.json"
    submission = read(submission_path)
    if submission.get("dataset_id") != specification["dataset_id"]:
        raise ValueError("package dataset differs from official contract")
    if submission.get("scoring_support") != {
        "status": "candidate",
        "release_id": candidate["release_id"],
        "manifest_url": candidate["manifest_url"],
        "manifest_sha256": candidate["manifest_sha256"],
    }:
        raise ValueError("package is not bound to the approved candidate support")
    for name in ("discretization.json", "metrics/cases.json"):
        path = directory / name
        value = read(path)
        if (
            value.get("scoring_support_release_id") != candidate["release_id"]
            or value.get("scoring_support_manifest_sha256") != candidate["manifest_sha256"]
        ):
            raise ValueError(f"{name} candidate support identity differs")
        value["scoring_support_release_id"] = support["release_id"]
        value["scoring_support_manifest_sha256"] = support["manifest_sha256"]
        digest = write(path, value)
        if name == "discretization.json":
            submission["spatial_discretization"]["sha256"] = digest
        else:
            submission["case_metrics"]["sha256"] = digest

    evidence_path = directory / "evaluation-evidence.json"
    evidence = read(evidence_path)
    if (
        evidence.get("scoring_support_release_id") != candidate["release_id"]
        or evidence.get("scoring_support_manifest_sha256") != candidate["manifest_sha256"]
    ):
        raise ValueError("evaluation evidence candidate identity differs")
    evidence["scoring_support_release_id"] = support["release_id"]
    evidence["scoring_support_manifest_sha256"] = support["manifest_sha256"]
    evidence["discretization_sha256"] = submission["spatial_discretization"]["sha256"]
    evidence["case_metrics_sha256"] = submission["case_metrics"]["sha256"]
    if specification["dataset_id"] == "drivaerml":
        evidence["profile_index_sha256"] = official_drivaerml_profiles(
            directory, submission
        )
    submission["evaluation"]["evidence_sha256"] = write(evidence_path, evidence)
    submission["scoring_support"] = {
        "status": "official",
        "release_id": support["release_id"],
        "manifest_url": support["manifest_url"],
        "manifest_sha256": support["manifest_sha256"],
    }
    write(submission_path, submission)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: finalize_official_native_package.py PACKAGE SPECIFICATION")
    finalize(Path(sys.argv[1]), Path(sys.argv[2]))
