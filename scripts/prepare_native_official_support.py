#!/usr/bin/env python3
"""Prepare an official support draft from a validated, immutable native candidate.

The resulting files are local release drafts. Publishing the manifest URL and
recording the dataset-specific loader receipt remain separate release steps.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.scoring_support import sha256_file
from scripts.validate_scoring_supports import validate_candidate_manifest_release

RELEASES = {
    "ahmedml": "ahmedml-native-support-v1",
    "drivaerml": "drivaerml-native-support-v3",
    "windsorml": "windsorml-native-support-v1",
    "drivaernetplusplus": "drivaernetplusplus-native-pressure-support-v1",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def prepare(dataset: str, output_root: Path, approval: dict, timestamp: str) -> Path:
    specification_root = ROOT / "benchmark-specs" / dataset
    specification = read(specification_root / "submission-spec.json")
    binding = specification["scoring_support"]["candidate_manifest"]
    source_manifest = specification_root / binding["manifest_file"]
    source_root = source_manifest.parent
    manifest = read(source_manifest)
    if sha256_file(source_manifest) != binding["manifest_sha256"]:
        raise ValueError(f"{dataset}: candidate manifest hash differs")
    errors = validate_candidate_manifest_release(
        specification, specification_root, source_manifest, manifest
    )
    if errors:
        raise ValueError(f"{dataset}: candidate support is invalid: {'; '.join(errors)}")

    release_id = RELEASES[dataset]
    destination = output_root / release_id
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_root, destination)
    published_sets = []
    for descriptor in manifest["case_sets"]:
        descriptor = copy.deepcopy(descriptor)
        index_path = destination / descriptor["index_file"]
        index = read(index_path)
        for chunk in index["chunks"]:
            chunk_path = index_path.parent / chunk["file"]
            chunk_content = read(chunk_path)
            chunk_content["release_id"] = release_id
            chunk["sha256"] = write(chunk_path, chunk_content)
        index["release_id"] = release_id
        descriptor["index_sha256"] = write(index_path, index)
        published_sets.append(descriptor)

    manifest["release_id"] = release_id
    manifest["status"] = "official"
    manifest["published_at"] = timestamp
    manifest["case_sets"] = published_sets
    manifest["owner_approval"] = approval
    manifest["notes"] = (
        "Official support draft prepared from the validated candidate case set. "
        "The case artifacts and scientific metric definitions retain their "
        "candidate bytes; release identity and hash chain are renewed. "
        "Public intake requires publication of the manifest and a separate "
        "dataset-specific loader validation receipt."
    )
    write(destination / "manifest.json", manifest)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(RELEASES))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--pull-request-url", required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    approval = {
        "approved_by": args.approved_by,
        "approved_at": args.approved_at,
        "pull_request_url": args.pull_request_url,
    }
    path = prepare(args.dataset, args.output_root, approval, args.timestamp)
    print(f"{path / 'manifest.json'} {sha256_file(path / 'manifest.json')}")


if __name__ == "__main__":
    main()
