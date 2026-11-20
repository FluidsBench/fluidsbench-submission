#!/usr/bin/env python3
"""Reproduce the WindsorML v2 support metadata from v1 and the v2 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SPEC_ROOT = Path(__file__).resolve().parents[1] / "benchmark-specs/windsorml"
V1 = SPEC_ROOT / "scoring-support/windsorml-native-support-v1"
V1_SHA256 = "b6b89d956d1ef9ba03ed86c20955a01a8d107095a8c510edbda99cc838b18cf2"


def pinned(path: Path, expected: str) -> dict:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{path}: SHA-256 differs")
    return json.loads(payload)


def write(path: Path, value: dict) -> str:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build(output: Path, *, candidate: bool = False) -> Path:
    spec = json.loads((SPEC_ROOT / "submission-spec.json").read_text())
    support = spec["scoring_support"]
    binding = support["candidate_manifest"] if candidate else support
    case_sets = {
        split["case_set_id"]: pinned(SPEC_ROOT / split["index_file"], split["sha256"])["case_ids"]
        for split in spec["splits"]
    }
    manifest = pinned(V1 / "manifest.json", V1_SHA256)
    release_id = binding["release_id"]
    manifest.update(
        status=binding["status"],
        release_id=release_id,
        dataset_version=spec["dataset_version"],
        published_at=support["owner_approval"]["approved_at"],
        notes="WindsorML v2 case lists. Native fields, quadrature, profile samples, and scoring formulas are preserved for retained cases.",
    )
    if candidate:
        del manifest["owner_approval"]
    else:
        manifest["owner_approval"] = support["owner_approval"]
    for definition in manifest["supports"]:
        parameters = definition["location_definition"].get("parameters", {})
        if "profile_support_manifest_sha256" in parameters:
            parameters["profile_support_manifest_sha256"] = support["profile_support"]["manifest_sha256"]

    for descriptor in manifest["case_sets"]:
        expected = case_sets[descriptor["id"]]
        retained = set(expected)
        index_file = Path(descriptor["index_file"])
        index = pinned(V1 / index_file, descriptor["index_sha256"])
        actual = []
        for chunk_descriptor in index["chunks"]:
            chunk_file = index_file.parent / chunk_descriptor["file"]
            chunk = pinned(V1 / chunk_file, chunk_descriptor["sha256"])
            chunk["release_id"] = release_id
            chunk["cases"] = [case for case in chunk["cases"] if case["case_id"] in retained]
            ids = [case["case_id"] for case in chunk["cases"]]
            if not ids:
                raise ValueError(f"{chunk_file}: no retained cases")
            actual.extend(ids)
            digest = write(output / chunk_file, chunk)
            chunk_descriptor.update(case_count=len(ids), case_ids=ids, sha256=digest)
        if actual != expected:
            raise ValueError(f"{index_file}: cases differ from the v2 split or are reordered")
        index.update(release_id=release_id, case_count=len(actual))
        descriptor.update(case_count=len(actual), index_sha256=write(output / index_file, index))

    path = output / "manifest.json"
    digest = write(path, manifest)
    if digest != binding["manifest_sha256"]:
        raise ValueError("Regenerated support metadata differs from the published binding")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    print(build(args.output, candidate=args.candidate))
