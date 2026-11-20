#!/usr/bin/env python3
"""Bind the pinned WindsorML v2 partitions to the FluidsBench evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_ID = "windsorml"
DATASET_VERSION = "windsorml-native-v2"
SCHEMA_VERSION = "1.1"
SOURCE_REVISION = "ba7e66955650b2ff0482517f256d84c1eda815e7"
SPEC_ROOT = Path(__file__).resolve().parents[1] / "benchmark-specs" / DATASET_ID

FAMILIES: tuple[tuple[str, str], ...] = (
    ("full", "Full"),
    ("medium", "Medium"),
    ("scarce", "Scarce"),
    ("super_scarce", "Super scarce"),
    ("geometry", "Geometry OOD"),
    ("high_drag", "High drag"),
    ("low_drag", "Low drag"),
    ("image_wake", "Image wake"),
)

# Families whose test population is the shared baseline test set.
SHARED_TEST_FAMILIES = frozenset({"full", "medium", "scarce", "super_scarce"})


def case_sort_key(case_id: str) -> int:
    return int(case_id.removeprefix("run_"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPEC_ROOT / "splits/v2/manifest.json")
    parser.add_argument("--release-record", type=Path, default=SPEC_ROOT / "splits/v2/release.json")
    parser.add_argument("--out-dir", type=Path, default=SPEC_ROOT / "splits")
    args = parser.parse_args()

    raw = args.manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    release = json.loads(args.release_record.read_text())
    if release.get("version") != 2 or digest != release.get("manifest_sha256"):
        raise SystemExit("Input does not match the WindsorML v2 manifest checksum.")
    manifest = json.loads(raw)
    eligible = set(release["eligible_run_ids"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for part in ("val", "test"):
        if len({tuple(manifest[f"{family}_{part}"]) for family in SHARED_TEST_FAMILIES}) != 1:
            raise SystemExit(f"Baseline splits must share the same {part} cases.")

    summary = []
    for family, label in FAMILIES:
        assignment = {
            part: sorted(manifest[f"{family}_{part}"], key=case_sort_key)
            for part in ("train", "val", "test")
        }
        all_ids = [case for part in assignment.values() for case in part]
        if len(all_ids) != len(set(all_ids)) or not set(all_ids) <= eligible:
            raise SystemExit(f"{family}: overlapping partitions or ineligible cases.")
        counts = {part: len(ids) for part, ids in assignment.items()}
        if counts != release["split_counts"][family]:
            raise SystemExit(f"{family}: counts differ from the published release record.")
        case_set_id = "baseline-test" if family in SHARED_TEST_FAMILIES else f"{family}-test"
        document = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "split_id": family,
            "case_set_id": case_set_id,
            "split_label": label,
            "case_id_status": "official_intersected_with_published",
            "source": {
                "description": "Canonical WindsorML v2 benchmark split assignments.",
                "path": "splits/v2/manifest.json",
                "repository_id": "neashton/windsorml",
                "revision": SOURCE_REVISION,
                "sha256": digest,
            },
            "official_train_count": counts["train"],
            "official_validation_count": counts["val"],
            "official_case_count": counts["test"],
            "train_count": counts["train"],
            "validation_count": counts["val"],
            "case_count": counts["test"],
            "excluded_case_ids": [],
            "excluded_reason": None,
            "train_case_ids": assignment["train"],
            "validation_case_ids": assignment["val"],
            "case_ids": assignment["test"],
        }
        payload = (json.dumps(document, indent=2) + "\n").encode()
        (args.out_dir / f"{family}.json").write_bytes(payload)
        summary.append({
            "id": family,
            "label": label,
            "index_file": f"splits/{family}.json",
            "case_count": counts["test"],
            "case_set_id": case_set_id,
            "case_id_status": document["case_id_status"],
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        print(f"{family}: {counts['train']} train / {counts['val']} validation / {counts['test']} test")
    (args.out_dir / "splits-index.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
