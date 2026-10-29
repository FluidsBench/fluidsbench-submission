#!/usr/bin/env python3
"""Derive FluidsBench WindsorML split bindings from the official manifest.

The dataset ships its own authoritative split manifest at
``splits/manifest.json`` in the public release. That manifest enumerates all 355
design variants, but ``run_350``..``run_354`` have no per-run payload, so two
test identifiers (``run_352`` in ``high_drag``, ``run_354`` in the shared
full/medium/scarce/super_scarce test set and in ``low_drag``) cannot be scored.

This writes the scored case sets as the intersection of the official manifest
with the published runs, and records the official counts and the excluded IDs in
each file so the reduction stays visible and auditable. The official manifest is
never modified or regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_ROOT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/data"
)
DATASET_ID = "windsorml"
DATASET_VERSION = "windsorml-native-v1-candidate"
SCHEMA_VERSION = "1.1"
PUBLISHED_RUN_IDS = range(350)

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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_ROOT / "splits" / "manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "benchmark-specs"
        / DATASET_ID
        / "splits",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    published = {f"run_{run_id}" for run_id in PUBLISHED_RUN_IDS}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The reduced-data families are defined to reuse the baseline test set;
    # assert that rather than trusting it.
    shared = {
        family: tuple(manifest[f"{family}_test"]) for family in SHARED_TEST_FAMILIES
    }
    if len(set(shared.values())) != 1:
        raise SystemExit(
            "full/medium/scarce/super_scarce no longer share one test set: "
            + ", ".join(f"{k}={len(v)}" for k, v in shared.items())
        )

    summary = []
    for family, label in FAMILIES:
        official_test = list(manifest[f"{family}_test"])
        scored = sorted((c for c in official_test if c in published), key=case_sort_key)
        excluded = sorted(
            (c for c in official_test if c not in published), key=case_sort_key
        )
        official_train = list(manifest[f"{family}_train"])
        official_val = list(manifest[f"{family}_val"])
        scored_train = sorted(
            (c for c in official_train if c in published), key=case_sort_key
        )
        scored_val = sorted(
            (c for c in official_val if c in published), key=case_sort_key
        )

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
                "description": (
                    "Official WindsorML split manifest shipped in the public release."
                ),
                "path": "splits/manifest.json",
                "repository_id": "neashton/windsorml",
            },
            "official_train_count": len(official_train),
            "official_validation_count": len(official_val),
            "official_case_count": len(official_test),
            "train_count": len(scored_train),
            "validation_count": len(scored_val),
            "case_count": len(scored),
            "excluded_case_ids": excluded,
            "excluded_reason": (
                "Identifier is assigned by the official manifest but has no per-run "
                "payload in the public release, so it cannot be scored."
            )
            if excluded
            else None,
            "case_ids": scored,
        }
        path = args.out_dir / f"{family}.json"
        payload = json.dumps(document, indent=2) + "\n"
        path.write_text(payload)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        summary.append(
            {
                "id": family,
                "label": label,
                "index_file": f"splits/{family}.json",
                "case_count": len(scored),
                "case_set_id": case_set_id,
                "case_id_status": "official_intersected_with_published",
                "sha256": digest,
            }
        )
        flag = f"  excluded {excluded}" if excluded else ""
        print(
            f"  {family:14s} train {len(official_train):3d}->{len(scored_train):3d}  "
            f"val {len(official_val):3d}->{len(scored_val):3d}  "
            f"test {len(official_test):3d}->{len(scored):3d}{flag}"
        )

    index = args.out_dir / "splits-index.json"
    index.write_text(json.dumps(summary, indent=2) + "\n")
    scored_union = set()
    for entry in summary:
        scored_union |= set(json.loads((args.out_dir / f"{entry['id']}.json").read_text())["case_ids"])
    print(f"\nunique scored test cases: {len(scored_union)}")
    print(f"wrote {len(summary)} split files + {index.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
