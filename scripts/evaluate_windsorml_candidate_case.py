#!/usr/bin/env python3
"""Evaluate complete WindsorML surface/volume predictions and frozen profiles."""

from __future__ import annotations
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
from reference.windsorml.contract import SOURCE_IDENTITY_SHA256, load_source_identity
from reference.windsorml.evaluator import evaluate_candidate_case
from reference.windsorml.profiles import load_profile_support
from scripts.score_windsorml_candidate_dataset import write_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--surface-manifest", required=True, type=Path)
    parser.add_argument("--volume-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        identity = load_source_identity(
            ROOT
            / "benchmark-specs/windsorml/public-source-identity/windsorml-public-source-identity-v1.json",
            expected_sha256=SOURCE_IDENTITY_SHA256,
        )
        support = load_profile_support(case_id=args.case_id)
        result = evaluate_candidate_case(
            case=identity.case(args.case_id),
            dataset_root=args.dataset_root,
            surface_manifest=args.surface_manifest,
            volume_manifest=args.volume_manifest,
            profile_support=support.document,
        )
        write_atomic(args.output.resolve(), result.to_json())
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
