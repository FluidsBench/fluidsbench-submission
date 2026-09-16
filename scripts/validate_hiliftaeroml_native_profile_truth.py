#!/usr/bin/env python3
"""Validate a complete inactive HiLiftAeroML native profile truth release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.native_profile_truth import (  # noqa: E402
    NativeProfileTruthError,
    load_case_universe,
    load_source_index,
    validate_release,
)
from reference.hiliftaeroml.native_profile_truth_materializer import (  # noqa: E402
    load_prerequisite_authority_index,
    materialize_case_truth,
)

DEFAULT_SUPPORT_MANIFEST = (
    ROOT
    / "benchmark-specs"
    / "hiliftaeroml"
    / "scoring-support"
    / "hiliftaeroml-native-all-splits-support-v1-candidate"
    / "manifest.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--authority-index",
        type=Path,
        help="Prediction-free production prerequisite authority index.",
    )
    source.add_argument(
        "--oracle-source-index",
        type=Path,
        help="Legacy evaluator-output source index, valid only for oracle fixtures.",
    )
    parser.add_argument(
        "--support-manifest", type=Path, default=DEFAULT_SUPPORT_MANIFEST
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip expensive replay against evaluator-native sources.",
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        universe = load_case_universe(args.support_manifest)
        replay_case_builder = None
        if args.authority_index is not None:
            source_body, source_sha = load_prerequisite_authority_index(
                args.authority_index, universe=universe, require_complete=True
            )

            def replay_case_builder(case_id: str, output_root: Path) -> dict:
                record, _ = materialize_case_truth(
                    case_id=case_id,
                    authority=source_body,
                    output_root=output_root,
                )
                return record

        else:
            source_body, source_sha = load_source_index(
                args.oracle_source_index, universe=universe, require_complete=True
            )
        result = validate_release(
            universe=universe,
            source=source_body,
            source_index_sha256=source_sha,
            output_root=args.output_dir,
            replay_sources=not args.metadata_only,
            replay_case_builder=replay_case_builder,
        )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.receipt is not None:
            if args.receipt.exists() or args.receipt.is_symlink():
                raise NativeProfileTruthError(
                    "refusing to overwrite validation receipt"
                )
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (NativeProfileTruthError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
