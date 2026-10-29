#!/usr/bin/env python3
"""Download and verify the frozen AhmedML development evaluator cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.ahmedml.distribution import (  # noqa: E402
    SPLITS,
    cache_root,
    ensure_case_support,
    load_distribution,
    select_cases,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--split", choices=SPLITS)
    selection.add_argument("--case-id", action="append", dest="case_ids")
    selection.add_argument(
        "--all", action="store_true", help="install the union of all test splits"
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show maximum download and installed sizes without writing or downloading",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify an already installed selection without network access",
    )
    args = parser.parse_args()
    try:
        cases = select_cases(split=args.split, case_ids=args.case_ids)
        index = load_distribution()
        root = cache_root(args.destination)
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "destination": str(root),
                    "maximum_download_bytes": sum(
                        index["cases"][c]["size_bytes"] for c in cases
                    ),
                    "installed_bytes": sum(
                        index["cases"][c]["expanded_bytes"] for c in cases
                    ),
                }
            ),
            flush=True,
        )
        if not args.dry_run:
            for number, case_id in enumerate(cases, 1):
                print(
                    f"[{number}/{len(cases)}] Verifying/installing {case_id}",
                    flush=True,
                )
                ensure_case_support(
                    case_id, destination=root, offline=args.offline, index=index
                )
            print(f"Verified {len(cases)} cases in {root}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"AhmedML support installation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
