#!/usr/bin/env python3
"""Print the ordered union of candidate AhmedML test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ORDER = (
    "full",
    "medium",
    "scarce",
    "super_scarce",
    "geometry",
    "high_drag",
    "low_drag",
    "image_wake",
)


def ordered_case_ids() -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for split_id in SPLIT_ORDER:
        path = ROOT / "benchmark-specs" / "ahmedml" / "splits" / f"{split_id}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        for case_id in value["case_ids"]:
            if case_id not in seen:
                seen.add(case_id)
                result.append(case_id)
    if len(result) != 316:
        raise SystemExit(f"expected 316 unique AhmedML test cases, found {len(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        help="print only the case at this zero-based position",
    )
    args = parser.parse_args()
    case_ids = ordered_case_ids()
    if args.index is None:
        print("\n".join(case_ids))
    elif 0 <= args.index < len(case_ids):
        print(case_ids[args.index])
    else:
        parser.error(f"index must lie in [0, {len(case_ids) - 1}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
