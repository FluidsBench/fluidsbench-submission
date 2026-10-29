#!/usr/bin/env python3
"""Calculate the balanced FluidsBench AirfRANS velocity-profile score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.airfrans_profiles import score_airfrans_velocity_profiles  # noqa: E402


DEFAULT_PROFILE_DEFINITION = ROOT / "benchmark-specs" / "airfrans" / "velocity-profiles-v1.json"
SPLITS_ROOT = ROOT / "benchmark-specs" / "airfrans" / "splits"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        nargs="+",
        required=True,
        help="One or more ground-truth profile JSON files.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prediction profile JSON files.",
    )
    parser.add_argument(
        "--profile-definition",
        type=Path,
        default=DEFAULT_PROFILE_DEFINITION,
        help="Pinned AirfRANS profile definition (default: repository contract).",
    )
    parser.add_argument(
        "--split-id",
        choices=("full", "scarce", "reynolds_extrapolation", "aoa_extrapolation"),
        required=True,
        help="Official AirfRANS evaluation split whose case coverage must be scored.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a strict subset for calibration only; never use for a leaderboard result.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    definition = load_json(arguments.profile_definition)
    split = load_json(SPLITS_ROOT / f"{arguments.split_id}.json")
    result = score_airfrans_velocity_profiles(
        [load_json(path) for path in arguments.ground_truth],
        [load_json(path) for path in arguments.predictions],
        station_ids=[station["id"] for station in definition["stations"]],
        quantity_ids=[quantity["id"] for quantity in definition["quantities"]],
        panel_id=definition["metric_binding"]["panel_id"],
        expected_case_ids=split["case_ids"],
        allow_partial_case_coverage=arguments.allow_partial,
    )
    result["split_id"] = arguments.split_id
    payload = json.dumps(result, indent=2) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
