#!/usr/bin/env python3
"""Check a complete DrivAerML 1/2/5/10 mm profile convergence study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.profile_convergence import (  # noqa: E402
    DEFAULT_CONTRACT_PROPOSAL,
    ProfileConvergenceError,
    evaluate_profile_convergence,
    load_input,
    write_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Compact deterministic JSON whose losses are indexed as "
            "[method][case][profile][spacing]."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--contract-proposal",
        type=Path,
        default=DEFAULT_CONTRACT_PROPOSAL,
        help="Candidate contract whose convergence constants are checked for drift.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document, input_sha256 = load_input(args.input)
        evidence = evaluate_profile_convergence(
            document,
            contract_proposal=args.contract_proposal,
            input_byte_sha256=input_sha256,
        )
        identity = write_evidence(evidence, args.output)
    except ProfileConvergenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    result = {
        "output": identity,
        "status": evidence["status"],
        "profile_resolution_activation_eligible": evidence[
            "profile_resolution_activation_eligible"
        ],
        "does_not_activate_scoring_contract": True,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if evidence["profile_resolution_activation_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
