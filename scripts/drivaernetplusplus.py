#!/usr/bin/env python3
"""DrivAerNet++ participant workflow: verify data, fetch support, evaluate, package.

Run ``python scripts/drivaernetplusplus.py <command> --help`` for each step;
benchmark-specs/drivaernetplusplus/QUICKSTART.md walks through all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaernetplusplus import participant_workflow as workflow  # noqa: E402
from reference.drivaernetplusplus.contract import SPLIT_ID, ContractError, official_split  # noqa: E402
from reference.drivaernetplusplus.distribution import fetch_support  # noqa: E402


def _checkpoints(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        identifier, separator, path = value.partition("=")
        if not separator or not identifier or not path or identifier in result:
            raise ContractError("every --checkpoint must be one unique METHOD_CHECKPOINT_ID=PATH")
        result[identifier] = Path(path).expanduser().resolve()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drivaernetplusplus.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("verify-data", help="check extracted SurfacePressureVTK files against pinned SHA-256s")
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--split-id", default=SPLIT_ID)
    command.add_argument("--size-only", action="store_true", help="skip hashing (fast, weaker)")

    command = commands.add_parser("fetch-support", help="download and verify the prepared scoring support")
    command.add_argument("--split-id", default=SPLIT_ID)
    command.add_argument("--destination", type=Path, help="support cache (default ~/.cache/fluidsbench/...)")
    command.add_argument("--mirror", type=Path, help="install archives from a local copy instead of downloading")
    command.add_argument("--offline", action="store_true", help="verify an existing cache without network access")
    command.add_argument("--dry-run", action="store_true")

    command = commands.add_parser("validate-entry", help="check entry.json and every prediction file")
    command.add_argument("entry_root", type=Path)

    command = commands.add_parser("evaluate-entry", help="score every case (resumable) and reduce the split")
    command.add_argument("entry_root", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--support-root", type=Path)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--workers", type=int, default=workflow.DEFAULT_WORKERS)

    command = commands.add_parser("discretization-template", help="write per-case discretization records")
    command.add_argument("--split-id", default=SPLIT_ID)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("package", help="assemble and verify a schema-v3 dev package")
    command.add_argument("--evaluation-root", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--discretization-cases", type=Path, required=True)
    command.add_argument("--checkpoint", action="append", default=[], metavar="ID=PATH")
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("verify-package", help="re-run the dev-intake checks on a package")
    command.add_argument("package_dir", type=Path)

    command = commands.add_parser("report", help="write a self-contained HTML case report")
    command.add_argument("--evaluation-root", type=Path, required=True)
    command.add_argument("--case-id", required=True)
    command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify-data":
            result = workflow.verify_data(args.dataset_root, args.split_id, check_hashes=not args.size_only)
        elif args.command == "fetch-support":
            result = fetch_support(
                official_split(args.split_id)["case_ids"],
                destination=args.destination,
                offline=args.offline,
                mirror=args.mirror,
                dry_run=args.dry_run,
            )
        elif args.command == "validate-entry":
            result = workflow.validate_entry(args.entry_root)
        elif args.command == "evaluate-entry":
            result = workflow.evaluate_entry(
                args.entry_root, args.output, support_root=args.support_root, resume=args.resume, workers=args.workers
            )
        elif args.command == "discretization-template":
            result = workflow.discretization_template(args.split_id, args.output)
        elif args.command == "package":
            result = workflow.package(
                args.evaluation_root, args.config, args.discretization_cases, args.output, _checkpoints(args.checkpoint)
            )
        elif args.command == "verify-package":
            result = workflow.verify_package(args.package_dir)
        else:
            result = workflow.report(args.evaluation_root, args.case_id, args.output)
    except (ContractError, OSError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
