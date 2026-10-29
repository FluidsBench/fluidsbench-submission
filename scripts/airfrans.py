#!/usr/bin/env python3
"""Prepare, evaluate, resume and package AirfRANS submissions on dev."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
# Running this file puts scripts/ first on sys.path, where this module would shadow
# the installed AirfRANS library ("import airfrans"). Drop it before importing
# anything; the per-case worker runs this same file, so it is covered too.
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPTS]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.airfrans import participant as workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser(
        "fetch-support", help="Download and verify the prepared support"
    )
    sub.add_argument("--destination", type=Path, required=True)
    sub.add_argument("--archive-dir", type=Path)
    sub.add_argument("--offline", action="store_true")
    sub.add_argument("--dry-run", action="store_true")
    for name in ("validate-entry", "evaluate-entry"):
        sub = commands.add_parser(name)
        sub.add_argument("entry_root", type=Path)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--support-root", type=Path, required=True)
        if name == "evaluate-entry":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("package")
    sub.add_argument("--evaluation-root", type=Path, required=True)
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument("--discretization-cases", type=Path, required=True)
    sub.add_argument("--checkpoint", action="append", default=[], metavar="ID=PATH")
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("verify-package")
    sub.add_argument("directory", type=Path)
    sub = commands.add_parser("report")
    sub.add_argument("--evaluation-root", type=Path, required=True)
    sub.add_argument("--case-id", required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("_evaluate-case", help=argparse.SUPPRESS)
    sub.add_argument("--case-id", required=True)
    sub.add_argument("--prediction", type=Path, required=True)
    sub.add_argument("--dataset-root", type=Path, required=True)
    sub.add_argument("--truth", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-support":
            result = workflow.fetch_support(
                args.destination,
                offline=args.offline,
                archive_dir=args.archive_dir,
                dry_run=args.dry_run,
            )
        elif args.command == "validate-entry":
            campaign = workflow.preflight(
                args.entry_root, args.dataset_root, args.support_root
            )
            result = {
                "status": "valid_entry",
                "case_count": len(campaign["case_ids"]),
                "entry": campaign["entry"],
            }
        elif args.command == "evaluate-entry":
            result = workflow.evaluate_entry(
                args.entry_root,
                args.dataset_root,
                args.support_root,
                args.output,
                resume=args.resume,
            )
        elif args.command == "package":
            checkpoints = {}
            for item in args.checkpoint:
                key, separator, value = item.partition("=")
                if not separator or not key or not value or key in checkpoints:
                    raise workflow.WorkflowError(
                        "checkpoint arguments must be unique ID=PATH pairs"
                    )
                checkpoints[key] = Path(value)
            result = workflow.package(
                args.evaluation_root,
                args.config,
                args.discretization_cases,
                args.output,
                checkpoints,
            )
        elif args.command == "verify-package":
            result = workflow.verify_package(args.directory)
        elif args.command == "report":
            result = workflow.report(args.evaluation_root, args.case_id, args.output)
        else:
            workflow.evaluate_case(
                args.case_id,
                args.prediction,
                args.dataset_root,
                args.truth,
                args.output,
            )
            result = {"status": "case_evaluated", "case_id": args.case_id}
    except (
        ValueError,
        RuntimeError,
        OSError,
        ImportError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
