"""The shared AhmedML/WindsorML participant command line."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reference.native_participant import ParticipantWorkflow


def main(dataset, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Prepare, evaluate, resume and package {dataset} submissions on dev."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch-data", "fetch-support"):
        sub = commands.add_parser(name)
        sub.add_argument("--split-id", required=True)
        sub.add_argument("--destination", type=Path, required=name == "fetch-data")
        sub.add_argument(
            "--dry-run" if name == "fetch-data" else "--offline", action="store_true"
        )
    for name in ("validate-entry", "evaluate-entry"):
        sub = commands.add_parser(name)
        sub.add_argument("entry_root", type=Path)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--support-root", type=Path)
        sub.add_argument("--offline", action="store_true")
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
    args = parser.parse_args(argv)
    workflow = ParticipantWorkflow(dataset)
    try:
        if args.command == "fetch-data":
            result = workflow.fetch_data(
                args.split_id, args.destination, dry_run=args.dry_run
            )
        elif args.command == "fetch-support":
            result = workflow.fetch_support(
                args.split_id, args.destination, offline=args.offline
            )
        elif args.command == "validate-entry":
            result = workflow.preflight(
                args.entry_root,
                args.dataset_root,
                args.support_root,
                offline=args.offline,
            )
        elif args.command == "evaluate-entry":
            result = workflow.evaluate_entry(
                args.entry_root,
                args.dataset_root,
                args.output,
                args.support_root,
                resume=args.resume,
                offline=args.offline,
            )
        elif args.command == "package":
            from scripts.assemble_ahmedml_schema_v3_candidate import (
                _checkpoint_arguments,
            )

            result = workflow.package(
                args.evaluation_root,
                args.config,
                args.discretization_cases,
                args.output,
                _checkpoint_arguments(args.checkpoint),
            )
        elif args.command == "verify-package":
            result = workflow.verify_package(args.directory)
        else:
            result = workflow.report(args.evaluation_root, args.case_id, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
