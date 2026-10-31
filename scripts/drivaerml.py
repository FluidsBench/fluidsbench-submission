#!/usr/bin/env python3
"""Prepare, evaluate, resume and package DrivAerML submissions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.drivaerml import participant_workflow as workflow


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ["fetch-data", "fetch-support"]:
        sub = commands.add_parser(name)
        sub.add_argument("--split-id", required=True)
        sub.add_argument("--destination", type=Path, required=True)
        if name == "fetch-data":
            sub.add_argument(
                "--prediction-scope",
                choices=workflow.SCOPES,
                default="surface_and_volume",
            )
            sub.add_argument("--dry-run", action="store_true")
        else:
            sub.add_argument("--offline", action="store_true")
    for name in ["validate-entry", "evaluate-entry"]:
        sub = commands.add_parser(name)
        sub.add_argument("entry_root", type=Path)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--support-root", type=Path)
        sub.add_argument("--offline", action="store_true")
        sub.add_argument("--maximum-chunk-rows", type=int, default=1_000_000)
        if name == "evaluate-entry":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("package")
    sub.add_argument("--evaluation-root", type=Path, required=True)
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument("--discretization-cases", type=Path, required=True)
    sub.add_argument("--checkpoint", action="append", default=[], metavar="ID=PATH")
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--support-root", type=Path)
    sub = commands.add_parser("verify-package")
    sub.add_argument("directory", type=Path)
    sub.add_argument("--support-root", type=Path)
    sub = commands.add_parser("report")
    sub.add_argument("--evaluation-root", type=Path, required=True)
    sub.add_argument("--case-id", required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--support-root", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch-data":
            result = workflow.fetch_data(
                args.split_id,
                args.destination,
                args.prediction_scope,
                dry_run=args.dry_run,
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
                maximum_chunk_rows=args.maximum_chunk_rows,
            )
        elif args.command == "evaluate-entry":
            result = workflow.evaluate_entry(
                args.entry_root,
                args.dataset_root,
                args.output,
                args.support_root,
                resume=args.resume,
                offline=args.offline,
                maximum_chunk_rows=args.maximum_chunk_rows,
            )
        elif args.command == "package":
            from scripts.assemble_windsorml_schema_v3_candidate import (
                _checkpoint_arguments,
            )

            result = workflow.package(
                args.evaluation_root,
                args.config,
                args.discretization_cases,
                args.output,
                _checkpoint_arguments(args.checkpoint),
                args.support_root,
            )
        elif args.command == "verify-package":
            result = workflow.verify_package(args.directory, args.support_root)
        else:
            result = workflow.report(
                args.evaluation_root, args.case_id, args.output, args.support_root
            )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
