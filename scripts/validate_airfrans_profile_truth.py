#!/usr/bin/env python3
"""Validate an AirfRANS candidate or official profile-truth release.

The default check walks the whole release from ``manifest.json`` down: every
file is listed with a matching SHA-256 and size, every truth file is a
``ground_truth_reference`` extraction that satisfies the pinned profile
definition (stations, quantities, 1,001 samples, runtime versions, exact-copy
check), each case set equals its official split file in order, and the case
universe is exactly their union.

``--source-replay`` additionally re-extracts every case from the public
AirfRANS dataset with ``examples/airfrans-profile-extraction/extract.py`` and
requires byte-identical output. It needs the pinned extraction environment
(``requirements-airfrans-evaluator.txt``), passed with ``--python``.

``--receipt`` writes a deterministic validation record outside the release.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.airfrans.profile_truth import (  # noqa: E402
    EXTRACTOR_PATH,
    RELEASE_ID,
    ProfileTruthError,
    canonical_json_bytes,
    validate_release,
)

DEFAULT_PROFILE_DEFINITION = ROOT / "benchmark-specs" / "airfrans" / "velocity-profiles-v1.json"


def source_replay(release: Path, dataset_root: Path, python: str, jobs: int) -> list[str]:
    """Re-extract every case; return the IDs whose output differs from the release."""

    index = json.loads((release / "index.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="airfrans-profile-replay-") as scratch:

        def replay(case_id: str) -> str | None:
            output = Path(scratch) / f"{case_id}.json"
            subprocess.run(
                [python, str(ROOT / EXTRACTOR_PATH), "--dataset-root", str(dataset_root),
                 "--case-name", case_id, "--output", str(output)],
                check=True,
                capture_output=True,
            )
            truth = release / index["case_locations"][case_id]["truth_artifact"]["file"]
            return None if output.read_bytes() == truth.read_bytes() else case_id

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(replay, index["case_ids"]))
    return [case_id for case_id in results if case_id is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--profile-definition", type=Path, default=DEFAULT_PROFILE_DEFINITION)
    parser.add_argument("--source-replay", action="store_true")
    parser.add_argument("--dataset-root", type=Path, help="AirfRANS Dataset/ directory, for --source-replay.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with the pinned extraction environment.")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--receipt", type=Path, help="Write the validation record here (outside the release).")
    args = parser.parse_args(argv)
    if args.source_replay and args.dataset_root is None:
        parser.error("--source-replay requires --dataset-root")
    if args.receipt is not None and args.receipt.resolve().is_relative_to(args.release.resolve()):
        parser.error("--receipt must be outside the release")

    try:
        summary = validate_release(
            release_root=args.release,
            profile_definition_path=args.profile_definition,
            repository_root=ROOT,
        )
    except (ProfileTruthError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    record = {
        "schema": "airfrans-native-profile-truth-validation-v1-candidate",
        "schema_version": 1,
        "release_id": summary["release_id"],
        "manifest_sha256": summary["manifest_sha256"],
        "case_count": summary["case_count"],
        "case_sets": summary["case_sets"],
        "file_count": summary["file_count"],
        "release_graph_valid": True,
        "source_replay_complete": False,
    }
    if args.source_replay:
        try:
            mismatches = source_replay(args.release, args.dataset_root, args.python, args.jobs)
        except subprocess.CalledProcessError as error:
            print(f"error: extraction failed: {error.stderr.decode(errors='replace')[-2000:]}", file=sys.stderr)
            return 2
        if mismatches:
            print(f"error: source replay differs for {len(mismatches)} case(s): {mismatches[:5]}", file=sys.stderr)
            return 1
        record["source_replay_complete"] = True
        record["source_replay_case_count"] = summary["case_count"]

    payload = canonical_json_bytes(record)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(payload)
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
