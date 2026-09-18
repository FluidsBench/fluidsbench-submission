#!/usr/bin/env python3
"""Reduce AhmedML case evidence for one closed candidate split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.dataset_scorer import (  # noqa: E402
    AhmedMLDatasetScorerError,
    score_candidate_dataset,
)


def write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--case-evidence-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--submission-specification",
        type=Path,
        default=ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json",
    )
    args = parser.parse_args()
    try:
        result = score_candidate_dataset(
            submission_specification=args.submission_specification,
            split_id=args.split_id,
            case_evidence_directory=args.case_evidence_directory,
        )
        write_atomic(args.output.expanduser().resolve(), result.to_json())
    except (AhmedMLDatasetScorerError, OSError, ValueError) as error:
        parser.error(str(error))
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
