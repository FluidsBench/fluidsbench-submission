#!/usr/bin/env python3
"""Evaluate one complete native AhmedML development case."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    AhmedMLContractError,
    load_source_identity,
)
from reference.ahmedml.evaluator import (  # noqa: E402
    AhmedMLCandidateEvaluatorError,
    evaluate_candidate_case,
)
from reference.ahmedml.support import (  # noqa: E402
    AhmedMLSupportError,
    load_case_support,
)


DEFAULT_SOURCE_IDENTITY = (
    ROOT
    / "benchmark-specs"
    / "ahmedml"
    / "public-source-identity"
    / "ahmedml-public-source-identity-v1.json"
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
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-support", type=Path, required=True)
    parser.add_argument("--surface-prediction-manifest", type=Path, required=True)
    parser.add_argument("--volume-prediction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, default=DEFAULT_SOURCE_IDENTITY)
    parser.add_argument("--maximum-prediction-chunk-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    try:
        identity = load_source_identity(args.source_identity)
        support = load_case_support(args.case_support, source_identity=identity)
        result = evaluate_candidate_case(
            case_id=args.case_id,
            dataset_root=args.dataset_root,
            source_identity=identity,
            case_support=support,
            surface_prediction_manifest=args.surface_prediction_manifest,
            volume_prediction_manifest=args.volume_prediction_manifest,
            maximum_prediction_chunk_rows=args.maximum_prediction_chunk_rows,
        )
        write_atomic(args.output.expanduser().resolve(), result.to_json())
    except (
        AhmedMLContractError,
        AhmedMLSupportError,
        AhmedMLCandidateEvaluatorError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
