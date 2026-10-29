#!/usr/bin/env python3
"""Generate the retained run_419 constant-series fixture support index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_drivaerml_run419_relative_fixture import (  # noqa: E402
    ExportError,
    build_run419_constant_series_support_index,
)


DEFAULT_OUTPUT = (
    ROOT
    / "benchmark-specs/drivaerml/support/relative-v3/"
    "run419-constant-series-support-index.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the compact run_419 constant-series support bindings from "
            "the exact constant velocity and Cp producer artifacts."
        )
    )
    parser.add_argument(
        "--constant-velocity-case-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the run_419 10 mm constant velocity mapping "
            "and receipt."
        ),
    )
    parser.add_argument(
        "--constant-cp-campaign-root",
        type=Path,
        required=True,
        help="Root of the constant Cp production_path_all484_v8 campaign.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless the retained output already equals the replayed bytes.",
    )
    return parser.parse_args()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        index = build_run419_constant_series_support_index(
            args.constant_velocity_case_dir,
            args.constant_cp_campaign_root,
        )
        payload = canonical_bytes(index)
        output = args.output.expanduser().resolve(strict=False)
        if args.check:
            try:
                retained = output.read_bytes()
            except OSError as error:
                raise ExportError(
                    f"cannot read retained constant support index: {error}"
                ) from error
            if retained != payload:
                raise ExportError(
                    "retained run_419 constant support index differs from producer replay"
                )
        else:
            write_atomic(output, payload)
    except (ExportError, OSError, ValueError) as error:
        raise SystemExit(f"constant support index generation failed: {error}") from error
    print(
        json.dumps(
            {
                "output": output.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "series_count": index["series_count"],
                "check": args.check,
                "relative_scoring_activated": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
