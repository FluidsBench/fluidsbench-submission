#!/usr/bin/env python3
"""Evaluate one closed-candidate DrivAerML case from native prediction chunks."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.evaluator import (  # noqa: E402
    DrivAerCandidateEvaluatorError,
    evaluate_candidate_case,
    validate_native_source_contract,
    write_candidate_case_evidence,
)
from reference.drivaerml.native_surface import (  # noqa: E402
    DrivAerNativeSurfaceError,
    audit_fixed_surface_area_file,
    load_native_surface_vtp,
)
from reference.drivaerml.prediction_chunks import PredictionChunkError  # noqa: E402
from reference.drivaerml.source import (  # noqa: E402
    NativeSourceError,
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
    open_verified_multipart,
)


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one DrivAerML case with the closed candidate evaluator. "
            "This does not create or validate an official submission."
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing the pinned run_N native files or multipart files.",
    )
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--monolithic-vtu",
        type=Path,
        help="Verified byte reconstruction; defaults to the pinned logical path.",
    )
    transport.add_argument(
        "--multipart",
        action="store_true",
        help="Open and concatenate the pinned .part files without materializing a VTU.",
    )
    parser.add_argument("--surface-area-npy", type=Path, required=True)
    parser.add_argument("--surface-prediction-manifest", type=Path, required=True)
    parser.add_argument("--volume-prediction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-prediction-chunk-rows",
        type=_positive_integer,
        default=1_000_000,
    )
    parser.add_argument(
        "--io-chunk-bytes",
        type=_positive_integer,
        default=8 * 1024 * 1024,
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    pin = load_native_source_pin(args.native_source_pin)
    validate_native_source_contract(pin)
    case = pin.case(args.case_id)
    resolved = pin.resolve(args.case_id, args.dataset_root)
    surface = load_native_surface_vtp(resolved.boundary_path)
    areas = audit_fixed_surface_area_file(
        surface,
        args.surface_area_npy,
        expected_area_sha256=case.surface_cell_area.sha256,
        source_boundary_sha256=case.surface_cell_area.source_boundary_sha256,
    )
    if args.multipart:
        volume_stream = open_verified_multipart(
            resolved,
            verification_chunk_size=args.io_chunk_bytes,
        )
    else:
        volume_stream = open_verified_monolithic(
            resolved,
            args.monolithic_vtu,
            verification_chunk_size=args.io_chunk_bytes,
        )
    with closing(volume_stream) as stream:
        vtk_index = index_inline_binary_vtk_xml(
            stream,
            scan_chunk_size=args.io_chunk_bytes,
        )
        if len(vtk_index.pieces) != 1:
            raise DrivAerCandidateEvaluatorError(
                "native DrivAerML volume must contain exactly one Piece"
            )
        evaluation = evaluate_candidate_case(
            case_id=args.case_id,
            native_source_pin=pin,
            native_surface=surface,
            fixed_surface_areas=areas,
            volume_stream=stream,
            volume_vtk_index=vtk_index,
            surface_prediction_manifest=args.surface_prediction_manifest,
            volume_prediction_manifest=args.volume_prediction_manifest,
            maximum_prediction_chunk_rows=args.maximum_prediction_chunk_rows,
            hash_chunk_bytes=args.io_chunk_bytes,
            validation_block_rows=args.maximum_prediction_chunk_rows,
            encoded_chunk_bytes=args.io_chunk_bytes,
        )
    return write_candidate_case_evidence(evaluation, args.output)


def main() -> int:
    args = parse_args()
    try:
        identity = run(args)
    except (
        DrivAerCandidateEvaluatorError,
        DrivAerNativeSurfaceError,
        NativeSourceError,
        PredictionChunkError,
    ) as error:
        raise SystemExit(f"candidate evaluation failed: {error}") from error
    print(
        f"PASS {args.case_id}: {identity['byte_size']} evidence bytes, "
        f"sha256={identity['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
