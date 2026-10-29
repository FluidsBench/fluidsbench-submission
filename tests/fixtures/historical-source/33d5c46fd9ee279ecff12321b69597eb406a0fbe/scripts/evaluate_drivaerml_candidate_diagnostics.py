#!/usr/bin/env python3
"""Evaluate one candidate DrivAerML AutoCFD5 diagnostic case.

The command consumes strict case-local Cp and 10 mm velocity mapping evidence,
complete candidate prediction chunk manifests, and immutable native truth.  It
emits candidate-only evidence and never activates scoring or creates an
official submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import closing
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.diagnostic_evaluator import (  # noqa: E402
    DrivAerDiagnosticEvaluatorError,
    SparseNativeField,
    evaluate_loaded_case_diagnostics,
    gather_sparse_inline_native_field,
    load_strict_cp_case_support,
    load_strict_velocity_10mm_mapping,
    write_candidate_diagnostic_evidence,
)
from reference.drivaerml.cp_mapping import (  # noqa: E402
    CpMappingError,
    RetainedVerifiedRegularFile,
)
from reference.drivaerml.evaluator import (  # noqa: E402
    validate_native_source_contract,
)
from reference.drivaerml.prediction_chunks import PredictionChunkError  # noqa: E402
from reference.drivaerml.source import (  # noqa: E402
    NativeSourceError,
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
    open_verified_multipart,
)
from scripts.build_drivaerml_cp_case_support import (  # noqa: E402
    CpCaseSupportError,
    read_boundary_pressure_selection,
)


DEFAULT_PROFILE = (
    ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
)


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _sha256_file(path: Path, *, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as source:
            while block := source.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise DrivAerDiagnosticEvaluatorError(
            f"cannot hash native source {path.name}"
        ) from error
    return digest.hexdigest()


def _read_verified_boundary_pressure(
    path: Path,
    raw_polygon_ids: Sequence[int],
    *,
    expected_sha256: str,
    chunk_bytes: int,
    reader=read_boundary_pressure_selection,
):
    """Hash and parse one boundary inode retained through the VTK pass.

    VTK accepts only a filename, so it receives a verified descriptor-filesystem
    alias.  Replacing ``path`` after it is opened therefore cannot redirect the
    parser, and in-place mutation is rejected by the post-read ``fstat`` check.
    """

    try:
        with RetainedVerifiedRegularFile.open(
            path, label="native boundary VTP"
        ) as retained:
            if retained.sha256(chunk_bytes=chunk_bytes) != expected_sha256:
                raise DrivAerDiagnosticEvaluatorError(
                    "native boundary bytes differ from the immutable source pin"
                )
            descriptor_path = retained.descriptor_path()
            boundary = reader(descriptor_path, raw_polygon_ids)
            retained.finalize_verification()
            return boundary
    except CpMappingError as error:
        raise DrivAerDiagnosticEvaluatorError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--autocfd5-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--cp-support-json", type=Path, required=True)
    parser.add_argument("--velocity-mapping-json", type=Path, required=True)
    parser.add_argument("--velocity-receipt-json", type=Path, required=True)
    parser.add_argument("--surface-prediction-manifest", type=Path, required=True)
    parser.add_argument("--volume-prediction-manifest", type=Path, required=True)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument(
        "--multipart",
        action="store_true",
        help="verify and concatenate the pinned volume part files without copying",
    )
    transport.add_argument(
        "--monolithic-vtu",
        type=Path,
        help="verify a reconstructed VTU against every pinned byte segment",
    )
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run one exact candidate case and return the output file identity."""

    pin_path = args.native_source_pin.expanduser().resolve()
    pin = load_native_source_pin(pin_path)
    pin_sha256 = validate_native_source_contract(pin)
    if _sha256_file(pin_path, chunk_bytes=args.io_chunk_bytes) != pin_sha256:
        raise DrivAerDiagnosticEvaluatorError(
            "native-source pin changed during diagnostic evaluation"
        )
    case = pin.case(args.case_id)
    resolved = pin.resolve(args.case_id, args.dataset_root)
    expected_part_sha256 = tuple(part.sha256 for part in case.volume_parts)
    cp_support = load_strict_cp_case_support(
        args.cp_support_json,
        autocfd5_profile=args.autocfd5_profile,
        case_id=args.case_id,
        expected_boundary_sha256=case.boundary.sha256,
    )
    velocity_mapping = load_strict_velocity_10mm_mapping(
        args.velocity_mapping_json,
        args.velocity_receipt_json,
        autocfd5_profile=args.autocfd5_profile,
        case_id=args.case_id,
        expected_source_pin_sha256=pin_sha256,
        expected_source_part_sha256=expected_part_sha256,
    )
    cp_raw_ids = tuple(
        row.raw_vtk_polygon_id
        for row in cp_support.rows
        if row.mapping_valid and row.raw_vtk_polygon_id is not None
    )
    boundary = _read_verified_boundary_pressure(
        resolved.boundary_path,
        cp_raw_ids,
        expected_sha256=case.boundary.sha256,
        chunk_bytes=args.io_chunk_bytes,
    )
    if (
        boundary.polygon_count != cp_support.boundary_polygon_count
        or boundary.tuple_count != boundary.polygon_count
        or boundary.component_count != 1
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "native boundary pressure support differs from strict Cp evidence"
        )
    selected_surface_ids = np.asarray(
        [raw_id for raw_id, _ in boundary.selected_values], dtype=np.int64
    )
    selected_surface_values = np.asarray(
        [value for _, value in boundary.selected_values], dtype=np.float64
    )
    native_surface = SparseNativeField(
        case_id=args.case_id,
        support_id="surface_native_cells",
        field_name="pMeanTrim",
        total_row_count=boundary.polygon_count,
        raw_cell_ids=selected_surface_ids,
        values=selected_surface_values,
        source_files=(case.boundary.path.name,),
        source_sha256=(case.boundary.sha256,),
        complete_source_identity_verified=True,
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
            stream, scan_chunk_size=args.io_chunk_bytes
        )
        if (
            vtk_index.dataset_type != "UnstructuredGrid"
            or len(vtk_index.pieces) != 1
            or vtk_index.pieces[0].number_of_cells
            != velocity_mapping.native_cell_count
        ):
            raise DrivAerDiagnosticEvaluatorError(
                "native volume geometry differs from velocity mapping evidence"
            )
        velocity_arrays = vtk_index.arrays_for(
            association="CellData", name="UMeanTrim"
        )
        if len(velocity_arrays) != 1:
            raise DrivAerDiagnosticEvaluatorError(
                "native volume must contain exactly one CellData UMeanTrim array"
            )
        velocity_raw_ids = tuple(
            row.raw_vtk_cell_id
            for row in velocity_mapping.rows
            if row.valid and row.raw_vtk_cell_id is not None
        )
        native_volume, _ = gather_sparse_inline_native_field(
            stream,
            vtk_index,
            velocity_arrays[0],
            velocity_raw_ids,
            case_id=args.case_id,
            support_id="volume_native_cells",
            field_name="UMeanTrim",
            expected_components=3,
            source_files=tuple(part.path.name for part in case.volume_parts),
            source_sha256=expected_part_sha256,
            encoded_chunk_bytes=args.io_chunk_bytes,
        )
        evaluation = evaluate_loaded_case_diagnostics(
            cp_support=cp_support,
            velocity_mapping=velocity_mapping,
            surface_prediction_manifest=args.surface_prediction_manifest,
            volume_prediction_manifest=args.volume_prediction_manifest,
            native_surface_pressure=native_surface,
            native_volume_velocity=native_volume,
            maximum_prediction_chunk_rows=args.maximum_prediction_chunk_rows,
            hash_chunk_bytes=args.io_chunk_bytes,
            validation_block_rows=args.maximum_prediction_chunk_rows,
        )
    return write_candidate_diagnostic_evidence(evaluation, args.output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        identity = run(args)
    except (
        CpCaseSupportError,
        DrivAerDiagnosticEvaluatorError,
        NativeSourceError,
        PredictionChunkError,
    ) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "candidate_diagnostics_not_active_or_official_submission",
                "case_id": args.case_id,
                "evidence_byte_size": identity["byte_size"],
                "evidence_sha256": identity["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
