#!/usr/bin/env python3
"""Generate one pinned DrivAerML case's candidate velocity-cell mappings.

This maintainer command is deliberately candidate-only.  It verifies every
ordered byte range of a reconstructed monolithic VTU against the immutable
native-source pin, then gives VTK a verified descriptor alias to that same
retained inode.  It then emits complete, deterministic assignment evidence for all sixteen
AutoCFD5 lines at 1, 2, 5, and 10 mm.  It does not claim profile-resolution
convergence, model-order preservation, an owner validity mask, scientific
sign-off, or activation of the scoring contract.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    AutoCFD5Error,
    load_autocfd5_submission_definition,
)
from reference.drivaerml.source import (  # noqa: E402
    NativeSourceError,
    SegmentVerification,
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
)


RECEIPT_SCHEMA = "drivaerml-velocity-cell-assignments-case-candidate-v1"
ARTIFACT_SCHEMA = "drivaerml-velocity-cell-mapping-candidate-v1"
RECEIPT_STATUS = "candidate_complete_geometry_mapping_not_activation_evidence"
EXPECTED_DIAGNOSTIC_PROFILE_SHA256 = (
    "df22bc807b62f925c32659d681ac44064e6acf46449038b8431b1e9139aba1e8"
)
# Compatibility name retained for downstream candidate tooling; it now binds
# the probe-free v9 diagnostic registry, not the historical v8 research file.
EXPECTED_AUTOCFD5_PROFILE_SHA256 = EXPECTED_DIAGNOSTIC_PROFILE_SHA256
EXPECTED_DATASET_REPOSITORY_ID = "neashton/drivaerml"
EXPECTED_DATASET_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
DEFAULT_PROFILE = (
    ROOT / "benchmark-specs" / "drivaerml" / "drivaerml-diagnostics-v9.json"
)
DEFAULT_IO_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_VALIDATION_CHUNK_CELLS = 1_000_000
RESOLUTIONS = (
    (1, 0.001, "01mm"),
    (2, 0.002, "02mm"),
    (5, 0.005, "05mm"),
    (10, 0.010, "10mm"),
)
EXPECTED_SAMPLE_COUNTS = {
    1: 37_416,
    2: 18_716,
    5: 7_496,
    10: 3_756,
}


class VelocityAssignmentCLIError(ValueError):
    """Raised when strict one-case assignment generation cannot continue."""


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise VelocityAssignmentCLIError(f"{label} must be a positive integer")
    return value


def _positive_integer_argument(text: str) -> int:
    try:
        return _positive_integer(int(text), "value")
    except (ValueError, VelocityAssignmentCLIError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as source:
            while block := source.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise VelocityAssignmentCLIError(f"cannot hash required input: {error}") from error
    return digest.hexdigest()


def _atomic_compact_json(path: Path, value: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verified_segment_binding(
    verifications: tuple[SegmentVerification, ...],
    expected_parts: Sequence[Any],
) -> list[dict[str, int | str]]:
    parts = tuple(expected_parts)
    if len(verifications) != len(parts):
        raise VelocityAssignmentCLIError(
            "verified monolithic source does not cover every pinned segment"
        )
    result: list[dict[str, int | str]] = []
    expected_offset = 0
    for part_index, (verification, part) in enumerate(
        zip(verifications, parts, strict=True)
    ):
        if (
            int(part.part_index) != part_index
            or int(verification.file_offset) != expected_offset
            or int(verification.size_bytes) != int(part.size_bytes)
            or verification.sha256 != part.sha256
        ):
            raise VelocityAssignmentCLIError(
                "verified monolithic segment differs from the native-source pin"
            )
        result.append(
            {
                "part_index": part_index,
                "byte_offset": expected_offset,
                "size_bytes": int(verification.size_bytes),
                "sha256": verification.sha256,
            }
        )
        expected_offset += int(verification.size_bytes)
    return result


def _profile_binding(profile_path: Path) -> tuple[Any, dict[str, object]]:
    initial_hash = _sha256_file(profile_path)
    if initial_hash != EXPECTED_DIAGNOSTIC_PROFILE_SHA256:
        raise VelocityAssignmentCLIError(
            "diagnostic profile does not match the exact probe-free v9 SHA-256: "
            f"expected {EXPECTED_DIAGNOSTIC_PROFILE_SHA256}, got {initial_hash}"
        )
    try:
        definition = load_autocfd5_submission_definition(profile_path)
    except AutoCFD5Error as error:
        raise VelocityAssignmentCLIError(
            f"v9 diagnostic registry validation failed: {error}"
        ) from error
    if _sha256_file(profile_path) != initial_hash:
        raise VelocityAssignmentCLIError(
            "diagnostic profile changed while its registries were validated"
        )
    return definition, {
        "profile_sha256": initial_hash,
        "source_registry_sha256": {
            name: digest for name, digest in definition.source_sha256
        },
        "line_count": len(definition.velocity_lines),
        "fixed_10mm_sample_count": len(definition.velocity_samples),
        "continuous_cp_cut_count": len(definition.pressure_cut_ids),
        "discrete_cp_probe_count": 0,
    }


def _native_source_binding(
    *,
    native_source_pin: Path,
    dataset_root: Path,
    case_id: str,
    monolithic_vtu: Path,
    io_chunk_bytes: int,
) -> tuple[
    Any,
    dict[str, object],
    dict[str, object],
    tuple[str, ...],
    Any,
    Any,
]:
    """Verify, index, and read geometry from one continuously retained inode."""

    pin_hash = _sha256_file(native_source_pin)
    try:
        pin = load_native_source_pin(native_source_pin)
    except NativeSourceError as error:
        raise VelocityAssignmentCLIError(
            f"native-source pin validation failed: {error}"
        ) from error
    if _sha256_file(native_source_pin) != pin_hash:
        raise VelocityAssignmentCLIError(
            "native-source pin changed while it was being validated"
        )
    if (
        pin.repository_id != EXPECTED_DATASET_REPOSITORY_ID
        or pin.repository_revision != EXPECTED_DATASET_REVISION
    ):
        raise VelocityAssignmentCLIError(
            "native-source pin must identify the immutable public DrivAerML release "
            f"{EXPECTED_DATASET_REPOSITORY_ID}@{EXPECTED_DATASET_REVISION}"
        )
    try:
        case = pin.case(case_id)
        resolved = pin.resolve(case_id, dataset_root)
    except NativeSourceError as error:
        raise VelocityAssignmentCLIError(str(error)) from error
    source = monolithic_vtu.expanduser().resolve()
    try:
        with open_verified_monolithic(
            resolved,
            source,
            verification_chunk_size=io_chunk_bytes,
        ) as verified_stream:
            segments = _verified_segment_binding(
                verified_stream.verification, case.volume_parts
            )
            verified_stream.seek(0)
            vtk_index = index_inline_binary_vtk_xml(
                verified_stream,
                scan_chunk_size=io_chunk_bytes,
            )
            if (
                vtk_index.dataset_type != "UnstructuredGrid"
                or len(vtk_index.pieces) != 1
            ):
                raise VelocityAssignmentCLIError(
                    "native volume must declare exactly one UnstructuredGrid Piece"
                )
            piece = vtk_index.pieces[0]
            if vtk_index.source_size_bytes != case.volume_total_size_bytes:
                raise VelocityAssignmentCLIError(
                    "indexed VTU byte count differs from the pinned reconstructed size"
                )

            # The VTK runtime was already checked before this large source scan.
            # Crucially, its geometry reader does not reopen ``source``: it
            # receives an fd-filesystem alias checked to resolve to the retained
            # verified device/inode, and is invoked only after segment checks.
            from reference.drivaerml.native_volume_geometry import (
                read_geometry_only_vtu,
            )

            vtk_source = verified_stream.single_file_descriptor_path()
            grid, reader_audit = read_geometry_only_vtu(vtk_source)
            verified_stream.assert_unchanged(
                context="while VTK read the verified monolithic source"
            )
    except VelocityAssignmentCLIError:
        raise
    except (ImportError, NativeSourceError, OSError, ValueError) as error:
        raise VelocityAssignmentCLIError(
            f"pinned retained-descriptor geometry load failed: {error}"
        ) from error
    source_hashes = tuple(part.sha256 for part in case.volume_parts)
    native_binding: dict[str, object] = {
        "pin_sha256": pin_hash,
        "repository_id": pin.repository_id,
        "repository_revision": pin.repository_revision,
        "case_id": case_id,
        "logical_size_bytes": case.volume_total_size_bytes,
        "ordered_verified_segments": segments,
        "verification": {
            "method": "exact_ordered_segment_size_and_sha256",
            "timing": "completed_before_vtk_geometry_reader",
            "vtk_input": "retained_verified_file_descriptor",
            "post_vtk_fstat": "unchanged",
        },
    }
    xml_binding: dict[str, object] = {
        "dataset_type": vtk_index.dataset_type,
        "version": vtk_index.version,
        "byte_order": vtk_index.byte_order,
        "header_type": vtk_index.header_type,
        "compressor": vtk_index.compressor,
        "piece_count": len(vtk_index.pieces),
        "declared_point_count": piece.number_of_points,
        "declared_cell_count": piece.number_of_cells,
    }
    return (
        case,
        native_binding,
        xml_binding,
        source_hashes,
        grid,
        reader_audit,
    )


def _samples_for_resolution(definition: Any, spacing_mm: int, spacing_m: float) -> tuple:
    # Keep the helper import local so generic CLI inspection remains lightweight.
    from reference.drivaerml.velocity_assignments import (
        generate_definition_velocity_samples,
    )

    if spacing_mm == 10:
        samples = definition.velocity_samples
    else:
        samples = generate_definition_velocity_samples(definition, spacing_m)
    expected = EXPECTED_SAMPLE_COUNTS[spacing_mm]
    if len(samples) != expected:
        raise VelocityAssignmentCLIError(
            f"{spacing_mm} mm grid has {len(samples)} samples; expected {expected}"
        )
    keys = tuple((sample.profile_id, sample.sample_index) for sample in samples)
    if len(keys) != len(set(keys)):
        raise VelocityAssignmentCLIError(
            f"{spacing_mm} mm grid contains duplicate line/sample IDs"
        )
    if len({sample.profile_id for sample in samples}) != 16:
        raise VelocityAssignmentCLIError(
            f"{spacing_mm} mm grid does not cover all sixteen lines"
        )
    return tuple(samples)


def _assignment_artifact(
    *,
    case_id: str,
    spacing_mm: int,
    spacing_m: float,
    sample_source: str,
    assignments: Sequence[Any],
    source_sha256: tuple[str, ...],
    registry_binding: dict[str, object],
    native_binding: dict[str, object],
    geometry_binding: dict[str, object],
    kernel_receipt: dict[str, object],
) -> dict[str, object]:
    from reference.drivaerml.velocity_assignments import (
        assignment_evidence_sha256,
    )

    records = tuple(assignments)
    expected = EXPECTED_SAMPLE_COUNTS[spacing_mm]
    if len(records) != expected:
        raise VelocityAssignmentCLIError(
            f"{spacing_mm} mm assignments omit required samples"
        )
    keys = tuple((row.profile_id, row.sample_index) for row in records)
    if len(keys) != len(set(keys)) or len({key[0] for key in keys}) != 16:
        raise VelocityAssignmentCLIError(
            f"{spacing_mm} mm assignments are not complete and duplicate-free"
        )
    if any(
        row.case_id != case_id
        or row.source_sha256 != source_sha256
        or not math.isclose(
            row.geometric_tolerance_m,
            POINT_IN_CELL_CLOSURE_TOLERANCE_M,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for row in records
    ):
        raise VelocityAssignmentCLIError(
            f"{spacing_mm} mm assignments lost constant evidence bindings"
        )
    invalid_reasons = collections.Counter(
        row.reason for row in records if not row.valid
    )
    line_sample_counts = collections.Counter(row.profile_id for row in records)
    candidate_histogram = collections.Counter(row.candidate_count for row in records)
    raw_ids = [
        int(row.raw_vtk_cell_id)
        for row in records
        if row.raw_vtk_cell_id is not None
    ]
    rows = [
        [
            row.profile_id,
            row.sample_index,
            list(row.point_m),
            row.distance_m,
            row.valid,
            row.reason,
            row.raw_vtk_cell_id,
            row.candidate_count,
        ]
        for row in records
    ]
    return {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": 1,
        "status": RECEIPT_STATUS,
        "case_id": case_id,
        "resolution": {
            "nominal_spacing_mm": spacing_mm,
            "nominal_spacing_m": spacing_m,
            "sample_source": sample_source,
            "line_count": 16,
            "sample_count": len(records),
        },
        "constant_evidence_fields": {
            "geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
            "source_sha256": list(source_sha256),
        },
        "row_fields": [
            "profile_id",
            "sample_index",
            "point_m",
            "distance_m",
            "valid",
            "reason",
            "raw_vtk_cell_id",
            "candidate_count",
        ],
        "rows": rows,
        "coverage": {
            "expected_sample_count": expected,
            "actual_sample_count": len(records),
            "unique_line_sample_key_count": len(set(keys)),
            "complete_duplicate_free_no_omissions": True,
            "line_sample_counts": {
                profile_id: count
                for profile_id, count in sorted(line_sample_counts.items())
            },
            "valid_count": sum(row.valid for row in records),
            "invalid_count": sum(not row.valid for row in records),
            "invalid_reason_counts": {
                reason: count for reason, count in sorted(invalid_reasons.items())
            },
            "candidate_count_histogram": {
                str(count): frequency
                for count, frequency in sorted(candidate_histogram.items())
            },
            "selected_raw_vtk_cell_id_min": min(raw_ids) if raw_ids else None,
            "selected_raw_vtk_cell_id_max": max(raw_ids) if raw_ids else None,
        },
        "assignment_evidence_sha256": assignment_evidence_sha256(records),
        "units": {"point_m": "m", "distance_m": "m"},
        "registries": registry_binding,
        "native_source_binding": native_binding,
        "geometry": geometry_binding,
        "kernel": kernel_receipt,
        "claims": {
            "resolution_convergence": False,
            "ranked_result_invariance": False,
            "model_ordering": False,
            "owner_validity_mask_complete": False,
            "owner_scientific_signoff": False,
            "scoring_contract_active": False,
            "official_submission": False,
        },
    }


def generate_pinned_velocity_assignments(
    *,
    case_id: str,
    native_source_pin: Path | str,
    dataset_root: Path | str,
    monolithic_vtu: Path | str,
    output_root: Path | str,
    autocfd5_profile: Path | str = DEFAULT_PROFILE,
    io_chunk_bytes: int = DEFAULT_IO_CHUNK_BYTES,
    validation_chunk_cells: int = DEFAULT_VALIDATION_CHUNK_CELLS,
) -> dict[str, object]:
    """Generate four complete path-free mapping artifacts for one pinned case."""

    if not isinstance(case_id, str) or not case_id.strip():
        raise VelocityAssignmentCLIError("case_id must be a non-empty string")
    io_size = _positive_integer(io_chunk_bytes, "io_chunk_bytes")
    validation_size = _positive_integer(
        validation_chunk_cells, "validation_chunk_cells"
    )
    destination = Path(output_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise VelocityAssignmentCLIError(f"output root already exists: {destination}")

    profile = Path(autocfd5_profile).expanduser().resolve()
    definition, registry_binding = _profile_binding(profile)
    # Fail before hashing/indexing a multi-gigabyte native source if this
    # process could only emit a receipt rejected by the strict aggregator.
    # Keep the resulting receipt and bind the exact same object into every
    # per-resolution artifact rather than probing the runtime again later.
    try:
        from reference.drivaerml.velocity_assignments import (
            NativeContainingCellKernel,
            QUERY_CACHE_KEY_ID,
            candidate_kernel_receipt,
        )

        kernel_receipt = candidate_kernel_receipt()
    except (ImportError, ValueError) as error:
        raise VelocityAssignmentCLIError(
            "frozen Python/NumPy/VTK runtime preflight failed: "
            f"{error}"
        ) from error
    pin_path = Path(native_source_pin).expanduser().resolve()
    (
        case,
        native_binding,
        xml_binding,
        source_hashes,
        grid,
        reader_audit,
    ) = _native_source_binding(
        native_source_pin=pin_path,
        dataset_root=Path(dataset_root),
        case_id=case_id,
        monolithic_vtu=Path(monolithic_vtu),
        io_chunk_bytes=io_size,
    )
    # The grid was loaded from the continuously retained verified descriptor.
    # Build the assignment kernel only after that source-bound read completes.
    try:
        kernel = NativeContainingCellKernel(
            grid, validation_chunk_size=validation_size
        )
    except (ImportError, ValueError) as error:
        raise VelocityAssignmentCLIError(
            f"exact VTK 9.5.2 native geometry load failed: {error}"
        ) from error
    if (
        int(grid.GetNumberOfPoints()) != xml_binding["declared_point_count"]
        or int(grid.GetNumberOfCells()) != xml_binding["declared_cell_count"]
    ):
        raise VelocityAssignmentCLIError(
            "VTK geometry tuple counts differ from the verified XML declarations"
        )
    geometry_binding: dict[str, object] = {
        **xml_binding,
        "vtk_loaded_point_count": int(grid.GetNumberOfPoints()),
        "vtk_loaded_cell_count": int(grid.GetNumberOfCells()),
        "reader_audit": reader_audit.as_dict(),
        "association": "native_volume_CellData",
        "raw_vtk_cell_id": "zero_based_native_GetCell_index",
        "remeshing": False,
        "reordering": False,
    }

    artifact_summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as staging_text:
        staging = Path(staging_text)
        for spacing_mm, spacing_m, label in RESOLUTIONS:
            samples = _samples_for_resolution(definition, spacing_mm, spacing_m)
            try:
                assignments = kernel.assign(
                    samples,
                    case_id=case_id,
                    source_sha256=source_hashes,
                    tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                )
            except ValueError as error:
                raise VelocityAssignmentCLIError(
                    f"{spacing_mm} mm containing-cell assignment failed: {error}"
                ) from error
            if tuple(
                (row.profile_id, row.sample_index, row.point_m, row.distance_m)
                for row in assignments
            ) != tuple(
                (
                    sample.profile_id,
                    sample.sample_index,
                    sample.point_m,
                    sample.distance_m,
                )
                for sample in samples
            ):
                raise VelocityAssignmentCLIError(
                    f"{spacing_mm} mm assignment order differs from its complete grid"
                )
            artifact = _assignment_artifact(
                case_id=case_id,
                spacing_mm=spacing_mm,
                spacing_m=spacing_m,
                sample_source=(
                    "expanded_autocfd5_v8_10mm_registry"
                    if spacing_mm == 10
                    else "endpoint_inclusive_equal_arc_from_v8_line_registry"
                ),
                assignments=assignments,
                source_sha256=source_hashes,
                registry_binding=registry_binding,
                native_binding=native_binding,
                geometry_binding=geometry_binding,
                kernel_receipt=kernel_receipt,
            )
            artifact_name = f"velocity-cell-mapping-{label}.json"
            artifact_path = staging / artifact_name
            _atomic_compact_json(artifact_path, artifact)
            coverage = artifact["coverage"]
            artifact_summaries.append(
                {
                    "nominal_spacing_mm": spacing_mm,
                    "nominal_spacing_m": spacing_m,
                    "artifact": artifact_name,
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": _sha256_file(artifact_path),
                    "assignment_evidence_sha256": artifact[
                        "assignment_evidence_sha256"
                    ],
                    "line_count": 16,
                    "sample_count": coverage["actual_sample_count"],
                    "valid_count": coverage["valid_count"],
                    "invalid_count": coverage["invalid_count"],
                    "invalid_reason_counts": coverage["invalid_reason_counts"],
                    "complete_duplicate_free_no_omissions": coverage[
                        "complete_duplicate_free_no_omissions"
                    ],
                }
            )
            del assignments, samples, artifact

        query_cache_audit = kernel.query_cache_audit()
        expected_total_rows = sum(EXPECTED_SAMPLE_COUNTS.values())
        if (
            query_cache_audit["enabled"] is not True
            or query_cache_audit["key_id"] != QUERY_CACHE_KEY_ID
            or query_cache_audit["total_rows"] != expected_total_rows
            or not isinstance(query_cache_audit["unique_query_keys"], int)
            or query_cache_audit["unique_query_keys"] < 1
            or query_cache_audit["unique_query_keys"] > expected_total_rows
            or query_cache_audit["cache_hits"]
            != expected_total_rows - query_cache_audit["unique_query_keys"]
        ):
            raise VelocityAssignmentCLIError(
                "cross-resolution containing-cell query-cache audit is inconsistent"
            )
        polyhedron_geometry_cache_audit = (
            kernel.polyhedron_geometry_cache_audit()
        )
        polyhedron_evaluation_audit = kernel.polyhedron_evaluation_audit()
        if (
            polyhedron_evaluation_audit["broad_phase_polyhedron_visit_count"]
            != polyhedron_geometry_cache_audit["cache_hits"]
            + polyhedron_geometry_cache_audit["cache_misses"]
            or polyhedron_evaluation_audit["ambiguous_count"]
            < polyhedron_geometry_cache_audit["fail_closed_preparations"]
        ):
            raise VelocityAssignmentCLIError(
                "polyhedron cache and evaluation audits are inconsistent"
            )

        receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": 1,
            "status": RECEIPT_STATUS,
            "case_id": case_id,
            "registries": registry_binding,
            "native_source_binding": native_binding,
            "geometry": geometry_binding,
            "kernel": kernel_receipt,
            "execution": {
                "io_chunk_bytes": io_size,
                "validation_chunk_cells": validation_size,
                "resolution_order_mm": [row[0] for row in RESOLUTIONS],
                "geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                "containing_cell_query_cache": query_cache_audit,
                "polyhedron_geometry_cache": polyhedron_geometry_cache_audit,
                "polyhedron_evaluation": polyhedron_evaluation_audit,
            },
            "artifacts": artifact_summaries,
            "coverage": {
                "resolution_count": len(artifact_summaries),
                "all_sixteen_lines_each_resolution": True,
                "all_samples_explicit_no_silent_omissions": True,
                "expected_sample_counts": {
                    str(key): value
                    for key, value in sorted(EXPECTED_SAMPLE_COUNTS.items())
                },
            },
            "claims": {
                "resolution_convergence": False,
                "ranked_result_invariance": False,
                "model_ordering": False,
                "owner_validity_mask_complete": False,
                "owner_scientific_signoff": False,
                "scoring_contract_active": False,
                "official_submission": False,
            },
        }
        _atomic_compact_json(staging / "receipt.json", receipt)
        if destination.exists():
            raise VelocityAssignmentCLIError(
                f"output root appeared during generation: {destination}"
            )
        os.replace(staging, destination)

    # Keep the resolved case alive through generation and make the assertion
    # explicit without putting any local path in the returned evidence.
    if case.case_id != case_id:
        raise VelocityAssignmentCLIError("native-source case identity changed")
    return receipt


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate closed-candidate, one-case AutoCFD5 native-cell assignment "
            "evidence for all 16 lines at 1, 2, 5, and 10 mm after exact source "
            "verification and a retained-descriptor VTK read. This does not prove resolution convergence, "
            "model ordering, or owner scientific approval."
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--monolithic-vtu", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--autocfd5-profile", type=Path, default=DEFAULT_PROFILE
    )
    parser.add_argument(
        "--io-chunk-bytes",
        type=_positive_integer_argument,
        default=DEFAULT_IO_CHUNK_BYTES,
    )
    parser.add_argument(
        "--validation-chunk-cells",
        type=_positive_integer_argument,
        default=DEFAULT_VALIDATION_CHUNK_CELLS,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        receipt = generate_pinned_velocity_assignments(
            case_id=args.case_id,
            native_source_pin=args.native_source_pin,
            dataset_root=args.dataset_root,
            monolithic_vtu=args.monolithic_vtu,
            output_root=args.output_root,
            autocfd5_profile=args.autocfd5_profile,
            io_chunk_bytes=args.io_chunk_bytes,
            validation_chunk_cells=args.validation_chunk_cells,
        )
    except VelocityAssignmentCLIError as error:
        parser.error(str(error))
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
