#!/usr/bin/env python3
"""Audit one reconstructed native DrivAerML volume with bounded memory."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.native_fields import (  # noqa: E402
    audit_and_compare_inline_native_cell_data,
)
from reference.drivaerml.source import (  # noqa: E402
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
)


def _relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), np.finfo(np.float64).tiny)
    return abs(left - right) / scale


def compare_metric_passes(left: object, right: object) -> dict[str, object]:
    if (
        left.field_name != right.field_name
        or left.statistics.entity_count != right.statistics.entity_count
        or left.statistics.component_count != right.statistics.component_count
        or left.source_payload.payload_sha256 != right.source_payload.payload_sha256
    ):
        raise ValueError("metric passes do not refer to the same complete native field")
    additive: dict[str, dict[str, float]] = {}
    maximum_relative = 0.0
    maximum_metric_absolute = 0.0
    left_sums = left.statistics.uniform
    right_sums = right.statistics.uniform
    if left_sums.entity_count != right_sums.entity_count:
        raise ValueError("metric partitions have different entity coverage")
    for name in (
        "absolute_error",
        "squared_error",
        "squared_truth",
        "total_weight",
    ):
        left_value = float(getattr(left_sums, name))
        right_value = float(getattr(right_sums, name))
        relative = _relative_difference(left_value, right_value)
        maximum_relative = max(maximum_relative, relative)
        additive[name] = {
            "absolute_difference": abs(left_value - right_value),
            "relative_difference": relative,
        }
    left_metrics = left.statistics.metric_values()["uniform"]
    right_metrics = right.statistics.metric_values()["uniform"]
    maximum_metric_absolute = max(
        abs(left_metrics[name] - right_metrics[name]) for name in left_metrics
    )
    return {
        "same_source_payload_sha256": True,
        "same_complete_entity_coverage": True,
        "equal_native_cell_additive_sums": additive,
        "maximum_additive_relative_difference": maximum_relative,
        "maximum_metric_absolute_difference": maximum_metric_absolute,
    }


def equal_native_cell_metric_pass(metric_pass: object) -> dict[str, object]:
    """Serialize only the frozen one-weight-per-native-cell statistic stream."""

    statistics = metric_pass.statistics
    return {
        "field_name": metric_pass.field_name,
        "chunk_entities": metric_pass.chunk_entities,
        "source_payload_sha256": metric_pass.source_payload.payload_sha256,
        "source_payload_bytes": metric_pass.source_payload.decoded_payload_bytes,
        "entity_count": statistics.entity_count,
        "component_count": statistics.component_count,
        "metrics": statistics.metric_values()["uniform"],
        "additive_sums": asdict(statistics.uniform),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--monolithic-vtu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-chunk-cells", type=int, default=1_000_003)
    parser.add_argument("--comparison-chunk-cells", type=int, default=777_779)
    parser.add_argument("--io-chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--invariance-relative-tolerance", type=float, default=2e-12)
    parser.add_argument("--invariance-metric-tolerance", type=float, default=2e-12)
    args = parser.parse_args()
    if (
        args.reference_chunk_cells < 1
        or args.comparison_chunk_cells < 1
        or args.reference_chunk_cells == args.comparison_chunk_cells
        or args.io_chunk_bytes < 4
    ):
        raise ValueError("chunk sizes must be positive and independently partitioned")

    started = time.time()
    pin = load_native_source_pin(args.pin)
    resolved = pin.resolve(args.case_id, args.dataset_root)
    with open_verified_monolithic(
        resolved,
        args.monolithic_vtu,
        verification_chunk_size=args.io_chunk_bytes,
    ) as stream:
        verified_segments = [
            {
                **asdict(row),
                "path": str(row.path),
            }
            for row in stream.verification
        ]
        indexed_at = time.time()
        vtk_index = index_inline_binary_vtk_xml(
            stream, scan_chunk_size=args.io_chunk_bytes
        )
        index_finished_at = time.time()
        if vtk_index.dataset_type != "UnstructuredGrid" or len(vtk_index.pieces) != 1:
            raise ValueError("native volume must be one UnstructuredGrid Piece")
        piece = vtk_index.pieces[0]
        field_audits: dict[str, object] = {}
        metric_passes: dict[str, object] = {}
        requirements = {
            "pMeanTrim": (1, "m^2/s^2"),
            "UMeanTrim": (3, "m/s"),
        }
        for name, (components, units) in requirements.items():
            arrays = vtk_index.arrays_for(association="CellData", name=name)
            if len(arrays) != 1:
                raise ValueError(
                    f"volume must contain exactly one CellData {name!r} array"
                )
            array = arrays[0]
            if array.vtk_type != "Float32" or array.number_of_components != components:
                raise ValueError(
                    f"CellData {name!r} must be Float32 with {components} component(s)"
                )
            audit, reference, comparison = audit_and_compare_inline_native_cell_data(
                stream,
                vtk_index,
                array,
                units=units,
                predictions=None,
                physical_weights=None,
                reference_chunk_entities=args.reference_chunk_cells,
                comparison_chunk_entities=args.comparison_chunk_cells,
                encoded_chunk_size=args.io_chunk_bytes,
            )
            field_audits[name] = audit
            invariance = compare_metric_passes(reference, comparison)
            if (
                invariance["maximum_additive_relative_difference"]
                > args.invariance_relative_tolerance
                or invariance["maximum_metric_absolute_difference"]
                > args.invariance_metric_tolerance
            ):
                raise ValueError("complete-case metrics depend on the chunk partition")
            metric_passes[name] = {
                "prediction_role": "all_zero_invariance_fixture_not_a_published_baseline",
                "reference_partition": equal_native_cell_metric_pass(reference),
                "comparison_partition": equal_native_cell_metric_pass(comparison),
                "invariance": invariance,
            }
        field_audit_finished_at = time.time()

    receipt = {
        "schema": "drivaerml-native-volume-case-audit-v2",
        "status": "passed_candidate_evaluator_case_audit",
        "case_id": args.case_id,
        "public_source": {
            "repository_id": pin.repository_id,
            "immutable_revision": pin.repository_revision,
            "logical_path": resolved.record.volume_logical_path.as_posix(),
            "logical_size_bytes": resolved.record.volume_total_size_bytes,
            "multipart_part_count": len(resolved.record.volume_parts),
            "verified_monolithic_segments": verified_segments,
        },
        "vtk": {
            "dataset_type": vtk_index.dataset_type,
            "version": vtk_index.version,
            "byte_order": vtk_index.byte_order,
            "header_type": vtk_index.header_type,
            "compressor": vtk_index.compressor,
            "piece": asdict(piece),
            "data_arrays_in_raw_xml_order": [asdict(row) for row in vtk_index.data_arrays],
        },
        "required_cell_data": {
            name: audit.to_json() for name, audit in field_audits.items()
        },
        "units": {"coordinates": "m", "pMeanTrim": "m^2/s^2", "UMeanTrim": "m/s"},
        "raw_cell_order": "zero-based VTK Piece CellData tuple order, unchanged",
        "coverage": {
            "expected_raw_id_interval": [0, piece.number_of_cells],
            "validation": "each metric pass finalized exact gap-free duplicate-free coverage",
        },
        "volume_weighting": {
            "weighting": "one_per_native_cell",
            "entity_count": piece.number_of_cells,
            "total_weight": float(piece.number_of_cells),
            "geometric_cell_volume_weights_used": False,
        },
        "metrics": metric_passes,
        "chunk_invariance_tolerances": {
            "maximum_additive_relative_difference": args.invariance_relative_tolerance,
            "maximum_metric_absolute_difference": args.invariance_metric_tolerance,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "segment_verification_seconds": indexed_at - started,
            "xml_index_seconds": index_finished_at - indexed_at,
            "field_audit_and_equal_cell_metric_seconds": (
                field_audit_finished_at - index_finished_at
            ),
            "total_seconds": time.time() - started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"PASS {args.case_id}: {piece.number_of_cells} cells, "
        f"{len(resolved.record.volume_parts)} verified parts"
    )


if __name__ == "__main__":
    main()
