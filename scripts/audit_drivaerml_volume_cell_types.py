#!/usr/bin/env python3
"""Stream the native VTU cell-type array into a deterministic histogram."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.evaluator import validate_native_source_contract  # noqa: E402
from reference.drivaerml.source import (  # noqa: E402
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
    open_verified_multipart,
    stream_inline_binary_payload,
)


_DTYPES = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
}


class _HistogramSink:
    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype
        self.pending = bytearray()
        self.counts: Counter[int] = Counter()

    def write(self, block: bytes) -> int:
        self.pending.extend(block)
        width = self.dtype.itemsize
        complete = len(self.pending) - len(self.pending) % width
        if complete:
            values = np.frombuffer(bytes(self.pending[:complete]), dtype=self.dtype)
            unique, counts = np.unique(values, return_counts=True)
            self.counts.update(
                {int(value): int(count) for value, count in zip(unique, counts, strict=True)}
            )
            del self.pending[:complete]
        return len(block)

    def finish(self) -> dict[str, int]:
        if self.pending:
            raise ValueError("cell-type payload ended within one scalar")
        return {str(key): self.counts[key] for key in sorted(self.counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--multipart", action="store_true")
    transport.add_argument("--monolithic-vtu", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--io-chunk-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

    pin = load_native_source_pin(args.native_source_pin)
    pin_sha256 = validate_native_source_contract(pin)
    case = pin.case(args.case_id)
    resolved = pin.resolve(args.case_id, args.dataset_root)
    stream = (
        open_verified_multipart(resolved, verification_chunk_size=args.io_chunk_bytes)
        if args.multipart
        else open_verified_monolithic(
            resolved,
            args.monolithic_vtu,
            verification_chunk_size=args.io_chunk_bytes,
        )
    )
    with closing(stream):
        index = index_inline_binary_vtk_xml(stream, scan_chunk_size=args.io_chunk_bytes)
        arrays = index.arrays_for(association="Cells", name="types")
        if len(arrays) != 1 or arrays[0].number_of_components != 1:
            parser.error("native volume must contain one scalar Cells/types array")
        type_code = arrays[0].vtk_type
        if type_code not in _DTYPES:
            parser.error(f"unsupported cell-type scalar {type_code!r}")
        byte_order = "<" if index.byte_order == "LittleEndian" else ">"
        dtype = np.dtype(byte_order + _DTYPES[type_code])
        sink = _HistogramSink(dtype)
        summary = stream_inline_binary_payload(
            stream,
            index,
            arrays[0],
            sink,
            encoded_chunk_size=args.io_chunk_bytes,
        )
        histogram = sink.finish()
        stream.assert_unchanged(context="after cell-type histogram")

    document = {
        "schema": "drivaerml-native-volume-cell-type-audit-v1",
        "status": "candidate_diagnostic_not_volume_weight_definition",
        "case_id": args.case_id,
        "native_source_pin_sha256": pin_sha256,
        "source_volume_part_sha256": [part.sha256 for part in case.volume_parts],
        "cell_count": index.pieces[0].number_of_cells,
        "cell_type_vtk_scalar": type_code,
        "cell_type_payload_sha256": summary.payload_sha256,
        "cell_type_histogram": histogram,
        "vtk_cell_type_names": {
            key: str(__import__("vtk").vtkCellTypes.GetClassNameFromTypeId(int(key)))
            for key in histogram
        },
    }
    if sum(histogram.values()) != document["cell_count"]:
        parser.error("cell-type histogram does not cover every native cell")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
