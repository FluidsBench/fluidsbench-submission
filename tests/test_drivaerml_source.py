from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.source import (
    InlineBinaryDecodeError,
    NativeSourceError,
    NativeSourceIntegrityError,
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
    open_verified_multipart,
    stream_inline_binary_payload,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inline_binary(payload: bytes, *, separate_header: bool = False) -> bytes:
    header = struct.pack("<Q", len(payload))
    if separate_header:
        return base64.b64encode(header) + b"\n" + base64.b64encode(payload)
    return base64.b64encode(header + payload)


def _vtk_document(*, incorrect_scalar_length: bool = False) -> tuple[bytes, dict[str, bytes]]:
    points = struct.pack("<6d", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    pressure = struct.pack("<2f", 3.5, -1.25)
    velocity = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    scalar_header = struct.pack(
        "<Q", len(pressure) + (4 if incorrect_scalar_length else 0)
    )
    encoded_pressure = base64.b64encode(scalar_header + pressure)
    encoded_velocity = _inline_binary(velocity, separate_header=True)
    document = b"".join(
        [
            b'<?xml version="1.0"?>\n',
            b'<VTKFile type="UnstructuredGrid" version="1.0" ',
            b'byte_order="LittleEndian" header_type="UInt64">\n',
            b"<UnstructuredGrid>\n",
            b"<FieldData>\n",
            b'<DataArray type="Float32" Name="TimeValue" NumberOfTuples="1" format="binary">\n',
            _inline_binary(struct.pack("<f", 3.6)),
            b"\n</DataArray>\n",
            b"</FieldData>\n",
            b'<Piece NumberOfPoints="2" NumberOfCells="2">\n',
            b"<Points>\n",
            b'<DataArray type="Float64" NumberOfComponents="3" format="binary">\n',
            _inline_binary(points),
            b"\n</DataArray>\n",
            b"</Points>\n",
            b"<CellData>\n",
            b'<DataArray type="Float32" Name="pMeanTrim" format="binary">\n',
            encoded_pressure,
            b"\n</DataArray>\n",
            b'<DataArray type="Float32" Name="UMeanTrim" ',
            b'NumberOfComponents="3" format="binary">\n',
            encoded_velocity,
            b"\n</DataArray>\n",
            b"</CellData>\n",
            b"</Piece>\n",
            b"</UnstructuredGrid>\n",
            b"</VTKFile>\n",
        ]
    )
    return document, {
        "points": points,
        "pMeanTrim": pressure,
        "UMeanTrim": velocity,
    }


def _pin_document(run_number: int, parts: list[bytes]) -> dict:
    case_id = f"run_{run_number}"
    boundary = b"synthetic-boundary"
    boundary_sha = _sha256(boundary)
    part_records = [
        {
            "part_index": index,
            "path": f"{case_id}/volume_{run_number}.vtu.{index:02d}.part",
            "size_bytes": len(payload),
            "lfs_sha256": _sha256(payload),
        }
        for index, payload in enumerate(parts)
    ]
    case = {
        "case_id": case_id,
        "run_number": run_number,
        "boundary": {
            "path": f"{case_id}/boundary_{run_number}.vtp",
            "size_bytes": len(boundary),
            "lfs_sha256": boundary_sha,
        },
        "surface_cell_area": {
            "path": f"{case_id}/boundary_cell_area_{run_number}.npy",
            "size_bytes": 132,
            "lfs_sha256": _sha256(b"synthetic-area"),
            "dtype": "<f4",
            "element_count": 1,
            "source_boundary_sha256": boundary_sha,
        },
        "volume": {
            "logical_path_after_assembly": f"{case_id}/volume_{run_number}.vtu",
            "part_count": len(parts),
            "parts": part_records,
            "total_size_bytes": sum(map(len, parts)),
        },
    }
    return {
        "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
        "schema_version": 1,
        "repository": {
            "repo_id": "synthetic/drivaerml",
            "revision": "a" * 40,
        },
        "case_scope": {
            "case_count": 1,
            "run_number_min": run_number,
            "run_number_max": run_number,
            "unavailable_or_held_back_run_numbers": [],
        },
        "cases": [case],
        "totals": {
            "boundary_file_count": 1,
            "boundary_bytes": len(boundary),
            "surface_cell_area_file_count": 1,
            "surface_cell_area_bytes": 132,
            "logical_volume_count": 1,
            "volume_part_file_count": len(parts),
            "reconstructed_volume_bytes": sum(map(len, parts)),
        },
    }


def _materialize(root: Path, run_number: int, parts: list[bytes]) -> Path:
    run_root = root / f"run_{run_number}"
    run_root.mkdir(parents=True)
    for index, payload in enumerate(parts):
        (run_root / f"volume_{run_number}.vtu.{index:02d}.part").write_bytes(
            payload
        )
    pin_path = root / "native-source-pin.json"
    pin_path.write_text(
        json.dumps(_pin_document(run_number, parts)), encoding="utf-8"
    )
    return pin_path


class DrivAerMLNativeSourceTests(unittest.TestCase):
    def test_two_part_reader_crosses_boundaries_and_indexes_split_tag(self) -> None:
        document, payloads = _vtk_document()
        tag = b'<DataArray type="Float32" Name="pMeanTrim"'
        split = document.index(tag) + 17
        parts = [document[:split], document[split:]]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = load_native_source_pin(_materialize(root, 1, parts))
            resolved = pin.resolve("run_1", root)
            with open_verified_multipart(
                resolved, verification_chunk_size=11
            ) as stream:
                self.assertEqual(len(stream.verification), 2)
                stream.seek(split - 5)
                self.assertEqual(stream.read(13), document[split - 5 : split + 8])
                stream.seek(9)
                index = index_inline_binary_vtk_xml(
                    stream, scan_chunk_size=7, max_tag_bytes=512
                )
                self.assertEqual(stream.tell(), 9)
                self.assertEqual(index.dataset_type, "UnstructuredGrid")
                self.assertEqual(index.header_type, "UInt64")
                self.assertEqual(
                    [(piece.number_of_points, piece.number_of_cells) for piece in index.pieces],
                    [(2, 2)],
                )
                self.assertEqual(
                    [array.association for array in index.data_arrays],
                    ["FieldData", "Points", "CellData", "CellData"],
                )
                self.assertEqual(
                    [array.name for array in index.data_arrays],
                    ["TimeValue", None, "pMeanTrim", "UMeanTrim"],
                )
                self.assertEqual(index.arrays_for(name="TimeValue")[0].piece_index, -1)

                for name in ("pMeanTrim", "UMeanTrim"):
                    array = index.arrays_for(name=name)[0]
                    output = io.BytesIO()
                    summary = stream_inline_binary_payload(
                        stream,
                        index,
                        array,
                        output,
                        encoded_chunk_size=5,
                    )
                    self.assertEqual(output.getvalue(), payloads[name])
                    self.assertEqual(summary.tuple_count, 2)
                    self.assertEqual(summary.payload_sha256, _sha256(payloads[name]))
                self.assertEqual(
                    index.arrays_for(name="UMeanTrim")[0].number_of_components, 3
                )

    def test_three_part_and_monolithic_segment_views_are_equivalent(self) -> None:
        document, _ = _vtk_document()
        first = document.index(b"<Points") + 3
        second = document.index(b"UMeanTrim") + 4
        parts = [document[:first], document[first:second], document[second:]]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = load_native_source_pin(_materialize(root, 44, parts))
            resolved = pin.resolve("run_44", root)
            monolithic = root / "reconstructed-run-44.vtu"
            monolithic.write_bytes(document)

            with open_verified_multipart(
                resolved, verification_chunk_size=13
            ) as multipart, open_verified_monolithic(
                resolved, monolithic, verification_chunk_size=13
            ) as assembled:
                self.assertEqual(len(multipart.segments), 3)
                self.assertEqual(len(assembled.verification), 3)
                for offset, size in ((0, 19), (first - 2, 11), (second - 3, 17)):
                    multipart.seek(offset)
                    assembled.seek(offset)
                    self.assertEqual(multipart.read(size), assembled.read(size))
                assembled.seek(-12, io.SEEK_END)
                self.assertEqual(assembled.read(12), document[-12:])

    def test_verified_reader_never_reopens_paths_and_detects_mutation(self) -> None:
        document, _ = _vtk_document()
        split = len(document) // 2
        parts = [document[:split], document[split:]]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = load_native_source_pin(_materialize(root, 1, parts))
            resolved = pin.resolve("run_1", root)
            with open_verified_multipart(
                resolved, verification_chunk_size=17
            ) as stream:
                # Any pathname reopen after verification would hit this guard.
                with mock.patch.object(
                    Path,
                    "open",
                    side_effect=AssertionError("verified reader reopened a path"),
                ):
                    stream.seek(split - 4)
                    self.assertEqual(
                        stream.read(12), document[split - 4 : split + 8]
                    )

            monolithic = root / "reconstructed.vtu"
            monolithic.write_bytes(document)
            with open_verified_monolithic(
                resolved, monolithic, verification_chunk_size=19
            ) as stream:
                with monolithic.open("ab") as handle:
                    handle.write(b"mutation")
                    handle.flush()
                    os.fsync(handle.fileno())
                stream.seek(0)
                with self.assertRaisesRegex(
                    NativeSourceIntegrityError, "changed before a logical-stream read"
                ):
                    stream.read(1)

    def test_rejects_malformed_part_order_and_materialized_hash(self) -> None:
        document, _ = _vtk_document()
        split = len(document) // 2
        parts = [document[:split], document[split:]]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_path = _materialize(root, 1, parts)
            malformed = json.loads(pin_path.read_text(encoding="utf-8"))
            malformed["cases"][0]["volume"]["parts"][0]["part_index"] = 1
            malformed_path = root / "malformed-order.json"
            malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(NativeSourceError, "concatenation order"):
                load_native_source_pin(malformed_path)

            wrong_hash = json.loads(pin_path.read_text(encoding="utf-8"))
            wrong_hash["cases"][0]["volume"]["parts"][1]["lfs_sha256"] = "0" * 64
            wrong_hash_path = root / "wrong-hash.json"
            wrong_hash_path.write_text(json.dumps(wrong_hash), encoding="utf-8")
            pin = load_native_source_pin(wrong_hash_path)
            with self.assertRaisesRegex(
                NativeSourceIntegrityError, "SHA-256 mismatch"
            ):
                open_verified_multipart(
                    pin.resolve("run_1", root), verification_chunk_size=7
                )

    def test_scalar_and_vector_tuple_counts_are_enforced(self) -> None:
        malformed, _ = _vtk_document(incorrect_scalar_length=True)
        with io.BytesIO(malformed) as stream:
            index = index_inline_binary_vtk_xml(stream, scan_chunk_size=9)
            pressure = index.arrays_for(name="pMeanTrim")[0]
            with self.assertRaisesRegex(
                InlineBinaryDecodeError, "declares 12 payload bytes, decoded 8"
            ):
                stream_inline_binary_payload(
                    stream, index, pressure, encoded_chunk_size=4
                )

        valid, payloads = _vtk_document()
        with io.BytesIO(valid) as stream:
            index = index_inline_binary_vtk_xml(stream, scan_chunk_size=8)
            points = index.arrays_for(association="Points")[0]
            output: list[bytes] = []
            summary = stream_inline_binary_payload(
                stream, index, points, output.append, encoded_chunk_size=7
            )
            self.assertEqual(b"".join(output), payloads["points"])
            self.assertEqual(summary.scalar_count, 6)
            self.assertEqual(summary.tuple_count, 2)
            self.assertEqual(summary.number_of_components, 3)

    def test_polydata_piece_uses_polygon_count_for_cell_data(self) -> None:
        pressure = struct.pack("<2f", 0.5, -0.5)
        document = b"".join(
            [
                b'<VTKFile type="PolyData" version="0.1" ',
                b'byte_order="LittleEndian" header_type="UInt64">',
                b"<PolyData>",
                b'<Piece NumberOfPoints="3" NumberOfPolys="2">',
                b"<CellData>",
                b'<DataArray type="Float32" Name="pMeanTrim" format="binary">',
                _inline_binary(pressure),
                b"</DataArray>",
                b"</CellData>",
                b"</Piece>",
                b"</PolyData>",
                b"</VTKFile>",
            ]
        )
        with io.BytesIO(document) as stream:
            index = index_inline_binary_vtk_xml(stream, scan_chunk_size=6)
            self.assertEqual(index.dataset_type, "PolyData")
            self.assertEqual(index.pieces[0].number_of_cells, 2)
            self.assertEqual(index.pieces[0].number_of_polys, 2)
            summary = stream_inline_binary_payload(
                stream,
                index,
                index.arrays_for(name="pMeanTrim")[0],
                encoded_chunk_size=3,
            )
            self.assertEqual(summary.tuple_count, 2)

    def test_real_public_pin_loads_with_expected_transport_shape(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pin = load_native_source_pin(
            root
            / "benchmark-specs"
            / "drivaerml"
            / "proposal"
            / "native-source-pin.json"
        )
        self.assertEqual(len(pin.cases), 484)
        self.assertEqual(len(pin.case("run_1").volume_parts), 2)
        self.assertEqual(len(pin.case("run_44").volume_parts), 3)
        self.assertEqual(
            sum(len(case.volume_parts) for case in pin.cases),
            978,
        )


if __name__ == "__main__":
    unittest.main()
