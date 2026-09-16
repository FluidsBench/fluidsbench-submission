from __future__ import annotations

import base64
import io
import struct
import unittest

import numpy as np

from reference.drivaerml.native_fields import (
    NativeFieldAuditError,
    audit_and_compare_inline_native_cell_data,
    audit_required_volume_cell_data,
    evaluate_inline_native_cell_data,
)
from reference.drivaerml.source import index_inline_binary_vtk_xml


def _encoded(payload: bytes) -> bytes:
    return base64.b64encode(struct.pack("<Q", len(payload)) + payload)


def _volume(*, nonfinite: bool = False) -> bytes:
    pressure = struct.pack("<2f", 1.5, float("nan") if nonfinite else -2.0)
    velocity = struct.pack("<6f", 1, 2, 3, 4, 5, 6)
    return b"".join(
        [
            b'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt64">',
            b'<UnstructuredGrid><Piece NumberOfPoints="1" NumberOfCells="2"><CellData>',
            b'<DataArray type="Float32" Name="pMeanTrim" format="binary">',
            _encoded(pressure),
            b'</DataArray>',
            b'<DataArray type="Float32" Name="UMeanTrim" NumberOfComponents="3" format="binary">',
            _encoded(velocity),
            b'</DataArray></CellData></Piece></UnstructuredGrid></VTKFile>',
        ]
    )


class DrivAerMLNativeFieldTests(unittest.TestCase):
    def test_scalar_vector_counts_ranges_and_hashes(self) -> None:
        stream = io.BytesIO(_volume())
        index = index_inline_binary_vtk_xml(stream, scan_chunk_size=11)
        audits = audit_required_volume_cell_data(
            stream, index, encoded_chunk_size=5
        )
        self.assertEqual(audits["pMeanTrim"].tuple_count, 2)
        self.assertEqual(audits["pMeanTrim"].minimum_by_component, (-2.0,))
        self.assertEqual(audits["pMeanTrim"].maximum_by_component, (1.5,))
        self.assertEqual(audits["UMeanTrim"].number_of_components, 3)
        self.assertEqual(audits["UMeanTrim"].minimum_by_component, (1.0, 2.0, 3.0))
        self.assertEqual(audits["UMeanTrim"].maximum_by_component, (4.0, 5.0, 6.0))
        self.assertEqual(audits["UMeanTrim"].raw_id_stop, 2)
        self.assertTrue(audits["UMeanTrim"].finite)

    def test_nonfinite_and_wrong_association_fail(self) -> None:
        stream = io.BytesIO(_volume(nonfinite=True))
        index = index_inline_binary_vtk_xml(stream)
        with self.assertRaisesRegex(NativeFieldAuditError, "non-finite"):
            audit_required_volume_cell_data(stream, index)

        malformed = _volume().replace(b"<CellData>", b"<PointData>").replace(
            b"</CellData>", b"</PointData>"
        )
        stream = io.BytesIO(malformed)
        index = index_inline_binary_vtk_xml(stream)
        with self.assertRaisesRegex(NativeFieldAuditError, "exactly one CellData"):
            audit_required_volume_cell_data(stream, index)

    def test_materialized_full_case_matches_bounded_native_chunks(self) -> None:
        stream = io.BytesIO(_volume())
        index = index_inline_binary_vtk_xml(stream, scan_chunk_size=7)
        pressure = index.arrays_for(association="CellData", name="pMeanTrim")[0]
        prediction = np.asarray([1.0, -1.0], dtype=np.float32)
        weights = np.asarray([2.0, 5.0], dtype=np.float64)
        full = evaluate_inline_native_cell_data(
            stream,
            index,
            pressure,
            predictions=prediction,
            physical_weights=weights,
            chunk_entities=2,
            encoded_chunk_size=3,
        )
        chunked = evaluate_inline_native_cell_data(
            stream,
            index,
            pressure,
            predictions=prediction,
            physical_weights=weights,
            chunk_entities=1,
            encoded_chunk_size=5,
        )
        self.assertEqual(full.source_payload.payload_sha256, chunked.source_payload.payload_sha256)
        self.assertEqual(full.statistics.metric_values(), chunked.statistics.metric_values())
        self.assertEqual(full.to_json()["entity_count"], 2)
        self.assertEqual(full.to_json()["component_count"], 1)

    def test_one_decode_broadcasts_audit_and_independent_partitions(self) -> None:
        stream = io.BytesIO(_volume())
        index = index_inline_binary_vtk_xml(stream, scan_chunk_size=7)
        velocity = index.arrays_for(association="CellData", name="UMeanTrim")[0]
        prediction = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        weights = np.asarray([2.0, 5.0], dtype=np.float64)
        audit, full, chunked = audit_and_compare_inline_native_cell_data(
            stream,
            index,
            velocity,
            units="m/s",
            predictions=prediction,
            physical_weights=weights,
            reference_chunk_entities=2,
            comparison_chunk_entities=1,
            encoded_chunk_size=3,
        )
        self.assertEqual(audit.tuple_count, 2)
        self.assertEqual(audit.payload_sha256, full.source_payload.payload_sha256)
        self.assertEqual(audit.payload_sha256, chunked.source_payload.payload_sha256)
        self.assertEqual(full.statistics.metric_values(), chunked.statistics.metric_values())
        self.assertIsNot(full.statistics, chunked.statistics)

        with self.assertRaisesRegex(NativeFieldAuditError, "distinct positive"):
            audit_and_compare_inline_native_cell_data(
                stream,
                index,
                velocity,
                units="m/s",
                predictions=prediction,
                physical_weights=weights,
                reference_chunk_entities=1,
                comparison_chunk_entities=1,
            )

    def test_metric_shape_weight_and_association_fail_closed(self) -> None:
        stream = io.BytesIO(_volume())
        index = index_inline_binary_vtk_xml(stream)
        pressure = index.arrays_for(name="pMeanTrim")[0]
        with self.assertRaisesRegex(NativeFieldAuditError, "prediction shape"):
            evaluate_inline_native_cell_data(
                stream,
                index,
                pressure,
                predictions=np.zeros(3),
                physical_weights=np.ones(2),
                chunk_entities=1,
            )
        with self.assertRaisesRegex(NativeFieldAuditError, "strictly positive"):
            evaluate_inline_native_cell_data(
                stream,
                index,
                pressure,
                predictions=np.zeros(2),
                physical_weights=np.asarray([1.0, 0.0]),
                chunk_entities=1,
            )


if __name__ == "__main__":
    unittest.main()
