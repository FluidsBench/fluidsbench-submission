from __future__ import annotations

import base64
import copy
import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from reference.drivaerml.coordinate_identity import (
    coordinate_array_identity_sha256,
)
from scripts import export_drivaerml_native_profile_truth as exporter
from scripts import validate_drivaerml_native_profile_truth as truth_validator


def _inline_array(
    name: str,
    values: np.ndarray,
    *,
    components: int,
) -> bytes:
    payload = np.asarray(values, dtype="<f4").tobytes(order="C")
    encoded = base64.b64encode(struct.pack("<Q", len(payload)) + payload)
    return (
        b'<DataArray type="Float32" Name="'
        + name.encode("ascii")
        + b'" NumberOfComponents="'
        + str(components).encode("ascii")
        + b'" format="binary">\n'
        + encoded
        + b"\n</DataArray>"
    )


def _miniature_vtu(target: np.ndarray) -> io.BytesIO:
    preceding = np.asarray([91.0, 92.0, 93.0, 94.0], dtype=np.float32)
    payload = b"".join(
        (
            b'<?xml version="1.0"?>\n',
            b'<VTKFile type="UnstructuredGrid" version="1.0" '
            b'byte_order="LittleEndian" header_type="UInt64">\n',
            b"<UnstructuredGrid>\n",
            b'<Piece NumberOfPoints="0" NumberOfCells="5">\n',
            b"<CellData>\n",
            _inline_array("preceding", preceding, components=1),
            b"\n",
            _inline_array("UMeanTrim", target, components=3),
            b"\n</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n",
        )
    )
    return io.BytesIO(payload)


def _miniature_vtp(target: np.ndarray) -> io.BytesIO:
    preceding = np.asarray([[1.0, 2.0, 3.0]] * 5, dtype=np.float32)
    payload = b"".join(
        (
            b'<?xml version="1.0"?>\n',
            b'<VTKFile type="PolyData" version="1.0" '
            b'byte_order="LittleEndian" header_type="UInt64">\n',
            b"<PolyData>\n",
            b'<Piece NumberOfPoints="8" NumberOfPolys="5">\n',
            b"<CellData>\n",
            _inline_array("preceding", preceding, components=3),
            b"\n",
            _inline_array("pMeanTrim", target, components=1),
            b"\n</CellData>\n</Piece>\n</PolyData>\n</VTKFile>\n",
        )
    )
    return io.BytesIO(payload)


class DrivAerMLNativeProfileTruthExporterTests(unittest.TestCase):
    def test_all_40_prediction_series_are_cross_bound_to_truth_coordinates(self) -> None:
        prediction_document, _ = exporter.load_json(
            truth_validator.PREDICTION_FIXTURE, "current run_419 prediction fixture"
        )
        prediction_case = prediction_document["cases"][0]
        generated_case = copy.deepcopy(prediction_case)
        for item in generated_case["series"]:
            if item["representation"] == "materialized":
                item["value"] = item.pop("prediction")
        truth_validator.cross_bind_all_prediction_series(
            generated_case, prediction_case
        )

        mutations = (
            (38, "coordinate", 0, 1234.0),
            (16, "coordinate", 0, 1234.0),
            (16, "coordinate_id", None, "wrong_coordinate"),
            (38, "support_identity_sha256", None, "f" * 64),
        )
        for series_index, field, element, replacement in mutations:
            with self.subTest(series_index=series_index, field=field):
                mutated = copy.deepcopy(generated_case)
                if element is None:
                    mutated["series"][series_index][field] = replacement
                else:
                    mutated["series"][series_index][field][element] = replacement
                with self.assertRaisesRegex(
                    truth_validator.TruthValidationError,
                    "support/coordinate/prediction binding differs",
                ):
                    truth_validator.cross_bind_all_prediction_series(
                        mutated, prediction_case
                    )

    def test_component_identity_encodings_are_platform_independent(self) -> None:
        values = [-0.0, 1.25, -3.5]
        self.assertEqual(
            exporter.binary64_array_identity_sha256(values),
            "5e6c753ac5402959bce2ed778fbd3c46ac2b206bfb0e32bea8d9c4454af57719",
        )
        self.assertEqual(
            exporter.integer_array_identity_sha256([0, 2, 2**63]),
            "a5a509ee25bd146bf330cac8c9aa09b2efa680351cf639b899e377e2c9c718d5",
        )
        self.assertEqual(
            coordinate_array_identity_sha256(values),
            "f971e8e435f0cbcb4820ad42f57bf5c20d068befcdf1953e9479f8151b967ca0",
        )
        self.assertEqual(
            exporter.binary64_array_identity_sha256(values),
            exporter.binary64_array_identity_sha256([0.0, 1.25, -3.5]),
        )
        with self.assertRaisesRegex(exporter.ExportError, "must be numeric"):
            exporter.binary64_array_identity_sha256([True])
        with self.assertRaisesRegex(exporter.ExportError, "non-negative"):
            exporter.integer_array_identity_sha256([-1])

    def test_compact_segments_preserve_gaps_and_interval_labels(self) -> None:
        segments = exporter.compact_segments(
            [0, 1, 3, 4, 5],
            [0.0, 0.1, 0.3, 0.4, 0.5],
            labels=["a", "a", "b", "b", "c"],
        )

        self.assertEqual(
            segments,
            [
                {
                    "segment_id": "a",
                    "emitted_index_start": 0,
                    "emitted_index_stop": 2,
                    "sample_index_start": 0,
                    "sample_index_stop": 2,
                    "coordinate_start": 0.0,
                    "coordinate_stop": 0.1,
                },
                {
                    "segment_id": "b",
                    "emitted_index_start": 2,
                    "emitted_index_stop": 4,
                    "sample_index_start": 3,
                    "sample_index_stop": 5,
                    "coordinate_start": 0.3,
                    "coordinate_stop": 0.4,
                },
                {
                    "segment_id": "c",
                    "emitted_index_start": 4,
                    "emitted_index_stop": 5,
                    "sample_index_start": 5,
                    "sample_index_stop": 6,
                    "coordinate_start": 0.5,
                    "coordinate_stop": 0.5,
                },
            ],
        )

    def test_coverage_summary_is_derived_and_expected_totals_are_enforced(self) -> None:
        family = "test-family"
        documents = [
            {
                "series": [
                    {
                        "family_id": family,
                        "representation": "materialized",
                        "sample_index": [0, 2],
                        "unsupported_samples": [
                            {"sample_index": 1, "coordinate": 0.5, "reason": "gap"}
                        ],
                        "segments": [{"segment_id": "a"}, {"segment_id": "b"}],
                    },
                    {"family_id": family, "representation": "shared_alias"},
                ]
            }
        ]
        expected = {
            family: {
                "materialized_series_count": 1,
                "shared_alias_series_count": 1,
                "sample_count": 2,
                "unsupported_sample_count": 1,
            }
        }
        with mock.patch.object(exporter, "EXPECTED_ALL484_COVERAGE", expected):
            summary = exporter._all484_coverage_summary(documents)
            self.assertEqual(summary["families"][family]["segment_count"], 2)
            self.assertEqual(summary["totals"]["requested_sample_count"], 3)
            wrong = copy.deepcopy(expected)
            wrong[family]["sample_count"] = 3
            with mock.patch.object(exporter, "EXPECTED_ALL484_COVERAGE", wrong):
                with self.assertRaisesRegex(exporter.ExportError, "sample_count differs"):
                    exporter._all484_coverage_summary(documents)

    def test_materialized_series_binds_every_array_and_gap(self) -> None:
        series = exporter.materialized_series(
            panel_id="velocity_profiles",
            family_id=exporter.RELATIVE_VELOCITY_FAMILY,
            placement_mode="relative",
            station_id="V1",
            quantity_id="velocity_ratio",
            quantity="velocity_magnitude_ratio",
            units="1",
            scoring_role="report_only",
            support_identity_sha256="1" * 64,
            placement_receipt_identity_sha256="2" * 64,
            coordinate_id="normalized_arc_length",
            coordinate_unit="1",
            sample_index=[0, 2],
            raw_native_cell_id=[7, 11],
            coordinate=[0.0, 1.0],
            value=[0.5, 1.5],
            segments=exporter.compact_segments([0, 2], [0.0, 1.0]),
            unsupported_samples=[
                {"sample_index": 1, "coordinate": 0.5, "reason": "no_native_cell"}
            ],
        )
        supplied = series["series_identity_sha256"]
        body = dict(series)
        body.pop("series_identity_sha256")
        self.assertEqual(supplied, exporter.series_identity_sha256(body))
        self.assertEqual(
            series["coordinate_identity_sha256"],
            coordinate_array_identity_sha256([0.0, 1.0]),
        )
        self.assertEqual(
            series["value_identity_sha256"],
            exporter.binary64_array_identity_sha256([0.5, 1.5]),
        )

        for field, replacement in (
            ("sample_index", [0, 3]),
            ("raw_native_cell_id", [7, 12]),
            ("coordinate_identity_sha256", "3" * 64),
            ("value_identity_sha256", "4" * 64),
            ("unsupported_samples", []),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(body)
                mutated[field] = replacement
                self.assertNotEqual(supplied, exporter.series_identity_sha256(mutated))

    def test_materialized_cp_binds_aligned_physical_x_for_display_only(self) -> None:
        series = exporter.materialized_series(
            panel_id="pressure_profiles",
            family_id=exporter.CONSTANT_CP_FAMILY,
            placement_mode="constant",
            station_id="upperbody_centerline",
            quantity_id="cp",
            quantity="pressure_coefficient",
            units="1",
            scoring_role="inherits_parent_candidate",
            support_identity_sha256="1" * 64,
            placement_receipt_identity_sha256="2" * 64,
            coordinate_id="arc_length_m",
            coordinate_unit="m",
            sample_index=[0, 1, 2],
            raw_native_cell_id=[7, 11, 13],
            coordinate=[0.1, 0.2, 0.3],
            value=[-0.2, 0.1, 0.4],
            segments=exporter.compact_segments(
                [0, 1, 2], [0.1, 0.2, 0.3], labels=["cut"] * 3
            ),
            unsupported_samples=[],
            display_coordinate_id=exporter.CP_DISPLAY_COORDINATE_ID,
            display_coordinate_unit=exporter.CP_DISPLAY_COORDINATE_UNIT,
            display_coordinate=[-0.0, 1.25, -3.5],
        )
        self.assertEqual(series["coordinate"], [0.1, 0.2, 0.3])
        self.assertEqual(
            series["display_coordinate_identity_sha256"],
            "f971e8e435f0cbcb4820ad42f57bf5c20d068befcdf1953e9479f8151b967ca0",
        )
        body = dict(series)
        supplied = body.pop("series_identity_sha256")
        self.assertEqual(supplied, exporter.series_identity_sha256(body))

        stale = copy.deepcopy(body)
        stale["display_coordinate"][1] = 1234.0
        self.assertEqual(supplied, exporter.series_identity_sha256(stale))
        self.assertNotEqual(
            stale["display_coordinate_identity_sha256"],
            coordinate_array_identity_sha256(stale["display_coordinate"]),
        )
        rebound = copy.deepcopy(stale)
        rebound["display_coordinate_identity_sha256"] = coordinate_array_identity_sha256(
            rebound["display_coordinate"]
        )
        self.assertNotEqual(supplied, exporter.series_identity_sha256(rebound))

        common = dict(
            panel_id="pressure_profiles",
            family_id=exporter.CONSTANT_CP_FAMILY,
            placement_mode="constant",
            station_id="upperbody_centerline",
            quantity_id="cp",
            quantity="pressure_coefficient",
            units="1",
            scoring_role="inherits_parent_candidate",
            support_identity_sha256="1" * 64,
            placement_receipt_identity_sha256="2" * 64,
            coordinate_id="arc_length_m",
            coordinate_unit="m",
            sample_index=[0, 1],
            raw_native_cell_id=[7, 11],
            coordinate=[0.1, 0.2],
            value=[-0.2, 0.1],
            segments=exporter.compact_segments([0, 1], [0.1, 0.2]),
            unsupported_samples=[],
        )
        with self.assertRaisesRegex(exporter.ExportError, "supplied together"):
            exporter.materialized_series(
                **common,
                display_coordinate_id=exporter.CP_DISPLAY_COORDINATE_ID,
            )
        with self.assertRaisesRegex(exporter.ExportError, "align"):
            exporter.materialized_series(
                **common,
                display_coordinate_id=exporter.CP_DISPLAY_COORDINATE_ID,
                display_coordinate_unit="m",
                display_coordinate=[0.0],
            )
        with self.assertRaisesRegex(exporter.ExportError, "finite"):
            exporter.materialized_series(
                **common,
                display_coordinate_id=exporter.CP_DISPLAY_COORDINATE_ID,
                display_coordinate_unit="m",
                display_coordinate=[0.0, float("nan")],
            )

    def test_cp_display_coordinate_uses_retained_segment_midpoint_x(self) -> None:
        row = {
            "endpoint_start_m": [-0.75, 0.0, 0.2],
            "endpoint_end_m": [-0.25, 0.0, 0.2],
            "length_m": 0.5,
        }
        self.assertEqual(exporter.cp_display_coordinate_x(row, "test row"), -0.5)
        for mutation, message in (
            ({"endpoint_start_m": [-0.75, 0.0]}, "three coordinates"),
            ({"endpoint_end_m": [-0.25, float("inf"), 0.2]}, "finite"),
            ({"length_m": 0.6}, "does not replay"),
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(exporter.ExportError, message):
                    exporter.cp_display_coordinate_x({**row, **mutation}, "test row")

    def test_alias_uses_only_the_exact_shared_support_reference(self) -> None:
        shared = {
            "canonical_family_id": exporter.CONSTANT_CP_FAMILY,
            "canonical_station_id": "upperbody_centerline",
            "canonical_support_identity_sha256": "3" * 64,
            "shared_support_id": "upperbody_centerline",
        }
        alias = exporter.alias_series(
            station_id="upperbody_centerline",
            placement_receipt_identity_sha256="4" * 64,
            shared_support_ref=shared,
        )
        self.assertEqual(alias["representation"], "shared_alias")
        self.assertEqual(alias["shared_support_ref"], shared)
        self.assertNotIn("coordinate", alias)
        self.assertNotIn("display_coordinate", alias)
        self.assertNotIn("value", alias)
        body = dict(alias)
        supplied = body.pop("series_identity_sha256")
        self.assertEqual(supplied, exporter.series_identity_sha256(body))
        malformed_body = copy.deepcopy(body)
        malformed_body["shared_support_ref"]["display_coordinate"] = [1234.0]
        with self.assertRaisesRegex(exporter.ExportError, "reference shape differs"):
            exporter.series_identity_sha256(malformed_body)
        with self.assertRaisesRegex(exporter.ExportError, "shape differs"):
            exporter.alias_series(
                station_id="upperbody_centerline",
                placement_receipt_identity_sha256="4" * 64,
                shared_support_ref={**shared, "unexpected": True},
            )

    def test_run419_display_validator_rejects_rebound_interior_coordinate(self) -> None:
        original = [-0.8, 0.25, 3.6]
        original_identity = coordinate_array_identity_sha256(original)
        with mock.patch.object(
            truth_validator,
            "RUN419_CP_DISPLAY_IDENTITY",
            {"test_station": {"count": 3, "sha256": original_identity}},
        ):
            truth_validator._validate_run419_cp_display(
                "test_station", original, original_identity, "test series"
            )
            mutated = [-0.8, 1234.0, 3.6]
            rebound_identity = coordinate_array_identity_sha256(mutated)
            with self.assertRaisesRegex(
                truth_validator.TruthValidationError,
                "exact retained producer midpoint array",
            ):
                truth_validator._validate_run419_cp_display(
                    "test_station", mutated, rebound_identity, "test series"
                )

    def test_direct_seek_reader_skips_prior_inline_array_and_gathers_exact_cells(self) -> None:
        target = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            dtype=np.float32,
        )
        stream = _miniature_vtu(target)
        located = exporter.locate_uncompressed_cell_field(
            stream,
            field_name="UMeanTrim",
            expected_components=3,
            expected_tuple_count=5,
        )
        self.assertEqual(located.declared_payload_bytes, 5 * 3 * 4)
        ids, values = exporter.direct_sparse_float32x3(
            stream, located, [4, 0, 2, 2]
        )
        np.testing.assert_array_equal(ids, np.asarray([0, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(values, target[[0, 2, 4]].astype(np.float64))

        expected = hashlib.sha256()
        expected.update(
            b'{"ids_shape":[3],"values_shape":[3,3]}'
        )
        expected.update(np.asarray([0, 2, 4], dtype="<i8").tobytes())
        expected.update(target[[0, 2, 4]].astype("<f8").tobytes())
        self.assertEqual(
            exporter.selected_array_evidence_sha256(ids, values),
            expected.hexdigest(),
        )
        with self.assertRaisesRegex(exporter.ExportError, "exceed"):
            exporter.direct_sparse_float32x3(stream, located, [5])

    def test_direct_seek_reader_gathers_native_polydata_scalar_by_polygon_id(self) -> None:
        target = np.asarray(
            [-1387.5, -0.0, 12.25, 759.4375, 1.0], dtype=np.float32
        )
        stream = _miniature_vtp(target)
        located = exporter.locate_uncompressed_cell_field(
            stream,
            field_name="pMeanTrim",
            expected_components=1,
            expected_tuple_count=None,
            expected_dataset_type="PolyData",
            piece_tuple_attribute="NumberOfPolys",
        )
        self.assertEqual(located.tuple_count, 5)
        self.assertEqual(located.declared_payload_bytes, 5 * 4)
        ids, values = exporter.direct_sparse_float32x1(
            stream, located, [3, 0, 3, 2]
        )
        np.testing.assert_array_equal(ids, np.asarray([0, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(values, target[[0, 2, 3]].astype(np.float64))
        with self.assertRaisesRegex(exporter.ExportError, "exceed"):
            exporter.direct_sparse_float32x1(stream, located, [5])

    def test_identity_bound_documents_replay_and_detect_mutation(self) -> None:
        document = exporter.identity_bound_document(
            {"schema": "example", "case_count": 484}, "document_identity"
        )
        supplied = document["document_identity"]["sha256"]
        self.assertEqual(
            exporter.document_identity_sha256(document, "document_identity"),
            supplied,
        )
        mutated = copy.deepcopy(document)
        mutated["case_count"] = 483
        with self.assertRaisesRegex(exporter.ExportError, "does not replay"):
            exporter.document_identity_sha256(mutated, "document_identity")

    def test_assembler_requires_the_full_set_and_replays_byte_for_byte(self) -> None:
        official = tuple(f"run_{index}" for index in range(1, 485))
        split_documents = {
            split_id: {
                "case_set_id": f"drivaerml-{split_id}-v1",
                "case_ids": list(official[offset:: len(exporter.EXPECTED_SPLITS)]),
            }
            for offset, split_id in enumerate(exporter.EXPECTED_SPLITS)
        }
        split_bindings = {
            split_id: {
                "path": f"{split_id}.json",
                "sha256": hashlib.sha256(split_id.encode("ascii")).hexdigest(),
                "size_bytes": len(split_id),
            }
            for split_id in exporter.EXPECTED_SPLITS
        }

        def fake_case(
            context: exporter.InputContext, case_id: str
        ) -> tuple[dict[str, object], dict[str, object]]:
            document: dict[str, object] = {
                "case_id": case_id,
                "input_bindings": {"test": case_id},
                "native_volume": {
                    "logical_path": f"DrivAerML/volume/{case_id}.vtu",
                    "logical_size_bytes": 100,
                    "retained_full_payload_sha256": hashlib.sha256(
                        f"payload:{case_id}".encode("ascii")
                    ).hexdigest(),
                    "selected_unique_raw_cell_id_count": 3,
                    "selected_values_sha256": hashlib.sha256(
                        f"selected:{case_id}".encode("ascii")
                    ).hexdigest(),
                },
                "native_boundary": {
                    "logical_path": f"{case_id}/boundary.vtp",
                    "logical_size_bytes": 200,
                    "source_sha256": hashlib.sha256(
                        f"boundary:{case_id}".encode("ascii")
                    ).hexdigest(),
                    "tuple_count": 9,
                    "referenced_producer_row_count": 5,
                    "selected_unique_raw_polygon_id_count": 3,
                    "selected_values_sha256": hashlib.sha256(
                        f"pressure:{case_id}".encode("ascii")
                    ).hexdigest(),
                },
                "series": [],
            }
            return document, {
                "path": f"cases/{case_id}.json",
                "sha256": hashlib.sha256(case_id.encode("ascii")).hexdigest(),
                "size_bytes": len(case_id),
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            context = exporter.InputContext(
                native_source_pin=output / "pin.json",
                dataset_root=output,
                constant_velocity_receipts_root=output,
                relative_producer_root=output,
                constant_cp_campaign_root=output,
                native_volume_audit=output / "audit.json",
                output_dir=output,
                evaluator_git_revision="a" * 40,
                official_case_ids=official,
                pin_sha256="b" * 64,
                pin_by_case={},
                native_audit_by_case={},
                relative_placement_by_case={},
                relative_mapping_by_case={},
                relative_cp_by_case={},
                constant_cp_by_case={},
                global_bindings={"test": {"sha256": "c" * 64}},
                split_indexes=split_documents,
                split_bindings=split_bindings,
            )
            test_coverage = {
                "derivation": "test",
                "producer_expected_counts_verified": True,
                "families": {},
                "totals": {},
            }
            with (
                mock.patch.object(
                    exporter, "_load_case_artifact", side_effect=fake_case
                ),
                mock.patch.object(
                    exporter,
                    "_all484_coverage_summary",
                    return_value=test_coverage,
                ),
            ):
                first = exporter.assemble(context, cases_per_chunk=8, check=False)
                second = exporter.assemble(context, cases_per_chunk=8, check=True)
            self.assertEqual(first, second)
            self.assertEqual(first["case_count"], 484)
            self.assertEqual(first["chunk_count"], 61)
            self.assertFalse(first["relative_scoring_activated"])

            index, _ = exporter.load_json(output / "index.json", "test index")
            chunk, _ = exporter.load_json(
                output / "chunks" / "chunk-000.json", "test chunk"
            )
            provenance, _ = exporter.load_json(
                output / "provenance.json", "test provenance"
            )
            split, _ = exporter.load_json(
                output / "splits" / "full.json", "test split"
            )
            release, _ = exporter.load_json(
                output / "release-receipt.json", "test release"
            )
            exporter.document_identity_sha256(index, "index_identity")
            exporter.document_identity_sha256(chunk, "chunk_identity")
            exporter.document_identity_sha256(provenance, "provenance_identity")
            exporter.document_identity_sha256(split, "split_identity")
            exporter.document_identity_sha256(release, "release_identity")
            self.assertEqual(index["case_count"], 484)
            self.assertEqual(index["series_count"], 484 * 40)
            self.assertEqual(len(index["chunks"]), 61)
            self.assertEqual(len(index["splits"]), 8)
            self.assertEqual(index["truth_source"], exporter.TRUTH_SOURCE)
            self.assertEqual(chunk["truth_source"], exporter.TRUTH_SOURCE)
            self.assertEqual(provenance["truth_source"], exporter.TRUTH_SOURCE)
            self.assertEqual(split["truth_source"], exporter.TRUTH_SOURCE)
            self.assertFalse(index["relative_scoring_activated"])
            self.assertEqual(release["truth_source"], exporter.TRUTH_SOURCE)
            self.assertFalse(release["relative_scoring_activated"])
            self.assertFalse(release["submissions_opened"])
            self.assertFalse(release["owner_approval_complete"])


if __name__ == "__main__":
    unittest.main()
