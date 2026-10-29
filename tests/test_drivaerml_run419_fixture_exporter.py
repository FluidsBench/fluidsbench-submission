from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import vtk  # noqa: F401
except ImportError:
    raise unittest.SkipTest(
        "requires optional VTK for DrivAerML Run419 fixture export"
    ) from None

from scripts import evaluate_drivaerml_candidate_case as candidate_cli
from scripts import export_drivaerml_run419_relative_fixture as exporter


class _PinStub:
    def case(self, case_id: str) -> object:
        if case_id != exporter.CASE_ID:
            raise AssertionError(case_id)
        return SimpleNamespace(case_id=case_id)


def _prediction_record(support_id: str, count: int, marker: str) -> dict[str, object]:
    digest = marker * 64
    return {
        "manifest_file": "manifest.json",
        "manifest_sha256": digest,
        "support_id": support_id,
        "field_components": {},
        "entity_count": count,
        "chunk_count": 1,
        "chunks": [
            {
                "chunk_index": 0,
                "file": "chunks/chunk-00000.npz",
                "sha256": digest,
                "row_count": count,
                "raw_cell_id_start": 0,
                "raw_cell_id_stop": count,
            }
        ],
        "coverage": {
            "raw_cell_id_start": 0,
            "raw_cell_id_stop": count,
            "complete_gap_free_duplicate_free": True,
        },
    }


def _minimal_evidence(
    surface: dict[str, object], volume: dict[str, object]
) -> dict[str, object]:
    return {
        "schema": "drivaerml-candidate-case-evaluation-v2",
        "schema_version": 2,
        "status": "candidate_evaluator_evidence_not_official_submission",
        "official_submission": False,
        "case_id": exporter.CASE_ID,
        "source": {
            "native_source_pin_sha256": exporter.NATIVE_SOURCE_PIN_SHA256,
            "repository_id": exporter.DATASET_REPOSITORY,
            "repository_revision": exporter.DATASET_REVISION,
            "boundary_sha256": exporter.BOUNDARY_SHA256,
            "surface_native": {
                "source_file": "boundary_419.vtp",
                "boundary_sha256": exporter.BOUNDARY_SHA256,
                "vtk_version": "9.5.2",
                "point_count": 1,
                "polygon_count": exporter.SURFACE_ENTITY_COUNT,
                "raw_cell_order": "zero_based_native_vtk_polygon_order_unchanged",
                "association": "CellData",
                "arrays": {},
                "available_point_arrays": [],
                "available_cell_arrays": [],
            },
            "surface_area": {
                "source_path": "boundary_cell_area_419.npy",
                "sha256": "c" * 64,
                "source_boundary_sha256": exporter.BOUNDARY_SHA256,
                "entity_count": exporter.SURFACE_ENTITY_COUNT,
                "area_sum_m2": 1.0,
                "area_min_m2": 1.0,
                "area_max_m2": 1.0,
                "dtype": "<f4",
                "role": "fixed_external_input_not_regenerated",
                "native_geometry_order_audit": {},
            },
            "volume_part_sha256": list(exporter.VOLUME_PART_SHA256),
            "volume_vtk": {},
            "volume_weighting": {},
            "volume_native_arrays": {},
        },
        "prediction_inputs": {
            "surface_native_cells": {
                "manifest_sha256": surface["manifest_sha256"],
                "chunk_sha256": [surface["chunks"][0]["sha256"]],
                "chunk_count": 1,
                "entity_count": exporter.SURFACE_ENTITY_COUNT,
            },
            "volume_native_cells": {
                "manifest_sha256": volume["manifest_sha256"],
                "chunk_sha256": [volume["chunks"][0]["sha256"]],
                "chunk_count": 1,
                "entity_count": exporter.VOLUME_ENTITY_COUNT,
            },
        },
        "coverage": {
            "surface": {
                "raw_cell_id_start": 0,
                "raw_cell_id_stop": exporter.SURFACE_ENTITY_COUNT,
                "complete_gap_free_duplicate_free": True,
            },
            "volume": {
                "raw_cell_id_start": 0,
                "raw_cell_id_stop": exporter.VOLUME_ENTITY_COUNT,
                "complete_gap_free_duplicate_free": True,
            },
        },
        "metric_values": {},
        "metric_sufficient_statistics": {},
        "additive_sums": {},
        "force_coefficients": {},
        "execution": {
            "maximum_prediction_chunk_rows": 1_000_000,
            "hash_chunk_bytes": 16_777_216,
            "validation_block_rows": 1_000_000,
            "encoded_chunk_bytes": 16_777_216,
        },
    }


def _write_json(path: Path, value: object) -> tuple[str, int]:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


class DrivAerMLRun419FixtureExporterTests(unittest.TestCase):
    def test_core_evaluator_receipt_is_deterministic_path_free_and_exclusive(self) -> None:
        implementation = (
            {
                "path": candidate_cli.CORE_EVALUATOR_GIT_PATHS[0],
                "sha256": "a" * 64,
                "size_bytes": 123,
            },
        )
        evidence = {
            "file": "run419-evidence.json",
            "sha256": "b" * 64,
            "size_bytes": 456,
            "schema": "drivaerml-candidate-case-evaluation-v2",
            "schema_version": 2,
        }
        runtime = {"python": "3.12.13", "numpy": "2.2.6", "vtk": "9.5.2"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            one = candidate_cli._write_implementation_receipt(
                first,
                case_id=exporter.CASE_ID,
                commit="c" * 40,
                implementation=implementation,
                evidence=evidence,
                runtime=runtime,
            )
            two = candidate_cli._write_implementation_receipt(
                second,
                case_id=exporter.CASE_ID,
                commit="c" * 40,
                implementation=implementation,
                evidence=evidence,
                runtime=runtime,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(one["sha256"], two["sha256"])
            document = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], candidate_cli.IMPLEMENTATION_RECEIPT_SCHEMA)
            self.assertFalse(document["official_submission"])
            self.assertNotIn(str(root), first.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(
                candidate_cli.CandidateImplementationBindingError,
                "refusing to overwrite",
            ):
                candidate_cli._write_implementation_receipt(
                    first,
                    case_id=exporter.CASE_ID,
                    commit="c" * 40,
                    implementation=implementation,
                    evidence=evidence,
                    runtime=runtime,
                )

    def test_evaluator_requires_revision_and_receipt_as_a_pair(self) -> None:
        args = SimpleNamespace(
            evaluator_git_revision="a" * 40,
            implementation_receipt=None,
        )
        with self.assertRaisesRegex(
            candidate_cli.CandidateImplementationBindingError,
            "must be supplied together",
        ):
            candidate_cli.run(args)  # type: ignore[arg-type]

    def test_skeletal_current_case_evidence_is_rejected_before_receipt(self) -> None:
        surface = _prediction_record(
            "surface_native_cells", exporter.SURFACE_ENTITY_COUNT, "a"
        )
        volume = _prediction_record(
            "volume_native_cells", exporter.VOLUME_ENTITY_COUNT, "b"
        )
        skeletal = {
            "schema": "drivaerml-candidate-case-evaluation-v2",
            "schema_version": 2,
            "status": "candidate_evaluator_evidence_not_official_submission",
            "official_submission": False,
            "case_id": exporter.CASE_ID,
            "source": {
                "native_source_pin_sha256": exporter.NATIVE_SOURCE_PIN_SHA256,
                "repository_id": exporter.DATASET_REPOSITORY,
                "repository_revision": exporter.DATASET_REVISION,
                "boundary_sha256": exporter.BOUNDARY_SHA256,
                "volume_part_sha256": list(exporter.VOLUME_PART_SHA256),
            },
            "prediction_inputs": {
                "surface_native_cells": {
                    "manifest_sha256": surface["manifest_sha256"],
                    "chunk_sha256": [surface["chunks"][0]["sha256"]],
                    "chunk_count": 1,
                    "entity_count": exporter.SURFACE_ENTITY_COUNT,
                },
                "volume_native_cells": {
                    "manifest_sha256": volume["manifest_sha256"],
                    "chunk_sha256": [volume["chunks"][0]["sha256"]],
                    "chunk_count": 1,
                    "entity_count": exporter.VOLUME_ENTITY_COUNT,
                },
            },
            "coverage": {
                "surface": {
                    "raw_cell_id_start": 0,
                    "raw_cell_id_stop": exporter.SURFACE_ENTITY_COUNT,
                    "complete_gap_free_duplicate_free": True,
                },
                "volume": {
                    "raw_cell_id_start": 0,
                    "raw_cell_id_stop": exporter.VOLUME_ENTITY_COUNT,
                    "complete_gap_free_duplicate_free": True,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.json"
            _write_json(evidence_path, skeletal)
            with mock.patch.object(
                exporter, "load_native_source_pin", return_value=_PinStub()
            ):
                with self.assertRaisesRegex(ValueError, "keys differ"):
                    exporter._verify_current_case_evidence(
                        evidence_path,
                        root / "absent-receipt.json",
                        evaluator_commit="d" * 40,
                        implementation=[],
                        native_source_pin_path=root / "pin.json",
                        surface=surface,
                        volume=volume,
                    )

    def test_receipt_must_bind_exact_evaluator_commit_and_evidence(self) -> None:
        surface = _prediction_record(
            "surface_native_cells", exporter.SURFACE_ENTITY_COUNT, "a"
        )
        volume = _prediction_record(
            "volume_native_cells", exporter.VOLUME_ENTITY_COUNT, "b"
        )
        evidence = _minimal_evidence(surface, volume)
        implementation = [
            {"path": path, "sha256": "d" * 64, "size_bytes": 1}
            for path in candidate_cli.CORE_EVALUATOR_GIT_PATHS
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.json"
            evidence_sha, evidence_size = _write_json(evidence_path, evidence)
            receipt_path = root / "receipt.json"
            receipt = {
                "schema": candidate_cli.IMPLEMENTATION_RECEIPT_SCHEMA,
                "schema_version": 1,
                "status": candidate_cli.IMPLEMENTATION_RECEIPT_STATUS,
                "official_submission": False,
                "case_id": exporter.CASE_ID,
                "evaluator": {
                    "repository": "neilashton/fluidsbench-submission",
                    "git_revision": "e" * 40,
                    "implementation": implementation,
                    "preflight_verified": True,
                    "postflight_verified": True,
                },
                "evidence": {
                    "file": evidence_path.name,
                    "sha256": evidence_sha,
                    "size_bytes": evidence_size,
                    "schema": evidence["schema"],
                    "schema_version": evidence["schema_version"],
                },
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "vtk": "9.5.2",
                },
            }
            _write_json(receipt_path, receipt)
            validated = SimpleNamespace(
                surface_count=exporter.SURFACE_ENTITY_COUNT,
                volume_count=exporter.VOLUME_ENTITY_COUNT,
            )
            with (
                mock.patch.object(
                    exporter, "load_native_source_pin", return_value=_PinStub()
                ),
                mock.patch.object(
                    exporter, "_validate_core_case", return_value=validated
                ),
            ):
                result = exporter._verify_current_case_evidence(
                    evidence_path,
                    receipt_path,
                    evaluator_commit="e" * 40,
                    implementation=implementation,
                    native_source_pin_path=root / "pin.json",
                    surface=surface,
                    volume=volume,
                )
                self.assertTrue(
                    result["complete_schema_native_audits_and_execution_validated"]
                )

                receipt["evaluator"]["git_revision"] = "f" * 40
                _write_json(receipt_path, receipt)
                with self.assertRaisesRegex(
                    exporter.ExportError, "does not bind the exact evaluator"
                ):
                    exporter._verify_current_case_evidence(
                        evidence_path,
                        receipt_path,
                        evaluator_commit="e" * 40,
                        implementation=implementation,
                        native_source_pin_path=root / "pin.json",
                        surface=surface,
                        volume=volume,
                    )


if __name__ == "__main__":
    unittest.main()
