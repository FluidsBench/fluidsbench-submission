from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reference.drivaerml.accumulators import (
    AdditiveFieldSums,
    FinalizedFieldStatistics,
)
from scripts.audit_drivaerml_native_volume_case import (
    compare_metric_passes,
    equal_native_cell_metric_pass,
)
import scripts.aggregate_drivaerml_native_volume_audits as aggregate_module

from scripts.aggregate_drivaerml_native_volume_audits import (
    CASE_AUDIT_SCHEMA,
    COMPARISON_CHUNK_CELLS,
    EQUAL_CELL_ALL_CASE_AUDIT_SCHEMA,
    EQUAL_CELL_PILOT_AGGREGATE_SCHEMA,
    MAX_ADDITIVE_RELATIVE_DIFFERENCE,
    MAX_METRIC_ABSOLUTE_DIFFERENCE,
    REFERENCE_CHUNK_CELLS,
    ZERO_PREDICTION_ROLE,
    NativeVolumeAuditAggregateError,
    aggregate_native_volume_equal_cell_all_case_audit,
    aggregate_native_volume_equal_cell_pilot,
    sha256_file,
    write_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_drivaerml_native_volume_audits.py"
CASE_SCRIPT = ROOT / "scripts" / "audit_drivaerml_native_volume_case.py"
REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
CASE_IDS = ("run_1", "run_44")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _difference_rows() -> dict[str, dict[str, float]]:
    return {
        name: {"absolute_difference": 0.0, "relative_difference": 0.0}
        for name in (
            "absolute_error",
            "squared_error",
            "squared_truth",
            "total_weight",
        )
    }


class NativeVolumeAuditFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cell_counts = {
            "run_1": REFERENCE_CHUNK_CELLS + 7,
            "run_44": REFERENCE_CHUNK_CELLS + 11,
        }
        self.pin = self._make_pin()
        self.pin_path = root / "native-source-pin.json"
        _write_json(self.pin_path, self.pin)
        self.receipt_paths = []
        for case_id in CASE_IDS:
            path = root / f"{case_id}-audit.json"
            _write_json(path, self.make_receipt(case_id))
            self.receipt_paths.append(path)

    def _make_pin(self) -> dict[str, object]:
        cases = []
        for case_id in CASE_IDS:
            number = int(case_id.removeprefix("run_"))
            part_count = 2 if case_id == "run_1" else 3
            part_sizes = [400 + index for index in range(part_count)]
            boundary_sha = _sha256(f"{case_id}-boundary".encode())
            cases.append(
                {
                    "case_id": case_id,
                    "run_number": number,
                    "boundary": {
                        "path": f"{case_id}/boundary_{number}.vtp",
                        "size_bytes": 100,
                        "lfs_sha256": boundary_sha,
                    },
                    "surface_cell_area": {
                        "path": f"{case_id}/boundary_cell_area_{number}.npy",
                        "size_bytes": 132,
                        "lfs_sha256": _sha256(f"{case_id}-area".encode()),
                        "dtype": "<f4",
                        "element_count": 1,
                        "source_boundary_sha256": boundary_sha,
                    },
                    "volume": {
                        "assembly": (
                            "byte concatenation of parts in listed order, with no "
                            "delimiter or transformation"
                        ),
                        "identity_contract": (
                            "ordered (path,size_bytes,lfs_sha256) tuples; no "
                            "assembled-file SHA-256 is asserted"
                        ),
                        "logical_path_after_assembly": f"{case_id}/volume_{number}.vtu",
                        "part_count": part_count,
                        "parts": [
                            {
                                "part_index": index,
                                "path": (
                                    f"{case_id}/volume_{number}.vtu.{index:02d}.part"
                                ),
                                "size_bytes": size,
                                "lfs_sha256": _sha256(
                                    f"{case_id}-part-{index}".encode()
                                ),
                            }
                            for index, size in enumerate(part_sizes)
                        ],
                        "total_size_bytes": sum(part_sizes),
                    },
                }
            )
        return {
            "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
            "schema_version": 1,
            "repository": {
                "provider": "Hugging Face Hub",
                "repo_id": "neashton/drivaerml",
                "repo_type": "dataset",
                "revision": REVISION,
            },
            "case_scope": {
                "case_count": len(cases),
                "run_number_min": 1,
                "run_number_max": 44,
                "unavailable_or_held_back_run_numbers": list(range(2, 44)),
            },
            "cases": cases,
            "totals": {
                "boundary_file_count": len(cases),
                "boundary_bytes": 200,
                "surface_cell_area_file_count": len(cases),
                "surface_cell_area_bytes": 264,
                "logical_volume_count": len(cases),
                "volume_part_file_count": 5,
                "reconstructed_volume_bytes": sum(
                    case["volume"]["total_size_bytes"] for case in cases
                ),
            },
        }

    def case_pin(self, case_id: str) -> dict[str, object]:
        return next(case for case in self.pin["cases"] if case["case_id"] == case_id)

    def field_audit(
        self, case_id: str, name: str, components: int, units: str
    ) -> dict[str, object]:
        count = self.cell_counts[case_id]
        return {
            "name": name,
            "association": "CellData",
            "vtk_type": "Float32",
            "number_of_components": components,
            "tuple_count": count,
            "scalar_count": count * components,
            "decoded_payload_bytes": count * components * 4,
            "payload_sha256": _sha256(f"{case_id}-{name}".encode()),
            "finite": True,
            "minimum_by_component": [-float(i + 1) for i in range(components)],
            "maximum_by_component": [float(i + 2) for i in range(components)],
            "units": units,
            "raw_id_start": 0,
            "raw_id_stop": count,
        }

    def metric_pass(
        self, case_id: str, field: dict[str, object], chunk_cells: int
    ) -> dict[str, object]:
        count = self.cell_counts[case_id]
        sums = {
            "absolute_error": 2.0 * count,
            "squared_error": 4.0 * count,
            "squared_truth": 4.0 * count,
            "entity_count": count,
            "total_weight": float(count),
        }
        return {
            "field_name": field["name"],
            "chunk_entities": chunk_cells,
            "source_payload_sha256": field["payload_sha256"],
            "source_payload_bytes": field["decoded_payload_bytes"],
            "entity_count": count,
            "component_count": field["number_of_components"],
            "metrics": {"relative_l2_percent": 100.0, "mae": 2.0, "rmse": 2.0},
            "additive_sums": sums,
        }

    def metric_fixture(
        self, case_id: str, field: dict[str, object]
    ) -> dict[str, object]:
        return {
            "prediction_role": ZERO_PREDICTION_ROLE,
            "reference_partition": self.metric_pass(
                case_id, field, REFERENCE_CHUNK_CELLS
            ),
            "comparison_partition": self.metric_pass(
                case_id, field, COMPARISON_CHUNK_CELLS
            ),
            "invariance": {
                "same_source_payload_sha256": True,
                "same_complete_entity_coverage": True,
                "equal_native_cell_additive_sums": _difference_rows(),
                "maximum_additive_relative_difference": 0.0,
                "maximum_metric_absolute_difference": 0.0,
            },
        }

    def make_receipt(self, case_id: str) -> dict[str, object]:
        case = self.case_pin(case_id)
        count = self.cell_counts[case_id]
        pressure = self.field_audit(case_id, "pMeanTrim", 1, "m^2/s^2")
        velocity = self.field_audit(case_id, "UMeanTrim", 3, "m/s")
        source_path = str(self.root / f"assembled-{case_id}.vtu")
        offset = 0
        segments = []
        for part in case["volume"]["parts"]:
            segments.append(
                {
                    "label": f"{case_id}:monolithic-segment:{part['part_index']}",
                    "path": source_path,
                    "file_offset": offset,
                    "size_bytes": part["size_bytes"],
                    "sha256": part["lfs_sha256"],
                }
            )
            offset += part["size_bytes"]
        arrays = []
        for index, (name, components, positions) in enumerate(
            (
                ("pMeanTrim", 1, (10, 20, 100, 110)),
                ("UMeanTrim", 3, (120, 130, 250, 260)),
            )
        ):
            arrays.append(
                {
                    "array_index": index,
                    "piece_index": 0,
                    "association": "CellData",
                    "name": name,
                    "vtk_type": "Float32",
                    "number_of_components": components,
                    "format": "binary",
                    "opening_tag_start": positions[0],
                    "encoded_start": positions[1],
                    "encoded_end": positions[2],
                    "closing_tag_end": positions[3],
                }
            )
        return {
            "schema": CASE_AUDIT_SCHEMA,
            "status": "passed_candidate_evaluator_case_audit",
            "case_id": case_id,
            "public_source": {
                "repository_id": "neashton/drivaerml",
                "immutable_revision": REVISION,
                "logical_path": case["volume"]["logical_path_after_assembly"],
                "logical_size_bytes": case["volume"]["total_size_bytes"],
                "multipart_part_count": case["volume"]["part_count"],
                "verified_monolithic_segments": segments,
            },
            "vtk": {
                "dataset_type": "UnstructuredGrid",
                "version": "1.0",
                "byte_order": "LittleEndian",
                "header_type": "UInt64",
                "compressor": None,
                "piece": {
                    "piece_index": 0,
                    "number_of_points": count // 2,
                    "number_of_cells": count,
                    "number_of_verts": 0,
                    "number_of_lines": 0,
                    "number_of_strips": 0,
                    "number_of_polys": 0,
                },
                "data_arrays_in_raw_xml_order": arrays,
            },
            "required_cell_data": {"pMeanTrim": pressure, "UMeanTrim": velocity},
            "units": {
                "coordinates": "m",
                "pMeanTrim": "m^2/s^2",
                "UMeanTrim": "m/s",
            },
            "raw_cell_order": "zero-based VTK Piece CellData tuple order, unchanged",
            "coverage": {
                "expected_raw_id_interval": [0, count],
                "validation": (
                    "each metric pass finalized exact gap-free duplicate-free coverage"
                ),
            },
            "volume_weighting": {
                "weighting": "one_per_native_cell",
                "entity_count": count,
                "total_weight": float(count),
                "geometric_cell_volume_weights_used": False,
            },
            "metrics": {
                "pMeanTrim": self.metric_fixture(case_id, pressure),
                "UMeanTrim": self.metric_fixture(case_id, velocity),
            },
            "chunk_invariance_tolerances": {
                "maximum_additive_relative_difference": (
                    MAX_ADDITIVE_RELATIVE_DIFFERENCE
                ),
                "maximum_metric_absolute_difference": (
                    MAX_METRIC_ABSOLUTE_DIFFERENCE
                ),
            },
            "runtime": {
                "python": "3.12.13",
                "numpy": "2.2.6",
                "segment_verification_seconds": 1.0,
                "xml_index_seconds": 2.0,
                "field_audit_and_equal_cell_metric_seconds": 3.0,
                "total_seconds": 6.0,
            },
        }


class DrivAerMLNativeVolumeAuditAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = NativeVolumeAuditFixture(self.root)

    def receipt(self, position: int) -> dict[str, object]:
        return json.loads(self.fixture.receipt_paths[position].read_text())

    def rewrite(self, position: int, value: dict[str, object]) -> None:
        _write_json(self.fixture.receipt_paths[position], value)

    def all_case(self, paths: list[Path] | None = None) -> dict[str, object]:
        return aggregate_native_volume_equal_cell_all_case_audit(
            native_source_pin_path=self.fixture.pin_path,
            receipt_paths=self.fixture.receipt_paths if paths is None else paths,
        )

    def test_all_case_is_deterministic_path_free_and_counts_multipart_sources(self) -> None:
        evidence = self.all_case(list(reversed(self.fixture.receipt_paths)))
        self.assertEqual(evidence["schema"], EQUAL_CELL_ALL_CASE_AUDIT_SCHEMA)
        self.assertEqual(evidence["schema_version"], 2)
        self.assertEqual(evidence["status"], "passed_all_case_equal_native_cell_audit")
        self.assertTrue(evidence["scope"]["complete_native_source_pin_scope"])
        self.assertEqual([row["case_id"] for row in evidence["cases"]], list(CASE_IDS))
        self.assertEqual(evidence["totals"]["two_part_case_count"], 1)
        self.assertEqual(evidence["totals"]["three_part_case_count"], 1)
        self.assertEqual(evidence["totals"]["verified_segment_count"], 5)
        self.assertEqual(
            evidence["totals"]["native_cell_count"],
            sum(self.fixture.cell_counts.values()),
        )
        self.assertEqual(
            {row["volume_weighting"]["weighting"] for row in evidence["cases"]},
            {"one_per_native_cell"},
        )
        serialized = json.dumps(evidence, sort_keys=True)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("secondary", serialized)
        self.assertNotIn("unresolved", serialized)

        first = self.root / "first.json"
        second = self.root / "second.json"
        write_evidence(first, evidence)
        write_evidence(second, self.all_case())
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_receipt_sha_and_validation_do_not_use_independent_path_reads(self) -> None:
        real_sha256_file = aggregate_module.sha256_file
        receipt_path = self.fixture.receipt_paths[0]
        independent_receipt_hash_called = False

        def mutate_after_independent_hash(path, *args, **kwargs):
            nonlocal independent_receipt_hash_called
            digest = real_sha256_file(path, *args, **kwargs)
            if Path(path) == receipt_path:
                independent_receipt_hash_called = True
                changed = self.fixture.make_receipt("run_1")
                changed["runtime"]["total_seconds"] = 7.0
                _write_json(receipt_path, changed)
            return digest

        with patch.object(
            aggregate_module,
            "sha256_file",
            side_effect=mutate_after_independent_hash,
        ):
            evidence = self.all_case()

        self.assertFalse(independent_receipt_hash_called)
        run_1 = next(row for row in evidence["cases"] if row["case_id"] == "run_1")
        self.assertEqual(run_1["receipt_sha256"], _sha256(receipt_path.read_bytes()))

    def test_receipt_path_replacement_during_parse_fails_closed(self) -> None:
        receipt_path = self.fixture.receipt_paths[0]
        replacement_path = self.root / "replacement-receipt.json"
        changed = self.fixture.make_receipt("run_1")
        changed["runtime"]["total_seconds"] = 7.0
        _write_json(replacement_path, changed)
        real_json_loads = json.loads
        replaced = False

        def replace_path_while_parsing(payload, *args, **kwargs):
            nonlocal replaced
            if not replaced:
                os.replace(replacement_path, receipt_path)
                replaced = True
            return real_json_loads(payload, *args, **kwargs)

        with patch.object(
            aggregate_module.json,
            "loads",
            side_effect=replace_path_while_parsing,
        ), self.assertRaisesRegex(
            NativeVolumeAuditAggregateError,
            "changed while its bytes were read and parsed",
        ):
            aggregate_module._read_json_with_sha256(
                receipt_path, "native-volume equal-cell receipt"
            )
        self.assertTrue(replaced)

    def test_selected_pilot_is_explicitly_incomplete(self) -> None:
        evidence = aggregate_native_volume_equal_cell_pilot(
            native_source_pin_path=self.fixture.pin_path,
            receipt_paths=[self.fixture.receipt_paths[0]],
            selected_case_ids=("run_1",),
        )
        self.assertEqual(evidence["schema"], EQUAL_CELL_PILOT_AGGREGATE_SCHEMA)
        self.assertEqual(evidence["status"], "passed_incomplete_equal_native_cell_pilot")
        self.assertFalse(evidence["scope"]["complete_native_source_pin_scope"])
        self.assertEqual(evidence["totals"]["two_part_case_count"], 1)
        self.assertEqual(evidence["totals"]["three_part_case_count"], 0)
        self.assertIn("all-case", evidence["scope_note"])
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "incomplete"):
            aggregate_native_volume_equal_cell_pilot(
                native_source_pin_path=self.fixture.pin_path,
                receipt_paths=self.fixture.receipt_paths,
                selected_case_ids=CASE_IDS,
            )

    def test_case_coverage_is_exact_and_duplicate_free(self) -> None:
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "missing=.*run_44"):
            self.all_case([self.fixture.receipt_paths[0]])
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "duplicate"):
            self.all_case(
                [
                    self.fixture.receipt_paths[0],
                    self.fixture.receipt_paths[0],
                    self.fixture.receipt_paths[1],
                ]
            )
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "pin order"):
            aggregate_native_volume_equal_cell_pilot(
                native_source_pin_path=self.fixture.pin_path,
                receipt_paths=self.fixture.receipt_paths,
                selected_case_ids=("run_44", "run_1"),
            )

    def test_volume_weighting_declaration_is_exact(self) -> None:
        mutations = (
            (lambda row: row.update(weighting="cell_volume_m3"), "exactly one weight"),
            (lambda row: row.update(entity_count=row["entity_count"] - 1), "exactly one weight"),
            (lambda row: row.update(total_weight=row["total_weight"] - 1.0), "exactly one weight"),
            (lambda row: row.update(total_weight=int(row["total_weight"])), "exactly one weight"),
            (
                lambda row: row.update(geometric_cell_volume_weights_used=True),
                "exactly one weight",
            ),
            (lambda row: row.update(status="audited"), "keys differ"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message, mutation=mutate):
                receipt = self.fixture.make_receipt("run_1")
                mutate(receipt["volume_weighting"])
                self.rewrite(0, receipt)
                with self.assertRaisesRegex(NativeVolumeAuditAggregateError, message):
                    self.all_case()

    def test_legacy_weight_and_physical_metric_vocabulary_is_rejected(self) -> None:
        receipt = self.fixture.make_receipt("run_1")
        receipt["volume_weights"] = receipt.pop("volume_weighting")
        self.rewrite(0, receipt)
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "keys differ"):
            self.all_case()

        receipt = self.fixture.make_receipt("run_1")
        receipt["metrics"]["pMeanTrim"]["reference_partition"]["additive_sums"] = {
            "uniform": receipt["metrics"]["pMeanTrim"]["reference_partition"][
                "additive_sums"
            ],
            "physical": {},
        }
        self.rewrite(0, receipt)
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "keys differ"):
            self.all_case()

        receipt = self.fixture.make_receipt("run_1")
        invariance = receipt["metrics"]["pMeanTrim"]["invariance"]
        invariance["additive_sums"] = invariance.pop(
            "equal_native_cell_additive_sums"
        )
        self.rewrite(0, receipt)
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "keys differ"):
            self.all_case()

    def test_source_field_coverage_and_invariance_tampering_fail_closed(self) -> None:
        mutations = (
            (
                lambda row: row["public_source"]["verified_monolithic_segments"][0].update(
                    sha256="f" * 64
                ),
                "segment 0 is not pinned",
            ),
            (
                lambda row: row["required_cell_data"]["pMeanTrim"].update(
                    association="PointData"
                ),
                "association/type/components",
            ),
            (
                lambda row: row["coverage"].update(
                    expected_raw_id_interval=[1, self.fixture.cell_counts["run_1"]]
                ),
                "coverage evidence is incomplete",
            ),
            (
                lambda row: row["metrics"]["pMeanTrim"]["comparison_partition"].update(
                    chunk_entities=REFERENCE_CHUNK_CELLS
                ),
                "partition binding",
            ),
            (
                lambda row: row["chunk_invariance_tolerances"].update(
                    maximum_additive_relative_difference=1.0e-6
                ),
                "thresholds are not frozen",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                receipt = self.fixture.make_receipt("run_1")
                mutate(receipt)
                self.rewrite(0, receipt)
                with self.assertRaisesRegex(NativeVolumeAuditAggregateError, message):
                    self.all_case()

    def test_output_rejects_absolute_paths(self) -> None:
        evidence = self.all_case()
        evidence["cases"][0]["source"]["logical_path"] = "/private/source.vtu"
        with self.assertRaisesRegex(NativeVolumeAuditAggregateError, "absolute path"):
            write_evidence(self.root / "must-not-write.json", evidence)

    def test_cli_exposes_only_equal_cell_pilot_and_all_case_modes(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--equal-cell-pilot", help_result.stdout)
        self.assertIn("--equal-cell-all-case-audit", help_result.stdout)
        self.assertNotIn("--volume-weight-aggregate", help_result.stdout)
        self.assertNotIn("--primary-only", help_result.stdout)

        output = self.root / "cli-all-case.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--native-source-pin",
                str(self.fixture.pin_path),
                "--equal-cell-all-case-audit",
                "--output",
                str(output),
                *(str(path) for path in reversed(self.fixture.receipt_paths)),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(output.read_text())["schema"],
            EQUAL_CELL_ALL_CASE_AUDIT_SCHEMA,
        )
        self.assertEqual(sha256_file(output), sha256_file(output))

    def test_case_auditor_serializes_only_equal_native_cell_statistics(self) -> None:
        sums = AdditiveFieldSums(
            absolute_error=4.0,
            squared_error=8.0,
            squared_truth=8.0,
            entity_count=2,
            total_weight=2.0,
        )
        statistics = FinalizedFieldStatistics(
            entity_count=2,
            component_count=1,
            uniform=sums,
            physical=sums,
        )
        metric_pass = SimpleNamespace(
            field_name="pMeanTrim",
            chunk_entities=1,
            source_payload=SimpleNamespace(
                payload_sha256="a" * 64,
                decoded_payload_bytes=8,
            ),
            statistics=statistics,
        )
        serialized = equal_native_cell_metric_pass(metric_pass)
        self.assertEqual(
            set(serialized),
            {
                "field_name",
                "chunk_entities",
                "source_payload_sha256",
                "source_payload_bytes",
                "entity_count",
                "component_count",
                "metrics",
                "additive_sums",
            },
        )
        self.assertNotIn("physical", json.dumps(serialized, sort_keys=True))
        invariance = compare_metric_passes(metric_pass, metric_pass)
        self.assertEqual(
            set(invariance["equal_native_cell_additive_sums"]),
            {"absolute_error", "squared_error", "squared_truth", "total_weight"},
        )
        self.assertNotIn("physical", json.dumps(invariance, sort_keys=True))

        help_result = subprocess.run(
            [sys.executable, str(CASE_SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertNotIn("--volume-weights", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
