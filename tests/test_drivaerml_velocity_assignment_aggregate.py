from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.autocfd5 import (
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    VelocityCellAssignmentEvidence,
    VelocitySampleDefinition,
    load_autocfd5_submission_definition,
)
from reference.drivaerml.velocity_assignments import (
    KERNEL_ID,
    NO_CLOSURE_CELL_REASON,
    QUERY_CACHE_KEY_ID,
    assignment_evidence_sha256,
    candidate_kernel_settings,
)
from scripts import aggregate_drivaerml_velocity_assignments as aggregate_module
from scripts.aggregate_drivaerml_velocity_assignments import (
    MANIFEST_SCHEMA,
    PINNED_KERNEL_SETTINGS_SHA256,
    PINNED_VERSIONS,
    VelocityAssignmentAggregateError,
    aggregate_velocity_assignments,
    sha256_file,
    write_manifest,
)
from scripts.generate_drivaerml_velocity_assignments import (
    ARTIFACT_SCHEMA,
    EXPECTED_AUTOCFD5_PROFILE_SHA256,
    EXPECTED_DATASET_REPOSITORY_ID,
    EXPECTED_DATASET_REVISION,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    RESOLUTIONS,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "benchmark-specs" / "drivaerml" / "drivaerml-diagnostics-v9.json"
CASE_ID = "run_44"
PROFILE_IDS = (
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "L1",
    "R1",
    "R2",
    "R3",
)
SMALL_SAMPLE_COUNTS = {1: 16, 2: 16, 5: 16, 10: 16}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class DrivAerMLVelocityAssignmentAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.receipts_root = self.root / "receipts"
        self.case_root = self.receipts_root / CASE_ID
        self.case_root.mkdir(parents=True)
        self.definition = load_autocfd5_submission_definition(PROFILE)
        self.samples = self._samples()
        self.pin_path, self.parts = self._pin()
        self.registry_binding = {
            "profile_sha256": EXPECTED_AUTOCFD5_PROFILE_SHA256,
            "source_registry_sha256": {
                name: digest for name, digest in self.definition.source_sha256
            },
            "line_count": 16,
            "fixed_10mm_sample_count": 16,
            "continuous_cp_cut_count": 4,
            "discrete_cp_probe_count": 0,
        }
        self.kernel = {
            "kernel_id": KERNEL_ID,
            "settings_sha256": PINNED_KERNEL_SETTINGS_SHA256,
            "versions": PINNED_VERSIONS,
            "settings": candidate_kernel_settings(),
        }
        self.source_binding = self._source_binding()
        self.geometry = {
            "dataset_type": "UnstructuredGrid",
            "version": "1.0",
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
            "compressor": "vtkZLibDataCompressor",
            "piece_count": 1,
            "declared_point_count": 8,
            "declared_cell_count": 2,
            "vtk_loaded_point_count": 8,
            "vtk_loaded_cell_count": 2,
            "reader_audit": {
                "disabled_point_array_count": 0,
                "disabled_point_arrays": [],
                "disabled_cell_array_count": 2,
                "disabled_cell_arrays": ["pMeanTrim", "UMeanTrim"],
            },
            "association": "native_volume_CellData",
            "raw_vtk_cell_id": "zero_based_native_GetCell_index",
            "remeshing": False,
            "reordering": False,
        }
        self._write_case()

    def _samples(self) -> dict[int, tuple[VelocitySampleDefinition, ...]]:
        result: dict[int, tuple[VelocitySampleDefinition, ...]] = {}
        for spacing_mm, spacing_m, _ in RESOLUTIONS:
            result[spacing_mm] = tuple(
                VelocitySampleDefinition(
                    profile_id=profile_id,
                    sample_index=0,
                    point_count=2,
                    line_fraction=0.0,
                    distance_m=0.0,
                    point_m=(float(index), spacing_m, 0.0),
                )
                for index, profile_id in enumerate(PROFILE_IDS)
            )
        return result

    def _pin(self) -> tuple[Path, list[bytes]]:
        parts = [b"synthetic-part-zero", b"synthetic-part-one", b"part-two"]
        boundary_sha = _sha256_bytes(b"boundary")
        part_rows = [
            {
                "part_index": index,
                "path": f"{CASE_ID}/volume_44.vtu.{index:02d}.part",
                "size_bytes": len(payload),
                "lfs_sha256": _sha256_bytes(payload),
            }
            for index, payload in enumerate(parts)
        ]
        pin = {
            "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
            "schema_version": 1,
            "repository": {
                "repo_id": EXPECTED_DATASET_REPOSITORY_ID,
                "revision": EXPECTED_DATASET_REVISION,
            },
            "case_scope": {
                "case_count": 1,
                "run_number_min": 44,
                "run_number_max": 44,
                "unavailable_or_held_back_run_numbers": [],
            },
            "cases": [
                {
                    "case_id": CASE_ID,
                    "run_number": 44,
                    "boundary": {
                        "path": f"{CASE_ID}/boundary_44.vtp",
                        "size_bytes": 8,
                        "lfs_sha256": boundary_sha,
                    },
                    "surface_cell_area": {
                        "path": f"{CASE_ID}/boundary_cell_area_44.npy",
                        "size_bytes": 132,
                        "lfs_sha256": _sha256_bytes(b"area"),
                        "dtype": "<f4",
                        "element_count": 1,
                        "source_boundary_sha256": boundary_sha,
                    },
                    "volume": {
                        "logical_path_after_assembly": f"{CASE_ID}/volume_44.vtu",
                        "part_count": 3,
                        "parts": part_rows,
                        "total_size_bytes": sum(map(len, parts)),
                    },
                }
            ],
            "totals": {
                "boundary_file_count": 1,
                "boundary_bytes": 8,
                "surface_cell_area_file_count": 1,
                "surface_cell_area_bytes": 132,
                "logical_volume_count": 1,
                "volume_part_file_count": 3,
                "reconstructed_volume_bytes": sum(map(len, parts)),
            },
        }
        pin_path = self.root / "native-source-pin.json"
        _write_json(pin_path, pin)
        return pin_path, parts

    def _source_binding(self) -> dict[str, object]:
        segments = []
        offset = 0
        for index, payload in enumerate(self.parts):
            segments.append(
                {
                    "part_index": index,
                    "byte_offset": offset,
                    "size_bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
            offset += len(payload)
        return {
            "pin_sha256": sha256_file(self.pin_path),
            "repository_id": EXPECTED_DATASET_REPOSITORY_ID,
            "repository_revision": EXPECTED_DATASET_REVISION,
            "case_id": CASE_ID,
            "logical_size_bytes": offset,
            "ordered_verified_segments": segments,
            "verification": {
                "method": "exact_ordered_segment_size_and_sha256",
                "timing": "completed_before_vtk_geometry_reader",
                "vtk_input": "retained_verified_file_descriptor",
                "post_vtk_fstat": "unchanged",
            },
        }

    def _rows_and_evidence(
        self, spacing_mm: int
    ) -> tuple[list[list[object]], list[VelocityCellAssignmentEvidence]]:
        rows: list[list[object]] = []
        evidence: list[VelocityCellAssignmentEvidence] = []
        source_hashes = tuple(_sha256_bytes(payload) for payload in self.parts)
        for position, sample in enumerate(self.samples[spacing_mm]):
            if position % 3 == 0:
                valid, reason, raw_cell_id, candidate_count = True, "", 0, 2
            elif position % 3 == 1:
                valid, reason, raw_cell_id, candidate_count = True, "", 1, 1
            else:
                valid = False
                reason = NO_CLOSURE_CELL_REASON
                raw_cell_id = None
                candidate_count = 0
            rows.append(
                [
                    sample.profile_id,
                    sample.sample_index,
                    list(sample.point_m),
                    sample.distance_m,
                    valid,
                    reason,
                    raw_cell_id,
                    candidate_count,
                ]
            )
            evidence.append(
                VelocityCellAssignmentEvidence(
                    case_id=CASE_ID,
                    profile_id=sample.profile_id,
                    sample_index=sample.sample_index,
                    point_m=sample.point_m,
                    distance_m=sample.distance_m,
                    valid=valid,
                    reason=reason,
                    raw_vtk_cell_id=raw_cell_id,
                    candidate_count=candidate_count,
                    geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                    source_sha256=source_hashes,
                )
            )
        return rows, evidence

    def _artifact(self, spacing_mm: int, spacing_m: float) -> dict[str, object]:
        rows, evidence = self._rows_and_evidence(spacing_mm)
        line_counts = {profile_id: 1 for profile_id in sorted(PROFILE_IDS)}
        return {
            "schema": ARTIFACT_SCHEMA,
            "schema_version": 1,
            "status": RECEIPT_STATUS,
            "case_id": CASE_ID,
            "resolution": {
                "nominal_spacing_mm": spacing_mm,
                "nominal_spacing_m": spacing_m,
                "sample_source": (
                    "expanded_autocfd5_v8_10mm_registry"
                    if spacing_mm == 10
                    else "endpoint_inclusive_equal_arc_from_v8_line_registry"
                ),
                "line_count": 16,
                "sample_count": 16,
            },
            "constant_evidence_fields": {
                "geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                "source_sha256": [
                    _sha256_bytes(payload) for payload in self.parts
                ],
            },
            "row_fields": list(aggregate_module.ROW_FIELDS),
            "rows": rows,
            "coverage": {
                "expected_sample_count": 16,
                "actual_sample_count": 16,
                "unique_line_sample_key_count": 16,
                "complete_duplicate_free_no_omissions": True,
                "line_sample_counts": line_counts,
                "valid_count": 11,
                "invalid_count": 5,
                "invalid_reason_counts": {NO_CLOSURE_CELL_REASON: 5},
                "candidate_count_histogram": {"0": 5, "1": 5, "2": 6},
                "selected_raw_vtk_cell_id_min": 0,
                "selected_raw_vtk_cell_id_max": 1,
            },
            "assignment_evidence_sha256": assignment_evidence_sha256(evidence),
            "units": {"point_m": "m", "distance_m": "m"},
            "registries": self.registry_binding,
            "native_source_binding": self.source_binding,
            "geometry": self.geometry,
            "kernel": self.kernel,
            "claims": dict(aggregate_module.FALSE_CASE_CLAIMS),
        }

    def _write_case(self) -> None:
        summaries = []
        for spacing_mm, spacing_m, label in RESOLUTIONS:
            artifact = self._artifact(spacing_mm, spacing_m)
            artifact_name = f"velocity-cell-mapping-{label}.json"
            artifact_path = self.case_root / artifact_name
            _write_json(artifact_path, artifact)
            summaries.append(
                {
                    "nominal_spacing_mm": spacing_mm,
                    "nominal_spacing_m": spacing_m,
                    "artifact": artifact_name,
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                    "assignment_evidence_sha256": artifact[
                        "assignment_evidence_sha256"
                    ],
                    "line_count": 16,
                    "sample_count": 16,
                    "valid_count": 11,
                    "invalid_count": 5,
                    "invalid_reason_counts": {NO_CLOSURE_CELL_REASON: 5},
                    "complete_duplicate_free_no_omissions": True,
                }
            )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": 1,
            "status": RECEIPT_STATUS,
            "case_id": CASE_ID,
            "registries": self.registry_binding,
            "native_source_binding": self.source_binding,
            "geometry": self.geometry,
            "kernel": self.kernel,
            "execution": {
                "io_chunk_bytes": 31,
                "validation_chunk_cells": 7,
                "resolution_order_mm": [1, 2, 5, 10],
                "geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                "containing_cell_query_cache": {
                    "enabled": True,
                    "key_id": QUERY_CACHE_KEY_ID,
                    "total_rows": 64,
                    "unique_query_keys": 64,
                    "cache_hits": 0,
                },
                "polyhedron_geometry_cache": {
                    "policy": "deterministic_least_recently_used",
                    "maximum_entries": 8_192,
                    "maximum_emitted_triangles": 131_072,
                    "current_entries": 0,
                    "current_emitted_triangles": 0,
                    "peak_entries": 0,
                    "peak_emitted_triangles": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "evictions": 0,
                    "oversized_entry_bypasses": 0,
                    "fail_closed_preparations": 0,
                    "vtk_objects_cached": False,
                },
                "polyhedron_evaluation": {
                    "scope": (
                        "broad_phase_polyhedron_visits_for_uncached_exact_xyz_"
                        "tolerance_queries"
                    ),
                    "broad_phase_polyhedron_visit_count": 0,
                    "boundary_count": 0,
                    "inside_count": 0,
                    "outside_count": 0,
                    "ambiguous_count": 0,
                    "winding_classified_count": 0,
                    "minimum_winding_classification_margin_steradian": None,
                    "classification_absolute_tolerance_steradian": 1.0e-3,
                },
            },
            "artifacts": summaries,
            "coverage": {
                "resolution_count": 4,
                "all_sixteen_lines_each_resolution": True,
                "all_samples_explicit_no_silent_omissions": True,
                "expected_sample_counts": {
                    str(key): value
                    for key, value in sorted(SMALL_SAMPLE_COUNTS.items())
                },
            },
            "claims": dict(aggregate_module.FALSE_CASE_CLAIMS),
        }
        _write_json(self.case_root / "receipt.json", receipt)

    def _patch_small_samples(self) -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        stack.enter_context(
            mock.patch.object(
                aggregate_module, "EXPECTED_SAMPLE_COUNTS", SMALL_SAMPLE_COUNTS
            )
        )
        stack.enter_context(
            mock.patch.object(
                aggregate_module,
                "_expected_samples_by_resolution",
                return_value=self.samples,
            )
        )
        return stack

    def _aggregate(self) -> dict[str, object]:
        with self._patch_small_samples():
            return aggregate_velocity_assignments(
                receipts_root=self.receipts_root,
                native_source_pin=self.pin_path,
                autocfd5_profile=PROFILE,
                pilot_case_ids=(CASE_ID,),
                official_case_ids=(CASE_ID,),
                expected_pin_sha256=None,
            )

    def _mutate_artifact(self, spacing_label: str, mutation: object) -> None:
        path = self.case_root / f"velocity-cell-mapping-{spacing_label}.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        mutation(artifact)
        _write_json(path, artifact)
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for summary in receipt["artifacts"]:
            if summary["artifact"] == path.name:
                summary["size_bytes"] = path.stat().st_size
                summary["sha256"] = sha256_file(path)
        _write_json(receipt_path, receipt)

    def test_explicit_pilot_emits_deterministic_compact_path_free_manifest(self) -> None:
        first = self._aggregate()
        second = self._aggregate()
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], MANIFEST_SCHEMA)
        self.assertEqual(
            first["status"],
            "incomplete_explicit_pilot_not_public_or_activation_evidence",
        )
        self.assertEqual(first["mode"], "explicit_non_public_pilot")
        self.assertFalse(first["case_scope"]["complete_484_public_case_set"])
        self.assertEqual(first["case_scope"]["explicit_pilot_case_ids"], [CASE_ID])
        self.assertEqual(first["totals"]["explicit_assignment_row_count"], 64)
        self.assertEqual(
            first["totals"]["containing_cell_query_cache"],
            {
                "key_id": QUERY_CACHE_KEY_ID,
                "expected_per_case": {
                    "total_rows": 64,
                    "unique_query_keys": 64,
                    "cache_hits": 0,
                },
                "audited_receipt_count": 1,
                "missing_pre_cache_pilot_receipt_count": 0,
                "total_rows": 64,
                "unique_query_keys_sum": 64,
                "cache_hits": 0,
            },
        )
        self.assertEqual(
            first["totals"]["polyhedron_geometry_cache"],
            {
                "audited_receipt_count": 1,
                "policy": "deterministic_least_recently_used",
                "maximum_entries_per_case": 8_192,
                "maximum_emitted_triangles_per_case": 131_072,
                "peak_entries_max": 0,
                "peak_emitted_triangles_max": 0,
                "current_entries": 0,
                "current_emitted_triangles": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "evictions": 0,
                "oversized_entry_bypasses": 0,
                "fail_closed_preparations": 0,
            },
        )
        self.assertEqual(
            first["totals"]["polyhedron_evaluation"],
            {
                "audited_receipt_count": 1,
                "classification_absolute_tolerance_steradian": 1.0e-3,
                "broad_phase_polyhedron_visit_count": 0,
                "boundary_count": 0,
                "inside_count": 0,
                "outside_count": 0,
                "ambiguous_count": 0,
                "winding_classified_count": 0,
                "minimum_winding_classification_margin_steradian": None,
            },
        )
        for spacing_mm in (1, 2, 5, 10):
            totals = first["totals"]["by_resolution_mm"][str(spacing_mm)]
            self.assertEqual(totals["sample_count"], 16)
            self.assertEqual(totals["valid_count"], 11)
            self.assertEqual(totals["invalid_count"], 5)
            self.assertEqual(totals["tie_assignment_count"], 6)
            self.assertEqual(totals["candidate_count_sum"], 17)
        self.assertFalse(
            first["tolerance_evidence"]["half_one_two_micrometre_replay_complete"]
        )
        self.assertTrue(all(value is False for value in first["claims"].values()))
        serialized = json.dumps(first, sort_keys=True)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn('"rows"', serialized)

        first_path = self.root / "first.json"
        second_path = self.root / "second.json"
        first_identity = write_manifest(first_path, first)
        second_identity = write_manifest(second_path, second)
        self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        self.assertEqual(first_identity, second_identity)

    def test_cli_requires_explicit_pilot_flag_and_writes_non_public_output(self) -> None:
        output = self.root / "pilot-manifest.json"
        stdout = io.StringIO()
        with self._patch_small_samples(), mock.patch.object(
            aggregate_module, "OFFICIAL_CASE_IDS", (CASE_ID,)
        ), mock.patch.object(
            aggregate_module, "OFFICIAL_NATIVE_SOURCE_PIN_SHA256", None
        ), contextlib.redirect_stdout(stdout):
            result = aggregate_module.main(
                [
                    "--receipts-root",
                    str(self.receipts_root),
                    "--native-source-pin",
                    str(self.pin_path),
                    "--autocfd5-profile",
                    str(PROFILE),
                    "--output",
                    str(output),
                    "--pilot-case",
                    CASE_ID,
                ]
            )
        self.assertEqual(result, 0)
        cli_summary = json.loads(stdout.getvalue())
        self.assertEqual(cli_summary["case_count"], 1)
        evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(evidence["mode"], "explicit_non_public_pilot")
        self.assertTrue(all(value is False for value in evidence["claims"].values()))

    def test_strict_default_rejects_any_partial_directory_set(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "exact official 484-case scope",
        ):
            aggregate_velocity_assignments(
                receipts_root=self.receipts_root,
                native_source_pin=self.pin_path,
                official_case_ids=(CASE_ID,),
                expected_pin_sha256=None,
            )
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "immutable official native-source pin hash",
        ):
            aggregate_velocity_assignments(
                receipts_root=empty,
                expected_pin_sha256=None,
            )
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "all 484 public receipt directories are not exact",
        ):
            aggregate_velocity_assignments(receipts_root=empty)

    def test_missing_cache_audit_is_only_accepted_in_explicit_pilot_mode(self) -> None:
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        del receipt["execution"]["containing_cell_query_cache"]
        _write_json(receipt_path, receipt)

        pilot = self._aggregate()
        self.assertEqual(pilot["mode"], "explicit_non_public_pilot")
        self.assertEqual(
            pilot["totals"]["containing_cell_query_cache"],
            {
                "key_id": QUERY_CACHE_KEY_ID,
                "expected_per_case": {
                    "total_rows": 64,
                    "unique_query_keys": 64,
                    "cache_hits": 0,
                },
                "audited_receipt_count": 0,
                "missing_pre_cache_pilot_receipt_count": 1,
                "total_rows": 0,
                "unique_query_keys_sum": 0,
                "cache_hits": 0,
            },
        )

        with self._patch_small_samples(), mock.patch.object(
            aggregate_module, "OFFICIAL_CASE_IDS", (CASE_ID,)
        ), mock.patch.object(
            aggregate_module, "OFFICIAL_NATIVE_SOURCE_PIN_SHA256", None
        ):
            with self.assertRaisesRegex(
                VelocityAssignmentAggregateError,
                "complete-mode receipt is missing.*query-cache audit",
            ):
                aggregate_velocity_assignments(
                    receipts_root=self.receipts_root,
                    native_source_pin=self.pin_path,
                    autocfd5_profile=PROFILE,
                    official_case_ids=(CASE_ID,),
                    expected_pin_sha256=None,
                )

    def test_query_cache_audit_counts_are_strict(self) -> None:
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["execution"]["containing_cell_query_cache"]["cache_hits"] = 1
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "query-cache audit differs from the exact registry-derived query keys",
        ):
            self._aggregate()

    def test_polyhedron_runtime_audits_are_strict(self) -> None:
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["execution"]["polyhedron_geometry_cache"][
            "maximum_entries"
        ] = 8_191
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "polyhedron geometry-cache audit is inconsistent",
        ):
            self._aggregate()

        self.setUp()
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["execution"]["polyhedron_evaluation"]["inside_count"] = 1
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "polyhedron evaluation audit is inconsistent",
        ):
            self._aggregate()

    def test_official_query_cache_counts_are_derived_from_exact_grids(self) -> None:
        expected_samples = aggregate_module._expected_samples_by_resolution(
            self.definition
        )
        self.assertEqual(
            aggregate_module._expected_query_cache_audit(expected_samples),
            {
                "key_id": QUERY_CACHE_KEY_ID,
                "total_rows": 67_384,
                "unique_query_keys": 39_362,
                "cache_hits": 28_022,
            },
        )

    def test_self_consistent_forged_query_cache_counts_fail(self) -> None:
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        audit = receipt["execution"]["containing_cell_query_cache"]
        audit["unique_query_keys"] = 1
        audit["cache_hits"] = audit["total_rows"] - 1
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError,
            "query-cache audit differs from the exact registry-derived query keys",
        ):
            self._aggregate()

    def test_rehashed_duplicate_and_invalid_raw_id_rows_fail(self) -> None:
        self._mutate_artifact(
            "01mm", lambda artifact: artifact["rows"].__setitem__(1, artifact["rows"][0])
        )
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError, "duplicate key"
        ):
            self._aggregate()

        self.setUp()
        self._mutate_artifact(
            "02mm", lambda artifact: artifact["rows"][0].__setitem__(6, 2)
        )
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError, "valid row is inconsistent"
        ):
            self._aggregate()

    def test_kernel_segment_geometry_and_artifact_hash_tampering_fail(self) -> None:
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["kernel"]["settings_sha256"] = "0" * 64
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError, "kernel settings hash"
        ):
            self._aggregate()

        self.setUp()
        receipt_path = self.case_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["native_source_binding"]["ordered_verified_segments"][1][
            "sha256"
        ] = "0" * 64
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError, "source segment 1 differs"
        ):
            self._aggregate()

        self.setUp()
        artifact_path = self.case_root / "velocity-cell-mapping-05mm.json"
        artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            VelocityAssignmentAggregateError, "artifact bytes differ"
        ):
            self._aggregate()

    def test_cell_evaluation_failure_reason_ids_are_canonical(self) -> None:
        prefix = "native_cell_closure_evaluation_failed_for_broad_phase_cells:"
        self.assertEqual(
            aggregate_module._evaluation_failure_ids(
                prefix + "3,9", cell_count=10, label="reason"
            ),
            (3, 9),
        )
        for suffix, message in (
            ("03", "non-canonical"),
            ("3,3", "unique and increasing"),
            ("9,3", "unique and increasing"),
            ("10", "out-of-range"),
        ):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(
                    VelocityAssignmentAggregateError, message
                ):
                    aggregate_module._evaluation_failure_ids(
                        prefix + suffix, cell_count=10, label="reason"
                    )


if __name__ == "__main__":
    unittest.main()
