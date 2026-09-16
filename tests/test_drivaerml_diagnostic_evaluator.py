from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import reference.drivaerml.diagnostic_evaluator as diagnostic_evaluator_module

from reference.drivaerml.autocfd5 import (
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    U_INF_M_PER_S,
    VelocityCellAssignmentEvidence,
    load_autocfd5_definition,
)
from reference.drivaerml.diagnostic_evaluator import (
    CANDIDATE_SCHEMA,
    CP_SUPPORT_SCHEMA,
    CP_SUPPORT_STATUS,
    EXPECTED_RESEARCH_PROFILE_SHA256,
    EXPECTED_SUBMISSION_PROFILE_SHA256,
    FALSE_CASE_CLAIMS,
    PINNED_KERNEL_VERSIONS,
    VELOCITY_ARTIFACT_SCHEMA,
    VELOCITY_RECEIPT_SCHEMA,
    VELOCITY_STATUS,
    DrivAerDiagnosticEvaluatorError,
    SparseNativeField,
    _evaluate_loaded_probe_research_diagnostics as evaluate_loaded_case_diagnostics,
    evaluate_loaded_case_diagnostics as evaluate_loaded_submission_diagnostics,
    evaluate_surface_only_case_diagnostics,
    gather_mapped_prediction_field,
    gather_sparse_inline_native_field,
    load_strict_cp_case_support,
    _load_research_velocity_10mm_mapping as load_research_velocity_10mm_mapping,
    load_strict_velocity_10mm_mapping as load_submission_velocity_10mm_mapping,
    sparse_native_field_from_array,
    write_candidate_diagnostic_evidence,
)
from reference.drivaerml.prediction_chunks import (
    CANDIDATE_ARTIFACT_ROLE,
    CANDIDATE_FORMAT,
    load_prediction_chunk_manifest,
)
from reference.drivaerml.source import index_inline_binary_vtk_xml
from reference.drivaerml.velocity_assignments import (
    CELL_EVALUATION_FAILURE_REASON_PREFIX,
    KERNEL_ID,
    NO_CLOSURE_CELL_REASON,
    POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES,
    POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES,
    POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE,
    QUERY_CACHE_KEY_ID,
    assignment_evidence_sha256,
    candidate_kernel_settings,
)
from scripts.build_drivaerml_cp_case_support import (
    CSV_SCHEMA as CP_CSV_SCHEMA,
    _algorithm_payload as cp_algorithm_payload,
    _csv_bytes as cp_csv_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
SUBMISSION_PROFILE = (
    ROOT / "benchmark-specs" / "drivaerml" / "drivaerml-diagnostics-v9.json"
)
CASE_ID = "run_44"
BOUNDARY_SHA = "b" * 64
STL_SHA = "c" * 64
PIN_SHA = "d" * 64
PART_SHA = ("e" * 64, "f" * 64)
SURFACE_COUNT = 256
VOLUME_COUNT = 3756


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_prediction_manifest(
    root: Path,
    *,
    support_id: str,
    partitions: tuple[int, ...],
    fields: dict[str, np.ndarray],
) -> Path:
    root.mkdir(parents=True)
    chunks = []
    start = 0
    for chunk_index, row_count in enumerate(partitions):
        stop = start + row_count
        relative = Path("chunks") / f"chunk-{chunk_index:05d}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            raw_cell_id=np.arange(start, stop, dtype=np.int64),
            **{name: values[start:stop] for name, values in fields.items()},
        )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "file": relative.as_posix(),
                "sha256": _sha256(path),
                "row_count": row_count,
                "raw_cell_id_start": start,
                "raw_cell_id_stop": stop,
            }
        )
        start = stop
    manifest = {
        "format": CANDIDATE_FORMAT,
        "format_version": 1,
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "case_id": CASE_ID,
        "support_id": support_id,
        "association": "CellData",
        "total_row_count": start,
        "field_components": {
            name: 1 if values.ndim == 1 else int(values.shape[1])
            for name, values in fields.items()
        },
        "chunks": chunks,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


class DiagnosticFixture:
    def __init__(
        self,
        root: Path,
        *,
        invalid_cp_position: int | None = None,
        invalid_velocity_position: int | None = None,
        invalid_velocity_reason: str = NO_CLOSURE_CELL_REASON,
        submission_velocity_profile: bool = False,
    ) -> None:
        self.root = root
        self.definition = load_autocfd5_definition(PROFILE)
        self.submission_velocity_profile = submission_velocity_profile
        self.surface_truth = np.linspace(-20.0, 40.0, SURFACE_COUNT, dtype=np.float64)
        pressure_delta = 0.05 * U_INF_M_PER_S * U_INF_M_PER_S
        self.surface_prediction = self.surface_truth + pressure_delta
        self.volume_truth = np.zeros((VOLUME_COUNT, 3), dtype=np.float64)
        self.volume_truth[:, 0] = U_INF_M_PER_S
        self.volume_prediction = np.zeros((VOLUME_COUNT, 3), dtype=np.float64)
        self.volume_prediction[:, 0] = 1.2 * U_INF_M_PER_S
        self.cp_path = root / "cp-support-run_44.json"
        self.velocity_root = root / "velocity"
        self.velocity_root.mkdir()
        self.velocity_path = self.velocity_root / "velocity-cell-mapping-10mm.json"
        self.receipt_path = self.velocity_root / "receipt.json"
        self._write_cp(invalid_cp_position)
        self._write_velocity(invalid_velocity_position, invalid_velocity_reason)

    def _write_cp(self, invalid_position: int | None) -> None:
        rows = []
        owner_status = {}
        pressure_values = []
        cp_values = []
        mapping_reasons = {}
        truth_reasons = {}
        solid_facet_counts = {
            rule.drivaerml_component: 300
            for rule in self.definition.cp_component_rules
        }
        for position, probe in enumerate(self.definition.cp_probes):
            rule = self.definition.cp_component_rules[position]
            raw_id = position % SURFACE_COUNT
            pressure = float(self.surface_truth[raw_id])
            truth_cp = 2.0 * pressure / (U_INF_M_PER_S * U_INF_M_PER_S)
            valid = position != invalid_position
            mapping_reason = (
                "" if valid else "no_native_polygon_bounds_candidate_within_2mm"
            )
            truth_valid = valid
            truth_reason = "" if valid else "mapping_invalid"
            if valid:
                pressure_values.append(pressure)
                cp_values.append(truth_cp)
            else:
                mapping_reasons[mapping_reason] = 1
                truth_reasons[truth_reason] = 1
            owner_status[rule.owner_review_status] = (
                owner_status.get(rule.owner_review_status, 0) + 1
            )
            rows.append(
                {
                    "case_id": CASE_ID,
                    "autocfd_probe_id": probe.autocfd_probe_id,
                    "nominal_point_m": list(probe.point_m),
                    "drivaerml_component": rule.drivaerml_component,
                    "projection_mode": rule.projection_mode,
                    "cut_axis": rule.cut_axis,
                    "cut_value_m": rule.cut_value_m,
                    "owner_review_status": rule.owner_review_status,
                    "mapping_valid": valid,
                    "mapping_reason": mapping_reason,
                    "mapped_stl_point_m": list(probe.point_m),
                    "raw_stl_triangle_id": position,
                    "stl_unit_normal": [1.0, 0.0, 0.0],
                    "nominal_displacement_m": 0.0,
                    "native_closest_point_m": list(probe.point_m) if valid else None,
                    "raw_vtk_polygon_id": raw_id if valid else None,
                    "native_polygon_unit_normal": (
                        [1.0, 0.0, 0.0] if valid else None
                    ),
                    "bridge_distance_m": 0.0 if valid else None,
                    "bridge_abs_normal_dot": 1.0 if valid else None,
                    "component_facet_count": solid_facet_counts[
                        rule.drivaerml_component
                    ],
                    "projection_candidate_count": 1,
                    "native_bounds_candidate_count": 1 if valid else 0,
                    "native_distance_pass_count": 1 if valid else 0,
                    "native_normal_pass_count": 1 if valid else 0,
                    "review_flags": [],
                    "truth_valid": truth_valid,
                    "truth_reason": truth_reason,
                    "pMeanTrim_m2_per_s2": pressure if valid else None,
                    "truth_Cp": truth_cp if valid else None,
                    "support_valid": valid,
                }
            )
        valid_count = len(rows) - (invalid_position is not None)
        document = {
            "schema": CP_SUPPORT_SCHEMA,
            "schema_version": "1.0",
            "case_id": CASE_ID,
            "status": CP_SUPPORT_STATUS,
            "owner_visual_signoff_claimed": False,
            "definition": {
                "profile_sha256": EXPECTED_RESEARCH_PROFILE_SHA256,
                "registry_sha256": dict(self.definition.source_sha256),
                "probe_count": len(rows),
                "rule_count": len(rows),
            },
            "sources": {
                "stl": {
                    "file": "drivaer_44.stl",
                    "sha256": STL_SHA,
                    "size_bytes": 10,
                    "solid_count": len(solid_facet_counts),
                    "facet_count": sum(solid_facet_counts.values()),
                    "solid_facet_counts": solid_facet_counts,
                },
                "boundary": {
                    "file": "boundary_44.vtp",
                    "sha256": BOUNDARY_SHA,
                    "size_bytes": 10,
                    "point_count": 100,
                    "polygon_count": SURFACE_COUNT,
                    "pMeanTrim_tuple_count": SURFACE_COUNT,
                    "pMeanTrim_component_count": 1,
                    "pMeanTrim_vtk_data_type": "float",
                },
            },
            "dependencies": {"vtk": "9.5.2", "numpy": "2.2.6"},
            "algorithm": cp_algorithm_payload(stl_chunk_facets=16_384),
            "summary": {
                "row_count": len(rows),
                "mapping_valid_count": valid_count,
                "mapping_invalid_count": len(rows) - valid_count,
                "mapping_invalid_reason_counts": mapping_reasons,
                "truth_valid_count": valid_count,
                "truth_invalid_count": len(rows) - valid_count,
                "truth_invalid_reason_counts": truth_reasons,
                "support_valid_count": valid_count,
                "owner_review_status_counts": dict(sorted(owner_status.items())),
                "review_flag_counts": {},
                "pMeanTrim_range_m2_per_s2": {
                    "minimum": min(pressure_values),
                    "maximum": max(pressure_values),
                },
                "truth_Cp_range": {
                    "minimum": min(cp_values),
                    "maximum": max(cp_values),
                },
            },
            "artifacts": {
                "mapping_csv_schema": CP_CSV_SCHEMA,
                "mapping_csv_row_count": len(rows),
                "mapping_csv_sha256": hashlib.sha256(cp_csv_bytes(rows)).hexdigest(),
            },
            "rows": rows,
        }
        _write_json(self.cp_path, document)

    def _write_velocity(
        self, invalid_position: int | None, invalid_reason: str
    ) -> None:
        settings = candidate_kernel_settings()
        settings_hash = hashlib.sha256(
            json.dumps(
                settings,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        source_registry_sha256 = dict(self.definition.source_sha256)
        if self.submission_velocity_profile:
            source_registry_sha256 = {
                key: value
                for key, value in source_registry_sha256.items()
                if key.startswith("velocity_")
            }
        registries = {
            "profile_sha256": (
                EXPECTED_SUBMISSION_PROFILE_SHA256
                if self.submission_velocity_profile
                else EXPECTED_RESEARCH_PROFILE_SHA256
            ),
            "source_registry_sha256": source_registry_sha256,
            "line_count": 16,
            "fixed_10mm_sample_count": 3756,
        }
        if self.submission_velocity_profile:
            registries.update(
                {
                    "continuous_cp_cut_count": 4,
                    "discrete_cp_probe_count": 0,
                }
            )
        source = {
            "pin_sha256": PIN_SHA,
            "repository_id": "neashton/drivaerml",
            "repository_revision": "7a5c0948ce27be709b1116a3a190f806e7a8f79f",
            "case_id": CASE_ID,
            "logical_size_bytes": 30,
            "ordered_verified_segments": [
                {
                    "part_index": 0,
                    "byte_offset": 0,
                    "size_bytes": 10,
                    "sha256": PART_SHA[0],
                },
                {
                    "part_index": 1,
                    "byte_offset": 10,
                    "size_bytes": 20,
                    "sha256": PART_SHA[1],
                },
            ],
            "verification": {
                "method": "exact_ordered_segment_size_and_sha256",
                "timing": "completed_before_vtk_geometry_reader",
                "vtk_input": "retained_verified_file_descriptor",
                "post_vtk_fstat": "unchanged",
            },
        }
        geometry = {
            "dataset_type": "UnstructuredGrid",
            "version": "1.0",
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
            "compressor": None,
            "piece_count": 1,
            "declared_point_count": 80,
            "declared_cell_count": VOLUME_COUNT,
            "vtk_loaded_point_count": 80,
            "vtk_loaded_cell_count": VOLUME_COUNT,
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
        kernel = {
            "kernel_id": KERNEL_ID,
            "settings_sha256": settings_hash,
            "versions": PINNED_KERNEL_VERSIONS,
            "settings": settings,
        }
        rows = []
        evidence = []
        line_counts = {}
        invalid_reasons = {}
        histogram = {}
        selected_ids = []
        for position, sample in enumerate(self.definition.velocity_samples):
            valid = position != invalid_position
            reason = "" if valid else invalid_reason
            raw_id = position if valid else None
            candidates = 1 if valid else 0
            rows.append(
                [
                    sample.profile_id,
                    sample.sample_index,
                    list(sample.point_m),
                    sample.distance_m,
                    valid,
                    reason,
                    raw_id,
                    candidates,
                ]
            )
            line_counts[sample.profile_id] = line_counts.get(sample.profile_id, 0) + 1
            histogram[str(candidates)] = histogram.get(str(candidates), 0) + 1
            if valid:
                selected_ids.append(raw_id)
            else:
                invalid_reasons[reason] = 1
            evidence.append(
                VelocityCellAssignmentEvidence(
                    case_id=CASE_ID,
                    profile_id=sample.profile_id,
                    sample_index=sample.sample_index,
                    point_m=sample.point_m,
                    distance_m=sample.distance_m,
                    valid=valid,
                    reason=reason,
                    raw_vtk_cell_id=raw_id,
                    candidate_count=candidates,
                    geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                    source_sha256=PART_SHA,
                )
            )
        valid_count = len(rows) - (invalid_position is not None)
        evidence_hash = assignment_evidence_sha256(evidence)
        artifact = {
            "schema": VELOCITY_ARTIFACT_SCHEMA,
            "schema_version": 1,
            "status": VELOCITY_STATUS,
            "case_id": CASE_ID,
            "resolution": {
                "nominal_spacing_mm": 10,
                "nominal_spacing_m": 0.01,
                "sample_source": "expanded_autocfd5_v8_10mm_registry",
                "line_count": 16,
                "sample_count": 3756,
            },
            "constant_evidence_fields": {
                "geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                "source_sha256": list(PART_SHA),
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
                "expected_sample_count": 3756,
                "actual_sample_count": 3756,
                "unique_line_sample_key_count": 3756,
                "complete_duplicate_free_no_omissions": True,
                "line_sample_counts": line_counts,
                "valid_count": valid_count,
                "invalid_count": len(rows) - valid_count,
                "invalid_reason_counts": invalid_reasons,
                "candidate_count_histogram": histogram,
                "selected_raw_vtk_cell_id_min": min(selected_ids),
                "selected_raw_vtk_cell_id_max": max(selected_ids),
            },
            "assignment_evidence_sha256": evidence_hash,
            "units": {"point_m": "m", "distance_m": "m"},
            "registries": registries,
            "native_source_binding": source,
            "geometry": geometry,
            "kernel": kernel,
            "claims": FALSE_CASE_CLAIMS,
        }
        _write_json(self.velocity_path, artifact)
        summary_10 = {
            "nominal_spacing_mm": 10,
            "nominal_spacing_m": 0.01,
            "artifact": self.velocity_path.name,
            "size_bytes": self.velocity_path.stat().st_size,
            "sha256": _sha256(self.velocity_path),
            "assignment_evidence_sha256": evidence_hash,
            "line_count": 16,
            "sample_count": 3756,
            "valid_count": valid_count,
            "invalid_count": 3756 - valid_count,
            "invalid_reason_counts": invalid_reasons,
            "complete_duplicate_free_no_omissions": True,
        }
        summaries = []
        for spacing, spacing_m, label, count in (
            (1, 0.001, "01mm", 37416),
            (2, 0.002, "02mm", 18716),
            (5, 0.005, "05mm", 7496),
        ):
            summaries.append(
                {
                    **summary_10,
                    "nominal_spacing_mm": spacing,
                    "nominal_spacing_m": spacing_m,
                    "artifact": f"velocity-cell-mapping-{label}.json",
                    "sample_count": count,
                }
            )
        summaries.append(summary_10)
        receipt = {
            "schema": VELOCITY_RECEIPT_SCHEMA,
            "schema_version": 1,
            "status": VELOCITY_STATUS,
            "case_id": CASE_ID,
            "registries": registries,
            "native_source_binding": source,
            "geometry": geometry,
            "kernel": kernel,
            "execution": {
                "io_chunk_bytes": 100,
                "validation_chunk_cells": 10,
                "resolution_order_mm": [1, 2, 5, 10],
                "geometric_tolerance_m": POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                "containing_cell_query_cache": {
                    "enabled": True,
                    "key_id": QUERY_CACHE_KEY_ID,
                    "total_rows": 67384,
                    "unique_query_keys": 39362,
                    "cache_hits": 28022,
                },
                "polyhedron_geometry_cache": {
                    "policy": "deterministic_least_recently_used",
                    "maximum_entries": POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES,
                    "maximum_emitted_triangles": (
                        POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES
                    ),
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
                    "classification_absolute_tolerance_steradian": (
                        POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
                    ),
                },
            },
            "artifacts": summaries,
            "coverage": {
                "resolution_count": 4,
                "all_sixteen_lines_each_resolution": True,
                "all_samples_explicit_no_silent_omissions": True,
                "expected_sample_counts": {
                    "1": 37416,
                    "2": 18716,
                    "5": 7496,
                    "10": 3756,
                },
            },
            "claims": FALSE_CASE_CLAIMS,
        }
        _write_json(self.receipt_path, receipt)

    def loaded(self):
        cp = load_strict_cp_case_support(
            self.cp_path,
            autocfd5_profile=PROFILE,
            case_id=CASE_ID,
            expected_boundary_sha256=BOUNDARY_SHA,
        )
        velocity = load_research_velocity_10mm_mapping(
            self.velocity_path,
            self.receipt_path,
            autocfd5_profile=PROFILE,
            case_id=CASE_ID,
            expected_source_pin_sha256=PIN_SHA,
            expected_source_part_sha256=PART_SHA,
        )
        return cp, velocity

    def loaded_submission_velocity(self):
        return load_submission_velocity_10mm_mapping(
            self.velocity_path,
            self.receipt_path,
            autocfd5_profile=SUBMISSION_PROFILE,
            case_id=CASE_ID,
            expected_source_pin_sha256=PIN_SHA,
            expected_source_part_sha256=PART_SHA,
        )

    def native_fields(self, cp, velocity):
        cp_ids = [
            row.raw_vtk_polygon_id
            for row in cp.rows
            if row.mapping_valid and row.raw_vtk_polygon_id is not None
        ]
        velocity_ids = [
            row.raw_vtk_cell_id
            for row in velocity.rows
            if row.valid and row.raw_vtk_cell_id is not None
        ]
        surface = sparse_native_field_from_array(
            self.surface_truth,
            cp_ids,
            case_id=CASE_ID,
            support_id="surface_native_cells",
            field_name="pMeanTrim",
            source_file="boundary_44.vtp",
            source_sha256=BOUNDARY_SHA,
        )
        unique_velocity = np.unique(np.asarray(velocity_ids, dtype=np.int64))
        volume = SparseNativeField(
            case_id=CASE_ID,
            support_id="volume_native_cells",
            field_name="UMeanTrim",
            total_row_count=VOLUME_COUNT,
            raw_cell_ids=unique_velocity,
            values=self.volume_truth[unique_velocity],
            source_files=("volume_44.vtu.00.part", "volume_44.vtu.01.part"),
            source_sha256=PART_SHA,
            complete_source_identity_verified=True,
        )
        return surface, volume

    def manifests(self, name: str, surface_parts, volume_parts):
        surface = _write_prediction_manifest(
            self.root / name / "surface",
            support_id="surface_native_cells",
            partitions=surface_parts,
            fields={
                "pMeanTrim": self.surface_prediction,
                "wallShearStressMeanTrim": np.zeros((SURFACE_COUNT, 3)),
            },
        )
        volume = _write_prediction_manifest(
            self.root / name / "volume",
            support_id="volume_native_cells",
            partitions=volume_parts,
            fields={
                "pMeanTrim": np.zeros(VOLUME_COUNT),
                "UMeanTrim": self.volume_prediction,
            },
        )
        return surface, volume


class DrivAerMLDiagnosticEvaluatorTests(unittest.TestCase):
    def test_surface_only_diagnostics_emit_no_volume_or_velocity_predictions(self) -> None:
        result = evaluate_surface_only_case_diagnostics(case_id=CASE_ID).to_json()

        self.assertEqual(result["schema"], "drivaerml-case-diagnostics-candidate-v4")
        self.assertEqual(result["prediction_scope"], "surface_only")
        self.assertEqual(
            result["mapping_inputs"]["velocity_10mm"]["status"],
            "not_loaded_surface_only",
        )
        self.assertEqual(result["profile_series"], [])
        self.assertEqual(
            result["metrics"]["velocity_profile_uinf_rmse"]["unavailable_reasons"][0][
                "reason"
            ],
            "not_submitted_surface_only",
        )

    def test_v9_submission_path_is_probe_free_and_cp_cuts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(
                Path(directory), submission_velocity_profile=True
            )
            cp = load_strict_cp_case_support(
                fixture.cp_path,
                autocfd5_profile=PROFILE,
                case_id=CASE_ID,
                expected_boundary_sha256=BOUNDARY_SHA,
            )
            velocity = fixture.loaded_submission_velocity()
            self.assertEqual(
                velocity.profile_sha256, EXPECTED_SUBMISSION_PROFILE_SHA256
            )
            _, native_volume = fixture.native_fields(cp, velocity)
            _, volume = fixture.manifests(
                "submission-v9", (SURFACE_COUNT,), (VOLUME_COUNT,)
            )
            result = evaluate_loaded_submission_diagnostics(
                velocity_mapping=velocity,
                volume_prediction_manifest=volume,
                native_volume_velocity=native_volume,
                maximum_prediction_chunk_rows=VOLUME_COUNT,
            ).to_json()
            self.assertEqual(result["schema"], "drivaerml-case-diagnostics-candidate-v4")
            self.assertEqual(result["schema_version"], 4)
            self.assertEqual(result["prediction_scope"], "surface_and_volume")
            self.assertEqual(set(result["mapping_inputs"]), {"velocity_10mm"})
            self.assertEqual(
                set(result["sparse_gather_evidence"]),
                {
                    "volume_prediction",
                    "volume_native_truth",
                    "only_unique_mapped_raw_ids_retained",
                    "prediction_manifest_fully_consumed",
                },
            )
            self.assertEqual(
                set(result["metrics"]),
                {
                    "cp_cut_rmse",
                    "velocity_profile_uinf_rmse",
                    "velocity_profile_experimental_subset_uinf_rmse",
                },
            )
            cp_cut = result["metrics"]["cp_cut_rmse"]
            self.assertFalse(cp_cut["ranked_value_available"])
            self.assertEqual(cp_cut["required_cut_count"], 4)
            self.assertEqual(
                cp_cut["weighting"], "native_cut_intersection_segment_length"
            )
            self.assertFalse(cp_cut["discrete_cp_probe_fallback_used"])
            self.assertEqual(cp_cut["cut_rmse"], [])
            self.assertEqual(
                set(cp_cut["unavailable_reasons"][0]),
                {"diagnostic", "stage", "reason"},
            )
            self.assertEqual(len(result["profile_series"]), 16)
            self.assertEqual(
                result["profile_series"][0]["station_id"], "autocfd5_v1"
            )
            self.assertEqual(
                result["profile_series"][-1]["station_id"], "autocfd5_r3"
            )
            for series in result["profile_series"]:
                self.assertEqual(series["panel_id"], "velocity_profiles")
                self.assertEqual(series["quantity_id"], "velocity_ratio")
                self.assertEqual(len(series["coordinate"]), len(series["prediction"]))
                self.assertGreaterEqual(len(series["coordinate"]), 2)
            self.assertNotIn("autocfd_probe_id", json.dumps(result, sort_keys=True))

    def test_mapping_json_mutation_during_parse_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mapping.json"
            _write_json(source, {"value": 1})
            original_json_load = diagnostic_evaluator_module.json.load

            def load_then_mutate(*args: object, **kwargs: object) -> object:
                value = original_json_load(*args, **kwargs)
                _write_json(source, {"value": "mutated-and-longer"})
                return value

            with mock.patch.object(
                diagnostic_evaluator_module.json,
                "load",
                side_effect=load_then_mutate,
            ), self.assertRaisesRegex(
                DrivAerDiagnosticEvaluatorError,
                "changed while its JSON was parsed",
            ):
                diagnostic_evaluator_module._read_json(source, "mapping JSON")

    def test_metrics_are_invariant_to_prediction_chunk_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(Path(directory))
            cp, velocity = fixture.loaded()
            self.assertEqual(
                velocity.profile_sha256, EXPECTED_RESEARCH_PROFILE_SHA256
            )
            native_surface, native_volume = fixture.native_fields(cp, velocity)
            surface_a, volume_a = fixture.manifests(
                "a", (SURFACE_COUNT,), (VOLUME_COUNT,)
            )
            surface_b, volume_b = fixture.manifests(
                "b", (17, 53, 186), (3, 7, 11, VOLUME_COUNT - 21)
            )
            first = evaluate_loaded_case_diagnostics(
                cp_support=cp,
                velocity_mapping=velocity,
                surface_prediction_manifest=surface_a,
                volume_prediction_manifest=volume_a,
                native_surface_pressure=native_surface,
                native_volume_velocity=native_volume,
                maximum_prediction_chunk_rows=VOLUME_COUNT,
                hash_chunk_bytes=19,
                validation_block_rows=13,
            ).to_json()
            second = evaluate_loaded_case_diagnostics(
                cp_support=cp,
                velocity_mapping=velocity,
                surface_prediction_manifest=surface_b,
                volume_prediction_manifest=volume_b,
                native_surface_pressure=native_surface,
                native_volume_velocity=native_volume,
                maximum_prediction_chunk_rows=VOLUME_COUNT,
                hash_chunk_bytes=23,
                validation_block_rows=11,
            ).to_json()
            self.assertEqual(first["schema"], CANDIDATE_SCHEMA)
            self.assertEqual(first["metrics"], second["metrics"])
            cp_metric = first["metrics"]["cp_probe_rmse"]
            cp_panel_metric = first["metrics"]["cp_panel_macro_rmse"]
            velocity_metric = first["metrics"]["velocity_profile_uinf_rmse"]
            experimental_velocity_metric = first["metrics"][
                "velocity_profile_experimental_subset_uinf_rmse"
            ]
            self.assertEqual(cp_metric["metric_id"], "cp_probe_rmse")
            self.assertEqual(cp_panel_metric["metric_id"], "cp_panel_macro_rmse")
            self.assertEqual(
                velocity_metric["metric_id"], "velocity_profile_uinf_rmse"
            )
            self.assertEqual(
                experimental_velocity_metric["metric_id"],
                "velocity_profile_experimental_subset_uinf_rmse",
            )
            self.assertTrue(cp_metric["ranked_value_available"])
            self.assertAlmostEqual(cp_metric["case_rmse"], 0.1, places=14)
            self.assertAlmostEqual(
                cp_panel_metric["case_equal_panel_mean_rmse"], 0.1, places=14
            )
            self.assertTrue(velocity_metric["ranked_value_available"])
            self.assertAlmostEqual(
                velocity_metric["case_equal_line_mean_rmse"], 0.2, places=14
            )
            self.assertEqual(len(velocity_metric["line_rmse"]), 16)
            self.assertAlmostEqual(
                experimental_velocity_metric[
                    "case_equal_experimental_line_mean_rmse"
                ],
                0.2,
                places=14,
            )
            self.assertEqual(len(experimental_velocity_metric["line_rmse"]), 11)
            self.assertNotEqual(
                first["sparse_gather_evidence"]["surface_prediction"][
                    "chunk_count"
                ],
                second["sparse_gather_evidence"]["surface_prediction"][
                    "chunk_count"
                ],
            )
            self.assertTrue(all(value is False for value in first["claims"].values()))
            self.assertNotIn(str(Path(directory)), json.dumps(first, sort_keys=True))
            output = Path(directory) / "diagnostics.json"
            identity = write_candidate_diagnostic_evidence(
                evaluate_loaded_case_diagnostics(
                    cp_support=cp,
                    velocity_mapping=velocity,
                    surface_prediction_manifest=surface_a,
                    volume_prediction_manifest=volume_a,
                    native_surface_pressure=native_surface,
                    native_volume_velocity=native_volume,
                    maximum_prediction_chunk_rows=VOLUME_COUNT,
                ),
                output,
            )
            self.assertRegex(identity["sha256"], r"^[0-9a-f]{64}$")

    def test_four_diagnostic_reductions_are_distinct_and_match_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(Path(directory))
            fixture.surface_prediction = fixture.surface_truth.copy()
            fixture.surface_prediction[0] += 0.5 * U_INF_M_PER_S**2
            fixture.volume_prediction = fixture.volume_truth.copy()
            experimental_ids = {
                line.profile_id
                for line in fixture.definition.velocity_lines
                if line.experimental_availability != "none"
            }
            for position, sample in enumerate(fixture.definition.velocity_samples):
                ratio_error = 0.1 if sample.profile_id in experimental_ids else 0.3
                fixture.volume_prediction[position, 0] = (
                    1.0 + ratio_error
                ) * U_INF_M_PER_S

            cp, velocity = fixture.loaded()
            native_surface, native_volume = fixture.native_fields(cp, velocity)
            surface, volume = fixture.manifests(
                "distinct", (SURFACE_COUNT,), (VOLUME_COUNT,)
            )
            result = evaluate_loaded_case_diagnostics(
                cp_support=cp,
                velocity_mapping=velocity,
                surface_prediction_manifest=surface,
                volume_prediction_manifest=volume,
                native_surface_pressure=native_surface,
                native_volume_velocity=native_volume,
                maximum_prediction_chunk_rows=VOLUME_COUNT,
            ).to_json()
            metrics = result["metrics"]
            cp_unique = metrics["cp_probe_rmse"]["case_rmse"]
            cp_panel = metrics["cp_panel_macro_rmse"][
                "case_equal_panel_mean_rmse"
            ]
            velocity_all = metrics["velocity_profile_uinf_rmse"][
                "case_equal_line_mean_rmse"
            ]
            velocity_experimental = metrics[
                "velocity_profile_experimental_subset_uinf_rmse"
            ]["case_equal_experimental_line_mean_rmse"]
            self.assertAlmostEqual(cp_unique, 1.0 / np.sqrt(209.0), places=14)
            panel_membership_counts = {}
            probe_zero_membership_counts = {}
            first_probe_id = fixture.definition.cp_probes[0].autocfd_probe_id
            for membership in fixture.definition.cp_panel_memberships:
                panel_membership_counts[membership.panel_id] = (
                    panel_membership_counts.get(membership.panel_id, 0) + 1
                )
                if membership.autocfd_probe_id == first_probe_id:
                    probe_zero_membership_counts[membership.panel_id] = (
                        probe_zero_membership_counts.get(membership.panel_id, 0) + 1
                    )
            expected_panel = sum(
                np.sqrt(
                    probe_zero_membership_counts.get(panel_id, 0)
                    / membership_count
                )
                for panel_id, membership_count in panel_membership_counts.items()
            ) / 15.0
            self.assertAlmostEqual(cp_panel, expected_panel, places=14)
            self.assertAlmostEqual(velocity_all, (11 * 0.1 + 5 * 0.3) / 16, places=14)
            self.assertAlmostEqual(velocity_experimental, 0.1, places=14)
            self.assertEqual(
                len(
                    {
                        round(float(cp_unique), 12),
                        round(float(cp_panel), 12),
                        round(float(velocity_all), 12),
                        round(float(velocity_experimental), 12),
                    }
                ),
                4,
            )

            specification = json.loads(
                (ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json")
                .read_text(encoding="utf-8")
            )
            definitions = {
                row["id"]: row for row in specification["metrics"]
            }
            for metric_id, output in metrics.items():
                self.assertEqual(output["metric_id"], metric_id)
                if metric_id.startswith("cp_"):
                    self.assertNotIn(metric_id, definitions)
                    continue
                self.assertEqual(
                    output["aggregation"], definitions[metric_id]["aggregation"]
                )
                self.assertEqual(
                    output["weighting"], definitions[metric_id]["weighting"]
                )

    def test_cp_loader_rejects_semantically_fabricated_evidence(self) -> None:
        mutations = (
            (
                "non-unit STL normal",
                lambda document: document["rows"][0].__setitem__(
                    "stl_unit_normal", [2.0, 0.0, 0.0]
                ),
                "STL normal is not unit",
            ),
            (
                "negative candidate count",
                lambda document: document["rows"][0].__setitem__(
                    "native_bounds_candidate_count", -1
                ),
                "must be an integer >= 0",
            ),
            (
                "fabricated review flag",
                lambda document: document["rows"][0].__setitem__(
                    "review_flags", ["native_bridge_distance_gt_0p5mm"]
                ),
                "review_flags are inconsistent",
            ),
            (
                "fabricated review summary",
                lambda document: document["summary"].__setitem__(
                    "review_flag_counts",
                    {"native_bridge_distance_gt_0p5mm": 1},
                ),
                "summary is inconsistent",
            ),
            (
                "fabricated artifact row count",
                lambda document: document["artifacts"].__setitem__(
                    "mapping_csv_row_count", 208
                ),
                "artifact declaration is not exact",
            ),
            (
                "fabricated artifact hash",
                lambda document: document["artifacts"].__setitem__(
                    "mapping_csv_sha256", "a" * 64
                ),
                "artifact declaration is not exact",
            ),
            (
                "owner review claim",
                lambda document: document.__setitem__(
                    "owner_visual_signoff_claimed", True
                ),
                "identity is not exact",
            ),
            (
                "fabricated source integrity",
                lambda document: document["algorithm"]["source_integrity"][
                    "stl"
                ].__setitem__("post_read_fstat", "unchecked"),
                "algorithm constants/status are not exact",
            ),
        )
        for label, mutate, expected_message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = DiagnosticFixture(Path(directory))
                document = json.loads(fixture.cp_path.read_text(encoding="utf-8"))
                mutate(document)
                _write_json(fixture.cp_path, document)
                with self.assertRaisesRegex(
                    DrivAerDiagnosticEvaluatorError, expected_message
                ):
                    load_strict_cp_case_support(
                        fixture.cp_path,
                        autocfd5_profile=PROFILE,
                        case_id=CASE_ID,
                    )

    def test_invalid_rows_are_explicit_and_make_ranked_values_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(
                Path(directory),
                invalid_cp_position=5,
                invalid_velocity_position=7,
                invalid_velocity_reason=(
                    CELL_EVALUATION_FAILURE_REASON_PREFIX + "3,9"
                ),
            )
            cp, velocity = fixture.loaded()
            native_surface, native_volume = fixture.native_fields(cp, velocity)
            surface, volume = fixture.manifests(
                "invalid", (100, 156), (15, VOLUME_COUNT - 15)
            )
            result = evaluate_loaded_case_diagnostics(
                cp_support=cp,
                velocity_mapping=velocity,
                surface_prediction_manifest=surface,
                volume_prediction_manifest=volume,
                native_surface_pressure=native_surface,
                native_volume_velocity=native_volume,
                maximum_prediction_chunk_rows=VOLUME_COUNT,
            ).to_json()
            cp_metric = result["metrics"]["cp_probe_rmse"]
            cp_panel_metric = result["metrics"]["cp_panel_macro_rmse"]
            velocity_metric = result["metrics"]["velocity_profile_uinf_rmse"]
            experimental_velocity_metric = result["metrics"][
                "velocity_profile_experimental_subset_uinf_rmse"
            ]
            self.assertFalse(cp_metric["ranked_value_available"])
            self.assertIsNone(cp_metric["case_rmse"])
            self.assertFalse(cp_metric["truth_equation_verified"])
            self.assertFalse(cp_panel_metric["truth_equation_verified"])
            self.assertEqual(
                cp_metric["unavailable_reasons"][0]["reason"],
                "no_native_polygon_bounds_candidate_within_2mm",
            )
            self.assertFalse(velocity_metric["ranked_value_available"])
            self.assertIsNone(velocity_metric["case_equal_line_mean_rmse"])
            self.assertEqual(
                velocity_metric["unavailable_reasons"][0]["reason"],
                CELL_EVALUATION_FAILURE_REASON_PREFIX + "3,9",
            )
            self.assertFalse(experimental_velocity_metric["value_available"])
            self.assertEqual(
                len(result["mapping_inputs"]["cp_support"]["invalid_rows"]), 1
            )
            self.assertEqual(
                len(result["mapping_inputs"]["velocity_10mm"]["invalid_rows"]),
                1,
            )

    def test_cp_equation_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(Path(directory))
            document = json.loads(fixture.cp_path.read_text(encoding="utf-8"))
            document["rows"][0]["truth_Cp"] += 1.0e-4
            _write_json(fixture.cp_path, document)
            with self.assertRaisesRegex(
                DrivAerDiagnosticEvaluatorError,
                "Cp=2\\*pMeanTrim/Uinf\\^2",
            ):
                load_strict_cp_case_support(
                    fixture.cp_path,
                    autocfd5_profile=PROFILE,
                    case_id=CASE_ID,
                )

    def test_failure_reason_ids_must_be_canonical_increasing_and_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(
                Path(directory),
                invalid_velocity_position=7,
                invalid_velocity_reason=(
                    CELL_EVALUATION_FAILURE_REASON_PREFIX + "9,3"
                ),
            )
            with self.assertRaisesRegex(
                DrivAerDiagnosticEvaluatorError,
                "unique, increasing, and in range",
            ):
                load_research_velocity_10mm_mapping(
                    fixture.velocity_path,
                    fixture.receipt_path,
                    autocfd5_profile=PROFILE,
                    case_id=CASE_ID,
                )

    def test_velocity_mapping_requires_retained_descriptor_source_receipt(self) -> None:
        for key in ("vtk_input", "post_vtk_fstat"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                fixture = DiagnosticFixture(Path(directory))
                for path in (fixture.velocity_path, fixture.receipt_path):
                    document = json.loads(path.read_text(encoding="utf-8"))
                    del document["native_source_binding"]["verification"][key]
                    _write_json(path, document)
                with self.assertRaisesRegex(
                    DrivAerDiagnosticEvaluatorError, "keys differ from schema"
                ):
                    load_research_velocity_10mm_mapping(
                        fixture.velocity_path,
                        fixture.receipt_path,
                        autocfd5_profile=PROFILE,
                        case_id=CASE_ID,
                    )

    def test_velocity_receipt_requires_consistent_v7_runtime_audits(self) -> None:
        mutations = (
            (
                "missing audit",
                lambda execution: execution.pop("polyhedron_evaluation"),
                "velocity execution keys differ from schema",
            ),
            (
                "query count",
                lambda execution: execution[
                    "containing_cell_query_cache"
                ].__setitem__("total_rows", 67_385),
                "query-cache audit is inconsistent",
            ),
            (
                "cache cap",
                lambda execution: execution[
                    "polyhedron_geometry_cache"
                ].__setitem__("maximum_entries", 8_191),
                "geometry-cache audit is inconsistent",
            ),
            (
                "cache accounting",
                lambda execution: execution[
                    "polyhedron_geometry_cache"
                ].__setitem__("current_entries", 1),
                "geometry-cache audit is inconsistent",
            ),
            (
                "evaluation accounting",
                lambda execution: execution[
                    "polyhedron_evaluation"
                ].__setitem__("broad_phase_polyhedron_visit_count", 1),
                "evaluation audit is inconsistent",
            ),
            (
                "classification margin",
                lambda execution: execution[
                    "polyhedron_evaluation"
                ].__setitem__(
                    "minimum_winding_classification_margin_steradian", 0.0
                ),
                "evaluation audit is inconsistent",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = DiagnosticFixture(Path(directory))
                receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
                mutate(receipt["execution"])
                _write_json(fixture.receipt_path, receipt)
                with self.assertRaisesRegex(
                    DrivAerDiagnosticEvaluatorError, message
                ):
                    load_research_velocity_10mm_mapping(
                        fixture.velocity_path,
                        fixture.receipt_path,
                        autocfd5_profile=PROFILE,
                        case_id=CASE_ID,
                    )

    def test_velocity_receipt_accepts_observed_v7_runtime_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(Path(directory))
            receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            receipt["execution"]["polyhedron_geometry_cache"].update(
                {
                    "cache_hits": 4520,
                    "cache_misses": 157,
                    "current_emitted_triangles": 3656,
                    "current_entries": 157,
                    "peak_emitted_triangles": 3656,
                    "peak_entries": 157,
                }
            )
            receipt["execution"]["polyhedron_evaluation"].update(
                {
                    "broad_phase_polyhedron_visit_count": 4677,
                    "boundary_count": 1735,
                    "inside_count": 1771,
                    "outside_count": 1171,
                    "winding_classified_count": 2942,
                    "minimum_winding_classification_margin_steradian": (
                        0.0009999999998383515
                    ),
                }
            )
            _write_json(fixture.receipt_path, receipt)
            mapping = load_research_velocity_10mm_mapping(
                fixture.velocity_path,
                fixture.receipt_path,
                autocfd5_profile=PROFILE,
                case_id=CASE_ID,
            )
            self.assertEqual(
                mapping.receipt_sha256, _sha256(fixture.receipt_path)
            )

    def test_unmapped_prediction_chunk_is_still_hashed_and_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(Path(directory))
            cp, velocity = fixture.loaded()
            native_surface, native_volume = fixture.native_fields(cp, velocity)
            surface, volume = fixture.manifests(
                "unmapped-tail", (209, 47), (VOLUME_COUNT,)
            )
            manifest = json.loads(surface.read_text(encoding="utf-8"))
            tail = surface.parent / manifest["chunks"][1]["file"]
            tail.write_bytes(tail.read_bytes() + b"tampered-unmapped-tail")
            with self.assertRaisesRegex(
                DrivAerDiagnosticEvaluatorError, "SHA-256 mismatch"
            ):
                evaluate_loaded_case_diagnostics(
                    cp_support=cp,
                    velocity_mapping=velocity,
                    surface_prediction_manifest=surface,
                    volume_prediction_manifest=volume,
                    native_surface_pressure=native_surface,
                    native_volume_velocity=native_volume,
                    maximum_prediction_chunk_rows=VOLUME_COUNT,
                )

    def test_preloaded_prediction_manifest_must_match_retained_file_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DiagnosticFixture(Path(directory))
            surface, _ = fixture.manifests(
                "preloaded-manifest", (SURFACE_COUNT,), (VOLUME_COUNT,)
            )
            preloaded = load_prediction_chunk_manifest(surface)

            # Replace the pathname with byte-distinct but semantically identical
            # JSON after preload. The old implementation evaluated preloaded chunk
            # descriptors while reporting the replacement file's SHA-256.
            document = json.loads(surface.read_text(encoding="utf-8"))
            replacement = surface.with_name("replacement.json")
            replacement.write_text(
                json.dumps(document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            replacement.replace(surface)

            with self.assertRaisesRegex(
                DrivAerDiagnosticEvaluatorError,
                "preloaded prediction manifest differs from a retained-file replay",
            ):
                gather_mapped_prediction_field(
                    preloaded,
                    [0],
                    case_id=CASE_ID,
                    support_id="surface_native_cells",
                    field_name="pMeanTrim",
                    expected_total_row_count=SURFACE_COUNT,
                )

    def test_sparse_inline_native_gather_scans_full_payload(self) -> None:
        velocity = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype="<f4",
        )
        payload = velocity.tobytes()
        encoded = base64.b64encode(struct.pack("<Q", len(payload)) + payload)
        document = b"".join(
            [
                b'<VTKFile type="UnstructuredGrid" version="0.1" ',
                b'byte_order="LittleEndian" header_type="UInt64">',
                b'<UnstructuredGrid><Piece NumberOfPoints="1" NumberOfCells="3">',
                b'<CellData><DataArray type="Float32" Name="UMeanTrim" ',
                b'NumberOfComponents="3" format="binary">',
                encoded,
                b"</DataArray></CellData></Piece></UnstructuredGrid></VTKFile>",
            ]
        )
        stream = io.BytesIO(document)
        index = index_inline_binary_vtk_xml(stream, scan_chunk_size=7)
        array = index.arrays_for(association="CellData", name="UMeanTrim")[0]
        field, summary = gather_sparse_inline_native_field(
            stream,
            index,
            array,
            [2, 0, 2],
            case_id=CASE_ID,
            support_id="volume_native_cells",
            field_name="UMeanTrim",
            expected_components=3,
            source_files=("volume_44.vtu.00.part", "volume_44.vtu.01.part"),
            source_sha256=PART_SHA,
            encoded_chunk_bytes=5,
        )
        np.testing.assert_array_equal(field.raw_cell_ids, [0, 2])
        np.testing.assert_allclose(field.values, velocity[[0, 2]])
        self.assertEqual(summary.tuple_count, 3)


if __name__ == "__main__":
    unittest.main()
