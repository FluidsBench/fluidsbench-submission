from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from reference.drivaerml.autocfd5 import (
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    U_INF_M_PER_S,
    VelocityCellAssignmentEvidence,
    load_autocfd5_definition,
)
from reference.drivaerml.diagnostic_evaluator import (
    SparseNativeField,
    gather_mapped_prediction_field,
)
from reference.drivaerml.profile_convergence import (
    canonical_sha256,
    method_set_sha256,
)
from reference.drivaerml.profile_convergence_evaluator import (
    CONSTRUCTED_EVIDENCE_SCHEMA,
    EXPECTED_SAMPLE_COUNTS,
    FALSE_CLAIMS,
    MAPPING_ARTIFACT_NAMES,
    CaseResolutionMappings,
    NativeProfileConvergenceError,
    ProfileMappingRow,
    ResolutionMapping,
    build_profile_convergence_from_predictions,
    case_profile_losses,
    finalize_profile_convergence_evidence,
    prediction_manifest_set_identity,
    validate_mapping_grids,
)
from reference.drivaerml.velocity_assignments import (
    assignment_evidence_sha256,
    generate_definition_velocity_samples,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
CONTRACT = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "proposal"
    / "contract-proposal.json"
)
CASE_ID = "run_1"
SOURCE_SHA = ("a" * 64, "b" * 64)
NATIVE_CELL_COUNT = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_volume_manifest(
    root: Path,
    velocity: np.ndarray,
    *,
    case_id: str = CASE_ID,
    partitions: tuple[int, ...] = (43, 85),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    chunks = []
    start = 0
    for index, row_count in enumerate(partitions):
        stop = start + row_count
        relative = Path("chunks") / f"chunk-{index:05d}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            raw_cell_id=np.arange(start, stop, dtype=np.int64),
            pMeanTrim=np.zeros(stop - start, dtype=np.float64),
            UMeanTrim=velocity[start:stop],
        )
        chunks.append(
            {
                "chunk_index": index,
                "file": relative.as_posix(),
                "sha256": _sha256(path),
                "row_count": row_count,
                "raw_cell_id_start": start,
                "raw_cell_id_stop": stop,
            }
        )
        start = stop
    manifest = {
        "format": "drivaerml-native-prediction-chunks-candidate",
        "format_version": 1,
        "artifact_role": "local_evaluator_input_not_official_submission_artifact",
        "case_id": case_id,
        "support_id": "volume_native_cells",
        "association": "CellData",
        "total_row_count": start,
        "field_components": {"pMeanTrim": 1, "UMeanTrim": 3},
        "chunks": chunks,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


def _assignment_sha256(rows: tuple[ProfileMappingRow, ...]) -> str:
    return assignment_evidence_sha256(
        tuple(
            VelocityCellAssignmentEvidence(
                case_id=CASE_ID,
                profile_id=row.profile_id,
                sample_index=row.sample_index,
                point_m=row.point_m,
                distance_m=row.distance_m,
                valid=row.valid,
                reason=row.reason,
                raw_vtk_cell_id=row.raw_vtk_cell_id,
                candidate_count=row.candidate_count,
                geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                source_sha256=SOURCE_SHA,
            )
            for row in rows
        )
    )


def _base_mappings(definition) -> CaseResolutionMappings:
    resolutions = []
    for spacing_mm in (1, 2, 5, 10):
        samples = (
            definition.velocity_samples
            if spacing_mm == 10
            else generate_definition_velocity_samples(
                definition, spacing_mm / 1000.0
            )
        )
        rows = tuple(
            ProfileMappingRow(
                profile_id=sample.profile_id,
                sample_index=sample.sample_index,
                point_m=sample.point_m,
                distance_m=sample.distance_m,
                valid=True,
                reason="",
                raw_vtk_cell_id=int(round(sample.distance_m * 1000.0))
                % NATIVE_CELL_COUNT,
                candidate_count=1,
            )
            for sample in samples
        )
        assignment_sha256 = _assignment_sha256(rows)
        resolutions.append(
            ResolutionMapping(
                case_id=CASE_ID,
                spacing_mm=spacing_mm,
                native_cell_count=NATIVE_CELL_COUNT,
                source_part_sha256=SOURCE_SHA,
                artifact_name=MAPPING_ARTIFACT_NAMES[spacing_mm],
                artifact_sha256=hashlib.sha256(
                    f"artifact-{spacing_mm}".encode()
                ).hexdigest(),
                assignment_evidence_sha256=assignment_sha256,
                rows=rows,
            )
        )
    return CaseResolutionMappings(CASE_ID, tuple(resolutions))


def _replace_resolution_row(
    mappings: CaseResolutionMappings,
    spacing_mm: int,
    position: int,
    **changes,
) -> CaseResolutionMappings:
    resolutions = list(mappings.resolutions)
    resolution_position = (1, 2, 5, 10).index(spacing_mm)
    original = resolutions[resolution_position]
    rows = list(original.rows)
    rows[position] = replace(rows[position], **changes)
    row_tuple = tuple(rows)
    resolutions[resolution_position] = replace(
        original,
        rows=row_tuple,
        assignment_evidence_sha256=_assignment_sha256(row_tuple),
    )
    return replace(mappings, resolutions=tuple(resolutions))


class DrivAerMLNativeProfileConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_autocfd5_definition(PROFILE)
        cls.mappings = _base_mappings(cls.definition)
        cls.required_ids = cls.mappings.required_raw_cell_ids()
        truth_values = np.zeros((len(cls.required_ids), 3), dtype=np.float64)
        truth_values[:, 0] = U_INF_M_PER_S
        cls.truth = SparseNativeField(
            case_id=CASE_ID,
            support_id="volume_native_cells",
            field_name="UMeanTrim",
            total_row_count=NATIVE_CELL_COUNT,
            raw_cell_ids=cls.required_ids,
            values=truth_values,
            source_files=("volume_1_part1.vtu", "volume_1_part2.vtu"),
            source_sha256=SOURCE_SHA,
            complete_source_identity_verified=True,
        )

    def test_exact_four_grids_are_stride_nested_and_assignment_invariant(self) -> None:
        result = validate_mapping_grids(self.mappings, self.definition)

        self.assertTrue(result["grids_stride_nested"])
        self.assertTrue(result["every_nested_geometric_assignment_invariant"])
        self.assertEqual(
            [row["spacing_mm"] for row in result["assignment_comparisons"]],
            [2, 5, 10],
        )
        self.assertEqual(
            [
                row["matched_nested_sample_count"]
                for row in result["assignment_comparisons"]
            ],
            [
                EXPECTED_SAMPLE_COUNTS[2],
                EXPECTED_SAMPLE_COUNTS[5],
                EXPECTED_SAMPLE_COUNTS[10],
            ],
        )

    def test_geometric_assignment_change_is_reported_separately(self) -> None:
        # First 2 mm row is also first 1 mm row and therefore a matched sample.
        changed = _replace_resolution_row(
            self.mappings, 2, 0, raw_vtk_cell_id=27
        )
        result = validate_mapping_grids(changed, self.definition)

        self.assertFalse(result["every_nested_geometric_assignment_invariant"])
        comparison = result["assignment_comparisons"][0]
        self.assertEqual(comparison["assignment_difference_count"], 1)
        self.assertEqual(
            comparison["first_assignment_differences"][0]["profile_id"], "V1"
        )

    def test_mapping_row_hash_and_invalid_reason_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            NativeProfileConvergenceError, "assignment-evidence SHA-256"
        ):
            replace(
                self.mappings.resolutions[0],
                assignment_evidence_sha256="f" * 64,
            )

        resolution = self.mappings.resolutions[0]
        rows = list(resolution.rows)
        rows[0] = replace(
            rows[0],
            valid=False,
            reason="submitter_inferred_invalidity",
            raw_vtk_cell_id=None,
            candidate_count=0,
        )
        row_tuple = tuple(rows)
        with self.assertRaisesRegex(
            NativeProfileConvergenceError, "unsupported invalid reason"
        ):
            replace(
                resolution,
                rows=row_tuple,
                assignment_evidence_sha256=_assignment_sha256(row_tuple),
            )

    def test_partial_loss_is_rejected_without_bound_owner_mask(self) -> None:
        resolutions = []
        for resolution in self.mappings.resolutions:
            rows = list(resolution.rows)
            v1_count = next(
                line.reference_1mm_sample_count
                if resolution.spacing_mm == 1
                else int((line.reference_1mm_sample_count - 1) / resolution.spacing_mm)
                + 1
                for line in self.definition.velocity_lines
                if line.profile_id == "V1"
            )
            gap_position = v1_count // 2
            rows[gap_position] = replace(
                rows[gap_position],
                valid=False,
                reason="outside_released_fluid_domain",
                raw_vtk_cell_id=None,
                candidate_count=0,
            )
            row_tuple = tuple(rows)
            resolutions.append(
                replace(
                    resolution,
                    rows=row_tuple,
                    assignment_evidence_sha256=_assignment_sha256(row_tuple),
                )
            )
        mappings = replace(self.mappings, resolutions=tuple(resolutions))
        required_ids = mappings.required_raw_cell_ids()
        truth = replace(
            self.truth,
            raw_cell_ids=required_ids,
            values=self.truth.values[
                np.searchsorted(self.truth.raw_cell_ids, required_ids)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
            velocity[:, 0] = 1.1 * U_INF_M_PER_S
            manifest = _write_volume_manifest(root / "prediction", velocity)
            prediction = gather_mapped_prediction_field(
                manifest,
                required_ids,
                case_id=CASE_ID,
                support_id="volume_native_cells",
                field_name="UMeanTrim",
                expected_total_row_count=NATIVE_CELL_COUNT,
            )

            with self.assertRaisesRegex(
                NativeProfileConvergenceError,
                "unresolved velocity mapping rows while no immutable owner validity mask",
            ):
                case_profile_losses(
                    mappings,
                    definition=self.definition,
                    native_truth=truth,
                    prediction=prediction,
                )

    def test_positive_arc_support_is_fail_closed(self) -> None:
        resolutions = []
        for resolution in self.mappings.resolutions:
            rows = tuple(
                replace(
                    row,
                    valid=False,
                    reason="outside_released_fluid_domain",
                    raw_vtk_cell_id=None,
                    candidate_count=0,
                )
                if row.profile_id == "V1"
                else row
                for row in resolution.rows
            )
            resolutions.append(
                replace(
                    resolution,
                    rows=rows,
                    assignment_evidence_sha256=_assignment_sha256(rows),
                )
            )
        invalid = replace(self.mappings, resolutions=tuple(resolutions))
        required = invalid.required_raw_cell_ids()
        truth = replace(
            self.truth,
            raw_cell_ids=required,
            values=self.truth.values[
                np.searchsorted(self.truth.raw_cell_ids, required)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
            velocity[:, 0] = U_INF_M_PER_S
            manifest = _write_volume_manifest(Path(temporary), velocity)
            prediction = gather_mapped_prediction_field(
                manifest,
                required,
                case_id=CASE_ID,
                support_id="volume_native_cells",
                field_name="UMeanTrim",
                expected_total_row_count=NATIVE_CELL_COUNT,
            )
            with self.assertRaisesRegex(
                NativeProfileConvergenceError,
                "unresolved velocity mapping rows while no immutable owner validity mask",
            ):
                case_profile_losses(
                    invalid,
                    definition=self.definition,
                    native_truth=truth,
                    prediction=prediction,
                )

    def test_manifest_set_pin_binds_order_counts_and_all_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
            first = _write_volume_manifest(Path(temporary) / "one", velocity)
            digest, parsed, records = prediction_manifest_set_identity(
                [CASE_ID], [first]
            )

            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(parsed[0].total_row_count, NATIVE_CELL_COUNT)
            self.assertEqual(records[0]["chunk_count"], 2)
            self.assertEqual(len(records[0]["ordered_chunk_sha256"]), 2)

    def test_complete_genuine_method_pilot_cannot_claim_owner_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = [
                "physics_null",
                "nearest_training_design_vector_control",
                "trained_model_checkpoint",
                "trained_model_checkpoint",
                "trained_model_checkpoint",
            ]
            manifests = {}
            methods = []
            for index, role in enumerate(roles):
                velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
                velocity[:, 0] = (1.0 + 0.05 * index) * U_INF_M_PER_S
                path = _write_volume_manifest(root / f"method-{index}", velocity)
                method_id = f"method-{index}"
                digest, _, _ = prediction_manifest_set_identity(
                    [CASE_ID], [path]
                )
                method = {
                    "method_id": method_id,
                    "role": role,
                    "prediction_artifact_sha256": digest,
                }
                if role == "trained_model_checkpoint":
                    method["model_id"] = f"architecture-{index}"
                    method["checkpoint_id"] = f"checkpoint-{index}"
                methods.append(method)
                manifests[method_id] = [path]
            method_set = {
                "pinned_before_study": True,
                "sha256": method_set_sha256(methods),
                "methods": methods,
            }

            result = build_profile_convergence_from_predictions(
                study_id="genuine-artifact-synthetic-values",
                method_set=method_set,
                case_order=[CASE_ID],
                case_mappings={CASE_ID: self.mappings},
                native_truth={CASE_ID: self.truth},
                prediction_manifests=manifests,
                definition=self.definition,
                contract_proposal=CONTRACT,
            )

            self.assertEqual(result["schema"], CONSTRUCTED_EVIDENCE_SCHEMA)
            self.assertEqual(
                result["status"],
                "pilot_or_incomplete_scope_not_eligible_for_owner_review",
            )
            self.assertFalse(
                result["profile_resolution_candidate_eligible_for_owner_review"]
            )
            self.assertIn(
                "incomplete_official_484_case_scope",
                result["blocking_reasons"],
            )
            self.assertIn(
                "owner_validity_mask_not_bound",
                result["blocking_reasons"],
            )
            self.assertEqual(result["claims"], FALSE_CLAIMS)
            self.assertFalse(result["claims"]["owner_scientific_approval"])
            self.assertTrue(
                result["geometric_assignment_convergence"]["passed"]
            )
            self.assertTrue(
                result["loss_and_method_order_convergence"]["passed"]
            )
            self.assertTrue(
                result["loss_and_method_order_convergence"][
                    "distinct_from_scope_and_method_set_eligibility"
                ]
            )

    def test_missing_genuine_models_is_truthfully_blocked_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = {}
            methods = []
            for index, role in enumerate(
                ("physics_null", "nearest_training_design_vector_control")
            ):
                velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
                velocity[:, 0] = (1.0 + 0.1 * index) * U_INF_M_PER_S
                path = _write_volume_manifest(root / f"method-{index}", velocity)
                digest, _, _ = prediction_manifest_set_identity(
                    [CASE_ID], [path]
                )
                method_id = f"method-{index}"
                methods.append(
                    {
                        "method_id": method_id,
                        "role": role,
                        "prediction_artifact_sha256": digest,
                    }
                )
                manifests[method_id] = [path]
            method_set = {
                "pinned_before_study": True,
                "sha256": method_set_sha256(methods),
                "methods": methods,
            }

            result = build_profile_convergence_from_predictions(
                study_id="outstanding-genuine-model-inputs",
                method_set=method_set,
                case_order=[CASE_ID],
                case_mappings={CASE_ID: self.mappings},
                native_truth={CASE_ID: self.truth},
                prediction_manifests=manifests,
                definition=self.definition,
                contract_proposal=CONTRACT,
            )

            self.assertEqual(
                result["status"],
                "pilot_or_incomplete_scope_not_eligible_for_owner_review",
            )
            self.assertIn(
                "missing_or_unpinned_genuine_method_predictions",
                result["blocking_reasons"],
            )
            self.assertFalse(
                result["profile_resolution_candidate_eligible_for_owner_review"]
            )
            method_summary = result["loss_and_method_order_convergence"][
                "evidence"
            ]["method_set"]
            self.assertEqual(
                method_summary["role_counts"]["trained_model_checkpoint"], 0
            )

    def test_declared_prediction_pin_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
            path = _write_volume_manifest(Path(temporary), velocity)
            methods = [
                {
                    "method_id": "physics-null",
                    "role": "physics_null",
                    "prediction_artifact_sha256": "f" * 64,
                }
            ]
            method_set = {
                "pinned_before_study": True,
                "sha256": method_set_sha256(methods),
                "methods": methods,
            }
            with self.assertRaisesRegex(
                NativeProfileConvergenceError, "prediction artifact pin mismatch"
            ):
                build_profile_convergence_from_predictions(
                    study_id="bad-pin",
                    method_set=method_set,
                    case_order=[CASE_ID],
                    case_mappings={CASE_ID: self.mappings},
                    native_truth={CASE_ID: self.truth},
                    prediction_manifests={"physics-null": [path]},
                    definition=self.definition,
                    contract_proposal=CONTRACT,
                )

    def test_finalizer_rejects_unbound_or_path_leaking_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            methods = []
            manifests = {}
            for method_id, role, velocity_ratio in (
                ("physics-null", "physics_null", 1.0),
                (
                    "nearest-control",
                    "nearest_training_design_vector_control",
                    1.125,
                ),
            ):
                velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
                velocity[:, 0] = velocity_ratio * U_INF_M_PER_S
                manifest = _write_volume_manifest(
                    root / method_id, velocity
                )
                digest, _, _ = prediction_manifest_set_identity(
                    [CASE_ID], [manifest]
                )
                methods.append(
                    {
                        "method_id": method_id,
                        "role": role,
                        "prediction_artifact_sha256": digest,
                    }
                )
                manifests[method_id] = [manifest]
            method_set = {
                "pinned_before_study": True,
                "sha256": method_set_sha256(methods),
                "methods": methods,
            }
            result = build_profile_convergence_from_predictions(
                study_id="strict-finalizer-fixture",
                method_set=method_set,
                case_order=[CASE_ID],
                case_mappings={CASE_ID: self.mappings},
                native_truth={CASE_ID: self.truth},
                prediction_manifests=manifests,
                definition=self.definition,
                contract_proposal=CONTRACT,
            )
            base = {
                "study_id": result["study_id"],
                "method_set": result["constructed_loss_input"]["method_set"],
                "case_order": result["constructed_loss_input"]["case_order"],
                "losses": result["constructed_loss_input"]["losses"],
                "prediction_bindings": result["prediction_artifacts"],
                "geometric_case_evidence": result[
                    "geometric_assignment_convergence"
                ]["cases"],
                "method_audits": result["loss_construction_audit"],
                "contract_proposal": CONTRACT,
            }
            variants = []
            missing_predictions = copy.deepcopy(base)
            missing_predictions["prediction_bindings"] = []
            variants.append(("prediction bindings", missing_predictions))
            minimal_geometry = copy.deepcopy(base)
            minimal_geometry["geometric_case_evidence"] = [
                {
                    "case_id": CASE_ID,
                    "every_nested_geometric_assignment_invariant": True,
                }
            ]
            variants.append(("keys differ", minimal_geometry))
            missing_audits = copy.deepcopy(base)
            missing_audits["method_audits"] = []
            variants.append(("loss-construction audit", missing_audits))
            changed_chunk = copy.deepcopy(base)
            changed_chunk["prediction_bindings"][0]["case_manifests"][0][
                "ordered_chunk_sha256"
            ][0] = "f" * 64
            variants.append(("artifact digest", changed_chunk))
            changed_count = copy.deepcopy(base)
            changed_count["prediction_bindings"][0]["case_manifests"][0][
                "native_cell_count"
            ] = 999
            changed_count_digest = canonical_sha256(
                changed_count["prediction_bindings"][0]["case_manifests"]
            )
            changed_count["prediction_bindings"][0][
                "prediction_artifact_sha256"
            ] = changed_count_digest
            changed_count["method_set"]["methods"][0][
                "prediction_artifact_sha256"
            ] = changed_count_digest
            changed_count["method_set"]["sha256"] = method_set_sha256(
                changed_count["method_set"]["methods"]
            )
            variants.append(("not bound to its manifest", changed_count))
            absolute_path = copy.deepcopy(base)
            absolute_path["prediction_bindings"][0]["case_manifests"][0][
                "manifest_file"
            ] = "/tmp/leaked-manifest.json"
            variants.append(("basename", absolute_path))
            traversal_name = copy.deepcopy(base)
            traversal_name["prediction_bindings"][0]["case_manifests"][0][
                "manifest_file"
            ] = ".."
            variants.append(("basename", traversal_name))
            changed_truth = copy.deepcopy(base)
            changed_truth["method_audits"][1]["cases"][0]["native_truth"][
                "selected_values_sha256"
            ] = "0" * 64
            variants.append(("native truth identity differs", changed_truth))
            changed_support = copy.deepcopy(base)
            changed_support["method_audits"][1]["cases"][0]["line_support"][0][
                "contributing_arc_length_m"
            ] += 1.0e-6
            variants.append(("line-support geometry differs", changed_support))
            impossible_selection = copy.deepcopy(base)
            impossible_case_audit = impossible_selection["method_audits"][0][
                "cases"
            ][0]
            impossible_case_audit["required_unique_raw_cell_id_count"] = 999
            impossible_case_audit["prediction_manifest"][
                "selected_unique_raw_cell_id_count"
            ] = 999
            impossible_case_audit["native_truth"][
                "selected_unique_raw_cell_id_count"
            ] = 999
            variants.append(("exceeds the native cell count", impossible_selection))

            for expected, arguments in variants:
                with self.subTest(expected=expected), self.assertRaisesRegex(
                    NativeProfileConvergenceError, expected
                ):
                    finalize_profile_convergence_evidence(**arguments)

            with (
                mock.patch(
                    "reference.drivaerml.profile_convergence.OFFICIAL_CASE_ORDER",
                    (CASE_ID,),
                ),
                mock.patch(
                    "reference.drivaerml.profile_convergence_evaluator."
                    "OFFICIAL_CASE_ORDER",
                    (CASE_ID,),
                ),
                self.assertRaisesRegex(
                    NativeProfileConvergenceError,
                    "exact official evidence requires a native truth payload",
                ),
            ):
                finalize_profile_convergence_evidence(**base)

    def test_profile_losses_are_invariant_to_prediction_chunk_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_cell_ids = np.arange(NATIVE_CELL_COUNT, dtype=np.float64)
            velocity = np.zeros((NATIVE_CELL_COUNT, 3), dtype=np.float64)
            velocity[:, 0] = U_INF_M_PER_S * (
                0.85 + raw_cell_ids / 512.0
            )
            velocity[:, 1] = U_INF_M_PER_S * (
                np.remainder(raw_cell_ids, 11.0) / 211.0
            )
            velocity[:, 2] = U_INF_M_PER_S * (
                np.remainder(raw_cell_ids, 7.0) / 307.0
            )
            self.assertEqual(
                hashlib.sha256(
                    velocity.astype("<f8", copy=False).tobytes(order="C")
                ).hexdigest(),
                "6310d98419db9efe6d1a5a7f58d1ed3ef06aa3f959be9aed10fc43b76104c8be",
            )
            manifests = [
                _write_volume_manifest(
                    root / "two-chunks", velocity, partitions=(43, 85)
                ),
                _write_volume_manifest(
                    root / "three-chunks", velocity, partitions=(7, 13, 108)
                ),
            ]
            results = []
            for index, manifest in enumerate(manifests):
                digest, parsed, _ = prediction_manifest_set_identity(
                    [CASE_ID], [manifest]
                )
                gathered = gather_mapped_prediction_field(
                    parsed[0],
                    self.required_ids,
                    case_id=CASE_ID,
                    support_id="volume_native_cells",
                    field_name="UMeanTrim",
                    expected_total_row_count=NATIVE_CELL_COUNT,
                )
                np.testing.assert_array_equal(
                    gathered.values, velocity[self.required_ids]
                )
                methods = [
                    {
                        "method_id": "physics-null",
                        "role": "physics_null",
                        "prediction_artifact_sha256": digest,
                    }
                ]
                results.append(
                    build_profile_convergence_from_predictions(
                        study_id=f"chunk-partition-{index}",
                        method_set={
                            "pinned_before_study": True,
                            "sha256": method_set_sha256(methods),
                            "methods": methods,
                        },
                        case_order=[CASE_ID],
                        case_mappings={CASE_ID: self.mappings},
                        native_truth={CASE_ID: self.truth},
                        prediction_manifests={"physics-null": [manifest]},
                        definition=self.definition,
                        contract_proposal=CONTRACT,
                    )
                )

            self.assertEqual(
                results[0]["constructed_loss_input"]["losses"],
                results[1]["constructed_loss_input"]["losses"],
            )
            first_support = results[0]["loss_construction_audit"][0]["cases"][0][
                "line_support"
            ]
            second_support = results[1]["loss_construction_audit"][0]["cases"][0][
                "line_support"
            ]
            self.assertEqual(
                [item["loss"] for item in first_support],
                [item["loss"] for item in second_support],
            )


if __name__ == "__main__":
    unittest.main()
