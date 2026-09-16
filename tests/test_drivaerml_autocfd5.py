from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from reference.drivaerml.autocfd5 import (
    AutoCFD5Error,
    CP_PANEL_COUNT,
    CP_PANEL_MEMBERSHIP_COUNT,
    CP_PROBE_COUNT,
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    U_INF_M_PER_S,
    VELOCITY_LINE_COUNT,
    VELOCITY_SAMPLE_COUNT,
    CpProbeMappingEvidence,
    VelocityCellAssignmentEvidence,
    cp_from_kinematic_pressure,
    cp_probe_rmse,
    cp_values_from_mappings,
    load_autocfd5_definition,
    sample_native_cell_data_zeroth_order,
    validate_cp_mapping_evidence,
    validate_velocity_assignment_evidence,
    velocity_magnitude_ratio,
    velocity_profile_rmse,
    velocity_ratios_from_assignments,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"
SOURCE_HASHES = ("0" * 64,)


class DrivAerMLAutoCFD5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_autocfd5_definition(PROFILE)

    def _cp_mapping(
        self,
        case_id: str,
        position: int,
        *,
        valid: bool = True,
    ) -> CpProbeMappingEvidence:
        probe = self.definition.cp_probes[position]
        return CpProbeMappingEvidence(
            case_id=case_id,
            autocfd_probe_id=probe.autocfd_probe_id,
            valid=valid,
            reason="" if valid else "owner_review_required",
            mapped_point_m=probe.point_m if valid else None,
            raw_stl_triangle_id=1000 + position if valid else None,
            raw_vtk_polygon_id=position if valid else None,
            nominal_displacement_m=0.001 if valid else None,
            bridge_distance_m=0.0002 if valid else None,
            bridge_abs_normal_dot=0.99 if valid else None,
            source_sha256=SOURCE_HASHES,
        )

    def _velocity_assignment(
        self,
        case_id: str,
        position: int,
        *,
        valid: bool = True,
    ) -> VelocityCellAssignmentEvidence:
        sample = self.definition.velocity_samples[position]
        return VelocityCellAssignmentEvidence(
            case_id=case_id,
            profile_id=sample.profile_id,
            sample_index=sample.sample_index,
            point_m=sample.point_m,
            distance_m=sample.distance_m,
            valid=valid,
            reason="" if valid else "inside_morphed_solid",
            raw_vtk_cell_id=position if valid else None,
            candidate_count=2 if valid else 0,
            geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
            source_sha256=SOURCE_HASHES,
        )

    def test_frozen_profile_and_all_hashed_registries_load(self) -> None:
        definition = self.definition
        self.assertEqual(len(definition.cp_probes), CP_PROBE_COUNT)
        self.assertEqual(
            len(definition.cp_panel_memberships), CP_PANEL_MEMBERSHIP_COUNT
        )
        self.assertEqual(
            len({row.panel_id for row in definition.cp_panel_memberships}),
            CP_PANEL_COUNT,
        )
        # The eight repeated display rows remain intentional panel membership,
        # while the ranked identity registry contains exactly 209 unique taps.
        panel_probe_ids = [
            row.autocfd_probe_id for row in definition.cp_panel_memberships
        ]
        self.assertEqual(len(panel_probe_ids) - len(set(panel_probe_ids)), 8)
        self.assertEqual(
            len({probe.autocfd_probe_id for probe in definition.cp_probes}),
            CP_PROBE_COUNT,
        )
        self.assertEqual(len(definition.cp_component_rules), CP_PROBE_COUNT)
        self.assertEqual(
            {rule.owner_review_status for rule in definition.cp_component_rules},
            {"pending_visual_signoff"},
        )

        self.assertEqual(len(definition.velocity_lines), VELOCITY_LINE_COUNT)
        self.assertEqual(len(definition.velocity_samples), VELOCITY_SAMPLE_COUNT)
        self.assertEqual(
            [line.profile_id for line in definition.velocity_lines],
            [
                "V1", "V2", "V3", "V4", "V5", "V6",
                "U1", "U2", "U3", "U4", "U5", "U6",
                "L1", "R1", "R2", "R3",
            ],
        )
        self.assertEqual(
            sum(line.sample_count for line in definition.velocity_lines),
            VELOCITY_SAMPLE_COUNT,
        )
        self.assertEqual(len(definition.source_sha256), 5)

    def test_cp_formula_uses_kinematic_pressure_and_fixed_uinf(self) -> None:
        dynamic_scale = 0.5 * U_INF_M_PER_S**2
        pressure = np.asarray([-dynamic_scale, 0.0, 2.0 * dynamic_scale])
        np.testing.assert_allclose(
            cp_from_kinematic_pressure(pressure),
            [-1.0, 0.0, 2.0],
            rtol=0.0,
            atol=2.0e-16,
        )
        self.assertEqual(cp_from_kinematic_pressure(dynamic_scale), 1.0)
        with self.assertRaisesRegex(AutoCFD5Error, "finite"):
            cp_from_kinematic_pressure([float("nan")])
        with self.assertRaisesRegex(AutoCFD5Error, "positive"):
            cp_from_kinematic_pressure([1.0], freestream_velocity_m_per_s=0.0)

    def test_cp_mapping_coverage_is_exact_duplicate_free_and_ordered(self) -> None:
        cases = ("run_1", "run_44")
        mappings = [
            self._cp_mapping(case_id, position)
            for case_id in reversed(cases)
            for position in reversed(range(CP_PROBE_COUNT))
        ]
        ordered = validate_cp_mapping_evidence(
            mappings,
            self.definition,
            expected_case_ids=cases,
        )
        self.assertEqual(tuple(ordered), cases)
        self.assertEqual(
            [item.autocfd_probe_id for item in ordered["run_1"]],
            [probe.autocfd_probe_id for probe in self.definition.cp_probes],
        )

        with self.assertRaisesRegex(AutoCFD5Error, "missing=1"):
            validate_cp_mapping_evidence(
                mappings[:-1], self.definition, expected_case_ids=cases
            )
        with self.assertRaisesRegex(AutoCFD5Error, "duplicate Cp"):
            validate_cp_mapping_evidence(
                mappings + [mappings[0]], self.definition, expected_case_ids=cases
            )

    def test_cp_invalid_mapping_is_retained_but_cannot_be_ranked(self) -> None:
        mappings = [self._cp_mapping("run_1", i) for i in range(CP_PROBE_COUNT)]
        mappings[17] = self._cp_mapping("run_1", 17, valid=False)
        with self.assertRaisesRegex(AutoCFD5Error, "invalid mappings"):
            validate_cp_mapping_evidence(
                mappings, self.definition, expected_case_ids=("run_1",)
            )
        audited = validate_cp_mapping_evidence(
            mappings,
            self.definition,
            expected_case_ids=("run_1",),
            require_all_valid=False,
        )
        self.assertFalse(audited["run_1"][17].valid)

        violating = [self._cp_mapping("run_1", i) for i in range(CP_PROBE_COUNT)]
        violating[0] = replace(violating[0], bridge_distance_m=0.00201)
        with self.assertRaisesRegex(AutoCFD5Error, "bridge distance"):
            validate_cp_mapping_evidence(
                violating, self.definition, expected_case_ids=("run_1",)
            )

    def test_cp_zeroth_order_sampling_and_unique_probe_rmse(self) -> None:
        mappings = tuple(
            self._cp_mapping("run_1", i) for i in range(CP_PROBE_COUNT)
        )
        target_cp = np.linspace(-2.0, 1.0, CP_PROBE_COUNT)
        native_pressure = 0.5 * U_INF_M_PER_S**2 * target_cp
        np.testing.assert_allclose(
            cp_values_from_mappings(native_pressure, mappings),
            target_cp,
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        self.assertAlmostEqual(cp_probe_rmse(target_cp + 0.25, target_cp), 0.25)
        with self.assertRaisesRegex(AutoCFD5Error, "shape"):
            cp_probe_rmse(target_cp[:-1], target_cp[:-1])

    def test_velocity_assignment_coverage_and_grid_identity_are_exact(self) -> None:
        assignments = [
            self._velocity_assignment("run_44", position)
            for position in reversed(range(VELOCITY_SAMPLE_COUNT))
        ]
        ordered = validate_velocity_assignment_evidence(
            assignments,
            self.definition,
            expected_case_ids=("run_44",),
        )
        self.assertEqual(len(ordered["run_44"]), VELOCITY_SAMPLE_COUNT)
        self.assertEqual(ordered["run_44"][0].profile_id, "V1")
        self.assertEqual(ordered["run_44"][-1].profile_id, "R3")

        with self.assertRaisesRegex(AutoCFD5Error, "missing=1"):
            validate_velocity_assignment_evidence(
                assignments[:-1],
                self.definition,
                expected_case_ids=("run_44",),
            )
        with self.assertRaisesRegex(AutoCFD5Error, "duplicate velocity"):
            validate_velocity_assignment_evidence(
                assignments + [assignments[0]],
                self.definition,
                expected_case_ids=("run_44",),
            )

        displaced = assignments.copy()
        displaced[0] = replace(
            displaced[0],
            point_m=(displaced[0].point_m[0] + 1.0e-5,) + displaced[0].point_m[1:],
        )
        with self.assertRaisesRegex(AutoCFD5Error, "fixed grid"):
            validate_velocity_assignment_evidence(
                displaced,
                self.definition,
                expected_case_ids=("run_44",),
            )

    def test_invalid_velocity_rows_fail_ranked_validation(self) -> None:
        assignments = [
            self._velocity_assignment("run_1", position)
            for position in range(VELOCITY_SAMPLE_COUNT)
        ]
        v1_count = self.definition.velocity_lines[0].sample_count
        for position in range(v1_count):
            assignments[position] = self._velocity_assignment(
                "run_1", position, valid=False
            )
        with self.assertRaisesRegex(
            AutoCFD5Error, "cannot contain unresolved invalid mappings"
        ):
            validate_velocity_assignment_evidence(
                assignments,
                self.definition,
                expected_case_ids=("run_1",),
            )
        with self.assertRaisesRegex(AutoCFD5Error, "no positive contributing"):
            validate_velocity_assignment_evidence(
                assignments,
                self.definition,
                expected_case_ids=("run_1",),
                allow_incomplete_audit=True,
            )
        with self.assertRaisesRegex(AutoCFD5Error, "must be Boolean"):
            validate_velocity_assignment_evidence(
                assignments,
                self.definition,
                expected_case_ids=("run_1",),
                allow_incomplete_audit=1,  # type: ignore[arg-type]
            )

        distance = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        truth = np.zeros(5)
        prediction = np.asarray([1.0, 1.0, np.nan, 3.0, 3.0])
        valid = np.asarray([True, True, False, True, True])
        self.assertAlmostEqual(
            velocity_profile_rmse(distance, prediction, truth, valid),
            math.sqrt(5.0),
        )
        with self.assertRaisesRegex(AutoCFD5Error, "no positive contributing"):
            velocity_profile_rmse(
                distance,
                np.asarray([1.0, 1.0, 1.0, 3.0, 3.0]),
                truth,
                np.asarray([True, False, True, False, True]),
            )

    def test_native_cell_sampling_is_direct_and_preserves_assignment_rows(self) -> None:
        cell_values = np.asarray(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, U_INF_M_PER_S],
                [U_INF_M_PER_S, 0.0, 0.0],
            ]
        )
        gathered = sample_native_cell_data_zeroth_order(cell_values, [2, 0, 2])
        np.testing.assert_array_equal(gathered, cell_values[[2, 0, 2]])
        np.testing.assert_allclose(
            velocity_magnitude_ratio(gathered),
            [1.0, 5.0 / U_INF_M_PER_S, 1.0],
        )

        samples = self.definition.velocity_samples[:3]
        assignments = (
            VelocityCellAssignmentEvidence(
                case_id="run_1",
                profile_id=samples[0].profile_id,
                sample_index=samples[0].sample_index,
                point_m=samples[0].point_m,
                distance_m=samples[0].distance_m,
                valid=True,
                reason="",
                raw_vtk_cell_id=0,
                candidate_count=1,
                geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                source_sha256=SOURCE_HASHES,
            ),
            VelocityCellAssignmentEvidence(
                case_id="run_1",
                profile_id=samples[1].profile_id,
                sample_index=samples[1].sample_index,
                point_m=samples[1].point_m,
                distance_m=samples[1].distance_m,
                valid=False,
                reason="outside_released_fluid_domain",
                raw_vtk_cell_id=None,
                candidate_count=0,
                geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                source_sha256=SOURCE_HASHES,
            ),
            VelocityCellAssignmentEvidence(
                case_id="run_1",
                profile_id=samples[2].profile_id,
                sample_index=samples[2].sample_index,
                point_m=samples[2].point_m,
                distance_m=samples[2].distance_m,
                valid=True,
                reason="",
                raw_vtk_cell_id=1,
                candidate_count=2,
                geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                source_sha256=SOURCE_HASHES,
            ),
        )
        ratios, valid = velocity_ratios_from_assignments(cell_values, assignments)
        np.testing.assert_array_equal(valid, [True, False, True])
        self.assertAlmostEqual(ratios[0], 5.0 / U_INF_M_PER_S)
        self.assertTrue(math.isnan(ratios[1]))
        self.assertEqual(ratios[2], 1.0)


if __name__ == "__main__":
    unittest.main()
