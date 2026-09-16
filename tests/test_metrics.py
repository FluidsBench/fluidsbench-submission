from __future__ import annotations

import unittest

import numpy as np

from reference.metrics import (
    aggregate_case_metrics,
    arc_length_rmse,
    field_rrmse,
    r2_score,
    relative_l1,
    relative_l2,
    scalar_rrmse,
    unique_probe_rmse,
    weighted_mae,
    weighted_mse,
    weighted_rmse,
)
from reference.scores import (
    arithmetic_mean,
    composite_component_group_scores,
    composite_overall_score,
    legacy_aero_scores,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = np.array([1.0, 2.0, 4.0])
        self.prediction = np.array([2.0, 2.0, 1.0])
        self.weights = np.array([1.0, 2.0, 1.0])

    def test_weighted_errors(self) -> None:
        self.assertAlmostEqual(weighted_mae(self.truth, self.prediction, self.weights), 1.0)
        self.assertAlmostEqual(weighted_mse(self.truth, self.prediction, self.weights), 2.5)
        self.assertAlmostEqual(weighted_rmse(self.truth, self.prediction, self.weights), np.sqrt(2.5))
        self.assertAlmostEqual(relative_l1(self.truth, self.prediction, self.weights), 100.0 / 9.0 * 4.0)
        self.assertAlmostEqual(relative_l2(self.truth, self.prediction, self.weights), 100.0 * np.sqrt(10.0 / 25.0))

    def test_r2_uses_weighted_mean(self) -> None:
        expected = 1.0 - 10.0 / 4.75
        self.assertAlmostEqual(r2_score(self.truth, self.prediction, self.weights), expected)

    def test_rrmse(self) -> None:
        truth = np.array([[1.0, 2.0], [2.0, 4.0]])
        prediction = np.array([[2.0, 2.0], [2.0, 6.0]])
        self.assertAlmostEqual(field_rrmse(truth, prediction), np.sqrt((0.125 + 0.125) / 2.0))
        self.assertAlmostEqual(scalar_rrmse([1.0, 2.0], [2.0, 2.0]), np.sqrt(0.5))

    def test_macro_average(self) -> None:
        cases = [([1.0], [2.0], None), ([10.0], [12.0], None)]
        self.assertAlmostEqual(aggregate_case_metrics(cases, weighted_mae), 1.5)

    def test_profile_rmse_reductions(self) -> None:
        coordinate = [0.0, 1.0, 3.0]
        truth = [0.0, 0.0, 0.0]
        prediction = [0.0, 2.0, 2.0]
        self.assertAlmostEqual(
            arc_length_rmse(coordinate, truth, prediction),
            np.sqrt(10.0 / 3.0),
        )
        self.assertAlmostEqual(
            unique_probe_rmse([0.0, 0.0, 0.0], [1.0, 2.0, 2.0]),
            np.sqrt(3.0),
        )

    def test_physics_null_skill_is_not_clipped(self) -> None:
        declaration = {
            "operation": "weighted_component_scores",
            "components": [
                {
                    "metric_id": "error",
                    "weight": 1.0,
                    "transform": "physics_null_skill",
                    "baseline_error": 2.0,
                }
            ],
        }
        self.assertEqual(composite_overall_score({"error": 0.0}, declaration), 100.0)
        self.assertEqual(composite_overall_score({"error": 2.0}, declaration), 0.0)
        self.assertEqual(composite_overall_score({"error": 3.0}, declaration), -50.0)

    def test_score_reductions(self) -> None:
        values = {
            "surface_pressure_rel_l2": 7.5,
            "surface_wall_shear_rel_l2": 10.0,
            "volume_velocity_rel_l2": 6.0,
            "volume_pressure_rel_l2": 7.5,
            "cd_r2": 0.9,
            "cl_r2": 0.8,
            "velocity_profile_r2": 0.7,
            "cp_cut_r2": 0.6,
        }
        scores = legacy_aero_scores(values)
        self.assertAlmostEqual(scores["field_score"], 50.0)
        self.assertAlmostEqual(scores["force_score"], 86.0)
        self.assertAlmostEqual(scores["diagnostic_score"], 66.0)
        self.assertAlmostEqual(scores["overall_score"], 63.0)
        flow_domain_values = dict(values)
        flow_domain_values["flow_domain_velocity_rel_l2"] = flow_domain_values.pop("volume_velocity_rel_l2")
        flow_domain_values["flow_domain_pressure_rel_l2"] = flow_domain_values.pop("volume_pressure_rel_l2")
        self.assertEqual(legacy_aero_scores(flow_domain_values), scores)
        self.assertAlmostEqual(arithmetic_mean({"a": 1.0, "b": 3.0}, ["a", "b"]), 2.0)

        overall_declaration = {
            "operation": "weighted_component_scores",
            "components": [
                {"metric_id": "surface_pressure_rel_l2", "weight": 0.15, "transform": "bounded_error", "cap": 15.0},
                {"metric_id": "surface_wall_shear_rel_l2", "weight": 0.10, "transform": "bounded_error", "cap": 20.0},
                {"metric_id": "volume_velocity_rel_l2", "weight": 0.15, "transform": "bounded_error", "cap": 12.0},
                {"metric_id": "volume_pressure_rel_l2", "weight": 0.10, "transform": "bounded_error", "cap": 15.0},
                {"metric_id": "cd_r2", "weight": 0.15, "transform": "bounded_quality"},
                {"metric_id": "cl_r2", "weight": 0.10, "transform": "bounded_quality"},
                {"metric_id": "velocity_profile_r2", "weight": 0.15, "transform": "bounded_quality"},
                {"metric_id": "cp_cut_r2", "weight": 0.10, "transform": "bounded_quality"},
            ],
        }
        group_declaration = {
            "operation": "normalized_weighted_component_scores",
            "groups": [
                {
                    "metric_id": "field_score",
                    "component_metric_ids": [
                        "surface_pressure_rel_l2",
                        "surface_wall_shear_rel_l2",
                        "volume_velocity_rel_l2",
                        "volume_pressure_rel_l2",
                    ],
                },
                {
                    "metric_id": "force_score",
                    "component_metric_ids": ["cd_r2", "cl_r2"],
                },
                {
                    "metric_id": "diagnostic_score",
                    "component_metric_ids": ["velocity_profile_r2", "cp_cut_r2"],
                },
            ],
        }
        declared_scores = composite_component_group_scores(
            values,
            overall_declaration,
            group_declaration,
        )
        self.assertEqual(declared_scores, {key: scores[key] for key in declared_scores})
        self.assertAlmostEqual(
            composite_overall_score(values, overall_declaration),
            scores["overall_score"],
        )

    def test_single_component_group_score(self) -> None:
        overall_declaration = {
            "operation": "weighted_component_scores",
            "components": [
                {"metric_id": "error", "weight": 0.67, "transform": "bounded_error", "cap": 20.0},
                {"metric_id": "quality", "weight": 0.22, "transform": "bounded_quality"},
                {"metric_id": "profile", "weight": 0.11, "transform": "bounded_quality"},
            ],
        }
        group_declaration = {
            "operation": "normalized_weighted_component_scores",
            "groups": [
                {"metric_id": "field_score", "component_metric_ids": ["error"]},
                {"metric_id": "force_score", "component_metric_ids": ["quality"]},
                {"metric_id": "diagnostic_score", "component_metric_ids": ["profile"]},
            ],
        }
        self.assertEqual(
            composite_component_group_scores(
                {"error": 10.0, "quality": 0.8, "profile": 0.7},
                overall_declaration,
                group_declaration,
            ),
            {"field_score": 50.0, "force_score": 80.0, "diagnostic_score": 70.0},
        )
        incomplete_groups = dict(group_declaration)
        incomplete_groups["groups"] = group_declaration["groups"][:-1]
        with self.assertRaisesRegex(ValueError, "do not cover overall components"):
            composite_component_group_scores(
                {"error": 10.0, "quality": 0.8, "profile": 0.7},
                overall_declaration,
                incomplete_groups,
            )

    def test_dataset_declared_composite_score(self) -> None:
        declaration = {
            "operation": "weighted_component_scores",
            "components": [
                {"metric_id": "error", "weight": 0.75, "transform": "bounded_error", "cap": 20.0},
                {"metric_id": "quality", "weight": 0.25, "transform": "bounded_quality"},
            ],
        }
        self.assertAlmostEqual(
            composite_overall_score({"error": 10.0, "quality": 0.8}, declaration),
            57.5,
        )
        self.assertAlmostEqual(
            composite_overall_score({"error": 40.0, "quality": 2.0}, declaration),
            25.0,
        )

        invalid = dict(declaration)
        invalid["components"] = [dict(declaration["components"][0], weight=0.5)]
        with self.assertRaises(ValueError):
            composite_overall_score({"error": 10.0}, invalid)

    def test_scope_unavailable_components_keep_their_weights_at_zero(self) -> None:
        declaration = {
            "operation": "weighted_component_scores",
            "components": [
                {"metric_id": "surface", "weight": 0.35, "transform": "bounded_quality"},
                {"metric_id": "force", "weight": 0.25, "transform": "bounded_quality"},
                {"metric_id": "volume", "weight": 0.25, "transform": "bounded_quality"},
                {"metric_id": "velocity", "weight": 0.15, "transform": "bounded_quality"},
            ],
        }
        groups = {
            "operation": "normalized_weighted_component_scores",
            "groups": [
                {"metric_id": "field_score", "component_metric_ids": ["surface", "volume"]},
                {"metric_id": "force_score", "component_metric_ids": ["force"]},
                {"metric_id": "diagnostic_score", "component_metric_ids": ["velocity"]},
            ],
        }
        # The unavailable raw values are intentionally absent.  With the two
        # available components perfect, their 0.60 published weight is the
        # exact overall ceiling; the groups still expose the zero components.
        values = {"surface": 1.0, "force": 1.0}
        unavailable = ("volume", "velocity")
        self.assertEqual(
            composite_overall_score(
                values, declaration, fixed_zero_component_ids=unavailable
            ),
            60.0,
        )
        self.assertEqual(
            composite_component_group_scores(
                values,
                declaration,
                groups,
                fixed_zero_component_ids=unavailable,
            ),
            {"field_score": 58.333333333333336, "force_score": 100.0, "diagnostic_score": 0.0},
        )

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            relative_l2([0.0], [1.0])
        with self.assertRaises(ValueError):
            r2_score([1.0, 1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            weighted_mae([1.0], [1.0], [-1.0])
        with self.assertRaises(ValueError):
            weighted_mae([np.nan], [1.0])


if __name__ == "__main__":
    unittest.main()
