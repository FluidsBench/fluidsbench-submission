from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from reference.scores import (
    composite_component_group_scores,
    composite_overall_score,
)

from scripts.generate_drivaerml_leaderboard_fixtures import (
    CASES_PER_CHUNK,
    DATASET_ROOT,
    GENERATOR_COMMAND,
    GENERATOR_REVISION,
    PRESSURE_SUPPORTS,
    SUBMISSION_SPLITS,
    SUBMISSIONS_ROOT,
    U_INF_M_PER_S,
    build_case,
    build_ground_truth_case,
    load_json,
    raw_macro_rmse,
    sha256_file,
    target_profile_metrics,
    velocity_contract,
)


ROOT = Path(__file__).resolve().parents[1]

if any(
    not (SUBMISSIONS_ROOT / submission_id / "submission.json").is_file()
    for submission_id in SUBMISSION_SPLITS
):
    raise unittest.SkipTest("requires retired DrivAerML leaderboard fixture packages")


def second_difference_is_nonzero(values: list[float]) -> bool:
    return any(
        abs(right - 2.0 * middle + left) > 1.0e-8
        for left, middle, right in zip(values, values[1:], values[2:])
    )


class DrivAerMLLeaderboardFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specification = load_json(DATASET_ROOT / "submission-spec.json")
        cls.diagnostics = load_json(
            DATASET_ROOT / cls.specification["profile_definition"]["file"]
        )
        cls.velocity_station_ids, cls.velocity_coordinates = velocity_contract(
            cls.specification
        )
        cls.pressure_station_ids = list(PRESSURE_SUPPORTS)

    def test_all_profiles_use_the_full_candidate_contract_and_are_curved(self) -> None:
        for submission_id in SUBMISSION_SPLITS:
            directory = SUBMISSIONS_ROOT / submission_id
            submission = load_json(directory / "submission.json")
            evidence = load_json(directory / "evaluation-evidence.json")
            index_path = directory / "profiles" / "index.json"
            index = load_json(index_path)
            with self.subTest(submission_id=submission_id):
                self.assertEqual(submission["evaluation"]["code_revision"], GENERATOR_REVISION)
                self.assertEqual(submission["evaluation"]["command"], GENERATOR_COMMAND)
                self.assertEqual(evidence["code_revision"], GENERATOR_REVISION)
                self.assertEqual(evidence["command"], GENERATOR_COMMAND)
                self.assertEqual(evidence["profile_index_sha256"], sha256_file(index_path))
                self.assertEqual(
                    submission["evaluation"]["evidence_sha256"],
                    sha256_file(directory / "evaluation-evidence.json"),
                )
                self.assertTrue(
                    all(len(chunk["case_ids"]) <= CASES_PER_CHUNK for chunk in index["chunks"])
                )

            for chunk_entry in index["chunks"]:
                chunk_path = directory / "profiles" / chunk_entry["file"]
                self.assertEqual(chunk_entry["sha256"], sha256_file(chunk_path))
                chunk = load_json(chunk_path)
                for case in chunk["cases"]:
                    series = {
                        (item["panel_id"], item["station_id"]): item
                        for item in case["series"]
                    }
                    self.assertEqual(len(series), 20)
                    for station_id in self.pressure_station_ids:
                        item = series[("pressure_profiles", station_id)]
                        self.assertGreaterEqual(len(item["coordinate"]), 100)
                        self.assertEqual(len(item["coordinate"]), len(item["prediction"]))
                        self.assertTrue(second_difference_is_nonzero(item["prediction"]))
                    for station_id in self.velocity_station_ids:
                        item = series[("velocity_profiles", station_id)]
                        expected_coordinate = self.velocity_coordinates[station_id]
                        self.assertEqual(item["coordinate"], expected_coordinate)
                        self.assertEqual(len(item["prediction"]), len(expected_coordinate))
                        self.assertTrue(second_difference_is_nonzero(item["prediction"]))

    def test_profile_metrics_are_physically_tied_to_dimensional_errors(self) -> None:
        for submission_id in SUBMISSION_SPLITS:
            submission = load_json(SUBMISSIONS_ROOT / submission_id / "submission.json")
            metrics = submission["metric_values"]
            cp_target, velocity_target = target_profile_metrics(metrics)
            with self.subTest(submission_id=submission_id):
                self.assertTrue(
                    math.isclose(
                        metrics["cp_cut_rmse"], cp_target, rel_tol=0.0, abs_tol=1.0e-7
                    )
                )
                self.assertTrue(
                    math.isclose(
                        metrics["velocity_profile_uinf_rmse"],
                        velocity_target,
                        rel_tol=0.0,
                        abs_tol=1.0e-7,
                    )
                )
                self.assertGreater(
                    metrics["velocity_profile_experimental_subset_uinf_rmse"], 0.0
                )
                self.assertLess(metrics["velocity_profile_uinf_rmse"], 0.355)
                for metric_id in (
                    "cd_r2",
                    "cl_r2",
                    "c_pitch_r2",
                    "velocity_profile_r2",
                    "cp_cut_r2",
                ):
                    self.assertTrue(math.isfinite(metrics[metric_id]))
                expected_groups = composite_component_group_scores(
                    metrics,
                    self.specification["overall_score_composite"],
                    self.specification["component_score_groups"],
                )
                for metric_id, expected in expected_groups.items():
                    self.assertTrue(
                        math.isclose(
                            metrics[metric_id], expected, rel_tol=0.0, abs_tol=1.0e-10
                        )
                    )
                self.assertTrue(
                    math.isclose(
                        metrics["overall_score"],
                        composite_overall_score(
                            metrics, self.specification["overall_score_composite"]
                        ),
                        rel_tol=0.0,
                        abs_tol=1.0e-10,
                    )
                )
                self.assertGreaterEqual(metrics["overall_score"], 0.0)
                self.assertLessEqual(metrics["overall_score"], 100.0)

                expected_cp_from_pressure_units = (
                    2.0
                    * metrics["drivaerml_surface_pressure_equal_entity_rmse"]
                    / U_INF_M_PER_S**2
                )
                self.assertTrue(
                    math.isclose(cp_target, expected_cp_from_pressure_units, abs_tol=1.0e-15)
                )

    def test_meshgraphnets_first_case_is_reproducible_from_the_generator(self) -> None:
        submission_id = "dummy-drivaerml-meshoperator-v1"
        directory = SUBMISSIONS_ROOT / submission_id
        submission = load_json(directory / "submission.json")
        split_entry = next(
            entry
            for entry in self.specification["splits"]
            if entry["id"] == submission["split_id"]
        )
        case_ids = load_json(DATASET_ROOT / split_entry["index_file"])["case_ids"]
        cp_target, velocity_target = target_profile_metrics(submission["metric_values"])
        cp_scale = cp_target / raw_macro_rmse(
            "pressure_profiles",
            self.pressure_station_ids,
            case_ids,
            submission_id,
            self.velocity_coordinates,
        )
        velocity_scale = velocity_target / raw_macro_rmse(
            "velocity_profiles",
            self.velocity_station_ids,
            case_ids,
            submission_id,
            self.velocity_coordinates,
        )
        expected_case, *_ = build_case(
            case_ids[0],
            submission_id,
            self.pressure_station_ids,
            self.velocity_station_ids,
            self.velocity_coordinates,
            cp_scale,
            velocity_scale,
        )
        first_chunk = json.loads(
            (directory / "profiles" / "chunk-000.json").read_text(encoding="utf-8")
        )
        self.assertEqual(first_chunk["cases"][0], expected_case)

    def test_ground_truth_case_uses_the_same_full_profile_support(self) -> None:
        case = build_ground_truth_case(
            "run_5",
            self.pressure_station_ids,
            self.velocity_station_ids,
            self.velocity_coordinates,
        )
        series = {
            (item["panel_id"], item["station_id"]): item
            for item in case["series"]
        }
        self.assertEqual(len(series), 20)
        for station_id in self.pressure_station_ids:
            item = series[("pressure_profiles", station_id)]
            self.assertGreaterEqual(len(item["coordinate"]), 100)
            self.assertEqual(len(item["coordinate"]), len(item["value"]))
            self.assertTrue(second_difference_is_nonzero(item["value"]))
        for station_id in self.velocity_station_ids:
            item = series[("velocity_profiles", station_id)]
            self.assertEqual(item["coordinate"], self.velocity_coordinates[station_id])
            self.assertEqual(len(item["coordinate"]), len(item["value"]))
            self.assertTrue(second_difference_is_nonzero(item["value"]))


if __name__ == "__main__":
    unittest.main()
