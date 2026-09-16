from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIRFRANS_ROOT = ROOT / "benchmark-specs" / "airfrans"
AIRFRANS_EXAMPLE_ROOT = ROOT / "examples" / "airfrans-profile-extraction"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AirfransContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = load_json(AIRFRANS_ROOT / "submission-spec.json")
        binding = self.specification["profile_definition"]
        self.profile_definition_path = AIRFRANS_ROOT / binding["file"]
        self.profile_definition = load_json(self.profile_definition_path)

    def test_profile_definition_is_pinned_and_matches_submission_panel(self) -> None:
        binding = self.specification["profile_definition"]
        self.assertEqual(binding["id"], "airfrans-boundary-layer-v1")
        self.assertEqual(binding["sha256"], sha256_file(self.profile_definition_path))
        self.assertEqual(binding["status"], "owner_review_required")

        source = self.profile_definition["source_library"]
        self.assertEqual(source["version"], "0.1.5.1")
        self.assertEqual(source["commit"], "d35d4035d8ba6fa98c1a6662be925c2fc777610b")
        self.assertEqual(
            source["source_file_sha256"],
            "77a74740ad482fa99adf8829ebdc4c4fe38b85de24d30d30a0c07806b1306c17",
        )

        extraction = self.profile_definition["extraction"]
        expected_arguments = {
            "y": 0.1,
            "extrado": True,
            "direction": "normals",
            "local_frame": False,
            "resolution": 1000,
            "compressible": False,
        }
        for call_name, reference in (
            ("ground_truth_call_arguments", True),
            ("prediction_call_arguments", False),
        ):
            self.assertEqual(
                extraction[call_name],
                {**expected_arguments, "reference": reference},
            )
        self.assertEqual(extraction["resolution_segments"], 1000)
        self.assertEqual(extraction["sample_count"], 1001)
        self.assertEqual(extraction["surface"], "extrados")
        self.assertEqual(extraction["line_direction"], "outward_airfoil_surface_normal")
        self.assertEqual(extraction["source_normal_orientation"], "inward_since_airfrans_0.1.4")
        self.assertEqual(
            extraction["normal_transform"],
            "negate_inward_normal_then_l2_normalize",
        )

        panel = self.specification["profile_panels"][0]
        self.assertTrue(panel["required"])
        self.assertFalse(panel["allow_unlisted_stations"])
        self.assertEqual(panel["profile_definition_id"], binding["id"])
        self.assertEqual(panel["metric_id"], "velocity_profile_r2")
        self.assertEqual(panel["sample_count"], 1001)
        self.assertEqual(panel["coordinate_interval"], [0.0, 0.1])
        self.assertEqual(
            panel["station_ids"],
            [station["id"] for station in self.profile_definition["stations"]],
        )
        self.assertEqual(
            panel["quantity_ids"],
            [quantity["id"] for quantity in self.profile_definition["quantities"]],
        )

        metric_binding = self.profile_definition["metric_binding"]
        aggregation = "equal_station_equal_quantity_mean_bounded_r2_across_split"
        self.assertEqual(metric_binding["metric_id"], panel["metric_id"])
        self.assertEqual(metric_binding["aggregation"], aggregation)
        self.assertEqual(
            metric_binding["grouping"],
            "calculate_one_r2_for_each_station_quantity_pair_across_all_cases_and_samples",
        )
        self.assertEqual(metric_binding["r2_bounds_before_averaging"], [0.0, 1.0])
        self.assertEqual(metric_binding["station_weighting"], "equal")
        self.assertEqual(metric_binding["quantity_weighting"], "equal")
        self.assertEqual(metric_binding["sample_weighting_within_group"], "equal")

        metric = next(
            item for item in self.specification["metrics"] if item["id"] == panel["metric_id"]
        )
        self.assertEqual(metric["aggregation"], aggregation)
        self.assertEqual(
            metric["weighting"],
            "stations_equal_quantities_equal_samples_within_each_station_quantity_group",
        )

    def test_profile_runtime_and_reference_fixture_are_hash_bound(self) -> None:
        runtime = self.profile_definition["sampling_runtime"]
        requirements_path = ROOT / runtime["requirements_file"]
        self.assertEqual(runtime["airfrans_distribution_version"], "0.1.5.1")
        self.assertEqual(runtime["airfrans_module_version"], "0.1.2")
        self.assertEqual(runtime["numpy_version"], "2.5.2")
        self.assertEqual(runtime["pyvista_version"], "0.48.4")
        self.assertEqual(runtime["vtk_version"], "9.6.2")
        self.assertEqual(runtime["requirements_sha256"], sha256_file(requirements_path))
        self.assertEqual(
            requirements_path.read_text(encoding="utf-8").splitlines(),
            [
                "airfrans==0.1.5.1",
                "numpy==2.5.2",
                "pyvista==0.48.4",
                "vtk==9.6.2",
            ],
        )

        binding = self.profile_definition["reference_fixture"]
        fixture_path = ROOT / binding["file"]
        self.assertEqual(fixture_path, AIRFRANS_EXAMPLE_ROOT / "example_extraction.json")
        self.assertEqual(binding["sha256"], sha256_file(fixture_path))

    def test_reference_fixture_has_valid_outward_normal_profiles(self) -> None:
        fixture_binding = self.profile_definition["reference_fixture"]
        fixture = load_json(ROOT / fixture_binding["file"])
        provenance = fixture["provenance"]
        runtime = self.profile_definition["sampling_runtime"]
        extraction = self.profile_definition["extraction"]
        validation = self.profile_definition["validation"]

        self.assertEqual(provenance["mode"], fixture_binding["mode"])
        self.assertEqual(
            provenance["profile_definition_id"],
            self.profile_definition["profile_definition_id"],
        )
        self.assertEqual(
            provenance["extractor_sha256"],
            sha256_file(AIRFRANS_EXAMPLE_ROOT / "extract.py"),
        )
        self.assertEqual(
            provenance["versions"],
            {
                "airfrans_distribution": runtime["airfrans_distribution_version"],
                "airfrans_module": runtime["airfrans_module_version"],
                "numpy": runtime["numpy_version"],
                "pyvista": runtime["pyvista_version"],
                "vtk": runtime["vtk_version"],
            },
        )
        self.assertEqual(
            provenance["airfrans_runtime_source_file_sha256"],
            runtime["airfrans_runtime_source_file_sha256"],
        )
        perfect_copy = provenance["perfect_copy_validation"]
        self.assertEqual(
            perfect_copy["absolute_tolerance"],
            validation["perfect_copy_absolute_tolerance"],
        )
        self.assertLessEqual(
            perfect_copy["maximum_absolute_difference"],
            perfect_copy["absolute_tolerance"],
        )

        expected_station_ids = [station["id"] for station in self.profile_definition["stations"]]
        sampling = provenance["sampling"]
        self.assertEqual(sampling["surface"], "extrados")
        self.assertEqual(sampling["normal_transform"], extraction["normal_transform"])
        self.assertEqual(
            [station["station_id"] for station in sampling["stations"]],
            expected_station_ids,
        )
        for station in sampling["stations"]:
            self.assertEqual(station["valid_sample_count"], extraction["sample_count"])
            self.assertEqual(len(station["origin_m"]), 3)
            self.assertEqual(len(station["outward_unit_normal"]), 2)
            self.assertTrue(
                math.isclose(
                    math.hypot(*station["outward_unit_normal"]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )

        self.assertEqual(len(fixture["cases"]), 1)
        self.assertEqual(fixture["cases"][0]["case_id"], fixture_binding["case_id"])
        series = fixture["cases"][0]["series"]
        expected_series = {
            (self.profile_definition["metric_binding"]["panel_id"], station_id, quantity["id"])
            for station_id in expected_station_ids
            for quantity in self.profile_definition["quantities"]
        }
        observed_series = {
            (entry["panel_id"], entry["station_id"], entry["quantity_id"])
            for entry in series
        }
        self.assertEqual(observed_series, expected_series)
        by_identity = {
            (entry["station_id"], entry["quantity_id"]): entry
            for entry in series
        }
        for entry in series:
            coordinates = entry["coordinate"]
            predictions = entry["prediction"]
            self.assertEqual(len(coordinates), extraction["sample_count"])
            self.assertEqual(len(predictions), extraction["sample_count"])
            self.assertEqual(coordinates[0], 0.0)
            self.assertEqual(coordinates[-1], extraction["line_length_m"])
            self.assertTrue(all(math.isfinite(value) for value in coordinates + predictions))
            spacing = extraction["line_length_m"] / extraction["resolution_segments"]
            self.assertTrue(
                all(
                    math.isclose(
                        coordinates[index + 1] - coordinates[index],
                        spacing,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    for index in range(len(coordinates) - 1)
                )
            )

        for station_id in expected_station_ids:
            velocity_x = by_identity[(station_id, "velocity_x_ratio")]["prediction"]
            velocity_y = by_identity[(station_id, "velocity_y_ratio")]["prediction"]
            self.assertFalse(
                any(x == 0.0 and y == 0.0 for x, y in zip(velocity_x, velocity_y, strict=True)),
                f"{station_id} contains VTK fill-value samples",
            )

    def test_split_ids_are_bound_to_the_official_manifest_member(self) -> None:
        expected_manifest_keys = {
            "full": "full_test",
            "scarce": "full_test",
            "reynolds_extrapolation": "reynolds_test",
            "aoa_extrapolation": "aoa_test",
        }
        spec_splits = {entry["id"]: entry for entry in self.specification["splits"]}
        for split_id, manifest_key in expected_manifest_keys.items():
            with self.subTest(split=split_id):
                entry = spec_splits[split_id]
                split_path = AIRFRANS_ROOT / entry["index_file"]
                split = load_json(split_path)
                source = split["source"]
                self.assertEqual(entry["sha256"], sha256_file(split_path))
                self.assertEqual(source["manifest_member"], "Dataset/manifest.json")
                self.assertEqual(
                    source["manifest_sha256"],
                    "3d1320005bcf3d94df80df2bf8338dd6ce6d394c6beafa14849b4ce1de7def13",
                )
                self.assertEqual(source["manifest_key"], manifest_key)
                self.assertEqual(
                    source["loader_commit"],
                    "d35d4035d8ba6fa98c1a6662be925c2fc777610b",
                )
                self.assertEqual(
                    source["loader_file_sha256"],
                    "21fee60ef9451fa5eb2634ccc80125acac2e387a1ba15e75ecc7ae5ec64c0675",
                )
                self.assertEqual(
                    source["case_ids_sha256"],
                    canonical_json_sha256(split["case_ids"]),
                )
                self.assertEqual(split["case_count"], len(split["case_ids"]))
        self.assertEqual(
            load_json(AIRFRANS_ROOT / spec_splits["full"]["index_file"])["case_ids"],
            load_json(AIRFRANS_ROOT / spec_splits["scarce"]["index_file"])["case_ids"],
        )

    def test_prototype_profiles_exercise_every_required_series(self) -> None:
        panel = self.specification["profile_panels"][0]
        expected_series = {
            (panel["id"], station_id, quantity_id)
            for station_id in panel["station_ids"]
            for quantity_id in panel["quantity_ids"]
        }
        fixture_root = ROOT / "submissions" / "airfrans"
        for submission_path in sorted(fixture_root.glob("*/submission.json")):
            with self.subTest(submission=submission_path.parent.name):
                submission = load_json(submission_path)
                self.assertEqual(submission["approval"]["status"], "prototype")
                index = load_json(submission_path.parent / submission["profile_data"]["index_file"])
                for chunk_entry in index["chunks"]:
                    chunk = load_json(submission_path.parent / "profiles" / chunk_entry["file"])
                    for case in chunk["cases"]:
                        observed = {
                            (series["panel_id"], series["station_id"], series["quantity_id"])
                            for series in case["series"]
                        }
                        self.assertEqual(observed, expected_series)
                        self.assertTrue(
                            all(series["coordinate"] == [0.0, 0.05, 0.1] for series in case["series"])
                        )

    def test_leaderboard_panel_matches_the_airfrans_contract(self) -> None:
        manifest = load_json(ROOT / "leaderboard" / "manifest.json")
        dataset = next(item for item in manifest["datasets"] if item["slug"] == "airfrans")
        self.assertEqual(dataset["profile_definition"], self.specification["profile_definition"])
        self.assertEqual([panel["id"] for panel in dataset["diagnostic_panels"]], ["velocity_profiles"])
        panel = dataset["diagnostic_panels"][0]
        self.assertEqual(panel["profile_definition_id"], "airfrans-boundary-layer-v1")
        self.assertEqual(
            [quantity["id"] for quantity in panel["quantities"]],
            ["velocity_x_ratio", "velocity_y_ratio"],
        )
        self.assertEqual(
            [station["id"] for station in panel["stations"]],
            ["upper_x_0_25c", "upper_x_0_50c", "upper_x_0_75c", "upper_x_0_95c"],
        )
        self.assertTrue(panel["required"])
        self.assertFalse(panel["allow_unlisted_stations"])
        self.assertEqual(
            dataset["metric_definition_overrides"]["velocity_profile_r2"]["label"],
            "Balanced velocity profile score",
        )


if __name__ == "__main__":
    unittest.main()
