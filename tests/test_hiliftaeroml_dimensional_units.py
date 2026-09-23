"""Known-answer SI exports and preservation of the registered native results."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from reference.hiliftaeroml import dimensional_units as units
from scripts import assemble_hiliftaeroml_schema_v3_candidate as assembler
from scripts import validate_submission as validator

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


class HiLiftDimensionalUnitsTests(unittest.TestCase):
    def test_independent_full_geotransolver_known_answers(self):
        # Current ICLR manuscript's corrected dimensional RMSE table.
        path = ROOT / "submissions/hiliftaeroml/hiliftaeroml-geotransolver-full360-candidate-v1/submission.json"
        values = load(path)["metric_values"]
        for metric, expected in {
            "surface_pressure_rmse": 87.3107201811254,
            "surface_wall_shear_rmse": 0.6836418675218421,
            "volume_pressure_rmse": 165.9088033434961,
            "volume_velocity_rmse": 2.6561332906973014,
        }.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(values[metric], expected, places=10)

    def test_invalid_or_unrelated_input_is_rejected(self):
        for value in (-1, True, float("nan"), float("inf"), "1", None, 1e308):
            with self.subTest(value=value), self.assertRaises(ValueError):
                units.native_error_to_si("surface_pressure_rmse", value)
        with self.assertRaises(ValueError):
            units.native_error_to_si("surface_pressure_rel_l2", 1)
        self.assertEqual(units.native_error_to_si("surface_pressure_rmse", 0), 0)

    def test_assembler_converts_dimensional_errors_and_preserves_relative_statistics(self):
        concise = {"domains": {
            "surface": {"fields": {name: {"dimensional": {"mae": 1, "rmse": 2}}
                                   for name in ("pressure", "tau_wall")}},
            "volume": {"fields": {name: {"dimensional": {"mae": 100, "rmse": 200}}
                                  for name in ("pressure", "velocity")}},
        }}
        sums = {"sum_squared_error": 1.0, "sum_squared_truth": 4.0, "weight_sum": 2.0}
        for domain, support, weighting, expected_factor in (
            ("surface", assembler.SURFACE_SUPPORT_ID, "dual_area", 574.5631077637795),
            ("volume", assembler.VOLUME_SUPPORT_ID, "equal_valid_node", 574.5631077637795),
        ):
            relative = f"{domain}_pressure_rel_l2"
            native = {"coverage": {"point_count": 2}, "weightings": {weighting: {"pressure": {
                "metrics": {"relative_l2_percent": 50.0}, "sufficient_statistics": sums}}}}
            bindings = {metric: {"case_evidence": "metric_value"}
                        for metric in units.METRIC_FACTORS if metric.startswith(domain)}
            bindings[relative] = {"case_evidence": "metric_value", "weighting": "uniform", "dataset_weighting": weighting}
            record, values = assembler._field_support_record(
                case_id="known-answer", support_id=support, native_support=native,
                concise_case=concise, bindings=bindings)
            self.assertEqual(values[relative], 50.0)
            self.assertEqual(record["metric_sufficient_statistics"][relative]["numerator"], 1.0)
            self.assertEqual(record["metric_sufficient_statistics"][relative]["denominator"], 4.0)
            magnitude = 1 if domain == "surface" else 100
            self.assertAlmostEqual(values[f"{domain}_pressure_mae"], magnitude * expected_factor)
            self.assertAlmostEqual(values[f"{domain}_pressure_rmse"], 2 * magnitude * expected_factor)
            if domain == "surface":
                self.assertAlmostEqual(values["surface_wall_shear_rmse"], 1149.126215527559)
            else:
                self.assertAlmostEqual(values["volume_velocity_mae"], 2.54)
                self.assertAlmostEqual(values["volume_velocity_rmse"], 5.08)

    def test_case_macro_average_and_double_conversion_guard(self):
        metrics = {key: 2.0 for key in units.METRIC_FACTORS}
        metrics.update(overall_score=42.0, surface_pressure_rel_l2=10.0)
        submission = {"dataset_id": "hiliftaeroml", "metric_values": metrics}
        evidence = copy.deepcopy(submission)
        cases = {"metric_values": copy.deepcopy(metrics), "cases": [
            {"case_id": name, "supports": [{"support_count": count,
             "metric_values": {**{key: value for key in units.METRIC_FACTORS}, "surface_pressure_rel_l2": 10.0},
             "metric_sufficient_statistics": {"numerator": count, "denominator": count * 100}}]}
            for name, count, value in (("small", 2, 1.0), ("large", 200, 3.0))
        ]}
        originals = copy.deepcopy((submission, evidence, cases))
        result, proof, corrected = units.corrected_documents(submission, evidence, cases)
        self.assertEqual((submission, evidence, cases), originals)
        self.assertEqual(result["metric_values"]["overall_score"], 42.0)
        self.assertEqual(result["metric_values"]["surface_pressure_rel_l2"], 10.0)
        self.assertAlmostEqual(result["metric_values"]["volume_velocity_rmse"], 0.0508)
        self.assertAlmostEqual(result["metric_values"]["surface_pressure_rmse"], 1149.126215527559)
        for before, after in zip(cases["cases"], corrected["cases"]):
            self.assertEqual(before["supports"][0]["metric_sufficient_statistics"], after["supports"][0]["metric_sufficient_statistics"])
        with self.assertRaisesRegex(ValueError, "already recorded"):
            units.corrected_documents(result, proof, corrected)
        cases["cases"][0]["supports"][0]["metric_values"].pop("surface_pressure_mae")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            units.corrected_documents(submission, evidence, cases)

    def test_all_registered_results_preserve_every_non_dimensional_metric(self):
        registry = load(ROOT / units.CORRECTION_PATH)
        self.assertEqual(len(registry["records"]), 23)
        self.assertEqual(sum(v["corrected"]["case_count"] for v in registry["records"].values()), 8142)
        original_order, corrected_order = [], []
        for config in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS:
            binding = config["binding"]
            path = ROOT / binding["submission_path"]
            corrected = load(path)
            source_path, _, new_binding = units.corrected_registration_view(ROOT, path, corrected, binding)
            original = load(source_path)
            self.assertEqual(new_binding["dimensional_export_version"], units.CONTRACT_ID)
            with self.subTest(submission_id=binding["submission_id"]):
                for metric, value in original["metric_values"].items():
                    if metric in units.METRIC_FACTORS:
                        self.assertTrue(math.isclose(corrected["metric_values"][metric], value * units.METRIC_FACTORS[metric], rel_tol=1e-12, abs_tol=1e-12))
                    else:
                        self.assertEqual(corrected["metric_values"][metric], value)
                original_order.append((original["split_id"], -original["metric_values"]["overall_score"], binding["submission_id"]))
                corrected_order.append((corrected["split_id"], -corrected["metric_values"]["overall_score"], binding["submission_id"]))
        self.assertEqual(sorted(original_order), sorted(corrected_order))

    def test_modified_registry_original_or_corrected_bytes_fail_closed(self):
        binding = validator.HILIFT_REGISTERED_PREVIEW_CONFIGS[0]["binding"]
        path = ROOT / binding["submission_path"]
        submission = load(path)
        source = load(ROOT / units.CORRECTION_PATH)["records"][binding["submission_id"]]["source"]
        real_digest = units.digest
        paths = [ROOT / units.CONTRACT_PATH, ROOT / units.CORRECTION_PATH,
                 ROOT / source["submission_file"], ROOT / source["evidence_file"],
                 path, path.parent / "evaluation-evidence.json", path.parent / "metrics/cases.json"]
        for changed in paths:
            with self.subTest(changed=changed), patch.object(units, "digest", side_effect=lambda value: "0" * 64 if value == changed else real_digest(value)):
                with self.assertRaises(ValueError):
                    units.corrected_registration_view(ROOT, path, submission, binding)


if __name__ == "__main__":
    unittest.main()
