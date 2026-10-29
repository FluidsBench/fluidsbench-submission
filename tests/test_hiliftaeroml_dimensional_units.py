"""Known-answer SI exports and preservation of the registered native results."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from reference.hiliftaeroml import dimensional_units as units
from tests.hilift_si_fixture import corrected_documents
from scripts import assemble_hiliftaeroml_schema_v3_candidate as assembler
from scripts import validate_submission as validator

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


class HiLiftDimensionalUnitsTests(unittest.TestCase):

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
        result, proof, corrected = corrected_documents(submission, evidence, cases)
        self.assertEqual((submission, evidence, cases), originals)
        self.assertEqual(result["metric_values"]["overall_score"], 42.0)
        self.assertEqual(result["metric_values"]["surface_pressure_rel_l2"], 10.0)
        self.assertAlmostEqual(result["metric_values"]["volume_velocity_rmse"], 0.0508)
        self.assertAlmostEqual(result["metric_values"]["surface_pressure_rmse"], 1149.126215527559)
        for before, after in zip(cases["cases"], corrected["cases"]):
            self.assertEqual(before["supports"][0]["metric_sufficient_statistics"], after["supports"][0]["metric_sufficient_statistics"])
        with self.assertRaisesRegex(ValueError, "already recorded"):
            corrected_documents(result, proof, corrected)
        cases["cases"][0]["supports"][0]["metric_values"].pop("surface_pressure_mae")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            corrected_documents(submission, evidence, cases)




if __name__ == "__main__":
    unittest.main()
