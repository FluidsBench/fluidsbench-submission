from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - dependency is installed in repository CI
    Draft202012Validator = None
    FormatChecker = None


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples" / "v3-template"


@unittest.skipIf(Draft202012Validator is None, "jsonschema dependency is not installed")
class V3SchemaExampleTests(unittest.TestCase):
    def assert_schema_valid(self, schema_path: Path, value: object) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            ).iter_errors(value),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(
            errors,
            [],
            "\n".join(
                f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
                for error in errors
            ),
        )

    def test_scoring_support_chain_and_prediction_artifact(self) -> None:
        checks = [
            (
                ROOT / "schemas/scoring-support/v1/manifest.schema.json",
                TEMPLATE / "support/manifest.json",
            ),
            (
                ROOT / "schemas/scoring-support/v1/case-index.schema.json",
                TEMPLATE / "support/case-sets/standard/index.json",
            ),
            (
                ROOT / "schemas/scoring-support/v1/case-chunk.schema.json",
                TEMPLATE / "support/case-sets/standard/chunk-000.json",
            ),
            (
                ROOT / "schemas/v3/prediction-artifact.schema.json",
                TEMPLATE / "predictions/manifest.json",
            ),
        ]
        for schema_path, value_path in checks:
            with self.subTest(value=value_path.name):
                self.assert_schema_valid(
                    schema_path,
                    json.loads(value_path.read_text(encoding="utf-8")),
                )

    def test_dataset_reference_reduction_requires_matching_rrmse_rule(self) -> None:
        schema = json.loads(
            (
                ROOT / "schemas/scoring-support/v1/manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        manifest = json.loads(
            (TEMPLATE / "support/manifest.json").read_text(encoding="utf-8")
        )
        binding = manifest["supports"][0]["metric_bindings"][0]
        binding.update(
            {
                "reduction": "dataset_reference",
                "case_evidence": "aggregate_only",
                "reference_rule": {
                    "id": "benchmark_field_rrmse_across_cases",
                    "version": "1.0",
                },
            }
        )
        self.assertTrue(list(validator.iter_errors(manifest)))

        binding["aggregation"] = "benchmark_field_rrmse_across_cases"
        self.assertEqual(list(validator.iter_errors(manifest)), [])
        binding["reduction"] = "mae"
        binding["case_evidence"] = "metric_value"
        self.assertTrue(list(validator.iter_errors(manifest)))

    def test_scoring_support_schema_accepts_two_dimensional_cell_area(self) -> None:
        schema = json.loads(
            (
                ROOT / "schemas/scoring-support/v1/manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (TEMPLATE / "support/manifest.json").read_text(encoding="utf-8")
        )
        manifest["supports"][0]["location_definition"]["weight_rule"] = {
            "kind": "geometric",
            "measure": "cell_area",
            "rule_id": "vtk-cell-area-v1",
            "rule_version": "1.0",
        }
        self.assertEqual(
            list(
                Draft202012Validator(
                    schema,
                    format_checker=FormatChecker(),
                ).iter_errors(manifest)
            ),
            [],
        )

    def test_v3_submission_package_examples(self) -> None:
        checks = [
            ("submission.schema.json", "submission.json"),
            ("evaluation-evidence.schema.json", "evaluation-evidence.json"),
            ("discretization.schema.json", "discretization.json"),
            ("case-metrics.schema.json", "metrics/cases.json"),
            ("maintainer-validation.schema.json", "maintainer-validation.json"),
            (
                "prediction-artifact-checks.schema.json",
                "prediction-artifact-checks.json",
            ),
        ]
        for schema_name, relative_value in checks:
            with self.subTest(value=relative_value):
                self.assert_schema_valid(
                    ROOT / "schemas" / "v3" / schema_name,
                    json.loads((TEMPLATE / relative_value).read_text(encoding="utf-8")),
                )

    def test_methodology_is_required_for_every_schema_v3_dataset(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v3/submission.schema.json").read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        generic = json.loads((TEMPLATE / "submission.json").read_text(encoding="utf-8"))
        self.assertEqual(list(validator.iter_errors(generic)), [])
        generic_without_method = copy.deepcopy(generic)
        generic_without_method.pop("methodology")
        self.assertTrue(
            any(
                "methodology" in error.message
                for error in validator.iter_errors(generic_without_method)
            )
        )

        drivaerml = copy.deepcopy(generic)
        drivaerml["dataset_id"] = "drivaerml"
        methodology = json.loads(
            (
                ROOT
                / "examples/drivaerml-v3-candidate/methodology.example.json"
            ).read_text(encoding="utf-8")
        )
        drivaerml["methodology"] = methodology
        drivaerml["parameter_count_millions"] = 42.612345
        self.assertEqual(list(validator.iter_errors(drivaerml)), [])

        invalid_hash = copy.deepcopy(drivaerml)
        invalid_hash["methodology"]["checkpoints"][0]["sha256"] = "not-a-hash"
        self.assertTrue(list(validator.iter_errors(invalid_hash)))

        unknown_key = copy.deepcopy(drivaerml)
        unknown_key["methodology"]["architecture"]["invented"] = True
        self.assertTrue(list(validator.iter_errors(unknown_key)))

        null_size = copy.deepcopy(drivaerml)
        null_size["parameter_count_millions"] = None
        self.assertTrue(list(validator.iter_errors(null_size)))

    def test_discretization_jsonl_records(self) -> None:
        schema_path = ROOT / "schemas/v3/discretization-case.schema.json"
        records = [
            json.loads(line)
            for line in (TEMPLATE / "discretization/cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        self.assertEqual(len(records), 2)
        for record in records:
            with self.subTest(case_id=record["case_id"]):
                self.assert_schema_valid(schema_path, record)


if __name__ == "__main__":
    unittest.main()
