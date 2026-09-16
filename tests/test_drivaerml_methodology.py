from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from reference.drivaerml.methodology import (
    DrivAerMethodologyError,
    derived_parameter_count_millions,
    methodology_errors,
)
from scripts import validate_submission as submission_validator
from tests.test_drivaerml_candidate_package_assembler import valid_methodology


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "benchmark-specs/drivaerml/methodology-contract.json").read_text(
        encoding="utf-8"
    )
)


def submission_for(methodology: dict, *, training_regime: str = "from_scratch") -> dict:
    return {
        "dataset_id": "drivaerml",
        "model_type": "Neural operator",
        "training_regime": training_regime,
        "methodology": methodology,
        "parameter_count_millions": derived_parameter_count_millions(methodology),
    }


def methodology_validator() -> Draft202012Validator:
    schema = json.loads(
        (ROOT / "schemas/v3/submission.schema.json").read_text(encoding="utf-8")
    )
    fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        "$ref": "#/$defs/fluidsbench_methodology",
    }
    return Draft202012Validator(fragment, format_checker=FormatChecker())


class DrivAerMLMethodologyTests(unittest.TestCase):
    def assert_semantically_valid(
        self, methodology: dict, *, case_count: int, training_regime: str = "from_scratch"
    ) -> None:
        self.assertEqual(list(methodology_validator().iter_errors(methodology)), [])
        self.assertEqual(
            methodology_errors(
                submission_for(methodology, training_regime=training_regime),
                expected_case_count=case_count,
                contract=CONTRACT,
            ),
            [],
        )

    def test_filled_example_is_valid_for_the_full_split(self) -> None:
        methodology = json.loads(
            (
                ROOT
                / "examples/drivaerml-v3-candidate/methodology.example.json"
            ).read_text(encoding="utf-8")
        )
        self.assert_semantically_valid(methodology, case_count=50)

    def test_surface_only_requires_only_the_two_native_surface_outputs(self) -> None:
        methodology = valid_methodology()
        methodology["architecture"]["predicted_fields"] = [
            field
            for field in methodology["architecture"]["predicted_fields"]
            if field["field_id"].startswith("surface_native_cells.")
        ]
        submission = submission_for(methodology)
        submission["prediction_scope"] = "surface_only"
        self.assertEqual(
            methodology_errors(
                submission,
                expected_case_count=2,
                contract=CONTRACT,
            ),
            [],
        )

    def test_training_actor_is_independent_of_target_data_regime(self) -> None:
        submitter_pretrained = valid_methodology()
        self.assert_semantically_valid(
            submitter_pretrained,
            case_count=2,
            training_regime="pretrained_zero_shot",
        )

        upstream_from_scratch = valid_methodology()
        upstream_from_scratch["architecture"][
            "submitter_trainable_parameter_count"
        ] = 0
        upstream_from_scratch["training"]["stages"] = [
            {
                "id": "upstream-training",
                "status": "performed_upstream",
                "component_ids": ["joint-predictor"],
                "description": "Original from-scratch training was performed upstream.",
                "upstream_reference": "https://example.org/upstream-method",
            }
        ]
        self.assert_semantically_valid(upstream_from_scratch, case_count=2)

    def test_separate_surface_and_volume_models_are_representable(self) -> None:
        methodology = valid_methodology()
        architecture = methodology["architecture"]
        architecture["components"] = [
            {
                "id": "surface-model",
                "family": "Surface graph network",
                "role": "Predict both surface fields.",
                "description": "Eight graph blocks and a two-field output head.",
                "parameter_count": 400_000,
            },
            {
                "id": "volume-model",
                "family": "Volume neural operator",
                "role": "Predict both volume fields.",
                "description": "Six operator blocks and a two-field output head.",
                "parameter_count": 834_567,
            },
        ]
        architecture["submitter_trainable_parameter_count"] = 400_000
        architecture["key_hyperparameters"] = [
            {
                "id": "surface-width",
                "component_ids": ["surface-model"],
                "name": "hidden_width",
                "value": 128,
                "description": "Surface latent width.",
            },
            {
                "id": "volume-width",
                "component_ids": ["volume-model"],
                "name": "hidden_width",
                "value": 256,
                "description": "Volume latent width.",
            },
        ]
        architecture["input_features"] = [
            {
                "id": "surface-coordinates",
                "component_ids": ["surface-model"],
                "name": "coordinates",
                "domain": "surface",
                "component_count": 3,
                "description": "Surface cell-centre coordinates.",
            },
            {
                "id": "volume-coordinates",
                "component_ids": ["volume-model"],
                "name": "coordinates",
                "domain": "volume",
                "component_count": 3,
                "description": "Volume cell-centre coordinates.",
            },
        ]
        for field in architecture["predicted_fields"]:
            field["component_ids"] = [
                "surface-model"
                if field["field_id"].startswith("surface_")
                else "volume-model"
            ]

        submitter_stage = methodology["training"]["stages"][0]
        submitter_stage["id"] = "surface-training"
        submitter_stage["component_ids"] = ["surface-model"]
        methodology["training"]["stages"] = [
            submitter_stage,
            {
                "id": "volume-upstream-training",
                "status": "performed_upstream",
                "component_ids": ["volume-model"],
                "description": "Volume checkpoint training was performed upstream.",
                "upstream_reference": "https://example.org/volume-method",
            },
        ]
        methodology["checkpoints"] = [
            {
                "id": "surface-checkpoint",
                "component_ids": ["surface-model"],
                "sha256": "1" * 64,
                "digest_scope": "raw_loaded_file",
                "bytes_description": "Raw surface checkpoint file.",
                "role": "Surface predictor weights.",
                "selection_rule": "Lowest validation loss before test inference.",
            },
            {
                "id": "volume-checkpoint",
                "component_ids": ["volume-model"],
                "sha256": "2" * 64,
                "digest_scope": "raw_loaded_file",
                "bytes_description": "Raw volume checkpoint file.",
                "role": "Volume predictor weights.",
                "selection_rule": "The fixed upstream release checkpoint.",
            },
        ]
        self.assert_semantically_valid(methodology, case_count=2)

    def test_cross_field_inconsistencies_fail_closed(self) -> None:
        mutations = {
            "component sum": (
                lambda method: method["architecture"]["components"][0].update(
                    parameter_count=1
                ),
                "must equal the sum",
            ),
            "trainable exceeds total": (
                lambda method: method["architecture"].update(
                    submitter_trainable_parameter_count=2_000_000
                ),
                "cannot exceed total_parameter_count",
            ),
            "unknown component": (
                lambda method: method["architecture"]["input_features"][0].update(
                    component_ids=["unknown-component"]
                ),
                "references unknown architecture components",
            ),
            "missing training provenance": (
                lambda method: method["training"].update(stages=[]),
                "must disclose training provenance",
            ),
            "missing checkpoint binding": (
                lambda method: method["checkpoints"][0].update(component_ids=[]),
                "must bind every parameterized architecture component",
            ),
            "seed count": (
                lambda method: method["training"]["stages"][0].update(run_count=2),
                "exactly one seed",
            ),
            "case count": (
                lambda method: method["inference_compute"].update(case_count=1),
                "official evaluation case count 2",
            ),
            "non-finite": (
                lambda method: method["inference_compute"].update(
                    campaign_wall_time_seconds=math.nan
                ),
                "must be a finite number",
            ),
            "training compute capacity": (
                lambda method: method["training"]["stages"][0]["compute"].update(
                    aggregate_device_hours=3.0
                ),
                "aggregate_device_hours cannot exceed",
            ),
            "inference compute capacity": (
                lambda method: method["inference_compute"].update(
                    aggregate_device_time_seconds=13.0
                ),
                "aggregate_device_time_seconds cannot exceed",
            ),
            "unrepresentable compute": (
                lambda method: method["inference_compute"].update(
                    aggregate_device_time_seconds=10**400
                ),
                "finite numeric representation",
            ),
        }
        for label, (mutate, expected) in mutations.items():
            with self.subTest(label=label):
                methodology = valid_methodology()
                mutate(methodology)
                errors = methodology_errors(
                    submission_for(methodology),
                    expected_case_count=2,
                    contract=CONTRACT,
                )
                self.assertIn(expected, "\n".join(errors))

    def test_extreme_parameter_count_is_a_normal_validation_error(self) -> None:
        methodology = valid_methodology()
        methodology["architecture"]["total_parameter_count"] = 10**400
        with self.assertRaisesRegex(
            DrivAerMethodologyError, "no greater than"
        ):
            derived_parameter_count_millions(methodology)

        submission = {
            "methodology": methodology,
            "parameter_count_millions": 0,
        }
        self.assertIn(
            "no greater than",
            "\n".join(methodology_errors(submission, expected_case_count=2)),
        )

    def test_schema_rejects_fractional_batch_and_non_raw_checkpoint_scope(self) -> None:
        validator = methodology_validator()
        methodology = valid_methodology()
        methodology["training"]["stages"][0]["procedure"]["batch"]["value"] = 0.5
        self.assertTrue(list(validator.iter_errors(methodology)))

        methodology = valid_methodology()
        methodology["checkpoints"][0]["digest_scope"] = "archive"
        self.assertTrue(list(validator.iter_errors(methodology)))

    def test_authoritative_submission_loader_rejects_ambiguous_json(self) -> None:
        for payload, expected in (
            ('{"methodology":{"format":"a","format":"b"}}\n', "duplicate JSON key"),
            (
                '{"methodology":{"inference_compute":{"seconds":NaN}}}\n',
                "forbidden non-finite JSON token NaN",
            ),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "submission.json"
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(
                    submission_validator.SubmissionJSONError, expected
                ):
                    submission_validator.load_submission_json(path)


if __name__ == "__main__":
    unittest.main()
