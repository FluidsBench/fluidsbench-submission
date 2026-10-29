from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reference.evaluate_predictions import (
    aggregate_metric,
    evaluate_prediction_artifact,
    evaluate_support,
    field_metric,
    relative_l2_from_sufficient_statistics,
    relative_l2_sufficient_statistics,
)
from reference.scoring_support import (
    LOCATION_MODES,
    ScoringSupport,
    ScoringSupportError,
    load_support_release,
    load_scoring_support,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples" / "v3-template"
SUPPORT_MANIFEST = TEMPLATE / "support" / "manifest.json"
PREDICTION_MANIFEST = TEMPLATE / "predictions" / "manifest.json"


def write_json(path: Path, value: object) -> None:
    path.write_text(f"{json.dumps(value, indent=2)}\n", encoding="utf-8")


class ScoringSupportV3Tests(unittest.TestCase):
    def evaluate_fixture(self) -> dict:
        return evaluate_prediction_artifact(
            support_manifest_path=SUPPORT_MANIFEST,
            case_set_id="standard",
            prediction_manifest_path=PREDICTION_MANIFEST,
            submission_id="synthetic-open-model-v1",
            split_id="default",
        )

    def test_all_four_location_modes_are_public_interface(self) -> None:
        self.assertEqual(
            LOCATION_MODES,
            {
                "native_entities",
                "materialized_table",
                "structured_grid",
                "reference_generator",
            },
        )

    def test_manifest_index_chunk_chain_loads(self) -> None:
        release = load_support_release(SUPPORT_MANIFEST, "standard")
        self.assertEqual(list(release.cases), ["case-001", "case-002"])
        self.assertEqual(list(release.supports), ["volume-points-v1"])

    def test_installed_archive_member_is_exact_and_cannot_escape_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "support"
            shutil.copytree(TEMPLATE / "support", copied)
            manifest = json.loads((copied / "manifest.json").read_text())
            case_set = manifest["case_sets"][0]
            index_path = copied / case_set["index_file"]
            index = json.loads(index_path.read_text())
            chunk_path = index_path.parent / index["chunks"][0]["file"]
            chunk = json.loads(chunk_path.read_text())
            artifact = chunk["cases"][0]["support_instances"][0]["artifacts"][0]
            original = chunk_path.parent / artifact.pop("path")
            destination = copied / "installed/table.json"
            destination.parent.mkdir()
            shutil.copyfile(original, destination)
            artifact["archive"] = {"url": "https://example.org/release/tables.tar",
                                   "sha256": "a" * 64, "member": "installed/table.json"}

            def load():
                write_json(chunk_path, chunk)
                index["chunks"][0]["sha256"] = sha256_file(chunk_path)
                write_json(index_path, index)
                case_set["index_sha256"] = sha256_file(index_path)
                write_json(copied / "manifest.json", manifest)
                release = load_support_release(copied / "manifest.json", "standard")
                return load_scoring_support(release, "case-001", "volume-points-v1")

            support = load()
            self.assertEqual(len(support.support_ids), 3)
            destination.write_text("{}")
            with self.assertRaisesRegex(ScoringSupportError, "SHA-256 mismatch"):
                load()
            artifact["archive"]["member"] = "../../outside.json"
            with self.assertRaisesRegex(ScoringSupportError, "escapes its declared directory"):
                load()

    def test_exact_id_join_and_case_macro_average(self) -> None:
        result = self.evaluate_fixture()
        expected = json.loads((TEMPLATE / "metrics" / "cases.json").read_text(encoding="utf-8"))
        self.assertEqual(result, expected)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(
            [case["case_id"] for case in result["cases"]],
            ["case-001", "case-002"],
        )
        self.assertAlmostEqual(result["metric_values"]["pressure_rel_l1"], 22.22222222222222)
        self.assertAlmostEqual(
            result["metric_values"]["pressure_equal_rel_l2"],
            24.87944682143987,
        )
        self.assertAlmostEqual(
            result["metric_values"]["pressure_physical_rel_l2"],
            24.692605883433632,
        )
        self.assertAlmostEqual(result["metric_values"]["pressure_mae"], 0.625)
        self.assertAlmostEqual(result["metric_values"]["pressure_rmse"], 0.7865660924854931)
        for case in result["cases"]:
            coverage = case["supports"][0]
            self.assertEqual(coverage["support_count"], coverage["scored_count"])
            self.assertEqual(coverage["coverage_fraction"], 1.0)
            self.assertEqual(coverage["unmapped_count"], 0)

    def test_missing_and_unknown_support_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "v3-template"
            shutil.copytree(TEMPLATE, copied)
            prediction_path = copied / "predictions" / "data" / "case-001.json"
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            prediction["columns"]["support_id"] = ["s002", "s999"]
            prediction["columns"]["pressure_pred"] = [5.0, 1.0]
            write_json(prediction_path, prediction)
            manifest_path = copied / "predictions" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["files"][0]["sha256"] = sha256_file(prediction_path)
            manifest["cases"][0]["files"][0]["row_count"] = 2
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                ScoringSupportError, "cover exactly the canonical support IDs"
            ):
                evaluate_prediction_artifact(
                    support_manifest_path=copied / "support" / "manifest.json",
                    case_set_id="standard",
                    prediction_manifest_path=manifest_path,
                    submission_id="synthetic-open-model-v1",
                    split_id="default",
                )

    def test_duplicate_prediction_support_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "v3-template"
            shutil.copytree(TEMPLATE, copied)
            prediction_path = copied / "predictions" / "data" / "case-001.json"
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            prediction["columns"]["support_id"] = ["s000", "s000", "s002"]
            write_json(prediction_path, prediction)
            manifest_path = copied / "predictions" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["files"][0]["sha256"] = sha256_file(prediction_path)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ScoringSupportError, "duplicate support IDs"):
                evaluate_prediction_artifact(
                    support_manifest_path=copied / "support" / "manifest.json",
                    case_set_id="standard",
                    prediction_manifest_path=manifest_path,
                    submission_id="synthetic-open-model-v1",
                    split_id="default",
                )

    def test_duplicate_support_definitions_and_manifest_self_hash_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "support"
            shutil.copytree(TEMPLATE / "support", copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["supports"].append(dict(manifest["supports"][0]))
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ScoringSupportError, "definition IDs must be unique"):
                load_support_release(manifest_path, "standard")

            manifest["supports"].pop()
            manifest["manifest_sha256"] = "a" * 64
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ScoringSupportError, "must not contain its own"):
                load_support_release(manifest_path, "standard")

    def test_vector_metrics_repeat_entity_weights_per_component(self) -> None:
        truth = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.asarray([[2.0, 0.0], [3.0, 6.0]])
        weights = np.asarray([1.0, 2.0])
        self.assertAlmostEqual(
            field_metric(
                truth,
                prediction,
                weights,
                reduction="relative_l1_percent",
                weighting="support_weights",
            ),
            100.0 * 7.0 / 17.0,
        )
        self.assertAlmostEqual(
            field_metric(
                truth,
                prediction,
                weights,
                reduction="relative_l2_percent",
                weighting="support_weights",
            ),
            100.0 * np.sqrt(13.0 / 55.0),
        )
        self.assertAlmostEqual(
            field_metric(
                truth,
                prediction,
                weights,
                reduction="mse",
                weighting="support_weights",
            ),
            13.0 / 6.0,
        )
        self.assertEqual(
            field_metric(
                truth,
                truth,
                weights,
                reduction="r2",
                weighting="support_weights",
            ),
            1.0,
        )

    def test_relative_l2_statistics_are_chunk_additive(self) -> None:
        truth = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        prediction = np.asarray([[2.0, 1.0], [3.0, 6.0], [4.0, 8.0]])
        weights = np.asarray([0.5, 2.0, 1.5])
        chunks = [
            relative_l2_sufficient_statistics(
                truth[index],
                prediction[index],
                weights[index],
                weighting="support_weights",
            )
            for index in (slice(0, 2), slice(2, 3))
        ]
        self.assertEqual(sum(item["entity_count"] for item in chunks), 3)
        self.assertAlmostEqual(sum(item["total_weight"] for item in chunks), 4.0)
        self.assertAlmostEqual(
            relative_l2_from_sufficient_statistics(chunks),
            field_metric(
                truth,
                prediction,
                weights,
                reduction="relative_l2_percent",
                weighting="support_weights",
            ),
        )

        uniform = relative_l2_sufficient_statistics(
            truth,
            prediction,
            weights,
            weighting="uniform",
        )
        self.assertEqual(uniform["entity_count"], 3)
        self.assertEqual(uniform["total_weight"], 3.0)

        zero_truth_chunk = relative_l2_sufficient_statistics(
            np.zeros((1, 2)),
            np.ones((1, 2)),
            np.ones(1),
            weighting="uniform",
        )
        self.assertEqual(zero_truth_chunk["denominator"], 0.0)
        with self.assertRaisesRegex(
            ScoringSupportError,
            "ground-truth denominator is zero",
        ):
            relative_l2_from_sufficient_statistics([zero_truth_chunk])

    def test_case_evidence_emits_relative_l2_sufficient_statistics(self) -> None:
        result = self.evaluate_fixture()
        support_statistics = result["cases"][0]["supports"][0][
            "metric_sufficient_statistics"
        ]
        self.assertEqual(
            support_statistics["pressure_equal_rel_l2"],
            {
                "reduction": "relative_l2_percent",
                "weighting": "uniform",
                "dataset_weighting": "entities_equal",
                "numerator": 2.0,
                "denominator": 21.0,
                "entity_count": 3,
                "total_weight": 3.0,
            },
        )
        statistics = support_statistics["pressure_physical_rel_l2"]
        self.assertEqual(
            statistics,
            {
                "reduction": "relative_l2_percent",
                "weighting": "support_weights",
                "dataset_weighting": "synthetic_support_weights",
                "numerator": 3.0,
                "denominator": 25.0,
                "entity_count": 3,
                "total_weight": 4.0,
            },
        )

    def test_one_support_can_emit_equal_entity_and_physical_relative_l2(self) -> None:
        support = ScoringSupport(
            case_id="case-001",
            support_name="surface",
            support_ids=np.asarray(["s0", "s1"]),
            coordinates=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            weights=np.asarray([1.0, 3.0]),
            targets={"pressure": np.asarray([[1.0], [2.0]])},
        )
        aligned = {"pressure": np.asarray([[2.0], [2.0]])}
        bindings = [
            {
                "metric_id": "pressure_equal_entity_rel_l2",
                "quantity_id": "pressure",
                "reduction": "relative_l2_percent",
                "weighting": "uniform",
                "dataset_weighting": "points_equal_within_case_cases_equal",
                "case_evidence": "metric_value",
            },
            {
                "metric_id": "pressure_area_rel_l2",
                "quantity_id": "pressure",
                "reduction": "relative_l2_percent",
                "weighting": "support_weights",
                "dataset_weighting": "surface_point_dual_area",
                "case_evidence": "metric_value",
            },
        ]
        values, statistics = evaluate_support(
            support,
            aligned,
            {"metric_bindings": bindings},
        )
        self.assertEqual(set(values), {binding["metric_id"] for binding in bindings})
        self.assertEqual(set(statistics), set(values))
        self.assertEqual(
            statistics["pressure_equal_entity_rel_l2"]["total_weight"],
            2.0,
        )
        self.assertEqual(statistics["pressure_area_rel_l2"]["total_weight"], 4.0)

    def test_published_cross_case_aggregation_rules(self) -> None:
        ones = np.ones(2)
        global_payloads = [
            (np.asarray([[0.0], [1.0]]), np.asarray([[0.0], [1.0]]), ones),
            (np.asarray([[2.0], [3.0]]), np.asarray([[1.0], [4.0]]), ones),
        ]
        self.assertAlmostEqual(
            aggregate_metric(
                global_payloads,
                {
                    "aggregation": "flatten_all_aligned_field_values",
                    "reduction": "r2",
                    "weighting": "uniform",
                },
            ),
            0.6,
        )
        field_payloads = [
            (np.asarray([[2.0], [4.0]]), np.asarray([[3.0], [3.0]]), ones),
            (np.asarray([[1.0], [2.0]]), np.asarray([[1.0], [4.0]]), ones),
        ]
        field_rule = "benchmark_field_rrmse_across_cases"
        self.assertAlmostEqual(
            aggregate_metric(
                field_payloads,
                {
                    "aggregation": field_rule,
                    "reduction": "dataset_reference",
                    "weighting": "uniform",
                    "reference_rule": {"id": field_rule, "version": "1.0"},
                },
            ),
            np.sqrt((0.0625 + 0.5) / 2.0),
        )
        scalar_rule = "benchmark_scalar_rrmse_across_cases"
        self.assertAlmostEqual(
            aggregate_metric(
                [
                    (
                        np.asarray([[2.0], [4.0]]),
                        np.asarray([[3.0], [2.0]]),
                        ones,
                    )
                ],
                {
                    "aggregation": scalar_rule,
                    "reduction": "dataset_reference",
                    "weighting": "uniform",
                    "reference_rule": {"id": scalar_rule, "version": "1.0"},
                },
            ),
            0.5,
        )

    def test_new_schema_files_are_valid_json(self) -> None:
        schema_paths = [
            *sorted((ROOT / "schemas" / "scoring-support" / "v1").glob("*.json")),
            *sorted((ROOT / "schemas" / "v3").glob("*.json")),
        ]
        self.assertGreaterEqual(len(schema_paths), 10)
        for path in schema_paths:
            with self.subTest(path=path.name):
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_template_hash_bindings_and_jsonl(self) -> None:
        submission = json.loads((TEMPLATE / "submission.json").read_text(encoding="utf-8"))
        evidence = json.loads(
            (TEMPLATE / "evaluation-evidence.json").read_text(encoding="utf-8")
        )
        discretization = json.loads(
            (TEMPLATE / "discretization.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            submission["evaluation"]["evidence_sha256"],
            sha256_file(TEMPLATE / "evaluation-evidence.json"),
        )
        self.assertEqual(
            submission["spatial_discretization"]["sha256"],
            sha256_file(TEMPLATE / "discretization.json"),
        )
        self.assertEqual(
            submission["case_metrics"]["sha256"],
            sha256_file(TEMPLATE / "metrics" / "cases.json"),
        )
        self.assertEqual(
            evidence["profile_index_sha256"],
            sha256_file(TEMPLATE / "profiles" / "index.json"),
        )
        self.assertEqual(
            discretization["case_manifest"]["sha256"],
            sha256_file(TEMPLATE / "discretization" / "cases.jsonl"),
        )
        lines = [
            json.loads(line)
            for line in (TEMPLATE / "discretization" / "cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        self.assertEqual([line["case_id"] for line in lines], ["case-001", "case-002"])


if __name__ == "__main__":
    unittest.main()
