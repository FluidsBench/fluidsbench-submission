"""DrivAerNet++ candidate support, evaluator, and resumable participant workflow.

Hermetic: the large support archives are never downloaded.  Release-chain
tests read only the hash-pinned records retained in Git; evaluator and
workflow tests use small synthetic supports.
"""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from reference.drivaernetplusplus import contract, dataset_scorer, distribution, evaluator, native
from reference.drivaernetplusplus import participant_workflow as workflow
from reference.drivaernetplusplus.support import (
    EXPECTED_ARRAYS,
    load_npz_bytes,
    npz_bytes,
    station_array_names,
    validate_case_support,
)
from reference.scoring_support import load_support_release
from scripts.validate_scoring_supports import validate_candidate_manifest_release

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "benchmark-specs" / "drivaernetplusplus" / "tools" / "publish_scoring_support.py"


def _reference_loop():
    spec = importlib.util.spec_from_file_location("dnpp_reference_publisher", PUBLISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mass_lumped_dual_areas


class CommittedReleaseChain(unittest.TestCase):
    """The Git-retained half of the candidate release is complete and bound."""

    @classmethod
    def setUpClass(cls):
        cls.spec = contract.load_specification()
        cls.split = contract.official_split()
        cls.manifest_path = contract.SPEC_ROOT / cls.spec["scoring_support"]["candidate_manifest"]["manifest_file"]

    def test_specification_is_closed_candidate_bound_to_manifest(self):
        support = self.spec["scoring_support"]
        self.assertEqual(self.spec["status"], "official")
        self.assertEqual(support["status"], "official")
        self.assertIs(support["submissions_open"], True)
        self.assertEqual(
            support["candidate_manifest"]["manifest_sha256"], contract.sha256_file(self.manifest_path)
        )
        self.assertEqual([s["id"] for s in self.spec["splits"]], ["official_test"])
        self.assertEqual(support["owner_approval"]["approved_by"], "Neil Ashton")

    def test_candidate_manifest_release_validates(self):
        errors = validate_candidate_manifest_release(
            self.spec, contract.SPEC_ROOT, self.manifest_path, contract.read_json(self.manifest_path)
        )
        self.assertEqual(errors, [])
        release = load_support_release(self.manifest_path, contract.CASE_SET_ID)
        self.assertEqual(list(release.cases), self.split["case_ids"])

    def test_distribution_covers_split_exactly_once(self):
        index = distribution.load_distribution()
        covered = [case for entry in index["archives"] for case in entry["case_ids"]]
        self.assertEqual(covered, self.split["case_ids"])
        self.assertTrue(all(entry["url"].startswith(contract.DEFAULT_BASE_URL) for entry in index["archives"]))

    def test_package_template_bindings_match_repository(self):
        template = contract.read_json(
            ROOT / "examples" / "drivaernetplusplus-v3-candidate" / "package-config.template.json"
        )
        owner = self.spec["scoring_support"]["candidate_manifest"]
        truth = contract.read_json(ROOT / "leaderboard" / "manifest.json")["data_release"]["profile_ground_truth"]
        self.assertEqual(
            template["release_bindings"],
            {
                "candidate_manifest": {k: owner[k] for k in ("status", "release_id", "manifest_url", "manifest_sha256")},
                "evaluator": {"reference_version": self.spec["evaluation_reference_version"]},
                "profile_ground_truth": {"release_id": truth["release_id"], "manifest_sha256": truth["manifest_sha256"]},
            },
        )

    def test_every_case_record_is_consistent(self):
        release = load_support_release(self.manifest_path, contract.CASE_SET_ID)
        identity = {row["case_id"]: row for row in workflow.source_identity()["cases"]}
        for case_id in self.split["case_ids"]:
            record, payload = distribution.retained_record(case_id)
            instance = release.cases[case_id]["support_instances"][0]
            self.assertEqual(instance["entity_count"], record["native"]["point_count"], case_id)
            self.assertEqual(instance["expected_support_sha256"], contract.sha256_bytes(payload), case_id)
            self.assertEqual(identity[case_id]["sha256"], record["source"]["sha256"], case_id)
            self.assertEqual(list(record["support_npz"]["arrays"]), list(EXPECTED_ARRAYS), case_id)
            self.assertGreaterEqual(len(record["native"]["connected_component_point_counts"]), 5, case_id)
            for station in native.PROFILE_STATIONS:
                self.assertGreaterEqual(
                    record["profile_stations"][station]["point_count"], native.MINIMUM_STATION_POINTS, case_id
                )


def _synthetic_support(case_id="E_S_WW_WM_075", n=40, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {
        "dual_area_m2": rng.uniform(1e-5, 2e-5, n).astype("<f8"),
        "p": rng.normal(-90.0, 120.0, n).astype("<f4"),
    }
    stations = {}
    for offset, station in enumerate(native.PROFILE_STATIONS):
        index_name, coordinate_name = station_array_names(station)
        index = np.arange(offset, n, 7, dtype="<i8")
        arrays[index_name] = index
        arrays[coordinate_name] = np.linspace(0.0, 1.0, len(index)).astype("<f8")
        stations[station] = {"coordinate_id": "x_over_L", "point_count": len(index)}
    payload = npz_bytes(arrays)
    record = {
        "schema": contract.CASE_SUPPORT_SCHEMA,
        "release_id": contract.RELEASE_ID,
        "dataset_id": contract.DATASET_ID,
        "case_id": case_id,
        "rules_version": native.SUPPORT_RULES_VERSION,
        "native": {"point_count": n},
        "support_npz": {
            "sha256": contract.sha256_bytes(payload),
            "arrays": {
                name: {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": native.array_sha256(a)}
                for name, a in arrays.items()
            },
        },
        "profile_stations": stations,
    }
    record_bytes = contract.json_bytes(record)
    loaded = load_npz_bytes(payload, case_id)
    validate_case_support(record, loaded, case_id)
    return distribution.CaseSupport(case_id, record, contract.sha256_bytes(record_bytes), loaded), record_bytes


def _write_prediction(path: Path, values, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, p=values, **extra)


class EvaluatorMathematics(unittest.TestCase):
    def setUp(self):
        self.support, _ = _synthetic_support()
        self.truth = self.support.arrays["p"].astype(np.float64)
        self.weights = self.support.arrays["dual_area_m2"]
        self.directory = Path(tempfile.mkdtemp())

    def score(self, values, **extra):
        path = self.directory / "prediction.npz"
        _write_prediction(path, values, **extra)
        return evaluator.score_case(self.support, path)

    def test_identity_prediction_is_exact(self):
        evidence = self.score(self.support.arrays["p"])
        self.assertEqual(evidence["metric_values"]["surface_pressure_rel_l2"], 0.0)
        self.assertEqual(evidence["metric_values"]["surface_pressure_rel_l1"], 0.0)
        self.assertEqual(evidence["r2_statistics"]["sum_squared_error"], 0.0)

    def test_single_point_perturbation_matches_hand_calculation(self):
        values = self.truth.copy()
        values[3] += 5.0
        evidence = self.score(values)
        expected = 100 * math.sqrt(self.weights[3] * 25.0 / np.sum(self.weights * self.truth**2))
        self.assertAlmostEqual(evidence["metric_values"]["surface_pressure_rel_l2"], expected, places=12)
        expected_equal = 100 * math.sqrt(25.0 / np.sum(self.truth**2))
        self.assertAlmostEqual(evidence["metric_values"]["surface_pressure_equal_entity_rel_l2"], expected_equal, places=12)

    def test_profiles_read_predictions_at_frozen_points(self):
        values = self.truth * 2
        evidence = self.score(values)
        for profile in evidence["profiles"]:
            index = self.support.arrays[station_array_names(profile["station_id"])[0]]
            np.testing.assert_allclose(profile["prediction_cp"], values[index] / 450.0)
            np.testing.assert_allclose(profile["truth_cp"], self.truth[index] / 450.0)

    def test_malformed_predictions_are_rejected(self):
        bad = {
            "short": dict(values=self.truth[:-1]),
            "nan": dict(values=np.where(np.arange(40) == 5, np.nan, self.truth)),
            "integer": dict(values=np.zeros(40, dtype=np.int32)),
            "extra array": dict(values=self.truth, extra=np.zeros(1)),
        }
        for label, kwargs in bad.items():
            with self.subTest(label), self.assertRaises(evaluator.PredictionError):
                extra = {"extra": kwargs.pop("extra")} if "extra" in kwargs else {}
                self.score(kwargs["values"], **extra)

    def test_pooled_r2_equals_direct_computation(self):
        cases, truth, prediction = [], [], []
        for seed in range(4):
            support, _ = _synthetic_support(n=30 + seed * 7, seed=seed)
            self.support = support
            values = support.arrays["p"].astype(np.float64) + np.random.default_rng(seed).normal(0, 9, support.point_count)
            cases.append(self.score(values))
            truth.append(support.arrays["p"].astype(np.float64))
            prediction.append(values)
        y, p = np.concatenate(truth), np.concatenate(prediction)
        direct = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
        self.assertAlmostEqual(dataset_scorer.pooled_r2(cases), direct, places=12)


class NativeRules(unittest.TestCase):
    def test_vectorized_dual_areas_equal_reference_loop(self):
        rng = np.random.default_rng(3)
        points = rng.normal(size=(60, 3)).astype(np.float32)
        polygons = [list(rng.choice(60, size=k, replace=False)) for k in rng.integers(3, 8, size=90)]
        polygons.append(list(range(60)))  # every point gets area
        offsets = np.cumsum([0] + [len(p) for p in polygons])
        surface = native.NativeSurface(
            points, offsets.astype(np.int64), np.concatenate(polygons).astype(np.int64), np.zeros(60, np.float32)
        )
        expected = _reference_loop()(points.astype(np.float64), [[int(v) for v in p] for p in polygons])
        # Identical operation order; vectorized norms may round differently by
        # CPU (seen on CI runners), so allow round-off. Scoring never recomputes
        # weights: participants read the published arrays.
        np.testing.assert_allclose(native.mass_lumped_dual_areas(surface), expected, rtol=1e-13, atol=0)

    def test_components_follow_polygon_connectivity(self):
        polygons = [[0, 1, 2], [2, 3, 4], [5, 6, 7], [8, 9, 10, 11]]
        offsets = np.cumsum([0] + [len(p) for p in polygons]).astype(np.int64)
        surface = native.NativeSurface(
            np.zeros((12, 3), np.float32), offsets, np.concatenate(polygons).astype(np.int64), np.zeros(12, np.float32)
        )
        labels = native.connected_components(surface)
        self.assertEqual(len(set(labels[:5])), 1)
        self.assertEqual(len(set(labels.tolist())), 3)


class ResumableWorkflow(unittest.TestCase):
    """Two-case split: evaluate, resume, detect changed predictions, reduce."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.cases = ["E_S_WW_WM_075", "F_D_WM_WW_0001"]
        self.supports = {}
        self.records = {}
        for seed, case_id in enumerate(self.cases):
            support, record_bytes = _synthetic_support(case_id, n=50 + seed, seed=seed)
            self.supports[case_id] = support
            self.records[case_id] = (support.record, record_bytes)
        self.entry = self.directory / "entry"
        self.entry.mkdir()
        (self.entry / "entry.json").write_text(
            json.dumps({"schema": workflow.ENTRY_SCHEMA, "submission_id": "dnpp-test-model", "split_id": "official_test"})
        )
        for case_id, support in self.supports.items():
            _write_prediction(workflow.prediction_path(self.entry, case_id), support.arrays["p"] * 1.1)
        split = {"case_ids": self.cases, "case_set_id": "official_test", "split_id": "official_test"}
        retained = lambda case_id: self.records[case_id]  # noqa: E731
        from concurrent.futures import ThreadPoolExecutor
        self.patches = [
            mock.patch.object(workflow.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor),
            mock.patch.object(workflow, "official_split", return_value=split),
            mock.patch.object(dataset_scorer, "official_split", return_value=split),
            mock.patch.object(workflow, "retained_record", side_effect=retained),
            mock.patch.object(dataset_scorer, "retained_record", side_effect=retained),
            mock.patch.object(workflow, "load_case_support", side_effect=lambda c, destination=None: self.supports[c]),
            mock.patch.object(workflow, "fetch_support", return_value={}),
            mock.patch.object(workflow, "check_runtime", return_value={"system": "Linux"}),
            mock.patch.object(dataset_scorer, "manifest_sha256", return_value="0" * 64),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in self.patches])

    def evaluate(self, output, **kwargs):
        return workflow.evaluate_entry(self.entry, output, workers=1, **kwargs)

    def test_complete_then_resume_reuses_checkpoints(self):
        output = self.directory / "work"
        first = self.evaluate(output)
        self.assertEqual(first["case_count"], 2)
        self.assertAlmostEqual(first["metric_values"]["surface_pressure_rel_l2"], 10.0, places=4)
        second = self.evaluate(output, resume=True)
        self.assertTrue(second.get("reused"))

    def test_interrupted_run_resumes_only_missing_case(self):
        output = self.directory / "work"
        with mock.patch.object(workflow, "_reduce", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                self.evaluate(output)
        evidence = output / "cases" / self.cases[0] / "evidence.json"
        before = evidence.stat().st_mtime_ns
        (output / "cases" / self.cases[1]).rename(output / "cases" / f".{self.cases[1]}-removed")
        self.evaluate(output, resume=True)
        self.assertEqual(evidence.stat().st_mtime_ns, before)

    def test_changed_prediction_is_refused_on_resume(self):
        output = self.directory / "work"
        self.evaluate(output)
        _write_prediction(workflow.prediction_path(self.entry, self.cases[0]), self.supports[self.cases[0]].arrays["p"])
        with self.assertRaises(workflow.WorkflowError):
            self.evaluate(output, resume=True)

    def test_non_empty_output_requires_resume(self):
        output = self.directory / "work"
        self.evaluate(output)
        with self.assertRaises(workflow.WorkflowError):
            self.evaluate(output)

    def test_entry_must_cover_split_exactly(self):
        (workflow.prediction_path(self.entry, self.cases[1])).unlink()
        (self.entry / "cases" / self.cases[1]).rmdir()
        with self.assertRaises(workflow.WorkflowError):
            workflow.validate_entry(self.entry)


if __name__ == "__main__":
    unittest.main()
