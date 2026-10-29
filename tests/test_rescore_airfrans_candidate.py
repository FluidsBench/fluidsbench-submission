from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.scores import composite_component_group_scores, composite_overall_score
from scripts.rescore_airfrans_candidate import (
    DEFAULT_SPEC,
    RescoreError,
    dump_json,
    rescore_package,
)

BASE_METRICS = {
    "surface_pressure_rel_l2": 3.0,
    "surface_wall_shear_rel_l2": 10.0,
    "flow_domain_velocity_rel_l2": 9.0,
    "flow_domain_pressure_rel_l2": 15.0,
    "cd_r2": 0.8,
    "cl_r2": 0.6,
    "velocity_profile_r2": 0.4,
    "c_drag_mae": 0.01,
}
STALE_DERIVED = {
    "overall_score": 1.0,
    "field_score": 2.0,
    "force_score": 3.0,
    "diagnostic_score": 4.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


OLD_MANIFEST_SHA = "a" * 64
RELEASE_ID = "demo-release-v1-candidate"


def write_package(root: Path) -> Path:
    package = root / "pkg"
    (package / "metrics").mkdir(parents=True)
    identity = {"submission_id": "demo-v1", "dataset_id": "airfrans", "split_id": "full"}
    metric_values = {**BASE_METRICS, **STALE_DERIVED}
    discretization = {
        **identity,
        "scoring_support_manifest_sha256": OLD_MANIFEST_SHA,
        "case_records": {"file": "discretization/cases.jsonl", "sha256": "b" * 64},
    }
    (package / "discretization.json").write_bytes(dump_json(discretization))
    case_metrics = {
        **identity,
        "scoring_support_manifest_sha256": OLD_MANIFEST_SHA,
        "case_count": 1,
        "cases": [{"case_id": "c0", "supports": [{"metric_values": {"rel_l2": 3.0}}]}],
        "metric_values": metric_values,
    }
    (package / "metrics/cases.json").write_bytes(dump_json(case_metrics))
    evidence = {
        **identity,
        "dataset_version": "prototype-1",
        "reference_version": "prototype-1",
        "command": "evaluate",
        "case_metrics_sha256": sha256(package / "metrics/cases.json"),
        "discretization_sha256": sha256(package / "discretization.json"),
        "scoring_support_manifest_sha256": OLD_MANIFEST_SHA,
        "notes": "Candidate.",
        "metric_values": metric_values,
    }
    (package / "evaluation-evidence.json").write_bytes(dump_json(evidence))
    submission = {
        **identity,
        "dataset_version": "prototype-1",
        "evaluation": {
            "reference_version": "prototype-1",
            "command": "evaluate",
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": sha256(package / "evaluation-evidence.json"),
        },
        "case_metrics": {"file": "metrics/cases.json", "sha256": sha256(package / "metrics/cases.json")},
        "spatial_discretization": {
            "file": "discretization.json",
            "sha256": sha256(package / "discretization.json"),
        },
        "scoring_support": {"release_id": RELEASE_ID, "manifest_sha256": OLD_MANIFEST_SHA},
        "metric_values": metric_values,
    }
    (package / "submission.json").write_bytes(dump_json(submission))
    return package


def write_manifest(root: Path, **overrides: str) -> Path:
    manifest = {
        "release_id": RELEASE_ID,
        "dataset_id": "airfrans",
        "dataset_version": "prototype-1",
        "evaluation_reference_version": "prototype-1",
        **overrides,
    }
    path = root / "manifest.json"
    path.write_bytes(dump_json(manifest))
    return path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class RescoreAirfransCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load(DEFAULT_SPEC)
        self._tmp = tempfile.TemporaryDirectory()
        self.package = write_package(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rescore_recomputes_derived_scores_and_rehashes(self) -> None:
        report = rescore_package(self.package, self.spec, write=True)
        self.assertEqual(
            sorted(report["changed_files"]),
            ["evaluation-evidence.json", "metrics/cases.json", "submission.json"],
        )
        case_metrics = load(self.package / "metrics/cases.json")
        evidence = load(self.package / "evaluation-evidence.json")
        submission = load(self.package / "submission.json")

        values = case_metrics["metric_values"]
        composite = self.spec["overall_score_composite"]
        self.assertAlmostEqual(values["overall_score"], composite_overall_score(values, composite))
        groups = composite_component_group_scores(
            values, composite, self.spec["component_score_groups"]
        )
        for metric_id, expected in groups.items():
            self.assertAlmostEqual(values[metric_id], expected)
        for metric_id, expected in BASE_METRICS.items():
            self.assertEqual(values[metric_id], expected)
        self.assertEqual(case_metrics["cases"][0]["supports"][0]["metric_values"], {"rel_l2": 3.0})

        version = self.spec["evaluation_reference_version"]
        self.assertEqual(evidence["reference_version"], version)
        self.assertEqual(submission["evaluation"]["reference_version"], version)
        self.assertEqual(evidence["dataset_version"], "prototype-1")
        self.assertEqual(evidence["metric_values"], values)
        self.assertEqual(submission["metric_values"], values)
        self.assertEqual(evidence["case_metrics_sha256"], sha256(self.package / "metrics/cases.json"))
        self.assertEqual(submission["case_metrics"]["sha256"], sha256(self.package / "metrics/cases.json"))
        self.assertEqual(
            submission["evaluation"]["evidence_sha256"], sha256(self.package / "evaluation-evidence.json")
        )
        self.assertIn("rescore_airfrans_candidate.py", evidence["command"])
        self.assertIn(version, evidence["notes"])
        self.assertEqual(submission["scoring_support"]["manifest_sha256"], OLD_MANIFEST_SHA)

    def test_rebinds_restamped_manifest_and_discretization_hash(self) -> None:
        version = self.spec["evaluation_reference_version"]
        manifest = write_manifest(Path(self._tmp.name), evaluation_reference_version=version)
        report = rescore_package(
            self.package, self.spec, write=True, scoring_support_manifest=manifest
        )
        self.assertIn("discretization.json", report["changed_files"])
        new_sha = sha256(manifest)
        case_metrics = load(self.package / "metrics/cases.json")
        evidence = load(self.package / "evaluation-evidence.json")
        submission = load(self.package / "submission.json")
        discretization = load(self.package / "discretization.json")
        for document in (case_metrics, evidence, discretization):
            self.assertEqual(document["scoring_support_manifest_sha256"], new_sha)
        self.assertEqual(submission["scoring_support"]["manifest_sha256"], new_sha)
        self.assertEqual(
            discretization["case_records"],
            {"file": "discretization/cases.jsonl", "sha256": "b" * 64},
        )
        discretization_sha = sha256(self.package / "discretization.json")
        self.assertEqual(evidence["discretization_sha256"], discretization_sha)
        self.assertEqual(submission["spatial_discretization"]["sha256"], discretization_sha)
        self.assertEqual(
            submission["evaluation"]["evidence_sha256"], sha256(self.package / "evaluation-evidence.json")
        )
        self.assertIn("--scoring-support-manifest", evidence["command"])

    def test_refuses_manifest_with_old_version(self) -> None:
        manifest = write_manifest(Path(self._tmp.name))
        with self.assertRaisesRegex(RescoreError, "evaluation_reference_version"):
            rescore_package(self.package, self.spec, write=True, scoring_support_manifest=manifest)

    def test_refuses_manifest_of_another_release(self) -> None:
        manifest = write_manifest(
            Path(self._tmp.name),
            release_id="other-release",
            evaluation_reference_version=self.spec["evaluation_reference_version"],
        )
        with self.assertRaisesRegex(RescoreError, "release_id"):
            rescore_package(self.package, self.spec, write=True, scoring_support_manifest=manifest)

    def test_second_run_is_a_no_op(self) -> None:
        rescore_package(self.package, self.spec, write=True)
        before = {p.name: p.read_bytes() for p in self.package.rglob("*.json")}
        report = rescore_package(self.package, self.spec, write=True)
        self.assertEqual(report["changed_files"], [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.package.rglob("*.json")})

    def test_check_mode_does_not_write(self) -> None:
        before = {p.name: p.read_bytes() for p in self.package.rglob("*.json")}
        report = rescore_package(self.package, self.spec, write=False)
        self.assertTrue(report["changed_files"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.package.rglob("*.json")})

    def test_refuses_broken_hash_chain(self) -> None:
        evidence_path = self.package / "evaluation-evidence.json"
        evidence = load(evidence_path)
        evidence["case_metrics_sha256"] = "0" * 64
        evidence_path.write_bytes(dump_json(evidence))
        with self.assertRaisesRegex(RescoreError, "case_metrics_sha256"):
            rescore_package(self.package, self.spec, write=True)

    def test_refuses_disagreeing_metric_values(self) -> None:
        submission_path = self.package / "submission.json"
        submission = load(submission_path)
        submission["metric_values"]["cd_r2"] = 0.9
        submission_path.write_bytes(dump_json(submission))
        with self.assertRaisesRegex(RescoreError, "metric_values differ"):
            rescore_package(self.package, self.spec, write=True)


if __name__ == "__main__":
    unittest.main()
