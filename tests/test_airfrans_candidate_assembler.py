from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.assemble_airfrans_schema_v3_candidate import (
    AssembleError,
    DEFAULT_SPEC,
    ROOT,
    assemble,
    build_discretization,
    main,
    profile_ground_truth_binding,
    sha256_file,
    write_json,
    write_jsonl,
)
from scripts.build_airfrans_scoring_support import build_manifest
from scripts.validate_submission import validate_v3_discretization


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class AirfransCandidateAssemblerTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.airfrans_fixture import synthetic_config
        self.config = synthetic_config()
        self.config["release_bindings"]["profile_ground_truth"] = read(DEFAULT_SPEC)["profile_definition"]["profile_ground_truth"]
        self.spec = read(DEFAULT_SPEC)
        from tests.airfrans_fixture import make_fixture
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = make_fixture(Path(self.temporary.name) / "input", case_limit=2)

    def validate_discretization(self, directory: Path, submission: dict, metrics: dict) -> list[str]:
        case_ids = [case["case_id"] for case in metrics["cases"]]
        support = build_manifest(len(case_ids), submission["case_set_id"], submission["scoring_support"]["release_id"])
        index = {"_loaded_cases": [
            {"case_id": case["case_id"], "support_instances": [
                {"support_id": item["support_id"], "entity_count": item["support_count"]}
                for item in case["supports"]
            ]}
            for case in metrics["cases"]
        ]}
        errors = []
        validate_v3_discretization(errors.append, directory, submission, case_ids, support, index)
        return errors

    def test_generated_discretization_passes_reference_validator(self) -> None:
        submission = read(self.package / "submission.json")
        metrics = read(self.package / "metrics/cases.json")
        records = []
        for case in metrics["cases"]:
            counts = {item["support_id"]: item["support_count"] for item in case["supports"]}
            records.append({"case_id": case["case_id"], "n_domain": counts["two-dimensional-domain-native"],
                            "n_curve": counts["airfoil-curve-native"]})
        support = submission["scoring_support"]
        summary, rows = build_discretization(self.config, submission["submission_id"], submission["split_id"],
                                             support["release_id"], support["manifest_sha256"], records)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary["case_manifest"]["sha256"] = write_jsonl(directory / "discretization/cases.jsonl", rows)
            submission["spatial_discretization"]["sha256"] = write_json(directory / "discretization.json", summary)
            self.assertEqual(self.validate_discretization(directory, submission, metrics), [])

    def test_checked_in_candidate_discretization_passes_reference_validator(self) -> None:
        for directory in sorted((ROOT / "submissions/airfrans").glob("transolverpp-*-v1")):
            with self.subTest(candidate=directory.name):
                submission = read(directory / "submission.json")
                evidence = read(directory / "evaluation-evidence.json")
                self.assertEqual(submission["evaluation"]["evidence_sha256"],
                                 sha256_file(directory / "evaluation-evidence.json"))
                self.assertEqual(evidence["discretization_sha256"], sha256_file(directory / "discretization.json"))
                self.assertEqual(self.validate_discretization(directory, submission,
                                                              read(directory / "metrics/cases.json")), [])

    def test_rejects_missing_unresolved_and_definition_based_truth_bindings(self) -> None:
        definition = self.spec["profile_definition"]
        bad_bindings = [
            {},
            {"release_id": definition["id"], "manifest_sha256_source": "profile_definition.sha256"},
            {"release_id": "test-profile-truth", "manifest_sha256": definition["sha256"]},
            {"release_id": "test-profile-truth", "manifest_sha256": "not-a-digest"},
            {"release_id": "__UNRESOLVED_AIRFRANS_PROFILE_GROUND_TRUTH_RELEASE_ID__",
             "manifest_sha256": "__UNRESOLVED_AIRFRANS_PROFILE_GROUND_TRUTH_MANIFEST_SHA256__"},
        ]
        for binding in bad_bindings:
            with self.subTest(binding=binding):
                self.config["release_bindings"]["profile_ground_truth"] = binding
                with self.assertRaises(AssembleError):
                    profile_ground_truth_binding(self.config, self.spec)

    def test_registered_example_uses_the_retained_data_release_binding(self) -> None:
        binding = profile_ground_truth_binding(self.config, self.spec)
        submission = read(self.package / "submission.json")
        self.assertEqual(binding["release_id"], submission["profile_data"]["profile_ground_truth_release_id"])
        self.assertEqual(binding["manifest_sha256"], submission["profile_data"]["profile_ground_truth_manifest_sha256"])

    def test_legacy_binding_is_reported_by_list_blockers(self) -> None:
        self.config["release_bindings"]["profile_ground_truth"] = {
            "release_id": self.spec["profile_definition"]["id"],
            "manifest_sha256_source": "profile_definition.sha256",
        }
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as output:
            path = Path(temporary) / "config.json"
            write_json(path, self.config)
            self.assertEqual(main(["--config", str(path), "--list-blockers"]), 1)
            self.assertIn("not the profile extraction definition", output.getvalue())

    def test_assembly_rejects_unresolved_truth_before_writing_output(self) -> None:
        self.config["release_bindings"]["profile_ground_truth"] = {
            "release_id": "__UNRESOLVED_AIRFRANS_PROFILE_GROUND_TRUTH_RELEASE_ID__",
            "manifest_sha256": "__UNRESOLVED_AIRFRANS_PROFILE_GROUND_TRUTH_MANIFEST_SHA256__",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "package"
            with self.assertRaises(AssembleError):
                assemble(config=self.config, scoring_support_manifest_path=Path("unused"),
                         metrics_cases_path=Path("unused"), predicted_profiles_dir=Path("unused"),
                         profile_score_path=Path("unused"), output_dir=output, spec_path=DEFAULT_SPEC)
            self.assertFalse(output.exists())

    def test_assembly_binds_explicit_truth_in_submission_and_evidence(self) -> None:
        # Small synthetic assembly fixture: this tests bindings, not field scoring.
        metrics = read(self.package / "metrics/cases.json")
        metrics["cases"] = metrics["cases"][:2]
        metrics["case_count"] = 2
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            truth = {"release_id": "test-profile-truth", "manifest_sha256": write_json(
                directory / "truth.json", {"data_release": {"id": "test-profile-truth"}})}
            config = copy.deepcopy(self.config)
            config["release_bindings"]["profile_ground_truth"] = truth
            write_json(directory / "support.json", {"release_id": metrics["scoring_support_release_id"]})
            write_json(directory / "metrics.json", metrics)
            write_json(directory / "score.json", {"value": metrics["metric_values"]["velocity_profile_r2"]})
            for case in metrics["cases"]:
                write_json(directory / "profiles" / (case["case_id"] + ".json"),
                           {"cases": [{"case_id": case["case_id"], "series": []}]})
            output = directory / "package"
            assemble(config=config, scoring_support_manifest_path=directory / "support.json",
                     metrics_cases_path=directory / "metrics.json", predicted_profiles_dir=directory / "profiles",
                     profile_score_path=directory / "score.json", output_dir=output, spec_path=DEFAULT_SPEC)
            submission = read(output / "submission.json")
            evidence = read(output / "evaluation-evidence.json")
            for binding in (submission["profile_data"], evidence):
                self.assertEqual(binding["profile_ground_truth_release_id"], truth["release_id"])
                self.assertEqual(binding["profile_ground_truth_manifest_sha256"], truth["manifest_sha256"])
            self.assertEqual(submission["evaluation"]["evidence_sha256"], sha256_file(output / "evaluation-evidence.json"))


if __name__ == "__main__":
    unittest.main()
