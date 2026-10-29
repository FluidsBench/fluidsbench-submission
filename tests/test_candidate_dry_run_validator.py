from __future__ import annotations

import copy
import io
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import validate_submission as validator


SOURCE_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = SOURCE_ROOT / "examples" / "v3-template"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class CandidateDryRunValidatorTests(unittest.TestCase):
    def make_registered_fixture(
        self,
        root: Path,
        *,
        lifecycle: str = "candidate",
    ) -> Path:
        """Create a tiny registered package with a complete local support chain."""

        self.assertIn(lifecycle, {"candidate", "official"})
        package = root / "submissions" / "synthetic" / "synthetic-open-model-v1"
        shutil.copytree(TEMPLATE, package)
        (package / "maintainer-validation.json").unlink()
        (package / "prediction-artifact-checks.json").unlink()

        dataset_directory = root / "benchmark-specs" / "synthetic"
        support_directory = (
            dataset_directory / "scoring-support" / "synthetic-support-v1"
        )
        shutil.copytree(TEMPLATE / "support", support_directory)
        support_manifest_path = support_directory / "manifest.json"
        support_manifest = load_json(support_manifest_path)
        approval = {
            "approved_by": "Synthetic fixture owner",
            "approved_at": "2026-07-27",
            "pull_request_url": "https://github.com/example/benchmark/pull/17",
        }
        support_manifest["status"] = lifecycle
        for support in support_manifest["supports"]:
            for binding in support["metric_bindings"]:
                if binding["weighting"] == "support_weights":
                    binding["dataset_weighting"] = "cell_volume"
        if lifecycle == "official":
            support_manifest["owner_approval"] = approval
        else:
            support_manifest.pop("owner_approval", None)
        write_json(support_manifest_path, support_manifest)
        shutil.copy2(support_manifest_path, package / "support" / "manifest.json")
        support_sha256 = validator.sha256_file(support_manifest_path)

        split_path = dataset_directory / "splits" / "default.json"
        write_json(
            split_path,
            {
                "schema_version": "1.0",
                "dataset_id": "synthetic",
                "split_id": "default",
                "case_set_id": "standard",
                "split_label": "Default",
                "case_id_status": "official",
                "case_count": 2,
                "case_ids": ["case-001", "case-002"],
            },
        )
        split_sha256 = validator.sha256_file(split_path)

        bindings = support_manifest["supports"][0]["metric_bindings"]
        metrics = [
            {
                "id": binding["metric_id"],
                "aggregation": binding["aggregation"],
                "weighting": binding["dataset_weighting"],
            }
            for binding in bindings
        ]
        ranking = {
            "metric_id": "pressure_rel_l1",
            "direction": "lower",
            "decimal_places": 3,
            "rounding": "decimal_half_up",
            "method": "competition",
        }
        manifest_url = (
            "https://example.org/scoring-support/"
            "synthetic-support-v1/manifest.json"
        )
        release_binding = {
            "status": lifecycle,
            "release_id": "synthetic-support-v1",
            "manifest_file": "scoring-support/synthetic-support-v1/manifest.json",
            "manifest_url": manifest_url,
            "manifest_sha256": support_sha256,
        }
        if lifecycle == "candidate":
            scoring_support = {
                "status": "owner_review_required",
                "submissions_open": False,
                "closed_reason": "Candidate support is awaiting dataset-owner review.",
                "owner_decisions_required": ["approve_candidate_support"],
                "candidate_manifest": release_binding,
            }
            dataset_status = "candidate"
        else:
            scoring_support = {
                **release_binding,
                "status": "official",
                "submissions_open": True,
                "owner_approval": approval,
            }
            dataset_status = "official"
        specification = {
            "schema_version": "1.1",
            "dataset_id": "synthetic",
            "dataset_name": "Synthetic candidate fixture",
            "dataset_version": "synthetic-1",
            "status": dataset_status,
            "evaluation_reference_version": "synthetic-evaluator-v1",
            "ranking": ranking,
            "metrics": metrics,
            "scoring_support": scoring_support,
            "splits": [
                {
                    "id": "default",
                    "label": "Default",
                    "index_file": "splits/default.json",
                    "case_count": 2,
                    "case_set_id": "standard",
                    "case_id_status": "official",
                    "sha256": split_sha256,
                }
            ],
            "profile_panels": [
                {
                    "id": "synthetic_profile",
                    "required": True,
                    "minimum_points": 2,
                    "station_ids": ["centreline"],
                    "quantity_ids": ["pressure"],
                }
            ],
        }
        write_json(dataset_directory / "submission-spec.json", specification)

        submission_path = package / "submission.json"
        submission = load_json(submission_path)
        write_json(
            dataset_directory / "methodology-contract.json",
            {
                "$schema": (
                    "https://fluidsbench.org/schemas/"
                    "methodology-contract-v1.schema.json"
                ),
                "format": "fluidsbench-methodology-contract-v1",
                "dataset_id": "synthetic",
                "allow_additional_predicted_fields": True,
                "required_predicted_fields": [
                    {
                        "field_id": "synthetic-support.pressure",
                        "domain": "domain",
                        "component_count": 1,
                        "description": (
                            "Scalar pressure on the synthetic scoring support."
                        ),
                    }
                ],
            },
        )
        submission["dataset"] = "Synthetic candidate fixture"
        submission["split"] = "Default"
        submission["split_sha256"] = split_sha256
        submission["methodology"]["record_kind"] = "submitter_reported"
        submission["methodology"]["record_note"] = (
            "Synthetic submitter-reported fixture used to exercise candidate validation."
        )
        submission["scoring_support"].update(
            {
                "status": lifecycle,
                "manifest_url": manifest_url,
                "manifest_sha256": support_sha256,
            }
        )
        submission.pop("approval", None)

        prediction_manifest_path = package / "predictions" / "manifest.json"
        prediction_manifest = load_json(prediction_manifest_path)
        prediction_manifest["support_manifest_sha256"] = support_sha256
        write_json(prediction_manifest_path, prediction_manifest)
        submission["prediction_artifacts"][0][
            "support_manifest_sha256"
        ] = support_sha256
        submission["prediction_artifacts"][0]["manifest_sha256"] = (
            validator.sha256_file(prediction_manifest_path)
        )

        case_metrics_path = package / "metrics" / "cases.json"
        case_metrics = load_json(case_metrics_path)
        case_metrics["scoring_support_manifest_sha256"] = support_sha256
        for case in case_metrics["cases"]:
            for support in case["supports"]:
                for statistics in support.get(
                    "metric_sufficient_statistics", {}
                ).values():
                    if statistics["weighting"] == "support_weights":
                        statistics["dataset_weighting"] = "cell_volume"
        write_json(case_metrics_path, case_metrics)
        case_metrics_sha256 = validator.sha256_file(case_metrics_path)
        submission["case_metrics"]["sha256"] = case_metrics_sha256

        discretization_path = package / "discretization.json"
        discretization = load_json(discretization_path)
        discretization["scoring_support_manifest_sha256"] = support_sha256
        discretization["inference"]["surface_input"][
            "case_record_id"
        ] = "surface-geometry-input"
        write_json(discretization_path, discretization)
        discretization_sha256 = validator.sha256_file(discretization_path)
        submission["spatial_discretization"]["sha256"] = discretization_sha256

        evidence_path = package / "evaluation-evidence.json"
        evidence = load_json(evidence_path)
        evidence["split_sha256"] = split_sha256
        evidence["scoring_support_manifest_sha256"] = support_sha256
        evidence["case_metrics_sha256"] = case_metrics_sha256
        evidence["discretization_sha256"] = discretization_sha256
        write_json(evidence_path, evidence)
        submission["evaluation"]["evidence_sha256"] = validator.sha256_file(
            evidence_path
        )
        write_json(submission_path, submission)

        metric_definitions = [
            {
                "id": metric["id"],
                "direction": "lower",
                "kind": "error",
                "unit": "%" if "rel_" in metric["id"] else "Pa",
            }
            for metric in metrics
        ]
        leaderboard_manifest = {
            "data_release": {
                "profile_ground_truth": {
                    "release_id": "synthetic-profile-ground-truth-v1",
                    "manifest_sha256": "b" * 64,
                }
            },
            "datasets": [
                {
                    "name": "Synthetic candidate fixture",
                    "slug": "synthetic",
                    "splits": [{"id": "default", "name": "Default"}],
                    "metric_ids": [metric["id"] for metric in metrics],
                    "ranking": ranking,
                    "scoring_support": copy.deepcopy(scoring_support),
                }
            ],
            "metric_definitions": metric_definitions,
        }
        write_json(root / "leaderboard" / "manifest.json", leaderboard_manifest)
        return submission_path

    @contextmanager
    def repository_view(self, root: Path):
        with patch.object(validator, "ROOT", root), patch.object(
            validator, "MANIFEST_PATH", root / "leaderboard" / "manifest.json"
        ):
            yield

    def test_candidate_mode_validates_registered_closed_candidate_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission_path = self.make_registered_fixture(root)
            with self.repository_view(root):
                errors, stats = validator.validate_submission_file(
                    submission_path,
                    candidate_dry_run=True,
                )
                normal_errors, _ = validator.validate_submission_file(submission_path)
                contributor_errors, _ = validator.validate_submission_file(
                    submission_path,
                    contributor_stage=True,
                )
            self.assertEqual(errors, [])
            self.assertEqual(stats, {"cases": 2, "series": 2})
            self.assertIn("submissions are closed", "\n".join(normal_errors))
            self.assertIn("submissions are closed", "\n".join(contributor_errors))

    def test_candidate_mode_validates_an_isolated_assembled_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registered_path = self.make_registered_fixture(root)
            isolated_package = root / "candidate-build" / registered_path.parent.name
            isolated_package.parent.mkdir()
            shutil.move(registered_path.parent, isolated_package)
            submission_path = isolated_package / "submission.json"

            with self.repository_view(root):
                candidate_errors, stats = validator.validate_submission_file(
                    submission_path,
                    candidate_dry_run=True,
                )
                normal_errors, _ = validator.validate_submission_file(submission_path)

            self.assertEqual(candidate_errors, [])
            self.assertEqual(stats, {"cases": 2, "series": 2})
            self.assertIn(
                "submission.json must be stored at submissions/synthetic/",
                "\n".join(normal_errors),
            )

    def test_candidate_cli_never_prints_official_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission_path = self.make_registered_fixture(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.repository_view(root), redirect_stdout(stdout), redirect_stderr(
                stderr
            ):
                result = validator.main(
                    ["--candidate-dry-run", str(submission_path.parent)]
                )
            self.assertEqual(result, 0, stderr.getvalue())
            self.assertIn("CANDIDATE DRY-RUN VALID", stdout.getvalue())
            self.assertIn("not official acceptance", stdout.getvalue())
            self.assertNotIn("PASS", stdout.getvalue())

    def test_candidate_dry_run_allows_omitted_participant_code_identity(self) -> None:
        """Evaluator revision belongs to owner support, not evaluation.code_revision."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission_path = self.make_registered_fixture(root)
            submission = load_json(submission_path)
            submission.pop("code_url", None)
            submission["reproducibility"].pop("code", None)
            submission["evaluation"].pop("code_revision", None)
            evidence_path = submission_path.parent / "evaluation-evidence.json"
            evidence = load_json(evidence_path)
            evidence.pop("code_revision", None)
            write_json(evidence_path, evidence)
            submission["evaluation"]["evidence_sha256"] = validator.sha256_file(
                evidence_path
            )
            write_json(submission_path, submission)
            with self.repository_view(root):
                errors, stats = validator.validate_submission_file(
                    submission_path,
                    candidate_dry_run=True,
                )
            self.assertEqual(errors, [])
            self.assertEqual(stats, {"cases": 2, "series": 2})

    def test_candidate_mode_rejects_unsafe_owner_and_maintainer_states(self) -> None:
        mutations = (
            ("owner official", "owner"),
            ("submissions open", "open"),
            ("missing binding", "binding"),
            ("wrong binding", "hash"),
            ("approval", "approval"),
            ("maintainer file", "maintainer"),
        )
        for label, mutation in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                submission_path = self.make_registered_fixture(root)
                specification_path = (
                    root / "benchmark-specs" / "synthetic" / "submission-spec.json"
                )
                specification = load_json(specification_path)
                submission = load_json(submission_path)
                if mutation == "owner":
                    specification["status"] = "official"
                    specification["scoring_support"]["status"] = "official"
                elif mutation == "open":
                    specification["scoring_support"]["submissions_open"] = True
                elif mutation == "binding":
                    specification["scoring_support"].pop("candidate_manifest")
                elif mutation == "hash":
                    specification["scoring_support"]["candidate_manifest"][
                        "manifest_sha256"
                    ] = "f" * 64
                elif mutation == "approval":
                    submission["approval"] = load_json(
                        TEMPLATE / "submission.json"
                    ).get("approval", {
                        "status": "approved",
                        "approved_by": "Maintainer",
                        "approved_at": "2026-07-27",
                        "pull_request_url": "https://github.com/example/repo/pull/1",
                        "validation": {
                            "evidence_file": "maintainer-validation.json",
                            "evidence_sha256": "a" * 64,
                        },
                    })
                else:
                    (submission_path.parent / "maintainer-validation.json").write_text(
                        "{}\n", encoding="utf-8"
                    )
                write_json(specification_path, specification)
                write_json(submission_path, submission)
                with self.repository_view(root):
                    errors, _ = validator.validate_submission_file(
                        submission_path,
                        candidate_dry_run=True,
                    )
                self.assertTrue(errors)
                self.assertIn("candidate", "\n".join(errors).lower())

    def test_official_package_uses_normal_path_and_candidate_mode_rejects_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission_path = self.make_registered_fixture(
                root,
                lifecycle="official",
            )
            with self.repository_view(root):
                normal_errors, stats = validator.validate_submission_file(
                    submission_path
                )
                candidate_errors, _ = validator.validate_submission_file(
                    submission_path,
                    candidate_dry_run=True,
                )
            self.assertEqual(normal_errors, [])
            self.assertEqual(stats, {"cases": 2, "series": 2})
            self.assertIn("never an official one", "\n".join(candidate_errors))
            self.assertIn(
                "requires submission scoring_support.status='candidate'",
                "\n".join(candidate_errors),
            )

    def test_lifecycle_flags_are_mutually_exclusive(self) -> None:
        errors, totals = validator.validate_many(
            [],
            contributor_stage=True,
            candidate_dry_run=True,
        )
        self.assertEqual(
            errors,
            ["--contributor-stage and --candidate-dry-run are mutually exclusive"],
        )
        self.assertEqual(totals["submissions"], 0)
        with self.assertRaises(SystemExit) as raised, redirect_stderr(io.StringIO()):
            validator.main(["--contributor-stage", "--candidate-dry-run"])
        self.assertEqual(raised.exception.code, 2)

    def test_candidate_schema_status_is_explicit_and_non_approving(self) -> None:
        submission = load_json(TEMPLATE / "submission.json")
        submission["scoring_support"]["status"] = "candidate"
        self.assertEqual(
            validator.schema_errors(
                submission,
                "submission.schema.json",
                schema_version="v3",
            ),
            [],
        )
        submission["approval"] = {
            "status": "approved",
            "approved_by": "Maintainer",
            "approved_at": "2026-07-27",
            "pull_request_url": "https://github.com/example/repo/pull/1",
            "validation": {
                "evidence_file": "maintainer-validation.json",
                "evidence_sha256": "a" * 64,
            },
        }
        self.assertTrue(
            validator.schema_errors(
                submission,
                "submission.schema.json",
                schema_version="v3",
            )
        )

        support_manifest = load_json(TEMPLATE / "support" / "manifest.json")
        support_manifest["status"] = "candidate"
        support_manifest.pop("owner_approval")
        self.assertEqual(
            validator.schema_errors(
                support_manifest,
                "manifest.schema.json",
                schema_version="scoring-support/v1",
            ),
            [],
        )
        support_manifest["owner_approval"] = {
            "approved_by": "Must not appear",
            "approved_at": "2026-07-27",
            "pull_request_url": "https://github.com/example/repo/pull/1",
        }
        self.assertTrue(
            validator.schema_errors(
                support_manifest,
                "manifest.schema.json",
                schema_version="scoring-support/v1",
            )
        )


if __name__ == "__main__":
    unittest.main()
