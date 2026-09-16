from __future__ import annotations

import hashlib
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import manage_leaderboard


class ManageLeaderboardTests(unittest.TestCase):
    def ranking_manifest(self, *, direction: str = "higher", decimal_places: int = 1) -> dict:
        return {
            "schema_version": "0.6.0",
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "metric_definitions": [
                {
                    "id": "score",
                    "unit": "points",
                    "digits": decimal_places,
                    "direction": direction,
                }
            ],
            "data_release": {"status": "prototype_dummy_data"},
            "datasets": [
                {
                    "name": "Example",
                    "slug": "example",
                    "ranking": {
                        "metric_id": "score",
                        "direction": direction,
                        "decimal_places": decimal_places,
                        "rounding": "decimal_half_up",
                        "method": "competition",
                    },
                }
            ],
        }

    def test_feed_row_pins_profile_index_bytes(self) -> None:
        path = manage_leaderboard.ROOT / "submissions" / "ahmedml" / "transolver" / "submission.json"
        manifest = {
            "data_release": {"status": "prototype_dummy_data"},
            "datasets": [{"name": "AhmedML"}],
        }
        with patch.object(manage_leaderboard, "submission_files", return_value=[path]):
            row = manage_leaderboard.source_rows_by_dataset(manifest)["AhmedML"][0]
        profile_index_path = manage_leaderboard.ROOT / row["profile_data"]["index_file"]
        self.assertEqual(row["profile_data"]["index_sha256"], manage_leaderboard.sha256_file(profile_index_path))

    def test_only_latest_revision_is_ranked_and_all_versions_remain_in_history(self) -> None:
        rows = {
            "Example": [
                {
                    "submission_id": "team-model-v1",
                    "dataset_id": "example",
                    "split_id": "full",
                    "submitted_at": "2026-08-01",
                    "result_revision": {
                        "series_id": "team-model",
                        "version": 1,
                        "supersedes": None,
                        "change_summary": "Initial result.",
                    },
                },
                {
                    "submission_id": "team-model-v2",
                    "dataset_id": "example",
                    "split_id": "full",
                    "submitted_at": "2026-08-02",
                    "result_revision": {
                        "series_id": "team-model",
                        "version": 2,
                        "supersedes": "team-model-v1",
                        "change_summary": "Updated result.",
                    },
                },
            ]
        }
        manage_leaderboard.annotate_result_revisions(rows)
        latest = manage_leaderboard.latest_rows_by_dataset(rows)
        self.assertEqual([row["submission_id"] for row in latest["Example"]], ["team-model-v2"])
        self.assertFalse(rows["Example"][0]["result_revision"]["is_latest"])
        self.assertTrue(rows["Example"][1]["result_revision"]["is_latest"])
        history = manage_leaderboard.revision_history_payload(
            {
                "data_release": {
                    "id": "example-release",
                    "status": "official",
                    "generated_at": "2026-08-02T00:00:00Z",
                }
            },
            rows,
        )
        self.assertEqual(history["record_count"], 2)
        self.assertEqual(history["series_count"], 1)
        self.assertEqual(manage_leaderboard.revision_history_semantic_errors(history), [])

    def test_prototype_feed_contains_only_prototype_rows(self) -> None:
        paths = [
            manage_leaderboard.ROOT / "submissions" / "ahmedml" / submission_id / "submission.json"
            for submission_id in ("approved", "prototype", "pending")
        ]
        submissions = {
            path: {
                "submission_id": path.parent.name,
                "dataset": "AhmedML",
                "approval": {"status": path.parent.name} if path.parent.name != "pending" else {},
                "profile_data": {"index_file": "profiles/index.json"},
            }
            for path in paths
        }
        manifest = {
            "data_release": {"status": "prototype_dummy_data"},
            "datasets": [{"name": "AhmedML"}],
        }
        with (
            patch.object(manage_leaderboard, "submission_files", return_value=paths),
            patch.object(manage_leaderboard, "load_json", side_effect=lambda path: submissions[Path(path)]),
        ):
            rows = manage_leaderboard.source_rows_by_dataset(manifest)["AhmedML"]
        self.assertEqual([row["submission_id"] for row in rows], ["prototype"])

    def test_official_feed_contains_only_approved_rows(self) -> None:
        paths = [
            manage_leaderboard.ROOT / "submissions" / "ahmedml" / submission_id / "submission.json"
            for submission_id in ("approved", "prototype", "pending")
        ]
        submissions = {
            path: {
                "submission_id": path.parent.name,
                "dataset": "AhmedML",
                "approval": {"status": path.parent.name} if path.parent.name != "pending" else {},
                "profile_data": {"index_file": "profiles/index.json"},
            }
            for path in paths
        }
        manifest = {"data_release": {"status": "official"}, "datasets": [{"name": "AhmedML"}]}
        submissions[paths[0]]["approval"]["validation"] = {
            "evidence_file": "maintainer-validation.json",
            "evidence_sha256": "a" * 64,
        }
        validation = {
            "schema_version": "2.0",
            "status": "validated",
            "contract_version": "open-reproducibility-2.0",
            "reference_version": "v1",
            "case_set_id": "standard",
            "profile_ground_truth_release_id": "gt-v1",
            "profile_ground_truth_manifest_sha256": "d" * 64,
            "validated_by": "Maintainer",
            "validated_at": "2026-07-20T00:00:00Z",
            "validation_scope": "submitted_data_only",
            "model_execution": "not_performed",
            "metric_recomputation": "not_performed",
            "reviewed_submission_sha256": "b" * 64,
            "evaluation_evidence_sha256": "e" * 64,
            "profile_index_sha256": "c" * 64,
        }
        original_load = lambda path: (
            validation if Path(path).name == "maintainer-validation.json" else submissions[Path(path)]
        )
        with (
            patch.object(manage_leaderboard, "submission_files", return_value=paths),
            patch.object(manage_leaderboard, "load_json", side_effect=original_load),
        ):
            rows = manage_leaderboard.source_rows_by_dataset(manifest)["AhmedML"]
        self.assertEqual([row["submission_id"] for row in rows], ["approved"])
        self.assertEqual(rows[0]["maintainer_validation"]["schema_version"], "2.0")
        self.assertEqual(rows[0]["maintainer_validation"]["status"], "validated")
        self.assertEqual(rows[0]["maintainer_validation"]["reviewed_submission_sha256"], "b" * 64)
        self.assertEqual(rows[0]["maintainer_validation"]["profile_index_sha256"], "c" * 64)
        self.assertEqual(rows[0]["maintainer_validation"]["model_execution"], "not_performed")
        self.assertEqual(rows[0]["maintainer_validation"]["profile_ground_truth_release_id"], "gt-v1")
        self.assertTrue(rows[0]["maintainer_validation"]["evidence_path"].endswith("maintainer-validation.json"))

    def test_prototype_release_id_contains_feed_digest(self) -> None:
        manifest = {
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "datasets": [{"name": "AhmedML", "file": "leaderboard/datasets/ahmedml.json"}],
            "all_file": "leaderboard/all.json",
            "data_release": {"status": "prototype_dummy_data"},
        }
        rows = {"AhmedML": [{"submission_id": "example", "submitted_at": "2026-07-15"}]}
        with patch.object(manage_leaderboard, "source_rows_by_dataset", return_value=rows):
            updated, _, all_rows, revision_history = manage_leaderboard.expected_outputs(manifest)
        digest = hashlib.sha256(manage_leaderboard.json_bytes(all_rows)).hexdigest()
        self.assertEqual(updated["data_release"]["feed_sha256"], digest)
        self.assertEqual(updated["data_release"]["id"], f"prototype-dev-2026-07-15-{digest[:12]}")
        self.assertEqual(revision_history["record_count"], 1)
        self.assertEqual(revision_history["series_count"], 1)

    def test_official_release_requires_archive_and_license_provenance(self) -> None:
        manifest = {"data_release": {"status": "official", "archive_url": None}}
        errors = manage_leaderboard.release_contract_errors(manifest)
        self.assertIn("an official data release requires an immutable HTTPS data_release.archive_url", errors)
        self.assertIn("an official data release requires a full immutable data_release.source_commit", errors)
        self.assertTrue(any("release_view_url" in error for error in errors))
        self.assertTrue(any("asset_base_url" in error for error in errors))
        self.assertTrue(any("profile_ground_truth" in error for error in errors))
        self.assertTrue(any("data_release.license" in error for error in errors))

    def test_prototype_release_may_have_no_archive(self) -> None:
        manifest = {
            "schema_version": "0.6.0",
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "metric_definitions": [],
            "datasets": [],
            "data_release": {
                "status": "prototype_dummy_data",
                "archive_url": None,
                "release_view_url": None,
                "asset_base_url": "https://raw.githubusercontent.com/neilashton/fluidsbench-submission/dev/",
                "reproducibility_contract_version": "open-reproducibility-2.0",
                "profile_ground_truth": {
                    "release_id": "prototype-profile-ground-truth-1",
                    "manifest_url": "https://fluidsbench.org/assets/data/profile-ground-truth/manifest.json",
                    "manifest_sha256": "a" * 64,
                },
                "license": {
                    "spdx_id": "Apache-2.0",
                    "name": "Apache License 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0",
                    "scope": "Leaderboard metadata and contributed profiles.",
                },
            }
        }
        self.assertEqual(manage_leaderboard.release_contract_errors(manifest), [])

    def test_official_release_uses_release_scoped_asset_and_view_urls(self) -> None:
        release_id = "fluidsbench-2026-07"
        generated_at = "2026-07-22T00:00:00Z"
        manifest = {
            "schema_version": "0.6.0",
            "generated_at": generated_at,
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "metric_definitions": [],
            "datasets": [],
            "data_release": {
                "id": release_id,
                "status": "official",
                "generated_at": generated_at,
                "archive_url": "https://doi.org/10.0000/example",
                "release_view_url": f"https://fluidsbench.org/releases/{release_id}/",
                "asset_base_url": f"https://assets.fluidsbench.org/releases/{release_id}/",
                "source_commit": "a" * 40,
                "reproducibility_contract_version": "open-reproducibility-2.0",
                "profile_ground_truth": {
                    "release_id": "ground-truth-1",
                    "manifest_url": "https://fluidsbench.org/releases/ground-truth-1/manifest.json",
                    "manifest_sha256": "b" * 64,
                },
                "license": {
                    "spdx_id": "Apache-2.0",
                    "name": "Apache License 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0",
                    "scope": "Published leaderboard metadata.",
                },
            },
        }
        self.assertEqual(manage_leaderboard.release_contract_errors(manifest), [])
        invalid_base = deepcopy(manifest)
        invalid_base["data_release"]["release_view_url"] += "?lang=~#old"
        invalid_base["data_release"]["asset_base_url"] = invalid_base["data_release"]["asset_base_url"].rstrip("/")
        invalid_errors = manage_leaderboard.release_contract_errors(invalid_base)
        self.assertTrue(any("release_view_url" in error for error in invalid_errors))
        self.assertTrue(any("asset_base_url" in error for error in invalid_errors))
        mutable_ground_truth = deepcopy(manifest)
        mutable_ground_truth["data_release"]["profile_ground_truth"]["manifest_url"] = (
            "https://fluidsbench.org/assets/data/profile-ground-truth/manifest.json"
        )
        self.assertTrue(
            any(
                "profile-ground-truth manifest_url" in error
                for error in manage_leaderboard.release_contract_errors(mutable_ground_truth)
            )
        )

        root_scoped = deepcopy(manifest)
        root_scoped["data_release"].update(
            {
                "id": "/",
                "release_view_url": "https://fluidsbench.org/",
                "asset_base_url": "https://assets.fluidsbench.org/",
            }
        )
        root_scoped["data_release"]["profile_ground_truth"].update(
            {"release_id": "/", "manifest_url": "https://fluidsbench.org/manifest.json"}
        )
        root_errors = manage_leaderboard.release_contract_errors(root_scoped)
        self.assertTrue(any("data_release.id" in error for error in root_errors))
        self.assertTrue(any("release_view_url" in error for error in root_errors))
        self.assertTrue(any("asset_base_url" in error for error in root_errors))
        self.assertTrue(any("profile-ground-truth" in error for error in root_errors))

        partial_segment = deepcopy(manifest)
        partial_segment["data_release"]["release_view_url"] = (
            f"https://fluidsbench.org/releases/{release_id}-old/"
        )
        partial_segment["data_release"]["asset_base_url"] = (
            f"https://assets.fluidsbench.org/releases/{release_id}-old/"
        )
        partial_errors = manage_leaderboard.release_contract_errors(partial_segment)
        self.assertTrue(any("release_view_url" in error for error in partial_errors))
        self.assertTrue(any("asset_base_url" in error for error in partial_errors))

        malformed_https = deepcopy(manifest)
        malformed_https["data_release"]["archive_url"] = "https://"
        malformed_https["data_release"]["profile_ground_truth"]["manifest_url"] = (
            "https:///releases/ground-truth-1/manifest.json"
        )
        malformed_errors = manage_leaderboard.release_contract_errors(malformed_https)
        self.assertTrue(any("archive_url" in error for error in malformed_errors))
        self.assertTrue(any("profile_ground_truth" in error for error in malformed_errors))

    def test_competition_ranking_uses_published_precision(self) -> None:
        manifest = self.ranking_manifest()
        rows = [
            {
                "submission_id": submission_id,
                "split_id": "full",
                "metric_values": {"score": value},
                "approval": {"status": "prototype"},
            }
            for submission_id, value in (
                ("model-a", 10.01),
                ("model-b", 9.94),
                ("model-c", 9.93),
                ("model-d", 9.84),
            )
        ]
        manage_leaderboard.add_release_rankings(manifest, {"Example": rows})
        by_id = {row["submission_id"]: row["ranking"] for row in rows}
        self.assertEqual([by_id[key]["rank"] for key in ("model-a", "model-b", "model-c", "model-d")], [1, 2, 2, 4])
        self.assertEqual(by_id["model-b"]["display_value"], "9.9")
        self.assertEqual(by_id["model-c"]["ranked_value"], 9.9)
        self.assertTrue(by_id["model-b"]["tied"])
        self.assertEqual(by_id["model-b"]["tie_count"], 2)
        self.assertEqual(by_id["model-a"]["ranked_result_count"], 4)
        self.assertFalse(rows[0]["claim_eligibility"]["academic_citation"])

    def test_decimal_half_up_is_not_binary_float_rounding(self) -> None:
        rounded, json_value, display = manage_leaderboard.published_metric_value(9.95, 1)
        self.assertEqual(str(rounded), "10.0")
        self.assertEqual(json_value, 10.0)
        self.assertEqual(display, "10.0")

    def test_published_metric_normalizes_signed_zero(self) -> None:
        negative, negative_value, negative_display = manage_leaderboard.published_metric_value(-0.0, 1)
        positive, positive_value, positive_display = manage_leaderboard.published_metric_value(0.0, 1)
        self.assertEqual(negative, positive)
        self.assertEqual(negative_value, positive_value)
        self.assertEqual(negative_display, "0.0")
        self.assertEqual(positive_display, "0.0")

        manifest = self.ranking_manifest()
        rows = [
            {
                "submission_id": submission_id,
                "split_id": "full",
                "metric_values": {"score": value},
                "approval": {"status": "prototype"},
            }
            for submission_id, value in (("negative-zero", -0.0), ("positive-zero", 0.0))
        ]
        manage_leaderboard.add_release_rankings(manifest, {"Example": rows})
        self.assertEqual([row["ranking"]["display_value"] for row in rows], ["0.0", "0.0"])
        self.assertEqual([row["ranking"]["rank"] for row in rows], [1, 1])
        self.assertEqual([row["ranking"]["tie_count"] for row in rows], [2, 2])

    def test_lower_is_better_ranks_are_scoped_to_split(self) -> None:
        manifest = self.ranking_manifest(direction="lower")
        rows = [
            {
                "submission_id": submission_id,
                "split_id": split_id,
                "metric_values": {"score": value},
                "approval": {"status": "prototype"},
            }
            for submission_id, split_id, value in (
                ("model-a", "full", 1.01),
                ("model-b", "full", 1.04),
                ("model-c", "full", 1.11),
                ("model-d", "scarce", 9.99),
            )
        ]
        manage_leaderboard.add_release_rankings(manifest, {"Example": rows})
        by_id = {row["submission_id"]: row["ranking"] for row in rows}
        self.assertEqual([by_id[key]["rank"] for key in ("model-a", "model-b", "model-c")], [1, 1, 3])
        self.assertEqual(by_id["model-a"]["ranked_result_count"], 3)
        self.assertEqual(by_id["model-d"]["rank"], 1)
        self.assertEqual(by_id["model-d"]["ranked_result_count"], 1)

    def test_result_and_claim_urls_use_clean_release_bases(self) -> None:
        release = {
            "status": "official",
            "release_view_url": "https://fluidsbench.org/releases/release-1/",
            "asset_base_url": "https://assets.fluidsbench.org/releases/release-1/",
        }
        row = {"dataset_id": "ahmedml", "split_id": "full", "submission_id": "model-a"}
        self.assertEqual(
            manage_leaderboard.result_permalink(release, row),
            "https://fluidsbench.org/releases/release-1/?view=result&dataset=ahmedml&split=full&result=model-a",
        )
        self.assertEqual(
            manage_leaderboard.immutable_claim_record_url(release, row),
            "https://assets.fluidsbench.org/releases/release-1/leaderboard/claims/ahmedml/full/model-a.json",
        )

    def test_ranking_precision_must_match_display_digits(self) -> None:
        manifest = self.ranking_manifest(decimal_places=1)
        manifest["datasets"][0]["ranking"]["decimal_places"] = 2
        errors = manage_leaderboard.release_contract_errors(manifest)
        self.assertTrue(any("must match the metric display digits" in error for error in errors))

    def test_metric_definition_overrides_are_presentation_only_and_dataset_scoped(self) -> None:
        manifest = self.ranking_manifest()
        manifest["metric_definitions"].append(
            {"id": "unused", "unit": "%", "digits": 2, "direction": "lower"}
        )
        dataset = manifest["datasets"][0]
        dataset["metric_ids"] = ["score"]
        dataset["metric_definition_overrides"] = {
            "score": {
                "label": "Dataset score",
                "description": "Dataset-specific presentation text.",
            }
        }
        valid_errors = manage_leaderboard.release_contract_errors(manifest)
        self.assertFalse(any("metric_definition_overrides" in error for error in valid_errors))

        unknown = deepcopy(manifest)
        unknown["datasets"][0]["metric_definition_overrides"] = {
            "missing": {"label": "Missing metric"}
        }
        self.assertTrue(
            any(
                "references unknown metric missing" in error
                for error in manage_leaderboard.release_contract_errors(unknown)
            )
        )

        inactive = deepcopy(manifest)
        inactive["datasets"][0]["metric_definition_overrides"] = {
            "unused": {"label": "Inactive metric"}
        }
        self.assertTrue(
            any(
                "metric unused is not in metric_ids" in error
                for error in manage_leaderboard.release_contract_errors(inactive)
            )
        )

        semantic_override = deepcopy(manifest)
        semantic_override["datasets"][0]["metric_definition_overrides"] = {
            "score": {"label": "Changed score", "direction": "lower"}
        }
        self.assertTrue(
            any(
                "may only define label and description" in error
                for error in manage_leaderboard.release_contract_errors(semantic_override)
            )
        )

        empty_text = deepcopy(manifest)
        empty_text["datasets"][0]["metric_definition_overrides"] = {
            "score": {"description": "  "}
        }
        self.assertTrue(
            any(
                "description must be a non-empty string" in error
                for error in manage_leaderboard.release_contract_errors(empty_text)
            )
        )

    def test_official_generation_preserves_explicit_timestamp_and_has_no_today_fallback(self) -> None:
        generated_at = "2026-07-20T12:00:00Z"
        manifest = self.ranking_manifest()
        manifest.update(
            {
                "generated_at": generated_at,
                "all_file": "leaderboard/all.json",
            }
        )
        manifest["datasets"][0]["file"] = "leaderboard/datasets/example.json"
        manifest["data_release"] = {
            "id": "release-1",
            "status": "official",
            "generated_at": generated_at,
        }
        with patch.object(
            manage_leaderboard,
            "source_rows_by_dataset",
            return_value={"Example": []},
        ):
            first, _, first_rows, _ = manage_leaderboard.expected_outputs(
                manifest, generated_at="2030-01-01T00:00:00Z"
            )
            second, _, second_rows, _ = manage_leaderboard.expected_outputs(
                manifest, generated_at="2040-01-01T00:00:00Z"
            )
        self.assertEqual(first, second)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first["generated_at"], generated_at)
        self.assertEqual(first["data_release"]["generated_at"], generated_at)
        self.assertIsNone(first["datasets"][0]["updated_at"])

    def test_claim_index_semantic_counts_are_verified(self) -> None:
        index = {
            "record_count": 7,
            "eligible_record_count": 99,
            "records": [{"eligible": True}, {"eligible": False}],
        }
        self.assertEqual(
            manage_leaderboard.claim_index_semantic_errors(index),
            [
                "record_count must equal the number of claim-index records",
                "eligible_record_count must equal the number of eligible claim-index records",
            ],
        )
        index.update({"record_count": 2, "eligible_record_count": 1})
        self.assertEqual(manage_leaderboard.claim_index_semantic_errors(index), [])

    def test_existing_official_release_is_sealed_but_a_new_id_is_allowed(self) -> None:
        record_path = "leaderboard/claims/example/full/model-a.json"
        original_record = {
            "release": {
                "id": "release-1",
                "status": "official",
                "published_at": "2026-07-20T00:00:00Z",
                "archive_url": "https://doi.org/10.0000/example",
                "release_view_url": "https://fluidsbench.org/releases/release-1/",
            },
            "claim_record_url": (
                "https://assets.fluidsbench.org/releases/release-1/"
                "leaderboard/claims/example/full/model-a.json"
            ),
        }
        original_sha256 = hashlib.sha256(manage_leaderboard.json_bytes(original_record)).hexdigest()
        original_index = {
            "release_id": "release-1",
            "release_status": "official",
            "feed_sha256": "a" * 64,
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "record_count": 1,
            "eligible_record_count": 1,
            "records": [
                {
                    "file": record_path,
                    "sha256": original_sha256,
                    "eligible": True,
                }
            ],
        }
        original_manifest = {"data_release": {"id": "release-1", "status": "official"}}

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index_path = root / "leaderboard" / "claims" / "index.json"
            manage_leaderboard.write_json(index_path, original_index)
            manage_leaderboard.write_json(root / record_path, original_record)
            with (
                patch.object(manage_leaderboard, "ROOT", root),
                patch.object(manage_leaderboard, "CLAIMS_INDEX_PATH", index_path),
            ):
                self.assertEqual(
                    manage_leaderboard.official_release_seal_errors(
                        original_manifest,
                        original_index,
                        {record_path: original_record},
                    ),
                    [],
                )

                changed_feed = deepcopy(original_index)
                changed_feed["feed_sha256"] = "b" * 64
                feed_errors = manage_leaderboard.official_release_seal_errors(
                    original_manifest,
                    changed_feed,
                    {record_path: original_record},
                )
                self.assertTrue(any("feed digest changed" in error for error in feed_errors))

                changed_record = deepcopy(original_record)
                changed_record["release"]["published_at"] = "2026-07-21T00:00:00Z"
                changed_index = deepcopy(original_index)
                changed_index["records"][0]["sha256"] = hashlib.sha256(
                    manage_leaderboard.json_bytes(changed_record)
                ).hexdigest()
                metadata_errors = manage_leaderboard.official_release_seal_errors(
                    original_manifest,
                    changed_index,
                    {record_path: changed_record},
                )
                self.assertTrue(any("published release metadata changed" in error for error in metadata_errors))

                new_manifest = {"data_release": {"id": "release-2", "status": "official"}}
                new_index = deepcopy(changed_index)
                new_index["release_id"] = "release-2"
                self.assertEqual(
                    manage_leaderboard.official_release_seal_errors(
                        new_manifest,
                        new_index,
                        {record_path: changed_record},
                    ),
                    [],
                )

    def test_claim_index_hashes_records_and_prototype_is_ineligible(self) -> None:
        row = {
            "submission_id": "model-a",
            "model": "Model A",
            "dataset": "Example",
            "dataset_id": "example",
            "split": "Full",
            "split_id": "full",
            "evaluation": {"evidence_file": "evaluation-evidence.json"},
            "profile_data": {"index_file": "submissions/example/model-a/profiles/index.json"},
            "ranking": {
                "metric_id": "score",
                "value": 9.94,
                "ranked_value": 9.9,
                "display_value": "9.9",
                "unit": "points",
                "direction": "higher",
                "decimal_places": 1,
                "rounding": "decimal_half_up",
                "method": "competition",
                "rank": 1,
                "ranked_result_count": 1,
                "tied": False,
                "tie_count": 1,
            },
            "claim_eligibility": manage_leaderboard.claim_eligibility("prototype_dummy_data", {}),
        }
        manifest = {
            "all_file": "leaderboard/all.json",
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "data_release": {
                "id": "prototype-1",
                "status": "prototype_dummy_data",
                "generated_at": "2026-07-22T00:00:00Z",
                "feed_sha256": "a" * 64,
                "archive_url": None,
                "release_view_url": None,
                "asset_base_url": "https://raw.githubusercontent.com/example/repo/dev/",
            },
        }
        with patch.object(
            manage_leaderboard,
            "available_file_binding",
            return_value={"path": "fixtures/example.json", "sha256": "c" * 64},
        ):
            index, records = manage_leaderboard.expected_claim_artifacts(manifest, [row])
        path = "leaderboard/claims/example/full/model-a.json"
        record = records[path]
        self.assertEqual(index["records"][0]["sha256"], hashlib.sha256(manage_leaderboard.json_bytes(record)).hexdigest())
        self.assertEqual(index["eligible_record_count"], 0)
        self.assertIsNone(record["result_permalink"])
        self.assertIsNone(record["claim_record_url"])
        self.assertEqual(
            manage_leaderboard.schema_errors(record, "result-claim.schema.json", schema_version="releases"),
            [],
        )
        self.assertEqual(
            manage_leaderboard.schema_errors(index, "claim-index.schema.json", schema_version="releases"),
            [],
        )

        official_record = deepcopy(record)
        official_record["release"]["status"] = "official"
        official_record["release"]["archive_url"] = "https://doi.org/10.0000/example"
        official_record["release"]["release_view_url"] = "https://fluidsbench.org/releases/release-1/"
        official_record["eligibility"] = {
            "academic_citation": True,
            "promotion": True,
            "reason_code": "approved_submitted_data_result",
            "reason": "Approved submitted-data result; model execution and metric recomputation were not performed.",
        }
        official_record["result_permalink"] = (
            "https://fluidsbench.org/releases/release-1/?view=result&dataset=example&split=full&result=model-a"
        )
        official_record["claim_record_url"] = (
            "https://assets.fluidsbench.org/releases/release-1/leaderboard/claims/example/full/model-a.json"
        )
        official_record["bindings"]["maintainer_validation"] = {
            "path": "submissions/example/model-a/maintainer-validation.json",
            "sha256": "b" * 64,
            "status": "validated",
            "validation_scope": "submitted_data_only",
            "model_execution": "not_performed",
            "metric_recomputation": "not_performed",
        }
        self.assertEqual(
            manage_leaderboard.schema_errors(
                official_record, "result-claim.schema.json", schema_version="releases"
            ),
            [],
        )

    def test_v3_claim_requires_all_spatial_bindings_and_records_prediction_status(self) -> None:
        row = {
            "schema_version": "3.0",
            "submission_id": "model-v3",
            "model": "Model v3",
            "dataset": "Example",
            "dataset_id": "example",
            "split": "Full",
            "split_id": "full",
            "evaluation": {"evidence_file": "evaluation-evidence.json"},
            "profile_data": {"index_file": "submissions/example/model-v3/profiles/index.json"},
            "scoring_support": {
                "release_id": "example-support-v1",
                "manifest_url": "https://example.org/releases/example-support-v1/manifest.json",
                "manifest_sha256": "a" * 64,
            },
            "spatial_discretization": {
                "file": "submissions/example/model-v3/discretization.json",
                "summary": {
                    "case_manifest": {
                        "file": "submissions/example/model-v3/discretization/cases.jsonl"
                    }
                },
            },
            "case_metrics": {"file": "submissions/example/model-v3/metrics/cases.json"},
            "prediction_artifact_status": {
                "sharing": "declared",
                "declared_artifact_count": 1,
                "maintainer_check_status": "not_recorded",
                "checked_artifact_count": 0,
            },
            "ranking": {
                "metric_id": "score",
                "value": 9.9,
                "ranked_value": 9.9,
                "display_value": "9.9",
                "unit": "points",
                "direction": "higher",
                "decimal_places": 1,
                "rounding": "decimal_half_up",
                "method": "competition",
                "rank": 1,
                "ranked_result_count": 1,
                "tied": False,
                "tie_count": 1,
            },
            "claim_eligibility": manage_leaderboard.claim_eligibility(
                "prototype_dummy_data", {}
            ),
        }
        manifest = {
            "all_file": "leaderboard/all.json",
            "ranking_contract": manage_leaderboard.ranking_contract(),
            "data_release": {
                "id": "prototype-v3",
                "status": "prototype_dummy_data",
                "generated_at": "2026-07-27T00:00:00Z",
                "feed_sha256": "b" * 64,
                "archive_url": None,
                "release_view_url": None,
                "asset_base_url": "https://example.org/releases/prototype-v3/",
            },
        }
        with (
            patch.object(
                manage_leaderboard,
                "available_file_binding",
                return_value={"path": "fixtures/file.json", "sha256": "c" * 64},
            ),
            patch.object(
                manage_leaderboard,
                "scoring_support_file_binding",
                return_value={"path": "fixtures/support.json", "sha256": "a" * 64},
            ),
        ):
            record = manage_leaderboard.build_claim_record(manifest, row, 0)
        self.assertEqual(record["result"]["submission_schema_version"], "3.0")
        self.assertEqual(record["prediction_artifacts"]["sharing"], "declared")
        self.assertEqual(
            set(record["bindings"]),
            {
                "result",
                "source_submission",
                "evaluation_evidence",
                "profile_index",
                "scoring_support",
                "spatial_discretization",
                "discretization_cases",
                "case_metrics",
            },
        )
        self.assertEqual(
            manage_leaderboard.schema_errors(
                record, "result-claim.schema.json", schema_version="releases"
            ),
            [],
        )
        record["bindings"].pop("case_metrics")
        self.assertTrue(
            any(
                "'case_metrics' is a required property" in error
                for error in manage_leaderboard.schema_errors(
                    record, "result-claim.schema.json", schema_version="releases"
                )
            )
        )

    def test_prediction_status_does_not_change_claim_eligibility(self) -> None:
        approved_row = {
            "approval": {"status": "approved"},
            "maintainer_validation": {"status": "validated"},
        }
        without_predictions = deepcopy(approved_row)
        without_predictions["prediction_artifact_status"] = {
            "sharing": "not_declared",
            "maintainer_check_status": "not_recorded",
        }
        with_failed_optional_check = deepcopy(approved_row)
        with_failed_optional_check["prediction_artifact_status"] = {
            "sharing": "declared",
            "maintainer_check_status": "recorded",
            "checks": [{"status": "failed"}],
        }
        self.assertEqual(
            manage_leaderboard.claim_eligibility("official", without_predictions),
            manage_leaderboard.claim_eligibility("official", with_failed_optional_check),
        )

    def test_approval_documents_bind_v3_evidence_without_prediction_gating(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            profile_path = directory / "profiles" / "index.json"
            manage_leaderboard.write_json(profile_path, {"schema_version": "1.0"})
            submission = {
                "submission_id": "model-v3",
                "dataset_id": "example",
                "split_id": "full",
                "case_set_id": "standard",
                "evaluation": {
                    "reference_version": "reference-v1",
                    "evidence_sha256": "a" * 64,
                },
                "reproducibility": {"contract_version": "open-reproducibility-3.0"},
                "profile_data": {
                    "index_file": "profiles/index.json",
                    "profile_ground_truth_release_id": "profiles-v1",
                    "profile_ground_truth_manifest_sha256": "b" * 64,
                },
                "scoring_support": {
                    "release_id": "support-v1",
                    "manifest_sha256": "c" * 64,
                },
                "spatial_discretization": {"sha256": "d" * 64},
                "case_metrics": {"sha256": "e" * 64},
                "prediction_artifacts": [{"artifact_id": "optional"}],
            }
            validation, approved = manage_leaderboard.approval_documents(
                submission,
                directory,
                validated_by="Validator",
                validated_at="2026-07-27T12:00:00Z",
                approved_by="Approver",
                approved_at="2026-07-27",
                pull_request_url="https://github.com/example/repo/pull/123",
            )
            self.assertEqual(
                validation["$schema"],
                "https://fluidsbench.org/schemas/v3/maintainer-validation.schema.json",
            )
            self.assertEqual(validation["schema_version"], "3.0")
            self.assertEqual(validation["discretization_sha256"], "d" * 64)
            self.assertEqual(validation["case_metrics_sha256"], "e" * 64)
            self.assertNotIn("prediction_artifacts", validation)
            self.assertEqual(
                manage_leaderboard.schema_errors(
                    validation,
                    "maintainer-validation.schema.json",
                    schema_version="v3",
                ),
                [],
            )
            self.assertEqual(
                approved["approval"]["validation"]["evidence_sha256"],
                hashlib.sha256(manage_leaderboard.json_bytes(validation)).hexdigest(),
            )

    def test_approve_submission_writes_validation_and_rebuilds(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "submissions" / "example" / "model-v1"
            profile_path = directory / "profiles" / "index.json"
            manifest_path = root / "leaderboard" / "manifest.json"
            manage_leaderboard.write_json(profile_path, {"schema_version": "1.0"})
            manage_leaderboard.write_json(manifest_path, {"data_release": {}})
            submission_path = directory / "submission.json"
            manage_leaderboard.write_json(
                submission_path,
                {
                    "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
                    "schema_version": "3.0",
                    "submission_id": "model-v1",
                    "result_revision": {
                        "series_id": "model",
                        "version": 1,
                        "supersedes": None,
                        "change_summary": "Initial result.",
                    },
                    "submitted_at": "2026-07-26",
                    "dataset_id": "example",
                    "split_id": "full",
                    "case_set_id": "standard",
                    "evaluation": {
                        "reference_version": "reference-v1",
                        "evidence_sha256": "a" * 64,
                    },
                    "reproducibility": {"contract_version": "open-reproducibility-3.0"},
                    "profile_data": {
                        "index_file": "profiles/index.json",
                        "profile_ground_truth_release_id": "profiles-v1",
                        "profile_ground_truth_manifest_sha256": "b" * 64,
                    },
                    "scoring_support": {
                        "release_id": "support-v1",
                        "manifest_sha256": "c" * 64,
                    },
                    "spatial_discretization": {"sha256": "d" * 64},
                    "case_metrics": {"sha256": "e" * 64},
                },
            )
            with (
                patch.object(manage_leaderboard, "ROOT", root),
                patch.object(manage_leaderboard, "MANIFEST_PATH", manifest_path),
                patch.object(
                    manage_leaderboard,
                    "validate_many",
                    side_effect=[([], {}), ([], {})],
                ),
                patch.object(manage_leaderboard, "release_contract_errors", return_value=[]),
                patch.object(manage_leaderboard, "build", return_value=[]) as build_mock,
            ):
                errors = manage_leaderboard.approve_submission(
                    directory,
                    validated_by="Validator",
                    validated_at="2026-07-27T12:00:00Z",
                    approved_by="Approver",
                    approved_at="2026-07-27",
                    pull_request_url="https://github.com/example/repo/pull/123",
                    dry_run=False,
                    update_existing=False,
                    rebuild_feeds=True,
                )
            self.assertEqual(errors, [])
            validation = manage_leaderboard.load_json(directory / "maintainer-validation.json")
            approved = manage_leaderboard.load_json(submission_path)
            self.assertEqual(validation["status"], "validated")
            self.assertEqual(approved["approval"]["status"], "approved")
            self.assertEqual(
                approved["approval"]["validation"]["evidence_sha256"],
                manage_leaderboard.sha256_file(directory / "maintainer-validation.json"),
            )
            build_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
