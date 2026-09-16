from __future__ import annotations

import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts import manage_leaderboard
from scripts import validate_submission as validator

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = tuple(
    ROOT / configuration["binding"]["submission_path"]
    for configuration in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS
)
EXPECTED_PREVIEWS = (
    (
        "full",
        "hiliftaeroml-transolver-full360-candidate-v1",
        "caseset-ac791749e527",
        360,
        5_400,
    ),
    (
        "single_aoa_4",
        "hiliftaeroml-transolver-aoa4-candidate-v1",
        "caseset-7a743a20b3bd",
        36,
        540,
    ),
    (
        "single_aoa_12",
        "hiliftaeroml-transolver-aoa12-candidate-v1",
        "caseset-02fc12ff3494",
        36,
        540,
    ),
    (
        "single_aoa_22",
        "hiliftaeroml-transolver-aoa22-candidate-v1",
        "caseset-85ecccd9ccda",
        36,
        540,
    ),
    (
        "super_scarce",
        "hiliftaeroml-transolver-super-scarce-candidate-v1",
        "caseset-ac791749e527",
        360,
        5_400,
    ),
    (
        "geometry_scarce",
        "hiliftaeroml-transolver-geometry-scarce-candidate-v1",
        "caseset-53990ea68fa6",
        360,
        5_400,
    ),
    (
        "geometry_super_scarce",
        "hiliftaeroml-transolver-geometry-super-scarce-candidate-v1",
        "caseset-53990ea68fa6",
        360,
        5_400,
    ),
    (
        "geometry",
        "hiliftaeroml-transolver-geometry-candidate-v1",
        "caseset-53990ea68fa6",
        360,
        5_400,
    ),
    (
        "aoa",
        "hiliftaeroml-transolver-ood-aoa-candidate-v1",
        "caseset-29693354ed8a",
        900,
        13_500,
    ),
    (
        "deflection",
        "hiliftaeroml-transolver-ood-deflection-candidate-v1",
        "caseset-c0ecb14de138",
        360,
        5_400,
    ),
    (
        "stall",
        "hiliftaeroml-transolver-ood-stall-candidate-v1",
        "caseset-804491c8956e",
        723,
        10_845,
    ),
    (
        "full",
        "hiliftaeroml-geotransolver-full360-candidate-v1",
        "caseset-ac791749e527",
        360,
        5_400,
    ),
    (
        "scarce",
        "hiliftaeroml-geotransolver-scarce-candidate-v1",
        "caseset-ac791749e527",
        360,
        5_400,
    ),
    (
        "single_aoa_4",
        "hiliftaeroml-geotransolver-aoa4-candidate-v1",
        "caseset-7a743a20b3bd",
        36,
        540,
    ),
    (
        "single_aoa_12",
        "hiliftaeroml-geotransolver-aoa12-candidate-v1",
        "caseset-02fc12ff3494",
        36,
        540,
    ),
    (
        "single_aoa_22",
        "hiliftaeroml-geotransolver-aoa22-candidate-v1",
        "caseset-85ecccd9ccda",
        36,
        540,
    ),
    (
        "geometry",
        "hiliftaeroml-geotransolver-geometry-candidate-v1",
        "caseset-53990ea68fa6",
        360,
        5_400,
    ),
    (
        "geometry_scarce",
        "hiliftaeroml-geotransolver-geometry-scarce-candidate-v1",
        "caseset-53990ea68fa6",
        360,
        5_400,
    ),
    (
        "geometry_super_scarce",
        "hiliftaeroml-geotransolver-geometry-super-scarce-candidate-v1",
        "caseset-53990ea68fa6",
        360,
        5_400,
    ),
    (
        "super_scarce",
        "hiliftaeroml-geotransolver-super-scarce-candidate-v1",
        "caseset-ac791749e527",
        360,
        5_400,
    ),
    (
        "aoa",
        "hiliftaeroml-geotransolver-ood-aoa-candidate-v1",
        "caseset-29693354ed8a",
        900,
        13_500,
    ),
    (
        "deflection",
        "hiliftaeroml-geotransolver-ood-deflection-candidate-v1",
        "caseset-c0ecb14de138",
        360,
        5_400,
    ),
    (
        "stall",
        "hiliftaeroml-geotransolver-ood-stall-candidate-v1",
        "caseset-804491c8956e",
        723,
        10_845,
    ),
)


def manifest() -> dict:
    return validator.manifest_with_benchmark_contract(
        validator.load_json(validator.MANIFEST_PATH)
    )


class HiLiftRegisteredPreviewTests(unittest.TestCase):
    def test_registered_inventory_is_exact_and_unique(self) -> None:
        actual = tuple(
            (
                configuration["split_id"],
                configuration["binding"]["submission_id"],
                configuration["case_set_id"],
                configuration["case_count"],
                configuration["profile_series_count"],
            )
            for configuration in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS
        )
        self.assertEqual(actual, EXPECTED_PREVIEWS)
        self.assertEqual(
            len({submission_id for _split, submission_id, *_rest in actual}),
            len(EXPECTED_PREVIEWS),
        )

    def test_all_pass_without_hidden_score_recomputation(self) -> None:
        current_manifest = manifest()
        with patch.object(
            validator,
            "score_hilift_compact_profile_directory",
            side_effect=AssertionError(
                "ordinary preview validation loaded hidden truth"
            ),
        ):
            for configuration, submission_path in zip(
                validator.HILIFT_REGISTERED_PREVIEW_CONFIGS,
                SUBMISSIONS,
                strict=True,
            ):
                with self.subTest(split_id=configuration["split_id"]):
                    submission = validator.load_json(submission_path)
                    binding = validator.registered_hiliftaeroml_preview(
                        submission_path,
                        submission,
                        current_manifest,
                    )
                    self.assertIsNotNone(binding)
                    errors, stats = validator.validate_submission_file(
                        submission_path,
                        current_manifest,
                    )
                    self.assertEqual(errors, [])
                    self.assertEqual(
                        stats,
                        {
                            "cases": configuration["case_count"],
                            "series": configuration["profile_series_count"],
                        },
                    )

    def test_registration_is_exact_and_lifecycle_bound(self) -> None:
        current_manifest = manifest()
        official_manifest = deepcopy(current_manifest)
        official_manifest["data_release"]["status"] = "official"
        original_load = validator.load_json
        for configuration, submission_path in zip(
            validator.HILIFT_REGISTERED_PREVIEW_CONFIGS,
            SUBMISSIONS,
            strict=True,
        ):
            with self.subTest(split_id=configuration["split_id"]):
                submission = original_load(submission_path)
                changed_submission = deepcopy(submission)
                changed_submission["approval"] = {"status": "prototype"}
                self.assertIsNone(
                    validator.registered_hiliftaeroml_preview(
                        submission_path,
                        changed_submission,
                        current_manifest,
                    )
                )
                self.assertIsNone(
                    validator.registered_hiliftaeroml_preview(
                        ROOT / "outside" / "submission.json",
                        submission,
                        current_manifest,
                    )
                )
                self.assertIsNone(
                    validator.registered_hiliftaeroml_preview(
                        submission_path,
                        submission,
                        official_manifest,
                    )
                )

                receipt_path = (
                    ROOT / configuration["validation_record_path"]
                )
                changed_receipt = deepcopy(original_load(receipt_path))
                changed_receipt["activation"]["owner_approval_complete"] = True

                def load_with_changed_receipt(path: Path) -> object:
                    if Path(path).resolve() == receipt_path.resolve():
                        return changed_receipt
                    return original_load(Path(path))

                with patch.object(
                    validator,
                    "load_json",
                    side_effect=load_with_changed_receipt,
                ):
                    self.assertIsNone(
                        validator.registered_hiliftaeroml_preview(
                            submission_path,
                            submission,
                            current_manifest,
                        )
                    )

    def test_feed_replaces_prototypes_with_ineligible_references(self) -> None:
        rows = manage_leaderboard.source_rows_by_dataset(manifest())["HiLiftAeroML"]
        self.assertEqual(
            len(rows), len(validator.HILIFT_REGISTERED_PREVIEW_CONFIGS)
        )
        rows_by_id = {row["submission_id"]: row for row in rows}
        self.assertEqual(
            set(rows_by_id),
            {
                configuration["binding"]["submission_id"]
                for configuration in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS
            },
        )
        for configuration in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS:
            row = rows_by_id[configuration["binding"]["submission_id"]]
            self.assertEqual(row["split_id"], configuration["split_id"])
            self.assertEqual(row["record_type"], "pre_release_reference")
            self.assertIsNone(row.get("approval"))

            eligibility = manage_leaderboard.claim_eligibility(
                "prototype_dummy_data",
                row,
            )
            self.assertFalse(eligibility["academic_citation"])
            self.assertFalse(eligibility["promotion"])
            self.assertEqual(
                eligibility["reason_code"], "pre_release_reference"
            )
            self.assertIn("HiLiftAeroML", eligibility["reason"])

        official_manifest = manifest()
        official_manifest["data_release"]["status"] = "official"
        self.assertEqual(
            manage_leaderboard.source_rows_by_dataset(official_manifest)[
                "HiLiftAeroML"
            ],
            [],
        )

    def test_generated_feed_has_every_registered_rank_per_split(self) -> None:
        rows = validator.load_json(
            ROOT / "leaderboard" / "datasets" / "hiliftaeroml.json"
        )
        self.assertEqual(
            len(rows), len(validator.HILIFT_REGISTERED_PREVIEW_CONFIGS)
        )
        rows_by_id = {row["submission_id"]: row for row in rows}
        split_counts = Counter(
            configuration["split_id"]
            for configuration in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS
        )
        for configuration in validator.HILIFT_REGISTERED_PREVIEW_CONFIGS:
            row = rows_by_id[configuration["binding"]["submission_id"]]
            self.assertEqual(row["split_id"], configuration["split_id"])
            self.assertEqual(row["record_type"], "pre_release_reference")
            self.assertGreaterEqual(row["ranking"]["rank"], 1)
            self.assertLessEqual(
                row["ranking"]["rank"], split_counts[configuration["split_id"]]
            )
            self.assertEqual(
                row["ranking"]["ranked_result_count"],
                split_counts[configuration["split_id"]],
            )
            self.assertFalse(row["claim_eligibility"]["academic_citation"])
            self.assertFalse(row["claim_eligibility"]["promotion"])

    def test_cannot_be_approved_while_support_is_closed(self) -> None:
        errors = manage_leaderboard.approve_submission(
            SUBMISSIONS[0],
            validated_by="Maintainer",
            validated_at="2026-09-03T12:00:00Z",
            approved_by="Dataset owner",
            approved_at="2026-09-03",
            pull_request_url=(
                "https://github.com/neilashton/fluidsbench-submission/pull/21"
            ),
            dry_run=True,
            update_existing=False,
            rebuild_feeds=False,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("approval is closed", errors[0])


if __name__ == "__main__":
    unittest.main()
