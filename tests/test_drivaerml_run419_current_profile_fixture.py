from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from reference.drivaerml.coordinate_identity import (
    coordinate_array_identity_sha256,
)
from reference.drivaerml.dataset_scorer import (
    DrivAerDatasetScorerError,
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_CONTRACT_SHA256,
    RELATIVE_PROFILE_FORMAT,
    RELATIVE_PROFILE_SCHEMA_VERSION,
    validate_schema_v3_relative_profile_chunk_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "evidence"
    / "transolver-run419-current-relative-profile-v3"
)
PROFILE_INDEX_PATH = FIXTURE_ROOT / "profiles" / "index.json"
PROFILE_CHUNK_PATH = FIXTURE_ROOT / "profiles" / "chunk-000.json"
PROVENANCE_PATH = FIXTURE_ROOT / "provenance.json"
CURRENT_EVALUATION_PATH = FIXTURE_ROOT / "current-case-evaluation.json"
IMPLEMENTATION_RECEIPT_PATH = (
    FIXTURE_ROOT / "current-case-implementation-receipt.json"
)
RELATIVE_SUPPORT_INDEX_PATH = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "series-support-index.json"
)
CONSTANT_SUPPORT_INDEX_PATH = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "run419-constant-series-support-index.json"
)

EXPECTED_FILE_SHA256 = {
    CURRENT_EVALUATION_PATH: (
        "6aa349290049ad5498dd3a578865fc14aee5521be1ebc43208b43082a7a24bdf"
    ),
    IMPLEMENTATION_RECEIPT_PATH: (
        "26b9b67be887fd991c63bbab04c68192768edceac65813815b2d1073a4568be1"
    ),
    PROFILE_INDEX_PATH: (
        "654d5afec080e0f89ba3fa5a50135e72af23eef4af0f9b1ba1bf22a3f757c548"
    ),
    PROFILE_CHUNK_PATH: (
        "3a5bca4c5f8730dcae4e29f69f415dc393050c6ebca6cf6e1303d307985bf262"
    ),
    PROVENANCE_PATH: (
        "62ef43e1f5b8fe30c8b03fc36d315c4ad1b6faa242b0ec56c1ee0cb41cd5576a"
    ),
}

VELOCITY_STATIONS = (
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "L1",
    "R1",
    "R2",
    "R3",
)
CONSTANT_CP_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
    "sidewall_z_0_15",
    "front_left_wheelhouse_y_neg_0_6",
)
RELATIVE_CP_REPRESENTATIONS = {
    "upperbody_centerline": "shared_alias",
    "underbody_centerline": "shared_alias",
    "sidewall_front_wheelhouse_relative": "materialized",
    "front_left_wheelhouse_relative": "materialized",
}
EXPECTED_SERIES_KEYS = {
    *{
        (
            "drivaerml-autocfd5-constant-v1",
            f"autocfd5_{station.lower()}",
            "materialized",
        )
        for station in VELOCITY_STATIONS
    },
    *{
        ("drivaerml-velocity-relative-v3", station, "materialized")
        for station in VELOCITY_STATIONS
    },
    *{
        ("drivaerml_cp_constant_v1", station, "materialized")
        for station in CONSTANT_CP_STATIONS
    },
    *{
        ("drivaerml_cp_relative_v1", station, representation)
        for station, representation in RELATIVE_CP_REPRESENTATIONS.items()
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrong_identity(label: str) -> str:
    return hashlib.sha256(f"deliberately-wrong:{label}".encode("utf-8")).hexdigest()


class DrivAerMLRun419CurrentProfileFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = load_json(PROFILE_INDEX_PATH)
        cls.chunk = load_json(PROFILE_CHUNK_PATH)
        cls.provenance = load_json(PROVENANCE_PATH)
        cls.series = cls.chunk["cases"][0]["series"]

        relative_document = load_json(RELATIVE_SUPPORT_INDEX_PATH)
        cls.relative_support = {
            (series["family_id"], series["station_id"]): series
            for case in relative_document["cases"]
            if case["case_id"] == "run_419"
            for series in case["series"]
        }
        constant_document = load_json(CONSTANT_SUPPORT_INDEX_PATH)
        cls.constant_support = {
            (series["family_id"], series["station_id"]): series
            for series in constant_document["series"]
        }

    def test_fixture_bytes_and_internal_receipts_are_hash_bound(self) -> None:
        for path, expected_sha256 in EXPECTED_FILE_SHA256.items():
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())
                self.assertEqual(sha256_file(path), expected_sha256)

        self.assertEqual(
            self.index["chunks"],
            [
                {
                    "case_ids": ["run_419"],
                    "file": "chunk-000.json",
                    "sha256": EXPECTED_FILE_SHA256[PROFILE_CHUNK_PATH],
                }
            ],
        )
        fixture_receipt = self.provenance["fixture"]
        self.assertEqual(
            fixture_receipt["index_sha256"],
            EXPECTED_FILE_SHA256[PROFILE_INDEX_PATH],
        )
        self.assertEqual(fixture_receipt["series_count"], 40)
        self.assertEqual(fixture_receipt["case_count"], 1)
        self.assertEqual(fixture_receipt["chunk_count"], 1)
        self.assertEqual(
            fixture_receipt["chunks"],
            [
                {
                    "file": "chunk-000.json",
                    "sha256": EXPECTED_FILE_SHA256[PROFILE_CHUNK_PATH],
                    "size_bytes": PROFILE_CHUNK_PATH.stat().st_size,
                }
            ],
        )

        current = self.provenance["evaluator"]["current_case_evaluation"]
        self.assertEqual(current["file"], CURRENT_EVALUATION_PATH.name)
        self.assertEqual(
            (current["sha256"], current["size_bytes"]),
            (
                EXPECTED_FILE_SHA256[CURRENT_EVALUATION_PATH],
                CURRENT_EVALUATION_PATH.stat().st_size,
            ),
        )
        receipt = current["implementation_receipt"]
        self.assertEqual(receipt["file"], IMPLEMENTATION_RECEIPT_PATH.name)
        self.assertEqual(
            (receipt["sha256"], receipt["size_bytes"]),
            (
                EXPECTED_FILE_SHA256[IMPLEMENTATION_RECEIPT_PATH],
                IMPLEMENTATION_RECEIPT_PATH.stat().st_size,
            ),
        )

    def test_real_fixture_passes_json_schemas_and_semantic_validation(self) -> None:
        index_schema = load_json(ROOT / "schemas" / "v1" / "profile-index.schema.json")
        chunk_schema = load_json(
            ROOT
            / "schemas"
            / "v1"
            / "drivaerml-relative-profile-chunk.schema.json"
        )
        self.assertEqual(
            list(Draft202012Validator(index_schema).iter_errors(self.index)), []
        )
        self.assertEqual(
            list(Draft202012Validator(chunk_schema).iter_errors(self.chunk)), []
        )
        normalized = validate_schema_v3_relative_profile_chunk_candidate(self.chunk)
        self.assertEqual(normalized, self.chunk)
        self.assertEqual(self.index["format"], RELATIVE_PROFILE_FORMAT)
        self.assertEqual(self.index["contract_id"], RELATIVE_PROFILE_CONTRACT_ID)
        self.assertEqual(
            self.index["contract_sha256"], RELATIVE_PROFILE_CONTRACT_SHA256
        )
        self.assertEqual(self.chunk["schema_version"], RELATIVE_PROFILE_SCHEMA_VERSION)
        self.assertEqual(self.chunk["cases"][0]["case_id"], "run_419")

    def test_exact_40_series_namespace_and_array_lengths(self) -> None:
        observed = [
            (
                series["family_id"],
                series["station_id"],
                series["representation"],
            )
            for series in self.series
        ]
        self.assertEqual(len(EXPECTED_SERIES_KEYS), 40)
        self.assertEqual(len(observed), 40)
        self.assertEqual(len(set(observed)), 40)
        self.assertEqual(set(observed), EXPECTED_SERIES_KEYS)
        self.assertEqual(
            Counter(
                (series["family_id"], series["representation"])
                for series in self.series
            ),
            {
                ("drivaerml-autocfd5-constant-v1", "materialized"): 16,
                ("drivaerml-velocity-relative-v3", "materialized"): 16,
                ("drivaerml_cp_constant_v1", "materialized"): 4,
                ("drivaerml_cp_relative_v1", "shared_alias"): 2,
                ("drivaerml_cp_relative_v1", "materialized"): 2,
            },
        )
        for series in self.series:
            label = f"{series['family_id']}/{series['station_id']}"
            with self.subTest(series=label):
                if series["representation"] == "materialized":
                    self.assertEqual(
                        len(series["coordinate"]), len(series["prediction"])
                    )
                    self.assertGreaterEqual(len(series["coordinate"]), 2)
                    self.assertTrue(all(math.isfinite(v) for v in series["coordinate"]))
                    self.assertTrue(all(math.isfinite(v) for v in series["prediction"]))
                else:
                    self.assertNotIn("coordinate", series)
                    self.assertNotIn("prediction", series)
                    self.assertNotIn("support_identity_sha256", series)

    def test_every_series_identity_matches_retained_run419_support(self) -> None:
        self.assertEqual(len(self.relative_support), 20)
        self.assertEqual(len(self.constant_support), 20)
        for series in self.series:
            key = (series["family_id"], series["station_id"])
            relative = series["family_id"] in {
                "drivaerml-velocity-relative-v3",
                "drivaerml_cp_relative_v1",
            }
            expected = (
                self.relative_support[key]
                if relative
                else self.constant_support[key]
            )
            label = f"{key[0]}/{key[1]}"
            with self.subTest(series=label):
                self.assertEqual(
                    series["placement_receipt_identity_sha256"],
                    expected["placement_receipt_identity_sha256"],
                )
                self.assertEqual(
                    series["representation"], expected["representation"]
                )
                if series["representation"] == "shared_alias":
                    self.assertEqual(
                        series["shared_support_ref"][
                            "canonical_support_identity_sha256"
                        ],
                        expected["support_identity_sha256"],
                    )
                else:
                    self.assertEqual(
                        series["support_identity_sha256"],
                        expected["support_identity_sha256"],
                    )
                    self.assertEqual(
                        len(series["coordinate"]), expected["coordinate_count"]
                    )
                    self.assertEqual(
                        coordinate_array_identity_sha256(series["coordinate"]),
                        expected["coordinate_identity_sha256"],
                    )

    def test_relative_cp_coordinate_mutation_to_1234_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.chunk)
        series = next(
            item
            for item in mutated["cases"][0]["series"]
            if item["family_id"] == "drivaerml_cp_relative_v1"
            and item["representation"] == "materialized"
        )
        retained_identities = (
            series["support_identity_sha256"],
            series["placement_receipt_identity_sha256"],
        )
        self.assertLess(series["coordinate"][-2], 1234.0)
        series["coordinate"][-1] = 1234.0
        self.assertEqual(
            (
                series["support_identity_sha256"],
                series["placement_receipt_identity_sha256"],
            ),
            retained_identities,
        )
        with self.assertRaisesRegex(
            DrivAerDatasetScorerError,
            "coordinate identity differs from the retained relative support index",
        ):
            validate_schema_v3_relative_profile_chunk_candidate(mutated)

    def test_relative_velocity_coordinate_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.chunk)
        series = next(
            item
            for item in mutated["cases"][0]["series"]
            if item["family_id"] == "drivaerml-velocity-relative-v3"
            and item["station_id"] == "V1"
        )
        index = len(series["coordinate"]) // 2
        original = series["coordinate"][index]
        series["coordinate"][index] = (
            original + series["coordinate"][index + 1]
        ) / 2.0
        self.assertNotEqual(series["coordinate"][index], original)
        with self.assertRaisesRegex(
            DrivAerDatasetScorerError,
            "coordinate identity differs from the retained relative support index",
        ):
            validate_schema_v3_relative_profile_chunk_candidate(mutated)

    def test_wrong_relative_support_and_receipt_identities_are_rejected(self) -> None:
        for field, message in (
            ("support_identity_sha256", "support identity differs"),
            (
                "placement_receipt_identity_sha256",
                "placement receipt identity differs",
            ),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.chunk)
                series = next(
                    item
                    for item in mutated["cases"][0]["series"]
                    if item["family_id"] == "drivaerml-velocity-relative-v3"
                    and item["station_id"] == "V1"
                )
                series[field] = wrong_identity(field)
                with self.assertRaisesRegex(DrivAerDatasetScorerError, message):
                    validate_schema_v3_relative_profile_chunk_candidate(mutated)

    def test_fixture_and_repository_remain_explicitly_inactive(self) -> None:
        self.assertEqual(
            self.provenance["status"],
            "complete_one_case_real_model_regression_fixture_not_submission",
        )
        self.assertEqual(
            self.provenance["claims"],
            {
                "model_quality_claim": False,
                "official_submission": False,
                "owner_approval_complete": False,
                "relative_scoring_activated": False,
                "scoring_weight_changed": False,
                "sparse_constant_velocity_is_activation_or_scoring_support": False,
                "submission_validation_opened": False,
                "three_model_sensitivity_gate_complete": False,
            },
        )
        self.assertFalse(self.provenance["fixture"]["official_submission"])
        self.assertTrue(self.provenance["fixture"]["candidate_only"])

        submission_spec = load_json(
            ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json"
        )
        relative = submission_spec["relative_diagnostics"]
        self.assertEqual(relative["status"], "support_verified_activation_pending")
        self.assertFalse(relative["profile_format_enabled"])
        self.assertEqual(relative["relative_composite_weight"], 0.0)
        self.assertFalse(submission_spec["scoring_support"]["submissions_open"])


if __name__ == "__main__":
    unittest.main()
