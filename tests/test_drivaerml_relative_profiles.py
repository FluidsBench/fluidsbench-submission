from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from reference.drivaerml.coordinate_identity import (
    CoordinateIdentityError,
    coordinate_array_identity_sha256,
)
from reference.drivaerml.dataset_scorer import (
    CandidateDatasetEvaluation,
    DrivAerDatasetScorerError,
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_CONTRACT_SHA256,
    RELATIVE_PROFILE_FORMAT,
    RELATIVE_PROFILE_SCHEMA_VERSION,
    _relative_profile_expected_keys,
    schema_v3_relative_profile_chunks_candidate_adapter,
    validate_schema_v3_relative_profile_chunk_candidate,
    write_schema_v3_profile_chunks_candidate,
)
from scripts.validate_submission import sha256_file, validate_profiles


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_INDEX_PATH = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "series-support-index.json"
)
SUPPORT_INDEX = json.loads(SUPPORT_INDEX_PATH.read_text(encoding="utf-8"))
SUPPORT_IDENTITIES = {
    (case["case_id"], series["family_id"], series["station_id"]): series
    for case in SUPPORT_INDEX["cases"]
    for series in case["series"]
}
CONSTANT_SUPPORT_INDEX_PATH = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "run419-constant-series-support-index.json"
)
CONSTANT_SUPPORT_INDEX = json.loads(
    CONSTANT_SUPPORT_INDEX_PATH.read_text(encoding="utf-8")
)
CONSTANT_SUPPORT_IDENTITIES = {
    (series["family_id"], series["station_id"]): series
    for series in CONSTANT_SUPPORT_INDEX["series"]
}
ALL_CONSTANT_SUPPORT_INDEX_PATH = (
    ROOT
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "constant-series-support-index.json"
)
ALL_CONSTANT_SUPPORT_INDEX = json.loads(
    ALL_CONSTANT_SUPPORT_INDEX_PATH.read_text(encoding="utf-8")
)
ALL_CONSTANT_SUPPORT_IDENTITIES = {
    (case["case_id"], series["family_id"], series["station_id"]): series
    for case in ALL_CONSTANT_SUPPORT_INDEX["cases"]
    for series in case["series"]
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sample_count(station_id: str) -> int:
    station = station_id.removeprefix("autocfd5_").upper()
    if station.startswith("V"):
        return 201
    if station.startswith("U"):
        return 301
    if station == "L1":
        return 651
    return 31


def constant_interval_end(station_id: str) -> float:
    station = station_id.removeprefix("autocfd5_").upper()
    if station.startswith("V"):
        return 2.0
    if station.startswith("U"):
        return 3.0
    if station == "L1":
        return 6.5
    return {
        "R1": 0.300000876107,
        "R2": 0.300000877067,
        "R3": 0.300000591286,
    }[station]


def namespaced_series(case_id: str = "run_1") -> list[dict]:
    result: list[dict] = []
    canonical_cp_digests: dict[str, str] = {}
    for panel_id, family_id, station_id, quantity_id, representation in (
        _relative_profile_expected_keys()
    ):
        relative = family_id in {
            "drivaerml-velocity-relative-v3",
            "drivaerml_cp_relative_v1",
        }
        retained_identity = (
            SUPPORT_IDENTITIES[(case_id, family_id, station_id)]
            if relative
            else None
        )
        common = {
            "panel_id": panel_id,
            "family_id": family_id,
            "placement_mode": "relative" if relative else "constant",
            "station_id": station_id,
            "quantity_id": quantity_id,
            "scoring_role": "report_only" if relative else "inherits_parent_candidate",
            "representation": representation,
            "placement_receipt_identity_sha256": (
                retained_identity["placement_receipt_identity_sha256"]
                if retained_identity is not None
                else digest(f"receipt:{family_id}:{station_id}")
            ),
        }
        if representation == "shared_alias":
            shared_id = {
                "upperbody_centerline": "drivaerml-cp-upperbody-centerline-y0-v1",
                "underbody_centerline": "drivaerml-cp-underbody-centerline-y0-v1",
            }[station_id]
            common["shared_support_ref"] = {
                "shared_support_id": shared_id,
                "canonical_family_id": "drivaerml_cp_constant_v1",
                "canonical_station_id": station_id,
                "canonical_support_identity_sha256": retained_identity[
                    "support_identity_sha256"
                ],
            }
        else:
            if retained_identity is not None:
                support_digest = retained_identity["support_identity_sha256"]
            elif (
                family_id == "drivaerml_cp_constant_v1"
                and station_id in {"upperbody_centerline", "underbody_centerline"}
            ):
                support_digest = SUPPORT_IDENTITIES[
                    (case_id, "drivaerml_cp_relative_v1", station_id)
                ]["support_identity_sha256"]
            else:
                support_digest = digest(f"support:{family_id}:{station_id}")
            common["support_identity_sha256"] = support_digest
            if family_id == "drivaerml_cp_constant_v1" and station_id in {
                "upperbody_centerline",
                "underbody_centerline",
            }:
                canonical_cp_digests[station_id] = support_digest
            if panel_id == "velocity_profiles":
                count = sample_count(station_id)
                if relative:
                    coordinates = [index / (count - 1) for index in range(count)]
                    common.update(
                        {
                            "coordinate_id": "normalized_arc_length",
                            "coordinate_unit": "1",
                        }
                    )
                else:
                    end = constant_interval_end(station_id)
                    coordinates = [end * index / (count - 1) for index in range(count)]
                    common.update(
                        {"coordinate_id": "distance_m", "coordinate_unit": "m"}
                    )
                common["coordinate"] = coordinates
                common["prediction"] = [0.5] * count
            else:
                common.update(
                    {
                        "coordinate_id": "arc_length_m",
                        "coordinate_unit": "m",
                        "coordinate": [0.0, 1.0],
                        "prediction": [0.1, 0.2],
                    }
                )
        result.append(common)
    return result


def chunk(case_id: str = "run_1") -> dict:
    return {
        "schema_version": RELATIVE_PROFILE_SCHEMA_VERSION,
        "contract_id": RELATIVE_PROFILE_CONTRACT_ID,
        "contract_sha256": RELATIVE_PROFILE_CONTRACT_SHA256,
        "cases": [{"case_id": case_id, "series": namespaced_series(case_id)}],
    }


def synthetic_retained_support(document: dict) -> dict:
    """Bind synthetic unit-test coordinates without weakening production pins."""

    retained = {key: dict(value) for key, value in SUPPORT_IDENTITIES.items()}
    for case in document["cases"]:
        for series in case["series"]:
            if (
                series["representation"] == "materialized"
                and series["family_id"]
                in {
                    "drivaerml-velocity-relative-v3",
                    "drivaerml_cp_relative_v1",
                }
            ):
                identity = retained[
                    (case["case_id"], series["family_id"], series["station_id"])
                ]
                identity["coordinate_count"] = len(series["coordinate"])
                identity["coordinate_identity_sha256"] = (
                    coordinate_array_identity_sha256(series["coordinate"])
                )
    return retained


def synthetic_constant_support(document: dict) -> dict:
    """Bind synthetic constant arrays while exercising production checks."""

    retained = {
        key: dict(value) for key, value in ALL_CONSTANT_SUPPORT_IDENTITIES.items()
    }
    for case in document["cases"]:
        for series in case["series"]:
            if (
                series["representation"] == "materialized"
                and series["family_id"]
                in {
                    "drivaerml-autocfd5-constant-v1",
                    "drivaerml_cp_constant_v1",
                }
            ):
                identity = retained[
                    (case["case_id"], series["family_id"], series["station_id"])
                ]
                identity["support_identity_sha256"] = series[
                    "support_identity_sha256"
                ]
                identity["placement_receipt_identity_sha256"] = series[
                    "placement_receipt_identity_sha256"
                ]
                identity["coordinate_count"] = len(series["coordinate"])
                identity["coordinate_identity_sha256"] = (
                    coordinate_array_identity_sha256(series["coordinate"])
                )
    return retained


@contextmanager
def synthetic_support_patch(document: dict):
    with patch(
        "reference.drivaerml.dataset_scorer._relative_series_support_index",
        return_value=synthetic_retained_support(document),
    ), patch(
        "reference.drivaerml.dataset_scorer._constant_series_support_index",
        return_value=synthetic_constant_support(document),
    ):
        yield


def validate_synthetic_chunk(document: dict) -> dict:
    with synthetic_support_patch(document):
        return validate_schema_v3_relative_profile_chunk_candidate(document)


class DrivAerMLRelativeProfileTests(unittest.TestCase):
    def test_coordinate_identity_encoding_is_canonical_and_fail_closed(self) -> None:
        self.assertEqual(
            coordinate_array_identity_sha256([0.0, 1.0]),
            "a6102c31dbc506d4e846fb590395fbbf677b50a7f554d0703b7f01077f4e4b0d",
        )
        self.assertEqual(
            coordinate_array_identity_sha256([-0.0, 1.0]),
            coordinate_array_identity_sha256([0.0, 1]),
        )
        self.assertNotEqual(
            coordinate_array_identity_sha256([0.0, 1.0]),
            coordinate_array_identity_sha256([0.0, 1.0, 0.0]),
        )
        for invalid in ([True, 1.0], [float("nan")], [float("inf")], ["0.0"]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(CoordinateIdentityError):
                    coordinate_array_identity_sha256(invalid)

    def test_retained_index_v2_binds_only_materialized_coordinates(self) -> None:
        self.assertEqual(
            SUPPORT_INDEX["schema"],
            "drivaerml-relative-series-support-index-v2",
        )
        self.assertEqual(SUPPORT_INDEX["schema_version"], 2)
        for case in SUPPORT_INDEX["cases"]:
            for series in case["series"]:
                if series["representation"] == "materialized":
                    self.assertGreaterEqual(series["coordinate_count"], 2)
                    self.assertRegex(
                        series["coordinate_identity_sha256"], r"^[0-9a-f]{64}$"
                    )
                else:
                    self.assertNotIn("coordinate_count", series)
                    self.assertNotIn("coordinate_identity_sha256", series)

    def test_retained_run419_constant_index_binds_all_materialized_series(self) -> None:
        self.assertEqual(
            CONSTANT_SUPPORT_INDEX["schema"],
            "drivaerml-run419-constant-series-support-index-v1",
        )
        self.assertEqual(CONSTANT_SUPPORT_INDEX["case_id"], "run_419")
        self.assertEqual(CONSTANT_SUPPORT_INDEX["series_count"], 20)
        self.assertEqual(len(CONSTANT_SUPPORT_IDENTITIES), 20)
        for series in CONSTANT_SUPPORT_INDEX["series"]:
            self.assertEqual(series["representation"], "materialized")
            self.assertGreaterEqual(series["coordinate_count"], 2)
            for field in (
                "support_identity_sha256",
                "placement_receipt_identity_sha256",
                "coordinate_identity_sha256",
            ):
                self.assertRegex(series[field], r"^[0-9a-f]{64}$")
                self.assertNotEqual(series[field], "0" * 64)

    def test_all_case_constant_index_binds_native_v3_support(self) -> None:
        self.assertEqual(
            ALL_CONSTANT_SUPPORT_INDEX["schema"],
            "drivaerml-constant-series-support-index-v1",
        )
        self.assertEqual(ALL_CONSTANT_SUPPORT_INDEX["case_count"], 484)
        self.assertEqual(ALL_CONSTANT_SUPPORT_INDEX["series_per_case"], 20)
        self.assertEqual(len(ALL_CONSTANT_SUPPORT_IDENTITIES), 484 * 20)
        for key, run419_identity in CONSTANT_SUPPORT_IDENTITIES.items():
            self.assertEqual(
                ALL_CONSTANT_SUPPORT_IDENTITIES[("run_419", *key)],
                run419_identity,
            )

    def test_contract_schema_and_closed_scoring_policy_are_bound(self) -> None:
        bindings = json.loads(
            (ROOT / "benchmark-specs" / "drivaerml" / "candidate-release-bindings.json").read_text()
        )
        contract_path = (
            ROOT
            / "benchmark-specs"
            / "drivaerml"
            / bindings["relative_diagnostics_v3"]["file"]
        )
        self.assertEqual(sha256_file(contract_path), RELATIVE_PROFILE_CONTRACT_SHA256)
        contract = json.loads(contract_path.read_text())
        self.assertFalse(contract["activation_by_this_file"])
        self.assertEqual(
            contract["scoring_and_rollout"]["constant_family_policy"], "unchanged"
        )
        self.assertEqual(
            contract["scoring_and_rollout"]["relative_composite_weight"], 0.0
        )
        self.assertEqual(
            contract["status"],
            "candidate_report_only_support_artifacts_verified_activation_pending",
        )
        self.assertEqual(
            contract["activation_gates"],
            {
                "all_484_velocity_placement_manifest_bound": True,
                "all_484_velocity_mapping_manifest_bound": True,
                "all_484_cp_manifest_bound": True,
                "genuine_model_sensitivity_review_complete": False,
                "owner_scientific_approval": False,
                "immutable_evaluator_revision_bound": True,
            },
        )
        self.assertEqual(bindings["relative_support"]["status"], "ready")
        velocity_contract = contract["support_implementation_bindings"][
            "relative_velocity_v3"
        ]["mapping_all484_manifest"]
        velocity_handoff = bindings["relative_support"][
            "velocity_mapping_manifest"
        ]
        self.assertEqual(
            velocity_contract["producer_path"], velocity_handoff["producer_file"]
        )
        self.assertEqual(
            velocity_handoff["manifest_sha256"],
            sha256_file(ROOT / "benchmark-specs" / "drivaerml" / velocity_handoff["manifest_file"]),
        )
        cp_contract = contract["support_implementation_bindings"][
            "relative_cp_v1"
        ]["cp_all484_manifest"]
        cp_handoff = bindings["relative_support"]["cp_manifest"]
        self.assertEqual(cp_contract["producer_path"], cp_handoff["producer_file"])
        self.assertEqual(
            cp_handoff["manifest_sha256"],
            sha256_file(ROOT / "benchmark-specs" / "drivaerml" / cp_handoff["manifest_file"]),
        )
        placement_contract = contract["support_implementation_bindings"][
            "relative_velocity_v3"
        ]["placement_all484_manifest"]
        placement_handoff = bindings["relative_support"][
            "velocity_placement_manifest"
        ]
        self.assertEqual(
            placement_contract["producer_path"], placement_handoff["producer_file"]
        )
        self.assertEqual(
            placement_handoff["manifest_sha256"],
            sha256_file(ROOT / "benchmark-specs" / "drivaerml" / placement_handoff["manifest_file"]),
        )

    def test_complete_namespaced_chunk_passes_schema_and_semantics(self) -> None:
        document = chunk()
        schema = json.loads(
            (ROOT / "schemas" / "v1" / "drivaerml-relative-profile-chunk.schema.json").read_text()
        )
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(document)), [])
        normalized = validate_synthetic_chunk(document)
        self.assertEqual(len(normalized["cases"][0]["series"]), 40)
        aliases = [
            series
            for series in normalized["cases"][0]["series"]
            if series["representation"] == "shared_alias"
        ]
        self.assertEqual(len(aliases), 2)

    def test_materialized_relative_coordinate_mutations_are_rejected(self) -> None:
        cp_document = chunk()
        retained = synthetic_retained_support(cp_document)
        relative_cp = next(
            series
            for series in cp_document["cases"][0]["series"]
            if series["family_id"] == "drivaerml_cp_relative_v1"
            and series["representation"] == "materialized"
        )
        relative_cp["coordinate"][0] = 1234.0
        with patch(
            "reference.drivaerml.dataset_scorer._relative_series_support_index",
            return_value=retained,
        ), patch(
            "reference.drivaerml.dataset_scorer._constant_series_support_index",
            return_value=synthetic_constant_support(cp_document),
        ):
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "coordinate identity differs"
            ):
                validate_schema_v3_relative_profile_chunk_candidate(cp_document)

        velocity_document = chunk()
        retained = synthetic_retained_support(velocity_document)
        relative_velocity = next(
            series
            for series in velocity_document["cases"][0]["series"]
            if series["family_id"] == "drivaerml-velocity-relative-v3"
        )
        relative_velocity["coordinate"][1] += 1.0e-6
        with patch(
            "reference.drivaerml.dataset_scorer._relative_series_support_index",
            return_value=retained,
        ), patch(
            "reference.drivaerml.dataset_scorer._constant_series_support_index",
            return_value=synthetic_constant_support(velocity_document),
        ):
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "coordinate identity differs"
            ):
                validate_schema_v3_relative_profile_chunk_candidate(
                    velocity_document
                )

    def test_constant_velocity_rejects_unbound_subsets_for_every_case(self) -> None:
        document = chunk()
        validate_synthetic_chunk(document)
        retained_relative = synthetic_retained_support(document)
        retained_constant = synthetic_constant_support(document)
        constant_velocity = next(
            series
            for series in document["cases"][0]["series"]
            if series["family_id"] == "drivaerml-autocfd5-constant-v1"
        )
        constant_velocity["coordinate"] = constant_velocity["coordinate"][:2]
        constant_velocity["prediction"] = constant_velocity["prediction"][:2]
        with patch(
            "reference.drivaerml.dataset_scorer._relative_series_support_index",
            return_value=retained_relative,
        ), patch(
            "reference.drivaerml.dataset_scorer._constant_series_support_index",
            return_value=retained_constant,
        ):
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "coordinate count differs from the retained all-case constant",
            ):
                validate_schema_v3_relative_profile_chunk_candidate(document)

    def test_run419_constant_series_require_exact_retained_bindings(self) -> None:
        synthetic = chunk("run_419")
        validate_synthetic_chunk(synthetic)

        wrong_support = copy.deepcopy(synthetic)
        constant_velocity = next(
            series
            for series in wrong_support["cases"][0]["series"]
            if series["family_id"] == "drivaerml-autocfd5-constant-v1"
        )
        retained_relative = synthetic_retained_support(wrong_support)
        retained_constant = synthetic_constant_support(wrong_support)
        constant_velocity["support_identity_sha256"] = digest("wrong-run419-support")
        with patch(
            "reference.drivaerml.dataset_scorer._relative_series_support_index",
            return_value=retained_relative,
        ), patch(
            "reference.drivaerml.dataset_scorer._constant_series_support_index",
            return_value=retained_constant,
        ):
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "support identity differs from the retained all-case constant",
            ):
                validate_schema_v3_relative_profile_chunk_candidate(wrong_support)

        sparse = chunk("run_419")
        sparse_velocity = next(
            series
            for series in sparse["cases"][0]["series"]
            if series["family_id"] == "drivaerml-autocfd5-constant-v1"
        )
        sparse_velocity["coordinate"] = [0.0, 0.02, 0.04]
        sparse_velocity["prediction"] = [0.5, 0.5, 0.5]
        retained_relative = synthetic_retained_support(sparse)
        retained_constant = synthetic_constant_support(sparse)
        with patch(
            "reference.drivaerml.dataset_scorer._relative_series_support_index",
            return_value=retained_relative,
        ), patch(
            "reference.drivaerml.dataset_scorer._constant_series_support_index",
            return_value=retained_constant,
        ):
            validate_schema_v3_relative_profile_chunk_candidate(sparse)
            sparse_velocity["coordinate"][1] = 0.03
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "coordinate identity differs from the retained all-case constant",
            ):
                validate_schema_v3_relative_profile_chunk_candidate(sparse)

        full_grid_with_unbound_identities = chunk("run_419")
        with patch(
            "reference.drivaerml.dataset_scorer._relative_series_support_index",
            return_value=synthetic_retained_support(
                full_grid_with_unbound_identities
            ),
        ):
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "differs from the retained all-case constant support index",
            ):
                validate_schema_v3_relative_profile_chunk_candidate(
                    full_grid_with_unbound_identities
                )

    def test_semantics_reject_stale_velocity_families_and_alias_identity_mismatch(self) -> None:
        for stale_family in (
            "drivaerml-velocity-relative-v1",
            "drivaerml-velocity-relative-v2",
        ):
            with self.subTest(stale_family=stale_family):
                stale = chunk()
                relative_velocity = next(
                    series
                    for series in stale["cases"][0]["series"]
                    if series["family_id"] == "drivaerml-velocity-relative-v3"
                )
                retained = synthetic_retained_support(stale)
                relative_velocity["family_id"] = stale_family
                with self.assertRaisesRegex(DrivAerDatasetScorerError, "undeclared"):
                    with patch(
                        "reference.drivaerml.dataset_scorer._relative_series_support_index",
                        return_value=retained,
                    ), patch(
                        "reference.drivaerml.dataset_scorer._constant_series_support_index",
                        return_value=synthetic_constant_support(stale),
                    ):
                        validate_schema_v3_relative_profile_chunk_candidate(stale)

        mismatched = chunk()
        alias = next(
            series
            for series in mismatched["cases"][0]["series"]
            if series["representation"] == "shared_alias"
        )
        alias["shared_support_ref"]["canonical_support_identity_sha256"] = digest(
            "wrong-canonical-support"
        )
        with self.assertRaisesRegex(DrivAerDatasetScorerError, "retained relative support index"):
            validate_synthetic_chunk(mismatched)

        drifted = chunk()
        constant_velocity = next(
            series
            for series in drifted["cases"][0]["series"]
            if series["family_id"] == "drivaerml-autocfd5-constant-v1"
        )
        constant_velocity["coordinate"][1] += 1.0e-4
        with self.assertRaisesRegex(
            DrivAerDatasetScorerError, "preserve the frozen constant coordinate grid"
        ):
            validate_synthetic_chunk(drifted)

        incomplete = chunk()
        incomplete["cases"][0]["series"].pop()
        with self.assertRaisesRegex(DrivAerDatasetScorerError, "exactly 40"):
            validate_synthetic_chunk(incomplete)

    def test_semantics_reject_wrong_and_all_zero_relative_identities(self) -> None:
        for mutation in (
            "wrong_support",
            "zero_support",
            "wrong_receipt",
            "zero_receipt",
            "zero_alias_support",
        ):
            with self.subTest(mutation=mutation):
                document = chunk()
                if mutation == "zero_alias_support":
                    series = next(
                        item
                        for item in document["cases"][0]["series"]
                        if item["representation"] == "shared_alias"
                    )
                    series["shared_support_ref"][
                        "canonical_support_identity_sha256"
                    ] = "0" * 64
                else:
                    series = next(
                        item
                        for item in document["cases"][0]["series"]
                        if item["family_id"] == "drivaerml-velocity-relative-v3"
                    )
                    if mutation.endswith("support"):
                        series["support_identity_sha256"] = (
                            "0" * 64
                            if mutation.startswith("zero")
                            else digest("hash-shaped-but-wrong-support")
                        )
                    else:
                        series["placement_receipt_identity_sha256"] = (
                            "0" * 64
                            if mutation.startswith("zero")
                            else digest("hash-shaped-but-wrong-receipt")
                        )
                with self.assertRaises(DrivAerDatasetScorerError):
                    validate_synthetic_chunk(document)

    def test_adapter_writes_40_series_per_case_without_changing_legacy_writer(self) -> None:
        evaluation = CandidateDatasetEvaluation(
            {
                "split": {
                    "split_id": "full",
                    "case_set_id": "standard",
                    "case_ids": ["run_1", "run_2"],
                }
            }
        )
        series_by_case = {
            "run_1": namespaced_series("run_1"),
            "run_2": namespaced_series("run_2"),
        }
        synthetic_document = {
            "cases": [
                {"case_id": case_id, "series": series}
                for case_id, series in series_by_case.items()
            ]
        }
        with synthetic_support_patch(synthetic_document):
            package = schema_v3_relative_profile_chunks_candidate_adapter(
                evaluation,
                submission_id="relative-profile-test",
                namespaced_series_by_case=series_by_case,
                cases_per_chunk=1,
            )
        self.assertEqual(package.index["format"], RELATIVE_PROFILE_FORMAT)
        with synthetic_support_patch(synthetic_document):
            with tempfile.TemporaryDirectory() as temporary:
                receipt = write_schema_v3_profile_chunks_candidate(
                    package, Path(temporary) / "profiles"
                )
        self.assertEqual(receipt["case_count"], 2)
        self.assertEqual(receipt["series_count"], 80)

    def test_submission_validator_rejects_two_flag_activation_bypass(self) -> None:
        evaluation = CandidateDatasetEvaluation(
            {
                "split": {
                    "split_id": "full",
                    "case_set_id": "standard",
                    "case_ids": ["run_1"],
                }
            }
        )
        run_1_series = namespaced_series("run_1")
        with synthetic_support_patch(
            {"cases": [{"case_id": "run_1", "series": run_1_series}]}
        ):
            package = schema_v3_relative_profile_chunks_candidate_adapter(
                evaluation,
                submission_id="relative-profile-test",
                namespaced_series_by_case={"run_1": run_1_series},
            )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            directory = temporary_root / "submissions" / "drivaerml" / "relative-profile-test"
            with synthetic_support_patch(
                {"cases": [{"case_id": "run_1", "series": run_1_series}]}
            ):
                write_schema_v3_profile_chunks_candidate(
                    package, directory / "profiles"
                )
            split_path = temporary_root / "benchmark-specs" / "drivaerml" / "splits" / "full.json"
            split = {
                "schema_version": "1.0",
                "dataset_id": "drivaerml",
                "split_id": "full",
                "case_set_id": "standard",
                "case_id_status": "official",
                "case_count": 1,
                "case_ids": ["run_1"],
            }
            write_json(split_path, split)
            submission = {
                "submission_id": "relative-profile-test",
                "dataset_id": "drivaerml",
                "split_id": "full",
                "case_set_id": "standard",
                "split_sha256": sha256_file(split_path),
                "profile_data": {
                    "format": RELATIVE_PROFILE_FORMAT,
                    "index_file": "profiles/index.json",
                    "case_count": 1,
                    "case_set_id": "standard",
                },
            }
            split_entry = {
                "index_file": "splits/full.json",
                "sha256": sha256_file(split_path),
                "case_set_id": "standard",
                "case_id_status": "official",
                "case_count": 1,
            }
            bypass_spec = {
                "profile_panels": [],
                "relative_diagnostics": {
                    "status": "activated",
                    "profile_format_enabled": True,
                    "contract": {"sha256": RELATIVE_PROFILE_CONTRACT_SHA256},
                },
            }
            errors: list[str] = []
            with synthetic_support_patch(
                {"cases": [{"case_id": "run_1", "series": run_1_series}]}
            ):
                with patch("scripts.validate_submission.ROOT", temporary_root):
                    stats = validate_profiles(
                        errors.append,
                        directory,
                        submission,
                        bypass_spec,
                        split_entry,
                    )
            self.assertEqual(stats, {"cases": 1, "series": 40})
            joined = "\n".join(errors)
            self.assertIn("activation_release must contain exactly", joined)
            self.assertIn("relative profile format is closed", joined)

            pending_spec = copy.deepcopy(bypass_spec)
            pending_spec["relative_diagnostics"].update(
                {"status": "support_pending", "profile_format_enabled": False}
            )
            errors = []
            with synthetic_support_patch(
                {"cases": [{"case_id": "run_1", "series": run_1_series}]}
            ):
                with patch("scripts.validate_submission.ROOT", temporary_root):
                    validate_profiles(
                        errors.append,
                        directory,
                        submission,
                        pending_spec,
                        split_entry,
                    )
            self.assertIn("relative profile format is closed", "\n".join(errors))

            candidate_errors: list[str] = []
            with synthetic_support_patch(
                {"cases": [{"case_id": "run_1", "series": run_1_series}]}
            ), patch("scripts.validate_submission.ROOT", temporary_root), patch(
                "scripts.validate_submission.validate_drivaerml_relative_activation_release",
                return_value=True,
            ):
                candidate_stats = validate_profiles(
                    candidate_errors.append,
                    directory,
                    submission,
                    pending_spec,
                    split_entry,
                    candidate_dry_run=True,
                )
            self.assertEqual(candidate_stats, {"cases": 1, "series": 40})
            self.assertNotIn(
                "relative profile format is closed", "\n".join(candidate_errors)
            )

            # A complete retained-inference reference may show its immutable
            # profiles before activation, but only when it carries both of the
            # explicit non-ranking lifecycle markers.  A record_type alone is
            # deliberately not a bypass.
            pre_release_submission = copy.deepcopy(submission)
            pre_release_submission["record_type"] = "pre_release_reference"
            pre_release_submission["approval"] = {"status": "prototype"}
            pre_release_errors: list[str] = []
            with synthetic_support_patch(
                {"cases": [{"case_id": "run_1", "series": run_1_series}]}
            ), patch("scripts.validate_submission.ROOT", temporary_root):
                pre_release_stats = validate_profiles(
                    pre_release_errors.append,
                    directory,
                    pre_release_submission,
                    pending_spec,
                    split_entry,
                )
            self.assertEqual(pre_release_stats, {"cases": 1, "series": 40})
            self.assertEqual(pre_release_errors, [])

            incomplete_reference = copy.deepcopy(pre_release_submission)
            incomplete_reference.pop("approval")
            incomplete_reference_errors: list[str] = []
            with synthetic_support_patch(
                {"cases": [{"case_id": "run_1", "series": run_1_series}]}
            ), patch("scripts.validate_submission.ROOT", temporary_root):
                validate_profiles(
                    incomplete_reference_errors.append,
                    directory,
                    incomplete_reference,
                    pending_spec,
                    split_entry,
                )
            self.assertIn(
                "relative profile format is closed",
                "\n".join(incomplete_reference_errors),
            )


if __name__ == "__main__":
    unittest.main()
