from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import promote_drivaerml_candidate as promoter


ROOT = Path(__file__).resolve().parents[1]
BINDINGS_PATH = ROOT / "benchmark-specs" / "drivaerml" / "candidate-release-bindings.json"
TEMPLATE = ROOT / "examples" / "v3-template"

RELATIVE_MANIFEST_FILES = {
    "velocity_placement_manifest": (
        "support/relative-v3/manifests/velocity-placement-all484-v1.json"
    ),
    "velocity_mapping_manifest": (
        "support/relative-v3/manifests/velocity-mapping-all484-v1.json"
    ),
    "cp_manifest": "support/relative-v3/manifests/cp-native-support-all484-v3.json",
}

RELATIVE_CONTRACT_BINDINGS = {
    "velocity_placement_manifest": (
        "relative_velocity_v3",
        "placement_all484_manifest",
    ),
    "velocity_mapping_manifest": (
        "relative_velocity_v3",
        "mapping_all484_manifest",
    ),
    "cp_manifest": ("relative_cp_v1", "cp_all484_manifest"),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class DrivAerMLCandidateReleaseBindingTests(unittest.TestCase):
    def write_relative_support_fixture(
        self,
        benchmark: Path,
        relative_contract_path: Path,
    ) -> dict[str, str]:
        pin_path = benchmark / "proposal" / "native-source-pin.json"
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            ROOT
            / "benchmark-specs"
            / "drivaerml"
            / "proposal"
            / "native-source-pin.json",
            pin_path,
        )
        case_ids = [case["case_id"] for case in load_json(pin_path)["cases"]]
        self.assertEqual(len(case_ids), 484)

        common = {
            "dataset_id": "drivaerml",
            "public_dataset_revision": promoter.DATASET_REVISION,
        }
        manifests = {
            "velocity_placement_manifest": {
                **common,
                "schema": (
                    "drivaerml-relative-velocity-v3-production-input-manifest-v1"
                ),
                "schema_version": 1,
                "family_id": "drivaerml-velocity-relative-v3",
                "case_count": 484,
                "case_ids": case_ids,
                "case_indices": list(range(484)),
                "cases": [
                    {"case_id": case_id, "case_index": case_index}
                    for case_index, case_id in enumerate(case_ids)
                ],
            },
            "velocity_mapping_manifest": {
                **common,
                "schema": (
                    "drivaerml-velocity-relative-v3-mapping-aggregate-v1"
                ),
                "schema_version": 1,
                "family_id": "drivaerml-velocity-relative-v3",
                "official_case_count": 484,
                "included_case_count": 484,
                "complete_official_case_coverage": True,
                "cases": [{"case_id": case_id} for case_id in case_ids],
            },
            "cp_manifest": {
                **common,
                "schema": "drivaerml-relative-cp-native-support-manifest-v3",
                "schema_version": 3,
                "family_id": "drivaerml_cp_relative_v1",
                "case_count": 484,
                "all_official_cases_generated_and_replayed": True,
                "cases": [
                    {"case_id": case_id, "case_index": case_index}
                    for case_index, case_id in enumerate(case_ids)
                ],
            },
        }

        digests: dict[str, str] = {}
        for label, manifest in manifests.items():
            manifest_path = benchmark / RELATIVE_MANIFEST_FILES[label]
            write_json(manifest_path, manifest)
            digests[label] = promoter.sha256_file(manifest_path)

        relative_contract = load_json(relative_contract_path)
        support_bindings = relative_contract["support_implementation_bindings"]
        for label, (family, binding) in RELATIVE_CONTRACT_BINDINGS.items():
            support_bindings[family][binding]["sha256"] = digests[label]
        write_json(relative_contract_path, relative_contract)
        return digests

    def rewrite_relative_support_manifest(
        self,
        bindings: dict,
        benchmark: Path,
        label: str,
        manifest: dict,
    ) -> None:
        manifest_binding = bindings["relative_support"][label]
        manifest_path = benchmark / manifest_binding["manifest_file"]
        write_json(manifest_path, manifest)
        digest = promoter.sha256_file(manifest_path)
        manifest_binding["manifest_sha256"] = digest

        relative_contract_path = (
            benchmark / bindings["relative_diagnostics_v3"]["file"]
        )
        relative_contract = load_json(relative_contract_path)
        family, contract_binding = RELATIVE_CONTRACT_BINDINGS[label]
        relative_contract["support_implementation_bindings"][family][
            contract_binding
        ]["sha256"] = digest
        write_json(relative_contract_path, relative_contract)
        bindings["relative_diagnostics_v3"]["sha256"] = promoter.sha256_file(
            relative_contract_path
        )

    def make_unresolved_v10_fixture(self, root: Path) -> tuple[dict, Path, Path]:
        benchmark = root / "benchmark-specs" / "drivaerml"
        benchmark.mkdir(parents=True)
        bindings = load_json(BINDINGS_PATH)
        profile_path = benchmark / bindings["profile_definition_v10"]["file"]
        shutil.copy2(
            ROOT / "benchmark-specs" / "drivaerml" / profile_path.name,
            profile_path,
        )
        relative_path = benchmark / bindings["relative_diagnostics_v3"]["file"]
        shutil.copy2(
            ROOT / "benchmark-specs" / "drivaerml" / relative_path.name,
            relative_path,
        )
        relative_schema = (
            root
            / bindings["relative_diagnostics_v3"]["profile_chunk_schema_file"]
        )
        relative_schema.parent.mkdir(parents=True)
        shutil.copy2(
            ROOT / bindings["relative_diagnostics_v3"]["profile_chunk_schema_file"],
            relative_schema,
        )
        source_pin = (
            ROOT
            / "benchmark-specs"
            / "drivaerml"
            / "proposal"
            / "native-source-pin.json"
        )
        retained_pin = benchmark / "proposal" / "native-source-pin.json"
        retained_pin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pin, retained_pin)
        for label in RELATIVE_MANIFEST_FILES:
            manifest_file = bindings["relative_support"][label].get(
                "manifest_file"
            )
            if not isinstance(manifest_file, str):
                continue
            source_manifest = ROOT / "benchmark-specs" / "drivaerml" / manifest_file
            if source_manifest.is_file():
                retained_manifest = benchmark / manifest_file
                retained_manifest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_manifest, retained_manifest)
        bindings_path = benchmark / "candidate-release-bindings.json"
        write_json(bindings_path, bindings)
        return bindings, bindings_path, profile_path

    def make_ready_fixture(
        self, root: Path
    ) -> tuple[dict, Path, Path, dict]:
        benchmark = root / "benchmark-specs" / "drivaerml"
        split_path = benchmark / "splits" / "default.json"
        split = {
            "schema_version": "1.0",
            "dataset_id": "drivaerml",
            "split_id": "default",
            "case_set_id": "standard",
            "split_label": "Default",
            "case_id_status": "official",
            "case_count": 2,
            "case_ids": ["case-001", "case-002"],
        }
        write_json(split_path, split)

        support_root = benchmark / "scoring-support" / "candidate-v1"
        shutil.copytree(TEMPLATE / "support", support_root)
        manifest_path = support_root / "manifest.json"
        release_id = "drivaerml-candidate-support-v1"
        manifest = load_json(manifest_path)
        manifest.update(
            {
                "status": "candidate",
                "release_id": release_id,
                "dataset_id": "drivaerml",
                "dataset_version": promoter.DATASET_VERSION,
                "evaluation_reference_version": promoter.EVALUATOR_VERSION,
            }
        )
        manifest.pop("owner_approval", None)
        for support in manifest["supports"]:
            for binding in support["metric_bindings"]:
                if binding["weighting"] == "support_weights":
                    binding["dataset_weighting"] = "cell_volume"

        chunk_path = support_root / "case-sets" / "standard" / "chunk-000.json"
        chunk = load_json(chunk_path)
        chunk.update({"release_id": release_id, "dataset_id": "drivaerml"})
        write_json(chunk_path, chunk)
        index_path = support_root / "case-sets" / "standard" / "index.json"
        index = load_json(index_path)
        index.update({"release_id": release_id, "dataset_id": "drivaerml"})
        index["chunks"][0]["sha256"] = promoter.sha256_file(chunk_path)
        write_json(index_path, index)
        manifest["case_sets"][0]["index_sha256"] = promoter.sha256_file(index_path)
        write_json(manifest_path, manifest)
        global_truth = {
            "release_id": "prototype-profile-ground-truth-2026-08-21",
            "manifest_sha256": "2" * 64,
        }
        global_manifest_path = root / "leaderboard" / "manifest.json"
        write_json(
            global_manifest_path,
            {"data_release": {"profile_ground_truth": global_truth}},
        )
        profile_path = benchmark / "drivaerml-diagnostics-v10.json"
        write_json(profile_path, {"id": "drivaerml-diagnostics-v10-candidate"})
        profile_sha256 = promoter.sha256_file(profile_path)
        relative_contract_path = benchmark / "drivaerml-relative-diagnostics-v3.json"
        shutil.copy2(
            ROOT
            / "benchmark-specs"
            / "drivaerml"
            / "drivaerml-relative-diagnostics-v3.json",
            relative_contract_path,
        )
        relative_schema_path = root / "schemas" / "v1" / "drivaerml-relative-profile-chunk.schema.json"
        relative_schema_path.parent.mkdir(parents=True)
        shutil.copy2(
            ROOT / "schemas" / "v1" / "drivaerml-relative-profile-chunk.schema.json",
            relative_schema_path,
        )
        relative_manifest_digests = self.write_relative_support_fixture(
            benchmark,
            relative_contract_path,
        )
        metrics = [
            {
                "id": binding["metric_id"],
                "aggregation": binding["aggregation"],
                "weighting": binding["dataset_weighting"],
            }
            for support in manifest["supports"]
            for binding in support["metric_bindings"]
        ]
        generated = {
            "schema_version": "1.1",
            "dataset_id": "drivaerml",
            "dataset_name": "DrivAerML",
            "dataset_version": promoter.DATASET_VERSION,
            "evaluation_reference_version": promoter.EVALUATOR_VERSION,
            "metrics": metrics,
            "splits": [
                {
                    "id": "default",
                    "label": "Default",
                    "index_file": "splits/default.json",
                    "case_count": 2,
                    "case_set_id": "standard",
                    "case_id_status": "official",
                    "sha256": promoter.sha256_file(split_path),
                }
            ],
            "profile_definition": {
                "id": "drivaerml-diagnostics-v10-candidate",
                "file": profile_path.name,
                "sha256": profile_sha256,
            },
            "scoring_support": {
                "status": "owner_review_required",
                "submissions_open": False,
                "dataset_evaluator_binding": {
                    "status": "pending_frozen_release",
                    "evaluator_reference_version": promoter.EVALUATOR_VERSION,
                    "evaluator_code_revision": None,
                },
                "profile_definition": {
                    "file": profile_path.name,
                    "sha256": profile_sha256,
                    "status": "candidate",
                },
            },
        }
        bindings = {
            "schema": "drivaerml-fluidsbench-candidate-release-bindings-v1",
            "status": "ready",
            "unresolved_token_prefix": promoter.UNRESOLVED_RELEASE_PREFIX,
            "candidate_manifest": {
                "status": "candidate",
                "release_id": release_id,
                "manifest_file": "scoring-support/candidate-v1/manifest.json",
                "manifest_url": (
                    "https://fluidsbench.org/assets/drivaerml/"
                    f"{release_id}/manifest.json"
                ),
                "manifest_sha256": promoter.sha256_file(manifest_path),
            },
            "evaluator": {
                "reference_version": promoter.EVALUATOR_VERSION,
                "code_revision": "3" * 40,
            },
            "profile_definition_v10": {
                "file": profile_path.name,
                "sha256": profile_sha256,
            },
            "relative_diagnostics_v3": {
                "file": relative_contract_path.name,
                "sha256": promoter.sha256_file(relative_contract_path),
                "profile_chunk_schema_file": (
                    "schemas/v1/drivaerml-relative-profile-chunk.schema.json"
                ),
                "profile_chunk_schema_sha256": promoter.sha256_file(
                    relative_schema_path
                ),
            },
            "relative_support": {
                "status": "ready",
                "required_case_count": 484,
                "velocity_placement_manifest": {
                    "family_id": "drivaerml-velocity-relative-v3",
                    "producer_file": (
                        "velocity_support_v3/production_campaign_v1/aggregate/"
                        "relative-velocity-v3-production-all484-inputs-v1.json"
                    ),
                    "expected_schema": (
                        "drivaerml-relative-velocity-v3-production-input-manifest-v1"
                    ),
                    "manifest_file": RELATIVE_MANIFEST_FILES[
                        "velocity_placement_manifest"
                    ],
                    "manifest_sha256": relative_manifest_digests[
                        "velocity_placement_manifest"
                    ],
                },
                "velocity_mapping_manifest": {
                    "family_id": "drivaerml-velocity-relative-v3",
                    "producer_file": (
                        "velocity_mapping_v3/aggregate/"
                        "relative-velocity-v3-mapping-all484-v1.json"
                    ),
                    "expected_schema": (
                        "drivaerml-velocity-relative-v3-mapping-aggregate-v1"
                    ),
                    "manifest_file": RELATIVE_MANIFEST_FILES[
                        "velocity_mapping_manifest"
                    ],
                    "manifest_sha256": relative_manifest_digests[
                        "velocity_mapping_manifest"
                    ],
                },
                "cp_manifest": {
                    "family_id": "drivaerml_cp_relative_v1",
                    "producer_file": (
                        "cp_support/campaign_v3/aggregate_v3/all484/"
                        "relative-cp-native-support-manifest-v3.json"
                    ),
                    "expected_schema": (
                        "drivaerml-relative-cp-native-support-manifest-v3"
                    ),
                    "manifest_file": RELATIVE_MANIFEST_FILES["cp_manifest"],
                    "manifest_sha256": relative_manifest_digests["cp_manifest"],
                },
            },
            "profile_ground_truth": global_truth,
        }
        return bindings, benchmark, global_manifest_path, generated

    def test_checked_in_unresolved_hand_off_never_leaks_into_active_spec(self) -> None:
        bindings = promoter.load_candidate_release_bindings(BINDINGS_PATH)
        self.assertEqual(bindings["status"], "unresolved")
        tokens = [
            item
            for item in promoter.unresolved_release_tokens(bindings)
            if item[0] != ("unresolved_token_prefix",)
        ]
        self.assertEqual(len(tokens), 7)
        self.assertFalse(
            any(path[:1] == ("relative_support",) for path, _token in tokens)
        )
        self.assertEqual(bindings["relative_support"]["status"], "ready")
        approval = bindings["scientific_approval"]
        self.assertEqual(approval["status"], "approved")
        self.assertEqual(approval["approved_by"], "neilashton")
        approval_path = BINDINGS_PATH.parent / approval["approval_record"]["file"]
        self.assertEqual(
            approval["approval_record"]["sha256"],
            promoter.sha256_file(approval_path),
        )
        active = load_json(ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json")
        self.assertEqual(promoter.unresolved_release_tokens(active), [])
        self.assertNotIn("candidate_manifest", active["scoring_support"])
        self.assertEqual(active["profile_definition"]["id"], "drivaerml-diagnostics-v9-candidate")
        profile_path = BINDINGS_PATH.parent / bindings["profile_definition_v10"]["file"]
        self.assertEqual(
            bindings["profile_definition_v10"]["sha256"],
            promoter.sha256_file(profile_path),
        )
        profile = load_json(profile_path)
        self.assertEqual(profile["id"], promoter.PROFILE_DEFINITION_V10_ID)
        self.assertEqual(profile["dataset_id"], "drivaerml")
        self.assertEqual(profile["pressure_cuts"]["truth_source_array"], "CpMeanTrim")
        self.assertEqual(profile["pressure_cuts"]["prediction_source_array"], "pMeanTrim")

    def test_scientific_approval_binding_is_exact_and_nonactivating(self) -> None:
        bindings = load_json(BINDINGS_PATH)
        promoter.validate_scientific_approval_binding(
            bindings["scientific_approval"],
            bindings_path=BINDINGS_PATH,
            release_bindings=bindings,
        )
        approval_path = (
            BINDINGS_PATH.parent
            / bindings["scientific_approval"]["approval_record"]["file"]
        )
        approval = load_json(approval_path)
        self.assertTrue(
            all(
                approval["non_activation"][field] is False
                for field in (
                    "submissions_opened",
                    "official_ranking_enabled",
                    "production_evaluator_release_bound",
                    "scoring_support_release_bound",
                    "genuine_three_model_sensitivity_complete",
                    "independent_full_submission_dry_run_complete",
                    "relative_profile_format_authorized",
                )
            )
        )

        changed = copy.deepcopy(bindings["scientific_approval"])
        changed["approval_record"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not the reviewed record"):
            promoter.validate_scientific_approval_binding(
                changed,
                bindings_path=BINDINGS_PATH,
                release_bindings=bindings,
            )

    def test_unresolved_generator_preserves_later_candidate_and_v10_bindings(self) -> None:
        bindings = promoter.load_candidate_release_bindings(BINDINGS_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            ready, benchmark, global_manifest_path, generated = self.make_ready_fixture(
                Path(temporary)
            )
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ):
                existing = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), ready, None
                )
                existing["scoring_support"]["dataset_evaluator_binding"][
                    "legacy_extra"
                ] = "validated but deliberately not copied"
                existing["scoring_support"]["profile_definition"][
                    "legacy_extra"
                ] = "validated but deliberately not copied"
                result = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), bindings, existing
                )
            self.assertEqual(
                result["scoring_support"]["candidate_manifest"],
                existing["scoring_support"]["candidate_manifest"],
            )
            self.assertEqual(
                result["scoring_support"]["dataset_evaluator_binding"],
                {
                    key: value
                    for key, value in existing["scoring_support"][
                        "dataset_evaluator_binding"
                    ].items()
                    if key != "legacy_extra"
                },
            )
            self.assertEqual(
                result["scoring_support"]["profile_definition"],
                {
                    key: value
                    for key, value in existing["scoring_support"][
                        "profile_definition"
                    ].items()
                    if key != "legacy_extra"
                },
            )
            self.assertNotIn(
                "legacy_extra",
                result["scoring_support"]["dataset_evaluator_binding"],
            )

    def test_ready_binding_is_idempotent_and_checks_manifest_and_global_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, benchmark, global_manifest_path, generated = (
                self.make_ready_fixture(Path(temporary))
            )
            bindings_path = benchmark / "candidate-release-bindings.json"
            write_json(bindings_path, bindings)
            loaded = promoter.load_candidate_release_bindings(bindings_path)
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ):
                first = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), loaded, None
                )
                second = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), loaded, copy.deepcopy(first)
                )
            self.assertEqual(first, second)
            self.assertEqual(
                first["scoring_support"]["candidate_manifest"],
                bindings["candidate_manifest"],
            )
            self.assertEqual(
                first["scoring_support"]["dataset_evaluator_binding"]["status"],
                "frozen",
            )

            bad_hash = copy.deepcopy(bindings)
            bad_hash["candidate_manifest"]["manifest_sha256"] = "7" * 64
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ), self.assertRaisesRegex(ValueError, "does not match its local file"):
                promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), bad_hash, None
                )

            bad_truth = copy.deepcopy(bindings)
            bad_truth["profile_ground_truth"]["manifest_sha256"] = "8" * 64
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ), self.assertRaisesRegex(ValueError, "global leaderboard binding"):
                promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), bad_truth, None
                )

    def test_ranked_candidate_readiness_does_not_require_relative_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, benchmark, _global_manifest_path, _generated = (
                self.make_ready_fixture(Path(temporary))
            )
            bindings["relative_support"]["status"] = "unresolved"
            bindings["relative_support"]["velocity_mapping_manifest"][
                "manifest_sha256"
            ] = "__UNRESOLVED_DRIVAERML_RELATIVE_VELOCITY_V3_MAPPING_MANIFEST_SHA256__"
            bindings_path = benchmark / "candidate-release-bindings.json"
            write_json(bindings_path, bindings)
            loaded = promoter.load_candidate_release_bindings(bindings_path)
            self.assertEqual(loaded["status"], "ready")
            self.assertEqual(loaded["relative_support"]["status"], "unresolved")

    def test_relative_support_ready_rejects_pending_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, bindings_path, _profile_path = self.make_unresolved_v10_fixture(
                Path(temporary)
            )
            bindings["relative_support"]["status"] = "ready"
            bindings["relative_support"]["velocity_mapping_manifest"][
                "manifest_sha256"
            ] = "__UNRESOLVED_DRIVAERML_RELATIVE_VELOCITY_V3_MAPPING_MANIFEST_SHA256__"
            write_json(bindings_path, bindings)
            with self.assertRaisesRegex(ValueError, "still contains unresolved tokens"):
                promoter.load_candidate_release_bindings(bindings_path)

    def test_relative_support_rejects_false_hash_shaped_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, benchmark, _global_manifest_path, _generated = (
                self.make_ready_fixture(Path(temporary))
            )
            bindings["relative_support"]["velocity_placement_manifest"][
                "manifest_sha256"
            ] = "0" * 64
            bindings_path = benchmark / "candidate-release-bindings.json"
            write_json(bindings_path, bindings)
            with self.assertRaisesRegex(ValueError, "does not match its retained file"):
                promoter.load_candidate_release_bindings(bindings_path)

    def test_relative_support_rejects_483_case_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, benchmark, _global_manifest_path, _generated = (
                self.make_ready_fixture(Path(temporary))
            )
            label = "velocity_placement_manifest"
            manifest_path = (
                benchmark / bindings["relative_support"][label]["manifest_file"]
            )
            manifest = load_json(manifest_path)
            manifest["cases"].pop()
            manifest["case_ids"].pop()
            manifest["case_indices"].pop()
            manifest["case_count"] = 483
            self.rewrite_relative_support_manifest(
                bindings,
                benchmark,
                label,
                manifest,
            )
            bindings_path = benchmark / "candidate-release-bindings.json"
            write_json(bindings_path, bindings)
            with self.assertRaisesRegex(ValueError, "official 484-case set"):
                promoter.load_candidate_release_bindings(bindings_path)

    def test_relative_support_rejects_duplicate_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, benchmark, _global_manifest_path, _generated = (
                self.make_ready_fixture(Path(temporary))
            )
            label = "velocity_mapping_manifest"
            manifest_path = (
                benchmark / bindings["relative_support"][label]["manifest_file"]
            )
            manifest = load_json(manifest_path)
            manifest["cases"][-1]["case_id"] = manifest["cases"][0]["case_id"]
            self.rewrite_relative_support_manifest(
                bindings,
                benchmark,
                label,
                manifest,
            )
            bindings_path = benchmark / "candidate-release-bindings.json"
            write_json(bindings_path, bindings)
            with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
                promoter.load_candidate_release_bindings(bindings_path)

    def test_relative_support_rejects_wrong_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bindings, benchmark, _global_manifest_path, _generated = (
                self.make_ready_fixture(Path(temporary))
            )
            label = "cp_manifest"
            manifest_path = (
                benchmark / bindings["relative_support"][label]["manifest_file"]
            )
            manifest = load_json(manifest_path)
            manifest["cases"][-1]["case_id"] = "run_999"
            self.rewrite_relative_support_manifest(
                bindings,
                benchmark,
                label,
                manifest,
            )
            bindings_path = benchmark / "candidate-release-bindings.json"
            write_json(bindings_path, bindings)
            with self.assertRaisesRegex(ValueError, "official 484-case set"):
                promoter.load_candidate_release_bindings(bindings_path)

    def test_ready_manifest_requires_valid_complete_candidate_support(self) -> None:
        for mutation, message in (
            ("missing_supports", "supports"),
            ("owner_approval", "must not claim owner approval"),
            ("semantic_binding", "dataset_weighting must match"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                bindings, benchmark, global_manifest_path, generated = (
                    self.make_ready_fixture(Path(temporary))
                )
                manifest_path = benchmark / bindings["candidate_manifest"][
                    "manifest_file"
                ]
                manifest = load_json(manifest_path)
                if mutation == "missing_supports":
                    manifest.pop("supports")
                else:
                    if mutation == "owner_approval":
                        manifest["owner_approval"] = {
                            "approved_by": "Not allowed for a candidate",
                            "approved_at": "2026-08-22",
                            "pull_request_url": "https://github.com/example/repo/pull/1",
                        }
                    else:
                        manifest["supports"][0]["metric_bindings"][0][
                            "dataset_weighting"
                        ] = "entities_equal"
                write_json(manifest_path, manifest)
                bindings["candidate_manifest"]["manifest_sha256"] = (
                    promoter.sha256_file(manifest_path)
                )
                with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                    promoter, "MANIFEST_PATH", global_manifest_path
                ), self.assertRaisesRegex(ValueError, message):
                    promoter.apply_release_managed_bindings(
                        copy.deepcopy(generated), bindings, None
                    )

    def test_release_handoff_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bindings.json"
            text = BINDINGS_PATH.read_text(encoding="utf-8").replace(
                "{\n",
                '{\n  "status": "unresolved",\n',
                1,
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(
                promoter.AuthoritativeJSONError, "duplicate key 'status'"
            ):
                promoter.load_candidate_release_bindings(path)

    def test_unresolved_profile_v10_pair_is_atomic_and_strict(self) -> None:
        for mutation, message in (
            ("file_only_token", "must be resolved together"),
            ("sha_only_token", "must be resolved together"),
            ("parent_escape", "must stay inside"),
            ("absolute", "must stay inside"),
            ("unsafe_hash", "incomplete or invalid"),
            ("wrong_hash", "SHA-256 changed"),
            ("invalid_id", "id must equal"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                bindings, bindings_path, profile_path = self.make_unresolved_v10_fixture(
                    Path(temporary)
                )
                profile_binding = bindings["profile_definition_v10"]
                if mutation == "file_only_token":
                    profile_binding["file"] = "__UNRESOLVED_DRIVAERML_PROFILE_FILE__"
                elif mutation == "sha_only_token":
                    profile_binding["sha256"] = "__UNRESOLVED_DRIVAERML_PROFILE_SHA256__"
                elif mutation == "parent_escape":
                    profile_binding["file"] = f"../{profile_path.name}"
                elif mutation == "absolute":
                    profile_binding["file"] = str(profile_path)
                elif mutation == "unsafe_hash":
                    profile_binding["sha256"] = "A" * 64
                elif mutation == "wrong_hash":
                    profile_binding["sha256"] = "0" * 64
                else:
                    write_json(profile_path, {"id": "drivaerml-diagnostics-v10-wrong"})
                    profile_binding["sha256"] = promoter.sha256_file(profile_path)
                write_json(bindings_path, bindings)
                with self.assertRaisesRegex(ValueError, message):
                    promoter.load_candidate_release_bindings(bindings_path)

    def test_unresolved_profile_v10_rejects_ambiguous_json_and_symlink_escape(self) -> None:
        for mutation, message in (
            ("duplicate", "duplicate key 'id'"),
            ("nonfinite", "forbidden non-finite token NaN"),
            ("symlink_escape", "must stay inside"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bindings, bindings_path, profile_path = self.make_unresolved_v10_fixture(root)
                if mutation == "duplicate":
                    profile_path.write_text(
                        '{"id":"drivaerml-diagnostics-v10-candidate",'
                        '"id":"drivaerml-diagnostics-v10-candidate"}\n',
                        encoding="utf-8",
                    )
                elif mutation == "nonfinite":
                    profile_path.write_text(
                        '{"id":"drivaerml-diagnostics-v10-candidate","bad":NaN}\n',
                        encoding="utf-8",
                    )
                else:
                    outside = root / "outside-v10.json"
                    write_json(outside, {"id": promoter.PROFILE_DEFINITION_V10_ID})
                    profile_path.unlink()
                    try:
                        profile_path.symlink_to(outside)
                    except OSError as error:
                        self.skipTest(f"symbolic links are unavailable: {error}")
                bindings["profile_definition_v10"]["sha256"] = promoter.sha256_file(
                    profile_path
                )
                write_json(bindings_path, bindings)
                with self.assertRaisesRegex(ValueError, message):
                    promoter.load_candidate_release_bindings(bindings_path)

    def test_ready_profile_symlink_cannot_escape_drivaerml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings, benchmark, global_manifest_path, _generated = (
                self.make_ready_fixture(root)
            )
            profile_path = benchmark / bindings["profile_definition_v10"]["file"]
            outside = root / "outside-v10.json"
            write_json(outside, {"id": "drivaerml-diagnostics-v10-candidate"})
            profile_path.unlink()
            try:
                profile_path.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")
            bindings["profile_definition_v10"]["sha256"] = promoter.sha256_file(
                outside
            )
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ), self.assertRaisesRegex(ValueError, "must stay inside"):
                promoter.select_generation_profile(bindings, None)

    def test_unresolved_selection_does_not_regress_an_existing_v10_file(self) -> None:
        bindings = promoter.load_candidate_release_bindings(BINDINGS_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            ready, benchmark, global_manifest_path, generated = self.make_ready_fixture(
                Path(temporary)
            )
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ):
                existing = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), ready, None
                )
                selected_path, selected_profile = promoter.select_generation_profile(
                    bindings, existing
                )
            self.assertEqual(
                selected_path,
                (benchmark / "drivaerml-diagnostics-v10.json").resolve(),
            )
            self.assertEqual(
                selected_profile["id"], "drivaerml-diagnostics-v10-candidate"
            )

    def test_unresolved_rerun_rejects_partial_or_stale_preserved_state(self) -> None:
        unresolved = promoter.load_candidate_release_bindings(BINDINGS_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            ready, benchmark, global_manifest_path, generated = self.make_ready_fixture(
                Path(temporary)
            )
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ):
                existing = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), ready, None
                )
                for mutation, message in (
                    ("candidate", "candidate manifest"),
                    ("token", "unresolved release tokens"),
                    ("extra_token", "unresolved release tokens"),
                    ("evaluator", "evaluator binding"),
                    ("profile", "profile-v10"),
                    ("profile_bytes", "SHA-256 changed"),
                ):
                    stale = copy.deepcopy(existing)
                    if mutation == "candidate":
                        stale["scoring_support"]["candidate_manifest"].pop(
                            "manifest_sha256"
                        )
                    elif mutation == "token":
                        stale["scoring_support"]["candidate_manifest"][
                            "release_id"
                        ] = "__UNRESOLVED_DRIVAERML_RELEASE_ID__"
                    elif mutation == "extra_token":
                        stale["scoring_support"]["dataset_evaluator_binding"][
                            "legacy_note"
                        ] = "__UNRESOLVED_DRIVAERML_HIDDEN_LEGACY_FIELD__"
                    elif mutation == "evaluator":
                        stale["scoring_support"]["dataset_evaluator_binding"][
                            "evaluator_reference_version"
                        ] = "wrong-evaluator"
                    elif mutation == "profile":
                        stale["profile_definition"]["sha256"] = "f" * 64
                    else:
                        write_json(
                            benchmark / stale["profile_definition"]["file"],
                            {"id": "drivaerml-diagnostics-v10-stale"},
                        )
                    with self.subTest(mutation=mutation), self.assertRaisesRegex(
                        ValueError, message
                    ):
                        promoter.apply_release_managed_bindings(
                            copy.deepcopy(generated), unresolved, stale
                        )

    def test_unresolved_rerun_validates_old_manifest_against_new_contract(self) -> None:
        unresolved = promoter.load_candidate_release_bindings(BINDINGS_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            ready, benchmark, global_manifest_path, generated = self.make_ready_fixture(
                Path(temporary)
            )
            with patch.object(promoter, "BENCHMARK_ROOT", benchmark), patch.object(
                promoter, "MANIFEST_PATH", global_manifest_path
            ):
                existing = promoter.apply_release_managed_bindings(
                    copy.deepcopy(generated), ready, None
                )
                changed_contract = copy.deepcopy(generated)
                changed_contract["metrics"][0]["weighting"] = "entities_equal"
                with self.assertRaisesRegex(
                    ValueError, "dataset_weighting must match"
                ):
                    promoter.apply_release_managed_bindings(
                        changed_contract, unresolved, existing
                    )


if __name__ == "__main__":
    unittest.main()
