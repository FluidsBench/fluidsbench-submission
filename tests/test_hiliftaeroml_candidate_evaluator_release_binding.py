from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "benchmark-specs" / "hiliftaeroml"
BINDING_PATH = DATASET_ROOT / "candidate-evaluator-release-binding.json"
IMPLEMENTATION_MANIFEST_PATH = (
    DATASET_ROOT / "evaluator-implementation-manifest.candidate.json"
)
RUNTIME_LOCK_PATH = ROOT / "requirements-hiliftaeroml-evaluator.lock.txt"
TOKEN_PREFIX = "__UNRESOLVED_HILIFTAEROML_"
EVALUATOR_REVISION = "68899f780d96b70f2badb5658971c87af0b17172"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unresolved_tokens(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            token
            for child in value.values()
            for token in unresolved_tokens(child)
        ]
    if isinstance(value, list):
        return [token for child in value for token in unresolved_tokens(child)]
    if (
        isinstance(value, str)
        and value != TOKEN_PREFIX
        and value.startswith(TOKEN_PREFIX)
    ):
        return [value]
    return []


def load_binding() -> dict:
    value = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_candidate_binding_is_nonactivating_and_fail_closed() -> None:
    binding = load_binding()
    assert binding["schema"] == (
        "hiliftaeroml-fluidsbench-candidate-evaluator-release-binding-v1"
    )
    assert binding["schema_version"] == 1
    assert binding["status"] == (
        "candidate_full360_replay_complete_owner_approval_pending"
    )
    assert binding["activation_effect"] == "none"
    assert binding["dataset_id"] == "hiliftaeroml"
    assert binding["unresolved_token_prefix"] == TOKEN_PREFIX

    repositories = binding["repositories"]
    adapter = repositories["fluidsbench_adapter"]
    assert adapter["observed_base_revision"] == (
        "f18f3aabf5ed34e4d0502f584c4df826250e9bd2"
    )
    assert adapter["observed_base_revision_contains_candidate_implementation"] is False
    assert adapter["candidate_implementation_state"] == (
        "committed_at_immutable_revision_and_cross_repository_source_bytes_"
        "manifest_bound"
    )
    assert adapter["immutable_evaluator_code_revision"] == EVALUATOR_REVISION

    native = repositories["native_evaluator"]
    assert native["observed_base_revision"] == (
        "651c62656dd4f0ce5de9cd5afd0dbc389ae105bc"
    )
    assert native["observed_base_revision_contains_candidate_implementation"] is False
    assert native["candidate_implementation_state"] == (
        "native_source_bytes_manifest_bound_but_native_git_revision_unresolved"
    )
    assert native["immutable_native_evaluator_code_revision"] == (
        "__UNRESOLVED_HILIFTAEROML_NATIVE_EVALUATOR_CODE_REVISION__"
    )

    gates = binding["activation_gates"]
    assert gates["force_truth_policy_selected"] is True
    assert gates["force_coefficient_normalization_selected"] is True
    assert gates["force_truth_policy_implemented_in_native_evaluator"] is True
    assert gates["full_360_force_and_overall_replay_complete"] is True
    assert gates["evaluator_implementation_manifest_complete"] is True
    assert gates["immutable_evaluator_revision_bound"] is True
    assert gates["owner_scientific_approval"] is False
    assert gates["public_profile_truth_published"] is False
    assert gates["submissions_open"] is False

    tokens = unresolved_tokens(binding)
    assert set(tokens) == {
        "__UNRESOLVED_HILIFTAEROML_NATIVE_EVALUATOR_CODE_REVISION__",
        "__UNRESOLVED_HILIFTAEROML_OWNER_APPROVAL_RECORD_SHA256__",
    }


def test_implementation_manifest_and_runtime_lock_are_exactly_bound() -> None:
    scope = load_binding()["implementation_scope_to_freeze"]
    assert scope["implementation_manifest_sha256"] == sha256_file(
        IMPLEMENTATION_MANIFEST_PATH
    ) == "f94cef375b45c8604bbd65bb59c679af417340b39dd941c27beb363d59f883d6"
    assert scope["runtime_dependency_lock_sha256"] == sha256_file(
        RUNTIME_LOCK_PATH
    ) == "155f81a0d0ea1d88eb4b12e14e135eca07532fcdfde297bcb7fbc56c9469807b"

    manifest = json.loads(IMPLEMENTATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_file_count"] == 860
    assert manifest["content_fingerprint"] == (
        "55d201297d3a71517a65b9021e2b458d99a0d642304c6b8d5c73933bb2e0e924"
    )
    assert manifest["source_inventory_fingerprint"] == (
        "615f1b8901fe663c6f386ceb2557863c1792c16ec535cc851ab4c59548bac516"
    )
    native_runtime = manifest["runtime_boundaries"]["native_gpu_container"]
    assert native_runtime["image_digest"] is None
    assert native_runtime["host_python_environment_lock"] is None


def test_retained_evidence_hashes_match_repository_bytes() -> None:
    binding = load_binding()
    evidence = binding["retained_evidence_bindings"]
    assert len({item["id"] for item in evidence}) == len(evidence)
    assert len({item["file"] for item in evidence}) == len(evidence)
    for item in evidence:
        relative = Path(item["file"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = DATASET_ROOT / relative
        assert path.is_file()
        assert not path.is_symlink()
        assert sha256_file(path) == item["sha256"]


def test_force_policy_preserves_perfect_prediction_and_complete_case_scoring() -> None:
    policy = load_binding()["force_truth_policy"]
    assert policy["status"] == (
        "implemented_exact_degree1_force_exact_degree2_pitch_case_qref_"
        "full360_prefreeze_passed"
    )
    assert policy["scored_truth_source"] == (
        "canonical_reintegration_of_native_surface_truth_fields_for_every_case"
    )
    assert policy["coefficient_normalization"] == "case_qRef_from_ref_values_csv"
    assert policy["monitor_comparison_protocol"] == (
        "hilift-native-truth-vs-monitor-ci95-v1"
    )
    assert policy["current_native_pitching_moment_status"] == (
        "exact_degree_2_ordered_fan_prediction_and_truth_reintegration_complete"
    )
    assert policy["pressure_force_quadrature"] == (
        "ordered_fan_exact_degree_1_linear_traction"
    )
    assert policy["viscous_force_quadrature"] == (
        "ordered_fan_exact_degree_1_linear_traction"
    )
    assert policy["pitching_moment_quadrature"] == (
        "hiliftaeroml-ordered-fan-exact-linear-traction-moment-v1"
    )
    assert policy["perfect_surface_prediction_must_receive_perfect_load_metrics"] is True
    assert policy["case_exclusion_or_downweighting_permitted"] is False
    assert policy["per_case_tolerance_inflation_permitted"] is False
    assert policy["published_force_moment_csv_role"] == (
        "released_reference_and_provenance_diagnostic; six documented "
        "mean-coefficient rows are canonical surface-integrated overrides"
    )
    assert len(policy["historical_public_monitor_mismatch_case_ids"]) == 5
    assert policy["historical_outside_current_union_monitor_mismatch_case_ids"] == [
        "geo_LHC129_AoA_4"
    ]
    assert policy["surface_reintegration_value_substitution_case_count"] == 6
    assert policy["implementation_evidence_sha256"] == (
        "9c9e98ed1eee7160632f36c907d9e2599dd30c5f9c6741330ee12e0975104186"
    )


def test_force_release_is_bound_separately_from_unchanged_vtu_archives() -> None:
    release = load_binding()["dataset_force_release_binding"]
    assert release["status"] == "published_and_remote_sha256_verified"
    assert release["release_revision"] == (
        "bbec30bcfc6103309c1375c5228b3ad0a586bfaf"
    )
    assert release["release_tag"] == "force-mom-overrides-v1.1"
    assert release["pre_update_revision"] == (
        "dd186e13724f39fc94186308519748cca32399eb"
    )
    assert release["pre_update_tag"] == "force-mom-monitor-v1"
    assert release["archive_source_inventory_revision"] == (
        "1c266d3869bc2968ff97d2107c9c3919be03ed32"
    )
    assert release["archive_source_objects_changed"] is False
    assert release["manifest"] == {
        "file": "force_mom_surface_overrides_v1.json",
        "sha256": "ccdb8e4f9f26176fd5eef8c87efbef5f1a9a211335d37694dcd8c1278a414f2f",
    }
    assert release["aggregate"]["changed_rows"] == 6
    assert release["aggregate"]["unchanged_rows"] == 1794
    assert len(release["per_case_csv_sha256"]) == 6


def test_external_audits_are_bound_but_not_misrepresented_as_final_freeze() -> None:
    binding = load_binding()
    external = binding["external_audit_bindings"]
    moment = external["exact_moment_all_1800"]
    assert moment["status"] == "complete_validation_passed"
    assert moment["case_count"] == 1800
    assert moment["fluidsbench_union_case_count"] == 1355
    assert moment["published_coefficients_modified"] is False
    assert moment["source_fields_modified"] is False

    full = external["transolver_full_360_legacy_native_aggregate"]
    assert full["case_count"] == 360
    assert full["force_score_computed"] is False
    assert full["overall_score_computed"] is False
    assert full["acceptable_as_final_freeze_evidence"] is False

    prefreeze = external["transolver_full_360_exact_native_prefreeze"]
    assert prefreeze["status"] == (
        "complete_exact_force_replay_overall_delegated_to_fluidsbench_assembler"
    )
    assert prefreeze["case_count"] == 360
    assert prefreeze["case_set_id"] == "caseset-ac791749e527"
    assert prefreeze["case_set_sha256"] == (
        "ac791749e5279ecf6746fcce20e3ec32408fd33b22127d5270de968be7842acf"
    )
    assert prefreeze["submission_results_sha256"] == (
        load_binding()["force_truth_policy"]["implementation_evidence_sha256"]
    )
    assert prefreeze["aggregate_a_logical_tree_sha256"] == prefreeze[
        "aggregate_b_logical_tree_sha256"
    ]
    assert prefreeze["receipts_a_logical_tree_sha256"] == prefreeze[
        "receipts_b_logical_tree_sha256"
    ]
    assert prefreeze["order_invariance_status"] == "pass"
    assert prefreeze["order_invariance_forward_sha256"] == prefreeze[
        "order_invariance_reverse_sha256"
    ]
    assert prefreeze["independent_aggregate_trees_byte_identical"] is True
    assert prefreeze["independent_receipt_trees_byte_identical"] is True
    assert prefreeze["force_score_computed"] is True
    assert prefreeze["overall_score_computed"] is False
    assert prefreeze["acceptable_as_force_policy_implementation_evidence"] is True
    assert prefreeze["acceptable_as_final_package_completion_evidence"] is False

    replay = external["transolver_full_360_schema_v3_candidate_replay"]
    assert replay["status"] == "complete_closed_candidate_reproducible"
    assert replay["submission_id"] == (
        "hiliftaeroml-transolver-full360-candidate-v1"
    )
    assert replay["dataset_version"] == "hiliftaeroml-native-v1-candidate"
    assert replay["split_id"] == "full"
    assert replay["case_count"] == 360
    assert replay["case_set_id"] == "caseset-ac791749e527"
    assert replay["case_set_sha256"] == (
        "ac791749e5279ecf6746fcce20e3ec32408fd33b22127d5270de968be7842acf"
    )
    assert replay["exact_surface_summary_content_fingerprint"] == (
        "85ac770b7330ad229682cb46291a5be4a6ae7f6141af737ed4fb4e9fcd20fd80"
    )
    assert replay["evaluator_reference_version"] == binding["reference_version"]
    assert replay["evaluator_code_revision"] == EVALUATOR_REVISION
    assert replay["package_config_sha256"] == (
        "b4b1d2085d90d5c8fac309b96cb4b041f949a0a1a0ef0f68c698784dc581fe86"
    )
    assert replay["submission_specification_sha256"] == (
        "59c6ce89d09c562486ebf5b344706638c648fe8655d5334a53e6b3af30133423"
    )
    assert replay["completion_receipt_sha256"] == (
        "d6f65d6ed83b8d480a89b9bb4044c39e46f46987c4dd283f62e20a4d74aa6eb1"
    )
    assert replay["completion_receipt_size_bytes"] == 13262
    assert replay["surface_native_point_count"] == 50_766_193_080
    assert replay["surface_scored_point_count"] == 50_766_193_080
    assert replay["volume_native_valid_point_count"] == 83_728_136_475
    assert replay["volume_scored_point_count"] == 83_728_136_475
    assert replay["incomplete_support_record_count"] == 0

    assert replay["aggregate_a_logical_tree_sha256"] == replay[
        "aggregate_b_logical_tree_sha256"
    ] == "e44f5fb4cfa68d9dbbd1704c804ed52b220948a84b7f510dcc25208285f8e38e"
    assert replay["receipts_a_logical_tree_sha256"] == replay[
        "receipts_b_logical_tree_sha256"
    ] == "17c549ec4a97fd78c7b3f884b42be84a6c57c998ae5acbad7902151c800ee689"
    assert replay["package_a_logical_tree_sha256"] == replay[
        "package_b_logical_tree_sha256"
    ] == "9c02d241eaf2ea6deecd4181816ed690d664b10f6eea0c2db4e5880aa2b1a0ab"
    assert replay["package_file_count_each"] == 763
    assert replay["package_directory_count_each"] == 364
    assert replay["package_size_bytes_each"] == 2_345_982_957
    assert replay["submission_json_sha256"] == (
        "11b678133ca086639ba444395fff2835559eb259d92974d60aa0b0039c0aef4c"
    )
    assert replay["evaluation_evidence_sha256"] == (
        "2135ca5a18ad93b1dfcb980f4bf4d315bca6231595561240d867779039f2bb36"
    )
    assert replay["case_metrics_sha256"] == (
        "61a97397fe845d1656acdaa2034829084d101134770402de35048e342d1a12dc"
    )
    assert replay["discretization_sha256"] == (
        "073e3bb91091738fa34c435681fe9e83a7492087b938a3924e6b8a89b1c7debc"
    )
    assert replay["discretization_case_records_sha256"] == (
        "3af8ab1321c5f072b3f5bfc10ce837a1a80c4b79d75be5f94cc461cfcbff5abd"
    )
    assert replay["profile_index_sha256"] == (
        "bd842829a02b63d6de7fa06f104dac5c80245b77e877e7f09b44e318aa21dfba"
    )
    assert replay["profile_ground_truth_manifest_sha256"] == (
        "3e20b857e12055e16f1d248d125b8df62a3669c604411dc67322fe98d9ab4477"
    )
    assert replay["scoring_support_manifest_sha256"] == (
        "98a9a8d015e42c80f5993e30da94201011bebf67ff44d5574a7ef68a8b5dfbee"
    )
    assert replay["regional_diagnostics_sha256"] == (
        "1bdde868bde838d9a14fa6267e4b11ae886e9353d2d086c3e05c017756a9e241"
    )
    assert replay["zip_a_sha256"] == replay["zip_b_sha256"] == (
        "a8dfd6ffbe6d103bc1a3123f6e6f756bacf960f92acc7079237b443bbbb62504"
    )
    assert replay["zip_size_bytes_each"] == 2_264_458_600
    assert replay["zip_member_count_each"] == 763
    assert replay["candidate_validation_pass_count"] == 2
    assert replay["computed_scores"] == {
        "overall_score": 69.15276784043088,
        "field_score": 48.167429082451235,
        "force_score": 99.52105782630258,
        "diagnostic_score": 80.75515537051848,
        "cp_cut_r2": 0.9636931485730299,
        "velocity_profile_r2": 0.7034571571266213,
    }
    assert all(replay["checks"].values())
    assert replay["lifecycle"] == {
        "owner_scientific_approval": False,
        "candidate_activated": False,
        "public_profile_truth_published": False,
        "submissions_opened": False,
        "official_submission": False,
        "uploaded": False,
        "published": False,
        "public_leaderboard_entry_created": False,
        "private_leaderboard_created": False,
    }
    assert replay["acceptable_as_final_package_completion_evidence"] is True
    assert replay["acceptable_as_owner_approval_or_activation_evidence"] is False
    serialized = json.dumps(replay, sort_keys=True)
    assert not any(prefix in serialized for prefix in ("/lustre/", "/home/", "/root/"))
