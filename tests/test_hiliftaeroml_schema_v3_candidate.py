from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from reference.hiliftaeroml.regional_aggregate import (
    HiLiftRegionalAggregateError,
    validate_aggregate_regional_diagnostics,
)
from reference.scoring_support import load_support_release
from scripts import assemble_hiliftaeroml_schema_v3_candidate as assembler
from scripts.prepare_hiliftaeroml_split_indices import SPLIT_BINDINGS
from scripts.validate_scoring_supports import validate_candidate_manifest_release


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-specs" / "hiliftaeroml"
SPEC = DATASET / "submission-spec.json"
PROFILE_FORMAT = DATASET / "native-profile-format-v1.json"
LEADERBOARD_MANIFEST = ROOT / "leaderboard" / "manifest.json"
SUBMISSION_SCHEMA = ROOT / "schemas" / "v3" / "submission.schema.json"
FULL_SPLIT = DATASET / "splits" / "full.json"
RELEASE = (
    DATASET
    / "scoring-support"
    / "hiliftaeroml-native-all-splits-support-v1-candidate"
    / "manifest.json"
)
CANDIDATE_EVALUATOR_BINDING = DATASET / "candidate-evaluator-release-binding.json"
CONCRETE_CONFIG = (
    ROOT
    / "examples"
    / "hiliftaeroml-v3-candidate"
    / "transolver-full360-candidate-config.json"
)
CASE_SET_ID = "caseset-ac791749e527"
CASE_SET_SHA256 = "ac791749e5279ecf6746fcce20e3ec32408fd33b22127d5270de968be7842acf"
REGION_SHA256 = "8cf926d06706b8cc7fd58d821f395bf8cb6565ef1f3aabe6be18e0a279101da4"
EVALUATOR_REVISION = "68899f780d96b70f2badb5658971c87af0b17172"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_strings(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)
    elif isinstance(value, str):
        yield value


def test_full_split_is_the_frozen_actual_order() -> None:
    split = load(FULL_SPLIT)
    case_ids = split["case_ids"]
    assert split["case_set_id"] == CASE_SET_ID
    assert split["case_id_status"] == "official"
    assert split["case_count"] == len(case_ids) == len(set(case_ids)) == 360
    assert hashlib.sha256(("\n".join(case_ids) + "\n").encode()).hexdigest() == CASE_SET_SHA256
    assert split["case_set_sha256"] == CASE_SET_SHA256


def test_candidate_release_is_closed_complete_and_semantically_valid() -> None:
    specification = load(SPEC)
    manifest = load(RELEASE)
    support = specification["scoring_support"]
    assert specification["status"] == "owner_review_required"
    assert support["status"] == "owner_review_required"
    assert support["submissions_open"] is False
    assert "owner_approval" not in support
    assert support["candidate_manifest"]["status"] == "candidate"
    assert support["candidate_manifest"]["release_id"] == (
        "hiliftaeroml-native-all-splits-support-v1-candidate"
    )
    assert support["candidate_manifest"]["manifest_file"] == (
        "scoring-support/hiliftaeroml-native-all-splits-support-v1-candidate/"
        "manifest.json"
    )
    assert support["candidate_manifest"]["manifest_url"].endswith(
        "/hiliftaeroml-native-all-splits-support-v1-candidate/manifest.json"
    )
    assert support["candidate_manifest"]["manifest_sha256"] == digest(RELEASE)
    assert manifest["status"] == "candidate"
    assert "owner_approval" not in manifest
    assert validate_candidate_manifest_release(
        specification,
        DATASET,
        RELEASE,
        manifest,
    ) == []

    expected_case_sets: dict[str, list[str]] = {}
    for binding in SPLIT_BINDINGS:
        split = load(DATASET / "splits" / f"{binding.split_id}.json")
        case_ids = split["case_ids"]
        assert split["case_set_id"] == binding.case_set_id
        assert split["case_count"] == len(case_ids) == binding.case_count
        assert split["case_set_sha256"] == binding.case_set_sha256
        previous = expected_case_sets.setdefault(binding.case_set_id, case_ids)
        assert previous == case_ids

    assert len(expected_case_sets) == 8
    assert [item["id"] for item in manifest["case_sets"]] == list(expected_case_sets)
    for descriptor in manifest["case_sets"]:
        case_set_id = descriptor["id"]
        case_ids = expected_case_sets[case_set_id]
        assert descriptor["case_count"] == len(case_ids)
        loaded = load_support_release(RELEASE, case_set_id)
        assert list(loaded.cases) == case_ids
        assert set(loaded.supports) == {
            "surface-native-points-v1",
            "volume-native-valid-points-v1",
            "aerodynamic-case-coefficients-v1",
        }


def test_frozen_evaluator_revision_is_consistent_and_nonactivating() -> None:
    specification = load(SPEC)
    manifest = load(LEADERBOARD_MANIFEST)
    binding = load(CANDIDATE_EVALUATOR_BINDING)
    config = load(CONCRETE_CONFIG)
    dataset = next(
        item for item in manifest["datasets"] if item["slug"] == "hiliftaeroml"
    )
    preview_rows = load(ROOT / "leaderboard" / "datasets" / "hiliftaeroml.json")
    support = specification["scoring_support"]
    evaluator = support["dataset_evaluator_binding"]

    assert dataset["scoring_support"] == support
    assert len(preview_rows) == 23
    assert dataset["submission_count"] == len(preview_rows)
    assert dataset["revision_count"] == len(preview_rows)
    assert dataset["updated_at"] == max(
        row["submitted_at"] for row in preview_rows
    )
    assert evaluator["status"] == "frozen"
    assert evaluator["evaluator_reference_version"] == specification[
        "evaluation_reference_version"
    ]
    assert evaluator["evaluator_code_revision"] == EVALUATOR_REVISION
    assert config["release_bindings"]["evaluator"] == {
        "reference_version": specification["evaluation_reference_version"],
        "code_revision": EVALUATOR_REVISION,
    }
    assert binding["repositories"]["fluidsbench_adapter"][
        "immutable_evaluator_code_revision"
    ] == EVALUATOR_REVISION
    assert binding["status"] == (
        "candidate_full360_replay_complete_owner_approval_pending"
    )
    assert binding["activation_effect"] == "none"
    assert binding["activation_gates"]["immutable_evaluator_revision_bound"] is True
    assert binding["activation_gates"][
        "full_360_force_and_overall_replay_complete"
    ] is True
    assert binding["activation_gates"]["owner_scientific_approval"] is False
    assert binding["activation_gates"]["public_profile_truth_published"] is False
    assert binding["activation_gates"]["submissions_open"] is False
    assert specification["status"] == "owner_review_required"
    assert support["status"] == "owner_review_required"
    assert support["submissions_open"] is False
    assert specification["profile_definition"]["profile_ground_truth"] == {
        "status": "not_published",
        "release_id": None,
        "manifest_sha256": None,
    }
    decisions = support["owner_decisions_required"]
    assert (
        "owner_approve_the_frozen_dataset_evaluator_git_revision_without_"
        "public_activation"
    ) in decisions
    assert "freeze_and_approve_the_immutable_dataset_evaluator_git_revision" not in decisions


def test_compact_command_and_evidence_are_additive_to_native_v1() -> None:
    config = load(CONCRETE_CONFIG)
    assembler._validate_config_envelope(config)
    native = assembler._selected_evaluation(
        config, compact_profile_mode=False
    )
    compact = assembler._selected_evaluation(
        config, compact_profile_mode=True
    )
    assert "--candidate-profile-truth-release" in native["command"]
    assert "--candidate-compact-profile-support-release" not in native["command"]
    assert "--candidate-compact-profile-support-release" in compact["command"]
    assert "--candidate-profile-truth-release" not in compact["command"]
    assert native != compact

    revision = config["release_bindings"]["evaluator"]["code_revision"]
    native_notes = assembler._profile_evidence_notes(
        evaluator_revision=revision, compact_profile_mode=False
    )
    assert native_notes == (
        "Evaluator identity is repository-frozen at "
        f"{revision}; profile topology contract is "
        f"{assembler.PROFILE_CONTRACT_SHA256}. Cp/velocity profile R2 was "
        "recomputed from prediction-only chunks against the explicitly supplied "
        "inactive local candidate truth release; this does not publish or "
        "activate profile intake."
    )
    compact_notes = assembler._profile_evidence_notes(
        evaluator_revision=revision, compact_profile_mode=True
    )
    assert assembler.COMPACT_PROFILE_CONTRACT_SHA256 in compact_notes
    assert "unbound worktree candidate" in compact_notes
    assert "no code revision or implementation-manifest SHA-256" in compact_notes
    assert (
        assembler.COMPACT_PROFILE_IMPLEMENTATION_BINDING
        == {
            "status": "unbound_worktree_candidate",
            "activation_effect": "none",
            "code_revision": None,
            "implementation_manifest_sha256": None,
            "base_dataset_evaluator_scope": (
                "native_v1_base_field_force_and_noncompact_scoring_only"
            ),
        }
    )


def test_primary_l2_policy_and_activation_gate_are_explicit() -> None:
    specification = load(SPEC)
    manifest = load(RELEASE)
    supports = {support["id"]: support for support in manifest["supports"]}
    surface = supports["surface-native-points-v1"]
    volume = supports["volume-native-valid-points-v1"]
    surface_bindings = {
        binding["metric_id"]: binding for binding in surface["metric_bindings"]
    }
    volume_bindings = {
        binding["metric_id"]: binding for binding in volume["metric_bindings"]
    }
    assert surface_bindings["surface_pressure_rel_l2"]["weighting"] == "support_weights"
    assert surface_bindings["surface_pressure_rel_l2"]["quantity_id"] == "pressure_coefficient"
    assert surface_bindings["surface_wall_shear_rel_l2"]["weighting"] == "support_weights"
    assert surface_bindings["surface_wall_shear_rel_l2"]["quantity_id"] == "wall_shear_coefficient"
    assert surface_bindings["surface_pressure_equal_entity_rel_l2"]["weighting"] == "uniform"
    assert volume_bindings["volume_pressure_rel_l2"]["weighting"] == "uniform"
    assert volume_bindings["volume_pressure_rel_l2"]["quantity_id"] == "pressure_coefficient"
    assert volume_bindings["volume_velocity_rel_l2"]["weighting"] == "uniform"
    assert volume_bindings["volume_velocity_rel_l2"]["quantity_id"] == "velocity_over_u_inf"
    assert surface_bindings["surface_pressure_mae"]["quantity_id"] == "pressure"
    assert surface_bindings["surface_wall_shear_rmse"]["quantity_id"] == "wall_shear_stress"
    assert volume_bindings["volume_pressure_mae"]["quantity_id"] == "pressure"
    assert volume_bindings["volume_velocity_rmse"]["quantity_id"] == "velocity"
    assert "volume_pressure_physical_rel_l2" not in volume_bindings
    assert "volume_velocity_physical_rel_l2" not in volume_bindings
    assert not {
        "volume_pressure_physical_rel_l2",
        "volume_velocity_physical_rel_l2",
    }.intersection(metric["id"] for metric in specification["metrics"])
    assert "not_part_of_closed_candidate" == specification["scoring_support"][
        "relative_l2_policy"
    ]["flow_domain_secondary"]
    assert "not an activated public scoring change" in specification[
        "scoring_support"
    ]["candidate_migration_notes"][0]
    bases = specification["scoring_support"]["field_metric_value_bases"]
    assert bases["relative_l1_l2"]["surface_pressure"] == "(P-p_inf)/q_inf"
    assert bases["relative_l1_l2"]["volume_velocity"] == "U/|U_inf|"
    assert "q_inf into Pa" in bases["dimensional_mae_rmse"]["surface_pressure"]
    assert "|U_inf| into m/s" in bases["dimensional_mae_rmse"]["volume_velocity"]
    assert "three-component" in bases["vector_reduction"]["relative_l2"]
    assert "per-component" in bases["vector_reduction"]["mae_rmse"]
    assert "selected split" in bases["relative_l1_l2"]["aggregation"]
    assert "selected split" in bases["dimensional_mae_rmse"]["aggregation"]
    assert "360" not in bases["relative_l1_l2"]["aggregation"]
    assert "360" not in bases["dimensional_mae_rmse"]["aggregation"]
    assert "additive sums only" in surface["notes"]
    assert "No epsilon" in surface["notes"]
    assert "additive sums only" in volume["notes"]
    assert "No epsilon" in volume["notes"]
    assert any(
        "content_sha256" in gate
        for gate in specification["scoring_support"]["owner_decisions_required"]
    )
    assert any(
        "all_1355_unique_evaluation_cases" in gate
        for gate in specification["scoring_support"]["owner_decisions_required"]
    )


def test_profile_contract_matches_the_frozen_table5_evaluator() -> None:
    specification = load(SPEC)
    panels = {panel["id"]: panel for panel in specification["profile_panels"]}
    pressure = panels["pressure_profiles"]
    velocity = panels["velocity_profiles"]
    assert pressure["source_station_map"] == {
        f"pressure_belt_{letter.lower()}": letter for letter in "ABCDEFGHIJ"
    }
    assert pressure["weighting"] == "physical_connected_cut_arc_length"
    assert "independently within each physical connected cut graph" in pressure[
        "r2_protocol"
    ]
    assert velocity["source_station_map"] == {
        "hlpw5_b_2": "B.2",
        "hlpw5_b_3": "B.3",
        "hlpw5_c_1": "C.1",
        "hlpw5_c_2": "C.2",
        "hlpw5_c_3": "C.3",
    }
    assert velocity["weighting"] == "physical_polyline_arc_length"
    assert velocity["quantity_source"] == "velocity_magnitude/|U_inf|"
    assert "center truth within each" in velocity["r2_protocol"]
    metrics = {metric["id"]: metric for metric in specification["metrics"]}
    assert metrics["cp_cut_r2"]["aggregation"] == (
        "per_case_connected_graph_centered_then_macro_average"
    )
    assert metrics["velocity_profile_r2"]["aggregation"] == (
        "per_case_station_centered_then_macro_average"
    )
    notes = specification["scoring_support"]["candidate_migration_notes"]
    assert any("stale prototype 16-station" in note for note in notes)


def test_leaderboard_declares_schema_v3_and_only_the_frozen_velocity_stations() -> None:
    specification = load(SPEC)
    manifest = load(LEADERBOARD_MANIFEST)
    submission_schema = load(SUBMISSION_SCHEMA)
    profile_format = load(PROFILE_FORMAT)
    dataset = next(
        item for item in manifest["datasets"] if item["slug"] == "hiliftaeroml"
    )
    velocity_panel = next(
        panel
        for panel in dataset["diagnostic_panels"]
        if panel["id"] == "velocity_profiles"
    )
    specification_panel = next(
        panel
        for panel in specification["profile_panels"]
        if panel["id"] == "velocity_profiles"
    )

    assert assembler.SUBMISSION_FORMAT == "hiliftaeroml_native_candidate_v3"
    assert specification["submission_format"] == assembler.SUBMISSION_FORMAT
    assert dataset["submission_format"] == assembler.SUBMISSION_FORMAT
    assert submission_schema["$id"].endswith("/schemas/v3/submission.schema.json")
    assert submission_schema["properties"]["schema_version"]["const"] == "3.0"

    expected_aliases = ["B.2", "B.3", "C.1", "C.2", "C.3"]
    expected_station_ids = list(specification_panel["source_station_map"])
    assert list(specification_panel["source_station_map"].values()) == expected_aliases
    assert profile_format["volume_velocity"]["station_order"] == expected_aliases
    assert [station["id"] for station in velocity_panel["stations"]] == (
        expected_station_ids
    )
    assert [station["label"].removeprefix("HLPW-5 ") for station in velocity_panel["stations"]] == (
        expected_aliases
    )


def test_candidate_provenance_is_sanitized_and_not_a_source_pin() -> None:
    manifest = load(RELEASE)
    forbidden_prefixes = ("/lustre/", "/home/", "/root/")
    checked_paths = [RELEASE]
    checked_case_ids: set[str] = set()
    for descriptor in manifest["case_sets"]:
        index_path = RELEASE.parent / descriptor["index_file"]
        index = load(index_path)
        assert index["case_count"] == descriptor["case_count"]
        assert digest(index_path) == descriptor["index_sha256"]
        chunk_paths = [index_path.parent / chunk["file"] for chunk in index["chunks"]]
        observed_case_ids = []
        for chunk_descriptor, chunk_path in zip(index["chunks"], chunk_paths):
            assert digest(chunk_path) == chunk_descriptor["sha256"]
            chunk = load(chunk_path)
            chunk_case_ids = [case["case_id"] for case in chunk["cases"]]
            assert chunk_case_ids == chunk_descriptor["case_ids"]
            observed_case_ids.extend(chunk_case_ids)
            for case in chunk["cases"]:
                checked_case_ids.add(case["case_id"])
                for instance in case["support_instances"]:
                    parameters = instance.get("parameters", {})
                    source = parameters.get("source_identity")
                    if source is not None:
                        assert source["content_sha256_computed"] is False
                        assert "path" not in source
        assert len(observed_case_ids) == descriptor["case_count"]
        provenance_path = index_path.parent / "provenance.json"
        provenance = load(provenance_path)
        assert provenance["case_set_id"] == descriptor["id"]
        assert provenance["case_count"] == descriptor["case_count"]
        checked_paths.extend((index_path, *chunk_paths, provenance_path))
    assert len(checked_case_ids) == 1355
    for path in checked_paths:
        assert not any(
            value.startswith(forbidden_prefixes)
            for value in walk_strings(load(path))
        )


def test_regional_contract_is_exactly_bound_and_zero_weight() -> None:
    specification = load(SPEC)
    declaration = specification["regional_diagnostics"]
    contract = DATASET / declaration["contract_file"]
    assert digest(contract) == declaration["contract_sha256"] == REGION_SHA256
    assert declaration["format"] == "hiliftaeroml-regional-diagnostics-aggregate-v2"
    assert declaration["definition_id"] == "hiliftaeroml-native-geometric-regions-v1"
    assert declaration["role"] == "report_only"
    assert declaration["weight"] == 0.0
    assert declaration["affects_official_metrics"] is False
    assert declaration["affects_official_score"] is False
    assert declaration["required_for_new_submissions"] is False


def regional_field(region_ids: tuple[str, ...]) -> dict:
    case_macro = {
        metric_id: {"value": value, "defined_case_count": 2}
        for metric_id, value in {
            "whole_support_normalized_rmse_percent": 1.5,
            "relative_l2_percent": 2.0,
            "r2": 0.8,
            "mae": 0.1,
            "rmse": 0.2,
        }.items()
    }
    case_distribution = {
        metric_id: {
            "defined_case_count": 2,
            "minimum": minimum,
            "median": median,
            "p90": maximum,
            "maximum": maximum,
        }
        for metric_id, (minimum, median, maximum) in {
            "whole_support_normalized_rmse_percent": (1.0, 1.5, 2.0),
            "relative_l2_percent": (1.0, 2.0, 3.0),
            "r2": (0.6, 0.8, 0.9),
        }.items()
    }
    return {
        "global": {"relative_l2_percent": 2.0},
        "regions": [
            {
                "region_id": region_id,
                "entity_fraction": 0.25,
                "weight_fraction": 0.25,
                "squared_error_fraction": 0.25,
                "whole_support_normalized_rmse_percent": 2.0,
                "relative_l2_percent": 2.0,
                "r2": 0.8,
                "r2_status": "ok",
                "mae": 0.1,
                "rmse": 0.2,
                "case_macro": case_macro,
                "case_distribution": case_distribution,
            }
            for region_id in region_ids
        ],
        "pooled": {"relative_l2_percent": 2.0},
        "macro": {"relative_l2_percent": 2.0},
        "case_distribution": {"minimum": 1.0, "median": 2.0, "p90": 3.0, "maximum": 4.0},
    }


def regional_report(case_ids: list[str]) -> dict:
    surface_regions = (
        "nacelle_installation_envelope_proxy",
        "inboard_high_lift_envelope_proxy",
        "outboard_high_lift_envelope_proxy",
        "fuselage_tail_and_remaining",
    )
    volume_regions = (
        "near_airframe_sdf_band",
        "aft_airframe_wake_envelope_proxy",
        "near_aircraft_flow_envelope",
        "farfield_and_remaining",
    )
    return {
        "schema": "hiliftaeroml-regional-diagnostics-aggregate-v2",
        "schema_version": 2,
        "status": "complete_report_only",
        "definition_id": "hiliftaeroml-native-geometric-regions-v1",
        "contract_sha256": REGION_SHA256,
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "prediction_scope": "surface_and_volume",
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "scoring": {
            "role": "report_only",
            "weight": 0.0,
            "official_metric_inputs_changed": False,
            "official_score_changed": False,
        },
        "surface": {
            "support_id": "surface_native_points",
            "region_order": list(surface_regions),
            "fields": {
                "pressure": regional_field(surface_regions),
                "tau_wall": regional_field(surface_regions),
            },
        },
        "volume": {
            "support_id": "volume_native_valid_points",
            "region_order": list(volume_regions),
            "fields": {
                "pressure": regional_field(volume_regions),
                "velocity": regional_field(volume_regions),
            },
        },
        "reconstruction": {
            "status": "pass",
            "relative_tolerance": 5.0e-12,
            "absolute_tolerance": 1.0e-12,
            "checked_case_count": len(case_ids),
            "checked_field_count": 4,
        },
    }


def test_hilift_regional_aggregate_validator_is_dataset_specific_and_fail_closed() -> None:
    case_ids = ["geo_LHC001_AoA_20", "geo_LHC039_AoA_4"]
    report = regional_report(case_ids)
    validate_aggregate_regional_diagnostics(
        report,
        expected_case_ids=case_ids,
        expected_split_id="full",
    )
    broken = copy.deepcopy(report)
    broken["volume"]["fields"]["velocity"]["regions"][0][
        "entity_fraction"
    ] = 0.30
    with pytest.raises(HiLiftRegionalAggregateError, match="does not sum to one"):
        validate_aggregate_regional_diagnostics(
            broken,
            expected_case_ids=case_ids,
            expected_split_id="full",
        )
    broken = copy.deepcopy(report)
    broken["volume"]["fields"]["pressure"]["regions"][0][
        "whole_support_normalized_rmse_percent"
    ] = 3.0
    with pytest.raises(
        HiLiftRegionalAggregateError,
        match="does not reconstruct global relative L2",
    ):
        validate_aggregate_regional_diagnostics(
            broken,
            expected_case_ids=case_ids,
            expected_split_id="full",
        )


def test_whole_support_regional_normalization_uses_global_truth_rms() -> None:
    value = assembler._whole_support_normalized_rmse_percent(
        {"weight_sum": 4.0, "sum_squared_error": 4.0},
        {"weight_sum": 25.0, "sum_squared_truth": 100.0},
        label="test pressure region",
    )
    assert value == pytest.approx(50.0)
    assert assembler._whole_support_normalized_rmse_percent(
        {"weight_sum": 0.0, "sum_squared_error": 0.0},
        {"weight_sum": 25.0, "sum_squared_truth": 100.0},
        label="empty region",
    ) is None


def test_regional_case_summaries_retain_defined_counts_and_negative_r2() -> None:
    assert assembler._regional_case_macro([1.0, None, 3.0]) == {
        "value": 2.0,
        "defined_case_count": 2,
    }
    assert assembler._regional_case_distribution([-2.0, None, 0.5]) == {
        "defined_case_count": 2,
        "minimum": -2.0,
        "median": -0.75,
        "p90": 0.5,
        "maximum": 0.5,
    }
