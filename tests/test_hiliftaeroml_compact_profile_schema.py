from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "fluidsbench-hiliftaeroml-compact-profile-chunks-v2-candidate"
CONTRACT_ID = "hiliftaeroml-compact-profile-predictions-v2-candidate"
CASE_ID = "geo_LHC001_AoA_4"
IMPLEMENTATION_BINDING = {
    "status": "unbound_worktree_candidate",
    "activation_effect": "none",
    "code_revision": None,
    "implementation_manifest_sha256": None,
    "base_dataset_evaluator_scope": (
        "native_v1_base_field_force_and_noncompact_scoring_only"
    ),
}


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def valid_chunk() -> dict:
    return {
        "schema": "hiliftaeroml-compact-profile-chunk-v2-candidate",
        "schema_version": "2.0",
        "format": FORMAT,
        "contract_id": CONTRACT_ID,
        "contract_sha256": "a" * 64,
        "submission_id": "hilift-compact-test",
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "case_set_id": "caseset-test",
        "cases": [
            {
                "case_id": CASE_ID,
                "artifact": {
                    "format": "numpy-npz-v1",
                    "file": f"artifacts/{CASE_ID}/compact-profile-predictions.npz",
                    "sha256": "b" * 64,
                    "byte_size": 1234,
                    "array_order": [
                        "cp_q_delta",
                        "velocity_speed_over_u_inf",
                    ],
                    "ownership": "participant",
                    "content": "predictions_only",
                },
                "surface_cp": {
                    "support_identity_sha256": "c" * 64,
                    "prediction_order_sha256": "d" * 64,
                    "physical_graph_count": 37,
                    "retained_branch_count": 52,
                    "retained_point_count": 4096,
                    "maximum_points_per_physical_graph": 128,
                    "quantization_scale": 1024,
                    "quantization_dtype": "int16",
                    "delta_dtype": "int16",
                    "prediction_array": "cp_q_delta",
                },
                "volume_velocity": {
                    "support_identity_sha256": "e" * 64,
                    "prediction_order_sha256": "f" * 64,
                    "station_order": ["B.2", "B.3", "C.1", "C.2", "C.3"],
                    "station_count": 5,
                    "row_count": 4005,
                    "valid_row_count": 3900,
                    "invalid_row_count": 105,
                    "prediction_dtype": "float32",
                    "prediction_array": "velocity_speed_over_u_inf",
                    "storage_dtype": "uint8",
                    "storage_encoding": (
                        "little_endian_float32_bits_unsigned_delta_modulo_"
                        "2pow32_byte_shuffle_v1"
                    ),
                    "stored_byte_count": 15600,
                },
            }
        ],
    }


def test_compact_profile_contract_is_additive_and_excludes_full_surface_l2() -> None:
    contract = load("benchmark-specs/hiliftaeroml/native-profile-format-v2.json")
    assert contract["format"] == FORMAT
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["status"] == "additive_candidate_not_bound"
    assert contract["backward_compatibility"]["native_profile_v1_is_unchanged"]
    full_surface = contract["scope"]["full_surface_cp_dual_area_l2"]
    assert full_surface == {
        "metric_id": "surface_pressure_rel_l2",
        "status": "unchanged_and_out_of_scope",
        "support": "all canonical native surface points",
        "weighting": "surface_point_dual_area",
        "profile_payload_used": False,
    }
    assert contract["evaluator_owned_support"]["included_in_participant_artifact"] is False
    assert contract["container"]["compression"] == "zip_deflate_method_8_level_9"
    identities = contract["evaluator_owned_support"]["identity_encoding"]
    assert identities["truth_arrays_included"] is False
    assert identities["volume_velocity_prediction_order_array_order"] == [
        "velocity_station_names",
        "velocity_station_row_offsets",
        "velocity_valid_row_indices",
    ]
    assert contract["surface_cp"]["retained_support"][
        "maximum_points_per_physical_graph"
    ] == 128
    assert contract["surface_cp"]["quantization"]["formula"] == (
        "q=int16(round(Cp*1024))"
    )
    velocity = contract["volume_velocity"]["prediction_array"]
    assert velocity["logical_dtype"] == "float32"
    assert velocity["storage"] == {
        **velocity["storage"],
        "dtype": "uint8",
        "shape": ["4*velocity_valid_row_count"],
        "encoding": (
            "little_endian_float32_bits_unsigned_delta_modulo_2pow32_"
            "byte_shuffle_v1"
        ),
        "lossless": True,
        "browser_native_container_decompression": True,
    }
    counts = contract["case_metadata"]["count_semantics"]
    assert counts["row_count"].startswith("4005 canonical source rows")
    assert counts["valid_row_count"].endswith(
        "decoded logical length of velocity_speed_over_u_inf"
    )
    assert counts["stored_byte_count"].startswith("4*valid_row_count")


def test_compact_profile_chunk_and_generic_index_are_schema_valid() -> None:
    chunk = valid_chunk()
    chunk_schema = load("schemas/v1/hiliftaeroml-compact-profile-chunk.schema.json")
    assert list(Draft202012Validator(chunk_schema).iter_errors(chunk)) == []

    index = {
        "schema_version": "1.0",
        "format": FORMAT,
        "contract_id": CONTRACT_ID,
        "contract_sha256": "a" * 64,
        "evaluator_support_release_id": "compact-support-v2-candidate",
        "evaluator_support_manifest_sha256": "3" * 64,
        "submission_id": "hilift-compact-test",
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "case_set_id": "caseset-test",
        "case_count": 1,
        "case_id_status": "official",
        "chunks": [
            {
                "file": "chunk-000.json",
                "case_ids": [CASE_ID],
                "sha256": "1" * 64,
            }
        ],
    }
    index_schema = load("schemas/v1/profile-index.schema.json")
    assert list(Draft202012Validator(index_schema).iter_errors(index)) == []

    submission_schema = load("schemas/v3/submission.schema.json")
    assert FORMAT in submission_schema["$defs"]["profile_data"]["properties"][
        "format"
    ]["enum"]

    compact_profile_data = {
        "format": FORMAT,
        "index_file": "profiles/index.json",
        "case_count": 1,
        "case_set_id": "caseset-test",
        "profile_ground_truth_release_id": "candidate-truth-v1",
        "profile_ground_truth_manifest_sha256": "2" * 64,
        "evaluator_support_release_id": "compact-support-v2-candidate",
        "evaluator_support_manifest_sha256": "3" * 64,
        "compact_profile_implementation_binding": IMPLEMENTATION_BINDING,
    }
    profile_fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": submission_schema["$defs"],
        "$ref": "#/$defs/profile_data",
    }
    profile_validator = Draft202012Validator(profile_fragment)
    assert list(profile_validator.iter_errors(compact_profile_data)) == []

    evidence_schema = load("schemas/v3/evaluation-evidence.schema.json")
    binding_fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": evidence_schema["$defs"],
        "$ref": "#/$defs/compact_profile_implementation_binding",
    }
    binding_validator = Draft202012Validator(binding_fragment)
    assert list(binding_validator.iter_errors(IMPLEMENTATION_BINDING)) == []
    extra_binding = {**IMPLEMENTATION_BINDING, "source": "unbound"}
    assert list(binding_validator.iter_errors(extra_binding))

    missing_support = copy.deepcopy(compact_profile_data)
    del missing_support["evaluator_support_manifest_sha256"]
    assert list(profile_validator.iter_errors(missing_support))

    missing_implementation = copy.deepcopy(compact_profile_data)
    del missing_implementation["compact_profile_implementation_binding"]
    assert list(profile_validator.iter_errors(missing_implementation))

    falsely_frozen = copy.deepcopy(compact_profile_data)
    falsely_frozen["compact_profile_implementation_binding"]["status"] = "frozen"
    falsely_frozen["compact_profile_implementation_binding"]["code_revision"] = (
        "a" * 40
    )
    assert list(profile_validator.iter_errors(falsely_frozen))

    native_with_compact_support = copy.deepcopy(compact_profile_data)
    native_with_compact_support["format"] = (
        "fluidsbench-hiliftaeroml-native-profile-chunks-v1-candidate"
    )
    assert list(profile_validator.iter_errors(native_with_compact_support))


def test_compact_profile_chunk_rejects_support_or_truth_metadata_and_wrong_encoding() -> None:
    schema = load("schemas/v1/hiliftaeroml-compact-profile-chunk.schema.json")
    validator = Draft202012Validator(schema)

    extra_source = valid_chunk()
    extra_source["cases"][0]["artifact"]["source_native_npz_sha256"] = "0" * 64
    assert list(validator.iter_errors(extra_source))

    truth_member = valid_chunk()
    truth_member["cases"][0]["artifact"]["array_order"].append("truth_cp")
    assert list(validator.iter_errors(truth_member))

    wrong_budget = valid_chunk()
    wrong_budget["cases"][0]["surface_cp"][
        "maximum_points_per_physical_graph"
    ] = 127
    assert list(validator.iter_errors(wrong_budget))

    wrong_velocity_dtype = copy.deepcopy(valid_chunk())
    wrong_velocity_dtype["cases"][0]["volume_velocity"][
        "prediction_dtype"
    ] = "float64"
    assert list(validator.iter_errors(wrong_velocity_dtype))
