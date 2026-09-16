from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import scripts.build_hiliftaeroml_implementation_manifest as builder


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / builder.DEFAULT_OUTPUT_RELATIVE
SCHEMA_PATH = ROOT / builder.MANIFEST_SCHEMA_RELATIVE
LOCK_PATH = ROOT / builder.RUNTIME_LOCK_RELATIVE
FROZEN_MANIFEST_SHA256 = (
    "f94cef375b45c8604bbd65bb59c679af417340b39dd941c27beb363d59f883d6"
)
FROZEN_REGIONAL_DECLARATION = {
    "format": "hiliftaeroml-regional-diagnostics-aggregate-v1",
    "contract_file": "regional-diagnostics-v1.json",
    "contract_sha256": (
        "1579b0262f3368fe3748eb53025aa5e46c0a32c8ff1616c9becdbb5dedd85650"
    ),
}


@pytest.fixture(scope="module")
def retained_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def group_paths(manifest: dict, group_id: str) -> set[str]:
    group = next(
        group for group in manifest["source_groups"] if group["group_id"] == group_id
    )
    return {item["path"] for item in group["files"]}


def group_identity(manifest: dict, group_id: str, path: str) -> dict:
    group = next(
        group for group in manifest["source_groups"] if group["group_id"] == group_id
    )
    return next(item for item in group["files"] if item["path"] == path)


def remove_postfreeze_additive_extensions(specification: dict) -> None:
    compact_definition = specification.pop("compact_profile_definition")
    assert compact_definition["status"] == "additive_candidate_not_bound"
    regional = specification["regional_diagnostics"]
    assert regional["role"] == "report_only"
    assert regional["weight"] == 0.0
    for key, value in FROZEN_REGIONAL_DECLARATION.items():
        regional[key] = value


def test_retained_manifest_is_schema_valid_and_byte_deterministic(
    retained_manifest,
) -> None:
    observed = retained_manifest
    payload = MANIFEST_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FROZEN_MANIFEST_SHA256
    assert payload == builder.pretty_bytes(observed)
    assert builder.pretty_bytes(observed) == builder.pretty_bytes(observed)
    builder.verify_fingerprint(observed)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(
            observed
        ),
        key=lambda error: list(error.absolute_path),
    )
    assert [error.message for error in errors] == []
    assert observed["source_group_count"] == len(observed["source_groups"]) == 10
    assert observed["source_file_count"] == sum(
        group["file_count"] for group in observed["source_groups"]
    )
    assert observed["source_identity_policy"]["normalized_json_exception_count"] == 2
    identities = [
        item for group in observed["source_groups"] for item in group["files"]
    ]
    assert {
        item["path"]
        for item in identities
        if item["identity_mode"]
        == "canonical_json_sha256_excluding_lifecycle_v1"
    } == {
        "benchmark-specs/hiliftaeroml/submission-spec.json",
        "leaderboard/manifest.json",
    }
    assert all(
        item["identity_mode"]
        in {
            "raw_bytes_sha256",
            "canonical_json_sha256_excluding_lifecycle_v1",
        }
        for item in identities
    )


def test_inventory_covers_native_and_fluidsbench_execution_surfaces(
    retained_manifest,
) -> None:
    manifest = retained_manifest
    native = group_paths(manifest, "native_submission_execution")
    assert {
        "inference/exact_surface_loads.py",
        "inference/infer_surface_submission.py",
        "inference/infer_volume_submission.py",
        "inference/recover_geo_LHC039_AoA_10_surface.py",
        "inference/submission_streaming.py",
        "contract/geo_LHC039_AoA_10_normal_underflow_repair_v1.json",
        "audits/geo_LHC039_AoA_10_normal_direction_comparison_v1.json",
        "tools/aggregate_case_set.py",
        "tools/aggregate_full360.py",
        "tools/build_exact_pitch_moment_weights.py",
        "tools/postprocess_full360_candidate.py",
    } <= native
    recipe = group_paths(manifest, "native_recipe_dependencies")
    assert {
        "src/infer_native_surface.py",
        "src/infer_native_velocity_profiles.py",
        "src/infer_native_volume_fields_profiles.py",
        "src/native_cp_profile_metrics.py",
        "src/native_velocity_profile_stencils.py",
        "src/weighted_metrics.py",
        "conf/model/transolver_surface.yaml",
        "conf/model/transolver_volume.yaml",
    } <= recipe

    framework = group_paths(manifest, "physicsnemo_host_framework")
    assert {
        "physicsnemo/models/transolver/transolver.py",
        "physicsnemo/utils/checkpoint.py",
        "physicsnemo/nn/functional/geometry/sdf.py",
        "physicsnemo/mesh/mesh.py",
    } <= framework
    assert len(framework) >= 700

    adapter = group_paths(manifest, "fluidsbench_assembler_scorer_validator")
    assert adapter == {
        "scripts/assemble_hiliftaeroml_schema_v3_candidate.py",
        "scripts/build_hiliftaeroml_submission_zip.py",
        "scripts/validate_scoring_supports.py",
        "scripts/validate_submission.py",
    }
    references = group_paths(manifest, "fluidsbench_scoring_references")
    assert {
        "reference/scores.py",
        "reference/scoring_support.py",
        "reference/hiliftaeroml/native_profile_evaluator.py",
        "reference/hiliftaeroml/native_profiles.py",
        "reference/hiliftaeroml/regional_aggregate.py",
    } <= references


def test_result_specific_example_config_is_transparently_excluded(
    retained_manifest,
) -> None:
    manifest = retained_manifest
    exclusions = {
        row.get("path"): row.get("reason")
        for row in manifest["excluded_by_design"]
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    path = (
        "examples/hiliftaeroml-v3-candidate/"
        "transolver-full360-candidate-config.json"
    )
    assert path in exclusions
    assert "postprocess receipt binds its exact bytes" in exclusions[path]


def test_every_hilift_split_support_and_runtime_schema_is_bound(
    retained_manifest,
) -> None:
    manifest = retained_manifest
    specifications = group_paths(manifest, "fluidsbench_hilift_specification")
    expected_splits = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "benchmark-specs/hiliftaeroml/splits").glob("*.json")
    }
    expected_support = {
        path.relative_to(ROOT).as_posix()
        for path in (
            ROOT / "benchmark-specs/hiliftaeroml/scoring-support"
        ).glob("**/*.json")
    }
    assert expected_splits <= specifications
    assert expected_support <= specifications
    assert len(expected_splits) == 14
    assert "benchmark-specs/hiliftaeroml/submission-spec.json" in specifications
    assert "benchmark-specs/hiliftaeroml/methodology-contract.json" in specifications

    schemas = group_paths(manifest, "fluidsbench_hilift_schemas")
    expected_v3 = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "schemas/v3").glob("*.schema.json")
    }
    expected_support_schemas = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "schemas/scoring-support/v1").glob("*.schema.json")
    }
    assert expected_v3 <= schemas
    assert expected_support_schemas <= schemas
    assert "schemas/methodology-contract-v1.schema.json" in schemas
    assert "schemas/v1/profile-index.schema.json" in schemas
    assert "schemas/v1/hiliftaeroml-native-profile-chunk.schema.json" in schemas


def test_runtime_lock_is_minimal_exact_and_container_is_separate(
    retained_manifest,
) -> None:
    manifest = retained_manifest
    pins = {
        line.split("==", 1)[0]: line.split("==", 1)[1]
        for line in LOCK_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert pins == {
        "attrs": "26.1.0",
        "jsonschema": "4.26.0",
        "jsonschema-specifications": "2025.9.1",
        "numpy": "2.5.2",
        "referencing": "0.37.0",
        "rpds-py": "2026.6.3",
        "typing_extensions": "4.16.0",
    }
    assert {"pytest", "pluggy", "pip", "packaging"}.isdisjoint(pins)
    runtime = manifest["runtime_boundaries"]
    lock = runtime["fluidsbench_evaluator_python"]
    assert lock["interpreter"]["version"] == "3.12.13"
    assert lock["lock_file"]["identity_mode"] == "raw_bytes_sha256"
    assert lock["lock_file"]["sha256"] == hashlib.sha256(
        LOCK_PATH.read_bytes()
    ).hexdigest()
    native = runtime["native_gpu_container"]
    assert native["status"] == "external_container_reference_not_content_locked"
    assert native["image_digest"] is None
    assert native["host_python_environment_reference"].endswith(
        "/.venv-hilift-eval"
    )
    assert native["host_python_environment_lock"] is None
    assert native["host_physicsnemo_sources_hash_bound"] is True


def test_lifecycle_normalization_breaks_revision_cycle_but_binds_science(
    tmp_path: Path, retained_manifest
) -> None:
    manifest = retained_manifest
    group_id = "fluidsbench_hilift_specification"

    spec_relative = "benchmark-specs/hiliftaeroml/submission-spec.json"
    spec_identity = group_identity(manifest, group_id, spec_relative)
    assert spec_identity["identity_mode"] == (
        "canonical_json_sha256_excluding_lifecycle_v1"
    )
    spec_rule = builder.JSON_NORMALIZATION_RULES[
        ("fluidsbench_adapter", spec_relative)
    ]
    spec = json.loads((ROOT / spec_relative).read_text(encoding="utf-8"))
    remove_postfreeze_additive_extensions(spec)
    spec["status"] = "future_activation_state"
    spec["evaluation_reference_version"] = "future-revision-label"
    spec["scoring_support"]["status"] = "future_status"
    spec["scoring_support"]["submissions_open"] = True
    spec["scoring_support"]["closed_reason"] = ""
    spec["scoring_support"]["candidate_manifest"]["status"] = "future_status"
    spec["scoring_support"]["dataset_evaluator_binding"] = {
        "status": "frozen",
        "evaluator_code_revision": "a" * 40,
        "implementation_manifest_sha256": "b" * 64,
    }
    spec["scoring_support"]["owner_decisions_required"] = []
    spec["profile_definition"]["status"] = "future_status"
    spec["profile_definition"]["profile_ground_truth"] = {"status": "published"}
    spec["profile_definition"]["candidate_dry_run_profile_ground_truth"] = {}
    spec["profile_definition"]["activation_rule"] = "future rule"
    spec["regional_diagnostics"]["status"] = "future_status"
    spec["regional_diagnostics"]["required_for_new_submissions"] = True
    spec["regional_diagnostics"]["activation_gate"] = "passed"
    lifecycle_spec = tmp_path / "submission-spec-lifecycle.json"
    lifecycle_spec.write_text(json.dumps(spec), encoding="utf-8")
    normalized_lifecycle_spec = builder.describe_normalized_json(
        lifecycle_spec, spec_relative, spec_rule
    )
    assert normalized_lifecycle_spec == spec_identity

    scientific_spec = json.loads((ROOT / spec_relative).read_text(encoding="utf-8"))
    remove_postfreeze_additive_extensions(scientific_spec)
    scientific_spec["overall_score_composite"]["components"][0]["weight"] += 0.001
    scientific_spec_path = tmp_path / "submission-spec-scientific.json"
    scientific_spec_path.write_text(json.dumps(scientific_spec), encoding="utf-8")
    normalized_scientific_spec = builder.describe_normalized_json(
        scientific_spec_path, spec_relative, spec_rule
    )
    assert normalized_scientific_spec["canonical_sha256"] != spec_identity[
        "canonical_sha256"
    ]

    leaderboard_relative = "leaderboard/manifest.json"
    leaderboard_identity = group_identity(manifest, group_id, leaderboard_relative)
    assert leaderboard_identity["selection"] == {
        "kind": "unique_array_object",
        "array_json_pointer": "/datasets",
        "match_key": "slug",
        "match_value": "hiliftaeroml",
    }
    leaderboard_rule = builder.JSON_NORMALIZATION_RULES[
        ("fluidsbench_adapter", leaderboard_relative)
    ]
    leaderboard = json.loads((ROOT / leaderboard_relative).read_text(encoding="utf-8"))
    hilift = next(
        item for item in leaderboard["datasets"] if item["slug"] == "hiliftaeroml"
    )
    expected_profile_definition = json.loads(
        (ROOT / spec_relative).read_text(encoding="utf-8")
    )["profile_definition"]
    derived_profile_definition = hilift.pop("profile_definition")
    assert derived_profile_definition == expected_profile_definition
    hilift["submission_count"] += 1
    hilift["updated_at"] = "2099-12-31"
    hilift["revision_count"] += 1
    hilift["scoring_support"]["status"] = "future_status"
    hilift["scoring_support"]["submissions_open"] = True
    hilift["scoring_support"]["closed_reason"] = ""
    hilift["scoring_support"]["candidate_manifest"]["status"] = "future_status"
    hilift["scoring_support"]["dataset_evaluator_binding"] = {
        "status": "frozen",
        "evaluator_code_revision": "a" * 40,
        "implementation_manifest_sha256": "b" * 64,
    }
    hilift["scoring_support"]["owner_decisions_required"] = []
    lifecycle_leaderboard = tmp_path / "leaderboard-lifecycle.json"
    lifecycle_leaderboard.write_text(json.dumps(leaderboard), encoding="utf-8")
    normalized_lifecycle_leaderboard = builder.describe_normalized_json(
        lifecycle_leaderboard, leaderboard_relative, leaderboard_rule
    )
    assert normalized_lifecycle_leaderboard == leaderboard_identity

    scientific_leaderboard = json.loads(
        (ROOT / leaderboard_relative).read_text(encoding="utf-8")
    )
    scientific_hilift = next(
        item
        for item in scientific_leaderboard["datasets"]
        if item["slug"] == "hiliftaeroml"
    )
    derived_profile_definition = scientific_hilift.pop("profile_definition")
    assert derived_profile_definition == expected_profile_definition
    scientific_hilift["overall_score_composite"]["components"][0]["weight"] += 0.001
    scientific_leaderboard_path = tmp_path / "leaderboard-scientific.json"
    scientific_leaderboard_path.write_text(
        json.dumps(scientific_leaderboard), encoding="utf-8"
    )
    normalized_scientific_leaderboard = builder.describe_normalized_json(
        scientific_leaderboard_path, leaderboard_relative, leaderboard_rule
    )
    assert normalized_scientific_leaderboard[
        "canonical_sha256"
    ] != leaderboard_identity["canonical_sha256"]


def test_manifest_cannot_activate_or_resolve_release_gates(retained_manifest) -> None:
    manifest = retained_manifest
    assert manifest["status"] == builder.STATUS
    assert manifest["activation_effect"] == "none"
    assert manifest["release_state"] == {
        "git_revisions_bound": False,
        "owner_approval_recorded": False,
        "submissions_opened": False,
        "implementation_manifest_gate_changed": False,
        "runtime_lock_gate_changed": False,
    }
    every_path = {
        item["path"]
        for group in manifest["source_groups"]
        for item in group["files"]
    }
    assert (
        "benchmark-specs/hiliftaeroml/candidate-evaluator-release-binding.json"
        not in every_path
    )
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in every_path)
    assert all(
        root["revision_binding"] == "pending_candidate_commit"
        for root in manifest["source_roots"].values()
    )


def test_check_mode_fails_closed_on_different_bytes(
    tmp_path: Path, retained_manifest
) -> None:
    output = tmp_path / "manifest.json"
    payload = builder.pretty_bytes(retained_manifest)
    assert builder.write_or_check(output, payload, check=False) == "written"
    assert builder.write_or_check(output, payload, check=True) == "validated_existing"
    output.write_bytes(payload + b"\n")
    with pytest.raises(builder.ImplementationManifestError, match="differs"):
        builder.write_or_check(output, payload, check=True)
