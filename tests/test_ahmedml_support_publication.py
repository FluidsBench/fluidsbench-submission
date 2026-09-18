from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_ahmedml_candidate_support import (
    RELEASE_ID,
    bind_candidate_manifest,
)
from reference.ahmedml.contract import (
    FORCE_ABSOLUTE_TOLERANCE,
    FORCE_REPLAY_EXCEPTIONS,
    AhmedMLContractError,
    classify_force_replay,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_candidate_manifest_binding_is_hash_exact_and_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark-specs" / "ahmedml"
    specification = dataset / "submission-spec.json"
    manifest = dataset / "scoring-support" / RELEASE_ID / "manifest.json"
    _write(
        specification,
        {
            "dataset_id": "ahmedml",
            "scoring_support": {
                "status": "owner_review_required",
                "submissions_open": False,
            },
        },
    )
    _write(
        manifest,
        {
            "release_id": RELEASE_ID,
            "dataset_id": "ahmedml",
            "status": "candidate",
        },
    )

    binding = bind_candidate_manifest(
        specification_path=specification,
        manifest_path=manifest,
    )

    assert binding["status"] == "candidate"
    assert binding["release_id"] == RELEASE_ID
    assert binding["manifest_file"] == f"scoring-support/{RELEASE_ID}/manifest.json"
    assert binding["manifest_sha256"] == sha256_file(manifest)
    assert (
        json.loads(specification.read_text(encoding="utf-8"))["scoring_support"][
            "candidate_manifest"
        ]
        == binding
    )


def test_derived_cache_contract_keeps_public_dataset_canonical() -> None:
    dataset = ROOT / "benchmark-specs" / "ahmedml"
    contract_path = dataset / "derived-cache-contract-v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    specification = json.loads(
        (dataset / "submission-spec.json").read_text(encoding="utf-8")
    )

    assert contract["schema"] == "fluidsbench-ahmedml-derived-cache-contract-v1"
    source = contract["canonical_source"]
    assert source["kind"] == "huggingface_dataset_revision"
    assert source["repository_id"] == "neashton/ahmedml"
    assert source["revision"] == "02688c727cdb8dc8678e28abc6bbbb7e93c5fa15"
    public_roles = {
        item["role"]: item for item in source["required_public_artifacts"]
    }
    assert public_roles["surface_scalar_polygon_area"] == {
        "role": "surface_scalar_polygon_area",
        "path_rule": "run_<N>/boundary_cell_area_<N>.npy",
        "canonical": True,
        "dtype": "<f4",
        "association": "native_boundary_polygon_order",
    }
    assert sha256_file(dataset / source["source_identity_file"]) == source[
        "source_identity_sha256"
    ]

    policy = contract["cache_policy"]
    assert policy["classification"] == "deterministic_rebuildable_evaluator_cache"
    assert policy["canonical_source_data"] is False
    assert policy["remote_payload_required"] is False
    assert policy["runtime_rebuild_during_submission_scoring"] is False
    assert (
        policy["production_requirement"]
        == "preinstall_and_verify_before_accepting_submissions"
    )

    implementation = contract["derivation_implementation"]
    assert len(implementation["git_revision_with_exact_builder"]) == 40
    assert sha256_file(ROOT / implementation["builder_file"]) == implementation[
        "builder_sha256"
    ]
    assert sha256_file(ROOT / implementation["requirements_file"]) == implementation[
        "requirements_sha256"
    ]
    assert implementation["runtime"] == {
        "python": "3.12.3",
        "numpy": "2.5.2",
        "vtk": "9.5.2",
        "byte_order": "little_endian",
    }

    expected = contract["expected_outputs"]
    assert expected["official_test_case_union_count"] == 316
    assert sha256_file(dataset / expected["candidate_manifest_file"]) == expected[
        "candidate_manifest_sha256"
    ]
    binding = specification["scoring_support"]["derived_cache_contract"]
    assert binding == {
        "status": "candidate",
        "classification": "deterministic_rebuildable_evaluator_cache",
        "contract_file": "derived-cache-contract-v1.json",
        "contract_sha256": sha256_file(contract_path),
        "remote_payload_required": False,
        "production_installation_required": True,
    }
    public_supports = {
        item["id"]: item for item in specification["scoring_support"]["public_supports"]
    }
    assert (
        public_supports["surface-native"]["physical_weight_public_file"]
        == "run_<case>/boundary_cell_area_<case>.npy"
    )
    assert public_supports["flow-domain-native"]["primary_ranked_weighting"] == (
        "equal_native_cell"
    )
    assert "publish_the_generated_scoring_support_release" not in specification[
        "scoring_support"
    ]["owner_decisions_required"]


def test_profile_panels_pin_exact_native_sampling_grids() -> None:
    specification = json.loads(
        (ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json").read_text(
            encoding="utf-8"
        )
    )
    panels = {panel["id"]: panel for panel in specification["profile_panels"]}

    expected_intervals = {
        "pressure_profiles": {
            "upper_body_centerline": [0.0, 1.0],
            "underbody_centerline": [0.0, 1.0],
            "rear_slant_centerline": [0.0, 1.0],
        },
        "velocity_profiles": {
            "wake_vertical_x_0p25_l": [0.0, 2.0],
            "wake_vertical_x_0p50_l": [0.0, 2.0],
            "wake_vertical_x_1p00_l": [0.0, 2.0],
            "wake_lateral_x_0p50_l_z_0p50_h": [-1.0, 1.0],
        },
    }
    for panel_id, station_intervals in expected_intervals.items():
        panel = panels[panel_id]
        assert panel["exact_points"] == 128
        assert panel["minimum_points"] == 128
        assert panel["station_coordinate_intervals"] == station_intervals
        assert panel["station_sample_counts"] == {
            station_id: 128 for station_id in station_intervals
        }
        assert panel["station_coordinate_spacings"] == {
            station_id: "uniform" for station_id in station_intervals
        }


def test_run492_force_csv_exception_is_exact_and_field_owned() -> None:
    exception = FORCE_REPLAY_EXCEPTIONS["run_492"]
    assert (
        classify_force_replay(
            case_id=exception.case_id,
            published_cd=exception.published_cd,
            published_cl=exception.published_cl,
            field_replay_cd=exception.field_replay_cd,
            field_replay_cl=exception.field_replay_cl,
        )
        == exception
    )
    assert (
        abs(exception.field_replay_cd - exception.published_cd)
        > FORCE_ABSOLUTE_TOLERANCE
    )
    assert (
        abs(exception.field_replay_cl - exception.published_cl)
        > FORCE_ABSOLUTE_TOLERANCE
    )
    specification = json.loads(
        (ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert specification["scoring_support"]["source_audit_exceptions"] == [
        {
            "id": exception.exception_id,
            "case_id": exception.case_id,
            "scope": "published_force_csv_replay_only",
            "published_csv_used_for_scoring": False,
            "scoring_truth_source": "native_surface_field_integration",
            "policy": (
                "checksum_and_value_bound_single_case_exception_"
                "no_global_tolerance_relaxation"
            ),
        }
    ]
    with pytest.raises(AhmedMLContractError, match="pinned force-replay exception"):
        classify_force_replay(
            case_id=exception.case_id,
            published_cd=exception.published_cd,
            published_cl=exception.published_cl,
            field_replay_cd=exception.field_replay_cd + 1.0e-3,
            field_replay_cl=exception.field_replay_cl,
        )
