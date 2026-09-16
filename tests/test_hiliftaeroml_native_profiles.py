from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from reference.hiliftaeroml.native_profiles import (
    CP_PUBLISHED_ARRAYS,
    CP_SOURCE_ARRAYS,
    NativeProfileError,
    PROFILE_CONTRACT_SHA256,
    VELOCITY_PUBLISHED_ARRAYS,
    VELOCITY_SOURCE_ARRAYS,
    build_profile_directory,
    validate_prediction_npz,
    validate_velocity_source,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "geo_LHC001_AoA_4"
STATIONS = ("B.2", "B.3", "C.1", "C.2", "C.3")
ROWS = tuple("ABCDEFGHIJ")
ROW_COUNT = 5 * 801


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_ordered_npz(path: Path, arrays: dict[str, np.ndarray], order: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert set(arrays) == set(order)
    np.savez(path, **{name: arrays[name] for name in order})


def make_cp_source(surface: Path) -> None:
    arrays = {
        "branch_closed": np.asarray([False], dtype=np.bool_),
        "branch_component_code": np.asarray([0], dtype=np.uint8),
        "branch_graph_component_code": np.asarray([0], dtype=np.int64),
        "branch_plane_piece_code": np.asarray([0], dtype=np.uint8),
        "branch_row_code": np.asarray([0], dtype=np.uint8),
        "branch_segment_offsets": np.asarray([0, 2], dtype=np.int64),
        "branch_side_code": np.asarray([0], dtype=np.uint8),
        "branch_topology_patch_code": np.asarray([0], dtype=np.uint8),
        "branch_vertex_ids": np.asarray([0, 1, 2], dtype=np.int64),
        "branch_vertex_offsets": np.asarray([0, 3], dtype=np.int64),
        "cut_xyz_in": np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "prediction_cp": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "segment_lengths_in": np.asarray([1.0, 1.0], dtype=np.float64),
        "segment_plane_piece_code": np.asarray([0, 0], dtype=np.uint8),
        "segment_vertex_ids": np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        "truth_cp": np.asarray([0.0, 0.25, 0.5], dtype=np.float64),
    }
    npz = surface / "cp_cut_values.npz"
    write_ordered_npz(npz, arrays, CP_SOURCE_ARRAYS)
    sha = digest(npz)
    metrics = {
        "cut_values": {
            "cut_vertex_count": 3,
            "path": str(npz.resolve()),
            "sha256": sha,
            "size_bytes": npz.stat().st_size,
        },
        "metrics": {
            "catalogs": {
                "components": ["main_wing"],
                "plane_pieces": ["wing"],
                "rows": list(ROWS),
                "sides": ["upper"],
                "topology_patches": ["WING"],
            },
            "all_rows": {
                "rows": list(ROWS),
                "metrics": {"r2_centered": 0.75},
            },
        },
        "profile_metric_schema_id": "hilift_native_cp_line_metrics_v3",
        "run_fingerprint": "1" * 64,
        "status": "complete",
        "stencil_content_sha256": "2" * 64,
        "stencil_payload_sha256": "3" * 64,
        "stencil_semantic_sha256": "4" * 64,
    }
    write_json(surface / "cp_profile_metrics.json", metrics)


def make_velocity_source(volume: Path, *, corrupt_invalid_gap: bool = False) -> None:
    valid = np.ones(ROW_COUNT, dtype=np.bool_)
    valid[100] = False
    xyz = np.zeros((ROW_COUNT, 3), dtype=np.float64)
    xyz[:, 0] = np.arange(ROW_COUNT, dtype=np.float64)
    weights = np.ones(ROW_COUNT, dtype=np.float64)
    weights[~valid] = 0.0
    predicted = np.column_stack(
        (
            np.full(ROW_COUNT, 1.0),
            np.full(ROW_COUNT, 2.0),
            np.full(ROW_COUNT, 2.0),
        )
    ).astype(np.float64)
    reference = predicted + 0.1
    predicted[~valid] = np.nan
    reference[~valid] = np.nan
    magnitude = np.linalg.norm(predicted, axis=1)
    reference_magnitude = np.linalg.norm(reference, axis=1)
    if corrupt_invalid_gap:
        predicted[100] = 0.0
    support_prediction = np.asarray(
        [[1.0, 2.0, 2.0], [1.1, 2.1, 2.1], [1.2, 2.2, 2.2]], dtype=np.float32
    )
    support_reference = support_prediction + np.float32(0.1)
    arrays = {
        "requested_xyz_in": xyz,
        "valid_mask": valid,
        "station_names": np.asarray(STATIONS),
        "station_row_offsets": np.arange(6, dtype=np.int64) * 801,
        "line_length_weights_in": weights,
        "predicted_velocity_nd": predicted,
        "reference_velocity_nd": reference,
        "predicted_velocity_physical": predicted * 2.0,
        "reference_velocity_physical": reference * 2.0,
        "predicted_velocity_dimensional": predicted * 2.0,
        "reference_velocity_dimensional": reference * 2.0,
        "predicted_velocity_magnitude_nd": magnitude,
        "reference_velocity_magnitude_nd": reference_magnitude,
        "u_x_over_U_inf_pred": predicted[:, 0],
        "u_x_over_U_inf_reference": reference[:, 0],
        "u_y_over_U_inf_pred": predicted[:, 1],
        "u_y_over_U_inf_reference": reference[:, 1],
        "u_z_over_U_inf_pred": predicted[:, 2],
        "u_z_over_U_inf_reference": reference[:, 2],
        "speed_over_U_inf_pred": magnitude,
        "speed_over_U_inf_reference": reference_magnitude,
        "u_inf_magnitude_physical": np.asarray(2.0, dtype=np.float64),
        "support_raw_point_ids": np.asarray([10, 20, 30], dtype=np.int64),
        "support_compact_point_ids": np.asarray([0, 1, 2], dtype=np.int64),
        "predicted_velocity_support_nd": support_prediction,
        "reference_velocity_support_nd": support_reference,
        "predicted_velocity_support_physical": support_prediction * np.float32(2.0),
        "reference_velocity_support_physical": support_reference * np.float32(2.0),
        "u_inf_vector_physical": np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        "velocity_physical_units": np.asarray("m/s"),
        "velocity_nondimensionalization": np.asarray("U/|U_inf|"),
        "context_ceiling": np.asarray(200000, dtype=np.int64),
        "partition_seed": np.asarray(42, dtype=np.int64),
        "run_fingerprint": np.asarray("5" * 64),
    }
    npz = volume / "velocity_profiles.npz"
    write_ordered_npz(npz, arrays, VELOCITY_SOURCE_ARRAYS)
    metrics = {
        "capture_plan_fingerprint": "6" * 64,
        "case_id": CASE_ID,
        "metric_value_basis": "velocity divided by |U_inf|",
        "pooled": {
            "velocity_magnitude": {
                "count": int(np.count_nonzero(valid)),
                "r2": 0.5,
            }
        },
        "profile_npz": {
            "filename": npz.name,
            "path": str(npz.resolve()),
            "sha256": digest(npz),
            "size_bytes": npz.stat().st_size,
        },
        "ref_values_csv_sha256": "7" * 64,
        "run_fingerprint": "5" * 64,
        "schema_id": "hilift_native_velocity_profile_metrics_v2",
        "stations": {station: {} for station in STATIONS},
        "stencil_record_sha256": "8" * 64,
    }
    write_json(volume / "velocity_profile_metrics.json", metrics)


def make_case(root: Path) -> Path:
    case = root / CASE_ID
    make_cp_source(case / "surface_submission_stream")
    make_velocity_source(case / "volume_submission_stream")
    return case


def retained_profile_hashes(outputs: Path) -> dict[str, dict[str, str]]:
    surface = outputs / CASE_ID / "surface_submission_stream"
    volume = outputs / CASE_ID / "volume_submission_stream"
    return {
        CASE_ID: {
            "cp_profile_metrics": digest(surface / "cp_profile_metrics.json"),
            "cp_cut_values": digest(surface / "cp_cut_values.npz"),
            "velocity_profile_metrics": digest(
                volume / "velocity_profile_metrics.json"
            ),
            "velocity_profiles": digest(volume / "velocity_profiles.npz"),
        }
    }


def test_native_profiles_are_prediction_only_deterministic_and_schema_valid(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    expected_source_hashes = retained_profile_hashes(outputs)
    first = tmp_path / "profiles-a"
    second = tmp_path / "profiles-b"
    first_index_sha, first_metrics = build_profile_directory(
        submission_id="hilift-test-v1",
        split_id="full",
        case_set_id="caseset-test",
        case_ids=[CASE_ID],
        outputs_root=outputs,
        profiles_root=first,
        cases_per_chunk=1,
        expected_case_artifact_sha256=expected_source_hashes,
    )
    second_index_sha, second_metrics = build_profile_directory(
        submission_id="hilift-test-v1",
        split_id="full",
        case_set_id="caseset-test",
        case_ids=[CASE_ID],
        outputs_root=outputs,
        profiles_root=second,
        cases_per_chunk=1,
        expected_case_artifact_sha256=expected_source_hashes,
    )
    assert first_index_sha == second_index_sha
    assert first_metrics == second_metrics == {
        CASE_ID: {"cp_cut_r2": 0.75, "velocity_profile_r2": 0.5}
    }
    assert digest(first / "chunk-000.json") == digest(second / "chunk-000.json")
    chunk = json.loads((first / "chunk-000.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schemas/v1/hiliftaeroml-native-profile-chunk.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(Draft202012Validator(schema).iter_errors(chunk)) == []
    assert chunk["contract_sha256"] == PROFILE_CONTRACT_SHA256

    cp = first / "artifacts" / CASE_ID / "surface-cp-predictions.npz"
    velocity = first / "artifacts" / CASE_ID / "volume-velocity-predictions.npz"
    assert digest(cp) == digest(second / "artifacts" / CASE_ID / cp.name)
    assert digest(velocity) == digest(second / "artifacts" / CASE_ID / velocity.name)
    cp_metadata = chunk["cases"][0]["surface_cp"]
    velocity_metadata = chunk["cases"][0]["volume_velocity"]
    validate_prediction_npz(
        cp,
        expected_arrays=CP_PUBLISHED_ARRAYS,
        expected_sha256=digest(cp),
        metadata=cp_metadata,
    )
    validate_prediction_npz(
        velocity,
        expected_arrays=VELOCITY_PUBLISHED_ARRAYS,
        expected_sha256=digest(velocity),
        metadata=velocity_metadata,
    )
    with np.load(cp, allow_pickle=False) as archive:
        assert "truth_cp" not in archive.files
    with np.load(velocity, allow_pickle=False) as archive:
        assert not any(
            name.startswith("reference_") or name.endswith("_reference")
            for name in archive.files
        )

    with np.load(velocity, allow_pickle=False) as archive:
        tampered = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    invalid_index = int(np.flatnonzero(~tampered["valid_mask"])[0])
    tampered["predicted_velocity_nd"][invalid_index] = 0.0
    velocity.unlink()
    write_ordered_npz(velocity, tampered, VELOCITY_PUBLISHED_ARRAYS)
    with pytest.raises(NativeProfileError, match="NaN on every invalid row"):
        validate_prediction_npz(
            velocity,
            expected_arrays=VELOCITY_PUBLISHED_ARRAYS,
            expected_sha256=digest(velocity),
            metadata=velocity_metadata,
        )

    wrong_source_hashes = retained_profile_hashes(outputs)
    wrong_source_hashes[CASE_ID]["cp_profile_metrics"] = "0" * 64
    with pytest.raises(NativeProfileError, match="retained receipt"):
        build_profile_directory(
            submission_id="hilift-test-v1",
            split_id="full",
            case_set_id="caseset-test",
            case_ids=[CASE_ID],
            outputs_root=outputs,
            profiles_root=tmp_path / "profiles-bad-receipt",
            cases_per_chunk=1,
            expected_case_artifact_sha256=wrong_source_hashes,
        )


def test_velocity_invalid_gap_cannot_be_filled(tmp_path: Path) -> None:
    volume = tmp_path / "volume_submission_stream"
    make_velocity_source(volume, corrupt_invalid_gap=True)
    with pytest.raises(NativeProfileError, match="NaN on every invalid row"):
        validate_velocity_source(
            case_id=CASE_ID,
            metrics_path=volume / "velocity_profile_metrics.json",
            npz_path=volume / "velocity_profiles.npz",
        )


def test_native_profiles_accept_separate_surface_and_volume_output_roots(
    tmp_path: Path,
) -> None:
    surface_root = tmp_path / "surface-outputs"
    volume_root = tmp_path / "volume-outputs"
    surface = surface_root / CASE_ID / "surface_submission_stream"
    volume = volume_root / CASE_ID / "volume_submission_stream"
    make_cp_source(surface)
    make_velocity_source(volume)
    expected = {
        CASE_ID: {
            "cp_profile_metrics": digest(surface / "cp_profile_metrics.json"),
            "cp_cut_values": digest(surface / "cp_cut_values.npz"),
            "velocity_profile_metrics": digest(
                volume / "velocity_profile_metrics.json"
            ),
            "velocity_profiles": digest(volume / "velocity_profiles.npz"),
        }
    }
    _, metrics = build_profile_directory(
        submission_id="hilift-split-root-test-v1",
        split_id="full",
        case_set_id="caseset-test",
        case_ids=[CASE_ID],
        outputs_root=None,
        surface_outputs_root=surface_root,
        volume_outputs_root=volume_root,
        profiles_root=tmp_path / "profiles",
        cases_per_chunk=1,
        expected_case_artifact_sha256=expected,
    )
    assert metrics == {
        CASE_ID: {"cp_cut_r2": 0.75, "velocity_profile_r2": 0.5}
    }
