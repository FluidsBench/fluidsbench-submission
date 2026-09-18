from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from reference.hiliftaeroml.compact_profiles import (
    CP_FIXED_POINT_SCALE,
    CP_POINTS_PER_GRAPH,
    PREDICTION_ARRAYS,
    SUPPORT_ARRAYS,
    VELOCITY_ROW_COUNT,
    VELOCITY_ROWS_PER_STATION,
    VELOCITY_STATIONS,
    CompactProfileError,
    allocate_branch_samples,
    build_compact_support,
    compact_case_metadata,
    decode_compact_predictions,
    decode_velocity_storage,
    deterministic_npz_bytes,
    deterministic_prediction_npz_bytes,
    encode_compact_predictions,
    encode_native_predictions,
    encode_velocity_storage,
    load_compact_prediction_npz,
    load_compact_support_npz,
    score_compact_profiles,
    surface_prediction_order_sha256,
    surface_support_identity_sha256,
    support_sha256,
    validate_compact_predictions,
    validate_compact_support,
    velocity_prediction_order_sha256,
    velocity_support_identity_sha256,
    write_compact_prediction_npz,
    write_compact_support_npz,
)
from reference.hiliftaeroml.native_profile_evaluator import _cp_r2, _velocity_r2


def _native_inputs() -> tuple[
    dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray
]:
    # Branches 0, 1, and 2 share physical graph (row=0, graph=0).  Although
    # branch 2 is less than 1e-4 of that graph's total length, the whole graph
    # has fewer than the configured point budget, so the lossless small-graph
    # rule keeps it.  Branch 3 is a second physical graph despite reusing graph
    # code zero on row 1.
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0005, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    truth_cp = np.asarray(
        [0.0, 1.0, 2.0, 2.0, 1.0, 9.0, 9.0, -1.0, 0.0, 1.0],
        dtype=np.float64,
    )
    prediction_cp = truth_cp + np.asarray(
        [0.125, 0.125, 0.125, 0.25, 0.25, 1.0, 1.0, 0.25, 0.25, 0.25],
        dtype=np.float64,
    )
    cp_native = {
        "cut_xyz_in": xyz,
        "branch_closed": np.zeros(4, dtype=np.bool_),
        "branch_vertex_offsets": np.asarray([0, 3, 5, 7, 10], dtype=np.int64),
        "branch_vertex_ids": np.arange(10, dtype=np.int64),
        "branch_segment_offsets": np.asarray([0, 2, 3, 4, 6], dtype=np.int64),
        "segment_lengths_in": np.asarray(
            [3.0, 3.0, 4.0, 0.0005, 2.0, 2.0], dtype=np.float64
        ),
        "branch_row_code": np.asarray([0, 0, 0, 1], dtype=np.uint8),
        "branch_graph_component_code": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "branch_component_code": np.asarray([0, 1, 2, 0], dtype=np.uint8),
        "branch_plane_piece_code": np.asarray([0, 1, 2, 0], dtype=np.uint8),
        "branch_side_code": np.asarray([0, 1, 0, 1], dtype=np.uint8),
        "branch_topology_patch_code": np.asarray([0, 1, 2, 0], dtype=np.uint8),
        "prediction_cp": prediction_cp,
    }

    offsets = (
        np.arange(len(VELOCITY_STATIONS) + 1, dtype=np.int64)
        * VELOCITY_ROWS_PER_STATION
    )
    valid = np.ones(VELOCITY_ROW_COUNT, dtype=np.bool_)
    for station_start in offsets[:-1]:
        valid[int(station_start) + 100] = False
    row = np.arange(VELOCITY_ROW_COUNT, dtype=np.float64)
    local_row = row % VELOCITY_ROWS_PER_STATION
    requested_xyz = np.column_stack(
        (local_row, row // VELOCITY_ROWS_PER_STATION, np.zeros_like(row))
    )
    weights = np.ones(VELOCITY_ROW_COUNT, dtype=np.float64)
    weights[~valid] = 0.0
    truth_speed = 0.5 + (local_row % 16.0) / 16.0
    predicted_speed = truth_speed + 0.125
    truth_velocity = np.column_stack(
        (truth_speed, np.zeros((VELOCITY_ROW_COUNT, 2), dtype=np.float64))
    )
    predicted_velocity = np.column_stack(
        (predicted_speed, np.zeros((VELOCITY_ROW_COUNT, 2), dtype=np.float64))
    )
    truth_velocity[~valid] = np.nan
    predicted_velocity[~valid] = np.nan
    predicted_speed = np.linalg.norm(predicted_velocity, axis=1)
    velocity_native = {
        "requested_xyz_in": requested_xyz,
        "valid_mask": valid,
        "station_names": np.asarray(VELOCITY_STATIONS),
        "station_row_offsets": offsets,
        "line_length_weights_in": weights,
        "predicted_velocity_nd": predicted_velocity,
        "speed_over_U_inf_pred": predicted_speed,
    }
    return cp_native, truth_cp, velocity_native, truth_velocity


@pytest.fixture
def compact_case() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    cp_native, truth_cp, velocity_native, truth_velocity = _native_inputs()
    support = build_compact_support(
        cp_native=cp_native,
        truth_cp=truth_cp,
        velocity_native=velocity_native,
        truth_velocity_nd=truth_velocity,
    )
    artifact = encode_native_predictions(
        support=support,
        cp_native=cp_native,
        velocity_native=velocity_native,
    )
    return support, artifact, cp_native, velocity_native


def _copy_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array(value, copy=True) for name, value in arrays.items()}


def _subset_digest(
    arrays: dict[str, np.ndarray], order: tuple[str, ...]
) -> str:
    payload = deterministic_npz_bytes(
        {name: arrays[name] for name in order}, order=order
    )
    return hashlib.sha256(payload).hexdigest()


def test_branch_allocation_prunes_only_tiny_branch_and_is_deterministic() -> None:
    keep, allocation = allocate_branch_samples([6.0, 4.0, 0.0005])
    repeated_keep, repeated_allocation = allocate_branch_samples(
        [6.0, 4.0, 0.0005]
    )
    assert keep.tolist() == [0, 1]
    np.testing.assert_array_equal(repeated_keep, keep)
    np.testing.assert_array_equal(repeated_allocation, allocation)
    assert np.all(allocation >= 2)
    assert int(allocation.sum()) == CP_POINTS_PER_GRAPH

    maximum_branches = CP_POINTS_PER_GRAPH // 2
    fragmented_count = maximum_branches + 6
    # More than the maximum number of equal branches cannot all retain both
    # endpoints. Stable ordering keeps the first eligible branches.
    keep, allocation = allocate_branch_samples(
        np.ones(fragmented_count, dtype=np.float64)
    )
    assert keep.tolist() == list(range(maximum_branches))
    assert allocation.tolist() == [2] * maximum_branches


def test_closed_native_branch_is_unwrapped_without_adding_a_scoring_edge() -> None:
    cp_native, truth_cp, velocity_native, truth_velocity = _native_inputs()
    cp_native["branch_closed"][3] = True
    cp_native["branch_segment_offsets"] = np.asarray(
        [0, 2, 3, 4, 7], dtype=np.int64
    )
    cp_native["segment_lengths_in"] = np.asarray(
        [3.0, 3.0, 4.0, 0.0005, 2.0, 2.0, 4.0], dtype=np.float64
    )

    support = build_compact_support(
        cp_native=cp_native,
        truth_cp=truth_cp,
        velocity_native=velocity_native,
        truth_velocity_nd=truth_velocity,
    )
    source_slot = support["cp_source_branch_index"].tolist().index(3)
    start = int(support["cp_branch_point_offsets"][source_slot])
    stop = int(support["cp_branch_point_offsets"][source_slot + 1])
    assert stop - start == 4
    np.testing.assert_array_equal(
        support["cp_xyz_in"][start], support["cp_xyz_in"][stop - 1]
    )
    assert support["cp_arc_length_in"][stop - 1] == 8.0
    assert support["cp_truth"][start] == support["cp_truth"][stop - 1]

    artifact = encode_native_predictions(
        support=support,
        cp_native=cp_native,
        velocity_native=velocity_native,
    )
    decoded_cp, _ = decode_compact_predictions(
        artifact,
        support=support,
        metadata=compact_case_metadata(support),
    )
    assert decoded_cp[start] == decoded_cp[stop - 1]


def test_support_and_prediction_roundtrip_are_deterministic_and_prediction_only(
    tmp_path: Path,
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    assert tuple(support) == SUPPORT_ARRAYS
    assert tuple(artifact) == PREDICTION_ARRAYS
    assert support["cp_source_branch_index"].tolist() == [0, 1, 2, 3]
    assert set(
        zip(
            support["cp_branch_row_code"].tolist(),
            support["cp_branch_graph_component_code"].tolist(),
            strict=True,
        )
    ) == {(0, 0), (1, 0)}
    offsets = support["cp_branch_point_offsets"]
    graph_zero_count = int(offsets[3] - offsets[0])
    graph_one_count = int(offsets[4] - offsets[3])
    assert graph_zero_count == 7
    assert graph_one_count == 3

    decoded_cp, decoded_speed = decode_compact_predictions(
        artifact, support=support, metadata=metadata
    )
    expected_cp = np.concatenate(
        [
            np.interp(
                support["cp_arc_length_in"][int(offsets[i]) : int(offsets[i + 1])],
                np.linspace(
                    0.0,
                    support["cp_arc_length_in"][int(offsets[i + 1]) - 1],
                    len(
                        cp_native["prediction_cp"][
                            cp_native["branch_vertex_offsets"][
                                support["cp_source_branch_index"][i]
                            ] : cp_native["branch_vertex_offsets"][
                                support["cp_source_branch_index"][i] + 1
                            ]
                        ]
                    ),
                ),
                cp_native["prediction_cp"][
                    cp_native["branch_vertex_offsets"][
                        support["cp_source_branch_index"][i]
                    ] : cp_native["branch_vertex_offsets"][
                        support["cp_source_branch_index"][i] + 1
                    ]
                ],
            )
            for i in range(len(offsets) - 1)
        ]
    )
    assert np.max(np.abs(decoded_cp - expected_cp)) <= (
        0.5 / CP_FIXED_POINT_SCALE + 2.0e-15
    )
    valid = support["velocity_valid_mask"]
    assert np.all(np.isfinite(decoded_speed[valid]))
    assert np.all(np.isnan(decoded_speed[~valid]))

    # The participant payload has no truth, coordinates, topology, masks, or
    # weights.  The chunk metadata, rather than the NPZ, binds those
    # evaluator-owned arrays and the exact prediction order by SHA-256.
    assert not any("truth" in name or "reference" in name for name in artifact)
    assert set(artifact) == {
        "cp_q_delta",
        "velocity_speed_over_u_inf",
    }
    assert metadata["surface_cp"] == {
        **metadata["surface_cp"],
        "support_identity_sha256": surface_support_identity_sha256(support),
        "prediction_order_sha256": surface_prediction_order_sha256(support),
    }
    assert metadata["volume_velocity"] == {
        **metadata["volume_velocity"],
        "support_identity_sha256": velocity_support_identity_sha256(support),
        "prediction_order_sha256": velocity_prediction_order_sha256(support),
    }
    surface_support_order = (
        "cp_xyz_in",
        "cp_arc_length_in",
        "cp_branch_point_offsets",
        "cp_branch_row_code",
        "cp_branch_graph_component_code",
        "cp_branch_component_code",
        "cp_branch_plane_piece_code",
        "cp_branch_side_code",
        "cp_branch_topology_patch_code",
        "cp_source_branch_index",
    )
    surface_prediction_order = (
        "cp_branch_point_offsets",
        "cp_branch_row_code",
        "cp_branch_graph_component_code",
        "cp_source_branch_index",
        "cp_xyz_in",
        "cp_arc_length_in",
    )
    velocity_support_order = (
        "velocity_requested_xyz_in",
        "velocity_valid_mask",
        "velocity_station_names",
        "velocity_station_row_offsets",
        "velocity_line_length_weights_in",
    )
    assert metadata["surface_cp"]["support_identity_sha256"] == (
        _subset_digest(support, surface_support_order)
    )
    assert metadata["surface_cp"]["prediction_order_sha256"] == (
        _subset_digest(support, surface_prediction_order)
    )
    assert metadata["volume_velocity"]["support_identity_sha256"] == (
        _subset_digest(support, velocity_support_order)
    )
    velocity_prediction_order = {
        "velocity_station_names": support["velocity_station_names"],
        "velocity_station_row_offsets": support[
            "velocity_station_row_offsets"
        ],
        "velocity_valid_row_indices": np.flatnonzero(
            support["velocity_valid_mask"]
        ).astype(np.int64),
    }
    assert metadata["volume_velocity"]["prediction_order_sha256"] == (
        _subset_digest(
            velocity_prediction_order, tuple(velocity_prediction_order)
        )
    )

    support_a = tmp_path / "support-a.npz"
    support_b = tmp_path / "support-b.npz"
    prediction_a = tmp_path / "prediction-a.npz"
    prediction_b = tmp_path / "prediction-b.npz"
    support_digest_a = write_compact_support_npz(support_a, support)
    support_digest_b = write_compact_support_npz(support_b, support)
    prediction_digest_a = write_compact_prediction_npz(
        prediction_a, artifact, support=support, metadata=metadata
    )
    prediction_digest_b = write_compact_prediction_npz(
        prediction_b, artifact, support=support, metadata=metadata
    )
    assert support_digest_a == support_digest_b == support_sha256(support)
    assert support_a.read_bytes() == support_b.read_bytes()
    assert prediction_digest_a == prediction_digest_b
    assert prediction_a.read_bytes() == prediction_b.read_bytes()
    with zipfile.ZipFile(prediction_a) as archive:
        assert archive.namelist() == [f"{name}.npy" for name in PREDICTION_ARRAYS]
        assert all(
            member.compress_type == zipfile.ZIP_DEFLATED
            for member in archive.infolist()
        )
        with archive.open("velocity_speed_over_u_inf.npy") as member:
            stored_speed = np.lib.format.read_array(member, allow_pickle=False)
        assert stored_speed.dtype == np.dtype(np.uint8)
        assert stored_speed.shape == (artifact["velocity_speed_over_u_inf"].size * 4,)
        np.testing.assert_array_equal(
            decode_velocity_storage(stored_speed).view(np.uint32),
            artifact["velocity_speed_over_u_inf"].view(np.uint32),
        )
        assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in archive.infolist())
        assert not any(
            "truth" in member.filename or "reference" in member.filename
            for member in archive.infolist()
        )

    loaded_support, observed_support_digest = load_compact_support_npz(support_a)
    loaded_metadata = compact_case_metadata(loaded_support)
    loaded_artifact, observed_prediction_digest = load_compact_prediction_npz(
        prediction_a, support=loaded_support, metadata=loaded_metadata
    )
    assert observed_support_digest == support_digest_a
    assert observed_prediction_digest == prediction_digest_a
    loaded_cp, loaded_speed = decode_compact_predictions(
        loaded_artifact, support=loaded_support, metadata=loaded_metadata
    )
    np.testing.assert_array_equal(loaded_cp, decoded_cp)
    np.testing.assert_array_equal(loaded_speed, decoded_speed)


def test_velocity_storage_roundtrip_preserves_every_float32_bit() -> None:
    values = np.asarray(
        [0.0, -0.0, 0.625, 1.0, 1.0000001192092896, 16.0],
        dtype=np.float32,
    )
    # Negative zero is numerically non-negative and its sign bit must survive.
    encoded = encode_velocity_storage(values)
    assert encoded.dtype == np.dtype(np.uint8)
    assert encoded.shape == (values.size * 4,)
    decoded = decode_velocity_storage(encoded)
    np.testing.assert_array_equal(decoded.view(np.uint32), values.view(np.uint32))


def test_compact_scores_match_native_quadrature_on_identical_compact_support(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, _cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    decoded_cp, decoded_speed = decode_compact_predictions(
        artifact, support=support, metadata=metadata
    )
    offsets = support["cp_branch_point_offsets"]
    endpoints: list[np.ndarray] = []
    branch_segment_offsets = [0]
    segment_lengths: list[np.ndarray] = []
    for start_value, end_value in zip(offsets[:-1], offsets[1:], strict=True):
        start = int(start_value)
        end = int(end_value)
        endpoints.append(
            np.column_stack(
                (
                    np.arange(start, end - 1, dtype=np.int64),
                    np.arange(start + 1, end, dtype=np.int64),
                )
            )
        )
        local_lengths = np.diff(support["cp_arc_length_in"][start:end])
        segment_lengths.append(local_lengths)
        branch_segment_offsets.append(
            branch_segment_offsets[-1] + len(local_lengths)
        )
    native_cp_arrays = {
        "prediction_cp": decoded_cp,
        "segment_vertex_ids": np.concatenate(endpoints),
        "segment_lengths_in": np.concatenate(segment_lengths),
        "branch_segment_offsets": np.asarray(
            branch_segment_offsets, dtype=np.int64
        ),
        "branch_row_code": support["cp_branch_row_code"],
        "branch_graph_component_code": support[
            "cp_branch_graph_component_code"
        ],
    }
    truth_speed = support["velocity_truth_speed_over_uinf"]
    native_velocity_arrays = {
        "predicted_velocity_nd": np.column_stack(
            (
                decoded_speed,
                np.zeros((VELOCITY_ROW_COUNT, 2), dtype=np.float64),
            )
        ),
        "line_length_weights_in": support["velocity_line_length_weights_in"],
        "station_row_offsets": support["velocity_station_row_offsets"],
    }
    truth_velocity = np.column_stack(
        (
            truth_speed,
            np.zeros((VELOCITY_ROW_COUNT, 2), dtype=np.float64),
        )
    )
    observed = score_compact_profiles(
        artifact, support=support, metadata=metadata
    )
    assert observed["cp_cut_r2"] == pytest.approx(
        _cp_r2(native_cp_arrays, support["cp_truth"]), abs=2.0e-14
    )
    assert observed["velocity_profile_r2"] == pytest.approx(
        _velocity_r2(native_velocity_arrays, truth_velocity), abs=2.0e-14
    )


def test_cp_delta_resets_at_every_retained_branch(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, _cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    cp, _speed = decode_compact_predictions(
        artifact, support=support, metadata=metadata
    )
    quantized = np.rint(cp * CP_FIXED_POINT_SCALE).astype(np.int64)
    offsets = support["cp_branch_point_offsets"]
    for start_value in offsets[:-1]:
        start = int(start_value)
        assert int(artifact["cp_q_delta"][start]) == int(quantized[start])


def test_prediction_validation_rejects_truth_leakage_and_wrong_support(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, _cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    leaked = _copy_arrays(artifact)
    leaked["cp_truth"] = np.array(support["cp_truth"], copy=True)
    with pytest.raises(CompactProfileError, match="inventory differs"):
        validate_compact_predictions(
            leaked, support=support, metadata=metadata
        )

    # The complete private support digest binds truth, while the four values
    # copied into participant chunk metadata deliberately do not expose or
    # hash truth.
    different_truth = _copy_arrays(support)
    different_truth["cp_truth"][0] += 1.0
    assert support_sha256(different_truth) != support_sha256(support)
    assert compact_case_metadata(different_truth) == metadata
    validate_compact_predictions(
        artifact, support=different_truth, metadata=metadata
    )

    # A change to public support is rejected by the unchanged chunk binding.
    different_support = _copy_arrays(support)
    different_support["cp_xyz_in"][0, 0] += 0.25
    with pytest.raises(CompactProfileError, match="metadata differs"):
        validate_compact_predictions(
            artifact, support=different_support, metadata=metadata
        )


def test_encoder_rejects_cp_absolute_and_delta_overflow(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, _cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    cp, speed = decode_compact_predictions(
        artifact, support=support, metadata=metadata
    )
    absolute_overflow = np.array(cp, copy=True)
    absolute_overflow[0] = 32.0
    with pytest.raises(CompactProfileError, match="quantized Cp.*overflows"):
        encode_compact_predictions(
            support=support,
            cp_prediction=absolute_overflow,
            velocity_speed_over_u_inf=speed,
        )

    delta_overflow = np.array(cp, copy=True)
    delta_overflow[:2] = [-31.0, 31.0]
    with pytest.raises(CompactProfileError, match="delta-coded Cp.*overflows"):
        encode_compact_predictions(
            support=support,
            cp_prediction=delta_overflow,
            velocity_speed_over_u_inf=speed,
        )

    malformed = _copy_arrays(artifact)
    malformed["cp_q_delta"][:2] = [32767, 1]
    with pytest.raises(CompactProfileError, match="reconstruction overflows"):
        validate_compact_predictions(
            malformed, support=support, metadata=metadata
        )


def test_encoder_rejects_filled_or_invalid_velocity_gaps(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, _cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    cp, speed = decode_compact_predictions(
        artifact, support=support, metadata=metadata
    )
    gap = int(np.flatnonzero(~support["velocity_valid_mask"])[0])
    valid_row = int(np.flatnonzero(support["velocity_valid_mask"])[0])

    filled_gap = np.array(speed, copy=True)
    filled_gap[gap] = 0.0
    with pytest.raises(CompactProfileError, match="retain NaN"):
        encode_compact_predictions(
            support=support,
            cp_prediction=cp,
            velocity_speed_over_u_inf=filled_gap,
        )

    missing_valid = np.array(speed, copy=True)
    missing_valid[valid_row] = np.nan
    with pytest.raises(CompactProfileError, match="finite and non-negative"):
        encode_compact_predictions(
            support=support,
            cp_prediction=cp,
            velocity_speed_over_u_inf=missing_valid,
        )

    negative = np.array(speed, copy=True)
    negative[valid_row] = -1.0
    with pytest.raises(CompactProfileError, match="finite and non-negative"):
        encode_compact_predictions(
            support=support,
            cp_prediction=cp,
            velocity_speed_over_u_inf=negative,
        )


def test_native_encoder_requires_exact_cp_and_velocity_support_alignment(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, _artifact, cp_native, velocity_native = compact_case

    wrong_cp_coordinates = _copy_arrays(cp_native)
    wrong_cp_coordinates["cut_xyz_in"][0, 1] += 1.0e-6
    with pytest.raises(CompactProfileError, match="Cp plotting coordinates differ"):
        encode_native_predictions(
            support=support,
            cp_native=wrong_cp_coordinates,
            velocity_native=velocity_native,
        )

    wrong_coordinates = _copy_arrays(velocity_native)
    wrong_coordinates["requested_xyz_in"][0, 0] += 1.0e-6
    with pytest.raises(CompactProfileError, match="plotting coordinates differ"):
        encode_native_predictions(
            support=support,
            cp_native=cp_native,
            velocity_native=wrong_coordinates,
        )

    wrong_mask = _copy_arrays(velocity_native)
    wrong_mask["valid_mask"][0] = False
    with pytest.raises(CompactProfileError, match="validity mask differ"):
        encode_native_predictions(
            support=support,
            cp_native=cp_native,
            velocity_native=wrong_mask,
        )


def test_support_validation_rejects_excess_resolution_and_gap_changes(
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, _artifact, _cp_native, _velocity_native = compact_case
    wrong_resolution = _copy_arrays(support)
    offsets = wrong_resolution["cp_branch_point_offsets"]
    old_first_end = int(offsets[1])
    new_first_count = CP_POINTS_PER_GRAPH - 4 + 1
    first_xyz = wrong_resolution["cp_xyz_in"][:old_first_end]
    first_arc = wrong_resolution["cp_arc_length_in"][:old_first_end]
    first_truth = wrong_resolution["cp_truth"][:old_first_end]
    replacement_arc = np.linspace(
        float(first_arc[0]), float(first_arc[-1]), new_first_count
    )
    replacement_xyz = np.column_stack(
        [
            np.interp(replacement_arc, first_arc, first_xyz[:, axis])
            for axis in range(3)
        ]
    )
    replacement_truth = np.interp(replacement_arc, first_arc, first_truth)
    wrong_resolution["cp_xyz_in"] = np.concatenate(
        (replacement_xyz, wrong_resolution["cp_xyz_in"][old_first_end:])
    )
    wrong_resolution["cp_arc_length_in"] = np.concatenate(
        (replacement_arc, wrong_resolution["cp_arc_length_in"][old_first_end:])
    )
    wrong_resolution["cp_truth"] = np.concatenate(
        (replacement_truth, wrong_resolution["cp_truth"][old_first_end:])
    )
    offset_delta = new_first_count - old_first_end
    wrong_resolution["cp_branch_point_offsets"][1:] += offset_delta
    with pytest.raises(
        CompactProfileError,
        match=rf"{CP_POINTS_PER_GRAPH}-point maximum",
    ):
        validate_compact_support(wrong_resolution)

    wrong_gap_weight = _copy_arrays(support)
    gap = int(np.flatnonzero(~wrong_gap_weight["velocity_valid_mask"])[0])
    wrong_gap_weight["velocity_line_length_weights_in"][gap] = 1.0
    with pytest.raises(CompactProfileError, match="zero on gaps"):
        validate_compact_support(wrong_gap_weight)


def test_loaders_reject_noncanonical_array_order(
    tmp_path: Path,
    compact_case: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
) -> None:
    support, artifact, _cp_native, _velocity_native = compact_case
    metadata = compact_case_metadata(support)
    path = tmp_path / "wrong-order.npz"
    path.write_bytes(
        deterministic_prediction_npz_bytes(
            artifact, order=tuple(reversed(PREDICTION_ARRAYS))
        )
    )
    with pytest.raises(CompactProfileError, match="order differs"):
        load_compact_prediction_npz(
            path, support=support, metadata=metadata
        )
