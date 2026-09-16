from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from reference.hiliftaeroml.native_profile_evaluator import (
    CANDIDATE_BINDING_PATH,
    CANDIDATE_TRUTH_STATUS,
    CANDIDATE_USAGE,
    NativeProfileEvaluationError,
    _cp_r2,
    _velocity_r2,
    load_candidate_truth_binding,
)


ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION = ROOT / "benchmark-specs" / "hiliftaeroml" / "submission-spec.json"


def test_candidate_binding_is_explicitly_complete_but_inactive() -> None:
    specification = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
    profile = specification["profile_definition"]
    declaration = profile["candidate_dry_run_profile_ground_truth"]
    binding = load_candidate_truth_binding(
        expected_sha256=declaration["binding_sha256"]
    )

    assert profile["profile_ground_truth"] == {
        "status": "not_published",
        "release_id": None,
        "manifest_sha256": None,
    }
    assert declaration["status"] == binding["status"] == CANDIDATE_TRUTH_STATUS
    assert declaration["usage"] == binding["usage"] == CANDIDATE_USAGE
    assert declaration["release_id"] == binding["release"]["release_id"]
    assert declaration["manifest_sha256"] == binding["release"]["manifest_sha256"]
    assert binding["activation"] == {
        "owner_approval_complete": False,
        "published": False,
        "submissions_opened": False,
        "candidate_validation_may_change_activation": False,
    }
    assert hashlib.sha256(CANDIDATE_BINDING_PATH.read_bytes()).hexdigest() == (
        declaration["binding_sha256"]
    )


def test_cp_r2_centers_each_physical_row_graph_independently() -> None:
    # The graph component code intentionally repeats across two station rows.
    # Each one-unit linear graph has SST=1/12.  Only the second graph has a
    # constant +1 prediction error (SSE=1), so R2 = 1 - 1/(1/6) = -5.
    arrays = {
        "prediction_cp": np.asarray([0.0, 1.0, 11.0, 12.0]),
        "segment_vertex_ids": np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        "segment_lengths_in": np.asarray([1.0, 1.0]),
        "branch_segment_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "branch_row_code": np.asarray([0, 1], dtype=np.uint8),
        "branch_graph_component_code": np.asarray([0, 0], dtype=np.int64),
    }
    truth = np.asarray([0.0, 1.0, 10.0, 11.0])
    assert _cp_r2(arrays, truth) == pytest.approx(-5.0, abs=2e-13)


def test_velocity_r2_centers_each_station_and_pools_physical_weights() -> None:
    # Each station has discrete weighted SST=0.5.  The second station has two
    # unit errors (SSE=2), hence the pooled station-centered R2 is -1.
    truth_magnitude = np.asarray([0.0, 1.0, 10.0, 11.0])
    prediction_magnitude = np.asarray([0.0, 1.0, 11.0, 12.0])
    truth = np.column_stack(
        (truth_magnitude, np.zeros((4, 2), dtype=np.float64))
    )
    prediction = np.column_stack(
        (prediction_magnitude, np.zeros((4, 2), dtype=np.float64))
    )
    arrays = {
        "predicted_velocity_nd": prediction,
        "line_length_weights_in": np.ones(4, dtype=np.float64),
        "station_row_offsets": np.asarray([0, 2, 4], dtype=np.int64),
    }
    assert _velocity_r2(arrays, truth) == pytest.approx(-1.0, abs=2e-13)


def test_profile_r2_refuses_zero_target_variance() -> None:
    arrays = {
        "predicted_velocity_nd": np.ones((2, 3), dtype=np.float64),
        "line_length_weights_in": np.ones(2, dtype=np.float64),
        "station_row_offsets": np.asarray([0, 2], dtype=np.int64),
    }
    truth = np.ones((2, 3), dtype=np.float64)
    with pytest.raises(NativeProfileEvaluationError, match="variance is zero"):
        _velocity_r2(arrays, truth)
