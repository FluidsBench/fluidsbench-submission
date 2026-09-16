from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reference.ahmedml.contract import (
    PROFILE_DEFINITION_SHA256,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
)
from reference.ahmedml.dataset_scorer import (
    AhmedMLDatasetScorerError,
    score_candidate_dataset,
)
from reference.ahmedml.evaluator import EVIDENCE_SCHEMA, EVIDENCE_STATUS
from reference.ahmedml.support import SURFACE_STATIONS, VOLUME_STATIONS


ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION = ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json"


def _profile_series(case_index: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    identities = (
        *(("pressure_profiles", station, "cp") for station in SURFACE_STATIONS),
        *(("velocity_profiles", station, "ux_over_uinf") for station in VOLUME_STATIONS),
    )
    for series_index, (panel_id, station_id, quantity_id) in enumerate(identities):
        if panel_id == "pressure_profiles":
            start, stop = 0.0, 1.0
        elif station_id == "wake_lateral_x_0p50_l_z_0p50_h":
            start, stop = -1.0, 1.0
        else:
            start, stop = 0.0, 2.0
        coordinate = np.linspace(start, stop, 128, dtype=np.float64).tolist()
        truth = [
            0.01 * case_index + 0.1 * series_index + value
            for value in coordinate
        ]
        result.append(
            {
                "panel_id": panel_id,
                "station_id": station_id,
                "quantity_id": quantity_id,
                "coordinate": coordinate,
                "truth": truth,
                "prediction": list(truth),
                "sample_count": 128,
                "source": "evaluator_derived_from_complete_native_fields",
            }
        )
    return result


def _case_evidence(case_id: str, case_index: int) -> dict[str, object]:
    value = 0.1 + 0.002 * case_index
    return {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": 1,
        "status": EVIDENCE_STATUS,
        "official_submission": False,
        "leaderboard_eligible": False,
        "case_id": case_id,
        "source": {
            "repository_revision": REPOSITORY_REVISION,
            "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        },
        "coverage": {
            "complete_case": True,
            "gap_free_duplicate_free": True,
        },
        "metric_values": {
            "surface_pressure_rel_l2": 0.0,
            "surface_wall_shear_rel_l2": 0.0,
            "volume_velocity_rel_l2": 0.0,
            "volume_pressure_rel_l2": 0.0,
        },
        "force_coefficients": {
            "truth_cd": value,
            "prediction_cd": value,
            "truth_cl": -0.5 * value,
            "prediction_cl": -0.5 * value,
        },
        "profiles": {
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
            "participant_profile_payload_accepted": False,
            "series": _profile_series(case_index),
        },
    }


def _write_full_evidence(root: Path) -> list[str]:
    split = json.loads(
        (ROOT / "benchmark-specs" / "ahmedml" / "splits" / "full.json").read_text(
            encoding="utf-8"
        )
    )
    case_ids = split["case_ids"]
    root.mkdir()
    for index, case_id in enumerate(case_ids):
        (root / f"{case_id}.json").write_text(
            json.dumps(_case_evidence(case_id, index)), encoding="utf-8"
        )
    return case_ids


def test_perfect_complete_full_split_scores_one_hundred(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    case_ids = _write_full_evidence(evidence)
    result = score_candidate_dataset(
        submission_specification=SPECIFICATION,
        split_id="full",
        case_evidence_directory=evidence,
    ).to_json()

    assert result["case_ids"] == case_ids
    assert result["case_count"] == 50
    assert result["leaderboard_eligible"] is False
    metrics = result["metric_values"]
    assert metrics["cd_r2"] == pytest.approx(1.0)
    assert metrics["cl_r2"] == pytest.approx(1.0)
    assert metrics["cp_cut_r2"] == pytest.approx(1.0)
    assert metrics["velocity_profile_r2"] == pytest.approx(1.0)
    assert metrics["overall_score"] == pytest.approx(100.0)


def test_rejects_case_evidence_that_claims_ranking_eligibility(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    case_ids = _write_full_evidence(evidence)
    path = evidence / f"{case_ids[0]}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["leaderboard_eligible"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AhmedMLDatasetScorerError, match="evidence header differs"):
        score_candidate_dataset(
            submission_specification=SPECIFICATION,
            split_id="full",
            case_evidence_directory=evidence,
        )


def test_rejects_a_monotonic_but_noncanonical_profile_grid(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    case_ids = _write_full_evidence(evidence)
    path = evidence / f"{case_ids[0]}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    coordinate = document["profiles"]["series"][0]["coordinate"]
    coordinate[64] = 0.5 * (coordinate[64] + coordinate[65])
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AhmedMLDatasetScorerError, match="exact 128-point grid"):
        score_candidate_dataset(
            submission_specification=SPECIFICATION,
            split_id="full",
            case_evidence_directory=evidence,
        )
