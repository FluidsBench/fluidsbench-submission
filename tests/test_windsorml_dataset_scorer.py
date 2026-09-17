"""WindsorML split-level reduction tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from reference.windsorml.dataset_scorer import (  # noqa: E402
    WindsorMLDatasetScorerError,
    score_candidate_dataset,
)

SPEC = REPO_ROOT / "benchmark-specs" / "windsorml" / "submission-spec.json"
SPLIT_ID = "full"


PROFILE_FAMILIES = {
    "windsorml_velocity_constant_v1": ("wake_vertical_x_0p05l", "ux_over_uinf"),
    "windsorml_velocity_relative_v1": ("wake_vertical_x_0p05l_relative", "ux_over_uinf"),
    "windsorml_cp_constant_v1": ("cp_centreline_upper", "cp"),
    "windsorml_cp_relative_v1": ("cp_centreline_upper_relative", "cp"),
}


def profile_block(index: int, profile_error: float) -> dict:
    """Synthetic profile evidence for all four placement families."""

    families = {}
    for family, (station, quantity) in PROFILE_FAMILIES.items():
        truth = [0.2 + 0.01 * index + 0.003 * s for s in range(8)]
        prediction = [value + profile_error for value in truth]
        families[family] = [
            {
                "station_id": station,
                "quantity_id": quantity,
                "sample_count": len(truth),
                "source": "evaluator_derived_from_complete_native_fields",
                "coordinate": [0.05 * s for s in range(8)],
                "truth": truth,
                "prediction": prediction,
            }
        ]
    return {
        "support_schema": "windsorml-profile-support-v2",
        "sample_count": 8,
        "body_height_m": 0.34342,
        "participant_profile_payload_accepted": False,
        "families": families,
    }


def case_evidence(
    case_id: str,
    *,
    index: int,
    field_error: float,
    force_error: float,
    profile_error: float = 0.0,
) -> dict:
    """One synthetic case evidence document.

    Truth coefficients vary with ``index`` so R^2 has a non-degenerate
    denominator; ``force_error`` offsets the prediction.
    """

    cd_truth = 0.25 + 0.001 * index
    cl_truth = 0.45 + 0.002 * index
    return {
        "schema": "windsorml-candidate-case-evaluation-v1",
        "schema_version": 1,
        "status": "non_ranked_development_evidence_not_official_submission",
        "official_submission_artifact": False,
        "case_id": case_id,
        "run_id": int(case_id.removeprefix("run_")),
        "surface": {
            "support_id": "windsorml_surface_native_points",
            "association": "PointData",
            "metrics": {
                "surface_pressure_rel_l2": field_error,
                "surface_wall_shear_rel_l2": field_error,
            },
        },
        "volume": {
            "support_id": "windsorml_volume_native_cells",
            "association": "CellData",
            "metrics": {
                "volume_velocity_rel_l2": field_error,
                "volume_pressure_rel_l2": field_error,
            },
        },
        "profiles": profile_block(index, profile_error),
        "forces": {
            "truth_integrated": {"cd": cd_truth, "cl": cl_truth, "cs": 0.0},
            "prediction_integrated": {
                "cd": cd_truth + force_error,
                "cl": cl_truth + force_error,
                "cs": 0.0,
            },
        },
    }


def write_evidence(
    root: Path, *, field_error: float, force_error: float, profile_error: float = 0.0
) -> list[str]:
    split = json.loads(
        (SPEC.parent / "splits" / f"{SPLIT_ID}.json").read_text()
    )
    case_ids = split["case_ids"]
    for index, case_id in enumerate(case_ids):
        (root / f"{case_id}.json").write_text(
            json.dumps(
                case_evidence(
                    case_id,
                    index=index,
                    field_error=field_error,
                    force_error=force_error,
                    profile_error=profile_error,
                )
            )
        )
    return case_ids


@unittest.skipIf(np is None, "requires NumPy")
class WindsorMLDatasetScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_perfect_evidence_scores_one_hundred(self) -> None:
        write_evidence(self.root, field_error=0.0, force_error=0.0)
        result = score_candidate_dataset(
            submission_specification=SPEC,
            split_id=SPLIT_ID,
            case_evidence_directory=self.root,
        ).to_json()
        values = result["metric_values"]
        self.assertAlmostEqual(values["cd_r2"], 1.0, places=12)
        self.assertAlmostEqual(values["cl_r2"], 1.0, places=12)
        self.assertAlmostEqual(values["surface_pressure_rel_l2"], 0.0, places=12)
        self.assertAlmostEqual(values["velocity_profile_r2"], 1.0, places=12)
        self.assertAlmostEqual(values["cp_cut_r2"], 1.0, places=12)
        self.assertAlmostEqual(values["velocity_profile_relative_r2"], 1.0, places=12)
        self.assertAlmostEqual(values["cp_cut_relative_r2"], 1.0, places=12)
        self.assertAlmostEqual(values["overall_score"], 100.0, places=9)
        self.assertAlmostEqual(values["field_score"], 100.0, places=9)
        self.assertAlmostEqual(values["force_score"], 100.0, places=9)
        self.assertAlmostEqual(values["diagnostic_score"], 100.0, places=9)

    def test_degraded_evidence_scores_below_perfect(self) -> None:
        write_evidence(
            self.root, field_error=5.0, force_error=0.01, profile_error=0.02
        )
        result = score_candidate_dataset(
            submission_specification=SPEC,
            split_id=SPLIT_ID,
            case_evidence_directory=self.root,
        ).to_json()
        values = result["metric_values"]
        self.assertLess(values["overall_score"], 100.0)
        self.assertGreater(values["overall_score"], 0.0)
        self.assertLess(values["cd_r2"], 1.0)
        self.assertAlmostEqual(values["surface_pressure_rel_l2"], 5.0, places=9)

    def test_split_binding_records_the_excluded_case(self) -> None:
        write_evidence(self.root, field_error=0.0, force_error=0.0)
        result = score_candidate_dataset(
            submission_specification=SPEC,
            split_id=SPLIT_ID,
            case_evidence_directory=self.root,
        ).to_json()
        self.assertEqual(result["case_count"], 35)
        self.assertEqual(result["official_case_count"], 36)
        self.assertEqual(result["excluded_case_ids"], ["run_354"])

    def test_relative_families_are_reported_but_never_weighted(self) -> None:
        """A degraded relative family must not move overall_score."""

        write_evidence(self.root, field_error=0.0, force_error=0.0)
        baseline = score_candidate_dataset(
            submission_specification=SPEC,
            split_id=SPLIT_ID,
            case_evidence_directory=self.root,
        ).to_json()["metric_values"]

        split = json.loads((SPEC.parent / "splits" / f"{SPLIT_ID}.json").read_text())
        for index, case_id in enumerate(split["case_ids"]):
            evidence = case_evidence(
                case_id, index=index, field_error=0.0, force_error=0.0
            )
            for family in ("windsorml_velocity_relative_v1", "windsorml_cp_relative_v1"):
                for station in evidence["profiles"]["families"][family]:
                    station["prediction"] = [v + 0.5 for v in station["truth"]]
            (self.root / f"{case_id}.json").write_text(json.dumps(evidence))

        degraded = score_candidate_dataset(
            submission_specification=SPEC,
            split_id=SPLIT_ID,
            case_evidence_directory=self.root,
        ).to_json()["metric_values"]

        self.assertLess(degraded["velocity_profile_relative_r2"], 1.0)
        self.assertLess(degraded["cp_cut_relative_r2"], 1.0)
        self.assertAlmostEqual(
            degraded["overall_score"], baseline["overall_score"], places=12
        )
        self.assertAlmostEqual(
            degraded["velocity_profile_r2"], baseline["velocity_profile_r2"], places=12
        )

    def test_profile_evidence_claiming_participant_payload_is_rejected(self) -> None:
        case_ids = write_evidence(self.root, field_error=0.0, force_error=0.0)
        path = self.root / f"{case_ids[0]}.json"
        evidence = json.loads(path.read_text())
        evidence["profiles"]["participant_profile_payload_accepted"] = True
        path.write_text(json.dumps(evidence))
        with self.assertRaises(WindsorMLDatasetScorerError) as caught:
            score_candidate_dataset(
                submission_specification=SPEC,
                split_id=SPLIT_ID,
                case_evidence_directory=self.root,
            )
        self.assertIn("evaluator-derived", str(caught.exception))

    def test_missing_case_evidence_is_rejected(self) -> None:
        case_ids = write_evidence(self.root, field_error=0.0, force_error=0.0)
        (self.root / f"{case_ids[0]}.json").unlink()
        with self.assertRaises(WindsorMLDatasetScorerError) as caught:
            score_candidate_dataset(
                submission_specification=SPEC,
                split_id=SPLIT_ID,
                case_evidence_directory=self.root,
            )
        self.assertIn("missing case evidence", str(caught.exception))

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaises(WindsorMLDatasetScorerError):
            score_candidate_dataset(
                submission_specification=SPEC,
                split_id="not_a_split",
                case_evidence_directory=self.root,
            )

    def test_every_declared_split_can_be_scored(self) -> None:
        """All eight families must reduce, not just the baseline test set."""

        specification = json.loads(SPEC.read_text())
        for entry in specification["splits"]:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                split = json.loads((SPEC.parent / entry["index_file"]).read_text())
                for index, case_id in enumerate(split["case_ids"]):
                    (root / f"{case_id}.json").write_text(
                        json.dumps(
                            case_evidence(
                                case_id, index=index, field_error=0.0, force_error=0.0
                            )
                        )
                    )
                result = score_candidate_dataset(
                    submission_specification=SPEC,
                    split_id=entry["id"],
                    case_evidence_directory=root,
                ).to_json()
                self.assertAlmostEqual(
                    result["metric_values"]["overall_score"], 100.0, places=9
                )
                self.assertEqual(result["case_count"], entry["case_count"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
