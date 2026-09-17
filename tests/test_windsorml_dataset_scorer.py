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


def case_evidence(
    case_id: str,
    *,
    index: int,
    field_error: float,
    force_error: float,
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
        "forces": {
            "truth_integrated": {"cd": cd_truth, "cl": cl_truth, "cs": 0.0},
            "prediction_integrated": {
                "cd": cd_truth + force_error,
                "cl": cl_truth + force_error,
                "cs": 0.0,
            },
        },
    }


def write_evidence(root: Path, *, field_error: float, force_error: float) -> list[str]:
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
        self.assertAlmostEqual(values["overall_score"], 100.0, places=9)
        self.assertAlmostEqual(values["field_score"], 100.0, places=9)
        self.assertAlmostEqual(values["force_score"], 100.0, places=9)

    def test_degraded_evidence_scores_below_perfect(self) -> None:
        write_evidence(self.root, field_error=5.0, force_error=0.01)
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
