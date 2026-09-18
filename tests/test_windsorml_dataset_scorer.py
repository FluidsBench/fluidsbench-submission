"""WindsorML split-level reduction tests."""

from __future__ import annotations

import json
from copy import deepcopy
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
from reference.windsorml.profiles import load_profile_support  # noqa: E402

SPEC = REPO_ROOT / "benchmark-specs" / "windsorml" / "submission-spec.json"
SPLIT_ID = "full"


def profile_block(case_id: str, profile_error: float) -> dict:
    """Perturb the real pinned truth on every required station and sample."""

    support = load_profile_support(case_id=case_id)
    families = {}
    for family in support.definition["families"]:
        family_id = family["family_id"]
        families[family_id] = []
        for station in family["station_ids"]:
            frozen = support.document["families"][family_id][station]
            key = (
                "truth_ux_over_uinf"
                if family["quantity_id"] == "ux_over_uinf"
                else "truth_cp"
            )
            truth = list(frozen[key])
            families[family_id].append(
                {
                    "station_id": station,
                    "quantity_id": family["quantity_id"],
                    "sample_count": len(truth),
                    "source": "evaluator_derived_from_complete_native_fields",
                    "coordinate": list(frozen["coordinate"]),
                    "truth": truth,
                    "prediction": [value + profile_error for value in truth],
                }
            )
    return {
        **support.evidence_binding(),
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
        "profiles": profile_block(case_id, profile_error),
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
    split = json.loads((SPEC.parent / "splits" / f"{SPLIT_ID}.json").read_text())
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
        write_evidence(self.root, field_error=5.0, force_error=0.01, profile_error=0.02)
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
            for family in (
                "windsorml_velocity_relative_v1",
                "windsorml_cp_relative_v1",
            ):
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

    def test_incomplete_or_modified_profile_evidence_is_rejected(self) -> None:
        """The former one-station/two-sample 100/100 bypass must fail closed."""

        case_ids = write_evidence(self.root, field_error=0.0, force_error=0.0)
        path = self.root / f"{case_ids[0]}.json"
        baseline = json.loads(path.read_text())
        family = "windsorml_velocity_constant_v1"

        def truncate(profiles):
            for stations in profiles["families"].values():
                del stations[1:]
                for key in ("truth", "prediction", "coordinate"):
                    stations[0][key] = stations[0][key][:2]
                stations[0]["sample_count"] = 2

        mutations = {
            "former perfect-score bypass": truncate,
            "missing station": lambda p: p["families"][family].pop(),
            "duplicate station": lambda p: p["families"][family].__setitem__(
                1, deepcopy(p["families"][family][0])
            ),
            "unknown station": lambda p: p["families"][family][0].update(
                station_id="unknown"
            ),
            "missing family": lambda p: p["families"].pop("windsorml_cp_relative_v1"),
            "extra family": lambda p: p["families"].update(unknown=[]),
            "truncated prediction": lambda p: p["families"][family][0][
                "prediction"
            ].pop(),
            "truncated truth": lambda p: p["families"][family][0]["truth"].pop(),
            "truncated coordinate": lambda p: p["families"][family][0][
                "coordinate"
            ].pop(),
            "wrong sample count": lambda p: p["families"][family][0].update(
                sample_count=2
            ),
            "wrong quantity": lambda p: p["families"][family][0].update(
                quantity_id="cp"
            ),
            "wrong source": lambda p: p["families"][family][0].update(
                source="participant"
            ),
            "changed coordinate": lambda p: p["families"][family][0][
                "coordinate"
            ].__setitem__(0, -10.0),
            "changed truth": lambda p: p["families"][family][0]["truth"].__setitem__(
                0, 9.9
            ),
            "boolean prediction": lambda p: p["families"][family][0][
                "prediction"
            ].__setitem__(0, True),
            "overflowing prediction": lambda p: p["families"][family][0][
                "prediction"
            ].__setitem__(0, 10**400),
            "missing case pin": lambda p: p.pop("case_support_sha256"),
            "wrong case pin": lambda p: p.update(case_support_sha256="0" * 64),
            "wrong manifest pin": lambda p: p.update(
                profile_support_manifest_sha256="0" * 64
            ),
            "wrong definition pin": lambda p: p.update(
                profile_definition_sha256="0" * 64
            ),
            "wrong source pin": lambda p: p.update(source_identity_sha256="0" * 64),
            "wrong case": lambda p: p.update(case_id="run_0"),
            "wrong schema": lambda p: p.update(
                support_schema="windsorml-profile-support-v2"
            ),
            "wrong body height": lambda p: p.update(body_height_m=0.1),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                evidence = deepcopy(baseline)
                mutate(evidence["profiles"])
                path.write_text(json.dumps(evidence))
                with self.assertRaises(WindsorMLDatasetScorerError):
                    score_candidate_dataset(
                        submission_specification=SPEC,
                        split_id=SPLIT_ID,
                        case_evidence_directory=self.root,
                    )

    def test_station_order_does_not_change_scores(self) -> None:
        write_evidence(self.root, field_error=0.0, force_error=0.0, profile_error=0.02)
        arguments = dict(
            submission_specification=SPEC,
            split_id=SPLIT_ID,
            case_evidence_directory=self.root,
        )
        before = score_candidate_dataset(**arguments).to_json()["metric_values"]
        for path in self.root.glob("*.json"):
            evidence = json.loads(path.read_text())
            for stations in evidence["profiles"]["families"].values():
                stations.reverse()
            path.write_text(json.dumps(evidence))
        self.assertEqual(
            before, score_candidate_dataset(**arguments).to_json()["metric_values"]
        )

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
