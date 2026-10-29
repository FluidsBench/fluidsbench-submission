#!/usr/bin/env python3
"""Finalize AirfRANS schema-v3 case metrics (Layer 3 orchestration).

All of the actual point-wise and force metric math -- relative-L2/L1
reductions, per-geometry-then-macro-average, and the all-test-cases R2/MAE
aggregation cd_r2/cl_r2/c_drag_mae/c_lift_mae need -- already exists as
dataset-agnostic code in ``reference/evaluate_predictions.py`` and
``reference/scores.py``. This script owns exactly the two things that generic
pipeline cannot know on its own:

1. ``velocity_profile_r2`` is not part of the ScoringSupport system at all --
   it comes from the separate, already-working boundary-layer profile
   extraction/scoring pipeline (``examples/airfrans-profile-extraction/
   extract.py`` + ``score.py``), so this script reads its output
   (``profile-score.json``) and folds the one number in.
2. The four composite scores (``overall_score``, ``field_score``,
   ``force_score``, ``diagnostic_score``) are computed by
   ``reference.scores.composite_overall_score``/
   ``composite_component_group_scores`` from
   ``benchmark-specs/airfrans/submission-spec.json``'s own
   ``overall_score_composite``/``component_score_groups`` declarations --
   fully generic, but something has to call them with the complete metric
   set (point-wise + force + profile) assembled.

This script needs no PyVista/VTK/airfrans -- only jsonschema/numpy, like the
rest of the schema-v3 validator tooling (see requirements.txt, not
requirements-airfrans-evaluator.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.evaluate_predictions import evaluate_prediction_artifact  # noqa: E402
from reference.scores import composite_component_group_scores, composite_overall_score  # noqa: E402
from reference.scoring_support import ScoringSupportError  # noqa: E402

DEFAULT_SPEC = ROOT / "benchmark-specs" / "airfrans" / "submission-spec.json"


class FinalizeError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def apply_profile_score_and_composites(
    case_metrics: dict[str, Any],
    profile_score: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Fold velocity_profile_r2 into metric_values and compute composite scores.

    Mutates and returns ``case_metrics``. Split out from ``finalize`` so it can
    be unit tested against small synthetic fixtures without needing the full
    ScoringSupport manifest/index/chunk machinery.
    """

    if profile_score.get("metric_id") != "velocity_profile_r2":
        raise FinalizeError(
            "profile score is not a velocity_profile_r2 score file "
            f"(metric_id={profile_score.get('metric_id')!r})"
        )
    if profile_score.get("case_coverage") != "complete_official_split":
        raise FinalizeError(
            "profile score does not cover the complete official split "
            f"(case_coverage={profile_score.get('case_coverage')!r})"
        )
    if profile_score.get("case_count") != case_metrics["case_count"]:
        raise FinalizeError(
            "profile score case_count "
            f"({profile_score.get('case_count')}) does not match the scored "
            f"case-metrics case_count ({case_metrics['case_count']})"
        )
    case_metrics["metric_values"]["velocity_profile_r2"] = float(profile_score["value"])

    overall_declaration = spec["overall_score_composite"]
    group_declaration = spec["component_score_groups"]
    case_metrics["metric_values"]["overall_score"] = composite_overall_score(
        case_metrics["metric_values"], overall_declaration
    )
    case_metrics["metric_values"].update(
        composite_component_group_scores(
            case_metrics["metric_values"], overall_declaration, group_declaration
        )
    )
    return case_metrics


def finalize(
    *,
    support_manifest_path: Path,
    case_set_id: str,
    prediction_manifest_path: Path,
    submission_id: str,
    split_id: str,
    profile_score_path: Path,
    spec_path: Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    case_metrics = evaluate_prediction_artifact(
        support_manifest_path=support_manifest_path,
        case_set_id=case_set_id,
        prediction_manifest_path=prediction_manifest_path,
        submission_id=submission_id,
        split_id=split_id,
    )
    profile_score = load_json(profile_score_path)
    spec = load_json(spec_path)
    return apply_profile_score_and_composites(case_metrics, profile_score, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--case-set", required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--profile-score", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        result = finalize(
            support_manifest_path=args.support_manifest,
            case_set_id=args.case_set,
            prediction_manifest_path=args.prediction_manifest,
            submission_id=args.submission_id,
            split_id=args.split_id,
            profile_score_path=args.profile_score,
            spec_path=args.spec,
        )
    except (ScoringSupportError, FinalizeError) as error:
        parser.error(str(error))

    write_json(args.output, result)

    print(f"Wrote {args.output}")
    print(f"case_count = {result['case_count']}")
    print("metric_values:")
    for key, value in result["metric_values"].items():
        print(f"  {key:45s} {value:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
