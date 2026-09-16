#!/usr/bin/env python3
"""Synchronize prototype overall scores with dataset composite declarations."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scores import composite_overall_score
from scripts.validate_submission import load_json, sha256_file


def write_json(path: Path, value: object) -> None:
    # newline="\n" keeps generated files byte-identical across platforms.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def metric_values_with_overall(metric_values: dict[str, float], score: float) -> dict[str, float]:
    return {"overall_score": score, **{key: value for key, value in metric_values.items() if key != "overall_score"}}


def main() -> int:
    updated_packages = 0
    for specification_path in sorted((ROOT / "benchmark-specs").glob("*/submission-spec.json")):
        specification = load_json(specification_path)
        declaration = specification.get("overall_score_composite")
        if not isinstance(declaration, dict):
            continue
        if declaration.get("status", "active") != "active":
            continue
        dataset_id = specification["dataset_id"]
        tolerance = float(declaration.get("tolerance", 1e-6))
        for submission_path in sorted((ROOT / "submissions" / dataset_id).glob("*/submission.json")):
            submission = load_json(submission_path)
            metric_values = submission["metric_values"]
            calculated = composite_overall_score(metric_values, declaration)
            existing = metric_values.get("overall_score")
            score = (
                float(existing)
                if isinstance(existing, (int, float))
                and not isinstance(existing, bool)
                and math.isclose(float(existing), calculated, rel_tol=0.0, abs_tol=tolerance)
                else calculated
            )
            synchronized_values = metric_values_with_overall(metric_values, score)

            evidence_path = submission_path.parent / submission["evaluation"]["evidence_file"]
            evidence = load_json(evidence_path)
            evidence_changed = evidence.get("metric_values") != synchronized_values
            if evidence_changed:
                evidence["metric_values"] = synchronized_values
                write_json(evidence_path, evidence)

            evidence_sha256 = sha256_file(evidence_path)
            submission_changed = (
                submission.get("metric_values") != synchronized_values
                or submission["evaluation"].get("evidence_sha256") != evidence_sha256
            )
            if submission_changed:
                submission["metric_values"] = synchronized_values
                submission["evaluation"]["evidence_sha256"] = evidence_sha256
                write_json(submission_path, submission)

            if evidence_changed or submission_changed:
                updated_packages += 1
                print(f"updated {submission_path.parent.relative_to(ROOT)}")

    print(f"updated {updated_packages} prototype packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
