#!/usr/bin/env python3
"""Regenerate AirfRANS dummy packages from the benchmark contracts.

The three-point profile arrays are intentionally abridged packaging fixtures.
They are permitted only because every generated submission is labelled
``approval.status=prototype``. Real submissions must follow the pinned
1,001-point profile definition in the AirfRANS benchmark specification.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scores import composite_component_group_scores, composite_overall_score
from scripts.prototype_methodologies import build_prototype_methodology


SUBMISSION_SPLITS = {
    "airfrans-pointnet-full": "full",
    "dummy-airfrans-airfoiloperator-v1": "full",
    "airfrans-graph-u-net-full": "scarce",
    "airfrans-graphsage-full": "reynolds_extrapolation",
    "airfrans-mlp-full": "aoa_extrapolation",
}
PROTOTYPE_COORDINATE_M = [0.0, 0.05, 0.1]
PROTOTYPE_PREDICTION = [1.0, 1.0, 1.0]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_series(panel: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "panel_id": panel["id"],
            "station_id": station_id,
            "quantity_id": quantity_id,
            "coordinate": PROTOTYPE_COORDINATE_M,
            "prediction": PROTOTYPE_PREDICTION,
        }
        for station_id in panel["station_ids"]
        for quantity_id in panel["quantity_ids"]
    ]


def main() -> None:
    specification = load_json(ROOT / "benchmark-specs" / "airfrans" / "submission-spec.json")
    methodology_contract = load_json(
        ROOT / "benchmark-specs" / "airfrans" / "methodology-contract.json"
    )
    split_entries = {entry["id"]: entry for entry in specification["splits"]}
    panel = next(item for item in specification["profile_panels"] if item["id"] == "velocity_profiles")
    expected_prototype_count = load_json(
        ROOT / "benchmark-specs" / "airfrans" / specification["profile_definition"]["file"]
    )["prototype_fixture_policy"]["abridged_sample_count"]
    if len(PROTOTYPE_COORDINATE_M) != expected_prototype_count:
        raise ValueError("prototype coordinate count differs from the profile definition")

    submissions_root = ROOT / "submissions" / "airfrans"
    actual_submission_ids = {
        path.name for path in submissions_root.iterdir() if (path / "submission.json").is_file()
    }
    if actual_submission_ids != set(SUBMISSION_SPLITS):
        raise ValueError(
            "SUBMISSION_SPLITS must exactly cover AirfRANS fixtures; "
            f"missing={sorted(actual_submission_ids - set(SUBMISSION_SPLITS))}, "
            f"unexpected={sorted(set(SUBMISSION_SPLITS) - actual_submission_ids)}"
        )

    for submission_id, split_id in SUBMISSION_SPLITS.items():
        directory = submissions_root / submission_id
        submission_path = directory / "submission.json"
        submission = load_json(submission_path)
        if submission.get("approval", {}).get("status") != "prototype":
            raise ValueError(f"refusing to create abridged profiles for non-prototype {submission_id}")
        if submission["split_id"] != split_id:
            raise ValueError(f"{submission_id} split_id does not match SUBMISSION_SPLITS")

        split_entry = split_entries[split_id]
        split_path = ROOT / "benchmark-specs" / "airfrans" / split_entry["index_file"]
        split = load_json(split_path)
        case_ids = split["case_ids"]
        if split["case_set_id"] != submission["case_set_id"]:
            raise ValueError(f"{submission_id} case_set_id does not match its benchmark split")

        profiles_directory = directory / "profiles"
        profiles_directory.mkdir(exist_ok=True)
        for stale_path in profiles_directory.glob("chunk-*.json"):
            stale_path.unlink()
        index_path = profiles_directory / "index.json"
        index_path.unlink(missing_ok=True)

        chunk_path = profiles_directory / "chunk-000.json"
        write_json(
            chunk_path,
            {
                "schema_version": "1.0",
                "cases": [
                    {
                        "case_id": case_id,
                        "series": profile_series(panel),
                    }
                    for case_id in case_ids
                ],
            },
        )
        write_json(
            index_path,
            {
                "schema_version": "1.0",
                "submission_id": submission_id,
                "dataset_id": "airfrans",
                "split_id": split_id,
                "case_set_id": split["case_set_id"],
                "case_count": len(case_ids),
                "chunks": [
                    {
                        "file": chunk_path.name,
                        "case_ids": case_ids,
                        "sha256": sha256_file(chunk_path),
                    }
                ],
            },
        )

        evidence_path = directory / submission["evaluation"]["evidence_file"]
        evidence = load_json(evidence_path)
        metrics = evidence["metric_values"]
        metrics.pop("cp_cut_r2", None)
        metrics.update(
            composite_component_group_scores(
                metrics,
                specification["overall_score_composite"],
                specification["component_score_groups"],
            )
        )
        metrics["overall_score"] = composite_overall_score(
            metrics,
            specification["overall_score_composite"],
        )
        evidence["profile_index_sha256"] = sha256_file(index_path)
        write_json(evidence_path, evidence)

        submission["split_sha256"] = sha256_file(split_path)
        submission["metric_values"] = metrics
        submission["evaluation"]["evidence_sha256"] = sha256_file(evidence_path)
        submission["profile_data"]["case_count"] = len(case_ids)
        submission["profile_data"]["case_set_id"] = split["case_set_id"]
        submission["profile_data"]["index_file"] = "profiles/index.json"
        submission["methodology"] = build_prototype_methodology(
            submission,
            methodology_contract,
        )
        write_json(submission_path, submission)
        print(f"Regenerated AirfRANS prototype fixture: {submission_id}")


if __name__ == "__main__":
    main()
