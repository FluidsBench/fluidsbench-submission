#!/usr/bin/env python3
"""Generate split-scoped DrivAerML force-truth statistics for prototype R2 rows.

The authoritative force table remains in the pinned public DrivAerML release.
This script records only the sufficient truth statistics needed to convert a
synthetic split-level RMSE into a mathematically consistent R2 value.  It does
not replace the casewise truth table used by the reference evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "benchmark-specs" / "drivaerml"
OUTPUT_PATH = DATASET_ROOT / "force-r2-truth-statistics.json"
EXPECTED_SHA256 = "4e9e003da38ccdcacad359451079888361eae221d3c8dad7fd5682250d257865"
EXPECTED_HEADER = ["run", "cd", "cl", "clf", "clr", "cs"]
EXPECTED_CASE_COUNT = 484
SOURCE_REPOSITORY = "neashton/drivaerml"
SOURCE_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_force_truth(path: Path) -> dict[str, dict[str, float]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"force table SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        )
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"unexpected force-table header: {reader.fieldnames!r}")
        for row in reader:
            run = int(row["run"])
            case_id = f"run_{run}"
            if case_id in rows:
                raise ValueError(f"duplicate force-table case: {case_id}")
            clf = float(row["clf"])
            clr = float(row["clr"])
            rows[case_id] = {
                "cd": float(row["cd"]),
                "cl": float(row["cl"]),
                "c_pitch": (clf - clr) / 2.0,
            }
    if len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"force table must contain {EXPECTED_CASE_COUNT} cases, got {len(rows)}"
        )
    return rows


def truth_statistics(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = math.fsum(values) / count
    squared_error_about_mean = math.fsum((value - mean) ** 2 for value in values)
    if squared_error_about_mean <= 0.0:
        raise ValueError("R2 truth support must have positive variance")
    return {
        "case_count": count,
        "truth_mean": mean,
        "truth_sum": math.fsum(values),
        "truth_squared_sum": math.fsum(value * value for value in values),
        "truth_sst": squared_error_about_mean,
        "truth_population_variance": squared_error_about_mean / count,
    }


def build_document(force_truth: dict[str, dict[str, float]]) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    split_paths = sorted((DATASET_ROOT / "splits").glob("*.json"))
    for split_path in split_paths:
        if split_path.name == "manifest.json":
            continue
        split = load_json(split_path)
        case_ids = split["case_ids"]
        missing = [case_id for case_id in case_ids if case_id not in force_truth]
        if missing:
            raise ValueError(f"{split_path.name} references missing force cases: {missing}")
        splits[split["split_id"]] = {
            "case_set_id": split["case_set_id"],
            "split_sha256": sha256_file(split_path),
            "targets": {
                target: truth_statistics([force_truth[case_id][target] for case_id in case_ids])
                for target in ("cd", "cl", "c_pitch")
            },
        }
    return {
        "schema_version": "1.0",
        "dataset_id": "drivaerml",
        "purpose": (
            "prototype_fixture_force_r2_truth_sufficient_statistics_only; "
            "official_evaluation_uses_casewise_authoritative_truth"
        ),
        "source": {
            "provider": "Hugging Face Hub",
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "file": "force_mom_constref_all.csv",
            "sha256": EXPECTED_SHA256,
            "case_count": EXPECTED_CASE_COUNT,
            "target_definitions": {
                "cd": "cd",
                "cl": "cl",
                "c_pitch": "(clf-clr)/2",
            },
        },
        "r2_equation": "1-squared_error_sum/truth_sst",
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("force_truth_csv", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    document = build_document(load_force_truth(args.force_truth_csv.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
