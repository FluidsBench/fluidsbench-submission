"""Compare a maintainer's evaluator replay with a submitted result.

This is a comparison helper, not an evaluator, approval command or badge writer.
Replay provenance and the official split/support must be reviewed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
REL_TOL = 1e-12
ABS_TOL = 1e-12
METRIC_CONTAINERS = {"metric_values", "nonspatial_metric_values", "metric_sufficient_statistics"}
EXACT_NUMBERS = {"entity_count", "support_count", "scored_count"}


class ComparisonError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ComparisonError(f"non-finite JSON number: {value}")


def read_object(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    document = json.loads(
        payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if not isinstance(document, dict):
        raise ComparisonError(f"expected a JSON object: {path}")
    return document, hashlib.sha256(payload).hexdigest()


def _validate_cases(document: dict[str, Any], label: str) -> None:
    schema, _ = read_object(ROOT / "schemas/v3/case-metrics.schema.json")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    error = next(validator.iter_errors(document), None)
    if error is not None:
        raise ComparisonError(f"{label}: {error.json_path}: {error.message}")
    cases = document["cases"]
    ids = [case["case_id"] for case in cases]
    if document["case_count"] != len(cases) or len(set(ids)) != len(ids):
        raise ComparisonError(f"{label}: case count or unique case identities differ")
    for case in cases:
        supports = case["supports"]
        names = [support["support_id"] for support in supports]
        if len(set(names)) != len(names):
            raise ComparisonError(f"{label}: duplicate supports in {case['case_id']}")
        for support in supports:
            if (
                support["support_count"] != support["scored_count"]
                or support["coverage_fraction"] != 1
                or support["weight_coverage_fraction"] != 1
                or support["unmapped_count"] != 0
            ):
                raise ComparisonError(f"{label}: incomplete support in {case['case_id']}")


def differences(expected: Any, actual: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Compare full evidence; only floating metric/statistic values have tolerance."""
    label = ".".join(path) or "$"
    if isinstance(expected, dict) and isinstance(actual, dict):
        errors = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected:
                errors.append(f"{label}.{key}: unexpected value")
            elif key not in actual:
                errors.append(f"{label}.{key}: missing value")
            else:
                errors.extend(differences(expected[key], actual[key], (*path, key)))
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{label}: length {len(actual)} differs from {len(expected)}"]
        return [
            error
            for index, (left, right) in enumerate(zip(expected, actual))
            for error in differences(left, right, (*path, str(index)))
        ]
    numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (expected, actual))
    if numeric:
        if not all(math.isfinite(value) for value in (expected, actual)):
            return [f"{label}: non-finite number"]
        tolerant = bool(METRIC_CONTAINERS.intersection(path)) and path[-1] not in EXACT_NUMBERS
        equal = math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL) if tolerant else expected == actual
    else:
        equal = type(expected) is type(actual) and expected == actual
    return [] if equal else [f"{label}: submitted {expected!r}, replay {actual!r}"]


def compare_replay(
    directory: Path,
    replay_path: Path,
    aggregate_path: Path | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    submission, submission_hash = read_object(directory / "submission.json")
    if submission.get("schema_version") != "3.0":
        raise ComparisonError("the comparison requires a schema-v3 submission")
    declaration = submission["case_metrics"]
    source_path = (directory / declaration["file"]).resolve()
    if not source_path.is_relative_to(directory):
        raise ComparisonError("submitted case-metrics path escapes its package")
    if replay_path.resolve() == source_path:
        raise ComparisonError("replay must be a separate evaluator output, not the submitted case file")
    if aggregate_path and aggregate_path.resolve() == (directory / "submission.json"):
        raise ComparisonError("replayed aggregates must not be the source submission")
    source, source_hash = read_object(source_path)
    if source_hash != declaration["sha256"]:
        raise ComparisonError("submitted case-metrics SHA-256 differs from its declaration")
    replay, replay_hash = read_object(replay_path)
    _validate_cases(source, "submitted cases")
    _validate_cases(replay, "replayed cases")
    identity = {
        key: submission[key]
        for key in ("submission_id", "dataset_id", "split_id", "case_set_id")
    }
    identity.update({
        "scoring_support_release_id": submission["scoring_support"]["release_id"],
        "scoring_support_manifest_sha256": submission["scoring_support"]["manifest_sha256"],
        "case_count": declaration["case_count"],
    })
    for key, value in identity.items():
        if source.get(key) != value:
            raise ComparisonError(f"submitted case evidence {key} differs from submission.json")
    source.pop("generated_at", None)
    replay.pop("generated_at", None)
    errors = differences(source, replay, ("case_metrics",))
    aggregate_document, aggregate_hash = read_object(aggregate_path) if aggregate_path else (replay, replay_hash)
    for key, value in identity.items():
        if key in aggregate_document:
            errors.extend(differences(value, aggregate_document[key], ("replayed_aggregates", key)))
    actual = aggregate_document.get("metric_values")
    expected = submission.get("metric_values")
    if not isinstance(expected, dict) or not expected or not isinstance(actual, dict):
        raise ComparisonError("submission and replay must contain a non-empty metric_values object")
    for label, values in (("submission", expected), ("replay", actual)):
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in values.values()):
            raise ComparisonError(f"{label} metric_values must be finite numbers")
    errors.extend(differences(expected, actual, ("metric_values",)))
    # A native evaluator may emit additional aggregate scores in a second file,
    # but its shared base metrics must also agree with its case report.
    common = replay["metric_values"].keys() & actual.keys()
    errors.extend(differences(
        {key: replay["metric_values"][key] for key in common},
        {key: actual[key] for key in common},
        ("replay_consistency", "metric_values"),
    ))
    return {
        "format": "fluidsbench-prediction-replay-comparison-v1",
        "status": "match" if not errors else "mismatch",
        "scope": "comparison_only_not_verification_or_approval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
        "submitted_metric_ids": sorted(expected),
        "replayed_metric_ids": sorted(actual),
        "input_sha256": {
            "submission": submission_hash,
            "submitted_case_metrics": source_hash,
            "replayed_case_metrics": replay_hash,
            "replayed_metric_values_document": aggregate_hash,
        },
        "tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "difference_count": len(errors),
        "differences": errors[:100],
        "differences_truncated": len(errors) > 100,
        "review_required": "Confirm independent evaluator provenance, official split/support, and all metric coverage before recording any maintainer check.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_directory", type=Path)
    parser.add_argument("--recomputed-case-metrics", type=Path, required=True)
    parser.add_argument("--recomputed-metrics", type=Path, help="separate evaluator JSON containing all metric_values, including derived scores")
    parser.add_argument("--output", type=Path, required=True, help="new comparison report outside the source submission")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve().is_relative_to(args.submission_directory.resolve()):
            raise ComparisonError("comparison output must be outside the source submission")
        report = compare_replay(args.submission_directory, args.recomputed_case_metrics, args.recomputed_metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"{report['status'].upper()}: {report['difference_count']} difference(s). Comparison only; no verification or approval recorded.")
    return 0 if report["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
