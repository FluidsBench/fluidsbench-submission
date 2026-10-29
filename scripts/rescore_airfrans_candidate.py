#!/usr/bin/env python3
"""Regenerate the derived scores of an assembled AirfRANS schema-v3 package.

When ``benchmark-specs/airfrans/submission-spec.json`` changes its composite
weights (and ``evaluation_reference_version``), an existing package keeps
valid base metrics but carries stale derived values. This script recomputes
only the derived values from the package's own base metrics, with the same
``reference.scores`` functions ``scripts/finalize_airfrans_metrics.py`` uses,
and then rewrites the hashes that depend on them:

- ``metrics/cases.json``: the ``overall_score`` and group scores
  (``field_score``/``force_score``/``diagnostic_score``) in ``metric_values``;
- ``evaluation-evidence.json``: ``metric_values``, ``reference_version``,
  ``case_metrics_sha256``, ``command`` and ``notes``;
- ``submission.json``: ``metric_values``, ``evaluation.reference_version``,
  ``evaluation.command``, ``evaluation.evidence_sha256`` and
  ``case_metrics.sha256``.

The validator also requires the scoring-support manifest to carry the same
``evaluation_reference_version`` as the package. When that manifest has been
re-stamped for the new version, pass it with ``--scoring-support-manifest``:
its SHA-256 is then rebound as ``scoring_support_manifest_sha256`` in
``discretization.json``, ``metrics/cases.json`` and
``evaluation-evidence.json`` and as ``scoring_support.manifest_sha256`` in
``submission.json``, and the ``discretization.json`` hash is updated where it
is bound. The manifest must have the package's release ID and the spec's
evaluation reference version.

Every base metric, per-case record, profile, discretization record and
other binding is left byte-for-byte unchanged; the script checks this before
writing. It refuses to run on a package whose metric values or hashes are
not mutually consistent, because a rescore must not paper over a broken
package. No predictions or AirfRANS dependencies are needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scores import composite_component_group_scores, composite_overall_score  # noqa: E402

DEFAULT_SPEC = ROOT / "benchmark-specs" / "airfrans" / "submission-spec.json"
CASE_METRICS_FILE = "metrics/cases.json"
MANIFEST_SHA_KEY = "scoring_support_manifest_sha256"


class RescoreError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derived_metric_ids(spec: Mapping[str, Any]) -> list[str]:
    groups = [group["metric_id"] for group in spec["component_score_groups"]["groups"]]
    return [spec["overall_score_composite"]["metric_id"], *groups]


def derived_scores(metric_values: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, float]:
    overall_declaration = spec["overall_score_composite"]
    scores = {
        overall_declaration["metric_id"]: composite_overall_score(metric_values, overall_declaration)
    }
    scores.update(
        composite_component_group_scores(
            metric_values, overall_declaration, spec["component_score_groups"]
        )
    )
    return scores


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RescoreError(message)


def _append_once(text: str, addition: str, separator: str) -> str:
    return text if addition in text else f"{text}{separator}{addition}"


def _check_hash_chain(documents: Mapping[str, Any], raw: Mapping[str, bytes]) -> None:
    case_metrics, evidence, submission, discretization = (
        documents["case_metrics"],
        documents["evidence"],
        documents["submission"],
        documents["discretization"],
    )
    case_metrics_sha = sha256_bytes(raw["case_metrics"])
    discretization_sha = sha256_bytes(raw["discretization"])
    _require(
        evidence["case_metrics_sha256"] == case_metrics_sha,
        "evaluation-evidence.json case_metrics_sha256 does not match metrics/cases.json",
    )
    _require(
        submission["case_metrics"]["sha256"] == case_metrics_sha,
        "submission.json case_metrics.sha256 does not match metrics/cases.json",
    )
    _require(
        evidence["discretization_sha256"] == discretization_sha,
        "evaluation-evidence.json discretization_sha256 does not match discretization.json",
    )
    _require(
        submission["spatial_discretization"]["sha256"] == discretization_sha,
        "submission.json spatial_discretization.sha256 does not match discretization.json",
    )
    _require(
        submission["evaluation"]["evidence_sha256"] == sha256_bytes(raw["evidence"]),
        "submission.json evaluation.evidence_sha256 does not match evaluation-evidence.json",
    )
    manifest_shas = {
        case_metrics[MANIFEST_SHA_KEY],
        evidence[MANIFEST_SHA_KEY],
        discretization[MANIFEST_SHA_KEY],
        submission["scoring_support"]["manifest_sha256"],
    }
    _require(
        len(manifest_shas) == 1,
        "scoring-support manifest SHA-256 differs between the package documents",
    )
    _require(
        case_metrics["metric_values"] == evidence["metric_values"] == submission["metric_values"],
        "metric_values differ between metrics/cases.json, evaluation-evidence.json and submission.json",
    )
    for key in ("submission_id", "dataset_id", "split_id"):
        _require(
            case_metrics[key] == evidence[key] == submission[key],
            f"{key} differs between metrics/cases.json, evaluation-evidence.json and submission.json",
        )


def rescore_payloads(
    documents: Mapping[str, Any],
    raw: Mapping[str, bytes],
    spec: Mapping[str, Any],
    command: str,
    manifest_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return rescored copies of the four package documents.

    ``documents``/``raw`` hold ``case_metrics``, ``evidence``, ``submission``
    and ``discretization``, parsed and as on-disk bytes; the bytes are used to
    check the existing hash chain before anything is changed.
    """

    _check_hash_chain(documents, raw)
    version = spec["evaluation_reference_version"]
    submission = documents["submission"]

    manifest_sha = submission["scoring_support"]["manifest_sha256"]
    if manifest_bytes is not None:
        manifest = json.loads(manifest_bytes)
        _require(
            manifest.get("release_id") == submission["scoring_support"]["release_id"],
            f"scoring-support manifest release_id {manifest.get('release_id')!r} does not match "
            f"the package's {submission['scoring_support']['release_id']!r}",
        )
        _require(
            manifest.get("dataset_id") == submission["dataset_id"],
            "scoring-support manifest dataset_id does not match the package",
        )
        _require(
            manifest.get("evaluation_reference_version") == version,
            "scoring-support manifest evaluation_reference_version "
            f"{manifest.get('evaluation_reference_version')!r} must equal the spec's {version!r}",
        )
        manifest_sha = sha256_bytes(manifest_bytes)

    derived_ids = derived_metric_ids(spec)
    missing = [
        metric_id
        for metric_id in derived_ids
        if metric_id not in documents["case_metrics"]["metric_values"]
    ]
    _require(not missing, f"package has no derived metric(s) {missing}")
    scores = derived_scores(documents["case_metrics"]["metric_values"], spec)
    for metric_id, value in scores.items():
        _require(math.isfinite(value), f"derived {metric_id} is not finite ({value})")

    new_discretization = copy.deepcopy(documents["discretization"])
    new_discretization[MANIFEST_SHA_KEY] = manifest_sha
    new_discretization_sha = sha256_bytes(dump_json(new_discretization))

    new_case_metrics = copy.deepcopy(documents["case_metrics"])
    new_case_metrics["metric_values"].update(scores)
    new_case_metrics[MANIFEST_SHA_KEY] = manifest_sha
    metric_values = new_case_metrics["metric_values"]
    new_case_metrics_sha = sha256_bytes(dump_json(new_case_metrics))

    evidence = documents["evidence"]
    new_evidence = copy.deepcopy(evidence)
    new_evidence["metric_values"] = copy.deepcopy(metric_values)
    new_evidence["reference_version"] = version
    new_evidence["case_metrics_sha256"] = new_case_metrics_sha
    new_evidence["discretization_sha256"] = new_discretization_sha
    new_evidence[MANIFEST_SHA_KEY] = manifest_sha
    new_evidence["command"] = _append_once(evidence["command"], command, " ; ")
    notes = _append_once(
        evidence.get("notes", ""),
        f"Derived group and overall scores regenerated under {version} from the unchanged base metrics.",
        " ",
    )
    if manifest_bytes is not None:
        notes = _append_once(
            notes,
            f"Scoring-support manifest SHA-256 rebound to the manifest stamped with {version}.",
            " ",
        )
    new_evidence["notes"] = notes.strip()

    new_submission = copy.deepcopy(submission)
    new_submission["metric_values"] = copy.deepcopy(metric_values)
    new_submission["evaluation"]["reference_version"] = version
    new_submission["evaluation"]["command"] = _append_once(
        submission["evaluation"]["command"], command, " ; "
    )
    new_submission["evaluation"]["evidence_sha256"] = sha256_bytes(dump_json(new_evidence))
    new_submission["case_metrics"]["sha256"] = new_case_metrics_sha
    new_submission["spatial_discretization"]["sha256"] = new_discretization_sha
    new_submission["scoring_support"]["manifest_sha256"] = manifest_sha

    _check_only_expected_changes(documents["case_metrics"], new_case_metrics, derived_ids)
    _check_only_expected_changes(documents["discretization"], new_discretization, [])
    return {
        "case_metrics": new_case_metrics,
        "evidence": new_evidence,
        "submission": new_submission,
        "discretization": new_discretization,
    }


def _check_only_expected_changes(
    before: Mapping[str, Any], after: Mapping[str, Any], derived_ids: list[str]
) -> None:
    def strip(document: Mapping[str, Any]) -> dict[str, Any]:
        stripped = copy.deepcopy(dict(document))
        stripped.pop(MANIFEST_SHA_KEY, None)
        for metric_id in derived_ids:
            stripped["metric_values"].pop(metric_id, None)
        return stripped

    _require(
        strip(before) == strip(after),
        "rescore changed something other than derived scores and the manifest binding",
    )


def rescore_package(
    package_dir: Path,
    spec: Mapping[str, Any],
    *,
    write: bool,
    scoring_support_manifest: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "case_metrics": package_dir / CASE_METRICS_FILE,
        "evidence": package_dir / "evaluation-evidence.json",
        "submission": package_dir / "submission.json",
        "discretization": package_dir / "discretization.json",
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    for name, data in raw.items():
        _require(
            dump_json(json.loads(data)) == data,
            f"{paths[name]} is not in canonical indent=2 form; refusing to rewrite it",
        )
    documents = {name: json.loads(data) for name, data in raw.items()}
    try:
        package_path = package_dir.resolve().relative_to(ROOT)
    except ValueError:
        package_path = package_dir
    command = f"python3 scripts/rescore_airfrans_candidate.py --package-dir {package_path.as_posix()}"
    manifest_bytes = None
    if scoring_support_manifest is not None:
        manifest_bytes = scoring_support_manifest.read_bytes()
        release_id = documents["submission"]["scoring_support"]["release_id"]
        command += f" --scoring-support-manifest <scoring-support>/{release_id}/manifest.json"

    new_documents = rescore_payloads(documents, raw, spec, command, manifest_bytes)
    new_raw = {name: dump_json(new_documents[name]) for name in paths}
    changed = [name for name in paths if new_raw[name] != raw[name]]
    if write:
        for name in changed:
            paths[name].write_bytes(new_raw[name])

    derived_ids = derived_metric_ids(spec)
    return {
        "package": str(package_path),
        "reference_version": spec["evaluation_reference_version"],
        "changed_files": [str(paths[name].relative_to(package_dir)) for name in changed],
        "written": write and bool(changed),
        "derived_before": {k: documents["case_metrics"]["metric_values"][k] for k in derived_ids},
        "derived_after": {k: new_documents["case_metrics"]["metric_values"][k] for k in derived_ids},
        "scoring_support_manifest_sha256": new_documents["submission"]["scoring_support"][
            "manifest_sha256"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--scoring-support-manifest",
        type=Path,
        help="Re-stamped scoring-support manifest whose SHA-256 the package should bind.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would change without writing; exit 1 if the package is stale.",
    )
    args = parser.parse_args(argv)

    spec = load_json(args.spec)
    try:
        report = rescore_package(
            args.package_dir,
            spec,
            write=not args.check,
            scoring_support_manifest=args.scoring_support_manifest,
        )
    except (RescoreError, KeyError) as error:
        print(f"error: {args.package_dir}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 1 if args.check and report["changed_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
