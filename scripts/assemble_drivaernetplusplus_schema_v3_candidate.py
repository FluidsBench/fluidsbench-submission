#!/usr/bin/env python3
"""Assemble a genuine-inference DrivAerNet++ schema-v3 candidate package.

Consumes a completed ``scripts/drivaernetplusplus.py evaluate-entry`` result,
re-reduces the split from its per-case evidence, writes evaluator-extracted
Cp profile chunks, and binds participant methodology, checkpoint bytes and
discretization records.  Fail-closed; it creates neither owner approval nor a
leaderboard claim.  Normally run through ``scripts/drivaernetplusplus.py
package``, which also verifies the result with the dev-intake checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaernetplusplus.contract import (  # noqa: E402
    DATASET_ID,
    DATASET_NAME,
    DATASET_VERSION,
    SPEC_ROOT,
    SUPPORT_ID,
    ContractError,
    load_specification,
    official_split,
    parse_json_bytes,
    read_json,
    sha256_file,
)
from reference.drivaernetplusplus.dataset_scorer import (  # noqa: E402
    MACRO_METRICS,
    score_candidate_dataset,
)
from reference.drivaernetplusplus.native import PROFILE_STATIONS  # noqa: E402
from reference.drivaernetplusplus.pre_release import GENUINE_INFERENCE_ATTESTATION  # noqa: E402
from reference.methodology import (  # noqa: E402
    MethodologyError,
    derived_parameter_count_millions,
    require_methodology,
)
from scripts.validate_scoring_supports import validate_candidate_manifest_release  # noqa: E402

CONFIG_SCHEMA = "drivaernetplusplus-fluidsbench-schema-v3-package-config-v1"
PROFILE_FORMAT = "fluidsbench-profile-chunks-v1"
PROFILE_PANEL = "pressure_profiles"
PREDICTION_SCOPE = "surface_only"
SCHEMA_ROOT = ROOT / "schemas"
TOKEN_PREFIX = "__REPLACE_"
SHA256_LENGTH = 64
ATTESTATION = {
    "actual_model_inference": True,
    "evaluation_truth_generated_predictions": False,
    "scope": "all_declared_cases_and_native_surface_points",
}
CONFIG_KEYS = {
    "schema",
    "split_id",
    "profile_cases_per_chunk",
    "inference_attestation",
    "participant",
    "evaluation",
    "spatial_discretization",
    "release_bindings",
}
FORBIDDEN_PARTICIPANT_KEYS = {
    "$schema",
    "schema_version",
    "dataset",
    "dataset_id",
    "dataset_version",
    "split",
    "split_id",
    "case_set_id",
    "split_sha256",
    "prediction_scope",
    "parameter_count_millions",
    "evaluation",
    "scoring_support",
    "spatial_discretization",
    "case_metrics",
    "metric_values",
    "profile_data",
    "approval",
    "record_type",
}


class PackageAssemblyError(ValueError):
    """Raised when a DrivAerNet++ package cannot be assembled honestly."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def unresolved_tokens(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        return [found for key in sorted(value) for found in unresolved_tokens(value[key], f"{path}.{key}")]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in unresolved_tokens(item, f"{path}[{index}]")]
    return [path] if isinstance(value, str) and value.startswith(TOKEN_PREFIX) else []


def _require_schema(value: Any, relative_path: str, label: str) -> None:
    schema = read_json(SCHEMA_ROOT / relative_path, "schema")
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise PackageAssemblyError(
            f"{label} does not satisfy {relative_path}: "
            + "; ".join(f"{list(e.absolute_path)}: {e.message}" for e in errors[:5])
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageAssemblyError(f"{label} must be an object")
    return value


def _release_bindings(config: Mapping[str, Any], specification: Mapping[str, Any]) -> tuple[dict, dict]:
    release = _mapping(config.get("release_bindings"), "release_bindings")
    if set(release) != {"candidate_manifest", "evaluator", "profile_ground_truth"}:
        raise PackageAssemblyError(
            "release_bindings keys must be candidate_manifest, evaluator and profile_ground_truth"
        )
    scoring_support = _mapping(specification.get("scoring_support"), "specification.scoring_support")
    if (
        not (specification.get("status") == "official"
             and scoring_support.get("status") == "official"
             and scoring_support.get("submissions_open") is True)
        and (specification.get("status") != "candidate_native_support"
             or scoring_support.get("status") != "owner_review_required"
             or scoring_support.get("submissions_open") is not False)
    ):
        raise PackageAssemblyError("the DrivAerNet++ specification is not the expected closed candidate")
    owner = _mapping(scoring_support.get("candidate_manifest"), "specification candidate manifest")
    candidate = dict(_mapping(release["candidate_manifest"], "candidate_manifest"))
    expected = {key: owner.get(key) for key in ("status", "release_id", "manifest_url", "manifest_sha256")}
    if candidate != expected:
        raise PackageAssemblyError("candidate_manifest does not match the repository binding; refresh package-config.json")
    if dict(release["evaluator"]) != {"reference_version": specification.get("evaluation_reference_version")}:
        raise PackageAssemblyError("evaluator identity differs from the specification")
    truth = read_json(ROOT / "leaderboard" / "manifest.json")["data_release"]["profile_ground_truth"]
    ground_truth = dict(_mapping(release["profile_ground_truth"], "profile_ground_truth"))
    if ground_truth != {"release_id": truth["release_id"], "manifest_sha256": truth["manifest_sha256"]}:
        raise PackageAssemblyError("profile_ground_truth differs from the dev release binding")
    return candidate, ground_truth


def _verify_checkpoints(methodology: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    declared = {}
    for checkpoint in methodology.get("checkpoints") or []:
        identifier, digest = checkpoint.get("id"), checkpoint.get("sha256")
        if not isinstance(identifier, str) or identifier in declared or not isinstance(digest, str) or len(digest) != SHA256_LENGTH:
            raise PackageAssemblyError("methodology checkpoint identity is invalid")
        declared[identifier] = digest
    if not declared:
        raise PackageAssemblyError("methodology must declare at least one checkpoint")
    if set(paths) != set(declared):
        raise PackageAssemblyError(
            "--checkpoint IDs must exactly cover methodology checkpoints; "
            f"missing={sorted(set(declared) - set(paths))}, unexpected={sorted(set(paths) - set(declared))}"
        )
    for identifier, expected in declared.items():
        path = paths[identifier]
        if path.is_symlink() or not path.is_file():
            raise PackageAssemblyError(f"checkpoint {identifier!r} is not a regular file")
        if sha256_file(path) != expected:
            raise PackageAssemblyError(f"checkpoint {identifier!r} bytes differ from methodology.sha256")


def _discretization_cases(source: Path, destination: Path, *, submission_id: str, split_id: str,
                          case_ids: Sequence[str], counts: Mapping[str, int]) -> str:
    try:
        records = [parse_json_bytes(line.encode(), "discretization case") for line in source.read_text().splitlines() if line.strip()]
    except (OSError, ContractError) as error:
        raise PackageAssemblyError(f"cannot read discretization cases: {error}") from error
    if [record.get("case_id") for record in records] != list(case_ids):
        raise PackageAssemblyError("discretization cases must match the exact ordered split case list")
    encoded = []
    for record in records:
        identity = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": DATASET_ID,
            "split_id": split_id,
        }
        for key, value in identity.items():
            if key in record and record[key] != value:
                raise PackageAssemblyError(f"discretization case {record['case_id']}.{key} conflicts with the package")
        record = {**identity, **record}
        mappings = record.get("inference", {}).get("mappings", [])
        native = counts[record["case_id"]]
        if [m.get("support_id") for m in mappings] != [SUPPORT_ID] or any(
            (m.get("support_count"), m.get("scored_count"), m.get("unmapped_count")) != (native, native, 0) for m in mappings
        ):
            raise PackageAssemblyError(
                f"{record['case_id']}: the mapping must cover all {native} native points of {SUPPORT_ID}"
            )
        _require_schema(record, "v3/discretization-case.schema.json", f"discretization case {record['case_id']}")
        encoded.append(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"\n".join(encoded) + b"\n")
    return sha256_file(destination)


def _case_metric_record(evidence: Mapping[str, Any]) -> dict[str, Any]:
    n = evidence["support"]["entity_count"]
    return {
        "case_id": evidence["case_id"],
        "supports": [
            {
                "support_id": SUPPORT_ID,
                "support_count": n,
                "scored_count": n,
                "coverage_fraction": 1.0,
                "weight_coverage_fraction": 1.0,
                "unmapped_count": 0,
                "extrapolated_count": 0,
                "metric_values": {metric: evidence["metric_values"][metric] for metric in MACRO_METRICS},
                "metric_sufficient_statistics": copy.deepcopy(evidence["metric_sufficient_statistics"]),
            }
        ],
    }


def assemble_package(
    *,
    config_path: Path,
    evaluation_root: Path,
    entry: Mapping[str, Any],
    discretization_cases_path: Path,
    checkpoint_paths: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    config = read_json(config_path, "package config")
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS or config.get("schema") != CONFIG_SCHEMA:
        raise PackageAssemblyError(f"package config must use schema {CONFIG_SCHEMA!r} and only keys {sorted(CONFIG_KEYS)}")
    blockers = unresolved_tokens(config)
    if blockers:
        raise PackageAssemblyError("package config contains unresolved __REPLACE_ tokens: " + ", ".join(blockers[:10]))
    if config.get("inference_attestation") != ATTESTATION:
        raise PackageAssemblyError(
            "inference_attestation must declare genuine model inference at every native surface point of every case"
        )
    chunk_size = config.get("profile_cases_per_chunk", 50)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 100:
        raise PackageAssemblyError("profile_cases_per_chunk must lie in [1, 100]")
    specification = load_specification()
    if specification.get("dataset_version") != DATASET_VERSION:
        raise PackageAssemblyError("assembler accepts only the active DrivAerNet++ contract")
    split_id = config.get("split_id")
    participant = config.get("participant")
    if not isinstance(participant, dict):
        raise PackageAssemblyError("config.participant must be an object")
    submission_id = participant.get("submission_id")
    if split_id != entry["split_id"] or submission_id != entry["submission_id"]:
        raise PackageAssemblyError("package config submission_id and split_id must match the evaluated entry")
    split = official_split(split_id)
    case_ids = list(split["case_ids"])
    declaration = next(item for item in specification["splits"] if item["id"] == split_id)
    candidate, ground_truth = _release_bindings(config, specification)
    manifest_path = SPEC_ROOT / specification["scoring_support"]["candidate_manifest"]["manifest_file"]
    if sha256_file(manifest_path) != candidate["manifest_sha256"]:
        raise PackageAssemblyError("candidate support manifest SHA-256 differs")
    support_errors = validate_candidate_manifest_release(
        specification, SPEC_ROOT, manifest_path, read_json(manifest_path)
    )
    if support_errors:
        raise PackageAssemblyError("candidate scoring support is invalid: " + "; ".join(support_errors))

    recomputed = score_candidate_dataset(
        split_id=split_id, case_evidence_directory=evaluation_root / "case-evidence", submission_id=submission_id
    )
    if recomputed != read_json(evaluation_root / "dataset-evidence.json"):
        raise PackageAssemblyError("retained dataset evidence differs from a fresh exact reduction")
    metric_values = {metric["id"]: float(recomputed["metric_values"][metric["id"]]) for metric in specification["metrics"]}
    if any(not math.isfinite(value) for value in metric_values.values()):
        raise PackageAssemblyError("dataset evidence contains non-finite metrics")

    overlap = sorted(FORBIDDEN_PARTICIPANT_KEYS.intersection(participant))
    if overlap:
        raise PackageAssemblyError(f"participant config must not override benchmark fields: {overlap}")
    methodology = participant.get("methodology")
    if not isinstance(methodology, Mapping) or methodology.get("record_kind") != "submitter_reported":
        raise PackageAssemblyError("genuine-inference assembly requires methodology.record_kind=submitter_reported")
    participant_submission = copy.deepcopy(participant)
    participant_submission["dataset_id"] = DATASET_ID
    participant_submission["prediction_scope"] = PREDICTION_SCOPE
    try:
        participant_submission["parameter_count_millions"] = derived_parameter_count_millions(methodology)
        require_methodology(
            participant_submission,
            expected_case_count=len(case_ids),
            contract=read_json(SPEC_ROOT / "methodology-contract.json"),
        )
    except MethodologyError as error:
        raise PackageAssemblyError(f"participant methodology is invalid: {error}") from error
    _verify_checkpoints(methodology, checkpoint_paths)
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    command, generated_at = evaluation.get("command"), evaluation.get("generated_at")
    if not isinstance(command, str) or not command:
        raise PackageAssemblyError("evaluation.command must be a non-empty string")
    try:
        if datetime.fromisoformat(str(generated_at).replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except ValueError as error:
        raise PackageAssemblyError("evaluation.generated_at must be an RFC 3339 date-time with timezone") from error

    evidence_by_case = {}
    for binding in recomputed["case_evidence"]:
        path = evaluation_root / "case-evidence" / f"{binding['case_id']}.json"
        if sha256_file(path) != binding["sha256"]:
            raise PackageAssemblyError(f"{binding['case_id']} evidence changed after reduction")
        evidence_by_case[binding["case_id"]] = read_json(path)
    counts = {case_id: evidence_by_case[case_id]["support"]["entity_count"] for case_id in case_ids}

    output_path.mkdir(parents=True)
    chunks = []
    for chunk_index, offset in enumerate(range(0, len(case_ids), chunk_size)):
        selected = case_ids[offset : offset + chunk_size]
        name = f"chunk-{chunk_index:03d}.json"
        cases = []
        for case_id in selected:
            profiles = evidence_by_case[case_id]["profiles"]
            if [p["station_id"] for p in profiles] != list(PROFILE_STATIONS):
                raise PackageAssemblyError(f"{case_id}: profile stations differ")
            cases.append(
                {
                    "case_id": case_id,
                    "series": [
                        {
                            "panel_id": PROFILE_PANEL,
                            "station_id": p["station_id"],
                            "quantity_id": "cp",
                            "coordinate": p["coordinate"],
                            "prediction": p["prediction_cp"],
                        }
                        for p in profiles
                    ],
                }
            )
        digest = _write_json(output_path / "profiles" / name, {"schema_version": "1.0", "cases": cases})
        chunks.append({"file": name, "case_ids": selected, "sha256": digest})
    profile_index_sha = _write_json(
        output_path / "profiles" / "index.json",
        {
            "schema_version": "1.0",
            "format": PROFILE_FORMAT,
            "submission_id": submission_id,
            "dataset_id": DATASET_ID,
            "split_id": split_id,
            "case_set_id": split["case_set_id"],
            "case_count": len(case_ids),
            "case_id_status": declaration["case_id_status"],
            "chunks": chunks,
        },
    )
    case_metrics = {
        "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": DATASET_ID,
        "split_id": split_id,
        "case_set_id": split["case_set_id"],
        "scoring_support_release_id": candidate["release_id"],
        "scoring_support_manifest_sha256": candidate["manifest_sha256"],
        "case_count": len(case_ids),
        "cases": [_case_metric_record(evidence_by_case[case_id]) for case_id in case_ids],
        "metric_values": metric_values,
        "generated_at": generated_at,
    }
    _require_schema(case_metrics, "v3/case-metrics.schema.json", "metrics/cases.json")
    case_metrics_sha = _write_json(output_path / "metrics" / "cases.json", case_metrics)
    cases_sha = _discretization_cases(
        discretization_cases_path,
        output_path / "discretization" / "cases.jsonl",
        submission_id=submission_id,
        split_id=split_id,
        case_ids=case_ids,
        counts=counts,
    )
    spatial = _mapping(config.get("spatial_discretization"), "spatial_discretization")
    discretization = {
        "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": DATASET_ID,
        "split_id": split_id,
        "scoring_support_release_id": candidate["release_id"],
        "scoring_support_manifest_sha256": candidate["manifest_sha256"],
        "training": copy.deepcopy(spatial.get("training")),
        "inference": copy.deepcopy(spatial.get("inference")),
        "case_manifest": {
            "format": "jsonl",
            "file": "discretization/cases.jsonl",
            "sha256": cases_sha,
            "case_count": len(case_ids),
        },
    }
    if "notes" in spatial:
        discretization["notes"] = spatial["notes"]
    _require_schema(discretization, "v3/discretization.schema.json", "discretization.json")
    discretization_sha = _write_json(output_path / "discretization.json", discretization)
    evidence = {
        "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
        "schema_version": "3.0",
        "submission_id": submission_id,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "split_id": split_id,
        "split_sha256": declaration["sha256"],
        "case_set_id": split["case_set_id"],
        "prediction_scope": PREDICTION_SCOPE,
        "reference_version": specification["evaluation_reference_version"],
        "command": command,
        "generated_at": generated_at,
        "status": "submitted_evaluation",
        "metric_values": metric_values,
        "profile_index_sha256": profile_index_sha,
        "profile_ground_truth_release_id": ground_truth["release_id"],
        "profile_ground_truth_manifest_sha256": ground_truth["manifest_sha256"],
        "scoring_support_release_id": candidate["release_id"],
        "scoring_support_manifest_sha256": candidate["manifest_sha256"],
        "discretization_sha256": discretization_sha,
        "case_metrics_sha256": case_metrics_sha,
        "notes": GENUINE_INFERENCE_ATTESTATION,
    }
    code = (participant.get("reproducibility") or {}).get("code")
    if isinstance(code, Mapping) and code.get("commit"):
        evidence["code_revision"] = code["commit"]
    _require_schema(evidence, "v3/evaluation-evidence.schema.json", "evaluation-evidence.json")
    evidence_sha = _write_json(output_path / "evaluation-evidence.json", evidence)
    submission = {
        "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
        "schema_version": "3.0",
        **participant_submission,
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "split": declaration["label"],
        "split_id": split_id,
        "case_set_id": split["case_set_id"],
        "split_sha256": declaration["sha256"],
        "prediction_scope": PREDICTION_SCOPE,
        "evaluation": {
            "reference_version": specification["evaluation_reference_version"],
            "command": command,
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": evidence_sha,
        },
        "scoring_support": {
            "status": "candidate",
            "release_id": candidate["release_id"],
            "manifest_url": candidate["manifest_url"],
            "manifest_sha256": candidate["manifest_sha256"],
        },
        "spatial_discretization": {
            "format": "fluidsbench-discretization-v1",
            "file": "discretization.json",
            "sha256": discretization_sha,
        },
        "case_metrics": {
            "format": "fluidsbench-case-metrics-v1",
            "file": "metrics/cases.json",
            "sha256": case_metrics_sha,
            "case_count": len(case_ids),
        },
        "metric_values": metric_values,
        "profile_data": {
            "format": PROFILE_FORMAT,
            "index_file": "profiles/index.json",
            "case_count": len(case_ids),
            "case_set_id": split["case_set_id"],
            "profile_ground_truth_release_id": ground_truth["release_id"],
            "profile_ground_truth_manifest_sha256": ground_truth["manifest_sha256"],
        },
    }
    if evidence.get("code_revision"):
        submission["evaluation"]["code_revision"] = evidence["code_revision"]
    _require_schema(submission, "v3/submission.schema.json", "submission.json")
    _write_json(output_path / "submission.json", submission)
    return {
        "submission_id": submission_id,
        "split_id": split_id,
        "case_count": len(case_ids),
        "metric_values": metric_values,
        "dev_intake_validation_command": (
            "python scripts/validate_submission.py --dev-intake --contributor-stage "
            f"submissions/{DATASET_ID}/{submission_id}"
        ),
    }
