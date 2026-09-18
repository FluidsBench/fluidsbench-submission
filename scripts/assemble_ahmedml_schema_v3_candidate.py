#!/usr/bin/env python3
"""Assemble a genuine-inference AhmedML schema-v3 candidate package.

The adapter consumes exact per-case evidence emitted by the AhmedML evaluator,
re-reduces the declared split, creates prediction-only 128-sample profile
chunks, and binds participant methodology and discretization records.  It is
fail-closed and creates neither owner approval nor an official leaderboard
claim.  AhmedML submissions remain closed while the native support is a
candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    DATASET_VERSION,
    PROFILE_DEFINITION_SHA256,
    REGION_DEFINITION_SHA256,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
)
from reference.ahmedml.dataset_scorer import (  # noqa: E402
    AhmedMLDatasetScorerError,
    score_candidate_dataset,
)
from reference.ahmedml.pre_release import (  # noqa: E402
    GENUINE_INFERENCE_ATTESTATION,
    sha256_file,
)
from reference.ahmedml.regional_aggregate import (  # noqa: E402
    AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DEFINITION_ID,
    build_aggregate_regional_diagnostics,
)
from reference.methodology import (  # noqa: E402
    MethodologyError,
    derived_parameter_count_millions,
    require_methodology,
)
from reference.scoring_support import (  # noqa: E402
    ScoringSupportError,
    load_support_release,
)
from scripts.assemble_ahmedml_schema_v3_dev_fixture import (  # noqa: E402
    AssemblyError as EvidenceNormalizationError,
    _case_metrics,
    _case_profiles,
    _support_counts,
    _write_json,
)
from scripts.validate_scoring_supports import (  # noqa: E402
    validate_candidate_manifest_release,
)


DEFAULT_SPECIFICATION = ROOT / "benchmark-specs" / "ahmedml" / "submission-spec.json"
CONFIG_SCHEMA = "ahmedml-fluidsbench-schema-v3-package-config-v1"
PROFILE_FORMAT = "fluidsbench-profile-chunks-v1"
SCHEMA_ROOT = ROOT / "schemas"
TOKEN_PREFIXES = ("__REPLACE_", "__UNRESOLVED_AHMEDML_")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class PackageAssemblyError(ValueError):
    """Raised when an AhmedML package cannot be assembled honestly."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageAssemblyError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> Any:
    raise PackageAssemblyError(f"JSON contains forbidden non-finite token {token}")


def load_json_with_sha256(
    path: Path, *, label: str | None = None
) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except PackageAssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageAssemblyError(
            f"cannot read {label or 'JSON'} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PackageAssemblyError(f"{label or path} must contain a JSON object")
    return value, hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    return load_json_with_sha256(path, label=label)[0]


def _json_path(parts: Iterable[str | int]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def unresolved_tokens(
    value: Any, path: tuple[str | int, ...] = ()
) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            found.extend(unresolved_tokens(value[key], (*path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(unresolved_tokens(item, (*path, index)))
    elif isinstance(value, str) and value.startswith(TOKEN_PREFIXES):
        found.append({"path": _json_path(path), "token": value})
    return found


def _require_schema(value: Any, relative_path: str, label: str) -> None:
    schema = load_json(SCHEMA_ROOT / relative_path, label="schema")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(value),
        key=lambda item: _json_path(item.absolute_path),
    )
    if errors:
        rendered = "; ".join(
            f"{_json_path(error.absolute_path)}: {error.message}"
            for error in errors
        )
        raise PackageAssemblyError(
            f"{label} does not satisfy {relative_path}: {rendered}"
        )


def _require_methodology_schema(value: Any) -> None:
    submission_schema = load_json(
        SCHEMA_ROOT / "v3" / "submission.schema.json", label="submission schema"
    )
    fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": submission_schema["$defs"],
        "$ref": "#/$defs/fluidsbench_methodology",
    }
    validator = Draft202012Validator(fragment, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(value),
        key=lambda item: _json_path(item.absolute_path),
    )
    if errors:
        rendered = "; ".join(
            f"{_json_path(error.absolute_path)}: {error.message}"
            for error in errors
        )
        raise PackageAssemblyError(
            f"participant methodology does not satisfy the schema: {rendered}"
        )


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PackageAssemblyError(f"{label} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PackageAssemblyError(f"{label} must stay inside {root}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise PackageAssemblyError(f"{label} must stay inside {root}")
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageAssemblyError(f"{label} must be an object")
    return value


def _find_split(
    specification: Mapping[str, Any],
    specification_path: Path,
    split_id: str,
) -> tuple[Mapping[str, Any], dict[str, Any], list[str], str]:
    matches = [
        split
        for split in specification.get("splits", [])
        if isinstance(split, Mapping) and split.get("id") == split_id
    ]
    if len(matches) != 1:
        raise PackageAssemblyError(
            f"split_id {split_id!r} is not one official AhmedML split"
        )
    declaration = matches[0]
    split_path = _safe_child(
        specification_path.parent, declaration.get("index_file"), "split index"
    )
    if not split_path.is_file():
        raise PackageAssemblyError(f"split index is missing: {split_path}")
    split, digest = load_json_with_sha256(split_path, label="split index")
    if digest != declaration.get("sha256"):
        raise PackageAssemblyError("split index SHA-256 differs from the specification")
    case_ids = split.get("case_ids")
    if (
        declaration.get("case_id_status") != "official"
        or split.get("dataset_id") != "ahmedml"
        or split.get("split_id") != split_id
        or split.get("case_set_id") != declaration.get("case_set_id")
        or not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(case_id, str) and case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
        or len(case_ids) != declaration.get("case_count")
        or len(case_ids) != split.get("case_count")
    ):
        raise PackageAssemblyError("split index identities or case order are invalid")
    return declaration, split, list(case_ids), digest


def _release_bindings(
    config: Mapping[str, Any],
    specification: Mapping[str, Any],
    specification_path: Path,
    profile_truth_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    release = _mapping(config.get("release_bindings"), "release_bindings")
    expected_keys = {
        "candidate_manifest",
        "evaluator",
        "profile_definition",
        "profile_ground_truth",
    }
    if set(release) != expected_keys:
        raise PackageAssemblyError(
            f"release_bindings keys must be {sorted(expected_keys)}"
        )
    candidate = dict(
        _mapping(release.get("candidate_manifest"), "candidate_manifest")
    )
    evaluator = dict(_mapping(release.get("evaluator"), "evaluator"))
    profile = dict(_mapping(release.get("profile_definition"), "profile_definition"))
    ground_truth = dict(
        _mapping(release.get("profile_ground_truth"), "profile_ground_truth")
    )
    scoring_support = _mapping(
        specification.get("scoring_support"), "specification.scoring_support"
    )
    if (
        specification.get("status") != "candidate_native_support"
        or scoring_support.get("status") != "owner_review_required"
        or scoring_support.get("submissions_open") is not False
    ):
        raise PackageAssemblyError(
            "the AhmedML specification is not the expected closed candidate"
        )
    owner_candidate = _mapping(
        scoring_support.get("candidate_manifest"),
        "specification candidate manifest",
    )
    for key in ("status", "release_id", "manifest_url", "manifest_sha256"):
        if candidate.get(key) != owner_candidate.get(key):
            raise PackageAssemblyError(
                f"candidate_manifest.{key} does not match the repository binding"
            )
    if set(candidate) != {"status", "release_id", "manifest_url", "manifest_sha256"}:
        raise PackageAssemblyError("candidate_manifest contains unsupported keys")
    if (
        evaluator
        != {"reference_version": specification.get("evaluation_reference_version")}
    ):
        raise PackageAssemblyError("evaluator identity differs from the specification")
    active_profile = _mapping(
        specification.get("profile_definition"), "profile_definition"
    )
    if profile != {
        "id": active_profile.get("id"),
        "file": active_profile.get("file"),
        "sha256": active_profile.get("sha256"),
    }:
        raise PackageAssemblyError(
            "profile definition identity differs from the active repository contract"
        )
    profile_path = _safe_child(
        specification_path.parent, profile["file"], "profile definition"
    )
    if not profile_path.is_file() or sha256_file(profile_path) != PROFILE_DEFINITION_SHA256:
        raise PackageAssemblyError("active profile definition bytes differ")

    global_truth = _mapping(
        load_json(ROOT / "leaderboard" / "manifest.json", label="leaderboard manifest")
        .get("data_release", {})
        .get("profile_ground_truth"),
        "global profile-ground-truth binding",
    )
    if ground_truth != {
        "release_id": global_truth.get("release_id"),
        "manifest_sha256": global_truth.get("manifest_sha256"),
    }:
        raise PackageAssemblyError(
            "profile-ground-truth identity differs from the dev release binding"
        )
    truth_manifest, truth_manifest_sha256 = load_json_with_sha256(
        profile_truth_manifest_path, label="profile-ground-truth manifest"
    )
    if (
        not isinstance(ground_truth.get("release_id"), str)
        or not isinstance(ground_truth.get("manifest_sha256"), str)
        or SHA256_PATTERN.fullmatch(ground_truth["manifest_sha256"]) is None
        or truth_manifest_sha256 != ground_truth["manifest_sha256"]
    ):
        raise PackageAssemblyError("profile-ground-truth manifest bytes differ")
    if truth_manifest.get("data_release", {}).get("id") != ground_truth["release_id"]:
        raise PackageAssemblyError("profile-ground-truth release ID differs")
    ahmed_truth = next(
        (
            dataset
            for dataset in truth_manifest.get("datasets", [])
            if isinstance(dataset, Mapping) and dataset.get("id") == "ahmedml"
        ),
        None,
    )
    native_truth = (
        ahmed_truth.get("native_profile_truth")
        if isinstance(ahmed_truth, Mapping)
        else None
    )
    if (
        not isinstance(native_truth, Mapping)
        or native_truth.get("dataset_revision") != REPOSITORY_REVISION
        or native_truth.get("source_identity_sha256") != SOURCE_IDENTITY_SHA256
        or native_truth.get("profile_definition_sha256")
        != PROFILE_DEFINITION_SHA256
        or native_truth.get("series_per_case") != 7
        or native_truth.get("samples_per_series") != 128
    ):
        raise PackageAssemblyError(
            "profile-ground-truth manifest lacks the exact AhmedML native release"
        )
    return candidate, ground_truth


def _checkpoint_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        identifier, separator, raw_path = value.partition("=")
        if not separator or not identifier or not raw_path or identifier in result:
            raise PackageAssemblyError(
                "every --checkpoint must use one unique METHOD_CHECKPOINT_ID=PATH"
            )
        result[identifier] = Path(raw_path).expanduser().resolve()
    return result


def _verify_checkpoints(methodology: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    checkpoints = methodology.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise PackageAssemblyError("methodology must declare at least one checkpoint")
    declared: dict[str, str] = {}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise PackageAssemblyError("methodology checkpoint must be an object")
        identifier = checkpoint.get("id")
        digest = checkpoint.get("sha256")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in declared
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise PackageAssemblyError("methodology checkpoint identity is invalid")
        declared[identifier] = digest
    if set(paths) != set(declared):
        raise PackageAssemblyError(
            "--checkpoint IDs must exactly cover methodology checkpoints; "
            f"missing={sorted(set(declared) - set(paths))}, "
            f"unexpected={sorted(set(paths) - set(declared))}"
        )
    for identifier, expected in declared.items():
        path = paths[identifier]
        if path.is_symlink() or not path.is_file():
            raise PackageAssemblyError(
                f"checkpoint {identifier!r} is not a retained regular file"
            )
        if sha256_file(path) != expected:
            raise PackageAssemblyError(
                f"checkpoint {identifier!r} bytes differ from methodology.sha256"
            )


def _normalize_discretization_cases(
    source: Path,
    destination: Path,
    *,
    submission_id: str,
    split_id: str,
    case_ids: Sequence[str],
    case_metrics: Mapping[str, Any],
) -> str:
    try:
        records = [
            json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except PackageAssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageAssemblyError(f"cannot read discretization cases: {error}") from error
    if not all(isinstance(record, dict) for record in records):
        raise PackageAssemblyError("every discretization case must be an object")
    if [record.get("case_id") for record in records] != list(case_ids):
        raise PackageAssemblyError(
            "discretization cases must match the exact ordered split case list"
        )
    metric_cases = {
        case["case_id"]: case for case in case_metrics.get("cases", [])
    }
    encoded: list[bytes] = []
    for record in records:
        identities = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": "ahmedml",
            "split_id": split_id,
        }
        for key, expected in identities.items():
            if key in record and record[key] != expected:
                raise PackageAssemblyError(
                    f"discretization case {record.get('case_id')}.{key} conflicts "
                    "with the package identity"
                )
        record.update(identities)
        metric_case = metric_cases.get(record["case_id"])
        if not isinstance(metric_case, Mapping):
            raise PackageAssemblyError("discretization case has no metric record")
        support_counts = {
            support["support_id"]: (
                support["support_count"],
                support["scored_count"],
            )
            for support in metric_case.get("supports", [])
            if isinstance(support, Mapping)
        }
        for mapping in record.get("inference", {}).get("mappings", []):
            expected = support_counts.get(mapping.get("support_id"))
            if expected is not None and (
                mapping.get("support_count"), mapping.get("scored_count")
            ) != expected:
                raise PackageAssemblyError(
                    "discretization mapping counts disagree with case metrics for "
                    f"{record['case_id']}"
                )
        _require_schema(
            record,
            "v3/discretization-case.schema.json",
            f"discretization case {record['case_id']}",
        )
        encoded.append(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"\n".join(encoded) + b"\n")
    return sha256_file(destination)


def _prediction_input_integrity(
    evidence: Mapping[str, Any], case_id: str, counts: Mapping[str, int]
) -> None:
    prediction_inputs = _mapping(
        evidence.get("prediction_inputs"), f"{case_id}.prediction_inputs"
    )
    for domain, support_id in (
        ("surface", "ahmedml-surface-native-cells-v1"),
        ("volume", "ahmedml-volume-native-cells-v1"),
    ):
        value = _mapping(prediction_inputs.get(domain), f"{case_id}.{domain}")
        chunks = value.get("chunk_sha256")
        if (
            not isinstance(value.get("manifest_sha256"), str)
            or SHA256_PATTERN.fullmatch(value["manifest_sha256"]) is None
            or not isinstance(chunks, list)
            or not chunks
            or any(
                not isinstance(digest, str)
                or SHA256_PATTERN.fullmatch(digest) is None
                for digest in chunks
            )
            or value.get("entity_count") != counts[support_id]
        ):
            raise PackageAssemblyError(
                f"{case_id} does not retain exact {domain} prediction input hashes"
            )


def assemble_package(
    *,
    config_path: Path,
    specification_path: Path,
    case_evidence_root: Path,
    discretization_cases_path: Path,
    profile_truth_manifest_path: Path,
    checkpoint_paths: Mapping[str, Path],
    output_path: Path,
    dataset_evidence_path: Path | None = None,
) -> dict[str, Any]:
    config = load_json(config_path, label="package config")
    if config.get("schema") != CONFIG_SCHEMA:
        raise PackageAssemblyError(f"config.schema must equal {CONFIG_SCHEMA!r}")
    blockers = unresolved_tokens(config)
    if blockers:
        raise PackageAssemblyError(
            "configuration contains unresolved tokens: "
            + ", ".join(item["path"] for item in blockers)
        )
    if output_path.exists() or output_path.is_symlink():
        raise PackageAssemblyError(f"output already exists: {output_path}")
    profile_cases_per_chunk = config.get("profile_cases_per_chunk", 10)
    if (
        isinstance(profile_cases_per_chunk, bool)
        or not isinstance(profile_cases_per_chunk, int)
        or not 1 <= profile_cases_per_chunk <= 50
    ):
        raise PackageAssemblyError("profile_cases_per_chunk must lie in [1, 50]")
    attestation = config.get("inference_attestation")
    if attestation != {
        "actual_model_inference": True,
        "evaluation_truth_generated_predictions": False,
        "scope": "all_declared_cases_and_surface_volume_fields",
    }:
        raise PackageAssemblyError(
            "config.inference_attestation must explicitly declare genuine complete "
            "surface-and-volume model inference with no truth-generated predictions"
        )

    specification = load_json(specification_path, label="AhmedML specification")
    if (
        specification.get("dataset_id") != "ahmedml"
        or specification.get("dataset_version") != DATASET_VERSION
    ):
        raise PackageAssemblyError("assembler accepts only the active AhmedML contract")
    split_id = config.get("split_id")
    if not isinstance(split_id, str):
        raise PackageAssemblyError("config.split_id must be a string")
    split_declaration, split, case_ids, split_sha = _find_split(
        specification, specification_path, split_id
    )
    candidate, ground_truth = _release_bindings(
        config,
        specification,
        specification_path,
        profile_truth_manifest_path,
    )
    support_manifest_path = _safe_child(
        specification_path.parent,
        specification["scoring_support"]["candidate_manifest"]["manifest_file"],
        "candidate support manifest",
    )
    support_manifest, support_manifest_sha256 = load_json_with_sha256(
        support_manifest_path, label="support manifest"
    )
    if support_manifest_sha256 != candidate["manifest_sha256"]:
        raise PackageAssemblyError("candidate support manifest SHA-256 differs")
    # The generic support validator covers field/case-scalar bindings. AhmedML
    # additionally declares evaluator-derived profile R2 bindings on profile
    # supports; those are recomputed and validated by the dataset scorer below.
    support_validation_manifest = copy.deepcopy(support_manifest)
    profile_metric_ids = {"cp_cut_r2", "velocity_profile_r2"}
    support_validation_manifest["supports"] = [
        support
        for support in support_validation_manifest.get("supports", [])
        if not (
            isinstance(support, dict)
            and isinstance(support.get("metric_bindings"), list)
            and support["metric_bindings"]
            and {
                binding.get("metric_id")
                for binding in support["metric_bindings"]
                if isinstance(binding, dict)
            }
            <= profile_metric_ids
        )
    ]
    support_errors = validate_candidate_manifest_release(
        specification,
        specification_path.parent,
        support_manifest_path,
        support_validation_manifest,
    )
    if support_errors:
        raise PackageAssemblyError(
            "candidate scoring support is invalid: " + "; ".join(support_errors)
        )
    release = load_support_release(support_manifest_path, split["case_set_id"])
    if list(release.cases) != case_ids:
        raise PackageAssemblyError(
            "candidate support case order differs from the declared split"
        )

    try:
        recomputed_evidence = score_candidate_dataset(
            submission_specification=specification_path,
            split_id=split_id,
            case_evidence_directory=case_evidence_root,
        ).to_json()
    except AhmedMLDatasetScorerError as error:
        raise PackageAssemblyError(str(error)) from error
    if dataset_evidence_path is not None:
        retained_evidence = load_json(
            dataset_evidence_path, label="retained dataset evidence"
        )
        if retained_evidence != recomputed_evidence:
            raise PackageAssemblyError(
                "retained dataset evidence differs from a fresh exact reduction"
            )

    participant = config.get("participant")
    if not isinstance(participant, dict):
        raise PackageAssemblyError("config.participant must be an object")
    forbidden = {
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
        "regional_diagnostics",
        "approval",
    }
    overlap = sorted(forbidden.intersection(participant))
    if overlap:
        raise PackageAssemblyError(
            f"participant config must not override benchmark fields: {overlap}"
        )
    submission_id = participant.get("submission_id")
    if not isinstance(submission_id, str):
        raise PackageAssemblyError("participant.submission_id must be a string")
    methodology = participant.get("methodology")
    if not isinstance(methodology, Mapping):
        raise PackageAssemblyError("participant.methodology must be an object")
    if methodology.get("record_kind") != "submitter_reported":
        raise PackageAssemblyError(
            "genuine-inference assembly requires methodology.record_kind=submitter_reported"
        )
    _require_methodology_schema(methodology)
    methodology_contract = load_json(
        specification_path.parent / "methodology-contract.json",
        label="methodology contract",
    )
    participant_submission = copy.deepcopy(participant)
    participant_submission["dataset_id"] = "ahmedml"
    participant_submission["prediction_scope"] = "surface_and_volume"
    try:
        participant_submission["parameter_count_millions"] = (
            derived_parameter_count_millions(methodology)
        )
        require_methodology(
            participant_submission,
            expected_case_count=len(case_ids),
            contract=methodology_contract,
        )
    except MethodologyError as error:
        raise PackageAssemblyError(f"participant methodology is invalid: {error}") from error
    _verify_checkpoints(methodology, checkpoint_paths)

    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    command = evaluation.get("command")
    generated_at = evaluation.get("generated_at")
    if not isinstance(command, str) or not command:
        raise PackageAssemblyError("evaluation.command must be a non-empty string")
    if not isinstance(generated_at, str):
        raise PackageAssemblyError("evaluation.generated_at must be an RFC3339 string")
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise PackageAssemblyError("evaluation.generated_at is invalid") from error
    if parsed.tzinfo is None:
        raise PackageAssemblyError("evaluation.generated_at must include a timezone")

    evidence_bindings = {
        item.get("case_id"): item.get("sha256")
        for item in recomputed_evidence.get("case_evidence", [])
        if isinstance(item, Mapping)
    }
    if set(evidence_bindings) != set(case_ids):
        raise PackageAssemblyError("dataset reduction case-evidence bindings differ")
    case_documents: dict[str, dict[str, Any]] = {}
    profile_cases: dict[str, list[dict[str, Any]]] = {}
    metric_records: list[dict[str, Any]] = []
    for case_id in case_ids:
        evidence, evidence_sha256 = load_json_with_sha256(
            case_evidence_root / f"{case_id}.json", label=f"{case_id} evidence"
        )
        if evidence_sha256 != evidence_bindings[case_id]:
            raise PackageAssemblyError(
                f"{case_id} evidence changed after the exact dataset reduction"
            )
        try:
            profiles, cp_r2, velocity_r2 = _case_profiles(evidence, case_id)
            counts = _support_counts(release, case_id)
            metric_record = _case_metrics(
                evidence,
                case_id=case_id,
                counts=counts,
                cp_r2=cp_r2,
                velocity_r2=velocity_r2,
            )
        except EvidenceNormalizationError as error:
            raise PackageAssemblyError(str(error)) from error
        _prediction_input_integrity(evidence, case_id, counts)
        case_documents[case_id] = evidence
        profile_cases[case_id] = profiles
        metric_records.append(metric_record)

    required_metric_ids = [
        metric["id"]
        for metric in specification.get("metrics", [])
        if isinstance(metric, Mapping) and isinstance(metric.get("id"), str)
    ]
    raw_metrics = recomputed_evidence.get("metric_values")
    if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != set(
        required_metric_ids
    ):
        raise PackageAssemblyError(
            "dataset evidence metric set differs from the AhmedML specification"
        )
    metric_values = {
        metric_id: float(raw_metrics[metric_id]) for metric_id in required_metric_ids
    }
    if any(not math.isfinite(value) for value in metric_values.values()):
        raise PackageAssemblyError("dataset evidence contains non-finite metrics")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    try:
        chunk_bindings: list[dict[str, Any]] = []
        for chunk_index, offset in enumerate(
            range(0, len(case_ids), profile_cases_per_chunk)
        ):
            selected = case_ids[offset : offset + profile_cases_per_chunk]
            name = f"chunk-{chunk_index:03d}.json"
            digest = _write_json(
                staging / "profiles" / name,
                {
                    "schema_version": "1.0",
                    "cases": [
                        {"case_id": case_id, "series": profile_cases[case_id]}
                        for case_id in selected
                    ],
                },
            )
            chunk_bindings.append(
                {"file": name, "case_ids": selected, "sha256": digest}
            )
        profile_index_sha = _write_json(
            staging / "profiles" / "index.json",
            {
                "schema_version": "1.0",
                "format": PROFILE_FORMAT,
                "submission_id": submission_id,
                "dataset_id": "ahmedml",
                "split_id": split_id,
                "case_set_id": split["case_set_id"],
                "case_count": len(case_ids),
                "case_id_status": "official",
                "chunks": chunk_bindings,
            },
        )

        case_metrics = {
            "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": "ahmedml",
            "split_id": split_id,
            "case_set_id": split["case_set_id"],
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "case_count": len(case_ids),
            "cases": metric_records,
            "metric_values": metric_values,
            "generated_at": generated_at,
        }
        _require_schema(case_metrics, "v3/case-metrics.schema.json", "metrics/cases.json")
        case_metrics_sha = _write_json(
            staging / "metrics" / "cases.json", case_metrics
        )

        cases_sha = _normalize_discretization_cases(
            discretization_cases_path,
            staging / "discretization" / "cases.jsonl",
            submission_id=submission_id,
            split_id=split_id,
            case_ids=case_ids,
            case_metrics=case_metrics,
        )
        spatial_config = _mapping(
            config.get("spatial_discretization"), "spatial_discretization"
        )
        discretization: dict[str, Any] = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": "ahmedml",
            "split_id": split_id,
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "training": copy.deepcopy(spatial_config.get("training")),
            "inference": copy.deepcopy(spatial_config.get("inference")),
            "case_manifest": {
                "format": "jsonl",
                "file": "discretization/cases.jsonl",
                "sha256": cases_sha,
                "case_count": len(case_ids),
            },
        }
        if "notes" in spatial_config:
            discretization["notes"] = spatial_config["notes"]
        _require_schema(
            discretization, "v3/discretization.schema.json", "discretization.json"
        )
        discretization_sha = _write_json(
            staging / "discretization.json", discretization
        )

        regional = build_aggregate_regional_diagnostics(
            case_ids=case_ids,
            case_evidence=case_documents,
            split_id=split_id,
        )
        regional_sha = _write_json(
            staging / "regional-diagnostics.json", regional
        )

        evidence: dict[str, Any] = {
            "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
            "schema_version": "3.0",
            "submission_id": submission_id,
            "dataset_id": "ahmedml",
            "dataset_version": DATASET_VERSION,
            "split_id": split_id,
            "split_sha256": split_sha,
            "case_set_id": split["case_set_id"],
            "prediction_scope": "surface_and_volume",
            "reference_version": specification["evaluation_reference_version"],
            "command": command,
            "generated_at": generated_at,
            "status": "submitted_evaluation",
            "metric_values": metric_values,
            "profile_index_sha256": profile_index_sha,
            "profile_ground_truth_release_id": ground_truth["release_id"],
            "profile_ground_truth_manifest_sha256": ground_truth[
                "manifest_sha256"
            ],
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "discretization_sha256": discretization_sha,
            "case_metrics_sha256": case_metrics_sha,
            "regional_diagnostics_sha256": regional_sha,
            "notes": GENUINE_INFERENCE_ATTESTATION,
        }
        reproducibility = participant.get("reproducibility")
        code = (
            reproducibility.get("code")
            if isinstance(reproducibility, Mapping)
            else None
        )
        participant_code_revision = (
            code.get("commit") if isinstance(code, Mapping) else None
        )
        if participant_code_revision is not None:
            evidence["code_revision"] = participant_code_revision
        _require_schema(
            evidence,
            "v3/evaluation-evidence.schema.json",
            "evaluation-evidence.json",
        )
        evidence_sha = _write_json(staging / "evaluation-evidence.json", evidence)

        submission: dict[str, Any] = {
            "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
            "schema_version": "3.0",
            **participant_submission,
            "dataset": specification["dataset_name"],
            "dataset_id": "ahmedml",
            "dataset_version": DATASET_VERSION,
            "split": split_declaration["label"],
            "split_id": split_id,
            "case_set_id": split["case_set_id"],
            "split_sha256": split_sha,
            "prediction_scope": "surface_and_volume",
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
                "profile_ground_truth_manifest_sha256": ground_truth[
                    "manifest_sha256"
                ],
            },
            "regional_diagnostics": {
                "format": AGGREGATE_REGIONAL_REPORT_SCHEMA,
                "file": "regional-diagnostics.json",
                "sha256": regional_sha,
                "contract_sha256": REGION_DEFINITION_SHA256,
                "definition_id": REGIONAL_DEFINITION_ID,
                "case_count": len(case_ids),
                "role": "report_only",
                "weight": 0.0,
                "official_score_changed": False,
            },
        }
        if participant_code_revision is not None:
            submission["evaluation"]["code_revision"] = participant_code_revision
        _require_schema(submission, "v3/submission.schema.json", "submission.json")
        _write_json(staging / "submission.json", submission)
        staging.rename(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    expected_registration_path = (
        ROOT / "submissions" / "ahmedml" / submission_id / "submission.json"
    ).resolve()
    assembled_submission_path = (output_path / "submission.json").resolve()
    registration_command = (
        "python scripts/register_ahmedml_pre_release.py "
        f"{assembled_submission_path}"
        if assembled_submission_path == expected_registration_path
        else None
    )
    return {
        "status": "candidate_package_assembled_not_approved",
        "output": str(output_path),
        "submission_id": submission_id,
        "split_id": split_id,
        "case_count": len(case_ids),
        "candidate_dry_run_command": (
            f"python scripts/validate_submission.py --candidate-dry-run {output_path}"
        ),
        "registration_command": registration_command,
        "registration_requirement": (
            "candidate must first be retained at "
            f"submissions/ahmedml/{submission_id}/submission.json"
            if registration_command is None
            else "exact conventional repository path satisfied"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--submission-specification", type=Path, default=DEFAULT_SPECIFICATION
    )
    parser.add_argument("--case-evidence-directory", type=Path)
    parser.add_argument("--dataset-evidence", type=Path)
    parser.add_argument("--discretization-cases", type=Path)
    parser.add_argument("--profile-ground-truth-manifest", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="repeat once for every methodology checkpoint",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--list-unresolved",
        action="store_true",
        help="print unresolved config tokens without reading evaluator products",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config.expanduser().resolve(), label="package config")
        if args.list_unresolved:
            found = unresolved_tokens(config)
            print(
                json.dumps(
                    {
                        "status": "blocked_by_tokens" if found else "tokens_resolved",
                        "unresolved": found,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        missing = [
            name
            for name in (
                "case_evidence_directory",
                "discretization_cases",
                "profile_ground_truth_manifest",
                "output",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise PackageAssemblyError(
                "assembly requires arguments: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        result = assemble_package(
            config_path=args.config.expanduser().resolve(),
            specification_path=args.submission_specification.expanduser().resolve(),
            case_evidence_root=args.case_evidence_directory.expanduser().resolve(),
            discretization_cases_path=args.discretization_cases.expanduser().resolve(),
            profile_truth_manifest_path=(
                args.profile_ground_truth_manifest.expanduser().resolve()
            ),
            checkpoint_paths=_checkpoint_arguments(args.checkpoint),
            output_path=args.output.expanduser().resolve(),
            dataset_evidence_path=(
                args.dataset_evidence.expanduser().resolve()
                if args.dataset_evidence is not None
                else None
            ),
        )
    except (
        PackageAssemblyError,
        ScoringSupportError,
        EvidenceNormalizationError,
        OSError,
        ValueError,
    ) as error:
        print(
            json.dumps({"status": "blocked", "error": str(error)}, sort_keys=True),
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
