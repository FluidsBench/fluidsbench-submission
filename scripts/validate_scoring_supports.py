#!/usr/bin/env python3
"""Validate each dataset's canonical scoring-support publication gate.

This check is deliberately dataset-local. A collaborator can make one dataset
official without editing or waiting for any other dataset. Non-official
datasets must remain closed to new submissions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scoring_support import (  # noqa: E402
    ScoringSupportError,
    load_scoring_support,
    load_support_release,
    sha256_file,
)
from reference.weightings import evaluator_weighting  # noqa: E402


SPEC_ROOT = ROOT / "benchmark-specs"
SCHEMA_ROOT = ROOT / "schemas" / "scoring-support" / "v1"
METHODOLOGY_CONTRACT_SCHEMA = ROOT / "schemas" / "methodology-contract-v1.schema.json"
ALLOWED_STATUSES = {"prototype", "owner_review_required", "official", "retired"}
OFFICIAL_FIELDS = (
    "release_id",
    "manifest_file",
    "manifest_url",
    "manifest_sha256",
)
OWNER_APPROVAL_FIELDS = ("approved_by", "approved_at", "pull_request_url")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
UNPUBLISHED_FIELDS = set(OFFICIAL_FIELDS) | {
    "owner_approval",
    "publication_validation",
}
SUPPORT_AGGREGATIONS = {
    "per_geometry_then_macro_average",
    "flatten_all_aligned_field_values",
    "all_test_cases",
    "benchmark_field_rrmse_across_cases",
    "benchmark_scalar_rrmse_across_cases",
}


class DuplicateJSONKeyError(ValueError):
    """Raised when an authoritative JSON object repeats a member name."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)


def json_path(parts: list[Any]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def schema_errors(value: Any, schema_name: str) -> list[str]:
    schema = load_json(SCHEMA_ROOT / schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{json_path(list(error.absolute_path))}: {error.message}"
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: json_path(list(item.absolute_path)),
        )
    ]


def validate_methodology_contract(
    errors: list[str],
    dataset_directory: Path,
    dataset_id: str,
) -> None:
    """Validate the dataset-owned list of outputs a method must disclose."""

    path = dataset_directory / "methodology-contract.json"
    label = f"{dataset_id}.methodology_contract"
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
        return
    try:
        contract = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
        errors.append(f"{label} cannot be read: {error}")
        return
    schema = load_json(METHODOLOGY_CONTRACT_SCHEMA)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for error in sorted(
        validator.iter_errors(contract),
        key=lambda item: json_path(list(item.absolute_path)),
    ):
        errors.append(f"{label} {json_path(list(error.absolute_path))}: {error.message}")
    if not isinstance(contract, dict):
        return
    if contract.get("dataset_id") != dataset_id:
        errors.append(f"{label}.dataset_id must equal {dataset_id!r}")
    field_ids = [
        item.get("field_id")
        for item in contract.get("required_predicted_fields", [])
        if isinstance(item, dict)
    ]
    if len(field_ids) != len(set(field_ids)):
        errors.append(f"{label}.required_predicted_fields field_id values must be unique")


def safe_dataset_path(
    errors: list[str],
    dataset_directory: Path,
    value: Any,
    *,
    label: str,
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty relative path")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label} must stay inside its dataset directory")
        return None
    candidate = (dataset_directory / relative).resolve()
    if not candidate.is_relative_to(dataset_directory.resolve()):
        errors.append(f"{label} must stay inside its dataset directory")
        return None
    return candidate


def validate_closed_gate(
    errors: list[str],
    support: dict[str, Any],
    *,
    label: str,
) -> None:
    if support.get("submissions_open") is not False:
        errors.append(f"{label}.submissions_open must be false unless status is official")
    reason = support.get("closed_reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append(f"{label}.closed_reason must explain why submissions are closed")


def validate_owner_approval(
    errors: list[str],
    approval: Any,
    *,
    label: str,
) -> None:
    if not isinstance(approval, dict):
        errors.append(f"{label} must be an object")
        return
    for field in OWNER_APPROVAL_FIELDS:
        if not isinstance(approval.get(field), str) or not approval[field].strip():
            errors.append(f"{label}.{field} must be a non-empty string")
    pull_request_url = approval.get("pull_request_url")
    if isinstance(pull_request_url, str) and not re.fullmatch(
        r"https://github\.com/[^/]+/[^/]+/pull/[0-9]+",
        pull_request_url,
    ):
        errors.append(f"{label}.pull_request_url must identify the approving GitHub pull request")


def validate_split_index(
    errors: list[str],
    dataset_directory: Path,
    dataset_id: str,
    split: Any,
) -> tuple[str | None, list[str] | None]:
    label = f"{dataset_id}.splits"
    if not isinstance(split, dict):
        errors.append(f"{label} entries must be objects")
        return None, None
    split_id = split.get("id")
    split_label = f"{dataset_id}.splits[{split_id!r}]"
    if not isinstance(split_id, str) or not split_id:
        errors.append(f"{split_label}.id must be a non-empty string")
        return None, None
    if split.get("case_id_status") != "official":
        errors.append(f"{split_label}.case_id_status must be official")
    case_set_id = split.get("case_set_id")
    if not isinstance(case_set_id, str) or not case_set_id:
        errors.append(f"{split_label}.case_set_id must be a non-empty string")
        return None, None
    index_path = safe_dataset_path(
        errors,
        dataset_directory,
        split.get("index_file"),
        label=f"{split_label}.index_file",
    )
    if index_path is None:
        return case_set_id, None
    if not index_path.is_file():
        errors.append(f"{split_label}.index_file does not exist: {index_path}")
        return case_set_id, None
    declared_sha256 = split.get("sha256")
    if not isinstance(declared_sha256, str) or not SHA256_PATTERN.fullmatch(declared_sha256):
        errors.append(f"{split_label}.sha256 must be a lowercase SHA-256")
    elif sha256_file(index_path) != declared_sha256:
        errors.append(f"{split_label}.sha256 does not match {index_path.name}")
    try:
        index = load_json(index_path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
        errors.append(f"{split_label}.index_file cannot be read: {error}")
        return case_set_id, None
    if not isinstance(index, dict):
        errors.append(f"{split_label}.index_file must contain a JSON object")
        return case_set_id, None
    identities = {
        "dataset_id": dataset_id,
        "split_id": split_id,
        "case_set_id": case_set_id,
        "case_id_status": "official",
    }
    for key, expected in identities.items():
        if index.get(key) != expected:
            errors.append(f"{split_label} index {key} must equal {expected!r}")
    case_ids = index.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids or not all(
        isinstance(case_id, str) and case_id for case_id in case_ids
    ):
        errors.append(f"{split_label} index case_ids must be a non-empty string list")
        return case_set_id, None
    if len(case_ids) != len(set(case_ids)):
        errors.append(f"{split_label} index case_ids must be unique")
    for source, count in (
        ("specification", split.get("case_count")),
        ("index", index.get("case_count")),
    ):
        if count != len(case_ids):
            errors.append(
                f"{split_label} {source} case_count must equal {len(case_ids)}"
            )
    return case_set_id, case_ids


def expected_metric_reduction(
    metric_id: str,
    metric: dict[str, Any],
) -> str | None:
    aggregation = metric.get("aggregation")
    if aggregation in {
        "benchmark_field_rrmse_across_cases",
        "benchmark_scalar_rrmse_across_cases",
    }:
        return "dataset_reference"
    if metric_id.endswith("_rel_l1"):
        return "relative_l1_percent"
    if metric_id.endswith("_rel_l2"):
        return "relative_l2_percent"
    if metric_id.endswith("_rrmse"):
        return "dataset_reference"
    if metric_id.endswith("_rmse"):
        return "rmse"
    if metric_id.endswith("_mse"):
        return "mse"
    if metric_id.endswith("_mae"):
        return "mae"
    if metric_id.endswith("_r2"):
        return "r2"
    return None


def validate_metric_contract(
    errors: list[str],
    specification: dict[str, Any],
    manifest: dict[str, Any],
    *,
    label: str,
) -> None:
    """Require an official release to implement the specification's scoring metrics."""

    dataset_metric_entries = [
        metric
        for metric in specification.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("id"), str)
    ]
    dataset_metric_ids = [metric["id"] for metric in dataset_metric_entries]
    if len(dataset_metric_ids) != len(set(dataset_metric_ids)):
        errors.append(f"{label} dataset metric IDs must be unique")
    dataset_metrics = {
        metric["id"]: metric
        for metric in dataset_metric_entries
    }
    required_metric_ids = {
        metric_id
        for metric_id, metric in dataset_metrics.items()
        if metric.get("aggregation") in SUPPORT_AGGREGATIONS
    }

    bound_metric_ids: list[str] = []
    for support in manifest.get("supports", []):
        if not isinstance(support, dict):
            continue
        support_id = support.get("id")
        quantities = [
            quantity
            for quantity in support.get("quantities", [])
            if isinstance(quantity, dict)
        ]
        quantity_ids = [
            quantity.get("id")
            for quantity in quantities
            if isinstance(quantity.get("id"), str)
        ]
        for binding in support.get("metric_bindings", []):
            if not isinstance(binding, dict):
                continue
            metric_id = binding.get("metric_id")
            if not isinstance(metric_id, str):
                continue
            bound_metric_ids.append(metric_id)
            metric = dataset_metrics.get(metric_id)
            if metric is None:
                errors.append(
                    f"{label} support {support_id!r} binds unknown metric_id "
                    f"{metric_id!r}"
                )
                continue
            if binding.get("quantity_id") not in quantity_ids:
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    "references an unknown quantity"
                )
            if binding.get("aggregation") != metric.get("aggregation"):
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    "aggregation must match the dataset specification"
                )
            if binding.get("dataset_weighting") != metric.get("weighting"):
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    "dataset_weighting must match the dataset specification"
                )
            expected_reduction = expected_metric_reduction(metric_id, metric)
            if expected_reduction is None:
                errors.append(
                    f"{label} dataset metric {metric_id!r} requires an explicit "
                    "supported scoring reduction"
                )
            elif binding.get("reduction") != expected_reduction:
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    f"reduction must be {expected_reduction!r}"
                )
            expected_weighting = evaluator_weighting(metric.get("weighting"))
            if binding.get("weighting") != expected_weighting:
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    f"weighting must be {expected_weighting!r}"
                )
            expected_case_evidence = (
                "aggregate_only"
                if expected_reduction in {"r2", "dataset_reference"}
                else "metric_value"
            )
            if binding.get("case_evidence") != expected_case_evidence:
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    f"case_evidence must be {expected_case_evidence!r}"
                )
            if expected_reduction == "dataset_reference":
                reference_rule = binding.get("reference_rule")
                if not isinstance(reference_rule, dict) or (
                    reference_rule.get("id") != metric.get("aggregation")
                    or reference_rule.get("version")
                    != specification.get("evaluation_reference_version")
                ):
                    errors.append(
                        f"{label} support {support_id!r} metric {metric_id!r} "
                        "must pin the dataset aggregation rule and evaluator version"
                    )
            scalar_aggregation = metric.get("aggregation") in {
                "all_test_cases",
                "benchmark_scalar_rrmse_across_cases",
            }
            if scalar_aggregation and support.get("domain") != "scalar_case":
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    "requires a scalar_case support"
                )
            if not scalar_aggregation and support.get("domain") == "scalar_case":
                errors.append(
                    f"{label} support {support_id!r} metric {metric_id!r} "
                    "cannot use a scalar_case support"
                )

    if len(bound_metric_ids) != len(set(bound_metric_ids)):
        errors.append(f"{label} each metric may be bound only once")
    bound_metric_id_set = set(bound_metric_ids)
    if bound_metric_id_set != required_metric_ids:
        errors.append(
            f"{label} metric bindings must exactly cover dataset field and "
            "case-scalar metrics; "
            f"missing={sorted(required_metric_ids - bound_metric_id_set)}, "
            f"unexpected={sorted(bound_metric_id_set - required_metric_ids)}"
        )


def validate_declared_manifest(
    errors: list[str],
    dataset_directory: Path,
    specification: dict[str, Any],
    support: dict[str, Any],
) -> None:
    dataset_id = specification["dataset_id"]
    label = f"{dataset_id}.scoring_support"
    for field in OFFICIAL_FIELDS:
        if not isinstance(support.get(field), str) or not support[field].strip():
            errors.append(f"{label}.{field} must be a non-empty string")
    declared_hash = support.get("manifest_sha256")
    if isinstance(declared_hash, str) and not SHA256_PATTERN.fullmatch(declared_hash):
        errors.append(f"{label}.manifest_sha256 must be a lowercase SHA-256")
    manifest_url = support.get("manifest_url")
    if isinstance(manifest_url, str):
        parsed = urlparse(manifest_url)
        release_id = support.get("release_id")
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not isinstance(release_id, str)
            or release_id not in [part for part in parsed.path.split("/") if part]
        ):
            errors.append(
                f"{label}.manifest_url must be a clean public HTTPS URL with "
                "release_id as an exact path segment and no credentials, query, or fragment"
            )
    validate_owner_approval(
        errors,
        support.get("owner_approval"),
        label=f"{label}.owner_approval",
    )
    manifest_path = safe_dataset_path(
        errors,
        dataset_directory,
        support.get("manifest_file"),
        label=f"{label}.manifest_file",
    )
    if manifest_path is None:
        return
    if not manifest_path.is_file():
        errors.append(f"{label}.manifest_file does not exist: {manifest_path}")
        return
    if (
        isinstance(declared_hash, str)
        and SHA256_PATTERN.fullmatch(declared_hash)
        and sha256_file(manifest_path) != declared_hash
    ):
        errors.append(f"{label}.manifest_sha256 does not match {manifest_path.name}")
    try:
        manifest = load_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
        errors.append(f"{label}.manifest_file cannot be read: {error}")
        return
    for error in schema_errors(manifest, "manifest.schema.json"):
        errors.append(f"{label} manifest {error}")
    if not isinstance(manifest, dict):
        return
    identities = {
        "release_id": support.get("release_id"),
        "status": "official",
        "dataset_id": dataset_id,
        "dataset_version": specification.get("dataset_version"),
        "evaluation_reference_version": specification.get(
            "evaluation_reference_version"
        ),
        "owner_approval": support.get("owner_approval"),
    }
    for key, expected in identities.items():
        if manifest.get(key) != expected:
            errors.append(f"{label} manifest {key} must equal {expected!r}")
    validate_metric_contract(
        errors,
        specification,
        manifest,
        label=f"{label} manifest",
    )

    resolved_dataset_directory = dataset_directory.resolve()
    manifest_relative_directory = manifest_path.parent.relative_to(
        resolved_dataset_directory
    )
    for case_set in manifest.get("case_sets", []):
        if not isinstance(case_set, dict):
            continue
        case_set_id = case_set.get("id")
        index_file = case_set.get("index_file")
        if not isinstance(index_file, str):
            continue
        index_path = safe_dataset_path(
            errors,
            dataset_directory,
            (manifest_relative_directory / index_file).as_posix(),
            label=f"{label} case set {case_set_id!r} index_file",
        )
        if index_path is None or not index_path.is_file():
            continue
        try:
            support_index = load_json(index_path)
        except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError):
            continue
        for error in schema_errors(support_index, "case-index.schema.json"):
            errors.append(f"{label} case set {case_set_id!r} index {error}")
        if not isinstance(support_index, dict):
            continue
        index_relative_directory = index_path.parent.relative_to(
            resolved_dataset_directory
        )
        for chunk in support_index.get("chunks", []):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("file"), str):
                continue
            chunk_path = safe_dataset_path(
                errors,
                dataset_directory,
                (index_relative_directory / chunk["file"]).as_posix(),
                label=(
                    f"{label} case set {case_set_id!r} "
                    f"chunk {chunk.get('file')!r}"
                ),
            )
            if chunk_path is None or not chunk_path.is_file():
                continue
            try:
                support_chunk = load_json(chunk_path)
            except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError):
                continue
            for error in schema_errors(support_chunk, "case-chunk.schema.json"):
                errors.append(
                    f"{label} case set {case_set_id!r} "
                    f"chunk {chunk.get('file')!r} {error}"
                )

    split_case_sets: dict[str, list[str]] = {}
    for split in specification.get("splits", []):
        case_set_id, case_ids = validate_split_index(
            errors,
            dataset_directory,
            dataset_id,
            split,
        )
        if case_set_id is None or case_ids is None:
            continue
        existing = split_case_sets.setdefault(case_set_id, case_ids)
        if existing != case_ids:
            errors.append(
                f"{dataset_id} splits bound to case set {case_set_id!r} "
                "must use the same ordered case IDs"
            )

    manifest_case_sets = {
        item.get("id")
        for item in manifest.get("case_sets", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if manifest_case_sets != set(split_case_sets):
        errors.append(
            f"{label} manifest case sets {sorted(manifest_case_sets)} must equal "
            f"the specification case sets {sorted(split_case_sets)}"
        )
    releases: dict[str, Any] = {}
    for case_set_id, expected_case_ids in split_case_sets.items():
        try:
            release = load_support_release(manifest_path, case_set_id)
        except (OSError, KeyError, TypeError, ScoringSupportError) as error:
            errors.append(f"{label} case set {case_set_id!r} is invalid: {error}")
            continue
        releases[case_set_id] = release
        actual_case_ids = list(release.cases)
        if actual_case_ids != expected_case_ids:
            errors.append(
                f"{label} case set {case_set_id!r} cases must exactly match "
                "the ordered official split index"
            )

    requires_external_validation = False
    for release in releases.values():
        for case_id, case in release.cases.items():
            instances = {
                instance.get("support_id"): instance
                for instance in case.get("support_instances", [])
                if isinstance(instance, dict)
            }
            for support_id, definition in release.supports.items():
                location = definition.get("location_definition", {})
                instance = instances.get(support_id, {})
                artifact_role = location.get("artifact_role")
                local_roles = {
                    artifact.get("role")
                    for artifact in instance.get("artifacts", [])
                    if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
                }
                generic_local = (
                    location.get("mode") == "materialized_table"
                    and isinstance(artifact_role, str)
                    and artifact_role in local_roles
                )
                if not generic_local:
                    requires_external_validation = True
                    continue
                try:
                    loaded = load_scoring_support(release, case_id, support_id)
                except (OSError, KeyError, TypeError, ScoringSupportError) as error:
                    errors.append(
                        f"{label} {case_id}/{support_id} cannot be loaded by the "
                        f"declared generic support contract: {error}"
                    )
                    continue
                expected_count = instance.get("entity_count")
                if len(loaded.support_ids) != expected_count:
                    errors.append(
                        f"{label} {case_id}/{support_id} loaded count must equal "
                        "the declared entity_count"
                    )

    if requires_external_validation:
        validation = support.get("publication_validation")
        validation_label = f"{label}.publication_validation"
        if not isinstance(validation, dict):
            errors.append(
                f"{validation_label} is required when an official release uses "
                "remote artifacts or a dataset-specific loader"
            )
            return
        required_validation_strings = (
            "validated_by",
            "validated_at",
            "validator_file",
            "validator_sha256",
            "normalized_support_sha256",
            "pull_request_url",
        )
        if validation.get("status") != "passed":
            errors.append(f"{validation_label}.status must be passed")
        for field in required_validation_strings:
            if not isinstance(validation.get(field), str) or not validation[field].strip():
                errors.append(f"{validation_label}.{field} must be a non-empty string")
        if validation.get("manifest_sha256") != support.get("manifest_sha256"):
            errors.append(
                f"{validation_label}.manifest_sha256 must match the official manifest"
            )
        expected_case_count = sum(len(release.cases) for release in releases.values())
        expected_support_instances = sum(
            len(release.cases) * len(release.supports)
            for release in releases.values()
        )
        if validation.get("case_count") != expected_case_count:
            errors.append(
                f"{validation_label}.case_count must equal {expected_case_count}"
            )
        if validation.get("support_instance_count") != expected_support_instances:
            errors.append(
                f"{validation_label}.support_instance_count must equal "
                f"{expected_support_instances}"
            )
        for field in ("validator_sha256", "normalized_support_sha256"):
            value = validation.get(field)
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                errors.append(f"{validation_label}.{field} must be a lowercase SHA-256")
        validator_path = safe_dataset_path(
            errors,
            dataset_directory,
            validation.get("validator_file"),
            label=f"{validation_label}.validator_file",
        )
        if validator_path is not None:
            if not validator_path.is_file():
                errors.append(f"{validation_label}.validator_file does not exist")
            elif (
                isinstance(validation.get("validator_sha256"), str)
                and SHA256_PATTERN.fullmatch(validation["validator_sha256"])
                and sha256_file(validator_path) != validation["validator_sha256"]
            ):
                errors.append(
                    f"{validation_label}.validator_sha256 does not match "
                    "the pinned validator file"
                )
        validate_owner_approval(
            errors,
            {
                "approved_by": validation.get("validated_by"),
                "approved_at": str(validation.get("validated_at", ""))[:10],
                "pull_request_url": validation.get("pull_request_url"),
            },
            label=validation_label,
        )


def validate_candidate_manifest_release(
    specification: dict[str, Any],
    dataset_directory: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[str]:
    """Validate a closed candidate with the existing support/spec semantics.

    Candidate publication deliberately omits owner approval, but its manifest,
    case-set chain, metric bindings, and exact official case coverage must
    already be as rigorous as an official release.  This helper is shared by
    the promotion generator and participant-package assembler so neither path
    can accept an identity-only placeholder manifest.
    """

    errors: list[str] = []
    dataset_id = specification.get("dataset_id")
    label = f"{dataset_id}.scoring_support.candidate_manifest"
    if not isinstance(dataset_id, str) or not dataset_id:
        return ["candidate specification dataset_id must be a non-empty string"]
    resolved_dataset = dataset_directory.resolve()
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.is_relative_to(resolved_dataset):
        return [f"{label} file must stay inside its dataset directory"]

    for error in schema_errors(manifest, "manifest.schema.json"):
        errors.append(f"{label} manifest {error}")
    if not isinstance(manifest, dict):
        return errors + [f"{label} manifest must contain a JSON object"]
    identities = {
        "status": "candidate",
        "dataset_id": dataset_id,
        "dataset_version": specification.get("dataset_version"),
        "evaluation_reference_version": specification.get(
            "evaluation_reference_version"
        ),
    }
    for key, expected in identities.items():
        if manifest.get(key) != expected:
            errors.append(f"{label} manifest {key} must equal {expected!r}")
    if "owner_approval" in manifest:
        errors.append(f"{label} manifest must not claim owner approval")

    if isinstance(manifest.get("supports"), list) and isinstance(
        specification.get("metrics"), list
    ):
        validate_metric_contract(
            errors,
            specification,
            manifest,
            label=f"{label} manifest",
        )
    else:
        errors.append(
            f"{label} manifest support semantics require list-valued supports "
            "and specification metrics"
        )

    split_case_sets: dict[str, list[str]] = {}
    split_items = specification.get("splits")
    if not isinstance(split_items, list):
        errors.append(f"{label} specification splits must be a list")
        split_items = []
    for split in split_items:
        case_set_id, case_ids = validate_split_index(
            errors,
            dataset_directory,
            dataset_id,
            split,
        )
        if case_set_id is None or case_ids is None:
            continue
        existing = split_case_sets.setdefault(case_set_id, case_ids)
        if existing != case_ids:
            errors.append(
                f"{dataset_id} splits bound to case set {case_set_id!r} "
                "must use the same ordered case IDs"
            )

    raw_case_sets = manifest.get("case_sets")
    case_set_items = (
        [item for item in raw_case_sets if isinstance(item, dict)]
        if isinstance(raw_case_sets, list)
        else []
    )
    case_set_ids = [
        item.get("id") for item in case_set_items if isinstance(item.get("id"), str)
    ]
    if len(case_set_ids) != len(set(case_set_ids)):
        errors.append(f"{label} manifest case-set IDs must be unique")
    if set(case_set_ids) != set(split_case_sets):
        errors.append(
            f"{label} manifest case sets {sorted(set(case_set_ids))} must equal "
            f"the specification case sets {sorted(split_case_sets)}"
        )

    manifest_relative_directory = resolved_manifest.parent.relative_to(
        resolved_dataset
    )
    for descriptor in case_set_items:
        case_set_id = descriptor.get("id")
        if not isinstance(case_set_id, str):
            continue
        expected_case_ids = split_case_sets.get(case_set_id)
        if expected_case_ids is None:
            continue
        if descriptor.get("case_count") != len(expected_case_ids):
            errors.append(
                f"{label} case set {case_set_id!r} case_count must equal "
                f"{len(expected_case_ids)}"
            )
        index_file = descriptor.get("index_file")
        if not isinstance(index_file, str):
            continue
        index_path = safe_dataset_path(
            errors,
            dataset_directory,
            (manifest_relative_directory / index_file).as_posix(),
            label=f"{label} case set {case_set_id!r} index_file",
        )
        if index_path is None or not index_path.is_file():
            if index_path is not None:
                errors.append(
                    f"{label} case set {case_set_id!r} index_file does not exist"
                )
            continue
        if sha256_file(index_path) != descriptor.get("index_sha256"):
            errors.append(
                f"{label} case set {case_set_id!r} index SHA-256 changed"
            )
        try:
            support_index = load_json(index_path)
        except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
            errors.append(
                f"{label} case set {case_set_id!r} index cannot be read: {error}"
            )
            continue
        for error in schema_errors(support_index, "case-index.schema.json"):
            errors.append(f"{label} case set {case_set_id!r} index {error}")
        index_relative_directory = index_path.parent.resolve().relative_to(
            resolved_dataset
        )
        raw_chunks = support_index.get("chunks")
        chunks = raw_chunks if isinstance(raw_chunks, list) else []
        for chunk in chunks:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("file"), str):
                continue
            chunk_path = safe_dataset_path(
                errors,
                dataset_directory,
                (index_relative_directory / chunk["file"]).as_posix(),
                label=(
                    f"{label} case set {case_set_id!r} "
                    f"chunk {chunk.get('file')!r}"
                ),
            )
            if chunk_path is None or not chunk_path.is_file():
                if chunk_path is not None:
                    errors.append(f"{label} scoring-support chunk is missing")
                continue
            if sha256_file(chunk_path) != chunk.get("sha256"):
                errors.append(
                    f"{label} case set {case_set_id!r} chunk "
                    f"{chunk['file']!r} SHA-256 changed"
                )
            try:
                support_chunk = load_json(chunk_path)
            except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
                errors.append(
                    f"{label} case set {case_set_id!r} chunk cannot be read: {error}"
                )
                continue
            for error in schema_errors(support_chunk, "case-chunk.schema.json"):
                errors.append(
                    f"{label} case set {case_set_id!r} chunk "
                    f"{chunk['file']!r} {error}"
                )

        try:
            release = load_support_release(resolved_manifest, case_set_id)
        except (OSError, KeyError, TypeError, ScoringSupportError) as error:
            errors.append(f"{label} case set {case_set_id!r} is invalid: {error}")
            continue
        if list(release.cases) != expected_case_ids:
            errors.append(
                f"{label} case set {case_set_id!r} cases must exactly match "
                "the ordered official split index"
            )
    return errors


def validate_specification(
    specification_path: Path,
    *,
    spec_root: Path = SPEC_ROOT,
) -> list[str]:
    errors: list[str] = []
    try:
        specification = load_json(specification_path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
        return [f"{specification_path}: cannot read specification: {error}"]
    if not isinstance(specification, dict):
        return [f"{specification_path}: specification must contain a JSON object"]
    dataset_id = specification.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        return [f"{specification_path}: dataset_id must be a non-empty string"]
    expected_directory = spec_root / dataset_id
    if specification_path.parent.resolve() != expected_directory.resolve():
        errors.append(
            f"{dataset_id}: specification directory must match its dataset_id"
        )
    validate_methodology_contract(errors, expected_directory, dataset_id)
    support = specification.get("scoring_support")
    if not isinstance(support, dict):
        return errors + [f"{dataset_id}.scoring_support must be an object"]
    status = support.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(
            f"{dataset_id}.scoring_support.status must be one of "
            f"{sorted(ALLOWED_STATUSES)}"
        )
        return errors
    if status != "official":
        validate_closed_gate(errors, support, label=f"{dataset_id}.scoring_support")
        if status in {"prototype", "owner_review_required"}:
            unexpected = sorted(UNPUBLISHED_FIELDS.intersection(support))
            if unexpected:
                errors.append(
                    f"{dataset_id}.scoring_support status {status!r} must not "
                    f"publish official fields: {unexpected}"
                )
        if status == "owner_review_required":
            decisions = support.get("owner_decisions_required")
            if (
                not isinstance(decisions, list)
                or not decisions
                or not all(isinstance(item, str) and item.strip() for item in decisions)
                or len(decisions) != len(set(decisions))
            ):
                errors.append(
                    f"{dataset_id}.scoring_support.owner_decisions_required must "
                    "be a non-empty list of unique decision IDs"
                )
        return errors
    if not isinstance(support.get("submissions_open"), bool):
        errors.append(
            f"{dataset_id}.scoring_support.submissions_open must be true or false"
        )
    elif support["submissions_open"] is False:
        reason = support.get("closed_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(
                f"{dataset_id}.scoring_support.closed_reason must explain why "
                "this official release is closed"
            )
    validate_declared_manifest(
        errors,
        expected_directory,
        specification,
        support,
    )
    return errors


def validate_all() -> tuple[list[str], int]:
    specification_paths = sorted(SPEC_ROOT.glob("*/submission-spec.json"))
    errors: list[str] = []
    for path in specification_paths:
        errors.extend(validate_specification(path))
    return errors, len(specification_paths)


def main() -> int:
    errors, dataset_count = validate_all()
    if errors:
        print("Scoring-support validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Validated publication gates and declared scoring-support releases "
        f"for {dataset_count} datasets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
