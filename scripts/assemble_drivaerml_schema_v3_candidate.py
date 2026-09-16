#!/usr/bin/env python3
"""Assemble a closed DrivAerML FluidsBench schema-v3 candidate package.

The assembler is intentionally fail-closed.  It creates participant-owned
result files only after the repository and the participant configuration agree
on immutable candidate support, evaluator, profile-v10, and profile-truth
identities.  It does not create scoring support or owner approval metadata.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.methodology import (
    MethodologyError,
    derived_parameter_count_millions,
    require_methodology,
)
from reference.drivaerml.dataset_scorer import (
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_FORMAT,
    DrivAerDatasetScorerError,
    validate_schema_v3_relative_profile_chunk_candidate,
)
from reference.drivaerml.regional_aggregate import (
    AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
    RegionalAggregateError,
    validate_aggregate_regional_diagnostics,
)
from reference.drivaerml.retained_file import RetainedFileError, RetainedVerifiedFile
from scripts.validate_scoring_supports import validate_candidate_manifest_release


DEFAULT_SPECIFICATION = ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json"
SCHEMA_ROOT = ROOT / "schemas"
CONFIG_SCHEMA = "drivaerml-fluidsbench-schema-v3-package-config-v2"
TOKEN_PREFIXES = ("__REPLACE_", "__UNRESOLVED_DRIVAERML_")
SAFE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,159}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")


class PackageAssemblyError(ValueError):
    """Raised when a package cannot be assembled without inventing data."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageAssemblyError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> Any:
    raise PackageAssemblyError(f"JSON contains forbidden non-finite token {token}")


def load_json_with_sha256(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    """Hash and parse the same retained regular-file descriptor."""

    try:
        with RetainedVerifiedFile.open(path, label=label) as retained:
            digest = retained.sha256(chunk_bytes=1024 * 1024)
            retained.handle.seek(0)
            value = json.load(
                retained.handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            retained.assert_unchanged(context="while its JSON was parsed")
    except PackageAssemblyError:
        raise
    except RetainedFileError as error:
        raise PackageAssemblyError(str(error)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageAssemblyError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PackageAssemblyError(f"{path} must contain a JSON object")
    return value, digest


def load_json(path: Path) -> dict[str, Any]:
    return load_json_with_sha256(path, label=str(path))[0]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PackageAssemblyError(f"value is not finite canonical JSON: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_path(parts: Iterable[str | int]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def unresolved_tokens(value: Any, path: tuple[str | int, ...] = ()) -> list[dict[str, str]]:
    """Return every deliberately unresolved string with its JSON path."""

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


def _schema_errors(value: Any, relative_path: str) -> list[str]:
    schema = load_json(SCHEMA_ROOT / relative_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{_json_path(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: _json_path(item.absolute_path),
        )
    ]


def _require_schema(value: Any, relative_path: str, label: str) -> None:
    errors = _schema_errors(value, relative_path)
    if errors:
        raise PackageAssemblyError(
            f"{label} does not satisfy {relative_path}: " + "; ".join(errors)
        )


def _require_methodology_schema(value: Any) -> None:
    """Validate the common method-record fragment before processing evidence."""

    submission_schema = load_json(SCHEMA_ROOT / "v3" / "submission.schema.json")
    fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": submission_schema["$defs"],
        "$ref": "#/$defs/fluidsbench_methodology",
    }
    validator = Draft202012Validator(fragment, format_checker=FormatChecker())
    errors = [
        f"{_json_path(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: _json_path(item.absolute_path),
        )
    ]
    if errors:
        raise PackageAssemblyError(
            "participant methodology does not satisfy the FluidsBench schema: "
            + "; ".join(errors)
        )


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PackageAssemblyError(
            f"{label} keys must be {sorted(keys)}; observed {actual}"
        )
    return value


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PackageAssemblyError(f"{label} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PackageAssemblyError(f"{label} must remain inside {root}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PackageAssemblyError(f"{label} must remain inside {root}")
    return resolved


def _require_identities(
    value: dict[str, Any], expected: dict[str, Any], label: str, *, allow_absent: bool = False
) -> None:
    """Reject conflicting identities before any canonical normalization."""

    for key, identity in expected.items():
        if key not in value and allow_absent:
            continue
        if value.get(key) != identity:
            raise PackageAssemblyError(
                f"{label}.{key} must equal {identity!r}; observed {value.get(key)!r}"
            )


def _find_split(
    specification: dict[str, Any], specification_path: Path, split_id: str
) -> tuple[dict[str, Any], list[str]]:
    matches = [item for item in specification.get("splits", []) if item.get("id") == split_id]
    if len(matches) != 1:
        raise PackageAssemblyError(f"split_id {split_id!r} is not one official DrivAerML split")
    split = matches[0]
    if split.get("case_id_status") != "official":
        raise PackageAssemblyError(f"split {split_id!r} does not have official case IDs")
    split_path = _safe_child(specification_path.parent, split.get("index_file"), "split index_file")
    if not split_path.is_file() or sha256_file(split_path) != split.get("sha256"):
        raise PackageAssemblyError(f"split {split_id!r} index is missing or its SHA-256 changed")
    index = load_json(split_path)
    case_ids = index.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(item, str) and item for item in case_ids)
        or len(case_ids) != len(set(case_ids))
        or index.get("case_count") != len(case_ids)
        or split.get("case_count") != len(case_ids)
    ):
        raise PackageAssemblyError(f"split {split_id!r} has an invalid case index")
    for key, expected in (
        ("dataset_id", "drivaerml"),
        ("split_id", split_id),
        ("case_set_id", split.get("case_set_id")),
        ("case_id_status", "official"),
    ):
        if index.get(key) != expected:
            raise PackageAssemblyError(f"split index {key} must equal {expected!r}")
    return split, list(case_ids)


def _release_binding(
    config: dict[str, Any],
    specification: dict[str, Any],
    specification_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    release = config.get("release_bindings")
    if not isinstance(release, dict):
        raise PackageAssemblyError("config.release_bindings must be an object")
    candidate = _require_exact_keys(
        release.get("candidate_manifest"),
        {"status", "release_id", "manifest_url", "manifest_sha256"},
        "release_bindings.candidate_manifest",
    )
    evaluator = _require_exact_keys(
        release.get("evaluator"),
        {"reference_version", "code_revision"},
        "release_bindings.evaluator",
    )
    profile = _require_exact_keys(
        release.get("profile_definition_v10"),
        {"file", "sha256"},
        "release_bindings.profile_definition_v10",
    )
    ground_truth = _require_exact_keys(
        release.get("profile_ground_truth"),
        {"release_id", "manifest_sha256"},
        "release_bindings.profile_ground_truth",
    )
    if candidate.get("status") != "candidate":
        raise PackageAssemblyError("candidate manifest status must be 'candidate'")
    if not isinstance(candidate.get("release_id"), str) or not SAFE_ID_PATTERN.fullmatch(
        candidate["release_id"]
    ):
        raise PackageAssemblyError("candidate manifest release_id must be a safe ID")
    if not isinstance(candidate.get("manifest_url"), str):
        raise PackageAssemblyError("candidate manifest_url must be a string")
    if not isinstance(candidate.get("manifest_sha256"), str) or not SHA256_PATTERN.fullmatch(
        candidate["manifest_sha256"]
    ):
        raise PackageAssemblyError("candidate manifest_sha256 must be a lowercase SHA-256")
    if (
        not isinstance(evaluator.get("reference_version"), str)
        or not isinstance(evaluator.get("code_revision"), str)
        or not GIT_REVISION_PATTERN.fullmatch(evaluator["code_revision"])
    ):
        raise PackageAssemblyError("candidate evaluator identity is invalid")
    if (
        not isinstance(profile.get("file"), str)
        or not isinstance(profile.get("sha256"), str)
        or not SHA256_PATTERN.fullmatch(profile["sha256"])
    ):
        raise PackageAssemblyError("profile-v10 identity is invalid")
    if (
        not isinstance(ground_truth.get("release_id"), str)
        or not SAFE_ID_PATTERN.fullmatch(ground_truth["release_id"])
        or not isinstance(ground_truth.get("manifest_sha256"), str)
        or not SHA256_PATTERN.fullmatch(ground_truth["manifest_sha256"])
    ):
        raise PackageAssemblyError("profile-ground-truth identity is invalid")

    support = specification.get("scoring_support")
    if not isinstance(support, dict):
        raise PackageAssemblyError("DrivAerML specification has no scoring_support object")
    if support.get("status") not in {"candidate", "owner_review_required"}:
        raise PackageAssemblyError("DrivAerML must remain a closed candidate for this workflow")
    if support.get("submissions_open") is not False:
        raise PackageAssemblyError("candidate assembly requires submissions_open=false")
    owner_candidate = support.get("candidate_manifest")
    if not isinstance(owner_candidate, dict):
        raise PackageAssemblyError(
            "the repository has not published scoring_support.candidate_manifest; "
            "release tokens must not be guessed"
        )
    for key, expected in candidate.items():
        if owner_candidate.get(key) != expected:
            raise PackageAssemblyError(
                f"config candidate_manifest.{key} does not match the repository binding"
            )
    manifest_path = _safe_child(
        specification_path.parent,
        owner_candidate.get("manifest_file"),
        "scoring_support.candidate_manifest.manifest_file",
    )
    if not manifest_path.is_file():
        raise PackageAssemblyError("candidate scoring-support manifest is missing")
    manifest, actual_manifest_sha256 = load_json_with_sha256(
        manifest_path,
        label="candidate scoring-support manifest",
    )
    if actual_manifest_sha256 != candidate["manifest_sha256"]:
        raise PackageAssemblyError("candidate scoring-support manifest is missing or its SHA-256 changed")
    parsed_manifest_url = urlparse(candidate["manifest_url"])
    if (
        parsed_manifest_url.scheme != "https"
        or not parsed_manifest_url.netloc
        or parsed_manifest_url.username is not None
        or parsed_manifest_url.password is not None
        or parsed_manifest_url.query
        or parsed_manifest_url.fragment
        or candidate["release_id"]
        not in [part for part in parsed_manifest_url.path.split("/") if part]
    ):
        raise PackageAssemblyError("candidate manifest URL is not a clean release-scoped HTTPS URL")
    for key, expected in {
        "status": "candidate",
        "release_id": candidate["release_id"],
        "dataset_id": "drivaerml",
        "dataset_version": specification.get("dataset_version"),
        "evaluation_reference_version": specification.get("evaluation_reference_version"),
    }.items():
        if manifest.get(key) != expected:
            raise PackageAssemblyError(
                f"candidate scoring-support manifest {key} must equal {expected!r}"
            )
    if "owner_approval" in manifest:
        raise PackageAssemblyError("candidate scoring-support manifest must not claim owner approval")
    support_errors = validate_candidate_manifest_release(
        specification,
        specification_path.parent,
        manifest_path,
        manifest,
    )
    if support_errors:
        raise PackageAssemblyError(
            "candidate scoring-support manifest is not valid support: "
            + "; ".join(support_errors)
        )

    owner_evaluator = support.get("dataset_evaluator_binding", {})
    if (
        evaluator.get("reference_version") != specification.get("evaluation_reference_version")
        or evaluator.get("reference_version") != owner_evaluator.get("evaluator_reference_version")
        or evaluator.get("code_revision") != owner_evaluator.get("evaluator_code_revision")
    ):
        raise PackageAssemblyError("config evaluator identity does not match the repository binding")
    if owner_evaluator.get("status") != "frozen":
        raise PackageAssemblyError("candidate evaluator is not yet frozen")

    active_profile = specification.get("profile_definition", {})
    if (
        profile.get("file") != active_profile.get("file")
        or profile.get("sha256") != active_profile.get("sha256")
        or "v10" not in str(profile.get("file"))
    ):
        raise PackageAssemblyError("config profile-v10 identity is not the active repository profile")

    owner_truth = support.get("profile_definition", {}).get("profile_ground_truth")
    if not isinstance(owner_truth, dict):
        raise PackageAssemblyError("repository profile-ground-truth release is not published")
    global_truth = load_json(ROOT / "leaderboard" / "manifest.json").get(
        "data_release", {}
    ).get("profile_ground_truth")
    if not isinstance(global_truth, dict):
        raise PackageAssemblyError("leaderboard manifest has no profile-ground-truth binding")
    for key, expected in ground_truth.items():
        if owner_truth.get(key) != expected:
            raise PackageAssemblyError(
                f"config profile_ground_truth.{key} does not match the repository binding"
            )
        if global_truth.get(key) != expected:
            raise PackageAssemblyError(
                f"config profile_ground_truth.{key} does not match the global leaderboard binding"
            )
    return candidate, evaluator, ground_truth


def _normalize_case_metrics(
    source: Path,
    *,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    case_ids: list[str],
    release_id: str,
    manifest_sha256: str,
    required_metric_ids: list[str],
) -> dict[str, Any]:
    value = load_json(source)
    identities = {
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": split_id,
        "case_set_id": case_set_id,
        "scoring_support_release_id": release_id,
        "scoring_support_manifest_sha256": manifest_sha256,
    }
    _require_identities(value, identities, "metrics/cases.json")
    _require_schema(value, "v3/case-metrics.schema.json", "metrics/cases.json")
    cases = value.get("cases")
    observed = [item.get("case_id") for item in cases] if isinstance(cases, list) else None
    if observed != case_ids or value.get("case_count") != len(case_ids):
        raise PackageAssemblyError("metrics/cases.json must contain the exact ordered split case list")
    metric_values = value.get("metric_values")
    if not isinstance(metric_values, dict) or set(metric_values) != set(required_metric_ids):
        missing = sorted(set(required_metric_ids) - set(metric_values or {}))
        extra = sorted(set(metric_values or {}) - set(required_metric_ids))
        raise PackageAssemblyError(
            "metrics/cases.json must contain every specification metric; "
            f"missing={missing}, extra={extra}. The current candidate reducer does "
            "not yet emit the composite/group/R2 set required for a package."
        )
    for metric_id, metric_value in metric_values.items():
        if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool) or not math.isfinite(metric_value):
            raise PackageAssemblyError(f"metric_values.{metric_id} must be finite")
    value["metric_values"] = {
        metric_id: metric_values[metric_id] for metric_id in required_metric_ids
    }
    _require_schema(value, "v3/case-metrics.schema.json", "metrics/cases.json")
    return value


def _required_profile_series(
    specification: dict[str, Any], *, prediction_scope: str
) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for panel in specification.get("profile_panels", []):
        if not panel.get("required"):
            continue
        if prediction_scope == "surface_only" and panel.get("id") == "velocity_profiles":
            continue
        for station in panel.get("station_ids", []):
            for quantity in panel.get("quantity_ids", []):
                result.append((panel["id"], station, quantity))
    if not result:
        raise PackageAssemblyError("DrivAerML specification has no required profile series")
    return result


def _copy_regional_diagnostics(
    source: Path,
    destination: Path,
    *,
    case_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    """Validate and copy one compact report without changing its values."""

    report = load_json(source)
    try:
        validate_aggregate_regional_diagnostics(
            report,
            expected_case_ids=case_ids,
        )
    except RegionalAggregateError as error:
        raise PackageAssemblyError(
            f"regional-diagnostics.json is invalid: {error}"
        ) from error
    write_json(destination, report)
    digest = sha256_file(destination)
    return digest, {
        "format": AGGREGATE_REGIONAL_REPORT_SCHEMA,
        "file": "regional-diagnostics.json",
        "sha256": digest,
        "contract_sha256": REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
        "case_count": len(case_ids),
        "role": "report_only",
        "weight": 0.0,
        "official_score_changed": False,
    }


def _copy_profiles(
    source: Path,
    destination: Path,
    *,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    case_ids: list[str],
    required_series: list[tuple[str, str, str]],
    relative_contract_sha256: str | None = None,
) -> tuple[str, str]:
    index = load_json(source / "index.json")
    profile_format = index.get("format") or "fluidsbench-profile-chunks-v1"
    if profile_format not in {
        "fluidsbench-profile-chunks-v1",
        RELATIVE_PROFILE_FORMAT,
    }:
        raise PackageAssemblyError(f"unsupported profile index format {profile_format!r}")
    relative_profile = profile_format == RELATIVE_PROFILE_FORMAT
    if relative_profile:
        if (
            not isinstance(relative_contract_sha256, str)
            or index.get("contract_id") != RELATIVE_PROFILE_CONTRACT_ID
            or index.get("contract_sha256") != relative_contract_sha256
        ):
            raise PackageAssemblyError(
                "relative profile index does not match the retained benchmark contract"
            )
    index_identities = {
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": split_id,
        "case_set_id": case_set_id,
        "case_count": len(case_ids),
        "case_id_status": "official",
    }
    _require_identities(index, index_identities, "profiles/index.json")
    _require_schema(index, "v1/profile-index.schema.json", "profiles/index.json")
    observed_cases: list[str] = []
    normalized_chunks: list[dict[str, Any]] = []
    for chunk_number, binding in enumerate(index.get("chunks", [])):
        expected_name = f"chunk-{chunk_number:03d}.json"
        if binding.get("file") != expected_name:
            raise PackageAssemblyError("profile chunks must be contiguous chunk-NNN.json files")
        source_chunk = _safe_child(source, expected_name, "profile chunk")
        if sha256_file(source_chunk) != binding.get("sha256"):
            raise PackageAssemblyError(
                f"source {expected_name} SHA-256 does not match its input index binding"
            )
        chunk = load_json(source_chunk)
        if relative_profile:
            _require_schema(
                chunk,
                "v1/drivaerml-relative-profile-chunk.schema.json",
                expected_name,
            )
            try:
                chunk = validate_schema_v3_relative_profile_chunk_candidate(
                    chunk,
                    expected_contract_sha256=relative_contract_sha256,
                )
            except DrivAerDatasetScorerError as error:
                raise PackageAssemblyError(
                    f"{expected_name} relative profile contract is invalid: {error}"
                ) from error
        else:
            _require_schema(chunk, "v1/profile-chunk.schema.json", expected_name)
        chunk_case_ids = [case.get("case_id") for case in chunk["cases"]]
        if binding.get("case_ids") != chunk_case_ids:
            raise PackageAssemblyError(f"{expected_name} case IDs do not match its index binding")
        for case in chunk["cases"]:
            if not relative_profile:
                by_identity = {
                    (series["panel_id"], series["station_id"], series["quantity_id"]): series
                    for series in case["series"]
                }
                if (
                    len(by_identity) != len(case["series"])
                    or set(by_identity) != set(required_series)
                ):
                    raise PackageAssemblyError(
                        f"profile case {case['case_id']} must contain each required series exactly once"
                    )
                case["series"] = [by_identity[identity] for identity in required_series]
            for series in case["series"]:
                if series.get("representation", "materialized") == "materialized" and (
                    len(series["coordinate"]) != len(series["prediction"])
                ):
                    raise PackageAssemblyError(
                        f"profile case {case['case_id']} has unequal coordinate/prediction lengths"
                    )
        destination_chunk = destination / expected_name
        write_json(destination_chunk, chunk)
        normalized_chunks.append(
            {
                "file": expected_name,
                "case_ids": chunk_case_ids,
                "sha256": sha256_file(destination_chunk),
            }
        )
        observed_cases.extend(chunk_case_ids)
    if observed_cases != case_ids:
        raise PackageAssemblyError("profile index must contain the exact ordered split case list")
    index["chunks"] = normalized_chunks
    _require_schema(index, "v1/profile-index.schema.json", "profiles/index.json")
    index_path = destination / "index.json"
    write_json(index_path, index)
    return sha256_file(index_path), profile_format


def _normalize_discretization_cases(
    source: Path,
    destination: Path,
    *,
    submission_id: str,
    split_id: str,
    case_ids: list[str],
    case_metrics: dict[str, Any],
) -> str:
    try:
        lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = [
            json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            for line in lines
        ]
    except PackageAssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageAssemblyError(f"cannot read discretization cases: {error}") from error
    if not all(isinstance(record, dict) for record in records):
        raise PackageAssemblyError("every discretization case line must be a JSON object")
    if [record.get("case_id") for record in records] != case_ids:
        raise PackageAssemblyError("discretization cases must match the exact ordered split case list")
    metric_cases = {case["case_id"]: case for case in case_metrics["cases"]}
    normalized_lines: list[bytes] = []
    for record in records:
        identities = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": "drivaerml",
            "split_id": split_id,
        }
        _require_identities(
            record,
            identities,
            f"discretization case {record.get('case_id')}",
            allow_absent=True,
        )
        record.update(identities)
        support_counts = {
            support["support_id"]: (support["support_count"], support["scored_count"])
            for support in metric_cases[record["case_id"]]["supports"]
        }
        for mapping in record.get("inference", {}).get("mappings", []):
            expected = support_counts.get(mapping.get("support_id"))
            if expected is not None and (
                mapping.get("support_count"), mapping.get("scored_count")
            ) != expected:
                raise PackageAssemblyError(
                    f"discretization mapping counts disagree with metrics for {record['case_id']}"
                )
        _require_schema(
            record,
            "v3/discretization-case.schema.json",
            f"discretization case {record['case_id']}",
        )
        normalized_lines.append(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"\n".join(normalized_lines) + b"\n")
    return sha256_file(destination)


def assemble_package(
    *,
    config_path: Path,
    specification_path: Path,
    case_metrics_path: Path,
    profiles_path: Path,
    discretization_cases_path: Path,
    output_path: Path,
    regional_diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise PackageAssemblyError(f"config.schema must equal {CONFIG_SCHEMA!r}")
    blockers = unresolved_tokens(config)
    if blockers:
        raise PackageAssemblyError(
            "configuration contains unresolved release or participant tokens: "
            + ", ".join(item["path"] for item in blockers)
        )
    if output_path.exists() or output_path.is_symlink():
        raise PackageAssemblyError(f"output already exists: {output_path}")

    specification = load_json(specification_path)
    if specification.get("dataset_id") != "drivaerml":
        raise PackageAssemblyError("assembler accepts only the DrivAerML specification")
    methodology_contract_path = specification_path.parent / "methodology-contract.json"
    if not methodology_contract_path.is_file():
        raise PackageAssemblyError("DrivAerML methodology contract is missing")
    methodology_contract = load_json(methodology_contract_path)
    split_id = config.get("split_id")
    if not isinstance(split_id, str):
        raise PackageAssemblyError("config.split_id must be a string")
    split, case_ids = _find_split(specification, specification_path, split_id)
    prediction_scope = config.get("prediction_scope", "surface_and_volume")
    scope_declarations = specification.get("scoring_support", {}).get(
        "prediction_scopes", {}
    )
    if not isinstance(scope_declarations, dict):
        scope_declarations = {}
    if prediction_scope not in {"surface_and_volume", "surface_only"} or (
        prediction_scope == "surface_only"
        and not isinstance(scope_declarations.get(prediction_scope), dict)
    ):
        raise PackageAssemblyError(
            "config.prediction_scope must be a supported DrivAerML prediction scope"
        )
    regional_contract = specification.get("regional_diagnostics")
    regional_required = (
        isinstance(regional_contract, dict)
        and regional_contract.get("required_for_new_submissions") is True
    )
    if isinstance(regional_contract, dict) and (
        regional_contract.get("format") != AGGREGATE_REGIONAL_REPORT_SCHEMA
        or regional_contract.get("contract_sha256")
        != REGIONAL_DIAGNOSTICS_CONTRACT_SHA256
        or regional_contract.get("role") != "report_only"
        or regional_contract.get("weight") != 0.0
        or regional_contract.get("affects_official_score") is not False
    ):
        raise PackageAssemblyError(
            "the DrivAerML regional diagnostics declaration differs from the "
            "frozen zero-weight contract"
        )
    if regional_required and regional_diagnostics_path is None:
        raise PackageAssemblyError(
            "the DrivAerML specification requires --regional-diagnostics"
        )
    candidate, evaluator, ground_truth = _release_binding(
        config, specification, specification_path
    )

    profile_config = config["release_bindings"]["profile_definition_v10"]
    profile_path = _safe_child(specification_path.parent, profile_config["file"], "profile_definition_v10.file")
    if not profile_path.is_file():
        raise PackageAssemblyError("active profile-v10 file is missing")
    profile_document, actual_profile_sha256 = load_json_with_sha256(
        profile_path,
        label="active profile-v10 artifact",
    )
    if actual_profile_sha256 != profile_config["sha256"]:
        raise PackageAssemblyError("active profile-v10 file is missing or its SHA-256 changed")
    active_profile = specification.get("profile_definition")
    if (
        not isinstance(active_profile, dict)
        or profile_document.get("id") != active_profile.get("id")
        or "v10" not in str(profile_document.get("id", ""))
    ):
        raise PackageAssemblyError(
            "active profile-v10 artifact identity does not match the specification"
        )

    participant = config.get("participant")
    if not isinstance(participant, dict):
        raise PackageAssemblyError("config.participant must be an object")
    forbidden = {
        "dataset", "dataset_id", "dataset_version", "split", "split_id",
        "case_set_id", "split_sha256", "evaluation", "scoring_support",
        "spatial_discretization", "case_metrics", "metric_values", "profile_data",
        "regional_diagnostics",
        "approval", "prediction_artifacts", "parameter_count_millions",
    }
    overlap = sorted(forbidden.intersection(participant))
    if overlap:
        raise PackageAssemblyError(f"participant config must not override benchmark fields: {overlap}")
    submission_id = participant.get("submission_id")
    if not isinstance(submission_id, str):
        raise PackageAssemblyError("participant.submission_id must be a string")
    participant_submission = copy.deepcopy(participant)
    participant_submission["dataset_id"] = "drivaerml"
    participant_submission["prediction_scope"] = prediction_scope
    try:
        participant_submission["parameter_count_millions"] = (
            derived_parameter_count_millions(participant.get("methodology"))
        )
        _require_methodology_schema(participant.get("methodology"))
        require_methodology(
            participant_submission,
            expected_case_count=len(case_ids),
            contract=methodology_contract,
        )
    except MethodologyError as error:
        raise PackageAssemblyError(
            f"participant methodology is invalid: {error}"
        ) from error
    unavailable_metric_ids = set(
        scope_declarations.get(prediction_scope, {}).get(
            "unavailable_component_metric_ids", []
        )
        if prediction_scope == "surface_only"
        else []
    )
    required_metric_ids = [
        metric["id"]
        for metric in specification.get("metrics", [])
        if metric["id"] not in unavailable_metric_ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent))
    try:
        case_metrics = _normalize_case_metrics(
            case_metrics_path,
            submission_id=submission_id,
            split_id=split_id,
            case_set_id=split["case_set_id"],
            case_ids=case_ids,
            release_id=candidate["release_id"],
            manifest_sha256=candidate["manifest_sha256"],
            required_metric_ids=required_metric_ids,
        )
        normalized_metrics_path = staging / "metrics" / "cases.json"
        write_json(normalized_metrics_path, case_metrics)
        case_metrics_sha256 = sha256_file(normalized_metrics_path)

        cases_sha256 = _normalize_discretization_cases(
            discretization_cases_path,
            staging / "discretization" / "cases.jsonl",
            submission_id=submission_id,
            split_id=split_id,
            case_ids=case_ids,
            case_metrics=case_metrics,
        )
        spatial = config.get("spatial_discretization")
        if not isinstance(spatial, dict):
            raise PackageAssemblyError("config.spatial_discretization must be an object")
        discretization = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": "drivaerml",
            "split_id": split_id,
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "training": copy.deepcopy(spatial.get("training")),
            "inference": copy.deepcopy(spatial.get("inference")),
            "case_manifest": {
                "format": "jsonl",
                "file": "discretization/cases.jsonl",
                "sha256": cases_sha256,
                "case_count": len(case_ids),
            },
        }
        if "notes" in spatial:
            discretization["notes"] = spatial["notes"]
        _require_schema(discretization, "v3/discretization.schema.json", "discretization.json")
        discretization_path = staging / "discretization.json"
        write_json(discretization_path, discretization)
        discretization_sha256 = sha256_file(discretization_path)

        relative_declaration = specification.get("relative_diagnostics", {})
        relative_contract = (
            relative_declaration.get("contract")
            if isinstance(relative_declaration, dict)
            else None
        )
        relative_contract_sha256 = (
            relative_contract.get("sha256")
            if isinstance(relative_contract, dict)
            else None
        )
        profile_index_sha256, profile_format = _copy_profiles(
            profiles_path,
            staging / "profiles",
            submission_id=submission_id,
            split_id=split_id,
            case_set_id=split["case_set_id"],
            case_ids=case_ids,
            required_series=_required_profile_series(
                specification, prediction_scope=prediction_scope
            ),
            relative_contract_sha256=relative_contract_sha256,
        )

        regional_sha256: str | None = None
        regional_declaration: dict[str, Any] | None = None
        if regional_diagnostics_path is not None:
            regional_sha256, regional_declaration = _copy_regional_diagnostics(
                regional_diagnostics_path,
                staging / "regional-diagnostics.json",
                case_ids=case_ids,
            )

        evaluation = config.get("evaluation")
        if not isinstance(evaluation, dict):
            raise PackageAssemblyError("config.evaluation must be an object")
        evidence = {
            "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
            "schema_version": "3.0",
            "submission_id": submission_id,
            "dataset_id": "drivaerml",
            "dataset_version": specification["dataset_version"],
            "split_id": split_id,
            "split_sha256": split["sha256"],
            "case_set_id": split["case_set_id"],
            "prediction_scope": prediction_scope,
            "reference_version": evaluator["reference_version"],
            "command": evaluation.get("command"),
            "generated_at": evaluation.get("generated_at"),
            "status": "submitted_evaluation",
            "metric_values": case_metrics["metric_values"],
            "profile_index_sha256": profile_index_sha256,
            "profile_ground_truth_release_id": ground_truth["release_id"],
            "profile_ground_truth_manifest_sha256": ground_truth["manifest_sha256"],
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "discretization_sha256": discretization_sha256,
            "case_metrics_sha256": case_metrics_sha256,
        }
        if regional_sha256 is not None:
            evidence["regional_diagnostics_sha256"] = regional_sha256
        reproducibility = participant.get("reproducibility")
        reproducibility_code = (
            reproducibility.get("code") if isinstance(reproducibility, dict) else None
        )
        participant_code_revision = (
            reproducibility_code.get("commit")
            if isinstance(reproducibility_code, dict)
            else None
        )
        if participant_code_revision is not None:
            evidence["code_revision"] = participant_code_revision
        _require_schema(evidence, "v3/evaluation-evidence.schema.json", "evaluation-evidence.json")
        evidence_path = staging / "evaluation-evidence.json"
        write_json(evidence_path, evidence)

        submission = {
            "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
            "schema_version": "3.0",
            **participant_submission,
            "dataset": specification["dataset_name"],
            "dataset_id": "drivaerml",
            "dataset_version": specification["dataset_version"],
            "split": split["label"],
            "split_id": split_id,
            "case_set_id": split["case_set_id"],
            "split_sha256": split["sha256"],
            "prediction_scope": prediction_scope,
            "evaluation": {
                "reference_version": evaluator["reference_version"],
                "command": evaluation.get("command"),
                "evidence_file": "evaluation-evidence.json",
                "evidence_sha256": sha256_file(evidence_path),
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
                "sha256": discretization_sha256,
            },
            "case_metrics": {
                "format": "fluidsbench-case-metrics-v1",
                "file": "metrics/cases.json",
                "sha256": case_metrics_sha256,
                "case_count": len(case_ids),
            },
            "metric_values": case_metrics["metric_values"],
            "profile_data": {
                "format": profile_format,
                "index_file": "profiles/index.json",
                "case_count": len(case_ids),
                "case_set_id": split["case_set_id"],
                "profile_ground_truth_release_id": ground_truth["release_id"],
                "profile_ground_truth_manifest_sha256": ground_truth["manifest_sha256"],
            },
        }
        if regional_declaration is not None:
            submission["regional_diagnostics"] = regional_declaration
        if participant_code_revision is not None:
            submission["evaluation"]["code_revision"] = participant_code_revision
        _require_schema(submission, "v3/submission.schema.json", "submission.json")
        write_json(staging / "submission.json", submission)

        staging.rename(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "candidate_package_assembled_not_approved",
        "output": str(output_path),
        "submission_id": submission_id,
        "split_id": split_id,
        "case_count": len(case_ids),
        "candidate_dry_run_command": (
            f"python scripts/validate_submission.py --candidate-dry-run {output_path}"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--submission-specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--case-metrics", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument(
        "--regional-diagnostics",
        type=Path,
        help="Split-level zero-weight regional-diagnostics.json from the evaluator.",
    )
    parser.add_argument("--discretization-cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--list-unresolved",
        action="store_true",
        help="print unresolved participant/release tokens as JSON without assembling",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_json(args.config)
        if args.list_unresolved:
            found = unresolved_tokens(config)
            print(
                json.dumps(
                    {
                        "status": (
                            "blocked_by_tokens"
                            if found
                            else "tokens_resolved_repository_and_inputs_not_validated"
                        ),
                        "unresolved": found,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        missing = [
            name
            for name in ("case_metrics", "profiles", "discretization_cases", "output")
            if getattr(args, name) is None
        ]
        if missing:
            raise PackageAssemblyError(
                "assembly requires arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        result = assemble_package(
            config_path=args.config,
            specification_path=args.submission_specification,
            case_metrics_path=args.case_metrics,
            profiles_path=args.profiles,
            discretization_cases_path=args.discretization_cases,
            output_path=args.output,
            regional_diagnostics_path=args.regional_diagnostics,
        )
    except PackageAssemblyError as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, sort_keys=True), file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
