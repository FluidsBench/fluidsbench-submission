#!/usr/bin/env python3
"""Validate source submissions and build compact public leaderboard feeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit

if __package__:
    from .validate_submission import (
        load_json,
        manifest_with_benchmark_contract,
        normalized_result_revision,
        registered_ahmedml_development_fixture,
        registered_ahmedml_pre_release_reference,
        registered_hiliftaeroml_preview,
        schema_errors,
        sha256_file,
        submission_files,
        validate_many,
        validate_submission_file,
    )
else:
    from validate_submission import (
        load_json,
        manifest_with_benchmark_contract,
        normalized_result_revision,
        registered_ahmedml_development_fixture,
        registered_ahmedml_pre_release_reference,
        registered_hiliftaeroml_preview,
        schema_errors,
        sha256_file,
        submission_files,
        validate_many,
        validate_submission_file,
    )


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "leaderboard" / "manifest.json"
CLAIMS_ROOT = ROOT / "leaderboard" / "claims"
CLAIMS_INDEX_PATH = CLAIMS_ROOT / "index.json"
REVISION_HISTORY_PATH = ROOT / "leaderboard" / "revisions.json"
CLAIM_SCHEMA_URL = "https://fluidsbench.org/schemas/releases/result-claim.schema.json"
CLAIM_INDEX_SCHEMA_URL = "https://fluidsbench.org/schemas/releases/claim-index.schema.json"
REVISION_HISTORY_SCHEMA_URL = "https://fluidsbench.org/schemas/releases/revision-history.schema.json"
RANKING_METHOD = "competition"
RANKING_ROUNDING = "decimal_half_up"
RANKING_SCOPE = ["release_id", "dataset_id", "split_id"]
RELEASE_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,158}[a-z0-9])?$")
METRIC_DEFINITION_OVERRIDE_KEYS = frozenset({"label", "description"})


def ranking_contract() -> dict[str, Any]:
    return {
        "version": "1.0",
        "scope": RANKING_SCOPE,
        "method": RANKING_METHOD,
        "tie_sequence_example": [1, 2, 2, 4],
        "comparison": "rounded_metric_value",
        "rounding": RANKING_ROUNDING,
    }


def is_release_id(value: Any) -> bool:
    return isinstance(value, str) and RELEASE_ID_PATTERN.fullmatch(value) is not None


def https_url_parts(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        parts.port
    except ValueError:
        return None
    if (
        parts.scheme != "https"
        or not parts.netloc
        or not hostname
        or parts.username is not None
        or parts.password is not None
    ):
        return None
    return parts


def path_has_release_segment(path: str, release_id: str, *, final: bool = False) -> bool:
    segments = [unquote(segment) for segment in path.split("/") if segment]
    if final:
        return bool(segments) and segments[-1] == release_id
    return release_id in segments


def is_clean_https_directory_base(value: Any, *, release_id: str | None = None) -> bool:
    parts = https_url_parts(value)
    return (
        parts is not None
        and parts.path.endswith("/")
        and not parts.query
        and not parts.fragment
        and (release_id is None or path_has_release_segment(parts.path, release_id, final=True))
    )


def is_valid_published_at(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def release_contract_errors(manifest: dict[str, Any]) -> list[str]:
    """Return release-level provenance errors that must block an official feed."""

    release = manifest.get("data_release", {})
    errors: list[str] = []
    if manifest.get("schema_version") != "0.6.0":
        errors.append("leaderboard/manifest.json schema_version must be 0.6.0")
    if manifest.get("ranking_contract") != ranking_contract():
        errors.append("leaderboard/manifest.json ranking_contract does not match the published ranking policy")
    if release.get("status") not in {"prototype_dummy_data", "official"}:
        errors.append("leaderboard/manifest.json data_release.status must be prototype_dummy_data or official")
    if release.get("reproducibility_contract_version") not in {
        "open-reproducibility-2.0",
        "open-reproducibility-3.0",
    }:
        errors.append(
            "leaderboard/manifest.json data_release.reproducibility_contract_version must be "
            "open-reproducibility-2.0 or open-reproducibility-3.0"
        )
    license_metadata = release.get("license")
    if not isinstance(license_metadata, dict) or any(
        not isinstance(license_metadata.get(key), str) or not license_metadata[key].strip()
        for key in ("spdx_id", "name", "url", "scope")
    ):
        errors.append("leaderboard/manifest.json data_release.license must define spdx_id, name, url, and scope")
    archive_url = release.get("archive_url")
    if release.get("status") == "official" and https_url_parts(archive_url) is None:
        errors.append("an official data release requires an immutable HTTPS data_release.archive_url")
    source_commit = release.get("source_commit")
    if release.get("status") == "official" and not (
        isinstance(source_commit, str) and re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", source_commit)
    ):
        errors.append("an official data release requires a full immutable data_release.source_commit")
    release_view_url = release.get("release_view_url")
    release_id = release.get("id")
    if release.get("status") == "official" and not is_release_id(release_id):
        errors.append(
            "an official data_release.id must be a lowercase release slug using only letters, digits, dots, and hyphens"
        )
    if release.get("status") == "official" and (
        not is_release_id(release_id)
        or not is_clean_https_directory_base(release_view_url, release_id=release_id)
    ):
        errors.append(
            "an official data release requires a clean immutable HTTPS directory data_release.release_view_url "
            "whose final path segment is the release ID, ending in '/', and having no query or fragment"
        )
    if release_view_url is not None and https_url_parts(release_view_url) is None:
        errors.append("data_release.release_view_url must be null or an HTTPS URL")
    asset_base_url = release.get("asset_base_url")
    if https_url_parts(asset_base_url) is None:
        errors.append("leaderboard/manifest.json data_release.asset_base_url must be an HTTPS URL")
    elif release.get("status") == "official" and (
        not is_release_id(release_id)
        or not is_clean_https_directory_base(asset_base_url, release_id=release_id)
    ):
        errors.append(
            "an official data_release.asset_base_url must be a clean HTTPS directory whose final path segment is "
            "the immutable release ID, ending in '/', and having no query or fragment"
        )
    if release.get("status") == "official":
        published_at = release.get("generated_at")
        if not is_valid_published_at(published_at):
            errors.append("an official data release requires an explicit timezone-qualified data_release.generated_at")
        if manifest.get("generated_at") != published_at:
            errors.append("official manifest generated_at must equal data_release.generated_at")
    profile_ground_truth = release.get("profile_ground_truth")
    if not isinstance(profile_ground_truth, dict) or any(
        not isinstance(profile_ground_truth.get(key), str) or not profile_ground_truth[key].strip()
        for key in ("release_id", "manifest_url", "manifest_sha256")
    ):
        errors.append(
            "leaderboard/manifest.json data_release.profile_ground_truth must define release_id, manifest_url, and manifest_sha256"
        )
    elif https_url_parts(profile_ground_truth["manifest_url"]) is None or not re.fullmatch(
        r"[a-f0-9]{64}", profile_ground_truth["manifest_sha256"]
    ):
        errors.append("data_release.profile_ground_truth requires an HTTPS manifest_url and lowercase SHA-256 digest")
    elif release.get("status") == "official":
        ground_truth_parts = urlsplit(profile_ground_truth["manifest_url"])
        if (
            not is_release_id(profile_ground_truth["release_id"])
            or not path_has_release_segment(ground_truth_parts.path, profile_ground_truth["release_id"])
            or ground_truth_parts.query
            or ground_truth_parts.fragment
        ):
            errors.append(
                "an official profile-ground-truth manifest_url must use a safe lowercase release ID appearing as "
                "an exact path segment, with no query or fragment"
            )
    if archive_url is not None and https_url_parts(archive_url) is None:
        errors.append("data_release.archive_url must be null or an HTTPS URL")

    definitions = {definition.get("id"): definition for definition in manifest.get("metric_definitions", [])}
    for dataset in manifest.get("datasets", []):
        ranking = dataset.get("ranking")
        label = dataset.get("slug", dataset.get("name", "unknown dataset"))
        overrides = dataset.get("metric_definition_overrides")
        if overrides is not None:
            if not isinstance(overrides, dict):
                errors.append(f"dataset {label} metric_definition_overrides must be an object")
            else:
                metric_ids = dataset.get("metric_ids", [])
                active_metric_ids = set(metric_ids) if isinstance(metric_ids, list) else set()
                for metric_id, override in overrides.items():
                    if metric_id not in definitions:
                        errors.append(
                            f"dataset {label} metric_definition_overrides references unknown metric {metric_id}"
                        )
                    elif metric_id not in active_metric_ids:
                        errors.append(
                            f"dataset {label} metric_definition_overrides metric {metric_id} is not in metric_ids"
                        )
                    if not isinstance(override, dict):
                        errors.append(
                            f"dataset {label} metric_definition_overrides.{metric_id} must be an object"
                        )
                        continue
                    unsupported_keys = set(override) - METRIC_DEFINITION_OVERRIDE_KEYS
                    if unsupported_keys:
                        errors.append(
                            f"dataset {label} metric_definition_overrides.{metric_id} may only define "
                            f"label and description; unsupported={sorted(unsupported_keys)}"
                        )
                    presentation_keys = set(override) & METRIC_DEFINITION_OVERRIDE_KEYS
                    if not presentation_keys:
                        errors.append(
                            f"dataset {label} metric_definition_overrides.{metric_id} must define label or description"
                        )
                    for key in presentation_keys:
                        if not isinstance(override[key], str) or not override[key].strip():
                            errors.append(
                                f"dataset {label} metric_definition_overrides.{metric_id}.{key} "
                                "must be a non-empty string"
                            )
        if not isinstance(ranking, dict):
            errors.append(f"dataset {label} must define its ranking policy")
            continue
        required_ranking = {
            "method": RANKING_METHOD,
            "rounding": RANKING_ROUNDING,
        }
        if any(ranking.get(key) != value for key, value in required_ranking.items()):
            errors.append(f"dataset {label} ranking must use competition ranking and decimal_half_up rounding")
        decimal_places = ranking.get("decimal_places")
        if not isinstance(decimal_places, int) or isinstance(decimal_places, bool) or not 0 <= decimal_places <= 12:
            errors.append(f"dataset {label} ranking.decimal_places must be an integer from 0 to 12")
        definition = definitions.get(ranking.get("metric_id"))
        if definition is None:
            errors.append(f"dataset {label} ranking.metric_id is not in metric_definitions")
        elif ranking.get("direction") != definition.get("direction"):
            errors.append(f"dataset {label} ranking.direction must match the metric definition")
        elif decimal_places != definition.get("digits"):
            errors.append(f"dataset {label} ranking.decimal_places must match the metric display digits")
    return errors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def json_bytes(value: Any) -> bytes:
    return f"{json.dumps(value, indent=2, ensure_ascii=True)}\n".encode()


def source_rows_by_dataset(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {dataset["name"]: [] for dataset in manifest["datasets"]}
    release_status = manifest.get("data_release", {}).get("status")
    allowed_approval_status = "prototype" if release_status == "prototype_dummy_data" else "approved"
    for path in submission_files():
        submission = load_json(path)
        registered_preview = registered_hiliftaeroml_preview(
            path,
            submission,
            manifest,
            root=ROOT,
        )
        ahmedml_pre_release = registered_ahmedml_pre_release_reference(
            path,
            submission,
            manifest,
            root=ROOT,
        )
        development_fixture = registered_ahmedml_development_fixture(
            path,
            submission,
            manifest,
            root=ROOT,
        )
        if (
            registered_preview is None
            and ahmedml_pre_release is None
            and development_fixture is None
            and submission.get("approval", {}).get("status")
            != allowed_approval_status
        ):
            continue
        row = deepcopy(submission)
        row.pop("$schema", None)
        if registered_preview is not None:
            row["record_type"] = registered_preview["record_type"]
        elif ahmedml_pre_release is not None:
            row["record_type"] = ahmedml_pre_release["record_type"]
        elif development_fixture is not None:
            row["record_type"] = development_fixture["record_type"]
        row["parameter_count"] = row.get("parameter_count_millions")
        row["profile_data"]["index_file"] = str(
            (path.parent / row["profile_data"]["index_file"]).relative_to(ROOT)
        )
        profile_index_path = ROOT / row["profile_data"]["index_file"]
        if profile_index_path.is_file():
            row["profile_data"]["index_sha256"] = sha256_file(profile_index_path)
        regional = row.get("regional_diagnostics")
        if isinstance(regional, dict):
            regional_path = path.parent / regional["file"]
            row["regional_diagnostics"]["file"] = str(
                regional_path.relative_to(ROOT)
            )
            if regional_path.is_file():
                row["regional_diagnostics"]["sha256"] = sha256_file(regional_path)
        if submission.get("schema_version") == "3.0":
            discretization_path = path.parent / submission["spatial_discretization"]["file"]
            row["spatial_discretization"]["file"] = str(discretization_path.relative_to(ROOT))
            if discretization_path.is_file():
                discretization = load_json(discretization_path)
                row["spatial_discretization"]["summary"] = discretization
                case_manifest = discretization.get("case_manifest", {})
                case_file = case_manifest.get("file")
                if isinstance(case_file, str):
                    row["spatial_discretization"]["summary"]["case_manifest"]["file"] = str(
                        (path.parent / case_file).relative_to(ROOT)
                    )
            case_metrics_path = path.parent / submission["case_metrics"]["file"]
            row["case_metrics"]["file"] = str(case_metrics_path.relative_to(ROOT))
            declared_artifacts = submission.get("prediction_artifacts", [])
            checks_path = path.parent / "prediction-artifact-checks.json"
            prediction_status: dict[str, Any] = {
                "sharing": "declared" if declared_artifacts else "not_declared",
                "declared_artifact_count": len(declared_artifacts),
                "maintainer_check_status": "not_recorded",
                "checked_artifact_count": 0,
            }
            if checks_path.is_file():
                checks = load_json(checks_path)
                successful_statuses = {
                    "accessible",
                    "format_checked",
                    "metrics_recomputed",
                }
                prediction_status.update(
                    {
                        "maintainer_check_status": "recorded",
                        "checked_artifact_count": sum(
                            1
                            for check in checks.get("checks", [])
                            if isinstance(check, dict)
                            and check.get("status") in successful_statuses
                        ),
                        "checks": deepcopy(checks.get("checks", [])),
                        "check_file": str(checks_path.relative_to(ROOT)),
                        "check_sha256": sha256_file(checks_path),
                    }
                )
            row["prediction_artifact_status"] = prediction_status
        if allowed_approval_status == "approved":
            validation_metadata = submission["approval"]["validation"]
            validation_path = path.parent / validation_metadata["evidence_file"]
            validation = load_json(validation_path)
            row["maintainer_validation"] = {
                "schema_version": validation["schema_version"],
                "status": validation["status"],
                "contract_version": validation["contract_version"],
                "reference_version": validation["reference_version"],
                "case_set_id": validation["case_set_id"],
                "profile_ground_truth_release_id": validation["profile_ground_truth_release_id"],
                "profile_ground_truth_manifest_sha256": validation["profile_ground_truth_manifest_sha256"],
                "validated_by": validation["validated_by"],
                "validated_at": validation["validated_at"],
                "validation_scope": validation["validation_scope"],
                "model_execution": validation["model_execution"],
                "metric_recomputation": validation["metric_recomputation"],
                "reviewed_submission_sha256": validation["reviewed_submission_sha256"],
                "evaluation_evidence_sha256": validation["evaluation_evidence_sha256"],
                "profile_index_sha256": validation["profile_index_sha256"],
                "evidence_path": str(validation_path.relative_to(ROOT)),
                "evidence_sha256": validation_metadata["evidence_sha256"],
            }
            for key in (
                "scoring_support_release_id",
                "scoring_support_manifest_sha256",
                "discretization_sha256",
                "case_metrics_sha256",
            ):
                if key in validation:
                    row["maintainer_validation"][key] = validation[key]
        rows[submission["dataset"]].append(row)
    annotate_result_revisions(rows)
    return rows


def annotate_result_revisions(rows_by_dataset: dict[str, list[dict[str, Any]]]) -> None:
    """Add derived current-version information without changing source packages."""

    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for rows in rows_by_dataset.values():
        for row in rows:
            revision = normalized_result_revision(row)
            row["result_revision"] = revision
            key = (row.get("dataset_id"), row.get("split_id"), revision.get("series_id"))
            groups.setdefault(key, []).append(row)

    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["result_revision"].get("version") or 0),
                str(row.get("submitted_at") or ""),
                str(row.get("submission_id") or ""),
            ),
        )
        latest = ordered[-1]
        for row in ordered:
            row["result_revision"].update(
                {
                    "is_latest": row is latest,
                    "latest_submission_id": latest.get("submission_id"),
                    "version_count": len(ordered),
                }
            )


def latest_rows_by_dataset(
    rows_by_dataset: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return only the current revision in each result series."""

    return {
        dataset: [
            deepcopy(row)
            for row in rows
            if row.get("result_revision", {}).get("is_latest") is True
        ]
        for dataset, rows in rows_by_dataset.items()
    }


def revision_history_payload(
    manifest: dict[str, Any],
    rows_by_dataset: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    records = sorted(
        (deepcopy(row) for rows in rows_by_dataset.values() for row in rows),
        key=lambda row: (
            str(row.get("dataset_id") or ""),
            str(row.get("split_id") or ""),
            str(row.get("result_revision", {}).get("series_id") or ""),
            int(row.get("result_revision", {}).get("version") or 0),
            str(row.get("submission_id") or ""),
        ),
    )
    series = {
        (
            row.get("dataset_id"),
            row.get("split_id"),
            row.get("result_revision", {}).get("series_id"),
        )
        for row in records
    }
    release = manifest.get("data_release", {})
    return {
        "$schema": REVISION_HISTORY_SCHEMA_URL,
        "schema_version": "1.0",
        "release_id": release.get("id"),
        "release_status": release.get("status"),
        "generated_at": release.get("generated_at"),
        "record_count": len(records),
        "series_count": len(series),
        "records": records,
    }


def published_metric_value(value: int | float, decimal_places: int) -> tuple[Decimal, float, str]:
    """Return the exact decimal used for ranking, its JSON number, and its display string."""

    quantum = Decimal(1).scaleb(-decimal_places)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded.is_zero():
        rounded = rounded.copy_abs()
    return rounded, float(rounded), f"{rounded:.{decimal_places}f}"


def claim_eligibility(release_status: str, row: dict[str, Any]) -> dict[str, Any]:
    if release_status == "prototype_dummy_data":
        if row.get("record_type") == "development_fixture":
            return {
                "academic_citation": False,
                "promotion": False,
                "reason_code": "non_ranked_development_fixture",
                "reason": (
                    "Synthetic truth-derived development fixture retained only "
                    "to exercise the AhmedML evaluator and dev dashboard; it is "
                    "not model inference or a benchmark submission."
                ),
            }
        if row.get("record_type") == "pre_release_reference":
            if row.get("dataset_id") == "hiliftaeroml":
                reason = (
                    "Genuine HiLiftAeroML pre-release reference data retained to "
                    "exercise the evaluator and leaderboard before submissions "
                    "open; it is not an official benchmark result or ranking claim."
                )
            elif row.get("dataset_id") == "ahmedml":
                reason = (
                    "Genuine AhmedML pre-release model-inference data retained "
                    "to exercise the evaluator and dev leaderboard while "
                    "submissions remain closed; it is not an official benchmark "
                    "result or ranking claim."
                )
            else:
                reason = (
                    "Genuine pre-release reference data retained to validate the "
                    "DrivAerML workflow before submissions open; it is not an "
                    "official benchmark result or ranking claim."
                )
            return {
                "academic_citation": False,
                "promotion": False,
                "reason_code": "pre_release_reference",
                "reason": reason,
            }
        return {
            "academic_citation": False,
            "promotion": False,
            "reason_code": "prototype_dummy_data",
            "reason": "Illustrative prototype data are not eligible for academic leaderboard claims or promotion.",
        }
    validation = row.get("maintainer_validation", {})
    eligible = row.get("approval", {}).get("status") == "approved" and validation.get("status") == "validated"
    if eligible:
        return {
            "academic_citation": True,
            "promotion": True,
            "reason_code": "approved_submitted_data_result",
            "reason": (
                "Approved submitted-data result in this immutable release. FluidsBench did not execute the model "
                "or recompute submitted base metrics."
            ),
        }
    return {
        "academic_citation": False,
        "promotion": False,
        "reason_code": "not_approved_and_validated",
        "reason": "The result is not both approved and covered by a maintainer submitted-data validation record.",
    }


def add_release_rankings(
    manifest: dict[str, Any], rows_by_dataset: dict[str, list[dict[str, Any]]]
) -> None:
    """Add deterministic release-scoped competition ranks to feed rows in place."""

    definitions = {definition["id"]: definition for definition in manifest.get("metric_definitions", [])}
    release_status = manifest.get("data_release", {}).get("status", "")
    for dataset in manifest.get("datasets", []):
        config = dataset.get("ranking")
        if not isinstance(config, dict):
            continue
        metric_id = config["metric_id"]
        decimal_places = config["decimal_places"]
        direction = config["direction"]
        definition = definitions[metric_id]
        groups: dict[str, list[tuple[dict[str, Any], Decimal, float, str]]] = {}
        for row in rows_by_dataset.get(dataset["name"], []):
            if row.get("record_type") == "development_fixture":
                row.pop("ranking", None)
                row["claim_eligibility"] = claim_eligibility(
                    release_status, row
                )
                continue
            raw_value = row["metric_values"][metric_id]
            rounded, ranked_value, display_value = published_metric_value(raw_value, decimal_places)
            groups.setdefault(row["split_id"], []).append((row, rounded, ranked_value, display_value))

        for group in groups.values():
            counts = Counter(rounded for _, rounded, _, _ in group)
            ordered = sorted(
                group,
                key=lambda item: (
                    -item[1] if direction == "higher" else item[1],
                    item[0]["submission_id"],
                ),
            )
            previous_value: Decimal | None = None
            rank = 0
            for position, (row, rounded, ranked_value, display_value) in enumerate(ordered, start=1):
                if rounded != previous_value:
                    rank = position
                    previous_value = rounded
                tie_count = counts[rounded]
                row["ranking"] = {
                    "metric_id": metric_id,
                    "value": row["metric_values"][metric_id],
                    "ranked_value": ranked_value,
                    "display_value": display_value,
                    "unit": definition.get("unit", ""),
                    "direction": direction,
                    "decimal_places": decimal_places,
                    "rounding": RANKING_ROUNDING,
                    "method": RANKING_METHOD,
                    "rank": rank,
                    "ranked_result_count": len(group),
                    "tied": tie_count > 1,
                    "tie_count": tie_count,
                }
                row["claim_eligibility"] = claim_eligibility(release_status, row)


def claim_record_path(row: dict[str, Any]) -> str:
    return f"leaderboard/claims/{row['dataset_id']}/{row['split_id']}/{row['submission_id']}.json"


def feed_result_revision(row: dict[str, Any]) -> dict[str, Any]:
    revision = deepcopy(row.get("result_revision") or normalized_result_revision(row))
    revision.setdefault("is_latest", True)
    revision.setdefault("latest_submission_id", row.get("submission_id"))
    revision.setdefault("version_count", 1)
    return revision


def result_permalink(release: dict[str, Any], row: dict[str, Any]) -> str | None:
    release_view_url = release.get("release_view_url")
    if release.get("status") != "official" or not isinstance(release_view_url, str):
        return None
    query = urlencode(
        (
            ("view", "result"),
            ("dataset", row["dataset_id"]),
            ("split", row["split_id"]),
            ("result", row["submission_id"]),
        )
    )
    return f"{release_view_url}?{query}"


def immutable_claim_record_url(release: dict[str, Any], row: dict[str, Any]) -> str | None:
    asset_base_url = release.get("asset_base_url")
    if release.get("status") != "official" or not isinstance(asset_base_url, str):
        return None
    parts = urlsplit(asset_base_url)
    claim_path = f"{parts.path.rstrip('/')}/{claim_record_path(row)}"
    return urlunsplit((parts.scheme, parts.netloc, claim_path, "", ""))


def available_file_binding(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def scoring_support_file_binding(row: dict[str, Any]) -> dict[str, str] | None:
    """Find the local benchmark-owned manifest already pinned by the feed row."""

    declaration = row.get("scoring_support")
    if not isinstance(declaration, dict):
        return None
    expected_sha256 = declaration.get("manifest_sha256")
    if not isinstance(expected_sha256, str):
        return None
    directory = ROOT / "benchmark-specs" / row["dataset_id"] / "scoring-support"
    for path in sorted(directory.glob("**/manifest.json")):
        if sha256_file(path) == expected_sha256:
            return {"path": str(path.relative_to(ROOT)), "sha256": expected_sha256}
    return None


def build_claim_record(manifest: dict[str, Any], row: dict[str, Any], row_index: int) -> dict[str, Any]:
    release = manifest["data_release"]
    result_revision = feed_result_revision(row)

    source_directory = ROOT / "submissions" / row["dataset_id"] / row["submission_id"]
    bindings: dict[str, Any] = {
        "result": {
            "feed_file": manifest["all_file"],
            "feed_sha256": release["feed_sha256"],
            "row_index": row_index,
        }
    }
    required_binding_paths = {
        "source_submission": source_directory / "submission.json",
        "evaluation_evidence": source_directory / row["evaluation"]["evidence_file"],
        "profile_index": ROOT / row["profile_data"]["index_file"],
    }
    for binding_name, binding_path in required_binding_paths.items():
        binding = available_file_binding(binding_path)
        if binding is None:
            raise FileNotFoundError(f"missing required claim binding: {binding_path}")
        bindings[binding_name] = binding
    validation = row.get("maintainer_validation")
    if isinstance(validation, dict):
        validation_path = ROOT / validation["evidence_path"]
        if validation_path.is_file():
            bindings["maintainer_validation"] = {
                "path": validation["evidence_path"],
                "sha256": sha256_file(validation_path),
                "status": validation["status"],
                "validation_scope": validation["validation_scope"],
                "model_execution": validation["model_execution"],
                "metric_recomputation": validation["metric_recomputation"],
            }
    if row.get("schema_version") == "3.0":
        v3_binding_paths = {
            "spatial_discretization": ROOT / row["spatial_discretization"]["file"],
            "discretization_cases": ROOT
            / row["spatial_discretization"]["summary"]["case_manifest"]["file"],
            "case_metrics": ROOT / row["case_metrics"]["file"],
        }
        regional = row.get("regional_diagnostics")
        if isinstance(regional, dict):
            v3_binding_paths["regional_diagnostics"] = ROOT / regional["file"]
        for binding_name, binding_path in v3_binding_paths.items():
            binding = available_file_binding(binding_path)
            if binding is None:
                raise FileNotFoundError(f"missing required schema v3 claim binding: {binding_path}")
            bindings[binding_name] = binding
        support_binding = scoring_support_file_binding(row)
        if support_binding is None:
            raise FileNotFoundError(
                f"missing scoring-support manifest for {row['dataset_id']}/{row['submission_id']}"
            )
        bindings["scoring_support"] = {
            "release_id": row["scoring_support"]["release_id"],
            "manifest_url": row["scoring_support"]["manifest_url"],
            "manifest_sha256": row["scoring_support"]["manifest_sha256"],
            "path": support_binding["path"],
        }
        prediction_status = row.get("prediction_artifact_status", {})
        prediction_check_file = prediction_status.get("check_file")
        if isinstance(prediction_check_file, str):
            prediction_binding = available_file_binding(ROOT / prediction_check_file)
            if prediction_binding is not None:
                bindings["prediction_artifact_checks"] = prediction_binding

    claim_id = "/".join(
        (release["id"], row["dataset_id"], row["split_id"], row["submission_id"])
    )
    record = {
        "$schema": CLAIM_SCHEMA_URL,
        "schema_version": "1.0",
        "claim_id": claim_id,
        "release": {
            "id": release["id"],
            "status": release["status"],
            "published_at": release["generated_at"],
            "archive_url": release.get("archive_url"),
            "release_view_url": release.get("release_view_url"),
        },
        "result": {
            "submission_id": row["submission_id"],
            "result_revision": result_revision,
            "model": row["model"],
            "dataset": row["dataset"],
            "dataset_id": row["dataset_id"],
            "split": row["split"],
            "split_id": row["split_id"],
            "submission_schema_version": row.get("schema_version", "1.0"),
            **(
                {"prediction_scope": row["prediction_scope"]}
                if row.get("prediction_scope") in {
                    "surface_and_volume",
                    "surface_only",
                }
                else {}
            ),
        },
        "ranking": deepcopy(row["ranking"]),
        "eligibility": deepcopy(row["claim_eligibility"]),
        "result_permalink": result_permalink(release, row),
        "claim_record_url": immutable_claim_record_url(release, row),
        "bindings": bindings,
    }
    if row.get("schema_version") == "3.0":
        record["prediction_artifacts"] = {
            key: deepcopy(value)
            for key, value in row.get("prediction_artifact_status", {}).items()
            if key
            in {
                "sharing",
                "declared_artifact_count",
                "maintainer_check_status",
                "checked_artifact_count",
            }
        }
    return record


def expected_claim_artifacts(
    manifest: dict[str, Any], all_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records = {
        claim_record_path(row): build_claim_record(manifest, row, row_index)
        for row_index, row in enumerate(all_rows)
        if isinstance(row.get("ranking"), dict)
    }
    entries = []
    for path_value, record in records.items():
        entries.append(
            {
                "claim_id": record["claim_id"],
                "submission_id": record["result"]["submission_id"],
                "series_id": record["result"]["result_revision"]["series_id"],
                "version": record["result"]["result_revision"]["version"],
                "dataset_id": record["result"]["dataset_id"],
                "split_id": record["result"]["split_id"],
                "eligible": record["eligibility"]["academic_citation"]
                and record["eligibility"]["promotion"],
                "file": path_value,
                "sha256": hashlib.sha256(json_bytes(record)).hexdigest(),
            }
        )
    entries.sort(
        key=lambda entry: (
            entry["dataset_id"],
            entry["split_id"],
            entry["series_id"],
            entry["version"],
            entry["submission_id"],
        )
    )
    eligible_count = sum(entry["eligible"] for entry in entries)
    index = {
        "$schema": CLAIM_INDEX_SCHEMA_URL,
        "schema_version": "1.0",
        "release_id": manifest["data_release"]["id"],
        "release_status": manifest["data_release"]["status"],
        "feed_file": manifest["all_file"],
        "feed_sha256": manifest["data_release"]["feed_sha256"],
        "ranking_contract": deepcopy(manifest["ranking_contract"]),
        "record_count": len(entries),
        "eligible_record_count": eligible_count,
        "records": entries,
    }
    return index, records


def claim_index_semantic_errors(index: dict[str, Any]) -> list[str]:
    records = index.get("records")
    if not isinstance(records, list):
        return []  # JSON Schema reports the structural error.
    errors = []
    if index.get("record_count") != len(records):
        errors.append("record_count must equal the number of claim-index records")
    eligible_count = sum(
        isinstance(entry, dict) and entry.get("eligible") is True for entry in records
    )
    if index.get("eligible_record_count") != eligible_count:
        errors.append("eligible_record_count must equal the number of eligible claim-index records")
    return errors


def revision_history_semantic_errors(history: dict[str, Any]) -> list[str]:
    records = history.get("records")
    if not isinstance(records, list):
        return []  # JSON Schema reports the structural error.
    errors: list[str] = []
    if history.get("record_count") != len(records):
        errors.append("record_count must equal the number of revision-history records")
    submission_ids = [record.get("submission_id") for record in records if isinstance(record, dict)]
    if len(submission_ids) != len(set(submission_ids)):
        errors.append("revision-history submission IDs must be unique")
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        revision = record.get("result_revision", {})
        key = (record.get("dataset_id"), record.get("split_id"), revision.get("series_id"))
        groups.setdefault(key, []).append(record)
    if history.get("series_count") != len(groups):
        errors.append("series_count must equal the number of dataset/split/result series")
    for key, group in groups.items():
        ordered = sorted(
            group,
            key=lambda record: record.get("result_revision", {}).get("version", 0),
        )
        versions = [record.get("result_revision", {}).get("version") for record in ordered]
        if not all(isinstance(version, int) and not isinstance(version, bool) for version in versions):
            continue
        if sorted(versions) != list(range(1, max(versions) + 1)):
            errors.append(f"revision series {key} must contain every sequential version from 1")
            continue
        latest = [record for record in ordered if record.get("result_revision", {}).get("is_latest") is True]
        if len(latest) != 1 or latest[0].get("result_revision", {}).get("version") != max(versions):
            errors.append(f"revision series {key} must mark only its highest version as latest")
            continue
        latest_submission_id = ordered[-1].get("submission_id")
        for index, record in enumerate(ordered):
            revision = record.get("result_revision", {})
            expected_supersedes = None if index == 0 else ordered[index - 1].get("submission_id")
            if revision.get("supersedes") != expected_supersedes:
                errors.append(f"revision series {key} must link each version to its exact predecessor")
                break
            if (
                revision.get("version_count") != len(ordered)
                or revision.get("latest_submission_id") != latest_submission_id
            ):
                errors.append(f"revision series {key} has inconsistent derived version metadata")
                break
    return errors


def official_release_seal_errors(
    expected_manifest: dict[str, Any],
    expected_index: dict[str, Any],
    expected_records: dict[str, dict[str, Any]],
) -> list[str]:
    """Refuse to overwrite a generated official release under the same release ID."""

    release = expected_manifest.get("data_release", {})
    if release.get("status") != "official" or not CLAIMS_INDEX_PATH.is_file():
        return []
    try:
        existing_index = load_json(CLAIMS_INDEX_PATH)
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot verify the existing official release seal: {error}"]
    if not isinstance(existing_index, dict):
        return ["cannot verify the existing official release seal: claim index is not an object"]
    if existing_index.get("release_id") != release.get("id"):
        return []  # A different ID is a deliberate new release.

    errors = [f"existing claim index {error}" for error in claim_index_semantic_errors(existing_index)]
    if existing_index.get("release_status") != "official":
        errors.append("the existing release ID is not an official sealed release; choose a new official release ID")
    if existing_index.get("feed_sha256") != expected_index.get("feed_sha256"):
        errors.append("the computed feed digest changed for an existing official release ID; choose a new release ID")
    if existing_index.get("ranking_contract") != expected_index.get("ranking_contract"):
        errors.append("the ranking contract changed for an existing official release ID; choose a new release ID")
    expected_history = expected_manifest.get("data_release", {}).get("revision_history")
    if isinstance(expected_history, dict):
        try:
            existing_manifest = load_json(MANIFEST_PATH)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot verify the existing official revision history: {error}")
        else:
            existing_history = existing_manifest.get("data_release", {}).get("revision_history")
            if existing_history != expected_history:
                errors.append("the revision history changed for an existing official release ID; choose a new release ID")
            elif isinstance(existing_history, dict):
                expected_digest = existing_history.get("sha256")
                if not REVISION_HISTORY_PATH.is_file() or sha256_file(REVISION_HISTORY_PATH) != expected_digest:
                    errors.append("the sealed revision-history bytes no longer match the existing official release")

    existing_entries = {
        entry.get("file"): entry
        for entry in existing_index.get("records", [])
        if isinstance(entry, dict) and isinstance(entry.get("file"), str)
    }
    if set(existing_entries) != set(expected_records):
        errors.append("the claim-record set changed for an existing official release ID; choose a new release ID")

    for path_value in sorted(set(existing_entries) & set(expected_records)):
        path = ROOT / path_value
        if not path.is_file():
            errors.append(f"sealed claim record {path_value} is missing")
            continue
        expected_sha256 = existing_entries[path_value].get("sha256")
        if sha256_file(path) != expected_sha256:
            errors.append(f"sealed claim record {path_value} no longer matches its existing index digest")
            continue
        try:
            existing_record = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot read sealed claim record {path_value}: {error}")
            continue
        if not isinstance(existing_record, dict):
            errors.append(f"sealed claim record {path_value} is not an object")
            continue
        expected_record = expected_records[path_value]
        if existing_record.get("release") != expected_record.get("release"):
            errors.append(
                f"published release metadata changed in {path_value} for an existing official release ID; "
                "choose a new release ID"
            )
        elif existing_record != expected_record:
            errors.append(f"sealed claim record {path_value} changed; choose a new release ID")

    if existing_index != expected_index:
        errors.append("the generated claim index changed for an existing official release ID; choose a new release ID")
    return list(dict.fromkeys(errors))


def latest_submission_date(rows: list[dict[str, Any]]) -> str | None:
    valid_dates = []
    for row in rows:
        try:
            valid_dates.append(date.fromisoformat(row.get("submitted_at", "")).isoformat())
        except (TypeError, ValueError):
            continue
    return max(valid_dates, default=None)


def stable_fallback_date(manifest: dict[str, Any]) -> str:
    for value in (
        manifest.get("data_release", {}).get("generated_at"),
        manifest.get("generated_at"),
    ):
        if not isinstance(value, str):
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            continue
    return "1970-01-01"


def expected_outputs(
    manifest: dict[str, Any],
    *,
    generated_at: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    updated_manifest = manifest_with_benchmark_contract(manifest)
    revision_rows_by_dataset = source_rows_by_dataset(updated_manifest)
    rows_by_dataset = latest_rows_by_dataset(revision_rows_by_dataset)
    add_release_rankings(updated_manifest, rows_by_dataset)
    all_rows: list[dict[str, Any]] = []
    latest_dates: list[str] = []
    for dataset in updated_manifest["datasets"]:
        rows = rows_by_dataset[dataset["name"]]
        revision_rows = revision_rows_by_dataset[dataset["name"]]
        latest = latest_submission_date(revision_rows)
        dataset["submission_count"] = len(rows)
        dataset["revision_count"] = len(revision_rows)
        dataset["updated_at"] = latest
        if latest is not None:
            latest_dates.append(latest)
        all_rows.extend(rows)
    latest_global = max(latest_dates, default=stable_fallback_date(manifest))
    release = updated_manifest.setdefault("data_release", {})
    if release.get("status") == "official":
        official_generated_at = release.get("generated_at")
        if not isinstance(official_generated_at, str):
            raise ValueError("official releases require an explicit data_release.generated_at")
        updated_manifest["generated_at"] = official_generated_at
    else:
        updated_manifest["generated_at"] = (
            generated_at or manifest.get("generated_at") or f"{latest_global}T00:00:00Z"
        )
    if release.get("status") == "prototype_dummy_data":
        updated_manifest["submission_schema_version"] = "1.0"
    elif release.get("reproducibility_contract_version") == "open-reproducibility-3.0":
        updated_manifest["submission_schema_version"] = "3.0"
    else:
        updated_manifest["submission_schema_version"] = "2.0"
    release["generated_at"] = updated_manifest["generated_at"]
    feed_sha256 = hashlib.sha256(json_bytes(all_rows)).hexdigest()
    if release.get("status") == "prototype_dummy_data":
        release["id"] = f"prototype-dev-{latest_global}-{feed_sha256[:12]}"
    release["feed_sha256"] = feed_sha256
    revision_history = revision_history_payload(updated_manifest, revision_rows_by_dataset)
    release["revision_history"] = {
        "schema_version": revision_history["schema_version"],
        "file": str(REVISION_HISTORY_PATH.relative_to(ROOT)),
        "sha256": hashlib.sha256(json_bytes(revision_history)).hexdigest(),
        "record_count": revision_history["record_count"],
        "series_count": revision_history["series_count"],
    }
    claim_index, _ = expected_claim_artifacts(updated_manifest, all_rows)
    release["claims"] = {
        "schema_version": claim_index["schema_version"],
        "index_file": "leaderboard/claims/index.json",
        "index_sha256": hashlib.sha256(json_bytes(claim_index)).hexdigest(),
        "record_count": claim_index["record_count"],
        "eligible_record_count": claim_index["eligible_record_count"],
    }
    return updated_manifest, rows_by_dataset, all_rows, revision_history


def build(manifest: dict[str, Any]) -> list[str]:
    generated_at = None
    if manifest.get("data_release", {}).get("status") != "official":
        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    updated_manifest, rows_by_dataset, all_rows, revision_history = expected_outputs(
        manifest, generated_at=generated_at
    )
    claim_index, claim_records = expected_claim_artifacts(updated_manifest, all_rows)
    seal_errors = official_release_seal_errors(updated_manifest, claim_index, claim_records)
    if seal_errors:
        return seal_errors
    for dataset in updated_manifest["datasets"]:
        write_json(ROOT / dataset["file"], rows_by_dataset[dataset["name"]])
    write_json(ROOT / updated_manifest["all_file"], all_rows)
    write_json(ROOT / "leaderboard.json", all_rows)
    write_json(REVISION_HISTORY_PATH, revision_history)
    expected_claim_paths = {ROOT / path_value for path_value in claim_records}
    for stale_path in CLAIMS_ROOT.glob("*/*/*.json"):
        if stale_path not in expected_claim_paths:
            stale_path.unlink()
    for path_value, record in claim_records.items():
        write_json(ROOT / path_value, record)
    write_json(CLAIMS_INDEX_PATH, claim_index)
    write_json(MANIFEST_PATH, updated_manifest)
    return []


def check_generated_feeds(manifest: dict[str, Any]) -> list[str]:
    expected_manifest, rows_by_dataset, all_rows, revision_history = expected_outputs(manifest)
    expected_claim_index, expected_claim_records = expected_claim_artifacts(expected_manifest, all_rows)
    errors = []
    for dataset in expected_manifest["datasets"]:
        path = ROOT / dataset["file"]
        if not path.exists() or load_json(path) != rows_by_dataset[dataset["name"]]:
            errors.append(f"{path.relative_to(ROOT)} is not synchronized with source submissions")
    for path_value in (expected_manifest["all_file"], "leaderboard.json"):
        path = ROOT / path_value
        if not path.exists() or load_json(path) != all_rows:
            errors.append(f"{path.relative_to(ROOT)} is not synchronized with source submissions")
        elif sha256_file(path) != expected_manifest["data_release"]["feed_sha256"]:
            errors.append(f"{path.relative_to(ROOT)} bytes do not match data_release.feed_sha256")
    revision_metadata = expected_manifest["data_release"]["revision_history"]
    if not REVISION_HISTORY_PATH.is_file() or load_json(REVISION_HISTORY_PATH) != revision_history:
        errors.append("leaderboard/revisions.json is not synchronized with source submission revisions")
    elif sha256_file(REVISION_HISTORY_PATH) != revision_metadata["sha256"]:
        errors.append("leaderboard/revisions.json does not match data_release.revision_history.sha256")
    if REVISION_HISTORY_PATH.is_file():
        actual_revision_history = load_json(REVISION_HISTORY_PATH)
        for error in schema_errors(
            actual_revision_history,
            "revision-history.schema.json",
            schema_version="releases",
        ):
            errors.append(f"leaderboard/revisions.json {error}")
        errors.extend(
            f"leaderboard/revisions.json {error}"
            for error in revision_history_semantic_errors(actual_revision_history)
        )
    actual_claim_index = load_json(CLAIMS_INDEX_PATH) if CLAIMS_INDEX_PATH.is_file() else None
    if actual_claim_index is None or actual_claim_index != expected_claim_index:
        errors.append("leaderboard/claims/index.json is not synchronized with the ranked scalar feed")
    elif sha256_file(CLAIMS_INDEX_PATH) != expected_manifest["data_release"]["claims"]["index_sha256"]:
        errors.append("leaderboard/claims/index.json does not match data_release.claims.index_sha256")
    if isinstance(actual_claim_index, dict):
        errors.extend(
            f"leaderboard/claims/index.json {error}"
            for error in claim_index_semantic_errors(actual_claim_index)
        )
    actual_claim_paths = set(CLAIMS_ROOT.glob("*/*/*.json"))
    expected_claim_paths = {ROOT / path_value for path_value in expected_claim_records}
    for missing_path in sorted(expected_claim_paths - actual_claim_paths):
        errors.append(f"{missing_path.relative_to(ROOT)} is missing")
    for stale_path in sorted(actual_claim_paths - expected_claim_paths):
        errors.append(f"{stale_path.relative_to(ROOT)} is not part of the current release")
    for path_value, expected_record in expected_claim_records.items():
        path = ROOT / path_value
        if path.is_file() and load_json(path) != expected_record:
            errors.append(f"{path_value} is not synchronized with its ranked feed row")
            continue
        if path.is_file():
            expected_sha256 = next(
                entry["sha256"] for entry in expected_claim_index["records"] if entry["file"] == path_value
            )
            if sha256_file(path) != expected_sha256:
                errors.append(f"{path_value} bytes do not match the claims-index digest")
            for error in schema_errors(load_json(path), "result-claim.schema.json", schema_version="releases"):
                errors.append(f"{path_value} {error}")
    if isinstance(actual_claim_index, dict):
        for error in schema_errors(
            actual_claim_index, "claim-index.schema.json", schema_version="releases"
        ):
            errors.append(f"leaderboard/claims/index.json {error}")
    for key in (
        "schema_version",
        "submission_schema_version",
        "generated_at",
        "data_release",
        "ranking_contract",
        "metric_catalog",
        "metric_definitions",
        "training_regimes",
        "datasets",
    ):
        if manifest.get(key) != expected_manifest.get(key):
            errors.append(f"leaderboard/manifest.json has stale {key}")
    return errors


def approval_documents(
    submission: dict[str, Any],
    directory: Path,
    *,
    validated_by: str,
    validated_at: str,
    approved_by: str,
    approved_at: str,
    pull_request_url: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the maintainer-owned v3 validation record and approval metadata."""

    reviewed_submission = deepcopy(submission)
    reviewed_submission.pop("approval", None)
    profile_index_path = directory / reviewed_submission["profile_data"]["index_file"]
    validation = {
        "$schema": "https://fluidsbench.org/schemas/v3/maintainer-validation.schema.json",
        "schema_version": "3.0",
        "contract_version": reviewed_submission["reproducibility"]["contract_version"],
        "submission_id": reviewed_submission["submission_id"],
        "dataset_id": reviewed_submission["dataset_id"],
        "split_id": reviewed_submission["split_id"],
        "case_set_id": reviewed_submission["case_set_id"],
        "reference_version": reviewed_submission["evaluation"]["reference_version"],
        "profile_ground_truth_release_id": reviewed_submission["profile_data"][
            "profile_ground_truth_release_id"
        ],
        "profile_ground_truth_manifest_sha256": reviewed_submission["profile_data"][
            "profile_ground_truth_manifest_sha256"
        ],
        "scoring_support_release_id": reviewed_submission["scoring_support"]["release_id"],
        "scoring_support_manifest_sha256": reviewed_submission["scoring_support"]["manifest_sha256"],
        "validated_by": validated_by,
        "validated_at": validated_at,
        "status": "validated",
        "validation_scope": "submitted_data_only",
        "model_execution": "not_performed",
        "metric_recomputation": "not_performed",
        "reviewed_submission_sha256": canonical_submission_sha256(reviewed_submission),
        "evaluation_evidence_sha256": reviewed_submission["evaluation"]["evidence_sha256"],
        "profile_index_sha256": sha256_file(profile_index_path),
        "discretization_sha256": reviewed_submission["spatial_discretization"]["sha256"],
        "case_metrics_sha256": reviewed_submission["case_metrics"]["sha256"],
    }
    regional = reviewed_submission.get("regional_diagnostics")
    if isinstance(regional, dict):
        validation["regional_diagnostics_sha256"] = regional["sha256"]
    validation_path = directory / "maintainer-validation.json"
    approved_submission = deepcopy(reviewed_submission)
    approved_submission["approval"] = {
        "status": "approved",
        "approved_by": approved_by,
        "approved_at": approved_at,
        "pull_request_url": pull_request_url,
        "validation": {
            "evidence_file": "maintainer-validation.json",
            "evidence_sha256": hashlib.sha256(json_bytes(validation)).hexdigest(),
        },
    }
    return validation, approved_submission


def canonical_submission_sha256(submission: dict[str, Any]) -> str:
    """Match validate_submission.canonical_json_sha256 without a circular import."""

    payload = json.dumps(
        submission,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approve_submission(
    path: Path,
    *,
    validated_by: str,
    validated_at: str,
    approved_by: str,
    approved_at: str,
    pull_request_url: str,
    dry_run: bool,
    update_existing: bool,
    rebuild_feeds: bool,
) -> list[str]:
    """Create a schema-v3 maintainer approval, validate it, and rebuild compact feeds."""

    path = path.resolve()
    submissions_root = (ROOT / "submissions").resolve()
    if path.is_dir():
        path = path / "submission.json"
    if not path.is_relative_to(submissions_root) or len(path.relative_to(submissions_root).parts) != 3:
        return ["approve requires exactly submissions/<dataset-id>/<submission-id>/submission.json"]
    if not path.is_file():
        return [f"submission file does not exist: {path}"]
    try:
        submission = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read submission: {error}"]
    if submission.get("schema_version") != "3.0":
        return ["new approvals require submission schema_version=3.0"]
    if submission.get("dataset_id") == "hiliftaeroml":
        specification = load_json(
            ROOT / "benchmark-specs" / "hiliftaeroml" / "submission-spec.json"
        )
        scoring_support = specification.get("scoring_support", {})
        profile_definition = specification.get("profile_definition", {})
        compact_definition = specification.get("compact_profile_definition", {})
        compact_profile = (
            submission.get("profile_data", {}).get("format")
            == "fluidsbench-hiliftaeroml-compact-profile-chunks-v2-candidate"
        )
        if (
            specification.get("status") != "official"
            or scoring_support.get("status") != "official"
            or scoring_support.get("submissions_open") is not True
            or profile_definition.get("profile_ground_truth", {}).get("status")
            != "published"
            or (
                compact_profile
                and compact_definition.get("evaluator_support", {}).get("status")
                != "published"
            )
        ):
            return [
                "HiLiftAeroML approval is closed until scoring support, profile "
                "truth, and any compact evaluator support are officially published"
            ]
    existing_approval = submission.get("approval")
    if existing_approval is not None and not update_existing:
        return ["submission already has approval metadata; pass --update-existing to update its PR URL"]
    validation_path = path.parent / "maintainer-validation.json"
    if validation_path.exists() and not update_existing:
        return ["maintainer-validation.json already exists; pass --update-existing to replace the same approval"]

    if not validated_by.strip() or not approved_by.strip():
        return ["validated-by and approved-by must be non-empty"]
    if not is_valid_published_at(validated_at):
        return ["validated-at must be an explicit timezone-qualified ISO date-time"]
    try:
        approved_date = date.fromisoformat(approved_at)
        validation_date = datetime.fromisoformat(validated_at.replace("Z", "+00:00")).date()
        submitted_date = date.fromisoformat(submission["submitted_at"])
    except (KeyError, TypeError, ValueError):
        return ["submitted-at, validated-at, and approved-at must be valid ISO dates"]
    if not submitted_date <= validation_date <= approved_date:
        return ["submission, validation, and approval dates must be chronological"]
    if re.fullmatch(r"https://github\.com/[^/]+/[^/]+/pull/[0-9]+", pull_request_url) is None:
        return ["pull-request-url must be a full GitHub pull request URL"]

    reviewed = deepcopy(submission)
    reviewed.pop("approval", None)
    pre_errors, _ = validate_many([path])
    if pre_errors:
        return [f"cannot approve an invalid source package: {error}" for error in pre_errors]

    validation, approved_submission = approval_documents(
        reviewed,
        path.parent,
        validated_by=validated_by,
        validated_at=validated_at,
        approved_by=approved_by,
        approved_at=approved_at,
        pull_request_url=pull_request_url,
    )
    if dry_run:
        print(json.dumps({"maintainer_validation": validation, "submission": approved_submission}, indent=2))
        return []

    previous_submission_bytes = path.read_bytes()
    previous_validation_bytes = validation_path.read_bytes() if validation_path.is_file() else None
    operation_errors: list[str] = []
    try:
        write_json(validation_path, validation)
        write_json(path, approved_submission)
        errors, _ = validate_many([path])
        if errors:
            operation_errors.extend(f"generated approval is invalid: {error}" for error in errors)
        if not operation_errors and rebuild_feeds:
            manifest = load_json(MANIFEST_PATH)
            operation_errors.extend(release_contract_errors(manifest))
        if not operation_errors and rebuild_feeds:
            operation_errors.extend(build(manifest))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        operation_errors.append(f"approval generation failed: {error}")
    if operation_errors:
        path.write_bytes(previous_submission_bytes)
        if previous_validation_bytes is None:
            if validation_path.is_file():
                validation_path.unlink()
        else:
            validation_path.write_bytes(previous_validation_bytes)
    return operation_errors


def print_errors(errors: list[str]) -> int:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate submission directories")
    validate_parser.add_argument("paths", nargs="*", type=Path)
    subparsers.add_parser("build", help="validate and regenerate all leaderboard feeds")
    subparsers.add_parser("check", help="validate submissions and verify generated feeds")
    approve_parser = subparsers.add_parser(
        "approve",
        help="create a maintainer validation/approval record and rebuild feeds",
    )
    approve_parser.add_argument("path", type=Path, help="exact schema-v3 submission directory")
    approve_parser.add_argument("--validated-by", required=True)
    approve_parser.add_argument("--validated-at", required=True, help="timezone-qualified ISO date-time")
    approve_parser.add_argument("--approved-by", required=True)
    approve_parser.add_argument("--approved-at", required=True, help="ISO date in YYYY-MM-DD form")
    approve_parser.add_argument("--pull-request-url", required=True)
    approve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the proposed records without modifying files or rebuilding feeds",
    )
    approve_parser.add_argument(
        "--update-existing",
        action="store_true",
        help="update the same generated approval, normally to replace the workflow's temporary PR URL",
    )
    approve_parser.add_argument(
        "--skip-feed-build",
        action="store_true",
        help="write and validate approval files but defer feed regeneration",
    )
    args = parser.parse_args()

    if args.command == "approve":
        errors = approve_submission(
            args.path,
            validated_by=args.validated_by,
            validated_at=args.validated_at,
            approved_by=args.approved_by,
            approved_at=args.approved_at,
            pull_request_url=args.pull_request_url,
            dry_run=args.dry_run,
            update_existing=args.update_existing,
            rebuild_feeds=not args.skip_feed_build,
        )
        if errors:
            return print_errors(errors)
        action = "Prepared" if args.dry_run else "Approved and rebuilt feeds for"
        print(f"{action} {args.path}.")
        return 0

    paths = getattr(args, "paths", None) or None
    manifest = manifest_with_benchmark_contract(load_json(MANIFEST_PATH))
    errors, totals = validate_many(paths, manifest=manifest)
    if errors:
        return print_errors(errors)
    errors = release_contract_errors(manifest)
    if errors:
        return print_errors(errors)
    if args.command == "build":
        errors = build(manifest)
        if errors:
            return print_errors(errors)
        print(f"Built compact leaderboard feeds from {totals['submissions']} validated submissions.")
    elif args.command == "check":
        errors = check_generated_feeds(manifest)
        if errors:
            return print_errors(errors)
        print(
            f"Validated {totals['submissions']} submissions, {totals['cases']} cases, and "
            f"{totals['series']} profile series; generated feeds are synchronized."
        )
    else:
        print(
            f"Validated {totals['submissions']} submissions, {totals['cases']} cases, and "
            f"{totals['series']} profile series."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
