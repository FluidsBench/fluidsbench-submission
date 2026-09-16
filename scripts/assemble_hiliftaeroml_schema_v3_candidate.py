#!/usr/bin/env python3
"""Assemble a HiLiftAeroML evaluator-native result as FluidsBench schema v3.

This is a closed-candidate adapter, not an evaluator.  It consumes the exact
per-case support/profile products and split aggregate written by the native
HiLiftAeroML evaluator, preserves all case and support identities, and creates
the participant-owned FluidsBench directory.  It never fills missing cases or
metrics.  In particular, incomplete surface-load truth coverage makes force,
overall, and package assembly unavailable rather than assigning a zero or
imputed value.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.native_profiles import (
    NativeProfileError,
    PROFILE_CONTRACT_ID,
    PROFILE_CONTRACT_SHA256,
    PROFILE_FORMAT,
    build_profile_directory,
)
from reference.hiliftaeroml.native_profile_evaluator import (
    CANDIDATE_TRUTH_STATUS,
    CANDIDATE_USAGE,
    NativeProfileEvaluationError,
    open_candidate_truth_release,
    score_native_profile_directory,
)
from reference.hiliftaeroml.compact_profile_evaluator import (
    COMPACT_PROFILE_CONTRACT_ID,
    COMPACT_PROFILE_CONTRACT_PATH,
    COMPACT_PROFILE_CONTRACT_SHA256,
    COMPACT_PROFILE_FORMAT,
    CompactSupportRelease,
    CompactProfileEvaluationError,
    build_compact_profile_directory,
    open_compact_support_release,
    score_compact_profile_directory,
)
from reference.hiliftaeroml.regional_aggregate import (
    AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DEFINITION_ID,
    REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
    HiLiftRegionalAggregateError,
    validate_aggregate_regional_diagnostics,
)
from reference.methodology import (
    MethodologyError,
    derived_parameter_count_millions,
    require_methodology,
)
from reference.scores import composite_component_group_scores, composite_overall_score
from reference.scoring_support import ScoringSupportError, load_support_release
from scripts.validate_scoring_supports import validate_candidate_manifest_release


DEFAULT_SPECIFICATION = ROOT / "benchmark-specs" / "hiliftaeroml" / "submission-spec.json"
SCHEMA_ROOT = ROOT / "schemas"
CONFIG_SCHEMA = "hiliftaeroml-fluidsbench-schema-v3-package-config-v1"
SUBMISSION_FORMAT = "hiliftaeroml_native_candidate_v3"
TOKEN_PREFIXES = ("__REPLACE_", "__UNRESOLVED_HILIFTAEROML_")
COMPACT_PROFILE_IMPLEMENTATION_BINDING = {
    "status": "unbound_worktree_candidate",
    "activation_effect": "none",
    "code_revision": None,
    "implementation_manifest_sha256": None,
    "base_dataset_evaluator_scope": (
        "native_v1_base_field_force_and_noncompact_scoring_only"
    ),
}
# Release-ready native artifacts are bound to an exact ordered case set, not a
# particular split label or a 360-case campaign.  The legacy identifiers remain
# readable only so retained Full-360 evidence can be diagnosed during migration;
# newly assembled/scored evidence uses the generic case-set family below.
NATIVE_RESULT_SCHEMA = "hiliftaeroml-transolver-case-set-results-v1"
NATIVE_CASE_METRICS_SCHEMA = "hiliftaeroml-transolver-case-set-case-metrics-v1"
NATIVE_AUXILIARY_SCHEMA = "hiliftaeroml-transolver-case-set-auxiliary-summary-v1"
NATIVE_ARTIFACT_SCHEMA = "hiliftaeroml-transolver-case-set-aggregate-artifacts-v1"
NATIVE_RECEIPT_SCHEMA = "hiliftaeroml-transolver-case-receipt-v1"
LEGACY_NATIVE_RESULT_SCHEMA = "hiliftaeroml-transolver-full360-results-v2"
LEGACY_NATIVE_CASE_METRICS_SCHEMA = (
    "hiliftaeroml-transolver-full360-case-metrics-v1"
)
LEGACY_NATIVE_AUXILIARY_SCHEMA = (
    "hiliftaeroml-transolver-full360-auxiliary-summary-v1"
)
LEGACY_NATIVE_ARTIFACT_SCHEMA = (
    "hiliftaeroml-transolver-full360-aggregate-artifacts-v1"
)
LEGACY_NATIVE_RECEIPT_SCHEMA = "hiliftaeroml-transolver-full360-case-receipt-v2"
LEGACY_FULL360_CASE_SET_ID = "caseset-ac791749e527"
NATIVE_REGIONAL_SCHEMA = "hiliftaeroml-regional-diagnostics-aggregate-v1"
NATIVE_REGIONAL_CASE_SCHEMA = "hiliftaeroml-regional-diagnostics-case-support-v1"
NATIVE_REGIONAL_CONTRACT_SHA256 = (
    "1579b0262f3368fe3748eb53025aa5e46c0a32c8ff1616c9becdbb5dedd85650"
)
NATIVE_SUPPORT_SCHEMA = "hiliftaeroml-native-support-score-v1"
NATIVE_AGGREGATE_CONTRACTS = {
    NATIVE_ARTIFACT_SCHEMA: {
        "family": "case_set_v1",
        "result_schema": NATIVE_RESULT_SCHEMA,
        "case_metrics_schema": NATIVE_CASE_METRICS_SCHEMA,
        "auxiliary_schema": NATIVE_AUXILIARY_SCHEMA,
        "auxiliary_statuses": frozenset({"complete"}),
        "require_complete_case_set_binding": True,
    },
    LEGACY_NATIVE_ARTIFACT_SCHEMA: {
        "family": "legacy_full360_v2",
        "result_schema": LEGACY_NATIVE_RESULT_SCHEMA,
        "case_metrics_schema": LEGACY_NATIVE_CASE_METRICS_SCHEMA,
        "auxiliary_schema": LEGACY_NATIVE_AUXILIARY_SCHEMA,
        "auxiliary_statuses": frozenset(
            {"complete_with_partial_report_only_load_coverage", "complete"}
        ),
        "require_complete_case_set_binding": False,
    },
}
NATIVE_AGGREGATE_FILES = frozenset(
    {
        "artifact_manifest.json",
        "auxiliary_summary.json",
        "case_metrics.json",
        "regional_diagnostics_aggregate.json",
        "submission_results.json",
    }
)
NATIVE_CASE_INPUT_SHA256_KEYS = frozenset(
    {
        "raw_contract",
        "receipt",
        "surface_loads",
        "surface_regional",
        "surface_support",
        "truth_contract",
        "volume_regional",
        "volume_support",
    }
)
SAFE_SHA256 = frozenset("0123456789abcdef")

SURFACE_SUMMARY_SCHEMA_VERSION = 7
LEGACY_SURFACE_SUMMARY_SCHEMA_VERSION = 6
VOLUME_SUMMARY_SCHEMA_VERSION = 2
EXACT_SURFACE_LOAD_RESULT_SCHEMA = "hiliftaeroml-exact-surface-load-result-v1"
EXACT_SURFACE_LOAD_ALGORITHM_ID = "hiliftaeroml-streaming-exact-surface-loads-v1"
EXACT_SURFACE_FORCE_BASIS_ID = (
    "ordered-fan-piecewise-linear-nodal-force-degree1-v1"
)
EXACT_SURFACE_PITCH_BASIS_ID = (
    "ordered-fan-piecewise-linear-nodal-body-y-moment-degree2-weights-v1"
)
EXACT_SURFACE_INPUT_BASIS_ID = "hiliftaeroml-training-qinf-cp-cf-v1"
EXACT_SURFACE_OUTPUT_BASIS_ID = (
    "hiliftaeroml-case-qref-force-moment-coefficients-v1"
)
EXACT_SURFACE_PRESSURE_CONVENTION_ID = (
    "positive-cp-times-ordered-oriented-area-vector-v1"
)

SURFACE_SUPPORT_ID = "surface-native-points-v1"
VOLUME_SUPPORT_ID = "volume-native-valid-points-v1"
SCALAR_SUPPORT_ID = "aerodynamic-case-coefficients-v1"
INFERENCE_INPUT_CASE_RECORD_IDS = {
    "surface_input": "surface-native-input",
    "volume_input": "volume-native-input",
}
INFERENCE_INPUT_SUPPORT_IDS = {
    "surface_input": SURFACE_SUPPORT_ID,
    "volume_input": VOLUME_SUPPORT_ID,
}
INFERENCE_OUTPUT_SUPPORT_IDS = {
    "surface": SURFACE_SUPPORT_ID,
    "volume": VOLUME_SUPPORT_ID,
}
NATIVE_SURFACE_SUPPORT_ID = "surface_native_points"
NATIVE_VOLUME_SUPPORT_ID = "volume_native_valid_points"
SURFACE_REGION_ORDER = (
    "nacelle_installation_envelope_proxy",
    "inboard_high_lift_envelope_proxy",
    "outboard_high_lift_envelope_proxy",
    "fuselage_tail_and_remaining",
)
VOLUME_REGION_ORDER = (
    "near_airframe_sdf_band",
    "aft_airframe_wake_envelope_proxy",
    "near_aircraft_flow_envelope",
    "farfield_and_remaining",
)

SURFACE_METRIC_SOURCE = {
    "surface_pressure_rel_l2": ("dual_area", "pressure", "relative_l2_percent"),
    "surface_pressure_equal_entity_rel_l2": (
        "equal_node",
        "pressure",
        "relative_l2_percent",
    ),
    "surface_wall_shear_rel_l2": ("dual_area", "tau_wall", "relative_l2_percent"),
    "surface_wall_shear_equal_entity_rel_l2": (
        "equal_node",
        "tau_wall",
        "relative_l2_percent",
    ),
    "surface_pressure_rel_l1": ("dual_area", "pressure", "relative_l1_percent"),
    "surface_wall_shear_rel_l1": ("dual_area", "tau_wall", "relative_l1_percent"),
}
VOLUME_METRIC_SOURCE = {
    "volume_velocity_rel_l2": (
        "equal_valid_node",
        "velocity",
        "relative_l2_percent",
    ),
    "volume_pressure_rel_l2": (
        "equal_valid_node",
        "pressure",
        "relative_l2_percent",
    ),
    "volume_velocity_rel_l1": (
        "equal_valid_node",
        "velocity",
        "relative_l1_percent",
    ),
    "volume_pressure_rel_l1": (
        "equal_valid_node",
        "pressure",
        "relative_l1_percent",
    ),
}
DIMENSIONAL_METRIC_SOURCE = {
    "surface_pressure_mae": ("surface", "pressure", "mae"),
    "surface_pressure_rmse": ("surface", "pressure", "rmse"),
    "surface_wall_shear_mae": ("surface", "tau_wall", "mae"),
    "surface_wall_shear_rmse": ("surface", "tau_wall", "rmse"),
    "volume_pressure_mae": ("volume", "pressure", "mae"),
    "volume_pressure_rmse": ("volume", "pressure", "rmse"),
    "volume_velocity_mae": ("volume", "velocity", "mae"),
    "volume_velocity_rmse": ("volume", "velocity", "rmse"),
}
RELATIVE_L2_IDS = {
    "surface_pressure_rel_l2",
    "surface_pressure_equal_entity_rel_l2",
    "surface_wall_shear_rel_l2",
    "surface_wall_shear_equal_entity_rel_l2",
    "volume_velocity_rel_l2",
    "volume_pressure_rel_l2",
}
VECTOR_METRIC_IDS = {
    "surface_wall_shear_rel_l2",
    "surface_wall_shear_equal_entity_rel_l2",
    "volume_velocity_rel_l2",
}
FORCE_METRIC_IDS = {
    "cd_r2",
    "cl_r2",
    "c_drag_mae",
    "c_lift_mae",
    "c_pitch_mae",
    "force_score",
    "overall_score",
}

HILIFT_FORCE_COEFFICIENT_FIELDS = (
    "predicted_c_drag",
    "truth_c_drag",
    "predicted_c_lift",
    "truth_c_lift",
    "predicted_c_pitch",
    "truth_c_pitch",
)


class HiLiftPackageAssemblyError(ValueError):
    """Raised when native evidence cannot produce an honest schema-v3 package."""


def _resolve_native_output_roots(
    *,
    outputs_root: Path | None,
    surface_outputs_root: Path | None = None,
    volume_outputs_root: Path | None = None,
) -> dict[str, Path]:
    """Resolve domain roots while retaining ``outputs_root`` as shorthand."""

    surface = (
        surface_outputs_root if surface_outputs_root is not None else outputs_root
    )
    volume = volume_outputs_root if volume_outputs_root is not None else outputs_root
    missing = [
        domain
        for domain, value in (("surface", surface), ("volume", volume))
        if value is None
    ]
    if missing:
        raise HiLiftPackageAssemblyError(
            "native output roots are incomplete; missing " + ", ".join(missing)
        )
    assert surface is not None and volume is not None
    return {"surface": surface, "volume": volume}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HiLiftPackageAssemblyError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise HiLiftPackageAssemblyError(f"JSON contains forbidden non-finite token {token}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HiLiftPackageAssemblyError(f"cannot encode canonical JSON: {error}") from error


def write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise HiLiftPackageAssemblyError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def load_json(
    path: Path,
    *,
    label: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HiLiftPackageAssemblyError(
            f"{label or 'JSON'} must be a regular non-symlink file: {path}"
        )
    before = path.stat()
    digest = sha256_file(path)
    if expected_sha256 is not None:
        _require_sha(expected_sha256, f"{label or path} expected digest")
        if digest != expected_sha256:
            raise HiLiftPackageAssemblyError(
                f"{label or path} SHA-256 differs from the aggregate case binding"
            )
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except HiLiftPackageAssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HiLiftPackageAssemblyError(f"cannot read {label or path}: {error}") from error
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(path) != digest:
        raise HiLiftPackageAssemblyError(f"{label or path} changed while it was read")
    if not isinstance(value, dict):
        raise HiLiftPackageAssemblyError(f"{label or path} must contain one JSON object")
    return value


def _json_path(parts: Sequence[str | int]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def unresolved_tokens(
    value: Any, path: tuple[str | int, ...] = ()
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            result.extend(unresolved_tokens(value[key], (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(unresolved_tokens(child, (*path, index)))
    elif isinstance(value, str) and value.startswith(TOKEN_PREFIXES):
        result.append({"path": _json_path(path), "token": value})
    return result


def _require_schema(value: Any, relative_path: str, label: str) -> None:
    schema = load_json(SCHEMA_ROOT / relative_path, label=f"schema {relative_path}")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda error: _json_path(error.absolute_path))
    if errors:
        rendered = "; ".join(
            f"{_json_path(error.absolute_path)}: {error.message}" for error in errors
        )
        raise HiLiftPackageAssemblyError(
            f"{label} does not satisfy {relative_path}: {rendered}"
        )


def _require_methodology_schema(value: Any) -> None:
    schema = load_json(SCHEMA_ROOT / "v3" / "submission.schema.json")
    fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        "$ref": "#/$defs/fluidsbench_methodology",
    }
    errors = sorted(
        Draft202012Validator(fragment, format_checker=FormatChecker()).iter_errors(value),
        key=lambda error: _json_path(error.absolute_path),
    )
    if errors:
        raise HiLiftPackageAssemblyError(
            "participant methodology does not satisfy the FluidsBench schema: "
            + "; ".join(
                f"{_json_path(error.absolute_path)}: {error.message}" for error in errors
            )
        )


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HiLiftPackageAssemblyError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise HiLiftPackageAssemblyError(f"{label} must remain inside {root}")
    result = (root / candidate).resolve()
    if not result.is_relative_to(root.resolve()):
        raise HiLiftPackageAssemblyError(f"{label} must remain inside {root}")
    return result


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value).issubset(SAFE_SHA256)
    ):
        raise HiLiftPackageAssemblyError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HiLiftPackageAssemblyError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise HiLiftPackageAssemblyError(f"{label} must be finite")
    return result


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _verify_fingerprint(value: Mapping[str, Any], label: str) -> None:
    expected = value.get("content_fingerprint")
    if not isinstance(expected, str):
        raise HiLiftPackageAssemblyError(f"{label} content fingerprint is absent")
    unsigned = dict(value)
    del unsigned["content_fingerprint"]
    if _canonical_fingerprint(unsigned) != expected:
        raise HiLiftPackageAssemblyError(f"{label} content fingerprint differs")


def _find_split(
    specification: Mapping[str, Any], specification_path: Path, split_id: str
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    matches = [
        item
        for item in specification.get("splits", [])
        if isinstance(item, dict) and item.get("id") == split_id
    ]
    if len(matches) != 1:
        raise HiLiftPackageAssemblyError(
            f"split_id {split_id!r} is not one official HiLiftAeroML split"
        )
    split = matches[0]
    if split.get("case_id_status") != "official":
        raise HiLiftPackageAssemblyError(f"split {split_id!r} lacks official case IDs")
    path = _safe_child(specification_path.parent, split.get("index_file"), "split index_file")
    if not path.is_file() or sha256_file(path) != split.get("sha256"):
        raise HiLiftPackageAssemblyError(
            f"split {split_id!r} index is absent or its SHA-256 changed"
        )
    index = load_json(path, label=f"split {split_id}")
    case_ids = index.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(case_id, str) and case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
        or len(case_ids) != split.get("case_count")
        or len(case_ids) != index.get("case_count")
    ):
        raise HiLiftPackageAssemblyError(f"split {split_id!r} case index is invalid")
    identities = {
        "dataset_id": "hiliftaeroml",
        "split_id": split_id,
        "case_set_id": split.get("case_set_id"),
        "case_id_status": "official",
    }
    for key, expected in identities.items():
        if index.get(key) != expected:
            raise HiLiftPackageAssemblyError(
                f"split {split_id!r} {key} must equal {expected!r}"
            )
    return split, list(case_ids), index


def _clean_https_release_url(value: Any, release_id: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and release_id in [part for part in parsed.path.split("/") if part]
    )


def _release_bindings(
    config: Mapping[str, Any], specification: Mapping[str, Any], specification_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bindings = config.get("release_bindings")
    if not isinstance(bindings, dict):
        raise HiLiftPackageAssemblyError("config.release_bindings must be an object")
    required = {
        "candidate_manifest",
        "evaluator",
        "native_profile_contract",
        "profile_ground_truth",
    }
    if set(bindings) != required:
        raise HiLiftPackageAssemblyError(
            f"release_bindings keys must be {sorted(required)}"
        )
    candidate = bindings["candidate_manifest"]
    evaluator = bindings["evaluator"]
    profile = bindings["native_profile_contract"]
    truth = bindings["profile_ground_truth"]
    if not all(isinstance(item, dict) for item in (candidate, evaluator, profile, truth)):
        raise HiLiftPackageAssemblyError("every release binding must be an object")
    if set(candidate) != {"status", "release_id", "manifest_url", "manifest_sha256"}:
        raise HiLiftPackageAssemblyError("candidate_manifest keys differ")
    if set(evaluator) != {"reference_version", "code_revision"}:
        raise HiLiftPackageAssemblyError("evaluator binding keys differ")
    if set(profile) != {"contract_id", "file", "format", "sha256"}:
        raise HiLiftPackageAssemblyError("native_profile_contract keys differ")
    if set(truth) != {"release_id", "manifest_sha256"}:
        raise HiLiftPackageAssemblyError("profile_ground_truth keys differ")
    if candidate.get("status") != "candidate":
        raise HiLiftPackageAssemblyError("candidate manifest status must be 'candidate'")
    release_id = candidate.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise HiLiftPackageAssemblyError("candidate release ID is invalid")
    _require_sha(candidate.get("manifest_sha256"), "candidate manifest digest")
    if not _clean_https_release_url(candidate.get("manifest_url"), release_id):
        raise HiLiftPackageAssemblyError("candidate manifest URL is not a clean release-scoped HTTPS URL")
    if (
        profile.get("contract_id") != PROFILE_CONTRACT_ID
        or profile.get("format") != PROFILE_FORMAT
        or profile.get("file") != "native-profile-format-v1.json"
        or profile.get("sha256") != PROFILE_CONTRACT_SHA256
    ):
        raise HiLiftPackageAssemblyError("native profile binding differs from the retained contract")
    _require_sha(truth.get("manifest_sha256"), "profile ground-truth manifest digest")
    if not isinstance(truth.get("release_id"), str) or not truth["release_id"]:
        raise HiLiftPackageAssemblyError("profile ground-truth release ID is invalid")

    support = specification.get("scoring_support")
    if not isinstance(support, dict):
        raise HiLiftPackageAssemblyError("HiLiftAeroML specification has no scoring_support")
    if support.get("submissions_open") is not False or support.get("status") not in {
        "candidate",
        "owner_review_required",
    }:
        raise HiLiftPackageAssemblyError("schema-v3 candidate assembly requires a closed candidate")
    owner_candidate = support.get("candidate_manifest")
    if not isinstance(owner_candidate, dict):
        raise HiLiftPackageAssemblyError("repository candidate support binding is absent")
    for key, expected in candidate.items():
        if owner_candidate.get(key) != expected:
            raise HiLiftPackageAssemblyError(
                f"config candidate_manifest.{key} differs from the repository binding"
            )
    manifest_path = _safe_child(
        specification_path.parent,
        owner_candidate.get("manifest_file"),
        "candidate manifest_file",
    )
    manifest = load_json(manifest_path, label="candidate scoring-support manifest")
    if sha256_file(manifest_path) != candidate["manifest_sha256"]:
        raise HiLiftPackageAssemblyError("candidate scoring-support manifest SHA-256 changed")
    errors = validate_candidate_manifest_release(
        dict(specification), specification_path.parent, manifest_path, manifest
    )
    if errors:
        raise HiLiftPackageAssemblyError(
            "candidate scoring-support release is invalid: " + "; ".join(errors)
        )

    owner_evaluator = support.get("dataset_evaluator_binding")
    if not isinstance(owner_evaluator, dict) or owner_evaluator.get("status") != "frozen":
        raise HiLiftPackageAssemblyError(
            "HiLiftAeroML evaluator revision is not frozen; force/overall and package output remain unavailable"
        )
    revision = evaluator.get("code_revision")
    if (
        evaluator.get("reference_version") != specification.get("evaluation_reference_version")
        or evaluator.get("reference_version")
        != owner_evaluator.get("evaluator_reference_version")
        or revision != owner_evaluator.get("evaluator_code_revision")
        or not isinstance(revision, str)
        or len(revision) not in {40, 64}
        or not set(revision).issubset(SAFE_SHA256)
    ):
        raise HiLiftPackageAssemblyError("config evaluator differs from the frozen repository binding")
    owner_profile = specification.get("profile_definition")
    if not isinstance(owner_profile, dict):
        raise HiLiftPackageAssemblyError(
            "HiLiftAeroML profile definition and ground-truth release are not published"
        )
    for key, expected in profile.items():
        if owner_profile.get(key) != expected:
            raise HiLiftPackageAssemblyError(
                f"config native_profile_contract.{key} differs from the repository binding"
            )
    # This adapter is deliberately local-candidate-only.  The public binding
    # remains unpublished and must never be treated as activated by this path.
    owner_truth = owner_profile.get("profile_ground_truth")
    if owner_truth != {
        "status": "not_published",
        "release_id": None,
        "manifest_sha256": None,
    }:
        raise HiLiftPackageAssemblyError(
            "HiLiftAeroML public profile-ground-truth activation boundary differs"
        )
    owner_candidate_truth = owner_profile.get(
        "candidate_dry_run_profile_ground_truth"
    )
    if (
        not isinstance(owner_candidate_truth, dict)
        or owner_candidate_truth.get("status") != CANDIDATE_TRUTH_STATUS
        or owner_candidate_truth.get("usage") != CANDIDATE_USAGE
    ):
        raise HiLiftPackageAssemblyError(
            "HiLiftAeroML local candidate profile-ground-truth binding is absent"
        )
    for key, expected in truth.items():
        if owner_candidate_truth.get(key) != expected:
            raise HiLiftPackageAssemblyError(
                "config profile_ground_truth."
                f"{key} differs from the local candidate repository binding"
            )
    return dict(candidate), dict(evaluator), dict(owner_candidate_truth)


def _compact_profile_declaration(
    specification: Mapping[str, Any], specification_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the additive compact contract without changing v1 activation."""

    definition = specification.get("compact_profile_definition")
    if not isinstance(definition, dict):
        raise HiLiftPackageAssemblyError(
            "HiLiftAeroML compact profile definition is absent"
        )
    expected_contract = {
        "contract_id": COMPACT_PROFILE_CONTRACT_ID,
        "file": COMPACT_PROFILE_CONTRACT_PATH.name,
        "format": COMPACT_PROFILE_FORMAT,
        "sha256": COMPACT_PROFILE_CONTRACT_SHA256,
    }
    if definition.get("status") != "additive_candidate_not_bound":
        raise HiLiftPackageAssemblyError(
            "compact profile definition must remain an unbound additive candidate"
        )
    for key, expected in expected_contract.items():
        if definition.get(key) != expected:
            raise HiLiftPackageAssemblyError(
                f"compact profile definition {key} differs from {expected!r}"
            )
    contract_path = _safe_child(
        specification_path.parent,
        definition.get("file"),
        "compact profile contract file",
    )
    if contract_path != COMPACT_PROFILE_CONTRACT_PATH.resolve():
        raise HiLiftPackageAssemblyError(
            "compact profile contract resolves outside its retained repository path"
        )
    if sha256_file(contract_path) != COMPACT_PROFILE_CONTRACT_SHA256:
        raise HiLiftPackageAssemblyError("compact profile contract SHA-256 changed")
    if definition.get("evaluator_support") != {
        "status": "not_published",
        "release_id": None,
        "manifest_sha256": None,
    }:
        raise HiLiftPackageAssemblyError(
            "public compact evaluator-support activation boundary differs"
        )
    candidate = definition.get("candidate_dry_run_evaluator_support")
    if (
        not isinstance(candidate, dict)
        or candidate.get("status") != "complete_candidate_not_published"
        or candidate.get("usage") != "maintainer_local_candidate_dry_run_only"
        or not isinstance(candidate.get("release_id"), str)
        or not candidate["release_id"]
    ):
        raise HiLiftPackageAssemblyError(
            "local candidate compact evaluator-support declaration is invalid"
        )
    _require_sha(
        candidate.get("manifest_sha256"),
        "candidate compact evaluator-support manifest digest",
    )
    return dict(definition), dict(candidate)


def _verify_native_aggregate(
    *,
    aggregate_root: Path,
    case_ids: Sequence[str],
    case_set_id: str,
    case_set_sha256: str,
) -> dict[str, dict[str, Any]]:
    _require_sha(case_set_sha256, "selected case-set digest")
    if not aggregate_root.is_dir() or aggregate_root.is_symlink():
        raise HiLiftPackageAssemblyError(
            f"native aggregate must be a regular directory: {aggregate_root}"
        )
    entries = {
        path.name
        for path in aggregate_root.iterdir()
        if not path.name.startswith(".")
    }
    if entries != NATIVE_AGGREGATE_FILES:
        raise HiLiftPackageAssemblyError(
            "native aggregate inventory differs; "
            f"missing={sorted(NATIVE_AGGREGATE_FILES-entries)}, "
            f"extra={sorted(entries-NATIVE_AGGREGATE_FILES)}"
        )
    documents = {
        name: load_json(aggregate_root / name, label=f"native {name}")
        for name in sorted(NATIVE_AGGREGATE_FILES)
    }
    artifact = documents["artifact_manifest.json"]
    contract = NATIVE_AGGREGATE_CONTRACTS.get(artifact.get("schema_id"))
    if contract is None or artifact.get("status") != "complete":
        raise HiLiftPackageAssemblyError("native aggregate artifact manifest differs")
    if (
        contract["family"] == "legacy_full360_v2"
        and (case_set_id != LEGACY_FULL360_CASE_SET_ID or len(case_ids) != 360)
    ):
        raise HiLiftPackageAssemblyError(
            "legacy Full360 native aggregate is bounded to its retained 360-case set"
        )
    _verify_fingerprint(artifact, "native aggregate artifact manifest")
    declared = artifact.get("artifacts")
    expected_declared = NATIVE_AGGREGATE_FILES - {"artifact_manifest.json"}
    if not isinstance(declared, dict) or set(declared) != expected_declared:
        raise HiLiftPackageAssemblyError("native aggregate declared artifact inventory differs")
    for name in expected_declared:
        descriptor = declared[name]
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("sha256") != sha256_file(aggregate_root / name)
        ):
            raise HiLiftPackageAssemblyError(f"native aggregate {name} SHA-256 differs")
    if artifact.get("case_set_id") != case_set_id:
        raise HiLiftPackageAssemblyError("native aggregate case-set ID differs")

    results = documents["submission_results.json"]
    cases = documents["case_metrics.json"]
    auxiliary = documents["auxiliary_summary.json"]
    regional = documents["regional_diagnostics_aggregate.json"]
    if (
        results.get("schema_id") != contract["result_schema"]
        or cases.get("schema_id") != contract["case_metrics_schema"]
        or auxiliary.get("schema_id") != contract["auxiliary_schema"]
        or regional.get("schema_id") != NATIVE_REGIONAL_SCHEMA
    ):
        raise HiLiftPackageAssemblyError(
            "native aggregate document schema family differs"
        )
    for label, value in (
        ("results", results),
        ("auxiliary", auxiliary),
        ("regional", regional),
    ):
        if value.get("case_set_id") != case_set_id or value.get("case_count") != len(case_ids):
            raise HiLiftPackageAssemblyError(f"native aggregate {label} case-set binding differs")
    raw_case_records = cases.get("cases")
    if (
        cases.get("case_set_id") != case_set_id
        or not isinstance(raw_case_records, list)
        or len(raw_case_records) != len(case_ids)
    ):
        raise HiLiftPackageAssemblyError("native aggregate cases case-set binding differs")
    if contract["require_complete_case_set_binding"]:
        for label, value in (
            ("artifact manifest", artifact),
            ("results", results),
            ("cases", cases),
            ("auxiliary", auxiliary),
            ("regional", regional),
        ):
            if (
                value.get("case_set_sha256") != case_set_sha256
                or value.get("case_count") != len(case_ids)
            ):
                raise HiLiftPackageAssemblyError(
                    f"native aggregate {label} exact case-set binding differs"
                )
    else:
        # Retained legacy documents did not repeat both values everywhere, but
        # any binding they do declare must still agree with the selected set.
        for label, value in (
            ("artifact manifest", artifact),
            ("results", results),
            ("cases", cases),
            ("auxiliary", auxiliary),
            ("regional", regional),
        ):
            if (
                "case_set_sha256" in value
                and value.get("case_set_sha256") != case_set_sha256
            ):
                raise HiLiftPackageAssemblyError(
                    f"native aggregate {label} case-set digest differs"
                )
    if (
        cases.get("status") != "complete"
        or auxiliary.get("status") not in contract["auxiliary_statuses"]
        or results.get("status") != "complete_closed_candidate"
        or regional.get("status") != "complete_report_only"
    ):
        raise HiLiftPackageAssemblyError("native aggregate completion status differs")
    if cases.get("case_order") != list(case_ids):
        raise HiLiftPackageAssemblyError("native aggregate case order differs from the selected split")
    expected_order_sha256 = _canonical_sha256(list(case_ids))
    if contract["require_complete_case_set_binding"]:
        if results.get("case_order_sha256") != expected_order_sha256:
            raise HiLiftPackageAssemblyError(
                "native aggregate ordered-case digest differs"
            )
    elif (
        "case_order_sha256" in results
        and results.get("case_order_sha256") != expected_order_sha256
    ):
        raise HiLiftPackageAssemblyError("native aggregate ordered-case digest differs")
    case_records = raw_case_records
    if (
        not isinstance(case_records, list)
        or [record.get("case_id") for record in case_records if isinstance(record, dict)]
        != list(case_ids)
    ):
        raise HiLiftPackageAssemblyError("native aggregate case records are incomplete or reordered")
    index_key = (
        "case_index"
        if contract["require_complete_case_set_binding"]
        else "full_case_index"
    )
    for case_index, record in enumerate(case_records):
        if record.get(index_key) != case_index:
            raise HiLiftPackageAssemblyError(
                f"native aggregate {record.get('case_id')} {index_key} differs"
            )
        inputs = record.get("input_sha256")
        if not isinstance(inputs, Mapping) or set(inputs) != NATIVE_CASE_INPUT_SHA256_KEYS:
            raise HiLiftPackageAssemblyError(
                f"native aggregate {record.get('case_id')} input SHA-256 inventory differs"
            )
        for key, digest in inputs.items():
            _require_sha(digest, f"native aggregate {record['case_id']}/{key}")
    return documents


def _case_input_sha256(
    case_records: Mapping[str, Mapping[str, Any]], *, case_id: str, artifact_id: str
) -> str:
    record = case_records.get(case_id)
    if not isinstance(record, Mapping):
        raise HiLiftPackageAssemblyError(
            f"native aggregate has no case record for {case_id}"
        )
    inputs = record.get("input_sha256")
    if not isinstance(inputs, Mapping) or set(inputs) != NATIVE_CASE_INPUT_SHA256_KEYS:
        raise HiLiftPackageAssemblyError(
            f"native aggregate {case_id} input SHA-256 inventory differs"
        )
    if artifact_id not in NATIVE_CASE_INPUT_SHA256_KEYS:
        raise HiLiftPackageAssemblyError(
            f"native aggregate artifact binding {artifact_id!r} is unknown"
        )
    return _require_sha(inputs.get(artifact_id), f"native aggregate {case_id}/{artifact_id}")


def _validate_live_artifact_descriptor(
    descriptor: Any,
    *,
    path: Path,
    label: str,
    expected_sha256: str | None = None,
) -> str:
    """Validate one retained artifact descriptor against the live regular file."""

    if not isinstance(descriptor, Mapping):
        raise HiLiftPackageAssemblyError(f"{label} descriptor is absent")
    declared_sha256 = _require_sha(descriptor.get("sha256"), f"{label} digest")
    if expected_sha256 is not None:
        _require_sha(expected_sha256, f"{label} expected digest")
        if declared_sha256 != expected_sha256:
            raise HiLiftPackageAssemblyError(
                f"{label} digest differs from its upstream provenance binding"
            )
    declared_path = descriptor.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise HiLiftPackageAssemblyError(f"{label} path is absent")
    if Path(declared_path).resolve() != path.resolve():
        raise HiLiftPackageAssemblyError(f"{label} path differs from the selected case")
    if "filename" in descriptor and descriptor.get("filename") != path.name:
        raise HiLiftPackageAssemblyError(f"{label} filename differs")
    declared_size = descriptor.get("size_bytes")
    if (
        not isinstance(declared_size, int)
        or isinstance(declared_size, bool)
        or declared_size < 0
    ):
        raise HiLiftPackageAssemblyError(f"{label} size is invalid")
    if not path.is_file() or path.is_symlink():
        raise HiLiftPackageAssemblyError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    before = path.stat()
    actual_sha256 = sha256_file(path)
    after = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise HiLiftPackageAssemblyError(f"{label} changed while it was hashed")
    if before.st_size != declared_size:
        raise HiLiftPackageAssemblyError(f"{label} size differs")
    if actual_sha256 != declared_sha256:
        raise HiLiftPackageAssemblyError(f"{label} SHA-256 differs")
    return declared_sha256


def _verify_native_receipts(
    *,
    receipts_root: Path,
    outputs_root: Path | None,
    case_ids: Sequence[str],
    case_set_id: str,
    case_set_sha256: str,
    case_records: Mapping[str, Mapping[str, Any]],
    surface_outputs_root: Path | None = None,
    volume_outputs_root: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Verify aggregate -> receipt -> summary -> consumed-artifact provenance."""

    if not receipts_root.is_dir() or receipts_root.is_symlink():
        raise HiLiftPackageAssemblyError(
            f"native receipts must be a regular directory: {receipts_root}"
        )
    output_roots = _resolve_native_output_roots(
        outputs_root=outputs_root,
        surface_outputs_root=surface_outputs_root,
        volume_outputs_root=volume_outputs_root,
    )
    for domain, root in output_roots.items():
        if not root.is_dir() or root.is_symlink():
            raise HiLiftPackageAssemblyError(
                f"native {domain} outputs must be a regular directory: {root}"
            )
    _require_sha(case_set_sha256, "selected case-set digest")
    profile_sha256: dict[str, dict[str, str]] = {}
    observed_receipt_schema: str | None = None
    for case_index, case_id in enumerate(case_ids):
        record = case_records.get(case_id)
        if not isinstance(record, Mapping):
            raise HiLiftPackageAssemblyError(
                f"native aggregate has no case record for {case_id}"
            )
        receipt = load_json(
            receipts_root / f"{case_id}.json",
            label=f"{case_id} native receipt",
            expected_sha256=_case_input_sha256(
                case_records, case_id=case_id, artifact_id="receipt"
            ),
        )
        receipt_schema = receipt.get("schema_id")
        if receipt_schema not in {
            NATIVE_RECEIPT_SCHEMA,
            LEGACY_NATIVE_RECEIPT_SCHEMA,
        }:
            raise HiLiftPackageAssemblyError(
                f"{case_id} native receipt schema differs"
            )
        if observed_receipt_schema is None:
            observed_receipt_schema = receipt_schema
        elif receipt_schema != observed_receipt_schema:
            raise HiLiftPackageAssemblyError(
                "native receipt schema family changes within the selected case set"
            )
        index_key = (
            "case_index"
            if receipt_schema == NATIVE_RECEIPT_SCHEMA
            else "full_case_index"
        )
        if (
            receipt.get("status") != "complete"
            or receipt.get("case_id") != case_id
            or receipt.get("case_set_id") != case_set_id
            or receipt.get("case_set_sha256") != case_set_sha256
            or receipt.get(index_key) != case_index
            or receipt.get(index_key) != record.get(index_key)
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} native receipt identity/completion binding differs"
            )
        _require_sha(
            receipt.get("campaign_content_fingerprint"),
            f"{case_id} campaign content fingerprint",
        )
        _verify_fingerprint(receipt, f"{case_id} native receipt")

        contracts = receipt.get("contracts")
        if not isinstance(contracts, Mapping):
            raise HiLiftPackageAssemblyError(f"{case_id} receipt contracts are absent")
        for contract_id, aggregate_id in (
            ("raw_id_sequences", "raw_contract"),
            ("truth_supports", "truth_contract"),
        ):
            descriptor = contracts.get(contract_id)
            if not isinstance(descriptor, Mapping):
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{contract_id} contract descriptor is absent"
                )
            expected = _case_input_sha256(
                case_records, case_id=case_id, artifact_id=aggregate_id
            )
            if descriptor.get("sha256") != expected:
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{contract_id} differs from the aggregate binding"
                )
            _require_sha(descriptor.get("sha256"), f"{case_id}/{contract_id} digest")
            size = descriptor.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{contract_id} contract size is invalid"
                )

        domains = receipt.get("domains")
        if not isinstance(domains, Mapping) or set(domains) != {"surface", "volume"}:
            raise HiLiftPackageAssemblyError(
                f"{case_id} receipt domain inventory differs"
            )
        case_profile_sha256: dict[str, str] = {}
        for domain in ("surface", "volume"):
            receipt_domain = domains[domain]
            if not isinstance(receipt_domain, Mapping):
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{domain} receipt domain is invalid"
                )
            run_fingerprint = _require_sha(
                receipt_domain.get("run_fingerprint"),
                f"{case_id}/{domain} run fingerprint",
            )
            stream = (
                output_roots[domain]
                / case_id
                / f"{domain}_submission_stream"
            )
            summary_path = stream / "summary.json"
            summary_sha256 = _validate_live_artifact_descriptor(
                receipt_domain.get("summary"),
                path=summary_path,
                label=f"{case_id}/{domain} summary",
            )
            summary = load_json(
                summary_path,
                label=f"{case_id}/{domain} summary",
                expected_sha256=summary_sha256,
            )
            if domain == "surface":
                allowed_summary_versions = (
                    {SURFACE_SUMMARY_SCHEMA_VERSION}
                    if receipt_schema == NATIVE_RECEIPT_SCHEMA
                    else {
                        LEGACY_SURFACE_SUMMARY_SCHEMA_VERSION,
                        SURFACE_SUMMARY_SCHEMA_VERSION,
                    }
                )
            else:
                allowed_summary_versions = {VOLUME_SUMMARY_SCHEMA_VERSION}
            if (
                summary.get("summary_schema_version")
                not in allowed_summary_versions
                or summary.get("status") != "complete"
                or summary.get("case_id") != case_id
                or summary.get("run_fingerprint") != run_fingerprint
            ):
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{domain} summary identity/completion binding differs"
                )

            summary_submission = summary.get("submission_artifacts")
            if not isinstance(summary_submission, Mapping):
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{domain} summary submission artifacts are absent"
                )
            for receipt_id, aggregate_id, filename in (
                ("submission_support_score", f"{domain}_support", "submission_support_score.json"),
                ("regional_diagnostics", f"{domain}_regional", "regional_diagnostics.json"),
            ):
                expected = _case_input_sha256(
                    case_records, case_id=case_id, artifact_id=aggregate_id
                )
                artifact_path = stream / filename
                _validate_live_artifact_descriptor(
                    receipt_domain.get(receipt_id),
                    path=artifact_path,
                    label=f"{case_id}/{domain} receipt {receipt_id}",
                    expected_sha256=expected,
                )
                _validate_live_artifact_descriptor(
                    summary_submission.get(receipt_id),
                    path=artifact_path,
                    label=f"{case_id}/{domain} summary {receipt_id}",
                    expected_sha256=expected,
                )

            if domain == "surface":
                auxiliary = summary.get("auxiliary_artifacts")
                diagnostic = receipt_domain.get("surface_load_diagnostic")
                if not isinstance(auxiliary, Mapping) or not isinstance(
                    diagnostic, Mapping
                ):
                    raise HiLiftPackageAssemblyError(
                        f"{case_id}/surface load/profile descriptors are absent"
                    )
                load_sha256 = _case_input_sha256(
                    case_records, case_id=case_id, artifact_id="surface_loads"
                )
                load_path = stream / "surface_loads.json"
                _validate_live_artifact_descriptor(
                    diagnostic.get("artifact"),
                    path=load_path,
                    label=f"{case_id}/surface receipt loads",
                    expected_sha256=load_sha256,
                )
                _validate_live_artifact_descriptor(
                    auxiliary.get("surface_loads"),
                    path=load_path,
                    label=f"{case_id}/surface summary loads",
                    expected_sha256=load_sha256,
                )
                profile_descriptors = auxiliary
                profile_files = (
                    ("cp_profile_metrics", "cp_profile_metrics.json"),
                    ("cp_cut_values", "cp_cut_values.npz"),
                )
            else:
                profile_descriptors = summary.get("artifacts")
                if not isinstance(profile_descriptors, Mapping):
                    raise HiLiftPackageAssemblyError(
                        f"{case_id}/volume profile descriptors are absent"
                    )
                profile_files = (
                    ("velocity_profile_metrics", "velocity_profile_metrics.json"),
                    ("velocity_profiles", "velocity_profiles.npz"),
                )
            for artifact_id, filename in profile_files:
                case_profile_sha256[artifact_id] = _validate_live_artifact_descriptor(
                    profile_descriptors.get(artifact_id),
                    path=stream / filename,
                    label=f"{case_id}/{domain} {artifact_id}",
                )
        profile_sha256[case_id] = case_profile_sha256
    return profile_sha256


def _load_native_support(
    path: Path,
    *,
    case_id: str,
    support_id: str,
    expected_sha256: str,
) -> dict[str, Any]:
    support = load_json(
        path,
        label=f"{case_id}/{support_id}",
        expected_sha256=expected_sha256,
    )
    if (
        support.get("schema") != NATIVE_SUPPORT_SCHEMA
        or support.get("status") != "complete"
        or support.get("case_id") != case_id
        or support.get("support_id") != support_id
        or support.get("coverage", {}).get("complete_exact_truth_membership") is not True
        or support.get("coverage", {}).get("gap_free_duplicate_free_support_positions") is not True
    ):
        raise HiLiftPackageAssemblyError(f"{case_id}/{support_id} support is incomplete")
    count = support.get("coverage", {}).get("point_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise HiLiftPackageAssemblyError(f"{case_id}/{support_id} point count is invalid")
    return support


def _native_metric(
    support: Mapping[str, Any], source: tuple[str, str, str], label: str
) -> float:
    weighting, field, metric = source
    try:
        value = support["weightings"][weighting][field]["metrics"][metric]
    except (KeyError, TypeError) as error:
        raise HiLiftPackageAssemblyError(f"{label} is absent") from error
    return _finite(value, label)


def _support_bindings(manifest: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for support in manifest.get("supports", []):
        if not isinstance(support, dict) or not isinstance(support.get("id"), str):
            continue
        result[support["id"]] = {
            binding["metric_id"]: binding
            for binding in support.get("metric_bindings", [])
            if isinstance(binding, dict) and isinstance(binding.get("metric_id"), str)
        }
    return result


def _relative_l2_statistics(
    *,
    native_support: Mapping[str, Any],
    metric_id: str,
    source: tuple[str, str, str],
    binding: Mapping[str, Any],
    support_count: int,
) -> dict[str, Any]:
    weighting, field, _ = source
    sums = native_support["weightings"][weighting][field]["sufficient_statistics"]
    numerator = _finite(sums.get("sum_squared_error"), f"{metric_id} numerator")
    denominator = _finite(sums.get("sum_squared_truth"), f"{metric_id} denominator")
    if numerator < 0.0 or denominator <= 0.0:
        raise HiLiftPackageAssemblyError(f"{metric_id} relative-L2 sufficient statistics are invalid")
    native_total_weight = _finite(sums.get("weight_sum"), f"{metric_id} total weight")
    component_count = 3 if metric_id in VECTOR_METRIC_IDS else 1
    total_weight = native_total_weight / component_count
    expected_total_weight = float(support_count) if binding.get("weighting") == "uniform" else None
    if expected_total_weight is not None and not math.isclose(
        total_weight, expected_total_weight, rel_tol=0.0, abs_tol=1e-9
    ):
        raise HiLiftPackageAssemblyError(
            f"{metric_id} uniform total weight differs from native support count"
        )
    return {
        "reduction": "relative_l2_percent",
        "weighting": binding.get("weighting"),
        "dataset_weighting": binding.get("dataset_weighting"),
        "numerator": numerator,
        "denominator": denominator,
        "entity_count": support_count,
        "total_weight": total_weight,
    }


def _field_support_record(
    *,
    case_id: str,
    support_id: str,
    native_support: Mapping[str, Any],
    concise_case: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    support_count = native_support["coverage"]["point_count"]
    source_map = SURFACE_METRIC_SOURCE if support_id == SURFACE_SUPPORT_ID else VOLUME_METRIC_SOURCE
    metric_values: dict[str, float] = {}
    statistics: dict[str, Any] = {}
    for metric_id, binding in bindings.items():
        if binding.get("case_evidence") != "metric_value":
            continue
        if metric_id in source_map:
            source = source_map[metric_id]
            value = _native_metric(native_support, source, f"{case_id}/{metric_id}")
            metric_values[metric_id] = value
            if metric_id in RELATIVE_L2_IDS:
                statistics[metric_id] = _relative_l2_statistics(
                    native_support=native_support,
                    metric_id=metric_id,
                    source=source,
                    binding=binding,
                    support_count=support_count,
                )
        elif metric_id in DIMENSIONAL_METRIC_SOURCE:
            domain, field, reduction = DIMENSIONAL_METRIC_SOURCE[metric_id]
            if (
                (support_id == SURFACE_SUPPORT_ID and domain != "surface")
                or (support_id == VOLUME_SUPPORT_ID and domain != "volume")
            ):
                continue
            try:
                raw = concise_case["domains"][domain]["fields"][field]["dimensional"][reduction]
            except (KeyError, TypeError) as error:
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{metric_id} dimensional native metric is absent"
                ) from error
            metric_values[metric_id] = _finite(raw, f"{case_id}/{metric_id}")
        else:
            raise HiLiftPackageAssemblyError(
                f"{case_id}/{support_id} has no native adapter for metric {metric_id!r}"
            )
    expected = {
        metric_id
        for metric_id, binding in bindings.items()
        if binding.get("case_evidence") == "metric_value"
    }
    if set(metric_values) != expected:
        raise HiLiftPackageAssemblyError(
            f"{case_id}/{support_id} adapted metric inventory differs; "
            f"missing={sorted(expected-set(metric_values))}"
        )
    record = {
        "support_id": support_id,
        "support_count": support_count,
        "scored_count": support_count,
        "coverage_fraction": 1.0,
        "weight_coverage_fraction": 1.0,
        "unmapped_count": 0,
        "extrapolated_count": 0,
        "metric_values": metric_values,
        "metric_sufficient_statistics": statistics,
    }
    return record, metric_values


def _r2(truth: Sequence[float], prediction: Sequence[float], label: str) -> float:
    if len(truth) != len(prediction) or len(truth) < 2:
        raise HiLiftPackageAssemblyError(f"{label} requires matching complete case arrays")
    truth_mean = math.fsum(truth) / len(truth)
    numerator = math.fsum((predicted_value - truth_value) ** 2 for truth_value, predicted_value in zip(truth, prediction, strict=True))
    denominator = math.fsum((truth_value - truth_mean) ** 2 for truth_value in truth)
    if denominator <= 0.0:
        raise HiLiftPackageAssemblyError(f"{label} target variance is zero")
    return 1.0 - numerator / denominator


def _force_coefficient_evidence(load: Mapping[str, Any]) -> dict[str, float]:
    """Return the complete per-case inputs needed to replay force metrics."""

    return {
        field: _finite(load[field], f"force coefficient evidence {field}")
        for field in HILIFT_FORCE_COEFFICIENT_FIELDS
    }


def _dataset_evaluator_evidence(evaluator: Mapping[str, Any]) -> dict[str, str]:
    """Separate the frozen dataset evaluator from participant code identity."""

    reference_version = evaluator.get("reference_version")
    code_revision = evaluator.get("code_revision")
    if not isinstance(reference_version, str) or not reference_version:
        raise HiLiftPackageAssemblyError("dataset evaluator reference version is absent")
    if (
        not isinstance(code_revision, str)
        or len(code_revision) not in {40, 64}
        or not set(code_revision).issubset(SAFE_SHA256)
    ):
        raise HiLiftPackageAssemblyError(
            "dataset evaluator code revision must be a lowercase full Git SHA-1 or SHA-256"
        )
    return {
        "status": "frozen",
        "reference_version": reference_version,
        "code_revision": code_revision,
    }


def _require_exact_surface_load_result(
    value: Any, *, case_id: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HiLiftPackageAssemblyError(
            f"{case_id} exact surface-load result is absent"
        )
    identities = {
        "schema_id": EXACT_SURFACE_LOAD_RESULT_SCHEMA,
        "status": "complete",
        "algorithm_id": EXACT_SURFACE_LOAD_ALGORITHM_ID,
        "force_basis_id": EXACT_SURFACE_FORCE_BASIS_ID,
        "pitch_moment_basis_id": EXACT_SURFACE_PITCH_BASIS_ID,
        "input_basis_id": EXACT_SURFACE_INPUT_BASIS_ID,
        "output_basis_id": EXACT_SURFACE_OUTPUT_BASIS_ID,
        "pressure_convention_id": EXACT_SURFACE_PRESSURE_CONVENTION_ID,
    }
    for key, expected in identities.items():
        if value.get(key) != expected:
            raise HiLiftPackageAssemblyError(
                f"{case_id} exact surface-load {key} differs"
            )
    if not isinstance(value.get("weight_support_id"), str) or not value[
        "weight_support_id"
    ]:
        raise HiLiftPackageAssemblyError(
            f"{case_id} exact surface-load weight-support identity is absent"
        )
    coverage = value.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or not isinstance(coverage.get("point_count"), int)
        or isinstance(coverage.get("point_count"), bool)
        or coverage["point_count"] < 1
        or not isinstance(coverage.get("chunk_count"), int)
        or isinstance(coverage.get("chunk_count"), bool)
        or coverage["chunk_count"] < 1
    ):
        raise HiLiftPackageAssemblyError(
            f"{case_id} exact surface-load coverage is invalid"
        )
    reference = value.get("reference")
    if not isinstance(reference, Mapping):
        raise HiLiftPackageAssemblyError(
            f"{case_id} exact surface-load reference is absent"
        )
    q_inf = _finite(reference.get("q_inf"), f"{case_id} load q_inf")
    q_ref = _finite(reference.get("q_ref"), f"{case_id} load q_ref")
    scale = _finite(
        reference.get("qinf_to_qref_scale"),
        f"{case_id} qinf-to-qRef scale",
    )
    if (
        q_inf <= 0.0
        or q_ref <= 0.0
        or scale <= 0.0
        or not math.isclose(scale, q_inf / q_ref, rel_tol=5e-15, abs_tol=0.0)
    ):
        raise HiLiftPackageAssemblyError(
            f"{case_id} exact surface-load qRef normalization differs"
        )
    loads = value.get("loads")
    if not isinstance(loads, Mapping) or set(loads) != {"prediction", "truth"}:
        raise HiLiftPackageAssemblyError(
            f"{case_id} exact surface-load side inventory differs"
        )
    for side_name in ("prediction", "truth"):
        side = loads[side_name]
        total = side.get("total") if isinstance(side, Mapping) else None
        if (
            not isinstance(total, Mapping)
            or total.get("pitch_moment_quadrature") != "exact_degree2"
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} {side_name} pitching moment is not exact degree two"
            )
    return value


def _load_complete_loads(
    *,
    case_ids: Sequence[str],
    outputs_root: Path,
    case_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    unavailable: list[dict[str, str]] = []
    for case_id in case_ids:
        stream = outputs_root / case_id / "surface_submission_stream"
        path = stream / "surface_loads.json"
        expected_sha256 = _case_input_sha256(
            case_records, case_id=case_id, artifact_id="surface_loads"
        )
        value = load_json(
            path,
            label=f"{case_id} surface loads",
            expected_sha256=expected_sha256,
        )
        summary = load_json(stream / "summary.json", label=f"{case_id} surface summary")
        summary_artifact = summary.get("auxiliary_artifacts", {}).get("surface_loads")
        run_fingerprint = _require_sha(
            value.get("run_fingerprint"), f"{case_id} surface-load run fingerprint"
        )
        if (
            summary.get("summary_schema_version")
            != SURFACE_SUMMARY_SCHEMA_VERSION
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} force scoring requires release-ready surface summary "
                f"schema {SURFACE_SUMMARY_SCHEMA_VERSION}"
            )
        if (
            summary.get("status") != "complete"
            or summary.get("case_id") != case_id
            or summary.get("run_fingerprint") != run_fingerprint
            or not isinstance(summary_artifact, Mapping)
            or summary_artifact.get("sha256") != expected_sha256
            or summary_artifact.get("size_bytes") != path.stat().st_size
            or not isinstance(summary_artifact.get("path"), str)
            or Path(summary_artifact["path"]).resolve() != path.resolve()
            or value.get("case_id") != case_id
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} surface-load summary/case/hash binding differs"
            )
        status = value.get("status")
        if status != "complete":
            unavailable.append(
                {
                    "case_id": case_id,
                    "status": str(status),
                    "reason_code": str(value.get("reason_code", "missing_reason_code")),
                }
            )
            continue
        summary_point_count = summary.get("number_of_points")
        if (
            isinstance(summary_point_count, bool)
            or not isinstance(summary_point_count, int)
            or summary_point_count < 1
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} surface summary point count is invalid"
            )
        area_reconciliation = summary.get("load_metric_area_reconciliation")
        if (
            not isinstance(area_reconciliation, Mapping)
            or area_reconciliation.get("status") != "pass"
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} load/metric area reconciliation did not pass"
            )
        exact_result = _require_exact_surface_load_result(
            value.get("result"), case_id=case_id
        )
        exact_point_count = exact_result["coverage"]["point_count"]
        if exact_point_count != summary_point_count:
            raise HiLiftPackageAssemblyError(
                f"{case_id} exact surface-load coverage differs from the "
                "surface summary point count"
            )
        prediction = exact_result["loads"]["prediction"]["total"]
        truth = exact_result["loads"]["truth"]["total"]
        record = {
            "_surface_point_count": exact_point_count,
            "predicted_c_drag": _finite(prediction.get("cd"), f"{case_id} predicted CD"),
            "predicted_c_lift": _finite(prediction.get("cl"), f"{case_id} predicted CL"),
            "predicted_c_pitch": _finite(prediction.get("cm_body_y"), f"{case_id} predicted CM"),
            "truth_c_drag": _finite(truth.get("cd"), f"{case_id} truth CD"),
            "truth_c_lift": _finite(truth.get("cl"), f"{case_id} truth CL"),
            "truth_c_pitch": _finite(truth.get("cm_body_y"), f"{case_id} truth CM"),
        }
        result[case_id] = record
    if unavailable:
        rendered = ", ".join(
            f"{item['case_id']}[{item['reason_code']}]" for item in unavailable
        )
        raise HiLiftPackageAssemblyError(
            "force/overall metrics unavailable: exact complete surface-load truth "
            f"coverage is required; unavailable={rendered}; imputation and zero-fill are forbidden"
        )
    return result


def _native_case_records(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        record["case_id"]: record
        for record in documents["case_metrics.json"]["cases"]
    }


def _case_metrics_and_values(
    *,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    case_ids: Sequence[str],
    support_release: Any,
    native_documents: Mapping[str, Mapping[str, Any]],
    surface_outputs_root: Path,
    volume_outputs_root: Path,
    profile_metrics: Mapping[str, Mapping[str, float]],
    specification: Mapping[str, Any],
    support_release_id: str,
    support_manifest_sha256: str,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    bindings = _support_bindings(support_release.manifest)
    if set(bindings) != {SURFACE_SUPPORT_ID, VOLUME_SUPPORT_ID, SCALAR_SUPPORT_ID}:
        raise HiLiftPackageAssemblyError("candidate scoring-support IDs differ")
    concise = _native_case_records(native_documents)
    loads = _load_complete_loads(
        case_ids=case_ids,
        outputs_root=surface_outputs_root,
        case_records=concise,
    )
    case_documents: list[dict[str, Any]] = []
    aggregate_inputs: dict[str, list[float]] = {}
    drag_truth: list[float] = []
    drag_prediction: list[float] = []
    lift_truth: list[float] = []
    lift_prediction: list[float] = []
    for case_id in case_ids:
        load = loads[case_id]
        native_surface = _load_native_support(
            surface_outputs_root
            / case_id
            / "surface_submission_stream"
            / "submission_support_score.json",
            case_id=case_id,
            support_id=NATIVE_SURFACE_SUPPORT_ID,
            expected_sha256=_case_input_sha256(
                concise, case_id=case_id, artifact_id="surface_support"
            ),
        )
        native_volume = _load_native_support(
            volume_outputs_root
            / case_id
            / "volume_submission_stream"
            / "submission_support_score.json",
            case_id=case_id,
            support_id=NATIVE_VOLUME_SUPPORT_ID,
            expected_sha256=_case_input_sha256(
                concise, case_id=case_id, artifact_id="volume_support"
            ),
        )
        expected_instances = {
            instance["support_id"]: instance["entity_count"]
            for instance in support_release.cases[case_id]["support_instances"]
        }
        if (
            native_surface["coverage"]["point_count"] != expected_instances[SURFACE_SUPPORT_ID]
            or native_volume["coverage"]["point_count"] != expected_instances[VOLUME_SUPPORT_ID]
            or expected_instances[SCALAR_SUPPORT_ID] != 1
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} native support count differs from scoring support"
            )
        if (
            load["_surface_point_count"]
            != native_surface["coverage"]["point_count"]
        ):
            raise HiLiftPackageAssemblyError(
                f"{case_id} exact surface-load coverage differs from native "
                "surface support"
            )
        surface_record, surface_values = _field_support_record(
            case_id=case_id,
            support_id=SURFACE_SUPPORT_ID,
            native_support=native_surface,
            concise_case=concise[case_id],
            bindings=bindings[SURFACE_SUPPORT_ID],
        )
        volume_record, volume_values = _field_support_record(
            case_id=case_id,
            support_id=VOLUME_SUPPORT_ID,
            native_support=native_volume,
            concise_case=concise[case_id],
            bindings=bindings[VOLUME_SUPPORT_ID],
        )
        scalar_values = {
            "c_drag_mae": abs(load["predicted_c_drag"] - load["truth_c_drag"]),
            "c_lift_mae": abs(load["predicted_c_lift"] - load["truth_c_lift"]),
            "c_pitch_mae": abs(load["predicted_c_pitch"] - load["truth_c_pitch"]),
        }
        expected_scalar = {
            metric_id
            for metric_id, binding in bindings[SCALAR_SUPPORT_ID].items()
            if binding.get("case_evidence") == "metric_value"
        }
        if set(scalar_values) != expected_scalar:
            raise HiLiftPackageAssemblyError("scalar per-case metric bindings differ")
        scalar_record = {
            "support_id": SCALAR_SUPPORT_ID,
            "support_count": 1,
            "scored_count": 1,
            "coverage_fraction": 1.0,
            "weight_coverage_fraction": 1.0,
            "unmapped_count": 0,
            "extrapolated_count": 0,
            "metric_values": scalar_values,
            "metric_sufficient_statistics": {},
        }
        profiles = profile_metrics.get(case_id)
        if not isinstance(profiles, Mapping) or set(profiles) != {
            "cp_cut_r2",
            "velocity_profile_r2",
        }:
            raise HiLiftPackageAssemblyError(f"{case_id} profile metrics are incomplete")
        nonspatial = {
            **scalar_values,
            "cp_cut_r2": _finite(profiles["cp_cut_r2"], f"{case_id} Cp R2"),
            "velocity_profile_r2": _finite(
                profiles["velocity_profile_r2"], f"{case_id} velocity R2"
            ),
        }
        case_documents.append(
            {
                "case_id": case_id,
                "supports": [surface_record, volume_record, scalar_record],
                "nonspatial_metric_values": nonspatial,
                "force_coefficients": _force_coefficient_evidence(load),
            }
        )
        for metric_id, value in {
            **surface_values,
            **volume_values,
            **nonspatial,
        }.items():
            aggregate_inputs.setdefault(metric_id, []).append(value)
        drag_truth.append(load["truth_c_drag"])
        drag_prediction.append(load["predicted_c_drag"])
        lift_truth.append(load["truth_c_lift"])
        lift_prediction.append(load["predicted_c_lift"])

    metric_values = {
        metric_id: math.fsum(values) / len(values)
        for metric_id, values in aggregate_inputs.items()
        if metric_id not in {"cd_r2", "cl_r2"}
    }
    metric_values["cd_r2"] = _r2(drag_truth, drag_prediction, "CD R2")
    metric_values["cl_r2"] = _r2(lift_truth, lift_prediction, "CL R2")
    composite = specification.get("overall_score_composite")
    groups = specification.get("component_score_groups")
    if not isinstance(composite, Mapping) or not isinstance(groups, Mapping):
        raise HiLiftPackageAssemblyError("HiLiftAeroML composite score declarations are absent")
    try:
        metric_values.update(
            composite_component_group_scores(metric_values, composite, groups)
        )
        metric_values[composite["metric_id"]] = composite_overall_score(
            metric_values, composite
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HiLiftPackageAssemblyError(
            f"cannot derive HiLiftAeroML component/overall scores: {error}"
        ) from error
    required_ids = [
        metric["id"]
        for metric in specification.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("id"), str)
    ]
    if set(metric_values) != set(required_ids):
        raise HiLiftPackageAssemblyError(
            "native adapter cannot provide every specification metric; "
            f"missing={sorted(set(required_ids)-set(metric_values))}, "
            f"unexpected={sorted(set(metric_values)-set(required_ids))}"
        )
    metric_values = {metric_id: metric_values[metric_id] for metric_id in required_ids}
    result = {
        "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": "hiliftaeroml",
        "split_id": split_id,
        "case_set_id": case_set_id,
        "scoring_support_release_id": support_release_id,
        "scoring_support_manifest_sha256": support_manifest_sha256,
        "case_count": len(case_ids),
        "cases": case_documents,
        "metric_values": metric_values,
        "generated_at": generated_at,
    }
    _require_schema(result, "v3/case-metrics.schema.json", "metrics/cases.json")
    return result, metric_values


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise HiLiftPackageAssemblyError("cannot summarize an empty regional metric")
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "maximum": ordered[-1],
    }


def _regional_case_macro(values: Sequence[float | None]) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None]
    return {
        "value": math.fsum(defined) / len(defined) if defined else None,
        "defined_case_count": len(defined),
    }


def _regional_case_distribution(values: Sequence[float | None]) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None]
    return {
        "defined_case_count": len(defined),
        **(
            _summary(defined)
            if defined
            else {"minimum": None, "median": None, "p90": None, "maximum": None}
        ),
    }


def _optional_regional_metric(
    document: Mapping[str, Any], metric_id: str, label: str
) -> float | None:
    value = document.get("metrics", {}).get(metric_id)
    return None if value is None else _finite(value, label)


def _whole_support_normalized_rmse_percent(
    regional_sums: Mapping[str, Any],
    global_sums: Mapping[str, Any],
    *,
    label: str,
) -> float | None:
    regional_weight = _finite(
        regional_sums.get("weight_sum"), f"{label} regional weight_sum"
    )
    regional_squared_error = _finite(
        regional_sums.get("sum_squared_error"),
        f"{label} regional sum_squared_error",
    )
    global_weight = _finite(
        global_sums.get("weight_sum"), f"{label} whole-support weight_sum"
    )
    global_squared_truth = _finite(
        global_sums.get("sum_squared_truth"),
        f"{label} whole-support sum_squared_truth",
    )
    if regional_weight <= 0.0:
        return None
    if global_weight <= 0.0 or global_squared_truth <= 0.0:
        return None
    if regional_squared_error < 0.0:
        raise HiLiftPackageAssemblyError(
            f"{label} regional sum_squared_error is negative"
        )
    regional_rmse = math.sqrt(regional_squared_error / regional_weight)
    whole_support_truth_rms = math.sqrt(global_squared_truth / global_weight)
    return 100.0 * regional_rmse / whole_support_truth_rms


def _convert_regional(
    *,
    native: Mapping[str, Any],
    case_ids: Sequence[str],
    split_id: str,
    surface_outputs_root: Path,
    volume_outputs_root: Path,
    case_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        native.get("schema_id") != NATIVE_REGIONAL_SCHEMA
        or native.get("status") != "complete_report_only"
        or native.get("scoring_weight") != 0.0
        or native.get("case_count") != len(case_ids)
        or native.get("contract_sha256") != NATIVE_REGIONAL_CONTRACT_SHA256
    ):
        raise HiLiftPackageAssemblyError("native regional aggregate identity differs")
    domains: dict[str, Any] = {}
    checked_fields = 0
    output_roots = {
        "surface": surface_outputs_root,
        "volume": volume_outputs_root,
    }
    for domain, support_id, expected_regions in (
        ("surface", NATIVE_SURFACE_SUPPORT_ID, SURFACE_REGION_ORDER),
        ("volume", NATIVE_VOLUME_SUPPORT_ID, VOLUME_REGION_ORDER),
    ):
        raw_domain = native.get("domains", {}).get(domain)
        if (
            not isinstance(raw_domain, Mapping)
            or raw_domain.get("support_id") != support_id
            or raw_domain.get("region_order") != list(expected_regions)
            or not isinstance(raw_domain.get("fields"), Mapping)
        ):
            raise HiLiftPackageAssemblyError(f"native regional {domain} identity differs")
        case_reports = [
            load_json(
                output_roots[domain]
                / case_id
                / f"{domain}_submission_stream"
                / "regional_diagnostics.json",
                label=f"{case_id}/{domain} regional diagnostics",
                expected_sha256=_case_input_sha256(
                    case_records,
                    case_id=case_id,
                    artifact_id=f"{domain}_regional",
                ),
            )
            for case_id in case_ids
        ]
        expected_field_ids = set(raw_domain.get("fields", {}))
        for case_id, report in zip(case_ids, case_reports, strict=True):
            if (
                report.get("schema") != NATIVE_REGIONAL_CASE_SCHEMA
                or report.get("schema_version") != 1
                or report.get("status") != "complete_report_only"
                or report.get("case_id") != case_id
                or report.get("support_id") != support_id
                or report.get("definition_id") != REGIONAL_DEFINITION_ID
                or report.get("contract_sha256")
                != NATIVE_REGIONAL_CONTRACT_SHA256
                or report.get("scoring_weight") != 0.0
                or report.get("region_order") != list(expected_regions)
                or report.get("coverage", {}).get("complete") is not True
                or report.get("reconstruction", {}).get("status") != "pass"
                or set(report.get("fields", {})) != expected_field_ids
            ):
                raise HiLiftPackageAssemblyError(
                    f"{case_id}/{domain} regional case identity or coverage differs"
                )
            for field_id, case_field in report["fields"].items():
                if (
                    not isinstance(case_field, Mapping)
                    or not isinstance(case_field.get("global"), Mapping)
                    or [
                        row.get("region_id")
                        for row in case_field.get("regions", [])
                        if isinstance(row, Mapping)
                    ]
                    != list(expected_regions)
                ):
                    raise HiLiftPackageAssemblyError(
                        f"{case_id}/{domain}/{field_id} regional structure differs"
                    )
        fields: dict[str, Any] = {}
        for field_id, aggregate_field in raw_domain.get("fields", {}).items():
            if (
                not isinstance(aggregate_field, Mapping)
                or not isinstance(aggregate_field.get("global"), Mapping)
                or not isinstance(
                    aggregate_field.get("global", {}).get("pooled_metrics"),
                    Mapping,
                )
                or [
                    row.get("region_id")
                    for row in aggregate_field.get("regions", [])
                    if isinstance(row, Mapping)
                ]
                != list(expected_regions)
            ):
                raise HiLiftPackageAssemblyError(
                    f"native regional {domain}.{field_id} structure differs"
                )
            checked_fields += 1
            regions = []
            for position, region_id in enumerate(expected_regions):
                raw_region = aggregate_field["regions"][position]
                pooled = raw_region["pooled_metrics"]
                case_region_documents = [
                    report["fields"][field_id]["regions"][position]
                    for report in case_reports
                ]
                case_global_documents = [
                    report["fields"][field_id]["global"]
                    for report in case_reports
                ]
                stable_case_values = [
                    _whole_support_normalized_rmse_percent(
                        case_region["sufficient_statistics"],
                        case_global["sufficient_statistics"],
                        label=(
                            f"regional {case_id}/{domain}/{field_id}/{region_id}"
                        ),
                    )
                    for case_id, case_region, case_global in zip(
                        case_ids,
                        case_region_documents,
                        case_global_documents,
                        strict=True,
                    )
                ]
                local_l2_case_values = [
                    _optional_regional_metric(
                        case_region,
                        "relative_l2_percent",
                        f"regional {case_id}/{domain}/{field_id}/{region_id} relative L2",
                    )
                    for case_id, case_region in zip(
                        case_ids, case_region_documents, strict=True
                    )
                ]
                r2_case_values = [
                    _optional_regional_metric(
                        case_region,
                        "r2",
                        f"regional {case_id}/{domain}/{field_id}/{region_id} R2",
                    )
                    for case_id, case_region in zip(
                        case_ids, case_region_documents, strict=True
                    )
                ]
                mae_case_values = [
                    _optional_regional_metric(
                        case_region,
                        "mae",
                        f"regional {case_id}/{domain}/{field_id}/{region_id} MAE",
                    )
                    for case_id, case_region in zip(
                        case_ids, case_region_documents, strict=True
                    )
                ]
                rmse_case_values = [
                    _optional_regional_metric(
                        case_region,
                        "rmse",
                        f"regional {case_id}/{domain}/{field_id}/{region_id} RMSE",
                    )
                    for case_id, case_region in zip(
                        case_ids, case_region_documents, strict=True
                    )
                ]
                regions.append(
                    {
                        "region_id": region_id,
                        "entity_fraction": raw_region.get("spatial_point_fraction"),
                        "weight_fraction": raw_region.get("weight_fraction"),
                        "squared_error_fraction": raw_region.get("squared_error_fraction"),
                        "whole_support_normalized_rmse_percent": (
                            _whole_support_normalized_rmse_percent(
                                raw_region["sufficient_statistics"],
                                aggregate_field["global"]["sufficient_statistics"],
                                label=f"regional pooled {domain}/{field_id}/{region_id}",
                            )
                        ),
                        "relative_l2_percent": pooled.get("relative_l2_percent"),
                        "r2": pooled.get("r2"),
                        "r2_status": pooled.get("r2_status"),
                        "mae": pooled.get("mae"),
                        "rmse": pooled.get("rmse"),
                        "case_macro": {
                            "whole_support_normalized_rmse_percent": (
                                _regional_case_macro(stable_case_values)
                            ),
                            "relative_l2_percent": _regional_case_macro(
                                local_l2_case_values
                            ),
                            "r2": _regional_case_macro(r2_case_values),
                            "mae": _regional_case_macro(mae_case_values),
                            "rmse": _regional_case_macro(rmse_case_values),
                        },
                        "case_distribution": {
                            "whole_support_normalized_rmse_percent": (
                                _regional_case_distribution(stable_case_values)
                            ),
                            "relative_l2_percent": _regional_case_distribution(
                                local_l2_case_values
                            ),
                            "r2": _regional_case_distribution(r2_case_values),
                        },
                    }
                )
            case_global = [
                report["fields"][field_id]["global"]["metrics"] for report in case_reports
            ]
            macro = {
                metric_id: math.fsum(
                    _finite(row[metric_id], f"regional {domain}.{field_id}.{metric_id}")
                    for row in case_global
                )
                / len(case_global)
                for metric_id in ("relative_l2_percent", "mae", "rmse")
            }
            distribution = _summary(
                [
                    _finite(
                        row["relative_l2_percent"],
                        f"regional {domain}.{field_id} relative L2",
                    )
                    for row in case_global
                ]
            )
            pooled_global = aggregate_field["global"]["pooled_metrics"]
            fields[field_id] = {
                "global": dict(pooled_global),
                "regions": regions,
                "pooled": dict(pooled_global),
                "macro": macro,
                "case_distribution": distribution,
            }
        domains[domain] = {
            "support_id": support_id,
            "region_order": list(expected_regions),
            "fields": fields,
        }
    report = {
        "schema": AGGREGATE_REGIONAL_REPORT_SCHEMA,
        "schema_version": 2,
        "status": "complete_report_only",
        "definition_id": REGIONAL_DEFINITION_ID,
        "contract_sha256": REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
        "dataset_id": "hiliftaeroml",
        "split_id": split_id,
        "prediction_scope": "surface_and_volume",
        "case_count": len(case_ids),
        "case_ids": list(case_ids),
        "scoring": {
            "role": "report_only",
            "weight": 0.0,
            "official_metric_inputs_changed": False,
            "official_score_changed": False,
        },
        "surface": domains["surface"],
        "volume": domains["volume"],
        "reconstruction": {
            "status": "pass",
            "relative_tolerance": 5e-12,
            "absolute_tolerance": 1e-12,
            "checked_case_count": len(case_ids),
            "checked_field_count": checked_fields,
        },
    }
    try:
        validate_aggregate_regional_diagnostics(
            report, expected_case_ids=case_ids, expected_split_id=split_id
        )
    except HiLiftRegionalAggregateError as error:
        raise HiLiftPackageAssemblyError(
            f"adapted regional report is invalid: {error}"
        ) from error
    return report


def _prepare_inference_discretization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HiLiftPackageAssemblyError(
            "config.spatial_discretization.inference must be an object"
        )
    inference = copy.deepcopy(value)
    for input_name, default_record_id in INFERENCE_INPUT_CASE_RECORD_IDS.items():
        representation = inference.get(input_name)
        if not isinstance(representation, dict):
            raise HiLiftPackageAssemblyError(
                f"config.spatial_discretization.inference.{input_name} must be an object"
            )
        if representation.get("used") is not True:
            continue
        representation.setdefault("case_record_id", default_record_id)
        case_record_id = representation.get("case_record_id")
        if not isinstance(case_record_id, str) or not case_record_id:
            raise HiLiftPackageAssemblyError(
                "config.spatial_discretization.inference."
                f"{input_name}.case_record_id must be a non-empty string when used"
            )
    return inference


def _case_representation(
    *,
    record_id: str,
    summary: Mapping[str, Any],
    support_count: int,
    label: str,
) -> dict[str, Any]:
    entities = [
        entry.get("entity")
        for entry in summary.get("entity_counts", [])
        if isinstance(entry, dict)
    ]
    if entities != ["points"]:
        raise HiLiftPackageAssemblyError(
            f"{label} must declare exactly the points entity"
        )
    result: dict[str, Any] = {
        "id": record_id,
        "entity_counts": [{"entity": "points", "count": support_count}],
    }
    native = summary.get("native_comparison")
    if isinstance(native, dict) and native.get("status") == "reported":
        native_entities = [
            entry.get("entity")
            for entry in native.get("native_entity_counts", [])
            if isinstance(entry, dict)
        ]
        fraction_entities = [
            entry.get("entity")
            for entry in native.get("fractions", [])
            if isinstance(entry, dict)
        ]
        if native_entities != entities or fraction_entities != entities:
            raise HiLiftPackageAssemblyError(
                f"{label} native entity IDs must match entity_counts in declared order"
            )
        result["native_entity_counts"] = [
            {"entity": "points", "count": support_count}
        ]
        result["native_fractions"] = [{"entity": "points", "fraction": 1.0}]
    domain = summary.get("domain")
    if isinstance(domain, dict):
        result["domain"] = copy.deepcopy(domain)
    return result


def _case_discretization_records(
    *,
    submission_id: str,
    split_id: str,
    case_ids: Sequence[str],
    case_metrics: Mapping[str, Any],
    inference_summary: Mapping[str, Any],
) -> bytes:
    by_case = {case["case_id"]: case for case in case_metrics["cases"]}
    lines: list[bytes] = []
    for case_id in case_ids:
        counts = {
            support["support_id"]: support["support_count"]
            for support in by_case[case_id]["supports"]
        }
        inputs = []
        for input_name, support_id in INFERENCE_INPUT_SUPPORT_IDS.items():
            representation = inference_summary.get(input_name)
            if not isinstance(representation, dict):
                raise HiLiftPackageAssemblyError(
                    f"inference summary {input_name} must be an object"
                )
            if representation.get("used") is not True:
                continue
            inputs.append(
                _case_representation(
                    record_id=representation["case_record_id"],
                    summary=representation,
                    support_count=counts[support_id],
                    label=f"inference.{input_name}",
                )
            )
        direct_outputs = []
        for output in inference_summary.get("direct_outputs", []):
            if not isinstance(output, dict):
                raise HiLiftPackageAssemblyError(
                    "inference summary direct_outputs entries must be objects"
                )
            domain = output.get("domain")
            support_id = INFERENCE_OUTPUT_SUPPORT_IDS.get(domain)
            if support_id is None:
                raise HiLiftPackageAssemblyError(
                    f"inference direct output {output.get('id')!r} has unsupported "
                    f"domain {domain!r}"
                )
            representation = output.get("representation")
            if not isinstance(representation, dict):
                raise HiLiftPackageAssemblyError(
                    f"inference direct output {output.get('id')!r} representation must be an object"
                )
            direct_outputs.append(
                _case_representation(
                    record_id=output["id"],
                    summary=representation,
                    support_count=counts[support_id],
                    label=f"inference.direct_outputs[{output.get('id')}]",
                )
            )
        mappings = []
        for mapping in inference_summary.get("mappings", []):
            if not isinstance(mapping, dict):
                raise HiLiftPackageAssemblyError(
                    "inference summary mappings entries must be objects"
                )
            support_id = mapping["support_id"]
            support_count = counts[support_id]
            mappings.append(
                {
                    "support_id": support_id,
                    "source_output_id": mapping["source_output_id"],
                    "support_count": support_count,
                    "scored_count": support_count,
                    "unmapped_count": 0,
                    "extrapolated_count": 0,
                    "final_coverage_fraction": 1.0,
                }
            )
        document = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
            "schema_version": "1.0",
            "submission_id": submission_id,
            "dataset_id": "hiliftaeroml",
            "split_id": split_id,
            "case_id": case_id,
            "inference": {
                "inputs": inputs,
                "direct_outputs": direct_outputs,
                "mappings": mappings,
            },
        }
        _require_schema(
            document,
            "v3/discretization-case.schema.json",
            f"discretization case {case_id}",
        )
        lines.append(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        )
    return b"\n".join(lines) + b"\n"


def _participant_submission(
    *, config: Mapping[str, Any], case_count: int, methodology_contract: Mapping[str, Any]
) -> dict[str, Any]:
    participant = config.get("participant")
    if not isinstance(participant, dict):
        raise HiLiftPackageAssemblyError("config.participant must be an object")
    forbidden = {
        "dataset",
        "dataset_id",
        "dataset_version",
        "split",
        "split_id",
        "case_set_id",
        "split_sha256",
        "evaluation",
        "scoring_support",
        "spatial_discretization",
        "case_metrics",
        "metric_values",
        "profile_data",
        "regional_diagnostics",
        "approval",
        "prediction_artifacts",
        "parameter_count_millions",
    }
    overlap = sorted(forbidden.intersection(participant))
    if overlap:
        raise HiLiftPackageAssemblyError(
            f"participant config must not override benchmark fields: {overlap}"
        )
    submission = copy.deepcopy(participant)
    submission["dataset_id"] = "hiliftaeroml"
    submission["prediction_scope"] = "surface_and_volume"
    try:
        submission["parameter_count_millions"] = derived_parameter_count_millions(
            participant.get("methodology")
        )
        _require_methodology_schema(participant.get("methodology"))
        require_methodology(
            submission, expected_case_count=case_count, contract=dict(methodology_contract)
        )
    except MethodologyError as error:
        raise HiLiftPackageAssemblyError(
            f"participant methodology is invalid: {error}"
        ) from error
    return submission


def _validate_config_envelope(config: Mapping[str, Any]) -> tuple[int, bool]:
    required = {
        "schema",
        "split_id",
        "participant",
        "evaluation",
        "spatial_discretization",
        "release_bindings",
    }
    optional = {
        "profile_cases_per_chunk",
        "include_regional_diagnostics",
        "compact_evaluation",
    }
    if not required.issubset(config) or not set(config).issubset(required | optional):
        raise HiLiftPackageAssemblyError(
            "package config keys differ; "
            f"missing={sorted(required-set(config))}, "
            f"unexpected={sorted(set(config)-(required|optional))}"
        )
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "command",
        "generated_at",
    }:
        raise HiLiftPackageAssemblyError(
            "config.evaluation keys must be command and generated_at"
        )
    compact_evaluation = config.get("compact_evaluation")
    if compact_evaluation is not None and (
        not isinstance(compact_evaluation, dict)
        or set(compact_evaluation) != {"command", "generated_at"}
    ):
        raise HiLiftPackageAssemblyError(
            "config.compact_evaluation keys must be command and generated_at"
        )
    spatial = config.get("spatial_discretization")
    if not isinstance(spatial, dict) or not {"training", "inference"}.issubset(
        spatial
    ) or not set(spatial).issubset({"training", "inference", "notes"}):
        raise HiLiftPackageAssemblyError(
            "config.spatial_discretization keys differ"
        )
    cases_per_chunk = config.get("profile_cases_per_chunk", 10)
    if (
        not isinstance(cases_per_chunk, int)
        or isinstance(cases_per_chunk, bool)
        or cases_per_chunk < 1
    ):
        raise HiLiftPackageAssemblyError(
            "config.profile_cases_per_chunk must be a positive integer"
        )
    include_regional = config.get("include_regional_diagnostics", True)
    if not isinstance(include_regional, bool):
        raise HiLiftPackageAssemblyError(
            "config.include_regional_diagnostics must be boolean"
        )
    return cases_per_chunk, include_regional


def _selected_evaluation(
    config: Mapping[str, Any], *, compact_profile_mode: bool
) -> Mapping[str, Any]:
    key = "compact_evaluation" if compact_profile_mode else "evaluation"
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise HiLiftPackageAssemblyError(f"config.{key} must be an object")
    return value


def _profile_evidence_notes(
    *, evaluator_revision: str, compact_profile_mode: bool
) -> str:
    if compact_profile_mode:
        return (
            "The frozen dataset-evaluator revision "
            f"{evaluator_revision} applies only to the native-v1 base field, "
            "force, and noncompact scoring implementation. The additive "
            "compact-v2 profile implementation is an unbound worktree "
            "candidate with no code revision or implementation-manifest "
            f"SHA-256; its profile topology contract is "
            f"{COMPACT_PROFILE_CONTRACT_SHA256}. Cp/velocity profile R2 was "
            "recomputed from prediction-only chunks against the explicitly "
            "supplied inactive local candidate evaluator-support release; "
            "this does not publish or activate profile intake."
        )
    return (
        "Evaluator identity is repository-frozen at "
        f"{evaluator_revision}; profile topology contract is "
        f"{PROFILE_CONTRACT_SHA256}. Cp/velocity profile R2 was recomputed "
        "from prediction-only chunks against the explicitly supplied "
        "inactive local candidate truth release; this does not publish or "
        "activate profile intake."
    )


def _open_or_validate_compact_support_release(
    *,
    release_root: Path,
    declaration: Mapping[str, Any],
    case_ids: Sequence[str],
    case_set_id: str,
    opened_release: CompactSupportRelease | None,
) -> CompactSupportRelease:
    """Return an exact compact-support handle for one package assembly.

    A batch invocation can provide a handle which was fully streamed and
    verified immediately before the two independent package assemblies.  The
    handle is still checked against this assembly's path, manifest, ordered
    cases, and case-set identity, so it cannot cross-bind a different split or
    support release.
    """

    expected_manifest = declaration["manifest_sha256"]
    if opened_release is None:
        try:
            return open_compact_support_release(
                release_root=release_root,
                expected_manifest_sha256=expected_manifest,
                expected_case_ids=case_ids,
                case_set_id=case_set_id,
            )
        except CompactProfileEvaluationError as error:
            raise HiLiftPackageAssemblyError(
                f"local compact evaluator-support release is invalid: {error}"
            ) from error
    if not isinstance(opened_release, CompactSupportRelease):
        raise HiLiftPackageAssemblyError("preopened compact support handle is invalid")
    try:
        expected_root = release_root.resolve(strict=True)
    except OSError as error:
        raise HiLiftPackageAssemblyError(
            f"cannot resolve compact evaluator-support release: {error}"
        ) from error
    if (
        opened_release.release_root != expected_root
        or opened_release.release_id != declaration["release_id"]
        or opened_release.manifest_sha256 != expected_manifest
        or opened_release.case_set_id != case_set_id
        or opened_release.case_ids != tuple(case_ids)
    ):
        raise HiLiftPackageAssemblyError(
            "preopened compact support handle differs from this package binding"
        )
    return opened_release


def assemble_package(
    *,
    config_path: Path,
    specification_path: Path,
    native_aggregate_root: Path,
    native_outputs_root: Path | None,
    native_receipts_root: Path,
    candidate_profile_truth_release_root: Path | None,
    output_path: Path,
    native_surface_outputs_root: Path | None = None,
    native_volume_outputs_root: Path | None = None,
    candidate_compact_profile_support_release_root: Path | None = None,
    opened_compact_support_release: CompactSupportRelease | None = None,
) -> dict[str, Any]:
    config = load_json(config_path, label="package config")
    if config.get("schema") != CONFIG_SCHEMA:
        raise HiLiftPackageAssemblyError(f"config.schema must equal {CONFIG_SCHEMA!r}")
    compact_profile_mode = (
        candidate_compact_profile_support_release_root is not None
    )
    if compact_profile_mode == (candidate_profile_truth_release_root is not None):
        raise HiLiftPackageAssemblyError(
            "assembly requires exactly one local profile release: native-v1 truth "
            "or compact-v2 evaluator support"
        )
    if opened_compact_support_release is not None and not compact_profile_mode:
        raise HiLiftPackageAssemblyError(
            "a preopened compact support handle requires compact-v2 evaluator support"
        )
    profile_cases_per_chunk, include_regional_diagnostics = _validate_config_envelope(
        config
    )
    blockers = unresolved_tokens(config)
    if blockers:
        raise HiLiftPackageAssemblyError(
            "configuration contains unresolved owner or participant tokens: "
            + ", ".join(item["path"] for item in blockers)
        )
    if output_path.exists() or output_path.is_symlink():
        raise HiLiftPackageAssemblyError(f"output already exists: {output_path}")
    output_roots = _resolve_native_output_roots(
        outputs_root=native_outputs_root,
        surface_outputs_root=native_surface_outputs_root,
        volume_outputs_root=native_volume_outputs_root,
    )
    specification = load_json(specification_path, label="HiLiftAeroML specification")
    if specification.get("dataset_id") != "hiliftaeroml":
        raise HiLiftPackageAssemblyError("assembler accepts only HiLiftAeroML")
    if specification.get("submission_format") != SUBMISSION_FORMAT:
        raise HiLiftPackageAssemblyError(
            f"HiLiftAeroML specification submission_format must equal {SUBMISSION_FORMAT!r}"
        )
    split_id = config.get("split_id")
    if not isinstance(split_id, str):
        raise HiLiftPackageAssemblyError("config.split_id must be a string")
    split, case_ids, split_index = _find_split(
        specification, specification_path, split_id
    )
    candidate, evaluator, profile_truth = _release_bindings(
        config, specification, specification_path
    )
    compact_support_declaration: dict[str, Any] | None = None
    compact_support_release = None
    if compact_profile_mode:
        _, compact_support_declaration = _compact_profile_declaration(
            specification, specification_path
        )
        assert candidate_compact_profile_support_release_root is not None
        compact_support_release = _open_or_validate_compact_support_release(
            release_root=candidate_compact_profile_support_release_root,
            declaration=compact_support_declaration,
            case_ids=case_ids,
            case_set_id=split["case_set_id"],
            opened_release=opened_compact_support_release,
        )
        if (
            compact_support_release.release_id
            != compact_support_declaration["release_id"]
            or compact_support_release.manifest_sha256
            != compact_support_declaration["manifest_sha256"]
            or compact_support_release.source_profile_truth_release_id
            != profile_truth["release_id"]
            or compact_support_release.source_profile_truth_manifest_sha256
            != profile_truth["manifest_sha256"]
        ):
            raise HiLiftPackageAssemblyError(
                "compact evaluator support differs from its contract or source-truth binding"
            )
    else:
        assert candidate_profile_truth_release_root is not None
        try:
            open_candidate_truth_release(
                release_root=candidate_profile_truth_release_root,
                candidate_declaration=profile_truth,
                expected_case_ids=case_ids,
                case_set_id=split["case_set_id"],
            )
        except NativeProfileEvaluationError as error:
            raise HiLiftPackageAssemblyError(
                f"local candidate profile-ground-truth release is invalid: {error}"
            ) from error
    selected_profile_format = (
        COMPACT_PROFILE_FORMAT if compact_profile_mode else PROFILE_FORMAT
    )
    methodology_contract = load_json(
        specification_path.parent / "methodology-contract.json",
        label="HiLiftAeroML methodology contract",
    )
    participant = _participant_submission(
        config=config,
        case_count=len(case_ids),
        methodology_contract=methodology_contract,
    )
    native_documents = _verify_native_aggregate(
        aggregate_root=native_aggregate_root,
        case_ids=case_ids,
        case_set_id=split["case_set_id"],
        case_set_sha256=split_index["case_set_sha256"],
    )
    native_case_records = _native_case_records(native_documents)
    profile_artifact_sha256 = _verify_native_receipts(
        receipts_root=native_receipts_root,
        outputs_root=None,
        surface_outputs_root=output_roots["surface"],
        volume_outputs_root=output_roots["volume"],
        case_ids=case_ids,
        case_set_id=split["case_set_id"],
        case_set_sha256=split_index["case_set_sha256"],
        case_records=native_case_records,
    )
    required_metric_ids = {
        metric["id"]
        for metric in specification.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("id"), str)
    }
    if required_metric_ids.intersection(FORCE_METRIC_IDS):
        # Run before profile serialization so any selected-case-set load gate
        # is reported without producing gigabytes of partial profile artifacts.
        _load_complete_loads(
            case_ids=case_ids,
            outputs_root=output_roots["surface"],
            case_records=native_case_records,
        )
    manifest_path = _safe_child(
        specification_path.parent,
        specification["scoring_support"]["candidate_manifest"]["manifest_file"],
        "candidate manifest_file",
    )
    try:
        support_release = load_support_release(manifest_path, split["case_set_id"])
    except ScoringSupportError as error:
        raise HiLiftPackageAssemblyError(f"cannot load scoring support: {error}") from error
    if list(support_release.cases) != case_ids:
        raise HiLiftPackageAssemblyError("scoring support case order differs from the split")

    evaluation = _selected_evaluation(
        config, compact_profile_mode=compact_profile_mode
    )
    spatial = config.get("spatial_discretization")
    if not isinstance(spatial, dict):
        raise HiLiftPackageAssemblyError(
            "config spatial_discretization must be an object"
        )
    command = evaluation.get("command")
    generated_at = evaluation.get("generated_at")
    if not isinstance(command, str) or not command or not isinstance(generated_at, str):
        raise HiLiftPackageAssemblyError("evaluation command/generated_at are required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    package_size_bytes: int | None = None
    try:
        if compact_profile_mode:
            assert candidate_compact_profile_support_release_root is not None
            assert compact_support_declaration is not None
            try:
                profile_index_sha, _compact_profile_metrics = (
                    build_compact_profile_directory(
                        submission_id=participant["submission_id"],
                        split_id=split_id,
                        case_set_id=split["case_set_id"],
                        case_ids=case_ids,
                        support_release_root=(
                            candidate_compact_profile_support_release_root
                        ),
                        support_manifest_sha256=compact_support_declaration[
                            "manifest_sha256"
                        ],
                        outputs_root=None,
                        surface_outputs_root=output_roots["surface"],
                        volume_outputs_root=output_roots["volume"],
                        profiles_root=staging / "profiles",
                        cases_per_chunk=profile_cases_per_chunk,
                        expected_case_artifact_sha256=profile_artifact_sha256,
                        opened_support_release=compact_support_release,
                    )
                )
                profile_metrics = score_compact_profile_directory(
                    profiles_root=staging / "profiles",
                    support_release_root=(
                        candidate_compact_profile_support_release_root
                    ),
                    support_manifest_sha256=compact_support_declaration[
                        "manifest_sha256"
                    ],
                    submission_id=participant["submission_id"],
                    split_id=split_id,
                    case_set_id=split["case_set_id"],
                    expected_case_ids=case_ids,
                    opened_support_release=compact_support_release,
                )
            except CompactProfileEvaluationError as error:
                raise HiLiftPackageAssemblyError(
                    "compact profile serialization/scoring failed: " + str(error)
                ) from error
        else:
            assert candidate_profile_truth_release_root is not None
            try:
                profile_index_sha, _native_profile_metrics = build_profile_directory(
                    submission_id=participant["submission_id"],
                    split_id=split_id,
                    case_set_id=split["case_set_id"],
                    case_ids=case_ids,
                    outputs_root=None,
                    surface_outputs_root=output_roots["surface"],
                    volume_outputs_root=output_roots["volume"],
                    profiles_root=staging / "profiles",
                    cases_per_chunk=profile_cases_per_chunk,
                    expected_case_artifact_sha256=profile_artifact_sha256,
                )
            except NativeProfileError as error:
                raise HiLiftPackageAssemblyError(
                    f"native profile serialization failed: {error}"
                ) from error
            try:
                profile_metrics = score_native_profile_directory(
                    profiles_root=staging / "profiles",
                    release_root=candidate_profile_truth_release_root,
                    candidate_declaration=profile_truth,
                    submission_id=participant["submission_id"],
                    split_id=split_id,
                    case_set_id=split["case_set_id"],
                    expected_case_ids=case_ids,
                )
            except NativeProfileEvaluationError as error:
                raise HiLiftPackageAssemblyError(
                    "hidden-truth native profile scoring failed: " + str(error)
                ) from error
        case_metrics, metric_values = _case_metrics_and_values(
            submission_id=participant["submission_id"],
            split_id=split_id,
            case_set_id=split["case_set_id"],
            case_ids=case_ids,
            support_release=support_release,
            native_documents=native_documents,
            surface_outputs_root=output_roots["surface"],
            volume_outputs_root=output_roots["volume"],
            profile_metrics=profile_metrics,
            specification=specification,
            support_release_id=candidate["release_id"],
            support_manifest_sha256=candidate["manifest_sha256"],
            generated_at=generated_at,
        )
        case_metrics_sha = write_json(staging / "metrics" / "cases.json", case_metrics)

        inference_discretization = _prepare_inference_discretization(
            spatial.get("inference")
        )
        case_lines = _case_discretization_records(
            submission_id=participant["submission_id"],
            split_id=split_id,
            case_ids=case_ids,
            case_metrics=case_metrics,
            inference_summary=inference_discretization,
        )
        cases_path = staging / "discretization" / "cases.jsonl"
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        cases_path.write_bytes(case_lines)
        cases_sha = sha256_file(cases_path)
        discretization = {
            "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
            "schema_version": "1.0",
            "submission_id": participant["submission_id"],
            "dataset_id": "hiliftaeroml",
            "split_id": split_id,
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "training": copy.deepcopy(spatial.get("training")),
            "inference": inference_discretization,
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
        discretization_sha = write_json(staging / "discretization.json", discretization)

        regional_sha: str | None = None
        regional_declaration: dict[str, Any] | None = None
        if include_regional_diagnostics:
            regional = _convert_regional(
                native=native_documents["regional_diagnostics_aggregate.json"],
                case_ids=case_ids,
                split_id=split_id,
                surface_outputs_root=output_roots["surface"],
                volume_outputs_root=output_roots["volume"],
                case_records=native_case_records,
            )
            regional_sha = write_json(staging / "regional-diagnostics.json", regional)
            regional_declaration = {
                "format": AGGREGATE_REGIONAL_REPORT_SCHEMA,
                "file": "regional-diagnostics.json",
                "sha256": regional_sha,
                "contract_sha256": REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
                "definition_id": REGIONAL_DEFINITION_ID,
                "case_count": len(case_ids),
                "role": "report_only",
                "weight": 0.0,
                "official_score_changed": False,
            }
        reproducibility = participant.get("reproducibility")
        code = reproducibility.get("code") if isinstance(reproducibility, dict) else None
        participant_revision = code.get("commit") if isinstance(code, dict) else None
        # Preserve native-v1 wording byte-for-byte while giving compact-v2 its
        # evaluator-support terminology and contract identity.
        evidence_notes = _profile_evidence_notes(
            evaluator_revision=evaluator["code_revision"],
            compact_profile_mode=compact_profile_mode,
        )
        evidence = {
            "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
            "schema_version": "3.0",
            "submission_id": participant["submission_id"],
            "dataset_id": "hiliftaeroml",
            "dataset_version": specification["dataset_version"],
            "split_id": split_id,
            "split_sha256": split["sha256"],
            "case_set_id": split["case_set_id"],
            "prediction_scope": "surface_and_volume",
            "reference_version": evaluator["reference_version"],
            "dataset_evaluator_binding": _dataset_evaluator_evidence(evaluator),
            "command": command,
            "generated_at": generated_at,
            "status": "submitted_evaluation",
            "metric_values": metric_values,
            "profile_index_sha256": profile_index_sha,
            "profile_ground_truth_release_id": profile_truth["release_id"],
            "profile_ground_truth_manifest_sha256": profile_truth["manifest_sha256"],
            "scoring_support_release_id": candidate["release_id"],
            "scoring_support_manifest_sha256": candidate["manifest_sha256"],
            "discretization_sha256": discretization_sha,
            "case_metrics_sha256": case_metrics_sha,
            "notes": evidence_notes,
        }
        if regional_sha is not None:
            evidence["regional_diagnostics_sha256"] = regional_sha
        if compact_profile_mode:
            evidence["compact_profile_implementation_binding"] = dict(
                COMPACT_PROFILE_IMPLEMENTATION_BINDING
            )
        if participant_revision is not None:
            evidence["code_revision"] = participant_revision
        _require_schema(evidence, "v3/evaluation-evidence.schema.json", "evaluation-evidence.json")
        evidence_sha = write_json(staging / "evaluation-evidence.json", evidence)
        submission = {
            "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
            "schema_version": "3.0",
            **participant,
            "dataset": specification["dataset_name"],
            "dataset_id": "hiliftaeroml",
            "dataset_version": specification["dataset_version"],
            "split": split["label"],
            "split_id": split_id,
            "case_set_id": split["case_set_id"],
            "split_sha256": split["sha256"],
            "prediction_scope": "surface_and_volume",
            "evaluation": {
                "reference_version": evaluator["reference_version"],
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
                "format": selected_profile_format,
                "index_file": "profiles/index.json",
                "case_count": len(case_ids),
                "case_set_id": split["case_set_id"],
                "profile_ground_truth_release_id": profile_truth["release_id"],
                "profile_ground_truth_manifest_sha256": profile_truth["manifest_sha256"],
            },
        }
        if compact_profile_mode:
            assert compact_support_release is not None
            submission["profile_data"].update(
                {
                    "evaluator_support_release_id": (
                        compact_support_release.release_id
                    ),
                    "evaluator_support_manifest_sha256": (
                        compact_support_release.manifest_sha256
                    ),
                    "compact_profile_implementation_binding": dict(
                        COMPACT_PROFILE_IMPLEMENTATION_BINDING
                    ),
                }
            )
        if regional_declaration is not None:
            submission["regional_diagnostics"] = regional_declaration
        if participant_revision is not None:
            submission["evaluation"]["code_revision"] = participant_revision
        _require_schema(submission, "v3/submission.schema.json", "submission.json")
        write_json(staging / "submission.json", submission)
        if compact_profile_mode:
            package_size_bytes = sum(
                path.stat().st_size
                for path in staging.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        os.replace(staging, output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "candidate_package_assembled_not_approved",
        "output": str(output_path.resolve()),
        "submission_id": participant["submission_id"],
        "split_id": split_id,
        "case_set_id": split["case_set_id"],
        "case_count": len(case_ids),
        "profile_format": selected_profile_format,
        "package_size_bytes": package_size_bytes,
        "compact_package_max_bytes": None,
        "candidate_dry_run_command": (
            "python scripts/validate_submission.py --candidate-dry-run "
            + (
                "--candidate-compact-profile-support-release "
                f"{candidate_compact_profile_support_release_root} "
                if compact_profile_mode
                else "--candidate-profile-truth-release "
                f"{candidate_profile_truth_release_root} "
            )
            + str(output_path)
        ),
    }


def inspect_blockers(
    *,
    config_path: Path,
    specification_path: Path,
    native_aggregate_root: Path | None,
    native_outputs_root: Path | None,
    native_receipts_root: Path | None,
    candidate_profile_truth_release_root: Path | None = None,
    candidate_compact_profile_support_release_root: Path | None = None,
    native_surface_outputs_root: Path | None = None,
    native_volume_outputs_root: Path | None = None,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    if (
        candidate_profile_truth_release_root is not None
        and candidate_compact_profile_support_release_root is not None
    ):
        blockers.append(
            {
                "gate": "profile_release_selection",
                "detail": (
                    "native-v1 truth and compact-v2 evaluator-support paths are "
                    "mutually exclusive"
                ),
            }
        )
    config = load_json(config_path, label="package config")
    for token in unresolved_tokens(config):
        blockers.append({"gate": "configuration_token", **token})
    specification = load_json(specification_path, label="HiLiftAeroML specification")
    evaluator = specification.get("scoring_support", {}).get("dataset_evaluator_binding", {})
    if evaluator.get("status") != "frozen":
        blockers.append(
            {
                "gate": "frozen_evaluator_revision",
                "status": evaluator.get("status"),
                "detail": "repository evaluator_code_revision is not frozen",
            }
        )
    profile = specification.get("profile_definition")
    if not isinstance(profile, dict):
        blockers.append(
            {
                "gate": "profile_contract_binding",
                "detail": "submission specification does not bind the native profile contract",
            }
        )
    candidate_profile_truth: dict[str, Any] | None = None
    if isinstance(profile, dict):
        public_truth = profile.get("profile_ground_truth")
        candidate_value = profile.get("candidate_dry_run_profile_ground_truth")
        if (
            public_truth
            != {
                "status": "not_published",
                "release_id": None,
                "manifest_sha256": None,
            }
            or not isinstance(candidate_value, dict)
            or candidate_value.get("status") != CANDIDATE_TRUTH_STATUS
            or candidate_value.get("usage") != CANDIDATE_USAGE
        ):
            blockers.append(
                {
                    "gate": "profile_ground_truth_release",
                    "detail": "local candidate/public profile-truth boundary differs",
                }
            )
        else:
            candidate_profile_truth = candidate_value
            if (
                candidate_profile_truth_release_root is None
                and candidate_compact_profile_support_release_root is None
            ):
                blockers.append(
                    {
                        "gate": "profile_ground_truth_release",
                        "detail": (
                            "inactive candidate truth is bound but its local release "
                            "path was not supplied; public profile intake remains closed"
                        ),
                    }
                )
    split_id = config.get("split_id")
    if isinstance(split_id, str):
        try:
            split, case_ids, split_index = _find_split(
                specification, specification_path, split_id
            )
        except HiLiftPackageAssemblyError as error:
            blockers.append({"gate": "split", "detail": str(error)})
        else:
            if (
                candidate_profile_truth is not None
                and candidate_profile_truth_release_root is not None
            ):
                try:
                    open_candidate_truth_release(
                        release_root=candidate_profile_truth_release_root,
                        candidate_declaration=candidate_profile_truth,
                        expected_case_ids=case_ids,
                        case_set_id=split["case_set_id"],
                    )
                except NativeProfileEvaluationError as error:
                    blockers.append(
                        {
                            "gate": "profile_ground_truth_release",
                            "detail": str(error),
                        }
                    )
            if candidate_compact_profile_support_release_root is not None:
                try:
                    _, compact_declaration = _compact_profile_declaration(
                        specification, specification_path
                    )
                    release = open_compact_support_release(
                        release_root=(
                            candidate_compact_profile_support_release_root
                        ),
                        expected_manifest_sha256=compact_declaration[
                            "manifest_sha256"
                        ],
                        expected_case_ids=case_ids,
                        case_set_id=split["case_set_id"],
                    )
                    if (
                        candidate_profile_truth is None
                        or release.release_id != compact_declaration["release_id"]
                        or release.source_profile_truth_release_id
                        != candidate_profile_truth.get("release_id")
                        or release.source_profile_truth_manifest_sha256
                        != candidate_profile_truth.get("manifest_sha256")
                    ):
                        raise HiLiftPackageAssemblyError(
                            "compact support source-truth binding differs"
                        )
                except (HiLiftPackageAssemblyError, CompactProfileEvaluationError) as error:
                    blockers.append(
                        {
                            "gate": "compact_profile_support_release",
                            "detail": str(error),
                        }
                    )
            native_case_records: dict[str, dict[str, Any]] | None = None
            if native_aggregate_root is not None:
                try:
                    native_documents = _verify_native_aggregate(
                        aggregate_root=native_aggregate_root,
                        case_ids=case_ids,
                        case_set_id=split["case_set_id"],
                        case_set_sha256=split_index["case_set_sha256"],
                    )
                    native_case_records = _native_case_records(native_documents)
                except HiLiftPackageAssemblyError as error:
                    blockers.append({"gate": "native_aggregate", "detail": str(error)})
            output_argument_supplied = any(
                root is not None
                for root in (
                    native_outputs_root,
                    native_surface_outputs_root,
                    native_volume_outputs_root,
                )
            )
            if output_argument_supplied or native_receipts_root is not None:
                if native_case_records is None:
                    blockers.append(
                        {
                            "gate": "native_output_provenance",
                            "detail": (
                                "native aggregate is required to hash-bind live "
                                "per-case outputs"
                            ),
                        }
                    )
                elif not output_argument_supplied or native_receipts_root is None:
                    blockers.append(
                        {
                            "gate": "native_output_provenance",
                            "detail": (
                                "both native outputs and native receipts are required "
                                "to verify per-case provenance"
                            ),
                        }
                    )
                else:
                    try:
                        output_roots = _resolve_native_output_roots(
                            outputs_root=native_outputs_root,
                            surface_outputs_root=native_surface_outputs_root,
                            volume_outputs_root=native_volume_outputs_root,
                        )
                        _verify_native_receipts(
                            receipts_root=native_receipts_root,
                            outputs_root=None,
                            surface_outputs_root=output_roots["surface"],
                            volume_outputs_root=output_roots["volume"],
                            case_ids=case_ids,
                            case_set_id=split["case_set_id"],
                            case_set_sha256=split_index["case_set_sha256"],
                            case_records=native_case_records,
                        )
                        _load_complete_loads(
                            case_ids=case_ids,
                            outputs_root=output_roots["surface"],
                            case_records=native_case_records,
                        )
                    except HiLiftPackageAssemblyError as error:
                        blockers.append(
                            {
                                "gate": (
                                    "force_and_overall"
                                    if str(error).startswith("force/overall metrics unavailable:")
                                    else "native_output_provenance"
                                ),
                                "detail": str(error),
                            }
                        )
    return {
        "status": "blocked" if blockers else "ready_for_full_assembly_validation",
        "dataset_id": "hiliftaeroml",
        "split_id": split_id,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "note": "No package is written by blocker inspection.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--submission-specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--native-aggregate", type=Path)
    parser.add_argument(
        "--native-outputs",
        type=Path,
        help="shared surface/volume output root (backward-compatible shorthand)",
    )
    parser.add_argument(
        "--native-surface-outputs",
        type=Path,
        help="surface output root; overrides --native-outputs for surface artifacts",
    )
    parser.add_argument(
        "--native-volume-outputs",
        type=Path,
        help="volume output root; overrides --native-outputs for volume artifacts",
    )
    parser.add_argument("--native-receipts", type=Path)
    parser.add_argument(
        "--candidate-profile-truth-release",
        type=Path,
        help=(
            "local inactive hidden-truth release root; candidate dry-run use only"
        ),
    )
    parser.add_argument(
        "--candidate-compact-profile-support-release",
        type=Path,
        help=(
            "local inactive compact evaluator-support release root; candidate "
            "dry-run use only and mutually exclusive with the native-v1 truth path"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help=(
            "participant package directory; repeat exactly twice to build two "
            "independent packages while reusing one already-validated compact "
            "support handle"
        ),
    )
    parser.add_argument(
        "--list-blockers",
        action="store_true",
        help="report unresolved owner/config/native gates without writing a package",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_blockers:
            result = inspect_blockers(
                config_path=args.config,
                specification_path=args.submission_specification,
                native_aggregate_root=args.native_aggregate,
                native_outputs_root=args.native_outputs,
                native_surface_outputs_root=args.native_surface_outputs,
                native_volume_outputs_root=args.native_volume_outputs,
                native_receipts_root=args.native_receipts,
                candidate_profile_truth_release_root=(
                    args.candidate_profile_truth_release
                ),
                candidate_compact_profile_support_release_root=(
                    args.candidate_compact_profile_support_release
                ),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        missing = [
            name
            for name in ("native_aggregate", "native_receipts")
            if getattr(args, name) is None
        ]
        if not args.output:
            missing.append("output")
        if missing:
            raise HiLiftPackageAssemblyError(
                "assembly requires arguments: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        supplied_profile_releases = sum(
            value is not None
            for value in (
                args.candidate_profile_truth_release,
                args.candidate_compact_profile_support_release,
            )
        )
        if supplied_profile_releases != 1:
            raise HiLiftPackageAssemblyError(
                "assembly requires exactly one of --candidate-profile-truth-release "
                "or --candidate-compact-profile-support-release"
            )
        _resolve_native_output_roots(
            outputs_root=args.native_outputs,
            surface_outputs_root=args.native_surface_outputs,
            volume_outputs_root=args.native_volume_outputs,
        )
        outputs = args.output
        if len(outputs) > 2:
            raise HiLiftPackageAssemblyError(
                "assembly accepts one output or exactly two independent outputs"
            )
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise HiLiftPackageAssemblyError(
                "assembly output directories must be distinct"
            )
        opened_compact_support_release: CompactSupportRelease | None = None
        if (
            args.candidate_compact_profile_support_release is not None
            and len(outputs) > 1
        ):
            config = load_json(args.config, label="package config")
            specification = load_json(
                args.submission_specification,
                label="HiLiftAeroML specification",
            )
            split_id = config.get("split_id")
            if not isinstance(split_id, str):
                raise HiLiftPackageAssemblyError("config.split_id must be a string")
            split, case_ids, _ = _find_split(
                specification, args.submission_specification, split_id
            )
            _, compact_declaration = _compact_profile_declaration(
                specification, args.submission_specification
            )
            opened_compact_support_release = _open_or_validate_compact_support_release(
                release_root=args.candidate_compact_profile_support_release,
                declaration=compact_declaration,
                case_ids=case_ids,
                case_set_id=split["case_set_id"],
                opened_release=None,
            )
        results = [
            assemble_package(
                config_path=args.config,
                specification_path=args.submission_specification,
                native_aggregate_root=args.native_aggregate,
                native_outputs_root=args.native_outputs,
                native_surface_outputs_root=args.native_surface_outputs,
                native_volume_outputs_root=args.native_volume_outputs,
                native_receipts_root=args.native_receipts,
                candidate_profile_truth_release_root=(
                    args.candidate_profile_truth_release
                ),
                candidate_compact_profile_support_release_root=(
                    args.candidate_compact_profile_support_release
                ),
                output_path=output,
                opened_compact_support_release=opened_compact_support_release,
            )
            for output in outputs
        ]
        result: dict[str, Any]
        if len(results) == 1:
            result = results[0]
        else:
            result = {
                "status": "candidate_packages_assembled_not_approved",
                "output_count": len(results),
                "packages": results,
            }
    except HiLiftPackageAssemblyError as error:
        print(
            json.dumps({"status": "blocked", "error": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
