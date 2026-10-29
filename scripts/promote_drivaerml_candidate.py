#!/usr/bin/env python3
"""Promote the reviewed DrivAerML proposal into the active closed candidate.

This migration is deliberately reproducible and idempotent. It activates the
real public split identities and the participant-facing scientific contract,
but it does not open submissions or manufacture missing profile supports or
all-case evaluator replays.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scores import (
    composite_component_group_scores,
    composite_overall_score,
)
from reference.drivaerml.retained_file import RetainedFileError, RetainedVerifiedFile
from scripts.validate_scoring_supports import validate_candidate_manifest_release


BENCHMARK_ROOT = ROOT / "benchmark-specs" / "drivaerml"
PROPOSAL_ROOT = BENCHMARK_ROOT / "proposal"
SUBMISSIONS_ROOT = ROOT / "submissions" / "drivaerml"
MANIFEST_PATH = ROOT / "leaderboard" / "manifest.json"
CANDIDATE_RELEASE_BINDINGS_PATH = BENCHMARK_ROOT / "candidate-release-bindings.json"

UNRESOLVED_RELEASE_PREFIX = "__UNRESOLVED_DRIVAERML_"
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
SAFE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,159}$")
PROFILE_DEFINITION_V10_ID = "drivaerml-diagnostics-v10-candidate"
RELATIVE_DIAGNOSTICS_V3_ID = "drivaerml-relative-diagnostics-v3-candidate"
RELATIVE_DIAGNOSTICS_V3_SHA256 = (
    "a6066fa83a7af0d18b84a18c11079f020023a6aed81292c4251186c14fc0461a"
)
RELATIVE_PROFILE_SCHEMA_SHA256 = (
    "ff5c5965bb00633303b9372360879d02535f7946c882b9cfcefe1ee55446a0d2"
)

DATASET_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
DATASET_VERSION = "drivaerml-native-v3-candidate"
EVALUATOR_VERSION = "drivaerml-evaluator-v3-candidate"
SPLIT_MANIFEST_SHA256 = "032a2e9f88926d9218a1943b51e650135cc78683cad6b0a38f3cf4f9dfba647d"
SOURCE_PIN_SHA256 = "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
SURFACE_AREA_MANIFEST_SHA256 = "1401c7e80bd86f3aa2d640289db9b088ce1e0825327e18eeb1ab2852de04323e"
FORCE_TABLE_SHA256 = "4e9e003da38ccdcacad359451079888361eae221d3c8dad7fd5682250d257865"
FORCE_R2_STATISTICS_SHA256 = "6deb7f61ef472eb1d6891927147bf9aafe7125ed6da4f96f0dcc05e0ba178b55"
OWNER_SCIENTIFIC_APPROVAL_SHA256 = (
    "448f2b852df7066fc311eed75ea94caa756ae6f63573d12a50798804c08929b3"
)
PROFILE_DEFINITION_V10_SHA256 = (
    "9136feac7ac276d91f7cbcfb6c558beffc97cfaf0f0174626fee67a1f2a52bd0"
)
NATIVE_V3_TRUTH_INDEX_SHA256 = (
    "e7cf14f161fc7dbf22794e6f66db4e157329be960d2a4368140e08cf0608a5ae"
)

RETIRED_DRIVAER_PHYSICAL_VOLUME_METRIC_IDS = frozenset(
    {
        "drivaerml_volume_velocity_physical_mae",
        "drivaerml_volume_velocity_physical_rmse",
        "drivaerml_volume_pressure_physical_mae",
        "drivaerml_volume_pressure_physical_rmse",
        "cp_probe_rmse",
        "cp_panel_macro_rmse",
    }
)

DRIVAER_DIMENSIONAL_CATALOG = (
    {
        "id": "drivaerml_surface_pmeantrim_native_area",
        "label": "Surface pMeanTrim",
        "unit": "m^2/s^2",
        "digits": 3,
        "statistics": ["mae", "rmse"],
        "weighting": "native_surface_polygon_area",
    },
    {
        "id": "drivaerml_surface_wallshearstressmeantrim_native_area",
        "label": "Surface wallShearStressMeanTrim",
        "unit": "m^2/s^2",
        "digits": 3,
        "statistics": ["mae", "rmse"],
        "weighting": "native_surface_polygon_area",
    },
    {
        "id": "drivaerml_volume_umeantrim_equal_cell",
        "label": "Volume UMeanTrim",
        "unit": "m/s",
        "digits": 3,
        "statistics": ["mae", "rmse"],
        "weighting": "equal_native_volume_cell",
    },
    {
        "id": "drivaerml_volume_pmeantrim_equal_cell",
        "label": "Volume pMeanTrim",
        "unit": "m^2/s^2",
        "digits": 3,
        "statistics": ["mae", "rmse"],
        "weighting": "equal_native_volume_cell",
    },
)

DRIVAER_COEFFICIENT_CATALOG = (
    {
        "id": "drivaerml_cd_equal_case_rmse",
        "label": "Field-integrated C_D",
        "unit": "",
        "digits": 5,
        "statistic": "rmse",
        "aggregation": "all_test_cases",
        "weighting": "cases_equal",
    },
    {
        "id": "drivaerml_cl_equal_case_rmse",
        "label": "Field-integrated C_L",
        "unit": "",
        "digits": 5,
        "statistic": "rmse",
        "aggregation": "all_test_cases",
        "weighting": "cases_equal",
    },
    {
        "id": "drivaerml_cmpitch_equal_case_rmse",
        "label": "Field-integrated C_M pitch",
        "unit": "",
        "digits": 5,
        "statistic": "rmse",
        "aggregation": "all_test_cases",
        "weighting": "cases_equal",
    },
)

SPLITS = {
    "full": ("Full", "standard"),
    "medium": ("Medium", "standard"),
    "scarce": ("Scarce", "standard"),
    "super_scarce": ("Super scarce", "standard"),
    "geometry": ("Geometry", "geometry"),
    "high_drag": ("High drag", "high_drag"),
    "low_drag": ("Low drag", "low_drag"),
    "rear_separation": ("Rear separation", "rear_separation"),
}

SCORE_EQUATIONS = {
    "overall_score": r"\sum_k \alpha_k S_k,\quad \sum_k \alpha_k=1",
    "field_score": r"\frac{\sum_{j\in F}\alpha_j S_j}{\sum_{j\in F}\alpha_j}",
    "force_score": r"\frac{\sum_{j\in C}\alpha_j S_j}{\sum_{j\in C}\alpha_j}",
    "diagnostic_score": r"\frac{\sum_{j\in D}\alpha_j S_j}{\sum_{j\in D}\alpha_j}",
}

RELATIVE_L2_EQUATION = (
    r"100\,\frac{\sqrt{\sum_i w_i\lVert\hat{\mathbf{y}}_i-"
    r"\mathbf{y}_i\rVert_2^2}}{\sqrt{\sum_i w_i\lVert\mathbf{y}_i\rVert_2^2}}"
)
MAE_EQUATION = (
    r"\frac{\sum_i w_i\lVert\hat{\mathbf{y}}_i-\mathbf{y}_i\rVert_2}{\sum_i w_i}"
)
RMSE_EQUATION = (
    r"\sqrt{\frac{\sum_i w_i\lVert\hat{\mathbf{y}}_i-\mathbf{y}_i\rVert_2^2}{\sum_i w_i}}"
)
CASE_RMSE_EQUATION = r"\sqrt{\frac{1}{N}\sum_c(\hat{y}_c-y_c)^2}"


class AuthoritativeJSONError(ValueError):
    """Raised when benchmark-owned JSON is ambiguous or unsafe to consume."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthoritativeJSONError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> Any:
    raise AuthoritativeJSONError(f"JSON contains forbidden non-finite token {token}")


def load_json_with_sha256(path: Path, *, label: str) -> tuple[Any, str]:
    """Hash and parse one retained regular-file descriptor."""

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
    except AuthoritativeJSONError:
        raise
    except RetainedFileError as error:
        raise AuthoritativeJSONError(str(error)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuthoritativeJSONError(f"cannot read valid {label} JSON: {path}") from error
    return value, digest


def load_json(path: Path) -> Any:
    return load_json_with_sha256(path, label=str(path))[0]


def resolved_benchmark_file(
    relative_value: Any,
    *,
    label: str,
    benchmark_root: Path | None = None,
) -> Path:
    """Resolve a release-managed path while proving dataset containment."""

    root = BENCHMARK_ROOT if benchmark_root is None else benchmark_root
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside DrivAerML")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = (root / relative).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} does not exist: {root / relative}") from error
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} must stay inside DrivAerML")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, indent=2, ensure_ascii=True)}\n",
        encoding="utf-8",
    )


def unresolved_release_tokens(
    value: Any, path: tuple[str | int, ...] = ()
) -> list[tuple[tuple[str | int, ...], str]]:
    """Return explicit release blockers without treating them as real values."""

    found: list[tuple[tuple[str | int, ...], str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            found.extend(unresolved_release_tokens(value[key], (*path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(unresolved_release_tokens(item, (*path, index)))
    elif isinstance(value, str) and value.startswith(UNRESOLVED_RELEASE_PREFIX):
        found.append((path, value))
    return found


def validate_relative_diagnostics_v3_binding(
    binding: Any,
    *,
    bindings_path: Path = CANDIDATE_RELEASE_BINDINGS_PATH,
) -> tuple[dict[str, Any], Path]:
    """Validate the retained report-only contract and its versioned schema."""

    required = {
        "file",
        "sha256",
        "profile_chunk_schema_file",
        "profile_chunk_schema_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError(
            "relative_diagnostics_v3 binding must contain exactly "
            f"{sorted(required)}"
        )
    contract_path = resolved_benchmark_file(
        binding["file"],
        label="relative_diagnostics_v3.file",
        benchmark_root=bindings_path.parent,
    )
    contract, contract_digest = load_json_with_sha256(
        contract_path,
        label="relative-diagnostics-v3 contract",
    )
    if contract_digest != binding["sha256"]:
        raise ValueError("relative_diagnostics_v3 SHA-256 does not match its file")
    scoring_and_rollout = (
        contract.get("scoring_and_rollout")
        if isinstance(contract, dict)
        else None
    )
    if (
        not isinstance(contract, dict)
        or not isinstance(scoring_and_rollout, dict)
        or contract.get("id") != "drivaerml-relative-diagnostics-v3-candidate"
        or contract.get("dataset_id") != "drivaerml"
        or contract.get("activation_by_this_file") is not False
        or scoring_and_rollout.get("relative_composite_weight") != 0.0
        or scoring_and_rollout.get("constant_family_policy") != "unchanged"
    ):
        raise ValueError("relative-diagnostics-v3 contract is not fail-closed")

    repository_root = bindings_path.parent.parent.parent
    schema_path = resolved_benchmark_file(
        binding["profile_chunk_schema_file"],
        label="relative_diagnostics_v3.profile_chunk_schema_file",
        benchmark_root=repository_root,
    )
    schema, schema_digest = load_json_with_sha256(
        schema_path,
        label="relative profile-chunk schema",
    )
    if schema_digest != binding["profile_chunk_schema_sha256"]:
        raise ValueError(
            "relative profile-chunk schema SHA-256 does not match its file"
        )
    if (
        not isinstance(schema, dict)
        or schema.get("$id")
        != "https://fluidsbench.org/schemas/v1/drivaerml-relative-profile-chunk.schema.json"
    ):
        raise ValueError("relative profile-chunk schema identity is invalid")
    return contract, schema_path


def validate_relative_support_bindings(
    value: Any,
    *,
    relative_contract: Any,
    bindings_path: Path = CANDIDATE_RELEASE_BINDINGS_PATH,
) -> bool:
    """Validate atomic relative-support manifest bindings.

    Returns true only when every manifest identity is resolved, every retained
    manifest byte stream and its complete official case scope have been
    verified, and the local relative-support status is ready.  No manifest is
    treated as scoring support by this hand-off.
    """

    required = {
        "status",
        "required_case_count",
        "velocity_placement_manifest",
        "velocity_mapping_manifest",
        "cp_manifest",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"relative_support must contain exactly {sorted(required)}"
        )
    if value["status"] not in {"unresolved", "ready"}:
        raise ValueError("relative_support.status must be unresolved or ready")
    if value["required_case_count"] != 484:
        raise ValueError("relative_support.required_case_count must remain 484")
    expected_manifests = {
        "velocity_placement_manifest": {
            "family_id": "drivaerml-velocity-relative-v3",
            "producer_file": (
                "velocity_support_v3/production_campaign_v1/aggregate/"
                "relative-velocity-v3-production-all484-inputs-v1.json"
            ),
            "expected_schema": (
                "drivaerml-relative-velocity-v3-production-input-manifest-v1"
            ),
            "expected_schema_version": 1,
            "manifest_file": (
                "support/relative-v3/manifests/velocity-placement-all484-v1.json"
            ),
            "contract_family": "relative_velocity_v3",
            "contract_binding": "placement_all484_manifest",
        },
        "velocity_mapping_manifest": {
            "family_id": "drivaerml-velocity-relative-v3",
            "producer_file": (
                "velocity_mapping_v3/aggregate/"
                "relative-velocity-v3-mapping-all484-v1.json"
            ),
            "expected_schema": (
                "drivaerml-velocity-relative-v3-mapping-aggregate-v1"
            ),
            "expected_schema_version": 1,
            "manifest_file": (
                "support/relative-v3/manifests/velocity-mapping-all484-v1.json"
            ),
            "contract_family": "relative_velocity_v3",
            "contract_binding": "mapping_all484_manifest",
        },
        "cp_manifest": {
            "family_id": "drivaerml_cp_relative_v1",
            "producer_file": (
                "cp_support/campaign_v3/aggregate_v3/all484/"
                "relative-cp-native-support-manifest-v3.json"
            ),
            "expected_schema": "drivaerml-relative-cp-native-support-manifest-v3",
            "expected_schema_version": 3,
            "manifest_file": (
                "support/relative-v3/manifests/cp-native-support-all484-v3.json"
            ),
            "contract_family": "relative_cp_v1",
            "contract_binding": "cp_all484_manifest",
        },
    }
    manifest_fields = {
        "family_id",
        "producer_file",
        "manifest_file",
        "expected_schema",
        "manifest_sha256",
    }
    support_implementations = (
        relative_contract.get("support_implementation_bindings")
        if isinstance(relative_contract, dict)
        else None
    )
    if not isinstance(support_implementations, dict):
        raise ValueError(
            "relative-diagnostics-v3 contract has no support implementation bindings"
        )

    official_case_ids: tuple[str, ...] | None = None

    def load_official_case_ids() -> tuple[str, ...]:
        nonlocal official_case_ids
        if official_case_ids is not None:
            return official_case_ids
        pin_path = resolved_benchmark_file(
            "proposal/native-source-pin.json",
            label="relative_support official native-source pin",
            benchmark_root=bindings_path.parent,
        )
        pin, pin_digest = load_json_with_sha256(
            pin_path,
            label="relative_support official native-source pin",
        )
        if pin_digest != SOURCE_PIN_SHA256:
            raise ValueError(
                "relative_support official native-source pin SHA-256 changed"
            )
        raw_cases = pin.get("cases") if isinstance(pin, dict) else None
        if (
            not isinstance(pin, dict)
            or pin.get("schema")
            != "drivaerml-fluidsbench-public-native-source-pin-v1"
            or pin.get("schema_version") != 1
            or pin.get("repository", {}).get("repo_id") != "neashton/drivaerml"
            or pin.get("repository", {}).get("revision") != DATASET_REVISION
            or not isinstance(raw_cases, list)
            or len(raw_cases) != 484
        ):
            raise ValueError(
                "relative_support official native-source pin identity is invalid"
            )
        case_ids: list[str] = []
        for position, case in enumerate(raw_cases):
            case_id = case.get("case_id") if isinstance(case, dict) else None
            if not isinstance(case_id, str) or re.fullmatch(
                r"run_[1-9][0-9]*", case_id
            ) is None:
                raise ValueError(
                    "relative_support official native-source pin contains an "
                    f"invalid case ID at position {position}"
                )
            case_ids.append(case_id)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "relative_support official native-source pin contains duplicate case IDs"
            )
        official_case_ids = tuple(case_ids)
        return official_case_ids

    def validate_manifest_cases(
        label: str,
        manifest: dict[str, Any],
        expected_case_ids: tuple[str, ...],
    ) -> None:
        raw_cases = manifest.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"relative_support.{label} cases must be an array")
        observed: list[str] = []
        for position, case in enumerate(raw_cases):
            case_id = case.get("case_id") if isinstance(case, dict) else None
            if not isinstance(case_id, str):
                raise ValueError(
                    f"relative_support.{label} case {position} has no case_id"
                )
            observed.append(case_id)
        if len(observed) != len(set(observed)):
            raise ValueError(
                f"relative_support.{label} contains duplicate case IDs"
            )
        if tuple(observed) != expected_case_ids:
            expected_set = set(expected_case_ids)
            observed_set = set(observed)
            raise ValueError(
                f"relative_support.{label} case coverage/order differs from the "
                "official 484-case set "
                f"(missing={sorted(expected_set - observed_set)[:5]}, "
                f"unexpected={sorted(observed_set - expected_set)[:5]})"
            )

        if label == "velocity_placement_manifest":
            if (
                manifest.get("case_count") != 484
                or manifest.get("case_ids") != list(expected_case_ids)
                or manifest.get("case_indices") != list(range(484))
                or any(
                    not isinstance(case, dict)
                    or case.get("case_index") != position
                    for position, case in enumerate(raw_cases)
                )
            ):
                raise ValueError(
                    "relative_support.velocity_placement_manifest count/order "
                    "fields differ from the official 484-case set"
                )
        elif label == "velocity_mapping_manifest":
            if (
                manifest.get("official_case_count") != 484
                or manifest.get("included_case_count") != 484
                or manifest.get("complete_official_case_coverage") is not True
            ):
                raise ValueError(
                    "relative_support.velocity_mapping_manifest count/completion "
                    "fields differ from the official 484-case set"
                )
        elif (
            manifest.get("case_count") != 484
            or manifest.get("all_official_cases_generated_and_replayed") is not True
            or any(
                not isinstance(case, dict) or case.get("case_index") != position
                for position, case in enumerate(raw_cases)
            )
        ):
            raise ValueError(
                "relative_support.cp_manifest count/order fields differ from the "
                "official 484-case set"
            )

    all_resolved = True
    for label, expected in expected_manifests.items():
        binding = value[label]
        if not isinstance(binding, dict) or set(binding) != manifest_fields:
            raise ValueError(
                f"relative_support.{label} must contain exactly {sorted(manifest_fields)}"
            )
        for field in ("family_id", "producer_file", "expected_schema", "manifest_file"):
            expected_value = expected[field]
            if binding[field] != expected_value:
                raise ValueError(
                    f"relative_support.{label}.{field} differs from the v3 contract"
                )
        producer_file = Path(binding["producer_file"])
        if producer_file.is_absolute() or ".." in producer_file.parts:
            raise ValueError(
                f"relative_support.{label}.producer_file must be a safe producer-relative path"
            )
        manifest_file = Path(binding["manifest_file"])
        if manifest_file.is_absolute() or ".." in manifest_file.parts:
            raise ValueError(
                f"relative_support.{label}.manifest_file must stay inside DrivAerML"
            )
        contract_family = support_implementations.get(expected["contract_family"])
        contract_binding = (
            contract_family.get(expected["contract_binding"])
            if isinstance(contract_family, dict)
            else None
        )
        if (
            not isinstance(contract_binding, dict)
            or contract_binding.get("producer_path") != binding["producer_file"]
        ):
            raise ValueError(
                f"relative_support.{label} producer path differs from the "
                "relative-diagnostics-v3 contract"
            )
        digest = binding["manifest_sha256"]
        if (
            isinstance(digest, str)
            and digest.startswith(UNRESOLVED_RELEASE_PREFIX)
        ):
            all_resolved = False
            continue
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"relative_support.{label} binding is invalid")
        retained_path = resolved_benchmark_file(
            binding["manifest_file"],
            label=f"relative_support.{label}.manifest_file",
            benchmark_root=bindings_path.parent,
        )
        manifest, actual_digest = load_json_with_sha256(
            retained_path,
            label=f"relative_support.{label} retained manifest",
        )
        if actual_digest != digest:
            raise ValueError(
                f"relative_support.{label} SHA-256 does not match its retained file"
            )
        if contract_binding.get("sha256") != actual_digest:
            raise ValueError(
                f"relative_support.{label} SHA-256 differs from the "
                "relative-diagnostics-v3 contract"
            )
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != binding["expected_schema"]
            or manifest.get("schema_version") != expected["expected_schema_version"]
            or manifest.get("dataset_id") != "drivaerml"
            or manifest.get("family_id") != binding["family_id"]
            or manifest.get("public_dataset_revision") != DATASET_REVISION
        ):
            raise ValueError(
                f"relative_support.{label} retained manifest schema/identity is invalid"
            )
        validate_manifest_cases(label, manifest, load_official_case_ids())
    if value["status"] == "ready" and not all_resolved:
        raise ValueError("ready relative_support still contains unresolved tokens")
    if value["status"] == "unresolved" and all_resolved:
        raise ValueError("resolved relative_support must set status=ready")
    return value["status"] == "ready"


def validate_scientific_approval_binding(
    value: Any,
    *,
    bindings_path: Path,
    release_bindings: dict[str, Any],
) -> None:
    """Validate the non-activating dataset-owner scientific approval."""

    required = {
        "status",
        "approved_by",
        "approved_at",
        "approval_record",
        "release_activation_effect",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "scientific_approval must contain exactly " f"{sorted(required)}"
        )
    if (
        value["status"] != "approved"
        or value["approved_by"] != "neilashton"
        or value["approved_at"] != "2026-08-28"
        or value["release_activation_effect"]
        != "none_until_remaining_release_gates_complete"
    ):
        raise ValueError("scientific_approval decision metadata is invalid")
    record_binding = value["approval_record"]
    record_fields = {"file", "sha256", "schema"}
    if not isinstance(record_binding, dict) or set(record_binding) != record_fields:
        raise ValueError(
            "scientific_approval.approval_record must contain exactly "
            f"{sorted(record_fields)}"
        )
    if record_binding["schema"] != "drivaerml-owner-scientific-approval-v1":
        raise ValueError("scientific approval record schema is invalid")
    if (
        record_binding["file"]
        != "evidence/owner-scientific-approval-2026-08-28.json"
        or record_binding["sha256"] != OWNER_SCIENTIFIC_APPROVAL_SHA256
    ):
        raise ValueError("scientific approval record identity is not the reviewed record")
    record_path = resolved_benchmark_file(
        record_binding["file"],
        label="scientific_approval.approval_record.file",
        benchmark_root=bindings_path.parent,
    )
    record, record_digest = load_json_with_sha256(
        record_path,
        label="DrivAerML scientific approval record",
    )
    if record_digest != record_binding["sha256"]:
        raise ValueError("scientific approval record SHA-256 does not match its file")
    if (
        not isinstance(record, dict)
        or record.get("schema") != record_binding["schema"]
        or record.get("schema_version") != 1
        or record.get("dataset_id") != "drivaerml"
        or record.get("status") != value["status"]
        or record.get("approved_by") != value["approved_by"]
        or record.get("approved_at") != value["approved_at"]
        or record.get("authority") != "dataset_owner"
    ):
        raise ValueError("scientific approval record identity is invalid")
    evidence = record.get("evidence_bindings")
    non_activation = record.get("non_activation")
    if not isinstance(evidence, dict) or not isinstance(non_activation, dict):
        raise ValueError("scientific approval record is incomplete")
    expected_evidence = {
        "profile_definition_v10": release_bindings["profile_definition_v10"],
        "velocity_placement_all484": {
            "file": release_bindings["relative_support"][
                "velocity_placement_manifest"
            ]["manifest_file"],
            "sha256": release_bindings["relative_support"][
                "velocity_placement_manifest"
            ]["manifest_sha256"],
        },
        "velocity_mapping_all484": {
            "file": release_bindings["relative_support"][
                "velocity_mapping_manifest"
            ]["manifest_file"],
            "sha256": release_bindings["relative_support"][
                "velocity_mapping_manifest"
            ]["manifest_sha256"],
        },
        "cp_support_all484": {
            "file": release_bindings["relative_support"]["cp_manifest"][
                "manifest_file"
            ],
            "sha256": release_bindings["relative_support"]["cp_manifest"][
                "manifest_sha256"
            ],
        },
    }
    for label, expected in expected_evidence.items():
        if evidence.get(label) != expected:
            raise ValueError(
                f"scientific approval {label} differs from the release hand-off"
            )
    for field in (
        "submissions_opened",
        "official_ranking_enabled",
        "production_evaluator_release_bound",
        "scoring_support_release_bound",
        "genuine_three_model_sensitivity_complete",
        "independent_full_submission_dry_run_complete",
        "relative_profile_format_authorized",
    ):
        if non_activation.get(field) is not False:
            raise ValueError(
                f"scientific approval must keep non_activation.{field}=false"
            )
    if non_activation.get("relative_composite_weight") != 0.0:
        raise ValueError("scientific approval must keep relative composite weight zero")


def load_candidate_release_bindings(
    path: Path = CANDIDATE_RELEASE_BINDINGS_PATH,
) -> dict[str, Any]:
    """Load the release hand-off while rejecting incoherent partial states.

    An unresolved hand-off may still pin the complete local profile-v10
    file/SHA-256 pair.  The pair is atomic: both fields must remain explicit
    unresolved tokens or both must resolve to the exact retained artifact next
    to this hand-off file.  The nested report-only relative_support lifecycle
    is validated independently and never blocks the ranked constant candidate.
    """

    value = load_json(path)
    if value.get("schema") != "drivaerml-fluidsbench-candidate-release-bindings-v1":
        raise ValueError("unsupported DrivAerML candidate release-binding schema")
    if value.get("status") not in {"unresolved", "ready"}:
        raise ValueError("candidate release-binding status must be unresolved or ready")
    if value.get("unresolved_token_prefix") != UNRESOLVED_RELEASE_PREFIX:
        raise ValueError("candidate release-binding unresolved_token_prefix is invalid")
    required_objects = {
        "candidate_manifest": {
            "status", "release_id", "manifest_file", "manifest_url", "manifest_sha256"
        },
        "evaluator": {"reference_version", "code_revision"},
        "profile_definition_v10": {"file", "sha256"},
        "relative_diagnostics_v3": {
            "file",
            "sha256",
            "profile_chunk_schema_file",
            "profile_chunk_schema_sha256",
        },
        "relative_support": {
            "status",
            "required_case_count",
            "velocity_placement_manifest",
            "velocity_mapping_manifest",
            "cp_manifest",
        },
        "profile_ground_truth": {"release_id", "manifest_sha256"},
    }
    for key, required in required_objects.items():
        item = value.get(key)
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"candidate release bindings require {key} fields {sorted(required)}")
    if value["candidate_manifest"].get("status") != "candidate":
        raise ValueError("candidate_manifest.status must remain candidate")
    if value["evaluator"].get("reference_version") != EVALUATOR_VERSION:
        raise ValueError("candidate evaluator reference version changed unexpectedly")
    profile_binding = value["profile_definition_v10"]
    if set(profile_binding) != {"file", "sha256"}:
        raise ValueError("profile_definition_v10 binding must contain exactly file and sha256")
    file_is_token = (
        isinstance(profile_binding.get("file"), str)
        and profile_binding["file"].startswith(UNRESOLVED_RELEASE_PREFIX)
    )
    sha256_is_token = (
        isinstance(profile_binding.get("sha256"), str)
        and profile_binding["sha256"].startswith(UNRESOLVED_RELEASE_PREFIX)
    )
    if file_is_token != sha256_is_token:
        raise ValueError(
            "profile_definition_v10 file and SHA-256 must be resolved together"
        )
    if not file_is_token:
        validate_profile_v10_binding(
            profile_binding,
            benchmark_root=path.parent,
        )
    relative_contract, _relative_schema_path = validate_relative_diagnostics_v3_binding(
        value["relative_diagnostics_v3"],
        bindings_path=path,
    )
    validate_relative_support_bindings(
        value["relative_support"],
        relative_contract=relative_contract,
        bindings_path=path,
    )
    if "scientific_approval" in value:
        validate_scientific_approval_binding(
            value["scientific_approval"],
            bindings_path=path,
            release_bindings=value,
        )
    tokens = [
        item
        for item in unresolved_release_tokens(value)
        if item[0] != ("unresolved_token_prefix",)
        and item[0][:1] != ("relative_support",)
    ]
    if value["status"] == "unresolved":
        if not tokens:
            raise ValueError("unresolved release bindings must retain explicit blocker tokens")
        return value
    if tokens:
        rendered = [".".join(map(str, location)) for location, _token in tokens]
        raise ValueError(f"ready release bindings still contain tokens: {rendered}")
    candidate = value["candidate_manifest"]
    if (
        not isinstance(candidate.get("release_id"), str)
        or not SAFE_ID_PATTERN.fullmatch(candidate["release_id"])
        or not isinstance(candidate.get("manifest_file"), str)
        or not isinstance(candidate.get("manifest_url"), str)
        or not isinstance(candidate.get("manifest_sha256"), str)
        or not SHA256_PATTERN.fullmatch(candidate["manifest_sha256"])
    ):
        raise ValueError("candidate manifest binding is invalid")
    if (
        not isinstance(value["evaluator"].get("code_revision"), str)
        or not GIT_REVISION_PATTERN.fullmatch(value["evaluator"]["code_revision"])
    ):
        raise ValueError("candidate evaluator Git revision is invalid")
    for label, digest in (
        ("profile_definition_v10", value["profile_definition_v10"]["sha256"]),
        ("profile_ground_truth", value["profile_ground_truth"]["manifest_sha256"]),
    ):
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"{label} SHA-256 is invalid")
    profile_file_value = value["profile_definition_v10"].get("file")
    if not isinstance(profile_file_value, str):
        raise ValueError("profile_definition_v10.file is invalid")
    profile_file = Path(profile_file_value)
    if (
        profile_file.is_absolute()
        or ".." in profile_file.parts
        or "v10" not in profile_file.name
    ):
        raise ValueError("profile_definition_v10.file must be a safe v10 dataset path")
    truth_release_id = value["profile_ground_truth"].get("release_id")
    if (
        not isinstance(truth_release_id, str)
        or not SAFE_ID_PATTERN.fullmatch(truth_release_id)
    ):
        raise ValueError("profile_ground_truth.release_id is invalid")
    return value


def validate_ready_candidate_manifest(
    release_bindings: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact local candidate bytes and their support semantics."""

    if release_bindings.get("status") != "ready":
        raise ValueError("candidate manifest materialization requires ready bindings")
    candidate = release_bindings["candidate_manifest"]
    release_id = candidate["release_id"]
    if not SAFE_ID_PATTERN.fullmatch(release_id):
        raise ValueError("candidate manifest release_id is not a safe ID")
    manifest_path = resolved_benchmark_file(
        candidate.get("manifest_file"),
        label="candidate manifest_file",
    )
    manifest, actual_sha256 = load_json_with_sha256(
        manifest_path,
        label="candidate scoring-support manifest",
    )
    if actual_sha256 != candidate["manifest_sha256"]:
        raise ValueError("candidate manifest SHA-256 does not match its local file")
    parsed = urlparse(candidate["manifest_url"])
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or release_id not in [part for part in parsed.path.split("/") if part]
    ):
        raise ValueError(
            "candidate manifest_url must be clean HTTPS and contain release_id "
            "as an exact path segment"
        )
    if not isinstance(manifest, dict):
        raise ValueError("candidate scoring-support manifest must be a JSON object")
    manifest_tokens = unresolved_release_tokens(manifest)
    if manifest_tokens:
        rendered = [
            ".".join(map(str, location)) for location, _token in manifest_tokens
        ]
        raise ValueError(
            "candidate scoring-support manifest contains unresolved release tokens: "
            + ", ".join(rendered)
        )
    identities = {
        "status": "candidate",
        "release_id": release_id,
        "dataset_id": "drivaerml",
        "dataset_version": DATASET_VERSION,
        "evaluation_reference_version": EVALUATOR_VERSION,
    }
    for key, expected in identities.items():
        if manifest.get(key) != expected:
            raise ValueError(f"candidate manifest {key} must equal {expected!r}")
    if "owner_approval" in manifest:
        raise ValueError("candidate manifest must not claim owner approval")
    semantic_errors = validate_candidate_manifest_release(
        specification,
        BENCHMARK_ROOT,
        manifest_path,
        manifest,
    )
    if semantic_errors:
        raise ValueError(
            "candidate scoring-support manifest is not valid support: "
            + "; ".join(semantic_errors)
        )
    return manifest


def upsert_catalog_entries(
    catalog_entries: list[dict[str, Any]],
    dataset_entries: tuple[dict[str, Any], ...],
) -> None:
    """Add or replace only dataset-prefixed metric-catalog entries."""

    positions = {
        entry.get("id"): index
        for index, entry in enumerate(catalog_entries)
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    for entry in dataset_entries:
        metric_id = entry["id"]
        if not metric_id.startswith("drivaerml_"):
            raise ValueError("DrivAerML catalog entries must use a dataset-prefixed ID")
        position = positions.get(metric_id)
        if position is None:
            positions[metric_id] = len(catalog_entries)
            catalog_entries.append(dict(entry))
        else:
            catalog_entries[position] = dict(entry)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_metric(metric_id: str) -> dict[str, Any]:
    return {
        "id": metric_id,
        "unit": "",
        "direction": "higher",
        "kind": "score",
        "equation": SCORE_EQUATIONS[metric_id],
        "aggregation": "derived_score_equation",
        "weighting": "dataset_declared_component_weights",
    }


def metric(
    metric_id: str,
    *,
    unit: str,
    equation: str,
    aggregation: str,
    weighting: str,
    availability: str | None = None,
) -> dict[str, Any]:
    result = {
        "id": metric_id,
        "unit": unit,
        "direction": "lower",
        "kind": "error",
        "equation": equation,
        "aggregation": aggregation,
        "weighting": weighting,
    }
    if availability is not None:
        result["availability"] = availability
    return result


def build_metrics() -> list[dict[str, Any]]:
    metrics = [score_metric(metric_id) for metric_id in SCORE_EQUATIONS]
    metrics.extend(
        [
            {
                "id": "cd_r2",
                "unit": "",
                "direction": "higher",
                "kind": "r2",
                "equation": r"1-\frac{\sum_c(C_{D,c}-\hat{C}_{D,c})^2}{\sum_c(C_{D,c}-\bar{C}_D)^2}",
                "aggregation": "all_test_cases",
                "weighting": "cases_equal",
            },
            {
                "id": "cl_r2",
                "unit": "",
                "direction": "higher",
                "kind": "r2",
                "equation": r"1-\frac{\sum_c(C_{L,c}-\hat{C}_{L,c})^2}{\sum_c(C_{L,c}-\bar{C}_L)^2}",
                "aggregation": "all_test_cases",
                "weighting": "cases_equal",
            },
            {
                "id": "c_pitch_r2",
                "unit": "",
                "direction": "higher",
                "kind": "r2",
                "equation": r"1-\frac{\sum_c(C_{M,c}-\hat{C}_{M,c})^2}{\sum_c(C_{M,c}-\bar{C}_M)^2}",
                "aggregation": "all_test_cases",
                "weighting": "cases_equal",
            },
            {
                "id": "velocity_profile_r2",
                "unit": "",
                "direction": "higher",
                "kind": "r2",
                "equation": r"1-\frac{\sum_{c,l}\int_{A_{cl}}(\hat{U}/U_\infty-U/U_\infty)^2\,ds/L_{cl}}{\sum_{c,l}\int_{A_{cl}}(U/U_\infty-\bar{U}_w)^2\,ds/L_{cl}}",
                "aggregation": "equal_case_equal_line_global_weighted_r2",
                "weighting": "equal_cases_equal_lines_trapezoidal_arc_length_within_line",
                "availability": (
                    "all_16_lines_available_for_every_case_after_owner_mask_"
                    "and_zero_unresolved_nonexcluded_samples"
                ),
            },
            {
                "id": "cp_cut_r2",
                "unit": "",
                "direction": "higher",
                "kind": "r2",
                "equation": r"1-\frac{\sum_{c,q}\sum_{s\in\Gamma_{cq}}\tilde{w}_{cqs}(\hat{C}_{p,s}-C_{p,s})^2}{\sum_{c,q}\sum_{s\in\Gamma_{cq}}\tilde{w}_{cqs}(C_{p,s}-\bar{C}_{p,w})^2}",
                "aggregation": "equal_case_equal_cut_global_weighted_r2",
                "weighting": "equal_cases_equal_cuts_normalized_native_cut_intersection_segment_length",
                "availability": (
                    "all_four_continuous_native_surface_cuts_available_for_every_"
                    "case_after_immutable_owner_extraction_support_is_published"
                ),
            },
        ]
    )
    metrics.extend(
        [
            metric(
                "surface_pressure_rel_l2",
                unit="%",
                equation=RELATIVE_L2_EQUATION,
                aggregation="per_geometry_then_macro_average",
                weighting="surface_face_area",
            ),
            metric(
                "surface_pressure_equal_entity_rel_l2",
                unit="%",
                equation=RELATIVE_L2_EQUATION,
                aggregation="per_geometry_then_macro_average",
                weighting="surface_entities_equal",
            ),
            metric(
                "surface_wall_shear_rel_l2",
                unit="%",
                equation=RELATIVE_L2_EQUATION,
                aggregation="per_geometry_then_macro_average",
                weighting="surface_face_area",
            ),
            metric(
                "surface_wall_shear_equal_entity_rel_l2",
                unit="%",
                equation=RELATIVE_L2_EQUATION,
                aggregation="per_geometry_then_macro_average",
                weighting="surface_entities_equal",
            ),
            metric(
                "volume_velocity_rel_l2",
                unit="%",
                equation=RELATIVE_L2_EQUATION,
                aggregation="per_geometry_then_macro_average",
                weighting="volume_cells_equal",
            ),
            metric(
                "volume_pressure_rel_l2",
                unit="%",
                equation=RELATIVE_L2_EQUATION,
                aggregation="per_geometry_then_macro_average",
                weighting="volume_cells_equal",
            ),
        ]
    )

    absolute_fields = [
        ("surface_pressure", "m^2/s^2", "area", "surface_face_area"),
        ("surface_pressure", "m^2/s^2", "equal_entity", "surface_entities_equal"),
        ("surface_wall_shear", "m^2/s^2", "area", "surface_face_area"),
        ("surface_wall_shear", "m^2/s^2", "equal_entity", "surface_entities_equal"),
        ("volume_velocity", "m/s", "equal_entity", "volume_cells_equal"),
        ("volume_pressure", "m^2/s^2", "equal_entity", "volume_cells_equal"),
    ]
    for field_id, unit, token, weighting in absolute_fields:
        prefix = f"drivaerml_{field_id}_{token}"
        metrics.extend(
            [
                metric(
                    f"{prefix}_mae",
                    unit=unit,
                    equation=MAE_EQUATION,
                    aggregation="per_geometry_then_macro_average",
                    weighting=weighting,
                ),
                metric(
                    f"{prefix}_rmse",
                    unit=unit,
                    equation=RMSE_EQUATION,
                    aggregation="per_geometry_then_macro_average",
                    weighting=weighting,
                ),
            ]
        )

    for metric_id in (
        "field_integrated_cd_rmse",
        "field_integrated_cl_rmse",
        "field_integrated_cmpitch_rmse",
        "field_integrated_clf_rmse",
        "field_integrated_clr_rmse",
    ):
        metrics.append(
            metric(
                metric_id,
                unit="",
                equation=CASE_RMSE_EQUATION,
                aggregation="all_test_cases",
                weighting="cases_equal",
            )
        )
    metrics.extend(
        [
            metric(
                "field_integrated_lift_closure_max_abs",
                unit="",
                equation=r"\max_c\lvert\hat{C}_{L,c}-(\hat{C}_{Lf,c}+\hat{C}_{Lr,c})\rvert",
                aggregation="all_test_cases",
                weighting="maximum_case_residual",
            ),
            metric(
                "velocity_profile_uinf_rmse",
                unit="",
                equation=(
                    r"\frac{1}{N}\sum_c\frac{1}{16}\sum_l"
                    r"\sqrt{\frac{\int_{A_{cl}}(\hat{U}/U_\infty-U/U_\infty)^2\,ds}{\int_{A_{cl}} ds}}"
                ),
                aggregation="equal_case_equal_line_macro_average",
                weighting="trapezoidal_arc_length_within_line",
                availability=(
                    "all_16_lines_available_for_every_case_after_owner_mask_"
                    "and_zero_unresolved_nonexcluded_samples"
                ),
            ),
            metric(
                "velocity_profile_experimental_subset_uinf_rmse",
                unit="",
                equation=(
                    r"\frac{1}{N}\sum_c\frac{1}{11}\sum_{l\in E}"
                    r"\sqrt{\frac{\int_{A_{cl}}(\hat{U}/U_\infty-U/U_\infty)^2\,ds}{\int_{A_{cl}} ds}}"
                ),
                aggregation="equal_case_equal_experimental_line_macro_average",
                weighting="trapezoidal_arc_length_within_line",
                availability=(
                    "all_11_experimental_lines_available_for_every_case_after_"
                    "owner_mask_and_zero_unresolved_nonexcluded_samples"
                ),
            ),
            metric(
                "cp_cut_rmse",
                unit="",
                equation=(
                    r"\frac{1}{N}\sum_c\frac{1}{4}\sum_q"
                    r"\sqrt{\frac{\sum_{s\in\Gamma_{cq}}\Delta s_s"
                    r"(\hat{C}_{p,s}-C_{p,s})^2}"
                    r"{\sum_{s\in\Gamma_{cq}}\Delta s_s}}"
                ),
                aggregation="equal_case_equal_cut_macro_average",
                weighting="native_cut_intersection_segment_length",
                availability=(
                    "all_four_continuous_native_surface_cuts_available_for_every_"
                    "case_after_immutable_owner_extraction_support_is_published"
                ),
            ),
        ]
    )
    return metrics


def build_profile_definition() -> dict[str, Any]:
    lines_path = PROPOSAL_ROOT / "autocfd_velocity_lines_nominal.csv"
    samples_path = PROPOSAL_ROOT / "autocfd_velocity_samples_10mm.csv"

    line_rows = read_csv(lines_path)
    sample_rows = read_csv(samples_path)
    samples_by_line: dict[str, list[dict[str, str]]] = {}
    for row in sample_rows:
        samples_by_line.setdefault(row["profile_id"], []).append(row)
    velocity_stations = []
    for row in line_rows:
        profile_id = row["profile_id"]
        samples = samples_by_line[profile_id]
        velocity_stations.append(
            {
                "id": f"autocfd5_{profile_id.lower()}",
                "source_profile_id": profile_id,
                "label": f"AutoCFD5 {profile_id}",
                "start_m": [float(row["start_x_m"]), float(row["start_y_m"]), float(row["start_z_m"])],
                "end_m": [float(row["end_x_m"]), float(row["end_y_m"]), float(row["end_z_m"])],
                "length_m": float(row["length_m"]),
                "scoring_spacing_m": float(row["scoring_spacing_m"]),
                "sample_count": int(row["scoring_point_count"]),
                "coordinate_interval_m": [0.0, float(samples[-1]["distance_m"])],
                "experimental_availability": row["autocfd_experimental_availability"],
            }
        )

    pressure_cuts = [
        {
            "id": "upperbody_centerline",
            "label": "Upper-body centreline (y = 0 m)",
            "plane": {"axis": "y", "value_m": 0.0},
            "anatomical_scope": "upper_body_external_surface",
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
        },
        {
            "id": "underbody_centerline",
            "label": "Underbody centreline (y = 0 m)",
            "plane": {"axis": "y", "value_m": 0.0},
            "anatomical_scope": "underbody_external_surface",
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
        },
        {
            "id": "sidewall_z_0_15",
            "label": "Sidewall (z = 0.15 m)",
            "plane": {"axis": "z", "value_m": 0.15},
            "anatomical_scope": "vehicle_sidewall_external_surface",
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
        },
        {
            "id": "front_left_wheelhouse_y_neg_0_6",
            "label": "Front-left wheelhouse (y = -0.6 m)",
            "plane": {"axis": "y", "value_m": -0.6},
            "anatomical_scope": "front_left_wheelhouse_surface",
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
        },
    ]

    return {
        "schema_version": "1.0",
        "id": "drivaerml-diagnostics-v9-candidate",
        "dataset_id": "drivaerml",
        "status": "candidate_velocity_and_cp_cut_support_pending_all_case_validation",
        "source": {
            "result_template_version": 8,
            "velocity_lines": {"file": "proposal/autocfd_velocity_lines_nominal.csv", "sha256": sha256_file(lines_path)},
            "velocity_scoring_grid": {"file": "proposal/autocfd_velocity_samples_10mm.csv", "sha256": sha256_file(samples_path)},
        },
        "velocity_profiles": {
            "definition_authority": "AutoCFD5",
            "ranked_metric_id": "velocity_profile_r2",
            "report_only_metric_id": "velocity_profile_uinf_rmse",
            "ranked_reduction": "equal_case_equal_line_global_R2_with_normalized_trapezoidal_arc_length_support_per_line",
            "quantity": "magnitude(UMeanTrim)/38.889",
            "scoring_grid": "fixed_10_mm_candidate_grid",
            "line_count": len(velocity_stations),
            "sample_count_per_case": sum(station["sample_count"] for station in velocity_stations),
            "stations": velocity_stations,
        },
        "pressure_cuts": {
            "definition_authority": "FluidsBench",
            "ranked_metric_id": "cp_cut_r2",
            "report_only_metric_id": "cp_cut_rmse",
            "quantity": "Cp=2*pMeanTrim/(38.889^2)",
            "cut_count": len(pressure_cuts),
            "association": "native_surface_VTP_CellData",
            "extraction_status": "pending_immutable_owner_cut_support",
            "reduction": "equal_case_equal_cut_global_R2_with_normalized_native_intersection_segment_length_support_per_cut",
            "stations": pressure_cuts,
        },
        "activation_requirements": [
            "publish_all_case_containing_cell_assignments_and_hashes",
            "pass_velocity_1_2_5_10mm_resolution_convergence",
            "publish_immutable_native_surface_cp_cut_extraction_support",
            "pass_all_case_cp_cut_replay_and_chunk_invariance",
        ],
        "excluded_diagnostics": {
            "cp_probes": "the_209_discrete_taps_are_not_part_of_the_drivaerml_submission_or_score"
        },
    }


def write_splits() -> list[dict[str, Any]]:
    owner = load_json(PROPOSAL_ROOT / "owner-published-splits.json")
    entries = []
    for split_id, (label, case_set_id) in SPLITS.items():
        train = owner[f"{split_id}_train"]
        validation = owner[f"{split_id}_val"]
        test = owner[f"{split_id}_test"]
        split = {
            "schema_version": "1.0",
            "dataset_id": "drivaerml",
            "split_id": split_id,
            "case_set_id": case_set_id,
            "split_label": label,
            "case_id_status": "official",
            "training_case_count": len(train),
            "validation_case_count": len(validation),
            "case_count": len(test),
            "train_case_ids": train,
            "validation_case_ids": validation,
            "case_ids": test,
            "source": {
                "repository": "neashton/drivaerml",
                "revision": DATASET_REVISION,
                "manifest_path": "splits/manifest.json",
                "manifest_sha256": SPLIT_MANIFEST_SHA256,
                "manifest_keys": [f"{split_id}_train", f"{split_id}_val", f"{split_id}_test"],
            },
            "usage": {
                "train": "model fitting and training-only preprocessing statistics",
                "validation": "checkpoint and hyperparameter selection",
                "test": "evaluation only; no fitting, tuning, or preprocessing statistics",
            },
        }
        path = BENCHMARK_ROOT / "splits" / f"{split_id}.json"
        write_json(path, split)
        entries.append(
            {
                "id": split_id,
                "label": label,
                "index_file": f"splits/{split_id}.json",
                "training_case_count": len(train),
                "validation_case_count": len(validation),
                "case_count": len(test),
                "case_set_id": case_set_id,
                "case_id_status": "official",
                "sha256": sha256_file(path),
            }
        )
    return entries


def build_profile_panels(profile: dict[str, Any]) -> list[dict[str, Any]]:
    pressure = profile["pressure_cuts"]["stations"]
    velocity = profile["velocity_profiles"]["stations"]
    return [
        {
            "id": "pressure_profiles",
            "required": True,
            "allow_unlisted_stations": False,
            "minimum_points": 2,
            "coordinate_order": "strictly_increasing",
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
            "metric_id": "cp_cut_r2",
            "station_ids": [station["id"] for station in pressure],
            "quantity_ids": ["cp"],
        },
        {
            "id": "velocity_profiles",
            "required": True,
            "allow_unlisted_stations": False,
            "minimum_points": 2,
            "coordinate_order": "strictly_increasing",
            "coordinate_id": "distance_m",
            "coordinate_unit": "m",
            "metric_id": "velocity_profile_r2",
            "station_ids": [station["id"] for station in velocity],
            "quantity_ids": ["velocity_ratio"],
            "station_sample_counts": {station["id"]: station["sample_count"] for station in velocity},
            "station_coordinate_intervals": {station["id"]: station["coordinate_interval_m"] for station in velocity},
            "station_coordinate_spacings": {station["id"]: "uniform" for station in velocity},
        },
    ]


def build_composite() -> dict[str, Any]:
    bounded_errors = [
        ("surface_pressure_rel_l2", 0.15, 15.0),
        ("surface_wall_shear_rel_l2", 0.10, 20.0),
        ("volume_velocity_rel_l2", 0.15, 12.0),
        ("volume_pressure_rel_l2", 0.10, 15.0),
    ]
    bounded_quality = [
        ("cd_r2", 0.15),
        ("cl_r2", 0.05),
        ("c_pitch_r2", 0.05),
        ("velocity_profile_r2", 0.15),
        ("cp_cut_r2", 0.10),
    ]
    return {
        "metric_id": "overall_score",
        "operation": "weighted_component_scores",
        "status": "active",
        "score_range": [0.0, 100.0],
        "components": [
            {
                "metric_id": metric_id,
                "weight": weight,
                "transform": "bounded_error",
                "cap": cap,
            }
            for metric_id, weight, cap in bounded_errors
        ]
        + [
            {
                "metric_id": metric_id,
                "weight": weight,
                "transform": "bounded_quality",
            }
            for metric_id, weight in bounded_quality
        ],
        "tolerance": 1e-6,
    }


def build_velocity_validity_policy() -> dict[str, Any]:
    """Return the closed-candidate velocity support and failure policy.

    The coordinate registry stays separately hash-pinned.  This policy defines
    how a future owner mask must bind those coordinates without pretending that
    the pending all-case mask already exists.
    """

    return {
        "id": "drivaerml-autocfd5-velocity-validity-v1",
        "status": "candidate_policy_defined_owner_mask_pending",
        "authority": "dataset_owner",
        "participant_may_modify": False,
        "mask_scope": {
            "case_count": 484,
            "line_count": 16,
            "master_spacing_m": 0.001,
            "master_samples_per_case": 37416,
            "master_row_count": 18109344,
            "ranked_spacing_m": 0.01,
            "ranked_samples_per_case": 3756,
            "ranked_row_count": 1817904,
            "row_key": ["case_id", "profile_id", "master_sample_index"],
            "study_grid_derivation": {
                "1_mm": 1,
                "2_mm": 2,
                "5_mm": 5,
                "10_mm": 10,
            },
        },
        "mask_binding": {
            "status": "pending_immutable_owner_release_no_exclusions_approved",
            "file": None,
            "sha256": None,
        },
        "allowed_owner_exclusion_reasons": [
            "inside_morphed_solid",
            "outside_released_fluid_domain",
        ],
        "mapping_failures_are_not_automatic_exclusions": [
            "no_native_cell_within_closure_tolerance",
            "native_cell_closure_evaluation_failed_for_broad_phase_cells:<sorted_raw_ids>",
            "ambiguous_polyhedron_solid_angle_classification",
        ],
        "included_sample_rule": (
            "every_non_excluded_sample_must_select_one_deterministic_raw_vtk_cell_id"
        ),
        "unresolved_sample_rule": (
            "an_unmapped_non_excluded_sample_makes_its_line_and_case_ranked_velocity_component_unavailable"
        ),
        "excluded_sample_rule": (
            "retain_the_row_explicitly_but_give_the_point_and_its_adjacent_edges_zero_metric_support"
        ),
        "arc_rule": (
            "a_trapezoidal_edge_contributes_only_when_both_adjacent_endpoints_are_owner_included_and_mapped"
        ),
        "line_minimum_support": (
            "at_least_one_positive_length_owner_included_mapped_adjacent_edge"
        ),
        "case_availability": "all_16_required_line_metrics_must_be_available",
        "forbidden_operations": [
            "silent_omission",
            "snapping",
            "interpolation",
            "extrapolation",
            "gap_bridging",
            "participant_defined_mask",
        ],
        "polyhedron_classification": {
            "status": "candidate_selected_all_case_replay_and_owner_approval_pending",
            "spatial_boundary_test_precedes_winding": True,
            "spatial_boundary_tolerance_m": 1.0e-6,
            "absolute_solid_angle_target_inside_steradian": 4.0 * math.pi,
            "absolute_solid_angle_target_outside_steradian": 0.0,
            "absolute_tolerance_steradian": 1.0e-3,
            "intermediate_or_nonfinite_action": (
                "retain_explicit_unresolved_invalid_row_no_cell_assignment"
            ),
        },
        "activation_requirement": (
            "publish_and_hash_the_complete_owner_mask_and_all_case_assignments_with_zero_unresolved_non_excluded_samples"
        ),
    }


def build_scoring_support(profile_path: Path) -> dict[str, Any]:
    return {
        "status": "owner_review_required",
        "submissions_open": False,
        "closed_reason": (
            "The participant contract, all-case native evidence and profile truth, "
            "scientific review, and bounded scoring equation are complete, but "
            "ranking remains closed until the production evaluator/scoring-support/"
            "truth release identities are frozen, the three-model sensitivity gate "
            "passes, and an independent full schema-v3 submission dry run is approved."
        ),
        "scientific_approval": {
            "status": "approved",
            "approved_by": "neilashton",
            "approved_at": "2026-08-28",
            "approval_record": {
                "file": "evidence/owner-scientific-approval-2026-08-28.json",
                "sha256": OWNER_SCIENTIFIC_APPROVAL_SHA256,
                "schema": "drivaerml-owner-scientific-approval-v1",
            },
            "activation_effect": (
                "scientific_review_complete_release_gates_unchanged"
            ),
        },
        "owner_approved_successor_profile": {
            "id": PROFILE_DEFINITION_V10_ID,
            "file": "drivaerml-diagnostics-v10.json",
            "sha256": PROFILE_DEFINITION_V10_SHA256,
            "status": "scientifically_approved_pending_immutable_release_binding",
            "published_native_v3_truth_index_sha256": (
                NATIVE_V3_TRUTH_INDEX_SHA256
            ),
        },
        "candidate_evidence_index_file": "evidence/README.md",
        "candidate_evidence_manifest_file": "evidence/manifest.json",
        "dataset_evaluator_binding": {
            "status": "pending_frozen_release",
            "repository_url": "https://github.com/neilashton/fluidsbench-submission",
            "evaluator_reference_version": EVALUATOR_VERSION,
            "evaluator_code_revision": None,
            "activation_rule": (
                "set status=frozen and publish the exact immutable evaluator Git "
                "revision only after the production evaluator is scientifically "
                "approved; participant evaluation evidence must match the frozen "
                "reference version, and optional maintainer recomputation receipts "
                "must match both values exactly"
            ),
        },
        "source_release": {
            "provider": "Hugging Face Hub",
            "repository": "neashton/drivaerml",
            "revision": DATASET_REVISION,
            "public_case_count": 484,
            "native_source_pin": {"file": "proposal/native-source-pin.json", "sha256": SOURCE_PIN_SHA256},
            "owner_split_manifest": {"file": "proposal/owner-published-splits.json", "sha256": SPLIT_MANIFEST_SHA256},
            "proposal_contract": {"file": "proposal/contract-proposal.json", "sha256": sha256_file(PROPOSAL_ROOT / "contract-proposal.json")},
        },
        "case_file_binding": {
            "case_id_pattern": "^run_([1-9][0-9]*)$",
            "case_id_variable": "<case_id>",
            "run_number_variable": "<run_number>",
            "example": {"case_id": "run_11", "run_number": 11},
        },
        "coverage_contract": {
            "dimensionality": "3D native surface and 3D native finite-volume domain",
            "rule": "every_required_native_CellData_tuple_exactly_once",
            "inference_may_be_chunked": True,
            "chunk_metrics_must_use_additive_sufficient_statistics": True,
            "complete_case_and_entity_coverage_required": True,
            "full_prediction_artifact_required": False,
            "case_aggregation": "calculate_each_complete_case_then_macro_average_cases_equally",
            "vector_error": "componentwise_difference_then_euclidean_norm_not_difference_of_magnitudes",
        },
        "public_supports": [
            {
                "id": "surface_native_cells",
                "public_file": "<case_id>/boundary_<run_number>.vtp",
                "format": "VTK PolyData",
                "association": "CellData",
                "entities": "all_native_polygons_in_raw_VTK_cell_order",
                "stable_id": ["case_id", "zero_based_raw_vtk_cell_id"],
                "arrays": ["pMeanTrim", "wallShearStressMeanTrim"],
                "array_components": {"pMeanTrim": 1, "wallShearStressMeanTrim": 3},
                "array_units": {"pMeanTrim": "m^2/s^2", "wallShearStressMeanTrim": "m^2/s^2"},
                "physical_weight_file": "<case_id>/boundary_cell_area_<run_number>.npy",
                "physical_weight_dtype": "little_endian_float32",
                "physical_weight_association": "one_positive_finite_area_per_native_polygon_in_identical_order",
                "physical_weight_manifest": {"path": "surface_cell_areas/manifest.json", "sha256": SURFACE_AREA_MANIFEST_SHA256},
                "physical_weight_release": {
                    "status": "public_complete",
                    "dataset_revision": DATASET_REVISION,
                    "case_count": 484,
                    "payload_file_count": 484,
                    "native_polygon_count": 4_159_517_910,
                    "payload_bytes": 16_638_133_592,
                },
            },
            {
                "id": "volume_native_cells",
                "logical_public_file": "<case_id>/volume_<run_number>.vtu",
                "transport_parts": "resolve the exact ordered .00.part, .01.part, and optional .02.part list from proposal/native-source-pin.json",
                "transport_assembly": "byte_concatenate_pinned_parts_in_part_index_order_with_no_delimiter_or_transformation",
                "part_count_policy": "474 cases have two parts and 10 pinned cases have three; never assume exactly two",
                "format": "VTK UnstructuredGrid",
                "association": "CellData",
                "entities": "all_native_finite_volume_cells_in_raw_VTK_cell_order",
                "stable_id": ["case_id", "zero_based_raw_vtk_cell_id"],
                "arrays": ["UMeanTrim", "pMeanTrim"],
                "array_components": {"UMeanTrim": 3, "pMeanTrim": 1},
                "array_units": {"UMeanTrim": "m/s", "pMeanTrim": "m^2/s^2"},
                "weighting": "one_per_native_cell",
                "geometric_cell_volume_weights_required": False,
                "candidate_primary_validation_evidence": {
                    "file": "evidence/native-volume-equal-cell-primary-all484.json",
                    "sha256": "bda42a125ffb4d6484756e77ac7e495974f39d4bce3e663154322e9e560827c7",
                    "provenance_file": "evidence/native-volume-equal-cell-primary-all484-provenance.json",
                    "provenance_sha256": "b5ffe2234bb1597cf041ff5d97458f3d0d6e81db7a30591a7e26f51ebc032fde",
                    "artifact_schema": "drivaerml-native-volume-equal-cell-primary-all-case-audit-v1",
                    "legacy_unit_weight_vocabulary": True,
                    "current_tool_aggregate_schema": "drivaerml-native-volume-equal-cell-all-case-audit-v2",
                    "normalization_status": "bound_historical_equal_cell_evidence_current_v2_tools_require_regeneration_or_explicit_normalization",
                    "status": "all_484_equal_cell_primary_audit_passed_not_scoring_support",
                    "case_count": 484,
                    "verified_segment_count": 978,
                    "native_cell_count": 68949662110,
                    "equal_native_cell_weighting_exercised": True,
                    "complete_all_484_cases": True,
                    "owner_scientific_approval": True,
                },
                "candidate_primary_pilot_evidence": {
                    "file": "evidence/native-volume-run1-run44-equal-cell-primary-pilot.json",
                    "sha256": "d009b6ac708fa320d21492b4cc44fc846b61e99b445836b5dedc704a934592d7",
                    "status": "superseded_scope_pilot_retained_for_two_part_three_part_implementation_reference",
                },
            },
            {
                "id": "field_integrated_force_truth",
                "authoritative_public_file": "force_mom_constref_all.csv",
                "sha256": FORCE_TABLE_SHA256,
                "prototype_r2_truth_statistics": {
                    "file": "force-r2-truth-statistics.json",
                    "sha256": FORCE_R2_STATISTICS_SHA256,
                    "scope": "prototype_fixture_calibration_only; official_evaluation_uses_casewise_authoritative_truth",
                },
                "per_case_mirror": "<case_id>/force_mom_constref_<run_number>.csv",
                "per_case_mirror_policy": "accepted_as_a_convenience_only_after_exact_row_replay_against_the_authoritative_aggregate",
                "exact_header": ["run", "cd", "cl", "clf", "clr", "cs"],
                "ranked_targets": {"Cd": "cd", "Cl": "cl", "CmPitch": "casewise_(clf-clr)/2"},
                "report_only_targets": {"Clf": "clf", "Clr": "clr"},
            },
        ],
        "scored_field_bindings": [
            {
                "target_id": "surface_pressure",
                "support_id": "surface_native_cells",
                "array": "pMeanTrim",
                "components": 1,
                "primary_metric": "surface_pressure_rel_l2",
                "primary_weighting": "surface_face_area",
                "secondary_metric": "surface_pressure_equal_entity_rel_l2",
                "secondary_weighting": "surface_entities_equal",
            },
            {
                "target_id": "surface_wall_shear",
                "support_id": "surface_native_cells",
                "array": "wallShearStressMeanTrim",
                "components": 3,
                "primary_metric": "surface_wall_shear_rel_l2",
                "primary_weighting": "surface_face_area",
                "secondary_metric": "surface_wall_shear_equal_entity_rel_l2",
                "secondary_weighting": "surface_entities_equal",
            },
            {
                "target_id": "volume_velocity",
                "support_id": "volume_native_cells",
                "array": "UMeanTrim",
                "components": 3,
                "primary_metric": "volume_velocity_rel_l2",
                "primary_weighting": "volume_cells_equal",
            },
            {
                "target_id": "volume_pressure",
                "support_id": "volume_native_cells",
                "array": "pMeanTrim",
                "components": 1,
                "primary_metric": "volume_pressure_rel_l2",
                "primary_weighting": "volume_cells_equal",
            },
        ],
        "relative_l2_policy": {
            "surface_primary": "surface_face_area",
            "surface_secondary": "equal_native_polygon",
            "volume": "equal_native_cell_only",
            "chunk_rule": "sum_case_numerators_denominators_counts_and_weights_across_chunks_then_apply_one_nonlinear_reduction",
            "forbidden_chunk_rule": "average_chunk_local_metric_values",
        },
        "force_integration": {
            "source_fields": ["surface pMeanTrim", "surface wallShearStressMeanTrim"],
            "freestream_velocity_m_per_s": 38.889,
            "density_kg_per_m3": 1.0,
            "reference_area_m2": 2.17,
            "reference_length_m": 2.78618,
            "centre_of_rotation_m": [1.40009, 0.0, -0.3176],
            "drag_axis": "+x",
            "lift_axis": "+z",
            "pitch_axis": "+y",
            "body_force_equation": "rho_inf*sum_f(pMeanTrim_f*A_f-wallShearStressMeanTrim_f*norm(A_f))",
            "body_moment_equation": "rho_inf*sum_f(cross(C_f-CoR,pMeanTrim_f*A_f-wallShearStressMeanTrim_f*norm(A_f)))",
            "face_area_and_centre_convention": "OpenFOAM_v2212_primitiveMeshTools_makeFaceCentresAndAreas",
            "ranked_reduction": "separate_equal_case_R2_for_Cd_Cl_and_CmPitch_with_RMSE_reported_as_diagnostics",
            "dependent_axle_loads": {"Clf": "Cl/2+CmPitch", "Clr": "Cl/2-CmPitch", "composite_weight": 0.0},
            "participant_json": {
                "file": "metrics/cases.json",
                "case_property": "force_coefficients",
                "required_prediction_keys": ["cd", "cl", "cm_pitch", "clf", "clr"],
                "ranked_keys": ["cd", "cl", "cm_pitch"],
                "report_only_keys": ["clf", "clr"],
                "generation": "participant_runs_the_frozen_reference_evaluator_locally",
            },
            "candidate_validation_evidence": {
                "file": "evidence/force-replay-all484.json",
                "sha256": "631cd02c3a4215b254489652c1d93dfd781ecdb9478ff1ab11f24294743e8a17",
                "status": "all_484_cases_passed_candidate_evaluator",
                "owner_scientific_approval": True,
            },
        },
        "profile_definition": {
            "file": str(profile_path.relative_to(BENCHMARK_ROOT)),
            "sha256": sha256_file(profile_path),
            "status": "candidate_velocity_and_continuous_cp_cut_support_pending_all_case_validation",
            "velocity_validity_policy": build_velocity_validity_policy(),
            "cp_cut_support": {
                "status": "pending_immutable_owner_release",
                "cut_count": 4,
                "metric_id": "cp_cut_r2",
                "discrete_cp_probe_support_accepted": False,
            },
        },
        "participant_process": [
            "select one official split and use only its train list for fitting and training statistics",
            "load every required boundary VTP CellData tuple and its same-order public area array",
            "reconstruct each logical volume VTU from the exact per-case part list; process native cells in bounded-memory chunks without resampling",
            "retain raw native cell IDs so chunks form one complete duplicate-free case partition",
            "accumulate additive sufficient statistics per chunk and reduce only after the complete case is assembled logically",
            "derive field-integrated forces, AutoCFD5 velocity diagnostics, and FluidsBench continuous Cp cuts from the participant's complete native predictions using the candidate evaluator for implementation evidence; official submissions must use the future frozen owner-approved evaluator",
            "record the repository-wide schema-v3 FluidsBench methodology disclosure and DrivAerML output-field contract with named architecture components, exact total and submitter-trainable parameter counts, scoped inputs and outputs, data handling, every submitter or upstream training stage, compute for every submitter-performed stage, every raw loaded checkpoint-file SHA-256 and selection rule, and complete-split inference compute; checkpoint publication remains optional",
            "submit participant-authored scalar metrics, per-case force coefficients and evidence, and complete velocity-profile and continuous-Cp-cut JSON chunks through the current FluidsBench schema; sharing revision-pinned prediction artifacts and maintainer native-evaluator recomputation are optional audits and do not affect approval, rank, citation, or promotion eligibility",
        ],
        "activation_gates": {
            "official_splits": "complete",
            "pinned_native_files": "complete",
            "surface_area_weights": "complete_public_all_484_pinned_order_count_hash_and_value_audit_passed",
            "volume_field_weighting": "complete_equal_native_cell_no_geometric_cell_volume_weights_required",
            "force_evaluator": "all_484_candidate_replay_passed_owner_scientifically_approved_release_binding_pending",
            "velocity_profiles": "all_484_fixed_and_relative_native_truth_and_support_published_owner_scientifically_approved_release_binding_pending",
            "cp_cuts": "all_484_four_cut_native_truth_and_support_published_owner_scientifically_approved_release_binding_pending",
            "bounded_score_definition": "complete_fixed_field_error_caps_and_bounded_force_profile_R2",
            "composite_sensitivity_and_bootstrap": "pending_frozen_evaluator_and_at_least_three_genuine_model_checkpoint_predictions",
            "schema_v3_participant_result_binding": "candidate_fail_closed_participant_authored_reductions_case_metrics_and_profile_hash_bindings_implemented_tests_pass_scientifically_approved_frozen_release_identity_pending",
            "independent_participant_dry_run": "pending",
            "owner_evaluator_approval": "scientific_method_approved_final_immutable_release_binding_pending",
        },
        "owner_decisions_required": [
            "provide_at_least_three_genuine_trained_model_checkpoint_predictions_for_sensitivity_and_method_ordering",
            "review_fixed_field_error_caps_and_bounded_R2_sensitivity_with_genuine_model_predictions_and_publish_bootstrap_indexes",
            "approve_immutable_evaluator_and_scoring_support_release_before_opening_submissions",
        ],
    }


def build_specification(
    split_entries: list[dict[str, Any]], profile_path: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    composite = build_composite()
    return {
        "schema_version": "1.1",
        "dataset_id": "drivaerml",
        "dataset_name": "DrivAerML",
        "dataset_version": DATASET_VERSION,
        "status": "candidate_scoring_contract",
        "contract_base_path": "benchmark-specs/drivaerml/",
        "contract_base_path_scope": "repository_relative_contract_artifact_fields_only; upstream_dataset_paths_resolve_within_source_release",
        "scoring_support": build_scoring_support(profile_path),
        "relative_diagnostics": {
            "status": "support_pending",
            "profile_format_enabled": False,
            "closed_reason": (
                "genuine-model sensitivity, owner approval, and an immutable "
                "evaluator revision are not all bound"
            ),
            "contract": {
                "id": RELATIVE_DIAGNOSTICS_V3_ID,
                "file": "drivaerml-relative-diagnostics-v3.json",
                "sha256": RELATIVE_DIAGNOSTICS_V3_SHA256,
            },
            "profile_chunk": {
                "format": (
                    "fluidsbench-drivaerml-relative-profile-chunks-v3-candidate"
                ),
                "schema_file": (
                    "schemas/v1/drivaerml-relative-profile-chunk.schema.json"
                ),
                "schema_sha256": RELATIVE_PROFILE_SCHEMA_SHA256,
                "series_per_case": 40,
            },
            "constant_scoring_policy": "unchanged",
            "relative_scoring_role": "report_only",
            "relative_composite_weight": 0.0,
        },
        "evaluation_reference_version": EVALUATOR_VERSION,
        "default_field_reduction": "per_geometry_then_macro_average",
        "ranking": {
            "metric_id": "overall_score",
            "direction": "higher",
            "decimal_places": 1,
            "rounding": "decimal_half_up",
            "method": "competition",
        },
        "overall_score_composite": composite,
        "component_score_groups": {
            "operation": "normalized_weighted_component_scores",
            "groups": [
                {
                    "metric_id": "field_score",
                    "component_metric_ids": [
                        "surface_pressure_rel_l2",
                        "surface_wall_shear_rel_l2",
                        "volume_velocity_rel_l2",
                        "volume_pressure_rel_l2",
                    ],
                },
                {
                    "metric_id": "force_score",
                    "component_metric_ids": [
                        "cd_r2",
                        "cl_r2",
                        "c_pitch_r2",
                    ],
                },
                {
                    "metric_id": "diagnostic_score",
                    "component_metric_ids": ["velocity_profile_r2", "cp_cut_r2"],
                },
            ],
            "tolerance": 1e-6,
        },
        "metrics": build_metrics(),
        "profile_definition": {
            "id": profile["id"],
            "file": str(profile_path.relative_to(BENCHMARK_ROOT)),
            "sha256": sha256_file(profile_path),
            "status": "candidate_velocity_and_continuous_cp_cut_support_pending_all_case_validation",
            "velocity_validity_policy_id": (
                "drivaerml-autocfd5-velocity-validity-v1"
            ),
        },
        "profile_panels": build_profile_panels(profile),
        "splits": split_entries,
    }


def select_generation_profile(
    release_bindings: dict[str, Any],
    existing_specification: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    """Select v9 now, but never regress an already bound v10 profile."""

    if release_bindings["status"] == "ready":
        return validate_profile_v10_binding(
            release_bindings["profile_definition_v10"]
        )

    preserved = validate_preserved_release_managed_bindings(existing_specification)
    if preserved is None:
        profile = build_profile_definition()
        profile_path = BENCHMARK_ROOT / "drivaerml-diagnostics-v9.json"
        write_json(profile_path, profile)
        return profile_path, profile
    return validate_profile_v10_binding(preserved["profile_definition_v10"])


def validate_profile_v10_binding(
    binding: dict[str, Any],
    expected_definition: dict[str, Any] | None = None,
    *,
    benchmark_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Read, hash, and parse the exact contained profile-v10 artifact."""

    if not isinstance(binding, dict) or set(binding) != {"file", "sha256"}:
        raise ValueError("profile_definition_v10 binding must be an object")
    file_value = binding.get("file")
    digest = binding.get("sha256")
    if (
        not isinstance(file_value, str)
        or "v10" not in Path(file_value).name
        or not isinstance(digest, str)
        or not SHA256_PATTERN.fullmatch(digest)
    ):
        raise ValueError("profile_definition_v10 binding is incomplete or invalid")
    profile_path = resolved_benchmark_file(
        file_value,
        label="profile_definition_v10.file",
        benchmark_root=benchmark_root,
    )
    profile, actual_sha256 = load_json_with_sha256(
        profile_path,
        label="profile-definition-v10 artifact",
    )
    if actual_sha256 != digest:
        raise ValueError(
            f"release-managed profile SHA-256 changed for {profile_path}: "
            f"{actual_sha256}"
        )
    if (
        not isinstance(profile, dict)
        or profile.get("id") != PROFILE_DEFINITION_V10_ID
    ):
        raise ValueError(
            "profile-definition-v10 artifact id must equal "
            f"{PROFILE_DEFINITION_V10_ID!r}"
        )
    if expected_definition is not None:
        for key, expected in (
            ("file", file_value),
            ("sha256", digest),
            ("id", profile["id"]),
        ):
            if expected_definition.get(key) != expected:
                raise ValueError(
                    f"active profile_definition.{key} does not match the bound "
                    "profile-v10 artifact"
                )
    return profile_path, profile


def validate_profile_ground_truth_binding(binding: dict[str, Any]) -> None:
    if (
        not isinstance(binding, dict)
        or set(binding) != {"release_id", "manifest_sha256"}
        or not isinstance(binding.get("release_id"), str)
        or not SAFE_ID_PATTERN.fullmatch(binding["release_id"])
        or not isinstance(binding.get("manifest_sha256"), str)
        or not SHA256_PATTERN.fullmatch(binding["manifest_sha256"])
    ):
        raise ValueError("profile-ground-truth binding is incomplete or invalid")
    leaderboard = load_json(MANIFEST_PATH)
    global_profile_truth = (
        leaderboard.get("data_release", {}).get("profile_ground_truth")
        if isinstance(leaderboard, dict)
        else None
    )
    if not isinstance(global_profile_truth, dict):
        raise ValueError("leaderboard manifest has no global profile-ground-truth binding")
    for key in ("release_id", "manifest_sha256"):
        if binding.get(key) != global_profile_truth.get(key):
            raise ValueError(
                "DrivAerML candidate profile ground truth must match the global "
                f"leaderboard binding for {key}"
            )


def validate_preserved_release_managed_bindings(
    existing_specification: dict[str, Any] | None,
    *,
    target_specification: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a coherent prior release state or reject partial/stale state.

    When a target is supplied, the old release must satisfy the newly generated
    contract before any whitelisted value is materialized into that contract.
    """

    if not isinstance(existing_specification, dict):
        return None
    existing_support = existing_specification.get("scoring_support")
    existing_profile = existing_specification.get("profile_definition")
    if not isinstance(existing_support, dict):
        return None
    candidate = existing_support.get("candidate_manifest")
    evaluator = existing_support.get("dataset_evaluator_binding")
    support_profile = existing_support.get("profile_definition")
    has_managed_state = any(
        (
            isinstance(candidate, dict),
            isinstance(evaluator, dict) and evaluator.get("status") == "frozen",
            isinstance(existing_profile, dict)
            and "v10" in Path(str(existing_profile.get("file", ""))).name,
            isinstance(support_profile, dict)
            and isinstance(support_profile.get("profile_ground_truth"), dict),
        )
    )
    if not has_managed_state:
        return None
    preserved_source = {
        "candidate_manifest": candidate,
        "dataset_evaluator_binding": evaluator,
        "profile_definition": existing_profile,
        "scoring_support_profile_definition": support_profile,
    }
    tokens = unresolved_release_tokens(preserved_source)
    if tokens:
        rendered = [".".join(map(str, location)) for location, _token in tokens]
        raise ValueError(
            "preserved release-managed state contains unresolved release tokens: "
            + ", ".join(rendered)
        )
    if not isinstance(candidate, dict) or set(candidate) != {
        "status",
        "release_id",
        "manifest_file",
        "manifest_url",
        "manifest_sha256",
    }:
        raise ValueError("preserved candidate manifest binding is incomplete")
    if (
        candidate.get("status") != "candidate"
        or not isinstance(candidate.get("release_id"), str)
        or not SAFE_ID_PATTERN.fullmatch(candidate["release_id"])
        or not isinstance(candidate.get("manifest_sha256"), str)
        or not SHA256_PATTERN.fullmatch(candidate["manifest_sha256"])
    ):
        raise ValueError("preserved candidate manifest binding is invalid")
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("status") != "frozen"
        or evaluator.get("evaluator_reference_version") != EVALUATOR_VERSION
        or not GIT_REVISION_PATTERN.fullmatch(
            str(evaluator.get("evaluator_code_revision", ""))
        )
    ):
        raise ValueError("preserved candidate evaluator binding is incomplete or stale")
    if not isinstance(existing_profile, dict) or not isinstance(support_profile, dict):
        raise ValueError("preserved profile-v10 binding is incomplete")
    for key in ("file", "sha256"):
        if support_profile.get(key) != existing_profile.get(key):
            raise ValueError(
                "preserved scoring-support and active profile-v10 bindings disagree"
            )
    profile_binding = {
        "file": existing_profile.get("file"),
        "sha256": existing_profile.get("sha256"),
    }
    validation_specification = (
        target_specification
        if isinstance(target_specification, dict)
        else existing_specification
    )
    validation_profile = validation_specification.get("profile_definition")
    if not isinstance(validation_profile, dict):
        raise ValueError("target contract has no valid profile-v10 definition")
    validate_profile_v10_binding(profile_binding, validation_profile)
    ground_truth = support_profile.get("profile_ground_truth")
    validate_profile_ground_truth_binding(ground_truth)
    preserved = {
        "schema": "drivaerml-fluidsbench-candidate-release-bindings-v1",
        "status": "ready",
        "unresolved_token_prefix": UNRESOLVED_RELEASE_PREFIX,
        "candidate_manifest": copy.deepcopy(candidate),
        "evaluator": {
            "reference_version": evaluator["evaluator_reference_version"],
            "code_revision": evaluator["evaluator_code_revision"],
        },
        "profile_definition_v10": profile_binding,
        "profile_ground_truth": copy.deepcopy(ground_truth),
    }
    validate_ready_candidate_manifest(preserved, validation_specification)
    return preserved


def preserve_relative_activation_declaration(
    specification: dict[str, Any],
    existing_specification: dict[str, Any] | None,
) -> None:
    """Preserve only a fully verified relative activation-release declaration.

    The activation record contains the evaluator revision, so it is added in a
    follow-up commit and cannot be hard-coded into the evaluator commit without
    creating a self-reference.  A later promotion rerun must nevertheless keep
    that reviewed record.  Validate its complete local chain before copying the
    lifecycle fields; ordinary unresolved or absent declarations are ignored.
    """

    if not isinstance(existing_specification, dict):
        return
    existing = existing_specification.get("relative_diagnostics")
    if not isinstance(existing, dict) or "activation_release" not in existing:
        return

    # Local import avoids coupling the normal promotion module import path to
    # the optional submission-schema runtime.
    from scripts.validate_submission import (
        validate_drivaerml_relative_activation_release,
    )

    problems: list[str] = []
    claims_activation = (
        existing.get("status") == "activated"
        or existing.get("profile_format_enabled") is True
    )
    repository_root = BENCHMARK_ROOT.parents[1]
    if not validate_drivaerml_relative_activation_release(
        problems.append,
        existing_specification,
        existing,
        repository_root=repository_root,
        require_active=claims_activation,
    ):
        raise ValueError(
            "existing relative activation release is invalid: "
            + "; ".join(problems)
        )

    target = specification.get("relative_diagnostics")
    if not isinstance(target, dict):
        raise ValueError("generated specification has no relative_diagnostics")
    for field in (
        "status",
        "profile_format_enabled",
        "closed_reason",
        "activation_release",
    ):
        if field in existing:
            target[field] = copy.deepcopy(existing[field])


def apply_release_managed_bindings(
    specification: dict[str, Any],
    release_bindings: dict[str, Any],
    existing_specification: dict[str, Any] | None,
) -> dict[str, Any]:
    """Materialize only real bindings and preserve reviewed bindings on rerun.

    The unresolved hand-off file is never copied into the active specification.
    This prevents token-shaped pseudo releases while also ensuring that a later
    candidate manifest or v10 ground-truth binding is not erased by rerunning
    this older promotion generator.
    """

    preserve_relative_activation_declaration(
        specification,
        existing_specification,
    )
    support = specification["scoring_support"]
    if release_bindings["status"] == "ready":
        validate_ready_candidate_manifest(release_bindings, specification)
        validate_profile_v10_binding(
            release_bindings["profile_definition_v10"],
            specification.get("profile_definition"),
        )
        validate_profile_ground_truth_binding(
            release_bindings["profile_ground_truth"]
        )
        support["candidate_manifest"] = copy.deepcopy(
            release_bindings["candidate_manifest"]
        )
        support["dataset_evaluator_binding"].update(
            {
                "status": "frozen",
                "evaluator_code_revision": release_bindings["evaluator"][
                    "code_revision"
                ],
            }
        )
        support["profile_definition"]["profile_ground_truth"] = copy.deepcopy(
            release_bindings["profile_ground_truth"]
        )
        return specification

    preserved = validate_preserved_release_managed_bindings(
        existing_specification,
        target_specification=specification,
    )
    if preserved is None:
        return specification
    support["candidate_manifest"] = copy.deepcopy(
        preserved["candidate_manifest"]
    )
    support["dataset_evaluator_binding"].update(
        {
            "status": "frozen",
            "evaluator_code_revision": preserved["evaluator"]["code_revision"],
        }
    )
    support["profile_definition"]["profile_ground_truth"] = copy.deepcopy(
        preserved["profile_ground_truth"]
    )
    return specification


def presentation_definition(metric_spec: dict[str, Any]) -> dict[str, Any]:
    metric_id = metric_spec["id"]
    labels = {
        "cd_r2": "Field-integrated C_D R2",
        "cl_r2": "Field-integrated C_L R2",
        "c_pitch_r2": "Field-integrated C_M,pitch R2",
        "velocity_profile_r2": "AutoCFD5 velocity-profile R2",
        "cp_cut_r2": "Continuous native-surface Cp-cut R2",
        "field_integrated_cd_rmse": "Field-integrated C_D RMSE",
        "field_integrated_cl_rmse": "Field-integrated C_L RMSE",
        "field_integrated_cmpitch_rmse": "Field-integrated C_M,pitch RMSE",
        "field_integrated_clf_rmse": "Field-integrated C_Lf RMSE",
        "field_integrated_clr_rmse": "Field-integrated C_Lr RMSE",
        "field_integrated_lift_closure_max_abs": "Lift closure max. error",
        "velocity_profile_uinf_rmse": "AutoCFD5 velocity-profile RMSE",
        "velocity_profile_experimental_subset_uinf_rmse": "Experimental-line velocity RMSE",
        "cp_cut_rmse": "Continuous native-surface Cp-cut RMSE",
    }
    if metric_id.startswith("drivaerml_"):
        readable = metric_id.removeprefix("drivaerml_").replace("_", " ")
        label = f"DrivAerML {readable}"
        column_group = "absolute"
        group = "absolute-rmse" if metric_id.endswith("rmse") else "absolute-mae"
        group_label = "Absolute RMSE" if metric_id.endswith("rmse") else "Absolute MAE"
        digits = 3
    else:
        label = labels[metric_id]
        column_group = "diagnostics" if "profile" in metric_id or metric_id.startswith("cp_") else "integral"
        quality = metric_spec["kind"] == "r2"
        group = (
            "profile-quality" if quality and column_group == "diagnostics"
            else "force-quality" if quality
            else "profile-errors" if column_group == "diagnostics"
            else "force-errors"
        )
        group_label = (
            "Profile R2" if quality and column_group == "diagnostics"
            else "Integral force / moment R2" if quality
            else "Profile errors" if column_group == "diagnostics"
            else "Integral force / moment errors"
        )
        digits = 5 if metric_spec["unit"] == "" else 3
    return {
        "id": metric_id,
        "label": label,
        "description": (
            f"DrivAerML candidate contract metric using {metric_spec['weighting']}. "
            f"{'Higher' if metric_spec['direction'] == 'higher' else 'Lower'} is better."
        ),
        "group": group,
        "group_label": group_label,
        "column_group": column_group,
        "unit": metric_spec["unit"],
        "digits": digits,
        "direction": metric_spec["direction"],
        "kind": metric_spec["kind"],
        "equation": metric_spec["equation"],
        "comparison_group": group,
        "comparison_group_label": group_label,
        "column_group_label": "Diagnostics" if column_group == "diagnostics" else ("Absolute" if column_group == "absolute" else "Integral forces / moments"),
    }


def diagnostic_panels(profile: dict[str, Any]) -> list[dict[str, Any]]:
    velocity_source_url = "https://autocfd5.s3.eu-west-1.amazonaws.com/"
    cut_source_url = "https://github.com/neilashton/fluidsbench-submission"
    pressure_stations = [
        {
            "id": station["id"],
            "label": station["label"],
            "x_label": "Arc length along continuous native-surface cut, m",
            "description": "Continuous native-surface Cp cut on published owner-approved all-case support; immutable production release binding is pending.",
            "basis": "candidate_fluidsbench_v9_continuous_cp_cut",
            "source_url": cut_source_url,
        }
        for station in profile["pressure_cuts"]["stations"]
    ]
    velocity_stations = [
        {
            "id": station["id"],
            "label": station["label"],
            "x_label": "Distance along line, m",
            "description": f"Pinned 10 mm AutoCFD5 line with {station['sample_count']} samples.",
            "basis": "pinned_autocfd5_v9_velocity_line",
            "source_url": velocity_source_url,
        }
        for station in profile["velocity_profiles"]["stations"]
    ]
    return [
        {
            "id": "pressure_profiles",
            "title": "Continuous pressure-coefficient cuts",
            "description": "Four continuous native-surface Cp cuts; the 209 discrete AutoCFD taps are excluded.",
            "data_key": "cp_cuts",
            "profile_definition_id": profile["id"],
            "coordinate_unit": "m",
            "x_keys": ["arc_length_m", "distance", "x"],
            "quantities": [{"id": "cp", "label": "Cp", "y_label": "Pressure coefficient, Cp", "y_keys": ["cp"], "unit": ""}],
            "stations": pressure_stations,
            "required": True,
            "allow_unlisted_stations": False,
            "required_series_fields": ["case_id", "station_id"],
            "reverse_y": True,
        },
        {
            "id": "velocity_profiles",
            "title": "AutoCFD5 velocity profiles",
            "description": "Sixteen fixed lines scored on the pinned 10 mm grid using equal-line global R2 with normalized trapezoidal arc-length support.",
            "data_key": "velocity_profiles",
            "profile_definition_id": profile["id"],
            "coordinate_unit": "m",
            "x_keys": ["distance_m", "distance", "x"],
            "quantities": [{"id": "velocity_ratio", "label": "|U|/Uinf", "y_label": "Velocity magnitude / freestream", "y_keys": ["velocity_ratio", "u_over_u_inf"], "unit": ""}],
            "stations": velocity_stations,
            "required": True,
            "allow_unlisted_stations": False,
            "required_series_fields": ["case_id", "station_id"],
        },
    ]


def update_manifest(specification: dict[str, Any], profile: dict[str, Any]) -> None:
    manifest = load_json(MANIFEST_PATH)
    catalog = manifest["metric_catalog"]
    upsert_catalog_entries(
        catalog["dimensional_fields"], DRIVAER_DIMENSIONAL_CATALOG
    )
    upsert_catalog_entries(
        catalog["coefficient_errors"], DRIVAER_COEFFICIENT_CATALOG
    )
    definitions = manifest["metric_definitions"]
    definitions[:] = [
        definition
        for definition in definitions
        if definition.get("id") not in RETIRED_DRIVAER_PHYSICAL_VOLUME_METRIC_IDS
    ]
    existing_ids = {definition["id"] for definition in definitions}
    for metric_spec in specification["metrics"]:
        metric_id = metric_spec["id"]
        if metric_id in existing_ids:
            continue
        definition = presentation_definition(metric_spec)
        definitions.append(definition)
        existing_ids.add(metric_id)

    dataset = next(item for item in manifest["datasets"] if item["slug"] == "drivaerml")
    dataset["contract_base_path"] = specification["contract_base_path"]
    dataset["contract_base_path_scope"] = specification["contract_base_path_scope"]
    fixture_documents = [
        load_json(path)
        for path in sorted(SUBMISSIONS_ROOT.glob("*/submission.json"))
    ]
    if not fixture_documents or any(
        not isinstance(document.get("submitted_at"), str)
        for document in fixture_documents
    ):
        raise ValueError("DrivAerML prototype fixtures require submitted_at dates")
    fixture_count = len(fixture_documents)
    dataset["submission_count"] = fixture_count
    dataset["updated_at"] = max(
        document["submitted_at"] for document in fixture_documents
    )
    dataset["submission_population"] = {
        "feed_row_count": fixture_count,
        "prototype_ineligible_fixture_count": fixture_count,
        "eligible_submission_count": 0,
        "scientific_result_count": 0,
        "submission_count_semantics": (
            "all_current_feed_rows_are_structural_prototype_fixtures_not_eligible_submissions"
        ),
        "updated_at_semantics": (
            "latest_submitted_at_date_among_prototype_fixture_rows_not_a_contract_or_scientific_evidence_update"
        ),
    }
    dataset["submission_format"] = "drivaerml_native_candidate_v3"
    dataset["metrics"] = {
        "dimensional_fields": [
            entry["id"] for entry in DRIVAER_DIMENSIONAL_CATALOG
        ],
        "coefficient_errors": [
            entry["id"] for entry in DRIVAER_COEFFICIENT_CATALOG
        ],
    }
    dataset["metric_ids"] = [item["id"] for item in specification["metrics"]]
    dataset["scoring_support"] = specification["scoring_support"]
    dataset["overall_score_composite"] = specification[
        "overall_score_composite"
    ]
    dataset["component_score_groups"] = specification["component_score_groups"]
    dataset["profile_definition"] = specification["profile_definition"]
    dataset["metric_definition_overrides"] = {
        "surface_pressure_rel_l2": {
            "label": "Surface p rel. L2 (area-weighted)",
            "description": "Per-case area-weighted relative L2 of native CellData pMeanTrim, then equal-case mean. Kinematic pressure is in m^2/s^2.",
        },
        "surface_wall_shear_rel_l2": {
            "label": "Surface WSS rel. L2 (area-weighted)",
            "description": "Per-case area-weighted relative L2 of the native three-component wallShearStressMeanTrim vector, then equal-case mean.",
        },
        "volume_velocity_rel_l2": {
            "label": "Volume velocity rel. L2 (equal-cell)",
            "description": "Per-case equal-native-cell relative L2 of the three-component UMeanTrim vector, then equal-case mean.",
        },
        "volume_pressure_rel_l2": {
            "label": "Volume p rel. L2 (equal-cell)",
            "description": "Per-case equal-native-cell relative L2 of CellData pMeanTrim, then equal-case mean.",
        },
    }
    dataset["diagnostic_panels"] = diagnostic_panels(profile)
    write_json(MANIFEST_PATH, manifest)


def first_and_last(series: dict[str, Any] | None, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(series, dict):
        predictions = series.get("prediction")
        if isinstance(predictions, list) and len(predictions) >= 2:
            return float(predictions[0]), float(predictions[-1])
    return fallback


def migrated_series(case: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    existing = [series for series in case.get("series", []) if isinstance(series, dict)]
    pressure_sources = [series for series in existing if series.get("panel_id") == "pressure_profiles"]
    velocity_sources = [series for series in existing if series.get("panel_id") == "velocity_profiles"]
    result = []
    for index, station in enumerate(profile["pressure_cuts"]["stations"]):
        exact = next((series for series in pressure_sources if series.get("station_id") == station["id"]), None)
        source = exact or (pressure_sources[index % len(pressure_sources)] if pressure_sources else None)
        left, right = first_and_last(source, (0.0, 0.0))
        offset = 0.0 if exact is not None else 0.0002 * index
        result.append(
            {
                "panel_id": "pressure_profiles",
                "station_id": station["id"],
                "quantity_id": "cp",
                "coordinate": [0.0, 1.0],
                "prediction": [round(left + offset, 7), round(right + offset, 7)],
            }
        )
    for index, station in enumerate(profile["velocity_profiles"]["stations"]):
        exact = next((series for series in velocity_sources if series.get("station_id") == station["id"]), None)
        source = exact or (velocity_sources[index % len(velocity_sources)] if velocity_sources else None)
        left, right = first_and_last(source, (1.0, 1.0))
        offset = 0.0 if exact is not None else 0.0002 * index
        result.append(
            {
                "panel_id": "velocity_profiles",
                "station_id": station["id"],
                "quantity_id": "velocity_ratio",
                "coordinate": station["coordinate_interval_m"],
                "prediction": [round(left + offset, 7), round(right + offset, 7)],
            }
        )
    return result


def value(values: dict[str, Any], metric_id: str, fallback_id: str | None, default: float) -> float:
    candidate = values.get(metric_id)
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
        candidate = values.get(fallback_id) if fallback_id else None
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool) or not math.isfinite(candidate):
        candidate = default
    return max(0.0, float(candidate))


def migrated_metric_values(old: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {metric_id: 0.0 for metric_id in SCORE_EQUATIONS}
    primaries = {
        "surface_pressure_rel_l2": 5.0,
        "surface_wall_shear_rel_l2": 8.0,
        "volume_velocity_rel_l2": 6.0,
        "volume_pressure_rel_l2": 7.0,
    }
    for metric_id, default in primaries.items():
        result[metric_id] = value(old, metric_id, None, default)
    result["surface_pressure_equal_entity_rel_l2"] = value(old, "surface_pressure_equal_entity_rel_l2", "surface_pressure_rel_l2", 5.0)
    result["surface_wall_shear_equal_entity_rel_l2"] = value(old, "surface_wall_shear_equal_entity_rel_l2", "surface_wall_shear_rel_l2", 8.0)

    old_absolute = {
        "surface_pressure": ("surface_pressure_mae", "surface_pressure_rmse", 25.0, 35.0),
        "surface_wall_shear": ("surface_wall_shear_mae", "surface_wall_shear_rmse", 0.8, 1.2),
        "volume_velocity": ("volume_velocity_mae", "volume_velocity_rmse", 2.0, 3.0),
        "volume_pressure": ("volume_pressure_mae", "volume_pressure_rmse", 35.0, 50.0),
    }
    tokens = {
        "surface_pressure": ("area", "equal_entity"),
        "surface_wall_shear": ("area", "equal_entity"),
        "volume_velocity": ("equal_entity",),
        "volume_pressure": ("equal_entity",),
    }
    for field_id, (old_mae, old_rmse, default_mae, default_rmse) in old_absolute.items():
        for token in tokens[field_id]:
            prefix = f"drivaerml_{field_id}_{token}"
            result[f"{prefix}_mae"] = value(old, f"{prefix}_mae", old_mae, default_mae)
            result[f"{prefix}_rmse"] = value(old, f"{prefix}_rmse", old_rmse, default_rmse)

    cd = value(old, "field_integrated_cd_rmse", "c_drag_mae", 0.005)
    cl = value(old, "field_integrated_cl_rmse", "c_lift_mae", 0.01)
    result.update(
        {
            "cd_r2": value(old, "cd_r2", None, 0.8),
            "cl_r2": value(old, "cl_r2", None, 0.8),
            "c_pitch_r2": value(old, "c_pitch_r2", None, 0.8),
            "field_integrated_cd_rmse": cd,
            "field_integrated_cl_rmse": cl,
            "field_integrated_cmpitch_rmse": value(old, "field_integrated_cmpitch_rmse", None, max(0.001, 0.75 * cl)),
            "field_integrated_clf_rmse": value(old, "field_integrated_clf_rmse", None, max(0.001, 0.8 * cl)),
            "field_integrated_clr_rmse": value(old, "field_integrated_clr_rmse", None, max(0.001, 0.8 * cl)),
            "field_integrated_lift_closure_max_abs": value(old, "field_integrated_lift_closure_max_abs", None, 1e-7),
        }
    )
    old_velocity_r2 = value(old, "velocity_profile_r2", None, 0.95)
    velocity_rmse = value(old, "velocity_profile_uinf_rmse", None, math.sqrt(max(0.0, 1.0 - old_velocity_r2)))
    result.update(
        {
            "velocity_profile_r2": old_velocity_r2,
            "cp_cut_r2": value(old, "cp_cut_r2", None, 0.8),
            "velocity_profile_uinf_rmse": velocity_rmse,
            "velocity_profile_experimental_subset_uinf_rmse": value(old, "velocity_profile_experimental_subset_uinf_rmse", None, velocity_rmse),
            "cp_cut_rmse": value(old, "cp_cut_rmse", None, 0.25),
        }
    )
    return result


def migrate_submission(directory: Path, specification: dict[str, Any], profile: dict[str, Any]) -> None:
    submission_path = directory / "submission.json"
    submission = load_json(submission_path)
    split = next(item for item in specification["splits"] if item["id"] == submission["split_id"])
    split_index = load_json(BENCHMARK_ROOT / split["index_file"])
    case_ids = split_index["case_ids"]

    index_path = directory / submission["profile_data"]["index_file"]
    index = load_json(index_path)
    cursor = 0
    for chunk_entry in index["chunks"]:
        chunk_path = index_path.parent / chunk_entry["file"]
        chunk = load_json(chunk_path)
        count = len(chunk["cases"])
        replacement_ids = case_ids[cursor : cursor + count]
        if len(replacement_ids) != count:
            raise ValueError(f"profile chunk partition exceeds official {submission['split_id']} test set")
        for case, case_id in zip(chunk["cases"], replacement_ids):
            case["case_id"] = case_id
            case["series"] = migrated_series(case, profile)
        chunk_entry["case_ids"] = replacement_ids
        write_json(chunk_path, chunk)
        chunk_entry["sha256"] = sha256_file(chunk_path)
        cursor += count
    if cursor != len(case_ids):
        raise ValueError(f"profile chunks cover {cursor} cases but split requires {len(case_ids)}")
    index.update(
        {
            "dataset_id": "drivaerml",
            "split_id": submission["split_id"],
            "case_set_id": split["case_set_id"],
            "case_count": len(case_ids),
            "case_id_status": "official",
        }
    )
    write_json(index_path, index)

    metric_values = migrated_metric_values(submission.get("metric_values", {}))
    metric_values.update(
        composite_component_group_scores(
            metric_values,
            specification["overall_score_composite"],
            specification["component_score_groups"],
        )
    )
    metric_values["overall_score"] = composite_overall_score(
        metric_values, specification["overall_score_composite"]
    )
    ordered_ids = [item["id"] for item in specification["metrics"]]
    metric_values = {metric_id: metric_values[metric_id] for metric_id in ordered_ids}
    submission.update(
        {
            "dataset_version": DATASET_VERSION,
            "split": split["label"],
            "case_set_id": split["case_set_id"],
            "split_sha256": split["sha256"],
            "metric_values": metric_values,
        }
    )
    submission["evaluation"]["reference_version"] = EVALUATOR_VERSION
    submission["profile_data"].update({"case_count": len(case_ids), "case_set_id": split["case_set_id"]})
    submission["approval"] = {
        "status": "prototype",
        "note": "Structural fixture for the closed DrivAerML candidate; no metric or score is an approved result.",
    }
    submission["note"] = (
        "Illustrative structural fixture migrated to the official split IDs and candidate DrivAerML metric/profile vocabulary. "
        "It is not a recomputation from native fields. Its bounded scores are structural placeholders, so this row is "
        "ineligible for ranking or citation."
    )

    evidence_path = directory / submission["evaluation"]["evidence_file"]
    evidence = load_json(evidence_path)
    evidence.update(
        {
            "split_id": submission["split_id"],
            "reference_version": EVALUATOR_VERSION,
            "metric_values": metric_values,
            "profile_index_sha256": sha256_file(index_path),
            "notes": "Structural dummy evidence for the closed DrivAerML candidate; not computed from public fields and not scientifically rankable.",
        }
    )
    write_json(evidence_path, evidence)
    submission["evaluation"]["evidence_sha256"] = sha256_file(evidence_path)
    write_json(submission_path, submission)


def main() -> int:
    for path, expected in (
        (PROPOSAL_ROOT / "native-source-pin.json", SOURCE_PIN_SHA256),
        (PROPOSAL_ROOT / "owner-published-splits.json", SPLIT_MANIFEST_SHA256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"proposal source digest changed for {path}: {actual}")

    release_bindings = load_candidate_release_bindings()
    specification_path = BENCHMARK_ROOT / "submission-spec.json"
    existing_specification = (
        load_json(specification_path) if specification_path.is_file() else None
    )
    profile_path, profile = select_generation_profile(
        release_bindings, existing_specification
    )
    split_entries = write_splits()
    specification = build_specification(split_entries, profile_path, profile)
    specification = apply_release_managed_bindings(
        specification, release_bindings, existing_specification
    )
    write_json(specification_path, specification)
    update_manifest(specification, profile)
    directories = sorted(path.parent for path in SUBMISSIONS_ROOT.glob("*/submission.json"))
    for directory in directories:
        migrate_submission(directory, specification, profile)
        print(f"migrated {directory.relative_to(ROOT)}")
    print(
        f"updated closed DrivAerML candidate with {len(split_entries)} official splits, "
        f"{len(specification['metrics'])} metrics, and {len(directories)} prototype fixtures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
