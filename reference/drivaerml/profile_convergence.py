"""Strict DrivAerML velocity-profile resolution-convergence evidence.

The candidate scientific contract compares the final profile loss on nested
2, 5, and 10 mm grids with a 1 mm reference grid.  Every candidate grid must
pass its loss and method-order gates before the retained 10 mm grid is eligible
for owner review.  This module consumes only already-computed case-line losses:
it does not manufacture predictions or silently discard cases, lines, spacings,
or methods.

Input losses are a rectangular array indexed as
``[method][case][profile][spacing]``.  The associated orders are declared once
in the same document.  That representation makes missing, duplicated, or
reordered coverage a schema error rather than something a reduction can hide.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .retained_file import RetainedFileError, RetainedVerifiedFile


INPUT_SCHEMA = "drivaerml-profile-resolution-convergence-input-v1"
OUTPUT_SCHEMA = "drivaerml-profile-resolution-convergence-evidence-v1"
SCHEMA_VERSION = 1
SPACINGS_MM = (1, 2, 5, 10)
REFERENCE_SPACING_MM = 1
ACTIVATION_CANDIDATE_SPACING_MM = 10
PROFILE_ORDER = (
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "L1",
    "R1",
    "R2",
    "R3",
)
ZERO_REFERENCE_TOLERANCE = 1.0e-12
SCORE_TIE_ABSOLUTE_TOLERANCE = 1.0e-12
AGGREGATE_RELATIVE_LIMIT = 0.005
CASE_MACRO_RELATIVE_LIMIT = 0.01
CASE_LINE_RELATIVE_LIMIT = 0.02
METHOD_ORDER_MINIMUM = 0.99
MAX_INPUT_BYTES = 128 * 1024 * 1024
OFFICIAL_UNAVAILABLE_RUNS = frozenset(
    {167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473}
)
OFFICIAL_CASE_ORDER = tuple(
    f"run_{run_number}"
    for run_number in range(1, 501)
    if run_number not in OFFICIAL_UNAVAILABLE_RUNS
)
DEFAULT_CONTRACT_PROPOSAL = (
    Path(__file__).resolve().parents[2]
    / "benchmark-specs"
    / "drivaerml"
    / "proposal"
    / "contract-proposal.json"
)

_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = {
    "physics_null",
    "nearest_training_design_vector_control",
    "trained_model_checkpoint",
}


class ProfileConvergenceError(ValueError):
    """Raised when convergence evidence is malformed or not reproducible."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileConvergenceError(
                f"JSON contains duplicate object key {key!r}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of the compact canonical JSON representation."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def method_set_sha256(methods: Sequence[Mapping[str, object]]) -> str:
    """Return the pin used to prove that the ordered method set was declared."""

    return canonical_sha256(list(methods))


def _read_retained_json(
    path: str | Path,
    *,
    label: str,
    maximum_bytes: int = MAX_INPUT_BYTES,
) -> tuple[dict[str, Any], str]:
    try:
        with RetainedVerifiedFile.open(path, label=label) as retained:
            if retained.snapshot.size_bytes > maximum_bytes:
                raise ProfileConvergenceError(
                    f"{label} exceeds the {maximum_bytes}-byte safety limit"
                )
            digest = retained.sha256()
            retained.handle.seek(0)
            payload = retained.handle.read(maximum_bytes + 1)
            retained.assert_unchanged(context="while reading JSON")
    except RetainedFileError as error:
        raise ProfileConvergenceError(str(error)) from error
    if len(payload) > maximum_bytes:
        raise ProfileConvergenceError(
            f"{label} exceeds the {maximum_bytes}-byte safety limit"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except ProfileConvergenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProfileConvergenceError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProfileConvergenceError(f"{label} must be a JSON object")
    return value, digest


def load_input(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one retained convergence input and return it with its byte hash."""

    return _read_retained_json(path, label="profile convergence input")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileConvergenceError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProfileConvergenceError(f"{label} must be an array")
    return value


def _exact_keys(
    value: object, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result))
        unexpected = sorted(set(result) - expected_set)
        raise ProfileConvergenceError(
            f"{label} keys differ from schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return result


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ProfileConvergenceError(f"{label} must be a safe non-empty identifier")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProfileConvergenceError(
            f"{label} must be a lowercase hexadecimal SHA-256"
        )
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProfileConvergenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ProfileConvergenceError(f"{label} must be finite and non-negative")
    return result


def _contract_binding(path: str | Path) -> dict[str, object]:
    document, byte_sha256 = _read_retained_json(
        path, label="DrivAerML contract proposal", maximum_bytes=16 * 1024 * 1024
    )
    try:
        profiles = document["deferred_contracts"]["profiles_and_cp_cuts"]
        velocity_value = profiles["velocity_profiles"]
        convergence = velocity_value["resolution_convergence"]
    except (KeyError, TypeError) as error:
        raise ProfileConvergenceError(
            "contract proposal lacks the velocity-profile convergence contract"
        ) from error
    velocity = _mapping(velocity_value, "contract velocity_profiles")
    convergence = _mapping(convergence, "contract resolution_convergence")
    expected = {
        "reference_spacing_m": 0.001,
        "candidate_spacings_m": [0.002, 0.005, 0.01],
        "nested_reference_strides": [2, 5, 10],
        "relative_change_equation": "abs(L_h-L_1mm)/L_1mm",
        "zero_reference_loss_rule": (
            "if_L_1mm_lte_1e-12_require_absolute_agreement_lte_1e-12"
        ),
        "aggregate_limit": AGGREGATE_RELATIVE_LIMIT,
        "every_case_macro_limit": CASE_MACRO_RELATIVE_LIMIT,
        "every_case_line_limit": CASE_LINE_RELATIVE_LIMIT,
        "method_order_statistic": "kendall_tau_b",
        "method_order_ranking_quantity": (
            "final_E_profile_not_nine_component_overall_score"
        ),
        "method_order_minimum": METHOD_ORDER_MINIMUM,
        "score_tie_absolute_tolerance": SCORE_TIE_ABSOLUTE_TOLERANCE,
        "method_set": (
            "pinned_before_study_including_physics_null_"
            "nearest_training_design_vector_control_and_at_least_three_"
            "distinct_trained_model_checkpoints"
        ),
    }
    for key, expected_value in expected.items():
        if convergence.get(key) != expected_value:
            raise ProfileConvergenceError(
                f"contract proposal has unexpected {key!r}; checker refuses drift"
            )
    validity_expected = {
        "validity_policy_id": "drivaerml-autocfd5-velocity-validity-v1",
        "owner_mask_status": (
            "pending_immutable_owner_release_no_exclusions_approved"
        ),
        "allowed_owner_exclusion_reasons": [
            "inside_morphed_solid",
            "outside_released_fluid_domain",
        ],
        "unresolved_mapping_failure_rule": (
            "unmapped_nonexcluded_sample_makes_case_line_and_complete_ranked_"
            "case_velocity_component_unavailable"
        ),
        "invalid_gap_rule": (
            "only_owner_excluded_endpoints_may_be_removed_and_trapezoid_edge_"
            "contributes_only_if_both_endpoints_owner_included_and_mapped_no_"
            "gap_bridging"
        ),
        "minimum_case_line_support": (
            "positive_owner_included_mapped_contributing_arc_length"
        ),
        "polyhedron_solid_angle_absolute_tolerance_steradian": 0.001,
        "polyhedron_ambiguous_action": (
            "retain_explicit_unresolved_invalid_row_no_cell_assignment"
        ),
    }
    for key, expected_value in validity_expected.items():
        if velocity.get(key) != expected_value:
            raise ProfileConvergenceError(
                f"contract proposal has unexpected {key!r}; checker refuses drift"
            )
    if velocity.get("line_count") != 16 or velocity.get("ranked_line_count") != 16:
        raise ProfileConvergenceError(
            "contract proposal must rank all 16 velocity-profile lines"
        )
    if velocity.get("ranked_error_equation") != (
        "mean_cases(mean_16_lines(case_line_error))"
    ):
        raise ProfileConvergenceError(
            "contract proposal has an unexpected profile macro reduction"
        )
    return {
        "path_role": "benchmark-specs/drivaerml/proposal/contract-proposal.json",
        "sha256": byte_sha256,
        "reference_spacing_mm": REFERENCE_SPACING_MM,
        "candidate_spacings_mm": list(SPACINGS_MM[1:]),
        "activation_candidate_spacing_mm": ACTIVATION_CANDIDATE_SPACING_MM,
        "validity_policy_id": velocity["validity_policy_id"],
        "owner_mask_status": velocity["owner_mask_status"],
    }


def _validate_methods(
    method_set_value: object,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    method_set = _exact_keys(
        method_set_value,
        ("pinned_before_study", "sha256", "methods"),
        "method_set",
    )
    if not isinstance(method_set["pinned_before_study"], bool):
        raise ProfileConvergenceError("method_set.pinned_before_study must be boolean")
    declared_sha = _sha256(method_set["sha256"], "method_set.sha256")
    raw_methods = _list(method_set["methods"], "method_set.methods")
    if not raw_methods:
        raise ProfileConvergenceError("method_set.methods cannot be empty")
    try:
        actual_method_set_sha = method_set_sha256(raw_methods)
    except (TypeError, ValueError) as error:
        raise ProfileConvergenceError(
            "method_set.methods cannot be represented as canonical JSON"
        ) from error
    if actual_method_set_sha != declared_sha:
        raise ProfileConvergenceError(
            "method_set.sha256 does not match the ordered method declarations"
        )

    methods: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    trained_pairs: set[tuple[str, str]] = set()
    trained_hashes: set[str] = set()
    duplicate_trained_pairs = False
    duplicate_trained_hashes = False
    for index, raw_method in enumerate(raw_methods):
        label = f"method_set.methods[{index}]"
        base = _mapping(raw_method, label)
        role = base.get("role")
        if role not in _ROLES:
            raise ProfileConvergenceError(f"{label}.role is not recognized")
        expected_keys = {"method_id", "role", "prediction_artifact_sha256"}
        if role == "trained_model_checkpoint":
            expected_keys.update(("model_id", "checkpoint_id"))
        method = dict(_exact_keys(base, expected_keys, label))
        method_id = _identifier(method["method_id"], f"{label}.method_id")
        if method_id in seen_ids:
            raise ProfileConvergenceError(f"duplicate method_id {method_id!r}")
        seen_ids.add(method_id)
        artifact_sha = _sha256(
            method["prediction_artifact_sha256"],
            f"{label}.prediction_artifact_sha256",
        )
        normalized: dict[str, Any] = {
            "method_id": method_id,
            "role": role,
            "prediction_artifact_sha256": artifact_sha,
        }
        if role == "trained_model_checkpoint":
            model_id = _identifier(method["model_id"], f"{label}.model_id")
            checkpoint_id = _identifier(
                method["checkpoint_id"], f"{label}.checkpoint_id"
            )
            pair = (model_id, checkpoint_id)
            if pair in trained_pairs:
                duplicate_trained_pairs = True
            trained_pairs.add(pair)
            if artifact_sha in trained_hashes:
                duplicate_trained_hashes = True
            trained_hashes.add(artifact_sha)
            normalized.update(model_id=model_id, checkpoint_id=checkpoint_id)
        methods.append(normalized)

    role_counts = {
        role: sum(method["role"] == role for method in methods)
        for role in sorted(_ROLES)
    }
    genuine_count = len(trained_pairs)
    checks = {
        "pinned_before_study": bool(method_set["pinned_before_study"]),
        "exactly_one_physics_null": role_counts["physics_null"] == 1,
        "exactly_one_nearest_training_design_vector_control": (
            role_counts["nearest_training_design_vector_control"] == 1
        ),
        "at_least_three_distinct_trained_model_checkpoints": (
            role_counts["trained_model_checkpoint"] >= 3
            and genuine_count >= 3
            and not duplicate_trained_pairs
            and not duplicate_trained_hashes
        ),
    }
    summary = {
        "sha256": declared_sha,
        "method_order": [method["method_id"] for method in methods],
        "role_counts": role_counts,
        "distinct_trained_model_checkpoint_pairs": genuine_count,
        "distinct_trained_prediction_artifacts": len(trained_hashes),
        "checks": checks,
        "requirements_passed": all(checks.values()),
        "genuine_method_semantics": (
            "trained methods are explicit distinct model/checkpoint declarations "
            "with distinct immutable prediction-artifact hashes; their scientific "
            "authenticity remains subject to evidence and owner review"
        ),
    }
    return methods, summary


def _validate_cases(value: object) -> list[str]:
    case_order = _list(value, "case_order")
    if not case_order:
        raise ProfileConvergenceError("case_order cannot be empty")
    normalized: list[str] = []
    for index, case_id in enumerate(case_order):
        if not isinstance(case_id, str) or _CASE_RE.fullmatch(case_id) is None:
            raise ProfileConvergenceError(
                f"case_order[{index}] must match run_<positive integer>"
            )
        normalized.append(case_id)
    if len(set(normalized)) != len(normalized):
        raise ProfileConvergenceError("case_order contains duplicate cases")
    return normalized


def _validate_losses(
    value: object,
    *,
    method_count: int,
    case_count: int,
) -> list[list[list[list[float]]]]:
    methods = _list(value, "losses")
    if len(methods) != method_count:
        raise ProfileConvergenceError(
            "losses must contain exactly one block per declared method"
        )
    normalized: list[list[list[list[float]]]] = []
    for method_index, method_value in enumerate(methods):
        cases = _list(method_value, f"losses[{method_index}]")
        if len(cases) != case_count:
            raise ProfileConvergenceError(
                f"losses[{method_index}] must contain exactly {case_count} cases"
            )
        method_rows: list[list[list[float]]] = []
        for case_index, case_value in enumerate(cases):
            label = f"losses[{method_index}][{case_index}]"
            profiles = _list(case_value, label)
            if len(profiles) != len(PROFILE_ORDER):
                raise ProfileConvergenceError(
                    f"{label} must contain exactly {len(PROFILE_ORDER)} profiles"
                )
            case_rows: list[list[float]] = []
            for profile_index, profile_value in enumerate(profiles):
                profile_label = f"{label}[{profile_index}]"
                spacings = _list(profile_value, profile_label)
                if len(spacings) != len(SPACINGS_MM):
                    raise ProfileConvergenceError(
                        f"{profile_label} must contain exactly four spacing losses"
                    )
                case_rows.append(
                    [
                        _finite_nonnegative(item, f"{profile_label}[{spacing_index}]")
                        for spacing_index, item in enumerate(spacings)
                    ]
                )
            method_rows.append(case_rows)
        normalized.append(method_rows)
    return normalized


def _validate_input(
    document: Mapping[str, Any],
) -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, object],
    list[str],
    list[list[list[list[float]]]],
]:
    root = _exact_keys(
        document,
        (
            "schema",
            "schema_version",
            "study_id",
            "method_set",
            "case_order",
            "profile_order",
            "spacings_mm",
            "losses",
        ),
        "root",
    )
    if (
        root["schema"] != INPUT_SCHEMA
        or not isinstance(root["schema_version"], int)
        or isinstance(root["schema_version"], bool)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise ProfileConvergenceError("unsupported profile convergence input schema")
    study_id = _identifier(root["study_id"], "study_id")
    if root["profile_order"] != list(PROFILE_ORDER):
        raise ProfileConvergenceError(
            "profile_order must be exactly V1-V6, U1-U6, L1, R1-R3"
        )
    spacings = root["spacings_mm"]
    if (
        not isinstance(spacings, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in spacings
        )
        or spacings != list(SPACINGS_MM)
    ):
        raise ProfileConvergenceError("spacings_mm must be exactly [1,2,5,10]")
    methods, method_summary = _validate_methods(root["method_set"])
    case_order = _validate_cases(root["case_order"])
    losses = _validate_losses(
        root["losses"], method_count=len(methods), case_count=len(case_order)
    )
    return study_id, methods, method_summary, case_order, losses


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ProfileConvergenceError("cannot reduce an empty loss collection")
    result = math.fsum(materialized) / len(materialized)
    if not math.isfinite(result) or result < 0.0:
        raise ProfileConvergenceError("loss macro reduction became invalid")
    return result


def _comparison(
    reference: float, candidate: float, relative_limit: float
) -> dict[str, object]:
    if reference <= ZERO_REFERENCE_TOLERANCE:
        change = abs(candidate - reference)
        return {
            "reference_loss": reference,
            "candidate_loss": candidate,
            "rule": "absolute_zero_reference",
            "change": change,
            "limit": ZERO_REFERENCE_TOLERANCE,
            "nominal_relative_limit": relative_limit,
            "passed": change <= ZERO_REFERENCE_TOLERANCE,
        }
    change = abs(candidate - reference) / reference
    return {
        "reference_loss": reference,
        "candidate_loss": candidate,
        "rule": "relative",
        "change": change,
        "limit": relative_limit,
        "passed": change <= relative_limit,
    }


def kendall_tau_b(
    reference_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    tie_tolerance: float = SCORE_TIE_ABSOLUTE_TOLERANCE,
) -> dict[str, object]:
    """Compute deterministic Kendall tau-b with an absolute tie tolerance.

    If either ranking has no non-tied pair, tau-b is undefined and represented
    as JSON ``null``.  The caller must fail closed rather than replacing it by
    one, including when all scores happen to be equal in both vectors.
    """

    if len(reference_scores) != len(candidate_scores) or not reference_scores:
        raise ProfileConvergenceError(
            "Kendall score vectors must be non-empty and have equal length"
        )
    tolerance = _finite_nonnegative(tie_tolerance, "Kendall tie tolerance")
    reference = tuple(
        _finite_nonnegative(value, f"reference Kendall score[{index}]")
        for index, value in enumerate(reference_scores)
    )
    candidate = tuple(
        _finite_nonnegative(value, f"candidate Kendall score[{index}]")
        for index, value in enumerate(candidate_scores)
    )
    concordant = 0
    discordant = 0
    tied_reference_only = 0
    tied_candidate_only = 0
    tied_both = 0
    for left in range(len(reference)):
        for right in range(left + 1, len(reference)):
            reference_difference = reference[left] - reference[right]
            candidate_difference = candidate[left] - candidate[right]
            reference_tied = abs(reference_difference) <= tolerance
            candidate_tied = abs(candidate_difference) <= tolerance
            if reference_tied and candidate_tied:
                tied_both += 1
            elif reference_tied:
                tied_reference_only += 1
            elif candidate_tied:
                tied_candidate_only += 1
            elif reference_difference * candidate_difference > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator_left = concordant + discordant + tied_reference_only
    denominator_right = concordant + discordant + tied_candidate_only
    denominator = math.sqrt(denominator_left * denominator_right)
    tau_b = (
        (concordant - discordant) / denominator if denominator > 0.0 else None
    )
    return {
        "tau_b": tau_b,
        "tie_absolute_tolerance": tolerance,
        "pair_counts": {
            "concordant": concordant,
            "discordant": discordant,
            "tied_reference_only": tied_reference_only,
            "tied_candidate_only": tied_candidate_only,
            "tied_both": tied_both,
            "total": len(reference) * (len(reference) - 1) // 2,
        },
        "defined": tau_b is not None,
    }


def evaluate_profile_convergence(
    document: Mapping[str, Any],
    *,
    contract_proposal: str | Path = DEFAULT_CONTRACT_PROPOSAL,
    input_byte_sha256: str | None = None,
) -> dict[str, object]:
    """Validate and reduce one complete candidate convergence study."""

    contract = _contract_binding(contract_proposal)
    study_id, methods, method_summary, case_order, losses = _validate_input(document)
    canonical_input_sha = canonical_sha256(document)
    if input_byte_sha256 is not None:
        _sha256(input_byte_sha256, "input_byte_sha256")

    spacing_results: list[dict[str, object]] = []
    spacing_index = {spacing: index for index, spacing in enumerate(SPACINGS_MM)}
    for spacing_mm in SPACINGS_MM[1:]:
        candidate_index = spacing_index[spacing_mm]
        method_results: list[dict[str, object]] = []
        reference_scores: list[float] = []
        candidate_scores: list[float] = []
        for method_index, method in enumerate(methods):
            case_results: list[dict[str, object]] = []
            reference_case_macros: list[float] = []
            candidate_case_macros: list[float] = []
            for case_index, case_id in enumerate(case_order):
                line_results: list[dict[str, object]] = []
                reference_line_losses: list[float] = []
                candidate_line_losses: list[float] = []
                for profile_index, profile_id in enumerate(PROFILE_ORDER):
                    values = losses[method_index][case_index][profile_index]
                    reference_loss = values[0]
                    candidate_loss = values[candidate_index]
                    reference_line_losses.append(reference_loss)
                    candidate_line_losses.append(candidate_loss)
                    line_results.append(
                        {
                            "profile_id": profile_id,
                            "comparison": _comparison(
                                reference_loss,
                                candidate_loss,
                                CASE_LINE_RELATIVE_LIMIT,
                            ),
                        }
                    )
                reference_case = _mean(reference_line_losses)
                candidate_case = _mean(candidate_line_losses)
                reference_case_macros.append(reference_case)
                candidate_case_macros.append(candidate_case)
                macro = _comparison(
                    reference_case, candidate_case, CASE_MACRO_RELATIVE_LIMIT
                )
                case_results.append(
                    {
                        "case_id": case_id,
                        "macro_comparison": macro,
                        "case_lines": line_results,
                        "case_lines_passed": all(
                            bool(line["comparison"]["passed"])
                            for line in line_results
                        ),
                    }
                )
            reference_aggregate = _mean(reference_case_macros)
            candidate_aggregate = _mean(candidate_case_macros)
            reference_scores.append(reference_aggregate)
            candidate_scores.append(candidate_aggregate)
            aggregate = _comparison(
                reference_aggregate,
                candidate_aggregate,
                AGGREGATE_RELATIVE_LIMIT,
            )
            method_passed = (
                bool(aggregate["passed"])
                and all(
                    bool(case["macro_comparison"]["passed"])
                    for case in case_results
                )
                and all(bool(case["case_lines_passed"]) for case in case_results)
            )
            method_results.append(
                {
                    "method_id": method["method_id"],
                    "role": method["role"],
                    "aggregate_comparison": aggregate,
                    "cases": case_results,
                    "passed": method_passed,
                }
            )
        ordering = kendall_tau_b(reference_scores, candidate_scores)
        tau = ordering["tau_b"]
        ordering["reference_final_E_profile_by_method"] = [
            {"method_id": method["method_id"], "value": score}
            for method, score in zip(methods, reference_scores, strict=True)
        ]
        ordering["candidate_final_E_profile_by_method"] = [
            {"method_id": method["method_id"], "value": score}
            for method, score in zip(methods, candidate_scores, strict=True)
        ]
        ordering["minimum"] = METHOD_ORDER_MINIMUM
        ordering["ranking_quantity"] = "final_E_profile"
        ordering["passed"] = tau is not None and tau >= METHOD_ORDER_MINIMUM
        spacing_passed = all(bool(item["passed"]) for item in method_results) and bool(
            ordering["passed"]
        )
        spacing_results.append(
            {
                "spacing_mm": spacing_mm,
                "methods": method_results,
                "method_ordering": ordering,
                "passed": spacing_passed,
            }
        )

    by_spacing = {
        int(result["spacing_mm"]): result for result in spacing_results
    }
    activation_spacing_passed = bool(
        by_spacing[ACTIVATION_CANDIDATE_SPACING_MM]["passed"]
    )
    all_candidate_spacings_passed = all(
        bool(result["passed"]) for result in spacing_results
    )
    method_requirements_passed = bool(method_summary["requirements_passed"])
    complete_official_case_scope = tuple(case_order) == OFFICIAL_CASE_ORDER
    owner_validity_mask_bound = (
        contract["owner_mask_status"] == "immutable_owner_release_bound"
    )
    activation_eligible = (
        all_candidate_spacings_passed
        and method_requirements_passed
        and complete_official_case_scope
        and owner_validity_mask_bound
    )
    passing_spacings = [
        int(result["spacing_mm"])
        for result in spacing_results
        if bool(result["passed"])
    ]
    coarsest_passing = max(passing_spacings) if passing_spacings else None
    blocking_reasons: list[str] = []
    if not complete_official_case_scope:
        blocking_reasons.append("incomplete_official_484_case_scope")
    if not method_requirements_passed:
        blocking_reasons.append("incomplete_or_unpinned_genuine_method_set")
    if not all_candidate_spacings_passed:
        blocking_reasons.append("resolution_threshold_or_method_order_failure")
    if not owner_validity_mask_bound:
        blocking_reasons.append("owner_validity_mask_not_bound")
    if activation_eligible:
        status = "eligible_for_owner_activation_review"
    elif not complete_official_case_scope:
        status = "ineligible_incomplete_official_484_case_scope"
    elif not method_requirements_passed:
        status = "ineligible_incomplete_or_unpinned_genuine_method_set"
    elif not all_candidate_spacings_passed:
        status = "ineligible_resolution_threshold_failure"
    else:
        status = "blocked_owner_validity_mask_not_bound"

    input_binding: dict[str, object] = {
        "canonical_json_sha256": canonical_input_sha,
    }
    if input_byte_sha256 is not None:
        input_binding["file_byte_sha256"] = input_byte_sha256
    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "profile_resolution_activation_eligible": activation_eligible,
        "does_not_activate_scoring_contract": True,
        "owner_scientific_approval_claimed": False,
        "independent_participant_dry_run_claimed": False,
        "input": input_binding,
        "contract": contract,
        "scope": {
            "case_order": case_order,
            "case_count": len(case_order),
            "required_official_case_count": len(OFFICIAL_CASE_ORDER),
            "complete_official_case_order": complete_official_case_scope,
            "profile_order": list(PROFILE_ORDER),
            "profile_count": len(PROFILE_ORDER),
            "spacings_mm": list(SPACINGS_MM),
            "complete_rectangular_coverage": True,
            "owner_validity_mask_bound": owner_validity_mask_bound,
        },
        "method_set": method_summary,
        "thresholds": {
            "aggregate_relative_change": AGGREGATE_RELATIVE_LIMIT,
            "every_case_macro_relative_change": CASE_MACRO_RELATIVE_LIMIT,
            "every_case_line_relative_change": CASE_LINE_RELATIVE_LIMIT,
            "zero_reference_loss_and_absolute_agreement": ZERO_REFERENCE_TOLERANCE,
            "method_order_kendall_tau_b_minimum": METHOD_ORDER_MINIMUM,
            "score_tie_absolute_tolerance": SCORE_TIE_ABSOLUTE_TOLERANCE,
        },
        "spacing_results": spacing_results,
        "selection": {
            "activation_candidate_spacing_mm": ACTIVATION_CANDIDATE_SPACING_MM,
            "activation_candidate_passed": activation_spacing_passed,
            "all_prescribed_candidate_spacings_passed": (
                all_candidate_spacings_passed
            ),
            "failed_candidate_spacings_mm": [
                int(result["spacing_mm"])
                for result in spacing_results
                if not bool(result["passed"])
            ],
            "coarsest_passing_candidate_spacing_mm": coarsest_passing,
            "owner_activation_review_rule": (
                "every prescribed 2, 5, and 10 mm comparison against the "
                "1 mm reference must pass its loss thresholds and method-order "
                "gate before the retained 10 mm grid is eligible"
            ),
            "failure_rule": (
                "diagnostic remains unranked until a new contract version globally "
                "adopts the coarsest finer passing grid"
            ),
        },
    }


def write_evidence(
    document: Mapping[str, object], path: str | Path
) -> dict[str, object]:
    """Atomically write compact deterministic JSON and return its identity."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(document) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(destination),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = [
    "ACTIVATION_CANDIDATE_SPACING_MM",
    "DEFAULT_CONTRACT_PROPOSAL",
    "INPUT_SCHEMA",
    "OFFICIAL_CASE_ORDER",
    "OFFICIAL_UNAVAILABLE_RUNS",
    "OUTPUT_SCHEMA",
    "PROFILE_ORDER",
    "ProfileConvergenceError",
    "SCHEMA_VERSION",
    "SPACINGS_MM",
    "canonical_sha256",
    "evaluate_profile_convergence",
    "kendall_tau_b",
    "load_input",
    "method_set_sha256",
    "write_evidence",
]
