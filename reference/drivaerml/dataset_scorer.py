"""Strict candidate-only dataset reductions for DrivAerML.

The native DrivAerML evaluator deliberately produces one compact evidence
document per case.  This module performs the next reduction level without
opening VTK or prediction arrays: it validates every case document against one
official split, the immutable native-source pin, the candidate diagnostic
registry, and a predeclared authoritative force-table hash.  Only then are
case metrics reduced.

This is *candidate* evidence.  A frozen scoring-support release, owner
approval, and an independent participant dry run are still required before an
official submission can exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from .coordinate_identity import (
    CoordinateIdentityError,
    coordinate_array_identity_sha256,
)
from .regional_aggregate import (
    RegionalAggregateError,
    aggregate_regional_diagnostics,
    validate_case_regional_envelope,
)
from .retained_file import RetainedFileError, RetainedVerifiedFile
from .source import (
    NativeCaseRecord,
    NativeSourceError,
    NativeSourcePin,
    load_native_source_pin,
)
CANDIDATE_DATASET_SCHEMA = "drivaerml-candidate-dataset-evaluation-v4"
LEGACY_CANDIDATE_DATASET_SCHEMA = "drivaerml-candidate-dataset-evaluation-v3"
CANDIDATE_DATASET_STATUS = (
    "candidate_dataset_evidence_not_active_or_official_submission"
)
CORE_CASE_SCHEMA = "drivaerml-candidate-case-evaluation-v4"
LEGACY_CORE_CASE_SCHEMA = "drivaerml-candidate-case-evaluation-v2"
CORE_CASE_STATUS = "candidate_evaluator_evidence_not_official_submission"
DIAGNOSTIC_CASE_SCHEMA = "drivaerml-case-diagnostics-candidate-v4"
LEGACY_DIAGNOSTIC_CASE_SCHEMA = "drivaerml-case-diagnostics-candidate-v3"
DIAGNOSTIC_CASE_STATUS = "candidate_diagnostics_not_active_or_official_submission"
RETAINED_INFERENCE_DIAGNOSTIC_CASE_SCHEMA = (
    "drivaerml-retained-inference-diagnostics-v1"
)
RETAINED_INFERENCE_DIAGNOSTIC_CASE_STATUS = (
    "candidate_retained_inference_evidence_not_official_submission"
)

OFFICIAL_REPOSITORY_ID = "neashton/drivaerml"
OFFICIAL_REPOSITORY_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
OFFICIAL_NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
OFFICIAL_FORCE_TRUTH_SHA256 = (
    "4e9e003da38ccdcacad359451079888361eae221d3c8dad7fd5682250d257865"
)
OFFICIAL_DIAGNOSTIC_PROFILE_SHA256 = (
    "df22bc807b62f925c32659d681ac44064e6acf46449038b8431b1e9139aba1e8"
)
OFFICIAL_SURFACE_AREA_MANIFEST_SHA256 = (
    "1401c7e80bd86f3aa2d640289db9b088ce1e0825327e18eeb1ab2852de04323e"
)
SURFACE_AREA_GEOMETRY_RELATIVE_TOLERANCE = 6.0e-8

PRIMARY_FIELD_METRIC_IDS = (
    "surface_pressure_rel_l2",
    "surface_wall_shear_rel_l2",
    "volume_velocity_rel_l2",
    "volume_pressure_rel_l2",
)
PREDICTION_SCOPE_FULL = "surface_and_volume"
PREDICTION_SCOPE_SURFACE_ONLY = "surface_only"
PREDICTION_SCOPES = frozenset(
    {PREDICTION_SCOPE_FULL, PREDICTION_SCOPE_SURFACE_ONLY}
)
RANKED_FORCE_METRIC_IDS = (
    "field_integrated_cd_rmse",
    "field_integrated_cl_rmse",
    "field_integrated_cmpitch_rmse",
)
REPORT_ONLY_FORCE_METRIC_IDS = (
    "field_integrated_clf_rmse",
    "field_integrated_clr_rmse",
    "field_integrated_lift_closure_max_abs",
)
DIAGNOSTIC_METRIC_IDS = (
    "velocity_profile_uinf_rmse",
    "velocity_profile_experimental_subset_uinf_rmse",
    "cp_cut_rmse",
)
RANKED_DIAGNOSTIC_METRIC_IDS = (
    "velocity_profile_uinf_rmse",
    "cp_cut_rmse",
)
REPORT_ONLY_DIAGNOSTIC_METRIC_IDS = (
    "velocity_profile_experimental_subset_uinf_rmse",
)
SCHEMA_V3_FORCE_COEFFICIENT_KEYS = (
    "cd",
    "cl",
    "cm_pitch",
    "clf",
    "clr",
)
_INTERNAL_FORCE_COEFFICIENT_KEYS = (
    "Cd",
    "Cl",
    "CmPitch",
    "Clf",
    "Clr",
)
NONSPATIAL_METRIC_IDS = (
    *RANKED_FORCE_METRIC_IDS,
    *REPORT_ONLY_FORCE_METRIC_IDS,
    *DIAGNOSTIC_METRIC_IDS,
)

_FIELD_LAYOUT = {
    "surface_pressure": {
        "support": "surface_native_cells",
        "component_count": 1,
        "primary_weighting": "physical",
        "secondary_weighting": "uniform",
        "primary_label": "area",
        "secondary_label": "equal_entity",
        "primary_dataset_weighting": "surface_face_area",
        "secondary_dataset_weighting": "surface_entities_equal",
    },
    "surface_wall_shear": {
        "support": "surface_native_cells",
        "component_count": 3,
        "primary_weighting": "physical",
        "secondary_weighting": "uniform",
        "primary_label": "area",
        "secondary_label": "equal_entity",
        "primary_dataset_weighting": "surface_face_area",
        "secondary_dataset_weighting": "surface_entities_equal",
    },
    "volume_pressure": {
        "support": "volume_native_cells",
        "component_count": 1,
        "primary_weighting": "uniform",
        "primary_label": "equal_entity",
        "primary_dataset_weighting": "volume_cells_equal",
    },
    "volume_velocity": {
        "support": "volume_native_cells",
        "component_count": 3,
        "primary_weighting": "uniform",
        "primary_label": "equal_entity",
        "primary_dataset_weighting": "volume_cells_equal",
    },
}


def _field_metric_ids(prefix: str) -> tuple[str, ...]:
    layout = _FIELD_LAYOUT[prefix]
    primary = (
        f"{prefix}_rel_l2",
        f"drivaerml_{prefix}_{layout['primary_label']}_mae",
        f"drivaerml_{prefix}_{layout['primary_label']}_rmse",
    )
    if layout["support"] == "volume_native_cells":
        return primary
    return (
        primary[0],
        f"{prefix}_equal_entity_rel_l2",
        primary[1],
        primary[2],
        f"drivaerml_{prefix}_{layout['secondary_label']}_mae",
        f"drivaerml_{prefix}_{layout['secondary_label']}_rmse",
    )


ALL_FIELD_METRIC_IDS = tuple(
    metric_id
    for prefix in _FIELD_LAYOUT
    for metric_id in _field_metric_ids(prefix)
)
ALL_RELATIVE_L2_METRIC_IDS = tuple(
    metric_id
    for metric_id in ALL_FIELD_METRIC_IDS
    if metric_id.endswith("_rel_l2")
)

_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}\Z")

RELATIVE_PROFILE_FORMAT = (
    "fluidsbench-drivaerml-relative-profile-chunks-v3-candidate"
)
RELATIVE_PROFILE_SCHEMA_VERSION = "3.0-drivaerml-relative-candidate"
RELATIVE_PROFILE_CONTRACT_ID = "drivaerml-relative-diagnostics-v3-candidate"
RELATIVE_PROFILE_CONTRACT_SHA256 = (
    "a6066fa83a7af0d18b84a18c11079f020023a6aed81292c4251186c14fc0461a"
)
RELATIVE_PROFILE_SERIES_PER_CASE = 40
RELATIVE_SERIES_SUPPORT_INDEX_SCHEMA = (
    "drivaerml-relative-series-support-index-v2"
)
RELATIVE_SERIES_SUPPORT_INDEX_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "series-support-index.json"
)
# Updated only when a separately reviewed retained index changes bytes.
RELATIVE_SERIES_SUPPORT_INDEX_SHA256 = (
    "ff7b8bdb0b963611ce7ecb2055090b6861d42477ff47976d7b164ff131d82632"
)
RELATIVE_SERIES_SUPPORT_PER_CASE = 20
CONSTANT_SERIES_SUPPORT_INDEX_SCHEMA = (
    "drivaerml-constant-series-support-index-v1"
)
CONSTANT_SERIES_SUPPORT_INDEX_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "constant-series-support-index.json"
)
# Updated only when the separately reviewed native-v3 truth release changes.
CONSTANT_SERIES_SUPPORT_INDEX_SHA256 = (
    "056b6488230c2224dfca07fca5e00eaab349bd0cda22cadffc0e609eb8a611e2"
)
CONSTANT_SERIES_SUPPORT_PER_CASE = 20
RUN419_CONSTANT_SERIES_SUPPORT_INDEX_SCHEMA = (
    "drivaerml-run419-constant-series-support-index-v1"
)
RUN419_CONSTANT_SERIES_SUPPORT_INDEX_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark-specs"
    / "drivaerml"
    / "support"
    / "relative-v3"
    / "run419-constant-series-support-index.json"
)
RUN419_CONSTANT_SERIES_SUPPORT_INDEX_SHA256 = (
    "c66adf17b73fbe0c4cba080e2ee44e2046f7c69cc7e5a5e797b2b973c826b211"
)
RUN419_CONSTANT_SERIES_SUPPORT_COUNT = 20
_ZERO_SHA256 = "0" * 64

_CONSTANT_VELOCITY_STATIONS = tuple(
    f"autocfd5_{station.lower()}"
    for station in (
        "V1", "V2", "V3", "V4", "V5", "V6",
        "U1", "U2", "U3", "U4", "U5", "U6",
        "L1", "R1", "R2", "R3",
    )
)
_RELATIVE_VELOCITY_STATIONS = (
    "V1", "V2", "V3", "V4", "V5", "V6",
    "U1", "U2", "U3", "U4", "U5", "U6",
    "L1", "R1", "R2", "R3",
)
_VELOCITY_SAMPLE_COUNTS = {
    **{station: 201 for station in ("V1", "V2", "V3", "V4", "V5", "V6")},
    **{station: 301 for station in ("U1", "U2", "U3", "U4", "U5", "U6")},
    "L1": 651,
    "R1": 31,
    "R2": 31,
    "R3": 31,
}
_CONSTANT_VELOCITY_INTERVAL_ENDS = {
    **{station: 2.0 for station in ("V1", "V2", "V3", "V4", "V5", "V6")},
    **{station: 3.0 for station in ("U1", "U2", "U3", "U4", "U5", "U6")},
    "L1": 6.5,
    "R1": 0.300000876107,
    "R2": 0.300000877067,
    "R3": 0.300000591286,
}
_CONSTANT_CP_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
    "sidewall_z_0_15",
    "front_left_wheelhouse_y_neg_0_6",
)
_RELATIVE_CP_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
    "sidewall_front_wheelhouse_relative",
    "front_left_wheelhouse_relative",
)
_SHARED_CP_SUPPORT_IDS = {
    "upperbody_centerline": "drivaerml-cp-upperbody-centerline-y0-v1",
    "underbody_centerline": "drivaerml-cp-underbody-centerline-y0-v1",
}
_SUBMISSION_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,79}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")


class DrivAerDatasetScorerError(ValueError):
    """Raised when dataset evidence cannot be reduced without ambiguity."""


@dataclass(frozen=True)
class CandidateDatasetEvaluation:
    """One deterministic, explicitly ineligible dataset-level result."""

    evidence: Mapping[str, object]
    regional_diagnostics: Mapping[str, object] | None = None

    def to_json(self) -> dict[str, object]:
        result = dict(self.evidence)
        _assert_no_absolute_paths(result)
        return result


@dataclass(frozen=True)
class CandidateProfileChunks:
    """A deterministic candidate profile index and its ordered chunks."""

    index: Mapping[str, object]
    chunks: tuple[tuple[str, Mapping[str, object]], ...]

    def index_json(self) -> dict[str, object]:
        result = dict(self.index)
        _assert_no_absolute_paths(result)
        return result

    def chunk_json(self, filename: str) -> dict[str, object]:
        matches = [document for name, document in self.chunks if name == filename]
        if len(matches) != 1:
            raise DrivAerDatasetScorerError(
                f"candidate profile package has no unique chunk {filename!r}"
            )
        result = dict(matches[0])
        _assert_no_absolute_paths(result)
        return result


@dataclass(frozen=True)
class CandidateDatasetImmutablePins:
    """Predeclared identities that cannot be learned from supplied inputs."""

    repository_id: str
    repository_revision: str
    native_source_pin_sha256: str
    force_truth_sha256: str
    diagnostic_profile_sha256: str
    surface_area_manifest_sha256: str


OFFICIAL_IMMUTABLE_PINS = CandidateDatasetImmutablePins(
    repository_id=OFFICIAL_REPOSITORY_ID,
    repository_revision=OFFICIAL_REPOSITORY_REVISION,
    native_source_pin_sha256=OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
    force_truth_sha256=OFFICIAL_FORCE_TRUTH_SHA256,
    diagnostic_profile_sha256=OFFICIAL_DIAGNOSTIC_PROFILE_SHA256,
    surface_area_manifest_sha256=OFFICIAL_SURFACE_AREA_MANIFEST_SHA256,
)


@dataclass(frozen=True)
class _Contract:
    specification_path: Path
    specification_sha256: str
    split_id: str
    case_set_id: str
    split_file: str
    split_sha256: str
    case_ids: tuple[str, ...]
    native_source_pin: NativeSourcePin
    native_source_pin_file: str
    native_source_pin_sha256: str
    force_truth_file: str
    force_truth_sha256: str
    profile_file: str
    profile_sha256: str
    profile_velocity_line_ids: tuple[str, ...]
    profile_velocity_station_ids: tuple[str, ...]
    profile_velocity_line_sample_counts: tuple[int, ...]
    profile_experimental_velocity_line_ids: tuple[str, ...]
    profile_velocity_sample_count: int
    profile_cp_cut_ids: tuple[str, ...]
    surface_area_manifest_sha256: str
    force_constants: Mapping[str, object]


@dataclass(frozen=True)
class _CoreCase:
    case_id: str
    input_file: str
    input_sha256: str
    prediction_scope: str
    surface_count: int
    volume_count: int | None
    field_metrics: Mapping[str, float]
    relative_l2_statistics: Mapping[str, Mapping[str, object]]
    force_coefficients: Mapping[str, float]
    surface_prediction_manifest_sha256: str
    volume_prediction_manifest_sha256: str | None
    surface_prediction_chunk_sha256: tuple[str, ...]
    volume_prediction_chunk_sha256: tuple[str, ...] | None
    boundary_sha256: str
    surface_area_sha256: str
    volume_part_sha256: tuple[str, ...] | None
    regional_diagnostics: Mapping[str, object] | None


@dataclass(frozen=True)
class _DiagnosticCase:
    case_id: str
    input_file: str
    input_sha256: str
    values: Mapping[str, float | None]
    unavailable_reasons: Mapping[str, tuple[Mapping[str, object], ...]]
    velocity_mapping_sha256: str | None
    velocity_receipt_sha256: str | None
    profile_series: tuple[Mapping[str, object], ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DrivAerDatasetScorerError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _read_json(path: Path | str, label: str) -> tuple[dict[str, Any], Path, str]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise DrivAerDatasetScorerError(f"{label} cannot be a symbolic link")
    try:
        with RetainedVerifiedFile.open(unresolved, label=label) as retained:
            source = retained.source_path
            initial = retained.sha256(chunk_bytes=1024 * 1024)
            retained.handle.seek(0)
            value = json.load(
                retained.handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    DrivAerDatasetScorerError(
                        f"{label} contains forbidden non-finite token {token}"
                    )
                ),
            )
            retained.assert_unchanged(context="while its JSON was parsed")
    except DrivAerDatasetScorerError:
        raise
    except RetainedFileError as error:
        raise DrivAerDatasetScorerError(str(error)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DrivAerDatasetScorerError(
            f"cannot read valid {label} JSON: {unresolved.name}"
        ) from error
    if not isinstance(value, dict):
        raise DrivAerDatasetScorerError(f"{label} must be a JSON object")
    return value, source, initial


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DrivAerDatasetScorerError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        raise DrivAerDatasetScorerError(
            f"{label} keys differ from the frozen candidate schema "
            f"(missing={sorted(expected_set - set(result))}, "
            f"unexpected={sorted(set(result) - expected_set)})"
        )
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DrivAerDatasetScorerError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DrivAerDatasetScorerError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DrivAerDatasetScorerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " finite and non-negative" if nonnegative else " finite"
        raise DrivAerDatasetScorerError(f"{label} must be{suffix}")
    return result


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise DrivAerDatasetScorerError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return result


def _case_id(value: object, label: str = "case_id") -> str:
    result = _string(value, label)
    if _CASE_RE.fullmatch(result) is None:
        raise DrivAerDatasetScorerError(f"{label} is not a canonical run_N ID")
    return result


def _basename(value: object, label: str) -> str:
    result = _string(value, label).replace("\\", "/")
    if (
        result.startswith("/")
        or _WINDOWS_ABSOLUTE_RE.match(result)
        or "/" in result
        or result in {".", ".."}
    ):
        raise DrivAerDatasetScorerError(f"{label} must be a basename")
    return result


def _same_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=2.0e-14)


def _require_same_float(actual: object, expected: float, label: str) -> float:
    value = _finite(actual, label, nonnegative=expected >= 0.0)
    if not _same_float(value, expected):
        raise DrivAerDatasetScorerError(
            f"{label} differs from its independently reconstructed value"
        )
    return value


def _declared_path(base: Path, value: object, label: str) -> Path:
    relative = PurePosixPath(_string(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise DrivAerDatasetScorerError(f"{label} must be a safe relative path")
    resolved_base = base.resolve()
    result = (resolved_base / Path(*relative.parts)).resolve()
    if resolved_base != result and resolved_base not in result.parents:
        raise DrivAerDatasetScorerError(f"{label} escapes the specification root")
    return result


def _unique_case_ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DrivAerDatasetScorerError(f"{label} must be a non-empty array")
    result = tuple(_case_id(item, label) for item in value)
    if len(result) != len(set(result)):
        raise DrivAerDatasetScorerError(f"{label} contains duplicate case IDs")
    return result


def _support_by_id(specification: Mapping[str, Any], support_id: str) -> Mapping[str, Any]:
    scoring = _mapping(specification.get("scoring_support"), "scoring_support")
    supports = scoring.get("public_supports")
    if not isinstance(supports, list):
        raise DrivAerDatasetScorerError("scoring_support.public_supports must be an array")
    matching = [item for item in supports if isinstance(item, Mapping) and item.get("id") == support_id]
    if len(matching) != 1:
        raise DrivAerDatasetScorerError(
            f"submission specification must define support {support_id!r} exactly once"
        )
    return matching[0]


def _load_contract(
    *,
    submission_specification: Path | str,
    split_id: str,
    force_truth_csv: Path | str,
    immutable_pins: CandidateDatasetImmutablePins,
) -> tuple[_Contract, Mapping[str, Mapping[str, float]]]:
    if not isinstance(immutable_pins, CandidateDatasetImmutablePins):
        raise DrivAerDatasetScorerError(
            "immutable_pins must be CandidateDatasetImmutablePins"
        )
    for label, digest in (
        ("native source pin", immutable_pins.native_source_pin_sha256),
        ("force truth", immutable_pins.force_truth_sha256),
        ("diagnostic profile", immutable_pins.diagnostic_profile_sha256),
        ("surface-area manifest", immutable_pins.surface_area_manifest_sha256),
    ):
        _sha256(digest, f"predeclared {label} SHA-256")
    if not isinstance(split_id, str) or _SAFE_ID_RE.fullmatch(split_id) is None:
        raise DrivAerDatasetScorerError("exactly one valid split_id is required")
    specification, spec_path, spec_sha = _read_json(
        submission_specification, "DrivAerML submission specification"
    )
    if specification.get("dataset_id") != "drivaerml":
        raise DrivAerDatasetScorerError("submission specification is not DrivAerML")
    if specification.get("status") != "candidate_scoring_contract":
        raise DrivAerDatasetScorerError(
            "dataset scorer only accepts the candidate DrivAerML contract"
        )
    scoring = _mapping(specification.get("scoring_support"), "scoring_support")
    if (
        scoring.get("status") != "owner_review_required"
        or scoring.get("submissions_open") is not False
    ):
        raise DrivAerDatasetScorerError(
            "candidate dataset scorer requires submissions to remain closed"
        )
    source_release = _mapping(scoring.get("source_release"), "source_release")
    if (
        source_release.get("repository") != immutable_pins.repository_id
        or source_release.get("revision") != immutable_pins.repository_revision
    ):
        raise DrivAerDatasetScorerError("source release repository identity is not frozen")

    split_declarations = specification.get("splits")
    if not isinstance(split_declarations, list):
        raise DrivAerDatasetScorerError("submission specification splits must be an array")
    matching_splits = [
        item
        for item in split_declarations
        if isinstance(item, Mapping) and item.get("id") == split_id
    ]
    if len(matching_splits) != 1:
        raise DrivAerDatasetScorerError(
            f"split {split_id!r} must select exactly one official split"
        )
    split_declaration = matching_splits[0]
    if split_declaration.get("case_id_status") != "official":
        raise DrivAerDatasetScorerError("selected split case IDs are not official")
    split_path = _declared_path(
        spec_path.parent,
        split_declaration.get("index_file"),
        "split index_file",
    )
    split_document, _, split_sha = _read_json(split_path, "official split index")
    expected_split_sha = _sha256(split_declaration.get("sha256"), "split SHA-256")
    if split_sha != expected_split_sha:
        raise DrivAerDatasetScorerError("official split index SHA-256 mismatch")
    split = _exact_keys(
        split_document,
        {
            "schema_version",
            "dataset_id",
            "split_id",
            "case_set_id",
            "split_label",
            "case_id_status",
            "training_case_count",
            "validation_case_count",
            "case_count",
            "train_case_ids",
            "validation_case_ids",
            "case_ids",
            "source",
            "usage",
        },
        "official split index",
    )
    if (
        split["schema_version"] != "1.0"
        or split["dataset_id"] != "drivaerml"
        or split["split_id"] != split_id
        or split["case_id_status"] != "official"
        or split["case_set_id"] != split_declaration.get("case_set_id")
    ):
        raise DrivAerDatasetScorerError("official split index identity mismatch")
    train = _unique_case_ids(split["train_case_ids"], "split train_case_ids")
    validation = _unique_case_ids(
        split["validation_case_ids"], "split validation_case_ids"
    )
    test = _unique_case_ids(split["case_ids"], "split case_ids")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise DrivAerDatasetScorerError("official split train/validation/test sets overlap")
    for key, values in (
        ("training_case_count", train),
        ("validation_case_count", validation),
        ("case_count", test),
    ):
        if split[key] != len(values) or split_declaration.get(key) != len(values):
            raise DrivAerDatasetScorerError(f"official split {key} is inconsistent")
    split_source = _mapping(split["source"], "official split source")
    if (
        split_source.get("repository") != immutable_pins.repository_id
        or split_source.get("revision") != immutable_pins.repository_revision
    ):
        raise DrivAerDatasetScorerError("official split source identity mismatch")

    pin_declaration = _mapping(
        source_release.get("native_source_pin"), "native source pin declaration"
    )
    pin_path = _declared_path(
        spec_path.parent, pin_declaration.get("file"), "native source pin file"
    )
    pin_document, _, pin_sha = _read_json(pin_path, "native source pin")
    expected_pin_sha = _sha256(
        pin_declaration.get("sha256"), "native source pin SHA-256"
    )
    if expected_pin_sha != immutable_pins.native_source_pin_sha256:
        raise DrivAerDatasetScorerError(
            "submission specification native-source pin differs from the "
            "predeclared evaluator identity"
        )
    if pin_sha != expected_pin_sha:
        raise DrivAerDatasetScorerError("native source pin SHA-256 mismatch")
    try:
        native_pin = load_native_source_pin(pin_path)
    except NativeSourceError as error:
        raise DrivAerDatasetScorerError(str(error)) from error
    if (
        native_pin.repository_id != immutable_pins.repository_id
        or native_pin.repository_revision != immutable_pins.repository_revision
    ):
        raise DrivAerDatasetScorerError("native source pin repository identity mismatch")
    pinned_ids = tuple(case.case_id for case in native_pin.cases)
    if any(case_id not in set(pinned_ids) for case_id in (*train, *validation, *test)):
        raise DrivAerDatasetScorerError("official split references a case absent from the native pin")

    force_support = _support_by_id(specification, "field_integrated_force_truth")
    force_sha = _sha256(force_support.get("sha256"), "force truth SHA-256")
    if force_sha != immutable_pins.force_truth_sha256:
        raise DrivAerDatasetScorerError(
            "submission specification force truth differs from the predeclared hash"
        )
    pin_supports = _mapping(
        pin_document.get("authoritative_support_files"),
        "native pin authoritative_support_files",
    )
    pin_force = _mapping(
        pin_supports.get("force_mom_constref_all"),
        "native pin force_mom_constref_all",
    )
    if pin_force.get("sha256") != force_sha:
        raise DrivAerDatasetScorerError("force truth hash differs between contract and native pin")
    truth, truth_path, truth_sha = _load_force_truth(
        force_truth_csv,
        expected_sha256=force_sha,
        expected_case_ids=pinned_ids,
    )

    surface_support = _support_by_id(specification, "surface_native_cells")
    surface_manifest = _mapping(
        surface_support.get("physical_weight_manifest"),
        "surface physical-weight manifest",
    )
    surface_manifest_sha = _sha256(
        surface_manifest.get("sha256"), "surface-area manifest SHA-256"
    )
    if surface_manifest_sha != immutable_pins.surface_area_manifest_sha256:
        raise DrivAerDatasetScorerError(
            "submission specification surface-area manifest differs from the "
            "predeclared hash"
        )
    pin_surface_manifest = _mapping(
        pin_supports.get("surface_cell_area_manifest"),
        "native pin surface-cell-area manifest",
    )
    if pin_surface_manifest.get("sha256") != surface_manifest_sha:
        raise DrivAerDatasetScorerError(
            "surface-area manifest hash differs between contract and native pin"
        )

    profile_declaration = _mapping(
        specification.get("profile_definition"), "profile_definition"
    )
    profile_path = _declared_path(
        spec_path.parent, profile_declaration.get("file"), "profile definition file"
    )
    profile, _, profile_sha = _read_json(profile_path, "diagnostic profile definition")
    declared_profile_sha = _sha256(
        profile_declaration.get("sha256"), "profile definition SHA-256"
    )
    if declared_profile_sha != immutable_pins.diagnostic_profile_sha256:
        raise DrivAerDatasetScorerError(
            "submission specification diagnostic profile differs from the "
            "predeclared hash"
        )
    if profile_sha != declared_profile_sha:
        raise DrivAerDatasetScorerError("diagnostic profile definition SHA-256 mismatch")
    if profile.get("id") != "drivaerml-diagnostics-v9-candidate":
        raise DrivAerDatasetScorerError("diagnostic profile identity mismatch")
    if "pressure_profiles" in profile:
        raise DrivAerDatasetScorerError(
            "active diagnostic profile cannot contain discrete Cp probe panels"
        )
    source_bindings = _mapping(profile.get("source"), "diagnostic profile source")
    if any(key.startswith("cp_") or "probe" in key for key in source_bindings):
        raise DrivAerDatasetScorerError(
            "active diagnostic profile cannot bind discrete Cp probe support"
        )
    pressure_profile = _mapping(profile.get("pressure_cuts"), "pressure_cuts")
    velocity_profile = _mapping(profile.get("velocity_profiles"), "velocity_profiles")
    if (
        pressure_profile.get("ranked_metric_id") != "cp_cut_r2"
        or pressure_profile.get("report_only_metric_id") != "cp_cut_rmse"
        or pressure_profile.get("definition_authority") != "FluidsBench"
        or pressure_profile.get("association") != "native_surface_VTP_CellData"
        or pressure_profile.get("extraction_status")
        != "pending_immutable_owner_cut_support"
        or pressure_profile.get("reduction")
        != "equal_case_equal_cut_global_R2_with_normalized_native_intersection_segment_length_support_per_cut"
        or velocity_profile.get("ranked_metric_id") != "velocity_profile_r2"
        or velocity_profile.get("report_only_metric_id")
        != "velocity_profile_uinf_rmse"
        or velocity_profile.get("ranked_reduction")
        != "equal_case_equal_line_global_R2_with_normalized_trapezoidal_arc_length_support_per_line"
        or velocity_profile.get("definition_authority") != "AutoCFD5"
    ):
        raise DrivAerDatasetScorerError("diagnostic canonical definitions are inconsistent")
    stations = velocity_profile.get("stations")
    if not isinstance(stations, list):
        raise DrivAerDatasetScorerError("velocity profile stations must be an array")
    normalized_stations = tuple(
        _mapping(item, "velocity station") for item in stations
    )
    line_ids = tuple(
        _string(
            item.get("source_profile_id", item.get("id")),
            "velocity station source profile ID",
        )
        for item in normalized_stations
    )
    station_ids = tuple(
        _string(item.get("id"), "velocity station ID")
        for item in normalized_stations
    )
    experimental_line_ids = tuple(
        line_id
        for line_id, item in zip(line_ids, normalized_stations, strict=True)
        if item.get("experimental_availability", "none") != "none"
    )
    line_sample_counts = tuple(
        _integer(item.get("sample_count"), "velocity station sample count", minimum=2)
        for item in normalized_stations
    )
    if (
        len(line_ids) != len(set(line_ids))
        or len(station_ids) != len(set(station_ids))
        or velocity_profile.get("line_count") != len(line_ids)
        or not experimental_line_ids
    ):
        raise DrivAerDatasetScorerError("velocity profile line registry is inconsistent")
    sample_count = _integer(
        velocity_profile.get("sample_count_per_case"),
        "velocity sample count",
        minimum=1,
    )
    pressure_stations = pressure_profile.get("stations")
    if not isinstance(pressure_stations, list):
        raise DrivAerDatasetScorerError("pressure cut stations must be an array")
    cp_cut_ids = tuple(
        _string(
            _mapping(item, "pressure cut station").get("id"),
            "pressure cut station ID",
        )
        for item in pressure_stations
    )
    expected_cp_cut_ids = (
        "upperbody_centerline",
        "underbody_centerline",
        "sidewall_z_0_15",
        "front_left_wheelhouse_y_neg_0_6",
    )
    if (
        cp_cut_ids != expected_cp_cut_ids
        or pressure_profile.get("cut_count") != len(cp_cut_ids)
    ):
        raise DrivAerDatasetScorerError("pressure cut registry is inconsistent")
    force_constants = _mapping(scoring.get("force_integration"), "force_integration")

    return (
        _Contract(
            specification_path=spec_path,
            specification_sha256=spec_sha,
            split_id=split_id,
            case_set_id=_string(split["case_set_id"], "case_set_id"),
            split_file=split_path.name,
            split_sha256=split_sha,
            case_ids=test,
            native_source_pin=native_pin,
            native_source_pin_file=pin_path.name,
            native_source_pin_sha256=pin_sha,
            force_truth_file=truth_path.name,
            force_truth_sha256=truth_sha,
            profile_file=profile_path.name,
            profile_sha256=profile_sha,
            profile_velocity_line_ids=line_ids,
            profile_velocity_station_ids=station_ids,
            profile_velocity_line_sample_counts=line_sample_counts,
            profile_experimental_velocity_line_ids=experimental_line_ids,
            profile_velocity_sample_count=sample_count,
            profile_cp_cut_ids=cp_cut_ids,
            surface_area_manifest_sha256=surface_manifest_sha,
            force_constants=force_constants,
        ),
        truth,
    )


def _load_force_truth(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_case_ids: Sequence[str],
) -> tuple[dict[str, Mapping[str, float]], Path, str]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise DrivAerDatasetScorerError("force truth cannot be a symbolic link")
    rows: dict[str, Mapping[str, float]] = {}
    ordered: list[str] = []
    try:
        with RetainedVerifiedFile.open(unresolved, label="force truth") as retained:
            source = retained.source_path
            initial = retained.sha256(chunk_bytes=1024 * 1024)
            if initial != _sha256(
                expected_sha256, "predeclared force truth SHA-256"
            ):
                raise DrivAerDatasetScorerError(
                    "authoritative force truth SHA-256 mismatch"
                )
            retained.handle.seek(0)
            text = retained.handle.read().decode("utf-8")
            reader = csv.DictReader(text.splitlines(keepends=True))
            if reader.fieldnames != ["run", "cd", "cl", "clf", "clr", "cs"]:
                raise DrivAerDatasetScorerError(
                    "authoritative force truth header is not exact"
                )
            for offset, row in enumerate(reader, start=2):
                if None in row or set(row) != set(reader.fieldnames):
                    raise DrivAerDatasetScorerError(
                        f"force truth row {offset} has the wrong number of columns"
                    )
                run = _string(row["run"], f"force truth row {offset} run")
                if not re.fullmatch(r"[1-9][0-9]*", run):
                    raise DrivAerDatasetScorerError(
                        f"force truth row {offset} run is not canonical"
                    )
                case_id = f"run_{int(run)}"
                if case_id in rows:
                    raise DrivAerDatasetScorerError(
                        f"force truth contains duplicate case {case_id}"
                    )
                values: dict[str, float] = {}
                for key in ("cd", "cl", "clf", "clr", "cs"):
                    try:
                        parsed = float(
                            _string(row[key], f"force truth {case_id}/{key}")
                        )
                    except ValueError as error:
                        raise DrivAerDatasetScorerError(
                            f"force truth {case_id}/{key} must be numeric"
                        ) from error
                    if not math.isfinite(parsed):
                        raise DrivAerDatasetScorerError(
                            f"force truth {case_id}/{key} must be finite"
                        )
                    values[key] = parsed
                rows[case_id] = values
                ordered.append(case_id)
            retained.assert_unchanged(context="while its CSV was parsed")
    except DrivAerDatasetScorerError:
        raise
    except RetainedFileError as error:
        raise DrivAerDatasetScorerError(str(error)) from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise DrivAerDatasetScorerError("cannot read authoritative force truth") from error
    if tuple(ordered) != tuple(expected_case_ids):
        missing = sorted(set(expected_case_ids) - set(ordered))
        extra = sorted(set(ordered) - set(expected_case_ids))
        raise DrivAerDatasetScorerError(
            "force truth case coverage/order differs from the native source pin "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    return rows, source, initial


def _load_evidence_documents(
    paths: Sequence[Path | str],
    *,
    expected_case_ids: Sequence[str],
    label: str,
) -> dict[str, tuple[Mapping[str, Any], Path, str]]:
    if isinstance(paths, (str, bytes, Path)) or not isinstance(paths, Sequence):
        raise DrivAerDatasetScorerError(f"{label} paths must be a sequence")
    loaded: dict[str, tuple[Mapping[str, Any], Path, str]] = {}
    seen_paths: set[Path] = set()
    for value in paths:
        document, path, sha = _read_json(value, label)
        if path in seen_paths:
            raise DrivAerDatasetScorerError(f"duplicate {label} path {path.name!r}")
        seen_paths.add(path)
        case_id = _case_id(document.get("case_id"), f"{label} case_id")
        if case_id in loaded:
            raise DrivAerDatasetScorerError(
                f"duplicate {label} for case {case_id}"
            )
        loaded[case_id] = (document, path, sha)
    expected = tuple(expected_case_ids)
    if len(loaded) != len(expected) or set(loaded) != set(expected):
        missing = [case_id for case_id in expected if case_id not in loaded]
        extra = [case_id for case_id in loaded if case_id not in set(expected)]
        raise DrivAerDatasetScorerError(
            f"{label} must cover the selected split exactly "
            f"(missing={missing}, extra={extra})"
        )
    return loaded


def _validate_additive_sums(
    value: object,
    *,
    label: str,
    entity_count: int,
    expected_weightings: tuple[str, ...],
) -> Mapping[str, Mapping[str, float | int]]:
    root = _exact_keys(value, expected_weightings, label)
    result: dict[str, Mapping[str, float | int]] = {}
    for weighting in expected_weightings:
        sums = _exact_keys(
            root[weighting],
            {"absolute_error", "squared_error", "squared_truth", "entity_count", "total_weight"},
            f"{label}.{weighting}",
        )
        count = _integer(
            sums["entity_count"], f"{label}.{weighting}.entity_count", minimum=1
        )
        if count != entity_count:
            raise DrivAerDatasetScorerError(
                f"{label}.{weighting}.entity_count differs from native coverage"
            )
        absolute = _finite(
            sums["absolute_error"], f"{label}.{weighting}.absolute_error", nonnegative=True
        )
        squared = _finite(
            sums["squared_error"], f"{label}.{weighting}.squared_error", nonnegative=True
        )
        truth = _finite(
            sums["squared_truth"], f"{label}.{weighting}.squared_truth", nonnegative=True
        )
        if truth <= 0.0:
            raise DrivAerDatasetScorerError(
                f"{label}.{weighting}.squared_truth must be positive"
            )
        total_weight = _finite(
            sums["total_weight"], f"{label}.{weighting}.total_weight", nonnegative=True
        )
        if total_weight <= 0.0:
            raise DrivAerDatasetScorerError(
                f"{label}.{weighting}.total_weight must be positive"
            )
        if weighting == "uniform" and total_weight != float(entity_count):
            raise DrivAerDatasetScorerError(
                f"{label}.uniform.total_weight must equal entity_count"
            )
        result[weighting] = {
            "absolute_error": absolute,
            "squared_error": squared,
            "squared_truth": truth,
            "entity_count": count,
            "total_weight": total_weight,
        }
    return result


def _validate_field_metrics(
    root: Mapping[str, Any],
    *,
    surface_count: int,
    volume_count: int | None,
    prediction_scope: str = PREDICTION_SCOPE_FULL,
) -> tuple[dict[str, float], dict[str, Mapping[str, object]]]:
    if prediction_scope not in PREDICTION_SCOPES:
        raise DrivAerDatasetScorerError("core prediction scope differs")
    prefixes = tuple(
        prefix
        for prefix, layout in _FIELD_LAYOUT.items()
        if prediction_scope == PREDICTION_SCOPE_FULL
        or layout["support"] == "surface_native_cells"
    )
    expected_metric_ids = tuple(
        metric_id for prefix in prefixes for metric_id in _field_metric_ids(prefix)
    )
    expected_relative_ids = tuple(
        metric_id for metric_id in expected_metric_ids if metric_id.endswith("_rel_l2")
    )
    metric_values = _exact_keys(
        root.get("metric_values"), expected_metric_ids, "core metric_values"
    )
    normalized_values = {
        metric_id: _finite(value, f"core metric_values.{metric_id}", nonnegative=True)
        for metric_id, value in metric_values.items()
    }
    additive = _exact_keys(
        root.get("additive_sums"), prefixes, "core additive_sums"
    )
    sufficient = _exact_keys(
        root.get("metric_sufficient_statistics"),
        expected_relative_ids,
        "core metric_sufficient_statistics",
    )
    normalized_statistics: dict[str, Mapping[str, object]] = {}
    for prefix in prefixes:
        layout = _FIELD_LAYOUT[prefix]
        entity_count = surface_count if layout["support"] == "surface_native_cells" else volume_count
        if entity_count is None:
            raise DrivAerDatasetScorerError("volume field metrics require volume coverage")
        expected_weightings = (
            ("uniform", "physical")
            if layout["support"] == "surface_native_cells"
            else ("uniform",)
        )
        sums = _validate_additive_sums(
            additive[prefix],
            label=f"core additive_sums.{prefix}",
            entity_count=entity_count,
            expected_weightings=expected_weightings,
        )
        primary_weighting = str(layout["primary_weighting"])
        primary = sums[primary_weighting]
        metric_ids = _field_metric_ids(prefix)
        primary_rel_id = metric_ids[0]
        primary_mae_id = (
            metric_ids[2]
            if layout["support"] == "surface_native_cells"
            else metric_ids[1]
        )
        primary_rmse_id = (
            metric_ids[3]
            if layout["support"] == "surface_native_cells"
            else metric_ids[2]
        )
        expected_values = {
            primary_rel_id: 100.0 * math.sqrt(float(primary["squared_error"]) / float(primary["squared_truth"])),
            # MAE and RMSE here consume DrivAerML's per-entity Euclidean-vector
            # sums.  They are intentionally not generic flattened-component
            # reductions.
            primary_mae_id: float(primary["absolute_error"]) / float(primary["total_weight"]),
            primary_rmse_id: math.sqrt(float(primary["squared_error"]) / float(primary["total_weight"])),
        }
        relative_statistics = [
            (
                primary_rel_id,
                primary_weighting,
                layout["primary_dataset_weighting"],
            )
        ]
        if layout["support"] == "surface_native_cells":
            secondary_weighting = str(layout["secondary_weighting"])
            secondary = sums[secondary_weighting]
            secondary_rel_id = metric_ids[1]
            secondary_mae_id = metric_ids[4]
            secondary_rmse_id = metric_ids[5]
            expected_values.update(
                {
                    secondary_rel_id: 100.0
                    * math.sqrt(
                        float(secondary["squared_error"])
                        / float(secondary["squared_truth"])
                    ),
                    secondary_mae_id: float(secondary["absolute_error"])
                    / float(secondary["total_weight"]),
                    secondary_rmse_id: math.sqrt(
                        float(secondary["squared_error"])
                        / float(secondary["total_weight"])
                    ),
                }
            )
            relative_statistics.append(
                (
                    secondary_rel_id,
                    secondary_weighting,
                    layout["secondary_dataset_weighting"],
                )
            )
        for metric_id, expected in expected_values.items():
            _require_same_float(
                normalized_values[metric_id], expected, f"core metric_values.{metric_id}"
            )
        for metric_id, weighting, dataset_weighting in relative_statistics:
            statistics = _exact_keys(
                sufficient[metric_id],
                {"reduction", "weighting", "dataset_weighting", "numerator", "denominator", "entity_count", "total_weight"},
                f"core metric_sufficient_statistics.{metric_id}",
            )
            source = sums[weighting]
            expected_weighting = "uniform" if weighting == "uniform" else "support_weights"
            if (
                statistics["reduction"] != "relative_l2_percent"
                or statistics["weighting"] != expected_weighting
                or statistics["dataset_weighting"] != dataset_weighting
                or statistics["entity_count"] != entity_count
            ):
                raise DrivAerDatasetScorerError(
                    f"core metric_sufficient_statistics.{metric_id} identity mismatch"
                )
            for key, expected in (
                ("numerator", float(source["squared_error"])),
                ("denominator", float(source["squared_truth"])),
                ("total_weight", float(source["total_weight"])),
            ):
                _require_same_float(
                    statistics[key], expected, f"core metric_sufficient_statistics.{metric_id}.{key}"
                )
            normalized_statistics[metric_id] = dict(statistics)
    return normalized_values, normalized_statistics


def _validate_surface_only_core_case(
    root: Mapping[str, Any],
    path: Path,
    digest: str,
    *,
    contract: _Contract,
    pinned: NativeCaseRecord,
) -> _CoreCase:
    """Validate v4 surface-only evidence without inspecting native volume data."""

    case_id = pinned.case_id
    source = _exact_keys(
        root["source"],
        {
            "native_source_pin_sha256", "repository_id", "repository_revision",
            "boundary_sha256", "surface_native", "surface_area", "volume_native",
        },
        f"core evidence {case_id}.source",
    )
    if (
        source["native_source_pin_sha256"] != contract.native_source_pin_sha256
        or source["repository_id"] != contract.native_source_pin.repository_id
        or source["repository_revision"] != contract.native_source_pin.repository_revision
        or source["boundary_sha256"] != pinned.boundary.sha256
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} source identity mismatch")
    if source["volume_native"] != {
        "status": "not_loaded_surface_only",
        "scientific_metric_values_fabricated": False,
    }:
        raise DrivAerDatasetScorerError(
            f"core evidence {case_id} surface-only volume status differs"
        )
    surface_native = _exact_keys(
        source["surface_native"],
        {
            "source_file", "boundary_sha256", "vtk_version", "point_count",
            "polygon_count", "raw_cell_order", "association", "arrays",
            "available_point_arrays", "available_cell_arrays",
        },
        f"core evidence {case_id}.surface_native",
    )
    surface_count = _integer(
        surface_native["polygon_count"], f"{case_id} surface count", minimum=1
    )
    if (
        surface_native["source_file"] != pinned.boundary.path.name
        or surface_native["boundary_sha256"] != pinned.boundary.sha256
        or surface_native["raw_cell_order"] != "zero_based_native_vtk_polygon_order_unchanged"
        or surface_native["association"] != "CellData"
        or surface_count != pinned.surface_cell_area.element_count
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} native surface mismatch")
    arrays = _mapping(surface_native["arrays"], f"{case_id} surface arrays")
    if set(arrays) != {"pMeanTrim", "wallShearStressMeanTrim"}:
        raise DrivAerDatasetScorerError(f"core evidence {case_id} surface arrays mismatch")
    for name, components in (("pMeanTrim", 1), ("wallShearStressMeanTrim", 3)):
        item = _mapping(arrays[name], f"{case_id} surface array {name}")
        if (
            item.get("components") != components
            or item.get("tuples") != surface_count
            or item.get("unit") != "m^2/s^2"
        ):
            raise DrivAerDatasetScorerError(
                f"core evidence {case_id} surface array {name} mismatch"
            )
    surface_area = _exact_keys(
        source["surface_area"],
        {
            "source_path", "sha256", "source_boundary_sha256", "entity_count",
            "area_sum_m2", "area_min_m2", "area_max_m2", "dtype", "role",
            "native_geometry_order_audit",
        },
        f"core evidence {case_id}.surface_area",
    )
    area_sum = _finite(surface_area["area_sum_m2"], f"{case_id} area sum")
    area_min = _finite(surface_area["area_min_m2"], f"{case_id} minimum area")
    area_max = _finite(surface_area["area_max_m2"], f"{case_id} maximum area")
    if (
        surface_area["source_path"] != pinned.surface_cell_area.path.name
        or surface_area["sha256"] != pinned.surface_cell_area.sha256
        or surface_area["source_boundary_sha256"] != pinned.boundary.sha256
        or surface_area["entity_count"] != surface_count
        or surface_area["dtype"] != "<f4"
        or surface_area["role"] != "fixed_external_input_not_regenerated"
        or area_min <= 0.0
        or area_max < area_min
        or area_sum <= 0.0
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} fixed surface-area mismatch")
    geometry_area = _exact_keys(
        surface_area["native_geometry_order_audit"],
        {
            "entity_count", "relative_tolerance", "calculated_sum_m2",
            "published_sum_m2", "maximum_absolute_difference_m2",
            "maximum_relative_difference", "raw_order_correspondence_verified",
            "published_values_role",
        },
        f"core evidence {case_id}.surface area geometry audit",
    )
    if (
        geometry_area["entity_count"] != surface_count
        or not _same_float(
            _finite(geometry_area["relative_tolerance"], f"{case_id} area tolerance"),
            SURFACE_AREA_GEOMETRY_RELATIVE_TOLERANCE,
        )
        or not _same_float(
            _finite(geometry_area["published_sum_m2"], f"{case_id} published area"),
            area_sum,
        )
        or _finite(geometry_area["calculated_sum_m2"], f"{case_id} calculated area") <= 0.0
        or _finite(geometry_area["maximum_absolute_difference_m2"], f"{case_id} area difference", nonnegative=True) < 0.0
        or _finite(geometry_area["maximum_relative_difference"], f"{case_id} area relative difference", nonnegative=True) > SURFACE_AREA_GEOMETRY_RELATIVE_TOLERANCE
        or geometry_area["raw_order_correspondence_verified"] is not True
        or geometry_area["published_values_role"] != "fixed_input_audited_not_regenerated"
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} geometry-area audit mismatch")

    predictions = _exact_keys(
        root["prediction_inputs"], {"surface_native_cells"},
        f"core evidence {case_id}.prediction_inputs",
    )
    record = _exact_keys(
        predictions["surface_native_cells"],
        {"manifest_sha256", "chunk_sha256", "chunk_count", "entity_count"},
        f"core evidence {case_id}.surface prediction",
    )
    chunks = record["chunk_sha256"]
    if not isinstance(chunks, list) or not chunks:
        raise DrivAerDatasetScorerError(f"{case_id} surface chunks must be non-empty")
    surface_chunks = tuple(
        _sha256(item, f"{case_id} surface chunk SHA-256") for item in chunks
    )
    if record["chunk_count"] != len(surface_chunks) or record["entity_count"] != surface_count:
        raise DrivAerDatasetScorerError(f"{case_id} surface prediction coverage mismatch")
    surface_manifest = _sha256(record["manifest_sha256"], f"{case_id} surface manifest SHA-256")
    coverage = _exact_keys(root["coverage"], {"surface", "volume"}, f"core evidence {case_id}.coverage")
    surface_coverage = _mapping(coverage["surface"], f"{case_id} surface coverage")
    if (
        surface_coverage.get("raw_cell_id_start") != 0
        or surface_coverage.get("raw_cell_id_stop") != surface_count
        or surface_coverage.get("complete_gap_free_duplicate_free") is not True
        or coverage["volume"] != {
            "status": "not_submitted_surface_only", "component_score": 0.0
        }
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} surface-only coverage mismatch")
    field_metrics, relative_statistics = _validate_field_metrics(
        root,
        surface_count=surface_count,
        volume_count=None,
        prediction_scope=PREDICTION_SCOPE_SURFACE_ONLY,
    )
    force = _exact_keys(
        root["force_coefficients"],
        {"entity_count", "force_n", "moment_about_forces_cor_n_m", "Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr", "lift_closure_abs"},
        f"core evidence {case_id}.force_coefficients",
    )
    if force["entity_count"] != surface_count:
        raise DrivAerDatasetScorerError(f"core evidence {case_id} force entity count mismatch")
    normalized_force = {
        key: _finite(force[key], f"{case_id} predicted {key}")
        for key in ("Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr", "lift_closure_abs")
    }
    vectors: dict[str, tuple[float, float, float]] = {}
    for key in ("force_n", "moment_about_forces_cor_n_m"):
        raw = force[key]
        if not isinstance(raw, list) or len(raw) != 3:
            raise DrivAerDatasetScorerError(f"{case_id} {key} must have three values")
        vectors[key] = tuple(_finite(value, f"{case_id} {key}") for value in raw)
    u_inf = _finite(contract.force_constants.get("freestream_velocity_m_per_s"), "force freestream velocity", nonnegative=True)
    density = _finite(contract.force_constants.get("density_kg_per_m3"), "force density", nonnegative=True)
    area = _finite(contract.force_constants.get("reference_area_m2"), "force reference area", nonnegative=True)
    length = _finite(contract.force_constants.get("reference_length_m"), "force reference length", nonnegative=True)
    if min(u_inf, density, area, length) <= 0.0:
        raise DrivAerDatasetScorerError("force normalization constants must be positive")
    q_area = 0.5 * density * u_inf**2 * area
    for key, expected in {
        "Cd": vectors["force_n"][0] / q_area,
        "Cl": vectors["force_n"][2] / q_area,
        "Cs": vectors["force_n"][1] / q_area,
        "CmPitch": vectors["moment_about_forces_cor_n_m"][1] / (q_area * length),
    }.items():
        _require_same_float(normalized_force[key], expected, f"{case_id} predicted {key}")
    _require_same_float(normalized_force["Clf"], normalized_force["Cl"] / 2.0 + normalized_force["CmPitch"], f"{case_id} predicted Clf")
    _require_same_float(normalized_force["Clr"], normalized_force["Cl"] / 2.0 - normalized_force["CmPitch"], f"{case_id} predicted Clr")
    _require_same_float(normalized_force["lift_closure_abs"], abs(normalized_force["Cl"] - normalized_force["Clf"] - normalized_force["Clr"]), f"{case_id} predicted lift closure")
    execution = _exact_keys(
        root["execution"],
        {"maximum_prediction_chunk_rows", "hash_chunk_bytes", "validation_block_rows"},
        f"core evidence {case_id}.execution",
    )
    for key, value in execution.items():
        _integer(value, f"core evidence {case_id}.execution.{key}", minimum=1)
    regional = _mapping(root["report_only_regional_diagnostics"], f"core evidence {case_id} regional diagnostics")
    try:
        validate_case_regional_envelope(
            regional, expected_case_id=case_id, expected_additive_sums=root["additive_sums"]
        )
    except RegionalAggregateError as error:
        raise DrivAerDatasetScorerError(
            f"core evidence {case_id} regional diagnostics are invalid: {error}"
        ) from error
    return _CoreCase(
        case_id=case_id,
        input_file=path.name,
        input_sha256=digest,
        prediction_scope=PREDICTION_SCOPE_SURFACE_ONLY,
        surface_count=surface_count,
        volume_count=None,
        field_metrics=field_metrics,
        relative_l2_statistics=relative_statistics,
        force_coefficients=normalized_force,
        surface_prediction_manifest_sha256=surface_manifest,
        volume_prediction_manifest_sha256=None,
        surface_prediction_chunk_sha256=surface_chunks,
        volume_prediction_chunk_sha256=None,
        boundary_sha256=pinned.boundary.sha256,
        surface_area_sha256=pinned.surface_cell_area.sha256,
        volume_part_sha256=None,
        regional_diagnostics=dict(regional),
    )


def _validate_core_case(
    document: Mapping[str, Any],
    path: Path,
    digest: str,
    *,
    contract: _Contract,
    pinned: NativeCaseRecord,
) -> _CoreCase:
    case_id = pinned.case_id
    legacy_keys = {
        "schema", "schema_version", "status", "official_submission", "case_id",
        "source", "prediction_inputs", "coverage", "metric_values",
        "metric_sufficient_statistics", "additive_sums", "force_coefficients",
        "execution",
    }
    v4_keys = legacy_keys | {"prediction_scope", "report_only_regional_diagnostics"}
    root = _mapping(document, f"core evidence {case_id}")
    legacy = set(root) == legacy_keys
    if set(root) != legacy_keys and set(root) != v4_keys:
        raise DrivAerDatasetScorerError(
            f"core evidence {case_id} keys differ from the frozen candidate schema"
        )
    prediction_scope = (
        PREDICTION_SCOPE_FULL if legacy else root.get("prediction_scope")
    )
    if (
        (legacy and (root["schema"] != LEGACY_CORE_CASE_SCHEMA or root["schema_version"] != 2))
        or (
            not legacy
            and (root["schema"] != CORE_CASE_SCHEMA or root["schema_version"] != 4)
        )
        or root["status"] != CORE_CASE_STATUS
        or root["official_submission"] is not False
        or root["case_id"] != case_id
        or prediction_scope not in PREDICTION_SCOPES
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} schema/status mismatch")
    if prediction_scope == PREDICTION_SCOPE_SURFACE_ONLY:
        return _validate_surface_only_core_case(
            root, path, digest, contract=contract, pinned=pinned
        )
    source = _exact_keys(
        root["source"],
        {
            "native_source_pin_sha256", "repository_id", "repository_revision", "boundary_sha256",
            "surface_native", "surface_area", "volume_part_sha256", "volume_vtk", "volume_weighting",
            "volume_native_arrays",
        },
        f"core evidence {case_id}.source",
    )
    if (
        source["native_source_pin_sha256"] != contract.native_source_pin_sha256
        or source["repository_id"] != contract.native_source_pin.repository_id
        or source["repository_revision"] != contract.native_source_pin.repository_revision
        or source["boundary_sha256"] != pinned.boundary.sha256
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} source identity mismatch")
    surface_native = _exact_keys(
        source["surface_native"],
        {"source_file", "boundary_sha256", "vtk_version", "point_count", "polygon_count", "raw_cell_order", "association", "arrays", "available_point_arrays", "available_cell_arrays"},
        f"core evidence {case_id}.surface_native",
    )
    surface_count = _integer(surface_native["polygon_count"], f"{case_id} surface count", minimum=1)
    if (
        surface_native["source_file"] != pinned.boundary.path.name
        or surface_native["boundary_sha256"] != pinned.boundary.sha256
        or surface_native["raw_cell_order"] != "zero_based_native_vtk_polygon_order_unchanged"
        or surface_native["association"] != "CellData"
        or surface_count != pinned.surface_cell_area.element_count
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} native surface mismatch")
    arrays = _mapping(surface_native["arrays"], f"{case_id} surface arrays")
    if set(arrays) != {"pMeanTrim", "wallShearStressMeanTrim"}:
        raise DrivAerDatasetScorerError(f"core evidence {case_id} surface arrays mismatch")
    for name, components in (("pMeanTrim", 1), ("wallShearStressMeanTrim", 3)):
        item = _mapping(arrays[name], f"{case_id} surface array {name}")
        if item.get("components") != components or item.get("tuples") != surface_count or item.get("unit") != "m^2/s^2":
            raise DrivAerDatasetScorerError(f"core evidence {case_id} surface array {name} mismatch")
    surface_area = _exact_keys(
        source["surface_area"],
        {
            "source_path", "sha256", "source_boundary_sha256", "entity_count",
            "area_sum_m2", "area_min_m2", "area_max_m2", "dtype", "role",
            "native_geometry_order_audit",
        },
        f"core evidence {case_id}.surface_area",
    )
    area_sum = _finite(surface_area["area_sum_m2"], f"{case_id} area sum")
    area_min = _finite(surface_area["area_min_m2"], f"{case_id} minimum area")
    area_max = _finite(surface_area["area_max_m2"], f"{case_id} maximum area")
    if (
        surface_area["source_path"] != pinned.surface_cell_area.path.name
        or surface_area["sha256"] != pinned.surface_cell_area.sha256
        or surface_area["source_boundary_sha256"] != pinned.boundary.sha256
        or surface_area["entity_count"] != surface_count
        or surface_area["dtype"] != "<f4"
        or surface_area["role"] != "fixed_external_input_not_regenerated"
        or area_min <= 0.0
        or area_max < area_min
        or area_sum <= 0.0
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} fixed surface-area mismatch")
    geometry_area = _exact_keys(
        surface_area["native_geometry_order_audit"],
        {
            "entity_count", "relative_tolerance", "calculated_sum_m2",
            "published_sum_m2", "maximum_absolute_difference_m2",
            "maximum_relative_difference", "raw_order_correspondence_verified",
            "published_values_role",
        },
        f"core evidence {case_id}.source.surface_area.native_geometry_order_audit",
    )
    geometry_tolerance = _finite(
        geometry_area["relative_tolerance"],
        f"{case_id} surface geometry relative tolerance",
        nonnegative=True,
    )
    calculated_sum = _finite(
        geometry_area["calculated_sum_m2"],
        f"{case_id} calculated geometry area sum",
    )
    published_sum = _finite(
        geometry_area["published_sum_m2"],
        f"{case_id} published geometry area sum",
    )
    maximum_absolute_difference = _finite(
        geometry_area["maximum_absolute_difference_m2"],
        f"{case_id} maximum absolute geometry-area difference",
        nonnegative=True,
    )
    maximum_relative_difference = _finite(
        geometry_area["maximum_relative_difference"],
        f"{case_id} maximum relative geometry-area difference",
        nonnegative=True,
    )
    if (
        geometry_area["entity_count"] != surface_count
        or not _same_float(
            geometry_tolerance, SURFACE_AREA_GEOMETRY_RELATIVE_TOLERANCE
        )
        or calculated_sum <= 0.0
        or published_sum <= 0.0
        or not _same_float(published_sum, area_sum)
        or maximum_absolute_difference < 0.0
        or maximum_relative_difference > geometry_tolerance
        or geometry_area["raw_order_correspondence_verified"] is not True
        or geometry_area["published_values_role"]
        != "fixed_input_audited_not_regenerated"
    ):
        raise DrivAerDatasetScorerError(
            f"core evidence {case_id} native geometry/area order audit mismatch"
        )

    expected_parts = tuple(part.sha256 for part in pinned.volume_parts)
    part_hashes = source["volume_part_sha256"]
    if not isinstance(part_hashes, list) or tuple(part_hashes) != expected_parts:
        raise DrivAerDatasetScorerError(f"core evidence {case_id} multipart identity mismatch")
    volume_vtk = _mapping(source["volume_vtk"], f"{case_id} volume_vtk")
    volume_count = _integer(volume_vtk.get("cell_count"), f"{case_id} volume count", minimum=1)
    if (
        volume_vtk.get("dataset_type") != "UnstructuredGrid"
        or volume_vtk.get("piece_count") != 1
        or volume_vtk.get("source_size_bytes") != pinned.volume_total_size_bytes
    ):
        raise DrivAerDatasetScorerError(f"core evidence {case_id} native volume mismatch")
    volume_weighting = _exact_keys(
        source["volume_weighting"],
        {
            "weighting",
            "entity_count",
            "total_weight",
            "geometric_cell_volume_weights_used",
        },
        f"core evidence {case_id}.volume_weighting",
    )
    volume_weighting_count = _integer(
        volume_weighting["entity_count"],
        f"{case_id} volume weighting entity count",
        minimum=1,
    )
    total_volume_weight = _finite(
        volume_weighting["total_weight"],
        f"{case_id} volume weighting total weight",
        nonnegative=True,
    )
    if (
        volume_weighting["weighting"] != "one_per_native_cell"
        or volume_weighting_count != volume_count
        or not isinstance(volume_weighting["total_weight"], float)
        or total_volume_weight != float(volume_count)
        or volume_weighting["geometric_cell_volume_weights_used"] is not False
    ):
        raise DrivAerDatasetScorerError(
            f"core evidence {case_id} volume weighting mismatch"
        )

    native_arrays = _mapping(source["volume_native_arrays"], f"{case_id} volume native arrays")
    if set(native_arrays) != {"pMeanTrim", "UMeanTrim"}:
        raise DrivAerDatasetScorerError(f"core evidence {case_id} native volume arrays mismatch")
    for name, components, unit in (("pMeanTrim", 1, "m^2/s^2"), ("UMeanTrim", 3, "m/s")):
        audit = _mapping(native_arrays[name], f"{case_id} native array {name}")
        if (
            audit.get("name") != name
            or audit.get("association") != "CellData"
            or audit.get("number_of_components") != components
            or audit.get("tuple_count") != volume_count
            or audit.get("scalar_count") != volume_count * components
            or audit.get("finite") is not True
            or audit.get("units") != unit
            or audit.get("raw_id_start") != 0
            or audit.get("raw_id_stop") != volume_count
        ):
            raise DrivAerDatasetScorerError(f"core evidence {case_id} native array {name} mismatch")
        _sha256(audit.get("payload_sha256"), f"{case_id} native array payload SHA-256")

    predictions = _exact_keys(
        root["prediction_inputs"],
        {"surface_native_cells", "volume_native_cells"},
        f"core evidence {case_id}.prediction_inputs",
    )
    prediction_records: dict[str, tuple[str, tuple[str, ...]]] = {}
    for support_id, count in (("surface_native_cells", surface_count), ("volume_native_cells", volume_count)):
        record = _exact_keys(
            predictions[support_id],
            {"manifest_sha256", "chunk_sha256", "chunk_count", "entity_count"},
            f"core evidence {case_id}.prediction_inputs.{support_id}",
        )
        chunks = record["chunk_sha256"]
        if not isinstance(chunks, list) or not chunks:
            raise DrivAerDatasetScorerError(f"{case_id}/{support_id} chunks must be non-empty")
        chunk_hashes = tuple(_sha256(item, f"{case_id}/{support_id} chunk SHA-256") for item in chunks)
        if record["chunk_count"] != len(chunk_hashes) or record["entity_count"] != count:
            raise DrivAerDatasetScorerError(f"{case_id}/{support_id} prediction coverage mismatch")
        prediction_records[support_id] = (
            _sha256(record["manifest_sha256"], f"{case_id}/{support_id} manifest SHA-256"),
            chunk_hashes,
        )
    coverage = _exact_keys(root["coverage"], {"surface", "volume"}, f"core evidence {case_id}.coverage")
    for key, count in (("surface", surface_count), ("volume", volume_count)):
        item = _mapping(coverage[key], f"{case_id} {key} coverage")
        if item.get("raw_cell_id_start") != 0 or item.get("raw_cell_id_stop") != count or item.get("complete_gap_free_duplicate_free") is not True:
            raise DrivAerDatasetScorerError(f"core evidence {case_id} {key} coverage mismatch")

    field_metrics, relative_statistics = _validate_field_metrics(
        root,
        surface_count=surface_count,
        volume_count=volume_count,
        prediction_scope=PREDICTION_SCOPE_FULL,
    )
    force = _exact_keys(
        root["force_coefficients"],
        {"entity_count", "force_n", "moment_about_forces_cor_n_m", "Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr", "lift_closure_abs"},
        f"core evidence {case_id}.force_coefficients",
    )
    if force["entity_count"] != surface_count:
        raise DrivAerDatasetScorerError(f"core evidence {case_id} force entity count mismatch")
    normalized_force = {
        key: _finite(force[key], f"{case_id} predicted {key}")
        for key in ("Cd", "Cl", "Cs", "CmPitch", "Clf", "Clr", "lift_closure_abs")
    }
    force_vectors: dict[str, tuple[float, float, float]] = {}
    for vector_name in ("force_n", "moment_about_forces_cor_n_m"):
        vector = force[vector_name]
        if not isinstance(vector, list) or len(vector) != 3:
            raise DrivAerDatasetScorerError(f"{case_id} {vector_name} must have three values")
        force_vectors[vector_name] = tuple(
            _finite(value, f"{case_id} {vector_name}[{axis}]")
            for axis, value in enumerate(vector)
        )
    u_inf = _finite(
        contract.force_constants.get("freestream_velocity_m_per_s"),
        "force freestream velocity",
        nonnegative=True,
    )
    density = _finite(
        contract.force_constants.get("density_kg_per_m3"),
        "force density",
        nonnegative=True,
    )
    reference_area = _finite(
        contract.force_constants.get("reference_area_m2"),
        "force reference area",
        nonnegative=True,
    )
    reference_length = _finite(
        contract.force_constants.get("reference_length_m"),
        "force reference length",
        nonnegative=True,
    )
    if min(u_inf, density, reference_area, reference_length) <= 0.0:
        raise DrivAerDatasetScorerError("force normalization constants must be positive")
    q_area = 0.5 * density * u_inf**2 * reference_area
    vector_coefficients = {
        "Cd": force_vectors["force_n"][0] / q_area,
        "Cl": force_vectors["force_n"][2] / q_area,
        "Cs": force_vectors["force_n"][1] / q_area,
        "CmPitch": force_vectors["moment_about_forces_cor_n_m"][1]
        / (q_area * reference_length),
    }
    for key, expected in vector_coefficients.items():
        _require_same_float(
            normalized_force[key], expected, f"{case_id} predicted {key}"
        )
    expected_clf = normalized_force["Cl"] / 2.0 + normalized_force["CmPitch"]
    expected_clr = normalized_force["Cl"] / 2.0 - normalized_force["CmPitch"]
    _require_same_float(normalized_force["Clf"], expected_clf, f"{case_id} predicted Clf")
    _require_same_float(normalized_force["Clr"], expected_clr, f"{case_id} predicted Clr")
    closure = abs(normalized_force["Cl"] - (normalized_force["Clf"] + normalized_force["Clr"]))
    _require_same_float(normalized_force["lift_closure_abs"], closure, f"{case_id} predicted lift closure")

    execution = _exact_keys(
        root["execution"],
        {"maximum_prediction_chunk_rows", "hash_chunk_bytes", "validation_block_rows", "encoded_chunk_bytes"},
        f"core evidence {case_id}.execution",
    )
    for key, value in execution.items():
        _integer(value, f"core evidence {case_id}.execution.{key}", minimum=1)
    regional_diagnostics: Mapping[str, object] | None = None
    if "report_only_regional_diagnostics" in root:
        regional_diagnostics = _mapping(
            root["report_only_regional_diagnostics"],
            f"core evidence {case_id}.report_only_regional_diagnostics",
        )
        try:
            validate_case_regional_envelope(
                regional_diagnostics,
                expected_case_id=case_id,
                expected_additive_sums=root["additive_sums"],
            )
        except RegionalAggregateError as error:
            raise DrivAerDatasetScorerError(
                f"core evidence {case_id} regional diagnostics are invalid: {error}"
            ) from error
    return _CoreCase(
        case_id=case_id,
        input_file=path.name,
        input_sha256=digest,
        prediction_scope=PREDICTION_SCOPE_FULL,
        surface_count=surface_count,
        volume_count=volume_count,
        field_metrics=field_metrics,
        relative_l2_statistics=relative_statistics,
        force_coefficients=normalized_force,
        surface_prediction_manifest_sha256=prediction_records["surface_native_cells"][0],
        volume_prediction_manifest_sha256=prediction_records["volume_native_cells"][0],
        surface_prediction_chunk_sha256=prediction_records["surface_native_cells"][1],
        volume_prediction_chunk_sha256=prediction_records["volume_native_cells"][1],
        boundary_sha256=pinned.boundary.sha256,
        surface_area_sha256=pinned.surface_cell_area.sha256,
        volume_part_sha256=expected_parts,
        regional_diagnostics=(
            dict(regional_diagnostics) if regional_diagnostics is not None else None
        ),
    )


def _validate_sparse_audit(
    value: object,
    *,
    label: str,
    support_id: str,
    field_name: str,
    total_count: int,
    source_files: tuple[str, ...] | None = None,
    source_hashes: tuple[str, ...] | None = None,
    manifest_sha256: str | None = None,
    chunk_sha256: tuple[str, ...] | None = None,
) -> None:
    if manifest_sha256 is not None:
        audit = _exact_keys(
            value,
            {"support_id", "field_name", "total_row_count", "manifest_file", "manifest_sha256", "chunk_count", "chunk_sha256", "complete_gap_free_duplicate_free_coverage", "selected_unique_raw_cell_id_count", "selected_values_sha256"},
            label,
        )
        _basename(audit["manifest_file"], f"{label}.manifest_file")
        chunks = audit["chunk_sha256"]
        if not isinstance(chunks, list) or tuple(chunks) != chunk_sha256:
            raise DrivAerDatasetScorerError(f"{label} chunk identity mismatch")
        if audit["manifest_sha256"] != manifest_sha256 or audit["chunk_count"] != len(chunk_sha256 or ()) or audit["complete_gap_free_duplicate_free_coverage"] is not True:
            raise DrivAerDatasetScorerError(f"{label} prediction identity mismatch")
    else:
        audit = _exact_keys(
            value,
            {"support_id", "field_name", "total_row_count", "selected_unique_raw_cell_id_count", "selected_values_sha256", "source_files", "source_sha256", "source_payload_sha256", "complete_source_identity_verified"},
            label,
        )
        if audit["source_files"] != list(source_files or ()) or audit["source_sha256"] != list(source_hashes or ()) or audit["complete_source_identity_verified"] is not True:
            raise DrivAerDatasetScorerError(f"{label} native source identity mismatch")
        if audit["source_payload_sha256"] is not None:
            _sha256(audit["source_payload_sha256"], f"{label}.source_payload_sha256")
    if audit["support_id"] != support_id or audit["field_name"] != field_name or audit["total_row_count"] != total_count:
        raise DrivAerDatasetScorerError(f"{label} support identity mismatch")
    selected = _integer(audit["selected_unique_raw_cell_id_count"], f"{label}.selected count", minimum=0)
    if selected > total_count:
        raise DrivAerDatasetScorerError(f"{label} selected count exceeds native support")
    _sha256(audit["selected_values_sha256"], f"{label}.selected_values_sha256")


def _validate_diagnostic_profile_series(
    value: object,
    *,
    case_id: str,
    contract: _Contract,
    velocity_available: bool,
    cp_cut_available: bool,
) -> tuple[Mapping[str, object], ...]:
    """Validate optional participant-display series retained by the evaluator.

    Case diagnostic evidence predates profile-series output, so an absent or
    empty array remains valid candidate evidence.  Once a family is present it
    must be complete, available, and in the canonical submission-panel order.
    The packaging adapter applies the stronger rule that both families must be
    present for every case.
    """

    if value is None:
        return ()
    if not isinstance(value, list):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id}.profile_series must be an array"
        )
    if not value:
        return ()

    pressure_keys = tuple(
        ("pressure_profiles", station_id, "cp")
        for station_id in contract.profile_cp_cut_ids
    )
    velocity_keys = tuple(
        ("velocity_profiles", station_id, "velocity_ratio")
        for station_id in contract.profile_velocity_station_ids
    )
    expected_by_key: dict[tuple[str, str, str], int | None] = {
        key: None for key in pressure_keys
    }
    expected_by_key.update(
        zip(
            velocity_keys,
            contract.profile_velocity_line_sample_counts,
            strict=True,
        )
    )
    normalized: list[Mapping[str, object]] = []
    observed_keys: list[tuple[str, str, str]] = []
    for position, raw_series in enumerate(value):
        series = _exact_keys(
            raw_series,
            {"panel_id", "station_id", "quantity_id", "coordinate", "prediction"},
            f"{case_id} profile series {position}",
        )
        key = (
            _string(series["panel_id"], f"{case_id} profile panel ID"),
            _string(series["station_id"], f"{case_id} profile station ID"),
            _string(series["quantity_id"], f"{case_id} profile quantity ID"),
        )
        if key not in expected_by_key:
            raise DrivAerDatasetScorerError(
                f"{case_id} profile series names an undeclared panel/station/quantity {key}"
            )
        if key in observed_keys:
            raise DrivAerDatasetScorerError(
                f"{case_id} profile series contains duplicate {key}"
            )
        coordinates = series["coordinate"]
        predictions = series["prediction"]
        if (
            not isinstance(coordinates, list)
            or not isinstance(predictions, list)
            or len(coordinates) < 2
            or len(coordinates) != len(predictions)
        ):
            raise DrivAerDatasetScorerError(
                f"{case_id}/{key[1]} coordinate and prediction arrays must have the same length >= 2"
            )
        normalized_coordinates = [
            _finite(item, f"{case_id}/{key[1]} coordinate {index}")
            for index, item in enumerate(coordinates)
        ]
        normalized_predictions = [
            _finite(item, f"{case_id}/{key[1]} prediction {index}")
            for index, item in enumerate(predictions)
        ]
        if any(
            right <= left
            for left, right in zip(
                normalized_coordinates, normalized_coordinates[1:]
            )
        ):
            raise DrivAerDatasetScorerError(
                f"{case_id}/{key[1]} profile coordinates must be strictly increasing"
            )
        expected_count = expected_by_key[key]
        if expected_count is not None and len(normalized_coordinates) != expected_count:
            raise DrivAerDatasetScorerError(
                f"{case_id}/{key[1]} profile series must contain exactly {expected_count} samples"
            )
        observed_keys.append(key)
        normalized.append(
            {
                "panel_id": key[0],
                "station_id": key[1],
                "quantity_id": key[2],
                "coordinate": normalized_coordinates,
                "prediction": normalized_predictions,
            }
        )

    observed_pressure = tuple(
        key for key in observed_keys if key[0] == "pressure_profiles"
    )
    observed_velocity = tuple(
        key for key in observed_keys if key[0] == "velocity_profiles"
    )
    if observed_pressure:
        if not cp_cut_available:
            raise DrivAerDatasetScorerError(
                f"{case_id} cannot publish Cp-cut profile series while cp_cut_rmse is unavailable"
            )
        if observed_pressure != pressure_keys:
            raise DrivAerDatasetScorerError(
                f"{case_id} Cp-cut profile series are incomplete or out of registry order"
            )
    if observed_velocity:
        if not velocity_available:
            raise DrivAerDatasetScorerError(
                f"{case_id} cannot publish velocity profile series while the ranked velocity metric is unavailable"
            )
        if observed_velocity != velocity_keys:
            raise DrivAerDatasetScorerError(
                f"{case_id} velocity profile series are incomplete or out of registry order"
            )
    canonical_observed = (
        (pressure_keys if observed_pressure else ())
        + (velocity_keys if observed_velocity else ())
    )
    if tuple(observed_keys) != canonical_observed:
        raise DrivAerDatasetScorerError(
            f"{case_id} profile series differ from canonical pressure-then-velocity order"
        )
    return tuple(normalized)


def _validate_surface_only_diagnostic_case(
    root: Mapping[str, Any],
    path: Path,
    digest: str,
    *,
    contract: _Contract,
    pinned: NativeCaseRecord,
    core: _CoreCase,
) -> _DiagnosticCase:
    """Validate the deliberate absence of every volume-derived diagnostic."""

    case_id = pinned.case_id
    if core.prediction_scope != PREDICTION_SCOPE_SURFACE_ONLY:
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} prediction scope differs from core evidence"
        )
    mappings = _exact_keys(root["mapping_inputs"], {"velocity_10mm"}, f"{case_id} mapping_inputs")
    if mappings["velocity_10mm"] != {
        "status": "not_loaded_surface_only",
        "scientific_metric_values_fabricated": False,
    }:
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} surface-only mapping state differs"
        )
    sparse = _exact_keys(
        root["sparse_gather_evidence"],
        {
            "volume_prediction", "volume_native_truth",
            "only_unique_mapped_raw_ids_retained", "prediction_manifest_fully_consumed",
        },
        f"{case_id} sparse gather evidence",
    )
    omitted = {
        "status": "not_submitted_surface_only",
        "scientific_metric_values_fabricated": False,
    }
    not_loaded = {
        "status": "not_loaded_surface_only",
        "scientific_metric_values_fabricated": False,
    }
    if (
        sparse["volume_prediction"] != omitted
        or sparse["volume_native_truth"] != not_loaded
        or sparse["only_unique_mapped_raw_ids_retained"] is not False
        or sparse["prediction_manifest_fully_consumed"] is not False
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} surface-only sparse state differs"
        )
    metrics = _exact_keys(
        root["metrics"],
        {
            "cp_cut_rmse", "velocity_profile_uinf_rmse",
            "velocity_profile_experimental_subset_uinf_rmse",
        },
        f"{case_id} diagnostics metrics",
    )
    cp = _mapping(metrics["cp_cut_rmse"], f"{case_id} Cp-cut metric")
    if (
        cp.get("metric_id") != "cp_cut_rmse"
        or cp.get("ranked_value_available") is not False
        or cp.get("case_equal_cut_mean_rmse") is not None
        or cp.get("cut_rmse") != []
        or cp.get("support_status") != "pending_immutable_owner_release"
    ):
        raise DrivAerDatasetScorerError(f"diagnostic evidence {case_id} Cp-cut state differs")
    expected_reason = {
        "diagnostic": "cp_cut_rmse",
        "stage": "benchmark_support",
        "reason": "immutable_native_cp_cut_extraction_support_not_published",
    }
    if cp.get("unavailable_reasons") != [expected_reason]:
        raise DrivAerDatasetScorerError(f"diagnostic evidence {case_id} Cp-cut reason differs")
    values: dict[str, float | None] = {"cp_cut_rmse": None}
    reasons: dict[str, tuple[Mapping[str, object], ...]] = {
        "cp_cut_rmse": (expected_reason,)
    }
    for metric_id, availability_key, value_key in (
        ("velocity_profile_uinf_rmse", "ranked_value_available", "case_equal_line_mean_rmse"),
        (
            "velocity_profile_experimental_subset_uinf_rmse",
            "value_available",
            "case_equal_experimental_line_mean_rmse",
        ),
    ):
        metric = _mapping(metrics[metric_id], f"{case_id} {metric_id}")
        expected = {
            "diagnostic": metric_id,
            "stage": "prediction_scope",
            "reason": "not_submitted_surface_only",
            "profile_id": None,
            "sample_index": None,
        }
        if (
            metric.get("metric_id") != metric_id
            or metric.get(availability_key) is not False
            or metric.get(value_key) is not None
            or metric.get("line_rmse") != []
            or metric.get("unavailable_reasons") != [expected]
        ):
            raise DrivAerDatasetScorerError(
                f"diagnostic evidence {case_id} {metric_id} surface-only state differs"
            )
        values[metric_id] = None
        reasons[metric_id] = (expected,)
    if root.get("profile_series") != []:
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} surface-only profile series must be empty"
        )
    return _DiagnosticCase(
        case_id=case_id,
        input_file=path.name,
        input_sha256=digest,
        values=values,
        unavailable_reasons=reasons,
        velocity_mapping_sha256=None,
        velocity_receipt_sha256=None,
        profile_series=(),
    )


def _validate_retained_inference_diagnostic_case(
    document: Mapping[str, Any],
    path: Path,
    digest: str,
    *,
    contract: _Contract,
    pinned: NativeCaseRecord,
    core: _CoreCase,
) -> _DiagnosticCase:
    """Validate compact profile evidence retained from a complete inference.

    A retained inference result can preserve exact field-manifest identities,
    profile coordinates, predictions, and sufficient statistics without also
    preserving the transient sparse raw-cell gather buffers used by a live
    evaluator invocation.  This narrow candidate-only representation keeps
    that distinction explicit rather than inventing a sparse-gather audit.

    Complete materialized profile arrays are independently bound by the
    FluidsBench profile-chunk contract.  This document records the per-case
    reductions that feed the dataset evaluator and binds them to the same
    field prediction manifests and current FluidsBench support identities.
    """

    case_id = pinned.case_id
    root = _exact_keys(
        document,
        {
            "schema",
            "schema_version",
            "status",
            "case_id",
            "official_submission",
            "prediction_scope",
            "prediction_inputs",
            "profile_support",
            "metrics",
            "claims",
        },
        f"retained inference diagnostic evidence {case_id}",
    )
    if (
        root["schema"] != RETAINED_INFERENCE_DIAGNOSTIC_CASE_SCHEMA
        or root["schema_version"] != 1
        or root["status"] != RETAINED_INFERENCE_DIAGNOSTIC_CASE_STATUS
        or root["case_id"] != case_id
        or root["official_submission"] is not False
        or root["prediction_scope"] != PREDICTION_SCOPE_FULL
        or core.prediction_scope != PREDICTION_SCOPE_FULL
    ):
        raise DrivAerDatasetScorerError(
            f"retained inference diagnostic evidence {case_id} schema/status mismatch"
        )

    prediction_inputs = _exact_keys(
        root["prediction_inputs"],
        {"surface_native_cells", "volume_native_cells"},
        f"retained inference diagnostic evidence {case_id}.prediction_inputs",
    )
    for support_id, expected_manifest, expected_chunks in (
        (
            "surface_native_cells",
            core.surface_prediction_manifest_sha256,
            core.surface_prediction_chunk_sha256,
        ),
        (
            "volume_native_cells",
            core.volume_prediction_manifest_sha256,
            core.volume_prediction_chunk_sha256,
        ),
    ):
        record = _exact_keys(
            prediction_inputs[support_id],
            {"manifest_sha256", "chunk_sha256"},
            f"retained inference diagnostic evidence {case_id}.{support_id}",
        )
        manifest = _sha256(
            record["manifest_sha256"],
            f"{case_id}/{support_id} retained prediction manifest SHA-256",
        )
        chunks = record["chunk_sha256"]
        if not isinstance(chunks, list):
            raise DrivAerDatasetScorerError(
                f"{case_id}/{support_id} retained prediction chunks must be an array"
            )
        chunk_hashes = tuple(
            _sha256(
                value,
                f"{case_id}/{support_id} retained prediction chunk SHA-256",
            )
            for value in chunks
        )
        if (
            expected_manifest is None
            or expected_chunks is None
            or manifest != expected_manifest
            or chunk_hashes != expected_chunks
        ):
            raise DrivAerDatasetScorerError(
                f"{case_id}/{support_id} retained prediction identity differs from core evidence"
            )

    profile_support = _exact_keys(
        root["profile_support"],
        {
            "native_source_pin_sha256",
            "profile_definition_sha256",
            "constant_series_support_index_sha256",
            "relative_series_support_index_sha256",
            "profile_series_per_case",
            "coordinates_rebound_without_resampling",
        },
        f"retained inference diagnostic evidence {case_id}.profile_support",
    )
    if (
        profile_support["native_source_pin_sha256"]
        != contract.native_source_pin_sha256
        or profile_support["profile_definition_sha256"] != contract.profile_sha256
        or profile_support["constant_series_support_index_sha256"]
        != CONSTANT_SERIES_SUPPORT_INDEX_SHA256
        or profile_support["relative_series_support_index_sha256"]
        != RELATIVE_SERIES_SUPPORT_INDEX_SHA256
        or profile_support["profile_series_per_case"]
        != RELATIVE_PROFILE_SERIES_PER_CASE
        or profile_support["coordinates_rebound_without_resampling"] is not True
    ):
        raise DrivAerDatasetScorerError(
            f"retained inference diagnostic evidence {case_id} profile support mismatch"
        )

    claims = _exact_keys(
        root["claims"],
        {
            "scoring_contract_active",
            "official_submission",
            "owner_scientific_approval",
            "profile_resolution_convergence",
            "three_real_model_ordering",
            "independent_participant_dry_run",
            "all_case_chunk_partition_invariance",
        },
        f"retained inference diagnostic evidence {case_id}.claims",
    )
    if any(value is not False for value in claims.values()):
        raise DrivAerDatasetScorerError(
            f"retained inference diagnostic evidence {case_id} makes an ineligible claim"
        )

    metrics = _exact_keys(
        root["metrics"],
        {
            "cp_cut_rmse",
            "velocity_profile_uinf_rmse",
            "velocity_profile_experimental_subset_uinf_rmse",
        },
        f"retained inference diagnostic evidence {case_id}.metrics",
    )
    cp_cut = _exact_keys(
        metrics["cp_cut_rmse"],
        {
            "metric_id",
            "ranked_value_available",
            "required_cut_count",
            "unavailable_reasons",
            "case_equal_cut_mean_rmse",
            "aggregation",
            "weighting",
            "support_status",
            "discrete_cp_probe_fallback_used",
            "cut_rmse",
        },
        f"retained inference diagnostic evidence {case_id}.cp_cut_rmse",
    )
    velocity = _exact_keys(
        metrics["velocity_profile_uinf_rmse"],
        {
            "metric_id",
            "ranked_value_available",
            "required_line_count",
            "required_sample_count",
            "unavailable_reasons",
            "line_rmse",
            "case_equal_line_mean_rmse",
            "quantity",
            "Uinf_m_per_s",
            "arc_rule",
            "aggregation",
            "weighting",
        },
        f"retained inference diagnostic evidence {case_id}.velocity_profile_uinf_rmse",
    )
    experimental_velocity = _exact_keys(
        metrics["velocity_profile_experimental_subset_uinf_rmse"],
        {
            "metric_id",
            "value_available",
            "required_line_count",
            "required_profile_ids",
            "unavailable_reasons",
            "line_rmse",
            "case_equal_experimental_line_mean_rmse",
            "quantity",
            "Uinf_m_per_s",
            "arc_rule",
            "aggregation",
            "weighting",
        },
        f"retained inference diagnostic evidence {case_id}.velocity_profile_experimental_subset_uinf_rmse",
    )

    if (
        cp_cut["metric_id"] != "cp_cut_rmse"
        or cp_cut["ranked_value_available"] is not True
        or cp_cut["required_cut_count"] != len(contract.profile_cp_cut_ids)
        or cp_cut["unavailable_reasons"] != []
        or cp_cut["aggregation"] != "equal_case_equal_cut_macro_average"
        or cp_cut["weighting"] != "native_cut_intersection_segment_length"
        or cp_cut["support_status"] != "retained_native_support_rebind"
        or cp_cut["discrete_cp_probe_fallback_used"] is not False
        or not isinstance(cp_cut["cut_rmse"], list)
        or len(cp_cut["cut_rmse"]) != len(contract.profile_cp_cut_ids)
    ):
        raise DrivAerDatasetScorerError(
            f"retained inference diagnostic evidence {case_id} Cp-cut contract mismatch"
        )
    cut_values: list[float] = []
    cut_ids: list[str] = []
    for position, row in enumerate(cp_cut["cut_rmse"]):
        item = _exact_keys(
            row,
            {"cut_id", "segment_count", "arc_length_m", "rmse"},
            f"retained inference diagnostic evidence {case_id} Cp-cut row {position}",
        )
        cut_ids.append(_string(item["cut_id"], f"{case_id} Cp-cut ID"))
        _integer(
            item["segment_count"],
            f"{case_id}/{item['cut_id']} retained Cp-cut segment count",
            minimum=1,
        )
        if _finite(
            item["arc_length_m"],
            f"{case_id}/{item['cut_id']} retained Cp-cut arc length",
            nonnegative=True,
        ) <= 0.0:
            raise DrivAerDatasetScorerError(
                f"{case_id}/{item['cut_id']} retained Cp-cut arc length must be positive"
            )
        cut_values.append(
            _finite(
                item["rmse"],
                f"{case_id}/{item['cut_id']} retained Cp-cut RMSE",
                nonnegative=True,
            )
        )
    if tuple(cut_ids) != contract.profile_cp_cut_ids:
        raise DrivAerDatasetScorerError(
            f"retained inference diagnostic evidence {case_id} Cp-cut order differs from the registry"
        )
    cp_value = _finite(
        cp_cut["case_equal_cut_mean_rmse"],
        f"{case_id}/retained_cp_cut_rmse",
        nonnegative=True,
    )
    _require_same_float(
        cp_value,
        math.fsum(cut_values) / len(cut_values),
        f"{case_id}/retained_cp_cut_rmse case metric",
    )

    u_inf = _finite(
        contract.force_constants.get("freestream_velocity_m_per_s"),
        "force freestream velocity",
        nonnegative=True,
    )
    if u_inf <= 0.0:
        raise DrivAerDatasetScorerError("force freestream velocity must be positive")

    def validate_velocity_metric(
        metric_id: str,
        metric: Mapping[str, Any],
        *,
        availability_key: str,
        value_key: str,
        expected_ids: tuple[str, ...],
        expected_operation: str,
    ) -> float:
        if (
            metric["metric_id"] != metric_id
            or metric[availability_key] is not True
            or metric["unavailable_reasons"] != []
            or metric["required_line_count"] != len(expected_ids)
            or metric["quantity"] != "magnitude(UMeanTrim)/Uinf"
            or metric["Uinf_m_per_s"] != u_inf
            or metric["arc_rule"]
            != "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging"
            or metric["aggregation"] != expected_operation
            or metric["weighting"] != "trapezoidal_arc_length_within_line"
            or not isinstance(metric["line_rmse"], list)
            or len(metric["line_rmse"]) != len(expected_ids)
        ):
            raise DrivAerDatasetScorerError(
                f"retained inference diagnostic evidence {case_id} {metric_id} contract mismatch"
            )
        rows: list[float] = []
        observed_ids: list[str] = []
        sample_count_by_id = dict(
            zip(
                contract.profile_velocity_line_ids,
                contract.profile_velocity_line_sample_counts,
                strict=True,
            )
        )
        for position, row in enumerate(metric["line_rmse"]):
            item = _exact_keys(
                row,
                {"profile_id", "sample_count", "arc_length_m", "rmse"},
                f"retained inference diagnostic evidence {case_id} {metric_id} row {position}",
            )
            profile_id = _string(item["profile_id"], f"{case_id} retained profile ID")
            observed_ids.append(profile_id)
            if item["sample_count"] != sample_count_by_id.get(profile_id):
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{metric_id}/{profile_id} retained sample count mismatch"
                )
            if _finite(
                item["arc_length_m"],
                f"{case_id}/{metric_id}/{profile_id} retained arc length",
                nonnegative=True,
            ) <= 0.0:
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{metric_id}/{profile_id} retained arc length must be positive"
                )
            rows.append(
                _finite(
                    item["rmse"],
                    f"{case_id}/{metric_id}/{profile_id} retained RMSE",
                    nonnegative=True,
                )
            )
        if tuple(observed_ids) != expected_ids:
            raise DrivAerDatasetScorerError(
                f"retained inference diagnostic evidence {case_id} {metric_id} line order differs from the registry"
            )
        value = _finite(metric[value_key], f"{case_id}/{metric_id}", nonnegative=True)
        _require_same_float(
            value,
            math.fsum(rows) / len(rows),
            f"{case_id}/{metric_id} retained case metric",
        )
        return value

    velocity_value = validate_velocity_metric(
        "velocity_profile_uinf_rmse",
        velocity,
        availability_key="ranked_value_available",
        value_key="case_equal_line_mean_rmse",
        expected_ids=contract.profile_velocity_line_ids,
        expected_operation="equal_case_equal_line_macro_average",
    )
    if experimental_velocity["required_profile_ids"] != list(
        contract.profile_experimental_velocity_line_ids
    ):
        raise DrivAerDatasetScorerError(
            f"retained inference diagnostic evidence {case_id} experimental profile IDs differ from the registry"
        )
    experimental_value = validate_velocity_metric(
        "velocity_profile_experimental_subset_uinf_rmse",
        experimental_velocity,
        availability_key="value_available",
        value_key="case_equal_experimental_line_mean_rmse",
        expected_ids=contract.profile_experimental_velocity_line_ids,
        expected_operation="equal_case_equal_experimental_line_macro_average",
    )
    return _DiagnosticCase(
        case_id=case_id,
        input_file=path.name,
        input_sha256=digest,
        values={
            "cp_cut_rmse": cp_value,
            "velocity_profile_uinf_rmse": velocity_value,
            "velocity_profile_experimental_subset_uinf_rmse": experimental_value,
        },
        unavailable_reasons={
            "cp_cut_rmse": (),
            "velocity_profile_uinf_rmse": (),
            "velocity_profile_experimental_subset_uinf_rmse": (),
        },
        velocity_mapping_sha256=None,
        velocity_receipt_sha256=None,
        profile_series=(),
    )


def _validate_diagnostic_case(
    document: Mapping[str, Any],
    path: Path,
    digest: str,
    *,
    contract: _Contract,
    pinned: NativeCaseRecord,
    core: _CoreCase,
) -> _DiagnosticCase:
    """Validate the probe-free v9 submission diagnostic evidence."""

    case_id = pinned.case_id
    required_root_keys = {
        "schema",
        "schema_version",
        "status",
        "case_id",
        "official_submission",
        "mapping_inputs",
        "sparse_gather_evidence",
        "metrics",
        "claims",
    }
    root = _mapping(document, f"diagnostic evidence {case_id}")
    if root.get("schema") == RETAINED_INFERENCE_DIAGNOSTIC_CASE_SCHEMA:
        return _validate_retained_inference_diagnostic_case(
            root,
            path,
            digest,
            contract=contract,
            pinned=pinned,
            core=core,
        )
    observed_root_keys = set(root)
    allowed_root_keys = required_root_keys | {"profile_series", "prediction_scope"}
    if not required_root_keys.issubset(observed_root_keys) or not observed_root_keys.issubset(
        allowed_root_keys
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} keys differ from the frozen candidate "
            f"schema (missing={sorted(required_root_keys - observed_root_keys)}, "
            f"unexpected={sorted(observed_root_keys - allowed_root_keys)})"
        )
    legacy = (
        root["schema"] == LEGACY_DIAGNOSTIC_CASE_SCHEMA
        and root["schema_version"] == 3
        and "prediction_scope" not in root
    )
    current = (
        root["schema"] == DIAGNOSTIC_CASE_SCHEMA
        and root["schema_version"] == 4
    )
    if (
        not (legacy or current)
        or root["status"] != DIAGNOSTIC_CASE_STATUS
        or root["case_id"] != case_id
        or root.get("prediction_scope", PREDICTION_SCOPE_FULL) not in PREDICTION_SCOPES
        or root["official_submission"] is not False
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} schema/status mismatch"
        )
    if root.get("prediction_scope", PREDICTION_SCOPE_FULL) == PREDICTION_SCOPE_SURFACE_ONLY:
        return _validate_surface_only_diagnostic_case(
            root, path, digest, contract=contract, pinned=pinned, core=core
        )
    if core.prediction_scope != PREDICTION_SCOPE_FULL:
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} prediction scope differs from core evidence"
        )
    claims = _exact_keys(
        root["claims"],
        {
            "scoring_contract_active",
            "official_submission",
            "owner_scientific_approval",
            "profile_resolution_convergence",
            "three_real_model_ordering",
            "independent_participant_dry_run",
            "all_case_chunk_partition_invariance",
        },
        f"diagnostic evidence {case_id}.claims",
    )
    if any(value is not False for value in claims.values()):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} makes an ineligible claim"
        )

    mappings = _exact_keys(
        root["mapping_inputs"], {"velocity_10mm"}, f"{case_id} mapping_inputs"
    )
    velocity_mapping = _exact_keys(
        mappings["velocity_10mm"],
        {
            "artifact_file",
            "artifact_sha256",
            "receipt_file",
            "receipt_sha256",
            "profile_sha256",
            "row_count",
            "invalid_rows",
        },
        f"{case_id} velocity_10mm",
    )
    _basename(velocity_mapping["artifact_file"], f"{case_id} velocity artifact file")
    _basename(velocity_mapping["receipt_file"], f"{case_id} velocity receipt file")
    if (
        velocity_mapping["profile_sha256"] != contract.profile_sha256
        or velocity_mapping["row_count"] != contract.profile_velocity_sample_count
        or not isinstance(velocity_mapping["invalid_rows"], list)
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} velocity mapping support mismatch"
        )
    velocity_artifact_sha = _sha256(
        velocity_mapping["artifact_sha256"], f"{case_id} velocity mapping SHA-256"
    )
    velocity_receipt_sha = _sha256(
        velocity_mapping["receipt_sha256"], f"{case_id} velocity receipt SHA-256"
    )

    sparse = _exact_keys(
        root["sparse_gather_evidence"],
        {
            "volume_prediction",
            "volume_native_truth",
            "only_unique_mapped_raw_ids_retained",
            "prediction_manifest_fully_consumed",
        },
        f"diagnostic evidence {case_id}.sparse_gather_evidence",
    )
    if (
        sparse["only_unique_mapped_raw_ids_retained"] is not True
        or sparse["prediction_manifest_fully_consumed"] is not True
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} sparse coverage mismatch"
        )
    _validate_sparse_audit(
        sparse["volume_prediction"],
        label=f"{case_id} volume prediction gather",
        support_id="volume_native_cells",
        field_name="UMeanTrim",
        total_count=core.volume_count,
        manifest_sha256=core.volume_prediction_manifest_sha256,
        chunk_sha256=core.volume_prediction_chunk_sha256,
    )
    _validate_sparse_audit(
        sparse["volume_native_truth"],
        label=f"{case_id} volume native truth",
        support_id="volume_native_cells",
        field_name="UMeanTrim",
        total_count=core.volume_count,
        source_files=tuple(part.path.name for part in pinned.volume_parts),
        source_hashes=core.volume_part_sha256,
    )

    metrics = _exact_keys(
        root["metrics"],
        {
            "cp_cut_rmse",
            "velocity_profile_uinf_rmse",
            "velocity_profile_experimental_subset_uinf_rmse",
        },
        f"{case_id} diagnostics metrics",
    )
    cp_cut = _exact_keys(
        metrics["cp_cut_rmse"],
        {
            "metric_id",
            "ranked_value_available",
            "required_cut_count",
            "unavailable_reasons",
            "case_equal_cut_mean_rmse",
            "aggregation",
            "weighting",
            "support_status",
            "discrete_cp_probe_fallback_used",
            "cut_rmse",
        },
        f"{case_id} Cp-cut metric",
    )
    velocity = _exact_keys(
        metrics["velocity_profile_uinf_rmse"],
        {
            "metric_id",
            "ranked_value_available",
            "required_line_count",
            "required_sample_count",
            "unavailable_reasons",
            "line_rmse",
            "case_equal_line_mean_rmse",
            "quantity",
            "Uinf_m_per_s",
            "arc_rule",
            "aggregation",
            "weighting",
        },
        f"{case_id} velocity metric",
    )
    experimental_velocity = _exact_keys(
        metrics["velocity_profile_experimental_subset_uinf_rmse"],
        {
            "metric_id",
            "value_available",
            "required_line_count",
            "required_profile_ids",
            "unavailable_reasons",
            "line_rmse",
            "case_equal_experimental_line_mean_rmse",
            "quantity",
            "Uinf_m_per_s",
            "arc_rule",
            "aggregation",
            "weighting",
        },
        f"{case_id} experimental velocity metric",
    )
    if (
        cp_cut["metric_id"] != "cp_cut_rmse"
        or cp_cut["required_cut_count"] != len(contract.profile_cp_cut_ids)
        or cp_cut["aggregation"] != "equal_case_equal_cut_macro_average"
        or cp_cut["weighting"] != "native_cut_intersection_segment_length"
        or cp_cut["discrete_cp_probe_fallback_used"] is not False
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} Cp-cut contract mismatch"
        )
    cp_cut_available = cp_cut["ranked_value_available"]
    cp_cut_reasons = cp_cut["unavailable_reasons"]
    cp_cut_rows = cp_cut["cut_rmse"]
    if (
        not isinstance(cp_cut_available, bool)
        or not isinstance(cp_cut_reasons, list)
        or not isinstance(cp_cut_rows, list)
    ):
        raise DrivAerDatasetScorerError(
            f"{case_id} Cp-cut availability evidence is malformed"
        )
    cp_cut_reason: Mapping[str, Any] | None = None
    cp_cut_value: float | None = None
    if not cp_cut_available:
        if len(cp_cut_reasons) != 1:
            raise DrivAerDatasetScorerError(
                f"{case_id} Cp-cut support must fail closed with one explicit reason"
            )
        cp_cut_reason = _exact_keys(
            cp_cut_reasons[0],
            {"diagnostic", "stage", "reason"},
            f"{case_id} Cp-cut unavailable reason",
        )
        if (
            cp_cut["case_equal_cut_mean_rmse"] is not None
            or cp_cut["support_status"] != "pending_immutable_owner_release"
            or cp_cut_rows != []
            or cp_cut_reason
            != {
                "diagnostic": "cp_cut_rmse",
                "stage": "benchmark_support",
                "reason": "immutable_native_cp_cut_extraction_support_not_published",
            }
        ):
            raise DrivAerDatasetScorerError(
                f"diagnostic evidence {case_id} Cp-cut fail-closed contract mismatch"
            )
    else:
        if (
            cp_cut_reasons != []
            or cp_cut["support_status"] != "immutable_owner_release"
            or len(cp_cut_rows) != len(contract.profile_cp_cut_ids)
        ):
            raise DrivAerDatasetScorerError(
                f"diagnostic evidence {case_id} available Cp-cut support is incomplete"
            )
        cut_values: list[float] = []
        observed_cut_ids: list[str] = []
        for position, row in enumerate(cp_cut_rows):
            item = _exact_keys(
                row,
                {"cut_id", "segment_count", "arc_length_m", "rmse"},
                f"{case_id} Cp-cut row {position}",
            )
            observed_cut_ids.append(
                _string(item["cut_id"], f"{case_id} Cp-cut ID")
            )
            _integer(
                item["segment_count"],
                f"{case_id}/{item['cut_id']} segment count",
                minimum=1,
            )
            if _finite(
                item["arc_length_m"],
                f"{case_id}/{item['cut_id']} arc length",
                nonnegative=True,
            ) <= 0.0:
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{item['cut_id']} arc length must be positive"
                )
            cut_values.append(
                _finite(
                    item["rmse"],
                    f"{case_id}/{item['cut_id']} Cp-cut RMSE",
                    nonnegative=True,
                )
            )
        if tuple(observed_cut_ids) != contract.profile_cp_cut_ids:
            raise DrivAerDatasetScorerError(
                f"{case_id} Cp-cut order differs from the registry"
            )
        cp_cut_value = _finite(
            cp_cut["case_equal_cut_mean_rmse"],
            f"{case_id}/cp_cut_rmse",
            nonnegative=True,
        )
        _require_same_float(
            cp_cut_value,
            math.fsum(cut_values) / len(cut_values),
            f"{case_id}/cp_cut_rmse case metric",
        )

    u_inf = _finite(
        contract.force_constants.get("freestream_velocity_m_per_s"),
        "force freestream velocity",
        nonnegative=True,
    )
    if (
        velocity["metric_id"] != "velocity_profile_uinf_rmse"
        or velocity["required_line_count"] != len(contract.profile_velocity_line_ids)
        or velocity["required_sample_count"] != contract.profile_velocity_sample_count
        or velocity["quantity"] != "magnitude(UMeanTrim)/Uinf"
        or velocity["arc_rule"]
        != "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging"
        or velocity["aggregation"] != "equal_case_equal_line_macro_average"
        or velocity["weighting"] != "trapezoidal_arc_length_within_line"
        or not _same_float(
            _finite(velocity["Uinf_m_per_s"], f"{case_id} velocity Uinf"), u_inf
        )
        or experimental_velocity["metric_id"]
        != "velocity_profile_experimental_subset_uinf_rmse"
        or experimental_velocity["required_line_count"]
        != len(contract.profile_experimental_velocity_line_ids)
        or experimental_velocity["required_profile_ids"]
        != list(contract.profile_experimental_velocity_line_ids)
        or experimental_velocity["quantity"] != "magnitude(UMeanTrim)/Uinf"
        or experimental_velocity["arc_rule"]
        != "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging"
        or experimental_velocity["aggregation"]
        != "equal_case_equal_experimental_line_macro_average"
        or experimental_velocity["weighting"]
        != "trapezoidal_arc_length_within_line"
        or not _same_float(
            _finite(
                experimental_velocity["Uinf_m_per_s"],
                f"{case_id} experimental velocity Uinf",
            ),
            u_inf,
        )
    ):
        raise DrivAerDatasetScorerError(
            f"diagnostic evidence {case_id} velocity contract mismatch"
        )

    values: dict[str, float | None] = {"cp_cut_rmse": cp_cut_value}
    reasons: dict[str, tuple[Mapping[str, object], ...]] = {
        "cp_cut_rmse": (() if cp_cut_reason is None else (dict(cp_cut_reason),))
    }
    for metric_id, metric, availability_key, value_key in (
        (
            "velocity_profile_uinf_rmse",
            velocity,
            "ranked_value_available",
            "case_equal_line_mean_rmse",
        ),
        (
            "velocity_profile_experimental_subset_uinf_rmse",
            experimental_velocity,
            "value_available",
            "case_equal_experimental_line_mean_rmse",
        ),
    ):
        available = metric[availability_key]
        unavailable = metric["unavailable_reasons"]
        if not isinstance(available, bool) or not isinstance(unavailable, list):
            raise DrivAerDatasetScorerError(
                f"{case_id}/{metric_id} availability is malformed"
            )
        if available:
            if unavailable:
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{metric_id} claims availability despite reasons"
                )
            values[metric_id] = _finite(
                metric[value_key], f"{case_id}/{metric_id}", nonnegative=True
            )
            reasons[metric_id] = ()
        else:
            if metric[value_key] is not None or not unavailable:
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{metric_id} unavailable evidence is incomplete"
                )
            values[metric_id] = None
            reasons[metric_id] = tuple(
                dict(_mapping(reason, f"{case_id}/{metric_id} unavailable reason"))
                for reason in unavailable
            )

    def unavailable_reason_keys(
        metric_id: str,
    ) -> tuple[tuple[str, int, str, str], ...]:
        result: list[tuple[str, int, str, str]] = []
        for position, reason in enumerate(reasons[metric_id]):
            item = _exact_keys(
                reason,
                {
                    "diagnostic",
                    "stage",
                    "reason",
                    "profile_id",
                    "sample_index",
                },
                f"{case_id}/{metric_id} unavailable reason {position}",
            )
            if (
                item["diagnostic"] != metric_id
                or item["profile_id"] is None
                or item["sample_index"] is None
            ):
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{metric_id} unavailable reason identity mismatch"
                )
            result.append(
                (
                    _string(item["profile_id"], f"{case_id} unavailable profile ID"),
                    _integer(
                        item["sample_index"],
                        f"{case_id} unavailable sample index",
                        minimum=0,
                    ),
                    _string(item["stage"], f"{case_id} unavailable stage"),
                    _string(item["reason"], f"{case_id} unavailable reason"),
                )
            )
        return tuple(result)

    velocity_invalid_keys: list[tuple[str, int, str, str]] = []
    for position, row in enumerate(velocity_mapping["invalid_rows"]):
        item = _exact_keys(
            row,
            {
                "profile_id",
                "sample_index",
                "valid",
                "reason",
                "raw_vtk_cell_id",
                "candidate_count",
            },
            f"{case_id} invalid velocity row {position}",
        )
        profile_id = _string(item["profile_id"], f"{case_id} invalid profile ID")
        if profile_id not in set(contract.profile_velocity_line_ids):
            raise DrivAerDatasetScorerError(
                f"{case_id} invalid velocity row names an unknown profile"
            )
        sample_index = _integer(
            item["sample_index"], f"{case_id} invalid sample index", minimum=0
        )
        if item["valid"] is not False:
            raise DrivAerDatasetScorerError(
                f"{case_id} invalid velocity row claims valid support"
            )
        velocity_invalid_keys.append(
            (
                profile_id,
                sample_index,
                "mapping",
                _string(item["reason"], f"{case_id} velocity invalid reason"),
            )
        )
    expected_experimental_keys = tuple(
        key
        for key in velocity_invalid_keys
        if key[0] in set(contract.profile_experimental_velocity_line_ids)
    )
    if (
        unavailable_reason_keys("velocity_profile_uinf_rmse")
        != tuple(velocity_invalid_keys)
        or unavailable_reason_keys("velocity_profile_experimental_subset_uinf_rmse")
        != expected_experimental_keys
    ):
        raise DrivAerDatasetScorerError(
            f"{case_id} velocity unavailable reasons differ from explicit invalid rows"
        )

    sample_count_by_id = dict(
        zip(
            contract.profile_velocity_line_ids,
            contract.profile_velocity_line_sample_counts,
            strict=True,
        )
    )

    def validate_line_reduction(
        metric_id: str,
        metric: Mapping[str, Any],
        expected_ids: tuple[str, ...],
    ) -> None:
        line_results = metric["line_rmse"]
        if values[metric_id] is None:
            if line_results != []:
                raise DrivAerDatasetScorerError(
                    f"{case_id} unavailable {metric_id} contains partial scores"
                )
            return
        if not isinstance(line_results, list) or len(line_results) != len(expected_ids):
            raise DrivAerDatasetScorerError(
                f"{case_id}/{metric_id} line evidence is incomplete"
            )
        line_values: list[float] = []
        observed_ids: list[str] = []
        for position, row in enumerate(line_results):
            item = _exact_keys(
                row,
                {"profile_id", "sample_count", "arc_length_m", "rmse"},
                f"{case_id}/{metric_id} line {position}",
            )
            profile_id = _string(item["profile_id"], f"{case_id} profile ID")
            observed_ids.append(profile_id)
            if item["sample_count"] != sample_count_by_id.get(profile_id):
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{metric_id}/{profile_id} sample count mismatch"
                )
            if (
                _finite(
                    item["arc_length_m"],
                    f"{case_id} profile arc length",
                    nonnegative=True,
                )
                <= 0.0
            ):
                raise DrivAerDatasetScorerError(
                    f"{case_id} profile arc length must be positive"
                )
            line_values.append(
                _finite(item["rmse"], f"{case_id} profile RMSE", nonnegative=True)
            )
        if tuple(observed_ids) != expected_ids:
            raise DrivAerDatasetScorerError(
                f"{case_id}/{metric_id} line order differs from the registry"
            )
        _require_same_float(
            values[metric_id],
            math.fsum(line_values) / len(line_values),
            f"{case_id}/{metric_id} case metric",
        )

    validate_line_reduction(
        "velocity_profile_uinf_rmse",
        velocity,
        contract.profile_velocity_line_ids,
    )
    validate_line_reduction(
        "velocity_profile_experimental_subset_uinf_rmse",
        experimental_velocity,
        contract.profile_experimental_velocity_line_ids,
    )
    profile_series = _validate_diagnostic_profile_series(
        root.get("profile_series"),
        case_id=case_id,
        contract=contract,
        velocity_available=values["velocity_profile_uinf_rmse"] is not None,
        cp_cut_available=values["cp_cut_rmse"] is not None,
    )
    return _DiagnosticCase(
        case_id=case_id,
        input_file=path.name,
        input_sha256=digest,
        values=values,
        unavailable_reasons=reasons,
        velocity_mapping_sha256=velocity_artifact_sha,
        velocity_receipt_sha256=velocity_receipt_sha,
        profile_series=profile_series,
    )


def _macro(values: Sequence[float]) -> float:
    if not values:
        raise DrivAerDatasetScorerError("cannot reduce an empty case metric")
    return math.fsum(values) / len(values)


def _rmse(errors: Sequence[float]) -> float:
    if not errors:
        raise DrivAerDatasetScorerError("cannot reduce an empty force metric")
    return math.sqrt(math.fsum(error * error for error in errors) / len(errors))


def evaluate_candidate_dataset(
    *,
    submission_specification: Path | str,
    split_id: str,
    force_truth_csv: Path | str,
    core_case_evidence: Sequence[Path | str],
    diagnostic_case_evidence: Sequence[Path | str],
    immutable_pins: CandidateDatasetImmutablePins = OFFICIAL_IMMUTABLE_PINS,
) -> CandidateDatasetEvaluation:
    """Validate and reduce exactly one official DrivAerML test split."""

    contract, truth = _load_contract(
        submission_specification=submission_specification,
        split_id=split_id,
        force_truth_csv=force_truth_csv,
        immutable_pins=immutable_pins,
    )
    core_documents = _load_evidence_documents(
        core_case_evidence, expected_case_ids=contract.case_ids, label="core case evidence"
    )
    diagnostic_documents = _load_evidence_documents(
        diagnostic_case_evidence,
        expected_case_ids=contract.case_ids,
        label="diagnostic case evidence",
    )
    core_cases: list[_CoreCase] = []
    diagnostic_cases: list[_DiagnosticCase] = []
    for case_id in contract.case_ids:
        pinned = contract.native_source_pin.case(case_id)
        core_document, core_path, core_sha = core_documents[case_id]
        core = _validate_core_case(
            core_document, core_path, core_sha, contract=contract, pinned=pinned
        )
        diagnostic_document, diagnostic_path, diagnostic_sha = diagnostic_documents[case_id]
        diagnostic = _validate_diagnostic_case(
            diagnostic_document,
            diagnostic_path,
            diagnostic_sha,
            contract=contract,
            pinned=pinned,
            core=core,
        )
        core_cases.append(core)
        diagnostic_cases.append(diagnostic)

    prediction_scopes = {case.prediction_scope for case in core_cases}
    if len(prediction_scopes) != 1:
        raise DrivAerDatasetScorerError(
            "all cases in a candidate split must use one prediction scope"
        )
    prediction_scope = prediction_scopes.pop()
    active_field_metric_ids = tuple(
        metric_id
        for prefix in _FIELD_LAYOUT
        if (
            prediction_scope == PREDICTION_SCOPE_FULL
            or _FIELD_LAYOUT[prefix]["support"] == "surface_native_cells"
        )
        for metric_id in _field_metric_ids(prefix)
    )
    active_primary_field_metric_ids = tuple(
        metric_id
        for metric_id in PRIMARY_FIELD_METRIC_IDS
        if (
            prediction_scope == PREDICTION_SCOPE_FULL
            or not metric_id.startswith("volume_")
        )
    )

    field_metric_values = {
        metric_id: _macro([case.field_metrics[metric_id] for case in core_cases])
        for metric_id in active_field_metric_ids
    }
    force_errors: dict[str, list[float]] = {
        key: [] for key in ("Cd", "Cl", "CmPitch", "Clf", "Clr")
    }
    force_case_rows: list[dict[str, object]] = []
    predicted_closures: list[float] = []
    truth_closures: list[float] = []
    for core in core_cases:
        expected = truth[core.case_id]
        expected_force = {
            "Cd": expected["cd"],
            "Cl": expected["cl"],
            "CmPitch": (expected["clf"] - expected["clr"]) / 2.0,
            "Clf": expected["clf"],
            "Clr": expected["clr"],
        }
        errors = {
            key: core.force_coefficients[key] - expected_force[key]
            for key in force_errors
        }
        for key in force_errors:
            force_errors[key].append(errors[key])
        predicted_closure = abs(
            core.force_coefficients["Cl"]
            - (core.force_coefficients["Clf"] + core.force_coefficients["Clr"])
        )
        truth_closure = abs(expected["cl"] - (expected["clf"] + expected["clr"]))
        predicted_closures.append(predicted_closure)
        truth_closures.append(truth_closure)
        force_case_rows.append(
            {
                "prediction": {key: core.force_coefficients[key] for key in force_errors},
                "truth": expected_force,
                "signed_error": errors,
                "predicted_lift_closure_abs": predicted_closure,
                "truth_lift_closure_abs": truth_closure,
            }
        )
    force_metric_values = {
        "field_integrated_cd_rmse": _rmse(force_errors["Cd"]),
        "field_integrated_cl_rmse": _rmse(force_errors["Cl"]),
        "field_integrated_cmpitch_rmse": _rmse(force_errors["CmPitch"]),
        "field_integrated_clf_rmse": _rmse(force_errors["Clf"]),
        "field_integrated_clr_rmse": _rmse(force_errors["Clr"]),
        "field_integrated_lift_closure_max_abs": max(predicted_closures),
    }

    diagnostic_metric_values: dict[str, float | None] = {}
    diagnostic_availability: dict[str, object] = {}
    for metric_id in DIAGNOSTIC_METRIC_IDS:
        unavailable = [
            {
                "case_id": case.case_id,
                "reasons": list(case.unavailable_reasons[metric_id]),
            }
            for case in diagnostic_cases
            if case.values[metric_id] is None
        ]
        if unavailable:
            diagnostic_metric_values[metric_id] = None
            diagnostic_availability[metric_id] = {
                "available": False,
                "required_case_count": len(contract.case_ids),
                "available_case_count": len(contract.case_ids) - len(unavailable),
                "unavailable_cases": unavailable,
                "omitted_case_count": 0,
            }
        else:
            values = [float(case.values[metric_id]) for case in diagnostic_cases]
            diagnostic_metric_values[metric_id] = _macro(values)
            diagnostic_availability[metric_id] = {
                "available": True,
                "required_case_count": len(contract.case_ids),
                "available_case_count": len(contract.case_ids),
                "unavailable_cases": [],
                "omitted_case_count": 0,
            }

    metric_values: dict[str, float | None] = {
        **field_metric_values,
        **force_metric_values,
        **diagnostic_metric_values,
    }
    case_rows: list[dict[str, object]] = []
    for core, diagnostic, force_row in zip(
        core_cases, diagnostic_cases, force_case_rows, strict=True
    ):
        case_rows.append(
            {
                "case_id": core.case_id,
                "inputs": {
                    "core_case_evidence": {"file": core.input_file, "sha256": core.input_sha256},
                    "diagnostic_case_evidence": {"file": diagnostic.input_file, "sha256": diagnostic.input_sha256},
                },
                "source_support": {
                    "boundary_sha256": core.boundary_sha256,
                    "surface_area_sha256": core.surface_area_sha256,
                    **(
                        {
                            "volume_part_sha256": list(core.volume_part_sha256 or ()),
                            "volume_weighting": {
                                "weighting": "one_per_native_cell",
                                "entity_count": core.volume_count,
                                "total_weight": float(core.volume_count or 0),
                                "geometric_cell_volume_weights_used": False,
                            },
                        }
                        if prediction_scope == PREDICTION_SCOPE_FULL
                        else {
                            "volume_native": {
                                "status": "not_loaded_surface_only",
                                "scientific_metric_values_fabricated": False,
                            }
                        }
                    ),
                    "velocity_mapping_sha256": diagnostic.velocity_mapping_sha256,
                    "velocity_receipt_sha256": diagnostic.velocity_receipt_sha256,
                },
                "support_counts": {
                    "surface_native_cells": core.surface_count,
                    **(
                        {"volume_native_cells": core.volume_count}
                        if prediction_scope == PREDICTION_SCOPE_FULL
                        else {}
                    ),
                },
                "prediction_scope": prediction_scope,
                "field_metric_values": dict(core.field_metrics),
                "field_metric_sufficient_statistics": {
                    key: dict(value) for key, value in core.relative_l2_statistics.items()
                },
                "force": force_row,
                "diagnostic_metric_values": dict(diagnostic.values),
                "profile_series": [dict(series) for series in diagnostic.profile_series],
            }
        )

    reductions: dict[str, object] = {}
    for metric_id in active_field_metric_ids:
        reductions[metric_id] = {
            "operation": "complete_case_then_equal_case_macro_average",
            "case_count": len(core_cases),
            "value": field_metric_values[metric_id],
        }
    for metric_id, force_key in (
        ("field_integrated_cd_rmse", "Cd"),
        ("field_integrated_cl_rmse", "Cl"),
        ("field_integrated_cmpitch_rmse", "CmPitch"),
        ("field_integrated_clf_rmse", "Clf"),
        ("field_integrated_clr_rmse", "Clr"),
    ):
        reductions[metric_id] = {
            "operation": "equal_case_rmse",
            "case_count": len(core_cases),
            "squared_error_sum": math.fsum(value * value for value in force_errors[force_key]),
            "value": force_metric_values[metric_id],
            "ranking_role": "ranked" if metric_id in RANKED_FORCE_METRIC_IDS else "report_only",
        }
    reductions["field_integrated_lift_closure_max_abs"] = {
        "operation": "maximum_case_absolute_residual",
        "case_count": len(core_cases),
        "value": force_metric_values["field_integrated_lift_closure_max_abs"],
        "ranking_role": "report_only",
    }
    diagnostic_operations = {
        "velocity_profile_uinf_rmse": "equal_case_equal_line_macro_average",
        "velocity_profile_experimental_subset_uinf_rmse": (
            "equal_case_equal_experimental_line_macro_average"
        ),
        "cp_cut_rmse": "equal_case_equal_cut_macro_average",
    }
    diagnostic_scopes = {
        "velocity_profile_uinf_rmse": {
            "line_count_per_case": len(contract.profile_velocity_line_ids),
            "within_line_weighting": "trapezoidal_arc_length",
        },
        "velocity_profile_experimental_subset_uinf_rmse": {
            "line_count_per_case": len(
                contract.profile_experimental_velocity_line_ids
            ),
            "within_line_weighting": "trapezoidal_arc_length",
        },
        "cp_cut_rmse": {
            "cut_count_per_case": len(contract.profile_cp_cut_ids),
            "within_cut_weighting": "native_cut_intersection_segment_length",
            "between_cut_weighting": "cuts_equal",
            "support_status": "pending_immutable_owner_release",
        },
    }
    for metric_id in DIAGNOSTIC_METRIC_IDS:
        reductions[metric_id] = {
            "operation": diagnostic_operations[metric_id],
            "case_count": len(core_cases),
            "value": diagnostic_metric_values[metric_id],
            "available": diagnostic_metric_values[metric_id] is not None,
            "ranking_role": (
                "ranked"
                if metric_id in RANKED_DIAGNOSTIC_METRIC_IDS
                else "report_only"
            ),
            "support_scope": diagnostic_scopes[metric_id],
        }

    uses_legacy_evidence = all(
        core.prediction_scope == PREDICTION_SCOPE_FULL
        and core.regional_diagnostics is None
        for core in core_cases
    )
    evidence: dict[str, object] = {
        "schema": (
            LEGACY_CANDIDATE_DATASET_SCHEMA
            if uses_legacy_evidence
            else CANDIDATE_DATASET_SCHEMA
        ),
        "schema_version": 3 if uses_legacy_evidence else 4,
        "status": CANDIDATE_DATASET_STATUS,
        "eligibility": {
            "official_submission": False,
            "leaderboard_eligible": False,
            "scoring_contract_active": False,
            "composite_score_available": False,
            "component_scores_available": False,
            "reason": "immutable profile support and owner activation approval are not frozen",
        },
        "prediction_scope": prediction_scope,
        "split": {
            "split_id": contract.split_id,
            "case_set_id": contract.case_set_id,
            "index_file": contract.split_file,
            "index_sha256": contract.split_sha256,
            "case_count": len(contract.case_ids),
            "case_ids": list(contract.case_ids),
            "exact_order_and_membership_verified": True,
        },
        "source_contract": {
            "submission_specification_file": contract.specification_path.name,
            "submission_specification_sha256": contract.specification_sha256,
            "native_source_pin_file": contract.native_source_pin_file,
            "native_source_pin_sha256": contract.native_source_pin_sha256,
            "repository_id": contract.native_source_pin.repository_id,
            "repository_revision": contract.native_source_pin.repository_revision,
            "surface_area_manifest_sha256": contract.surface_area_manifest_sha256,
            "volume_weighting": "one_per_native_cell",
            "geometric_cell_volume_weights_used": False,
            "diagnostic_profile_file": contract.profile_file,
            "diagnostic_profile_sha256": contract.profile_sha256,
            "diagnostic_profile_series_contract": {
                "pressure_station_ids": list(contract.profile_cp_cut_ids),
                "velocity_station_ids": list(contract.profile_velocity_station_ids),
                "velocity_station_sample_counts": list(
                    contract.profile_velocity_line_sample_counts
                ),
                "canonical_order": "pressure_profiles_then_velocity_profiles",
            },
            "force_truth_file": contract.force_truth_file,
            "force_truth_sha256": contract.force_truth_sha256,
        },
        "metric_values": metric_values,
        "primary_field_metric_ids": list(active_primary_field_metric_ids),
        "ranked_force_metric_ids": list(RANKED_FORCE_METRIC_IDS),
        "report_only_force_metric_ids": list(REPORT_ONLY_FORCE_METRIC_IDS),
        "diagnostic_metric_ids": list(DIAGNOSTIC_METRIC_IDS),
        "ranked_diagnostic_metric_ids": list(RANKED_DIAGNOSTIC_METRIC_IDS),
        "report_only_diagnostic_metric_ids": list(
            REPORT_ONLY_DIAGNOSTIC_METRIC_IDS
        ),
        "metric_reductions": reductions,
        "diagnostic_availability": diagnostic_availability,
        "force_truth_audit": {
            "case_count": len(contract.native_source_pin.cases),
            "selected_case_count": len(contract.case_ids),
            "authoritative_table_hash_verified_before_reduction": True,
            "CmPitch_truth_equation": "(clf-clr)/2",
            "maximum_authoritative_lift_closure_abs_selected_cases": max(truth_closures),
            "maximum_predicted_lift_closure_abs_selected_cases": max(predicted_closures),
        },
        "cases": case_rows,
        "claims": {
            "bounded_score_definition_active": True,
            "overall_score_computed": False,
            "field_force_or_diagnostic_component_score_computed": False,
            "scoring_contract_active": False,
            "official_submission": False,
            "leaderboard_eligible": False,
            "owner_scientific_approval": False,
            "independent_participant_dry_run": False,
        },
    }
    regional_diagnostics: Mapping[str, object] | None = None
    if all(core.regional_diagnostics is not None for core in core_cases):
        try:
            regional_diagnostics = aggregate_regional_diagnostics(
                [
                    {
                        "case_id": core.case_id,
                        "core": {
                            "report_only_regional_diagnostics": dict(
                                core.regional_diagnostics or {}
                            )
                        },
                    }
                    for core in core_cases
                ],
                case_ids=contract.case_ids,
            )
        except RegionalAggregateError as error:
            raise DrivAerDatasetScorerError(
                f"cannot aggregate zero-weight regional diagnostics: {error}"
            ) from error
    _assert_no_absolute_paths(evidence)
    if regional_diagnostics is not None:
        _assert_no_absolute_paths(regional_diagnostics)
    return CandidateDatasetEvaluation(
        evidence=evidence,
        regional_diagnostics=regional_diagnostics,
    )


def schema_v3_case_metrics_candidate_adapter(
    evaluation: CandidateDatasetEvaluation,
    *,
    submission_id: str,
    candidate_support_release_id: str,
    candidate_support_manifest_sha256: str,
) -> dict[str, object]:
    """Return schema-v3 case-metrics-shaped *candidate* evidence.

    The caller must provide a candidate support identity; this helper rejects a
    release ID without the word ``candidate`` so its output cannot silently be
    presented as an activated support release.  Unavailable diagnostic metrics
    are omitted from every case and the dataset aggregate rather than reduced
    over a subset.
    """

    if not isinstance(evaluation, CandidateDatasetEvaluation):
        raise DrivAerDatasetScorerError("evaluation must be CandidateDatasetEvaluation")
    if not isinstance(submission_id, str) or _SUBMISSION_ID_RE.fullmatch(submission_id) is None:
        raise DrivAerDatasetScorerError("submission_id is not schema-v3 compatible")
    if (
        not isinstance(candidate_support_release_id, str)
        or _SAFE_ID_RE.fullmatch(candidate_support_release_id) is None
        or "candidate" not in candidate_support_release_id
    ):
        raise DrivAerDatasetScorerError(
            "candidate_support_release_id must be a safe ID explicitly containing 'candidate'"
        )
    manifest_sha = _sha256(
        candidate_support_manifest_sha256, "candidate support manifest SHA-256"
    )
    evidence = evaluation.to_json()
    prediction_scope = evidence.get("prediction_scope")
    if prediction_scope not in PREDICTION_SCOPES:
        raise DrivAerDatasetScorerError("candidate dataset prediction scope is invalid")
    split = _mapping(evidence["split"], "candidate dataset split")
    aggregate_values = _mapping(evidence["metric_values"], "candidate metric_values")
    numeric_aggregate = {
        metric_id: float(value)
        for metric_id, value in aggregate_values.items()
        if value is not None
    }
    unavailable_diagnostics = {
        metric_id
        for metric_id in DIAGNOSTIC_METRIC_IDS
        if aggregate_values.get(metric_id) is None
    }
    cases: list[dict[str, object]] = []
    raw_cases = evidence.get("cases")
    if not isinstance(raw_cases, list):
        raise DrivAerDatasetScorerError("candidate dataset evidence cases are malformed")
    surface_ids = tuple(
        metric_id for metric_id in ALL_FIELD_METRIC_IDS if metric_id.startswith("surface_") or metric_id.startswith("drivaerml_surface_")
    )
    volume_ids = tuple(metric_id for metric_id in ALL_FIELD_METRIC_IDS if metric_id not in surface_ids)
    for raw_case in raw_cases:
        case = _mapping(raw_case, "candidate dataset case")
        fields = _mapping(case["field_metric_values"], "candidate case field metrics")
        statistics = _mapping(case["field_metric_sufficient_statistics"], "candidate case statistics")
        counts = _mapping(case["support_counts"], "candidate case support counts")
        force = _mapping(case["force"], "candidate case force")
        predicted = _mapping(force["prediction"], "candidate predicted force")
        truth = _mapping(force["truth"], "candidate force truth")
        diagnostics = _mapping(case["diagnostic_metric_values"], "candidate diagnostics")
        nonspatial = {
            # A one-case RMSE is the absolute error.  Keep the declared metric
            # IDs here so per-case sufficient evidence remains traceable to the
            # equal-case aggregate rather than introducing undeclared aliases.
            "field_integrated_cd_rmse": abs(float(predicted["Cd"]) - float(truth["Cd"])),
            "field_integrated_cl_rmse": abs(float(predicted["Cl"]) - float(truth["Cl"])),
            "field_integrated_cmpitch_rmse": abs(float(predicted["CmPitch"]) - float(truth["CmPitch"])),
            "field_integrated_clf_rmse": abs(float(predicted["Clf"]) - float(truth["Clf"])),
            "field_integrated_clr_rmse": abs(float(predicted["Clr"]) - float(truth["Clr"])),
            "field_integrated_lift_closure_max_abs": float(force["predicted_lift_closure_abs"]),
        }
        for metric_id in DIAGNOSTIC_METRIC_IDS:
            if metric_id not in unavailable_diagnostics:
                nonspatial[metric_id] = float(diagnostics[metric_id])
        supports = []
        support_layout = [("surface_native_cells", surface_ids)]
        if prediction_scope == PREDICTION_SCOPE_FULL:
            support_layout.append(("volume_native_cells", volume_ids))
        for support_id, metric_ids in support_layout:
            count = _integer(counts[support_id], f"{case['case_id']}/{support_id} count", minimum=1)
            support_statistics = {
                metric_id: dict(statistics[metric_id])
                for metric_id in metric_ids
                if metric_id in statistics
            }
            supports.append(
                {
                    "support_id": support_id,
                    "support_count": count,
                    "scored_count": count,
                    "coverage_fraction": 1.0,
                    "weight_coverage_fraction": 1.0,
                    "unmapped_count": 0,
                    "extrapolated_count": 0,
                    "metric_values": {metric_id: float(fields[metric_id]) for metric_id in metric_ids},
                    "metric_sufficient_statistics": support_statistics,
                }
            )
        cases.append(
            {
                "case_id": case["case_id"],
                "supports": supports,
                "nonspatial_metric_values": nonspatial,
                "force_coefficients": {
                    output_key: _finite(
                        predicted[internal_key],
                        f"{case['case_id']} predicted force coefficient {internal_key}",
                    )
                    for output_key, internal_key in zip(
                        SCHEMA_V3_FORCE_COEFFICIENT_KEYS,
                        _INTERNAL_FORCE_COEFFICIENT_KEYS,
                        strict=True,
                    )
                },
            }
        )
    result: dict[str, object] = {
        "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": split["split_id"],
        "case_set_id": split["case_set_id"],
        "scoring_support_release_id": candidate_support_release_id,
        "scoring_support_manifest_sha256": manifest_sha,
        "case_count": split["case_count"],
        "cases": cases,
        "metric_values": numeric_aggregate,
    }
    _assert_no_absolute_paths(result)
    return result


def _canonical_json_payload(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _relative_profile_expected_keys() -> tuple[tuple[str, str, str, str, str], ...]:
    """Return the closed logical namespace for one relative-v3 case."""

    return (
        *(
            (
                "velocity_profiles",
                "drivaerml-autocfd5-constant-v1",
                station,
                "velocity_ratio",
                "materialized",
            )
            for station in _CONSTANT_VELOCITY_STATIONS
        ),
        *(
            (
                "velocity_profiles",
                "drivaerml-velocity-relative-v3",
                station,
                "velocity_ratio",
                "materialized",
            )
            for station in _RELATIVE_VELOCITY_STATIONS
        ),
        *(
            (
                "pressure_profiles",
                "drivaerml_cp_constant_v1",
                station,
                "cp",
                "materialized",
            )
            for station in _CONSTANT_CP_STATIONS
        ),
        *(
            (
                "pressure_profiles",
                "drivaerml_cp_relative_v1",
                station,
                "cp",
                "shared_alias" if station in _SHARED_CP_SUPPORT_IDS else "materialized",
            )
            for station in _RELATIVE_CP_STATIONS
        ),
    )


def _relative_support_expected_keys() -> tuple[tuple[str, str, str], ...]:
    """Return the relative family/station/representation identity namespace."""

    return tuple(
        (family_id, station_id, representation)
        for _panel_id, family_id, station_id, _quantity_id, representation in (
            _relative_profile_expected_keys()
        )
        if family_id
        in {"drivaerml-velocity-relative-v3", "drivaerml_cp_relative_v1"}
    )


def _constant_support_expected_keys() -> tuple[tuple[str, str], ...]:
    """Return the constant family/station identity namespace."""

    return tuple(
        (family_id, station_id)
        for _panel_id, family_id, station_id, _quantity_id, representation in (
            _relative_profile_expected_keys()
        )
        if representation == "materialized"
        and family_id
        in {"drivaerml-autocfd5-constant-v1", "drivaerml_cp_constant_v1"}
    )


@lru_cache(maxsize=4)
def _load_relative_series_support_index_cached(
    index_path_text: str,
    expected_sha256: str,
) -> Mapping[tuple[str, str, str], Mapping[str, object]]:
    """Load and fully validate the immutable all-case relative support index."""

    expected_digest = _sha256(
        expected_sha256,
        "retained relative series support index SHA-256",
    )
    index_path = Path(index_path_text)
    root, _source, actual_digest = _read_json(
        index_path,
        "retained relative series support index",
    )
    if actual_digest != expected_digest:
        raise DrivAerDatasetScorerError(
            "retained relative series support index SHA-256 mismatch"
        )
    root = _exact_keys(
        root,
        {
            "schema",
            "schema_version",
            "dataset_id",
            "contract_id",
            "scope",
            "case_count",
            "series_per_case",
            "source_bindings",
            "cases",
        },
        "retained relative series support index",
    )
    if (
        root["schema"] != RELATIVE_SERIES_SUPPORT_INDEX_SCHEMA
        or root["schema_version"] != 2
        or root["dataset_id"] != "drivaerml"
        or root["contract_id"] != RELATIVE_PROFILE_CONTRACT_ID
        or root["scope"] != "relative_families_only"
        or root["case_count"] != 484
        or root["series_per_case"] != RELATIVE_SERIES_SUPPORT_PER_CASE
        or not isinstance(root["source_bindings"], Mapping)
    ):
        raise DrivAerDatasetScorerError(
            "retained relative series support index identity is invalid"
        )
    source_bindings = _exact_keys(
        root["source_bindings"],
        {"manifests", "producer_identity_fields", "public_dataset"},
        "retained relative series support index source_bindings",
    )
    expected_manifest_bindings = [
        {
            "path": (
                "benchmark-specs/drivaerml/support/relative-v3/manifests/"
                "velocity-placement-all484-v1.json"
            ),
            "role": "velocity_placement",
            "schema": "drivaerml-relative-velocity-v3-production-input-manifest-v1",
            "schema_version": 1,
            "sha256": "70627d6af9e6b254739b29d54d856b470d6152066e09a1097ebe20c034179925",
        },
        {
            "path": (
                "benchmark-specs/drivaerml/support/relative-v3/manifests/"
                "velocity-mapping-all484-v1.json"
            ),
            "role": "velocity_mapping",
            "schema": "drivaerml-velocity-relative-v3-mapping-aggregate-v1",
            "schema_version": 1,
            "sha256": "9b88c36e2268bf72c9baec9418ef2d9afc9faba1d98c9c12ed73b79e683a6a3d",
        },
        {
            "path": (
                "benchmark-specs/drivaerml/support/relative-v3/manifests/"
                "cp-native-support-all484-v3.json"
            ),
            "role": "cp",
            "schema": "drivaerml-relative-cp-native-support-manifest-v3",
            "schema_version": 3,
            "sha256": "4e6a4c3495ea4938895868162480dcb20b5bbea42114c94013a2c76e26128c90",
        },
    ]
    if source_bindings["manifests"] != expected_manifest_bindings:
        raise DrivAerDatasetScorerError(
            "retained relative series support index manifest provenance is invalid"
        )
    expected_identity_fields = {
        "relative_cp_materialized_coordinate": (
            "support.moving_cuts[].rows[].interval_arc_end_m encoded by "
            "fluidsbench-drivaerml-coordinate-array-v1"
        ),
        "relative_cp_moving_support": (
            "support.moving_cuts[].support_identity_sha256"
        ),
        "relative_cp_placement_receipt": "receipt_identity.sha256",
        "relative_cp_shared_alias_support": (
            "support.centerline_aliases[].canonical_cut_support_sha256"
        ),
        "relative_velocity_placement_receipt": (
            "sha256(exact placement receipt bytes)"
        ),
        "relative_velocity_materialized_coordinate": (
            "placement CSV line_fraction selected in order by valid rows from the "
            "aggregate-bound 10mm mapping, encoded by "
            "fluidsbench-drivaerml-coordinate-array-v1"
        ),
        "relative_velocity_support": (
            "profiles[].coordinates_binary64_be_sha256"
        ),
    }
    if source_bindings["producer_identity_fields"] != expected_identity_fields:
        raise DrivAerDatasetScorerError(
            "retained relative series support index identity provenance is invalid"
        )
    expected_public_dataset = {
        "native_source_pin_path": (
            "benchmark-specs/drivaerml/proposal/native-source-pin.json"
        ),
        "native_source_pin_sha256": OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
        "repository": OFFICIAL_REPOSITORY_ID,
        "revision": OFFICIAL_REPOSITORY_REVISION,
    }
    if source_bindings["public_dataset"] != expected_public_dataset:
        raise DrivAerDatasetScorerError(
            "retained relative series support index dataset provenance is invalid"
        )

    benchmark_root = index_path.parents[2]
    pin, _pin_source, pin_digest = _read_json(
        benchmark_root / "proposal" / "native-source-pin.json",
        "official DrivAerML native-source pin",
    )
    raw_official_cases = pin.get("cases")
    if (
        pin_digest != OFFICIAL_NATIVE_SOURCE_PIN_SHA256
        or pin.get("schema")
        != "drivaerml-fluidsbench-public-native-source-pin-v1"
        or pin.get("schema_version") != 1
        or not isinstance(raw_official_cases, list)
    ):
        raise DrivAerDatasetScorerError(
            "official DrivAerML native-source pin identity is invalid"
        )
    official_case_ids = tuple(
        _case_id(
            case.get("case_id") if isinstance(case, Mapping) else None,
            f"official DrivAerML case {position}",
        )
        for position, case in enumerate(raw_official_cases)
    )
    if len(official_case_ids) != 484 or len(set(official_case_ids)) != 484:
        raise DrivAerDatasetScorerError(
            "official DrivAerML native-source pin does not contain 484 unique cases"
        )

    raw_cases = root["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != 484:
        raise DrivAerDatasetScorerError(
            "retained relative series support index must contain exactly 484 cases"
        )
    expected_series_keys = _relative_support_expected_keys()
    result: dict[tuple[str, str, str], Mapping[str, object]] = {}
    observed_case_ids: list[str] = []
    for case_position, raw_case in enumerate(raw_cases):
        case = _exact_keys(
            raw_case,
            {"case_id", "series"},
            f"retained relative support case {case_position}",
        )
        case_id = _case_id(
            case["case_id"],
            f"retained relative support case {case_position} ID",
        )
        observed_case_ids.append(case_id)
        raw_series = case["series"]
        if (
            not isinstance(raw_series, list)
            or len(raw_series) != RELATIVE_SERIES_SUPPORT_PER_CASE
        ):
            raise DrivAerDatasetScorerError(
                f"retained relative support case {case_id} must contain exactly "
                f"{RELATIVE_SERIES_SUPPORT_PER_CASE} series"
            )
        observed_series_keys: list[tuple[str, str, str]] = []
        for series_position, raw_series_item in enumerate(raw_series):
            label = (
                f"retained relative support {case_id} series {series_position}"
            )
            raw_series_mapping = _mapping(raw_series_item, label)
            raw_representation = _string(
                raw_series_mapping.get("representation"),
                f"{label} representation",
            )
            expected_fields = {
                "family_id",
                "station_id",
                "representation",
                "support_identity_sha256",
                "placement_receipt_identity_sha256",
            }
            if raw_representation == "materialized":
                expected_fields |= {
                    "coordinate_count",
                    "coordinate_identity_sha256",
                }
            elif raw_representation != "shared_alias":
                raise DrivAerDatasetScorerError(
                    f"{label} representation is invalid"
                )
            series = _exact_keys(raw_series_mapping, expected_fields, label)
            family_id = _string(series["family_id"], f"{label} family_id")
            station_id = _string(series["station_id"], f"{label} station_id")
            representation = _string(
                series["representation"], f"{label} representation"
            )
            logical_key = (family_id, station_id, representation)
            observed_series_keys.append(logical_key)
            support_digest = _sha256(
                series["support_identity_sha256"],
                f"{label} support identity",
            )
            receipt_digest = _sha256(
                series["placement_receipt_identity_sha256"],
                f"{label} placement receipt identity",
            )
            if support_digest == _ZERO_SHA256 or receipt_digest == _ZERO_SHA256:
                raise DrivAerDatasetScorerError(
                    f"{label} contains an all-zero identity"
                )
            retained_identity: dict[str, object] = {
                "representation": representation,
                "support_identity_sha256": support_digest,
                "placement_receipt_identity_sha256": receipt_digest,
            }
            if representation == "materialized":
                coordinate_digest = _sha256(
                    series["coordinate_identity_sha256"],
                    f"{label} coordinate identity",
                )
                if coordinate_digest == _ZERO_SHA256:
                    raise DrivAerDatasetScorerError(
                        f"{label} contains an all-zero coordinate identity"
                    )
                retained_identity.update(
                    {
                        "coordinate_count": _integer(
                            series["coordinate_count"],
                            f"{label} coordinate count",
                            minimum=2,
                        ),
                        "coordinate_identity_sha256": coordinate_digest,
                    }
                )
            result[(case_id, family_id, station_id)] = retained_identity
        if tuple(observed_series_keys) != expected_series_keys:
            raise DrivAerDatasetScorerError(
                f"retained relative support {case_id} series coverage/order "
                "differs from the relative-v3 contract"
            )
    if tuple(observed_case_ids) != official_case_ids:
        raise DrivAerDatasetScorerError(
            "retained relative series support index case coverage/order differs "
            "from the official 484-case set"
        )
    if len(result) != 484 * RELATIVE_SERIES_SUPPORT_PER_CASE:
        raise DrivAerDatasetScorerError(
            "retained relative series support index contains duplicate logical keys"
        )
    return result


def _relative_series_support_index() -> Mapping[
    tuple[str, str, str], Mapping[str, object]
]:
    """Return the release-pinned relative support lookup used by submissions."""

    return _load_relative_series_support_index_cached(
        str(RELATIVE_SERIES_SUPPORT_INDEX_PATH),
        RELATIVE_SERIES_SUPPORT_INDEX_SHA256,
    )


@lru_cache(maxsize=4)
def _load_constant_series_support_index_cached(
    index_path_text: str,
    expected_sha256: str,
) -> Mapping[tuple[str, str, str], Mapping[str, object]]:
    """Load the immutable all-case constant support identities."""

    expected_digest = _sha256(
        expected_sha256,
        "retained constant series support index SHA-256",
    )
    index_path = Path(index_path_text)
    root, _source, actual_digest = _read_json(
        index_path,
        "retained constant series support index",
    )
    if actual_digest != expected_digest:
        raise DrivAerDatasetScorerError(
            "retained constant series support index SHA-256 mismatch"
        )
    root = _exact_keys(
        root,
        {
            "schema",
            "schema_version",
            "dataset_id",
            "contract_id",
            "scope",
            "case_count",
            "series_per_case",
            "source_bindings",
            "cases",
        },
        "retained constant series support index",
    )
    if (
        root["schema"] != CONSTANT_SERIES_SUPPORT_INDEX_SCHEMA
        or root["schema_version"] != 1
        or root["dataset_id"] != "drivaerml"
        or root["contract_id"] != RELATIVE_PROFILE_CONTRACT_ID
        or root["scope"] != "constant_families_all_official_cases"
        or root["case_count"] != 484
        or root["series_per_case"] != CONSTANT_SERIES_SUPPORT_PER_CASE
    ):
        raise DrivAerDatasetScorerError(
            "retained constant series support index identity is invalid"
        )
    expected_sources = {
        "native_profile_truth": {
            "master_index_path": (
                "assets/data/profile-ground-truth/datasets/drivaerml/"
                "native-v3/index.json"
            ),
            "master_index_sha256": (
                "e7cf14f161fc7dbf22794e6f66db4e157329be960d2a4368140e08cf0608a5ae"
            ),
            "provenance_sha256": (
                "6030df1dce11c4fbf6028d4e17ab39394cfdd3e7bb4d5a7ff81f33408849f221"
            ),
            "release_receipt_sha256": (
                "ef59ca838c828ac1c5505ef4d35f1b91a55e07ad90df359491599114ec0d878c"
            ),
            "dataset_repository": OFFICIAL_REPOSITORY_ID,
            "dataset_revision": OFFICIAL_REPOSITORY_REVISION,
        },
        "public_dataset": {
            "native_source_pin_path": (
                "benchmark-specs/drivaerml/proposal/native-source-pin.json"
            ),
            "native_source_pin_sha256": OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
            "repository": OFFICIAL_REPOSITORY_ID,
            "revision": OFFICIAL_REPOSITORY_REVISION,
        },
        "coordinate_identity_encoding": (
            "fluidsbench-drivaerml-coordinate-array-v1"
        ),
    }
    if root["source_bindings"] != expected_sources:
        raise DrivAerDatasetScorerError(
            "retained constant series support provenance is invalid"
        )

    benchmark_root = index_path.parents[2]
    pin, _pin_source, pin_digest = _read_json(
        benchmark_root / "proposal" / "native-source-pin.json",
        "official DrivAerML native-source pin",
    )
    raw_official_cases = pin.get("cases")
    if (
        pin_digest != OFFICIAL_NATIVE_SOURCE_PIN_SHA256
        or pin.get("schema")
        != "drivaerml-fluidsbench-public-native-source-pin-v1"
        or pin.get("schema_version") != 1
        or not isinstance(raw_official_cases, list)
    ):
        raise DrivAerDatasetScorerError(
            "official DrivAerML native-source pin identity is invalid"
        )
    official_case_ids = tuple(
        _case_id(
            case.get("case_id") if isinstance(case, Mapping) else None,
            f"official DrivAerML case {position}",
        )
        for position, case in enumerate(raw_official_cases)
    )
    if len(official_case_ids) != 484 or len(set(official_case_ids)) != 484:
        raise DrivAerDatasetScorerError(
            "official DrivAerML native-source pin does not contain 484 unique cases"
        )

    raw_cases = root["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != 484:
        raise DrivAerDatasetScorerError(
            "retained constant series support index must contain exactly 484 cases"
        )
    expected_series_keys = _constant_support_expected_keys()
    result: dict[tuple[str, str, str], Mapping[str, object]] = {}
    observed_case_ids: list[str] = []
    for case_position, raw_case in enumerate(raw_cases):
        case = _exact_keys(
            raw_case,
            {"case_id", "series"},
            f"retained constant support case {case_position}",
        )
        case_id = _case_id(
            case["case_id"],
            f"retained constant support case {case_position} ID",
        )
        observed_case_ids.append(case_id)
        raw_series = case["series"]
        if (
            not isinstance(raw_series, list)
            or len(raw_series) != CONSTANT_SERIES_SUPPORT_PER_CASE
        ):
            raise DrivAerDatasetScorerError(
                f"retained constant support case {case_id} must contain exactly "
                f"{CONSTANT_SERIES_SUPPORT_PER_CASE} series"
            )
        observed_series_keys: list[tuple[str, str]] = []
        for series_position, raw_item in enumerate(raw_series):
            label = f"retained constant support {case_id} series {series_position}"
            item = _exact_keys(
                raw_item,
                {
                    "family_id",
                    "station_id",
                    "representation",
                    "support_identity_sha256",
                    "placement_receipt_identity_sha256",
                    "coordinate_count",
                    "coordinate_identity_sha256",
                },
                label,
            )
            family_id = _string(item["family_id"], f"{label} family_id")
            station_id = _string(item["station_id"], f"{label} station_id")
            key = (family_id, station_id)
            observed_series_keys.append(key)
            if item["representation"] != "materialized":
                raise DrivAerDatasetScorerError(f"{label} must be materialized")
            support_digest = _sha256(
                item["support_identity_sha256"], f"{label} support identity"
            )
            receipt_digest = _sha256(
                item["placement_receipt_identity_sha256"],
                f"{label} placement receipt identity",
            )
            coordinate_digest = _sha256(
                item["coordinate_identity_sha256"],
                f"{label} coordinate identity",
            )
            if _ZERO_SHA256 in {
                support_digest,
                receipt_digest,
                coordinate_digest,
            }:
                raise DrivAerDatasetScorerError(
                    f"{label} contains an all-zero identity"
                )
            result[(case_id, family_id, station_id)] = {
                "representation": "materialized",
                "support_identity_sha256": support_digest,
                "placement_receipt_identity_sha256": receipt_digest,
                "coordinate_count": _integer(
                    item["coordinate_count"],
                    f"{label} coordinate count",
                    minimum=2,
                ),
                "coordinate_identity_sha256": coordinate_digest,
            }
        if tuple(observed_series_keys) != expected_series_keys:
            raise DrivAerDatasetScorerError(
                f"retained constant support {case_id} series coverage/order "
                "differs from the contract"
            )
    if tuple(observed_case_ids) != official_case_ids:
        raise DrivAerDatasetScorerError(
            "retained constant series support index case coverage/order differs "
            "from the official 484-case set"
        )
    if len(result) != 484 * CONSTANT_SERIES_SUPPORT_PER_CASE:
        raise DrivAerDatasetScorerError(
            "retained constant series support index contains duplicate logical keys"
        )
    return result


def _constant_series_support_index() -> Mapping[
    tuple[str, str, str], Mapping[str, object]
]:
    """Return the native-v3-bound constant support lookup."""

    return _load_constant_series_support_index_cached(
        str(CONSTANT_SERIES_SUPPORT_INDEX_PATH),
        CONSTANT_SERIES_SUPPORT_INDEX_SHA256,
    )


@lru_cache(maxsize=4)
def _load_run419_constant_series_support_index_cached(
    index_path_text: str,
    expected_sha256: str,
) -> Mapping[tuple[str, str], Mapping[str, object]]:
    """Load the immutable constant support used only by the real run_419 fixture."""

    expected_digest = _sha256(
        expected_sha256,
        "retained run_419 constant series support index SHA-256",
    )
    index_path = Path(index_path_text)
    root, _source, actual_digest = _read_json(
        index_path,
        "retained run_419 constant series support index",
    )
    if actual_digest != expected_digest:
        raise DrivAerDatasetScorerError(
            "retained run_419 constant series support index SHA-256 mismatch"
        )
    root = _exact_keys(
        root,
        {
            "schema",
            "schema_version",
            "dataset_id",
            "contract_id",
            "scope",
            "case_id",
            "series_count",
            "source_bindings",
            "series",
        },
        "retained run_419 constant series support index",
    )
    if (
        root["schema"] != RUN419_CONSTANT_SERIES_SUPPORT_INDEX_SCHEMA
        or root["schema_version"] != 1
        or root["dataset_id"] != "drivaerml"
        or root["contract_id"] != RELATIVE_PROFILE_CONTRACT_ID
        or root["scope"] != "run_419_report_only_valid_native_support"
        or root["case_id"] != "run_419"
        or root["series_count"] != RUN419_CONSTANT_SERIES_SUPPORT_COUNT
    ):
        raise DrivAerDatasetScorerError(
            "retained run_419 constant series support index identity is invalid"
        )
    expected_sources = {
        "public_dataset": {
            "native_source_pin_path": (
                "benchmark-specs/drivaerml/proposal/native-source-pin.json"
            ),
            "native_source_pin_sha256": OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
            "repository": OFFICIAL_REPOSITORY_ID,
            "revision": OFFICIAL_REPOSITORY_REVISION,
        },
        "constant_velocity": {
            "mapping_sha256": (
                "9866147358a34540e6f6be4b95e3cdc46681693a22bd93f0d5ef790144bddcec"
            ),
            "receipt_sha256": (
                "63df3170bb71edcdb1a9892f64ed33f4ecd028f7bbb172e91e06efb34bee96bf"
            ),
            "profile_registry_sha256": (
                "df22bc807b62f925c32659d681ac44064e6acf46449038b8431b1e9139aba1e8"
            ),
            "sample_count": 3_756,
            "valid_count": 3_684,
            "invalid_count": 72,
            "coordinate_selection": (
                "ordered 10mm frozen-grid mapping rows with valid=true; "
                "unsupported rows are omitted and interior gaps are preserved"
            ),
        },
        "constant_cp": {
            "aggregate_sha256": (
                "bc2a7337ae87942e9b9ae4f57a5ba408bbdf82d03307d47110fcc85edd449c7c"
            ),
            "case_support_json_sha256": (
                "e2fb791194151aab95e03b21cf7689e4b78994d21073817c663599e12cc942e8"
            ),
            "case_support_csv_sha256": (
                "e2e0c84f4e3306ad4ed3b67a7de3ad10dc947c090103cb2de7c5b91fe7e27b8a"
            ),
            "case_support_identity_sha256": (
                "b5ed95dbbcff30ad6d1ebc07c1fa1c484665cd0416b1d6d2d53945ef64b61e57"
            ),
            "receipt_file_sha256": (
                "b1ea8c26f753935e5c9826495dfb77cf97bd542fcba1fceeab575030cefd9195"
            ),
            "receipt_identity_sha256": (
                "4c120f02ed318a55025330c232c3a46b16820baadc81515dc729d31cc822926d"
            ),
        },
        "coordinate_identity_encoding": (
            "fluidsbench-drivaerml-coordinate-array-v1"
        ),
    }
    if root["source_bindings"] != expected_sources:
        raise DrivAerDatasetScorerError(
            "retained run_419 constant series support provenance is invalid"
        )
    expected_keys = tuple(
        (family_id, station_id)
        for _panel, family_id, station_id, _quantity, representation in (
            _relative_profile_expected_keys()
        )
        if representation == "materialized"
        and family_id
        in {"drivaerml-autocfd5-constant-v1", "drivaerml_cp_constant_v1"}
    )
    raw_series = root["series"]
    if (
        not isinstance(raw_series, list)
        or len(raw_series) != RUN419_CONSTANT_SERIES_SUPPORT_COUNT
    ):
        raise DrivAerDatasetScorerError(
            "retained run_419 constant support must contain exactly 20 series"
        )
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    observed_keys: list[tuple[str, str]] = []
    for position, raw_item in enumerate(raw_series):
        label = f"retained run_419 constant support series {position}"
        item = _exact_keys(
            raw_item,
            {
                "family_id",
                "station_id",
                "representation",
                "support_identity_sha256",
                "placement_receipt_identity_sha256",
                "coordinate_count",
                "coordinate_identity_sha256",
            },
            label,
        )
        family_id = _string(item["family_id"], f"{label} family_id")
        station_id = _string(item["station_id"], f"{label} station_id")
        key = (family_id, station_id)
        observed_keys.append(key)
        if item["representation"] != "materialized":
            raise DrivAerDatasetScorerError(f"{label} must be materialized")
        support_digest = _sha256(
            item["support_identity_sha256"], f"{label} support identity"
        )
        receipt_digest = _sha256(
            item["placement_receipt_identity_sha256"],
            f"{label} placement receipt identity",
        )
        coordinate_digest = _sha256(
            item["coordinate_identity_sha256"], f"{label} coordinate identity"
        )
        if _ZERO_SHA256 in {support_digest, receipt_digest, coordinate_digest}:
            raise DrivAerDatasetScorerError(f"{label} contains an all-zero identity")
        result[key] = {
            "representation": "materialized",
            "support_identity_sha256": support_digest,
            "placement_receipt_identity_sha256": receipt_digest,
            "coordinate_count": _integer(
                item["coordinate_count"], f"{label} coordinate count", minimum=2
            ),
            "coordinate_identity_sha256": coordinate_digest,
        }
    if tuple(observed_keys) != expected_keys or len(result) != len(expected_keys):
        raise DrivAerDatasetScorerError(
            "retained run_419 constant support coverage/order differs from the contract"
        )
    return result


def _run419_constant_series_support_index() -> Mapping[
    tuple[str, str], Mapping[str, object]
]:
    """Return the pinned lookup for the report-only real run_419 fixture."""

    return _load_run419_constant_series_support_index_cached(
        str(RUN419_CONSTANT_SERIES_SUPPORT_INDEX_PATH),
        RUN419_CONSTANT_SERIES_SUPPORT_INDEX_SHA256,
    )


def _normalize_relative_profile_series(
    raw_series: object,
    *,
    case_id: str,
    position: int,
) -> tuple[tuple[str, str, str, str, str], dict[str, object]]:
    """Validate one namespaced series and return its logical key and JSON form."""

    label = f"{case_id} relative profile series {position}"
    series = _mapping(raw_series, label)
    representation = _string(series.get("representation"), f"{label} representation")
    common = {
        "panel_id",
        "family_id",
        "placement_mode",
        "station_id",
        "quantity_id",
        "scoring_role",
        "representation",
        "placement_receipt_identity_sha256",
    }
    if representation == "materialized":
        expected_fields = common | {
            "coordinate_id",
            "coordinate_unit",
            "coordinate",
            "prediction",
            "support_identity_sha256",
        }
    elif representation == "shared_alias":
        expected_fields = common | {"shared_support_ref"}
    else:
        raise DrivAerDatasetScorerError(
            f"{label} representation must be materialized or shared_alias"
        )
    series = _exact_keys(series, expected_fields, label)
    panel_id = _string(series["panel_id"], f"{label} panel_id")
    family_id = _string(series["family_id"], f"{label} family_id")
    station_id = _string(series["station_id"], f"{label} station_id")
    quantity_id = _string(series["quantity_id"], f"{label} quantity_id")
    key = (panel_id, family_id, station_id, quantity_id, representation)
    expected_keys = set(_relative_profile_expected_keys())
    if key not in expected_keys:
        raise DrivAerDatasetScorerError(
            f"{label} has an undeclared family/station representation {key}"
        )
    relative = family_id in {
        "drivaerml-velocity-relative-v3",
        "drivaerml_cp_relative_v1",
    }
    expected_mode = "relative" if relative else "constant"
    expected_role = "report_only" if relative else "inherits_parent_candidate"
    if series["placement_mode"] != expected_mode:
        raise DrivAerDatasetScorerError(
            f"{label} placement_mode must be {expected_mode!r}"
        )
    if series["scoring_role"] != expected_role:
        raise DrivAerDatasetScorerError(
            f"{label} scoring_role must be {expected_role!r}"
        )
    receipt_digest = series["placement_receipt_identity_sha256"]
    if (
        not isinstance(receipt_digest, str)
        or _SHA256_RE.fullmatch(receipt_digest) is None
        or receipt_digest == _ZERO_SHA256
    ):
        raise DrivAerDatasetScorerError(
            f"{label} placement receipt identity must be a nonzero lowercase SHA-256"
        )

    normalized = dict(series)
    if representation == "shared_alias":
        reference = _exact_keys(
            series["shared_support_ref"],
            {
                "shared_support_id",
                "canonical_family_id",
                "canonical_station_id",
                "canonical_support_identity_sha256",
            },
            f"{label} shared_support_ref",
        )
        expected_shared_id = _SHARED_CP_SUPPORT_IDS[station_id]
        if (
            reference["shared_support_id"] != expected_shared_id
            or reference["canonical_family_id"] != "drivaerml_cp_constant_v1"
            or reference["canonical_station_id"] != station_id
            or not isinstance(reference["canonical_support_identity_sha256"], str)
            or _SHA256_RE.fullmatch(reference["canonical_support_identity_sha256"])
            is None
            or reference["canonical_support_identity_sha256"] == _ZERO_SHA256
        ):
            raise DrivAerDatasetScorerError(
                f"{label} shared alias does not bind its canonical constant Cp support"
            )
        normalized["shared_support_ref"] = dict(reference)
        return key, normalized

    support_digest = series["support_identity_sha256"]
    if (
        not isinstance(support_digest, str)
        or _SHA256_RE.fullmatch(support_digest) is None
        or support_digest == _ZERO_SHA256
    ):
        raise DrivAerDatasetScorerError(
            f"{label} support identity must be a nonzero lowercase SHA-256"
        )
    coordinates = series["coordinate"]
    predictions = series["prediction"]
    if (
        not isinstance(coordinates, list)
        or not isinstance(predictions, list)
        or len(coordinates) < 2
        or len(coordinates) != len(predictions)
    ):
        raise DrivAerDatasetScorerError(
            f"{label} coordinate and prediction arrays must have equal length >= 2"
        )
    normalized_coordinates = [
        _finite(value, f"{label} coordinate {index}")
        for index, value in enumerate(coordinates)
    ]
    normalized_predictions = [
        _finite(value, f"{label} prediction {index}")
        for index, value in enumerate(predictions)
    ]
    # A relative moving Cp station may contain several explicitly ordered
    # producer intervals whose local arc coordinate resets at each boundary.
    # Its exact order is bound below by the retained coordinate identity.
    coordinate_must_be_strictly_increasing = (
        family_id != "drivaerml_cp_relative_v1"
    )
    if coordinate_must_be_strictly_increasing and any(
        right <= left
        for left, right in zip(normalized_coordinates, normalized_coordinates[1:])
    ):
        raise DrivAerDatasetScorerError(
            f"{label} coordinates must be strictly increasing"
        )
    if panel_id == "velocity_profiles":
        station_key = (
            station_id.removeprefix("autocfd5_").upper()
            if family_id == "drivaerml-autocfd5-constant-v1"
            else station_id
        )
        full_grid_count = _VELOCITY_SAMPLE_COUNTS[station_key]
        expected_coordinate = (
            ("distance_m", "m")
            if family_id == "drivaerml-autocfd5-constant-v1"
            else ("normalized_arc_length", "1")
        )
        if (series["coordinate_id"], series["coordinate_unit"]) != expected_coordinate:
            raise DrivAerDatasetScorerError(
                f"{label} coordinate identity must be {expected_coordinate}"
            )
        if family_id == "drivaerml-velocity-relative-v3":
            if any(value < 0.0 or value > 1.0 for value in normalized_coordinates):
                raise DrivAerDatasetScorerError(
                    f"{label} normalized arc coordinates must lie in [0, 1]"
                )
        else:
            interval_end = _CONSTANT_VELOCITY_INTERVAL_ENDS[station_key]
            denominator = full_grid_count - 1
            grid_indices: list[int] = []
            for value in normalized_coordinates:
                nearest_index = round(value * denominator / interval_end)
                if (
                    nearest_index < 0
                    or nearest_index >= full_grid_count
                    or not math.isclose(
                        value,
                        interval_end * nearest_index / denominator,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise DrivAerDatasetScorerError(
                        f"{label} must preserve the frozen constant coordinate grid"
                    )
                grid_indices.append(nearest_index)
            if any(
                right <= left
                for left, right in zip(grid_indices, grid_indices[1:])
            ):
                raise DrivAerDatasetScorerError(
                    f"{label} frozen constant coordinate-grid indices must be unique "
                    "and increasing"
                )
    elif (series["coordinate_id"], series["coordinate_unit"]) != ("arc_length_m", "m"):
        raise DrivAerDatasetScorerError(
            f"{label} Cp coordinate identity must be ('arc_length_m', 'm')"
        )
    normalized["coordinate"] = normalized_coordinates
    normalized["prediction"] = normalized_predictions
    return key, normalized


def validate_schema_v3_relative_profile_chunk_candidate(
    document: Mapping[str, object],
    *,
    expected_contract_sha256: str = RELATIVE_PROFILE_CONTRACT_SHA256,
) -> dict[str, object]:
    """Validate one complete DrivAerML constant+relative candidate chunk."""

    root = _exact_keys(
        document,
        {"schema_version", "contract_id", "contract_sha256", "cases"},
        "relative profile chunk",
    )
    if root["schema_version"] != RELATIVE_PROFILE_SCHEMA_VERSION:
        raise DrivAerDatasetScorerError("relative profile chunk schema_version is invalid")
    if root["contract_id"] != RELATIVE_PROFILE_CONTRACT_ID:
        raise DrivAerDatasetScorerError("relative profile chunk contract_id is invalid")
    if (
        not isinstance(expected_contract_sha256, str)
        or _SHA256_RE.fullmatch(expected_contract_sha256) is None
        or root["contract_sha256"] != expected_contract_sha256
    ):
        raise DrivAerDatasetScorerError(
            "relative profile chunk contract_sha256 does not match the retained contract"
        )
    raw_cases = root["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DrivAerDatasetScorerError("relative profile chunk cases must be non-empty")
    expected_keys = set(_relative_profile_expected_keys())
    retained_support = _relative_series_support_index()
    retained_constant_support = _constant_series_support_index()
    normalized_cases: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    for case_position, raw_case in enumerate(raw_cases):
        case = _exact_keys(
            raw_case,
            {"case_id", "series"},
            f"relative profile case {case_position}",
        )
        case_id = _case_id(case["case_id"], f"relative profile case {case_position} ID")
        if case_id in seen_cases:
            raise DrivAerDatasetScorerError(
                f"relative profile chunk contains duplicate case {case_id}"
            )
        seen_cases.add(case_id)
        raw_series = case["series"]
        if not isinstance(raw_series, list) or len(raw_series) != RELATIVE_PROFILE_SERIES_PER_CASE:
            raise DrivAerDatasetScorerError(
                f"{case_id} must provide exactly {RELATIVE_PROFILE_SERIES_PER_CASE} namespaced series"
            )
        normalized_series: list[dict[str, object]] = []
        observed_keys: list[tuple[str, str, str, str, str]] = []
        canonical_supports: dict[str, str] = {}
        aliases: list[tuple[str, Mapping[str, Any]]] = []
        for series_position, raw_item in enumerate(raw_series):
            key, normalized = _normalize_relative_profile_series(
                raw_item,
                case_id=case_id,
                position=series_position,
            )
            observed_keys.append(key)
            normalized_series.append(normalized)
            panel_id, family_id, station_id, _quantity_id, representation = key
            if family_id in {
                "drivaerml-velocity-relative-v3",
                "drivaerml_cp_relative_v1",
            }:
                expected_identity = retained_support.get(
                    (case_id, family_id, station_id)
                )
                if expected_identity is None:
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} is absent from the "
                        "retained relative support index"
                    )
                if expected_identity["representation"] != representation:
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} representation differs "
                        "from the retained relative support index"
                    )
                if (
                    normalized["placement_receipt_identity_sha256"]
                    != expected_identity["placement_receipt_identity_sha256"]
                ):
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} placement receipt "
                        "identity differs from the retained relative support index"
                    )
                submitted_support = (
                    normalized["shared_support_ref"][
                        "canonical_support_identity_sha256"
                    ]
                    if representation == "shared_alias"
                    else normalized["support_identity_sha256"]
                )
                if submitted_support != expected_identity["support_identity_sha256"]:
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} support identity "
                        "differs from the retained relative support index"
                    )
                if representation == "materialized":
                    submitted_coordinates = normalized["coordinate"]
                    expected_coordinate_count = expected_identity[
                        "coordinate_count"
                    ]
                    if len(submitted_coordinates) != expected_coordinate_count:
                        raise DrivAerDatasetScorerError(
                            f"{case_id}/{family_id}/{station_id} coordinate count "
                            "differs from the retained relative support index"
                        )
                    try:
                        submitted_coordinate_identity = (
                            coordinate_array_identity_sha256(
                                submitted_coordinates
                            )
                        )
                    except CoordinateIdentityError as error:
                        raise DrivAerDatasetScorerError(
                            f"{case_id}/{family_id}/{station_id} coordinate array "
                            f"cannot be canonically encoded: {error}"
                        ) from error
                    if (
                        submitted_coordinate_identity
                        != expected_identity["coordinate_identity_sha256"]
                    ):
                        raise DrivAerDatasetScorerError(
                            f"{case_id}/{family_id}/{station_id} coordinate identity "
                            "differs from the retained relative support index"
                        )
            if representation == "materialized" and family_id in {
                "drivaerml-autocfd5-constant-v1",
                "drivaerml_cp_constant_v1",
            }:
                expected_constant = retained_constant_support.get(
                    (case_id, family_id, station_id)
                )
                if expected_constant is None:
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} is absent from the "
                        "retained all-case constant support index"
                    )
                if (
                    normalized["placement_receipt_identity_sha256"]
                    != expected_constant["placement_receipt_identity_sha256"]
                ):
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} placement receipt "
                        "identity differs from the retained all-case constant "
                        "support index"
                    )
                if (
                    normalized["support_identity_sha256"]
                    != expected_constant["support_identity_sha256"]
                ):
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} support identity "
                        "differs from the retained all-case constant support index"
                    )
                submitted_constant_coordinates = normalized["coordinate"]
                if (
                    len(submitted_constant_coordinates)
                    != expected_constant["coordinate_count"]
                ):
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} coordinate count "
                        "differs from the retained all-case constant support index"
                    )
                try:
                    submitted_constant_coordinate_identity = (
                        coordinate_array_identity_sha256(
                            submitted_constant_coordinates
                        )
                    )
                except CoordinateIdentityError as error:
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} coordinate array "
                        f"cannot be canonically encoded: {error}"
                    ) from error
                if (
                    submitted_constant_coordinate_identity
                    != expected_constant["coordinate_identity_sha256"]
                ):
                    raise DrivAerDatasetScorerError(
                        f"{case_id}/{family_id}/{station_id} coordinate identity "
                        "differs from the retained all-case constant support index"
                    )
            if (
                panel_id == "pressure_profiles"
                and family_id == "drivaerml_cp_constant_v1"
                and station_id in _SHARED_CP_SUPPORT_IDS
            ):
                canonical_supports[station_id] = str(
                    normalized["support_identity_sha256"]
                )
            if representation == "shared_alias":
                aliases.append((station_id, normalized["shared_support_ref"]))
        if len(observed_keys) != len(set(observed_keys)):
            raise DrivAerDatasetScorerError(
                f"{case_id} contains duplicate namespaced profile series"
            )
        missing = sorted(expected_keys - set(observed_keys))
        unexpected = sorted(set(observed_keys) - expected_keys)
        if missing or unexpected:
            raise DrivAerDatasetScorerError(
                f"{case_id} relative profile coverage differs from the closed contract "
                f"(missing={missing}, unexpected={unexpected})"
            )
        for station_id, reference in aliases:
            if reference["canonical_support_identity_sha256"] != canonical_supports.get(station_id):
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{station_id} shared alias identity differs from its canonical constant Cp support"
                )
        normalized_cases.append({"case_id": case_id, "series": normalized_series})
    result: dict[str, object] = {
        "schema_version": RELATIVE_PROFILE_SCHEMA_VERSION,
        "contract_id": RELATIVE_PROFILE_CONTRACT_ID,
        "contract_sha256": expected_contract_sha256,
        "cases": normalized_cases,
    }
    _assert_no_absolute_paths(result)
    return result


def schema_v3_relative_profile_chunks_candidate_adapter(
    evaluation: CandidateDatasetEvaluation,
    *,
    submission_id: str,
    namespaced_series_by_case: Mapping[str, Sequence[Mapping[str, object]]],
    contract_sha256: str = RELATIVE_PROFILE_CONTRACT_SHA256,
    cases_per_chunk: int = 16,
) -> CandidateProfileChunks:
    """Package evaluator-derived constant and report-only relative diagnostics.

    Geometry placement and field sampling remain evaluator responsibilities.
    This boundary accepts only the complete, namespaced 40-series result for
    every selected case and therefore cannot fabricate or subset-reduce a
    missing relative family.
    """

    if not isinstance(evaluation, CandidateDatasetEvaluation):
        raise DrivAerDatasetScorerError("evaluation must be CandidateDatasetEvaluation")
    if evaluation.to_json().get("prediction_scope") == PREDICTION_SCOPE_SURFACE_ONLY:
        raise DrivAerDatasetScorerError(
            "surface-only submissions have no velocity profiles to package"
        )
    if not isinstance(submission_id, str) or _SUBMISSION_ID_RE.fullmatch(submission_id) is None:
        raise DrivAerDatasetScorerError("submission_id is not schema-v3 compatible")
    if not isinstance(namespaced_series_by_case, Mapping):
        raise DrivAerDatasetScorerError("namespaced_series_by_case must be an object")
    if any(not isinstance(case_id, str) for case_id in namespaced_series_by_case):
        raise DrivAerDatasetScorerError(
            "namespaced_series_by_case keys must be case-ID strings"
        )
    if contract_sha256 != RELATIVE_PROFILE_CONTRACT_SHA256:
        raise DrivAerDatasetScorerError(
            "relative profile contract SHA-256 is not the retained v3 contract"
        )
    chunk_size = _integer(cases_per_chunk, "cases_per_chunk", minimum=1)
    evidence = evaluation.to_json()
    split = _mapping(evidence.get("split"), "candidate dataset split")
    raw_case_ids = split.get("case_ids")
    if not isinstance(raw_case_ids, list) or not raw_case_ids:
        raise DrivAerDatasetScorerError("candidate split case IDs are malformed")
    case_ids = tuple(_case_id(value, "candidate split case ID") for value in raw_case_ids)
    if set(namespaced_series_by_case) != set(case_ids):
        missing = sorted(set(case_ids) - set(namespaced_series_by_case))
        unexpected = sorted(set(namespaced_series_by_case) - set(case_ids))
        raise DrivAerDatasetScorerError(
            "namespaced relative profile cases differ from the selected split "
            f"(missing={missing}, unexpected={unexpected})"
        )
    packaged_cases = [
        {
            "case_id": case_id,
            "series": list(namespaced_series_by_case[case_id]),
        }
        for case_id in case_ids
    ]
    chunk_count = math.ceil(len(packaged_cases) / chunk_size)
    if chunk_count > 1000:
        raise DrivAerDatasetScorerError(
            "cases_per_chunk would exceed the three-digit profile chunk namespace"
        )
    chunks: list[tuple[str, Mapping[str, object]]] = []
    chunk_index: list[dict[str, object]] = []
    for chunk_number, start in enumerate(range(0, len(packaged_cases), chunk_size)):
        filename = f"chunk-{chunk_number:03d}.json"
        document = validate_schema_v3_relative_profile_chunk_candidate(
            {
                "schema_version": RELATIVE_PROFILE_SCHEMA_VERSION,
                "contract_id": RELATIVE_PROFILE_CONTRACT_ID,
                "contract_sha256": contract_sha256,
                "cases": packaged_cases[start : start + chunk_size],
            },
            expected_contract_sha256=contract_sha256,
        )
        digest = hashlib.sha256(_canonical_json_payload(document)).hexdigest()
        chunks.append((filename, document))
        chunk_index.append(
            {
                "file": filename,
                "case_ids": [case["case_id"] for case in document["cases"]],
                "sha256": digest,
            }
        )
    index: dict[str, object] = {
        "schema_version": "1.0",
        "format": RELATIVE_PROFILE_FORMAT,
        "contract_id": RELATIVE_PROFILE_CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": split["split_id"],
        "case_set_id": split["case_set_id"],
        "case_count": len(packaged_cases),
        "case_id_status": "official",
        "chunks": chunk_index,
    }
    _assert_no_absolute_paths(index)
    return CandidateProfileChunks(index=index, chunks=tuple(chunks))


def schema_v3_profile_chunks_candidate_adapter(
    evaluation: CandidateDatasetEvaluation,
    *,
    submission_id: str,
    cases_per_chunk: int = 16,
) -> CandidateProfileChunks:
    """Project complete evaluator series into FluidsBench profile chunks.

    Every selected full-field case must contain all sixteen velocity lines and
    all four continuous Cp cuts in the evaluator's canonical order. A
    surface-only case contains the four Cp cuts only: velocity is explicitly
    unavailable and is never replaced by dummy series. In either scope,
    unavailable Cp support raises instead of fabricating display values.
    """

    if not isinstance(evaluation, CandidateDatasetEvaluation):
        raise DrivAerDatasetScorerError(
            "evaluation must be CandidateDatasetEvaluation"
        )
    if (
        not isinstance(submission_id, str)
        or _SUBMISSION_ID_RE.fullmatch(submission_id) is None
    ):
        raise DrivAerDatasetScorerError(
            "submission_id is not schema-v3 compatible"
        )
    chunk_size = _integer(cases_per_chunk, "cases_per_chunk", minimum=1)
    evidence = evaluation.to_json()
    prediction_scope = evidence.get("prediction_scope")
    if prediction_scope not in PREDICTION_SCOPES:
        raise DrivAerDatasetScorerError("candidate dataset prediction scope is invalid")
    split = _mapping(evidence.get("split"), "candidate dataset split")
    source_contract = _mapping(
        evidence.get("source_contract"), "candidate source contract"
    )
    series_contract = _exact_keys(
        source_contract.get("diagnostic_profile_series_contract"),
        {
            "pressure_station_ids",
            "velocity_station_ids",
            "velocity_station_sample_counts",
            "canonical_order",
        },
        "candidate profile-series contract",
    )
    if series_contract["canonical_order"] != "pressure_profiles_then_velocity_profiles":
        raise DrivAerDatasetScorerError(
            "candidate profile-series contract has an unsupported order"
        )

    def safe_id_array(value: object, label: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise DrivAerDatasetScorerError(f"{label} must be a non-empty array")
        result = tuple(_string(item, label) for item in value)
        if len(result) != len(set(result)):
            raise DrivAerDatasetScorerError(f"{label} contains duplicate IDs")
        return result

    pressure_ids = safe_id_array(
        series_contract["pressure_station_ids"], "pressure station IDs"
    )
    velocity_ids = safe_id_array(
        series_contract["velocity_station_ids"], "velocity station IDs"
    )
    raw_velocity_counts = series_contract["velocity_station_sample_counts"]
    if not isinstance(raw_velocity_counts, list):
        raise DrivAerDatasetScorerError(
            "velocity station sample counts must be an array"
        )
    velocity_counts = tuple(
        _integer(item, "velocity station sample count", minimum=2)
        for item in raw_velocity_counts
    )
    if (
        len(pressure_ids) != 4
        or len(velocity_ids) != 16
        or len(velocity_counts) != len(velocity_ids)
    ):
        raise DrivAerDatasetScorerError(
            "candidate profile-series contract must define four Cp cuts and sixteen velocity lines"
        )
    expected_keys = tuple(
        ("pressure_profiles", station_id, "cp") for station_id in pressure_ids
    ) + (
        tuple(
            ("velocity_profiles", station_id, "velocity_ratio")
            for station_id in velocity_ids
        )
        if prediction_scope == PREDICTION_SCOPE_FULL
        else ()
    )
    expected_counts = {
        ("velocity_profiles", station_id, "velocity_ratio"): sample_count
        for station_id, sample_count in zip(
            velocity_ids, velocity_counts, strict=True
        )
    }

    raw_cases = evidence.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DrivAerDatasetScorerError(
            "candidate dataset evidence cases are malformed"
        )
    split_case_ids = split.get("case_ids")
    if not isinstance(split_case_ids, list):
        raise DrivAerDatasetScorerError("candidate split case IDs are malformed")
    packaged_cases: list[dict[str, object]] = []
    for position, raw_case in enumerate(raw_cases):
        case = _mapping(raw_case, f"candidate dataset case {position}")
        case_id = _case_id(case.get("case_id"), f"candidate case {position} ID")
        if position >= len(split_case_ids) or case_id != split_case_ids[position]:
            raise DrivAerDatasetScorerError(
                "candidate dataset cases differ from the selected split order"
            )
        diagnostics = _mapping(
            case.get("diagnostic_metric_values"),
            f"candidate case {case_id} diagnostic values",
        )
        required_diagnostics = ["cp_cut_rmse"]
        if prediction_scope == PREDICTION_SCOPE_FULL:
            required_diagnostics.insert(0, "velocity_profile_uinf_rmse")
        unavailable = [
            metric_id
            for metric_id in required_diagnostics
            if diagnostics.get(metric_id) is None
        ]
        if unavailable:
            raise DrivAerDatasetScorerError(
                f"cannot package {case_id} profiles because required diagnostics "
                f"are unavailable: {unavailable}"
            )
        for metric_id in required_diagnostics:
            _finite(
                diagnostics[metric_id],
                f"{case_id}/{metric_id}",
                nonnegative=True,
            )
        raw_series = case.get("profile_series")
        if not isinstance(raw_series, list):
            raise DrivAerDatasetScorerError(
                f"candidate case {case_id} profile_series must be an array"
            )
        if len(raw_series) != len(expected_keys):
            raise DrivAerDatasetScorerError(
                f"candidate case {case_id} must provide the complete profile series for its prediction scope"
            )
        normalized_series: list[dict[str, object]] = []
        observed_keys: list[tuple[str, str, str]] = []
        for series_position, raw_series_item in enumerate(raw_series):
            series = _exact_keys(
                raw_series_item,
                {
                    "panel_id",
                    "station_id",
                    "quantity_id",
                    "coordinate",
                    "prediction",
                },
                f"{case_id} profile series {series_position}",
            )
            key = (
                _string(series["panel_id"], f"{case_id} panel ID"),
                _string(series["station_id"], f"{case_id} station ID"),
                _string(series["quantity_id"], f"{case_id} quantity ID"),
            )
            observed_keys.append(key)
            coordinates = series["coordinate"]
            predictions = series["prediction"]
            if (
                not isinstance(coordinates, list)
                or not isinstance(predictions, list)
                or len(coordinates) < 2
                or len(coordinates) != len(predictions)
            ):
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{key[1]} profile arrays must have the same length >= 2"
                )
            normalized_coordinates = [
                _finite(item, f"{case_id}/{key[1]} coordinate {index}")
                for index, item in enumerate(coordinates)
            ]
            normalized_predictions = [
                _finite(item, f"{case_id}/{key[1]} prediction {index}")
                for index, item in enumerate(predictions)
            ]
            if any(
                right <= left
                for left, right in zip(
                    normalized_coordinates, normalized_coordinates[1:]
                )
            ):
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{key[1]} profile coordinates must be strictly increasing"
                )
            expected_count = expected_counts.get(key)
            if expected_count is not None and len(normalized_coordinates) != expected_count:
                raise DrivAerDatasetScorerError(
                    f"{case_id}/{key[1]} profile series must contain exactly {expected_count} samples"
                )
            normalized_series.append(
                {
                    "panel_id": key[0],
                    "station_id": key[1],
                    "quantity_id": key[2],
                    "coordinate": normalized_coordinates,
                    "prediction": normalized_predictions,
                }
            )
        if tuple(observed_keys) != expected_keys:
            raise DrivAerDatasetScorerError(
                f"candidate case {case_id} profile series are incomplete, duplicated, or out of canonical order"
            )
        packaged_cases.append({"case_id": case_id, "series": normalized_series})
    if len(packaged_cases) != len(split_case_ids) or split.get("case_count") != len(
        packaged_cases
    ):
        raise DrivAerDatasetScorerError(
            "candidate profile package does not cover the complete selected split"
        )

    chunk_count = math.ceil(len(packaged_cases) / chunk_size)
    if chunk_count > 1000:
        raise DrivAerDatasetScorerError(
            "cases_per_chunk would exceed the three-digit profile chunk namespace"
        )
    chunks: list[tuple[str, Mapping[str, object]]] = []
    chunk_index: list[dict[str, object]] = []
    for chunk_number, start in enumerate(range(0, len(packaged_cases), chunk_size)):
        filename = f"chunk-{chunk_number:03d}.json"
        selected = packaged_cases[start : start + chunk_size]
        document: dict[str, object] = {
            "schema_version": "1.0",
            "cases": selected,
        }
        digest = hashlib.sha256(_canonical_json_payload(document)).hexdigest()
        chunks.append((filename, document))
        chunk_index.append(
            {
                "file": filename,
                "case_ids": [case["case_id"] for case in selected],
                "sha256": digest,
            }
        )
    index: dict[str, object] = {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": "drivaerml",
        "split_id": split["split_id"],
        "case_set_id": split["case_set_id"],
        "case_count": len(packaged_cases),
        "case_id_status": "official",
        "chunks": chunk_index,
    }
    _assert_no_absolute_paths(index)
    for _, document in chunks:
        _assert_no_absolute_paths(document)
    return CandidateProfileChunks(index=index, chunks=tuple(chunks))


def validate_schema_v3_candidate_nonspatial_metrics(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate DrivAerML schema-v3 nonspatial values and their reductions.

    Generic schema-v3 validation can verify spatial sufficient statistics but
    cannot infer dataset-specific force and AutoCFD5 reductions.  This helper
    makes that projection fail closed: every force metric must be present for
    every case, each diagnostic must be present for either every case or no
    case, no undeclared nonspatial metric is accepted, and every dataset value
    is recomputed from the per-case values.

    The returned digest covers the ordered case IDs, their exact nonspatial
    values, and the participant-reported force coefficients.  It can be used
    to bind an optional maintainer audit, but native-field sharing and
    maintainer recomputation are not required for a normal submission.
    """

    root = _mapping(document, "schema-v3 case metrics")
    if root.get("dataset_id") != "drivaerml":
        raise DrivAerDatasetScorerError(
            "schema-v3 nonspatial validation only accepts dataset_id='drivaerml'"
        )
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DrivAerDatasetScorerError(
            "schema-v3 DrivAerML cases must be a non-empty array"
        )
    if root.get("case_count") != len(raw_cases):
        raise DrivAerDatasetScorerError(
            "schema-v3 DrivAerML case_count differs from cases"
        )
    aggregate = _mapping(root.get("metric_values"), "schema-v3 metric_values")
    known = set(NONSPATIAL_METRIC_IDS)
    required_force = set((*RANKED_FORCE_METRIC_IDS, *REPORT_ONLY_FORCE_METRIC_IDS))
    per_metric: dict[str, list[float]] = {
        metric_id: [] for metric_id in NONSPATIAL_METRIC_IDS
    }
    canonical_cases: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _mapping(raw_case, f"schema-v3 cases[{index}]")
        case_id = _case_id(case.get("case_id"), f"schema-v3 cases[{index}].case_id")
        if case_id in seen_case_ids:
            raise DrivAerDatasetScorerError(
                f"schema-v3 DrivAerML cases contain duplicate {case_id}"
            )
        seen_case_ids.add(case_id)
        values = _mapping(
            case.get("nonspatial_metric_values"),
            f"schema-v3 {case_id} nonspatial_metric_values",
        )
        force_coefficients = _exact_keys(
            case.get("force_coefficients"),
            SCHEMA_V3_FORCE_COEFFICIENT_KEYS,
            f"schema-v3 {case_id} force_coefficients",
        )
        canonical_force = {
            key: _finite(
                force_coefficients[key],
                f"schema-v3 {case_id} force_coefficients.{key}",
            )
            for key in SCHEMA_V3_FORCE_COEFFICIENT_KEYS
        }
        _require_same_float(
            canonical_force["clf"],
            canonical_force["cl"] / 2.0 + canonical_force["cm_pitch"],
            f"schema-v3 {case_id} force_coefficients.clf",
        )
        _require_same_float(
            canonical_force["clr"],
            canonical_force["cl"] / 2.0 - canonical_force["cm_pitch"],
            f"schema-v3 {case_id} force_coefficients.clr",
        )
        observed = set(values)
        if not required_force.issubset(observed):
            raise DrivAerDatasetScorerError(
                f"schema-v3 {case_id} nonspatial metrics are missing force IDs: "
                f"{sorted(required_force - observed)}"
            )
        if not observed.issubset(known):
            raise DrivAerDatasetScorerError(
                f"schema-v3 {case_id} contains undeclared DrivAerML nonspatial "
                f"metrics: {sorted(observed - known)}"
            )
        canonical_values: dict[str, float] = {}
        for metric_id in NONSPATIAL_METRIC_IDS:
            if metric_id not in values:
                continue
            value = _finite(
                values[metric_id],
                f"schema-v3 {case_id} nonspatial_metric_values.{metric_id}",
                nonnegative=True,
            )
            per_metric[metric_id].append(value)
            canonical_values[metric_id] = value
        canonical_cases.append(
            {
                "case_id": case_id,
                "force_coefficients": canonical_force,
                "nonspatial_metric_values": canonical_values,
            }
        )

        closure_value = canonical_values.get(
            "field_integrated_lift_closure_max_abs"
        )
        if closure_value is not None:
            expected_closure = abs(
                canonical_force["cl"]
                - canonical_force["clf"]
                - canonical_force["clr"]
            )
            if not _same_float(closure_value, expected_closure):
                raise DrivAerDatasetScorerError(
                    f"schema-v3 {case_id} lift-closure metric differs from "
                    "the explicit force coefficients"
                )

    case_count = len(raw_cases)
    for metric_id in DIAGNOSTIC_METRIC_IDS:
        count = len(per_metric[metric_id])
        if count not in {0, case_count}:
            raise DrivAerDatasetScorerError(
                f"schema-v3 diagnostic {metric_id!r} must be present for every "
                "case or omitted for every case"
            )
    present_ids = tuple(
        metric_id for metric_id in NONSPATIAL_METRIC_IDS if per_metric[metric_id]
    )
    aggregate_ids = set(aggregate) & known
    if aggregate_ids != set(present_ids):
        raise DrivAerDatasetScorerError(
            "schema-v3 aggregate DrivAerML nonspatial metric IDs differ from "
            f"the complete per-case values (missing={sorted(set(present_ids) - aggregate_ids)}, "
            f"unexpected={sorted(aggregate_ids - set(present_ids))})"
        )

    expected_values: dict[str, float] = {}
    for metric_id in (*RANKED_FORCE_METRIC_IDS, "field_integrated_clf_rmse", "field_integrated_clr_rmse"):
        values = per_metric[metric_id]
        expected_values[metric_id] = math.sqrt(
            math.fsum(value * value for value in values) / case_count
        )
    expected_values["field_integrated_lift_closure_max_abs"] = max(
        per_metric["field_integrated_lift_closure_max_abs"]
    )
    for metric_id in DIAGNOSTIC_METRIC_IDS:
        values = per_metric[metric_id]
        if values:
            expected_values[metric_id] = math.fsum(values) / case_count
    for metric_id in present_ids:
        submitted = _finite(
            aggregate[metric_id],
            f"schema-v3 aggregate metric_values.{metric_id}",
            nonnegative=True,
        )
        if not _same_float(submitted, expected_values[metric_id]):
            raise DrivAerDatasetScorerError(
                f"schema-v3 aggregate metric_values.{metric_id} differs from "
                "the DrivAerML per-case reduction"
            )

    payload = {
        "schema": "drivaerml-schema-v3-nonspatial-values-v1",
        "metric_ids": list(present_ids),
        "cases": canonical_cases,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "case_count": case_count,
        "metric_ids": list(present_ids),
        "force_coefficient_keys": list(SCHEMA_V3_FORCE_COEFFICIENT_KEYS),
        "nonspatial_values_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _assert_no_absolute_paths(value: object, context: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_paths(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, f"{context}[{index}]")
    elif isinstance(value, str):
        if value.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
            raise DrivAerDatasetScorerError(
                f"candidate evidence contains an absolute path at {context}"
            )


def write_candidate_dataset_evidence(
    evaluation: CandidateDatasetEvaluation,
    path: Path | str,
) -> dict[str, object]:
    """Atomically write deterministic candidate dataset evidence."""

    if not isinstance(evaluation, CandidateDatasetEvaluation):
        raise DrivAerDatasetScorerError("evaluation must be CandidateDatasetEvaluation")
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise DrivAerDatasetScorerError("dataset evidence output must use .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            evaluation.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "status": CANDIDATE_DATASET_STATUS,
        "official_submission": False,
    }


def write_regional_diagnostics(
    evaluation: CandidateDatasetEvaluation,
    path: Path | str,
) -> dict[str, object]:
    """Atomically write the compact zero-weight split-level regional report."""

    if not isinstance(evaluation, CandidateDatasetEvaluation):
        raise DrivAerDatasetScorerError(
            "evaluation must be CandidateDatasetEvaluation"
        )
    if evaluation.regional_diagnostics is None:
        raise DrivAerDatasetScorerError(
            "candidate evaluation does not contain regional diagnostics"
        )
    destination = Path(path)
    if destination.name != "regional-diagnostics.json":
        raise DrivAerDatasetScorerError(
            "regional diagnostics output must be named regional-diagnostics.json"
        )
    payload = _canonical_json_payload(evaluation.regional_diagnostics)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "role": "report_only",
        "weight": 0.0,
        "official_score_changed": False,
    }


def write_schema_v3_case_metrics_candidate(
    document: Mapping[str, object], path: Path | str
) -> dict[str, object]:
    """Write a deterministic candidate schema-v3 case-metrics adapter."""

    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise DrivAerDatasetScorerError("case-metrics output must use .json")
    payload = (
        json.dumps(
            dict(document), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "candidate_only": True,
        "official_submission": False,
    }


def write_schema_v3_profile_chunks_candidate(
    package: CandidateProfileChunks, directory: Path | str
) -> dict[str, object]:
    """Atomically publish one candidate profile index and its chunks.

    The destination must not already exist.  Refusing replacement avoids stale
    or unindexed chunk files and prevents a failed write from damaging an
    existing participant package.
    """

    if not isinstance(package, CandidateProfileChunks):
        raise DrivAerDatasetScorerError(
            "package must be CandidateProfileChunks"
        )
    relative_format = package.index.get("format") == RELATIVE_PROFILE_FORMAT
    index_fields = {
        "schema_version",
        "submission_id",
        "dataset_id",
        "split_id",
        "case_set_id",
        "case_count",
        "case_id_status",
        "chunks",
    }
    if relative_format:
        index_fields |= {"format", "contract_id", "contract_sha256"}
    index = _exact_keys(package.index, index_fields, "candidate profile index")
    raw_index_chunks = index["chunks"]
    if (
        index["schema_version"] != "1.0"
        or index["dataset_id"] != "drivaerml"
        or index["case_id_status"] != "official"
        or not isinstance(raw_index_chunks, list)
        or len(raw_index_chunks) != len(package.chunks)
    ):
        raise DrivAerDatasetScorerError(
            "candidate profile package index is inconsistent"
        )
    if relative_format and (
        index["contract_id"] != RELATIVE_PROFILE_CONTRACT_ID
        or index["contract_sha256"] != RELATIVE_PROFILE_CONTRACT_SHA256
    ):
        raise DrivAerDatasetScorerError(
            "relative candidate profile index does not bind the retained v3 contract"
        )
    series_count = 0
    for position, ((filename, document), raw_entry) in enumerate(
        zip(package.chunks, raw_index_chunks, strict=True)
    ):
        entry = _exact_keys(
            raw_entry,
            {"file", "case_ids", "sha256"},
            f"candidate profile index chunk {position}",
        )
        if relative_format:
            chunk = validate_schema_v3_relative_profile_chunk_candidate(document)
        else:
            chunk = _exact_keys(
                document,
                {"schema_version", "cases"},
                f"candidate profile chunk {position}",
            )
        cases = chunk["cases"]
        if not isinstance(cases, list) or not cases:
            raise DrivAerDatasetScorerError(
                f"candidate profile chunk {position} cases are malformed"
            )
        case_ids = [
            _case_id(
                _mapping(case, f"candidate profile chunk {position} case").get(
                    "case_id"
                ),
                f"candidate profile chunk {position} case ID",
            )
            for case in cases
        ]
        digest = hashlib.sha256(_canonical_json_payload(document)).hexdigest()
        if (
            (
                chunk["schema_version"] != RELATIVE_PROFILE_SCHEMA_VERSION
                if relative_format
                else chunk["schema_version"] != "1.0"
            )
            or entry["file"] != filename
            or entry["case_ids"] != case_ids
            or entry["sha256"] != digest
        ):
            raise DrivAerDatasetScorerError(
                f"candidate profile chunk {position} differs from its index binding"
            )
        series_count += sum(
            len(_mapping(case, "candidate profile case").get("series", []))
            for case in cases
        )
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise DrivAerDatasetScorerError(
            "profile output directory must not already exist"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
    )
    try:
        chunk_receipts: list[dict[str, object]] = []
        for filename, document in package.chunks:
            if re.fullmatch(r"chunk-[0-9]{3}\.json", filename) is None:
                raise DrivAerDatasetScorerError(
                    f"invalid candidate profile chunk filename {filename!r}"
                )
            payload = _canonical_json_payload(document)
            (temporary / filename).write_bytes(payload)
            chunk_receipts.append(
                {
                    "file": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        index_payload = _canonical_json_payload(package.index)
        (temporary / "index.json").write_bytes(index_payload)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "index_file": "index.json",
        "index_sha256": hashlib.sha256(index_payload).hexdigest(),
        "chunk_count": len(package.chunks),
        "case_count": package.index["case_count"],
        "series_count": series_count,
        "chunks": chunk_receipts,
        "candidate_only": True,
        "official_submission": False,
    }
