"""Candidate one-case AutoCFD5 diagnostic evaluator for DrivAerML.

This module is intentionally separate from :mod:`reference.drivaerml.evaluator`.
It joins strict candidate mapping evidence to complete native-order prediction
chunk manifests, retaining only the mapped raw-cell values after every chunk
has been verified.  It can also select sparse native volume truth while the
complete inline-binary ``UMeanTrim`` payload is decoded and hashed.

The outputs remain candidate evidence.  They do not activate scoring support,
claim owner approval, establish profile-resolution convergence, or represent
an official submission.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np

from .autocfd5 import (
    CP_PANEL_COUNT,
    CP_PANEL_MEMBERSHIP_COUNT,
    CP_PROBE_COUNT,
    NATIVE_BRIDGE_MAX_DISTANCE_M,
    NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT,
    NOMINAL_PROBE_DISPLACEMENT_MAX_M,
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    U_INF_M_PER_S,
    VELOCITY_LINE_COUNT,
    VELOCITY_SAMPLE_COUNT,
    AutoCFD5Definition,
    AutoCFD5Error,
    CpProbeMappingEvidence,
    VelocityCellAssignmentEvidence,
    cp_from_kinematic_pressure,
    cp_probe_rmse,
    load_autocfd5_definition,
    validate_cp_mapping_evidence,
    velocity_magnitude_ratio,
    velocity_profile_rmse,
)
from .cp_mapping import (
    CUT_VERTEX_PLANE_TOLERANCE_M,
    NATIVE_BRIDGE_DISTANCE_REVIEW_M,
    NATIVE_DISTANCE_TIE_M,
    NATIVE_NORMAL_TIE,
    NOMINAL_PROBE_DISPLACEMENT_REVIEW_M,
    REQUIRED_VTK_VERSION,
    STL_DISTANCE_TIE_M,
    UNNORMALIZED_AREA_VECTOR_MIN_M2,
)
from .prediction_chunks import (
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    PredictionChunkError,
    PredictionChunkManifest,
    iter_prediction_chunks,
    load_prediction_chunk_manifest,
)
from .retained_file import RetainedFileError, RetainedVerifiedFile
from .source import (
    InlineBinaryDecodeError,
    InlineBinaryPayloadSummary,
    VTKDataArrayIndex,
    VTKXMLIndex,
    stream_inline_binary_payload,
)
from .velocity_assignments import (
    CELL_EVALUATION_FAILURE_REASON_PREFIX,
    KERNEL_ID,
    NO_CLOSURE_CELL_REASON,
    OWNER_INVALID_REASONS,
    POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES,
    POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES,
    POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE,
    QUERY_CACHE_KEY_ID,
    assignment_evidence_sha256,
    candidate_kernel_settings,
)


CANDIDATE_SCHEMA = "drivaerml-autocfd5-case-diagnostics-candidate-v1"
CANDIDATE_STATUS = "candidate_diagnostics_not_active_or_official_submission"
CP_SUPPORT_SCHEMA = "drivaerml-autocfd5-cp-case-support-candidate-v1"
CP_SUPPORT_STATUS = "candidate_not_owner_approved_not_active_scoring_support"
VELOCITY_RECEIPT_SCHEMA = (
    "drivaerml-velocity-cell-assignments-case-candidate-v1"
)
VELOCITY_ARTIFACT_SCHEMA = "drivaerml-velocity-cell-mapping-candidate-v1"
VELOCITY_STATUS = "candidate_complete_geometry_mapping_not_activation_evidence"
VELOCITY_ARTIFACT_BASENAME = "velocity-cell-mapping-10mm.json"
EXPECTED_PROFILE_SHA256 = (
    "17d830087d11e83e3cba75358f33fdd827421be6698ba1624e547ae36f359184"
)
EXPECTED_REPOSITORY_ID = "neashton/drivaerml"
EXPECTED_REPOSITORY_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
PINNED_KERNEL_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.2.6",
    "vtk": "9.5.2",
    "vtk_source": "vtk version 9.5.2",
}
PINNED_KERNEL_SETTINGS_SHA256 = (
    "6cbd2b2fb56fc782fd9e9990bd43f2bad7fd040b355f128555ecf101b369fa1b"
)
EXPECTED_ROW_FIELDS = [
    "profile_id",
    "sample_index",
    "point_m",
    "distance_m",
    "valid",
    "reason",
    "raw_vtk_cell_id",
    "candidate_count",
]
FALSE_CASE_CLAIMS = {
    "resolution_convergence": False,
    "ranked_result_invariance": False,
    "model_ordering": False,
    "owner_validity_mask_complete": False,
    "owner_scientific_signoff": False,
    "scoring_contract_active": False,
    "official_submission": False,
}
OUTPUT_FALSE_CLAIMS = {
    "scoring_contract_active": False,
    "official_submission": False,
    "owner_scientific_approval": False,
    "profile_resolution_convergence": False,
    "three_real_model_ordering": False,
    "independent_participant_dry_run": False,
    "all_case_chunk_partition_invariance": False,
}

_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")
_CP_MAPPING_INVALID_REASONS = frozenset(
    {
        "declared_component_absent",
        "declared_cut_has_no_finite_nondegenerate_intersection",
        "declared_component_has_no_finite_nondegenerate_triangle",
        "nominal_displacement_exceeds_10pct_wheelbase",
        "no_native_polygon_bounds_candidate_within_2mm",
        "no_finite_nondegenerate_native_polygon",
        "native_bridge_distance_exceeds_2mm",
        "native_bridge_normal_agreement_below_cos30",
    }
)
_CP_REVIEW_FLAGS = frozenset(
    {
        "nominal_displacement_gt_7p5pct_wheelbase",
        "native_bridge_distance_gt_0p5mm",
    }
)
_ALLOWED_PRESSURE_VTK_TYPES = frozenset({"float", "double"})
_CP_CSV_SCHEMA = "drivaerml-autocfd5-cp-case-mapping-candidate-v1"
_EXPERIMENTAL_VELOCITY_PROFILE_IDS = (
    "V1",
    "V2",
    "V3",
    "V5",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "L1",
)
_CP_ROW_KEYS = frozenset(
    {
        "case_id",
        "autocfd_probe_id",
        "nominal_point_m",
        "drivaerml_component",
        "projection_mode",
        "cut_axis",
        "cut_value_m",
        "owner_review_status",
        "mapping_valid",
        "mapping_reason",
        "mapped_stl_point_m",
        "raw_stl_triangle_id",
        "stl_unit_normal",
        "nominal_displacement_m",
        "native_closest_point_m",
        "raw_vtk_polygon_id",
        "native_polygon_unit_normal",
        "bridge_distance_m",
        "bridge_abs_normal_dot",
        "component_facet_count",
        "projection_candidate_count",
        "native_bounds_candidate_count",
        "native_distance_pass_count",
        "native_normal_pass_count",
        "review_flags",
        "truth_valid",
        "truth_reason",
        "pMeanTrim_m2_per_s2",
        "truth_Cp",
        "support_valid",
    }
)
_CP_CSV_FIELDS = (
    "schema",
    "case_id",
    "autocfd_probe_id",
    "nominal_x_m",
    "nominal_y_m",
    "nominal_z_m",
    "drivaerml_component",
    "projection_mode",
    "cut_axis",
    "cut_value_m",
    "owner_review_status",
    "mapping_valid",
    "mapping_reason",
    "mapped_stl_x_m",
    "mapped_stl_y_m",
    "mapped_stl_z_m",
    "raw_stl_triangle_id",
    "stl_normal_x",
    "stl_normal_y",
    "stl_normal_z",
    "nominal_displacement_m",
    "native_closest_x_m",
    "native_closest_y_m",
    "native_closest_z_m",
    "raw_vtk_polygon_id",
    "native_normal_x",
    "native_normal_y",
    "native_normal_z",
    "bridge_distance_m",
    "bridge_abs_normal_dot",
    "component_facet_count",
    "projection_candidate_count",
    "native_bounds_candidate_count",
    "native_distance_pass_count",
    "native_normal_pass_count",
    "review_flags",
    "truth_valid",
    "truth_reason",
    "pMeanTrim_m2_per_s2",
    "truth_Cp",
    "support_valid",
)


class DrivAerDiagnosticEvaluatorError(ValueError):
    """Raised when candidate diagnostic evidence cannot be evaluated exactly."""


@dataclass(frozen=True)
class CpSupportRow:
    autocfd_probe_id: int
    mapping_valid: bool
    mapping_reason: str
    raw_vtk_polygon_id: int | None
    truth_valid: bool
    truth_reason: str
    pressure_m2_per_s2: float | None
    truth_cp: float | None
    support_valid: bool


@dataclass(frozen=True)
class StrictCpCaseSupport:
    path: Path
    sha256: str
    case_id: str
    profile_sha256: str
    boundary_file: str
    boundary_sha256: str
    boundary_polygon_count: int
    panel_memberships: tuple[tuple[str, int], ...]
    rows: tuple[CpSupportRow, ...]


@dataclass(frozen=True)
class VelocityMappingRow:
    profile_id: str
    sample_index: int
    distance_m: float
    valid: bool
    reason: str
    raw_vtk_cell_id: int | None
    candidate_count: int


@dataclass(frozen=True)
class StrictVelocityMapping:
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    case_id: str
    profile_sha256: str
    source_pin_sha256: str
    source_part_sha256: tuple[str, ...]
    native_cell_count: int
    experimental_profile_ids: tuple[str, ...]
    rows: tuple[VelocityMappingRow, ...]


@dataclass(frozen=True)
class SparseNativeField:
    """Mapped native truth values plus path-free source identity evidence."""

    case_id: str
    support_id: str
    field_name: str
    total_row_count: int
    raw_cell_ids: np.ndarray
    values: np.ndarray
    source_files: tuple[str, ...]
    source_sha256: tuple[str, ...]
    source_payload_sha256: str | None = None
    complete_source_identity_verified: bool = True

    def __post_init__(self) -> None:
        _case_id(self.case_id)
        _string(self.support_id, "support_id")
        _string(self.field_name, "field_name")
        total = _positive_integer(self.total_row_count, "total_row_count")
        ids = _raw_ids(self.raw_cell_ids, "raw_cell_ids", upper_bound=total)
        if len(ids) and not np.all(np.diff(ids) > 0):
            raise DrivAerDiagnosticEvaluatorError(
                "SparseNativeField raw_cell_ids must be sorted and unique"
            )
        values = np.asarray(self.values)
        if values.ndim not in {1, 2} or values.shape[0] != len(ids):
            raise DrivAerDiagnosticEvaluatorError(
                "SparseNativeField values must have one row per raw cell ID"
            )
        if values.dtype.kind not in {"i", "u", "f"}:
            raise DrivAerDiagnosticEvaluatorError(
                "SparseNativeField values must be numeric"
            )
        if not np.all(np.isfinite(values)):
            raise DrivAerDiagnosticEvaluatorError(
                "SparseNativeField values must be finite"
            )
        if not self.source_files or len(self.source_files) != len(self.source_sha256):
            raise DrivAerDiagnosticEvaluatorError(
                "SparseNativeField source files and hashes must be non-empty and aligned"
            )
        for name in self.source_files:
            _basename(name, "native source file")
        for digest in self.source_sha256:
            _sha256(digest, "native source SHA-256")
        if self.source_payload_sha256 is not None:
            _sha256(self.source_payload_sha256, "native payload SHA-256")
        if self.complete_source_identity_verified is not True:
            raise DrivAerDiagnosticEvaluatorError(
                "native sparse truth must come from a completely verified source identity"
            )
        ids.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "raw_cell_ids", ids)
        object.__setattr__(self, "values", values)

    def values_for(self, raw_cell_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        requested = _raw_ids(
            raw_cell_ids,
            "requested native raw_cell_ids",
            upper_bound=self.total_row_count,
            require_unique=False,
        )
        positions = np.searchsorted(self.raw_cell_ids, requested)
        if len(positions) and (
            np.any(positions >= len(self.raw_cell_ids))
            or np.any(self.raw_cell_ids[positions] != requested)
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"native truth field {self.field_name!r} omits a mapped raw cell ID"
            )
        return np.take(self.values, positions, axis=0)

    def audit_record(self) -> dict[str, object]:
        return {
            "support_id": self.support_id,
            "field_name": self.field_name,
            "total_row_count": self.total_row_count,
            "selected_unique_raw_cell_id_count": len(self.raw_cell_ids),
            "selected_values_sha256": _array_evidence_sha256(
                self.raw_cell_ids, self.values
            ),
            "source_files": list(self.source_files),
            "source_sha256": list(self.source_sha256),
            "source_payload_sha256": self.source_payload_sha256,
            "complete_source_identity_verified": True,
        }


@dataclass(frozen=True)
class SparsePredictionGather:
    case_id: str
    support_id: str
    field_name: str
    total_row_count: int
    raw_cell_ids: np.ndarray
    values: np.ndarray
    manifest_file: str
    manifest_sha256: str
    chunk_sha256: tuple[str, ...]

    def values_for(self, raw_cell_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        requested = _raw_ids(
            raw_cell_ids,
            "requested prediction raw_cell_ids",
            upper_bound=self.total_row_count,
            require_unique=False,
        )
        positions = np.searchsorted(self.raw_cell_ids, requested)
        if len(positions) and (
            np.any(positions >= len(self.raw_cell_ids))
            or np.any(self.raw_cell_ids[positions] != requested)
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"prediction field {self.field_name!r} omits a mapped raw cell ID"
            )
        return np.take(self.values, positions, axis=0)

    def audit_record(self) -> dict[str, object]:
        return {
            "support_id": self.support_id,
            "field_name": self.field_name,
            "total_row_count": self.total_row_count,
            "manifest_file": self.manifest_file,
            "manifest_sha256": self.manifest_sha256,
            "chunk_count": len(self.chunk_sha256),
            "chunk_sha256": list(self.chunk_sha256),
            "complete_gap_free_duplicate_free_coverage": True,
            "selected_unique_raw_cell_id_count": len(self.raw_cell_ids),
            "selected_values_sha256": _array_evidence_sha256(
                self.raw_cell_ids, self.values
            ),
        }


@dataclass(frozen=True)
class CandidateCaseDiagnostics:
    """Compact path-free candidate diagnostic evidence for one case."""

    evidence: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        result = dict(self.evidence)
        _assert_no_absolute_paths(result)
        return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DrivAerDiagnosticEvaluatorError(
                f"JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise DrivAerDiagnosticEvaluatorError(f"{label} cannot be a symbolic link")
    try:
        with RetainedVerifiedFile.open(unresolved, label=label) as retained:
            initial = retained.sha256(chunk_bytes=1024 * 1024)
            retained.handle.seek(0)
            value = json.load(
                retained.handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    DrivAerDiagnosticEvaluatorError(
                        f"{label} contains forbidden non-finite token {token}"
                    )
                ),
            )
            retained.assert_unchanged(context="while its JSON was parsed")
    except DrivAerDiagnosticEvaluatorError:
        raise
    except RetainedFileError as error:
        raise DrivAerDiagnosticEvaluatorError(str(error)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DrivAerDiagnosticEvaluatorError(
            f"cannot read valid {label}: {unresolved.name}"
        ) from error
    if not isinstance(value, dict):
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be a JSON object")
    return value, initial


def _sha256_file(path: Path | str, *, chunk_bytes: int = 1024 * 1024) -> str:
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
        raise DrivAerDiagnosticEvaluatorError("hash chunk size must be positive")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb", buffering=0) as source:
            while block := source.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise DrivAerDiagnosticEvaluatorError("cannot hash required input") from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    expected_set = set(expected)
    if set(result) != expected_set:
        missing = sorted(expected_set - set(result))
        unexpected = sorted(set(result) - expected_set)
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} keys differ from schema "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return result


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be {qualifier}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be Boolean")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _positive_integer(value: object, label: str) -> int:
    return _integer(value, label, minimum=1)


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be {qualifier}")
    return result


def _optional_finite(
    value: object, label: str, *, nonnegative: bool = False
) -> float | None:
    return None if value is None else _finite(value, label, nonnegative=nonnegative)


def _point3(value: object, label: str, *, optional: bool = False) -> tuple[float, ...] | None:
    if optional and value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} must contain exactly three finite values"
        )
    return tuple(_finite(item, label) for item in value)


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return result


def _case_id(value: object) -> str:
    result = _string(value, "case_id")
    if _CASE_RE.fullmatch(result) is None:
        raise DrivAerDiagnosticEvaluatorError(
            "case_id must have the form run_<positive integer>"
        )
    return result


def _basename(value: object, label: str, *, expected: str | None = None) -> str:
    result = _string(value, label)
    if (
        result.startswith("/")
        or _WINDOWS_ABSOLUTE_RE.match(result)
        or "/" in result
        or "\\" in result
        or result in {".", ".."}
    ):
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be a path-free basename")
    if expected is not None and result != expected:
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} must equal {expected!r}, got {result!r}"
        )
    return result


def _raw_ids(
    value: Sequence[int] | np.ndarray,
    label: str,
    *,
    upper_bound: int,
    require_unique: bool = True,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} must be a one-dimensional integer array"
        )
    if array.dtype.kind == "i" and np.any(array < 0):
        raise DrivAerDiagnosticEvaluatorError(f"{label} cannot contain negative IDs")
    if np.any(array >= upper_bound):
        raise DrivAerDiagnosticEvaluatorError(f"{label} exceeds native support")
    result = array.astype(np.int64, copy=False)
    if require_unique and len(result) != len(np.unique(result)):
        raise DrivAerDiagnosticEvaluatorError(f"{label} must be unique")
    return result


def _array_evidence_sha256(raw_ids: np.ndarray, values: np.ndarray) -> str:
    ids = np.asarray(raw_ids, dtype="<i8")
    array = np.asarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(_canonical_json({"ids_shape": ids.shape, "values_shape": array.shape}))
    digest.update(ids.tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _assert_no_absolute_paths(value: object, label: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} contains an absolute path"
            )


def _load_profile(path: Path | str) -> tuple[AutoCFD5Definition, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    digest = _sha256_file(source)
    if digest != EXPECTED_PROFILE_SHA256:
        raise DrivAerDiagnosticEvaluatorError(
            "AutoCFD5 profile is not the exact v8 candidate identity"
        )
    try:
        definition = load_autocfd5_definition(source)
    except AutoCFD5Error as error:
        raise DrivAerDiagnosticEvaluatorError(
            "AutoCFD5 v8 registry validation failed"
        ) from error
    if _sha256_file(source) != digest:
        raise DrivAerDiagnosticEvaluatorError(
            "AutoCFD5 profile changed during validation"
        )
    return definition, {
        "profile_sha256": digest,
        "source_registry_sha256": dict(definition.source_sha256),
        "line_count": VELOCITY_LINE_COUNT,
        "fixed_10mm_sample_count": VELOCITY_SAMPLE_COUNT,
    }


def _range_from_values(values: Sequence[float]) -> dict[str, float] | None:
    return {"minimum": min(values), "maximum": max(values)} if values else None


def _validate_range(value: object, expected: dict[str, float] | None, label: str) -> None:
    if expected is None:
        if value is not None:
            raise DrivAerDiagnosticEvaluatorError(f"{label} must be null")
        return
    result = _exact_keys(value, {"minimum", "maximum"}, label)
    if (
        _finite(result["minimum"], f"{label}.minimum") != expected["minimum"]
        or _finite(result["maximum"], f"{label}.maximum") != expected["maximum"]
    ):
        raise DrivAerDiagnosticEvaluatorError(f"{label} is inconsistent with rows")


def _validate_counter(value: object, label: str) -> dict[str, int]:
    source = _mapping(value, label)
    result: dict[str, int] = {}
    for key, count in source.items():
        name = _string(key, f"{label} key")
        result[name] = _integer(count, f"{label}.{name}", minimum=1)
    return result


def _is_unit_vector(value: tuple[float, ...] | None) -> bool:
    return value is None or math.isclose(
        math.sqrt(math.fsum(component * component for component in value)),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    )


def _expected_cp_algorithm(*, stl_chunk_facets: int) -> dict[str, object]:
    return {
        "id": "drivaerml-autocfd5-cp-geometric-mapping-candidate-v1",
        "scientific_status": (
            "candidate_not_owner_approved_not_active_scoring_support"
        ),
        "owner_visual_signoff_claimed": False,
        "source_integrity": {
            "stl": {
                "pathname_open": "single_open_retained_through_all_reads",
                "file_type": "regular_file_verified_by_fstat",
                "hash_input": "retained_verified_file_descriptor",
                "parser_inputs": [
                    "strict_ascii_stl_inventory",
                    "component_constrained_geometric_mapping",
                ],
                "descriptor_transport": (
                    "verified_procfs_or_devfs_fd_alias_same_device_inode"
                ),
                "identity_fields": [
                    "device",
                    "inode",
                    "mode",
                    "size_bytes",
                    "mtime_ns",
                    "ctime_ns",
                ],
                "post_read_fstat": "unchanged",
            },
            "boundary_vtp": {
                "pathname_open": "single_open_retained_through_all_reads",
                "file_type": "regular_file_verified_by_fstat",
                "hash_input": "retained_verified_file_descriptor",
                "parser_inputs": [
                    "vtk_polygon_locator",
                    "vtk_CellData_pMeanTrim_selection",
                ],
                "descriptor_transport": (
                    "verified_procfs_or_devfs_fd_alias_same_device_inode"
                ),
                "identity_fields": [
                    "device",
                    "inode",
                    "mode",
                    "size_bytes",
                    "mtime_ns",
                    "ctime_ns",
                ],
                "post_read_fstat": "unchanged",
            },
        },
        "stl": {
            "format": "strict_ASCII_multi_solid",
            "stream_chunk_facets": stl_chunk_facets,
            "raw_facet_identity": (
                "zero_based_global_facet_order_across_named_solids"
            ),
            "component_match": (
                "exact_named_solid_only_no_unrestricted_fallback"
            ),
            "projection_modes": ["component_closest_3d", "cut_plane_closest"],
            "cut_vertex_plane_tolerance_m": CUT_VERTEX_PLANE_TOLERANCE_M,
            "distance_tie_m": STL_DISTANCE_TIE_M,
            "tie_break": "smallest_raw_stl_triangle_id",
            "unnormalized_area_vector_invalid_lte_m2": (
                UNNORMALIZED_AREA_VECTOR_MIN_M2
            ),
            "nominal_displacement_review_m": (
                NOMINAL_PROBE_DISPLACEMENT_REVIEW_M
            ),
            "nominal_displacement_invalid_gt_m": (
                NOMINAL_PROBE_DISPLACEMENT_MAX_M
            ),
        },
        "native_vtp": {
            "association": "raw_native_VTK_CellData_polygon_order",
            "candidate_discovery": (
                "vtkStaticCellLocator_FindCellsWithinBounds_candidate_only"
            ),
            "required_vtk_version": REQUIRED_VTK_VERSION,
            "polygon_closest_point": (
                "ordered_first_vertex_triangle_fan_binary64"
            ),
            "polygon_normal": "normalize(sum_i(cross(v_i,v_i_plus_1)))",
            "bridge_distance_max_m": NATIVE_BRIDGE_MAX_DISTANCE_M,
            "bridge_distance_review_m": NATIVE_BRIDGE_DISTANCE_REVIEW_M,
            "bridge_abs_normal_dot_min": NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT,
            "distance_tie_m": NATIVE_DISTANCE_TIE_M,
            "normal_tie": NATIVE_NORMAL_TIE,
            "tie_break": (
                "distance_tie_then_decreasing_abs_normal_dot_then_"
                "smallest_raw_polygon_id"
            ),
        },
        "truth": {
            "source": "native_CellData_pMeanTrim_at_mapped_raw_polygon_id",
            "pMeanTrim_unit": "m2/s2",
            "equation": "Cp=2*pMeanTrim/Uinf^2",
            "Uinf_m_per_s": U_INF_M_PER_S,
            "Cp_unit": "dimensionless",
        },
    }


def _optional_csv_float(value: object) -> str:
    return "" if value is None else repr(float(value))


def _optional_csv_integer(value: object) -> str:
    return "" if value is None else str(value)


def _cp_csv_point_columns(prefix: str, value: object) -> dict[str, str]:
    if value is None:
        return {f"{prefix}_{axis}_m": "" for axis in "xyz"}
    point = tuple(value)  # type: ignore[arg-type]
    return {
        f"{prefix}_{axis}_m": repr(float(coordinate))
        for axis, coordinate in zip("xyz", point, strict=True)
    }


def _cp_csv_vector_columns(prefix: str, value: object) -> dict[str, str]:
    if value is None:
        return {f"{prefix}_{axis}": "" for axis in "xyz"}
    vector = tuple(value)  # type: ignore[arg-type]
    return {
        f"{prefix}_{axis}": repr(float(coordinate))
        for axis, coordinate in zip("xyz", vector, strict=True)
    }


def _cp_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CP_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for payload in rows:
        nominal = tuple(payload["nominal_point_m"])
        row = {
            "schema": _CP_CSV_SCHEMA,
            "case_id": str(payload["case_id"]),
            "autocfd_probe_id": str(payload["autocfd_probe_id"]),
            "nominal_x_m": repr(float(nominal[0])),
            "nominal_y_m": repr(float(nominal[1])),
            "nominal_z_m": repr(float(nominal[2])),
            "drivaerml_component": str(payload["drivaerml_component"]),
            "projection_mode": str(payload["projection_mode"]),
            "cut_axis": (
                "" if payload["cut_axis"] is None else str(payload["cut_axis"])
            ),
            "cut_value_m": _optional_csv_float(payload["cut_value_m"]),
            "owner_review_status": str(payload["owner_review_status"]),
            "mapping_valid": "true" if payload["mapping_valid"] else "false",
            "mapping_reason": str(payload["mapping_reason"]),
            "raw_stl_triangle_id": _optional_csv_integer(
                payload["raw_stl_triangle_id"]
            ),
            "nominal_displacement_m": _optional_csv_float(
                payload["nominal_displacement_m"]
            ),
            "raw_vtk_polygon_id": _optional_csv_integer(
                payload["raw_vtk_polygon_id"]
            ),
            "bridge_distance_m": _optional_csv_float(
                payload["bridge_distance_m"]
            ),
            "bridge_abs_normal_dot": _optional_csv_float(
                payload["bridge_abs_normal_dot"]
            ),
            "component_facet_count": str(payload["component_facet_count"]),
            "projection_candidate_count": str(
                payload["projection_candidate_count"]
            ),
            "native_bounds_candidate_count": str(
                payload["native_bounds_candidate_count"]
            ),
            "native_distance_pass_count": str(
                payload["native_distance_pass_count"]
            ),
            "native_normal_pass_count": str(
                payload["native_normal_pass_count"]
            ),
            "review_flags": ";".join(payload["review_flags"]),
            "truth_valid": "true" if payload["truth_valid"] else "false",
            "truth_reason": str(payload["truth_reason"]),
            "pMeanTrim_m2_per_s2": _optional_csv_float(
                payload["pMeanTrim_m2_per_s2"]
            ),
            "truth_Cp": _optional_csv_float(payload["truth_Cp"]),
            "support_valid": "true" if payload["support_valid"] else "false",
        }
        row.update(
            _cp_csv_point_columns("mapped_stl", payload["mapped_stl_point_m"])
        )
        row.update(_cp_csv_vector_columns("stl_normal", payload["stl_unit_normal"]))
        row.update(
            _cp_csv_point_columns(
                "native_closest", payload["native_closest_point_m"]
            )
        )
        row.update(
            _cp_csv_vector_columns(
                "native_normal", payload["native_polygon_unit_normal"]
            )
        )
        if set(row) != set(_CP_CSV_FIELDS):
            raise DrivAerDiagnosticEvaluatorError(
                "internal Cp mapping CSV schema mismatch"
            )
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _load_cp_support(
    path: Path,
    *,
    definition: AutoCFD5Definition,
    profile_binding: Mapping[str, object],
    case_id: str,
    expected_boundary_sha256: str | None,
) -> StrictCpCaseSupport:
    document, digest = _read_json(path, "Cp case support")
    root = _exact_keys(
        document,
        {
            "schema",
            "schema_version",
            "case_id",
            "status",
            "owner_visual_signoff_claimed",
            "definition",
            "sources",
            "dependencies",
            "algorithm",
            "summary",
            "artifacts",
            "rows",
        },
        "Cp case support",
    )
    if (
        root["schema"] != CP_SUPPORT_SCHEMA
        or root["schema_version"] != "1.0"
        or root["case_id"] != case_id
        or root["status"] != CP_SUPPORT_STATUS
        or root["owner_visual_signoff_claimed"] is not False
    ):
        raise DrivAerDiagnosticEvaluatorError("Cp case support identity is not exact")
    definition_record = _exact_keys(
        root["definition"],
        {"profile_sha256", "registry_sha256", "probe_count", "rule_count"},
        "Cp definition binding",
    )
    if (
        definition_record["profile_sha256"] != profile_binding["profile_sha256"]
        or definition_record["registry_sha256"]
        != profile_binding["source_registry_sha256"]
        or definition_record["probe_count"] != CP_PROBE_COUNT
        or definition_record["rule_count"] != CP_PROBE_COUNT
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp support differs from the exact v8 registry binding"
        )
    dependencies = _exact_keys(root["dependencies"], {"vtk", "numpy"}, "Cp dependencies")
    if dependencies != {"vtk": "9.5.2", "numpy": "2.2.6"}:
        raise DrivAerDiagnosticEvaluatorError("Cp support dependencies are not pinned")
    sources = _exact_keys(root["sources"], {"stl", "boundary"}, "Cp sources")
    stl = _exact_keys(
        sources["stl"],
        {
            "file",
            "sha256",
            "size_bytes",
            "solid_count",
            "facet_count",
            "solid_facet_counts",
        },
        "Cp STL source",
    )
    boundary = _exact_keys(
        sources["boundary"],
        {
            "file",
            "sha256",
            "size_bytes",
            "point_count",
            "polygon_count",
            "pMeanTrim_tuple_count",
            "pMeanTrim_component_count",
            "pMeanTrim_vtk_data_type",
        },
        "Cp boundary source",
    )
    run_number = case_id.removeprefix("run_")
    _basename(stl["file"], "STL file", expected=f"drivaer_{run_number}.stl")
    stl_sha256 = _sha256(stl["sha256"], "STL SHA-256")
    _positive_integer(stl["size_bytes"], "STL size")
    stl_facet_count = _positive_integer(stl["facet_count"], "STL facet count")
    solid_counts_source = _mapping(stl["solid_facet_counts"], "STL solid counts")
    solid_facet_counts: dict[str, int] = {}
    for raw_name, raw_count in solid_counts_source.items():
        name = _string(raw_name, "STL solid name")
        solid_facet_counts[name] = _positive_integer(
            raw_count, f"STL solid {name!r} facet count"
        )
    if (
        _positive_integer(stl["solid_count"], "STL solid count")
        != len(solid_facet_counts)
        or math.fsum(solid_facet_counts.values()) != stl_facet_count
    ):
        raise DrivAerDiagnosticEvaluatorError("Cp STL inventory does not close")
    boundary_sha256 = _sha256(boundary["sha256"], "boundary SHA-256")
    if expected_boundary_sha256 is not None and boundary_sha256 != _sha256(
        expected_boundary_sha256, "expected boundary SHA-256"
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp support boundary hash differs from the native-source pin"
        )
    boundary_file = _basename(
        boundary["file"], "boundary file", expected=f"boundary_{run_number}.vtp"
    )
    _positive_integer(boundary["point_count"], "boundary point count")
    polygon_count = _positive_integer(boundary["polygon_count"], "polygon_count")
    if (
        boundary["pMeanTrim_tuple_count"] != polygon_count
        or boundary["pMeanTrim_component_count"] != 1
        or _positive_integer(boundary["size_bytes"], "boundary size") < 1
        or _string(boundary["pMeanTrim_vtk_data_type"], "pMeanTrim VTK type")
        not in _ALLOWED_PRESSURE_VTK_TYPES
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp boundary pMeanTrim metadata is inconsistent"
        )
    algorithm = _mapping(root["algorithm"], "Cp algorithm")
    stl_algorithm = _mapping(algorithm.get("stl"), "Cp STL algorithm")
    stl_chunk_facets = _positive_integer(
        stl_algorithm.get("stream_chunk_facets"), "Cp STL stream chunk facets"
    )
    if dict(algorithm) != _expected_cp_algorithm(
        stl_chunk_facets=stl_chunk_facets
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp algorithm constants/status are not exact"
        )

    raw_rows = root["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != CP_PROBE_COUNT:
        raise DrivAerDiagnosticEvaluatorError(
            "Cp support must contain all 209 unique probes"
        )
    expected_probe_ids = tuple(
        probe.autocfd_probe_id for probe in definition.cp_probes
    )
    rule_by_id = {
        rule.autocfd_probe_id: rule for rule in definition.cp_component_rules
    }
    parsed: list[CpSupportRow] = []
    evidence: list[CpProbeMappingEvidence] = []
    mapping_reasons: collections.Counter[str] = collections.Counter()
    truth_reasons: collections.Counter[str] = collections.Counter()
    owner_statuses: collections.Counter[str] = collections.Counter()
    review_flag_counts: collections.Counter[str] = collections.Counter()
    validated_rows: list[Mapping[str, Any]] = []
    pressure_values: list[float] = []
    cp_values: list[float] = []
    for position, (raw, expected_probe_id) in enumerate(
        zip(raw_rows, expected_probe_ids, strict=True)
    ):
        label = f"Cp row {position}"
        row = _exact_keys(raw, _CP_ROW_KEYS, label)
        probe_id = _integer(row["autocfd_probe_id"], f"{label}.autocfd_probe_id", minimum=1)
        if row["case_id"] != case_id or probe_id != expected_probe_id:
            raise DrivAerDiagnosticEvaluatorError(
                "Cp rows must use exact case and ordered unique v8 probe IDs"
            )
        probe = definition.cp_probes[position]
        rule = rule_by_id[probe_id]
        if _point3(row["nominal_point_m"], f"{label}.nominal_point_m") != tuple(
            probe.point_m
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} differs from the nominal v8 point"
            )
        if (
            row["drivaerml_component"] != rule.drivaerml_component
            or row["projection_mode"] != rule.projection_mode
            or row["cut_axis"] != rule.cut_axis
            or row["cut_value_m"] != rule.cut_value_m
            or row["owner_review_status"] != rule.owner_review_status
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} differs from its v8 component rule"
            )
        owner_statuses[str(row["owner_review_status"])] += 1
        mapping_valid = _boolean(row["mapping_valid"], f"{label}.mapping_valid")
        mapping_reason = _string(
            row["mapping_reason"], f"{label}.mapping_reason", allow_empty=True
        )
        if mapping_valid and mapping_reason:
            raise DrivAerDiagnosticEvaluatorError(f"{label} valid mapping has a reason")
        if not mapping_valid:
            if mapping_reason not in _CP_MAPPING_INVALID_REASONS:
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} has an unknown mapping-invalid reason"
                )
            mapping_reasons[mapping_reason] += 1
        raw_polygon_id = _optional_integer(
            row["raw_vtk_polygon_id"], f"{label}.raw_vtk_polygon_id"
        )
        mapped_point = _point3(
            row["mapped_stl_point_m"],
            f"{label}.mapped_stl_point_m",
            optional=True,
        )
        raw_stl_triangle_id = _optional_integer(
            row["raw_stl_triangle_id"], f"{label}.raw_stl_triangle_id"
        )
        stl_normal = _point3(
            row["stl_unit_normal"], f"{label}.stl_unit_normal", optional=True
        )
        displacement = _optional_finite(
            row["nominal_displacement_m"],
            f"{label}.nominal_displacement_m",
            nonnegative=True,
        )
        native_point = _point3(
            row["native_closest_point_m"],
            f"{label}.native_closest_point_m",
            optional=True,
        )
        native_normal = _point3(
            row["native_polygon_unit_normal"],
            f"{label}.native_polygon_unit_normal",
            optional=True,
        )
        bridge_distance = _optional_finite(
            row["bridge_distance_m"],
            f"{label}.bridge_distance_m",
            nonnegative=True,
        )
        normal_dot = _optional_finite(
            row["bridge_abs_normal_dot"],
            f"{label}.bridge_abs_normal_dot",
            nonnegative=True,
        )
        if raw_stl_triangle_id is not None and raw_stl_triangle_id >= stl_facet_count:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} raw STL triangle ID is out of range"
            )
        if raw_polygon_id is not None and raw_polygon_id >= polygon_count:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} raw VTK polygon ID is out of range"
            )
        if normal_dot is not None and normal_dot > 1.0:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} normal agreement exceeds one"
            )
        if not _is_unit_vector(stl_normal):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} STL normal is not unit"
            )
        if not _is_unit_vector(native_normal):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} native polygon normal is not unit"
            )

        component_count = _integer(
            row["component_facet_count"], f"{label}.component_facet_count"
        )
        projection_count = _integer(
            row["projection_candidate_count"],
            f"{label}.projection_candidate_count",
        )
        bounds_count = _integer(
            row["native_bounds_candidate_count"],
            f"{label}.native_bounds_candidate_count",
        )
        distance_count = _integer(
            row["native_distance_pass_count"],
            f"{label}.native_distance_pass_count",
        )
        normal_count = _integer(
            row["native_normal_pass_count"],
            f"{label}.native_normal_pass_count",
        )
        if component_count != solid_facet_counts.get(rule.drivaerml_component, 0):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} component facet count differs from the STL inventory"
            )
        if component_count > stl_facet_count or projection_count > component_count:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} STL candidate counts do not close"
            )
        if not 0 <= normal_count <= distance_count <= bounds_count:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} native candidate counts do not close"
            )
        flags_value = row["review_flags"]
        if (
            not isinstance(flags_value, list)
            or any(
                not isinstance(flag, str) or flag not in _CP_REVIEW_FLAGS
                for flag in flags_value
            )
            or len(flags_value) != len(set(flags_value))
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} review_flags are invalid"
            )
        expected_flags: list[str] = []
        if (
            displacement is not None
            and displacement > NOMINAL_PROBE_DISPLACEMENT_REVIEW_M
        ):
            expected_flags.append("nominal_displacement_gt_7p5pct_wheelbase")
        if (
            bridge_distance is not None
            and bridge_distance > NATIVE_BRIDGE_DISTANCE_REVIEW_M
        ):
            expected_flags.append("native_bridge_distance_gt_0p5mm")
        if flags_value != expected_flags:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} review_flags are inconsistent with mapping values"
            )
        review_flag_counts.update(flags_value)
        stl_fields = (
            mapped_point,
            raw_stl_triangle_id,
            stl_normal,
            displacement,
        )
        native_fields = (
            native_point,
            raw_polygon_id,
            native_normal,
            bridge_distance,
            normal_dot,
        )
        if not (all(value is None for value in stl_fields) or all(
            value is not None for value in stl_fields
        )):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} STL mapping fields are only partially populated"
            )
        if not (all(value is None for value in native_fields) or all(
            value is not None for value in native_fields
        )):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} native mapping fields are only partially populated"
            )
        if mapping_valid:
            if (
                any(value is None for value in stl_fields)
                or any(value is None for value in native_fields)
                or projection_count < 1
                or normal_count < 1
            ):
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} valid mapping fields/counts are incomplete"
                )
        else:
            if any(value is not None for value in native_fields):
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} invalid mapping cannot claim a selected native polygon"
                )
            if mapping_reason == "declared_component_absent":
                reason_consistent = (
                    component_count == 0
                    and projection_count == 0
                    and bounds_count == 0
                    and all(value is None for value in stl_fields)
                )
            elif mapping_reason in {
                "declared_cut_has_no_finite_nondegenerate_intersection",
                "declared_component_has_no_finite_nondegenerate_triangle",
            }:
                reason_consistent = (
                    component_count > 0
                    and projection_count == 0
                    and bounds_count == 0
                    and all(value is None for value in stl_fields)
                )
            elif mapping_reason == "nominal_displacement_exceeds_10pct_wheelbase":
                reason_consistent = (
                    projection_count > 0
                    and all(value is not None for value in stl_fields)
                    and displacement is not None
                    and displacement > NOMINAL_PROBE_DISPLACEMENT_MAX_M
                    and bounds_count == 0
                )
            elif mapping_reason == "no_native_polygon_bounds_candidate_within_2mm":
                reason_consistent = (
                    projection_count > 0
                    and all(value is not None for value in stl_fields)
                    and bounds_count == 0
                )
            elif mapping_reason in {
                "no_finite_nondegenerate_native_polygon",
                "native_bridge_distance_exceeds_2mm",
            }:
                reason_consistent = (
                    projection_count > 0
                    and all(value is not None for value in stl_fields)
                    and bounds_count > 0
                    and distance_count == 0
                )
            else:
                assert mapping_reason == "native_bridge_normal_agreement_below_cos30"
                reason_consistent = (
                    projection_count > 0
                    and all(value is not None for value in stl_fields)
                    and distance_count > 0
                    and normal_count == 0
                )
            if not reason_consistent:
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} mapping reason is inconsistent with its fields/counts"
                )
        try:
            evidence.append(
                CpProbeMappingEvidence(
                    case_id=case_id,
                    autocfd_probe_id=probe_id,
                    valid=mapping_valid,
                    reason=mapping_reason,
                    mapped_point_m=mapped_point,
                    raw_stl_triangle_id=raw_stl_triangle_id,
                    raw_vtk_polygon_id=raw_polygon_id,
                    nominal_displacement_m=displacement,
                    bridge_distance_m=bridge_distance,
                    bridge_abs_normal_dot=normal_dot,
                    source_sha256=(stl_sha256, boundary_sha256),
                )
            )
        except AutoCFD5Error as error:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} violates the Cp mapping evidence schema"
            ) from error
        truth_valid = _boolean(row["truth_valid"], f"{label}.truth_valid")
        truth_reason = _string(
            row["truth_reason"], f"{label}.truth_reason", allow_empty=True
        )
        pressure = _optional_finite(
            row["pMeanTrim_m2_per_s2"], f"{label}.pMeanTrim_m2_per_s2"
        )
        truth_cp = _optional_finite(row["truth_Cp"], f"{label}.truth_Cp")
        support_valid = _boolean(row["support_valid"], f"{label}.support_valid")
        if support_valid != (mapping_valid and truth_valid):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} support_valid is inconsistent"
            )
        if truth_valid:
            if not mapping_valid or truth_reason or pressure is None or truth_cp is None:
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} valid truth fields are incomplete"
                )
            expected_cp = cp_from_kinematic_pressure(pressure)
            if not math.isclose(truth_cp, float(expected_cp), rel_tol=0.0, abs_tol=1.0e-15):
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} violates Cp=2*pMeanTrim/Uinf^2"
                )
            pressure_values.append(pressure)
            cp_values.append(truth_cp)
        else:
            expected_reason = (
                "nonfinite_native_pMeanTrim" if mapping_valid else "mapping_invalid"
            )
            if (
                truth_reason != expected_reason
                or pressure is not None
                or truth_cp is not None
            ):
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} invalid truth fields are inconsistent"
                )
            truth_reasons[truth_reason] += 1
        parsed.append(
            CpSupportRow(
                autocfd_probe_id=probe_id,
                mapping_valid=mapping_valid,
                mapping_reason=mapping_reason,
                raw_vtk_polygon_id=raw_polygon_id,
                truth_valid=truth_valid,
                truth_reason=truth_reason,
                pressure_m2_per_s2=pressure,
                truth_cp=truth_cp,
                support_valid=support_valid,
            )
        )
        validated_rows.append(row)
    try:
        validate_cp_mapping_evidence(
            evidence,
            definition,
            expected_case_ids=(case_id,),
            require_all_valid=False,
        )
    except AutoCFD5Error as error:
        raise DrivAerDiagnosticEvaluatorError(
            "Cp mappings fail the fixed v8 geometric gates"
        ) from error

    summary = _exact_keys(
        root["summary"],
        {
            "row_count",
            "mapping_valid_count",
            "mapping_invalid_count",
            "mapping_invalid_reason_counts",
            "truth_valid_count",
            "truth_invalid_count",
            "truth_invalid_reason_counts",
            "support_valid_count",
            "owner_review_status_counts",
            "review_flag_counts",
            "pMeanTrim_range_m2_per_s2",
            "truth_Cp_range",
        },
        "Cp summary",
    )
    mapping_valid_count = sum(row.mapping_valid for row in parsed)
    truth_valid_count = sum(row.truth_valid for row in parsed)
    support_valid_count = sum(row.support_valid for row in parsed)
    for key in (
        "row_count",
        "mapping_valid_count",
        "mapping_invalid_count",
        "truth_valid_count",
        "truth_invalid_count",
        "support_valid_count",
    ):
        _integer(summary[key], f"Cp summary.{key}")
    for key in (
        "mapping_invalid_reason_counts",
        "truth_invalid_reason_counts",
        "owner_review_status_counts",
        "review_flag_counts",
    ):
        _validate_counter(summary[key], f"Cp summary.{key}")
    if (
        summary["row_count"] != CP_PROBE_COUNT
        or summary["mapping_valid_count"] != mapping_valid_count
        or summary["mapping_invalid_count"] != CP_PROBE_COUNT - mapping_valid_count
        or summary["mapping_invalid_reason_counts"]
        != dict(sorted(mapping_reasons.items()))
        or summary["truth_valid_count"] != truth_valid_count
        or summary["truth_invalid_count"] != CP_PROBE_COUNT - truth_valid_count
        or summary["truth_invalid_reason_counts"] != dict(sorted(truth_reasons.items()))
        or summary["support_valid_count"] != support_valid_count
        or summary["owner_review_status_counts"]
        != dict(sorted(owner_statuses.items()))
        or summary["review_flag_counts"]
        != dict(sorted(review_flag_counts.items()))
    ):
        raise DrivAerDiagnosticEvaluatorError("Cp summary is inconsistent with rows")
    _validate_range(
        summary["pMeanTrim_range_m2_per_s2"],
        _range_from_values(pressure_values),
        "Cp pressure range",
    )
    _validate_range(
        summary["truth_Cp_range"],
        _range_from_values(cp_values),
        "Cp truth range",
    )
    artifacts = _exact_keys(
        root["artifacts"],
        {"mapping_csv_schema", "mapping_csv_row_count", "mapping_csv_sha256"},
        "Cp artifacts",
    )
    expected_csv_sha256 = hashlib.sha256(_cp_csv_bytes(validated_rows)).hexdigest()
    if (
        artifacts["mapping_csv_schema"] != _CP_CSV_SCHEMA
        or _positive_integer(
            artifacts["mapping_csv_row_count"], "Cp mapping CSV row count"
        )
        != CP_PROBE_COUNT
        or _sha256(artifacts["mapping_csv_sha256"], "Cp mapping CSV SHA-256")
        != expected_csv_sha256
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp mapping CSV artifact declaration is not exact"
        )
    panel_memberships = tuple(
        (membership.panel_id, membership.autocfd_probe_id)
        for membership in definition.cp_panel_memberships
    )
    if (
        len(panel_memberships) != CP_PANEL_MEMBERSHIP_COUNT
        or len({panel_id for panel_id, _ in panel_memberships}) != CP_PANEL_COUNT
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp panel membership registry is not exact"
        )
    return StrictCpCaseSupport(
        path=path.expanduser().resolve(),
        sha256=digest,
        case_id=case_id,
        profile_sha256=EXPECTED_PROFILE_SHA256,
        boundary_file=boundary_file,
        boundary_sha256=boundary_sha256,
        boundary_polygon_count=polygon_count,
        panel_memberships=panel_memberships,
        rows=tuple(parsed),
    )


def load_strict_cp_case_support(
    path: Path | str,
    *,
    autocfd5_profile: Path | str,
    case_id: str,
    expected_boundary_sha256: str | None = None,
) -> StrictCpCaseSupport:
    """Load and validate one complete 209-row candidate Cp support receipt."""

    canonical_case = _case_id(case_id)
    definition, profile_binding = _load_profile(autocfd5_profile)
    return _load_cp_support(
        Path(path),
        definition=definition,
        profile_binding=profile_binding,
        case_id=canonical_case,
        expected_boundary_sha256=expected_boundary_sha256,
    )


def _validate_false_case_claims(value: object, label: str) -> None:
    claims = _exact_keys(value, FALSE_CASE_CLAIMS, label)
    if dict(claims) != FALSE_CASE_CLAIMS:
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} must retain every candidate limitation as false"
        )


def _validate_velocity_source_binding(
    value: object,
    *,
    case_id: str,
    expected_source_pin_sha256: str | None,
    expected_source_part_sha256: Sequence[str] | None,
) -> tuple[str, tuple[str, ...]]:
    binding = _exact_keys(
        value,
        {
            "pin_sha256",
            "repository_id",
            "repository_revision",
            "case_id",
            "logical_size_bytes",
            "ordered_verified_segments",
            "verification",
        },
        "velocity native-source binding",
    )
    pin_sha256 = _sha256(binding["pin_sha256"], "velocity source-pin SHA-256")
    if expected_source_pin_sha256 is not None and pin_sha256 != _sha256(
        expected_source_pin_sha256, "expected source-pin SHA-256"
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping source-pin hash differs from the expected identity"
        )
    if (
        binding["repository_id"] != EXPECTED_REPOSITORY_ID
        or binding["repository_revision"] != EXPECTED_REPOSITORY_REVISION
        or binding["case_id"] != case_id
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping native-source identity is not exact"
        )
    logical_size = _positive_integer(
        binding["logical_size_bytes"], "velocity logical source size"
    )
    raw_segments = binding["ordered_verified_segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) not in {2, 3}:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping must bind two or three ordered source segments"
        )
    hashes: list[str] = []
    offset = 0
    for part_index, raw_segment in enumerate(raw_segments):
        segment = _exact_keys(
            raw_segment,
            {"part_index", "byte_offset", "size_bytes", "sha256"},
            f"velocity source segment {part_index}",
        )
        size = _positive_integer(segment["size_bytes"], "source segment size")
        if (
            segment["part_index"] != part_index
            or segment["byte_offset"] != offset
        ):
            raise DrivAerDiagnosticEvaluatorError(
                "velocity source segments are not contiguous and ordered"
            )
        hashes.append(_sha256(segment["sha256"], "source segment SHA-256"))
        offset += size
    if offset != logical_size:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity source segments do not close the logical byte extent"
        )
    if expected_source_part_sha256 is not None:
        expected_hashes = tuple(
            _sha256(value, "expected volume-part SHA-256")
            for value in expected_source_part_sha256
        )
        if tuple(hashes) != expected_hashes:
            raise DrivAerDiagnosticEvaluatorError(
                "velocity mapping source segments differ from the native-source pin"
            )
    verification = _exact_keys(
        binding["verification"],
        {"method", "timing", "vtk_input", "post_vtk_fstat"},
        "source verification",
    )
    if dict(verification) != {
        "method": "exact_ordered_segment_size_and_sha256",
        "timing": "completed_before_vtk_geometry_reader",
        "vtk_input": "retained_verified_file_descriptor",
        "post_vtk_fstat": "unchanged",
    }:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping source verification is not exact"
        )
    return pin_sha256, tuple(hashes)


def _validate_velocity_geometry(value: object, case_id: str) -> int:
    geometry = _exact_keys(
        value,
        {
            "dataset_type",
            "version",
            "byte_order",
            "header_type",
            "compressor",
            "piece_count",
            "declared_point_count",
            "declared_cell_count",
            "vtk_loaded_point_count",
            "vtk_loaded_cell_count",
            "reader_audit",
            "association",
            "raw_vtk_cell_id",
            "remeshing",
            "reordering",
        },
        "velocity geometry",
    )
    point_count = _positive_integer(
        geometry["declared_point_count"], "velocity point count"
    )
    cell_count = _positive_integer(
        geometry["declared_cell_count"], "velocity cell count"
    )
    if (
        geometry["dataset_type"] != "UnstructuredGrid"
        or geometry["byte_order"] != "LittleEndian"
        or geometry["header_type"] != "UInt64"
        or geometry["piece_count"] != 1
        or geometry["vtk_loaded_point_count"] != point_count
        or geometry["vtk_loaded_cell_count"] != cell_count
        or geometry["association"] != "native_volume_CellData"
        or geometry["raw_vtk_cell_id"] != "zero_based_native_GetCell_index"
        or geometry["remeshing"] is not False
        or geometry["reordering"] is not False
    ):
        raise DrivAerDiagnosticEvaluatorError(
            f"{case_id} velocity geometry semantics or counts are not exact"
        )
    audit = _exact_keys(
        geometry["reader_audit"],
        {
            "disabled_point_array_count",
            "disabled_point_arrays",
            "disabled_cell_array_count",
            "disabled_cell_arrays",
        },
        "velocity geometry reader audit",
    )
    for association in ("point", "cell"):
        arrays = audit[f"disabled_{association}_arrays"]
        if (
            not isinstance(arrays, list)
            or any(not isinstance(item, str) or not item for item in arrays)
            or len(arrays) != len(set(arrays))
            or audit[f"disabled_{association}_array_count"] != len(arrays)
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"velocity disabled {association} array audit is inconsistent"
            )
    if not {"pMeanTrim", "UMeanTrim"}.issubset(
        set(audit["disabled_cell_arrays"])
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity geometry reader did not disable native solution CellData"
        )
    return cell_count


def _validate_velocity_kernel(value: object) -> None:
    kernel = _exact_keys(
        value,
        {"kernel_id", "settings_sha256", "versions", "settings"},
        "velocity kernel",
    )
    expected_settings = candidate_kernel_settings()
    expected_hash = hashlib.sha256(_canonical_json(expected_settings)).hexdigest()
    if expected_hash != PINNED_KERNEL_SETTINGS_SHA256:
        raise DrivAerDiagnosticEvaluatorError(
            "local velocity candidate-v7 settings differ from the frozen hash"
        )
    if (
        kernel["kernel_id"] != KERNEL_ID
        or kernel["settings"] != expected_settings
        or kernel["settings_sha256"] != expected_hash
        or kernel["versions"] != PINNED_KERNEL_VERSIONS
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping kernel settings, hash, or versions are not pinned"
        )


def _validate_velocity_invalid_reason(
    reason: str, *, cell_count: int, candidate_count: int, label: str
) -> None:
    if reason in {NO_CLOSURE_CELL_REASON, *OWNER_INVALID_REASONS}:
        if reason == NO_CLOSURE_CELL_REASON and candidate_count != 0:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} no-closure assignment cannot retain candidates"
            )
        return
    if not reason.startswith(CELL_EVALUATION_FAILURE_REASON_PREFIX):
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} has an unknown invalid-assignment reason"
        )
    suffix = reason.removeprefix(CELL_EVALUATION_FAILURE_REASON_PREFIX)
    tokens = suffix.split(",")
    if (
        not suffix
        or any(re.fullmatch(r"0|[1-9][0-9]*", token) is None for token in tokens)
    ):
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} closure-evaluation failure IDs are not canonical unsigned decimals"
        )
    failed_ids = tuple(int(token) for token in tokens)
    if (
        tuple(sorted(set(failed_ids))) != failed_ids
        or any(raw_id >= cell_count for raw_id in failed_ids)
    ):
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} closure-evaluation failure IDs must be unique, increasing, and in range"
        )
    if candidate_count > cell_count - len(failed_ids):
        raise DrivAerDiagnosticEvaluatorError(
            f"{label} candidate/failure counts exceed the native cell count"
        )


def _validate_velocity_coverage(
    value: object,
    *,
    rows: Sequence[VelocityMappingRow],
    expected_line_counts: Mapping[str, int],
) -> None:
    coverage = _exact_keys(
        value,
        {
            "expected_sample_count",
            "actual_sample_count",
            "unique_line_sample_key_count",
            "complete_duplicate_free_no_omissions",
            "line_sample_counts",
            "valid_count",
            "invalid_count",
            "invalid_reason_counts",
            "candidate_count_histogram",
            "selected_raw_vtk_cell_id_min",
            "selected_raw_vtk_cell_id_max",
        },
        "velocity artifact coverage",
    )
    count = len(rows)
    valid_count = sum(row.valid for row in rows)
    invalid_reasons = collections.Counter(
        row.reason for row in rows if not row.valid
    )
    candidate_histogram = collections.Counter(row.candidate_count for row in rows)
    selected_ids = [
        row.raw_vtk_cell_id
        for row in rows
        if row.raw_vtk_cell_id is not None
    ]
    if (
        coverage["expected_sample_count"] != count
        or coverage["actual_sample_count"] != count
        or coverage["unique_line_sample_key_count"] != count
        or coverage["complete_duplicate_free_no_omissions"] is not True
        or coverage["line_sample_counts"] != dict(expected_line_counts)
        or coverage["valid_count"] != valid_count
        or coverage["invalid_count"] != count - valid_count
        or coverage["invalid_reason_counts"]
        != dict(sorted(invalid_reasons.items()))
        or coverage["candidate_count_histogram"]
        != {
            str(key): value
            for key, value in sorted(candidate_histogram.items())
        }
        or coverage["selected_raw_vtk_cell_id_min"]
        != (min(selected_ids) if selected_ids else None)
        or coverage["selected_raw_vtk_cell_id_max"]
        != (max(selected_ids) if selected_ids else None)
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity artifact coverage is inconsistent with explicit rows"
        )


def _validate_velocity_execution(value: object) -> None:
    """Validate the candidate-v7 bounded deterministic runtime audits."""

    execution = _exact_keys(
        value,
        {
            "io_chunk_bytes",
            "validation_chunk_cells",
            "resolution_order_mm",
            "geometric_tolerance_m",
            "containing_cell_query_cache",
            "polyhedron_geometry_cache",
            "polyhedron_evaluation",
        },
        "velocity execution",
    )
    if (
        _positive_integer(execution["io_chunk_bytes"], "io_chunk_bytes") < 1
        or _positive_integer(
            execution["validation_chunk_cells"], "validation_chunk_cells"
        )
        < 1
        or execution["resolution_order_mm"] != [1, 2, 5, 10]
        or execution["geometric_tolerance_m"]
        != POINT_IN_CELL_CLOSURE_TOLERANCE_M
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity receipt execution declaration is not exact"
        )

    query_cache = _exact_keys(
        execution["containing_cell_query_cache"],
        {"enabled", "key_id", "total_rows", "unique_query_keys", "cache_hits"},
        "velocity containing-cell query-cache audit",
    )
    total_rows = _positive_integer(query_cache["total_rows"], "total_rows")
    unique_queries = _positive_integer(
        query_cache["unique_query_keys"], "unique_query_keys"
    )
    query_hits = _integer(query_cache["cache_hits"], "cache_hits")
    expected_rows = 37_416 + 18_716 + 7_496 + VELOCITY_SAMPLE_COUNT
    expected_unique_queries = 39_362
    if (
        query_cache["enabled"] is not True
        or query_cache["key_id"] != QUERY_CACHE_KEY_ID
        or total_rows != expected_rows
        or unique_queries != expected_unique_queries
        or query_hits != expected_rows - expected_unique_queries
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity containing-cell query-cache audit is inconsistent"
        )

    geometry_cache = _exact_keys(
        execution["polyhedron_geometry_cache"],
        {
            "policy",
            "maximum_entries",
            "maximum_emitted_triangles",
            "current_entries",
            "current_emitted_triangles",
            "peak_entries",
            "peak_emitted_triangles",
            "cache_hits",
            "cache_misses",
            "evictions",
            "oversized_entry_bypasses",
            "fail_closed_preparations",
            "vtk_objects_cached",
        },
        "velocity polyhedron geometry-cache audit",
    )
    maximum_entries = _positive_integer(
        geometry_cache["maximum_entries"], "maximum_entries"
    )
    maximum_triangles = _positive_integer(
        geometry_cache["maximum_emitted_triangles"],
        "maximum_emitted_triangles",
    )
    current_entries = _integer(
        geometry_cache["current_entries"], "current_entries"
    )
    current_triangles = _integer(
        geometry_cache["current_emitted_triangles"],
        "current_emitted_triangles",
    )
    peak_entries = _integer(geometry_cache["peak_entries"], "peak_entries")
    peak_triangles = _integer(
        geometry_cache["peak_emitted_triangles"], "peak_emitted_triangles"
    )
    cache_hits = _integer(geometry_cache["cache_hits"], "cache_hits")
    cache_misses = _integer(geometry_cache["cache_misses"], "cache_misses")
    evictions = _integer(geometry_cache["evictions"], "evictions")
    oversized_bypasses = _integer(
        geometry_cache["oversized_entry_bypasses"], "oversized_entry_bypasses"
    )
    preparation_failures = _integer(
        geometry_cache["fail_closed_preparations"], "fail_closed_preparations"
    )
    if (
        geometry_cache["policy"] != "deterministic_least_recently_used"
        or geometry_cache["vtk_objects_cached"] is not False
        or maximum_entries != POLYHEDRON_GEOMETRY_CACHE_MAX_ENTRIES
        or maximum_triangles != POLYHEDRON_GEOMETRY_CACHE_MAX_TRIANGLES
        or current_entries > maximum_entries
        or peak_entries < current_entries
        or peak_entries > maximum_entries
        or current_triangles > maximum_triangles
        or peak_triangles < current_triangles
        or peak_triangles > maximum_triangles
        or oversized_bypasses > cache_misses
        or evictions > cache_misses - oversized_bypasses
        or current_entries != cache_misses - oversized_bypasses - evictions
        or preparation_failures > cache_misses
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity polyhedron geometry-cache audit is inconsistent"
        )

    evaluation = _exact_keys(
        execution["polyhedron_evaluation"],
        {
            "scope",
            "broad_phase_polyhedron_visit_count",
            "boundary_count",
            "inside_count",
            "outside_count",
            "ambiguous_count",
            "winding_classified_count",
            "minimum_winding_classification_margin_steradian",
            "classification_absolute_tolerance_steradian",
        },
        "velocity polyhedron evaluation audit",
    )
    visits = _integer(
        evaluation["broad_phase_polyhedron_visit_count"],
        "broad_phase_polyhedron_visit_count",
    )
    boundary = _integer(evaluation["boundary_count"], "boundary_count")
    inside = _integer(evaluation["inside_count"], "inside_count")
    outside = _integer(evaluation["outside_count"], "outside_count")
    ambiguous = _integer(evaluation["ambiguous_count"], "ambiguous_count")
    winding = _integer(
        evaluation["winding_classified_count"], "winding_classified_count"
    )
    margin = _optional_finite(
        evaluation["minimum_winding_classification_margin_steradian"],
        "minimum_winding_classification_margin_steradian",
        nonnegative=True,
    )
    if (
        evaluation["scope"]
        != "broad_phase_polyhedron_visits_for_uncached_exact_xyz_tolerance_queries"
        or evaluation["classification_absolute_tolerance_steradian"]
        != POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
        or visits != boundary + inside + outside + ambiguous
        or visits != cache_hits + cache_misses
        or winding != inside + outside
        or ambiguous < preparation_failures
        or (winding == 0) != (margin is None)
        or (
            margin is not None
            and margin > POLYHEDRON_SOLID_ANGLE_ABSOLUTE_TOLERANCE
        )
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity polyhedron evaluation audit is inconsistent"
        )


def _load_velocity_mapping(
    artifact_path: Path,
    receipt_path: Path,
    *,
    definition: AutoCFD5Definition,
    profile_binding: Mapping[str, object],
    case_id: str,
    expected_source_pin_sha256: str | None,
    expected_source_part_sha256: Sequence[str] | None,
) -> StrictVelocityMapping:
    receipt, receipt_sha256 = _read_json(receipt_path, "velocity receipt")
    artifact, artifact_sha256 = _read_json(
        artifact_path, "10 mm velocity mapping artifact"
    )
    if receipt_path.name != "receipt.json":
        raise DrivAerDiagnosticEvaluatorError(
            "velocity receipt must use the path-free basename receipt.json"
        )
    _basename(
        artifact_path.name,
        "velocity artifact",
        expected=VELOCITY_ARTIFACT_BASENAME,
    )
    receipt_root = _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "status",
            "case_id",
            "registries",
            "native_source_binding",
            "geometry",
            "kernel",
            "execution",
            "artifacts",
            "coverage",
            "claims",
        },
        "velocity receipt",
    )
    if (
        receipt_root["schema"] != VELOCITY_RECEIPT_SCHEMA
        or receipt_root["schema_version"] != 1
        or receipt_root["status"] != VELOCITY_STATUS
        or receipt_root["case_id"] != case_id
        or receipt_root["registries"] != profile_binding
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity receipt identity or registry binding is not exact"
        )
    _validate_false_case_claims(receipt_root["claims"], "velocity receipt claims")
    pin_sha256, source_part_sha256 = _validate_velocity_source_binding(
        receipt_root["native_source_binding"],
        case_id=case_id,
        expected_source_pin_sha256=expected_source_pin_sha256,
        expected_source_part_sha256=expected_source_part_sha256,
    )
    cell_count = _validate_velocity_geometry(receipt_root["geometry"], case_id)
    _validate_velocity_kernel(receipt_root["kernel"])
    _validate_velocity_execution(receipt_root["execution"])
    receipt_coverage = _exact_keys(
        receipt_root["coverage"],
        {
            "resolution_count",
            "all_sixteen_lines_each_resolution",
            "all_samples_explicit_no_silent_omissions",
            "expected_sample_counts",
        },
        "velocity receipt coverage",
    )
    if (
        receipt_coverage["resolution_count"] != 4
        or receipt_coverage["all_sixteen_lines_each_resolution"] is not True
        or receipt_coverage["all_samples_explicit_no_silent_omissions"] is not True
        or receipt_coverage["expected_sample_counts"]
        != {"1": 37416, "2": 18716, "5": 7496, "10": VELOCITY_SAMPLE_COUNT}
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity receipt does not declare exact four-grid coverage"
        )
    raw_summaries = receipt_root["artifacts"]
    if not isinstance(raw_summaries, list) or len(raw_summaries) != 4:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity receipt must contain four ordered artifact summaries"
        )
    expected_resolutions = (
        (1, 0.001, "velocity-cell-mapping-01mm.json"),
        (2, 0.002, "velocity-cell-mapping-02mm.json"),
        (5, 0.005, "velocity-cell-mapping-05mm.json"),
        (10, 0.010, VELOCITY_ARTIFACT_BASENAME),
    )
    summary_10mm: Mapping[str, Any] | None = None
    for position, (raw, expected) in enumerate(
        zip(raw_summaries, expected_resolutions, strict=True)
    ):
        summary = _exact_keys(
            raw,
            {
                "nominal_spacing_mm",
                "nominal_spacing_m",
                "artifact",
                "size_bytes",
                "sha256",
                "assignment_evidence_sha256",
                "line_count",
                "sample_count",
                "valid_count",
                "invalid_count",
                "invalid_reason_counts",
                "complete_duplicate_free_no_omissions",
            },
            f"velocity artifact summary {position}",
        )
        spacing_mm, spacing_m, filename = expected
        if (
            summary["nominal_spacing_mm"] != spacing_mm
            or summary["nominal_spacing_m"] != spacing_m
            or _basename(summary["artifact"], "velocity artifact name") != filename
            or summary["line_count"] != VELOCITY_LINE_COUNT
            or summary["complete_duplicate_free_no_omissions"] is not True
        ):
            raise DrivAerDiagnosticEvaluatorError(
                "velocity receipt artifact order or identity is not exact"
            )
        if spacing_mm == 10:
            summary_10mm = summary
    assert summary_10mm is not None
    if (
        summary_10mm["size_bytes"] != artifact_path.stat().st_size
        or summary_10mm["sha256"] != artifact_sha256
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "10 mm velocity artifact bytes differ from its receipt"
        )

    artifact_root = _exact_keys(
        artifact,
        {
            "schema",
            "schema_version",
            "status",
            "case_id",
            "resolution",
            "constant_evidence_fields",
            "row_fields",
            "rows",
            "coverage",
            "assignment_evidence_sha256",
            "units",
            "registries",
            "native_source_binding",
            "geometry",
            "kernel",
            "claims",
        },
        "10 mm velocity artifact",
    )
    if (
        artifact_root["schema"] != VELOCITY_ARTIFACT_SCHEMA
        or artifact_root["schema_version"] != 1
        or artifact_root["status"] != VELOCITY_STATUS
        or artifact_root["case_id"] != case_id
        or artifact_root["registries"] != receipt_root["registries"]
        or artifact_root["native_source_binding"]
        != receipt_root["native_source_binding"]
        or artifact_root["geometry"] != receipt_root["geometry"]
        or artifact_root["kernel"] != receipt_root["kernel"]
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "10 mm velocity artifact identity or receipt bindings are not exact"
        )
    _validate_false_case_claims(artifact_root["claims"], "velocity artifact claims")
    resolution = _exact_keys(
        artifact_root["resolution"],
        {
            "nominal_spacing_mm",
            "nominal_spacing_m",
            "sample_source",
            "line_count",
            "sample_count",
        },
        "10 mm velocity resolution",
    )
    if dict(resolution) != {
        "nominal_spacing_mm": 10,
        "nominal_spacing_m": 0.01,
        "sample_source": "expanded_autocfd5_v8_10mm_registry",
        "line_count": VELOCITY_LINE_COUNT,
        "sample_count": VELOCITY_SAMPLE_COUNT,
    }:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping is not the exact 10 mm scoring grid"
        )
    constants = _exact_keys(
        artifact_root["constant_evidence_fields"],
        {"geometric_tolerance_m", "source_sha256"},
        "velocity constant evidence",
    )
    if (
        constants["geometric_tolerance_m"]
        != POINT_IN_CELL_CLOSURE_TOLERANCE_M
        or constants["source_sha256"] != list(source_part_sha256)
        or artifact_root["row_fields"] != EXPECTED_ROW_FIELDS
        or artifact_root["units"] != {"point_m": "m", "distance_m": "m"}
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity artifact constants, row order, or units are not exact"
        )
    raw_rows = artifact_root["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != VELOCITY_SAMPLE_COUNT:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity artifact must retain all 3,756 10 mm samples"
        )
    parsed: list[VelocityMappingRow] = []
    evidence: list[VelocityCellAssignmentEvidence] = []
    line_counts: collections.Counter[str] = collections.Counter()
    for position, (raw, sample) in enumerate(
        zip(raw_rows, definition.velocity_samples, strict=True)
    ):
        label = f"velocity row {position}"
        if not isinstance(raw, list) or len(raw) != len(EXPECTED_ROW_FIELDS):
            raise DrivAerDiagnosticEvaluatorError(f"{label} has the wrong shape")
        profile_id = _string(raw[0], f"{label}.profile_id")
        sample_index = _integer(raw[1], f"{label}.sample_index")
        point = _point3(raw[2], f"{label}.point_m")
        distance = _finite(raw[3], f"{label}.distance_m", nonnegative=True)
        valid = _boolean(raw[4], f"{label}.valid")
        reason = _string(raw[5], f"{label}.reason", allow_empty=True)
        raw_cell_id = _optional_integer(raw[6], f"{label}.raw_vtk_cell_id")
        candidate_count = _integer(raw[7], f"{label}.candidate_count")
        if (
            (profile_id, sample_index) != (sample.profile_id, sample.sample_index)
            or point != tuple(sample.point_m)
            or distance != sample.distance_m
        ):
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} differs from the exact ordered 10 mm registry"
            )
        if candidate_count > cell_count:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} candidate_count exceeds native cell count"
            )
        if valid:
            if (
                reason
                or raw_cell_id is None
                or raw_cell_id >= cell_count
                or candidate_count < 1
            ):
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} valid assignment is inconsistent"
                )
        else:
            if raw_cell_id is not None:
                raise DrivAerDiagnosticEvaluatorError(
                    f"{label} invalid assignment is inconsistent"
                )
            _validate_velocity_invalid_reason(
                reason,
                cell_count=cell_count,
                candidate_count=candidate_count,
                label=label,
            )
        row = VelocityMappingRow(
            profile_id=profile_id,
            sample_index=sample_index,
            distance_m=distance,
            valid=valid,
            reason=reason,
            raw_vtk_cell_id=raw_cell_id,
            candidate_count=candidate_count,
        )
        parsed.append(row)
        line_counts[profile_id] += 1
        try:
            evidence.append(
                VelocityCellAssignmentEvidence(
                    case_id=case_id,
                    profile_id=profile_id,
                    sample_index=sample_index,
                    point_m=tuple(point),  # type: ignore[arg-type]
                    distance_m=distance,
                    valid=valid,
                    reason=reason,
                    raw_vtk_cell_id=raw_cell_id,
                    candidate_count=candidate_count,
                    geometric_tolerance_m=POINT_IN_CELL_CLOSURE_TOLERANCE_M,
                    source_sha256=source_part_sha256,
                )
            )
        except AutoCFD5Error as error:
            raise DrivAerDiagnosticEvaluatorError(
                f"{label} violates the velocity evidence schema"
            ) from error
    evidence_sha256 = assignment_evidence_sha256(evidence)
    if (
        artifact_root["assignment_evidence_sha256"] != evidence_sha256
        or summary_10mm["assignment_evidence_sha256"] != evidence_sha256
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity assignment evidence SHA-256 is inconsistent"
        )
    expected_line_counts = {
        line.profile_id: line.sample_count for line in definition.velocity_lines
    }
    if dict(line_counts) != expected_line_counts:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping does not cover every line exactly"
        )
    _validate_velocity_coverage(
        artifact_root["coverage"],
        rows=parsed,
        expected_line_counts=expected_line_counts,
    )
    valid_count = sum(row.valid for row in parsed)
    invalid_reasons = collections.Counter(
        row.reason for row in parsed if not row.valid
    )
    if (
        summary_10mm["sample_count"] != VELOCITY_SAMPLE_COUNT
        or summary_10mm["valid_count"] != valid_count
        or summary_10mm["invalid_count"] != VELOCITY_SAMPLE_COUNT - valid_count
        or summary_10mm["invalid_reason_counts"]
        != dict(sorted(invalid_reasons.items()))
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity receipt 10 mm summary is inconsistent with explicit rows"
        )
    experimental_profile_ids = tuple(
        line.profile_id
        for line in definition.velocity_lines
        if line.experimental_availability != "none"
    )
    if experimental_profile_ids != _EXPERIMENTAL_VELOCITY_PROFILE_IDS:
        raise DrivAerDiagnosticEvaluatorError(
            "velocity experimental subset differs from the exact v8 eleven-line set"
        )
    return StrictVelocityMapping(
        artifact_path=artifact_path.expanduser().resolve(),
        artifact_sha256=artifact_sha256,
        receipt_path=receipt_path.expanduser().resolve(),
        receipt_sha256=receipt_sha256,
        case_id=case_id,
        profile_sha256=EXPECTED_PROFILE_SHA256,
        source_pin_sha256=pin_sha256,
        source_part_sha256=source_part_sha256,
        native_cell_count=cell_count,
        experimental_profile_ids=experimental_profile_ids,
        rows=tuple(parsed),
    )


def load_strict_velocity_10mm_mapping(
    artifact_path: Path | str,
    receipt_path: Path | str,
    *,
    autocfd5_profile: Path | str,
    case_id: str,
    expected_source_pin_sha256: str | None = None,
    expected_source_part_sha256: Sequence[str] | None = None,
) -> StrictVelocityMapping:
    """Load strict receipt-bound assignments for all 3,756 10 mm samples."""

    canonical_case = _case_id(case_id)
    definition, profile_binding = _load_profile(autocfd5_profile)
    return _load_velocity_mapping(
        Path(artifact_path),
        Path(receipt_path),
        definition=definition,
        profile_binding=profile_binding,
        case_id=canonical_case,
        expected_source_pin_sha256=expected_source_pin_sha256,
        expected_source_part_sha256=expected_source_part_sha256,
    )


def gather_mapped_prediction_field(
    manifest: PredictionChunkManifest | Path | str,
    raw_cell_ids: Sequence[int] | np.ndarray,
    *,
    case_id: str,
    support_id: str,
    field_name: str,
    expected_total_row_count: int,
    maximum_chunk_rows: int | None = None,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
) -> SparsePredictionGather:
    """Exhaust a complete prediction manifest while retaining mapped IDs only."""

    canonical_case = _case_id(case_id)
    expected_count = _positive_integer(
        expected_total_row_count, "expected_total_row_count"
    )
    if isinstance(manifest, PredictionChunkManifest):
        parsed = load_prediction_chunk_manifest(manifest.path)
        if parsed != manifest:
            raise DrivAerDiagnosticEvaluatorError(
                "preloaded prediction manifest differs from a retained-file replay"
            )
    else:
        parsed = load_prediction_chunk_manifest(manifest)
    if (
        parsed.case_id != canonical_case
        or parsed.support_id != support_id
        or parsed.total_row_count != expected_count
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "prediction manifest case, support, or native count is not exact"
        )
    if field_name not in parsed.field_components:
        raise DrivAerDiagnosticEvaluatorError(
            f"prediction support {support_id!r} has no field {field_name!r}"
        )
    if maximum_chunk_rows is not None:
        maximum = _positive_integer(maximum_chunk_rows, "maximum_chunk_rows")
        if any(descriptor.row_count > maximum for descriptor in parsed.chunks):
            raise DrivAerDiagnosticEvaluatorError(
                f"prediction chunk exceeds maximum_chunk_rows={maximum}"
            )
    requested = _raw_ids(
        raw_cell_ids,
        "mapped prediction raw_cell_ids",
        upper_bound=expected_count,
        require_unique=False,
    )
    unique = np.unique(requested)
    components = parsed.field_components[field_name]
    selected = np.empty(
        (len(unique),) if components == 1 else (len(unique), components),
        dtype=np.float64,
    )
    found = np.zeros(len(unique), dtype=bool)
    initial_manifest_hash = parsed.sha256
    try:
        for chunk in iter_prediction_chunks(
            parsed,
            hash_chunk_bytes=hash_chunk_bytes,
            validation_block_rows=validation_block_rows,
        ):
            start = chunk.descriptor.raw_cell_id_start
            stop = chunk.descriptor.raw_cell_id_stop
            first = int(np.searchsorted(unique, start, side="left"))
            last = int(np.searchsorted(unique, stop, side="left"))
            if first < last:
                ids = unique[first:last]
                local = ids - start
                selected[first:last] = np.asarray(
                    chunk.field(field_name)[local], dtype=np.float64
                )
                found[first:last] = True
            del chunk
    except PredictionChunkError as error:
        raise DrivAerDiagnosticEvaluatorError(str(error)) from error
    if not np.all(found):
        raise DrivAerDiagnosticEvaluatorError(
            "complete prediction manifest did not yield every mapped raw cell ID"
        )
    if _sha256_file(parsed.path, chunk_bytes=hash_chunk_bytes) != initial_manifest_hash:
        raise DrivAerDiagnosticEvaluatorError(
            "prediction manifest changed during sparse gather"
        )
    if not np.all(np.isfinite(selected)):
        raise DrivAerDiagnosticEvaluatorError(
            "mapped prediction values contain non-finite values"
        )
    unique.setflags(write=False)
    selected.setflags(write=False)
    return SparsePredictionGather(
        case_id=canonical_case,
        support_id=support_id,
        field_name=field_name,
        total_row_count=expected_count,
        raw_cell_ids=unique,
        values=selected,
        manifest_file=_basename(parsed.path.name, "prediction manifest file"),
        manifest_sha256=initial_manifest_hash,
        chunk_sha256=tuple(descriptor.sha256 for descriptor in parsed.chunks),
    )


class _SparseNativeTupleSink:
    """Select sorted raw IDs while validating every decoded native tuple."""

    def __init__(
        self,
        *,
        raw_cell_ids: np.ndarray,
        expected_tuple_count: int,
        components: int,
        byte_order: str,
    ) -> None:
        prefix = "<" if byte_order == "LittleEndian" else ">"
        self.dtype = np.dtype(prefix + "f4")
        self.components = components
        self.tuple_bytes = self.dtype.itemsize * components
        self.raw_cell_ids = raw_cell_ids
        self.expected_tuple_count = expected_tuple_count
        self.values = np.empty(
            (len(raw_cell_ids),)
            if components == 1
            else (len(raw_cell_ids), components),
            dtype=np.float64,
        )
        self.found = np.zeros(len(raw_cell_ids), dtype=bool)
        self.pending = bytearray()
        self.cursor = 0

    def _consume(self, tuple_count: int) -> None:
        if tuple_count < 1:
            return
        byte_count = tuple_count * self.tuple_bytes
        payload = bytes(self.pending[:byte_count])
        del self.pending[:byte_count]
        block = np.frombuffer(payload, dtype=self.dtype).reshape(
            tuple_count, self.components
        )
        if not np.all(np.isfinite(block)):
            raise DrivAerDiagnosticEvaluatorError(
                "native diagnostic truth field contains non-finite values"
            )
        start = self.cursor
        stop = start + tuple_count
        if stop > self.expected_tuple_count:
            raise DrivAerDiagnosticEvaluatorError(
                "native diagnostic truth exceeds its declared tuple count"
            )
        first = int(np.searchsorted(self.raw_cell_ids, start, side="left"))
        last = int(np.searchsorted(self.raw_cell_ids, stop, side="left"))
        if first < last:
            local = self.raw_cell_ids[first:last] - start
            selected = block[local]
            if self.components == 1:
                selected = selected[:, 0]
            self.values[first:last] = selected
            self.found[first:last] = True
        self.cursor = stop

    def write(self, payload: bytes) -> int:
        self.pending.extend(payload)
        complete_tuples = len(self.pending) // self.tuple_bytes
        self._consume(complete_tuples)
        return len(payload)

    def finish(self) -> np.ndarray:
        if self.pending:
            raise DrivAerDiagnosticEvaluatorError(
                "native diagnostic truth payload ends within a tuple"
            )
        if self.cursor != self.expected_tuple_count or not np.all(self.found):
            raise DrivAerDiagnosticEvaluatorError(
                "native diagnostic truth does not cover every mapped raw cell ID"
            )
        return self.values


def gather_sparse_inline_native_field(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    raw_cell_ids: Sequence[int] | np.ndarray,
    *,
    case_id: str,
    support_id: str,
    field_name: str,
    expected_components: int,
    source_files: Sequence[str],
    source_sha256: Sequence[str],
    encoded_chunk_bytes: int = 8 * 1024 * 1024,
) -> tuple[SparseNativeField, InlineBinaryPayloadSummary]:
    """Decode a complete native field and retain only sorted mapped tuples."""

    if array.association != "CellData" or array.name != field_name:
        raise DrivAerDiagnosticEvaluatorError(
            "native sparse gather requires the requested CellData field"
        )
    if array.vtk_type != "Float32" or array.number_of_components != expected_components:
        raise DrivAerDiagnosticEvaluatorError(
            f"native {field_name} must be Float32 with {expected_components} components"
        )
    if array.piece_index < 0 or array.piece_index >= len(vtk_index.pieces):
        raise DrivAerDiagnosticEvaluatorError(
            "native diagnostic field references a missing VTK Piece"
        )
    expected_count = vtk_index.pieces[array.piece_index].number_of_cells
    ids = np.unique(
        _raw_ids(
            raw_cell_ids,
            "mapped native raw_cell_ids",
            upper_bound=expected_count,
            require_unique=False,
        )
    )
    sink = _SparseNativeTupleSink(
        raw_cell_ids=ids,
        expected_tuple_count=expected_count,
        components=expected_components,
        byte_order=vtk_index.byte_order,
    )
    try:
        payload = stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            sink,
            encoded_chunk_size=_positive_integer(
                encoded_chunk_bytes, "encoded_chunk_bytes"
            ),
        )
        values = sink.finish()
    except InlineBinaryDecodeError as error:
        raise DrivAerDiagnosticEvaluatorError(str(error)) from error
    if payload.tuple_count != expected_count:
        raise DrivAerDiagnosticEvaluatorError(
            "native diagnostic field payload tuple count is inconsistent"
        )
    field = SparseNativeField(
        case_id=_case_id(case_id),
        support_id=support_id,
        field_name=field_name,
        total_row_count=expected_count,
        raw_cell_ids=ids,
        values=values,
        source_files=tuple(source_files),
        source_sha256=tuple(source_sha256),
        source_payload_sha256=payload.payload_sha256,
        complete_source_identity_verified=True,
    )
    return field, payload


def sparse_native_field_from_array(
    values: object,
    raw_cell_ids: Sequence[int] | np.ndarray,
    *,
    case_id: str,
    support_id: str,
    field_name: str,
    source_file: str,
    source_sha256: str,
) -> SparseNativeField:
    """Construct exact sparse native evidence from an in-memory test/source array."""

    array = np.asarray(values)
    if array.ndim not in {1, 2} or array.shape[0] < 1:
        raise DrivAerDiagnosticEvaluatorError(
            "native field array must have shape [cell] or [cell, component]"
        )
    ids = np.unique(
        _raw_ids(
            raw_cell_ids,
            "mapped native raw_cell_ids",
            upper_bound=array.shape[0],
            require_unique=False,
        )
    )
    return SparseNativeField(
        case_id=case_id,
        support_id=support_id,
        field_name=field_name,
        total_row_count=array.shape[0],
        raw_cell_ids=ids,
        values=np.asarray(array[ids], dtype=np.float64),
        source_files=(source_file,),
        source_sha256=(source_sha256,),
        complete_source_identity_verified=True,
    )


def _require_exact_native_selection(
    field: SparseNativeField,
    *,
    case_id: str,
    support_id: str,
    field_name: str,
    total_row_count: int,
    required_ids: np.ndarray,
) -> None:
    if (
        field.case_id != case_id
        or field.support_id != support_id
        or field.field_name != field_name
        or field.total_row_count != total_row_count
        or not np.array_equal(field.raw_cell_ids, np.unique(required_ids))
    ):
        raise DrivAerDiagnosticEvaluatorError(
            f"native sparse field {field_name!r} must contain exactly the mapped IDs"
        )


def _unavailable_reason(
    *,
    diagnostic: str,
    stage: str,
    reason: str,
    autocfd_probe_id: int | None = None,
    profile_id: str | None = None,
    sample_index: int | None = None,
) -> dict[str, object]:
    return {
        "diagnostic": diagnostic,
        "stage": stage,
        "reason": reason,
        "autocfd_probe_id": autocfd_probe_id,
        "profile_id": profile_id,
        "sample_index": sample_index,
    }


def _cp_unavailable_reasons(
    rows: Sequence[CpSupportRow],
    *,
    diagnostic: str = "cp_probe_rmse",
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        if not row.mapping_valid:
            result.append(
                _unavailable_reason(
                    diagnostic=diagnostic,
                    stage="mapping",
                    reason=row.mapping_reason,
                    autocfd_probe_id=row.autocfd_probe_id,
                )
            )
        elif not row.truth_valid:
            result.append(
                _unavailable_reason(
                    diagnostic=diagnostic,
                    stage="native_truth",
                    reason=row.truth_reason,
                    autocfd_probe_id=row.autocfd_probe_id,
                )
            )
    return result


def _velocity_unavailable_reasons(
    rows: Sequence[VelocityMappingRow],
    *,
    profile_ids: frozenset[str] | None = None,
    diagnostic: str = "velocity_profile_uinf_rmse",
) -> list[dict[str, object]]:
    return [
        _unavailable_reason(
            diagnostic=diagnostic,
            stage="mapping",
            reason=row.reason,
            profile_id=row.profile_id,
            sample_index=row.sample_index,
        )
        for row in rows
        if not row.valid
        and (profile_ids is None or row.profile_id in profile_ids)
    ]


def evaluate_loaded_case_diagnostics(
    *,
    cp_support: StrictCpCaseSupport,
    velocity_mapping: StrictVelocityMapping,
    surface_prediction_manifest: PredictionChunkManifest | Path | str,
    volume_prediction_manifest: PredictionChunkManifest | Path | str,
    native_surface_pressure: SparseNativeField,
    native_volume_velocity: SparseNativeField,
    maximum_prediction_chunk_rows: int | None = 1_000_000,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
) -> CandidateCaseDiagnostics:
    """Evaluate strict loaded support without omitting any invalid mapping row."""

    if cp_support.case_id != velocity_mapping.case_id:
        raise DrivAerDiagnosticEvaluatorError(
            "Cp and velocity mapping evidence use different case IDs"
        )
    case_id = cp_support.case_id
    cp_ids = np.asarray(
        [
            row.raw_vtk_polygon_id
            for row in cp_support.rows
            if row.mapping_valid and row.raw_vtk_polygon_id is not None
        ],
        dtype=np.int64,
    )
    velocity_ids = np.asarray(
        [
            row.raw_vtk_cell_id
            for row in velocity_mapping.rows
            if row.valid and row.raw_vtk_cell_id is not None
        ],
        dtype=np.int64,
    )
    _require_exact_native_selection(
        native_surface_pressure,
        case_id=case_id,
        support_id="surface_native_cells",
        field_name="pMeanTrim",
        total_row_count=cp_support.boundary_polygon_count,
        required_ids=cp_ids,
    )
    _require_exact_native_selection(
        native_volume_velocity,
        case_id=case_id,
        support_id="volume_native_cells",
        field_name="UMeanTrim",
        total_row_count=velocity_mapping.native_cell_count,
        required_ids=velocity_ids,
    )
    if native_surface_pressure.source_sha256 != (cp_support.boundary_sha256,):
        raise DrivAerDiagnosticEvaluatorError(
            "native surface truth hash differs from the strict Cp support"
        )
    if native_volume_velocity.source_sha256 != velocity_mapping.source_part_sha256:
        raise DrivAerDiagnosticEvaluatorError(
            "native volume truth part hashes differ from the velocity mapping"
        )
    surface_prediction = gather_mapped_prediction_field(
        surface_prediction_manifest,
        cp_ids,
        case_id=case_id,
        support_id="surface_native_cells",
        field_name="pMeanTrim",
        expected_total_row_count=cp_support.boundary_polygon_count,
        maximum_chunk_rows=maximum_prediction_chunk_rows,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
    )
    volume_prediction = gather_mapped_prediction_field(
        volume_prediction_manifest,
        velocity_ids,
        case_id=case_id,
        support_id="volume_native_cells",
        field_name="UMeanTrim",
        expected_total_row_count=velocity_mapping.native_cell_count,
        maximum_chunk_rows=maximum_prediction_chunk_rows,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
    )

    if velocity_mapping.experimental_profile_ids != (
        _EXPERIMENTAL_VELOCITY_PROFILE_IDS
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "velocity mapping does not retain the exact v8 experimental subset"
        )
    if (
        len(cp_support.panel_memberships) != CP_PANEL_MEMBERSHIP_COUNT
        or len({panel for panel, _ in cp_support.panel_memberships})
        != CP_PANEL_COUNT
    ):
        raise DrivAerDiagnosticEvaluatorError(
            "Cp support does not retain the exact panel membership scope"
        )
    cp_reasons = _cp_unavailable_reasons(cp_support.rows)
    cp_panel_reasons = _cp_unavailable_reasons(
        cp_support.rows, diagnostic="cp_panel_macro_rmse"
    )
    velocity_reasons = _velocity_unavailable_reasons(velocity_mapping.rows)
    experimental_velocity_reasons = _velocity_unavailable_reasons(
        velocity_mapping.rows,
        profile_ids=frozenset(velocity_mapping.experimental_profile_ids),
        diagnostic="velocity_profile_experimental_subset_uinf_rmse",
    )
    truth_valid_positions = [
        position
        for position, row in enumerate(cp_support.rows)
        if row.truth_valid
    ]
    if truth_valid_positions:
        truth_valid_ids = np.asarray(
            [
                cp_support.rows[position].raw_vtk_polygon_id
                for position in truth_valid_positions
            ],
            dtype=np.int64,
        )
        actual_valid_pressure = native_surface_pressure.values_for(truth_valid_ids)
        support_valid_pressure = np.asarray(
            [
                cp_support.rows[position].pressure_m2_per_s2
                for position in truth_valid_positions
            ],
            dtype=np.float64,
        )
        if not np.array_equal(actual_valid_pressure, support_valid_pressure):
            raise DrivAerDiagnosticEvaluatorError(
                "Cp support pressure replay differs from the verified native boundary"
            )
        support_valid_cp = np.asarray(
            [cp_support.rows[position].truth_cp for position in truth_valid_positions],
            dtype=np.float64,
        )
        derived_valid_cp = cp_from_kinematic_pressure(actual_valid_pressure)
        if not np.allclose(
            derived_valid_cp,
            support_valid_cp,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise DrivAerDiagnosticEvaluatorError(
                "Cp truth does not satisfy 2*pMeanTrim/Uinf^2"
            )

    all_required_cp_truth_verified = (
        not cp_reasons and len(truth_valid_positions) == CP_PROBE_COUNT
    )
    cp_metric: dict[str, object] = {
        "metric_id": "cp_probe_rmse",
        "ranked_value_available": not cp_reasons,
        "required_unique_probe_count": CP_PROBE_COUNT,
        "unavailable_reasons": cp_reasons,
        "case_rmse": None,
        "aggregation": "per_case_unique_probe_rmse_then_macro_average",
        "weighting": "209_unique_probes_equal",
        "equation": "Cp=2*pMeanTrim/Uinf^2",
        "Uinf_m_per_s": U_INF_M_PER_S,
        "valid_truth_row_equation_replay_count": len(truth_valid_positions),
        "truth_equation_verified": all_required_cp_truth_verified,
    }
    cp_panel_metric: dict[str, object] = {
        "metric_id": "cp_panel_macro_rmse",
        "value_available": not cp_panel_reasons,
        "required_panel_count": CP_PANEL_COUNT,
        "required_panel_membership_row_count": CP_PANEL_MEMBERSHIP_COUNT,
        "required_unique_probe_count": CP_PROBE_COUNT,
        "unavailable_reasons": cp_panel_reasons,
        "panel_rmse": [],
        "case_equal_panel_mean_rmse": None,
        "aggregation": "per_case_panel_rmse_then_equal_panel_and_case_average",
        "weighting": "panel_membership_rows_equal",
        "truth_equation_verified": all_required_cp_truth_verified,
    }
    if not cp_reasons:
        ordered_ids = np.asarray(
            [row.raw_vtk_polygon_id for row in cp_support.rows], dtype=np.int64
        )
        predicted_pressure = surface_prediction.values_for(ordered_ids)
        predicted_cp = cp_from_kinematic_pressure(predicted_pressure)
        truth_cp_support = np.asarray(
            [row.truth_cp for row in cp_support.rows], dtype=np.float64
        )
        cp_metric["case_rmse"] = cp_probe_rmse(predicted_cp, truth_cp_support)
        position_by_probe = {
            row.autocfd_probe_id: position
            for position, row in enumerate(cp_support.rows)
        }
        panel_positions: dict[str, list[int]] = collections.defaultdict(list)
        for panel_id, probe_id in cp_support.panel_memberships:
            if probe_id not in position_by_probe:
                raise DrivAerDiagnosticEvaluatorError(
                    "Cp panel membership references a missing unique probe"
                )
            panel_positions[panel_id].append(position_by_probe[probe_id])
        if (
            len(panel_positions) != CP_PANEL_COUNT
            or sum(len(values) for values in panel_positions.values())
            != CP_PANEL_MEMBERSHIP_COUNT
        ):
            raise DrivAerDiagnosticEvaluatorError(
                "Cp panel reduction coverage is not exactly 217 rows/15 panels"
            )
        squared_error = (predicted_cp - truth_cp_support) ** 2
        panel_results: list[dict[str, object]] = []
        for panel_id, positions in panel_positions.items():
            panel_rmse = math.sqrt(
                math.fsum(float(squared_error[position]) for position in positions)
                / len(positions)
            )
            panel_results.append(
                {
                    "panel_id": panel_id,
                    "membership_row_count": len(positions),
                    "rmse": panel_rmse,
                }
            )
        cp_panel_metric["panel_rmse"] = panel_results
        cp_panel_metric["case_equal_panel_mean_rmse"] = math.fsum(
            float(row["rmse"]) for row in panel_results
        ) / CP_PANEL_COUNT

    velocity_metric: dict[str, object] = {
        "metric_id": "velocity_profile_uinf_rmse",
        "ranked_value_available": not velocity_reasons,
        "required_line_count": VELOCITY_LINE_COUNT,
        "required_sample_count": VELOCITY_SAMPLE_COUNT,
        "unavailable_reasons": velocity_reasons,
        "line_rmse": [],
        "case_equal_line_mean_rmse": None,
        "quantity": "magnitude(UMeanTrim)/Uinf",
        "Uinf_m_per_s": U_INF_M_PER_S,
        "arc_rule": "trapezoidal_squared_error_over_complete_line_arc_length",
        "aggregation": "equal_case_equal_line_macro_average",
        "weighting": "trapezoidal_arc_length_within_line",
    }
    experimental_velocity_metric: dict[str, object] = {
        "metric_id": "velocity_profile_experimental_subset_uinf_rmse",
        "value_available": not experimental_velocity_reasons,
        "required_line_count": len(_EXPERIMENTAL_VELOCITY_PROFILE_IDS),
        "required_profile_ids": list(_EXPERIMENTAL_VELOCITY_PROFILE_IDS),
        "unavailable_reasons": experimental_velocity_reasons,
        "line_rmse": [],
        "case_equal_experimental_line_mean_rmse": None,
        "quantity": "magnitude(UMeanTrim)/Uinf",
        "Uinf_m_per_s": U_INF_M_PER_S,
        "arc_rule": "trapezoidal_squared_error_over_complete_line_arc_length",
        "aggregation": "equal_case_equal_experimental_line_macro_average",
        "weighting": "trapezoidal_arc_length_within_line",
    }
    if not velocity_reasons or not experimental_velocity_reasons:
        valid_positions = [
            position
            for position, row in enumerate(velocity_mapping.rows)
            if row.valid and row.raw_vtk_cell_id is not None
        ]
        valid_ids = np.asarray(
            [velocity_mapping.rows[position].raw_vtk_cell_id for position in valid_positions],
            dtype=np.int64,
        )
        prediction_valid_ratio = velocity_magnitude_ratio(
            volume_prediction.values_for(valid_ids)
        )
        truth_valid_ratio = velocity_magnitude_ratio(
            native_volume_velocity.values_for(valid_ids)
        )
        prediction_ratio = np.full(len(velocity_mapping.rows), np.nan, dtype=np.float64)
        truth_ratio = np.full(len(velocity_mapping.rows), np.nan, dtype=np.float64)
        prediction_ratio[valid_positions] = prediction_valid_ratio
        truth_ratio[valid_positions] = truth_valid_ratio
        by_line: dict[str, list[int]] = collections.defaultdict(list)
        for position, row in enumerate(velocity_mapping.rows):
            by_line[row.profile_id].append(position)

        def line_results_for(profile_ids: Sequence[str]) -> list[dict[str, object]]:
            results: list[dict[str, object]] = []
            for profile_id in profile_ids:
                positions = by_line.get(profile_id, [])
                if len(positions) < 2 or any(
                    not velocity_mapping.rows[position].valid for position in positions
                ):
                    raise DrivAerDiagnosticEvaluatorError(
                        f"velocity line {profile_id!r} is unavailable during reduction"
                    )
                indices = np.asarray(positions, dtype=np.int64)
                distances = np.asarray(
                    [
                        velocity_mapping.rows[position].distance_m
                        for position in positions
                    ],
                    dtype=np.float64,
                )
                rmse = velocity_profile_rmse(
                    distances,
                    prediction_ratio[indices],
                    truth_ratio[indices],
                    np.ones(len(indices), dtype=bool),
                )
                results.append(
                    {
                        "profile_id": profile_id,
                        "sample_count": len(indices),
                        "arc_length_m": float(distances[-1] - distances[0]),
                        "rmse": rmse,
                    }
                )
            return results

        if not velocity_reasons:
            all_profile_ids = tuple(by_line)
            line_results = line_results_for(all_profile_ids)
            if len(line_results) != VELOCITY_LINE_COUNT:
                raise DrivAerDiagnosticEvaluatorError(
                    "velocity reduction did not produce exactly sixteen lines"
                )
            velocity_metric["line_rmse"] = line_results
            velocity_metric["case_equal_line_mean_rmse"] = math.fsum(
                float(row["rmse"]) for row in line_results
            ) / VELOCITY_LINE_COUNT
        if not experimental_velocity_reasons:
            experimental_results = line_results_for(
                velocity_mapping.experimental_profile_ids
            )
            if len(experimental_results) != len(
                _EXPERIMENTAL_VELOCITY_PROFILE_IDS
            ):
                raise DrivAerDiagnosticEvaluatorError(
                    "velocity experimental reduction did not produce eleven lines"
                )
            experimental_velocity_metric["line_rmse"] = experimental_results
            experimental_velocity_metric[
                "case_equal_experimental_line_mean_rmse"
            ] = math.fsum(
                float(row["rmse"]) for row in experimental_results
            ) / len(_EXPERIMENTAL_VELOCITY_PROFILE_IDS)

    invalid_cp_rows = [
        {
            "autocfd_probe_id": row.autocfd_probe_id,
            "mapping_valid": row.mapping_valid,
            "mapping_reason": row.mapping_reason,
            "raw_vtk_polygon_id": row.raw_vtk_polygon_id,
            "truth_valid": row.truth_valid,
            "truth_reason": row.truth_reason,
            "support_valid": row.support_valid,
        }
        for row in cp_support.rows
        if not row.support_valid
    ]
    invalid_velocity_rows = [
        {
            "profile_id": row.profile_id,
            "sample_index": row.sample_index,
            "valid": row.valid,
            "reason": row.reason,
            "raw_vtk_cell_id": row.raw_vtk_cell_id,
            "candidate_count": row.candidate_count,
        }
        for row in velocity_mapping.rows
        if not row.valid
    ]
    evidence: dict[str, object] = {
        "schema": CANDIDATE_SCHEMA,
        "schema_version": 1,
        "status": CANDIDATE_STATUS,
        "case_id": case_id,
        "official_submission": False,
        "mapping_inputs": {
            "cp_support": {
                "file": _basename(cp_support.path.name, "Cp support file"),
                "sha256": cp_support.sha256,
                "profile_sha256": cp_support.profile_sha256,
                "row_count": len(cp_support.rows),
                "invalid_rows": invalid_cp_rows,
            },
            "velocity_10mm": {
                "artifact_file": _basename(
                    velocity_mapping.artifact_path.name,
                    "velocity artifact file",
                ),
                "artifact_sha256": velocity_mapping.artifact_sha256,
                "receipt_file": _basename(
                    velocity_mapping.receipt_path.name,
                    "velocity receipt file",
                ),
                "receipt_sha256": velocity_mapping.receipt_sha256,
                "profile_sha256": velocity_mapping.profile_sha256,
                "row_count": len(velocity_mapping.rows),
                "invalid_rows": invalid_velocity_rows,
            },
        },
        "sparse_gather_evidence": {
            "surface_prediction": surface_prediction.audit_record(),
            "volume_prediction": volume_prediction.audit_record(),
            "surface_native_truth": native_surface_pressure.audit_record(),
            "volume_native_truth": native_volume_velocity.audit_record(),
            "only_unique_mapped_raw_ids_retained": True,
            "prediction_manifests_fully_consumed": True,
        },
        "metrics": {
            "cp_probe_rmse": cp_metric,
            "cp_panel_macro_rmse": cp_panel_metric,
            "velocity_profile_uinf_rmse": velocity_metric,
            "velocity_profile_experimental_subset_uinf_rmse": (
                experimental_velocity_metric
            ),
        },
        "claims": dict(OUTPUT_FALSE_CLAIMS),
    }
    _assert_no_absolute_paths(evidence)
    return CandidateCaseDiagnostics(MappingProxyType(evidence))


def evaluate_case_diagnostics(
    *,
    case_id: str,
    autocfd5_profile: Path | str,
    cp_support_json: Path | str,
    velocity_mapping_json: Path | str,
    velocity_receipt_json: Path | str,
    surface_prediction_manifest: PredictionChunkManifest | Path | str,
    volume_prediction_manifest: PredictionChunkManifest | Path | str,
    native_surface_pressure: SparseNativeField,
    native_volume_velocity: SparseNativeField,
    expected_source_pin_sha256: str | None = None,
    expected_boundary_sha256: str | None = None,
    expected_source_part_sha256: Sequence[str] | None = None,
    maximum_prediction_chunk_rows: int | None = 1_000_000,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
) -> CandidateCaseDiagnostics:
    """Load strict mapping inputs and evaluate one candidate case."""

    cp_support = load_strict_cp_case_support(
        cp_support_json,
        autocfd5_profile=autocfd5_profile,
        case_id=case_id,
        expected_boundary_sha256=expected_boundary_sha256,
    )
    velocity_mapping = load_strict_velocity_10mm_mapping(
        velocity_mapping_json,
        velocity_receipt_json,
        autocfd5_profile=autocfd5_profile,
        case_id=case_id,
        expected_source_pin_sha256=expected_source_pin_sha256,
        expected_source_part_sha256=expected_source_part_sha256,
    )
    return evaluate_loaded_case_diagnostics(
        cp_support=cp_support,
        velocity_mapping=velocity_mapping,
        surface_prediction_manifest=surface_prediction_manifest,
        volume_prediction_manifest=volume_prediction_manifest,
        native_surface_pressure=native_surface_pressure,
        native_volume_velocity=native_volume_velocity,
        maximum_prediction_chunk_rows=maximum_prediction_chunk_rows,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
    )


def write_candidate_diagnostic_evidence(
    evaluation: CandidateCaseDiagnostics, path: Path | str
) -> dict[str, object]:
    """Atomically write deterministic candidate-only path-free JSON evidence."""

    if not isinstance(evaluation, CandidateCaseDiagnostics):
        raise DrivAerDiagnosticEvaluatorError(
            "evaluation must be CandidateCaseDiagnostics"
        )
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise DrivAerDiagnosticEvaluatorError(
            "diagnostic evidence output must use the .json suffix"
        )
    document = evaluation.to_json()
    encoded = _canonical_json(document) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "byte_size": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


__all__ = [
    "CANDIDATE_SCHEMA",
    "CANDIDATE_STATUS",
    "CandidateCaseDiagnostics",
    "CpSupportRow",
    "DrivAerDiagnosticEvaluatorError",
    "SparseNativeField",
    "SparsePredictionGather",
    "StrictCpCaseSupport",
    "StrictVelocityMapping",
    "VelocityMappingRow",
    "evaluate_case_diagnostics",
    "evaluate_loaded_case_diagnostics",
    "gather_mapped_prediction_field",
    "gather_sparse_inline_native_field",
    "load_strict_cp_case_support",
    "load_strict_velocity_10mm_mapping",
    "sparse_native_field_from_array",
    "write_candidate_diagnostic_evidence",
]
