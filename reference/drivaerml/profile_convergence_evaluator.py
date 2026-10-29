"""Build DrivAerML profile-convergence losses from native predictions.

The lower-level :mod:`reference.drivaerml.profile_convergence` checker accepts
an already-computed rectangular loss tensor.  This module supplies the missing
dataset-specific construction step.  It binds each declared method to complete
native ``CellData`` prediction manifests, samples the frozen 1/2/5/10 mm
containing-cell mappings in raw VTK cell order, and computes the prescribed
no-gap arc-weighted profile loss.

Geometric assignment invariance and loss/method-order convergence are retained
as separate evidence.  A numerically converged loss study cannot hide a change
of native cell, validity, invalidity reason, or closure-candidate count at a
nested sample.  None of the outputs in this module activate the benchmark or
claim that a declared trained checkpoint is scientifically genuine.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .autocfd5 import (
    POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    U_INF_M_PER_S,
    AutoCFD5Definition,
    AutoCFD5Error,
    VelocityCellAssignmentEvidence,
    velocity_magnitude_ratio,
    velocity_profile_rmse,
)
from .diagnostic_evaluator import (
    DrivAerDiagnosticEvaluatorError,
    SparseNativeField,
    SparsePredictionGather,
    gather_mapped_prediction_field,
)
from .prediction_chunks import (
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    PredictionChunkManifest,
    load_prediction_chunk_manifest,
)
from .profile_convergence import (
    INPUT_SCHEMA,
    OFFICIAL_CASE_ORDER,
    PROFILE_ORDER,
    SPACINGS_MM,
    ProfileConvergenceError,
    canonical_sha256,
    evaluate_profile_convergence,
    method_set_sha256,
)
from .velocity_assignments import (
    CELL_EVALUATION_FAILURE_REASON_PREFIX,
    NO_CLOSURE_CELL_REASON,
    OWNER_INVALID_REASONS,
    assignment_evidence_sha256,
    generate_definition_velocity_samples,
)


PREDICTION_STUDY_SCHEMA = (
    "drivaerml-profile-resolution-native-prediction-study-v1"
)
CONSTRUCTED_EVIDENCE_SCHEMA = (
    "drivaerml-profile-resolution-native-prediction-evidence-v1"
)
SCHEMA_VERSION = 1
SUPPORT_ID = "volume_native_cells"
FIELD_NAME = "UMeanTrim"
ASSOCIATION = "CellData"
NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M = 2.0e-11
EXPECTED_SAMPLE_COUNTS = {1: 37_416, 2: 18_716, 5: 7_496, 10: 3_756}
NESTED_STRIDES = {2: 2, 5: 5, 10: 10}
MAPPING_ARTIFACT_NAMES = {
    1: "velocity-cell-mapping-01mm.json",
    2: "velocity-cell-mapping-02mm.json",
    5: "velocity-cell-mapping-05mm.json",
    10: "velocity-cell-mapping-10mm.json",
}
FALSE_CLAIMS = {
    "scoring_contract_active": False,
    "submissions_open": False,
    "official_submission": False,
    "owner_scientific_approval": False,
    "independent_participant_dry_run": False,
    "trained_method_scientific_authenticity_verified": False,
}

_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class NativeProfileConvergenceError(ValueError):
    """Raised when native prediction evidence cannot be reduced safely."""


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NativeProfileConvergenceError(
            f"{label} must be a lowercase hexadecimal SHA-256"
        )
    return value


def _case_id(value: object, label: str = "case_id") -> str:
    if not isinstance(value, str) or _CASE_RE.fullmatch(value) is None:
        raise NativeProfileConvergenceError(
            f"{label} must match run_<positive integer>"
        )
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NativeProfileConvergenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise NativeProfileConvergenceError(
            f"{label} must be finite and non-negative"
        )
    return result


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise NativeProfileConvergenceError(f"{label} must be a positive integer")
    return value


def _exact_mapping(
    value: object, expected: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NativeProfileConvergenceError(f"{label} must be an object")
    if set(value) != expected:
        raise NativeProfileConvergenceError(
            f"{label} keys differ from the constructed-evidence schema"
        )
    return value


def _basename(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise NativeProfileConvergenceError(f"{label} must be a basename")
    return value


def _assert_no_absolute_paths(value: object, label: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        if value.replace("\\", "/").startswith("/") or _WINDOWS_ABSOLUTE_RE.match(
            value
        ):
            raise NativeProfileConvergenceError(
                f"{label} contains an absolute path"
            )


@dataclass(frozen=True)
class ProfileMappingRow:
    """One explicit containing-cell result at one velocity-line sample."""

    profile_id: str
    sample_index: int
    point_m: tuple[float, float, float]
    distance_m: float
    valid: bool
    reason: str
    raw_vtk_cell_id: int | None
    candidate_count: int

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILE_ORDER:
            raise NativeProfileConvergenceError(
                f"unknown velocity profile ID {self.profile_id!r}"
            )
        if (
            not isinstance(self.sample_index, int)
            or isinstance(self.sample_index, bool)
            or self.sample_index < 0
        ):
            raise NativeProfileConvergenceError(
                "mapping sample_index must be a non-negative integer"
            )
        if not isinstance(self.point_m, tuple) or len(self.point_m) != 3:
            raise NativeProfileConvergenceError(
                "mapping point_m must be a three-coordinate tuple"
            )
        for coordinate in self.point_m:
            if not math.isfinite(float(coordinate)):
                raise NativeProfileConvergenceError(
                    "mapping point_m must contain finite coordinates"
                )
        _finite_nonnegative(self.distance_m, "mapping distance_m")
        if not isinstance(self.valid, bool):
            raise NativeProfileConvergenceError("mapping valid must be Boolean")
        if not isinstance(self.reason, str):
            raise NativeProfileConvergenceError("mapping reason must be a string")
        if (
            not isinstance(self.candidate_count, int)
            or isinstance(self.candidate_count, bool)
            or self.candidate_count < 0
        ):
            raise NativeProfileConvergenceError(
                "mapping candidate_count must be a non-negative integer"
            )
        if self.valid:
            if (
                self.reason
                or not isinstance(self.raw_vtk_cell_id, int)
                or isinstance(self.raw_vtk_cell_id, bool)
                or self.raw_vtk_cell_id < 0
                or self.candidate_count < 1
            ):
                raise NativeProfileConvergenceError(
                    "valid mapping row has inconsistent reason, raw ID, or count"
                )
        elif (
            not self.reason
            or self.raw_vtk_cell_id is not None
        ):
            raise NativeProfileConvergenceError(
                "invalid mapping row must retain a reason and no selected raw ID"
            )

    def assignment_key(self) -> tuple[bool, str, int | None, int]:
        """Return the geometry result compared at nested sample locations."""

        return (
            self.valid,
            self.reason,
            self.raw_vtk_cell_id,
            self.candidate_count,
        )


@dataclass(frozen=True)
class ResolutionMapping:
    """One complete case/grid mapping after receipt-level validation."""

    case_id: str
    spacing_mm: int
    native_cell_count: int
    source_part_sha256: tuple[str, ...]
    artifact_name: str
    artifact_sha256: str
    assignment_evidence_sha256: str
    rows: tuple[ProfileMappingRow, ...]

    def __post_init__(self) -> None:
        _case_id(self.case_id)
        if self.spacing_mm not in SPACINGS_MM:
            raise NativeProfileConvergenceError(
                "mapping spacing must be exactly 1, 2, 5, or 10 mm"
            )
        count = _positive_integer(self.native_cell_count, "native_cell_count")
        if not self.source_part_sha256:
            raise NativeProfileConvergenceError(
                "mapping must retain at least one native source part hash"
            )
        for index, digest in enumerate(self.source_part_sha256):
            _sha256(digest, f"source_part_sha256[{index}]")
        if self.artifact_name != MAPPING_ARTIFACT_NAMES[self.spacing_mm]:
            raise NativeProfileConvergenceError(
                "mapping artifact basename differs from its resolution"
            )
        _sha256(self.artifact_sha256, "mapping artifact SHA-256")
        _sha256(
            self.assignment_evidence_sha256,
            "mapping assignment-evidence SHA-256",
        )
        if len(self.rows) != EXPECTED_SAMPLE_COUNTS[self.spacing_mm]:
            raise NativeProfileConvergenceError(
                f"{self.spacing_mm} mm mapping must contain exactly "
                f"{EXPECTED_SAMPLE_COUNTS[self.spacing_mm]} rows"
            )
        for row in self.rows:
            if row.raw_vtk_cell_id is not None and row.raw_vtk_cell_id >= count:
                raise NativeProfileConvergenceError(
                    "mapping raw VTK cell ID exceeds native CellData extent"
                )
            if row.candidate_count > count:
                raise NativeProfileConvergenceError(
                    "mapping candidate count exceeds native cell count"
                )
            if not row.valid:
                if row.reason == NO_CLOSURE_CELL_REASON:
                    if row.candidate_count != 0:
                        raise NativeProfileConvergenceError(
                            "no-closure mapping row must have zero candidates"
                        )
                elif row.reason in OWNER_INVALID_REASONS:
                    pass
                elif row.reason.startswith(
                    CELL_EVALUATION_FAILURE_REASON_PREFIX
                ):
                    suffix = row.reason[
                        len(CELL_EVALUATION_FAILURE_REASON_PREFIX) :
                    ]
                    tokens = suffix.split(",") if suffix else []
                    if any(
                        not token.isascii()
                        or not token.isdecimal()
                        or token != str(int(token))
                        for token in tokens
                    ):
                        raise NativeProfileConvergenceError(
                            "mapping evaluation-failure reason has malformed raw IDs"
                        )
                    failed_ids = tuple(int(token) for token in tokens)
                    if (
                        not failed_ids
                        or failed_ids != tuple(sorted(set(failed_ids)))
                        or any(raw_id >= count for raw_id in failed_ids)
                        or row.candidate_count > count - len(failed_ids)
                    ):
                        raise NativeProfileConvergenceError(
                            "mapping evaluation-failure reason/count is inconsistent"
                        )
                else:
                    raise NativeProfileConvergenceError(
                        f"mapping row has unsupported invalid reason {row.reason!r}"
                    )
        try:
            actual_assignment_sha256 = assignment_evidence_sha256(
                tuple(
                    VelocityCellAssignmentEvidence(
                        case_id=self.case_id,
                        profile_id=row.profile_id,
                        sample_index=row.sample_index,
                        point_m=row.point_m,
                        distance_m=row.distance_m,
                        valid=row.valid,
                        reason=row.reason,
                        raw_vtk_cell_id=row.raw_vtk_cell_id,
                        candidate_count=row.candidate_count,
                        geometric_tolerance_m=(
                            POINT_IN_CELL_CLOSURE_TOLERANCE_M
                        ),
                        source_sha256=self.source_part_sha256,
                    )
                    for row in self.rows
                )
            )
        except (AutoCFD5Error, ValueError) as error:
            raise NativeProfileConvergenceError(
                "mapping rows violate assignment evidence semantics"
            ) from error
        if actual_assignment_sha256 != self.assignment_evidence_sha256:
            raise NativeProfileConvergenceError(
                "mapping assignment-evidence SHA-256 does not match explicit rows"
            )

    def rows_by_profile(self) -> dict[str, tuple[ProfileMappingRow, ...]]:
        result: dict[str, list[ProfileMappingRow]] = {
            profile_id: [] for profile_id in PROFILE_ORDER
        }
        for row in self.rows:
            result[row.profile_id].append(row)
        normalized: dict[str, tuple[ProfileMappingRow, ...]] = {}
        for profile_id in PROFILE_ORDER:
            rows = tuple(result[profile_id])
            if len(rows) < 2 or tuple(row.sample_index for row in rows) != tuple(
                range(len(rows))
            ):
                raise NativeProfileConvergenceError(
                    f"{self.spacing_mm} mm profile {profile_id} does not have "
                    "complete ordered sample indices"
                )
            if any(
                right.distance_m <= left.distance_m
                for left, right in zip(rows, rows[1:])
            ):
                raise NativeProfileConvergenceError(
                    f"{self.spacing_mm} mm profile {profile_id} distances are not "
                    "strictly increasing"
                )
            normalized[profile_id] = rows
        return normalized


@dataclass(frozen=True)
class CaseResolutionMappings:
    """The exact four candidate grids for one native DrivAerML case."""

    case_id: str
    resolutions: tuple[ResolutionMapping, ...]

    def __post_init__(self) -> None:
        _case_id(self.case_id)
        if tuple(item.spacing_mm for item in self.resolutions) != SPACINGS_MM:
            raise NativeProfileConvergenceError(
                "case mappings must be ordered exactly [1,2,5,10] mm"
            )
        if any(item.case_id != self.case_id for item in self.resolutions):
            raise NativeProfileConvergenceError(
                "case mapping resolutions use inconsistent case IDs"
            )
        counts = {item.native_cell_count for item in self.resolutions}
        sources = {item.source_part_sha256 for item in self.resolutions}
        if len(counts) != 1 or len(sources) != 1:
            raise NativeProfileConvergenceError(
                "case mapping resolutions use inconsistent native sources"
            )

    @property
    def native_cell_count(self) -> int:
        return self.resolutions[0].native_cell_count

    @property
    def source_part_sha256(self) -> tuple[str, ...]:
        return self.resolutions[0].source_part_sha256

    def required_raw_cell_ids(self) -> np.ndarray:
        ids = np.asarray(
            [
                row.raw_vtk_cell_id
                for resolution in self.resolutions
                for row in resolution.rows
                if row.valid and row.raw_vtk_cell_id is not None
            ],
            dtype=np.int64,
        )
        result = np.unique(ids)
        result.setflags(write=False)
        return result


def validate_mapping_grids(
    case: CaseResolutionMappings,
    definition: AutoCFD5Definition,
) -> dict[str, object]:
    """Validate exact grids and report nested geometric assignment invariance.

    The 10 mm CSV stores rounded decimal coordinates.  Its points are accepted
    as nested only within the same 2e-11 m registry tolerance used by the
    AutoCFD5 loader.  Assignment identity itself is exact and includes the
    validity bit, invalidity reason, selected raw cell ID, and candidate count.
    """

    if not isinstance(definition, AutoCFD5Definition):
        raise NativeProfileConvergenceError(
            "definition must be a validated AutoCFD5Definition"
        )
    expected_by_spacing: dict[int, tuple[Any, ...]] = {}
    for spacing_mm in SPACINGS_MM:
        samples = (
            tuple(definition.velocity_samples)
            if spacing_mm == 10
            else generate_definition_velocity_samples(
                definition, spacing_mm / 1000.0
            )
        )
        if len(samples) != EXPECTED_SAMPLE_COUNTS[spacing_mm]:
            raise NativeProfileConvergenceError(
                f"registry-derived {spacing_mm} mm sample count drifted"
            )
        expected_by_spacing[spacing_mm] = samples

    by_spacing: dict[int, dict[str, tuple[ProfileMappingRow, ...]]] = {}
    for resolution in case.resolutions:
        expected = expected_by_spacing[resolution.spacing_mm]
        if len(resolution.rows) != len(expected):
            raise NativeProfileConvergenceError("mapping row count drifted")
        for position, (row, sample) in enumerate(
            zip(resolution.rows, expected, strict=True)
        ):
            if (
                (row.profile_id, row.sample_index)
                != (sample.profile_id, sample.sample_index)
                or row.distance_m != sample.distance_m
                or row.point_m != sample.point_m
            ):
                raise NativeProfileConvergenceError(
                    f"{case.case_id} {resolution.spacing_mm} mm mapping row "
                    f"{position} differs from the exact registry grid"
                )
        by_spacing[resolution.spacing_mm] = resolution.rows_by_profile()

    reference = by_spacing[1]
    comparisons: list[dict[str, object]] = []
    every_assignment_invariant = True
    for spacing_mm in SPACINGS_MM[1:]:
        stride = NESTED_STRIDES[spacing_mm]
        matched_count = 0
        assignment_difference_count = 0
        first_differences: list[dict[str, object]] = []
        max_point_difference = 0.0
        max_distance_difference = 0.0
        for profile_id in PROFILE_ORDER:
            fine = reference[profile_id]
            coarse = by_spacing[spacing_mm][profile_id]
            nested = fine[::stride]
            if len(nested) != len(coarse):
                raise NativeProfileConvergenceError(
                    f"{case.case_id} {profile_id} {spacing_mm} mm grid is not a "
                    f"stride-{stride} subset of 1 mm"
                )
            for coarse_row, fine_row in zip(coarse, nested, strict=True):
                point_difference = max(
                    abs(left - right)
                    for left, right in zip(coarse_row.point_m, fine_row.point_m)
                )
                distance_difference = abs(
                    coarse_row.distance_m - fine_row.distance_m
                )
                max_point_difference = max(max_point_difference, point_difference)
                max_distance_difference = max(
                    max_distance_difference, distance_difference
                )
                if (
                    point_difference > NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M
                    or distance_difference
                    > NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M
                ):
                    raise NativeProfileConvergenceError(
                        f"{case.case_id} {profile_id} sample "
                        f"{coarse_row.sample_index} is not geometrically nested"
                    )
                matched_count += 1
                if coarse_row.assignment_key() != fine_row.assignment_key():
                    assignment_difference_count += 1
                    if len(first_differences) < 16:
                        first_differences.append(
                            {
                                "profile_id": profile_id,
                                "coarse_sample_index": coarse_row.sample_index,
                                "reference_sample_index": fine_row.sample_index,
                                "reference_assignment": list(
                                    fine_row.assignment_key()
                                ),
                                "candidate_assignment": list(
                                    coarse_row.assignment_key()
                                ),
                            }
                        )
        invariant = assignment_difference_count == 0
        every_assignment_invariant &= invariant
        comparisons.append(
            {
                "spacing_mm": spacing_mm,
                "reference_stride": stride,
                "matched_nested_sample_count": matched_count,
                "assignment_difference_count": assignment_difference_count,
                "assignment_invariant": invariant,
                "maximum_point_coordinate_difference_m": max_point_difference,
                "maximum_arc_distance_difference_m": max_distance_difference,
                "coordinate_absolute_tolerance_m": (
                    NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M
                ),
                "first_assignment_differences": first_differences,
            }
        )
    return {
        "case_id": case.case_id,
        "grid_counts_exact": True,
        "grids_stride_nested": True,
        "native_cell_count": case.native_cell_count,
        "native_source_part_sha256": list(case.source_part_sha256),
        "mapping_artifacts": [
            {
                "spacing_mm": resolution.spacing_mm,
                "artifact_name": resolution.artifact_name,
                "artifact_sha256": resolution.artifact_sha256,
                "assignment_evidence_sha256": (
                    resolution.assignment_evidence_sha256
                ),
                "sample_count": len(resolution.rows),
            }
            for resolution in case.resolutions
        ],
        "assignment_comparisons": comparisons,
        "every_nested_geometric_assignment_invariant": (
            every_assignment_invariant
        ),
    }


def _require_sparse_native_truth(
    truth: SparseNativeField,
    case: CaseResolutionMappings,
    required_ids: np.ndarray,
) -> None:
    if not isinstance(truth, SparseNativeField):
        raise NativeProfileConvergenceError(
            "native truth must be a verified SparseNativeField"
        )
    if (
        truth.case_id != case.case_id
        or truth.support_id != SUPPORT_ID
        or truth.field_name != FIELD_NAME
        or truth.total_row_count != case.native_cell_count
        or truth.source_sha256 != case.source_part_sha256
        or truth.complete_source_identity_verified is not True
    ):
        raise NativeProfileConvergenceError(
            f"{case.case_id} native truth identity differs from its mappings"
        )
    if not np.array_equal(truth.raw_cell_ids, required_ids):
        raise NativeProfileConvergenceError(
            f"{case.case_id} native truth selection is not the exact union of "
            "valid mapped raw cell IDs"
        )
    if truth.values.shape != (len(required_ids), 3):
        raise NativeProfileConvergenceError(
            f"{case.case_id} native UMeanTrim selection must have shape [N,3]"
        )


def _require_prediction_gather(
    prediction: SparsePredictionGather,
    case: CaseResolutionMappings,
    required_ids: np.ndarray,
) -> None:
    if (
        prediction.case_id != case.case_id
        or prediction.support_id != SUPPORT_ID
        or prediction.field_name != FIELD_NAME
        or prediction.total_row_count != case.native_cell_count
        or not np.array_equal(prediction.raw_cell_ids, required_ids)
        or prediction.values.shape != (len(required_ids), 3)
    ):
        raise NativeProfileConvergenceError(
            f"{case.case_id} prediction gather does not exactly cover mapped "
            "native UMeanTrim values"
        )


def case_profile_losses(
    case: CaseResolutionMappings,
    *,
    definition: AutoCFD5Definition,
    native_truth: SparseNativeField,
    prediction: SparsePredictionGather,
    validated_geometry: Mapping[str, object] | None = None,
) -> tuple[list[list[float]], dict[str, object]]:
    """Compute all 16 x four losses for one method/case.

    The current candidate has no hash-bound owner validity mask.  Consequently
    every requested row must map: a raw invalid row is unresolved and cannot be
    converted into a partial ranked convergence loss.  Once a non-empty owner
    mask is published, it requires an explicit versioned binding that separates
    approved exclusions from locator failures before masked arcs can be ranked.
    """

    if validated_geometry is None:
        geometry = validate_mapping_grids(case, definition)
    else:
        if (
            validated_geometry.get("case_id") != case.case_id
            or validated_geometry.get("grid_counts_exact") is not True
            or validated_geometry.get("grids_stride_nested") is not True
        ):
            raise NativeProfileConvergenceError(
                "prevalidated geometry record is not exact for this case"
            )
        geometry = dict(validated_geometry)
    invalid_rows = [
        (resolution.spacing_mm, row.profile_id, row.sample_index, row.reason)
        for resolution in case.resolutions
        for row in resolution.rows
        if not row.valid
    ]
    if invalid_rows:
        spacing_mm, profile_id, sample_index, reason = invalid_rows[0]
        raise NativeProfileConvergenceError(
            f"{case.case_id} contains {len(invalid_rows)} unresolved velocity "
            "mapping rows while no immutable owner validity mask is bound; "
            "partial profile convergence is forbidden "
            f"(first={spacing_mm} mm/{profile_id}/{sample_index}: {reason})"
        )
    required_ids = case.required_raw_cell_ids()
    if len(required_ids) == 0:
        raise NativeProfileConvergenceError(
            f"{case.case_id} mappings contain no valid native cell assignment"
        )
    _require_sparse_native_truth(native_truth, case, required_ids)
    _require_prediction_gather(prediction, case, required_ids)

    truth_ratio_by_id = velocity_magnitude_ratio(native_truth.values)
    prediction_ratio_by_id = velocity_magnitude_ratio(prediction.values)
    position_by_id = {
        int(raw_id): position for position, raw_id in enumerate(required_ids)
    }
    losses_by_profile: dict[str, list[float]] = {
        profile_id: [] for profile_id in PROFILE_ORDER
    }
    line_support: list[dict[str, object]] = []
    for resolution in case.resolutions:
        by_profile = resolution.rows_by_profile()
        for profile_id in PROFILE_ORDER:
            rows = by_profile[profile_id]
            distance = np.asarray(
                [row.distance_m for row in rows], dtype=np.float64
            )
            valid = np.asarray([row.valid for row in rows], dtype=bool)
            prediction_ratio = np.full(len(rows), np.nan, dtype=np.float64)
            truth_ratio = np.full(len(rows), np.nan, dtype=np.float64)
            for position, row in enumerate(rows):
                if row.valid:
                    assert row.raw_vtk_cell_id is not None
                    selected = position_by_id[row.raw_vtk_cell_id]
                    prediction_ratio[position] = prediction_ratio_by_id[selected]
                    truth_ratio[position] = truth_ratio_by_id[selected]
            edge_mask = valid[:-1] & valid[1:]
            contributing_arc = math.fsum(
                float(value)
                for value in np.diff(distance)[edge_mask]
            )
            if not math.isfinite(contributing_arc) or contributing_arc <= 0.0:
                raise NativeProfileConvergenceError(
                    f"{case.case_id} {profile_id} {resolution.spacing_mm} mm "
                    "has no positive contributing arc length"
                )
            try:
                loss = velocity_profile_rmse(
                    distance,
                    prediction_ratio,
                    truth_ratio,
                    valid,
                )
            except AutoCFD5Error as error:
                raise NativeProfileConvergenceError(
                    f"{case.case_id} {profile_id} {resolution.spacing_mm} mm "
                    f"profile reduction failed: {error}"
                ) from error
            losses_by_profile[profile_id].append(loss)
            line_support.append(
                {
                    "profile_id": profile_id,
                    "spacing_mm": resolution.spacing_mm,
                    "sample_count": len(rows),
                    "valid_sample_count": int(np.count_nonzero(valid)),
                    "invalid_sample_count": int(np.count_nonzero(~valid)),
                    "contributing_edge_count": int(np.count_nonzero(edge_mask)),
                    "contributing_arc_length_m": contributing_arc,
                    "no_invalid_gap_bridging": True,
                    "loss": loss,
                }
            )
    losses = [losses_by_profile[profile_id] for profile_id in PROFILE_ORDER]
    if any(len(values) != len(SPACINGS_MM) for values in losses):
        raise NativeProfileConvergenceError(
            "profile loss construction did not cover all four resolutions"
        )
    return losses, {
        "case_id": case.case_id,
        "geometry": geometry,
        "required_unique_raw_cell_id_count": len(required_ids),
        "line_support": line_support,
    }


def _validate_prediction_binding_records(
    prediction_bindings: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[Mapping[str, object]],
    cases: Sequence[str],
) -> dict[tuple[str, str], dict[str, object]]:
    if not isinstance(prediction_bindings, (list, tuple)) or len(
        prediction_bindings
    ) != len(methods):
        raise NativeProfileConvergenceError(
            "prediction bindings must contain exactly one ordered record per method"
        )
    result: dict[tuple[str, str], dict[str, object]] = {}
    for method_position, (raw_binding, method) in enumerate(
        zip(prediction_bindings, methods, strict=True)
    ):
        binding = _exact_mapping(
            raw_binding,
            {
                "method_id",
                "role",
                "prediction_artifact_sha256",
                "case_manifests",
            },
            f"prediction binding {method_position}",
        )
        method_id = method["method_id"]
        if (
            binding["method_id"] != method_id
            or binding["role"] != method["role"]
            or _sha256(
                binding["prediction_artifact_sha256"],
                f"prediction binding {method_id!r} artifact SHA-256",
            )
            != method["prediction_artifact_sha256"]
        ):
            raise NativeProfileConvergenceError(
                f"prediction binding {method_id!r} differs from its method declaration"
            )
        raw_cases = binding["case_manifests"]
        if not isinstance(raw_cases, list) or len(raw_cases) != len(cases):
            raise NativeProfileConvergenceError(
                f"prediction binding {method_id!r} does not cover case_order"
            )
        for case_position, (raw_case, case_id) in enumerate(
            zip(raw_cases, cases, strict=True)
        ):
            record = _exact_mapping(
                raw_case,
                {
                    "case_id",
                    "manifest_file",
                    "manifest_sha256",
                    "native_cell_count",
                    "chunk_count",
                    "ordered_chunk_sha256",
                    "complete_native_cell_coverage_declared",
                },
                f"prediction binding {method_id!r} case {case_position}",
            )
            if record["case_id"] != case_id:
                raise NativeProfileConvergenceError(
                    f"prediction binding {method_id!r} case order differs"
                )
            _basename(record["manifest_file"], "prediction manifest file")
            _sha256(record["manifest_sha256"], "prediction manifest SHA-256")
            _positive_integer(record["native_cell_count"], "native cell count")
            chunk_count = _positive_integer(
                record["chunk_count"], "prediction chunk count"
            )
            chunk_sha256 = record["ordered_chunk_sha256"]
            if not isinstance(chunk_sha256, list) or len(chunk_sha256) != chunk_count:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} ordered chunk hashes are incomplete"
                )
            for chunk_position, digest in enumerate(chunk_sha256):
                _sha256(
                    digest,
                    f"{method_id}/{case_id} chunk {chunk_position} SHA-256",
                )
            if record["complete_native_cell_coverage_declared"] is not True:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} manifest does not declare full coverage"
                )
            result[(str(method_id), case_id)] = dict(record)
        if canonical_sha256(raw_cases) != binding["prediction_artifact_sha256"]:
            raise NativeProfileConvergenceError(
                f"prediction binding {method_id!r} artifact digest does not "
                "match its ordered case-manifest records"
            )
    return result


def _validate_geometric_case_records(
    records: Sequence[Mapping[str, object]],
    *,
    cases: Sequence[str],
) -> tuple[list[dict[str, object]], bool]:
    if not isinstance(records, (list, tuple)) or len(records) != len(cases):
        raise NativeProfileConvergenceError(
            "geometric evidence must contain exactly one ordered record per case"
        )
    normalized: list[dict[str, object]] = []
    all_invariant = True
    for case_position, (raw_record, case_id) in enumerate(
        zip(records, cases, strict=True)
    ):
        record = _exact_mapping(
            raw_record,
            {
                "case_id",
                "grid_counts_exact",
                "grids_stride_nested",
                "native_cell_count",
                "native_source_part_sha256",
                "mapping_artifacts",
                "assignment_comparisons",
                "every_nested_geometric_assignment_invariant",
            },
            f"geometric evidence case {case_position}",
        )
        if (
            record["case_id"] != case_id
            or record["grid_counts_exact"] is not True
            or record["grids_stride_nested"] is not True
        ):
            raise NativeProfileConvergenceError(
                f"geometric evidence for {case_id} is not exact"
            )
        _positive_integer(record["native_cell_count"], "native cell count")
        source_hashes = record["native_source_part_sha256"]
        if not isinstance(source_hashes, list) or not source_hashes:
            raise NativeProfileConvergenceError(
                f"geometric evidence for {case_id} lacks source hashes"
            )
        for source_position, digest in enumerate(source_hashes):
            _sha256(digest, f"{case_id} source part {source_position} SHA-256")

        raw_artifacts = record["mapping_artifacts"]
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(
            SPACINGS_MM
        ):
            raise NativeProfileConvergenceError(
                f"geometric evidence for {case_id} lacks four mapping artifacts"
            )
        for artifact_position, (raw_artifact, spacing_mm) in enumerate(
            zip(raw_artifacts, SPACINGS_MM, strict=True)
        ):
            artifact = _exact_mapping(
                raw_artifact,
                {
                    "spacing_mm",
                    "artifact_name",
                    "artifact_sha256",
                    "assignment_evidence_sha256",
                    "sample_count",
                },
                f"{case_id} mapping artifact {artifact_position}",
            )
            if (
                artifact["spacing_mm"] != spacing_mm
                or artifact["artifact_name"] != MAPPING_ARTIFACT_NAMES[spacing_mm]
                or artifact["sample_count"] != EXPECTED_SAMPLE_COUNTS[spacing_mm]
            ):
                raise NativeProfileConvergenceError(
                    f"{case_id} mapping artifact order or coverage differs"
                )
            _basename(artifact["artifact_name"], "mapping artifact name")
            _sha256(artifact["artifact_sha256"], "mapping artifact SHA-256")
            _sha256(
                artifact["assignment_evidence_sha256"],
                "mapping assignment-evidence SHA-256",
            )

        raw_comparisons = record["assignment_comparisons"]
        if not isinstance(raw_comparisons, list) or len(raw_comparisons) != 3:
            raise NativeProfileConvergenceError(
                f"geometric evidence for {case_id} lacks three nested comparisons"
            )
        comparison_invariants: list[bool] = []
        for comparison_position, (raw_comparison, spacing_mm) in enumerate(
            zip(raw_comparisons, SPACINGS_MM[1:], strict=True)
        ):
            comparison = _exact_mapping(
                raw_comparison,
                {
                    "spacing_mm",
                    "reference_stride",
                    "matched_nested_sample_count",
                    "assignment_difference_count",
                    "assignment_invariant",
                    "maximum_point_coordinate_difference_m",
                    "maximum_arc_distance_difference_m",
                    "coordinate_absolute_tolerance_m",
                    "first_assignment_differences",
                },
                f"{case_id} nested comparison {comparison_position}",
            )
            difference_count = comparison["assignment_difference_count"]
            if (
                comparison["spacing_mm"] != spacing_mm
                or comparison["reference_stride"] != NESTED_STRIDES[spacing_mm]
                or comparison["matched_nested_sample_count"]
                != EXPECTED_SAMPLE_COUNTS[spacing_mm]
                or not isinstance(difference_count, int)
                or isinstance(difference_count, bool)
                or difference_count < 0
                or comparison["assignment_invariant"]
                is not (difference_count == 0)
                or comparison["coordinate_absolute_tolerance_m"]
                != NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M
            ):
                raise NativeProfileConvergenceError(
                    f"{case_id} nested comparison {spacing_mm} mm is inconsistent"
                )
            for key in (
                "maximum_point_coordinate_difference_m",
                "maximum_arc_distance_difference_m",
            ):
                maximum = _finite_nonnegative(
                    comparison[key], f"{case_id} {spacing_mm} mm {key}"
                )
                if maximum > NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M:
                    raise NativeProfileConvergenceError(
                        f"{case_id} nested comparison exceeds coordinate tolerance"
                    )
            differences = comparison["first_assignment_differences"]
            if (
                not isinstance(differences, list)
                or len(differences) != min(difference_count, 16)
            ):
                raise NativeProfileConvergenceError(
                    f"{case_id} nested assignment differences are incomplete"
                )
            comparison_invariants.append(difference_count == 0)
        invariant = all(comparison_invariants)
        if record["every_nested_geometric_assignment_invariant"] is not invariant:
            raise NativeProfileConvergenceError(
                f"{case_id} aggregate geometric-invariance flag is inconsistent"
            )
        all_invariant &= invariant
        normalized.append(dict(record))
    return normalized, all_invariant


def _validate_method_audit_records(
    records: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[Mapping[str, object]],
    cases: Sequence[str],
    losses: Sequence[Sequence[Sequence[Sequence[float]]]],
    geometry: Sequence[Mapping[str, object]],
    prediction_records: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(records, (list, tuple)) or len(records) != len(methods):
        raise NativeProfileConvergenceError(
            "loss-construction audit must contain one ordered record per method"
        )
    normalized: list[dict[str, object]] = []
    native_truth_identity_by_case: dict[str, str] = {}
    line_support_geometry_by_case: dict[str, str] = {}
    require_native_payload_sha256 = tuple(cases) == OFFICIAL_CASE_ORDER
    for method_position, (raw_method_audit, method) in enumerate(
        zip(records, methods, strict=True)
    ):
        method_id = str(method["method_id"])
        method_audit = _exact_mapping(
            raw_method_audit,
            {"method_id", "case_count", "cases"},
            f"loss audit method {method_position}",
        )
        raw_cases = method_audit["cases"]
        if (
            method_audit["method_id"] != method_id
            or method_audit["case_count"] != len(cases)
            or not isinstance(raw_cases, list)
            or len(raw_cases) != len(cases)
        ):
            raise NativeProfileConvergenceError(
                f"loss audit for {method_id!r} does not cover case_order"
            )
        for case_position, (raw_case_audit, case_id, geometry_record) in enumerate(
            zip(raw_cases, cases, geometry, strict=True)
        ):
            case_audit = _exact_mapping(
                raw_case_audit,
                {
                    "case_id",
                    "geometry",
                    "required_unique_raw_cell_id_count",
                    "line_support",
                    "prediction_manifest",
                    "native_truth",
                },
                f"loss audit {method_id}/{case_id}",
            )
            if (
                case_audit["case_id"] != case_id
                or canonical_sha256(case_audit["geometry"])
                != canonical_sha256(geometry_record)
            ):
                raise NativeProfileConvergenceError(
                    f"loss audit {method_id}/{case_id} geometry binding differs"
                )
            required_count = _positive_integer(
                case_audit["required_unique_raw_cell_id_count"],
                f"{method_id}/{case_id} required mapped ID count",
            )
            if required_count > geometry_record["native_cell_count"]:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} required mapped ID count exceeds "
                    "the native cell count"
                )
            prediction = _exact_mapping(
                case_audit["prediction_manifest"],
                {
                    "support_id",
                    "field_name",
                    "total_row_count",
                    "manifest_file",
                    "manifest_sha256",
                    "chunk_count",
                    "chunk_sha256",
                    "complete_gap_free_duplicate_free_coverage",
                    "selected_unique_raw_cell_id_count",
                    "selected_values_sha256",
                },
                f"{method_id}/{case_id} prediction audit",
            )
            bound_prediction = prediction_records[(method_id, case_id)]
            if (
                prediction["support_id"] != SUPPORT_ID
                or prediction["field_name"] != FIELD_NAME
                or prediction["total_row_count"] != geometry_record["native_cell_count"]
                or bound_prediction["native_cell_count"]
                != geometry_record["native_cell_count"]
                or prediction["manifest_file"] != bound_prediction["manifest_file"]
                or prediction["manifest_sha256"]
                != bound_prediction["manifest_sha256"]
                or prediction["chunk_count"] != bound_prediction["chunk_count"]
                or prediction["chunk_sha256"]
                != bound_prediction["ordered_chunk_sha256"]
                or prediction["complete_gap_free_duplicate_free_coverage"] is not True
                or prediction["selected_unique_raw_cell_id_count"] != required_count
            ):
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} prediction audit is not bound to its manifest"
                )
            _sha256(
                prediction["selected_values_sha256"],
                f"{method_id}/{case_id} prediction selected-values audit SHA-256",
            )

            truth = _exact_mapping(
                case_audit["native_truth"],
                {
                    "support_id",
                    "field_name",
                    "total_row_count",
                    "selected_unique_raw_cell_id_count",
                    "selected_values_sha256",
                    "source_files",
                    "source_sha256",
                    "source_payload_sha256",
                    "complete_source_identity_verified",
                },
                f"{method_id}/{case_id} native truth audit",
            )
            source_files = truth["source_files"]
            source_hashes = truth["source_sha256"]
            if (
                truth["support_id"] != SUPPORT_ID
                or truth["field_name"] != FIELD_NAME
                or truth["total_row_count"] != geometry_record["native_cell_count"]
                or truth["selected_unique_raw_cell_id_count"] != required_count
                or not isinstance(source_files, list)
                or not isinstance(source_hashes, list)
                or len(source_files) != len(source_hashes)
                or source_hashes != geometry_record["native_source_part_sha256"]
                or truth["complete_source_identity_verified"] is not True
            ):
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} native truth audit is not source-bound"
                )
            for source_position, (source_file, digest) in enumerate(
                zip(source_files, source_hashes, strict=True)
            ):
                _basename(source_file, "native source file")
                _sha256(
                    digest,
                    f"{method_id}/{case_id} source {source_position} SHA-256",
                )
            _sha256(
                truth["selected_values_sha256"],
                f"{method_id}/{case_id} truth selected-values audit SHA-256",
            )
            if truth["source_payload_sha256"] is not None:
                _sha256(
                    truth["source_payload_sha256"],
                    f"{method_id}/{case_id} truth payload SHA-256",
                )
            elif require_native_payload_sha256:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} exact official evidence requires "
                    "a native truth payload SHA-256"
                )
            truth_identity = canonical_sha256(truth)
            previous_truth_identity = native_truth_identity_by_case.setdefault(
                case_id, truth_identity
            )
            if truth_identity != previous_truth_identity:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} native truth identity differs "
                    "across methods"
                )

            line_support = case_audit["line_support"]
            if not isinstance(line_support, list) or len(line_support) != (
                len(PROFILE_ORDER) * len(SPACINGS_MM)
            ):
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} line-support audit is incomplete"
                )
            support_position = 0
            support_geometry: list[dict[str, object]] = []
            for spacing_position, spacing_mm in enumerate(SPACINGS_MM):
                for profile_position, profile_id in enumerate(PROFILE_ORDER):
                    support = _exact_mapping(
                        line_support[support_position],
                        {
                            "profile_id",
                            "spacing_mm",
                            "sample_count",
                            "valid_sample_count",
                            "invalid_sample_count",
                            "contributing_edge_count",
                            "contributing_arc_length_m",
                            "no_invalid_gap_bridging",
                            "loss",
                        },
                        f"{method_id}/{case_id} line support {support_position}",
                    )
                    sample_count = _positive_integer(
                        support["sample_count"], "profile sample count"
                    )
                    valid_count = support["valid_sample_count"]
                    invalid_count = support["invalid_sample_count"]
                    edge_count = support["contributing_edge_count"]
                    expected_loss = float(
                        losses[method_position][case_position][profile_position][
                            spacing_position
                        ]
                    )
                    if (
                        support["profile_id"] != profile_id
                        or support["spacing_mm"] != spacing_mm
                        or not isinstance(valid_count, int)
                        or isinstance(valid_count, bool)
                        or not isinstance(invalid_count, int)
                        or isinstance(invalid_count, bool)
                        or valid_count < 0
                        or invalid_count < 0
                        or valid_count + invalid_count != sample_count
                        or not isinstance(edge_count, int)
                        or isinstance(edge_count, bool)
                        or edge_count < 1
                        or edge_count >= sample_count
                        or _finite_nonnegative(
                            support["contributing_arc_length_m"],
                            "contributing arc length",
                        )
                        <= 0.0
                        or support["no_invalid_gap_bridging"] is not True
                        or _finite_nonnegative(support["loss"], "profile loss")
                        != expected_loss
                    ):
                        raise NativeProfileConvergenceError(
                            f"{method_id}/{case_id} line support {support_position} "
                            "is inconsistent with the constructed loss tensor"
                        )
                    support_geometry.append(
                        {
                            key: support[key]
                            for key in (
                                "profile_id",
                                "spacing_mm",
                                "sample_count",
                                "valid_sample_count",
                                "invalid_sample_count",
                                "contributing_edge_count",
                                "contributing_arc_length_m",
                                "no_invalid_gap_bridging",
                            )
                        }
                    )
                    support_position += 1
            support_geometry_identity = canonical_sha256(support_geometry)
            previous_support_geometry_identity = (
                line_support_geometry_by_case.setdefault(
                    case_id, support_geometry_identity
                )
            )
            if support_geometry_identity != previous_support_geometry_identity:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} line-support geometry differs "
                    "across methods"
                )
        normalized.append(dict(method_audit))
    return normalized


def finalize_profile_convergence_evidence(
    *,
    study_id: str,
    method_set: Mapping[str, object],
    case_order: Sequence[str],
    losses: Sequence[Sequence[Sequence[Sequence[float]]]],
    prediction_bindings: Sequence[Mapping[str, object]],
    geometric_case_evidence: Sequence[Mapping[str, object]],
    method_audits: Sequence[Mapping[str, object]],
    contract_proposal: str | Path,
) -> dict[str, object]:
    """Finalize separately constructed geometry and loss evidence.

    This is the bounded-case-memory entry point used by the real-data CLI.  It
    deliberately accepts only already-computed finite losses and path-free
    audit records; the existing convergence checker revalidates the complete
    rectangular tensor and the exact method declarations.
    """

    if not isinstance(study_id, str) or not study_id:
        raise NativeProfileConvergenceError("study_id must be a non-empty string")
    cases = tuple(_case_id(item, "case_order entry") for item in case_order)
    if not cases or len(cases) != len(set(cases)):
        raise NativeProfileConvergenceError(
            "case_order must be non-empty and duplicate-free"
        )
    if not isinstance(method_set, Mapping):
        raise NativeProfileConvergenceError("method_set must be an object")
    loss_input: dict[str, object] = {
        "schema": INPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "method_set": dict(method_set),
        "case_order": list(cases),
        "profile_order": list(PROFILE_ORDER),
        "spacings_mm": list(SPACINGS_MM),
        "losses": losses,
    }
    try:
        convergence = evaluate_profile_convergence(
            loss_input, contract_proposal=contract_proposal
        )
    except ProfileConvergenceError as error:
        raise NativeProfileConvergenceError(str(error)) from error
    raw_methods = method_set.get("methods")
    if not isinstance(raw_methods, list) or any(
        not isinstance(item, Mapping) for item in raw_methods
    ):
        raise NativeProfileConvergenceError(
            "validated method_set.methods must remain an ordered object array"
        )
    methods = [dict(item) for item in raw_methods]
    prediction_records = _validate_prediction_binding_records(
        prediction_bindings,
        methods=methods,
        cases=cases,
    )
    geometry_records, geometry_invariant = _validate_geometric_case_records(
        geometric_case_evidence,
        cases=cases,
    )
    validated_method_audits = _validate_method_audit_records(
        method_audits,
        methods=methods,
        cases=cases,
        losses=losses,
        geometry=geometry_records,
        prediction_records=prediction_records,
    )
    numerical_loss_convergence_passed = bool(
        convergence["selection"][
            "all_prescribed_candidate_spacings_passed"
        ]
    )
    complete_official_scope = bool(
        convergence["scope"]["complete_official_case_order"]
    )
    method_requirements_passed = bool(
        convergence["method_set"]["requirements_passed"]
    )
    owner_validity_mask_bound = bool(
        convergence["scope"]["owner_validity_mask_bound"]
    )
    combined_eligible = (
        geometry_invariant
        and numerical_loss_convergence_passed
        and complete_official_scope
        and method_requirements_passed
        and owner_validity_mask_bound
    )
    blocking_reasons: list[str] = []
    if not complete_official_scope:
        blocking_reasons.append("incomplete_official_484_case_scope")
    if not geometry_invariant:
        blocking_reasons.append("nested_geometric_assignment_changed")
    if not method_requirements_passed:
        blocking_reasons.append(
            "missing_or_unpinned_genuine_method_predictions"
        )
    if not numerical_loss_convergence_passed:
        blocking_reasons.append("profile_loss_or_method_order_convergence_failed")
    if not owner_validity_mask_bound:
        blocking_reasons.append("owner_validity_mask_not_bound")
    if combined_eligible:
        status = "eligible_for_owner_review_not_active"
    elif not complete_official_scope:
        status = "pilot_or_incomplete_scope_not_eligible_for_owner_review"
    elif not geometry_invariant:
        status = "ineligible_nested_geometric_assignment_changed"
    elif not method_requirements_passed:
        status = "blocked_missing_or_unpinned_genuine_method_predictions"
    elif not numerical_loss_convergence_passed:
        status = "ineligible_profile_loss_or_method_order_convergence_failed"
    else:
        status = "blocked_owner_validity_mask_not_bound"
    result = {
        "schema": CONSTRUCTED_EVIDENCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "profile_resolution_candidate_eligible_for_owner_review": combined_eligible,
        "claims": dict(FALSE_CLAIMS),
        "construction": {
            "association": ASSOCIATION,
            "support_id": SUPPORT_ID,
            "field_name": FIELD_NAME,
            "native_field_units": "m/s",
            "quantity": "magnitude(UMeanTrim)/Uinf",
            "Uinf_m_per_s": U_INF_M_PER_S,
            "case_order": list(cases),
            "required_official_case_count": len(OFFICIAL_CASE_ORDER),
            "complete_official_case_order": complete_official_scope,
            "profile_order": list(PROFILE_ORDER),
            "spacings_mm": list(SPACINGS_MM),
            "bounded_memory_prediction_access": True,
            "bounded_memory_case_access_supported": True,
            "complete_prediction_manifests_exhausted": True,
            "validity_policy_id": (
                "drivaerml-autocfd5-velocity-validity-v1"
            ),
            "owner_mask_status": (
                "pending_immutable_owner_release_no_exclusions_approved"
            ),
            "invalid_mapping_action": (
                "reject_partial_convergence_until_hash_bound_owner_mask_exists"
            ),
            "loss_input_canonical_json_sha256": canonical_sha256(loss_input),
        },
        "prediction_artifacts": [dict(item) for item in prediction_bindings],
        "geometric_assignment_convergence": {
            "passed": geometry_invariant,
            "distinct_from_loss_and_method_order_convergence": True,
            "cases": geometry_records,
        },
        "loss_and_method_order_convergence": {
            "passed": numerical_loss_convergence_passed,
            "distinct_from_scope_and_method_set_eligibility": True,
            "evidence": convergence,
        },
        "loss_construction_audit": validated_method_audits,
        "constructed_loss_input": loss_input,
    }
    _assert_no_absolute_paths(result)
    return result


def prediction_manifest_set_identity(
    case_order: Sequence[str],
    manifests: Sequence[PredictionChunkManifest | str | Path],
) -> tuple[str, tuple[PredictionChunkManifest, ...], list[dict[str, object]]]:
    """Bind one method to its ordered complete per-case manifest identities."""

    cases = tuple(_case_id(value, "case_order entry") for value in case_order)
    if len(cases) == 0 or len(cases) != len(set(cases)):
        raise NativeProfileConvergenceError(
            "case_order must be non-empty and duplicate-free"
        )
    if len(manifests) != len(cases):
        raise NativeProfileConvergenceError(
            "prediction manifest set must contain one manifest per case"
        )
    parsed: list[PredictionChunkManifest] = []
    records: list[dict[str, object]] = []
    for case_id, value in zip(cases, manifests, strict=True):
        manifest = (
            load_prediction_chunk_manifest(value.path)
            if isinstance(value, PredictionChunkManifest)
            else load_prediction_chunk_manifest(value)
        )
        if (
            manifest.case_id != case_id
            or manifest.support_id != SUPPORT_ID
            or manifest.association != ASSOCIATION
            or manifest.field_components.get(FIELD_NAME) != 3
        ):
            raise NativeProfileConvergenceError(
                f"{case_id} prediction manifest does not describe native "
                "CellData UMeanTrim"
            )
        parsed.append(manifest)
        records.append(
            {
                "case_id": case_id,
                "manifest_file": manifest.path.name,
                "manifest_sha256": manifest.sha256,
                "native_cell_count": manifest.total_row_count,
                "chunk_count": len(manifest.chunks),
                "ordered_chunk_sha256": [
                    item.sha256 for item in manifest.chunks
                ],
                "complete_native_cell_coverage_declared": True,
            }
        )
    digest = canonical_sha256(records)
    return digest, tuple(parsed), records


def build_profile_convergence_from_predictions(
    *,
    study_id: str,
    method_set: Mapping[str, object],
    case_order: Sequence[str],
    case_mappings: Mapping[str, CaseResolutionMappings],
    native_truth: Mapping[str, SparseNativeField],
    prediction_manifests: Mapping[
        str, Sequence[PredictionChunkManifest | str | Path]
    ],
    definition: AutoCFD5Definition,
    contract_proposal: str | Path,
    maximum_prediction_chunk_rows: int | None = 1_000_000,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
) -> dict[str, object]:
    """Construct losses from genuine artifacts and run the frozen checker."""

    if not isinstance(study_id, str) or not study_id:
        raise NativeProfileConvergenceError("study_id must be a non-empty string")
    cases = tuple(_case_id(item, "case_order entry") for item in case_order)
    if not cases or len(cases) != len(set(cases)):
        raise NativeProfileConvergenceError(
            "case_order must be non-empty and duplicate-free"
        )
    if set(case_mappings) != set(cases) or set(native_truth) != set(cases):
        raise NativeProfileConvergenceError(
            "mapping and native-truth case sets must exactly equal case_order"
        )
    if not isinstance(method_set, Mapping):
        raise NativeProfileConvergenceError("method_set must be an object")
    if set(method_set) != {"pinned_before_study", "sha256", "methods"}:
        raise NativeProfileConvergenceError("method_set keys differ from schema")
    raw_methods = method_set["methods"]
    if not isinstance(raw_methods, list) or not raw_methods:
        raise NativeProfileConvergenceError("method_set.methods cannot be empty")
    if method_set_sha256(raw_methods) != _sha256(
        method_set["sha256"], "method_set.sha256"
    ):
        raise NativeProfileConvergenceError(
            "method_set.sha256 does not match ordered method declarations"
        )
    method_ids: list[str] = []
    for index, method in enumerate(raw_methods):
        if not isinstance(method, Mapping):
            raise NativeProfileConvergenceError(
                f"method_set.methods[{index}] must be an object"
            )
        method_id = method.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            raise NativeProfileConvergenceError(
                f"method_set.methods[{index}].method_id must be non-empty"
            )
        method_ids.append(method_id)
    if len(method_ids) != len(set(method_ids)) or set(prediction_manifests) != set(
        method_ids
    ):
        raise NativeProfileConvergenceError(
            "prediction-manifest method set must exactly match unique method IDs"
        )

    parsed_by_method: dict[str, tuple[PredictionChunkManifest, ...]] = {}
    prediction_bindings: list[dict[str, object]] = []
    for method, method_id in zip(raw_methods, method_ids, strict=True):
        digest, parsed, records = prediction_manifest_set_identity(
            cases, prediction_manifests[method_id]
        )
        declared = _sha256(
            method.get("prediction_artifact_sha256"),
            f"method {method_id!r} prediction_artifact_sha256",
        )
        if digest != declared:
            raise NativeProfileConvergenceError(
                f"method {method_id!r} prediction artifact pin mismatch: "
                f"declared {declared}, actual {digest}"
            )
        for case_id, manifest in zip(cases, parsed, strict=True):
            if manifest.total_row_count != case_mappings[case_id].native_cell_count:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} prediction native cell count differs "
                    "from the receipt-bound mapping"
                )
        parsed_by_method[method_id] = parsed
        prediction_bindings.append(
            {
                "method_id": method_id,
                "role": method.get("role"),
                "prediction_artifact_sha256": digest,
                "case_manifests": records,
            }
        )

    geometry_by_case = {
        case_id: validate_mapping_grids(case_mappings[case_id], definition)
        for case_id in cases
    }
    losses: list[list[list[list[float]]]] = []
    method_audits: list[dict[str, object]] = []
    for method_index, method_id in enumerate(method_ids):
        method_losses: list[list[list[float]]] = []
        case_audits: list[dict[str, object]] = []
        for case_index, case_id in enumerate(cases):
            mapping = case_mappings[case_id]
            required_ids = mapping.required_raw_cell_ids()
            try:
                prediction = gather_mapped_prediction_field(
                    parsed_by_method[method_id][case_index],
                    required_ids,
                    case_id=case_id,
                    support_id=SUPPORT_ID,
                    field_name=FIELD_NAME,
                    expected_total_row_count=mapping.native_cell_count,
                    maximum_chunk_rows=maximum_prediction_chunk_rows,
                    hash_chunk_bytes=hash_chunk_bytes,
                    validation_block_rows=validation_block_rows,
                )
            except DrivAerDiagnosticEvaluatorError as error:
                raise NativeProfileConvergenceError(
                    f"{method_id}/{case_id} complete prediction gather failed: "
                    f"{error}"
                ) from error
            case_losses, audit = case_profile_losses(
                mapping,
                definition=definition,
                native_truth=native_truth[case_id],
                prediction=prediction,
                validated_geometry=geometry_by_case[case_id],
            )
            method_losses.append(case_losses)
            audit["prediction_manifest"] = prediction.audit_record()
            audit["native_truth"] = native_truth[case_id].audit_record()
            case_audits.append(audit)
        losses.append(method_losses)
        method_audits.append(
            {
                "method_id": method_id,
                "case_count": len(cases),
                "cases": case_audits,
            }
        )

    return finalize_profile_convergence_evidence(
        study_id=study_id,
        method_set=method_set,
        case_order=cases,
        losses=losses,
        prediction_bindings=prediction_bindings,
        geometric_case_evidence=[geometry_by_case[case_id] for case_id in cases],
        method_audits=method_audits,
        contract_proposal=contract_proposal,
    )


__all__ = [
    "ASSOCIATION",
    "CONSTRUCTED_EVIDENCE_SCHEMA",
    "CaseResolutionMappings",
    "EXPECTED_SAMPLE_COUNTS",
    "FALSE_CLAIMS",
    "FIELD_NAME",
    "MAPPING_ARTIFACT_NAMES",
    "NESTED_COORDINATE_ABSOLUTE_TOLERANCE_M",
    "NESTED_STRIDES",
    "NativeProfileConvergenceError",
    "PREDICTION_STUDY_SCHEMA",
    "ProfileMappingRow",
    "ResolutionMapping",
    "SCHEMA_VERSION",
    "SUPPORT_ID",
    "build_profile_convergence_from_predictions",
    "case_profile_losses",
    "finalize_profile_convergence_evidence",
    "prediction_manifest_set_identity",
    "validate_mapping_grids",
]
