"""Deterministic, zero-weight regional diagnostics for native DrivAerML fields.

This module deliberately has no connection to :mod:`autocfd5_aiml.scores`.
It partitions already required native-cell predictions while they are streamed
through the evaluator and retains additive report statistics.  The official
whole-field accumulators remain the sole source of scored field metrics.

Two immutable four-region partitions are defined:

* native surface polygons use the OpenFOAM face-centre ``z`` coordinate and
  the absolute ``z`` component of the native-order unit normal;
* native volume cells use the retained VTK cell centre in the native order.

The names are intentionally geometric.  They must not be presented as audited
OpenFOAM patches or as exact semantic underfloor, wheel, roof, or wake masks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

REGIONAL_REPORT_SCHEMA = "autocfd5-aiml-regional-diagnostics-v1"
REGIONAL_REPORT_SCHEMA_VERSION = 1
REGIONAL_REPORT_STATUS = "complete_report_only"
REGIONAL_SCORING_WEIGHT = 0.0

SURFACE_Z_THRESHOLD_M = 0.75
SURFACE_HORIZONTAL_NORMAL_ABS_Z_MIN = 0.5
VOLUME_X_BODY_MIN_M = -0.85
VOLUME_X_BODY_MAX_M = 3.65
VOLUME_BODY_ABS_Y_MAX_M = 1.25
VOLUME_Z_LOW_MAX_M = 0.75
VOLUME_Z_UPPER_MAX_M = 2.0
VOLUME_X_WAKE_MAX_M = 6.0
VOLUME_WAKE_ABS_Y_MAX_M = 2.0
VOLUME_WAKE_Z_MAX_M = 2.5
U_INF_M_PER_S = 38.889
VELOCITY_DIRECTION_MIN_SPEED_M_PER_S = 0.05 * U_INF_M_PER_S

DEFAULT_RECONSTRUCTION_RTOL = 5.0e-12
DEFAULT_RECONSTRUCTION_ATOL = 1.0e-12
_NORMAL_BOUND_TOLERANCE = 2.0e-12
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID_RE = re.compile(r"run_[1-9][0-9]*\Z")
_ASSIGNMENT_ENCODING = (
    "domain-prefixed uint8 region codes in zero-based native entity order"
)


class RegionalDiagnosticError(ValueError):
    """Raised when a regional partition or report fails closed validation."""


@dataclass(frozen=True)
class RegionRule:
    """One stable region code, identifier, and human-readable predicate."""

    code: int
    region_id: str
    predicate: str

    def to_json(self) -> dict[str, object]:
        return {
            "code": self.code,
            "region_id": self.region_id,
            "predicate": self.predicate,
        }


@dataclass(frozen=True)
class RegionDefinition:
    """Immutable scientific definition of one exhaustive regional partition."""

    definition_id: str
    support_id: str
    coordinate_frame: str
    coordinate_method: str
    coordinate_unit: str
    rules: tuple[RegionRule, ...]
    semantic_limit: str

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(rule.region_id for rule in self.rules)

    def to_json(self) -> dict[str, object]:
        return {
            "definition_id": self.definition_id,
            "support_id": self.support_id,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_method": self.coordinate_method,
            "coordinate_unit": self.coordinate_unit,
            "regions_in_code_order": [rule.to_json() for rule in self.rules],
            "partition_properties": "mutually_exclusive_and_exhaustive",
            "semantic_limit": self.semantic_limit,
            "scoring_role": "report_only_zero_weight",
            "scoring_weight": REGIONAL_SCORING_WEIGHT,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_json())).hexdigest()


SURFACE_REGION_DEFINITION = RegionDefinition(
    definition_id="drivaerml-surface-four-geometric-regions-v1",
    support_id="surface_native_cells",
    coordinate_frame="raw_native_VTP_coordinates",
    coordinate_method=(
        "OpenFOAM_v2212_primitiveMeshTools_makeFaceCentresAndAreas; "
        "signed unit normal is native-order oriented area vector divided by "
        "its calculated magnitude"
    ),
    coordinate_unit="m",
    rules=(
        RegionRule(
            0,
            "low_z_horizontal_normal",
            "face_centre_z_m < 0.75 and abs(signed_unit_normal_z) >= 0.5",
        ),
        RegionRule(
            1,
            "low_z_other_normal",
            "face_centre_z_m < 0.75 and abs(signed_unit_normal_z) < 0.5",
        ),
        RegionRule(
            2,
            "high_z_horizontal_normal",
            "face_centre_z_m >= 0.75 and abs(signed_unit_normal_z) >= 0.5",
        ),
        RegionRule(
            3,
            "high_z_other_normal",
            "face_centre_z_m >= 0.75 and abs(signed_unit_normal_z) < 0.5",
        ),
    ),
    semantic_limit=(
        "geometric orientation bins only; not OpenFOAM patches and not "
        "validated underfloor, wheel, roof, or base semantic labels"
    ),
)

VOLUME_REGION_DEFINITION = RegionDefinition(
    definition_id="drivaerml-volume-four-geometric-regions-v1",
    support_id="volume_native_cells",
    coordinate_frame="raw_native_VTU_coordinates",
    coordinate_method="VTK cell parametric centre in native GetCell order",
    coordinate_unit="m",
    rules=(
        RegionRule(
            0,
            "underbody_and_wheels",
            "-0.85 <= x < 3.65 and abs(y) < 1.25 and z < 0.75",
        ),
        RegionRule(
            1,
            "near_body_upper",
            "-0.85 <= x < 3.65 and abs(y) < 1.25 and 0.75 <= z < 2.0",
        ),
        RegionRule(
            2,
            "near_wake",
            "3.65 <= x < 6.0 and abs(y) < 2.0 and z < 2.5",
        ),
        RegionRule(
            3,
            "upstream_and_outer",
            "all remaining native volume cells",
        ),
    ),
    semantic_limit=(
        "coarse Cartesian envelopes only; underbody_and_wheels is not an "
        "exclusive floor-boundary-layer or wheel-patch mask"
    ),
)

REGION_DEFINITIONS: Mapping[str, RegionDefinition] = {
    SURFACE_REGION_DEFINITION.definition_id: SURFACE_REGION_DEFINITION,
    VOLUME_REGION_DEFINITION.definition_id: VOLUME_REGION_DEFINITION,
}

REGIONAL_FIELD_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "surface_pressure": {
        "quantity": "pMeanTrim",
        "unit": "m2 s-2",
        "component_labels": ["p"],
        "primary_weighting": "physical",
    },
    "surface_wall_shear": {
        "quantity": "wallShearStressMeanTrim",
        "unit": "m2 s-2",
        "component_labels": ["x", "y", "z"],
        "primary_weighting": "physical",
    },
    "volume_pressure": {
        "quantity": "pMeanTrim",
        "unit": "m2 s-2",
        "component_labels": ["p"],
        "primary_weighting": "equal_entity",
    },
    "volume_velocity": {
        "quantity": "UMeanTrim",
        "unit": "m s-1",
        "component_labels": ["Ux", "Uy", "Uz"],
        "primary_weighting": "equal_entity",
    },
}

_FIELD_DEFINITION_IDS: Mapping[str, str] = {
    "surface_pressure": SURFACE_REGION_DEFINITION.definition_id,
    "surface_wall_shear": SURFACE_REGION_DEFINITION.definition_id,
    "volume_pressure": VOLUME_REGION_DEFINITION.definition_id,
    "volume_velocity": VOLUME_REGION_DEFINITION.definition_id,
}

_WEIGHTING_KEYS = frozenset(
    {
        "absolute_error",
        "squared_error",
        "squared_truth",
        "relative_l2_percent",
        "mae",
        "rmse",
        "component_squared_error",
        "component_squared_truth",
        "component_relative_l2_percent",
        "component_rmse",
        "component_fraction_of_region_squared_error",
    }
)
_PHYSICAL_FRACTION_KEYS = frozenset(
    {
        "fraction_of_case_squared_error",
        "fraction_of_case_squared_truth",
    }
)
_VELOCITY_MAGNITUDE_KEYS = frozenset(
    {
        "absolute_error",
        "squared_error",
        "relative_l2_percent_with_vector_truth_norm",
        "mae",
        "rmse",
    }
)
_VELOCITY_DIRECTION_KEYS = frozenset(
    {
        "definition",
        "minimum_speed_m_per_s",
        "defined_entity_count",
        "defined_entity_fraction",
        "cosine_sum",
        "angular_error_sum_degrees",
        "mean_cosine_similarity",
        "mean_angular_error_degrees",
    }
)
_VELOCITY_DIRECTION_DEFINITION = (
    "both truth and prediction speed meet the threshold"
)

REGIONAL_METRIC_SEMANTICS: Mapping[str, object] = {
    "relative_l2_percent": (
        "100*sqrt(sum(weight*||prediction-truth||^2)/"
        "sum(weight*||truth||^2))"
    ),
    "rmse": "sqrt(sum(weight*||prediction-truth||^2)/sum(weight))",
    "mae": "sum(weight*||prediction-truth||)/sum(weight)",
    "component_metrics": "same formulas applied independently to ordered components",
    "speed_magnitude_error": (
        "abs(||prediction||-||truth||); relative L2 denominator is vector truth "
        "squared norm"
    ),
    "direction_selection": (
        "both truth and prediction speed >= 1.94445 m s-1 (0.05*38.889)"
    ),
    "direction_metrics": (
        "mean cosine similarity and mean angular error in degrees over selected cells"
    ),
    "case_reconstruction_tolerance": {
        "relative": DEFAULT_RECONSTRUCTION_RTOL,
        "absolute": DEFAULT_RECONSTRUCTION_ATOL,
    },
    "aggregate_pooled": (
        "sum additive numerators/denominators over exact split cases before "
        "deriving metrics"
    ),
    "aggregate_macro": "equal arithmetic mean of complete per-case regional metrics",
    "aggregate_distribution": (
        "linear order-statistic interpolation at minimum, median, p90 and maximum"
    ),
    "regional_scoring_weight": REGIONAL_SCORING_WEIGHT,
}


def regional_contract_projection() -> dict[str, object]:
    """Return the executable semantics frozen in the release JSON contract."""

    return {
        "definitions": {
            "surface": SURFACE_REGION_DEFINITION.to_json(),
            "volume": VOLUME_REGION_DEFINITION.to_json(),
        },
        "fields": {
            field_id: dict(values)
            for field_id, values in REGIONAL_FIELD_CONTRACTS.items()
        },
        "metric_semantics": dict(REGIONAL_METRIC_SEMANTICS),
    }


def canonical_json_bytes(value: object) -> bytes:
    """Return the evaluator's platform-independent compact JSON encoding."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RegionalDiagnosticError(
            "regional definition/report is not finite JSON"
        ) from error


def _numeric_array(value: object, *, label: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise RegionalDiagnosticError(f"{label} must be numeric") from error
    if array.ndim != ndim or array.dtype.kind not in {"i", "u", "f"}:
        raise RegionalDiagnosticError(f"{label} must be a numeric {ndim}-D array")
    normalized = np.asarray(array, dtype=np.float64)
    if normalized.size == 0 or not np.all(np.isfinite(normalized)):
        raise RegionalDiagnosticError(f"{label} must be non-empty and finite")
    return normalized


def surface_region_codes(
    face_centre_z_m: object,
    signed_unit_normal_z: object,
) -> np.ndarray:
    """Classify surface polygons from aligned face-centre and unit-normal data."""

    z_m = _numeric_array(face_centre_z_m, label="face_centre_z_m", ndim=1)
    normal_z = _numeric_array(
        signed_unit_normal_z, label="signed_unit_normal_z", ndim=1
    )
    if z_m.shape != normal_z.shape:
        raise RegionalDiagnosticError(
            "face-centre and signed-unit-normal arrays must have identical shapes"
        )
    if np.any(np.abs(normal_z) > 1.0 + _NORMAL_BOUND_TOLERANCE):
        raise RegionalDiagnosticError("signed_unit_normal_z lies outside [-1, 1]")
    low = z_m < SURFACE_Z_THRESHOLD_M
    horizontal = np.abs(normal_z) >= SURFACE_HORIZONTAL_NORMAL_ABS_Z_MIN
    codes = np.full(z_m.size, 3, dtype=np.uint8)
    codes[low] = 1
    codes[low & horizontal] = 0
    codes[(~low) & horizontal] = 2
    return codes


def classify_surface_geometry(
    face_centres_m: object,
    oriented_area_vectors_m2: object,
    calculated_areas_m2: object,
) -> np.ndarray:
    """Classify an existing ``SurfaceGeometryChunk`` without recomputing it."""

    centres = _numeric_array(face_centres_m, label="face_centres_m", ndim=2)
    area_vectors = _numeric_array(
        oriented_area_vectors_m2, label="oriented_area_vectors_m2", ndim=2
    )
    areas = _numeric_array(calculated_areas_m2, label="calculated_areas_m2", ndim=1)
    if (
        centres.shape[1:] != (3,)
        or area_vectors.shape != centres.shape
        or areas.shape != (centres.shape[0],)
    ):
        raise RegionalDiagnosticError(
            "surface geometry must contain aligned [polygon,3] centres/vectors "
            "and [polygon] areas"
        )
    if np.any(areas <= 0.0):
        raise RegionalDiagnosticError("calculated surface areas must be positive")
    normal_z = area_vectors[:, 2] / areas
    return surface_region_codes(centres[:, 2], normal_z)


def volume_region_codes(cell_centres_m: object) -> np.ndarray:
    """Classify native volume cells into the validated four Cartesian views."""

    centres = _numeric_array(cell_centres_m, label="cell_centres_m", ndim=2)
    if centres.shape[1:] != (3,):
        raise RegionalDiagnosticError("cell_centres_m must have shape [cell, 3]")
    x = centres[:, 0]
    y_abs = np.abs(centres[:, 1])
    z = centres[:, 2]
    body_xy = (
        (x >= VOLUME_X_BODY_MIN_M)
        & (x < VOLUME_X_BODY_MAX_M)
        & (y_abs < VOLUME_BODY_ABS_Y_MAX_M)
    )
    low = body_xy & (z < VOLUME_Z_LOW_MAX_M)
    upper = body_xy & (z >= VOLUME_Z_LOW_MAX_M) & (z < VOLUME_Z_UPPER_MAX_M)
    wake = (
        (x >= VOLUME_X_BODY_MAX_M)
        & (x < VOLUME_X_WAKE_MAX_M)
        & (y_abs < VOLUME_WAKE_ABS_Y_MAX_M)
        & (z < VOLUME_WAKE_Z_MAX_M)
    )
    codes = np.full(centres.shape[0], 3, dtype=np.uint8)
    codes[low] = 0
    codes[upper] = 1
    codes[wake] = 2
    return codes


def _region_codes(value: object, expected_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (expected_rows,):
        raise RegionalDiagnosticError("region codes must contain one code per entity")
    if array.dtype.kind not in {"i", "u"}:
        raise RegionalDiagnosticError("region codes must use an integer dtype")
    if np.any(array < 0) or np.any(array >= 4):
        raise RegionalDiagnosticError("region codes must lie in [0, 3]")
    return np.asarray(array, dtype=np.uint8)


class RegionAssignmentHasher:
    """Hash uint8 region codes in complete zero-based native-entity order."""

    def __init__(self, definition: RegionDefinition, expected_entity_count: int) -> None:
        if definition not in REGION_DEFINITIONS.values():
            raise RegionalDiagnosticError("definition is not an immutable release definition")
        if (
            not isinstance(expected_entity_count, int)
            or isinstance(expected_entity_count, bool)
            or expected_entity_count < 1
        ):
            raise RegionalDiagnosticError("expected_entity_count must be positive")
        self.definition = definition
        self.expected_entity_count = expected_entity_count
        self._cursor = 0
        self._digest = hashlib.sha256()
        self._digest.update(b"autocfd5-regional-assignment-uint8-v1\x00")
        self._digest.update(definition.sha256.encode("ascii"))
        self._digest.update(b"\x00")

    @property
    def cursor(self) -> int:
        return self._cursor

    def add_chunk(self, raw_id_start: int, codes: object) -> None:
        if (
            not isinstance(raw_id_start, int)
            or isinstance(raw_id_start, bool)
            or raw_id_start != self._cursor
        ):
            raise RegionalDiagnosticError(
                "region assignments must arrive in complete zero-based native order"
            )
        array = np.asarray(codes)
        if array.ndim != 1:
            raise RegionalDiagnosticError("region assignment chunk must be one-dimensional")
        normalized = _region_codes(array, int(array.size))
        stop = raw_id_start + normalized.size
        if stop > self.expected_entity_count:
            raise RegionalDiagnosticError("region assignments exceed expected_entity_count")
        self._digest.update(normalized.tobytes(order="C"))
        self._cursor = stop

    def finalize(self) -> dict[str, object]:
        if self._cursor != self.expected_entity_count:
            raise RegionalDiagnosticError(
                "region assignments do not cover every native entity"
            )
        return {
            "encoding": _ASSIGNMENT_ENCODING,
            "definition_sha256": self.definition.sha256,
            "entity_count": self.expected_entity_count,
            "complete_gap_free_duplicate_free": True,
            "sha256": self._digest.hexdigest(),
        }


@dataclass(frozen=True)
class _ChunkSums:
    raw_id_start: int
    raw_id_stop: int
    count: np.ndarray
    weight: np.ndarray
    uniform_absolute_error: np.ndarray
    uniform_squared_error: np.ndarray
    uniform_squared_truth: np.ndarray
    physical_absolute_error: np.ndarray
    physical_squared_error: np.ndarray
    physical_squared_truth: np.ndarray
    uniform_component_squared_error: np.ndarray
    uniform_component_squared_truth: np.ndarray
    physical_component_squared_error: np.ndarray
    physical_component_squared_truth: np.ndarray
    speed_absolute_error: np.ndarray | None
    speed_squared_error: np.ndarray | None
    direction_count: np.ndarray | None
    cosine_sum: np.ndarray | None
    angle_sum_degrees: np.ndarray | None


def _bins(codes: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(codes, weights=weights, minlength=4)


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RegionalDiagnosticError(f"{label} must be finite and non-negative")
    return result


def _expected_sums(value: object, label: str) -> dict[str, float | int]:
    if isinstance(value, Mapping):
        source = value
    else:
        source = {
            name: getattr(value, name, None)
            for name in (
                "absolute_error",
                "squared_error",
                "squared_truth",
                "entity_count",
                "total_weight",
            )
        }
    if set(source) != {
        "absolute_error",
        "squared_error",
        "squared_truth",
        "entity_count",
        "total_weight",
    }:
        raise RegionalDiagnosticError(f"{label} has incomplete additive sums")
    count = source["entity_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise RegionalDiagnosticError(f"{label}.entity_count must be positive")
    result: dict[str, float | int] = {"entity_count": count}
    for name in ("absolute_error", "squared_error", "squared_truth", "total_weight"):
        numeric = _finite_nonnegative(float(source[name]), f"{label}.{name}")
        if name == "total_weight" and numeric <= 0.0:
            raise RegionalDiagnosticError(f"{label}.total_weight must be positive")
        result[name] = numeric
    return result


def _merge(rows: Sequence[np.ndarray], width: int = 4) -> np.ndarray:
    if not rows:
        raise RegionalDiagnosticError("cannot finalize empty regional statistics")
    return np.asarray(
        [math.fsum(float(row[index]) for row in rows) for index in range(width)],
        dtype=np.float64,
    )


def _merge_components(
    rows: Sequence[np.ndarray], components: int, width: int = 4
) -> np.ndarray:
    if not rows:
        raise RegionalDiagnosticError("cannot finalize empty component statistics")
    return np.asarray(
        [
            [
                math.fsum(float(row[index, component]) for row in rows)
                for component in range(components)
            ]
            for index in range(width)
        ],
        dtype=np.float64,
    )


def _close(actual: float, expected: float, *, rtol: float, atol: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise RegionalDiagnosticError(
            f"{label} reconstruction differs: actual={actual!r}, expected={expected!r}"
        )


class RegionalFieldAccumulator:
    """Collect four-region scalar/vector sufficient statistics in bounded memory."""

    def __init__(
        self,
        definition: RegionDefinition,
        expected_entity_count: int,
        component_labels: Sequence[str],
        *,
        velocity_diagnostics: bool = False,
        direction_min_speed_m_per_s: float = VELOCITY_DIRECTION_MIN_SPEED_M_PER_S,
    ) -> None:
        if definition not in REGION_DEFINITIONS.values():
            raise RegionalDiagnosticError("definition is not an immutable release definition")
        if (
            not isinstance(expected_entity_count, int)
            or isinstance(expected_entity_count, bool)
            or expected_entity_count < 1
        ):
            raise RegionalDiagnosticError("expected_entity_count must be positive")
        labels = tuple(component_labels)
        if not labels or len(labels) != len(set(labels)) or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise RegionalDiagnosticError("component_labels must be unique non-empty strings")
        if velocity_diagnostics and len(labels) != 3:
            raise RegionalDiagnosticError("velocity diagnostics require exactly 3 components")
        if (
            not math.isfinite(direction_min_speed_m_per_s)
            or direction_min_speed_m_per_s <= 0.0
        ):
            raise RegionalDiagnosticError("direction threshold must be finite and positive")
        self.definition = definition
        self.expected_entity_count = expected_entity_count
        self.component_labels = labels
        self.velocity_diagnostics = velocity_diagnostics
        self.direction_min_speed_m_per_s = float(direction_min_speed_m_per_s)
        self._cursor = 0
        self._chunks: list[_ChunkSums] = []

    @property
    def cursor(self) -> int:
        return self._cursor

    def add_chunk(
        self,
        raw_id_start: int,
        codes: object,
        truth: object,
        prediction: object,
        physical_weights: object | None = None,
    ) -> None:
        if (
            not isinstance(raw_id_start, int)
            or isinstance(raw_id_start, bool)
            or raw_id_start != self._cursor
        ):
            raise RegionalDiagnosticError(
                "regional field chunks must arrive in complete zero-based native order"
            )
        truth_array = _numeric_array(truth, label="truth", ndim=np.asarray(truth).ndim)
        prediction_array = _numeric_array(
            prediction, label="prediction", ndim=np.asarray(prediction).ndim
        )
        if truth_array.ndim not in {1, 2} or truth_array.shape != prediction_array.shape:
            raise RegionalDiagnosticError(
                "truth and prediction must have identical [entity] or [entity,component] shapes"
            )
        if truth_array.ndim == 1:
            truth_array = truth_array[:, None]
            prediction_array = prediction_array[:, None]
        count, components = truth_array.shape
        if components != len(self.component_labels):
            raise RegionalDiagnosticError("field component count differs from component_labels")
        normalized_codes = _region_codes(codes, count)
        stop = raw_id_start + count
        if stop > self.expected_entity_count:
            raise RegionalDiagnosticError("regional field chunk exceeds expected_entity_count")
        if physical_weights is None:
            weights = np.ones(count, dtype=np.float64)
        else:
            weights = _numeric_array(
                physical_weights, label="physical_weights", ndim=1
            )
            if weights.shape != (count,) or np.any(weights <= 0.0):
                raise RegionalDiagnosticError(
                    "physical_weights must contain one positive value per entity"
                )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = prediction_array - truth_array
            component_squared_error = error * error
            component_squared_truth = truth_array * truth_array
            squared_error = np.einsum(
                "ij,ij->i", error, error, optimize=False
            )
            squared_truth = np.einsum(
                "ij,ij->i", truth_array, truth_array, optimize=False
            )
            absolute_error = np.sqrt(squared_error)

        arrays = (
            component_squared_error,
            component_squared_truth,
            squared_error,
            squared_truth,
            absolute_error,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise RegionalDiagnosticError("regional field arithmetic became non-finite")

        uniform_component_error = np.empty((4, components), dtype=np.float64)
        uniform_component_truth = np.empty_like(uniform_component_error)
        physical_component_error = np.empty_like(uniform_component_error)
        physical_component_truth = np.empty_like(uniform_component_error)
        for component in range(components):
            uniform_component_error[:, component] = _bins(
                normalized_codes, component_squared_error[:, component]
            )
            uniform_component_truth[:, component] = _bins(
                normalized_codes, component_squared_truth[:, component]
            )
            physical_component_error[:, component] = _bins(
                normalized_codes, weights * component_squared_error[:, component]
            )
            physical_component_truth[:, component] = _bins(
                normalized_codes, weights * component_squared_truth[:, component]
            )

        speed_absolute: np.ndarray | None = None
        speed_squared: np.ndarray | None = None
        direction_count: np.ndarray | None = None
        cosine_sum: np.ndarray | None = None
        angle_sum: np.ndarray | None = None
        if self.velocity_diagnostics:
            truth_speed = np.sqrt(squared_truth)
            prediction_speed = np.linalg.norm(prediction_array, axis=1)
            speed_error = np.abs(prediction_speed - truth_speed)
            stable = (
                (truth_speed >= self.direction_min_speed_m_per_s)
                & (prediction_speed >= self.direction_min_speed_m_per_s)
            )
            speed_absolute = _bins(normalized_codes, speed_error)
            speed_squared = _bins(normalized_codes, speed_error * speed_error)
            direction_count = np.zeros(4, dtype=np.int64)
            cosine_sum = np.zeros(4, dtype=np.float64)
            angle_sum = np.zeros(4, dtype=np.float64)
            if np.any(stable):
                stable_codes = normalized_codes[stable]
                cosine = np.einsum(
                    "ij,ij->i",
                    prediction_array[stable],
                    truth_array[stable],
                    optimize=False,
                ) / (prediction_speed[stable] * truth_speed[stable])
                np.clip(cosine, -1.0, 1.0, out=cosine)
                angle = np.degrees(np.arccos(cosine))
                direction_count = _bins(stable_codes).astype(np.int64, copy=False)
                cosine_sum = _bins(stable_codes, cosine)
                angle_sum = _bins(stable_codes, angle)

        self._chunks.append(
            _ChunkSums(
                raw_id_start=raw_id_start,
                raw_id_stop=stop,
                count=_bins(normalized_codes).astype(np.int64, copy=False),
                weight=_bins(normalized_codes, weights),
                uniform_absolute_error=_bins(normalized_codes, absolute_error),
                uniform_squared_error=_bins(normalized_codes, squared_error),
                uniform_squared_truth=_bins(normalized_codes, squared_truth),
                physical_absolute_error=_bins(
                    normalized_codes, weights * absolute_error
                ),
                physical_squared_error=_bins(
                    normalized_codes, weights * squared_error
                ),
                physical_squared_truth=_bins(
                    normalized_codes, weights * squared_truth
                ),
                uniform_component_squared_error=uniform_component_error,
                uniform_component_squared_truth=uniform_component_truth,
                physical_component_squared_error=physical_component_error,
                physical_component_squared_truth=physical_component_truth,
                speed_absolute_error=speed_absolute,
                speed_squared_error=speed_squared,
                direction_count=direction_count,
                cosine_sum=cosine_sum,
                angle_sum_degrees=angle_sum,
            )
        )
        self._cursor = stop

    def finalize(
        self,
        *,
        expected_uniform: object | None = None,
        expected_physical: object | None = None,
        require_all_regions: bool = True,
        reconstruction_rtol: float = DEFAULT_RECONSTRUCTION_RTOL,
        reconstruction_atol: float = DEFAULT_RECONSTRUCTION_ATOL,
    ) -> dict[str, object]:
        """Finalize regional fields and prove reconstruction of global sums."""

        if self._cursor != self.expected_entity_count:
            raise RegionalDiagnosticError(
                "regional field chunks do not cover every native entity"
            )
        if (expected_uniform is None) != (expected_physical is None):
            raise RegionalDiagnosticError(
                "expected uniform and physical sums must be supplied together"
            )
        if (
            not math.isfinite(reconstruction_rtol)
            or reconstruction_rtol < 0.0
            or not math.isfinite(reconstruction_atol)
            or reconstruction_atol < 0.0
        ):
            raise RegionalDiagnosticError("reconstruction tolerances must be non-negative")

        counts_float = _merge([chunk.count for chunk in self._chunks])
        counts = counts_float.astype(np.int64)
        if not np.array_equal(counts_float, counts.astype(np.float64)):
            raise RegionalDiagnosticError("regional entity counts are not integral")
        if int(np.sum(counts, dtype=np.int64)) != self.expected_entity_count:
            raise RegionalDiagnosticError("regional entity counts do not reconstruct coverage")
        if require_all_regions and np.any(counts == 0):
            raise RegionalDiagnosticError("at least one configured region is empty")

        weight = _merge([chunk.weight for chunk in self._chunks])
        uniform_absolute = _merge(
            [chunk.uniform_absolute_error for chunk in self._chunks]
        )
        uniform_error = _merge(
            [chunk.uniform_squared_error for chunk in self._chunks]
        )
        uniform_truth = _merge(
            [chunk.uniform_squared_truth for chunk in self._chunks]
        )
        physical_absolute = _merge(
            [chunk.physical_absolute_error for chunk in self._chunks]
        )
        physical_error = _merge(
            [chunk.physical_squared_error for chunk in self._chunks]
        )
        physical_truth = _merge(
            [chunk.physical_squared_truth for chunk in self._chunks]
        )
        uniform_component_error = _merge_components(
            [chunk.uniform_component_squared_error for chunk in self._chunks],
            len(self.component_labels),
        )
        uniform_component_truth = _merge_components(
            [chunk.uniform_component_squared_truth for chunk in self._chunks],
            len(self.component_labels),
        )
        physical_component_error = _merge_components(
            [chunk.physical_component_squared_error for chunk in self._chunks],
            len(self.component_labels),
        )
        physical_component_truth = _merge_components(
            [chunk.physical_component_squared_truth for chunk in self._chunks],
            len(self.component_labels),
        )

        reconstructed_uniform: dict[str, float | int] = {
            "absolute_error": math.fsum(uniform_absolute.tolist()),
            "squared_error": math.fsum(uniform_error.tolist()),
            "squared_truth": math.fsum(uniform_truth.tolist()),
            "entity_count": int(np.sum(counts, dtype=np.int64)),
            "total_weight": float(np.sum(counts, dtype=np.int64)),
        }
        reconstructed_physical: dict[str, float | int] = {
            "absolute_error": math.fsum(physical_absolute.tolist()),
            "squared_error": math.fsum(physical_error.tolist()),
            "squared_truth": math.fsum(physical_truth.tolist()),
            "entity_count": int(np.sum(counts, dtype=np.int64)),
            "total_weight": math.fsum(weight.tolist()),
        }

        expected_mappings: dict[str, dict[str, float | int]] = {}
        if expected_uniform is not None and expected_physical is not None:
            expected_mappings = {
                "uniform": _expected_sums(expected_uniform, "expected_uniform"),
                "physical": _expected_sums(expected_physical, "expected_physical"),
            }
            for weighting, actual in (
                ("uniform", reconstructed_uniform),
                ("physical", reconstructed_physical),
            ):
                expected = expected_mappings[weighting]
                for name, actual_value in actual.items():
                    expected_value = expected[name]
                    if name == "entity_count":
                        if actual_value != expected_value:
                            raise RegionalDiagnosticError(
                                f"{weighting}.entity_count reconstruction differs"
                            )
                    else:
                        _close(
                            float(actual_value),
                            float(expected_value),
                            rtol=reconstruction_rtol,
                            atol=reconstruction_atol,
                            label=f"{weighting}.{name}",
                        )

        total_area = math.fsum(weight.tolist())
        total_physical_error = math.fsum(physical_error.tolist())
        total_physical_truth = math.fsum(physical_truth.tolist())
        regions: list[dict[str, object]] = []
        for index, region_id in enumerate(self.definition.region_ids):
            count = int(counts[index])
            region_weight = float(weight[index])
            uniform = self._weighting_report(
                count=float(count),
                absolute=float(uniform_absolute[index]),
                squared_error=float(uniform_error[index]),
                squared_truth=float(uniform_truth[index]),
                component_error=uniform_component_error[index],
                component_truth=uniform_component_truth[index],
            )
            physical = self._weighting_report(
                count=region_weight,
                absolute=float(physical_absolute[index]),
                squared_error=float(physical_error[index]),
                squared_truth=float(physical_truth[index]),
                component_error=physical_component_error[index],
                component_truth=physical_component_truth[index],
            )
            physical["fraction_of_case_squared_error"] = (
                float(physical_error[index]) / total_physical_error
                if total_physical_error > 0.0
                else None
            )
            physical["fraction_of_case_squared_truth"] = (
                float(physical_truth[index]) / total_physical_truth
                if total_physical_truth > 0.0
                else None
            )
            row: dict[str, object] = {
                "region_id": region_id,
                "entity_count": count,
                "entity_fraction": count / self.expected_entity_count,
                "physical_weight": region_weight,
                "physical_weight_fraction": region_weight / total_area,
                "equal_entity": uniform,
                "physical": physical,
            }
            if self.velocity_diagnostics:
                row["velocity"] = self._velocity_report(index, count, uniform_truth)
            regions.append(row)

        reconstruction: dict[str, object] = {
            "tolerance": {
                "relative": reconstruction_rtol,
                "absolute": reconstruction_atol,
            },
            "all_four_regions_nonempty": bool(np.all(counts > 0)),
            "mutually_exclusive_exhaustive": True,
            "reconstructed_uniform": reconstructed_uniform,
            "reconstructed_physical": reconstructed_physical,
            "expected_global_sums_supplied": bool(expected_mappings),
            "matches_expected_global_sums": bool(expected_mappings),
        }
        if expected_mappings:
            reconstruction["expected_uniform"] = expected_mappings["uniform"]
            reconstruction["expected_physical"] = expected_mappings["physical"]
        report = {
            "definition_id": self.definition.definition_id,
            "component_labels": list(self.component_labels),
            "entity_count": self.expected_entity_count,
            "region_count": 4,
            "regions": regions,
            "reconstruction": reconstruction,
        }
        validate_regional_field_report(report, self.definition)
        return report

    def _weighting_report(
        self,
        *,
        count: float,
        absolute: float,
        squared_error: float,
        squared_truth: float,
        component_error: np.ndarray,
        component_truth: np.ndarray,
    ) -> dict[str, object]:
        if count <= 0.0:
            return {
                "absolute_error": absolute,
                "squared_error": squared_error,
                "squared_truth": squared_truth,
                "relative_l2_percent": None,
                "mae": None,
                "rmse": None,
                "component_squared_error": {
                    label: float(component_error[offset])
                    for offset, label in enumerate(self.component_labels)
                },
                "component_squared_truth": {
                    label: float(component_truth[offset])
                    for offset, label in enumerate(self.component_labels)
                },
                "component_relative_l2_percent": {
                    label: None for label in self.component_labels
                },
                "component_rmse": {label: None for label in self.component_labels},
                "component_fraction_of_region_squared_error": {
                    label: None for label in self.component_labels
                },
            }
        component_relative = {
            label: (
                100.0
                * math.sqrt(
                    float(component_error[offset]) / float(component_truth[offset])
                )
                if component_truth[offset] > 0.0
                else None
            )
            for offset, label in enumerate(self.component_labels)
        }
        return {
            "absolute_error": absolute,
            "squared_error": squared_error,
            "squared_truth": squared_truth,
            "relative_l2_percent": (
                100.0 * math.sqrt(squared_error / squared_truth)
                if squared_truth > 0.0
                else None
            ),
            "mae": absolute / count,
            "rmse": math.sqrt(squared_error / count),
            "component_squared_error": {
                label: float(component_error[offset])
                for offset, label in enumerate(self.component_labels)
            },
            "component_squared_truth": {
                label: float(component_truth[offset])
                for offset, label in enumerate(self.component_labels)
            },
            "component_relative_l2_percent": component_relative,
            "component_rmse": {
                label: math.sqrt(float(component_error[offset]) / count)
                for offset, label in enumerate(self.component_labels)
            },
            "component_fraction_of_region_squared_error": {
                label: (
                    float(component_error[offset]) / squared_error
                    if squared_error > 0.0
                    else None
                )
                for offset, label in enumerate(self.component_labels)
            },
        }

    def _velocity_report(
        self, index: int, count: int, uniform_truth: np.ndarray
    ) -> dict[str, object]:
        speed_absolute = _merge(
            [
                chunk.speed_absolute_error
                for chunk in self._chunks
                if chunk.speed_absolute_error is not None
            ]
        )
        speed_squared = _merge(
            [
                chunk.speed_squared_error
                for chunk in self._chunks
                if chunk.speed_squared_error is not None
            ]
        )
        direction_count_float = _merge(
            [
                chunk.direction_count
                for chunk in self._chunks
                if chunk.direction_count is not None
            ]
        )
        direction_count = direction_count_float.astype(np.int64)
        cosine_sum = _merge(
            [chunk.cosine_sum for chunk in self._chunks if chunk.cosine_sum is not None]
        )
        angle_sum = _merge(
            [
                chunk.angle_sum_degrees
                for chunk in self._chunks
                if chunk.angle_sum_degrees is not None
            ]
        )
        stable_count = int(direction_count[index])
        truth_energy = float(uniform_truth[index])
        return {
            "speed_magnitude": {
                "absolute_error": float(speed_absolute[index]),
                "squared_error": float(speed_squared[index]),
                "relative_l2_percent_with_vector_truth_norm": (
                    100.0 * math.sqrt(float(speed_squared[index]) / truth_energy)
                    if truth_energy > 0.0
                    else None
                ),
                "mae": float(speed_absolute[index]) / count if count else None,
                "rmse": math.sqrt(float(speed_squared[index]) / count)
                if count
                else None,
            },
            "direction": {
                "definition": _VELOCITY_DIRECTION_DEFINITION,
                "minimum_speed_m_per_s": self.direction_min_speed_m_per_s,
                "defined_entity_count": stable_count,
                "defined_entity_fraction": stable_count / count if count else None,
                "cosine_sum": float(cosine_sum[index]),
                "angular_error_sum_degrees": float(angle_sum[index]),
                "mean_cosine_similarity": (
                    float(cosine_sum[index]) / stable_count if stable_count else None
                ),
                "mean_angular_error_degrees": (
                    float(angle_sum[index]) / stable_count if stable_count else None
                ),
            },
        }


def validate_reconstruction(
    field_report: Mapping[str, object],
    *,
    expected_uniform: object,
    expected_physical: object,
    rtol: float = DEFAULT_RECONSTRUCTION_RTOL,
    atol: float = DEFAULT_RECONSTRUCTION_ATOL,
) -> None:
    """Independently check retained report sums against official accumulators."""

    expected = {
        "uniform": _expected_sums(expected_uniform, "expected_uniform"),
        "physical": _expected_sums(expected_physical, "expected_physical"),
    }
    reconstruction = field_report.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise RegionalDiagnosticError("field report has no reconstruction object")
    for weighting in ("uniform", "physical"):
        actual = reconstruction.get(f"reconstructed_{weighting}")
        if not isinstance(actual, Mapping) or set(actual) != set(expected[weighting]):
            raise RegionalDiagnosticError(
                f"field report reconstructed_{weighting} keys differ"
            )
        for key, expected_value in expected[weighting].items():
            actual_value = actual[key]
            if key == "entity_count":
                if actual_value != expected_value:
                    raise RegionalDiagnosticError(
                        f"{weighting}.entity_count reconstruction differs"
                    )
            else:
                _close(
                    float(actual_value),
                    float(expected_value),
                    rtol=rtol,
                    atol=atol,
                    label=f"{weighting}.{key}",
                )


def _finite_json_number(value: object, label: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise RegionalDiagnosticError(f"{label} must be a finite JSON number")


def _check_derived_number(
    actual: object,
    expected: float | None,
    *,
    label: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise RegionalDiagnosticError(f"{label} must be null")
        return
    _finite_json_number(actual, label)
    _close(
        float(actual),
        expected,
        rtol=DEFAULT_RECONSTRUCTION_RTOL,
        atol=DEFAULT_RECONSTRUCTION_ATOL,
        label=label,
    )


def validate_regional_field_report(
    report: Mapping[str, object],
    definition: RegionDefinition,
    *,
    field_id: str | None = None,
) -> None:
    """Strictly validate one compact scalar/vector regional field report."""

    expected_keys = {
        "definition_id",
        "component_labels",
        "entity_count",
        "region_count",
        "regions",
        "reconstruction",
    }
    if not isinstance(report, Mapping) or set(report) != expected_keys:
        raise RegionalDiagnosticError("regional field report keys differ")
    if report.get("definition_id") != definition.definition_id:
        raise RegionalDiagnosticError("regional field definition_id differs")
    field_contract: Mapping[str, object] | None = None
    if field_id is not None:
        field_contract = REGIONAL_FIELD_CONTRACTS.get(field_id)
        if field_contract is None:
            raise RegionalDiagnosticError("regional field_id is not released")
        if _FIELD_DEFINITION_IDS[field_id] != definition.definition_id:
            raise RegionalDiagnosticError(
                "regional field is bound to a different support definition"
            )
    components = report.get("component_labels")
    if (
        not isinstance(components, list)
        or not components
        or len(components) != len(set(components))
        or any(not isinstance(value, str) or not value for value in components)
    ):
        raise RegionalDiagnosticError("regional field component labels are invalid")
    if field_contract is not None and components != field_contract["component_labels"]:
        raise RegionalDiagnosticError(
            "regional field component labels/order differ from released contract"
        )
    entity_count = report.get("entity_count")
    if (
        not isinstance(entity_count, int)
        or isinstance(entity_count, bool)
        or entity_count < 1
        or report.get("region_count") != 4
    ):
        raise RegionalDiagnosticError("regional field counts are invalid")
    regions = report.get("regions")
    if not isinstance(regions, list) or len(regions) != 4:
        raise RegionalDiagnosticError("regional field must contain exactly four regions")
    if [row.get("region_id") for row in regions if isinstance(row, Mapping)] != list(
        definition.region_ids
    ):
        raise RegionalDiagnosticError("regional field IDs/order differ from definition")
    if sum(int(row.get("entity_count", -1)) for row in regions) != entity_count:
        raise RegionalDiagnosticError("regional field entity counts do not reconstruct")
    for offset, row in enumerate(regions):
        if not isinstance(row, Mapping):
            raise RegionalDiagnosticError("regional field row is not an object")
        required = {
            "region_id",
            "entity_count",
            "entity_fraction",
            "physical_weight",
            "physical_weight_fraction",
            "equal_entity",
            "physical",
        }
        expected_row_keys = required | (
            {"velocity"} if field_id == "volume_velocity" else set()
        )
        if field_id is None:
            expected_row_keys |= {"velocity"} if "velocity" in row else set()
        if set(row) != expected_row_keys:
            raise RegionalDiagnosticError(
                f"regional field row {offset} keys differ"
            )
        count = row.get("entity_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RegionalDiagnosticError("regional field entity_count is invalid")
        for name in (
            "entity_fraction",
            "physical_weight",
            "physical_weight_fraction",
        ):
            _finite_json_number(row.get(name), f"region.{name}")
        count = int(count)
        physical_weight = float(row["physical_weight"])
        if count < 1 or physical_weight <= 0.0:
            raise RegionalDiagnosticError("every regional field bin must be non-empty")
        _check_derived_number(
            row["entity_fraction"],
            count / entity_count,
            label="region.entity_fraction",
        )
        for weighting in ("equal_entity", "physical"):
            values = row.get(weighting)
            if not isinstance(values, Mapping):
                raise RegionalDiagnosticError(f"region.{weighting} is not an object")
            expected_weighting_keys = _WEIGHTING_KEYS | (
                _PHYSICAL_FRACTION_KEYS if weighting == "physical" else frozenset()
            )
            if set(values) != expected_weighting_keys:
                raise RegionalDiagnosticError(
                    f"region.{weighting} keys differ"
                )
            for name in ("absolute_error", "squared_error", "squared_truth"):
                _finite_json_number(values.get(name), f"region.{weighting}.{name}")
            for name in ("relative_l2_percent", "mae", "rmse"):
                _finite_json_number(
                    values.get(name), f"region.{weighting}.{name}", nullable=True
                )
            for name in (
                "component_squared_error",
                "component_squared_truth",
                "component_relative_l2_percent",
                "component_rmse",
                "component_fraction_of_region_squared_error",
            ):
                mapping = values.get(name)
                if not isinstance(mapping, Mapping) or set(mapping) != set(components):
                    raise RegionalDiagnosticError(
                        f"region.{weighting}.{name} component set differs"
                    )
                for component, value in mapping.items():
                    _finite_json_number(
                        value,
                        f"region.{weighting}.{name}.{component}",
                        nullable=True,
                    )
            denominator = float(count) if weighting == "equal_entity" else physical_weight
            squared_error = float(values["squared_error"])
            squared_truth = float(values["squared_truth"])
            absolute_error = float(values["absolute_error"])
            if min(squared_error, squared_truth, absolute_error) < 0.0:
                raise RegionalDiagnosticError(
                    f"region.{weighting} additive sums must be non-negative"
                )
            component_error = values["component_squared_error"]
            component_truth = values["component_squared_truth"]
            if any(
                float(component_error[label]) < 0.0
                or float(component_truth[label]) < 0.0
                for label in components
            ):
                raise RegionalDiagnosticError(
                    f"region.{weighting} component sums must be non-negative"
                )
            _close(
                math.fsum(float(component_error[label]) for label in components),
                squared_error,
                rtol=DEFAULT_RECONSTRUCTION_RTOL,
                atol=DEFAULT_RECONSTRUCTION_ATOL,
                label=f"region.{weighting}.component_squared_error",
            )
            _close(
                math.fsum(float(component_truth[label]) for label in components),
                squared_truth,
                rtol=DEFAULT_RECONSTRUCTION_RTOL,
                atol=DEFAULT_RECONSTRUCTION_ATOL,
                label=f"region.{weighting}.component_squared_truth",
            )
            _check_derived_number(
                values["relative_l2_percent"],
                100.0 * math.sqrt(squared_error / squared_truth)
                if squared_truth > 0.0
                else None,
                label=f"region.{weighting}.relative_l2_percent",
            )
            _check_derived_number(
                values["mae"],
                absolute_error / denominator,
                label=f"region.{weighting}.mae",
            )
            _check_derived_number(
                values["rmse"],
                math.sqrt(squared_error / denominator),
                label=f"region.{weighting}.rmse",
            )
            for component in components:
                component_error_value = float(component_error[component])
                component_truth_value = float(component_truth[component])
                _check_derived_number(
                    values["component_relative_l2_percent"][component],
                    100.0
                    * math.sqrt(component_error_value / component_truth_value)
                    if component_truth_value > 0.0
                    else None,
                    label=(
                        f"region.{weighting}.component_relative_l2_percent."
                        f"{component}"
                    ),
                )
                _check_derived_number(
                    values["component_rmse"][component],
                    math.sqrt(component_error_value / denominator),
                    label=f"region.{weighting}.component_rmse.{component}",
                )
                _check_derived_number(
                    values["component_fraction_of_region_squared_error"][component],
                    component_error_value / squared_error
                    if squared_error > 0.0
                    else None,
                    label=(
                        "region."
                        f"{weighting}.component_fraction_of_region_squared_error."
                        f"{component}"
                    ),
                )
        physical = row["physical"]
        for name in (
            "fraction_of_case_squared_error",
            "fraction_of_case_squared_truth",
        ):
            _finite_json_number(physical.get(name), f"region.physical.{name}", nullable=True)
        if "velocity" in row:
            velocity = row["velocity"]
            if not isinstance(velocity, Mapping) or set(velocity) != {
                "speed_magnitude",
                "direction",
            }:
                raise RegionalDiagnosticError("velocity regional report keys differ")
            magnitude = velocity["speed_magnitude"]
            direction = velocity["direction"]
            if not isinstance(magnitude, Mapping) or not isinstance(direction, Mapping):
                raise RegionalDiagnosticError("velocity diagnostic sections must be objects")
            if set(magnitude) != _VELOCITY_MAGNITUDE_KEYS:
                raise RegionalDiagnosticError(
                    "velocity magnitude diagnostic keys differ"
                )
            if set(direction) != _VELOCITY_DIRECTION_KEYS:
                raise RegionalDiagnosticError(
                    "velocity direction diagnostic keys differ"
                )
            if direction.get("definition") != _VELOCITY_DIRECTION_DEFINITION:
                raise RegionalDiagnosticError(
                    "velocity direction definition differs"
                )
            if (
                direction.get("minimum_speed_m_per_s")
                != VELOCITY_DIRECTION_MIN_SPEED_M_PER_S
            ):
                raise RegionalDiagnosticError(
                    "velocity direction threshold differs"
                )
            for name in (
                "absolute_error",
                "squared_error",
                "relative_l2_percent_with_vector_truth_norm",
                "mae",
                "rmse",
            ):
                _finite_json_number(magnitude.get(name), f"velocity.{name}", nullable=True)
            direction_count = direction.get("defined_entity_count")
            if (
                not isinstance(direction_count, int)
                or isinstance(direction_count, bool)
                or direction_count < 0
                or direction_count > count
            ):
                raise RegionalDiagnosticError("velocity direction count is invalid")
            for name in (
                "minimum_speed_m_per_s",
                "defined_entity_fraction",
                "cosine_sum",
                "angular_error_sum_degrees",
                "mean_cosine_similarity",
                "mean_angular_error_degrees",
            ):
                _finite_json_number(direction.get(name), f"direction.{name}", nullable=True)
            speed_squared = float(magnitude["squared_error"])
            speed_absolute = float(magnitude["absolute_error"])
            if speed_squared < 0.0 or speed_absolute < 0.0:
                raise RegionalDiagnosticError(
                    "velocity magnitude additive sums must be non-negative"
                )
            cosine_sum = float(direction["cosine_sum"])
            angle_sum = float(direction["angular_error_sum_degrees"])
            if (
                abs(cosine_sum) > direction_count + DEFAULT_RECONSTRUCTION_ATOL
                or angle_sum < 0.0
                or angle_sum
                > 180.0 * direction_count + DEFAULT_RECONSTRUCTION_ATOL
            ):
                raise RegionalDiagnosticError(
                    "velocity direction additive sums lie outside physical bounds"
                )
            vector_truth = float(row["equal_entity"]["squared_truth"])
            _check_derived_number(
                magnitude["relative_l2_percent_with_vector_truth_norm"],
                100.0 * math.sqrt(speed_squared / vector_truth)
                if vector_truth > 0.0
                else None,
                label="velocity.relative_l2_percent_with_vector_truth_norm",
            )
            _check_derived_number(
                magnitude["mae"],
                speed_absolute / count,
                label="velocity.mae",
            )
            _check_derived_number(
                magnitude["rmse"],
                math.sqrt(speed_squared / count),
                label="velocity.rmse",
            )
            _check_derived_number(
                direction["defined_entity_fraction"],
                direction_count / count,
                label="direction.defined_entity_fraction",
            )
            _check_derived_number(
                direction["mean_cosine_similarity"],
                cosine_sum / direction_count
                if direction_count
                else None,
                label="direction.mean_cosine_similarity",
            )
            _check_derived_number(
                direction["mean_angular_error_degrees"],
                angle_sum / direction_count
                if direction_count
                else None,
                label="direction.mean_angular_error_degrees",
            )
    reconstruction = report.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise RegionalDiagnosticError("regional field reconstruction is absent")
    required_reconstruction_keys = {
        "tolerance",
        "all_four_regions_nonempty",
        "mutually_exclusive_exhaustive",
        "reconstructed_uniform",
        "reconstructed_physical",
        "expected_global_sums_supplied",
        "matches_expected_global_sums",
        "expected_uniform",
        "expected_physical",
    }
    if set(reconstruction) != required_reconstruction_keys or (
        reconstruction.get("mutually_exclusive_exhaustive") is not True
        or reconstruction.get("all_four_regions_nonempty") is not True
        or reconstruction.get("expected_global_sums_supplied") is not True
        or reconstruction.get("matches_expected_global_sums") is not True
        or reconstruction.get("tolerance")
        != {
            "relative": DEFAULT_RECONSTRUCTION_RTOL,
            "absolute": DEFAULT_RECONSTRUCTION_ATOL,
        }
    ):
        raise RegionalDiagnosticError("regional field reconstruction is incomplete")
    total_physical_weight = math.fsum(
        float(row["physical_weight"]) for row in regions
    )
    total_physical_squared_error = math.fsum(
        float(row["physical"]["squared_error"]) for row in regions
    )
    total_physical_squared_truth = math.fsum(
        float(row["physical"]["squared_truth"]) for row in regions
    )
    for row in regions:
        _check_derived_number(
            row["physical_weight_fraction"],
            float(row["physical_weight"]) / total_physical_weight,
            label="region.physical_weight_fraction",
        )
        physical = row["physical"]
        _check_derived_number(
            physical["fraction_of_case_squared_error"],
            float(physical["squared_error"]) / total_physical_squared_error
            if total_physical_squared_error > 0.0
            else None,
            label="region.physical.fraction_of_case_squared_error",
        )
        _check_derived_number(
            physical["fraction_of_case_squared_truth"],
            float(physical["squared_truth"]) / total_physical_squared_truth
            if total_physical_squared_truth > 0.0
            else None,
            label="region.physical.fraction_of_case_squared_truth",
        )
    derived_uniform = {
        "absolute_error": math.fsum(
            float(row["equal_entity"]["absolute_error"]) for row in regions
        ),
        "squared_error": math.fsum(
            float(row["equal_entity"]["squared_error"]) for row in regions
        ),
        "squared_truth": math.fsum(
            float(row["equal_entity"]["squared_truth"]) for row in regions
        ),
        "entity_count": entity_count,
        "total_weight": float(entity_count),
    }
    derived_physical = {
        "absolute_error": math.fsum(
            float(row["physical"]["absolute_error"]) for row in regions
        ),
        "squared_error": math.fsum(
            float(row["physical"]["squared_error"]) for row in regions
        ),
        "squared_truth": math.fsum(
            float(row["physical"]["squared_truth"]) for row in regions
        ),
        "entity_count": entity_count,
        "total_weight": total_physical_weight,
    }
    validate_reconstruction(
        report,
        expected_uniform=derived_uniform,
        expected_physical=derived_physical,
    )
    validate_reconstruction(
        report,
        expected_uniform=reconstruction["expected_uniform"],
        expected_physical=reconstruction["expected_physical"],
    )
    canonical_json_bytes(report)


def build_regional_report(
    *,
    case_id: str,
    supports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build and validate a complete case-level zero-weight report object."""

    report: dict[str, object] = {
        "schema": REGIONAL_REPORT_SCHEMA,
        "schema_version": REGIONAL_REPORT_SCHEMA_VERSION,
        "status": REGIONAL_REPORT_STATUS,
        "case_id": case_id,
        "scoring": {
            "role": "report_only",
            "weight": REGIONAL_SCORING_WEIGHT,
            "official_metric_inputs_changed": False,
            "official_score_changed": False,
        },
        "supports": dict(supports),
    }
    validate_regional_report(report, expected_case_id=case_id)
    return report


def validate_regional_report(
    report: Mapping[str, object],
    *,
    expected_case_id: str | None = None,
    required_definition_ids: Sequence[str] | None = None,
) -> None:
    """Strict, offline validation for a retained case regional report."""

    if not isinstance(report, Mapping) or set(report) != {
        "schema",
        "schema_version",
        "status",
        "case_id",
        "scoring",
        "supports",
    }:
        raise RegionalDiagnosticError("regional report top-level keys differ")
    case_id = report.get("case_id")
    if (
        report.get("schema") != REGIONAL_REPORT_SCHEMA
        or report.get("schema_version") != REGIONAL_REPORT_SCHEMA_VERSION
        or report.get("status") != REGIONAL_REPORT_STATUS
        or not isinstance(case_id, str)
        or _CASE_ID_RE.fullmatch(case_id) is None
        or (expected_case_id is not None and case_id != expected_case_id)
    ):
        raise RegionalDiagnosticError("regional report identity differs")
    scoring = report.get("scoring")
    if not isinstance(scoring, Mapping) or scoring != {
        "role": "report_only",
        "weight": 0.0,
        "official_metric_inputs_changed": False,
        "official_score_changed": False,
    }:
        raise RegionalDiagnosticError("regional report scoring boundary differs")
    supports = report.get("supports")
    if not isinstance(supports, Mapping) or not supports:
        raise RegionalDiagnosticError("regional report has no supports")
    required = (
        tuple(required_definition_ids)
        if required_definition_ids is not None
        else tuple(supports)
    )
    if set(supports) != set(required):
        raise RegionalDiagnosticError("regional report support membership differs")
    for definition_id, support in supports.items():
        definition = REGION_DEFINITIONS.get(definition_id)
        if definition is None or not isinstance(support, Mapping) or set(support) != {
            "definition",
            "definition_sha256",
            "assignment",
            "fields",
        }:
            raise RegionalDiagnosticError("regional report support contract differs")
        if support.get("definition") != definition.to_json():
            raise RegionalDiagnosticError("regional report definition bytes differ")
        if support.get("definition_sha256") != definition.sha256:
            raise RegionalDiagnosticError("regional report definition hash differs")
        assignment = support.get("assignment")
        if not isinstance(assignment, Mapping) or set(assignment) != {
            "encoding",
            "definition_sha256",
            "entity_count",
            "complete_gap_free_duplicate_free",
            "sha256",
        }:
            raise RegionalDiagnosticError("regional report assignment contract differs")
        if (
            assignment.get("encoding") != _ASSIGNMENT_ENCODING
            or assignment.get("definition_sha256") != definition.sha256
            or assignment.get("complete_gap_free_duplicate_free") is not True
            or not isinstance(assignment.get("entity_count"), int)
            or isinstance(assignment.get("entity_count"), bool)
            or assignment["entity_count"] < 1
            or not isinstance(assignment.get("sha256"), str)
            or _SHA256_RE.fullmatch(assignment["sha256"]) is None
        ):
            raise RegionalDiagnosticError("regional report assignment identity differs")
        fields = support.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            raise RegionalDiagnosticError("regional report support has no fields")
        for field_id, field in fields.items():
            if not isinstance(field_id, str) or not isinstance(field, Mapping):
                raise RegionalDiagnosticError("regional report field is invalid")
            field_contract = REGIONAL_FIELD_CONTRACTS.get(field_id)
            if field_contract is None:
                raise RegionalDiagnosticError("regional report field_id is not released")
            if _FIELD_DEFINITION_IDS[field_id] != definition_id:
                raise RegionalDiagnosticError(
                    "regional report field is bound to a different support definition"
                )
            if set(field) != {"quantity", "unit", "primary_weighting", "statistics"}:
                raise RegionalDiagnosticError("regional report field contract differs")
            expected_metadata = {
                name: field_contract[name]
                for name in ("quantity", "unit", "primary_weighting")
            }
            if {
                name: field.get(name)
                for name in ("quantity", "unit", "primary_weighting")
            } != expected_metadata:
                raise RegionalDiagnosticError(
                    "regional report field metadata differs from released contract"
                )
            statistics = field.get("statistics")
            if not isinstance(statistics, Mapping):
                raise RegionalDiagnosticError("regional report field statistics are invalid")
            validate_regional_field_report(
                statistics,
                definition,
                field_id=field_id,
            )
            if statistics.get("entity_count") != assignment.get("entity_count"):
                raise RegionalDiagnosticError(
                    "regional field and assignment entity counts differ"
                )
    canonical_json_bytes(report)


__all__ = [
    "DEFAULT_RECONSTRUCTION_ATOL",
    "DEFAULT_RECONSTRUCTION_RTOL",
    "REGIONAL_REPORT_SCHEMA",
    "REGIONAL_REPORT_SCHEMA_VERSION",
    "REGIONAL_SCORING_WEIGHT",
    "REGION_DEFINITIONS",
    "SURFACE_HORIZONTAL_NORMAL_ABS_Z_MIN",
    "SURFACE_REGION_DEFINITION",
    "SURFACE_Z_THRESHOLD_M",
    "U_INF_M_PER_S",
    "VELOCITY_DIRECTION_MIN_SPEED_M_PER_S",
    "VOLUME_REGION_DEFINITION",
    "RegionAssignmentHasher",
    "RegionDefinition",
    "RegionRule",
    "RegionalDiagnosticError",
    "RegionalFieldAccumulator",
    "build_regional_report",
    "canonical_json_bytes",
    "classify_surface_geometry",
    "surface_region_codes",
    "validate_reconstruction",
    "validate_regional_field_report",
    "validate_regional_report",
    "volume_region_codes",
]
