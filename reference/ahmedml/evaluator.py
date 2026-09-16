"""Bounded-memory candidate evaluator for complete native AhmedML fields."""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Iterator, Mapping

import numpy as np

from reference.drivaerml.accumulators import (
    DrivAerAccumulatorError,
    FinalizedFieldStatistics,
    StreamingFieldAccumulator,
)
from reference.drivaerml.prediction_chunks import (
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    PredictionChunk,
    PredictionChunkError,
    PredictionChunkManifest,
    iter_prediction_chunks,
    load_prediction_chunk_manifest,
)
from reference.drivaerml.retained_file import RetainedFileError, RetainedVerifiedFile
from reference.drivaerml.source import (
    InlineBinaryDecodeError,
    VTKDataArrayIndex,
    VTKXMLIndex,
    VTKXMLIndexError,
    index_inline_binary_vtk_xml,
    stream_inline_binary_payload,
)

from .contract import (
    PROFILE_DEFINITION_SHA256,
    REGION_DEFINITION_SHA256,
    AhmedMLSourceCase,
    AhmedMLSourceIdentity,
    SourceFileIdentity,
)
from .support import (
    SURFACE_STATIONS,
    VOLUME_STATIONS,
    AhmedMLCaseSupport,
    AhmedMLSupportError,
    load_profile_support,
    open_support_array,
)


EVIDENCE_SCHEMA = "ahmedml-candidate-case-evaluation-v1"
EVIDENCE_STATUS = "non_ranked_development_evidence_not_official_submission"
SURFACE_SUPPORT_ID = "ahmedml_surface_native_cells"
VOLUME_SUPPORT_ID = "ahmedml_volume_native_cells"
DEFAULT_MAX_PREDICTION_CHUNK_ROWS = 1_000_000
DEFAULT_ENCODED_CHUNK_BYTES = 8 * 1024 * 1024
Q_REF = 0.5
A_REF = 0.112032
FORCE_DENOMINATOR = Q_REF * A_REF
REGION_IDS = ("near_body", "wake", "farfield")
_VTK_DTYPES = {
    ("LittleEndian", "Float32"): np.dtype("<f4"),
    ("LittleEndian", "Float64"): np.dtype("<f8"),
    ("BigEndian", "Float32"): np.dtype(">f4"),
    ("BigEndian", "Float64"): np.dtype(">f8"),
}


class AhmedMLCandidateEvaluatorError(ValueError):
    """Raised when native AhmedML evidence cannot be produced exactly."""


@dataclass(frozen=True)
class _FieldResult:
    field_id: str
    source_payload_sha256: str
    source_payload_bytes: int
    statistics: FinalizedFieldStatistics
    relative_l1_uniform_percent: float
    relative_l1_physical_percent: float
    regional_diagnostics: Mapping[str, object] | None
    sampled_truth: np.ndarray | None
    sampled_prediction: np.ndarray | None
    truth_force_xyz: np.ndarray | None
    prediction_force_xyz: np.ndarray | None
    chunk_sha256: tuple[str, ...]


@dataclass(frozen=True)
class CandidateCaseEvaluation:
    """Complete one-case evidence, explicitly ineligible for ranking."""

    value: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        return dict(self.value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AhmedMLCandidateEvaluatorError(f"{label} must be a positive integer")
    return value


@contextlib.contextmanager
def _open_verified_source(
    identity: SourceFileIdentity,
    dataset_root: str | Path,
    *,
    label: str,
) -> Iterator[RetainedVerifiedFile]:
    declared = identity.resolve(dataset_root)
    try:
        materialized = declared.resolve(strict=True)
        with RetainedVerifiedFile.open(materialized, label=label) as retained:
            if retained.snapshot.size_bytes != identity.size_bytes:
                raise AhmedMLCandidateEvaluatorError(
                    f"{label} size differs from the public source identity"
                )
            if retained.sha256() != identity.sha256:
                raise AhmedMLCandidateEvaluatorError(
                    f"{label} SHA-256 differs from the public source identity"
                )
            retained.handle.seek(0)
            yield retained
            retained.assert_unchanged(context="while it was evaluated")
    except AhmedMLCandidateEvaluatorError:
        raise
    except (OSError, RetainedFileError) as error:
        raise AhmedMLCandidateEvaluatorError(f"cannot verify {label}: {error}") from error


def _index_source(
    source: BinaryIO,
    *,
    expected_type: str,
    expected_cells: int,
) -> VTKXMLIndex:
    try:
        index = index_inline_binary_vtk_xml(source)
    except VTKXMLIndexError as error:
        raise AhmedMLCandidateEvaluatorError(str(error)) from error
    if index.dataset_type != expected_type or len(index.pieces) != 1:
        raise AhmedMLCandidateEvaluatorError(
            f"native source must contain one {expected_type} Piece"
        )
    if index.pieces[0].number_of_cells != expected_cells:
        raise AhmedMLCandidateEvaluatorError(
            "native source tuple count differs from the public source identity"
        )
    return index


def _required_array(
    index: VTKXMLIndex,
    name: str,
    components: int,
) -> VTKDataArrayIndex:
    matches = index.arrays_for(association="CellData", name=name)
    if len(matches) != 1:
        raise AhmedMLCandidateEvaluatorError(
            f"native VTK source must contain exactly one CellData {name!r}"
        )
    result = matches[0]
    if (
        result.piece_index != 0
        or result.number_of_components != components
        or result.vtk_type not in {"Float32", "Float64"}
    ):
        raise AhmedMLCandidateEvaluatorError(
            f"native CellData {name!r} has the wrong piece, type, or components"
        )
    return result


def _manifest(
    value: PredictionChunkManifest | str | Path,
) -> PredictionChunkManifest:
    if isinstance(value, PredictionChunkManifest):
        # Reparse the retained pathname so callers cannot hand the evaluator a
        # stale in-memory declaration after changing its file.
        parsed = load_prediction_chunk_manifest(value.path)
        if parsed.sha256 != value.sha256:
            raise AhmedMLCandidateEvaluatorError("prediction manifest changed")
        return parsed
    return load_prediction_chunk_manifest(value)


def _validate_manifest(
    manifest: PredictionChunkManifest,
    *,
    case_id: str,
    support_id: str,
    expected_count: int,
    maximum_rows: int,
) -> None:
    if (
        manifest.case_id != case_id
        or manifest.support_id != support_id
        or manifest.association != "CellData"
        or manifest.total_row_count != expected_count
    ):
        raise AhmedMLCandidateEvaluatorError(
            f"prediction manifest does not bind {case_id}/{support_id}"
        )
    if any(chunk.row_count > maximum_rows for chunk in manifest.chunks):
        raise AhmedMLCandidateEvaluatorError(
            f"prediction chunks may contain at most {maximum_rows} rows"
        )


class _RegionalSums:
    """Three-region additive scalar/vector statistics."""

    def __init__(self, components: int) -> None:
        self.components = components
        self.count = np.zeros(3, dtype=np.int64)
        self.weight = np.zeros(3, dtype=np.float64)
        self.uniform_absolute_error = np.zeros(3, dtype=np.float64)
        self.uniform_squared_error = np.zeros(3, dtype=np.float64)
        self.uniform_squared_truth = np.zeros(3, dtype=np.float64)
        self.physical_absolute_error = np.zeros(3, dtype=np.float64)
        self.physical_squared_error = np.zeros(3, dtype=np.float64)
        self.physical_squared_truth = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _bins(codes: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(codes, weights=values, minlength=3).astype(
            np.float64, copy=False
        )

    def add(
        self,
        codes: np.ndarray,
        truth: np.ndarray,
        prediction: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        normalized = np.asarray(codes)
        if normalized.dtype != np.uint8 or normalized.ndim != 1:
            raise AhmedMLCandidateEvaluatorError("region codes must be one-dimensional uint8")
        if np.any(normalized > 2):
            raise AhmedMLCandidateEvaluatorError("region codes must lie in [0, 2]")
        truth_2d = truth[:, None] if truth.ndim == 1 else truth
        pred_2d = prediction[:, None] if prediction.ndim == 1 else prediction
        if truth_2d.shape != pred_2d.shape or truth_2d.shape[1] != self.components:
            raise AhmedMLCandidateEvaluatorError("regional field shape differs")
        error = pred_2d.astype(np.float64) - truth_2d.astype(np.float64)
        truth64 = truth_2d.astype(np.float64)
        squared_error = np.einsum("ij,ij->i", error, error, optimize=False)
        squared_truth = np.einsum("ij,ij->i", truth64, truth64, optimize=False)
        absolute_error = np.sqrt(squared_error)
        self.count += self._bins(normalized).astype(np.int64)
        self.weight += self._bins(normalized, weights)
        self.uniform_absolute_error += self._bins(normalized, absolute_error)
        self.uniform_squared_error += self._bins(normalized, squared_error)
        self.uniform_squared_truth += self._bins(normalized, squared_truth)
        self.physical_absolute_error += self._bins(normalized, weights * absolute_error)
        self.physical_squared_error += self._bins(normalized, weights * squared_error)
        self.physical_squared_truth += self._bins(normalized, weights * squared_truth)

    @staticmethod
    def _metrics(
        count_or_weight: float,
        absolute_error: float,
        squared_error: float,
        squared_truth: float,
    ) -> dict[str, float | None]:
        return {
            # Retain additive sufficient statistics so a split-level regional
            # report can be reconstructed exactly without reopening the very
            # large native fields or averaging already-normalized errors.
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "squared_truth": squared_truth,
            "total_weight": count_or_weight,
            "relative_l2_percent": (
                100.0 * math.sqrt(squared_error / squared_truth)
                if squared_truth > 0.0
                else None
            ),
            "mae": absolute_error / count_or_weight,
            "rmse": math.sqrt(squared_error / count_or_weight),
        }

    def finalize(
        self,
        expected_uniform: object,
        expected_physical: object,
    ) -> Mapping[str, object]:
        if np.any(self.count <= 0) or np.any(self.weight <= 0.0):
            raise AhmedMLCandidateEvaluatorError("every AhmedML volume region must be non-empty")
        if int(np.sum(self.count)) != int(getattr(expected_uniform, "entity_count")):
            raise AhmedMLCandidateEvaluatorError("regional counts do not reconstruct the field")
        checks = (
            (float(np.sum(self.uniform_squared_error)), float(getattr(expected_uniform, "squared_error"))),
            (float(np.sum(self.uniform_squared_truth)), float(getattr(expected_uniform, "squared_truth"))),
            (float(np.sum(self.physical_squared_error)), float(getattr(expected_physical, "squared_error"))),
            (float(np.sum(self.physical_squared_truth)), float(getattr(expected_physical, "squared_truth"))),
        )
        for actual, expected in checks:
            if not math.isclose(actual, expected, rel_tol=5.0e-12, abs_tol=1.0e-12):
                raise AhmedMLCandidateEvaluatorError(
                    "regional sufficient statistics do not reconstruct the whole field"
                )
        regions: dict[str, object] = {}
        for code, region_id in enumerate(REGION_IDS):
            regions[region_id] = {
                "code": code,
                "entity_count": int(self.count[code]),
                "physical_weight_sum": float(self.weight[code]),
                "uniform": self._metrics(
                    float(self.count[code]),
                    float(self.uniform_absolute_error[code]),
                    float(self.uniform_squared_error[code]),
                    float(self.uniform_squared_truth[code]),
                ),
                "physical": self._metrics(
                    float(self.weight[code]),
                    float(self.physical_absolute_error[code]),
                    float(self.physical_squared_error[code]),
                    float(self.physical_squared_truth[code]),
                ),
            }
        return MappingProxyType(regions)


class _FieldSink:
    """Join a decoded native truth payload to verified prediction chunks."""

    def __init__(
        self,
        *,
        vtk_index: VTKXMLIndex,
        array: VTKDataArrayIndex,
        manifest: PredictionChunkManifest,
        field_name: str,
        weights: np.ndarray,
        region_codes: np.ndarray | None,
        sample_ids: np.ndarray | None,
        sample_component: int | None,
        sample_scale: float,
        area_vectors: np.ndarray | None,
        force_sign: float | None,
        hash_chunk_bytes: int,
        validation_block_rows: int,
    ) -> None:
        self.vtk_index = vtk_index
        self.array = array
        self.manifest = manifest
        self.field_name = field_name
        self.components = array.number_of_components
        try:
            self.dtype = _VTK_DTYPES[(vtk_index.byte_order, array.vtk_type)]
        except KeyError as error:
            raise AhmedMLCandidateEvaluatorError("unsupported native field dtype") from error
        self.tuple_bytes = self.dtype.itemsize * self.components
        self.weights = weights
        self.region_codes = region_codes
        self.area_vectors = area_vectors
        self.force_sign = force_sign
        self.accumulator = StreamingFieldAccumulator(
            expected_entity_count=manifest.total_row_count,
            component_count=self.components,
        )
        self.regional = (
            _RegionalSums(self.components) if region_codes is not None else None
        )
        self.iterator = iter_prediction_chunks(
            manifest,
            hash_chunk_bytes=hash_chunk_bytes,
            validation_block_rows=validation_block_rows,
        )
        self.current = next(self.iterator, None)
        self.pending = bytearray()
        self.cursor = 0
        self.chunk_sha256: list[str] = []
        self.uniform_absolute_truth = 0.0
        self.physical_absolute_truth = 0.0
        self.truth_force = np.zeros(3, dtype=np.float64) if force_sign is not None else None
        self.prediction_force = (
            np.zeros(3, dtype=np.float64) if force_sign is not None else None
        )
        self.sample_ids = None if sample_ids is None else np.asarray(sample_ids).reshape(-1)
        self.sample_component = sample_component
        self.sample_scale = float(sample_scale)
        self.sample_truth = (
            None
            if sample_ids is None
            else np.full(self.sample_ids.shape, np.nan, dtype=np.float64)
        )
        self.sample_prediction = (
            None
            if sample_ids is None
            else np.full(self.sample_ids.shape, np.nan, dtype=np.float64)
        )

    def _capture_samples(
        self,
        start: int,
        stop: int,
        truth: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        if self.sample_ids is None:
            return
        positions = np.nonzero((self.sample_ids >= start) & (self.sample_ids < stop))[0]
        if not len(positions):
            return
        local = self.sample_ids[positions] - start
        if self.sample_component is None:
            truth_values = truth[local]
            prediction_values = prediction[local]
        else:
            truth_values = truth[local, self.sample_component]
            prediction_values = prediction[local, self.sample_component]
        assert self.sample_truth is not None and self.sample_prediction is not None
        self.sample_truth[positions] = np.asarray(truth_values, dtype=np.float64) * self.sample_scale
        self.sample_prediction[positions] = (
            np.asarray(prediction_values, dtype=np.float64) * self.sample_scale
        )

    def _accumulate_force(
        self,
        start: int,
        stop: int,
        truth: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        if self.force_sign is None:
            return
        assert self.truth_force is not None and self.prediction_force is not None
        if self.components == 1:
            if self.area_vectors is None:
                raise AhmedMLCandidateEvaluatorError("pressure force needs area vectors")
            geometry = self.area_vectors[start:stop]
            truth_force = np.asarray(truth, dtype=np.float64)[:, None] * geometry
            prediction_force = np.asarray(prediction, dtype=np.float64)[:, None] * geometry
        elif self.components == 3:
            geometry = np.asarray(self.weights[start:stop], dtype=np.float64)[:, None]
            truth_force = np.asarray(truth, dtype=np.float64) * geometry
            prediction_force = np.asarray(prediction, dtype=np.float64) * geometry
        else:  # pragma: no cover - constructor contracts permit only 1 or 3.
            raise AhmedMLCandidateEvaluatorError("force fields must have 1 or 3 components")
        self.truth_force += self.force_sign * np.sum(truth_force, axis=0, dtype=np.float64)
        self.prediction_force += self.force_sign * np.sum(
            prediction_force, axis=0, dtype=np.float64
        )

    def _consume_current(self) -> None:
        chunk = self.current
        if chunk is None:
            return
        descriptor = chunk.descriptor
        required = descriptor.row_count * self.tuple_bytes
        payload = bytes(self.pending[:required])
        del self.pending[:required]
        truth = np.frombuffer(payload, dtype=self.dtype)
        truth = truth.reshape(descriptor.row_count, self.components)
        if self.components == 1:
            truth = truth[:, 0]
        prediction = chunk.field(self.field_name)
        start, stop = descriptor.raw_cell_id_start, descriptor.raw_cell_id_stop
        weights = np.asarray(self.weights[start:stop])
        self.accumulator.add_chunk(chunk.raw_cell_id, truth, prediction, weights)
        truth_2d = truth[:, None] if truth.ndim == 1 else truth
        truth_norm = np.linalg.norm(truth_2d.astype(np.float64), axis=1)
        self.uniform_absolute_truth += float(np.sum(truth_norm, dtype=np.float64))
        self.physical_absolute_truth += float(
            np.sum(weights.astype(np.float64) * truth_norm, dtype=np.float64)
        )
        if self.regional is not None:
            assert self.region_codes is not None
            self.regional.add(
                np.asarray(self.region_codes[start:stop]),
                truth,
                prediction,
                weights,
            )
        self._capture_samples(start, stop, truth, prediction)
        self._accumulate_force(start, stop, truth, prediction)
        self.chunk_sha256.append(descriptor.sha256)
        self.cursor = stop
        self.current = next(self.iterator, None)

    def write(self, payload: bytes) -> int:
        self.pending.extend(payload)
        while self.current is not None:
            required = self.current.descriptor.row_count * self.tuple_bytes
            if len(self.pending) < required:
                break
            self._consume_current()
        if self.current is None and self.pending:
            raise AhmedMLCandidateEvaluatorError(
                f"native truth {self.field_name!r} exceeds prediction coverage"
            )
        return len(payload)

    def finish(self) -> tuple[
        FinalizedFieldStatistics,
        Mapping[str, object] | None,
        float,
        float,
    ]:
        if self.pending or self.current is not None or self.cursor != self.manifest.total_row_count:
            raise AhmedMLCandidateEvaluatorError(
                f"prediction coverage differs from native truth {self.field_name!r}"
            )
        statistics = self.accumulator.finalize()
        if self.uniform_absolute_truth <= 0.0 or self.physical_absolute_truth <= 0.0:
            raise AhmedMLCandidateEvaluatorError(
                f"relative L1 denominator is zero for {self.field_name!r}"
            )
        regional = (
            None
            if self.regional is None
            else self.regional.finalize(statistics.uniform, statistics.physical)
        )
        if self.sample_truth is not None and (
            np.any(~np.isfinite(self.sample_truth))
            or np.any(~np.isfinite(self.sample_prediction))
        ):
            raise AhmedMLCandidateEvaluatorError(
                f"profile samples for {self.field_name!r} were not covered"
            )
        return (
            statistics,
            regional,
            100.0 * statistics.uniform.absolute_error / self.uniform_absolute_truth,
            100.0 * statistics.physical.absolute_error / self.physical_absolute_truth,
        )


def _evaluate_field(
    *,
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    manifest: PredictionChunkManifest,
    field_id: str,
    field_name: str,
    weights: np.ndarray,
    region_codes: np.ndarray | None,
    sample_ids: np.ndarray | None,
    sample_component: int | None,
    sample_scale: float,
    area_vectors: np.ndarray | None,
    force_sign: float | None,
    hash_chunk_bytes: int,
    validation_block_rows: int,
    encoded_chunk_bytes: int,
) -> _FieldResult:
    sink = _FieldSink(
        vtk_index=vtk_index,
        array=array,
        manifest=manifest,
        field_name=field_name,
        weights=weights,
        region_codes=region_codes,
        sample_ids=sample_ids,
        sample_component=sample_component,
        sample_scale=sample_scale,
        area_vectors=area_vectors,
        force_sign=force_sign,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
    )
    try:
        payload = stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            sink,
            encoded_chunk_size=encoded_chunk_bytes,
        )
        statistics, regional, relative_l1_uniform, relative_l1_physical = sink.finish()
    except (
        InlineBinaryDecodeError,
        DrivAerAccumulatorError,
        PredictionChunkError,
    ) as error:
        raise AhmedMLCandidateEvaluatorError(str(error)) from error
    if payload.tuple_count != manifest.total_row_count:
        raise AhmedMLCandidateEvaluatorError("native field tuple coverage differs")
    return _FieldResult(
        field_id=field_id,
        source_payload_sha256=payload.payload_sha256,
        source_payload_bytes=payload.decoded_payload_bytes,
        statistics=statistics,
        relative_l1_uniform_percent=relative_l1_uniform,
        relative_l1_physical_percent=relative_l1_physical,
        regional_diagnostics=regional,
        sampled_truth=sink.sample_truth,
        sampled_prediction=sink.sample_prediction,
        truth_force_xyz=sink.truth_force,
        prediction_force_xyz=sink.prediction_force,
        chunk_sha256=tuple(sink.chunk_sha256),
    )


def _field_evidence(result: _FieldResult) -> dict[str, object]:
    statistics = result.statistics
    return {
        "source_payload_sha256": result.source_payload_sha256,
        "source_payload_bytes": result.source_payload_bytes,
        "entity_count": statistics.entity_count,
        "component_count": statistics.component_count,
        "uniform": {
            **asdict(statistics.uniform),
            **statistics.metric_values()["uniform"],
            "relative_l1_percent": result.relative_l1_uniform_percent,
        },
        "physical": {
            **asdict(statistics.physical),
            **statistics.metric_values()["physical"],
            "relative_l1_percent": result.relative_l1_physical_percent,
        },
    }


def _metric_values(
    surface_pressure: _FieldResult,
    surface_shear: _FieldResult,
    volume_pressure: _FieldResult,
    volume_velocity: _FieldResult,
) -> dict[str, float]:
    sp = surface_pressure.statistics.metric_values()
    sw = surface_shear.statistics.metric_values()
    vp = volume_pressure.statistics.metric_values()
    vv = volume_velocity.statistics.metric_values()
    return {
        "surface_pressure_rel_l2": sp["physical"]["relative_l2_percent"],
        "surface_pressure_equal_entity_rel_l2": sp["uniform"]["relative_l2_percent"],
        "surface_wall_shear_rel_l2": sw["physical"]["relative_l2_percent"],
        "surface_wall_shear_equal_entity_rel_l2": sw["uniform"]["relative_l2_percent"],
        "volume_pressure_rel_l2": vp["uniform"]["relative_l2_percent"],
        "volume_pressure_physical_rel_l2": vp["physical"]["relative_l2_percent"],
        "volume_velocity_rel_l2": vv["uniform"]["relative_l2_percent"],
        "volume_velocity_physical_rel_l2": vv["physical"]["relative_l2_percent"],
        "surface_pressure_rel_l1": surface_pressure.relative_l1_physical_percent,
        "surface_wall_shear_rel_l1": surface_shear.relative_l1_physical_percent,
        "volume_pressure_rel_l1": volume_pressure.relative_l1_uniform_percent,
        "volume_velocity_rel_l1": volume_velocity.relative_l1_uniform_percent,
        "surface_pressure_mae": sp["physical"]["mae"],
        "surface_pressure_rmse": sp["physical"]["rmse"],
        "surface_wall_shear_mae": sw["physical"]["mae"],
        "surface_wall_shear_rmse": sw["physical"]["rmse"],
        "volume_pressure_mae": vp["uniform"]["mae"],
        "volume_pressure_rmse": vp["uniform"]["rmse"],
        "volume_velocity_mae": vv["uniform"]["mae"],
        "volume_velocity_rmse": vv["uniform"]["rmse"],
    }


def _profiles(
    profile_support: Mapping[str, np.ndarray],
    surface_pressure: _FieldResult,
    volume_velocity: _FieldResult,
) -> dict[str, object]:
    if surface_pressure.sampled_truth is None or surface_pressure.sampled_prediction is None:
        raise AhmedMLCandidateEvaluatorError("surface profile samples are unavailable")
    if volume_velocity.sampled_truth is None or volume_velocity.sampled_prediction is None:
        raise AhmedMLCandidateEvaluatorError("volume profile samples are unavailable")
    surface_truth = surface_pressure.sampled_truth.reshape(3, 128)
    surface_prediction = surface_pressure.sampled_prediction.reshape(3, 128)
    volume_truth = volume_velocity.sampled_truth.reshape(4, 128)
    volume_prediction = volume_velocity.sampled_prediction.reshape(4, 128)
    if not np.allclose(
        surface_truth,
        profile_support["surface_truth_cp"],
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise AhmedMLCandidateEvaluatorError(
            "decoded native surface truth differs from frozen profile truth"
        )
    if not np.allclose(
        volume_truth,
        profile_support["volume_truth_ux_over_uinf"],
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise AhmedMLCandidateEvaluatorError(
            "decoded native volume truth differs from frozen profile truth"
        )
    series: list[dict[str, object]] = []
    for index, station_id in enumerate(SURFACE_STATIONS):
        series.append(
            {
                "panel_id": "pressure_profiles",
                "station_id": station_id,
                "quantity_id": "cp",
                "coordinate": profile_support["surface_coordinate"][index].tolist(),
                "truth": surface_truth[index].tolist(),
                "prediction": surface_prediction[index].tolist(),
                "sample_count": 128,
                "source": "evaluator_derived_from_complete_native_fields",
            }
        )
    for index, station_id in enumerate(VOLUME_STATIONS):
        series.append(
            {
                "panel_id": "velocity_profiles",
                "station_id": station_id,
                "quantity_id": "ux_over_uinf",
                "coordinate": profile_support["volume_coordinate"][index].tolist(),
                "truth": volume_truth[index].tolist(),
                "prediction": volume_prediction[index].tolist(),
                "sample_count": 128,
                "source": "evaluator_derived_from_complete_native_fields",
            }
        )
    return {
        "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
        "participant_profile_payload_accepted": False,
        "series": series,
    }


def _force_result(
    pressure: _FieldResult,
    shear: _FieldResult,
) -> dict[str, object]:
    arrays = (
        pressure.truth_force_xyz,
        pressure.prediction_force_xyz,
        shear.truth_force_xyz,
        shear.prediction_force_xyz,
    )
    if any(value is None for value in arrays):
        raise AhmedMLCandidateEvaluatorError("surface force accumulation is unavailable")
    truth = np.asarray(pressure.truth_force_xyz) + np.asarray(shear.truth_force_xyz)
    prediction = np.asarray(pressure.prediction_force_xyz) + np.asarray(
        shear.prediction_force_xyz
    )
    return {
        "traction_rule": "pMean*A_outward - wallShearStressMean*|A|",
        "reference_dynamic_pressure_times_area": FORCE_DENOMINATOR,
        "truth_force_xyz": truth.tolist(),
        "prediction_force_xyz": prediction.tolist(),
        "truth_cd": float(truth[0] / FORCE_DENOMINATOR),
        "prediction_cd": float(prediction[0] / FORCE_DENOMINATOR),
        "truth_cl": float(truth[2] / FORCE_DENOMINATOR),
        "prediction_cl": float(prediction[2] / FORCE_DENOMINATOR),
    }


def evaluate_candidate_case(
    *,
    case_id: str,
    dataset_root: str | Path,
    source_identity: AhmedMLSourceIdentity,
    case_support: AhmedMLCaseSupport,
    surface_prediction_manifest: PredictionChunkManifest | str | Path,
    volume_prediction_manifest: PredictionChunkManifest | str | Path,
    maximum_prediction_chunk_rows: int = DEFAULT_MAX_PREDICTION_CHUNK_ROWS,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
    encoded_chunk_bytes: int = DEFAULT_ENCODED_CHUNK_BYTES,
) -> CandidateCaseEvaluation:
    """Evaluate one complete native case and derive its profile payload.

    This function intentionally has no ranked/official mode.  Promotion of a
    future frozen evaluator is a separate owner-reviewed release operation.
    """

    case = source_identity.case(case_id)
    if case_support.case_id != case_id:
        raise AhmedMLCandidateEvaluatorError("case support belongs to another case")
    maximum_rows = _positive_integer(
        maximum_prediction_chunk_rows, "maximum_prediction_chunk_rows"
    )
    hash_bytes = _positive_integer(hash_chunk_bytes, "hash_chunk_bytes")
    validation_rows = _positive_integer(
        validation_block_rows, "validation_block_rows"
    )
    encoded_bytes = _positive_integer(encoded_chunk_bytes, "encoded_chunk_bytes")
    try:
        surface_manifest = _manifest(surface_prediction_manifest)
        volume_manifest = _manifest(volume_prediction_manifest)
    except PredictionChunkError as error:
        raise AhmedMLCandidateEvaluatorError(str(error)) from error
    _validate_manifest(
        surface_manifest,
        case_id=case_id,
        support_id=SURFACE_SUPPORT_ID,
        expected_count=case.surface_entity_count,
        maximum_rows=maximum_rows,
    )
    _validate_manifest(
        volume_manifest,
        case_id=case_id,
        support_id=VOLUME_SUPPORT_ID,
        expected_count=case.volume_entity_count,
        maximum_rows=maximum_rows,
    )
    try:
        profile_support = load_profile_support(case_support)
        with (
            _open_verified_source(
                case.boundary, dataset_root, label=f"{case_id} boundary"
            ) as boundary,
            _open_verified_source(
                case.surface_cell_area,
                dataset_root,
                label=f"{case_id} surface area",
            ) as surface_area_source,
            _open_verified_source(
                case.volume, dataset_root, label=f"{case_id} volume"
            ) as volume,
            open_support_array(case_support, "surface_area_vector") as area_vectors,
            open_support_array(case_support, "volume_cell_volume") as cell_volumes,
            open_support_array(case_support, "volume_region_code") as region_codes,
        ):
            surface_area_source.handle.seek(0)
            surface_areas = np.load(
                surface_area_source.handle, mmap_mode=None, allow_pickle=False
            )
            if (
                surface_areas.dtype != np.dtype("float32")
                or surface_areas.shape != (case.surface_entity_count,)
                or np.any(~np.isfinite(surface_areas))
                or np.any(surface_areas <= 0.0)
            ):
                raise AhmedMLCandidateEvaluatorError(
                    "public surface area array violates its native contract"
                )
            surface_area_source.assert_unchanged(
                context="while its area array was consumed"
            )
            boundary_index = _index_source(
                boundary.handle,
                expected_type="PolyData",
                expected_cells=case.surface_entity_count,
            )
            volume_index = _index_source(
                volume.handle,
                expected_type="UnstructuredGrid",
                expected_cells=case.volume_entity_count,
            )
            surface_pressure_array = _required_array(boundary_index, "pMean", 1)
            surface_shear_array = _required_array(
                boundary_index, "wallShearStressMean", 3
            )
            volume_pressure_array = _required_array(volume_index, "pMean", 1)
            volume_velocity_array = _required_array(volume_index, "UMean", 3)

            surface_pressure = _evaluate_field(
                stream=boundary.handle,
                vtk_index=boundary_index,
                array=surface_pressure_array,
                manifest=surface_manifest,
                field_id="surface_pressure",
                field_name="pMean",
                weights=surface_areas,
                region_codes=None,
                sample_ids=profile_support["surface_raw_cell_id"],
                sample_component=None,
                sample_scale=2.0,
                area_vectors=area_vectors,
                force_sign=1.0,
                hash_chunk_bytes=hash_bytes,
                validation_block_rows=validation_rows,
                encoded_chunk_bytes=encoded_bytes,
            )
            surface_shear = _evaluate_field(
                stream=boundary.handle,
                vtk_index=boundary_index,
                array=surface_shear_array,
                manifest=surface_manifest,
                field_id="surface_wall_shear",
                field_name="wallShearStressMean",
                weights=surface_areas,
                region_codes=None,
                sample_ids=None,
                sample_component=None,
                sample_scale=1.0,
                area_vectors=None,
                force_sign=-1.0,
                hash_chunk_bytes=hash_bytes,
                validation_block_rows=validation_rows,
                encoded_chunk_bytes=encoded_bytes,
            )
            volume_pressure = _evaluate_field(
                stream=volume.handle,
                vtk_index=volume_index,
                array=volume_pressure_array,
                manifest=volume_manifest,
                field_id="volume_pressure",
                field_name="pMean",
                weights=cell_volumes,
                region_codes=region_codes,
                sample_ids=None,
                sample_component=None,
                sample_scale=1.0,
                area_vectors=None,
                force_sign=None,
                hash_chunk_bytes=hash_bytes,
                validation_block_rows=validation_rows,
                encoded_chunk_bytes=encoded_bytes,
            )
            volume_velocity = _evaluate_field(
                stream=volume.handle,
                vtk_index=volume_index,
                array=volume_velocity_array,
                manifest=volume_manifest,
                field_id="volume_velocity",
                field_name="UMean",
                weights=cell_volumes,
                region_codes=region_codes,
                sample_ids=profile_support["volume_raw_cell_id"],
                sample_component=0,
                sample_scale=1.0,
                area_vectors=None,
                force_sign=None,
                hash_chunk_bytes=hash_bytes,
                validation_block_rows=validation_rows,
                encoded_chunk_bytes=encoded_bytes,
            )
    except (AhmedMLSupportError, RetainedFileError) as error:
        raise AhmedMLCandidateEvaluatorError(str(error)) from error

    fields = {
        result.field_id: _field_evidence(result)
        for result in (
            surface_pressure,
            surface_shear,
            volume_pressure,
            volume_velocity,
        )
    }
    metric_values = _metric_values(
        surface_pressure, surface_shear, volume_pressure, volume_velocity
    )
    profiles = _profiles(profile_support, surface_pressure, volume_velocity)
    force = _force_result(surface_pressure, surface_shear)
    evidence: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": 1,
        "status": EVIDENCE_STATUS,
        "official_submission": False,
        "leaderboard_eligible": False,
        "case_id": case_id,
        "source": {
            "repository_id": source_identity.repository_id,
            "repository_revision": source_identity.repository_revision,
            "source_identity_sha256": source_identity.sha256,
            "boundary_sha256": case.boundary.sha256,
            "surface_cell_area_sha256": case.surface_cell_area.sha256,
            "volume_sha256": case.volume.sha256,
        },
        "support": {
            "case_support_sha256": case_support.manifest_sha256,
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
            "regional_definition_sha256": REGION_DEFINITION_SHA256,
        },
        "prediction_inputs": {
            "surface": {
                "manifest_sha256": surface_manifest.sha256,
                "chunk_sha256": list(surface_pressure.chunk_sha256),
                "entity_count": case.surface_entity_count,
            },
            "volume": {
                "manifest_sha256": volume_manifest.sha256,
                "chunk_sha256": list(volume_pressure.chunk_sha256),
                "entity_count": case.volume_entity_count,
            },
        },
        "coverage": {
            "complete_case": True,
            "surface_raw_cell_interval": [0, case.surface_entity_count],
            "volume_raw_cell_interval": [0, case.volume_entity_count],
            "gap_free_duplicate_free": True,
        },
        "metric_values": metric_values,
        "field_statistics": fields,
        "force_coefficients": force,
        "profiles": profiles,
        "report_only_regional_diagnostics": {
            "definition_sha256": REGION_DEFINITION_SHA256,
            "ranking_effect": "none",
            "volume_pressure": dict(volume_pressure.regional_diagnostics or {}),
            "volume_velocity": dict(volume_velocity.regional_diagnostics or {}),
        },
        "execution": {
            "maximum_prediction_chunk_rows": maximum_rows,
            "encoded_truth_chunk_bytes": encoded_bytes,
            "prediction_hash_chunk_bytes": hash_bytes,
            "prediction_validation_block_rows": validation_rows,
        },
    }
    return CandidateCaseEvaluation(MappingProxyType(evidence))


__all__ = [
    "AhmedMLCandidateEvaluatorError",
    "CandidateCaseEvaluation",
    "DEFAULT_ENCODED_CHUNK_BYTES",
    "DEFAULT_MAX_PREDICTION_CHUNK_ROWS",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_STATUS",
    "SURFACE_SUPPORT_ID",
    "VOLUME_SUPPORT_ID",
    "evaluate_candidate_case",
]
