"""Bounded-memory candidate evaluator for complete native WindsorML fields.

Three structural differences from :mod:`reference.ahmedml.evaluator`:

* **The surface support is PointData.** Raw IDs index native boundary points,
  and the per-point quadrature weights are the published
  ``boundary_dual_area_N.npy`` sidecar rather than an evaluator-generated array.
  Entity counts are therefore checked against ``NumberOfPoints``.

* **No generated support artifacts are required for v1.** The surface weights
  ship with the dataset, and the volume metric is equal-cell weighted by
  decision, so there are no cell-volume sidecars to build. Only the profile
  panels would need generated support, and those stay out of v1 while the
  velocity stations are under owner review.

* **Vector metrics are assembled from separately streamed scalar components.**
  WindsorML stores ``cfxavg``/``cfyavg``/``cfzavg`` and
  ``velocityxavg``/``velocityyavg``/``velocityzavg`` as independent scalar
  arrays, and the inline-binary reader streams one array at a time. Relative L2
  recombines exactly, because both its numerator and denominator are sums of
  per-component squares. Euclidean MAE and RMSE do **not** recombine that way
  (``|a| + |b| != |(a, b)|``), so those are reported per component only and a
  vector-magnitude MAE is deliberately never claimed.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from reference.drivaerml.accumulators import (
    DrivAerAccumulatorError,
    FinalizedFieldStatistics,
    StreamingFieldAccumulator,
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
    DRAG_AXIS_INDEX,
    FORCE_REFERENCE_AREA_M2,
    LIFT_AXIS_INDEX,
    SIDE_AXIS_INDEX,
    SourceFileIdentity,
    WindsorMLContractError,
    WindsorMLSourceCase,
    classify_force_replay,
)
from .prediction_chunks import (
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    WINDSORML_SURFACE_SUPPORT_ID,
    WINDSORML_VOLUME_SUPPORT_ID,
    PredictionChunkError,
    PredictionChunkManifest,
    iter_prediction_chunks,
    load_prediction_chunk_manifest,
)


EVIDENCE_SCHEMA = "windsorml-candidate-case-evaluation-v1"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_STATUS = "non_ranked_development_evidence_not_official_submission"
DEFAULT_MAX_PREDICTION_CHUNK_ROWS = 1_000_000
DEFAULT_ENCODED_CHUNK_BYTES = 8 * 1024 * 1024

SURFACE_PRESSURE_FIELD = "cpavg"
SURFACE_SHEAR_FIELDS = ("cfxavg", "cfyavg", "cfzavg")
VOLUME_VELOCITY_FIELDS = ("velocityxavg", "velocityyavg", "velocityzavg")
VOLUME_PRESSURE_FIELD = "pressureavg"
SURFACE_NORMALS_ARRAY = "Normals"

_VTK_DTYPES = {
    ("LittleEndian", "Float32"): np.dtype("<f4"),
    ("LittleEndian", "Float64"): np.dtype("<f8"),
    ("BigEndian", "Float32"): np.dtype(">f4"),
    ("BigEndian", "Float64"): np.dtype(">f8"),
}


class WindsorMLCandidateEvaluatorError(ValueError):
    """Raised when native WindsorML evidence cannot be produced exactly."""


@dataclass(frozen=True)
class _ScalarResult:
    """One complete scalar field's statistics and force contribution."""

    field_name: str
    source_payload_sha256: str
    source_payload_bytes: int
    statistics: FinalizedFieldStatistics
    truth_force_xyz: np.ndarray | None
    prediction_force_xyz: np.ndarray | None
    sampled_truth: np.ndarray | None
    sampled_prediction: np.ndarray | None
    chunk_sha256: tuple[str, ...]


@dataclass(frozen=True)
class CandidateCaseEvaluation:
    """Complete one-case evidence, explicitly ineligible for ranking."""

    value: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        return dict(self.value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WindsorMLCandidateEvaluatorError(f"{label} must be a positive integer")
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
                raise WindsorMLCandidateEvaluatorError(
                    f"{label} size differs from the public source identity"
                )
            if retained.sha256() != identity.sha256:
                raise WindsorMLCandidateEvaluatorError(
                    f"{label} SHA-256 differs from the public source identity"
                )
            retained.handle.seek(0)
            yield retained
            retained.assert_unchanged(context="while it was evaluated")
    except WindsorMLCandidateEvaluatorError:
        raise
    except (OSError, RetainedFileError) as error:
        raise WindsorMLCandidateEvaluatorError(
            f"cannot verify {label}: {error}"
        ) from error


def _index_source(
    source: BinaryIO,
    *,
    expected_entities: int,
    association: str,
) -> VTKXMLIndex:
    """Index one inline-binary VTU and bind its entity count.

    ``association`` selects which Piece count the support's raw IDs index:
    ``PointData`` for the WindsorML surface, ``CellData`` for the volume.
    """

    try:
        index = index_inline_binary_vtk_xml(source)
    except VTKXMLIndexError as error:
        raise WindsorMLCandidateEvaluatorError(str(error)) from error
    if index.dataset_type != "UnstructuredGrid" or len(index.pieces) != 1:
        raise WindsorMLCandidateEvaluatorError(
            "native source must contain one UnstructuredGrid Piece"
        )
    piece = index.pieces[0]
    actual = (
        piece.number_of_points if association == "PointData" else piece.number_of_cells
    )
    if actual != expected_entities:
        raise WindsorMLCandidateEvaluatorError(
            f"native {association} entity count {actual} differs from the public "
            f"source identity ({expected_entities})"
        )
    return index


def _required_array(
    index: VTKXMLIndex,
    name: str,
    components: int,
    association: str,
) -> VTKDataArrayIndex:
    matches = index.arrays_for(association=association, name=name)
    if len(matches) != 1:
        raise WindsorMLCandidateEvaluatorError(
            f"native VTK source must contain exactly one {association} {name!r}"
        )
    result = matches[0]
    if (
        result.piece_index != 0
        or result.number_of_components != components
        or result.vtk_type not in {"Float32", "Float64"}
    ):
        raise WindsorMLCandidateEvaluatorError(
            f"native {association} {name!r} has the wrong piece, type, or components"
        )
    return result


def _manifest(value: PredictionChunkManifest | str | Path) -> PredictionChunkManifest:
    if isinstance(value, PredictionChunkManifest):
        # Reparse the retained pathname so callers cannot hand the evaluator a
        # stale in-memory declaration after changing its file.
        parsed = load_prediction_chunk_manifest(value.path)
        if parsed.sha256 != value.sha256:
            raise WindsorMLCandidateEvaluatorError("prediction manifest changed")
        return parsed
    return load_prediction_chunk_manifest(value)


def _validate_manifest(
    manifest: PredictionChunkManifest,
    *,
    case_id: str,
    support_id: str,
    association: str,
    expected_count: int,
    maximum_rows: int,
) -> None:
    if (
        manifest.case_id != case_id
        or manifest.support_id != support_id
        or manifest.association != association
        or manifest.total_row_count != expected_count
    ):
        raise WindsorMLCandidateEvaluatorError(
            f"prediction manifest does not bind {case_id}/{support_id}"
        )
    if any(chunk.row_count > maximum_rows for chunk in manifest.chunks):
        raise WindsorMLCandidateEvaluatorError(
            f"prediction chunks may contain at most {maximum_rows} rows"
        )


class _ScalarSink:
    """Join one decoded native scalar truth payload to verified predictions.

    ``force_vectors`` optionally supplies a per-entity 3-vector that converts
    this scalar into a force contribution: the outward area vector for pressure
    (``-cp * n * dA``) or a single-axis weight vector for a shear component
    (``cf_c * dA`` on axis ``c``).
    """

    def __init__(
        self,
        *,
        vtk_index: VTKXMLIndex,
        array: VTKDataArrayIndex,
        manifest: PredictionChunkManifest,
        field_name: str,
        weights: np.ndarray,
        force_vectors: np.ndarray | None,
        hash_chunk_bytes: int,
        validation_block_rows: int,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        if array.number_of_components != 1:
            raise WindsorMLCandidateEvaluatorError(
                f"{field_name!r} must be a scalar native array"
            )
        self.field_name = field_name
        try:
            self.dtype = _VTK_DTYPES[(vtk_index.byte_order, array.vtk_type)]
        except KeyError as error:
            raise WindsorMLCandidateEvaluatorError(
                "unsupported native field dtype"
            ) from error
        self.tuple_bytes = self.dtype.itemsize
        self.manifest = manifest
        self.weights = weights
        self.force_vectors = force_vectors
        self.accumulator = StreamingFieldAccumulator(
            expected_entity_count=manifest.total_row_count,
            component_count=1,
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
        self.truth_force = (
            np.zeros(3, dtype=np.float64) if force_vectors is not None else None
        )
        self.prediction_force = (
            np.zeros(3, dtype=np.float64) if force_vectors is not None else None
        )
        # Profile samples are captured during the same single pass: the frozen
        # support names native entity IDs, and both truth and prediction are
        # read at those IDs so a profile can never be a second prediction path.
        self.sample_ids = None if sample_ids is None else np.asarray(sample_ids).reshape(-1)
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
        assert self.sample_truth is not None and self.sample_prediction is not None
        self.sample_truth[positions] = np.asarray(truth, dtype=np.float64)[local]
        self.sample_prediction[positions] = np.asarray(prediction, dtype=np.float64)[local]

    def _accumulate_force(
        self,
        start: int,
        stop: int,
        truth: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        if self.force_vectors is None:
            return
        assert self.truth_force is not None and self.prediction_force is not None
        geometry = self.force_vectors[start:stop]
        self.truth_force += np.sum(
            np.asarray(truth, dtype=np.float64)[:, None] * geometry,
            axis=0,
            dtype=np.float64,
        )
        self.prediction_force += np.sum(
            np.asarray(prediction, dtype=np.float64)[:, None] * geometry,
            axis=0,
            dtype=np.float64,
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
        prediction = chunk.field(self.field_name)
        start, stop = descriptor.raw_cell_id_start, descriptor.raw_cell_id_stop
        weights = np.asarray(self.weights[start:stop])
        self.accumulator.add_chunk(chunk.raw_cell_id, truth, prediction, weights)
        self._accumulate_force(start, stop, truth, prediction)
        self._capture_samples(start, stop, truth, prediction)
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
            raise WindsorMLCandidateEvaluatorError(
                f"native truth {self.field_name!r} exceeds prediction coverage"
            )
        return len(payload)

    def finish(self) -> FinalizedFieldStatistics:
        if (
            self.pending
            or self.current is not None
            or self.cursor != self.manifest.total_row_count
        ):
            raise WindsorMLCandidateEvaluatorError(
                f"prediction coverage differs from native truth {self.field_name!r}"
            )
        if self.sample_truth is not None and (
            np.any(~np.isfinite(self.sample_truth))
            or np.any(~np.isfinite(self.sample_prediction))
        ):
            raise WindsorMLCandidateEvaluatorError(
                f"profile samples for {self.field_name!r} were not covered"
            )
        return self.accumulator.finalize()


def _evaluate_scalar(
    *,
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    manifest: PredictionChunkManifest,
    field_name: str,
    weights: np.ndarray,
    force_vectors: np.ndarray | None,
    hash_chunk_bytes: int,
    validation_block_rows: int,
    encoded_chunk_bytes: int,
    sample_ids: np.ndarray | None = None,
) -> _ScalarResult:
    sink = _ScalarSink(
        vtk_index=vtk_index,
        array=array,
        manifest=manifest,
        field_name=field_name,
        weights=weights,
        force_vectors=force_vectors,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
        sample_ids=sample_ids,
    )
    try:
        payload = stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            sink,
            encoded_chunk_size=encoded_chunk_bytes,
        )
        statistics = sink.finish()
    except (
        InlineBinaryDecodeError,
        DrivAerAccumulatorError,
        PredictionChunkError,
    ) as error:
        raise WindsorMLCandidateEvaluatorError(str(error)) from error
    if payload.tuple_count != manifest.total_row_count:
        raise WindsorMLCandidateEvaluatorError("native field tuple coverage differs")
    return _ScalarResult(
        field_name=field_name,
        source_payload_sha256=payload.payload_sha256,
        source_payload_bytes=payload.decoded_payload_bytes,
        statistics=statistics,
        truth_force_xyz=sink.truth_force,
        prediction_force_xyz=sink.prediction_force,
        sampled_truth=sink.sample_truth,
        sampled_prediction=sink.sample_prediction,
        chunk_sha256=tuple(sink.chunk_sha256),
    )


def _read_inline_array(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    *,
    encoded_chunk_bytes: int,
) -> np.ndarray:
    """Materialize one native array in full.

    Only used for the boundary ``Normals`` array (about 26 MB for a 2.2M-point
    surface), never for a volume payload.
    """

    collected = bytearray()

    class _Collector:
        def write(self, payload: bytes) -> int:
            collected.extend(payload)
            return len(payload)

    try:
        dtype = _VTK_DTYPES[(vtk_index.byte_order, array.vtk_type)]
    except KeyError as error:
        raise WindsorMLCandidateEvaluatorError("unsupported native dtype") from error
    try:
        stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            _Collector(),
            encoded_chunk_size=encoded_chunk_bytes,
        )
    except InlineBinaryDecodeError as error:
        raise WindsorMLCandidateEvaluatorError(str(error)) from error
    values = np.frombuffer(bytes(collected), dtype=dtype)
    return values.reshape(-1, array.number_of_components).astype(np.float64)


def _vector_relative_l2_percent(
    components: Sequence[FinalizedFieldStatistics],
    *,
    weighting: str,
) -> float:
    """Recombine a vector relative L2 from independent scalar component sums.

    Exact, because the numerator and denominator are both sums of per-component
    squares over the same entities and weights.
    """

    squared_error = 0.0
    squared_truth = 0.0
    for statistics in components:
        sums = getattr(statistics, weighting)
        squared_error += sums.squared_error
        squared_truth += sums.squared_truth
    if squared_truth <= 0.0:
        raise WindsorMLCandidateEvaluatorError(
            "vector relative L2 is undefined when the ground-truth norm is zero"
        )
    value = 100.0 * math.sqrt(squared_error / squared_truth)
    if not math.isfinite(value):
        raise WindsorMLCandidateEvaluatorError("vector relative L2 is non-finite")
    return value


def _scalar_evidence(result: _ScalarResult) -> dict[str, object]:
    statistics = result.statistics
    metrics = statistics.metric_values()
    return {
        "source_payload_sha256": result.source_payload_sha256,
        "source_payload_bytes": result.source_payload_bytes,
        "entity_count": statistics.entity_count,
        "component_count": statistics.component_count,
        "uniform": {**asdict(statistics.uniform), **metrics["uniform"]},
        "physical": {**asdict(statistics.physical), **metrics["physical"]},
    }


CP_FAMILIES = ("windsorml_cp_constant_v1", "windsorml_cp_relative_v1")
VELOCITY_FAMILIES = (
    "windsorml_velocity_constant_v1",
    "windsorml_velocity_relative_v1",
)
PROFILE_SUPPORT_SCHEMA = "windsorml-profile-support-v3"


def _load_profile_support(
    value: Mapping[str, object] | str | Path, *, case_id: str
) -> Mapping[str, object]:
    if isinstance(value, (str, Path)):
        document = json.loads(Path(value).read_text())
    else:
        document = dict(value)
    if document.get("schema") != PROFILE_SUPPORT_SCHEMA:
        raise WindsorMLCandidateEvaluatorError("profile support schema is unexpected")
    if document.get("case_id") != case_id:
        raise WindsorMLCandidateEvaluatorError(
            f"profile support binds {document.get('case_id')!r}, not {case_id!r}"
        )
    families = document.get("families")
    if not isinstance(families, Mapping):
        raise WindsorMLCandidateEvaluatorError("profile support has no families")
    missing = [f for f in (*CP_FAMILIES, *VELOCITY_FAMILIES) if f not in families]
    if missing:
        raise WindsorMLCandidateEvaluatorError(
            f"profile support is missing families {missing}"
        )
    return document


def _gather_sample_ids(
    support: Mapping[str, object],
    families: tuple[str, ...],
    id_key: str,
    *,
    entity_count: int,
) -> tuple[np.ndarray, dict[tuple[str, str], slice]]:
    """Flatten every station's native IDs into one capture vector."""

    ids: list[int] = []
    slices: dict[tuple[str, str], slice] = {}
    for family in families:
        for station, payload in support["families"][family].items():
            values = payload[id_key]
            start = len(ids)
            ids.extend(int(v) for v in values)
            slices[(family, station)] = slice(start, len(ids))
    array = np.asarray(ids, dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= entity_count):
        raise WindsorMLCandidateEvaluatorError(
            "profile support references a native ID outside the case"
        )
    return array, slices


def _profile_series(
    support: Mapping[str, object],
    families: tuple[str, ...],
    slices: dict[tuple[str, str], slice],
    sampled_truth: np.ndarray,
    sampled_prediction: np.ndarray,
    *,
    truth_key: str,
    scale: float,
    quantity_id: str,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for family in families:
        stations = []
        for station, payload in support["families"][family].items():
            window = slices[(family, station)]
            truth = sampled_truth[window] / scale
            prediction = sampled_prediction[window] / scale
            declared = np.asarray(payload[truth_key], dtype=np.float64)
            # The support's stored truth must agree with the truth streamed from
            # the pinned source, or the support is stale.
            if not np.allclose(truth, declared, rtol=0.0, atol=1e-5):
                raise WindsorMLCandidateEvaluatorError(
                    f"{family}/{station} support truth disagrees with the native field"
                )
            stations.append(
                {
                    "station_id": station,
                    "quantity_id": quantity_id,
                    "sample_count": int(truth.shape[0]),
                    "source": "evaluator_derived_from_complete_native_fields",
                    "coordinate": list(payload["coordinate"]),
                    "truth": truth.tolist(),
                    "prediction": prediction.tolist(),
                }
            )
        out[family] = stations
    return out


def _coefficients(force_xyz: np.ndarray) -> dict[str, float]:
    """Convert an integrated force vector into WindsorML coefficients.

    The vertical axis is +y; mapping lift to +z instead collapses it from about
    0.49 to about -0.01 on run_0.
    """

    scaled = np.asarray(force_xyz, dtype=np.float64) / FORCE_REFERENCE_AREA_M2
    return {
        "cd": float(scaled[DRAG_AXIS_INDEX]),
        "cl": float(scaled[LIFT_AXIS_INDEX]),
        "cs": float(scaled[SIDE_AXIS_INDEX]),
    }


def evaluate_candidate_case(
    *,
    case: WindsorMLSourceCase,
    dataset_root: str | Path,
    surface_manifest: PredictionChunkManifest | str | Path,
    volume_manifest: PredictionChunkManifest | str | Path | None = None,
    profile_support: Mapping[str, object] | str | Path | None = None,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
    encoded_chunk_bytes: int = DEFAULT_ENCODED_CHUNK_BYTES,
    maximum_prediction_chunk_rows: int = DEFAULT_MAX_PREDICTION_CHUNK_ROWS,
) -> CandidateCaseEvaluation:
    """Score one complete native WindsorML case against verified predictions."""

    _positive_integer(hash_chunk_bytes, "hash_chunk_bytes")
    _positive_integer(validation_block_rows, "validation_block_rows")
    _positive_integer(encoded_chunk_bytes, "encoded_chunk_bytes")
    _positive_integer(maximum_prediction_chunk_rows, "maximum_prediction_chunk_rows")

    surface = _manifest(surface_manifest)
    _validate_manifest(
        surface,
        case_id=case.case_id,
        support_id=WINDSORML_SURFACE_SUPPORT_ID,
        association="PointData",
        expected_count=case.surface_entity_count,
        maximum_rows=maximum_prediction_chunk_rows,
    )

    # The published per-point dual areas are the canonical surface weights; the
    # dataset README warns that recomputing polygon areas with VTK drifts.
    dual_area_path = case.surface_dual_area.resolve(dataset_root)
    with RetainedVerifiedFile.open(dual_area_path, label="surface_dual_area") as retained:
        if retained.snapshot.size_bytes != case.surface_dual_area.size_bytes:
            raise WindsorMLCandidateEvaluatorError("surface dual-area size differs")
        if retained.sha256() != case.surface_dual_area.sha256:
            raise WindsorMLCandidateEvaluatorError("surface dual-area SHA-256 differs")
        retained.handle.seek(0)
        dual_area = np.load(retained.handle, allow_pickle=False).astype(np.float64)
        retained.assert_unchanged(context="while surface weights were read")
    if dual_area.ndim != 1 or dual_area.shape[0] != case.surface_entity_count:
        raise WindsorMLCandidateEvaluatorError(
            "surface dual-area array does not match the native point count"
        )
    if not np.all(np.isfinite(dual_area)) or np.any(dual_area <= 0.0):
        raise WindsorMLCandidateEvaluatorError(
            "surface dual-area weights must be finite and positive"
        )

    support = (
        None
        if profile_support is None
        else _load_profile_support(profile_support, case_id=case.case_id)
    )
    cp_sample_ids: np.ndarray | None = None
    cp_slices: dict[tuple[str, str], slice] = {}
    if support is not None:
        cp_sample_ids, cp_slices = _gather_sample_ids(
            support,
            CP_FAMILIES,
            "native_point_ids",
            entity_count=case.surface_entity_count,
        )

    surface_results: dict[str, _ScalarResult] = {}
    with _open_verified_source(case.boundary, dataset_root, label="boundary") as retained:
        index = _index_source(
            retained.handle,
            expected_entities=case.surface_entity_count,
            association="PointData",
        )
        normals_array = _required_array(index, SURFACE_NORMALS_ARRAY, 3, "PointData")
        retained.handle.seek(0)
        normals = _read_inline_array(
            retained.handle, index, normals_array, encoded_chunk_bytes=encoded_chunk_bytes
        )
        if normals.shape != (case.surface_entity_count, 3):
            raise WindsorMLCandidateEvaluatorError("native Normals shape differs")
        norms = np.linalg.norm(normals, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise WindsorMLCandidateEvaluatorError("native Normals must be unit vectors")

        # Pressure acts along -n; each shear component acts along its own axis.
        area_vectors = normals * dual_area[:, None]
        axis_vectors = {
            field: np.zeros((case.surface_entity_count, 3), dtype=np.float64)
            for field in SURFACE_SHEAR_FIELDS
        }
        for axis, field in enumerate(SURFACE_SHEAR_FIELDS):
            axis_vectors[field][:, axis] = dual_area

        for field in (SURFACE_PRESSURE_FIELD, *SURFACE_SHEAR_FIELDS):
            array = _required_array(index, field, 1, "PointData")
            retained.handle.seek(0)
            force_vectors = (
                -area_vectors if field == SURFACE_PRESSURE_FIELD else axis_vectors[field]
            )
            surface_results[field] = _evaluate_scalar(
                stream=retained.handle,
                vtk_index=index,
                array=array,
                manifest=surface,
                field_name=field,
                weights=dual_area,
                force_vectors=force_vectors,
                hash_chunk_bytes=hash_chunk_bytes,
                validation_block_rows=validation_block_rows,
                encoded_chunk_bytes=encoded_chunk_bytes,
                sample_ids=cp_sample_ids if field == SURFACE_PRESSURE_FIELD else None,
            )

    truth_force = np.zeros(3, dtype=np.float64)
    prediction_force = np.zeros(3, dtype=np.float64)
    for result in surface_results.values():
        assert result.truth_force_xyz is not None
        assert result.prediction_force_xyz is not None
        truth_force += result.truth_force_xyz
        prediction_force += result.prediction_force_xyz
    truth_coefficients = _coefficients(truth_force)
    prediction_coefficients = _coefficients(prediction_force)

    # Audit the truth integration against the published CSV. Scoring uses the
    # integrated truth, so this never enters a metric; it refuses a case whose
    # mesh, weights, or axis mapping has drifted.
    try:
        replay_audit = classify_force_replay(
            case_id=case.case_id,
            published=case.force_truth,
            replay_cd=truth_coefficients["cd"],
            replay_cl=truth_coefficients["cl"],
        )
    except WindsorMLContractError as error:
        raise WindsorMLCandidateEvaluatorError(str(error)) from error

    evidence: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": EVIDENCE_STATUS,
        "official_submission_artifact": False,
        "case_id": case.case_id,
        "run_id": case.run_id,
        "surface": {
            "support_id": WINDSORML_SURFACE_SUPPORT_ID,
            "association": "PointData",
            "entity_count": case.surface_entity_count,
            "weighting": "published_point_dual_area",
            "fields": {
                field: _scalar_evidence(result)
                for field, result in surface_results.items()
            },
            "metrics": {
                # Surface primary weighting is physical (dual area) per the
                # dataset's relative-L2 policy; equal-entity is the secondary.
                "surface_pressure_rel_l2": surface_results[SURFACE_PRESSURE_FIELD]
                .statistics.physical.relative_l2_percent(),
                "surface_pressure_equal_entity_rel_l2": surface_results[
                    SURFACE_PRESSURE_FIELD
                ].statistics.uniform.relative_l2_percent(),
                "surface_wall_shear_rel_l2": _vector_relative_l2_percent(
                    [surface_results[f].statistics for f in SURFACE_SHEAR_FIELDS],
                    weighting="physical",
                ),
                "surface_wall_shear_equal_entity_rel_l2": _vector_relative_l2_percent(
                    [surface_results[f].statistics for f in SURFACE_SHEAR_FIELDS],
                    weighting="uniform",
                ),
            },
        },
        "forces": {
            "reference_area_m2": FORCE_REFERENCE_AREA_M2,
            "axes": {"drag": "+x", "lift": "+y", "side": "+z"},
            "truth_integrated": truth_coefficients,
            "prediction_integrated": prediction_coefficients,
            "published_audit": {
                "cd": case.force_truth.cd,
                "cl": case.force_truth.cl,
                "cs": case.force_truth.cs,
                "relative_delta": dict(replay_audit),
            },
        },
    }

    if volume_manifest is not None:
        volume = _manifest(volume_manifest)
        _validate_manifest(
            volume,
            case_id=case.case_id,
            support_id=WINDSORML_VOLUME_SUPPORT_ID,
            association="CellData",
            expected_count=case.volume_entity_count,
            maximum_rows=maximum_prediction_chunk_rows,
        )
        # v1 scores the volume with equal-cell weighting by decision, so the
        # physical weights are unit and no cell-volume sidecar is required.
        volume_weights = np.ones(case.volume_entity_count, dtype=np.float64)
        velocity_sample_ids: np.ndarray | None = None
        velocity_slices: dict[tuple[str, str], slice] = {}
        if support is not None:
            velocity_sample_ids, velocity_slices = _gather_sample_ids(
                support,
                VELOCITY_FAMILIES,
                "native_cell_ids",
                entity_count=case.volume_entity_count,
            )
        volume_results: dict[str, _ScalarResult] = {}
        with _open_verified_source(case.volume, dataset_root, label="volume") as retained:
            volume_index = _index_source(
                retained.handle,
                expected_entities=case.volume_entity_count,
                association="CellData",
            )
            for field in (*VOLUME_VELOCITY_FIELDS, VOLUME_PRESSURE_FIELD):
                array = _required_array(volume_index, field, 1, "CellData")
                retained.handle.seek(0)
                volume_results[field] = _evaluate_scalar(
                    stream=retained.handle,
                    vtk_index=volume_index,
                    array=array,
                    manifest=volume,
                    field_name=field,
                    weights=volume_weights,
                    force_vectors=None,
                    hash_chunk_bytes=hash_chunk_bytes,
                    validation_block_rows=validation_block_rows,
                    encoded_chunk_bytes=encoded_chunk_bytes,
                    sample_ids=(
                        velocity_sample_ids
                        if field == VOLUME_VELOCITY_FIELDS[0]
                        else None
                    ),
                )
        evidence["volume"] = {
            "support_id": WINDSORML_VOLUME_SUPPORT_ID,
            "association": "CellData",
            "entity_count": case.volume_entity_count,
            "weighting": "equal_native_entity",
            "fields": {
                field: _scalar_evidence(result)
                for field, result in volume_results.items()
            },
            "metrics": {
                "volume_velocity_rel_l2": _vector_relative_l2_percent(
                    [volume_results[f].statistics for f in VOLUME_VELOCITY_FIELDS],
                    weighting="uniform",
                ),
                "volume_pressure_rel_l2": volume_results[VOLUME_PRESSURE_FIELD]
                .statistics.uniform.relative_l2_percent(),
            },
        }

    if support is not None:
        profiles: dict[str, object] = {
            "support_schema": PROFILE_SUPPORT_SCHEMA,
            "sample_count": support["sample_count"],
            "body_height_m": support["body_height_m"],
            "participant_profile_payload_accepted": False,
            "families": {},
        }
        cp_result = surface_results[SURFACE_PRESSURE_FIELD]
        assert cp_result.sampled_truth is not None
        profiles["families"].update(
            _profile_series(
                support,
                CP_FAMILIES,
                cp_slices,
                cp_result.sampled_truth,
                cp_result.sampled_prediction,
                truth_key="truth_cp",
                scale=1.0,
                quantity_id="cp",
            )
        )
        if volume_manifest is not None:
            velocity_result = volume_results[VOLUME_VELOCITY_FIELDS[0]]
            assert velocity_result.sampled_truth is not None
            profiles["families"].update(
                _profile_series(
                    support,
                    VELOCITY_FAMILIES,
                    velocity_slices,
                    velocity_result.sampled_truth,
                    velocity_result.sampled_prediction,
                    truth_key="truth_ux_over_uinf",
                    scale=float(support["reference_velocity_m_s"]),
                    quantity_id="ux_over_uinf",
                )
            )
        evidence["profiles"] = profiles

    return CandidateCaseEvaluation(value=MappingProxyType(evidence))


__all__ = [
    "CandidateCaseEvaluation",
    "DEFAULT_ENCODED_CHUNK_BYTES",
    "DEFAULT_MAX_PREDICTION_CHUNK_ROWS",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_STATUS",
    "SURFACE_PRESSURE_FIELD",
    "SURFACE_SHEAR_FIELDS",
    "VOLUME_PRESSURE_FIELD",
    "VOLUME_VELOCITY_FIELDS",
    "WindsorMLCandidateEvaluatorError",
    "evaluate_candidate_case",
]
