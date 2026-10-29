"""Bounded-memory native DrivAerML CellData array audits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, BinaryIO, Mapping, Protocol

import numpy as np

from .accumulators import (
    DrivAerAccumulatorError,
    FinalizedFieldStatistics,
    StreamingFieldAccumulator,
)
from .source import (
    InlineBinaryDecodeError,
    InlineBinaryPayloadSummary,
    VTKDataArrayIndex,
    VTKXMLIndex,
    stream_inline_binary_payload,
)


class NativeFieldAuditError(ValueError):
    """Raised when required native field metadata or values are invalid."""


_NUMPY_DTYPES: Mapping[str, str] = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
}


@dataclass(frozen=True)
class NativeArrayAudit:
    """Compact audit evidence for one raw native VTK DataArray."""

    name: str
    association: str
    vtk_type: str
    number_of_components: int
    tuple_count: int
    scalar_count: int
    decoded_payload_bytes: int
    payload_sha256: str
    finite: bool
    minimum_by_component: tuple[float | int, ...]
    maximum_by_component: tuple[float | int, ...]
    units: str
    raw_id_start: int
    raw_id_stop: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NativeFieldMetricPass:
    """One complete native field reduction under a fixed chunk partition."""

    field_name: str
    chunk_entities: int
    source_payload: InlineBinaryPayloadSummary
    statistics: FinalizedFieldStatistics

    def to_json(self) -> dict[str, object]:
        values = self.statistics.metric_values()
        return {
            "field_name": self.field_name,
            "chunk_entities": self.chunk_entities,
            "source_payload_sha256": self.source_payload.payload_sha256,
            "source_payload_bytes": self.source_payload.decoded_payload_bytes,
            "entity_count": self.statistics.entity_count,
            "component_count": self.statistics.component_count,
            "metrics": values,
            "additive_sums": {
                weighting: asdict(getattr(self.statistics, weighting))
                for weighting in ("uniform", "physical")
            },
        }


class _NumericAuditSink:
    def __init__(
        self,
        *,
        vtk_type: str,
        components: int,
        byte_order: str,
    ) -> None:
        code = _NUMPY_DTYPES.get(vtk_type)
        if code is None:
            raise NativeFieldAuditError(f"unsupported VTK numeric type {vtk_type!r}")
        if components < 1:
            raise NativeFieldAuditError("component count must be positive")
        prefix = "<" if byte_order == "LittleEndian" else ">"
        if code.endswith("1"):
            prefix = "|"
        self.dtype = np.dtype(prefix + code)
        self.components = components
        self.tuple_bytes = self.dtype.itemsize * components
        self.pending = bytearray()
        self.tuple_count = 0
        self.finite = True
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def write(self, payload: bytes) -> int:
        self.pending.extend(payload)
        complete_bytes = len(self.pending) - len(self.pending) % self.tuple_bytes
        if complete_bytes:
            # Detach complete tuples before creating NumPy's exported view.  A
            # view into ``self.pending`` would keep the bytearray exported and
            # make the subsequent bounded-buffer deletion raise BufferError.
            complete = bytes(self.pending[:complete_bytes])
            del self.pending[:complete_bytes]
            values = np.frombuffer(complete, dtype=self.dtype).reshape(
                -1, self.components
            )
            if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
                self.finite = False
            block_min = np.min(values, axis=0)
            block_max = np.max(values, axis=0)
            if self.minimum is None:
                self.minimum = np.array(block_min, copy=True)
                self.maximum = np.array(block_max, copy=True)
            else:
                self.minimum = np.minimum(self.minimum, block_min)
                self.maximum = np.maximum(self.maximum, block_max)
            self.tuple_count += values.shape[0]
        return len(payload)

    def finish(self) -> tuple[int, bool, tuple[float | int, ...], tuple[float | int, ...]]:
        if self.pending:
            raise NativeFieldAuditError("decoded numeric payload ends within a tuple")
        if self.tuple_count < 1 or self.minimum is None or self.maximum is None:
            raise NativeFieldAuditError("decoded numeric payload contains no tuples")
        if not self.finite:
            raise NativeFieldAuditError("native field contains non-finite values")
        if self.dtype.kind == "f":
            minimum = tuple(float(value) for value in self.minimum)
            maximum = tuple(float(value) for value in self.maximum)
        else:
            minimum = tuple(int(value) for value in self.minimum)
            maximum = tuple(int(value) for value in self.maximum)
        return self.tuple_count, self.finite, minimum, maximum


class _NativeMetricSink:
    """Convert decoded VTK tuples into bounded raw-ID metric chunks."""

    def __init__(
        self,
        *,
        expected_entity_count: int,
        vtk_type: str,
        components: int,
        byte_order: str,
        chunk_entities: int,
        predictions: Any | None,
        physical_weights: Any | None,
    ) -> None:
        code = _NUMPY_DTYPES.get(vtk_type)
        if code is None:
            raise NativeFieldAuditError(f"unsupported VTK numeric type {vtk_type!r}")
        if expected_entity_count < 1 or chunk_entities < 1 or components < 1:
            raise NativeFieldAuditError(
                "expected_entity_count, components, and chunk_entities must be positive"
            )
        prefix = "<" if byte_order == "LittleEndian" else ">"
        if code.endswith("1"):
            prefix = "|"
        self.dtype = np.dtype(prefix + code)
        self.components = components
        self.tuple_bytes = self.dtype.itemsize * components
        self.expected_entity_count = expected_entity_count
        self.chunk_entities = chunk_entities
        self.predictions = None if predictions is None else np.asarray(predictions)
        expected_shape = (
            (expected_entity_count,)
            if components == 1
            else (expected_entity_count, components)
        )
        if self.predictions is not None:
            if self.predictions.shape != expected_shape:
                raise NativeFieldAuditError(
                    f"prediction shape must be {expected_shape}, got {self.predictions.shape}"
                )
            if self.predictions.dtype.kind not in {"i", "u", "f"}:
                raise NativeFieldAuditError("predictions must be numeric")
        self.physical_weights = (
            None if physical_weights is None else np.asarray(physical_weights)
        )
        if self.physical_weights is not None:
            if self.physical_weights.shape != (expected_entity_count,):
                raise NativeFieldAuditError(
                    "physical weights must contain one value per native cell"
                )
            if self.physical_weights.dtype.kind not in {"i", "u", "f"}:
                raise NativeFieldAuditError("physical weights must be numeric")
        self.pending = bytearray()
        self.raw_id_cursor = 0
        self.accumulator = StreamingFieldAccumulator(
            expected_entity_count, component_count=components
        )

    def _consume(self, tuple_count: int) -> None:
        if tuple_count < 1:
            return
        byte_count = tuple_count * self.tuple_bytes
        complete = bytes(self.pending[:byte_count])
        del self.pending[:byte_count]
        truth = np.frombuffer(complete, dtype=self.dtype).reshape(
            tuple_count, self.components
        )
        if self.components == 1:
            truth = truth[:, 0]
        start = self.raw_id_cursor
        stop = start + tuple_count
        if stop > self.expected_entity_count:
            raise NativeFieldAuditError(
                "decoded native field contains more tuples than its Piece declares"
            )
        prediction = (
            np.zeros_like(truth)
            if self.predictions is None
            else self.predictions[start:stop]
        )
        weights = (
            np.ones(tuple_count, dtype=np.float64)
            if self.physical_weights is None
            else self.physical_weights[start:stop]
        )
        self.accumulator.add_chunk(
            np.arange(start, stop, dtype=np.int64),
            truth,
            prediction,
            weights,
        )
        self.raw_id_cursor = stop

    def write(self, payload: bytes) -> int:
        self.pending.extend(payload)
        target_bytes = self.chunk_entities * self.tuple_bytes
        while len(self.pending) >= target_bytes:
            self._consume(self.chunk_entities)
        return len(payload)

    def finish(self) -> FinalizedFieldStatistics:
        if len(self.pending) % self.tuple_bytes:
            raise NativeFieldAuditError("decoded native field ends within a tuple")
        self._consume(len(self.pending) // self.tuple_bytes)
        if self.raw_id_cursor != self.expected_entity_count:
            raise NativeFieldAuditError(
                "decoded native field does not cover every raw cell exactly once"
            )
        return self.accumulator.finalize()


class _WritableSink(Protocol):
    """Minimal sink protocol accepted by the inline VTK decoder."""

    def write(self, payload: bytes) -> int: ...


class _BroadcastSink:
    """Feed one decoded payload stream to independent bounded-memory sinks."""

    def __init__(self, *sinks: _WritableSink) -> None:
        if not sinks:
            raise NativeFieldAuditError("a broadcast sink needs at least one consumer")
        self.sinks = sinks

    def write(self, payload: bytes) -> int:
        for sink in self.sinks:
            consumed = sink.write(payload)
            if consumed != len(payload):
                raise NativeFieldAuditError(
                    "a decoded-payload consumer did not accept the complete block"
                )
        return len(payload)


def audit_inline_numeric_array(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    *,
    units: str,
    encoded_chunk_size: int = 8 * 1024 * 1024,
) -> NativeArrayAudit:
    """Decode, hash, and finite-check one array while preserving tuple order."""

    if array.name is None:
        raise NativeFieldAuditError("required native field array must have a name")
    sink = _NumericAuditSink(
        vtk_type=array.vtk_type,
        components=array.number_of_components,
        byte_order=vtk_index.byte_order,
    )
    try:
        payload = stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            sink,
            encoded_chunk_size=encoded_chunk_size,
        )
    except (InlineBinaryDecodeError, DrivAerAccumulatorError) as error:
        raise NativeFieldAuditError(str(error)) from error
    tuple_count, finite, minimum, maximum = sink.finish()
    if tuple_count != payload.tuple_count:
        raise NativeFieldAuditError("numeric audit tuple count differs from VTK payload")
    return NativeArrayAudit(
        name=array.name,
        association=array.association,
        vtk_type=array.vtk_type,
        number_of_components=array.number_of_components,
        tuple_count=tuple_count,
        scalar_count=payload.scalar_count,
        decoded_payload_bytes=payload.decoded_payload_bytes,
        payload_sha256=payload.payload_sha256,
        finite=finite,
        minimum_by_component=minimum,
        maximum_by_component=maximum,
        units=units,
        raw_id_start=0,
        raw_id_stop=tuple_count,
    )


def audit_required_volume_cell_data(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    *,
    encoded_chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, NativeArrayAudit]:
    """Audit canonical volume ``pMeanTrim`` and ``UMeanTrim`` CellData."""

    if vtk_index.dataset_type != "UnstructuredGrid":
        raise NativeFieldAuditError("volume source must be a VTK UnstructuredGrid")
    if len(vtk_index.pieces) != 1:
        raise NativeFieldAuditError("native DrivAerML volume must contain one Piece")
    if vtk_index.compressor is not None:
        raise NativeFieldAuditError("compressed volume payloads are not in the frozen release")
    requirements = {
        "pMeanTrim": (1, "m^2/s^2"),
        "UMeanTrim": (3, "m/s"),
    }
    result: dict[str, NativeArrayAudit] = {}
    for name, (components, units) in requirements.items():
        arrays = vtk_index.arrays_for(association="CellData", name=name)
        if len(arrays) != 1:
            raise NativeFieldAuditError(
                f"volume must contain exactly one CellData {name!r} array"
            )
        array = arrays[0]
        if array.vtk_type != "Float32" or array.number_of_components != components:
            raise NativeFieldAuditError(
                f"CellData {name!r} must be Float32 with {components} component(s)"
            )
        result[name] = audit_inline_numeric_array(
            stream,
            vtk_index,
            array,
            units=units,
            encoded_chunk_size=encoded_chunk_size,
        )
    expected = vtk_index.pieces[0].number_of_cells
    if any(audit.tuple_count != expected for audit in result.values()):
        raise NativeFieldAuditError("volume field tuple count differs from native cells")
    return result


def evaluate_inline_native_cell_data(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    *,
    predictions: Any | None,
    physical_weights: Any | None,
    chunk_entities: int,
    encoded_chunk_size: int = 8 * 1024 * 1024,
) -> NativeFieldMetricPass:
    """Reduce one native CellData field without materializing its truth array.

    Passing ``predictions=None`` selects an explicit all-zero pilot prediction;
    passing ``physical_weights=None`` selects unit weights.  Real evaluation
    supplies same-order predictions.  Surface workflows may also supply the
    fixed same-order polygon areas; native volume evaluation deliberately uses
    unit weights and does not supply a geometric cell-volume array.
    """

    if array.association != "CellData" or array.piece_index < 0:
        raise NativeFieldAuditError("native metrics require a Piece CellData array")
    if array.name is None:
        raise NativeFieldAuditError("native metric field must have a name")
    try:
        piece = vtk_index.pieces[array.piece_index]
    except IndexError as error:
        raise NativeFieldAuditError("native field references a missing Piece") from error
    sink = _NativeMetricSink(
        expected_entity_count=piece.number_of_cells,
        vtk_type=array.vtk_type,
        components=array.number_of_components,
        byte_order=vtk_index.byte_order,
        chunk_entities=chunk_entities,
        predictions=predictions,
        physical_weights=physical_weights,
    )
    try:
        payload = stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            sink,
            encoded_chunk_size=encoded_chunk_size,
        )
        statistics = sink.finish()
    except (InlineBinaryDecodeError, DrivAerAccumulatorError) as error:
        raise NativeFieldAuditError(str(error)) from error
    if payload.tuple_count != statistics.entity_count:
        raise NativeFieldAuditError(
            "metric coverage differs from the decoded VTK tuple count"
        )
    return NativeFieldMetricPass(
        field_name=array.name,
        chunk_entities=chunk_entities,
        source_payload=payload,
        statistics=statistics,
    )


def audit_and_compare_inline_native_cell_data(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    *,
    units: str,
    predictions: Any | None,
    physical_weights: Any | None,
    reference_chunk_entities: int,
    comparison_chunk_entities: int,
    encoded_chunk_size: int = 8 * 1024 * 1024,
) -> tuple[NativeArrayAudit, NativeFieldMetricPass, NativeFieldMetricPass]:
    """Audit one field and evaluate two partitions from one decoded byte pass.

    The two metric sinks remain independent: each constructs and validates its
    own complete raw-ID partition.  Broadcasting only avoids decoding and
    reading the identical source payload three times; it does not share any
    additive accumulator state between the partitions.
    """

    if array.association != "CellData" or array.piece_index < 0:
        raise NativeFieldAuditError("native metrics require a Piece CellData array")
    if array.name is None:
        raise NativeFieldAuditError("native metric field must have a name")
    if (
        not isinstance(reference_chunk_entities, int)
        or isinstance(reference_chunk_entities, bool)
        or reference_chunk_entities < 1
        or not isinstance(comparison_chunk_entities, int)
        or isinstance(comparison_chunk_entities, bool)
        or comparison_chunk_entities < 1
        or reference_chunk_entities == comparison_chunk_entities
    ):
        raise NativeFieldAuditError(
            "reference and comparison chunk sizes must be distinct positive integers"
        )
    try:
        piece = vtk_index.pieces[array.piece_index]
    except IndexError as error:
        raise NativeFieldAuditError("native field references a missing Piece") from error

    numeric = _NumericAuditSink(
        vtk_type=array.vtk_type,
        components=array.number_of_components,
        byte_order=vtk_index.byte_order,
    )
    reference = _NativeMetricSink(
        expected_entity_count=piece.number_of_cells,
        vtk_type=array.vtk_type,
        components=array.number_of_components,
        byte_order=vtk_index.byte_order,
        chunk_entities=reference_chunk_entities,
        predictions=predictions,
        physical_weights=physical_weights,
    )
    comparison = _NativeMetricSink(
        expected_entity_count=piece.number_of_cells,
        vtk_type=array.vtk_type,
        components=array.number_of_components,
        byte_order=vtk_index.byte_order,
        chunk_entities=comparison_chunk_entities,
        predictions=predictions,
        physical_weights=physical_weights,
    )
    try:
        payload = stream_inline_binary_payload(
            stream,
            vtk_index,
            array,
            _BroadcastSink(numeric, reference, comparison),
            encoded_chunk_size=encoded_chunk_size,
        )
        tuple_count, finite, minimum, maximum = numeric.finish()
        reference_statistics = reference.finish()
        comparison_statistics = comparison.finish()
    except (InlineBinaryDecodeError, DrivAerAccumulatorError) as error:
        raise NativeFieldAuditError(str(error)) from error

    if (
        tuple_count != payload.tuple_count
        or reference_statistics.entity_count != payload.tuple_count
        or comparison_statistics.entity_count != payload.tuple_count
    ):
        raise NativeFieldAuditError(
            "audit or metric coverage differs from the decoded VTK tuple count"
        )
    audit = NativeArrayAudit(
        name=array.name,
        association=array.association,
        vtk_type=array.vtk_type,
        number_of_components=array.number_of_components,
        tuple_count=tuple_count,
        scalar_count=payload.scalar_count,
        decoded_payload_bytes=payload.decoded_payload_bytes,
        payload_sha256=payload.payload_sha256,
        finite=finite,
        minimum_by_component=minimum,
        maximum_by_component=maximum,
        units=units,
        raw_id_start=0,
        raw_id_stop=tuple_count,
    )
    return (
        audit,
        NativeFieldMetricPass(
            field_name=array.name,
            chunk_entities=reference_chunk_entities,
            source_payload=payload,
            statistics=reference_statistics,
        ),
        NativeFieldMetricPass(
            field_name=array.name,
            chunk_entities=comparison_chunk_entities,
            source_payload=payload,
            statistics=comparison_statistics,
        ),
    )
