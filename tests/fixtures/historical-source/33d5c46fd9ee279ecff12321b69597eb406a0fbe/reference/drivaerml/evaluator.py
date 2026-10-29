"""Candidate one-case evaluator for native DrivAerML prediction chunks.

This module joins the local candidate NPZ transport to the pinned native
surface and volume sources.  It is deliberately not an official FluidsBench
submission evaluator.  Prediction files are opened one at a time, reduced to
additive statistics, and released before the next NPZ is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .accumulators import (
    DrivAerAccumulatorError,
    FinalizedFieldStatistics,
    StreamingFieldAccumulator,
)
from .native_fields import NativeFieldAuditError, audit_required_volume_cell_data
from .native_surface import FixedSurfaceAreas, NativeSurface
from .prediction_chunks import (
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    PredictionChunk,
    PredictionChunkError,
    PredictionChunkManifest,
    iter_prediction_chunks,
    load_prediction_chunk_manifest,
)
from .source import (
    InlineBinaryDecodeError,
    NativeCaseRecord,
    NativeSourcePin,
    SegmentedReader,
    VTKDataArrayIndex,
    VTKXMLIndex,
    stream_inline_binary_payload,
)
from .surface_forces import (
    DrivAerSurfaceForceError,
    audit_fixed_surface_areas,
    finalize_force_coefficients,
    force_moment_chunk,
    surface_geometry_chunk_validated,
)


CANDIDATE_EVIDENCE_SCHEMA = "drivaerml-candidate-case-evaluation-v2"
CANDIDATE_STATUS = "candidate_evaluator_evidence_not_official_submission"
DEFAULT_MAX_PREDICTION_CHUNK_ROWS = 1_000_000
DEFAULT_ENCODED_CHUNK_BYTES = 8 * 1024 * 1024
SURFACE_AREA_RELATIVE_TOLERANCE = 6.0e-8
OFFICIAL_NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
OFFICIAL_REPOSITORY_ID = "neashton/drivaerml"
OFFICIAL_REPOSITORY_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"


class DrivAerCandidateEvaluatorError(ValueError):
    """Raised when one candidate case cannot be evaluated exactly."""


@dataclass(frozen=True)
class NativeSourceContract:
    """Expected immutable source-pin identity for an evaluator invocation."""

    pin_sha256: str
    repository_id: str
    repository_revision: str


OFFICIAL_NATIVE_SOURCE_CONTRACT = NativeSourceContract(
    pin_sha256=OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
    repository_id=OFFICIAL_REPOSITORY_ID,
    repository_revision=OFFICIAL_REPOSITORY_REVISION,
)


@dataclass(frozen=True)
class _VolumeFieldResult:
    field_name: str
    source_payload_sha256: str
    source_payload_bytes: int
    statistics: FinalizedFieldStatistics


@dataclass(frozen=True)
class CandidateCaseEvaluation:
    """Compact evidence inputs plus complete case field/force reductions."""

    case_id: str
    source_pin_sha256: str
    repository_id: str
    repository_revision: str
    boundary_sha256: str
    surface_native_audit: dict[str, object]
    surface_area_audit: dict[str, object]
    volume_part_sha256: tuple[str, ...]
    volume_vtk_audit: dict[str, object]
    volume_weighting_audit: dict[str, object]
    surface_prediction_manifest_sha256: str
    volume_prediction_manifest_sha256: str
    surface_prediction_chunk_sha256: tuple[str, ...]
    volume_prediction_chunk_sha256: tuple[str, ...]
    surface_entity_count: int
    volume_entity_count: int
    surface_chunk_count: int
    volume_chunk_count: int
    volume_native_array_audits: dict[str, dict[str, object]]
    metric_values: dict[str, float]
    metric_sufficient_statistics: dict[str, dict[str, float | int | str]]
    additive_sums: dict[str, dict[str, dict[str, float | int]]]
    force_coefficients: dict[str, float | int | list[float]]
    execution: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "schema": CANDIDATE_EVIDENCE_SCHEMA,
            "schema_version": 2,
            "status": CANDIDATE_STATUS,
            "official_submission": False,
            "case_id": self.case_id,
            "source": {
                "native_source_pin_sha256": self.source_pin_sha256,
                "repository_id": self.repository_id,
                "repository_revision": self.repository_revision,
                "boundary_sha256": self.boundary_sha256,
                "surface_native": self.surface_native_audit,
                "surface_area": self.surface_area_audit,
                "volume_part_sha256": list(self.volume_part_sha256),
                "volume_vtk": self.volume_vtk_audit,
                "volume_weighting": self.volume_weighting_audit,
                "volume_native_arrays": self.volume_native_array_audits,
            },
            "prediction_inputs": {
                "surface_native_cells": {
                    "manifest_sha256": self.surface_prediction_manifest_sha256,
                    "chunk_sha256": list(self.surface_prediction_chunk_sha256),
                    "chunk_count": self.surface_chunk_count,
                    "entity_count": self.surface_entity_count,
                },
                "volume_native_cells": {
                    "manifest_sha256": self.volume_prediction_manifest_sha256,
                    "chunk_sha256": list(self.volume_prediction_chunk_sha256),
                    "chunk_count": self.volume_chunk_count,
                    "entity_count": self.volume_entity_count,
                },
            },
            "coverage": {
                "surface": {
                    "raw_cell_id_start": 0,
                    "raw_cell_id_stop": self.surface_entity_count,
                    "complete_gap_free_duplicate_free": True,
                },
                "volume": {
                    "raw_cell_id_start": 0,
                    "raw_cell_id_stop": self.volume_entity_count,
                    "complete_gap_free_duplicate_free": True,
                },
            },
            "metric_values": self.metric_values,
            "metric_sufficient_statistics": self.metric_sufficient_statistics,
            "additive_sums": self.additive_sums,
            "force_coefficients": self.force_coefficients,
            "execution": self.execution,
        }


def _sha256_file(path: Path | str, *, chunk_bytes: int = 1024 * 1024) -> str:
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
        raise DrivAerCandidateEvaluatorError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DrivAerCandidateEvaluatorError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def validate_native_source_contract(
    pin: NativeSourcePin,
    contract: NativeSourceContract = OFFICIAL_NATIVE_SOURCE_CONTRACT,
) -> str:
    """Require a loaded pin to match a predeclared immutable identity."""

    if not isinstance(pin, NativeSourcePin):
        raise DrivAerCandidateEvaluatorError(
            "native source must be a validated NativeSourcePin"
        )
    if not isinstance(contract, NativeSourceContract):
        raise DrivAerCandidateEvaluatorError(
            "source_contract must be a NativeSourceContract"
        )
    expected_sha = _sha256_digest(contract.pin_sha256, "source contract pin_sha256")
    actual_sha = _sha256_file(pin.source_path)
    if actual_sha != expected_sha:
        raise DrivAerCandidateEvaluatorError(
            "native-source pin SHA-256 differs from the evaluator source contract"
        )
    if (
        pin.repository_id != contract.repository_id
        or pin.repository_revision != contract.repository_revision
    ):
        raise DrivAerCandidateEvaluatorError(
            "native-source repository identity differs from the evaluator source contract"
        )
    return actual_sha


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DrivAerCandidateEvaluatorError(f"{label} must be a positive integer")
    return value


def _manifest(
    value: PredictionChunkManifest | Path | str,
) -> PredictionChunkManifest:
    if not isinstance(value, PredictionChunkManifest):
        return load_prediction_chunk_manifest(value)
    replay = load_prediction_chunk_manifest(value.path)
    if replay != value:
        raise DrivAerCandidateEvaluatorError(
            "preloaded prediction manifest differs from a retained-file replay"
        )
    return replay


def _validate_manifest_binding(
    manifest: PredictionChunkManifest,
    *,
    case_id: str,
    support_id: str,
    expected_count: int,
    maximum_chunk_rows: int,
) -> None:
    if manifest.case_id != case_id:
        raise DrivAerCandidateEvaluatorError(
            f"{support_id} prediction case_id differs from {case_id!r}"
        )
    if manifest.support_id != support_id:
        raise DrivAerCandidateEvaluatorError(
            f"expected prediction support_id {support_id!r}, found {manifest.support_id!r}"
        )
    if manifest.total_row_count != expected_count:
        raise DrivAerCandidateEvaluatorError(
            f"{support_id} prediction count {manifest.total_row_count} differs "
            f"from native count {expected_count}"
        )
    for descriptor in manifest.chunks:
        if descriptor.row_count > maximum_chunk_rows:
            raise DrivAerCandidateEvaluatorError(
                f"{support_id} prediction chunk {descriptor.chunk_index} exceeds "
                f"maximum_prediction_chunk_rows={maximum_chunk_rows}"
            )


def _next_chunk(
    iterator: Iterator[PredictionChunk],
) -> PredictionChunk | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _field_metric_values(
    prefix: str,
    statistics: FinalizedFieldStatistics,
    *,
    physical_first: bool,
    include_secondary: bool = True,
) -> dict[str, float]:
    primary = statistics.physical if physical_first else statistics.uniform
    secondary = statistics.uniform if physical_first else statistics.physical
    primary_label = "area" if physical_first else "equal_entity"
    secondary_label = "equal_entity" if physical_first else "physical"
    result = {
        f"{prefix}_rel_l2": primary.relative_l2_percent(),
        f"drivaerml_{prefix}_{primary_label}_mae": primary.mae(),
        f"drivaerml_{prefix}_{primary_label}_rmse": primary.rmse(),
    }
    if include_secondary:
        result.update(
            {
                (
                    f"{prefix}_"
                    f"{'equal_entity' if physical_first else 'physical'}_rel_l2"
                ): secondary.relative_l2_percent(),
                f"drivaerml_{prefix}_{secondary_label}_mae": secondary.mae(),
                f"drivaerml_{prefix}_{secondary_label}_rmse": secondary.rmse(),
            }
        )
    return result


def _relative_l2_evidence(
    metric_id: str,
    statistics: FinalizedFieldStatistics,
    *,
    physical_first: bool,
    physical_dataset_weighting: str,
    uniform_dataset_weighting: str,
    include_secondary: bool = True,
) -> dict[str, dict[str, float | int | str]]:
    primary = statistics.physical if physical_first else statistics.uniform
    secondary = statistics.uniform if physical_first else statistics.physical
    result = {
        metric_id: primary.relative_l2_evidence(
            weighting="support_weights" if physical_first else "uniform",
            dataset_weighting=(
                physical_dataset_weighting
                if physical_first
                else uniform_dataset_weighting
            ),
        )
    }
    if include_secondary:
        result[
            (
                f"{metric_id.rsplit('_rel_l2', 1)[0]}_"
                f"{'equal_entity' if physical_first else 'physical'}_rel_l2"
            )
        ] = secondary.relative_l2_evidence(
            weighting="uniform" if physical_first else "support_weights",
            dataset_weighting=(
                uniform_dataset_weighting
                if physical_first
                else physical_dataset_weighting
            ),
        )
    return result


def _additive_sums(
    statistics: FinalizedFieldStatistics,
    *,
    include_physical: bool = True,
) -> dict[str, dict[str, float | int]]:
    return {
        weighting: asdict(getattr(statistics, weighting))
        for weighting in (("uniform", "physical") if include_physical else ("uniform",))
    }


def _evaluate_surface_chunks(
    manifest: PredictionChunkManifest,
    surface: NativeSurface,
    areas: FixedSurfaceAreas,
    *,
    hash_chunk_bytes: int,
    validation_block_rows: int,
) -> tuple[
    FinalizedFieldStatistics,
    FinalizedFieldStatistics,
    dict[str, float | int | list[float]],
    dict[str, float | int],
]:
    count = surface.polygon_count
    pressure = StreamingFieldAccumulator(count, component_count=1)
    shear = StreamingFieldAccumulator(count, component_count=3)
    force_chunks = []
    area_audits: list[dict[str, float | int]] = []
    iterator = iter_prediction_chunks(
        manifest,
        hash_chunk_bytes=hash_chunk_bytes,
        validation_block_rows=validation_block_rows,
    )
    while True:
        chunk = _next_chunk(iterator)
        if chunk is None:
            break
        start = chunk.descriptor.raw_cell_id_start
        stop = chunk.descriptor.raw_cell_id_stop
        try:
            weights = areas.values_m2[start:stop]
            geometry = surface_geometry_chunk_validated(
                surface.points_m,
                surface.connectivity,
                surface.offsets,
                start,
                stop,
            )
            area_audits.append(
                audit_fixed_surface_areas(
                    geometry.areas_m2,
                    weights,
                    rtol=SURFACE_AREA_RELATIVE_TOLERANCE,
                )
            )
            pressure.add_chunk(
                chunk.raw_cell_id,
                surface.pressure_m2_per_s2[start:stop],
                chunk.field("pMeanTrim"),
                weights,
            )
            shear.add_chunk(
                chunk.raw_cell_id,
                surface.wall_shear_m2_per_s2[start:stop],
                chunk.field("wallShearStressMeanTrim"),
                weights,
            )
            force_chunks.append(
                force_moment_chunk(
                    geometry,
                    chunk.field("pMeanTrim"),
                    chunk.field("wallShearStressMeanTrim"),
                )
            )
        finally:
            # Do not request the next NPZ until all references to this one go away.
            del chunk

    if sum(int(row["entity_count"]) for row in area_audits) != count:
        raise DrivAerCandidateEvaluatorError(
            "surface-area geometry audit did not cover every native polygon"
        )
    area_audit = {
        "entity_count": count,
        "relative_tolerance": SURFACE_AREA_RELATIVE_TOLERANCE,
        "calculated_sum_m2": math.fsum(
            float(row["calculated_sum_m2"]) for row in area_audits
        ),
        "published_sum_m2": math.fsum(
            float(row["published_sum_m2"]) for row in area_audits
        ),
        "maximum_absolute_difference_m2": max(
            (float(row["maximum_absolute_difference_m2"]) for row in area_audits),
            default=0.0,
        ),
        "maximum_relative_difference": max(
            (float(row["maximum_relative_difference"]) for row in area_audits),
            default=0.0,
        ),
        "raw_order_correspondence_verified": True,
        "published_values_role": "fixed_input_audited_not_regenerated",
    }
    return (
        pressure.finalize(),
        shear.finalize(),
        finalize_force_coefficients(force_chunks, expected_entity_count=count),
        area_audit,
    )


class _VolumePredictionSink:
    """Align decoded native truth to one verified prediction NPZ at a time."""

    def __init__(
        self,
        *,
        vtk_index: VTKXMLIndex,
        array: VTKDataArrayIndex,
        manifest: PredictionChunkManifest,
        field_name: str,
        hash_chunk_bytes: int,
        validation_block_rows: int,
    ) -> None:
        if array.vtk_type != "Float32":
            raise DrivAerCandidateEvaluatorError(
                f"native volume field {field_name!r} must use Float32"
            )
        prefix = "<" if vtk_index.byte_order == "LittleEndian" else ">"
        self.dtype = np.dtype(prefix + "f4")
        self.components = array.number_of_components
        self.tuple_bytes = self.dtype.itemsize * self.components
        self.field_name = field_name
        self.pending = bytearray()
        self.cursor = 0
        self.accumulator = StreamingFieldAccumulator(
            manifest.total_row_count,
            component_count=self.components,
        )
        self.iterator = iter_prediction_chunks(
            manifest,
            hash_chunk_bytes=hash_chunk_bytes,
            validation_block_rows=validation_block_rows,
        )
        self.current = _next_chunk(self.iterator)

    def _consume_current(self) -> None:
        chunk = self.current
        if chunk is None:
            raise DrivAerCandidateEvaluatorError(
                f"native truth field {self.field_name!r} exceeds prediction coverage"
            )
        descriptor = chunk.descriptor
        if descriptor.raw_cell_id_start != self.cursor:
            raise DrivAerCandidateEvaluatorError(
                f"prediction cursor mismatch for field {self.field_name!r}"
            )
        required_bytes = descriptor.row_count * self.tuple_bytes
        complete = bytes(self.pending[:required_bytes])
        del self.pending[:required_bytes]
        truth = np.frombuffer(complete, dtype=self.dtype).reshape(
            descriptor.row_count,
            self.components,
        )
        if self.components == 1:
            truth = truth[:, 0]
        prediction = chunk.field(self.field_name)
        # DrivAerML volume fields use one equal weight per native cell.  The
        # generic accumulator still accepts a weight vector because surface
        # fields use fixed polygon areas; no geometric cell-volume array is
        # constructed or consumed here.
        weights = np.ones(descriptor.row_count, dtype=np.float64)
        self.accumulator.add_chunk(
            chunk.raw_cell_id,
            truth,
            prediction,
            weights,
        )
        self.cursor = descriptor.raw_cell_id_stop
        self.current = None
        del weights, prediction, truth, complete, chunk
        # The old arrays are now unreferenced before the generator opens another NPZ.
        self.current = _next_chunk(self.iterator)

    def write(self, payload: bytes) -> int:
        self.pending.extend(payload)
        while self.current is not None:
            required_bytes = self.current.descriptor.row_count * self.tuple_bytes
            if len(self.pending) < required_bytes:
                break
            self._consume_current()
        if self.current is None and self.pending:
            raise DrivAerCandidateEvaluatorError(
                f"native truth field {self.field_name!r} exceeds prediction coverage"
            )
        return len(payload)

    def finish(self) -> FinalizedFieldStatistics:
        if self.pending:
            raise DrivAerCandidateEvaluatorError(
                f"native truth field {self.field_name!r} ends within a prediction chunk"
            )
        if self.current is not None or self.cursor != self.accumulator.expected_entity_count:
            raise DrivAerCandidateEvaluatorError(
                f"prediction coverage exceeds native truth field {self.field_name!r}"
            )
        return self.accumulator.finalize()


def _evaluate_volume_field(
    stream: BinaryIO,
    vtk_index: VTKXMLIndex,
    array: VTKDataArrayIndex,
    manifest: PredictionChunkManifest,
    *,
    hash_chunk_bytes: int,
    validation_block_rows: int,
    encoded_chunk_bytes: int,
) -> _VolumeFieldResult:
    if array.name is None:
        raise DrivAerCandidateEvaluatorError("native volume field must have a name")
    sink = _VolumePredictionSink(
        vtk_index=vtk_index,
        array=array,
        manifest=manifest,
        field_name=array.name,
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
        statistics = sink.finish()
    except (InlineBinaryDecodeError, DrivAerAccumulatorError) as error:
        raise DrivAerCandidateEvaluatorError(str(error)) from error
    if payload.tuple_count != statistics.entity_count:
        raise DrivAerCandidateEvaluatorError(
            f"native volume field {array.name!r} tuple coverage mismatch"
        )
    return _VolumeFieldResult(
        field_name=array.name,
        source_payload_sha256=payload.payload_sha256,
        source_payload_bytes=payload.decoded_payload_bytes,
        statistics=statistics,
    )


def _validate_case_sources(
    pin: NativeSourcePin,
    case: NativeCaseRecord,
    surface: NativeSurface,
    areas: FixedSurfaceAreas,
    volume_stream: SegmentedReader,
    vtk_index: VTKXMLIndex,
) -> int:
    areas.assert_source_unchanged(context="before candidate evaluation")
    if surface.boundary_sha256 != case.boundary.sha256:
        raise DrivAerCandidateEvaluatorError(
            "loaded boundary SHA-256 differs from the native-source pin"
        )
    if surface.source_path.name != case.boundary.path.name:
        raise DrivAerCandidateEvaluatorError(
            "loaded boundary filename differs from the native-source pin"
        )
    if (
        areas.sha256 != case.surface_cell_area.sha256
        or areas.source_boundary_sha256 != case.boundary.sha256
        or areas.entity_count != case.surface_cell_area.element_count
        or areas.entity_count != surface.polygon_count
    ):
        raise DrivAerCandidateEvaluatorError(
            "fixed surface areas differ from the native-source case binding"
        )
    if not isinstance(volume_stream, SegmentedReader):
        raise DrivAerCandidateEvaluatorError(
            "volume_stream must come from open_verified_multipart or open_verified_monolithic"
        )
    if volume_stream.logical_size_bytes != case.volume_total_size_bytes:
        raise DrivAerCandidateEvaluatorError(
            "verified volume stream size differs from the native-source pin"
        )
    if len(volume_stream.verification) != len(case.volume_parts):
        raise DrivAerCandidateEvaluatorError(
            "volume stream does not contain verification for every pinned part"
        )
    for verification, part in zip(
        volume_stream.verification, case.volume_parts, strict=True
    ):
        if verification.sha256 != part.sha256 or verification.size_bytes != part.size_bytes:
            raise DrivAerCandidateEvaluatorError(
                "verified volume segment differs from the pinned part order"
            )
    if vtk_index.source_size_bytes != case.volume_total_size_bytes:
        raise DrivAerCandidateEvaluatorError(
            "VTK XML index source size differs from the pinned logical volume"
        )
    if vtk_index.dataset_type != "UnstructuredGrid" or len(vtk_index.pieces) != 1:
        raise DrivAerCandidateEvaluatorError(
            "native volume must be one VTK UnstructuredGrid Piece"
        )
    volume_count = vtk_index.pieces[0].number_of_cells
    if volume_count < 1:
        raise DrivAerCandidateEvaluatorError("native volume contains no cells")
    if pin.case(case.case_id) is not case:
        # NativeSourcePin.case returns the exact immutable record held by the pin.
        raise DrivAerCandidateEvaluatorError("case record is not owned by the source pin")
    return volume_count


def evaluate_candidate_case(
    *,
    case_id: str,
    native_source_pin: NativeSourcePin,
    native_surface: NativeSurface,
    fixed_surface_areas: FixedSurfaceAreas,
    volume_stream: SegmentedReader,
    volume_vtk_index: VTKXMLIndex,
    surface_prediction_manifest: PredictionChunkManifest | Path | str,
    volume_prediction_manifest: PredictionChunkManifest | Path | str,
    maximum_prediction_chunk_rows: int = DEFAULT_MAX_PREDICTION_CHUNK_ROWS,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    validation_block_rows: int = DEFAULT_VALIDATION_BLOCK_ROWS,
    encoded_chunk_bytes: int = DEFAULT_ENCODED_CHUNK_BYTES,
    source_contract: NativeSourceContract = OFFICIAL_NATIVE_SOURCE_CONTRACT,
) -> CandidateCaseEvaluation:
    """Evaluate one exact candidate case without materializing predictions."""

    if not isinstance(native_source_pin, NativeSourcePin):
        raise DrivAerCandidateEvaluatorError(
            "native_source_pin must be a validated NativeSourcePin"
        )
    source_pin_sha256 = validate_native_source_contract(
        native_source_pin, source_contract
    )
    case = native_source_pin.case(case_id)
    maximum_rows = _positive_integer(
        maximum_prediction_chunk_rows, "maximum_prediction_chunk_rows"
    )
    hash_rows = _positive_integer(hash_chunk_bytes, "hash_chunk_bytes")
    validation_rows = _positive_integer(
        validation_block_rows, "validation_block_rows"
    )
    encoded_bytes = _positive_integer(encoded_chunk_bytes, "encoded_chunk_bytes")
    volume_count = _validate_case_sources(
        native_source_pin,
        case,
        native_surface,
        fixed_surface_areas,
        volume_stream,
        volume_vtk_index,
    )
    surface_manifest = _manifest(surface_prediction_manifest)
    volume_manifest = _manifest(volume_prediction_manifest)
    surface_manifest_sha256 = surface_manifest.sha256
    volume_manifest_sha256 = volume_manifest.sha256
    _validate_manifest_binding(
        surface_manifest,
        case_id=case_id,
        support_id="surface_native_cells",
        expected_count=native_surface.polygon_count,
        maximum_chunk_rows=maximum_rows,
    )
    _validate_manifest_binding(
        volume_manifest,
        case_id=case_id,
        support_id="volume_native_cells",
        expected_count=volume_count,
        maximum_chunk_rows=maximum_rows,
    )

    try:
        volume_audits = audit_required_volume_cell_data(
            volume_stream,
            volume_vtk_index,
            encoded_chunk_size=encoded_bytes,
        )
        (
            surface_pressure,
            surface_shear,
            coefficients,
            surface_geometry_area_audit,
        ) = _evaluate_surface_chunks(
            surface_manifest,
            native_surface,
            fixed_surface_areas,
            hash_chunk_bytes=hash_rows,
            validation_block_rows=validation_rows,
        )
        pressure_array = volume_vtk_index.arrays_for(
            association="CellData", name="pMeanTrim"
        )[0]
        velocity_array = volume_vtk_index.arrays_for(
            association="CellData", name="UMeanTrim"
        )[0]
        volume_pressure = _evaluate_volume_field(
            volume_stream,
            volume_vtk_index,
            pressure_array,
            volume_manifest,
            hash_chunk_bytes=hash_rows,
            validation_block_rows=validation_rows,
            encoded_chunk_bytes=encoded_bytes,
        )
        volume_velocity = _evaluate_volume_field(
            volume_stream,
            volume_vtk_index,
            velocity_array,
            volume_manifest,
            hash_chunk_bytes=hash_rows,
            validation_block_rows=validation_rows,
            encoded_chunk_bytes=encoded_bytes,
        )
    except (
        DrivAerAccumulatorError,
        DrivAerSurfaceForceError,
        NativeFieldAuditError,
        PredictionChunkError,
    ) as error:
        raise DrivAerCandidateEvaluatorError(str(error)) from error
    for name, result in (
        ("pMeanTrim", volume_pressure),
        ("UMeanTrim", volume_velocity),
    ):
        if result.source_payload_sha256 != volume_audits[name].payload_sha256:
            raise DrivAerCandidateEvaluatorError(
                f"volume field {name!r} changed between audit and metric passes"
            )

    statistics = {
        "surface_pressure": surface_pressure,
        "surface_wall_shear": surface_shear,
        "volume_pressure": volume_pressure.statistics,
        "volume_velocity": volume_velocity.statistics,
    }
    try:
        metric_values: dict[str, float] = {}
        metric_values.update(
            _field_metric_values(
                "surface_pressure", surface_pressure, physical_first=True
            )
        )
        metric_values.update(
            _field_metric_values(
                "surface_wall_shear", surface_shear, physical_first=True
            )
        )
        metric_values.update(
            _field_metric_values(
                "volume_pressure",
                volume_pressure.statistics,
                physical_first=False,
                include_secondary=False,
            )
        )
        metric_values.update(
            _field_metric_values(
                "volume_velocity",
                volume_velocity.statistics,
                physical_first=False,
                include_secondary=False,
            )
        )
        sufficient_statistics: dict[
            str, dict[str, float | int | str]
        ] = {}
        sufficient_statistics.update(
            _relative_l2_evidence(
                "surface_pressure_rel_l2",
                surface_pressure,
                physical_first=True,
                physical_dataset_weighting="surface_face_area",
                uniform_dataset_weighting="surface_entities_equal",
            )
        )
        sufficient_statistics.update(
            _relative_l2_evidence(
                "surface_wall_shear_rel_l2",
                surface_shear,
                physical_first=True,
                physical_dataset_weighting="surface_face_area",
                uniform_dataset_weighting="surface_entities_equal",
            )
        )
        sufficient_statistics.update(
            _relative_l2_evidence(
                "volume_pressure_rel_l2",
                volume_pressure.statistics,
                physical_first=False,
                physical_dataset_weighting="not_applicable_equal_native_cell_only",
                uniform_dataset_weighting="volume_cells_equal",
                include_secondary=False,
            )
        )
        sufficient_statistics.update(
            _relative_l2_evidence(
                "volume_velocity_rel_l2",
                volume_velocity.statistics,
                physical_first=False,
                physical_dataset_weighting="not_applicable_equal_native_cell_only",
                uniform_dataset_weighting="volume_cells_equal",
                include_secondary=False,
            )
        )
    except DrivAerAccumulatorError as error:
        raise DrivAerCandidateEvaluatorError(str(error)) from error

    if _sha256_file(native_source_pin.source_path) != source_pin_sha256:
        raise DrivAerCandidateEvaluatorError(
            "native-source pin changed during candidate evaluation"
        )
    fixed_surface_areas.assert_source_unchanged(
        context="while candidate surface metrics were reduced"
    )
    if _sha256_file(surface_manifest.path) != surface_manifest_sha256:
        raise DrivAerCandidateEvaluatorError(
            "surface prediction manifest changed during candidate evaluation"
        )
    if _sha256_file(volume_manifest.path) != volume_manifest_sha256:
        raise DrivAerCandidateEvaluatorError(
            "volume prediction manifest changed during candidate evaluation"
        )

    return CandidateCaseEvaluation(
        case_id=case_id,
        source_pin_sha256=source_pin_sha256,
        repository_id=native_source_pin.repository_id,
        repository_revision=native_source_pin.repository_revision,
        boundary_sha256=native_surface.boundary_sha256,
        surface_native_audit=native_surface.audit_record(),
        surface_area_audit={
            **fixed_surface_areas.audit_record(),
            "native_geometry_order_audit": surface_geometry_area_audit,
        },
        volume_part_sha256=tuple(part.sha256 for part in case.volume_parts),
        volume_vtk_audit={
            "dataset_type": volume_vtk_index.dataset_type,
            "version": volume_vtk_index.version,
            "byte_order": volume_vtk_index.byte_order,
            "header_type": volume_vtk_index.header_type,
            "compressor": volume_vtk_index.compressor,
            "piece_count": len(volume_vtk_index.pieces),
            "cell_count": volume_count,
            "source_size_bytes": volume_vtk_index.source_size_bytes,
        },
        volume_weighting_audit={
            "weighting": "one_per_native_cell",
            "entity_count": volume_count,
            "total_weight": float(volume_count),
            "geometric_cell_volume_weights_used": False,
        },
        surface_prediction_manifest_sha256=surface_manifest_sha256,
        volume_prediction_manifest_sha256=volume_manifest_sha256,
        surface_prediction_chunk_sha256=tuple(
            descriptor.sha256 for descriptor in surface_manifest.chunks
        ),
        volume_prediction_chunk_sha256=tuple(
            descriptor.sha256 for descriptor in volume_manifest.chunks
        ),
        surface_entity_count=native_surface.polygon_count,
        volume_entity_count=volume_count,
        surface_chunk_count=len(surface_manifest.chunks),
        volume_chunk_count=len(volume_manifest.chunks),
        volume_native_array_audits={
            name: audit.to_json() for name, audit in sorted(volume_audits.items())
        },
        metric_values=metric_values,
        metric_sufficient_statistics=sufficient_statistics,
        additive_sums={
            name: _additive_sums(
                value,
                include_physical=not name.startswith("volume_"),
            )
            for name, value in statistics.items()
        },
        force_coefficients=coefficients,
        execution={
            "maximum_prediction_chunk_rows": maximum_rows,
            "hash_chunk_bytes": hash_rows,
            "validation_block_rows": validation_rows,
            "encoded_chunk_bytes": encoded_bytes,
        },
    )


def write_candidate_case_evidence(
    evaluation: CandidateCaseEvaluation,
    path: Path | str,
) -> dict[str, object]:
    """Atomically write deterministic compact JSON and return its identity."""

    if not isinstance(evaluation, CandidateCaseEvaluation):
        raise DrivAerCandidateEvaluatorError(
            "evaluation must be a CandidateCaseEvaluation"
        )
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise DrivAerCandidateEvaluatorError("evidence output must use the .json suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
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
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file": destination.name,
        "byte_size": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


__all__ = [
    "CANDIDATE_EVIDENCE_SCHEMA",
    "CANDIDATE_STATUS",
    "DEFAULT_ENCODED_CHUNK_BYTES",
    "DEFAULT_MAX_PREDICTION_CHUNK_ROWS",
    "CandidateCaseEvaluation",
    "DrivAerCandidateEvaluatorError",
    "NativeSourceContract",
    "OFFICIAL_NATIVE_SOURCE_CONTRACT",
    "OFFICIAL_NATIVE_SOURCE_PIN_SHA256",
    "evaluate_candidate_case",
    "validate_native_source_contract",
    "write_candidate_case_evidence",
]
