"""WindsorML-owned interface to the shared bounded NPZ prediction transport.

The byte-level parser is shared with DrivAerML. WindsorML is the first dataset
whose surface support is ``PointData`` rather than ``CellData``: its raw IDs
index native boundary *points*, because the published ``cpavg``/``cf*avg``
fields and the ``boundary_dual_area_N.npy`` quadrature weights are both
per-point. The volume support remains ``CellData``.

WindsorML callers import this module so the dataset-facing contract, allowed
supports, and future versioning do not leak through a DrivAerML namespace.
"""

from __future__ import annotations

from pathlib import Path

from reference.drivaerml.prediction_chunks import (
    CANDIDATE_ARTIFACT_ROLE,
    CANDIDATE_FORMAT_VERSION,
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_MAX_MANIFEST_BYTES,
    DEFAULT_MAX_NPY_HEADER_BYTES,
    DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES,
    DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    RAW_CELL_ID_FIELD,
    WINDSORML_CANDIDATE_FORMAT,
    PredictionChunk,
    PredictionChunkDescriptor,
    PredictionChunkError,
    PredictionChunkManifest,
    PredictionChunkValidation,
    iter_prediction_chunks,
    load_prediction_chunk_manifest as _load_shared_manifest,
    support_association,
    support_field_components,
    validate_prediction_chunks,
)


WINDSORML_SURFACE_SUPPORT_ID = "windsorml_surface_native_points"
WINDSORML_VOLUME_SUPPORT_ID = "windsorml_volume_native_cells"
WINDSORML_SUPPORT_IDS = frozenset(
    {WINDSORML_SURFACE_SUPPORT_ID, WINDSORML_VOLUME_SUPPORT_ID}
)


def load_prediction_chunk_manifest(path: str | Path) -> PredictionChunkManifest:
    """Load one manifest and fail closed unless it is WindsorML-owned."""

    manifest = _load_shared_manifest(path)
    if manifest.support_id not in WINDSORML_SUPPORT_IDS:
        raise PredictionChunkError(
            f"support_id {manifest.support_id!r} is not a WindsorML native support"
        )
    return manifest


__all__ = [
    "CANDIDATE_ARTIFACT_ROLE",
    "CANDIDATE_FORMAT_VERSION",
    "DEFAULT_HASH_CHUNK_BYTES",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_NPY_HEADER_BYTES",
    "DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES",
    "DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES",
    "DEFAULT_VALIDATION_BLOCK_ROWS",
    "RAW_CELL_ID_FIELD",
    "WINDSORML_CANDIDATE_FORMAT",
    "WINDSORML_SUPPORT_IDS",
    "WINDSORML_SURFACE_SUPPORT_ID",
    "WINDSORML_VOLUME_SUPPORT_ID",
    "PredictionChunk",
    "PredictionChunkDescriptor",
    "PredictionChunkError",
    "PredictionChunkManifest",
    "PredictionChunkValidation",
    "iter_prediction_chunks",
    "load_prediction_chunk_manifest",
    "support_association",
    "support_field_components",
    "validate_prediction_chunks",
]
