"""AhmedML-owned interface to the shared bounded NPZ prediction transport.

The byte-level parser is shared with DrivAerML because both datasets retain
native ``CellData`` ordering.  AhmedML callers import this module so the
dataset-facing contract, allowed supports, and future versioning do not leak
through a DrivAerML namespace.
"""

from __future__ import annotations

from pathlib import Path

from reference.drivaerml.prediction_chunks import (
    AHMEDML_CANDIDATE_FORMAT,
    CANDIDATE_ARTIFACT_ROLE,
    CANDIDATE_FORMAT_VERSION,
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_MAX_MANIFEST_BYTES,
    DEFAULT_MAX_NPY_HEADER_BYTES,
    DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES,
    DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    PredictionChunk,
    PredictionChunkDescriptor,
    PredictionChunkError,
    PredictionChunkManifest,
    PredictionChunkValidation,
    RAW_CELL_ID_FIELD,
    iter_prediction_chunks,
    load_prediction_chunk_manifest as _load_shared_manifest,
    support_field_components,
    validate_prediction_chunks,
)


AHMEDML_SUPPORT_IDS = frozenset(
    {"ahmedml_surface_native_cells", "ahmedml_volume_native_cells"}
)


def load_prediction_chunk_manifest(
    path: str | Path,
) -> PredictionChunkManifest:
    """Load one manifest and fail closed unless it is AhmedML-owned."""

    manifest = _load_shared_manifest(path)
    if manifest.support_id not in AHMEDML_SUPPORT_IDS:
        raise PredictionChunkError(
            f"support_id {manifest.support_id!r} is not an AhmedML native support"
        )
    return manifest


__all__ = [
    "AHMEDML_CANDIDATE_FORMAT",
    "AHMEDML_SUPPORT_IDS",
    "CANDIDATE_ARTIFACT_ROLE",
    "CANDIDATE_FORMAT_VERSION",
    "DEFAULT_HASH_CHUNK_BYTES",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_NPY_HEADER_BYTES",
    "DEFAULT_MAX_NPZ_CENTRAL_DIRECTORY_BYTES",
    "DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES",
    "DEFAULT_VALIDATION_BLOCK_ROWS",
    "PredictionChunk",
    "PredictionChunkDescriptor",
    "PredictionChunkError",
    "PredictionChunkManifest",
    "PredictionChunkValidation",
    "RAW_CELL_ID_FIELD",
    "iter_prediction_chunks",
    "load_prediction_chunk_manifest",
    "support_field_components",
    "validate_prediction_chunks",
]
