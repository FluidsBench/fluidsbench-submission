#!/usr/bin/env python3
"""Create deterministic all-zero DrivAerML candidate prediction chunks.

The output is only a local transport fixture for the closed candidate
evaluator.  This script does not read native truth, evaluate model quality, or
create an official FluidsBench submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.prediction_chunks import (  # noqa: E402
    CANDIDATE_ARTIFACT_ROLE,
    CANDIDATE_FORMAT,
    CANDIDATE_FORMAT_VERSION,
    PredictionChunkError,
    support_field_components,
    validate_prediction_chunks,
)


RECEIPT_SCHEMA = "drivaerml-candidate-dummy-predictions-v1"
RECEIPT_STATUS = "candidate_all_zero_transport_fixture_validated"
DEFAULT_MAX_CHUNK_ROWS = 1_000_000
_CASE_ID_PATTERN = re.compile(r"run_[1-9][0-9]*\Z")
_MAX_INT64 = int(np.iinfo(np.int64).max)
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_COMPRESSION_LEVEL = 9


class DummyPredictionError(ValueError):
    """Raised when deterministic dummy generation cannot proceed safely."""


def _positive_int64(value: object, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > _MAX_INT64
    ):
        raise DummyPredictionError(
            f"{label} must be a positive integer within the signed int64 range"
        )
    return value


def _positive_int64_argument(text: str) -> int:
    try:
        value = int(text)
        return _positive_int64(value, "value")
    except (ValueError, DummyPredictionError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _raw_id_ranges(entity_count: int, max_chunk_rows: int) -> Iterator[tuple[int, int]]:
    for start in range(0, entity_count, max_chunk_rows):
        yield start, min(entity_count, start + max_chunk_rows)


def _zip_member(name: str) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIME)
    member.compress_type = zipfile.ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = 0o100644 << 16
    member.internal_attr = 0
    member.flag_bits = 0
    member._compresslevel = _ZIP_COMPRESSION_LEVEL
    return member


def _write_array_member(
    archive: zipfile.ZipFile,
    name: str,
    array: np.ndarray,
) -> None:
    with archive.open(_zip_member(name), mode="w", force_zip64=True) as member:
        np.lib.format.write_array(
            member,
            array,
            version=(1, 0),
            allow_pickle=False,
        )


def _atomic_dummy_npz(
    destination: Path,
    *,
    support_id: str,
    raw_cell_id_start: int,
    raw_cell_id_stop: int,
) -> None:
    """Write one deterministic archive while retaining one array at a time."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = raw_cell_id_stop - raw_cell_id_start
    vector_name = (
        "wallShearStressMeanTrim"
        if support_id == "surface_native_cells"
        else "UMeanTrim"
    )
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=_ZIP_COMPRESSION_LEVEL,
                allowZip64=True,
                strict_timestamps=True,
            ) as archive:
                raw_ids = np.arange(
                    raw_cell_id_start,
                    raw_cell_id_stop,
                    dtype="<i8",
                )
                _write_array_member(archive, "raw_cell_id", raw_ids)
                del raw_ids

                pressure = np.zeros(row_count, dtype="<f4")
                _write_array_member(archive, "pMeanTrim", pressure)
                del pressure

                vector = np.zeros((row_count, 3), dtype="<f4")
                _write_array_member(archive, vector_name, vector)
                del vector
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(destination: Path, value: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
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
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _create_support(
    root: Path,
    *,
    case_id: str,
    support_id: str,
    entity_count: int,
    max_chunk_rows: int,
) -> Path:
    support_root = root / support_id
    chunks: list[dict[str, object]] = []
    for chunk_index, (start, stop) in enumerate(
        _raw_id_ranges(entity_count, max_chunk_rows)
    ):
        relative_file = Path("chunks") / f"chunk-{chunk_index:05d}.npz"
        chunk_path = support_root / relative_file
        _atomic_dummy_npz(
            chunk_path,
            support_id=support_id,
            raw_cell_id_start=start,
            raw_cell_id_stop=stop,
        )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "file": relative_file.as_posix(),
                "sha256": _sha256_file(chunk_path),
                "row_count": stop - start,
                "raw_cell_id_start": start,
                "raw_cell_id_stop": stop,
            }
        )
    manifest: dict[str, object] = {
        "format": CANDIDATE_FORMAT,
        "format_version": CANDIDATE_FORMAT_VERSION,
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "case_id": case_id,
        "support_id": support_id,
        "association": "CellData",
        "total_row_count": entity_count,
        "field_components": dict(support_field_components(support_id)),
        "chunks": chunks,
    }
    manifest_path = support_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def create_dummy_predictions(
    *,
    case_id: str,
    surface_count: int,
    volume_count: int,
    output_root: Path | str,
    max_chunk_rows: int = DEFAULT_MAX_CHUNK_ROWS,
) -> dict[str, object]:
    """Create and validate two all-zero candidate-support manifests atomically."""

    if not isinstance(case_id, str) or _CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise DummyPredictionError("case_id must have the form run_<positive integer>")
    counts = {
        "surface_native_cells": _positive_int64(surface_count, "surface_count"),
        "volume_native_cells": _positive_int64(volume_count, "volume_count"),
    }
    maximum = _positive_int64(max_chunk_rows, "max_chunk_rows")
    destination = Path(output_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DummyPredictionError(f"output root already exists: {destination}")

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as staging_text:
        staging = Path(staging_text)
        manifests = {
            support_id: _create_support(
                staging,
                case_id=case_id,
                support_id=support_id,
                entity_count=entity_count,
                max_chunk_rows=maximum,
            )
            for support_id, entity_count in counts.items()
        }
        support_receipts: dict[str, object] = {}
        for support_id, manifest_path in manifests.items():
            validation = validate_prediction_chunks(
                manifest_path,
                validation_block_rows=maximum,
            )
            support_receipts[support_id] = {
                "manifest": f"{support_id}/manifest.json",
                "manifest_sha256": validation.manifest_sha256,
                "entity_count": validation.total_row_count,
                "chunk_count": validation.chunk_count,
                "chunk_sha256": list(validation.chunk_sha256),
                "complete_gap_free_duplicate_free_coverage": (
                    validation.complete_gap_free_duplicate_free_coverage
                ),
            }
        receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": 1,
            "status": RECEIPT_STATUS,
            "case_id": case_id,
            "artifact_role": CANDIDATE_ARTIFACT_ROLE,
            "official_submission": False,
            "reads_native_truth": False,
            "claims_model_quality": False,
            "prediction_semantics": "deterministic_all_zero_transport_fixture_only",
            "array_dtypes": {
                "raw_cell_id": "<i8",
                "prediction_fields": "<f4",
            },
            "max_chunk_rows": maximum,
            "archive_determinism": {
                "member_order_by_support": {
                    "surface_native_cells": [
                        "raw_cell_id.npy",
                        "pMeanTrim.npy",
                        "wallShearStressMeanTrim.npy",
                    ],
                    "volume_native_cells": [
                        "raw_cell_id.npy",
                        "pMeanTrim.npy",
                        "UMeanTrim.npy",
                    ],
                },
                "npy_format_version": "1.0",
                "zip_compression": "deflate",
                "zip_compression_level": _ZIP_COMPRESSION_LEVEL,
                "zip_member_timestamp": "1980-01-01T00:00:00",
            },
            "supports": support_receipts,
        }
        _atomic_json(staging / "receipt.json", receipt)
        if destination.exists():
            raise DummyPredictionError(
                f"output root appeared during generation: {destination}"
            )
        os.replace(staging, destination)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic all-zero local candidate chunks for both "
            "DrivAerML supports. No truth is read and no official submission is made."
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--surface-count", type=_positive_int64_argument, required=True)
    parser.add_argument("--volume-count", type=_positive_int64_argument, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-chunk-rows",
        type=_positive_int64_argument,
        default=DEFAULT_MAX_CHUNK_ROWS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = create_dummy_predictions(
            case_id=args.case_id,
            surface_count=args.surface_count,
            volume_count=args.volume_count,
            output_root=args.output_root,
            max_chunk_rows=args.max_chunk_rows,
        )
    except (DummyPredictionError, OSError, PredictionChunkError) as error:
        raise SystemExit(f"dummy prediction generation failed: {error}") from error
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_CHUNK_ROWS",
    "RECEIPT_SCHEMA",
    "RECEIPT_STATUS",
    "DummyPredictionError",
    "create_dummy_predictions",
    "main",
    "parse_args",
]
