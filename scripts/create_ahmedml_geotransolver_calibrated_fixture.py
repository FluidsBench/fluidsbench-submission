#!/usr/bin/env python3
"""Create a complete, non-ranked AhmedML evaluator-development fixture.

This command does *not* run the preserved GeoTransolver checkpoint.  The exact
training source and field exporter were not preserved.  Instead, it verifies
that checkpoint's immutable identity and applies deterministic signed scaling
errors to every native truth tuple.  The four error magnitudes are calibrated
to the checkpoint's recorded best-evaluation field errors.  The resulting
complete native fields are useful for exercising submission validation,
bounded-memory scoring, force integration, regions, and evaluator-owned
profiles without representing the artifact as genuine model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import vtk  # type: ignore[import-not-found]
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import-not-found]
except ImportError as error:  # pragma: no cover - command dependency check.
    raise SystemExit(f"VTK 9.5.2 is required: {error}") from error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    SOURCE_IDENTITY_SHA256,
    AhmedMLContractError,
    SourceFileIdentity,
    load_source_identity,
    sha256_file,
)
from reference.ahmedml.prediction_chunks import (  # noqa: E402
    AHMEDML_CANDIDATE_FORMAT,
    CANDIDATE_ARTIFACT_ROLE,
)


FIXTURE_SCHEMA = "ahmedml-geotransolver-calibrated-development-fixture-v1"
FIXTURE_ID = "geotransolver-calibrated-synthetic-dev-v1"
FIXTURE_STATUS = "synthetic_non_ranked_not_model_inference"
REQUIRED_VTK_VERSION = "9.5.2"
EXPECTED_CHECKPOINT_SHA256 = (
    "37ee4418ef1ee069b5c086322c4c5ef9c35f1a59ea4755b470b7d6358c642ac5"
)
EXPECTED_CHECKPOINT_SIZE_BYTES = 80_893_878
DEFAULT_CHUNK_ROWS = 1_000_000

# Ratios reported for the preserved checkpoint at its best, test-selected
# evaluation step.  The WSS and velocity values are the recorded vector-
# magnitude errors; this fixture applies them to the corresponding vectors.
TARGET_RELATIVE_L2: Mapping[str, float] = {
    "surface.pMean": 0.03506908565759659,
    "surface.wallShearStressMean": 0.05472808,
    "volume.pMean": 0.09651642,
    "volume.UMean": 0.05140058,
}
PATTERN_CYCLES: Mapping[str, int] = {
    "surface.pMean": 37,
    "surface.wallShearStressMean": 53,
    "volume.pMean": 71,
    "volume.UMean": 89,
}


class FixtureBuildError(ValueError):
    """Raised when the development fixture cannot be built exactly."""


def _canonical_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _verify_file(path: Path, identity: SourceFileIdentity, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError as error:
        raise FixtureBuildError(f"cannot resolve {label}: {error}") from error
    if size != identity.size_bytes or sha256_file(resolved) != identity.sha256:
        raise FixtureBuildError(f"{label} differs from the pinned public source identity")
    return resolved


def _read_cell_fields(
    path: Path,
    *,
    dataset_type: str,
    expected_count: int,
    fields: Mapping[str, int],
) -> dict[str, np.ndarray]:
    if dataset_type == "PolyData":
        reader = vtk.vtkXMLPolyDataReader()
    elif dataset_type == "UnstructuredGrid":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:  # pragma: no cover - closed internal call sites.
        raise FixtureBuildError(f"unsupported VTK dataset type {dataset_type!r}")
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    reader.GetPointDataArraySelection().DisableAllArrays()
    selection = reader.GetCellDataArraySelection()
    selection.DisableAllArrays()
    for name in fields:
        if not selection.ArrayExists(name):
            raise FixtureBuildError(f"{path.name} has no CellData array {name!r}")
        selection.EnableArray(name)
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfCells() != expected_count:
        raise FixtureBuildError(f"{path.name} native cell count differs")
    result: dict[str, np.ndarray] = {}
    for name, components in fields.items():
        vtk_array = data.GetCellData().GetArray(name)
        if vtk_array is None:
            raise FixtureBuildError(f"VTK did not load CellData array {name!r}")
        array = np.asarray(vtk_to_numpy(vtk_array))
        expected_shape = (expected_count,) if components == 1 else (
            expected_count,
            components,
        )
        if array.dtype != np.dtype("float32") or array.shape != expected_shape:
            raise FixtureBuildError(
                f"{name!r} has {array.dtype} {array.shape}, expected float32 "
                f"{expected_shape}"
            )
        if np.any(~np.isfinite(array)):
            raise FixtureBuildError(f"{name!r} contains non-finite truth values")
        result[name] = array
    return result


def _signed_scale(
    truth: np.ndarray,
    *,
    start: int,
    total: int,
    ratio: float,
    cycles: int,
    phase: float,
) -> np.ndarray:
    ids = np.arange(start, start + len(truth), dtype=np.float64)
    angle = 2.0 * math.pi * cycles * ((ids + 0.5) / total) + phase
    sign = np.where(np.sin(angle) >= 0.0, 1.0, -1.0).astype(np.float32)
    multiplier = np.float32(1.0) + np.float32(ratio) * sign
    if truth.ndim == 2:
        multiplier = multiplier[:, None]
    prediction = np.asarray(truth, dtype=np.float32) * multiplier
    if prediction.dtype != np.dtype("float32") or np.any(~np.isfinite(prediction)):
        raise FixtureBuildError("synthetic fixture transform produced invalid values")
    return prediction


def _write_support(
    root: Path,
    *,
    case_id: str,
    support_id: str,
    fields: Mapping[str, np.ndarray],
    field_prefix: str,
    run_id: int,
    chunk_rows: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    support_root = root / ("surface" if "surface" in support_id else "volume")
    chunk_root = support_root / "chunks"
    chunk_root.mkdir(parents=True)
    counts = {len(value) for value in fields.values()}
    if len(counts) != 1:
        raise FixtureBuildError(f"{support_id} field tuple counts differ")
    total = counts.pop()
    descriptors: list[dict[str, object]] = []
    error_numerator = {name: 0.0 for name in fields}
    truth_denominator = {name: 0.0 for name in fields}
    for chunk_index, start in enumerate(range(0, total, chunk_rows)):
        stop = min(start + chunk_rows, total)
        raw_ids = np.arange(start, stop, dtype="<i8")
        payload: dict[str, np.ndarray] = {"raw_cell_id": raw_ids}
        for field_index, (name, truth_all) in enumerate(fields.items()):
            key = f"{field_prefix}.{name}"
            truth = np.asarray(truth_all[start:stop], dtype=np.float32)
            phase = 2.0 * math.pi * (
                ((run_id * 17 + field_index * 29) % 101) / 101.0
            )
            prediction = _signed_scale(
                truth,
                start=start,
                total=total,
                ratio=TARGET_RELATIVE_L2[key],
                cycles=PATTERN_CYCLES[key],
                phase=phase,
            )
            payload[name] = prediction.astype("<f4", copy=False)
            difference = prediction.astype(np.float64) - truth.astype(np.float64)
            error_numerator[name] += float(np.sum(difference * difference))
            truth64 = truth.astype(np.float64)
            truth_denominator[name] += float(np.sum(truth64 * truth64))
        relative = Path("chunks") / f"chunk-{chunk_index:05d}.npz"
        path = support_root / relative
        # Stored NPZ is intentionally used here: this is a local development
        # fixture and fast deterministic generation is more useful than size.
        np.savez(path, **payload)
        descriptors.append(
            {
                "chunk_index": chunk_index,
                "file": relative.as_posix(),
                "sha256": sha256_file(path),
                "row_count": stop - start,
                "raw_cell_id_start": start,
                "raw_cell_id_stop": stop,
            }
        )
    components = {
        name: (1 if value.ndim == 1 else int(value.shape[1]))
        for name, value in fields.items()
    }
    manifest = {
        "format": AHMEDML_CANDIDATE_FORMAT,
        "format_version": 1,
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "case_id": case_id,
        "support_id": support_id,
        "association": "CellData",
        "total_row_count": total,
        "field_components": components,
        "chunks": descriptors,
    }
    manifest_sha = _canonical_json(support_root / "manifest.json", manifest)
    achieved = {
        name: math.sqrt(error_numerator[name] / truth_denominator[name])
        for name in fields
    }
    return (
        {
            "manifest": f"{support_root.name}/manifest.json",
            "manifest_sha256": manifest_sha,
            "chunk_count": len(descriptors),
            "entity_count": total,
        },
        achieved,
    )


def create_fixture(
    *,
    case_id: str,
    dataset_root: Path,
    output_root: Path,
    source_identity_path: Path,
    checkpoint: Path,
    chunk_rows: int,
) -> Path:
    if vtk.vtkVersion.GetVTKVersion() != REQUIRED_VTK_VERSION:
        raise FixtureBuildError(
            f"fixture generation requires VTK {REQUIRED_VTK_VERSION}, got "
            f"{vtk.vtkVersion.GetVTKVersion()}"
        )
    if chunk_rows < 1 or chunk_rows > DEFAULT_CHUNK_ROWS:
        raise FixtureBuildError(
            f"chunk_rows must lie in [1, {DEFAULT_CHUNK_ROWS}]"
        )
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if (
        checkpoint.stat().st_size != EXPECTED_CHECKPOINT_SIZE_BYTES
        or sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256
    ):
        raise FixtureBuildError("checkpoint identity differs from the audited model_best.pt")
    identity = load_source_identity(source_identity_path)
    case = identity.case(case_id)
    boundary = _verify_file(
        case.boundary.resolve(dataset_root), case.boundary, f"{case_id} boundary"
    )
    volume = _verify_file(
        case.volume.resolve(dataset_root), case.volume, f"{case_id} volume"
    )
    output_root = output_root.expanduser().resolve()
    final = output_root / case_id
    if final.exists():
        raise FixtureBuildError(f"output already exists: {final}")
    output_root.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{case_id}-partial-", dir=output_root))
    try:
        surface_fields = _read_cell_fields(
            boundary,
            dataset_type="PolyData",
            expected_count=case.surface_entity_count,
            fields={"pMean": 1, "wallShearStressMean": 3},
        )
        surface_artifact, surface_achieved = _write_support(
            partial,
            case_id=case_id,
            support_id="ahmedml_surface_native_cells",
            fields=surface_fields,
            field_prefix="surface",
            run_id=case.run_id,
            chunk_rows=chunk_rows,
        )
        del surface_fields
        volume_fields = _read_cell_fields(
            volume,
            dataset_type="UnstructuredGrid",
            expected_count=case.volume_entity_count,
            fields={"pMean": 1, "UMean": 3},
        )
        volume_artifact, volume_achieved = _write_support(
            partial,
            case_id=case_id,
            support_id="ahmedml_volume_native_cells",
            fields=volume_fields,
            field_prefix="volume",
            run_id=case.run_id,
            chunk_rows=chunk_rows,
        )
        del volume_fields
        provenance = {
            "schema": FIXTURE_SCHEMA,
            "schema_version": 1,
            "fixture_id": FIXTURE_ID,
            "status": FIXTURE_STATUS,
            "actual_model_inference": False,
            "official_submission": False,
            "leaderboard_eligible": False,
            "purpose": (
                "exercise the complete AhmedML candidate evaluator and dev dashboard "
                "with checkpoint-calibrated but synthetic native fields"
            ),
            "case_id": case_id,
            "run_id": case.run_id,
            "source": {
                "repository_id": identity.repository_id,
                "repository_revision": identity.repository_revision,
                "source_identity_sha256": SOURCE_IDENTITY_SHA256,
                "boundary_sha256": case.boundary.sha256,
                "volume_sha256": case.volume.sha256,
            },
            "calibration_checkpoint": {
                "file_name": checkpoint.name,
                "sha256": EXPECTED_CHECKPOINT_SHA256,
                "size_bytes": EXPECTED_CHECKPOINT_SIZE_BYTES,
                "best_selection_split": "legacy_configured_test_split",
                "warning": (
                    "the checkpoint was not executed; exact training source and a field "
                    "exporter were not preserved"
                ),
            },
            "transform": {
                "id": "deterministic_signed_multiplicative_error_v1",
                "equation": "prediction = truth * (1 + target_ratio * sign(sin(pattern)))",
                "pattern_basis": "complete_native_raw_cell_id_and_run_id",
                "target_relative_l2_ratio": dict(TARGET_RELATIVE_L2),
                "pattern_cycles": dict(PATTERN_CYCLES),
            },
            "achieved_unweighted_relative_l2_ratio": {
                "surface.pMean": surface_achieved["pMean"],
                "surface.wallShearStressMean": surface_achieved[
                    "wallShearStressMean"
                ],
                "volume.pMean": volume_achieved["pMean"],
                "volume.UMean": volume_achieved["UMean"],
            },
            "prediction_inputs": {
                "surface": surface_artifact,
                "volume": volume_artifact,
            },
        }
        provenance_sha = _canonical_json(partial / "fixture-provenance.json", provenance)
        os.rename(partial, final)
    except Exception:
        # Keep a failed partial for forensic diagnosis rather than silently
        # deleting potentially useful evidence.
        raise
    print(
        json.dumps(
            {
                "case_id": case_id,
                "output": str(final),
                "fixture_provenance_sha256": provenance_sha,
                "actual_model_inference": False,
                "leaderboard_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--source-identity",
        type=Path,
        default=(
            ROOT
            / "benchmark-specs"
            / "ahmedml"
            / "public-source-identity"
            / "ahmedml-public-source-identity-v1.json"
        ),
    )
    args = parser.parse_args()
    try:
        create_fixture(
            case_id=args.case_id,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            source_identity_path=args.source_identity,
            checkpoint=args.checkpoint,
            chunk_rows=args.chunk_rows,
        )
    except (AhmedMLContractError, FixtureBuildError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
