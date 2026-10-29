#!/usr/bin/env python3
"""Export the real Transolver run_419 current-format profile fixture.

This release-maintainer utility intentionally has no discovery or best-effort
mode.  It accepts the original run_419 prediction transport and the retained
producer supports, verifies their exact identities, and uses the evaluator in
this repository to gather the native-cell predictions.  Only the compact
schema-v3 profile package and a path-free provenance record are written.

The output is one-case regression evidence.  It is not an official
submission, sensitivity evidence, or an activation record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    U_INF_M_PER_S,
    cp_from_kinematic_pressure,
    velocity_magnitude_ratio,
)
from reference.drivaerml.coordinate_identity import (  # noqa: E402
    CoordinateIdentityError,
    coordinate_array_identity_sha256,
)
from reference.drivaerml.dataset_scorer import (  # noqa: E402
    CandidateDatasetEvaluation,
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_FORMAT,
    RUN419_CONSTANT_SERIES_SUPPORT_INDEX_PATH,
    RUN419_CONSTANT_SERIES_SUPPORT_INDEX_SHA256,
    _relative_profile_expected_keys,
    _relative_series_support_index,
    _run419_constant_series_support_index,
    _validate_core_case,
    schema_v3_relative_profile_chunks_candidate_adapter,
    write_schema_v3_profile_chunks_candidate,
)
from reference.drivaerml.diagnostic_evaluator import (  # noqa: E402
    gather_mapped_prediction_field,
)
from reference.drivaerml.prediction_chunks import (  # noqa: E402
    PredictionChunkManifest,
    load_prediction_chunk_manifest,
)
from reference.drivaerml.source import load_native_source_pin  # noqa: E402
from reference.drivaerml.surface_forces import (  # noqa: E402
    REFERENCE_AREA_M2,
    REFERENCE_LENGTH_M,
    RHO_INF_KG_PER_M3,
)
from scripts.evaluate_drivaerml_candidate_case import (  # noqa: E402
    CORE_EVALUATOR_GIT_PATHS,
    IMPLEMENTATION_RECEIPT_SCHEMA,
    IMPLEMENTATION_RECEIPT_STATUS,
)


CASE_ID = "run_419"
DATASET_REPOSITORY = "neashton/drivaerml"
DATASET_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
NATIVE_SOURCE_PIN_SHA256 = (
    "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
)
BOUNDARY_SHA256 = (
    "e6ee75027b0dd87d44fc3f49188ecec95a5193ac0de148babe9dca95424da1c5"
)
VOLUME_PART_SHA256 = (
    "8dd6cd2f62969ee11824adcd760b66f62f983ccaa69b9be91070557b12e79570",
    "421a23dd4bed3e312df137d2a390bad21b7d1f8e98f99f5f3ee751db11641e5f",
)
SURFACE_ENTITY_COUNT = 7_284_102
VOLUME_ENTITY_COUNT = 121_635_947

SURFACE_MANIFEST_SHA256 = (
    "299733656cba6fc397100cc1aa8d0538167f9299f964d64453ea11c3f8289ced"
)
VOLUME_MANIFEST_SHA256 = (
    "d1e5792b5b4aa96090d761513059103e1021516f087080ae22df849b124598dd"
)
SURFACE_CHUNK_COUNT = 8
VOLUME_CHUNK_COUNT = 122

CHECKPOINT_PROVENANCE_SHA256 = (
    "230f454cc023fad53cda37f1dc3213471f18661a30b60bf2bef085dbb90cde26"
)
MODEL_REPOSITORY = "nvidia/transolver_drivaerml"
MODEL_REVISION = "96477aeb86d24c26ccf0797bca1b3851268017d0"
SURFACE_CHECKPOINT_SHA256 = (
    "eb98f399a050a8f8a24919335c61642e4a835bd4044f7e21abec231aa31fd82c"
)
SURFACE_CHECKPOINT_BYTES = 39_188_235
SURFACE_STATS_SHA256 = (
    "c4b19eaca6158219570232ce094e6ef48b8705eb727b335441a310c7479b6b25"
)
VOLUME_CHECKPOINT_SHA256 = (
    "436804a02c1fe4cc3f70072826df933a39b0e98bc88bd4edf231eeb1d10e55d3"
)
VOLUME_CHECKPOINT_BYTES = 39_191_307
VOLUME_STATS_SHA256 = (
    "adedca100650444b53e304cf9d85870a8299fbac0545e4f8a2e6f7c28086127f"
)
SURFACE_EXECUTION_RECEIPT_SHA256 = (
    "0cda6f3ea9477d9cc2ee63528a3cbffcd3380d59f1dc8ac061163269bc8145bd"
)
SURFACE_CONVERSION_RECEIPT_SHA256 = (
    "39fb1b7089384a0a1d1fa733c6228c9aebe39d3e8bc7072723f93f27743fc7aa"
)
VOLUME_INFERENCE_RECEIPT_SHA256 = (
    "b0053034762cc4c0bc97c721a2233f04630c5a47a53d83ee2e3a5e660d3c8306"
)

CONSTANT_VELOCITY_MAPPING_SHA256 = (
    "9866147358a34540e6f6be4b95e3cdc46681693a22bd93f0d5ef790144bddcec"
)
CONSTANT_VELOCITY_RECEIPT_SHA256 = (
    "63df3170bb71edcdb1a9892f64ed33f4ecd028f7bbb172e91e06efb34bee96bf"
)
CONSTANT_VELOCITY_PROFILE_SHA256 = (
    "df22bc807b62f925c32659d681ac44064e6acf46449038b8431b1e9139aba1e8"
)

VELOCITY_PLACEMENT_MANIFEST_SHA256 = (
    "70627d6af9e6b254739b29d54d856b470d6152066e09a1097ebe20c034179925"
)
VELOCITY_MAPPING_MANIFEST_SHA256 = (
    "9b88c36e2268bf72c9baec9418ef2d9afc9faba1d98c9c12ed73b79e683a6a3d"
)
CP_SUPPORT_MANIFEST_SHA256 = (
    "4e6a4c3495ea4938895868162480dcb20b5bbea42114c94013a2c76e26128c90"
)
RELATIVE_VELOCITY_COORDINATE_CSV_SHA256 = (
    "440e24531a559276006ef0e249796c77707ddd8cdedae11ebdb6b4f8dbee93a0"
)
RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256 = (
    "4a2697c25cfe6ffb2c46bd8a4cd4ccc6c64ea76b728c55148d58b5d2c6146e41"
)
RELATIVE_VELOCITY_MAPPING_SHA256 = (
    "7b61b6a444379cf7edb1cb0e07a7a6b202cf87084b7d55720e7c225cf591ce1e"
)
RELATIVE_VELOCITY_MAPPING_RECEIPT_SHA256 = (
    "f7df45705d896236c31c4ab7c13039c001ddd5d0845f599a6225599230729833"
)

CONSTANT_CP_AGGREGATE_SHA256 = (
    "bc2a7337ae87942e9b9ae4f57a5ba408bbdf82d03307d47110fcc85edd449c7c"
)
CONSTANT_CP_JSON_SHA256 = (
    "e2fb791194151aab95e03b21cf7689e4b78994d21073817c663599e12cc942e8"
)
CONSTANT_CP_CSV_SHA256 = (
    "e2e0c84f4e3306ad4ed3b67a7de3ad10dc947c090103cb2de7c5b91fe7e27b8a"
)
CONSTANT_CP_RECEIPT_FILE_SHA256 = (
    "b1ea8c26f753935e5c9826495dfb77cf97bd542fcba1fceeab575030cefd9195"
)
CONSTANT_CP_RECEIPT_IDENTITY_SHA256 = (
    "4c120f02ed318a55025330c232c3a46b16820baadc81515dc729d31cc822926d"
)
CONSTANT_CP_CASE_SUPPORT_SHA256 = (
    "b5ed95dbbcff30ad6d1ebc07c1fa1c484665cd0416b1d6d2d53945ef64b61e57"
)
RELATIVE_CP_JSON_SHA256 = (
    "1d947e4d5c1dd6fe5f1cd5eb13a0fd7e38f156c282f5b5c58273b753f67a15bd"
)
RELATIVE_CP_CSV_SHA256 = (
    "febf0dacbaa185e3cadad26ee32b39e7c9e65cc4d4480e8ef56fab73f096dea4"
)
RELATIVE_CP_ARTIFACT_RECEIPT_SHA256 = (
    "c05bba6fec3e269fe437e32835cb170fce17e0cfacbd2a11a4613d74eab65adb"
)
RELATIVE_CP_PLACEMENT_FILE_SHA256 = (
    "33235e7c660bac5755d27ee5e5fd6d8ddcea5e8508c138c454fce01eaee706e4"
)
RELATIVE_CP_CASE_SUPPORT_SHA256 = (
    "c633a0a9174643fbbd1e80b78aa26ab56980bd4a915b08f7203a21aab1846d56"
)

VELOCITY_STATIONS = (
    "V1", "V2", "V3", "V4", "V5", "V6",
    "U1", "U2", "U3", "U4", "U5", "U6",
    "L1", "R1", "R2", "R3",
)
VELOCITY_SAMPLE_COUNTS = {
    **{station: 201 for station in VELOCITY_STATIONS[:6]},
    **{station: 301 for station in VELOCITY_STATIONS[6:12]},
    "L1": 651,
    "R1": 31,
    "R2": 31,
    "R3": 31,
}
CONSTANT_CP_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
    "sidewall_z_0_15",
    "front_left_wheelhouse_y_neg_0_6",
)
RELATIVE_CP_ALIAS_STATIONS = (
    "upperbody_centerline",
    "underbody_centerline",
)
RELATIVE_CP_MOVING_STATIONS = (
    "sidewall_front_wheelhouse_relative",
    "front_left_wheelhouse_relative",
)
SHARED_SUPPORT_IDS = {
    "upperbody_centerline": "drivaerml-cp-upperbody-centerline-y0-v1",
    "underbody_centerline": "drivaerml-cp-underbody-centerline-y0-v1",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

EVALUATOR_GIT_PATHS = (
    "scripts/evaluate_drivaerml_candidate_case.py",
    "scripts/export_drivaerml_run419_relative_fixture.py",
    "reference/drivaerml/__init__.py",
    "reference/drivaerml/accumulators.py",
    "reference/drivaerml/autocfd5.py",
    "reference/drivaerml/coordinate_identity.py",
    "reference/drivaerml/cp_mapping.py",
    "reference/drivaerml/dataset_scorer.py",
    "reference/drivaerml/diagnostic_evaluator.py",
    "reference/drivaerml/evaluator.py",
    "reference/drivaerml/native_fields.py",
    "reference/drivaerml/native_surface.py",
    "reference/drivaerml/prediction_chunks.py",
    "reference/drivaerml/retained_file.py",
    "reference/drivaerml/source.py",
    "reference/drivaerml/surface_forces.py",
    "reference/drivaerml/velocity_assignments.py",
    "scripts/generate_drivaerml_run419_constant_support_index.py",
    "benchmark-specs/drivaerml/support/relative-v3/series-support-index.json",
    (
        "benchmark-specs/drivaerml/support/relative-v3/"
        "run419-constant-series-support-index.json"
    ),
)


class ExportError(ValueError):
    """Raised when run_419 inputs cannot prove the requested fixture."""


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the original Transolver run_419 native prediction transport "
            "and export a compact current schema-v3 40-series fixture."
        )
    )
    parser.add_argument(
        "--native-source-pin",
        type=Path,
        default=ROOT / "benchmark-specs/drivaerml/proposal/native-source-pin.json",
    )
    parser.add_argument("--current-case-evaluation-evidence", type=Path, required=True)
    parser.add_argument(
        "--current-case-implementation-receipt",
        type=Path,
        required=True,
        help=(
            "Deterministic receipt emitted by the current full-case evaluator "
            "that binds the evidence bytes to --evaluator-git-revision."
        ),
    )
    parser.add_argument("--surface-prediction-manifest", type=Path, required=True)
    parser.add_argument("--volume-prediction-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--surface-checkpoint", type=Path, required=True)
    parser.add_argument("--surface-stats", type=Path, required=True)
    parser.add_argument("--volume-checkpoint", type=Path, required=True)
    parser.add_argument("--volume-stats", type=Path, required=True)
    parser.add_argument("--surface-execution-receipt", type=Path, required=True)
    parser.add_argument("--surface-conversion-receipt", type=Path, required=True)
    parser.add_argument("--volume-inference-receipt", type=Path, required=True)
    parser.add_argument(
        "--constant-velocity-case-dir",
        type=Path,
        required=True,
        help="Directory containing run_419 receipt.json and velocity-cell-mapping-10mm.json.",
    )
    parser.add_argument(
        "--relative-producer-root",
        type=Path,
        required=True,
        help="Root containing velocity_support_v3, velocity_mapping_v3, and cp_support.",
    )
    parser.add_argument(
        "--constant-cp-campaign-root",
        type=Path,
        required=True,
        help="The production_path_all484_v8 root.",
    )
    parser.add_argument(
        "--relative-support-index",
        type=Path,
        default=(
            ROOT
            / "benchmark-specs/drivaerml/support/relative-v3/series-support-index.json"
        ),
    )
    parser.add_argument(
        "--constant-support-index",
        type=Path,
        default=RUN419_CONSTANT_SERIES_SUPPORT_INDEX_PATH,
        help="Repository-retained run_419 constant-series support binding.",
    )
    parser.add_argument(
        "--evaluator-git-revision",
        required=True,
        help="Exact commit whose evaluator/exporter bytes must match this checkout.",
    )
    parser.add_argument(
        "--submission-id",
        default="transolver-run419-current-relative-fixture",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-prediction-chunk-rows",
        type=_positive_integer,
        default=1_000_000,
    )
    parser.add_argument(
        "--hash-chunk-bytes", type=_positive_integer, default=1024 * 1024
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ExportError(f"{label} cannot be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ExportError(f"{label} must be a regular file")
    return resolved


def _directory(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ExportError(f"{label} cannot be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ExportError(f"{label} must be a directory")
    return resolved


def _read_bytes(path: Path, label: str, expected_sha256: str | None = None) -> bytes:
    source = _regular(path, label)
    digest = hashlib.sha256()
    parts: list[bytes] = []
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            parts.append(block)
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ExportError(f"{label} SHA-256 differs: {actual}")
    return b"".join(parts)


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ExportError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExportError(f"{label} contains non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ExportError(f"{label} must be a JSON object")
    return value


def _load_json(
    path: Path, label: str, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str, int]:
    payload = _read_bytes(path, label, expected_sha256)
    return _strict_json_bytes(payload, label), _sha256_bytes(payload), len(payload)


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExportError(f"{label} must be finite")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExportError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ExportError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_mapping(
    value: object, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError(f"{label} must be an object")
    if set(value) != expected:
        raise ExportError(
            f"{label} keys differ "
            f"(missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)})"
        )
    return value


def _one_case(document: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list):
        raise ExportError(f"{label} cases must be an array")
    matches = [
        item
        for item in raw_cases
        if isinstance(item, Mapping) and item.get("case_id") == CASE_ID
    ]
    if len(matches) != 1:
        raise ExportError(f"{label} must contain {CASE_ID} exactly once")
    return matches[0]


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.decode("utf-8", "replace").strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise ExportError(f"git {' '.join(arguments)} failed: {detail}") from error


def _verify_evaluator_revision(revision: str) -> tuple[str, list[dict[str, object]]]:
    if not isinstance(revision, str) or not revision:
        raise ExportError("evaluator Git revision is empty")
    commit = _git("rev-parse", "--verify", f"{revision}^{{commit}}").stdout.decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ExportError("evaluator Git revision did not resolve to a full commit")
    if _git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode != 0:
        raise ExportError("evaluator Git revision is not reachable from checkout HEAD")
    implementation: list[dict[str, object]] = []
    for relative_path in EVALUATOR_GIT_PATHS:
        committed = _git("show", f"{commit}:{relative_path}").stdout
        local = _read_bytes(ROOT / relative_path, f"current {relative_path}")
        if committed != local:
            raise ExportError(
                f"current {relative_path} bytes differ from evaluator commit {commit}"
            )
        implementation.append(
            {
                "path": relative_path,
                "sha256": _sha256_bytes(local),
                "size_bytes": len(local),
            }
        )
    return commit, implementation


def _verify_native_pin(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    document, _, _ = _load_json(path, "native source pin", NATIVE_SOURCE_PIN_SHA256)
    repository = document.get("repository")
    if (
        document.get("schema") != "drivaerml-fluidsbench-public-native-source-pin-v1"
        or document.get("schema_version") != 1
        or not isinstance(repository, Mapping)
        or repository.get("repo_id") != DATASET_REPOSITORY
        or repository.get("repo_type") != "dataset"
        or repository.get("revision") != DATASET_REVISION
    ):
        raise ExportError("native source pin does not bind the exact dataset revision")
    case = _one_case(document, "native source pin")
    boundary = case.get("boundary")
    volume = case.get("volume")
    parts = volume.get("parts") if isinstance(volume, Mapping) else None
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("lfs_sha256") != BOUNDARY_SHA256
        or boundary.get("size_bytes") != 543_851_361
        or not isinstance(volume, Mapping)
        or volume.get("total_size_bytes") != 40_392_954_997
        or not isinstance(parts, list)
        or tuple(item.get("lfs_sha256") for item in parts if isinstance(item, Mapping))
        != VOLUME_PART_SHA256
        or tuple(item.get("size_bytes") for item in parts if isinstance(item, Mapping))
        != (26_843_545_600, 13_549_409_397)
    ):
        raise ExportError("native source pin run_419 bytes differ")
    return document, case


def _manifest_record(
    manifest: PredictionChunkManifest,
    *,
    expected_sha256: str,
    expected_support: str,
    expected_count: int,
    expected_chunks: int,
) -> dict[str, object]:
    if (
        manifest.sha256 != expected_sha256
        or manifest.case_id != CASE_ID
        or manifest.support_id != expected_support
        or manifest.total_row_count != expected_count
        or len(manifest.chunks) != expected_chunks
    ):
        raise ExportError(f"{expected_support} prediction manifest identity differs")
    cursor = 0
    chunks: list[dict[str, object]] = []
    for position, descriptor in enumerate(manifest.chunks):
        if (
            descriptor.chunk_index != position
            or descriptor.raw_cell_id_start != cursor
            or descriptor.raw_cell_id_stop
            != descriptor.raw_cell_id_start + descriptor.row_count
        ):
            raise ExportError(
                f"{expected_support} prediction chunks are not ordered gap-free intervals"
            )
        cursor = descriptor.raw_cell_id_stop
        chunks.append(
            {
                "chunk_index": descriptor.chunk_index,
                "file": descriptor.relative_file.as_posix(),
                "sha256": descriptor.sha256,
                "row_count": descriptor.row_count,
                "raw_cell_id_start": descriptor.raw_cell_id_start,
                "raw_cell_id_stop": descriptor.raw_cell_id_stop,
            }
        )
    if cursor != expected_count:
        raise ExportError(f"{expected_support} prediction coverage does not end at N")
    return {
        "manifest_file": manifest.path.name,
        "manifest_sha256": manifest.sha256,
        "support_id": expected_support,
        "field_components": dict(manifest.field_components),
        "entity_count": expected_count,
        "chunk_count": expected_chunks,
        "chunks": chunks,
        "coverage": {
            "raw_cell_id_start": 0,
            "raw_cell_id_stop": expected_count,
            "complete_gap_free_duplicate_free": True,
        },
    }


def _verify_current_case_evidence(
    path: Path,
    implementation_receipt_path: Path,
    *,
    evaluator_commit: str,
    implementation: Sequence[Mapping[str, object]],
    native_source_pin_path: Path,
    surface: Mapping[str, object],
    volume: Mapping[str, object],
) -> dict[str, object]:
    evidence_path = _regular(path, "current full-case evaluation evidence")
    evidence, evidence_sha, evidence_size = _load_json(
        evidence_path, "current full-case evaluation evidence"
    )

    # Reuse the repository's strict dataset-side validator rather than keeping
    # a second, shallower interpretation of the v2 core-evidence schema here.
    # The validator closes every source/native-array/area audit, all sufficient
    # statistics and additive sums, force closure, prediction declarations,
    # complete coverage, and the four execution settings.
    native_pin = load_native_source_pin(native_source_pin_path)
    contract = SimpleNamespace(
        native_source_pin_sha256=NATIVE_SOURCE_PIN_SHA256,
        native_source_pin=native_pin,
        force_constants={
            "freestream_velocity_m_per_s": U_INF_M_PER_S,
            "density_kg_per_m3": RHO_INF_KG_PER_M3,
            "reference_area_m2": REFERENCE_AREA_M2,
            "reference_length_m": REFERENCE_LENGTH_M,
        },
    )
    validated = _validate_core_case(
        evidence,
        evidence_path,
        evidence_sha,
        contract=contract,  # type: ignore[arg-type]
        pinned=native_pin.case(CASE_ID),
    )
    if (
        validated.surface_count != SURFACE_ENTITY_COUNT
        or validated.volume_count != VOLUME_ENTITY_COUNT
    ):
        raise ExportError("current full-case native entity counts differ")

    source = _exact_mapping(
        evidence["source"],
        {
            "native_source_pin_sha256",
            "repository_id",
            "repository_revision",
            "boundary_sha256",
            "surface_native",
            "surface_area",
            "volume_part_sha256",
            "volume_vtk",
            "volume_weighting",
            "volume_native_arrays",
        },
        "current full-case source",
    )
    prediction = _exact_mapping(
        evidence["prediction_inputs"],
        {"surface_native_cells", "volume_native_cells"},
        "current full-case prediction inputs",
    )
    coverage = _exact_mapping(
        evidence["coverage"],
        {"surface", "volume"},
        "current full-case coverage",
    )
    for support_id, expected in (
        ("surface_native_cells", surface),
        ("volume_native_cells", volume),
    ):
        binding = prediction.get(support_id)
        span = coverage.get("surface" if support_id.startswith("surface") else "volume")
        if (
            not isinstance(binding, Mapping)
            or binding.get("manifest_sha256") != expected["manifest_sha256"]
            or binding.get("chunk_sha256")
            != [row["sha256"] for row in expected["chunks"]]  # type: ignore[index]
            or binding.get("chunk_count") != expected["chunk_count"]
            or binding.get("entity_count") != expected["entity_count"]
            or not isinstance(span, Mapping)
            or span.get("raw_cell_id_start") != 0
            or span.get("raw_cell_id_stop") != expected["entity_count"]
            or span.get("complete_gap_free_duplicate_free") is not True
        ):
            raise ExportError(
                f"current full-case evidence does not bind {support_id} exactly"
            )

    receipt, receipt_sha, receipt_size = _load_json(
        implementation_receipt_path,
        "current full-case implementation receipt",
    )
    receipt = _exact_mapping(
        receipt,
        {
            "schema",
            "schema_version",
            "status",
            "official_submission",
            "case_id",
            "evaluator",
            "evidence",
            "runtime",
        },
        "current full-case implementation receipt",
    )
    receipt_evaluator = _exact_mapping(
        receipt["evaluator"],
        {
            "repository",
            "git_revision",
            "implementation",
            "preflight_verified",
            "postflight_verified",
        },
        "current full-case implementation receipt evaluator",
    )
    receipt_evidence = _exact_mapping(
        receipt["evidence"],
        {"file", "sha256", "size_bytes", "schema", "schema_version"},
        "current full-case implementation receipt evidence",
    )
    receipt_runtime = _exact_mapping(
        receipt["runtime"],
        {"python", "numpy", "vtk"},
        "current full-case implementation receipt runtime",
    )
    implementation_by_path = {
        row.get("path"): dict(row) for row in implementation
    }
    expected_core_implementation = [
        implementation_by_path.get(relative_path)
        for relative_path in CORE_EVALUATOR_GIT_PATHS
    ]
    surface_native = _exact_mapping(
        source["surface_native"],
        {
            "source_file",
            "boundary_sha256",
            "vtk_version",
            "point_count",
            "polygon_count",
            "raw_cell_order",
            "association",
            "arrays",
            "available_point_arrays",
            "available_cell_arrays",
        },
        "current full-case native surface audit",
    )
    expected_evidence_binding = {
        "file": evidence_path.name,
        "sha256": evidence_sha,
        "size_bytes": evidence_size,
        "schema": evidence["schema"],
        "schema_version": evidence["schema_version"],
    }
    if (
        receipt["schema"] != IMPLEMENTATION_RECEIPT_SCHEMA
        or receipt["schema_version"] != 1
        or receipt["status"] != IMPLEMENTATION_RECEIPT_STATUS
        or receipt["official_submission"] is not False
        or receipt["case_id"] != CASE_ID
        or receipt_evaluator["repository"]
        != "neilashton/fluidsbench-submission"
        or receipt_evaluator["git_revision"] != evaluator_commit
        or receipt_evaluator["preflight_verified"] is not True
        or receipt_evaluator["postflight_verified"] is not True
        or any(row is None for row in expected_core_implementation)
        or receipt_evaluator["implementation"] != expected_core_implementation
        or receipt_evidence != expected_evidence_binding
        or receipt_runtime["python"] != platform.python_version()
        or receipt_runtime["numpy"] != np.__version__
        or receipt_runtime["vtk"] != surface_native["vtk_version"]
    ):
        raise ExportError(
            "current full-case implementation receipt does not bind the exact "
            "evaluator, evidence, and runtime"
        )

    surface_area = _exact_mapping(
        source["surface_area"],
        {
            "source_path",
            "sha256",
            "source_boundary_sha256",
            "entity_count",
            "area_sum_m2",
            "area_min_m2",
            "area_max_m2",
            "dtype",
            "role",
            "native_geometry_order_audit",
        },
        "current full-case surface-area audit",
    )
    execution = _exact_mapping(
        evidence["execution"],
        {
            "maximum_prediction_chunk_rows",
            "hash_chunk_bytes",
            "validation_block_rows",
            "encoded_chunk_bytes",
        },
        "current full-case execution",
    )
    return {
        "file": evidence_path.name,
        "sha256": evidence_sha,
        "size_bytes": evidence_size,
        "schema": evidence["schema"],
        "schema_version": evidence["schema_version"],
        "surface_area_sha256": surface_area["sha256"],
        "execution": dict(execution),
        "implementation_receipt": {
            "file": Path(implementation_receipt_path).name,
            "sha256": receipt_sha,
            "size_bytes": receipt_size,
            "schema": receipt["schema"],
            "git_revision": evaluator_commit,
            "runtime": dict(receipt_runtime),
        },
        "complete_schema_native_audits_and_execution_validated": True,
        "complete_surface_and_volume_replay": True,
    }


def _verify_model_inputs(args: argparse.Namespace) -> dict[str, object]:
    provenance, provenance_sha, provenance_size = _load_json(
        args.checkpoint_provenance,
        "checkpoint provenance",
        CHECKPOINT_PROVENANCE_SHA256,
    )
    source = provenance.get("source")
    files = provenance.get("files")
    if (
        provenance.get("artifact_kind") != "drivaerml_transolver_checkpoint_pair"
        or not isinstance(source, Mapping)
        or source.get("repository") != MODEL_REPOSITORY
        or source.get("immutable_revision") != MODEL_REVISION
        or not isinstance(files, list)
        or len(files) != 4
    ):
        raise ExportError("checkpoint provenance does not bind the expected model revision")
    expected_files = {
        "surface_checkpoint": (SURFACE_CHECKPOINT_SHA256, SURFACE_CHECKPOINT_BYTES),
        "surface_normalization_stats": (SURFACE_STATS_SHA256, 1_375),
        "volume_checkpoint": (VOLUME_CHECKPOINT_SHA256, VOLUME_CHECKPOINT_BYTES),
        "volume_normalization_stats": (VOLUME_STATS_SHA256, 1_532),
    }
    observed = {
        row.get("role"): (row.get("sha256"), row.get("size_bytes"))
        for row in files
        if isinstance(row, Mapping)
    }
    if observed != expected_files:
        raise ExportError("checkpoint provenance file inventory differs")
    local = (
        (args.surface_checkpoint, "surface checkpoint", *expected_files["surface_checkpoint"]),
        (args.surface_stats, "surface stats", *expected_files["surface_normalization_stats"]),
        (args.volume_checkpoint, "volume checkpoint", *expected_files["volume_checkpoint"]),
        (args.volume_stats, "volume stats", *expected_files["volume_normalization_stats"]),
    )
    for path, label, expected_sha, expected_size in local:
        payload = _read_bytes(path, label, expected_sha)
        if len(payload) != expected_size:
            raise ExportError(f"{label} byte size differs")

    surface_execution, surface_execution_sha, surface_execution_size = _load_json(
        args.surface_execution_receipt,
        "surface execution receipt",
        SURFACE_EXECUTION_RECEIPT_SHA256,
    )
    surface_conversion, surface_conversion_sha, surface_conversion_size = _load_json(
        args.surface_conversion_receipt,
        "surface conversion receipt",
        SURFACE_CONVERSION_RECEIPT_SHA256,
    )
    volume_inference, volume_inference_sha, volume_inference_size = _load_json(
        args.volume_inference_receipt,
        "volume inference receipt",
        VOLUME_INFERENCE_RECEIPT_SHA256,
    )
    surface_model = surface_execution.get("model")
    surface_conversion_manifest = surface_conversion.get("manifest")
    volume_model = volume_inference.get("model")
    volume_prediction = volume_inference.get("prediction")
    if (
        surface_execution.get("case_id") != CASE_ID
        or not isinstance(surface_model, Mapping)
        or surface_model.get("checkpoint_sha256") != SURFACE_CHECKPOINT_SHA256
        or surface_model.get("global_stats_sha256") != SURFACE_STATS_SHA256
        or surface_model.get("official_checkpoint_provenance_sha256")
        != CHECKPOINT_PROVENANCE_SHA256
        or surface_conversion.get("case_id") != CASE_ID
        or surface_conversion.get("status")
        != "local_candidate_evaluator_input_not_official_submission"
        or not isinstance(surface_conversion_manifest, Mapping)
        or surface_conversion_manifest.get("sha256") != SURFACE_MANIFEST_SHA256
        or surface_conversion_manifest.get("chunk_count") != SURFACE_CHUNK_COUNT
        or volume_inference.get("case_id") != CASE_ID
        or volume_inference.get("status")
        != "complete_candidate_evaluator_input_not_official_submission"
        or not isinstance(volume_model, Mapping)
        or volume_model.get("checkpoint_sha256") != VOLUME_CHECKPOINT_SHA256
        or volume_model.get("stats_sha256") != VOLUME_STATS_SHA256
        or not isinstance(volume_prediction, Mapping)
        or volume_prediction.get("manifest_sha256") != VOLUME_MANIFEST_SHA256
        or volume_prediction.get("chunk_count") != VOLUME_CHUNK_COUNT
        or volume_prediction.get("entity_count") != VOLUME_ENTITY_COUNT
        or volume_prediction.get("complete_gap_free_duplicate_free") is not True
    ):
        raise ExportError("inference/conversion receipts do not bind the model inputs")
    return {
        "source": {
            "provider": "Hugging Face",
            "repository": MODEL_REPOSITORY,
            "immutable_revision": MODEL_REVISION,
        },
        "checkpoint_provenance": {
            "file": args.checkpoint_provenance.name,
            "sha256": provenance_sha,
            "size_bytes": provenance_size,
        },
        "surface": {
            "checkpoint_sha256": SURFACE_CHECKPOINT_SHA256,
            "checkpoint_size_bytes": SURFACE_CHECKPOINT_BYTES,
            "stats_sha256": SURFACE_STATS_SHA256,
            "execution_receipt": {
                "file": args.surface_execution_receipt.name,
                "sha256": surface_execution_sha,
                "size_bytes": surface_execution_size,
            },
            "conversion_receipt": {
                "file": args.surface_conversion_receipt.name,
                "sha256": surface_conversion_sha,
                "size_bytes": surface_conversion_size,
            },
        },
        "volume": {
            "checkpoint_sha256": VOLUME_CHECKPOINT_SHA256,
            "checkpoint_size_bytes": VOLUME_CHECKPOINT_BYTES,
            "stats_sha256": VOLUME_STATS_SHA256,
            "inference_receipt": {
                "file": args.volume_inference_receipt.name,
                "sha256": volume_inference_sha,
                "size_bytes": volume_inference_size,
            },
        },
        "checkpoint_pair_count": 1,
        "genuine_three_model_sensitivity_gate_satisfied": False,
    }


def _verify_retained_manifests(
    producer_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, object]]:
    definitions = {
        "velocity_placement": (
            "velocity_support_v3/production_campaign_v1/aggregate/"
            "relative-velocity-v3-production-all484-inputs-v1.json",
            "benchmark-specs/drivaerml/support/relative-v3/manifests/"
            "velocity-placement-all484-v1.json",
            VELOCITY_PLACEMENT_MANIFEST_SHA256,
            "drivaerml-relative-velocity-v3-production-input-manifest-v1",
            1,
        ),
        "velocity_mapping": (
            "velocity_mapping_v3/aggregate/"
            "relative-velocity-v3-mapping-all484-v1.json",
            "benchmark-specs/drivaerml/support/relative-v3/manifests/"
            "velocity-mapping-all484-v1.json",
            VELOCITY_MAPPING_MANIFEST_SHA256,
            "drivaerml-velocity-relative-v3-mapping-aggregate-v1",
            1,
        ),
        "relative_cp": (
            "cp_support/campaign_v3/aggregate_v3/all484/"
            "relative-cp-native-support-manifest-v3.json",
            "benchmark-specs/drivaerml/support/relative-v3/manifests/"
            "cp-native-support-all484-v3.json",
            CP_SUPPORT_MANIFEST_SHA256,
            "drivaerml-relative-cp-native-support-manifest-v3",
            3,
        ),
    }
    documents: dict[str, dict[str, Any]] = {}
    records: dict[str, object] = {}
    for role, (producer_relative, retained_relative, expected_sha, schema, version) in (
        definitions.items()
    ):
        producer_bytes = _read_bytes(
            producer_root / producer_relative,
            f"{role} producer manifest",
            expected_sha,
        )
        retained_bytes = _read_bytes(
            ROOT / retained_relative,
            f"{role} retained manifest",
            expected_sha,
        )
        if producer_bytes != retained_bytes:
            raise ExportError(f"{role} producer and retained manifest bytes differ")
        document = _strict_json_bytes(retained_bytes, f"{role} retained manifest")
        if (
            document.get("schema") != schema
            or document.get("schema_version") != version
            or document.get("public_dataset_revision") != DATASET_REVISION
            or len(document.get("cases", [])) != 484
        ):
            raise ExportError(f"{role} manifest schema/revision/coverage differs")
        _one_case(document, f"{role} retained manifest")
        documents[role] = document
        records[role] = {
            "repository_path": retained_relative,
            "producer_path": producer_relative,
            "sha256": expected_sha,
            "size_bytes": len(retained_bytes),
            "schema": schema,
            "schema_version": version,
            "case_count": 484,
            "producer_and_retained_bytes_equal": True,
        }
    return documents, records


def _verify_constant_velocity(
    case_dir: Path,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    mapping_path = case_dir / "velocity-cell-mapping-10mm.json"
    receipt_path = case_dir / "receipt.json"
    mapping, _, mapping_size = _load_json(
        mapping_path, "constant velocity 10 mm mapping", CONSTANT_VELOCITY_MAPPING_SHA256
    )
    receipt, _, receipt_size = _load_json(
        receipt_path, "constant velocity receipt", CONSTANT_VELOCITY_RECEIPT_SHA256
    )
    native = mapping.get("native_source_binding")
    resolution = mapping.get("resolution")
    coverage = mapping.get("coverage")
    rows = mapping.get("rows")
    summaries = receipt.get("artifacts")
    ten_mm = [
        row
        for row in summaries
        if isinstance(row, Mapping) and row.get("nominal_spacing_mm") == 10
    ] if isinstance(summaries, list) else []
    if (
        mapping.get("schema") != "drivaerml-velocity-cell-mapping-candidate-v1"
        or mapping.get("schema_version") != 1
        or mapping.get("case_id") != CASE_ID
        or mapping.get("status")
        != "candidate_complete_geometry_mapping_not_activation_evidence"
        or mapping.get("row_fields")
        != [
            "profile_id", "sample_index", "point_m", "distance_m",
            "valid", "reason", "raw_vtk_cell_id", "candidate_count",
        ]
        or not isinstance(native, Mapping)
        or native.get("pin_sha256") != NATIVE_SOURCE_PIN_SHA256
        or native.get("repository_id") != DATASET_REPOSITORY
        or native.get("repository_revision") != DATASET_REVISION
        or not isinstance(resolution, Mapping)
        or resolution.get("sample_count") != 3_756
        or resolution.get("line_count") != 16
        or resolution.get("nominal_spacing_mm") != 10
        or not isinstance(coverage, Mapping)
        or coverage.get("complete_duplicate_free_no_omissions") is not True
        or coverage.get("valid_count") != 3_684
        or coverage.get("invalid_count") != 72
        or not isinstance(rows, list)
        or len(rows) != 3_756
        or receipt.get("schema")
        != "drivaerml-velocity-cell-assignments-case-candidate-v2"
        or receipt.get("schema_version") != 2
        or receipt.get("case_id") != CASE_ID
        or len(ten_mm) != 1
        or ten_mm[0].get("sha256") != CONSTANT_VELOCITY_MAPPING_SHA256
        or ten_mm[0].get("valid_count") != 3_684
        or ten_mm[0].get("invalid_count") != 72
    ):
        raise ExportError("constant velocity support declaration differs")
    by_station: dict[str, list[dict[str, object]]] = defaultdict(list)
    invalid_reasons: Counter[str] = Counter()
    for position, raw in enumerate(rows):
        if not isinstance(raw, list) or len(raw) != 8:
            raise ExportError(f"constant velocity row {position} shape differs")
        station, sample_index, point_m, distance, valid, reason, raw_id, candidates = raw
        if station not in VELOCITY_SAMPLE_COUNTS:
            raise ExportError(f"constant velocity row {position} station differs")
        expected_index = len(by_station[station])
        if (
            sample_index != expected_index
            or not isinstance(point_m, list)
            or len(point_m) != 3
            or any(not math.isfinite(_finite(value, "constant point")) for value in point_m)
            or not isinstance(valid, bool)
            or isinstance(candidates, bool)
            or not isinstance(candidates, int)
            or candidates < 0
        ):
            raise ExportError(f"constant velocity row {position} is malformed")
        row = {
            "sample_index": sample_index,
            "distance_m": _finite(distance, f"constant velocity row {position} distance"),
            "valid": valid,
            "raw_vtk_cell_id": raw_id,
        }
        if valid:
            if reason != "" or not isinstance(raw_id, int) or isinstance(raw_id, bool):
                raise ExportError(f"constant velocity row {position} valid binding differs")
            if raw_id < 0 or raw_id >= VOLUME_ENTITY_COUNT or candidates < 1:
                raise ExportError(f"constant velocity row {position} raw ID differs")
        else:
            if not isinstance(reason, str) or not reason or raw_id is not None:
                raise ExportError(f"constant velocity row {position} invalid binding differs")
            invalid_reasons[reason] += 1
        by_station[station].append(row)
    if (
        tuple(by_station) != VELOCITY_STATIONS
        or {station: len(rows_) for station, rows_ in by_station.items()}
        != VELOCITY_SAMPLE_COUNTS
        or dict(invalid_reasons) != coverage.get("invalid_reason_counts")
    ):
        raise ExportError("constant velocity station coverage/order differs")
    support_identity: dict[str, str] = {}
    for station, station_rows in by_station.items():
        valid_rows = [row for row in station_rows if row["valid"]]
        distances = [float(row["distance_m"]) for row in valid_rows]
        if len(distances) < 2 or any(
            right <= left for left, right in zip(distances, distances[1:])
        ):
            raise ExportError(f"constant velocity {station} valid coordinate subset differs")
        support_identity[station] = _sha256_bytes(
            _canonical_bytes(
                {
                    "schema": "drivaerml-run419-constant-velocity-support-identity-v1",
                    "case_id": CASE_ID,
                    "family_id": "drivaerml-autocfd5-constant-v1",
                    "station_id": f"autocfd5_{station.lower()}",
                    "mapping_sha256": CONSTANT_VELOCITY_MAPPING_SHA256,
                    "ordered_rows": station_rows,
                }
            )
        )
    return dict(by_station), {
        "mapping": {
            "file": mapping_path.name,
            "sha256": CONSTANT_VELOCITY_MAPPING_SHA256,
            "size_bytes": mapping_size,
        },
        "receipt": {
            "file": receipt_path.name,
            "sha256": CONSTANT_VELOCITY_RECEIPT_SHA256,
            "size_bytes": receipt_size,
        },
        "profile_registry_sha256": CONSTANT_VELOCITY_PROFILE_SHA256,
        "sample_count": 3_756,
        "valid_count": 3_684,
        "invalid_count": 72,
        "per_station_materialized_support_identity_sha256": support_identity,
        "support_identity_encoding": (
            "UTF-8 sorted-key compact JSON plus LF over the exact receipt-bound "
            "ordered station mapping rows"
        ),
    }


def _read_relative_coordinate_csv(path: Path) -> dict[str, list[dict[str, object]]]:
    source = _regular(path, "relative velocity coordinate CSV")
    if _sha256_bytes(source.read_bytes()) != RELATIVE_VELOCITY_COORDINATE_CSV_SHA256:
        raise ExportError("relative velocity coordinate CSV SHA-256 differs")
    fields = [
        "family_id", "placement_mode", "profile_id", "sample_index", "point_count",
        "line_fraction", "distance_m", "x_m", "y_m", "z_m",
    ]
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ExportError("relative velocity coordinate CSV columns differ")
        for position, raw in enumerate(reader):
            station = raw["profile_id"]
            if (
                raw["family_id"] != "drivaerml-velocity-relative-v3"
                or raw["placement_mode"] != "relative"
                or station not in VELOCITY_SAMPLE_COUNTS
                or int(raw["sample_index"]) != len(result[station])
                or int(raw["point_count"]) != VELOCITY_SAMPLE_COUNTS[station]
            ):
                raise ExportError(f"relative velocity coordinate row {position} differs")
            result[station].append(
                {
                    "sample_index": int(raw["sample_index"]),
                    "line_fraction": _finite(float(raw["line_fraction"]), "line fraction"),
                    "distance_m": _finite(float(raw["distance_m"]), "relative distance"),
                    "point_m": [
                        _finite(float(raw[field]), f"relative point {field}")
                        for field in ("x_m", "y_m", "z_m")
                    ],
                }
            )
    if (
        tuple(result) != VELOCITY_STATIONS
        or {station: len(rows) for station, rows in result.items()}
        != VELOCITY_SAMPLE_COUNTS
    ):
        raise ExportError("relative velocity coordinate CSV coverage differs")
    return dict(result)


def _verify_relative_velocity(
    producer_root: Path,
    manifests: Mapping[str, dict[str, Any]],
    retained: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    placement_case = _one_case(manifests["velocity_placement"], "velocity placement")
    mapping_case = _one_case(manifests["velocity_mapping"], "velocity mapping")
    placement_root = (
        producer_root / "velocity_support_v3/production_campaign_v1/cases" / CASE_ID
    )
    mapping_root = producer_root / "velocity_mapping_v3/cases" / CASE_ID
    csv_path = placement_root / f"{CASE_ID}-relative-v3-velocity-10mm-coordinates.csv"
    placement_receipt_path = placement_root / f"{CASE_ID}-relative-v3-velocity-receipt.json"
    mapping_path = mapping_root / "velocity-relative-v3-cell-mapping-10mm.json"
    mapping_receipt_path = mapping_root / "receipt.json"
    coordinates = _read_relative_coordinate_csv(csv_path)
    placement_receipt, _, placement_size = _load_json(
        placement_receipt_path,
        "relative velocity placement receipt",
        RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256,
    )
    mapping, _, mapping_size = _load_json(
        mapping_path,
        "relative velocity 10 mm mapping",
        RELATIVE_VELOCITY_MAPPING_SHA256,
    )
    mapping_receipt, _, mapping_receipt_size = _load_json(
        mapping_receipt_path,
        "relative velocity mapping receipt",
        RELATIVE_VELOCITY_MAPPING_RECEIPT_SHA256,
    )
    ten_mm = [
        row
        for row in mapping_case.get("artifacts", [])
        if isinstance(row, Mapping) and row.get("nominal_spacing_mm") == 10
    ]
    receipt_ten_mm = [
        row
        for row in mapping_receipt.get("artifacts", [])
        if isinstance(row, Mapping) and row.get("nominal_spacing_mm") == 10
    ]
    resolution = mapping.get("resolution")
    coverage = mapping.get("coverage")
    relative_binding = mapping.get("relative_placement_support")
    rows = mapping.get("rows")
    if (
        placement_case.get("coordinates_sha256")
        != RELATIVE_VELOCITY_COORDINATE_CSV_SHA256
        or placement_case.get("receipt_sha256")
        != RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256
        or placement_receipt.get("schema")
        != "drivaerml-relative-velocity-v3-case-receipt-v1"
        or placement_receipt.get("case_id") != CASE_ID
        or placement_receipt.get("family_id") != "drivaerml-velocity-relative-v3"
        or placement_receipt.get("placement_mode") != "relative"
        or placement_receipt.get("public_dataset", {}).get("revision") != DATASET_REVISION
        or len(ten_mm) != 1
        or ten_mm[0].get("sha256") != RELATIVE_VELOCITY_MAPPING_SHA256
        or mapping_case.get("coordinate_csv_sha256")
        != RELATIVE_VELOCITY_COORDINATE_CSV_SHA256
        or mapping_case.get("placement_receipt_sha256")
        != RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256
        or mapping_case.get("receipt_sha256")
        != RELATIVE_VELOCITY_MAPPING_RECEIPT_SHA256
        or mapping.get("schema") != "drivaerml-velocity-relative-v3-cell-mapping-v1"
        or mapping.get("schema_version") != 1
        or mapping.get("case_id") != CASE_ID
        or mapping.get("family_id") != "drivaerml-velocity-relative-v3"
        or mapping.get("placement_mode") != "relative"
        or not isinstance(relative_binding, Mapping)
        or relative_binding.get("coordinate_csv_sha256")
        != RELATIVE_VELOCITY_COORDINATE_CSV_SHA256
        or relative_binding.get("placement_receipt_sha256")
        != RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256
        or not isinstance(resolution, Mapping)
        or resolution.get("sample_count") != 3_756
        or resolution.get("per_profile_counts") != VELOCITY_SAMPLE_COUNTS
        or not isinstance(coverage, Mapping)
        or coverage.get("complete_duplicate_free_no_omissions") is not True
        or coverage.get("valid_count") != 3_678
        or coverage.get("invalid_count") != 78
        or not isinstance(rows, list)
        or len(rows) != 3_756
        or mapping_receipt.get("schema")
        != "drivaerml-velocity-relative-v3-cell-mapping-case-v1"
        or mapping_receipt.get("public_dataset_revision") != DATASET_REVISION
        or len(receipt_ten_mm) != 1
        or receipt_ten_mm[0].get("sha256") != RELATIVE_VELOCITY_MAPPING_SHA256
    ):
        raise ExportError("relative velocity support declaration differs")
    expected_order = [
        (station, row)
        for station in VELOCITY_STATIONS
        for row in coordinates[station]
    ]
    materialized: dict[str, list[dict[str, object]]] = defaultdict(list)
    invalid_reasons: Counter[str] = Counter()
    for position, (raw, (expected_station, coordinate)) in enumerate(
        zip(rows, expected_order, strict=True)
    ):
        if not isinstance(raw, list) or len(raw) != 8:
            raise ExportError(f"relative velocity row {position} shape differs")
        station, sample_index, point_m, distance, valid, reason, raw_id, candidates = raw
        if (
            station != expected_station
            or sample_index != coordinate["sample_index"]
            or point_m != coordinate["point_m"]
            or distance != coordinate["distance_m"]
            or not isinstance(valid, bool)
            or isinstance(candidates, bool)
            or not isinstance(candidates, int)
            or candidates < 0
        ):
            raise ExportError(f"relative velocity row {position} differs from placement")
        if valid:
            if (
                reason != ""
                or isinstance(raw_id, bool)
                or not isinstance(raw_id, int)
                or raw_id < 0
                or raw_id >= VOLUME_ENTITY_COUNT
                or candidates < 1
            ):
                raise ExportError(f"relative velocity row {position} valid binding differs")
            materialized[station].append(
                {
                    "sample_index": sample_index,
                    "coordinate": coordinate["line_fraction"],
                    "raw_vtk_cell_id": raw_id,
                }
            )
        else:
            if not isinstance(reason, str) or not reason or raw_id is not None:
                raise ExportError(f"relative velocity row {position} invalid binding differs")
            invalid_reasons[reason] += 1
    if (
        tuple(materialized) != VELOCITY_STATIONS
        or sum(len(rows_) for rows_ in materialized.values()) != 3_678
        or dict(invalid_reasons) != coverage.get("invalid_reason_counts")
    ):
        raise ExportError("relative velocity valid/invalid coverage differs")
    profile_receipts = placement_receipt.get("profiles")
    if not isinstance(profile_receipts, list) or len(profile_receipts) != 16:
        raise ExportError("relative velocity profile receipts differ")
    receipt_by_station = {
        row.get("profile_id"): row for row in profile_receipts if isinstance(row, Mapping)
    }
    for station in VELOCITY_STATIONS:
        identity = retained.get((CASE_ID, "drivaerml-velocity-relative-v3", station))
        profile_receipt = receipt_by_station.get(station)
        emitted_coordinates = [row["coordinate"] for row in materialized[station]]
        try:
            coordinate_identity = coordinate_array_identity_sha256(emitted_coordinates)
        except CoordinateIdentityError as error:
            raise ExportError(f"relative velocity {station} coordinate encoding failed") from error
        if (
            identity is None
            or identity.get("representation") != "materialized"
            or identity.get("placement_receipt_identity_sha256")
            != RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256
            or not isinstance(profile_receipt, Mapping)
            or identity.get("support_identity_sha256")
            != profile_receipt.get("coordinates_binary64_be_sha256")
            or identity.get("coordinate_count") != len(emitted_coordinates)
            or identity.get("coordinate_identity_sha256") != coordinate_identity
        ):
            raise ExportError(f"relative velocity {station} retained identity differs")
    return dict(materialized), {
        "coordinate_csv": {
            "file": csv_path.name,
            "sha256": RELATIVE_VELOCITY_COORDINATE_CSV_SHA256,
            "size_bytes": csv_path.stat().st_size,
        },
        "placement_receipt": {
            "file": placement_receipt_path.name,
            "sha256": RELATIVE_VELOCITY_PLACEMENT_RECEIPT_SHA256,
            "size_bytes": placement_size,
        },
        "mapping": {
            "file": mapping_path.name,
            "sha256": RELATIVE_VELOCITY_MAPPING_SHA256,
            "size_bytes": mapping_size,
        },
        "mapping_receipt": {
            "file": mapping_receipt_path.name,
            "sha256": RELATIVE_VELOCITY_MAPPING_RECEIPT_SHA256,
            "size_bytes": mapping_receipt_size,
        },
        "sample_count": 3_756,
        "valid_count": 3_678,
        "invalid_count": 78,
        "coordinates_selected_from_verified_mapping": True,
    }


def _verify_cp_supports(
    constant_root: Path,
    relative_root: Path,
    manifests: Mapping[str, dict[str, Any]],
    retained: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    dict[str, object],
]:
    cuts, constant_record = _verify_constant_cp_support(constant_root)

    relative_case_root = (
        relative_root / "cp_support/campaign_v3/native_support_v3/cases" / CASE_ID
    )
    relative_json_path = relative_case_root / "relative-case-support.json"
    relative_csv_path = relative_case_root / "relative-case-support.csv"
    relative_receipt_path = relative_case_root / "receipt.json"
    relative_placement_path = relative_case_root / "placement-receipt.json"
    relative_envelope, _, relative_json_size = _load_json(
        relative_json_path, "relative Cp case support", RELATIVE_CP_JSON_SHA256
    )
    relative_csv = _read_bytes(
        relative_csv_path, "relative Cp case support CSV", RELATIVE_CP_CSV_SHA256
    )
    relative_receipt, _, relative_receipt_size = _load_json(
        relative_receipt_path,
        "relative Cp artifact receipt",
        RELATIVE_CP_ARTIFACT_RECEIPT_SHA256,
    )
    relative_placement, _, relative_placement_size = _load_json(
        relative_placement_path,
        "relative Cp placement receipt",
        RELATIVE_CP_PLACEMENT_FILE_SHA256,
    )
    relative_manifest_case = _one_case(manifests["relative_cp"], "relative Cp manifest")
    relative_support = relative_envelope.get("support")
    relative_identities = relative_envelope.get("identities")
    aliases = (
        relative_support.get("centerline_aliases")
        if isinstance(relative_support, Mapping)
        else None
    )
    moving = relative_support.get("moving_cuts") if isinstance(relative_support, Mapping) else None
    artifacts = relative_receipt.get("artifacts")
    placement_body = dict(relative_placement)
    placement_identity = placement_body.pop("receipt_identity", None)
    if (
        relative_envelope.get("envelope_schema")
        != "drivaerml-relative-cp-case-support-envelope-v1"
        or not isinstance(relative_identities, Mapping)
        or relative_identities.get("case_support_sha256")
        != RELATIVE_CP_CASE_SUPPORT_SHA256
        or relative_identities.get("moving_rows_csv_sha256") != RELATIVE_CP_CSV_SHA256
        or not isinstance(relative_support, Mapping)
        or relative_support.get("schema") != "drivaerml-relative-cp-case-support-v1"
        or relative_support.get("case_id") != CASE_ID
        or relative_support.get("freestream_velocity_m_per_s") != U_INF_M_PER_S
        or _sha256_bytes(_canonical_bytes(relative_support))
        != RELATIVE_CP_CASE_SUPPORT_SHA256
        or not isinstance(aliases, list)
        or not isinstance(moving, list)
        or len(aliases) != 2
        or len(moving) != 2
        or [item.get("station_id") for item in aliases if isinstance(item, Mapping)]
        != list(RELATIVE_CP_ALIAS_STATIONS)
        or [item.get("station_id") for item in moving if isinstance(item, Mapping)]
        != list(RELATIVE_CP_MOVING_STATIONS)
        or relative_manifest_case.get("relative_case_support_sha256")
        != RELATIVE_CP_CASE_SUPPORT_SHA256
        or relative_manifest_case.get("relative_case_receipt_sha256")
        != RELATIVE_CP_ARTIFACT_RECEIPT_SHA256
        or relative_manifest_case.get("constant_case_support_sha256")
        != CONSTANT_CP_CASE_SUPPORT_SHA256
        or not isinstance(artifacts, Mapping)
        or artifacts.get("relative-case-support.json", {}).get("sha256")
        != RELATIVE_CP_JSON_SHA256
        or artifacts.get("relative-case-support.csv", {}).get("sha256")
        != RELATIVE_CP_CSV_SHA256
        or artifacts.get("placement-receipt.json", {}).get("sha256")
        != RELATIVE_CP_PLACEMENT_FILE_SHA256
        or not isinstance(placement_identity, Mapping)
        or placement_identity.get("sha256")
        != _sha256_bytes(_canonical_bytes(placement_body))
        or relative_placement.get("case_id") != CASE_ID
        or relative_placement.get("family_id") != "drivaerml_cp_relative_v1"
    ):
        raise ExportError("relative Cp support declaration or identity differs")

    constant_by_station = {
        cut["definition"]["cut_id"]: cut for cut in cuts if isinstance(cut, Mapping)
    }
    for alias in aliases:
        station = alias["station_id"]
        expected_support = _sha256_bytes(_canonical_bytes(constant_by_station[station]))
        identity = retained.get((CASE_ID, "drivaerml_cp_relative_v1", station))
        if (
            alias.get("canonical_cut_support_sha256") != expected_support
            or identity is None
            or identity.get("representation") != "shared_alias"
            or identity.get("support_identity_sha256") != expected_support
            or identity.get("placement_receipt_identity_sha256")
            != placement_identity["sha256"]
        ):
            raise ExportError(f"relative Cp alias {station} identity differs")
    for cut in moving:
        station = cut["station_id"]
        cut_body = dict(cut)
        supplied_support = cut_body.pop("support_identity_sha256", None)
        rows = cut.get("rows")
        if (
            supplied_support != _sha256_bytes(_canonical_bytes(cut_body))
            or not isinstance(rows, list)
            or len(rows) < 2
        ):
            raise ExportError(f"relative Cp moving cut {station} identity differs")
        coordinates = [row.get("interval_arc_end_m") for row in rows if isinstance(row, Mapping)]
        try:
            coordinate_identity = coordinate_array_identity_sha256(coordinates)
        except CoordinateIdentityError as error:
            raise ExportError(f"relative Cp {station} coordinate encoding failed") from error
        identity = retained.get((CASE_ID, "drivaerml_cp_relative_v1", station))
        if (
            identity is None
            or identity.get("representation") != "materialized"
            or identity.get("support_identity_sha256") != supplied_support
            or identity.get("placement_receipt_identity_sha256")
            != placement_identity["sha256"]
            or identity.get("coordinate_count") != len(coordinates)
            or identity.get("coordinate_identity_sha256") != coordinate_identity
        ):
            raise ExportError(f"relative Cp moving cut {station} retained identity differs")
    return cuts, aliases, moving, {
        "constant": constant_record,
        "relative": {
            "case_support_json": {
                "file": relative_json_path.name,
                "sha256": RELATIVE_CP_JSON_SHA256,
                "size_bytes": relative_json_size,
            },
            "case_support_csv": {
                "file": relative_csv_path.name,
                "sha256": RELATIVE_CP_CSV_SHA256,
                "size_bytes": len(relative_csv),
            },
            "artifact_receipt": {
                "file": relative_receipt_path.name,
                "sha256": RELATIVE_CP_ARTIFACT_RECEIPT_SHA256,
                "size_bytes": relative_receipt_size,
            },
            "placement_receipt": {
                "file": relative_placement_path.name,
                "sha256": RELATIVE_CP_PLACEMENT_FILE_SHA256,
                "size_bytes": relative_placement_size,
                "receipt_identity_sha256": placement_identity["sha256"],
            },
            "case_support_identity_sha256": RELATIVE_CP_CASE_SUPPORT_SHA256,
        },
    }


def _verify_constant_cp_support(
    constant_root: Path,
) -> tuple[list[Mapping[str, Any]], dict[str, object]]:
    """Verify and return run_419 constant Cp producer support only."""

    constant_case_root = constant_root / "cases" / CASE_ID
    constant_json_path = constant_case_root / "case-support.json"
    constant_csv_path = constant_case_root / "case-support.csv"
    constant_receipt_path = constant_case_root / "receipt.json"
    constant_aggregate_path = (
        constant_root / "aggregate/continuous-cp-cut-all484-aggregate-candidate-v8.json"
    )
    constant_envelope, _, constant_json_size = _load_json(
        constant_json_path, "constant Cp case support", CONSTANT_CP_JSON_SHA256
    )
    constant_csv = _read_bytes(
        constant_csv_path, "constant Cp case support CSV", CONSTANT_CP_CSV_SHA256
    )
    constant_receipt, _, constant_receipt_size = _load_json(
        constant_receipt_path,
        "constant Cp case receipt",
        CONSTANT_CP_RECEIPT_FILE_SHA256,
    )
    constant_aggregate, _, constant_aggregate_size = _load_json(
        constant_aggregate_path,
        "constant Cp all-484 aggregate",
        CONSTANT_CP_AGGREGATE_SHA256,
    )
    constant_support = constant_envelope.get("support")
    constant_identities = constant_envelope.get("identities")
    constant_case = _one_case(constant_aggregate, "constant Cp aggregate")
    cuts = constant_support.get("cuts") if isinstance(constant_support, Mapping) else None
    if (
        constant_envelope.get("envelope_schema")
        != "drivaerml-continuous-cp-cut-case-support-candidate-v2"
        or not isinstance(constant_identities, Mapping)
        or constant_identities.get("case_support_sha256")
        != CONSTANT_CP_CASE_SUPPORT_SHA256
        or constant_identities.get("segment_csv_sha256") != CONSTANT_CP_CSV_SHA256
        or not isinstance(constant_support, Mapping)
        or constant_support.get("case_id") != CASE_ID
        or constant_support.get("freestream_velocity_m_per_s") != U_INF_M_PER_S
        or not isinstance(cuts, list)
        or len(cuts) != 4
        or [cut.get("definition", {}).get("cut_id") for cut in cuts if isinstance(cut, Mapping)]
        != list(CONSTANT_CP_STATIONS)
        or constant_receipt.get("receipt_identity", {}).get("sha256")
        != CONSTANT_CP_RECEIPT_IDENTITY_SHA256
        or constant_receipt.get("case_support", {}).get("json_sha256")
        != CONSTANT_CP_JSON_SHA256
        or constant_receipt.get("case_support", {}).get("csv_sha256")
        != CONSTANT_CP_CSV_SHA256
        or constant_case.get("case_support_sha256") != CONSTANT_CP_CASE_SUPPORT_SHA256
        or constant_case.get("receipt_body_sha256")
        != CONSTANT_CP_RECEIPT_IDENTITY_SHA256
    ):
        raise ExportError("constant Cp support declaration or identity differs")

    return cuts, {
            "aggregate": {
                "file": constant_aggregate_path.name,
                "sha256": CONSTANT_CP_AGGREGATE_SHA256,
                "size_bytes": constant_aggregate_size,
            },
            "case_support_json": {
                "file": constant_json_path.name,
                "sha256": CONSTANT_CP_JSON_SHA256,
                "size_bytes": constant_json_size,
            },
            "case_support_csv": {
                "file": constant_csv_path.name,
                "sha256": CONSTANT_CP_CSV_SHA256,
                "size_bytes": len(constant_csv),
            },
            "receipt": {
                "file": constant_receipt_path.name,
                "sha256": CONSTANT_CP_RECEIPT_FILE_SHA256,
                "size_bytes": constant_receipt_size,
                "receipt_identity_sha256": CONSTANT_CP_RECEIPT_IDENTITY_SHA256,
            },
            "case_support_identity_sha256": CONSTANT_CP_CASE_SUPPORT_SHA256,
    }


def build_run419_constant_series_support_index(
    constant_velocity_case_dir: Path,
    constant_cp_campaign_root: Path,
) -> dict[str, object]:
    """Build the compact constant-series binding from verified producer bytes.

    The constant velocity producer explicitly marks 72 of the frozen-grid
    points as unsupported.  Those rows are retained in the support identity,
    while only the ordered valid rows form the report-only coordinate array.
    Consequently a materialized array may contain interior gaps; it is not a
    new constant-family scoring support.
    """

    velocity, velocity_record = _verify_constant_velocity(
        _directory(constant_velocity_case_dir, "constant velocity case directory")
    )
    cuts, cp_record = _verify_constant_cp_support(
        _directory(constant_cp_campaign_root, "constant Cp campaign root")
    )
    raw_velocity_identities = velocity_record.get(
        "per_station_materialized_support_identity_sha256"
    )
    if not isinstance(raw_velocity_identities, Mapping):
        raise ExportError("constant velocity station identities are malformed")

    series: list[dict[str, object]] = []
    for station in VELOCITY_STATIONS:
        station_rows = velocity[station]
        valid_rows = [row for row in station_rows if row["valid"]]
        coordinates = [row["distance_m"] for row in valid_rows]
        series.append(
            {
                "family_id": "drivaerml-autocfd5-constant-v1",
                "station_id": f"autocfd5_{station.lower()}",
                "representation": "materialized",
                "support_identity_sha256": raw_velocity_identities[station],
                "placement_receipt_identity_sha256": (
                    CONSTANT_VELOCITY_RECEIPT_SHA256
                ),
                "coordinate_count": len(coordinates),
                "coordinate_identity_sha256": (
                    coordinate_array_identity_sha256(coordinates)
                ),
            }
        )
    for cut in cuts:
        definition = cut.get("definition")
        rows = cut.get("segments")
        if not isinstance(definition, Mapping) or not isinstance(rows, list):
            raise ExportError("constant Cp cut structure is malformed")
        station = definition.get("cut_id")
        if station not in CONSTANT_CP_STATIONS or len(rows) < 2:
            raise ExportError("constant Cp cut coverage is malformed")
        coordinates = [
            row.get("arc_length_end_m")
            for row in rows
            if isinstance(row, Mapping)
        ]
        if len(coordinates) != len(rows):
            raise ExportError(f"constant Cp {station} contains a malformed segment")
        series.append(
            {
                "family_id": "drivaerml_cp_constant_v1",
                "station_id": station,
                "representation": "materialized",
                "support_identity_sha256": _sha256_bytes(_canonical_bytes(cut)),
                "placement_receipt_identity_sha256": (
                    CONSTANT_CP_RECEIPT_IDENTITY_SHA256
                ),
                "coordinate_count": len(coordinates),
                "coordinate_identity_sha256": (
                    coordinate_array_identity_sha256(coordinates)
                ),
            }
        )
    if len(series) != 20:
        raise ExportError("run_419 constant support index must contain 20 series")
    return {
        "schema": "drivaerml-run419-constant-series-support-index-v1",
        "schema_version": 1,
        "dataset_id": "drivaerml",
        "contract_id": RELATIVE_PROFILE_CONTRACT_ID,
        "scope": "run_419_report_only_valid_native_support",
        "case_id": CASE_ID,
        "series_count": 20,
        "source_bindings": {
            "public_dataset": {
                "native_source_pin_path": (
                    "benchmark-specs/drivaerml/proposal/native-source-pin.json"
                ),
                "native_source_pin_sha256": NATIVE_SOURCE_PIN_SHA256,
                "repository": DATASET_REPOSITORY,
                "revision": DATASET_REVISION,
            },
            "constant_velocity": {
                "mapping_sha256": CONSTANT_VELOCITY_MAPPING_SHA256,
                "receipt_sha256": CONSTANT_VELOCITY_RECEIPT_SHA256,
                "profile_registry_sha256": CONSTANT_VELOCITY_PROFILE_SHA256,
                "sample_count": velocity_record["sample_count"],
                "valid_count": velocity_record["valid_count"],
                "invalid_count": velocity_record["invalid_count"],
                "coordinate_selection": (
                    "ordered 10mm frozen-grid mapping rows with valid=true; "
                    "unsupported rows are omitted and interior gaps are preserved"
                ),
            },
            "constant_cp": {
                "aggregate_sha256": CONSTANT_CP_AGGREGATE_SHA256,
                "case_support_json_sha256": CONSTANT_CP_JSON_SHA256,
                "case_support_csv_sha256": CONSTANT_CP_CSV_SHA256,
                "case_support_identity_sha256": CONSTANT_CP_CASE_SUPPORT_SHA256,
                "receipt_file_sha256": CONSTANT_CP_RECEIPT_FILE_SHA256,
                "receipt_identity_sha256": CONSTANT_CP_RECEIPT_IDENTITY_SHA256,
            },
            "coordinate_identity_encoding": (
                "fluidsbench-drivaerml-coordinate-array-v1"
            ),
        },
        "series": series,
    }


def _valid_raw_ids(rows_by_station: Mapping[str, Sequence[Mapping[str, object]]]) -> list[int]:
    result: list[int] = []
    for station in VELOCITY_STATIONS:
        for row in rows_by_station[station]:
            if row.get("valid", True):
                result.append(_integer(row.get("raw_vtk_cell_id"), "velocity raw ID"))
    return result


def _cp_raw_ids(cuts: Sequence[Mapping[str, Any]], row_key: str) -> list[int]:
    return [
        _integer(row.get("raw_polygon_id"), "Cp raw polygon ID")
        for cut in cuts
        for row in cut[row_key]
    ]


def _constant_velocity_series(
    rows_by_station: Mapping[str, Sequence[Mapping[str, object]]],
    gathered: Any,
    support_record: Mapping[str, object],
    retained: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    identities = support_record["per_station_materialized_support_identity_sha256"]
    if not isinstance(identities, Mapping):
        raise ExportError("constant velocity per-station support identities are malformed")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for station in VELOCITY_STATIONS:
        valid = [row for row in rows_by_station[station] if row["valid"]]
        raw_ids = np.asarray([row["raw_vtk_cell_id"] for row in valid], dtype=np.int64)
        prediction = velocity_magnitude_ratio(gathered.values_for(raw_ids)).tolist()
        station_id = f"autocfd5_{station.lower()}"
        coordinates = [row["distance_m"] for row in valid]
        identity = retained.get(
            ("drivaerml-autocfd5-constant-v1", station_id)
        )
        if (
            identity is None
            or identity.get("support_identity_sha256") != identities[station]
            or identity.get("placement_receipt_identity_sha256")
            != CONSTANT_VELOCITY_RECEIPT_SHA256
            or identity.get("coordinate_count") != len(coordinates)
            or identity.get("coordinate_identity_sha256")
            != coordinate_array_identity_sha256(coordinates)
        ):
            raise ExportError(
                f"constant velocity {station} retained fixture identity differs"
            )
        result[("drivaerml-autocfd5-constant-v1", station_id)] = {
            "panel_id": "velocity_profiles",
            "family_id": "drivaerml-autocfd5-constant-v1",
            "placement_mode": "constant",
            "station_id": station_id,
            "quantity_id": "velocity_ratio",
            "scoring_role": "inherits_parent_candidate",
            "representation": "materialized",
            "placement_receipt_identity_sha256": identity[
                "placement_receipt_identity_sha256"
            ],
            "support_identity_sha256": identity["support_identity_sha256"],
            "coordinate_id": "distance_m",
            "coordinate_unit": "m",
            "coordinate": coordinates,
            "prediction": prediction,
        }
    return result


def _relative_velocity_series(
    rows_by_station: Mapping[str, Sequence[Mapping[str, object]]],
    gathered: Any,
    retained: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for station in VELOCITY_STATIONS:
        rows = rows_by_station[station]
        raw_ids = np.asarray([row["raw_vtk_cell_id"] for row in rows], dtype=np.int64)
        identity = retained[(CASE_ID, "drivaerml-velocity-relative-v3", station)]
        result[("drivaerml-velocity-relative-v3", station)] = {
            "panel_id": "velocity_profiles",
            "family_id": "drivaerml-velocity-relative-v3",
            "placement_mode": "relative",
            "station_id": station,
            "quantity_id": "velocity_ratio",
            "scoring_role": "report_only",
            "representation": "materialized",
            "placement_receipt_identity_sha256": identity[
                "placement_receipt_identity_sha256"
            ],
            "support_identity_sha256": identity["support_identity_sha256"],
            "coordinate_id": "normalized_arc_length",
            "coordinate_unit": "1",
            "coordinate": [row["coordinate"] for row in rows],
            "prediction": velocity_magnitude_ratio(gathered.values_for(raw_ids)).tolist(),
        }
    return result


def _constant_cp_series(
    cuts: Sequence[Mapping[str, Any]],
    gathered: Any,
    retained: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for cut in cuts:
        station = cut["definition"]["cut_id"]
        rows = cut["segments"]
        raw_ids = np.asarray([row["raw_polygon_id"] for row in rows], dtype=np.int64)
        support_identity = _sha256_bytes(_canonical_bytes(cut))
        coordinates = [row["arc_length_end_m"] for row in rows]
        identity = retained.get(("drivaerml_cp_constant_v1", station))
        if (
            identity is None
            or identity.get("support_identity_sha256") != support_identity
            or identity.get("placement_receipt_identity_sha256")
            != CONSTANT_CP_RECEIPT_IDENTITY_SHA256
            or identity.get("coordinate_count") != len(coordinates)
            or identity.get("coordinate_identity_sha256")
            != coordinate_array_identity_sha256(coordinates)
        ):
            raise ExportError(f"constant Cp {station} retained fixture identity differs")
        result[("drivaerml_cp_constant_v1", station)] = {
            "panel_id": "pressure_profiles",
            "family_id": "drivaerml_cp_constant_v1",
            "placement_mode": "constant",
            "station_id": station,
            "quantity_id": "cp",
            "scoring_role": "inherits_parent_candidate",
            "representation": "materialized",
            "placement_receipt_identity_sha256": identity[
                "placement_receipt_identity_sha256"
            ],
            "support_identity_sha256": identity["support_identity_sha256"],
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
            "coordinate": coordinates,
            "prediction": cp_from_kinematic_pressure(gathered.values_for(raw_ids)).tolist(),
        }
    return result


def _relative_cp_series(
    aliases: Sequence[Mapping[str, Any]],
    moving: Sequence[Mapping[str, Any]],
    gathered: Any,
    retained: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for alias in aliases:
        station = alias["station_id"]
        identity = retained[(CASE_ID, "drivaerml_cp_relative_v1", station)]
        result[("drivaerml_cp_relative_v1", station)] = {
            "panel_id": "pressure_profiles",
            "family_id": "drivaerml_cp_relative_v1",
            "placement_mode": "relative",
            "station_id": station,
            "quantity_id": "cp",
            "scoring_role": "report_only",
            "representation": "shared_alias",
            "placement_receipt_identity_sha256": identity[
                "placement_receipt_identity_sha256"
            ],
            "shared_support_ref": {
                "shared_support_id": SHARED_SUPPORT_IDS[station],
                "canonical_family_id": "drivaerml_cp_constant_v1",
                "canonical_station_id": station,
                "canonical_support_identity_sha256": identity[
                    "support_identity_sha256"
                ],
            },
        }
    for cut in moving:
        station = cut["station_id"]
        rows = cut["rows"]
        raw_ids = np.asarray([row["raw_polygon_id"] for row in rows], dtype=np.int64)
        identity = retained[(CASE_ID, "drivaerml_cp_relative_v1", station)]
        result[("drivaerml_cp_relative_v1", station)] = {
            "panel_id": "pressure_profiles",
            "family_id": "drivaerml_cp_relative_v1",
            "placement_mode": "relative",
            "station_id": station,
            "quantity_id": "cp",
            "scoring_role": "report_only",
            "representation": "materialized",
            "placement_receipt_identity_sha256": identity[
                "placement_receipt_identity_sha256"
            ],
            "support_identity_sha256": identity["support_identity_sha256"],
            "coordinate_id": "arc_length_m",
            "coordinate_unit": "m",
            "coordinate": [row["interval_arc_end_m"] for row in rows],
            "prediction": cp_from_kinematic_pressure(gathered.values_for(raw_ids)).tolist(),
        }
    return result


def _ordered_series(
    *groups: Mapping[tuple[str, str], dict[str, object]]
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for group in groups:
        for key, series in group.items():
            if key in by_key:
                raise ExportError(f"duplicate generated family/station {key}")
            by_key[key] = series
    result: list[dict[str, object]] = []
    for panel, family, station, quantity, representation in _relative_profile_expected_keys():
        series = by_key.get((family, station))
        if (
            series is None
            or series["panel_id"] != panel
            or series["quantity_id"] != quantity
            or series["representation"] != representation
        ):
            raise ExportError(f"generated series namespace differs at {family}/{station}")
        result.append(series)
    if len(result) != 40 or len(by_key) != 40:
        raise ExportError("generated run_419 fixture does not contain exactly 40 series")
    return result


def _series_lineage(series: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in series:
        representation = str(item["representation"])
        record: dict[str, object] = {
            "family_id": item["family_id"],
            "station_id": item["station_id"],
            "representation": representation,
            "placement_receipt_identity_sha256": item[
                "placement_receipt_identity_sha256"
            ],
        }
        if representation == "shared_alias":
            reference = item["shared_support_ref"]
            if not isinstance(reference, Mapping):
                raise ExportError("generated shared support reference is malformed")
            record["support_identity_sha256"] = reference[
                "canonical_support_identity_sha256"
            ]
            record["coordinate_count"] = 0
            record["prediction_count"] = 0
            record["support_source"] = "canonical_constant_cp_shared_support"
        else:
            coordinate = item["coordinate"]
            prediction = item["prediction"]
            if not isinstance(coordinate, list) or not isinstance(prediction, list):
                raise ExportError("generated materialized series arrays are malformed")
            record.update(
                {
                    "support_identity_sha256": item["support_identity_sha256"],
                    "coordinate_count": len(coordinate),
                    "prediction_count": len(prediction),
                    "coordinate_identity_sha256": coordinate_array_identity_sha256(
                        coordinate
                    ),
                    "support_source": (
                        "retained_relative_series_support_index"
                        if item["placement_mode"] == "relative"
                        else "retained_run419_constant_series_support_index"
                    ),
                }
            )
        records.append(record)
    return records


def _write_json_exclusive(path: Path, value: object) -> dict[str, object]:
    payload = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {
        "file": path.name,
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_dir.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise ExportError(f"refusing to overwrite output directory: {output}")
    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    evaluator_commit, implementation = _verify_evaluator_revision(
        args.evaluator_git_revision
    )
    _, native_case = _verify_native_pin(args.native_source_pin)
    model = _verify_model_inputs(args)

    surface_manifest = load_prediction_chunk_manifest(args.surface_prediction_manifest)
    volume_manifest = load_prediction_chunk_manifest(args.volume_prediction_manifest)
    surface_record = _manifest_record(
        surface_manifest,
        expected_sha256=SURFACE_MANIFEST_SHA256,
        expected_support="surface_native_cells",
        expected_count=SURFACE_ENTITY_COUNT,
        expected_chunks=SURFACE_CHUNK_COUNT,
    )
    volume_record = _manifest_record(
        volume_manifest,
        expected_sha256=VOLUME_MANIFEST_SHA256,
        expected_support="volume_native_cells",
        expected_count=VOLUME_ENTITY_COUNT,
        expected_chunks=VOLUME_CHUNK_COUNT,
    )
    case_evidence = _verify_current_case_evidence(
        args.current_case_evaluation_evidence,
        args.current_case_implementation_receipt,
        evaluator_commit=evaluator_commit,
        implementation=implementation,
        native_source_pin_path=args.native_source_pin,
        surface=surface_record,
        volume=volume_record,
    )

    producer_root = _directory(args.relative_producer_root, "relative producer root")
    constant_cp_root = _directory(
        args.constant_cp_campaign_root, "constant Cp campaign root"
    )
    manifests, manifest_records = _verify_retained_manifests(producer_root)

    support_index_path = _regular(args.relative_support_index, "relative support index")
    expected_index_path = (
        ROOT / "benchmark-specs/drivaerml/support/relative-v3/series-support-index.json"
    ).resolve()
    if support_index_path != expected_index_path:
        raise ExportError("relative support index must be the evaluator-retained repository file")
    retained = _relative_series_support_index()
    run419_relative = {
        key: value for key, value in retained.items() if key[0] == CASE_ID
    }
    if len(run419_relative) != 20:
        raise ExportError("retained support index does not contain 20 run_419 relative entries")
    support_index_bytes = _read_bytes(support_index_path, "relative support index")

    constant_support_index_path = _regular(
        args.constant_support_index, "run_419 constant support index"
    )
    if constant_support_index_path != RUN419_CONSTANT_SERIES_SUPPORT_INDEX_PATH.resolve():
        raise ExportError(
            "constant support index must be the evaluator-retained repository file"
        )
    constant_support_index_bytes = _read_bytes(
        constant_support_index_path,
        "run_419 constant support index",
        RUN419_CONSTANT_SERIES_SUPPORT_INDEX_SHA256,
    )
    retained_constant = _run419_constant_series_support_index()
    if len(retained_constant) != 20:
        raise ExportError(
            "retained run_419 constant support index does not contain 20 entries"
        )

    constant_velocity, constant_velocity_record = _verify_constant_velocity(
        _directory(args.constant_velocity_case_dir, "constant velocity case directory")
    )
    relative_velocity, relative_velocity_record = _verify_relative_velocity(
        producer_root, manifests, retained
    )
    constant_cp, relative_cp_aliases, relative_cp_moving, cp_record = _verify_cp_supports(
        constant_cp_root, producer_root, manifests, retained
    )
    replayed_constant_index = build_run419_constant_series_support_index(
        args.constant_velocity_case_dir,
        constant_cp_root,
    )
    if _canonical_bytes(replayed_constant_index) != constant_support_index_bytes:
        raise ExportError(
            "retained run_419 constant support index differs from producer replay"
        )

    volume_raw_ids = [
        *_valid_raw_ids(constant_velocity),
        *_valid_raw_ids(relative_velocity),
    ]
    surface_raw_ids = [
        *_cp_raw_ids(constant_cp, "segments"),
        *_cp_raw_ids(relative_cp_moving, "rows"),
    ]
    volume_gather = gather_mapped_prediction_field(
        volume_manifest,
        volume_raw_ids,
        case_id=CASE_ID,
        support_id="volume_native_cells",
        field_name="UMeanTrim",
        expected_total_row_count=VOLUME_ENTITY_COUNT,
        maximum_chunk_rows=args.maximum_prediction_chunk_rows,
        hash_chunk_bytes=args.hash_chunk_bytes,
        validation_block_rows=args.maximum_prediction_chunk_rows,
    )
    surface_gather = gather_mapped_prediction_field(
        surface_manifest,
        surface_raw_ids,
        case_id=CASE_ID,
        support_id="surface_native_cells",
        field_name="pMeanTrim",
        expected_total_row_count=SURFACE_ENTITY_COUNT,
        maximum_chunk_rows=args.maximum_prediction_chunk_rows,
        hash_chunk_bytes=args.hash_chunk_bytes,
        validation_block_rows=args.maximum_prediction_chunk_rows,
    )
    if (
        tuple(volume_gather.chunk_sha256)
        != tuple(row["sha256"] for row in volume_record["chunks"])  # type: ignore[index]
        or tuple(surface_gather.chunk_sha256)
        != tuple(row["sha256"] for row in surface_record["chunks"])  # type: ignore[index]
    ):
        raise ExportError("evaluator sparse gathers differ from prediction manifests")

    series = _ordered_series(
        _constant_velocity_series(
            constant_velocity,
            volume_gather,
            constant_velocity_record,
            retained_constant,
        ),
        _relative_velocity_series(relative_velocity, volume_gather, retained),
        _constant_cp_series(constant_cp, surface_gather, retained_constant),
        _relative_cp_series(
            relative_cp_aliases, relative_cp_moving, surface_gather, retained
        ),
    )
    evaluation = CandidateDatasetEvaluation(
        {
            "split": {
                "split_id": "run419-regression",
                "case_set_id": "standard",
                "case_ids": [CASE_ID],
            }
        }
    )
    package = schema_v3_relative_profile_chunks_candidate_adapter(
        evaluation,
        submission_id=args.submission_id,
        namespaced_series_by_case={CASE_ID: series},
        cases_per_chunk=1,
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        profile_receipt = write_schema_v3_profile_chunks_candidate(
            package, temporary / "profiles"
        )
        provenance = {
            "schema": "drivaerml-transolver-run419-current-relative-profile-fixture-v1",
            "schema_version": 1,
            "status": "complete_one_case_real_model_regression_fixture_not_submission",
            "case_id": CASE_ID,
            "profile_format": RELATIVE_PROFILE_FORMAT,
            "series_count": 40,
            "series_composition": {
                "constant_velocity_materialized": 16,
                "relative_velocity_materialized": 16,
                "constant_cp_materialized": 4,
                "relative_cp_shared_alias": 2,
                "relative_cp_materialized": 2,
            },
            "evaluator": {
                "repository": "neilashton/fluidsbench-submission",
                "git_revision": evaluator_commit,
                "implementation": implementation,
                "current_case_evaluation": case_evidence,
                "profile_assembler": (
                    "reference.drivaerml.dataset_scorer."
                    "schema_v3_relative_profile_chunks_candidate_adapter"
                ),
                "profile_writer": (
                    "reference.drivaerml.dataset_scorer."
                    "write_schema_v3_profile_chunks_candidate"
                ),
                "sparse_prediction_gather": (
                    "reference.drivaerml.diagnostic_evaluator."
                    "gather_mapped_prediction_field"
                ),
            },
            "dataset": {
                "provider": "Hugging Face",
                "repository": DATASET_REPOSITORY,
                "immutable_revision": DATASET_REVISION,
                "native_source_pin": {
                    "repository_path": (
                        "benchmark-specs/drivaerml/proposal/native-source-pin.json"
                    ),
                    "sha256": NATIVE_SOURCE_PIN_SHA256,
                },
                "run419": {
                    "boundary_sha256": BOUNDARY_SHA256,
                    "boundary_size_bytes": native_case["boundary"]["size_bytes"],
                    "volume_part_sha256": list(VOLUME_PART_SHA256),
                    "volume_total_size_bytes": native_case["volume"]["total_size_bytes"],
                },
                "pin_verified_before_prediction_gather": True,
            },
            "model": model,
            "prediction_inputs": {
                "surface": {
                    **surface_record,
                    "gather": surface_gather.audit_record(),
                },
                "volume": {
                    **volume_record,
                    "gather": volume_gather.audit_record(),
                },
                "observed_chunk_count_discrepancy": False,
                "expected_and_verified_chunk_counts": {"surface": 8, "volume": 122},
            },
            "support": {
                "retained_all484_manifests": manifest_records,
                "relative_series_support_index": {
                    "repository_path": (
                        "benchmark-specs/drivaerml/support/relative-v3/"
                        "series-support-index.json"
                    ),
                    "sha256": _sha256_bytes(support_index_bytes),
                    "size_bytes": len(support_index_bytes),
                    "run419_relative_series_count": 20,
                },
                "run419_constant_series_support_index": {
                    "repository_path": (
                        "benchmark-specs/drivaerml/support/relative-v3/"
                        "run419-constant-series-support-index.json"
                    ),
                    "sha256": _sha256_bytes(constant_support_index_bytes),
                    "size_bytes": len(constant_support_index_bytes),
                    "series_count": 20,
                    "producer_replay_bytes_equal": True,
                    "scope": "one_case_report_only_regression_fixture",
                    "constant_velocity_valid_count": 3_684,
                    "constant_velocity_unsupported_count": 72,
                    "constant_velocity_interior_gaps_preserved": True,
                },
                "constant_velocity": constant_velocity_record,
                "relative_velocity": relative_velocity_record,
                "cp": cp_record,
            },
            "quantities": {
                "cp": "2*pMeanTrim/(38.889^2)",
                "velocity_ratio": "magnitude(UMeanTrim)/38.889",
                "native_association": "CellData",
                "remeshing_or_interpolation": False,
            },
            "series_lineage": _series_lineage(series),
            "fixture": {
                "directory": "profiles",
                **profile_receipt,
            },
            "claims": {
                "official_submission": False,
                "relative_scoring_activated": False,
                "submission_validation_opened": False,
                "scoring_weight_changed": False,
                "three_model_sensitivity_gate_complete": False,
                "owner_approval_complete": False,
                "model_quality_claim": False,
                "sparse_constant_velocity_is_activation_or_scoring_support": False,
            },
        }
        provenance_receipt = _write_json_exclusive(
            temporary / "provenance.json", provenance
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_directory": output.name,
        "profiles": profile_receipt,
        "provenance": provenance_receipt,
        "evaluator_git_revision": evaluator_commit,
        "dataset_revision": DATASET_REVISION,
        "surface_prediction_manifest_sha256": SURFACE_MANIFEST_SHA256,
        "volume_prediction_manifest_sha256": VOLUME_MANIFEST_SHA256,
        "series_count": 40,
        "official_submission": False,
        "relative_scoring_activated": False,
    }


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except (ExportError, CoordinateIdentityError, OSError, ValueError) as error:
        raise SystemExit(f"run_419 fixture export failed: {error}") from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
