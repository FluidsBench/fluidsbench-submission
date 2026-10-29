#!/usr/bin/env python3
"""Evaluate exactly the real run_1 and run_44 candidate pilot cases.

This is a fail-closed orchestration example.  It reuses the repository's core
and AutoCFD5 candidate evaluator entry points, always reconstructs the pinned
multipart VTU byte stream, and never downloads or generates scientific data.
The DrivAerML scoring contract remains closed, so the command writes candidate
evidence only and never writes ``submission.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reference.drivaerml.diagnostic_evaluator import (  # noqa: E402
    CANDIDATE_SCHEMA as DIAGNOSTIC_EVIDENCE_SCHEMA,
    DrivAerDiagnosticEvaluatorError,
    EXPECTED_SUBMISSION_PROFILE_SHA256,
    load_strict_velocity_10mm_mapping,
)
from reference.drivaerml.evaluator import (  # noqa: E402
    CANDIDATE_EVIDENCE_SCHEMA as CORE_EVIDENCE_SCHEMA,
    DrivAerCandidateEvaluatorError,
    validate_native_source_contract,
)
from reference.drivaerml.native_surface import DrivAerNativeSurfaceError  # noqa: E402
from reference.drivaerml.prediction_chunks import (  # noqa: E402
    PredictionChunkError,
    load_prediction_chunk_manifest,
)
from reference.drivaerml.source import (  # noqa: E402
    NativeSourceError,
    load_native_source_pin,
)
from scripts.evaluate_drivaerml_candidate_case import run as _run_core_case  # noqa: E402
from scripts.evaluate_drivaerml_candidate_diagnostics import (  # noqa: E402
    DEFAULT_PROFILE,
    run as _run_diagnostic_case,
)


INPUT_SCHEMA = "drivaerml-run1-run44-reference-inputs-v3"
OUTPUT_SCHEMA = "drivaerml-run1-run44-reference-evidence-v4"
OUTPUT_STATUS = "candidate_pilot_evidence_not_official_submission"
EVALUATOR_REFERENCE_VERSION = "drivaerml-evaluator-v3-candidate"
REPOSITORY_URL = "https://github.com/neilashton/fluidsbench-submission"
REQUIRED_CASE_PART_COUNTS = {"run_1": 2, "run_44": 3}
PREDICTION_SUPPORT_IDS = ("surface_native_cells", "volume_native_cells")
IMPLEMENTATION_FILES = (
    "examples/drivaerml-candidate-native-chunks/real_reference_driver.py",
    "reference/drivaerml/evaluator.py",
    "reference/drivaerml/diagnostic_evaluator.py",
    "scripts/evaluate_drivaerml_candidate_case.py",
    "scripts/evaluate_drivaerml_candidate_diagnostics.py",
    "benchmark-specs/drivaerml/submission-spec.json",
    "requirements-drivaerml-evaluator.txt",
)
CASE_INPUT_KEYS = frozenset(
    {
        "case_id",
        "surface_area_npy",
        "surface_prediction_manifest",
        "volume_prediction_manifest",
        "velocity_mapping_json",
        "velocity_receipt_json",
    }
)


class RealReferenceDriverError(ValueError):
    """Raised before or during a real candidate pilot orchestration."""


@dataclass(frozen=True)
class RealCaseInputs:
    case_id: str
    surface_area_npy: Path
    surface_prediction_manifest: Path
    volume_prediction_manifest: Path
    velocity_mapping_json: Path
    velocity_receipt_json: Path


@dataclass(frozen=True)
class _PredictionEvidenceIdentity:
    manifest_sha256: str
    chunk_sha256: tuple[str, ...]
    entity_count: int


@dataclass(frozen=True)
class _PreflightResult:
    pin: Any
    config_path: Path
    config_sha256: str
    native_source_pin_sha256: str
    diagnostic_profile_path: Path
    diagnostic_profile_sha256: str
    cases: tuple[RealCaseInputs, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RealReferenceDriverError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _implementation_identities() -> dict[str, str]:
    """Hash every repository file that defines this orchestration path."""

    return {
        relative_path: _sha256_file(
            _regular_file(REPOSITORY_ROOT / relative_path, relative_path)
        )
        for relative_path in IMPLEMENTATION_FILES
    }


def _repository_identity() -> dict[str, object]:
    """Record the local Git revision without making Git a runtime requirement."""

    def invoke(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    try:
        revision_result = invoke(["rev-parse", "--verify", "HEAD"])
        status_result = invoke(
            ["status", "--porcelain=v1", "--untracked-files=no"]
        )
    except OSError:
        return {
            "git_metadata_available": False,
            "git_revision": None,
            "tracked_worktree_clean": None,
        }
    revision = revision_result.stdout.strip()
    if (
        revision_result.returncode != 0
        or status_result.returncode != 0
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        return {
            "git_metadata_available": False,
            "git_revision": None,
            "tracked_worktree_clean": None,
        }
    return {
        "git_metadata_available": True,
        "git_revision": revision,
        "tracked_worktree_clean": status_result.stdout == "",
    }


def _require_unchanged(
    identities: dict[Path, tuple[str, str]], *, phase: str
) -> None:
    """Fail if a preflight input changed before evidence publication."""

    for path, (label, expected_sha256) in identities.items():
        if _sha256_file(path) != expected_sha256:
            raise RealReferenceDriverError(f"{label} changed {phase}")


def _require_repository_identity_unchanged(
    expected: dict[str, object], *, phase: str
) -> None:
    if _repository_identity() != expected:
        raise RealReferenceDriverError(
            f"repository revision or tracked state changed {phase}"
        )


def _require_clean_evidence_repository(identity: dict[str, object]) -> None:
    if not identity.get("git_metadata_available"):
        raise RealReferenceDriverError(
            "real candidate evidence requires an identifiable Git checkout"
        )
    if identity.get("tracked_worktree_clean") is not True:
        raise RealReferenceDriverError(
            "real candidate evidence requires a clean tracked Git worktree"
        )


def _validate_candidate_evaluator_binding() -> None:
    path = REPOSITORY_ROOT / "benchmark-specs" / "drivaerml" / "submission-spec.json"
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        observed = document["scoring_support"]["dataset_evaluator_binding"][
            "evaluator_reference_version"
        ]
    except RealReferenceDriverError:
        raise
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise RealReferenceDriverError(
            "cannot load the candidate evaluator binding from submission-spec.json"
        ) from error
    if observed != EVALUATOR_REFERENCE_VERSION:
        raise RealReferenceDriverError(
            "candidate evaluator reference version differs from submission-spec.json"
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RealReferenceDriverError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_evidence_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RealReferenceDriverError(f"{label} must be a positive integer")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealReferenceDriverError(f"{label} must be an object")
    return value


def _generated_evidence_document(
    path: Path,
    identity: dict[str, object],
    *,
    case_id: str,
    label: str,
) -> dict[str, Any]:
    """Reload one generated output and bind it to the writer's identity."""

    source = _regular_file(path, label)
    expected_sha256 = _sha256(identity.get("sha256"), f"{label} SHA-256")
    expected_size = _positive_evidence_count(
        identity.get("byte_size"), f"{label} byte_size"
    )
    initial_sha256 = _sha256_file(source)
    if initial_sha256 != expected_sha256 or source.stat().st_size != expected_size:
        raise RealReferenceDriverError(
            f"{label} differs from the evaluator writer identity"
        )
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except RealReferenceDriverError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RealReferenceDriverError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict) or document.get("case_id") != case_id:
        raise RealReferenceDriverError(f"{label} case identity mismatch")
    if _sha256_file(source) != initial_sha256:
        raise RealReferenceDriverError(f"{label} changed while it was reloaded")
    return document


def _prediction_evidence_identities(
    document: dict[str, Any],
    *,
    case_id: str,
    diagnostic: bool,
) -> dict[str, _PredictionEvidenceIdentity]:
    """Extract the prediction identities common to core and diagnostics."""

    support_ids = (
        ("volume_native_cells",) if diagnostic else PREDICTION_SUPPORT_IDS
    )
    if diagnostic:
        root = _mapping(
            document.get("sparse_gather_evidence"),
            f"{case_id} diagnostic sparse_gather_evidence",
        )
        records = {
            "volume_native_cells": root.get("volume_prediction"),
        }
    else:
        root = _mapping(
            document.get("prediction_inputs"),
            f"{case_id} core prediction_inputs",
        )
        records = {
            support_id: root.get(support_id)
            for support_id in PREDICTION_SUPPORT_IDS
        }

    result: dict[str, _PredictionEvidenceIdentity] = {}
    for support_id in support_ids:
        record = _mapping(
            records[support_id], f"{case_id}/{support_id} prediction evidence"
        )
        chunks = record.get("chunk_sha256")
        if not isinstance(chunks, list) or not chunks:
            raise RealReferenceDriverError(
                f"{case_id}/{support_id} prediction evidence has no chunk identities"
            )
        chunk_sha256 = tuple(
            _sha256(value, f"{case_id}/{support_id} chunk SHA-256")
            for value in chunks
        )
        chunk_count = _positive_evidence_count(
            record.get("chunk_count"), f"{case_id}/{support_id} chunk_count"
        )
        if chunk_count != len(chunk_sha256):
            raise RealReferenceDriverError(
                f"{case_id}/{support_id} chunk count differs from its identities"
            )
        result[support_id] = _PredictionEvidenceIdentity(
            manifest_sha256=_sha256(
                record.get("manifest_sha256"),
                f"{case_id}/{support_id} manifest SHA-256",
            ),
            chunk_sha256=chunk_sha256,
            entity_count=_positive_evidence_count(
                record.get("total_row_count" if diagnostic else "entity_count"),
                f"{case_id}/{support_id} entity count",
            ),
        )
    return result


def _validated_case_provenance(
    core_document: dict[str, Any],
    diagnostic_document: dict[str, Any],
    *,
    case_id: str,
    native_source_pin_sha256: str,
    diagnostic_profile_sha256: str,
) -> str:
    """Verify generated evidence used the preflight pin and profile."""

    if (
        core_document.get("schema") != CORE_EVIDENCE_SCHEMA
        or core_document.get("schema_version") != 4
        or core_document.get("prediction_scope") != "surface_and_volume"
    ):
        raise RealReferenceDriverError(
            f"{case_id} core evidence schema differs from the candidate evaluator"
        )
    if (
        diagnostic_document.get("schema") != DIAGNOSTIC_EVIDENCE_SCHEMA
        or diagnostic_document.get("schema_version") != 4
        or diagnostic_document.get("prediction_scope") != "surface_and_volume"
    ):
        raise RealReferenceDriverError(
            f"{case_id} diagnostic evidence schema differs from the candidate evaluator"
        )
    source = _mapping(core_document.get("source"), f"{case_id} core source")
    observed_pin = _sha256(
        source.get("native_source_pin_sha256"),
        f"{case_id} core native-source pin SHA-256",
    )
    if observed_pin != native_source_pin_sha256:
        raise RealReferenceDriverError(
            f"{case_id} core evidence used a different native-source pin"
        )
    surface_native = _mapping(
        source.get("surface_native"), f"{case_id} core native surface"
    )
    vtk_version = surface_native.get("vtk_version")
    if not isinstance(vtk_version, str) or not vtk_version:
        raise RealReferenceDriverError(
            f"{case_id} core evidence has no VTK runtime version"
        )

    mapping_inputs = _mapping(
        diagnostic_document.get("mapping_inputs"),
        f"{case_id} diagnostic mapping_inputs",
    )
    if set(mapping_inputs) != {"velocity_10mm"}:
        raise RealReferenceDriverError(
            f"{case_id} diagnostic evidence must use velocity-only v9 mapping input"
        )
    velocity_input = _mapping(
        mapping_inputs.get("velocity_10mm"),
        f"{case_id} diagnostic velocity_10mm",
    )
    observed_profile = _sha256(
        velocity_input.get("profile_sha256"),
        f"{case_id} diagnostic velocity_10mm profile SHA-256",
    )
    if observed_profile != diagnostic_profile_sha256:
        raise RealReferenceDriverError(
            f"{case_id} diagnostic evidence used a different v9 profile"
        )

    metrics = _mapping(
        diagnostic_document.get("metrics"), f"{case_id} diagnostic metrics"
    )
    cp_cut = _mapping(metrics.get("cp_cut_rmse"), f"{case_id} Cp-cut metric")
    unavailable_reasons = cp_cut.get("unavailable_reasons")
    if (
        cp_cut.get("metric_id") != "cp_cut_rmse"
        or cp_cut.get("ranked_value_available") is not False
        or cp_cut.get("required_cut_count") != 4
        or cp_cut.get("case_equal_cut_mean_rmse") is not None
        or cp_cut.get("weighting") != "native_cut_intersection_segment_length"
        or cp_cut.get("support_status") != "pending_immutable_owner_release"
        or cp_cut.get("discrete_cp_probe_fallback_used") is not False
        or not isinstance(unavailable_reasons, list)
        or len(unavailable_reasons) != 1
        or not isinstance(unavailable_reasons[0], dict)
        or unavailable_reasons[0].get("reason")
        != "immutable_native_cp_cut_extraction_support_not_published"
    ):
        raise RealReferenceDriverError(
            f"{case_id} diagnostic evidence must keep four Cp cuts explicitly unavailable"
        )
    return vtk_version


def _runtime_identity(vtk_versions: set[str]) -> dict[str, str]:
    if len(vtk_versions) != 1:
        raise RealReferenceDriverError(
            "core evidence did not report one consistent VTK runtime version"
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "vtk": next(iter(vtk_versions)),
        "byte_order": sys.byteorder,
    }


def _regular_file(path: Path | str, label: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        if candidate.is_symlink():
            raise RealReferenceDriverError(f"{label} must not be a symlink")
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RealReferenceDriverError(f"{label} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise RealReferenceDriverError(f"{label} must be a regular file")
    return resolved


def _config_file(config_directory: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RealReferenceDriverError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    return _regular_file(path, label)


def load_case_inputs(
    path: Path | str,
) -> tuple[Path, str, tuple[RealCaseInputs, ...]]:
    """Load a closed two-case config and preflight every declared input file."""

    source = _regular_file(path, "case-input config")
    try:
        source_bytes = source.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        document = json.loads(
            source_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except RealReferenceDriverError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RealReferenceDriverError("case-input config is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"schema", "cases"}:
        raise RealReferenceDriverError(
            "case-input config must contain exactly 'schema' and 'cases'"
        )
    if document["schema"] != INPUT_SCHEMA:
        raise RealReferenceDriverError(f"case-input schema must equal {INPUT_SCHEMA!r}")
    rows = document["cases"]
    if not isinstance(rows, list) or len(rows) != len(REQUIRED_CASE_PART_COUNTS):
        raise RealReferenceDriverError("case-input config must contain exactly two cases")
    parsed: dict[str, RealCaseInputs] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != CASE_INPUT_KEYS:
            missing = sorted(CASE_INPUT_KEYS - set(row) if isinstance(row, dict) else CASE_INPUT_KEYS)
            unknown = sorted(set(row) - CASE_INPUT_KEYS if isinstance(row, dict) else [])
            raise RealReferenceDriverError(
                f"cases[{index}] has a closed schema; missing={missing}, unknown={unknown}"
            )
        case_id = row["case_id"]
        if not isinstance(case_id, str) or case_id not in REQUIRED_CASE_PART_COUNTS:
            raise RealReferenceDriverError("the only permitted cases are run_1 and run_44")
        if case_id in parsed:
            raise RealReferenceDriverError(f"duplicate case {case_id}")
        values: dict[str, Path] = {
            key: _config_file(source.parent, row[key], f"{case_id}.{key}")
            for key in sorted(CASE_INPUT_KEYS - {"case_id"})
        }
        parsed[case_id] = RealCaseInputs(case_id=case_id, **values)
    if set(parsed) != set(REQUIRED_CASE_PART_COUNTS):
        raise RealReferenceDriverError("case-input config must cover run_1 and run_44")
    return (
        source,
        source_sha256,
        tuple(parsed[case_id] for case_id in REQUIRED_CASE_PART_COUNTS),
    )


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-inputs", type=Path, required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--diagnostic-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-prediction-chunk-rows",
        type=_positive_integer,
        default=1_000_000,
    )
    parser.add_argument(
        "--io-chunk-bytes",
        type=_positive_integer,
        default=8 * 1024 * 1024,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _preflight(
    args: argparse.Namespace,
) -> _PreflightResult:
    config_path, config_sha256, cases = load_case_inputs(args.case_inputs)
    pin_path = _regular_file(args.native_source_pin, "native-source pin")
    profile_path = _regular_file(args.diagnostic_profile, "diagnostic profile")
    pin_sha256 = _sha256_file(pin_path)
    profile_sha256 = _sha256_file(profile_path)
    if profile_sha256 != EXPECTED_SUBMISSION_PROFILE_SHA256:
        raise RealReferenceDriverError(
            "diagnostic profile SHA-256 differs from the candidate evaluator profile"
        )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise RealReferenceDriverError("dataset root must be an existing directory")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RealReferenceDriverError(f"output must not already exist: {output}")
    pin = load_native_source_pin(pin_path)
    validated_pin_sha256 = validate_native_source_contract(pin)
    if validated_pin_sha256 != pin_sha256:
        raise RealReferenceDriverError(
            "native-source pin changed while it was loaded and validated"
        )
    for case_id, expected_parts in REQUIRED_CASE_PART_COUNTS.items():
        if len(pin.case(case_id).volume_parts) != expected_parts:
            raise RealReferenceDriverError(
                f"pinned {case_id} must contain exactly {expected_parts} volume parts"
            )
        pin.resolve(case_id, dataset_root)
    args.native_source_pin = pin_path
    args.dataset_root = dataset_root
    args.diagnostic_profile = profile_path
    args.output = output
    return _PreflightResult(
        pin=pin,
        config_path=config_path,
        config_sha256=config_sha256,
        native_source_pin_sha256=pin_sha256,
        diagnostic_profile_path=profile_path,
        diagnostic_profile_sha256=profile_sha256,
        cases=cases,
    )


def _preflight_case_inputs(
    preflight: _PreflightResult,
) -> dict[Path, tuple[str, str]]:
    """Validate every compact case input before any native-field evaluation.

    The core evaluator reads multi-gigabyte native fields. Validate both cases'
    velocity mappings and prediction-manifest metadata first, so a stale
    later-case artifact cannot waste an earlier case's core pass. Prediction
    NPZ payloads remain lazily verified by the evaluators themselves. The four
    continuous Cp cuts have no immutable support yet, and the excluded 209
    discrete probes are deliberately not accepted as a fallback input.
    """

    retained: dict[Path, tuple[str, str]] = {}
    velocity_mappings: dict[str, Any] = {}

    for inputs in preflight.cases:
        case = preflight.pin.case(inputs.case_id)
        mapping = load_strict_velocity_10mm_mapping(
            inputs.velocity_mapping_json,
            inputs.velocity_receipt_json,
            autocfd5_profile=preflight.diagnostic_profile_path,
            case_id=inputs.case_id,
            expected_source_pin_sha256=preflight.native_source_pin_sha256,
            expected_source_part_sha256=tuple(
                part.sha256 for part in case.volume_parts
            ),
        )
        velocity_mappings[inputs.case_id] = mapping
        retained[mapping.artifact_path] = (
            f"{inputs.case_id} 10 mm velocity mapping",
            mapping.artifact_sha256,
        )
        retained[mapping.receipt_path] = (
            f"{inputs.case_id} velocity receipt",
            mapping.receipt_sha256,
        )

    for inputs in preflight.cases:
        case = preflight.pin.case(inputs.case_id)
        velocity_mapping = velocity_mappings[inputs.case_id]
        for path, expected_support_id, expected_count in (
            (
                inputs.surface_prediction_manifest,
                "surface_native_cells",
                case.surface_cell_area.element_count,
            ),
            (
                inputs.volume_prediction_manifest,
                "volume_native_cells",
                velocity_mapping.native_cell_count,
            ),
        ):
            manifest = load_prediction_chunk_manifest(path)
            if (
                manifest.case_id != inputs.case_id
                or manifest.support_id != expected_support_id
                or manifest.total_row_count != expected_count
            ):
                raise RealReferenceDriverError(
                    f"{inputs.case_id} {expected_support_id} prediction manifest "
                    "identity/count differs from the strict native support"
                )
            retained[manifest.path] = (
                f"{inputs.case_id} {expected_support_id} prediction manifest",
                manifest.sha256,
            )
    return retained


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run both evaluator paths in a staged directory, then publish evidence."""

    preflight = _preflight(args)
    pin = preflight.pin
    implementation_sha256 = _implementation_identities()
    _validate_candidate_evaluator_binding()
    repository_identity = _repository_identity()
    _require_clean_evidence_repository(repository_identity)
    retained_inputs: dict[Path, tuple[str, str]] = {
        preflight.config_path: ("case-input config", preflight.config_sha256),
        args.native_source_pin: (
            "native-source pin",
            preflight.native_source_pin_sha256,
        ),
        args.diagnostic_profile: (
            "diagnostic profile",
            preflight.diagnostic_profile_sha256,
        ),
    }
    retained_inputs.update(
        {
            (REPOSITORY_ROOT / relative_path).resolve(): (
                f"implementation file {relative_path}",
                digest,
            )
            for relative_path, digest in implementation_sha256.items()
        }
    )
    retained_inputs.update(_preflight_case_inputs(preflight))
    _require_unchanged(retained_inputs, phase="during all-case preflight")
    _require_repository_identity_unchanged(
        repository_identity, phase="during all-case preflight"
    )
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as staging_text:
        staging = Path(staging_text)
        case_results: list[dict[str, object]] = []
        vtk_versions: set[str] = set()
        for case_inputs in preflight.cases:
            case_directory = staging / "cases" / case_inputs.case_id
            core_output = case_directory / "core-evaluation.json"
            diagnostic_output = case_directory / "diagnostic-evaluation.json"
            core_identity = _run_core_case(
                SimpleNamespace(
                    case_id=case_inputs.case_id,
                    native_source_pin=args.native_source_pin,
                    dataset_root=args.dataset_root,
                    monolithic_vtu=None,
                    multipart=True,
                    surface_area_npy=case_inputs.surface_area_npy,
                    surface_prediction_manifest=(
                        case_inputs.surface_prediction_manifest
                    ),
                    volume_prediction_manifest=case_inputs.volume_prediction_manifest,
                    output=core_output,
                    maximum_prediction_chunk_rows=(
                        args.maximum_prediction_chunk_rows
                    ),
                    io_chunk_bytes=args.io_chunk_bytes,
                )
            )
            diagnostic_identity = _run_diagnostic_case(
                SimpleNamespace(
                    case_id=case_inputs.case_id,
                    native_source_pin=args.native_source_pin,
                    dataset_root=args.dataset_root,
                    diagnostic_profile=args.diagnostic_profile,
                    velocity_mapping_json=case_inputs.velocity_mapping_json,
                    velocity_receipt_json=case_inputs.velocity_receipt_json,
                    volume_prediction_manifest=case_inputs.volume_prediction_manifest,
                    multipart=True,
                    monolithic_vtu=None,
                    output=diagnostic_output,
                    maximum_prediction_chunk_rows=(
                        args.maximum_prediction_chunk_rows
                    ),
                    io_chunk_bytes=args.io_chunk_bytes,
                )
            )
            core_document = _generated_evidence_document(
                core_output,
                core_identity,
                case_id=case_inputs.case_id,
                label=f"{case_inputs.case_id} core evidence",
            )
            diagnostic_document = _generated_evidence_document(
                diagnostic_output,
                diagnostic_identity,
                case_id=case_inputs.case_id,
                label=f"{case_inputs.case_id} diagnostic evidence",
            )
            vtk_versions.add(
                _validated_case_provenance(
                    core_document,
                    diagnostic_document,
                    case_id=case_inputs.case_id,
                    native_source_pin_sha256=(
                        preflight.native_source_pin_sha256
                    ),
                    diagnostic_profile_sha256=(
                        preflight.diagnostic_profile_sha256
                    ),
                )
            )
            core_prediction_identities = _prediction_evidence_identities(
                core_document,
                case_id=case_inputs.case_id,
                diagnostic=False,
            )
            diagnostic_prediction_identities = _prediction_evidence_identities(
                diagnostic_document,
                case_id=case_inputs.case_id,
                diagnostic=True,
            )
            if diagnostic_prediction_identities != {
                "volume_native_cells": core_prediction_identities[
                    "volume_native_cells"
                ]
            }:
                raise RealReferenceDriverError(
                    f"{case_inputs.case_id} core and diagnostic evaluations used "
                    "different volume-prediction manifest, chunk, or entity identities"
                )
            case_results.append(
                {
                    "case_id": case_inputs.case_id,
                    "pinned_volume_part_count": len(
                        pin.case(case_inputs.case_id).volume_parts
                    ),
                    "transport": "verified_ordered_multipart_byte_stream",
                    "cross_evaluator_volume_prediction_identity_verified": True,
                    "core_evidence": {
                        **core_identity,
                        "file": core_output.relative_to(staging).as_posix(),
                    },
                    "diagnostic_evidence": {
                        **diagnostic_identity,
                        "file": diagnostic_output.relative_to(staging).as_posix(),
                    },
                }
            )
        _require_unchanged(retained_inputs, phase="after it was parsed")
        _require_repository_identity_unchanged(
            repository_identity, phase="after evaluation"
        )
        receipt: dict[str, object] = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": 4,
            "status": OUTPUT_STATUS,
            "official_submission": False,
            "scoring_contract_active": False,
            "owner_scientific_approval": False,
            "independent_participant_dry_run": False,
            "downloads_performed": False,
            "fabricated_scientific_results": False,
            "complete_484_case_split_evaluated": False,
            "public_scoring_support_eligible": False,
            "native_source_pin_sha256": preflight.native_source_pin_sha256,
            "diagnostic_profile_sha256": preflight.diagnostic_profile_sha256,
            "case_input_config_sha256": preflight.config_sha256,
            "evaluator_binding": {
                "repository_url": REPOSITORY_URL,
                "reference_version": EVALUATOR_REFERENCE_VERSION,
                "frozen_release": False,
                "implementation_files_sha256": implementation_sha256,
                "repository": repository_identity,
            },
            "runtime": _runtime_identity(vtk_versions),
            "volume_weighting": "one_per_native_cell",
            "geometric_cell_volume_weights_used": False,
            "continuous_cp_cuts": {
                "required_cut_count": 4,
                "ranked_value_available": False,
                "support_status": "pending_immutable_owner_release",
                "participant_field_required": False,
                "derived_from": "surface_native_cells.pMeanTrim",
                "weighting": "native_cut_intersection_segment_length",
                "discrete_cp_probe_fallback_used": False,
            },
            "cases": case_results,
        }
        receipt_path = staging / "validation-receipt.json"
        receipt_path.write_text(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise RealReferenceDriverError(f"output appeared during run: {output}")
        _require_unchanged(retained_inputs, phase="before evidence publication")
        _require_repository_identity_unchanged(
            repository_identity, phase="before evidence publication"
        )
        os.replace(staging, output)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipt = run(args)
    except (
        DrivAerCandidateEvaluatorError,
        DrivAerDiagnosticEvaluatorError,
        DrivAerNativeSurfaceError,
        NativeSourceError,
        PredictionChunkError,
        RealReferenceDriverError,
        OSError,
    ) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cases": [case["case_id"] for case in receipt["cases"]],
                "receipt": str(args.output.resolve() / "validation-receipt.json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_INPUT_KEYS",
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    "OUTPUT_STATUS",
    "REQUIRED_CASE_PART_COUNTS",
    "RealCaseInputs",
    "RealReferenceDriverError",
    "load_case_inputs",
    "main",
    "parse_args",
    "run",
]
