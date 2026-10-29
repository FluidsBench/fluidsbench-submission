#!/usr/bin/env python3
"""Evaluate DrivAerML profile resolution from native prediction artifacts.

This maintainer command validates the receipt-bound 1/2/5/10 mm mappings,
streams actual pinned multipart VTU ``UMeanTrim`` truth, exhausts every native
prediction chunk manifest, and then runs the frozen loss/Kendall convergence
checker.  It remains candidate-only and never opens submissions or activates
the scoring contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.diagnostic_evaluator import (  # noqa: E402
    DrivAerDiagnosticEvaluatorError,
    SparseNativeField,
    gather_mapped_prediction_field,
    gather_sparse_inline_native_field,
)
from reference.drivaerml.evaluator import (  # noqa: E402
    validate_native_source_contract,
)
from reference.drivaerml.prediction_chunks import (  # noqa: E402
    PredictionChunkError,
    PredictionChunkManifest,
)
from reference.drivaerml.profile_convergence import (  # noqa: E402
    DEFAULT_CONTRACT_PROPOSAL,
    OFFICIAL_CASE_ORDER,
    canonical_sha256,
    method_set_sha256,
    write_evidence,
)
from reference.drivaerml.profile_convergence_evaluator import (  # noqa: E402
    FIELD_NAME,
    MAPPING_ARTIFACT_NAMES,
    PREDICTION_STUDY_SCHEMA,
    SCHEMA_VERSION,
    SUPPORT_ID,
    CaseResolutionMappings,
    NativeProfileConvergenceError,
    ProfileMappingRow,
    ResolutionMapping,
    case_profile_losses,
    finalize_profile_convergence_evidence,
    prediction_manifest_set_identity,
    validate_mapping_grids,
)
from reference.drivaerml.retained_file import (  # noqa: E402
    RetainedFileError,
    RetainedVerifiedFile,
)
from reference.drivaerml.source import (  # noqa: E402
    NativeSourceError,
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_multipart,
)
from scripts.aggregate_drivaerml_velocity_assignments import (  # noqa: E402
    DEFAULT_AUTOCFD5_PROFILE,
    DEFAULT_NATIVE_SOURCE_PIN,
    OFFICIAL_CASE_IDS,
    OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
    VelocityAssignmentAggregateError,
    aggregate_velocity_assignments,
)
from reference.drivaerml.autocfd5 import load_autocfd5_definition  # noqa: E402


CONFIG_SCHEMA = PREDICTION_STUDY_SCHEMA
MAX_CONFIG_BYTES = 16 * 1024 * 1024
REQUIRED_PYTHON_VERSION = "3.12.13"
REQUIRED_NUMPY_VERSION = "2.2.6"
_CONFIG_KEYS = {
    "schema",
    "schema_version",
    "study_id",
    "method_set",
    "case_order",
    "prediction_manifests",
}
_METHOD_ROLES = {
    "physics_null",
    "nearest_training_design_vector_control",
    "trained_model_checkpoint",
}
_IMPLEMENTATION_FILES = (
    "scripts/evaluate_drivaerml_profile_resolution_from_predictions.py",
    "scripts/aggregate_drivaerml_velocity_assignments.py",
    "scripts/generate_drivaerml_velocity_assignments.py",
    "reference/drivaerml/profile_convergence_evaluator.py",
    "reference/drivaerml/profile_convergence.py",
    "reference/drivaerml/diagnostic_evaluator.py",
    "reference/drivaerml/evaluator.py",
    "reference/drivaerml/prediction_chunks.py",
    "reference/drivaerml/retained_file.py",
    "reference/drivaerml/source.py",
    "reference/drivaerml/autocfd5.py",
    "reference/drivaerml/velocity_assignments.py",
)


class NativeProfileConvergenceCLIError(ValueError):
    """Raised when the real-data convergence command cannot proceed safely."""


def _validated_runtime_binding() -> dict[str, object]:
    """Fail closed unless the frozen XML-streaming runtime is exact.

    This check intentionally has no filesystem side effects and must remain the
    first operation in :func:`run`, before study descriptors, prediction
    manifests, mappings, or native source bytes are opened.  VTK is not part of
    this reduction: the command streams the inline XML arrays directly.
    """

    python_version = platform.python_version()
    numpy_version = np.__version__
    binding = {
        "python": python_version,
        "numpy": numpy_version,
        "required_python": REQUIRED_PYTHON_VERSION,
        "required_numpy": REQUIRED_NUMPY_VERSION,
        "vtk_used_by_this_reduction": False,
    }
    if (
        python_version != REQUIRED_PYTHON_VERSION
        or numpy_version != REQUIRED_NUMPY_VERSION
    ):
        raise NativeProfileConvergenceCLIError(
            "native profile convergence requires exactly Python "
            f"{REQUIRED_PYTHON_VERSION} and NumPy {REQUIRED_NUMPY_VERSION}; "
            f"observed Python {python_version} and NumPy {numpy_version}"
        )
    return binding


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_binding() -> dict[str, object]:
    files = [
        {
            "file": relative,
            "sha256": _sha256_file(ROOT / relative),
        }
        for relative in _IMPLEMENTATION_FILES
    ]
    git_revision: str | None = None
    git_worktree_clean: bool | None = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if len(revision) == 40 and all(
            character in "0123456789abcdef" for character in revision
        ):
            git_revision = revision
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_worktree_clean = status == ""
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_revision": git_revision,
        "git_worktree_clean": git_worktree_clean,
        "source_files": files,
    }


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeProfileConvergenceCLIError(
                f"JSON contains duplicate object key {key!r}"
            )
        result[key] = value
    return result


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeProfileConvergenceCLIError(f"{label} must be an object")
    if set(value) != expected:
        raise NativeProfileConvergenceCLIError(
            f"{label} keys differ from schema "
            f"(missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)})"
        )
    return value


def _case_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("run_")
        or not value[4:].isascii()
        or not value[4:].isdecimal()
        or value[4:] != str(int(value[4:]))
        or int(value[4:]) < 1
    ):
        raise NativeProfileConvergenceCLIError(
            f"{label} must match run_<positive integer>"
        )
    return value


def _read_retained_json(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAX_CONFIG_BYTES,
) -> tuple[dict[str, Any], str]:
    try:
        with RetainedVerifiedFile.open(path, label=label) as retained:
            if retained.snapshot.size_bytes > maximum_bytes:
                raise NativeProfileConvergenceCLIError(
                    f"{label} exceeds {maximum_bytes} bytes"
                )
            digest = retained.sha256(chunk_bytes=8 * 1024 * 1024)
            retained.handle.seek(0)
            payload = retained.handle.read(maximum_bytes + 1)
            retained.assert_unchanged(context="while parsing JSON")
    except RetainedFileError as error:
        raise NativeProfileConvergenceCLIError(str(error)) from error
    if len(payload) > maximum_bytes:
        raise NativeProfileConvergenceCLIError(
            f"{label} exceeds {maximum_bytes} bytes"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except NativeProfileConvergenceCLIError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NativeProfileConvergenceCLIError(
            f"{label} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise NativeProfileConvergenceCLIError(f"{label} must be an object")
    return value, digest


def _resolve_input_path(value: object, *, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise NativeProfileConvergenceCLIError(f"{label} must be a non-empty path")
    unresolved = Path(value).expanduser()
    if not unresolved.is_absolute():
        unresolved = root / unresolved
    try:
        result = unresolved.resolve(strict=True)
    except OSError as error:
        raise NativeProfileConvergenceCLIError(
            f"{label} does not exist: {unresolved}"
        ) from error
    if not result.is_file():
        raise NativeProfileConvergenceCLIError(f"{label} must be a regular file")
    return result


def load_study_config(
    path: Path | str,
) -> tuple[
    dict[str, Any],
    str,
    tuple[str, ...],
    list[dict[str, Any]],
    dict[str, tuple[Path, ...]],
]:
    """Load the closed execution descriptor and resolve its local manifests."""

    config_path = Path(path).expanduser().resolve()
    value, byte_sha256 = _read_retained_json(
        config_path, label="native profile convergence study config"
    )
    root = _exact_keys(value, _CONFIG_KEYS, "study config")
    if (
        root["schema"] != CONFIG_SCHEMA
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise NativeProfileConvergenceCLIError(
            "unsupported native profile convergence config schema"
        )
    if not isinstance(root["study_id"], str) or not root["study_id"]:
        raise NativeProfileConvergenceCLIError("study_id must be non-empty")
    case_values = root["case_order"]
    if not isinstance(case_values, list) or not case_values:
        raise NativeProfileConvergenceCLIError("case_order must be non-empty")
    case_order = tuple(
        _case_id(item, f"case_order[{index}]")
        for index, item in enumerate(case_values)
    )
    if len(case_order) != len(set(case_order)):
        raise NativeProfileConvergenceCLIError("case_order contains duplicates")
    method_set = _exact_keys(
        root["method_set"],
        {"pinned_before_study", "sha256", "methods"},
        "method_set",
    )
    if not isinstance(method_set["pinned_before_study"], bool):
        raise NativeProfileConvergenceCLIError(
            "method_set.pinned_before_study must be Boolean"
        )
    methods = method_set["methods"]
    if not isinstance(methods, list) or not methods:
        raise NativeProfileConvergenceCLIError("method_set.methods cannot be empty")
    try:
        actual_method_set_sha = method_set_sha256(methods)
    except (TypeError, ValueError) as error:
        raise NativeProfileConvergenceCLIError(
            "method declarations cannot be represented as canonical JSON"
        ) from error
    if method_set["sha256"] != actual_method_set_sha:
        raise NativeProfileConvergenceCLIError(
            "method_set.sha256 does not match the ordered declarations"
        )
    method_ids: list[str] = []
    for index, raw in enumerate(methods):
        if not isinstance(raw, dict):
            raise NativeProfileConvergenceCLIError(
                f"method_set.methods[{index}] must be an object"
            )
        role = raw.get("role")
        expected = {"method_id", "role", "prediction_artifact_sha256"}
        if role == "trained_model_checkpoint":
            expected.update({"model_id", "checkpoint_id"})
        if role not in _METHOD_ROLES or set(raw) != expected:
            raise NativeProfileConvergenceCLIError(
                f"method_set.methods[{index}] has invalid role or keys"
            )
        method_id = raw.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            raise NativeProfileConvergenceCLIError(
                f"method_set.methods[{index}].method_id must be non-empty"
            )
        method_ids.append(method_id)
    if len(method_ids) != len(set(method_ids)):
        raise NativeProfileConvergenceCLIError("method IDs contain duplicates")

    raw_prediction_sets = root["prediction_manifests"]
    if not isinstance(raw_prediction_sets, list) or len(raw_prediction_sets) != len(
        method_ids
    ):
        raise NativeProfileConvergenceCLIError(
            "prediction_manifests must contain one ordered block per method"
        )
    paths_by_method: dict[str, tuple[Path, ...]] = {}
    for method_position, (raw_set, expected_method_id) in enumerate(
        zip(raw_prediction_sets, method_ids, strict=True)
    ):
        block = _exact_keys(
            raw_set,
            {"method_id", "case_manifests"},
            f"prediction_manifests[{method_position}]",
        )
        if block["method_id"] != expected_method_id:
            raise NativeProfileConvergenceCLIError(
                "prediction manifest method order differs from method_set"
            )
        raw_cases = block["case_manifests"]
        if not isinstance(raw_cases, list) or len(raw_cases) != len(case_order):
            raise NativeProfileConvergenceCLIError(
                f"{expected_method_id} must provide one manifest per case"
            )
        paths: list[Path] = []
        for case_position, (raw_case, expected_case_id) in enumerate(
            zip(raw_cases, case_order, strict=True)
        ):
            case_record = _exact_keys(
                raw_case,
                {"case_id", "manifest"},
                f"{expected_method_id} case_manifests[{case_position}]",
            )
            if case_record["case_id"] != expected_case_id:
                raise NativeProfileConvergenceCLIError(
                    f"{expected_method_id} case manifest order differs from case_order"
                )
            paths.append(
                _resolve_input_path(
                    case_record["manifest"],
                    root=config_path.parent,
                    label=f"{expected_method_id}/{expected_case_id} manifest",
                )
            )
        paths_by_method[expected_method_id] = tuple(paths)
    return value, byte_sha256, case_order, methods, paths_by_method


def _mapping_from_validated_case(
    case_directory: Path,
    *,
    case_summary: Mapping[str, Any],
    source_part_sha256: tuple[str, ...],
) -> CaseResolutionMappings:
    """Extract rows only after the strict aggregate replay has accepted bytes."""

    case_id = _case_id(case_summary.get("case_id"), "aggregate case_id")
    geometry = case_summary.get("geometry")
    if not isinstance(geometry, Mapping):
        raise NativeProfileConvergenceCLIError("aggregate geometry is missing")
    native_cell_count = geometry.get("cell_count")
    if (
        not isinstance(native_cell_count, int)
        or isinstance(native_cell_count, bool)
        or native_cell_count < 1
    ):
        raise NativeProfileConvergenceCLIError(
            "aggregate native cell count is invalid"
        )
    raw_summaries = case_summary.get("resolutions")
    if not isinstance(raw_summaries, list) or len(raw_summaries) != 4:
        raise NativeProfileConvergenceCLIError(
            "aggregate case must contain four resolution summaries"
        )
    resolutions: list[ResolutionMapping] = []
    for spacing_mm, summary in zip((1, 2, 5, 10), raw_summaries, strict=True):
        if not isinstance(summary, Mapping) or summary.get(
            "nominal_spacing_mm"
        ) != spacing_mm:
            raise NativeProfileConvergenceCLIError(
                "aggregate resolution order is not exact"
            )
        artifact_name = MAPPING_ARTIFACT_NAMES[spacing_mm]
        if summary.get("artifact_name") != artifact_name:
            raise NativeProfileConvergenceCLIError(
                "aggregate mapping artifact name differs from resolution"
            )
        artifact_path = case_directory / artifact_name
        artifact, artifact_sha256 = _read_retained_json(
            artifact_path,
            label=f"{case_id} {spacing_mm} mm validated mapping artifact",
            maximum_bytes=128 * 1024 * 1024,
        )
        if artifact_sha256 != summary.get("artifact_sha256"):
            raise NativeProfileConvergenceCLIError(
                f"{case_id} {spacing_mm} mm artifact changed after strict replay"
            )
        raw_rows = artifact.get("rows")
        if not isinstance(raw_rows, list):
            raise NativeProfileConvergenceCLIError("mapping rows must be an array")
        rows: list[ProfileMappingRow] = []
        for position, raw in enumerate(raw_rows):
            if not isinstance(raw, list) or len(raw) != 8:
                raise NativeProfileConvergenceCLIError(
                    f"mapping row {position} has the wrong shape"
                )
            point = raw[2]
            if not isinstance(point, list) or len(point) != 3:
                raise NativeProfileConvergenceCLIError(
                    f"mapping row {position} point has the wrong shape"
                )
            try:
                rows.append(
                    ProfileMappingRow(
                        profile_id=raw[0],
                        sample_index=raw[1],
                        point_m=tuple(float(value) for value in point),
                        distance_m=raw[3],
                        valid=raw[4],
                        reason=raw[5],
                        raw_vtk_cell_id=raw[6],
                        candidate_count=raw[7],
                    )
                )
            except (TypeError, ValueError) as error:
                raise NativeProfileConvergenceCLIError(
                    f"mapping row {position} cannot be reconstructed"
                ) from error
        resolutions.append(
            ResolutionMapping(
                case_id=case_id,
                spacing_mm=spacing_mm,
                native_cell_count=native_cell_count,
                source_part_sha256=source_part_sha256,
                artifact_name=artifact_name,
                artifact_sha256=artifact_sha256,
                assignment_evidence_sha256=summary.get(
                    "assignment_evidence_sha256"
                ),
                rows=tuple(rows),
            )
        )
    return CaseResolutionMappings(case_id, tuple(resolutions))


def _native_truth_for_case(
    *,
    pin: Any,
    case_id: str,
    dataset_root: Path,
    mappings: CaseResolutionMappings,
    io_chunk_bytes: int,
) -> SparseNativeField:
    pinned_case = pin.case(case_id)
    resolved = pin.resolve(case_id, dataset_root)
    required_ids = mappings.required_raw_cell_ids()
    with closing(
        open_verified_multipart(
            resolved, verification_chunk_size=io_chunk_bytes
        )
    ) as stream:
        vtk_index = index_inline_binary_vtk_xml(
            stream, scan_chunk_size=io_chunk_bytes
        )
        if (
            vtk_index.dataset_type != "UnstructuredGrid"
            or len(vtk_index.pieces) != 1
            or vtk_index.pieces[0].number_of_cells != mappings.native_cell_count
        ):
            raise NativeProfileConvergenceCLIError(
                f"{case_id} native volume geometry differs from mappings"
            )
        arrays = vtk_index.arrays_for(association="CellData", name=FIELD_NAME)
        if len(arrays) != 1:
            raise NativeProfileConvergenceCLIError(
                f"{case_id} native volume must have exactly one CellData {FIELD_NAME}"
            )
        try:
            truth, _ = gather_sparse_inline_native_field(
                stream,
                vtk_index,
                arrays[0],
                required_ids,
                case_id=case_id,
                support_id=SUPPORT_ID,
                field_name=FIELD_NAME,
                expected_components=3,
                source_files=tuple(
                    part.path.name for part in pinned_case.volume_parts
                ),
                source_sha256=tuple(
                    part.sha256 for part in pinned_case.volume_parts
                ),
                encoded_chunk_bytes=io_chunk_bytes,
            )
        except DrivAerDiagnosticEvaluatorError as error:
            raise NativeProfileConvergenceCLIError(str(error)) from error
    return truth


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the complete case-streaming real-data study."""

    runtime = _validated_runtime_binding()
    implementation_start = _implementation_binding()
    if (
        implementation_start["git_revision"] is None
        or implementation_start["git_worktree_clean"] is not True
    ):
        raise NativeProfileConvergenceCLIError(
            "real-data convergence evidence requires a clean Git checkout "
            "with a resolved 40-character revision"
        )
    if OFFICIAL_CASE_IDS != OFFICIAL_CASE_ORDER:
        raise NativeProfileConvergenceCLIError(
            "profile checker and velocity aggregator official case orders drifted"
        )
    (
        config,
        config_byte_sha256,
        case_order,
        methods,
        manifest_paths,
    ) = load_study_config(args.input)
    method_set = config["method_set"]
    method_ids = [method["method_id"] for method in methods]

    # Bind all method artifacts before any multi-gigabyte native source read.
    parsed_by_method: dict[str, tuple[PredictionChunkManifest, ...]] = {}
    prediction_bindings: list[dict[str, object]] = []
    for method in methods:
        method_id = method["method_id"]
        try:
            digest, parsed, records = prediction_manifest_set_identity(
                case_order, manifest_paths[method_id]
            )
        except (NativeProfileConvergenceError, PredictionChunkError) as error:
            raise NativeProfileConvergenceCLIError(str(error)) from error
        if digest != method["prediction_artifact_sha256"]:
            raise NativeProfileConvergenceCLIError(
                f"{method_id} prediction artifact pin mismatch: "
                f"declared {method['prediction_artifact_sha256']}, actual {digest}"
            )
        parsed_by_method[method_id] = parsed
        prediction_bindings.append(
            {
                "method_id": method_id,
                "role": method["role"],
                "prediction_artifact_sha256": digest,
                "case_manifests": records,
            }
        )

    native_source_pin = args.native_source_pin.expanduser().resolve()
    autocfd5_profile = args.autocfd5_profile.expanduser().resolve()
    pilot_cases = () if tuple(case_order) == OFFICIAL_CASE_IDS else case_order
    try:
        mapping_aggregate = aggregate_velocity_assignments(
            receipts_root=args.mappings_root,
            native_source_pin=native_source_pin,
            autocfd5_profile=autocfd5_profile,
            pilot_case_ids=pilot_cases,
            expected_pin_sha256=OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
        )
    except VelocityAssignmentAggregateError as error:
        raise NativeProfileConvergenceCLIError(str(error)) from error
    mapping_by_case = {
        item["case_id"]: item for item in mapping_aggregate["cases"]
    }

    try:
        pin = load_native_source_pin(native_source_pin)
        pin_sha256 = validate_native_source_contract(pin)
        definition = load_autocfd5_definition(autocfd5_profile)
    except (NativeSourceError, ValueError) as error:
        raise NativeProfileConvergenceCLIError(str(error)) from error
    if pin_sha256 != OFFICIAL_NATIVE_SOURCE_PIN_SHA256:
        raise NativeProfileConvergenceCLIError(
            "native-source pin differs from immutable public release pin"
        )

    losses_by_method: dict[str, list[list[list[float]]]] = {
        method_id: [] for method_id in method_ids
    }
    audits_by_method: dict[str, list[dict[str, object]]] = {
        method_id: [] for method_id in method_ids
    }
    geometry_evidence: list[dict[str, object]] = []
    mappings_root = args.mappings_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    for case_index, case_id in enumerate(case_order):
        pinned_case = pin.case(case_id)
        mapping = _mapping_from_validated_case(
            mappings_root / case_id,
            case_summary=mapping_by_case[case_id],
            source_part_sha256=tuple(
                part.sha256 for part in pinned_case.volume_parts
            ),
        )
        geometry = validate_mapping_grids(mapping, definition)
        geometry_evidence.append(geometry)
        truth = _native_truth_for_case(
            pin=pin,
            case_id=case_id,
            dataset_root=dataset_root,
            mappings=mapping,
            io_chunk_bytes=args.io_chunk_bytes,
        )
        required_ids = mapping.required_raw_cell_ids()
        for method_id in method_ids:
            manifest = parsed_by_method[method_id][case_index]
            if manifest.total_row_count != mapping.native_cell_count:
                raise NativeProfileConvergenceCLIError(
                    f"{method_id}/{case_id} native cell count differs from mappings"
                )
            try:
                prediction = gather_mapped_prediction_field(
                    manifest,
                    required_ids,
                    case_id=case_id,
                    support_id=SUPPORT_ID,
                    field_name=FIELD_NAME,
                    expected_total_row_count=mapping.native_cell_count,
                    maximum_chunk_rows=args.maximum_prediction_chunk_rows,
                    hash_chunk_bytes=args.io_chunk_bytes,
                    validation_block_rows=args.maximum_prediction_chunk_rows,
                )
                case_losses, audit = case_profile_losses(
                    mapping,
                    definition=definition,
                    native_truth=truth,
                    prediction=prediction,
                    validated_geometry=geometry,
                )
            except (
                DrivAerDiagnosticEvaluatorError,
                NativeProfileConvergenceError,
                PredictionChunkError,
            ) as error:
                raise NativeProfileConvergenceCLIError(
                    f"{method_id}/{case_id}: {error}"
                ) from error
            audit["prediction_manifest"] = prediction.audit_record()
            audit["native_truth"] = truth.audit_record()
            losses_by_method[method_id].append(case_losses)
            audits_by_method[method_id].append(audit)
        # Explicitly release all case-local rows and selected truth before the
        # next 100+ GB logical VTU is verified and decoded.
        del truth, mapping, required_ids

    result = finalize_profile_convergence_evidence(
        study_id=config["study_id"],
        method_set=method_set,
        case_order=case_order,
        losses=[losses_by_method[method_id] for method_id in method_ids],
        prediction_bindings=prediction_bindings,
        geometric_case_evidence=geometry_evidence,
        method_audits=[
            {
                "method_id": method_id,
                "case_count": len(case_order),
                "cases": audits_by_method[method_id],
            }
            for method_id in method_ids
        ],
        contract_proposal=args.contract_proposal,
    )
    implementation_end = _implementation_binding()
    if implementation_end != implementation_start:
        raise NativeProfileConvergenceCLIError(
            "evaluator source files or Git state changed during the study"
        )
    result["execution_input"] = {
        "config_schema": CONFIG_SCHEMA,
        "config_file_sha256": config_byte_sha256,
        "config_canonical_json_sha256": canonical_sha256(config),
        "native_source_pin_sha256": pin_sha256,
        "mapping_aggregate_canonical_json_sha256": canonical_sha256(
            mapping_aggregate
        ),
        "mapping_aggregate_mode": mapping_aggregate["mode"],
        "mapping_aggregate_status": mapping_aggregate["status"],
        "actual_native_multipart_vtu_truth_streamed": True,
        "case_streaming_bounded_memory": True,
        "execution_limits": {
            "io_chunk_bytes": args.io_chunk_bytes,
            "maximum_prediction_chunk_rows": (
                args.maximum_prediction_chunk_rows
            ),
        },
        "runtime": runtime,
        "implementation": implementation_start,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mappings-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--native-source-pin", type=Path, default=DEFAULT_NATIVE_SOURCE_PIN
    )
    parser.add_argument(
        "--autocfd5-profile", type=Path, default=DEFAULT_AUTOCFD5_PROFILE
    )
    parser.add_argument(
        "--contract-proposal", type=Path, default=DEFAULT_CONTRACT_PROPOSAL
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-prediction-chunk-rows",
        type=_positive_integer,
        default=1_000_000,
    )
    parser.add_argument(
        "--io-chunk-bytes", type=_positive_integer, default=16 * 1024 * 1024
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
        identity = write_evidence(result, args.output)
    except (
        NativeProfileConvergenceCLIError,
        NativeProfileConvergenceError,
        NativeSourceError,
        PredictionChunkError,
        RetainedFileError,
        VelocityAssignmentAggregateError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "profile_resolution_candidate_eligible_for_owner_review": result[
                    "profile_resolution_candidate_eligible_for_owner_review"
                ],
                "does_not_activate_scoring_contract": True,
                "output": identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return (
        0
        if result["profile_resolution_candidate_eligible_for_owner_review"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
