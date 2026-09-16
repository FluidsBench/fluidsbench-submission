#!/usr/bin/env python3
"""Evaluate one closed-candidate DrivAerML case from native prediction chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.evaluator import (  # noqa: E402
    DrivAerCandidateEvaluatorError,
    evaluate_candidate_case,
    evaluate_surface_only_candidate_case,
    validate_native_source_contract,
    write_candidate_case_evidence,
)
from reference.drivaerml.native_surface import (  # noqa: E402
    DrivAerNativeSurfaceError,
    audit_fixed_surface_area_file,
    load_native_surface_vtp,
)
from reference.drivaerml.prediction_chunks import PredictionChunkError  # noqa: E402
from reference.drivaerml.source import (  # noqa: E402
    NativeSourceError,
    index_inline_binary_vtk_xml,
    load_native_source_pin,
    open_verified_monolithic,
    open_verified_multipart,
)


IMPLEMENTATION_RECEIPT_SCHEMA = (
    "drivaerml-candidate-case-evaluation-implementation-receipt-v1"
)
IMPLEMENTATION_RECEIPT_STATUS = (
    "complete_commit_bound_candidate_evaluation_not_official_submission"
)
CORE_EVALUATOR_GIT_PATHS = (
    "scripts/evaluate_drivaerml_candidate_case.py",
    "reference/drivaerml/__init__.py",
    "reference/drivaerml/accumulators.py",
    "reference/drivaerml/evaluator.py",
    "reference/drivaerml/native_fields.py",
    "reference/drivaerml/native_surface.py",
    "reference/drivaerml/prediction_chunks.py",
    "reference/drivaerml/regional_aggregate.py",
    "reference/drivaerml/regional_diagnostics.py",
    "reference/drivaerml/retained_file.py",
    "reference/drivaerml/source.py",
    "reference/drivaerml/surface_forces.py",
    "reference/drivaerml/volume_regions.py",
    "benchmark-specs/drivaerml/regional-diagnostics-v1.json",
)


class CandidateImplementationBindingError(ValueError):
    """Raised when a full-case replay cannot be bound to committed code."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_bytes(path: Path, label: str) -> bytes:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise CandidateImplementationBindingError(
            f"{label} cannot be a symbolic link"
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise CandidateImplementationBindingError(
            f"cannot resolve {label}: {error}"
        ) from error
    if not resolved.is_file():
        raise CandidateImplementationBindingError(
            f"{label} must be a regular file"
        )
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise CandidateImplementationBindingError(
            f"cannot read {label}: {error}"
        ) from error


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
        raise CandidateImplementationBindingError(
            f"git {' '.join(arguments)} failed: {detail}"
        ) from error


def _implementation_binding(revision: str) -> tuple[str, tuple[dict[str, object], ...]]:
    """Resolve a reachable commit and require exact local implementation bytes."""

    if not isinstance(revision, str) or not revision:
        raise CandidateImplementationBindingError(
            "evaluator Git revision must be non-empty"
        )
    commit = (
        _git("rev-parse", "--verify", f"{revision}^{{commit}}")
        .stdout.decode("ascii")
        .strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise CandidateImplementationBindingError(
            "evaluator Git revision did not resolve to a full commit"
        )
    if _git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode != 0:
        raise CandidateImplementationBindingError(
            "evaluator Git revision is not reachable from checkout HEAD"
        )
    records: list[dict[str, object]] = []
    for relative_path in CORE_EVALUATOR_GIT_PATHS:
        committed = _git("show", f"{commit}:{relative_path}").stdout
        local = _read_regular_bytes(
            ROOT / relative_path, f"current evaluator file {relative_path}"
        )
        if committed != local:
            raise CandidateImplementationBindingError(
                f"current {relative_path} bytes differ from evaluator commit {commit}"
            )
        records.append(
            {
                "path": relative_path,
                "sha256": _sha256_bytes(local),
                "size_bytes": len(local),
            }
        )
    return commit, tuple(records)


def _assert_implementation_unchanged(
    commit: str, expected: tuple[dict[str, object], ...]
) -> None:
    replay_commit, replay = _implementation_binding(commit)
    if replay_commit != commit or replay != expected:
        raise CandidateImplementationBindingError(
            "evaluator implementation changed during the full-case replay"
        )


def _evidence_identity(
    path: Path, *, case_id: str, expected: dict[str, object]
) -> dict[str, object]:
    payload = _read_regular_bytes(path, "candidate case evidence")
    digest = _sha256_bytes(payload)
    if (
        expected.get("file") != path.name
        or expected.get("sha256") != digest
        or expected.get("byte_size") != len(payload)
    ):
        raise CandidateImplementationBindingError(
            "candidate case evidence differs from its evaluator writer identity"
        )

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise CandidateImplementationBindingError(
                    f"candidate case evidence contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CandidateImplementationBindingError(
                    f"candidate case evidence contains non-finite token {token}"
                )
            ),
        )
    except CandidateImplementationBindingError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CandidateImplementationBindingError(
            "candidate case evidence is not strict UTF-8 JSON"
        ) from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != "drivaerml-candidate-case-evaluation-v4"
        or document.get("schema_version") != 4
        or document.get("case_id") != case_id
        or document.get("official_submission") is not False
    ):
        raise CandidateImplementationBindingError(
            "candidate case evidence schema or case identity differs"
        )
    return {
        "file": path.name,
        "sha256": digest,
        "size_bytes": len(payload),
        "schema": document["schema"],
        "schema_version": document["schema_version"],
    }


def _runtime_binding(evidence_path: Path) -> dict[str, str]:
    document = json.loads(evidence_path.read_bytes())
    vtk_version = document.get("source", {}).get("surface_native", {}).get(
        "vtk_version"
    )
    if not isinstance(vtk_version, str) or not vtk_version:
        raise CandidateImplementationBindingError(
            "candidate case evidence does not record the VTK runtime version"
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "vtk": vtk_version,
    }


def _write_implementation_receipt(
    path: Path,
    *,
    case_id: str,
    commit: str,
    implementation: tuple[dict[str, object], ...],
    evidence: dict[str, object],
    runtime: dict[str, str],
) -> dict[str, object]:
    destination = path.expanduser()
    if destination.suffix.lower() != ".json":
        raise CandidateImplementationBindingError(
            "implementation receipt output must use the .json suffix"
        )
    if destination.exists() or destination.is_symlink():
        raise CandidateImplementationBindingError(
            f"refusing to overwrite implementation receipt: {destination}"
        )
    receipt = {
        "schema": IMPLEMENTATION_RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": IMPLEMENTATION_RECEIPT_STATUS,
        "official_submission": False,
        "case_id": case_id,
        "evaluator": {
            "repository": "neilashton/fluidsbench-submission",
            "git_revision": commit,
            "implementation": list(implementation),
            "preflight_verified": True,
            "postflight_verified": True,
        },
        "evidence": evidence,
        "runtime": runtime,
    }
    payload = (
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except OSError as error:
        raise CandidateImplementationBindingError(
            f"cannot create implementation receipt: {error}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "file": destination.name,
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one DrivAerML case with the closed candidate evaluator. "
            "This does not create or validate an official submission."
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--native-source-pin", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing the pinned run_N native files or multipart files.",
    )
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--monolithic-vtu",
        type=Path,
        help="Verified byte reconstruction; defaults to the pinned logical path.",
    )
    transport.add_argument(
        "--multipart",
        action="store_true",
        help="Open and concatenate the pinned .part files without materializing a VTU.",
    )
    parser.add_argument("--surface-area-npy", type=Path, required=True)
    parser.add_argument(
        "--prediction-scope",
        choices=("surface_and_volume", "surface_only"),
        default="surface_and_volume",
        help=(
            "surface_only evaluates only complete native-surface predictions; "
            "the volume prediction and velocity-profile components remain zero."
        ),
    )
    parser.add_argument("--surface-prediction-manifest", type=Path, required=True)
    parser.add_argument("--volume-prediction-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluator-git-revision",
        help=(
            "Reachable commit whose evaluator bytes are bound to this replay. "
            "Must be supplied together with --implementation-receipt."
        ),
    )
    parser.add_argument(
        "--implementation-receipt",
        type=Path,
        help=(
            "Exclusive path for a deterministic evidence-to-commit receipt. "
            "Must be supplied together with --evaluator-git-revision."
        ),
    )
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
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    requested_revision = getattr(args, "evaluator_git_revision", None)
    receipt_path = getattr(args, "implementation_receipt", None)
    if (requested_revision is None) != (receipt_path is None):
        raise CandidateImplementationBindingError(
            "--evaluator-git-revision and --implementation-receipt must be "
            "supplied together"
        )
    if (
        args.prediction_scope == "surface_and_volume"
        and args.volume_prediction_manifest is None
    ):
        raise DrivAerCandidateEvaluatorError(
            "surface_and_volume scope requires --volume-prediction-manifest"
        )
    if (
        args.prediction_scope == "surface_only"
        and args.volume_prediction_manifest is not None
    ):
        raise DrivAerCandidateEvaluatorError(
            "surface_only scope must not receive --volume-prediction-manifest"
        )
    implementation_binding: tuple[
        str, tuple[dict[str, object], ...]
    ] | None = None
    if requested_revision is not None:
        assert receipt_path is not None
        if receipt_path.exists() or receipt_path.is_symlink():
            raise CandidateImplementationBindingError(
                f"refusing to overwrite implementation receipt: {receipt_path}"
            )
        implementation_binding = _implementation_binding(requested_revision)

    pin = load_native_source_pin(args.native_source_pin)
    validate_native_source_contract(pin)
    case = pin.case(args.case_id)
    resolved = pin.resolve(args.case_id, args.dataset_root)
    surface = load_native_surface_vtp(resolved.boundary_path)
    areas = audit_fixed_surface_area_file(
        surface,
        args.surface_area_npy,
        expected_area_sha256=case.surface_cell_area.sha256,
        source_boundary_sha256=case.surface_cell_area.source_boundary_sha256,
    )
    if args.prediction_scope == "surface_only":
        if args.multipart or args.monolithic_vtu is not None:
            raise DrivAerCandidateEvaluatorError(
                "surface_only scope must not receive a native volume transport"
            )
        evaluation = evaluate_surface_only_candidate_case(
            case_id=args.case_id,
            native_source_pin=pin,
            native_surface=surface,
            fixed_surface_areas=areas,
            surface_prediction_manifest=args.surface_prediction_manifest,
            maximum_prediction_chunk_rows=args.maximum_prediction_chunk_rows,
            hash_chunk_bytes=args.io_chunk_bytes,
            validation_block_rows=args.maximum_prediction_chunk_rows,
        )
    else:
        if args.multipart:
            volume_stream = open_verified_multipart(
                resolved,
                verification_chunk_size=args.io_chunk_bytes,
            )
        else:
            volume_stream = open_verified_monolithic(
                resolved,
                args.monolithic_vtu,
                verification_chunk_size=args.io_chunk_bytes,
            )
        with closing(volume_stream) as stream:
            vtk_index = index_inline_binary_vtk_xml(
                stream,
                scan_chunk_size=args.io_chunk_bytes,
            )
            if len(vtk_index.pieces) != 1:
                raise DrivAerCandidateEvaluatorError(
                    "native DrivAerML volume must contain exactly one Piece"
                )
            evaluation = evaluate_candidate_case(
                case_id=args.case_id,
                native_source_pin=pin,
                native_surface=surface,
                fixed_surface_areas=areas,
                volume_stream=stream,
                volume_vtk_index=vtk_index,
                surface_prediction_manifest=args.surface_prediction_manifest,
                volume_prediction_manifest=args.volume_prediction_manifest,
                maximum_prediction_chunk_rows=args.maximum_prediction_chunk_rows,
                hash_chunk_bytes=args.io_chunk_bytes,
                validation_block_rows=args.maximum_prediction_chunk_rows,
                encoded_chunk_bytes=args.io_chunk_bytes,
            )
    identity = write_candidate_case_evidence(evaluation, args.output)
    if implementation_binding is None:
        return identity

    commit, implementation = implementation_binding
    evidence_path = args.output.expanduser().resolve(strict=True)
    evidence = _evidence_identity(
        evidence_path,
        case_id=args.case_id,
        expected=identity,
    )
    runtime = _runtime_binding(evidence_path)
    _assert_implementation_unchanged(commit, implementation)
    # Re-read the evidence immediately before publication so the receipt never
    # binds a stale pre-mutation identity.
    if (
        _evidence_identity(
            evidence_path,
            case_id=args.case_id,
            expected=identity,
        )
        != evidence
    ):
        raise CandidateImplementationBindingError(
            "candidate case evidence changed before receipt publication"
        )
    receipt_identity = _write_implementation_receipt(
        receipt_path,
        case_id=args.case_id,
        commit=commit,
        implementation=implementation,
        evidence=evidence,
        runtime=runtime,
    )
    try:
        _assert_implementation_unchanged(commit, implementation)
        if (
            _evidence_identity(
                evidence_path,
                case_id=args.case_id,
                expected=identity,
            )
            != evidence
            or _sha256_bytes(
                _read_regular_bytes(
                    receipt_path, "candidate implementation receipt"
                )
            )
            != receipt_identity["sha256"]
        ):
            raise CandidateImplementationBindingError(
                "evidence or receipt changed during receipt publication"
            )
    except BaseException:
        receipt_path.unlink(missing_ok=True)
        raise
    return {**identity, "implementation_receipt": receipt_identity}


def main() -> int:
    args = parse_args()
    try:
        identity = run(args)
    except (
        DrivAerCandidateEvaluatorError,
        DrivAerNativeSurfaceError,
        NativeSourceError,
        PredictionChunkError,
        CandidateImplementationBindingError,
    ) as error:
        raise SystemExit(f"candidate evaluation failed: {error}") from error
    print(
        f"PASS {args.case_id}: {identity['byte_size']} evidence bytes, "
        f"sha256={identity['sha256']}"
    )
    if "implementation_receipt" in identity:
        receipt = identity["implementation_receipt"]
        print(
            f"BOUND {receipt['file']}: {receipt['size_bytes']} bytes, "
            f"sha256={receipt['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
