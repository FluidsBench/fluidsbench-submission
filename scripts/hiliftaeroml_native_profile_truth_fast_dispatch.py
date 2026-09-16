#!/usr/bin/env python3
"""Fail-closed worker for the eight-node HiLift profile-truth campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.native_profile_truth import (  # noqa: E402
    NativeProfileTruthError,
    load_case_universe,
)
from reference.hiliftaeroml.native_profile_truth_materializer import (  # noqa: E402
    load_prerequisite_authority_index,
    materialize_case_truth,
)

EXPECTED_CASE_COUNT = 1355
EXPECTED_CASE_SET_COUNT = 8
EXPECTED_TASK_COUNT = 8
EXPECTED_CPUS_PER_TASK = 2
APPROVAL_SENTINEL = "YES_OWNER_APPROVED_FAST_PROFILE_TRUTH_1355"
PREFLIGHT_SCHEMA = (
    "hiliftaeroml-native-profile-truth-prerequisite-preflight-v1-candidate"
)
VALIDATION_STATUS = "pass"
SHA256 = re.compile(r"[0-9a-f]{64}")
SUPPORT_MANIFEST = (
    ROOT
    / "benchmark-specs"
    / "hiliftaeroml"
    / "scoring-support"
    / "hiliftaeroml-native-all-splits-support-v1-candidate"
    / "manifest.json"
)


def striped_indices(
    rank: int,
    *,
    task_count: int = EXPECTED_TASK_COUNT,
    case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[int, ...]:
    """Return rank, rank + task_count, ... over the frozen global universe."""

    if task_count != EXPECTED_TASK_COUNT:
        raise NativeProfileTruthError(
            f"dispatcher task count must be {EXPECTED_TASK_COUNT}"
        )
    if case_count != EXPECTED_CASE_COUNT:
        raise NativeProfileTruthError(
            f"dispatcher case count must be {EXPECTED_CASE_COUNT}"
        )
    if rank < 0 or rank >= task_count:
        raise NativeProfileTruthError("dispatcher rank is out of range")
    return tuple(range(rank, case_count, task_count))


def _absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise NativeProfileTruthError(f"{label} must be absolute")
    return path


def _regular_json(path: Path, label: str) -> dict[str, Any]:
    _absolute(path, label)
    if path.is_symlink() or not path.is_file():
        raise NativeProfileTruthError(f"{label} must be a regular non-symlink file")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise NativeProfileTruthError(f"{label} must contain a JSON object")
    return body


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_sha(value: str) -> str:
    if SHA256.fullmatch(value) is None:
        raise NativeProfileTruthError("pinned authority SHA-256 is invalid")
    return value


def _require_approval() -> None:
    if os.environ.get("HILIFT_TRUTH_FAST_APPROVED") != APPROVAL_SENTINEL:
        raise NativeProfileTruthError("fast truth approval sentinel differs")


def _load_ready_sources(
    *, authority_index: Path, preflight_receipt: Path, pinned_sha256: str
) -> tuple[tuple[str, ...], dict[str, Any]]:
    _require_approval()
    _absolute(authority_index, "authority index")
    if authority_index.is_symlink() or not authority_index.is_file():
        raise NativeProfileTruthError(
            "authority index must be a regular non-symlink file"
        )
    pinned_sha256 = _pinned_sha(pinned_sha256)
    observed_sha256 = _sha256_file(authority_index)
    if observed_sha256 != pinned_sha256:
        raise NativeProfileTruthError("authority-index SHA-256 differs")

    universe = load_case_universe(SUPPORT_MANIFEST)
    if (
        len(universe.case_ids) != EXPECTED_CASE_COUNT
        or len(universe.case_sets) != EXPECTED_CASE_SET_COUNT
    ):
        raise NativeProfileTruthError("frozen case-universe cardinality differs")
    authority, loader_sha256 = load_prerequisite_authority_index(
        authority_index, universe=universe, require_complete=True
    )
    if (
        loader_sha256 != pinned_sha256
        or authority.get("status") != "complete"
        or authority.get("expected_case_count") != EXPECTED_CASE_COUNT
        or authority.get("available_case_count") != EXPECTED_CASE_COUNT
        or authority.get("missing_case_count") != 0
        or authority.get("missing") != {}
        or authority.get("prediction_bearing_evaluator_outputs_used_as_source")
        is not False
    ):
        raise NativeProfileTruthError("complete authority contract differs")

    receipt = _regular_json(preflight_receipt, "preflight receipt")
    if (
        receipt.get("schema") != PREFLIGHT_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("authority_index_sha256") != pinned_sha256
        or receipt.get("case_count") != EXPECTED_CASE_COUNT
        or receipt.get("case_set_count") != EXPECTED_CASE_SET_COUNT
        or receipt.get("available_case_count") != EXPECTED_CASE_COUNT
        or receipt.get("missing_case_count") != 0
        or receipt.get("missing") != {}
        or receipt.get("activation_changed") is not False
        or receipt.get("owner_approval_complete") is not False
        or receipt.get("published") is not False
        or receipt.get("source_io_contract", {}).get("model_inference") is not False
        or receipt.get("source_io_contract", {}).get("prediction_outputs_as_authority")
        is not False
    ):
        raise NativeProfileTruthError("ready preflight receipt differs")
    return universe.case_ids, authority


def _resolved_without_strict(path: Path) -> Path:
    return path.resolve(strict=False)


def _validate_output_boundary(
    *, output_dir: Path, validation_receipt: Path, require_fresh: bool
) -> None:
    _absolute(output_dir, "output directory")
    _absolute(validation_receipt, "validation receipt")
    output_resolved = _resolved_without_strict(output_dir)
    receipt_resolved = _resolved_without_strict(validation_receipt)
    if receipt_resolved == output_resolved or receipt_resolved.is_relative_to(
        output_resolved
    ):
        raise NativeProfileTruthError(
            "validation receipt must be outside the release output"
        )
    if output_dir.is_symlink():
        raise NativeProfileTruthError("release output must not be a symlink")
    if require_fresh:
        if output_dir.exists():
            raise NativeProfileTruthError("isolated release output already exists")
    elif output_dir.exists() and not output_dir.is_dir():
        raise NativeProfileTruthError("release output is not a directory")
    if validation_receipt.exists() or validation_receipt.is_symlink():
        raise NativeProfileTruthError("validation receipt already exists")


def _slurm_rank() -> int:
    try:
        rank = int(os.environ["SLURM_PROCID"])
        task_count = int(os.environ["SLURM_NTASKS"])
        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, ValueError) as error:
        raise NativeProfileTruthError(
            "striped worker requires exact Slurm allocation metadata"
        ) from error
    if (
        task_count != EXPECTED_TASK_COUNT
        or node_count != EXPECTED_TASK_COUNT
        or cpus_per_task != EXPECTED_CPUS_PER_TASK
    ):
        raise NativeProfileTruthError("striped Slurm allocation differs")
    striped_indices(rank)
    return rank


def _validation_receipt(path: Path) -> dict[str, Any]:
    receipt = _regular_json(path, "validation receipt")
    if (
        receipt.get("status") != VALIDATION_STATUS
        or receipt.get("case_count") != EXPECTED_CASE_COUNT
        or receipt.get("case_set_count") != EXPECTED_CASE_SET_COUNT
        or receipt.get("source_replay_complete") is not True
    ):
        raise NativeProfileTruthError("full source-replay validation receipt differs")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode", required=True, choices=("preflight", "generate", "check", "receipt")
    )
    result.add_argument("--authority-index", required=True, type=Path)
    result.add_argument("--authority-sha256", required=True)
    result.add_argument("--preflight-receipt", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--validation-receipt", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        case_ids, authority = _load_ready_sources(
            authority_index=args.authority_index,
            preflight_receipt=args.preflight_receipt,
            pinned_sha256=args.authority_sha256,
        )
        if args.mode == "preflight":
            _validate_output_boundary(
                output_dir=args.output_dir,
                validation_receipt=args.validation_receipt,
                require_fresh=True,
            )
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "case_count": len(case_ids),
                        "case_set_count": EXPECTED_CASE_SET_COUNT,
                        "task_count": EXPECTED_TASK_COUNT,
                        "authority_index_sha256": args.authority_sha256,
                        "isolated_output_absent": True,
                        "publication_or_activation_changed": False,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.mode == "receipt":
            _validate_output_boundary_after_validation(
                output_dir=args.output_dir,
                validation_receipt=args.validation_receipt,
            )
            print(
                json.dumps(_validation_receipt(args.validation_receipt), sort_keys=True)
            )
            return 0

        _validate_output_boundary(
            output_dir=args.output_dir,
            validation_receipt=args.validation_receipt,
            require_fresh=False,
        )
        rank = _slurm_rank()
        indices = striped_indices(rank)
        check = args.mode == "check"
        for case_index in indices:
            case_id = case_ids[case_index]
            try:
                materialize_case_truth(
                    case_id=case_id,
                    authority=authority,
                    output_root=args.output_dir,
                    check=check,
                )
            except (NativeProfileTruthError, OSError, ValueError) as error:
                raise NativeProfileTruthError(
                    f"rank {rank} case index {case_index} ({case_id}) failed: {error}"
                ) from error
        print(
            json.dumps(
                {
                    "status": "pass" if check else "complete",
                    "mode": args.mode,
                    "rank": rank,
                    "task_count": EXPECTED_TASK_COUNT,
                    "case_count": len(indices),
                    "first_case_index": indices[0],
                    "last_case_index": indices[-1],
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        NativeProfileTruthError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def _validate_output_boundary_after_validation(
    *, output_dir: Path, validation_receipt: Path
) -> None:
    _absolute(output_dir, "output directory")
    _absolute(validation_receipt, "validation receipt")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise NativeProfileTruthError("validated release output is absent")
    output_resolved = _resolved_without_strict(output_dir)
    receipt_resolved = _resolved_without_strict(validation_receipt)
    if receipt_resolved == output_resolved or receipt_resolved.is_relative_to(
        output_resolved
    ):
        raise NativeProfileTruthError(
            "validation receipt must be outside the release output"
        )


if __name__ == "__main__":
    raise SystemExit(main())
