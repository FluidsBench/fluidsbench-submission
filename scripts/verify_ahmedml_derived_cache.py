#!/usr/bin/env python3
"""Verify a preinstalled AhmedML derived cache against immutable Git metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    AhmedMLContractError,
    load_source_identity,
    read_json,
    sha256_file,
)
from reference.ahmedml.support import (  # noqa: E402
    AhmedMLSupportError,
    load_case_support,
    load_profile_support,
    open_support_array,
)
from reference.scoring_support import (  # noqa: E402
    ScoringSupportError,
    load_support_release,
)
from scripts.list_ahmedml_support_cases import ordered_case_ids  # noqa: E402


DATASET_DIR = ROOT / "benchmark-specs" / "ahmedml"
DEFAULT_CONTRACT = DATASET_DIR / "derived-cache-contract-v1.json"
DEFAULT_SPECIFICATION = DATASET_DIR / "submission-spec.json"
CASE_SET_IDS = (
    "full-test",
    "geometry-test",
    "high_drag-test",
    "low_drag-test",
    "image_wake-test",
)
NPY_ROLES = (
    "surface_area_vector",
    "volume_cell_volume",
    "volume_region_code",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class CacheVerificationError(ValueError):
    """Raised when the installed derived cache differs from its contract."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheVerificationError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheVerificationError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise CacheVerificationError(f"{label} must be a lowercase SHA-256")
    return result


def _repository_path(value: object, label: str) -> Path:
    text = _string(value, label)
    candidate = (ROOT / text).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise CacheVerificationError(f"{label} escapes the repository") from error
    if not candidate.is_file():
        raise CacheVerificationError(f"{label} is missing: {candidate}")
    return candidate


def _dataset_path(contract_path: Path, value: object, label: str) -> Path:
    text = _string(value, label)
    candidate = (contract_path.parent / text).resolve()
    try:
        candidate.relative_to(contract_path.parent.resolve())
    except ValueError as error:
        raise CacheVerificationError(f"{label} escapes the AhmedML contract directory") from error
    if not candidate.is_file():
        raise CacheVerificationError(f"{label} is missing: {candidate}")
    return candidate


def _validate_contract(
    *,
    contract_path: Path,
    specification_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    contract_path = contract_path.expanduser().resolve()
    specification_path = specification_path.expanduser().resolve()
    contract = read_json(contract_path, label="AhmedML derived-cache contract")
    if (
        contract.get("schema") != "fluidsbench-ahmedml-derived-cache-contract-v1"
        or contract.get("schema_version") != 1
        or contract.get("status") != "candidate"
        or contract.get("dataset_id") != "ahmedml"
    ):
        raise CacheVerificationError("derived-cache contract header differs")

    source = _mapping(contract.get("canonical_source"), "canonical_source")
    if (
        source.get("kind") != "huggingface_dataset_revision"
        or source.get("repository_id") != "neashton/ahmedml"
        or source.get("revision") != "02688c727cdb8dc8678e28abc6bbbb7e93c5fa15"
        or source.get("public_case_count") != 500
    ):
        raise CacheVerificationError("canonical public source binding differs")
    source_identity_path = _dataset_path(
        contract_path,
        source.get("source_identity_file"),
        "canonical_source.source_identity_file",
    )
    if sha256_file(source_identity_path) != _digest(
        source.get("source_identity_sha256"),
        "canonical_source.source_identity_sha256",
    ):
        raise CacheVerificationError("public source-identity SHA-256 differs")

    policy = _mapping(contract.get("cache_policy"), "cache_policy")
    if (
        policy.get("classification") != "deterministic_rebuildable_evaluator_cache"
        or policy.get("canonical_source_data") is not False
        or policy.get("remote_payload_required") is not False
        or policy.get("runtime_rebuild_during_submission_scoring") is not False
        or policy.get("production_requirement")
        != "preinstall_and_verify_before_accepting_submissions"
    ):
        raise CacheVerificationError("derived-cache deployment policy differs")

    implementation = _mapping(
        contract.get("derivation_implementation"), "derivation_implementation"
    )
    revision = _string(
        implementation.get("git_revision_with_exact_builder"),
        "derivation_implementation.git_revision_with_exact_builder",
    )
    if _GIT_SHA_RE.fullmatch(revision) is None:
        raise CacheVerificationError("builder Git revision must be a full commit SHA")
    builder = _repository_path(
        implementation.get("builder_file"),
        "derivation_implementation.builder_file",
    )
    if sha256_file(builder) != _digest(
        implementation.get("builder_sha256"),
        "derivation_implementation.builder_sha256",
    ):
        raise CacheVerificationError("AhmedML support builder SHA-256 differs")
    requirements = _repository_path(
        implementation.get("requirements_file"),
        "derivation_implementation.requirements_file",
    )
    if sha256_file(requirements) != _digest(
        implementation.get("requirements_sha256"),
        "derivation_implementation.requirements_sha256",
    ):
        raise CacheVerificationError("AhmedML support-builder requirements SHA-256 differs")
    runtime = _mapping(implementation.get("runtime"), "derivation_implementation.runtime")
    if runtime != {
        "python": "3.12.3",
        "numpy": "2.5.2",
        "vtk": "9.5.2",
        "byte_order": "little_endian",
    }:
        raise CacheVerificationError("derived-cache builder runtime differs")

    expected = _mapping(contract.get("expected_outputs"), "expected_outputs")
    if (
        expected.get("official_test_case_union_count") != 316
        or expected.get("artifacts_per_case") != 4
    ):
        raise CacheVerificationError("derived-cache expected output counts differ")
    candidate_manifest_path = _dataset_path(
        contract_path,
        expected.get("candidate_manifest_file"),
        "expected_outputs.candidate_manifest_file",
    )
    candidate_sha = _digest(
        expected.get("candidate_manifest_sha256"),
        "expected_outputs.candidate_manifest_sha256",
    )
    if sha256_file(candidate_manifest_path) != candidate_sha:
        raise CacheVerificationError("candidate scoring-support manifest SHA-256 differs")

    specification = read_json(
        specification_path, label="AhmedML submission specification"
    )
    scoring_support = _mapping(
        specification.get("scoring_support"), "submission scoring_support"
    )
    binding = _mapping(
        scoring_support.get("derived_cache_contract"),
        "scoring_support.derived_cache_contract",
    )
    try:
        relative_contract = contract_path.relative_to(specification_path.parent)
    except ValueError as error:
        raise CacheVerificationError(
            "derived-cache contract must remain inside the AhmedML contract directory"
        ) from error
    if (
        binding.get("status") != "candidate"
        or binding.get("classification")
        != "deterministic_rebuildable_evaluator_cache"
        or binding.get("contract_file") != relative_contract.as_posix()
        or binding.get("contract_sha256") != sha256_file(contract_path)
        or binding.get("remote_payload_required") is not False
        or binding.get("production_installation_required") is not True
    ):
        raise CacheVerificationError("submission specification cache binding differs")
    return contract, source_identity_path, candidate_manifest_path


def _expected_case_manifest_hashes(manifest_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for case_set_id in CASE_SET_IDS:
        release = load_support_release(manifest_path, case_set_id)
        for case_id, case in release.cases.items():
            instances = case.get("support_instances")
            if not isinstance(instances, list) or not instances:
                raise CacheVerificationError(
                    f"{case_set_id}/{case_id} has no support instances"
                )
            digests = {
                instance.get("expected_support_sha256")
                for instance in instances
                if isinstance(instance, Mapping)
            }
            if len(digests) != 1:
                raise CacheVerificationError(
                    f"{case_set_id}/{case_id} support manifest hashes differ"
                )
            digest = _digest(
                next(iter(digests)),
                f"{case_set_id}/{case_id}.expected_support_sha256",
            )
            previous = expected.setdefault(case_id, digest)
            if previous != digest:
                raise CacheVerificationError(
                    f"{case_id} has inconsistent hashes across case sets"
                )
    return expected


def verify_cache(
    *,
    cache_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    specification_path: Path = DEFAULT_SPECIFICATION,
    deep: bool,
) -> dict[str, object]:
    """Verify all official AhmedML test-case cache entries and return a receipt."""

    cache_root = cache_root.expanduser().resolve()
    if not cache_root.is_dir():
        raise CacheVerificationError(f"cache root is not a directory: {cache_root}")
    contract, source_identity_path, candidate_manifest_path = _validate_contract(
        contract_path=contract_path,
        specification_path=specification_path,
    )
    identity = load_source_identity(source_identity_path)
    expected_hashes = _expected_case_manifest_hashes(candidate_manifest_path)
    case_ids = ordered_case_ids()
    if set(expected_hashes) != set(case_ids) or len(expected_hashes) != 316:
        raise CacheVerificationError(
            "candidate scoring-support cases differ from the 316-case official union"
        )

    artifact_count = 0
    payload_bytes = 0
    for case_id in case_ids:
        support = load_case_support(
            cache_root / case_id / "case-support.json",
            source_identity=identity,
        )
        if support.case_id != case_id:
            raise CacheVerificationError(f"cache entry {case_id} resolved as {support.case_id}")
        if support.manifest_sha256 != expected_hashes[case_id]:
            raise CacheVerificationError(f"{case_id} case-support SHA-256 differs")
        artifact_count += len(support.artifacts)
        payload_bytes += sum(item.size_bytes for item in support.artifacts.values())
        if deep:
            for role in NPY_ROLES:
                with open_support_array(support, role):
                    pass
            load_profile_support(support)

    expected = _mapping(contract.get("expected_outputs"), "expected_outputs")
    if artifact_count != len(case_ids) * int(expected["artifacts_per_case"]):
        raise CacheVerificationError("installed cache artifact count differs")
    implementation = _mapping(
        contract.get("derivation_implementation"), "derivation_implementation"
    )
    return {
        "schema": "fluidsbench-ahmedml-derived-cache-verification-v1",
        "schema_version": 1,
        "status": "verified",
        "verification_mode": "deep_payload_sha256" if deep else "metadata_and_size",
        "verified_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dataset_id": "ahmedml",
        "dataset_revision": identity.repository_revision,
        "contract_sha256": sha256_file(contract_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "builder_git_revision": implementation["git_revision_with_exact_builder"],
        "builder_sha256": implementation["builder_sha256"],
        "cache_root": str(cache_root),
        "case_count": len(case_ids),
        "artifact_count": artifact_count,
        "payload_bytes": payload_bytes,
    }


def _write_atomic(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--submission-specification", type=Path, default=DEFAULT_SPECIFICATION
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also hash every generated NPY/NPZ payload; use for production installation",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_cache(
            cache_root=args.cache_root,
            contract_path=args.contract,
            specification_path=args.submission_specification,
            deep=args.deep,
        )
        if args.output is None:
            print(json.dumps(receipt, indent=2, sort_keys=False))
        else:
            _write_atomic(args.output, receipt)
            print(args.output.expanduser().resolve())
    except (
        AhmedMLContractError,
        AhmedMLSupportError,
        CacheVerificationError,
        ScoringSupportError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
