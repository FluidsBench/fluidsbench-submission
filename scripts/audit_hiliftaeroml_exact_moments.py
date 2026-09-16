#!/usr/bin/env python3
"""Storage-light exact pitching-moment audit for all HiLiftAeroML cases.

Successful workers retain one compact, fingerprinted JSON receipt per case;
they never copy a VTU or alter the released fields/force CSVs.  The full
campaign uses rank striding rather than a Slurm array because the site array
limit is smaller than the 1,800-case universe.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The production venv deliberately uses NumPy 2.2.6.  The host's unrelated
# NumPy-1.x SciPy must be removed before importing NumPy/Numba; otherwise
# Numba's BLAS probe emits an ABI traceback even though this kernel needs no
# SciPy functionality.
if os.environ.get("HILIFT_EXCLUDE_SYSTEM_DIST_PACKAGES") == "1":
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != Path("/usr/lib/python3/dist-packages")
    ]

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.exact_moment_audit import (  # noqa: E402
    AGGREGATE_SCHEMA,
    ALGORITHM_ID,
    CASE_RECEIPT_SCHEMA,
    EXPECTED_ALL_CASE_COUNT,
    EXPECTED_ALL_CASE_SET_SHA256,
    EXPECTED_PUBLIC_CASE_COUNT,
    EXPECTED_PUBLIC_NUMERIC_SHA256,
    EXPECTED_SUPPORT_LEXICAL_SHA256,
    FORCE_EXCEPTION_CASES,
    CompensatedIntegrals,
    ExactMomentAuditError,
    canonical_json_bytes,
    case_sort_key,
    coefficient_document,
    comparison_record,
    content_fingerprint,
    decimal_half_quantum,
    newline_case_set_sha256,
    published_force_component_closure_audit,
    validate_all_case_universe,
    validate_case_receipt,
)

PILOT_CASES = (*FORCE_EXCEPTION_CASES, "geo_LHC001_AoA_4")
REQUIRED_POINT_FIELDS = (
    "PROJ(AVG(P))",
    "AVG(TAU_WALL(0))",
    "AVG(TAU_WALL(1))",
    "AVG(TAU_WALL(2))",
)
REQUIRED_FORCE_KEYS = ("cd", "cl", "cm", "cdp", "cdv", "clp", "clv")
PUBLIC_IDENTITY_RELATIVE = Path(
    "benchmark-specs/hiliftaeroml/public-source-identity/"
    "hiliftaeroml-public-source-identity-v1.json"
)
SUPPORT_MANIFEST_RELATIVE = Path(
    "benchmark-specs/hiliftaeroml/scoring-support/"
    "hiliftaeroml-native-all-splits-support-v1-candidate/manifest.json"
)
SUCCESS_DIRNAME = "cases"
FAILURE_DIRNAME = "failures"
LOCK_DIRNAME = "locks"
VALIDATION_SCHEMA = "hiliftaeroml-exact-moment-campaign-validation-v1"
RANK_RECEIPT_SCHEMA = "hiliftaeroml-exact-moment-rank-completion-v1"
PILOT_COMPARISON_SCHEMA = "hiliftaeroml-exact-moment-pilot-comparison-v1"
PILOT_COMPARISON_ATOL = 2.0e-10
PILOT_COMPARISON_RTOL = 2.0e-10
APPROVAL_FULL = "YES_OWNER_APPROVED_EXACT_MOMENT_AUDIT_ALL1800"
APPROVAL_PILOT = "YES_OWNER_APPROVED_EXACT_MOMENT_AUDIT_PILOT6"
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
EXPECTED_PUBLIC_IDENTITY_SHA256 = (
    "44f62b90d58ab8e070c3df2a37c98435c576519222b1853a511e79b2124dc881"
)
EXPECTED_ALL1800_INVENTORY_SHA256 = (
    "bf31c77ff240764c4772bcebe7d49698e573023500bbad4d4e9591df4c2cca9b"
)
ALL1800_INVENTORY_RELATIVE = Path(
    "eval_runs/hilift_native_prerequisites_all1800_v1/case_inventory_all1800.json"
)


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ExactMomentAuditError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ExactMomentAuditError(f"{label} is not a regular file: {resolved}")
    return resolved


def _absolute_directory(path: Path, label: str, *, create: bool = False) -> Path:
    if not path.is_absolute():
        raise ExactMomentAuditError(f"{label} must be absolute")
    if path.is_symlink():
        raise ExactMomentAuditError(f"{label} must not be a symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=False)
    if resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise ExactMomentAuditError(f"{label} is not a regular directory")
    return resolved


def _atomic_create(
    path: Path, payload: bytes, *, accept_identical_existing: bool = False
) -> str:
    """Create a durable file, optionally accepting byte-identical replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if not accept_identical_existing:
                raise ExactMomentAuditError(
                    f"refusing to replace artifact: {path}"
                ) from error
            existing = _regular_file(path, "existing derived artifact").read_bytes()
            if existing != payload:
                raise ExactMomentAuditError(
                    f"existing derived artifact differs byte-for-byte: {path}"
                ) from error
            return "verified_identical_existing"
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return "created"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExactMomentAuditError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ExactMomentAuditError(f"{label} must contain an object")
    return value, _sha256_file(path)


def _one_csv_row(path: Path, label: str) -> dict[str, str]:
    path = _regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ExactMomentAuditError(f"{label} must contain exactly one data row")
    return rows[0]


def _finite_csv(row: Mapping[str, str], key: str, source: Path) -> float:
    raw = row.get(key)
    if raw is None or not str(raw).strip():
        raise ExactMomentAuditError(f"{source} lacks finite field {key!r}")
    try:
        value = float(raw)
    except ValueError as error:
        raise ExactMomentAuditError(f"{source} field {key!r} is not numeric") from error
    if not math.isfinite(value):
        raise ExactMomentAuditError(f"{source} field {key!r} is not finite")
    return value


def _parse_reference_vector(raw: str, source: Path) -> np.ndarray:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise ExactMomentAuditError(f"invalid forcesCoR in {source}") from error
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ExactMomentAuditError(
            f"forcesCoR in {source} is not a finite three-vector"
        )
    return result


def _support_union(manifest_path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    manifest, manifest_sha = _load_json(manifest_path, "scoring-support manifest")
    descriptors = manifest.get("case_sets")
    if manifest.get("dataset_id") != "hiliftaeroml" or not isinstance(
        descriptors, list
    ):
        raise ExactMomentAuditError("scoring-support manifest identity differs")
    release_root = manifest_path.resolve().parent
    union: set[str] = set()
    case_sets: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ExactMomentAuditError("case-set descriptor is invalid")
        index_file = descriptor.get("index_file")
        if (
            not isinstance(index_file, str)
            or Path(index_file).is_absolute()
            or ".." in Path(index_file).parts
        ):
            raise ExactMomentAuditError("case-set index path is invalid")
        index_path = release_root / index_file
        index, index_sha = _load_json(index_path, "case-set index")
        if index_sha != descriptor.get("index_sha256"):
            raise ExactMomentAuditError("case-set index SHA-256 differs")
        selected: list[str] = []
        chunks = index.get("chunks")
        if not isinstance(chunks, list):
            raise ExactMomentAuditError("case-set chunks are absent")
        for chunk in chunks:
            ids = chunk.get("case_ids") if isinstance(chunk, Mapping) else None
            if not isinstance(ids, list) or not all(
                isinstance(case, str) for case in ids
            ):
                raise ExactMomentAuditError("case-set chunk IDs differ")
            selected.extend(ids)
        if len(selected) != descriptor.get("case_count") or len(selected) != len(
            set(selected)
        ):
            raise ExactMomentAuditError("case-set count or uniqueness differs")
        for case_id in selected:
            case_sort_key(case_id)
        union.update(selected)
        case_sets.append(
            {
                "case_set_id": descriptor.get("id"),
                "case_count": len(selected),
                "index_sha256": index_sha,
            }
        )
    lexical = tuple(sorted(union))
    digest = newline_case_set_sha256(lexical)
    if (
        len(lexical) != EXPECTED_PUBLIC_CASE_COUNT
        or digest != EXPECTED_SUPPORT_LEXICAL_SHA256
    ):
        raise ExactMomentAuditError("scoring-support lexical union differs")
    return lexical, {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "case_set_count": len(case_sets),
        "case_count": len(lexical),
        "ordering": "Python lexical case_id order (legacy scoring-support union)",
        "newline_case_set_sha256": digest,
        "case_sets": case_sets,
    }


def load_universe_contract(
    *,
    dataset_root: Path,
    public_identity_path: Path,
    support_manifest_path: Path,
    all1800_inventory_path: Path,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    """Reconcile the 1800 local cases and both 1355-case order conventions."""

    dataset_root = _absolute_directory(dataset_root, "dataset root")
    local_ids: list[str] = []
    for candidate in dataset_root.iterdir():
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        try:
            case_sort_key(candidate.name)
        except ExactMomentAuditError:
            continue
        local_ids.append(candidate.name)
    all_ids = validate_all_case_universe(tuple(sorted(local_ids, key=case_sort_key)))

    frozen, frozen_sha = _load_json(all1800_inventory_path, "all-1800 inventory")
    frozen_cases = frozen.get("cases")
    if (
        frozen_sha != EXPECTED_ALL1800_INVENTORY_SHA256
        or frozen.get("status") != "complete"
        or frozen.get("case_count") != EXPECTED_ALL_CASE_COUNT
        or frozen.get("case_set_sha256") != EXPECTED_ALL_CASE_SET_SHA256
        or not isinstance(frozen_cases, list)
    ):
        raise ExactMomentAuditError("frozen all-1800 inventory identity differs")
    source_bindings: dict[str, dict[str, Any]] = {}
    frozen_ids: list[str] = []
    for expected_index, record in enumerate(frozen_cases):
        if (
            not isinstance(record, Mapping)
            or record.get("case_index") != expected_index
        ):
            raise ExactMomentAuditError("all-1800 inventory index/order differs")
        case_id = record.get("case_id")
        surface = record.get("surface_vtu")
        if not isinstance(case_id, str) or not isinstance(surface, Mapping):
            raise ExactMomentAuditError("all-1800 surface identity is absent")
        case_sort_key(case_id)
        expected_path = dataset_root / case_id / f"boundary_{case_id}.vtu"
        expected_stat = {
            "path": str(expected_path.resolve()),
            "size_bytes": surface.get("size_bytes"),
            "mtime_ns": surface.get("mtime_ns"),
        }
        if (
            surface.get("path") != str(expected_path.resolve())
            or _source_stat(_regular_file(expected_path, "frozen native surface VTU"))
            != expected_stat
        ):
            raise ExactMomentAuditError(
                f"{case_id} live surface stat differs from the frozen all-1800 inventory"
            )
        frozen_ids.append(case_id)
        source_bindings[case_id] = {
            "frozen_local_surface": expected_stat,
            "frozen_all1800_inventory_sha256": frozen_sha,
            "public_surface_archive": None,
            "public_source_inventory_sha256": None,
            "public_archive_equivalence_to_local_extracted_vtu": "not_established",
        }
    if tuple(frozen_ids) != all_ids:
        raise ExactMomentAuditError("all-1800 inventory membership/order differs")

    public, public_sha = _load_json(public_identity_path, "public-source identity")
    if public_sha != EXPECTED_PUBLIC_IDENTITY_SHA256:
        raise ExactMomentAuditError("public-source inventory SHA-256 differs")
    public_cases = public.get("cases")
    if not isinstance(public_cases, list):
        raise ExactMomentAuditError("public-source cases are absent")
    public_ids: list[str] = []
    for expected_index, record in enumerate(public_cases):
        if (
            not isinstance(record, Mapping)
            or record.get("case_index") != expected_index
        ):
            raise ExactMomentAuditError("public-source case index/order differs")
        case_id = record.get("case_id")
        if not isinstance(case_id, str):
            raise ExactMomentAuditError("public-source case ID is invalid")
        case_sort_key(case_id)
        public_ids.append(case_id)
        surface = record.get("surface")
        archive = surface.get("archive") if isinstance(surface, Mapping) else None
        member = surface.get("member") if isinstance(surface, Mapping) else None
        frozen_size = source_bindings[case_id]["frozen_local_surface"]["size_bytes"]
        if (
            not isinstance(archive, Mapping)
            or not isinstance(member, Mapping)
            or member.get("path") != f"boundary_{case_id}.vtu"
            or member.get("type") != "regular_file"
            or member.get("declared_size_bytes") != frozen_size
            or member.get("extraction_rule")
            != "exactly-one-regular-member-exact-basename-no-links-v1"
        ):
            raise ExactMomentAuditError(
                "public-source surface archive binding is absent"
            )
        source_bindings[case_id]["public_surface_archive"] = {
            "repository_path": archive.get("repository_path"),
            "size_bytes": archive.get("size_bytes"),
            "lfs_sha256": archive.get("lfs_sha256"),
            "member": {
                "path": member.get("path"),
                "type": member.get("type"),
                "declared_size_bytes": member.get("declared_size_bytes"),
                "extraction_rule": member.get("extraction_rule"),
            },
        }
        source_bindings[case_id]["public_source_inventory_sha256"] = public_sha
        source_bindings[case_id][
            "public_archive_equivalence_to_local_extracted_vtu"
        ] = "exact_member_basename_and_size_only; content_equivalence_not_established"
    numeric_public = tuple(public_ids)
    declared = public.get("case_set")
    if not isinstance(declared, Mapping):
        raise ExactMomentAuditError("public-source case-set descriptor is absent")
    public_digest = newline_case_set_sha256(numeric_public)
    if (
        len(numeric_public) != EXPECTED_PUBLIC_CASE_COUNT
        or tuple(sorted(numeric_public, key=case_sort_key)) != numeric_public
        or len(set(numeric_public)) != len(numeric_public)
        or declared.get("case_count") != EXPECTED_PUBLIC_CASE_COUNT
        or declared.get("case_set_sha256") != EXPECTED_PUBLIC_NUMERIC_SHA256
        or public_digest != EXPECTED_PUBLIC_NUMERIC_SHA256
    ):
        raise ExactMomentAuditError("public-source numeric case universe differs")

    lexical, support_contract = _support_union(support_manifest_path)
    if set(lexical) != set(numeric_public):
        raise ExactMomentAuditError(
            "public-source and scoring-support 1355-case memberships differ"
        )
    return (
        all_ids,
        numeric_public,
        {
            "all_1800": {
                "inventory": str(all1800_inventory_path.resolve()),
                "inventory_sha256": frozen_sha,
                "case_count": len(all_ids),
                "ordering": "geometry_id ascending, then numeric AoA ascending",
                "newline_case_set_sha256": EXPECTED_ALL_CASE_SET_SHA256,
            },
            "fluidsbench_public_1355": {
                "inventory": str(public_identity_path.resolve()),
                "inventory_sha256": public_sha,
                "case_count": len(numeric_public),
                "ordering": "geometry_id ascending, then numeric AoA ascending",
                "newline_case_set_sha256": public_digest,
            },
            "scoring_support_legacy_1355": support_contract,
            "membership_reconciliation": {
                "same_1355_membership": True,
                "orders_intentionally_distinct": numeric_public != lexical,
                "numeric_public_sha256": EXPECTED_PUBLIC_NUMERIC_SHA256,
                "lexical_support_sha256": EXPECTED_SUPPORT_LEXICAL_SHA256,
            },
        },
        source_bindings,
    )


def striped_indices(case_count: int, *, rank: int, world_size: int) -> tuple[int, ...]:
    if case_count <= 0:
        raise ExactMomentAuditError("case_count must be positive")
    if world_size not in {1, 8, 10}:
        raise ExactMomentAuditError("world_size must be one, eight, or ten")
    if rank < 0 or rank >= world_size:
        raise ExactMomentAuditError("rank is outside the dispatcher world")
    return tuple(range(rank, case_count, world_size))


def validate_stripe_coverage(case_count: int, *, world_size: int) -> tuple[int, ...]:
    combined = tuple(
        sorted(
            index
            for rank in range(world_size)
            for index in striped_indices(case_count, rank=rank, world_size=world_size)
        )
    )
    if combined != tuple(range(case_count)):
        raise ExactMomentAuditError("rank stripes do not cover every case exactly once")
    return combined


def _configure_isolated_numba_cache() -> Path | None:
    """Select one preflight/rank cache directory before importing Numba."""

    raw_parent = os.environ.get("HILIFT_NUMBA_CACHE_PARENT")
    if raw_parent is None:
        return None
    parent = _absolute_directory(Path(raw_parent), "Numba cache parent", create=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    if job_id is None or not job_id.isdigit() or not restart.isdigit():
        raise ExactMomentAuditError(
            "isolated Numba cache requires numeric Slurm job/restart metadata"
        )
    role = os.environ.get("HILIFT_NUMBA_CACHE_ROLE")
    if role == "preflight":
        leaf = "preflight"
    elif role is None:
        rank = os.environ.get("SLURM_PROCID")
        if rank is None or not rank.isdigit():
            raise ExactMomentAuditError(
                "rank-isolated Numba cache requires a numeric Slurm process ID"
            )
        leaf = f"rank-{int(rank):03d}"
    else:
        raise ExactMomentAuditError("unknown HILIFT_NUMBA_CACHE_ROLE")
    target = _absolute_directory(
        parent / f"{job_id}-r{int(restart)}" / leaf,
        "isolated Numba cache",
        create=True,
    )
    os.environ["NUMBA_CACHE_DIR"] = str(target)
    return target


def _backend(recipe_root: Path) -> dict[str, Any]:
    if os.environ.get("HILIFT_EXCLUDE_SYSTEM_DIST_PACKAGES") != "1" or any(
        Path(entry or ".").resolve() == SYSTEM_DIST_PACKAGES for entry in sys.path
    ):
        raise ExactMomentAuditError(
            "production backend requires early system-dist-packages exclusion"
        )
    numba_cache = _configure_isolated_numba_cache()
    recipe_root = _absolute_directory(recipe_root, "recipe root")
    source_dir = recipe_root / "src"
    scripts_dir = recipe_root / "scripts"
    for directory in (source_dir, scripts_dir):
        if not directory.is_dir() or directory.is_symlink():
            raise ExactMomentAuditError(
                f"backend directory is unavailable: {directory}"
            )
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    memmap_module = importlib.import_module("native_vtu_memmap")
    geometry_module = importlib.import_module("build_hilift_surface_dual_area")
    numba_module = importlib.import_module("reference.hiliftaeroml.exact_moment_numba")
    numba_package = importlib.import_module("numba")
    vtk_package = importlib.import_module("vtk")
    return {
        "NativeVTUPointReader": memmap_module.NativeVTUPointReader,
        "geometry_only": geometry_module._geometry_only_vtu,
        "validate_geometry": geometry_module._validate_geometry,
        "geometry_hash": geometry_module._canonical_array_hash,
        "integrate_cell_range": numba_module.integrate_native_cell_range,
        "kernel_self_check": numba_module.warmup_and_self_check,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba_package.__version__,
            "vtk": vtk_package.vtkVersion.GetVTKVersion(),
            "system_dist_packages_excluded": True,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
            "python_unbuffered": os.environ.get("PYTHONUNBUFFERED") == "1",
            "numba_cache_dir": (
                str(numba_cache)
                if numba_cache is not None
                else os.environ.get("NUMBA_CACHE_DIR")
            ),
            "sys_path": list(sys.path),
        },
        "files": {
            "native_vtu_memmap": {
                "path": str(Path(memmap_module.__file__).resolve()),
                "sha256": _sha256_file(Path(memmap_module.__file__).resolve()),
            },
            "build_hilift_surface_dual_area": {
                "path": str(Path(geometry_module.__file__).resolve()),
                "sha256": _sha256_file(Path(geometry_module.__file__).resolve()),
            },
            "exact_moment_audit": {
                "path": str(
                    (ROOT / "reference/hiliftaeroml/exact_moment_audit.py").resolve()
                ),
                "sha256": _sha256_file(
                    ROOT / "reference/hiliftaeroml/exact_moment_audit.py"
                ),
            },
            "exact_moment_numba": {
                "path": str(
                    (ROOT / "reference/hiliftaeroml/exact_moment_numba.py").resolve()
                ),
                "sha256": _sha256_file(
                    ROOT / "reference/hiliftaeroml/exact_moment_numba.py"
                ),
            },
            "campaign_driver": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
    }


def _source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _logical_array_sha256(name: str, array: np.ndarray) -> str:
    """Hash one logical native array in tuple order without retaining a copy."""

    value = np.asarray(array)
    if not value.flags.c_contiguous or value.dtype.hasobject:
        raise ExactMomentAuditError(f"native field {name!r} is not a plain C array")
    digest = hashlib.sha256()
    digest.update(b"hiliftaeroml-native-logical-point-array-v1\0")
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0"
    )
    raw = value.view(np.uint8).reshape(-1)
    block_bytes = 64 * 1024 * 1024
    for start in range(0, raw.size, block_bytes):
        digest.update(memoryview(raw[start : min(start + block_bytes, raw.size)]))
    return digest.hexdigest()


def _preflight_identity(
    *,
    dataset_root: Path,
    case_id: str,
    p_inf: float,
    cell_chunk: int,
    backend: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path, Path]:
    case_sort_key(case_id)
    case_dir = dataset_root / case_id
    vtu = _regular_file(case_dir / f"boundary_{case_id}.vtu", "native surface VTU")
    ref = _regular_file(case_dir / f"ref_values_{case_id}.csv", "reference CSV")
    force = _regular_file(case_dir / f"force_mom_{case_id}.csv", "force CSV")
    observed_stat = _source_stat(vtu)
    if source_binding.get("frozen_local_surface") != observed_stat:
        raise ExactMomentAuditError(
            f"{case_id} source differs from the frozen all-1800 inventory"
        )
    return (
        {
            "source_vtu_stat": observed_stat,
            "frozen_source_binding": dict(source_binding),
            "reference_csv": {
                "path": str(ref),
                "size_bytes": ref.stat().st_size,
                "sha256": _sha256_file(ref),
            },
            "force_csv": {
                "path": str(force),
                "size_bytes": force.stat().st_size,
                "sha256": _sha256_file(force),
            },
            "p_inf": p_inf,
            "cell_chunk": cell_chunk,
            "backend_files": backend["files"],
        },
        vtu,
        ref,
        force,
    )


def _array_spec(spec: Any) -> dict[str, Any]:
    return {
        "vtk_type": spec.vtk_type,
        "dtype": spec.dtype.str,
        "components": spec.components,
        "tuples": spec.tuples,
        "payload_offset": spec.payload_offset,
        "expected_nbytes": spec.expected_nbytes,
    }


def _native_integrals(
    *,
    backend: Mapping[str, Any],
    vtu: Path,
    moment_reference: np.ndarray,
    p_inf: float,
    q_ref: float,
    cell_chunk: int,
) -> tuple[Any, dict[str, Any]]:
    Reader = backend["NativeVTUPointReader"]
    reader = Reader(vtu, validate_byte_counts=False)
    missing = [
        name for name in REQUIRED_POINT_FIELDS if name not in reader.point_data_specs
    ]
    if missing:
        raise ExactMomentAuditError(f"native surface lacks point fields: {missing}")
    reader.validate_byte_counts(REQUIRED_POINT_FIELDS)
    grid, points, connectivity, offsets, cell_types = backend["geometry_only"](vtu)
    try:
        topology = backend["validate_geometry"](
            points, connectivity, offsets, cell_types, chunk_values=2_000_000
        )
        if points.shape[0] != reader.number_of_points:
            raise ExactMomentAuditError(
                "VTK geometry and raw point-field counts differ"
            )
        geometry_sha = backend["geometry_hash"](
            (
                ("points", points),
                ("connectivity", connectivity),
                ("offsets", offsets),
                ("cell_types", cell_types),
            )
        )
        pressure = reader.point_data_memmap(REQUIRED_POINT_FIELDS[0])
        tau = tuple(
            reader.point_data_memmap(name) for name in REQUIRED_POINT_FIELDS[1:]
        )
        logical_field_sha256 = {
            REQUIRED_POINT_FIELDS[0]: _logical_array_sha256(
                REQUIRED_POINT_FIELDS[0], pressure
            ),
            **{
                name: _logical_array_sha256(name, component)
                for name, component in zip(REQUIRED_POINT_FIELDS[1:], tau, strict=True)
            },
        }
        accumulator = CompensatedIntegrals()
        cell_count = offsets.size - 1
        for begin in range(0, cell_count, cell_chunk):
            end = min(begin + cell_chunk, cell_count)
            accumulator.add(
                backend["integrate_cell_range"](
                    points=points,
                    connectivity=connectivity,
                    offsets=offsets,
                    pressure=pressure,
                    tau_x=tau[0],
                    tau_y=tau[1],
                    tau_z=tau[2],
                    moment_reference=moment_reference,
                    p_inf=p_inf,
                    q_ref=q_ref,
                    begin_cell=begin,
                    end_cell=end,
                )
            )
        result = accumulator.finalize()
        if result.triangle_count != topology["n_fan_triangles"]:
            raise ExactMomentAuditError(
                "integrated triangle count differs from topology"
            )
        field_specs = {
            name: _array_spec(reader.point_data_specs[name])
            for name in REQUIRED_POINT_FIELDS
        }
        return result, {
            "geometry_sha256": geometry_sha,
            "topology": topology,
            "field_specs": field_specs,
            "logical_field_sha256": logical_field_sha256,
            "source_byte_count_convention": reader.byte_count_convention,
        }
    finally:
        del grid


def _force_comparisons(
    coefficients: Mapping[str, Any], published: Mapping[str, float]
) -> dict[str, Any]:
    force = coefficients["force"]
    calculated = {
        "cdp": force["pressure"]["cd"],
        "clp": force["pressure"]["cl"],
        "cdv": force["viscous"]["cd"],
        "clv": force["viscous"]["cl"],
        "cd": force["total"]["cd"],
        "cl": force["total"]["cl"],
    }
    return {
        name: {
            "calculated": float(calculated[name]),
            "published": float(published[name]),
            "calculated_minus_published": float(calculated[name] - published[name]),
            "relative_error_percent": (
                100.0 * (calculated[name] - published[name]) / abs(published[name])
                if published[name] != 0.0
                else None
            ),
        }
        for name in calculated
    }


def _canonical_force_acceptance(
    coefficients: Mapping[str, Any], force_row: Mapping[str, str], force_path: Path
) -> dict[str, Any]:
    """Replay the frozen native-truth-vs-monitor CI95 force policy."""

    force = coefficients["force"]
    calculated = {
        "cdp": float(force["pressure"]["cd"]),
        "clp": float(force["pressure"]["cl"]),
        "cdv": float(force["viscous"]["cd"]),
        "clv": float(force["viscous"]["cl"]),
        "cd": float(force["total"]["cd"]),
        "cl": float(force["total"]["cl"]),
    }
    published = {name: _finite_csv(force_row, name, force_path) for name in calculated}
    axes = {
        "drag": ("cd", "cdp", "cdv", "cd_ci95"),
        "lift": ("cl", "clp", "clv", "cl_ci95"),
    }
    tokens = {name: force_row[name].strip() for name in calculated}
    closure_audit = published_force_component_closure_audit(published, tokens)
    checks: dict[str, Any] = {}
    for axis, (total, pressure, viscous, ci_key) in axes.items():
        closure = closure_audit["axes"][axis]["absolute_residual"]
        ci95 = _finite_csv(force_row, ci_key, force_path)
        if ci95 < 0.0:
            raise ExactMomentAuditError("published force CI95 must be non-negative")
        for key, role in (
            (total, "total"),
            (pressure, "pressure"),
            (viscous, "viscous"),
        ):
            half_quantum = decimal_half_quantum(force_row[key])
            numerical = (
                1.0e-4 * max(1.0, abs(calculated[key]), abs(published[key]))
                + half_quantum
                + closure
            )
            allowance = ci95 if role in {"total", "pressure"} else 0.0
            absolute = abs(calculated[key] - published[key])
            checks[key] = {
                "axis": axis,
                "role": role,
                "calculated": calculated[key],
                "published": published[key],
                "calculated_minus_published": calculated[key] - published[key],
                "absolute_error": absolute,
                "numerical_tolerance": numerical,
                "published_axis_ci95_allowance": allowance,
                "acceptance_tolerance": numerical + allowance,
                "numerical_status": "pass" if absolute <= numerical else "fail",
                "status": "pass" if absolute <= numerical + allowance else "fail",
            }
    failed = sorted(key for key, value in checks.items() if value["status"] != "pass")
    return {
        "protocol_id": "hilift-native-truth-vs-monitor-ci95-v1",
        "status": "pass" if not failed else "fail",
        "failed_keys": failed,
        "published_component_closure": closure_audit,
        "checks": checks,
    }


def build_case_receipt(
    *,
    dataset_root: Path,
    recipe_root: Path,
    case_id: str,
    p_inf: float,
    cell_chunk: int,
    source_binding: Mapping[str, Any],
    backend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one native VTU and return a deterministic compact receipt."""

    if not math.isfinite(p_inf):
        raise ExactMomentAuditError("p_inf must be finite")
    if (
        isinstance(cell_chunk, bool)
        or not isinstance(cell_chunk, int)
        or cell_chunk <= 0
    ):
        raise ExactMomentAuditError("cell_chunk must be a positive integer")
    dataset_root = _absolute_directory(dataset_root, "dataset root")
    backend = _backend(recipe_root) if backend is None else backend
    preflight, vtu, ref_path, force_path = _preflight_identity(
        dataset_root=dataset_root,
        case_id=case_id,
        p_inf=p_inf,
        cell_chunk=cell_chunk,
        backend=backend,
        source_binding=source_binding,
    )
    ref = _one_csv_row(ref_path, "reference CSV")
    force = _one_csv_row(force_path, "force CSV")
    q_ref = _finite_csv(ref, "qRef", ref_path)
    area_ref = _finite_csv(ref, "areaRef", ref_path)
    length_ref = _finite_csv(ref, "chordRef", ref_path)
    alpha_deg = _finite_csv(ref, "alphaRef", ref_path)
    if q_ref <= 0.0 or area_ref <= 0.0 or length_ref <= 0.0:
        raise ExactMomentAuditError("qRef, areaRef, and chordRef must be positive")
    if not math.isclose(alpha_deg, float(case_sort_key(case_id)[1]), abs_tol=1.0e-10):
        raise ExactMomentAuditError("alphaRef differs from the case ID")
    moment_reference = _parse_reference_vector(ref.get("forcesCoR", ""), ref_path)
    published = {
        name: _finite_csv(force, name, force_path) for name in REQUIRED_FORCE_KEYS
    }
    cm_ci95 = _finite_csv(force, "cm_ci95", force_path)
    cm_stderr = _finite_csv(force, "cm_stderr", force_path)
    cd_ci95 = _finite_csv(force, "cd_ci95", force_path)
    cl_ci95 = _finite_csv(force, "cl_ci95", force_path)
    cd_stderr = _finite_csv(force, "cd_stderr", force_path)
    cl_stderr = _finite_csv(force, "cl_stderr", force_path)
    if (
        cm_stderr < 0.0
        or cd_stderr < 0.0
        or cl_stderr < 0.0
        or cm_ci95 < cm_stderr
        or cd_ci95 < cd_stderr
        or cl_ci95 < cl_stderr
    ):
        raise ExactMomentAuditError("published moment uncertainty must be non-negative")

    integrals, native = _native_integrals(
        backend=backend,
        vtu=vtu,
        moment_reference=moment_reference,
        p_inf=p_inf,
        q_ref=q_ref,
        cell_chunk=cell_chunk,
    )
    if _source_stat(vtu) != preflight["source_vtu_stat"]:
        raise ExactMomentAuditError("source VTU stat changed during integration")
    if _sha256_file(ref_path) != preflight["reference_csv"]["sha256"]:
        raise ExactMomentAuditError("reference CSV changed during integration")
    if _sha256_file(force_path) != preflight["force_csv"]["sha256"]:
        raise ExactMomentAuditError("force CSV changed during integration")

    coefficients = coefficient_document(
        integrals,
        reference_area=area_ref,
        reference_length=length_ref,
        alpha_rad=math.radians(alpha_deg),
    )
    exact_cm = float(coefficients["moment_exact_degree2"]["total"]["cm_body_y"])
    vertex_cm = float(coefficients["moment_vertex_lumped"]["total"]["cm_body_y"])
    cm_token = force["cm"].strip()
    cm_half_quantum = decimal_half_quantum(cm_token)
    exact_comparison = comparison_record(
        exact_cm,
        published["cm"],
        cm_ci95,
        published_half_quantum=cm_half_quantum,
    )
    vertex_comparison = comparison_record(
        vertex_cm,
        published["cm"],
        cm_ci95,
        published_half_quantum=cm_half_quantum,
    )
    receipt: dict[str, Any] = {
        "schema_id": CASE_RECEIPT_SCHEMA,
        "status": "complete",
        "algorithm_id": ALGORITHM_ID,
        "case_id": case_id,
        "input_identity": {"preflight": preflight, "native": native},
        "reference": {
            "p_inf": p_inf,
            "q_ref": q_ref,
            "area_ref_in2": area_ref,
            "length_ref_in": length_ref,
            "alpha_deg": alpha_deg,
            "forcesCoR_in": moment_reference.tolist(),
        },
        "quadrature": {
            "surface_triangulation": "ordered fan (v0, vj, vj+1)",
            "field_interpolation": "piecewise-linear nodal",
            "force": "exact degree-1 barycentric integration",
            "moment_exact": (
                "exact degree-2 triangle mass matrix; integral(lambda_i*lambda_j) "
                "is A/6 for i=j and A/12 otherwise"
            ),
            "moment_legacy_comparison": "vertex-lumped r cross nodal force",
            "pressure_sign": "+Cp times ordered oriented area vector",
            "cm_convention": "positive body-y moment about forcesCoR divided by areaRef*chordRef",
        },
        "integral_diagnostics": {
            "triangle_count": integrals.triangle_count,
            "zero_area_triangle_count": integrals.zero_area_triangle_count,
            "surface_area_in2": integrals.area_sum,
            "oriented_area_sum_in2": integrals.oriented_area_sum.tolist(),
            "pressure_force_numerator_in2": integrals.pressure_force.tolist(),
            "viscous_force_numerator_in2": integrals.viscous_force.tolist(),
            "pressure_moment_exact_in3": integrals.pressure_moment_exact.tolist(),
            "viscous_moment_exact_in3": integrals.viscous_moment_exact.tolist(),
            "pressure_moment_vertex_lumped_in3": (
                integrals.pressure_moment_vertex_lumped.tolist()
            ),
            "viscous_moment_vertex_lumped_in3": (
                integrals.viscous_moment_vertex_lumped.tolist()
            ),
        },
        "coefficients": coefficients,
        "published": {
            **published,
            "cd_stderr": cd_stderr,
            "cl_stderr": cl_stderr,
            "cm_stderr": cm_stderr,
            "cd_ci95": cd_ci95,
            "cl_ci95": cl_ci95,
            "cm_ci95": cm_ci95,
            "cm_csv_token": cm_token,
            "force_csv_tokens": {
                name: force[name].strip() for name in REQUIRED_FORCE_KEYS
            },
        },
        "comparison": {
            "scientific_role": (
                "diagnostic audit; disagreement does not mutate fields or published CSVs"
            ),
            "exact_vs_published": exact_comparison,
            "vertex_lumped_vs_published": vertex_comparison,
            "exact_minus_vertex_lumped_cm": exact_cm - vertex_cm,
            "exact_minus_vertex_lumped_relative_to_published_percent": (
                100.0 * (exact_cm - vertex_cm) / abs(published["cm"])
                if published["cm"] != 0.0
                else None
            ),
            "force_vs_published": _force_comparisons(coefficients, published),
            "force_acceptance": _canonical_force_acceptance(
                coefficients, force, force_path
            ),
        },
        "source_fields_modified": False,
        "published_coefficients_modified": False,
    }
    receipt["content_fingerprint"] = content_fingerprint(receipt)
    validate_case_receipt(receipt, expected_case_id=case_id)
    return receipt


def _existing_receipt(
    path: Path, *, case_id: str, expected_preflight: Mapping[str, Any]
) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    receipt, _ = _load_json(path, f"{case_id} receipt")
    validate_case_receipt(receipt, expected_case_id=case_id)
    if receipt.get("input_identity", {}).get("preflight") != expected_preflight:
        raise ExactMomentAuditError(
            f"existing {case_id} receipt input identity differs"
        )
    return True


def _write_failure(output_root: Path, case_id: str, error: BaseException) -> Path:
    attempt = (
        os.environ.get("SLURM_JOB_ID", "local")
        + "-"
        + os.environ.get("SLURM_PROCID", "0")
        + f"-{os.getpid()}-{time.time_ns()}"
    )
    path = output_root / FAILURE_DIRNAME / case_id / f"attempt-{attempt}.json"
    payload = {
        "schema_id": "hiliftaeroml-exact-moment-failed-attempt-v1",
        "status": "failed",
        "algorithm_id": ALGORITHM_ID,
        "case_id": case_id,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "source_fields_modified": False,
    }
    _atomic_create(path, canonical_json_bytes(payload))
    return path


def run_case(
    *,
    dataset_root: Path,
    recipe_root: Path,
    output_root: Path,
    case_id: str,
    p_inf: float,
    cell_chunk: int,
    source_binding: Mapping[str, Any],
    backend: Mapping[str, Any] | None = None,
) -> str:
    output_root = _absolute_directory(output_root, "output root", create=True)
    dataset_root = _absolute_directory(dataset_root, "dataset root")
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ExactMomentAuditError("audit output must be outside the dataset root")
    backend = _backend(recipe_root) if backend is None else backend
    preflight, _, _, _ = _preflight_identity(
        dataset_root=dataset_root,
        case_id=case_id,
        p_inf=p_inf,
        cell_chunk=cell_chunk,
        backend=backend,
        source_binding=source_binding,
    )
    receipt_path = output_root / SUCCESS_DIRNAME / f"{case_id}.json"
    if _existing_receipt(receipt_path, case_id=case_id, expected_preflight=preflight):
        return "skipped_valid_complete"

    lock_path = output_root / LOCK_DIRNAME / f"{case_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExactMomentAuditError(f"case lock is active: {lock_path}") from error
        os.ftruncate(lock_fd, 0)
        os.write(
            lock_fd,
            canonical_json_bytes(
                {
                    "case_id": case_id,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_procid": os.environ.get("SLURM_PROCID"),
                }
            ),
        )
        os.fsync(lock_fd)
        if _existing_receipt(
            receipt_path, case_id=case_id, expected_preflight=preflight
        ):
            return "skipped_valid_complete"
        receipt = build_case_receipt(
            dataset_root=dataset_root,
            recipe_root=recipe_root,
            case_id=case_id,
            p_inf=p_inf,
            cell_chunk=cell_chunk,
            source_binding=source_binding,
            backend=backend,
        )
        _atomic_create(receipt_path, canonical_json_bytes(receipt))
        return "completed"
    except BaseException as error:
        _write_failure(output_root, case_id, error)
        raise
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        # Keep the inert lock inode.  Kernel/Slurm termination releases the OS
        # advisory lock automatically, so stale files never block resumption.


def _read_receipts(
    receipt_root: Path, case_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        path = receipt_root / SUCCESS_DIRNAME / f"{case_id}.json"
        receipt, _ = _load_json(path, f"{case_id} receipt")
        validate_case_receipt(receipt, expected_case_id=case_id)
        result[case_id] = receipt
    present = {
        path.stem
        for path in (receipt_root / SUCCESS_DIRNAME).glob("*.json")
        if path.is_file() and not path.is_symlink()
    }
    extras = sorted(present.difference(case_ids))
    if extras:
        raise ExactMomentAuditError(f"unexpected successful receipts: {extras[:10]}")
    return result


def _ordered_receipt_fingerprint_sha256(
    output_root: Path, case_ids: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for case_id in case_ids:
        receipt, _ = _load_json(
            output_root / SUCCESS_DIRNAME / f"{case_id}.json",
            f"{case_id} rank receipt dependency",
        )
        validate_case_receipt(receipt, expected_case_id=case_id)
        digest.update(
            (case_id + " " + receipt["content_fingerprint"] + "\n").encode("utf-8")
        )
    return digest.hexdigest()


def _slurm_attempt_identity() -> tuple[str, int, str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    restart_raw = os.environ.get("SLURM_RESTART_COUNT", "0")
    if not job_id or not job_id.isdigit() or not restart_raw.isdigit():
        raise ExactMomentAuditError(
            "rank completion requires numeric Slurm job/restart metadata"
        )
    restart = int(restart_raw)
    return job_id, restart, f"{job_id}-r{restart}"


def _write_rank_completion(
    *,
    output_root: Path,
    selected: Sequence[str],
    rank: int,
    world: int,
    completed: int,
    skipped: int,
    backend_files: Mapping[str, Any],
    runtime: Mapping[str, Any],
    kernel_self_check: Mapping[str, Any],
    per_case_elapsed_seconds: Mapping[str, float],
    rank_elapsed_seconds: float,
) -> Path:
    job_id, restart, attempt_id = _slurm_attempt_identity()
    indices = striped_indices(len(selected), rank=rank, world_size=world)
    case_ids = tuple(selected[index] for index in indices)
    if set(per_case_elapsed_seconds) != set(case_ids) or any(
        not math.isfinite(value) or value < 0.0
        for value in per_case_elapsed_seconds.values()
    ):
        raise ExactMomentAuditError("rank per-case timing coverage is invalid")
    if not math.isfinite(rank_elapsed_seconds) or rank_elapsed_seconds + 1.0e-9 < sum(
        per_case_elapsed_seconds.values()
    ):
        raise ExactMomentAuditError("rank elapsed time does not cover case timings")
    document: dict[str, Any] = {
        "schema_id": RANK_RECEIPT_SCHEMA,
        "status": "complete",
        "algorithm_id": ALGORITHM_ID,
        "slurm_job_id": job_id,
        "slurm_restart_count": restart,
        "slurm_attempt_id": attempt_id,
        "rank": rank,
        "world_size": world,
        "node_count": int(os.environ["SLURM_JOB_NUM_NODES"]),
        "cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "hostname": socket.gethostname(),
        "assigned_indices": list(indices),
        "assigned_case_ids": list(case_ids),
        "assigned_case_count": len(case_ids),
        "assigned_case_set_sha256": newline_case_set_sha256(case_ids),
        "completed_count": completed,
        "skipped_valid_complete_count": skipped,
        "ordered_case_receipt_fingerprints_sha256": (
            _ordered_receipt_fingerprint_sha256(output_root, case_ids)
        ),
        "backend_files": backend_files,
        "runtime": dict(runtime),
        "execution_provenance": {
            "scientific_role": "timing/provenance only; excluded from case receipts",
            "kernel_self_check": dict(kernel_self_check),
            "rank_elapsed_seconds": rank_elapsed_seconds,
            "per_case_elapsed_seconds": {
                case_id: per_case_elapsed_seconds[case_id] for case_id in case_ids
            },
        },
    }
    if completed + skipped != len(case_ids):
        raise ExactMomentAuditError("rank completion counts do not close")
    document["content_fingerprint"] = content_fingerprint(document)
    path = output_root / "ranks" / attempt_id / f"rank-{rank:03d}.json"
    _atomic_create(path, canonical_json_bytes(document))
    return path


def validate_rank_coverage(
    *, output_root: Path, all_case_ids: Sequence[str], attempt_id: str
) -> dict[str, Any]:
    attempt_match = re.fullmatch(r"(?P<job>[0-9]+)-r(?P<restart>[0-9]+)", attempt_id)
    if attempt_match is None:
        raise ExactMomentAuditError("rank attempt ID must have form <job>-r<restart>")
    job_id = attempt_match.group("job")
    restart = int(attempt_match.group("restart"))
    observed_indices: list[int] = []
    rank_fingerprints: list[str] = []
    common_runtime_contract: dict[str, Any] | None = None
    expected_kernel_self_check = {
        "status": "pass",
        "points_dtype": "<f4",
        "connectivity_dtype": "<i8",
        "offsets_dtype": "<i8",
        "point_field_dtype": "<f4",
        "point_fields_read_only": True,
        "moment_reference_dtype": "<f8",
        "triangle_count": 1,
    }
    for rank in range(8):
        path = output_root / "ranks" / attempt_id / f"rank-{rank:03d}.json"
        document, _ = _load_json(path, f"rank {rank} completion")
        fingerprint = document.get("content_fingerprint")
        runtime = document.get("runtime")
        runtime_contract = (
            {
                key: runtime.get(key)
                for key in (
                    "python_executable",
                    "python",
                    "numpy",
                    "numba",
                    "vtk",
                    "system_dist_packages_excluded",
                    "python_no_user_site",
                    "python_unbuffered",
                    "sys_path",
                )
            }
            if isinstance(runtime, Mapping)
            else None
        )
        cache_dir = (
            runtime.get("numba_cache_dir") if isinstance(runtime, Mapping) else None
        )
        if (
            document.get("schema_id") != RANK_RECEIPT_SCHEMA
            or document.get("status") != "complete"
            or document.get("algorithm_id") != ALGORITHM_ID
            or document.get("slurm_job_id") != job_id
            or document.get("slurm_restart_count") != restart
            or document.get("slurm_attempt_id") != attempt_id
            or document.get("rank") != rank
            or document.get("world_size") != 8
            or document.get("node_count") != 8
            or document.get("cpus_per_task") != 2
            or not isinstance(runtime, Mapping)
            or runtime.get("system_dist_packages_excluded") is not True
            or runtime.get("python_no_user_site") is not True
            or runtime.get("python_unbuffered") is not True
            or not all(
                isinstance(runtime.get(key), str) and runtime.get(key)
                for key in (
                    "python_executable",
                    "python",
                    "numpy",
                    "numba",
                    "vtk",
                )
            )
            or not isinstance(runtime.get("sys_path"), list)
            or not all(isinstance(entry, str) for entry in runtime.get("sys_path", []))
            or any(
                Path(entry or ".").resolve() == SYSTEM_DIST_PACKAGES
                for entry in runtime.get("sys_path", [])
            )
            or not isinstance(cache_dir, str)
            or not Path(cache_dir).is_absolute()
            or Path(cache_dir).parts[-2:] != (attempt_id, f"rank-{rank:03d}")
            or not isinstance(fingerprint, str)
            or content_fingerprint(document) != fingerprint
        ):
            raise ExactMomentAuditError(f"rank {rank} completion contract differs")
        if common_runtime_contract is None:
            common_runtime_contract = runtime_contract
        elif runtime_contract != common_runtime_contract:
            raise ExactMomentAuditError("rank runtime/version contracts differ")
        indices = striped_indices(len(all_case_ids), rank=rank, world_size=8)
        cases = tuple(all_case_ids[index] for index in indices)
        first_case_receipt, _ = _load_json(
            output_root / SUCCESS_DIRNAME / f"{cases[0]}.json",
            f"rank {rank} first case receipt",
        )
        expected_backend_files = first_case_receipt["input_identity"]["preflight"][
            "backend_files"
        ]
        execution = document.get("execution_provenance")
        timings = (
            execution.get("per_case_elapsed_seconds")
            if isinstance(execution, Mapping)
            else None
        )
        if (
            document.get("assigned_indices") != list(indices)
            or document.get("assigned_case_ids") != list(cases)
            or document.get("assigned_case_count") != len(cases)
            or document.get("assigned_case_set_sha256")
            != newline_case_set_sha256(cases)
            or document.get("completed_count", 0)
            + document.get("skipped_valid_complete_count", 0)
            != len(cases)
            or document.get("ordered_case_receipt_fingerprints_sha256")
            != _ordered_receipt_fingerprint_sha256(output_root, cases)
            or document.get("backend_files") != expected_backend_files
            or not isinstance(execution, Mapping)
            or execution.get("kernel_self_check") != expected_kernel_self_check
            or not isinstance(timings, Mapping)
            or set(timings) != set(cases)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                for value in timings.values()
            )
            or isinstance(execution.get("rank_elapsed_seconds"), bool)
            or not isinstance(execution.get("rank_elapsed_seconds"), (int, float))
            or not math.isfinite(execution["rank_elapsed_seconds"])
            or execution["rank_elapsed_seconds"] + 1.0e-9 < sum(timings.values())
        ):
            raise ExactMomentAuditError(f"rank {rank} assigned coverage differs")
        observed_indices.extend(indices)
        rank_fingerprints.append(fingerprint)
    if tuple(sorted(observed_indices)) != tuple(range(EXPECTED_ALL_CASE_COUNT)) or len(
        observed_indices
    ) != len(set(observed_indices)):
        raise ExactMomentAuditError(
            "actual rank union does not cover 0..1799 exactly once"
        )
    return {
        "schema_id": "hiliftaeroml-exact-moment-rank-coverage-v1",
        "status": "pass",
        "slurm_job_id": job_id,
        "slurm_restart_count": restart,
        "slurm_attempt_id": attempt_id,
        "rank_count": 8,
        "case_count": len(observed_indices),
        "exact_union_0_through_1799_once": True,
        "rank_content_fingerprints_sha256": hashlib.sha256(
            "".join(
                f"{rank} {fingerprint}\n"
                for rank, fingerprint in enumerate(rank_fingerprints)
            ).encode("utf-8")
        ).hexdigest(),
        "common_runtime_contract": common_runtime_contract,
    }


def _require_no_active_locks(output_root: Path, case_ids: Sequence[str]) -> None:
    active: list[str] = []
    for case_id in case_ids:
        path = output_root / LOCK_DIRNAME / f"{case_id}.lock"
        if not path.exists():
            continue
        descriptor = os.open(path, os.O_RDWR)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active.append(case_id)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    if active:
        raise ExactMomentAuditError(f"active per-case workers remain: {active[:20]}")


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ExactMomentAuditError("aggregate distribution input is invalid")
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(values * values, dtype=np.float64))),
        "median": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _campaign_homogeneity(
    case_ids: Sequence[str],
    public_case_ids: Sequence[str],
    expected_source_bindings: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    public_set = set(public_case_ids)
    if (
        len(public_case_ids) != EXPECTED_PUBLIC_CASE_COUNT
        or len(public_set) != EXPECTED_PUBLIC_CASE_COUNT
        or not public_set.issubset(case_ids)
        or set(expected_source_bindings) != set(case_ids)
    ):
        raise ExactMomentAuditError("campaign expected source membership differs")
    first = receipts[case_ids[0]]
    first_preflight = first["input_identity"]["preflight"]
    expected = {
        "backend_files": first_preflight["backend_files"],
        "p_inf": first_preflight["p_inf"],
        "cell_chunk": first_preflight["cell_chunk"],
        "quadrature": first["quadrature"],
        "source_byte_count_convention": first["input_identity"]["native"][
            "source_byte_count_convention"
        ],
    }
    if expected["p_inf"] != 176.352:
        raise ExactMomentAuditError("campaign p_inf differs from the canonical value")
    public_binding_count = 0
    for case_id in case_ids:
        receipt = receipts[case_id]
        preflight = receipt["input_identity"]["preflight"]
        binding = preflight["frozen_source_binding"]
        expected_binding = expected_source_bindings[case_id]
        observed = {
            "backend_files": preflight["backend_files"],
            "p_inf": preflight["p_inf"],
            "cell_chunk": preflight["cell_chunk"],
            "quadrature": receipt["quadrature"],
            "source_byte_count_convention": receipt["input_identity"]["native"][
                "source_byte_count_convention"
            ],
        }
        if observed != expected:
            raise ExactMomentAuditError(
                f"{case_id} campaign code/normalization/chunk contract differs"
            )
        if binding != expected_binding:
            raise ExactMomentAuditError(
                f"{case_id} frozen per-case source descriptor differs"
            )
        if binding.get(
            "frozen_all1800_inventory_sha256"
        ) != EXPECTED_ALL1800_INVENTORY_SHA256 or (
            binding.get("public_surface_archive") is not None
            and binding.get("public_source_inventory_sha256")
            != EXPECTED_PUBLIC_IDENTITY_SHA256
        ):
            raise ExactMomentAuditError(
                f"{case_id} frozen source inventory binding differs"
            )
        is_public_binding = binding.get("public_surface_archive") is not None
        if is_public_binding != (case_id in public_set):
            raise ExactMomentAuditError(
                f"{case_id} public-source membership binding differs"
            )
        if is_public_binding:
            public_binding_count += 1
    if public_binding_count != EXPECTED_PUBLIC_CASE_COUNT:
        raise ExactMomentAuditError("campaign public-source binding count differs")
    return {
        **expected,
        "case_count": len(case_ids),
        "all_receipts_identical_for_campaign_contract": True,
        "all1800_inventory_sha256": EXPECTED_ALL1800_INVENTORY_SHA256,
        "public_inventory_sha256": EXPECTED_PUBLIC_IDENTITY_SHA256,
        "public_source_bound_case_count": public_binding_count,
    }


def _scope_summary(
    case_ids: Sequence[str], receipts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    exact_delta = np.asarray(
        [
            receipts[case]["comparison"]["exact_vs_published"][
                "calculated_minus_published"
            ]
            for case in case_ids
        ],
        dtype=np.float64,
    )
    vertex_delta = np.asarray(
        [
            receipts[case]["comparison"]["vertex_lumped_vs_published"][
                "calculated_minus_published"
            ]
            for case in case_ids
        ],
        dtype=np.float64,
    )
    correction = np.asarray(
        [
            receipts[case]["comparison"]["exact_minus_vertex_lumped_cm"]
            for case in case_ids
        ],
        dtype=np.float64,
    )
    exact_numerical_fail = [
        case
        for case in case_ids
        if receipts[case]["comparison"]["exact_vs_published"]["numerical_status"]
        == "fail"
    ]
    exact_ci_fail = [
        case
        for case in case_ids
        if receipts[case]["comparison"]["exact_vs_published"]["ci95_status"] == "fail"
    ]
    order = np.argsort(np.abs(exact_delta))[::-1][:20]
    force_failures = [
        case
        for case in case_ids
        if receipts[case]["comparison"]["force_acceptance"]["status"] != "pass"
    ]
    force_numerical_failures = [
        case
        for case in case_ids
        if any(
            check["numerical_status"] != "pass"
            for check in receipts[case]["comparison"]["force_acceptance"][
                "checks"
            ].values()
        )
    ]
    force_failed_key_counts = {
        key: sum(
            receipts[case]["comparison"]["force_acceptance"]["checks"][key]["status"]
            != "pass"
            for case in case_ids
        )
        for key in ("cd", "cl", "cdp", "clp", "cdv", "clv")
    }
    return {
        "case_count": len(case_ids),
        "exact_minus_published_cm": _distribution(exact_delta),
        "absolute_exact_minus_published_cm": _distribution(np.abs(exact_delta)),
        "vertex_lumped_minus_published_cm": _distribution(vertex_delta),
        "exact_minus_vertex_lumped_cm": _distribution(correction),
        "exact_numerical_pass_count": len(case_ids) - len(exact_numerical_fail),
        "exact_numerical_fail_count": len(exact_numerical_fail),
        "exact_ci95_pass_count": len(case_ids) - len(exact_ci_fail),
        "exact_ci95_fail_count": len(exact_ci_fail),
        "exact_numerical_fail_case_ids": exact_numerical_fail,
        "exact_ci95_fail_case_ids": exact_ci_fail,
        "force_acceptance": {
            "protocol_id": "hilift-native-truth-vs-monitor-ci95-v1",
            "pass_count": len(case_ids) - len(force_failures),
            "fail_count": len(force_failures),
            "fail_case_ids": force_failures,
            "numerical_only_fail_count": len(force_numerical_failures),
            "numerical_only_fail_case_ids": force_numerical_failures,
            "failed_key_counts": force_failed_key_counts,
        },
        "worst_20_absolute_exact_minus_published": [
            {
                "case_id": case_ids[int(index)],
                "calculated_minus_published": float(exact_delta[index]),
                "absolute_error": float(abs(exact_delta[index])),
                "published_cm": receipts[case_ids[int(index)]]["published"]["cm"],
                "exact_cm": receipts[case_ids[int(index)]]["comparison"][
                    "exact_vs_published"
                ]["calculated"],
            }
            for index in order
        ],
    }


def aggregate_document(
    *,
    all_case_ids: Sequence[str],
    public_case_ids: Sequence[str],
    universe_contract: Mapping[str, Any],
    expected_source_bindings: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    preserved_failed_attempt_count: int = 0,
    rank_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_all_case_universe(tuple(all_case_ids))
    if tuple(public_case_ids) != tuple(sorted(public_case_ids, key=case_sort_key)):
        raise ExactMomentAuditError("public aggregate order differs")
    if newline_case_set_sha256(public_case_ids) != EXPECTED_PUBLIC_NUMERIC_SHA256:
        raise ExactMomentAuditError("public aggregate case digest differs")
    if set(receipts) != set(all_case_ids):
        raise ExactMomentAuditError("aggregate receipt membership differs")
    for case_id in all_case_ids:
        validate_case_receipt(receipts[case_id], expected_case_id=case_id)
    homogeneity = _campaign_homogeneity(
        all_case_ids, public_case_ids, expected_source_bindings, receipts
    )
    document: dict[str, Any] = {
        "schema_id": AGGREGATE_SCHEMA,
        "status": "complete",
        "algorithm_id": ALGORITHM_ID,
        "scientific_role": (
            "read-only exact moment audit; no source field or published coefficient was modified"
        ),
        "universe_contract": universe_contract,
        "campaign_homogeneity": homogeneity,
        "all_1800_summary": _scope_summary(all_case_ids, receipts),
        "fluidsbench_union_1355_summary": _scope_summary(public_case_ids, receipts),
        "five_force_exception_cases": {
            case: {
                "exact_vs_published": receipts[case]["comparison"][
                    "exact_vs_published"
                ],
                "vertex_lumped_vs_published": receipts[case]["comparison"][
                    "vertex_lumped_vs_published"
                ],
                "exact_minus_vertex_lumped_cm": receipts[case]["comparison"][
                    "exact_minus_vertex_lumped_cm"
                ],
                "force_vs_published": receipts[case]["comparison"][
                    "force_vs_published"
                ],
            }
            for case in FORCE_EXCEPTION_CASES
        },
        "case_receipts": {
            "directory": SUCCESS_DIRNAME,
            "case_count": len(receipts),
            "ordered_fingerprints_sha256": hashlib.sha256(
                "".join(
                    case + " " + str(receipts[case]["content_fingerprint"]) + "\n"
                    for case in all_case_ids
                ).encode("utf-8")
            ).hexdigest(),
        },
        "rank_coverage": dict(rank_coverage or {}),
        "preserved_failed_attempts": {
            "scientific_role": "execution provenance only; valid completed receipt wins",
            "count": preserved_failed_attempt_count,
        },
        "source_fields_modified": False,
        "published_coefficients_modified": False,
    }
    document["content_fingerprint"] = content_fingerprint(document)
    return document


def compare_pilot_receipts(*, pilot_root_a: Path, pilot_root_b: Path) -> dict[str, Any]:
    """Compare two independent six-case runs with different fixed chunks."""

    roots = (
        _absolute_directory(pilot_root_a, "pilot root A"),
        _absolute_directory(pilot_root_b, "pilot root B"),
    )
    receipt_sets = tuple(_read_receipts(root, PILOT_CASES) for root in roots)
    cases: dict[str, Any] = {}
    global_max_absolute = 0.0
    global_max_relative = 0.0
    for case_id in PILOT_CASES:
        left = receipt_sets[0][case_id]
        right = receipt_sets[1][case_id]
        chunk_left = left["input_identity"]["preflight"]["cell_chunk"]
        chunk_right = right["input_identity"]["preflight"]["cell_chunk"]
        if chunk_left == chunk_right:
            raise ExactMomentAuditError("pilot replays must use different cell chunks")
        preflight_left = dict(left["input_identity"]["preflight"])
        preflight_right = dict(right["input_identity"]["preflight"])
        preflight_left.pop("cell_chunk")
        preflight_right.pop("cell_chunk")
        contract_left = {
            "algorithm_id": left["algorithm_id"],
            "case_id": left["case_id"],
            "preflight_except_cell_chunk": preflight_left,
            "native": left["input_identity"]["native"],
            "reference": left["reference"],
            "quadrature": left["quadrature"],
            "published": left["published"],
            "source_fields_modified": left["source_fields_modified"],
            "published_coefficients_modified": left["published_coefficients_modified"],
        }
        contract_right = {
            "algorithm_id": right["algorithm_id"],
            "case_id": right["case_id"],
            "preflight_except_cell_chunk": preflight_right,
            "native": right["input_identity"]["native"],
            "reference": right["reference"],
            "quadrature": right["quadrature"],
            "published": right["published"],
            "source_fields_modified": right["source_fields_modified"],
            "published_coefficients_modified": right["published_coefficients_modified"],
        }
        if contract_left != contract_right:
            raise ExactMomentAuditError(
                f"{case_id} pilot replay code/source/reference contract differs"
            )
        replay_contract_sha256 = hashlib.sha256(
            canonical_json_bytes(contract_left)
        ).hexdigest()
        values_left = np.asarray(
            [
                left["coefficients"]["force"][part][axis]
                for part in ("pressure", "viscous", "total")
                for axis in ("cd", "cy", "cl")
            ]
            + [
                left["coefficients"][quadrature][part]["cm_body_y"]
                for quadrature in ("moment_exact_degree2", "moment_vertex_lumped")
                for part in ("pressure", "viscous", "total")
            ],
            dtype=np.float64,
        )
        values_right = np.asarray(
            [
                right["coefficients"]["force"][part][axis]
                for part in ("pressure", "viscous", "total")
                for axis in ("cd", "cy", "cl")
            ]
            + [
                right["coefficients"][quadrature][part]["cm_body_y"]
                for quadrature in ("moment_exact_degree2", "moment_vertex_lumped")
                for part in ("pressure", "viscous", "total")
            ],
            dtype=np.float64,
        )
        absolute = np.abs(values_left - values_right)
        scale = np.maximum(np.maximum(np.abs(values_left), np.abs(values_right)), 1.0)
        relative = absolute / scale
        tolerance = PILOT_COMPARISON_ATOL + PILOT_COMPARISON_RTOL * scale
        if np.any(absolute > tolerance):
            raise ExactMomentAuditError(
                f"{case_id} pilot regrouping exceeds the recorded tolerance"
            )
        case_max_absolute = float(np.max(absolute))
        case_max_relative = float(np.max(relative))
        global_max_absolute = max(global_max_absolute, case_max_absolute)
        global_max_relative = max(global_max_relative, case_max_relative)
        cases[case_id] = {
            "cell_chunk_a": chunk_left,
            "cell_chunk_b": chunk_right,
            "replay_contract_sha256": replay_contract_sha256,
            "max_absolute_coefficient_difference": case_max_absolute,
            "max_scaled_relative_coefficient_difference": case_max_relative,
            "status": "pass",
        }
    document: dict[str, Any] = {
        "schema_id": PILOT_COMPARISON_SCHEMA,
        "status": "pass",
        "algorithm_id": ALGORITHM_ID,
        "case_count": len(PILOT_CASES),
        "case_ids": list(PILOT_CASES),
        "root_a": str(roots[0]),
        "root_b": str(roots[1]),
        "absolute_tolerance": PILOT_COMPARISON_ATOL,
        "scaled_relative_tolerance": PILOT_COMPARISON_RTOL,
        "global_max_absolute_coefficient_difference": global_max_absolute,
        "global_max_scaled_relative_coefficient_difference": global_max_relative,
        "cases": cases,
    }
    document["content_fingerprint"] = content_fingerprint(document)
    return document


def _required_files(case_ids: Sequence[str], dataset_root: Path) -> None:
    missing: list[str] = []
    for case_id in case_ids:
        case_dir = dataset_root / case_id
        for name in (
            f"boundary_{case_id}.vtu",
            f"ref_values_{case_id}.csv",
            f"force_mom_{case_id}.csv",
        ):
            path = case_dir / name
            if path.is_symlink() or not path.is_file():
                missing.append(str(path))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        raise ExactMomentAuditError(f"required case inputs are missing: {missing}")


def _slurm_rank(scope: str) -> tuple[int, int]:
    try:
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ["SLURM_NTASKS"])
        nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, ValueError) as error:
        raise ExactMomentAuditError(
            "dispatcher requires Slurm allocation metadata"
        ) from error
    expected_world = 1 if scope == "pilot" else 8
    if world != expected_world or nodes != world or cpus != 2:
        raise ExactMomentAuditError("Slurm allocation differs from the audit contract")
    return rank, world


def _require_approval(scope: str) -> None:
    expected = APPROVAL_PILOT if scope == "pilot" else APPROVAL_FULL
    if os.environ.get("HILIFT_MOMENT_AUDIT_APPROVED") != expected:
        raise ExactMomentAuditError("moment-audit approval sentinel differs")


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    dataset_root = _absolute_directory(args.dataset_root, "dataset root")
    recipe_root = _absolute_directory(args.recipe_root, "recipe root")
    output_root = _absolute_directory(
        args.output_root,
        "output root",
        create=args.mode in {"case", "dispatch", "aggregate", "validate"},
    )
    public = _regular_file(args.public_identity, "public-source identity")
    support = _regular_file(args.support_manifest, "support manifest")
    inventory_raw = (
        args.all1800_inventory
        if args.all1800_inventory is not None
        else recipe_root / ALL1800_INVENTORY_RELATIVE
    )
    inventory = _regular_file(inventory_raw, "all-1800 inventory")
    return dataset_root, recipe_root, output_root, public, support, inventory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode",
        required=True,
        choices=(
            "preflight",
            "case",
            "dispatch",
            "compare-pilots",
            "aggregate",
            "validate",
        ),
    )
    result.add_argument("--dataset-root", required=True, type=Path)
    result.add_argument("--recipe-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument(
        "--public-identity", type=Path, default=ROOT / PUBLIC_IDENTITY_RELATIVE
    )
    result.add_argument(
        "--support-manifest", type=Path, default=ROOT / SUPPORT_MANIFEST_RELATIVE
    )
    result.add_argument("--all1800-inventory", type=Path)
    result.add_argument("--scope", choices=("pilot", "full"), default="full")
    result.add_argument("--case-id")
    result.add_argument("--p-inf", type=float, default=176.352)
    result.add_argument("--cell-chunk", type=int, default=100_000)
    result.add_argument("--aggregate-output", type=Path)
    result.add_argument("--validation-receipt", type=Path)
    result.add_argument("--pilot-root-a", type=Path)
    result.add_argument("--pilot-root-b", type=Path)
    result.add_argument("--pilot-comparison-output", type=Path)
    result.add_argument("--rank-attempt-id")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        dataset_root, recipe_root, output_root, public, support, inventory = _paths(
            args
        )
        all_ids, public_ids, universe, source_bindings = load_universe_contract(
            dataset_root=dataset_root,
            public_identity_path=public,
            support_manifest_path=support,
            all1800_inventory_path=inventory,
        )
        selected = PILOT_CASES if args.scope == "pilot" else all_ids
        if args.mode == "preflight":
            backend = _backend(recipe_root)
            kernel_self_check = backend["kernel_self_check"]()
            _required_files(selected, dataset_root)
            world = 1 if args.scope == "pilot" else 8
            validate_stripe_coverage(len(selected), world_size=world)
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "scope": args.scope,
                        "case_count": len(selected),
                        "recommended_world_size": world,
                        "kernel_self_check": kernel_self_check,
                        "runtime": backend["runtime"],
                        "universe_contract": universe,
                        "source_fields_will_be_modified": False,
                        "job_submitted": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "case":
            if args.case_id is None or args.case_id not in selected:
                raise ExactMomentAuditError(
                    "--case-id must belong to the selected scope"
                )
            _require_approval(args.scope)
            status = run_case(
                dataset_root=dataset_root,
                recipe_root=recipe_root,
                output_root=output_root,
                case_id=args.case_id,
                p_inf=args.p_inf,
                cell_chunk=args.cell_chunk,
                source_binding=source_bindings[args.case_id],
            )
            print(
                json.dumps({"case_id": args.case_id, "status": status}, sort_keys=True)
            )
            return 0
        if args.mode == "dispatch":
            _require_approval(args.scope)
            rank, world = _slurm_rank(args.scope)
            validate_stripe_coverage(len(selected), world_size=world)
            rank_started = time.monotonic()
            backend = _backend(recipe_root)
            kernel_self_check = backend["kernel_self_check"]()
            completed = 0
            skipped = 0
            per_case_elapsed_seconds: dict[str, float] = {}
            for index in striped_indices(len(selected), rank=rank, world_size=world):
                case_id = selected[index]
                case_started = time.monotonic()
                status = run_case(
                    dataset_root=dataset_root,
                    recipe_root=recipe_root,
                    output_root=output_root,
                    case_id=case_id,
                    p_inf=args.p_inf,
                    cell_chunk=args.cell_chunk,
                    source_binding=source_bindings[case_id],
                    backend=backend,
                )
                elapsed_seconds = time.monotonic() - case_started
                per_case_elapsed_seconds[case_id] = elapsed_seconds
                completed += status == "completed"
                skipped += status == "skipped_valid_complete"
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "world_size": world,
                            "case_index": index,
                            "case_id": case_id,
                            "status": status,
                            "elapsed_seconds": elapsed_seconds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "status": "complete",
                        "completed": completed,
                        "skipped": skipped,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            rank_receipt = _write_rank_completion(
                output_root=output_root,
                selected=selected,
                rank=rank,
                world=world,
                completed=completed,
                skipped=skipped,
                backend_files=backend["files"],
                runtime=backend["runtime"],
                kernel_self_check=kernel_self_check,
                per_case_elapsed_seconds=per_case_elapsed_seconds,
                rank_elapsed_seconds=time.monotonic() - rank_started,
            )
            print(
                json.dumps(
                    {"rank": rank, "rank_completion": str(rank_receipt)},
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        if args.mode == "compare-pilots":
            if (
                args.scope != "pilot"
                or args.pilot_root_a is None
                or args.pilot_root_b is None
                or args.pilot_comparison_output is None
                or not args.pilot_comparison_output.is_absolute()
            ):
                raise ExactMomentAuditError(
                    "pilot comparison requires --scope pilot, both roots, and an absolute output"
                )
            comparison = compare_pilot_receipts(
                pilot_root_a=args.pilot_root_a, pilot_root_b=args.pilot_root_b
            )
            write_status = _atomic_create(
                args.pilot_comparison_output,
                canonical_json_bytes(comparison),
                accept_identical_existing=True,
            )
            print(
                json.dumps(
                    {**comparison, "artifact_write_status": write_status},
                    sort_keys=True,
                )
            )
            return 0
        if args.scope != "full":
            raise ExactMomentAuditError("aggregate and validate require --scope full")
        if args.aggregate_output is None or not args.aggregate_output.is_absolute():
            raise ExactMomentAuditError("--aggregate-output must be absolute")
        _require_no_active_locks(output_root, all_ids)
        if args.rank_attempt_id is None:
            raise ExactMomentAuditError(
                "aggregate/validate require --rank-attempt-id from the eight-rank run"
            )
        rank_coverage = validate_rank_coverage(
            output_root=output_root,
            all_case_ids=all_ids,
            attempt_id=args.rank_attempt_id,
        )
        receipts = _read_receipts(output_root, all_ids)
        failed_attempt_count = sum(
            1
            for path in (output_root / FAILURE_DIRNAME).glob("*/attempt-*.json")
            if path.is_file() and not path.is_symlink()
        )
        aggregate = aggregate_document(
            all_case_ids=all_ids,
            public_case_ids=public_ids,
            universe_contract=universe,
            expected_source_bindings=source_bindings,
            receipts=receipts,
            preserved_failed_attempt_count=failed_attempt_count,
            rank_coverage=rank_coverage,
        )
        aggregate_bytes = canonical_json_bytes(aggregate)
        if args.mode == "aggregate":
            write_status = _atomic_create(
                args.aggregate_output,
                aggregate_bytes,
                accept_identical_existing=True,
            )
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "case_count": len(all_ids),
                        "aggregate": str(args.aggregate_output),
                        "aggregate_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
                        "artifact_write_status": write_status,
                    },
                    sort_keys=True,
                )
            )
            return 0
        existing = _regular_file(args.aggregate_output, "aggregate output").read_bytes()
        if existing != aggregate_bytes:
            raise ExactMomentAuditError("aggregate source replay differs byte-for-byte")
        if args.validation_receipt is None or not args.validation_receipt.is_absolute():
            raise ExactMomentAuditError("--validation-receipt must be absolute")
        validation: dict[str, Any] = {
            "schema_id": VALIDATION_SCHEMA,
            "status": "pass",
            "algorithm_id": ALGORITHM_ID,
            "case_count": len(all_ids),
            "all_case_set_sha256": EXPECTED_ALL_CASE_SET_SHA256,
            "fluidsbench_case_count": len(public_ids),
            "fluidsbench_numeric_case_set_sha256": EXPECTED_PUBLIC_NUMERIC_SHA256,
            "aggregate_replay_complete": True,
            "source_reintegration_performed": False,
            "aggregate": {
                "path": str(args.aggregate_output.resolve()),
                "size_bytes": len(existing),
                "sha256": hashlib.sha256(existing).hexdigest(),
            },
            "source_fields_modified": False,
            "published_coefficients_modified": False,
        }
        validation["content_fingerprint"] = content_fingerprint(validation)
        write_status = _atomic_create(
            args.validation_receipt,
            canonical_json_bytes(validation),
            accept_identical_existing=True,
        )
        print(
            json.dumps(
                {**validation, "artifact_write_status": write_status}, sort_keys=True
            )
        )
        return 0
    except (ExactMomentAuditError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
