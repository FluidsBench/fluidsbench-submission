"""Download and verify the prepared DrivAerNet++ scoring support.

Participants never regenerate the support.  ``support-distribution-v1.json``
pins every archive (URL, size, SHA-256, case list); the per-case
``case-support.json`` records retained in Git pin every member inside them.
Archives stay compressed in the cache and are read member by member, so the
cache needs only the archive bytes (about 6 GB for official_test).
"""

from __future__ import annotations

import fcntl
import http.client
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contract import (
    CASE_SET_ID,
    DATASET_ID,
    DISTRIBUTION_PATH,
    DISTRIBUTION_SCHEMA,
    RELEASE_DIR,
    RELEASE_ID,
    RELEASE_TAG,
    ContractError,
    parse_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .native import SUPPORT_RULES_VERSION
from .support import load_npz_bytes, validate_case_support

BLOCK_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024**3
ARCHIVE_NAME_RE = re.compile(rf"{DATASET_ID}-support-\d{{3}}\.zip\Z")


class SupportDistributionError(ContractError):
    """Raised when prepared support is missing, altered, or unsafe."""


def manifest_sha256() -> str:
    return sha256_file(RELEASE_DIR / "manifest.json")


@lru_cache(maxsize=1)
def load_distribution() -> dict[str, Any]:
    index = read_json(DISTRIBUTION_PATH, "DrivAerNet++ support distribution")
    header = {
        "schema": DISTRIBUTION_SCHEMA,
        "schema_version": 1,
        "status": "dev_prerelease",
        "dataset_id": DATASET_ID,
        "scientific_release_id": RELEASE_ID,
        "scientific_manifest_sha256": manifest_sha256(),
        "rules_version": SUPPORT_RULES_VERSION,
        "release_tag": RELEASE_TAG,
    }
    if any(index.get(key) != value for key, value in header.items()):
        raise SupportDistributionError("support distribution bindings differ from the candidate manifest")
    archives = index.get("archives")
    if not isinstance(archives, list) or not archives:
        raise SupportDistributionError("support distribution lists no archives")
    seen: set[str] = set()
    for entry in archives:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("file"), str)
            or ARCHIVE_NAME_RE.fullmatch(entry["file"]) is None
            or not isinstance(entry.get("url"), str)
            or not entry["url"].startswith("https://")
            or not entry["url"].endswith("/" + entry["file"])
            or type(entry.get("size_bytes")) is not int
            or not 0 < entry["size_bytes"] < MAX_ARCHIVE_BYTES
            or not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or not isinstance(entry.get("case_ids"), list)
            or not entry["case_ids"]
        ):
            raise SupportDistributionError("invalid support archive binding")
        if seen.intersection(entry["case_ids"]):
            raise SupportDistributionError("a case appears in more than one support archive")
        seen.update(entry["case_ids"])
    if index.get("case_count") != len(seen):
        raise SupportDistributionError("support distribution case count differs")
    return index


def archive_for_case(case_id: str) -> dict[str, Any]:
    for entry in load_distribution()["archives"]:
        if case_id in entry["case_ids"]:
            return entry
    raise SupportDistributionError(f"no support archive contains {case_id}")


def cache_root(destination: Path | str | None = None) -> Path:
    if destination is not None:
        return Path(destination).expanduser().resolve()
    configured = os.environ.get("FLUIDSBENCH_DRIVAERNETPLUSPLUS_SUPPORT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return (base.expanduser() / "fluidsbench" / DATASET_ID / manifest_sha256()).resolve()


def _verify_archive_file(path: Path, entry: dict[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise SupportDistributionError(f"missing or unsafe support archive: {path}")
    if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
        raise SupportDistributionError(f"support archive size or SHA-256 differs: {path}")


class _HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise SupportDistributionError("support download redirected outside HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(entry: dict[str, Any], target: Path) -> None:
    opener = urllib.request.build_opener(_HTTPSRedirects())
    for attempt in range(3):
        try:
            size = 0
            request = urllib.request.Request(
                entry["url"], headers={"User-Agent": "FluidsBench-DrivAerNetPlusPlus-support/1"}
            )
            with opener.open(request, timeout=60) as response, target.open("wb") as output:
                while block := response.read(BLOCK_BYTES):
                    size += len(block)
                    if size > entry["size_bytes"]:
                        raise SupportDistributionError("support archive exceeds its pinned size")
                    output.write(block)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            if attempt == 2:
                raise SupportDistributionError(
                    f"support download failed after three attempts: {error}"
                ) from error
            time.sleep(attempt + 1)


def _install_from_mirror(entry: dict[str, Any], mirror: Path, target: Path) -> None:
    source = (mirror / entry["file"]).resolve()
    _verify_archive_file(source, entry)
    try:
        os.link(source, target)  # no second copy when the mirror shares the filesystem
    except OSError:
        shutil.copyfile(source, target)


def ensure_archive(
    entry: dict[str, Any],
    *,
    destination: Path | str | None = None,
    offline: bool = False,
    mirror: Path | str | None = None,
) -> Path:
    """Return a verified local archive, downloading it at most once."""

    root = cache_root(destination)
    path = root / entry["file"]
    if offline:
        _verify_archive_file(path, entry)
        return path
    root.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    with os.fdopen(os.open(root / f".{entry['file']}.lock", flags, 0o600), "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() or path.is_symlink():
            try:
                _verify_archive_file(path, entry)
                return path
            except SupportDistributionError:
                # Keep a damaged archive for inspection rather than deleting it.
                path.rename(root / f".{entry['file']}.replaced-{time.time_ns()}")
        if shutil.disk_usage(root).free < entry["size_bytes"]:
            raise SupportDistributionError(f"insufficient space for {entry['file']}")
        partial = root / f".{entry['file']}.part"
        partial.unlink(missing_ok=True)
        try:
            if mirror is not None:
                _install_from_mirror(entry, Path(mirror).expanduser(), partial)
            else:
                _download(entry, partial)
            _verify_archive_file(partial, entry)
            partial.rename(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def fetch_support(
    case_ids: Iterable[str],
    *,
    destination: Path | str | None = None,
    offline: bool = False,
    mirror: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    wanted = list(case_ids)
    entries = []
    for entry in load_distribution()["archives"]:
        if set(entry["case_ids"]).intersection(wanted):
            entries.append(entry)
    covered = {case_id for entry in entries for case_id in entry["case_ids"]}
    missing = sorted(set(wanted) - covered)
    if missing:
        raise SupportDistributionError(f"support distribution omits {missing[:5]}")
    plan = {
        "root": str(cache_root(destination)),
        "release_id": RELEASE_ID,
        "manifest_sha256": manifest_sha256(),
        "archive_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "case_count": len(wanted),
    }
    if not dry_run:
        for number, entry in enumerate(entries, 1):
            ensure_archive(entry, destination=destination, offline=offline, mirror=mirror)
            print(f"[{number}/{len(entries)}] {entry['file']}: verified", flush=True)
    return plan


@dataclass(frozen=True)
class CaseSupport:
    case_id: str
    record: dict[str, Any]
    record_sha256: str
    arrays: dict[str, np.ndarray]

    @property
    def point_count(self) -> int:
        return int(self.record["native"]["point_count"])


def retained_record(case_id: str) -> tuple[dict[str, Any], bytes]:
    path = RELEASE_DIR / "case-sets" / CASE_SET_ID / "case-artifacts" / case_id / "case-support.json"
    payload = path.read_bytes()
    return parse_json_bytes(payload, f"{case_id} case-support record"), payload


def load_case_support(case_id: str, *, destination: Path | str | None = None) -> CaseSupport:
    """Read one case from its verified local archive; never touches the network."""

    entry = archive_for_case(case_id)
    archive_path = cache_root(destination) / entry["file"]
    if archive_path.is_symlink() or not archive_path.is_file() or archive_path.stat().st_size != entry["size_bytes"]:
        raise SupportDistributionError(
            f"support archive for {case_id} is missing; run fetch-support first ({archive_path})"
        )
    record, record_bytes = retained_record(case_id)
    expected = {f"{case_id}/support.npz", f"{case_id}/case-support.json"}
    with zipfile.ZipFile(archive_path) as archive:
        members = {item.filename: item for item in archive.infolist() if item.filename.startswith(case_id + "/")}
        if set(members) != expected:
            raise SupportDistributionError(f"{case_id}: archive members differ from the contract")
        for item in members.values():
            mode = item.external_attr >> 16
            if item.is_dir() or item.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise SupportDistributionError(f"{case_id}: unsafe archive member {item.filename}")
        if members[f"{case_id}/support.npz"].file_size != record["support_npz"]["size_bytes"]:
            raise SupportDistributionError(f"{case_id}: support NPZ size differs")
        archived_record = archive.read(f"{case_id}/case-support.json")
        payload = archive.read(f"{case_id}/support.npz")
    if archived_record != record_bytes:
        raise SupportDistributionError(f"{case_id}: archived case-support record differs from Git")
    if sha256_bytes(payload) != record["support_npz"]["sha256"]:
        raise SupportDistributionError(f"{case_id}: support NPZ SHA-256 differs")
    arrays = load_npz_bytes(payload, f"{case_id} support")
    validate_case_support(record, arrays, case_id)
    return CaseSupport(case_id, record, sha256_bytes(record_bytes), arrays)
