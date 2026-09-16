#!/usr/bin/env python3
"""Freeze and audit HiLiftAeroML public archive/member identities.

The authoritative, inexpensive binding is the immutable Hugging Face dataset
revision, the archive's LFS SHA-256 and byte size, and one exact regular VTU
member name and declared uncompressed size.  An evaluator must extract from the
verified archive object.  A hash of a pre-existing extracted VTU is optional
local evidence and does not, by itself, prove equivalence to the public object.

No command submits a job, downloads an archive, or mutates source data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "benchmark-specs" / "hiliftaeroml"
DEFAULT_SPLIT_DIR = DATASET_DIR / "splits"
DEFAULT_SUPPORT_DIR = (
    DATASET_DIR
    / "scoring-support"
    / "hiliftaeroml-native-all-splits-support-v1-candidate"
)
REPOSITORY_ID = "nvidia/HiLiftAeroML"
REVISION = "1c266d3869bc2968ff97d2107c9c3919be03ed32"
REVISION_URL = f"https://huggingface.co/datasets/{REPOSITORY_ID}/tree/{REVISION}"
SNAPSHOT_SCHEMA = "hiliftaeroml-hf-archive-metadata-snapshot-v1"
INVENTORY_SCHEMA = "hiliftaeroml-public-source-identity-v1"
LOCAL_RECEIPT_SCHEMA = "hiliftaeroml-local-extracted-vtu-attestation-v1"
ARCHIVE_RECEIPT_SCHEMA = "hiliftaeroml-verified-archive-member-audit-v1"
EXTRACTION_RULE = "exactly-one-regular-member-exact-basename-no-links-v1"
CASE_SET_SHA256_RULE = "sha256(utf8(case_id + newline) in listed order)"
BLOCK_SIZE = 16 * 1024 * 1024
EXPECTED_CASE_COUNT = 1355
EXPECTED_RELEASE_CASE_COUNT = 1800
EXPECTED_SPLITS = {
    "aoa",
    "deflection",
    "full",
    "geometry",
    "geometry_medium",
    "geometry_scarce",
    "geometry_super_scarce",
    "medium",
    "scarce",
    "single_aoa_12",
    "single_aoa_22",
    "single_aoa_4",
    "stall",
    "super_scarce",
}
CASE_RE = re.compile(r"^geo_LHC(?P<geometry>[0-9]{3})_AoA_(?P<aoa>[0-9]+)$")
ARCHIVE_RE = re.compile(
    r"^(?P<case>geo_LHC[0-9]{3}_AoA_[0-9]+)/"
    r"(?P<domain>boundary|volume)_(?P=case)\.vtu\.tgz$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")


class SourceIdentityError(ValueError):
    """Raised when public or local source identity evidence is ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceIdentityError(f"cannot encode canonical JSON: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceIdentityError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SourceIdentityError(f"{path} must contain one JSON object")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_size: int = BLOCK_SIZE) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(block_size), b""):
                digest.update(block)
    except OSError as error:
        raise SourceIdentityError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SourceIdentityError(f"{label} is not a lowercase SHA-256")
    return value


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o644)
    try:
        if path.exists():
            observed = path.read_bytes()
            if observed == payload:
                temporary.unlink()
                return
            raise SourceIdentityError(f"refusing to replace differing output {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def case_key(case_id: str) -> tuple[int, int]:
    match = CASE_RE.fullmatch(case_id)
    if match is None:
        raise SourceIdentityError(f"malformed HiLiftAeroML case ID {case_id!r}")
    geometry = int(match.group("geometry"))
    aoa = int(match.group("aoa"))
    if not 1 <= geometry <= 180 or aoa not in range(4, 23, 2):
        raise SourceIdentityError(f"out-of-domain HiLiftAeroML case ID {case_id!r}")
    return geometry, aoa


def ordered_case_digest(case_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{case_id}\n" for case_id in case_ids).encode("utf-8")
    ).hexdigest()


def exact_evaluation_union(split_dir: Path) -> list[str]:
    observed_files = {path.stem for path in split_dir.glob("*.json")}
    if observed_files != EXPECTED_SPLITS:
        raise SourceIdentityError(
            "split file set differs: "
            f"missing={sorted(EXPECTED_SPLITS - observed_files)}, "
            f"extra={sorted(observed_files - EXPECTED_SPLITS)}"
        )
    union: set[str] = set()
    for split_id in sorted(EXPECTED_SPLITS):
        document = read_json(split_dir / f"{split_id}.json")
        case_ids = document.get("case_ids")
        if (
            document.get("split_id") != split_id
            or not isinstance(case_ids, list)
            or any(not isinstance(case_id, str) for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
            or document.get("case_count") != len(case_ids)
            or document.get("case_set_sha256") != ordered_case_digest(case_ids)
        ):
            raise SourceIdentityError(f"split {split_id!r} is malformed or unpinned")
        for case_id in case_ids:
            case_key(case_id)
        union.update(case_ids)
    ordered = sorted(union, key=case_key)
    if len(ordered) != EXPECTED_CASE_COUNT:
        raise SourceIdentityError(
            f"evaluation union count differs: expected {EXPECTED_CASE_COUNT}, "
            f"observed {len(ordered)}"
        )
    return ordered


def _next_link(header: str | None) -> str | None:
    if not header:
        return None
    for item in header.split(","):
        match = re.match(r"\s*<([^>]+)>;\s*rel=\"next\"\s*$", item)
        if match:
            return match.group(1)
    return None


def iter_hf_tree_pages(
    *,
    opener: Any = urllib.request.urlopen,
) -> Iterator[list[dict[str, Any]]]:
    query = urllib.parse.urlencode(
        {"recursive": "true", "expand": "false", "limit": "1000"}
    )
    url = (
        f"https://huggingface.co/api/datasets/{REPOSITORY_ID}/tree/"
        f"{REVISION}?{query}"
    )
    while url is not None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise SourceIdentityError(f"refusing unexpected pagination URL {url!r}")
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "fluidsbench/1"}
        )
        try:
            with opener(request, timeout=60) as response:
                payload = response.read()
                link = response.headers.get("Link")
        except OSError as error:
            raise SourceIdentityError(f"Hugging Face tree request failed: {error}") from error
        try:
            page = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SourceIdentityError("Hugging Face tree returned invalid JSON") from error
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise SourceIdentityError("Hugging Face tree page is not an object list")
        yield page
        url = _next_link(link)


def build_hf_snapshot(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    archives: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str) or ARCHIVE_RE.fullmatch(path) is None:
            continue
        if path in archives:
            raise SourceIdentityError(f"duplicate public archive path {path!r}")
        match = ARCHIVE_RE.fullmatch(path)
        assert match is not None
        case_id = match.group("case")
        geometry, aoa = case_key(case_id)
        lfs = entry.get("lfs")
        if not isinstance(lfs, dict):
            raise SourceIdentityError(f"public archive {path!r} has no LFS identity")
        size = entry.get("size")
        if not isinstance(size, int) or size < 1 or lfs.get("size") != size:
            raise SourceIdentityError(f"public archive {path!r} has conflicting size")
        git_oid = entry.get("oid")
        if not isinstance(git_oid, str) or not GIT_OID_RE.fullmatch(git_oid):
            raise SourceIdentityError(f"public archive {path!r} has malformed Git oid")
        record: dict[str, Any] = {
            "case_id": case_id,
            "geometry_id": geometry,
            "aoa_degrees": aoa,
            "domain": "surface" if match.group("domain") == "boundary" else "volume",
            "repository_path": path,
            "size_bytes": size,
            "lfs_sha256": require_sha256(lfs.get("oid"), f"{path} LFS oid"),
            "git_blob_oid": git_oid,
        }
        xet_hash = entry.get("xetHash")
        if xet_hash is not None:
            record["xet_hash"] = require_sha256(xet_hash, f"{path} Xet hash")
        archives[path] = record
    expected = 2 * EXPECTED_RELEASE_CASE_COUNT
    if len(archives) != expected:
        raise SourceIdentityError(
            f"public archive count differs: expected {expected}, observed {len(archives)}"
        )
    ordered = sorted(
        archives.values(),
        key=lambda record: (
            record["geometry_id"],
            record["aoa_degrees"],
            record["domain"],
        ),
    )
    domains = {"surface": 0, "volume": 0}
    totals = {"surface": 0, "volume": 0}
    seen: set[tuple[str, str]] = set()
    for record in ordered:
        key = (record["case_id"], record["domain"])
        if key in seen:
            raise SourceIdentityError(f"duplicate public case/domain {key!r}")
        seen.add(key)
        domains[record["domain"]] += 1
        totals[record["domain"]] += record["size_bytes"]
    if domains != {
        "surface": EXPECTED_RELEASE_CASE_COUNT,
        "volume": EXPECTED_RELEASE_CASE_COUNT,
    }:
        raise SourceIdentityError(f"public domain coverage differs: {domains}")
    return {
        "$schema": (
            "https://fluidsbench.org/schemas/hiliftaeroml/"
            "hf-archive-metadata-snapshot-v1.schema.json"
        ),
        "schema_id": SNAPSHOT_SCHEMA,
        "schema_version": "1.0",
        "repository": {
            "id": REPOSITORY_ID,
            "type": "dataset",
            "revision": REVISION,
            "revision_url": REVISION_URL,
        },
        "archive_count": len(ordered),
        "archive_count_by_domain": domains,
        "archive_total_size_bytes_by_domain": totals,
        "archives": ordered,
    }


def fetch_hf_snapshot(*, opener: Any = urllib.request.urlopen) -> dict[str, Any]:
    return build_hf_snapshot(
        entry for page in iter_hf_tree_pages(opener=opener) for entry in page
    )


def _support_member_sizes(release_dir: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for chunk_path in sorted((release_dir / "case-sets").glob("caseset-*/chunk-*.json")):
        chunk = read_json(chunk_path)
        cases = chunk.get("cases")
        if not isinstance(cases, list):
            raise SourceIdentityError(f"support chunk {chunk_path} has no cases")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
                raise SourceIdentityError(f"support chunk {chunk_path} has malformed case")
            case_id = case["case_id"]
            domains: dict[str, int] = {}
            supports = case.get("support_instances")
            if not isinstance(supports, list):
                raise SourceIdentityError(f"support case {case_id} has no supports")
            for support in supports:
                if not isinstance(support, dict):
                    continue
                support_id = support.get("support_id")
                domain = {
                    "surface-native-points-v1": "surface",
                    "volume-native-valid-points-v1": "volume",
                }.get(support_id)
                if domain is None:
                    continue
                parameters = support.get("parameters")
                source = parameters.get("source_identity") if isinstance(parameters, dict) else None
                size = source.get("size_bytes") if isinstance(source, dict) else None
                filename = source.get("filename") if isinstance(source, dict) else None
                expected_filename = (
                    f"boundary_{case_id}.vtu" if domain == "surface" else f"volume_{case_id}.vtu"
                )
                if not isinstance(size, int) or size < 1 or filename != expected_filename:
                    raise SourceIdentityError(f"support source identity differs for {case_id} {domain}")
                domains[domain] = size
            if set(domains) != {"surface", "volume"}:
                raise SourceIdentityError(f"support domains are incomplete for {case_id}")
            old = result.setdefault(case_id, domains)
            if old != domains:
                raise SourceIdentityError(f"support member sizes conflict for {case_id}")
    return result


def _volume_records(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    payload = path.read_bytes()
    manifest = json.loads(payload)
    source_dataset = manifest.get("source_dataset") if isinstance(manifest, dict) else None
    if (
        not isinstance(source_dataset, dict)
        or source_dataset.get("repository_id") != REPOSITORY_ID
        or source_dataset.get("revision") != REVISION
        or manifest.get("case_count") != EXPECTED_RELEASE_CASE_COUNT
    ):
        raise SourceIdentityError("published volume provenance identity differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise SourceIdentityError("published volume provenance cases are absent")
    result = {
        case.get("case_id"): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    if len(result) != EXPECTED_RELEASE_CASE_COUNT:
        raise SourceIdentityError("published volume provenance cases are not unique/complete")
    return result, sha256_bytes(payload)


def _snapshot_records(snapshot: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    repository = snapshot.get("repository")
    if (
        snapshot.get("schema_id") != SNAPSHOT_SCHEMA
        or not isinstance(repository, dict)
        or repository.get("id") != REPOSITORY_ID
        or repository.get("revision") != REVISION
    ):
        raise SourceIdentityError("Hugging Face archive snapshot identity differs")
    archives = snapshot.get("archives")
    if not isinstance(archives, list):
        raise SourceIdentityError("Hugging Face archive snapshot has no archives")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in archives:
        if not isinstance(record, dict):
            raise SourceIdentityError("Hugging Face archive record is malformed")
        key = (record.get("case_id"), record.get("domain"))
        if not isinstance(key[0], str) or key[1] not in {"surface", "volume"}:
            raise SourceIdentityError("Hugging Face archive record key is malformed")
        if key in result:
            raise SourceIdentityError(f"duplicate Hugging Face archive record {key!r}")
        result[key] = record
    if len(result) != 2 * EXPECTED_RELEASE_CASE_COUNT:
        raise SourceIdentityError("Hugging Face archive snapshot is incomplete")
    return result


def _bound_file(
    descriptor: dict[str, Any],
    *,
    label: str,
    hash_content: bool,
) -> tuple[Path, int, str]:
    path_value = descriptor.get("path")
    size = descriptor.get("size_bytes")
    expected_sha = descriptor.get("sha256")
    if not isinstance(path_value, str) or not isinstance(size, int) or size < 1:
        raise SourceIdentityError(f"{label} descriptor is malformed")
    expected_sha = require_sha256(expected_sha, f"{label} descriptor")
    path = Path(path_value)
    try:
        observed_size = path.stat().st_size
    except OSError as error:
        raise SourceIdentityError(f"cannot stat {label} {path}: {error}") from error
    if observed_size != size or not path.is_file():
        raise SourceIdentityError(f"{label} size/type differs: {path}")
    if hash_content and sha256_file(path) != expected_sha:
        raise SourceIdentityError(f"{label} content SHA-256 differs: {path}")
    return path, size, expected_sha


def volume_stencil_attestations(
    *,
    receipt_dir: Path,
    case_ids: Sequence[str],
    expected_volume_sizes: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate the existing 1,355-case full-volume SHA receipt chain.

    Record and audit JSON are rehashed because they are small.  NPZ payloads
    are stat-bound to their receipt; their contents are not reread because the
    source VTU SHA lives in the receipt-bound record, not in the NPZ payload.
    """

    expected = set(case_ids)
    receipt_paths = {path.stem: path for path in receipt_dir.glob("*.json")}
    if set(receipt_paths) != expected:
        raise SourceIdentityError(
            "volume stencil receipt set differs: "
            f"missing={sorted(expected - set(receipt_paths))[:5]}, "
            f"extra={sorted(set(receipt_paths) - expected)[:5]}"
        )
    result: dict[str, dict[str, Any]] = {}
    chain_lines: list[str] = []
    inventory_bindings: set[tuple[str, int]] = set()
    campaign_bindings: set[tuple[str, int]] = set()
    origins: dict[str, int] = {}
    record_bytes = 0
    audit_bytes = 0
    payload_bytes = 0
    finalizer_reverified_count = 0
    for case_id in case_ids:
        receipt_path = receipt_paths[case_id]
        receipt_payload = receipt_path.read_bytes()
        receipt_sha = sha256_bytes(receipt_payload)
        try:
            receipt = json.loads(receipt_payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SourceIdentityError(f"cannot decode stencil receipt {receipt_path}") from error
        validation = receipt.get("validation")
        summary = validation.get("summary") if isinstance(validation, dict) else None
        source = summary.get("source") if isinstance(summary, dict) else None
        files = validation.get("files") if isinstance(validation, dict) else None
        if (
            receipt.get("case_id") != case_id
            or receipt.get("schema_id") != "hilift_native_prerequisite_case_receipt_v1"
            or receipt.get("domain") != "volume"
            or receipt.get("family") != "volume_velocity_profile_stencils"
            or receipt.get("status") != "complete"
            or not isinstance(summary, dict)
            or summary.get("case_id") != case_id
            or summary.get("status") != "complete"
            or summary.get("audit_status") != "pass"
            or not isinstance(source, dict)
            or not isinstance(files, dict)
        ):
            raise SourceIdentityError(f"volume stencil receipt fields differ for {case_id}")
        content_sha = require_sha256(source.get("sha256"), f"{case_id} source VTU")
        source_size = source.get("size_bytes")
        expected_relative = f"{case_id}/volume_{case_id}.vtu"
        if source_size != expected_volume_sizes.get(case_id) or source.get(
            "relative_path"
        ) != expected_relative:
            raise SourceIdentityError(f"volume stencil source identity differs for {case_id}")
        record_descriptor = files.get("record")
        audit_descriptor = files.get("audit")
        payload_descriptor = files.get("payload")
        if not all(
            isinstance(value, dict)
            for value in (record_descriptor, audit_descriptor, payload_descriptor)
        ):
            raise SourceIdentityError(f"volume stencil file bindings absent for {case_id}")
        record_path, record_size, record_sha = _bound_file(
            record_descriptor, label=f"{case_id} stencil record", hash_content=True
        )
        audit_path, audit_size, audit_sha = _bound_file(
            audit_descriptor, label=f"{case_id} stencil audit", hash_content=True
        )
        payload_path, payload_size, payload_sha = _bound_file(
            payload_descriptor, label=f"{case_id} stencil payload", hash_content=False
        )
        if record_path.name != f"native_velocity_profile_stencil_{case_id}.json":
            raise SourceIdentityError(f"stencil record path differs for {case_id}")
        if audit_path.name != f"{case_id}.json":
            raise SourceIdentityError(f"stencil audit path differs for {case_id}")
        if payload_path.name != f"native_velocity_profile_stencil_{case_id}.npz":
            raise SourceIdentityError(f"stencil payload path differs for {case_id}")
        record = read_json(record_path)
        audit = read_json(audit_path)
        if (
            record.get("schema_id")
            != "hilift_native_volume_velocity_profile_stencil_v1"
            or record.get("case_id") != case_id
            or record.get("status") != "complete"
            or record.get("source_vtu_sha256") != content_sha
            or record.get("source_vtu_size_bytes") != source_size
            or record.get("npz_sha256") != payload_sha
            or audit.get("case_id") != case_id
            or audit.get("status") != "pass"
        ):
            raise SourceIdentityError(f"stencil record/audit chain differs for {case_id}")
        inventory = receipt.get("inventory")
        campaign = receipt.get("table5_campaign")
        if not isinstance(inventory, dict) or not isinstance(campaign, dict):
            raise SourceIdentityError(f"stencil campaign bindings absent for {case_id}")
        inventory_bindings.add(
            (
                require_sha256(inventory.get("sha256"), f"{case_id} inventory"),
                inventory.get("size_bytes"),
            )
        )
        campaign_bindings.add(
            (
                require_sha256(campaign.get("sha256"), f"{case_id} Table-5 campaign"),
                campaign.get("size_bytes"),
            )
        )
        origin = receipt.get("origin")
        if not isinstance(origin, str):
            raise SourceIdentityError(f"stencil receipt origin absent for {case_id}")
        origins[origin] = origins.get(origin, 0) + 1
        reverified = source.get("full_sha256_reverified_by_finalizer")
        if reverified not in {True, False}:
            raise SourceIdentityError(f"stencil finalizer hash status absent for {case_id}")
        finalizer_reverified_count += int(reverified)
        result[case_id] = {
            "content_sha256": content_sha,
            "size_bytes": source_size,
            "attestation_origin": "velocity_profile_stencil_generator_full_file_sha256",
            "receipt_sha256": receipt_sha,
            "record_sha256": record_sha,
            "audit_sha256": audit_sha,
            "payload_sha256": payload_sha,
            "public_archive_equivalence": "not_established",
        }
        chain_lines.append(
            f"{case_id} {receipt_sha} {record_sha} {audit_sha} {payload_sha} "
            f"{content_sha}\n"
        )
        record_bytes += record_size
        audit_bytes += audit_size
        payload_bytes += payload_size
    if len(inventory_bindings) != 1 or len(campaign_bindings) != 1:
        raise SourceIdentityError("volume stencil receipts have conflicting campaign bindings")
    unique_source_sha_count = len(
        {attestation["content_sha256"] for attestation in result.values()}
    )
    inventory_sha, inventory_size = next(iter(inventory_bindings))
    campaign_sha, campaign_size = next(iter(campaign_bindings))
    evidence = {
        "schema_id": "hiliftaeroml-volume-stencil-source-content-attestation-set-v1",
        "case_count": len(result),
        "unique_source_content_sha256_count": unique_source_sha_count,
        "chain_sha256": sha256_bytes("".join(chain_lines).encode("utf-8")),
        "chain_sha256_rule": (
            "sha256(utf8(case_id receipt_sha256 record_sha256 audit_sha256 "
            "payload_sha256 source_content_sha256 + newline) in inventory case order)"
        ),
        "inventory_sha256": inventory_sha,
        "inventory_size_bytes": inventory_size,
        "table5_campaign_sha256": campaign_sha,
        "table5_campaign_size_bytes": campaign_size,
        "origins": dict(sorted(origins.items())),
        "record_bytes_rehashed": record_bytes,
        "audit_bytes_rehashed": audit_bytes,
        "payload_bytes_stat_bound": payload_bytes,
        "source_full_sha256_reverified_by_finalizer_count": finalizer_reverified_count,
        "source_hash_origin": (
            "Each receipt-bound stencil record reports the SHA-256 streamed over the "
            "complete source volume VTU by the original stencil generator."
        ),
        "public_archive_equivalence": "not_established",
    }
    return result, evidence


def build_inventory(
    *,
    split_dir: Path,
    support_release_dir: Path,
    volume_manifest_path: Path,
    hf_snapshot_path: Path,
    volume_stencil_receipt_dir: Path | None = None,
) -> dict[str, Any]:
    case_ids = exact_evaluation_union(split_dir)
    support_sizes = _support_member_sizes(support_release_dir)
    volume_records, volume_manifest_sha = _volume_records(volume_manifest_path)
    hf_payload = hf_snapshot_path.read_bytes()
    try:
        hf_snapshot = json.loads(hf_payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SourceIdentityError("cannot decode Hugging Face archive snapshot") from error
    hf_records = _snapshot_records(hf_snapshot)
    expected_volume_sizes = {
        case_id: support_sizes[case_id]["volume"] for case_id in case_ids
    }
    if volume_stencil_receipt_dir is None:
        volume_attestations: dict[str, dict[str, Any]] = {}
        volume_attestation_evidence: dict[str, Any] | None = None
    else:
        volume_attestations, volume_attestation_evidence = volume_stencil_attestations(
            receipt_dir=volume_stencil_receipt_dir,
            case_ids=case_ids,
            expected_volume_sizes=expected_volume_sizes,
        )
    cases: list[dict[str, Any]] = []
    aggregate = {
        "surface_archive_size_bytes": 0,
        "volume_archive_size_bytes": 0,
        "surface_member_declared_size_bytes": 0,
        "volume_member_declared_size_bytes": 0,
        "optional_extracted_content_sha256_count": 0,
        "local_surface_content_sha256_count": 0,
        "local_volume_content_sha256_count": 0,
        "local_extracted_content_sha256_count": 0,
    }
    for index, case_id in enumerate(case_ids):
        geometry, aoa = case_key(case_id)
        if case_id not in support_sizes or case_id not in volume_records:
            raise SourceIdentityError(f"source evidence is absent for {case_id}")
        domains: dict[str, Any] = {}
        for domain in ("surface", "volume"):
            public = hf_records.get((case_id, domain))
            if public is None:
                raise SourceIdentityError(f"public archive is absent for {case_id} {domain}")
            prefix = "boundary" if domain == "surface" else "volume"
            archive_path = f"{case_id}/{prefix}_{case_id}.vtu.tgz"
            member_path = f"{prefix}_{case_id}.vtu"
            if public.get("repository_path") != archive_path:
                raise SourceIdentityError(f"public archive path differs for {case_id} {domain}")
            member_size = support_sizes[case_id][domain]
            if domain == "volume":
                source = volume_records[case_id].get("source")
                if not isinstance(source, dict):
                    raise SourceIdentityError(f"volume provenance source absent for {case_id}")
                expected = {
                    "volume_archive_path": archive_path,
                    "volume_archive_size_bytes": public.get("size_bytes"),
                    "volume_archive_sha256": public.get("lfs_sha256"),
                    "extracted_volume_vtu_filename": member_path,
                    "extracted_volume_vtu_size_bytes": member_size,
                }
                if any(source.get(key) != value for key, value in expected.items()):
                    raise SourceIdentityError(
                        f"public and published volume provenance differ for {case_id}"
                    )
            archive_size = public.get("size_bytes")
            if not isinstance(archive_size, int) or archive_size < 1:
                raise SourceIdentityError(f"archive size is malformed for {case_id} {domain}")
            archive_sha = require_sha256(
                public.get("lfs_sha256"), f"{case_id} {domain} archive"
            )
            domains[domain] = {
                "archive": {
                    "repository_path": archive_path,
                    "size_bytes": archive_size,
                    "lfs_sha256": archive_sha,
                },
                "member": {
                    "path": member_path,
                    "type": "regular_file",
                    "declared_size_bytes": member_size,
                    "extraction_rule": EXTRACTION_RULE,
                    "extracted_content_sha256": None,
                    "extracted_content_sha256_status": "optional_not_computed",
                    "local_content_attestation": (
                        volume_attestations.get(case_id) if domain == "volume" else None
                    ),
                },
            }
            aggregate[f"{domain}_archive_size_bytes"] += archive_size
            aggregate[f"{domain}_member_declared_size_bytes"] += member_size
            if domains[domain]["member"]["local_content_attestation"] is not None:
                aggregate[f"local_{domain}_content_sha256_count"] += 1
                aggregate["local_extracted_content_sha256_count"] += 1
        cases.append(
            {
                "case_index": index,
                "case_id": case_id,
                "geometry_id": geometry,
                "aoa_degrees": aoa,
                "surface": domains["surface"],
                "volume": domains["volume"],
            }
        )
    aggregate["member_declared_size_bytes"] = (
        aggregate["surface_member_declared_size_bytes"]
        + aggregate["volume_member_declared_size_bytes"]
    )
    return {
        "$schema": (
            "https://fluidsbench.org/schemas/hiliftaeroml/"
            "public-source-identity-v1.schema.json"
        ),
        "schema_id": INVENTORY_SCHEMA,
        "schema_version": "1.0",
        "status": "complete_archive_member_binding",
        "dataset": {
            "repository_id": REPOSITORY_ID,
            "repository_type": "dataset",
            "revision": REVISION,
            "revision_url": REVISION_URL,
            "archive_hash_algorithm": "SHA-256",
            "archive_hash_source": "Hugging Face LFS object oid",
        },
        "binding_contract": {
            "authoritative_tier": "verified_archive_exact_member-v1",
            "rule": (
                "Verify the exact LFS object SHA-256 and byte size, then stream-extract "
                "exactly one regular member whose basename and declared size equal the "
                "case record; reject links, directories, additional members, path "
                "components, truncation, and trailing member ambiguity."
            ),
            "preexisting_extracted_file_policy": (
                "A pre-existing extracted VTU is not asserted equivalent unless it is "
                "re-extracted from the verified archive or its optional SHA-256 is "
                "compared with a SHA-256 streamed from that verified archive member."
            ),
            "optional_expensive_tier": "extracted-member-content-sha256-v1",
        },
        "source_evidence": {
            "hf_archive_snapshot_sha256": sha256_bytes(hf_payload),
            "published_volume_manifest_sha256": volume_manifest_sha,
            "volume_stencil_source_content_attestations": volume_attestation_evidence,
        },
        "case_set": {
            "case_count": len(case_ids),
            "case_ordering": "geometry_id ascending, then numeric AoA ascending",
            "case_set_sha256": ordered_case_digest(case_ids),
            "case_set_sha256_rule": CASE_SET_SHA256_RULE,
        },
        "aggregate": aggregate,
        "cases": cases,
    }


def validate_inventory(
    inventory: dict[str, Any],
    *,
    hf_snapshot_payload: bytes | None = None,
) -> None:
    if inventory.get("schema_id") != INVENTORY_SCHEMA:
        raise SourceIdentityError("inventory schema identity differs")
    dataset = inventory.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("repository_id") != REPOSITORY_ID
        or dataset.get("revision") != REVISION
    ):
        raise SourceIdentityError("inventory dataset identity differs")
    cases = inventory.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise SourceIdentityError("inventory case coverage is incomplete")
    case_ids: list[str] = []
    archive_totals = {"surface": 0, "volume": 0}
    member_totals = {"surface": 0, "volume": 0}
    extracted_hash_count = 0
    local_hash_counts = {"surface": 0, "volume": 0}
    snapshot_records: dict[tuple[str, str], dict[str, Any]] | None = None
    if hf_snapshot_payload is not None:
        expected_snapshot_sha = inventory.get("source_evidence", {}).get(
            "hf_archive_snapshot_sha256"
        )
        if sha256_bytes(hf_snapshot_payload) != expected_snapshot_sha:
            raise SourceIdentityError("Hugging Face archive snapshot digest differs")
        snapshot = json.loads(hf_snapshot_payload)
        snapshot_records = _snapshot_records(snapshot)
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise SourceIdentityError("inventory case is malformed")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case.get("case_index") != index:
            raise SourceIdentityError("inventory case order/index differs")
        geometry, aoa = case_key(case_id)
        if case.get("geometry_id") != geometry or case.get("aoa_degrees") != aoa:
            raise SourceIdentityError(f"inventory case attributes differ for {case_id}")
        case_ids.append(case_id)
        for domain, prefix in (("surface", "boundary"), ("volume", "volume")):
            record = case.get(domain)
            archive = record.get("archive") if isinstance(record, dict) else None
            member = record.get("member") if isinstance(record, dict) else None
            expected_archive = f"{case_id}/{prefix}_{case_id}.vtu.tgz"
            expected_member = f"{prefix}_{case_id}.vtu"
            if (
                not isinstance(archive, dict)
                or not isinstance(member, dict)
                or archive.get("repository_path") != expected_archive
                or member.get("path") != expected_member
                or PurePosixPath(expected_member).parts != (expected_member,)
                or member.get("type") != "regular_file"
                or member.get("extraction_rule") != EXTRACTION_RULE
            ):
                raise SourceIdentityError(f"archive/member path ambiguity for {case_id} {domain}")
            archive_size = archive.get("size_bytes")
            member_size = member.get("declared_size_bytes")
            if not isinstance(archive_size, int) or archive_size < 1:
                raise SourceIdentityError(f"archive size differs for {case_id} {domain}")
            if not isinstance(member_size, int) or member_size < 1:
                raise SourceIdentityError(f"member size differs for {case_id} {domain}")
            archive_sha = require_sha256(
                archive.get("lfs_sha256"), f"{case_id} {domain} archive"
            )
            content_sha = member.get("extracted_content_sha256")
            status = member.get("extracted_content_sha256_status")
            if content_sha is None:
                if status != "optional_not_computed":
                    raise SourceIdentityError(f"member hash status differs for {case_id} {domain}")
            else:
                require_sha256(content_sha, f"{case_id} {domain} member")
                if status != "complete_verified_archive_member":
                    raise SourceIdentityError(f"member hash provenance differs for {case_id} {domain}")
                extracted_hash_count += 1
            local_attestation = member.get("local_content_attestation")
            if local_attestation is not None:
                if (
                    not isinstance(local_attestation, dict)
                    or local_attestation.get("size_bytes") != member_size
                    or local_attestation.get("attestation_origin")
                    != "velocity_profile_stencil_generator_full_file_sha256"
                    or local_attestation.get("public_archive_equivalence")
                    != "not_established"
                ):
                    raise SourceIdentityError(
                        f"local member attestation differs for {case_id} {domain}"
                    )
                for key in (
                    "content_sha256",
                    "receipt_sha256",
                    "record_sha256",
                    "audit_sha256",
                    "payload_sha256",
                ):
                    require_sha256(
                        local_attestation.get(key),
                        f"{case_id} {domain} local attestation {key}",
                    )
                local_hash_counts[domain] += 1
            if snapshot_records is not None:
                public = snapshot_records.get((case_id, domain))
                if (
                    public is None
                    or public.get("repository_path") != expected_archive
                    or public.get("size_bytes") != archive_size
                    or public.get("lfs_sha256") != archive_sha
                ):
                    raise SourceIdentityError(f"public archive swap/tamper for {case_id} {domain}")
            archive_totals[domain] += archive_size
            member_totals[domain] += member_size
    if len(case_ids) != len(set(case_ids)) or case_ids != sorted(case_ids, key=case_key):
        raise SourceIdentityError("inventory case IDs are duplicate or misordered")
    case_set = inventory.get("case_set")
    if (
        not isinstance(case_set, dict)
        or case_set.get("case_count") != len(case_ids)
        or case_set.get("case_set_sha256") != ordered_case_digest(case_ids)
    ):
        raise SourceIdentityError("inventory case-set binding differs")
    aggregate = inventory.get("aggregate")
    expected_aggregate = {
        "surface_archive_size_bytes": archive_totals["surface"],
        "volume_archive_size_bytes": archive_totals["volume"],
        "surface_member_declared_size_bytes": member_totals["surface"],
        "volume_member_declared_size_bytes": member_totals["volume"],
        "member_declared_size_bytes": member_totals["surface"] + member_totals["volume"],
        "optional_extracted_content_sha256_count": extracted_hash_count,
        "local_surface_content_sha256_count": local_hash_counts["surface"],
        "local_volume_content_sha256_count": local_hash_counts["volume"],
        "local_extracted_content_sha256_count": sum(local_hash_counts.values()),
    }
    if aggregate != expected_aggregate:
        raise SourceIdentityError("inventory aggregate differs from case records")


def inventory_case(inventory: dict[str, Any], case_index: int) -> dict[str, Any]:
    cases = inventory.get("cases")
    if not isinstance(cases, list) or not 0 <= case_index < len(cases):
        raise SourceIdentityError(f"case index {case_index} is outside the inventory")
    case = cases[case_index]
    if not isinstance(case, dict) or case.get("case_index") != case_index:
        raise SourceIdentityError(f"case index {case_index} does not bind one case")
    return case


def hash_preexisting_case(
    *,
    inventory_path: Path,
    expected_inventory_sha256: str,
    dataset_root: Path,
    case_index: int,
    domains_to_hash: Sequence[str] = ("surface", "volume"),
) -> dict[str, Any]:
    payload = inventory_path.read_bytes()
    observed_inventory_sha = sha256_bytes(payload)
    if observed_inventory_sha != require_sha256(
        expected_inventory_sha256, "expected inventory"
    ):
        raise SourceIdentityError("inventory SHA-256 differs from the Slurm plan")
    inventory = json.loads(payload)
    validate_inventory(inventory)
    case = inventory_case(inventory, case_index)
    case_id = case["case_id"]
    domains: dict[str, Any] = {}
    if not domains_to_hash or any(
        domain not in {"surface", "volume"} for domain in domains_to_hash
    ):
        raise SourceIdentityError("domains_to_hash must contain surface and/or volume")
    if len(set(domains_to_hash)) != len(domains_to_hash):
        raise SourceIdentityError("domains_to_hash contains duplicates")
    for domain in domains_to_hash:
        member = case[domain]["member"]
        path = dataset_root / case_id / member["path"]
        try:
            before = path.stat()
        except OSError as error:
            raise SourceIdentityError(f"cannot stat {path}: {error}") from error
        if before.st_size != member["declared_size_bytes"] or not path.is_file():
            raise SourceIdentityError(f"pre-existing extracted size/type differs for {path}")
        digest = sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise SourceIdentityError(f"pre-existing extracted file changed while hashing: {path}")
        domains[domain] = {
            "relative_path": f"{case_id}/{member['path']}",
            "size_bytes": after.st_size,
            "content_sha256": digest,
            "equivalence_to_verified_public_archive": "not_established",
        }
    return {
        "schema_id": LOCAL_RECEIPT_SCHEMA,
        "schema_version": "1.0",
        "inventory_sha256": observed_inventory_sha,
        "case_index": case_index,
        "case_id": case_id,
        "attestation_scope": "pre-existing-local-extracted-files",
        "public_equivalence_policy": (
            "These hashes are local evidence only until compared with hashes streamed "
            "from exact members of the verified public LFS archives."
        ),
        "domains": domains,
    }


class DigestingReader(io.RawIOBase):
    """Read-only wrapper that hashes every compressed byte consumed."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        self.byte_count += len(data)
        return data


def audit_local_archive(
    *,
    archive_path: Path,
    case_record: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    if domain not in {"surface", "volume"}:
        raise SourceIdentityError(f"unsupported domain {domain!r}")
    case_id = case_record.get("case_id")
    source = case_record.get(domain)
    archive = source.get("archive") if isinstance(source, dict) else None
    expected_member = source.get("member") if isinstance(source, dict) else None
    if not isinstance(archive, dict) or not isinstance(expected_member, dict):
        raise SourceIdentityError("case archive/member record is malformed")
    if archive_path.stat().st_size != archive.get("size_bytes"):
        raise SourceIdentityError("local archive byte size differs before audit")
    member_records: list[dict[str, Any]] = []
    with archive_path.open("rb") as raw:
        digesting = DigestingReader(raw)
        try:
            with tarfile.open(fileobj=digesting, mode="r|gz") as tar:
                for member in tar:
                    if (
                        not member.isfile()
                        or PurePosixPath(member.name).parts != (member.name,)
                        or member.name != expected_member.get("path")
                        or member.size != expected_member.get("declared_size_bytes")
                    ):
                        raise SourceIdentityError(
                            f"archive contains an ambiguous/nonconforming member {member.name!r}"
                        )
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise SourceIdentityError("regular archive member cannot be opened")
                    member_digest = hashlib.sha256()
                    member_bytes = 0
                    for block in iter(lambda: extracted.read(BLOCK_SIZE), b""):
                        member_digest.update(block)
                        member_bytes += len(block)
                    if member_bytes != member.size:
                        raise SourceIdentityError("archive member is truncated")
                    member_records.append(
                        {
                            "path": member.name,
                            "size_bytes": member_bytes,
                            "content_sha256": member_digest.hexdigest(),
                        }
                    )
        except (tarfile.TarError, OSError) as error:
            raise SourceIdentityError(f"cannot stream-audit archive {archive_path}: {error}") from error
        while digesting.read(BLOCK_SIZE):
            pass
        archive_sha = digesting.digest.hexdigest()
        archive_bytes = digesting.byte_count
    if len(member_records) != 1:
        raise SourceIdentityError(
            f"archive must contain exactly one regular member, observed {len(member_records)}"
        )
    if archive_bytes != archive.get("size_bytes") or archive_sha != archive.get("lfs_sha256"):
        raise SourceIdentityError("local archive LFS identity differs")
    return {
        "schema_id": ARCHIVE_RECEIPT_SCHEMA,
        "schema_version": "1.0",
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "case_id": case_id,
        "domain": domain,
        "archive": {
            "repository_path": archive["repository_path"],
            "size_bytes": archive_bytes,
            "lfs_sha256": archive_sha,
        },
        "member": member_records[0],
        "public_equivalence": "established_by_verified_archive_stream_extraction",
    }


def resource_plan(inventory: dict[str, Any]) -> dict[str, Any]:
    validate_inventory(inventory)
    cases = inventory["cases"]
    per_case = sorted(
        sum(
            case[domain]["member"]["declared_size_bytes"]
            for domain in ("surface", "volume")
            if case[domain]["member"]["local_content_attestation"] is None
        )
        for case in cases
    )
    total = sum(per_case)
    def percentile(fraction: float) -> int:
        return per_case[min(len(per_case) - 1, int(fraction * (len(per_case) - 1)))]
    return {
        "status": "plan_only_not_submitted",
        "activation_binding_requires_bulk_member_hash": False,
        "optional_local_attestation": {
            "case_count": len(per_case),
            "domains_without_existing_full_content_sha256": [
                domain
                for domain in ("surface", "volume")
                if any(
                    case[domain]["member"]["local_content_attestation"] is None
                    for case in cases
                )
            ],
            "read_bytes": total,
            "read_TB_decimal": total / 1_000_000_000_000,
            "read_TiB": total / (1024**4),
            "per_case_bytes": {
                "minimum": per_case[0],
                "median": (per_case[677] if len(per_case) == 1355 else percentile(0.5)),
                "p95": percentile(0.95),
                "maximum": per_case[-1],
            },
            "aggregate_read_floor_hours": {
                str(mbps): total / (mbps * 1_000_000) / 3600
                for mbps in (250, 500, 750, 1000)
            },
            "slurm": {
                "partition": "cpu",
                "cpus_per_task": 1,
                "memory": "2G",
                "time_limit": "03:00:00",
                "maximum_concurrent_array_tasks": 2,
                "array_ranges": ["0-1023%2", "1024-1354%2 afterok first array"],
                "gpu_count": 0,
            },
        },
    }


def _case_index_argument(value: int | None) -> int:
    if value is not None:
        return value
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise SourceIdentityError("--case-index or SLURM_ARRAY_TASK_ID is required")
    try:
        return int(raw)
    except ValueError as error:
        raise SourceIdentityError("SLURM_ARRAY_TASK_ID is not an integer") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-hf-snapshot")
    fetch.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    build.add_argument("--support-release-dir", type=Path, default=DEFAULT_SUPPORT_DIR)
    build.add_argument("--volume-manifest", type=Path, required=True)
    build.add_argument("--hf-snapshot", type=Path, required=True)
    build.add_argument("--volume-stencil-receipt-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--inventory", type=Path, required=True)
    verify.add_argument("--hf-snapshot", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--inventory", type=Path, required=True)
    local_hash = commands.add_parser("hash-extracted-case")
    local_hash.add_argument("--inventory", type=Path, required=True)
    local_hash.add_argument("--expected-inventory-sha256", required=True)
    local_hash.add_argument("--dataset-root", type=Path, required=True)
    local_hash.add_argument("--case-index", type=int)
    local_hash.add_argument(
        "--domain",
        choices=("surface", "volume", "both"),
        default="both",
    )
    local_hash.add_argument("--receipt-dir", type=Path, required=True)
    archive = commands.add_parser("audit-local-archive")
    archive.add_argument("--inventory", type=Path, required=True)
    archive.add_argument("--expected-inventory-sha256", required=True)
    archive.add_argument("--case-index", type=int, required=True)
    archive.add_argument("--domain", choices=("surface", "volume"), required=True)
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fetch-hf-snapshot":
            payload = canonical_json_bytes(fetch_hf_snapshot())
            write_atomic(args.output, payload)
            print(json.dumps({"status": "complete", "sha256": sha256_bytes(payload)}))
        elif args.command == "build":
            inventory = build_inventory(
                split_dir=args.split_dir,
                support_release_dir=args.support_release_dir,
                volume_manifest_path=args.volume_manifest,
                hf_snapshot_path=args.hf_snapshot,
                volume_stencil_receipt_dir=args.volume_stencil_receipt_dir,
            )
            validate_inventory(inventory, hf_snapshot_payload=args.hf_snapshot.read_bytes())
            payload = canonical_json_bytes(inventory)
            write_atomic(args.output, payload)
            print(json.dumps({"status": inventory["status"], "sha256": sha256_bytes(payload)}))
        elif args.command == "verify":
            payload = args.inventory.read_bytes()
            inventory = json.loads(payload)
            validate_inventory(inventory, hf_snapshot_payload=args.hf_snapshot.read_bytes())
            print(json.dumps({"status": "pass", "sha256": sha256_bytes(payload)}))
        elif args.command == "plan":
            print(json.dumps(resource_plan(read_json(args.inventory)), indent=2, sort_keys=True))
        elif args.command == "hash-extracted-case":
            receipt = hash_preexisting_case(
                inventory_path=args.inventory,
                expected_inventory_sha256=args.expected_inventory_sha256,
                dataset_root=args.dataset_root,
                case_index=_case_index_argument(args.case_index),
                domains_to_hash=(
                    ("surface", "volume")
                    if args.domain == "both"
                    else (args.domain,)
                ),
            )
            domain_tag = "-".join(receipt["domains"])
            path = args.receipt_dir / (
                f"{receipt['case_index']:04d}-{receipt['case_id']}-{domain_tag}.json"
            )
            payload = canonical_json_bytes(receipt)
            write_atomic(path, payload)
            print(json.dumps({"status": "complete", "receipt": str(path), "sha256": sha256_bytes(payload)}))
        elif args.command == "audit-local-archive":
            payload = args.inventory.read_bytes()
            if sha256_bytes(payload) != require_sha256(
                args.expected_inventory_sha256, "expected inventory"
            ):
                raise SourceIdentityError("inventory SHA-256 differs")
            inventory = json.loads(payload)
            validate_inventory(inventory)
            receipt = audit_local_archive(
                archive_path=args.archive,
                case_record=inventory_case(inventory, args.case_index),
                domain=args.domain,
            )
            output_payload = canonical_json_bytes(receipt)
            write_atomic(args.output, output_payload)
            print(json.dumps({"status": "complete", "sha256": sha256_bytes(output_payload)}))
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (OSError, SourceIdentityError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
