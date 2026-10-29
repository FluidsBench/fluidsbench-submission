"""Frozen identities of the candidate DrivAerNet++ pressure-only contract."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "drivaernetplusplus"
DATASET_NAME = "DrivAerNet++"
SPEC_ROOT = ROOT / "benchmark-specs" / DATASET_ID
SPEC_PATH = SPEC_ROOT / "submission-spec.json"

DATASET_VERSION = "drivaernetplusplus-native-pressure-v1-candidate"
EVALUATION_REFERENCE_VERSION = "drivaernetplusplus-evaluator-v0.1-candidate"
RELEASE_ID = "drivaernetplusplus-native-pressure-support-v1-candidate"
SUPPORT_ID = "drivaernetplusplus-surface-native-points-v1"
QUANTITY_ID = "surface_pressure"
SPLIT_ID = "official_test"
CASE_SET_ID = "official_test"
RELEASE_DIR = SPEC_ROOT / "scoring-support" / RELEASE_ID
DISTRIBUTION_PATH = SPEC_ROOT / "support-distribution-v1.json"
SOURCE_IDENTITY_PATH = SPEC_ROOT / "public-source-identity" / "drivaernetplusplus-public-source-identity-v1.json"

DISTRIBUTION_SCHEMA = "fluidsbench-drivaernetplusplus-support-distribution-v1"
CASE_SUPPORT_SCHEMA = "fluidsbench-drivaernetplusplus-case-support-v1"
SOURCE_IDENTITY_SCHEMA = "fluidsbench-drivaernetplusplus-public-source-identity-v1"
RELEASE_TAG = "drivaernetplusplus-support-v1-dev"
DEFAULT_BASE_URL = (
    "https://github.com/neilashton/fluidsbench-submission/releases/download/"
    f"{RELEASE_TAG}/"
)

CASE_ID_RE = re.compile(r"(?:[EFN]_S_(?:WW|WWC|WWS)_WM_\d{3}|F_D_WM_WW_\d{4})\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SUBMISSION_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}\Z")


class ContractError(ValueError):
    """Raised when a DrivAerNet++ contract artifact differs from its pin."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(token: str) -> None:
    raise ContractError(f"non-finite JSON value {token!r}")


def parse_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot parse {label}: {error}") from error


def read_json(path: Path, label: str | None = None) -> Any:
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read {label or path}: {error}") from error
    return parse_json_bytes(payload, label or str(path))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> str:
    payload = json_bytes(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)
    return sha256_bytes(payload)


def digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def load_specification() -> dict[str, Any]:
    spec = read_json(SPEC_PATH, "DrivAerNet++ specification")
    if spec.get("dataset_id") != DATASET_ID:
        raise ContractError("specification dataset_id differs")
    return spec


def official_split(split_id: str = SPLIT_ID) -> dict[str, Any]:
    """Return the split index after checking it against the specification pin."""

    spec = load_specification()
    matches = [item for item in spec.get("splits", []) if item.get("id") == split_id]
    if len(matches) != 1:
        raise ContractError(
            f"split_id must be one of {[item.get('id') for item in spec.get('splits', [])]}"
        )
    declaration = matches[0]
    path = SPEC_ROOT / declaration["index_file"]
    if sha256_file(path) != declaration["sha256"]:
        raise ContractError("official split SHA-256 differs from the specification")
    split = read_json(path, "split index")
    case_ids = split.get("case_ids")
    if (
        split.get("dataset_id") != DATASET_ID
        or split.get("split_id") != split_id
        or split.get("case_set_id") != declaration.get("case_set_id")
        or not isinstance(case_ids, list)
        or len(case_ids) != declaration.get("case_count")
        or len(set(case_ids)) != len(case_ids)
        or any(not isinstance(c, str) or CASE_ID_RE.fullmatch(c) is None for c in case_ids)
    ):
        raise ContractError("split index identity or case list is invalid")
    return split
