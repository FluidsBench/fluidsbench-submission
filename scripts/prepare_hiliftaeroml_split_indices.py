#!/usr/bin/env python3
"""Freeze every HiLiftAeroML evaluation split from the Table-5 campaign.

The campaign contains eight distinct ordered evaluation case sets.  Several
benchmark labels differ only in training-data availability and therefore share
one evaluation case set.  This command expands those eight immutable sources
into all fourteen FluidsBench split files and updates only the corresponding
``submission-spec.json`` split descriptors.

The input is deliberately pinned by SHA-256.  It is internal release evidence,
not a participant-facing dependency and not a substitute for public source-file
content pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "benchmark-specs" / "hiliftaeroml"
CAMPAIGN_SCHEMA_ID = "hilift_native_table5_campaign_v1"
CAMPAIGN_SHA256 = "ba11fd7197270209b83890ec798528cf0a3e7afc78eaffa2fd123d517d73db98"
CASE_SET_SHA256_RULE = "sha256(utf8(case_id + newline) in listed order)"


class SplitPreparationError(ValueError):
    """Raised when the immutable source cannot produce the declared splits."""


@dataclass(frozen=True)
class SplitBinding:
    split_id: str
    split_label: str
    case_set_id: str
    case_count: int
    case_set_sha256: str


STANDARD_CASE_SET_ID = "caseset-ac791749e527"
STANDARD_CASE_SET_SHA256 = (
    "ac791749e5279ecf6746fcce20e3ec32408fd33b22127d5270de968be7842acf"
)
GEOMETRY_CASE_SET_ID = "caseset-53990ea68fa6"
GEOMETRY_CASE_SET_SHA256 = (
    "53990ea68fa6043879c9e080542ddeffe9836421dcf29d417c6ab47245c4c231"
)


SPLIT_BINDINGS: tuple[SplitBinding, ...] = (
    SplitBinding("full", "Full", STANDARD_CASE_SET_ID, 360, STANDARD_CASE_SET_SHA256),
    SplitBinding("medium", "Medium", STANDARD_CASE_SET_ID, 360, STANDARD_CASE_SET_SHA256),
    SplitBinding("scarce", "Scarce", STANDARD_CASE_SET_ID, 360, STANDARD_CASE_SET_SHA256),
    SplitBinding(
        "super_scarce",
        "Super scarce",
        STANDARD_CASE_SET_ID,
        360,
        STANDARD_CASE_SET_SHA256,
    ),
    SplitBinding(
        "geometry",
        "Geometry",
        GEOMETRY_CASE_SET_ID,
        360,
        GEOMETRY_CASE_SET_SHA256,
    ),
    SplitBinding(
        "geometry_medium",
        "Geometry medium",
        GEOMETRY_CASE_SET_ID,
        360,
        GEOMETRY_CASE_SET_SHA256,
    ),
    SplitBinding(
        "geometry_scarce",
        "Geometry scarce",
        GEOMETRY_CASE_SET_ID,
        360,
        GEOMETRY_CASE_SET_SHA256,
    ),
    SplitBinding(
        "geometry_super_scarce",
        "Geometry super scarce",
        GEOMETRY_CASE_SET_ID,
        360,
        GEOMETRY_CASE_SET_SHA256,
    ),
    SplitBinding(
        "single_aoa_4",
        "AoA 4",
        "caseset-7a743a20b3bd",
        36,
        "7a743a20b3bd4d4dbdb8b0316da982f0d3b38c6d38f479644c44d052d6539892",
    ),
    SplitBinding(
        "single_aoa_12",
        "AoA 12",
        "caseset-02fc12ff3494",
        36,
        "02fc12ff3494cc1d8d7c50cc1ee40bd8c1caaef85b7e97dbb8e5d0b88d28eb53",
    ),
    SplitBinding(
        "single_aoa_22",
        "AoA 22",
        "caseset-85ecccd9ccda",
        36,
        "85ecccd9ccda92ca6bb44eaffbb2de583ac81fc12edd24cad1f581d0ab13b5c4",
    ),
    SplitBinding(
        "aoa",
        "AoA extrapolation",
        "caseset-29693354ed8a",
        900,
        "29693354ed8ae4e08abb15efa54a818f78148e4a328dbdd88602ac505f004c44",
    ),
    SplitBinding(
        "deflection",
        "Deflection",
        "caseset-c0ecb14de138",
        360,
        "c0ecb14de138973cb20f64f6a7d1d6f9e4e6ea7442eba2620acdba39dd48d272",
    ),
    SplitBinding(
        "stall",
        "Stall",
        "caseset-804491c8956e",
        723,
        "804491c8956e7d4c1a3604ffe91d4ceb06bd03931c7d2b3bbcaf094c8d4e4f8e",
    ),
)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SplitPreparationError(f"cannot encode canonical JSON: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ordered_case_digest(case_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{case_id}\n" for case_id in case_ids).encode("utf-8")
    ).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SplitPreparationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SplitPreparationError(f"{path} must contain one JSON object")
    return value


def read_pinned_campaign(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SplitPreparationError(f"cannot read campaign {path}: {error}") from error
    observed_sha256 = sha256_bytes(payload)
    if observed_sha256 != CAMPAIGN_SHA256:
        raise SplitPreparationError(
            "Table-5 campaign SHA-256 differs: "
            f"expected {CAMPAIGN_SHA256}, observed {observed_sha256}"
        )
    try:
        campaign = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SplitPreparationError(f"cannot decode campaign {path}: {error}") from error
    if not isinstance(campaign, dict):
        raise SplitPreparationError("Table-5 campaign must contain one JSON object")
    if campaign.get("schema_id") != CAMPAIGN_SCHEMA_ID:
        raise SplitPreparationError("Table-5 campaign schema identity differs")
    return campaign


def case_sets_from_campaign(campaign: dict[str, Any]) -> dict[str, list[str]]:
    raw_case_sets = campaign.get("case_sets")
    if not isinstance(raw_case_sets, list):
        raise SplitPreparationError("Table-5 campaign case_sets must be a list")
    requested = {binding.case_set_id for binding in SPLIT_BINDINGS}
    matches: dict[str, dict[str, Any]] = {}
    for raw_case_set in raw_case_sets:
        if not isinstance(raw_case_set, dict):
            continue
        case_set_id = raw_case_set.get("case_set_id")
        if case_set_id not in requested:
            continue
        if case_set_id in matches:
            raise SplitPreparationError(f"duplicate campaign case set {case_set_id!r}")
        matches[case_set_id] = raw_case_set
    missing = sorted(requested - set(matches))
    if missing:
        raise SplitPreparationError(f"campaign case sets are missing: {missing}")

    expected_by_id = {
        binding.case_set_id: (binding.case_count, binding.case_set_sha256)
        for binding in SPLIT_BINDINGS
    }
    result: dict[str, list[str]] = {}
    for case_set_id, (expected_count, expected_sha256) in expected_by_id.items():
        entry = matches[case_set_id]
        case_ids = entry.get("ordered_case_ids")
        if (
            not isinstance(case_ids, list)
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
        ):
            raise SplitPreparationError(
                f"campaign case set {case_set_id!r} has malformed or duplicate IDs"
            )
        observed_sha256 = ordered_case_digest(case_ids)
        if len(case_ids) != expected_count or entry.get("case_count") != expected_count:
            raise SplitPreparationError(
                f"campaign case set {case_set_id!r} count differs: "
                f"expected {expected_count}, observed {len(case_ids)}"
            )
        if (
            observed_sha256 != expected_sha256
            or entry.get("case_set_sha256") != expected_sha256
        ):
            raise SplitPreparationError(
                f"campaign case set {case_set_id!r} digest differs: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        result[case_set_id] = list(case_ids)
    return result


def split_document(binding: SplitBinding, case_ids: Sequence[str]) -> dict[str, Any]:
    if len(case_ids) != binding.case_count:
        raise SplitPreparationError(f"{binding.split_id} case count differs")
    if ordered_case_digest(case_ids) != binding.case_set_sha256:
        raise SplitPreparationError(f"{binding.split_id} ordered case digest differs")
    return {
        "schema_version": "1.0",
        "dataset_id": "hiliftaeroml",
        "split_id": binding.split_id,
        "case_set_id": binding.case_set_id,
        "split_label": binding.split_label,
        "case_id_status": "official",
        "case_count": binding.case_count,
        "case_set_sha256": binding.case_set_sha256,
        "case_set_sha256_rule": CASE_SET_SHA256_RULE,
        "case_ids": list(case_ids),
    }


def build_split_documents(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    case_sets = case_sets_from_campaign(campaign)
    return {
        binding.split_id: split_document(binding, case_sets[binding.case_set_id])
        for binding in SPLIT_BINDINGS
    }


def update_specification_splits(
    specification: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_splits = specification.get("splits")
    if not isinstance(raw_splits, list):
        raise SplitPreparationError("submission specification splits must be a list")
    expected_ids = [binding.split_id for binding in SPLIT_BINDINGS]
    observed_ids = [
        split.get("id") if isinstance(split, dict) else None for split in raw_splits
    ]
    if observed_ids != expected_ids:
        raise SplitPreparationError(
            "submission specification split order differs: "
            f"expected {expected_ids}, observed {observed_ids}"
        )
    by_id = {binding.split_id: binding for binding in SPLIT_BINDINGS}
    for descriptor in raw_splits:
        split_id = descriptor["id"]
        binding = by_id[split_id]
        payload = canonical_json_bytes(documents[split_id])
        descriptor.update(
            {
                "label": binding.split_label,
                "index_file": f"splits/{split_id}.json",
                "case_count": binding.case_count,
                "case_set_id": binding.case_set_id,
                "case_id_status": "official",
                "sha256": sha256_bytes(payload),
            }
        )
    return specification


def prepare(
    campaign: dict[str, Any],
    dataset_dir: Path,
    *,
    check: bool,
) -> dict[str, Any]:
    documents = build_split_documents(campaign)
    specification_path = dataset_dir / "submission-spec.json"
    specification = update_specification_splits(
        read_json_object(specification_path), documents
    )
    expected_files = {
        dataset_dir / "splits" / f"{split_id}.json": canonical_json_bytes(document)
        for split_id, document in documents.items()
    }
    expected_files[specification_path] = canonical_json_bytes(specification)

    differences: list[str] = []
    for path, payload in expected_files.items():
        try:
            current = path.read_bytes()
        except OSError:
            current = None
        if current == payload:
            continue
        differences.append(str(path))
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    if check and differences:
        raise SplitPreparationError(
            "generated split state differs from the repository: "
            + ", ".join(differences)
        )

    return {
        "campaign_sha256": CAMPAIGN_SHA256,
        "case_set_count": len({binding.case_set_id for binding in SPLIT_BINDINGS}),
        "split_count": len(SPLIT_BINDINGS),
        "changed_files": differences,
        "splits": [
            {
                "split_id": binding.split_id,
                "case_count": binding.case_count,
                "case_set_id": binding.case_set_id,
                "case_set_sha256": binding.case_set_sha256,
                "split_file_sha256": sha256_bytes(
                    canonical_json_bytes(documents[binding.split_id])
                ),
            }
            for binding in SPLIT_BINDINGS
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-json",
        type=Path,
        required=True,
        help="Pinned hilift_native_table5_campaign_20260810.json",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="HiLiftAeroML benchmark-spec directory to update",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files differ; do not write anything",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = read_pinned_campaign(args.campaign_json)
    report = prepare(campaign, args.dataset_dir, check=args.check)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
