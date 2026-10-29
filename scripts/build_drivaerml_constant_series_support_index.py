#!/usr/bin/env python3
"""Build the all-case DrivAerML constant-profile support identity index.

The input is the published native-v3 ground-truth directory from the website
repository.  Only compact identities and counts are retained here; prediction
arrays are never copied into the submission repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.coordinate_identity import (  # noqa: E402
    coordinate_array_identity_sha256,
)


MASTER_INDEX_SHA256 = "e7cf14f161fc7dbf22794e6f66db4e157329be960d2a4368140e08cf0608a5ae"
PROVENANCE_SHA256 = "6030df1dce11c4fbf6028d4e17ab39394cfdd3e7bb4d5a7ff81f33408849f221"
RELEASE_RECEIPT_SHA256 = "ef59ca838c828ac1c5505ef4d35f1b91a55e07ad90df359491599114ec0d878c"
DATASET_REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
NATIVE_SOURCE_PIN_SHA256 = "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
CONTRACT_ID = "drivaerml-relative-diagnostics-v3-candidate"
CONSTANT_FAMILIES = {
    "drivaerml-autocfd5-constant-v1",
    "drivaerml_cp_constant_v1",
}


class BuildError(ValueError):
    """Raised when the retained release cannot be reproduced exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildError(f"{path} must contain a JSON object")
    return value


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def safe_child(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BuildError("native-v3 chunk path must be a non-empty string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise BuildError(f"unsafe native-v3 chunk path: {relative!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise BuildError(f"native-v3 chunk escapes its release: {relative!r}")
    return resolved


def official_case_ids() -> list[str]:
    pin_path = ROOT / "benchmark-specs" / "drivaerml" / "proposal" / "native-source-pin.json"
    if sha256_file(pin_path) != NATIVE_SOURCE_PIN_SHA256:
        raise BuildError("official native-source pin SHA-256 changed")
    pin = load_json(pin_path)
    cases = pin.get("cases")
    if not isinstance(cases, list):
        raise BuildError("official native-source pin has no case list")
    result = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(result) != 484 or len(set(result)) != 484 or not all(
        isinstance(case_id, str) for case_id in result
    ):
        raise BuildError("official native-source pin case coverage is invalid")
    return result


def build(ground_truth_root: Path) -> dict[str, Any]:
    ground_truth_root = ground_truth_root.resolve()
    index_path = ground_truth_root / "index.json"
    provenance_path = ground_truth_root / "provenance.json"
    receipt_path = ground_truth_root / "release-receipt.json"
    expected_files = {
        index_path: MASTER_INDEX_SHA256,
        provenance_path: PROVENANCE_SHA256,
        receipt_path: RELEASE_RECEIPT_SHA256,
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise BuildError(f"published native-v3 artifact changed: {path}")

    index = load_json(index_path)
    case_ids = official_case_ids()
    if (
        index.get("schema") != "fluidsbench-drivaerml-native-profile-truth-index-v3"
        or index.get("schema_version") != "3.0"
        or index.get("dataset_id") != "drivaerml"
        or index.get("dataset_repository") != "neashton/drivaerml"
        or index.get("dataset_revision") != DATASET_REVISION
        or index.get("case_count") != 484
        or index.get("series_per_case") != 40
        or index.get("case_ids") != case_ids
        or index.get("profile_set", {}).get("drivaerml-autocfd5-constant-v1") != 16
        or index.get("profile_set", {}).get("drivaerml_cp_constant_v1") != 4
    ):
        raise BuildError("published native-v3 master index identity is invalid")

    retained_cases: list[dict[str, Any]] = []
    observed_case_ids: list[str] = []
    expected_series_keys: list[tuple[str, str]] | None = None
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 61:
        raise BuildError("published native-v3 release must contain exactly 61 chunks")
    for binding in chunks:
        if not isinstance(binding, dict):
            raise BuildError("native-v3 chunk binding must be an object")
        chunk_path = safe_child(ground_truth_root, binding.get("path"))
        if (
            not chunk_path.is_file()
            or chunk_path.stat().st_size != binding.get("size_bytes")
            or sha256_file(chunk_path) != binding.get("sha256")
        ):
            raise BuildError(f"native-v3 chunk binding changed: {chunk_path}")
        chunk = load_json(chunk_path)
        raw_cases = chunk.get("cases")
        if not isinstance(raw_cases, list):
            raise BuildError(f"{chunk_path} has no cases")
        if [case.get("case_id") for case in raw_cases] != binding.get("case_ids"):
            raise BuildError(f"{chunk_path} case order differs from its binding")
        for case in raw_cases:
            case_id = case.get("case_id")
            raw_series = case.get("series")
            if not isinstance(case_id, str) or not isinstance(raw_series, list):
                raise BuildError(f"{chunk_path} has an invalid case record")
            constant_series = [
                series
                for series in raw_series
                if isinstance(series, dict)
                and series.get("family_id") in CONSTANT_FAMILIES
            ]
            if len(constant_series) != 20:
                raise BuildError(f"{case_id} must contain exactly 20 constant series")
            series_keys = [
                (series.get("family_id"), series.get("station_id"))
                for series in constant_series
            ]
            if len(set(series_keys)) != 20:
                raise BuildError(f"{case_id} has duplicate constant series")
            if expected_series_keys is None:
                expected_series_keys = series_keys
            elif series_keys != expected_series_keys:
                raise BuildError(f"{case_id} constant series order differs")
            retained_series: list[dict[str, Any]] = []
            for series in constant_series:
                coordinate = series.get("coordinate")
                if (
                    series.get("representation") != "materialized"
                    or not isinstance(coordinate, list)
                    or len(coordinate) < 2
                ):
                    raise BuildError(f"{case_id} has an invalid constant series")
                coordinate_digest = coordinate_array_identity_sha256(coordinate)
                if coordinate_digest != series.get("coordinate_identity_sha256"):
                    raise BuildError(
                        f"{case_id}/{series.get('family_id')}/{series.get('station_id')} "
                        "coordinate identity changed"
                    )
                retained_series.append(
                    {
                        "family_id": series["family_id"],
                        "station_id": series["station_id"],
                        "representation": "materialized",
                        "support_identity_sha256": series["support_identity_sha256"],
                        "placement_receipt_identity_sha256": series[
                            "placement_receipt_identity_sha256"
                        ],
                        "coordinate_count": len(coordinate),
                        "coordinate_identity_sha256": coordinate_digest,
                    }
                )
            retained_cases.append({"case_id": case_id, "series": retained_series})
            observed_case_ids.append(case_id)
    if observed_case_ids != case_ids:
        raise BuildError("native-v3 chunk coverage differs from the official case order")

    return {
        "schema": "drivaerml-constant-series-support-index-v1",
        "schema_version": 1,
        "dataset_id": "drivaerml",
        "contract_id": CONTRACT_ID,
        "scope": "constant_families_all_official_cases",
        "case_count": 484,
        "series_per_case": 20,
        "source_bindings": {
            "native_profile_truth": {
                "master_index_path": (
                    "assets/data/profile-ground-truth/datasets/drivaerml/"
                    "native-v3/index.json"
                ),
                "master_index_sha256": MASTER_INDEX_SHA256,
                "provenance_sha256": PROVENANCE_SHA256,
                "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
                "dataset_repository": "neashton/drivaerml",
                "dataset_revision": DATASET_REVISION,
            },
            "public_dataset": {
                "native_source_pin_path": (
                    "benchmark-specs/drivaerml/proposal/native-source-pin.json"
                ),
                "native_source_pin_sha256": NATIVE_SOURCE_PIN_SHA256,
                "repository": "neashton/drivaerml",
                "revision": DATASET_REVISION,
            },
            "coordinate_identity_encoding": (
                "fluidsbench-drivaerml-coordinate-array-v1"
            ),
        },
        "cases": retained_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = canonical_json(build(args.ground_truth_root))
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != encoded:
            raise BuildError(f"generated index differs from {args.output}")
        print(f"verified {args.output} ({hashlib.sha256(encoded).hexdigest()})")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"wrote {args.output} ({hashlib.sha256(encoded).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
