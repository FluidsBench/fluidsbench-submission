#!/usr/bin/env python3
"""Validate the complete hosted candidate support before a separate activation.

Reads cached archives without extracting or changing them. The receipt is evidence
for the candidate manifest only; it is never installed into the active spec.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "reference").is_dir() and (p / "schemas").is_dir())
sys.path.insert(0, str(ROOT))

from reference.drivaernetplusplus import contract, distribution
from reference.drivaernetplusplus.native import PROFILE_STATIONS
from reference.drivaernetplusplus.support import EXPECTED_ARRAYS, station_array_names
from reference.scoring_support import load_support_release
from scripts.validate_scoring_supports import validate_candidate_manifest_release

PREPARATION = contract.SPEC_ROOT / "releases/drivaernetplusplus-v1-preparation"
RECEIPT_SCHEMA = "fluidsbench-drivaernetplusplus-release-preparation-v1"
APPROVAL_PR = "https://github.com/neilashton/fluidsbench-submission/pull/61"
INPUTS = [
    Path(__file__),
    ROOT / "scripts/validate_scoring_supports.py",
    ROOT / "reference/scoring_support.py",
    *(ROOT / "reference/drivaernetplusplus" / name for name in
      ("contract.py", "distribution.py", "native.py", "support.py")),
    contract.SPEC_PATH, contract.DISTRIBUTION_PATH, contract.SOURCE_IDENTITY_PATH,
    contract.RELEASE_DIR / "manifest.json", contract.SPEC_ROOT / "splits/official_test.json",
    contract.SPEC_ROOT / "OWNER_APPROVAL.md", PREPARATION / "approval-record.json",
    PREPARATION / "hosted-assets.json",
]


def require(condition, message):
    if not condition:
        raise contract.ContractError(message)


def input_hashes():
    return {p.relative_to(ROOT).as_posix(): contract.sha256_file(p) for p in INPUTS}


def check_hosted_assets(hosted, index):
    require(hosted["tag_name"] == contract.RELEASE_TAG and not hosted["draft"]
            and hosted["prerelease"], "hosted release must remain a published dev prerelease")
    assets = {a["name"]: a for a in hosted["assets"]}
    require(len(assets) == len(hosted["assets"]), "duplicate hosted asset names")
    for entry in index["archives"]:
        asset = assets.get(entry["file"], {})
        require(asset.get("size") == entry["size_bytes"]
                and asset.get("digest") == "sha256:" + entry["sha256"]
                and asset.get("browser_download_url") == entry["url"],
                f"hosted archive binding differs: {entry['file']}")


def check_contract():
    spec = contract.load_specification()
    support = spec["scoring_support"]
    require(spec["status"] == "candidate_native_support"
            and support["status"] == "owner_review_required"
            and support["submissions_open"] is False
            and "publication_validation" not in support
            and "owner_approval" not in support, "active contract must stay closed and candidate")
    manifest_path = contract.RELEASE_DIR / "manifest.json"
    errors = validate_candidate_manifest_release(
        spec, contract.SPEC_ROOT, manifest_path, contract.read_json(manifest_path))
    require(not errors, "candidate manifest validation failed: " + "; ".join(errors))
    ids = contract.official_split()["case_ids"]
    index = distribution.load_distribution()
    require([c for entry in index["archives"] for c in entry["case_ids"]] == ids,
            "archive coverage must equal the ordered official test split")
    source = contract.read_json(contract.SOURCE_IDENTITY_PATH)
    require([c["case_id"] for c in source["cases"]] == ids, "source identity coverage differs")
    approval = contract.read_json(PREPARATION / "approval-record.json")
    require(approval["status"] == "prepared_dev_only"
            and approval["candidate_manifest_sha256"] == distribution.manifest_sha256()
            and approval["candidate_release_id"] == contract.RELEASE_ID
            and approval["profile_station_ids"] == list(PROFILE_STATIONS)
            and approval["official_test_case_count"] == len(ids)
            and approval["pressure_extremes_scored_as_stored"] is True
            and approval["official_submissions_open"] is False,
            "prepared owner approval differs from the closed candidate")
    check_hosted_assets(contract.read_json(PREPARATION / "hosted-assets.json"), index)
    return ids, index, {c["case_id"]: c for c in source["cases"]}


def check_receipt(receipt):
    ids, index, _ = check_contract()
    require(receipt["schema"] == RECEIPT_SCHEMA and receipt["status"] == "passed"
            and receipt["publication_status"] == "prepared_dev_only"
            and receipt["official_submissions_open"] is False,
            "receipt must be passed dev-only preparation evidence")
    require(receipt["input_sha256"] == input_hashes(), "receipt inputs changed; rerun full validation")
    require(receipt["manifest_sha256"] == distribution.manifest_sha256()
            and receipt["case_count"] == receipt["support_instance_count"] == len(ids)
            and receipt["archive_count"] == len(index["archives"]), "receipt coverage differs")
    require(contract.SHA256_RE.fullmatch(receipt["normalized_support_sha256"]) is not None,
            "normalized support digest is invalid")


def validate(support_root):
    ids, index, source = check_contract()
    release = load_support_release(contract.RELEASE_DIR / "manifest.json", contract.CASE_SET_ID)
    for entry in index["archives"]:
        distribution.ensure_archive(entry, destination=support_root, offline=True)
    digest = hashlib.sha256()
    point_count = series_count = extreme_points = extreme_cases = 0
    for number, case_id in enumerate(ids, 1):
        support = distribution.load_case_support(case_id, destination=support_root)
        record = support.record
        instance = release.cases[case_id]["support_instances"][0]
        require(instance["expected_support_sha256"] == support.record_sha256
                and instance["entity_count"] == support.point_count == source[case_id]["point_count"]
                and record["source"]["sha256"] == source[case_id]["sha256"]
                and record["source"]["size_bytes"] == source[case_id]["size_bytes"],
                f"{case_id}: source/manifest binding differs")
        require(np.all(np.isfinite(support.arrays["dual_area_m2"])), f"{case_id}: nonfinite weights")
        # Native point IDs are exactly range(point_count), in published VTK order.
        digest.update(contract.json_bytes([case_id, support.point_count, "source_entity_index"]))
        for name in EXPECTED_ARRAYS:
            array = support.arrays[name]
            digest.update(contract.json_bytes([name, array.dtype.str, list(array.shape)]))
            digest.update(array.tobytes(order="C"))
        for station in PROFILE_STATIONS:
            point_ids, coordinates = (support.arrays[n] for n in station_array_names(station))
            require(np.all(np.isfinite(coordinates)) and len(set(point_ids.tolist())) == len(point_ids)
                    and len(point_ids) == record["profile_stations"][station]["point_count"],
                    f"{case_id}: invalid native profile {station}")
            series_count += 1
        truth = support.arrays["p"]
        extremes = int(np.count_nonzero((truth > 460) | (truth < -4500)))
        extreme_points += extremes
        extreme_cases += int(extremes > 0)
        point_count += support.point_count
        if number % 100 == 0 or number == len(ids):
            print(f"Validated {number}/{len(ids)} native cases", flush=True)
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "passed", "publication_status": "prepared_dev_only",
        "validated_by": "Codex automated artifact validation",
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pull_request_url": APPROVAL_PR,
        "validator_file": Path(__file__).relative_to(contract.SPEC_ROOT).as_posix(),
        "validator_sha256": contract.sha256_file(Path(__file__)),
        "release_id": contract.RELEASE_ID, "manifest_sha256": distribution.manifest_sha256(),
        "case_count": len(ids), "support_instance_count": len(ids), "native_point_count": point_count,
        "archive_count": len(index["archives"]), "archive_bytes": sum(e["size_bytes"] for e in index["archives"]),
        "normalized_support_sha256": digest.hexdigest(), "profile_series_count": series_count,
        "pressure_extremes": {"points": extreme_points, "cases": extreme_cases, "masked": False},
        "official_submissions_open": False, "support_payload_bytes_changed": False,
        "source_mesh_replay_performed": False, "model_inference_performed": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.system()},
        "input_sha256": input_hashes(),
    }
    check_receipt(receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--output", type=Path, default=PREPARATION / "publication-validation.json")
    parser.add_argument("--check-receipt", action="store_true",
                        help="check retained evidence bindings; does not reread support archives")
    args = parser.parse_args()
    if args.check_receipt:
        check_receipt(contract.read_json(args.output))
        print("Prepared publication receipt matches the closed candidate contract and retained evidence.")
    else:
        if args.support_root is None:
            parser.error("--support-root is required for full archive validation")
        contract.write_json(args.output, validate(args.support_root))
        print(f"Written full validation receipt: {args.output}")


if __name__ == "__main__":
    main()
