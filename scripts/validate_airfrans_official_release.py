#!/usr/bin/env python3
"""Check the approved AirfRANS draft and its active dev intake contract.

CI validates the committed hash graph and full-artifact validation receipt. Arrays
remain in prepared local archives; their complete load was performed at preparation,
and can be repeated with publish_airfrans_official_release.verify_native.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.airfrans.profile_truth import validate_release
from reference.hosting import support_url
from reference.scoring_support import sha256_file
from scripts.validate_scoring_supports import validate_specification


def check(root=ROOT):
    spec_dir = root / "benchmark-specs/airfrans"
    active = json.loads((spec_dir / "submission-spec.json").read_text())
    declaration = active.get("approved_release")
    if declaration is None:
        return None

    def require(value, message):
        if not value:
            raise ValueError(message)

    def bound(entry):
        path = (spec_dir / entry["file"]).resolve()
        require(path.is_relative_to(spec_dir.resolve()) and not path.is_symlink(), "release path escapes AirfRANS")
        require(path.is_file() and sha256_file(path) == entry["sha256"], f"release digest differs: {entry['file']}")
        if "byte_size" in entry:
            require(path.stat().st_size == entry["byte_size"], "release byte size differs")
        return path

    binding_path = bound({"file": declaration["binding_file"], "sha256": declaration["binding_sha256"]})
    binding = json.loads(binding_path.read_text())
    require(binding["schema"] == "airfrans-approved-release-binding-v1", "unknown AirfRANS binding")
    require(binding["status"] == "dev_approved_draft" and binding["submissions_opened"] is False and binding["publicly_published"] is False,
            "dev drafts cannot claim publication or open intake")
    require(binding["enabled_split_ids"] == ["full", "scarce", "aoa_extrapolation"], "official split scope differs")
    contract_path = bound(binding["contract"])
    contract = json.loads(contract_path.read_text())
    require(contract["status"] == "official" and contract["scoring_support"]["submissions_open"] is False,
            "approved contract must be official and closed")
    require(contract["publication_status"] == "prepared_dev_only", "contract must remain a dev draft")
    require([s["id"] for s in contract["splits"]] == binding["enabled_split_ids"], "contract split scope differs")
    require("approved_release" not in contract, "a release contract cannot recursively select another release")
    errors = validate_specification(contract_path, spec_root=contract_path.parent.parent)
    require(not errors, "\n".join(errors))
    approval = json.loads(bound(binding["approval"]).read_text())
    require(approval["status"] == "approved" and approval["contract_sha256"] == binding["contract"]["sha256"],
            "approval does not bind the contract")
    require(approval["owner_approval"] == contract["scoring_support"]["owner_approval"], "owner approval differs")
    require(approval["submissions_opened"] is False and approval["participant_results_approved"] is False,
            "release approval cannot approve participant results")
    require(approval["publicly_published"] is False, "approval cannot claim public publication")
    receipt = json.loads(bound(binding["publication_validation"]).read_text())
    require(receipt["status"] == "passed" and receipt["publicly_published"] is False, "draft validation did not pass")
    native = receipt["native_support"]
    require((native["case_count"], native["unique_case_count"], native["support_instance_count"]) == (396, 355, 1188),
            "native publication coverage differs")
    require(native["normalized_support_sha256"] == contract["scoring_support"]["publication_validation"]["normalized_support_sha256"],
            "normalized native support differs")
    require(receipt["native_table_bytes_changed"] is False and receipt["profile_payload_bytes_changed"] is False,
            "publication changed source payloads")
    truth_root = contract_path.parent / "profile-truth/airfrans-native-profile-truth-v1"
    truth = validate_release(release_root=truth_root, profile_definition_path=contract_path.parent / "velocity-profiles-v1.json",
                             repository_root=root, metadata_only=True)
    truth_binding = contract["profile_definition"]["profile_ground_truth"]
    require(truth["manifest_sha256"] == truth_binding["manifest_sha256"] == receipt["profile_truth_manifest_sha256"],
            "profile truth manifest differs")
    require(receipt["profile_truth"]["truth_arrays_checked"] is True and truth["case_count"] == 355,
            "full profile truth publication validation is missing")
    support_manifest = contract_path.parent / contract["scoring_support"]["manifest_file"]
    require(sha256_file(support_manifest) == receipt["scoring_support_manifest_sha256"] == approval["scoring_support_manifest_sha256"],
            "native support manifest differs")
    support_body = json.loads(support_manifest.read_text())
    archive_bindings = set()
    for case_set in support_body["case_sets"]:
        index_path = support_manifest.parent / case_set["index_file"]
        index = json.loads(index_path.read_text())
        for chunk in index["chunks"]:
            body = json.loads((index_path.parent / chunk["file"]).read_text())
            for case in body["cases"]:
                for instance in case["support_instances"]:
                    archive = instance["artifacts"][0]["archive"]
                    archive_bindings.add((archive["url"], archive["sha256"]))
    assets = {a["url"]: a for a in binding["assets"]}
    require(all(url in assets and assets[url]["sha256"] == digest for url, digest in archive_bindings),
            "native table archive differs from the published asset binding")
    require(all(a["url"].startswith("https://github.com/neilashton/fluidsbench-submission/releases/download/" + a["release_id"] + "/")
                and "?" not in a["url"] and "#" not in a["url"] for a in assets.values()), "asset URL must identify its versioned dev prerelease")
    if active.get("publication_status") in {"dev_intake", "official_intake"}:
        branch = active.get("intake_branch")
        require(branch in {"dev", "main"}, "AirfRANS intake branch must be dev or main")
        require(active["publication_status"] != "dev_intake" or branch == "dev",
                "legacy dev intake must target dev")
        expected = json.loads(json.dumps(contract))
        expected["scoring_support"]["manifest_url"] = support_url(expected["scoring_support"]["manifest_url"])
        expected["profile_definition"]["profile_ground_truth"]["manifest_url"] = support_url(expected["profile_definition"]["profile_ground_truth"]["manifest_url"])
        expected["publication_status"] = active["publication_status"]
        expected["intake_branch"] = branch
        expected["approved_release"] = declaration
        expected["scoring_support"]["submissions_open"] = True
        expected["scoring_support"].pop("closed_reason", None)
        prefix = "releases/airfrans-evaluation-v1/airfrans/"
        expected["scoring_support"]["manifest_file"] = prefix + expected["scoring_support"]["manifest_file"]
        expected["scoring_support"]["publication_validation"]["validator_file"] = prefix + "publication-validator.py"
        require(active == expected, "intake differs from the approved scoring contract")
    else:
        require(active["scoring_support"]["submissions_open"] is False, "unknown AirfRANS activation")
    return {"contract_id": binding["contract_id"], "enabled_split_ids": binding["enabled_split_ids"],
            "unique_cases": 355, "intake_open": active["scoring_support"]["submissions_open"], "publicly_published": False}


if __name__ == "__main__":
    try:
        print(json.dumps(check(), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
