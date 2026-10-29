#!/usr/bin/env python3
"""Prepare approved AirfRANS evaluation drafts on dev without publishing.

Only release metadata changes. Native tables and lossless extraction JSON retain
their candidate bytes. Archives have deterministic headers; no network upload
or participant approval is performed by this command.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "reference").is_dir() and (p / "schemas").is_dir())
sys.path.insert(0, str(ROOT))

from reference.airfrans.profile_truth import validate_release as validate_truth
from reference.scoring_support import load_support_release, load_scoring_support, sha256_file
from scripts.validate_scoring_supports import validate_specification

SUPPORT_ID = "airfrans-native-support-v1"
TRUTH_ID = "airfrans-native-profile-truth-v1"
CONTRACT_ID = "airfrans-evaluation-v1"
EVALUATOR_VERSION = "airfrans-scoring-v2"
DATASET_VERSION = "airfrans-native-v1"
EVALUATOR_COMMIT = "b320575e35d6f61f94ac852fdb084dec080e4ecb"
ENABLED_SPLITS = ["full", "scarce", "aoa_extrapolation"]
RELEASE_BASE = "https://github.com/neilashton/fluidsbench-submission/releases/download/"
SPEC = ROOT / "benchmark-specs/airfrans"
RELEASE_DIR = SPEC / "releases" / CONTRACT_ID


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return sha256_file(path)


def descriptor(path, root):
    return {"file": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
            "byte_size": path.stat().st_size}


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError(f"copy differs: {source}")


def make_tar(path, files, *, compressed=False):
    """Archive only explicitly selected regular files with fixed headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        stream = gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) if compressed else raw
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for member, source in sorted(files.items()):
                if source.is_symlink() or not source.is_file() or Path(member).is_absolute() or ".." in Path(member).parts:
                    raise ValueError(f"unsafe archive file: {member}")
                info = tarfile.TarInfo(member)
                info.size = source.stat().st_size
                info.mode = 0o644
                info.mtime = info.uid = info.gid = 0
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        if compressed:
            stream.close()
    return sha256_file(path)


def verify_native(release_root):
    """Load every case/support and digest normalized IDs, coordinates, weights and targets."""
    digest = hashlib.sha256()
    manifest = read(release_root / "manifest.json")
    count = instances = 0
    unique = set()
    for case_set in manifest["case_sets"]:
        release = load_support_release(release_root / "manifest.json", case_set["id"])
        for case_id in release.cases:
            count += 1
            unique.add(case_id)
            for support_id in sorted(release.supports):
                support = load_scoring_support(release, case_id, support_id)
                digest.update(json.dumps([case_set["id"], case_id, support_id]).encode())
                arrays = [("ids", support.support_ids), ("coordinates", support.coordinates),
                          ("weights", support.weights), *sorted(support.targets.items())]
                for name, array in arrays:
                    digest.update(json.dumps([name, str(array.dtype), array.shape]).encode())
                    digest.update(array.tobytes(order="C"))
                instances += 1
    return {"case_count": count, "unique_case_count": len(unique),
            "support_instance_count": instances, "normalized_support_sha256": digest.hexdigest()}


def native_release(full, aoa, output, approval, timestamp):
    root = output / SUPPORT_ID
    manifests = [read(full / "manifest.json"), read(aoa / "manifest.json")]
    if manifests[0]["supports"] != manifests[1]["supports"]:
        raise ValueError("candidate support definitions differ")
    tables = {}
    prepared = []
    for source, manifest in zip((full, aoa), manifests):
        case_set = manifest["case_sets"][0]
        original = load_support_release(source / "manifest.json", case_set["id"])
        chunk = read(source / "case-sets" / case_set["id"] / "chunk-000.json")
        chunk["release_id"] = SUPPORT_ID
        for case in chunk["cases"]:
            for instance in case["support_instances"]:
                for artifact in instance["artifacts"]:
                    source_path = original.case_directories[case["case_id"]] / artifact.pop("path")
                    if sha256_file(source_path) != artifact["sha256"]:
                        raise ValueError(f"candidate table differs: {source_path}")
                    member = "tables/" + source_path.name
                    if member in tables and sha256_file(tables[member]) != artifact["sha256"]:
                        raise ValueError(f"overlapping case table differs: {member}")
                    tables[member] = source_path
                    artifact["archive"] = {"url": RELEASE_BASE + SUPPORT_ID + "/native-tables.tar",
                                           "sha256": "pending", "member": member}
                    artifact["byte_size"] = source_path.stat().st_size
        prepared.append((case_set, chunk))
    archive_hash = make_tar(output / "support-assets/native-tables.tar", tables)
    for member, source in tables.items():
        copy_exact(source, root / member)
    descriptors = []
    for case_set, chunk in prepared:
        for case in chunk["cases"]:
            for instance in case["support_instances"]:
                for artifact in instance["artifacts"]:
                    artifact["archive"]["sha256"] = archive_hash
        directory = root / "case-sets" / case_set["id"]
        chunk_hash = write(directory / "chunk-000.json", chunk)
        index = read((full if case_set["id"] == "standard" else aoa) / case_set["index_file"])
        index["release_id"] = SUPPORT_ID
        index["chunks"][0]["sha256"] = chunk_hash
        index_hash = write(directory / "index.json", index)
        descriptors.append({**case_set, "index_sha256": index_hash})
    manifest = copy.deepcopy(manifests[0])
    manifest.update(release_id=SUPPORT_ID, status="official", published_at=timestamp,
                    dataset_version=DATASET_VERSION, evaluation_reference_version=EVALUATOR_VERSION,
                    owner_approval=approval, case_sets=descriptors,
                    notes="Approved dev-only draft support for Full, Scarce and AoA. Public publication and intake remain pending. Native table bytes are unchanged from the verified public-CFD candidate handover; download URLs are reserved destinations, not published assets.")
    write(root / "manifest.json", manifest)
    return root, verify_native(root)


def promote_truth(candidate, output, support_root, approval, timestamp):
    validate_truth(release_root=candidate, profile_definition_path=SPEC / "velocity-profiles-v1.json", repository_root=ROOT)
    root = output / TRUTH_ID
    candidate_manifest = read(candidate / "manifest.json")
    index = read(candidate / "index.json")
    universe = index["case_ids"]

    def official(body):
        body = copy.deepcopy(body)
        body["release_id"] = TRUTH_ID
        if "schema" in body:
            body["schema"] = body["schema"].removesuffix("-candidate")
        if "format" in body:
            body["format"] = "fluidsbench-airfrans-native-profile-truth-v1"
        if "status" in body:
            body["status"] = "official"
        return body

    for case_id in universe:
        copy_exact(candidate / f"cases/{case_id}.json", root / f"cases/{case_id}.json")
        location = index["case_locations"][case_id]
        record = official(read(candidate / location["case_record_file"]))
        location["case_record_sha256"] = write(root / location["case_record_file"], record)
    for chunk in index["chunks"]:
        body = official(read(candidate / chunk["file"]))
        for case in body["cases"]:
            case["case_record_sha256"] = index["case_locations"][case["case_id"]]["case_record_sha256"]
        write(root / chunk["file"], body)
        chunk.update(descriptor(root / chunk["file"], root))
        for case_id in chunk["case_ids"]:
            index["case_locations"][case_id]["chunk_sha256"] = chunk["sha256"]
    index = official(index)
    write(root / "index.json", index)
    case_sets = []
    support_manifest = read(support_root / "manifest.json")
    for entry in candidate_manifest["case_sets"]:
        body = official(read(candidate / entry["file"]))
        body["master_index_sha256"] = sha256_file(root / "index.json")
        support_set = next(s for s in support_manifest["case_sets"] if s["id"] == entry["case_set_id"])
        body["scoring_support"] = {"release_id": SUPPORT_ID, "case_set_index_sha256": support_set["index_sha256"]}
        write(root / entry["file"], body)
        case_sets.append({**entry, **descriptor(root / entry["file"], root)})
    provenance = official(read(candidate / "provenance.json"))
    provenance["promotion"] = {"candidate_release_id": candidate_manifest["release_id"],
                               "candidate_manifest_sha256": sha256_file(candidate / "manifest.json"),
                               "truth_payload_bytes_changed": False, "source_replay_performed": False}
    write(root / "provenance.json", provenance)
    manifest = official(candidate_manifest)
    manifest.update(owner_approval=approval, prepared_at=timestamp, publication_status="prepared_dev_only", enabled_split_ids=ENABLED_SPLITS,
                    master_index=descriptor(root / "index.json", root),
                    provenance=descriptor(root / "provenance.json", root), case_sets=case_sets,
                    activation={"owner_approval_complete": True, "published": False, "submissions_opened": False})
    manifest_hash = write(root / "manifest.json", manifest)
    receipt = official(read(candidate / "release-receipt.json"))
    receipt.update(manifest_sha256=manifest_hash, status="official",
                   owner_approval_complete=True, published=False, submissions_opened=False)
    write(root / "release-receipt.json", receipt)
    summary = validate_truth(release_root=root, profile_definition_path=SPEC / "velocity-profiles-v1.json", repository_root=ROOT)
    make_tar(output / "truth-assets/profile-truth.tar.gz",
             {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}, compressed=True)
    return root, summary


def build(args):
    if not re.fullmatch(r"https://github\.com/neilashton/fluidsbench-submission/pull/[1-9][0-9]*", args.approval_pr):
        raise ValueError("approval must identify an actual submission-repository PR")
    approval = {"approved_by": args.approved_by, "approved_at": args.approved_at,
                "pull_request_url": args.approval_pr}
    output = args.output.resolve()
    native, normalized = native_release(args.full_support, args.aoa_support, output, approval, args.published_at)
    truth, truth_summary = promote_truth(args.profile_truth, output, native, approval, args.published_at)
    destination = RELEASE_DIR / "airfrans"
    support_dest = destination / "scoring-support" / SUPPORT_ID
    truth_dest = destination / "profile-truth" / TRUTH_ID
    for source, target, omitted in ((native, support_dest, "tables"), (truth, truth_dest, "cases")):
        for path in source.rglob("*"):
            if path.is_file() and path.relative_to(source).parts[0] != omitted:
                copy_exact(path, target / path.relative_to(source))
    for filename in ("methodology-contract.json", "velocity-profiles-v1.json"):
        copy_exact(SPEC / filename, destination / filename)
    contract = read(SPEC / "submission-spec.json")
    contract.pop("approved_release", None)
    contract.update(status="official", publication_status="prepared_dev_only", dataset_version=DATASET_VERSION, evaluation_reference_version=EVALUATOR_VERSION)
    contract["splits"] = [s for s in contract["splits"] if s["id"] in ENABLED_SPLITS]
    for split in contract["splits"]:
        copy_exact(SPEC / split["index_file"], destination / split["index_file"])
    copy_exact(Path(__file__), destination / "publication-validator.py")
    support = contract["scoring_support"]
    support.pop("owner_decisions_required", None)
    support.update(status="official", submissions_open=False,
                   closed_reason="Approved dev-only draft. Public publication and the contributor/approval trial remain pending. Reynolds is outside this release.",
                   release_id=SUPPORT_ID, manifest_file=f"scoring-support/{SUPPORT_ID}/manifest.json",
                   manifest_url=RELEASE_BASE + SUPPORT_ID + "/manifest.json",
                   manifest_sha256=sha256_file(native / "manifest.json"), owner_approval=approval,
                   publication_validation={"status": "passed", "validated_by": "Codex automated artifact validation",
                       "validated_at": args.published_at, "validator_file": "publication-validator.py",
                       "validator_sha256": sha256_file(Path(__file__)), **normalized,
                       "manifest_sha256": sha256_file(native / "manifest.json"), "pull_request_url": args.approval_pr})
    contract["profile_definition"].update(status="official", profile_ground_truth={
        "status": "prepared_dev_only", "release_id": TRUTH_ID, "manifest_sha256": sha256_file(truth / "manifest.json"),
        "manifest_url": RELEASE_BASE + TRUTH_ID + "/manifest.json"})
    write(destination / "submission-spec.json", contract)
    errors = validate_specification(destination / "submission-spec.json", spec_root=RELEASE_DIR)
    if errors:
        raise ValueError("\n".join(errors))
    approval_record = {"schema": "airfrans-official-scoring-approval-v1", "status": "approved",
        "dataset_id": "airfrans", "contract_id": CONTRACT_ID, "owner_approval": approval,
        "approval_basis": "Neil Ashton explicitly requested approval and official publication on 2026-10-07.",
        "enabled_split_ids": ENABLED_SPLITS, "excluded_split_ids": ["reynolds_extrapolation"],
        "evaluator": {"version": EVALUATOR_VERSION, "repository": "https://github.com/neilashton/fluidsbench-submission",
                      "scientific_implementation_commit": EVALUATOR_COMMIT},
        "publication_status": "prepared_dev_only", "publicly_published": False,
        "contract_sha256": sha256_file(destination / "submission-spec.json"),
        "scoring_support_manifest_sha256": sha256_file(native / "manifest.json"),
        "profile_truth_manifest_sha256": sha256_file(truth / "manifest.json"),
        "submissions_opened": False, "participant_results_approved": False,
        "source_reextraction_performed": False, "model_inference_performed": False}
    write(RELEASE_DIR / "approval.json", approval_record)
    receipt = {"schema": "airfrans-official-publication-validation-v1", "status": "passed", "publicly_published": False,
        "validated_at": args.published_at, "native_support": normalized, "profile_truth": truth_summary,
        "native_table_bytes_changed": False, "profile_payload_bytes_changed": False,
        "source_reextraction_performed": False, "model_inference_performed": False,
        "scoring_support_manifest_sha256": sha256_file(native / "manifest.json"),
        "profile_truth_manifest_sha256": sha256_file(truth / "manifest.json"),
        "input_manifests": {str(p.name): sha256_file(p / "manifest.json")
                            for p in (args.full_support, args.aoa_support, args.profile_truth)}}
    write(RELEASE_DIR / "publication-validation.json", receipt)
    metadata_files = {p.relative_to(native).as_posix(): p for p in native.rglob("*")
                      if p.is_file() and p.relative_to(native).parts[0] != "tables"}
    make_tar(output / "support-assets/support-metadata.tar.gz", metadata_files, compressed=True)
    for category, source in (("support", native), ("truth", truth)):
        assets = output / (category + "-assets")
        for name, path in (("manifest.json", source / "manifest.json"),
                           ("approval.json", RELEASE_DIR / "approval.json"),
                           ("publication-validation.json", RELEASE_DIR / "publication-validation.json")):
            copy_exact(path, assets / name)
        if category == "support":
            for schema in ("manifest", "case-index", "case-chunk"):
                copy_exact(ROOT / "schemas/scoring-support/v1" / (schema + ".schema.json"),
                           assets / (schema + ".schema.json"))
        archive_files = sorted(p for p in assets.iterdir() if p.is_file() and p.name != "SHA256SUMS")
        (assets / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in archive_files))
    binding = {"schema": "airfrans-approved-release-binding-v1", "status": "dev_approved_draft", "publicly_published": False,
        "contract_id": CONTRACT_ID, "enabled_split_ids": ENABLED_SPLITS, "submissions_opened": False,
        "contract": descriptor(destination / "submission-spec.json", SPEC),
        "approval": descriptor(RELEASE_DIR / "approval.json", SPEC),
        "publication_validation": descriptor(RELEASE_DIR / "publication-validation.json", SPEC),
        "assets": [{"release_id": SUPPORT_ID if category == "support" else TRUTH_ID,
                    "url": RELEASE_BASE + (SUPPORT_ID if category == "support" else TRUTH_ID) + "/" + p.name,
                    "sha256": sha256_file(p), "byte_size": p.stat().st_size}
                   for category in ("support", "truth")
                   for p in sorted((output / (category + "-assets")).iterdir()) if p.is_file()]}
    write(SPEC / "official-release-binding.json", binding)
    active = read(SPEC / "submission-spec.json")
    active["approved_release"] = {"binding_file": "official-release-binding.json",
                                  "binding_sha256": sha256_file(SPEC / "official-release-binding.json")}
    write(SPEC / "submission-spec.json", active)
    return {"output": str(output), "native_support": normalized, "profile_truth": truth_summary}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-native":
        print(json.dumps(verify_native(Path(sys.argv[2])), indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-support", type=Path, required=True)
    parser.add_argument("--aoa-support", type=Path, required=True)
    parser.add_argument("--profile-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--approval-pr", required=True)
    parser.add_argument("--published-at", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
