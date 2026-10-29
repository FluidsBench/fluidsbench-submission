#!/usr/bin/env python3
"""Validate native support metadata and local artifacts for a release receipt.

Remote CFD and prepared-cache payloads are hash-pinned by the support chain;
this validator checks their declarations but does not download them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.scoring_support import load_support_release, sha256_file
from scripts.validate_scoring_supports import validate_candidate_manifest_release


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def validate(dataset: str, release_id: str) -> dict:
    dataset_root = ROOT / "benchmark-specs" / dataset
    manifest_path = dataset_root / "scoring-support" / release_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    specification = json.loads((dataset_root / "submission-spec.json").read_text())
    if manifest["status"] != "official" or manifest["release_id"] != release_id:
        raise ValueError("support manifest is not the selected official release")
    if specification["dataset_id"] != dataset:
        raise ValueError("dataset specification differs")
    # The candidate gate validates exact split membership, metric definitions,
    # and chained release artifacts. Use its semantic checks with the already
    # approved scientific definitions and official manifest status adjusted in
    # memory only; retained official release bytes are never changed.
    candidate_view = dict(manifest, status="candidate")
    candidate_view.pop("owner_approval", None)
    errors = validate_candidate_manifest_release(
        specification, dataset_root, manifest_path, candidate_view
    )
    if errors:
        raise ValueError("; ".join(errors))
    digest = hashlib.sha256()
    cases = instances = local_artifacts = remote_artifacts = 0
    for descriptor in manifest["case_sets"]:
        case_set = descriptor["id"]
        release = load_support_release(manifest_path, case_set)
        if len(release.cases) != descriptor["case_count"]:
            raise ValueError(f"{case_set}: case count differs")
        for case_id, case in release.cases.items():
            cases += 1
            for instance in case["support_instances"]:
                instances += 1
                for artifact in instance["artifacts"]:
                    expected = artifact.get("sha256")
                    if not isinstance(expected, str) or len(expected) != 64:
                        raise ValueError(f"{case_id}: artifact has no SHA-256")
                    if "path" in artifact:
                        path = (release.case_directories[case_id] / artifact["path"]).resolve()
                        if not path.is_relative_to(manifest_path.parent.resolve()):
                            raise ValueError(f"{case_id}: artifact escapes release root")
                        if sha256_file(path) != expected:
                            raise ValueError(f"{case_id}: local artifact hash differs")
                        local_artifacts += 1
                    elif "url" in artifact or "archive" in artifact:
                        remote = artifact.get("archive", artifact)
                        url = urlsplit(remote["url"])
                        if url.scheme != "https" or not url.netloc or url.username or url.password:
                            raise ValueError(f"{case_id}: remote artifact URL is invalid")
                        remote_artifacts += 1
                    else:
                        raise ValueError(f"{case_id}: artifact has neither path nor URL")
                digest.update(canonical([case_set, case_id, instance]))
    return {
        "case_count": cases,
        "support_instance_count": instances,
        "local_artifact_count": local_artifacts,
        "remote_artifact_count": remote_artifacts,
        "normalized_support_sha256": digest.hexdigest(),
        "manifest_sha256": sha256_file(manifest_path),
        "validation_scope": "exact_case_metric_and_hash_chain_local_artifact_bytes_remote_hash_pins_not_downloaded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("ahmedml", "drivaerml", "windsorml", "drivaernetplusplus"))
    parser.add_argument("release_id")
    parser.add_argument("--validated-by", required=True)
    parser.add_argument("--pull-request-url", required=True)
    parser.add_argument("--validator-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.dataset, args.release_id)
    validator_path = ROOT / "benchmark-specs" / args.dataset / args.validator_file
    result.update(
        status="passed",
        validated_by=args.validated_by,
        validated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        validator_file=args.validator_file,
        validator_sha256=sha256_file(validator_path),
        pull_request_url=args.pull_request_url,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
