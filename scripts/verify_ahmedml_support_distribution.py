#!/usr/bin/env python3
"""Qualify every public AhmedML archive on Linux using bounded local storage."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.ahmedml.distribution import (  # noqa: E402
    RELEASE_SHA256,
    ensure_case_support,
    load_distribution,
    select_cases,
)
from scripts.verify_ahmedml_derived_cache import verify_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error(
            "deep native-array verification requires Linux descriptor semantics"
        )
    index = load_distribution()
    receipts = []
    cases = select_cases()
    for position, case_id in enumerate(cases, 1):
        print(
            f"[{position}/{len(cases)}] Downloading and deeply verifying {case_id}",
            flush=True,
        )
        # Each temporary cache is genuinely empty. Do not use maintainer files
        # or restore an Actions cache. Keep the runner's disk requirement small.
        with tempfile.TemporaryDirectory(prefix="ahmedml-public-cache-") as temporary:
            root = Path(temporary)
            ensure_case_support(case_id, destination=root, index=index)
            receipt = verify_cache(cache_root=root, deep=True, case_ids=[case_id])
            receipts.append(receipt)
    report = {
        "status": "verified",
        "scientific_manifest_sha256": RELEASE_SHA256,
        "case_count": len(receipts),
        "artifact_count": sum(r["artifact_count"] for r in receipts),
        "payload_bytes": sum(r["payload_bytes"] for r in receipts),
        "cases": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Verified {report['case_count']} cases and {report['artifact_count']} payloads"
    )


if __name__ == "__main__":
    main()
