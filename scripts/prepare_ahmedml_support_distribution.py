#!/usr/bin/env python3
"""Package an existing verified AhmedML cache without regenerating its data."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import REPOSITORY_REVISION, sha256_file  # noqa: E402
from reference.ahmedml.distribution import (  # noqa: E402
    BASE_URL,
    BLOCK_BYTES,
    CONTRACT_SHA256,
    RELEASE_ID,
    RELEASE_SHA256,
    RELEASE_TAG,
    expected_files,
    select_cases,
    verify_case,
)


def package_case(source: Path, output: Path, case_id: str) -> tuple[str, dict]:
    files = expected_files(case_id)
    verify_case(source / case_id, files)
    archive = output / f"ahmedml-support-{case_id}.zip"
    temporary = archive.with_suffix(".zip.partial")
    if archive.exists():
        raise ValueError(f"refusing to overwrite {archive}")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as bundle:
            for name in sorted(files):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 1
                digest = hashlib.sha256()
                with (source / case_id / name).open("rb") as original, bundle.open(
                    info, "w"
                ) as sink:
                    while block := original.read(BLOCK_BYTES):
                        digest.update(block)
                        sink.write(block)
                if digest.hexdigest() != files[name]["sha256"]:
                    raise ValueError(
                        f"source changed during packaging: {case_id}/{name}"
                    )
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return case_id, {
        "url": BASE_URL + archive.name,
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
        "expanded_bytes": sum(f["size_bytes"] for f in files.values()),
        "case_manifest_sha256": files["case-support.json"]["sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = select_cases()
    # Load and validate canonical metadata once before workers start.
    expected_files(cases[0])
    entries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(package_case, args.cache_root, output, c) for c in cases]
        for future in concurrent.futures.as_completed(futures):
            case_id, entry = future.result()
            entries[case_id] = entry
            print(
                f"[{len(entries)}/{len(cases)}] {case_id}: {entry['size_bytes']} bytes",
                flush=True,
            )
    index = {
        "schema": "fluidsbench-ahmedml-support-distribution-v1",
        "schema_version": 1,
        "status": "dev_prerelease",
        "dataset_revision": REPOSITORY_REVISION,
        "scientific_release_id": RELEASE_ID,
        "scientific_manifest_sha256": RELEASE_SHA256,
        "derived_cache_contract_sha256": CONTRACT_SHA256,
        "release_tag": RELEASE_TAG,
        "cases": {c: entries[c] for c in cases},
    }
    path = output / "support-distribution-v1.json"
    path.write_text(json.dumps(index, indent=2) + "\n")
    checksums = [
        f"{entry['sha256']}  ahmedml-support-{case_id}.zip"
        for case_id, entry in index["cases"].items()
    ]
    checksums.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "archive_bytes": sum(e["size_bytes"] for e in entries.values()),
                "expanded_bytes": sum(e["expanded_bytes"] for e in entries.values()),
                "index_sha256": sha256_file(path),
            }
        )
    )


if __name__ == "__main__":
    main()
