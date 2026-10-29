#!/usr/bin/env python3
"""Download the exact profile manifest pinned by this submission checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def fetch_manifest(output: Path, *, release_manifest: Path | None = None) -> Path:
    release_manifest = release_manifest or ROOT / "leaderboard" / "manifest.json"
    binding = json.loads(release_manifest.read_text())["data_release"]["profile_ground_truth"]
    url = binding["manifest_url"]
    if urlsplit(url).scheme != "https":
        raise ValueError("profile-ground-truth manifest URL must use HTTPS")
    with urlopen(url, timeout=30) as response:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("profile-ground-truth download redirected outside HTTPS")
        payload = response.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("profile-ground-truth manifest exceeds the download size limit")
    if hashlib.sha256(payload).hexdigest() != binding["manifest_sha256"]:
        raise ValueError("downloaded profile-ground-truth manifest SHA-256 differs from the repository binding")
    if json.loads(payload).get("data_release", {}).get("id") != binding["release_id"]:
        raise ValueError("downloaded profile-ground-truth release ID differs from the repository binding")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact bytes; never leave a partial or unverified output file.
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.close()
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = fetch_manifest(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Profile-ground-truth download failed: {error}\n")
    print(f"Verified profile-ground-truth manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
