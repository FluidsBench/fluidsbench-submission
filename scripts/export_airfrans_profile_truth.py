#!/usr/bin/env python3
"""Package AirfRANS ground-truth velocity profiles into a candidate profile-truth release.

The inputs are the ``ground_truth_reference`` JSON files written by
``examples/airfrans-profile-extraction/extract.py`` for each official split,
i.e. the files passed to ``score.py --ground-truth``. They are copied byte for
byte into the HiLiftAeroML-style release layout described in
``reference/airfrans/profile_truth.py``; cases shared by several splits are
stored once. Packaging is deterministic: rerun with ``--check`` to compare
every byte of an existing release without writing anything.

Example::

    python3 scripts/export_airfrans_profile_truth.py \\
      --source full=/path/to/full/reference-profiles \\
      --source aoa_extrapolation=/path/to/aoa/reference-profiles \\
      --scoring-support-manifest full=/path/to/airfrans-native-ground-truth-v1-candidate/manifest.json \\
      --scoring-support-manifest aoa_extrapolation=/path/to/airfrans-native-ground-truth-aoa-v1-candidate/manifest.json \\
      --extractor-commit 5f69ea140b061ec39b6827686a51c88d3ad657cd \\
      --output /path/to/airfrans-native-profile-truth-v1-candidate

Packaging does not publish the release, record owner approval or open
submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.airfrans.profile_truth import (  # noqa: E402
    EXTRACTOR_PATH,
    CaseSetSource,
    ProfileTruthError,
    build_release,
)

DEFAULT_PROFILE_DEFINITION = ROOT / "benchmark-specs" / "airfrans" / "velocity-profiles-v1.json"
SPLITS_ROOT = ROOT / "benchmark-specs" / "airfrans" / "splits"


def _pairs(values: list[str], option: str) -> dict[str, Path]:
    pairs: dict[str, Path] = {}
    for value in values:
        split_id, separator, path = value.partition("=")
        if not separator or not split_id or not path:
            raise SystemExit(f"{option} expects SPLIT_ID=PATH, got {value!r}")
        if split_id in pairs:
            raise SystemExit(f"{option} names split {split_id!r} twice")
        pairs[split_id] = Path(path)
    return pairs


def extractor_file_sha256(commit: str) -> str:
    """SHA-256 of ``extract.py`` as committed at ``commit``."""

    try:
        payload = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{EXTRACTOR_PATH}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot read {EXTRACTOR_PATH} at commit {commit}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", action="append", required=True, metavar="SPLIT_ID=TRUTH_DIR")
    parser.add_argument(
        "--scoring-support-manifest", action="append", default=[], metavar="SPLIT_ID=MANIFEST"
    )
    parser.add_argument("--extractor-commit", required=True)
    parser.add_argument("--profile-definition", type=Path, default=DEFAULT_PROFILE_DEFINITION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare every byte of an existing release instead of writing one.",
    )
    args = parser.parse_args(argv)

    truth_dirs = _pairs(args.source, "--source")
    support_manifests = _pairs(args.scoring_support_manifest, "--scoring-support-manifest")
    unknown = sorted(set(support_manifests) - set(truth_dirs))
    if unknown:
        parser.error(f"--scoring-support-manifest names splits without --source: {unknown}")
    if not args.check and args.output.exists():
        parser.error(f"output already exists; use --check to compare: {args.output}")

    sources = [
        CaseSetSource(
            split_file=SPLITS_ROOT / f"{split_id}.json",
            truth_dir=truth_dir,
            scoring_support_manifest=support_manifests.get(split_id),
        )
        for split_id, truth_dir in truth_dirs.items()
    ]
    try:
        manifest = build_release(
            sources=sources,
            profile_definition_path=args.profile_definition,
            repository_root=ROOT,
            extractor_commit=args.extractor_commit,
            extractor_commit_file_sha256=extractor_file_sha256(args.extractor_commit),
            output_root=args.output,
            check=args.check,
        )
    except (ProfileTruthError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    manifest_sha = hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "mode": "check" if args.check else "write",
                "release_id": manifest["release_id"],
                "manifest_sha256": manifest_sha,
                "case_count": manifest["case_count"],
                "case_sets": {entry["case_set_id"]: entry["case_count"] for entry in manifest["case_sets"]},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
