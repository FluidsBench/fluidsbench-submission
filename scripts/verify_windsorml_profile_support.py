#!/usr/bin/env python3
"""Verify the generated WindsorML profile support and emit its pinned manifest.

Checks every scored case against the frozen definition: schema, case binding,
family and station completeness, sample counts, native-ID ranges, finite truth,
and the expected ground-plane fallback pattern. Then writes a manifest that
hash-pins each case file so the evaluator can bind to an immutable support set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reference.windsorml.contract import (  # noqa: E402
    SOURCE_IDENTITY_SHA256,
    load_source_identity,
)

SPEC_DIR = REPO_ROOT / "benchmark-specs" / "windsorml"
DEFINITION = SPEC_DIR / "profile-definition-v2.json"
SCHEMA = "windsorml-profile-support-v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=Path(
            "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton"
            "/windsorml/fluidsbench/profile_support_v3"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SPEC_DIR / "profile-support" / "windsorml-profile-support-v3-manifest.json",
    )
    args = parser.parse_args()

    definition = json.loads(DEFINITION.read_text())
    sample_count = definition["sampling"]["sample_count"]
    families = {f["family_id"]: set(f["station_ids"]) for f in definition["families"]}

    identity = load_source_identity(
        SPEC_DIR / "public-source-identity" / "windsorml-public-source-identity-v1.json",
        expected_sha256=SOURCE_IDENTITY_SHA256,
    )

    scored: set[str] = set()
    for entry in json.loads((SPEC_DIR / "splits" / "splits-index.json").read_text()):
        scored |= set(json.loads((SPEC_DIR / entry["index_file"]).read_text())["case_ids"])
    ordered = sorted(scored, key=lambda c: int(c.removeprefix("run_")))

    problems: list[str] = []
    entries = []
    total_fallbacks = 0
    heights = []

    for case_id in ordered:
        path = args.support_dir / f"{case_id}.json"
        if not path.is_file():
            problems.append(f"{case_id}: missing")
            continue
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            problems.append(f"{case_id}: unreadable ({error})")
            continue

        if document.get("schema") != SCHEMA:
            problems.append(f"{case_id}: schema {document.get('schema')!r}")
        if document.get("case_id") != case_id:
            problems.append(f"{case_id}: binds {document.get('case_id')!r}")
        if document.get("sample_count") != sample_count:
            problems.append(f"{case_id}: sample_count {document.get('sample_count')}")

        case = identity.case(case_id)
        height = document.get("body_height_m")
        if not isinstance(height, (int, float)) or not 0.29 < height < 0.50:
            problems.append(f"{case_id}: implausible body height {height}")
        else:
            heights.append(float(height))

        got = document.get("families", {})
        if set(got) != set(families):
            problems.append(f"{case_id}: families {sorted(got)}")
            continue
        for family, expected_stations in families.items():
            if set(got[family]) != expected_stations:
                problems.append(f"{case_id}/{family}: stations {sorted(got[family])}")
                continue
            for station, payload in got[family].items():
                key = "native_cell_ids" if "native_cell_ids" in payload else "native_point_ids"
                ids = payload[key]
                truth_key = "truth_ux_over_uinf" if key == "native_cell_ids" else "truth_cp"
                truth = payload[truth_key]
                if len(ids) != sample_count or len(truth) != sample_count:
                    problems.append(f"{case_id}/{family}/{station}: length")
                limit = (
                    case.volume_entity_count
                    if key == "native_cell_ids"
                    else case.surface_entity_count
                )
                if min(ids) < 0 or max(ids) >= limit:
                    problems.append(f"{case_id}/{family}/{station}: id out of range")
                if any(v != v or v in (float("inf"), float("-inf")) for v in truth):
                    problems.append(f"{case_id}/{family}/{station}: non-finite truth")

        fallbacks = sum(
            v for k, v in (document.get("diagnostics") or {}).items() if "fallback" in k
        )
        total_fallbacks += fallbacks
        # Each of the 8 vertical stations starts at y = 0 on the ground plane,
        # which lies outside the fluid mesh; the 2 lateral ones do not.
        if fallbacks != 8:
            problems.append(f"{case_id}: {fallbacks} fallbacks, expected 8")

        entries.append(
            {
                "case_id": case_id,
                "file": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "body_height_m": height,
            }
        )

    print(f"cases checked : {len(ordered)}")
    print(f"cases valid   : {len(entries)}")
    print(f"problems      : {len(problems)}")
    for problem in problems[:20]:
        print(f"    {problem}")
    if heights:
        print(f"body height   : {min(heights):.5f} .. {max(heights):.5f} m")
    print(f"fallbacks     : {total_fallbacks} total ({total_fallbacks / max(len(entries),1):.1f}/case)")

    if problems:
        print("\nRESULT: INCOMPLETE")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "windsorml-profile-support-manifest-v1",
        "dataset_id": "windsorml",
        "support_schema": SCHEMA,
        "profile_definition_sha256": sha256(DEFINITION),
        "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "sample_count": sample_count,
        "case_count": len(entries),
        "cases": entries,
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print(f"  manifest SHA-256: {sha256(args.out)}")
    print("\nRESULT: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
