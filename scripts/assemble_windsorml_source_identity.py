#!/usr/bin/env python3
"""Assemble the pinned WindsorML public-source identity from case-support shards.

Consumes the shard JSON written by ``build_windsorml_case_support.py``, checks
that all 350 published runs are present exactly once, validates every case
against the frozen contract (field inventory, axis convention, force replay
tolerance), and emits
``benchmark-specs/windsorml/public-source-identity/windsorml-public-source-identity-v1.json``.

It also reports the observed force-replay distribution, which is what
:data:`reference.windsorml.contract.FORCE_REPLAY_RELATIVE_TOLERANCE` must be
set from -- run this once, read the reported maximum, then pin the constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reference.windsorml.contract import (  # noqa: E402
    AXIS_CONVENTION,
    CASE_COUNT,
    DATASET_ID,
    DATASET_VERSION,
    FORCE_REFERENCE_AREA_M2,
    FORCE_REPLAY_ABSOLUTE_TOLERANCE,
    FORCE_REPLAY_RELATIVE_TOLERANCE,
    FORCE_TRUTH_SOURCE,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    SOURCE_SCHEMA,
    SURFACE_ASSOCIATION,
    SURFACE_FIELDS,
    VOLUME_ASSOCIATION,
    VOLUME_FIELDS,
    sha256_file,
)


SURFACE_FIELD_UNITS = {
    "cpavg": "1",
    "cfxavg": "1",
    "cfyavg": "1",
    "cfzavg": "1",
}
VOLUME_FIELD_UNITS = {
    "velocityxavg": "m s-1",
    "velocityyavg": "m s-1",
    "velocityzavg": "m s-1",
    "pressureavg": "Pa",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path(
            "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton"
            "/windsorml/fluidsbench/case_support"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "benchmark-specs"
        / DATASET_ID
        / "public-source-identity"
        / "windsorml-public-source-identity-v1.json",
    )
    args = parser.parse_args()

    shards = sorted(args.shard_dir.glob("shard-*.json"))
    if not shards:
        raise SystemExit(f"no shard JSON found in {args.shard_dir}")

    records: dict[int, dict] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        if payload["repository_revision"] != REPOSITORY_REVISION:
            raise SystemExit(f"{shard.name} pins a different revision")
        if payload.get("volume_hash_skipped"):
            raise SystemExit(f"{shard.name} was built with --skip-volume-hash")
        for case in payload["cases"]:
            run_id = case["run_id"]
            if run_id in records:
                raise SystemExit(f"run_{run_id} appears in more than one shard")
            records[run_id] = case

    missing = [r for r in range(CASE_COUNT) if r not in records]
    if missing or len(records) != CASE_COUNT:
        raise SystemExit(
            f"expected {CASE_COUNT} cases, have {len(records)}; missing {missing[:10]}"
        )

    # Replay distribution -- this is what the frozen tolerances come from.
    # Absolute deltas matter more than relative ones because lift and side
    # force can be legitimately near zero (run_306 publishes cl = +0.000106,
    # where a negligible 0.0003 offset is a 310% relative difference).
    absolute = {"cd": [], "cl": []}
    relative = {"cd": [], "cl": []}
    for run_id in sorted(records):
        for key in absolute:
            absolute[key].append(abs(records[run_id]["replay_delta"][key]))
            value = records[run_id]["replay_relative_delta"][key]
            if value is not None:
                relative[key].append(abs(value))

    print(f"force replay deltas over {len(records)} published runs:")
    for key in ("cd", "cl"):
        values = absolute[key]
        ordered = sorted(values)
        print(
            f"  {key} |absolute|: median {statistics.median(values):.6f}  "
            f"p95 {ordered[int(0.95 * len(ordered))]:.6f}  max {max(values):.6f}"
        )
    for key in ("cd", "cl"):
        values = relative[key]
        ordered = sorted(values)
        print(
            f"  {key} |relative|: median {statistics.median(values):.4%}  "
            f"p95 {ordered[int(0.95 * len(ordered))]:.4%}  max {max(values):.4%}"
        )

    # Report the tightest constants that would still admit every run.
    worst_gap = 0.0
    for run_id, record in records.items():
        published = record["published_forces"]
        for key in ("cd", "cl"):
            need = abs(record["replay_delta"][key]) - (
                FORCE_REPLAY_RELATIVE_TOLERANCE * abs(published[key])
            )
            worst_gap = max(worst_gap, need)
    print(
        f"  with RELATIVE={FORCE_REPLAY_RELATIVE_TOLERANCE}, the minimum viable "
        f"ABSOLUTE is {worst_gap:.6f} (frozen at {FORCE_REPLAY_ABSOLUTE_TOLERANCE})"
    )
    if worst_gap > FORCE_REPLAY_ABSOLUTE_TOLERANCE:
        raise SystemExit(
            "frozen FORCE_REPLAY_ABSOLUTE_TOLERANCE is too tight for the observed data"
        )
    print()

    cases = []
    for run_id in sorted(records):
        record = records[run_id]
        files = record["files"]
        for role in (
            "boundary",
            "volume",
            "surface_dual_area",
            "geometry_parameters",
            "force_coefficients",
        ):
            if not files[role]["sha256"]:
                raise SystemExit(f"run_{run_id} {role} has no SHA-256")
        published = record["published_forces"]
        cases.append(
            {
                "case_id": record["case_id"],
                "run_id": run_id,
                "surface_entity_count": record["surface_entity_count"],
                "volume_entity_count": record["volume_entity_count"],
                "boundary": files["boundary"],
                "surface_dual_area": files["surface_dual_area"],
                "volume": files["volume"],
                "geometry_parameters": files["geometry_parameters"],
                "force_coefficients": files["force_coefficients"],
                "force_truth": {
                    "cd": published["cd"],
                    "cl": published["cl"],
                    "cs": published["cs"],
                    "cmy": published["cmy"],
                },
            }
        )

    document = {
        "schema": SOURCE_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "repository": {
            "id": REPOSITORY_ID,
            "revision": REPOSITORY_REVISION,
            "revision_kind": "git_commit",
            "license_spdx": "CC-BY-SA-4.0",
        },
        "surface_association": SURFACE_ASSOCIATION,
        "volume_association": VOLUME_ASSOCIATION,
        "surface_fields": {
            name: {
                "components": components,
                "dtype": "Float32",
                "unit": SURFACE_FIELD_UNITS[name],
            }
            for name, components in SURFACE_FIELDS.items()
        },
        "volume_fields": {
            name: {
                "components": components,
                "dtype": "Float32",
                "unit": VOLUME_FIELD_UNITS[name],
            }
            for name, components in VOLUME_FIELDS.items()
        },
        "force_convention": {
            "reference_area_m2": FORCE_REFERENCE_AREA_M2,
            "axes": dict(AXIS_CONVENTION),
            "truth_source": FORCE_TRUTH_SOURCE,
            "audit_reference": "published_force_mom_csv",
            "replay_note": (
                "Coefficients are scored by integrating the truth and the "
                "prediction identically against the published barycentric dual "
                "areas, so the point quadrature's ~0.26% offset from the solver's "
                "exact cell integration cancels and a perfect prediction scores "
                "R^2 = 1. The published force_mom CSV is retained as the audit "
                "anchor: the truth integration must reproduce it within the "
                "frozen relative tolerance."
            ),
        },
        "case_count": len(cases),
        "cases": cases,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    digest = sha256_file(args.out)
    print(f"wrote {args.out}")
    print(f"  cases: {len(cases)}")
    print(f"  SHA-256: {digest}")
    print("\nPin this digest as SOURCE_IDENTITY_SHA256 in reference/windsorml/contract.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
