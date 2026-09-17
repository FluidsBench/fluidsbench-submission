#!/usr/bin/env python3
"""End-to-end smoke test of the WindsorML evaluator on one real case.

Builds a *perfect* surface prediction from the case's own native truth fields
and feeds it back through the evaluator. Every relative-L2 must come out at
zero and the integrated prediction coefficients must equal the integrated truth
exactly, on real 2.2M-point inline-binary data with the published dual areas --
which the synthetic unit tests cannot demonstrate.

Too heavy for the unit suite (a single boundary file is ~400 MB), so it lives
here and is run deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reference.windsorml.contract import (  # noqa: E402
    SOURCE_IDENTITY_SHA256,
    load_source_identity,
)
from reference.windsorml.evaluator import evaluate_candidate_case  # noqa: E402
from reference.windsorml.prediction_chunks import (  # noqa: E402
    CANDIDATE_ARTIFACT_ROLE,
    WINDSORML_CANDIDATE_FORMAT,
    WINDSORML_SURFACE_SUPPORT_ID,
)

DATASET_ROOT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/data"
)
SURFACE_FIELDS = ("cpavg", "cfxavg", "cfyavg", "cfzavg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_surface_truth(path: Path) -> dict[str, np.ndarray]:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    point_data = reader.GetOutput().GetPointData()
    fields = {}
    for name in SURFACE_FIELDS:
        array = point_data.GetArray(name)
        if array is None:
            raise SystemExit(f"{path.name} lacks PointData {name!r}")
        fields[name] = vtk_to_numpy(array).astype(np.float32)
    return fields


def write_chunks(root: Path, case_id: str, fields: dict[str, np.ndarray], chunks: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    total = next(iter(fields.values())).shape[0]
    bounds = np.array_split(np.arange(total), chunks)
    descriptors = []
    for index, rows in enumerate(bounds):
        chunk = root / f"chunk-{index:05d}.npz"
        np.savez(
            chunk,
            raw_cell_id=rows.astype(np.int64),
            **{name: values[rows] for name, values in fields.items()},
        )
        descriptors.append(
            {
                "chunk_index": index,
                "file": chunk.name,
                "sha256": sha256(chunk),
                "row_count": int(rows.shape[0]),
                "raw_cell_id_start": int(rows[0]),
                "raw_cell_id_stop": int(rows[-1]) + 1,
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": WINDSORML_CANDIDATE_FORMAT,
                "format_version": 1,
                "artifact_role": CANDIDATE_ARTIFACT_ROLE,
                "case_id": case_id,
                "support_id": WINDSORML_SURFACE_SUPPORT_ID,
                "association": "PointData",
                "total_row_count": total,
                "field_components": {name: 1 for name in fields},
                "chunks": descriptors,
            }
        )
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="run_0")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()

    identity = load_source_identity(
        REPO_ROOT
        / "benchmark-specs"
        / "windsorml"
        / "public-source-identity"
        / "windsorml-public-source-identity-v1.json",
        expected_sha256=SOURCE_IDENTITY_SHA256,
    )
    case = identity.case(args.case_id)
    boundary = case.boundary.resolve(args.dataset_root)
    print(f"case {case.case_id}: {case.surface_entity_count:,} native surface points")

    truth = read_surface_truth(boundary)
    for name, values in truth.items():
        if values.shape[0] != case.surface_entity_count:
            raise SystemExit(f"{name} has {values.shape[0]} rows")
    print(f"read {len(truth)} native surface fields")

    with tempfile.TemporaryDirectory(prefix="wml-smoke-") as temporary:
        manifest = write_chunks(
            Path(temporary) / "surface", case.case_id, truth, args.chunks
        )
        print(f"wrote a perfect prediction in {args.chunks} chunks")
        evidence = evaluate_candidate_case(
            case=case,
            dataset_root=args.dataset_root,
            surface_manifest=manifest,
        ).to_json()

    metrics = evidence["surface"]["metrics"]
    print("\nsurface metrics (all must be 0):")
    failures = []
    for name, value in metrics.items():
        flag = "" if abs(value) < 1e-9 else "   <== NON-ZERO"
        if flag:
            failures.append(name)
        print(f"  {name:44s} {value:.12f}{flag}")

    forces = evidence["forces"]
    print("\nforces:")
    for key in ("cd", "cl", "cs"):
        t = forces["truth_integrated"][key]
        p = forces["prediction_integrated"][key]
        print(f"  {key}: truth {t:+.9f}  prediction {p:+.9f}  identical={t == p}")
        if t != p:
            failures.append(f"force_{key}")
    audit = forces["published_audit"]
    print("\npublished-CSV audit (absolute deltas):")
    for key, delta in audit["relative_delta"].items():
        print(f"  {key}: published {audit[key]:+.9f}  |delta| {delta:.9f}")

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
