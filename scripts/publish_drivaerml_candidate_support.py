#!/usr/bin/env python3
"""Publish compact candidate support metadata; never regenerate native arrays."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.drivaerml.dev_profiles import sha256
from reference.weightings import evaluator_weighting
from scripts.validate_scoring_supports import expected_metric_reduction

SPEC_ROOT = ROOT / "benchmark-specs/drivaerml"
RELEASE = "drivaerml-native-dev-v2-candidate"
SOURCE_COMMIT = "781637418bd9bde43b6c9a7aeb2e3c2563404e79"
BASE = f"https://raw.githubusercontent.com/neilashton/fluidsbench-submission/{SOURCE_COMMIT}/benchmark-specs/drivaerml/"
FORCE = "drivaerml-force-coefficients-v1"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return sha256(path)


def source_artifact(file, role):
    path = SPEC_ROOT / file
    return {
        "role": role,
        "url": BASE + file,
        "sha256": sha256(path),
        "byte_size": path.stat().st_size,
        "format": "json",
    }


def definitions(spec):
    result = []
    for sid, domain in (
        ("surface_native_cells", "surface"),
        ("volume_native_cells", "volume"),
        (FORCE, "scalar_case"),
    ):
        metrics = [
            m
            for m in spec["metrics"]
            if (
                m["aggregation"] == "all_test_cases"
                if domain == "scalar_case"
                else m["aggregation"] == "per_geometry_then_macro_average"
                and (
                    m["id"].startswith(domain + "_")
                    or m["id"].startswith("drivaerml_" + domain + "_")
                )
            )
        ]
        bindings = [
            {
                "metric_id": m["id"],
                "quantity_id": (
                    "forces"
                    if domain == "scalar_case"
                    else (
                        "pressure"
                        if "pressure" in m["id"]
                        else "wall_shear" if domain == "surface" else "velocity"
                    )
                ),
                "reduction": expected_metric_reduction(m["id"], m),
                "weighting": evaluator_weighting(m["weighting"]),
                "dataset_weighting": m["weighting"],
                "aggregation": m["aggregation"],
                "case_evidence": (
                    "aggregate_only" if m["kind"] == "r2" else "metric_value"
                ),
            }
            for m in metrics
        ]
        location = {
            "mode": "reference_generator",
            "generator_id": (
                "drivaerml-native-force-integration"
                if domain == "scalar_case"
                else "drivaerml-pinned-native-" + domain
            ),
            "generator_version": spec["evaluation_reference_version"],
            "parameters": {
                "source_pin_sha256": sha256(
                    SPEC_ROOT / "proposal/native-source-pin.json"
                )
            },
            "support_id_rule": {"kind": "source_entity_index"},
            "weight_rule": (
                {
                    "kind": "canonical_rule",
                    "rule_id": "published-native-polygon-area",
                    "rule_version": "1",
                }
                if domain == "surface"
                else {"kind": "uniform"}
            ),
        }

        def quantity(qid, unit, fields):
            return {
                "id": qid,
                "unit": unit,
                "components": [
                    {
                        "id": label,
                        "target_field": field,
                        "prediction_field": field,
                        "target_association": (
                            "generated" if domain == "scalar_case" else "cell_data"
                        ),
                    }
                    for label, field in fields
                ],
            }

        if domain == "scalar_case":
            quantities = [
                quantity(
                    "forces",
                    "1",
                    [(x, x) for x in ("cd", "cl", "cm_pitch", "clf", "clr")],
                )
            ]
        else:
            quantities = [quantity("pressure", "m^2/s^2", [("p", "pMeanTrim")])]
            if domain == "surface":
                quantities.append(
                    quantity(
                        "wall_shear",
                        "m^2/s^2",
                        [
                            (x, f"wallShearStressMeanTrim[{i}]")
                            for i, x in enumerate(("x", "y", "z"))
                        ],
                    )
                )
            else:
                quantities.append(
                    quantity(
                        "velocity",
                        "m/s",
                        [(x, f"UMeanTrim[{i}]") for i, x in enumerate(("x", "y", "z"))],
                    )
                )
        result.append(
            {
                "id": sid,
                "domain": domain,
                "location_definition": location,
                "quantities": quantities,
                "metric_bindings": bindings,
                "required_coverage": {
                    "count_fraction": 1.0,
                    "weight_fraction": 1.0,
                    "unmapped_count": 0,
                },
                "extrapolation_policy": "forbidden",
            }
        )
    return result


def publish(output):
    if output.exists():
        raise ValueError("output already exists")
    spec = json.loads((SPEC_ROOT / "submission-spec.json").read_text())
    pin = json.loads((SPEC_ROOT / "proposal/native-source-pin.json").read_text())
    audit = json.loads(
        (
            SPEC_ROOT / "evidence/native-volume-equal-cell-primary-all484.json"
        ).read_text()
    )
    volumes = {
        c["case_id"]: c["required_cell_data"]["UMeanTrim"]["tuple_count"]
        for c in audit["cases"]
    }
    pinned = {c["case_id"]: c for c in pin["cases"]}
    identity = source_artifact("proposal/native-source-pin.json", "native-source-pin")
    volume_audit = source_artifact(
        "evidence/native-volume-equal-cell-primary-all484.json", "native-volume-counts"
    )
    sets, seen = [], set()
    for split in spec["splits"]:
        sid = split["case_set_id"]
        if sid in seen:
            continue
        seen.add(sid)
        path = SPEC_ROOT / split["index_file"]
        if sha256(path) != split["sha256"]:
            raise ValueError("split hash differs")
        ids = json.loads(path.read_text())["case_ids"]
        chunks = []
        for start in range(0, len(ids), 25):
            cases = []
            for cid in ids[start : start + 25]:
                native = pinned[cid]
                cases.append(
                    {
                        "case_id": cid,
                        "support_instances": [
                            {
                                "support_id": "surface_native_cells",
                                "entity_count": native["surface_cell_area"][
                                    "element_count"
                                ],
                                "artifacts": [identity],
                                "parameters": {
                                    "boundary_sha256": native["boundary"]["lfs_sha256"],
                                    "area_sha256": native["surface_cell_area"][
                                        "lfs_sha256"
                                    ],
                                    "area_path": native["surface_cell_area"]["path"],
                                    "association": "CellData",
                                },
                            },
                            {
                                "support_id": "volume_native_cells",
                                "entity_count": volumes[cid],
                                "artifacts": [identity, volume_audit],
                                "parameters": {
                                    "weighting": "one_per_native_cell",
                                    "association": "CellData",
                                    "ordered_parts": native["volume"]["parts"],
                                },
                            },
                            {
                                "support_id": FORCE,
                                "entity_count": 1,
                                "artifacts": [identity],
                                "parameters": {
                                    "truth": "force_mom_constref_all.csv",
                                    "moment": "CmPitch=(Clf-Clr)/2",
                                },
                            },
                        ],
                    }
                )
            name = f"chunk-{start//25:03d}.json"
            digest = write(
                output / "case-sets" / sid / name,
                {
                    "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
                    "schema_version": "1.0",
                    "release_id": RELEASE,
                    "dataset_id": "drivaerml",
                    "case_set_id": sid,
                    "cases": cases,
                },
            )
            chunks.append(
                {
                    "file": name,
                    "sha256": digest,
                    "case_count": len(cases),
                    "case_ids": [c["case_id"] for c in cases],
                }
            )
        index = f"case-sets/{sid}/index.json"
        digest = write(
            output / index,
            {
                "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
                "schema_version": "1.0",
                "release_id": RELEASE,
                "dataset_id": "drivaerml",
                "case_set_id": sid,
                "case_count": len(ids),
                "chunks": chunks,
            },
        )
        sets.append(
            {
                "id": sid,
                "case_count": len(ids),
                "index_file": index,
                "index_sha256": digest,
            }
        )
    write(
        output / "manifest.json",
        {
            "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
            "schema_version": "1.0",
            "release_id": RELEASE,
            "status": "candidate",
            "published_at": "2026-10-12T00:00:00Z",
            "dataset_id": "drivaerml",
            "dataset_version": spec["dataset_version"],
            "evaluation_reference_version": spec["evaluation_reference_version"],
            "coordinate_frame": {
                "id": "drivaerml-native-frame-v1",
                "axis_order": ["x", "y", "z"],
                "handedness": "right",
                "length_unit": "m",
                "normalization": "Native metres; published polygon areas; equal volume cells",
            },
            "supports": definitions(spec),
            "case_sets": sets,
            "notes": "Dev-only candidate; immutable native-v3 profiles supply exact extraction supports and zero-weight relative diagnostics. Official release gates remain closed.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    publish(parser.parse_args().output)
