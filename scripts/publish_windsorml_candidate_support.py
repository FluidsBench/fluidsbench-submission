#!/usr/bin/env python3
"""Publish compact schema-v3 support metadata from WindsorML's pinned contracts.

No CFD arrays are regenerated. Surface quadrature uses the published NPY bytes
named and hashed in the immutable source identity; volume weighting is uniform.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
from reference.windsorml.contract import (
    DATASET_VERSION,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
    load_source_identity,
    sha256_file,
)
from reference.windsorml.profiles import (
    PROFILE_DEFINITION_SHA256,
    PROFILE_MANIFEST_SHA256,
    load_profile_support,
)
from reference.windsorml.dataset_scorer import PROFILE_FAMILY_METRICS

SPEC_ROOT = ROOT / "benchmark-specs/windsorml"
RELEASE_ID = "windsorml-native-support-v1-candidate"
EVALUATOR_VERSION = "windsorml-evaluator-v0.1-candidate"
# These pre-existing source/profile files are immutable at this commit.
SOURCE_COMMIT = "8a8c4a052c4aa1e2739753ceaaf8575058da344f"
SOURCE_URL = f"https://raw.githubusercontent.com/neilashton/fluidsbench-submission/{SOURCE_COMMIT}/benchmark-specs/windsorml/"
IDENTITY_PATH = (
    SPEC_ROOT / "public-source-identity/windsorml-public-source-identity-v1.json"
)
SURFACE = "windsorml_surface_native_points"
VOLUME = "windsorml_volume_native_cells"
FORCE = "windsorml-force-coefficients-v1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def quantity(identifier, fields, association, unit=""):
    return {
        "id": identifier,
        "unit": unit,
        "components": [
            {
                "id": "value" if len(fields) == 1 else axis,
                "target_field": name,
                "target_association": association,
                "prediction_field": name,
            }
            for axis, name in zip(("x", "y", "z"), fields)
        ],
    }


def support(identifier, domain, location, quantities, metrics, spec):
    declared = {m["id"]: m for m in spec["metrics"]}
    bindings = []
    for metric, qid in metrics:
        m = declared[metric]
        reduction = (
            "relative_l2_percent"
            if metric.endswith("rel_l2")
            else "r2" if m["kind"] == "r2" else "mae"
        )
        bindings.append(
            {
                "metric_id": metric,
                "quantity_id": qid,
                "reduction": reduction,
                "weighting": (
                    "support_weights"
                    if m["weighting"] == "surface_point_dual_area"
                    else "uniform"
                ),
                "dataset_weighting": m["weighting"],
                "aggregation": m["aggregation"],
                "case_evidence": (
                    "aggregate_only" if reduction == "r2" else "metric_value"
                ),
            }
        )
    return {
        "id": identifier,
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


def native_location(surface):
    return {
        "mode": "native_entities",
        "format": "vtk_xml_unstructured_grid",
        "artifact_role": "public-boundary-vtu" if surface else "public-volume-vtu",
        "entity": "point" if surface else "cell",
        "selection": {"kind": "all_entities"},
        "coordinate_rule": "source_point" if surface else "cell_centres_v1",
        "ordering": "source_entity_index",
        "support_id_rule": {"kind": "source_entity_index"},
        "weight_rule": (
            {
                "kind": "canonical_rule",
                "rule_id": "windsorml-published-point-dual-area",
                "rule_version": "1",
            }
            if surface
            else {"kind": "uniform"}
        ),
    }


def generated_location(identifier):
    return {
        "mode": "reference_generator",
        "generator_id": identifier,
        "generator_version": EVALUATOR_VERSION,
        "parameters": {
            "source_identity_sha256": SOURCE_IDENTITY_SHA256,
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
            "profile_support_manifest_sha256": PROFILE_MANIFEST_SHA256,
        },
        "support_id_rule": {"kind": "source_entity_index"},
        "weight_rule": {"kind": "uniform"},
    }


def definitions(spec):
    result = [
        support(
            SURFACE,
            "surface",
            native_location(True),
            [
                quantity("cp", ["cpavg"], "point_data"),
                quantity("cf", ["cfxavg", "cfyavg", "cfzavg"], "point_data"),
            ],
            [
                (m, "cp" if "pressure" in m else "cf")
                for m in (
                    "surface_pressure_rel_l2",
                    "surface_pressure_equal_entity_rel_l2",
                    "surface_wall_shear_rel_l2",
                    "surface_wall_shear_equal_entity_rel_l2",
                )
            ],
            spec,
        ),
        support(
            VOLUME,
            "volume",
            native_location(False),
            [
                quantity("pressure", ["pressureavg"], "cell_data", "native_export_units_unresolved"),
                quantity(
                    "velocity",
                    ["velocityxavg", "velocityyavg", "velocityzavg"],
                    "cell_data",
                    "m/s",
                ),
            ],
            [
                ("volume_pressure_rel_l2", "pressure"),
                ("volume_velocity_rel_l2", "velocity"),
            ],
            spec,
        ),
        support(
            FORCE,
            "scalar_case",
            generated_location("windsorml-native-field-force-integration-v1"),
            [quantity("cd", ["cd"], "generated"), quantity("cl", ["cl"], "generated")],
            [
                ("cd_r2", "cd"),
                ("cl_r2", "cl"),
                ("c_drag_mae", "cd"),
                ("c_lift_mae", "cl"),
            ],
            spec,
        ),
    ]
    for family, metric in PROFILE_FAMILY_METRICS.items():
        qid = "cp" if "_cp_" in family else "ux_over_uinf"
        result.append(
            support(
                family,
                "line",
                generated_location(family),
                [quantity(qid, [qid], "generated")],
                [(metric, qid)],
                spec,
            )
        )
    return result


def instances(case):
    profiles = load_profile_support(case_id=case.case_id)
    identity = {
        "role": "public-source-identity",
        "url": SOURCE_URL
        + "public-source-identity/windsorml-public-source-identity-v1.json",
        "sha256": SOURCE_IDENTITY_SHA256,
        "byte_size": IDENTITY_PATH.stat().st_size,
        "format": "json",
    }
    profile_path = SPEC_ROOT / "profile-support" / f"{case.case_id}.json"
    profile = {
        "role": "profile-support",
        "url": SOURCE_URL + f"profile-support/{case.case_id}.json",
        "sha256": profiles.sha256,
        "byte_size": profile_path.stat().st_size,
        "format": "json",
    }

    def native_artifact(source, role):
        return {
            "role": role,
            "url": f"https://huggingface.co/datasets/neashton/windsorml/resolve/{REPOSITORY_REVISION}/{source.path}",
            "sha256": source.sha256,
            "byte_size": source.size_bytes,
            "format": "vtk_xml_unstructured_grid",
        }

    result = [
        {
            "support_id": SURFACE,
            "entity_count": case.surface_entity_count,
            "artifacts": [
                identity,
                native_artifact(case.boundary, "public-boundary-vtu"),
            ],
            "parameters": {
                "association": "PointData",
                "dual_area_path": str(case.surface_dual_area.path),
                "dual_area_sha256": case.surface_dual_area.sha256,
                "dual_area_size_bytes": case.surface_dual_area.size_bytes,
                "weight_rule": "published_npy_in_native_point_order_no_regeneration",
            },
        },
        {
            "support_id": VOLUME,
            "entity_count": case.volume_entity_count,
            "artifacts": [identity, native_artifact(case.volume, "public-volume-vtu")],
            "parameters": {
                "association": "CellData",
                "weight_rule": "equal_native_cell",
            },
        },
        {
            "support_id": FORCE,
            "entity_count": 1,
            "artifacts": [identity],
            "parameters": {
                "reference_area_m2": 0.112,
                "axes": {"drag": "+x", "lift": "+y", "side": "+z"},
                "truth": "native_field_integration",
                "published_csv": "audit_only",
            },
        },
    ]
    for family in PROFILE_FAMILY_METRICS:
        result.append(
            {
                "support_id": family,
                "entity_count": len(profiles.document["families"][family]) * 128,
                "artifacts": [profile],
                "expected_support_sha256": profiles.sha256,
            }
        )
    return result


def publish(output: Path, *, published_at="2026-10-11T00:00:00Z"):
    spec = json.loads((SPEC_ROOT / "submission-spec.json").read_text())
    identity = load_source_identity(
        IDENTITY_PATH, expected_sha256=SOURCE_IDENTITY_SHA256
    )
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".windsorml-support-", dir=output.parent))
    seen, case_sets, cache = set(), [], {}
    try:
        for split in spec["splits"]:
            set_id = split["case_set_id"]
            if set_id in seen:
                continue
            seen.add(set_id)
            split_path = SPEC_ROOT / split["index_file"]
            if sha256_file(split_path) != split["sha256"]:
                raise ValueError("split digest differs")
            case_ids = json.loads(split_path.read_text())["case_ids"]
            chunks = []
            for start in range(0, len(case_ids), 25):
                selected = case_ids[start : start + 25]
                cases = []
                for cid in selected:
                    if cid not in cache:
                        cache[cid] = instances(identity.case(cid))
                    cases.append({"case_id": cid, "support_instances": cache[cid]})
                name = f"chunk-{start//25:03d}.json"
                digest = write_json(
                    stage / "case-sets" / set_id / name,
                    {
                        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
                        "schema_version": "1.0",
                        "release_id": RELEASE_ID,
                        "dataset_id": "windsorml",
                        "case_set_id": set_id,
                        "cases": cases,
                    },
                )
                chunks.append(
                    {
                        "file": name,
                        "sha256": digest,
                        "case_count": len(selected),
                        "case_ids": selected,
                    }
                )
            index = f"case-sets/{set_id}/index.json"
            digest = write_json(
                stage / index,
                {
                    "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
                    "schema_version": "1.0",
                    "release_id": RELEASE_ID,
                    "dataset_id": "windsorml",
                    "case_set_id": set_id,
                    "case_count": len(case_ids),
                    "chunks": chunks,
                },
            )
            case_sets.append(
                {
                    "id": set_id,
                    "case_count": len(case_ids),
                    "index_file": index,
                    "index_sha256": digest,
                }
            )
        if len(cache) != 233:
            raise ValueError("expected exactly 233 unique scored cases")
        write_json(
            stage / "manifest.json",
            {
                "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
                "schema_version": "1.0",
                "release_id": RELEASE_ID,
                "status": "candidate",
                "published_at": published_at,
                "dataset_id": "windsorml",
                "dataset_version": DATASET_VERSION,
                "evaluation_reference_version": EVALUATOR_VERSION,
                "coordinate_frame": {
                    "id": "windsorml-native-body-frame-v1",
                    "axis_order": ["x", "y", "z"],
                    "handedness": "right",
                    "length_unit": "m",
                    "normalization": "native SI coordinates; dimensionless surface coefficients; +y vertical",
                },
                "supports": definitions(spec),
                "case_sets": case_sets,
                "notes": "Closed dev-only candidate. Published point dual areas are canonical; volume uses equal-cell weighting. Fixed profiles contribute to scoring; relative profiles are report-only.",
            },
        )
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output / "manifest.json"


def bind(manifest):
    path = SPEC_ROOT / "submission-spec.json"
    spec = json.loads(path.read_text())
    if spec["scoring_support"]["submissions_open"] is not False:
        raise ValueError("binding requires a closed candidate")
    relative = manifest.resolve().relative_to(SPEC_ROOT).as_posix()
    spec["scoring_support"]["candidate_manifest"] = {
        "status": "candidate",
        "release_id": RELEASE_ID,
        "manifest_file": relative,
        "manifest_url": f"https://github.com/neilashton/fluidsbench-submission/blob/dev/benchmark-specs/windsorml/{relative}",
        "manifest_sha256": sha256_file(manifest),
    }
    spec["profile_definition"] = {
        "id": "windsorml-profile-definition-v2",
        "file": "profile-definition-v2.json",
        "sha256": PROFILE_DEFINITION_SHA256,
    }
    definition = json.loads((SPEC_ROOT / "profile-definition-v2.json").read_text())
    for panel in spec["profile_panels"]:
        panel["station_ids"] = [
            s
            for f in definition["families"]
            if f["panel_id"] == panel["id"]
            for s in f["station_ids"]
        ]
    write_json(path, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=SPEC_ROOT / "scoring-support-candidate"
    )
    parser.add_argument("--bind", action="store_true")
    args = parser.parse_args()
    manifest = publish(args.output.resolve())
    if args.bind:
        bind(manifest)
    print(manifest)


if __name__ == "__main__":
    main()
