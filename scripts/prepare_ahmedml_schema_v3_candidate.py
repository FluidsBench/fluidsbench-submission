#!/usr/bin/env python3
"""Prepare the AhmedML native schema-v3 candidate contract.

This command is intentionally deterministic.  It binds the FluidsBench
development contract to one immutable public AhmedML revision, replaces the
prototype split identifiers with the dataset-published ``run_N`` identifiers,
and records the exact native source identities without rereading multi-terabyte
payloads.  Hugging Face local-dir metadata stores the content SHA-256 as its
ETag; critical small files without metadata are hashed directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "benchmark-specs" / "ahmedml"
REVISION = "02688c727cdb8dc8678e28abc6bbbb7e93c5fa15"
REPOSITORY_ID = "neashton/ahmedml"
DATASET_VERSION = "ahmedml-native-v1-candidate"
EVALUATOR_VERSION = "ahmedml-evaluator-v0.1-candidate"
PROFILE_DEFINITION_ID = "ahmedml-native-profiles-v1-candidate"
REGION_DEFINITION_ID = "ahmedml-native-regions-v2-candidate"
VOLUME_REGION_DEFINITION_ID = "ahmedml-native-regions-v1-candidate"
PROFILE_SAMPLE_COUNT = 128

SPLITS = (
    "full",
    "medium",
    "scarce",
    "super_scarce",
    "geometry",
    "high_drag",
    "low_drag",
    "image_wake",
)
SPLIT_LABELS = {
    "full": "Full",
    "medium": "Medium",
    "scarce": "Scarce",
    "super_scarce": "Super scarce",
    "geometry": "Geometry OOD",
    "high_drag": "High drag OOD",
    "low_drag": "Low drag OOD",
    "image_wake": "Image wake OOD",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> str:
    encoded = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def hf_content_sha256(dataset_root: Path, relative: Path) -> str:
    metadata = (
        dataset_root
        / ".cache"
        / "huggingface"
        / "download"
        / relative.parent
        / f"{relative.name}.metadata"
    )
    if metadata.is_file():
        lines = metadata.read_text().splitlines()
        if len(lines) >= 2 and re.fullmatch(r"[0-9a-f]{64}", lines[1]):
            if lines[0] != REVISION:
                raise ValueError(
                    f"{relative}: metadata revision {lines[0]!r} != {REVISION!r}"
                )
            return lines[1]
    return sha256_file(dataset_root / relative)


def vtk_declared_count(path: Path, *, key: str) -> int:
    with path.open("rb") as source:
        prefix = source.read(16 * 1024)
    match = re.search(rb"\b" + key.encode("ascii") + rb"=['\"]([0-9]+)['\"]", prefix)
    if match is None:
        raise ValueError(f"cannot find {key} in {path}")
    return int(match.group(1))


def source_file(dataset_root: Path, relative: Path) -> dict[str, Any]:
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative.as_posix(),
        "sha256": hf_content_sha256(dataset_root, relative),
        "size_bytes": path.stat().st_size,
    }


def profile_definition() -> dict[str, Any]:
    return {
        "schema": "ahmedml-native-profile-definition-v1",
        "definition_id": PROFILE_DEFINITION_ID,
        "status": "candidate_owner_review_required",
        "dataset_id": "ahmedml",
        "dataset_version": DATASET_VERSION,
        "dataset_repository": REPOSITORY_ID,
        "dataset_revision": REVISION,
        "association": "CellData",
        "sample_count": PROFILE_SAMPLE_COUNT,
        "coordinate_frame": {
            "axis_order": ["x", "y", "z"],
            "streamwise_axis": "+x",
            "lateral_axis": "y",
            "vertical_axis": "+z",
            "ground_plane_m": 0.0,
            "nominal_rear_plane_m": 0.0,
            "length_unit": "m",
            "geometry_parameter_input_unit": "mm",
        },
        "normalization": {
            "u_infinity_m_per_s": 1.0,
            "dynamic_pressure_kinematic_m2_per_s2": 0.5,
            "cp_from_pmean": "Cp = pMean / 0.5 = 2*pMean",
            "body_length": "body-length / 1000",
            "body_height": "body-height / 1000",
            "body_width": "body-width / 1000",
            "slant_dx": "slant-angle-length / 1000",
            "slant_dz": "slant-angle-height / 1000",
        },
        "mapping": {
            "algorithm_id": "ahmedml-vtk-static-cell-locator-v1",
            "surface_rule": (
                "find the nearest native boundary polygon to each target point; "
                "freeze its zero-based source CellData index and projected point"
            ),
            "volume_rule": (
                "find the native volume cell containing each target point; if a "
                "roundoff-only miss occurs, freeze the nearest native cell"
            ),
            "tie_breaking": (
                "the generated mapping artifact is authoritative; evaluator runtime "
                "never repeats geometric nearest-neighbour selection"
            ),
            "prediction_source": (
                "sample the same complete native CellData prediction arrays used for "
                "field scoring; participant-provided profile values are forbidden"
            ),
        },
        "panels": [
            {
                "panel_id": "pressure_profiles",
                "quantity_id": "cp",
                "source_support_id": "ahmedml_surface_native_cells",
                "source_field": "pMean",
                "transform": "2*pMean",
                "stations": [
                    {
                        "station_id": "upper_body_centerline",
                        "coordinate_id": "x_over_l",
                        "coordinate_range": [0.0, 1.0],
                        "target_rule": (
                            "x=-L+t*(L-slant_dx), y=0, z=z_surface_max+0.05*L; "
                            "nearest boundary polygon"
                        ),
                    },
                    {
                        "station_id": "underbody_centerline",
                        "coordinate_id": "x_over_l",
                        "coordinate_range": [0.0, 1.0],
                        "target_rule": (
                            "x=-L+t*L, y=0, z=0; nearest boundary polygon"
                        ),
                    },
                    {
                        "station_id": "rear_slant_centerline",
                        "coordinate_id": "s_over_slant",
                        "coordinate_range": [0.0, 1.0],
                        "target_rule": (
                            "x=-slant_dx+t*slant_dx, y=0, "
                            "z=z_surface_max-t*slant_dz; nearest boundary polygon"
                        ),
                    },
                ],
            },
            {
                "panel_id": "velocity_profiles",
                "quantity_id": "ux_over_uinf",
                "source_support_id": "ahmedml_volume_native_cells",
                "source_field": "UMean",
                "component": 0,
                "transform": "Ux / U_infinity",
                "stations": [
                    {
                        "station_id": "wake_vertical_x_0p25_l",
                        "coordinate_id": "z_over_h",
                        "coordinate_range": [0.0, 2.0],
                        "target_rule": "x=0.25*L, y=0, z=t*H",
                    },
                    {
                        "station_id": "wake_vertical_x_0p50_l",
                        "coordinate_id": "z_over_h",
                        "coordinate_range": [0.0, 2.0],
                        "target_rule": "x=0.50*L, y=0, z=t*H",
                    },
                    {
                        "station_id": "wake_vertical_x_1p00_l",
                        "coordinate_id": "z_over_h",
                        "coordinate_range": [0.0, 2.0],
                        "target_rule": "x=1.00*L, y=0, z=t*H",
                    },
                    {
                        "station_id": "wake_lateral_x_0p50_l_z_0p50_h",
                        "coordinate_id": "y_over_w",
                        "coordinate_range": [-1.0, 1.0],
                        "target_rule": "x=0.50*L, y=t*W, z=z_surface_min+0.50*H",
                    },
                ],
            },
        ],
    }


def regional_definition() -> dict[str, Any]:
    return {
        "schema": "ahmedml-native-region-definition-v1",
        "definition_id": VOLUME_REGION_DEFINITION_ID,
        "status": "candidate_report_only",
        "dataset_id": "ahmedml",
        "dataset_revision": REVISION,
        "association": "CellData",
        "coordinate_source": "native volume cell centres",
        "precedence": ["near_body", "wake", "farfield"],
        "regions": [
            {
                "id": "near_body",
                "code": 0,
                "rule": (
                    "-1.25*L <= x <= 0 and abs(y) <= 0.75*W and "
                    "0 <= z <= z_surface_min + 2*H"
                ),
            },
            {
                "id": "wake",
                "code": 1,
                "rule": (
                    "0 < x <= 2*L and abs(y) <= W and "
                    "0 <= z <= z_surface_min + 2*H"
                ),
            },
            {"id": "farfield", "code": 2, "rule": "all remaining cells"},
        ],
        "ranking_effect": "none",
        "notes": (
            "Regions are mutually exclusive and exhaustive. They diagnose field "
            "error localization but do not alter the candidate overall score."
        ),
    }


def build_splits(dataset_root: Path, manifest: dict[str, list[str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    split_dir = DATASET_DIR / "splits"
    for split in SPLITS:
        train = manifest[f"{split}_train"]
        validation = manifest[f"{split}_val"]
        test = manifest[f"{split}_test"]
        case_set_id = "full-test" if split in {
            "full", "medium", "scarce", "super_scarce"
        } else f"{split}-test"
        document = {
            "schema_version": "1.1",
            "dataset_id": "ahmedml",
            "dataset_version": DATASET_VERSION,
            "split_id": split,
            "case_set_id": case_set_id,
            "split_label": SPLIT_LABELS[split],
            "case_id_status": "official",
            "train_count": len(train),
            "validation_count": len(validation),
            "case_count": len(test),
            "case_ids": test,
            "source": {
                "repository_id": REPOSITORY_ID,
                "revision": REVISION,
                "manifest_file": "splits/manifest.json",
                "manifest_key": f"{split}_test",
            },
        }
        path = split_dir / f"{split}.json"
        digest = write_json(path, document)
        result.append(
            {
                "id": split,
                "label": SPLIT_LABELS[split],
                "index_file": f"splits/{split}.json",
                "case_count": len(test),
                "case_set_id": case_set_id,
                "case_id_status": "official",
                "sha256": digest,
            }
        )
    return result


def build_source_identity(dataset_root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for run_id in range(1, 501):
        run = f"run_{run_id}"
        boundary_rel = Path(run) / f"boundary_{run_id}.vtp"
        volume_rel = Path(run) / f"volume_{run_id}.vtu"
        area_rel = Path(run) / f"boundary_cell_area_{run_id}.npy"
        geometry_rel = Path(run) / f"geo_parameters_{run_id}.csv"
        force_rel = Path(run) / f"force_mom_{run_id}.csv"
        boundary = source_file(dataset_root, boundary_rel)
        volume = source_file(dataset_root, volume_rel)
        areas = source_file(dataset_root, area_rel)
        geometry = source_file(dataset_root, geometry_rel)
        force = source_file(dataset_root, force_rel)
        boundary_count = vtk_declared_count(
            dataset_root / boundary_rel, key="NumberOfPolys"
        )
        volume_count = vtk_declared_count(
            dataset_root / volume_rel, key="NumberOfCells"
        )
        area_array = np.load(dataset_root / area_rel, mmap_mode="r", allow_pickle=False)
        if area_array.dtype != np.dtype("<f4") or area_array.shape != (boundary_count,):
            raise ValueError(f"{area_rel}: area array does not match boundary polygons")
        cases.append(
            {
                "case_id": run,
                "run_id": run_id,
                "surface_entity_count": boundary_count,
                "volume_entity_count": volume_count,
                "boundary": boundary,
                "surface_cell_area": areas,
                "volume": volume,
                "geometry_parameters": geometry,
                "force_coefficients": force,
            }
        )
    return {
        "schema": "ahmedml-public-native-source-identity-v1",
        "dataset_id": "ahmedml",
        "dataset_version": DATASET_VERSION,
        "repository": {
            "id": REPOSITORY_ID,
            "revision": REVISION,
            "revision_kind": "git_commit",
            "license_spdx": "CC-BY-SA-4.0",
        },
        "association": "CellData",
        "surface_fields": {
            "pMean": {"components": 1, "dtype": "Float32", "unit": "m2 s-2"},
            "wallShearStressMean": {
                "components": 3,
                "dtype": "Float32",
                "unit": "m2 s-2",
            },
        },
        "volume_fields": {
            "pMean": {"components": 1, "dtype": "Float32", "unit": "m2 s-2"},
            "UMean": {"components": 3, "dtype": "Float32", "unit": "m s-1"},
        },
        "force_reference": {
            "rho_kg_per_m3": 1.0,
            "u_infinity_m_per_s": 1.0,
            "reference_area_m2": 0.112032,
            "reference_length_m": 1.04,
            "centre_of_rotation_m": [-0.502, 0.0, 0.0],
            "drag_direction": [1.0, 0.0, 0.0],
            "lift_direction": [0.0, 0.0, 1.0],
            "traction_rule": "pMean*A_outward - wallShearStressMean*|A|",
        },
        "case_count": len(cases),
        "cases": cases,
    }


def update_submission_spec(
    split_records: list[dict[str, Any]],
    profile_sha256: str,
    region_sha256: str,
    source_sha256: str,
) -> None:
    path = DATASET_DIR / "submission-spec.json"
    spec = json.loads(path.read_text())
    spec["schema_version"] = "1.2"
    spec["dataset_version"] = DATASET_VERSION
    spec["status"] = "candidate_native_support"
    spec["evaluation_reference_version"] = EVALUATOR_VERSION
    support = spec["scoring_support"]
    support["status"] = "owner_review_required"
    support["submissions_open"] = False
    support["closed_reason"] = (
        "AhmedML native evaluator and support are under owner review; only the "
        "explicit non-ranked development fixture may be exercised."
    )
    support["dataset_source"] = {
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "identity_file": "public-source-identity/ahmedml-public-source-identity-v1.json",
        "identity_sha256": source_sha256,
    }
    support["owner_decisions_required"] = [
        "review_and_approve_profile_station_geometry",
        "review_and_approve_candidate_score_weights_and_caps",
        "publish_the_generated_scoring_support_release",
        "approve_evaluator_implementation_and_open_submissions",
    ]
    for public_support in support["public_supports"]:
        public_support["verification"] = "pinned_public_revision_and_source_identity"
    for metric in spec["metrics"]:
        if metric["id"] in {
            "surface_pressure_mae",
            "surface_pressure_rmse",
            "surface_wall_shear_mae",
            "surface_wall_shear_rmse",
            "volume_pressure_mae",
            "volume_pressure_rmse",
        }:
            metric["unit"] = "m2/s2"
    spec["profile_definition"] = {
        "id": PROFILE_DEFINITION_ID,
        "file": "profile-definition-v1.json",
        "sha256": profile_sha256,
        "sample_count_per_series": PROFILE_SAMPLE_COUNT,
        "prediction_source": "evaluator_derived_from_complete_native_fields",
    }
    spec["regional_diagnostics"] = {
        "status": "candidate_report_only",
        "required_for_new_submissions": True,
        "format": "ahmedml-regional-diagnostics-aggregate-v2",
        "definition_id": REGION_DEFINITION_ID,
        "contract_file": "regional-diagnostics-v2.json",
        "contract_sha256": region_sha256,
        "role": "report_only",
        "weight": 0.0,
        "affects_official_metrics": False,
        "affects_official_score": False,
        "requires_new_inference": False,
        "prediction_format_changed": False,
        "activation_gate": (
            "Candidate evaluator support remains closed to real submissions until "
            "owner review; the development fixture exercises this required report."
        ),
    }
    spec["profile_panels"] = [
        {
            "id": "pressure_profiles",
            "required": True,
            "allow_unlisted_stations": False,
            "minimum_points": PROFILE_SAMPLE_COUNT,
            "exact_points": PROFILE_SAMPLE_COUNT,
            "coordinate_order": "strictly_increasing",
            "station_ids": [
                "upper_body_centerline",
                "underbody_centerline",
                "rear_slant_centerline",
            ],
            "station_sample_counts": {
                "upper_body_centerline": PROFILE_SAMPLE_COUNT,
                "underbody_centerline": PROFILE_SAMPLE_COUNT,
                "rear_slant_centerline": PROFILE_SAMPLE_COUNT,
            },
            "station_coordinate_intervals": {
                "upper_body_centerline": [0.0, 1.0],
                "underbody_centerline": [0.0, 1.0],
                "rear_slant_centerline": [0.0, 1.0],
            },
            "station_coordinate_spacings": {
                "upper_body_centerline": "uniform",
                "underbody_centerline": "uniform",
                "rear_slant_centerline": "uniform",
            },
            "quantity_ids": ["cp"],
        },
        {
            "id": "velocity_profiles",
            "required": True,
            "allow_unlisted_stations": False,
            "minimum_points": PROFILE_SAMPLE_COUNT,
            "exact_points": PROFILE_SAMPLE_COUNT,
            "coordinate_order": "strictly_increasing",
            "station_ids": [
                "wake_vertical_x_0p25_l",
                "wake_vertical_x_0p50_l",
                "wake_vertical_x_1p00_l",
                "wake_lateral_x_0p50_l_z_0p50_h",
            ],
            "station_sample_counts": {
                "wake_vertical_x_0p25_l": PROFILE_SAMPLE_COUNT,
                "wake_vertical_x_0p50_l": PROFILE_SAMPLE_COUNT,
                "wake_vertical_x_1p00_l": PROFILE_SAMPLE_COUNT,
                "wake_lateral_x_0p50_l_z_0p50_h": PROFILE_SAMPLE_COUNT,
            },
            "station_coordinate_intervals": {
                "wake_vertical_x_0p25_l": [0.0, 2.0],
                "wake_vertical_x_0p50_l": [0.0, 2.0],
                "wake_vertical_x_1p00_l": [0.0, 2.0],
                "wake_lateral_x_0p50_l_z_0p50_h": [-1.0, 1.0],
            },
            "station_coordinate_spacings": {
                "wake_vertical_x_0p25_l": "uniform",
                "wake_vertical_x_0p50_l": "uniform",
                "wake_vertical_x_1p00_l": "uniform",
                "wake_lateral_x_0p50_l_z_0p50_h": "uniform",
            },
            "quantity_ids": ["ux_over_uinf"],
        },
    ]
    spec["splits"] = split_records
    write_json(path, spec)


def update_embedded_leaderboard_manifest() -> None:
    """Keep the repository's generated leaderboard contract projection aligned."""

    specification = json.loads((DATASET_DIR / "submission-spec.json").read_text())
    path = ROOT / "leaderboard" / "manifest.json"
    manifest = json.loads(path.read_text())
    matches = [
        item
        for item in manifest["datasets"]
        if item.get("slug") == "ahmedml"
    ]
    if len(matches) != 1:
        raise ValueError("leaderboard manifest must contain exactly one AhmedML dataset")
    dataset = matches[0]
    dataset["scoring_support"] = specification["scoring_support"]
    dataset["diagnostic_panels"] = [
        {
            "id": "pressure_profiles",
            "title": "Pressure-coefficient profiles",
            "description": (
                "Evaluator-derived Cp traces on frozen native surface-cell mappings."
            ),
            "data_key": "cp_cuts",
            "x_keys": ["x_over_l", "s_over_slant", "x"],
            "quantities": [
                {
                    "id": "cp",
                    "label": "Cp",
                    "y_label": "Pressure coefficient, Cp",
                    "y_keys": ["cp"],
                    "unit": "",
                }
            ],
            "stations": [
                {
                    "id": "upper_body_centerline",
                    "label": "Upper-body centreline",
                    "x_label": "Normalized upper-body coordinate",
                    "description": "Nearest native surface polygons along the upper-body centreline.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
                {
                    "id": "underbody_centerline",
                    "label": "Underbody centreline",
                    "x_label": "x/L",
                    "description": "Nearest native surface polygons along the underbody centreline.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
                {
                    "id": "rear_slant_centerline",
                    "label": "Rear-slant centreline",
                    "x_label": "s/slant",
                    "description": "Nearest native surface polygons along the rear-slant centreline.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
            ],
            "required": True,
            "allow_unlisted_stations": False,
            "required_series_fields": ["case_id", "station_id"],
            "reverse_y": True,
        },
        {
            "id": "velocity_profiles",
            "title": "Wake velocity profiles",
            "description": (
                "Evaluator-derived Ux/Uinf traces on frozen native volume-cell mappings."
            ),
            "data_key": "velocity_profiles",
            "x_keys": ["z_over_h", "y_over_w", "x"],
            "quantities": [
                {
                    "id": "ux_over_uinf",
                    "label": "Ux/Uinf",
                    "y_label": "Streamwise velocity, Ux/Uinf",
                    "y_keys": ["ux_over_uinf", "u_over_u_inf"],
                    "unit": "",
                }
            ],
            "stations": [
                {
                    "id": "wake_vertical_x_0p25_l",
                    "label": "Vertical wake, x/L = 0.25",
                    "x_label": "z/H",
                    "description": "Vertical centreplane wake profile 0.25 body lengths downstream.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
                {
                    "id": "wake_vertical_x_0p50_l",
                    "label": "Vertical wake, x/L = 0.50",
                    "x_label": "z/H",
                    "description": "Vertical centreplane wake profile 0.50 body lengths downstream.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
                {
                    "id": "wake_vertical_x_1p00_l",
                    "label": "Vertical wake, x/L = 1.00",
                    "x_label": "z/H",
                    "description": "Vertical centreplane wake profile 1.00 body lengths downstream.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
                {
                    "id": "wake_lateral_x_0p50_l_z_0p50_h",
                    "label": "Lateral wake, x/L = 0.50",
                    "x_label": "y/W",
                    "description": "Lateral wake profile at x/L = 0.50 and z/H = 0.50 above the body minimum.",
                    "basis": "fluidsbench_candidate_native_mapping",
                },
            ],
            "required": True,
            "allow_unlisted_stations": False,
            "required_series_fields": ["case_id", "station_id"],
        },
    ]
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = dataset_root / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    profile_sha = write_json(DATASET_DIR / "profile-definition-v1.json", profile_definition())
    volume_region_sha = write_json(
        DATASET_DIR / "regional-diagnostics-v1.json", regional_definition()
    )
    regional_v2_path = DATASET_DIR / "regional-diagnostics-v2.json"
    region_sha = write_json(
        regional_v2_path, json.loads(regional_v2_path.read_text(encoding="utf-8"))
    )
    split_records = build_splits(dataset_root, manifest)
    source_document = build_source_identity(dataset_root)
    source_sha = write_json(
        DATASET_DIR
        / "public-source-identity"
        / "ahmedml-public-source-identity-v1.json",
        source_document,
    )
    update_submission_spec(split_records, profile_sha, region_sha, source_sha)
    update_embedded_leaderboard_manifest()
    print(
        json.dumps(
            {
                "dataset_revision": REVISION,
                "source_identity_sha256": source_sha,
                "profile_definition_sha256": profile_sha,
                "regional_definition_sha256": region_sha,
                "legacy_volume_region_definition_sha256": volume_region_sha,
                "split_count": len(split_records),
                "source_case_count": source_document["case_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
