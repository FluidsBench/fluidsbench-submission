#!/usr/bin/env python3
"""Publish compact metadata for the generated AhmedML candidate support.

The large NPY evaluator arrays remain outside Git.  This publisher validates
every one of the 316 unique local case-support artifacts, copies their compact
hash-bound JSON manifests, and emits the five official test-case-set indices
used by the eight AhmedML split families.  The result remains an owner-review
candidate and does not open submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "benchmark-specs" / "ahmedml"
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    FORCE_ABSOLUTE_TOLERANCE,
    PROFILE_DEFINITION_SHA256,
    REPOSITORY_REVISION,
    SOURCE_IDENTITY_SHA256,
    AhmedMLContractError,
    classify_force_replay,
    load_source_identity,
    sha256_file,
)
from reference.ahmedml.support import (  # noqa: E402
    load_case_support,
    load_profile_support,
)


RELEASE_ID = "ahmedml-native-all-splits-support-v1-candidate"
DATASET_VERSION = "ahmedml-native-v1-candidate"
EVALUATOR_VERSION = "ahmedml-evaluator-v0.1-candidate"
CASE_SET_SPLITS = {
    "full-test": "full",
    "geometry-test": "geometry",
    "high_drag-test": "high_drag",
    "low_drag-test": "low_drag",
    "image_wake-test": "image_wake",
}
_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)


class SupportPublicationError(ValueError):
    """Raised when compact candidate support cannot be published."""


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SupportPublicationError(f"{label} must be a finite number")
    return float(value)


def _validate_case_audits(support: Any) -> None:
    """Reject a generated case whose build-time physical audits did not close."""

    audits = support.audits
    expected = {
        "vtk_version",
        "surface_bounds_m",
        "volume_bounds_m",
        "surface_area",
        "force_truth",
        "volume_cell_volume",
        "volume_regions",
        "profile_mapping",
    }
    if set(audits) != expected or audits.get("vtk_version") != "9.5.2":
        raise SupportPublicationError(f"{support.case_id} build audits are incomplete")
    for name in ("surface_bounds_m", "volume_bounds_m"):
        bounds = audits.get(name)
        if not isinstance(bounds, list) or len(bounds) != 6:
            raise SupportPublicationError(f"{support.case_id} {name} differs")
        for index, value in enumerate(bounds):
            _finite_number(value, f"{support.case_id}.{name}[{index}]")

    area = audits.get("surface_area")
    if not isinstance(area, dict):
        raise SupportPublicationError(f"{support.case_id} surface-area audit differs")
    relative_tolerance = _finite_number(
        area.get("relative_tolerance"),
        f"{support.case_id}.surface_area.relative_tolerance",
    )
    relative_difference = _finite_number(
        area.get("maximum_relative_magnitude_difference"),
        f"{support.case_id}.surface_area.maximum_relative_magnitude_difference",
    )
    if (
        relative_tolerance != 2e-6
        or relative_difference < 0.0
        or relative_difference > relative_tolerance
        or _finite_number(
            area.get("calculated_sum_m2"),
            f"{support.case_id}.surface_area.calculated_sum_m2",
        )
        <= 0.0
        or _finite_number(
            area.get("published_sum_m2"),
            f"{support.case_id}.surface_area.published_sum_m2",
        )
        <= 0.0
    ):
        raise SupportPublicationError(f"{support.case_id} surface-area replay failed")

    force = audits.get("force_truth")
    if not isinstance(force, dict):
        raise SupportPublicationError(f"{support.case_id} force audit differs")
    force_tolerance = _finite_number(
        force.get("absolute_tolerance"),
        f"{support.case_id}.force_truth.absolute_tolerance",
    )
    if force_tolerance != FORCE_ABSOLUTE_TOLERANCE:
        raise SupportPublicationError(f"{support.case_id} force tolerance differs")
    published_cd = _finite_number(
        force.get("published_cd"), f"{support.case_id}.force_truth.published_cd"
    )
    published_cl = _finite_number(
        force.get("published_cl"), f"{support.case_id}.force_truth.published_cl"
    )
    replay_cd = _finite_number(
        force.get("field_replay_cd"),
        f"{support.case_id}.force_truth.field_replay_cd",
    )
    replay_cl = _finite_number(
        force.get("field_replay_cl"),
        f"{support.case_id}.force_truth.field_replay_cl",
    )
    try:
        exception = classify_force_replay(
            case_id=support.case_id,
            published_cd=published_cd,
            published_cl=published_cl,
            field_replay_cd=replay_cd,
            field_replay_cl=replay_cl,
        )
    except AhmedMLContractError as error:
        raise SupportPublicationError(str(error)) from error
    exception_record = force.get("source_exception")
    if exception is None:
        if exception_record is not None:
            raise SupportPublicationError(
                f"{support.case_id} declares an unexpected force-replay exception"
            )
    elif exception_record != {
        "exception_id": exception.exception_id,
        "scope": "published_force_csv_replay_only",
        "scoring_truth_source": "native_surface_field_integration",
        "published_csv_used_for_scoring": False,
        "reason": exception.reason,
    }:
        raise SupportPublicationError(
            f"{support.case_id} force-replay exception declaration differs"
        )

    cell_volume = audits.get("volume_cell_volume")
    if (
        not isinstance(cell_volume, dict)
        or cell_volume.get("storage_dtype") != "<f4"
        or _finite_number(
            cell_volume.get("sum_before_float32_m3"),
            f"{support.case_id}.volume_cell_volume.sum_before_float32_m3",
        )
        <= 0.0
        or _finite_number(
            cell_volume.get("sum_after_float32_m3"),
            f"{support.case_id}.volume_cell_volume.sum_after_float32_m3",
        )
        <= 0.0
        or _finite_number(
            cell_volume.get("minimum_m3"),
            f"{support.case_id}.volume_cell_volume.minimum_m3",
        )
        <= 0.0
        or _finite_number(
            cell_volume.get("maximum_m3"),
            f"{support.case_id}.volume_cell_volume.maximum_m3",
        )
        <= 0.0
    ):
        raise SupportPublicationError(f"{support.case_id} cell-volume audit failed")

    regions = audits.get("volume_regions")
    region_ids = ("near_body", "wake", "farfield")
    if (
        not isinstance(regions, dict)
        or set(regions) != {*region_ids, "sum"}
        or any(
            isinstance(regions.get(region_id), bool)
            or not isinstance(regions.get(region_id), int)
            or regions[region_id] <= 0
            for region_id in region_ids
        )
        or regions.get("sum") != sum(regions[region_id] for region_id in region_ids)
        or regions.get("sum") != support.volume_entity_count
    ):
        raise SupportPublicationError(f"{support.case_id} regional partition failed")

    profile = audits.get("profile_mapping")
    if (
        not isinstance(profile, dict)
        or profile.get("runtime_geometric_remapping_forbidden") is not True
        or profile.get("volume_nearest_cell_fallback_count") != 0
        or _finite_number(
            profile.get("surface_maximum_projection_distance_m"),
            f"{support.case_id}.profile_mapping.surface_maximum_projection_distance_m",
        )
        < 0.0
        or _finite_number(
            profile.get("volume_maximum_projection_distance_m"),
            f"{support.case_id}.profile_mapping.volume_maximum_projection_distance_m",
        )
        != 0.0
    ):
        raise SupportPublicationError(f"{support.case_id} profile mapping audit failed")


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _artifact(manifest_path: Path, relative_path: str) -> dict[str, object]:
    return {
        "role": "candidate-case-support-manifest",
        "path": relative_path,
        "sha256": sha256_file(manifest_path),
        "byte_size": manifest_path.stat().st_size,
        "format": "json",
    }


def _support_instances(
    document: dict[str, Any],
    artifact: dict[str, object],
) -> list[dict[str, object]]:
    counts = document["entity_counts"]
    source = document["source"]
    definitions = document["definitions"]
    manifest_sha = str(artifact["sha256"])
    shared = {
        "source_identity_sha256": source["source_identity_sha256"],
        "profile_definition_sha256": definitions["profile_definition_sha256"],
        "regional_definition_sha256": definitions["regional_definition_sha256"],
        "case_support_manifest_sha256": manifest_sha,
        "large_array_distribution": "maintainer_local_candidate_not_published",
    }
    boundary = source["boundary"]
    volume = source["volume"]
    boundary_artifact = {
        "role": "public-boundary-vtp",
        "url": (
            "https://huggingface.co/datasets/neashton/ahmedml/resolve/"
            f"{REPOSITORY_REVISION}/{boundary['path']}"
        ),
        "sha256": boundary["sha256"],
        "byte_size": boundary["size_bytes"],
        "format": "vtk_xml_polydata",
    }
    volume_artifact = {
        "role": "public-volume-vtu",
        "url": (
            "https://huggingface.co/datasets/neashton/ahmedml/resolve/"
            f"{REPOSITORY_REVISION}/{volume['path']}"
        ),
        "sha256": volume["sha256"],
        "byte_size": volume["size_bytes"],
        "format": "vtk_xml_unstructured_grid",
    }
    return [
        {
            "support_id": "ahmedml-surface-native-cells-v1",
            "entity_count": counts["surface"],
            "artifacts": [artifact, boundary_artifact],
            "parameters": {
                **shared,
                "association": "CellData",
                "selection": "all_native_boundary_polygons",
                "source_sha256": source["boundary"]["sha256"],
                "surface_area_vector_sha256": document["artifacts"][
                    "surface_area_vector"
                ]["sha256"],
            },
            "expected_support_sha256": manifest_sha,
        },
        {
            "support_id": "ahmedml-volume-native-cells-v1",
            "entity_count": counts["volume"],
            "artifacts": [artifact, volume_artifact],
            "parameters": {
                **shared,
                "association": "CellData",
                "selection": "all_native_volume_cells",
                "source_sha256": source["volume"]["sha256"],
                "cell_volume_sha256": document["artifacts"]["volume_cell_volume"][
                    "sha256"
                ],
                "region_code_sha256": document["artifacts"]["volume_region_code"][
                    "sha256"
                ],
            },
            "expected_support_sha256": manifest_sha,
        },
        {
            "support_id": "ahmedml-force-coefficients-v1",
            "entity_count": 1,
            "artifacts": [artifact],
            "parameters": {
                **shared,
                "integration": "pMean*A_outward-wallShearStressMean*area",
                "force_truth_sha256": source["force_coefficients"]["sha256"],
            },
            "expected_support_sha256": manifest_sha,
        },
        {
            "support_id": "ahmedml-surface-cp-profiles-v1",
            "entity_count": 3 * 128,
            "artifacts": [artifact],
            "parameters": {
                **shared,
                "station_count": 3,
                "samples_per_station": 128,
                "mapping_and_truth_sha256": document["artifacts"][
                    "profile_mapping_and_truth"
                ]["sha256"],
            },
            "expected_support_sha256": manifest_sha,
        },
        {
            "support_id": "ahmedml-volume-velocity-profiles-v1",
            "entity_count": 4 * 128,
            "artifacts": [artifact],
            "parameters": {
                **shared,
                "station_count": 4,
                "samples_per_station": 128,
                "mapping_and_truth_sha256": document["artifacts"][
                    "profile_mapping_and_truth"
                ]["sha256"],
            },
            "expected_support_sha256": manifest_sha,
        },
    ]


def _surface_support() -> dict[str, object]:
    components = lambda name, count: [  # noqa: E731 - compact declarative helper.
        {
            "id": component,
            "target_field": name,
            "target_association": "cell_data",
            "prediction_field": name,
        }
        for component in (("value",) if count == 1 else ("x", "y", "z"))
    ]
    bindings: list[dict[str, object]] = []
    for metric_id, quantity_id, reduction, weighting, dataset_weighting in (
        ("surface_pressure_rel_l2", "pmean", "relative_l2_percent", "support_weights", "surface_face_area"),
        ("surface_pressure_equal_entity_rel_l2", "pmean", "relative_l2_percent", "uniform", "surface_entities_equal"),
        ("surface_wall_shear_rel_l2", "wall_shear", "relative_l2_percent", "support_weights", "surface_face_area"),
        ("surface_wall_shear_equal_entity_rel_l2", "wall_shear", "relative_l2_percent", "uniform", "surface_entities_equal"),
        ("surface_pressure_rel_l1", "pmean", "relative_l1_percent", "support_weights", "surface_face_area"),
        ("surface_wall_shear_rel_l1", "wall_shear", "relative_l1_percent", "support_weights", "surface_face_area"),
        ("surface_pressure_mae", "pmean", "mae", "support_weights", "surface_face_area"),
        ("surface_pressure_rmse", "pmean", "rmse", "support_weights", "surface_face_area"),
        ("surface_wall_shear_mae", "wall_shear", "mae", "support_weights", "surface_face_area"),
        ("surface_wall_shear_rmse", "wall_shear", "rmse", "support_weights", "surface_face_area"),
    ):
        bindings.append(
            {
                "metric_id": metric_id,
                "quantity_id": quantity_id,
                "reduction": reduction,
                "weighting": weighting,
                "dataset_weighting": dataset_weighting,
                "aggregation": "per_geometry_then_macro_average",
                "case_evidence": "metric_value",
            }
        )
    return {
        "id": "ahmedml-surface-native-cells-v1",
        "domain": "surface",
        "location_definition": {
            "mode": "native_entities",
            "format": "vtk_xml_polydata",
            "artifact_role": "public-boundary-vtp",
            "entity": "surface_face",
            "selection": {"kind": "all_entities"},
            "coordinate_rule": "face_centres_v1",
            "ordering": "source_entity_index",
            "support_id_rule": {"kind": "source_entity_index"},
            "weight_rule": {
                "kind": "geometric",
                "measure": "face_area",
                "rule_id": "native-polygon-area-v1",
                "rule_version": "1",
            },
        },
        "quantities": [
            {"id": "pmean", "unit": "m2/s2", "components": components("pMean", 1)},
            {
                "id": "wall_shear",
                "unit": "m2/s2",
                "components": components("wallShearStressMean", 3),
            },
        ],
        "metric_bindings": bindings,
        "required_coverage": {
            "count_fraction": 1.0,
            "weight_fraction": 1.0,
            "unmapped_count": 0,
        },
        "extrapolation_policy": "forbidden",
        "notes": "Every native boundary polygon is scored in source CellData order.",
    }


def _volume_support() -> dict[str, object]:
    def components(name: str, count: int) -> list[dict[str, object]]:
        labels = ("value",) if count == 1 else ("x", "y", "z")
        return [
            {
                "id": component,
                "target_field": name,
                "target_association": "cell_data",
                "prediction_field": name,
            }
            for component in labels
        ]

    bindings: list[dict[str, object]] = []
    for metric_id, quantity_id, reduction, weighting, dataset_weighting in (
        ("volume_pressure_rel_l2", "pmean", "relative_l2_percent", "uniform", "volume_cells_equal"),
        ("volume_pressure_physical_rel_l2", "pmean", "relative_l2_percent", "support_weights", "cell_volume"),
        ("volume_velocity_rel_l2", "umean", "relative_l2_percent", "uniform", "volume_cells_equal"),
        ("volume_velocity_physical_rel_l2", "umean", "relative_l2_percent", "support_weights", "cell_volume"),
        ("volume_pressure_rel_l1", "pmean", "relative_l1_percent", "uniform", "volume_cells_equal"),
        ("volume_velocity_rel_l1", "umean", "relative_l1_percent", "uniform", "volume_cells_equal"),
        ("volume_pressure_mae", "pmean", "mae", "uniform", "volume_cells_equal"),
        ("volume_pressure_rmse", "pmean", "rmse", "uniform", "volume_cells_equal"),
        ("volume_velocity_mae", "umean", "mae", "uniform", "volume_cells_equal"),
        ("volume_velocity_rmse", "umean", "rmse", "uniform", "volume_cells_equal"),
    ):
        bindings.append(
            {
                "metric_id": metric_id,
                "quantity_id": quantity_id,
                "reduction": reduction,
                "weighting": weighting,
                "dataset_weighting": dataset_weighting,
                "aggregation": "per_geometry_then_macro_average",
                "case_evidence": "metric_value",
            }
        )
    return {
        "id": "ahmedml-volume-native-cells-v1",
        "domain": "volume",
        "location_definition": {
            "mode": "native_entities",
            "format": "vtk_xml_unstructured_grid",
            "artifact_role": "public-volume-vtu",
            "entity": "cell",
            "selection": {"kind": "all_entities"},
            "coordinate_rule": "cell_centres_v1",
            "ordering": "source_entity_index",
            "support_id_rule": {"kind": "source_entity_index"},
            "weight_rule": {
                "kind": "geometric",
                "measure": "cell_volume",
                "rule_id": "vtk-cell-size-filter-v1",
                "rule_version": "9.5.2",
            },
        },
        "quantities": [
            {"id": "pmean", "unit": "m2/s2", "components": components("pMean", 1)},
            {"id": "umean", "unit": "m/s", "components": components("UMean", 3)},
        ],
        "metric_bindings": bindings,
        "required_coverage": {
            "count_fraction": 1.0,
            "weight_fraction": 1.0,
            "unmapped_count": 0,
        },
        "extrapolation_policy": "forbidden",
        "notes": "Equal-native-cell relative L2 is primary; cell-volume weighting is secondary.",
    }


def _derived_supports() -> list[dict[str, object]]:
    def derived(
        support_id: str,
        domain: str,
        generator_id: str,
        quantities: list[dict[str, object]],
        bindings: list[dict[str, object]],
        notes: str,
    ) -> dict[str, object]:
        return {
            "id": support_id,
            "domain": domain,
            "location_definition": {
                "mode": "reference_generator",
                "generator_id": generator_id,
                "generator_version": EVALUATOR_VERSION,
                "parameters": {
                    "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
                    "source_identity_sha256": SOURCE_IDENTITY_SHA256,
                },
                "support_id_rule": {"kind": "source_entity_index"},
                "weight_rule": {"kind": "uniform"},
            },
            "quantities": quantities,
            "metric_bindings": bindings,
            "required_coverage": {
                "count_fraction": 1.0,
                "weight_fraction": 1.0,
                "unmapped_count": 0,
            },
            "extrapolation_policy": "forbidden",
            "notes": notes,
        }

    def aggregate_binding(
        metric_id: str,
        quantity_id: str,
        reduction: str,
        *,
        aggregation: str = "all_test_cases",
        dataset_weighting: str = "cases_equal",
    ) -> dict[str, object]:
        return {
            "metric_id": metric_id,
            "quantity_id": quantity_id,
            "reduction": reduction,
            "weighting": "uniform",
            "dataset_weighting": dataset_weighting,
            "aggregation": aggregation,
            "case_evidence": "aggregate_only" if reduction == "r2" else "metric_value",
        }

    scalar = lambda field: [  # noqa: E731 - compact declarative helper.
        {
            "id": "value",
            "target_field": field,
            "target_association": "generated",
            "prediction_field": field,
        }
    ]
    return [
        derived(
            "ahmedml-force-coefficients-v1",
            "scalar_case",
            "ahmedml-surface-force-integration-v1",
            [
                {"id": "cd", "unit": "", "components": scalar("cd")},
                {"id": "cl", "unit": "", "components": scalar("cl")},
            ],
            [
                aggregate_binding("cd_r2", "cd", "r2"),
                aggregate_binding("cl_r2", "cl", "r2"),
                aggregate_binding("c_drag_mae", "cd", "mae"),
                aggregate_binding("c_lift_mae", "cl", "mae"),
            ],
            "Cd and Cl are integrated from the same complete native surface predictions.",
        ),
        derived(
            "ahmedml-surface-cp-profiles-v1",
            "line",
            "ahmedml-frozen-surface-profile-mapping-v1",
            [{"id": "cp", "unit": "", "components": scalar("Cp")}],
            [
                aggregate_binding(
                    "cp_cut_r2",
                    "cp",
                    "r2",
                    aggregation="flatten_all_required_profile_samples",
                    dataset_weighting="samples_equal",
                )
            ],
            "Three evaluator-owned 128-sample Cp cuts; participant profile payloads are forbidden.",
        ),
        derived(
            "ahmedml-volume-velocity-profiles-v1",
            "line",
            "ahmedml-frozen-volume-profile-mapping-v1",
            [
                {
                    "id": "ux_over_uinf",
                    "unit": "",
                    "components": scalar("Ux_over_Uinf"),
                }
            ],
            [
                aggregate_binding(
                    "velocity_profile_r2",
                    "ux_over_uinf",
                    "r2",
                    aggregation="flatten_all_required_profile_samples",
                    dataset_weighting="samples_equal",
                )
            ],
            "Four evaluator-owned 128-sample wake profiles; participant profile payloads are forbidden.",
        ),
    ]


def publish(
    *,
    support_root: Path,
    output: Path,
    source_identity_path: Path,
    published_at: str,
    chunk_cases: int,
) -> Path:
    if _TIMESTAMP_RE.fullmatch(published_at) is None:
        raise SupportPublicationError("published_at must be UTC YYYY-MM-DDTHH:MM:SSZ")
    if chunk_cases < 1 or chunk_cases > 100:
        raise SupportPublicationError("chunk_cases must lie in [1, 100]")
    output = output.expanduser().resolve()
    if output.exists():
        raise SupportPublicationError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-partial-", dir=output.parent))
    identity = load_source_identity(source_identity_path)
    validated: set[str] = set()
    case_sets: list[dict[str, object]] = []
    for case_set_id, split_id in CASE_SET_SPLITS.items():
        split = json.loads(
            (ROOT / "benchmark-specs" / "ahmedml" / "splits" / f"{split_id}.json").read_text(
                encoding="utf-8"
            )
        )
        if split.get("case_set_id") != case_set_id:
            raise SupportPublicationError(f"{split_id} case_set_id differs")
        case_ids = split["case_ids"]
        case_set_root = temporary / "case-sets" / case_set_id
        chunks: list[dict[str, object]] = []
        for chunk_index, offset in enumerate(range(0, len(case_ids), chunk_cases)):
            selected = case_ids[offset : offset + chunk_cases]
            cases: list[dict[str, object]] = []
            for case_id in selected:
                source_manifest = support_root / case_id / "case-support.json"
                support = load_case_support(source_manifest, source_identity=identity)
                destination_relative = (
                    Path("case-artifacts") / case_id / "case-support.json"
                )
                destination = case_set_root / destination_relative
                if case_id not in validated:
                    _validate_case_audits(support)
                    # This compact NPZ is the only generated payload published to
                    # the browser later.  Verify its hash, member contract, exact
                    # 128-point grids, and finite truth arrays before accepting
                    # the case manifest.  The much larger field-weight arrays
                    # remain local and are re-verified when the evaluator opens
                    # them for scoring.
                    load_profile_support(support)
                    validated.add(case_id)
                # Generic scoring-support paths are deliberately confined to
                # the directory containing each case-set chunk. Duplicate
                # these small manifests across overlapping case sets instead
                # of introducing an escaping ../../ reference.
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_manifest, destination)
                artifact = _artifact(source_manifest, destination_relative.as_posix())
                document = json.loads(source_manifest.read_text(encoding="utf-8"))
                if support.manifest_sha256 != artifact["sha256"]:
                    raise SupportPublicationError(f"{case_id} support digest changed")
                cases.append(
                    {
                        "case_id": case_id,
                        "support_instances": _support_instances(document, artifact),
                    }
                )
            chunk_name = f"chunk-{chunk_index:03d}.json"
            chunk_sha = _write_json(
                case_set_root / chunk_name,
                {
                    "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
                    "schema_version": "1.0",
                    "release_id": RELEASE_ID,
                    "dataset_id": "ahmedml",
                    "case_set_id": case_set_id,
                    "cases": cases,
                },
            )
            chunks.append(
                {
                    "file": chunk_name,
                    "sha256": chunk_sha,
                    "case_count": len(selected),
                    "case_ids": selected,
                }
            )
        index_path = case_set_root / "index.json"
        index_sha = _write_json(
            index_path,
            {
                "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
                "schema_version": "1.0",
                "release_id": RELEASE_ID,
                "dataset_id": "ahmedml",
                "case_set_id": case_set_id,
                "case_count": len(case_ids),
                "chunks": chunks,
            },
        )
        case_sets.append(
            {
                "id": case_set_id,
                "case_count": len(case_ids),
                "index_file": f"case-sets/{case_set_id}/index.json",
                "index_sha256": index_sha,
            }
        )
    if len(validated) != 316:
        raise SupportPublicationError(
            f"expected 316 unique candidate support cases, validated {len(validated)}"
        )
    manifest = {
        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "status": "candidate",
        "published_at": published_at,
        "dataset_id": "ahmedml",
        "dataset_version": DATASET_VERSION,
        "evaluation_reference_version": EVALUATOR_VERSION,
        "coordinate_frame": {
            "id": "ahmedml-native-body-frame-v1",
            "axis_order": ["x", "y", "z"],
            "handedness": "right",
            "length_unit": "m",
            "normalization": "none for scoring; Cp and profile coordinates use declared derived transforms",
        },
        "supports": [_surface_support(), _volume_support(), *_derived_supports()],
        "case_sets": case_sets,
        "notes": (
            "Closed owner-review candidate. Public source bytes are pinned exactly. Compact "
            "per-case manifests bind all generated array hashes, while the approximately "
            "40 GB large-array payload remains maintainer-local until owner review. The "
            "run_492 force CSV/native-field discrepancy is one checksum- and value-bound "
            "source exception; native surface-field integration remains force truth."
        ),
    }
    _write_json(temporary / "manifest.json", manifest)
    os.rename(temporary, output)
    print(output / "manifest.json")
    return output


def bind_candidate_manifest(
    *,
    specification_path: Path,
    manifest_path: Path,
) -> dict[str, str]:
    """Bind an emitted candidate release into the benchmark specification."""

    specification_path = specification_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    try:
        relative = manifest_path.relative_to(specification_path.parent)
    except ValueError as error:
        raise SupportPublicationError(
            "candidate support manifest must remain inside the AhmedML specification directory"
        ) from error
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    support = specification.get("scoring_support")
    if (
        specification.get("dataset_id") != "ahmedml"
        or not isinstance(support, dict)
        or support.get("status") != "owner_review_required"
        or support.get("submissions_open") is not False
    ):
        raise SupportPublicationError(
            "submission specification is not the closed AhmedML owner-review contract"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("release_id") != RELEASE_ID
        or manifest.get("dataset_id") != "ahmedml"
        or manifest.get("status") != "candidate"
    ):
        raise SupportPublicationError("published manifest identity differs")
    binding = {
        "status": "candidate",
        "release_id": RELEASE_ID,
        "manifest_file": relative.as_posix(),
        "manifest_url": (
            "https://github.com/neilashton/fluidsbench-submission/blob/dev/"
            f"benchmark-specs/ahmedml/{relative.as_posix()}"
        ),
        "manifest_sha256": sha256_file(manifest_path),
    }
    support["candidate_manifest"] = binding
    write_path = specification_path.with_name(
        f".{specification_path.name}.partial-{os.getpid()}"
    )
    try:
        _write_json(write_path, specification)
        os.replace(write_path, specification_path)
    finally:
        write_path.unlink(missing_ok=True)
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--chunk-cases", type=int, default=25)
    parser.add_argument(
        "--bind-submission-specification",
        action="store_true",
        help="atomically add the emitted candidate manifest binding to submission-spec.json",
    )
    parser.add_argument(
        "--submission-specification",
        type=Path,
        default=DATASET_DIR / "submission-spec.json",
    )
    parser.add_argument(
        "--source-identity",
        type=Path,
        default=(
            ROOT
            / "benchmark-specs"
            / "ahmedml"
            / "public-source-identity"
            / "ahmedml-public-source-identity-v1.json"
        ),
    )
    args = parser.parse_args()
    try:
        published = publish(
            support_root=args.support_root.expanduser().resolve(),
            output=args.output,
            source_identity_path=args.source_identity,
            published_at=args.published_at,
            chunk_cases=args.chunk_cases,
        )
        if args.bind_submission_specification:
            binding = bind_candidate_manifest(
                specification_path=args.submission_specification,
                manifest_path=published / "manifest.json",
            )
            print(json.dumps({"candidate_manifest": binding}, indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
