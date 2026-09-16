#!/usr/bin/env python3
"""Prepare the closed HiLiftAeroML all-split native schema-v3 candidate.

This command deliberately separates *candidate readiness* from dataset-owner
activation.  It binds all fourteen exact split labels to eight unique ordered
case sets and builds one FluidsBench scoring-support release from the existing
all-1,800 integrity audits.  The audits are provenance, not a substitute for
immutable public source-file content hashes; that missing binding remains an
explicit gate in the generated benchmark specification.

No participant metric value or prediction is created by this command.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.prepare_hiliftaeroml_split_indices import (
        CAMPAIGN_SHA256 as SPLIT_CAMPAIGN_SHA256,
        SPLIT_BINDINGS,
        build_split_documents,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from prepare_hiliftaeroml_split_indices import (  # type: ignore[no-redef]
        CAMPAIGN_SHA256 as SPLIT_CAMPAIGN_SHA256,
        SPLIT_BINDINGS,
        build_split_documents,
    )


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "benchmark-specs" / "hiliftaeroml"
SPEC_PATH = DATASET_DIR / "submission-spec.json"
RELEASE_ID = "hiliftaeroml-native-all-splits-support-v1-candidate"
RELEASE_DIR = DATASET_DIR / "scoring-support" / RELEASE_ID
DATASET_VERSION = "hiliftaeroml-native-v1-candidate"
EVALUATOR_VERSION = "hiliftaeroml-evaluator-v0.1-candidate"
REGIONAL_CONTRACT_SHA256 = (
    "8cf926d06706b8cc7fd58d821f395bf8cb6565ef1f3aabe6be18e0a279101da4"
)
REGIONAL_CONTRACT_PATH = DATASET_DIR / "regional-diagnostics-v2.json"
LEADERBOARD_MANIFEST_PATH = ROOT / "leaderboard" / "manifest.json"
MANIFEST_URL = (
    "https://github.com/neilashton/fluidsbench-submission/blob/dev/"
    "benchmark-specs/hiliftaeroml/scoring-support/"
    f"{RELEASE_ID}/manifest.json"
)

SCHEMA_MANIFEST = (
    "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json"
)
SCHEMA_CASE_INDEX = (
    "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json"
)
SCHEMA_CASE_CHUNK = (
    "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json"
)

PHYSICAL_WEIGHTINGS = {
    "surface_point_dual_area",
    "volume_point_dual_volume",
}
REMOVED_CANDIDATE_METRIC_IDS = {
    "volume_velocity_physical_rel_l2",
    "volume_pressure_physical_rel_l2",
}


class CandidatePreparationError(ValueError):
    """Raised when retained evidence cannot produce an honest candidate."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CandidatePreparationError(f"cannot encode canonical JSON: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidatePreparationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CandidatePreparationError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CandidatePreparationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CandidatePreparationError(f"{label} is not a lowercase SHA-256")
    return value


def exact_case_sets(campaign: dict[str, Any]) -> dict[str, list[str]]:
    """Return the eight unique sets in first-bound split order."""

    try:
        documents = build_split_documents(campaign)
    except ValueError as error:
        raise CandidatePreparationError(str(error)) from error
    result: dict[str, list[str]] = {}
    for binding in SPLIT_BINDINGS:
        case_ids = documents[binding.split_id]["case_ids"]
        previous = result.setdefault(binding.case_set_id, case_ids)
        if previous != case_ids:
            raise CandidatePreparationError(
                f"shared case set {binding.case_set_id!r} has conflicting order"
            )
    return result


def validate_split_documents(campaign: dict[str, Any]) -> dict[str, str]:
    """Fail closed unless every checked-in split is the canonical expansion."""

    try:
        documents = build_split_documents(campaign)
    except ValueError as error:
        raise CandidatePreparationError(str(error)) from error
    split_sha256s: dict[str, str] = {}
    for binding in SPLIT_BINDINGS:
        path = DATASET_DIR / "splits" / f"{binding.split_id}.json"
        expected = canonical_json_bytes(documents[binding.split_id])
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise CandidatePreparationError(f"cannot read split {path}: {error}") from error
        if observed != expected:
            raise CandidatePreparationError(
                f"split {binding.split_id!r} differs from the pinned campaign"
            )
        split_sha256s[binding.split_id] = hashlib.sha256(observed).hexdigest()
    return split_sha256s


def indexed_cases(
    manifest: dict[str, Any],
    case_ids: Iterable[str],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if manifest.get("case_count") != 1800:
        raise CandidatePreparationError(f"{label} manifest must cover 1,800 cases")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 1800:
        raise CandidatePreparationError(f"{label} manifest cases are incomplete")
    result = {
        case.get("case_id"): case
        for case in raw_cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    if len(result) != len(raw_cases):
        raise CandidatePreparationError(f"{label} manifest case IDs are not unique")
    missing = [case_id for case_id in case_ids if case_id not in result]
    if missing:
        raise CandidatePreparationError(
            f"{label} manifest misses selected evaluation cases: {missing[:5]}"
        )
    return result


def source_summary(record: dict[str, Any], *, label: str) -> dict[str, Any]:
    source = record.get("source")
    if not isinstance(source, dict):
        raise CandidatePreparationError(f"{label} source audit is absent")
    filename = source.get("filename")
    size_bytes = source.get("size_bytes")
    mtime_ns = source.get("mtime_ns")
    content_hashed = source.get("content_sha256_computed")
    if (
        not isinstance(filename, str)
        or not filename
        or not isinstance(size_bytes, int)
        or size_bytes < 1
        or not isinstance(mtime_ns, int)
        or mtime_ns < 1
        or content_hashed is not False
    ):
        raise CandidatePreparationError(f"{label} source audit identity is malformed")
    return {
        "filename": filename,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "content_sha256_computed": False,
    }


def surface_parameters(
    entry: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    if audit.get("status") != "pass" or audit.get("case_id") != entry.get("case_id"):
        raise CandidatePreparationError(
            f"surface audit failed for {entry.get('case_id')!r}"
        )
    dual = audit.get("dual_area")
    if not isinstance(dual, dict):
        raise CandidatePreparationError("surface dual-area audit is absent")
    raw_count = entry.get("raw_point_count")
    if (
        not isinstance(raw_count, int)
        or raw_count < 1
        or entry.get("scored_point_count") != raw_count
        or audit.get("source_number_of_points") != raw_count
    ):
        raise CandidatePreparationError("surface point-count closure failed")
    payload_shape = dual.get("payload_shape")
    if payload_shape != [raw_count]:
        raise CandidatePreparationError("surface dual-area shape differs from support")
    return {
        "association": "PointData",
        "selection": "all_native_boundary_points",
        "raw_point_count": raw_count,
        "scored_point_count": raw_count,
        "source_identity": source_summary(audit, label="surface"),
        "integrity_audit_sha256": require_sha256(
            entry.get("audit_sha256"), "surface audit digest"
        ),
        "dual_area": {
            "algorithm_id": dual.get("algorithm_id"),
            "payload_sha256": require_sha256(
                dual.get("payload_sha256"), "surface dual-area digest"
            ),
            "payload_dtype": dual.get("payload_dtype"),
            "payload_shape": payload_shape,
            "zero_weight_point_count": dual.get("zero_weight_point_count"),
            "negative_weight_point_count": dual.get("negative_weight_point_count"),
            "nonfinite_weight_point_count": dual.get("nonfinite_weight_point_count"),
        },
        "activation_gate": "public_source_vtu_content_sha256_not_yet_pinned",
    }


def volume_parameters(
    entry: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("status") != "complete" or receipt.get("case_id") != entry.get("case_id"):
        raise CandidatePreparationError(
            f"volume receipt failed for {entry.get('case_id')!r}"
        )
    summary = receipt.get("summary")
    if not isinstance(summary, dict) or summary.get("case_id") != entry.get("case_id"):
        raise CandidatePreparationError("volume receipt summary identity differs")
    raw_count = entry.get("raw_point_count")
    scored_count = entry.get("scored_point_count")
    excluded_count = entry.get("excluded_point_count")
    if (
        not all(isinstance(value, int) for value in (raw_count, scored_count, excluded_count))
        or raw_count < 1
        or scored_count < 1
        or excluded_count < 0
        or raw_count != scored_count + excluded_count
        or summary.get("raw_point_count") != raw_count
        or summary.get("scored_point_count") != scored_count
        or summary.get("excluded_point_count") != excluded_count
    ):
        raise CandidatePreparationError("volume retained-point count closure failed")
    return {
        "association": "PointData",
        "selection": "raw_float32_avg(P)_not_equal_to_zero",
        "raw_point_count": raw_count,
        "scored_point_count": scored_count,
        "excluded_point_count": excluded_count,
        "source_identity": source_summary(summary, label="volume"),
        "receipt_sha256": require_sha256(
            entry.get("receipt", {}).get("sha256"), "volume receipt digest"
        ),
        "validity_mask_sha256": require_sha256(
            entry.get("mask_payload_sha256"), "volume validity-mask digest"
        ),
        "mapping_fingerprint": require_sha256(
            entry.get("mapping_fingerprint"), "volume mapping fingerprint"
        ),
        "excluded_raw_ids_sha256": require_sha256(
            entry.get("excluded_raw_ids_sha256"), "volume excluded-ID digest"
        ),
        "zero_fill_all_fields_verified": entry.get(
            "zero_fill_all_fields_verified_in_this_campaign"
        )
        is True,
        "activation_gate": "public_source_vtu_content_sha256_not_yet_pinned",
    }


def metric_binding(metric: dict[str, Any], quantity_id: str) -> dict[str, Any]:
    metric_id = metric["id"]
    if metric_id.endswith("_rel_l1"):
        reduction = "relative_l1_percent"
    elif metric_id.endswith("_rel_l2"):
        reduction = "relative_l2_percent"
    elif metric_id.endswith("_mae"):
        reduction = "mae"
    elif metric_id.endswith("_rmse"):
        reduction = "rmse"
    elif metric_id.endswith("_r2"):
        reduction = "r2"
    else:
        raise CandidatePreparationError(
            f"metric {metric_id!r} has no schema-v3 reduction"
        )
    return {
        "metric_id": metric_id,
        "quantity_id": quantity_id,
        "reduction": reduction,
        "weighting": (
            "support_weights"
            if metric.get("weighting") in PHYSICAL_WEIGHTINGS
            else "uniform"
        ),
        "dataset_weighting": metric["weighting"],
        "aggregation": metric["aggregation"],
        "case_evidence": "aggregate_only" if reduction == "r2" else "metric_value",
    }


def support_definitions(specification: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = {
        metric["id"]: metric
        for metric in specification.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("id"), str)
    }
    surface_ids = [
        metric_id
        for metric_id, metric in metrics.items()
        if metric.get("aggregation") == "per_geometry_then_macro_average"
        and metric_id.startswith("surface_")
    ]
    volume_ids = [
        metric_id
        for metric_id, metric in metrics.items()
        if metric.get("aggregation") == "per_geometry_then_macro_average"
        and metric_id.startswith("volume_")
    ]
    scalar_ids = [
        metric_id
        for metric_id, metric in metrics.items()
        if metric.get("aggregation") == "all_test_cases"
    ]
    required = {
        metric_id
        for metric_id, metric in metrics.items()
        if metric.get("aggregation")
        in {"per_geometry_then_macro_average", "all_test_cases"}
    }
    observed = set(surface_ids) | set(volume_ids) | set(scalar_ids)
    if observed != required:
        raise CandidatePreparationError(
            "candidate support cannot route all field/case metrics; "
            f"missing={sorted(required - observed)}, unexpected={sorted(observed - required)}"
        )

    def surface_quantity(metric_id: str) -> str:
        relative = "_rel_l" in metric_id
        if "wall_shear" in metric_id:
            return "wall_shear_coefficient" if relative else "wall_shear_stress"
        return "pressure_coefficient" if relative else "pressure"

    def volume_quantity(metric_id: str) -> str:
        relative = "_rel_l" in metric_id
        if "velocity" in metric_id:
            return "velocity_over_u_inf" if relative else "velocity"
        return "pressure_coefficient" if relative else "pressure"

    def scalar_quantity(metric_id: str) -> str:
        if metric_id in {"cd_r2", "c_drag_mae"}:
            return "c_drag"
        if metric_id in {"cl_r2", "c_lift_mae"}:
            return "c_lift"
        if metric_id == "c_pitch_mae":
            return "c_pitch"
        raise CandidatePreparationError(f"unknown scalar metric {metric_id!r}")

    common_coverage = {
        "count_fraction": 1.0,
        "weight_fraction": 1.0,
        "unmapped_count": 0,
    }
    return [
        {
            "id": "surface-native-points-v1",
            "domain": "surface",
            "location_definition": {
                "mode": "reference_generator",
                "generator_id": "hiliftaeroml-native-surface-pointdata-v1",
                "generator_version": EVALUATOR_VERSION,
                "parameters": {
                    "association": "PointData",
                    "selection": "all_native_boundary_points",
                    "ordering": "raw_source_point_index_ascending",
                    "source_pattern": "geo_LHC<id>_AoA_<angle>/boundary_geo_LHC<id>_AoA_<angle>.vtu",
                    "candidate_source_content_pinning_complete": False,
                },
                "support_id_rule": {"kind": "source_entity_index"},
                "weight_rule": {
                    "kind": "canonical_rule",
                    "rule_id": "native-point-fan-barycentric-dual-area-v1",
                    "rule_version": "1",
                },
            },
            "quantities": [
                {
                    "id": "pressure_coefficient",
                    "unit": "",
                    "components": [
                        {
                            "id": "pressure_coefficient",
                            "target_field": "(PROJ(AVG(P))-p_inf)/q_inf",
                            "target_association": "generated",
                            "prediction_field": "pressure_coefficient",
                        }
                    ],
                },
                {
                    "id": "pressure",
                    "unit": "Pa",
                    "components": [
                        {
                            "id": "pressure",
                            "target_field": "PROJ(AVG(P))",
                            "target_association": "point_data",
                            "prediction_field": "pressure",
                        }
                    ],
                },
                {
                    "id": "wall_shear_coefficient",
                    "unit": "",
                    "components": [
                        {
                            "id": axis,
                            "target_field": f"AVG(TAU_WALL({index}))/q_inf",
                            "target_association": "generated",
                            "prediction_field": f"tau_wall_coefficient_{axis}",
                        }
                        for index, axis in enumerate(("x", "y", "z"))
                    ],
                },
                {
                    "id": "wall_shear_stress",
                    "unit": "Pa",
                    "components": [
                        {
                            "id": axis,
                            "target_field": f"AVG(TAU_WALL({index}))",
                            "target_association": "point_data",
                            "prediction_field": f"tau_wall_{axis}",
                        }
                        for index, axis in enumerate(("x", "y", "z"))
                    ],
                },
            ],
            "metric_bindings": [
                metric_binding(metrics[metric_id], surface_quantity(metric_id))
                for metric_id in surface_ids
            ],
            "required_coverage": common_coverage,
            "extrapolation_policy": "forbidden",
            "notes": (
                "Primary relative L2 uses released nodal dual area; equal-node "
                "relative L2 is diagnostic. Relative L1/L2 use the frozen Table-5 "
                "nondimensional targets (Cp and tau_wall/q_inf); dimensional MAE/RMSE "
                "use Pa after the per-case q_inf inverse scale. Chunks contribute "
                "additive sums only, followed by one square root after a complete "
                "case. No epsilon is used."
            ),
        },
        {
            "id": "volume-native-valid-points-v1",
            "domain": "volume",
            "location_definition": {
                "mode": "reference_generator",
                "generator_id": "hiliftaeroml-native-volume-pointdata-v1",
                "generator_version": EVALUATOR_VERSION,
                "parameters": {
                    "association": "PointData",
                    "selection": "raw_float32_avg(P)_not_equal_to_zero",
                    "ordering": "retained_raw_source_point_index_ascending",
                    "source_pattern": "geo_LHC<id>_AoA_<angle>/volume_geo_LHC<id>_AoA_<angle>.vtu",
                    "candidate_source_content_pinning_complete": False,
                },
                "support_id_rule": {"kind": "source_entity_index"},
                "weight_rule": {"kind": "uniform"},
            },
            "quantities": [
                {
                    "id": "pressure_coefficient",
                    "unit": "",
                    "components": [
                        {
                            "id": "pressure_coefficient",
                            "target_field": "(avg(P)-p_inf)/q_inf",
                            "target_association": "generated",
                            "prediction_field": "pressure_coefficient",
                        }
                    ],
                },
                {
                    "id": "pressure",
                    "unit": "Pa",
                    "components": [
                        {
                            "id": "pressure",
                            "target_field": "avg(P)",
                            "target_association": "point_data",
                            "prediction_field": "pressure",
                        }
                    ],
                },
                {
                    "id": "velocity_over_u_inf",
                    "unit": "",
                    "components": [
                        {
                            "id": axis,
                            "target_field": f"avg(u)[{index}]/|U_inf|",
                            "target_association": "generated",
                            "prediction_field": f"velocity_over_u_inf_{axis}",
                        }
                        for index, axis in enumerate(("x", "y", "z"))
                    ],
                },
                {
                    "id": "velocity",
                    "unit": "m/s",
                    "components": [
                        {
                            "id": axis,
                            "target_field": f"avg(u)[{index}]",
                            "target_association": "point_data",
                            "prediction_field": f"velocity_{axis}",
                        }
                        for index, axis in enumerate(("x", "y", "z"))
                    ],
                },
            ],
            "metric_bindings": [
                metric_binding(metrics[metric_id], volume_quantity(metric_id))
                for metric_id in volume_ids
            ],
            "required_coverage": common_coverage,
            "extrapolation_policy": "forbidden",
            "notes": (
                "Primary relative L2 weights every retained valid native PointData "
                "node equally. Relative L1/L2 use the frozen Table-5 nondimensional "
                "targets ((P-p_inf)/q_inf and U/|U_inf|); dimensional MAE/RMSE use "
                "Pa or m/s after the per-case inverse scale. Chunks contribute additive "
                "sums only, followed by one square root after a complete case. No "
                "epsilon is used."
            ),
        },
        {
            "id": "aerodynamic-case-coefficients-v1",
            "domain": "scalar_case",
            "location_definition": {
                "mode": "reference_generator",
                "generator_id": "hiliftaeroml-native-surface-loads-v1",
                "generator_version": EVALUATOR_VERSION,
                "parameters": {
                    "derivation": "canonical_force_and_moment_integration_from_native_surface_fields",
                    "candidate_source_content_pinning_complete": False,
                },
                "support_id_rule": {
                    "kind": "canonical_rule",
                    "rule_id": "one-coefficient-row-per-case-v1",
                    "rule_version": "1",
                },
                "weight_rule": {"kind": "uniform"},
            },
            "quantities": [
                {
                    "id": quantity,
                    "unit": "",
                    "components": [
                        {
                            "id": quantity,
                            "target_field": f"truth_{quantity}",
                            "target_association": "generated",
                            "prediction_field": f"predicted_{quantity}",
                        }
                    ],
                }
                for quantity in ("c_drag", "c_lift", "c_pitch")
            ],
            "metric_bindings": [
                metric_binding(metrics[metric_id], scalar_quantity(metric_id))
                for metric_id in scalar_ids
            ],
            "required_coverage": common_coverage,
            "extrapolation_policy": "forbidden",
            "notes": (
                "Dataset-level R2 uses every case in the selected split; scalar "
                "MAE weights selected cases equally."
            ),
        },
    ]


def provenance_document(
    case_set_id: str,
    case_ids: list[str],
    campaign_path: Path,
    surface_manifest_path: Path,
    surface_manifest: dict[str, Any],
    volume_manifest_path: Path,
    volume_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "hiliftaeroml-all-splits-native-candidate-provenance-v1",
        "schema_version": 1,
        "status": "candidate_provenance_not_public_source_content_pin",
        "dataset_id": "hiliftaeroml",
        "case_set_id": case_set_id,
        "case_set_sha256": hashlib.sha256(
            ("\n".join(case_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "case_count": len(case_ids),
        "inputs": {
            "campaign": {
                "filename": campaign_path.name,
                "sha256": sha256_file(campaign_path),
            },
            "surface_all1800_manifest": {
                "filename": surface_manifest_path.name,
                "schema_id": surface_manifest.get("schema_id"),
                "case_count": surface_manifest.get("case_count"),
                "case_set_sha256": surface_manifest.get("case_set_sha256"),
                "sha256": sha256_file(surface_manifest_path),
            },
            "volume_all1800_manifest": {
                "filename": volume_manifest_path.name,
                "schema_id": volume_manifest.get("schema_id"),
                "case_count": volume_manifest.get("case_count"),
                "case_set_sha256": volume_manifest.get("case_set_sha256"),
                "sha256": sha256_file(volume_manifest_path),
            },
        },
        "retained_scope": (
            "Only per-case counts, mapping/weight payload digests, receipt/audit "
            "digests, and filename/size/mtime source identities are retained in "
            "the support chunks. Private absolute paths are discarded."
        ),
        "activation_gate": (
            "The original public boundary and volume source members were bound by "
            "filename, byte size and mtime, not content SHA-256. Their immutable "
            "archive/member content hashes must be published and matched before "
            "dataset-owner approval or opening submissions."
        ),
    }


def build_release(
    specification: dict[str, Any],
    case_sets: dict[str, list[str]],
    surface_entries: dict[str, dict[str, Any]],
    volume_entries: dict[str, dict[str, Any]],
    *,
    campaign_path: Path,
    surface_manifest_path: Path,
    surface_manifest: dict[str, Any],
    volume_manifest_path: Path,
    volume_manifest: dict[str, Any],
    published_at: str,
) -> str:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    case_set_descriptors: list[dict[str, Any]] = []
    verified_surface: dict[str, dict[str, Any]] = {}
    verified_volume: dict[str, dict[str, Any]] = {}
    for case_set_id, case_ids in case_sets.items():
        case_directory = RELEASE_DIR / "case-sets" / case_set_id
        case_directory.mkdir(parents=True, exist_ok=True)
        provenance = provenance_document(
            case_set_id,
            case_ids,
            campaign_path,
            surface_manifest_path,
            surface_manifest,
            volume_manifest_path,
            volume_manifest,
        )
        provenance_path = case_directory / "provenance.json"
        provenance_sha256 = write_json(provenance_path, provenance)
        shared_artifact = {
            "role": "candidate-provenance",
            "path": "provenance.json",
            "sha256": provenance_sha256,
            "byte_size": provenance_path.stat().st_size,
            "format": "json",
        }

        chunk_descriptors: list[dict[str, Any]] = []
        for chunk_number, start in enumerate(range(0, len(case_ids), 90)):
            selected = case_ids[start : start + 90]
            cases = []
            for case_id in selected:
                if case_id not in verified_surface:
                    surface_entry = surface_entries[case_id]
                    surface_audit_path = Path(surface_entry["audit_path"])
                    if sha256_file(surface_audit_path) != surface_entry.get(
                        "audit_sha256"
                    ):
                        raise CandidatePreparationError(
                            f"surface audit bytes changed for {case_id}"
                        )
                    verified_surface[case_id] = surface_parameters(
                        surface_entry, read_json(surface_audit_path)
                    )
                if case_id not in verified_volume:
                    volume_entry = volume_entries[case_id]
                    volume_receipt_path = Path(volume_entry["receipt"]["path"])
                    if sha256_file(volume_receipt_path) != volume_entry[
                        "receipt"
                    ].get("sha256"):
                        raise CandidatePreparationError(
                            f"volume receipt bytes changed for {case_id}"
                        )
                    verified_volume[case_id] = volume_parameters(
                        volume_entry, read_json(volume_receipt_path)
                    )
                surface = verified_surface[case_id]
                volume = verified_volume[case_id]
                cases.append(
                    {
                        "case_id": case_id,
                        "support_instances": [
                            {
                                "support_id": "surface-native-points-v1",
                                "entity_count": surface["scored_point_count"],
                                "artifacts": [dict(shared_artifact)],
                                "parameters": surface,
                            },
                            {
                                "support_id": "volume-native-valid-points-v1",
                                "entity_count": volume["scored_point_count"],
                                "artifacts": [dict(shared_artifact)],
                                "parameters": volume,
                            },
                            {
                                "support_id": "aerodynamic-case-coefficients-v1",
                                "entity_count": 1,
                                "artifacts": [dict(shared_artifact)],
                                "parameters": {
                                    "case_id": case_id,
                                    "status": "candidate_surface_load_support",
                                    "activation_gate": (
                                        "public_source_vtu_content_sha256_not_yet_pinned"
                                    ),
                                },
                            },
                        ],
                    }
                )
            chunk = {
                "$schema": SCHEMA_CASE_CHUNK,
                "schema_version": "1.0",
                "release_id": RELEASE_ID,
                "dataset_id": "hiliftaeroml",
                "case_set_id": case_set_id,
                "cases": cases,
            }
            filename = f"chunk-{chunk_number:03d}.json"
            chunk_path = case_directory / filename
            chunk_sha256 = write_json(chunk_path, chunk)
            chunk_descriptors.append(
                {
                    "file": filename,
                    "sha256": chunk_sha256,
                    "case_count": len(selected),
                    "case_ids": selected,
                }
            )

        case_index = {
            "$schema": SCHEMA_CASE_INDEX,
            "schema_version": "1.0",
            "release_id": RELEASE_ID,
            "dataset_id": "hiliftaeroml",
            "case_set_id": case_set_id,
            "case_count": len(case_ids),
            "chunks": chunk_descriptors,
        }
        index_path = case_directory / "index.json"
        index_sha256 = write_json(index_path, case_index)
        case_set_descriptors.append(
            {
                "id": case_set_id,
                "case_count": len(case_ids),
                "index_file": f"case-sets/{case_set_id}/index.json",
                "index_sha256": index_sha256,
            }
        )

    manifest = {
        "$schema": SCHEMA_MANIFEST,
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "status": "candidate",
        "published_at": published_at,
        "dataset_id": "hiliftaeroml",
        "dataset_version": DATASET_VERSION,
        "evaluation_reference_version": EVALUATOR_VERSION,
        "coordinate_frame": {
            "id": "hiliftaeroml-native-body-frame-v1",
            "axis_order": ["x", "y", "z"],
            "handedness": "right",
            "length_unit": "in",
            "normalization": (
                "none for scoring; checkpoint normalization is model-owned and "
                "does not change native support identity"
            ),
        },
        "supports": support_definitions(specification),
        "case_sets": case_set_descriptors,
        "notes": (
            "Closed owner-review candidate. Per-case provenance is filtered from "
            "the complete all-1,800 internal integrity manifests; no private path "
            "is part of the participant contract. Public source archive/member "
            "content pinning remains mandatory before activation."
        ),
    }
    return write_json(RELEASE_DIR / "manifest.json", manifest)


def update_specification(
    specification: dict[str, Any],
    *,
    split_sha256s: dict[str, str],
    manifest_sha256: str,
) -> None:
    specification["metrics"] = [
        metric
        for metric in specification.get("metrics", [])
        if isinstance(metric, dict)
        and metric.get("id") not in REMOVED_CANDIDATE_METRIC_IDS
    ]
    metrics_by_id = {
        metric.get("id"): metric
        for metric in specification["metrics"]
        if isinstance(metric.get("id"), str)
    }
    cp_metric = metrics_by_id.get("cp_cut_r2")
    velocity_profile_metric = metrics_by_id.get("velocity_profile_r2")
    if not isinstance(cp_metric, dict) or not isinstance(velocity_profile_metric, dict):
        raise CandidatePreparationError("required profile metrics are absent")
    cp_metric.update(
        {
            "equation": (
                "1-\\frac{\\sum_g\\sum_i w_{g,i}(y_{g,i}-\\hat{y}_{g,i})^2}"
                "{\\sum_g\\sum_i w_{g,i}(y_{g,i}-\\bar{y}_g)^2}"
            ),
            "aggregation": "per_case_connected_graph_centered_then_macro_average",
            "weighting": "physical_connected_cut_arc_length",
        }
    )
    velocity_profile_metric.update(
        {
            "equation": (
                "1-\\frac{\\sum_s\\sum_i w_{s,i}(y_{s,i}-\\hat{y}_{s,i})^2}"
                "{\\sum_s\\sum_i w_{s,i}(y_{s,i}-\\bar{y}_s)^2}"
            ),
            "aggregation": "per_case_station_centered_then_macro_average",
            "weighting": "physical_polyline_arc_length",
        }
    )
    panels = {
        panel.get("id"): panel
        for panel in specification.get("profile_panels", [])
        if isinstance(panel, dict) and isinstance(panel.get("id"), str)
    }
    pressure_panel = panels.get("pressure_profiles")
    velocity_panel = panels.get("velocity_profiles")
    if not isinstance(pressure_panel, dict) or not isinstance(velocity_panel, dict):
        raise CandidatePreparationError("required profile panels are absent")
    pressure_station_ids = [f"pressure_belt_{letter}" for letter in "abcdefghij"]
    pressure_panel.update(
        {
            "station_ids": pressure_station_ids,
            "source_station_map": {
                station_id: letter
                for station_id, letter in zip(pressure_station_ids, "ABCDEFGHIJ")
            },
            "weighting": "physical_connected_cut_arc_length",
            "r2_protocol": (
                "center truth independently within each physical connected cut "
                "graph, pool graph SSE/SST across rows A-J within a case, then "
                "macro-average complete-case R2 values equally"
            ),
            "serialization_gate": (
                "freeze a lossless disconnected-graph profile-chunk mapping and "
                "matching public profile-ground-truth release before activation"
            ),
        }
    )
    velocity_station_map = {
        "hlpw5_b_2": "B.2",
        "hlpw5_b_3": "B.3",
        "hlpw5_c_1": "C.1",
        "hlpw5_c_2": "C.2",
        "hlpw5_c_3": "C.3",
    }
    velocity_panel.update(
        {
            "station_ids": list(velocity_station_map),
            "source_station_map": velocity_station_map,
            "quantity_source": "velocity_magnitude/|U_inf|",
            "weighting": "physical_polyline_arc_length",
            "r2_protocol": (
                "center truth within each of B.2/B.3/C.1/C.2/C.3, pool station "
                "SSE/SST within a case, then macro-average complete-case R2 values equally"
            ),
        }
    )
    specification["dataset_version"] = DATASET_VERSION
    specification["status"] = "owner_review_required"
    specification["evaluation_reference_version"] = EVALUATOR_VERSION
    split_descriptors = specification.get("splits")
    expected_split_ids = [binding.split_id for binding in SPLIT_BINDINGS]
    if not isinstance(split_descriptors, list) or [
        split.get("id") if isinstance(split, dict) else None
        for split in split_descriptors
    ] != expected_split_ids:
        raise CandidatePreparationError(
            "submission specification does not contain the fourteen exact splits"
        )
    binding_by_id = {binding.split_id: binding for binding in SPLIT_BINDINGS}
    for descriptor in split_descriptors:
        binding = binding_by_id[descriptor["id"]]
        descriptor.update(
            {
                "label": binding.split_label,
                "index_file": f"splits/{binding.split_id}.json",
                "case_count": binding.case_count,
                "case_set_id": binding.case_set_id,
                "case_id_status": "official",
                "sha256": split_sha256s[binding.split_id],
            }
        )
    old_support = specification.get("scoring_support")
    if not isinstance(old_support, dict):
        raise CandidatePreparationError("submission specification scoring_support is absent")
    public_supports = copy.deepcopy(old_support.get("public_supports"))
    if not isinstance(public_supports, list):
        raise CandidatePreparationError("public_supports must be a list")
    volume_public = next(
        (
            support
            for support in public_supports
            if isinstance(support, dict) and support.get("id") == "flow-domain-native"
        ),
        None,
    )
    if volume_public is None:
        raise CandidatePreparationError("flow-domain-native public support is absent")
    volume_public.pop("physical_weight", None)
    volume_public["entities"] = (
        "retained_native_volume_points_with_raw_float32_avg(P)_not_equal_to_zero"
    )
    volume_public["primary_weight"] = "one_per_retained_valid_native_point"
    relative_l2_policy = copy.deepcopy(old_support.get("relative_l2_policy"))
    if not isinstance(relative_l2_policy, dict):
        raise CandidatePreparationError("relative_l2_policy must be an object")
    relative_l2_policy["flow_domain_primary"] = "equal_valid_native_entity"
    relative_l2_policy["flow_domain_secondary"] = "not_part_of_closed_candidate"
    relative_l2_policy["point_or_vertex_physical_weights"] = (
        "surface_only_deterministic_mass_lumped_dual_area_from_the_exact_public_mesh"
    )
    specification["scoring_support"] = {
        "status": "owner_review_required",
        "submissions_open": False,
        "closed_reason": (
            "Closed owner-review candidate: immutable public source-content pins, "
            "owner scientific approval, and an immutable evaluator revision are pending."
        ),
        "candidate_manifest": {
            "status": "candidate",
            "release_id": RELEASE_ID,
            "manifest_file": f"scoring-support/{RELEASE_ID}/manifest.json",
            "manifest_url": MANIFEST_URL,
            "manifest_sha256": manifest_sha256,
        },
        "dataset_evaluator_binding": {
            "status": "pending_frozen_release",
            "repository_url": "https://github.com/neilashton/fluidsbench-submission",
            "evaluator_reference_version": EVALUATOR_VERSION,
            "evaluator_code_revision": None,
            "activation_rule": (
                "Publish and owner-approve one immutable evaluator Git revision; "
                "participant evidence must bind that exact revision."
            ),
        },
        "coverage_contract": old_support.get("coverage_contract"),
        "public_supports": public_supports,
        "relative_l2_policy": relative_l2_policy,
        "field_metric_value_bases": {
            "relative_l1_l2": {
                "surface_pressure": "(P-p_inf)/q_inf",
                "surface_wall_shear": "tau_wall/q_inf",
                "volume_pressure": "(P-p_inf)/q_inf",
                "volume_velocity": "U/|U_inf|",
                "aggregation": (
                    "form each complete-case ratio from additive sufficient "
                    "statistics, then macro-average all cases in the selected "
                    "split equally"
                ),
            },
            "dimensional_mae_rmse": {
                "surface_pressure": "inverse-scale each case by q_inf into Pa",
                "surface_wall_shear": "inverse-scale each case by q_inf into Pa",
                "volume_pressure": "inverse-scale each case by q_inf into Pa",
                "volume_velocity": "inverse-scale each case by |U_inf| into m/s",
                "aggregation": (
                    "derive each dimensional complete-case value before "
                    "macro-averaging all cases in the selected split equally"
                ),
            },
            "vector_reduction": {
                "relative_l1": "component-flattened absolute numerator and denominator",
                "relative_l2": (
                    "one nodal weight multiplies the squared three-component "
                    "vector error and truth magnitude"
                ),
                "mae_rmse": (
                    "current native-evaluator per-component convention over 3*N "
                    "weighted scalar components"
                ),
            },
        },
        "candidate_migration_notes": [
            (
                "The closed schema-v3 candidate removes the legacy diagnostic "
                "volume_velocity_physical_rel_l2 and "
                "volume_pressure_physical_rel_l2 requirements. The native "
                "Table-5 evaluator has never published or consumed volume "
                "dual-volume support; its frozen primary contract is equal valid "
                "native nodes. This candidate migration is not an activated public "
                "scoring change because submissions remain closed and unapproved."
            ),
            (
                "Relative field metrics are bound to the frozen Table-5 "
                "nondimensional targets, while the existing paper-compatible "
                "MAE/RMSE metrics remain dimensional. A candidate aggregate must "
                "emit and audit both bases from the same per-case inference stream."
            ),
            (
                "The closed candidate replaces the stale prototype 16-station "
                "velocity panel with the five frozen Table-5 stations B.2, B.3, "
                "C.1, C.2, and C.3. Velocity profiles use physical polyline "
                "arc length and within-station centering; Cp retains exact rows "
                "A-J with physical connected-cut arc length and independently "
                "centered connected graphs. This is not an activated public "
                "profile-contract change while submissions remain closed."
            )
        ],
        "owner_decisions_required": [
            "pin_exact_public_release_archive_and_extracted_member_content_sha256_for_all_1355_unique_evaluation_cases",
            "confirm_public_source_array_names_associations_retained_volume_mask_and_raw_point_order",
            "publish_and_approve_authoritative_surface_dual_area_support",
            "approve_all_exact_split_bindings_and_candidate_scoring_support_release",
            "exercise_and_audit_same_stream_nondimensional_relative_and_dimensional_absolute_field_aggregation",
            "freeze_lossless_cp_connected_graph_and_five_station_velocity_profile_serialization_and_public_truth_release",
            "freeze_and_approve_the_immutable_dataset_evaluator_git_revision",
            "approve_the_zero_weight_regional_contract_and_complete_split_aggregator",
        ],
    }
    write_json(SPEC_PATH, specification)


def update_leaderboard_manifest(specification: dict[str, Any]) -> None:
    manifest = read_json(LEADERBOARD_MANIFEST_PATH)
    matches = [
        dataset
        for dataset in manifest.get("datasets", [])
        if isinstance(dataset, dict) and dataset.get("slug") == "hiliftaeroml"
    ]
    if len(matches) != 1:
        raise CandidatePreparationError(
            "leaderboard manifest must define HiLiftAeroML exactly once"
        )
    dataset = matches[0]
    dataset["metric_ids"] = [metric["id"] for metric in specification["metrics"]]
    dataset["scoring_support"] = copy.deepcopy(specification["scoring_support"])
    dataset["ranking"] = copy.deepcopy(specification["ranking"])
    write_json(LEADERBOARD_MANIFEST_PATH, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-json", type=Path, required=True)
    parser.add_argument("--surface-manifest", type=Path, required=True)
    parser.add_argument("--volume-manifest", type=Path, required=True)
    parser.add_argument(
        "--published-at",
        default="2026-08-31T00:00:00Z",
        help="Candidate manifest timestamp; fixed by default for reproducible output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in (
        (args.campaign_json, "campaign"),
        (args.surface_manifest, "surface manifest"),
        (args.volume_manifest, "volume manifest"),
    ):
        if not path.is_file():
            raise CandidatePreparationError(f"{label} does not exist: {path}")
    if sha256_file(args.campaign_json) != SPLIT_CAMPAIGN_SHA256:
        raise CandidatePreparationError("Table-5 campaign SHA-256 differs")
    if sha256_file(REGIONAL_CONTRACT_PATH) != REGIONAL_CONTRACT_SHA256:
        raise CandidatePreparationError("regional contract SHA-256 differs")

    campaign = read_json(args.campaign_json)
    case_sets = exact_case_sets(campaign)
    split_sha256s = validate_split_documents(campaign)
    unique_case_ids = list(
        dict.fromkeys(
            case_id for case_ids in case_sets.values() for case_id in case_ids
        )
    )
    surface_manifest = read_json(args.surface_manifest)
    volume_manifest = read_json(args.volume_manifest)
    surface_entries = indexed_cases(
        surface_manifest, unique_case_ids, label="surface all1800"
    )
    volume_entries = indexed_cases(
        volume_manifest, unique_case_ids, label="volume all1800"
    )
    specification = read_json(SPEC_PATH)
    specification["metrics"] = [
        metric
        for metric in specification.get("metrics", [])
        if isinstance(metric, dict)
        and metric.get("id") not in REMOVED_CANDIDATE_METRIC_IDS
    ]

    manifest_sha256 = build_release(
        specification,
        case_sets,
        surface_entries,
        volume_entries,
        campaign_path=args.campaign_json,
        surface_manifest_path=args.surface_manifest,
        surface_manifest=surface_manifest,
        volume_manifest_path=args.volume_manifest,
        volume_manifest=volume_manifest,
        published_at=args.published_at,
    )
    update_specification(
        specification,
        split_sha256s=split_sha256s,
        manifest_sha256=manifest_sha256,
    )
    update_leaderboard_manifest(specification)
    print(
        json.dumps(
            {
                "split_count": len(SPLIT_BINDINGS),
                "case_set_count": len(case_sets),
                "unique_evaluation_case_count": len(unique_case_ids),
                "case_sets": [
                    {
                        "case_set_id": case_set_id,
                        "case_count": len(case_ids),
                        "case_set_sha256": hashlib.sha256(
                            ("\n".join(case_ids) + "\n").encode("utf-8")
                        ).hexdigest(),
                    }
                    for case_set_id, case_ids in case_sets.items()
                ],
                "split_sha256s": split_sha256s,
                "release_id": RELEASE_ID,
                "manifest_sha256": manifest_sha256,
                "regional_contract_sha256": REGIONAL_CONTRACT_SHA256,
                "submissions_open": False,
                "owner_approved": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
