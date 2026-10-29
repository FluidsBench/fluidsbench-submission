"""Normalize WindsorML evaluator evidence into the common schema-v3 envelope."""

from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from reference.windsorml.contract import (
    SOURCE_IDENTITY_SHA256,
    classify_force_replay,
    load_source_identity,
)
from reference.windsorml.dataset_scorer import PROFILE_FAMILY_METRICS, _r2
from reference.windsorml.profiles import load_profile_support, validate_profile_evidence
from scripts.assemble_ahmedml_schema_v3_dev_fixture import (
    AssemblyError,
    _finite,
    _support_counts,
    _support_record,
    _write_json,
)

SURFACE = "windsorml_surface_native_points"
VOLUME = "windsorml_volume_native_cells"
FORCE = "windsorml-force-coefficients-v1"


@lru_cache(maxsize=1)
def source_identity():
    return load_source_identity(
        Path(__file__).resolve().parents[2]
        / "benchmark-specs/windsorml/public-source-identity/windsorml-public-source-identity-v1.json",
        expected_sha256=SOURCE_IDENTITY_SHA256,
    )


def _case_profiles(evidence, case_id):
    try:
        values = validate_profile_evidence(evidence.get("profiles"), case_id=case_id)
        support = load_profile_support(case_id=case_id)
        panels = {f["family_id"]: f["panel_id"] for f in support.definition["families"]}
        series = [
            dict(
                panel_id=panels[family],
                **{
                    k: item[k]
                    for k in ("station_id", "quantity_id", "coordinate", "prediction")
                },
            )
            for family, items in evidence["profiles"]["families"].items()
            for item in items
        ]
        return series, {
            PROFILE_FAMILY_METRICS[f]: _r2(t, p, f) for f, (t, p) in values.items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AssemblyError(f"{case_id}: invalid profile evidence: {error}") from error


def _case_metrics(evidence, *, case_id, counts, profile_metrics):
    case = source_identity().case(case_id)
    supports = []
    for domain, sid, expected_count, association in (
        ("surface", SURFACE, case.surface_entity_count, "PointData"),
        ("volume", VOLUME, case.volume_entity_count, "CellData"),
    ):
        block = evidence[domain]
        if (
            counts[sid] != expected_count
            or block.get("entity_count") != expected_count
            or block.get("association") != association
            or block.get("support_id") != sid
        ):
            raise AssemblyError(
                f"{case_id}: {domain} native support/count/association differs"
            )
        metrics, statistics = {}, {}
        if domain == "surface":
            declarations = [
                (
                    "surface_pressure_rel_l2",
                    ("cpavg",),
                    "physical",
                    "surface_point_dual_area",
                ),
                (
                    "surface_pressure_equal_entity_rel_l2",
                    ("cpavg",),
                    "uniform",
                    "surface_entities_equal",
                ),
                (
                    "surface_wall_shear_rel_l2",
                    ("cfxavg", "cfyavg", "cfzavg"),
                    "physical",
                    "surface_point_dual_area",
                ),
                (
                    "surface_wall_shear_equal_entity_rel_l2",
                    ("cfxavg", "cfyavg", "cfzavg"),
                    "uniform",
                    "surface_entities_equal",
                ),
            ]
        else:
            declarations = [
                (
                    "volume_pressure_rel_l2",
                    ("pressureavg",),
                    "uniform",
                    "volume_cells_equal",
                ),
                (
                    "volume_velocity_rel_l2",
                    ("velocityxavg", "velocityyavg", "velocityzavg"),
                    "uniform",
                    "volume_cells_equal",
                ),
            ]
        for metric, names, weighting, dataset_weighting in declarations:
            fields = [block["fields"][name] for name in names]
            if any(
                f.get("entity_count") != expected_count or f.get("component_count") != 1
                for f in fields
            ):
                raise AssemblyError(
                    f"{case_id}: component coverage differs for {metric}"
                )
            pieces = [f[weighting] for f in fields]
            weights = [_finite(p.get("total_weight"), metric) for p in pieces]
            if any(w <= 0 or w != weights[0] for w in weights) or (
                weighting == "uniform" and weights[0] != expected_count
            ):
                raise AssemblyError(f"{case_id}: component weights differ for {metric}")
            numerator = sum(_finite(p.get("squared_error"), metric) for p in pieces)
            denominator = sum(_finite(p.get("squared_truth"), metric) for p in pieces)
            if numerator < 0 or denominator <= 0:
                raise AssemblyError(f"{case_id}: invalid squared sums for {metric}")
            observed = _finite(block["metrics"].get(metric), metric)
            recomputed = 100 * math.sqrt(numerator / denominator)
            if not math.isclose(observed, recomputed, rel_tol=1e-9, abs_tol=1e-9):
                raise AssemblyError(
                    f"{case_id}: {metric} differs from additive sufficient statistics"
                )
            metrics[metric] = observed
            statistics[metric] = {
                "reduction": "relative_l2_percent",
                "weighting": "uniform" if weighting == "uniform" else "support_weights",
                "dataset_weighting": dataset_weighting,
                "numerator": numerator,
                "denominator": denominator,
                "entity_count": expected_count,
                "total_weight": weights[0],
            }
        supports.append(
            _support_record(
                support_id=sid,
                count=expected_count,
                metric_values=metrics,
                statistics=statistics,
            )
        )
    forces = evidence["forces"]
    if (
        forces.get("axes") != {"drag": "+x", "lift": "+y", "side": "+z"}
        or forces.get("reference_area_m2") != 0.112
    ):
        raise AssemblyError(f"{case_id}: force convention differs")
    truth, prediction = forces["truth_integrated"], forces["prediction_integrated"]
    coefficients = {
        f"{kind}_{axis}": _finite(values.get(axis), f"{case_id}.{kind}.{axis}")
        for kind, values in (("truth", truth), ("prediction", prediction))
        for axis in ("cd", "cl")
    }
    try:
        classify_force_replay(
            case_id=case_id,
            published=case.force_truth,
            replay_cd=coefficients["truth_cd"],
            replay_cl=coefficients["truth_cl"],
        )
    except ValueError as error:
        raise AssemblyError(str(error)) from error
    errors = {
        "c_drag_mae": abs(coefficients["prediction_cd"] - coefficients["truth_cd"]),
        "c_lift_mae": abs(coefficients["prediction_cl"] - coefficients["truth_cl"]),
    }
    supports.append(
        _support_record(support_id=FORCE, count=1, metric_values=errors, statistics={})
    )
    for family in PROFILE_FAMILY_METRICS:
        supports.append(
            _support_record(
                support_id=family, count=counts[family], metric_values={}, statistics={}
            )
        )
    return {
        "case_id": case_id,
        "supports": supports,
        "nonspatial_metric_values": {**errors, **profile_metrics},
        "force_coefficients": coefficients,
    }


def validate_packaged_profiles(cases, metric_values):
    """Recompute all four profile metrics against immutable native truth.

    Standard v1 chunks carry predictions only. Distinct station IDs identify
    the families; coordinates and complete family coverage must match exactly.
    """
    truth = {f: [] for f in PROFILE_FAMILY_METRICS}
    prediction = {f: [] for f in PROFILE_FAMILY_METRICS}
    for case in cases:
        support = load_profile_support(case_id=case["case_id"])
        expected = {}
        for family in support.definition["families"]:
            fid = family["family_id"]
            for station in family["station_ids"]:
                expected[(family["panel_id"], station, family["quantity_id"])] = (
                    fid,
                    support.document["families"][fid][station],
                )
        observed = set()
        for series in case["series"]:
            key = (series["panel_id"], series["station_id"], series["quantity_id"])
            if key in observed or key not in expected:
                raise ValueError(
                    "WindsorML profile series differs from frozen families"
                )
            observed.add(key)
            fid, frozen = expected[key]
            if series["coordinate"] != frozen["coordinate"]:
                raise ValueError(
                    "WindsorML profile coordinates differ from frozen support"
                )
            values = series["prediction"]
            if len(values) != 128 or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in values
            ):
                raise ValueError(
                    "WindsorML profiles require 128 finite predictions per series"
                )
            truth[fid].extend(
                frozen["truth_cp" if key[2] == "cp" else "truth_ux_over_uinf"]
            )
            prediction[fid].extend(values)
        if observed != set(expected):
            raise ValueError(
                "WindsorML requires all sixteen fixed and relative profile series per case"
            )
    for family, metric in PROFILE_FAMILY_METRICS.items():
        computed = _r2(truth[family], prediction[family], metric)
        # Native streamed Float32 truth and frozen support agree to 1e-5.
        # Their profile R2 must agree within the same numerical precision.
        declared = _finite(metric_values.get(metric), metric)
        if not math.isclose(computed, declared, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"WindsorML {metric} differs from packaged predictions and pinned native truth"
            )
