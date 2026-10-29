"""Complete, non-approving reductions for the DrivAerML development workflow."""

from __future__ import annotations

import math
import json
import csv
from pathlib import Path

from reference.drivaerml.dev_profiles import (
    SCHEMA as PROFILE_EVIDENCE_SCHEMA,
    INDEX_SHA256,
    compare_profiles,
    truth_case,
    WORKFLOW_VERSION,
    validate_scoring_contract,
)
from reference.scores import composite_overall_score, composite_component_group_scores

CP = "drivaerml_cp_constant_v1"
VELOCITY = "drivaerml-autocfd5-constant-v1"


def unavailable_metrics(specification):
    return {
        m["id"]
        for m in specification["metrics"]
        if m["id"].startswith(("volume_", "drivaerml_volume_", "velocity_profile_"))
    }


def profile_metrics(case_id, series, scope):
    statistics, rmses = compare_profiles(case_id, series, scope)
    mean = lambda xs: math.fsum(xs) / len(xs)
    values = {"cp_cut_rmse": mean([v for _, v in rmses[CP]])}
    if scope == "surface_and_volume":
        values["velocity_profile_uinf_rmse"] = mean([v for _, v in rmses[VELOCITY]])
        experimental = {
            "autocfd5_v1",
            "autocfd5_v2",
            "autocfd5_v3",
            "autocfd5_v5",
            "autocfd5_u1",
            "autocfd5_u2",
            "autocfd5_u3",
            "autocfd5_u4",
            "autocfd5_u5",
            "autocfd5_u6",
            "autocfd5_l1",
        }
        values["velocity_profile_experimental_subset_uinf_rmse"] = mean(
            [v for station, v in rmses[VELOCITY] if station in experimental]
        )
    return values, statistics


def validate_diagnostic_case(document, path, digest, *, contract, pinned, core):
    from reference.drivaerml.dataset_scorer import (
        _DiagnosticCase,
        DrivAerDatasetScorerError,
    )

    if (
        set(document)
        != {
            "schema",
            "case_id",
            "prediction_scope",
            "native_truth_index_sha256",
            "prediction_inputs",
            "profile_series",
        }
        or document["schema"] != PROFILE_EVIDENCE_SCHEMA
        or document["case_id"] != core.case_id
        or document["prediction_scope"] != core.prediction_scope
        or document["native_truth_index_sha256"] != INDEX_SHA256
    ):
        raise DrivAerDatasetScorerError(
            "native development profile evidence identity differs"
        )
    inputs = document["prediction_inputs"]
    expected = {
        "surface_native_cells": (
            core.surface_prediction_manifest_sha256,
            core.surface_prediction_chunk_sha256,
        )
    }
    if core.prediction_scope == "surface_and_volume":
        expected["volume_native_cells"] = (
            core.volume_prediction_manifest_sha256,
            core.volume_prediction_chunk_sha256,
        )
    if set(inputs) != set(expected):
        raise DrivAerDatasetScorerError(
            "native profile prediction supports differ from core evidence"
        )
    for support, (manifest_sha, chunk_shas) in expected.items():
        binding = inputs[support]
        audit = binding.get("audit", {})
        if (
            binding.get("manifest_sha256") != manifest_sha
            or audit.get("manifest_sha256") != manifest_sha
            or tuple(audit.get("chunk_sha256", [])) != tuple(chunk_shas)
        ):
            raise DrivAerDatasetScorerError(
                "profile predictions differ from field-evaluation manifests/chunks"
            )
    try:
        values, _ = profile_metrics(
            core.case_id, document["profile_series"], core.prediction_scope
        )
    except (ValueError, KeyError, TypeError) as error:
        raise DrivAerDatasetScorerError(str(error)) from error
    for metric in (
        "velocity_profile_uinf_rmse",
        "velocity_profile_experimental_subset_uinf_rmse",
    ):
        values.setdefault(metric, None)
    return _DiagnosticCase(
        core.case_id,
        Path(path).name,
        digest,
        values,
        {
            k: (() if v is not None else ({"reason": "not_in_surface_only_scope"},))
            for k, v in values.items()
        },
        None,
        None,
        tuple(document["profile_series"]),
    )


def r2(truth, prediction):
    mean = math.fsum(truth) / len(truth)
    variance = math.fsum((v - mean) ** 2 for v in truth)
    if len(truth) < 2 or variance <= 0:
        raise ValueError("R2 requires nonzero truth variance across the complete split")
    return (
        1
        - math.fsum((p - t) ** 2 for t, p in zip(truth, prediction, strict=True))
        / variance
    )


def complete_metric_values(evidence, specification):
    validate_scoring_contract(specification)
    cases = evidence["cases"]
    values = evidence["metric_values"]
    scope = evidence["prediction_scope"]
    for metric, coefficient in (
        ("cd_r2", "Cd"),
        ("cl_r2", "Cl"),
        ("c_pitch_r2", "CmPitch"),
    ):
        values[metric] = r2(
            [c["force"]["truth"][coefficient] for c in cases],
            [c["force"]["prediction"][coefficient] for c in cases],
        )
    families = {CP: "cp_cut_r2"}
    if scope == "surface_and_volume":
        families[VELOCITY] = "velocity_profile_r2"
    statistics = {family: [] for family in families}
    for case in cases:
        _, per_family = profile_metrics(case["case_id"], case["profile_series"], scope)
        for family in families:
            # Every line/cut has unit weight; every case contains exactly the
            # frozen complete family, so no case or long line dominates.
            statistics[family].extend(per_family[family])
    for family, metric in families.items():
        stats = statistics[family]
        error, first, second = [
            math.fsum(s[i] for s in stats) / len(stats) for i in range(3)
        ]
        variance = second - first**2
        if variance <= 0:
            raise ValueError(f"{metric} has zero native truth variance")
        values[metric] = 1 - error / variance
    values = {k: v for k, v in values.items() if v is not None}
    fixed_zero = (
        tuple(
            specification["overall_score_composite"]["surface_only_policy"][
                "unavailable_component_metric_ids"
            ]
        )
        if scope == "surface_only"
        else ()
    )
    values["overall_score"] = composite_overall_score(
        values,
        specification["overall_score_composite"],
        fixed_zero_component_ids=fixed_zero,
    )
    values.update(
        composite_component_group_scores(
            values,
            specification["overall_score_composite"],
            specification["component_score_groups"],
            fixed_zero_component_ids=fixed_zero,
        )
    )
    required = {m["id"] for m in specification["metrics"]}
    if scope == "surface_only":
        required -= unavailable_metrics(specification)
    if set(values) != required:
        raise ValueError(
            f"incomplete dev metrics: missing={sorted(required-set(values))}, unexpected={sorted(set(values)-required)}"
        )
    evidence["metric_values"] = values
    evidence["dev_workflow_version"] = WORKFLOW_VERSION
    cp_reduction = evidence.get("metric_reductions", {}).get("cp_cut_rmse")
    if cp_reduction is not None:
        cp_reduction["support_scope"]["within_cut_weighting"] = "trapezoidal_arc_length"
        cp_reduction["support_scope"]["support_status"] = "published_immutable_native_v3"
    evidence["native_profile_truth_index_sha256"] = INDEX_SHA256
    evidence["eligibility"][
        "reason"
    ] = "Development candidate only; official release gates remain closed"
    evidence["claims"].update(
        overall_score_computed=True,
        field_force_or_diagnostic_component_score_computed=True,
    )
    evidence["eligibility"].update(
        composite_score_available=True, component_scores_available=True
    )


def validate_package_metrics(directory, submission, specification):
    """Recompute the package's force/profile values from immutable truth."""
    validate_scoring_contract(specification)
    from reference.drivaerml.dev_profiles import sha256
    from reference.drivaerml.dataset_scorer import OFFICIAL_FORCE_TRUTH_SHA256

    source = (
        Path(__file__).resolve().parents[2]
        / "benchmark-specs/drivaerml/reference-data/force_mom_constref_all.csv"
    )
    if sha256(source) != OFFICIAL_FORCE_TRUTH_SHA256:
        raise ValueError("retained force truth digest differs")
    with source.open() as handle:
        truth = {
            "run_" + row["run"]: {k: float(v) for k, v in row.items() if k != "run"}
            for row in csv.DictReader(handle)
        }
    profile_index = directory / submission["profile_data"]["index_file"]
    index = json.loads(profile_index.read_text())
    profiles = {}
    for binding in index["chunks"]:
        for case in json.loads((profile_index.parent / binding["file"]).read_text())[
            "cases"
        ]:
            if case["case_id"] in profiles:
                raise ValueError("duplicate profile case")
            profiles[case["case_id"]] = case["series"]
    metrics = json.loads((directory / submission["case_metrics"]["file"]).read_text())
    scope = submission["prediction_scope"]
    cases, diagnostics, force_errors, closures = [], {}, {}, []
    for case in metrics["cases"]:
        cid = case["case_id"]
        t, p = truth[cid], case["force_coefficients"]
        native = {
            "Cd": t["cd"],
            "Cl": t["cl"],
            "CmPitch": (t["clf"] - t["clr"]) / 2,
            "Clf": t["clf"],
            "Clr": t["clr"],
        }
        predicted = {
            "Cd": p["cd"],
            "Cl": p["cl"],
            "CmPitch": p["cm_pitch"],
            "Clf": p["clf"],
            "Clr": p["clr"],
        }
        for coefficient, expected in (
            ("clf", p["cl"] / 2 + p["cm_pitch"]),
            ("clr", p["cl"] / 2 - p["cm_pitch"]),
        ):
            if not math.isclose(p[coefficient], expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(
                    f"{cid}: force coefficients violate lift/moment closure"
                )
        force_supports = [
            s
            for s in case["supports"]
            if s["support_id"] == "drivaerml-force-coefficients-v1"
        ]
        if len(force_supports) != 1:
            raise ValueError(f"{cid}: expected one force-coefficient support")
        force_support = force_supports[0]
        for metric, value in force_support["metric_values"].items():
            if not math.isclose(
                value,
                case["nonspatial_metric_values"][metric],
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"{cid}: force support differs from nonspatial evidence"
                )
        for key, metric in (
            ("Cd", "cd"),
            ("Cl", "cl"),
            ("CmPitch", "cmpitch"),
            ("Clf", "clf"),
            ("Clr", "clr"),
        ):
            error = predicted[key] - native[key]
            metric = "field_integrated_" + metric + "_rmse"
            force_errors.setdefault(metric, []).append(error)
            if not math.isclose(
                case["nonspatial_metric_values"][metric],
                abs(error),
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{cid}: force error differs from native truth")
        closure = abs(p["cl"] - p["clf"] - p["clr"])
        if not math.isclose(
            case["nonspatial_metric_values"]["field_integrated_lift_closure_max_abs"],
            closure,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{cid}: lift closure differs from force coefficients")
        closures.append(closure)
        values, _ = profile_metrics(cid, profiles[cid], scope)
        for metric, value in values.items():
            diagnostics.setdefault(metric, []).append(value)
            if not math.isclose(
                case["nonspatial_metric_values"].get(metric, float("nan")),
                value,
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"{cid}: {metric} differs from native profile predictions"
                )
        cases.append(
            {
                "case_id": cid,
                "profile_series": profiles[cid],
                "force": {"truth": native, "prediction": predicted},
            }
        )
    if set(profiles) != {c["case_id"] for c in cases}:
        raise ValueError("profile cases differ from metric cases")
    expected = dict(submission["metric_values"])
    expected.update(
        {
            k: math.sqrt(math.fsum(v * v for v in vs) / len(vs))
            for k, vs in force_errors.items()
        }
    )
    expected.update({k: math.fsum(vs) / len(vs) for k, vs in diagnostics.items()})
    expected["field_integrated_lift_closure_max_abs"] = max(closures)
    evidence = {
        "cases": cases,
        "metric_values": expected,
        "prediction_scope": scope,
        "claims": {},
        "eligibility": {},
    }
    complete_metric_values(evidence, specification)
    for metric, value in evidence["metric_values"].items():
        if not math.isclose(
            submission["metric_values"][metric], value, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(f"{metric} differs from independently recomputed values")
