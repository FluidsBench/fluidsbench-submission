"""Synthetic full-split packages test dev intake, never publish model results."""

import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from reference.drivaerml import dataset_scorer as scorer
from reference.drivaerml import dev_profiles as profiles
from reference.drivaerml.dev_workflow import (
    complete_metric_values,
    profile_metrics,
    validate_package_metrics,
)
from reference.drivaerml.regional_diagnostics import (
    RegionalFieldAccumulator,
    RegionAssignmentHasher,
    SURFACE_REGION_DEFINITION,
    VOLUME_REGION_DEFINITION,
    REGIONAL_FIELD_CONTRACTS,
    build_regional_report,
)
from reference.drivaerml.regional_aggregate import (
    build_case_regional_envelope,
    aggregate_regional_diagnostics,
)
from scripts import assemble_drivaerml_schema_v3_candidate as assembler
from scripts import validate_submission as validator, manage_leaderboard as manager
from scripts import register_drivaerml_pre_release as registrar
from scripts.assemble_ahmedml_schema_v3_dev_fixture import (
    _case_representation,
    _case_mapping,
    _representation_summary,
    _mapping_summary,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "benchmark-specs/drivaerml/submission-spec.json"


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def region_fixture(cid, scope):
    supports, sums = {}, {}
    audit = None
    for definition, fields in (
        (SURFACE_REGION_DEFINITION, ["surface_pressure", "surface_wall_shear"]),
        (VOLUME_REGION_DEFINITION, ["volume_pressure", "volume_velocity"]),
    ):
        if scope == "surface_only" and definition is VOLUME_REGION_DEFINITION:
            continue
        codes = np.arange(4, dtype=np.uint8)
        hasher = RegionAssignmentHasher(definition, 4)
        hasher.add_chunk(0, codes)
        assignment = hasher.finalize()
        field_records = {}
        for field in fields:
            vector = field.endswith(("velocity", "shear"))
            contract = REGIONAL_FIELD_CONTRACTS[field]
            acc = RegionalFieldAccumulator(
                definition,
                4,
                contract["component_labels"],
                velocity_diagnostics=field == "volume_velocity",
            )
            truth = np.ones((4, 3) if vector else 4)
            acc.add_chunk(0, codes, truth, truth + 0.01, np.ones(4))
            difference = truth + 0.01 - truth
            squared = difference**2 if not vector else (difference**2).sum(axis=1)
            expected = {
                "entity_count": 4,
                "total_weight": 4.0,
                "absolute_error": float(np.sqrt(squared).sum()),
                "squared_error": float(squared.sum()),
                "squared_truth": float((truth**2).sum()),
            }
            stats = acc.finalize(expected_uniform=expected, expected_physical=expected)
            field_records[field] = {
                k: contract[k] for k in ("quantity", "unit", "primary_weighting")
            }
            field_records[field]["statistics"] = stats
            sums[field] = {
                w: stats["reconstruction"]["reconstructed_" + w]
                for w in ["uniform", "physical"]
            }
        supports[definition.definition_id] = {
            "definition": definition.to_json(),
            "definition_sha256": definition.sha256,
            "assignment": assignment,
            "fields": field_records,
        }
        if definition is VOLUME_REGION_DEFINITION:
            audit = {
                "entity_count": 4,
                "all_cells_assigned_exactly_once": True,
                "coordinate_identity_sha256": "a" * 64,
                "assignment_identity_sha256": assignment["sha256"],
                "region_labels": list(definition.region_ids),
                "region_entity_count": dict.fromkeys(definition.region_ids, 1),
            }
    report = build_regional_report(case_id=cid, supports=supports)
    return build_case_regional_envelope(
        case_id=cid,
        case_report=report,
        prediction_scope=scope,
        volume_geometry_audit=audit,
        official_additive_sums=sums,
    )


def inputs(root, scope):
    spec = load(SPEC)
    split = load(SPEC.parent / "splits/full.json")
    ids = split["case_ids"]
    config = load(ROOT / "examples/drivaerml-v3-candidate/package-config.template.json")
    original = load(ROOT / "examples/v3-template/submission.json")
    config["participant"] = {
        k: original[k] for k in config["participant"] if k in original
    }
    p = config["participant"]
    p["submission_id"] = "drivaerml-workflow-test-v1"
    p["result_revision"]["series_id"] = "drivaerml-workflow-test"
    p["model"] = "Synthetic integration test; no trained model"
    p["code_url"] = p["reproducibility"]["code"]["repository_url"]
    p["note"] = "Disposable test evidence only; never published."
    method = p["methodology"]
    method["record_kind"] = "submitter_reported"
    fields = load(SPEC.parent / "methodology-contract.json")[
        "required_predicted_fields"
    ]
    method["architecture"]["predicted_fields"] = [
        {
            **f,
            "component_ids": ["coordinate-network"],
            "production": "direct_model_output",
        }
        for f in fields
        if scope != "surface_only" or f.get("domain") != "volume"
    ]
    method["inference_compute"]["case_count"] = len(ids)
    checkpoint = root / "test-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic test checkpoint, not real weights")
    method["checkpoints"][0]["sha256"] = profiles.sha256(checkpoint)
    config.update(
        split_id="full",
        prediction_scope=scope,
        evaluation={
            "command": "python scripts/score_drivaerml_candidate_dataset.py --split-id full",
            "generated_at": "2026-10-11T00:00:00Z",
        },
    )
    with (SPEC.parent / "reference-data/force_mom_constref_all.csv").open() as f:
        forces = {
            "run_" + r["run"]: {k: float(v) for k, v in r.items() if k != "run"}
            for r in csv.DictReader(f)
        }
    counts, cases, regional = {}, [], []
    metric_values = {}
    for cid in ids:
        truth = profiles.truth_case(cid)
        selected = profiles.selected_series(truth, scope)
        series = [
            profiles.prediction_series(
                s, [v + 0.01 for v in s["value"]] if "value" in s else None
            )
            for s in selected
        ]
        diagnostics, _ = profile_metrics(cid, series, scope)
        counts[cid] = {
            "surface_native_cells": truth["native_boundary"]["tuple_count"],
            "drivaerml-force-coefficients-v1": 1,
        }
        if scope == "surface_and_volume":
            counts[cid]["volume_native_cells"] = truth["native_volume"]["tuple_count"]
        field_values, statistics = {}, {}
        for metric in scorer.ALL_FIELD_METRIC_IDS:
            volume = metric.startswith(("volume_", "drivaerml_volume_"))
            if volume and scope == "surface_only":
                continue
            field_values[metric] = 5.0
            if metric.endswith("rel_l2"):
                count = counts[cid][
                    "volume_native_cells" if volume else "surface_native_cells"
                ]
                weighting = next(
                    m["weighting"] for m in spec["metrics"] if m["id"] == metric
                )
                statistics[metric] = {
                    "reduction": "relative_l2_percent",
                    "weighting": (
                        "support_weights"
                        if weighting == "surface_face_area"
                        else "uniform"
                    ),
                    "dataset_weighting": weighting,
                    "numerator": 0.25,
                    "denominator": 100.0,
                    "entity_count": count,
                    "total_weight": float(count),
                }
        f = forces[cid]
        native = {
            "Cd": f["cd"],
            "Cl": f["cl"],
            "CmPitch": (f["clf"] - f["clr"]) / 2,
            "Clf": f["clf"],
            "Clr": f["clr"],
        }
        predicted = {
            **native,
            "Cd": native["Cd"] + 0.001,
            "Clf": native["Cl"] / 2 + native["CmPitch"],
            "Clr": native["Cl"] / 2 - native["CmPitch"],
        }
        case = {
            "case_id": cid,
            "prediction_scope": scope,
            "support_counts": counts[cid],
            "field_metric_values": field_values,
            "field_metric_sufficient_statistics": statistics,
            "force": {
                "prediction": predicted,
                "truth": native,
                "predicted_lift_closure_abs": abs(
                    predicted["Cl"] - predicted["Clf"] - predicted["Clr"]
                ),
            },
            "diagnostic_metric_values": diagnostics,
            "profile_series": series,
        }
        cases.append(case)
        regional.append(
            {
                "case_id": cid,
                "core": {
                    "report_only_regional_diagnostics": region_fixture(cid, scope)
                },
            }
        )
    metric_values.update(cases[0]["field_metric_values"])
    for coefficient, name in [
        ("Cd", "cd"),
        ("Cl", "cl"),
        ("CmPitch", "cmpitch"),
        ("Clf", "clf"),
        ("Clr", "clr"),
    ]:
        metric_values["field_integrated_" + name + "_rmse"] = (
            sum(
                (
                    c["force"]["prediction"][coefficient]
                    - c["force"]["truth"][coefficient]
                )
                ** 2
                for c in cases
            )
            / len(cases)
        ) ** 0.5
    metric_values["field_integrated_lift_closure_max_abs"] = max(
        c["force"]["predicted_lift_closure_abs"] for c in cases
    )
    for metric in cases[0]["diagnostic_metric_values"]:
        metric_values[metric] = sum(
            c["diagnostic_metric_values"][metric] for c in cases
        ) / len(cases)
    evidence = {
        "cases": cases,
        "prediction_scope": scope,
        "metric_values": metric_values,
        "split": {
            "split_id": "full",
            "case_set_id": split["case_set_id"],
            "case_count": len(ids),
            "case_ids": ids,
        },
        "claims": {},
        "eligibility": {},
    }
    complete_metric_values(evidence, spec)
    evaluation = scorer.CandidateDatasetEvaluation(
        evidence=evidence, regional_diagnostics=None
    )
    binding = config["release_bindings"]["candidate_manifest"]
    metrics = scorer.schema_v3_case_metrics_candidate_adapter(
        evaluation,
        submission_id=p["submission_id"],
        candidate_support_release_id=binding["release_id"],
        candidate_support_manifest_sha256=binding["manifest_sha256"],
    )
    write(root / "metrics.json", metrics)
    package = scorer.schema_v3_relative_profile_chunks_candidate_adapter(
        evaluation,
        submission_id=p["submission_id"],
        namespaced_series_by_case={c["case_id"]: c["profile_series"] for c in cases},
    )
    scorer.write_schema_v3_profile_chunks_candidate(package, root / "profiles")
    write(
        root / "regional.json", aggregate_regional_diagnostics(regional, case_ids=ids)
    )
    spatial = load(ROOT / "examples/v3-template/discretization.json")
    spatial["inference"]["surface_input"]["case_record_id"] = "surface-geometry-input"
    domains = ["surface"] + (["volume"] if scope == "surface_and_volume" else [])
    spatial["inference"]["direct_outputs"] = [
        {
            "id": d + "-output",
            "domain": d,
            "representation": _representation_summary(
                description="Synthetic test native coverage",
                entity="cells",
                counts=[counts[c][d + "_native_cells"] for c in ids],
            ),
            "queries_per_forward_pass": {"kind": "fixed", "value": 1000},
        }
        for d in domains
    ]

    def out(sid):
        return "volume-output" if sid == "volume_native_cells" else "surface-output"

    spatial["inference"]["mappings"] = [
        _mapping_summary(
            sid, out(sid), derived=sid == "drivaerml-force-coefficients-v1"
        )
        for sid in counts[ids[0]]
    ]
    config["spatial_discretization"] = {
        k: spatial[k] for k in ("training", "inference")
    }
    records = [
        {
            "case_id": cid,
            "inference": {
                "inputs": [
                    {
                        "id": "surface-geometry-input",
                        "entity_counts": [{"entity": "points", "count": 3}],
                        "domain": {"kind": "full_dataset_domain"},
                    }
                ],
                "direct_outputs": [
                    _case_representation(
                        d + "-output", "cells", counts[cid][d + "_native_cells"]
                    )
                    for d in domains
                ],
                "mappings": [
                    _case_mapping(sid, out(sid), n) for sid, n in counts[cid].items()
                ],
            },
        }
        for cid in ids
    ]
    (root / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in records))
    write(root / "config.json", config)
    return config


@pytest.fixture(scope="module", params=["surface_only", "surface_and_volume"])
def completed(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("drivaerml-" + request.param)
    inputs(root, request.param)
    output = root / "package"
    assembler.assemble_package(
        config_path=root / "config.json",
        specification_path=SPEC,
        case_metrics_path=root / "metrics.json",
        profiles_path=root / "profiles",
        discretization_cases_path=root / "cases.jsonl",
        output_path=output,
        regional_diagnostics_path=root / "regional.json",
        checkpoint_paths={
            "coordinate-network-checkpoint": root / "test-checkpoint.bin"
        },
    )
    return root, output, request.param


@pytest.fixture
def repository(tmp_path, completed, monkeypatch):
    _, package, scope = completed
    shutil.copytree(SPEC.parent, tmp_path / "benchmark-specs/drivaerml")
    shutil.copy2(ROOT / "tests/fixtures/candidate-contracts/drivaerml.json", tmp_path / "benchmark-specs/drivaerml/submission-spec.json")
    shutil.copytree(ROOT / "schemas", tmp_path / "schemas")
    write(
        tmp_path / "leaderboard/manifest.json", load(ROOT / "leaderboard/manifest.json")
    )
    manifest_path = tmp_path / "leaderboard/manifest.json"
    candidate_manifest = load(manifest_path)
    candidate_manifest["data_release"].update(status="prototype_dummy_data", source_ref="dev")
    write(manifest_path, candidate_manifest)
    output = tmp_path / "submissions/drivaerml/drivaerml-workflow-test-v1"
    shutil.copytree(package, output)
    for module in (validator, manager, registrar):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in (
        "MANIFEST_PATH",
        "CLAIMS_ROOT",
        "CLAIMS_INDEX_PATH",
        "REVISION_HISTORY_PATH",
    ):
        monkeypatch.setattr(
            manager, name, tmp_path / getattr(manager, name).relative_to(ROOT)
        )
    monkeypatch.setattr(
        registrar,
        "DEFAULT_REGISTRY",
        tmp_path / "benchmark-specs/drivaerml/pre-release-reference-registry.json",
    )
    # The isolated repository has no .git; only the existing frozen relative evaluator is accepted.
    monkeypatch.setattr(
        validator,
        "_drivaerml_git_commit_is_bound",
        lambda root, rev: rev == "b9db402a3fa0efbb94fe36be2ba1fe6f7b4bc1e1",
    )
    return (
        tmp_path,
        output / "submission.json",
        load(tmp_path / "leaderboard/manifest.json"),
        scope,
    )


def test_full_split_intake_registration_publication(repository):
    root, path, manifest, scope = repository
    for options in (
        {"candidate_dry_run": True},
        {"dev_intake": True, "contributor_stage": True},
    ):
        errors, stats = validator.validate_submission_file(path, manifest, **options)
        assert not errors, "\n".join(errors)
        assert stats == {
            "cases": 50,
            "series": 400 if scope == "surface_only" else 2000,
        }
    assert manager.source_rows_by_dataset(manifest)["DrivAerML"] == []
    registrar.register(path, registry_path=registrar.DEFAULT_REGISTRY, write=True)
    errors, _ = validator.validate_submission_file(path, manifest)
    assert not errors, "\n".join(errors)
    assert manager.build(manifest) == []
    published = load(root / "leaderboard/datasets/drivaerml.json")
    assert (
        len(published) == 1 and published[0]["record_type"] == "pre_release_reference"
    )
    assert published[0]["claim_eligibility"]["academic_citation"] is False
    assert manager.check_generated_feeds(load(root / "leaderboard/manifest.json")) == []
    if scope == "surface_only":
        assert published[0]["metric_values"]["overall_score"] <= 60
        assert "volume_pressure_rel_l2" not in published[0]["metric_values"]
    (path.parent / "unexpected.txt").write_text("changed tree")
    assert manager.source_rows_by_dataset(manifest)["DrivAerML"] == []


def test_main_and_automatic_approval_rejected(repository):
    _, path, manifest, _ = repository
    errors, _ = validator.validate_submission_file(path, manifest)
    assert any("submissions are closed" in e for e in errors)
    manifest["data_release"]["source_ref"] = "main"
    errors, _ = validator.validate_submission_file(path, manifest, dev_intake=True)
    assert any("prototype dev release" in e for e in errors)


def test_prediction_or_coordinate_tampering_rejected(completed):
    root, path, scope = completed
    submission = load(path / "submission.json")
    validate_package_metrics(path, submission, load(SPEC))
    submission["metric_values"]["cp_cut_r2"] -= 0.1
    with pytest.raises(ValueError, match="cp_cut_r2"):
        validate_package_metrics(path, submission, load(SPEC))
    case = load(path / "profiles/chunk-000.json")["cases"][0]
    series = copy.deepcopy(case["series"])
    materialized = next(s for s in series if "coordinate" in s)
    materialized["coordinate"][0] += 0.001
    with pytest.raises(ValueError, match="metadata differs"):
        profiles.compare_profiles(case["case_id"], series, scope)


def test_native_sampling_uses_complete_prediction_stream_and_binds_core(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from test_drivaerml_diagnostic_evaluator import _write_prediction_manifest
    from reference.drivaerml.dev_workflow import validate_diagnostic_case
    from reference.drivaerml.diagnostic_evaluator import DrivAerDiagnosticEvaluatorError

    truth = copy.deepcopy(profiles.truth_case("run_44"))
    truth["native_boundary"]["tuple_count"] = 16
    truth["native_volume"]["tuple_count"] = 16
    for series in truth["series"]:
        if series["representation"] == "materialized":
            series["raw_native_cell_id"] = [i % 16 for i in range(len(series["value"]))]
    monkeypatch.setattr(profiles, "truth_case", lambda _: truth)
    pressure = np.arange(16, dtype=float) + 10
    velocity = np.column_stack((pressure, pressure + 1, pressure + 2))
    surface = _write_prediction_manifest(
        tmp_path / "surface",
        support_id="surface_native_cells",
        partitions=(7, 9),
        fields={"pMeanTrim": pressure, "wallShearStressMeanTrim": velocity},
    )
    volume = _write_prediction_manifest(
        tmp_path / "volume",
        support_id="volume_native_cells",
        partitions=(5, 11),
        fields={"pMeanTrim": pressure, "UMeanTrim": velocity},
    )
    result = profiles.evaluate_profiles("run_44", "surface_and_volume", surface, volume)
    assert len(result["profile_series"]) == 40
    for source, series in zip(truth["series"], result["profile_series"], strict=True):
        if source["representation"] == "shared_alias":
            assert "prediction" not in series
            continue
        ids = np.array(source["raw_native_cell_id"])
        expected = (
            2 * pressure[ids] / profiles.UINF**2
            if source["panel_id"] == "pressure_profiles"
            else np.linalg.norm(velocity[ids], axis=1) / profiles.UINF
        )
        np.testing.assert_allclose(series["prediction"], expected)
    core = SimpleNamespace(
        case_id="run_44",
        prediction_scope="surface_and_volume",
        volume_count=16,
        surface_prediction_manifest_sha256=profiles.sha256(surface),
        surface_prediction_chunk_sha256=tuple(
            c["sha256"] for c in load(surface)["chunks"]
        ),
        volume_prediction_manifest_sha256=profiles.sha256(volume),
        volume_prediction_chunk_sha256=tuple(
            c["sha256"] for c in load(volume)["chunks"]
        ),
    )
    checked = validate_diagnostic_case(
        result, "run_44.json", "a" * 64, contract=None, pinned=None, core=core
    )
    assert checked.values["cp_cut_rmse"] > 0
    changed = copy.deepcopy(result)
    changed["prediction_inputs"]["surface_native_cells"]["audit"]["chunk_sha256"][0] = (
        "0" * 64
    )
    with pytest.raises(
        scorer.DrivAerDatasetScorerError, match="differ from field-evaluation"
    ):
        validate_diagnostic_case(
            changed, "run_44.json", "a" * 64, contract=None, pinned=None, core=core
        )
    chunk = volume.parent / load(volume)["chunks"][-1]["file"]
    chunk.write_bytes(chunk.read_bytes() + b"changed")
    with pytest.raises(DrivAerDiagnosticEvaluatorError, match="SHA-256 mismatch"):
        profiles.evaluate_profiles("run_44", "surface_and_volume", surface, volume)


def test_profile_publication_never_overwrites_existing_evidence(tmp_path):
    output = tmp_path / "run_1.json"
    profiles.write_profile_evidence(output, {"complete": True})
    with pytest.raises(FileExistsError):
        profiles.write_profile_evidence(output, {"complete": False})
    assert load(output) == {"complete": True}


def test_checkpoint_bytes_must_match_before_assembly(completed, tmp_path):
    root, _, _ = completed
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the declared checkpoint")
    with pytest.raises(assembler.PackageAssemblyError, match="checkpoint"):
        assembler.assemble_package(
            config_path=root / "config.json",
            specification_path=SPEC,
            case_metrics_path=root / "metrics.json",
            profiles_path=root / "profiles",
            discretization_cases_path=root / "cases.jsonl",
            output_path=tmp_path / "bad-package",
            regional_diagnostics_path=root / "regional.json",
            checkpoint_paths={"coordinate-network-checkpoint": wrong},
        )
    assert not (tmp_path / "bad-package").exists()


def test_participant_command_packages_and_verifies_both_scopes(completed, tmp_path):
    from reference.drivaerml import participant_workflow as workflow

    root, _, scope = completed
    config = load(root / "config.json")
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "metrics").mkdir()
    shutil.copyfile(root / "metrics.json", evaluation / "metrics/cases.json")
    shutil.copyfile(root / "regional.json", evaluation / "regional-diagnostics.json")
    shutil.copytree(root / "profiles", evaluation / "package-profiles")
    workflow.write_json(
        evaluation / "workflow.json",
        {
            "entry": {
                "schema": workflow.ENTRY_SCHEMA,
                "split_id": "full",
                "prediction_scope": scope,
                "submission_id": config["participant"]["submission_id"],
            },
            "implementation": workflow.implementation_identity(),
            "outputs": {
                p.relative_to(evaluation).as_posix(): profiles.sha256(p)
                for p in evaluation.rglob("*")
                if p.is_file()
            },
        },
    )
    output = tmp_path / "ready-for-pr"
    result = workflow.package(
        evaluation,
        root / "config.json",
        root / "cases.jsonl",
        output,
        {"coordinate-network-checkpoint": root / "test-checkpoint.bin"},
    )
    assert result["status"] == "validated_official_package"
    assert load(output / "submission.json")["prediction_scope"] == scope
    assert workflow.verify_package(output)["status"] == "valid_official_package"
    document = load(output / "submission.json")
    document["metric_values"]["cp_cut_r2"] -= 0.01
    write(output / "submission.json", document)
    import subprocess

    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        workflow.verify_package(output)
