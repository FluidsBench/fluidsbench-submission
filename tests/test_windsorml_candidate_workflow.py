"""Disposable synthetic evidence exercises packaging, intake, and dev publication.

No test package is a model result or is retained in the repository submissions.
"""

from __future__ import annotations
import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts import assemble_windsorml_schema_v3_candidate as assembler
from scripts import validate_submission as validator, manage_leaderboard as manager
from scripts import register_windsorml_pre_release as registrar
from scripts.assemble_ahmedml_schema_v3_dev_fixture import (
    _case_representation,
    _case_mapping,
    _representation_summary,
    _mapping_summary,
)
from reference.windsorml.package import source_identity
from reference.windsorml.pre_release import build_registry_entry
from reference.windsorml.profiles import load_profile_support
from reference.windsorml.dataset_scorer import (
    score_candidate_dataset,
    PROFILE_FAMILY_METRICS,
)
from scripts.publish_windsorml_candidate_support import publish
from tests.test_windsorml_dataset_scorer import case_evidence

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "benchmark-specs/windsorml/submission-spec.json"
TEMPLATE = ROOT / "examples/windsorml-v3-candidate/package-config.template.json"
TRUTH = ROOT / "tests/fixtures/windsorml/profile-ground-truth-manifest.json"


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs(root):
    config = load(TEMPLATE)
    original = load(ROOT / "examples/v3-template/submission.json")
    config["participant"] = {
        k: original[k] for k in config["participant"] if k in original
    }
    participant = config["participant"]
    participant["submission_id"] = "windsorml-workflow-test-v1"
    participant["result_revision"]["series_id"] = "windsorml-workflow-test"
    participant["code_url"] = original["code_url"]
    participant["model"] = "Synthetic test evidence; no model inference performed"
    participant["note"] = "Disposable integration test only, never a scientific result."
    methodology = participant["methodology"]
    methodology["record_kind"] = (
        "submitter_reported"  # Simulates contributor metadata only.
    )
    fields = load(SPEC.parent / "methodology-contract.json")[
        "required_predicted_fields"
    ]
    methodology["architecture"]["predicted_fields"] = [
        {
            **f,
            "component_ids": ["coordinate-network"],
            "production": "direct_model_output",
        }
        for f in fields
    ]
    methodology["inference_compute"]["case_count"] = 35
    checkpoint = root / "test-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic test fixture; not trained weights")
    methodology["checkpoints"][0]["sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    config["split_id"] = "full"
    config["evaluation"] = {
        "command": "python scripts/score_windsorml_candidate_dataset.py --split-id full --case-evidence-directory evidence --output dataset.json",
        "generated_at": "2026-10-11T00:00:00Z",
    }
    ids = load(SPEC.parent / "splits/full.json")["case_ids"]
    source = source_identity()
    cases = [source.case(cid) for cid in ids]
    support_counts = {
        cid: {
            "windsorml_surface_native_points": source.case(cid).surface_entity_count,
            "windsorml_volume_native_cells": source.case(cid).volume_entity_count,
            "windsorml-force-coefficients-v1": 1,
            **{
                f: len(load_profile_support(case_id=cid).document["families"][f]) * 128
                for f in PROFILE_FAMILY_METRICS
            },
        }
        for cid in ids
    }

    def output_id(sid):
        return (
            "surface-output"
            if sid == "windsorml_surface_native_points"
            or "force" in sid
            or "_cp_" in sid
            else "volume-output"
        )

    spatial = load(ROOT / "examples/v3-template/discretization.json")
    inference = spatial["inference"]
    inference["surface_input"]["case_record_id"] = "surface-geometry-input"
    inference["direct_outputs"] = [
        {
            "id": domain + "-output",
            "domain": domain,
            "representation": _representation_summary(
                description="Synthetic native coverage for testing",
                entity=entity,
                counts=counts,
            ),
            "queries_per_forward_pass": {"kind": "fixed", "value": 1000},
        }
        for domain, entity, counts in (
            ("surface", "points", [c.surface_entity_count for c in cases]),
            ("volume", "cells", [c.volume_entity_count for c in cases]),
        )
    ]
    inference["mappings"] = [
        _mapping_summary(
            sid,
            output_id(sid),
            derived=sid
            not in ("windsorml_surface_native_points", "windsorml_volume_native_cells"),
        )
        for sid in support_counts[ids[0]]
    ]
    for item in inference["mappings"]:
        if item["method"]["kind"] == "reference_rule":
            item["method"]["rule_id"] = "windsorml-evaluator-derived-support-v1"
    config["spatial_discretization"] = {
        k: spatial[k] for k in ("training", "inference")
    }
    records = []
    for i, cid in enumerate(ids):
        case = source.case(cid)
        e = case_evidence(
            cid, index=i, field_error=5, force_error=0.001, profile_error=0.01
        )
        e["forces"]["truth_integrated"] = {
            "cd": case.force_truth.cd,
            "cl": case.force_truth.cl,
            "cs": case.force_truth.cs,
        }
        e["forces"]["prediction_integrated"] = {
            k: v + 0.001 for k, v in e["forces"]["truth_integrated"].items()
        }
        e["forces"].update(
            reference_area_m2=0.112, axes={"drag": "+x", "lift": "+y", "side": "+z"}
        )
        e["prediction_inputs"] = {}
        for domain, count, names in (
            (
                "surface",
                case.surface_entity_count,
                ("cpavg", "cfxavg", "cfyavg", "cfzavg"),
            ),
            (
                "volume",
                case.volume_entity_count,
                ("velocityxavg", "velocityyavg", "velocityzavg", "pressureavg"),
            ),
        ):
            e[domain]["entity_count"] = count
            e[domain]["fields"] = {
                n: {
                    "entity_count": count,
                    "component_count": 1,
                    **{
                        w: {
                            "squared_error": 0.25,
                            "squared_truth": 100.0,
                            "total_weight": count,
                        }
                        for w in ("uniform", "physical")
                    },
                }
                for n in names
            }
            e["prediction_inputs"][domain] = {
                "manifest_sha256": "a" * 64,
                "chunk_sha256": ["b" * 64],
                "entity_count": count,
            }
        write(root / "evidence" / f"{cid}.json", e)
        records.append(
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
                            "surface-output", "points", case.surface_entity_count
                        ),
                        _case_representation(
                            "volume-output", "cells", case.volume_entity_count
                        ),
                    ],
                    "mappings": [
                        _case_mapping(sid, output_id(sid), count)
                        for sid, count in support_counts[cid].items()
                    ],
                },
            }
        )
    write(root / "config.json", config)
    (root / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return config, checkpoint


def assemble(root, output):
    return assembler.assemble_package(
        config_path=root / "config.json",
        specification_path=SPEC,
        case_evidence_root=root / "evidence",
        discretization_cases_path=root / "cases.jsonl",
        profile_truth_manifest_path=TRUTH,
        checkpoint_paths={
            "coordinate-network-checkpoint": root / "test-checkpoint.bin"
        },
        output_path=output,
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("windsor-inputs")
    inputs(root)
    output = root / "package"
    result = assemble(root, output)
    assert result["case_count"] == 35
    return root, output


@pytest.fixture
def repository(tmp_path, completed, monkeypatch):
    _, package = completed
    (tmp_path / "benchmark-specs").mkdir()
    shutil.copytree(SPEC.parent, tmp_path / "benchmark-specs/windsorml")
    shutil.copy2(ROOT / "tests/fixtures/candidate-contracts/windsorml.json", tmp_path / "benchmark-specs/windsorml/submission-spec.json")
    (tmp_path / "leaderboard").mkdir()
    shutil.copy2(
        ROOT / "leaderboard/manifest.json", tmp_path / "leaderboard/manifest.json"
    )
    manifest_path = tmp_path / "leaderboard/manifest.json"
    candidate_manifest = load(manifest_path)
    candidate_manifest["data_release"].update(status="prototype_dummy_data", source_ref="dev")
    write(manifest_path, candidate_manifest)
    output = tmp_path / "submissions/windsorml/windsorml-workflow-test-v1"
    shutil.copytree(package, output)
    for module in (validator, manager, registrar):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        if hasattr(module, "MANIFEST_PATH"):
            monkeypatch.setattr(
                module, "MANIFEST_PATH", tmp_path / "leaderboard/manifest.json"
            )
    monkeypatch.setattr(
        registrar,
        "DEFAULT_REGISTRY",
        tmp_path / "benchmark-specs/windsorml/pre-release-reference-registry.json",
    )
    for name in ("CLAIMS_ROOT", "CLAIMS_INDEX_PATH", "REVISION_HISTORY_PATH"):
        relative = getattr(manager, name).relative_to(ROOT)
        monkeypatch.setattr(manager, name, tmp_path / relative)
    return (
        tmp_path,
        output / "submission.json",
        load(tmp_path / "leaderboard/manifest.json"),
    )


def test_complete_package_intake_and_registration(repository):
    root, path, manifest = repository
    for args in (
        {"candidate_dry_run": True},
        {"dev_intake": True},
        {"dev_intake": True, "contributor_stage": True},
    ):
        errors, stats = validator.validate_submission_file(path, manifest, **args)
        assert errors == [], "\n".join(errors)
        assert stats == {"cases": 35, "series": 560}
    assert manager.source_rows_by_dataset(manifest)["WindsorML"] == []
    proposed = registrar.register(
        path, registry_path=registrar.DEFAULT_REGISTRY, write=False
    )
    assert proposed["status"] == "proposed_not_written"
    registered = registrar.register(
        path, registry_path=registrar.DEFAULT_REGISTRY, write=True
    )
    assert registered["written"] is True
    errors, stats = validator.validate_submission_file(path, manifest)
    assert errors == [], "\n".join(errors)
    rows = manager.source_rows_by_dataset(manifest)["WindsorML"]
    assert len(rows) == 1 and rows[0]["record_type"] == "pre_release_reference"
    eligibility = manager.claim_eligibility("prototype_dummy_data", rows[0])
    assert (
        eligibility["academic_citation"] is False and eligibility["promotion"] is False
    )
    assert "WindsorML" in eligibility["reason"]
    assert manager.build(manifest) == []
    generated = load(root / "leaderboard/manifest.json")
    assert manager.check_generated_feeds(generated) == []
    published = load(root / "leaderboard/datasets/windsorml.json")
    assert len(published) == 1
    assert published[0]["profile_data"]["index_sha256"] == validator.sha256_file(
        path.parent / "profiles/index.json"
    )
    assert published[0]["claim_eligibility"]["academic_citation"] is False
    (path.parent / "unexpected.txt").write_text("changed package")
    errors, _ = validator.validate_submission_file(path, manifest)
    assert any("package tree" in e for e in errors)
    assert manager.source_rows_by_dataset(manifest)["WindsorML"] == []


def test_main_and_closed_validation_reject(repository):
    _, path, manifest = repository
    errors, _ = validator.validate_submission_file(path, manifest)
    assert any("submissions are closed" in e for e in errors)
    for key, value in (("source_ref", "main"), ("status", "official")):
        wrong = copy.deepcopy(manifest)
        wrong["data_release"][key] = value
        errors, _ = validator.validate_submission_file(path, wrong, dev_intake=True)
        assert any("requires the prototype dev release" in e for e in errors)


def test_changed_profile_hash_rejected(repository):
    _, path, manifest = repository
    with (path.parent / "profiles/chunk-000.json").open("a") as f:
        f.write(" ")
    errors, _ = validator.validate_submission_file(path, manifest, dev_intake=True)
    assert any("profile chunk digest differs" in e for e in errors)


def test_profile_metrics_and_coordinates_bound(completed):
    from reference.windsorml.package import validate_packaged_profiles

    _, package = completed
    cases = [
        c
        for p in sorted((package / "profiles").glob("chunk-*.json"))
        for c in load(p)["cases"]
    ]
    metrics = load(package / "submission.json")["metric_values"]
    validate_packaged_profiles(cases, metrics)
    wrong = dict(metrics, cp_cut_relative_r2=0.0)
    with pytest.raises(ValueError, match="cp_cut_relative_r2 differs"):
        validate_packaged_profiles(cases, wrong)
    cases[0]["series"][0]["coordinate"][0] += 1e-6
    with pytest.raises(ValueError, match="coordinates differ"):
        validate_packaged_profiles(cases, metrics)


@pytest.mark.parametrize(
    "mutation",
    ["checkpoint", "missing_case", "missing_relative", "statistics", "approval"],
)
def test_assembly_rejects_incomplete_or_conflicting_inputs(
    tmp_path, completed, mutation
):
    source, _ = completed
    shutil.copytree(source / "evidence", tmp_path / "evidence")
    for name in ("config.json", "cases.jsonl", "test-checkpoint.bin"):
        shutil.copy2(source / name, tmp_path / name)
    first = next((tmp_path / "evidence").glob("*.json"))
    if mutation == "checkpoint":
        (tmp_path / "test-checkpoint.bin").write_bytes(b"wrong")
    elif mutation == "missing_case":
        first.unlink()
    elif mutation == "approval":
        config = load(tmp_path / "config.json")
        config["participant"]["approval"] = {}
        write(tmp_path / "config.json", config)
    else:
        evidence = load(first)
        if mutation == "missing_relative":
            del evidence["profiles"]["families"]["windsorml_cp_relative_v1"]
        else:
            evidence["surface"]["fields"]["cpavg"]["physical"]["squared_error"] = 100
        write(first, evidence)
    with pytest.raises((assembler.PackageAssemblyError, ValueError)):
        assemble(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_support_publication_is_reproducible(tmp_path):
    manifest = publish(tmp_path / "support")
    assert (
        manifest.read_bytes()
        == (SPEC.parent / "scoring-support-candidate/manifest.json").read_bytes()
    )
    from reference.scoring_support import load_support_release

    for split in load(SPEC)["splits"]:
        release = load_support_release(manifest, split["case_set_id"])
        assert (
            list(release.cases) == load(SPEC.parent / split["index_file"])["case_ids"]
        )


def test_template_bindings_and_all_splits():
    config = load(TEMPLATE)
    spec = load(SPEC)
    assert assembler.unresolved_tokens(config)
    assembler._release_bindings(config, spec, SPEC, TRUTH)
    for split in spec["splits"]:
        _, _, ids, _ = assembler._find_split(spec, SPEC, split["id"])
        assert len(ids) == split["case_count"]
