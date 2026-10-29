"""Complete synthetic campaigns exercise the real reducers, assemblers and intake.

The per-case inference step is simulated; these packages are never model results.
"""

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from reference import native_participant as n
from scripts.assemble_ahmedml_schema_v3_dev_fixture import (
    _case_representation,
    _case_mapping,
    _representation_summary,
    _mapping_summary,
)
from tests.test_windsorml_candidate_workflow import inputs as windsor_inputs, TRUTH
from tests.test_ahmedml_dataset_scorer import _case_evidence
from tests.test_ahmedml_regional_aggregate import _case as regional_case, _metrics


def ahmed_inputs(root):
    w = n.ParticipantWorkflow("ahmedml")
    config = n.load_json(
        n.ROOT / "examples/ahmedml-v3-candidate/package-config.template.json"
    )
    config["split_id"] = "full"
    original = n.load_json(n.ROOT / "examples/v3-template/submission.json")
    config["participant"] = {
        k: original[k] for k in config["participant"] if k in original
    }
    participant = config["participant"]
    participant["code_url"] = original["code_url"]
    participant["submission_id"] = "ahmedml-workflow-test-v1"
    participant["result_revision"]["series_id"] = "ahmedml-workflow-test"
    participant["model"] = "Synthetic integration test; no model inference"
    participant["note"] = "Disposable test only, never a scientific result."
    methodology = participant["methodology"]
    methodology["record_kind"] = "submitter_reported"
    fields = n.load_json(w.spec_root / "methodology-contract.json")[
        "required_predicted_fields"
    ]
    methodology["architecture"]["predicted_fields"] = [
        {
            **field,
            "component_ids": ["coordinate-network"],
            "production": "direct_model_output",
        }
        for field in fields
    ]
    ids = w.official_split("full")["case_ids"]
    methodology["inference_compute"]["case_count"] = len(ids)
    checkpoint = root / "test-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic test only; not trained weights")
    methodology["checkpoints"][0]["sha256"] = n.sha256_file(checkpoint)
    config["evaluation"] = {
        "command": "synthetic workflow integration test",
        "generated_at": "2026-10-12T00:00:00Z",
    }
    source = w.source_identity()
    cases = [source.case(cid) for cid in ids]
    counts = {
        c.case_id: {
            "ahmedml-surface-native-cells-v1": c.surface_entity_count,
            "ahmedml-volume-native-cells-v1": c.volume_entity_count,
            "ahmedml-force-coefficients-v1": 1,
            "ahmedml-surface-cp-profiles-v1": 384,
            "ahmedml-volume-velocity-profiles-v1": 512,
        }
        for c in cases
    }

    def output_id(sid):
        return "volume-output" if "volume" in sid or "wake" in sid else "surface-output"

    spatial = n.load_json(n.ROOT / "examples/v3-template/discretization.json")
    spatial["inference"]["surface_input"]["case_record_id"] = "surface-geometry-input"
    spatial["inference"]["direct_outputs"] = [
        {
            "id": domain + "-output",
            "domain": domain,
            "representation": _representation_summary(
                description="Synthetic test coverage",
                entity="cells",
                counts=[getattr(c, domain + "_entity_count") for c in cases],
            ),
            "queries_per_forward_pass": {"kind": "fixed", "value": 1000},
        }
        for domain in ("surface", "volume")
    ]
    spatial["inference"]["mappings"] = [
        _mapping_summary(sid, output_id(sid), derived="native" not in sid)
        for sid in counts[ids[0]]
    ]
    config["spatial_discretization"] = {
        k: spatial[k] for k in ("training", "inference")
    }
    records = []
    for i, case in enumerate(cases):
        cid = case.case_id
        evidence = _case_evidence(cid, i)
        regional = regional_case(cid)
        evidence.update(regional)
        for field_id, field in evidence["field_statistics"].items():
            count = getattr(case, field_id.split("_")[0] + "_entity_count")
            regions = evidence["report_only_regional_diagnostics"][field_id]
            for j, region in enumerate(regions.values()):
                region["entity_count"] = count // len(regions) + (
                    j < count % len(regions)
                )
                old = region["uniform"]
                region["uniform"] = _metrics(
                    absolute=old["absolute_error"],
                    squared=old["squared_error"],
                    truth=old["squared_truth"],
                    weight=region["entity_count"],
                )
            field["entity_count"] = count
            for weighting in ("uniform", "physical"):
                sums = {
                    key: sum(r[weighting][key] for r in regions.values())
                    for key in (
                        "absolute_error",
                        "squared_error",
                        "squared_truth",
                        "total_weight",
                    )
                }
                field[weighting] = _metrics(
                    absolute=sums["absolute_error"],
                    squared=sums["squared_error"],
                    truth=sums["squared_truth"],
                    weight=sums["total_weight"],
                )
            primary = "physical" if field_id.startswith("surface") else "uniform"
            secondary = "equal_entity" if primary == "physical" else "physical"
            values = evidence["metric_values"]
            values[field_id + "_rel_l2"] = field[primary]["relative_l2_percent"]
            values[field_id + "_" + secondary + "_rel_l2"] = field[
                "uniform" if primary == "physical" else "physical"
            ]["relative_l2_percent"]
            values[field_id + "_rel_l1"] = 1.0
            values[field_id + "_mae"] = field[primary]["mae"]
            values[field_id + "_rmse"] = field[primary]["rmse"]
        evidence["prediction_inputs"] = {
            domain: {
                "manifest_sha256": "a" * 64,
                "chunk_sha256": ["b" * 64],
                "entity_count": getattr(case, domain + "_entity_count"),
            }
            for domain in ("surface", "volume")
        }
        n.write_json(root / "evidence" / f"{cid}.json", evidence)
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
                            domain + "-output",
                            "cells",
                            getattr(case, domain + "_entity_count"),
                        )
                        for domain in ("surface", "volume")
                    ],
                    "mappings": [
                        _case_mapping(sid, output_id(sid), count)
                        for sid, count in counts[cid].items()
                    ],
                },
            }
        )
    n.write_json(root / "config.json", config)
    (root / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return config, checkpoint


@pytest.mark.parametrize("dataset", ["ahmedml", "windsorml"])
def test_complete_campaign_package_and_real_validator(tmp_path, monkeypatch, dataset):
    config, checkpoint = (ahmed_inputs if dataset == "ahmedml" else windsor_inputs)(
        tmp_path
    )
    w = n.ParticipantWorkflow(dataset)
    entry = {
        "schema": f"fluidsbench-{dataset}-entry-v1",
        "submission_id": config["participant"]["submission_id"],
        "split_id": "full",
        "prediction_scope": n.SCOPE,
        "force_prediction_source": "field_integrated",
    }
    monkeypatch.setattr(n, "check_runtime", lambda: None)
    monkeypatch.setattr(
        w,
        "preflight",
        lambda *a, **kw: {
            "entry": entry,
            "support": {
                "root": str(tmp_path),
                "profile_ground_truth_manifest": str(TRUTH),
            },
        },
    )
    monkeypatch.setattr(w, "prediction_manifests", lambda *a: {})
    # Only native inference is simulated. Reduction, receipts, package creation,
    # checkpoint checks and the validator subprocess are the production paths.
    monkeypatch.setattr(
        w,
        "run_case",
        lambda cid, manifests, data, support, out: shutil.copyfile(
            tmp_path / "evidence" / f"{cid}.json", out
        ),
    )
    result = w.evaluate_entry(tmp_path, tmp_path, tmp_path / "work")
    root = Path(result["output"])
    assert (
        w.evaluate_entry(tmp_path, tmp_path, tmp_path / "work", resume=True)[
            "reused_cases"
        ]
        == result["case_count"]
    )
    package = tmp_path / "package"
    outcome = w.package(
        root,
        tmp_path / "config.json",
        tmp_path / "cases.jsonl",
        package,
        {"coordinate-network-checkpoint": checkpoint},
    )
    assert outcome["status"] == "validated_official_package"
    assert w.verify_package(package)["status"] == "valid_official_package"
    report = w.report(
        root, w.official_split("full")["case_ids"][0], tmp_path / "report.html"
    )
    assert report["profile_count"] == (7 if dataset == "ahmedml" else 16)
    submission = n.load_json(package / "submission.json")
    submission["metric_values"]["cd_r2"] -= 0.1
    n.write_json(package / "submission.json", submission)
    with pytest.raises((ValueError, n.subprocess.CalledProcessError)):
        w.verify_package(package)
