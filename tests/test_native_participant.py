"""Recovery and policy tests with tiny native-order prediction campaigns."""

import json
import shutil
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from reference import native_participant as n
from reference.native_participant_cli import main

TRUTH = n.ROOT / "tests/fixtures/windsorml/profile-ground-truth-manifest.json"


def entry(w, path, **updates):
    value = dict(
        schema=f"fluidsbench-{w.dataset}-entry-v1",
        submission_id=f"{w.dataset}-test-v1",
        split_id="full",
        prediction_scope="surface_and_volume",
        force_prediction_source="field_integrated",
    )
    value.update(updates)
    n.write_json(path / "entry.json", value)
    return value


@pytest.fixture(params=["ahmedml", "windsorml"])
def workflow(request):
    return n.ParticipantWorkflow(request.param)


def test_all_prescribed_splits_resolve(workflow, tmp_path):
    declarations = n.load_json(workflow.spec_path)["splits"]
    assert len(declarations) == 8
    for item in declarations:
        value = entry(workflow, tmp_path, split_id=item["id"])
        assert workflow.load_entry(tmp_path) == value
        assert (
            len(workflow.official_split(item["id"])["case_ids"]) == item["case_count"]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"split_id": "custom"},
        {"split_id": "../full"},
        {"test_case_ids": ["run_1"]},
        {"train_case_ids": []},
        {"direct_force_coefficients": {"cd": 0.3}},
        {"force_prediction_source": "direct"},
        {"prediction_scope": "surface_only"},
    ],
)
def test_custom_splits_direct_forces_and_surface_only_rejected(
    workflow, tmp_path, changes
):
    entry(workflow, tmp_path, **changes)
    with pytest.raises(ValueError):
        workflow.load_entry(tmp_path)


def test_download_plan_is_offline_and_pinned(workflow, tmp_path, monkeypatch):
    run = Mock(side_effect=AssertionError("offline plan must not download"))
    monkeypatch.setattr(n.subprocess, "run", run)
    plan = workflow.fetch_data("full", tmp_path, dry_run=True)
    assert plan["case_count"] == (50 if workflow.dataset == "ahmedml" else 35)
    assert len(plan["files"]) == 5 * plan["case_count"]
    assert plan["revision"] == workflow.contract.REPOSITORY_REVISION
    assert plan["total_bytes"] > 1_000_000
    run.assert_not_called()


def test_fetch_data_uses_exact_plan(workflow, tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(n.subprocess, "run", run)
    monkeypatch.setattr(n.shutil, "which", lambda _: "/bin/hf")
    plan = workflow.fetch_data("full", tmp_path)
    command = run.call_args.args[0]
    assert command[:3] == ["/bin/hf", "download", plan["repository"]]
    assert command[3 : 3 + len(plan["files"])] == plan["files"]
    assert command[-4:] == [
        "--revision",
        plan["revision"],
        "--local-dir",
        str(tmp_path),
    ]


def test_profile_manifest_offline_reuse_and_corruption(workflow, tmp_path, monkeypatch):
    from scripts import fetch_profile_ground_truth as fetch

    monkeypatch.setattr(
        fetch, "fetch_manifest", Mock(side_effect=AssertionError("offline"))
    )
    with pytest.raises(ValueError, match="missing offline"):
        workflow.profile_manifest(tmp_path, offline=True)
    digest = n.sha256_file(TRUTH)
    target = tmp_path / "profile-ground-truth" / digest / "manifest.json"
    target.parent.mkdir(parents=True)
    shutil.copyfile(TRUTH, target)
    assert workflow.profile_manifest(tmp_path, offline=True) == target
    target.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        workflow.profile_manifest(tmp_path, offline=True)


def manifest(w, root, cid, domain):
    points = w.dataset == "windsorml" and domain == "surface"
    support = f"{w.dataset}_{domain}_native_{'points' if points else 'cells'}"
    components = dict(w.predictions.support_field_components(support))
    root.mkdir(parents=True)
    chunk = root / "chunk.npz"
    fields = {
        key: np.ones(4 if size == 1 else (4, size)) for key, size in components.items()
    }
    np.savez(chunk, raw_cell_id=np.arange(4, dtype=np.int64), **fields)
    n.write_json(
        root / "manifest.json",
        {
            "format": f"{w.dataset}-native-prediction-chunks-candidate",
            "format_version": 1,
            "artifact_role": "local_evaluator_input_not_official_submission_artifact",
            "case_id": cid,
            "support_id": support,
            "association": "PointData" if points else "CellData",
            "total_row_count": 4,
            "field_components": components,
            "chunks": [
                {
                    "chunk_index": 0,
                    "file": "chunk.npz",
                    "sha256": n.sha256_file(chunk),
                    "row_count": 4,
                    "raw_cell_id_start": 0,
                    "raw_cell_id_stop": 4,
                }
            ],
        },
    )


@pytest.fixture
def campaign(workflow, tmp_path, monkeypatch):
    w = workflow
    monkeypatch.setattr(n, "check_runtime", lambda: None)
    entry_root, data, output = [tmp_path / name for name in ("entry", "data", "output")]
    value = entry(w, entry_root)
    cases = ["run_1", "run_2"]
    records = {}
    for cid in cases:
        for domain in ("surface", "volume"):
            manifest(w, entry_root / "cases" / cid / domain, cid, domain)
        files = {}
        for name in (
            "boundary",
            "surface_cell_area",
            "surface_dual_area",
            "volume",
            "geometry_parameters",
            "force_coefficients",
        ):
            path = data / cid / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"native fixture")
            files[name] = SimpleNamespace(path=PurePosixPath(cid) / name, size_bytes=14)
        records[cid] = SimpleNamespace(
            case_id=cid, surface_entity_count=4, volume_entity_count=4, **files
        )
    monkeypatch.setattr(w, "official_split", lambda _: {"case_ids": cases})
    monkeypatch.setattr(
        w, "source_identity", lambda: SimpleNamespace(case=records.__getitem__)
    )
    monkeypatch.setattr(w, "implementation_identity", lambda: "test-evaluator")
    monkeypatch.setattr(
        w,
        "fetch_support",
        lambda *a, **k: {
            "root": str(tmp_path),
            "profile_ground_truth_manifest": str(TRUTH),
        },
    )
    calls, reductions = [], []

    def run(cid, manifests, dataset, support, out):
        calls.append(cid)
        s = {
            "panel_id": "pressure_profiles",
            "station_id": "station",
            "quantity_id": "cp",
            "coordinate": [0, 1],
            "truth": [1, 2],
            "prediction": [1.1, 2.1],
        }
        profiles = (
            {"series": [s]}
            if w.dataset == "ahmedml"
            else {"families": {"constant": [s]}}
        )
        n.write_json(out, {"case_id": cid, "profiles": profiles})

    def reduce(e, source, out):
        reductions.append(e)
        n.write_json(out, {"metric_values": {"test_metric": 0.1}})

    monkeypatch.setattr(w, "run_case", run)
    monkeypatch.setattr(w, "reduce_cases", reduce)
    return SimpleNamespace(
        w=w,
        entry=entry_root,
        data=data,
        output=output,
        calls=calls,
        reductions=reductions,
        cases=cases,
        value=value,
    )


def evaluate(c, **kwargs):
    return c.w.evaluate_entry(c.entry, c.data, c.output, **kwargs)


def test_case_interruption_and_reduction_interruption_resume(campaign, monkeypatch):
    c = campaign
    original = c.w.run_case

    def interrupted(cid, *args):
        if cid == "run_2":
            args[-1].write_text("partial")
            raise RuntimeError("case interrupted")
        original(cid, *args)

    monkeypatch.setattr(c.w, "run_case", interrupted)
    with pytest.raises(RuntimeError, match="case interrupted"):
        evaluate(c)
    assert (c.output / "cases/run_1/checkpoint.json").exists()
    assert not (c.output / "cases/run_2").exists()
    monkeypatch.setattr(c.w, "run_case", original)
    reduce = c.w.reduce_cases
    monkeypatch.setattr(
        c.w, "reduce_cases", Mock(side_effect=RuntimeError("reduction interrupted"))
    )
    with pytest.raises(RuntimeError, match="reduction interrupted"):
        evaluate(c, resume=True)
    assert not (c.output / "evaluation").exists()
    monkeypatch.setattr(c.w, "reduce_cases", reduce)
    result = evaluate(c, resume=True)
    assert result["reused_cases"] == 2
    assert c.calls == c.cases
    assert evaluate(c, resume=True)["reused_cases"] == 2
    assert len(c.reductions) == 1
    report = c.w.report(Path(result["output"]), "run_1", c.output / "report.html")
    assert report["profile_count"] == 1
    assert (c.output / "report.html").read_text().count("<polyline") == 2


@pytest.mark.parametrize(
    "change",
    [
        "prediction",
        "manifest",
        "case",
        "checkpoint",
        "evaluation",
        "extra_evaluation",
        "implementation",
        "entry",
    ],
)
def test_resume_rejects_changed_inputs_and_outputs(campaign, monkeypatch, change):
    c = campaign
    evaluate(c)
    if change == "implementation":
        monkeypatch.setattr(c.w, "implementation_identity", lambda: "changed")
    elif change == "entry":
        entry(c.w, c.entry, submission_id="changed")
    else:
        path = {
            "prediction": c.entry / "cases/run_1/surface/chunk.npz",
            "manifest": c.entry / "cases/run_1/surface/manifest.json",
            "case": c.output / "cases/run_1/evidence.json",
            "checkpoint": c.output / "cases/run_1/checkpoint.json",
            "evaluation": c.output / "evaluation/dataset-evidence.json",
            "extra_evaluation": c.output / "evaluation/unexpected.json",
        }[change]
        path.write_text("{}")
    with pytest.raises(ValueError):
        evaluate(c, resume=True)


def test_preflight_rejects_missing_case_native_file_and_direct_force(campaign):
    c = campaign
    (c.entry / "cases/run_1/forces.json").write_text("{}")
    with pytest.raises(ValueError, match="direct force"):
        evaluate(c)
    (c.entry / "cases/run_1/forces.json").unlink()
    (c.data / "run_1/boundary").unlink()
    with pytest.raises(ValueError, match="native file"):
        evaluate(c)
    shutil.rmtree(c.entry / "cases/run_2")
    with pytest.raises(ValueError, match="exactly match"):
        evaluate(c)
    assert c.calls == []


def test_concurrent_campaign_is_rejected(campaign):
    with n.campaign_lock(campaign.output):
        with pytest.raises(ValueError, match="another evaluation"):
            evaluate(campaign)


def test_cli_rejects_unsupported_split_before_download(workflow, tmp_path):
    assert (
        main(
            workflow.dataset,
            [
                "fetch-data",
                "--split-id",
                "custom",
                "--destination",
                str(tmp_path),
                "--dry-run",
            ],
        )
        == 2
    )


@pytest.mark.parametrize(
    "change",
    [
        {"test_case_ids": ["run_1"]},
        {"force_prediction_source": "direct_coefficients"},
        {"direct_force_coefficients": {"cd": 0.3}},
        {"prediction_scope": "surface_only"},
    ],
)
def test_existing_assemblers_reject_policy_overrides(workflow, tmp_path, change):
    import importlib

    assembler = importlib.import_module(
        f"scripts.assemble_{workflow.dataset}_schema_v3_candidate"
    )
    config = n.load_json(
        n.ROOT
        / f"examples/{workflow.dataset}-v3-candidate/package-config.template.json"
    )
    config.update(change)
    path = tmp_path / "config.json"
    n.write_json(path, config)
    with pytest.raises(
        ValueError, match="forbidden|field_integrated|surface_and_volume"
    ):
        assembler.assemble_package(
            config_path=path,
            specification_path=workflow.spec_path,
            case_evidence_root=tmp_path,
            discretization_cases_path=tmp_path / "cases.jsonl",
            profile_truth_manifest_path=TRUTH,
            checkpoint_paths={},
            output_path=tmp_path / "package",
        )
