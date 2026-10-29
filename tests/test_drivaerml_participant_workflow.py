from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from reference.drivaerml import participant_workflow as w
from reference.drivaerml import dev_profiles as profiles
from scripts.create_drivaerml_dummy_predictions import _create_support
from scripts import drivaerml as cli


def entry(root, **updates):
    value = {
        "schema": w.ENTRY_SCHEMA,
        "submission_id": "workflow-test-v2",
        "split_id": "full",
        "prediction_scope": "surface_only",
    }
    value.update(updates)
    w.write_json(root / "entry.json", value)
    return value


@pytest.mark.parametrize(
    "split_id",
    [
        "full",
        "medium",
        "scarce",
        "super_scarce",
        "geometry",
        "high_drag",
        "low_drag",
        "rear_separation",
    ],
)
def test_only_published_splits_resolve(tmp_path, split_id):
    expected = entry(tmp_path, split_id=split_id)
    assert w.load_entry(tmp_path) == expected
    assert len(w.official_split(split_id)["case_ids"]) in (50, 97)


@pytest.mark.parametrize(
    "changes",
    [
        {"split_id": "custom"},
        {"split_id": "../full"},
        {"test_case_ids": ["run_1"]},
        {"train_case_ids": ["run_2"]},
        {"force_prediction_source": "direct_coefficients"},
        {"direct_force_coefficients": {"Cd": 0.3}},
    ],
)
def test_custom_splits_and_direct_forces_rejected(tmp_path, changes):
    entry(tmp_path, **changes)
    with pytest.raises(ValueError):
        w.load_entry(tmp_path)


def test_fetch_plan_is_offline_and_scope_aware(tmp_path, monkeypatch):
    run = Mock(side_effect=AssertionError("dry run must not download"))
    monkeypatch.setattr(w.subprocess, "run", run)
    surface = w.fetch_data("full", tmp_path, "surface_only", dry_run=True)
    full = w.fetch_data("full", tmp_path, "surface_and_volume", dry_run=True)
    assert surface["case_count"] == full["case_count"] == 50
    assert len(surface["files"]) == 100
    assert not any(".part" in x for x in surface["files"])
    assert set(surface["files"]) < set(full["files"])
    assert surface["total_bytes"] < full["total_bytes"]
    assert full["revision"] == "7a5c0948ce27be709b1116a3a190f806e7a8f79f"


def test_support_fetch_is_repeatable_verified_and_offline(tmp_path, monkeypatch):
    body = {"cases": []}
    w.write_json(tmp_path / "chunks/one.json", body)
    index = {
        "case_locations": [
            {
                "case_id": "run_1",
                "chunk_path": "chunks/one.json",
                "chunk_sha256": profiles.sha256(tmp_path / "chunks/one.json"),
            }
        ]
    }
    w.write_json(tmp_path / "index.json", index)
    monkeypatch.setattr(
        profiles, "INDEX_SHA256", profiles.sha256(tmp_path / "index.json")
    )
    monkeypatch.setattr(w, "official_split", lambda sid: {"case_ids": ["run_1"]})
    monkeypatch.setattr(
        profiles.urllib.request, "urlopen", Mock(side_effect=AssertionError("offline"))
    )
    assert w.fetch_support("full", tmp_path, offline=True) == w.fetch_support(
        "full", tmp_path, offline=True
    )
    (tmp_path / "chunks/one.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        w.fetch_support("full", tmp_path, offline=True)
    (tmp_path / "chunks/one.json").unlink()
    with pytest.raises(ValueError, match="missing offline"):
        w.fetch_support("full", tmp_path, offline=True)


@pytest.fixture(params=["surface_only", "surface_and_volume"])
def small_campaign(tmp_path, monkeypatch, request):
    monkeypatch.setattr(w, "check_runtime", lambda: None)
    entry_root, dataset, output = [
        tmp_path / name for name in ["entry", "data", "output"]
    ]
    value = entry(entry_root, prediction_scope=request.param)
    cases = ["run_1", "run_2"]
    records = {}
    for cid in cases:
        case_dir = entry_root / "cases" / cid
        for domain in ["surface"] + (
            ["volume"] if request.param == "surface_and_volume" else []
        ):
            _create_support(
                case_dir,
                case_id=cid,
                support_id=domain + "_native_cells",
                entity_count=4,
                max_chunk_rows=2,
            )
            (case_dir / (domain + "_native_cells")).rename(case_dir / domain)
        files = []
        for name in ["boundary", "area", "volume.part1"]:
            path = dataset / cid / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"native fixture")
            files.append(
                SimpleNamespace(
                    path=PurePosixPath(cid) / name, size_bytes=14, element_count=4
                )
            )
        records[cid] = SimpleNamespace(
            boundary=files[0], surface_cell_area=files[1], volume_parts=files[2:]
        )
    monkeypatch.setattr(w, "official_split", lambda sid: {"case_ids": cases})
    monkeypatch.setattr(
        w,
        "load_native_source_pin",
        lambda path: SimpleNamespace(case=records.__getitem__),
    )
    monkeypatch.setattr(
        w, "fetch_support", lambda *a, **k: {"root": str(tmp_path / "truth")}
    )
    monkeypatch.setattr(w, "implementation_identity", lambda: "evaluator-v2")
    calls = []

    def run_case(e, cid, manifests, native, data, truth, staging, limit):
        calls.append(cid)
        w.write_json(
            staging / "core.json", {"case_id": cid, "scope": e["prediction_scope"]}
        )
        w.write_json(staging / "profiles.json", {"case_id": cid})

    reductions = []

    def reduce(e, completed, staging, truth):
        reductions.append(list(completed))
        w.write_json(staging / "result.json", {"cases": list(completed)})

    monkeypatch.setattr(w, "run_case", run_case)
    monkeypatch.setattr(w, "reduce_cases", reduce)
    return SimpleNamespace(
        entry=entry_root,
        data=dataset,
        output=output,
        calls=calls,
        reductions=reductions,
        run_case=run_case,
        records=records,
        scope=request.param,
    )


def evaluate(campaign, **kwargs):
    return w.evaluate_entry(campaign.entry, campaign.data, campaign.output, **kwargs)


def test_interrupted_case_and_reduction_resume_without_manual_cleanup(
    small_campaign, monkeypatch
):
    c = small_campaign

    def interrupted(*args):
        c.run_case(*args)
        if args[1] == "run_2":
            raise RuntimeError("interrupted after writing both outputs")

    monkeypatch.setattr(w, "run_case", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate(c)
    assert (c.output / "cases/run_1/checkpoint.json").is_file()
    assert not (c.output / "cases/run_2").exists()
    monkeypatch.setattr(w, "run_case", c.run_case)
    reduce = w.reduce_cases
    monkeypatch.setattr(
        w, "reduce_cases", Mock(side_effect=RuntimeError("reducer interrupted"))
    )
    with pytest.raises(RuntimeError, match="reducer"):
        evaluate(c, resume=True)
    monkeypatch.setattr(w, "reduce_cases", reduce)
    assert evaluate(c, resume=True)["reused_cases"] == 2
    assert c.calls == ["run_1", "run_2", "run_2"]
    assert evaluate(c, resume=True)["reused_cases"] == 2
    assert len(c.reductions) == 1
    with pytest.raises(ValueError, match="--resume"):
        evaluate(c)


@pytest.mark.parametrize(
    "changed", ["prediction", "manifest", "core", "evaluator", "aggregate"]
)
def test_resume_rejects_changed_inputs_and_outputs(
    small_campaign, monkeypatch, changed
):
    c = small_campaign
    evaluate(c)
    if changed == "prediction":
        file = next((c.entry / "cases/run_1/surface").rglob("*.npz"))
        data = bytearray(file.read_bytes())
        data[-1] ^= 1
        file.write_bytes(data)
    elif changed == "manifest":
        file = c.entry / "cases/run_1/surface/manifest.json"
        file.write_text(file.read_text() + "\n")
    elif changed == "core":
        (c.output / "cases/run_1/core.json").write_text("{}")
    elif changed == "evaluator":
        monkeypatch.setattr(w, "implementation_identity", lambda: "different-code")
    else:
        (c.output / "evaluation/result.json").write_text("{}")
    with pytest.raises(ValueError):
        evaluate(c, resume=True)
    assert c.calls == ["run_1", "run_2"]


def test_preflight_rejects_missing_cases_native_data_and_direct_force_files(
    small_campaign,
):
    c = small_campaign
    assert w.preflight(c.entry, c.data)["case_count"] == 2
    force = c.entry / "cases/run_1/direct-force-coefficients.json"
    force.write_text('{"Cd":0.3,"Clf":0.1,"Clr":0.1}')
    with pytest.raises(ValueError, match="direct force"):
        w.preflight(c.entry, c.data)
    force.unlink()
    boundary = c.data / "run_2/boundary"
    boundary.unlink()
    with pytest.raises(ValueError, match="native file"):
        w.preflight(c.entry, c.data)
    assert not c.output.exists()


def test_cli_rejects_custom_split_without_downloading(tmp_path):
    assert (
        cli.main(
            [
                "fetch-data",
                "--split-id",
                "custom",
                "--destination",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 2
    )
