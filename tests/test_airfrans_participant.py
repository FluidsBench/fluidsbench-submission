"""Participant policy/recovery tests; synthetic fixtures are never submitted."""

import copy
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reference.airfrans import participant as w


@pytest.mark.parametrize(
    "split,count", [("full", 200), ("scarce", 200), ("aoa_extrapolation", 196)]
)
def test_official_splits(split, count):
    assert len(w.official_split(split)["case_ids"]) == count


@pytest.mark.parametrize(
    "split", ["reynolds_extrapolation", "custom", "default", "../full"]
)
def test_closed_and_custom_splits(split):
    with pytest.raises(w.WorkflowError, match="enabled official"):
        w.official_split(split)


@pytest.mark.parametrize(
    "change",
    [
        {"case_ids": []},
        {"force_prediction_source": "direct"},
        {"split_id": "reynolds_extrapolation"},
        {"submission_id": "../bad"},
    ],
)
def test_entry_policy(tmp_path, change):
    entry = {
        "schema": "fluidsbench-airfrans-entry-v1",
        "submission_id": "test",
        "split_id": "full",
        **change,
    }
    w.write_json(tmp_path / "entry.json", entry)
    with pytest.raises(w.WorkflowError):
        w.load_entry(tmp_path)


def prediction(path):
    np.savez(
        path,
        velocity=np.ones((3, 2)),
        pressure=np.ones(3),
        internal_points=np.ones((3, 3)),
        airfoil_points=np.ones((2, 3)),
    )


@pytest.mark.parametrize("problem", ["force", "missing", "nan", "shape", "object"])
def test_prediction_preflight(tmp_path, problem):
    path = tmp_path / "prediction.npz"
    prediction(path)
    with np.load(path) as source:
        arrays = dict(source)
    if problem == "force":
        arrays["cd"] = np.ones(1)
    elif problem == "missing":
        del arrays["pressure"]
    elif problem == "nan":
        arrays["pressure"][0] = np.nan
    elif problem == "shape":
        arrays["internal_points"] = np.ones((2, 2))
    else:
        arrays["pressure"] = np.array([object()], dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        w.validate_prediction(path, (3, 2))


@pytest.mark.parametrize(
    "kind,name",
    [("file", "../escape"), ("file", "/absolute"), ("link", "link"), ("fifo", "fifo")],
)
def test_unsafe_archives(tmp_path, kind, name):
    archive = tmp_path / "test.tar"
    with tarfile.open(archive, "w") as target:
        member = tarfile.TarInfo(name)
        if kind == "link":
            member.type, member.linkname = tarfile.SYMTYPE, "outside"
        elif kind == "fifo":
            member.type = tarfile.FIFOTYPE
        target.addfile(member)
    with pytest.raises(w.WorkflowError):
        w.unpack(archive, tmp_path / "output")


def test_archive_never_overwrites(tmp_path):
    archive = tmp_path / "test.tar"
    with tarfile.open(archive, "w") as target:
        for _ in range(2):
            target.addfile(tarfile.TarInfo("same"))
    with pytest.raises(FileExistsError):
        w.unpack(archive, tmp_path / "output")


def test_download_plan_is_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        w.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network accessed")
    )
    result = w.fetch_support(tmp_path / "support", dry_run=True)
    assert result["total_bytes"] == 1529509241
    assert not (tmp_path / "support").exists()


def test_offline_download_fails_without_installation(tmp_path):
    with pytest.raises(w.WorkflowError, match="offline support archive missing"):
        w.fetch_support(tmp_path / "support", offline=True)
    assert not (tmp_path / "support/installed").exists()


def campaign_fixture(tmp_path, monkeypatch):
    entry_root, data, support, output = [
        tmp_path / s for s in ("entry", "data", "support", "work")
    ]
    (entry_root / "predictions").mkdir(parents=True)
    ids = ["case_a", "case_b"]
    entry = {
        "schema": "fluidsbench-airfrans-entry-v1",
        "submission_id": "test",
        "split_id": "full",
    }
    w.write_json(entry_root / "entry.json", entry)
    inputs, truths = {}, []
    for cid in ids:
        pred = entry_root / "predictions" / f"{cid}.npz"
        prediction(pred)
        mesh_hashes = {}
        for name, path in w.mesh_paths(data, cid).items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            mesh_hashes[name] = w.sha256_file(path)
        truth = support / "installed/profiles" / f"{cid}.json"
        w.write_json(
            truth,
            {
                "cases": [{"case_id": cid, "series": []}],
                "provenance": {"input_hashes": mesh_hashes},
            },
        )
        truths.append(truth)
        inputs[cid] = {
            "prediction_sha256": w.sha256_file(pred),
            "meshes": mesh_hashes,
            "truth_sha256": w.sha256_file(truth),
        }
    campaign = {
        "schema": "fluidsbench-airfrans-campaign-v1",
        "entry": entry,
        "case_ids": ids,
        "case_set_id": "standard",
        "inputs": inputs,
        "bindings": w.bindings(),
        "evaluator": w.evaluator_identity(),
    }
    monkeypatch.setattr(
        w, "official_split", lambda _: {"case_ids": ids, "case_set_id": "standard"}
    )
    monkeypatch.setattr(w, "verify_support", lambda _: {})
    monkeypatch.setattr(w, "resolve_case_set_truth", lambda *a: truths)
    instances = [
        {"support_id": k, "entity_count": n}
        for k, n in (("two-dimensional-domain-native", 3), ("airfoil-curve-native", 2))
    ]
    monkeypatch.setattr(
        w,
        "load_support_release",
        lambda *a: SimpleNamespace(
            cases={cid: {"support_instances": instances} for cid in ids}
        ),
    )
    monkeypatch.setattr(w, "check_runtime", lambda: {"test": "synthetic"})
    calls = []

    def worker(args, **kwargs):
        cid = args[args.index("--case-id") + 1]
        stage = Path(args[args.index("--output") + 1])
        calls.append(cid)
        w.write_json(stage / "case.json", {"case_id": cid, "files": []})
        w.write_json(
            stage / "profiles.json", {"cases": [{"case_id": cid, "series": []}]}
        )

    def reducer(output, campaign, support):
        root = output / "evaluation"
        root.mkdir()
        w.write_json(root / "metrics-cases.json", {"metric_values": {"example": 1}})
        w.write_json(
            root / "workflow.json",
            {
                "schema": "fluidsbench-airfrans-evaluation-v1",
                "campaign": campaign,
                "files": w.file_tree(root),
                "case_checkpoints": {
                    cid: w.sha256_file(output / "cases" / cid / "checkpoint.json")
                    for cid in ids
                },
            },
        )

    monkeypatch.setattr(w.subprocess, "run", worker)
    monkeypatch.setattr(w, "reduce_campaign", reducer)
    return SimpleNamespace(
        entry=entry_root,
        data=data,
        support=support,
        output=output,
        calls=calls,
        worker=worker,
        reducer=reducer,
        campaign=campaign,
    )


def test_preflight_checks_native_mesh_hashes(tmp_path, monkeypatch):
    f = campaign_fixture(tmp_path, monkeypatch)
    assert w.preflight(f.entry, f.data, f.support) == f.campaign
    w.mesh_paths(f.data, "case_a")["internal_vtu"].write_text("changed")
    with pytest.raises(w.WorkflowError, match="native mesh missing or differs"):
        w.preflight(f.entry, f.data, f.support)


@pytest.mark.parametrize("problem", ["extra", "missing", "link"])
def test_exact_prediction_inventory(tmp_path, monkeypatch, problem):
    f = campaign_fixture(tmp_path, monkeypatch)
    p = f.entry / "predictions/case_a.npz"
    if problem == "extra":
        (f.entry / "predictions/direct-force.json").write_text("{}")
    elif problem == "missing":
        p.unlink()
    else:
        p.unlink()
        p.symlink_to(f.entry / "predictions/case_b.npz")
    with pytest.raises(w.WorkflowError):
        w.preflight(f.entry, f.data, f.support)


def test_interrupted_case_resumes_without_repeating_completed_cases(
    tmp_path, monkeypatch
):
    f = campaign_fixture(tmp_path, monkeypatch)

    def fail_second(args, **kwargs):
        if args[args.index("--case-id") + 1] == "case_b":
            raise subprocess.CalledProcessError(1, args)
        f.worker(args, **kwargs)

    monkeypatch.setattr(w.subprocess, "run", fail_second)
    with pytest.raises(subprocess.CalledProcessError):
        w.evaluate_entry(f.entry, f.data, f.support, f.output)
    assert (f.output / "cases/case_a/checkpoint.json").exists()
    assert not (f.output / "cases/case_b").exists()
    monkeypatch.setattr(w.subprocess, "run", f.worker)
    w.evaluate_entry(f.entry, f.data, f.support, f.output, resume=True)
    assert f.calls == ["case_a", "case_b"]
    assert (
        w.evaluate_entry(f.entry, f.data, f.support, f.output, resume=True)["status"]
        == "verified_completed_evaluation"
    )
    assert f.calls == ["case_a", "case_b"]


def test_interrupted_reduction_reuses_all_cases(tmp_path, monkeypatch):
    f = campaign_fixture(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("interrupted reduction")

    monkeypatch.setattr(w, "reduce_campaign", fail)
    with pytest.raises(RuntimeError):
        w.evaluate_entry(f.entry, f.data, f.support, f.output)
    monkeypatch.setattr(w, "reduce_campaign", f.reducer)
    w.evaluate_entry(f.entry, f.data, f.support, f.output, resume=True)
    assert f.calls == ["case_a", "case_b"]


@pytest.mark.parametrize("change", ["predictions", "case", "evaluation", "code"])
def test_resume_rejects_changed_inputs_and_outputs(tmp_path, monkeypatch, change):
    f = campaign_fixture(tmp_path, monkeypatch)
    w.evaluate_entry(f.entry, f.data, f.support, f.output)
    if change == "predictions":
        p = f.entry / "predictions/case_a.npz"
        with np.load(p) as source:
            arrays = dict(source)
        arrays["velocity"] += 1
        np.savez(p, **arrays)
    elif change == "case":
        (f.output / "cases/case_a/case.json").write_text("{}")
    elif change == "evaluation":
        (f.output / "evaluation/metrics-cases.json").write_text("{}")
    else:
        monkeypatch.setattr(w, "evaluator_identity", lambda: {"different": "evaluator"})
    with pytest.raises(w.WorkflowError):
        w.evaluate_entry(f.entry, f.data, f.support, f.output, resume=True)


def test_concurrent_campaign_is_rejected(tmp_path):
    with w.campaign_lock(tmp_path):
        with pytest.raises(w.WorkflowError, match="another evaluation"):
            with w.campaign_lock(tmp_path):
                pass
