"""Run the real field, force and profile reducers on tiny synthetic native tables."""

import copy

import numpy as np
import pytest

from reference.airfrans import participant as w
from scripts.build_airfrans_scoring_support import build_manifest


def test_real_reduction_and_atomic_output(tmp_path, monkeypatch):
    ids = ["synthetic_a", "synthetic_b"]
    release_id = "synthetic-airfrans-test-v1"
    case_set = "standard"
    support_root = tmp_path / "support"
    native = support_root / "native"
    output = tmp_path / "campaign"
    manifest = build_manifest(2, case_set, release_id)
    identity = {
        "schema_version": "1.0",
        "dataset_id": "airfrans",
        "release_id": release_id,
        "case_set_id": case_set,
    }
    chunk = {**identity, "cases": []}
    inputs, truth_paths = {}, []
    definition = w.load_json(w.PROFILE_PATH)
    for ordinal, cid in enumerate(ids):
        instances, predictions = [], []
        case_root = output / "cases" / cid
        for support in manifest["supports"]:
            is_force = support["id"] == "airfoil-force-coefficients-native"
            size, suffix = (1, "json") if is_force else (3, "npz")
            columns = {
                "support_id": np.array([f"{n:07d}" for n in range(size)]),
                "weight": np.ones(size),
            }
            for field in support["location_definition"]["coordinate_fields"]:
                columns[field] = np.arange(size, dtype=float)
            pred = {"support_id": columns["support_id"]}
            for quantity in support["quantities"]:
                for component in quantity["components"]:
                    values = np.arange(size, dtype=float) + ordinal + 1
                    columns[component["target_field"]] = values
                    pred[component["prediction_field"]] = values.copy()
            relative = f"data/{cid}-{support['id']}.{suffix}"
            truth = native / "case-sets/standard" / relative
            predicted = case_root / relative
            truth.parent.mkdir(parents=True, exist_ok=True)
            predicted.parent.mkdir(parents=True, exist_ok=True)
            if is_force:
                w.write_json(
                    truth, {"columns": {k: v.tolist() for k, v in columns.items()}}
                )
                w.write_json(
                    predicted, {"columns": {k: v.tolist() for k, v in pred.items()}}
                )
            else:
                np.savez(truth, **columns)
                np.savez(predicted, **pred)
            instances.append(
                {
                    "support_id": support["id"],
                    "entity_count": size,
                    "artifacts": [
                        {
                            "role": "ground_truth_table",
                            "path": relative,
                            "sha256": w.sha256_file(truth),
                            "format": suffix,
                        }
                    ],
                }
            )
            predictions.append(
                {
                    "support_id": support["id"],
                    "file": relative,
                    "sha256": w.sha256_file(predicted),
                    "format": suffix,
                    "row_count": size,
                }
            )
        chunk["cases"].append({"case_id": cid, "support_instances": instances})
        w.write_json(case_root / "case.json", {"case_id": cid, "files": predictions})
        x = np.linspace(0, 0.1, 1001)
        series = [
            {
                "panel_id": definition["metric_binding"]["panel_id"],
                "station_id": station["id"],
                "quantity_id": quantity["id"],
                "coordinate": x.tolist(),
                "prediction": (x + ordinal + 1).tolist(),
            }
            for station in definition["stations"]
            for quantity in definition["quantities"]
        ]
        profiles = {"cases": [{"case_id": cid, "series": series}]}
        w.write_json(case_root / "profiles.json", profiles)
        path = support_root / "profiles" / f"{cid}.json"
        w.write_json(path, profiles)
        truth_paths.append(path)
        inputs[cid] = {"truth_sha256": w.sha256_file(path)}
    chunk_path = native / "case-sets/standard/chunk-000.json"
    w.write_json(chunk_path, chunk)
    index = {
        **identity,
        "case_count": 2,
        "chunks": [
            {
                "file": "chunk-000.json",
                "case_count": 2,
                "case_ids": ids,
                "sha256": w.sha256_file(chunk_path),
            }
        ],
    }
    w.write_json(native / "case-sets/standard/index.json", index)
    manifest["case_sets"][0]["index_sha256"] = w.sha256_file(
        native / "case-sets/standard/index.json"
    )
    w.write_json(native / "manifest.json", manifest)
    campaign = {
        "entry": {"submission_id": "synthetic-test", "split_id": "full"},
        "case_ids": ids,
        "case_set_id": case_set,
        "inputs": inputs,
        "bindings": {"native": w.sha256_file(native / "manifest.json")},
    }
    for cid in ids:
        path = output / "cases" / cid
        w.write_json(
            path / "checkpoint.json",
            {
                "campaign_sha256": w.digest(campaign),
                "input": inputs[cid],
                "files": w.file_tree(path),
            },
        )
    monkeypatch.setattr(w, "resolve_case_set_truth", lambda *a: truth_paths)
    w.reduce_campaign(output, campaign, support_root)
    result = w.load_json(output / "evaluation/metrics-cases.json")
    assert result["case_count"] == 2
    assert result["metric_values"]["overall_score"] == pytest.approx(100)
    assert result["metric_values"]["velocity_profile_r2"] == pytest.approx(1)
    assert result["metric_values"]["cd_r2"] == pytest.approx(1)
    assert result["metric_values"]["surface_pressure_rel_l2"] == pytest.approx(0)
    assert (
        len(w.load_json(output / "evaluation/prediction-manifest.json")["cases"]) == 2
    )
    assert not list(output.glob(".reduce-*"))
