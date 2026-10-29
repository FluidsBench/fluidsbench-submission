from __future__ import annotations
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from reference.drivaerml.accumulators import StreamingFieldAccumulator
from reference.drivaerml.regional_diagnostics import (
    RegionalFieldAccumulator,
    SURFACE_REGION_DEFINITION,
    RegionalDiagnosticError,
    validate_regional_field_report,
)
from reference.drivaerml import dev_profiles as profiles


def _global_statistics(truth, prediction, weights, chunks):
    accumulator = StreamingFieldAccumulator(
        len(truth), component_count=1 if truth.ndim == 1 else truth.shape[1]
    )
    for start, stop in chunks:
        accumulator.add_chunk(
            np.arange(start, stop, dtype=np.int64),
            truth[start:stop],
            prediction[start:stop],
            weights[start:stop],
        )
    return accumulator.finalize()


@pytest.mark.parametrize("physical_weight", [1.0, 0.3])
@pytest.mark.parametrize("component_labels", [("p",), ("x", "y", "z")])
def test_million_row_regions_reconstruct_without_relaxing_tolerance(
    component_labels: tuple[str, ...],
    physical_weight: float,
) -> None:
    # A supported-size chunk with repeated nonbinary contributions exposes
    # sequential summation drift, even though every input and region is valid.
    count = 1_000_000
    codes = np.zeros(count, dtype=np.uint8)
    codes[-3:] = [1, 2, 3]
    if len(component_labels) == 1:
        truth = np.full(count, 4.0)
        prediction = truth + math.sqrt(0.6)
    else:
        truth = np.tile([4.0, 3.0, 0.0], (count, 1))
        prediction = truth + np.sqrt([0.2, 0.2, 0.6])
    weights = np.full(count, physical_weight)
    global_statistics = _global_statistics(truth, prediction, weights, ((0, count),))
    accumulator = RegionalFieldAccumulator(
        SURFACE_REGION_DEFINITION, count, component_labels
    )
    accumulator.add_chunk(0, codes, truth, prediction, weights)
    report = accumulator.finalize(
        expected_uniform=global_statistics.uniform,
        expected_physical=global_statistics.physical,
    )

    assert [row["entity_count"] for row in report["regions"]] == [count - 3, 1, 1, 1]
    reconstruction = report["reconstruction"]
    assert reconstruction["tolerance"] == {"relative": 5.0e-12, "absolute": 1.0e-12}
    assert reconstruction["expected_uniform"] == asdict(global_statistics.uniform)
    assert reconstruction["expected_physical"] == asdict(global_statistics.physical)
    validate_regional_field_report(report, SURFACE_REGION_DEFINITION)

    altered = copy.deepcopy(report)
    altered["regions"][0]["equal_entity"]["component_squared_error"][
        component_labels[0]
    ] *= 1.01
    with pytest.raises(
        RegionalDiagnosticError, match="region.equal_entity.component_squared_error"
    ):
        validate_regional_field_report(altered, SURFACE_REGION_DEFINITION)


@pytest.mark.parametrize("scope", ["surface_only", "surface_and_volume"])
def test_native_run419_replays_autocfd5_profiles(monkeypatch, scope):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/drivaerml-autocfd5-run419-golden.json"
        ).read_text()
    )
    truth = {"series": fixture["support_projection"]["series"]}
    monkeypatch.setattr(profiles, "truth_case", lambda cid: truth)
    by_key = {
        (r["family_id"], r["station_id"]): r
        for r in fixture["profile_prediction_series"]
    }
    predictions = [
        profiles.prediction_series(
            s, by_key[(s["family_id"], s["station_id"])].get("prediction")
        )
        for s in profiles.selected_series(truth, scope)
    ]
    statistics, rmses = profiles.compare_profiles("run_419", predictions, scope)
    # These expected values are independently frozen by AutoCFD5's own golden test.
    expected = {"drivaerml_cp_constant_v1": (0.9955028959702529, 0.015373708203805573)}
    if scope == "surface_and_volume":
        expected["drivaerml-autocfd5-constant-v1"] = (
            0.9957833983449247,
            0.012722685649733181,
        )
    for family, (r2, rmse) in expected.items():
        rows = statistics[family]
        actual = 1 - math.fsum(r[0] for r in rows) / (
            math.fsum(r[2] for r in rows)
            - math.fsum(r[1] for r in rows) ** 2 / len(rows)
        )
        assert actual == pytest.approx(r2, abs=2e-14, rel=0)
        assert math.fsum(v for _, v in rmses[family]) / len(rows) == pytest.approx(
            rmse, abs=2e-14, rel=0
        )


def test_cp_weights_do_not_bridge_gaps_or_add_an_origin_edge():
    series = {
        "panel_id": "pressure_profiles",
        "coordinate": [5.0, 7.0, 20.0, 24.0],
        "segments": [
            {"emitted_index_start": 0, "emitted_index_stop": 2},
            {"emitted_index_start": 2, "emitted_index_stop": 4},
        ],
    }
    assert profiles.weights(series) == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 3])
    series["coordinate"][1] = 4.0
    with pytest.raises(ValueError, match="increase"):
        profiles.weights(series)
