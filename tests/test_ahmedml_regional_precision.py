"""Supported chunk sizes must not create false regional reconstruction failures."""

import math
import numpy as np
import pytest
from reference.ahmedml.evaluator import _RegionalSums, AhmedMLCandidateEvaluatorError
from reference.drivaerml.accumulators import StreamingFieldAccumulator


@pytest.mark.parametrize("components", [1, 3])
@pytest.mark.parametrize("weight", [1.0, 0.3])
@pytest.mark.parametrize("regions", [3, 4])
def test_million_row_reconstruction_keeps_existing_tolerance(
    components, weight, regions
):
    n = 1_000_000
    codes = np.zeros(n, dtype=np.uint8)
    codes[-regions + 1 :] = np.arange(1, regions)
    truth = np.full(n, 4.0) if components == 1 else np.tile([4.0, 3.0, 0.0], (n, 1))
    prediction = truth + (
        math.sqrt(0.6) if components == 1 else np.sqrt([0.2, 0.2, 0.6])
    )
    weights = np.full(n, weight)
    whole = StreamingFieldAccumulator(n, component_count=components)
    whole.add_chunk(np.arange(n, dtype=np.int64), truth, prediction, weights)
    expected = whole.finalize()
    accumulator = _RegionalSums(
        components, tuple(str(i) for i in range(regions)), "test"
    )
    accumulator.add(codes, truth, prediction, weights)
    report = accumulator.finalize(expected.uniform, expected.physical)
    assert sum(r["entity_count"] for r in report.values()) == n
    for weighting in ("uniform", "physical"):
        observed = sum(r[weighting]["squared_error"] for r in report.values())
        assert math.isclose(
            observed,
            getattr(expected, weighting).squared_error,
            rel_tol=5e-12,
            abs_tol=1e-12,
        )
    accumulator.uniform_squared_error[0] *= 1.01
    with pytest.raises(AhmedMLCandidateEvaluatorError, match="do not reconstruct"):
        accumulator.finalize(expected.uniform, expected.physical)
