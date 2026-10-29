"""Minimal dual-weight and chunk-safe relative-L2 example."""

from __future__ import annotations

import numpy as np

from reference.evaluate_predictions import (
    relative_l2_from_sufficient_statistics,
    relative_l2_sufficient_statistics,
)


ground_truth = np.array([101325.0, 100980.0, 100410.0])
prediction = np.array([101300.0, 101020.0, 100500.0])
benchmark_face_areas = np.array([0.10, 0.15, 0.08])

uniform_statistics = relative_l2_sufficient_statistics(
    ground_truth[:, None],
    prediction[:, None],
    benchmark_face_areas,
    weighting="uniform",
)
area_statistics = relative_l2_sufficient_statistics(
    ground_truth[:, None],
    prediction[:, None],
    benchmark_face_areas,
    weighting="support_weights",
)

print(
    "Equal-face relative L2: "
    f"{relative_l2_from_sufficient_statistics([uniform_statistics]):.6f}%"
)
print(
    "Face-area-weighted relative L2: "
    f"{relative_l2_from_sufficient_statistics([area_statistics]):.6f}%"
)

# Chunks retain additive numerators and denominators; chunk L2 values are not averaged.
chunk_statistics = [
    relative_l2_sufficient_statistics(
        ground_truth[index, None],
        prediction[index, None],
        benchmark_face_areas[index],
        weighting="support_weights",
    )
    for index in (slice(0, 2), slice(2, 3))
]
print(
    "Chunked face-area-weighted relative L2: "
    f"{relative_l2_from_sufficient_statistics(chunk_statistics):.6f}%"
)
print(f"Chunk sufficient statistics: {chunk_statistics}")
