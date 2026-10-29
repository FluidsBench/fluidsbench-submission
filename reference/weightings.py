"""Shared dataset-weighting vocabulary for executable scoring contracts."""

from __future__ import annotations


PHYSICAL_WEIGHTINGS = frozenset(
    {
        "surface_face_area",
        "cell_volume",
        "boundary_line_length",
        "interior_cell_area",
        "surface_point_dual_area",
        "volume_point_dual_volume",
        "boundary_point_dual_length",
        "interior_point_dual_area",
    }
)


def evaluator_weighting(dataset_weighting: object) -> str:
    """Map a published dataset-weighting token to the evaluator mode."""

    return "support_weights" if dataset_weighting in PHYSICAL_WEIGHTINGS else "uniform"
