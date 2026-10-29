from __future__ import annotations

import unittest

import numpy as np
import pytest

try:
    import numba  # noqa: F401
except ImportError:
    raise unittest.SkipTest("requires an importable numba") from None

from reference.hiliftaeroml.exact_moment_audit import (
    ExactMomentAuditError,
    integrate_ordered_fan_mesh,
)
from reference.hiliftaeroml.exact_moment_numba import (
    integrate_native_cell_range,
    warmup_and_self_check,
)


def _inputs():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.3],
            [0.0, 1.0, 0.1],
            [-0.4, 0.3, 1.2],
        ],
        dtype=np.float64,
    )
    connectivity = np.asarray([0, 1, 2, 3, 0, 3, 4], dtype=np.int64)
    offsets = np.asarray([0, 4, 7], dtype=np.int64)
    pressure = np.asarray([2.1, 2.8, 1.6, 3.0, 1.9], dtype=np.float32)
    tau = (
        np.asarray([0.1, 0.4, -0.3, 0.9, 0.2], dtype=np.float32),
        np.asarray([0.2, -0.2, 0.7, 0.2, -0.4], dtype=np.float32),
        np.asarray([0.3, 0.8, 0.1, -0.5, 0.6], dtype=np.float32),
    )
    reference = np.asarray([0.1, 0.2, -0.3], dtype=np.float64)
    return points, connectivity, offsets, pressure, tau, reference


def test_direct_kernel_matches_numpy_reference() -> None:
    points, connectivity, offsets, pressure, tau, reference = _inputs()
    p_inf = 1.2
    q_ref = 2.5
    observed = integrate_native_cell_range(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure=pressure,
        tau_x=tau[0],
        tau_y=tau[1],
        tau_z=tau[2],
        moment_reference=reference,
        p_inf=p_inf,
        q_ref=q_ref,
        begin_cell=0,
        end_cell=2,
    )
    expected = integrate_ordered_fan_mesh(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure_coefficient=(pressure.astype(np.float64) - p_inf) / q_ref,
        wall_shear_coefficient=np.column_stack(tau).astype(np.float64) / q_ref,
        moment_reference=reference,
        cell_chunk=2,
    )
    assert observed.triangle_count == expected.triangle_count
    assert observed.zero_area_triangle_count == expected.zero_area_triangle_count
    assert observed.area_sum == pytest.approx(expected.area_sum, rel=3e-15, abs=3e-15)
    for left, right in zip(observed.vectors(), expected.vectors(), strict=True):
        np.testing.assert_allclose(left, right, rtol=5e-15, atol=5e-15)


def test_direct_kernel_chunk_composition_and_self_check() -> None:
    warmup_and_self_check()
    points, connectivity, offsets, pressure, tau, reference = _inputs()
    pieces = [
        integrate_native_cell_range(
            points=points,
            connectivity=connectivity,
            offsets=offsets,
            pressure=pressure,
            tau_x=tau[0],
            tau_y=tau[1],
            tau_z=tau[2],
            moment_reference=reference,
            p_inf=1.2,
            q_ref=2.5,
            begin_cell=index,
            end_cell=index + 1,
        )
        for index in range(2)
    ]
    combined = integrate_native_cell_range(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure=pressure,
        tau_x=tau[0],
        tau_y=tau[1],
        tau_z=tau[2],
        moment_reference=reference,
        p_inf=1.2,
        q_ref=2.5,
        begin_cell=0,
        end_cell=2,
    )
    for name in (
        "area_sum",
        "oriented_area_sum",
        "pressure_force",
        "viscous_force",
        "pressure_moment_exact",
        "viscous_moment_exact",
        "pressure_moment_vertex_lumped",
        "viscous_moment_vertex_lumped",
    ):
        np.testing.assert_allclose(
            getattr(combined, name),
            getattr(pieces[0], name) + getattr(pieces[1], name),
            rtol=5e-15,
            atol=5e-15,
        )


def test_direct_kernel_invalid_inputs_fail_closed() -> None:
    points, connectivity, offsets, pressure, tau, reference = _inputs()
    with pytest.raises(ExactMomentAuditError, match="q_ref"):
        integrate_native_cell_range(
            points=points,
            connectivity=connectivity,
            offsets=offsets,
            pressure=pressure,
            tau_x=tau[0],
            tau_y=tau[1],
            tau_z=tau[2],
            moment_reference=reference,
            p_inf=1.2,
            q_ref=0.0,
            begin_cell=0,
            end_cell=2,
        )
    bad = pressure.copy()
    bad[1] = np.nan
    with pytest.raises(ExactMomentAuditError, match="non-finite"):
        integrate_native_cell_range(
            points=points,
            connectivity=connectivity,
            offsets=offsets,
            pressure=bad,
            tau_x=tau[0],
            tau_y=tau[1],
            tau_z=tau[2],
            moment_reference=reference,
            p_inf=1.2,
            q_ref=2.5,
            begin_cell=0,
            end_cell=2,
        )
