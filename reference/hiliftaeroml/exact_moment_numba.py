# SPDX-License-Identifier: Apache-2.0
"""Direct-cell Numba kernel for the production HiLift exact-moment audit.

This optional campaign backend is intentionally separate from the portable
NumPy reference implementation.  It never materializes the roughly 0.5
trillion ordered-fan triangles in the full 1,800-case campaign and keeps the
three wall-shear components as separate read-only memmaps.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from reference.hiliftaeroml.exact_moment_audit import (
    ExactMomentAuditError,
    TriangleBatchIntegrals,
)

# Float statistic layout returned by the compiled kernel.
_AREA = 0
_ORIENTED = slice(1, 4)
_PRESSURE_FORCE = slice(4, 7)
_VISCOUS_FORCE = slice(7, 10)
_PRESSURE_EXACT = slice(10, 13)
_VISCOUS_EXACT = slice(13, 16)
_PRESSURE_VERTEX = slice(16, 19)
_VISCOUS_VERTEX = slice(19, 22)
_STAT_COUNT = 22


@njit(cache=True, fastmath=False, nogil=True)
def _integrate_native_cell_range(
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    pressure: np.ndarray,
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    tau_z: np.ndarray,
    reference: np.ndarray,
    p_inf: float,
    q_ref: float,
    begin_cell: int,
    end_cell: int,
) -> tuple[int, int, int, int, np.ndarray]:
    """Return exact sufficient statistics for a contiguous native cell range."""

    statistics = np.zeros(_STAT_COUNT, dtype=np.float64)
    triangle_count = 0
    zero_area_count = 0
    nonfinite_count = 0
    invalid_connectivity_count = 0
    n_points = points.shape[0]
    one_third = 1.0 / 3.0
    one_twelfth = 1.0 / 12.0

    for cell_id in range(begin_cell, end_cell):
        begin = int(offsets[cell_id])
        end = int(offsets[cell_id + 1])
        if end - begin < 3:
            invalid_connectivity_count += 1
            continue
        point_a = int(connectivity[begin])
        if point_a < 0 or point_a >= n_points:
            invalid_connectivity_count += 1
            continue
        for local_j in range(1, end - begin - 1):
            point_b = int(connectivity[begin + local_j])
            point_c = int(connectivity[begin + local_j + 1])
            triangle_count += 1
            if point_b < 0 or point_b >= n_points or point_c < 0 or point_c >= n_points:
                invalid_connectivity_count += 1
                continue

            ax = np.float64(points[point_a, 0])
            ay = np.float64(points[point_a, 1])
            az = np.float64(points[point_a, 2])
            bx = np.float64(points[point_b, 0])
            by = np.float64(points[point_b, 1])
            bz = np.float64(points[point_b, 2])
            cx = np.float64(points[point_c, 0])
            cy = np.float64(points[point_c, 1])
            cz = np.float64(points[point_c, 2])
            abx = bx - ax
            aby = by - ay
            abz = bz - az
            acx = cx - ax
            acy = cy - ay
            acz = cz - az
            oriented_x = 0.5 * (aby * acz - abz * acy)
            oriented_y = 0.5 * (abz * acx - abx * acz)
            oriented_z = 0.5 * (abx * acy - aby * acx)
            area = math.sqrt(
                oriented_x * oriented_x
                + oriented_y * oriented_y
                + oriented_z * oriented_z
            )

            cp_a = (np.float64(pressure[point_a]) - p_inf) / q_ref
            cp_b = (np.float64(pressure[point_b]) - p_inf) / q_ref
            cp_c = (np.float64(pressure[point_c]) - p_inf) / q_ref
            tx_a = np.float64(tau_x[point_a]) / q_ref
            ty_a = np.float64(tau_y[point_a]) / q_ref
            tz_a = np.float64(tau_z[point_a]) / q_ref
            tx_b = np.float64(tau_x[point_b]) / q_ref
            ty_b = np.float64(tau_y[point_b]) / q_ref
            tz_b = np.float64(tau_z[point_b]) / q_ref
            tx_c = np.float64(tau_x[point_c]) / q_ref
            ty_c = np.float64(tau_y[point_c]) / q_ref
            tz_c = np.float64(tau_z[point_c]) / q_ref
            if not (
                math.isfinite(ax)
                and math.isfinite(ay)
                and math.isfinite(az)
                and math.isfinite(bx)
                and math.isfinite(by)
                and math.isfinite(bz)
                and math.isfinite(cx)
                and math.isfinite(cy)
                and math.isfinite(cz)
                and math.isfinite(area)
                and math.isfinite(cp_a)
                and math.isfinite(cp_b)
                and math.isfinite(cp_c)
                and math.isfinite(tx_a)
                and math.isfinite(ty_a)
                and math.isfinite(tz_a)
                and math.isfinite(tx_b)
                and math.isfinite(ty_b)
                and math.isfinite(tz_b)
                and math.isfinite(tx_c)
                and math.isfinite(ty_c)
                and math.isfinite(tz_c)
            ):
                nonfinite_count += 1
                continue
            if area == 0.0:
                zero_area_count += 1

            statistics[_AREA] += area
            statistics[1] += oriented_x
            statistics[2] += oriented_y
            statistics[3] += oriented_z
            cp_sum_third = (cp_a + cp_b + cp_c) * one_third
            statistics[4] += cp_sum_third * oriented_x
            statistics[5] += cp_sum_third * oriented_y
            statistics[6] += cp_sum_third * oriented_z
            statistics[7] += area * (tx_a + tx_b + tx_c) * one_third
            statistics[8] += area * (ty_a + ty_b + ty_c) * one_third
            statistics[9] += area * (tz_a + tz_b + tz_c) * one_third

            ra_x = ax - reference[0]
            ra_y = ay - reference[1]
            ra_z = az - reference[2]
            rb_x = bx - reference[0]
            rb_y = by - reference[1]
            rb_z = bz - reference[2]
            rc_x = cx - reference[0]
            rc_y = cy - reference[1]
            rc_z = cz - reference[2]
            sum_x = ra_x + rb_x + rc_x
            sum_y = ra_y + rb_y + rc_y
            sum_z = ra_z + rb_z + rc_z

            # Each traction node j uses (sum_i r_i + r_j)/12.  Keeping the
            # node loop explicit avoids any triangle or traction allocation.
            for node in range(3):
                if node == 0:
                    rx = ra_x
                    ry = ra_y
                    rz = ra_z
                    cp_value = cp_a
                    tx = tx_a
                    ty = ty_a
                    tz = tz_a
                elif node == 1:
                    rx = rb_x
                    ry = rb_y
                    rz = rb_z
                    cp_value = cp_b
                    tx = tx_b
                    ty = ty_b
                    tz = tz_b
                else:
                    rx = rc_x
                    ry = rc_y
                    rz = rc_z
                    cp_value = cp_c
                    tx = tx_c
                    ty = ty_c
                    tz = tz_c

                wx = (sum_x + rx) * one_twelfth
                wy = (sum_y + ry) * one_twelfth
                wz = (sum_z + rz) * one_twelfth
                pfx = cp_value * oriented_x
                pfy = cp_value * oriented_y
                pfz = cp_value * oriented_z
                statistics[10] += wy * pfz - wz * pfy
                statistics[11] += wz * pfx - wx * pfz
                statistics[12] += wx * pfy - wy * pfx
                statistics[13] += area * (wy * tz - wz * ty)
                statistics[14] += area * (wz * tx - wx * tz)
                statistics[15] += area * (wx * ty - wy * tx)

                vertex_scale = one_third
                statistics[16] += vertex_scale * (ry * pfz - rz * pfy)
                statistics[17] += vertex_scale * (rz * pfx - rx * pfz)
                statistics[18] += vertex_scale * (rx * pfy - ry * pfx)
                statistics[19] += area * vertex_scale * (ry * tz - rz * ty)
                statistics[20] += area * vertex_scale * (rz * tx - rx * tz)
                statistics[21] += area * vertex_scale * (rx * ty - ry * tx)

    return (
        triangle_count,
        zero_area_count,
        nonfinite_count,
        invalid_connectivity_count,
        statistics,
    )


def integrate_native_cell_range(
    *,
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    pressure: np.ndarray,
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    tau_z: np.ndarray,
    moment_reference: np.ndarray,
    p_inf: float,
    q_ref: float,
    begin_cell: int,
    end_cell: int,
) -> TriangleBatchIntegrals:
    """Validated Python boundary around the direct compiled cell kernel."""

    if not math.isfinite(p_inf):
        raise ExactMomentAuditError("p_inf must be finite")
    if not math.isfinite(q_ref) or q_ref <= 0.0:
        raise ExactMomentAuditError("q_ref must be finite and positive")
    reference = np.asarray(moment_reference, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ExactMomentAuditError("moment reference must contain three finite values")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ExactMomentAuditError("points must have shape (N, 3)")
    point_count = points.shape[0]
    for name, value in (
        ("pressure", pressure),
        ("tau_x", tau_x),
        ("tau_y", tau_y),
        ("tau_z", tau_z),
    ):
        if value.ndim != 1 or value.shape[0] != point_count:
            raise ExactMomentAuditError(f"{name} must be a point vector")
    if connectivity.ndim != 1 or offsets.ndim != 1:
        raise ExactMomentAuditError("connectivity and offsets must be vectors")
    if not (0 <= begin_cell <= end_cell <= offsets.size - 1):
        raise ExactMomentAuditError("cell range is invalid")
    triangle_count, zero_count, nonfinite_count, invalid_count, statistics = (
        _integrate_native_cell_range(
            points,
            connectivity,
            offsets,
            pressure,
            tau_x,
            tau_y,
            tau_z,
            reference,
            float(p_inf),
            float(q_ref),
            begin_cell,
            end_cell,
        )
    )
    if nonfinite_count:
        raise ExactMomentAuditError(
            f"direct kernel found {nonfinite_count} non-finite fan triangles"
        )
    if invalid_count:
        raise ExactMomentAuditError(
            f"direct kernel found {invalid_count} invalid cells/triangles"
        )
    if statistics.shape != (_STAT_COUNT,) or not np.all(np.isfinite(statistics)):
        raise ExactMomentAuditError("direct kernel returned invalid statistics")
    return TriangleBatchIntegrals(
        triangle_count=int(triangle_count),
        zero_area_triangle_count=int(zero_count),
        area_sum=float(statistics[_AREA]),
        oriented_area_sum=statistics[_ORIENTED].copy(),
        pressure_force=statistics[_PRESSURE_FORCE].copy(),
        viscous_force=statistics[_VISCOUS_FORCE].copy(),
        pressure_moment_exact=statistics[_PRESSURE_EXACT].copy(),
        viscous_moment_exact=statistics[_VISCOUS_EXACT].copy(),
        pressure_moment_vertex_lumped=statistics[_PRESSURE_VERTEX].copy(),
        viscous_moment_vertex_lumped=statistics[_VISCOUS_VERTEX].copy(),
    )


def warmup_and_self_check() -> dict[str, object]:
    """Compile the production signature and verify one analytic constant case."""

    points = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    connectivity = np.asarray([0, 1, 2], dtype=np.int64)
    offsets = np.asarray([0, 3], dtype=np.int64)
    pressure = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    tau_x = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    tau_y = np.zeros(3, dtype=np.float32)
    tau_z = np.zeros(3, dtype=np.float32)
    for field in (pressure, tau_x, tau_y, tau_z):
        field.setflags(write=False)
    reference = np.zeros(3, dtype=np.float64)
    result = integrate_native_cell_range(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure=pressure,
        tau_x=tau_x,
        tau_y=tau_y,
        tau_z=tau_z,
        moment_reference=reference,
        p_inf=1.0,
        q_ref=1.0,
        begin_cell=0,
        end_cell=1,
    )
    if result.triangle_count != 1 or not np.allclose(
        result.pressure_force, np.asarray([0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-15
    ):
        raise ExactMomentAuditError("direct Numba kernel self-check failed")
    return {
        "status": "pass",
        "points_dtype": points.dtype.str,
        "connectivity_dtype": connectivity.dtype.str,
        "offsets_dtype": offsets.dtype.str,
        "point_field_dtype": pressure.dtype.str,
        "point_fields_read_only": not pressure.flags.writeable,
        "moment_reference_dtype": reference.dtype.str,
        "triangle_count": result.triangle_count,
    }


__all__ = ["integrate_native_cell_range", "warmup_and_self_check"]
