from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest
from reference.hiliftaeroml.exact_moment_audit import (
    ALGORITHM_ID,
    CASE_RECEIPT_SCHEMA,
    ExactMomentAuditError,
    body_to_wind,
    coefficient_document,
    comparison_record,
    content_fingerprint,
    integrate_ordered_fan_mesh,
    integrate_triangle_batch,
    ordered_fan_triangle_ids,
    published_force_component_closure_audit,
    validate_case_receipt,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [[0.2, -0.4, 0.7], [2.1, 0.3, 0.1], [0.4, 1.8, 1.3]],
            [[-0.5, 0.2, 0.1], [0.7, 0.1, 2.0], [1.2, 1.4, 0.6]],
        ],
        dtype=np.float64,
    )
    cp = np.asarray([[0.7, -0.2, 1.4], [-0.8, 0.4, 1.1]], dtype=np.float64)
    shear = np.asarray(
        [
            [[0.1, -0.2, 0.3], [0.7, 0.1, -0.4], [-0.3, 0.8, 0.2]],
            [[-0.2, 0.4, 0.5], [0.9, -0.1, 0.2], [0.3, 0.6, -0.7]],
        ],
        dtype=np.float64,
    )
    reference = np.asarray([0.35, -0.15, 0.25], dtype=np.float64)
    return points, cp, shear, reference


def _degree2_quadrature(
    points: np.ndarray,
    cp: np.ndarray,
    shear: np.ndarray,
    reference: np.ndarray,
) -> dict[str, np.ndarray]:
    """Independent symmetric three-point rule, exact through degree two."""

    barycentric = np.asarray(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ]
    )
    output = {
        "pressure_force": np.zeros(3),
        "viscous_force": np.zeros(3),
        "pressure_moment": np.zeros(3),
        "viscous_moment": np.zeros(3),
    }
    for xyz, nodal_cp, nodal_shear in zip(points, cp, shear, strict=True):
        oriented = 0.5 * np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        area = np.linalg.norm(oriented)
        unit_normal = oriented / area
        for bary in barycentric:
            location = bary @ xyz
            pressure_traction = float(bary @ nodal_cp) * unit_normal
            viscous_traction = bary @ nodal_shear
            weight = area / 3.0
            output["pressure_force"] += weight * pressure_traction
            output["viscous_force"] += weight * viscous_traction
            output["pressure_moment"] += weight * np.cross(
                location - reference, pressure_traction
            )
            output["viscous_moment"] += weight * np.cross(
                location - reference, viscous_traction
            )
    return output


def test_exact_kernel_matches_independent_degree2_quadrature() -> None:
    points, cp, shear, reference = _fixture()
    result = integrate_triangle_batch(
        triangle_points=points,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference,
    )
    oracle = _degree2_quadrature(points, cp, shear, reference)
    np.testing.assert_allclose(
        result.pressure_force, oracle["pressure_force"], rtol=2e-15, atol=2e-15
    )
    np.testing.assert_allclose(
        result.viscous_force, oracle["viscous_force"], rtol=2e-15, atol=2e-15
    )
    np.testing.assert_allclose(
        result.pressure_moment_exact, oracle["pressure_moment"], rtol=3e-15, atol=3e-15
    )
    np.testing.assert_allclose(
        result.viscous_moment_exact, oracle["viscous_moment"], rtol=3e-15, atol=3e-15
    )
    assert (
        np.linalg.norm(
            result.pressure_moment_exact - result.pressure_moment_vertex_lumped
        )
        > 1.0e-4
    )
    assert (
        np.linalg.norm(
            result.viscous_moment_exact - result.viscous_moment_vertex_lumped
        )
        > 1.0e-4
    )


def test_constant_traction_moment_equals_vertex_lumping() -> None:
    points, _, _, reference = _fixture()
    cp = np.full((2, 3), 0.75)
    shear = np.broadcast_to(np.asarray([0.2, -0.1, 0.4]), (2, 3, 3)).copy()
    result = integrate_triangle_batch(
        triangle_points=points,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference,
    )
    np.testing.assert_allclose(
        result.pressure_moment_exact,
        result.pressure_moment_vertex_lumped,
        rtol=3e-15,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        result.viscous_moment_exact,
        result.viscous_moment_vertex_lumped,
        rtol=3e-15,
        atol=3e-15,
    )


def test_translation_and_reference_point_identities() -> None:
    points, cp, shear, reference = _fixture()
    base = integrate_triangle_batch(
        triangle_points=points,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference,
    )
    translation = np.asarray([13.0, -7.5, 2.25])
    translated = integrate_triangle_batch(
        triangle_points=points + translation,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference + translation,
    )
    for name in (
        "pressure_force",
        "viscous_force",
        "pressure_moment_exact",
        "viscous_moment_exact",
        "pressure_moment_vertex_lumped",
        "viscous_moment_vertex_lumped",
    ):
        np.testing.assert_allclose(
            getattr(base, name), getattr(translated, name), atol=2e-14
        )

    reference_shift = np.asarray([0.4, -0.3, 0.2])
    shifted = integrate_triangle_batch(
        triangle_points=points,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference + reference_shift,
    )
    np.testing.assert_allclose(
        shifted.pressure_moment_exact,
        base.pressure_moment_exact - np.cross(reference_shift, base.pressure_force),
        atol=3e-15,
    )
    np.testing.assert_allclose(
        shifted.viscous_moment_exact,
        base.viscous_moment_exact - np.cross(reference_shift, base.viscous_force),
        atol=3e-15,
    )


def test_ordered_fan_and_streamed_mesh_match_direct_triangles() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.3], [0.0, 1.0, 0.1]]
    )
    connectivity = np.asarray([0, 1, 2, 3], dtype=np.int64)
    offsets = np.asarray([0, 4], dtype=np.int64)
    triangles = ordered_fan_triangle_ids(
        connectivity, offsets, point_count=points.shape[0]
    )
    np.testing.assert_array_equal(triangles, np.asarray([[0, 1, 2], [0, 2, 3]]))
    cp = np.asarray([0.2, 0.9, -0.4, 1.3])
    shear = np.asarray(
        [[0.1, 0.2, 0.3], [0.4, -0.2, 0.8], [-0.3, 0.7, 0.1], [0.9, 0.2, -0.5]]
    )
    reference = np.asarray([0.1, 0.2, -0.3])
    direct = integrate_triangle_batch(
        triangle_points=points[triangles],
        pressure_coefficient=cp[triangles],
        wall_shear_coefficient=shear[triangles],
        moment_reference=reference,
    )
    streamed = integrate_ordered_fan_mesh(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference,
        cell_chunk=1,
    )
    assert streamed.triangle_count == 2
    for name in asdict(direct):
        expected = getattr(direct, name)
        observed = getattr(streamed, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
        else:
            assert observed == expected


def test_force_and_component_closure_and_rotation() -> None:
    points, cp, shear, reference = _fixture()
    result = integrate_triangle_batch(
        triangle_points=points,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference,
    )
    document = coefficient_document(
        result, reference_area=7.0, reference_length=2.5, alpha_rad=math.radians(12.0)
    )
    force = document["force"]
    np.testing.assert_allclose(
        force["total"]["body"],
        np.asarray(force["pressure"]["body"]) + np.asarray(force["viscous"]["body"]),
        atol=0.0,
    )
    exact = document["moment_exact_degree2"]
    np.testing.assert_allclose(
        exact["total"]["body"],
        np.asarray(exact["pressure"]["body"]) + np.asarray(exact["viscous"]["body"]),
        atol=0.0,
    )
    vector = np.asarray([2.0, 3.0, 5.0])
    np.testing.assert_allclose(body_to_wind(vector, 0.0), vector)


def test_kernel_is_bitwise_deterministic() -> None:
    inputs = _fixture()
    first = integrate_triangle_batch(
        triangle_points=inputs[0],
        pressure_coefficient=inputs[1],
        wall_shear_coefficient=inputs[2],
        moment_reference=inputs[3],
    )
    second = integrate_triangle_batch(
        triangle_points=inputs[0],
        pressure_coefficient=inputs[1],
        wall_shear_coefficient=inputs[2],
        moment_reference=inputs[3],
    )
    assert first.triangle_count == second.triangle_count
    assert first.area_sum == second.area_sum
    for left, right in zip(first.vectors(), second.vectors(), strict=True):
        assert np.array_equal(left, right)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, c, s, r: (p[:, :2], c, s, r), "triangle_points"),
        (lambda p, c, s, r: (p, c[:, :2], s, r), "pressure_coefficient"),
        (lambda p, c, s, r: (p, c, s[:, :, :2], r), "wall_shear_coefficient"),
        (lambda p, c, s, r: (p, c, s, [0.0, 1.0]), "moment reference"),
        (
            lambda p, c, s, r: (
                p,
                np.where(np.arange(c.size).reshape(c.shape) == 0, np.nan, c),
                s,
                r,
            ),
            "finite",
        ),
    ],
)
def test_invalid_triangle_inputs_fail_closed(change, message: str) -> None:
    with pytest.raises(ExactMomentAuditError, match=message):
        integrate_triangle_batch(
            **dict(
                zip(
                    (
                        "triangle_points",
                        "pressure_coefficient",
                        "wall_shear_coefficient",
                        "moment_reference",
                    ),
                    change(*_fixture()),
                    strict=True,
                )
            )
        )


def test_invalid_topology_and_normalization_fail_closed() -> None:
    with pytest.raises(ExactMomentAuditError, match="at least three"):
        ordered_fan_triangle_ids(np.asarray([0, 1]), np.asarray([0, 2]), point_count=2)
    with pytest.raises(ExactMomentAuditError, match="invalid point ID"):
        ordered_fan_triangle_ids(
            np.asarray([0, 1, 4]), np.asarray([0, 3]), point_count=3
        )
    with pytest.raises(ExactMomentAuditError, match="offset endpoints"):
        ordered_fan_triangle_ids(
            np.asarray([0, 1, 2]), np.asarray([1, 3]), point_count=3
        )
    points, cp, shear, reference = _fixture()
    result = integrate_triangle_batch(
        triangle_points=points,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference,
    )
    with pytest.raises(ExactMomentAuditError, match="reference area"):
        coefficient_document(
            result, reference_area=0.0, reference_length=1.0, alpha_rad=0.0
        )
    with pytest.raises(ExactMomentAuditError, match="CI95"):
        comparison_record(1.0, 1.0, -1.0)


def test_receipt_fingerprint_tamper_detection() -> None:
    receipt = {
        "schema_id": CASE_RECEIPT_SCHEMA,
        "status": "complete",
        "algorithm_id": ALGORITHM_ID,
        "case_id": "geo_LHC001_AoA_4",
        "comparison": {
            "exact_vs_published": {"ci95_status": "pass"},
            "vertex_lumped_vs_published": {"ci95_status": "fail"},
        },
        "input_identity": {},
        "reference": {},
        "quadrature": {},
        "integral_diagnostics": {},
        "coefficients": {},
        "published": {},
        "source_fields_modified": False,
        "published_coefficients_modified": False,
    }
    receipt["content_fingerprint"] = content_fingerprint(receipt)
    receipt["comparison"]["exact_vs_published"]["ci95_status"] = "fail"
    with pytest.raises(ExactMomentAuditError, match="fingerprint differs"):
        validate_case_receipt(receipt, expected_case_id="geo_LHC001_AoA_4")


def test_published_force_component_closure_guard_records_positive_case() -> None:
    published = {
        "cd": 0.3,
        "cdp": 0.1,
        "cdv": 0.2,
        "cl": 1.5,
        "clp": 1.2,
        "clv": 0.3,
    }
    tokens = {key: str(value) for key, value in published.items()}
    audit = published_force_component_closure_audit(published, tokens)
    assert audit["status"] == "pass"
    for axis in ("drag", "lift"):
        item = audit["axes"][axis]
        assert item["status"] == "pass"
        assert item["absolute_residual"] <= item["acceptance_limit"]
        assert item["acceptance_limit"] == pytest.approx(
            item["csv_rounding_bound"] + item["floating_point_slack_64eps"],
            rel=0.0,
            abs=0.0,
        )


def test_published_force_component_closure_guard_accepts_rounding_boundary() -> None:
    published = {
        "cd": 0.3,
        "cdp": 0.12,
        "cdv": 0.12,
        "cl": 1.0,
        "clp": 0.6,
        "clv": 0.4,
    }
    tokens = {
        "cd": "0.3",
        "cdp": "0.12",
        "cdv": "0.12",
        "cl": "1.0",
        "clp": "0.6",
        "clv": "0.4",
    }
    audit = published_force_component_closure_audit(published, tokens)
    drag = audit["axes"]["drag"]
    assert drag["status"] == "pass"
    assert drag["absolute_residual"] == pytest.approx(0.06)
    assert drag["csv_rounding_bound"] == pytest.approx(0.06)


def test_published_force_component_closure_guard_rejects_beyond_rounding() -> None:
    published = {
        "cd": 0.4,
        "cdp": 0.12,
        "cdv": 0.12,
        "cl": 1.0,
        "clp": 0.6,
        "clv": 0.4,
    }
    tokens = {
        "cd": "0.4",
        "cdp": "0.12",
        "cdv": "0.12",
        "cl": "1.0",
        "clp": "0.6",
        "clv": "0.4",
    }
    with pytest.raises(ExactMomentAuditError, match="exceeds its CSV rounding bound"):
        published_force_component_closure_audit(published, tokens)
