from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from reference.hiliftaeroml.exact_moment_audit import (
    ALGORITHM_ID,
    CASE_RECEIPT_SCHEMA,
    EXPECTED_ALL_CASE_SET_SHA256,
    FORCE_EXCEPTION_CASES,
    ExactMomentAuditError,
    canonical_json_bytes,
    coefficient_document,
    comparison_record,
    content_fingerprint,
    decimal_half_quantum,
    integrate_triangle_batch,
    newline_case_set_sha256,
    validate_case_receipt,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPOSITORY / "scripts/audit_hiliftaeroml_exact_moments.py"
SPEC = importlib.util.spec_from_file_location(
    "hilift_exact_moment_campaign", DRIVER_PATH
)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def _all_case_ids() -> tuple[str, ...]:
    return tuple(
        f"geo_LHC{geometry:03d}_AoA_{aoa}"
        for geometry in range(1, 181)
        for aoa in (4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    )


def _receipt(case_id: str, *, cell_chunk: int) -> dict[str, object]:
    alpha = int(case_id.rsplit("_", 1)[1])
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.5]]],
        dtype=np.float64,
    )
    cp = np.asarray([[0.7, -0.2, 1.4]], dtype=np.float64)
    shear = np.asarray(
        [[[0.1, -0.2, 0.3], [0.7, 0.1, -0.4], [-0.3, 0.8, 0.2]]],
        dtype=np.float64,
    )
    reference_vector = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
    integrals = integrate_triangle_batch(
        triangle_points=triangle,
        pressure_coefficient=cp,
        wall_shear_coefficient=shear,
        moment_reference=reference_vector,
    )
    coefficients = coefficient_document(
        integrals,
        reference_area=2.0,
        reference_length=3.0,
        alpha_rad=math.radians(alpha),
    )
    force = coefficients["force"]
    published_values = {
        "cdp": force["pressure"]["cd"],
        "clp": force["pressure"]["cl"],
        "cdv": force["viscous"]["cd"],
        "clv": force["viscous"]["cl"],
        "cd": force["total"]["cd"],
        "cl": force["total"]["cl"],
        "cm": coefficients["moment_exact_degree2"]["total"]["cm_body_y"],
    }
    tokens = {name: repr(float(value)) for name, value in published_values.items()}
    force_row = {
        **tokens,
        "cd_ci95": "0.0",
        "cl_ci95": "0.0",
    }
    exact_cm = float(coefficients["moment_exact_degree2"]["total"]["cm_body_y"])
    vertex_cm = float(coefficients["moment_vertex_lumped"]["total"]["cm_body_y"])
    quadrature = {
        "surface_triangulation": "ordered fan (v0, vj, vj+1)",
        "field_interpolation": "piecewise-linear nodal",
        "force": "exact degree-1 barycentric integration",
        "moment_exact": (
            "exact degree-2 triangle mass matrix; integral(lambda_i*lambda_j) "
            "is A/6 for i=j and A/12 otherwise"
        ),
        "moment_legacy_comparison": "vertex-lumped r cross nodal force",
        "pressure_sign": "+Cp times ordered oriented area vector",
        "cm_convention": (
            "positive body-y moment about forcesCoR divided by areaRef*chordRef"
        ),
    }
    source_stat = {
        "path": f"/frozen/{case_id}/boundary_{case_id}.vtu",
        "size_bytes": 123,
        "mtime_ns": 456,
    }
    field_specs = {
        name: {
            "vtk_type": "Float32",
            "dtype": "<f4",
            "components": 1,
            "tuples": 3,
            "payload_offset": index * 12,
            "expected_nbytes": 12,
        }
        for index, name in enumerate(campaign.REQUIRED_POINT_FIELDS)
    }
    diagnostics = {
        "triangle_count": integrals.triangle_count,
        "zero_area_triangle_count": integrals.zero_area_triangle_count,
        "surface_area_in2": integrals.area_sum,
        "oriented_area_sum_in2": integrals.oriented_area_sum.tolist(),
        "pressure_force_numerator_in2": integrals.pressure_force.tolist(),
        "viscous_force_numerator_in2": integrals.viscous_force.tolist(),
        "pressure_moment_exact_in3": integrals.pressure_moment_exact.tolist(),
        "viscous_moment_exact_in3": integrals.viscous_moment_exact.tolist(),
        "pressure_moment_vertex_lumped_in3": (
            integrals.pressure_moment_vertex_lumped.tolist()
        ),
        "viscous_moment_vertex_lumped_in3": (
            integrals.viscous_moment_vertex_lumped.tolist()
        ),
    }
    half_quantum = decimal_half_quantum(tokens["cm"])
    receipt: dict[str, object] = {
        "schema_id": CASE_RECEIPT_SCHEMA,
        "status": "complete",
        "algorithm_id": ALGORITHM_ID,
        "case_id": case_id,
        "input_identity": {
            "preflight": {
                "source_vtu_stat": source_stat,
                "frozen_source_binding": {
                    "frozen_all1800_inventory_sha256": (
                        campaign.EXPECTED_ALL1800_INVENTORY_SHA256
                    ),
                    "frozen_local_surface": source_stat,
                    "public_source_inventory_sha256": None,
                    "public_surface_archive": None,
                },
                "reference_csv": {
                    "path": f"/frozen/{case_id}/ref.csv",
                    "size_bytes": 10,
                    "sha256": "1" * 64,
                },
                "force_csv": {
                    "path": f"/frozen/{case_id}/force.csv",
                    "size_bytes": 10,
                    "sha256": "2" * 64,
                },
                "p_inf": 176.352,
                "cell_chunk": cell_chunk,
                "backend_files": {
                    name: {"path": f"/frozen/{name}.py", "sha256": "3" * 64}
                    for name in (
                        "native_vtu_memmap",
                        "build_hilift_surface_dual_area",
                        "exact_moment_audit",
                        "exact_moment_numba",
                        "campaign_driver",
                    )
                },
            },
            "native": {
                "geometry_sha256": "4" * 64,
                "topology": {
                    "n_points": 3,
                    "n_cells": 1,
                    "n_connectivity_values": 3,
                    "n_fan_triangles": 1,
                    "connectivity_min": 0,
                    "connectivity_max": 2,
                    "cell_type_histogram": {"5": 1},
                    "cell_arity_histogram": {"3": 1},
                },
                "field_specs": field_specs,
                "logical_field_sha256": {
                    name: str(index + 5) * 64
                    for index, name in enumerate(campaign.REQUIRED_POINT_FIELDS)
                },
                "source_byte_count_convention": "payload",
            },
        },
        "reference": {
            "p_inf": 176.352,
            "q_ref": 2.0,
            "area_ref_in2": 2.0,
            "length_ref_in": 3.0,
            "alpha_deg": float(alpha),
            "forcesCoR_in": reference_vector.tolist(),
        },
        "quadrature": quadrature,
        "integral_diagnostics": diagnostics,
        "coefficients": coefficients,
        "published": {
            **published_values,
            "cd_stderr": 0.0,
            "cl_stderr": 0.0,
            "cm_stderr": 0.0,
            "cd_ci95": 0.0,
            "cl_ci95": 0.0,
            "cm_ci95": 0.0,
            "cm_csv_token": tokens["cm"],
            "force_csv_tokens": tokens,
        },
        "comparison": {
            "scientific_role": (
                "diagnostic audit; disagreement does not mutate fields or published CSVs"
            ),
            "exact_vs_published": comparison_record(
                exact_cm,
                float(published_values["cm"]),
                0.0,
                published_half_quantum=half_quantum,
            ),
            "vertex_lumped_vs_published": comparison_record(
                vertex_cm,
                float(published_values["cm"]),
                0.0,
                published_half_quantum=half_quantum,
            ),
            "exact_minus_vertex_lumped_cm": exact_cm - vertex_cm,
            "exact_minus_vertex_lumped_relative_to_published_percent": (
                100.0 * (exact_cm - vertex_cm) / abs(published_values["cm"])
                if published_values["cm"] != 0.0
                else None
            ),
            "force_vs_published": campaign._force_comparisons(
                coefficients, published_values
            ),
            "force_acceptance": campaign._canonical_force_acceptance(
                coefficients, force_row, Path("synthetic-force.csv")
            ),
        },
        "source_fields_modified": False,
        "published_coefficients_modified": False,
    }
    receipt["content_fingerprint"] = content_fingerprint(receipt)
    validate_case_receipt(receipt, expected_case_id=case_id)
    return receipt


def test_exact_scope_and_eight_rank_stripes() -> None:
    all_ids = _all_case_ids()
    assert len(all_ids) == 1800
    assert newline_case_set_sha256(all_ids) == EXPECTED_ALL_CASE_SET_SHA256
    assert campaign.validate_stripe_coverage(1800, world_size=8) == tuple(range(1800))
    assert [
        len(campaign.striped_indices(1800, rank=rank, world_size=8))
        for rank in range(8)
    ] == [225] * 8
    assert set(campaign.PILOT_CASES) == {
        *FORCE_EXCEPTION_CASES,
        "geo_LHC001_AoA_4",
    }
    with pytest.raises(ExactMomentAuditError, match="world_size"):
        campaign.striped_indices(1800, rank=0, world_size=3)


def test_semantic_receipt_validation_rejects_rehashed_scientific_tamper() -> None:
    receipt = _receipt("geo_LHC001_AoA_4", cell_chunk=100_000)
    receipt["coefficients"]["moment_exact_degree2"]["total"]["cm_body_y"] += 1.0
    receipt["content_fingerprint"] = content_fingerprint(receipt)
    with pytest.raises(ExactMomentAuditError, match="coefficients do not close"):
        validate_case_receipt(receipt, expected_case_id="geo_LHC001_AoA_4")

    closure_tamper = _receipt("geo_LHC001_AoA_4", cell_chunk=100_000)
    closure_tamper["comparison"]["force_acceptance"]["published_component_closure"][
        "axes"
    ]["drag"]["status"] = "fail"
    closure_tamper["content_fingerprint"] = content_fingerprint(closure_tamper)
    with pytest.raises(ExactMomentAuditError, match="force acceptance differs"):
        validate_case_receipt(closure_tamper, expected_case_id="geo_LHC001_AoA_4")


def test_pilot_replays_allow_only_chunk_and_numerical_regrouping(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "pilot-a", tmp_path / "pilot-b")
    for root, chunk in zip(roots, (100_000, 131_071), strict=True):
        case_dir = root / campaign.SUCCESS_DIRNAME
        case_dir.mkdir(parents=True)
        for case_id in campaign.PILOT_CASES:
            (case_dir / f"{case_id}.json").write_bytes(
                canonical_json_bytes(_receipt(case_id, cell_chunk=chunk))
            )
    comparison = campaign.compare_pilot_receipts(
        pilot_root_a=roots[0], pilot_root_b=roots[1]
    )
    assert comparison["status"] == "pass"
    assert comparison["case_count"] == 6
    assert all(
        len(item["replay_contract_sha256"]) == 64
        for item in comparison["cases"].values()
    )

    tampered_path = roots[1] / campaign.SUCCESS_DIRNAME / "geo_LHC001_AoA_4.json"
    tampered = _receipt("geo_LHC001_AoA_4", cell_chunk=131_071)
    tampered["input_identity"]["preflight"]["backend_files"]["campaign_driver"][
        "sha256"
    ] = "a" * 64
    tampered["content_fingerprint"] = content_fingerprint(tampered)
    tampered_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ExactMomentAuditError, match="contract differs"):
        campaign.compare_pilot_receipts(pilot_root_a=roots[0], pilot_root_b=roots[1])


def test_atomic_derived_artifact_replay_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "aggregate.json"
    payload = b'{"status":"complete"}\n'
    assert campaign._atomic_create(target, payload) == "created"
    assert (
        campaign._atomic_create(target, payload, accept_identical_existing=True)
        == "verified_identical_existing"
    )
    with pytest.raises(ExactMomentAuditError, match="differs byte-for-byte"):
        campaign._atomic_create(
            target,
            b'{"status":"different"}\n',
            accept_identical_existing=True,
        )


def test_slurm_restart_and_rank_cache_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("HILIFT_NUMBA_CACHE_PARENT", str(tmp_path / "cache"))
    monkeypatch.delenv("HILIFT_NUMBA_CACHE_ROLE", raising=False)
    assert campaign._slurm_attempt_identity() == ("12345", 2, "12345-r2")
    expected = tmp_path / "cache/12345-r2/rank-007"
    assert campaign._configure_isolated_numba_cache() == expected
    assert os.environ["NUMBA_CACHE_DIR"] == str(expected)


def test_actual_eight_rank_receipts_prove_exact_1800_union(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    all_ids = _all_case_ids()
    case_root = tmp_path / campaign.SUCCESS_DIRNAME
    case_root.mkdir()
    for case_id in all_ids:
        (case_root / f"{case_id}.json").write_bytes(
            canonical_json_bytes(_receipt(case_id, cell_chunk=100_000))
        )

    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    backend_files = _receipt(all_ids[0], cell_chunk=100_000)["input_identity"][
        "preflight"
    ]["backend_files"]
    kernel_self_check = {
        "status": "pass",
        "points_dtype": "<f4",
        "connectivity_dtype": "<i8",
        "offsets_dtype": "<i8",
        "point_field_dtype": "<f4",
        "point_fields_read_only": True,
        "moment_reference_dtype": "<f8",
        "triangle_count": 1,
    }
    for rank in range(8):
        assigned = tuple(
            all_ids[index]
            for index in campaign.striped_indices(1800, rank=rank, world_size=8)
        )
        monkeypatch.setenv("SLURM_PROCID", str(rank))
        runtime = {
            "python_executable": "/frozen/python",
            "python": "3.12.3",
            "numpy": "2.2.6",
            "numba": "0.61.2",
            "vtk": "9.5.2",
            "system_dist_packages_excluded": True,
            "python_no_user_site": True,
            "python_unbuffered": True,
            "numba_cache_dir": str(tmp_path / f"numba-cache/98765-r3/rank-{rank:03d}"),
            "sys_path": ["/frozen/repository"],
        }
        path = campaign._write_rank_completion(
            output_root=tmp_path,
            selected=all_ids,
            rank=rank,
            world=8,
            completed=len(assigned),
            skipped=0,
            backend_files=backend_files,
            runtime=runtime,
            kernel_self_check=kernel_self_check,
            per_case_elapsed_seconds={case_id: 0.001 for case_id in assigned},
            rank_elapsed_seconds=1.0,
        )
        assert path.parent.name == "98765-r3"

    coverage = campaign.validate_rank_coverage(
        output_root=tmp_path,
        all_case_ids=all_ids,
        attempt_id="98765-r3",
    )
    assert coverage["status"] == "pass"
    assert coverage["case_count"] == 1800
    assert coverage["exact_union_0_through_1799_once"] is True


def test_full_campaign_homogeneity_rejects_mixed_chunk() -> None:
    all_ids = _all_case_ids()
    public_ids = all_ids[: campaign.EXPECTED_PUBLIC_CASE_COUNT]
    backend_files = {"driver": {"path": "/driver.py", "sha256": "a" * 64}}
    quadrature = {"moment_exact": "degree two"}
    receipts = {}
    expected_bindings = {}
    for index, case_id in enumerate(all_ids):
        public = index < campaign.EXPECTED_PUBLIC_CASE_COUNT
        binding = {
            "frozen_all1800_inventory_sha256": (
                campaign.EXPECTED_ALL1800_INVENTORY_SHA256
            ),
            "frozen_local_surface": {
                "path": f"/frozen/{case_id}.vtu",
                "size_bytes": index + 1,
                "mtime_ns": index + 2,
            },
            "public_surface_archive": (
                {"repository_path": f"surface/{case_id}.tar"} if public else None
            ),
            "public_source_inventory_sha256": (
                campaign.EXPECTED_PUBLIC_IDENTITY_SHA256 if public else None
            ),
        }
        expected_bindings[case_id] = binding
        receipts[case_id] = {
            "input_identity": {
                "preflight": {
                    "backend_files": backend_files,
                    "p_inf": 176.352,
                    "cell_chunk": 100_000,
                    "frozen_source_binding": binding,
                },
                "native": {"source_byte_count_convention": "payload"},
            },
            "quadrature": quadrature,
        }
    result = campaign._campaign_homogeneity(
        all_ids, public_ids, expected_bindings, receipts
    )
    assert result["case_count"] == 1800
    assert result["public_source_bound_case_count"] == 1355
    mixed = copy.deepcopy(receipts)
    mixed[all_ids[-1]]["input_identity"]["preflight"]["cell_chunk"] = 131_071
    with pytest.raises(ExactMomentAuditError, match="contract differs"):
        campaign._campaign_homogeneity(all_ids, public_ids, expected_bindings, mixed)

    swapped = copy.deepcopy(receipts)
    swapped[public_ids[0]]["input_identity"]["preflight"]["frozen_source_binding"] = (
        copy.deepcopy(expected_bindings[all_ids[-1]])
    )
    swapped[all_ids[-1]]["input_identity"]["preflight"]["frozen_source_binding"] = (
        copy.deepcopy(expected_bindings[public_ids[0]])
    )
    with pytest.raises(ExactMomentAuditError, match="descriptor differs"):
        campaign._campaign_homogeneity(all_ids, public_ids, expected_bindings, swapped)


def test_launchers_are_gated_storage_light_and_requeue_safe() -> None:
    full = (
        REPOSITORY / "scripts/run_hiliftaeroml_exact_moment_full.sbatch"
    ).read_text()
    submit = (
        REPOSITORY / "scripts/submit_hiliftaeroml_exact_moment_full.sh"
    ).read_text()
    driver = DRIVER_PATH.read_text()
    assert "#SBATCH --qos=cpu-short" in full
    assert "#SBATCH --nodes=8" in full
    assert "#SBATCH --ntasks=8" in full
    assert "#SBATCH --time=04:00:00" in full
    assert "${SLURM_JOB_ID}-r${restart_count}" in full
    assert "PYTHONNOUSERSITE=1" in full
    assert "PYTHONUNBUFFERED=1" in full
    assert "HILIFT_NUMBA_CACHE_PARENT" in full
    assert submit.count("sbatch --parsable") == 1
    assert 'mode="${1:---dry-run}"' in submit
    assert "HILIFT_MOMENT_FULL_SUBMIT_APPROVED" in submit
    assert "point_data_memmap" in driver
    assert "integrate_cell_range" in driver
    assert "logical_field_sha256" in driver
    assert "np.column_stack(tau)" not in driver
    for forbidden in ("git push", "huggingface", "leaderboard", "upload_file"):
        assert forbidden not in full.lower()
        assert forbidden not in submit.lower()
