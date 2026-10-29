from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

from reference.drivaerml import dataset_scorer as dataset_scorer_module
from reference.drivaerml.dataset_scorer import (
    ALL_FIELD_METRIC_IDS,
    CANDIDATE_DATASET_STATUS,
    CandidateDatasetImmutablePins,
    DrivAerDatasetScorerError,
    evaluate_candidate_dataset,
    schema_v3_case_metrics_candidate_adapter,
    schema_v3_profile_chunks_candidate_adapter,
    validate_schema_v3_candidate_nonspatial_metrics,
    write_candidate_dataset_evidence,
    write_schema_v3_profile_chunks_candidate,
)


REPOSITORY = "neashton/drivaerml"
REVISION = "7a5c0948ce27be709b1116a3a190f806e7a8f79f"
SURFACE_MANIFEST_SHA = "a" * 64
PROFILE_SHA_PLACEHOLDER = "d" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class DatasetFixture:
    case_ids = ("run_1", "run_2")
    surface_count = 2
    volume_count = 2

    def __init__(self, root: Path) -> None:
        self.root = root
        self.core_dir = root / "core"
        self.diagnostic_dir = root / "diagnostic"
        self.core_dir.mkdir()
        self.diagnostic_dir.mkdir()
        self.boundary_sha = {case_id: str(index) * 64 for index, case_id in enumerate(self.case_ids, start=1)}
        self.area_sha = {case_id: str(index + 2) * 64 for index, case_id in enumerate(self.case_ids, start=1)}
        self.part_sha = {
            case_id: (str(index + 4) * 64, str(index + 6) * 64)
            for index, case_id in enumerate(self.case_ids, start=1)
        }

        self.profile_path = root / "drivaerml-diagnostics-v9.json"
        stations = [
            {
                "id": f"line_{index:02d}",
                "source_profile_id": f"line_{index:02d}",
                "sample_count": 2,
                "experimental_availability": (
                    "case_2a" if index <= 11 else "none"
                ),
            }
            for index in range(1, 17)
        ]
        profile = {
            "schema_version": "1.0",
            "id": "drivaerml-diagnostics-v9-candidate",
            "dataset_id": "drivaerml",
            "status": "candidate_support_pending_all_case_validation",
            "source": {},
            "pressure_cuts": {
                "definition_authority": "FluidsBench",
                "ranked_metric_id": "cp_cut_r2",
                "report_only_metric_id": "cp_cut_rmse",
                "quantity": "Cp=2*pMeanTrim/(38.889^2)",
                "cut_count": 4,
                "association": "native_surface_VTP_CellData",
                "extraction_status": "pending_immutable_owner_cut_support",
                "reduction": "equal_case_equal_cut_global_R2_with_normalized_native_intersection_segment_length_support_per_cut",
                "stations": [
                    {"id": "upperbody_centerline"},
                    {"id": "underbody_centerline"},
                    {"id": "sidewall_z_0_15"},
                    {"id": "front_left_wheelhouse_y_neg_0_6"},
                ],
            },
            "velocity_profiles": {
                "definition_authority": "AutoCFD5",
                "ranked_metric_id": "velocity_profile_r2",
                "report_only_metric_id": "velocity_profile_uinf_rmse",
                "ranked_reduction": "equal_case_equal_line_global_R2_with_normalized_trapezoidal_arc_length_support_per_line",
                "quantity": "magnitude(UMeanTrim)/Uinf",
                "scoring_grid": {},
                "line_count": 16,
                "sample_count_per_case": 32,
                "stations": stations,
            },
            "activation_requirements": [],
        }
        _write_json(self.profile_path, profile)
        self.profile_sha = _sha256(self.profile_path)

        self.force_path = root / "force_mom_constref_all.csv"
        self.force_path.write_text(
            "run,cd,cl,clf,clr,cs\n"
            "1,0.1,0.2,0.12,0.08,0.0\n"
            "2,0.2,0.4,0.23,0.17,0.0\n",
            encoding="utf-8",
        )
        self.force_sha = _sha256(self.force_path)

        self.pin_path = root / "native-source-pin.json"
        pin_cases = []
        for index, case_id in enumerate(self.case_ids, start=1):
            pin_cases.append(
                {
                    "case_id": case_id,
                    "run_number": index,
                    "boundary": {
                        "path": f"{case_id}/boundary_{index}.vtp",
                        "size_bytes": 10 + index,
                        "lfs_sha256": self.boundary_sha[case_id],
                    },
                    "surface_cell_area": {
                        "path": f"{case_id}/boundary_cell_area_{index}.npy",
                        "size_bytes": 128 + 4 * self.surface_count,
                        "lfs_sha256": self.area_sha[case_id],
                        "dtype": "<f4",
                        "element_count": self.surface_count,
                        "source_boundary_sha256": self.boundary_sha[case_id],
                    },
                    "volume": {
                        "logical_path_after_assembly": f"{case_id}/volume_{index}.vtu",
                        "part_count": 2,
                        "parts": [
                            {
                                "part_index": part,
                                "path": f"{case_id}/volume_{index}.vtu.{part:02d}.part",
                                "size_bytes": part + 2,
                                "lfs_sha256": self.part_sha[case_id][part],
                            }
                            for part in range(2)
                        ],
                        "total_size_bytes": 5,
                    },
                }
            )
        pin = {
            "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
            "schema_version": 1,
            "repository": {"repo_id": REPOSITORY, "revision": REVISION},
            "case_scope": {
                "case_count": 2,
                "run_number_min": 1,
                "run_number_max": 2,
                "unavailable_or_held_back_run_numbers": [],
            },
            "cases": pin_cases,
            "authoritative_support_files": {
                "force_mom_constref_all": {
                    "path": self.force_path.name,
                    "sha256": self.force_sha,
                },
                "surface_cell_area_manifest": {
                    "path": "surface_cell_areas/manifest.json",
                    "sha256": SURFACE_MANIFEST_SHA,
                },
            },
            "totals": {
                "boundary_file_count": 2,
                "boundary_bytes": 23,
                "surface_cell_area_file_count": 2,
                "surface_cell_area_bytes": 272,
                "logical_volume_count": 2,
                "volume_part_file_count": 4,
                "reconstructed_volume_bytes": 10,
            },
        }
        _write_json(self.pin_path, pin)
        self.pin_sha = _sha256(self.pin_path)

        self.split_path = root / "splits" / "tiny.json"
        split = {
            "schema_version": "1.0",
            "dataset_id": "drivaerml",
            "split_id": "tiny",
            "case_set_id": "standard",
            "split_label": "Tiny test fixture",
            "case_id_status": "official",
            "training_case_count": 1,
            "validation_case_count": 1,
            "case_count": 2,
            "train_case_ids": ["run_1"],
            "validation_case_ids": ["run_2"],
            "case_ids": list(self.case_ids),
            "source": {"repository": REPOSITORY, "revision": REVISION},
            "usage": {"train": "fixture", "validation": "fixture", "test": "fixture"},
        }
        # The production split sets are disjoint. A two-case fixture cannot
        # cover all three roles without overlap, so use synthetic pinned cases
        # outside the test set for train/validation and extend the pin below.
        split["train_case_ids"] = ["run_3"]
        split["validation_case_ids"] = ["run_4"]
        _write_json(self.split_path, split)
        self.split_sha = _sha256(self.split_path)

        # Extend the native pin with train/validation identity-only records so
        # the strict split validation is realistic. Force truth covers all pin
        # cases, therefore rebuild both inputs in canonical run order.
        for index in (3, 4):
            case_id = f"run_{index}"
            boundary_sha = str(index) * 64
            area_sha = str(index + 2) * 64
            parts = tuple(
                hashlib.sha256(f"{case_id}-part-{part}".encode()).hexdigest()
                for part in range(2)
            )
            pin_cases.append(
                {
                    "case_id": case_id,
                    "run_number": index,
                    "boundary": {"path": f"{case_id}/boundary_{index}.vtp", "size_bytes": 10 + index, "lfs_sha256": boundary_sha},
                    "surface_cell_area": {"path": f"{case_id}/boundary_cell_area_{index}.npy", "size_bytes": 128 + 4 * self.surface_count, "lfs_sha256": area_sha, "dtype": "<f4", "element_count": self.surface_count, "source_boundary_sha256": boundary_sha},
                    "volume": {"logical_path_after_assembly": f"{case_id}/volume_{index}.vtu", "part_count": 2, "parts": [{"part_index": part, "path": f"{case_id}/volume_{index}.vtu.{part:02d}.part", "size_bytes": part + 2, "lfs_sha256": parts[part]} for part in range(2)], "total_size_bytes": 5},
                }
            )
        self.force_path.write_text(
            "run,cd,cl,clf,clr,cs\n"
            "1,0.1,0.2,0.12,0.08,0.0\n"
            "2,0.2,0.4,0.23,0.17,0.0\n"
            "3,0.0,0.0,0.0,0.0,0.0\n"
            "4,0.0,0.0,0.0,0.0,0.0\n",
            encoding="utf-8",
        )
        self.force_sha = _sha256(self.force_path)
        pin["case_scope"].update({"case_count": 4, "run_number_max": 4})
        pin["authoritative_support_files"]["force_mom_constref_all"]["sha256"] = self.force_sha
        pin["totals"].update(
            {
                "boundary_file_count": 4,
                "boundary_bytes": 50,
                "surface_cell_area_file_count": 4,
                "surface_cell_area_bytes": 544,
                "logical_volume_count": 4,
                "volume_part_file_count": 8,
                "reconstructed_volume_bytes": 20,
            }
        )
        _write_json(self.pin_path, pin)
        self.pin_sha = _sha256(self.pin_path)

        self.spec_path = root / "submission-spec.json"
        specification = {
            "dataset_id": "drivaerml",
            "status": "candidate_scoring_contract",
            "scoring_support": {
                "status": "owner_review_required",
                "submissions_open": False,
                "source_release": {
                    "repository": REPOSITORY,
                    "revision": REVISION,
                    "native_source_pin": {"file": self.pin_path.name, "sha256": self.pin_sha},
                },
                "public_supports": [
                    {
                        "id": "surface_native_cells",
                        "physical_weight_manifest": {"path": "surface_cell_areas/manifest.json", "sha256": SURFACE_MANIFEST_SHA},
                    },
                    {
                        "id": "field_integrated_force_truth",
                        "authoritative_public_file": self.force_path.name,
                        "sha256": self.force_sha,
                    },
                ],
                "force_integration": {
                    "freestream_velocity_m_per_s": 38.889,
                    "density_kg_per_m3": 1.0,
                    "reference_area_m2": 2.17,
                    "reference_length_m": 2.78618,
                },
            },
            "profile_definition": {"file": self.profile_path.name, "sha256": self.profile_sha},
            "splits": [
                {
                    "id": "tiny",
                    "index_file": "splits/tiny.json",
                    "training_case_count": 1,
                    "validation_case_count": 1,
                    "case_count": 2,
                    "case_set_id": "standard",
                    "case_id_status": "official",
                    "sha256": self.split_sha,
                }
            ],
        }
        _write_json(self.spec_path, specification)
        self.immutable_pins = CandidateDatasetImmutablePins(
            repository_id=REPOSITORY,
            repository_revision=REVISION,
            native_source_pin_sha256=self.pin_sha,
            force_truth_sha256=self.force_sha,
            diagnostic_profile_sha256=self.profile_sha,
            surface_area_manifest_sha256=SURFACE_MANIFEST_SHA,
        )

        for index, case_id in enumerate(self.case_ids, start=1):
            _write_json(self.core_dir / f"{case_id}.json", self._core(case_id, index))
            _write_json(
                self.diagnostic_dir / f"{case_id}.json",
                self._diagnostic(case_id, index),
            )

    def _field_sums(self, case_index: int) -> dict[str, dict[str, dict[str, float | int]]]:
        targets = {
            1: {"surface_pressure": 10.0, "surface_wall_shear": 20.0, "volume_pressure": 40.0},
            2: {"surface_pressure": 14.0, "surface_wall_shear": 18.0, "volume_pressure": 38.0},
        }
        result = {}
        for prefix in ("surface_pressure", "surface_wall_shear", "volume_pressure"):
            primary = targets[case_index][prefix]
            # Surface primary is physical; volume pressure primary is uniform.
            uniform_rel = primary if prefix == "volume_pressure" else primary + 1.0
            physical_rel = primary if prefix.startswith("surface") else primary + 1.0
            weighting_sums = {
                "uniform": {"absolute_error": 2.0 + case_index, "squared_error": uniform_rel**2, "squared_truth": 10000.0, "entity_count": 2, "total_weight": 2.0},
            }
            if prefix.startswith("surface"):
                weighting_sums["physical"] = {
                    "absolute_error": 3.0 + case_index,
                    "squared_error": physical_rel**2,
                    "squared_truth": 10000.0,
                    "entity_count": 2,
                    "total_weight": 3.0,
                }
            result[prefix] = weighting_sums
        # The first case is the non-axis-aligned vector golden: errors
        # (3,4,0) and (0,0,12) give per-entity norms 5 and 12.
        if case_index == 1:
            velocity_uniform = {"absolute_error": 17.0, "squared_error": 169.0, "squared_truth": 10000.0, "entity_count": 2, "total_weight": 2.0}
        else:
            velocity_uniform = {"absolute_error": 10.0, "squared_error": 50.0, "squared_truth": 10000.0, "entity_count": 2, "total_weight": 2.0}
        result["volume_velocity"] = {"uniform": velocity_uniform}
        return result

    def _metrics(self, sums: dict[str, dict[str, dict[str, float | int]]]):
        values = {}
        statistics = {}
        for prefix, weighting_sums in sums.items():
            surface = prefix.startswith("surface")
            primary_weighting = "physical" if surface else "uniform"
            primary_label = "area" if surface else "equal_entity"
            relative_metrics = [
                (
                    f"{prefix}_rel_l2",
                    primary_weighting,
                    "surface_face_area" if surface else "volume_cells_equal",
                )
            ]
            if surface:
                relative_metrics.append(
                    (
                        f"{prefix}_equal_entity_rel_l2",
                        "uniform",
                        "surface_entities_equal",
                    )
                )
            for metric_id, weighting, dataset_weighting in relative_metrics:
                source = weighting_sums[weighting]
                values[metric_id] = 100.0 * math.sqrt(source["squared_error"] / source["squared_truth"])
                statistics[metric_id] = {
                    "reduction": "relative_l2_percent",
                    "weighting": "uniform" if weighting == "uniform" else "support_weights",
                    "dataset_weighting": dataset_weighting,
                    "numerator": source["squared_error"],
                    "denominator": source["squared_truth"],
                    "entity_count": 2,
                    "total_weight": source["total_weight"],
                }
            reductions = [(primary_label, primary_weighting)]
            if surface:
                reductions.append(("equal_entity", "uniform"))
            for label, weighting in reductions:
                source = weighting_sums[weighting]
                values[f"drivaerml_{prefix}_{label}_mae"] = source["absolute_error"] / source["total_weight"]
                values[f"drivaerml_{prefix}_{label}_rmse"] = math.sqrt(source["squared_error"] / source["total_weight"])
        self.assert_metric_keys(values)
        return values, statistics

    @staticmethod
    def assert_metric_keys(values: dict[str, float]) -> None:
        if set(values) != set(ALL_FIELD_METRIC_IDS):
            raise AssertionError((set(values) - set(ALL_FIELD_METRIC_IDS), set(ALL_FIELD_METRIC_IDS) - set(values)))

    def _core(self, case_id: str, case_index: int) -> dict[str, object]:
        sums = self._field_sums(case_index)
        metric_values, statistics = self._metrics(sums)
        boundary = self.boundary_sha[case_id]
        areas = self.area_sha[case_id]
        parts = self.part_sha[case_id]
        surface_manifest = ("8" if case_index == 1 else "9") * 64
        volume_manifest = ("a" if case_index == 1 else "b") * 64
        surface_chunk = ("c" if case_index == 1 else "d") * 64
        volume_chunk = ("e" if case_index == 1 else "f") * 64
        force = (
            {"Cd": 0.11, "Cl": 0.22, "CmPitch": 0.02}
            if case_index == 1
            else {"Cd": 0.17, "Cl": 0.36, "CmPitch": 0.02}
        )
        force["Clf"] = force["Cl"] / 2.0 + force["CmPitch"]
        force["Clr"] = force["Cl"] / 2.0 - force["CmPitch"]
        force["Cs"] = 0.0
        force["lift_closure_abs"] = abs(force["Cl"] - (force["Clf"] + force["Clr"]))
        q_area = 0.5 * 38.889**2 * 2.17
        force_vector = [force["Cd"] * q_area, force["Cs"] * q_area, force["Cl"] * q_area]
        moment_vector = [0.0, force["CmPitch"] * q_area * 2.78618, 0.0]
        return {
            "schema": "drivaerml-candidate-case-evaluation-v2",
            "schema_version": 2,
            "status": "candidate_evaluator_evidence_not_official_submission",
            "official_submission": False,
            "case_id": case_id,
            "source": {
                "native_source_pin_sha256": self.pin_sha,
                "repository_id": REPOSITORY,
                "repository_revision": REVISION,
                "boundary_sha256": boundary,
                "surface_native": {
                    "source_file": f"boundary_{case_index}.vtp",
                    "boundary_sha256": boundary,
                    "vtk_version": "9.5.2",
                    "point_count": 4,
                    "polygon_count": 2,
                    "raw_cell_order": "zero_based_native_vtk_polygon_order_unchanged",
                    "association": "CellData",
                    "arrays": {
                        "pMeanTrim": {"components": 1, "tuples": 2, "dtype": "<f4", "unit": "m^2/s^2"},
                        "wallShearStressMeanTrim": {"components": 3, "tuples": 2, "dtype": "<f4", "unit": "m^2/s^2"},
                    },
                    "available_point_arrays": [],
                    "available_cell_arrays": ["pMeanTrim", "wallShearStressMeanTrim"],
                },
                "surface_area": {
                    "source_path": f"boundary_cell_area_{case_index}.npy",
                    "sha256": areas,
                    "source_boundary_sha256": boundary,
                    "entity_count": 2,
                    "area_sum_m2": 3.0,
                    "area_min_m2": 1.0,
                    "area_max_m2": 2.0,
                    "dtype": "<f4",
                    "role": "fixed_external_input_not_regenerated",
                    "native_geometry_order_audit": {
                        "entity_count": 2,
                        "relative_tolerance": 6.0e-8,
                        "calculated_sum_m2": 3.0,
                        "published_sum_m2": 3.0,
                        "maximum_absolute_difference_m2": 0.0,
                        "maximum_relative_difference": 0.0,
                        "raw_order_correspondence_verified": True,
                        "published_values_role": "fixed_input_audited_not_regenerated",
                    },
                },
                "volume_part_sha256": list(parts),
                "volume_vtk": {"dataset_type": "UnstructuredGrid", "piece_count": 1, "cell_count": 2, "source_size_bytes": 5},
                "volume_weighting": {
                    "weighting": "one_per_native_cell",
                    "entity_count": 2,
                    "total_weight": 2.0,
                    "geometric_cell_volume_weights_used": False,
                },
                "volume_native_arrays": {
                    "pMeanTrim": {"name": "pMeanTrim", "association": "CellData", "number_of_components": 1, "tuple_count": 2, "scalar_count": 2, "finite": True, "units": "m^2/s^2", "raw_id_start": 0, "raw_id_stop": 2, "payload_sha256": "4" * 64},
                    "UMeanTrim": {"name": "UMeanTrim", "association": "CellData", "number_of_components": 3, "tuple_count": 2, "scalar_count": 6, "finite": True, "units": "m/s", "raw_id_start": 0, "raw_id_stop": 2, "payload_sha256": "5" * 64},
                },
            },
            "prediction_inputs": {
                "surface_native_cells": {"manifest_sha256": surface_manifest, "chunk_sha256": [surface_chunk], "chunk_count": 1, "entity_count": 2},
                "volume_native_cells": {"manifest_sha256": volume_manifest, "chunk_sha256": [volume_chunk], "chunk_count": 1, "entity_count": 2},
            },
            "coverage": {
                "surface": {"raw_cell_id_start": 0, "raw_cell_id_stop": 2, "complete_gap_free_duplicate_free": True},
                "volume": {"raw_cell_id_start": 0, "raw_cell_id_stop": 2, "complete_gap_free_duplicate_free": True},
            },
            "metric_values": metric_values,
            "metric_sufficient_statistics": statistics,
            "additive_sums": sums,
            "force_coefficients": {"entity_count": 2, "force_n": force_vector, "moment_about_forces_cor_n_m": moment_vector, **force},
            "execution": {"maximum_prediction_chunk_rows": 10, "hash_chunk_bytes": 10, "validation_block_rows": 2, "encoded_chunk_bytes": 10},
        }

    def _diagnostic(self, case_id: str, case_index: int) -> dict[str, object]:
        core = self._core(case_id, case_index)
        predictions = core["prediction_inputs"]
        parts = self.part_sha[case_id]
        velocity_value = 0.1 if case_index == 1 else 0.3
        claims = {
            "scoring_contract_active": False,
            "official_submission": False,
            "owner_scientific_approval": False,
            "profile_resolution_convergence": False,
            "three_real_model_ordering": False,
            "independent_participant_dry_run": False,
            "all_case_chunk_partition_invariance": False,
        }

        def prediction_audit(support: str, field: str):
            item = predictions[support]
            return {
                "support_id": support,
                "field_name": field,
                "total_row_count": 2,
                "manifest_file": "manifest.json",
                "manifest_sha256": item["manifest_sha256"],
                "chunk_count": 1,
                "chunk_sha256": item["chunk_sha256"],
                "complete_gap_free_duplicate_free_coverage": True,
                "selected_unique_raw_cell_id_count": 2,
                "selected_values_sha256": "6" * 64,
            }

        def native_audit(support: str, field: str, files, hashes):
            return {
                "support_id": support,
                "field_name": field,
                "total_row_count": 2,
                "selected_unique_raw_cell_id_count": 2,
                "selected_values_sha256": "7" * 64,
                "source_files": files,
                "source_sha256": hashes,
                "source_payload_sha256": "8" * 64,
                "complete_source_identity_verified": True,
            }

        return {
            "schema": "drivaerml-case-diagnostics-candidate-v3",
            "schema_version": 3,
            "status": "candidate_diagnostics_not_active_or_official_submission",
            "case_id": case_id,
            "official_submission": False,
            "mapping_inputs": {
                "velocity_10mm": {
                    "artifact_file": "velocity-cell-mapping-10mm.json",
                    "artifact_sha256": "a" * 64,
                    "receipt_file": f"{case_id}-velocity.json",
                    "receipt_sha256": "b" * 64,
                    "profile_sha256": self.profile_sha,
                    "row_count": 32,
                    "invalid_rows": [],
                }
            },
            "sparse_gather_evidence": {
                "volume_prediction": prediction_audit(
                    "volume_native_cells", "UMeanTrim"
                ),
                "volume_native_truth": native_audit(
                    "volume_native_cells",
                    "UMeanTrim",
                    [
                        f"volume_{case_index}.vtu.00.part",
                        f"volume_{case_index}.vtu.01.part",
                    ],
                    list(parts),
                ),
                "only_unique_mapped_raw_ids_retained": True,
                "prediction_manifest_fully_consumed": True,
            },
            "metrics": {
                "cp_cut_rmse": {
                    "metric_id": "cp_cut_rmse",
                    "ranked_value_available": False,
                    "required_cut_count": 4,
                    "unavailable_reasons": [
                        {
                            "diagnostic": "cp_cut_rmse",
                            "stage": "benchmark_support",
                            "reason": "immutable_native_cp_cut_extraction_support_not_published",
                        }
                    ],
                    "case_equal_cut_mean_rmse": None,
                    "aggregation": "equal_case_equal_cut_macro_average",
                    "weighting": "native_cut_intersection_segment_length",
                    "support_status": "pending_immutable_owner_release",
                    "discrete_cp_probe_fallback_used": False,
                    "cut_rmse": [],
                },
                "velocity_profile_uinf_rmse": {
                    "metric_id": "velocity_profile_uinf_rmse",
                    "ranked_value_available": True,
                    "required_line_count": 16,
                    "required_sample_count": 32,
                    "unavailable_reasons": [],
                    "line_rmse": [
                        {
                            "profile_id": f"line_{line:02d}",
                            "sample_count": 2,
                            "arc_length_m": 1.0,
                            "rmse": velocity_value,
                        }
                        for line in range(1, 17)
                    ],
                    "case_equal_line_mean_rmse": velocity_value,
                    "quantity": "magnitude(UMeanTrim)/Uinf",
                    "Uinf_m_per_s": 38.889,
                    "arc_rule": "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging",
                    "aggregation": "equal_case_equal_line_macro_average",
                    "weighting": "trapezoidal_arc_length_within_line",
                },
                "velocity_profile_experimental_subset_uinf_rmse": {
                    "metric_id": "velocity_profile_experimental_subset_uinf_rmse",
                    "value_available": True,
                    "required_line_count": 11,
                    "required_profile_ids": [
                        f"line_{line:02d}" for line in range(1, 12)
                    ],
                    "unavailable_reasons": [],
                    "line_rmse": [
                        {
                            "profile_id": f"line_{line:02d}",
                            "sample_count": 2,
                            "arc_length_m": 1.0,
                            "rmse": velocity_value,
                        }
                        for line in range(1, 12)
                    ],
                    "case_equal_experimental_line_mean_rmse": velocity_value,
                    "quantity": "magnitude(UMeanTrim)/Uinf",
                    "Uinf_m_per_s": 38.889,
                    "arc_rule": "trapezoidal_squared_error_over_owner_included_mapped_arc_no_gap_bridging",
                    "aggregation": "equal_case_equal_experimental_line_macro_average",
                    "weighting": "trapezoidal_arc_length_within_line",
                },
            },
            "claims": claims,
        }

    def enable_complete_profile_series(self) -> None:
        cut_ids = (
            "upperbody_centerline",
            "underbody_centerline",
            "sidewall_z_0_15",
            "front_left_wheelhouse_y_neg_0_6",
        )
        for case_index, case_id in enumerate(self.case_ids, start=1):
            path = self.diagnostic_dir / f"{case_id}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            cp_value = 0.2 * case_index
            document["metrics"]["cp_cut_rmse"] = {
                "metric_id": "cp_cut_rmse",
                "ranked_value_available": True,
                "required_cut_count": 4,
                "unavailable_reasons": [],
                "case_equal_cut_mean_rmse": cp_value,
                "aggregation": "equal_case_equal_cut_macro_average",
                "weighting": "native_cut_intersection_segment_length",
                "support_status": "immutable_owner_release",
                "discrete_cp_probe_fallback_used": False,
                "cut_rmse": [
                    {
                        "cut_id": cut_id,
                        "segment_count": 2,
                        "arc_length_m": 1.0,
                        "rmse": cp_value,
                    }
                    for cut_id in cut_ids
                ],
            }
            pressure_series = [
                {
                    "panel_id": "pressure_profiles",
                    "station_id": cut_id,
                    "quantity_id": "cp",
                    "coordinate": [0.0, 1.0],
                    "prediction": [0.1 * case_index, -0.1 * case_index],
                }
                for cut_id in cut_ids
            ]
            velocity_series = [
                {
                    "panel_id": "velocity_profiles",
                    "station_id": f"line_{line:02d}",
                    "quantity_id": "velocity_ratio",
                    "coordinate": [0.0, 1.0],
                    "prediction": [0.5, 1.0 + 0.01 * line],
                }
                for line in range(1, 17)
            ]
            document["profile_series"] = pressure_series + velocity_series
            _write_json(path, document)


    def evaluate(self):
        return evaluate_candidate_dataset(
            submission_specification=self.spec_path,
            split_id="tiny",
            force_truth_csv=self.force_path,
            core_case_evidence=tuple(self.core_dir.glob("*.json")),
            diagnostic_case_evidence=tuple(self.diagnostic_dir.glob("*.json")),
            immutable_pins=self.immutable_pins,
        )


class DrivAerDatasetScorerTests(unittest.TestCase):
    def test_json_and_force_truth_mutation_during_parse_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_path = root / "document.json"
            _write_json(document_path, {"value": 1})
            original_json_load = dataset_scorer_module.json.load

            def load_then_mutate(*args: object, **kwargs: object) -> object:
                value = original_json_load(*args, **kwargs)
                _write_json(document_path, {"value": "mutated-and-longer"})
                return value

            with mock.patch.object(
                dataset_scorer_module.json,
                "load",
                side_effect=load_then_mutate,
            ), self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "changed while its JSON was parsed",
            ):
                dataset_scorer_module._read_json(document_path, "test document")

            force_path = root / "force.csv"
            force_path.write_text(
                "run,cd,cl,clf,clr,cs\n1,0.1,0.2,0.12,0.08,0.0\n",
                encoding="utf-8",
            )
            expected_sha256 = _sha256(force_path)
            original_dict_reader = dataset_scorer_module.csv.DictReader

            def reader_then_mutate(*args: object, **kwargs: object) -> object:
                reader = original_dict_reader(*args, **kwargs)
                force_path.write_text(
                    "run,cd,cl,clf,clr,cs\n1,9,9,9,9,9\n",
                    encoding="utf-8",
                )
                return reader

            with mock.patch.object(
                dataset_scorer_module.csv,
                "DictReader",
                side_effect=reader_then_mutate,
            ), self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "changed while its CSV was parsed",
            ):
                dataset_scorer_module._load_force_truth(
                    force_path,
                    expected_sha256=expected_sha256,
                    expected_case_ids=("run_1",),
                )

    def test_json_and_force_truth_reject_non_regular_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            non_regular = Path(directory) / "directory-input"
            non_regular.mkdir()
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "not a regular file|Is a directory"
            ):
                dataset_scorer_module._read_json(non_regular, "test document")
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "not a regular file|Is a directory"
            ):
                dataset_scorer_module._load_force_truth(
                    non_regular,
                    expected_sha256="a" * 64,
                    expected_case_ids=("run_1",),
                )

    def test_known_numeric_golden_is_deterministic_and_candidate_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            evaluation = fixture.evaluate()
            evidence = evaluation.to_json()
            values = evidence["metric_values"]
            self.assertEqual(values["surface_pressure_rel_l2"], 12.0)
            self.assertEqual(values["surface_wall_shear_rel_l2"], 19.0)
            self.assertEqual(values["volume_pressure_rel_l2"], 39.0)
            self.assertAlmostEqual(
                values["volume_velocity_rel_l2"],
                (13.0 + 100.0 * math.sqrt(50.0 / 10000.0)) / 2.0,
            )
            self.assertAlmostEqual(values["field_integrated_cd_rmse"], math.sqrt(0.0005))
            self.assertAlmostEqual(values["field_integrated_cl_rmse"], math.sqrt(0.001))
            self.assertAlmostEqual(values["field_integrated_cmpitch_rmse"], math.sqrt(0.00005))
            self.assertAlmostEqual(values["field_integrated_clf_rmse"], math.sqrt(0.0005))
            self.assertAlmostEqual(values["field_integrated_clr_rmse"], 0.01)
            self.assertAlmostEqual(values["velocity_profile_uinf_rmse"], 0.2)
            self.assertAlmostEqual(
                values["velocity_profile_experimental_subset_uinf_rmse"], 0.2
            )
            self.assertIsNone(values["cp_cut_rmse"])
            expected_operations = {
                "velocity_profile_uinf_rmse": "equal_case_equal_line_macro_average",
                "velocity_profile_experimental_subset_uinf_rmse": (
                    "equal_case_equal_experimental_line_macro_average"
                ),
                "cp_cut_rmse": "equal_case_equal_cut_macro_average",
            }
            for metric_id, operation in expected_operations.items():
                self.assertEqual(
                    evidence["metric_reductions"][metric_id]["operation"],
                    operation,
                )
            self.assertEqual(
                evidence["metric_reductions"]["velocity_profile_uinf_rmse"][
                    "support_scope"
                ]["line_count_per_case"],
                16,
            )
            self.assertEqual(
                evidence["metric_reductions"][
                    "velocity_profile_experimental_subset_uinf_rmse"
                ]["support_scope"]["line_count_per_case"],
                11,
            )
            self.assertEqual(
                evidence["metric_reductions"]["cp_cut_rmse"]["support_scope"],
                {
                    "cut_count_per_case": 4,
                    "within_cut_weighting": "native_cut_intersection_segment_length",
                    "between_cut_weighting": "cuts_equal",
                    "support_status": "pending_immutable_owner_release",
                },
            )
            self.assertFalse(evidence["eligibility"]["official_submission"])
            self.assertFalse(evidence["eligibility"]["composite_score_available"])
            self.assertNotIn("overall_score", values)
            self.assertEqual(evidence["status"], CANDIDATE_DATASET_STATUS)
            self.assertEqual(
                evidence["schema"],
                "drivaerml-candidate-dataset-evaluation-v3",
            )
            self.assertEqual(evidence["schema_version"], 3)
            self.assertFalse(
                any("physical" in metric_id for metric_id in ALL_FIELD_METRIC_IDS)
            )
            self.assertEqual(
                {
                    metric_id
                    for metric_id in ALL_FIELD_METRIC_IDS
                    if "volume_" in metric_id
                },
                {
                    "volume_pressure_rel_l2",
                    "drivaerml_volume_pressure_equal_entity_mae",
                    "drivaerml_volume_pressure_equal_entity_rmse",
                    "volume_velocity_rel_l2",
                    "drivaerml_volume_velocity_equal_entity_mae",
                    "drivaerml_volume_velocity_equal_entity_rmse",
                },
            )
            source_contract = evidence["source_contract"]
            self.assertEqual(
                source_contract["volume_weighting"], "one_per_native_cell"
            )
            self.assertFalse(
                source_contract["geometric_cell_volume_weights_used"]
            )
            for removed_key in (
                "volume_weight_aggregate_sha256",
                "volume_weight_algorithm_sha256",
                "volume_weight_native_cell_type_payload_manifest_sha256",
                "volume_weight_per_vtk_cell_type",
                "volume_weight_aggregate_predeclared_in_submission_spec",
            ):
                self.assertNotIn(removed_key, source_contract)
            first_source = evidence["cases"][0]["source_support"]
            self.assertEqual(
                first_source["volume_weighting"],
                {
                    "weighting": "one_per_native_cell",
                    "entity_count": 2,
                    "total_weight": 2.0,
                    "geometric_cell_volume_weights_used": False,
                },
            )
            self.assertNotIn("volume_weight_sha256", first_source)
            self.assertNotIn("volume_weight_receipt_sha256", first_source)

            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            one = write_candidate_dataset_evidence(evaluation, first)
            two = write_candidate_dataset_evidence(fixture.evaluate(), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(one["sha256"], two["sha256"])

    def test_equal_native_cell_volume_contract_is_exact_and_versioned(self) -> None:
        mutations = (
            ("legacy schema", lambda value: value.update({"schema": "drivaerml-candidate-case-evaluation-v1", "schema_version": 1}), "schema/status mismatch"),
            ("wrong weighting", lambda value: value["source"]["volume_weighting"].update({"weighting": "cell_volume"}), "volume weighting mismatch"),
            ("wrong count", lambda value: value["source"]["volume_weighting"].update({"entity_count": 1}), "volume weighting mismatch"),
            ("non-float total", lambda value: value["source"]["volume_weighting"].update({"total_weight": 2}), "volume weighting mismatch"),
            ("geometric weights", lambda value: value["source"]["volume_weighting"].update({"geometric_cell_volume_weights_used": True}), "volume weighting mismatch"),
            ("algorithm identity", lambda value: value["source"]["volume_weighting"].update({"algorithm_sha256": "a" * 64}), "volume_weighting keys differ"),
        )
        for label, mutate, error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = DatasetFixture(Path(directory))
                path = fixture.core_dir / "run_1.json"
                document = json.loads(path.read_text(encoding="utf-8"))
                mutate(document)
                _write_json(path, document)
                with self.assertRaisesRegex(DrivAerDatasetScorerError, error):
                    fixture.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            path = fixture.core_dir / "run_1.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["source"]["volume_weights"] = document["source"].pop(
                "volume_weighting"
            )
            _write_json(path, document)
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "source keys differ"
            ):
                fixture.evaluate()

        legacy_physical_mutations = (
            (
                "metric value",
                lambda value: value["metric_values"].update(
                    {"volume_pressure_physical_rel_l2": 1.0}
                ),
                "metric_values keys differ",
            ),
            (
                "additive sums",
                lambda value: value["additive_sums"]["volume_pressure"].update(
                    {"physical": copy.deepcopy(value["additive_sums"]["volume_pressure"]["uniform"])}
                ),
                "additive_sums.volume_pressure keys differ",
            ),
            (
                "sufficient statistics",
                lambda value: value["metric_sufficient_statistics"].update(
                    {
                        "volume_pressure_physical_rel_l2": copy.deepcopy(
                            value["metric_sufficient_statistics"][
                                "volume_pressure_rel_l2"
                            ]
                        )
                    }
                ),
                "metric_sufficient_statistics keys differ",
            ),
        )
        for label, mutate, error in legacy_physical_mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = DatasetFixture(Path(directory))
                path = fixture.core_dir / "run_1.json"
                document = json.loads(path.read_text(encoding="utf-8"))
                mutate(document)
                _write_json(path, document)
                with self.assertRaisesRegex(DrivAerDatasetScorerError, error):
                    fixture.evaluate()

    def test_exact_split_coverage_rejects_wrong_missing_extra_and_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            core = tuple(fixture.core_dir.glob("*.json"))
            diagnostics = tuple(fixture.diagnostic_dir.glob("*.json"))
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "exactly one official split"):
                evaluate_candidate_dataset(
                    submission_specification=fixture.spec_path,
                    split_id="wrong",
                    force_truth_csv=fixture.force_path,
                    core_case_evidence=core,
                    diagnostic_case_evidence=diagnostics,
                    immutable_pins=fixture.immutable_pins,
                )
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "missing=.*run_2"):
                evaluate_candidate_dataset(
                    submission_specification=fixture.spec_path,
                    split_id="tiny",
                    force_truth_csv=fixture.force_path,
                    core_case_evidence=(fixture.core_dir / "run_1.json",),
                    diagnostic_case_evidence=diagnostics,
                    immutable_pins=fixture.immutable_pins,
                )
            extra = Path(directory) / "run_99.json"
            value = json.loads((fixture.core_dir / "run_1.json").read_text())
            value["case_id"] = "run_99"
            _write_json(extra, value)
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "extra=.*run_99"):
                evaluate_candidate_dataset(
                    submission_specification=fixture.spec_path,
                    split_id="tiny",
                    force_truth_csv=fixture.force_path,
                    core_case_evidence=(*core, extra),
                    diagnostic_case_evidence=diagnostics,
                    immutable_pins=fixture.immutable_pins,
                )
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "duplicate core case evidence path"):
                evaluate_candidate_dataset(
                    submission_specification=fixture.spec_path,
                    split_id="tiny",
                    force_truth_csv=fixture.force_path,
                    core_case_evidence=(core[0], core[0], core[1]),
                    diagnostic_case_evidence=diagnostics,
                    immutable_pins=fixture.immutable_pins,
                )
            duplicate_case = Path(directory) / "duplicate" / "run_1-copy.json"
            duplicate_case.parent.mkdir()
            duplicate_case.write_bytes((fixture.core_dir / "run_1.json").read_bytes())
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "duplicate core case evidence for case run_1",
            ):
                evaluate_candidate_dataset(
                    submission_specification=fixture.spec_path,
                    split_id="tiny",
                    force_truth_csv=fixture.force_path,
                    core_case_evidence=(core[0], duplicate_case, core[1]),
                    diagnostic_case_evidence=diagnostics,
                    immutable_pins=fixture.immutable_pins,
                )

    def test_source_and_cross_evidence_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.force_path.write_text(
                fixture.force_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "authoritative force truth SHA-256 mismatch",
            ):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            path = fixture.core_dir / "run_2.json"
            core = json.loads(path.read_text())
            core["source"]["native_source_pin_sha256"] = "f" * 64
            _write_json(path, core)
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "source identity mismatch"):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            path = fixture.diagnostic_dir / "run_2.json"
            diagnostic = json.loads(path.read_text())
            diagnostic["sparse_gather_evidence"]["volume_prediction"]["manifest_sha256"] = "f" * 64
            _write_json(path, diagnostic)
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "prediction identity mismatch"):
                fixture.evaluate()

        for metric_id, key, forged in (
            (
                "velocity_profile_uinf_rmse",
                "quantity",
                "interpolated_velocity/Uinf",
            ),
            (
                "velocity_profile_experimental_subset_uinf_rmse",
                "arc_rule",
                "average_available_points",
            ),
        ):
            with self.subTest(
                metric_id=metric_id, key=key
            ), tempfile.TemporaryDirectory() as directory:
                fixture = DatasetFixture(Path(directory))
                path = fixture.diagnostic_dir / "run_2.json"
                diagnostic = json.loads(path.read_text())
                diagnostic["metrics"][metric_id][key] = forged
                _write_json(path, diagnostic)
                with self.assertRaisesRegex(
                    DrivAerDatasetScorerError,
                    "velocity contract mismatch",
                ):
                    fixture.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            path = fixture.core_dir / "run_2.json"
            core = json.loads(path.read_text())
            core["source"]["surface_area"]["native_geometry_order_audit"][
                "raw_order_correspondence_verified"
            ] = False
            _write_json(path, core)
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "native geometry/area order audit mismatch"
            ):
                fixture.evaluate()

    def test_non_axis_aligned_vector_metrics_survive_schema_v3_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            evaluation = fixture.evaluate()
            adapter = schema_v3_case_metrics_candidate_adapter(
                evaluation,
                submission_id="synthetic-drivaerml-vectors",
                candidate_support_release_id="drivaerml-candidate-support-v1",
                candidate_support_manifest_sha256="e" * 64,
            )
            first_case = adapter["cases"][0]
            volume = next(
                support
                for support in first_case["supports"]
                if support["support_id"] == "volume_native_cells"
            )
            # Errors (3,4,0) and (0,0,12): mean per-entity Euclidean norm is
            # (5+12)/2 = 8.5 and RMS is sqrt((25+144)/2). These are not the
            # flattened-component results 19/6 and sqrt(169/6).
            self.assertEqual(
                volume["metric_values"]["drivaerml_volume_velocity_equal_entity_mae"],
                8.5,
            )
            self.assertAlmostEqual(
                volume["metric_values"]["drivaerml_volume_velocity_equal_entity_rmse"],
                math.sqrt(169.0 / 2.0),
            )
            self.assertNotAlmostEqual(
                volume["metric_values"]["drivaerml_volume_velocity_equal_entity_mae"],
                19.0 / 6.0,
            )
            self.assertEqual(adapter["case_count"], 2)
            self.assertNotIn("overall_score", adapter["metric_values"])
            nonspatial = first_case["nonspatial_metric_values"]
            self.assertEqual(
                first_case["force_coefficients"],
                {
                    "cd": 0.11,
                    "cl": 0.22,
                    "cm_pitch": 0.02,
                    "clf": 0.13,
                    "clr": 0.09,
                },
            )
            for metric_id in (
                "field_integrated_cd_rmse",
                "field_integrated_cl_rmse",
                "field_integrated_cmpitch_rmse",
                "field_integrated_clf_rmse",
                "field_integrated_clr_rmse",
                "field_integrated_lift_closure_max_abs",
            ):
                self.assertIn(metric_id, nonspatial)
            self.assertFalse(
                any(metric_id.endswith("_abs_error") for metric_id in nonspatial)
            )
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "schemas"
                    / "v3"
                    / "case-metrics.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                list(Draft202012Validator(schema).iter_errors(adapter)), []
            )
            with self.assertRaisesRegex(DrivAerDatasetScorerError, "explicitly containing 'candidate'"):
                schema_v3_case_metrics_candidate_adapter(
                    evaluation,
                    submission_id="synthetic-drivaerml-vectors",
                    candidate_support_release_id="official-support-v1",
                    candidate_support_manifest_sha256="e" * 64,
                )

    def test_schema_v3_nonspatial_values_are_semantically_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            adapter = schema_v3_case_metrics_candidate_adapter(
                fixture.evaluate(),
                submission_id="synthetic-drivaerml-binding",
                candidate_support_release_id="drivaerml-candidate-support-v1",
                candidate_support_manifest_sha256="e" * 64,
            )
            first = validate_schema_v3_candidate_nonspatial_metrics(adapter)
            second = validate_schema_v3_candidate_nonspatial_metrics(
                copy.deepcopy(adapter)
            )
            self.assertEqual(first, second)
            self.assertEqual(first["case_count"], 2)
            self.assertEqual(len(first["nonspatial_values_sha256"]), 64)

            tampered = copy.deepcopy(adapter)
            tampered["cases"][0]["nonspatial_metric_values"][
                "field_integrated_cd_rmse"
            ] += 0.25
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "field_integrated_cd_rmse differs from the DrivAerML per-case reduction",
            ):
                validate_schema_v3_candidate_nonspatial_metrics(tampered)

            partial = copy.deepcopy(adapter)
            del partial["cases"][0]["nonspatial_metric_values"][
                "velocity_profile_uinf_rmse"
            ]
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "must be present for every case or omitted for every case",
            ):
                validate_schema_v3_candidate_nonspatial_metrics(partial)

            invented = copy.deepcopy(adapter)
            invented["cases"][0]["nonspatial_metric_values"]["invented"] = 1.0
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "contains undeclared DrivAerML nonspatial metrics",
            ):
                validate_schema_v3_candidate_nonspatial_metrics(invented)

            forged_force = copy.deepcopy(adapter)
            forged_force["cases"][0]["force_coefficients"]["clf"] += 0.1
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "force_coefficients.clf differs",
            ):
                validate_schema_v3_candidate_nonspatial_metrics(forged_force)

            missing_force = copy.deepcopy(adapter)
            del missing_force["cases"][0]["force_coefficients"]["cm_pitch"]
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "force_coefficients.*missing",
            ):
                validate_schema_v3_candidate_nonspatial_metrics(missing_force)

    def test_profile_chunk_adapter_fails_closed_then_packages_all_twenty_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "required diagnostics are unavailable.*cp_cut_rmse",
            ):
                schema_v3_profile_chunks_candidate_adapter(
                    fixture.evaluate(),
                    submission_id="synthetic-drivaerml-profiles",
                )

            fixture.enable_complete_profile_series()
            package = schema_v3_profile_chunks_candidate_adapter(
                fixture.evaluate(),
                submission_id="synthetic-drivaerml-profiles",
                cases_per_chunk=1,
            )
            index = package.index_json()
            self.assertEqual(index["case_count"], 2)
            self.assertEqual(len(index["chunks"]), 2)
            self.assertEqual(
                index["chunks"][0]["case_ids"], ["run_1"]
            )
            first_chunk = package.chunk_json("chunk-000.json")
            self.assertEqual(len(first_chunk["cases"]), 1)
            self.assertEqual(len(first_chunk["cases"][0]["series"]), 20)
            self.assertEqual(
                first_chunk["cases"][0]["series"][0]["station_id"],
                "upperbody_centerline",
            )
            self.assertEqual(
                first_chunk["cases"][0]["series"][-1]["station_id"],
                "line_16",
            )

            schema_root = Path(__file__).resolve().parents[1] / "schemas" / "v1"
            index_schema = json.loads(
                (schema_root / "profile-index.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            chunk_schema = json.loads(
                (schema_root / "profile-chunk.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                list(Draft202012Validator(index_schema).iter_errors(index)), []
            )
            self.assertEqual(
                list(
                    Draft202012Validator(chunk_schema).iter_errors(first_chunk)
                ),
                [],
            )

            output = Path(directory) / "profiles"
            receipt = write_schema_v3_profile_chunks_candidate(package, output)
            self.assertEqual(receipt["case_count"], 2)
            self.assertEqual(receipt["series_count"], 40)
            self.assertEqual(receipt["chunk_count"], 2)
            self.assertTrue((output / "index.json").is_file())
            self.assertEqual(
                _sha256(output / "index.json"), receipt["index_sha256"]
            )
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError, "must not already exist"
            ):
                write_schema_v3_profile_chunks_candidate(package, output)

    def test_profile_series_reject_partial_cp_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.enable_complete_profile_series()
            path = fixture.diagnostic_dir / "run_1.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            del document["profile_series"][2]
            _write_json(path, document)
            with self.assertRaisesRegex(
                DrivAerDatasetScorerError,
                "Cp-cut profile series are incomplete",
            ):
                fixture.evaluate()

    def test_incomplete_diagnostic_support_is_never_subset_reduced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            path = fixture.diagnostic_dir / "run_2.json"
            document = json.loads(path.read_text())
            document["mapping_inputs"]["velocity_10mm"]["invalid_rows"] = [
                {
                    "profile_id": "line_01",
                    "sample_index": 0,
                    "valid": False,
                    "reason": "synthetic_mapping_failure",
                    "raw_vtk_cell_id": None,
                    "candidate_count": 0,
                }
            ]
            for metric_id, availability_key, value_key in (
                (
                    "velocity_profile_uinf_rmse",
                    "ranked_value_available",
                    "case_equal_line_mean_rmse",
                ),
                (
                    "velocity_profile_experimental_subset_uinf_rmse",
                    "value_available",
                    "case_equal_experimental_line_mean_rmse",
                ),
            ):
                metric = document["metrics"][metric_id]
                metric[availability_key] = False
                metric[value_key] = None
                metric["line_rmse"] = []
                metric["unavailable_reasons"] = [
                    {
                        "diagnostic": metric_id,
                        "stage": "mapping",
                        "reason": "synthetic_mapping_failure",
                        "profile_id": "line_01",
                        "sample_index": 0,
                    }
                ]
            _write_json(path, document)

            evaluation = fixture.evaluate()
            evidence = evaluation.to_json()
            self.assertIsNone(evidence["metric_values"]["cp_cut_rmse"])
            cp_availability = evidence["diagnostic_availability"]["cp_cut_rmse"]
            self.assertFalse(cp_availability["available"])
            self.assertEqual(cp_availability["available_case_count"], 0)
            velocity_availability = evidence["diagnostic_availability"][
                "velocity_profile_uinf_rmse"
            ]
            self.assertFalse(velocity_availability["available"])
            self.assertEqual(velocity_availability["available_case_count"], 1)
            self.assertEqual(velocity_availability["omitted_case_count"], 0)
            adapter = schema_v3_case_metrics_candidate_adapter(
                evaluation,
                submission_id="synthetic-incomplete-support",
                candidate_support_release_id="drivaerml-candidate-support-v1",
                candidate_support_manifest_sha256="e" * 64,
            )
            for metric_id in (
                "cp_cut_rmse",
                "velocity_profile_uinf_rmse",
                "velocity_profile_experimental_subset_uinf_rmse",
            ):
                self.assertNotIn(metric_id, adapter["metric_values"])
                for case in adapter["cases"]:
                    self.assertNotIn(
                        metric_id, case["nonspatial_metric_values"]
                    )



if __name__ == "__main__":
    unittest.main()
