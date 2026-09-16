from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark-specs" / "drivaerml"
EVIDENCE_ROOT = BENCHMARK / "evidence"
MANIFEST_PATH = EVIDENCE_ROOT / "manifest.json"
SPEC_PATH = BENCHMARK / "submission-spec.json"

VELOCITY_PATH = (
    EVIDENCE_ROOT / "velocity-mapping-run1-run44-candidate-v7-pilot.json"
)
EXPECTED_VELOCITY_SHA256 = (
    "91a7bb4cb7b7c6167c3577f62745021409c93cebc6b2685a57cdfe1debc06d76"
)
PROVENANCE_PATH = (
    EVIDENCE_ROOT
    / "velocity-run1-run44-candidate-v7-pilot-provenance.json"
)
EXPECTED_PROVENANCE_SHA256 = (
    "be785d01618584691a60bba9faa8125376ac788dae1cfeeaf763e97fc7ff767e"
)
DRIVER_PATH = (
    EVIDENCE_ROOT
    / "real-reference-driver-run1-run44-candidate-v7-pilot.json"
)
EXPECTED_DRIVER_SHA256 = (
    "da80aebb11eb18d3bb66b7a968a94e9a779a6e60f5413ef73d2fbd6485d76572"
)
DRIVER_PROVENANCE_PATH = (
    EVIDENCE_ROOT
    / "real-reference-driver-run1-run44-candidate-v7-pilot-provenance.json"
)
EXPECTED_DRIVER_PROVENANCE_SHA256 = (
    "24fd18d70098e5943959c8f37e2cd0ade211322e0785734d0dd035200db8fff2"
)

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_LOCAL_PATH_FRAGMENTS = ("/home/", "/lustre/", "/tmp/", "file://")

EXPECTED_CASES = {
    "run_1": {
        "point_count": 165_769_088,
        "cell_count": 147_449_586,
        "part_count": 2,
        "receipt_sha256": (
            "00de7962877c5285f24f56566b2b5e4a39705baa4e35591130920784b916391f"
        ),
        "artifacts": {
            1: "99ea27601049a32f3df6d0aff321ab27c5e1393bbcae4384d1d598370c1e79a7",
            2: "9e83a3f42b12b112f452b883bf73174f9477eb751c890f9d56b806610424102d",
            5: "9f93858d701d409971d44528dced9d1d086f483aef47d29fed45aa6c052bf91e",
            10: "4b90cb0ed4dc79d3bbaf007668ee283306c4ef13688ca8f678405a86ca632cef",
        },
    },
    "run_44": {
        "point_count": 183_505_555,
        "cell_count": 163_398_798,
        "part_count": 3,
        "receipt_sha256": (
            "3654486f3e9abff1fcc725d9420a8c7ed9aa8543fe3e6066582a54807bb654a2"
        ),
        "artifacts": {
            1: "ebc61a165b0989fdde921cb125bc627401b11c5032e987f090d57c78f5a8dfbd",
            2: "fd999ea1a4eae8013e9e30b29efa3e5469727aabe1986cbd398f8a89a44f2f29",
            5: "d2427d62e4b26e6a9a35dd7a8fab561b348fff6dc93d3c06a28ccbe37982e3b6",
            10: "5d059b882edfc09ee485af8ef4f9b6d4de60d3affa4a81d68a046f3b0362f5a9",
        },
    },
}

EXPECTED_DRIVER_CHILDREN = {
    "run_1": {
        "part_count": 2,
        "core": (
            18_382,
            "d134ae14f8e2e4656f0cf2fdce7da35294fb5a7caed45e626d4183da5d16c7e3",
        ),
        "diagnostic": (
            91_082,
            "4ed73efb18607dcde63324c699386d508191be57e39edcc50b046933a249038e",
        ),
    },
    "run_44": {
        "part_count": 3,
        "core": (
            19_693,
            "d8a6f064b600e1f210da04d5c0c39df2fa521e5567ec0686e37f284cacd65639",
        ),
        "diagnostic": (
            63_979,
            "8ea80bb6676ab7774704587bd3dc9b0a726877fc1c7dd7eda7a38daee2db3b81",
        ),
    },
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_git_blob(revision: str, path: str) -> str:
    payload = subprocess.run(
        ("git", "show", f"{revision}:{path}"),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _walk_strings(value: object, location: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{location}[{index}]")
    elif isinstance(value, str):
        yield location, value


def _manifest_entry(file_name: str) -> dict:
    matches = [
        entry
        for entry in _load(MANIFEST_PATH)["artifacts"]
        if entry["file"] == file_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one manifest entry for {file_name!r}, got {len(matches)}"
        )
    return matches[0]


class DrivAerMLVelocityPilotEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.velocity = _load(VELOCITY_PATH)

    def assert_path_free(self, value: object) -> None:
        for location, text in _walk_strings(value):
            with self.subTest(location=location):
                self.assertFalse(text.startswith(("/", "~", "\\\\")), text)
                self.assertIsNone(re.match(r"^[A-Za-z]:[\\/]", text), text)
                for fragment in FORBIDDEN_LOCAL_PATH_FRAGMENTS:
                    self.assertNotIn(fragment, text, text)

    def test_velocity_aggregate_is_exact_nonactivating_pilot(self) -> None:
        evidence = self.velocity
        self.assertEqual(_sha256(VELOCITY_PATH), EXPECTED_VELOCITY_SHA256)
        self.assertEqual(
            evidence["schema"],
            "drivaerml-velocity-cell-assignments-all-case-candidate-v1",
        )
        self.assertEqual(evidence["schema_version"], 1)
        self.assertEqual(evidence["mode"], "explicit_non_public_pilot")
        self.assertEqual(
            evidence["status"],
            "incomplete_explicit_pilot_not_public_or_activation_evidence",
        )
        self.assertEqual(
            evidence["case_scope"],
            {
                "case_count": 2,
                "complete_484_public_case_set": False,
                "explicit_pilot_case_ids": ["run_1", "run_44"],
            },
        )
        self.assertEqual(
            set(evidence["claims"]),
            {
                "all_case_tolerance_replay_complete",
                "independent_participant_dry_run",
                "model_ordering",
                "official_submission",
                "owner_scientific_signoff",
                "owner_validity_mask_complete",
                "ranked_result_invariance",
                "resolution_convergence",
                "scoring_contract_active",
                "three_real_model_ordering",
            },
        )
        self.assertTrue(all(claim is False for claim in evidence["claims"].values()))
        self.assert_path_free(evidence)

    def test_velocity_source_kernel_and_tolerance_bindings_are_exact(self) -> None:
        source = self.velocity["source_bindings"]
        self.assertEqual(source["repository_id"], "neashton/drivaerml")
        self.assertEqual(
            source["repository_revision"],
            "7a5c0948ce27be709b1116a3a190f806e7a8f79f",
        )
        self.assertEqual(
            source["native_source_pin_sha256"],
            "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd",
        )
        self.assertEqual(
            source["autocfd5"]["profile_sha256"],
            "17d830087d11e83e3cba75358f33fdd827421be6698ba1624e547ae36f359184",
        )
        self.assertEqual(source["autocfd5"]["line_count"], 16)
        self.assertEqual(source["autocfd5"]["fixed_10mm_sample_count"], 3_756)

        kernel = self.velocity["kernel"]
        self.assertEqual(
            kernel["kernel_id"], "drivaerml-native-containing-cell-candidate-v7"
        )
        self.assertEqual(
            kernel["settings_sha256"],
            "6cbd2b2fb56fc782fd9e9990bd43f2bad7fd040b355f128555ecf101b369fa1b",
        )
        self.assertEqual(
            kernel["versions"],
            {
                "numpy": "2.2.6",
                "python": "3.12.13",
                "vtk": "9.5.2",
                "vtk_source": "vtk version 9.5.2",
            },
        )
        settings = kernel["settings"]
        self.assertEqual(settings["primary_tolerance_m"], 1e-6)
        self.assertEqual(settings["required_tolerance_replay_m"], [5e-7, 1e-6, 2e-6])
        self.assertEqual(
            settings["status"],
            "candidate_pending_all_case_replay_and_owner_approval",
        )
        polyhedron = settings["closure"]["native_vtk_polyhedron"]
        self.assertEqual(
            polyhedron["classification_absolute_tolerance_steradian"], 1e-3
        )
        self.assertEqual(
            polyhedron["classification_tolerance_status"],
            "candidate_scientific_choice_pending_owner_approval",
        )
        self.assertFalse(polyhedron["vtk_9_5_2_IsInside_called"])
        self.assertFalse(polyhedron["geometry_cache"]["vtk_objects_cached"])

        tolerance = self.velocity["tolerance_evidence"]
        self.assertEqual(tolerance["primary_geometric_tolerance_m"], 1e-6)
        self.assertEqual(tolerance["kernel_required_replay_m"], [5e-7, 1e-6, 2e-6])
        self.assertTrue(tolerance["requested_case_primary_tolerance_rows_complete"])
        for key in (
            "all_484_primary_tolerance_rows_complete",
            "half_one_two_micrometre_replay_complete",
            "non_face_assignment_invariance_claim",
            "ranked_result_invariance_claim",
        ):
            self.assertFalse(tolerance[key])

    def test_velocity_case_and_resolution_coverage_is_exact(self) -> None:
        evidence = self.velocity
        cases = {case["case_id"]: case for case in evidence["cases"]}
        self.assertEqual(set(cases), set(EXPECTED_CASES))
        expected_samples = {1: 37_416, 2: 18_716, 5: 7_496, 10: 3_756}

        for case_id, expected in EXPECTED_CASES.items():
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                self.assertEqual(
                    case["geometry"]["point_count"], expected["point_count"]
                )
                self.assertEqual(case["geometry"]["cell_count"], expected["cell_count"])
                self.assertEqual(
                    case["geometry"]["ordered_verified_segment_count"],
                    expected["part_count"],
                )
                self.assertEqual(case["receipt_sha256"], expected["receipt_sha256"])
                self.assertRegex(case["receipt_sha256"], HEX_SHA256)
                resolutions = {
                    row["nominal_spacing_mm"]: row for row in case["resolutions"]
                }
                self.assertEqual(set(resolutions), {1, 2, 5, 10})
                for spacing_mm, row in resolutions.items():
                    self.assertEqual(row["line_count"], 16)
                    self.assertEqual(row["sample_count"], expected_samples[spacing_mm])
                    self.assertEqual(
                        row["valid_count"] + row["invalid_count"],
                        row["sample_count"],
                    )
                    self.assertEqual(
                        sum(row["invalid_reason_counts"].values()),
                        row["invalid_count"],
                    )
                    self.assertTrue(row["complete_duplicate_free_no_omissions"])
                    self.assertEqual(
                        row["artifact_name"],
                        f"velocity-cell-mapping-{spacing_mm:02d}mm.json",
                    )
                    self.assertEqual(
                        row["artifact_sha256"], expected["artifacts"][spacing_mm]
                    )
                    self.assertRegex(row["assignment_evidence_sha256"], HEX_SHA256)

        totals = evidence["totals"]
        self.assertEqual(totals["case_count"], 2)
        self.assertEqual(totals["ordered_verified_segment_count"], 5)
        self.assertEqual(totals["geometry_point_count_sum"], 349_274_643)
        self.assertEqual(totals["geometry_cell_count_sum"], 310_848_384)
        self.assertEqual(totals["explicit_assignment_row_count"], 134_768)
        expected_totals = {
            1: (74_832, 72_355, 2_477),
            2: (37_432, 36_184, 1_248),
            5: (14_992, 14_487, 505),
            10: (7_512, 7_253, 259),
        }
        by_resolution = {
            int(spacing): row
            for spacing, row in totals["by_resolution_mm"].items()
        }
        self.assertEqual(set(by_resolution), set(expected_totals))
        for spacing_mm, (samples, valid, invalid) in expected_totals.items():
            row = by_resolution[spacing_mm]
            self.assertEqual(row["case_count"], 2)
            self.assertEqual(
                (row["sample_count"], row["valid_count"], row["invalid_count"]),
                (samples, valid, invalid),
            )
            self.assertEqual(valid + invalid, samples)
        self.assertEqual(
            sum(row["sample_count"] for row in by_resolution.values()),
            totals["explicit_assignment_row_count"],
        )
        self.assertEqual(sum(totals["invalid_reason_counts"].values()), 4_489)
        self.assertEqual(
            totals["invalid_reason_counts"][
                "no_native_cell_within_closure_tolerance"
            ],
            4_476,
        )
        self.assertEqual(
            sum(
                count
                for reason, count in totals["invalid_reason_counts"].items()
                if reason != "no_native_cell_within_closure_tolerance"
            ),
            13,
        )

    def test_velocity_runtime_audits_are_complete_and_internally_consistent(self) -> None:
        totals = self.velocity["totals"]
        query_cache = totals["containing_cell_query_cache"]
        self.assertEqual(query_cache["audited_receipt_count"], 2)
        self.assertEqual(query_cache["missing_pre_cache_pilot_receipt_count"], 0)
        self.assertEqual(query_cache["total_rows"], 134_768)
        self.assertEqual(query_cache["unique_query_keys_sum"], 78_724)
        self.assertEqual(query_cache["cache_hits"], 56_044)
        self.assertEqual(
            query_cache["unique_query_keys_sum"] + query_cache["cache_hits"],
            query_cache["total_rows"],
        )

        geometry_cache = totals["polyhedron_geometry_cache"]
        self.assertEqual(geometry_cache["audited_receipt_count"], 2)
        self.assertEqual(geometry_cache["policy"], "deterministic_least_recently_used")
        self.assertEqual(geometry_cache["cache_hits"], 9_390)
        self.assertEqual(geometry_cache["cache_misses"], 451)
        self.assertEqual(geometry_cache["current_entries"], 451)
        self.assertEqual(geometry_cache["current_emitted_triangles"], 10_246)
        self.assertEqual(geometry_cache["peak_entries_max"], 294)
        self.assertEqual(geometry_cache["peak_emitted_triangles_max"], 6_590)
        for key in (
            "evictions",
            "fail_closed_preparations",
            "oversized_entry_bypasses",
        ):
            self.assertEqual(geometry_cache[key], 0)
        self.assertLessEqual(
            geometry_cache["peak_entries_max"],
            geometry_cache["maximum_entries_per_case"],
        )
        self.assertLessEqual(
            geometry_cache["peak_emitted_triangles_max"],
            geometry_cache["maximum_emitted_triangles_per_case"],
        )

        evaluation = totals["polyhedron_evaluation"]
        self.assertEqual(evaluation["audited_receipt_count"], 2)
        self.assertEqual(evaluation["broad_phase_polyhedron_visit_count"], 9_841)
        self.assertEqual(evaluation["boundary_count"], 3_621)
        self.assertEqual(evaluation["inside_count"], 3_601)
        self.assertEqual(evaluation["outside_count"], 2_619)
        self.assertEqual(evaluation["ambiguous_count"], 0)
        self.assertEqual(evaluation["winding_classified_count"], 6_220)
        self.assertEqual(
            evaluation["inside_count"] + evaluation["outside_count"],
            evaluation["winding_classified_count"],
        )
        self.assertEqual(
            evaluation["boundary_count"]
            + evaluation["winding_classified_count"]
            + evaluation["ambiguous_count"],
            evaluation["broad_phase_polyhedron_visit_count"],
        )
        self.assertEqual(
            geometry_cache["cache_hits"] + geometry_cache["cache_misses"],
            evaluation["broad_phase_polyhedron_visit_count"],
        )
        self.assertEqual(
            evaluation["classification_absolute_tolerance_steradian"], 1e-3
        )
        self.assertLessEqual(
            abs(
                evaluation["minimum_winding_classification_margin_steradian"]
                - evaluation["classification_absolute_tolerance_steradian"]
            ),
            3e-13,
        )

        for case in self.velocity["cases"]:
            query = case["containing_cell_query_cache"]
            self.assertTrue(query["enabled"])
            self.assertEqual(query["total_rows"], 67_384)
            self.assertEqual(query["unique_query_keys"], 39_362)
            self.assertEqual(query["cache_hits"], 28_022)
            self.assertEqual(
                query["unique_query_keys"] + query["cache_hits"],
                query["total_rows"],
            )
            cache = case["polyhedron_geometry_cache"]
            audit = case["polyhedron_evaluation"]
            self.assertFalse(cache["vtk_objects_cached"])
            self.assertEqual(cache["evictions"], 0)
            self.assertEqual(cache["fail_closed_preparations"], 0)
            self.assertEqual(cache["oversized_entry_bypasses"], 0)
            self.assertEqual(
                cache["cache_hits"] + cache["cache_misses"],
                audit["broad_phase_polyhedron_visit_count"],
            )
            self.assertEqual(
                audit["boundary_count"]
                + audit["inside_count"]
                + audit["outside_count"]
                + audit["ambiguous_count"],
                audit["broad_phase_polyhedron_visit_count"],
            )

    def test_velocity_aggregate_manifest_binding_is_non_public(self) -> None:
        entry = _manifest_entry(VELOCITY_PATH.name)
        self.assertEqual(entry["sha256"], EXPECTED_VELOCITY_SHA256)
        self.assertEqual(entry["scope"], "run_1_run_44_non_public_pilot")
        self.assertEqual(
            entry["role"],
            "candidate_velocity_containing_cell_geometry_assignment_four_resolution_pilot",
        )
        self.assertFalse(entry["public_scoring_support_eligible"])

    @unittest.skipUnless(PROVENANCE_PATH.is_file(), "pilot provenance not packaged yet")
    def test_velocity_provenance_is_path_free_and_manifest_bound(self) -> None:
        provenance = _load(PROVENANCE_PATH)
        self.assertEqual(_sha256(PROVENANCE_PATH), EXPECTED_PROVENANCE_SHA256)
        self.assert_path_free(provenance)
        entry = _manifest_entry(PROVENANCE_PATH.name)
        self.assertEqual(entry["sha256"], EXPECTED_PROVENANCE_SHA256)
        self.assertEqual(entry["scope"], "run_1_run_44_non_public_pilot")
        self.assertEqual(
            entry["role"],
            "path_free_code_runtime_launcher_repeat_and_external_velocity_pilot_identity_binding",
        )
        self.assertFalse(entry["public_scoring_support_eligible"])
        self.assertEqual(
            provenance["schema"],
            "drivaerml-velocity-run1-run44-candidate-v7-pilot-provenance-v1",
        )
        self.assertEqual(provenance["schema_version"], 1)
        self.assertEqual(
            provenance["status"], "complete_path_free_non_public_pilot_provenance"
        )
        self.assertEqual(
            provenance["aggregate"],
            {
                "file": VELOCITY_PATH.name,
                "size_bytes": 17_981,
                "sha256": EXPECTED_VELOCITY_SHA256,
            },
        )

        external = provenance["external_case_artifacts"]
        self.assertEqual(external["case_count"], 2)
        self.assertEqual(external["case_ids"], ["run_1", "run_44"])
        self.assertEqual(external["total_receipt_size_bytes"], 23_674)
        external_cases = {case["case_id"]: case for case in external["cases"]}
        self.assertEqual(set(external_cases), set(EXPECTED_CASES))
        for case_id, expected in EXPECTED_CASES.items():
            case = external_cases[case_id]
            self.assertEqual(case["receipt"]["sha256"], expected["receipt_sha256"])
            mappings = {
                int(row["file"].split("-")[-1][:-7]): row
                for row in case["mappings"]
            }
            self.assertEqual(set(mappings), {1, 2, 5, 10})
            for spacing_mm, digest in expected["artifacts"].items():
                self.assertEqual(mappings[spacing_mm]["sha256"], digest)

        implementation = provenance["implementation"]
        revision = "31ab982b7874a3abe4e7a74b40359be9526f3072"
        self.assertEqual(implementation["git_revision"], revision)
        self.assertTrue(implementation["snapshot_clean_at_execution"])
        self.assertEqual(
            implementation["runtime"],
            {
                "python": "3.12.13",
                "numpy": "2.2.6",
                "vtk": "9.5.2",
                "vtk_source": "vtk version 9.5.2",
            },
        )
        self.assertEqual(len(implementation["source_files"]), 6)
        for source in implementation["source_files"]:
            self.assertTrue(source["matches_git_revision"])
            self.assertEqual(
                _sha256_git_blob(revision, source["file"]), source["sha256"]
            )
        for immutable_input in implementation["immutable_inputs"]:
            path = ROOT / immutable_input["file"]
            self.assertEqual(path.stat().st_size, immutable_input["size_bytes"])
            self.assertEqual(_sha256(path), immutable_input["sha256"])
        self.assertEqual(len(implementation["launchers"]), 4)
        for launcher in implementation["launchers"]:
            self.assertEqual(
                launcher["storage"], "external_campaign_file_not_bundled_in_git"
            )
            self.assertRegex(launcher["sha256"], HEX_SHA256)
            self.assertGreater(launcher["size_bytes"], 0)

        execution = provenance["execution"]
        self.assertEqual(
            execution["scheduler"],
            {
                "account": "coreai_modulus_cae",
                "partition": "cpu",
                "qos": "cpu-normal",
                "cpus_per_task": 1,
                "thread_caps_per_process": 1,
            },
        )
        primary_tasks = execution["primary_mapping_array"]["tasks"]
        self.assertEqual(
            [
                (task["case_id"], task["state"], task["exit_code"])
                for task in primary_tasks
            ],
            [("run_1", "COMPLETED", "0:0"), ("run_44", "COMPLETED", "0:0")],
        )
        repeat = execution["separate_process_repeat"]
        self.assertEqual(repeat["case_id"], "run_1")
        self.assertEqual((repeat["state"], repeat["exit_code"]), ("COMPLETED", "0:0"))
        aggregate = execution["strict_aggregation"]
        self.assertEqual(
            (aggregate["state"], aggregate["exit_code"]), ("COMPLETED", "0:0")
        )
        comparison = execution["repeat_comparison"]
        self.assertEqual(
            (comparison["state"], comparison["exit_code"]), ("COMPLETED", "0:0")
        )
        self.assertTrue(comparison["all_five_files_identical"])
        comparison_identities = {
            row["file"]: row["sha256"] for row in comparison["identities"]
        }
        run_1 = external_cases["run_1"]
        expected_repeat_identities = {
            run_1["receipt"]["file"]: run_1["receipt"]["sha256"],
            **{row["file"]: row["sha256"] for row in run_1["mappings"]},
        }
        self.assertEqual(comparison_identities, expected_repeat_identities)
        self.assertEqual(
            execution["parameters"],
            {
                "io_chunk_bytes": 16_777_216,
                "validation_chunk_cells": 1_000_000,
                "primary_geometric_tolerance_m": 1e-6,
                "resolution_order_mm": [1, 2, 5, 10],
            },
        )

        scope = provenance["scientific_scope"]
        self.assertTrue(scope["explicit_non_public_pilot"])
        self.assertTrue(scope["native_geometry_only_no_truth_field_arrays_loaded"])
        self.assertTrue(scope["requested_case_primary_tolerance_rows_complete"])
        self.assertEqual(scope["case_count"], 2)
        self.assertEqual(
            scope["candidate_polyhedron_classification_absolute_tolerance_steradian"],
            1e-3,
        )
        for key in (
            "model_checkpoint_used",
            "model_predictions_used",
            "complete_484_case_mapping",
            "all_484_primary_tolerance_rows_complete",
            "required_half_one_two_micrometre_replay_complete",
            "candidate_polyhedron_classification_tolerance_owner_approved",
            "owner_validity_mask_complete",
            "resolution_convergence",
            "ranked_result_invariance",
            "three_real_model_ordering",
            "physics_null_baseline",
            "independent_participant_dry_run",
            "public_scoring_support_eligible",
            "official_submission_scoring_enabled",
            "scoring_contract_active",
            "owner_scientific_approval",
        ):
            self.assertFalse(scope[key])

    @unittest.skipUnless(
        DRIVER_PATH.is_file(), "reference-driver pilot not packaged yet"
    )
    def test_reference_driver_pilot_is_nonactivating_and_manifest_bound(self) -> None:
        receipt = _load(DRIVER_PATH)
        self.assertEqual(DRIVER_PATH.stat().st_size, 2_972)
        self.assertEqual(_sha256(DRIVER_PATH), EXPECTED_DRIVER_SHA256)
        self.assert_path_free(receipt)
        entry = _manifest_entry(DRIVER_PATH.name)
        self.assertEqual(entry["sha256"], EXPECTED_DRIVER_SHA256)
        self.assertEqual(entry["scope"], "run_1_run_44_non_public_pilot")
        self.assertEqual(
            entry["role"],
            "real_native_inputs_zero_prediction_transport_and_reduction_reference_driver_pilot",
        )
        self.assertFalse(entry["public_scoring_support_eligible"])
        self.assertEqual(
            receipt["schema"], "drivaerml-run1-run44-reference-evidence-v3"
        )
        self.assertEqual(receipt["schema_version"], 3)
        self.assertEqual(
            receipt["status"], "candidate_pilot_evidence_not_official_submission"
        )
        self.assertEqual(
            receipt["native_source_pin_sha256"],
            "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd",
        )
        self.assertEqual(
            receipt["autocfd5_profile_sha256"],
            "17d830087d11e83e3cba75358f33fdd827421be6698ba1624e547ae36f359184",
        )
        self.assertEqual(
            receipt["case_input_config_sha256"],
            "4cc9cc69d71a64f122802120d4e65c9b11b186fbca39b0c42f3f44a2e3f3f3e8",
        )
        self.assertEqual(receipt["volume_weighting"], "one_per_native_cell")
        for key in (
            "complete_484_case_split_evaluated",
            "downloads_performed",
            "fabricated_scientific_results",
            "geometric_cell_volume_weights_used",
            "independent_participant_dry_run",
            "official_submission",
            "owner_scientific_approval",
            "public_scoring_support_eligible",
            "scoring_contract_active",
        ):
            self.assertFalse(receipt[key])
        self.assertFalse(receipt["evaluator_binding"]["frozen_release"])
        repository = receipt["evaluator_binding"]["repository"]
        self.assertTrue(repository["git_metadata_available"])
        self.assertTrue(repository["tracked_worktree_clean"])
        revision = "33d5c46fd9ee279ecff12321b69597eb406a0fbe"
        self.assertEqual(repository["git_revision"], revision)
        implementation_files = receipt["evaluator_binding"][
            "implementation_files_sha256"
        ]
        self.assertEqual(
            set(implementation_files),
            {
                "benchmark-specs/drivaerml/submission-spec.json",
                "examples/drivaerml-candidate-native-chunks/real_reference_driver.py",
                "reference/drivaerml/diagnostic_evaluator.py",
                "reference/drivaerml/evaluator.py",
                "requirements-drivaerml-evaluator.txt",
                "scripts/evaluate_drivaerml_candidate_case.py",
                "scripts/evaluate_drivaerml_candidate_diagnostics.py",
            },
        )
        for file_name, digest in implementation_files.items():
            self.assertEqual(_sha256_git_blob(revision, file_name), digest)
        self.assertEqual(
            receipt["runtime"],
            {
                "byte_order": "little",
                "numpy": "2.2.6",
                "python": "3.12.13",
                "vtk": "9.5.2",
            },
        )
        cases = {case["case_id"]: case for case in receipt["cases"]}
        self.assertEqual(set(cases), set(EXPECTED_DRIVER_CHILDREN))
        for case_id, expected in EXPECTED_DRIVER_CHILDREN.items():
            case = cases[case_id]
            self.assertEqual(
                case["pinned_volume_part_count"], expected["part_count"]
            )
            self.assertEqual(
                case["transport"], "verified_ordered_multipart_byte_stream"
            )
            self.assertTrue(case["cross_evaluator_prediction_identity_verified"])
            for evidence_kind, file_name in (
                ("core_evidence", "core-evaluation.json"),
                ("diagnostic_evidence", "diagnostic-evaluation.json"),
            ):
                child = case[evidence_kind]
                self.assertEqual(child["file"], f"cases/{case_id}/{file_name}")
                expected_size, expected_sha256 = expected[
                    "core" if evidence_kind == "core_evidence" else "diagnostic"
                ]
                self.assertEqual(child["byte_size"], expected_size)
                self.assertEqual(child["sha256"], expected_sha256)

    @unittest.skipUnless(
        DRIVER_PROVENANCE_PATH.is_file(),
        "reference-driver pilot provenance not packaged yet",
    )
    def test_reference_driver_provenance_binds_fixture_job_and_children(self) -> None:
        provenance = _load(DRIVER_PROVENANCE_PATH)
        self.assertEqual(DRIVER_PROVENANCE_PATH.stat().st_size, 12_227)
        self.assertEqual(
            _sha256(DRIVER_PROVENANCE_PATH), EXPECTED_DRIVER_PROVENANCE_SHA256
        )
        self.assert_path_free(provenance)
        entry = _manifest_entry(DRIVER_PROVENANCE_PATH.name)
        self.assertEqual(entry["sha256"], EXPECTED_DRIVER_PROVENANCE_SHA256)
        self.assertEqual(entry["scope"], "run_1_run_44_non_public_pilot")
        self.assertEqual(
            entry["role"],
            "path_free_code_runtime_launcher_input_fixture_job_and_external_child_evidence_binding_for_reference_driver_pilot",
        )
        self.assertFalse(entry["public_scoring_support_eligible"])
        self.assertEqual(
            provenance["schema"],
            "drivaerml-real-reference-driver-run1-run44-candidate-v7-pilot-provenance-v1",
        )
        self.assertEqual(provenance["schema_version"], 1)
        self.assertEqual(
            provenance["status"], "complete_path_free_non_public_pilot_provenance"
        )
        self.assertEqual(
            provenance["validation_receipt"],
            {
                "file": DRIVER_PATH.name,
                "storage": "byte_identical_copy_of_external_campaign_receipt",
                "campaign": "drivaerml_fluidsbench_pr4_candidate_20260820",
                "external_relative_file": (
                    "real_reference_driver_equal_cell_v11_bound_paths_33d5c46/"
                    "validation-receipt.json"
                ),
                "size_bytes": 2_972,
                "sha256": EXPECTED_DRIVER_SHA256,
                "copied_byte_identically": True,
            },
        )

        receipt = _load(DRIVER_PATH)
        receipt_cases = {case["case_id"]: case for case in receipt["cases"]}
        children = {
            case["case_id"]: case
            for case in provenance["external_child_evidence"]["cases"]
        }
        self.assertEqual(set(children), set(EXPECTED_DRIVER_CHILDREN))
        for case_id, child_case in children.items():
            for evidence_kind in ("core_evidence", "diagnostic_evidence"):
                external_identity = child_case[evidence_kind]
                receipt_identity = receipt_cases[case_id][evidence_kind]
                self.assertEqual(
                    external_identity["file"], Path(receipt_identity["file"]).name
                )
                self.assertEqual(
                    external_identity["size_bytes"], receipt_identity["byte_size"]
                )
                self.assertEqual(
                    external_identity["sha256"], receipt_identity["sha256"]
                )

        implementation = provenance["implementation"]
        revision = "33d5c46fd9ee279ecff12321b69597eb406a0fbe"
        self.assertEqual(implementation["git_revision"], revision)
        self.assertTrue(implementation["tracked_snapshot_clean_before_execution"])
        for source_map_name in (
            "launcher_verified_source_files_sha256",
            "receipt_bound_implementation_files_sha256",
        ):
            for file_name, digest in implementation[source_map_name].items():
                self.assertEqual(_sha256_git_blob(revision, file_name), digest)
        self.assertEqual(
            implementation["receipt_bound_implementation_files_sha256"],
            receipt["evaluator_binding"]["implementation_files_sha256"],
        )
        self.assertEqual(
            implementation["runtime"],
            {
                "python": "3.12.13",
                "numpy": "2.2.6",
                "vtk": "9.5.2",
                "vtk_smp_backend": "Sequential",
                "byte_order": "little",
            },
        )
        self.assertEqual(
            implementation["launcher"],
            {
                "storage": "external_campaign_file_not_bundled_in_git",
                "campaign": "drivaerml_fluidsbench_pr4_candidate_20260820",
                "file": "run_real_reference_driver_equal_cell_v11_selfcontained.sbatch",
                "size_bytes": 2_563,
                "sha256": (
                    "7c877be741b0921aa7a129a4748bedadffa85f3a7706538e46603437018bd471"
                ),
            },
        )

        execution = provenance["execution"]
        self.assertEqual(execution["scheduler_job_id"], "6401365")
        self.assertEqual(execution["account"], "coreai_modulus_cae")
        self.assertEqual((execution["partition"], execution["qos"]), ("cpu", "cpu-normal"))
        self.assertEqual(execution["cpus_per_task"], 1)
        self.assertEqual(execution["requested_memory"], "16G")
        self.assertEqual(execution["time_limit"], "01:00:00")
        self.assertEqual(execution["thread_caps_per_process"], 1)
        self.assertEqual((execution["state"], execution["exit_code"]), ("COMPLETED", "0:0"))
        self.assertEqual(execution["elapsed"], "00:10:21")
        self.assertEqual(execution["python_step_elapsed"], "00:10:08")
        self.assertEqual(execution["python_step_max_rss"], "5756096K")
        self.assertEqual(execution["maximum_prediction_chunk_rows"], 1_000_000)
        self.assertEqual(execution["io_chunk_bytes"], 16_777_216)
        self.assertEqual(execution["volume_weighting"], "one_per_native_cell")
        self.assertFalse(execution["geometric_cell_volume_arrays_used"])
        self.assertTrue(execution["final_validation_passed"])

        inputs = provenance["inputs"]
        self.assertEqual(
            inputs["case_input_config"]["sha256"],
            receipt["case_input_config_sha256"],
        )
        for input_name, receipt_key in (
            ("native_source_pin", "native_source_pin_sha256"),
            ("autocfd5_profile", "autocfd5_profile_sha256"),
        ):
            identity = inputs[input_name]
            path = ROOT / identity["file"]
            self.assertEqual(path.stat().st_size, identity["size_bytes"])
            self.assertEqual(_sha256(path), identity["sha256"])
            self.assertEqual(identity["sha256"], receipt[receipt_key])
        fixture = inputs["prediction_fixture_semantics"]
        self.assertEqual(fixture["schema"], "drivaerml-candidate-dummy-predictions-v1")
        self.assertEqual(
            fixture["prediction_semantics"],
            "deterministic_all_zero_transport_fixture_only",
        )
        self.assertEqual(fixture["maximum_chunk_rows"], 1_000_000)
        for key in (
            "reads_native_truth",
            "model_checkpoint_used",
            "claims_model_quality",
            "official_submission",
        ):
            self.assertFalse(fixture[key])

        expected_fixture_cases = {
            "run_1": {
                "receipt": (
                    11_866,
                    "c9bd450fc55d5857500c8875cd097f05f0b600e9401a8f1079ba73eef9996ca6",
                ),
                "surface": (
                    8_828_095,
                    9,
                    2_131,
                    "d99257b2bcc3081ed8e73a23ff76bbb538c9009af056d06636a0b87c6e22686a",
                ),
                "volume": (
                    147_449_586,
                    148,
                    30_617,
                    "9165ce61425270f2c94f197e4d334df4194ca196a5d772fd234e0248ce09d4cd",
                ),
                "area": (
                    35_312_508,
                    "786c7e55dd4252c8f60a4ef549e4008b1c607a8e5f83d1ab65c8b0b513946f10",
                ),
                "cp": (326_222, "2d63798ecd34ea06b54eaccce82e8bd2e1ce095ed72b0c1f3e861eaa032348a7", 205),
                "velocity": (224_794, EXPECTED_CASES["run_1"]["artifacts"][10], 3_595, 161),
            },
            "run_44": {
                "receipt": (
                    13_075,
                    "985993f768948222df6a9c8f93d62936099af9aad79d06cde4eb3ab8bb994493",
                ),
                "surface": (
                    10_075_774,
                    11,
                    2_538,
                    "1b432858863bcd92ab4fd78894b7c7de4146d806e42bf3e6f40f2d1983572e70",
                ),
                "volume": (
                    163_398_798,
                    164,
                    33_930,
                    "0f53d179bd02ff2de9673bc763fe17c3ed5936b643230c15b0f4e17d53f99be1",
                ),
                "area": (
                    40_303_224,
                    "1018a7ab0a31ec8bc89189303f67eca71ec5ae082c0dab72213976d731f037fd",
                ),
                "cp": (326_509, "d76ead3b210c2bbf542e00872f9ef566deb08d800edc1e7270b14b216ef52e0a", 206),
                "velocity": (221_826, EXPECTED_CASES["run_44"]["artifacts"][10], 3_658, 98),
            },
        }
        input_cases = {case["case_id"]: case for case in inputs["cases"]}
        self.assertEqual(set(input_cases), set(expected_fixture_cases))
        for case_id, expected in expected_fixture_cases.items():
            case = input_cases[case_id]
            self.assertEqual(
                case["pinned_volume_part_count"],
                EXPECTED_DRIVER_CHILDREN[case_id]["part_count"],
            )
            prediction = case["prediction_fixture"]
            self.assertEqual(
                (prediction["receipt"]["size_bytes"], prediction["receipt"]["sha256"]),
                expected["receipt"],
            )
            for support_name, expected_key in (
                ("surface_native_cells", "surface"),
                ("volume_native_cells", "volume"),
            ):
                support = prediction[support_name]
                entity_count, chunk_count, size_bytes, digest = expected[expected_key]
                self.assertEqual(support["entity_count"], entity_count)
                self.assertEqual(support["chunk_count"], chunk_count)
                self.assertEqual(support["manifest_size_bytes"], size_bytes)
                self.assertEqual(support["manifest_sha256"], digest)
                self.assertTrue(support["complete_gap_free_duplicate_free_coverage"])
            area_size, area_sha = expected["area"]
            self.assertEqual(case["fixed_surface_area"]["size_bytes"], area_size)
            self.assertEqual(case["fixed_surface_area"]["sha256"], area_sha)
            self.assertEqual(
                case["fixed_surface_area"]["native_polygon_count"],
                prediction["surface_native_cells"]["entity_count"],
            )
            cp_size, cp_sha, cp_valid = expected["cp"]
            self.assertEqual(case["cp_support"]["size_bytes"], cp_size)
            self.assertEqual(case["cp_support"]["sha256"], cp_sha)
            self.assertEqual(case["cp_support"]["probe_row_count"], 209)
            self.assertEqual(case["cp_support"]["support_valid_count"], cp_valid)
            self.assertFalse(case["cp_support"]["owner_visual_signoff"])
            velocity_size, velocity_sha, velocity_valid, velocity_invalid = expected[
                "velocity"
            ]
            velocity = case["velocity_support"]
            self.assertEqual(velocity["mapping_size_bytes"], velocity_size)
            self.assertEqual(velocity["mapping_sha256"], velocity_sha)
            self.assertEqual(velocity["row_count"], 3_756)
            self.assertEqual(velocity["valid_count"], velocity_valid)
            self.assertEqual(velocity["invalid_count"], velocity_invalid)
            self.assertEqual(velocity_valid + velocity_invalid, 3_756)

        scope = provenance["scientific_scope"]
        self.assertTrue(scope["two_case_transport_and_reduction_fixture_only"])
        self.assertTrue(scope["actual_pinned_native_surface_and_volume_inputs_used"])
        for key in (
            "complete_484_case_split_evaluated",
            "trained_model_checkpoint_used",
            "model_inference_result",
            "model_quality_claim",
            "physics_null_baseline",
            "physics_null_denominator",
            "three_real_model_sensitivity",
            "independent_participant_dry_run",
            "frozen_evaluator_release",
            "public_scoring_support_eligible",
            "official_submission",
            "scoring_contract_active",
            "owner_scientific_approval",
            "fabricated_scientific_results",
            "downloads_performed",
            "geometric_cell_volume_weights_used",
        ):
            self.assertFalse(scope[key])

    def test_submission_remains_closed_and_evaluator_unfrozen(self) -> None:
        specification = _load(SPEC_PATH)
        self.assertEqual(specification["status"], "candidate_scoring_contract")
        support = specification["scoring_support"]
        self.assertEqual(support["status"], "owner_review_required")
        self.assertFalse(support["submissions_open"])
        binding = support["dataset_evaluator_binding"]
        self.assertEqual(binding["status"], "pending_frozen_release")
        self.assertIsNone(binding["evaluator_code_revision"])
        self.assertEqual(
            support["activation_gates"]["velocity_profiles"],
            "all_484_fixed_and_relative_native_truth_and_support_published_owner_scientifically_approved_release_binding_pending",
        )
        self.assertEqual(
            support["activation_gates"]["independent_participant_dry_run"],
            "pending",
        )
        self.assertEqual(
            support["activation_gates"]["owner_evaluator_approval"],
            "scientific_method_approved_final_immutable_release_binding_pending",
        )


if __name__ == "__main__":
    unittest.main()
