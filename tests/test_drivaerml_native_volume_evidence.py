from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark-specs" / "drivaerml"
EVIDENCE = BENCHMARK / "evidence" / "native-volume-equal-cell-primary-all484.json"
EXPECTED_SHA256 = "bda42a125ffb4d6484756e77ac7e495974f39d4bce3e663154322e9e560827c7"
PROVENANCE = (
    BENCHMARK
    / "evidence"
    / "native-volume-equal-cell-primary-all484-provenance.json"
)
EXPECTED_PROVENANCE_SHA256 = (
    "b5ffe2234bb1597cf041ff5d97458f3d0d6e81db7a30591a7e26f51ebc032fde"
)
FAILURE_EVIDENCE = (
    BENCHMARK / "evidence" / "volume-weight-vtk-run_1-failure-diagnostic.json"
)
EXPECTED_FAILURE_SHA256 = (
    "aa2a209cafbd598930bfbe2dd1a73e8c06108188aff69c30841bfc46bfe7927e"
)
VTK96_PROBE_EVIDENCE = (
    BENCHMARK / "evidence" / "volume-weight-vtk96-run_1-wedge-probe.json"
)
EXPECTED_VTK96_PROBE_SHA256 = (
    "2970c507038bc1c3978542cc8e07c6682230db496f646a4887467645304a58a6"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_git_blob(revision: str, path: str) -> str:
    payload = subprocess.run(
        ("git", "show", f"{revision}:{path}"),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


class DrivAerNativeVolumeEvidenceTests(unittest.TestCase):
    def test_all_case_equal_cell_evidence_is_bound_and_nonactivating(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        manifest = json.loads(
            (BENCHMARK / "evidence" / "manifest.json").read_text(encoding="utf-8")
        )
        specification = json.loads(
            (BENCHMARK / "submission-spec.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sha256_file(EVIDENCE), EXPECTED_SHA256)
        self.assertEqual(
            evidence["schema"],
            "drivaerml-native-volume-equal-cell-primary-all-case-audit-v1",
        )
        self.assertEqual(evidence["status"], "passed_all_case_equal_cell_primary_audit")
        self.assertEqual(evidence["activation_status"], "does_not_activate_scoring_contract")
        self.assertFalse(evidence["public_scoring_support_eligible"])
        self.assertFalse(evidence["fixture_semantics"]["physics_null_baseline"])
        self.assertFalse(
            evidence["fixture_semantics"]["physical_volume_secondary_exercised"]
        )

        totals = evidence["totals"]
        self.assertEqual(totals["case_count"], 484)
        self.assertEqual(totals["two_part_case_count"], 474)
        self.assertEqual(totals["three_part_case_count"], 10)
        self.assertEqual(totals["verified_segment_count"], 978)
        self.assertEqual(totals["logical_volume_size_bytes"], 22_932_775_011_362)
        self.assertEqual(totals["native_cell_count"], 68_949_662_110)
        self.assertLessEqual(totals["maximum_additive_relative_difference"], 2e-12)
        self.assertLessEqual(totals["maximum_metric_absolute_difference"], 2e-12)

        cases = evidence["cases"]
        self.assertEqual(len(cases), 484)
        self.assertEqual(len({case["case_id"] for case in cases}), 484)
        self.assertEqual(
            sum(case["vtk"]["piece"]["number_of_cells"] for case in cases),
            totals["native_cell_count"],
        )
        for case in cases:
            cell_count = case["vtk"]["piece"]["number_of_cells"]
            self.assertEqual(case["volume_weights"]["cell_count"], cell_count)
            self.assertEqual(
                case["volume_weights"]["status"],
                "equal_cell_primary_only_physical_secondary_not_exercised",
            )
            for field_name, components, units in (
                ("pMeanTrim", 1, "m^2/s^2"),
                ("UMeanTrim", 3, "m/s"),
            ):
                field = case["required_cell_data"][field_name]
                self.assertEqual(field["association"], "CellData")
                self.assertEqual(field["tuple_count"], cell_count)
                self.assertEqual(field["number_of_components"], components)
                self.assertEqual(field["units"], units)
                self.assertTrue(field["finite"])
                self.assertEqual(field["raw_id_start"], 0)
                self.assertEqual(field["raw_id_stop"], cell_count)

        entry = next(
            item
            for item in manifest["artifacts"]
            if item["file"] == EVIDENCE.name
        )
        self.assertEqual(entry["sha256"], EXPECTED_SHA256)
        self.assertEqual(entry["scope"], "all_484_cases")
        self.assertEqual(
            entry["role"],
            "multipart_native_volume_fields_raw_order_coverage_and_equal_cell_primary_invariance_audit",
        )
        self.assertFalse(entry["public_scoring_support_eligible"])

        support = next(
            item
            for item in specification["scoring_support"]["public_supports"]
            if item["id"] == "volume_native_cells"
        )
        binding = support["candidate_primary_validation_evidence"]
        self.assertEqual(binding["file"], f"evidence/{EVIDENCE.name}")
        self.assertEqual(binding["sha256"], EXPECTED_SHA256)
        self.assertEqual(
            binding["provenance_file"], f"evidence/{PROVENANCE.name}"
        )
        self.assertEqual(
            binding["provenance_sha256"], EXPECTED_PROVENANCE_SHA256
        )
        self.assertTrue(binding["complete_all_484_cases"])
        self.assertTrue(binding["equal_native_cell_weighting_exercised"])
        self.assertTrue(binding["owner_scientific_approval"])
        self.assertEqual(support["weighting"], "one_per_native_cell")
        self.assertFalse(support["geometric_cell_volume_weights_required"])
        self.assertNotIn("candidate_secondary_weight_status", support)

    def test_all_case_provenance_binds_code_runtime_and_receipts(self) -> None:
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        manifest = json.loads(
            (BENCHMARK / "evidence" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sha256_file(PROVENANCE), EXPECTED_PROVENANCE_SHA256)
        self.assertEqual(provenance["aggregate"]["sha256"], EXPECTED_SHA256)
        self.assertTrue(
            provenance["aggregate"][
                "strict_reaggregation_reproduced_byte_identically"
            ]
        )
        self.assertEqual(provenance["case_receipts"]["case_count"], 484)
        self.assertEqual(
            provenance["case_receipts"]["total_size_bytes"], 10_717_655
        )
        self.assertEqual(
            provenance["implementation"]["git_revision"],
            "7067ba927195c9983a06c2c69264fdcf0c1d4f59",
        )
        runtime = provenance["implementation"]["runtime"]
        self.assertEqual(runtime["python"], "3.12.13")
        self.assertEqual(runtime["numpy"], "2.2.6")
        self.assertFalse(runtime["vtk_used_by_this_audit"])
        revision = provenance["implementation"]["git_revision"]
        for source in provenance["implementation"]["case_generator_snapshot"][
            "files"
        ]:
            self.assertEqual(
                sha256_git_blob(revision, source["file"]), source["sha256"]
            )
        aggregator = provenance["implementation"]["strict_aggregator"]
        self.assertEqual(
            sha256_git_blob(revision, aggregator["file"]), aggregator["sha256"]
        )
        self.assertTrue(
            provenance["execution"][
                "all_484_final_task_states_completed_exit_zero"
            ]
        )
        self.assertFalse(
            provenance["scientific_scope"][
                "physical_volume_secondary_weights_exercised"
            ]
        )
        self.assertFalse(
            provenance["scientific_scope"]["owner_scientific_approval"]
        )
        entry = next(
            item
            for item in manifest["artifacts"]
            if item["file"] == PROVENANCE.name
        )
        self.assertEqual(entry["sha256"], EXPECTED_PROVENANCE_SHA256)
        self.assertFalse(entry["public_scoring_support_eligible"])

    def test_historical_vtk_weight_artifacts_are_superseded_without_contract_role(
        self,
    ) -> None:
        evidence = json.loads(FAILURE_EVIDENCE.read_text(encoding="utf-8"))
        manifest = json.loads(
            (BENCHMARK / "evidence" / "manifest.json").read_text(encoding="utf-8")
        )
        specification = json.loads(
            (BENCHMARK / "submission-spec.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sha256_file(FAILURE_EVIDENCE), EXPECTED_FAILURE_SHA256)
        self.assertEqual(evidence["status"], "candidate_algorithm_rejected_fail_closed")
        self.assertFalse(evidence["published_outputs"]["volume_weight_array"])
        self.assertFalse(evidence["published_outputs"]["success_receipt"])
        self.assertEqual(
            evidence["implementation_binding"]["source_snapshot"]["file"],
            "code_weights_detailed/reference/drivaerml/volume_weights.py",
        )
        self.assertEqual(
            evidence["implementation_binding"]["source_reader_snapshot"]["file"],
            "code_weights_detailed/reference/drivaerml/source.py",
        )
        self.assertEqual(
            evidence["external_execution_logs"]["stderr"]["file"],
            "logs/volume_weights_detailed_normal_6360522_0.err",
        )
        self.assertEqual(
            evidence["external_execution_logs"]["stdout"]["file"],
            "logs/volume_weights_detailed_normal_6360522_0.out",
        )
        failed = evidence["checks"]["failed"]
        self.assertEqual(failed["negative_count"], 1)
        self.assertEqual(failed["first_invalid"][0]["raw_cell_id"], 124_707_859)
        self.assertEqual(failed["first_invalid"][0]["vtk_cell_type_name"], "vtkWedge")
        self.assertFalse(evidence["public_scoring_support_eligible"])
        self.assertFalse(evidence["official_submission_scoring_enabled"])
        self.assertFalse(evidence["owner_scientific_approval"])

        entry = next(
            item
            for item in manifest["artifacts"]
            if item["file"] == FAILURE_EVIDENCE.name
        )
        self.assertEqual(entry["sha256"], EXPECTED_FAILURE_SHA256)
        self.assertEqual(
            entry["role"],
            "superseded_historical_geometric_volume_weight_rejection_diagnostic_no_contract_role",
        )
        self.assertFalse(entry["public_scoring_support_eligible"])

        vtk96_entry = next(
            item
            for item in manifest["artifacts"]
            if item["file"] == VTK96_PROBE_EVIDENCE.name
        )
        self.assertEqual(
            sha256_file(VTK96_PROBE_EVIDENCE), EXPECTED_VTK96_PROBE_SHA256
        )
        self.assertEqual(vtk96_entry["sha256"], EXPECTED_VTK96_PROBE_SHA256)
        self.assertEqual(
            vtk96_entry["role"],
            "superseded_historical_vtk96_exact_wedge_probe_no_contract_role",
        )
        self.assertFalse(vtk96_entry["public_scoring_support_eligible"])

        support = next(
            item
            for item in specification["scoring_support"]["public_supports"]
            if item["id"] == "volume_native_cells"
        )
        self.assertEqual(support["weighting"], "one_per_native_cell")
        self.assertFalse(support["geometric_cell_volume_weights_required"])
        self.assertNotIn("candidate_secondary_weight_status", support)
        self.assertNotIn(FAILURE_EVIDENCE.name, json.dumps(support, sort_keys=True))
        self.assertNotIn(VTK96_PROBE_EVIDENCE.name, json.dumps(support, sort_keys=True))
        self.assertFalse(
            any(
                "volume" in decision and "weight" in decision
                for decision in specification["scoring_support"][
                    "owner_decisions_required"
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
