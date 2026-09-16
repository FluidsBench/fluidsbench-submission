from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.aggregate_drivaerml_force_replay import (
    CHUNK_INVARIANCE_ABSOLUTE_TOLERANCE,
    COEFFICIENT_ABSOLUTE_TOLERANCE,
    ForceAggregateError,
    aggregate_force_replay,
    sha256_file,
    write_evidence,
)


CASE_IDS = ("run_1", "run_44")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_truth(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            case_id = f"run_{int(row['run'])}"
            result[case_id] = {
                "Cd": float(row["cd"]),
                "Cl": float(row["cl"]),
                "Clf": float(row["clf"]),
                "Clr": float(row["clr"]),
                "Cs": float(row["cs"]),
            }
            result[case_id]["CmPitch"] = (
                result[case_id]["Clf"] - result[case_id]["Clr"]
            ) / 2.0
    return result


class DrivAerMLForceAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.truth_path = self.root / "force_mom_constref_all.csv"
        self.truth_path.write_text(
            "run,cd,cl,clf,clr,cs\n"
            "1,0.31,0.07,-0.045,0.115,0.04\n"
            "44,0.30,0.06,-0.03,0.09,0.035\n",
            encoding="utf-8",
        )
        self.truth = _read_truth(self.truth_path)
        self.pin = self._make_pin()
        self.pin_path = self.root / "native-source-pin.json"
        _write_json(self.pin_path, self.pin)
        self.receipt_paths = []
        for position, case_id in enumerate(CASE_IDS):
            receipt_path = self.root / f"{case_id}-receipt.json"
            _write_json(
                receipt_path,
                self._make_receipt(case_id, position),
            )
            self.receipt_paths.append(receipt_path)

    def _make_pin(self) -> dict[str, object]:
        cases = []
        for position, case_id in enumerate(CASE_IDS):
            run_number = int(case_id.removeprefix("run_"))
            entity_count = 10 + position
            boundary_sha = str(position + 1) * 64
            area_sha = chr(ord("a") + position) * 64
            cases.append(
                {
                    "case_id": case_id,
                    "run_number": run_number,
                    "boundary": {
                        "path": f"{case_id}/boundary_{run_number}.vtp",
                        "lfs_sha256": boundary_sha,
                        "size_bytes": 1000 + position,
                    },
                    "surface_cell_area": {
                        "path": f"{case_id}/boundary_cell_area_{run_number}.npy",
                        "lfs_sha256": area_sha,
                        "source_boundary_sha256": boundary_sha,
                        "dtype": "<f4",
                        "element_count": entity_count,
                        "size_bytes": 128 + 4 * entity_count,
                    },
                }
            )
        return {
            "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
            "schema_version": 1,
            "repository": {
                "provider": "Hugging Face Hub",
                "repo_id": "neashton/drivaerml",
                "repo_type": "dataset",
                "revision": "7a5c0948ce27be709b1116a3a190f806e7a8f79f",
            },
            "case_scope": {"case_count": len(CASE_IDS)},
            "authoritative_support_files": {
                "force_mom_constref_all": {
                    "path": "force_mom_constref_all.csv",
                    "sha256": sha256_file(self.truth_path),
                    "size_bytes": self.truth_path.stat().st_size,
                }
            },
            "cases": cases,
        }

    def _make_receipt(self, case_id: str, position: int) -> dict[str, object]:
        truth = self.truth[case_id]
        reconstructed_cl = truth["Cl"] + (position + 1) * 2.0e-8
        reconstructed_pitch = truth["CmPitch"] - (position + 1) * 1.0e-8
        coefficients = {
            "Cd": truth["Cd"] + (position + 1) * 1.0e-8,
            "Cl": reconstructed_cl,
            "CmPitch": reconstructed_pitch,
            "Clf": reconstructed_cl / 2.0 + reconstructed_pitch,
            "Clr": reconstructed_cl / 2.0 - reconstructed_pitch,
            "Cs": truth["Cs"] - (position + 1) * 1.0e-8,
        }
        differences = {
            key: abs(coefficients[key] - truth[key])
            for key in ("Cd", "Cl", "CmPitch", "Clf", "Clr", "Cs")
        }
        lift_closure = abs(
            coefficients["Cl"] - coefficients["Clf"] - coefficients["Clr"]
        )
        run_number = int(case_id.removeprefix("run_"))
        pinned = self.pin["cases"][position]  # type: ignore[index]
        entity_count = pinned["surface_cell_area"]["element_count"]  # type: ignore[index]
        chunk_differences = {
            key: (position + 1) * (index + 1) * 1.0e-15
            for index, key in enumerate(("Cd", "Cl", "CmPitch", "Clf", "Clr", "Cs"))
        }
        return {
            "schema": "drivaerml-native-surface-force-case-audit-v1",
            "status": "passed_candidate_evaluator_case_audit",
            "case_id": case_id,
            "source": {
                "boundary": {
                    "path": f"/private/native-cache/{case_id}/boundary_{run_number}.vtp",
                    "sha256": pinned["boundary"]["lfs_sha256"],  # type: ignore[index]
                },
                "surface_areas": {
                    "path": (
                        f"/private/area-cache/{case_id}/"
                        f"boundary_cell_area_{run_number}.npy"
                    ),
                    "sha256": pinned["surface_cell_area"]["lfs_sha256"],  # type: ignore[index]
                    "role": "fixed_input_not_regenerated",
                },
                "truth": {
                    "path": (
                        f"/private/native-cache/{case_id}/"
                        f"force_mom_constref_{run_number}.csv"
                    ),
                    "sha256": str(8 + position) * 64,
                },
            },
            "mesh_and_fields": {
                "vtk_version": "9.5.2",
                "point_count": entity_count + 3,
                "polygon_count": entity_count,
                "available_point_arrays": [],
                "available_cell_arrays": [
                    "CpMeanTrim",
                    "pMeanTrim",
                    "wallShearStressMeanTrim",
                ],
                "pressure_dtype": "float32",
                "wall_shear_dtype": "float32",
            },
            "raw_cell_order": "unchanged zero-based VTK CellData tuple order",
            "units": {
                "coordinates": "m",
                "surface_areas": "m^2",
                "pMeanTrim": "m^2/s^2",
                "wallShearStressMeanTrim": "m^2/s^2",
            },
            "area_audit": {
                "entity_count": entity_count,
                "calculated_sum_m2": 10.0 + position,
                "published_sum_m2": 10.0 + position + 1.0e-8,
                "maximum_absolute_difference_m2": (position + 1) * 1.0e-12,
                "maximum_relative_difference": (position + 1) * 2.0e-8,
            },
            "coefficients": {
                **coefficients,
                "entity_count": entity_count,
                "force_n": [1.0, 2.0, 3.0],
                "lift_closure_abs": lift_closure,
                "moment_about_forces_cor_n_m": [4.0, 5.0, 6.0],
            },
            "truth": {
                "cd": truth["Cd"],
                "cl": truth["Cl"],
                "clf": truth["Clf"],
                "clr": truth["Clr"],
                "cmpitch": truth["CmPitch"],
                "cs": truth["Cs"],
            },
            "absolute_truth_difference": differences,
            "coefficient_absolute_tolerance": COEFFICIENT_ABSOLUTE_TOLERANCE,
            "chunk_invariance": {
                "chunk_polygons": [7, 4],
                "absolute_difference": chunk_differences,
                "maximum_absolute_difference": max(chunk_differences.values()),
                "tolerance": CHUNK_INVARIANCE_ABSOLUTE_TOLERANCE,
            },
            "runtime": {
                "python": "3.12.13",
                "numpy": "2.2.6",
                "vtk": "9.5.2",
                "elapsed_seconds": 10.0 + position,
            },
        }

    def _aggregate(self, receipt_paths: list[Path] | None = None) -> dict[str, object]:
        return aggregate_force_replay(
            native_source_pin_path=self.pin_path,
            authoritative_truth_path=self.truth_path,
            receipt_paths=self.receipt_paths if receipt_paths is None else receipt_paths,
            expected_case_ids=CASE_IDS,
            expected_pin_sha256=None,
        )

    def _rewrite_receipt(self, position: int, receipt: dict[str, object]) -> None:
        _write_json(self.receipt_paths[position], receipt)

    def _receipt(self, position: int) -> dict[str, object]:
        return json.loads(self.receipt_paths[position].read_text(encoding="utf-8"))

    def _repin_truth(self) -> None:
        pin = json.loads(self.pin_path.read_text(encoding="utf-8"))
        binding = pin["authoritative_support_files"]["force_mom_constref_all"]
        binding["sha256"] = sha256_file(self.truth_path)
        binding["size_bytes"] = self.truth_path.stat().st_size
        _write_json(self.pin_path, pin)

    def test_complete_receipts_emit_deterministic_compact_path_free_evidence(self) -> None:
        evidence = self._aggregate(list(reversed(self.receipt_paths)))
        self.assertEqual(evidence["case_count"], 2)
        self.assertEqual(evidence["status"], "passed_requested_case_set_force_replay")
        self.assertEqual(
            [row["case_id"] for row in evidence["case_receipts"]],
            list(CASE_IDS),
        )
        self.assertEqual(
            evidence["mirror_truth_validation"]["status"],
            "exact_match_to_authoritative_aggregate",
        )
        self.assertEqual(evidence["surface_area_audit"]["entity_count"], 21)
        self.assertEqual(
            evidence["coefficient_absolute_difference"]["Cd"][
                "case_id_at_maximum"
            ],
            "run_44",
        )
        self.assertEqual(
            evidence["chunk_invariance"]["chunk_polygon_pairs"], [[7, 4]]
        )
        serialized = json.dumps(evidence, sort_keys=True)
        self.assertNotIn("/private/", serialized)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("elapsed_seconds", serialized)

        first = self.root / "evidence-first.json"
        second = self.root / "evidence-second.json"
        write_evidence(first, evidence)
        write_evidence(second, self._aggregate())
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_missing_duplicate_and_unexpected_receipts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ForceAggregateError, "missing=.*run_44"):
            self._aggregate([self.receipt_paths[0]])
        with self.assertRaisesRegex(ForceAggregateError, "duplicate force receipt"):
            self._aggregate([self.receipt_paths[0], self.receipt_paths[0]])

        receipt = self._receipt(1)
        receipt["case_id"] = "run_999"
        self._rewrite_receipt(1, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "unexpected force receipt"):
            self._aggregate()

    def test_mirror_truth_must_match_authoritative_row_exactly(self) -> None:
        receipt = self._receipt(0)
        receipt["truth"]["cd"] += 1.0e-8
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(
            ForceAggregateError, "mirror truth Cd differs from authoritative"
        ):
            self._aggregate()

    def test_coefficient_and_declared_tolerances_are_recomputed(self) -> None:
        receipt = self._receipt(0)
        receipt["coefficient_absolute_tolerance"] = 2.0e-6
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "tolerance is not frozen"):
            self._aggregate()

        receipt = self._make_receipt("run_1", 0)
        receipt["coefficients"]["Cd"] = self.truth["run_1"]["Cd"] + 2.0e-6
        receipt["absolute_truth_difference"]["Cd"] = 2.0e-6
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "Cd difference exceeds"):
            self._aggregate()

    def test_chunk_and_area_audit_tolerance_failures_are_hard_errors(self) -> None:
        receipt = self._receipt(0)
        receipt["chunk_invariance"]["absolute_difference"]["Cl"] = 3.0e-12
        receipt["chunk_invariance"]["maximum_absolute_difference"] = 3.0e-12
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "chunk invariance exceeds"):
            self._aggregate()

        receipt = self._make_receipt("run_1", 0)
        receipt["area_audit"]["maximum_relative_difference"] = 6.1e-8
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "surface-area audit exceeds"):
            self._aggregate()

    def test_source_hashes_counts_and_raw_order_are_bound_to_pin(self) -> None:
        receipt = self._receipt(0)
        receipt["source"]["boundary"]["sha256"] = "f" * 64
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "boundary SHA-256 mismatch"):
            self._aggregate()

        receipt = self._make_receipt("run_1", 0)
        receipt["mesh_and_fields"]["polygon_count"] += 1
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "polygon count differs"):
            self._aggregate()

        receipt = self._make_receipt("run_1", 0)
        receipt["raw_cell_order"] = "remeshed"
        self._rewrite_receipt(0, receipt)
        with self.assertRaisesRegex(ForceAggregateError, "raw native cell order"):
            self._aggregate()

    def test_truth_schema_duplicate_cases_and_pin_hash_are_strict(self) -> None:
        self.truth_path.write_text(
            "run,cd,cl,clf,clr,cs\n"
            "1,0.31,0.07,-0.045,0.115,0.04\n"
            "1,0.30,0.06,-0.03,0.09,0.035\n",
            encoding="utf-8",
        )
        self._repin_truth()
        with self.assertRaisesRegex(ForceAggregateError, "duplicate authoritative"):
            self._aggregate()

        self.truth_path.write_text(
            "run,cd,cl,clf,clr,cs,extra\n"
            "1,0.31,0.07,-0.045,0.115,0.04,x\n"
            "44,0.30,0.06,-0.03,0.09,0.035,x\n",
            encoding="utf-8",
        )
        self._repin_truth()
        with self.assertRaisesRegex(ForceAggregateError, "header must be exactly"):
            self._aggregate()

        # The production defaults never accept this two-case synthetic pin.
        with self.assertRaisesRegex(ForceAggregateError, "native source pin SHA-256"):
            aggregate_force_replay(
                native_source_pin_path=self.pin_path,
                authoritative_truth_path=self.truth_path,
                receipt_paths=self.receipt_paths,
            )


if __name__ == "__main__":
    unittest.main()
