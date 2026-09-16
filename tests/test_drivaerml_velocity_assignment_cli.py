from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.velocity_assignments import (
    NO_CLOSURE_CELL_REASON,
    QUERY_CACHE_KEY_ID,
    REQUIRED_VTK_VERSION,
    VelocityAssignmentError,
    vtk_available,
)
from scripts.generate_drivaerml_velocity_assignments import (
    ARTIFACT_SCHEMA,
    EXPECTED_AUTOCFD5_PROFILE_SHA256,
    EXPECTED_DATASET_REPOSITORY_ID,
    EXPECTED_DATASET_REVISION,
    EXPECTED_SAMPLE_COUNTS,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    VelocityAssignmentCLIError,
    _profile_binding,
    generate_pinned_velocity_assignments,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_drivaerml_velocity_assignments.py"
PROFILE = ROOT / "benchmark-specs" / "drivaerml" / "drivaerml-diagnostics-v9.json"
VTK_READY = vtk_available()

if VTK_READY:
    import vtk


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


class DrivAerMLVelocityAssignmentCLIBaseTests(unittest.TestCase):
    def test_profile_binding_is_probe_free_v9(self) -> None:
        definition, binding = _profile_binding(PROFILE)
        self.assertEqual(
            set(binding),
            {
                "profile_sha256",
                "source_registry_sha256",
                "line_count",
                "fixed_10mm_sample_count",
                "continuous_cp_cut_count",
                "discrete_cp_probe_count",
            },
        )
        self.assertEqual(
            set(binding["source_registry_sha256"]),
            {"velocity_lines", "velocity_scoring_grid"},
        )
        self.assertEqual(binding["profile_sha256"], EXPECTED_AUTOCFD5_PROFILE_SHA256)
        self.assertEqual(binding["line_count"], 16)
        self.assertEqual(binding["fixed_10mm_sample_count"], 3756)
        self.assertEqual(binding["continuous_cp_cut_count"], 4)
        self.assertEqual(binding["discrete_cp_probe_count"], 0)
        self.assertEqual(len(definition.pressure_cut_ids), 4)

    def test_help_is_strict_candidate_only(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        help_text = " ".join(completed.stdout.split())
        self.assertIn("closed-candidate", help_text)
        self.assertIn("retained-descriptor VTK read", help_text)
        self.assertIn("all 16 lines at 1, 2, 5, and 10 mm", help_text)
        self.assertIn("does not prove", help_text)
        for option in (
            "--case-id",
            "--native-source-pin",
            "--dataset-root",
            "--monolithic-vtu",
            "--output-root",
            "--autocfd5-profile",
        ):
            self.assertIn(option, completed.stdout)

    def test_profile_is_pinned_and_existing_output_is_preserved(self) -> None:
        self.assertEqual(_sha256_file(PROFILE), EXPECTED_AUTOCFD5_PROFILE_SHA256)
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "unrelated.txt"
            marker.write_text("preserve me", encoding="utf-8")
            with self.assertRaisesRegex(
                VelocityAssignmentCLIError, "output root already exists"
            ):
                generate_pinned_velocity_assignments(
                    case_id="run_44",
                    native_source_pin=root / "missing-pin.json",
                    dataset_root=root,
                    monolithic_vtu=root / "missing.vtu",
                    output_root=existing,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

            tampered_profile = root / "drivaerml-diagnostics-v9.json"
            tampered_profile.write_bytes(PROFILE.read_bytes() + b"\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-id",
                    "run_44",
                    "--native-source-pin",
                    str(root / "missing-pin.json"),
                    "--dataset-root",
                    str(root),
                    "--monolithic-vtu",
                    str(root / "missing.vtu"),
                    "--output-root",
                    str(root / "rejected"),
                    "--autocfd5-profile",
                    str(tampered_profile),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exact probe-free v9 SHA-256", completed.stderr)
            self.assertFalse((root / "rejected").exists())

    def test_runtime_mismatch_fails_before_native_source_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            output = root / "must-not-exist"
            with mock.patch(
                "scripts.generate_drivaerml_velocity_assignments._profile_binding",
                return_value=(object(), {}),
            ) as profile_binding, mock.patch(
                "reference.drivaerml.velocity_assignments.candidate_kernel_receipt",
                side_effect=VelocityAssignmentError(
                    "DrivAerML velocity assignment generation requires NumPy "
                    "2.2.6, got 2.2.5"
                ),
            ) as runtime_receipt, mock.patch(
                "scripts.generate_drivaerml_velocity_assignments._native_source_binding"
            ) as native_binding:
                with self.assertRaisesRegex(
                    VelocityAssignmentCLIError,
                    "runtime preflight failed.*NumPy 2.2.6, got 2.2.5",
                ):
                    generate_pinned_velocity_assignments(
                        case_id="run_44",
                        native_source_pin=root / "native-source-pin.json",
                        dataset_root=root,
                        monolithic_vtu=root / "volume_44.vtu",
                        output_root=output,
                        autocfd5_profile=root / "drivaerml-diagnostics-v9.json",
                    )
            profile_binding.assert_called_once()
            runtime_receipt.assert_called_once_with()
            native_binding.assert_not_called()
            self.assertFalse(output.exists())


@unittest.skipUnless(VTK_READY, f"optional VTK {REQUIRED_VTK_VERSION} is unavailable")
class DrivAerMLVelocityAssignmentCLIVTKTests(unittest.TestCase):
    @staticmethod
    def _write_small_hex(path: Path) -> None:
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for point in (
            (-0.25, -0.25, -0.25),
            (0.25, -0.25, -0.25),
            (0.25, 0.25, -0.25),
            (-0.25, 0.25, -0.25),
            (-0.25, -0.25, 0.25),
            (0.25, -0.25, 0.25),
            (0.25, 0.25, 0.25),
            (-0.25, 0.25, 0.25),
        ):
            points.InsertNextPoint(*point)
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, list(range(8)))

        point_values = vtk.vtkFloatArray()
        point_values.SetName("UMeanTrim")
        point_values.SetNumberOfComponents(3)
        point_values.SetNumberOfTuples(grid.GetNumberOfPoints())
        point_values.Fill(3.0)
        grid.GetPointData().AddArray(point_values)
        cell_values = vtk.vtkFloatArray()
        cell_values.SetName("pMeanTrim")
        cell_values.SetNumberOfTuples(grid.GetNumberOfCells())
        cell_values.Fill(7.0)
        grid.GetCellData().AddArray(cell_values)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(grid)
        writer.SetDataModeToAscii()
        writer.SetCompressorTypeToNone()
        writer.SetHeaderTypeToUInt64()
        if writer.Write() != 1:
            raise AssertionError("synthetic VTU writer failed")

    @classmethod
    def _fixture(cls, root: Path) -> tuple[Path, Path, list[bytes]]:
        monolithic = root / "synthetic-native-volume.vtu"
        cls._write_small_hex(monolithic)
        logical = monolithic.read_bytes()
        first = len(logical) // 3
        second = 2 * len(logical) // 3
        parts = [logical[:first], logical[first:second], logical[second:]]
        boundary = b"synthetic-boundary"
        boundary_sha = _sha256_bytes(boundary)
        case_id = "run_44"
        part_rows = [
            {
                "part_index": index,
                "path": f"{case_id}/volume_44.vtu.{index:02d}.part",
                "size_bytes": len(payload),
                "lfs_sha256": _sha256_bytes(payload),
            }
            for index, payload in enumerate(parts)
        ]
        pin = {
            "schema": "drivaerml-fluidsbench-public-native-source-pin-v1",
            "schema_version": 1,
            "repository": {
                "repo_id": EXPECTED_DATASET_REPOSITORY_ID,
                "revision": EXPECTED_DATASET_REVISION,
            },
            "case_scope": {
                "case_count": 1,
                "run_number_min": 44,
                "run_number_max": 44,
                "unavailable_or_held_back_run_numbers": [],
            },
            "cases": [
                {
                    "case_id": case_id,
                    "run_number": 44,
                    "boundary": {
                        "path": f"{case_id}/boundary_44.vtp",
                        "size_bytes": len(boundary),
                        "lfs_sha256": boundary_sha,
                    },
                    "surface_cell_area": {
                        "path": f"{case_id}/boundary_cell_area_44.npy",
                        "size_bytes": 132,
                        "lfs_sha256": _sha256_bytes(b"synthetic-area"),
                        "dtype": "<f4",
                        "element_count": 1,
                        "source_boundary_sha256": boundary_sha,
                    },
                    "volume": {
                        "logical_path_after_assembly": f"{case_id}/volume_44.vtu",
                        "part_count": len(parts),
                        "parts": part_rows,
                        "total_size_bytes": len(logical),
                    },
                }
            ],
            "totals": {
                "boundary_file_count": 1,
                "boundary_bytes": len(boundary),
                "surface_cell_area_file_count": 1,
                "surface_cell_area_bytes": 132,
                "logical_volume_count": 1,
                "volume_part_file_count": len(parts),
                "reconstructed_volume_bytes": len(logical),
            },
        }
        pin_path = root / "synthetic-native-source-pin.json"
        pin_path.write_text(json.dumps(pin), encoding="utf-8")
        return pin_path, monolithic, parts

    @staticmethod
    def _run(
        root: Path,
        pin: Path,
        monolithic: Path,
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--case-id",
                "run_44",
                "--native-source-pin",
                str(pin),
                "--dataset-root",
                str(root),
                "--monolithic-vtu",
                str(monolithic),
                "--output-root",
                str(output),
                "--io-chunk-bytes",
                "31",
                "--validation-chunk-cells",
                "1",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_emits_deterministic_complete_path_free_resolution_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            pin, monolithic, parts = self._fixture(root)
            first = root / "first-output"
            second = root / "second-output"
            first_run = self._run(root, pin, monolithic, first)
            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            second_run = self._run(root, pin, monolithic, second)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            first_receipt = json.loads(first_run.stdout)
            second_receipt = json.loads(second_run.stdout)
            self.assertEqual(first_receipt, second_receipt)
            self.assertEqual(_relative_hashes(first), _relative_hashes(second))
            self.assertEqual(first_receipt["schema"], RECEIPT_SCHEMA)
            self.assertEqual(first_receipt["status"], RECEIPT_STATUS)
            self.assertEqual(first_receipt["case_id"], "run_44")
            self.assertEqual(
                first_receipt["registries"]["profile_sha256"],
                EXPECTED_AUTOCFD5_PROFILE_SHA256,
            )
            self.assertEqual(
                first_receipt["native_source_binding"]["ordered_verified_segments"],
                [
                    {
                        "part_index": index,
                        "byte_offset": sum(len(value) for value in parts[:index]),
                        "size_bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                    }
                    for index, payload in enumerate(parts)
                ],
            )
            self.assertEqual(
                first_receipt["native_source_binding"]["verification"]["timing"],
                "completed_before_vtk_geometry_reader",
            )
            self.assertEqual(
                first_receipt["native_source_binding"]["verification"]["vtk_input"],
                "retained_verified_file_descriptor",
            )
            self.assertEqual(
                first_receipt["native_source_binding"]["verification"]["post_vtk_fstat"],
                "unchanged",
            )
            self.assertEqual(first_receipt["geometry"]["declared_cell_count"], 1)
            self.assertEqual(first_receipt["geometry"]["vtk_loaded_cell_count"], 1)
            self.assertEqual(
                first_receipt["geometry"]["reader_audit"],
                {
                    "disabled_point_array_count": 1,
                    "disabled_point_arrays": ["UMeanTrim"],
                    "disabled_cell_array_count": 1,
                    "disabled_cell_arrays": ["pMeanTrim"],
                },
            )
            self.assertFalse(first_receipt["geometry"]["remeshing"])
            self.assertFalse(first_receipt["geometry"]["reordering"])
            self.assertEqual(
                first_receipt["kernel"]["versions"]["vtk"], REQUIRED_VTK_VERSION
            )
            self.assertEqual(
                [row["nominal_spacing_mm"] for row in first_receipt["artifacts"]],
                [1, 2, 5, 10],
            )
            self.assertEqual(
                first_receipt["execution"]["containing_cell_query_cache"],
                {
                    "enabled": True,
                    "key_id": QUERY_CACHE_KEY_ID,
                    "total_rows": 67_384,
                    "unique_query_keys": 39_362,
                    "cache_hits": 28_022,
                },
            )
            self.assertEqual(
                first_receipt["execution"]["polyhedron_geometry_cache"],
                {
                    "policy": "deterministic_least_recently_used",
                    "maximum_entries": 8_192,
                    "maximum_emitted_triangles": 131_072,
                    "current_entries": 0,
                    "current_emitted_triangles": 0,
                    "peak_entries": 0,
                    "peak_emitted_triangles": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "evictions": 0,
                    "oversized_entry_bypasses": 0,
                    "fail_closed_preparations": 0,
                    "vtk_objects_cached": False,
                },
            )
            self.assertEqual(
                first_receipt["execution"]["polyhedron_evaluation"],
                {
                    "scope": (
                        "broad_phase_polyhedron_visits_for_uncached_exact_xyz_"
                        "tolerance_queries"
                    ),
                    "broad_phase_polyhedron_visit_count": 0,
                    "boundary_count": 0,
                    "inside_count": 0,
                    "outside_count": 0,
                    "ambiguous_count": 0,
                    "winding_classified_count": 0,
                    "minimum_winding_classification_margin_steradian": None,
                    "classification_absolute_tolerance_steradian": 1.0e-3,
                },
            )

            for summary in first_receipt["artifacts"]:
                spacing_mm = summary["nominal_spacing_mm"]
                artifact_path = first / summary["artifact"]
                artifact_bytes = artifact_path.read_bytes()
                artifact = json.loads(artifact_bytes)
                self.assertEqual(artifact["schema"], ARTIFACT_SCHEMA)
                self.assertEqual(summary["sha256"], _sha256_bytes(artifact_bytes))
                self.assertEqual(summary["size_bytes"], len(artifact_bytes))
                self.assertEqual(artifact["kernel"], first_receipt["kernel"])
                self.assertEqual(
                    artifact["assignment_evidence_sha256"],
                    summary["assignment_evidence_sha256"],
                )
                self.assertEqual(
                    artifact["constant_evidence_fields"]["source_sha256"],
                    [_sha256_bytes(payload) for payload in parts],
                )
                self.assertEqual(
                    summary["sample_count"], EXPECTED_SAMPLE_COUNTS[spacing_mm]
                )
                self.assertEqual(len(artifact["rows"]), summary["sample_count"])
                self.assertEqual(
                    len({(row[0], row[1]) for row in artifact["rows"]}),
                    summary["sample_count"],
                )
                self.assertEqual(
                    len({row[0] for row in artifact["rows"]}), 16
                )
                self.assertEqual(
                    sum(artifact["coverage"]["line_sample_counts"].values()),
                    summary["sample_count"],
                )
                self.assertGreater(summary["invalid_count"], 0)
                self.assertEqual(
                    summary["valid_count"] + summary["invalid_count"],
                    summary["sample_count"],
                )
                self.assertIn(
                    NO_CLOSURE_CELL_REASON, summary["invalid_reason_counts"]
                )
                for row in artifact["rows"]:
                    valid, reason, raw_cell_id, candidate_count = row[4:8]
                    if valid:
                        self.assertEqual(reason, "")
                        self.assertEqual(raw_cell_id, 0)
                        self.assertEqual(candidate_count, 1)
                    else:
                        self.assertTrue(reason)
                        self.assertIsNone(raw_cell_id)
                        self.assertEqual(candidate_count, 0)
                self.assertFalse(artifact["claims"]["resolution_convergence"])
                self.assertFalse(artifact["claims"]["model_ordering"])
                self.assertFalse(artifact["claims"]["owner_scientific_signoff"])
                self.assertFalse(artifact["claims"]["scoring_contract_active"])

            serialized = b"".join(
                path.read_bytes()
                for path in sorted(item for item in first.iterdir() if item.is_file())
            )
            self.assertNotIn(str(root).encode(), serialized)
            self.assertNotIn(monolithic.name.encode(), serialized)
            self.assertNotIn(pin.name.encode(), serialized)

    def test_same_size_tamper_is_rejected_before_vtk_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            pin, monolithic, _ = self._fixture(root)
            payload = bytearray(monolithic.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            monolithic.write_bytes(payload)
            output = root / "must-not-exist"
            completed = self._run(root, pin, monolithic, output)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SHA-256 mismatch", completed.stderr)
            self.assertFalse(output.exists())

    def test_path_replacement_during_vtk_read_is_rejected_and_vtk_uses_fd(self) -> None:
        from reference.drivaerml.native_volume_geometry import read_geometry_only_vtu

        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            pin, monolithic, _ = self._fixture(root)
            replacement = root / "replacement.vtu"
            replacement.write_bytes(b"x" * monolithic.stat().st_size)
            output = root / "must-not-exist"
            observed: dict[str, object] = {}

            def replace_path_then_read(vtk_source: Path):
                source = Path(vtk_source)
                observed["source"] = source
                observed["descriptor_identity"] = (
                    source.stat().st_dev,
                    source.stat().st_ino,
                )
                observed["original_identity"] = (
                    monolithic.stat().st_dev,
                    monolithic.stat().st_ino,
                )
                replacement.replace(monolithic)
                return read_geometry_only_vtu(source)

            with mock.patch(
                "reference.drivaerml.native_volume_geometry.read_geometry_only_vtu",
                side_effect=replace_path_then_read,
            ):
                with self.assertRaisesRegex(
                    VelocityAssignmentCLIError,
                    "changed while VTK read the verified monolithic source",
                ):
                    generate_pinned_velocity_assignments(
                        case_id="run_44",
                        native_source_pin=pin,
                        dataset_root=root,
                        monolithic_vtu=monolithic,
                        output_root=output,
                        io_chunk_bytes=31,
                        validation_chunk_cells=1,
                    )

            source = observed["source"]
            self.assertIsInstance(source, Path)
            self.assertIn(source.parent, {Path("/proc/self/fd"), Path("/dev/fd")})
            self.assertEqual(
                observed["descriptor_identity"], observed["original_identity"]
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
