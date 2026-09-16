from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.cp_mapping import CpMappingRecord
from scripts import build_drivaerml_cp_case_support as cli


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "benchmark-specs" / "drivaerml" / "autocfd5-profiles-v8.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_triangle_stl() -> str:
    return """solid Body
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid Body
"""


class _FakeLocator:
    polygon_count = 208

    def __init__(self, path: Path) -> None:
        self.path = path

    def candidate_polygon_ids(self, point_m: object, radius_m: float) -> tuple[int, ...]:
        raise AssertionError("the injected mapping producer should own candidate lookup")

    def polygon_vertices(self, raw_polygon_id: int) -> object:
        raise AssertionError("the injected mapping producer should own polygon lookup")


def _synthetic_mappings(
    case_id: str,
    stl_path: Path,
    locator: object,
    probes: object,
    rules: object,
    *,
    source_sha256: tuple[str, ...],
    chunk_facets: int,
) -> tuple[CpMappingRecord, ...]:
    del stl_path, locator, chunk_facets
    probe_rows = tuple(probes)  # type: ignore[arg-type]
    rules_by_id = {
        rule.autocfd_probe_id: rule for rule in rules  # type: ignore[union-attr]
    }
    if len(probe_rows) != 209 or len(rules_by_id) != 209:
        raise AssertionError("the CLI did not load all 209 hash-bound v8 registry rows")
    result: list[CpMappingRecord] = []
    for index, probe in enumerate(probe_rows):
        rule = rules_by_id[probe.autocfd_probe_id]
        valid = index < 208
        point = tuple(probe.point_m)
        result.append(
            CpMappingRecord(
                case_id=case_id,
                autocfd_probe_id=probe.autocfd_probe_id,
                nominal_point_m=point,
                drivaerml_component=rule.drivaerml_component,
                projection_mode=rule.projection_mode,
                cut_axis=rule.cut_axis,
                cut_value_m=rule.cut_value_m,
                owner_review_status=rule.owner_review_status,
                valid=valid,
                reason="" if valid else "declared_component_absent",
                mapped_point_m=point if valid else None,
                raw_stl_triangle_id=0 if valid else None,
                stl_unit_normal=(0.0, 0.0, 1.0) if valid else None,
                nominal_displacement_m=0.0 if valid else None,
                native_closest_point_m=point if valid else None,
                raw_vtk_polygon_id=index if valid else None,
                native_polygon_unit_normal=(0.0, 0.0, 1.0) if valid else None,
                bridge_distance_m=0.0 if valid else None,
                bridge_abs_normal_dot=1.0 if valid else None,
                native_bounds_candidate_count=1 if valid else 0,
                native_distance_pass_count=1 if valid else 0,
                native_normal_pass_count=1 if valid else 0,
                component_facet_count=1 if valid else 0,
                projection_candidate_count=1 if valid else 0,
                review_flags=("synthetic_review_flag",) if index == 0 else (),
                source_sha256=source_sha256,
            )
        )
    return tuple(result)


def _synthetic_boundary(
    path: Path, raw_polygon_ids: object
) -> cli.BoundaryPressureSelection:
    del path
    raw_ids = tuple(raw_polygon_ids)  # type: ignore[arg-type]
    if raw_ids != tuple(range(208)):
        raise AssertionError("the CLI did not request each valid raw polygon exactly once")
    values = tuple(
        (raw_id, float("nan") if raw_id == 7 else float(raw_id))
        for raw_id in raw_ids
    )
    return cli.BoundaryPressureSelection(
        vtk_version="9.5.2",
        point_count=210,
        polygon_count=208,
        tuple_count=208,
        component_count=1,
        vtk_data_type="float",
        selected_values=values,
    )


class BuildDrivAerMLCpCaseSupportTests(unittest.TestCase):
    def _inputs(self, directory: Path) -> tuple[Path, Path]:
        stl = directory / "drivaer_44.stl"
        boundary = directory / "boundary_44.vtp"
        stl.write_text(_one_triangle_stl(), encoding="ascii")
        boundary.write_bytes(b"synthetic native boundary identity\n")
        return stl, boundary

    def _argv(
        self,
        stl: Path,
        boundary: Path,
        output_json: Path,
        output_csv: Path,
    ) -> list[str]:
        return [
            "--case-id",
            "run_44",
            "--profile",
            str(PROFILE),
            "--case-stl",
            str(stl),
            "--boundary-vtp",
            str(boundary),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--chunk-facets",
            "1",
            "--expected-stl-sha256",
            _sha256(stl),
            "--expected-boundary-sha256",
            _sha256(boundary),
        ]

    def test_source_verification_algorithm_schema_is_exact(self) -> None:
        expected = cli._expected_source_verification()
        self.assertIn(
            "source_integrity",
            cli._algorithm_payload(stl_chunk_facets=1),
        )
        del expected["boundary_vtp"]["post_read_fstat"]  # type: ignore[index]
        with self.assertRaisesRegex(
            cli.CpCaseSupportError,
            "source verification receipt is not the exact candidate method",
        ):
            cli._algorithm_payload(
                stl_chunk_facets=1,
                source_verification=expected,
            )

    def test_cli_emits_deterministic_path_free_complete_candidate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            first_json = directory / "first.json"
            first_csv = directory / "first.csv"
            second_json = directory / "second.json"
            second_csv = directory / "second.csv"

            with (
                mock.patch.object(cli, "Vtk95VtpPolygonLocator", _FakeLocator),
                mock.patch.object(cli, "map_cp_probes_for_case", _synthetic_mappings),
                mock.patch.object(
                    cli, "read_boundary_pressure_selection", _synthetic_boundary
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    cli.main(self._argv(stl, boundary, first_json, first_csv)), 0
                )
                self.assertEqual(
                    cli.main(self._argv(stl, boundary, second_json, second_csv)), 0
                )

            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())
            json_text = first_json.read_text(encoding="utf-8")
            csv_text = first_csv.read_text(encoding="utf-8")
            self.assertNotIn(temporary, json_text)
            self.assertNotIn(temporary, csv_text)

            evidence = json.loads(json_text)
            self.assertEqual(evidence["case_id"], "run_44")
            self.assertEqual(
                evidence["status"],
                "candidate_not_owner_approved_not_active_scoring_support",
            )
            self.assertFalse(evidence["owner_visual_signoff_claimed"])
            self.assertFalse(evidence["algorithm"]["owner_visual_signoff_claimed"])
            self.assertEqual(evidence["definition"]["probe_count"], 209)
            self.assertEqual(evidence["definition"]["rule_count"], 209)
            self.assertEqual(len(evidence["definition"]["registry_sha256"]), 5)
            self.assertEqual(evidence["dependencies"]["vtk"], "9.5.2")
            self.assertEqual(evidence["algorithm"]["stl"]["stream_chunk_facets"], 1)
            source_integrity = evidence["algorithm"]["source_integrity"]
            self.assertEqual(
                set(source_integrity), {"stl", "boundary_vtp"}
            )
            self.assertEqual(
                source_integrity["stl"]["pathname_open"],
                "single_open_retained_through_all_reads",
            )
            self.assertEqual(
                source_integrity["stl"]["parser_inputs"],
                [
                    "strict_ascii_stl_inventory",
                    "component_constrained_geometric_mapping",
                ],
            )
            self.assertEqual(
                source_integrity["boundary_vtp"]["parser_inputs"],
                ["vtk_polygon_locator", "vtk_CellData_pMeanTrim_selection"],
            )
            for receipt in source_integrity.values():
                self.assertEqual(receipt["file_type"], "regular_file_verified_by_fstat")
                self.assertEqual(receipt["post_read_fstat"], "unchanged")
                self.assertEqual(
                    receipt["descriptor_transport"],
                    "verified_procfs_or_devfs_fd_alias_same_device_inode",
                )
            self.assertEqual(
                evidence["algorithm"]["truth"]["equation"],
                "Cp=2*pMeanTrim/Uinf^2",
            )
            self.assertEqual(evidence["sources"]["stl"]["facet_count"], 1)
            self.assertEqual(evidence["sources"]["stl"]["file"], "drivaer_44.stl")
            self.assertEqual(evidence["sources"]["boundary"]["polygon_count"], 208)
            self.assertEqual(
                evidence["sources"]["boundary"]["file"], "boundary_44.vtp"
            )
            self.assertEqual(evidence["summary"]["row_count"], 209)
            self.assertEqual(evidence["summary"]["mapping_valid_count"], 208)
            self.assertEqual(evidence["summary"]["mapping_invalid_count"], 1)
            self.assertEqual(evidence["summary"]["truth_valid_count"], 207)
            self.assertEqual(evidence["summary"]["truth_invalid_count"], 2)
            self.assertEqual(len(evidence["rows"]), 209)

            nonfinite = evidence["rows"][7]
            self.assertTrue(nonfinite["mapping_valid"])
            self.assertFalse(nonfinite["truth_valid"])
            self.assertEqual(nonfinite["truth_reason"], "nonfinite_native_pMeanTrim")
            self.assertIsNone(nonfinite["pMeanTrim_m2_per_s2"])
            self.assertIsNone(nonfinite["truth_Cp"])
            invalid = evidence["rows"][-1]
            self.assertFalse(invalid["mapping_valid"])
            self.assertEqual(invalid["mapping_reason"], "declared_component_absent")
            self.assertEqual(invalid["truth_reason"], "mapping_invalid")

            finite = evidence["rows"][1]
            self.assertAlmostEqual(
                finite["truth_Cp"], 2.0 / (38.889 * 38.889), places=16
            )
            self.assertEqual(
                evidence["artifacts"]["mapping_csv_sha256"], _sha256(first_csv)
            )
            with first_csv.open(newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 209)
            self.assertEqual(csv_rows[7]["truth_reason"], "nonfinite_native_pMeanTrim")
            self.assertEqual(csv_rows[7]["pMeanTrim_m2_per_s2"], "")
            self.assertEqual(csv_rows[-1]["mapping_reason"], "declared_component_absent")

    def test_cli_rejects_source_hash_mismatch_before_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"
            argv = self._argv(stl, boundary, output_json, output_csv)
            argv[-3] = "0" * 64
            locator = mock.Mock(side_effect=AssertionError("must not construct locator"))
            with mock.patch.object(cli, "Vtk95VtpPolygonLocator", locator):
                with self.assertRaisesRegex(
                    cli.CpCaseSupportError, "STL SHA-256 does not match"
                ):
                    cli.main(argv)
            locator.assert_not_called()
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())

    def test_pathname_replacement_does_not_redirect_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            original_stl = stl.read_bytes()
            original_boundary = boundary.read_bytes()
            original_stl_hash = _sha256(stl)
            original_boundary_hash = _sha256(boundary)
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"
            seen: dict[str, Path] = {}

            def replacing_locator(descriptor_path: Path) -> _FakeLocator:
                seen["locator"] = Path(descriptor_path)
                retained_original_stl = directory / "retained-original.stl"
                stl.replace(retained_original_stl)
                boundary.replace(directory / "retained-original.vtp")
                stl.write_bytes(b"replacement pathname bytes\n")
                boundary.write_bytes(b"replacement pathname bytes\n")
                # Some test filesystems quantize ctime.  Move mtime forward
                # explicitly so the required post-read metadata comparison is
                # deterministic while the retained bytes remain untouched.
                retained_stat = retained_original_stl.stat()
                os.utime(
                    retained_original_stl,
                    ns=(retained_stat.st_atime_ns, retained_stat.st_mtime_ns + 1_000_000_000),
                )
                self.assertEqual(Path(descriptor_path).read_bytes(), original_boundary)
                return _FakeLocator(Path(descriptor_path))

            def retained_mapping(*args: object, **kwargs: object) -> tuple[CpMappingRecord, ...]:
                descriptor_path = Path(args[1])
                seen["mapping"] = descriptor_path
                self.assertEqual(descriptor_path.read_bytes(), original_stl)
                return _synthetic_mappings(*args, **kwargs)  # type: ignore[arg-type]

            def retained_pressure(
                descriptor_path: Path, raw_polygon_ids: object
            ) -> cli.BoundaryPressureSelection:
                seen["pressure"] = Path(descriptor_path)
                self.assertEqual(Path(descriptor_path).read_bytes(), original_boundary)
                return _synthetic_boundary(descriptor_path, raw_polygon_ids)

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    cli.CpCaseSupportError, "retained case STL .* changed"
                ):
                    cli.build_case_support(
                        case_id="run_44",
                        profile_path=PROFILE,
                        stl_path=stl,
                        boundary_path=boundary,
                        output_json=output_json,
                        output_csv=output_csv,
                        chunk_facets=1,
                        expected_stl_sha256=original_stl_hash,
                        expected_boundary_sha256=original_boundary_hash,
                        locator_factory=replacing_locator,
                        mapping_builder=retained_mapping,
                        boundary_reader=retained_pressure,
                    )
            self.assertEqual(
                {path.parent for path in seen.values()},
                {next(iter(seen.values())).parent},
            )
            self.assertIn(
                next(iter(seen.values())).parent,
                {Path("/proc/self/fd"), Path("/dev/fd")},
            )
            self.assertNotEqual(_sha256(stl), original_stl_hash)
            self.assertNotEqual(_sha256(boundary), original_boundary_hash)
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())

    def test_in_place_stl_mutation_fails_before_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"

            def mutate_stl(*args: object, **kwargs: object) -> tuple[CpMappingRecord, ...]:
                stl.write_bytes(stl.read_bytes() + b"mutation\n")
                return _synthetic_mappings(*args, **kwargs)  # type: ignore[arg-type]

            with (
                mock.patch.object(cli, "Vtk95VtpPolygonLocator", _FakeLocator),
                mock.patch.object(cli, "map_cp_probes_for_case", mutate_stl),
                mock.patch.object(
                    cli, "read_boundary_pressure_selection", _synthetic_boundary
                ),
            ):
                with self.assertRaisesRegex(
                    cli.CpCaseSupportError, "retained case STL .* changed"
                ):
                    cli.main(self._argv(stl, boundary, output_json, output_csv))
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())

    def test_in_place_boundary_mutation_fails_before_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"

            def mutate_boundary(
                descriptor_path: Path, raw_polygon_ids: object
            ) -> cli.BoundaryPressureSelection:
                boundary.write_bytes(boundary.read_bytes() + b"mutation\n")
                return _synthetic_boundary(descriptor_path, raw_polygon_ids)

            with (
                mock.patch.object(cli, "Vtk95VtpPolygonLocator", _FakeLocator),
                mock.patch.object(cli, "map_cp_probes_for_case", _synthetic_mappings),
                mock.patch.object(
                    cli, "read_boundary_pressure_selection", mutate_boundary
                ),
            ):
                with self.assertRaisesRegex(
                    cli.CpCaseSupportError, "retained boundary VTP .* changed"
                ):
                    cli.main(self._argv(stl, boundary, output_json, output_csv))
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())

    def test_cli_binds_case_id_to_released_source_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            stl.rename(directory / "drivaer_45.stl")
            wrong_stl = directory / "drivaer_45.stl"
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"
            with self.assertRaisesRegex(
                cli.CpCaseSupportError, "must be named drivaer_44.stl"
            ):
                cli.main(
                    self._argv(wrong_stl, boundary, output_json, output_csv)
                )
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())

    def test_cli_rejects_a_modified_profile_before_registry_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            profile = directory / "autocfd5-profiles-v8.json"
            profile.write_bytes(PROFILE.read_bytes() + b"\n")
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"
            argv = self._argv(stl, boundary, output_json, output_csv)
            argv[argv.index("--profile") + 1] = str(profile)
            with mock.patch.object(
                cli,
                "load_autocfd5_definition",
                side_effect=AssertionError("modified profile must fail before loading"),
            ):
                with self.assertRaisesRegex(
                    cli.CpCaseSupportError, "exact candidate v8 SHA-256"
                ):
                    cli.main(argv)
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())

    def test_cli_rejects_missing_probe_row_without_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stl, boundary = self._inputs(directory)
            output_json = directory / "case.json"
            output_csv = directory / "case.csv"

            def incomplete(*args: object, **kwargs: object) -> tuple[CpMappingRecord, ...]:
                return _synthetic_mappings(*args, **kwargs)[:-1]  # type: ignore[arg-type]

            with (
                mock.patch.object(cli, "Vtk95VtpPolygonLocator", _FakeLocator),
                mock.patch.object(cli, "map_cp_probes_for_case", incomplete),
            ):
                with self.assertRaisesRegex(
                    cli.CpCaseSupportError, "returned 208 rows, expected 209"
                ):
                    cli.main(self._argv(stl, boundary, output_json, output_csv))
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())


if __name__ == "__main__":
    unittest.main()
