from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.autocfd5 import load_autocfd5_definition
from reference.drivaerml.cp_mapping import CpMappingRecord
from scripts import build_drivaerml_cp_case_support as case_cli
from scripts.aggregate_drivaerml_cp_support import (
    AGGREGATE_SCHEMA,
    DEFAULT_AUTOCFD5_PROFILE,
    STL_INVENTORY_SCHEMA,
    CpSupportAggregateError,
    aggregate_cp_support_receipts,
    sha256_file,
    write_evidence,
)


CASE_IDS = ("run_1", "run_2")
INVALID_POSITION = 11
NONFINITE_NATIVE_ID = 7


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _named_ascii_stl(components: list[str]) -> str:
    lines: list[str] = []
    for component in components:
        lines.extend(
            (
                f"solid {component}",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 0 0",
                "      vertex 0 1 0",
                "    endloop",
                "  endfacet",
                f"endsolid {component}",
            )
        )
    return "\n".join(lines) + "\n"


class _FakeLocator:
    polygon_count = 208

    def __init__(self, path: Path) -> None:
        self.path = path

    def candidate_polygon_ids(self, point_m: object, radius_m: float) -> tuple[int, ...]:
        raise AssertionError("the injected mapper owns native candidate lookup")

    def polygon_vertices(self, raw_polygon_id: int) -> object:
        raise AssertionError("the injected mapper owns native polygon lookup")


class DrivAerMLCpSupportAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.definition = load_autocfd5_definition(DEFAULT_AUTOCFD5_PROFILE)
        invalid_component = self.definition.cp_component_rules[
            INVALID_POSITION
        ].drivaerml_component
        all_components = sorted(
            {rule.drivaerml_component for rule in self.definition.cp_component_rules}
        )
        self.components = [
            component for component in all_components if component != invalid_component
        ]
        self.raw_stl_id = {
            component: index for index, component in enumerate(self.components)
        }

        self.receipt_pairs: list[tuple[Path, Path]] = []
        self.boundaries: dict[str, Path] = {}
        for case_id in CASE_IDS:
            run_number = int(case_id.removeprefix("run_"))
            case_directory = self.root / case_id
            case_directory.mkdir()
            stl = case_directory / f"drivaer_{run_number}.stl"
            boundary = case_directory / f"boundary_{run_number}.vtp"
            stl.write_text(_named_ascii_stl(self.components), encoding="ascii")
            boundary.write_bytes(f"synthetic-boundary-{case_id}\n".encode("ascii"))
            output_json = self.root / f"cp-support-{case_id}.json"
            output_csv = self.root / f"cp-support-{case_id}.csv"
            with mock.patch.object(case_cli.np, "__version__", "2.2.6"):
                case_cli.build_case_support(
                    case_id=case_id,
                    profile_path=DEFAULT_AUTOCFD5_PROFILE,
                    stl_path=stl,
                    boundary_path=boundary,
                    output_json=output_json,
                    output_csv=output_csv,
                    chunk_facets=2,
                    expected_stl_sha256=sha256_file(stl),
                    expected_boundary_sha256=sha256_file(boundary),
                    locator_factory=_FakeLocator,
                    mapping_builder=self._mappings,
                    boundary_reader=self._boundary,
                )
            self.receipt_pairs.append((output_json, output_csv))
            self.boundaries[case_id] = boundary

        self.pin = self._make_pin()
        self.pin_path = self.root / "native-source-pin.json"
        _write_json(self.pin_path, self.pin)
        self.pristine_json = [
            json.loads(path.read_text(encoding="utf-8"))
            for path, _ in self.receipt_pairs
        ]
        self.pristine_csv = [path.read_bytes() for _, path in self.receipt_pairs]

    def _mappings(
        self,
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
        result: list[CpMappingRecord] = []
        native_id = 0
        for position, probe in enumerate(probe_rows):
            rule = rules_by_id[probe.autocfd_probe_id]
            valid = position != INVALID_POSITION
            point = tuple(probe.point_m)
            raw_polygon_id = native_id if valid else None
            if valid:
                native_id += 1
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
                    raw_stl_triangle_id=(
                        self.raw_stl_id[rule.drivaerml_component] if valid else None
                    ),
                    stl_unit_normal=(0.0, 0.0, 1.0) if valid else None,
                    nominal_displacement_m=0.0 if valid else None,
                    native_closest_point_m=point if valid else None,
                    raw_vtk_polygon_id=raw_polygon_id,
                    native_polygon_unit_normal=(0.0, 0.0, 1.0) if valid else None,
                    bridge_distance_m=0.0 if valid else None,
                    bridge_abs_normal_dot=1.0 if valid else None,
                    native_bounds_candidate_count=1 if valid else 0,
                    native_distance_pass_count=1 if valid else 0,
                    native_normal_pass_count=1 if valid else 0,
                    component_facet_count=1 if valid else 0,
                    projection_candidate_count=1 if valid else 0,
                    review_flags=(),
                    source_sha256=source_sha256,
                )
            )
        self.assertEqual(native_id, 208)
        return tuple(result)

    def _boundary(
        self, path: Path, raw_polygon_ids: object
    ) -> case_cli.BoundaryPressureSelection:
        del path
        raw_ids = tuple(raw_polygon_ids)  # type: ignore[arg-type]
        self.assertEqual(raw_ids, tuple(range(208)))
        return case_cli.BoundaryPressureSelection(
            vtk_version="9.5.2",
            point_count=210,
            polygon_count=208,
            tuple_count=208,
            component_count=1,
            vtk_data_type="float",
            selected_values=tuple(
                (
                    raw_id,
                    float("nan")
                    if raw_id == NONFINITE_NATIVE_ID
                    else float(raw_id),
                )
                for raw_id in raw_ids
            ),
        )

    def _make_pin(self) -> dict[str, object]:
        cases: list[dict[str, object]] = []
        for position, case_id in enumerate(CASE_IDS):
            run_number = int(case_id.removeprefix("run_"))
            boundary = self.boundaries[case_id]
            volume_size = 30 + 2 * position
            cases.append(
                {
                    "case_id": case_id,
                    "run_number": run_number,
                    "boundary": {
                        "path": f"{case_id}/boundary_{run_number}.vtp",
                        "size_bytes": boundary.stat().st_size,
                        "lfs_sha256": sha256_file(boundary),
                    },
                    "surface_cell_area": {
                        "path": f"{case_id}/boundary_cell_area_{run_number}.npy",
                        "size_bytes": 128 + 4 * 208,
                        "lfs_sha256": str(position + 3) * 64,
                        "dtype": "<f4",
                        "element_count": 208,
                        "source_boundary_sha256": sha256_file(boundary),
                    },
                    "volume": {
                        "logical_path_after_assembly": f"{case_id}/volume_{run_number}.vtu",
                        "part_count": 2,
                        "parts": [
                            {
                                "part_index": 0,
                                "path": f"{case_id}/volume_{run_number}.vtu.00.part",
                                "size_bytes": 20 + position,
                                "lfs_sha256": str(position + 5) * 64,
                            },
                            {
                                "part_index": 1,
                                "path": f"{case_id}/volume_{run_number}.vtu.01.part",
                                "size_bytes": 10 + position,
                                "lfs_sha256": str(position + 7) * 64,
                            },
                        ],
                        "total_size_bytes": volume_size,
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
            "case_scope": {
                "case_count": 2,
                "run_number_min": 1,
                "run_number_max": 2,
                "unavailable_or_held_back_run_numbers": [],
            },
            "cases": cases,
            "totals": {
                "boundary_file_count": 2,
                "boundary_bytes": sum(path.stat().st_size for path in self.boundaries.values()),
                "surface_cell_area_file_count": 2,
                "surface_cell_area_bytes": 2 * (128 + 4 * 208),
                "logical_volume_count": 2,
                "volume_part_file_count": 4,
                "reconstructed_volume_bytes": 62,
            },
        }

    def _aggregate(
        self,
        pairs: list[tuple[Path, Path]] | None = None,
        *,
        pilot_case_ids: tuple[str, ...] | None = None,
        profile_path: Path = DEFAULT_AUTOCFD5_PROFILE,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return aggregate_cp_support_receipts(
            native_source_pin_path=self.pin_path,
            profile_path=profile_path,
            receipt_pairs=self.receipt_pairs if pairs is None else pairs,
            pilot_case_ids=pilot_case_ids,
            official_case_ids=CASE_IDS,
            expected_pin_sha256=None,
        )

    def _receipt(self, position: int) -> dict[str, object]:
        return copy.deepcopy(self.pristine_json[position])

    def _rewrite(self, position: int, receipt: dict[str, object]) -> None:
        _write_json(self.receipt_pairs[position][0], receipt)

    def _restore(self, position: int = 0) -> None:
        self._rewrite(position, self._receipt(position))
        self.receipt_pairs[position][1].write_bytes(self.pristine_csv[position])

    def test_complete_aggregate_and_separate_inventory_are_deterministic(self) -> None:
        aggregate, inventory = self._aggregate(list(reversed(self.receipt_pairs)))
        self.assertEqual(aggregate["schema"], AGGREGATE_SCHEMA)
        self.assertEqual(inventory["schema"], STL_INVENTORY_SCHEMA)
        self.assertTrue(aggregate["complete"])
        self.assertTrue(inventory["complete"])
        self.assertFalse(aggregate["public_scoring_support_eligible"])
        self.assertFalse(aggregate["owner_visual_signoff_claimed"])
        self.assertTrue(inventory["separate_from_native_source_pin"])
        self.assertEqual(aggregate["case_count"], 2)
        self.assertEqual(aggregate["aggregate"]["probe_row_count"], 418)
        self.assertEqual(aggregate["aggregate"]["mapping_valid_count"], 416)
        self.assertEqual(aggregate["aggregate"]["mapping_invalid_count"], 2)
        self.assertEqual(aggregate["aggregate"]["truth_valid_count"], 414)
        self.assertEqual(aggregate["aggregate"]["truth_invalid_count"], 4)
        self.assertEqual(
            aggregate["aggregate"]["owner_review_status_counts"],
            {"pending_visual_signoff": 418},
        )
        self.assertEqual(
            aggregate["aggregate"]["pMeanTrim_vtk_data_type_counts"], {"float": 2}
        )
        self.assertEqual(
            [row["case_id"] for row in aggregate["cases"]], list(CASE_IDS)
        )
        self.assertEqual(
            [row["file"] for row in inventory["cases"]],
            ["drivaer_1.stl", "drivaer_2.stl"],
        )
        for position, case in enumerate(aggregate["cases"]):
            self.assertEqual(
                case["receipt_json_sha256"], sha256_file(self.receipt_pairs[position][0])
            )
            self.assertEqual(
                case["mapping_csv_sha256"], sha256_file(self.receipt_pairs[position][1])
            )
            self.assertEqual(
                case["boundary"]["sha256"],
                self.pin["cases"][position]["boundary"]["lfs_sha256"],
            )

        serialized = json.dumps([aggregate, inventory], sort_keys=True)
        self.assertNotIn(str(self.root), serialized)
        first_aggregate = self.root / "first-aggregate.json"
        first_inventory = self.root / "first-inventory.json"
        second_aggregate = self.root / "second-aggregate.json"
        second_inventory = self.root / "second-inventory.json"
        write_evidence(first_inventory, inventory)
        write_evidence(first_aggregate, aggregate)
        aggregate_again, inventory_again = self._aggregate()
        write_evidence(second_inventory, inventory_again)
        write_evidence(second_aggregate, aggregate_again)
        self.assertEqual(first_inventory.read_bytes(), second_inventory.read_bytes())
        self.assertEqual(first_aggregate.read_bytes(), second_aggregate.read_bytes())
        self.assertEqual(
            aggregate["candidate_stl_inventory"]["sha256"],
            hashlib.sha256(first_inventory.read_bytes()).hexdigest(),
        )

    def test_partial_mode_is_explicit_strict_subset_and_non_public(self) -> None:
        with self.assertRaisesRegex(CpSupportAggregateError, "missing=.*run_2"):
            self._aggregate([self.receipt_pairs[0]])

        aggregate, inventory = self._aggregate(
            [self.receipt_pairs[0]], pilot_case_ids=("run_1",)
        )
        self.assertEqual(aggregate["mode"], "partial_pilot")
        self.assertEqual(aggregate["status"], "incomplete_non_public_pilot")
        self.assertFalse(aggregate["complete"])
        self.assertFalse(aggregate["public_evidence_eligible"])
        self.assertFalse(inventory["public_evidence_eligible"])
        self.assertEqual(aggregate["omitted_official_case_count"], 1)
        self.assertIn("not public", aggregate["pilot_warning"])

        with self.assertRaisesRegex(CpSupportAggregateError, "strict subset"):
            self._aggregate(pilot_case_ids=CASE_IDS)
        with self.assertRaisesRegex(CpSupportAggregateError, "official pinned"):
            self._aggregate(
                [self.receipt_pairs[0]], pilot_case_ids=("run_999",)
            )

    def test_missing_duplicate_and_unexpected_pairs_fail_closed(self) -> None:
        with self.assertRaisesRegex(CpSupportAggregateError, "duplicate"):
            self._aggregate([self.receipt_pairs[0], self.receipt_pairs[0]])

        receipt = self._receipt(1)
        receipt["case_id"] = "run_999"
        self._rewrite(1, receipt)
        with self.assertRaisesRegex(CpSupportAggregateError, "unexpected"):
            self._aggregate()

    def test_profile_definition_source_algorithm_and_dependencies_are_exact(self) -> None:
        bad_profile = self.root / "autocfd5-profiles-v8.json"
        bad_profile.write_bytes(DEFAULT_AUTOCFD5_PROFILE.read_bytes() + b"\n")
        with self.assertRaisesRegex(CpSupportAggregateError, "exact candidate v8"):
            self._aggregate(profile_path=bad_profile)

        mutations = (
            (
                lambda receipt: receipt["definition"].__setitem__(
                    "profile_sha256", "f" * 64
                ),
                "v8 definition",
            ),
            (
                lambda receipt: receipt["definition"]["registry_sha256"].__setitem__(
                    "cp_nominal_registry", "f" * 64
                ),
                "v8 definition",
            ),
            (
                lambda receipt: receipt["dependencies"].__setitem__("numpy", "2.5.2"),
                "dependency versions",
            ),
            (
                lambda receipt: receipt["algorithm"]["native_vtp"].__setitem__(
                    "distance_tie_m", 9.0
                ),
                "algorithm constants",
            ),
            (
                lambda receipt: receipt["sources"]["stl"].__setitem__(
                    "file", "wrong.stl"
                ),
                "wrong named STL",
            ),
            (
                lambda receipt: receipt["sources"]["boundary"].__setitem__(
                    "sha256", "f" * 64
                ),
                "boundary identity",
            ),
            (
                lambda receipt: receipt["sources"]["boundary"].__setitem__(
                    "polygon_count", 207
                ),
                "polygon count differs",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                receipt = self._receipt(0)
                mutation(receipt)
                self._rewrite(0, receipt)
                with self.assertRaisesRegex(CpSupportAggregateError, message):
                    self._aggregate()
                self._restore()

    def test_all_ordered_rows_reasons_owner_status_and_cp_are_revalidated(self) -> None:
        mutations = (
            (
                lambda receipt: receipt["rows"].pop(),
                "exactly 209",
            ),
            (
                lambda receipt: receipt["rows"].__setitem__(
                    1, copy.deepcopy(receipt["rows"][0])
                ),
                "ordered unique",
            ),
            (
                lambda receipt: receipt["rows"][0].__setitem__(
                    "owner_review_status", "approved"
                ),
                "differs from its v8 mapping rule",
            ),
            (
                lambda receipt: receipt["rows"][INVALID_POSITION].__setitem__(
                    "mapping_reason", "silently_omitted"
                ),
                "unknown invalid reason",
            ),
            (
                lambda receipt: receipt["rows"][0].__setitem__("truth_Cp", 0.25),
                "truth Cp does not match",
            ),
            (
                lambda receipt: receipt["rows"][0].__setitem__(
                    "raw_vtk_polygon_id", 208
                ),
                "raw VTK polygon ID is out of range",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                receipt = self._receipt(0)
                mutation(receipt)
                self._rewrite(0, receipt)
                with self.assertRaisesRegex(CpSupportAggregateError, message):
                    self._aggregate()
                self._restore()

    def test_csv_hash_schema_rows_and_json_summary_must_all_close(self) -> None:
        receipt = self._receipt(0)
        receipt["summary"]["truth_valid_count"] += 1
        self._rewrite(0, receipt)
        with self.assertRaisesRegex(CpSupportAggregateError, "summary does not match"):
            self._aggregate()
        self._restore()

        csv_path = self.receipt_pairs[0][1]
        csv_path.write_bytes(self.pristine_csv[0] + b"\n")
        receipt = self._receipt(0)
        receipt["artifacts"]["mapping_csv_sha256"] = sha256_file(csv_path)
        self._rewrite(0, receipt)
        with self.assertRaisesRegex(
            CpSupportAggregateError, "CSV schema/order/rows differ"
        ):
            self._aggregate()

    def test_cli_refuses_to_overwrite_a_validated_input(self) -> None:
        from scripts import aggregate_drivaerml_cp_support as aggregate_cli

        with self.assertRaisesRegex(CpSupportAggregateError, "must not overwrite"):
            aggregate_cli.main(
                [
                    "--native-source-pin",
                    str(self.pin_path),
                    "--profile",
                    str(DEFAULT_AUTOCFD5_PROFILE),
                    "--output",
                    str(self.receipt_pairs[0][0]),
                    "--stl-inventory-output",
                    str(self.root / "inventory.json"),
                    "--receipt",
                    str(self.receipt_pairs[0][0]),
                    str(self.receipt_pairs[0][1]),
                ]
            )


if __name__ == "__main__":
    unittest.main()
