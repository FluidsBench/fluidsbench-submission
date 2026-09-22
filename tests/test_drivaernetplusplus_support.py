"""DrivAerNet++ review checks: official split integrity and the dataset publisher.

The publisher tests run hermetically on tiny synthetic ASCII legacy-VTK
PolyData files; no public data download is required.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import weakref
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "benchmark-specs" / "drivaernetplusplus"
PUBLISHER_PATH = SPEC_DIR / "tools" / "publish_scoring_support.py"

from reference import metrics
from reference.evaluate_predictions import (
    relative_l2_from_sufficient_statistics,
    relative_l2_sufficient_statistics,
)
from reference.scoring_support import (
    ScoredPredictions,
    ScoringSupportError,
    align_predictions,
    load_scoring_support,
    load_support_release,
)


def _load_publisher():
    spec = importlib.util.spec_from_file_location("dnpp_publisher", PUBLISHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = _load_publisher()


def _write_ascii_vtk(path: Path, points: np.ndarray, polygons: list[list[int]], pressure: np.ndarray) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        "synthetic drivaernetplusplus fixture",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(" ".join(f"{value:.6f}" for value in row) for row in points)
    total = sum(len(polygon) + 1 for polygon in polygons)
    lines.append(f"POLYGONS {len(polygons)} {total}")
    lines.extend(
        " ".join(str(value) for value in [len(polygon), *polygon]) for polygon in polygons
    )
    lines.append(f"POINT_DATA {len(points)}")
    lines.append("SCALARS p float 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(f"{value:.6f}" for value in pressure)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_ascii_vtk_v5(
    path: Path,
    points: np.ndarray,
    polygons: list[list[int]],
    pressure: np.ndarray,
) -> None:
    """Write the offset/connectivity cell layout used by legacy VTK 5.x."""

    connectivity = [vertex for polygon in polygons for vertex in polygon]
    offsets = [0]
    for polygon in polygons:
        offsets.append(offsets[-1] + len(polygon))
    lines = [
        "# vtk DataFile Version 5.1",
        "synthetic drivaernetplusplus VTK 5 fixture",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(" ".join(f"{value:.6f}" for value in row) for row in points)
    lines.extend(
        [
            f"POLYGONS {len(offsets)} {len(connectivity)}",
            "OFFSETS vtktypeint64",
            " ".join(str(value) for value in offsets),
            "CONNECTIVITY vtktypeint64",
            " ".join(str(value) for value in connectivity),
            f"POINT_DATA {len(points)}",
            "SCALARS p float 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend(f"{value:.6f}" for value in pressure)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _synthetic_case(seed: int) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    # Unit-square split into two triangles plus one skewed quad sharing an edge.
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    polygons = [[0, 1, 2], [0, 2, 3], [1, 4, 5, 2]]
    pressure = rng.normal(loc=-94.5, scale=117.25, size=len(points))
    return points, polygons, pressure


def _split_document(case_ids: list[str]) -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": "drivaernetplusplus",
        "split_id": "sample",
        "case_set_id": "sample",
        "split_label": "Sample",
        "case_id_status": "official",
        "case_count": len(case_ids),
        "case_ids": case_ids,
    }


class TestOfficialSplit(unittest.TestCase):
    def setUp(self):
        with (SPEC_DIR / "submission-spec.json").open(encoding="utf-8") as handle:
            self.spec = json.load(handle)
        self.split_path = SPEC_DIR / "splits" / "official_test.json"
        with self.split_path.open(encoding="utf-8") as handle:
            self.split = json.load(handle)

    def test_split_counts_and_uniqueness(self):
        case_ids = self.split["case_ids"]
        self.assertEqual(len(case_ids), 1154)
        self.assertEqual(self.split["case_count"], 1154)
        self.assertEqual(len(set(case_ids)), 1154)

    def test_split_is_sorted_ascending(self):
        case_ids = self.split["case_ids"]
        self.assertEqual(case_ids, sorted(case_ids))

    def test_split_ids_match_design_id_grammar(self):
        import re

        pattern = re.compile(r"^(?:[EFN]_S_(?:WW|WWC|WWS)_WM_\d{3}|F_D_WM_WW_\d{4})$")
        offenders = [case_id for case_id in self.split["case_ids"] if not pattern.fullmatch(case_id)]
        self.assertEqual(offenders, [])

    def test_spec_entry_matches_split_file(self):
        entry = next(item for item in self.spec["splits"] if item["id"] == "official_test")
        self.assertEqual(entry["case_count"], self.split["case_count"])
        self.assertEqual(entry["case_set_id"], self.split["case_set_id"])
        self.assertEqual(entry["case_id_status"], self.split["case_id_status"])
        digest = hashlib.sha256(self.split_path.read_bytes()).hexdigest()
        self.assertEqual(entry["sha256"], digest)

    def test_prototype_split_still_present_during_review(self):
        ids = [item["id"] for item in self.spec["splits"]]
        self.assertIn("default", ids)
        self.assertIn("official_test", ids)


class TestPublisherSyntheticRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.vtk_dir = root / "SurfacePressureVTK"
        self.vtk_dir.mkdir()
        self.case_ids = ["E_S_WW_WM_001", "F_D_WM_WW_0002"]
        self.fixtures = {}
        for seed, case_id in enumerate(self.case_ids, start=1):
            points, polygons, pressure = _synthetic_case(seed)
            self.fixtures[case_id] = (points, polygons, pressure)
            _write_ascii_vtk(
                self.vtk_dir / f"DrivAer_{case_id}.vtk", points, polygons, pressure
            )
        self.split_path = root / "sample_split.json"
        self.split_path.write_text(
            json.dumps(_split_document(self.case_ids)), encoding="utf-8"
        )
        self.out_dir = root / "release"
        publisher.main(
            [
                "--vtk-dir",
                str(self.vtk_dir),
                "--split",
                str(self.split_path),
                "--out",
                str(self.out_dir),
                "--chunk-size",
                "1",
                "--no-self-check",
            ]
        )
        self.release = load_support_release(self.out_dir / "manifest.json", "sample")

    def _support(self, case_id):
        return load_scoring_support(self.release, case_id, publisher.SUPPORT_ID)

    def test_release_loads_with_generic_loader(self):
        self.assertEqual(list(self.release.cases), self.case_ids)

    def test_every_native_point_appears_exactly_once(self):
        for case_id in self.case_ids:
            support = self._support(case_id)
            points, _, _ = self.fixtures[case_id]
            self.assertEqual(len(support.support_ids), len(points))
            self.assertEqual(sorted(set(support.support_ids.tolist())), support.support_ids.tolist())

    def test_dual_area_weights_match_hand_calculation(self):
        # Two unit right triangles (area 0.5 each) and one 1x1 skewed quad (area 1.0).
        support = self._support(self.case_ids[0])
        expected = np.zeros(6)
        for polygon, area in (([0, 1, 2], 0.5), ([0, 2, 3], 0.5), ([1, 4, 5, 2], 1.0)):
            for vertex in polygon:
                expected[vertex] += area / len(polygon)
        np.testing.assert_allclose(support.weights, expected, rtol=1e-12)

    def test_ground_truth_matches_source_pressure(self):
        for case_id in self.case_ids:
            support = self._support(case_id)
            _, _, pressure = self.fixtures[case_id]
            np.testing.assert_allclose(
                support.targets[publisher.QUANTITY_ID][:, 0], pressure, rtol=0, atol=5e-6
            )

    def test_identity_prediction_is_exactly_perfect(self):
        support = self._support(self.case_ids[0])
        truth = support.targets[publisher.QUANTITY_ID]
        aligned = align_predictions(
            support,
            ScoredPredictions(
                case_id=support.case_id,
                support_name=support.support_name,
                support_ids=support.support_ids,
                values={publisher.QUANTITY_ID: truth.copy()},
            ),
        )[publisher.QUANTITY_ID]
        self.assertEqual(
            metrics.relative_l2(truth[:, 0], aligned[:, 0], support.weights), 0.0
        )
        self.assertEqual(
            metrics.relative_l1(truth[:, 0], aligned[:, 0], support.weights), 0.0
        )
        self.assertEqual(metrics.r2_score(truth[:, 0], aligned[:, 0]), 1.0)

    def test_single_point_perturbation_matches_hand_calculation(self):
        support = self._support(self.case_ids[0])
        truth = support.targets[publisher.QUANTITY_ID][:, 0]
        delta = 2.5
        perturbed = truth.copy()
        perturbed[3] += delta
        expected = 100.0 * np.sqrt(
            support.weights[3] * delta**2 / np.sum(support.weights * truth**2)
        )
        observed = metrics.relative_l2(truth, perturbed, support.weights)
        self.assertAlmostEqual(observed, expected, places=10)

    def test_chunked_statistics_equal_single_pass(self):
        support = self._support(self.case_ids[1])
        truth = support.targets[publisher.QUANTITY_ID]
        prediction = truth + 0.1
        pieces = np.array_split(np.arange(len(truth)), 3)
        chunked = [
            relative_l2_sufficient_statistics(
                truth[piece], prediction[piece], support.weights[piece],
                weighting="support_weights",
            )
            for piece in pieces
        ]
        single = relative_l2_sufficient_statistics(
            truth, prediction, support.weights, weighting="support_weights"
        )
        self.assertAlmostEqual(
            sum(chunk["numerator"] for chunk in chunked), single["numerator"], places=12
        )
        self.assertAlmostEqual(
            sum(chunk["denominator"] for chunk in chunked), single["denominator"], places=9
        )
        self.assertAlmostEqual(
            relative_l2_from_sufficient_statistics(chunked),
            relative_l2_from_sufficient_statistics([single]),
            places=10,
        )

    def test_incomplete_prediction_coverage_fails(self):
        support = self._support(self.case_ids[0])
        truth = support.targets[publisher.QUANTITY_ID]
        with self.assertRaises(ScoringSupportError):
            align_predictions(
                support,
                ScoredPredictions(
                    case_id=support.case_id,
                    support_name=support.support_name,
                    support_ids=support.support_ids[:-1],
                    values={publisher.QUANTITY_ID: truth[:-1].copy()},
                ),
            )

    def test_unknown_prediction_ids_fail(self):
        support = self._support(self.case_ids[0])
        truth = support.targets[publisher.QUANTITY_ID]
        bad_ids = support.support_ids.copy()
        bad_ids[0] = "p99999999"
        with self.assertRaises(ScoringSupportError):
            align_predictions(
                support,
                ScoredPredictions(
                    case_id=support.case_id,
                    support_name=support.support_name,
                    support_ids=bad_ids,
                    values={publisher.QUANTITY_ID: truth.copy()},
                ),
            )


class TestLegacyVtkParser(unittest.TestCase):
    def test_vtk_5_offsets_connectivity_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            vtk_path = Path(tmp) / "DrivAer_E_S_WW_WM_001.vtk"
            points, polygons, pressure = _synthetic_case(3)
            _write_ascii_vtk_v5(vtk_path, points, polygons, pressure)

            observed_points, observed_polygons, point_data = (
                publisher.parse_legacy_vtk_polydata(vtk_path)
            )
            np.testing.assert_allclose(observed_points, points, rtol=0, atol=1e-12)
            self.assertEqual(observed_polygons, polygons)
            np.testing.assert_allclose(point_data["p"], pressure, rtol=0, atol=5e-6)

            columns = publisher.build_case_columns(vtk_path)
            expected_weights = publisher.mass_lumped_dual_areas(points, polygons)
            np.testing.assert_allclose(columns["weight"], expected_weights, rtol=1e-12)


class TestPublisherFailureModes(unittest.TestCase):
    def test_missing_case_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vtk_dir = root / "SurfacePressureVTK"
            vtk_dir.mkdir()
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(_split_document(["E_S_WW_WM_999"])), encoding="utf-8"
            )
            with self.assertRaises(publisher.PublisherError):
                publisher.main(
                    [
                        "--vtk-dir", str(vtk_dir),
                        "--split", str(split_path),
                        "--out", str(root / "release"),
                        "--no-self-check",
                    ]
                )
            self.assertFalse((root / "release").exists())

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "release"
            out_dir.mkdir()
            stale = out_dir / "case-sets" / "old" / "chunk-000.json"
            stale.parent.mkdir(parents=True)
            stale.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                publisher.PublisherError, "output directory must be empty"
            ):
                publisher.write_release(
                    out_dir,
                    ["E_S_WW_WM_001"],
                    root,
                    release_id="test-release",
                    case_set_id="sample",
                    chunk_size=1,
                    published_at="1970-01-01T00:00:00Z",
                )
            self.assertEqual(stale.read_text(encoding="utf-8"), "{}\n")

    def test_nonpositive_numeric_options_are_rejected(self):
        for option, value in (
            ("--chunk-size", "0"),
            ("--chunk-size", "-1"),
            ("--limit", "0"),
            ("--limit", "-1"),
        ):
            with self.subTest(option=option, value=value):
                with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    publisher.main(
                        [
                            "--vtk-dir", ".",
                            "--out", "unused-output",
                            option, value,
                        ]
                    )

    def test_streaming_writer_releases_each_case_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vtk_dir = root / "vtk"
            vtk_dir.mkdir()
            case_ids = ["case-a", "case-b", "case-c"]
            for case_id in case_ids:
                (vtk_dir / f"DrivAer_{case_id}.vtk").touch()

            array_references: list[weakref.ReferenceType] = []

            def build_columns(_path):
                gc.collect()
                self.assertTrue(all(reference() is None for reference in array_references))
                columns = {
                    "support_id": np.asarray(["p00000000", "p00000001"]),
                    "x": np.asarray([0.0, 1.0]),
                    "y": np.asarray([0.0, 0.0]),
                    "z": np.asarray([0.0, 0.0]),
                    "weight": np.asarray([0.5, 0.5]),
                    "p_true": np.asarray([1.0, 2.0]),
                }
                array_references.append(weakref.ref(columns["p_true"]))
                return columns

            with mock.patch.object(
                publisher, "build_case_columns", side_effect=build_columns
            ):
                publisher.write_release(
                    root / "release",
                    case_ids,
                    vtk_dir,
                    release_id="test-release",
                    case_set_id="sample",
                    chunk_size=3,
                    published_at="1970-01-01T00:00:00Z",
                )
            gc.collect()
            self.assertTrue(all(reference() is None for reference in array_references))

    def test_missing_pressure_array_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vtk_path = root / "DrivAer_E_S_WW_WM_001.vtk"
            points, polygons, pressure = _synthetic_case(1)
            _write_ascii_vtk(vtk_path, points, polygons, pressure)
            content = vtk_path.read_text(encoding="ascii").replace("SCALARS p ", "SCALARS q ")
            vtk_path.write_text(content, encoding="ascii")
            with self.assertRaises(publisher.PublisherError):
                publisher.build_case_columns(vtk_path)

    def test_nonfinite_pressure_array_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            vtk_path = Path(tmp) / "DrivAer_E_S_WW_WM_001.vtk"
            points, polygons, pressure = _synthetic_case(1)
            pressure[0] = np.nan
            _write_ascii_vtk(vtk_path, points, polygons, pressure)
            with self.assertRaisesRegex(publisher.PublisherError, "pressure array"):
                publisher.build_case_columns(vtk_path)

    def test_self_check_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vtk_dir = root / "SurfacePressureVTK"
            vtk_dir.mkdir()
            points, polygons, pressure = _synthetic_case(7)
            _write_ascii_vtk(vtk_dir / "DrivAer_N_S_WWC_WM_010.vtk", points, polygons, pressure)
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(_split_document(["N_S_WWC_WM_010"])), encoding="utf-8"
            )
            exit_code = publisher.main(
                [
                    "--vtk-dir", str(vtk_dir),
                    "--split", str(split_path),
                    "--out", str(root / "release"),
                ]
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
