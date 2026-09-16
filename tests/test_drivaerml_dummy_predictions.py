from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from reference.drivaerml.prediction_chunks import (
    iter_prediction_chunks,
    validate_prediction_chunks,
)
from scripts.create_drivaerml_dummy_predictions import (
    DummyPredictionError,
    RECEIPT_STATUS,
    create_dummy_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "create_drivaerml_dummy_predictions.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


class DrivAerMLDummyPredictionTests(unittest.TestCase):
    def test_separate_outputs_have_identical_bytes_and_fixed_zip_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_receipt = create_dummy_predictions(
                case_id="run_44",
                surface_count=7,
                volume_count=11,
                output_root=first,
                max_chunk_rows=4,
            )
            second_receipt = create_dummy_predictions(
                case_id="run_44",
                surface_count=7,
                volume_count=11,
                output_root=second,
                max_chunk_rows=4,
            )
            self.assertEqual(first_receipt, second_receipt)
            self.assertEqual(_relative_hashes(first), _relative_hashes(second))
            self.assertEqual(first_receipt["status"], RECEIPT_STATUS)
            self.assertFalse(first_receipt["official_submission"])
            self.assertFalse(first_receipt["reads_native_truth"])
            self.assertFalse(first_receipt["claims_model_quality"])
            encoded_receipt = json.dumps(first_receipt, sort_keys=True)
            self.assertNotIn("volume_weight", encoded_receipt)
            self.assertNotIn("cell_volume", encoded_receipt)

            archive_path = (
                first
                / "surface_native_cells"
                / "chunks"
                / "chunk-00000.npz"
            )
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "raw_cell_id.npy",
                        "pMeanTrim.npy",
                        "wallShearStressMeanTrim.npy",
                    ],
                )
                for member in archive.infolist():
                    self.assertEqual(member.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(member.compress_type, zipfile.ZIP_DEFLATED)

    def test_outputs_validate_with_expected_shapes_coverage_and_chunk_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dummy"
            receipt = create_dummy_predictions(
                case_id="run_1",
                surface_count=5,
                volume_count=8,
                output_root=root,
                max_chunk_rows=3,
            )
            expectations = {
                "surface_native_cells": (
                    5,
                    {"pMeanTrim": (1,), "wallShearStressMeanTrim": (3,)},
                ),
                "volume_native_cells": (
                    8,
                    {"pMeanTrim": (1,), "UMeanTrim": (3,)},
                ),
            }
            for support_id, (entity_count, component_shapes) in expectations.items():
                manifest_path = root / support_id / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest["field_components"],
                    {
                        field_name: component_shape[0]
                        for field_name, component_shape in component_shapes.items()
                    },
                )
                self.assertNotIn(
                    "volume_weight",
                    json.dumps(manifest, sort_keys=True),
                )
                self.assertNotIn(
                    "cell_volume",
                    json.dumps(manifest, sort_keys=True),
                )
                validation = validate_prediction_chunks(
                    manifest_path,
                    validation_block_rows=2,
                )
                self.assertEqual(validation.total_row_count, entity_count)
                self.assertEqual(
                    validation.manifest_sha256,
                    receipt["supports"][support_id]["manifest_sha256"],
                )
                cursor = 0
                for chunk in iter_prediction_chunks(
                    manifest_path,
                    hash_chunk_bytes=7,
                    validation_block_rows=2,
                ):
                    rows = chunk.descriptor.row_count
                    self.assertLessEqual(rows, 3)
                    np.testing.assert_array_equal(
                        chunk.raw_cell_id,
                        np.arange(cursor, cursor + rows, dtype=np.int64),
                    )
                    self.assertEqual(chunk.raw_cell_id.dtype, np.dtype("<i8"))
                    for field_name, tail_shape in component_shapes.items():
                        field = chunk.field(field_name)
                        expected_shape = (
                            (rows,)
                            if tail_shape == (1,)
                            else (rows, *tail_shape)
                        )
                        self.assertEqual(field.shape, expected_shape)
                        self.assertEqual(field.dtype, np.dtype("<f4"))
                        self.assertTrue(np.all(field == 0.0))
                    cursor += rows
                self.assertEqual(cursor, entity_count)

            stored_receipt = json.loads(
                (root / "receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored_receipt, receipt)
            self.assertEqual(
                set(stored_receipt["supports"]),
                {"surface_native_cells", "volume_native_cells"},
            )
            self.assertNotIn(b"\n ", (root / "receipt.json").read_bytes())

    def test_invalid_arguments_and_existing_output_are_rejected(self) -> None:
        invalid = (
            {"case_id": "run_0"},
            {"case_id": "case_1"},
            {"surface_count": 0},
            {"volume_count": -1},
            {"max_chunk_rows": 0},
            {"surface_count": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, replacement in enumerate(invalid):
                arguments = {
                    "case_id": "run_1",
                    "surface_count": 3,
                    "volume_count": 4,
                    "output_root": root / f"invalid-{index}",
                    "max_chunk_rows": 2,
                    **replacement,
                }
                with self.subTest(replacement=replacement), self.assertRaises(
                    DummyPredictionError
                ):
                    create_dummy_predictions(**arguments)
                self.assertFalse(Path(arguments["output_root"]).exists())

            existing = root / "existing"
            existing.mkdir()
            marker = existing / "unrelated.txt"
            marker.write_text("preserve me", encoding="utf-8")
            with self.assertRaisesRegex(DummyPredictionError, "already exists"):
                create_dummy_predictions(
                    case_id="run_1",
                    surface_count=3,
                    volume_count=4,
                    output_root=existing,
                    max_chunk_rows=2,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

    def test_cli_generates_both_supports_in_one_command_and_rejects_bad_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cli"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-id",
                    "run_44",
                    "--surface-count",
                    "4",
                    "--volume-count",
                    "6",
                    "--output-root",
                    str(output),
                    "--max-chunk-rows",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            stdout_receipt = json.loads(completed.stdout)
            self.assertEqual(stdout_receipt["case_id"], "run_44")
            self.assertEqual(
                set(stdout_receipt["supports"]),
                {"surface_native_cells", "volume_native_cells"},
            )
            self.assertTrue(
                (output / "surface_native_cells" / "manifest.json").is_file()
            )
            self.assertTrue(
                (output / "volume_native_cells" / "manifest.json").is_file()
            )

            rejected = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-id",
                    "run_1",
                    "--surface-count",
                    "0",
                    "--volume-count",
                    "1",
                    "--output-root",
                    str(root / "bad"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("positive integer", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
