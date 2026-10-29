from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "examples" / "airfrans-profile-extraction" / "extract.py"
REFERENCE_FIXTURE = EXTRACTOR.with_name("example_extraction.json")
DATASET_ROOT = os.environ.get("AIRFRANS_DATASET_ROOT")
CASE_NAME = "airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548"


@unittest.skipUnless(
    DATASET_ROOT,
    "set AIRFRANS_DATASET_ROOT to run the official-data AirfRANS extraction test",
)
class AirfransExtractorIntegrationTests(unittest.TestCase):
    def test_official_case_reproduces_the_committed_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "example_extraction.json"
            subprocess.run(
                [
                    sys.executable,
                    str(EXTRACTOR),
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(output_path.read_bytes(), REFERENCE_FIXTURE.read_bytes())

    def test_native_numpy_velocity_uses_the_prediction_path(self) -> None:
        import airfrans as af

        simulation = af.Simulation(root=str(DATASET_ROOT), name=CASE_NAME)
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            velocity_path = temporary_root / "velocity.npy"
            output_path = temporary_root / "predicted_profiles.json"
            np.save(
                velocity_path,
                np.asarray(
                    simulation.internal.point_data["U"][:, :2], dtype=np.float64
                ),
                allow_pickle=False,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(EXTRACTOR),
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--velocity-predictions",
                    str(velocity_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            prediction = json.loads(output_path.read_text(encoding="utf-8"))
            reference = json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
            self.assertEqual(prediction["provenance"]["mode"], "model_prediction")
            self.assertIn(
                "velocity_predictions", prediction["provenance"]["input_hashes"]
            )
            self.assertEqual(prediction["cases"], reference["cases"])


if __name__ == "__main__":
    unittest.main()
