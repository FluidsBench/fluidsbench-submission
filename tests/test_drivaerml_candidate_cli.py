from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import vtk  # noqa: F401
except ImportError:
    raise unittest.SkipTest("requires optional VTK for DrivAerML evaluator imports") from None

from reference.drivaerml.diagnostic_evaluator import (
    DrivAerDiagnosticEvaluatorError,
)
from scripts import evaluate_drivaerml_candidate_diagnostics as diagnostic_cli


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_drivaerml_candidate_case.py"


class DrivAerMLCandidateCLITests(unittest.TestCase):
    def test_help_is_candidate_only_and_lists_both_transports(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("closed candidate evaluator", completed.stdout)
        self.assertIn("--monolithic-vtu", completed.stdout)
        self.assertIn("--multipart", completed.stdout)
        self.assertIn("--surface-prediction-manifest", completed.stdout)
        self.assertIn("--volume-prediction-manifest", completed.stdout)
        for obsolete_option in (
            "--volume-weight-npy",
            "--volume-weight-receipt",
            "--volume-weight-aggregate",
            "--pilot-volume-weight-aggregate-sha256",
            "--allow-incomplete-volume-weight-pilot",
            "--volume-weight-sha256",
        ):
            self.assertNotIn(obsolete_option, completed.stdout)

    def test_diagnostic_cli_uses_v9_registry_and_rejects_probe_inputs(self) -> None:
        required = [
            "--case-id",
            "run_1",
            "--native-source-pin",
            "native-source-pin.json",
            "--dataset-root",
            "dataset",
            "--velocity-mapping-json",
            "velocity.json",
            "--velocity-receipt-json",
            "receipt.json",
            "--volume-prediction-manifest",
            "volume-manifest.json",
            "--multipart",
            "--output",
            "diagnostics.json",
        ]
        args = diagnostic_cli.parse_args(required)
        self.assertEqual(
            args.diagnostic_profile,
            ROOT
            / "benchmark-specs"
            / "drivaerml"
            / "drivaerml-diagnostics-v9.json",
        )
        with self.assertRaises(SystemExit):
            diagnostic_cli.parse_args(
                [*required, "--cp-support-json", "legacy-probes.json"]
            )
        with self.assertRaises(SystemExit):
            diagnostic_cli.parse_args(
                [*required, "--surface-prediction-manifest", "surface.json"]
            )



if __name__ == "__main__":
    unittest.main()
