from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import vtk  # noqa: F401
except ImportError:
    raise unittest.SkipTest(
        "requires optional VTK for DrivAerML prediction convergence tests"
    ) from None

from reference.drivaerml.profile_convergence import method_set_sha256
from reference.drivaerml.profile_convergence_evaluator import (
    FALSE_CLAIMS,
    PREDICTION_STUDY_SCHEMA,
    prediction_manifest_set_identity,
)
from scripts.evaluate_drivaerml_profile_resolution_from_predictions import (
    _IMPLEMENTATION_FILES,
    REQUIRED_NUMPY_VERSION,
    REQUIRED_PYTHON_VERSION,
    NativeProfileConvergenceCLIError,
    _validated_runtime_binding,
    load_study_config,
    main,
    run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _manifest(root: Path) -> Path:
    root.mkdir()
    chunk = root / "chunk.npz"
    np.savez(
        chunk,
        raw_cell_id=np.arange(2, dtype=np.int64),
        pMeanTrim=np.zeros(2, dtype=np.float64),
        UMeanTrim=np.zeros((2, 3), dtype=np.float64),
    )
    document = {
        "format": "drivaerml-native-prediction-chunks-candidate",
        "format_version": 1,
        "artifact_role": "local_evaluator_input_not_official_submission_artifact",
        "case_id": "run_1",
        "support_id": "volume_native_cells",
        "association": "CellData",
        "total_row_count": 2,
        "field_components": {"pMeanTrim": 1, "UMeanTrim": 3},
        "chunks": [
            {
                "chunk_index": 0,
                "file": "chunk.npz",
                "sha256": _sha256(chunk),
                "row_count": 2,
                "raw_cell_id_start": 0,
                "raw_cell_id_stop": 2,
            }
        ],
    }
    path = root / "manifest.json"
    _write_json(path, document)
    return path


def _config(root: Path) -> tuple[Path, Path]:
    manifest = _manifest(root / "prediction")
    digest, _, _ = prediction_manifest_set_identity(["run_1"], [manifest])
    methods = [
        {
            "method_id": "physics-null",
            "role": "physics_null",
            "prediction_artifact_sha256": digest,
        }
    ]
    document = {
        "schema": PREDICTION_STUDY_SCHEMA,
        "schema_version": 1,
        "study_id": "one-case-candidate",
        "method_set": {
            "pinned_before_study": True,
            "sha256": method_set_sha256(methods),
            "methods": methods,
        },
        "case_order": ["run_1"],
        "prediction_manifests": [
            {
                "method_id": "physics-null",
                "case_manifests": [
                    {"case_id": "run_1", "manifest": "prediction/manifest.json"}
                ],
            }
        ],
    }
    path = root / "study.json"
    _write_json(path, document)
    return path, manifest


class DrivAerMLNativeProfileConvergenceCLITests(unittest.TestCase):
    def test_runtime_binding_is_exact_and_does_not_require_vtk(self) -> None:
        with (
            mock.patch(
                "scripts.evaluate_drivaerml_profile_resolution_"
                "from_predictions.platform.python_version",
                return_value=REQUIRED_PYTHON_VERSION,
            ),
            mock.patch.object(np, "__version__", REQUIRED_NUMPY_VERSION),
        ):
            self.assertEqual(
                _validated_runtime_binding(),
                {
                    "python": "3.12.13",
                    "numpy": "2.2.6",
                    "required_python": "3.12.13",
                    "required_numpy": "2.2.6",
                    "vtk_used_by_this_reduction": False,
                },
            )

    def test_runtime_mismatch_precedes_manifest_and_native_source_io(self) -> None:
        mismatches = (
            ("3.12.12", REQUIRED_NUMPY_VERSION),
            (REQUIRED_PYTHON_VERSION, "2.2.5"),
        )
        for python_version, numpy_version in mismatches:
            with (
                self.subTest(
                    python_version=python_version,
                    numpy_version=numpy_version,
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.platform.python_version",
                    return_value=python_version,
                ),
                mock.patch.object(np, "__version__", numpy_version),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._implementation_binding"
                ) as implementation_binding,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.load_study_config"
                ) as config_loader,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.prediction_manifest_set_identity"
                ) as manifest_loader,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.aggregate_velocity_assignments"
                ) as mapping_loader,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.load_native_source_pin"
                ) as source_loader,
                self.assertRaisesRegex(
                    NativeProfileConvergenceCLIError,
                    "requires exactly Python 3.12.13 and NumPy 2.2.6",
                ),
            ):
                run(SimpleNamespace())
            implementation_binding.assert_not_called()
            config_loader.assert_not_called()
            manifest_loader.assert_not_called()
            mapping_loader.assert_not_called()
            source_loader.assert_not_called()

    def test_implementation_binding_covers_direct_local_dependencies(self) -> None:
        required = {
            "scripts/evaluate_drivaerml_profile_resolution_from_predictions.py",
            "scripts/aggregate_drivaerml_velocity_assignments.py",
            "scripts/generate_drivaerml_velocity_assignments.py",
            "reference/drivaerml/profile_convergence_evaluator.py",
            "reference/drivaerml/profile_convergence.py",
            "reference/drivaerml/diagnostic_evaluator.py",
            "reference/drivaerml/evaluator.py",
            "reference/drivaerml/prediction_chunks.py",
            "reference/drivaerml/retained_file.py",
            "reference/drivaerml/source.py",
            "reference/drivaerml/autocfd5.py",
            "reference/drivaerml/velocity_assignments.py",
        }
        self.assertTrue(required.issubset(_IMPLEMENTATION_FILES))
        self.assertEqual(len(_IMPLEMENTATION_FILES), len(set(_IMPLEMENTATION_FILES)))
        root = Path(__file__).resolve().parents[1]
        for relative in _IMPLEMENTATION_FILES:
            with self.subTest(relative=relative):
                self.assertTrue((root / relative).is_file())

    def test_real_data_run_rejects_dirty_or_unresolved_implementation(self) -> None:
        for binding in (
            {
                "git_revision": None,
                "git_worktree_clean": True,
                "source_files": [],
            },
            {
                "git_revision": "c" * 40,
                "git_worktree_clean": False,
                "source_files": [],
            },
        ):
            with (
                self.subTest(binding=binding),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._validated_runtime_binding",
                    return_value={
                        "python": REQUIRED_PYTHON_VERSION,
                        "numpy": REQUIRED_NUMPY_VERSION,
                        "required_python": REQUIRED_PYTHON_VERSION,
                        "required_numpy": REQUIRED_NUMPY_VERSION,
                        "vtk_used_by_this_reduction": False,
                    },
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._implementation_binding",
                    return_value=binding,
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.load_study_config"
                ) as config_loader,
                self.assertRaisesRegex(
                    NativeProfileConvergenceCLIError, "clean Git checkout"
                ),
            ):
                run(SimpleNamespace())
            config_loader.assert_not_called()

    def test_config_is_closed_ordered_and_resolves_relative_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, manifest = _config(root)
            value, digest, cases, methods, manifests = load_study_config(path)

            self.assertEqual(value["study_id"], "one-case-candidate")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(cases, ("run_1",))
            self.assertEqual(methods[0]["method_id"], "physics-null")
            self.assertEqual(manifests["physics-null"], (manifest.resolve(),))

    def test_config_rejects_duplicate_keys_and_case_order_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema":"one","schema":"two"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                NativeProfileConvergenceCLIError, "duplicate"
            ):
                load_study_config(duplicate)

            path, _ = _config(root)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["prediction_manifests"][0]["case_manifests"][0][
                "case_id"
            ] = "run_2"
            _write_json(path, document)
            with self.assertRaisesRegex(
                NativeProfileConvergenceCLIError, "differs from case_order"
            ):
                load_study_config(path)

    def test_cli_distinguishes_valid_blocked_study_from_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "evidence.json"
            candidate = {
                "schema": "drivaerml-profile-resolution-native-prediction-evidence-v1",
                "schema_version": 1,
                "study_id": "blocked-no-models",
                "status": "blocked_missing_or_unpinned_genuine_method_predictions",
                "profile_resolution_candidate_eligible_for_owner_review": False,
                "claims": dict(FALSE_CLAIMS),
            }
            standard_output = io.StringIO()
            standard_error = io.StringIO()
            with (
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.run",
                    return_value=candidate,
                ),
                contextlib.redirect_stdout(standard_output),
                contextlib.redirect_stderr(standard_error),
            ):
                return_code = main(
                    [
                        "--input",
                        str(root / "unused.json"),
                        "--mappings-root",
                        str(root),
                        "--dataset-root",
                        str(root),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(return_code, 1)
            self.assertEqual(standard_error.getvalue(), "")
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["claims"], FALSE_CLAIMS)
            self.assertFalse(
                written["profile_resolution_candidate_eligible_for_owner_review"]
            )

            standard_error = io.StringIO()
            with (
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.run",
                    side_effect=NativeProfileConvergenceCLIError("broken study"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(standard_error),
            ):
                return_code = main(
                    [
                        "--input",
                        str(root / "unused.json"),
                        "--mappings-root",
                        str(root),
                        "--dataset-root",
                        str(root),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(return_code, 2)
            self.assertIn("broken study", standard_error.getvalue())

    def test_run_wires_pilot_scope_limits_runtime_and_source_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = _config(root)
            repo = Path(__file__).resolve().parents[1]
            mapping = SimpleNamespace(
                native_cell_count=2,
                required_raw_cell_ids=lambda: np.arange(2, dtype=np.int64),
            )
            native_truth = SimpleNamespace(
                audit_record=lambda: {"kind": "native-truth"}
            )
            prediction = SimpleNamespace(
                audit_record=lambda: {"kind": "prediction"}
            )
            pinned_case = SimpleNamespace(
                volume_parts=(
                    SimpleNamespace(sha256="a" * 64),
                    SimpleNamespace(sha256="b" * 64),
                )
            )
            pin = SimpleNamespace(case=lambda case_id: pinned_case)
            aggregate = {
                "mode": "explicit_non_public_pilot",
                "status": "incomplete_explicit_pilot_not_public_or_activation_evidence",
                "cases": [{"case_id": "run_1"}],
            }
            finalized = {
                "status": "pilot_or_incomplete_scope_not_eligible_for_owner_review",
                "profile_resolution_candidate_eligible_for_owner_review": False,
            }
            args = SimpleNamespace(
                input=config,
                mappings_root=root / "mappings",
                dataset_root=root,
                native_source_pin=(
                    repo
                    / "benchmark-specs"
                    / "drivaerml"
                    / "proposal"
                    / "native-source-pin.json"
                ),
                autocfd5_profile=(
                    repo
                    / "benchmark-specs"
                    / "drivaerml"
                    / "autocfd5-profiles-v8.json"
                ),
                contract_proposal=(
                    repo
                    / "benchmark-specs"
                    / "drivaerml"
                    / "proposal"
                    / "contract-proposal.json"
                ),
                maximum_prediction_chunk_rows=17,
                io_chunk_bytes=4096,
            )
            with (
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._validated_runtime_binding",
                    return_value={
                        "python": REQUIRED_PYTHON_VERSION,
                        "numpy": REQUIRED_NUMPY_VERSION,
                        "required_python": REQUIRED_PYTHON_VERSION,
                        "required_numpy": REQUIRED_NUMPY_VERSION,
                        "vtk_used_by_this_reduction": False,
                    },
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.aggregate_velocity_assignments",
                    return_value=aggregate,
                ) as aggregate_call,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.load_native_source_pin",
                    return_value=pin,
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.validate_native_source_contract",
                    return_value=(
                        "4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd"
                    ),
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.load_autocfd5_definition",
                    return_value=object(),
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._mapping_from_validated_case",
                    return_value=mapping,
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.validate_mapping_grids",
                    return_value={"case_id": "run_1"},
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._native_truth_for_case",
                    return_value=native_truth,
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.gather_mapped_prediction_field",
                    return_value=prediction,
                ) as gather_call,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.case_profile_losses",
                    return_value=(
                        [[[0.0, 0.0, 0.0, 0.0] for _ in range(16)]][0],
                        {"case_id": "run_1"},
                    ),
                ),
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions.finalize_profile_convergence_evidence",
                    return_value=finalized,
                ) as finalize_call,
                mock.patch(
                    "scripts.evaluate_drivaerml_profile_resolution_"
                    "from_predictions._implementation_binding",
                    return_value={
                        "git_revision": "c" * 40,
                        "git_worktree_clean": True,
                        "source_files": [],
                    },
                ) as implementation_call,
            ):
                result = run(args)

            aggregate_call.assert_called_once()
            self.assertEqual(
                aggregate_call.call_args.kwargs["pilot_case_ids"], ("run_1",)
            )
            gather_call.assert_called_once()
            finalize_call.assert_called_once()
            self.assertEqual(implementation_call.call_count, 2)
            execution = result["execution_input"]
            self.assertEqual(
                execution["mapping_aggregate_mode"],
                "explicit_non_public_pilot",
            )
            self.assertEqual(
                execution["execution_limits"],
                {
                    "io_chunk_bytes": 4096,
                    "maximum_prediction_chunk_rows": 17,
                },
            )
            self.assertEqual(execution["implementation"]["git_revision"], "c" * 40)
            self.assertEqual(
                execution["runtime"],
                {
                    "python": "3.12.13",
                    "numpy": "2.2.6",
                    "required_python": "3.12.13",
                    "required_numpy": "2.2.6",
                    "vtk_used_by_this_reduction": False,
                },
            )
            self.assertFalse(
                result["profile_resolution_candidate_eligible_for_owner_review"]
            )


if __name__ == "__main__":
    unittest.main()
