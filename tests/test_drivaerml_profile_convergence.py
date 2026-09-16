from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from reference.drivaerml.profile_convergence import (
    INPUT_SCHEMA,
    OFFICIAL_CASE_ORDER,
    OUTPUT_SCHEMA,
    PROFILE_ORDER,
    ProfileConvergenceError,
    evaluate_profile_convergence,
    kendall_tau_b,
    load_input,
    method_set_sha256,
)
from scripts.check_drivaerml_profile_resolution_convergence import main as cli_main


def _artifact_sha(index: int) -> str:
    return hashlib.sha256(f"artifact-{index}".encode()).hexdigest()


def _methods(*, trained_count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "method_id": "physics-null",
            "role": "physics_null",
            "prediction_artifact_sha256": _artifact_sha(0),
        },
        {
            "method_id": "nearest-training",
            "role": "nearest_training_design_vector_control",
            "prediction_artifact_sha256": _artifact_sha(1),
        },
        *[
            {
                "method_id": f"model-{index}",
                "role": "trained_model_checkpoint",
                "model_id": f"architecture-{index}",
                "checkpoint_id": f"checkpoint-{index}",
                "prediction_artifact_sha256": _artifact_sha(index + 2),
            }
            for index in range(trained_count)
        ],
    ]


def _document(
    *,
    trained_count: int = 3,
    case_order: tuple[str, ...] = ("run_1", "run_44"),
) -> dict[str, object]:
    methods = _methods(trained_count=trained_count)
    losses: list[list[list[list[float]]]] = []
    for method_index in range(len(methods)):
        method_cases: list[list[list[float]]] = []
        for case_index in range(len(case_order)):
            case_lines: list[list[float]] = []
            for profile_index in range(len(PROFILE_ORDER)):
                reference = (
                    1.0
                    + 0.2 * method_index
                    + 0.01 * case_index
                    + 0.0001 * profile_index
                )
                case_lines.append(
                    [
                        reference,
                        reference * 1.001,
                        reference * 1.002,
                        reference * 1.003,
                    ]
                )
            method_cases.append(case_lines)
        losses.append(method_cases)
    return {
        "schema": INPUT_SCHEMA,
        "schema_version": 1,
        "study_id": "synthetic-genuine-resolution-study",
        "method_set": {
            "pinned_before_study": True,
            "sha256": method_set_sha256(methods),
            "methods": methods,
        },
        "case_order": list(case_order),
        "profile_order": list(PROFILE_ORDER),
        "spacings_mm": [1, 2, 5, 10],
        "losses": losses,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class DrivAerMLProfileConvergenceTests(unittest.TestCase):
    def test_reduced_pilot_is_never_activation_review_eligible(self) -> None:
        result = evaluate_profile_convergence(_document())

        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertEqual(
            result["status"],
            "ineligible_incomplete_official_484_case_scope",
        )
        self.assertFalse(result["scope"]["complete_official_case_order"])
        self.assertIn(
            "incomplete_official_484_case_scope", result["blocking_reasons"]
        )
        self.assertTrue(result["selection"]["activation_candidate_passed"])

    def test_484_case_membership_and_order_must_both_be_exact(self) -> None:
        permuted = list(OFFICIAL_CASE_ORDER)
        permuted[0], permuted[1] = permuted[1], permuted[0]
        result = evaluate_profile_convergence(
            _document(case_order=tuple(permuted))
        )

        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertFalse(result["scope"]["complete_official_case_order"])
        self.assertEqual(
            result["status"],
            "ineligible_incomplete_official_484_case_scope",
        )

    def test_complete_passing_study_still_requires_owner_mask(self) -> None:
        result = evaluate_profile_convergence(
            _document(case_order=OFFICIAL_CASE_ORDER)
        )

        self.assertEqual(result["schema"], OUTPUT_SCHEMA)
        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertEqual(result["status"], "blocked_owner_validity_mask_not_bound")
        self.assertIn("owner_validity_mask_not_bound", result["blocking_reasons"])
        self.assertFalse(result["scope"]["owner_validity_mask_bound"])
        self.assertTrue(result["does_not_activate_scoring_contract"])
        self.assertFalse(result["owner_scientific_approval_claimed"])
        self.assertEqual(
            [item["spacing_mm"] for item in result["spacing_results"]],
            [2, 5, 10],
        )
        self.assertTrue(all(item["passed"] for item in result["spacing_results"]))
        ten_mm = result["spacing_results"][2]
        self.assertEqual(ten_mm["method_ordering"]["tau_b"], 1.0)
        self.assertEqual(ten_mm["method_ordering"]["pair_counts"]["total"], 10)
        self.assertEqual(len(ten_mm["methods"]), 5)
        for method in ten_mm["methods"]:
            self.assertEqual(len(method["cases"]), len(OFFICIAL_CASE_ORDER))
            self.assertEqual(len(method["cases"][0]["case_lines"]), 16)

    def test_exact_484_study_fails_if_finer_candidate_fails(self) -> None:
        document = _document(case_order=OFFICIAL_CASE_ORDER)
        reference = document["losses"][0][0][0][0]
        # Fail only the 2 mm case-line gate.  The retained 10 mm candidate
        # still passes, so this is a regression guard against selecting it
        # without enforcing the complete prescribed 2/5/10 mm envelope.
        document["losses"][0][0][0][1] = reference * 1.0200001

        result = evaluate_profile_convergence(document)

        self.assertFalse(result["spacing_results"][0]["passed"])
        self.assertTrue(result["spacing_results"][1]["passed"])
        self.assertTrue(result["spacing_results"][2]["passed"])
        self.assertTrue(result["selection"]["activation_candidate_passed"])
        self.assertFalse(
            result["selection"]["all_prescribed_candidate_spacings_passed"]
        )
        self.assertEqual(
            result["selection"]["failed_candidate_spacings_mm"], [2]
        )
        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertEqual(result["status"], "ineligible_resolution_threshold_failure")
        self.assertIn(
            "resolution_threshold_or_method_order_failure",
            result["blocking_reasons"],
        )

    def test_missing_genuine_models_reports_ineligible_without_inventing_them(
        self,
    ) -> None:
        document = _document(
            trained_count=0, case_order=OFFICIAL_CASE_ORDER
        )
        result = evaluate_profile_convergence(document)

        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertEqual(
            result["status"],
            "ineligible_incomplete_or_unpinned_genuine_method_set",
        )
        self.assertEqual(
            result["method_set"]["role_counts"]["trained_model_checkpoint"], 0
        )
        self.assertFalse(
            result["method_set"]["checks"][
                "at_least_three_distinct_trained_model_checkpoints"
            ]
        )
        self.assertTrue(result["spacing_results"][2]["passed"])

    def test_unpinned_method_set_is_ineligible(self) -> None:
        document = _document()
        document["method_set"]["pinned_before_study"] = False
        result = evaluate_profile_convergence(document)
        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertFalse(result["method_set"]["checks"]["pinned_before_study"])

    def test_duplicate_trained_pair_is_not_counted_as_distinct(self) -> None:
        document = _document()
        methods = document["method_set"]["methods"]
        methods[4]["model_id"] = methods[3]["model_id"]
        methods[4]["checkpoint_id"] = methods[3]["checkpoint_id"]
        document["method_set"]["sha256"] = method_set_sha256(methods)

        result = evaluate_profile_convergence(document)
        self.assertFalse(result["profile_resolution_activation_eligible"])
        self.assertEqual(
            result["method_set"]["distinct_trained_model_checkpoint_pairs"], 2
        )

    def test_zero_reference_uses_absolute_rule_at_every_reduction_level(self) -> None:
        document = _document(case_order=("run_1",))
        # Give all 16 lines of one method a zero reference so its case and
        # aggregate macros also exercise the zero-reference rule.
        for line in document["losses"][0][0]:
            line[:] = [0.0, 0.5e-12, 0.75e-12, 1.0e-12]
        result = evaluate_profile_convergence(document)
        method = result["spacing_results"][2]["methods"][0]
        self.assertEqual(
            method["aggregate_comparison"]["rule"], "absolute_zero_reference"
        )
        self.assertTrue(method["aggregate_comparison"]["passed"])
        self.assertEqual(
            method["cases"][0]["macro_comparison"]["rule"],
            "absolute_zero_reference",
        )
        self.assertEqual(
            method["cases"][0]["case_lines"][0]["comparison"]["rule"],
            "absolute_zero_reference",
        )

    def test_zero_reference_absolute_disagreement_fails(self) -> None:
        document = _document()
        document["losses"][0][0][0] = [0.0, 0.0, 0.0, 1.0000001e-12]
        result = evaluate_profile_convergence(document)
        comparison = result["spacing_results"][2]["methods"][0]["cases"][0][
            "case_lines"
        ][0]["comparison"]
        self.assertEqual(comparison["rule"], "absolute_zero_reference")
        self.assertFalse(comparison["passed"])
        self.assertFalse(result["profile_resolution_activation_eligible"])

    def test_case_line_threshold_failure_cannot_be_hidden_by_macros(self) -> None:
        document = _document()
        reference = document["losses"][0][0][0][0]
        document["losses"][0][0][0][3] = reference * 1.0200001

        result = evaluate_profile_convergence(document)
        method = result["spacing_results"][2]["methods"][0]
        self.assertTrue(method["aggregate_comparison"]["passed"])
        self.assertTrue(method["cases"][0]["macro_comparison"]["passed"])
        self.assertFalse(
            method["cases"][0]["case_lines"][0]["comparison"]["passed"]
        )
        self.assertFalse(method["passed"])
        self.assertFalse(result["profile_resolution_activation_eligible"])

    def test_ranking_reversal_fails_even_when_all_loss_thresholds_pass(self) -> None:
        document = _document(case_order=("run_1",))
        method_count = len(document["method_set"]["methods"])
        for method_index in range(method_count):
            reference = 1.0 + 0.001 * method_index
            candidate = 1.0 + 0.001 * (method_count - 1 - method_index)
            for line in document["losses"][method_index][0]:
                line[:] = [reference, reference, reference, candidate]

        result = evaluate_profile_convergence(document)
        ten_mm = result["spacing_results"][2]
        self.assertTrue(all(method["passed"] for method in ten_mm["methods"]))
        self.assertEqual(ten_mm["method_ordering"]["tau_b"], -1.0)
        self.assertFalse(ten_mm["method_ordering"]["passed"])
        self.assertFalse(result["profile_resolution_activation_eligible"])

    def test_kendall_tau_b_uses_absolute_ties(self) -> None:
        result = kendall_tau_b(
            [1.0, 1.0 + 0.5e-12, 2.0],
            [3.0, 3.0 + 0.75e-12, 4.0],
        )
        self.assertEqual(result["tau_b"], 1.0)
        self.assertEqual(result["pair_counts"]["tied_both"], 1)
        self.assertEqual(result["pair_counts"]["concordant"], 2)

        asymmetric = kendall_tau_b(
            [1.0, 1.0 + 0.5e-12, 2.0],
            [1.0, 1.0 + 2.0e-12, 2.0],
        )
        self.assertAlmostEqual(asymmetric["tau_b"], 2.0 / math.sqrt(6.0))
        self.assertEqual(asymmetric["pair_counts"]["tied_reference_only"], 1)

    def test_completely_tied_ranking_is_undefined_and_fails_closed(self) -> None:
        document = _document(case_order=("run_1",))
        for method_losses in document["losses"]:
            for line in method_losses[0]:
                line[:] = [1.0, 1.0, 1.0, 1.0]
        result = evaluate_profile_convergence(document)
        ordering = result["spacing_results"][2]["method_ordering"]
        self.assertIsNone(ordering["tau_b"])
        self.assertFalse(ordering["defined"])
        self.assertFalse(ordering["passed"])
        self.assertFalse(result["profile_resolution_activation_eligible"])

    def test_kendall_rejects_non_numeric_scores_and_invalid_tolerance(self) -> None:
        with self.assertRaisesRegex(ProfileConvergenceError, "must be numeric"):
            kendall_tau_b([True, 2.0], [1.0, 2.0])
        with self.assertRaisesRegex(ProfileConvergenceError, "non-negative"):
            kendall_tau_b([1.0, 2.0], [1.0, 2.0], tie_tolerance=-1.0)

    def test_missing_profile_loss_is_a_hard_coverage_error(self) -> None:
        document = _document()
        document["losses"][0][0].pop()
        with self.assertRaisesRegex(ProfileConvergenceError, "exactly 16 profiles"):
            evaluate_profile_convergence(document)

    def test_missing_spacing_and_nonfinite_or_negative_losses_are_rejected(
        self,
    ) -> None:
        missing = _document()
        missing["losses"][0][0][0].pop()
        with self.assertRaisesRegex(ProfileConvergenceError, "four spacing losses"):
            evaluate_profile_convergence(missing)

        for bad_value in (float("nan"), float("inf"), -1.0, True):
            with self.subTest(bad_value=bad_value):
                invalid = _document()
                invalid["losses"][0][0][0][0] = bad_value
                with self.assertRaisesRegex(
                    ProfileConvergenceError, "finite and non-negative|numeric"
                ):
                    evaluate_profile_convergence(invalid)

    def test_order_and_method_pin_are_exact(self) -> None:
        reordered = _document()
        reordered["profile_order"][0], reordered["profile_order"][1] = (
            reordered["profile_order"][1],
            reordered["profile_order"][0],
        )
        with self.assertRaisesRegex(ProfileConvergenceError, "profile_order"):
            evaluate_profile_convergence(reordered)

        bad_pin = _document()
        bad_pin["method_set"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ProfileConvergenceError, "does not match"):
            evaluate_profile_convergence(bad_pin)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProfileConvergenceError, "duplicate"):
                load_input(path)

    def test_cli_writes_evidence_and_distinguishes_ineligible_from_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "study.json"
            output_path = root / "evidence.json"
            _write_json(
                input_path, _document(case_order=OFFICIAL_CASE_ORDER)
            )
            standard_output = io.StringIO()
            standard_error = io.StringIO()
            with (
                contextlib.redirect_stdout(standard_output),
                contextlib.redirect_stderr(standard_error),
            ):
                return_code = cli_main(
                    ["--input", str(input_path), "--output", str(output_path)]
                )
            self.assertEqual(return_code, 1)
            self.assertEqual(standard_error.getvalue(), "")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(result["profile_resolution_activation_eligible"])
            self.assertIn(
                "owner_validity_mask_not_bound", result["blocking_reasons"]
            )
            self.assertRegex(result["input"]["file_byte_sha256"], r"^[0-9a-f]{64}$")

            ineligible = _document(
                trained_count=0, case_order=OFFICIAL_CASE_ORDER
            )
            _write_json(input_path, ineligible)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                return_code = cli_main(
                    ["--input", str(input_path), "--output", str(output_path)]
                )
            self.assertEqual(return_code, 1)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(result["profile_resolution_activation_eligible"])

            input_path.write_text("not JSON\n", encoding="utf-8")
            standard_error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                standard_error
            ):
                return_code = cli_main(
                    ["--input", str(input_path), "--output", str(output_path)]
                )
            self.assertEqual(return_code, 2)
            self.assertIn("not valid", standard_error.getvalue())

    def test_contract_threshold_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_path = Path(temporary) / "contract.json"
            original = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "benchmark-specs"
                    / "drivaerml"
                    / "proposal"
                    / "contract-proposal.json"
                ).read_text(encoding="utf-8")
            )
            drifted = copy.deepcopy(original)
            drifted["deferred_contracts"]["profiles_and_cp_cuts"][
                "velocity_profiles"
            ]["resolution_convergence"]["aggregate_limit"] = 0.006
            _write_json(contract_path, drifted)
            with self.assertRaisesRegex(ProfileConvergenceError, "refuses drift"):
                evaluate_profile_convergence(
                    _document(), contract_proposal=contract_path
                )


if __name__ == "__main__":
    unittest.main()
