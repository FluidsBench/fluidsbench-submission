from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_drivaerml_velocity_assignment_aggregate as aggregate_fixture  # noqa: E402
from scripts import build_drivaerml_velocity_owner_review_atlas as atlas_module  # noqa: E402
from scripts.aggregate_drivaerml_velocity_assignments import (  # noqa: E402
    write_manifest as write_assignment_manifest,
)
from scripts.build_drivaerml_velocity_owner_review_atlas import (  # noqa: E402
    ATLAS_ACTIVATION_STATUS,
    ATLAS_SCHEMA,
    ATLAS_STATUS,
    FAILURE_COLUMNS,
    MATPLOTLIB_VERSION,
    NUMPY_VERSION,
    VelocityAtlasError,
    _canonical_json_bytes,
    _refuse_output_collisions,
    analyse_atlas,
    build_manifest,
    load_atlas_inputs,
    render_atlas_pdf,
    write_invalid_samples_csv,
)


def _renderer_is_pinned() -> bool:
    previous_config = os.environ.get("MPLCONFIGDIR")
    try:
        with tempfile.TemporaryDirectory(prefix="drivaerml-velocity-atlas-test-mpl-") as config:
            if previous_config is None:
                os.environ["MPLCONFIGDIR"] = config
            try:
                import matplotlib
                import numpy
            except ImportError:
                return False
    finally:
        if previous_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
    return matplotlib.__version__ == MATPLOTLIB_VERSION and numpy.__version__ == NUMPY_VERSION


class DrivAerMLVelocityOwnerReviewAtlasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = aggregate_fixture.DrivAerMLVelocityAssignmentAggregateTests(
            "test_explicit_pilot_emits_deterministic_compact_path_free_manifest"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.aggregate = self.fixture._aggregate()
        self.aggregate_path = self.fixture.root / "velocity-aggregate.json"
        write_assignment_manifest(self.aggregate_path, self.aggregate)
        with self.fixture._patch_small_samples():
            self.inputs = load_atlas_inputs(
                aggregate_path=self.aggregate_path,
                receipts_root=self.fixture.receipts_root,
                native_source_pin=self.fixture.pin_path,
                autocfd5_profile=aggregate_fixture.PROFILE,
                pilot_case_ids=(aggregate_fixture.CASE_ID,),
                official_case_ids=(aggregate_fixture.CASE_ID,),
                expected_pin_sha256=None,
            )
        self.analysis = analyse_atlas(self.inputs)

    def test_exact_complete_duplicate_free_row_accounting(self) -> None:
        self.assertEqual(self.inputs.mode, "explicit_non_public_pilot")
        self.assertEqual(self.inputs.case_ids, (aggregate_fixture.CASE_ID,))
        self.assertEqual(self.analysis.profile_ids, aggregate_fixture.PROFILE_IDS)
        self.assertEqual(self.analysis.resolutions_mm, (1, 2, 5, 10))
        self.assertEqual(self.analysis.retained_row_count, 64)
        self.assertEqual(self.analysis.valid_count, 44)
        self.assertEqual(self.analysis.invalid_count, 20)
        self.assertEqual(self.analysis.tie_assignment_count, 24)
        self.assertEqual(self.analysis.candidate_count_sum, 68)
        self.assertEqual(
            self.analysis.invalid_reason_counts,
            {aggregate_fixture.NO_CLOSURE_CELL_REASON: 20},
        )
        self.assertEqual(len(self.analysis.per_case), 1)
        self.assertEqual(len(self.analysis.per_profile_resolution), 64)
        for resolution in self.analysis.by_resolution:
            self.assertEqual(resolution["sample_count"], 16)
            self.assertEqual(resolution["valid_count"], 11)
            self.assertEqual(resolution["invalid_count"], 5)

    def test_invalid_csv_retains_every_failure_and_is_byte_deterministic(self) -> None:
        first = self.fixture.root / "invalid-a.csv"
        second = self.fixture.root / "invalid-b.csv"
        identity_a = write_invalid_samples_csv(
            first, inputs=self.inputs, analysis=self.analysis
        )
        identity_b = write_invalid_samples_csv(
            second, inputs=self.inputs, analysis=self.analysis
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(identity_a, identity_b)
        self.assertEqual(identity_a["row_count"], 20)
        with first.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(tuple(rows[0]), FAILURE_COLUMNS)
        self.assertEqual(len(rows), 20)
        self.assertTrue(
            all(row["reason"] == aggregate_fixture.NO_CLOSURE_CELL_REASON for row in rows)
        )
        self.assertTrue(all(row["raw_vtk_cell_id"] == "" for row in rows))
        self.assertEqual(
            [(row["nominal_spacing_mm"], row["profile_id"]) for row in rows[:2]],
            [("1", "V3"), ("1", "V6")],
        )

    def test_manifest_is_path_free_nonactivating_and_never_approved(self) -> None:
        failure_path = self.fixture.root / "invalid.csv"
        failures = write_invalid_samples_csv(
            failure_path, inputs=self.inputs, analysis=self.analysis
        )
        renderer = {
            "matplotlib": MATPLOTLIB_VERSION,
            "numpy": NUMPY_VERSION,
            "bundled_dejavu_sans_sha256": "a" * 64,
            "page_count": 6,
            "pdf_sha256": "b" * 64,
            "pdf_size_bytes": 12345,
        }
        first = build_manifest(
            inputs=self.inputs,
            analysis=self.analysis,
            renderer=renderer,
            invalid_samples_artifact=failures,
        )
        second = build_manifest(
            inputs=self.inputs,
            analysis=self.analysis,
            renderer=copy.deepcopy(renderer),
            invalid_samples_artifact=copy.deepcopy(failures),
        )
        self.assertEqual(first["schema"], ATLAS_SCHEMA)
        self.assertEqual(first["status"], ATLAS_STATUS)
        self.assertEqual(first["activation_status"], ATLAS_ACTIVATION_STATUS)
        self.assertFalse(first["owner_visual_signoff_claimed"])
        self.assertFalse(first["scientific_approval_claimed"])
        self.assertFalse(first["public_scoring_support_eligible"])
        self.assertFalse(first["scope"]["complete_484_public_case_set"])
        self.assertTrue(first["scope"]["complete_duplicate_free_no_omissions"])
        self.assertEqual(first["scope"]["retained_assignment_row_count"], 64)
        self.assertEqual(first["artifacts"]["invalid_samples"]["row_count"], 20)
        self.assertFalse(first["source"]["source_geometry_rendered"])
        self.assertTrue(all(value is False for value in first["claims"].values()))
        self.assertEqual(_canonical_json_bytes(first), _canonical_json_bytes(second))
        serialized = _canonical_json_bytes(first).decode("utf-8")
        self.assertNotIn(str(self.fixture.root), serialized)
        self.assertNotIn("/tmp/", serialized)

    def test_pilot_is_explicit_and_supplied_aggregate_must_equal_replay(self) -> None:
        with self.fixture._patch_small_samples():
            with self.assertRaisesRegex(VelocityAtlasError, "strict assignment replay"):
                load_atlas_inputs(
                    aggregate_path=self.aggregate_path,
                    receipts_root=self.fixture.receipts_root,
                    native_source_pin=self.fixture.pin_path,
                    autocfd5_profile=aggregate_fixture.PROFILE,
                    official_case_ids=(aggregate_fixture.CASE_ID,),
                    expected_pin_sha256=None,
                )

        tampered = json.loads(self.aggregate_path.read_text(encoding="utf-8"))
        tampered["status"] = "invented"
        self.aggregate_path.write_text(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.fixture._patch_small_samples():
            with self.assertRaisesRegex(VelocityAtlasError, "differs from strict replay"):
                load_atlas_inputs(
                    aggregate_path=self.aggregate_path,
                    receipts_root=self.fixture.receipts_root,
                    native_source_pin=self.fixture.pin_path,
                    autocfd5_profile=aggregate_fixture.PROFILE,
                    pilot_case_ids=(aggregate_fixture.CASE_ID,),
                    official_case_ids=(aggregate_fixture.CASE_ID,),
                    expected_pin_sha256=None,
                )

    def test_bound_artifact_tamper_and_output_collisions_fail_closed(self) -> None:
        artifact = self.fixture.case_root / "velocity-cell-mapping-01mm.json"
        artifact.write_bytes(artifact.read_bytes() + b" ")
        with self.assertRaisesRegex(VelocityAtlasError, "changed before row replay"):
            analyse_atlas(self.inputs)

        with self.assertRaisesRegex(VelocityAtlasError, "must not overwrite"):
            _refuse_output_collisions(
                aggregate_path=self.aggregate_path,
                receipts_root=self.fixture.receipts_root,
                native_source_pin=self.fixture.pin_path,
                autocfd5_profile=aggregate_fixture.PROFILE,
                output_pdf=self.fixture.case_root / "atlas.pdf",
                output_invalid_samples=self.fixture.root / "invalid.csv",
                output_manifest=self.fixture.root / "manifest.json",
            )

    @unittest.skipUnless(
        _renderer_is_pinned(),
        f"requires numpy=={NUMPY_VERSION} and matplotlib=={MATPLOTLIB_VERSION}",
    )
    def test_pinned_pdf_is_byte_deterministic_and_renders_every_case(self) -> None:
        first = self.fixture.root / "atlas-a.pdf"
        second = self.fixture.root / "atlas-b.pdf"
        renderer_a = render_atlas_pdf(
            output_path=first, inputs=self.inputs, analysis=self.analysis
        )
        renderer_b = render_atlas_pdf(
            output_path=second, inputs=self.inputs, analysis=self.analysis
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(renderer_a, renderer_b)
        self.assertEqual(renderer_a["page_count"], 6)
        self.assertEqual(
            renderer_a["pdf_sha256"], hashlib.sha256(first.read_bytes()).hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
