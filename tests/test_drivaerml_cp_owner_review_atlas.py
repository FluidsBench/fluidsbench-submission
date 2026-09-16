from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_drivaerml_cp_support_aggregate as aggregate_fixture  # noqa: E402
from scripts.aggregate_drivaerml_cp_support import (  # noqa: E402
    DEFAULT_AUTOCFD5_PROFILE,
    write_evidence,
)
from scripts.build_drivaerml_cp_owner_review_atlas import (  # noqa: E402
    ATLAS_ACTIVATION_STATUS,
    ATLAS_SCHEMA,
    ATLAS_STATUS,
    MATPLOTLIB_VERSION,
    NUMPY_VERSION,
    CpAtlasError,
    _canonical_json_bytes,
    _refuse_output_collisions,
    _row_category,
    analyse_atlas,
    build_manifest,
    load_atlas_inputs,
    render_atlas_pdf,
)


def _renderer_is_pinned() -> bool:
    previous_config = os.environ.get("MPLCONFIGDIR")
    try:
        with tempfile.TemporaryDirectory(prefix="drivaerml-cp-atlas-test-mpl-") as config:
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


class DrivAerMLCpOwnerReviewAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = aggregate_fixture.DrivAerMLCpSupportAggregateTests(
            "test_complete_aggregate_and_separate_inventory_are_deterministic"
        )
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.aggregate, _ = cls.fixture._aggregate(
            list(reversed(cls.fixture.receipt_pairs))
        )
        cls.aggregate_path = cls.fixture.root / "cp-aggregate.json"
        write_evidence(cls.aggregate_path, cls.aggregate)
        cls.model = load_atlas_inputs(
            aggregate_path=cls.aggregate_path,
            profile_path=DEFAULT_AUTOCFD5_PROFILE,
            receipt_paths=[
                pair[0] for pair in reversed(cls.fixture.receipt_pairs)
            ],
            official_case_ids=aggregate_fixture.CASE_IDS,
        )
        cls.analysis = analyse_atlas(cls.model)

    def test_exact_coverage_order_categories_and_duplicate_flags(self) -> None:
        self.assertEqual(self.model.case_ids, aggregate_fixture.CASE_IDS)
        self.assertEqual(len(self.analysis.probe_ids), 209)
        self.assertEqual(len(self.model.definition.cp_panel_memberships), 217)
        panel_counts: dict[int, int] = {}
        for membership in self.model.definition.cp_panel_memberships:
            panel_counts[membership.autocfd_probe_id] = (
                panel_counts.get(membership.autocfd_probe_id, 0) + 1
            )
        self.assertEqual(
            {probe_id for probe_id, count in panel_counts.items() if count == 2},
            {38, 39, 40, 42, 133, 198, 260, 263},
        )
        self.assertEqual(set(panel_counts), set(self.analysis.probe_ids))
        self.assertEqual(
            sum(len(row) for row in self.analysis.category_matrix), 2 * 209
        )
        self.assertEqual(len(self.analysis.invalid_rows), 4)
        self.assertEqual(
            {row["category"] for row in self.analysis.invalid_rows},
            {
                "mapping:declared_component_absent",
                "truth:nonfinite_native_pMeanTrim",
            },
        )
        self.assertGreater(len(self.analysis.duplicate_groups), 0)
        self.assertTrue(
            all(len(group["probe_ids"]) >= 2 for group in self.analysis.duplicate_groups)
        )

    def test_review_flag_is_distinct_and_does_not_override_invalidity(self) -> None:
        valid_review = {
            "mapping_valid": True,
            "truth_valid": True,
            "review_flags": ["native_bridge_distance_gt_0p5mm"],
        }
        invalid_review = {
            "mapping_valid": False,
            "mapping_reason": "native_bridge_distance_exceeds_2mm",
            "truth_valid": False,
            "truth_reason": "mapping_invalid",
            "review_flags": ["native_bridge_distance_gt_0p5mm"],
        }
        self.assertEqual(
            _row_category(valid_review),
            "automated_valid_review_flag_pending_owner_review",
        )
        self.assertEqual(
            _row_category(invalid_review),
            "mapping:native_bridge_distance_exceeds_2mm",
        )

    def test_manifest_is_compact_path_free_and_truthfully_candidate(self) -> None:
        renderer = {
            "matplotlib": MATPLOTLIB_VERSION,
            "numpy": NUMPY_VERSION,
            "bundled_dejavu_sans_sha256": "a" * 64,
            "page_count": 6,
            "pdf_sha256": "b" * 64,
            "pdf_size_bytes": 12345,
        }
        first = build_manifest(
            inputs=self.model, analysis=self.analysis, renderer=renderer
        )
        second = build_manifest(
            inputs=self.model, analysis=self.analysis, renderer=copy.deepcopy(renderer)
        )
        self.assertEqual(first["schema"], ATLAS_SCHEMA)
        self.assertEqual(first["status"], ATLAS_STATUS)
        self.assertFalse(first["owner_visual_signoff_claimed"])
        self.assertFalse(first["scientific_approval_claimed"])
        self.assertEqual(first["activation_status"], ATLAS_ACTIVATION_STATUS)
        self.assertFalse(first["source"]["source_geometry_rendered"])
        self.assertEqual(first["scope"]["retained_row_count"], 418)
        self.assertEqual(first["scope"]["expected_row_count"], 418)
        self.assertEqual(first["artifact"]["page_count"], 6)
        self.assertEqual(_canonical_json_bytes(first), _canonical_json_bytes(second))
        serialized = _canonical_json_bytes(first).decode("utf-8")
        self.assertNotIn(str(self.fixture.root), serialized)
        self.assertNotIn("/tmp/", serialized)

    def test_explicit_pilot_remains_incomplete_and_non_activating(self) -> None:
        aggregate, _ = self.fixture._aggregate(
            [self.fixture.receipt_pairs[0]], pilot_case_ids=("run_1",)
        )
        aggregate_path = self.fixture.root / "pilot-aggregate.json"
        write_evidence(aggregate_path, aggregate)
        model = load_atlas_inputs(
            aggregate_path=aggregate_path,
            profile_path=DEFAULT_AUTOCFD5_PROFILE,
            receipt_paths=[self.fixture.receipt_pairs[0][0]],
            pilot_case_ids=("run_1",),
            official_case_ids=aggregate_fixture.CASE_IDS,
        )
        analysis = analyse_atlas(model)
        renderer = {
            "matplotlib": MATPLOTLIB_VERSION,
            "numpy": NUMPY_VERSION,
            "bundled_dejavu_sans_sha256": "a" * 64,
            "page_count": 5,
            "pdf_sha256": "b" * 64,
            "pdf_size_bytes": 123,
        }
        manifest = build_manifest(inputs=model, analysis=analysis, renderer=renderer)
        self.assertEqual(manifest["mode"], "partial_pilot")
        self.assertEqual(manifest["scope"]["case_ids"], ["run_1"])
        self.assertFalse(manifest["owner_visual_signoff_claimed"])
        self.assertEqual(manifest["activation_status"], ATLAS_ACTIVATION_STATUS)
        with self.assertRaisesRegex(CpAtlasError, "mode/status"):
            load_atlas_inputs(
                aggregate_path=aggregate_path,
                profile_path=DEFAULT_AUTOCFD5_PROFILE,
                receipt_paths=[self.fixture.receipt_pairs[0][0]],
                official_case_ids=aggregate_fixture.CASE_IDS,
            )

    def test_receipt_tamper_and_output_collision_fail_closed(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tampered = root / "run_1.json"
        shutil.copyfile(self.fixture.receipt_pairs[0][0], tampered)
        value = json.loads(tampered.read_text(encoding="utf-8"))
        value["rows"][0]["nominal_point_m"][0] += 1.0e-6
        tampered.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CpAtlasError, "receipt identity differs"):
            load_atlas_inputs(
                aggregate_path=self.aggregate_path,
                profile_path=DEFAULT_AUTOCFD5_PROFILE,
                receipt_paths=[tampered, self.fixture.receipt_pairs[1][0]],
                official_case_ids=aggregate_fixture.CASE_IDS,
            )
        with self.assertRaisesRegex(CpAtlasError, "must not overwrite"):
            _refuse_output_collisions(
                aggregate_path=self.aggregate_path,
                profile_path=DEFAULT_AUTOCFD5_PROFILE,
                receipt_paths=[pair[0] for pair in self.fixture.receipt_pairs],
                output_pdf=self.fixture.receipt_pairs[0][0],
                output_manifest=root / "manifest.json",
            )

    @unittest.skipUnless(
        _renderer_is_pinned(),
        f"requires numpy=={NUMPY_VERSION} and matplotlib=={MATPLOTLIB_VERSION}",
    )
    def test_pinned_pdf_and_manifest_are_byte_deterministic(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pdf_a = root / "atlas-a.pdf"
        pdf_b = root / "atlas-b.pdf"
        renderer_a = render_atlas_pdf(
            output_path=pdf_a, inputs=self.model, analysis=self.analysis
        )
        renderer_b = render_atlas_pdf(
            output_path=pdf_b, inputs=self.model, analysis=self.analysis
        )
        self.assertEqual(pdf_a.read_bytes(), pdf_b.read_bytes())
        self.assertEqual(renderer_a, renderer_b)
        self.assertEqual(renderer_a["page_count"], 6)
        self.assertEqual(
            renderer_a["pdf_sha256"], hashlib.sha256(pdf_a.read_bytes()).hexdigest()
        )
        manifest_a = build_manifest(
            inputs=self.model, analysis=self.analysis, renderer=renderer_a
        )
        manifest_b = build_manifest(
            inputs=self.model, analysis=self.analysis, renderer=renderer_b
        )
        self.assertEqual(_canonical_json_bytes(manifest_a), _canonical_json_bytes(manifest_b))


if __name__ == "__main__":
    unittest.main()
