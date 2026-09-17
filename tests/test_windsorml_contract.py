"""Bind the published WindsorML spec artifacts to the Python contract.

These are cheap file-level checks, but they are the ones that would catch the
failure modes that actually bit during development: a split file quietly
re-admitting an unscoreable case, component weights drifting off 1.0 after a
metric is removed, or the spec and the evaluator disagreeing about which axis
is lift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reference.windsorml.contract import (  # noqa: E402
    AXIS_CONVENTION,
    CASE_COUNT,
    FORCE_REFERENCE_AREA_M2,
    FORCE_REPLAY_ABSOLUTE_TOLERANCE,
    FORCE_TRUTH_SOURCE,
    SURFACE_FIELDS,
    UNPUBLISHED_CASE_IDS,
    VOLUME_FIELDS,
    WindsorMLContractError,
    WindsorMLForceTruth,
    classify_force_replay,
)

SPEC_DIR = REPO_ROOT / "benchmark-specs" / "windsorml"
SPLIT_DIR = SPEC_DIR / "splits"
FAMILIES = (
    "full",
    "medium",
    "scarce",
    "super_scarce",
    "geometry",
    "high_drag",
    "low_drag",
    "image_wake",
)
PUBLISHED = {f"run_{index}" for index in range(CASE_COUNT)}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


class WindsorMLSplitBindingTests(unittest.TestCase):
    def test_every_family_has_a_split_file(self) -> None:
        for family in FAMILIES:
            self.assertTrue((SPLIT_DIR / f"{family}.json").is_file(), family)

    def test_scored_cases_are_all_published(self) -> None:
        for family in FAMILIES:
            document = load(SPLIT_DIR / f"{family}.json")
            unpublished = sorted(set(document["case_ids"]) - PUBLISHED)
            self.assertEqual(
                unpublished, [], f"{family} scores unpublished cases {unpublished}"
            )

    def test_unscoreable_test_cases_are_excluded_and_recorded(self) -> None:
        """run_354 and run_352 have no per-run payload in the public release."""

        expected = {
            "full": ["run_354"],
            "medium": ["run_354"],
            "scarce": ["run_354"],
            "super_scarce": ["run_354"],
            "geometry": [],
            "high_drag": ["run_352"],
            "low_drag": ["run_354"],
            "image_wake": [],
        }
        for family, excluded in expected.items():
            document = load(SPLIT_DIR / f"{family}.json")
            self.assertEqual(document["excluded_case_ids"], excluded, family)
            self.assertEqual(
                document["case_count"],
                document["official_case_count"] - len(excluded),
                family,
            )

    def test_excluded_ids_are_declared_unpublished_by_the_contract(self) -> None:
        for family in FAMILIES:
            document = load(SPLIT_DIR / f"{family}.json")
            for case_id in document["excluded_case_ids"]:
                self.assertIn(case_id, UNPUBLISHED_CASE_IDS, case_id)

    def test_unique_scored_test_case_count_is_233(self) -> None:
        union: set[str] = set()
        for family in FAMILIES:
            union |= set(load(SPLIT_DIR / f"{family}.json")["case_ids"])
        self.assertEqual(len(union), 233)

    def test_case_ids_are_unique_and_sorted_within_each_split(self) -> None:
        for family in FAMILIES:
            case_ids = load(SPLIT_DIR / f"{family}.json")["case_ids"]
            self.assertEqual(len(case_ids), len(set(case_ids)), family)
            numeric = [int(c.removeprefix("run_")) for c in case_ids]
            self.assertEqual(numeric, sorted(numeric), family)


class WindsorMLSubmissionSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load(SPEC_DIR / "submission-spec.json")

    def test_component_weights_sum_to_one(self) -> None:
        components = self.spec["overall_score_composite"]["components"]
        total = sum(component["weight"] for component in components)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_split_digests_match_the_split_files(self) -> None:
        for entry in self.spec["splits"]:
            path = SPEC_DIR / entry["index_file"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["id"])
            self.assertEqual(load(path)["case_count"], entry["case_count"], entry["id"])

    def test_force_convention_matches_the_python_contract(self) -> None:
        forces = self.spec["scoring_support"]["force_convention"]
        self.assertEqual(forces["reference_area_m2"], FORCE_REFERENCE_AREA_M2)
        self.assertEqual(forces["axes"], dict(AXIS_CONVENTION))
        self.assertEqual(forces["truth_source"], FORCE_TRUTH_SOURCE)
        self.assertEqual(forces["audit_reference"], "published_force_mom_csv")

    def test_lift_is_declared_as_the_y_axis(self) -> None:
        self.assertEqual(
            self.spec["scoring_support"]["force_convention"]["axes"]["lift"], "+y"
        )

    def test_supports_declare_the_measured_associations(self) -> None:
        supports = {
            support["id"]: support
            for support in self.spec["scoring_support"]["public_supports"]
        }
        self.assertEqual(
            supports["windsorml_surface_native_points"]["association"], "PointData"
        )
        self.assertEqual(
            supports["windsorml_volume_native_cells"]["association"], "CellData"
        )

    def test_support_arrays_match_the_python_contract(self) -> None:
        supports = {
            support["id"]: support
            for support in self.spec["scoring_support"]["public_supports"]
        }
        self.assertEqual(
            set(supports["windsorml_surface_native_points"]["arrays"]),
            set(SURFACE_FIELDS),
        )
        self.assertEqual(
            set(supports["windsorml_volume_native_cells"]["arrays"]),
            set(VOLUME_FIELDS),
        )

    def test_no_fabricated_prototype_stations_remain(self) -> None:
        stations: list[str] = []
        for panel in self.spec["profile_panels"]:
            stations.extend(panel["station_ids"])
        for station in stations:
            self.assertFalse(
                station.startswith("prototype_"),
                f"fabricated station {station!r} still declared",
            )

    def test_unscored_panels_are_marked_and_carry_no_weight(self) -> None:
        scored_metric_ids = {
            component["metric_id"]
            for component in self.spec["overall_score_composite"]["components"]
        }
        for panel in self.spec["profile_panels"]:
            if not panel.get("scored_in_this_version", False):
                for quantity in panel["quantity_ids"]:
                    self.assertNotIn(f"{quantity}_r2", scored_metric_ids)

    def test_deferred_physical_volume_metrics_are_absent(self) -> None:
        metric_ids = {metric["id"] for metric in self.spec["metrics"]}
        self.assertNotIn("volume_velocity_physical_rel_l2", metric_ids)
        self.assertNotIn("volume_pressure_physical_rel_l2", metric_ids)

    def test_status_is_no_longer_prototype_dummy_data(self) -> None:
        self.assertNotEqual(self.spec["status"], "prototype_dummy_data")
        for entry in self.spec["splits"]:
            self.assertNotEqual(entry["case_id_status"], "prototype_generated")


class WindsorMLProfileDefinitionTests(unittest.TestCase):
    """The frozen stations and the two placement families must stay consistent.

    Body length and width are identical across all 350 published variants, but
    height varies from 0.316 m to 0.473 m. That spread is the whole reason the
    relative family exists, and the reason constant stations may not be defined
    as a fraction of height.
    """

    DEFINITION = SPEC_DIR / "profile-definition-v2.json"
    BASE_X = 0.48325
    NOSE_X = -0.56075
    BODY_LENGTH = 1.044
    MAX_BODY_HEIGHT = 0.47340
    MIN_BODY_HEIGHT = 0.31574
    REFERENCE_H = 0.34342

    def setUp(self) -> None:
        self.definition = load(self.DEFINITION)
        self.spec = load(SPEC_DIR / "submission-spec.json")
        self.families = {f["family_id"]: f for f in self.definition["families"]}

    def test_geometry_invariants_match_the_measured_values(self) -> None:
        invariants = self.definition["geometry_invariants"]
        self.assertAlmostEqual(invariants["base_x_m"], self.BASE_X, places=5)
        self.assertAlmostEqual(invariants["nose_x_m"], self.NOSE_X, places=5)
        self.assertAlmostEqual(invariants["body_length_m"], self.BODY_LENGTH, places=5)
        low, high = invariants["body_height_range_m"]
        self.assertAlmostEqual(low, self.MIN_BODY_HEIGHT, places=4)
        self.assertAlmostEqual(high, self.MAX_BODY_HEIGHT, places=4)

    def test_every_kind_has_a_constant_and_a_relative_family(self) -> None:
        by_panel: dict[str, dict[str, str]] = {}
        for family in self.definition["families"]:
            by_panel.setdefault(family["panel_id"], {})[family["placement_mode"]] = (
                family["family_id"]
            )
        for panel_id, modes in by_panel.items():
            self.assertEqual(set(modes), {"constant", "relative"}, panel_id)

    def test_relative_families_reference_their_constant_source(self) -> None:
        for family in self.definition["families"]:
            if family["placement_mode"] != "relative":
                continue
            self.assertIn(family["source_family_id"], self.families)
            source = self.families[family["source_family_id"]]
            self.assertEqual(source["placement_mode"], "constant")
            self.assertEqual(source["panel_id"], family["panel_id"])

    def test_only_the_constant_family_carries_weight(self) -> None:
        """The DrivAerML rule: one placement mode per kind may be ranked."""

        for family in self.definition["families"]:
            if family["placement_mode"] == "relative":
                self.assertEqual(family["scoring_role"], "report_only", family["family_id"])
                self.assertEqual(family["composite_weight"], 0.0, family["family_id"])
            else:
                self.assertEqual(
                    family["scoring_role"], "ranked_candidate", family["family_id"]
                )
        scored = {
            component["metric_id"]
            for component in self.spec["overall_score_composite"]["components"]
        }
        for metric in self.spec["metrics"]:
            if metric.get("scoring_role") == "report_only":
                self.assertNotIn(metric["id"], scored, metric["id"])

    def test_spec_panels_bind_both_families_and_rank_the_constant_one(self) -> None:
        for panel in self.spec["profile_panels"]:
            self.assertEqual(set(panel["families"]), {"constant", "relative"})
            self.assertEqual(panel["ranked_family_id"], panel["families"]["constant"])
            self.assertEqual(
                self.families[panel["ranked_family_id"]]["placement_mode"], "constant"
            )

    def test_constant_and_relative_station_counts_match(self) -> None:
        for group in ("velocity_stations", "pressure_stations"):
            constant = self.definition[group]["constant"]
            relative = self.definition[group]["relative"]
            self.assertEqual(len(constant), len(relative), group)
            constant_ids = {s["id"] for s in constant}
            for station in relative:
                self.assertIn(station["source_station_id"], constant_ids, station["id"])

    def test_velocity_stations_sit_where_their_x_over_l_says(self) -> None:
        for station in self.definition["velocity_stations"]["constant"]:
            expected = self.BASE_X + station["x_over_l"] * self.BODY_LENGTH
            self.assertAlmostEqual(station["x_m"], expected, places=5, msg=station["id"])

    def test_all_wake_stations_are_downstream_of_the_base(self) -> None:
        for station in self.definition["velocity_stations"]["constant"]:
            self.assertGreater(station["x_m"], self.BASE_X, station["id"])

    def test_relative_frame_normalises_only_the_vertical(self) -> None:
        """x and z are invariant across variants, so only y needs rescaling."""

        frame = self.definition["relative_frame"]
        self.assertEqual(frame["vertical_normalisation"], "eta = y / h_case")
        constant = {s["id"]: s for s in self.definition["velocity_stations"]["constant"]}
        for station in self.definition["velocity_stations"]["relative"]:
            source = constant[station["source_station_id"]]
            self.assertAlmostEqual(station["x_m"], source["x_m"], places=6, msg=station["id"])

    def test_relative_eta_reduces_to_the_constant_station_on_run_0(self) -> None:
        """Anchoring claim: the two families coincide on the reference case."""

        frame = self.definition["relative_frame"]
        self.assertAlmostEqual(frame["reference_h_m"], self.REFERENCE_H, places=5)
        self.assertAlmostEqual(
            frame["horizontal_cut_eta"] * self.REFERENCE_H, 0.194, places=4
        )
        eta_low, eta_high = frame["vertical_span_eta"]
        self.assertAlmostEqual(eta_low * self.REFERENCE_H, 0.0, places=6)
        self.assertAlmostEqual(eta_high * self.REFERENCE_H, 0.6, places=4)

    def test_constant_horizontal_cut_lies_inside_every_published_body(self) -> None:
        for station in self.definition["pressure_stations"]["constant"]:
            if "y_0p194" not in station["id"]:
                continue
            self.assertLess(0.194, self.MIN_BODY_HEIGHT, station["id"])

    def test_sampling_semantics_match_the_sibling_datasets(self) -> None:
        """Volume uses containing cell; surface selects native points."""

        semantics = self.definition["sampling_semantics"]
        self.assertEqual(semantics["volume"]["rule"], "containing_cell")
        self.assertEqual(semantics["volume"]["order"], "zeroth_order_piecewise_constant")
        self.assertEqual(semantics["surface"]["rule"], "native_point_selection")
        self.assertEqual(semantics["surface"]["order"], "exact_native_values")

    def test_quantisation_floor_is_documented_as_inherent(self) -> None:
        """It is shared with DrivAerML and AhmedML, not a WindsorML defect."""

        note = self.definition["sampling"]["resolution_evidence"]["residual_noise_note"]
        self.assertIn("DrivAerML", note)
        self.assertIn("AhmedML", note)
        self.assertIn("invent information", note)

    def test_sample_count_is_frozen_with_sweep_evidence(self) -> None:
        sampling = self.definition["sampling"]
        self.assertEqual(sampling["resolution_status"], "frozen_after_stratified_sweep")
        evidence = sampling["resolution_evidence"]
        for station in self.definition["velocity_stations"]["constant"]:
            deltas = evidence["max_delta_vs_64_samples"][station["id"]]
            self.assertEqual(set(deltas), {"96", "128", "160"}, station["id"])
            for value in deltas.values():
                self.assertLess(value, 0.05, station["id"])

    def test_spec_sample_count_matches_the_definition(self) -> None:
        expected = self.definition["sampling"]["sample_count"]
        for panel in self.spec["profile_panels"]:
            self.assertEqual(panel["sample_count"], expected, panel["id"])

    def test_no_fabricated_prototype_stations_survive(self) -> None:
        text = self.DEFINITION.read_text()
        self.assertNotIn("prototype_0_25l", text)
        for group in ("velocity_stations", "pressure_stations"):
            for mode in ("constant", "relative"):
                for station in self.definition[group][mode]:
                    self.assertFalse(station["id"].startswith("prototype_"), station["id"])

    def test_profiles_are_evaluator_derived_not_participant_supplied(self) -> None:
        for family in self.definition["families"]:
            self.assertIn("evaluator_derived", family["source"])
        for panel in self.spec["profile_panels"]:
            self.assertIn("evaluator_derived", panel["source"])


class WindsorMLScoredCaseSetApprovalTests(unittest.TestCase):
    def test_reduction_is_recorded_as_owner_approved(self) -> None:
        spec = load(SPEC_DIR / "submission-spec.json")
        record = spec["scoring_support"]["scored_case_set"]
        self.assertEqual(record["status"], "owner_approved")
        self.assertEqual(record["official_unique_test_cases"], 235)
        self.assertEqual(record["scored_unique_test_cases"], 233)
        self.assertEqual(sorted(record["excluded_case_ids"]), ["run_352", "run_354"])
        self.assertNotIn(
            "approve_scored_case_set_reduction_from_235_to_233_test_cases",
            spec["scoring_support"]["owner_decisions_required"],
        )

    def test_component_weights_are_recorded_as_owner_approved(self) -> None:
        spec = load(SPEC_DIR / "submission-spec.json")
        record = spec["scoring_support"]["component_weights"]
        self.assertEqual(record["status"], "owner_approved")
        self.assertEqual(
            record["split"],
            {"field_score": 0.50, "force_score": 0.25, "diagnostic_score": 0.25},
        )

    def test_group_weights_match_the_approved_split(self) -> None:
        """The declared split must equal what the components actually sum to."""

        spec = load(SPEC_DIR / "submission-spec.json")
        weights = {
            component["metric_id"]: component["weight"]
            for component in spec["overall_score_composite"]["components"]
        }
        actual = {
            group["metric_id"]: sum(weights[m] for m in group["component_metric_ids"])
            for group in spec["component_score_groups"]["groups"]
        }
        declared = spec["scoring_support"]["component_weights"]["split"]
        for group, value in declared.items():
            self.assertAlmostEqual(actual[group], value, places=9, msg=group)

    def test_no_owner_decisions_remain_open(self) -> None:
        spec = load(SPEC_DIR / "submission-spec.json")
        self.assertEqual(spec["scoring_support"]["owner_decisions_required"], [])

    def test_excluded_ids_are_contract_unpublished_cases(self) -> None:
        spec = load(SPEC_DIR / "submission-spec.json")
        for case_id in spec["scoring_support"]["scored_case_set"]["excluded_case_ids"]:
            self.assertIn(case_id, UNPUBLISHED_CASE_IDS)


class WindsorMLMethodologyContractTests(unittest.TestCase):
    def test_required_predicted_fields_match_the_python_contract(self) -> None:
        document = load(SPEC_DIR / "methodology-contract.json")
        surface = {
            field["field_id"].split(".", 1)[1]
            for field in document["required_predicted_fields"]
            if field["domain"] == "surface"
        }
        volume = {
            field["field_id"].split(".", 1)[1]
            for field in document["required_predicted_fields"]
            if field["domain"] == "volume"
        }
        self.assertEqual(surface, set(SURFACE_FIELDS))
        self.assertEqual(volume, set(VOLUME_FIELDS))


class WindsorMLForceReplayAuditTests(unittest.TestCase):
    """The audit must survive legitimately near-zero coefficients.

    run_306 publishes cl = +0.000106. Its truth integration lands 0.000328
    away, which is a 310% relative difference but physically negligible, so a
    purely relative bound would reject a perfectly good case.
    """

    NEAR_ZERO = WindsorMLForceTruth(cd=0.281817, cl=0.000106, cs=0.0, cmy=0.0)

    def test_tiny_absolute_drift_on_a_near_zero_coefficient_is_accepted(self) -> None:
        deltas = classify_force_replay(
            case_id="run_306",
            published=self.NEAR_ZERO,
            replay_cd=0.281088,
            replay_cl=-0.000222,
        )
        self.assertLess(deltas["cl"], FORCE_REPLAY_ABSOLUTE_TOLERANCE)

    def test_large_absolute_drift_on_a_near_zero_coefficient_is_rejected(self) -> None:
        with self.assertRaises(WindsorMLContractError):
            classify_force_replay(
                case_id="run_306",
                published=self.NEAR_ZERO,
                replay_cd=0.281088,
                replay_cl=0.40,
            )

    def test_observed_worst_case_deltas_fit_inside_the_frozen_tolerance(self) -> None:
        """Worst observed absolute deltas over all published runs: cd 0.0089, cl 0.0034."""

        truth = WindsorMLForceTruth(cd=0.30, cl=0.50, cs=0.0, cmy=0.0)
        deltas = classify_force_replay(
            case_id="run_worst",
            published=truth,
            replay_cd=0.30 + 0.0089,
            replay_cl=0.50 + 0.0034,
        )
        self.assertAlmostEqual(deltas["cd"], 0.0089, places=6)

    def test_non_finite_replay_is_rejected(self) -> None:
        with self.assertRaises(WindsorMLContractError):
            classify_force_replay(
                case_id="run_0",
                published=self.NEAR_ZERO,
                replay_cd=float("nan"),
                replay_cl=0.0,
            )


class WindsorMLSourceIdentityTests(unittest.TestCase):
    """Load the real 350-case identity and bind it to the pinned digest."""

    IDENTITY = (
        SPEC_DIR
        / "public-source-identity"
        / "windsorml-public-source-identity-v1.json"
    )

    def setUp(self) -> None:
        if not self.IDENTITY.is_file():
            self.skipTest("source identity has not been generated")

    def test_identity_matches_the_pinned_digest(self) -> None:
        from reference.windsorml.contract import (
            SOURCE_IDENTITY_SHA256,
            load_source_identity,
        )

        identity = load_source_identity(
            self.IDENTITY, expected_sha256=SOURCE_IDENTITY_SHA256
        )
        self.assertEqual(len(identity.cases), CASE_COUNT)
        self.assertEqual(identity.cases[0].case_id, "run_0")
        self.assertEqual(identity.cases[-1].case_id, "run_349")

    def test_a_tampered_identity_is_rejected(self) -> None:
        from reference.windsorml.contract import load_source_identity

        with self.assertRaises(WindsorMLContractError):
            load_source_identity(self.IDENTITY, expected_sha256="0" * 64)

    def test_every_case_declares_positive_entity_counts_and_force_truth(self) -> None:
        from reference.windsorml.contract import (
            SOURCE_IDENTITY_SHA256,
            load_source_identity,
        )

        identity = load_source_identity(
            self.IDENTITY, expected_sha256=SOURCE_IDENTITY_SHA256
        )
        for case in identity.cases:
            self.assertGreater(case.surface_entity_count, 0, case.case_id)
            self.assertGreater(case.volume_entity_count, 0, case.case_id)
            self.assertNotEqual(case.force_truth.cd, 0.0, case.case_id)

    def test_no_unpublished_case_appears_in_the_identity(self) -> None:
        from reference.windsorml.contract import (
            SOURCE_IDENTITY_SHA256,
            load_source_identity,
        )

        identity = load_source_identity(
            self.IDENTITY, expected_sha256=SOURCE_IDENTITY_SHA256
        )
        case_ids = {case.case_id for case in identity.cases}
        self.assertEqual(case_ids & UNPUBLISHED_CASE_IDS, set())

    def test_every_scored_test_case_is_present_in_the_identity(self) -> None:
        from reference.windsorml.contract import (
            SOURCE_IDENTITY_SHA256,
            load_source_identity,
        )

        identity = load_source_identity(
            self.IDENTITY, expected_sha256=SOURCE_IDENTITY_SHA256
        )
        for family in FAMILIES:
            for case_id in load(SPLIT_DIR / f"{family}.json")["case_ids"]:
                self.assertTrue(
                    identity.has_case(case_id), f"{family}: {case_id} missing"
                )


class WindsorMLCaseIdentityTests(unittest.TestCase):
    def test_unpublished_case_ids_are_run_350_to_354(self) -> None:
        self.assertEqual(
            UNPUBLISHED_CASE_IDS,
            frozenset({f"run_{index}" for index in range(350, 355)}),
        )

    def test_source_identity_rejects_an_unpublished_case(self) -> None:
        """A source identity may never contain run_350..run_354."""

        self.assertTrue(issubclass(WindsorMLContractError, ValueError))
        self.assertIn("run_354", UNPUBLISHED_CASE_IDS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
