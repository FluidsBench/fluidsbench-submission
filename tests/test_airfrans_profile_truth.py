from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.airfrans.profile_truth import (
    CaseSetSource,
    ProfileTruthError,
    build_release,
    load_profile_definition,
    resolve_case_set_truth,
    runtime_versions,
    validate_release,
)

PROFILE_DEFINITION = ROOT / "benchmark-specs" / "airfrans" / "velocity-profiles-v1.json"
EXTRACTOR_SHA = "e" * 64
SOURCE = {
    "dataset_release": "AirfRANS NeurIPS 2022 public release",
    "archive_url": "https://example.org/Dataset.zip",
    "archive_size_bytes": 1,
    "archive_last_modified": "2022-08-11T15:01:09Z",
    "manifest_member": "Dataset/manifest.json",
    "manifest_sha256": "d" * 64,
}


def truth_document(case_id: str, *, scale: float = 1.0, mode: str = "ground_truth_reference") -> dict:
    definition = load_profile_definition(PROFILE_DEFINITION)
    samples = definition.sample_count
    return {
        "schema_version": "1.0",
        "provenance": {
            "mode": mode,
            "profile_definition_id": definition.profile_definition_id,
            "extractor_sha256": EXTRACTOR_SHA,
            "versions": runtime_versions(definition),
            "airfrans_runtime_source_file_sha256": definition.sampling_runtime[
                "airfrans_runtime_source_file_sha256"
            ],
            "input_hashes": {"internal_vtu": "a" * 64, "aerofoil_vtp": "b" * 64},
            "perfect_copy_validation": {"absolute_tolerance": 1e-12, "maximum_absolute_difference": 0.0},
        },
        "cases": [
            {
                "case_id": case_id,
                "series": [
                    {
                        "panel_id": definition.panel_id,
                        "station_id": station_id,
                        "quantity_id": quantity_id,
                        "coordinate": [index / (samples - 1) for index in range(samples)],
                        "prediction": [scale * index / samples for index in range(samples)],
                    }
                    for station_id in definition.station_ids
                    for quantity_id in definition.quantity_ids
                ],
            }
        ],
    }


class AirfransProfileTruthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.repo = self.base / "repo"
        splits = self.repo / "benchmark-specs" / "airfrans" / "splits"
        splits.mkdir(parents=True)
        shutil.copy(PROFILE_DEFINITION, self.repo / "benchmark-specs" / "airfrans" / PROFILE_DEFINITION.name)
        self.profile_definition = self.repo / "benchmark-specs" / "airfrans" / PROFILE_DEFINITION.name
        # Two splits in non-sorted order sharing case "c2", as Full and AoA share 41 cases.
        self.split_cases = {"full": ["c3", "c1", "c2"], "aoa": ["c4", "c2"]}
        self.sources = []
        for split_id, case_set_id in (("full", "standard"), ("aoa", "aoa_extrapolation")):
            split = {"split_id": split_id, "case_set_id": case_set_id, "source": SOURCE,
                     "case_ids": self.split_cases[split_id]}
            (splits / f"{split_id}.json").write_text(json.dumps(split), encoding="utf-8")
            truth_dir = self.base / f"truth-{split_id}"
            truth_dir.mkdir()
            for case_id in self.split_cases[split_id]:
                (truth_dir / f"{case_id}.json").write_text(json.dumps(truth_document(case_id)), encoding="utf-8")
            self.sources.append(CaseSetSource(split_file=splits / f"{split_id}.json", truth_dir=truth_dir))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, output: Path, **overrides) -> dict:
        arguments = {
            "sources": self.sources,
            "profile_definition_path": self.profile_definition,
            "repository_root": self.repo,
            "extractor_commit": "f" * 40,
            "extractor_commit_file_sha256": EXTRACTOR_SHA,
            "output_root": output,
        }
        arguments.update(overrides)
        return build_release(**arguments)

    def validate(self, release: Path) -> dict:
        return validate_release(
            release_root=release, profile_definition_path=self.profile_definition, repository_root=self.repo
        )

    def test_build_deduplicates_and_validates(self) -> None:
        release = self.base / "release"
        manifest = self.build(release)
        self.assertEqual(manifest["case_count"], 4)
        self.assertEqual(
            {entry["case_set_id"]: entry["case_count"] for entry in manifest["case_sets"]},
            {"standard": 3, "aoa_extrapolation": 2},
        )
        self.assertEqual(sorted(path.name for path in (release / "cases").iterdir()),
                         ["c1.json", "c2.json", "c3.json", "c4.json"])
        summary = self.validate(release)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["case_sets"], {"standard": 3, "aoa_extrapolation": 2})
        receipt = json.loads((release / "release-receipt.json").read_text())
        self.assertEqual(receipt["manifest_sha256"], hashlib.sha256((release / "manifest.json").read_bytes()).hexdigest())

    def test_truth_files_are_copied_byte_for_byte(self) -> None:
        release = self.base / "release"
        self.build(release)
        source = self.sources[0].truth_dir / "c3.json"
        self.assertEqual((release / "cases" / "c3.json").read_bytes(), source.read_bytes())

    def test_case_set_resolves_in_official_order(self) -> None:
        release = self.base / "release"
        self.build(release)
        resolved = resolve_case_set_truth(release, "standard")
        self.assertEqual([path.stem for path in resolved], ["c3", "c1", "c2"])

    def test_packaging_is_deterministic(self) -> None:
        first, second = self.base / "first", self.base / "second"
        self.build(first)
        self.build(second)
        self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())
        self.build(first, check=True)

    def test_check_mode_detects_a_changed_byte(self) -> None:
        release = self.base / "release"
        self.build(release)
        record = release / "case-records" / "c1.json"
        record.write_bytes(record.read_bytes().replace(b'"c1"', b'"c9"', 1))
        with self.assertRaisesRegex(ProfileTruthError, "determinism check differs"):
            self.build(release, check=True)

    def test_refuses_to_overwrite(self) -> None:
        release = self.base / "release"
        self.build(release)
        with self.assertRaisesRegex(ProfileTruthError, "refusing to overwrite"):
            self.build(release)

    def test_refuses_shared_case_with_different_truth(self) -> None:
        (self.sources[1].truth_dir / "c2.json").write_text(
            json.dumps(truth_document("c2", scale=2.0)), encoding="utf-8"
        )
        with self.assertRaisesRegex(ProfileTruthError, "truth files differ"):
            self.build(self.base / "release")

    def test_refuses_prediction_mode_truth(self) -> None:
        (self.sources[0].truth_dir / "c1.json").write_text(
            json.dumps(truth_document("c1", mode="prediction")), encoding="utf-8"
        )
        with self.assertRaisesRegex(ProfileTruthError, "provenance.mode"):
            self.build(self.base / "release")

    def test_refuses_short_series(self) -> None:
        document = truth_document("c1")
        document["cases"][0]["series"][0]["prediction"].pop()
        (self.sources[0].truth_dir / "c1.json").write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ProfileTruthError, "finite samples"):
            self.build(self.base / "release")

    def test_refuses_mismatched_extractor_commit(self) -> None:
        with self.assertRaisesRegex(ProfileTruthError, "does not match the extractor"):
            self.build(self.base / "release", extractor_commit_file_sha256="0" * 64)

    def test_validation_detects_tampered_truth(self) -> None:
        release = self.base / "release"
        self.build(release)
        truth = release / "cases" / "c4.json"
        truth.write_text(json.dumps(truth_document("c4", scale=3.0)), encoding="utf-8")
        with self.assertRaisesRegex(ProfileTruthError, "c4 truth artifact"):
            self.validate(release)

    def test_validation_detects_extra_file(self) -> None:
        release = self.base / "release"
        self.build(release)
        (release / "cases" / "stray.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ProfileTruthError, "inventory differs"):
            self.validate(release)

    def test_validation_detects_changed_split(self) -> None:
        release = self.base / "release"
        self.build(release)
        split_path = self.sources[0].split_file
        split = json.loads(split_path.read_text())
        split["case_ids"] = ["c1", "c3", "c2"]
        split_path.write_text(json.dumps(split), encoding="utf-8")
        with self.assertRaisesRegex(ProfileTruthError, "official split file differs"):
            self.validate(release)


if __name__ == "__main__":
    unittest.main()
