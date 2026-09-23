from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reference.drivaerml.dataset_scorer import (
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_FORMAT,
)
from scripts.promote_drivaerml_candidate import (
    preserve_relative_activation_declaration,
)
from scripts.validate_submission import (
    DRIVAERML_OFFICIAL_CASE_REGISTRY_SCHEMA,
    DRIVAERML_RELATIVE_ACTIVATION_GATE_IDS,
    DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA,
    DRIVAERML_RELATIVE_MANIFEST_SCHEMAS,
    DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
    DRIVAERML_RELATIVE_SUPPORT_INDEX_SCHEMA,
    validate_drivaerml_relative_activation_release,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PREFIX = Path("benchmark-specs/drivaerml")
MANIFEST_FILES = {
    "velocity_placement_manifest": (
        "support/relative-v3/manifests/velocity-placement-all484-v1.json"
    ),
    "velocity_mapping_manifest": (
        "support/relative-v3/manifests/velocity-mapping-all484-v1.json"
    ),
    "cp_manifest": "support/relative-v3/manifests/cp-native-support-all484-v3.json",
}


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DrivAerMLActivationReleaseTests(unittest.TestCase):
    def test_promotion_rerun_preserves_verified_pending_release(self) -> None:
        existing = json.loads(
            (ROOT / DATASET_PREFIX / "submission-spec.json").read_text()
        )
        generated = {
            "relative_diagnostics": copy.deepcopy(
                existing["relative_diagnostics"]
            )
        }
        generated_declaration = generated["relative_diagnostics"]
        generated_declaration.pop("activation_release")
        generated_declaration["status"] = "support_pending"
        generated_declaration.pop("closed_reason")

        preserve_relative_activation_declaration(generated, existing)

        for field in (
            "status",
            "profile_format_enabled",
            "closed_reason",
            "activation_release",
        ):
            self.assertEqual(
                generated_declaration[field],
                existing["relative_diagnostics"][field],
            )

    def test_checked_in_pending_release_is_verified_and_fail_closed(self) -> None:
        spec = json.loads(
            (ROOT / DATASET_PREFIX / "submission-spec.json").read_text()
        )
        declaration = spec["relative_diagnostics"]
        errors: list[str] = []
        self.assertTrue(
            validate_drivaerml_relative_activation_release(
                errors.append,
                spec,
                declaration,
                repository_root=ROOT,
                require_active=False,
            ),
            "\n".join(errors),
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            declaration["status"], "support_verified_activation_pending"
        )
        self.assertIs(declaration["profile_format_enabled"], False)

        errors = []
        self.assertFalse(
            validate_drivaerml_relative_activation_release(
                errors.append,
                spec,
                declaration,
                repository_root=ROOT,
                require_active=True,
            )
        )
        self.assertIn("record status is not activated", "\n".join(errors))

    def build_release(self, root: Path, *, active: bool) -> tuple[dict, Path]:
        retained_files = [
            Path("schemas/v1/drivaerml-relative-profile-chunk.schema.json"),
            DATASET_PREFIX / "proposal/native-source-pin.json",
            DATASET_PREFIX / "support/relative-v3/series-support-index.json",
            *(DATASET_PREFIX / value for value in MANIFEST_FILES.values()),
        ]
        for relative in retained_files:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)

        dataset_directory = root / DATASET_PREFIX
        contract = json.loads(
            (ROOT / DATASET_PREFIX / "drivaerml-relative-diagnostics-v3.json").read_text()
        )
        gates = {
            gate: (
                active
                or gate
                in {
                    "all_484_velocity_placement_manifest_bound",
                    "all_484_velocity_mapping_manifest_bound",
                    "all_484_cp_manifest_bound",
                    "immutable_evaluator_revision_bound",
                }
            )
            for gate in DRIVAERML_RELATIVE_ACTIVATION_GATE_IDS
        }
        contract["activation_gates"] = gates
        contract_path = dataset_directory / "drivaerml-relative-diagnostics-v3.json"
        contract_digest = write_json(contract_path, contract)

        schema_file = root / "schemas/v1/drivaerml-relative-profile-chunk.schema.json"
        registry_file = dataset_directory / "proposal/native-source-pin.json"
        index_file = dataset_directory / "support/relative-v3/series-support-index.json"
        bindings: dict[str, dict] = {
            "contract": {
                "file": "drivaerml-relative-diagnostics-v3.json",
                "sha256": contract_digest,
            },
            "profile_chunk_schema": {
                "file": "../../schemas/v1/drivaerml-relative-profile-chunk.schema.json",
                "sha256": file_sha256(schema_file),
            },
            "official_case_registry": {
                "file": "proposal/native-source-pin.json",
                "sha256": file_sha256(registry_file),
                "schema": DRIVAERML_OFFICIAL_CASE_REGISTRY_SCHEMA,
                "case_count": 484,
            },
            "series_support_index": {
                "file": "support/relative-v3/series-support-index.json",
                "sha256": file_sha256(index_file),
                "schema": DRIVAERML_RELATIVE_SUPPORT_INDEX_SCHEMA,
                "case_count": 484,
            },
        }
        for name, relative in MANIFEST_FILES.items():
            path = dataset_directory / relative
            bindings[name] = {
                "file": relative,
                "sha256": file_sha256(path),
                "schema": DRIVAERML_RELATIVE_MANIFEST_SCHEMAS[name],
                "case_count": 484,
            }

        sensitivity: dict[str, object]
        approval: dict[str, object]
        if active:
            evidence = {
                "schema": DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
                "status": "passed",
                "model_checkpoint_count": 3,
                "bootstrap_replicate_count": 10000,
            }
            evidence_file = dataset_directory / "support/relative-v3/sensitivity.json"
            evidence_digest = write_json(evidence_file, evidence)
            sensitivity = {
                **evidence,
                "file": "support/relative-v3/sensitivity.json",
                "sha256": evidence_digest,
            }
            approval = {
                "status": "approved",
                "approved_by": "benchmark-owner",
                "approved_at": "2026-08-26",
                "pull_request_url": (
                    "https://github.com/neilashton/fluidsbench-submission/pull/99"
                ),
            }
        else:
            sensitivity = {
                "status": "pending",
                "file": None,
                "sha256": None,
                "schema": DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
                "model_checkpoint_count": 0,
                "bootstrap_replicate_count": 0,
            }
            approval = {
                "status": "pending",
                "approved_by": None,
                "approved_at": None,
                "pull_request_url": None,
            }

        evaluator_revision = "1" * 40
        record = {
            "schema": DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA,
            "schema_version": 1,
            "release_id": "drivaerml-relative-v3-test",
            "dataset_id": "drivaerml",
            "status": "activated" if active else "support_verified_activation_pending",
            "profile_format_authorized": active,
            "submissions_opened_by_this_record": False,
            "relative_composite_weight": 0.0,
            "bindings": bindings,
            "evaluator": {
                "status": "frozen",
                "repository": (
                    "https://github.com/neilashton/fluidsbench-submission"
                ),
                "git_revision": evaluator_revision,
                "reference_version": "drivaerml-evaluator-v3-candidate",
            },
            "sensitivity_evidence": sensitivity,
            "owner_approval": approval,
            "activation_gates": gates,
        }
        record_path = dataset_directory / "support/relative-v3/activation-release.json"
        record_digest = write_json(record_path, record)
        declaration = {
            "status": (
                "activated"
                if active
                else "support_verified_activation_pending"
            ),
            "profile_format_enabled": active,
            "contract": {
                "id": RELATIVE_PROFILE_CONTRACT_ID,
                "file": "drivaerml-relative-diagnostics-v3.json",
                "sha256": contract_digest,
            },
            "profile_chunk": {
                "format": RELATIVE_PROFILE_FORMAT,
                "schema_file": (
                    "schemas/v1/drivaerml-relative-profile-chunk.schema.json"
                ),
                "schema_sha256": file_sha256(schema_file),
                "series_per_case": 40,
            },
            "activation_release": {
                "file": "support/relative-v3/activation-release.json",
                "sha256": record_digest,
            },
        }
        dataset_spec = {
            "evaluation_reference_version": "drivaerml-evaluator-v3-candidate",
            "scoring_support": {
                "dataset_evaluator_binding": {
                    "status": "frozen" if active else "pending_frozen_release",
                    "repository_url": (
                        "https://github.com/neilashton/fluidsbench-submission"
                    ),
                    "evaluator_reference_version": (
                        "drivaerml-evaluator-v3-candidate"
                    ),
                    "evaluator_code_revision": evaluator_revision if active else None,
                }
            },
            "relative_diagnostics": declaration,
        }
        return dataset_spec, record_path

    def test_complete_active_release_is_content_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, _record_path = self.build_release(root, active=True)
            errors: list[str] = []
            ready = validate_drivaerml_relative_activation_release(
                errors.append,
                spec,
                spec["relative_diagnostics"],
                repository_root=root,
                require_active=True,
            )
            self.assertTrue(ready, "\n".join(errors))
            self.assertEqual(errors, [])

    def test_transferred_repository_approval_url_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            canonical = "https://github.com/FluidsBench/fluidsbench-submission"
            record["evaluator"]["repository"] = canonical
            record["owner_approval"]["pull_request_url"] = canonical + "/pull/99"
            spec["scoring_support"]["dataset_evaluator_binding"]["repository_url"] = canonical
            spec["relative_diagnostics"]["activation_release"]["sha256"] = write_json(
                record_path, record
            )
            errors: list[str] = []
            self.assertTrue(
                validate_drivaerml_relative_activation_release(
                    errors.append, spec, spec["relative_diagnostics"],
                    repository_root=root, require_active=True,
                ),
                "\n".join(errors),
            )
            record["owner_approval"]["pull_request_url"] = (
                "https://github.com/unrelated-owner/fluidsbench-submission/pull/99"
            )
            spec["relative_diagnostics"]["activation_release"]["sha256"] = write_json(
                record_path, record
            )
            errors = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append, spec, spec["relative_diagnostics"],
                    repository_root=root, require_active=True,
                )
            )
            self.assertIn("approving repository PR", "\n".join(errors))

    def test_false_record_digest_and_incomplete_gate_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)

            wrong_digest_spec = copy.deepcopy(spec)
            wrong_digest_spec["relative_diagnostics"]["activation_release"][
                "sha256"
            ] = "f" * 64
            errors: list[str] = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    wrong_digest_spec,
                    wrong_digest_spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=True,
                )
            )
            self.assertIn("does not match the retained record bytes", "\n".join(errors))

            record = json.loads(record_path.read_text())
            record["activation_gates"]["owner_scientific_approval"] = False
            spec["relative_diagnostics"]["activation_release"]["sha256"] = write_json(
                record_path, record
            )
            errors = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    spec,
                    spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=True,
                )
            )
            self.assertIn("every activation gate must be true", "\n".join(errors))

    def test_pending_release_is_offline_verifiable_but_cannot_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=False)
            errors: list[str] = []
            self.assertTrue(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    spec,
                    spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=False,
                ),
                "\n".join(errors),
            )

            bypass_spec = copy.deepcopy(spec)
            bypass_spec["relative_diagnostics"].update(
                {"status": "activated", "profile_format_enabled": True}
            )
            errors = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    bypass_spec,
                    bypass_spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=False,
                )
            )
            self.assertIn(
                "pending release record requires the benchmark declaration",
                "\n".join(errors),
            )

            record = json.loads(record_path.read_text())
            record["profile_format_authorized"] = True
            spec["relative_diagnostics"]["activation_release"]["sha256"] = write_json(
                record_path, record
            )
            errors = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    spec,
                    spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=False,
                )
            )
            self.assertIn("pending release record must not authorize", "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
