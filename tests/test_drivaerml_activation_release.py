from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.dataset_scorer import (
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_FORMAT,
)
from scripts.promote_drivaerml_candidate import (
    preserve_relative_activation_declaration,
)
from scripts.validate_submission import (
    DRIVAERML_CONSTANT_SUPPORT_INDEX_SCHEMA,
    DRIVAERML_OFFICIAL_CASE_REGISTRY_SCHEMA,
    DRIVAERML_RELATIVE_ACTIVATION_GATE_IDS,
    DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA,
    DRIVAERML_RELATIVE_MANIFEST_SCHEMAS,
    DRIVAERML_RELATIVE_OWNER_APPROVAL_SCHEMA,
    DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
    DRIVAERML_RELATIVE_SUPPORT_INDEX_SCHEMA,
    DRIVAERML_SENSITIVITY_PREDICTION_MANIFEST_SCHEMA,
    DRIVAERML_TRAINED_CHECKPOINT_PROVENANCE_SCHEMA,
    canonical_json_sha256,
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


def run_git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


class DrivAerMLActivationReleaseTests(unittest.TestCase):
    def validate_synthetic_active_fixture(
        self,
        errors: list[str],
        spec: dict,
        record_path: Path,
        root: Path,
    ) -> bool:
        """Validate gate behavior with the test-only activated contract.

        Production validation pins the checked-in pending candidate contract.
        The active fixture necessarily changes that contract's gates, so only
        this unit-test helper substitutes the resulting synthetic contract
        identity while exercising the rest of the activation chain.
        """

        record = json.loads(record_path.read_text())
        synthetic_contract_digest = record["bindings"]["contract"]["sha256"]
        with mock.patch(
            "scripts.validate_submission.RELATIVE_PROFILE_CONTRACT_SHA256",
            synthetic_contract_digest,
        ):
            return validate_drivaerml_relative_activation_release(
                errors.append,
                spec,
                spec["relative_diagnostics"],
                repository_root=root,
                require_active=True,
            )

    def errors_for_index_mutation(self, mutate: object) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=False)
            dataset_directory = root / DATASET_PREFIX
            index_file = (
                dataset_directory
                / "support/relative-v3/series-support-index.json"
            )
            index = json.loads(index_file.read_text())
            mutate(index)  # type: ignore[operator]
            record = json.loads(record_path.read_text())
            record["bindings"]["series_support_index"]["sha256"] = write_json(
                index_file,
                index,
            )
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )
            errors: list[str] = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    spec,
                    spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=False,
                )
            )
            return errors

    def errors_for_sensitivity_mutation(self, mutate: object) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            evidence_file = (
                root
                / DATASET_PREFIX
                / record["sensitivity_evidence"]["file"]
            )
            evidence = json.loads(evidence_file.read_text())
            mutate(evidence)  # type: ignore[operator]
            record["sensitivity_evidence"]["sha256"] = write_json(
                evidence_file,
                evidence,
            )
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )
            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            return errors

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

    def test_v2_coordinate_index_fields_are_fail_closed(self) -> None:
        def first_materialized(index: dict) -> dict:
            return next(
                item
                for item in index["cases"][0]["series"]
                if item["representation"] == "materialized"
            )

        def first_alias(index: dict) -> dict:
            return next(
                item
                for item in index["cases"][0]["series"]
                if item["representation"] == "shared_alias"
            )

        mutations = {
            "null cases array": (
                lambda index: index.__setitem__("cases", None),
                "series_support_index.cases must be an array",
            ),
            "missing coordinate digest": (
                lambda index: first_materialized(index).pop(
                    "coordinate_identity_sha256"
                ),
                "series 0 fields are invalid",
            ),
            "zero coordinate digest": (
                lambda index: first_materialized(index).__setitem__(
                    "coordinate_identity_sha256", "0" * 64
                ),
                "has an invalid coordinate_identity_sha256",
            ),
            "boolean coordinate count": (
                lambda index: first_materialized(index).__setitem__(
                    "coordinate_count", True
                ),
                "has an invalid coordinate_count",
            ),
            "alias coordinate fields": (
                lambda index: first_alias(index).update(
                    {
                        "coordinate_count": 2,
                        "coordinate_identity_sha256": hashlib.sha256(
                            b"alias-must-not-materialize"
                        ).hexdigest(),
                    }
                ),
                "fields are invalid",
            ),
        }
        for label, (mutate, expected) in mutations.items():
            with self.subTest(label=label):
                errors = self.errors_for_index_mutation(mutate)
                self.assertIn(expected, "\n".join(errors))

    def test_candidate_contract_and_schema_bytes_are_independently_pinned(self) -> None:
        mutations = {
            "contract": (
                "drivaerml-relative-diagnostics-v3.json",
                lambda document: document.__setitem__("drift_marker", True),
                "record contract digest differs from the independently pinned",
            ),
            "profile schema": (
                "../../schemas/v1/drivaerml-relative-profile-chunk.schema.json",
                lambda document: document.__setitem__(
                    "description", "unreviewed schema drift"
                ),
                "record profile schema digest differs from the independently pinned",
            ),
        }
        for label, (relative, mutate, expected) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec, record_path = self.build_release(root, active=False)
                record = json.loads(record_path.read_text())
                binding_name = (
                    "contract" if label == "contract" else "profile_chunk_schema"
                )
                artifact_path = (
                    root / DATASET_PREFIX / Path(relative)
                ).resolve()
                document = json.loads(artifact_path.read_text())
                mutate(document)
                digest = write_json(artifact_path, document)
                record["bindings"][binding_name]["sha256"] = digest
                if binding_name == "contract":
                    spec["relative_diagnostics"]["contract"]["sha256"] = digest
                else:
                    spec["relative_diagnostics"]["profile_chunk"][
                        "schema_sha256"
                    ] = digest
                spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                    write_json(record_path, record)
                )

                errors: list[str] = []
                self.assertFalse(
                    validate_drivaerml_relative_activation_release(
                        errors.append,
                        spec,
                        spec["relative_diagnostics"],
                        repository_root=root,
                        require_active=False,
                    )
                )
                self.assertIn(expected, "\n".join(errors))

    def test_null_contract_shared_support_groups_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=False)
            contract_path = (
                root / DATASET_PREFIX / "drivaerml-relative-diagnostics-v3.json"
            )
            contract = json.loads(contract_path.read_text())
            contract["shared_support_groups"] = None
            contract_digest = write_json(contract_path, contract)
            record = json.loads(record_path.read_text())
            record["bindings"]["contract"]["sha256"] = contract_digest
            spec["relative_diagnostics"]["contract"]["sha256"] = contract_digest
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                validate_drivaerml_relative_activation_release(
                    errors.append,
                    spec,
                    spec["relative_diagnostics"],
                    repository_root=root,
                    require_active=False,
                )
            )
            self.assertIn(
                "contract shared_support_groups must be an array",
                "\n".join(errors),
            )

    def build_release(self, root: Path, *, active: bool) -> tuple[dict, Path]:
        retained_files = [
            Path("schemas/v1/drivaerml-relative-profile-chunk.schema.json"),
            DATASET_PREFIX / "proposal/native-source-pin.json",
            DATASET_PREFIX / "support/relative-v3/series-support-index.json",
            DATASET_PREFIX / "support/relative-v3/constant-series-support-index.json",
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
        contract_path = dataset_directory / "drivaerml-relative-diagnostics-v3.json"
        if active:
            contract["activation_gates"] = gates
            contract_digest = write_json(contract_path, contract)
        else:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                ROOT / DATASET_PREFIX / "drivaerml-relative-diagnostics-v3.json",
                contract_path,
            )
            contract_digest = file_sha256(contract_path)

        schema_file = root / "schemas/v1/drivaerml-relative-profile-chunk.schema.json"
        registry_file = dataset_directory / "proposal/native-source-pin.json"
        index_file = dataset_directory / "support/relative-v3/series-support-index.json"
        constant_index_file = (
            dataset_directory
            / "support/relative-v3/constant-series-support-index.json"
        )
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
            "constant_series_support_index": {
                "file": "support/relative-v3/constant-series-support-index.json",
                "sha256": file_sha256(constant_index_file),
                "schema": DRIVAERML_CONSTANT_SUPPORT_INDEX_SCHEMA,
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

        run_git(root, "init", "-q")
        run_git(root, "config", "user.name", "Activation Test")
        run_git(root, "config", "user.email", "activation-test@example.invalid")
        evaluator_marker = root / "evaluator-commit.json"
        write_json(
            evaluator_marker,
            {"reference_version": "drivaerml-evaluator-v3-candidate"},
        )
        run_git(root, "add", "evaluator-commit.json")
        run_git(root, "commit", "-q", "-m", "Retain evaluator test commit")
        evaluator_revision = run_git(root, "rev-parse", "HEAD")

        sensitivity: dict[str, object]
        approval: dict[str, object]
        if active:
            registry = json.loads(registry_file.read_text())
            source_revision = registry["repository"]["revision"]
            sensitivity_case_ids = [
                case["case_id"] for case in registry["cases"][:2]
            ]
            case_scope = {
                "case_count": len(sensitivity_case_ids),
                "case_ids": sensitivity_case_ids,
                "case_ids_sha256": canonical_json_sha256(
                    sensitivity_case_ids
                ),
            }
            checkpoints: list[dict[str, object]] = []
            retained_sensitivity_files: list[Path] = []
            checkpoint_names = ("alpha", "bravo", "charlie")
            for checkpoint_position, checkpoint_name in enumerate(
                checkpoint_names
            ):
                checkpoint_id = f"transolver-campaign-{checkpoint_name}"
                checkpoint_revision = hashlib.sha256(
                    f"immutable-provider-revision-{checkpoint_name}".encode()
                ).hexdigest()[:40]
                checkpoint_sha256 = hashlib.sha256(
                    f"trained-checkpoint-bytes-{checkpoint_name}".encode()
                ).hexdigest()
                checkpoint_repository = (
                    "https://huggingface.co/neashton/"
                    f"drivaerml-transolver-{checkpoint_name}"
                )
                checkpoint_path = (
                    f"checkpoints/campaign-{checkpoint_name}/model.mdlus"
                )
                training_run_id = f"campaign-2026/run-{checkpoint_name}"
                provenance = {
                    "schema": DRIVAERML_TRAINED_CHECKPOINT_PROVENANCE_SCHEMA,
                    "schema_version": 1,
                    "dataset_id": "drivaerml",
                    "dataset_revision": source_revision,
                    "artifact_kind": "trained_model_checkpoint",
                    "checkpoint_id": checkpoint_id,
                    "model_family": "transolver",
                    "training_run_id": training_run_id,
                    "checkpoint_repository": checkpoint_repository,
                    "checkpoint_revision": checkpoint_revision,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_completed": True,
                    "training_case_count": 400 + checkpoint_position,
                    "optimizer_step_count": 12000 + checkpoint_position,
                }
                provenance_file = (
                    dataset_directory
                    / "support/relative-v3/sensitivity/checkpoints"
                    / f"{checkpoint_id}-provenance.json"
                )
                provenance_digest = write_json(provenance_file, provenance)

                prediction_cases = [
                    {
                        "case_id": case_id,
                        "surface_prediction_sha256": hashlib.sha256(
                            f"{checkpoint_id}/{case_id}/surface".encode()
                        ).hexdigest(),
                        "volume_prediction_sha256": hashlib.sha256(
                            f"{checkpoint_id}/{case_id}/volume".encode()
                        ).hexdigest(),
                    }
                    for case_id in sensitivity_case_ids
                ]
                prediction_manifest = {
                    "schema": (
                        DRIVAERML_SENSITIVITY_PREDICTION_MANIFEST_SCHEMA
                    ),
                    "schema_version": 1,
                    "dataset_id": "drivaerml",
                    "dataset_revision": source_revision,
                    "evaluator": {
                        "repository": (
                            "https://github.com/neilashton/"
                            "fluidsbench-submission"
                        ),
                        "git_revision": evaluator_revision,
                        "reference_version": (
                            "drivaerml-evaluator-v3-candidate"
                        ),
                    },
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "case_scope": case_scope,
                    "cases": prediction_cases,
                }
                prediction_file = (
                    dataset_directory
                    / "support/relative-v3/sensitivity/predictions"
                    / f"{checkpoint_id}-manifest.json"
                )
                prediction_digest = write_json(
                    prediction_file,
                    prediction_manifest,
                )
                checkpoints.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "model_family": "transolver",
                        "training_run_id": training_run_id,
                        "artifact_kind": "trained_model_checkpoint",
                        "checkpoint_repository": checkpoint_repository,
                        "checkpoint_revision": checkpoint_revision,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_provenance_file": provenance_file.relative_to(
                            dataset_directory
                        ).as_posix(),
                        "checkpoint_provenance_sha256": provenance_digest,
                        "prediction_manifest_file": prediction_file.relative_to(
                            dataset_directory
                        ).as_posix(),
                        "prediction_manifest_sha256": prediction_digest,
                    }
                )
                retained_sensitivity_files.extend(
                    (provenance_file, prediction_file)
                )
            evidence = {
                "schema": DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
                "schema_version": 1,
                "status": "passed",
                "dataset_revision": source_revision,
                "evaluator": {
                    "repository": (
                        "https://github.com/neilashton/fluidsbench-submission"
                    ),
                    "git_revision": evaluator_revision,
                    "reference_version": "drivaerml-evaluator-v3-candidate",
                },
                "case_scope": case_scope,
                "model_checkpoint_count": 3,
                "checkpoints": checkpoints,
                "bootstrap_replicate_count": 10000,
                "paired_bootstrap": {
                    "paired": True,
                    "replicate_count": 10000,
                    "random_seed": 419,
                    "resampling_unit": "official_case_id",
                    "comparisons": [
                        {
                            "left_checkpoint_id": checkpoints[0]["checkpoint_id"],
                            "right_checkpoint_id": checkpoints[1]["checkpoint_id"],
                            "metric_id": "relative_diagnostics_composite",
                            "observed_delta": 0.1,
                            "confidence_interval_95": [-0.2, 0.4],
                            "replicate_count": 10000,
                        },
                        {
                            "left_checkpoint_id": checkpoints[1]["checkpoint_id"],
                            "right_checkpoint_id": checkpoints[2]["checkpoint_id"],
                            "metric_id": "relative_diagnostics_composite",
                            "observed_delta": -0.05,
                            "confidence_interval_95": [-0.3, 0.2],
                            "replicate_count": 10000,
                        },
                    ],
                },
                "conclusion": {
                    "outcome": "passed",
                    "summary": "Three distinct trained checkpoints passed review.",
                },
            }
            evidence_file = dataset_directory / "support/relative-v3/sensitivity.json"
            evidence_digest = write_json(evidence_file, evidence)
            sensitivity = {
                "status": "passed",
                "file": "support/relative-v3/sensitivity.json",
                "sha256": evidence_digest,
                "schema": DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
                "model_checkpoint_count": 3,
                "bootstrap_replicate_count": 10000,
            }
            approval_document = {
                "schema": DRIVAERML_RELATIVE_OWNER_APPROVAL_SCHEMA,
                "schema_version": 1,
                "activation_release_schema": (
                    DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA
                ),
                "activation_release_schema_version": 1,
                "dataset_id": "drivaerml",
                "release_id": "drivaerml-relative-v3-test",
                "status": "approved",
                "approved_by": "neilashton",
                "approved_at": "2026-08-26",
                "contract_sha256": contract_digest,
                "evaluator_git_revision": evaluator_revision,
                "release_bindings_sha256": canonical_json_sha256(bindings),
                "sensitivity_evidence_sha256": evidence_digest,
            }
            approval_file = (
                dataset_directory
                / "support/relative-v3/owner-approval.json"
            )
            approval_digest = write_json(approval_file, approval_document)
            for retained_file in (
                *retained_sensitivity_files,
                evidence_file,
                approval_file,
            ):
                run_git(
                    root,
                    "add",
                    retained_file.relative_to(root).as_posix(),
                )
            run_git(root, "commit", "-q", "-m", "Retain owner approval record")
            approval_revision = run_git(root, "rev-parse", "HEAD")
            run_git(
                root,
                "remote",
                "add",
                "origin",
                "https://github.com/neilashton/fluidsbench-submission.git",
            )
            run_git(
                root,
                "update-ref",
                "refs/remotes/origin/dev",
                approval_revision,
            )
            approval = {
                "status": "approved",
                "approved_by": "neilashton",
                "approved_at": "2026-08-26",
                "approval_record": {
                    "file": "support/relative-v3/owner-approval.json",
                    "sha256": approval_digest,
                    "schema": DRIVAERML_RELATIVE_OWNER_APPROVAL_SCHEMA,
                    "git_revision": approval_revision,
                },
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
            spec, record_path = self.build_release(root, active=True)
            errors: list[str] = []
            ready = self.validate_synthetic_active_fixture(
                errors,
                spec,
                record_path,
                root,
            )
            self.assertTrue(ready, "\n".join(errors))
            self.assertEqual(errors, [])

    def test_owner_approval_binds_complete_release_binding_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            index_path = (
                root
                / DATASET_PREFIX
                / "support/relative-v3/series-support-index.json"
            )
            index = json.loads(index_path.read_text())
            materialized = next(
                item
                for item in index["cases"][0]["series"]
                if item["representation"] == "materialized"
            )
            materialized["coordinate_identity_sha256"] = hashlib.sha256(
                b"attacker-chosen-coordinate-array"
            ).hexdigest()
            materialized["support_identity_sha256"] = hashlib.sha256(
                b"attacker-chosen-support"
            ).hexdigest()

            record = json.loads(record_path.read_text())
            record["bindings"]["series_support_index"]["sha256"] = write_json(
                index_path,
                index,
            )
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "owner approval record does not exactly bind this release",
                "\n".join(errors),
            )

    def test_feature_commit_cannot_claim_owner_approval(self) -> None:
        """A PR-authored commit is not a trusted owner/merge record."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            marker = root / "feature-branch-marker.json"
            write_json(marker, {"claim": "self-authored approval"})
            run_git(root, "add", marker.relative_to(root).as_posix())
            run_git(root, "commit", "-q", "-m", "Unreviewed feature commit")
            feature_revision = run_git(root, "rev-parse", "HEAD")

            record = json.loads(record_path.read_text())
            record["owner_approval"]["approval_record"][
                "git_revision"
            ] = feature_revision
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "must already be retained on fetched origin/dev",
                "\n".join(errors),
            )

    def test_owner_approval_commit_must_retain_reviewed_evidence_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            dataset_directory = root / DATASET_PREFIX
            original = dataset_directory / record["sensitivity_evidence"]["file"]
            uncommitted_copy = (
                dataset_directory
                / "support/relative-v3/unreviewed/sensitivity.json"
            )
            uncommitted_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, uncommitted_copy)
            record["sensitivity_evidence"]["file"] = uncommitted_copy.relative_to(
                dataset_directory
            ).as_posix()
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "sensitivity evidence bytes were not retained by the owner approval commit",
                "\n".join(errors),
            )

    def test_checkpoint_locators_must_be_distinct(self) -> None:
        def duplicate_locator(evidence: dict) -> None:
            first = evidence["checkpoints"][0]
            second = evidence["checkpoints"][1]
            for field in (
                "checkpoint_repository",
                "checkpoint_revision",
                "checkpoint_path",
            ):
                second[field] = first[field]

        errors = self.errors_for_sensitivity_mutation(duplicate_locator)
        self.assertIn(
            "immutable checkpoint locators must be distinct",
            "\n".join(errors),
        )

    def test_owner_identity_is_independently_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            record["owner_approval"]["approved_by"] = "plausible-contributor"
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )
            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "approved_by is not an independently pinned benchmark owner",
                "\n".join(errors),
            )

    def test_nonexistent_evaluator_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            nonexistent_revision = "f" * 40
            record["evaluator"]["git_revision"] = nonexistent_revision
            spec["scoring_support"]["dataset_evaluator_binding"][
                "evaluator_code_revision"
            ] = nonexistent_revision
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "evaluator git_revision is not a commit reachable",
                "\n".join(errors),
            )

    def test_skeletal_sensitivity_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            sensitivity_file = (
                root
                / DATASET_PREFIX
                / record["sensitivity_evidence"]["file"]
            )
            skeletal = {
                "schema": DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA,
                "status": "passed",
                "model_checkpoint_count": 3,
                "bootstrap_replicate_count": 10000,
            }
            record["sensitivity_evidence"]["sha256"] = write_json(
                sensitivity_file,
                skeletal,
            )
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "sensitivity evidence must implement the complete v1 schema",
                "\n".join(errors),
            )

    def test_placeholder_checkpoint_metadata_is_rejected(self) -> None:
        mutations = {
            "repeated revision": (
                lambda evidence: evidence["checkpoints"][0].__setitem__(
                    "checkpoint_revision", "1" * 40
                ),
                "checkpoint_revision looks placeholder-shaped",
            ),
            "fake repository": (
                lambda evidence: evidence["checkpoints"][0].__setitem__(
                    "checkpoint_repository",
                    "https://example.invalid/test/model",
                ),
                "checkpoint_repository looks placeholder-shaped",
            ),
            "dummy path": (
                lambda evidence: evidence["checkpoints"][0].__setitem__(
                    "checkpoint_path", "checkpoints/dummy-model.pt"
                ),
                "checkpoint_path looks placeholder-shaped",
            ),
        }
        for label, (mutate, expected) in mutations.items():
            with self.subTest(label=label):
                errors = self.errors_for_sensitivity_mutation(mutate)
                self.assertIn(expected, "\n".join(errors))

    def test_sensitivity_requires_non_degenerate_bound_prediction_cohort(self) -> None:
        def one_case(evidence: dict) -> None:
            case_ids = ["run_419"]
            evidence["case_scope"] = {
                "case_count": 1,
                "case_ids": case_ids,
                "case_ids_sha256": canonical_json_sha256(case_ids),
            }

        errors = self.errors_for_sensitivity_mutation(one_case)
        self.assertIn(
            "sensitivity evidence must use at least two official cases",
            "\n".join(errors),
        )

        def wrong_prediction_manifest_digest(evidence: dict) -> None:
            evidence["checkpoints"][0]["prediction_manifest_sha256"] = (
                hashlib.sha256(b"different-retained-prediction-manifest").hexdigest()
            )

        errors = self.errors_for_sensitivity_mutation(
            wrong_prediction_manifest_digest
        )
        self.assertIn(
            "prediction_manifest_sha256 does not match the retained bytes",
            "\n".join(errors),
        )

    def test_pr_shaped_owner_metadata_without_committed_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, record_path = self.build_release(root, active=True)
            record = json.loads(record_path.read_text())
            record["owner_approval"] = {
                "status": "approved",
                "approved_by": "neilashton",
                "approved_at": "2026-08-26",
                "pull_request_url": (
                    "https://github.com/neilashton/fluidsbench-submission/pull/999999"
                ),
            }
            spec["relative_diagnostics"]["activation_release"]["sha256"] = (
                write_json(record_path, record)
            )

            errors: list[str] = []
            self.assertFalse(
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
                )
            )
            self.assertIn(
                "approved owner_approval must bind a committed approval record",
                "\n".join(errors),
            )

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
                self.validate_synthetic_active_fixture(
                    errors,
                    wrong_digest_spec,
                    record_path,
                    root,
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
                self.validate_synthetic_active_fixture(
                    errors,
                    spec,
                    record_path,
                    root,
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
