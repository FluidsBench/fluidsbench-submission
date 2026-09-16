#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the non-activating HiLiftAeroML evaluator implementation manifest.

The manifest binds source bytes, not Git revisions or approval state.  Its
inventory is deliberately split across the FluidsBench adapter, native
campaign implementation, evaluator core, recipe helpers, and the host-mounted
PhysicsNeMo framework.  Third-party GPU runtime libraries remain a separately
declared, unresolved container boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "hiliftaeroml-evaluator-implementation-manifest-v1"
STATUS = "candidate_source_bytes_bound_revisions_unresolved"
DEFAULT_OUTPUT_RELATIVE = Path(
    "benchmark-specs/hiliftaeroml/evaluator-implementation-manifest.candidate.json"
)
RUNTIME_LOCK_RELATIVE = Path("requirements-hiliftaeroml-evaluator.lock.txt")
MANIFEST_SCHEMA_RELATIVE = Path(
    "schemas/hiliftaeroml/evaluator-implementation-manifest-v1.schema.json"
)
NATIVE_CONTAINER_IMAGE = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/containers/"
    "pytorch-26.04.sqsh"
)
NATIVE_PYTHON_ENVIRONMENT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/"
    "neil-highlifttraining/physicsnemo/examples/cfd/external_aerodynamics/"
    "unified_external_aero_recipe/.venv-hilift-eval"
)


class ImplementationManifestError(RuntimeError):
    """Raised when an inventory input or generated manifest differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ImplementationManifestError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sign(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("content_fingerprint", None)
    result["content_fingerprint"] = canonical_sha256(result)
    return result


def verify_fingerprint(document: Mapping[str, Any]) -> None:
    fingerprint = document.get("content_fingerprint")
    require(
        isinstance(fingerprint, str) and len(fingerprint) == 64,
        "manifest content_fingerprint is absent",
    )
    unsigned = dict(document)
    del unsigned["content_fingerprint"]
    require(
        canonical_sha256(unsigned) == fingerprint,
        "manifest content_fingerprint differs",
    )


def _physicsnemo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if candidate.name == "physicsnemo" and (candidate / "physicsnemo").is_dir():
            return candidate
    raise ImplementationManifestError(
        "cannot infer PhysicsNeMo root; pass --physicsnemo-root"
    )


@dataclass(frozen=True)
class BuildRoots:
    fluidsbench_adapter: Path
    native_submission: Path
    native_evaluator_core: Path
    native_recipe: Path
    physicsnemo_framework: Path

    def as_mapping(self) -> dict[str, Path]:
        return {
            "fluidsbench_adapter": self.fluidsbench_adapter.resolve(),
            "native_submission": self.native_submission.resolve(),
            "native_evaluator_core": self.native_evaluator_core.resolve(),
            "native_recipe": self.native_recipe.resolve(),
            "physicsnemo_framework": self.physicsnemo_framework.resolve(),
        }


@dataclass(frozen=True)
class JsonNormalizationRule:
    selection_kind: str
    excluded_relative_json_pointers: tuple[str, ...]
    array_json_pointer: str | None = None
    match_key: str | None = None
    match_value: str | None = None


@dataclass(frozen=True)
class SourceGroup:
    group_id: str
    root_id: str
    purpose: str
    includes: tuple[str, ...]


_SPEC_LIFECYCLE_POINTERS = (
    "/status",
    "/evaluation_reference_version",
    "/scoring_support/status",
    "/scoring_support/submissions_open",
    "/scoring_support/closed_reason",
    "/scoring_support/candidate_manifest/status",
    "/scoring_support/dataset_evaluator_binding",
    "/scoring_support/owner_decisions_required",
    "/profile_definition/status",
    "/profile_definition/profile_ground_truth",
    "/profile_definition/candidate_dry_run_profile_ground_truth",
    "/profile_definition/activation_rule",
    "/regional_diagnostics/status",
    "/regional_diagnostics/required_for_new_submissions",
    "/regional_diagnostics/activation_gate",
)
_LEADERBOARD_LIFECYCLE_POINTERS = (
    "/submission_count",
    "/updated_at",
    "/revision_count",
    "/scoring_support/status",
    "/scoring_support/submissions_open",
    "/scoring_support/closed_reason",
    "/scoring_support/candidate_manifest/status",
    "/scoring_support/dataset_evaluator_binding",
    "/scoring_support/owner_decisions_required",
)


JSON_NORMALIZATION_RULES: dict[tuple[str, str], JsonNormalizationRule] = {
    (
        "fluidsbench_adapter",
        "benchmark-specs/hiliftaeroml/submission-spec.json",
    ): JsonNormalizationRule(
        selection_kind="document_root",
        excluded_relative_json_pointers=_SPEC_LIFECYCLE_POINTERS,
    ),
    (
        "fluidsbench_adapter",
        "leaderboard/manifest.json",
    ): JsonNormalizationRule(
        selection_kind="unique_array_object",
        array_json_pointer="/datasets",
        match_key="slug",
        match_value="hiliftaeroml",
        excluded_relative_json_pointers=_LEADERBOARD_LIFECYCLE_POINTERS,
    ),
}


SOURCE_GROUPS = (
    SourceGroup(
        "native_submission_execution",
        "native_submission",
        "Exact native surface/volume/profile execution and deterministic case-set aggregation.",
        (
            "inference/exact_surface_loads.py",
            "inference/infer_surface_submission.py",
            "inference/infer_volume_submission.py",
            "inference/recover_geo_LHC039_AoA_10_surface.py",
            "inference/submission_streaming.py",
            "contract/geo_LHC039_AoA_10_normal_underflow_repair_v1.json",
            "audits/geo_LHC039_AoA_10_normal_direction_comparison_v1.json",
            "tools/aggregate_case_set.py",
            "tools/aggregate_full360.py",
            "tools/audit_full360_case.py",
            "tools/build_exact_pitch_moment_weights.py",
            "tools/full360_case_contracts.py",
            "tools/postprocess_full360_candidate.py",
        ),
    ),
    SourceGroup(
        "native_campaign_launchers",
        "native_submission",
        "Slurm command lines and resource/runtime wiring for the exact Full360 execution.",
        (
            "scripts/sbatch_exact_surface_full360_20260902.sbatch",
            "scripts/sbatch_exact_surface_pilot_20260902.sbatch",
            "scripts/sbatch_exact_weight_full360_20260902.sbatch",
            "scripts/sbatch_exact_weight_full360_parallel4_20260902.sbatch",
            "scripts/sbatch_full360_transolver_submission.sbatch",
        ),
    ),
    SourceGroup(
        "native_evaluator_core",
        "native_evaluator_core",
        "Streaming weighted metrics, regional reporting, result persistence, and packaging core.",
        (
            "src/hiliftaeroml_evaluator/*.py",
            "contract/regional-diagnostics.json",
            "pyproject.toml",
        ),
    ),
    SourceGroup(
        "native_recipe_dependencies",
        "native_recipe",
        "Direct and transitive campaign-owned helpers/configuration loaded by native execution.",
        (
            "src/hilift_stl_components.py",
            "src/infer_native_surface.py",
            "src/infer_native_velocity_profiles.py",
            "src/infer_native_volume_fields_profiles.py",
            "src/native_cp_profile_metrics.py",
            "src/native_cp_stencils.py",
            "src/native_point_partition.py",
            "src/native_point_permutation_cache.py",
            "src/native_point_random_partition.py",
            "src/native_surface_auxiliary_point_random.py",
            "src/native_velocity_profile_stencils.py",
            "src/native_vtu_memmap.py",
            "src/profile_quadrature.py",
            "src/weighted_metrics.py",
            "conf/model/transolver_surface.yaml",
            "conf/model/transolver_volume.yaml",
            "datasets/highlift_volume.yaml",
        ),
    ),
    SourceGroup(
        "physicsnemo_host_framework",
        "physicsnemo_framework",
        "Complete host-mounted PhysicsNeMo Python package containing Transolver, mesh/SDF, and checkpoint loading.",
        (
            "physicsnemo/**/*.py",
        ),
    ),
    SourceGroup(
        "fluidsbench_assembler_scorer_validator",
        "fluidsbench_adapter",
        "HiLift package assembly/scoring plus the common fail-closed submission validator.",
        (
            "scripts/assemble_hiliftaeroml_schema_v3_candidate.py",
            "scripts/build_hiliftaeroml_submission_zip.py",
            "scripts/validate_scoring_supports.py",
            "scripts/validate_submission.py",
        ),
    ),
    SourceGroup(
        "fluidsbench_scoring_references",
        "fluidsbench_adapter",
        "Complete local import closure used by HiLift assembly and validation.",
        (
            "reference/__init__.py",
            "reference/metrics.py",
            "reference/methodology.py",
            "reference/scores.py",
            "reference/scoring_support.py",
            "reference/weightings.py",
            "reference/drivaerml/__init__.py",
            "reference/drivaerml/accumulators.py",
            "reference/drivaerml/coordinate_identity.py",
            "reference/drivaerml/dataset_scorer.py",
            "reference/drivaerml/methodology.py",
            "reference/drivaerml/regional_aggregate.py",
            "reference/drivaerml/regional_diagnostics.py",
            "reference/drivaerml/retained_file.py",
            "reference/drivaerml/source.py",
            "reference/hiliftaeroml/__init__.py",
            "reference/hiliftaeroml/native_profile_evaluator.py",
            "reference/hiliftaeroml/native_profile_truth.py",
            "reference/hiliftaeroml/native_profiles.py",
            "reference/hiliftaeroml/regional_aggregate.py",
        ),
    ),
    SourceGroup(
        "fluidsbench_hilift_schemas",
        "fluidsbench_adapter",
        "Every JSON Schema used by HiLift schema-v3 package creation and validation.",
        (
            "schemas/v3/*.schema.json",
            "schemas/v1/profile-index.schema.json",
            "schemas/v1/hiliftaeroml-native-profile-chunk.schema.json",
            "schemas/scoring-support/v1/*.schema.json",
            "schemas/methodology-contract-v1.schema.json",
        ),
    ),
    SourceGroup(
        "fluidsbench_hilift_specification",
        "fluidsbench_adapter",
        "Scoring equations, split definitions, profile/regional contracts, and all candidate support rows.",
        (
            "benchmark-specs/hiliftaeroml/submission-spec.json",
            "benchmark-specs/hiliftaeroml/methodology-contract.json",
            "benchmark-specs/hiliftaeroml/native-profile-format-v1.json",
            "benchmark-specs/hiliftaeroml/native-profile-truth-release-format-v1.json",
            "benchmark-specs/hiliftaeroml/candidate-profile-truth-binding.json",
            "benchmark-specs/hiliftaeroml/regional-diagnostics-v1.json",
            "benchmark-specs/hiliftaeroml/splits/*.json",
            "benchmark-specs/hiliftaeroml/scoring-support/**/*.json",
            "leaderboard/manifest.json",
            "examples/hiliftaeroml-v3-candidate/package-config.template.json",
        ),
    ),
    SourceGroup(
        "freeze_tooling",
        "fluidsbench_adapter",
        "Deterministic manifest builder and its validation schema.",
        (
            "scripts/build_hiliftaeroml_implementation_manifest.py",
            str(MANIFEST_SCHEMA_RELATIVE),
        ),
    ),
)


ROOT_DECLARATIONS: dict[str, dict[str, str]] = {
    "fluidsbench_adapter": {
        "repository": "https://github.com/neilashton/fluidsbench-submission",
        "role": "FluidsBench assembly, scoring, validation, schema, and specification",
        "revision_binding": "pending_candidate_commit",
    },
    "native_submission": {
        "repository": "https://github.com/NVIDIA/physicsnemo",
        "role": "HiLift native execution workspace",
        "revision_binding": "pending_candidate_commit",
    },
    "native_evaluator_core": {
        "repository": "https://github.com/NVIDIA/physicsnemo",
        "role": "HiLift streaming evaluator core",
        "revision_binding": "pending_candidate_commit",
    },
    "native_recipe": {
        "repository": "https://github.com/NVIDIA/physicsnemo",
        "role": "Unified external-aerodynamics native helper implementation",
        "revision_binding": "pending_candidate_commit",
    },
    "physicsnemo_framework": {
        "repository": "https://github.com/NVIDIA/physicsnemo",
        "role": "Host-mounted model, mesh/SDF, and checkpoint framework source",
        "revision_binding": "pending_candidate_commit",
    },
}


def describe_file(root: Path, relative: str) -> dict[str, Any]:
    relative_path = Path(relative)
    require(not relative_path.is_absolute(), f"absolute inventory path: {relative}")
    require(
        ".." not in relative_path.parts,
        f"parent traversal in inventory path: {relative}",
    )
    path = root / relative_path
    require(path.is_file() and not path.is_symlink(), f"missing regular source: {path}")
    return {
        "path": relative_path.as_posix(),
        "identity_mode": "raw_bytes_sha256",
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _json_pointer_parts(pointer: str) -> list[str]:
    require(pointer.startswith("/"), f"JSON pointer is not absolute: {pointer}")
    require(pointer != "/", "root JSON-pointer removal is forbidden")
    return [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer[1:].split("/")
    ]


def _select_pointer(document: Any, pointer: str) -> Any:
    current = document
    for part in _json_pointer_parts(pointer):
        if isinstance(current, Mapping):
            require(part in current, f"JSON selection pointer is absent: {pointer}")
            current = current[part]
        elif isinstance(current, list):
            require(part.isdigit(), f"JSON selection index is invalid: {pointer}")
            index = int(part)
            require(index < len(current), f"JSON selection index is absent: {pointer}")
            current = current[index]
        else:
            raise ImplementationManifestError(
                f"JSON selection traverses a scalar: {pointer}"
            )
    return current


def _remove_pointer(document: Any, pointer: str) -> None:
    parts = _json_pointer_parts(pointer)
    parent = document
    for part in parts[:-1]:
        require(
            isinstance(parent, dict) and part in parent,
            f"normalized lifecycle pointer is absent: {pointer}",
        )
        parent = parent[part]
    leaf = parts[-1]
    require(
        isinstance(parent, dict) and leaf in parent,
        f"normalized lifecycle pointer is absent: {pointer}",
    )
    del parent[leaf]


def describe_normalized_json(
    path: Path,
    relative: str,
    rule: JsonNormalizationRule,
) -> dict[str, Any]:
    relative_path = Path(relative)
    require(not relative_path.is_absolute(), f"absolute inventory path: {relative}")
    require(
        ".." not in relative_path.parts,
        f"parent traversal in inventory path: {relative}",
    )
    require(path.is_file() and not path.is_symlink(), f"missing regular source: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ImplementationManifestError(
            f"invalid normalized JSON source: {path}"
        ) from error

    if rule.selection_kind == "document_root":
        require(
            rule.array_json_pointer is None
            and rule.match_key is None
            and rule.match_value is None,
            f"document-root normalization selector differs: {relative}",
        )
        selected = document
        selection = {"kind": "document_root"}
    elif rule.selection_kind == "unique_array_object":
        require(
            isinstance(rule.array_json_pointer, str)
            and isinstance(rule.match_key, str)
            and isinstance(rule.match_value, str),
            f"array-object normalization selector is incomplete: {relative}",
        )
        candidates = _select_pointer(document, rule.array_json_pointer)
        require(
            isinstance(candidates, list),
            f"normalization selector is not an array: {relative}",
        )
        matches = [
            value
            for value in candidates
            if isinstance(value, dict) and value.get(rule.match_key) == rule.match_value
        ]
        require(
            len(matches) == 1,
            f"normalization selector is not unique: {relative}",
        )
        selected = matches[0]
        selection = {
            "kind": "unique_array_object",
            "array_json_pointer": rule.array_json_pointer,
            "match_key": rule.match_key,
            "match_value": rule.match_value,
        }
    else:
        raise ImplementationManifestError(
            f"unknown JSON normalization selector: {rule.selection_kind}"
        )

    require(
        isinstance(selected, dict),
        f"normalized JSON selection is not an object: {relative}",
    )
    exclusions = sorted(rule.excluded_relative_json_pointers)
    require(
        len(exclusions) == len(set(exclusions)),
        f"duplicate normalized lifecycle pointer: {relative}",
    )
    for pointer in exclusions:
        _remove_pointer(selected, pointer)
    normalized = canonical_bytes(selected)
    return {
        "path": relative_path.as_posix(),
        "identity_mode": "canonical_json_sha256_excluding_lifecycle_v1",
        "selection": selection,
        "excluded_relative_json_pointers": exclusions,
        "canonical_size_bytes": len(normalized),
        "canonical_sha256": hashlib.sha256(normalized).hexdigest(),
    }


def describe_source_file(root_id: str, root: Path, relative: str) -> dict[str, Any]:
    rule = JSON_NORMALIZATION_RULES.get((root_id, Path(relative).as_posix()))
    if rule is None:
        return describe_file(root, relative)
    return describe_normalized_json(root / relative, relative, rule)


def expand_group(group: SourceGroup, root: Path) -> list[dict[str, Any]]:
    relatives: set[str] = set()
    for expression in group.includes:
        if any(character in expression for character in "*?["):
            matches = [path for path in root.glob(expression) if path.is_file()]
            require(
                matches,
                f"inventory pattern matched no files: {group.root_id}:{expression}",
            )
            for path in matches:
                require(not path.is_symlink(), f"symlinked inventory source: {path}")
                relatives.add(path.relative_to(root).as_posix())
        else:
            relatives.add(Path(expression).as_posix())
    return [
        describe_source_file(group.root_id, root, relative)
        for relative in sorted(relatives)
    ]


def build_manifest(roots: BuildRoots) -> dict[str, Any]:
    root_paths = roots.as_mapping()
    require(set(root_paths) == set(ROOT_DECLARATIONS), "source-root inventory differs")
    for root_id, root in root_paths.items():
        require(
            root.is_dir() and not root.is_symlink(),
            f"missing source root {root_id}: {root}",
        )

    groups: list[dict[str, Any]] = []
    flattened: list[dict[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    for specification in SOURCE_GROUPS:
        files = expand_group(specification, root_paths[specification.root_id])
        for file_identity in files:
            key = (specification.root_id, file_identity["path"])
            require(key not in observed, f"source appears in multiple groups: {key}")
            observed.add(key)
            flattened.append(
                {
                    "group_id": specification.group_id,
                    "root_id": specification.root_id,
                    **file_identity,
                }
            )
        groups.append(
            {
                "group_id": specification.group_id,
                "root_id": specification.root_id,
                "purpose": specification.purpose,
                "file_count": len(files),
                "files": files,
                "group_fingerprint": canonical_sha256(files),
            }
        )

    runtime_lock = describe_file(
        root_paths["fluidsbench_adapter"], RUNTIME_LOCK_RELATIVE.as_posix()
    )
    document = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "status": STATUS,
        "activation_effect": "none",
        "dataset_id": "hiliftaeroml",
        "source_roots": ROOT_DECLARATIONS,
        "source_groups": groups,
        "source_group_count": len(groups),
        "source_file_count": len(flattened),
        "source_inventory_fingerprint": canonical_sha256(flattened),
        "source_identity_policy": {
            "default_mode": "raw_bytes_sha256",
            "normalized_json_exception_mode": (
                "canonical_json_sha256_excluding_lifecycle_v1"
            ),
            "normalized_json_exception_count": len(JSON_NORMALIZATION_RULES),
            "purpose": (
                "Prevent the later evaluator-revision metadata commit from "
                "invalidating its own implementation freeze while retaining all "
                "HiLift scientific, split, metric, profile, and scoring-support content."
            ),
        },
        "runtime_boundaries": {
            "fluidsbench_evaluator_python": {
                "status": "exact_versions_locked_from_tested_environment",
                "scope": "assembler_scorer_validator_only",
                "interpreter": {
                    "implementation": "CPython",
                    "version": "3.12.13",
                    "platform_observed": "Linux aarch64 glibc 2.39",
                },
                "install_contract": "fresh isolated environment; install every listed package with --no-deps",
                "artifact_hashes_in_lock": False,
                "lock_file": runtime_lock,
            },
            "native_gpu_container": {
                "status": "external_container_reference_not_content_locked",
                "scope": "native_model_inference_third_party_runtime",
                "image_reference": NATIVE_CONTAINER_IMAGE,
                "image_digest": None,
                "host_python_environment_reference": NATIVE_PYTHON_ENVIRONMENT,
                "host_python_environment_lock": None,
                "host_physicsnemo_sources_hash_bound": True,
                "note": (
                    "Campaign-owned and host-mounted PhysicsNeMo source bytes are in the "
                    "inventory. The container-supplied interpreter/shared libraries and the "
                    "host-mounted native Python environment/site-packages are not locked; "
                    "they require a later container digest plus environment artifact or "
                    "runtime export."
                ),
            },
        },
        "release_state": {
            "git_revisions_bound": False,
            "owner_approval_recorded": False,
            "submissions_opened": False,
            "implementation_manifest_gate_changed": False,
            "runtime_lock_gate_changed": False,
        },
        "excluded_by_design": [
            {
                "path": "benchmark-specs/hiliftaeroml/candidate-evaluator-release-binding.json",
                "reason": "excluded to avoid a manifest-to-release-binding hash cycle",
            },
            {
                "path": (
                    "examples/hiliftaeroml-v3-candidate/"
                    "transolver-full360-candidate-config.json"
                ),
                "reason": (
                    "result-specific example input, including the evaluator Git "
                    "revision fixed after the implementation commit; the final "
                    "postprocess receipt binds its exact bytes separately"
                ),
            },
            {
                "category": "owner_approval_and_activation_records",
                "reason": "governance state is not implementation source and remains pending",
            },
            {
                "category": "checkpoints_datasets_hidden_truth_and_execution_outputs",
                "reason": "bound separately by case/release evidence rather than this source manifest",
            },
            {
                "category": "documentation_and_tests",
                "reason": "not evaluator runtime inputs",
            },
        ],
    }
    return sign(document)


def verify_manifest_files(document: Mapping[str, Any], roots: BuildRoots) -> None:
    verify_fingerprint(document)
    root_paths = roots.as_mapping()
    groups = document.get("source_groups")
    require(isinstance(groups, list), "manifest source_groups is absent")
    flattened: list[dict[str, Any]] = []
    for group in groups:
        require(isinstance(group, Mapping), "invalid source group")
        root_id = group.get("root_id")
        require(root_id in root_paths, f"unknown source root: {root_id}")
        files = group.get("files")
        require(isinstance(files, list), "source-group files are absent")
        require(group.get("file_count") == len(files), "source-group file count differs")
        require(group.get("group_fingerprint") == canonical_sha256(files), "source-group fingerprint differs")
        for declared in files:
            require(isinstance(declared, Mapping), "invalid source identity")
            observed = describe_source_file(
                str(root_id),
                root_paths[str(root_id)],
                str(declared.get("path", "")),
            )
            require(observed == declared, f"source identity differs: {root_id}:{declared.get('path')}")
            flattened.append(
                {
                    "group_id": group.get("group_id"),
                    "root_id": root_id,
                    **observed,
                }
            )
    require(document.get("source_file_count") == len(flattened), "source file count differs")
    require(
        document.get("source_inventory_fingerprint") == canonical_sha256(flattened),
        "source inventory fingerprint differs",
    )
    runtime = document.get("runtime_boundaries", {}).get(
        "fluidsbench_evaluator_python", {}
    )
    expected_lock = describe_file(
        root_paths["fluidsbench_adapter"], RUNTIME_LOCK_RELATIVE.as_posix()
    )
    require(runtime.get("lock_file") == expected_lock, "runtime lock identity differs")


def write_or_check(path: Path, payload: bytes, *, check: bool) -> str:
    path = path.resolve()
    if check:
        require(path.is_file() and not path.is_symlink(), f"manifest is absent: {path}")
        require(path.read_bytes() == payload, f"generated manifest differs: {path}")
        return "validated_existing"
    require(not path.is_symlink(), f"refusing symlink manifest output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def parser() -> argparse.ArgumentParser:
    script_root = Path(__file__).resolve().parents[1]
    workspace = script_root.parent
    physicsnemo = _physicsnemo_root(script_root)
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fluidsbench-root", type=Path, default=script_root)
    result.add_argument(
        "--native-submission-root",
        type=Path,
        default=workspace / "hiliftaeroml-transolver-submission-v1",
    )
    result.add_argument(
        "--native-evaluator-core-root",
        type=Path,
        default=workspace / "hiliftaeroml-evaluator-v0.1",
    )
    result.add_argument(
        "--native-recipe-root",
        type=Path,
        default=(
            physicsnemo
            / "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
        ),
    )
    result.add_argument("--physicsnemo-root", type=Path, default=physicsnemo)
    result.add_argument("--output", type=Path, default=script_root / DEFAULT_OUTPUT_RELATIVE)
    result.add_argument("--check", action="store_true")
    return result


def roots_from_args(args: argparse.Namespace) -> BuildRoots:
    return BuildRoots(
        fluidsbench_adapter=args.fluidsbench_root,
        native_submission=args.native_submission_root,
        native_evaluator_core=args.native_evaluator_core_root,
        native_recipe=args.native_recipe_root,
        physicsnemo_framework=args.physicsnemo_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    roots = roots_from_args(args)
    manifest = build_manifest(roots)
    verify_fingerprint(manifest)
    payload = pretty_bytes(manifest)
    mode = write_or_check(args.output, payload, check=args.check)
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": mode,
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "content_fingerprint": manifest["content_fingerprint"],
                "source_group_count": manifest["source_group_count"],
                "source_file_count": manifest["source_file_count"],
                "source_inventory_fingerprint": manifest[
                    "source_inventory_fingerprint"
                ],
                "activation_effect": "none",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImplementationManifestError as error:
        raise SystemExit(f"ERROR: {error}") from error
