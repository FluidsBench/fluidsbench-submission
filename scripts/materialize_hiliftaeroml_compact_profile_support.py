#!/usr/bin/env python3
"""Materialize one inactive evaluator-owned compact-profile support release.

Support can be reconstructed either from validated native evaluator outputs or
directly, without predictions, from the frozen prerequisite authority used to
build the native-profile truth release.  Ground truth is always joined
exclusively from that separately bound truth-v1 release.  Output creation is
exclusive and the canonical manifest is written last by the reference
evaluator release writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.compact_profile_evaluator import (  # noqa: E402
    COMPACT_PROFILE_CONTRACT_ID,
    COMPACT_PROFILE_CONTRACT_SHA256,
    COMPACT_SUPPORT_RELEASE_ID,
    COMPACT_SUPPORT_RELEASE_STATUS,
    COMPACT_SUPPORT_RELEASE_USAGE,
    CompactProfileEvaluationError,
    open_compact_support_release,
    write_compact_support_release,
)
from reference.hiliftaeroml.compact_profiles import (  # noqa: E402
    COMPACT_PROFILE_FORMAT,
    CP_POINTS_PER_GRAPH,
    CompactProfileError,
    build_compact_support,
)
from reference.hiliftaeroml.native_profile_evaluator import (  # noqa: E402
    CandidateTruthRelease,
    NativeProfileEvaluationError,
    _truth_case,
    _validate_cp_alignment,
    _validate_velocity_alignment,
    open_candidate_truth_release,
)
from reference.hiliftaeroml.native_profile_truth import (  # noqa: E402
    CaseUniverse,
    TRUTH_FORMAT,
    NativeProfileTruthError,
)
from reference.hiliftaeroml.native_profile_truth_materializer import (  # noqa: E402
    load_compact_profile_support_inputs,
    load_prerequisite_authority_index,
)
from reference.hiliftaeroml.native_profiles import (  # noqa: E402
    PROFILE_CONTRACT_ID,
    PROFILE_CONTRACT_SHA256,
    ROWS,
    SAFE_CASE_ID,
    NativeProfileError,
    load_json,
    sha256_file,
    validate_cp_source,
    validate_velocity_source,
)


CASE_SET_SHA256_RULE = "sha256(utf8(case_id + newline) in listed order)"
AUTHORITY_SOURCE_HASH_SLOTS = {
    "cp_profile_metrics": "cp_stencil_record",
    "cp_cut_values": "cp_stencil_payload",
    "velocity_profile_metrics": "velocity_stencil_record",
    "velocity_profiles": "velocity_stencil_payload",
}


def _fail(message: str) -> None:
    raise CompactProfileEvaluationError(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label} must be a regular non-symlink directory: {path}")
    return path.resolve()


def _safe_declared_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        _fail(f"{label} must remain inside the benchmark-spec directory")
    resolved_root = root.resolve()
    unresolved = resolved_root / relative
    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            _fail(f"{label} traverses a symlink")
    path = unresolved.resolve()
    if resolved_root not in path.parents or not path.is_file() or path.is_symlink():
        _fail(f"{label} must name a regular file inside the benchmark-spec directory")
    return path


def _case_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        _fail("benchmark split case_ids must be a non-empty unique list")
    for case_id in value:
        if not isinstance(case_id, str) or SAFE_CASE_ID.fullmatch(case_id) is None:
            _fail(f"benchmark split contains invalid case ID {case_id!r}")
    return tuple(value)


def _case_set_sha256(case_ids: Sequence[str]) -> str:
    payload = "".join(f"{case_id}\n" for case_id in case_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_benchmark_bindings(
    *, submission_spec_path: Path, split_path: Path
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    tuple[str, ...],
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
]:
    try:
        spec, spec_sha = load_json(submission_spec_path, label="submission spec")
        split, split_sha = load_json(split_path, label="benchmark split")
    except NativeProfileError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    if spec.get("dataset_id") != "hiliftaeroml":
        _fail("submission spec dataset identity differs")
    expected_split_keys = {
        "schema_version",
        "dataset_id",
        "split_id",
        "case_set_id",
        "split_label",
        "case_id_status",
        "case_count",
        "case_set_sha256",
        "case_set_sha256_rule",
        "case_ids",
    }
    if set(split) != expected_split_keys:
        _fail("benchmark split keys differ")
    case_ids = _case_ids(split.get("case_ids"))
    if (
        split.get("schema_version") != "1.0"
        or split.get("dataset_id") != "hiliftaeroml"
        or split.get("case_id_status") != "official"
        or split.get("case_count") != len(case_ids)
        or split.get("case_set_sha256_rule") != CASE_SET_SHA256_RULE
        or split.get("case_set_sha256") != _case_set_sha256(case_ids)
    ):
        _fail("benchmark split identity, count, order, or case-set digest differs")

    split_declarations = spec.get("splits")
    if not isinstance(split_declarations, list):
        _fail("submission spec split declarations are absent")
    matches = [
        item
        for item in split_declarations
        if isinstance(item, Mapping) and item.get("id") == split.get("split_id")
    ]
    if len(matches) != 1:
        _fail("submission spec does not declare the selected split exactly once")
    declaration = matches[0]
    if set(declaration) != {
        "id",
        "label",
        "index_file",
        "case_count",
        "case_set_id",
        "case_id_status",
        "sha256",
    }:
        _fail("submission spec split declaration keys differ")
    declared_split_path = _safe_declared_file(
        submission_spec_path.parent,
        declaration.get("index_file"),
        "submission spec split index_file",
    )
    if declared_split_path != split_path.resolve():
        _fail("selected split path differs from the submission-spec declaration")
    if (
        declaration.get("label") != split.get("split_label")
        or declaration.get("case_count") != len(case_ids)
        or declaration.get("case_set_id") != split.get("case_set_id")
        or declaration.get("case_id_status") != "official"
        or declaration.get("sha256") != split_sha
    ):
        _fail("selected split differs from its submission-spec binding")

    compact = _require_mapping(
        spec.get("compact_profile_definition"), "compact_profile_definition"
    )
    contract_path = _safe_declared_file(
        submission_spec_path.parent,
        compact.get("file"),
        "compact profile contract file",
    )
    try:
        contract_sha = sha256_file(contract_path)
    except NativeProfileError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    if (
        compact.get("contract_id") != COMPACT_PROFILE_CONTRACT_ID
        or compact.get("format") != COMPACT_PROFILE_FORMAT
        or compact.get("sha256") != COMPACT_PROFILE_CONTRACT_SHA256
        or contract_sha != COMPACT_PROFILE_CONTRACT_SHA256
    ):
        _fail("compact profile contract differs from the reference evaluator")
    support_declaration = _require_mapping(
        compact.get("candidate_dry_run_evaluator_support"),
        "candidate_dry_run_evaluator_support",
    )
    if set(support_declaration) != {
        "status",
        "usage",
        "release_id",
        "manifest_sha256",
    }:
        _fail("candidate evaluator-support declaration keys differ")
    if (
        support_declaration.get("status") != COMPACT_SUPPORT_RELEASE_STATUS
        or support_declaration.get("usage") != COMPACT_SUPPORT_RELEASE_USAGE
        or support_declaration.get("release_id") != COMPACT_SUPPORT_RELEASE_ID
    ):
        _fail("candidate evaluator-support declaration identity differs")
    declared_support_sha = support_declaration.get("manifest_sha256")
    if declared_support_sha is not None:
        _require_sha(
            declared_support_sha,
            "candidate evaluator-support manifest_sha256",
        )

    native = _require_mapping(spec.get("profile_definition"), "profile_definition")
    if (
        native.get("contract_id") != PROFILE_CONTRACT_ID
        or native.get("sha256") != PROFILE_CONTRACT_SHA256
    ):
        _fail("native profile-v1 contract binding differs")
    truth_declaration = _require_mapping(
        native.get("candidate_dry_run_profile_ground_truth"),
        "candidate_dry_run_profile_ground_truth",
    )
    binding_path = _safe_declared_file(
        submission_spec_path.parent,
        truth_declaration.get("binding_file"),
        "candidate profile-truth binding file",
    )
    return (
        spec,
        spec_sha,
        split,
        split_sha,
        case_ids,
        support_declaration,
        binding_path,
        truth_declaration,
    )


class _StreamingCaseMaterializer:
    """Build at most one case support at a time for the release writer."""

    def __init__(
        self,
        *,
        case_ids: Sequence[str],
        surface_outputs_root: Path | None = None,
        volume_outputs_root: Path | None = None,
        truth_release: CandidateTruthRelease | None = None,
        case_sources: Mapping[
            str, tuple[Path, Path, CandidateTruthRelease]
        ]
        | None = None,
    ) -> None:
        self.case_ids = tuple(case_ids)
        self._case_set = set(self.case_ids)
        self._positions = {
            case_id: position
            for position, case_id in enumerate(self.case_ids, start=1)
        }
        if case_sources is None:
            if (
                surface_outputs_root is None
                or volume_outputs_root is None
                or truth_release is None
            ):
                _fail("single-route compact support inputs are incomplete")
            surface = _regular_directory(
                surface_outputs_root, "surface native-output root"
            )
            volume = _regular_directory(
                volume_outputs_root, "volume native-output root"
            )
            self._case_sources = {
                case_id: (surface, volume, truth_release)
                for case_id in self.case_ids
            }
        else:
            if any(
                value is not None
                for value in (
                    surface_outputs_root,
                    volume_outputs_root,
                    truth_release,
                )
            ):
                _fail("routed compact support cannot also use single-route inputs")
            if set(case_sources) != self._case_set:
                _fail("routed compact support case mapping differs")
            regular_roots: dict[Path, Path] = {}

            def regular(path: Path, label: str) -> Path:
                resolved = path.resolve()
                if resolved not in regular_roots:
                    regular_roots[resolved] = _regular_directory(path, label)
                return regular_roots[resolved]

            self._case_sources = {}
            for case_id in self.case_ids:
                raw = case_sources[case_id]
                if not isinstance(raw, tuple) or len(raw) != 3:
                    _fail(f"{case_id} routed compact support source is invalid")
                surface_root, volume_root, selected_truth = raw
                if not isinstance(selected_truth, CandidateTruthRelease):
                    _fail(f"{case_id} routed compact support truth handle is invalid")
                self._case_sources[case_id] = (
                    regular(
                        surface_root,
                        f"{case_id} surface native-output root",
                    ),
                    regular(
                        volume_root,
                        f"{case_id} volume native-output root",
                    ),
                    selected_truth,
                )
        self._cached_case_id: str | None = None
        self._cached_support: dict[str, np.ndarray] | None = None
        self._cached_hashes: dict[str, str] | None = None

    @staticmethod
    def _stream(root: Path, case_id: str, name: str) -> Path:
        case_root = root / case_id
        stream = case_root / name
        if (
            case_root.is_symlink()
            or not case_root.is_dir()
            or stream.is_symlink()
            or not stream.is_dir()
        ):
            _fail(f"{case_id} native-output stream is absent or non-regular: {stream}")
        return stream

    def _build(self, case_id: str) -> None:
        if case_id not in self._case_set:
            raise KeyError(case_id)
        if self._cached_case_id == case_id:
            return
        # Drop the prior support before loading the next native case.  The two
        # Mapping views deliberately share this cache, so the writer can ask
        # for hashes and support without causing a second build.
        self._cached_case_id = None
        self._cached_support = None
        self._cached_hashes = None
        surface_outputs_root, volume_outputs_root, truth_release = (
            self._case_sources[case_id]
        )
        surface = self._stream(
            surface_outputs_root, case_id, "surface_submission_stream"
        )
        volume = self._stream(
            volume_outputs_root, case_id, "volume_submission_stream"
        )
        cp_arrays, cp_metadata = validate_cp_source(
            metrics_path=surface / "cp_profile_metrics.json",
            npz_path=surface / "cp_cut_values.npz",
        )
        cp_metadata["station_rows"] = list(ROWS)
        velocity_arrays, velocity_metadata = validate_velocity_source(
            case_id=case_id,
            metrics_path=volume / "velocity_profile_metrics.json",
            npz_path=volume / "velocity_profiles.npz",
        )
        record, truth_cp, truth_velocity = _truth_case(truth_release, case_id)
        _validate_cp_alignment(case_id, cp_arrays, cp_metadata, record)
        _validate_velocity_alignment(
            case_id,
            velocity_arrays,
            velocity_metadata,
            record,
            truth_velocity,
        )
        support = build_compact_support(
            cp_native=cp_arrays,
            truth_cp=truth_cp,
            velocity_native=velocity_arrays,
            truth_velocity_nd=truth_velocity,
        )
        self._cached_case_id = case_id
        self._cached_support = support
        self._cached_hashes = {
            "cp_profile_metrics": cp_metadata["source_metrics_sha256"],
            "cp_cut_values": cp_metadata["source_npz_sha256"],
            "velocity_profile_metrics": velocity_metadata[
                "source_metrics_sha256"
            ],
            "velocity_profiles": velocity_metadata["source_npz_sha256"],
        }
        print(
            f"compact-support {self._positions[case_id]}/{len(self.case_ids)} "
            f"{case_id}",
            file=sys.stderr,
            flush=True,
        )

    def support(self, case_id: str) -> Mapping[str, np.ndarray]:
        self._build(case_id)
        assert self._cached_support is not None
        return self._cached_support

    def source_hashes(self, case_id: str) -> Mapping[str, str]:
        self._build(case_id)
        assert self._cached_hashes is not None
        return self._cached_hashes


def _authority_truth_binding(
    *,
    case_id: str,
    truth_record: Mapping[str, Any],
    authority_evidence: Mapping[str, str],
) -> None:
    truth_authority = _require_mapping(
        truth_record.get("truth_authority"), f"{case_id} truth authority"
    )
    surface_source = _require_mapping(
        _require_mapping(
            truth_record.get("surface_cp"), f"{case_id} truth surface Cp"
        ).get("source"),
        f"{case_id} truth surface Cp source",
    )
    velocity_source = _require_mapping(
        _require_mapping(
            truth_record.get("volume_velocity"),
            f"{case_id} truth volume velocity",
        ).get("source"),
        f"{case_id} truth volume velocity source",
    )
    expected = {
        "authority_case_identity_sha256": truth_authority.get(
            "authority_identity_sha256"
        ),
        "cp_stencil_identity_sha256": surface_source.get(
            "cp_stencil_identity_sha256"
        ),
        "velocity_stencil_identity_sha256": velocity_source.get(
            "velocity_stencil_identity_sha256"
        ),
        "validity_identity_sha256": velocity_source.get(
            "validity_identity_sha256"
        ),
    }
    if expected != dict(authority_evidence):
        _fail(f"{case_id} prerequisite authority and profile truth differ")


def _authority_source_hashes(
    *, case_id: str, authority: Mapping[str, Any]
) -> dict[str, str]:
    cases = _require_mapping(authority.get("cases"), "prerequisite authority cases")
    entry = _require_mapping(cases.get(case_id), f"{case_id} prerequisite authority")
    if entry.get("prediction_bearing_evaluator_outputs_used_as_source") is not False:
        _fail(f"{case_id} prerequisite authority is prediction-bearing")
    artifacts = _require_mapping(
        entry.get("artifacts"), f"{case_id} prerequisite authority artifacts"
    )
    result: dict[str, str] = {}
    for compatibility_slot, authority_name in AUTHORITY_SOURCE_HASH_SLOTS.items():
        descriptor = _require_mapping(
            artifacts.get(authority_name),
            f"{case_id} prerequisite authority {authority_name}",
        )
        result[compatibility_slot] = _require_sha(
            descriptor.get("sha256"),
            f"{case_id} prerequisite authority {authority_name} SHA-256",
        )
    return result


def _load_bound_prerequisite_authority(
    *, authority_index_path: Path, truth_release: CandidateTruthRelease
) -> tuple[dict[str, Any], str]:
    layout = _require_mapping(
        truth_release.binding.get("campaign_layout"),
        "profile-truth campaign layout",
    )
    descriptor = _require_mapping(
        layout.get("authority"), "profile-truth prerequisite authority descriptor"
    )
    relative = descriptor.get("file")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or "." in Path(relative).parts
    ):
        _fail("profile-truth prerequisite authority path is unsafe")
    expected_path = (truth_release.campaign_root / relative).resolve()
    supplied_path = authority_index_path.resolve()
    if (
        authority_index_path.is_symlink()
        or not supplied_path.is_file()
        or supplied_path != expected_path
    ):
        _fail("supplied prerequisite authority differs from the profile-truth binding")
    expected_sha = _require_sha(
        descriptor.get("sha256"), "profile-truth prerequisite authority SHA-256"
    )
    universe = CaseUniverse(
        case_ids=tuple(truth_release.index.get("case_ids", ())),
        case_sets=(),
        support_manifest_sha256=_require_sha(
            truth_release.index.get("support_manifest_sha256"),
            "profile-truth scoring-support manifest SHA-256",
        ),
        case_universe_sha256=_require_sha(
            truth_release.index.get("case_universe_sha256"),
            "profile-truth case-universe SHA-256",
        ),
    )
    try:
        authority, observed_sha = load_prerequisite_authority_index(
            supplied_path,
            universe=universe,
            require_complete=True,
        )
    except NativeProfileTruthError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    if observed_sha != expected_sha:
        _fail("prerequisite authority digest differs from the profile-truth binding")
    return authority, observed_sha


class _AuthorityCaseMaterializer:
    """Build support prediction-free from the truth release's frozen authority."""

    def __init__(
        self,
        *,
        case_ids: Sequence[str],
        truth_releases: Mapping[str, CandidateTruthRelease],
        authority: Mapping[str, Any],
    ) -> None:
        self.case_ids = tuple(case_ids)
        if set(truth_releases) != set(self.case_ids):
            _fail("authority compact support truth mapping differs")
        self._truth_releases = dict(truth_releases)
        self._authority = authority
        self._positions = {
            case_id: position
            for position, case_id in enumerate(self.case_ids, start=1)
        }
        self._cached_case_id: str | None = None
        self._cached_support: dict[str, np.ndarray] | None = None
        self._cached_hashes: dict[str, str] | None = None

    def _build(self, case_id: str) -> None:
        if case_id not in self._truth_releases:
            raise KeyError(case_id)
        if self._cached_case_id == case_id:
            return
        self._cached_case_id = None
        self._cached_support = None
        self._cached_hashes = None
        try:
            cp_native, velocity_native, authority_evidence = (
                load_compact_profile_support_inputs(
                    case_id=case_id,
                    authority=self._authority,
                )
            )
        except NativeProfileTruthError as error:
            raise CompactProfileEvaluationError(str(error)) from error
        record, truth_cp, truth_velocity = _truth_case(
            self._truth_releases[case_id], case_id
        )
        _authority_truth_binding(
            case_id=case_id,
            truth_record=record,
            authority_evidence=authority_evidence,
        )
        support = build_compact_support(
            cp_native=cp_native,
            truth_cp=truth_cp,
            velocity_native=velocity_native,
            truth_velocity_nd=truth_velocity,
        )
        self._cached_case_id = case_id
        self._cached_support = support
        self._cached_hashes = _authority_source_hashes(
            case_id=case_id, authority=self._authority
        )
        print(
            f"compact-support {self._positions[case_id]}/{len(self.case_ids)} "
            f"{case_id}",
            file=sys.stderr,
            flush=True,
        )

    def support(self, case_id: str) -> Mapping[str, np.ndarray]:
        self._build(case_id)
        assert self._cached_support is not None
        return self._cached_support

    def source_hashes(self, case_id: str) -> Mapping[str, str]:
        self._build(case_id)
        assert self._cached_hashes is not None
        return self._cached_hashes


class _SupportView(Mapping[str, Mapping[str, np.ndarray]]):
    def __init__(
        self, source: _StreamingCaseMaterializer | _AuthorityCaseMaterializer
    ) -> None:
        self.source = source

    def __getitem__(self, case_id: str) -> Mapping[str, np.ndarray]:
        return self.source.support(case_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self.source.case_ids)

    def __len__(self) -> int:
        return len(self.source.case_ids)


class _SourceHashView(Mapping[str, Mapping[str, str]]):
    def __init__(
        self, source: _StreamingCaseMaterializer | _AuthorityCaseMaterializer
    ) -> None:
        self.source = source

    def __getitem__(self, case_id: str) -> Mapping[str, str]:
        return self.source.source_hashes(case_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self.source.case_ids)

    def __len__(self) -> int:
        return len(self.source.case_ids)


def materialize_compact_support_release(
    *,
    submission_spec_path: Path,
    split_path: Path,
    surface_outputs_root: Path,
    volume_outputs_root: Path,
    source_truth_release: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build one exact split support release and return its binding receipt."""

    (
        _spec,
        spec_sha,
        split,
        split_sha,
        case_ids,
        support_declaration,
        binding_path,
        truth_declaration,
    ) = _load_benchmark_bindings(
        submission_spec_path=submission_spec_path,
        split_path=split_path,
    )
    try:
        truth_release = open_candidate_truth_release(
            release_root=source_truth_release,
            candidate_declaration=truth_declaration,
            expected_case_ids=case_ids,
            case_set_id=split["case_set_id"],
            binding_path=binding_path,
        )
    except NativeProfileEvaluationError as error:
        raise CompactProfileEvaluationError(str(error)) from error
    source_truth_release_id = truth_release.manifest.get("release_id")
    source_truth_manifest_sha = truth_declaration.get("manifest_sha256")
    if (
        source_truth_release_id != truth_declaration.get("release_id")
        or truth_release.manifest.get("format") != TRUTH_FORMAT
    ):
        _fail("opened source profile-truth release identity differs")
    source_truth_manifest_sha = _require_sha(
        source_truth_manifest_sha, "source profile-truth manifest SHA-256"
    )

    source = _StreamingCaseMaterializer(
        case_ids=case_ids,
        surface_outputs_root=surface_outputs_root,
        volume_outputs_root=volume_outputs_root,
        truth_release=truth_release,
    )
    try:
        manifest_sha = write_compact_support_release(
            release_root=output_root,
            case_ids=case_ids,
            case_sets={split["case_set_id"]: case_ids},
            supports=_SupportView(source),
            source_artifact_sha256=_SourceHashView(source),
            source_profile_truth_release_id=source_truth_release_id,
            source_profile_truth_manifest_sha256=source_truth_manifest_sha,
        )
    except (CompactProfileError, NativeProfileError) as error:
        raise CompactProfileEvaluationError(str(error)) from error
    declared_manifest_sha = support_declaration.get("manifest_sha256")
    if declared_manifest_sha is not None and manifest_sha != declared_manifest_sha:
        # The writer reached this path only after exclusively creating the
        # requested root.  Remove those newly produced, but incorrectly bound,
        # bytes rather than leaving a plausible-looking release behind.
        if output_root.is_dir() and not output_root.is_symlink():
            shutil.rmtree(output_root)
        _fail("materialized support manifest differs from its submission-spec pin")
    handle = open_compact_support_release(
        release_root=output_root,
        expected_manifest_sha256=manifest_sha,
        expected_case_ids=case_ids,
        case_set_id=split["case_set_id"],
    )
    if (
        handle.source_profile_truth_release_id != source_truth_release_id
        or handle.source_profile_truth_manifest_sha256
        != source_truth_manifest_sha
    ):
        _fail("materialized support source-truth binding differs")
    return {
        "status": COMPACT_SUPPORT_RELEASE_STATUS,
        "usage": COMPACT_SUPPORT_RELEASE_USAGE,
        "dataset_id": "hiliftaeroml",
        "evaluator_support_release_id": COMPACT_SUPPORT_RELEASE_ID,
        "evaluator_support_manifest_sha256": manifest_sha,
        "profile_contract_id": COMPACT_PROFILE_CONTRACT_ID,
        "profile_contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
        "source_profile_truth_release_id": source_truth_release_id,
        "source_profile_truth_manifest_sha256": source_truth_manifest_sha,
        "submission_spec_sha256": spec_sha,
        "split_id": split["split_id"],
        "split_sha256": split_sha,
        "case_set_id": split["case_set_id"],
        "case_set_sha256": split["case_set_sha256"],
        "case_count": len(case_ids),
        "points_per_physical_graph": CP_POINTS_PER_GRAPH,
        "output_root": str(output_root.resolve()),
        "activation": {
            "owner_approval_complete": False,
            "published": False,
            "submissions_opened": False,
            "materialization_changes_activation": False,
        },
    }


def materialize_routed_compact_support_release(
    *,
    submission_spec_path: Path,
    split_paths: Sequence[Path],
    surface_outputs_roots: Sequence[Path],
    volume_outputs_roots: Sequence[Path],
    source_truth_release: Path,
    output_root: Path,
    allow_manifest_rebind_from: str | None = None,
    prerequisite_authority_index: Path | None = None,
) -> dict[str, Any]:
    """Build one support release covering several exact benchmark case sets.

    In native-output mode the three route sequences are positional: each split
    uses the surface and volume roots at the same position.  Authority mode has
    no output roots and reconstructs support from the prediction-free source
    bound into the truth release.  If a case occurs in more than one split, its
    first declared truth handle is retained.  Case-set member order remains
    exactly official, while the deduplicated release inventory is sorted for a
    stable master-case traversal.
    """

    route_count = len(split_paths)
    authority_mode = prerequisite_authority_index is not None
    if route_count == 0:
        _fail("compact support requires at least one split")
    if authority_mode and (surface_outputs_roots or volume_outputs_roots):
        _fail("prerequisite-authority mode cannot also use native-output roots")
    if not authority_mode and (
        len(surface_outputs_roots) != route_count
        or len(volume_outputs_roots) != route_count
    ):
        _fail("compact support split and native-output route counts differ")
    if allow_manifest_rebind_from is not None:
        allow_manifest_rebind_from = _require_sha(
            allow_manifest_rebind_from,
            "allowed prior compact-support manifest SHA-256",
        )

    loaded: list[
        tuple[
            Path,
            Path | None,
            Path | None,
            str,
            Mapping[str, Any],
            str,
            tuple[str, ...],
            Mapping[str, Any],
            Path,
            Mapping[str, Any],
        ]
    ] = []
    common_spec_sha: str | None = None
    common_support_declaration: Mapping[str, Any] | None = None
    common_binding_path: Path | None = None
    common_truth_declaration: Mapping[str, Any] | None = None
    seen_split_ids: set[str] = set()
    for route_index, split_path in enumerate(split_paths):
        surface_root = (
            None if authority_mode else surface_outputs_roots[route_index]
        )
        volume_root = None if authority_mode else volume_outputs_roots[route_index]
        (
            _spec,
            spec_sha,
            split,
            split_sha,
            case_ids,
            support_declaration,
            binding_path,
            truth_declaration,
        ) = _load_benchmark_bindings(
            submission_spec_path=submission_spec_path,
            split_path=split_path,
        )
        split_id = split.get("split_id")
        if not isinstance(split_id, str) or split_id in seen_split_ids:
            _fail("routed compact support split IDs must be unique")
        seen_split_ids.add(split_id)
        if common_spec_sha is None:
            common_spec_sha = spec_sha
            common_support_declaration = support_declaration
            common_binding_path = binding_path
            common_truth_declaration = truth_declaration
        elif (
            spec_sha != common_spec_sha
            or support_declaration != common_support_declaration
            or binding_path != common_binding_path
            or truth_declaration != common_truth_declaration
        ):
            _fail("routed compact support benchmark bindings differ")
        loaded.append(
            (
                split_path,
                surface_root,
                volume_root,
                split_sha,
                split,
                split_id,
                case_ids,
                support_declaration,
                binding_path,
                truth_declaration,
            )
        )

    if (
        common_spec_sha is None
        or common_support_declaration is None
        or common_binding_path is None
        or common_truth_declaration is None
    ):
        _fail("routed compact support benchmark bindings are absent")
    declared_manifest_sha = common_support_declaration.get("manifest_sha256")
    if (
        allow_manifest_rebind_from is not None
        and declared_manifest_sha != allow_manifest_rebind_from
    ):
        _fail("allowed prior compact-support manifest differs from the live pin")

    truth_handles: dict[str, CandidateTruthRelease] = {}
    case_sets: dict[str, tuple[str, ...]] = {}
    case_sources: dict[
        str, tuple[Path, Path, CandidateTruthRelease]
    ] = {}
    authority_truth_releases: dict[str, CandidateTruthRelease] = {}
    route_receipts: list[dict[str, Any]] = []
    source_truth_release_id: str | None = None
    source_truth_manifest_sha = _require_sha(
        common_truth_declaration.get("manifest_sha256"),
        "source profile-truth manifest SHA-256",
    )
    for (
        split_path,
        surface_root,
        volume_root,
        split_sha,
        split,
        split_id,
        case_ids,
        _support_declaration,
        binding_path,
        truth_declaration,
    ) in loaded:
        case_set_id = split.get("case_set_id")
        if not isinstance(case_set_id, str):
            _fail(f"{split_id} routed compact support case-set ID is invalid")
        previous_members = case_sets.get(case_set_id)
        if previous_members is not None and previous_members != case_ids:
            _fail("duplicate compact support case-set ID has different members")
        case_sets.setdefault(case_set_id, case_ids)
        truth_handle = truth_handles.get(case_set_id)
        if truth_handle is None:
            try:
                truth_handle = open_candidate_truth_release(
                    release_root=source_truth_release,
                    candidate_declaration=truth_declaration,
                    expected_case_ids=case_ids,
                    case_set_id=case_set_id,
                    binding_path=binding_path,
                )
            except NativeProfileEvaluationError as error:
                raise CompactProfileEvaluationError(str(error)) from error
            truth_handles[case_set_id] = truth_handle
        opened_release_id = truth_handle.manifest.get("release_id")
        if (
            opened_release_id != truth_declaration.get("release_id")
            or truth_handle.manifest.get("format") != TRUTH_FORMAT
        ):
            _fail("opened source profile-truth release identity differs")
        if source_truth_release_id is None:
            source_truth_release_id = opened_release_id
        elif opened_release_id != source_truth_release_id:
            _fail("routed compact support source truth releases differ")
        selected_count = 0
        for case_id in case_ids:
            if authority_mode and case_id not in authority_truth_releases:
                authority_truth_releases[case_id] = truth_handle
                selected_count += 1
            elif not authority_mode and case_id not in case_sources:
                assert surface_root is not None and volume_root is not None
                case_sources[case_id] = (
                    surface_root,
                    volume_root,
                    truth_handle,
                )
                selected_count += 1
        route_receipt = {
            "split_id": split_id,
            "split_sha256": split_sha,
            "split_path": str(split_path.resolve()),
            "case_set_id": case_set_id,
            "case_set_sha256": split["case_set_sha256"],
            "case_count": len(case_ids),
            "selected_source_case_count": selected_count,
        }
        if authority_mode:
            route_receipt["support_source"] = "frozen_prerequisite_authority"
        else:
            assert surface_root is not None and volume_root is not None
            route_receipt.update(
                {
                    "support_source": "validated_native_evaluator_outputs",
                    "surface_outputs_root": str(surface_root.resolve()),
                    "volume_outputs_root": str(volume_root.resolve()),
                }
            )
        route_receipts.append(route_receipt)
    if source_truth_release_id is None:
        _fail("routed compact support source truth release is absent")

    selected_cases = authority_truth_releases if authority_mode else case_sources
    master_case_ids = tuple(sorted(selected_cases))
    authority_sha: str | None = None
    if authority_mode:
        assert prerequisite_authority_index is not None
        if not truth_handles:
            _fail("prerequisite-authority support has no truth handle")
        first_truth_handle = next(iter(truth_handles.values()))
        authority, authority_sha = _load_bound_prerequisite_authority(
            authority_index_path=prerequisite_authority_index,
            truth_release=first_truth_handle,
        )
        for truth_handle in truth_handles.values():
            if (
                truth_handle.campaign_root != first_truth_handle.campaign_root
                or truth_handle.release_root != first_truth_handle.release_root
                or truth_handle.index != first_truth_handle.index
            ):
                _fail("routed compact support profile-truth handles differ")
        source: _StreamingCaseMaterializer | _AuthorityCaseMaterializer = (
            _AuthorityCaseMaterializer(
                case_ids=master_case_ids,
                truth_releases=authority_truth_releases,
                authority=authority,
            )
        )
    else:
        source = _StreamingCaseMaterializer(
            case_ids=master_case_ids,
            case_sources=case_sources,
        )
    try:
        manifest_sha = write_compact_support_release(
            release_root=output_root,
            case_ids=master_case_ids,
            case_sets=case_sets,
            supports=_SupportView(source),
            source_artifact_sha256=_SourceHashView(source),
            source_profile_truth_release_id=source_truth_release_id,
            source_profile_truth_manifest_sha256=source_truth_manifest_sha,
        )
    except (CompactProfileError, NativeProfileError) as error:
        raise CompactProfileEvaluationError(str(error)) from error
    if (
        declared_manifest_sha is not None
        and manifest_sha != declared_manifest_sha
        and allow_manifest_rebind_from is None
    ):
        if output_root.is_dir() and not output_root.is_symlink():
            shutil.rmtree(output_root)
        _fail("materialized support manifest differs from its submission-spec pin")

    try:
        first_case_set_id = next(iter(case_sets))
        handle = open_compact_support_release(
            release_root=output_root,
            expected_manifest_sha256=manifest_sha,
            expected_case_ids=case_sets[first_case_set_id],
            case_set_id=first_case_set_id,
        )
        if (
            handle.source_profile_truth_release_id != source_truth_release_id
            or handle.source_profile_truth_manifest_sha256
            != source_truth_manifest_sha
        ):
            _fail("materialized support source-truth binding differs")
        indexed_case_sets = {
            descriptor["case_set_id"]: tuple(descriptor["case_ids"])
            for descriptor in handle.index["case_sets"]
        }
        expected_case_sets = {
            case_set_id: tuple(members)
            for case_set_id, members in case_sets.items()
        }
        if indexed_case_sets != expected_case_sets:
            _fail("materialized support case-set bindings differ")
    except CompactProfileEvaluationError:
        if output_root.is_dir() and not output_root.is_symlink():
            shutil.rmtree(output_root)
        raise
    return {
        "status": COMPACT_SUPPORT_RELEASE_STATUS,
        "usage": COMPACT_SUPPORT_RELEASE_USAGE,
        "dataset_id": "hiliftaeroml",
        "evaluator_support_release_id": COMPACT_SUPPORT_RELEASE_ID,
        "evaluator_support_manifest_sha256": manifest_sha,
        "profile_contract_id": COMPACT_PROFILE_CONTRACT_ID,
        "profile_contract_sha256": COMPACT_PROFILE_CONTRACT_SHA256,
        "source_profile_truth_release_id": source_truth_release_id,
        "source_profile_truth_manifest_sha256": source_truth_manifest_sha,
        "submission_spec_sha256": common_spec_sha,
        "split_count": len(route_receipts),
        "case_set_count": len(case_sets),
        "case_count": len(master_case_ids),
        "points_per_physical_graph": CP_POINTS_PER_GRAPH,
        "support_source": (
            {
                "kind": "frozen_prerequisite_authority",
                "prediction_bearing_evaluator_outputs_used_as_source": False,
                "authority_index": str(prerequisite_authority_index.resolve()),
                "authority_index_sha256": authority_sha,
                "legacy_source_hash_slot_mapping": dict(
                    AUTHORITY_SOURCE_HASH_SLOTS
                ),
            }
            if authority_mode and prerequisite_authority_index is not None
            else {
                "kind": "validated_native_evaluator_outputs",
                "prediction_bearing_evaluator_outputs_used_as_source": True,
            }
        ),
        "routes": route_receipts,
        "output_root": str(output_root.resolve()),
        "manifest_rebind": {
            "allowed": allow_manifest_rebind_from is not None,
            "prior_manifest_sha256": declared_manifest_sha,
            "materialized_manifest_sha256": manifest_sha,
        },
        "activation": {
            "owner_approval_complete": False,
            "published": False,
            "submissions_opened": False,
            "materialization_changes_activation": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--submission-spec", required=True, type=Path)
    result.add_argument("--split", required=True, action="append", type=Path)
    result.add_argument(
        "--surface-outputs-root", action="append", type=Path, default=[]
    )
    result.add_argument(
        "--volume-outputs-root", action="append", type=Path, default=[]
    )
    result.add_argument(
        "--prerequisite-authority-index",
        type=Path,
        help=(
            "prediction-free prerequisite authority bound into the native-profile "
            "truth release; mutually exclusive with native-output roots"
        ),
    )
    result.add_argument("--source-truth-release", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument(
        "--allow-manifest-rebind-from",
        help=(
            "explicitly permit a routed release to replace this currently "
            "pinned compact-support manifest digest"
        ),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if (
            args.prerequisite_authority_index is None
            and len(args.split) == 1
            and len(args.surface_outputs_root) == 1
            and len(args.volume_outputs_root) == 1
            and args.allow_manifest_rebind_from is None
        ):
            receipt = materialize_compact_support_release(
                submission_spec_path=args.submission_spec,
                split_path=args.split[0],
                surface_outputs_root=args.surface_outputs_root[0],
                volume_outputs_root=args.volume_outputs_root[0],
                source_truth_release=args.source_truth_release,
                output_root=args.output_root,
            )
        else:
            receipt = materialize_routed_compact_support_release(
                submission_spec_path=args.submission_spec,
                split_paths=args.split,
                surface_outputs_roots=args.surface_outputs_root,
                volume_outputs_roots=args.volume_outputs_root,
                source_truth_release=args.source_truth_release,
                output_root=args.output_root,
                allow_manifest_rebind_from=args.allow_manifest_rebind_from,
                prerequisite_authority_index=args.prerequisite_authority_index,
            )
    except (
        CompactProfileEvaluationError,
        CompactProfileError,
        NativeProfileEvaluationError,
        NativeProfileTruthError,
        NativeProfileError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
