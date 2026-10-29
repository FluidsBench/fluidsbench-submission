"""Hidden-truth scoring for prediction-only HiLiftAeroML profile chunks.

The participant package contains only Cp and velocity predictions plus their
native topology/alignment arrays.  This module joins those bytes to an
explicitly supplied benchmark-owned truth release, verifies the complete
inactive-candidate evidence chain, and recomputes the two profile R2 metrics.
It deliberately has no path that changes publication or activation state.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reference.hiliftaeroml.native_profile_truth import (
    TRUTH_ARRAYS,
    TRUTH_FORMAT,
    TRUTH_RELEASE_ID,
    NativeProfileTruthError,
    _combined_array_identity,
    _document_identity,
    _gap_runs,
    _load_case_record,
    _load_descriptor,
    array_identity,
    load_json,
    sha256_file,
)
from reference.hiliftaeroml.native_profiles import (
    CP_PUBLISHED_ARRAYS,
    PROFILE_CHUNK_SCHEMA,
    PROFILE_CONTRACT_ID,
    PROFILE_CONTRACT_SHA256,
    PROFILE_FORMAT,
    VELOCITY_PUBLISHED_ARRAYS,
    NativeProfileError,
    validate_prediction_npz,
)


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_BINDING_PATH = (
    ROOT
    / "benchmark-specs"
    / "hiliftaeroml"
    / "candidate-profile-truth-binding.json"
)
CANDIDATE_BINDING_SCHEMA = "hiliftaeroml-local-candidate-profile-truth-binding-v1"
CANDIDATE_TRUTH_STATUS = "complete_candidate_not_published"
CANDIDATE_USAGE = "maintainer_local_candidate_dry_run_only"
TRUTH_MANIFEST_SCHEMA = "hiliftaeroml-native-profile-truth-manifest-v1-candidate"
TRUTH_INDEX_SCHEMA = "hiliftaeroml-native-profile-truth-index-v1-candidate"
TRUTH_CASE_SCHEMA = "hiliftaeroml-native-profile-truth-case-v1-candidate"
TRUTH_RECEIPT_SCHEMA = (
    "hiliftaeroml-native-profile-truth-release-receipt-v1-candidate"
)
TRUTH_PROVENANCE_SCHEMA = "hiliftaeroml-native-profile-truth-provenance-v1-candidate"
TRUTH_AUTHORITY_SCHEMA = (
    "hiliftaeroml-native-profile-truth-prerequisite-authority-index-v1-candidate"
)
TRUTH_PREFLIGHT_SCHEMA = (
    "hiliftaeroml-native-profile-truth-prerequisite-preflight-v1-candidate"
)


class NativeProfileEvaluationError(ValueError):
    """Raised when hidden truth cannot safely score a prediction package."""


@dataclass(frozen=True)
class CandidateTruthRelease:
    """Validated local handle to one inactive hidden-truth candidate."""

    campaign_root: Path
    release_root: Path
    binding: dict[str, Any]
    manifest: dict[str, Any]
    index: dict[str, Any]
    case_set_id: str
    case_ids: tuple[str, ...]


def _fail(message: str) -> None:
    raise NativeProfileEvaluationError(message)


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        _fail(f"{label} must remain inside its root")
    resolved_root = root.resolve()
    unresolved = resolved_root / candidate
    resolved = unresolved.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        _fail(f"{label} resolves outside its root")
    cursor = resolved_root
    for part in candidate.parts:
        cursor /= part
        if cursor.is_symlink():
            _fail(f"{label} traverses a symlink")
    return resolved


def _load_campaign_descriptor(
    campaign_root: Path, descriptor: Any, *, label: str
) -> tuple[Path, str]:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "file",
        "sha256",
        "byte_size",
    }:
        _fail(f"{label} descriptor keys differ")
    path = _safe_child(campaign_root, descriptor.get("file"), f"{label}.file")
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} must be a regular non-symlink file")
    digest = _require_sha(descriptor.get("sha256"), f"{label}.sha256")
    size = _positive_int(descriptor.get("byte_size"), f"{label}.byte_size")
    before = path.stat()
    observed = sha256_file(path)
    after = path.stat()
    identity = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if identity(before) != identity(after):
        _fail(f"{label} changed while it was read")
    if before.st_size != size or observed != digest:
        _fail(f"{label} live bytes differ")
    return path, digest


def load_candidate_truth_binding(
    *,
    binding_path: Path = CANDIDATE_BINDING_PATH,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the repository binding without interpreting it as activation."""

    try:
        binding, digest = load_json(binding_path, label="candidate profile-truth binding")
    except NativeProfileTruthError as error:
        raise NativeProfileEvaluationError(str(error)) from error
    if expected_sha256 is not None and digest != _require_sha(
        expected_sha256, "candidate profile-truth binding digest"
    ):
        _fail("candidate profile-truth binding SHA-256 changed")
    if set(binding) != {
        "schema",
        "schema_version",
        "status",
        "usage",
        "dataset_id",
        "release",
        "campaign_layout",
        "validation_claims",
        "activation",
    }:
        _fail("candidate profile-truth binding keys differ")
    if (
        binding.get("schema") != CANDIDATE_BINDING_SCHEMA
        or binding.get("schema_version") != 1
        or binding.get("status") != CANDIDATE_TRUTH_STATUS
        or binding.get("usage") != CANDIDATE_USAGE
        or binding.get("dataset_id") != "hiliftaeroml"
    ):
        _fail("candidate profile-truth binding identity differs")
    release = binding.get("release")
    if not isinstance(release, Mapping) or set(release) != {
        "release_id",
        "format",
        "manifest_sha256",
        "profile_contract_id",
        "profile_contract_sha256",
        "case_count",
        "case_set_count",
    }:
        _fail("candidate profile-truth release binding differs")
    if (
        release.get("release_id") != TRUTH_RELEASE_ID
        or release.get("format") != TRUTH_FORMAT
        or release.get("profile_contract_id") != PROFILE_CONTRACT_ID
        or release.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or release.get("case_count") != 1355
        or release.get("case_set_count") != 8
    ):
        _fail("candidate profile-truth release identity differs")
    _require_sha(release.get("manifest_sha256"), "candidate truth manifest digest")
    claims = binding.get("validation_claims")
    if claims != {
        "source_replay_complete": True,
        "deterministic_packaging": True,
        "prediction_bearing_evaluator_outputs_used_as_truth_source": False,
    }:
        _fail("candidate profile-truth validation claims differ")
    activation = binding.get("activation")
    if activation != {
        "owner_approval_complete": False,
        "published": False,
        "submissions_opened": False,
        "candidate_validation_may_change_activation": False,
    }:
        _fail("candidate profile-truth activation boundary differs")
    return dict(binding)


def _require_candidate_declaration(
    declaration: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    if set(declaration) != {
        "status",
        "usage",
        "release_id",
        "manifest_sha256",
        "binding_file",
        "binding_sha256",
    }:
        _fail("candidate profile-ground-truth declaration keys differ")
    expected = {
        "status": CANDIDATE_TRUTH_STATUS,
        "usage": CANDIDATE_USAGE,
        "release_id": binding["release"]["release_id"],
        "manifest_sha256": binding["release"]["manifest_sha256"],
    }
    for key, value in expected.items():
        if declaration.get(key) != value:
            _fail(f"candidate profile-ground-truth declaration {key} differs")


def open_candidate_truth_release(
    *,
    release_root: Path,
    candidate_declaration: Mapping[str, Any],
    expected_case_ids: Sequence[str],
    case_set_id: str,
    binding_path: Path = CANDIDATE_BINDING_PATH,
) -> CandidateTruthRelease:
    """Validate one local inactive release and the selected exact case set."""

    binding_sha = candidate_declaration.get("binding_sha256")
    binding = load_candidate_truth_binding(
        binding_path=binding_path,
        expected_sha256=_require_sha(binding_sha, "candidate binding digest"),
    )
    _require_candidate_declaration(candidate_declaration, binding)
    if candidate_declaration.get("binding_file") != binding_path.name:
        _fail("candidate profile-ground-truth binding filename differs")
    if not expected_case_ids or len(expected_case_ids) != len(set(expected_case_ids)):
        _fail("selected profile truth case IDs must be non-empty and unique")
    if not isinstance(case_set_id, str) or not case_set_id:
        _fail("selected profile truth case-set ID is invalid")

    root = release_root.resolve()
    if release_root.is_symlink() or not root.is_dir():
        _fail("candidate profile-truth release must be a regular non-symlink directory")
    campaign_root = root.parent.parent
    layout = binding.get("campaign_layout")
    if not isinstance(layout, Mapping) or set(layout) != {
        "release_root",
        "release_files",
        "authority",
        "authority_preflight",
        "validation",
    }:
        _fail("candidate profile-truth campaign layout differs")
    expected_root = _safe_child(
        campaign_root, layout.get("release_root"), "candidate release root"
    )
    if expected_root != root:
        _fail("candidate profile-truth release is outside the bound campaign layout")

    release_files = layout.get("release_files")
    if not isinstance(release_files, Mapping) or set(release_files) != {
        "manifest",
        "index",
        "provenance",
        "release_receipt",
    }:
        _fail("candidate profile-truth release-file binding differs")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name, descriptor in release_files.items():
        paths[name], digests[name] = _load_campaign_descriptor(
            campaign_root, descriptor, label=f"candidate truth {name}"
        )
        if paths[name].parent != root:
            _fail(f"candidate truth {name} is outside the selected release root")
    authority_path, authority_sha = _load_campaign_descriptor(
        campaign_root, layout.get("authority"), label="candidate truth authority"
    )
    preflight_path, _ = _load_campaign_descriptor(
        campaign_root,
        layout.get("authority_preflight"),
        label="candidate truth authority preflight",
    )
    validation_path, _ = _load_campaign_descriptor(
        campaign_root, layout.get("validation"), label="candidate truth validation"
    )
    if digests["manifest"] != binding["release"]["manifest_sha256"]:
        _fail("candidate truth manifest differs from the repository binding")

    try:
        manifest, _ = load_json(paths["manifest"], label="candidate truth manifest")
        index, _ = load_json(paths["index"], label="candidate truth index")
        provenance, _ = load_json(paths["provenance"], label="candidate truth provenance")
        receipt, _ = load_json(
            paths["release_receipt"], label="candidate truth release receipt"
        )
        authority, _ = load_json(authority_path, label="candidate truth authority")
        preflight, _ = load_json(preflight_path, label="candidate truth preflight")
        validation, _ = load_json(validation_path, label="candidate truth validation")
    except NativeProfileTruthError as error:
        raise NativeProfileEvaluationError(str(error)) from error

    release = binding["release"]
    inactive = {
        "owner_approval_complete": False,
        "published": False,
        "submissions_opened": False,
    }
    if (
        manifest.get("schema") != TRUTH_MANIFEST_SCHEMA
        or manifest.get("status") != "candidate_owner_review_required"
        or manifest.get("dataset_id") != "hiliftaeroml"
        or manifest.get("release_id") != release["release_id"]
        or manifest.get("format") != release["format"]
        or manifest.get("profile_contract_id") != PROFILE_CONTRACT_ID
        or manifest.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or manifest.get("case_count") != release["case_count"]
        or manifest.get("case_set_count") != release["case_set_count"]
        or manifest.get("activation") != inactive
    ):
        _fail("candidate truth manifest identity or inactive status differs")
    if (
        manifest.get("master_index", {}).get("sha256") != digests["index"]
        or manifest.get("provenance", {}).get("sha256") != digests["provenance"]
    ):
        _fail("candidate truth manifest artifact binding differs")
    if (
        index.get("schema") != TRUTH_INDEX_SCHEMA
        or index.get("status") != "candidate_owner_review_required"
        or index.get("release_id") != release["release_id"]
        or index.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or index.get("case_count") != release["case_count"]
        or not isinstance(index.get("case_ids"), list)
        or len(index["case_ids"]) != len(set(index["case_ids"]))
    ):
        _fail("candidate truth master index differs")
    if (
        provenance.get("schema") != TRUTH_PROVENANCE_SCHEMA
        or provenance.get("status") != "candidate_owner_review_required"
        or provenance.get("release_id") != release["release_id"]
        or provenance.get("truth_source_index_sha256") != authority_sha
        or provenance.get("participant_artifacts_remain_prediction_only") is not True
    ):
        _fail("candidate truth provenance differs")
    if (
        receipt.get("schema") != TRUTH_RECEIPT_SCHEMA
        or receipt.get("status") != CANDIDATE_TRUTH_STATUS
        or receipt.get("release_id") != release["release_id"]
        or receipt.get("manifest_sha256") != digests["manifest"]
        or receipt.get("truth_source_index_sha256") != authority_sha
        or receipt.get("case_count") != release["case_count"]
        or receipt.get("case_set_count") != release["case_set_count"]
        or receipt.get("deterministic_packaging") is not True
        or any(receipt.get(key) is not False for key in inactive)
    ):
        _fail("candidate truth release receipt differs")
    if (
        authority.get("schema") != TRUTH_AUTHORITY_SCHEMA
        or authority.get("status") != "complete"
        or authority.get("dataset_id") != "hiliftaeroml"
        or authority.get("expected_case_count") != release["case_count"]
        or authority.get("available_case_count") != release["case_count"]
        or authority.get("missing_case_count") != 0
        or authority.get("case_ids") != index["case_ids"]
    ):
        _fail("candidate truth authority identity or coverage differs")
    if (
        preflight.get("schema") != TRUTH_PREFLIGHT_SCHEMA
        or preflight.get("status") != "ready"
        or preflight.get("case_count") != release["case_count"]
        or preflight.get("case_set_count") != release["case_set_count"]
        or preflight.get("authority_index_sha256") != authority_sha
        or preflight.get("owner_approval_complete") is not False
        or preflight.get("published") is not False
        or preflight.get("activation_changed") is not False
        or preflight.get("source_io_contract", {}).get(
            "prediction_outputs_as_authority"
        )
        is not False
    ):
        _fail("candidate truth authority preflight differs")
    if validation != {
        "case_count": release["case_count"],
        "case_set_count": release["case_set_count"],
        "manifest_sha256": digests["manifest"],
        "release_id": release["release_id"],
        "source_replay_complete": True,
        "status": "pass",
    }:
        _fail("candidate truth validation receipt differs")

    descriptors = manifest.get("case_sets")
    matches = [
        item
        for item in descriptors if isinstance(item, Mapping) and item.get("case_set_id") == case_set_id
    ] if isinstance(descriptors, list) else []
    if len(matches) != 1:
        _fail("candidate truth manifest does not contain the selected case set exactly once")
    try:
        case_set_path, _ = _load_descriptor(
            {
                key: matches[0][key]
                for key in ("file", "sha256", "byte_size")
            },
            label=f"candidate truth {case_set_id}",
            root=root,
        )
        case_set, _ = load_json(case_set_path, label=f"candidate truth {case_set_id}")
    except (KeyError, NativeProfileTruthError) as error:
        raise NativeProfileEvaluationError(str(error)) from error
    if (
        matches[0].get("case_count") != len(expected_case_ids)
        or case_set.get("release_id") != release["release_id"]
        or case_set.get("case_set_id") != case_set_id
        or case_set.get("case_count") != len(expected_case_ids)
        or case_set.get("case_ids") != list(expected_case_ids)
        or case_set.get("master_index_sha256") != digests["index"]
    ):
        _fail("candidate truth selected case-set binding differs")
    if not set(expected_case_ids).issubset(index["case_ids"]):
        _fail("candidate truth master index does not cover the selected case set")
    return CandidateTruthRelease(
        campaign_root=campaign_root,
        release_root=root,
        binding=binding,
        manifest=manifest,
        index=index,
        case_set_id=case_set_id,
        case_ids=tuple(expected_case_ids),
    )


def _truth_case(
    release: CandidateTruthRelease, case_id: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if case_id not in release.case_ids:
        _fail(f"{case_id} is outside the selected candidate truth case set")
    try:
        record, record_sha = _load_case_record(
            release.release_root, case_id, validate_artifact=False
        )
    except NativeProfileTruthError as error:
        raise NativeProfileEvaluationError(str(error)) from error
    location = release.index.get("case_locations", {}).get(case_id)
    if (
        not isinstance(location, Mapping)
        or location.get("case_record_sha256") != record_sha
        or location.get("truth_artifact") != record.get("truth_artifact")
    ):
        _fail(f"{case_id} candidate truth master-index location differs")
    if (
        record.get("schema") != TRUTH_CASE_SCHEMA
        or record.get("status") != "candidate_owner_review_required"
        or record.get("release_id") != TRUTH_RELEASE_ID
        or record.get("truth_boundary")
        != {
            "benchmark_owned": True,
            "participant_visible": False,
            "participant_prediction_artifacts_include_truth": False,
        }
    ):
        _fail(f"{case_id} candidate truth boundary differs")
    try:
        artifact_path, artifact_sha = _load_descriptor(
            {
                key: record["truth_artifact"][key]
                for key in ("file", "sha256", "byte_size")
            },
            label=f"{case_id} candidate truth artifact",
            root=release.release_root,
        )
        with np.load(artifact_path, allow_pickle=False) as archive:
            if archive.files != list(TRUTH_ARRAYS):
                _fail(f"{case_id} candidate truth array inventory/order differs")
            truth_cp = np.array(archive["truth_cp"], copy=True)
            truth_velocity = np.array(archive["reference_velocity_nd"], copy=True)
    except (KeyError, NativeProfileTruthError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise NativeProfileEvaluationError(
            f"cannot read {case_id} candidate truth artifact: {error}"
        ) from error
    if (
        artifact_sha != record["truth_artifact"]["sha256"]
        or sha256_file(artifact_path) != artifact_sha
    ):
        _fail(f"{case_id} candidate truth artifact digest differs")
    if (
        truth_cp.dtype.str != record.get("surface_cp", {}).get("truth_dtype")
        or list(truth_cp.shape) != record.get("surface_cp", {}).get("truth_shape")
        or array_identity(
            truth_cp, namespace="fluidsbench-hiliftaeroml-cp-truth-v1"
        )
        != record.get("surface_cp", {}).get("truth_identity_sha256")
        or truth_velocity.dtype.str
        != record.get("volume_velocity", {}).get("truth_dtype")
        or list(truth_velocity.shape)
        != record.get("volume_velocity", {}).get("truth_shape")
        or array_identity(
            truth_velocity,
            namespace="fluidsbench-hiliftaeroml-velocity-truth-v1",
        )
        != record.get("volume_velocity", {}).get("truth_identity_sha256")
    ):
        _fail(f"{case_id} candidate truth array identity differs")
    return record, truth_cp, truth_velocity


def _validate_cp_alignment(
    case_id: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    alignment = record.get("surface_cp", {}).get("alignment")
    if not isinstance(alignment, Mapping):
        _fail(f"{case_id} candidate Cp alignment is absent")
    support_names = tuple(
        name for name in CP_PUBLISHED_ARRAYS if name != "prediction_cp"
    )
    observed_array_ids = {
        name: array_identity(
            arrays[name], namespace=f"fluidsbench-hiliftaeroml-cp-support-v1/{name}"
        )
        for name in support_names
    }
    if alignment.get("array_identities_sha256") != observed_array_ids:
        _fail(f"{case_id} submitted Cp support differs from hidden-truth alignment")
    topology = (
        "branch_closed",
        "branch_component_code",
        "branch_graph_component_code",
        "branch_plane_piece_code",
        "branch_row_code",
        "branch_segment_offsets",
        "branch_side_code",
        "branch_topology_patch_code",
        "branch_vertex_ids",
        "branch_vertex_offsets",
        "segment_plane_piece_code",
        "segment_vertex_ids",
    )
    if alignment.get("topology_identity_sha256") != _combined_array_identity(
        arrays,
        topology,
        namespace="fluidsbench-hiliftaeroml-cp-topology-v1",
    ):
        _fail(f"{case_id} submitted Cp topology differs from hidden truth")
    if alignment.get("geometry_identity_sha256") != _combined_array_identity(
        arrays,
        ("cut_xyz_in", "segment_lengths_in"),
        namespace="fluidsbench-hiliftaeroml-cp-geometry-v1",
    ):
        _fail(f"{case_id} submitted Cp geometry differs from hidden truth")
    catalogs = metadata.get("code_catalogs")
    if not isinstance(catalogs, Mapping) or alignment.get(
        "code_catalogs_identity_sha256"
    ) != _document_identity(
        "fluidsbench-hiliftaeroml-cp-code-catalogs-v1", dict(catalogs)
    ):
        _fail(f"{case_id} submitted Cp code catalogs differ from hidden truth")
    expected_counts = {
        "vertex_count": metadata.get("vertex_count"),
        "segment_count": metadata.get("segment_count"),
        "branch_count": metadata.get("branch_count"),
        "graph_count": metadata.get("graph_count"),
        "station_rows": metadata.get("station_rows"),
    }
    if any(alignment.get(key) != value for key, value in expected_counts.items()):
        _fail(f"{case_id} submitted Cp topology counts differ from hidden truth")


def _validate_velocity_alignment(
    case_id: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    record: Mapping[str, Any],
    truth: np.ndarray,
) -> None:
    alignment = record.get("volume_velocity", {}).get("alignment")
    if not isinstance(alignment, Mapping):
        _fail(f"{case_id} candidate velocity alignment is absent")
    names = (
        "requested_xyz_in",
        "valid_mask",
        "station_names",
        "station_row_offsets",
        "line_length_weights_in",
    )
    if alignment.get("profile_alignment_identity_sha256") != _combined_array_identity(
        arrays,
        names,
        namespace="fluidsbench-hiliftaeroml-velocity-profile-alignment-v1",
    ):
        _fail(f"{case_id} submitted velocity support differs from hidden truth")
    expected = {
        "station_order": metadata.get("station_order"),
        "row_count": metadata.get("row_count"),
        "valid_row_count": metadata.get("valid_row_count"),
        "invalid_row_count": metadata.get("invalid_row_count"),
    }
    if alignment.get("rows_per_station") != 801 or any(
        alignment.get(key) != value for key, value in expected.items()
    ):
        _fail(f"{case_id} submitted velocity counts differ from hidden truth")
    stations = alignment.get("stations")
    offsets = arrays["station_row_offsets"]
    valid = arrays["valid_mask"]
    if not isinstance(stations, list) or len(stations) != 5:
        _fail(f"{case_id} hidden velocity station alignment differs")
    for index, station in enumerate(stations):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        station_id = str(arrays["station_names"][index])
        expected_station = {
            "station_id": station_id,
            "row_start": start,
            "row_end": end,
            "row_count": end - start,
            "valid_row_count": int(np.count_nonzero(valid[start:end])),
            "invalid_row_count": int(np.count_nonzero(~valid[start:end])),
            "invalid_gap_runs_half_open": _gap_runs(valid[start:end]),
            "coordinate_identity_sha256": array_identity(
                arrays["requested_xyz_in"][start:end],
                namespace=(
                    "fluidsbench-hiliftaeroml-velocity-station-coordinate-v1/"
                    f"{station_id}"
                ),
            ),
            "validity_identity_sha256": array_identity(
                valid[start:end],
                namespace=(
                    "fluidsbench-hiliftaeroml-velocity-station-validity-v1/"
                    f"{station_id}"
                ),
            ),
            "weight_identity_sha256": array_identity(
                arrays["line_length_weights_in"][start:end],
                namespace=(
                    "fluidsbench-hiliftaeroml-velocity-station-weight-v1/"
                    f"{station_id}"
                ),
            ),
            "truth_identity_sha256": array_identity(
                truth[start:end],
                namespace=(
                    "fluidsbench-hiliftaeroml-velocity-station-truth-v1/"
                    f"{station_id}"
                ),
            ),
        }
        expected_station["gap_identity_sha256"] = _document_identity(
            "fluidsbench-hiliftaeroml-velocity-station-gaps-v1",
            expected_station["invalid_gap_runs_half_open"],
        )
        if station != expected_station:
            _fail(f"{case_id}/{station_id} submitted velocity alignment differs")


def _roundoff_safe_centered(
    squared_truth: float,
    truth_integral: float,
    support_weight: float,
    *,
    unit_floor: bool,
) -> float:
    correction = truth_integral * truth_integral / support_weight
    centered = squared_truth - correction
    floor = 1.0 if unit_floor else np.finfo(np.float64).tiny
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        abs(squared_truth), abs(correction), floor
    )
    if centered < -tolerance:
        _fail("profile truth centered sum of squares is materially negative")
    return max(centered, 0.0)


def _cp_r2(arrays: Mapping[str, np.ndarray], truth: np.ndarray) -> float:
    prediction = arrays["prediction_cp"]
    endpoints = arrays["segment_vertex_ids"]
    lengths = arrays["segment_lengths_in"]
    branch_offsets = arrays["branch_segment_offsets"]
    row_codes = arrays["branch_row_code"]
    graph_codes = arrays["branch_graph_component_code"]
    # Graph component codes are local to a Cp station row.  Junction-split
    # branches sharing (row, graph component) are one physical graph and must
    # be centered together, while equal component codes on different rows
    # remain independent.
    states: dict[tuple[int, int], list[float]] = {}
    for branch_index, (row_value, graph_value) in enumerate(
        zip(row_codes.tolist(), graph_codes.tolist(), strict=True)
    ):
        start = int(branch_offsets[branch_index])
        end = int(branch_offsets[branch_index + 1])
        selected = endpoints[start:end]
        segment_lengths = lengths[start:end]
        pred = prediction[selected]
        target = truth[selected]
        error = pred - target
        graph_key = (int(row_value), int(graph_value))
        state = states.setdefault(graph_key, [0.0, 0.0, 0.0, 0.0])
        state[0] += float(np.sum(segment_lengths, dtype=np.float64))
        state[1] += float(
            np.sum(
                segment_lengths
                * (
                    error[:, 0] * error[:, 0]
                    + error[:, 0] * error[:, 1]
                    + error[:, 1] * error[:, 1]
                )
                / 3.0,
                dtype=np.float64,
            )
        )
        state[2] += float(
            np.sum(
                segment_lengths
                * (
                    target[:, 0] * target[:, 0]
                    + target[:, 0] * target[:, 1]
                    + target[:, 1] * target[:, 1]
                )
                / 3.0,
                dtype=np.float64,
            )
        )
        state[3] += float(
            np.sum(
                segment_lengths * (target[:, 0] + target[:, 1]) / 2.0,
                dtype=np.float64,
            )
        )
    sse = 0.0
    sst = 0.0
    for graph in sorted(states):
        length, graph_sse, squared_truth, truth_integral = states[graph]
        if length <= 0.0:
            _fail("Cp graph has no positive physical support")
        sse += graph_sse
        sst += _roundoff_safe_centered(
            squared_truth, truth_integral, length, unit_floor=False
        )
    if not math.isfinite(sse) or not math.isfinite(sst) or sst <= 0.0:
        _fail("Cp profile R2 target variance is zero or invalid")
    return 1.0 - sse / sst


def _velocity_r2(arrays: Mapping[str, np.ndarray], truth: np.ndarray) -> float:
    prediction = np.linalg.norm(arrays["predicted_velocity_nd"], axis=1)
    target = np.linalg.norm(truth, axis=1)
    weights = arrays["line_length_weights_in"]
    offsets = arrays["station_row_offsets"]
    sse = 0.0
    grouped_sst = 0.0
    for start_value, end_value in zip(offsets[:-1], offsets[1:], strict=True):
        start = int(start_value)
        end = int(end_value)
        local_weights = weights[start:end]
        active = (
            (local_weights > 0.0)
            & np.isfinite(prediction[start:end])
            & np.isfinite(target[start:end])
        )
        if not np.any(active):
            _fail("velocity station has no positive physical support")
        weight = local_weights[active]
        pred = prediction[start:end][active]
        target_values = target[start:end][active]
        error = pred - target_values
        weight_sum = float(np.sum(weight, dtype=np.float64))
        station_sse = float(np.sum(weight * error * error, dtype=np.float64))
        squared_truth = float(
            np.sum(weight * target_values * target_values, dtype=np.float64)
        )
        truth_integral = float(np.sum(weight * target_values, dtype=np.float64))
        sse += station_sse
        grouped_sst += _roundoff_safe_centered(
            squared_truth, truth_integral, weight_sum, unit_floor=True
        )
    if not math.isfinite(sse) or not math.isfinite(grouped_sst) or grouped_sst <= 0.0:
        _fail("velocity profile R2 target variance is zero or invalid")
    return 1.0 - sse / grouped_sst


def score_native_profile_case(
    *,
    profiles_root: Path,
    case: Mapping[str, Any],
    release: CandidateTruthRelease,
) -> dict[str, float]:
    """Recompute one case's two official profile diagnostics."""

    case_id = case.get("case_id")
    if not isinstance(case_id, str):
        _fail("profile case ID is absent")
    record, truth_cp, truth_velocity = _truth_case(release, case_id)
    surface = case.get("surface_cp")
    volume = case.get("volume_velocity")
    if not isinstance(surface, Mapping) or not isinstance(volume, Mapping):
        _fail(f"{case_id} prediction profile metadata is incomplete")
    cp_artifact = surface.get("artifact")
    velocity_artifact = volume.get("artifact")
    if not isinstance(cp_artifact, Mapping) or not isinstance(
        velocity_artifact, Mapping
    ):
        _fail(f"{case_id} prediction profile artifact descriptors are absent")
    artifact_keys = {
        "format",
        "file",
        "sha256",
        "byte_size",
        "source_native_npz_sha256",
        "ground_truth_included",
    }
    for label, descriptor in (
        ("Cp", cp_artifact),
        ("velocity", velocity_artifact),
    ):
        if (
            set(descriptor) != artifact_keys
            or descriptor.get("format") != "numpy-npz-v1"
            or descriptor.get("ground_truth_included") is not False
        ):
            _fail(f"{case_id} prediction {label} artifact descriptor differs")
        _require_sha(
            descriptor.get("source_native_npz_sha256"),
            f"{case_id} prediction {label} source digest",
        )
    cp_relative = f"artifacts/{case_id}/surface-cp-predictions.npz"
    velocity_relative = f"artifacts/{case_id}/volume-velocity-predictions.npz"
    if cp_artifact.get("file") != cp_relative or velocity_artifact.get(
        "file"
    ) != velocity_relative:
        _fail(f"{case_id} prediction profile artifact path differs")
    cp_path = _safe_child(profiles_root, cp_relative, f"{case_id} Cp artifact")
    velocity_path = _safe_child(
        profiles_root, velocity_relative, f"{case_id} velocity artifact"
    )
    try:
        cp_arrays = validate_prediction_npz(
            cp_path,
            expected_arrays=CP_PUBLISHED_ARRAYS,
            expected_sha256=cp_artifact.get("sha256"),
            metadata=surface,
        )
        velocity_arrays = validate_prediction_npz(
            velocity_path,
            expected_arrays=VELOCITY_PUBLISHED_ARRAYS,
            expected_sha256=velocity_artifact.get("sha256"),
            metadata=volume,
        )
    except NativeProfileError as error:
        raise NativeProfileEvaluationError(str(error)) from error
    if cp_path.stat().st_size != cp_artifact.get("byte_size") or velocity_path.stat(
    ).st_size != velocity_artifact.get("byte_size"):
        _fail(f"{case_id} prediction profile artifact byte size differs")
    if truth_cp.shape != cp_arrays["prediction_cp"].shape:
        _fail(f"{case_id} Cp prediction/truth shape differs")
    if truth_velocity.shape != velocity_arrays["predicted_velocity_nd"].shape:
        _fail(f"{case_id} velocity prediction/truth shape differs")
    _validate_cp_alignment(case_id, cp_arrays, surface, record)
    _validate_velocity_alignment(
        case_id, velocity_arrays, volume, record, truth_velocity
    )
    return {
        "cp_cut_r2": _cp_r2(cp_arrays, truth_cp),
        "velocity_profile_r2": _velocity_r2(velocity_arrays, truth_velocity),
    }


def score_native_profile_directory(
    *,
    profiles_root: Path,
    release_root: Path,
    candidate_declaration: Mapping[str, Any],
    submission_id: str,
    split_id: str,
    case_set_id: str,
    expected_case_ids: Sequence[str],
    binding_path: Path = CANDIDATE_BINDING_PATH,
) -> dict[str, dict[str, float]]:
    """Score a complete prediction-only profile directory against hidden truth."""

    if profiles_root.is_symlink() or not profiles_root.is_dir():
        _fail("prediction profile root must be a regular non-symlink directory")
    release = open_candidate_truth_release(
        release_root=release_root,
        candidate_declaration=candidate_declaration,
        expected_case_ids=expected_case_ids,
        case_set_id=case_set_id,
        binding_path=binding_path,
    )
    try:
        index, _ = load_json(profiles_root / "index.json", label="prediction profile index")
    except NativeProfileTruthError as error:
        raise NativeProfileEvaluationError(str(error)) from error
    expected_index = {
        "format": PROFILE_FORMAT,
        "contract_id": PROFILE_CONTRACT_ID,
        "contract_sha256": PROFILE_CONTRACT_SHA256,
        "submission_id": submission_id,
        "dataset_id": "hiliftaeroml",
        "split_id": split_id,
        "case_set_id": case_set_id,
        "case_count": len(expected_case_ids),
        "case_id_status": "official",
    }
    if any(index.get(key) != value for key, value in expected_index.items()):
        _fail("prediction profile index identity differs")
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        _fail("prediction profile index has no chunks")
    scores: dict[str, dict[str, float]] = {}
    referenced_chunks: set[str] = set()
    referenced_artifacts: set[str] = set()
    indexed_case_ids: list[str] = []
    for descriptor in chunks:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "file",
            "case_ids",
            "sha256",
        }:
            _fail("prediction profile chunk descriptor differs")
        filename = descriptor.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            _fail("prediction profile chunk filename is invalid")
        if filename in referenced_chunks:
            _fail("prediction profile chunk is referenced more than once")
        referenced_chunks.add(filename)
        path = _safe_child(profiles_root, filename, "prediction profile chunk")
        if sha256_file(path) != _require_sha(
            descriptor.get("sha256"), "prediction profile chunk digest"
        ):
            _fail(f"prediction profile chunk {filename} SHA-256 differs")
        try:
            chunk, _ = load_json(path, label=f"prediction profile chunk {filename}")
        except NativeProfileTruthError as error:
            raise NativeProfileEvaluationError(str(error)) from error
        identity = {
            "schema": PROFILE_CHUNK_SCHEMA,
            "schema_version": "1.0",
            "format": PROFILE_FORMAT,
            "contract_id": PROFILE_CONTRACT_ID,
            "contract_sha256": PROFILE_CONTRACT_SHA256,
            "submission_id": submission_id,
            "dataset_id": "hiliftaeroml",
            "split_id": split_id,
            "case_set_id": case_set_id,
        }
        if any(chunk.get(key) != value for key, value in identity.items()):
            _fail(f"prediction profile chunk {filename} identity differs")
        cases = chunk.get("cases")
        if not isinstance(cases, list):
            _fail(f"prediction profile chunk {filename} cases are absent")
        case_ids = [
            case.get("case_id") if isinstance(case, Mapping) else None for case in cases
        ]
        if case_ids != descriptor.get("case_ids"):
            _fail(f"prediction profile chunk {filename} case order differs")
        indexed_case_ids.extend(case_ids)
        for case in cases:
            if not isinstance(case, Mapping):
                _fail(f"prediction profile chunk {filename} case is invalid")
            case_id = case.get("case_id")
            if case_id in scores:
                _fail(f"prediction profile case {case_id} is duplicated")
            scores[str(case_id)] = score_native_profile_case(
                profiles_root=profiles_root,
                case=case,
                release=release,
            )
            referenced_artifacts.update(
                {
                    f"artifacts/{case_id}/surface-cp-predictions.npz",
                    f"artifacts/{case_id}/volume-velocity-predictions.npz",
                }
            )
    if indexed_case_ids != list(expected_case_ids):
        _fail("prediction profile case coverage/order differs from the selected split")
    actual_chunks = {path.name for path in profiles_root.glob("chunk-*.json")}
    if actual_chunks != referenced_chunks:
        _fail("prediction profile chunk inventory differs")
    actual_artifacts = {
        path.relative_to(profiles_root).as_posix()
        for path in (profiles_root / "artifacts").glob("**/*.npz")
        if path.is_file() and not path.is_symlink()
    }
    if actual_artifacts != referenced_artifacts:
        _fail("prediction profile artifact inventory differs")
    expected_files = {"index.json", *referenced_chunks, *referenced_artifacts}
    actual_files = {
        path.relative_to(profiles_root).as_posix()
        for path in profiles_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        _fail("prediction profile file inventory differs")
    return scores


__all__ = [
    "CANDIDATE_BINDING_PATH",
    "CANDIDATE_TRUTH_STATUS",
    "CANDIDATE_USAGE",
    "CandidateTruthRelease",
    "NativeProfileEvaluationError",
    "load_candidate_truth_binding",
    "open_candidate_truth_release",
    "score_native_profile_case",
    "score_native_profile_directory",
]
