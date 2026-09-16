"""Deterministic benchmark-owned HiLiftAeroML native profile truth release.

Participant profile artifacts are prediction-only.  This module creates the
separate hidden truth side of that contract from evaluator-native, receipt-
bound Cp and velocity profile artifacts.  Every physical case is stored once;
split/case-set indexes only reference the shared case universe.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reference.hiliftaeroml.native_profiles import (
    CONTRACT_PATH,
    CP_SOURCE_ARRAYS,
    PROFILE_CONTRACT_ID,
    PROFILE_CONTRACT_SHA256,
    ROWS,
    ROWS_PER_STATION,
    SAFE_CASE_ID,
    STATIONS,
    NativeProfileError,
    canonical_json_bytes,
    load_json,
    sha256_file,
    validate_cp_source,
    validate_velocity_source,
)

TRUTH_FORMAT = "fluidsbench-hiliftaeroml-native-profile-truth-v1-candidate"
TRUTH_ARRAYS = ("truth_cp", "reference_velocity_nd")
TRUTH_CASE_SCHEMA = "hiliftaeroml-native-profile-truth-case-v1-candidate"
TRUTH_CHUNK_SCHEMA = "hiliftaeroml-native-profile-truth-chunk-v1-candidate"
TRUTH_INDEX_SCHEMA = "hiliftaeroml-native-profile-truth-index-v1-candidate"
TRUTH_MANIFEST_SCHEMA = "hiliftaeroml-native-profile-truth-manifest-v1-candidate"
TRUTH_CASE_SET_SCHEMA = "hiliftaeroml-native-profile-truth-case-set-index-v1-candidate"
TRUTH_SOURCE_INDEX_SCHEMA = (
    "hiliftaeroml-native-profile-truth-source-index-v1-candidate"
)
TRUTH_RELEASE_ID = "hiliftaeroml-native-profile-truth-v1-candidate"
EXPECTED_UNIQUE_CASE_COUNT = 1355
EXPECTED_CASE_SET_COUNT = 8


class NativeProfileTruthError(ValueError):
    """Raised when hidden native profile truth does not close exactly."""


@dataclass(frozen=True)
class CaseUniverse:
    """One deduplicated case universe plus its exact case-set memberships."""

    case_ids: tuple[str, ...]
    case_sets: tuple[tuple[str, tuple[str, ...], str], ...]
    support_manifest_sha256: str
    case_universe_sha256: str


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeProfileTruthError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise NativeProfileTruthError(f"{label} must be a {qualifier} integer")
    return value


def _descriptor(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise NativeProfileTruthError(f"artifact must be a regular file: {path}")
    file_value = (
        str(path.resolve()) if root is None else path.relative_to(root).as_posix()
    )
    return {
        "file": file_value,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _load_descriptor(
    value: Any,
    *,
    label: str,
    root: Path | None = None,
    require_absolute: bool = False,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"file", "sha256", "byte_size"}:
        raise NativeProfileTruthError(f"{label} descriptor keys differ")
    file_value = value.get("file")
    if not isinstance(file_value, str) or not file_value:
        raise NativeProfileTruthError(f"{label}.file is invalid")
    path = Path(file_value)
    if require_absolute and not path.is_absolute():
        raise NativeProfileTruthError(f"{label}.file must be absolute")
    if root is not None:
        if path.is_absolute() or ".." in path.parts:
            raise NativeProfileTruthError(f"{label}.file must be release-relative")
        resolved_root = root.resolve()
        unresolved_path = resolved_root / path
        path = unresolved_path.resolve()
        if path != resolved_root and resolved_root not in path.parents:
            raise NativeProfileTruthError(
                f"{label}.file resolves outside the release root"
            )
        cursor = resolved_root
        for part in unresolved_path.relative_to(resolved_root).parts:
            cursor /= part
            if cursor.is_symlink():
                raise NativeProfileTruthError(
                    f"{label}.file traverses a release symlink"
                )
    if not path.is_file() or path.is_symlink():
        raise NativeProfileTruthError(f"{label} is not a regular file: {path}")
    digest = _sha(value.get("sha256"), f"{label}.sha256")
    size = _positive_int(value.get("byte_size"), f"{label}.byte_size")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise NativeProfileTruthError(f"{label} live bytes differ")
    return path, digest


def _document_identity(namespace: str, body: Any) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(body))
    return digest.hexdigest()


def _case_universe_identity(case_ids: Sequence[str]) -> str:
    return _document_identity(
        "fluidsbench-hiliftaeroml-case-universe-v1",
        {"case_ids": list(case_ids)},
    )


def _normalized_array(value: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise NativeProfileTruthError("object arrays are forbidden")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array) | np.isnan(array)):
        raise NativeProfileTruthError("infinite array values are forbidden")
    dtype = array.dtype
    if dtype.kind in "iufc" and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    return dtype.str, tuple(int(size) for size in array.shape), array.tobytes(order="C")


def array_identity(value: np.ndarray, *, namespace: str) -> str:
    """Hash dtype, shape and canonical little-endian C-order array bytes."""

    dtype, shape, payload = _normalized_array(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    encoded_dtype = dtype.encode("ascii")
    digest.update(struct.pack(">H", len(encoded_dtype)))
    digest.update(encoded_dtype)
    digest.update(struct.pack(">H", len(shape)))
    for size in shape:
        digest.update(struct.pack(">Q", size))
    digest.update(payload)
    return digest.hexdigest()


def _combined_array_identity(
    arrays: Mapping[str, np.ndarray], names: Sequence[str], *, namespace: str
) -> str:
    body = {
        "arrays": [
            {
                "name": name,
                "identity_sha256": array_identity(
                    arrays[name], namespace=f"{namespace}/{name}"
                ),
            }
            for name in names
        ]
    }
    return _document_identity(namespace, body)


def _gap_runs(mask: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = []
    start: int | None = None
    for index, valid in enumerate(mask.tolist()):
        if not valid and start is None:
            start = index
        elif valid and start is not None:
            result.append([start, index])
            start = None
    if start is not None:
        result.append([start, len(mask)])
    return result


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != TRUTH_ARRAYS:
        raise NativeProfileTruthError("truth NPZ array inventory/order differs")
    raw = io.BytesIO()
    with zipfile.ZipFile(
        raw,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in TRUTH_ARRAYS:
            member = io.BytesIO()
            np.lib.format.write_array(
                member, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                member.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return raw.getvalue()


def _atomic_bytes(path: Path, payload: bytes, *, check: bool) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if check:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise NativeProfileTruthError(f"determinism check differs: {path}")
        return digest
    if path.exists() or path.is_symlink():
        raise NativeProfileTruthError(f"refusing to overwrite release artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise NativeProfileTruthError(
                f"refusing concurrent overwrite of release artifact: {path}"
            ) from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _atomic_json(path: Path, body: Mapping[str, Any], *, check: bool) -> str:
    return _atomic_bytes(path, canonical_json_bytes(body), check=check)


def load_case_universe(
    support_manifest_path: Path,
    *,
    enforce_official_scope: bool = True,
) -> CaseUniverse:
    manifest, manifest_sha = load_json(
        support_manifest_path, label="HiLift scoring-support manifest"
    )
    case_sets_raw = manifest.get("case_sets")
    if (
        manifest.get("dataset_id") != "hiliftaeroml"
        or not isinstance(case_sets_raw, list)
        or not case_sets_raw
    ):
        raise NativeProfileTruthError("HiLift scoring-support manifest differs")
    release_root = support_manifest_path.parent
    case_sets: list[tuple[str, tuple[str, ...], str]] = []
    union: set[str] = set()
    for descriptor in case_sets_raw:
        if not isinstance(descriptor, Mapping):
            raise NativeProfileTruthError("case-set descriptor is invalid")
        case_set_id = descriptor.get("id")
        index_file = descriptor.get("index_file")
        if (
            not isinstance(case_set_id, str)
            or not case_set_id.startswith("caseset-")
            or not isinstance(index_file, str)
            or Path(index_file).is_absolute()
            or ".." in Path(index_file).parts
        ):
            raise NativeProfileTruthError("case-set identity/path is invalid")
        index_path = release_root / index_file
        index, index_sha = load_json(index_path, label=f"{case_set_id} index")
        if index_sha != _sha(
            descriptor.get("index_sha256"), f"{case_set_id} index digest"
        ):
            raise NativeProfileTruthError(f"{case_set_id} index digest differs")
        chunks = index.get("chunks")
        if (
            index.get("case_set_id") != case_set_id
            or not isinstance(chunks, list)
            or not chunks
        ):
            raise NativeProfileTruthError(f"{case_set_id} index differs")
        case_ids: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping) or not isinstance(
                chunk.get("case_ids"), list
            ):
                raise NativeProfileTruthError(f"{case_set_id} chunk index differs")
            selected = chunk["case_ids"]
            if not all(
                isinstance(case, str) and SAFE_CASE_ID.fullmatch(case)
                for case in selected
            ):
                raise NativeProfileTruthError(
                    f"{case_set_id} contains an invalid case ID"
                )
            if chunk.get("case_count") != len(selected):
                raise NativeProfileTruthError(f"{case_set_id} chunk count differs")
            case_ids.extend(selected)
        expected_count = _positive_int(
            descriptor.get("case_count"), f"{case_set_id} count"
        )
        if (
            len(case_ids) != expected_count
            or index.get("case_count") != expected_count
            or len(case_ids) != len(set(case_ids))
        ):
            raise NativeProfileTruthError(f"{case_set_id} case inventory differs")
        case_tuple = tuple(case_ids)
        case_sets.append((case_set_id, case_tuple, index_sha))
        union.update(case_tuple)
    ordered_union = tuple(sorted(union))
    if enforce_official_scope and (
        len(case_sets) != EXPECTED_CASE_SET_COUNT
        or len(ordered_union) != EXPECTED_UNIQUE_CASE_COUNT
    ):
        raise NativeProfileTruthError(
            "official HiLift profile scope must contain exactly eight case sets and "
            "1,355 unique cases"
        )
    return CaseUniverse(
        case_ids=ordered_union,
        case_sets=tuple(case_sets),
        support_manifest_sha256=manifest_sha,
        case_universe_sha256=_case_universe_identity(ordered_union),
    )


def _profile_paths(case_root: Path) -> dict[str, Path]:
    return {
        "cp_metrics": case_root
        / "surface_submission_stream"
        / "cp_profile_metrics.json",
        "cp_native": case_root / "surface_submission_stream" / "cp_cut_values.npz",
        "velocity_metrics": case_root
        / "volume_submission_stream"
        / "velocity_profile_metrics.json",
        "velocity_native": case_root
        / "volume_submission_stream"
        / "velocity_profiles.npz",
    }


def build_source_index(
    *,
    universe: CaseUniverse,
    outputs_roots: Sequence[Path],
    output_path: Path,
    check: bool = False,
) -> dict[str, Any]:
    """Inventory materialized evaluator-native sources without publishing paths."""

    if not outputs_roots:
        raise NativeProfileTruthError(
            "at least one evaluator-native outputs root is required"
        )
    roots = [root.resolve() for root in outputs_roots]
    cases: dict[str, Any] = {}
    missing: list[str] = []
    for case_id in universe.case_ids:
        candidates: list[tuple[Path, dict[str, Path], dict[str, Any]]] = []
        for root in roots:
            paths = _profile_paths(root / case_id)
            if all(path.is_file() and not path.is_symlink() for path in paths.values()):
                artifacts = {key: _descriptor(path) for key, path in paths.items()}
                candidates.append((root, paths, artifacts))
        if not candidates:
            missing.append(case_id)
            continue
        source_identities = {
            tuple(
                (key, descriptor["sha256"], descriptor["byte_size"])
                for key, descriptor in sorted(artifacts.items())
            )
            for _, _, artifacts in candidates
        }
        if len(source_identities) != 1:
            roots_text = ", ".join(str(root) for root, _, _ in candidates)
            raise NativeProfileTruthError(
                f"{case_id} has conflicting complete source roots: {roots_text}"
            )
        root, _, artifacts = candidates[0]
        cases[case_id] = {
            "outputs_root": str(root),
            "case_output_root": str((root / case_id).resolve()),
            "artifacts": artifacts,
            "source_selection": "first_byte-identical_complete_root_in_declared_order",
            "duplicate_complete_root_count": len(candidates),
        }
    body = {
        "schema": TRUTH_SOURCE_INDEX_SCHEMA,
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "profile_contract_id": PROFILE_CONTRACT_ID,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "support_manifest_sha256": universe.support_manifest_sha256,
        "case_universe_sha256": universe.case_universe_sha256,
        "expected_case_count": len(universe.case_ids),
        "available_case_count": len(cases),
        "missing_case_count": len(missing),
        "case_ids": list(universe.case_ids),
        "missing_case_ids": missing,
        "cases": cases,
    }
    _atomic_json(output_path, body, check=check)
    return body


def load_source_index(
    path: Path,
    *,
    universe: CaseUniverse,
    require_complete: bool,
) -> tuple[dict[str, Any], str]:
    source, source_sha = load_json(path, label="HiLift profile truth source index")
    cases = source.get("cases")
    missing = source.get("missing_case_ids")
    if (
        source.get("schema") != TRUTH_SOURCE_INDEX_SCHEMA
        or source.get("dataset_id") != "hiliftaeroml"
        or source.get("release_id") != TRUTH_RELEASE_ID
        or source.get("profile_contract_id") != PROFILE_CONTRACT_ID
        or source.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or source.get("support_manifest_sha256") != universe.support_manifest_sha256
        or source.get("case_universe_sha256") != universe.case_universe_sha256
        or source.get("case_ids") != list(universe.case_ids)
        or not isinstance(cases, dict)
        or not isinstance(missing, list)
    ):
        raise NativeProfileTruthError("profile truth source index binding differs")
    if set(cases).intersection(missing) or set(cases).union(missing) != set(
        universe.case_ids
    ):
        raise NativeProfileTruthError(
            "source index available/missing partition differs"
        )
    if (
        source.get("available_case_count") != len(cases)
        or source.get("missing_case_count") != len(missing)
        or source.get("expected_case_count") != len(universe.case_ids)
    ):
        raise NativeProfileTruthError("source index counts differ")
    if require_complete and (missing or source.get("status") != "complete"):
        raise NativeProfileTruthError(
            f"profile truth source index is incomplete ({len(missing)} cases missing)"
        )
    return source, source_sha


def _source_case_paths(
    source: Mapping[str, Any], case_id: str
) -> tuple[dict[str, Path], dict[str, str]]:
    cases = source.get("cases")
    if not isinstance(cases, Mapping) or not isinstance(cases.get(case_id), Mapping):
        raise NativeProfileTruthError(f"source index lacks {case_id}")
    entry = cases[case_id]
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "cp_metrics",
        "cp_native",
        "velocity_metrics",
        "velocity_native",
    }:
        raise NativeProfileTruthError(f"{case_id} source artifact inventory differs")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for key in sorted(artifacts):
        path, digest = _load_descriptor(
            artifacts[key], label=f"{case_id}/{key}", require_absolute=True
        )
        paths[key] = path
        digests[key] = digest
    case_root = entry.get("case_output_root")
    if not isinstance(case_root, str) or not Path(case_root).is_absolute():
        raise NativeProfileTruthError(f"{case_id} source root is invalid")
    expected = _profile_paths(Path(case_root))
    if any(paths[key].resolve() != expected[key].resolve() for key in paths):
        raise NativeProfileTruthError(f"{case_id} source paths differ from its root")
    return paths, digests


def _cp_alignment(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    support_names = tuple(
        name for name in CP_SOURCE_ARRAYS if name not in {"prediction_cp", "truth_cp"}
    )
    array_ids = {
        name: array_identity(
            arrays[name], namespace=f"fluidsbench-hiliftaeroml-cp-support-v1/{name}"
        )
        for name in support_names
    }
    catalogs = metadata.get("code_catalogs")
    if not isinstance(catalogs, Mapping):
        raise NativeProfileTruthError("Cp code catalogs are absent")
    catalogs_sha = _document_identity(
        "fluidsbench-hiliftaeroml-cp-code-catalogs-v1", dict(catalogs)
    )
    return {
        "topology_identity_sha256": _combined_array_identity(
            arrays,
            (
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
            ),
            namespace="fluidsbench-hiliftaeroml-cp-topology-v1",
        ),
        "geometry_identity_sha256": _combined_array_identity(
            arrays,
            ("cut_xyz_in", "segment_lengths_in"),
            namespace="fluidsbench-hiliftaeroml-cp-geometry-v1",
        ),
        "array_identities_sha256": array_ids,
        "code_catalogs_identity_sha256": catalogs_sha,
        "station_rows": list(ROWS),
        "vertex_count": int(metadata["vertex_count"]),
        "segment_count": int(metadata["segment_count"]),
        "branch_count": int(metadata["branch_count"]),
        "graph_count": int(metadata["graph_count"]),
    }


def _velocity_alignment(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    valid = arrays["valid_mask"]
    offsets = arrays["station_row_offsets"]
    truth = arrays["reference_velocity_nd"]
    stations: list[dict[str, Any]] = []
    for station_index, station_id in enumerate(STATIONS):
        start = int(offsets[station_index])
        end = int(offsets[station_index + 1])
        station_valid = valid[start:end]
        gaps = _gap_runs(station_valid)
        station = {
            "station_id": station_id,
            "row_start": start,
            "row_end": end,
            "row_count": end - start,
            "valid_row_count": int(np.count_nonzero(station_valid)),
            "invalid_row_count": int(np.count_nonzero(~station_valid)),
            "invalid_gap_runs_half_open": gaps,
            "coordinate_identity_sha256": array_identity(
                arrays["requested_xyz_in"][start:end],
                namespace=f"fluidsbench-hiliftaeroml-velocity-station-coordinate-v1/{station_id}",
            ),
            "validity_identity_sha256": array_identity(
                station_valid,
                namespace=f"fluidsbench-hiliftaeroml-velocity-station-validity-v1/{station_id}",
            ),
            "weight_identity_sha256": array_identity(
                arrays["line_length_weights_in"][start:end],
                namespace=f"fluidsbench-hiliftaeroml-velocity-station-weight-v1/{station_id}",
            ),
            "truth_identity_sha256": array_identity(
                truth[start:end],
                namespace=f"fluidsbench-hiliftaeroml-velocity-station-truth-v1/{station_id}",
            ),
        }
        station["gap_identity_sha256"] = _document_identity(
            "fluidsbench-hiliftaeroml-velocity-station-gaps-v1", gaps
        )
        stations.append(station)
    return {
        "profile_alignment_identity_sha256": _combined_array_identity(
            arrays,
            (
                "requested_xyz_in",
                "valid_mask",
                "station_names",
                "station_row_offsets",
                "line_length_weights_in",
            ),
            namespace="fluidsbench-hiliftaeroml-velocity-profile-alignment-v1",
        ),
        "station_order": list(STATIONS),
        "rows_per_station": ROWS_PER_STATION,
        "row_count": int(metadata["row_count"]),
        "valid_row_count": int(metadata["valid_row_count"]),
        "invalid_row_count": int(metadata["invalid_row_count"]),
        "stations": stations,
    }


def build_case_truth_from_arrays(
    *,
    case_id: str,
    cp_arrays: Mapping[str, np.ndarray],
    cp_metadata: Mapping[str, Any],
    velocity_arrays: Mapping[str, np.ndarray],
    velocity_metadata: Mapping[str, Any],
    surface_source: Mapping[str, Any],
    volume_source: Mapping[str, Any],
    authority_identity_sha256: str,
    output_root: Path,
    check: bool = False,
) -> dict[str, Any]:
    """Write one truth case from prediction-free, prerequisite-derived arrays."""

    if SAFE_CASE_ID.fullmatch(case_id) is None:
        raise NativeProfileTruthError(f"invalid case ID {case_id!r}")
    if sha256_file(CONTRACT_PATH) != PROFILE_CONTRACT_SHA256:
        raise NativeProfileTruthError("native profile contract SHA-256 changed")
    authority_identity_sha256 = _sha(
        authority_identity_sha256, f"{case_id} authority identity"
    )
    truth_cp = np.asarray(cp_arrays.get("truth_cp"))
    reference_velocity = np.asarray(velocity_arrays.get("reference_velocity_nd"))
    valid = np.asarray(velocity_arrays.get("valid_mask"))
    if truth_cp.dtype != np.dtype("<f8") or truth_cp.ndim != 1:
        raise NativeProfileTruthError(f"{case_id} Cp truth dtype/shape differs")
    if reference_velocity.dtype != np.dtype("<f8") or reference_velocity.shape != (
        len(STATIONS) * ROWS_PER_STATION,
        3,
    ):
        raise NativeProfileTruthError(f"{case_id} velocity truth dtype/shape differs")
    if valid.dtype != np.dtype(np.bool_) or valid.shape != reference_velocity.shape[:1]:
        raise NativeProfileTruthError(f"{case_id} velocity validity differs")
    if not np.all(np.isfinite(truth_cp)):
        raise NativeProfileTruthError(f"{case_id} Cp truth contains a non-finite value")
    if not np.all(np.isfinite(reference_velocity[valid])) or not np.all(
        np.isnan(reference_velocity[~valid])
    ):
        raise NativeProfileTruthError(f"{case_id} velocity truth gap encoding differs")
    artifact_payload = _deterministic_npz_bytes(
        {"truth_cp": truth_cp, "reference_velocity_nd": reference_velocity}
    )
    artifact_path = output_root / "cases" / f"{case_id}.npz"
    artifact_sha = _atomic_bytes(artifact_path, artifact_payload, check=check)
    body: dict[str, Any] = {
        "schema": TRUTH_CASE_SCHEMA,
        "schema_version": 1,
        "format": TRUTH_FORMAT,
        "status": "candidate_owner_review_required",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "case_id": case_id,
        "profile_contract_id": PROFILE_CONTRACT_ID,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "truth_artifact": {
            "file": artifact_path.relative_to(output_root).as_posix(),
            "sha256": artifact_sha,
            "byte_size": len(artifact_payload),
            "format": "numpy-npz-v1",
            "array_order": list(TRUTH_ARRAYS),
            "participant_visible": False,
        },
        "surface_cp": {
            "truth_identity_sha256": array_identity(
                truth_cp, namespace="fluidsbench-hiliftaeroml-cp-truth-v1"
            ),
            "truth_dtype": truth_cp.dtype.str,
            "truth_shape": list(truth_cp.shape),
            "alignment": _cp_alignment(cp_arrays, cp_metadata),
            "source": dict(surface_source),
        },
        "volume_velocity": {
            "truth_identity_sha256": array_identity(
                reference_velocity,
                namespace="fluidsbench-hiliftaeroml-velocity-truth-v1",
            ),
            "truth_dtype": reference_velocity.dtype.str,
            "truth_shape": list(reference_velocity.shape),
            "alignment": _velocity_alignment(velocity_arrays, velocity_metadata),
            "source": dict(volume_source),
        },
        "truth_authority": {
            "kind": "benchmark_truth_only_prerequisite_replay",
            "authority_identity_sha256": authority_identity_sha256,
            "prediction_bearing_evaluator_outputs_used_as_source": False,
        },
        "truth_boundary": {
            "benchmark_owned": True,
            "participant_visible": False,
            "participant_prediction_artifacts_include_truth": False,
        },
    }
    body["case_identity_sha256"] = _document_identity(
        "fluidsbench-hiliftaeroml-native-profile-truth-case-v1", body
    )
    _atomic_json(output_root / "case-records" / f"{case_id}.json", body, check=check)
    return body


def build_case_truth(
    *,
    case_id: str,
    source: Mapping[str, Any],
    output_root: Path,
    check: bool = False,
) -> dict[str, Any]:
    if SAFE_CASE_ID.fullmatch(case_id) is None:
        raise NativeProfileTruthError(f"invalid case ID {case_id!r}")
    if sha256_file(CONTRACT_PATH) != PROFILE_CONTRACT_SHA256:
        raise NativeProfileTruthError("native profile contract SHA-256 changed")
    paths, source_digests = _source_case_paths(source, case_id)
    try:
        cp_arrays, cp_metadata = validate_cp_source(
            metrics_path=paths["cp_metrics"],
            npz_path=paths["cp_native"],
            expected_metrics_sha256=source_digests["cp_metrics"],
            expected_npz_sha256=source_digests["cp_native"],
        )
        velocity_arrays, velocity_metadata = validate_velocity_source(
            case_id=case_id,
            metrics_path=paths["velocity_metrics"],
            npz_path=paths["velocity_native"],
            expected_metrics_sha256=source_digests["velocity_metrics"],
            expected_npz_sha256=source_digests["velocity_native"],
        )
    except NativeProfileError as error:
        raise NativeProfileTruthError(str(error)) from error
    truth_cp = np.array(cp_arrays["truth_cp"], copy=True)
    reference_velocity = np.array(velocity_arrays["reference_velocity_nd"], copy=True)
    valid = velocity_arrays["valid_mask"]
    if not np.all(np.isfinite(truth_cp)):
        raise NativeProfileTruthError(f"{case_id} Cp truth contains a non-finite value")
    if not np.all(np.isfinite(reference_velocity[valid])) or not np.all(
        np.isnan(reference_velocity[~valid])
    ):
        raise NativeProfileTruthError(f"{case_id} velocity truth gap encoding differs")
    artifact_payload = _deterministic_npz_bytes(
        {"truth_cp": truth_cp, "reference_velocity_nd": reference_velocity}
    )
    artifact_path = output_root / "cases" / f"{case_id}.npz"
    artifact_sha = _atomic_bytes(artifact_path, artifact_payload, check=check)
    cp_alignment = _cp_alignment(cp_arrays, cp_metadata)
    velocity_alignment = _velocity_alignment(velocity_arrays, velocity_metadata)
    body: dict[str, Any] = {
        "schema": TRUTH_CASE_SCHEMA,
        "schema_version": 1,
        "format": TRUTH_FORMAT,
        "status": "candidate_owner_review_required",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "case_id": case_id,
        "profile_contract_id": PROFILE_CONTRACT_ID,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "truth_artifact": {
            "file": artifact_path.relative_to(output_root).as_posix(),
            "sha256": artifact_sha,
            "byte_size": len(artifact_payload),
            "format": "numpy-npz-v1",
            "array_order": list(TRUTH_ARRAYS),
            "participant_visible": False,
        },
        "surface_cp": {
            "truth_identity_sha256": array_identity(
                truth_cp, namespace="fluidsbench-hiliftaeroml-cp-truth-v1"
            ),
            "truth_dtype": truth_cp.dtype.str,
            "truth_shape": list(truth_cp.shape),
            "alignment": cp_alignment,
            "source": {
                "metrics_schema": cp_metadata["source_metrics_schema"],
                "metrics_sha256": cp_metadata["source_metrics_sha256"],
                "native_npz_sha256": cp_metadata["source_npz_sha256"],
                "run_fingerprint": cp_metadata["run_fingerprint"],
                "stencil_content_sha256": cp_metadata["stencil_content_sha256"],
                "stencil_payload_sha256": cp_metadata["stencil_payload_sha256"],
                "stencil_semantic_sha256": cp_metadata["stencil_semantic_sha256"],
            },
        },
        "volume_velocity": {
            "truth_identity_sha256": array_identity(
                reference_velocity,
                namespace="fluidsbench-hiliftaeroml-velocity-truth-v1",
            ),
            "truth_dtype": reference_velocity.dtype.str,
            "truth_shape": list(reference_velocity.shape),
            "alignment": velocity_alignment,
            "source": {
                "metrics_schema": velocity_metadata["source_metrics_schema"],
                "metrics_sha256": velocity_metadata["source_metrics_sha256"],
                "native_npz_sha256": velocity_metadata["source_npz_sha256"],
                "run_fingerprint": velocity_metadata["run_fingerprint"],
                "capture_plan_fingerprint": velocity_metadata[
                    "capture_plan_fingerprint"
                ],
                "stencil_record_sha256": velocity_metadata["stencil_record_sha256"],
                "ref_values_csv_sha256": velocity_metadata["ref_values_csv_sha256"],
            },
        },
        "truth_boundary": {
            "benchmark_owned": True,
            "participant_visible": False,
            "participant_prediction_artifacts_include_truth": False,
        },
    }
    body["case_identity_sha256"] = _document_identity(
        "fluidsbench-hiliftaeroml-native-profile-truth-case-v1", body
    )
    record_path = output_root / "case-records" / f"{case_id}.json"
    _atomic_json(record_path, body, check=check)
    return body


def _load_case_record(
    output_root: Path,
    case_id: str,
    *,
    validate_artifact: bool,
) -> tuple[dict[str, Any], str]:
    path = output_root / "case-records" / f"{case_id}.json"
    record, digest = load_json(path, label=f"{case_id} truth record")
    identity = record.get("case_identity_sha256")
    body = dict(record)
    body.pop("case_identity_sha256", None)
    if (
        record.get("schema") != TRUTH_CASE_SCHEMA
        or record.get("format") != TRUTH_FORMAT
        or record.get("case_id") != case_id
        or record.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or identity
        != _document_identity(
            "fluidsbench-hiliftaeroml-native-profile-truth-case-v1", body
        )
    ):
        raise NativeProfileTruthError(f"{case_id} truth record identity differs")
    if validate_artifact:
        _load_case_truth_artifact(output_root, case_id, record)
    return record, digest


def _load_case_truth_artifact(
    output_root: Path,
    case_id: str,
    record: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    artifact_path, artifact_sha = _load_descriptor(
        {
            key: record["truth_artifact"][key]
            for key in ("file", "sha256", "byte_size")
        },
        label=f"{case_id} truth artifact",
        root=output_root,
    )
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            if archive.files != list(TRUTH_ARRAYS):
                raise NativeProfileTruthError(
                    f"{case_id} truth artifact array inventory/order differs"
                )
            truth_cp = np.array(archive["truth_cp"], copy=True)
            velocity = np.array(archive["reference_velocity_nd"], copy=True)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise NativeProfileTruthError(
            f"cannot read {case_id} truth artifact: {error}"
        ) from error
    if artifact_sha != record["truth_artifact"]["sha256"]:
        raise NativeProfileTruthError(f"{case_id} truth artifact digest differs")
    if (
        list(truth_cp.shape) != record["surface_cp"].get("truth_shape")
        or truth_cp.dtype.str != record["surface_cp"].get("truth_dtype")
        or array_identity(
            truth_cp, namespace="fluidsbench-hiliftaeroml-cp-truth-v1"
        )
        != record["surface_cp"].get("truth_identity_sha256")
        or list(velocity.shape) != record["volume_velocity"].get("truth_shape")
        or velocity.dtype.str != record["volume_velocity"].get("truth_dtype")
        or array_identity(
            velocity, namespace="fluidsbench-hiliftaeroml-velocity-truth-v1"
        )
        != record["volume_velocity"].get("truth_identity_sha256")
    ):
        raise NativeProfileTruthError(f"{case_id} truth array identity differs")
    return {
        "truth_cp": truth_cp,
        "reference_velocity_nd": velocity,
    }, artifact_sha


def load_case_truth_arrays(
    output_root: Path,
    case_id: str,
) -> tuple[dict[str, Any], str, dict[str, np.ndarray], str]:
    """Load one identity-checked truth record and its two canonical arrays."""

    record, record_sha = _load_case_record(
        output_root, case_id, validate_artifact=False
    )
    arrays, artifact_sha = _load_case_truth_artifact(
        output_root, case_id, record
    )
    return record, record_sha, arrays, artifact_sha


def assemble_release(
    *,
    universe: CaseUniverse,
    source_index_sha256: str,
    output_root: Path,
    cases_per_chunk: int = 20,
    check: bool = False,
) -> dict[str, Any]:
    _positive_int(cases_per_chunk, "cases_per_chunk")
    source_index_sha256 = _sha(source_index_sha256, "source index digest")
    records: list[tuple[str, dict[str, Any], str]] = []
    for case_id in universe.case_ids:
        record, record_sha = _load_case_record(
            output_root, case_id, validate_artifact=True
        )
        records.append((case_id, record, record_sha))
    chunks: list[dict[str, Any]] = []
    case_locations: dict[str, dict[str, Any]] = {}
    for chunk_index, start in enumerate(range(0, len(records), cases_per_chunk)):
        selected = records[start : start + cases_per_chunk]
        chunk_body = {
            "schema": TRUTH_CHUNK_SCHEMA,
            "schema_version": 1,
            "format": TRUTH_FORMAT,
            "dataset_id": "hiliftaeroml",
            "release_id": TRUTH_RELEASE_ID,
            "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
            "cases": [
                {
                    "case_id": case_id,
                    "case_record_sha256": record_sha,
                    "truth_artifact_sha256": record["truth_artifact"]["sha256"],
                }
                for case_id, record, record_sha in selected
            ],
        }
        filename = f"chunks/chunk-{chunk_index:03d}.json"
        chunk_sha = _atomic_json(output_root / filename, chunk_body, check=check)
        case_ids = [case_id for case_id, _, _ in selected]
        chunks.append(
            {
                "file": filename,
                "sha256": chunk_sha,
                "byte_size": (output_root / filename).stat().st_size,
                "case_count": len(case_ids),
                "case_ids": case_ids,
            }
        )
        for case_id, record, record_sha in selected:
            case_locations[case_id] = {
                "chunk_file": filename,
                "chunk_sha256": chunk_sha,
                "case_record_sha256": record_sha,
                "truth_artifact": record["truth_artifact"],
            }
    index = {
        "schema": TRUTH_INDEX_SCHEMA,
        "schema_version": 1,
        "format": TRUTH_FORMAT,
        "status": "candidate_owner_review_required",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "profile_contract_id": PROFILE_CONTRACT_ID,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "support_manifest_sha256": universe.support_manifest_sha256,
        "case_universe_sha256": universe.case_universe_sha256,
        "case_count": len(universe.case_ids),
        "case_ids": list(universe.case_ids),
        "chunks": chunks,
        "case_locations": case_locations,
    }
    index_sha = _atomic_json(output_root / "index.json", index, check=check)
    split_descriptors: list[dict[str, Any]] = []
    for case_set_id, case_ids, source_case_set_index_sha in universe.case_sets:
        split = {
            "schema": TRUTH_CASE_SET_SCHEMA,
            "schema_version": 1,
            "format": TRUTH_FORMAT,
            "dataset_id": "hiliftaeroml",
            "release_id": TRUTH_RELEASE_ID,
            "case_set_id": case_set_id,
            "case_count": len(case_ids),
            "case_ids": list(case_ids),
            "master_index_sha256": index_sha,
            "source_scoring_support_case_set_index_sha256": source_case_set_index_sha,
            "resolution": "resolve every case through master index case_locations",
        }
        filename = f"case-sets/{case_set_id}.json"
        digest = _atomic_json(output_root / filename, split, check=check)
        split_descriptors.append(
            {
                "case_set_id": case_set_id,
                "case_count": len(case_ids),
                "file": filename,
                "sha256": digest,
                "byte_size": (output_root / filename).stat().st_size,
            }
        )
    provenance = {
        "schema": "hiliftaeroml-native-profile-truth-provenance-v1-candidate",
        "schema_version": 1,
        "status": "candidate_owner_review_required",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "profile_contract": {
            "id": PROFILE_CONTRACT_ID,
            "sha256": PROFILE_CONTRACT_SHA256,
        },
        "truth_source_index_sha256": source_index_sha256,
        "support_manifest_sha256": universe.support_manifest_sha256,
        "case_universe_sha256": universe.case_universe_sha256,
        "truth_method": {
            "surface_cp": "exact benchmark-owned truth aligned to the disconnected Cp cut graph",
            "volume_velocity": "exact benchmark-owned velocity truth with native invalid gaps retained as NaN",
            "topology_storage": "not duplicated; full identities are bound in each case record",
            "case_deduplication": "one artifact per unique physical case; case-set indexes are references only",
        },
        "participant_artifacts_remain_prediction_only": True,
    }
    provenance_sha = _atomic_json(
        output_root / "provenance.json", provenance, check=check
    )
    truth_bytes = sum(
        int(record[1]["truth_artifact"]["byte_size"]) for record in records
    )
    public_metadata_bytes = sum(item["byte_size"] for item in chunks) + sum(
        item["byte_size"] for item in split_descriptors
    )
    manifest = {
        "schema": TRUTH_MANIFEST_SCHEMA,
        "schema_version": 1,
        "format": TRUTH_FORMAT,
        "status": "candidate_owner_review_required",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "profile_contract_id": PROFILE_CONTRACT_ID,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "case_count": len(universe.case_ids),
        "case_set_count": len(universe.case_sets),
        "master_index": {
            "file": "index.json",
            "sha256": index_sha,
            "byte_size": (output_root / "index.json").stat().st_size,
        },
        "provenance": {
            "file": "provenance.json",
            "sha256": provenance_sha,
            "byte_size": (output_root / "provenance.json").stat().st_size,
        },
        "case_sets": split_descriptors,
        "storage": {
            "truth_artifact_bytes": truth_bytes,
            "chunk_and_case_set_metadata_bytes": public_metadata_bytes,
            "topology_payloads_duplicated": 0,
            "case_artifacts_duplicated_across_case_sets": 0,
        },
        "activation": {
            "owner_approval_complete": False,
            "published": False,
            "submissions_opened": False,
        },
    }
    manifest_sha = _atomic_json(output_root / "manifest.json", manifest, check=check)
    receipt = {
        "schema": "hiliftaeroml-native-profile-truth-release-receipt-v1-candidate",
        "schema_version": 1,
        "status": "complete_candidate_not_published",
        "dataset_id": "hiliftaeroml",
        "release_id": TRUTH_RELEASE_ID,
        "manifest_sha256": manifest_sha,
        "truth_source_index_sha256": source_index_sha256,
        "case_count": len(universe.case_ids),
        "case_set_count": len(universe.case_sets),
        "deterministic_packaging": True,
        "owner_approval_complete": False,
        "published": False,
        "submissions_opened": False,
    }
    _atomic_json(output_root / "release-receipt.json", receipt, check=check)
    return manifest


def _require_exact_release_tree(output_root: Path, expected_files: set[str]) -> None:
    """Reject untracked files, directories, links, and interrupted-write debris."""

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in output_root.rglob("*"):
        relative = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise NativeProfileTruthError(
                f"profile truth release contains a symlink: {relative}"
            )
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise NativeProfileTruthError(
                f"profile truth release contains a non-file entry: {relative}"
            )
    expected_directories = {
        parent.as_posix()
        for filename in expected_files
        for parent in Path(filename).parents
        if parent != Path(".")
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        extra_files = sorted(observed_files - expected_files)
        missing_files = sorted(expected_files - observed_files)
        extra_directories = sorted(observed_directories - expected_directories)
        missing_directories = sorted(expected_directories - observed_directories)
        raise NativeProfileTruthError(
            "profile truth release inventory differs: "
            f"extra_files={extra_files}, missing_files={missing_files}, "
            f"extra_directories={extra_directories}, "
            f"missing_directories={missing_directories}"
        )


def validate_release(
    *,
    universe: CaseUniverse,
    source: Mapping[str, Any],
    source_index_sha256: str,
    output_root: Path,
    replay_sources: bool = True,
    replay_case_builder: Callable[[str, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate all release bytes and optionally replay every source binding."""

    if output_root.is_symlink() or not output_root.is_dir():
        raise NativeProfileTruthError(
            "profile truth output must be a regular non-symlink directory"
        )
    expected_release_files = {"manifest.json", "release-receipt.json"}
    manifest, manifest_sha = load_json(
        output_root / "manifest.json", label="profile truth manifest"
    )
    if (
        manifest.get("schema") != TRUTH_MANIFEST_SCHEMA
        or manifest.get("format") != TRUTH_FORMAT
        or manifest.get("release_id") != TRUTH_RELEASE_ID
        or manifest.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or manifest.get("case_count") != len(universe.case_ids)
        or manifest.get("case_set_count") != len(universe.case_sets)
        or manifest.get("activation", {}).get("published") is not False
        or manifest.get("activation", {}).get("owner_approval_complete") is not False
        or manifest.get("activation", {}).get("submissions_opened") is not False
    ):
        raise NativeProfileTruthError("profile truth manifest differs")
    index_path, index_sha = _load_descriptor(
        manifest.get("master_index"), label="master index", root=output_root
    )
    provenance_path, _ = _load_descriptor(
        manifest.get("provenance"), label="provenance", root=output_root
    )
    expected_release_files.update(
        {
            index_path.relative_to(output_root.resolve()).as_posix(),
            provenance_path.relative_to(output_root.resolve()).as_posix(),
        }
    )
    index, _ = load_json(index_path, label="profile truth index")
    provenance, _ = load_json(provenance_path, label="profile truth provenance")
    if (
        index.get("schema") != TRUTH_INDEX_SCHEMA
        or index.get("case_ids") != list(universe.case_ids)
        or index.get("case_universe_sha256") != universe.case_universe_sha256
        or provenance.get("truth_source_index_sha256") != source_index_sha256
        or provenance.get("support_manifest_sha256") != universe.support_manifest_sha256
    ):
        raise NativeProfileTruthError("profile truth index/provenance binding differs")
    chunk_by_case: dict[str, tuple[str, str, str, str]] = {}
    chunks = index.get("chunks")
    if not isinstance(chunks, list):
        raise NativeProfileTruthError("profile truth chunk inventory is absent")
    for descriptor in chunks:
        path, digest = _load_descriptor(
            {key: descriptor[key] for key in ("file", "sha256", "byte_size")},
            label="profile truth chunk",
            root=output_root,
        )
        expected_release_files.add(path.relative_to(output_root.resolve()).as_posix())
        chunk, _ = load_json(path, label="profile truth chunk")
        cases = chunk.get("cases")
        if chunk.get("schema") != TRUTH_CHUNK_SCHEMA or not isinstance(cases, list):
            raise NativeProfileTruthError("profile truth chunk differs")
        observed_ids = []
        for entry in cases:
            if not isinstance(entry, Mapping) or set(entry) != {
                "case_id",
                "case_record_sha256",
                "truth_artifact_sha256",
            }:
                raise NativeProfileTruthError(
                    "profile truth chunk case descriptor differs"
                )
            observed_ids.append(entry.get("case_id"))
        if observed_ids != descriptor.get("case_ids") or len(cases) != descriptor.get(
            "case_count"
        ):
            raise NativeProfileTruthError("profile truth chunk case inventory differs")
        for case_id, entry in zip(observed_ids, cases, strict=True):
            if not isinstance(case_id, str) or case_id in chunk_by_case:
                raise NativeProfileTruthError(
                    "profile truth chunk case duplication differs"
                )
            chunk_by_case[case_id] = (
                descriptor["file"],
                digest,
                _sha(
                    entry.get("case_record_sha256"),
                    f"{case_id} chunk case-record digest",
                ),
                _sha(
                    entry.get("truth_artifact_sha256"),
                    f"{case_id} chunk truth-artifact digest",
                ),
            )
    if set(chunk_by_case) != set(universe.case_ids):
        raise NativeProfileTruthError(
            "profile truth chunks do not cover the case universe"
        )
    for case_id in universe.case_ids:
        record, record_sha = _load_case_record(
            output_root, case_id, validate_artifact=True
        )
        expected_release_files.add(f"case-records/{case_id}.json")
        expected_release_files.add(record["truth_artifact"]["file"])
        location = index.get("case_locations", {}).get(case_id)
        chunk_file, chunk_sha, chunk_record_sha, chunk_artifact_sha = chunk_by_case[
            case_id
        ]
        if (
            not isinstance(location, Mapping)
            or (location.get("chunk_file"), location.get("chunk_sha256"))
            != (chunk_file, chunk_sha)
            or location.get("case_record_sha256") != record_sha
            or location.get("truth_artifact") != record.get("truth_artifact")
            or chunk_record_sha != record_sha
            or chunk_artifact_sha != record["truth_artifact"]["sha256"]
        ):
            raise NativeProfileTruthError(f"{case_id} master index location differs")
        if replay_sources:
            with tempfile.TemporaryDirectory(
                prefix="hilift-truth-replay-"
            ) as temporary:
                replay_root = Path(temporary)
                replay = (
                    replay_case_builder(case_id, replay_root)
                    if replay_case_builder is not None
                    else build_case_truth(
                        case_id=case_id,
                        source=source,
                        output_root=replay_root,
                    )
                )
                if replay != record:
                    raise NativeProfileTruthError(f"{case_id} source replay differs")
    case_set_descriptors = manifest.get("case_sets")
    if not isinstance(case_set_descriptors, list) or len(case_set_descriptors) != len(
        universe.case_sets
    ):
        raise NativeProfileTruthError(
            "profile truth case-set descriptor inventory differs"
        )
    expected_sets = {item[0]: item for item in universe.case_sets}
    seen_case_sets: set[str] = set()
    for descriptor in case_set_descriptors:
        case_set_id = (
            descriptor.get("case_set_id") if isinstance(descriptor, Mapping) else None
        )
        if case_set_id not in expected_sets:
            raise NativeProfileTruthError("profile truth case-set identity differs")
        if case_set_id in seen_case_sets:
            raise NativeProfileTruthError(
                "profile truth case-set descriptor is duplicated"
            )
        seen_case_sets.add(case_set_id)
        path, _ = _load_descriptor(
            {key: descriptor[key] for key in ("file", "sha256", "byte_size")},
            label=f"{case_set_id} truth index",
            root=output_root,
        )
        expected_release_files.add(path.relative_to(output_root.resolve()).as_posix())
        split, _ = load_json(path, label=f"{case_set_id} truth index")
        _, expected_ids, source_case_set_sha = expected_sets[case_set_id]
        if (
            descriptor.get("case_count") != len(expected_ids)
            or split.get("schema") != TRUTH_CASE_SET_SCHEMA
            or split.get("format") != TRUTH_FORMAT
            or split.get("dataset_id") != "hiliftaeroml"
            or split.get("release_id") != TRUTH_RELEASE_ID
            or split.get("case_set_id") != case_set_id
            or split.get("case_count") != len(expected_ids)
            or split.get("case_ids") != list(expected_ids)
            or split.get("master_index_sha256") != index_sha
            or split.get("source_scoring_support_case_set_index_sha256")
            != source_case_set_sha
        ):
            raise NativeProfileTruthError(f"{case_set_id} truth index differs")
    if seen_case_sets != set(expected_sets):
        raise NativeProfileTruthError("profile truth case-set inventory differs")
    receipt, _ = load_json(
        output_root / "release-receipt.json", label="profile truth release receipt"
    )
    if (
        receipt.get("manifest_sha256") != manifest_sha
        or receipt.get("truth_source_index_sha256") != source_index_sha256
        or receipt.get("published") is not False
        or receipt.get("owner_approval_complete") is not False
        or receipt.get("submissions_opened") is not False
    ):
        raise NativeProfileTruthError("profile truth release receipt differs")
    _require_exact_release_tree(output_root, expected_release_files)
    return {
        "status": "pass",
        "release_id": TRUTH_RELEASE_ID,
        "manifest_sha256": manifest_sha,
        "case_count": len(universe.case_ids),
        "case_set_count": len(universe.case_sets),
        "source_replay_complete": replay_sources,
    }


def preflight(
    *, universe: CaseUniverse, source: Mapping[str, Any], source_index_sha256: str
) -> dict[str, Any]:
    cases = source.get("cases")
    missing = source.get("missing_case_ids")
    if not isinstance(cases, Mapping) or not isinstance(missing, list):
        raise NativeProfileTruthError("source index inventory differs")
    source_bytes = 0
    for case in cases.values():
        if not isinstance(case, Mapping) or not isinstance(
            case.get("artifacts"), Mapping
        ):
            raise NativeProfileTruthError("source index case entry differs")
        for descriptor in case["artifacts"].values():
            if not isinstance(descriptor, Mapping):
                raise NativeProfileTruthError("source descriptor differs")
            source_bytes += _positive_int(
                descriptor.get("byte_size"), "source artifact byte size"
            )
    return {
        "schema": "hiliftaeroml-native-profile-truth-export-preflight-v1-candidate",
        "status": "ready" if not missing else "blocked_missing_materialized_sources",
        "release_id": TRUTH_RELEASE_ID,
        "source_index_sha256": source_index_sha256,
        "case_count": len(universe.case_ids),
        "case_set_count": len(universe.case_sets),
        "available_source_case_count": len(cases),
        "missing_source_case_count": len(missing),
        "missing_source_case_ids": missing,
        "available_source_artifact_bytes": source_bytes,
        "case_array_job": {
            "array": f"0-{len(universe.case_ids) - 1}",
            "one_case_per_task": True,
            "safe_parallelism": "I/O limited; begin at 8 concurrent tasks",
            "recommended_cpus_per_task": 1,
            "recommended_memory_gib": 4,
            "recommended_walltime": "00:15:00",
        },
        "assembly_job": {
            "dependency": "afterok:<case-array-job-id>",
            "recommended_cpus": 2,
            "recommended_memory_gib": 8,
            "recommended_walltime": "01:00:00",
        },
        "activation_changed": False,
        "owner_approval_complete": False,
        "published": False,
    }


__all__ = [
    "EXPECTED_CASE_SET_COUNT",
    "EXPECTED_UNIQUE_CASE_COUNT",
    "NativeProfileTruthError",
    "TRUTH_ARRAYS",
    "TRUTH_FORMAT",
    "TRUTH_RELEASE_ID",
    "array_identity",
    "assemble_release",
    "build_case_truth",
    "build_case_truth_from_arrays",
    "build_source_index",
    "load_case_universe",
    "load_case_truth_arrays",
    "load_source_index",
    "preflight",
    "validate_release",
]
