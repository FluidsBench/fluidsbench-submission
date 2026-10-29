# SPDX-License-Identifier: Apache-2.0
"""Prediction-free HiLiftAeroML native-profile truth materialization.

The production authority is the frozen geometry-only Cp/velocity stencils,
the frozen training-validity map, and the original benchmark truth fields.
Completed model-evaluator outputs may be supplied only as exact replay
oracles; their predictions and their truth copies never become source inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reference.hiliftaeroml.native_profile_truth import (
    CaseUniverse,
    NativeProfileTruthError,
    _atomic_bytes,
    _atomic_json,
    _document_identity,
    _sha,
    array_identity,
    build_case_truth_from_arrays,
)
from reference.hiliftaeroml.native_profiles import (
    CP_SOURCE_ARRAYS,
    ROWS_PER_STATION,
    SAFE_CASE_ID,
    STATIONS,
    load_json,
    sha256_file,
)

AUTHORITY_SCHEMA = (
    "hiliftaeroml-native-profile-truth-prerequisite-authority-index-v1-candidate"
)
SUPPORT_MAP_SCHEMA = "hiliftaeroml-native-profile-pdmsh-support-map-v1-candidate"
AUTHORITY_KIND = "benchmark_truth_only_prerequisite_replay"
PDM_RANDOMIZATION_SEED = 42
GLOBAL_DATA_NAMES = ("U_inf", "p_inf", "rho_inf", "T_inf", "L_ref")
CP_STENCIL_ARRAYS = (
    "branch_closed",
    "branch_component_code",
    "branch_graph_component_code",
    "branch_length_in",
    "branch_plane_piece_code",
    "branch_row_code",
    "branch_segment_offsets",
    "branch_side_code",
    "branch_topology_patch_code",
    "branch_vertex_ids",
    "branch_vertex_offsets",
    "branch_vertex_quadrature_weights_in",
    "cut_support_node_ids",
    "cut_support_offsets",
    "cut_support_weights",
    "cut_xyz_in",
    "segment_lengths_in",
    "segment_plane_piece_code",
    "segment_vertex_ids",
)
VELOCITY_STENCIL_ARRAYS = (
    "station_names",
    "station_row_offsets",
    "station_surface_xyz_in",
    "station_z_bounds_in",
    "requested_xyz_in",
    "valid_mask",
    "containing_cell_ids",
    "sub_ids",
    "parametric_coordinates",
    "csr_offsets",
    "raw_point_ids",
    "interpolation_weights",
    "support_raw_point_ids",
)
SUPPORT_MAP_ARRAYS = (
    "support_raw_point_ids",
    "support_compact_point_ids",
    "support_pdmsh_rows",
)


def _regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise NativeProfileTruthError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise NativeProfileTruthError(f"{label} must be a regular file: {resolved}")
    return resolved


def _absolute_descriptor(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _stat_descriptor(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    stat = path.stat()
    return {
        "file": str(path),
        "byte_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_absolute_descriptor(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "sha256",
        "byte_size",
    }:
        raise NativeProfileTruthError(f"{label} descriptor differs")
    raw_path = value.get("file")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise NativeProfileTruthError(f"{label}.file must be absolute")
    path = _regular_file(Path(raw_path), label)
    digest = _sha(value.get("sha256"), f"{label}.sha256")
    if path.stat().st_size != value.get("byte_size") or sha256_file(path) != digest:
        raise NativeProfileTruthError(f"{label} live bytes differ")
    return path, digest


def _validate_stat(value: Any, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "byte_size",
        "mtime_ns",
    }:
        raise NativeProfileTruthError(f"{label} stat descriptor differs")
    raw_path = value.get("file")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise NativeProfileTruthError(f"{label}.file must be absolute")
    path = _regular_file(Path(raw_path), label)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        value.get("byte_size"),
        value.get("mtime_ns"),
    ):
        raise NativeProfileTruthError(f"{label} live stat differs")
    return path


def _record_payload(
    *, family: str, record_path: Path, expected_case_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata, _ = load_json(record_path, label=f"{expected_case_id} {family} record")
    if metadata.get("case_id") != expected_case_id or (
        family != "volume_training_validity" and metadata.get("status") != "complete"
    ):
        raise NativeProfileTruthError(
            f"{expected_case_id} {family} record case/status differs"
        )
    if family == "surface_cp_stencils":
        payload = metadata.get("payload")
        if not isinstance(payload, Mapping):
            raise NativeProfileTruthError(f"{expected_case_id} Cp payload is absent")
        filename = payload.get("path")
        expected_sha = payload.get("sha256")
        expected_size = payload.get("byte_size")
    else:
        filename = metadata.get("npz_file")
        expected_sha = metadata.get("npz_sha256")
        expected_size = None
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(f"_{expected_case_id}.npz")
    ):
        raise NativeProfileTruthError(
            f"{expected_case_id} {family} payload name differs"
        )
    payload_path = _regular_file(
        record_path.parent / filename, f"{expected_case_id} {family} payload"
    )
    payload_descriptor = _absolute_descriptor(
        payload_path, f"{expected_case_id} {family} payload"
    )
    if payload_descriptor["sha256"] != _sha(
        expected_sha, f"{expected_case_id} {family} payload digest"
    ) or (
        expected_size is not None and payload_descriptor["byte_size"] != expected_size
    ):
        raise NativeProfileTruthError(
            f"{expected_case_id} {family} payload identity differs"
        )
    return (
        metadata,
        _absolute_descriptor(record_path, f"{expected_case_id} {family} record"),
        payload_descriptor,
    )


def _select_record(
    *,
    family: str,
    case_id: str,
    roots: Sequence[Path],
    filename: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for root in roots:
        path = root.expanduser().resolve() / case_id / filename
        if path.is_file() and not path.is_symlink():
            candidates.append(
                _record_payload(
                    family=family, record_path=path, expected_case_id=case_id
                )
            )
    if not candidates:
        return None
    identities = {
        (
            candidate[1]["sha256"],
            candidate[1]["byte_size"],
            candidate[2]["sha256"],
            candidate[2]["byte_size"],
        )
        for candidate in candidates
    }
    if len(identities) != 1:
        raise NativeProfileTruthError(
            f"{case_id} has conflicting complete {family} roots"
        )
    return candidates[0]


def _pdmsh_root(training_points: Path) -> Path:
    expected = ("_tensordict", "interior", "_tensordict", "points.memmap")
    if tuple(training_points.parts[-4:]) != expected:
        raise NativeProfileTruthError(
            f"training-points path is not a canonical PDMsh payload: {training_points}"
        )
    root = training_points.parents[3]
    if not root.is_dir() or root.is_symlink():
        raise NativeProfileTruthError(f"invalid PDMsh root: {root}")
    return root.resolve()


def _historical_pdmsh_root(validity_metadata: Mapping[str, Any], case_id: str) -> Path:
    """Locate PDMsh data without binding the unused historical points payload."""

    training_points_value = validity_metadata.get("training_points_path")
    if not isinstance(training_points_value, str) or not training_points_value:
        raise NativeProfileTruthError(
            f"{case_id} historical training-points path is absent"
        )
    training_points = Path(training_points_value)
    if not training_points.is_absolute():
        raise NativeProfileTruthError(
            f"{case_id} historical training-points path is not absolute"
        )
    if (
        validity_metadata.get("training_points_role")
        != "historical PDMsh provenance only"
        or validity_metadata.get("training_points_used_for_native_filter") is not False
        or validity_metadata.get("training_points_used_for_partition_order")
        is not False
        or validity_metadata.get("training_points_coordinate_order_equality_required")
        is not False
    ):
        raise NativeProfileTruthError(
            f"{case_id} historical training-points role differs"
        )
    # The native validity payload, rather than points.memmap, binds the raw-to-
    # compact mapping. Only the canonical path shape is used to locate the
    # separately content-bound freestream inputs and optional PDMsh audit data.
    return _pdmsh_root(training_points)


def _pdmsh_velocity_source(
    pdmsh_root: Path, *, valid_count: int, case_id: str
) -> dict[str, Any]:
    point_data = pdmsh_root / "_tensordict" / "interior" / "_tensordict" / "point_data"
    meta_path = _regular_file(point_data / "meta.json", f"{case_id} PDMsh metadata")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    avg_u = metadata.get("avg(u)")
    if not isinstance(avg_u, Mapping):
        raise NativeProfileTruthError(f"{case_id} PDMsh metadata lacks avg(u)")
    if avg_u.get("dtype") != "torch.float32" or avg_u.get("shape") != [
        valid_count,
        3,
    ]:
        raise NativeProfileTruthError(f"{case_id} PDMsh avg(u) contract differs")
    path = _regular_file(point_data / "avg(u).memmap", f"{case_id} PDMsh avg(u)")
    if path.stat().st_size != valid_count * 3 * np.dtype("<f4").itemsize:
        raise NativeProfileTruthError(f"{case_id} PDMsh avg(u) byte count differs")
    return {
        "root": str(pdmsh_root),
        "point_data_metadata": _absolute_descriptor(
            meta_path, f"{case_id} PDMsh metadata"
        ),
        "avg_u": {
            **_stat_descriptor(path, f"{case_id} PDMsh avg(u)"),
            "dtype": "<f4",
            "shape": [valid_count, 3],
        },
    }


def _global_data_sources(pdmsh_root: Path, case_id: str) -> dict[str, Any]:
    root = pdmsh_root / "_tensordict" / "global_data"
    result: dict[str, Any] = {}
    for name in GLOBAL_DATA_NAMES:
        path = root / f"{name}.memmap"
        result[f"{name}.memmap"] = _absolute_descriptor(
            path, f"{case_id} global data {name}"
        )
    return result


def build_prerequisite_authority_index(
    *,
    universe: CaseUniverse,
    cp_record_roots: Sequence[Path],
    velocity_record_roots: Sequence[Path],
    validity_record_roots: Sequence[Path],
    output_path: Path,
    check: bool = False,
) -> dict[str, Any]:
    """Inventory all prediction-free prerequisites for the exact case universe."""

    if not cp_record_roots or not velocity_record_roots or not validity_record_roots:
        raise NativeProfileTruthError(
            "all three prerequisite root families are required"
        )
    cases: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for case_id in universe.case_ids:
        cp = _select_record(
            family="surface_cp_stencils",
            case_id=case_id,
            roots=cp_record_roots,
            filename=f"native_cp_cut_stencil_{case_id}.json",
        )
        velocity = _select_record(
            family="volume_velocity_profile_stencils",
            case_id=case_id,
            roots=velocity_record_roots,
            filename=f"native_velocity_profile_stencil_{case_id}.json",
        )
        validity = _select_record(
            family="volume_training_validity",
            case_id=case_id,
            roots=validity_record_roots,
            filename=f"native_volume_training_validity_{case_id}.json",
        )
        absent = [
            name
            for name, selected in (
                ("surface_cp_stencils", cp),
                ("volume_velocity_profile_stencils", velocity),
                ("volume_training_validity", validity),
            )
            if selected is None
        ]
        if absent:
            missing[case_id] = absent
            continue
        if cp is None or velocity is None or validity is None:
            raise NativeProfileTruthError(
                f"{case_id} prerequisite selection changed during inventory"
            )
        cp_metadata, cp_record, cp_payload = cp
        velocity_metadata, velocity_record, velocity_payload = velocity
        validity_metadata, validity_record, validity_payload = validity

        cp_source = cp_metadata.get("source")
        if not isinstance(cp_source, Mapping):
            raise NativeProfileTruthError(f"{case_id} Cp source record is absent")
        surface_vtu = _regular_file(
            Path(str(cp_source.get("vtu_path"))), f"{case_id} surface VTU"
        )
        surface_stat = surface_vtu.stat()
        if (surface_stat.st_size, surface_stat.st_mtime_ns) != (
            cp_source.get("vtu_size_bytes"),
            cp_source.get("vtu_mtime_ns"),
        ):
            raise NativeProfileTruthError(f"{case_id} surface VTU stat differs")

        volume_vtu = _regular_file(
            Path(str(velocity_metadata.get("source_vtu"))), f"{case_id} volume VTU"
        )
        if (
            volume_vtu.resolve()
            != Path(str(validity_metadata.get("source_vtu"))).resolve()
        ):
            raise NativeProfileTruthError(f"{case_id} volume source paths differ")
        if volume_vtu.stat().st_size != velocity_metadata.get("source_vtu_size_bytes"):
            raise NativeProfileTruthError(f"{case_id} volume VTU size differs")
        volume_sha = _sha(
            velocity_metadata.get("source_vtu_sha256"),
            f"{case_id} frozen volume VTU digest",
        )
        raw_count = int(validity_metadata.get("raw_count", -1))
        valid_count = int(validity_metadata.get("valid_count", -1))
        if (
            raw_count != velocity_metadata.get("source_number_of_points")
            or valid_count <= 0
            or raw_count < valid_count
            or raw_count <= 0
        ):
            raise NativeProfileTruthError(f"{case_id} volume population counts differ")
        pdmsh_root = _historical_pdmsh_root(validity_metadata, case_id)
        entry: dict[str, Any] = {
            "case_id": case_id,
            "authority_kind": AUTHORITY_KIND,
            "artifacts": {
                "cp_stencil_record": cp_record,
                "cp_stencil_payload": cp_payload,
                "velocity_stencil_record": velocity_record,
                "velocity_stencil_payload": velocity_payload,
                "validity_record": validity_record,
                "validity_payload": validity_payload,
            },
            "surface_truth_source": {
                "vtu": _stat_descriptor(surface_vtu, f"{case_id} surface VTU"),
                "point_count": cp_source.get("topology", {}).get("n_points"),
                "pressure_field": "PROJ(AVG(P))",
                "geometry_sha256": _sha(
                    cp_source.get("geometry_sha256"),
                    f"{case_id} surface geometry digest",
                ),
                "normal_sha256": _sha(
                    cp_source.get("normal_sha256"),
                    f"{case_id} surface normal digest",
                ),
            },
            "volume_truth_source": {
                "vtu": {
                    **_stat_descriptor(volume_vtu, f"{case_id} volume VTU"),
                    "frozen_sha256": volume_sha,
                },
                "raw_count": raw_count,
                "valid_count": valid_count,
                "pdmsh": _pdmsh_velocity_source(
                    pdmsh_root, valid_count=valid_count, case_id=case_id
                ),
                "global_data": _global_data_sources(pdmsh_root, case_id),
                "permutation_contract": {
                    "algorithm": "torch.randperm",
                    "device": "cpu",
                    "seed": PDM_RANDOMIZATION_SEED,
                    "population": "ascending compact IDs after avg(P) != 0 validity filter",
                },
            },
            "prediction_bearing_evaluator_outputs_used_as_source": False,
        }
        entry["authority_case_identity_sha256"] = _document_identity(
            "fluidsbench-hiliftaeroml-profile-truth-authority-case-v1", entry
        )
        cases[case_id] = entry
    body = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "dataset_id": "hiliftaeroml",
        "authority_kind": AUTHORITY_KIND,
        "support_manifest_sha256": universe.support_manifest_sha256,
        "case_universe_sha256": universe.case_universe_sha256,
        "expected_case_count": len(universe.case_ids),
        "available_case_count": len(cases),
        "missing_case_count": len(missing),
        "case_ids": list(universe.case_ids),
        "missing": missing,
        "cases": cases,
        "prediction_bearing_evaluator_outputs_used_as_source": False,
    }
    _atomic_json(output_path, body, check=check)
    return body


def load_prerequisite_authority_index(
    path: Path, *, universe: CaseUniverse, require_complete: bool
) -> tuple[dict[str, Any], str]:
    body, digest = load_json(path, label="HiLift profile truth authority index")
    cases = body.get("cases")
    missing = body.get("missing")
    if (
        body.get("schema") != AUTHORITY_SCHEMA
        or body.get("authority_kind") != AUTHORITY_KIND
        or body.get("support_manifest_sha256") != universe.support_manifest_sha256
        or body.get("case_universe_sha256") != universe.case_universe_sha256
        or body.get("case_ids") != list(universe.case_ids)
        or body.get("prediction_bearing_evaluator_outputs_used_as_source") is not False
        or not isinstance(cases, Mapping)
        or not isinstance(missing, Mapping)
        or set(cases).intersection(missing)
        or set(cases).union(missing) != set(universe.case_ids)
        or body.get("available_case_count") != len(cases)
        or body.get("missing_case_count") != len(missing)
    ):
        raise NativeProfileTruthError("profile truth authority index binding differs")
    for case_id, entry in cases.items():
        if not isinstance(entry, Mapping) or entry.get("case_id") != case_id:
            raise NativeProfileTruthError(f"{case_id} authority entry differs")
        recorded = entry.get("authority_case_identity_sha256")
        identity_body = dict(entry)
        identity_body.pop("authority_case_identity_sha256", None)
        if recorded != _document_identity(
            "fluidsbench-hiliftaeroml-profile-truth-authority-case-v1",
            identity_body,
        ):
            raise NativeProfileTruthError(f"{case_id} authority identity differs")
    if require_complete and (missing or body.get("status") != "complete"):
        raise NativeProfileTruthError(
            f"profile truth authority index is incomplete ({len(missing)} cases missing)"
        )
    return body, digest


def _entry(authority: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    cases = authority.get("cases")
    if not isinstance(cases, Mapping) or not isinstance(cases.get(case_id), Mapping):
        raise NativeProfileTruthError(f"authority index lacks {case_id}")
    return cases[case_id]


def _load_artifact_pair(
    entry: Mapping[str, Any], record_name: str, payload_name: str, label: str
) -> tuple[dict[str, Any], Path, str]:
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise NativeProfileTruthError(f"{label} artifacts are absent")
    record_path, record_sha = _load_absolute_descriptor(
        artifacts.get(record_name), f"{label} record"
    )
    payload_path, payload_sha = _load_absolute_descriptor(
        artifacts.get(payload_name), f"{label} payload"
    )
    metadata = json.loads(record_path.read_text(encoding="utf-8"))
    if payload_path.parent != record_path.parent:
        raise NativeProfileTruthError(f"{label} record/payload roots differ")
    return (
        metadata,
        payload_path,
        _document_identity(
            f"fluidsbench-hiliftaeroml-{label}-artifact-pair-v1",
            {"record_sha256": record_sha, "payload_sha256": payload_sha},
        ),
    )


def _load_cp_stencil(
    entry: Mapping[str, Any], case_id: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    metadata, payload_path, pair_identity = _load_artifact_pair(
        entry,
        "cp_stencil_record",
        "cp_stencil_payload",
        f"{case_id}-cp-stencil",
    )
    if metadata.get("case_id") != case_id or metadata.get("status") != "complete":
        raise NativeProfileTruthError(f"{case_id} Cp stencil record differs")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            if tuple(archive.files) != CP_STENCIL_ARRAYS:
                raise NativeProfileTruthError(
                    f"{case_id} Cp stencil array inventory/order differs"
                )
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise NativeProfileTruthError(
            f"cannot read {case_id} Cp stencil: {error}"
        ) from error
    declared = metadata.get("payload", {}).get("arrays")
    if not isinstance(declared, Mapping) or set(declared) != set(arrays):
        raise NativeProfileTruthError(f"{case_id} Cp stencil array contract differs")
    for name, array in arrays.items():
        if declared[name] != {"dtype": array.dtype.str, "shape": list(array.shape)}:
            raise NativeProfileTruthError(f"{case_id} Cp stencil {name} differs")
    offsets = arrays["cut_support_offsets"]
    support_ids = arrays["cut_support_node_ids"]
    weights = arrays["cut_support_weights"]
    if (
        offsets.dtype != np.dtype("<i8")
        or offsets.shape != (len(arrays["cut_xyz_in"]) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(support_ids)
        or support_ids.dtype != np.dtype("<i8")
        or weights.dtype != np.dtype("<f8")
        or support_ids.shape != weights.shape
        or np.any(np.diff(offsets) <= 0)
        or np.any(support_ids < 0)
        or not np.all(np.isfinite(weights))
    ):
        raise NativeProfileTruthError(f"{case_id} Cp support CSR differs")
    return arrays, metadata, pair_identity


def _native_stencil_array_sha(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _load_velocity_stencil(
    entry: Mapping[str, Any], case_id: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    metadata, payload_path, pair_identity = _load_artifact_pair(
        entry,
        "velocity_stencil_record",
        "velocity_stencil_payload",
        f"{case_id}-velocity-stencil",
    )
    if (
        metadata.get("case_id") != case_id
        or metadata.get("status") != "complete"
        or metadata.get("schema_id")
        != "hilift_native_volume_velocity_profile_stencil_v1"
    ):
        raise NativeProfileTruthError(f"{case_id} velocity stencil record differs")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            if tuple(archive.files) != VELOCITY_STENCIL_ARRAYS:
                raise NativeProfileTruthError(
                    f"{case_id} velocity stencil array inventory/order differs"
                )
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise NativeProfileTruthError(
            f"cannot read {case_id} velocity stencil: {error}"
        ) from error
    declared_hashes = metadata.get("array_sha256")
    if not isinstance(declared_hashes, Mapping) or set(declared_hashes) != set(arrays):
        raise NativeProfileTruthError(f"{case_id} velocity stencil hashes differ")
    for name, value in arrays.items():
        if _native_stencil_array_sha(value) != declared_hashes[name]:
            raise NativeProfileTruthError(
                f"{case_id} velocity stencil {name} hash differs"
            )
    if (
        arrays["requested_xyz_in"].shape != (len(STATIONS) * ROWS_PER_STATION, 3)
        or arrays["requested_xyz_in"].dtype != np.dtype("<f8")
        or arrays["valid_mask"].dtype != np.dtype(np.bool_)
        or arrays["station_names"].tolist() != list(STATIONS)
        or arrays["station_row_offsets"].tolist()
        != [index * ROWS_PER_STATION for index in range(len(STATIONS) + 1)]
        or not np.array_equal(
            arrays["support_raw_point_ids"], np.unique(arrays["raw_point_ids"])
        )
    ):
        raise NativeProfileTruthError(f"{case_id} velocity stencil alignment differs")
    return arrays, metadata, pair_identity


def _load_validity(
    entry: Mapping[str, Any], case_id: str, support_raw: np.ndarray
) -> tuple[np.ndarray, dict[str, Any], str]:
    metadata, payload_path, pair_identity = _load_artifact_pair(
        entry,
        "validity_record",
        "validity_payload",
        f"{case_id}-validity",
    )
    if (
        metadata.get("case_id") != case_id
        or metadata.get("schema_id") != "hilift_native_volume_training_validity_v2"
    ):
        raise NativeProfileTruthError(f"{case_id} validity record differs")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            if archive.files != ["excluded_raw_point_ids"]:
                raise NativeProfileTruthError(
                    f"{case_id} validity array inventory differs"
                )
            excluded = np.array(archive["excluded_raw_point_ids"], copy=True)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise NativeProfileTruthError(
            f"cannot read {case_id} validity: {error}"
        ) from error
    if (
        excluded.dtype != np.dtype("<i8")
        or excluded.ndim != 1
        or (excluded.size and np.any(np.diff(excluded) <= 0))
        or len(excluded) != metadata.get("excluded_count")
    ):
        raise NativeProfileTruthError(f"{case_id} excluded raw IDs differ")
    slots = np.searchsorted(excluded, support_raw)
    safe_slots = np.minimum(slots, max(len(excluded) - 1, 0))
    if excluded.size and np.any(
        (slots < len(excluded)) & (excluded[safe_slots] == support_raw)
    ):
        raise NativeProfileTruthError(
            f"{case_id} velocity support contains excluded training points"
        )
    support_compact = support_raw - slots
    valid_count = int(metadata.get("valid_count", -1))
    if (
        support_compact.dtype != np.dtype("<i8")
        or np.any(support_compact < 0)
        or np.any(support_compact >= valid_count)
        or np.any(np.diff(support_compact) <= 0)
    ):
        raise NativeProfileTruthError(f"{case_id} support compact mapping differs")
    return support_compact, metadata, pair_identity


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != SUPPORT_MAP_ARRAYS:
        raise NativeProfileTruthError("support-map NPZ inventory/order differs")
    raw = io.BytesIO()
    with zipfile.ZipFile(
        raw,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in SUPPORT_MAP_ARRAYS:
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


def _regenerate_support_rows(
    *, support_compact: np.ndarray, valid_count: int, seed: int
) -> tuple[np.ndarray, str, str]:
    try:
        import torch
    except ImportError as error:
        raise NativeProfileTruthError(
            "PyTorch is required only to regenerate the frozen PDMsh permutation"
        ) from error
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation_tensor = torch.randperm(valid_count, generator=generator)
    permutation = permutation_tensor.numpy()
    if permutation.dtype != np.dtype(np.int64):
        raise NativeProfileTruthError("torch.randperm dtype changed")
    rows = np.full(len(support_compact), -1, dtype=np.int64)
    digest = hashlib.sha256()
    block_size = 10_000_000
    for start in range(0, valid_count, block_size):
        stop = min(start + block_size, valid_count)
        block = np.asarray(permutation[start:stop], dtype=np.int64)
        digest.update(np.asarray(block, dtype="<i8").tobytes(order="C"))
        slots = np.searchsorted(support_compact, block)
        in_range = slots < len(support_compact)
        local = np.flatnonzero(
            in_range
            & (support_compact[np.minimum(slots, len(support_compact) - 1)] == block)
        )
        if local.size:
            rows[slots[local]] = start + local
    if np.any(rows < 0) or not np.array_equal(permutation[rows], support_compact):
        raise NativeProfileTruthError("failed to invert every PDMsh support row")
    return rows, digest.hexdigest(), str(torch.__version__)


def build_support_map(
    *,
    case_id: str,
    authority: Mapping[str, Any],
    output_root: Path,
    seed: int = PDM_RANDOMIZATION_SEED,
    check: bool = False,
) -> dict[str, Any]:
    """Create a compact truth-only native-support to PDMsh-row prerequisite."""

    if seed != PDM_RANDOMIZATION_SEED:
        raise NativeProfileTruthError("PDMsh permutation seed must remain 42")
    entry = _entry(authority, case_id)
    velocity_arrays, _, velocity_identity = _load_velocity_stencil(entry, case_id)
    support_raw = velocity_arrays["support_raw_point_ids"]
    support_compact, validity, validity_identity = _load_validity(
        entry, case_id, support_raw
    )
    valid_count = int(validity["valid_count"])
    rows, permutation_sha, torch_version = _regenerate_support_rows(
        support_compact=support_compact, valid_count=valid_count, seed=seed
    )
    arrays = {
        "support_raw_point_ids": support_raw,
        "support_compact_point_ids": support_compact,
        "support_pdmsh_rows": rows,
    }
    payload = _deterministic_npz_bytes(arrays)
    payload_path = output_root / "support-maps" / f"{case_id}.npz"
    payload_sha = _atomic_bytes(payload_path, payload, check=check)
    body: dict[str, Any] = {
        "schema": SUPPORT_MAP_SCHEMA,
        "schema_version": 1,
        "status": "complete_truth_only_prerequisite",
        "case_id": case_id,
        "authority_case_identity_sha256": entry["authority_case_identity_sha256"],
        "velocity_stencil_identity_sha256": velocity_identity,
        "validity_identity_sha256": validity_identity,
        "permutation": {
            "algorithm": "torch.randperm",
            "device": "cpu",
            "seed": seed,
            "count": valid_count,
            "dtype": "<i8",
            "sha256": permutation_sha,
            "torch_version": torch_version,
        },
        "support_count": len(support_raw),
        "support_raw_identity_sha256": array_identity(
            support_raw,
            namespace="fluidsbench-hiliftaeroml-volume-support-raw-v1",
        ),
        "support_compact_identity_sha256": array_identity(
            support_compact,
            namespace="fluidsbench-hiliftaeroml-volume-support-compact-v1",
        ),
        "support_pdmsh_row_identity_sha256": array_identity(
            rows,
            namespace="fluidsbench-hiliftaeroml-volume-support-pdmsh-row-v1",
        ),
        "payload": {
            "file": payload_path.name,
            "sha256": payload_sha,
            "byte_size": len(payload),
            "array_order": list(SUPPORT_MAP_ARRAYS),
        },
        "prediction_bearing_evaluator_outputs_used_as_source": False,
    }
    body["support_map_identity_sha256"] = _document_identity(
        "fluidsbench-hiliftaeroml-pdmsh-support-map-v1", body
    )
    _atomic_json(output_root / "support-maps" / f"{case_id}.json", body, check=check)
    return body


def _load_support_map(
    *, case_id: str, entry: Mapping[str, Any], support_map_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    record_path = support_map_root / "support-maps" / f"{case_id}.json"
    record, record_sha = load_json(record_path, label=f"{case_id} support map")
    identity = record.get("support_map_identity_sha256")
    identity_body = dict(record)
    identity_body.pop("support_map_identity_sha256", None)
    if (
        record.get("schema") != SUPPORT_MAP_SCHEMA
        or record.get("case_id") != case_id
        or record.get("authority_case_identity_sha256")
        != entry.get("authority_case_identity_sha256")
        or record.get("prediction_bearing_evaluator_outputs_used_as_source")
        is not False
        or identity
        != _document_identity(
            "fluidsbench-hiliftaeroml-pdmsh-support-map-v1", identity_body
        )
    ):
        raise NativeProfileTruthError(f"{case_id} support-map record differs")
    descriptor = record.get("payload")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("file") != f"{case_id}.npz"
    ):
        raise NativeProfileTruthError(f"{case_id} support-map payload differs")
    payload_path = _regular_file(
        record_path.parent / str(descriptor["file"]), f"{case_id} support-map payload"
    )
    if payload_path.stat().st_size != descriptor.get("byte_size") or sha256_file(
        payload_path
    ) != descriptor.get("sha256"):
        raise NativeProfileTruthError(f"{case_id} support-map payload bytes differ")
    with np.load(payload_path, allow_pickle=False) as archive:
        if tuple(archive.files) != SUPPORT_MAP_ARRAYS:
            raise NativeProfileTruthError(f"{case_id} support-map arrays differ")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name, namespace, identity_key in (
        (
            "support_raw_point_ids",
            "fluidsbench-hiliftaeroml-volume-support-raw-v1",
            "support_raw_identity_sha256",
        ),
        (
            "support_compact_point_ids",
            "fluidsbench-hiliftaeroml-volume-support-compact-v1",
            "support_compact_identity_sha256",
        ),
        (
            "support_pdmsh_rows",
            "fluidsbench-hiliftaeroml-volume-support-pdmsh-row-v1",
            "support_pdmsh_row_identity_sha256",
        ),
    ):
        if array_identity(arrays[name], namespace=namespace) != record.get(
            identity_key
        ):
            raise NativeProfileTruthError(f"{case_id} support-map {name} differs")
    return arrays, record, record_sha


@dataclass(frozen=True)
class _VTUArraySpec:
    dtype: np.dtype
    components: int
    tuples: int
    header_offset: int
    payload_offset: int

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            (self.tuples,) if self.components == 1 else (self.tuples, self.components)
        )

    @property
    def nbytes(self) -> int:
        return self.tuples * self.components * self.dtype.itemsize


_VTK_DTYPES = {
    "Int8": np.dtype("i1"),
    "UInt8": np.dtype("u1"),
    "Int16": np.dtype("<i2"),
    "UInt16": np.dtype("<u2"),
    "Int32": np.dtype("<i4"),
    "UInt32": np.dtype("<u4"),
    "Int64": np.dtype("<i8"),
    "UInt64": np.dtype("<u8"),
    "Float32": np.dtype("<f4"),
    "Float64": np.dtype("<f8"),
}


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if child.tag.rsplit("}", 1)[-1] == name]


def _single(element: ET.Element, name: str, label: str) -> ET.Element:
    found = _children(element, name)
    if len(found) != 1:
        raise NativeProfileTruthError(f"expected one {name} below {label}")
    return found[0]


def _vtu_preamble(
    path: Path, maximum_bytes: int = 16 * 1024 * 1024
) -> tuple[ET.Element, int]:
    marker = b"<AppendedData"
    buffer = bytearray()
    with path.open("rb") as stream:
        while len(buffer) < maximum_bytes:
            block = stream.read(min(64 * 1024, maximum_bytes - len(buffer)))
            if not block:
                break
            buffer.extend(block)
            marker_at = buffer.find(marker)
            if marker_at < 0:
                continue
            tag_end = buffer.find(b">", marker_at + len(marker))
            if tag_end < 0:
                continue
            sentinel = tag_end + 1
            while sentinel < len(buffer) and buffer[sentinel] in b" \t\r\n":
                sentinel += 1
            if sentinel == len(buffer):
                continue
            if buffer[sentinel] != ord("_"):
                raise NativeProfileTruthError("VTU appended-data sentinel differs")
            try:
                root = ET.fromstring(  # noqa: S314 - frozen local VTK XML preamble
                    bytes(buffer[: tag_end + 1]) + b"</AppendedData></VTKFile>"
                )
            except ET.ParseError as error:
                raise NativeProfileTruthError(
                    f"invalid VTU preamble: {error}"
                ) from error
            return root, sentinel + 1
    raise NativeProfileTruthError("VTU raw-appended preamble is absent")


class _SparseVTUReader:
    def __init__(self, path: Path) -> None:
        self.path = _regular_file(path, "native VTU")
        root, appended_base = _vtu_preamble(self.path)
        if (
            root.tag.rsplit("}", 1)[-1] != "VTKFile"
            or root.get("type") != "UnstructuredGrid"
            or root.get("byte_order") != "LittleEndian"
            or root.get("header_type") != "UInt64"
            or root.get("compressor")
        ):
            raise NativeProfileTruthError("unsupported native VTU encoding")
        unstructured = _single(root, "UnstructuredGrid", "VTKFile")
        piece = _single(unstructured, "Piece", "UnstructuredGrid")
        self.number_of_points = int(str(piece.get("NumberOfPoints")))
        if self.number_of_points <= 0:
            raise NativeProfileTruthError("native VTU point count is invalid")
        appended = _single(root, "AppendedData", "VTKFile")
        if appended.get("encoding") != "raw":
            raise NativeProfileTruthError("native VTU appended encoding differs")

        def parse(element: ET.Element) -> _VTUArraySpec:
            vtk_type = element.get("type")
            if vtk_type not in _VTK_DTYPES or element.get("format") != "appended":
                raise NativeProfileTruthError("native VTU array encoding differs")
            components = int(element.get("NumberOfComponents", "1"))
            offset = int(str(element.get("offset")))
            header = appended_base + offset
            return _VTUArraySpec(
                dtype=_VTK_DTYPES[vtk_type],
                components=components,
                tuples=self.number_of_points,
                header_offset=header,
                payload_offset=header + 8,
            )

        points = _single(piece, "Points", "Piece")
        coordinate_elements = _children(points, "DataArray")
        if len(coordinate_elements) != 1:
            raise NativeProfileTruthError("native VTU coordinate array differs")
        coordinate_spec = parse(coordinate_elements[0])
        declared = self._declared(coordinate_spec)
        if declared == coordinate_spec.nbytes:
            self._header_bytes = 0
        elif declared == coordinate_spec.nbytes + 8:
            self._header_bytes = 8
        else:
            raise NativeProfileTruthError("native VTU byte-count convention differs")
        sections = _children(piece, "PointData")
        if len(sections) != 1:
            raise NativeProfileTruthError("native VTU PointData section differs")
        self.specs: dict[str, _VTUArraySpec] = {}
        for element in _children(sections[0], "DataArray"):
            name = element.get("Name")
            if not name or name in self.specs:
                raise NativeProfileTruthError("native VTU PointData names differ")
            self.specs[name] = parse(element)

    def _declared(self, spec: _VTUArraySpec) -> int:
        with self.path.open("rb", buffering=0) as stream:
            stream.seek(spec.header_offset)
            raw = stream.read(8)
        if len(raw) != 8:
            raise NativeProfileTruthError("native VTU block header is truncated")
        return struct.unpack("<Q", raw)[0]

    def gather(
        self, name: str, indices: np.ndarray
    ) -> tuple[np.ndarray, _VTUArraySpec]:
        if name not in self.specs:
            raise NativeProfileTruthError(f"native VTU lacks PointData {name!r}")
        spec = self.specs[name]
        if self._declared(spec) != spec.nbytes + self._header_bytes:
            raise NativeProfileTruthError(f"native VTU {name} byte count differs")
        selected = np.asarray(indices, dtype=np.int64)
        if (
            selected.ndim != 1
            or (selected.size and np.any(np.diff(selected) <= 0))
            or (
                selected.size
                and (selected[0] < 0 or selected[-1] >= self.number_of_points)
            )
        ):
            raise NativeProfileTruthError(f"native VTU {name} sparse IDs differ")
        source = np.memmap(
            self.path,
            mode="r",
            dtype=spec.dtype,
            offset=spec.payload_offset,
            shape=spec.shape,
        )
        try:
            result = np.asarray(source[selected]).copy()
        finally:
            mmap = getattr(source, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return result, spec


def _freestream(
    entry: Mapping[str, Any], case_id: str
) -> tuple[dict[str, float], dict[str, str]]:
    source = entry.get("volume_truth_source")
    global_data = source.get("global_data") if isinstance(source, Mapping) else None
    if not isinstance(global_data, Mapping) or set(global_data) != {
        f"{name}.memmap" for name in GLOBAL_DATA_NAMES
    }:
        raise NativeProfileTruthError(f"{case_id} global-data inventory differs")
    values: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in GLOBAL_DATA_NAMES:
        path, digest = _load_absolute_descriptor(
            global_data[f"{name}.memmap"], f"{case_id} global data {name}"
        )
        count = 3 if name == "U_inf" else 1
        if path.stat().st_size != count * 4:
            raise NativeProfileTruthError(f"{case_id} global data {name} size differs")
        values[name] = np.asarray(
            np.memmap(path, mode="r", dtype="<f4", shape=(count,))
        ).copy()
        digests[name] = digest
    u_inf = values["U_inf"]
    speed = math.sqrt(sum(float(value) ** 2 for value in u_inf))
    rho = float(values["rho_inf"][0])
    result = {
        "p_inf": float(values["p_inf"][0]),
        "rho_inf": rho,
        "T_inf": float(values["T_inf"][0]),
        "L_ref": float(values["L_ref"][0]),
        "u_inf_magnitude": speed,
        "q_inf": 0.5 * rho * speed**2,
    }
    if not all(
        math.isfinite(value) and value > 0 for value in (speed, rho, result["q_inf"])
    ):
        raise NativeProfileTruthError(f"{case_id} freestream differs")
    return result, digests


def _reconstruct_cp(
    stencil: Mapping[str, np.ndarray], native_cp: np.ndarray
) -> np.ndarray:
    support = np.unique(stencil["cut_support_node_ids"])
    slots = np.searchsorted(support, stencil["cut_support_node_ids"])
    weighted = stencil["cut_support_weights"] * native_cp[slots]
    return np.asarray(
        np.add.reduceat(weighted, stencil["cut_support_offsets"][:-1]),
        dtype=np.float64,
    )


def _reconstruct_velocity(
    stencil: Mapping[str, np.ndarray], support_values: np.ndarray
) -> np.ndarray:
    support = stencil["support_raw_point_ids"]
    raw_ids = stencil["raw_point_ids"]
    slots = np.searchsorted(support, raw_ids)
    if raw_ids.size and not np.array_equal(support[slots], raw_ids):
        raise NativeProfileTruthError("velocity CSR escaped sparse support")
    offsets = stencil["csr_offsets"]
    valid = stencil["valid_mask"]
    output = np.full((len(valid), 3), np.nan, dtype=np.float64)
    for row in np.flatnonzero(valid):
        begin = int(offsets[row])
        end = int(offsets[row + 1])
        output[row] = np.tensordot(
            np.asarray(stencil["interpolation_weights"][begin:end], dtype=np.float64),
            np.asarray(support_values[slots[begin:end]], dtype=np.float64),
            axes=(0, 0),
        )
    return output


def _polyline_weights(points: np.ndarray) -> np.ndarray:
    segment = np.linalg.norm(
        np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=1
    )
    if np.any(segment <= 0) or not np.all(np.isfinite(segment)):
        raise NativeProfileTruthError("velocity profile contains a degenerate segment")
    weights = np.empty(len(points), dtype=np.float64)
    weights[0] = 0.5 * segment[0]
    weights[-1] = 0.5 * segment[-1]
    weights[1:-1] = 0.5 * (segment[:-1] + segment[1:])
    return weights


def _line_weights(stencil: Mapping[str, np.ndarray]) -> np.ndarray:
    points = stencil["requested_xyz_in"]
    valid = stencil["valid_mask"]
    offsets = stencil["station_row_offsets"]
    result = np.zeros(len(valid), dtype=np.float64)
    for station_start, station_stop in zip(offsets[:-1], offsets[1:], strict=True):
        local_valid = valid[station_start:station_stop]
        transitions = np.diff(
            np.concatenate(([False], local_valid, [False])).astype(np.int8)
        )
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        for local_start, local_stop in zip(starts, stops, strict=True):
            if local_stop - local_start < 2:
                continue
            begin = int(station_start + local_start)
            end = int(station_start + local_stop)
            result[begin:end] = _polyline_weights(points[begin:end])
    return result


def load_compact_profile_support_inputs(
    *,
    case_id: str,
    authority: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Load prediction-free native support needed by compact profile-v2.

    The compact sampling rule needs geometry/topology from the frozen Cp
    stencil and coordinates, gaps, and line weights from the frozen velocity
    stencil.  This helper deliberately does not read a model-output directory
    or either full truth field.  The caller joins truth values from the
    separately validated native-profile truth release.

    At most one case's stencil payloads are resident at a time.  That makes
    this entry point suitable for streaming the complete 1,355-case public
    plotting projection without recreating the 1.755 GB lossless archive in
    memory.
    """

    if SAFE_CASE_ID.fullmatch(case_id) is None:
        raise NativeProfileTruthError(f"invalid case ID {case_id!r}")
    entry = _entry(authority, case_id)
    cp_stencil, _cp_record, cp_identity = _load_cp_stencil(entry, case_id)
    velocity_stencil, _velocity_record, velocity_identity = (
        _load_velocity_stencil(entry, case_id)
    )
    _support_compact, _validity, validity_identity = _load_validity(
        entry, case_id, velocity_stencil["support_raw_point_ids"]
    )

    cp_arrays = {
        name: cp_stencil[name]
        for name in (
            "cut_xyz_in",
            "branch_closed",
            "branch_vertex_offsets",
            "branch_vertex_ids",
            "branch_segment_offsets",
            "segment_lengths_in",
            "branch_row_code",
            "branch_graph_component_code",
            "branch_component_code",
            "branch_plane_piece_code",
            "branch_side_code",
            "branch_topology_patch_code",
        )
    }
    velocity_arrays = {
        "requested_xyz_in": velocity_stencil["requested_xyz_in"],
        "valid_mask": velocity_stencil["valid_mask"],
        "station_names": velocity_stencil["station_names"],
        "station_row_offsets": velocity_stencil["station_row_offsets"],
        "line_length_weights_in": _line_weights(velocity_stencil),
    }
    evidence = {
        "authority_case_identity_sha256": _sha(
            entry.get("authority_case_identity_sha256"),
            f"{case_id} authority identity",
        ),
        "cp_stencil_identity_sha256": cp_identity,
        "velocity_stencil_identity_sha256": velocity_identity,
        "validity_identity_sha256": validity_identity,
    }
    return cp_arrays, velocity_arrays, evidence


def _oracle_exact(
    *, case_id: str, oracle_case_root: Path, truth_cp: np.ndarray, velocity: np.ndarray
) -> dict[str, Any]:
    cp_path = oracle_case_root / "surface_submission_stream" / "cp_cut_values.npz"
    velocity_path = (
        oracle_case_root / "volume_submission_stream" / "velocity_profiles.npz"
    )
    with np.load(cp_path, allow_pickle=False) as archive:
        oracle_cp = np.asarray(archive["truth_cp"])
    with np.load(velocity_path, allow_pickle=False) as archive:
        oracle_velocity = np.asarray(archive["reference_velocity_nd"])
    if not np.array_equal(truth_cp, oracle_cp) or not np.array_equal(
        velocity, oracle_velocity, equal_nan=True
    ):
        raise NativeProfileTruthError(
            f"{case_id} truth-only replay differs from oracle"
        )
    return {
        "status": "exact_array_equality",
        "case_id": case_id,
        "cp_oracle_sha256": sha256_file(cp_path),
        "velocity_oracle_sha256": sha256_file(velocity_path),
        "oracle_used_as_truth_source": False,
    }


def materialize_case_truth(
    *,
    case_id: str,
    authority: Mapping[str, Any],
    output_root: Path,
    support_map_root: Path | None = None,
    oracle_case_root: Path | None = None,
    check: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Materialize exact Cp/velocity truth using only sparse source-field reads."""

    if SAFE_CASE_ID.fullmatch(case_id) is None:
        raise NativeProfileTruthError(f"invalid case ID {case_id!r}")
    entry = _entry(authority, case_id)
    cp_stencil, cp_record, cp_identity = _load_cp_stencil(entry, case_id)
    velocity_stencil, velocity_record, velocity_identity = _load_velocity_stencil(
        entry, case_id
    )
    support_compact, validity, validity_identity = _load_validity(
        entry, case_id, velocity_stencil["support_raw_point_ids"]
    )
    freestream, freestream_digests = _freestream(entry, case_id)

    surface_source = entry.get("surface_truth_source")
    if not isinstance(surface_source, Mapping):
        raise NativeProfileTruthError(f"{case_id} surface truth source is absent")
    surface_vtu = _validate_stat(surface_source.get("vtu"), f"{case_id} surface VTU")
    surface_reader = _SparseVTUReader(surface_vtu)
    if surface_reader.number_of_points != surface_source.get("point_count"):
        raise NativeProfileTruthError(f"{case_id} surface point count differs")
    cp_support = np.unique(cp_stencil["cut_support_node_ids"])
    raw_pressure, pressure_spec = surface_reader.gather(
        str(surface_source.get("pressure_field")), cp_support
    )
    if raw_pressure.dtype != np.dtype("<f4") or raw_pressure.shape != cp_support.shape:
        raise NativeProfileTruthError(f"{case_id} sparse pressure field differs")
    native_cp = (
        np.asarray(raw_pressure, dtype=np.float32) - freestream["p_inf"]
    ) / freestream["q_inf"]
    truth_cp = _reconstruct_cp(cp_stencil, native_cp)

    volume_source = entry.get("volume_truth_source")
    if not isinstance(volume_source, Mapping) or not isinstance(
        volume_source.get("vtu"), Mapping
    ):
        raise NativeProfileTruthError(f"{case_id} volume truth source is absent")
    volume_vtu_record = volume_source["vtu"]
    volume_vtu = _validate_stat(
        {key: volume_vtu_record[key] for key in ("file", "byte_size", "mtime_ns")},
        f"{case_id} volume VTU",
    )
    volume_reader = _SparseVTUReader(volume_vtu)
    if volume_reader.number_of_points != validity.get("raw_count"):
        raise NativeProfileTruthError(f"{case_id} volume point count differs")
    raw_velocity, velocity_spec = volume_reader.gather(
        "avg(u)", velocity_stencil["support_raw_point_ids"]
    )
    if raw_velocity.dtype != np.dtype("<f4") or raw_velocity.shape != (
        len(support_compact),
        3,
    ):
        raise NativeProfileTruthError(f"{case_id} sparse velocity field differs")
    reference_support_nd = (
        np.asarray(raw_velocity, dtype=np.float32) / freestream["u_inf_magnitude"]
    )
    reference_velocity = _reconstruct_velocity(velocity_stencil, reference_support_nd)
    line_weights = _line_weights(velocity_stencil)

    cp_arrays = {
        name: cp_stencil[name]
        for name in CP_SOURCE_ARRAYS
        if name not in {"prediction_cp", "truth_cp"}
    }
    cp_arrays["truth_cp"] = truth_cp
    graph_count = len(
        set(
            zip(
                cp_stencil["branch_row_code"].tolist(),
                cp_stencil["branch_graph_component_code"].tolist(),
                strict=True,
            )
        )
    )
    cp_metadata = {
        "vertex_count": len(cp_stencil["cut_xyz_in"]),
        "segment_count": len(cp_stencil["segment_vertex_ids"]),
        "branch_count": len(cp_stencil["branch_closed"]),
        "graph_count": graph_count,
        "code_catalogs": cp_record["catalogs"],
    }
    velocity_arrays = {
        "requested_xyz_in": velocity_stencil["requested_xyz_in"],
        "valid_mask": velocity_stencil["valid_mask"],
        "station_names": velocity_stencil["station_names"],
        "station_row_offsets": velocity_stencil["station_row_offsets"],
        "line_length_weights_in": line_weights,
        "reference_velocity_nd": reference_velocity,
    }
    velocity_metadata = {
        "row_count": len(reference_velocity),
        "valid_row_count": int(np.count_nonzero(velocity_stencil["valid_mask"])),
        "invalid_row_count": int(np.count_nonzero(~velocity_stencil["valid_mask"])),
    }
    authority_identity = _sha(
        entry.get("authority_case_identity_sha256"),
        f"{case_id} authority identity",
    )
    surface_provenance = {
        "authority_kind": AUTHORITY_KIND,
        "cp_stencil_identity_sha256": cp_identity,
        "surface_geometry_sha256": surface_source["geometry_sha256"],
        "surface_normal_sha256": surface_source["normal_sha256"],
        "surface_vtu_stat": dict(surface_source["vtu"]),
        "pressure_field": surface_source["pressure_field"],
        "pressure_field_dtype": pressure_spec.dtype.str,
        "pressure_field_shape": list(pressure_spec.shape),
        "selected_native_pressure_identity_sha256": array_identity(
            raw_pressure,
            namespace="fluidsbench-hiliftaeroml-selected-native-pressure-v1",
        ),
        "freestream_file_sha256": freestream_digests,
        "prediction_bearing_evaluator_outputs_used_as_source": False,
    }
    volume_provenance = {
        "authority_kind": AUTHORITY_KIND,
        "velocity_stencil_identity_sha256": velocity_identity,
        "validity_identity_sha256": validity_identity,
        "frozen_volume_vtu_sha256": volume_source["vtu"]["frozen_sha256"],
        "volume_vtu_stat": {
            key: volume_vtu_record[key] for key in ("file", "byte_size", "mtime_ns")
        },
        "velocity_field": "avg(u)",
        "velocity_field_dtype": velocity_spec.dtype.str,
        "velocity_field_shape": list(velocity_spec.shape),
        "selected_native_velocity_identity_sha256": array_identity(
            raw_velocity,
            namespace="fluidsbench-hiliftaeroml-selected-native-velocity-v1",
        ),
        "freestream_file_sha256": freestream_digests,
        "prediction_bearing_evaluator_outputs_used_as_source": False,
    }
    record = build_case_truth_from_arrays(
        case_id=case_id,
        cp_arrays=cp_arrays,
        cp_metadata=cp_metadata,
        velocity_arrays=velocity_arrays,
        velocity_metadata=velocity_metadata,
        surface_source=surface_provenance,
        volume_source=volume_provenance,
        authority_identity_sha256=authority_identity,
        output_root=output_root,
        check=check,
    )
    audit: dict[str, Any] = {}
    if support_map_root is not None:
        support_map, support_record, support_record_sha = _load_support_map(
            case_id=case_id, entry=entry, support_map_root=support_map_root
        )
        if not np.array_equal(
            support_map["support_raw_point_ids"],
            velocity_stencil["support_raw_point_ids"],
        ) or not np.array_equal(
            support_map["support_compact_point_ids"], support_compact
        ):
            raise NativeProfileTruthError(f"{case_id} support map/stencil differs")
        pdmsh = volume_source.get("pdmsh")
        avg_u = pdmsh.get("avg_u") if isinstance(pdmsh, Mapping) else None
        if not isinstance(avg_u, Mapping):
            raise NativeProfileTruthError(f"{case_id} PDMsh velocity source is absent")
        avg_u_path = _validate_stat(
            {key: avg_u[key] for key in ("file", "byte_size", "mtime_ns")},
            f"{case_id} PDMsh avg(u)",
        )
        valid_count = int(validity["valid_count"])
        pdmsh_source = np.memmap(
            avg_u_path, mode="r", dtype="<f4", shape=(valid_count, 3)
        )
        try:
            pdmsh_velocity = np.asarray(
                pdmsh_source[support_map["support_pdmsh_rows"]]
            ).copy()
        finally:
            mmap = getattr(pdmsh_source, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if not np.array_equal(raw_velocity, pdmsh_velocity):
            raise NativeProfileTruthError(
                f"{case_id} native/PDMsh sparse velocity differs"
            )
        audit["pdmsh_sparse_crosscheck"] = {
            "status": "exact_array_equality",
            "support_map_record_sha256": support_record_sha,
            "support_map_identity_sha256": support_record[
                "support_map_identity_sha256"
            ],
            "support_count": len(raw_velocity),
            "used_as_truth_source": False,
        }
    if oracle_case_root is not None:
        audit["evaluator_output_oracle"] = _oracle_exact(
            case_id=case_id,
            oracle_case_root=oracle_case_root,
            truth_cp=truth_cp,
            velocity=reference_velocity,
        )
    return record, audit or None


def prerequisite_preflight(
    *, universe: CaseUniverse, authority: Mapping[str, Any], authority_sha256: str
) -> dict[str, Any]:
    cases = authority.get("cases")
    missing = authority.get("missing")
    if not isinstance(cases, Mapping) or not isinstance(missing, Mapping):
        raise NativeProfileTruthError("authority preflight inventory differs")
    support_counts: list[int] = []
    valid_counts: list[int] = []
    cp_vertex_counts: list[int] = []
    for case_id, entry in cases.items():
        cp_stencil, _, _ = _load_cp_stencil(entry, case_id)
        velocity, _, _ = _load_velocity_stencil(entry, case_id)
        support_compact, validity, _ = _load_validity(
            entry, case_id, velocity["support_raw_point_ids"]
        )
        support_counts.append(len(support_compact))
        valid_counts.append(int(validity["valid_count"]))
        cp_vertex_counts.append(len(cp_stencil["cut_xyz_in"]))
    estimated_truth_bytes = sum(count * 8 for count in cp_vertex_counts) + len(
        cases
    ) * (len(STATIONS) * ROWS_PER_STATION * 3 * 8)
    return {
        "schema": "hiliftaeroml-native-profile-truth-prerequisite-preflight-v1-candidate",
        "status": "ready" if not missing else "blocked_missing_prerequisites",
        "authority_index_sha256": _sha(authority_sha256, "authority index digest"),
        "case_count": len(universe.case_ids),
        "case_set_count": len(universe.case_sets),
        "available_case_count": len(cases),
        "missing_case_count": len(missing),
        "missing": dict(missing),
        "support_count_range": (
            [min(support_counts), max(support_counts)] if support_counts else None
        ),
        "cp_cut_vertex_count_range": (
            [min(cp_vertex_counts), max(cp_vertex_counts)] if cp_vertex_counts else None
        ),
        "pdmsh_valid_count_range": (
            [min(valid_counts), max(valid_counts)] if valid_counts else None
        ),
        "uncompressed_truth_array_bytes": estimated_truth_bytes,
        "source_io_contract": {
            "surface": "random-access only unique Cp stencil support pressure values",
            "volume": "random-access only unique velocity-stencil support avg(u) values from the frozen native VTU",
            "full_vtu_or_pdmsh_field_scan_per_case": False,
            "model_inference": False,
            "prediction_outputs_as_authority": False,
        },
        "case_array_job": {
            "one_case_per_task": True,
            "recommended_cpus_per_task": 1,
            "recommended_memory_gib": 4,
            "recommended_walltime": "00:05:00",
            "safe_initial_parallelism": 8,
            "dominant_resource": "shared-filesystem sparse reads and stencil NPZ decompression",
            "launch_allowed_by_preflight": False,
            "reason": "owner review is required before the 1,355-case export",
        },
        "activation_changed": False,
        "owner_approval_complete": False,
        "published": False,
    }


__all__ = [
    "AUTHORITY_KIND",
    "AUTHORITY_SCHEMA",
    "SUPPORT_MAP_SCHEMA",
    "build_prerequisite_authority_index",
    "build_support_map",
    "load_prerequisite_authority_index",
    "load_compact_profile_support_inputs",
    "materialize_case_truth",
    "prerequisite_preflight",
]
