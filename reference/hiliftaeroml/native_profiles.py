"""Lossless prediction-only HiLiftAeroML native profile serialization.

The native evaluator writes two NPZ artifacts per case.  The surface artifact
contains a disconnected Cp cut graph and the volume artifact contains five
fixed velocity stations with explicit invalid gaps.  Flattening either artifact
into the generic profile-v1 curve shape changes the scientific support.  This
module validates the evaluator-native artifacts and emits deterministic,
prediction-only NPZ sidecars without joining graphs, filling gaps, or retaining
ground truth.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "benchmark-specs" / "hiliftaeroml" / "native-profile-format-v1.json"
)
PROFILE_FORMAT = "fluidsbench-hiliftaeroml-native-profile-chunks-v1-candidate"
PROFILE_CONTRACT_ID = "hiliftaeroml-native-profile-predictions-v1-candidate"
PROFILE_CONTRACT_SHA256 = (
    "39a1e79edf951b3e4f5d1b5fa456cbd1fad524e93dac1bdca58534e01647ae25"
)
PROFILE_CHUNK_SCHEMA = "hiliftaeroml-native-profile-chunk-v1-candidate"
SAFE_CASE_ID = re.compile(r"^geo_LHC[0-9]{3}_AoA_(?:4|6|8|10|12|14|16|18|20|22)$")
STATIONS = ("B.2", "B.3", "C.1", "C.2", "C.3")
ROWS = tuple("ABCDEFGHIJ")
ROWS_PER_STATION = 801

CP_SOURCE_ARRAYS = (
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
    "cut_xyz_in",
    "prediction_cp",
    "segment_lengths_in",
    "segment_plane_piece_code",
    "segment_vertex_ids",
    "truth_cp",
)
CP_PUBLISHED_ARRAYS = tuple(name for name in CP_SOURCE_ARRAYS if name != "truth_cp")
VELOCITY_SOURCE_ARRAYS = (
    "requested_xyz_in",
    "valid_mask",
    "station_names",
    "station_row_offsets",
    "line_length_weights_in",
    "predicted_velocity_nd",
    "reference_velocity_nd",
    "predicted_velocity_physical",
    "reference_velocity_physical",
    "predicted_velocity_dimensional",
    "reference_velocity_dimensional",
    "predicted_velocity_magnitude_nd",
    "reference_velocity_magnitude_nd",
    "u_x_over_U_inf_pred",
    "u_x_over_U_inf_reference",
    "u_y_over_U_inf_pred",
    "u_y_over_U_inf_reference",
    "u_z_over_U_inf_pred",
    "u_z_over_U_inf_reference",
    "speed_over_U_inf_pred",
    "speed_over_U_inf_reference",
    "u_inf_magnitude_physical",
    "support_raw_point_ids",
    "support_compact_point_ids",
    "predicted_velocity_support_nd",
    "reference_velocity_support_nd",
    "predicted_velocity_support_physical",
    "reference_velocity_support_physical",
    "u_inf_vector_physical",
    "velocity_physical_units",
    "velocity_nondimensionalization",
    "context_ceiling",
    "partition_seed",
    "run_fingerprint",
)
VELOCITY_PUBLISHED_ARRAYS = (
    "requested_xyz_in",
    "valid_mask",
    "station_names",
    "station_row_offsets",
    "line_length_weights_in",
    "predicted_velocity_nd",
    "predicted_velocity_magnitude_nd",
    "support_raw_point_ids",
    "support_compact_point_ids",
    "predicted_velocity_support_nd",
    "u_inf_vector_physical",
    "velocity_nondimensionalization",
    "context_ceiling",
    "partition_seed",
    "run_fingerprint",
)


class NativeProfileError(ValueError):
    """Raised when a native profile cannot be serialized without changing it."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeProfileError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise NativeProfileError(f"JSON contains forbidden non-finite token {token}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeProfileError(f"cannot encode canonical JSON: {error}") from error


def write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise NativeProfileError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    _require_regular(path, label)
    before = path.stat()
    digest = sha256_file(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except NativeProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeProfileError(f"cannot read {label} {path}: {error}") from error
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(path) != digest:
        raise NativeProfileError(f"{label} changed while it was read: {path}")
    if not isinstance(value, dict):
        raise NativeProfileError(f"{label} must contain one JSON object")
    return value, digest


def _require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise NativeProfileError(f"{label} must be a regular non-symlink file: {path}")


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeProfileError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NativeProfileError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise NativeProfileError(f"{label} must be finite")
    return result


def _load_npz_exact(
    path: Path, *, expected_arrays: Sequence[str], label: str
) -> tuple[dict[str, np.ndarray], str]:
    _require_regular(path, label)
    before = path.stat()
    digest = sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != list(expected_arrays):
                raise NativeProfileError(
                    f"{label} array inventory or order differs: {archive.files}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in expected_arrays}
    except NativeProfileError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise NativeProfileError(f"cannot read {label} {path}: {error}") from error
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(path) != digest:
        raise NativeProfileError(f"{label} changed while it was read: {path}")
    return arrays, digest


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype[Any] | type[Any] | None = None,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    value = arrays[name]
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise NativeProfileError(
            f"{name} dtype must be {np.dtype(dtype)}, observed {value.dtype}"
        )
    if ndim is not None and value.ndim != ndim:
        raise NativeProfileError(f"{name} must have {ndim} dimensions")
    if shape is not None and (
        value.ndim != len(shape)
        or any(expected is not None and observed != expected for observed, expected in zip(value.shape, shape, strict=True))
    ):
        raise NativeProfileError(f"{name} shape must match {shape}, observed {value.shape}")
    return value


def _validate_offsets(value: np.ndarray, *, end: int, label: str) -> None:
    if value.ndim != 1 or value.dtype != np.int64 or len(value) < 2:
        raise NativeProfileError(f"{label} must be a non-empty int64 offset vector")
    if value[0] != 0 or value[-1] != end or np.any(np.diff(value) <= 0):
        raise NativeProfileError(
            f"{label} must start at zero, end at {end}, and increase strictly"
        )


def _validate_identity(
    identity: Any, *, path: Path, digest: str, label: str, filename_key: str
) -> None:
    if not isinstance(identity, Mapping):
        raise NativeProfileError(f"{label} identity is absent")
    declared_filename = identity.get(filename_key)
    expected_filename = path.name
    if (
        not isinstance(declared_filename, str)
        or Path(declared_filename).name != expected_filename
    ):
        raise NativeProfileError(f"{label} filename differs")
    if identity.get("sha256") != digest:
        raise NativeProfileError(f"{label} SHA-256 differs")
    if identity.get("size_bytes") != path.stat().st_size:
        raise NativeProfileError(f"{label} byte size differs")
    declared_path = identity.get("path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != path.resolve():
        raise NativeProfileError(f"{label} path differs")


def _catalogs(metrics: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = metrics.get("metrics", {}).get("catalogs")
    required = {
        "components",
        "plane_pieces",
        "rows",
        "sides",
        "topology_patches",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise NativeProfileError("Cp code catalogs differ")
    result: dict[str, list[str]] = {}
    for key in sorted(required):
        values = raw[key]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise NativeProfileError(f"Cp catalog {key} is invalid")
        result[key] = list(values)
    if result["rows"] != list(ROWS):
        raise NativeProfileError("Cp row catalog must be exact A-J")
    return result


def validate_cp_source(
    *,
    metrics_path: Path,
    npz_path: Path,
    expected_metrics_sha256: str | None = None,
    expected_npz_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metrics, metrics_sha = load_json(metrics_path, label="Cp metrics")
    arrays, source_sha = _load_npz_exact(
        npz_path, expected_arrays=CP_SOURCE_ARRAYS, label="Cp native NPZ"
    )
    if expected_metrics_sha256 is not None and metrics_sha != _require_sha(
        expected_metrics_sha256, "receipt-bound Cp metrics digest"
    ):
        raise NativeProfileError("Cp metrics SHA-256 differs from its retained receipt")
    if expected_npz_sha256 is not None and source_sha != _require_sha(
        expected_npz_sha256, "receipt-bound Cp NPZ digest"
    ):
        raise NativeProfileError("Cp NPZ SHA-256 differs from its retained receipt")
    if (
        metrics.get("profile_metric_schema_id") != "hilift_native_cp_line_metrics_v3"
        or metrics.get("status") != "complete"
    ):
        raise NativeProfileError("Cp metrics schema/status differs")
    _validate_identity(
        metrics.get("cut_values"),
        path=npz_path,
        digest=source_sha,
        label="Cp native NPZ",
        filename_key="path",
    )
    # The native record uses ``path`` rather than ``filename`` for this identity.
    # Re-check the basename separately while retaining exact absolute-path binding.
    if Path(str(metrics["cut_values"]["path"])).name != npz_path.name:
        raise NativeProfileError("Cp native NPZ basename differs")

    branch_closed = _require_array(arrays, "branch_closed", dtype=np.bool_, ndim=1)
    branch_count = len(branch_closed)
    for name, dtype in (
        ("branch_component_code", np.uint8),
        ("branch_plane_piece_code", np.uint8),
        ("branch_row_code", np.uint8),
        ("branch_side_code", np.uint8),
        ("branch_topology_patch_code", np.uint8),
    ):
        _require_array(arrays, name, dtype=dtype, shape=(branch_count,))
    graph_codes = _require_array(
        arrays, "branch_graph_component_code", dtype=np.int64, shape=(branch_count,)
    )
    if np.any(graph_codes < 0):
        raise NativeProfileError("Cp graph component codes must be non-negative")

    vertex_ids = _require_array(arrays, "branch_vertex_ids", dtype=np.int64, ndim=1)
    segment_ids = _require_array(
        arrays, "segment_vertex_ids", dtype=np.int64, shape=(None, 2)
    )
    segment_count = len(segment_ids)
    _validate_offsets(
        _require_array(
            arrays, "branch_vertex_offsets", dtype=np.int64, shape=(branch_count + 1,)
        ),
        end=len(vertex_ids),
        label="branch_vertex_offsets",
    )
    _validate_offsets(
        _require_array(
            arrays, "branch_segment_offsets", dtype=np.int64, shape=(branch_count + 1,)
        ),
        end=segment_count,
        label="branch_segment_offsets",
    )
    xyz = _require_array(arrays, "cut_xyz_in", dtype=np.float64, shape=(None, 3))
    vertex_count = len(xyz)
    prediction = _require_array(
        arrays, "prediction_cp", dtype=np.float64, shape=(vertex_count,)
    )
    truth = _require_array(arrays, "truth_cp", dtype=np.float64, shape=(vertex_count,))
    lengths = _require_array(
        arrays, "segment_lengths_in", dtype=np.float64, shape=(segment_count,)
    )
    _require_array(
        arrays, "segment_plane_piece_code", dtype=np.uint8, shape=(segment_count,)
    )
    if not all(np.all(np.isfinite(value)) for value in (xyz, prediction, truth, lengths)):
        raise NativeProfileError("Cp numeric arrays must be finite")
    if np.any(lengths <= 0.0):
        raise NativeProfileError("Cp segment lengths must be positive")
    for value, label in ((vertex_ids, "branch_vertex_ids"), (segment_ids, "segment_vertex_ids")):
        if np.any(value < 0) or np.any(value >= vertex_count):
            raise NativeProfileError(f"{label} contains an out-of-range vertex ID")

    catalogs = _catalogs(metrics)
    limits = {
        "branch_component_code": len(catalogs["components"]),
        "branch_plane_piece_code": len(catalogs["plane_pieces"]),
        "branch_row_code": len(catalogs["rows"]),
        "branch_side_code": len(catalogs["sides"]),
        "branch_topology_patch_code": len(catalogs["topology_patches"]),
        "segment_plane_piece_code": len(catalogs["plane_pieces"]),
    }
    for name, limit in limits.items():
        if np.any(arrays[name] >= limit):
            raise NativeProfileError(f"{name} exceeds its frozen code catalog")

    cut_values = metrics.get("cut_values", {})
    if cut_values.get("cut_vertex_count") != vertex_count:
        raise NativeProfileError("Cp metrics vertex count differs from its NPZ")
    all_rows = metrics.get("metrics", {}).get("all_rows")
    if not isinstance(all_rows, Mapping) or all_rows.get("rows") != list(ROWS):
        raise NativeProfileError("Cp all-rows metric scope differs")
    cp_r2 = _finite_number(
        all_rows.get("metrics", {}).get("r2_centered"), "Cp all-rows R2"
    )
    metadata = {
        "source_npz_sha256": source_sha,
        "source_metrics_sha256": metrics_sha,
        "source_metrics_schema": metrics["profile_metric_schema_id"],
        "run_fingerprint": _require_sha(metrics.get("run_fingerprint"), "Cp run fingerprint"),
        "stencil_content_sha256": _require_sha(
            metrics.get("stencil_content_sha256"), "Cp stencil content digest"
        ),
        "stencil_payload_sha256": _require_sha(
            metrics.get("stencil_payload_sha256"), "Cp stencil payload digest"
        ),
        "stencil_semantic_sha256": _require_sha(
            metrics.get("stencil_semantic_sha256"), "Cp stencil semantic digest"
        ),
        "vertex_count": vertex_count,
        "segment_count": segment_count,
        "branch_count": branch_count,
        "graph_count": len(set(zip(arrays["branch_row_code"].tolist(), graph_codes.tolist(), strict=True))),
        "code_catalogs": catalogs,
        "cp_cut_r2": cp_r2,
    }
    return arrays, metadata


def _require_nan_gap(array: np.ndarray, valid: np.ndarray, label: str) -> None:
    flattened = array if array.ndim == 1 else array.reshape(len(valid), -1)
    if not np.all(np.isfinite(flattened[valid])):
        raise NativeProfileError(f"{label} must be finite on every valid row")
    if not np.all(np.isnan(flattened[~valid])):
        raise NativeProfileError(f"{label} must retain NaN on every invalid row")


def validate_velocity_source(
    *,
    case_id: str,
    metrics_path: Path,
    npz_path: Path,
    expected_metrics_sha256: str | None = None,
    expected_npz_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metrics, metrics_sha = load_json(metrics_path, label="velocity metrics")
    arrays, source_sha = _load_npz_exact(
        npz_path, expected_arrays=VELOCITY_SOURCE_ARRAYS, label="velocity native NPZ"
    )
    if expected_metrics_sha256 is not None and metrics_sha != _require_sha(
        expected_metrics_sha256, "receipt-bound velocity metrics digest"
    ):
        raise NativeProfileError(
            "velocity metrics SHA-256 differs from its retained receipt"
        )
    if expected_npz_sha256 is not None and source_sha != _require_sha(
        expected_npz_sha256, "receipt-bound velocity NPZ digest"
    ):
        raise NativeProfileError("velocity NPZ SHA-256 differs from its retained receipt")
    if (
        metrics.get("schema_id") != "hilift_native_velocity_profile_metrics_v2"
        or metrics.get("case_id") != case_id
        or metrics.get("metric_value_basis") != "velocity divided by |U_inf|"
    ):
        raise NativeProfileError("velocity metrics identity differs")
    _validate_identity(
        metrics.get("profile_npz"),
        path=npz_path,
        digest=source_sha,
        label="velocity native NPZ",
        filename_key="filename",
    )

    row_count = len(STATIONS) * ROWS_PER_STATION
    xyz = _require_array(arrays, "requested_xyz_in", dtype=np.float64, shape=(row_count, 3))
    valid = _require_array(arrays, "valid_mask", dtype=np.bool_, shape=(row_count,))
    station_names = _require_array(arrays, "station_names", ndim=1, shape=(len(STATIONS),))
    if station_names.dtype.kind != "U" or station_names.tolist() != list(STATIONS):
        raise NativeProfileError("velocity station names/order must be exact B.2/B.3/C.1/C.2/C.3")
    offsets = _require_array(
        arrays, "station_row_offsets", dtype=np.int64, shape=(len(STATIONS) + 1,)
    )
    if offsets.tolist() != [index * ROWS_PER_STATION for index in range(len(STATIONS) + 1)]:
        raise NativeProfileError("velocity station offsets must preserve five 801-row stations")
    weights = _require_array(
        arrays, "line_length_weights_in", dtype=np.float64, shape=(row_count,)
    )
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise NativeProfileError("velocity coordinates/weights must be finite and non-negative")
    if np.any(weights[~valid] != 0.0):
        raise NativeProfileError("velocity invalid rows must retain zero line-length weight")
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if not np.any(valid[start:end]):
            raise NativeProfileError("every velocity station must contain valid rows")

    predicted = _require_array(
        arrays, "predicted_velocity_nd", dtype=np.float64, shape=(row_count, 3)
    )
    magnitude = _require_array(
        arrays, "predicted_velocity_magnitude_nd", dtype=np.float64, shape=(row_count,)
    )
    _require_nan_gap(predicted, valid, "predicted_velocity_nd")
    _require_nan_gap(magnitude, valid, "predicted_velocity_magnitude_nd")
    if not np.allclose(
        magnitude[valid],
        np.linalg.norm(predicted[valid], axis=1),
        rtol=2e-13,
        atol=2e-13,
    ):
        raise NativeProfileError("velocity magnitude is inconsistent with vector prediction")
    for index, axis in enumerate(("x", "y", "z")):
        alias = _require_array(
            arrays, f"u_{axis}_over_U_inf_pred", dtype=np.float64, shape=(row_count,)
        )
        _require_nan_gap(alias, valid, f"u_{axis}_over_U_inf_pred")
        if not np.array_equal(alias[valid], predicted[valid, index]):
            raise NativeProfileError(f"velocity {axis} alias differs from vector prediction")
    speed_alias = _require_array(
        arrays, "speed_over_U_inf_pred", dtype=np.float64, shape=(row_count,)
    )
    _require_nan_gap(speed_alias, valid, "speed_over_U_inf_pred")
    if not np.array_equal(speed_alias[valid], magnitude[valid]):
        raise NativeProfileError("velocity speed alias differs from magnitude prediction")

    for name, shape in (
        ("reference_velocity_nd", (row_count, 3)),
        ("predicted_velocity_physical", (row_count, 3)),
        ("reference_velocity_physical", (row_count, 3)),
        ("predicted_velocity_dimensional", (row_count, 3)),
        ("reference_velocity_dimensional", (row_count, 3)),
        ("reference_velocity_magnitude_nd", (row_count,)),
        ("u_x_over_U_inf_reference", (row_count,)),
        ("u_y_over_U_inf_reference", (row_count,)),
        ("u_z_over_U_inf_reference", (row_count,)),
        ("speed_over_U_inf_reference", (row_count,)),
    ):
        value = _require_array(arrays, name, dtype=np.float64, shape=shape)
        _require_nan_gap(value, valid, name)

    raw_ids = _require_array(arrays, "support_raw_point_ids", dtype=np.int64, ndim=1)
    compact_ids = _require_array(
        arrays, "support_compact_point_ids", dtype=np.int64, shape=(len(raw_ids),)
    )
    support_prediction = _require_array(
        arrays,
        "predicted_velocity_support_nd",
        dtype=np.float32,
        shape=(len(raw_ids), 3),
    )
    if (
        len(raw_ids) < 1
        or len(set(raw_ids.tolist())) != len(raw_ids)
        or len(set(compact_ids.tolist())) != len(compact_ids)
        or np.any(raw_ids < 0)
        or np.any(compact_ids < 0)
        or not np.all(np.isfinite(support_prediction))
    ):
        raise NativeProfileError("velocity native support mapping is invalid")
    _require_array(
        arrays, "reference_velocity_support_nd", dtype=np.float32, shape=(len(raw_ids), 3)
    )
    _require_array(
        arrays, "predicted_velocity_support_physical", dtype=np.float32, shape=(len(raw_ids), 3)
    )
    _require_array(
        arrays, "reference_velocity_support_physical", dtype=np.float32, shape=(len(raw_ids), 3)
    )
    _require_array(arrays, "u_inf_vector_physical", dtype=np.float32, shape=(3,))
    if arrays["velocity_nondimensionalization"].shape != () or str(
        arrays["velocity_nondimensionalization"].item()
    ) != "U/|U_inf|":
        raise NativeProfileError("velocity nondimensionalization differs")
    for scalar_name, dtype in (("context_ceiling", np.int64), ("partition_seed", np.int64)):
        _require_array(arrays, scalar_name, dtype=dtype, shape=())
    if arrays["run_fingerprint"].shape != () or str(arrays["run_fingerprint"].item()) != metrics.get("run_fingerprint"):
        raise NativeProfileError("velocity run fingerprint differs")

    stations = metrics.get("stations")
    if not isinstance(stations, Mapping) or list(stations) != list(STATIONS):
        raise NativeProfileError("velocity metrics station inventory/order differs")
    pooled = metrics.get("pooled", {}).get("velocity_magnitude")
    if not isinstance(pooled, Mapping) or pooled.get("count") != int(np.count_nonzero(valid)):
        raise NativeProfileError("velocity pooled profile count differs")
    velocity_r2 = _finite_number(pooled.get("r2"), "velocity pooled R2")
    metadata = {
        "source_npz_sha256": source_sha,
        "source_metrics_sha256": metrics_sha,
        "source_metrics_schema": metrics["schema_id"],
        "run_fingerprint": _require_sha(
            metrics.get("run_fingerprint"), "velocity run fingerprint"
        ),
        "capture_plan_fingerprint": _require_sha(
            metrics.get("capture_plan_fingerprint"), "velocity capture-plan fingerprint"
        ),
        "stencil_record_sha256": _require_sha(
            metrics.get("stencil_record_sha256"), "velocity stencil record digest"
        ),
        "ref_values_csv_sha256": _require_sha(
            metrics.get("ref_values_csv_sha256"), "velocity reference-values digest"
        ),
        "station_order": list(STATIONS),
        "station_row_offsets": offsets.tolist(),
        "row_count": row_count,
        "valid_row_count": int(np.count_nonzero(valid)),
        "invalid_row_count": int(np.count_nonzero(~valid)),
        "native_support_point_count": len(raw_ids),
        "velocity_profile_r2": velocity_r2,
    }
    return arrays, metadata


def write_deterministic_npz(
    path: Path, arrays: Mapping[str, np.ndarray], *, order: Sequence[str]
) -> str:
    if set(arrays) != set(order):
        raise NativeProfileError("deterministic NPZ array inventory differs from contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as raw:
            with zipfile.ZipFile(
                raw,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                strict_timestamps=True,
            ) as archive:
                for name in order:
                    buffer = io.BytesIO()
                    np.lib.format.write_array(
                        buffer,
                        np.asarray(arrays[name]),
                        allow_pickle=False,
                    )
                    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            raw.flush()
            os.fsync(raw.fileno())
    except FileExistsError as error:
        raise NativeProfileError(f"refusing to overwrite profile artifact: {path}") from error
    return sha256_file(path)


def _artifact_descriptor(path: Path, *, profiles_root: Path, source_sha256: str) -> dict[str, Any]:
    relative = path.relative_to(profiles_root).as_posix()
    return {
        "format": "numpy-npz-v1",
        "file": relative,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "source_native_npz_sha256": source_sha256,
        "ground_truth_included": False,
    }


def serialize_case_profiles(
    *,
    case_id: str,
    case_output_root: Path | None,
    profiles_root: Path,
    expected_artifact_sha256: Mapping[str, str] | None = None,
    surface_case_output_root: Path | None = None,
    volume_case_output_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if SAFE_CASE_ID.fullmatch(case_id) is None:
        raise NativeProfileError(f"invalid HiLiftAeroML case ID {case_id!r}")
    surface_case_root = surface_case_output_root or case_output_root
    volume_case_root = volume_case_output_root or case_output_root
    if surface_case_root is None or volume_case_root is None:
        raise NativeProfileError(
            "surface and volume case-output roots are both required"
        )
    surface = surface_case_root / "surface_submission_stream"
    volume = volume_case_root / "volume_submission_stream"
    expected = expected_artifact_sha256 or {}
    if expected_artifact_sha256 is not None and set(expected) != {
        "cp_profile_metrics",
        "cp_cut_values",
        "velocity_profile_metrics",
        "velocity_profiles",
    }:
        raise NativeProfileError("receipt-bound profile artifact inventory differs")
    cp_arrays, cp = validate_cp_source(
        metrics_path=surface / "cp_profile_metrics.json",
        npz_path=surface / "cp_cut_values.npz",
        expected_metrics_sha256=expected.get("cp_profile_metrics"),
        expected_npz_sha256=expected.get("cp_cut_values"),
    )
    velocity_arrays, velocity = validate_velocity_source(
        case_id=case_id,
        metrics_path=volume / "velocity_profile_metrics.json",
        npz_path=volume / "velocity_profiles.npz",
        expected_metrics_sha256=expected.get("velocity_profile_metrics"),
        expected_npz_sha256=expected.get("velocity_profiles"),
    )
    case_artifacts = profiles_root / "artifacts" / case_id
    cp_path = case_artifacts / "surface-cp-predictions.npz"
    velocity_path = case_artifacts / "volume-velocity-predictions.npz"
    write_deterministic_npz(
        cp_path,
        {name: cp_arrays[name] for name in CP_PUBLISHED_ARRAYS},
        order=CP_PUBLISHED_ARRAYS,
    )
    write_deterministic_npz(
        velocity_path,
        {name: velocity_arrays[name] for name in VELOCITY_PUBLISHED_ARRAYS},
        order=VELOCITY_PUBLISHED_ARRAYS,
    )
    entry = {
        "case_id": case_id,
        "surface_cp": {
            "artifact": _artifact_descriptor(
                cp_path,
                profiles_root=profiles_root,
                source_sha256=cp["source_npz_sha256"],
            ),
            **{key: value for key, value in cp.items() if key != "cp_cut_r2"},
            "station_rows": list(ROWS),
            "topology": "disconnected_graphs_with_branch_and_segment_offsets",
        },
        "volume_velocity": {
            "artifact": _artifact_descriptor(
                velocity_path,
                profiles_root=profiles_root,
                source_sha256=velocity["source_npz_sha256"],
            ),
            **{
                key: value
                for key, value in velocity.items()
                if key != "velocity_profile_r2"
            },
            "validity": "explicit_valid_mask_with_unfilled_invalid_rows",
        },
    }
    return entry, {
        "cp_cut_r2": cp["cp_cut_r2"],
        "velocity_profile_r2": velocity["velocity_profile_r2"],
    }


def build_profile_directory(
    *,
    submission_id: str,
    split_id: str,
    case_set_id: str,
    case_ids: Sequence[str],
    outputs_root: Path | None,
    profiles_root: Path,
    cases_per_chunk: int = 10,
    expected_case_artifact_sha256: Mapping[str, Mapping[str, str]] | None = None,
    surface_outputs_root: Path | None = None,
    volume_outputs_root: Path | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    if profiles_root.exists() or profiles_root.is_symlink():
        raise NativeProfileError(f"profile output already exists: {profiles_root}")
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise NativeProfileError("profile case IDs must be non-empty and unique")
    if not isinstance(cases_per_chunk, int) or isinstance(cases_per_chunk, bool) or cases_per_chunk < 1:
        raise NativeProfileError("cases_per_chunk must be a positive integer")
    if sha256_file(CONTRACT_PATH) != PROFILE_CONTRACT_SHA256:
        raise NativeProfileError("native profile contract SHA-256 changed")
    if expected_case_artifact_sha256 is not None and set(
        expected_case_artifact_sha256
    ) != set(case_ids):
        raise NativeProfileError("receipt-bound profile case inventory differs")
    surface_root = surface_outputs_root or outputs_root
    volume_root = volume_outputs_root or outputs_root
    if surface_root is None or volume_root is None:
        raise NativeProfileError(
            "surface and volume native-output roots are both required"
        )
    profiles_root.mkdir(parents=True)
    chunks: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    for chunk_number, start in enumerate(range(0, len(case_ids), cases_per_chunk)):
        selected = list(case_ids[start : start + cases_per_chunk])
        entries = []
        for case_id in selected:
            entry, case_metrics = serialize_case_profiles(
                case_id=case_id,
                case_output_root=None,
                surface_case_output_root=surface_root / case_id,
                volume_case_output_root=volume_root / case_id,
                profiles_root=profiles_root,
                expected_artifact_sha256=(
                    expected_case_artifact_sha256[case_id]
                    if expected_case_artifact_sha256 is not None
                    else None
                ),
            )
            entries.append(entry)
            metrics[case_id] = case_metrics
        chunk = {
            "schema": PROFILE_CHUNK_SCHEMA,
            "schema_version": "1.0",
            "format": PROFILE_FORMAT,
            "contract_id": PROFILE_CONTRACT_ID,
            "contract_sha256": PROFILE_CONTRACT_SHA256,
            "submission_id": submission_id,
            "dataset_id": "hiliftaeroml",
            "split_id": split_id,
            "case_set_id": case_set_id,
            "cases": entries,
        }
        filename = f"chunk-{chunk_number:03d}.json"
        digest = write_json(profiles_root / filename, chunk)
        chunks.append({"file": filename, "case_ids": selected, "sha256": digest})
    index = {
        "schema_version": "1.0",
        "format": PROFILE_FORMAT,
        "contract_id": PROFILE_CONTRACT_ID,
        "contract_sha256": PROFILE_CONTRACT_SHA256,
        "submission_id": submission_id,
        "dataset_id": "hiliftaeroml",
        "split_id": split_id,
        "case_set_id": case_set_id,
        "case_count": len(case_ids),
        "case_id_status": "official",
        "chunks": chunks,
    }
    index_sha = write_json(profiles_root / "index.json", index)
    return index_sha, metrics


def validate_prediction_npz(
    path: Path,
    *,
    expected_arrays: Sequence[str],
    expected_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    arrays, digest = _load_npz_exact(
        path, expected_arrays=expected_arrays, label="published prediction NPZ"
    )
    if digest != expected_sha256:
        raise NativeProfileError("published prediction NPZ SHA-256 differs")
    forbidden = {"truth_cp"} | {
        name for name in arrays if name.startswith("reference_") or name.endswith("_reference")
    }
    if forbidden.intersection(arrays):
        raise NativeProfileError("published prediction NPZ contains ground truth")
    if metadata is None:
        return arrays
    if tuple(expected_arrays) == CP_PUBLISHED_ARRAYS:
        _validate_published_cp_arrays(arrays, metadata)
    elif tuple(expected_arrays) == VELOCITY_PUBLISHED_ARRAYS:
        _validate_published_velocity_arrays(arrays, metadata)
    else:
        raise NativeProfileError("published prediction NPZ contract is unknown")
    return arrays


def _metadata_count(metadata: Mapping[str, Any], name: str) -> int:
    value = metadata.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NativeProfileError(f"published profile metadata {name} is invalid")
    return value


def _validate_published_cp_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> None:
    vertex_count = _metadata_count(metadata, "vertex_count")
    segment_count = _metadata_count(metadata, "segment_count")
    branch_count = _metadata_count(metadata, "branch_count")
    graph_count = _metadata_count(metadata, "graph_count")
    if min(vertex_count, segment_count, branch_count, graph_count) < 1:
        raise NativeProfileError("published Cp topology counts must be positive")
    branch_closed = _require_array(
        arrays, "branch_closed", dtype=np.bool_, shape=(branch_count,)
    )
    del branch_closed
    for name, dtype in (
        ("branch_component_code", np.uint8),
        ("branch_plane_piece_code", np.uint8),
        ("branch_row_code", np.uint8),
        ("branch_side_code", np.uint8),
        ("branch_topology_patch_code", np.uint8),
    ):
        _require_array(arrays, name, dtype=dtype, shape=(branch_count,))
    graph_codes = _require_array(
        arrays,
        "branch_graph_component_code",
        dtype=np.int64,
        shape=(branch_count,),
    )
    if np.any(graph_codes < 0):
        raise NativeProfileError("published Cp graph component codes must be non-negative")
    branch_vertex_ids = _require_array(
        arrays, "branch_vertex_ids", dtype=np.int64, ndim=1
    )
    branch_vertex_offsets = _require_array(
        arrays,
        "branch_vertex_offsets",
        dtype=np.int64,
        shape=(branch_count + 1,),
    )
    branch_segment_offsets = _require_array(
        arrays,
        "branch_segment_offsets",
        dtype=np.int64,
        shape=(branch_count + 1,),
    )
    _validate_offsets(
        branch_vertex_offsets,
        end=len(branch_vertex_ids),
        label="published branch_vertex_offsets",
    )
    _validate_offsets(
        branch_segment_offsets,
        end=segment_count,
        label="published branch_segment_offsets",
    )
    xyz = _require_array(
        arrays, "cut_xyz_in", dtype=np.float64, shape=(vertex_count, 3)
    )
    prediction = _require_array(
        arrays, "prediction_cp", dtype=np.float64, shape=(vertex_count,)
    )
    lengths = _require_array(
        arrays, "segment_lengths_in", dtype=np.float64, shape=(segment_count,)
    )
    segment_vertex_ids = _require_array(
        arrays,
        "segment_vertex_ids",
        dtype=np.int64,
        shape=(segment_count, 2),
    )
    _require_array(
        arrays,
        "segment_plane_piece_code",
        dtype=np.uint8,
        shape=(segment_count,),
    )
    if not all(np.all(np.isfinite(value)) for value in (xyz, prediction, lengths)):
        raise NativeProfileError("published Cp numeric arrays must be finite")
    if np.any(lengths <= 0.0):
        raise NativeProfileError("published Cp segment lengths must be positive")
    for value, label in (
        (branch_vertex_ids, "branch_vertex_ids"),
        (segment_vertex_ids, "segment_vertex_ids"),
    ):
        if np.any(value < 0) or np.any(value >= vertex_count):
            raise NativeProfileError(f"published {label} contains an out-of-range ID")
    catalogs = metadata.get("code_catalogs")
    if not isinstance(catalogs, Mapping) or set(catalogs) != {
        "components",
        "plane_pieces",
        "rows",
        "sides",
        "topology_patches",
    }:
        raise NativeProfileError("published Cp code catalogs differ")
    if catalogs.get("rows") != list(ROWS):
        raise NativeProfileError("published Cp row catalog must be exact A-J")
    limits = {
        "branch_component_code": len(catalogs["components"]),
        "branch_plane_piece_code": len(catalogs["plane_pieces"]),
        "branch_row_code": len(catalogs["rows"]),
        "branch_side_code": len(catalogs["sides"]),
        "branch_topology_patch_code": len(catalogs["topology_patches"]),
        "segment_plane_piece_code": len(catalogs["plane_pieces"]),
    }
    if any(limit < 1 for limit in limits.values()):
        raise NativeProfileError("published Cp code catalog is empty")
    for name, limit in limits.items():
        if np.any(arrays[name] >= limit):
            raise NativeProfileError(f"published {name} exceeds its code catalog")
    observed_graph_count = len(
        set(
            zip(
                arrays["branch_row_code"].tolist(),
                graph_codes.tolist(),
                strict=True,
            )
        )
    )
    if observed_graph_count != graph_count:
        raise NativeProfileError("published Cp graph count differs from metadata")


def _validate_published_velocity_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> None:
    row_count = _metadata_count(metadata, "row_count")
    valid_count = _metadata_count(metadata, "valid_row_count")
    invalid_count = _metadata_count(metadata, "invalid_row_count")
    support_count = _metadata_count(metadata, "native_support_point_count")
    expected_row_count = len(STATIONS) * ROWS_PER_STATION
    if (
        row_count != expected_row_count
        or valid_count < 1
        or valid_count + invalid_count != row_count
        or support_count < 1
    ):
        raise NativeProfileError("published velocity row/support counts differ")
    xyz = _require_array(
        arrays, "requested_xyz_in", dtype=np.float64, shape=(row_count, 3)
    )
    valid = _require_array(arrays, "valid_mask", dtype=np.bool_, shape=(row_count,))
    if int(np.count_nonzero(valid)) != valid_count:
        raise NativeProfileError("published velocity valid-row count differs")
    station_names = _require_array(
        arrays, "station_names", ndim=1, shape=(len(STATIONS),)
    )
    if station_names.dtype.kind != "U" or station_names.tolist() != list(STATIONS):
        raise NativeProfileError("published velocity station order differs")
    offsets = _require_array(
        arrays,
        "station_row_offsets",
        dtype=np.int64,
        shape=(len(STATIONS) + 1,),
    )
    expected_offsets = [
        index * ROWS_PER_STATION for index in range(len(STATIONS) + 1)
    ]
    if offsets.tolist() != expected_offsets or metadata.get(
        "station_row_offsets"
    ) != expected_offsets or metadata.get("station_order") != list(STATIONS):
        raise NativeProfileError("published velocity station offsets differ")
    weights = _require_array(
        arrays, "line_length_weights_in", dtype=np.float64, shape=(row_count,)
    )
    if (
        not np.all(np.isfinite(xyz))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(weights[~valid] != 0.0)
    ):
        raise NativeProfileError("published velocity coordinates/weights differ")
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if not np.any(valid[start:end]):
            raise NativeProfileError("published velocity station has no valid rows")
    predicted = _require_array(
        arrays, "predicted_velocity_nd", dtype=np.float64, shape=(row_count, 3)
    )
    magnitude = _require_array(
        arrays,
        "predicted_velocity_magnitude_nd",
        dtype=np.float64,
        shape=(row_count,),
    )
    _require_nan_gap(predicted, valid, "published predicted_velocity_nd")
    _require_nan_gap(
        magnitude, valid, "published predicted_velocity_magnitude_nd"
    )
    if not np.allclose(
        magnitude[valid],
        np.linalg.norm(predicted[valid], axis=1),
        rtol=2e-13,
        atol=2e-13,
    ):
        raise NativeProfileError("published velocity magnitude differs from vector")
    raw_ids = _require_array(
        arrays, "support_raw_point_ids", dtype=np.int64, shape=(support_count,)
    )
    compact_ids = _require_array(
        arrays,
        "support_compact_point_ids",
        dtype=np.int64,
        shape=(support_count,),
    )
    support_prediction = _require_array(
        arrays,
        "predicted_velocity_support_nd",
        dtype=np.float32,
        shape=(support_count, 3),
    )
    if (
        len(set(raw_ids.tolist())) != support_count
        or len(set(compact_ids.tolist())) != support_count
        or np.any(raw_ids < 0)
        or np.any(compact_ids < 0)
        or not np.all(np.isfinite(support_prediction))
    ):
        raise NativeProfileError("published velocity support mapping differs")
    u_inf = _require_array(
        arrays, "u_inf_vector_physical", dtype=np.float32, shape=(3,)
    )
    if not np.all(np.isfinite(u_inf)):
        raise NativeProfileError("published velocity freestream vector is non-finite")
    if arrays["velocity_nondimensionalization"].shape != () or str(
        arrays["velocity_nondimensionalization"].item()
    ) != "U/|U_inf|":
        raise NativeProfileError("published velocity nondimensionalization differs")
    for scalar_name in ("context_ceiling", "partition_seed"):
        _require_array(arrays, scalar_name, dtype=np.int64, shape=())
    if arrays["run_fingerprint"].shape != () or str(
        arrays["run_fingerprint"].item()
    ) != metadata.get("run_fingerprint"):
        raise NativeProfileError("published velocity run fingerprint differs")


__all__ = [
    "CP_PUBLISHED_ARRAYS",
    "NativeProfileError",
    "PROFILE_CHUNK_SCHEMA",
    "PROFILE_CONTRACT_ID",
    "PROFILE_CONTRACT_SHA256",
    "PROFILE_FORMAT",
    "VELOCITY_PUBLISHED_ARRAYS",
    "build_profile_directory",
    "serialize_case_profiles",
    "validate_cp_source",
    "validate_prediction_npz",
    "validate_velocity_source",
    "write_deterministic_npz",
]
