"""Strict loader for evaluator-owned AhmedML case support."""

from __future__ import annotations

import contextlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np

from reference.drivaerml.retained_file import RetainedFileError, RetainedVerifiedFile

from .contract import (
    PROFILE_DEFINITION_SHA256,
    REGION_DEFINITION_SHA256,
    SOURCE_IDENTITY_SHA256,
    AhmedMLSourceCase,
    AhmedMLSourceIdentity,
    read_json,
    sha256_file,
)


CASE_SUPPORT_SCHEMA = "ahmedml-native-case-support-v1"
CASE_SUPPORT_STATUS = "candidate_evaluator_support_not_official"
PROFILE_SAMPLE_COUNT = 128
SURFACE_STATIONS = (
    "upper_body_centerline",
    "underbody_centerline",
    "rear_slant_centerline",
)
VOLUME_STATIONS = (
    "wake_vertical_x_0p25_l",
    "wake_vertical_x_0p50_l",
    "wake_vertical_x_1p00_l",
    "wake_lateral_x_0p50_l_z_0p50_h",
)
SURFACE_COORDINATE_INTERVALS = ((0.0, 1.0),) * len(SURFACE_STATIONS)
VOLUME_COORDINATE_INTERVALS = (
    (0.0, 2.0),
    (0.0, 2.0),
    (0.0, 2.0),
    (-1.0, 1.0),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_ARRAYS: Mapping[str, tuple[str, tuple[int, ...]]] = MappingProxyType(
    {
        "surface_raw_cell_id": ("<i8", (3, 128)),
        "surface_coordinate": ("<f8", (3, 128)),
        "surface_target_xyz_m": ("<f8", (3, 128, 3)),
        "surface_projected_xyz_m": ("<f8", (3, 128, 3)),
        "surface_distance_m": ("<f8", (3, 128)),
        "surface_truth_cp": ("<f4", (3, 128)),
        "volume_raw_cell_id": ("<i8", (4, 128)),
        "volume_coordinate": ("<f8", (4, 128)),
        "volume_target_xyz_m": ("<f8", (4, 128, 3)),
        "volume_projected_xyz_m": ("<f8", (4, 128, 3)),
        "volume_distance_m": ("<f8", (4, 128)),
        "volume_truth_ux_over_uinf": ("<f4", (4, 128)),
    }
)


class AhmedMLSupportError(ValueError):
    """Raised when generated support is incomplete or has changed."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AhmedMLSupportError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AhmedMLSupportError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AhmedMLSupportError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise AhmedMLSupportError(f"{label} must be a lowercase SHA-256")
    return result


def _relative_file(value: object, label: str) -> PurePosixPath:
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {"", ".", ".."}
        or path.as_posix() != text
    ):
        raise AhmedMLSupportError(f"{label} must be one normalized basename")
    return path


@dataclass(frozen=True)
class SupportArtifact:
    """Identity and array contract for one generated support file."""

    role: str
    path: Path
    sha256: str
    size_bytes: int
    format: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    arrays: Mapping[str, tuple[str, tuple[int, ...]]] | None = None


@dataclass(frozen=True)
class AhmedMLCaseSupport:
    """Validated metadata for one exact native case support directory."""

    manifest_path: Path
    manifest_sha256: str
    case_id: str
    run_id: int
    surface_entity_count: int
    volume_entity_count: int
    geometry: Mapping[str, float]
    artifacts: Mapping[str, SupportArtifact]
    audits: Mapping[str, Any]

    def artifact(self, role: str) -> SupportArtifact:
        try:
            return self.artifacts[role]
        except KeyError as error:
            raise AhmedMLSupportError(f"case support has no artifact {role!r}") from error


def _validate_source_binding(value: object, case: AhmedMLSourceCase) -> None:
    source = _mapping(value, "source")
    if source.get("source_identity_sha256") != SOURCE_IDENTITY_SHA256:
        raise AhmedMLSupportError("case support source identity SHA-256 differs")
    bindings = {
        "boundary": case.boundary,
        "surface_cell_area": case.surface_cell_area,
        "volume": case.volume,
        "geometry_parameters": case.geometry_parameters,
        "force_coefficients": case.force_coefficients,
    }
    if set(source) != {"source_identity_sha256", *bindings}:
        raise AhmedMLSupportError("case support source binding keys differ")
    for name, expected in bindings.items():
        actual = _mapping(source[name], f"source.{name}")
        if (
            actual.get("path") != expected.path.as_posix()
            or actual.get("sha256") != expected.sha256
            or actual.get("size_bytes") != expected.size_bytes
        ):
            raise AhmedMLSupportError(f"case support source.{name} differs")


def _artifact(
    role: str,
    value: object,
    root: Path,
) -> SupportArtifact:
    item = _mapping(value, f"artifacts.{role}")
    relative = _relative_file(item.get("path"), f"artifacts.{role}.path")
    path = root / relative.name
    try:
        stat = path.lstat()
    except OSError as error:
        raise AhmedMLSupportError(f"missing support artifact {path}: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise AhmedMLSupportError(f"support artifact must be a regular non-symlink: {path}")
    size = _integer(item.get("size_bytes"), f"artifacts.{role}.size_bytes", minimum=1)
    if stat.st_size != size:
        raise AhmedMLSupportError(f"support artifact {role!r} size differs")
    digest = _digest(item.get("sha256"), f"artifacts.{role}.sha256")
    data_format = _string(item.get("format"), f"artifacts.{role}.format")
    if data_format == "npy":
        dtype = _string(item.get("dtype"), f"artifacts.{role}.dtype")
        raw_shape = item.get("shape")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise AhmedMLSupportError(f"artifacts.{role}.shape must be an array")
        shape = tuple(
            _integer(part, f"artifacts.{role}.shape", minimum=1)
            for part in raw_shape
        )
        arrays = None
    elif data_format == "npz":
        dtype = None
        shape = None
        raw_arrays = _mapping(item.get("arrays"), f"artifacts.{role}.arrays")
        arrays_dict: dict[str, tuple[str, tuple[int, ...]]] = {}
        for name, raw in raw_arrays.items():
            declaration = _mapping(raw, f"artifacts.{role}.arrays.{name}")
            raw_shape = declaration.get("shape")
            if not isinstance(raw_shape, list) or not raw_shape:
                raise AhmedMLSupportError(
                    f"artifacts.{role}.arrays.{name}.shape must be an array"
                )
            arrays_dict[str(name)] = (
                _string(
                    declaration.get("dtype"),
                    f"artifacts.{role}.arrays.{name}.dtype",
                ),
                tuple(
                    _integer(part, f"artifacts.{role}.arrays.{name}.shape", minimum=1)
                    for part in raw_shape
                ),
            )
        arrays = MappingProxyType(arrays_dict)
    else:
        raise AhmedMLSupportError(f"unsupported artifact format {data_format!r}")
    return SupportArtifact(
        role=role,
        path=path,
        sha256=digest,
        size_bytes=size,
        format=data_format,
        dtype=dtype,
        shape=shape,
        arrays=arrays,
    )


def load_case_support(
    path: str | Path,
    *,
    source_identity: AhmedMLSourceIdentity,
) -> AhmedMLCaseSupport:
    """Validate one generated case-support manifest and its file declarations."""

    manifest_path = Path(path).expanduser().resolve()
    manifest_sha = sha256_file(manifest_path)
    document = read_json(manifest_path, label="AhmedML case support")
    if (
        document.get("schema") != CASE_SUPPORT_SCHEMA
        or document.get("schema_version") != 1
        or document.get("status") != CASE_SUPPORT_STATUS
        or document.get("official_submission_support") is not False
        or document.get("dataset_revision") != source_identity.repository_revision
    ):
        raise AhmedMLSupportError("case-support header is inconsistent")
    case_id = _string(document.get("case_id"), "case_id")
    case = source_identity.case(case_id)
    if document.get("run_id") != case.run_id:
        raise AhmedMLSupportError("case-support run_id differs from source identity")
    _validate_source_binding(document.get("source"), case)
    definitions = _mapping(document.get("definitions"), "definitions")
    if (
        definitions.get("profile_definition_sha256") != PROFILE_DEFINITION_SHA256
        or definitions.get("regional_definition_sha256") != REGION_DEFINITION_SHA256
        or definitions.get("profile_sample_count") != PROFILE_SAMPLE_COUNT
        or definitions.get("surface_station_ids") != list(SURFACE_STATIONS)
        or definitions.get("volume_station_ids") != list(VOLUME_STATIONS)
    ):
        raise AhmedMLSupportError("case-support definitions differ from the evaluator")
    counts = _mapping(document.get("entity_counts"), "entity_counts")
    if (
        counts.get("surface") != case.surface_entity_count
        or counts.get("volume") != case.volume_entity_count
    ):
        raise AhmedMLSupportError("case-support entity counts differ from source identity")
    raw_geometry = _mapping(
        document.get("geometry_parameters_mm_or_degrees"), "geometry parameters"
    )
    geometry: dict[str, float] = {}
    for name, value in raw_geometry.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise AhmedMLSupportError("geometry parameters must be positive finite numbers")
        geometry[name] = float(value)
    raw_artifacts = _mapping(document.get("artifacts"), "artifacts")
    expected_roles = {
        "surface_area_vector",
        "volume_cell_volume",
        "volume_region_code",
        "profile_mapping_and_truth",
    }
    if set(raw_artifacts) != expected_roles:
        raise AhmedMLSupportError("case-support artifact roles differ")
    artifacts = {
        role: _artifact(role, raw_artifacts[role], manifest_path.parent)
        for role in sorted(expected_roles)
    }
    expected_npy = {
        "surface_area_vector": ("<f8", (case.surface_entity_count, 3)),
        "volume_cell_volume": ("<f4", (case.volume_entity_count,)),
        "volume_region_code": ("|u1", (case.volume_entity_count,)),
    }
    for role, (dtype, shape) in expected_npy.items():
        artifact = artifacts[role]
        if artifact.format != "npy" or artifact.dtype != dtype or artifact.shape != shape:
            raise AhmedMLSupportError(f"artifact {role!r} array contract differs")
    profiles = artifacts["profile_mapping_and_truth"]
    if profiles.format != "npz" or dict(profiles.arrays or {}) != dict(_PROFILE_ARRAYS):
        raise AhmedMLSupportError("profile mapping array contract differs")
    return AhmedMLCaseSupport(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        case_id=case_id,
        run_id=case.run_id,
        surface_entity_count=case.surface_entity_count,
        volume_entity_count=case.volume_entity_count,
        geometry=MappingProxyType(geometry),
        artifacts=MappingProxyType(artifacts),
        audits=MappingProxyType(dict(_mapping(document.get("audits"), "audits"))),
    )


def _verify_retained_artifact(
    artifact: SupportArtifact,
    retained: RetainedVerifiedFile,
) -> None:
    if retained.snapshot.size_bytes != artifact.size_bytes:
        raise AhmedMLSupportError(f"support artifact {artifact.role!r} size changed")
    if retained.sha256() != artifact.sha256:
        raise AhmedMLSupportError(f"support artifact {artifact.role!r} SHA-256 changed")


@contextlib.contextmanager
def open_support_array(
    support: AhmedMLCaseSupport,
    role: str,
) -> Iterator[np.ndarray]:
    """Yield one verified memory-mapped NPY support array."""

    artifact = support.artifact(role)
    if artifact.format != "npy" or artifact.dtype is None or artifact.shape is None:
        raise AhmedMLSupportError(f"support artifact {role!r} is not NPY")
    try:
        with RetainedVerifiedFile.open(artifact.path, label=role) as retained:
            _verify_retained_artifact(artifact, retained)
            array = np.load(
                retained.descriptor_path(), mmap_mode="r", allow_pickle=False
            )
            if array.dtype.str != artifact.dtype or array.shape != artifact.shape:
                raise AhmedMLSupportError(f"support artifact {role!r} payload differs")
            array.flags.writeable = False
            yield array
            retained.assert_unchanged(context="while its NPY array was consumed")
            del array
    except AhmedMLSupportError:
        raise
    except (OSError, ValueError, RetainedFileError) as error:
        raise AhmedMLSupportError(f"cannot load support artifact {role!r}: {error}") from error


def load_profile_support(support: AhmedMLCaseSupport) -> Mapping[str, np.ndarray]:
    """Load and validate the compact frozen mapping/truth NPZ."""

    artifact = support.artifact("profile_mapping_and_truth")
    try:
        with RetainedVerifiedFile.open(artifact.path, label=artifact.role) as retained:
            _verify_retained_artifact(artifact, retained)
            retained.handle.seek(0)
            with np.load(retained.handle, allow_pickle=False) as archive:
                if set(archive.files) != set(_PROFILE_ARRAYS):
                    raise AhmedMLSupportError("profile NPZ member names differ")
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in _PROFILE_ARRAYS
                }
            retained.assert_unchanged(context="while its profile NPZ was consumed")
    except AhmedMLSupportError:
        raise
    except (OSError, ValueError, RetainedFileError) as error:
        raise AhmedMLSupportError(f"cannot load profile support: {error}") from error

    for name, (dtype, shape) in _PROFILE_ARRAYS.items():
        array = arrays[name]
        if array.dtype.str != dtype or array.shape != shape:
            raise AhmedMLSupportError(
                f"profile array {name!r} has {array.dtype.str} {array.shape}, "
                f"expected {dtype} {shape}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise AhmedMLSupportError(f"profile array {name!r} contains non-finite values")
        array.flags.writeable = False
    surface_ids = arrays["surface_raw_cell_id"]
    volume_ids = arrays["volume_raw_cell_id"]
    if np.any(surface_ids < 0) or np.any(surface_ids >= support.surface_entity_count):
        raise AhmedMLSupportError("surface profile mapping contains an invalid raw cell ID")
    if np.any(volume_ids < 0) or np.any(volume_ids >= support.volume_entity_count):
        raise AhmedMLSupportError("volume profile mapping contains an invalid raw cell ID")
    if np.any(np.diff(arrays["surface_coordinate"], axis=1) <= 0.0):
        raise AhmedMLSupportError("surface profile coordinates must increase strictly")
    if np.any(np.diff(arrays["volume_coordinate"], axis=1) <= 0.0):
        raise AhmedMLSupportError("volume profile coordinates must increase strictly")
    expected_surface_coordinates = np.vstack(
        [
            np.linspace(start, stop, PROFILE_SAMPLE_COUNT, dtype=np.float64)
            for start, stop in SURFACE_COORDINATE_INTERVALS
        ]
    )
    expected_volume_coordinates = np.vstack(
        [
            np.linspace(start, stop, PROFILE_SAMPLE_COUNT, dtype=np.float64)
            for start, stop in VOLUME_COORDINATE_INTERVALS
        ]
    )
    if not np.array_equal(arrays["surface_coordinate"], expected_surface_coordinates):
        raise AhmedMLSupportError(
            "surface profile coordinates differ from the exact 128-point grids"
        )
    if not np.array_equal(arrays["volume_coordinate"], expected_volume_coordinates):
        raise AhmedMLSupportError(
            "volume profile coordinates differ from the exact 128-point grids"
        )
    for name in ("surface_distance_m", "volume_distance_m"):
        if np.any(arrays[name] < 0.0):
            raise AhmedMLSupportError(f"profile array {name!r} cannot be negative")
    return MappingProxyType(arrays)


__all__ = [
    "AhmedMLCaseSupport",
    "AhmedMLSupportError",
    "CASE_SUPPORT_SCHEMA",
    "CASE_SUPPORT_STATUS",
    "PROFILE_SAMPLE_COUNT",
    "SURFACE_COORDINATE_INTERVALS",
    "SURFACE_STATIONS",
    "SupportArtifact",
    "VOLUME_COORDINATE_INTERVALS",
    "VOLUME_STATIONS",
    "load_case_support",
    "load_profile_support",
    "open_support_array",
]
