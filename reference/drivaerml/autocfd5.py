"""Deterministic AutoCFD5 registry, evidence, and diagnostic primitives.

The candidate DrivAerML contract freezes the diagnostic locations and
reductions, but it does *not* yet freeze the geometric search implementation
used to locate containing volume cells or project pressure taps onto each
morphed surface.  This module deliberately does not guess those algorithms.
It instead:

* verifies the hashed candidate registries;
* validates complete, duplicate-free owner-published assignment evidence;
* samples native ``CellData`` by raw VTK ID with the prescribed zeroth-order
  reconstruction; and
* evaluates the frozen Cp transform and profile/probe reductions.

Invalid velocity samples remain explicit records and invalid gaps are never
bridged.  Invalid Cp mappings also remain explicit, but make the ranked Cp
component unavailable unless the caller is auditing incomplete evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


U_INF_M_PER_S = 38.889
CP_PROBE_COUNT = 209
CP_PANEL_COUNT = 15
CP_PANEL_MEMBERSHIP_COUNT = 217
VELOCITY_LINE_COUNT = 16
VELOCITY_SAMPLE_COUNT = 3756
POINT_IN_CELL_CLOSURE_TOLERANCE_M = 1.0e-6
NATIVE_BRIDGE_MAX_DISTANCE_M = 2.0e-3
NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT = 0.8660254037844387
NOMINAL_PROBE_DISPLACEMENT_MAX_M = 0.278618

_PROFILE_ID = "drivaerml-autocfd5-v8-candidate"
_SUBMISSION_PROFILE_ID = "drivaerml-diagnostics-v9-candidate"
_VELOCITY_PROFILE_IDS = (
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "L1",
    "R1",
    "R2",
    "R3",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class AutoCFD5Error(ValueError):
    """Raised when a registry, assignment, or diagnostic violates the contract."""


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutoCFD5Error(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise AutoCFD5Error(f"{label} must be a {qualifier} integer")
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AutoCFD5Error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AutoCFD5Error(f"{label} must be finite")
    return result


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise AutoCFD5Error(f"{label} must contain exactly three coordinates")
    return tuple(_finite(item, label) for item in value)  # type: ignore[return-value]


def _optional_nonnegative_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _optional_finite(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _optional_point3(
    value: object, label: str
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _point3(value, label)


def _close(actual: float, expected: float, *, tolerance: float = 2.0e-11) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _same_point(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
    *,
    tolerance: float = 2.0e-11,
) -> bool:
    return all(_close(a, b, tolerance=tolerance) for a, b in zip(actual, expected))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(required_columns - fields)
            if missing:
                raise AutoCFD5Error(
                    f"{path.name} is missing required columns: {', '.join(missing)}"
                )
            return list(reader)
    except OSError as error:
        raise AutoCFD5Error(f"cannot read {path}") from error


def _csv_int(row: Mapping[str, str], key: str, path: Path) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise AutoCFD5Error(f"{path.name} column {key!r} must contain integers") from error
    return value


def _csv_float(row: Mapping[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise AutoCFD5Error(f"{path.name} column {key!r} must contain numbers") from error
    if not math.isfinite(value):
        raise AutoCFD5Error(f"{path.name} column {key!r} contains a non-finite value")
    return value


@dataclass(frozen=True)
class CpProbeDefinition:
    """One of the 209 unique nominal AutoCFD pressure taps."""

    autocfd_probe_id: int
    station_id: str
    ansa_pid: int
    station_point_index: int
    point_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        _integer(self.autocfd_probe_id, "autocfd_probe_id", minimum=1)
        _nonempty_string(self.station_id, "station_id")
        _integer(self.ansa_pid, "ansa_pid", minimum=1)
        _integer(self.station_point_index, "station_point_index", minimum=1)
        _point3(self.point_m, "point_m")


@dataclass(frozen=True)
class CpPanelMembership:
    """One ordered display-panel row; repeated probe IDs are intentional."""

    panel_id: str
    panel_point_index: int
    autocfd_probe_id: int
    point_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        _nonempty_string(self.panel_id, "panel_id")
        _integer(self.panel_point_index, "panel_point_index", minimum=1)
        _integer(self.autocfd_probe_id, "autocfd_probe_id", minimum=1)
        _point3(self.point_m, "point_m")


@dataclass(frozen=True)
class CpComponentRule:
    """Candidate anatomical component and projection rule for one tap."""

    autocfd_probe_id: int
    drivaerml_component: str
    projection_mode: str
    cut_axis: str | None
    cut_value_m: float | None
    owner_review_status: str

    def __post_init__(self) -> None:
        _integer(self.autocfd_probe_id, "autocfd_probe_id", minimum=1)
        _nonempty_string(self.drivaerml_component, "drivaerml_component")
        if self.projection_mode not in {"cut_plane_closest", "component_closest_3d"}:
            raise AutoCFD5Error("unsupported Cp projection_mode")
        if self.projection_mode == "cut_plane_closest":
            if self.cut_axis not in {"x", "y", "z"}:
                raise AutoCFD5Error("cut-plane rules require cut_axis x, y, or z")
            _finite(self.cut_value_m, "cut_value_m")
        elif self.cut_axis is not None or self.cut_value_m is not None:
            raise AutoCFD5Error("3-D closest rules cannot carry a cut plane")
        _nonempty_string(self.owner_review_status, "owner_review_status")


@dataclass(frozen=True)
class VelocityLineDefinition:
    """One fixed AutoCFD velocity line and its candidate scoring grid size."""

    profile_id: str
    station_id: str
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    length_m: float
    scoring_spacing_m: float
    sample_count: int
    reference_1mm_sample_count: int
    experimental_availability: str

    def __post_init__(self) -> None:
        _nonempty_string(self.profile_id, "profile_id")
        _nonempty_string(self.station_id, "station_id")
        start = _point3(self.start_m, "start_m")
        end = _point3(self.end_m, "end_m")
        length = _finite(self.length_m, "length_m")
        spacing = _finite(self.scoring_spacing_m, "scoring_spacing_m")
        if length <= 0.0 or spacing <= 0.0:
            raise AutoCFD5Error("line length and spacing must be positive")
        _integer(self.sample_count, "sample_count", minimum=2)
        _integer(self.reference_1mm_sample_count, "reference_1mm_sample_count", minimum=2)
        _nonempty_string(self.experimental_availability, "experimental_availability")
        coordinate_length = math.dist(start, end)
        if not _close(coordinate_length, length, tolerance=2.0e-9):
            raise AutoCFD5Error("line endpoint distance differs from declared length")
        if not _close(spacing, length / (self.sample_count - 1), tolerance=2.0e-11):
            raise AutoCFD5Error("line spacing differs from endpoint-inclusive grid")


@dataclass(frozen=True)
class VelocitySampleDefinition:
    """One explicit point on the fixed candidate 10 mm velocity grid."""

    profile_id: str
    sample_index: int
    point_count: int
    line_fraction: float
    distance_m: float
    point_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        _nonempty_string(self.profile_id, "profile_id")
        _integer(self.sample_index, "sample_index")
        _integer(self.point_count, "point_count", minimum=2)
        fraction = _finite(self.line_fraction, "line_fraction")
        distance = _finite(self.distance_m, "distance_m")
        if not 0.0 <= fraction <= 1.0:
            raise AutoCFD5Error("line_fraction must lie in [0, 1]")
        if distance < 0.0:
            raise AutoCFD5Error("distance_m must be non-negative")
        _point3(self.point_m, "point_m")


@dataclass(frozen=True)
class AutoCFD5Definition:
    """Validated fixed registries bound by ``autocfd5-profiles-v8.json``."""

    cp_probes: tuple[CpProbeDefinition, ...]
    cp_panel_memberships: tuple[CpPanelMembership, ...]
    cp_component_rules: tuple[CpComponentRule, ...]
    velocity_lines: tuple[VelocityLineDefinition, ...]
    velocity_samples: tuple[VelocitySampleDefinition, ...]
    source_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.cp_probes) != CP_PROBE_COUNT:
            raise AutoCFD5Error(f"Cp registry must contain {CP_PROBE_COUNT} probes")
        if len(self.cp_panel_memberships) != CP_PANEL_MEMBERSHIP_COUNT:
            raise AutoCFD5Error(
                f"Cp panel registry must contain {CP_PANEL_MEMBERSHIP_COUNT} rows"
            )
        if len({row.panel_id for row in self.cp_panel_memberships}) != CP_PANEL_COUNT:
            raise AutoCFD5Error(f"Cp panel registry must contain {CP_PANEL_COUNT} panels")
        if len(self.cp_component_rules) != CP_PROBE_COUNT:
            raise AutoCFD5Error("Cp component registry must contain one rule per probe")
        if len(self.velocity_lines) != VELOCITY_LINE_COUNT:
            raise AutoCFD5Error(
                f"velocity registry must contain {VELOCITY_LINE_COUNT} lines"
            )
        if len(self.velocity_samples) != VELOCITY_SAMPLE_COUNT:
            raise AutoCFD5Error(
                f"velocity grid must contain {VELOCITY_SAMPLE_COUNT} samples"
            )


@dataclass(frozen=True)
class AutoCFD5SubmissionDefinition:
    """Cp-probe-free v9 submission registry for velocity lines and true Cp cuts."""

    velocity_lines: tuple[VelocityLineDefinition, ...]
    velocity_samples: tuple[VelocitySampleDefinition, ...]
    pressure_cut_ids: tuple[str, ...]
    source_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.velocity_lines) != VELOCITY_LINE_COUNT:
            raise AutoCFD5Error(
                f"velocity registry must contain {VELOCITY_LINE_COUNT} lines"
            )
        if len(self.velocity_samples) != VELOCITY_SAMPLE_COUNT:
            raise AutoCFD5Error(
                f"velocity grid must contain {VELOCITY_SAMPLE_COUNT} samples"
            )
        if self.pressure_cut_ids != (
            "upperbody_centerline",
            "underbody_centerline",
            "sidewall_z_0_15",
            "front_left_wheelhouse_y_neg_0_6",
        ):
            raise AutoCFD5Error("submission profile must contain the four Cp cuts")


def _load_cp_probes(path: Path) -> tuple[CpProbeDefinition, ...]:
    rows = _csv_rows(
        path,
        {
            "station_id",
            "ansa_pid",
            "station_point_index",
            "autocfd_probe_id",
            "x_m",
            "y_m",
            "z_m",
        },
    )
    result = tuple(
        CpProbeDefinition(
            autocfd_probe_id=_csv_int(row, "autocfd_probe_id", path),
            station_id=row["station_id"],
            ansa_pid=_csv_int(row, "ansa_pid", path),
            station_point_index=_csv_int(row, "station_point_index", path),
            point_m=tuple(_csv_float(row, key, path) for key in ("x_m", "y_m", "z_m")),  # type: ignore[arg-type]
        )
        for row in rows
    )
    probe_ids = [probe.autocfd_probe_id for probe in result]
    if len(probe_ids) != len(set(probe_ids)):
        raise AutoCFD5Error("Cp nominal registry contains duplicate probe IDs")
    station_keys = [(probe.station_id, probe.station_point_index) for probe in result]
    if len(station_keys) != len(set(station_keys)):
        raise AutoCFD5Error("Cp nominal registry contains duplicate station point IDs")
    return result


def _load_cp_panels(
    path: Path, probes: tuple[CpProbeDefinition, ...]
) -> tuple[CpPanelMembership, ...]:
    rows = _csv_rows(
        path,
        {
            "panel_id",
            "panel_point_index",
            "autocfd_probe_id",
            "x_m",
            "y_m",
            "z_m",
        },
    )
    result = tuple(
        CpPanelMembership(
            panel_id=row["panel_id"],
            panel_point_index=_csv_int(row, "panel_point_index", path),
            autocfd_probe_id=_csv_int(row, "autocfd_probe_id", path),
            point_m=tuple(_csv_float(row, key, path) for key in ("x_m", "y_m", "z_m")),  # type: ignore[arg-type]
        )
        for row in rows
    )
    probe_by_id = {probe.autocfd_probe_id: probe for probe in probes}
    seen: set[tuple[str, int]] = set()
    indices_by_panel: dict[str, list[int]] = {}
    for membership in result:
        key = (membership.panel_id, membership.panel_point_index)
        if key in seen:
            raise AutoCFD5Error("Cp panel registry contains a duplicate panel row")
        seen.add(key)
        indices_by_panel.setdefault(membership.panel_id, []).append(
            membership.panel_point_index
        )
        probe = probe_by_id.get(membership.autocfd_probe_id)
        if probe is None:
            raise AutoCFD5Error("Cp panel registry references an unknown probe ID")
        if not _same_point(membership.point_m, probe.point_m):
            raise AutoCFD5Error("Cp panel coordinates differ from the nominal probe")
    for panel_id, indices in indices_by_panel.items():
        if indices != list(range(1, len(indices) + 1)):
            raise AutoCFD5Error(f"Cp panel {panel_id!r} is not consecutively ordered")
    return result


def _load_cp_component_rules(
    path: Path, probes: tuple[CpProbeDefinition, ...]
) -> tuple[CpComponentRule, ...]:
    rows = _csv_rows(
        path,
        {
            "autocfd_probe_id",
            "ansa_pid",
            "station_id",
            "station_point_index",
            "nominal_x_m",
            "nominal_y_m",
            "nominal_z_m",
            "drivaerml_component",
            "projection_mode",
            "cut_axis",
            "cut_value_m",
            "owner_review_status",
        },
    )
    probe_by_id = {probe.autocfd_probe_id: probe for probe in probes}
    result: list[CpComponentRule] = []
    seen: set[int] = set()
    for row in rows:
        probe_id = _csv_int(row, "autocfd_probe_id", path)
        if probe_id in seen:
            raise AutoCFD5Error("Cp component registry contains duplicate probe IDs")
        seen.add(probe_id)
        probe = probe_by_id.get(probe_id)
        if probe is None:
            raise AutoCFD5Error("Cp component registry references an unknown probe ID")
        point = tuple(
            _csv_float(row, key, path)
            for key in ("nominal_x_m", "nominal_y_m", "nominal_z_m")
        )
        if (
            _csv_int(row, "ansa_pid", path) != probe.ansa_pid
            or row["station_id"] != probe.station_id
            or _csv_int(row, "station_point_index", path) != probe.station_point_index
            or not _same_point(point, probe.point_m)  # type: ignore[arg-type]
        ):
            raise AutoCFD5Error("Cp component rule identity differs from nominal registry")
        cut_axis = row["cut_axis"] or None
        cut_value = (
            _csv_float(row, "cut_value_m", path) if row["cut_value_m"] else None
        )
        result.append(
            CpComponentRule(
                autocfd_probe_id=probe_id,
                drivaerml_component=row["drivaerml_component"],
                projection_mode=row["projection_mode"],
                cut_axis=cut_axis,
                cut_value_m=cut_value,
                owner_review_status=row["owner_review_status"],
            )
        )
    if seen != set(probe_by_id):
        raise AutoCFD5Error("Cp component registry does not cover every nominal probe")
    return tuple(result)


def _load_velocity_lines(path: Path) -> tuple[VelocityLineDefinition, ...]:
    rows = _csv_rows(
        path,
        {
            "profile_id",
            "start_x_m",
            "start_y_m",
            "start_z_m",
            "end_x_m",
            "end_y_m",
            "end_z_m",
            "length_m",
            "autocfd_experimental_availability",
            "scoring_spacing_m",
            "scoring_point_count",
            "reference_1mm_point_count",
        },
    )
    result = tuple(
        VelocityLineDefinition(
            profile_id=row["profile_id"],
            station_id=f"autocfd5_{row['profile_id'].lower()}",
            start_m=tuple(
                _csv_float(row, key, path)
                for key in ("start_x_m", "start_y_m", "start_z_m")
            ),  # type: ignore[arg-type]
            end_m=tuple(
                _csv_float(row, key, path)
                for key in ("end_x_m", "end_y_m", "end_z_m")
            ),  # type: ignore[arg-type]
            length_m=_csv_float(row, "length_m", path),
            scoring_spacing_m=_csv_float(row, "scoring_spacing_m", path),
            sample_count=_csv_int(row, "scoring_point_count", path),
            reference_1mm_sample_count=_csv_int(
                row, "reference_1mm_point_count", path
            ),
            experimental_availability=row["autocfd_experimental_availability"],
        )
        for row in rows
    )
    if tuple(line.profile_id for line in result) != _VELOCITY_PROFILE_IDS:
        raise AutoCFD5Error("velocity line IDs or ordering differ from the v8 registry")
    return result


def _load_velocity_samples(
    path: Path, lines: tuple[VelocityLineDefinition, ...]
) -> tuple[VelocitySampleDefinition, ...]:
    rows = _csv_rows(
        path,
        {
            "profile_id",
            "sample_index",
            "point_count",
            "line_fraction",
            "distance_m",
            "x_m",
            "y_m",
            "z_m",
        },
    )
    result = tuple(
        VelocitySampleDefinition(
            profile_id=row["profile_id"],
            sample_index=_csv_int(row, "sample_index", path),
            point_count=_csv_int(row, "point_count", path),
            line_fraction=_csv_float(row, "line_fraction", path),
            distance_m=_csv_float(row, "distance_m", path),
            point_m=tuple(_csv_float(row, key, path) for key in ("x_m", "y_m", "z_m")),  # type: ignore[arg-type]
        )
        for row in rows
    )
    line_by_id = {line.profile_id: line for line in lines}
    by_line: dict[str, list[VelocitySampleDefinition]] = {}
    for sample in result:
        by_line.setdefault(sample.profile_id, []).append(sample)
    if tuple(by_line) != _VELOCITY_PROFILE_IDS:
        raise AutoCFD5Error("velocity sample line IDs or ordering differ from v8")
    for profile_id, samples in by_line.items():
        line = line_by_id[profile_id]
        if len(samples) != line.sample_count:
            raise AutoCFD5Error(f"velocity line {profile_id} has the wrong sample count")
        for expected_index, sample in enumerate(samples):
            expected_fraction = expected_index / (line.sample_count - 1)
            expected_point = tuple(
                start + expected_fraction * (end - start)
                for start, end in zip(line.start_m, line.end_m)
            )
            expected_distance = expected_fraction * math.dist(line.start_m, line.end_m)
            if (
                sample.sample_index != expected_index
                or sample.point_count != line.sample_count
                or not _close(sample.line_fraction, expected_fraction)
                or not _close(sample.distance_m, expected_distance)
                or not _same_point(sample.point_m, expected_point)
            ):
                raise AutoCFD5Error(
                    f"velocity line {profile_id} sample {expected_index} differs from "
                    "the endpoint-inclusive equal-arc grid"
                )
    return result


def _validate_profile_json(
    profile: Mapping[str, Any], definition: AutoCFD5Definition
) -> None:
    pressure = profile.get("pressure_profiles")
    velocity = profile.get("velocity_profiles")
    if not isinstance(pressure, Mapping) or not isinstance(velocity, Mapping):
        raise AutoCFD5Error("profile must define pressure_profiles and velocity_profiles")
    if (
        pressure.get("unique_probe_count") != CP_PROBE_COUNT
        or pressure.get("panel_count") != CP_PANEL_COUNT
        or pressure.get("panel_membership_row_count") != CP_PANEL_MEMBERSHIP_COUNT
        or pressure.get("quantity") != "Cp=2*pMeanTrim/(38.889^2)"
    ):
        raise AutoCFD5Error("pressure profile metadata differs from the frozen contract")
    if (
        velocity.get("line_count") != VELOCITY_LINE_COUNT
        or velocity.get("sample_count_per_case") != VELOCITY_SAMPLE_COUNT
        or velocity.get("quantity") != "magnitude(UMeanTrim)/38.889"
    ):
        raise AutoCFD5Error("velocity profile metadata differs from the frozen contract")

    panel_rows: dict[str, list[CpPanelMembership]] = {}
    for row in definition.cp_panel_memberships:
        panel_rows.setdefault(row.panel_id, []).append(row)
    stations = pressure.get("stations")
    if not isinstance(stations, list) or [item.get("id") for item in stations] != list(
        panel_rows
    ):
        raise AutoCFD5Error("pressure station IDs or ordering differ from panel registry")
    for station in stations:
        rows = panel_rows[station["id"]]
        probes = station.get("probes")
        if station.get("sample_count") != len(rows) or not isinstance(probes, list):
            raise AutoCFD5Error("pressure station sample count differs from panel registry")
        expected = [
            (row.panel_point_index, row.autocfd_probe_id, list(row.point_m))
            for row in rows
        ]
        actual = [
            (item.get("panel_point_index"), item.get("autocfd_probe_id"), item.get("point_m"))
            for item in probes
        ]
        if actual != expected:
            raise AutoCFD5Error("pressure station rows differ from panel registry")

    line_by_id = {line.profile_id: line for line in definition.velocity_lines}
    stations = velocity.get("stations")
    if not isinstance(stations, list) or [item.get("source_profile_id") for item in stations] != list(
        _VELOCITY_PROFILE_IDS
    ):
        raise AutoCFD5Error("velocity station IDs or ordering differ from line registry")
    for station in stations:
        line = line_by_id[station["source_profile_id"]]
        if (
            station.get("id") != line.station_id
            or station.get("start_m") != list(line.start_m)
            or station.get("end_m") != list(line.end_m)
            or station.get("sample_count") != line.sample_count
            or station.get("experimental_availability")
            != line.experimental_availability
        ):
            raise AutoCFD5Error("velocity station metadata differs from line registry")


def load_autocfd5_definition(profile_path: str | Path) -> AutoCFD5Definition:
    """Load and cross-check the profile plus all five hashed source registries."""

    path = Path(profile_path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoCFD5Error(f"cannot read valid JSON profile {path}") from error
    if not isinstance(profile, Mapping) or profile.get("id") != _PROFILE_ID:
        raise AutoCFD5Error(f"profile id must be {_PROFILE_ID!r}")
    source = profile.get("source")
    if not isinstance(source, Mapping):
        raise AutoCFD5Error("profile source bindings are missing")

    expected_bindings = {
        "cp_nominal_registry",
        "cp_panel_membership",
        "cp_component_registry",
        "velocity_lines",
        "velocity_scoring_grid",
    }
    source_paths: dict[str, Path] = {}
    source_hashes: list[tuple[str, str]] = []
    for key in expected_bindings:
        binding = source.get(key)
        if not isinstance(binding, Mapping):
            raise AutoCFD5Error(f"profile source binding {key!r} is missing")
        relative_path = binding.get("file")
        expected_hash = binding.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise AutoCFD5Error(f"profile source binding {key!r} is malformed")
        if _SHA256_RE.fullmatch(expected_hash) is None:
            raise AutoCFD5Error(f"profile source binding {key!r} has an invalid SHA-256")
        source_path = path.parent / relative_path
        actual_hash = _sha256_file(source_path)
        if actual_hash != expected_hash:
            raise AutoCFD5Error(
                f"profile source binding {key!r} SHA-256 mismatch: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        source_paths[key] = source_path
        source_hashes.append((key, actual_hash))

    probes = _load_cp_probes(source_paths["cp_nominal_registry"])
    panels = _load_cp_panels(source_paths["cp_panel_membership"], probes)
    component_rules = _load_cp_component_rules(
        source_paths["cp_component_registry"], probes
    )
    lines = _load_velocity_lines(source_paths["velocity_lines"])
    samples = _load_velocity_samples(source_paths["velocity_scoring_grid"], lines)
    definition = AutoCFD5Definition(
        cp_probes=probes,
        cp_panel_memberships=panels,
        cp_component_rules=component_rules,
        velocity_lines=lines,
        velocity_samples=samples,
        source_sha256=tuple(sorted(source_hashes)),
    )
    _validate_profile_json(profile, definition)
    return definition


def load_autocfd5_submission_definition(
    profile_path: str | Path,
) -> AutoCFD5SubmissionDefinition:
    """Load the v9 submission profile without accepting v8 Cp-probe registries."""

    path = Path(profile_path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoCFD5Error(f"cannot read valid JSON profile {path}") from error
    if not isinstance(profile, Mapping) or profile.get("id") != _SUBMISSION_PROFILE_ID:
        raise AutoCFD5Error(
            f"submission profile id must be {_SUBMISSION_PROFILE_ID!r}"
        )
    if profile.get("status") != (
        "candidate_velocity_and_cp_cut_support_pending_all_case_validation"
    ):
        raise AutoCFD5Error("submission profile status must retain both pending supports")
    source = profile.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "result_template_version",
        "velocity_lines",
        "velocity_scoring_grid",
    }:
        raise AutoCFD5Error(
            "submission profile must bind only the two velocity registries"
        )
    if source.get("result_template_version") != 8:
        raise AutoCFD5Error("submission profile result-template version is not 8")

    source_paths: dict[str, Path] = {}
    source_hashes: list[tuple[str, str]] = []
    for key in ("velocity_lines", "velocity_scoring_grid"):
        binding = source.get(key)
        if not isinstance(binding, Mapping) or set(binding) != {"file", "sha256"}:
            raise AutoCFD5Error(f"submission profile binding {key!r} is malformed")
        relative_path = binding.get("file")
        expected_hash = binding.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise AutoCFD5Error(f"submission profile binding {key!r} is malformed")
        if _SHA256_RE.fullmatch(expected_hash) is None:
            raise AutoCFD5Error(
                f"submission profile binding {key!r} has an invalid SHA-256"
            )
        source_path = path.parent / relative_path
        actual_hash = _sha256_file(source_path)
        if actual_hash != expected_hash:
            raise AutoCFD5Error(
                f"submission profile binding {key!r} SHA-256 mismatch: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        source_paths[key] = source_path
        source_hashes.append((key, actual_hash))

    lines = _load_velocity_lines(source_paths["velocity_lines"])
    samples = _load_velocity_samples(source_paths["velocity_scoring_grid"], lines)
    velocity = profile.get("velocity_profiles")
    if not isinstance(velocity, Mapping) or (
        velocity.get("ranked_metric_id") != "velocity_profile_r2"
        or velocity.get("report_only_metric_id") != "velocity_profile_uinf_rmse"
        or velocity.get("ranked_reduction")
        != "equal_case_equal_line_global_R2_with_normalized_trapezoidal_arc_length_support_per_line"
        or velocity.get("definition_authority") != "AutoCFD5"
        or velocity.get("line_count") != VELOCITY_LINE_COUNT
        or velocity.get("sample_count_per_case") != VELOCITY_SAMPLE_COUNT
        or velocity.get("quantity") != "magnitude(UMeanTrim)/38.889"
    ):
        raise AutoCFD5Error("v9 velocity profile metadata is inconsistent")
    stations = velocity.get("stations")
    if not isinstance(stations, list) or [
        item.get("source_profile_id") for item in stations if isinstance(item, Mapping)
    ] != list(_VELOCITY_PROFILE_IDS):
        raise AutoCFD5Error("v9 velocity station IDs or ordering are inconsistent")

    cuts = profile.get("pressure_cuts")
    if not isinstance(cuts, Mapping) or (
        cuts.get("ranked_metric_id") != "cp_cut_r2"
        or cuts.get("report_only_metric_id") != "cp_cut_rmse"
        or cuts.get("definition_authority") != "FluidsBench"
        or cuts.get("cut_count") != 4
        or cuts.get("quantity") != "Cp=2*pMeanTrim/(38.889^2)"
        or cuts.get("association") != "native_surface_VTP_CellData"
        or cuts.get("extraction_status") != "pending_immutable_owner_cut_support"
        or cuts.get("reduction")
        != "equal_case_equal_cut_global_R2_with_normalized_native_intersection_segment_length_support_per_cut"
    ):
        raise AutoCFD5Error("v9 Cp-cut metadata is inconsistent")
    cut_stations = cuts.get("stations")
    if not isinstance(cut_stations, list):
        raise AutoCFD5Error("v9 Cp-cut stations must be an array")
    cut_ids = tuple(
        _nonempty_string(item.get("id"), "Cp-cut station id")
        for item in cut_stations
        if isinstance(item, Mapping)
    )
    if len(cut_ids) != len(cut_stations):
        raise AutoCFD5Error("v9 Cp-cut station is malformed")
    if "pressure_profiles" in profile or any(
        key.startswith("cp_") and key != "cp_probes"
        for key in source
    ):
        raise AutoCFD5Error("v9 submission profile cannot contain Cp-probe registries")
    excluded = profile.get("excluded_diagnostics")
    if not isinstance(excluded, Mapping) or excluded.get("cp_probes") != (
        "the_209_discrete_taps_are_not_part_of_the_drivaerml_submission_or_score"
    ):
        raise AutoCFD5Error("v9 profile must explicitly exclude the 209 Cp probes")
    return AutoCFD5SubmissionDefinition(
        velocity_lines=lines,
        velocity_samples=samples,
        pressure_cut_ids=cut_ids,
        source_sha256=tuple(sorted(source_hashes)),
    )


def _validate_source_hashes(value: tuple[str, ...], label: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise AutoCFD5Error(f"{label} must contain at least one source SHA-256")
    if any(not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None for item in value):
        raise AutoCFD5Error(f"{label} must contain lowercase SHA-256 hex digests")


@dataclass(frozen=True)
class CpProbeMappingEvidence:
    """One explicit case/tap projection-and-bridge result.

    The geometric producer is intentionally external while its output schema
    retains the normative identifiers and mapping gates needed for validation.
    """

    case_id: str
    autocfd_probe_id: int
    valid: bool
    reason: str
    mapped_point_m: tuple[float, float, float] | None
    raw_stl_triangle_id: int | None
    raw_vtk_polygon_id: int | None
    nominal_displacement_m: float | None
    bridge_distance_m: float | None
    bridge_abs_normal_dot: float | None
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.case_id, "case_id")
        _integer(self.autocfd_probe_id, "autocfd_probe_id", minimum=1)
        if not isinstance(self.valid, bool):
            raise AutoCFD5Error("valid must be Boolean")
        if not isinstance(self.reason, str):
            raise AutoCFD5Error("reason must be a string")
        _optional_point3(self.mapped_point_m, "mapped_point_m")
        _optional_nonnegative_integer(self.raw_stl_triangle_id, "raw_stl_triangle_id")
        _optional_nonnegative_integer(self.raw_vtk_polygon_id, "raw_vtk_polygon_id")
        for value, label in (
            (self.nominal_displacement_m, "nominal_displacement_m"),
            (self.bridge_distance_m, "bridge_distance_m"),
            (self.bridge_abs_normal_dot, "bridge_abs_normal_dot"),
        ):
            result = _optional_finite(value, label)
            if result is not None and result < 0.0:
                raise AutoCFD5Error(f"{label} must be non-negative")
        _validate_source_hashes(self.source_sha256, "source_sha256")
        required = (
            self.mapped_point_m,
            self.raw_stl_triangle_id,
            self.raw_vtk_polygon_id,
            self.nominal_displacement_m,
            self.bridge_distance_m,
            self.bridge_abs_normal_dot,
        )
        if self.valid and any(value is None for value in required):
            raise AutoCFD5Error("valid Cp mappings must include every mapping field")
        if not self.valid and not self.reason.strip():
            raise AutoCFD5Error("invalid Cp mappings must include a reason")


@dataclass(frozen=True)
class VelocityCellAssignmentEvidence:
    """One explicit case/line/sample containing-cell assignment result."""

    case_id: str
    profile_id: str
    sample_index: int
    point_m: tuple[float, float, float]
    distance_m: float
    valid: bool
    reason: str
    raw_vtk_cell_id: int | None
    candidate_count: int
    geometric_tolerance_m: float
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.case_id, "case_id")
        _nonempty_string(self.profile_id, "profile_id")
        _integer(self.sample_index, "sample_index")
        _point3(self.point_m, "point_m")
        if _finite(self.distance_m, "distance_m") < 0.0:
            raise AutoCFD5Error("distance_m must be non-negative")
        if not isinstance(self.valid, bool):
            raise AutoCFD5Error("valid must be Boolean")
        if not isinstance(self.reason, str):
            raise AutoCFD5Error("reason must be a string")
        _optional_nonnegative_integer(self.raw_vtk_cell_id, "raw_vtk_cell_id")
        _integer(self.candidate_count, "candidate_count")
        if _finite(self.geometric_tolerance_m, "geometric_tolerance_m") <= 0.0:
            raise AutoCFD5Error("geometric_tolerance_m must be positive")
        _validate_source_hashes(self.source_sha256, "source_sha256")
        if self.valid:
            if self.raw_vtk_cell_id is None or self.candidate_count < 1:
                raise AutoCFD5Error(
                    "valid velocity assignments require a raw cell ID and candidates"
                )
        elif not self.reason.strip():
            raise AutoCFD5Error("invalid velocity assignments must include a reason")


def _case_ids(case_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_nonempty_string(case_id, "case_id") for case_id in case_ids)
    if not result:
        raise AutoCFD5Error("expected case IDs cannot be empty")
    if len(result) != len(set(result)):
        raise AutoCFD5Error("expected case IDs must be unique")
    return result


def validate_cp_mapping_evidence(
    assignments: Iterable[CpProbeMappingEvidence],
    definition: AutoCFD5Definition,
    *,
    expected_case_ids: Sequence[str],
    require_all_valid: bool = True,
) -> dict[str, tuple[CpProbeMappingEvidence, ...]]:
    """Require exactly one mapping for every expected case and unique probe.

    The returned tuples use nominal registry order, independent of evidence
    input order.  Duplicate, missing, and unexpected records all fail.
    """

    cases = _case_ids(expected_case_ids)
    probe_ids = tuple(probe.autocfd_probe_id for probe in definition.cp_probes)
    expected_keys = {(case_id, probe_id) for case_id in cases for probe_id in probe_ids}
    by_key: dict[tuple[str, int], CpProbeMappingEvidence] = {}
    for assignment in assignments:
        if not isinstance(assignment, CpProbeMappingEvidence):
            raise AutoCFD5Error("Cp mapping evidence contains the wrong record type")
        key = (assignment.case_id, assignment.autocfd_probe_id)
        if key in by_key:
            raise AutoCFD5Error(f"duplicate Cp mapping evidence for {key}")
        by_key[key] = assignment
        if assignment.valid:
            assert assignment.nominal_displacement_m is not None
            assert assignment.bridge_distance_m is not None
            assert assignment.bridge_abs_normal_dot is not None
            if assignment.nominal_displacement_m > NOMINAL_PROBE_DISPLACEMENT_MAX_M:
                raise AutoCFD5Error(f"Cp mapping {key} exceeds nominal displacement gate")
            if assignment.bridge_distance_m > NATIVE_BRIDGE_MAX_DISTANCE_M:
                raise AutoCFD5Error(f"Cp mapping {key} exceeds native bridge distance gate")
            if not (
                NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT
                <= assignment.bridge_abs_normal_dot
                <= 1.0
            ):
                raise AutoCFD5Error(f"Cp mapping {key} fails native normal agreement gate")
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        raise AutoCFD5Error(
            "Cp mapping evidence must cover every expected case/probe exactly once "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if require_all_valid:
        invalid = [key for key, value in by_key.items() if not value.valid]
        if invalid:
            raise AutoCFD5Error(
                f"ranked Cp evidence cannot contain invalid mappings (count={len(invalid)})"
            )
    return {
        case_id: tuple(by_key[(case_id, probe_id)] for probe_id in probe_ids)
        for case_id in cases
    }


def validate_velocity_assignment_evidence(
    assignments: Iterable[VelocityCellAssignmentEvidence],
    definition: AutoCFD5Definition,
    *,
    expected_case_ids: Sequence[str],
    expected_tolerance_m: float = POINT_IN_CELL_CLOSURE_TOLERANCE_M,
    allow_incomplete_audit: bool = False,
) -> dict[str, tuple[VelocityCellAssignmentEvidence, ...]]:
    """Require exact rows and, by default, a valid mapping for every sample.

    ``allow_incomplete_audit`` retains the historical masked-support checks for
    non-ranking discovery evidence.  It is not permission to score before a
    hash-bound owner validity mask distinguishes approved exclusions from
    unresolved mapping failures.
    """

    cases = _case_ids(expected_case_ids)
    if not isinstance(allow_incomplete_audit, bool):
        raise AutoCFD5Error("allow_incomplete_audit must be Boolean")
    tolerance = _finite(expected_tolerance_m, "expected_tolerance_m")
    if tolerance <= 0.0:
        raise AutoCFD5Error("expected_tolerance_m must be positive")
    sample_keys = tuple((sample.profile_id, sample.sample_index) for sample in definition.velocity_samples)
    sample_by_key = {
        (sample.profile_id, sample.sample_index): sample
        for sample in definition.velocity_samples
    }
    expected_keys = {
        (case_id, profile_id, sample_index)
        for case_id in cases
        for profile_id, sample_index in sample_keys
    }
    by_key: dict[tuple[str, str, int], VelocityCellAssignmentEvidence] = {}
    for assignment in assignments:
        if not isinstance(assignment, VelocityCellAssignmentEvidence):
            raise AutoCFD5Error(
                "velocity assignment evidence contains the wrong record type"
            )
        key = (assignment.case_id, assignment.profile_id, assignment.sample_index)
        if key in by_key:
            raise AutoCFD5Error(f"duplicate velocity assignment evidence for {key}")
        by_key[key] = assignment
        sample = sample_by_key.get((assignment.profile_id, assignment.sample_index))
        if sample is not None and (
            not _same_point(assignment.point_m, sample.point_m)
            or not _close(assignment.distance_m, sample.distance_m)
        ):
            raise AutoCFD5Error(f"velocity assignment {key} differs from fixed grid")
        if not _close(assignment.geometric_tolerance_m, tolerance, tolerance=1.0e-15):
            raise AutoCFD5Error(f"velocity assignment {key} uses the wrong tolerance")
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        raise AutoCFD5Error(
            "velocity assignment evidence must cover every expected case/sample "
            f"exactly once (missing={missing}, unexpected={unexpected})"
        )

    invalid_keys = [key for key, value in by_key.items() if not value.valid]
    if invalid_keys and not allow_incomplete_audit:
        raise AutoCFD5Error(
            "ranked velocity evidence cannot contain unresolved invalid mappings "
            "before an immutable owner validity mask is bound "
            f"(count={len(invalid_keys)})"
        )

    samples_by_line: dict[str, list[VelocitySampleDefinition]] = {}
    for sample in definition.velocity_samples:
        samples_by_line.setdefault(sample.profile_id, []).append(sample)
    for case_id in cases:
        for profile_id, samples in samples_by_line.items():
            valid = np.asarray(
                [by_key[(case_id, profile_id, sample.sample_index)].valid for sample in samples],
                dtype=bool,
            )
            if not np.any(valid[:-1] & valid[1:]):
                raise AutoCFD5Error(
                    f"velocity evidence for {(case_id, profile_id)} has no positive "
                    "contributing arc length"
                )
    return {
        case_id: tuple(
            by_key[(case_id, profile_id, sample_index)]
            for profile_id, sample_index in sample_keys
        )
        for case_id in cases
    }


def cp_from_kinematic_pressure(
    pressure_m2_per_s2: Any,
    *,
    freestream_velocity_m_per_s: float = U_INF_M_PER_S,
) -> float | np.ndarray:
    """Return ``Cp = 2*pMeanTrim/Uinf^2`` from kinematic pressure."""

    velocity = _finite(freestream_velocity_m_per_s, "freestream_velocity_m_per_s")
    if velocity <= 0.0:
        raise AutoCFD5Error("freestream_velocity_m_per_s must be positive")
    try:
        pressure = np.asarray(pressure_m2_per_s2, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AutoCFD5Error("pressure must be numeric") from error
    if not np.all(np.isfinite(pressure)):
        raise AutoCFD5Error("pressure must contain only finite values")
    result = 2.0 * pressure / (velocity * velocity)
    if result.ndim == 0:
        return float(result)
    return result


def velocity_magnitude_ratio(
    velocity_m_per_s: Any,
    *,
    freestream_velocity_m_per_s: float = U_INF_M_PER_S,
) -> np.ndarray:
    """Return ``magnitude(UMeanTrim)/Uinf`` for vectors with final extent 3."""

    velocity_scale = _finite(
        freestream_velocity_m_per_s, "freestream_velocity_m_per_s"
    )
    if velocity_scale <= 0.0:
        raise AutoCFD5Error("freestream_velocity_m_per_s must be positive")
    try:
        values = np.asarray(velocity_m_per_s, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AutoCFD5Error("velocity must be numeric") from error
    if values.ndim < 1 or values.shape[-1] != 3:
        raise AutoCFD5Error("velocity must have final shape extent 3")
    if not np.all(np.isfinite(values)):
        raise AutoCFD5Error("velocity must contain only finite values")
    return np.linalg.norm(values, axis=-1) / velocity_scale


def sample_native_cell_data_zeroth_order(values: Any, raw_vtk_cell_ids: Any) -> np.ndarray:
    """Gather native CellData by raw ID without point interpolation or remeshing."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise AutoCFD5Error("native CellData values must be array-like") from error
    if array.ndim not in {1, 2} or array.shape[0] == 0:
        raise AutoCFD5Error("native CellData must have shape [cell] or [cell, component]")
    if array.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(array)):
        raise AutoCFD5Error("native CellData must contain finite numeric values")
    ids = np.asarray(raw_vtk_cell_ids)
    if ids.ndim != 1 or ids.dtype.kind not in {"i", "u"}:
        raise AutoCFD5Error("raw_vtk_cell_ids must be a one-dimensional integer array")
    if ids.dtype.kind == "i" and np.any(ids < 0):
        raise AutoCFD5Error("raw_vtk_cell_ids cannot be negative")
    if np.any(ids >= array.shape[0]):
        raise AutoCFD5Error("raw_vtk_cell_ids exceed the native CellData extent")
    return np.take(array, ids.astype(np.int64, copy=False), axis=0)


def cp_values_from_mappings(
    native_pressure_m2_per_s2: Any,
    mappings: Sequence[CpProbeMappingEvidence],
) -> np.ndarray:
    """Gather one complete valid case's pressure taps and apply the Cp transform."""

    if len(mappings) != CP_PROBE_COUNT:
        raise AutoCFD5Error(f"Cp mapping sequence must contain {CP_PROBE_COUNT} rows")
    if any(not mapping.valid or mapping.raw_vtk_polygon_id is None for mapping in mappings):
        raise AutoCFD5Error("Cp values cannot be scored from invalid mappings")
    ids = np.asarray([mapping.raw_vtk_polygon_id for mapping in mappings], dtype=np.int64)
    pressure = sample_native_cell_data_zeroth_order(native_pressure_m2_per_s2, ids)
    if pressure.ndim != 1:
        raise AutoCFD5Error("native pMeanTrim CellData must be scalar")
    result = cp_from_kinematic_pressure(pressure)
    assert isinstance(result, np.ndarray)
    return result


def velocity_ratios_from_assignments(
    native_velocity_m_per_s: Any,
    assignments: Sequence[VelocityCellAssignmentEvidence],
) -> tuple[np.ndarray, np.ndarray]:
    """Gather valid samples while retaining invalid rows as an explicit mask."""

    ratios = np.full(len(assignments), np.nan, dtype=np.float64)
    valid = np.asarray([assignment.valid for assignment in assignments], dtype=bool)
    valid_ids = [
        assignment.raw_vtk_cell_id for assignment in assignments if assignment.valid
    ]
    if any(raw_id is None for raw_id in valid_ids):
        raise AutoCFD5Error("valid velocity assignment is missing a raw cell ID")
    if valid_ids:
        gathered = sample_native_cell_data_zeroth_order(
            native_velocity_m_per_s,
            np.asarray(valid_ids, dtype=np.int64),
        )
        ratios[valid] = velocity_magnitude_ratio(gathered)
    return ratios, valid


def cp_probe_rmse(prediction: Any, truth: Any) -> float:
    """Return equal-probe RMSE for one case's 209 unique Cp probes."""

    try:
        prediction_array = np.asarray(prediction, dtype=np.float64)
        truth_array = np.asarray(truth, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AutoCFD5Error("Cp values must be numeric") from error
    expected_shape = (CP_PROBE_COUNT,)
    if prediction_array.shape != expected_shape or truth_array.shape != expected_shape:
        raise AutoCFD5Error(f"Cp values must have shape {expected_shape}")
    if not np.all(np.isfinite(prediction_array)) or not np.all(np.isfinite(truth_array)):
        raise AutoCFD5Error("Cp values must contain only finite values")
    error = prediction_array - truth_array
    result = math.sqrt(float(np.sum(error * error, dtype=np.float64)) / CP_PROBE_COUNT)
    if not math.isfinite(result):
        raise AutoCFD5Error("Cp RMSE became non-finite")
    return result


def velocity_profile_rmse(
    distance_m: Any,
    prediction_ratio: Any,
    truth_ratio: Any,
    valid: Any,
) -> float:
    """Return a low-level masked RMSE without bridging invalid gaps.

    Ranked use requires the Boolean mask to come from the immutable owner
    validity support.  A raw locator failure must not be converted into a false
    entry merely to obtain a partial score.
    """

    try:
        distance = np.asarray(distance_m, dtype=np.float64)
        prediction = np.asarray(prediction_ratio, dtype=np.float64)
        truth = np.asarray(truth_ratio, dtype=np.float64)
        valid_mask = np.asarray(valid)
    except (TypeError, ValueError) as error:
        raise AutoCFD5Error("profile arrays must be numeric") from error
    if distance.ndim != 1 or len(distance) < 2:
        raise AutoCFD5Error("distance_m must contain at least two samples")
    if prediction.shape != distance.shape or truth.shape != distance.shape:
        raise AutoCFD5Error("profile values must match distance_m shape")
    if valid_mask.shape != distance.shape or valid_mask.dtype.kind != "b":
        raise AutoCFD5Error("valid must be a Boolean mask matching distance_m")
    if not np.all(np.isfinite(distance)) or np.any(np.diff(distance) <= 0.0):
        raise AutoCFD5Error("distance_m must be finite and strictly increasing")
    if not np.all(np.isfinite(prediction[valid_mask])) or not np.all(
        np.isfinite(truth[valid_mask])
    ):
        raise AutoCFD5Error("valid profile samples must contain finite values")
    edges = valid_mask[:-1] & valid_mask[1:]
    if not np.any(edges):
        raise AutoCFD5Error("profile has no positive contributing arc length")
    ds = np.diff(distance)[edges]
    error_squared = (prediction - truth) ** 2
    numerator = float(
        np.sum(
            ds * (error_squared[:-1][edges] + error_squared[1:][edges]) * 0.5,
            dtype=np.float64,
        )
    )
    denominator = float(np.sum(ds, dtype=np.float64))
    result = math.sqrt(numerator / denominator)
    if not math.isfinite(result):
        raise AutoCFD5Error("velocity profile RMSE became non-finite")
    return result


__all__ = [
    "AutoCFD5Definition",
    "AutoCFD5Error",
    "AutoCFD5SubmissionDefinition",
    "CP_PANEL_COUNT",
    "CP_PANEL_MEMBERSHIP_COUNT",
    "CP_PROBE_COUNT",
    "CpComponentRule",
    "CpPanelMembership",
    "CpProbeDefinition",
    "CpProbeMappingEvidence",
    "NATIVE_BRIDGE_MAX_DISTANCE_M",
    "NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT",
    "NOMINAL_PROBE_DISPLACEMENT_MAX_M",
    "POINT_IN_CELL_CLOSURE_TOLERANCE_M",
    "U_INF_M_PER_S",
    "VELOCITY_LINE_COUNT",
    "VELOCITY_SAMPLE_COUNT",
    "VelocityCellAssignmentEvidence",
    "VelocityLineDefinition",
    "VelocitySampleDefinition",
    "cp_from_kinematic_pressure",
    "cp_probe_rmse",
    "cp_values_from_mappings",
    "load_autocfd5_definition",
    "load_autocfd5_submission_definition",
    "sample_native_cell_data_zeroth_order",
    "validate_cp_mapping_evidence",
    "validate_velocity_assignment_evidence",
    "velocity_magnitude_ratio",
    "velocity_profile_rmse",
    "velocity_ratios_from_assignments",
]
