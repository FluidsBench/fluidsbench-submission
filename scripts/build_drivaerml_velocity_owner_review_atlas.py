#!/usr/bin/env python3
"""Build a deterministic DrivAerML velocity-assignment owner-review atlas.

The atlas is a review aid, not owner approval and not active scoring support.
It replays the strict velocity-assignment aggregate before reading any rows,
retains every invalid sample in a hash-bound CSV, and renders all supplied
cases, resolutions, lines, and samples without loading the all-case corpus at
once.  Omit ``--pilot-case`` only when all 484 official cases are present.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    VELOCITY_LINE_COUNT,
    load_autocfd5_submission_definition,
)
from scripts import aggregate_drivaerml_velocity_assignments as aggregate_module  # noqa: E402
from scripts.aggregate_drivaerml_velocity_assignments import (  # noqa: E402
    DEFAULT_AUTOCFD5_PROFILE,
    DEFAULT_NATIVE_SOURCE_PIN,
    MANIFEST_SCHEMA as ASSIGNMENT_MANIFEST_SCHEMA,
    OFFICIAL_CASE_IDS,
    OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
    ROW_FIELDS,
    VelocityAssignmentAggregateError,
    _finite,
    _integer,
    _mapping,
    _read_json,
    _string,
    aggregate_velocity_assignments,
    sha256_file,
)
from scripts.generate_drivaerml_velocity_assignments import (  # noqa: E402
    ARTIFACT_SCHEMA,
    RECEIPT_STATUS,
    RESOLUTIONS,
)


ATLAS_SCHEMA = "drivaerml-autocfd5-velocity-owner-review-atlas-candidate-v1"
ATLAS_STATUS = "candidate_not_owner_approved_not_active_scoring_support"
ATLAS_ACTIVATION_STATUS = "does_not_activate_scoring_contract"
ATLAS_ALGORITHM_ID = "drivaerml-autocfd5-velocity-owner-review-atlas-v1"
INVALID_SAMPLE_SCHEMA = "drivaerml-autocfd5-velocity-invalid-samples-candidate-v1"
MATPLOTLIB_VERSION = "3.11.1"
NUMPY_VERSION = "2.2.6"
FIXED_PDF_TIMESTAMP = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)

FAILURE_COLUMNS = (
    "case_id",
    "nominal_spacing_mm",
    "profile_id",
    "sample_index",
    "x_m",
    "y_m",
    "z_m",
    "distance_m",
    "reason",
    "raw_vtk_cell_id",
    "candidate_count",
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")


class VelocityAtlasError(VelocityAssignmentAggregateError):
    """Raised when atlas inputs, coverage, or rendering are not exact."""


@dataclass(frozen=True)
class BoundArtifact:
    case_id: str
    spacing_mm: int
    spacing_m: float
    name: str
    path: Path
    size_bytes: int
    sha256: str
    assignment_evidence_sha256: str
    sample_count: int
    valid_count: int
    invalid_count: int
    tie_assignment_count: int
    candidate_count_sum: int


@dataclass(frozen=True)
class BoundCase:
    case_id: str
    receipt_path: Path
    receipt_sha256: str
    receipt_size_bytes: int
    point_count: int
    cell_count: int
    ordered_verified_segment_count: int
    artifacts: tuple[BoundArtifact, ...]


@dataclass(frozen=True)
class AtlasInputs:
    definition: Any
    mode: str
    case_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    aggregate_sha256: str
    aggregate_size_bytes: int
    aggregate: Mapping[str, Any]
    cases: tuple[BoundCase, ...]
    expected_keys_by_resolution: Mapping[int, tuple[tuple[str, int], ...]]


@dataclass(frozen=True)
class AtlasAnalysis:
    case_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    resolutions_mm: tuple[int, ...]
    retained_row_count: int
    valid_count: int
    invalid_count: int
    tie_assignment_count: int
    candidate_count_sum: int
    invalid_reason_counts: Mapping[str, int]
    selected_raw_vtk_cell_id_min: int | None
    selected_raw_vtk_cell_id_max: int | None
    by_resolution: tuple[Mapping[str, Any], ...]
    per_case: tuple[Mapping[str, Any], ...]
    per_profile_resolution: tuple[Mapping[str, Any], ...]
    invalid_fraction_matrices: Mapping[int, tuple[tuple[float, ...], ...]]
    tie_fraction_matrices: Mapping[int, tuple[tuple[float, ...], ...]]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_size(path: Path, label: str) -> int:
    try:
        return path.stat().st_size
    except OSError as error:
        raise VelocityAtlasError(f"cannot stat {label}: {path.name}") from error


def _assert_no_absolute_paths(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        if value.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
            raise VelocityAtlasError(f"{label} contains an absolute path")


def _artifact_descriptor(
    *, case_id: str, case_directory: Path, raw: Mapping[str, Any]
) -> BoundArtifact:
    spacing_mm = _integer(raw.get("nominal_spacing_mm"), "nominal_spacing_mm", minimum=1)
    name = _string(raw.get("artifact_name"), "artifact_name")
    if Path(name).name != name or name in {".", ".."}:
        raise VelocityAtlasError(f"{case_id} artifact name is not path-free")
    path = case_directory / name
    size = _integer(raw.get("artifact_size_bytes"), "artifact_size_bytes", minimum=1)
    digest = _string(raw.get("artifact_sha256"), "artifact_sha256")
    if _file_size(path, f"{case_id} artifact") != size or sha256_file(path) != digest:
        raise VelocityAtlasError(f"{case_id} {spacing_mm} mm artifact identity changed")
    return BoundArtifact(
        case_id=case_id,
        spacing_mm=spacing_mm,
        spacing_m=_finite(raw.get("nominal_spacing_m"), "nominal_spacing_m"),
        name=name,
        path=path,
        size_bytes=size,
        sha256=digest,
        assignment_evidence_sha256=_string(
            raw.get("assignment_evidence_sha256"), "assignment_evidence_sha256"
        ),
        sample_count=_integer(raw.get("sample_count"), "sample_count", minimum=1),
        valid_count=_integer(raw.get("valid_count"), "valid_count"),
        invalid_count=_integer(raw.get("invalid_count"), "invalid_count"),
        tie_assignment_count=_integer(
            raw.get("tie_assignment_count"), "tie_assignment_count"
        ),
        candidate_count_sum=_integer(
            raw.get("candidate_count_sum"), "candidate_count_sum"
        ),
    )


def load_atlas_inputs(
    *,
    aggregate_path: Path | str,
    receipts_root: Path | str,
    native_source_pin: Path | str = DEFAULT_NATIVE_SOURCE_PIN,
    autocfd5_profile: Path | str = DEFAULT_AUTOCFD5_PROFILE,
    pilot_case_ids: Sequence[str] = (),
    official_case_ids: Sequence[str] = OFFICIAL_CASE_IDS,
    expected_pin_sha256: str | None = OFFICIAL_NATIVE_SOURCE_PIN_SHA256,
) -> AtlasInputs:
    """Replay strict validation and bind row-bearing inputs without retaining rows."""

    aggregate_path = Path(aggregate_path).expanduser().resolve()
    receipts_root = Path(receipts_root).expanduser().resolve()
    native_source_pin = Path(native_source_pin).expanduser().resolve()
    autocfd5_profile = Path(autocfd5_profile).expanduser().resolve()
    aggregate_digest = sha256_file(aggregate_path)
    aggregate_size = _file_size(aggregate_path, "velocity aggregate")
    provided = _read_json(aggregate_path, "velocity aggregate")
    try:
        replayed = aggregate_velocity_assignments(
            receipts_root=receipts_root,
            native_source_pin=native_source_pin,
            autocfd5_profile=autocfd5_profile,
            pilot_case_ids=pilot_case_ids,
            official_case_ids=official_case_ids,
            expected_pin_sha256=expected_pin_sha256,
        )
    except VelocityAssignmentAggregateError as error:
        raise VelocityAtlasError(f"strict assignment replay failed: {error}") from error
    if provided != replayed:
        raise VelocityAtlasError(
            "supplied aggregate differs from strict replay over current receipts"
        )
    if sha256_file(aggregate_path) != aggregate_digest:
        raise VelocityAtlasError("velocity aggregate changed during validation")
    _assert_no_absolute_paths(provided, "velocity aggregate")

    source_bindings = _mapping(replayed["source_bindings"], "source bindings")
    expected_pin_digest = _string(
        source_bindings["native_source_pin_sha256"], "native-source pin SHA-256"
    )
    autocfd_binding = _mapping(source_bindings["autocfd5"], "AutoCFD5 binding")
    expected_profile_digest = _string(
        autocfd_binding["profile_sha256"], "AutoCFD5 profile SHA-256"
    )
    if sha256_file(native_source_pin) != expected_pin_digest:
        raise VelocityAtlasError("native-source pin changed after strict replay")
    if sha256_file(autocfd5_profile) != expected_profile_digest:
        raise VelocityAtlasError("AutoCFD5 profile changed after strict replay")
    try:
        definition = load_autocfd5_submission_definition(autocfd5_profile)
    except ValueError as error:
        raise VelocityAtlasError("AutoCFD5 profile changed during atlas binding") from error
    if sha256_file(autocfd5_profile) != expected_profile_digest:
        raise VelocityAtlasError("AutoCFD5 profile changed during atlas binding")
    profile_ids = tuple(line.profile_id for line in definition.velocity_lines)
    if len(profile_ids) != VELOCITY_LINE_COUNT or len(set(profile_ids)) != len(profile_ids):
        raise VelocityAtlasError("AutoCFD5 profile IDs are not exactly sixteen unique lines")
    expected_samples = aggregate_module._expected_samples_by_resolution(definition)
    expected_keys = {
        spacing_mm: tuple((sample.profile_id, sample.sample_index) for sample in samples)
        for spacing_mm, samples in expected_samples.items()
    }

    case_ids = tuple(str(case["case_id"]) for case in replayed["cases"])
    expected_case_ids = tuple(pilot_case_ids) if pilot_case_ids else tuple(official_case_ids)
    expected_case_ids = tuple(sorted(expected_case_ids, key=aggregate_module._case_number))
    if case_ids != expected_case_ids or len(case_ids) != len(set(case_ids)):
        raise VelocityAtlasError("aggregate case coverage/order is not exact")
    mode = str(replayed["mode"])
    expected_mode = "explicit_non_public_pilot" if pilot_case_ids else "complete_484_case_default"
    if mode != expected_mode:
        raise VelocityAtlasError("aggregate mode differs from requested atlas scope")

    bound_cases: list[BoundCase] = []
    for raw_case in replayed["cases"]:
        case_id = str(raw_case["case_id"])
        case_directory = receipts_root / case_id
        receipt_path = case_directory / "receipt.json"
        receipt_sha = str(raw_case["receipt_sha256"])
        receipt_size = _file_size(receipt_path, f"{case_id} receipt")
        if sha256_file(receipt_path) != receipt_sha:
            raise VelocityAtlasError(f"{case_id} receipt changed after strict replay")
        geometry = _mapping(raw_case["geometry"], f"{case_id} geometry")
        artifacts = tuple(
            _artifact_descriptor(
                case_id=case_id,
                case_directory=case_directory,
                raw=_mapping(raw, f"{case_id} resolution"),
            )
            for raw in raw_case["resolutions"]
        )
        expected_resolutions = tuple(spacing for spacing, _, _ in RESOLUTIONS)
        if tuple(artifact.spacing_mm for artifact in artifacts) != expected_resolutions:
            raise VelocityAtlasError(f"{case_id} artifact resolution order is not exact")
        if any(
            artifact.sample_count != len(expected_keys[artifact.spacing_mm])
            for artifact in artifacts
        ):
            raise VelocityAtlasError(f"{case_id} artifact sample coverage is not exact")
        if sha256_file(receipt_path) != receipt_sha:
            raise VelocityAtlasError(f"{case_id} receipt changed during atlas binding")
        bound_cases.append(
            BoundCase(
                case_id=case_id,
                receipt_path=receipt_path,
                receipt_sha256=receipt_sha,
                receipt_size_bytes=receipt_size,
                point_count=_integer(geometry["point_count"], "point_count", minimum=1),
                cell_count=_integer(geometry["cell_count"], "cell_count", minimum=1),
                ordered_verified_segment_count=_integer(
                    geometry["ordered_verified_segment_count"],
                    "ordered_verified_segment_count",
                    minimum=1,
                ),
                artifacts=artifacts,
            )
        )
    return AtlasInputs(
        definition=definition,
        mode=mode,
        case_ids=case_ids,
        profile_ids=profile_ids,
        aggregate_sha256=aggregate_digest,
        aggregate_size_bytes=aggregate_size,
        aggregate=replayed,
        cases=tuple(bound_cases),
        expected_keys_by_resolution=expected_keys,
    )


def _load_bound_rows(
    inputs: AtlasInputs, case: BoundCase, artifact: BoundArtifact
) -> list[list[Any]]:
    if (
        _file_size(artifact.path, f"{case.case_id} artifact") != artifact.size_bytes
        or sha256_file(artifact.path) != artifact.sha256
    ):
        raise VelocityAtlasError(
            f"{case.case_id} {artifact.spacing_mm} mm artifact changed before row replay"
        )
    value = _read_json(
        artifact.path, f"{case.case_id} {artifact.spacing_mm} mm assignment artifact"
    )
    if (
        value.get("schema") != ARTIFACT_SCHEMA
        or value.get("status") != RECEIPT_STATUS
        or value.get("case_id") != case.case_id
        or value.get("row_fields") != ROW_FIELDS
        or _mapping(value.get("resolution"), "resolution").get("nominal_spacing_mm")
        != artifact.spacing_mm
    ):
        raise VelocityAtlasError(
            f"{case.case_id} {artifact.spacing_mm} mm artifact identity is not exact"
        )
    rows = value.get("rows")
    expected_keys = inputs.expected_keys_by_resolution[artifact.spacing_mm]
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise VelocityAtlasError(
            f"{case.case_id} {artifact.spacing_mm} mm rows are incomplete"
        )
    seen: set[tuple[str, int]] = set()
    for position, (row, expected_key) in enumerate(zip(rows, expected_keys, strict=True)):
        if not isinstance(row, list) or len(row) != len(ROW_FIELDS):
            raise VelocityAtlasError(
                f"{case.case_id} {artifact.spacing_mm} mm row {position} shape is invalid"
            )
        key = (_string(row[0], "profile_id"), _integer(row[1], "sample_index"))
        if key in seen:
            raise VelocityAtlasError(
                f"{case.case_id} {artifact.spacing_mm} mm contains duplicate row {key}"
            )
        seen.add(key)
        if key != expected_key:
            raise VelocityAtlasError(
                f"{case.case_id} {artifact.spacing_mm} mm row order/coverage changed"
            )
    coverage = _mapping(value.get("coverage"), "coverage")
    if (
        coverage.get("complete_duplicate_free_no_omissions") is not True
        or coverage.get("actual_sample_count") != len(rows)
        or coverage.get("unique_line_sample_key_count") != len(seen)
    ):
        raise VelocityAtlasError(
            f"{case.case_id} {artifact.spacing_mm} mm coverage is not complete"
        )
    if sha256_file(artifact.path) != artifact.sha256:
        raise VelocityAtlasError(
            f"{case.case_id} {artifact.spacing_mm} mm artifact changed during row replay"
        )
    return rows


def _empty_counts() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "tie_assignment_count": 0,
        "candidate_count_sum": 0,
        "invalid_reason_counts": collections.Counter(),
        "selected_raw_vtk_cell_id_min": None,
        "selected_raw_vtk_cell_id_max": None,
    }


def _add_row(counts: dict[str, Any], row: Sequence[Any]) -> None:
    valid = bool(row[4])
    candidate_count = int(row[7])
    counts["sample_count"] += 1
    counts["candidate_count_sum"] += candidate_count
    if valid:
        counts["valid_count"] += 1
        counts["tie_assignment_count"] += candidate_count > 1
        raw_cell_id = int(row[6])
        current_min = counts["selected_raw_vtk_cell_id_min"]
        current_max = counts["selected_raw_vtk_cell_id_max"]
        counts["selected_raw_vtk_cell_id_min"] = (
            raw_cell_id if current_min is None else min(current_min, raw_cell_id)
        )
        counts["selected_raw_vtk_cell_id_max"] = (
            raw_cell_id if current_max is None else max(current_max, raw_cell_id)
        )
    else:
        counts["invalid_count"] += 1
        counts["invalid_reason_counts"][str(row[5])] += 1


def _finalize_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(counts["sample_count"]),
        "valid_count": int(counts["valid_count"]),
        "invalid_count": int(counts["invalid_count"]),
        "tie_assignment_count": int(counts["tie_assignment_count"]),
        "candidate_count_sum": int(counts["candidate_count_sum"]),
        "invalid_reason_counts": dict(sorted(counts["invalid_reason_counts"].items())),
        "selected_raw_vtk_cell_id_min": counts["selected_raw_vtk_cell_id_min"],
        "selected_raw_vtk_cell_id_max": counts["selected_raw_vtk_cell_id_max"],
    }


def analyse_atlas(inputs: AtlasInputs) -> AtlasAnalysis:
    """Stream validated artifacts into compact deterministic review summaries."""

    resolutions = tuple(spacing for spacing, _, _ in RESOLUTIONS)
    by_resolution_counts = {spacing: _empty_counts() for spacing in resolutions}
    profile_counts = {
        (spacing, profile_id): _empty_counts()
        for spacing in resolutions
        for profile_id in inputs.profile_ids
    }
    invalid_matrices: dict[int, list[tuple[float, ...]]] = {
        spacing: [] for spacing in resolutions
    }
    tie_matrices: dict[int, list[tuple[float, ...]]] = {
        spacing: [] for spacing in resolutions
    }
    case_summaries: list[Mapping[str, Any]] = []
    global_counts = _empty_counts()

    for case in inputs.cases:
        case_counts = _empty_counts()
        resolution_summaries: list[Mapping[str, Any]] = []
        for artifact in case.artifacts:
            rows = _load_bound_rows(inputs, case, artifact)
            resolution_counts = _empty_counts()
            per_line = {profile_id: _empty_counts() for profile_id in inputs.profile_ids}
            for row in rows:
                profile_id = str(row[0])
                if profile_id not in per_line:
                    raise VelocityAtlasError(f"unexpected velocity profile {profile_id!r}")
                for counts in (
                    global_counts,
                    case_counts,
                    resolution_counts,
                    by_resolution_counts[artifact.spacing_mm],
                    profile_counts[(artifact.spacing_mm, profile_id)],
                    per_line[profile_id],
                ):
                    _add_row(counts, row)
            summary = _finalize_counts(resolution_counts)
            if any(
                summary[key] != getattr(artifact, key)
                for key in (
                    "sample_count",
                    "valid_count",
                    "invalid_count",
                    "tie_assignment_count",
                    "candidate_count_sum",
                )
            ):
                raise VelocityAtlasError(
                    f"{case.case_id} {artifact.spacing_mm} mm atlas replay does not close"
                )
            line_summaries = []
            invalid_row: list[float] = []
            tie_row: list[float] = []
            for profile_id in inputs.profile_ids:
                line = _finalize_counts(per_line[profile_id])
                if line["sample_count"] < 1:
                    raise VelocityAtlasError(
                        f"{case.case_id} {artifact.spacing_mm} mm omits {profile_id}"
                    )
                invalid_row.append(line["invalid_count"] / line["sample_count"])
                tie_row.append(line["tie_assignment_count"] / line["sample_count"])
                line_summaries.append({"profile_id": profile_id, **line})
            invalid_matrices[artifact.spacing_mm].append(tuple(invalid_row))
            tie_matrices[artifact.spacing_mm].append(tuple(tie_row))
            resolution_summaries.append(
                {
                    "nominal_spacing_mm": artifact.spacing_mm,
                    **summary,
                    "per_line": line_summaries,
                }
            )
        case_summaries.append(
            {
                "case_id": case.case_id,
                "native_point_count": case.point_count,
                "native_cell_count": case.cell_count,
                "ordered_verified_segment_count": case.ordered_verified_segment_count,
                **_finalize_counts(case_counts),
                "by_resolution_mm": resolution_summaries,
            }
        )

    by_resolution = []
    aggregate_totals = _mapping(inputs.aggregate["totals"], "aggregate totals")
    aggregate_by_resolution = _mapping(
        aggregate_totals["by_resolution_mm"], "aggregate by-resolution totals"
    )
    for spacing in resolutions:
        summary = _finalize_counts(by_resolution_counts[spacing])
        expected = _mapping(aggregate_by_resolution[str(spacing)], "aggregate resolution")
        for key in (
            "sample_count",
            "valid_count",
            "invalid_count",
            "tie_assignment_count",
            "candidate_count_sum",
        ):
            if summary[key] != expected[key]:
                raise VelocityAtlasError(
                    f"{spacing} mm atlas totals differ from strict aggregate"
                )
        by_resolution.append({"nominal_spacing_mm": spacing, **summary})

    final_global = _finalize_counts(global_counts)
    if (
        final_global["sample_count"] != aggregate_totals["explicit_assignment_row_count"]
        or final_global["invalid_reason_counts"] != aggregate_totals["invalid_reason_counts"]
    ):
        raise VelocityAtlasError("atlas row totals differ from strict aggregate")
    profile_summaries = tuple(
        {
            "nominal_spacing_mm": spacing,
            "profile_id": profile_id,
            **_finalize_counts(profile_counts[(spacing, profile_id)]),
        }
        for spacing in resolutions
        for profile_id in inputs.profile_ids
    )
    return AtlasAnalysis(
        case_ids=inputs.case_ids,
        profile_ids=inputs.profile_ids,
        resolutions_mm=resolutions,
        retained_row_count=final_global["sample_count"],
        valid_count=final_global["valid_count"],
        invalid_count=final_global["invalid_count"],
        tie_assignment_count=final_global["tie_assignment_count"],
        candidate_count_sum=final_global["candidate_count_sum"],
        invalid_reason_counts=final_global["invalid_reason_counts"],
        selected_raw_vtk_cell_id_min=final_global["selected_raw_vtk_cell_id_min"],
        selected_raw_vtk_cell_id_max=final_global["selected_raw_vtk_cell_id_max"],
        by_resolution=tuple(by_resolution),
        per_case=tuple(case_summaries),
        per_profile_resolution=profile_summaries,
        invalid_fraction_matrices={
            spacing: tuple(rows) for spacing, rows in invalid_matrices.items()
        },
        tie_fraction_matrices={
            spacing: tuple(rows) for spacing, rows in tie_matrices.items()
        },
    )


def _format_float(value: object) -> str:
    return format(_finite(value, "CSV float"), ".17g")


def write_invalid_samples_csv(
    output_path: Path | str, *, inputs: AtlasInputs, analysis: AtlasAnalysis
) -> dict[str, Any]:
    """Atomically write every invalid assignment row in deterministic order."""

    destination = Path(output_path)
    if destination.suffix.lower() != ".csv":
        raise VelocityAtlasError("invalid-sample output must use .csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    row_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(FAILURE_COLUMNS)
            for case in inputs.cases:
                for artifact in case.artifacts:
                    for row in _load_bound_rows(inputs, case, artifact):
                        if bool(row[4]):
                            continue
                        point = row[2]
                        writer.writerow(
                            (
                                case.case_id,
                                artifact.spacing_mm,
                                row[0],
                                row[1],
                                _format_float(point[0]),
                                _format_float(point[1]),
                                _format_float(point[2]),
                                _format_float(row[3]),
                                row[5],
                                "" if row[6] is None else row[6],
                                row[7],
                            )
                        )
                        row_count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if row_count != analysis.invalid_count:
            raise VelocityAtlasError(
                "invalid-sample CSV row count differs from atlas analysis"
            )
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "schema": INVALID_SAMPLE_SCHEMA,
        "media_type": "text/csv; charset=utf-8",
        "column_order": list(FAILURE_COLUMNS),
        "row_count": row_count,
        "sha256": sha256_file(destination),
        "size_bytes": _file_size(destination, "invalid-sample CSV"),
    }


def _load_renderer() -> tuple[Any, Any, Any, Any, str]:
    try:
        import numpy as np
    except ImportError as error:
        raise VelocityAtlasError("deterministic atlas rendering requires pinned NumPy") from error
    if np.__version__ != NUMPY_VERSION:
        raise VelocityAtlasError(
            f"atlas rendering requires numpy=={NUMPY_VERSION}; found {np.__version__}"
        )
    previous_config_dir = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="drivaerml-velocity-atlas-mpl-") as config_dir:
        os.environ["MPLCONFIGDIR"] = config_dir
        try:
            import matplotlib
            if matplotlib.__version__ != MATPLOTLIB_VERSION:
                raise VelocityAtlasError(
                    f"atlas rendering requires matplotlib=={MATPLOTLIB_VERSION}; "
                    f"found {matplotlib.__version__}"
                )
            matplotlib.use("Agg", force=True)
            from matplotlib import pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.lines import Line2D
        except ImportError as error:
            raise VelocityAtlasError(
                f"atlas rendering requires matplotlib=={MATPLOTLIB_VERSION}"
            ) from error
        finally:
            if previous_config_dir is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_config_dir
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "figure.dpi": 100,
            "path.simplify": False,
            "pdf.compression": 6,
            "pdf.fonttype": 42,
            "savefig.dpi": 100,
        }
    )
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return np, plt, PdfPages, Line2D, sha256_file(font_path)


def _status_banner(fig: Any, subtitle: str = "") -> None:
    fig.text(
        0.5,
        0.992,
        "CANDIDATE - NOT OWNER APPROVED - NOT ACTIVE SCORING SUPPORT",
        ha="center",
        va="top",
        color="#9c1c1c",
        fontsize=10,
        fontweight="bold",
    )
    if subtitle:
        fig.text(0.5, 0.972, subtitle, ha="center", va="top", fontsize=7)


def _sparse_ticks(count: int, maximum: int) -> list[int]:
    if count <= maximum:
        return list(range(count))
    step = math.ceil(count / maximum)
    ticks = list(range(0, count, step))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    return ticks


def _render_summary_page(pdf: Any, plt: Any, inputs: AtlasInputs, analysis: AtlasAnalysis) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _status_banner(fig)
    lines = [
        "DrivAerML AutoCFD5 velocity candidate owner-review atlas",
        "",
        f"Scope: {inputs.mode}; {len(inputs.case_ids)} case(s); 16 lines; "
        "resolutions 1, 2, 5, 10 mm",
        f"Rows retained: {analysis.retained_row_count:,} / {analysis.retained_row_count:,}",
        f"Valid / invalid: {analysis.valid_count:,} / {analysis.invalid_count:,}",
        "Valid rows with more than one containing-cell candidate: "
        f"{analysis.tie_assignment_count:,}",
        "",
        "All invalid rows are retained in the hash-bound CSV and shown as red x marks.",
        "Every valid row is represented in the overview summaries and case pages.",
        "The PDF uses receipt coordinates and raw cell IDs; source mesh geometry is not rendered.",
        "This does not assert owner sign-off, scientific approval, or scoring activation.",
        "",
        f"Assignment aggregate SHA-256: {inputs.aggregate_sha256}",
        f"Status: {ATLAS_STATUS}",
    ]
    fig.text(0.075, 0.87, "\n".join(lines), va="top", family="monospace", fontsize=9)
    reasons = "\n".join(
        f"  {reason}: {count:,}" for reason, count in analysis.invalid_reason_counts.items()
    ) or "  none"
    fig.text(0.075, 0.30, "Invalid reasons:\n" + reasons, va="top", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def _render_overview_page(
    pdf: Any, np: Any, plt: Any, inputs: AtlasInputs, analysis: AtlasAnalysis, spacing_mm: int
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.0), sharex=True)
    _status_banner(fig, f"{spacing_mm} mm: all-case, all-line assignment review")
    arrays = (
        (
            np.asarray(analysis.invalid_fraction_matrices[spacing_mm], dtype=float),
            "Invalid-sample fraction (failures retained; 0=none, 1=all)",
            "Reds",
        ),
        (
            np.asarray(analysis.tie_fraction_matrices[spacing_mm], dtype=float),
            "Valid-sample multi-candidate fraction (deterministic lowest raw ID selected)",
            "Purples",
        ),
    )
    for ax, (values, title, cmap) in zip(axes, arrays, strict=True):
        image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
        yticks = _sparse_ticks(len(inputs.case_ids), 30)
        ax.set_yticks(yticks, [inputs.case_ids[index] for index in yticks])
        ax.set_ylabel("case")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.012)
    axes[-1].set_xticks(
        range(len(inputs.profile_ids)), inputs.profile_ids, rotation=0
    )
    axes[-1].set_xlabel("AutoCFD5 velocity line (all 16 retained)")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _render_case_page(
    pdf: Any, plt: Any, Line2D: Any, inputs: AtlasInputs, analysis: AtlasAnalysis,
    case: BoundCase, case_index: int,
) -> None:
    colors = {1: "#08519c", 2: "#3182bd", 5: "#31a354", 10: "#756bb1"}
    rows_by_resolution_profile: dict[int, dict[str, list[list[Any]]]] = {}
    for artifact in case.artifacts:
        by_profile = {profile_id: [] for profile_id in inputs.profile_ids}
        for row in _load_bound_rows(inputs, case, artifact):
            by_profile[str(row[0])].append(row)
        rows_by_resolution_profile[artifact.spacing_mm] = by_profile

    fig, axes = plt.subplots(4, 4, figsize=(17, 11), sharey=True)
    summary = analysis.per_case[case_index]
    _status_banner(
        fig,
        f"{case.case_id}: all 16 lines and four resolutions; "
        f"valid {summary['valid_count']}/{summary['sample_count']}; "
        f"invalid {summary['invalid_count']}; multi-candidate {summary['tie_assignment_count']}",
    )
    for ax, profile_id in zip(axes.flat, inputs.profile_ids, strict=True):
        invalid_for_profile = 0
        tie_for_profile = 0
        for spacing_mm, _, _ in RESOLUTIONS:
            rows = rows_by_resolution_profile[spacing_mm][profile_id]
            valid = [row for row in rows if bool(row[4])]
            invalid = [row for row in rows if not bool(row[4])]
            ties = [row for row in valid if int(row[7]) > 1]
            invalid_for_profile += len(invalid)
            tie_for_profile += len(ties)
            if valid:
                ax.scatter(
                    [float(row[3]) for row in valid],
                    [int(row[6]) / case.cell_count for row in valid],
                    s=2.2,
                    color=colors[spacing_mm],
                    marker=".",
                    linewidths=0.0,
                    rasterized=True,
                )
            if ties:
                ax.scatter(
                    [float(row[3]) for row in ties],
                    [int(row[6]) / case.cell_count for row in ties],
                    s=7.0,
                    facecolors="none",
                    edgecolors=colors[spacing_mm],
                    linewidths=0.35,
                    rasterized=True,
                )
            if invalid:
                ax.scatter(
                    [float(row[3]) for row in invalid],
                    [-0.045] * len(invalid),
                    s=6.0,
                    color="#cb181d",
                    marker="x",
                    linewidths=0.45,
                    rasterized=True,
                )
        ax.set_title(f"{profile_id}  invalid={invalid_for_profile} tie={tie_for_profile}")
        ax.set_ylim(-0.08, 1.02)
        ax.grid(True, linewidth=0.25, color="#e0e0e0")
        ax.tick_params(labelsize=5)
        ax.set_xlabel("line distance [m]", fontsize=6)
        ax.set_ylabel("raw cell ID / cell count", fontsize=6)
    handles = [
        Line2D([], [], linestyle="none", marker=".", color=colors[spacing], label=f"{spacing} mm")
        for spacing, _, _ in RESOLUTIONS
    ]
    handles.extend(
        (
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                color="#636363",
                label="multi-candidate",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="x",
                color="#cb181d",
                label="invalid (exact row in CSV)",
            ),
        )
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.95), h_pad=1.0, w_pad=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def render_atlas_pdf(
    *, output_path: Path | str, inputs: AtlasInputs, analysis: AtlasAnalysis
) -> dict[str, Any]:
    """Render an atomic, path-free-evidenced deterministic review PDF."""

    np, plt, PdfPages, Line2D, font_sha256 = _load_renderer()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.pdf", dir=destination.parent
    )
    os.close(descriptor)
    metadata = {
        "Title": "DrivAerML AutoCFD5 velocity candidate owner-review atlas",
        "Author": "FluidsBench candidate evaluator",
        "Subject": ATLAS_STATUS,
        "Keywords": "DrivAerML AutoCFD5 velocity candidate owner review",
        "Creator": ATLAS_ALGORITHM_ID,
        "Producer": f"matplotlib {MATPLOTLIB_VERSION}",
        "CreationDate": FIXED_PDF_TIMESTAMP,
        "ModDate": FIXED_PDF_TIMESTAMP,
    }
    try:
        with PdfPages(temporary_name, metadata=metadata) as pdf:
            _render_summary_page(pdf, plt, inputs, analysis)
            for spacing_mm, _, _ in RESOLUTIONS:
                _render_overview_page(pdf, np, plt, inputs, analysis, spacing_mm)
            for case_index, case in enumerate(inputs.cases):
                _render_case_page(
                    pdf, plt, Line2D, inputs, analysis, case, case_index
                )
        with Path(temporary_name).open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "matplotlib": MATPLOTLIB_VERSION,
        "numpy": NUMPY_VERSION,
        "bundled_dejavu_sans_sha256": font_sha256,
        "page_count": 5 + len(inputs.case_ids),
        "pdf_sha256": sha256_file(destination),
        "pdf_size_bytes": _file_size(destination, "velocity atlas PDF"),
    }


def build_manifest(
    *,
    inputs: AtlasInputs,
    analysis: AtlasAnalysis,
    renderer: Mapping[str, Any],
    invalid_samples_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact path-free manifest that cannot imply owner approval."""

    source_receipts = []
    for case in inputs.cases:
        source_receipts.append(
            {
                "case_id": case.case_id,
                "receipt_json_sha256": case.receipt_sha256,
                "receipt_json_size_bytes": case.receipt_size_bytes,
                "artifacts": [
                    {
                        "nominal_spacing_mm": artifact.spacing_mm,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                        "assignment_evidence_sha256": artifact.assignment_evidence_sha256,
                    }
                    for artifact in case.artifacts
                ],
            }
        )
    complete = inputs.mode == "complete_484_case_default"
    manifest: dict[str, Any] = {
        "schema": ATLAS_SCHEMA,
        "schema_version": 1,
        "mode": inputs.mode,
        "status": ATLAS_STATUS,
        "owner_visual_signoff_claimed": False,
        "scientific_approval_claimed": False,
        "public_scoring_support_eligible": False,
        "activation_status": ATLAS_ACTIVATION_STATUS,
        "scope": {
            "case_count": len(inputs.case_ids),
            "case_ids": list(inputs.case_ids),
            "complete_484_public_case_set": complete,
            "line_count": VELOCITY_LINE_COUNT,
            "profile_ids": list(inputs.profile_ids),
            "resolutions_mm": list(analysis.resolutions_mm),
            "expected_assignment_row_count": int(
                inputs.aggregate["totals"]["explicit_assignment_row_count"]
            ),
            "retained_assignment_row_count": analysis.retained_row_count,
            "complete_duplicate_free_no_omissions": True,
        },
        "source": {
            "aggregate": {
                "schema": ASSIGNMENT_MANIFEST_SCHEMA,
                "sha256": inputs.aggregate_sha256,
                "size_bytes": inputs.aggregate_size_bytes,
            },
            "bindings": inputs.aggregate["source_bindings"],
            "kernel": inputs.aggregate["kernel"],
            "receipts": source_receipts,
            "source_geometry_rendered": False,
        },
        "algorithm": {
            "id": ATLAS_ALGORITHM_ID,
            "case_order": "increasing_run_number",
            "resolution_order_mm": list(analysis.resolutions_mm),
            "profile_order": "exact_v9_diagnostic_velocity_line_registry_order",
            "sample_order": "exact_endpoint_inclusive_registry_order",
            "invalid_sample_policy": "retain_every_failure_in_hash_bound_csv_and_mark_in_pdf",
            "valid_sample_policy": "represent_every_row_in_summaries_and_case_page",
            "case_page_raw_cell_axis": "selected_raw_vtk_cell_id_divided_by_native_cell_count",
            "geometry_policy": "receipt_sample_coordinates_only_no_source_mesh_geometry_rendered",
            "pdf_point_policy": "all_points_rendered_with_rasterized_markers_no_sampling",
            "pdf_timestamp_utc": "2000-01-01T00:00:00Z",
        },
        "dependencies": {
            "numpy": renderer["numpy"],
            "matplotlib": renderer["matplotlib"],
            "bundled_dejavu_sans_sha256": renderer[
                "bundled_dejavu_sans_sha256"
            ],
        },
        "artifacts": {
            "pdf": {
                "media_type": "application/pdf",
                "sha256": renderer["pdf_sha256"],
                "size_bytes": renderer["pdf_size_bytes"],
                "page_count": renderer["page_count"],
            },
            "invalid_samples": dict(invalid_samples_artifact),
        },
        "summary": {
            "assignment_row_count": analysis.retained_row_count,
            "valid_count": analysis.valid_count,
            "invalid_count": analysis.invalid_count,
            "tie_assignment_count": analysis.tie_assignment_count,
            "candidate_count_sum": analysis.candidate_count_sum,
            "invalid_reason_counts": dict(analysis.invalid_reason_counts),
            "selected_raw_vtk_cell_id_min": analysis.selected_raw_vtk_cell_id_min,
            "selected_raw_vtk_cell_id_max": analysis.selected_raw_vtk_cell_id_max,
            "by_resolution_mm": list(analysis.by_resolution),
        },
        "per_case": list(analysis.per_case),
        "per_profile_resolution": list(analysis.per_profile_resolution),
        "claims": {
            "owner_visual_signoff": False,
            "owner_scientific_approval": False,
            "scoring_contract_active": False,
            "official_submission_support": False,
            "independent_participant_dry_run": False,
            "resolution_convergence": False,
            "model_ordering": False,
        },
    }
    if (
        manifest["scope"]["expected_assignment_row_count"]
        != manifest["scope"]["retained_assignment_row_count"]
        or invalid_samples_artifact.get("row_count") != analysis.invalid_count
    ):
        raise VelocityAtlasError("atlas manifest would omit assignment or failure rows")
    _assert_no_absolute_paths(manifest, "velocity atlas manifest")
    return manifest


def write_manifest(path: Path | str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise VelocityAtlasError("atlas manifest output must use .json")
    _assert_no_absolute_paths(manifest, "velocity atlas manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "size_bytes": _file_size(destination, "velocity atlas manifest"),
        "sha256": sha256_file(destination),
    }


def _refuse_output_collisions(
    *,
    aggregate_path: Path | str,
    receipts_root: Path | str,
    native_source_pin: Path | str,
    autocfd5_profile: Path | str,
    output_pdf: Path | str,
    output_invalid_samples: Path | str,
    output_manifest: Path | str,
) -> None:
    source_files = {
        Path(aggregate_path).expanduser().resolve(),
        Path(native_source_pin).expanduser().resolve(),
        Path(autocfd5_profile).expanduser().resolve(),
    }
    receipt_root = Path(receipts_root).expanduser().resolve()
    outputs = tuple(
        Path(path).expanduser().resolve()
        for path in (output_pdf, output_invalid_samples, output_manifest)
    )
    if len(set(outputs)) != len(outputs):
        raise VelocityAtlasError("atlas PDF, failure CSV, and manifest outputs must differ")
    for output in outputs:
        if output in source_files or output == receipt_root or output.is_relative_to(receipt_root):
            raise VelocityAtlasError("atlas output must not overwrite validated inputs")
        if output.is_symlink():
            raise VelocityAtlasError("atlas output cannot be a symbolic link")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--native-source-pin", type=Path, default=DEFAULT_NATIVE_SOURCE_PIN)
    parser.add_argument("--autocfd5-profile", type=Path, default=DEFAULT_AUTOCFD5_PROFILE)
    parser.add_argument(
        "--pilot-case",
        action="append",
        default=[],
        help="explicit run_N for an incomplete, non-public pilot atlas",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-invalid-samples", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        _refuse_output_collisions(
            aggregate_path=args.aggregate,
            receipts_root=args.receipts_root,
            native_source_pin=args.native_source_pin,
            autocfd5_profile=args.autocfd5_profile,
            output_pdf=args.output_pdf,
            output_invalid_samples=args.output_invalid_samples,
            output_manifest=args.output_manifest,
        )
        inputs = load_atlas_inputs(
            aggregate_path=args.aggregate,
            receipts_root=args.receipts_root,
            native_source_pin=args.native_source_pin,
            autocfd5_profile=args.autocfd5_profile,
            pilot_case_ids=args.pilot_case,
        )
        analysis = analyse_atlas(inputs)
        failures = write_invalid_samples_csv(
            args.output_invalid_samples, inputs=inputs, analysis=analysis
        )
        renderer = render_atlas_pdf(
            output_path=args.output_pdf, inputs=inputs, analysis=analysis
        )
        manifest = build_manifest(
            inputs=inputs,
            analysis=analysis,
            renderer=renderer,
            invalid_samples_artifact=failures,
        )
        manifest_identity = write_manifest(args.output_manifest, manifest)
    except VelocityAssignmentAggregateError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": ATLAS_STATUS,
                "case_count": len(inputs.case_ids),
                "invalid_sample_count": analysis.invalid_count,
                "pdf": str(args.output_pdf),
                "invalid_samples": str(args.output_invalid_samples),
                "manifest": str(args.output_manifest),
                "pdf_sha256": renderer["pdf_sha256"],
                "invalid_samples_sha256": failures["sha256"],
                "manifest_sha256": manifest_identity["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
