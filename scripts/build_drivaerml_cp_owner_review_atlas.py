#!/usr/bin/env python3
"""Build a deterministic candidate AutoCFD5 Cp owner-review atlas.

The atlas is a review aid, not owner approval and not active scoring support.
It consumes the strict all-case (or explicit pilot) aggregate plus every
hash-bound per-case JSON receipt because the aggregate intentionally omits the
209 row coordinates.  Invalid and review rows remain visible in every output.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    CP_PANEL_COUNT,
    CP_PROBE_COUNT,
    NATIVE_BRIDGE_MAX_DISTANCE_M,
    NOMINAL_PROBE_DISPLACEMENT_MAX_M,
)
from scripts.aggregate_drivaerml_cp_support import (  # noqa: E402
    AGGREGATE_SCHEMA,
    DEFAULT_AUTOCFD5_PROFILE,
    OFFICIAL_CASE_IDS,
    OFFICIAL_REGISTRY_SHA256,
    PINNED_DEPENDENCIES,
    RECEIPT_STATUS,
    CpSupportAggregateError,
    _assert_no_absolute_paths,
    _boolean,
    _case_number,
    _canonical_json_bytes,
    _exact_keys,
    _integer,
    _load_definition,
    _mapping,
    _ordered_cases,
    _read_json,
    _sha256,
    _string,
    _validate_rows,
    sha256_file,
    write_evidence,
)
from scripts.build_drivaerml_cp_case_support import EVIDENCE_SCHEMA  # noqa: E402


ATLAS_SCHEMA = "drivaerml-autocfd5-cp-owner-review-atlas-candidate-v1"
ATLAS_STATUS = "candidate_not_owner_approved_not_active_scoring_support"
ATLAS_ACTIVATION_STATUS = "does_not_activate_scoring_contract"
ATLAS_ALGORITHM_ID = "drivaerml-autocfd5-cp-owner-review-atlas-v1"
MATPLOTLIB_VERSION = "3.11.1"
NUMPY_VERSION = "2.2.6"
FIXED_PDF_TIMESTAMP = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)

_MAPPING_REASONS = (
    "declared_component_absent",
    "declared_cut_has_no_finite_nondegenerate_intersection",
    "declared_component_has_no_finite_nondegenerate_triangle",
    "nominal_displacement_exceeds_10pct_wheelbase",
    "no_native_polygon_bounds_candidate_within_2mm",
    "no_finite_nondegenerate_native_polygon",
    "native_bridge_distance_exceeds_2mm",
    "native_bridge_normal_agreement_below_cos30",
)
_CATEGORY_ORDER = (
    "automated_valid_pending_owner_review",
    "automated_valid_review_flag_pending_owner_review",
    *(f"mapping:{reason}" for reason in _MAPPING_REASONS),
    "truth:nonfinite_native_pMeanTrim",
)
_CATEGORY_COLORS = (
    "#3182bd",
    "#f2b134",
    "#cb181d",
    "#e6550d",
    "#a63603",
    "#7f0000",
    "#b2182b",
    "#ef3b2c",
    "#fb6a4a",
    "#99000d",
    "#756bb1",
)
_SHORT_REASON = {
    "declared_component_absent": "component absent",
    "declared_cut_has_no_finite_nondegenerate_intersection": "no cut intersection",
    "declared_component_has_no_finite_nondegenerate_triangle": "no component facet",
    "nominal_displacement_exceeds_10pct_wheelbase": "displacement gate",
    "no_native_polygon_bounds_candidate_within_2mm": "no native candidate",
    "no_finite_nondegenerate_native_polygon": "no native polygon",
    "native_bridge_distance_exceeds_2mm": "bridge gate",
    "native_bridge_normal_agreement_below_cos30": "normal gate",
    "nonfinite_native_pMeanTrim": "non-finite pMeanTrim",
}

_AGGREGATE_KEYS = {
    "schema",
    "mode",
    "status",
    "complete",
    "public_evidence_eligible",
    "public_scoring_support_eligible",
    "owner_visual_signoff_claimed",
    "activation_status",
    "case_count",
    "official_case_count",
    "omitted_official_case_count",
    "case_order",
    "source",
    "dependencies",
    "candidate_stl_inventory",
    "aggregate",
    "cases",
}
_CASE_KEYS = {
    "case_id",
    "receipt_json_sha256",
    "receipt_json_size_bytes",
    "mapping_csv_sha256",
    "mapping_csv_size_bytes",
    "stl",
    "boundary",
    "validity",
}
_RECEIPT_KEYS = {
    "schema",
    "schema_version",
    "case_id",
    "status",
    "owner_visual_signoff_claimed",
    "definition",
    "sources",
    "dependencies",
    "algorithm",
    "summary",
    "artifacts",
    "rows",
}


class CpAtlasError(CpSupportAggregateError):
    """Raised when atlas inputs or deterministic rendering are not exact."""


@dataclass(frozen=True)
class AtlasInputs:
    definition: Any
    mode: str
    case_ids: tuple[str, ...]
    aggregate_sha256: str
    aggregate_size_bytes: int
    profile_sha256: str
    registry_sha256: Mapping[str, str]
    aggregate: Mapping[str, Any]
    aggregate_cases: Mapping[str, Mapping[str, Any]]
    receipt_evidence: tuple[Mapping[str, Any], ...]
    rows_by_case: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True)
class AtlasAnalysis:
    case_ids: tuple[str, ...]
    probe_ids: tuple[int, ...]
    categories: tuple[str, ...]
    category_matrix: tuple[tuple[int, ...], ...]
    displacement_matrix_m: tuple[tuple[float | None, ...], ...]
    bridge_matrix_m: tuple[tuple[float | None, ...], ...]
    duplicate_matrix: tuple[tuple[int, ...], ...]
    adjacent_duplicate_matrix: tuple[tuple[bool, ...], ...]
    duplicate_groups: tuple[Mapping[str, Any], ...]
    invalid_rows: tuple[Mapping[str, Any], ...]
    review_rows: tuple[Mapping[str, Any], ...]
    case_summaries: tuple[Mapping[str, Any], ...]
    probe_summaries: tuple[Mapping[str, Any], ...]


def _file_size(path: Path, label: str) -> int:
    try:
        return path.stat().st_size
    except OSError as error:
        raise CpAtlasError(f"cannot stat {label}: {path}") from error


def _exact_counter(value: object, label: str) -> dict[str, int]:
    mapping = _mapping(value, label)
    result: dict[str, int] = {}
    for key, count in mapping.items():
        result[_string(key, f"{label} key")] = _integer(
            count, f"{label}.{key}", minimum=1
        )
    return dict(sorted(result.items()))


def _stats(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "finite_count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    finite.sort()
    return {
        "finite_count": len(finite),
        "minimum": finite[0],
        "median": float(statistics.median(finite)),
        "maximum": finite[-1],
    }


def _validate_scope(
    aggregate: Mapping[str, Any],
    *,
    pilot_case_ids: Sequence[str] | None,
    official_case_ids: Sequence[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    official = _ordered_cases(official_case_ids, "official case IDs")
    if pilot_case_ids is None:
        mode = "complete"
        selected = official
        expected_status = (
            "complete_all_official_cases_candidate_evidence_pending_owner_visual_review"
        )
        complete = True
    else:
        mode = "partial_pilot"
        selected = _ordered_cases(pilot_case_ids, "pilot case IDs")
        if not set(selected).issubset(official):
            raise CpAtlasError("pilot cases must be official pinned cases")
        if selected == official:
            raise CpAtlasError(
                "pilot mode must be a strict subset; omit --pilot-case for complete mode"
            )
        expected_status = "incomplete_non_public_pilot"
        complete = False
    if aggregate.get("mode") != mode or aggregate.get("status") != expected_status:
        raise CpAtlasError("aggregate mode/status differs from the requested atlas scope")
    if _boolean(aggregate.get("complete"), "aggregate.complete") != complete:
        raise CpAtlasError("aggregate complete flag differs from its mode")
    return mode, selected, official


def _validate_receipt(
    *,
    path: Path,
    case_id: str,
    case_record: Mapping[str, Any],
    definition: Any,
    profile_sha256: str,
    registry_sha256: Mapping[str, str],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    expected_hash = _sha256(
        case_record.get("receipt_json_sha256"),
        f"{case_id} aggregate receipt SHA-256",
    )
    expected_size = _integer(
        case_record.get("receipt_json_size_bytes"),
        f"{case_id} aggregate receipt size",
        minimum=1,
    )
    if sha256_file(path) != expected_hash or _file_size(path, f"{case_id} receipt") != expected_size:
        raise CpAtlasError(f"{case_id} receipt identity differs from the strict aggregate")
    receipt = _read_json(path, f"{case_id} Cp receipt")
    _exact_keys(receipt, _RECEIPT_KEYS, f"{case_id} receipt")
    if (
        receipt["schema"] != EVIDENCE_SCHEMA
        or receipt["schema_version"] != "1.0"
        or receipt["case_id"] != case_id
        or receipt["status"] != RECEIPT_STATUS
        or _boolean(
            receipt["owner_visual_signoff_claimed"],
            f"{case_id}.owner_visual_signoff_claimed",
        )
    ):
        raise CpAtlasError(f"{case_id} receipt schema/identity/status is not exact")

    receipt_definition = _exact_keys(
        receipt["definition"],
        {"profile_sha256", "registry_sha256", "probe_count", "rule_count"},
        f"{case_id} receipt definition",
    )
    if (
        receipt_definition["profile_sha256"] != profile_sha256
        or dict(_mapping(receipt_definition["registry_sha256"], "registry SHA-256"))
        != dict(registry_sha256)
        or receipt_definition["probe_count"] != CP_PROBE_COUNT
        or receipt_definition["rule_count"] != CP_PROBE_COUNT
    ):
        raise CpAtlasError(f"{case_id} receipt is not bound to the exact v8 definition")
    if dict(_mapping(receipt["dependencies"], f"{case_id} dependencies")) != PINNED_DEPENDENCIES:
        raise CpAtlasError(f"{case_id} receipt dependency versions are not pinned")

    aggregate_stl = _exact_keys(
        case_record.get("stl"),
        {"file", "sha256", "size_bytes", "facet_count", "solid_count", "solid_facet_counts"},
        f"{case_id} aggregate STL",
    )
    aggregate_boundary = _exact_keys(
        case_record.get("boundary"),
        {
            "file",
            "sha256",
            "size_bytes",
            "point_count",
            "polygon_count",
            "pMeanTrim_tuple_count",
            "pMeanTrim_component_count",
            "pMeanTrim_vtk_data_type",
        },
        f"{case_id} aggregate boundary",
    )
    sources = _exact_keys(receipt["sources"], {"stl", "boundary"}, f"{case_id} sources")
    receipt_stl = _mapping(sources["stl"], f"{case_id} receipt STL")
    receipt_boundary = _mapping(sources["boundary"], f"{case_id} receipt boundary")
    for key in ("file", "sha256", "size_bytes", "facet_count", "solid_count", "solid_facet_counts"):
        if receipt_stl.get(key) != aggregate_stl[key]:
            raise CpAtlasError(f"{case_id} receipt STL differs from the strict aggregate")
    aggregate_boundary_file = _string(
        aggregate_boundary["file"], f"{case_id} aggregate boundary file"
    )
    receipt_boundary_file = _string(
        receipt_boundary.get("file"), f"{case_id} receipt boundary file"
    )
    if Path(aggregate_boundary_file).name != receipt_boundary_file:
        raise CpAtlasError(f"{case_id} receipt boundary name differs from the strict aggregate")
    for key in (
        "sha256",
        "size_bytes",
        "point_count",
        "polygon_count",
        "pMeanTrim_tuple_count",
        "pMeanTrim_component_count",
        "pMeanTrim_vtk_data_type",
    ):
        if receipt_boundary.get(key) != aggregate_boundary[key]:
            raise CpAtlasError(f"{case_id} receipt boundary differs from the strict aggregate")

    solid_counts = {
        _string(key, f"{case_id} solid name"): _integer(
            value, f"{case_id} solid facet count", minimum=1
        )
        for key, value in _mapping(
            aggregate_stl["solid_facet_counts"], f"{case_id} solid facet counts"
        ).items()
    }
    rows, computed_summary = _validate_rows(
        receipt["rows"],
        case_id=case_id,
        definition=definition,
        stl_sha256=_sha256(aggregate_stl["sha256"], f"{case_id} STL SHA-256"),
        boundary_sha256=_sha256(
            aggregate_boundary["sha256"], f"{case_id} boundary SHA-256"
        ),
        stl_facet_count=_integer(
            aggregate_stl["facet_count"], f"{case_id} STL facet count", minimum=1
        ),
        solid_facet_counts=solid_counts,
        boundary_polygon_count=_integer(
            aggregate_boundary["polygon_count"],
            f"{case_id} boundary polygon count",
            minimum=1,
        ),
    )
    if dict(_mapping(receipt["summary"], f"{case_id} receipt summary")) != computed_summary:
        raise CpAtlasError(f"{case_id} receipt summary differs from its 209 rows")
    if dict(_mapping(case_record.get("validity"), f"{case_id} aggregate validity")) != computed_summary:
        raise CpAtlasError(f"{case_id} aggregate validity differs from its receipt rows")

    artifacts = _mapping(receipt["artifacts"], f"{case_id} receipt artifacts")
    mapping_csv_hash = _sha256(
        case_record.get("mapping_csv_sha256"), f"{case_id} aggregate CSV SHA-256"
    )
    if (
        artifacts.get("mapping_csv_sha256") != mapping_csv_hash
        or artifacts.get("mapping_csv_row_count") != CP_PROBE_COUNT
    ):
        raise CpAtlasError(f"{case_id} receipt CSV binding differs from the aggregate")
    if sha256_file(path) != expected_hash:
        raise CpAtlasError(f"{case_id} receipt changed during atlas validation")
    evidence = {
        "case_id": case_id,
        "receipt_json_sha256": expected_hash,
        "receipt_json_size_bytes": expected_size,
        "mapping_csv_sha256": mapping_csv_hash,
        "mapping_csv_size_bytes": _integer(
            case_record.get("mapping_csv_size_bytes"),
            f"{case_id} aggregate CSV size",
            minimum=1,
        ),
    }
    return tuple(rows), evidence


def load_atlas_inputs(
    *,
    aggregate_path: Path,
    profile_path: Path,
    receipt_paths: Sequence[Path],
    pilot_case_ids: Sequence[str] | None = None,
    official_case_ids: Sequence[str] = OFFICIAL_CASE_IDS,
) -> AtlasInputs:
    """Load and fail-closed validate all row-bearing atlas inputs."""

    aggregate_path = Path(aggregate_path)
    profile_path = Path(profile_path)
    aggregate_sha256 = sha256_file(aggregate_path)
    aggregate_size = _file_size(aggregate_path, "Cp aggregate")
    aggregate = _read_json(aggregate_path, "Cp aggregate")
    expected_aggregate_keys = set(_AGGREGATE_KEYS)
    if aggregate.get("mode") == "partial_pilot":
        expected_aggregate_keys.add("pilot_warning")
    _exact_keys(aggregate, expected_aggregate_keys, "Cp aggregate")
    if aggregate["schema"] != AGGREGATE_SCHEMA:
        raise CpAtlasError("Cp aggregate schema is not exact")
    mode, selected_cases, official_cases = _validate_scope(
        aggregate,
        pilot_case_ids=pilot_case_ids,
        official_case_ids=official_case_ids,
    )
    if (
        _boolean(
            aggregate["public_scoring_support_eligible"],
            "aggregate.public_scoring_support_eligible",
        )
        or _boolean(
            aggregate["owner_visual_signoff_claimed"],
            "aggregate.owner_visual_signoff_claimed",
        )
        or aggregate["activation_status"] != ATLAS_ACTIVATION_STATUS
    ):
        raise CpAtlasError("aggregate improperly claims approval or active scoring support")
    expected_public_evidence = mode == "complete"
    if _boolean(aggregate["public_evidence_eligible"], "public evidence flag") != expected_public_evidence:
        raise CpAtlasError("aggregate public-evidence flag differs from its mode")
    if (
        _integer(aggregate["case_count"], "aggregate case_count", minimum=1)
        != len(selected_cases)
        or _integer(
            aggregate["official_case_count"], "aggregate official_case_count", minimum=1
        )
        != len(official_cases)
        or _integer(
            aggregate["omitted_official_case_count"],
            "aggregate omitted_official_case_count",
        )
        != len(official_cases) - len(selected_cases)
        or aggregate["case_order"] != "native_source_pin_increasing_run_number"
    ):
        raise CpAtlasError("aggregate case scope/counts are not exact")
    if dict(_mapping(aggregate["dependencies"], "aggregate dependencies")) != PINNED_DEPENDENCIES:
        raise CpAtlasError("aggregate dependency versions are not pinned")

    definition, profile_sha256, registry_sha256 = _load_definition(profile_path)
    source = _mapping(aggregate["source"], "aggregate source")
    autocfd = _mapping(source.get("autocfd5"), "aggregate AutoCFD5 source")
    if (
        autocfd.get("profile_sha256") != profile_sha256
        or dict(_mapping(autocfd.get("registry_sha256"), "aggregate registries"))
        != dict(registry_sha256)
        or autocfd.get("probe_count") != CP_PROBE_COUNT
    ):
        raise CpAtlasError("aggregate is not bound to the exact AutoCFD5 v8 registries")
    if dict(registry_sha256) != OFFICIAL_REGISTRY_SHA256:
        raise CpAtlasError("AutoCFD5 registry identities are not official")

    raw_cases = aggregate["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != len(selected_cases):
        raise CpAtlasError("aggregate cases do not cover the requested scope")
    case_records: dict[str, Mapping[str, Any]] = {}
    for position, (raw_case, expected_case_id) in enumerate(
        zip(raw_cases, selected_cases, strict=True)
    ):
        record = _exact_keys(raw_case, _CASE_KEYS, f"aggregate case {position}")
        if record["case_id"] != expected_case_id:
            raise CpAtlasError("aggregate cases are not in exact requested order")
        case_records[expected_case_id] = record

    paths_by_case: dict[str, Path] = {}
    for raw_path in receipt_paths:
        path = Path(raw_path)
        identity = _read_json(path, "Cp receipt identity")
        case_id = _string(identity.get("case_id"), "Cp receipt case_id")
        if case_id in paths_by_case:
            raise CpAtlasError(f"duplicate receipt JSON for {case_id}")
        paths_by_case[case_id] = path
    if set(paths_by_case) != set(selected_cases):
        missing = sorted(set(selected_cases) - set(paths_by_case), key=_case_number)
        unexpected = sorted(set(paths_by_case) - set(selected_cases), key=_case_number)
        raise CpAtlasError(
            "receipt JSON inputs must cover the requested case set exactly once "
            f"(missing={missing}, unexpected={unexpected})"
        )

    rows_by_case: dict[str, tuple[Mapping[str, Any], ...]] = {}
    receipt_evidence: list[Mapping[str, Any]] = []
    for case_id in selected_cases:
        rows, evidence = _validate_receipt(
            path=paths_by_case[case_id],
            case_id=case_id,
            case_record=case_records[case_id],
            definition=definition,
            profile_sha256=profile_sha256,
            registry_sha256=registry_sha256,
        )
        rows_by_case[case_id] = rows
        receipt_evidence.append(evidence)

    counters: dict[str, collections.Counter[str]] = {
        "mapping_invalid_reason_counts": collections.Counter(),
        "truth_invalid_reason_counts": collections.Counter(),
        "owner_review_status_counts": collections.Counter(),
        "review_flag_counts": collections.Counter(),
        "pMeanTrim_vtk_data_type_counts": collections.Counter(),
    }
    sums = collections.Counter()
    for case_id in selected_cases:
        record = case_records[case_id]
        validity = _mapping(record["validity"], f"{case_id} validity")
        for key in (
            "mapping_valid_count",
            "mapping_invalid_count",
            "truth_valid_count",
            "truth_invalid_count",
            "support_valid_count",
        ):
            sums[key] += _integer(validity.get(key), f"{case_id}.{key}")
        for key in (
            "mapping_invalid_reason_counts",
            "truth_invalid_reason_counts",
            "owner_review_status_counts",
            "review_flag_counts",
        ):
            counters[key].update(_exact_counter(validity.get(key), f"{case_id}.{key}"))
        boundary = _mapping(record["boundary"], f"{case_id} boundary")
        counters["pMeanTrim_vtk_data_type_counts"].update(
            [_string(boundary.get("pMeanTrim_vtk_data_type"), "pMeanTrim VTK type")]
        )

    aggregate_summary = _mapping(aggregate["aggregate"], "aggregate summary")
    expected_summary: dict[str, Any] = {
        "probe_row_count": CP_PROBE_COUNT * len(selected_cases),
        **{key: sums[key] for key in sums},
        **{key: dict(sorted(counter.items())) for key, counter in counters.items()},
        "receipt_json_size_bytes": sum(
            _integer(record["receipt_json_size_bytes"], "receipt JSON size", minimum=1)
            for record in case_records.values()
        ),
        "mapping_csv_size_bytes": sum(
            _integer(record["mapping_csv_size_bytes"], "mapping CSV size", minimum=1)
            for record in case_records.values()
        ),
        "stl_size_bytes": sum(
            _integer(_mapping(record["stl"], "STL")["size_bytes"], "STL size", minimum=1)
            for record in case_records.values()
        ),
        "stl_facet_count": sum(
            _integer(_mapping(record["stl"], "STL")["facet_count"], "STL facets", minimum=1)
            for record in case_records.values()
        ),
        "boundary_size_bytes": sum(
            _integer(_mapping(record["boundary"], "boundary")["size_bytes"], "boundary size", minimum=1)
            for record in case_records.values()
        ),
        "boundary_polygon_count": sum(
            _integer(
                _mapping(record["boundary"], "boundary")["polygon_count"],
                "boundary polygons",
                minimum=1,
            )
            for record in case_records.values()
        ),
    }
    if dict(aggregate_summary) != expected_summary:
        raise CpAtlasError("aggregate summary does not close against its case receipts")
    if sha256_file(aggregate_path) != aggregate_sha256:
        raise CpAtlasError("Cp aggregate changed during atlas validation")
    _assert_no_absolute_paths(aggregate, "Cp aggregate")
    return AtlasInputs(
        definition=definition,
        mode=mode,
        case_ids=selected_cases,
        aggregate_sha256=aggregate_sha256,
        aggregate_size_bytes=aggregate_size,
        profile_sha256=profile_sha256,
        registry_sha256=dict(registry_sha256),
        aggregate=aggregate,
        aggregate_cases=case_records,
        receipt_evidence=tuple(receipt_evidence),
        rows_by_case=rows_by_case,
    )


def _row_category(row: Mapping[str, Any]) -> str:
    if not row["mapping_valid"]:
        return f"mapping:{row['mapping_reason']}"
    if not row["truth_valid"]:
        return f"truth:{row['truth_reason']}"
    return (
        "automated_valid_review_flag_pending_owner_review"
        if row["review_flags"]
        else "automated_valid_pending_owner_review"
    )


def _duplicate_groups_for_case(
    *, case_id: str, rows: Sequence[Mapping[str, Any]], definition: Any
) -> tuple[list[dict[str, Any]], dict[int, int], set[int]]:
    row_by_probe = {int(row["autocfd_probe_id"]): row for row in rows}
    adjacent_by_key: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    panel_rows: dict[str, list[Any]] = collections.defaultdict(list)
    for membership in definition.cp_panel_memberships:
        panel_rows[membership.panel_id].append(membership)
    for panel_id, memberships in panel_rows.items():
        for first, second in zip(memberships, memberships[1:]):
            if first.autocfd_probe_id == second.autocfd_probe_id:
                continue
            for support, key in (
                ("stl", "raw_stl_triangle_id"),
                ("native", "raw_vtk_polygon_id"),
            ):
                first_id = row_by_probe[first.autocfd_probe_id][key]
                second_id = row_by_probe[second.autocfd_probe_id][key]
                if first_id is not None and first_id == second_id:
                    adjacent_by_key[(support, int(first_id))].append(
                        {
                            "panel_id": panel_id,
                            "first_probe_id": first.autocfd_probe_id,
                            "second_probe_id": second.autocfd_probe_id,
                        }
                    )

    groups: list[dict[str, Any]] = []
    flags = {int(row["autocfd_probe_id"]): 0 for row in rows}
    adjacent_probes: set[int] = set()
    for support, key, bit in (
        ("stl", "raw_stl_triangle_id", 1),
        ("native", "raw_vtk_polygon_id", 2),
    ):
        ids: dict[int, set[int]] = collections.defaultdict(set)
        for row in rows:
            raw_id = row[key]
            if raw_id is not None:
                ids[int(raw_id)].add(int(row["autocfd_probe_id"]))
        for raw_id, probe_ids in sorted(ids.items()):
            if len(probe_ids) < 2:
                continue
            ordered_probe_ids = sorted(probe_ids)
            for probe_id in ordered_probe_ids:
                flags[probe_id] |= bit
            adjacent = sorted(
                adjacent_by_key.get((support, raw_id), []),
                key=lambda item: (
                    item["panel_id"],
                    item["first_probe_id"],
                    item["second_probe_id"],
                ),
            )
            for item in adjacent:
                adjacent_probes.update(
                    (item["first_probe_id"], item["second_probe_id"])
                )
            groups.append(
                {
                    "case_id": case_id,
                    "support": support,
                    "raw_id": raw_id,
                    "probe_ids": ordered_probe_ids,
                    "adjacent_pairs": adjacent,
                }
            )
    groups.sort(key=lambda item: (item["support"], item["raw_id"]))
    return groups, flags, adjacent_probes


def analyse_atlas(inputs: AtlasInputs) -> AtlasAnalysis:
    """Build deterministic all-row review matrices and compact summaries."""

    probe_ids = tuple(probe.autocfd_probe_id for probe in inputs.definition.cp_probes)
    category_index = {name: index for index, name in enumerate(_CATEGORY_ORDER)}
    category_matrix: list[tuple[int, ...]] = []
    displacement_matrix: list[tuple[float | None, ...]] = []
    bridge_matrix: list[tuple[float | None, ...]] = []
    duplicate_matrix: list[tuple[int, ...]] = []
    adjacent_matrix: list[tuple[bool, ...]] = []
    duplicate_groups: list[Mapping[str, Any]] = []
    invalid_rows: list[Mapping[str, Any]] = []
    review_rows: list[Mapping[str, Any]] = []
    case_summaries: list[Mapping[str, Any]] = []
    rows_by_probe: dict[int, list[tuple[str, Mapping[str, Any], int, bool]]] = {
        probe_id: [] for probe_id in probe_ids
    }

    for case_id in inputs.case_ids:
        rows = inputs.rows_by_case[case_id]
        groups, duplicate_flags, adjacent_probes = _duplicate_groups_for_case(
            case_id=case_id, rows=rows, definition=inputs.definition
        )
        duplicate_groups.extend(groups)
        categories: list[int] = []
        displacements: list[float | None] = []
        bridges: list[float | None] = []
        duplicate_row: list[int] = []
        adjacent_row: list[bool] = []
        mapping_reasons: collections.Counter[str] = collections.Counter()
        truth_reasons: collections.Counter[str] = collections.Counter()
        review_flags: collections.Counter[str] = collections.Counter()
        support_valid_count = 0
        for row in rows:
            probe_id = int(row["autocfd_probe_id"])
            category = _row_category(row)
            if category not in category_index:
                raise CpAtlasError(f"unexpected validated category {category!r}")
            categories.append(category_index[category])
            displacement = row["nominal_displacement_m"]
            bridge = row["bridge_distance_m"]
            displacements.append(None if displacement is None else float(displacement))
            bridges.append(None if bridge is None else float(bridge))
            duplicate_flag = duplicate_flags[probe_id]
            adjacent = probe_id in adjacent_probes
            duplicate_row.append(duplicate_flag)
            adjacent_row.append(adjacent)
            rows_by_probe[probe_id].append((case_id, row, duplicate_flag, adjacent))
            support_valid_count += int(bool(row["support_valid"]))
            if not row["mapping_valid"]:
                mapping_reasons[str(row["mapping_reason"])] += 1
            if not row["truth_valid"]:
                truth_reasons[str(row["truth_reason"])] += 1
            review_flags.update(str(flag) for flag in row["review_flags"])
            if not row["support_valid"]:
                invalid_rows.append(
                    {
                        "case_id": case_id,
                        "autocfd_probe_id": probe_id,
                        "category": category,
                        "mapping_reason": str(row["mapping_reason"]),
                        "truth_reason": str(row["truth_reason"]),
                    }
                )
            if row["review_flags"]:
                review_rows.append(
                    {
                        "case_id": case_id,
                        "autocfd_probe_id": probe_id,
                        "review_flags": list(row["review_flags"]),
                    }
                )
        category_matrix.append(tuple(categories))
        displacement_matrix.append(tuple(displacements))
        bridge_matrix.append(tuple(bridges))
        duplicate_matrix.append(tuple(duplicate_row))
        adjacent_matrix.append(tuple(adjacent_row))
        case_summaries.append(
            {
                "case_id": case_id,
                "row_count": len(rows),
                "support_valid_count": support_valid_count,
                "support_invalid_count": len(rows) - support_valid_count,
                "mapping_invalid_reason_counts": dict(sorted(mapping_reasons.items())),
                "truth_invalid_reason_counts": dict(sorted(truth_reasons.items())),
                "review_flag_counts": dict(sorted(review_flags.items())),
                "nominal_displacement_m": _stats(
                    value for value in displacements if value is not None
                ),
                "native_bridge_distance_m": _stats(
                    value for value in bridges if value is not None
                ),
                "duplicate_group_count": len(groups),
                "duplicate_probe_count": sum(flag != 0 for flag in duplicate_row),
                "panel_adjacent_duplicate_probe_count": sum(adjacent_row),
            }
        )

    probe_summaries: list[Mapping[str, Any]] = []
    for probe_id in probe_ids:
        records = rows_by_probe[probe_id]
        mapping_reasons: collections.Counter[str] = collections.Counter()
        truth_reasons: collections.Counter[str] = collections.Counter()
        review_flags: collections.Counter[str] = collections.Counter()
        for _, row, _, _ in records:
            if not row["mapping_valid"]:
                mapping_reasons[str(row["mapping_reason"])] += 1
            if not row["truth_valid"]:
                truth_reasons[str(row["truth_reason"])] += 1
            review_flags.update(str(flag) for flag in row["review_flags"])
        probe_summaries.append(
            {
                "autocfd_probe_id": probe_id,
                "case_count": len(records),
                "support_valid_count": sum(bool(row["support_valid"]) for _, row, _, _ in records),
                "support_invalid_count": sum(not bool(row["support_valid"]) for _, row, _, _ in records),
                "mapping_invalid_reason_counts": dict(sorted(mapping_reasons.items())),
                "truth_invalid_reason_counts": dict(sorted(truth_reasons.items())),
                "review_flag_counts": dict(sorted(review_flags.items())),
                "nominal_displacement_m": _stats(
                    float(row["nominal_displacement_m"])
                    for _, row, _, _ in records
                    if row["nominal_displacement_m"] is not None
                ),
                "native_bridge_distance_m": _stats(
                    float(row["bridge_distance_m"])
                    for _, row, _, _ in records
                    if row["bridge_distance_m"] is not None
                ),
                "stl_duplicate_case_count": sum(bool(flag & 1) for _, _, flag, _ in records),
                "native_duplicate_case_count": sum(bool(flag & 2) for _, _, flag, _ in records),
                "panel_adjacent_duplicate_case_count": sum(adjacent for _, _, _, adjacent in records),
            }
        )

    expected_rows = len(inputs.case_ids) * CP_PROBE_COUNT
    if sum(len(row) for row in category_matrix) != expected_rows:
        raise CpAtlasError("atlas analysis omitted validated Cp rows")
    return AtlasAnalysis(
        case_ids=inputs.case_ids,
        probe_ids=probe_ids,
        categories=_CATEGORY_ORDER,
        category_matrix=tuple(category_matrix),
        displacement_matrix_m=tuple(displacement_matrix),
        bridge_matrix_m=tuple(bridge_matrix),
        duplicate_matrix=tuple(duplicate_matrix),
        adjacent_duplicate_matrix=tuple(adjacent_matrix),
        duplicate_groups=tuple(duplicate_groups),
        invalid_rows=tuple(invalid_rows),
        review_rows=tuple(review_rows),
        case_summaries=tuple(case_summaries),
        probe_summaries=tuple(probe_summaries),
    )


def _load_renderer() -> tuple[Any, Any, Any, Any, Any, Any, str]:
    try:
        import numpy as np
    except ImportError as error:
        raise CpAtlasError("deterministic atlas rendering requires pinned NumPy") from error
    if np.__version__ != NUMPY_VERSION:
        raise CpAtlasError(
            f"atlas rendering requires numpy=={NUMPY_VERSION}; found {np.__version__}"
        )
    previous_config_dir = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="drivaerml-cp-atlas-mpl-") as config_dir:
        os.environ["MPLCONFIGDIR"] = config_dir
        try:
            import matplotlib
            if matplotlib.__version__ != MATPLOTLIB_VERSION:
                raise CpAtlasError(
                    f"atlas rendering requires matplotlib=={MATPLOTLIB_VERSION}; "
                    f"found {matplotlib.__version__}"
                )
            matplotlib.use("Agg", force=True)
            from matplotlib import pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.colors import BoundaryNorm, ListedColormap
            from matplotlib.lines import Line2D
        except ImportError as error:
            raise CpAtlasError(
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
    font_sha256 = sha256_file(font_path)
    return np, plt, PdfPages, BoundaryNorm, ListedColormap, Line2D, font_sha256


def _sparse_ticks(count: int, maximum: int) -> list[int]:
    if count <= maximum:
        return list(range(count))
    step = math.ceil(count / maximum)
    ticks = list(range(0, count, step))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    return ticks


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


def _render_summary_page(pdf: Any, plt: Any, inputs: AtlasInputs, analysis: AtlasAnalysis) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _status_banner(fig)
    valid = len(inputs.case_ids) * CP_PROBE_COUNT - len(analysis.invalid_rows)
    lines = [
        "DrivAerML AutoCFD5 Cp candidate owner-review atlas",
        "",
        f"Scope: {inputs.mode}; {len(inputs.case_ids)} case(s); {CP_PROBE_COUNT} unique probes; {CP_PANEL_COUNT} panels",
        f"Rows retained: {len(inputs.case_ids) * CP_PROBE_COUNT:,} / {len(inputs.case_ids) * CP_PROBE_COUNT:,}",
        f"Support valid / invalid: {valid:,} / {len(analysis.invalid_rows):,}",
        f"Rows carrying review flags: {len(analysis.review_rows):,}",
        f"Duplicate/collapse groups: {len(analysis.duplicate_groups):,}",
        "",
        "This PDF is a deterministic visual review aid over strict hash-bound candidate receipts.",
        "It does not assert owner visual sign-off, scientific approval, or scoring activation.",
        "Invalid rows are shown explicitly; no failed case or probe is silently omitted.",
        "",
        f"Aggregate SHA-256: {inputs.aggregate_sha256}",
        f"AutoCFD5 profile SHA-256: {inputs.profile_sha256}",
    ]
    fig.text(0.08, 0.86, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.text(
        0.08,
        0.12,
        "Pages: validity/reason heatmap; displacement and native-bridge summaries; "
        "duplicate/collapse flags; then one 15-panel nominal-vs-mapped page per case.",
        va="bottom",
        fontsize=9,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _render_validity_page(
    pdf: Any, np: Any, plt: Any, BoundaryNorm: Any, ListedColormap: Any,
    inputs: AtlasInputs, analysis: AtlasAnalysis,
) -> None:
    fig, ax = plt.subplots(figsize=(17, 10))
    _status_banner(fig, "Case x probe validity and explicit failure reason")
    cmap = ListedColormap(_CATEGORY_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(_CATEGORY_ORDER) + 0.5), len(_CATEGORY_ORDER))
    image = ax.imshow(np.asarray(analysis.category_matrix), aspect="auto", cmap=cmap, norm=norm)
    case_index = {case_id: index for index, case_id in enumerate(analysis.case_ids)}
    probe_index = {
        probe_id: index for index, probe_id in enumerate(analysis.probe_ids)
    }
    if analysis.review_rows:
        review_x = [
            probe_index[int(row["autocfd_probe_id"])] for row in analysis.review_rows
        ]
        review_y = [case_index[str(row["case_id"])] for row in analysis.review_rows]
        ax.scatter(
            review_x,
            review_y,
            marker="o",
            s=5,
            facecolors="none",
            edgecolors="#ffbf00",
            linewidths=0.4,
        )
    xticks = _sparse_ticks(len(analysis.probe_ids), 24)
    yticks = _sparse_ticks(len(analysis.case_ids), 28)
    ax.set_xticks(xticks, [str(analysis.probe_ids[index]) for index in xticks], rotation=90)
    ax.set_yticks(yticks, [analysis.case_ids[index] for index in yticks])
    ax.set_xlabel(
        "AutoCFD5 unique probe ID (all 209 columns retained); amber ring = automated review flag"
    )
    ax.set_ylabel("DrivAerML case (all supplied cases retained)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_ticks(range(len(_CATEGORY_ORDER)), labels=_CATEGORY_ORDER)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def _distance_array(np: Any, values: Sequence[Sequence[float | None]]) -> Any:
    return np.asarray(
        [[np.nan if item is None else 1000.0 * item for item in row] for row in values],
        dtype=float,
    )


def _render_distance_page(pdf: Any, np: Any, plt: Any, inputs: AtlasInputs, analysis: AtlasAnalysis) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(17, 10), sharex=True)
    _status_banner(fig, "All-row nominal displacement and native bridge distance [mm]")
    arrays = (
        (
            _distance_array(np, analysis.displacement_matrix_m),
            1000.0 * NOMINAL_PROBE_DISPLACEMENT_MAX_M,
            "Nominal-to-STL displacement [mm]; candidate gate shown by upper color limit",
        ),
        (
            _distance_array(np, analysis.bridge_matrix_m),
            1000.0 * NATIVE_BRIDGE_MAX_DISTANCE_M,
            "STL-to-native bridge distance [mm]; candidate gate shown by upper color limit",
        ),
    )
    for ax, (values, vmax, title) in zip(axes, arrays, strict=True):
        cmap = plt.get_cmap("viridis").with_extremes(bad="#d9d9d9")
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        outlier_y, outlier_x = np.where(values > vmax)
        if len(outlier_x):
            ax.scatter(
                outlier_x,
                outlier_y,
                marker="x",
                s=4,
                color="#dd1c77",
                linewidths=0.35,
            )
        yticks = _sparse_ticks(len(inputs.case_ids), 22)
        ax.set_yticks(yticks, [inputs.case_ids[index] for index in yticks])
        ax.set_ylabel("case")
        ax.set_title(title + "; magenta x = above displayed gate (exact value retained)")
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    xticks = _sparse_ticks(len(analysis.probe_ids), 24)
    axes[-1].set_xticks(xticks, [str(analysis.probe_ids[index]) for index in xticks], rotation=90)
    axes[-1].set_xlabel("AutoCFD5 unique probe ID (gray means unavailable, not omitted)")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def _render_duplicate_page(
    pdf: Any, np: Any, plt: Any, BoundaryNorm: Any, ListedColormap: Any,
    inputs: AtlasInputs, analysis: AtlasAnalysis,
) -> None:
    fig, ax = plt.subplots(figsize=(17, 10))
    _status_banner(fig, "Distinct probes selecting the same raw STL/native cell IDs")
    cmap = ListedColormap(("#f7f7f7", "#9ecae1", "#a1d99b", "#756bb1"))
    norm = BoundaryNorm(np.arange(-0.5, 4.5), 4)
    image = ax.imshow(np.asarray(analysis.duplicate_matrix), aspect="auto", cmap=cmap, norm=norm)
    adjacent_y, adjacent_x = np.where(np.asarray(analysis.adjacent_duplicate_matrix))
    if len(adjacent_x):
        ax.scatter(adjacent_x, adjacent_y, marker="x", s=4, color="#de2d26", linewidths=0.35)
    xticks = _sparse_ticks(len(analysis.probe_ids), 24)
    yticks = _sparse_ticks(len(inputs.case_ids), 28)
    ax.set_xticks(xticks, [str(analysis.probe_ids[index]) for index in xticks], rotation=90)
    ax.set_yticks(yticks, [inputs.case_ids[index] for index in yticks])
    ax.set_xlabel("AutoCFD5 unique probe ID; red x = panel-adjacent collapse")
    ax.set_ylabel("DrivAerML case")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_ticks(
        (0, 1, 2, 3),
        labels=("none", "same STL ID", "same native ID", "both"),
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def _panel_axes(memberships: Sequence[Any]) -> tuple[int, int]:
    ranges = []
    for axis in range(3):
        coordinates = [float(row.point_m[axis]) for row in memberships]
        ranges.append(max(coordinates) - min(coordinates))
    selected = sorted(range(3), key=lambda axis: (-ranges[axis], axis))[:2]
    return tuple(sorted(selected))  # type: ignore[return-value]


def _render_case_page(
    pdf: Any, plt: Any, Line2D: Any, inputs: AtlasInputs, analysis: AtlasAnalysis,
    case_index: int,
) -> None:
    case_id = inputs.case_ids[case_index]
    rows = inputs.rows_by_case[case_id]
    row_by_probe = {int(row["autocfd_probe_id"]): row for row in rows}
    duplicate_by_probe = {
        probe_id: analysis.duplicate_matrix[case_index][position]
        for position, probe_id in enumerate(analysis.probe_ids)
    }
    panels: dict[str, list[Any]] = collections.defaultdict(list)
    for membership in inputs.definition.cp_panel_memberships:
        panels[membership.panel_id].append(membership)
    fig, axes = plt.subplots(5, 3, figsize=(17, 11))
    summary = analysis.case_summaries[case_index]
    _status_banner(
        fig,
        f"{case_id}: 15 panels; valid {summary['support_valid_count']}/{CP_PROBE_COUNT}; "
        f"review rows {sum(summary['review_flag_counts'].values())}; "
        f"duplicate groups {summary['duplicate_group_count']}",
    )
    axis_names = ("x", "y", "z")
    for ax, (panel_id, memberships) in zip(axes.flat, panels.items(), strict=True):
        first_axis, second_axis = _panel_axes(memberships)
        nominal_x = [row.point_m[first_axis] for row in memberships]
        nominal_y = [row.point_m[second_axis] for row in memberships]
        ax.plot(nominal_x, nominal_y, "--o", color="#636363", markersize=2.2, linewidth=0.7,
                markerfacecolor="none", label="nominal")
        stl_x: list[float] = []
        stl_y: list[float] = []
        native_x: list[float] = []
        native_y: list[float] = []
        for membership in memberships:
            row = row_by_probe[membership.autocfd_probe_id]
            stl_point = row["mapped_stl_point_m"]
            native_point = row["native_closest_point_m"]
            stl_x.append(math.nan if stl_point is None else float(stl_point[first_axis]))
            stl_y.append(math.nan if stl_point is None else float(stl_point[second_axis]))
            native_x.append(math.nan if native_point is None else float(native_point[first_axis]))
            native_y.append(math.nan if native_point is None else float(native_point[second_axis]))
        ax.plot(stl_x, stl_y, "-o", color="#2171b5", markersize=2.0, linewidth=0.65, label="mapped STL")
        ax.plot(native_x, native_y, "+", color="#238b45", markersize=3.0, markeredgewidth=0.7,
                label="mapped native")
        for membership, x_nominal, y_nominal in zip(
            memberships, nominal_x, nominal_y, strict=True
        ):
            probe_id = membership.autocfd_probe_id
            row = row_by_probe[probe_id]
            labels: list[str] = []
            if not row["support_valid"]:
                reason = row["mapping_reason"] or row["truth_reason"]
                labels.append(f"I:{_SHORT_REASON.get(str(reason), str(reason))}")
                ax.plot(x_nominal, y_nominal, "x", color="#cb181d", markersize=5, markeredgewidth=1.0)
            if row["review_flags"]:
                labels.append("R:" + ",".join(str(flag) for flag in row["review_flags"]))
                ax.plot(x_nominal, y_nominal, "s", markerfacecolor="none", markeredgecolor="#f16913", markersize=5)
            if duplicate_by_probe[probe_id]:
                labels.append(f"D:{duplicate_by_probe[probe_id]}")
                ax.plot(x_nominal, y_nominal, "d", markerfacecolor="none", markeredgecolor="#6a51a3", markersize=4)
            if labels:
                ax.annotate(
                    f"{probe_id} " + " ".join(labels),
                    (x_nominal, y_nominal),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=4.2,
                    color="#7f0000" if not row["support_valid"] else "#7f2704",
                )
        ax.set_title(panel_id.replace("_", " "), fontsize=7)
        ax.set_xlabel(f"{axis_names[first_axis]} [m]", fontsize=6)
        ax.set_ylabel(f"{axis_names[second_axis]} [m]", fontsize=6)
        ax.grid(True, linewidth=0.25, color="#d9d9d9")
        ax.tick_params(labelsize=5)
    handles = (
        Line2D([], [], linestyle="--", marker="o", color="#636363", markerfacecolor="none", label="nominal"),
        Line2D([], [], linestyle="-", marker="o", color="#2171b5", label="mapped STL"),
        Line2D([], [], linestyle="none", marker="+", color="#238b45", label="mapped native"),
        Line2D([], [], linestyle="none", marker="x", color="#cb181d", label="I invalid"),
        Line2D([], [], linestyle="none", marker="s", markerfacecolor="none", markeredgecolor="#f16913", label="R review flag"),
        Line2D([], [], linestyle="none", marker="d", markerfacecolor="none", markeredgecolor="#6a51a3", label="D duplicate ID bit: STL=1, native=2"),
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0.025, 0.05, 0.975, 0.95), h_pad=1.0, w_pad=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def render_atlas_pdf(
    *, output_path: Path, inputs: AtlasInputs, analysis: AtlasAnalysis
) -> Mapping[str, Any]:
    """Render an atomic deterministic PDF and return path-free renderer evidence."""

    np, plt, PdfPages, BoundaryNorm, ListedColormap, Line2D, font_sha256 = _load_renderer()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp.pdf", dir=output_path.parent
    )
    os.close(descriptor)
    metadata = {
        "Title": "DrivAerML AutoCFD5 Cp candidate owner-review atlas",
        "Author": "FluidsBench candidate evaluator",
        "Subject": ATLAS_STATUS,
        "Keywords": "DrivAerML AutoCFD5 Cp candidate owner review",
        "Creator": ATLAS_ALGORITHM_ID,
        "Producer": f"matplotlib {MATPLOTLIB_VERSION}",
        "CreationDate": FIXED_PDF_TIMESTAMP,
        "ModDate": FIXED_PDF_TIMESTAMP,
    }
    try:
        with PdfPages(temporary_name, metadata=metadata) as pdf:
            _render_summary_page(pdf, plt, inputs, analysis)
            _render_validity_page(
                pdf, np, plt, BoundaryNorm, ListedColormap, inputs, analysis
            )
            _render_distance_page(pdf, np, plt, inputs, analysis)
            _render_duplicate_page(
                pdf, np, plt, BoundaryNorm, ListedColormap, inputs, analysis
            )
            for case_index in range(len(inputs.case_ids)):
                _render_case_page(pdf, plt, Line2D, inputs, analysis, case_index)
        with Path(temporary_name).open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    page_count = 4 + len(inputs.case_ids)
    return {
        "matplotlib": MATPLOTLIB_VERSION,
        "numpy": NUMPY_VERSION,
        "bundled_dejavu_sans_sha256": font_sha256,
        "page_count": page_count,
        "pdf_sha256": sha256_file(output_path),
        "pdf_size_bytes": _file_size(output_path, "atlas PDF"),
    }


def build_manifest(
    *, inputs: AtlasInputs, analysis: AtlasAnalysis, renderer: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the compact, path-free, hashable atlas manifest/summary."""

    displacement_values = (
        value
        for row in analysis.displacement_matrix_m
        for value in row
        if value is not None
    )
    bridge_values = (
        value
        for row in analysis.bridge_matrix_m
        for value in row
        if value is not None
    )
    owner_statuses: collections.Counter[str] = collections.Counter()
    for case_id in inputs.case_ids:
        owner_statuses.update(
            str(row["owner_review_status"]) for row in inputs.rows_by_case[case_id]
        )
    manifest: dict[str, Any] = {
        "schema": ATLAS_SCHEMA,
        "mode": inputs.mode,
        "status": ATLAS_STATUS,
        "owner_visual_signoff_claimed": False,
        "scientific_approval_claimed": False,
        "activation_status": ATLAS_ACTIVATION_STATUS,
        "scope": {
            "case_count": len(inputs.case_ids),
            "case_ids": list(inputs.case_ids),
            "unique_probe_count": CP_PROBE_COUNT,
            "panel_count": CP_PANEL_COUNT,
            "expected_row_count": len(inputs.case_ids) * CP_PROBE_COUNT,
            "retained_row_count": sum(len(rows) for rows in inputs.rows_by_case.values()),
        },
        "source": {
            "aggregate": {
                "schema": AGGREGATE_SCHEMA,
                "sha256": inputs.aggregate_sha256,
                "size_bytes": inputs.aggregate_size_bytes,
            },
            "autocfd5": {
                "profile_sha256": inputs.profile_sha256,
                "registry_sha256": dict(inputs.registry_sha256),
            },
            "receipts": list(inputs.receipt_evidence),
            "source_geometry_rendered": False,
        },
        "algorithm": {
            "id": ATLAS_ALGORITHM_ID,
            "case_order": "increasing_run_number",
            "probe_order": "exact_autocfd5_v8_unique_probe_registry_order",
            "panel_order": "exact_autocfd5_v8_panel_membership_registry_order",
            "invalid_row_policy": "retain_and_label_never_omit",
            "geometry_policy": "receipt_coordinates_only_no_source_mesh_geometry_rendered",
            "duplicate_policy": "group_distinct_probe_ids_by_equal_raw_STL_or_native_VTK_cell_ID",
            "status_categories": list(_CATEGORY_ORDER),
            "automated_valid_semantics": "mapping_and_truth_valid_but_still_pending_owner_review",
            "review_overlay_policy": "review_flags_overlay_valid_or_invalid_status_without_changing_validity",
            "pdf_timestamp_utc": "2000-01-01T00:00:00Z",
        },
        "dependencies": {
            "numpy": renderer["numpy"],
            "matplotlib": renderer["matplotlib"],
            "bundled_dejavu_sans_sha256": renderer["bundled_dejavu_sans_sha256"],
        },
        "artifact": {
            "media_type": "application/pdf",
            "sha256": renderer["pdf_sha256"],
            "size_bytes": renderer["pdf_size_bytes"],
            "page_count": renderer["page_count"],
        },
        "summary": {
            "support_valid_count": len(inputs.case_ids) * CP_PROBE_COUNT - len(analysis.invalid_rows),
            "support_invalid_count": len(analysis.invalid_rows),
            "review_row_count": len(analysis.review_rows),
            "owner_review_status_counts": dict(sorted(owner_statuses.items())),
            "nominal_displacement_m": _stats(displacement_values),
            "native_bridge_distance_m": _stats(bridge_values),
            "duplicate_group_count": len(analysis.duplicate_groups),
            "panel_adjacent_duplicate_group_count": sum(
                bool(group["adjacent_pairs"]) for group in analysis.duplicate_groups
            ),
        },
        "invalid_rows": list(analysis.invalid_rows),
        "review_rows": list(analysis.review_rows),
        "duplicate_groups": list(analysis.duplicate_groups),
        "per_case": list(analysis.case_summaries),
        "per_probe": list(analysis.probe_summaries),
    }
    if manifest["scope"]["retained_row_count"] != manifest["scope"]["expected_row_count"]:
        raise CpAtlasError("manifest would omit one or more validated Cp rows")
    _assert_no_absolute_paths(manifest, "atlas manifest")
    return manifest


def _refuse_output_collisions(
    *, aggregate_path: Path, profile_path: Path, receipt_paths: Sequence[Path],
    output_pdf: Path, output_manifest: Path,
) -> None:
    inputs = [Path(aggregate_path), Path(profile_path), *(Path(path) for path in receipt_paths)]
    outputs = [Path(output_pdf), Path(output_manifest)]
    if outputs[0].resolve() == outputs[1].resolve():
        raise CpAtlasError("atlas PDF and manifest outputs must differ")
    input_paths = {path.resolve() for path in inputs}
    for output in outputs:
        if output.resolve() in input_paths:
            raise CpAtlasError("atlas output must not overwrite a validated input")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_AUTOCFD5_PROFILE)
    parser.add_argument(
        "--receipt", type=Path, action="append", required=True,
        help="strict per-case JSON receipt; repeat exactly once per selected case",
    )
    parser.add_argument(
        "--pilot-case", action="append", default=None,
        help="explicit official case ID; repeat for an incomplete pilot such as run_1/run_44",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _refuse_output_collisions(
        aggregate_path=args.aggregate,
        profile_path=args.profile,
        receipt_paths=args.receipt,
        output_pdf=args.output_pdf,
        output_manifest=args.output_manifest,
    )
    inputs = load_atlas_inputs(
        aggregate_path=args.aggregate,
        profile_path=args.profile,
        receipt_paths=args.receipt,
        pilot_case_ids=args.pilot_case,
    )
    analysis = analyse_atlas(inputs)
    renderer = render_atlas_pdf(
        output_path=args.output_pdf, inputs=inputs, analysis=analysis
    )
    manifest = build_manifest(inputs=inputs, analysis=analysis, renderer=renderer)
    write_evidence(args.output_manifest, manifest)
    print(
        json.dumps(
            {
                "status": ATLAS_STATUS,
                "pdf": str(args.output_pdf),
                "manifest": str(args.output_manifest),
                "pdf_sha256": renderer["pdf_sha256"],
                "manifest_sha256": hashlib.sha256(
                    _canonical_json_bytes(manifest)
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
