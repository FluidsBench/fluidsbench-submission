#!/usr/bin/env python3
"""Build one candidate DrivAerML AutoCFD5 Cp mapping/support receipt.

This maintainer CLI is deliberately case-local.  It verifies the SHA-bound v8
registries, maps all 209 nominal taps through the named case STL to native VTP
polygons, and gathers truth from native ``CellData pMeanTrim``.  Its outputs
are candidate evidence for aggregation and visual review; they do not assert
owner sign-off or activate scoring support.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaerml.autocfd5 import (  # noqa: E402
    CP_PROBE_COUNT,
    NATIVE_BRIDGE_MAX_DISTANCE_M,
    NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT,
    NOMINAL_PROBE_DISPLACEMENT_MAX_M,
    U_INF_M_PER_S,
    load_autocfd5_definition,
    validate_cp_mapping_evidence,
)
from reference.drivaerml.cp_mapping import (  # noqa: E402
    CUT_VERTEX_PLANE_TOLERANCE_M,
    NATIVE_BRIDGE_DISTANCE_REVIEW_M,
    NATIVE_DISTANCE_TIE_M,
    NATIVE_NORMAL_TIE,
    NOMINAL_PROBE_DISPLACEMENT_REVIEW_M,
    REQUIRED_VTK_VERSION,
    STL_DISTANCE_TIE_M,
    UNNORMALIZED_AREA_VECTOR_MIN_M2,
    AsciiStlTriangleStream,
    CpMappingError,
    CpMappingRecord,
    NativePolygonLocator,
    RetainedVerifiedRegularFile,
    Vtk95VtpPolygonLocator,
    map_cp_probes_for_case,
    sha256_file,
)


CASE_ID_RE = re.compile(r"run_[1-9][0-9]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EVIDENCE_SCHEMA = "drivaerml-autocfd5-cp-case-support-candidate-v1"
CSV_SCHEMA = "drivaerml-autocfd5-cp-case-mapping-candidate-v1"
ALGORITHM_ID = "drivaerml-autocfd5-cp-geometric-mapping-candidate-v1"
EXPECTED_AUTOCFD5_PROFILE_SHA256 = (
    "17d830087d11e83e3cba75358f33fdd827421be6698ba1624e547ae36f359184"
)


class CpCaseSupportError(ValueError):
    """Raised when a case support replay cannot produce complete evidence."""


@dataclass(frozen=True)
class StlInventory:
    sha256: str
    size_bytes: int
    facet_count: int
    solid_facet_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BoundaryPressureSelection:
    vtk_version: str
    point_count: int
    polygon_count: int
    tuple_count: int
    component_count: int
    vtk_data_type: str
    selected_values: tuple[tuple[int, float], ...]

    def values_by_raw_id(self) -> dict[int, float]:
        return dict(self.selected_values)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha(value: str | None, label: str) -> str | None:
    if value is not None and SHA256_RE.fullmatch(value) is None:
        raise CpCaseSupportError(f"{label} must be a lowercase SHA-256")
    return value


def inventory_ascii_stl(path: Path, *, chunk_facets: int) -> StlInventory:
    """Hash and count one strict named ASCII STL in bounded memory."""

    stream = AsciiStlTriangleStream(path, chunk_facets=chunk_facets)
    counts: Counter[str] = Counter()
    for chunk in stream:
        counts[chunk.solid_name] += len(chunk.raw_facet_ids)
    assert stream.sha256 is not None and stream.facet_count is not None
    if sum(counts.values()) != stream.facet_count:
        raise CpCaseSupportError("STL facet inventory did not close")
    return StlInventory(
        sha256=stream.sha256,
        size_bytes=path.stat().st_size,
        facet_count=stream.facet_count,
        solid_facet_counts=tuple(sorted(counts.items())),
    )


def read_boundary_pressure_selection(
    path: Path, raw_polygon_ids: Sequence[int]
) -> BoundaryPressureSelection:
    """Read selected native ``CellData pMeanTrim`` tuples using exact VTK."""

    try:
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkCommonCore import vtkVersion
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    except ImportError as error:  # pragma: no cover - optional evaluator dependency
        raise CpCaseSupportError(
            f"Cp support replay requires optional VTK {REQUIRED_VTK_VERSION}"
        ) from error
    vtk_version = vtkVersion.GetVTKVersion()
    if vtk_version != REQUIRED_VTK_VERSION:  # pragma: no cover - version-specific
        raise CpCaseSupportError(
            f"Cp support replay requires VTK {REQUIRED_VTK_VERSION}, found {vtk_version}"
        )

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    point_selection = reader.GetPointDataArraySelection()
    cell_selection = reader.GetCellDataArraySelection()
    for selection in (point_selection, cell_selection):
        for index in range(selection.GetNumberOfArrays()):
            selection.DisableArray(selection.GetArrayName(index))
    available_cell = {
        cell_selection.GetArrayName(index)
        for index in range(cell_selection.GetNumberOfArrays())
    }
    if "pMeanTrim" not in available_cell:
        raise CpCaseSupportError("boundary VTP is missing CellData pMeanTrim")
    cell_selection.EnableArray("pMeanTrim")
    reader.Update()
    if reader.GetErrorCode():
        raise CpCaseSupportError(
            f"VTK XML boundary reader failed with code {reader.GetErrorCode()}"
        )
    polydata = reader.GetOutput()
    polygon_count = int(polydata.GetNumberOfPolys())
    if (
        polygon_count < 1
        or int(polydata.GetNumberOfCells()) != polygon_count
        or polydata.GetNumberOfVerts()
        or polydata.GetNumberOfLines()
        or polydata.GetNumberOfStrips()
    ):
        raise CpCaseSupportError(
            "boundary VTP must contain native polygons only so raw IDs are stable"
        )
    array = polydata.GetCellData().GetArray("pMeanTrim")
    if array is None:
        raise CpCaseSupportError("VTK did not load CellData pMeanTrim")
    tuple_count = int(array.GetNumberOfTuples())
    component_count = int(array.GetNumberOfComponents())
    if tuple_count != polygon_count or component_count != 1:
        raise CpCaseSupportError(
            "pMeanTrim must contain one scalar tuple per native polygon"
        )
    ids = tuple(sorted(set(int(value) for value in raw_polygon_ids)))
    if any(value < 0 or value >= tuple_count for value in ids):
        raise CpCaseSupportError("a mapped raw polygon ID is outside pMeanTrim support")
    values = np.asarray(vtk_to_numpy(array))
    selected = tuple((raw_id, float(values[raw_id])) for raw_id in ids)
    return BoundaryPressureSelection(
        vtk_version=vtk_version,
        point_count=int(polydata.GetNumberOfPoints()),
        polygon_count=polygon_count,
        tuple_count=tuple_count,
        component_count=component_count,
        vtk_data_type=str(array.GetDataTypeAsString()),
        selected_values=selected,
    )


def _optional_float(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _point_columns(prefix: str, value: tuple[float, float, float] | None) -> dict[str, str]:
    if value is None:
        return {f"{prefix}_{axis}_m": "" for axis in "xyz"}
    return {
        f"{prefix}_{axis}_m": repr(float(coordinate))
        for axis, coordinate in zip("xyz", value)
    }


def _vector_columns(prefix: str, value: tuple[float, float, float] | None) -> dict[str, str]:
    if value is None:
        return {f"{prefix}_{axis}": "" for axis in "xyz"}
    return {
        f"{prefix}_{axis}": repr(float(coordinate))
        for axis, coordinate in zip("xyz", value)
    }


def _json_point(value: tuple[float, float, float] | None) -> list[float] | None:
    return list(value) if value is not None else None


def _row_payload(
    record: CpMappingRecord,
    pressure: float | None,
    truth_cp: float | None,
    *,
    truth_valid: bool,
    truth_reason: str,
) -> dict[str, object]:
    return {
        "case_id": record.case_id,
        "autocfd_probe_id": record.autocfd_probe_id,
        "nominal_point_m": list(record.nominal_point_m),
        "drivaerml_component": record.drivaerml_component,
        "projection_mode": record.projection_mode,
        "cut_axis": record.cut_axis,
        "cut_value_m": record.cut_value_m,
        "owner_review_status": record.owner_review_status,
        "mapping_valid": record.valid,
        "mapping_reason": record.reason,
        "mapped_stl_point_m": _json_point(record.mapped_point_m),
        "raw_stl_triangle_id": record.raw_stl_triangle_id,
        "stl_unit_normal": _json_point(record.stl_unit_normal),
        "nominal_displacement_m": record.nominal_displacement_m,
        "native_closest_point_m": _json_point(record.native_closest_point_m),
        "raw_vtk_polygon_id": record.raw_vtk_polygon_id,
        "native_polygon_unit_normal": _json_point(record.native_polygon_unit_normal),
        "bridge_distance_m": record.bridge_distance_m,
        "bridge_abs_normal_dot": record.bridge_abs_normal_dot,
        "component_facet_count": record.component_facet_count,
        "projection_candidate_count": record.projection_candidate_count,
        "native_bounds_candidate_count": record.native_bounds_candidate_count,
        "native_distance_pass_count": record.native_distance_pass_count,
        "native_normal_pass_count": record.native_normal_pass_count,
        "review_flags": list(record.review_flags),
        "truth_valid": truth_valid,
        "truth_reason": truth_reason,
        "pMeanTrim_m2_per_s2": pressure,
        "truth_Cp": truth_cp,
        "support_valid": record.valid and truth_valid,
    }


CSV_FIELDS = (
    "schema",
    "case_id",
    "autocfd_probe_id",
    "nominal_x_m",
    "nominal_y_m",
    "nominal_z_m",
    "drivaerml_component",
    "projection_mode",
    "cut_axis",
    "cut_value_m",
    "owner_review_status",
    "mapping_valid",
    "mapping_reason",
    "mapped_stl_x_m",
    "mapped_stl_y_m",
    "mapped_stl_z_m",
    "raw_stl_triangle_id",
    "stl_normal_x",
    "stl_normal_y",
    "stl_normal_z",
    "nominal_displacement_m",
    "native_closest_x_m",
    "native_closest_y_m",
    "native_closest_z_m",
    "raw_vtk_polygon_id",
    "native_normal_x",
    "native_normal_y",
    "native_normal_z",
    "bridge_distance_m",
    "bridge_abs_normal_dot",
    "component_facet_count",
    "projection_candidate_count",
    "native_bounds_candidate_count",
    "native_distance_pass_count",
    "native_normal_pass_count",
    "review_flags",
    "truth_valid",
    "truth_reason",
    "pMeanTrim_m2_per_s2",
    "truth_Cp",
    "support_valid",
)


def _csv_row(payload: dict[str, object]) -> dict[str, str]:
    nominal = tuple(payload["nominal_point_m"])  # type: ignore[arg-type]
    row = {
        "schema": CSV_SCHEMA,
        "case_id": str(payload["case_id"]),
        "autocfd_probe_id": str(payload["autocfd_probe_id"]),
        "nominal_x_m": repr(float(nominal[0])),
        "nominal_y_m": repr(float(nominal[1])),
        "nominal_z_m": repr(float(nominal[2])),
        "drivaerml_component": str(payload["drivaerml_component"]),
        "projection_mode": str(payload["projection_mode"]),
        "cut_axis": "" if payload["cut_axis"] is None else str(payload["cut_axis"]),
        "cut_value_m": _optional_float(payload["cut_value_m"]),  # type: ignore[arg-type]
        "owner_review_status": str(payload["owner_review_status"]),
        "mapping_valid": "true" if payload["mapping_valid"] else "false",
        "mapping_reason": str(payload["mapping_reason"]),
        "raw_stl_triangle_id": _optional_int(
            payload["raw_stl_triangle_id"]  # type: ignore[arg-type]
        ),
        "nominal_displacement_m": _optional_float(
            payload["nominal_displacement_m"]  # type: ignore[arg-type]
        ),
        "raw_vtk_polygon_id": _optional_int(
            payload["raw_vtk_polygon_id"]  # type: ignore[arg-type]
        ),
        "bridge_distance_m": _optional_float(
            payload["bridge_distance_m"]  # type: ignore[arg-type]
        ),
        "bridge_abs_normal_dot": _optional_float(
            payload["bridge_abs_normal_dot"]  # type: ignore[arg-type]
        ),
        "component_facet_count": str(payload["component_facet_count"]),
        "projection_candidate_count": str(payload["projection_candidate_count"]),
        "native_bounds_candidate_count": str(payload["native_bounds_candidate_count"]),
        "native_distance_pass_count": str(payload["native_distance_pass_count"]),
        "native_normal_pass_count": str(payload["native_normal_pass_count"]),
        "review_flags": ";".join(payload["review_flags"]),  # type: ignore[arg-type]
        "truth_valid": "true" if payload["truth_valid"] else "false",
        "truth_reason": str(payload["truth_reason"]),
        "pMeanTrim_m2_per_s2": _optional_float(
            payload["pMeanTrim_m2_per_s2"]  # type: ignore[arg-type]
        ),
        "truth_Cp": _optional_float(payload["truth_Cp"]),  # type: ignore[arg-type]
        "support_valid": "true" if payload["support_valid"] else "false",
    }
    row.update(
        _point_columns("mapped_stl", payload["mapped_stl_point_m"])  # type: ignore[arg-type]
    )
    row.update(_vector_columns("stl_normal", payload["stl_unit_normal"]))  # type: ignore[arg-type]
    row.update(
        _point_columns(
            "native_closest",
            payload["native_closest_point_m"],  # type: ignore[arg-type]
        )
    )
    row.update(
        _vector_columns(
            "native_normal",
            payload["native_polygon_unit_normal"],  # type: ignore[arg-type]
        )
    )
    if set(row) != set(CSV_FIELDS):
        raise CpCaseSupportError("internal Cp mapping CSV schema mismatch")
    return row


def _csv_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_csv_row(row) for row in rows)
    return stream.getvalue().encode("utf-8")


def _finite_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"minimum": min(values), "maximum": max(values)}


def _expected_source_verification() -> dict[str, object]:
    common: dict[str, object] = {
        "pathname_open": "single_open_retained_through_all_reads",
        "file_type": "regular_file_verified_by_fstat",
        "hash_input": "retained_verified_file_descriptor",
        "descriptor_transport": "verified_procfs_or_devfs_fd_alias_same_device_inode",
        "identity_fields": [
            "device",
            "inode",
            "mode",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        ],
        "post_read_fstat": "unchanged",
    }
    return {
        "stl": {
            **common,
            "parser_inputs": [
                "strict_ascii_stl_inventory",
                "component_constrained_geometric_mapping",
            ],
        },
        "boundary_vtp": {
            **common,
            "parser_inputs": [
                "vtk_polygon_locator",
                "vtk_CellData_pMeanTrim_selection",
            ],
        },
    }


def _algorithm_payload(
    *,
    stl_chunk_facets: int,
    source_verification: Mapping[str, object] | None = None,
) -> dict[str, object]:
    verification = (
        _expected_source_verification()
        if source_verification is None
        else dict(source_verification)
    )
    if verification != _expected_source_verification():
        raise CpCaseSupportError(
            "retained source verification receipt is not the exact candidate method"
        )
    return {
        "id": ALGORITHM_ID,
        "scientific_status": "candidate_not_owner_approved_not_active_scoring_support",
        "owner_visual_signoff_claimed": False,
        "source_integrity": verification,
        "stl": {
            "format": "strict_ASCII_multi_solid",
            "stream_chunk_facets": stl_chunk_facets,
            "raw_facet_identity": "zero_based_global_facet_order_across_named_solids",
            "component_match": "exact_named_solid_only_no_unrestricted_fallback",
            "projection_modes": ["component_closest_3d", "cut_plane_closest"],
            "cut_vertex_plane_tolerance_m": CUT_VERTEX_PLANE_TOLERANCE_M,
            "distance_tie_m": STL_DISTANCE_TIE_M,
            "tie_break": "smallest_raw_stl_triangle_id",
            "unnormalized_area_vector_invalid_lte_m2": UNNORMALIZED_AREA_VECTOR_MIN_M2,
            "nominal_displacement_review_m": NOMINAL_PROBE_DISPLACEMENT_REVIEW_M,
            "nominal_displacement_invalid_gt_m": NOMINAL_PROBE_DISPLACEMENT_MAX_M,
        },
        "native_vtp": {
            "association": "raw_native_VTK_CellData_polygon_order",
            "candidate_discovery": "vtkStaticCellLocator_FindCellsWithinBounds_candidate_only",
            "required_vtk_version": REQUIRED_VTK_VERSION,
            "polygon_closest_point": "ordered_first_vertex_triangle_fan_binary64",
            "polygon_normal": "normalize(sum_i(cross(v_i,v_i_plus_1)))",
            "bridge_distance_max_m": NATIVE_BRIDGE_MAX_DISTANCE_M,
            "bridge_distance_review_m": NATIVE_BRIDGE_DISTANCE_REVIEW_M,
            "bridge_abs_normal_dot_min": NATIVE_BRIDGE_MIN_ABS_NORMAL_DOT,
            "distance_tie_m": NATIVE_DISTANCE_TIE_M,
            "normal_tie": NATIVE_NORMAL_TIE,
            "tie_break": "distance_tie_then_decreasing_abs_normal_dot_then_smallest_raw_polygon_id",
        },
        "truth": {
            "source": "native_CellData_pMeanTrim_at_mapped_raw_polygon_id",
            "pMeanTrim_unit": "m2/s2",
            "equation": "Cp=2*pMeanTrim/Uinf^2",
            "Uinf_m_per_s": U_INF_M_PER_S,
            "Cp_unit": "dimensionless",
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


LocatorFactory = Callable[[Path], NativePolygonLocator]
MappingBuilder = Callable[..., tuple[CpMappingRecord, ...]]
BoundaryReader = Callable[[Path, Sequence[int]], BoundaryPressureSelection]


def _build_case_support_with_retained_sources(
    *,
    case_id: str,
    profile_path: Path,
    stl_path: Path,
    boundary_path: Path,
    retained_stl: RetainedVerifiedRegularFile,
    retained_boundary: RetainedVerifiedRegularFile,
    output_json: Path,
    output_csv: Path,
    chunk_facets: int,
    expected_stl_sha256: str | None = None,
    expected_boundary_sha256: str | None = None,
    locator_factory: LocatorFactory | None = None,
    mapping_builder: MappingBuilder | None = None,
    boundary_reader: BoundaryReader | None = None,
) -> dict[str, object]:
    """Build, validate, and atomically write one deterministic case receipt."""

    if CASE_ID_RE.fullmatch(case_id) is None:
        raise CpCaseSupportError("case_id must match run_<positive integer>")
    case_number = case_id.removeprefix("run_")
    if stl_path.name != f"drivaer_{case_number}.stl":
        raise CpCaseSupportError(
            f"case STL must be named drivaer_{case_number}.stl for {case_id}"
        )
    if boundary_path.name != f"boundary_{case_number}.vtp":
        raise CpCaseSupportError(
            f"boundary VTP must be named boundary_{case_number}.vtp for {case_id}"
        )
    if output_json.resolve() == output_csv.resolve():
        raise CpCaseSupportError("JSON and CSV output paths must differ")
    if not profile_path.is_file():
        raise CpCaseSupportError(
            "AutoCFD5 profile does not exist or is not a file"
        )
    if not isinstance(chunk_facets, int) or isinstance(chunk_facets, bool) or chunk_facets < 1:
        raise CpCaseSupportError("chunk_facets must be a positive integer")
    if locator_factory is None:
        locator_factory = Vtk95VtpPolygonLocator
    if mapping_builder is None:
        mapping_builder = map_cp_probes_for_case
    if boundary_reader is None:
        boundary_reader = read_boundary_pressure_selection
    expected_stl_sha256 = _validate_sha(expected_stl_sha256, "expected STL SHA-256")
    expected_boundary_sha256 = _validate_sha(
        expected_boundary_sha256, "expected boundary SHA-256"
    )

    profile_sha256 = sha256_file(profile_path)
    if profile_sha256 != EXPECTED_AUTOCFD5_PROFILE_SHA256:
        raise CpCaseSupportError(
            "AutoCFD5 profile does not match the exact candidate v8 SHA-256"
        )
    definition = load_autocfd5_definition(profile_path)
    if sha256_file(profile_path) != profile_sha256:
        raise CpCaseSupportError(
            "AutoCFD5 profile changed while its registries were validated"
        )
    if len(definition.cp_probes) != CP_PROBE_COUNT:
        raise CpCaseSupportError(
            f"AutoCFD5 definition must contain exactly {CP_PROBE_COUNT} probes"
        )
    if len(definition.cp_component_rules) != CP_PROBE_COUNT:
        raise CpCaseSupportError(
            f"AutoCFD5 definition must contain exactly {CP_PROBE_COUNT} rules"
        )

    stl_descriptor = retained_stl.descriptor_path()
    boundary_descriptor = retained_boundary.descriptor_path()
    stl = inventory_ascii_stl(stl_descriptor, chunk_facets=chunk_facets)
    retained_stl.assert_unchanged(context="while the STL inventory parser ran")
    if stl.size_bytes != retained_stl.snapshot.size_bytes:
        raise CpCaseSupportError(
            "STL inventory size differs from the retained pre-hash fstat snapshot"
        )
    boundary_sha256 = retained_boundary.sha256()
    if expected_stl_sha256 is not None and stl.sha256 != expected_stl_sha256:
        raise CpCaseSupportError("case STL SHA-256 does not match the expected identity")
    if (
        expected_boundary_sha256 is not None
        and boundary_sha256 != expected_boundary_sha256
    ):
        raise CpCaseSupportError(
            "boundary VTP SHA-256 does not match the expected identity"
        )
    source_sha256 = (stl.sha256, boundary_sha256)

    locator = locator_factory(boundary_descriptor)
    mappings = mapping_builder(
        case_id,
        stl_descriptor,
        locator,
        definition.cp_probes,
        definition.cp_component_rules,
        source_sha256=source_sha256,
        chunk_facets=chunk_facets,
    )
    if len(mappings) != CP_PROBE_COUNT:
        raise CpCaseSupportError(
            f"mapping producer returned {len(mappings)} rows, expected {CP_PROBE_COUNT}"
        )
    mapping_by_id: dict[int, CpMappingRecord] = {}
    for mapping in mappings:
        if mapping.case_id != case_id:
            raise CpCaseSupportError("mapping producer returned the wrong case ID")
        if mapping.source_sha256 != source_sha256:
            raise CpCaseSupportError("mapping producer returned the wrong source hashes")
        if mapping.autocfd_probe_id in mapping_by_id:
            raise CpCaseSupportError("mapping producer returned a duplicate probe")
        mapping_by_id[mapping.autocfd_probe_id] = mapping
    expected_ids = tuple(probe.autocfd_probe_id for probe in definition.cp_probes)
    if set(mapping_by_id) != set(expected_ids):
        raise CpCaseSupportError("mapping producer did not cover all 209 registry probes")
    mappings = tuple(mapping_by_id[probe_id] for probe_id in expected_ids)
    validate_cp_mapping_evidence(
        (mapping.to_evidence() for mapping in mappings),
        definition,
        expected_case_ids=(case_id,),
        require_all_valid=False,
    )

    raw_ids = tuple(
        mapping.raw_vtk_polygon_id
        for mapping in mappings
        if mapping.valid and mapping.raw_vtk_polygon_id is not None
    )
    native_polygon_count = locator.polygon_count
    del locator
    gc.collect()
    boundary = boundary_reader(boundary_descriptor, raw_ids)
    if boundary.vtk_version != REQUIRED_VTK_VERSION:
        raise CpCaseSupportError(
            f"boundary reader must use VTK {REQUIRED_VTK_VERSION}"
        )
    if boundary.polygon_count != native_polygon_count:
        raise CpCaseSupportError(
            "boundary pressure replay polygon count differs from mapping support"
        )
    if boundary.tuple_count != boundary.polygon_count or boundary.component_count != 1:
        raise CpCaseSupportError(
            "boundary pMeanTrim metadata must contain one scalar tuple per polygon"
        )
    if len(boundary.selected_values) != len(set(raw_ids)):
        raise CpCaseSupportError(
            "boundary pressure selection returned duplicate or extra raw polygon IDs"
        )
    values_by_id = boundary.values_by_raw_id()
    if set(values_by_id) != set(raw_ids):
        raise CpCaseSupportError(
            "boundary pressure selection did not cover every mapped raw polygon ID"
        )

    retained_stl.finalize_verification()
    retained_boundary.finalize_verification()
    source_verification = {
        "stl": retained_stl.verification_receipt(
            consumers=(
                "strict_ascii_stl_inventory",
                "component_constrained_geometric_mapping",
            )
        ),
        "boundary_vtp": retained_boundary.verification_receipt(
            consumers=(
                "vtk_polygon_locator",
                "vtk_CellData_pMeanTrim_selection",
            )
        ),
    }

    rows: list[dict[str, object]] = []
    truth_values: list[float] = []
    pressure_values: list[float] = []
    for mapping in mappings:
        pressure: float | None = None
        truth_cp: float | None = None
        truth_valid = False
        truth_reason = "mapping_invalid"
        if mapping.valid:
            assert mapping.raw_vtk_polygon_id is not None
            native_pressure = values_by_id[mapping.raw_vtk_polygon_id]
            if math.isfinite(native_pressure):
                pressure = native_pressure
                truth_cp = 2.0 * native_pressure / (U_INF_M_PER_S * U_INF_M_PER_S)
                truth_valid = math.isfinite(truth_cp)
                truth_reason = "" if truth_valid else "nonfinite_derived_Cp"
            else:
                truth_reason = "nonfinite_native_pMeanTrim"
            if truth_valid:
                assert truth_cp is not None
                pressure_values.append(pressure)
                truth_values.append(truth_cp)
        rows.append(
            _row_payload(
                mapping,
                pressure,
                truth_cp,
                truth_valid=truth_valid,
                truth_reason=truth_reason,
            )
        )
    if len(rows) != CP_PROBE_COUNT:
        raise CpCaseSupportError("internal row construction omitted a Cp probe")

    csv_payload = _csv_bytes(rows)
    mapping_invalid_reasons = Counter(
        str(row["mapping_reason"]) for row in rows if not row["mapping_valid"]
    )
    truth_invalid_reasons = Counter(
        str(row["truth_reason"]) for row in rows if not row["truth_valid"]
    )
    review_flags = Counter(
        str(flag) for row in rows for flag in row["review_flags"]  # type: ignore[union-attr]
    )
    owner_status = Counter(str(row["owner_review_status"]) for row in rows)
    evidence: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": "1.0",
        "case_id": case_id,
        "status": "candidate_not_owner_approved_not_active_scoring_support",
        "owner_visual_signoff_claimed": False,
        "definition": {
            "profile_sha256": profile_sha256,
            "registry_sha256": {
                key: value for key, value in definition.source_sha256
            },
            "probe_count": len(definition.cp_probes),
            "rule_count": len(definition.cp_component_rules),
        },
        "sources": {
            "stl": {
                "file": stl_path.name,
                "sha256": stl.sha256,
                "size_bytes": stl.size_bytes,
                "solid_count": len(stl.solid_facet_counts),
                "facet_count": stl.facet_count,
                "solid_facet_counts": {
                    key: value for key, value in stl.solid_facet_counts
                },
            },
            "boundary": {
                "file": boundary_path.name,
                "sha256": boundary_sha256,
                "size_bytes": retained_boundary.snapshot.size_bytes,
                "point_count": boundary.point_count,
                "polygon_count": boundary.polygon_count,
                "pMeanTrim_tuple_count": boundary.tuple_count,
                "pMeanTrim_component_count": boundary.component_count,
                "pMeanTrim_vtk_data_type": boundary.vtk_data_type,
            },
        },
        "dependencies": {
            "vtk": boundary.vtk_version,
            "numpy": np.__version__,
        },
        "algorithm": _algorithm_payload(
            stl_chunk_facets=chunk_facets,
            source_verification=source_verification,
        ),
        "summary": {
            "row_count": len(rows),
            "mapping_valid_count": sum(bool(row["mapping_valid"]) for row in rows),
            "mapping_invalid_count": sum(not bool(row["mapping_valid"]) for row in rows),
            "mapping_invalid_reason_counts": dict(sorted(mapping_invalid_reasons.items())),
            "truth_valid_count": sum(bool(row["truth_valid"]) for row in rows),
            "truth_invalid_count": sum(not bool(row["truth_valid"]) for row in rows),
            "truth_invalid_reason_counts": dict(sorted(truth_invalid_reasons.items())),
            "support_valid_count": sum(bool(row["support_valid"]) for row in rows),
            "owner_review_status_counts": dict(sorted(owner_status.items())),
            "review_flag_counts": dict(sorted(review_flags.items())),
            "pMeanTrim_range_m2_per_s2": _finite_summary(pressure_values),
            "truth_Cp_range": _finite_summary(truth_values),
        },
        "artifacts": {
            "mapping_csv_schema": CSV_SCHEMA,
            "mapping_csv_row_count": len(rows),
            "mapping_csv_sha256": _sha256_bytes(csv_payload),
        },
        "rows": rows,
    }
    json_payload = (
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(output_csv, csv_payload)
    _atomic_write(output_json, json_payload)
    return evidence


def build_case_support(
    *,
    case_id: str,
    profile_path: Path,
    stl_path: Path,
    boundary_path: Path,
    output_json: Path,
    output_csv: Path,
    chunk_facets: int,
    expected_stl_sha256: str | None = None,
    expected_boundary_sha256: str | None = None,
    locator_factory: LocatorFactory | None = None,
    mapping_builder: MappingBuilder | None = None,
    boundary_reader: BoundaryReader | None = None,
) -> dict[str, object]:
    """Open each geometry pathname once and retain both inodes to completion."""

    try:
        with RetainedVerifiedRegularFile.open(
            stl_path, label="case STL"
        ) as retained_stl, RetainedVerifiedRegularFile.open(
            boundary_path, label="boundary VTP"
        ) as retained_boundary:
            return _build_case_support_with_retained_sources(
                case_id=case_id,
                profile_path=profile_path,
                stl_path=stl_path,
                boundary_path=boundary_path,
                retained_stl=retained_stl,
                retained_boundary=retained_boundary,
                output_json=output_json,
                output_csv=output_csv,
                chunk_facets=chunk_facets,
                expected_stl_sha256=expected_stl_sha256,
                expected_boundary_sha256=expected_boundary_sha256,
                locator_factory=locator_factory,
                mapping_builder=mapping_builder,
                boundary_reader=boundary_reader,
            )
    except CpMappingError as error:
        raise CpCaseSupportError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--case-stl", type=Path, required=True)
    parser.add_argument("--boundary-vtp", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--chunk-facets", type=int, default=16_384)
    parser.add_argument("--expected-stl-sha256")
    parser.add_argument("--expected-boundary-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = build_case_support(
        case_id=args.case_id,
        profile_path=args.profile,
        stl_path=args.case_stl,
        boundary_path=args.boundary_vtp,
        output_json=args.output_json,
        output_csv=args.output_csv,
        chunk_facets=args.chunk_facets,
        expected_stl_sha256=args.expected_stl_sha256,
        expected_boundary_sha256=args.expected_boundary_sha256,
    )
    print(
        json.dumps(
            {
                "case_id": evidence["case_id"],
                "status": evidence["status"],
                "summary": evidence["summary"],
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
