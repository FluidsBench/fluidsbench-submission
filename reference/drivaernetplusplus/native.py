"""Read native DrivAerNet++ surface files and derive the frozen scoring support.

Everything here runs on the maintainer side when the candidate support is
published.  Participants and the evaluator consume only the frozen arrays it
produces (dual areas, reference pressure, profile point indices), so scoring
never depends on a VTK reader or filter version.

Rules (versioned by ``SUPPORT_RULES_VERSION``):

* native order       -- the PointData order of ``SurfacePressureVTK`` files;
* reference pressure -- PointData ``p`` exactly as stored (float32, m^2/s^2);
* dual-area weights  -- every polygon's area (fan triangulation from its first
  vertex) is split equally among its vertices.  This is the agreed rule of
  ``benchmark-specs/drivaernetplusplus/tools/publish_scoring_support.py``,
  vectorized; the two implementations agree to floating-point round-off;
* profile stations   -- see ``extract_profile_stations``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORT_RULES_VERSION = "drivaernetplusplus-native-support-rules-v1"
PRESSURE_ARRAY = "p"
DYNAMIC_HEAD = 450.0  # 0.5 * U_inf^2 at U_inf = 30 m/s, kinematic (m^2/s^2)

# Profile-station rule constants.  Frozen with SUPPORT_RULES_VERSION.
CENTRELINE_BAND = 2e-3  # |y| <= CENTRELINE_BAND * L
NORMAL_Z_THRESHOLD = 0.2
NORMAL_X_THRESHOLD = 0.5
ENVELOPE_BIN = 5e-3  # bin width in units of L along the station coordinate
REAR_REGION = 0.9  # rear-base points must satisfy (x - x_min) / L >= REAR_REGION
MINIMUM_STATION_POINTS = 20
MINIMUM_COMPONENTS = 5  # body plus four wheels; tiny fragments may add more
FRAGMENT_POINTS = 100  # smaller components are mesh slivers (scored, never profiled)
PROFILE_STATIONS = (
    "upper_body_centerline",
    "underbody_centerline",
    "rear_body_centerline",
)


class NativeSupportError(ValueError):
    """Raised when a public file cannot produce the agreed support."""


@dataclass(frozen=True)
class NativeSurface:
    points: np.ndarray  # (N, 3) float32, as stored
    offsets: np.ndarray  # (C + 1,) int64 polygon offsets
    connectivity: np.ndarray  # (offsets[-1],) int64 point indices
    pressure: np.ndarray  # (N,) float32, as stored

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    @property
    def cell_count(self) -> int:
        return int(len(self.offsets) - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Digest of the little-endian C-order bytes of an array."""

    value = np.ascontiguousarray(array)
    if value.dtype.byteorder == ">":
        value = value.astype(value.dtype.newbyteorder("<"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _vtk_array(array) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy

    return np.array(vtk_to_numpy(array), copy=True)


def read_native_surface(path: Path) -> NativeSurface:
    """Read one legacy-VTK PolyData pressure file without reordering anything."""

    import vtk

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    if reader.GetErrorCode() != 0:
        raise NativeSupportError(f"{path}: VTK reader failed")
    mesh = reader.GetOutput()
    if mesh is None or mesh.GetNumberOfPoints() <= 0:
        raise NativeSupportError(f"{path}: no points")
    if mesh.GetNumberOfVerts() or mesh.GetNumberOfLines() or mesh.GetNumberOfStrips():
        raise NativeSupportError(f"{path}: only polygon cells are supported")
    polys = mesh.GetPolys()
    offsets = _vtk_array(polys.GetOffsetsArray()).astype(np.int64)
    connectivity = _vtk_array(polys.GetConnectivityArray()).astype(np.int64)
    points = _vtk_array(mesh.GetPoints().GetData())
    pressure_array = mesh.GetPointData().GetArray(PRESSURE_ARRAY)
    if pressure_array is None:
        raise NativeSupportError(f"{path}: PointData array {PRESSURE_ARRAY!r} not found")
    pressure = _vtk_array(pressure_array)
    n = len(points)
    if points.shape != (n, 3) or pressure.shape != (n,):
        raise NativeSupportError(f"{path}: point or pressure array shape is invalid")
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(connectivity):
        raise NativeSupportError(f"{path}: polygon offsets are inconsistent")
    sizes = np.diff(offsets)
    if np.any(sizes < 3):
        raise NativeSupportError(f"{path}: polygon with fewer than 3 vertices")
    if connectivity.min() < 0 or connectivity.max() >= n:
        raise NativeSupportError(f"{path}: polygon references a missing point")
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(pressure))):
        raise NativeSupportError(f"{path}: non-finite coordinates or pressure")
    return NativeSurface(points, offsets, connectivity, pressure)


def mass_lumped_dual_areas(surface: NativeSurface) -> np.ndarray:
    """Split each polygon's fan-triangulated area equally among its vertices."""

    points = surface.points.astype(np.float64)
    offsets, connectivity = surface.offsets, surface.connectivity
    sizes = np.diff(offsets)
    starts = offsets[:-1]
    base = points[connectivity[starts]]
    areas = np.zeros(len(sizes), dtype=np.float64)
    # Fan triangle j spans vertices (0, j, j + 1); accumulate in ascending j,
    # the same order as the reference loop implementation.
    for j in range(1, int(sizes.max()) - 1):
        rows = np.flatnonzero(sizes >= j + 2)
        left = points[connectivity[starts[rows] + j]] - base[rows]
        right = points[connectivity[starts[rows] + j + 1]] - base[rows]
        areas[rows] += 0.5 * np.linalg.norm(np.cross(left, right), axis=1)
    shares = np.repeat(areas / sizes, sizes)
    weights = np.bincount(connectivity, weights=shares, minlength=surface.point_count)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise NativeSupportError("every native point must receive a positive dual area")
    return weights


def connected_components(surface: NativeSurface) -> np.ndarray:
    """Label points by polygon connectivity (numpy hook-and-compress union-find)."""

    offsets, connectivity = surface.offsets, surface.connectivity
    sizes = np.diff(offsets)
    first = np.repeat(connectivity[offsets[:-1]], sizes)
    parent = np.arange(surface.point_count, dtype=np.int64)
    while True:
        left, right = parent[first], parent[connectivity]
        low, high = np.minimum(left, right), np.maximum(left, right)
        changed = low != high
        if not np.any(changed):
            break
        np.minimum.at(parent, high[changed], low[changed])
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    _, labels = np.unique(parent, return_inverse=True)
    return labels.astype(np.int64)


def oriented_point_normals(path: Path, surface: NativeSurface) -> np.ndarray:
    """Outward unit point normals from VTK's consistent, auto-oriented filter."""

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(triangles.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    output = normals.GetOutput()
    result = np.array(vtk_to_numpy(output.GetPointData().GetNormals()), dtype=np.float64)
    out_points = vtk_to_numpy(output.GetPoints().GetData())
    if result.shape != surface.points.shape or not np.array_equal(out_points, surface.points):
        raise NativeSupportError(f"{path}: normal filter changed the native point order")
    return result


@dataclass(frozen=True)
class ProfileStation:
    station_id: str
    point_index: np.ndarray  # int64 native point indices, ordered by coordinate
    coordinate: np.ndarray  # float64, strictly increasing
    coordinate_id: str


def _envelope(candidates: np.ndarray, along: np.ndarray, across: np.ndarray,
              *, lower: float, length: float, keep_max: bool) -> np.ndarray:
    """One native point per bin of ``along``: the extreme of ``across``.

    Ties on ``across`` resolve to the lowest native index, so the selection is
    a pure function of the stored file.
    """

    if len(candidates) == 0:
        return candidates
    bins = np.floor((along[candidates] - lower) / (ENVELOPE_BIN * length)).astype(np.int64)
    key = -across[candidates] if keep_max else across[candidates]
    order = np.lexsort((candidates, key, bins))
    ordered_bins = bins[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = ordered_bins[1:] != ordered_bins[:-1]
    return candidates[order[first]]


def extract_profile_stations(
    surface: NativeSurface, labels: np.ndarray, normals: np.ndarray
) -> list[ProfileStation]:
    """Freeze the centreline pressure-profile stations for one design.

    * The body is the largest connected component.  There must be at least
      ``MINIMUM_COMPONENTS`` (body plus four wheels); every component of at
      least ``FRAGMENT_POINTS`` points must have outward normals on average.
      Smaller slivers (one 7-point fragment exists in official_test) are
      scored like every other point but never contribute to a profile.
    * Centreline band: body points with ``|y| <= CENTRELINE_BAND * L``.
    * Upper body: band points with ``n_z > 0.2``; underbody ``n_z < -0.2``;
      rear base ``n_x > 0.5`` and ``(x - x_min) / L >= REAR_REGION`` (the
      cowl step and roof trailing edge also face rearward on the centreline).
    * Outer envelope: per ``ENVELOPE_BIN * L`` bin of the station coordinate,
      keep the single highest (upper), lowest (underbody) or rearmost (rear)
      point.  This keeps the outer wetted surface and removes internal or
      overhanging surfaces (engine bay, tunnels, spoiler undersides) that the
      band would otherwise make multi-valued.
    * Coordinates: ``(x - x_min) / L`` for upper and underbody,
      ``(z - z_min) / L`` for the rear base.
    """

    points = surface.points.astype(np.float64)
    sizes = np.bincount(labels)
    if len(sizes) < MINIMUM_COMPONENTS:
        raise NativeSupportError(
            f"expected at least {MINIMUM_COMPONENTS} connected components, found {len(sizes)}"
        )
    for component in np.flatnonzero(sizes >= FRAGMENT_POINTS):
        member = labels == component
        centred = points[member] - points[member].mean(axis=0)
        if float(np.mean(np.sum(normals[member] * centred, axis=1))) <= 0.0:
            raise NativeSupportError(f"component {component} normals are not outward")
    body = int(np.argmax(sizes))
    x_min, z_min = float(points[:, 0].min()), float(points[:, 2].min())
    length = float(points[:, 0].max()) - x_min
    band = np.flatnonzero(
        (labels == body) & (np.abs(points[:, 1]) <= CENTRELINE_BAND * length)
    )
    nx, nz = normals[band, 0], normals[band, 2]
    x, z = points[:, 0], points[:, 2]
    selections = {
        "upper_body_centerline": (
            _envelope(band[nz > NORMAL_Z_THRESHOLD], x, z, lower=x_min, length=length, keep_max=True),
            (x - x_min) / length,
            "x_over_L",
        ),
        "underbody_centerline": (
            _envelope(band[nz < -NORMAL_Z_THRESHOLD], x, z, lower=x_min, length=length, keep_max=False),
            (x - x_min) / length,
            "x_over_L",
        ),
        "rear_body_centerline": (
            _envelope(
                band[(nx > NORMAL_X_THRESHOLD) & ((x[band] - x_min) / length >= REAR_REGION)],
                z, x, lower=z_min, length=length, keep_max=True,
            ),
            (z - z_min) / length,
            "z_over_L",
        ),
    }
    stations = []
    for station_id in PROFILE_STATIONS:
        selected, coordinate, coordinate_id = selections[station_id]
        values = coordinate[selected]
        order = np.argsort(values, kind="stable")
        selected, values = selected[order], values[order]
        if len(selected) < MINIMUM_STATION_POINTS or np.any(np.diff(values) <= 0.0):
            raise NativeSupportError(
                f"{station_id}: {len(selected)} points or non-increasing coordinates"
            )
        stations.append(ProfileStation(station_id, selected.astype(np.int64), values, coordinate_id))
    return stations
