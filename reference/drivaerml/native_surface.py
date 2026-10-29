"""Native DrivAerML VTP loading, fixed-area auditing, and field evaluation.

Importing this module does not require VTK.  Reading the native boundary does:
the candidate evaluator is pinned to exactly VTK 9.5.2.  Required surface
fields are loaded as ``CellData`` without triangulating or reordering the
polygon support.  Published area arrays remain external fixed inputs; this
module audits their count and hash binding but never regenerates them.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .accumulators import (
    DrivAerAccumulatorError,
    FinalizedFieldStatistics,
    StreamingFieldAccumulator,
)
from .retained_file import RetainedFileError, RetainedVerifiedFile
from .surface_forces import (
    DrivAerSurfaceForceError,
    audit_fixed_surface_areas,
    finalize_force_coefficients,
    force_moment_chunk,
    iter_raw_id_ranges,
    surface_geometry_chunk_validated,
    validate_polygon_topology,
)

try:  # VTK is an optional, separately pinned evaluator dependency.
    import vtk  # type: ignore[import-not-found]
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import-not-found]
except ImportError as error:  # pragma: no cover - exercised in the base environment.
    vtk = None
    vtk_to_numpy = None
    _VTK_IMPORT_ERROR: ImportError | None = error
else:
    _VTK_IMPORT_ERROR = None


REQUIRED_VTK_VERSION = "9.5.2"
PRESSURE_ARRAY_NAME = "pMeanTrim"
WALL_SHEAR_ARRAY_NAME = "wallShearStressMeanTrim"
DEFAULT_CHUNK_POLYGONS = 500_000
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class DrivAerNativeSurfaceError(ValueError):
    """Raised when native surface inputs or predictions violate the contract."""


@dataclass(frozen=True)
class NativeSurface:
    """Validated raw-order native polygon support and required truth fields."""

    vtk_owner: Any
    source_path: Path
    boundary_sha256: str
    points_m: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    pressure_m2_per_s2: np.ndarray
    wall_shear_m2_per_s2: np.ndarray
    available_point_arrays: tuple[str, ...]
    available_cell_arrays: tuple[str, ...]
    vtk_version: str

    @property
    def point_count(self) -> int:
        return int(self.points_m.shape[0])

    @property
    def polygon_count(self) -> int:
        return int(self.offsets.size - 1)

    def audit_record(self) -> dict[str, object]:
        return {
            "source_file": self.source_path.name,
            "boundary_sha256": self.boundary_sha256,
            "vtk_version": self.vtk_version,
            "point_count": self.point_count,
            "polygon_count": self.polygon_count,
            "raw_cell_order": "zero_based_native_vtk_polygon_order_unchanged",
            "association": "CellData",
            "arrays": {
                PRESSURE_ARRAY_NAME: {
                    "components": 1,
                    "tuples": self.polygon_count,
                    "dtype": self.pressure_m2_per_s2.dtype.str,
                    "unit": "m^2/s^2",
                },
                WALL_SHEAR_ARRAY_NAME: {
                    "components": 3,
                    "tuples": self.polygon_count,
                    "dtype": self.wall_shear_m2_per_s2.dtype.str,
                    "unit": "m^2/s^2",
                },
            },
            "available_point_arrays": list(self.available_point_arrays),
            "available_cell_arrays": list(self.available_cell_arrays),
        }


@dataclass(frozen=True)
class FixedSurfaceAreas:
    """Hash-bound published float32 areas in the identical native order."""

    values_m2: np.ndarray
    source_path: Path
    sha256: str
    source_boundary_sha256: str
    entity_count: int
    area_sum_m2: float
    area_min_m2: float
    area_max_m2: float
    _retained_file: RetainedVerifiedFile = field(repr=False, compare=False)

    def assert_source_unchanged(self, *, context: str) -> None:
        try:
            self._retained_file.assert_unchanged(context=context)
        except RetainedFileError as error:
            raise DrivAerNativeSurfaceError(str(error)) from error

    def close(self) -> None:
        self._retained_file.close()

    def audit_record(self) -> dict[str, object]:
        return {
            "source_path": self.source_path.name,
            "sha256": self.sha256,
            "source_boundary_sha256": self.source_boundary_sha256,
            "entity_count": self.entity_count,
            "area_sum_m2": self.area_sum_m2,
            "area_min_m2": self.area_min_m2,
            "area_max_m2": self.area_max_m2,
            "dtype": "<f4",
            "role": "fixed_external_input_not_regenerated",
        }


@dataclass(frozen=True)
class NativeSurfaceEvaluation:
    """Complete native-surface metrics and field-derived force coefficients."""

    entity_count: int
    chunk_polygons: int
    chunk_count: int
    metric_values: dict[str, float]
    metric_sufficient_statistics: dict[str, dict[str, float | int | str]]
    pressure_statistics: FinalizedFieldStatistics
    wall_shear_statistics: FinalizedFieldStatistics
    force_coefficients: dict[str, float | int | list[float]]

    def to_json(self) -> dict[str, object]:
        return {
            "entity_count": self.entity_count,
            "raw_id_start": 0,
            "raw_id_stop": self.entity_count,
            "complete_duplicate_free_coverage": True,
            "chunk_polygons": self.chunk_polygons,
            "chunk_count": self.chunk_count,
            "metric_values": self.metric_values,
            "metric_sufficient_statistics": self.metric_sufficient_statistics,
            "additive_sums": {
                "pressure": {
                    weighting: asdict(
                        getattr(self.pressure_statistics, weighting)
                    )
                    for weighting in ("uniform", "physical")
                },
                "wall_shear": {
                    weighting: asdict(
                        getattr(self.wall_shear_statistics, weighting)
                    )
                    for weighting in ("uniform", "physical")
                },
            },
            "force_coefficients": self.force_coefficients,
        }


def _vtk_version() -> str | None:
    if vtk is None:
        return None
    return str(vtk.vtkVersion.GetVTKVersion())


def vtk_available() -> bool:
    """Return whether the exact optional VTK dependency is importable."""

    return _vtk_version() == REQUIRED_VTK_VERSION


def _require_vtk() -> None:
    if vtk is None or vtk_to_numpy is None:
        detail = f": {_VTK_IMPORT_ERROR}" if _VTK_IMPORT_ERROR is not None else ""
        raise DrivAerNativeSurfaceError(
            f"native DrivAerML surfaces require optional VTK {REQUIRED_VTK_VERSION}{detail}"
        )
    actual = _vtk_version()
    if actual != REQUIRED_VTK_VERSION:
        raise DrivAerNativeSurfaceError(
            f"native DrivAerML surfaces require VTK {REQUIRED_VTK_VERSION}, got {actual}"
        )


def sha256_file(path: Path | str, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
        raise DrivAerNativeSurfaceError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _OpenSurfaceSnapshot:
    """Mutation-sensitive identity of one retained boundary descriptor."""

    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_handle(cls, handle: BinaryIO) -> "_OpenSurfaceSnapshot":
        try:
            value = os.fstat(handle.fileno())
        except OSError as error:
            raise DrivAerNativeSurfaceError(
                f"cannot fstat retained boundary VTP descriptor: {error}"
            ) from error
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )


@dataclass
class _RetainedSurfaceFile:
    """One open boundary inode retained through hashing and the VTK pass."""

    source_path: Path
    handle: BinaryIO
    snapshot: _OpenSurfaceSnapshot

    @classmethod
    def open(cls, source_path: Path) -> "_RetainedSurfaceFile":
        try:
            handle = source_path.open("rb", buffering=0)
        except OSError as error:
            raise DrivAerNativeSurfaceError(
                f"cannot open boundary VTP {source_path}: {error}"
            ) from error
        try:
            snapshot = _OpenSurfaceSnapshot.from_handle(handle)
            if not stat.S_ISREG(snapshot.mode):
                raise DrivAerNativeSurfaceError(
                    f"boundary VTP is not a regular file: {source_path}"
                )
            return cls(source_path=source_path, handle=handle, snapshot=snapshot)
        except Exception:
            handle.close()
            raise

    def __enter__(self) -> "_RetainedSurfaceFile":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.handle.close()

    def assert_unchanged(self, *, context: str) -> None:
        current = _OpenSurfaceSnapshot.from_handle(self.handle)
        try:
            pathname = os.stat(self.source_path)
            pathname_identity = (int(pathname.st_dev), int(pathname.st_ino))
        except OSError:
            pathname_identity = None
        if current != self.snapshot or pathname_identity != (
            self.snapshot.device,
            self.snapshot.inode,
        ):
            raise DrivAerNativeSurfaceError(
                f"retained boundary VTP {self.source_path} changed {context}; "
                "the pathname no longer names the retained inode, or device, "
                "inode, mode, size, mtime, or ctime differs from the pre-hash "
                "fstat snapshot"
            )

    def sha256(self, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
        if (
            not isinstance(chunk_bytes, int)
            or isinstance(chunk_bytes, bool)
            or chunk_bytes < 1
        ):
            raise DrivAerNativeSurfaceError("chunk_bytes must be a positive integer")
        self.assert_unchanged(context="before hashing")
        digest = hashlib.sha256()
        try:
            self.handle.seek(0)
            while block := self.handle.read(chunk_bytes):
                digest.update(block)
        except OSError as error:
            raise DrivAerNativeSurfaceError(
                f"cannot hash retained boundary VTP descriptor: {error}"
            ) from error
        self.assert_unchanged(context="while hashing")
        return digest.hexdigest()

    def descriptor_path(self) -> Path:
        """Expose this exact retained inode to the in-process VTK reader."""

        self.assert_unchanged(context="before exposing its descriptor path")
        descriptor = self.handle.fileno()
        for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = directory / str(descriptor)
            try:
                target = os.stat(candidate)
            except OSError:
                continue
            if (
                int(target.st_dev) == self.snapshot.device
                and int(target.st_ino) == self.snapshot.inode
            ):
                return candidate
        raise DrivAerNativeSurfaceError(
            "no safe descriptor filesystem exposes the retained boundary VTP; "
            "expected /proc/self/fd or /dev/fd"
        )


def _expected_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DrivAerNativeSurfaceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _algorithm_error(algorithm: Any, label: str) -> None:
    error_code = int(algorithm.GetErrorCode())
    if error_code == 0:
        return
    description = vtk.vtkErrorCode.GetStringFromErrorCode(error_code)
    raise DrivAerNativeSurfaceError(
        f"{label} failed: {description} ({error_code})"
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _load_native_surface_vtp_from_descriptor(
    vtk_source: Path,
    *,
    source_path: Path,
    boundary_sha256: str,
) -> NativeSurface:
    """Parse a boundary through a path proven to name its retained descriptor."""

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtk_source))
    reader.UpdateInformation()
    _algorithm_error(reader, "vtkXMLPolyDataReader metadata pass")
    point_arrays = tuple(
        str(reader.GetPointArrayName(index))
        for index in range(reader.GetNumberOfPointArrays())
    )
    cell_arrays = tuple(
        str(reader.GetCellArrayName(index))
        for index in range(reader.GetNumberOfCellArrays())
    )
    required = (PRESSURE_ARRAY_NAME, WALL_SHEAR_ARRAY_NAME)
    missing = sorted(set(required) - set(cell_arrays))
    if missing:
        raise DrivAerNativeSurfaceError(
            f"boundary VTP is missing required CellData arrays: {missing}"
        )
    for name in point_arrays:
        reader.SetPointArrayStatus(name, 0)
    for name in cell_arrays:
        reader.SetCellArrayStatus(name, 0)
    for name in required:
        reader.SetCellArrayStatus(name, 1)
    reader.Update()
    _algorithm_error(reader, "vtkXMLPolyDataReader data pass")
    output = reader.GetOutput()
    if output is None or not output.IsA("vtkPolyData"):
        raise DrivAerNativeSurfaceError("VTP reader did not produce vtkPolyData")

    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(output)
    polygon_count = int(polydata.GetNumberOfPolys())
    if (
        polygon_count < 1
        or int(polydata.GetNumberOfCells()) != polygon_count
        or int(polydata.GetNumberOfVerts()) != 0
        or int(polydata.GetNumberOfLines()) != 0
        or int(polydata.GetNumberOfStrips()) != 0
    ):
        raise DrivAerNativeSurfaceError(
            "boundary VTP must contain native polygons only"
        )
    if polydata.GetPoints() is None or polydata.GetPoints().GetData() is None:
        raise DrivAerNativeSurfaceError("boundary VTP contains no points")

    cell_data = polydata.GetCellData()
    pressure_array = cell_data.GetArray(PRESSURE_ARRAY_NAME)
    shear_array = cell_data.GetArray(WALL_SHEAR_ARRAY_NAME)
    if pressure_array is None or shear_array is None:
        raise DrivAerNativeSurfaceError("required surface CellData did not load")
    if (
        int(pressure_array.GetNumberOfComponents()) != 1
        or int(pressure_array.GetNumberOfTuples()) != polygon_count
    ):
        raise DrivAerNativeSurfaceError(
            "pMeanTrim must contain one scalar per native polygon"
        )
    if (
        int(shear_array.GetNumberOfComponents()) != 3
        or int(shear_array.GetNumberOfTuples()) != polygon_count
    ):
        raise DrivAerNativeSurfaceError(
            "wallShearStressMeanTrim must contain one 3-vector per native polygon"
        )

    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()))
    connectivity = np.asarray(vtk_to_numpy(polydata.GetPolys().GetConnectivityArray()))
    offsets = np.asarray(vtk_to_numpy(polydata.GetPolys().GetOffsetsArray()))
    pressure = np.asarray(vtk_to_numpy(pressure_array))
    wall_shear = np.asarray(vtk_to_numpy(shear_array))
    if pressure.shape != (polygon_count,) or wall_shear.shape != (polygon_count, 3):
        raise DrivAerNativeSurfaceError(
            "surface CellData tuple/component counts changed during conversion"
        )
    if pressure.dtype.kind not in {"i", "u", "f"} or wall_shear.dtype.kind not in {
        "i",
        "u",
        "f",
    }:
        raise DrivAerNativeSurfaceError("surface CellData must be numeric")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(wall_shear)):
        raise DrivAerNativeSurfaceError("surface CellData contains non-finite values")
    try:
        points, connectivity, offsets = validate_polygon_topology(
            points, connectivity, offsets
        )
    except DrivAerSurfaceForceError as error:
        raise DrivAerNativeSurfaceError(str(error)) from error
    if offsets.size - 1 != polygon_count:
        raise DrivAerNativeSurfaceError(
            "polygon connectivity count differs from native CellData count"
        )

    # Only the required fields remain resident.  The retained VTK owner keeps
    # the zero-copy NumPy field views valid for the lifetime of NativeSurface.
    polydata.GetPointData().Initialize()
    for index in reversed(range(cell_data.GetNumberOfArrays())):
        name = cell_data.GetArrayName(index)
        if name not in required:
            cell_data.RemoveArray(name)
    return NativeSurface(
        vtk_owner=polydata,
        source_path=source_path,
        boundary_sha256=boundary_sha256,
        points_m=_readonly(points),
        connectivity=_readonly(connectivity),
        offsets=_readonly(offsets),
        pressure_m2_per_s2=_readonly(pressure),
        wall_shear_m2_per_s2=_readonly(wall_shear),
        available_point_arrays=point_arrays,
        available_cell_arrays=cell_arrays,
        vtk_version=str(_vtk_version()),
    )


def load_native_surface_vtp(path: Path | str) -> NativeSurface:
    """Load raw-order native polygon ``CellData`` from one verified inode.

    The boundary is opened exactly once by this function.  Its SHA-256 is
    calculated through that retained descriptor and VTK receives a validated
    ``/proc/self/fd`` (or ``/dev/fd``) path to the same inode.  Mutation-sensitive
    ``fstat`` metadata is checked before and after both passes, so pathname
    replacement cannot redirect VTK and in-place mutation fails closed.
    """

    _require_vtk()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DrivAerNativeSurfaceError(f"boundary VTP does not exist: {source}")
    if source.suffix.lower() != ".vtp":
        raise DrivAerNativeSurfaceError("native surface source must use the .vtp suffix")
    with _RetainedSurfaceFile.open(source) as retained:
        boundary_sha256 = retained.sha256()
        vtk_source = retained.descriptor_path()
        surface = _load_native_surface_vtp_from_descriptor(
            vtk_source,
            source_path=source,
            boundary_sha256=boundary_sha256,
        )
        retained.assert_unchanged(context="while VTK read the verified boundary")
        return surface


def audit_fixed_surface_area_file(
    surface: NativeSurface,
    path: Path | str,
    *,
    expected_area_sha256: str,
    source_boundary_sha256: str,
    validation_chunk_entities: int = 1_000_000,
) -> FixedSurfaceAreas:
    """Audit the published area file by hash/count/source binding only.

    No area is calculated from geometry here.  The source boundary digest must
    come from the immutable area manifest and must match the VTP just loaded.
    """

    if not isinstance(surface, NativeSurface):
        raise DrivAerNativeSurfaceError("surface must be a NativeSurface")
    if (
        not isinstance(validation_chunk_entities, int)
        or isinstance(validation_chunk_entities, bool)
        or validation_chunk_entities < 1
    ):
        raise DrivAerNativeSurfaceError(
            "validation_chunk_entities must be a positive integer"
        )
    expected_area = _expected_sha256(expected_area_sha256, "expected_area_sha256")
    expected_boundary = _expected_sha256(
        source_boundary_sha256, "source_boundary_sha256"
    )
    if expected_boundary != surface.boundary_sha256:
        raise DrivAerNativeSurfaceError(
            "area manifest source boundary SHA-256 differs from the loaded VTP"
        )
    source = Path(path).resolve()
    if not source.is_file():
        raise DrivAerNativeSurfaceError(f"surface-area file does not exist: {source}")
    if source.suffix.lower() != ".npy":
        raise DrivAerNativeSurfaceError("surface-area input must use the .npy suffix")
    try:
        retained = RetainedVerifiedFile.open(
            source, label="fixed surface-area NPY"
        )
        actual_area = retained.sha256()
    except RetainedFileError as error:
        raise DrivAerNativeSurfaceError(str(error)) from error
    if actual_area != expected_area:
        retained.close()
        raise DrivAerNativeSurfaceError("surface-area SHA-256 mismatch")
    try:
        values = np.load(
            retained.descriptor_path(), mmap_mode="r", allow_pickle=False
        )
        retained.assert_unchanged(context="while NumPy opened the memory map")
    except (OSError, ValueError, RetainedFileError) as error:
        retained.close()
        raise DrivAerNativeSurfaceError(
            f"cannot read fixed surface-area NPY: {error}"
        ) from error
    if values.dtype != np.dtype("<f4"):
        retained.close()
        raise DrivAerNativeSurfaceError(
            "published surface areas must use little-endian float32"
        )
    if values.shape != (surface.polygon_count,):
        retained.close()
        raise DrivAerNativeSurfaceError(
            "surface-area count differs from the native polygon count"
        )

    partial_sums: list[float] = []
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, values.size, validation_chunk_entities):
        stop = min(start + validation_chunk_entities, values.size)
        block = np.asarray(values[start:stop])
        if not np.all(np.isfinite(block)):
            retained.close()
            raise DrivAerNativeSurfaceError("surface areas must be finite")
        if np.any(block <= 0.0):
            retained.close()
            raise DrivAerNativeSurfaceError("surface areas must be strictly positive")
        partial_sums.append(float(np.sum(block, dtype=np.float64)))
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
    area_sum = math.fsum(partial_sums)
    if not math.isfinite(area_sum) or area_sum <= 0.0:
        retained.close()
        raise DrivAerNativeSurfaceError("surface-area sum must be finite and positive")
    try:
        retained.assert_unchanged(context="while validating every area")
    except RetainedFileError as error:
        retained.close()
        raise DrivAerNativeSurfaceError(str(error)) from error
    return FixedSurfaceAreas(
        values_m2=values,
        source_path=source,
        sha256=actual_area,
        source_boundary_sha256=expected_boundary,
        entity_count=surface.polygon_count,
        area_sum_m2=area_sum,
        area_min_m2=minimum,
        area_max_m2=maximum,
        _retained_file=retained,
    )


def _prediction_arrays(
    raw_cell_ids: object,
    pressure_prediction: object,
    wall_shear_prediction: object,
    *,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_ids = np.asarray(raw_cell_ids)
    pressure = np.asarray(pressure_prediction)
    shear = np.asarray(wall_shear_prediction)
    if raw_ids.shape != (expected_count,) or raw_ids.dtype.kind not in {"i", "u"}:
        raise DrivAerNativeSurfaceError(
            "raw_cell_ids must contain one integer ID per native polygon"
        )
    if pressure.shape != (expected_count,) or pressure.dtype.kind not in {"i", "u", "f"}:
        raise DrivAerNativeSurfaceError(
            "pressure predictions must contain one numeric scalar per native polygon"
        )
    if shear.shape != (expected_count, 3) or shear.dtype.kind not in {"i", "u", "f"}:
        raise DrivAerNativeSurfaceError(
            "wall-shear predictions must contain one numeric 3-vector per native polygon"
        )
    return raw_ids, pressure, shear


def _metric_values(
    pressure: FinalizedFieldStatistics,
    wall_shear: FinalizedFieldStatistics,
) -> dict[str, float]:
    return {
        "surface_pressure_rel_l2": pressure.physical.relative_l2_percent(),
        "surface_pressure_equal_entity_rel_l2": pressure.uniform.relative_l2_percent(),
        "surface_wall_shear_rel_l2": wall_shear.physical.relative_l2_percent(),
        "surface_wall_shear_equal_entity_rel_l2": wall_shear.uniform.relative_l2_percent(),
        "drivaerml_surface_pressure_area_mae": pressure.physical.mae(),
        "drivaerml_surface_pressure_area_rmse": pressure.physical.rmse(),
        "drivaerml_surface_pressure_equal_entity_mae": pressure.uniform.mae(),
        "drivaerml_surface_pressure_equal_entity_rmse": pressure.uniform.rmse(),
        "drivaerml_surface_wall_shear_area_mae": wall_shear.physical.mae(),
        "drivaerml_surface_wall_shear_area_rmse": wall_shear.physical.rmse(),
        "drivaerml_surface_wall_shear_equal_entity_mae": wall_shear.uniform.mae(),
        "drivaerml_surface_wall_shear_equal_entity_rmse": wall_shear.uniform.rmse(),
    }


def evaluate_native_surface_predictions(
    surface: NativeSurface,
    fixed_areas: FixedSurfaceAreas,
    *,
    raw_cell_ids: object,
    pressure_prediction: object,
    wall_shear_prediction: object,
    chunk_polygons: int = DEFAULT_CHUNK_POLYGONS,
) -> NativeSurfaceEvaluation:
    """Score same-order predictions and integrate their forces in bounded chunks."""

    if not isinstance(surface, NativeSurface):
        raise DrivAerNativeSurfaceError("surface must be a NativeSurface")
    if not isinstance(fixed_areas, FixedSurfaceAreas):
        raise DrivAerNativeSurfaceError("fixed_areas must be FixedSurfaceAreas")
    fixed_areas.assert_source_unchanged(context="before surface reduction")
    if fixed_areas.entity_count != surface.polygon_count:
        raise DrivAerNativeSurfaceError("surface-area and VTP entity counts differ")
    if fixed_areas.source_boundary_sha256 != surface.boundary_sha256:
        raise DrivAerNativeSurfaceError(
            "surface areas are not bound to this boundary VTP"
        )
    if (
        not isinstance(chunk_polygons, int)
        or isinstance(chunk_polygons, bool)
        or chunk_polygons < 1
    ):
        raise DrivAerNativeSurfaceError("chunk_polygons must be a positive integer")
    count = surface.polygon_count
    raw_ids, pressure_prediction_array, shear_prediction_array = _prediction_arrays(
        raw_cell_ids,
        pressure_prediction,
        wall_shear_prediction,
        expected_count=count,
    )
    pressure_accumulator = StreamingFieldAccumulator(count, component_count=1)
    shear_accumulator = StreamingFieldAccumulator(count, component_count=3)
    force_chunks = []
    chunk_count = 0
    try:
        for start, stop in iter_raw_id_ranges(count, chunk_polygons):
            expected_ids = np.arange(start, stop, dtype=np.int64)
            if not np.array_equal(raw_ids[start:stop], expected_ids):
                raise DrivAerNativeSurfaceError(
                    "prediction raw_cell_ids must equal zero-based native polygon order"
                )
            weights = fixed_areas.values_m2[start:stop]
            pressure_accumulator.add_chunk(
                expected_ids,
                surface.pressure_m2_per_s2[start:stop],
                pressure_prediction_array[start:stop],
                weights,
            )
            shear_accumulator.add_chunk(
                expected_ids,
                surface.wall_shear_m2_per_s2[start:stop],
                shear_prediction_array[start:stop],
                weights,
            )
            geometry = surface_geometry_chunk_validated(
                surface.points_m,
                surface.connectivity,
                surface.offsets,
                start,
                stop,
            )
            audit_fixed_surface_areas(
                geometry.areas_m2,
                weights,
                rtol=6.0e-8,
            )
            force_chunks.append(
                force_moment_chunk(
                    geometry,
                    pressure_prediction_array[start:stop],
                    shear_prediction_array[start:stop],
                )
            )
            chunk_count += 1
        pressure = pressure_accumulator.finalize()
        wall_shear = shear_accumulator.finalize()
        coefficients = finalize_force_coefficients(
            force_chunks, expected_entity_count=count
        )
    except (DrivAerAccumulatorError, DrivAerSurfaceForceError) as error:
        raise DrivAerNativeSurfaceError(str(error)) from error

    fixed_areas.assert_source_unchanged(context="while surface metrics were reduced")

    try:
        metric_values = _metric_values(pressure, wall_shear)
        sufficient_statistics = {
            "surface_pressure_rel_l2": pressure.physical.relative_l2_evidence(
                weighting="support_weights",
                dataset_weighting="surface_face_area",
            ),
            "surface_pressure_equal_entity_rel_l2": pressure.uniform.relative_l2_evidence(
                weighting="uniform",
                dataset_weighting="surface_entities_equal",
            ),
            "surface_wall_shear_rel_l2": wall_shear.physical.relative_l2_evidence(
                weighting="support_weights",
                dataset_weighting="surface_face_area",
            ),
            "surface_wall_shear_equal_entity_rel_l2": wall_shear.uniform.relative_l2_evidence(
                weighting="uniform",
                dataset_weighting="surface_entities_equal",
            ),
        }
    except DrivAerAccumulatorError as error:
        raise DrivAerNativeSurfaceError(str(error)) from error
    return NativeSurfaceEvaluation(
        entity_count=count,
        chunk_polygons=chunk_polygons,
        chunk_count=chunk_count,
        metric_values=metric_values,
        metric_sufficient_statistics=sufficient_statistics,
        pressure_statistics=pressure,
        wall_shear_statistics=wall_shear,
        force_coefficients=coefficients,
    )


__all__ = [
    "DEFAULT_CHUNK_POLYGONS",
    "PRESSURE_ARRAY_NAME",
    "REQUIRED_VTK_VERSION",
    "WALL_SHEAR_ARRAY_NAME",
    "DrivAerNativeSurfaceError",
    "FixedSurfaceAreas",
    "NativeSurface",
    "NativeSurfaceEvaluation",
    "audit_fixed_surface_area_file",
    "evaluate_native_surface_predictions",
    "load_native_surface_vtp",
    "sha256_file",
    "vtk_available",
]
