"""Geometry-only native DrivAerML VTU loading.

The reader disables every advertised point- and cell-data array before loading
an unstructured grid.  It preserves the native points, topology, cell types,
and raw VTK cell order without attaching weights or redefining benchmark
metrics.  Callers own any task-specific Python, NumPy, and VTK runtime pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # VTK is an optional dependency used only by native-geometry workflows.
    import vtk  # type: ignore[import-not-found]
except ImportError as error:  # pragma: no cover - exercised without optional VTK.
    vtk = None
    _VTK_IMPORT_ERROR: ImportError | None = error
else:
    _VTK_IMPORT_ERROR = None


class NativeVolumeGeometryError(ValueError):
    """Raised when a native VTU geometry read cannot be trusted."""


@dataclass(frozen=True)
class GeometryReadAudit:
    """Source arrays disabled by the geometry-only XML reader."""

    disabled_point_arrays: tuple[str, ...]
    disabled_cell_arrays: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "disabled_point_array_count": len(self.disabled_point_arrays),
            "disabled_point_arrays": list(self.disabled_point_arrays),
            "disabled_cell_array_count": len(self.disabled_cell_arrays),
            "disabled_cell_arrays": list(self.disabled_cell_arrays),
        }


def _require_vtk() -> None:
    if vtk is None:
        detail = f": {_VTK_IMPORT_ERROR}" if _VTK_IMPORT_ERROR is not None else ""
        raise NativeVolumeGeometryError(
            f"native VTU geometry reading requires optional VTK{detail}"
        )


def _algorithm_error(algorithm: Any, label: str) -> None:
    error_code = int(algorithm.GetErrorCode())
    if error_code == 0:
        return
    description = vtk.vtkErrorCode.GetStringFromErrorCode(error_code)
    raise NativeVolumeGeometryError(
        f"{label} failed: {description} ({error_code})"
    )


def _vtk_event_message(arguments: tuple[object, ...]) -> str:
    """Return a bounded single-line diagnostic from VTK observer arguments."""

    if not arguments:
        return "no message supplied"
    value = arguments[-1]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    message = " ".join(str(value).split())
    return (message or "no message supplied")[:1000]


def _run_vtk_algorithm_stage(
    algorithm: Any,
    label: str,
    operation: Callable[[], object],
) -> None:
    """Fail on VTK events, global diagnostics, exceptions, or an error code."""

    if vtk is None:
        raise NativeVolumeGeometryError(f"{label} cannot run without VTK")
    events: list[tuple[str, str]] = []

    def capture(_caller: object, event: str, *arguments: object) -> None:
        events.append((str(event), _vtk_event_message(arguments)))

    observer_ids: list[int] = []
    previous_output_window = vtk.vtkOutputWindow.GetInstance()
    diagnostic_window = vtk.vtkStringOutputWindow()
    operation_error: Exception | None = None
    diagnostics = ""
    try:
        vtk.vtkOutputWindow.SetInstance(diagnostic_window)
        for event in ("WarningEvent", "ErrorEvent"):
            observer_ids.append(int(algorithm.AddObserver(event, capture)))
        try:
            operation()
        except Exception as error:  # Inspect diagnostics before propagating.
            operation_error = error
    except Exception as error:
        operation_error = error
    finally:
        try:
            for observer_id in observer_ids:
                algorithm.RemoveObserver(observer_id)
        finally:
            try:
                diagnostics = " ".join(str(diagnostic_window.GetOutput()).split())
            finally:
                vtk.vtkOutputWindow.SetInstance(previous_output_window)
    if diagnostics:
        error = NativeVolumeGeometryError(
            f"{label} emitted VTK global diagnostic: {diagnostics[:1000]}"
        )
        if operation_error is not None:
            raise error from operation_error
        raise error
    if events:
        event, message = events[0]
        error = NativeVolumeGeometryError(
            f"{label} emitted VTK {event}: {message}; event_count={len(events)}"
        )
        if operation_error is not None:
            raise error from operation_error
        raise error
    if operation_error is not None:
        if isinstance(operation_error, NativeVolumeGeometryError):
            raise operation_error
        raise NativeVolumeGeometryError(
            f"{label} raised an exception: {operation_error}"
        ) from operation_error
    _algorithm_error(algorithm, label)


def read_geometry_only_vtu(path: Path | str) -> tuple[Any, GeometryReadAudit]:
    """Read native VTU geometry after disabling every advertised data array."""

    _require_vtk()
    source = Path(path)
    if not source.is_file():
        raise NativeVolumeGeometryError(f"VTU source does not exist: {source}")
    is_retained_descriptor = source.parent in {
        Path("/proc/self/fd"),
        Path("/dev/fd"),
    }
    if source.suffix.lower() != ".vtu" and not is_retained_descriptor:
        raise NativeVolumeGeometryError("geometry source must use the .vtu suffix")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(source))
    _run_vtk_algorithm_stage(
        reader,
        "vtkXMLUnstructuredGridReader metadata pass",
        reader.UpdateInformation,
    )

    point_arrays = tuple(
        str(reader.GetPointArrayName(index))
        for index in range(reader.GetNumberOfPointArrays())
    )
    cell_arrays = tuple(
        str(reader.GetCellArrayName(index))
        for index in range(reader.GetNumberOfCellArrays())
    )
    for name in point_arrays:
        reader.SetPointArrayStatus(name, 0)
    for name in cell_arrays:
        reader.SetCellArrayStatus(name, 0)

    _run_vtk_algorithm_stage(
        reader,
        "vtkXMLUnstructuredGridReader geometry pass",
        reader.Update,
    )
    output = reader.GetOutput()
    if output is None or not output.IsA("vtkUnstructuredGrid"):
        raise NativeVolumeGeometryError(
            "VTU reader did not produce an unstructured grid"
        )

    grid = vtk.vtkUnstructuredGrid()
    grid.ShallowCopy(output)
    grid.GetPointData().Initialize()
    grid.GetCellData().Initialize()
    if grid.GetNumberOfCells() < 1:
        raise NativeVolumeGeometryError(
            "VTU geometry contains no native volume cells"
        )
    if grid.GetNumberOfPoints() < 1:
        raise NativeVolumeGeometryError("VTU geometry contains no points")
    return grid, GeometryReadAudit(point_arrays, cell_arrays)


__all__ = [
    "GeometryReadAudit",
    "NativeVolumeGeometryError",
    "read_geometry_only_vtu",
]
