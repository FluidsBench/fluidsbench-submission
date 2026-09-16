from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any

import airfrans as af
import numpy as np
import pyvista as pv
import vtk
from airfrans.naca_generator import camber_line

DEFAULT_CASE_NAME = "airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548"
HASH_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_versions() -> dict[str, str]:
    return {
        "airfrans_distribution": importlib.metadata.version("airfrans"),
        "airfrans_module": af.__version__,
        "numpy": np.__version__,
        "pyvista": pv.__version__,
        "vtk": vtk.vtkVersion.GetVTKVersion(),
    }


def validate_runtime(spec: dict[str, Any], repo_root: Path) -> dict[str, str]:
    runtime = spec["sampling_runtime"]
    observed = runtime_versions()
    expected = {
        "airfrans_distribution": runtime["airfrans_distribution_version"],
        "airfrans_module": runtime["airfrans_module_version"],
        "numpy": runtime["numpy_version"],
        "pyvista": runtime["pyvista_version"],
        "vtk": runtime["vtk_version"],
    }
    mismatches = [
        f"{name}: expected {expected[name]}, observed {observed[name]}"
        for name in expected
        if observed[name] != expected[name]
    ]

    source_path = Path(inspect.getsourcefile(af.Simulation) or "")
    source_sha256 = sha256_file(source_path) if source_path.is_file() else ""
    if source_sha256 != runtime["airfrans_runtime_source_file_sha256"]:
        mismatches.append(
            "airfrans runtime source: expected "
            f"{runtime['airfrans_runtime_source_file_sha256']}, observed {source_sha256 or 'missing'}"
        )

    requirements_path = repo_root / runtime["requirements_file"]
    requirements_sha256 = (
        sha256_file(requirements_path) if requirements_path.is_file() else ""
    )
    if requirements_sha256 != runtime["requirements_sha256"]:
        mismatches.append(
            "runtime requirements: expected "
            f"{runtime['requirements_sha256']}, observed {requirements_sha256 or 'missing'}"
        )

    if mismatches:
        raise RuntimeError(
            "Runtime does not match the pinned profile contract:\n- "
            + "\n- ".join(mismatches)
        )
    return observed


def outward_unit_normals(simulation: af.Simulation) -> np.ndarray:
    inward = np.asarray(simulation.airfoil_normals, dtype=np.float64)
    norms = np.linalg.norm(inward, axis=1, keepdims=True)
    if (
        not np.isfinite(inward).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0.0)
    ):
        raise ValueError("AirfRANS airfoil normals must be finite and non-zero")
    return -inward / norms


def extrados_station_index(simulation: af.Simulation, x_over_c: float) -> int:
    digits = np.asarray(
        list(map(float, simulation.name.split("_")[4:-1])), dtype=np.float64
    )
    camber = camber_line(digits, simulation.airfoil_position[:, 0])[0]
    extrados_indices = np.flatnonzero(simulation.airfoil_position[:, 1] > camber)
    if extrados_indices.size == 0:
        raise ValueError(f"No extrados points found for {simulation.name}")
    local_index = int(
        np.argmin(np.abs(simulation.airfoil_position[extrados_indices, 0] - x_over_c))
    )
    return int(extrados_indices[local_index])


def validate_sampling_line(
    simulation: af.Simulation,
    station_id: str,
    x_over_c: float,
    outward_normals: np.ndarray,
    line_length_m: float,
    resolution_segments: int,
    expected_sample_count: int,
) -> dict[str, Any]:
    station_index = extrados_station_index(simulation, x_over_c)
    origin = np.concatenate(
        [
            np.asarray(simulation.airfoil_position[station_index], dtype=np.float64),
            np.asarray([simulation.internal.points[0, 2]], dtype=np.float64),
        ]
    )
    direction = outward_normals[station_index]
    endpoint = origin + line_length_m * np.concatenate([direction, np.asarray([0.0])])
    sampled = simulation.internal.sample_over_line(
        origin,
        endpoint,
        resolution=resolution_segments,
    )
    if "vtkValidPointMask" not in sampled.point_data:
        raise RuntimeError(f"{station_id}: VTK did not return vtkValidPointMask")
    valid_mask = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    if valid_mask.size != expected_sample_count:
        raise ValueError(
            f"{station_id}: expected {expected_sample_count} validity entries, "
            f"observed {valid_mask.size}"
        )
    invalid_indices = np.flatnonzero(~valid_mask)
    if invalid_indices.size:
        preview = ", ".join(map(str, invalid_indices[:10]))
        raise ValueError(
            f"{station_id}: {invalid_indices.size} of {expected_sample_count} points are outside "
            f"the fluid mesh (first invalid indices: {preview})"
        )
    return {
        "station_id": station_id,
        "surface_point_index": station_index,
        "origin_m": origin.tolist(),
        "outward_unit_normal": direction.tolist(),
        "valid_sample_count": int(valid_mask.sum()),
    }


def extract_profiles(
    simulation: af.Simulation,
    spec: dict[str, Any],
    *,
    reference: bool,
    velocity: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extraction = spec["extraction"]
    expected_sample_count = int(extraction["sample_count"])
    resolution_segments = int(extraction["resolution_segments"])
    if expected_sample_count != resolution_segments + 1:
        raise ValueError(
            "The profile contract must define one more sample than line segments"
        )
    if extraction["normal_transform"] != "negate_inward_normal_then_l2_normalize":
        raise ValueError("Unsupported AirfRANS normal transform")

    if reference and velocity is not None:
        raise ValueError("Reference extraction cannot accept a prediction array")
    prediction = None if velocity is None else np.asarray(velocity, dtype=np.float64)
    if not reference:
        if prediction is None:
            raise ValueError("Prediction extraction requires a velocity array")
        if prediction.shape != simulation.velocity.shape:
            raise ValueError(
                f"Prediction shape must be {simulation.velocity.shape}, observed {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction velocity must contain only finite values")

    outward_normals = outward_unit_normals(simulation)
    original_normals = simulation.airfoil_normals
    original_velocity = simulation.velocity
    simulation.airfoil_normals = outward_normals
    if prediction is not None:
        simulation.velocity = prediction

    coordinate = np.linspace(
        0.0,
        float(extraction["line_length_m"]),
        expected_sample_count,
        dtype=np.float64,
    ).tolist()
    call_arguments = extraction[
        "ground_truth_call_arguments" if reference else "prediction_call_arguments"
    ]
    panel_id = spec["metric_binding"]["panel_id"]
    series: list[dict[str, Any]] = []
    sampling_records: list[dict[str, Any]] = []

    try:
        for station in spec["stations"]:
            station_id = station["id"]
            x_over_c = float(station["x_over_c"])
            sampling_records.append(
                validate_sampling_line(
                    simulation,
                    station_id,
                    x_over_c,
                    outward_normals,
                    float(extraction["line_length_m"]),
                    resolution_segments,
                    expected_sample_count,
                )
            )
            _, velocity_x, velocity_y, _, _ = simulation.boundary_layer(
                x=x_over_c,
                **call_arguments,
            )
            velocity_outputs = {
                1: np.asarray(velocity_x, dtype=np.float64),
                2: np.asarray(velocity_y, dtype=np.float64),
            }
            for quantity in spec["quantities"]:
                output_index = int(quantity["routine_output_index"])
                if output_index not in velocity_outputs:
                    raise ValueError(f"Unsupported routine output index {output_index}")
                values = velocity_outputs[output_index]
                if values.size != expected_sample_count:
                    raise ValueError(
                        f"{station_id}/{quantity['id']}: expected {expected_sample_count} samples, "
                        f"observed {values.size}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{station_id}/{quantity['id']}: samples must be finite"
                    )
                series.append(
                    {
                        "panel_id": panel_id,
                        "station_id": station_id,
                        "quantity_id": quantity["id"],
                        "coordinate": coordinate,
                        "prediction": values.tolist(),
                    }
                )
    finally:
        simulation.airfoil_normals = original_normals
        simulation.velocity = original_velocity

    return series, sampling_records


def maximum_series_difference(
    reference_series: list[dict[str, Any]],
    prediction_series: list[dict[str, Any]],
) -> float:
    if len(reference_series) != len(prediction_series):
        raise ValueError("Reference and prediction series counts differ")
    maximum = 0.0
    for reference, prediction in zip(reference_series, prediction_series, strict=True):
        identity = ("panel_id", "station_id", "quantity_id", "coordinate")
        if any(reference[key] != prediction[key] for key in identity):
            raise ValueError("Reference and prediction series identities differ")
        difference = np.max(
            np.abs(
                np.asarray(reference["prediction"], dtype=np.float64)
                - np.asarray(prediction["prediction"], dtype=np.float64)
            )
        )
        maximum = max(maximum, float(difference))
    return maximum


def load_velocity_predictions(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{path} does not contain the requested array {key!r}")
            return np.asarray(archive[key], dtype=np.float64)
    raise ValueError("Velocity predictions must be stored as .npy or .npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract FluidsBench AirfRANS extrados velocity profiles."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the main AirfRANS Dataset folder.",
    )
    parser.add_argument("--case-name", default=DEFAULT_CASE_NAME)
    parser.add_argument(
        "--velocity-predictions",
        type=Path,
        help="Optional .npy or .npz Cartesian velocity array in native internal-mesh point order.",
    )
    parser.add_argument(
        "--prediction-key",
        default="velocity",
        help="Array key used when --velocity-predictions points to an .npz archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to the committed example fixture.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    spec_path = repo_root / "benchmark-specs" / "airfrans" / "velocity-profiles-v1.json"
    spec = load_json(spec_path)
    versions = validate_runtime(spec, repo_root)

    case_dir = args.dataset_root / args.case_name
    internal_vtu = case_dir / f"{args.case_name}_internal.vtu"
    aerofoil_vtp = case_dir / f"{args.case_name}_aerofoil.vtp"
    missing = [str(path) for path in (internal_vtu, aerofoil_vtp) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required AirfRANS files:\n- " + "\n- ".join(missing)
        )

    simulation = af.Simulation(root=str(args.dataset_root), name=args.case_name)
    ground_truth_series, sampling_records = extract_profiles(
        simulation, spec, reference=True
    )
    perfect_copy_series, perfect_copy_sampling = extract_profiles(
        simulation,
        spec,
        reference=False,
        velocity=np.asarray(
            simulation.internal.point_data["U"][:, :2], dtype=np.float64
        ),
    )
    if sampling_records != perfect_copy_sampling:
        raise RuntimeError("Reference and prediction sampling geometry differ")
    perfect_copy_max_abs_difference = maximum_series_difference(
        ground_truth_series,
        perfect_copy_series,
    )
    tolerance = float(spec["validation"]["perfect_copy_absolute_tolerance"])
    if perfect_copy_max_abs_difference > tolerance:
        raise RuntimeError(
            "Perfect-copy prediction extraction differs from reference extraction: "
            f"{perfect_copy_max_abs_difference} > {tolerance}"
        )

    prediction_path = args.velocity_predictions
    if prediction_path is None:
        selected_series = ground_truth_series
        mode = "ground_truth_reference"
    else:
        velocity = load_velocity_predictions(prediction_path, args.prediction_key)
        selected_series, prediction_sampling = extract_profiles(
            simulation,
            spec,
            reference=False,
            velocity=velocity,
        )
        if sampling_records != prediction_sampling:
            raise RuntimeError(
                "Ground-truth and model-prediction sampling geometry differ"
            )
        mode = "model_prediction"

    input_hashes = {
        "internal_vtu": sha256_file(internal_vtu),
        "aerofoil_vtp": sha256_file(aerofoil_vtp),
    }
    if prediction_path is not None:
        input_hashes["velocity_predictions"] = sha256_file(prediction_path)

    output = {
        "schema_version": "1.0",
        "provenance": {
            "mode": mode,
            "profile_definition_id": spec["profile_definition_id"],
            "extractor_sha256": sha256_file(Path(__file__)),
            "versions": versions,
            "airfrans_runtime_source_file_sha256": sha256_file(
                Path(inspect.getsourcefile(af.Simulation) or "")
            ),
            "input_hashes": input_hashes,
            "sampling": {
                "surface": spec["extraction"]["surface"],
                "normal_transform": spec["extraction"]["normal_transform"],
                "stations": sampling_records,
            },
            "perfect_copy_validation": {
                "absolute_tolerance": tolerance,
                "maximum_absolute_difference": perfect_copy_max_abs_difference,
            },
        },
        "cases": [{"case_id": args.case_name, "series": selected_series}],
    }

    output_path = args.output or Path(__file__).with_name("example_extraction.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output_path)


if __name__ == "__main__":
    main()
