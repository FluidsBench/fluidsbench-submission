"""One explicit native-to-SI export stage; never alter relative statistics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ID = "hiliftaeroml-dimensional-export-si-v1"
CONTRACT_PATH = Path("benchmark-specs/hiliftaeroml/dimensional-export-si-v1.json")
CONTRACT_SHA256 = "a25f25b24228b72b0798704a155745ea5aab40db928e3e27a01904ec868acc59"
PRESSURE_TO_PA = 14.5939029372 / 0.0254
VELOCITY_TO_MPS = 0.0254
METRIC_FACTORS = {
    f"{field}_{reduction}": VELOCITY_TO_MPS if field == "volume_velocity" else PRESSURE_TO_PA
    for field in ("surface_pressure", "surface_wall_shear", "volume_pressure", "volume_velocity")
    for reduction in ("mae", "rmse")
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_binding() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "stage": "fluidsbench_package_export",
        "input_basis": "solver_native_dimensional",
        "output_basis": "SI",
        "application_count": 1,
    }


def validate_contract(root: Path = ROOT) -> None:
    path = root / CONTRACT_PATH
    if digest(path) != CONTRACT_SHA256:
        raise ValueError("HiLiftAeroML dimensional export contract hash differs")
    if json.loads(path.read_text())["metric_factors"] != METRIC_FACTORS:
        raise ValueError("HiLiftAeroML dimensional conversion factors differ")


def native_error_to_si(metric_id: str, value: float) -> float:
    """Accept only the eight dimensional metrics from the native assembler input."""
    if metric_id not in METRIC_FACTORS:
        raise ValueError(f"{metric_id}: not a dimensional HiLiftAeroML field error")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{metric_id}: expected a finite nonnegative native error")
    converted = value * METRIC_FACTORS[metric_id]
    if not math.isfinite(converted):
        raise ValueError(f"{metric_id}: SI conversion overflow")
    return converted
