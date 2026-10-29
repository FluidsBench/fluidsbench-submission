"""Pinned native profile sampling for DrivAerML submissions.

The published native-v3 truth also binds the raw cell selection, supported
segments and Cp arc lengths. Predictions are gathered independently from the
participant's complete native prediction streams; truth is used only to score.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

from reference.drivaerml.diagnostic_evaluator import gather_mapped_prediction_field

TRUTH_COMMIT = "scientific-truth-v1"
TRUTH_BASE = f"https://raw.githubusercontent.com/FluidsBench/fluidsbench/{TRUTH_COMMIT}/assets/data/profile-ground-truth/datasets/drivaerml/native-v3/"
INDEX_SHA256 = "e7cf14f161fc7dbf22794e6f66db4e157329be960d2a4368140e08cf0608a5ae"
SCHEMA = "drivaerml-native-profile-evidence-v2"
WORKFLOW_VERSION = 2
RELEASE_ID = "drivaerml-profile-scoring-v2"
SCORING_CONTRACT_SHA256 = (
    "c67ba550c661d0925c7e54d2c7ebda217a343dabd4a3d1f23e80618d2cb01449"
)
UINF = 38.889
COMMON = (
    "panel_id",
    "family_id",
    "placement_mode",
    "station_id",
    "quantity_id",
    "scoring_role",
    "representation",
    "placement_receipt_identity_sha256",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_scoring_contract(specification):
    contract = specification.get("profile_scoring", {})
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmark-specs/drivaerml/profile-scoring-v2.json"
    )
    if (
        contract.get("file") != "profile-scoring-v2.json"
        or contract.get("status") != "official"
        or contract.get("sha256") != SCORING_CONTRACT_SHA256
        or contract.get("release_id") != RELEASE_ID
        or sha256(path) != SCORING_CONTRACT_SHA256
    ):
        raise ValueError("DrivAerML profile scoring contract differs")


def retained_json(relative, digest, *, root=None, offline=None):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("profile truth path must stay within the pinned release")
    environment_root = os.environ.get("FLUIDSBENCH_DRIVAERML_TRUTH_ROOT")
    if offline is None:
        offline = bool(environment_root)
    root = (
        Path(root)
        if root is not None
        else (
            Path(environment_root)
            if environment_root
            else Path.home() / ".cache/fluidsbench/drivaerml" / INDEX_SHA256
        )
    )
    path = root / relative
    if not path.is_file():
        if offline:
            raise ValueError(f"missing offline profile truth: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TRUTH_BASE + relative, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("downloaded profile truth SHA-256 differs")
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"profile truth SHA-256 differs: {relative}")
    return json.loads(payload)


@lru_cache(maxsize=1)
def truth_index():
    return retained_json("index.json", INDEX_SHA256)


@lru_cache(maxsize=2)
def truth_chunk(relative, digest):
    return retained_json(relative, digest)


@lru_cache(maxsize=64)
def truth_case(case_id):
    entries = [x for x in truth_index()["case_locations"] if x["case_id"] == case_id]
    if len(entries) != 1:
        raise ValueError(f"case absent from pinned native truth: {case_id}")
    entry = entries[0]
    case = truth_chunk(entry["chunk_path"], entry["chunk_sha256"])["cases"][
        entry["case_offset"]
    ]
    if case["case_id"] != case_id:
        raise ValueError("native truth case identity differs")
    return case


def selected_series(case, scope):
    if scope not in {"surface_only", "surface_and_volume"}:
        raise ValueError("invalid prediction scope")
    return [
        s
        for s in case["series"]
        if scope != "surface_only" or s["panel_id"] == "pressure_profiles"
    ]


def prediction_series(truth, values=None):
    series = {key: copy.deepcopy(truth[key]) for key in COMMON}
    if truth["representation"] == "shared_alias":
        series["shared_support_ref"] = copy.deepcopy(truth["shared_support_ref"])
    else:
        for key in (
            "coordinate",
            "coordinate_id",
            "coordinate_unit",
            "support_identity_sha256",
        ):
            series[key] = copy.deepcopy(truth[key])
        series["prediction"] = [float(x) for x in values]
    return series


def weights(series):
    coordinate = np.asarray(series["coordinate"], dtype=float)
    result = np.zeros(len(coordinate), dtype=float)
    # Dev v2 uses the AutoCFD5 trapezoidal convention for both families.
    # A segment's first coordinate need not be zero. Never invent an edge
    # from zero to that first point, or bridge unsupported gaps.
    for segment in series["segments"]:
        start, stop = segment["emitted_index_start"], segment["emitted_index_stop"]
        edges = np.diff(coordinate[start:stop]) / 2
        if not np.isfinite(edges).all() or (edges <= 0).any():
            raise ValueError("native profile segment coordinates must increase")
        result[start : stop - 1] += edges
        result[start + 1 : stop] += edges
    if not np.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
        raise ValueError("native profile has invalid scoring weights")
    return result / result.sum()


def compare_profiles(case_id, series, scope):
    truth = selected_series(truth_case(case_id), scope)
    key = lambda s: (s["family_id"], s["station_id"])
    predictions = {key(s): s for s in series}
    if len(predictions) != len(series) or set(predictions) != {key(s) for s in truth}:
        raise ValueError(f"{case_id}: incomplete or duplicate native profile families")
    statistics, rmse = {}, {}
    for source in truth:
        observed = predictions[key(source)]
        values = observed.get("prediction")
        if prediction_series(source, values or []) != observed:
            raise ValueError(
                f"{case_id}/{key(source)}: profile metadata differs from native truth"
            )
        if source["representation"] == "shared_alias":
            continue
        predicted = np.asarray(values, dtype=float)
        target = np.asarray(source["value"], dtype=float)
        if predicted.shape != target.shape or not np.isfinite(predicted).all():
            raise ValueError(
                f"{case_id}/{key(source)}: incomplete or nonfinite predictions"
            )
        weight = weights(source)
        squared_error = float(np.sum(weight * (predicted - target) ** 2))
        stats = [
            squared_error,
            float(np.sum(weight * target)),
            float(np.sum(weight * target * target)),
        ]
        family = source["family_id"]
        statistics.setdefault(family, []).append(stats)
        rmse.setdefault(family, []).append(
            (source["station_id"], math.sqrt(squared_error))
        )
    return statistics, rmse


def evaluate_profiles(case_id, scope, surface_manifest, volume_manifest=None):
    case = truth_case(case_id)
    truth = selected_series(case, scope)
    gathered, bindings = {}, {}
    for panel, manifest, support, field, native in (
        (
            "pressure_profiles",
            surface_manifest,
            "surface_native_cells",
            "pMeanTrim",
            "native_boundary",
        ),
        (
            "velocity_profiles",
            volume_manifest,
            "volume_native_cells",
            "UMeanTrim",
            "native_volume",
        ),
    ):
        sources = [
            s
            for s in truth
            if s["panel_id"] == panel and s["representation"] == "materialized"
        ]
        if not sources:
            if manifest is not None:
                raise ValueError("surface_only must not supply volume predictions")
            continue
        if manifest is None:
            raise ValueError(f"missing {support} predictions")
        ids = np.array(
            sorted({x for s in sources for x in s["raw_native_cell_id"]}),
            dtype=np.int64,
        )
        field_values = gather_mapped_prediction_field(
            Path(manifest),
            ids,
            case_id=case_id,
            support_id=support,
            field_name=field,
            expected_total_row_count=case[native]["tuple_count"],
        )
        # SparseNativeField retains values in sorted raw-ID order.
        values = field_values.values
        gathered[panel] = {
            int(raw): value for raw, value in zip(ids, values, strict=True)
        }
        bindings[support] = {
            "manifest_sha256": sha256(manifest),
            "audit": field_values.audit_record(),
        }
    series = []
    for source in truth:
        values = None
        if source["representation"] == "materialized":
            raw = np.asarray(
                [gathered[source["panel_id"]][i] for i in source["raw_native_cell_id"]]
            )
            values = (
                2 * raw.reshape(-1) / UINF**2
                if source["panel_id"] == "pressure_profiles"
                else np.linalg.norm(raw, axis=1) / UINF
            )
        series.append(prediction_series(source, values))
    compare_profiles(case_id, series, scope)
    return {
        "schema": SCHEMA,
        "case_id": case_id,
        "prediction_scope": scope,
        "native_truth_index_sha256": INDEX_SHA256,
        "prediction_inputs": bindings,
        "profile_series": series,
    }


def write_profile_evidence(path, document):
    """Publish complete evidence atomically without replacing an earlier run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".profiles-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
