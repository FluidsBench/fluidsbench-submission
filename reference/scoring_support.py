"""Canonical scoring-support loading and exact prediction alignment.

The benchmark owns coordinates, targets, and weights.  A prediction artifact
contains only stable support IDs and predicted quantities.  Joining by those
IDs is exact: this module never interpolates or resamples ground truth.

Four location modes are part of the public interface:

* ``native_entities``
* ``materialized_table``
* ``structured_grid``
* ``reference_generator``

The generic implementation below intentionally loads only safe local
``materialized_table`` artifacts in JSON, CSV, or NPZ form. Dataset-specific
reference releases may add the other modes without changing the normalized
``ScoringSupport`` return type.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


LOCATION_MODES = frozenset(
    {"native_entities", "materialized_table", "structured_grid", "reference_generator"}
)
TABLE_FORMATS = frozenset({"json", "csv", "npz"})


class ScoringSupportError(ValueError):
    """Raised when benchmark support or predictions violate the contract."""


@dataclass(frozen=True)
class ScoringSupport:
    """Normalized benchmark-owned support for one case and support definition."""

    case_id: str
    support_name: str
    support_ids: np.ndarray
    coordinates: np.ndarray
    weights: np.ndarray
    targets: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ScoredPredictions:
    """Submitter predictions keyed by stable benchmark support IDs."""

    case_id: str
    support_name: str
    support_ids: np.ndarray
    values: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SupportRelease:
    """A semantically checked manifest/index/chunk release."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    supports: Mapping[str, Mapping[str, Any]]
    cases: Mapping[str, Mapping[str, Any]]
    case_directories: Mapping[str, Path]


class ScoringSupportLoader(Protocol):
    """Interface shared by all scoring-location modes."""

    def __call__(
        self,
        release: SupportRelease,
        case_id: str,
        support_name: str,
    ) -> ScoringSupport: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ScoringSupportError(f"cannot read JSON {path}: {error}") from error


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ScoringSupportError(f"{label} must be unique")


def _safe_relative(base: Path, value: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise ScoringSupportError(f"path escapes its declared directory: {value!r}") from error
    return candidate


def _check_identity(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ScoringSupportError(
                f"{label} {key} must equal {wanted!r}, got {value.get(key)!r}"
            )


def load_support_release(manifest_path: Path, case_set_id: str) -> SupportRelease:
    """Load and semantically verify a manifest → index → chunk chain."""

    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if "manifest_sha256" in manifest:
        raise ScoringSupportError("a scoring-support manifest must not contain its own SHA-256")

    support_items = manifest.get("supports")
    if not isinstance(support_items, list) or not support_items:
        raise ScoringSupportError("manifest supports must be a non-empty list")
    support_ids = [item.get("id") for item in support_items if isinstance(item, dict)]
    if len(support_ids) != len(support_items) or any(not isinstance(item, str) for item in support_ids):
        raise ScoringSupportError("every support definition requires a string id")
    _unique(support_ids, "support definition IDs")
    supports = dict(zip(support_ids, support_items, strict=True))

    metric_ids: list[str] = []
    for support_name, support in supports.items():
        location = support.get("location_definition", {})
        mode = location.get("mode") if isinstance(location, dict) else None
        if mode not in LOCATION_MODES:
            raise ScoringSupportError(f"{support_name} has unsupported location mode {mode!r}")
        quantities = support.get("quantities", [])
        quantity_ids = [quantity.get("id") for quantity in quantities if isinstance(quantity, dict)]
        if len(quantity_ids) != len(quantities):
            raise ScoringSupportError(f"{support_name} has an invalid quantity")
        _unique(quantity_ids, f"{support_name} quantity IDs")
        for quantity in quantities:
            components = quantity.get("components", [])
            component_ids = [
                component.get("id") for component in components if isinstance(component, dict)
            ]
            if len(component_ids) != len(components):
                raise ScoringSupportError(f"{support_name}/{quantity.get('id')} has an invalid component")
            _unique(component_ids, f"{support_name}/{quantity.get('id')} component IDs")
        for binding in support.get("metric_bindings", []):
            metric_id = binding.get("metric_id")
            if binding.get("quantity_id") not in quantity_ids:
                raise ScoringSupportError(
                    f"{support_name} metric {metric_id!r} references an unknown quantity"
                )
            metric_ids.append(metric_id)
    _unique(metric_ids, "metric bindings across scoring supports")

    case_sets = manifest.get("case_sets")
    if not isinstance(case_sets, list) or not case_sets:
        raise ScoringSupportError("manifest case_sets must be a non-empty list")
    case_set_ids = [item.get("id") for item in case_sets if isinstance(item, dict)]
    if len(case_set_ids) != len(case_sets):
        raise ScoringSupportError("every case set requires an id")
    _unique(case_set_ids, "case-set IDs")
    case_set = next((item for item in case_sets if item.get("id") == case_set_id), None)
    if case_set is None:
        raise ScoringSupportError(f"manifest does not define case set {case_set_id!r}")

    index_path = _safe_relative(manifest_path.parent, case_set["index_file"])
    if not index_path.is_file():
        raise ScoringSupportError(f"missing scoring-support case index: {index_path}")
    if sha256_file(index_path) != case_set["index_sha256"]:
        raise ScoringSupportError("scoring-support case index SHA-256 mismatch")
    index = _load_json(index_path)
    expected_identity = {
        "release_id": manifest.get("release_id"),
        "dataset_id": manifest.get("dataset_id"),
        "case_set_id": case_set_id,
    }
    _check_identity(index, expected_identity, "case index")

    chunks = index.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ScoringSupportError("case index chunks must be a non-empty list")
    chunk_files = [chunk.get("file") for chunk in chunks if isinstance(chunk, dict)]
    if len(chunk_files) != len(chunks):
        raise ScoringSupportError("every case-index chunk requires a file")
    _unique(chunk_files, "case-index chunk files")

    indexed_case_ids: list[str] = []
    loaded_case_ids: list[str] = []
    cases: dict[str, Mapping[str, Any]] = {}
    case_directories: dict[str, Path] = {}
    for descriptor in chunks:
        descriptor_ids = descriptor.get("case_ids", [])
        if descriptor.get("case_count") != len(descriptor_ids):
            raise ScoringSupportError(
                f"{descriptor.get('file')} case_count does not match its case_ids"
            )
        indexed_case_ids.extend(descriptor_ids)
        chunk_path = _safe_relative(index_path.parent, descriptor["file"])
        if not chunk_path.is_file():
            raise ScoringSupportError(f"missing scoring-support chunk: {chunk_path}")
        if sha256_file(chunk_path) != descriptor.get("sha256"):
            raise ScoringSupportError(f"{descriptor['file']} SHA-256 mismatch")
        chunk = _load_json(chunk_path)
        _check_identity(chunk, expected_identity, f"chunk {descriptor['file']}")
        chunk_cases = chunk.get("cases", [])
        chunk_case_ids = [
            case.get("case_id") for case in chunk_cases if isinstance(case, dict)
        ]
        if chunk_case_ids != descriptor_ids:
            raise ScoringSupportError(
                f"{descriptor['file']} case order does not match its index descriptor"
            )
        loaded_case_ids.extend(chunk_case_ids)
        for case in chunk_cases:
            case_id = case["case_id"]
            instances = case.get("support_instances", [])
            instance_ids = [
                instance.get("support_id") for instance in instances if isinstance(instance, dict)
            ]
            if len(instance_ids) != len(instances):
                raise ScoringSupportError(f"{case_id} has an invalid support instance")
            _unique(instance_ids, f"{case_id} support-instance IDs")
            if set(instance_ids) != set(support_ids):
                raise ScoringSupportError(
                    f"{case_id} support instances differ from manifest definitions"
                )
            for instance in instances:
                artifacts = instance.get("artifacts", [])
                roles = [
                    artifact.get("role")
                    for artifact in artifacts
                    if isinstance(artifact, dict)
                ]
                if len(roles) != len(artifacts):
                    raise ScoringSupportError(
                        f"{case_id}/{instance.get('support_id')} has an invalid artifact"
                    )
                _unique(roles, f"{case_id}/{instance.get('support_id')} artifact roles")
                for artifact in artifacts:
                    if "path" not in artifact:
                        continue
                    artifact_path = _safe_relative(chunk_path.parent, artifact["path"])
                    if not artifact_path.is_file():
                        raise ScoringSupportError(
                            f"missing local scoring-support artifact: {artifact_path}"
                        )
                    if sha256_file(artifact_path) != artifact.get("sha256"):
                        raise ScoringSupportError(
                            f"local scoring-support artifact SHA-256 mismatch: {artifact_path}"
                        )
            cases[case_id] = case
            case_directories[case_id] = chunk_path.parent

    _unique(indexed_case_ids, "case IDs across scoring-support chunks")
    if loaded_case_ids != indexed_case_ids:
        raise ScoringSupportError("loaded case order differs from the scoring-support index")
    if index.get("case_count") != len(indexed_case_ids):
        raise ScoringSupportError("case index case_count does not match indexed cases")
    if case_set.get("case_count") != len(indexed_case_ids):
        raise ScoringSupportError("manifest case-set count does not match indexed cases")

    return SupportRelease(
        manifest_path=manifest_path,
        manifest=manifest,
        supports=supports,
        cases=cases,
        case_directories=case_directories,
    )


def _columns_from_json(path: Path) -> dict[str, np.ndarray]:
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("columns"), dict):
        raw_columns = payload["columns"]
    else:
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            raise ScoringSupportError(
                f"{path} must contain a non-empty row list or a columns object"
            )
        keys = list(rows[0])
        if any(set(row) != set(keys) for row in rows):
            raise ScoringSupportError(f"{path} rows do not have identical fields")
        raw_columns = {key: [row[key] for row in rows] for key in keys}
    if not raw_columns:
        raise ScoringSupportError(f"{path} contains no columns")
    return {str(key): np.asarray(value) for key, value in raw_columns.items()}


def _columns_from_csv(path: Path) -> dict[str, np.ndarray]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError as error:
        raise ScoringSupportError(f"cannot read CSV {path}: {error}") from error
    if not reader.fieldnames or not rows:
        raise ScoringSupportError(f"{path} must contain a header and at least one row")
    return {
        name: np.asarray([row[name] for row in rows])
        for name in reader.fieldnames
    }


def _columns_from_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not payload.files:
                raise ScoringSupportError(f"{path} contains no arrays")
            return {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ScoringSupportError):
            raise
        raise ScoringSupportError(f"cannot read NPZ {path}: {error}") from error


def load_table_columns(path: Path, file_format: str) -> dict[str, np.ndarray]:
    """Load a safe local materialized table."""

    if file_format == "json":
        return _columns_from_json(path)
    if file_format == "csv":
        return _columns_from_csv(path)
    if file_format == "npz":
        return _columns_from_npz(path)
    if file_format == "parquet":
        raise ScoringSupportError(
            "Parquet is a declared interchange format but requires a dataset evaluator "
            "with a pinned Parquet dependency; the dependency-light generic loader accepts "
            "JSON, CSV, and NPZ"
        )
    raise ScoringSupportError(f"unsupported materialized table format {file_format!r}")


def _column(
    columns: Mapping[str, np.ndarray],
    name: str,
    *,
    numeric: bool,
    expected_count: int | None = None,
) -> np.ndarray:
    if name not in columns:
        raise ScoringSupportError(f"table is missing required field {name!r}")
    value = np.asarray(columns[name])
    if value.ndim != 1:
        raise ScoringSupportError(f"field {name!r} must be one-dimensional")
    if expected_count is not None and len(value) != expected_count:
        raise ScoringSupportError(f"field {name!r} has the wrong row count")
    if numeric:
        try:
            value = value.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ScoringSupportError(f"field {name!r} must be numeric") from error
        if not np.all(np.isfinite(value)):
            raise ScoringSupportError(f"field {name!r} contains non-finite values")
    return value


def _stable_ids(value: np.ndarray, label: str) -> np.ndarray:
    ids = np.asarray(value).astype(str)
    if ids.ndim != 1 or len(ids) == 0:
        raise ScoringSupportError(f"{label} support IDs must be a non-empty one-dimensional array")
    if any(not item for item in ids.tolist()):
        raise ScoringSupportError(f"{label} support IDs cannot be empty")
    if len(set(ids.tolist())) != len(ids):
        raise ScoringSupportError(f"{label} contains duplicate support IDs")
    return ids


def _case_instance(release: SupportRelease, case_id: str, support_name: str) -> Mapping[str, Any]:
    case = release.cases.get(case_id)
    if case is None:
        raise ScoringSupportError(f"unknown scoring-support case {case_id!r}")
    return next(
        instance
        for instance in case["support_instances"]
        if instance["support_id"] == support_name
    )


def _local_artifact(
    release: SupportRelease,
    case_id: str,
    instance: Mapping[str, Any],
    role: str,
) -> tuple[Path, str]:
    artifact = next(
        (item for item in instance["artifacts"] if item.get("role") == role),
        None,
    )
    if artifact is None:
        raise ScoringSupportError(
            f"{case_id}/{instance.get('support_id')} is missing artifact role {role!r}"
        )
    if "archive" in artifact:
        # Archive members are installed relative to the manifest, never the
        # current directory. Downloading/extracting is a separate, hash-checked
        # operation; evaluation cannot fetch mutable remote bytes implicitly.
        path = _safe_relative(release.manifest_path.parent, artifact["archive"]["member"])
    elif "path" in artifact:
        path = _safe_relative(release.case_directories[case_id], artifact["path"])
    else:
        raise ScoringSupportError(
            "the generic evaluator accepts local support artifacts only; fetch and verify "
            "the immutable release before evaluation"
        )
    if not path.is_file():
        raise ScoringSupportError(f"missing scoring-support artifact: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise ScoringSupportError(f"scoring-support artifact SHA-256 mismatch: {path}")
    return path, artifact["format"]


def load_materialized_support(
    release: SupportRelease,
    case_id: str,
    support_name: str,
) -> ScoringSupport:
    """Load benchmark-owned locations, targets, and weights from a materialized table."""

    definition = release.supports.get(support_name)
    if definition is None:
        raise ScoringSupportError(f"unknown support definition {support_name!r}")
    location = definition["location_definition"]
    if location.get("mode") != "materialized_table":
        raise ScoringSupportError(
            f"{support_name} uses {location.get('mode')!r}; the generic loader implements "
            "materialized_table only"
        )
    instance = _case_instance(release, case_id, support_name)
    path, artifact_format = _local_artifact(
        release, case_id, instance, location["artifact_role"]
    )
    if artifact_format != location["format"]:
        raise ScoringSupportError(
            f"{case_id}/{support_name} artifact format differs from its support definition"
        )
    columns = load_table_columns(path, artifact_format)
    id_rule = location["support_id_rule"]
    if id_rule.get("kind") != "artifact_field":
        raise ScoringSupportError(
            "materialized_table support requires support_id_rule.kind=artifact_field"
        )
    ids = _stable_ids(_column(columns, id_rule["field"], numeric=False), "ground truth")
    expected_count = len(ids)
    if instance.get("entity_count") != expected_count:
        raise ScoringSupportError(
            f"{case_id}/{support_name} entity_count does not match its table"
        )
    if ids.tolist() != sorted(ids.tolist()):
        raise ScoringSupportError(
            f"{case_id}/{support_name} support IDs must use support_id_ascending ordering"
        )
    coordinate_fields = location["coordinate_fields"]
    coordinates = np.column_stack(
        [
            _column(columns, field, numeric=True, expected_count=expected_count)
            for field in coordinate_fields
        ]
    )
    weight_rule = location["weight_rule"]
    if weight_rule["kind"] == "uniform":
        weights = np.ones(expected_count, dtype=np.float64)
    elif weight_rule["kind"] == "artifact_field":
        if weight_rule["artifact_role"] != location["artifact_role"]:
            raise ScoringSupportError(
                "generic materialized loader requires weights in the ground-truth table"
            )
        weights = _column(
            columns,
            weight_rule["field"],
            numeric=True,
            expected_count=expected_count,
        )
    else:
        raise ScoringSupportError(
            f"generic materialized loader cannot execute weight rule {weight_rule['kind']!r}"
        )
    if np.any(weights < 0) or not float(np.sum(weights)) > 0:
        raise ScoringSupportError("support weights must be non-negative with a positive sum")

    targets: dict[str, np.ndarray] = {}
    for quantity in definition["quantities"]:
        target_components = []
        for component in quantity["components"]:
            if component["target_association"] != "table_field":
                raise ScoringSupportError(
                    f"generic materialized loader requires table_field targets, got "
                    f"{component['target_association']!r}"
                )
            target_components.append(
                _column(
                    columns,
                    component["target_field"],
                    numeric=True,
                    expected_count=expected_count,
                )
            )
        targets[quantity["id"]] = np.column_stack(target_components)

    support = ScoringSupport(
        case_id=case_id,
        support_name=support_name,
        support_ids=ids,
        coordinates=coordinates,
        weights=np.asarray(weights, dtype=np.float64),
        targets=targets,
    )
    expected_digest = instance.get("expected_support_sha256")
    if expected_digest and normalized_support_sha256(support) != expected_digest:
        raise ScoringSupportError(
            f"{case_id}/{support_name} normalized support SHA-256 mismatch"
        )
    return support


LOADERS: dict[str, ScoringSupportLoader] = {
    "materialized_table": load_materialized_support,
}


def load_scoring_support(
    release: SupportRelease,
    case_id: str,
    support_name: str,
) -> ScoringSupport:
    """Load any registered location mode into the common support representation."""

    definition = release.supports.get(support_name)
    if definition is None:
        raise ScoringSupportError(f"unknown support definition {support_name!r}")
    mode = definition["location_definition"]["mode"]
    loader = LOADERS.get(mode)
    if loader is None:
        raise ScoringSupportError(
            f"location mode {mode!r} is declared by the interface but requires the "
            "dataset's pinned reference loader"
        )
    return loader(release, case_id, support_name)


def load_scored_predictions(
    path: Path,
    file_format: str,
    definition: Mapping[str, Any],
    *,
    case_id: str,
    support_name: str,
) -> ScoredPredictions:
    """Load submitter predictions; coordinates, targets, and weights are ignored."""

    columns = load_table_columns(path, file_format)
    id_rule = definition["location_definition"]["support_id_rule"]
    if id_rule.get("kind") != "artifact_field":
        raise ScoringSupportError(
            "generic prediction loading requires support_id_rule.kind=artifact_field"
        )
    ids = _stable_ids(_column(columns, id_rule["field"], numeric=False), "predictions")
    count = len(ids)
    values: dict[str, np.ndarray] = {}
    for quantity in definition["quantities"]:
        values[quantity["id"]] = np.column_stack(
            [
                _column(
                    columns,
                    component["prediction_field"],
                    numeric=True,
                    expected_count=count,
                )
                for component in quantity["components"]
            ]
        )
    return ScoredPredictions(
        case_id=case_id,
        support_name=support_name,
        support_ids=ids,
        values=values,
    )


def align_predictions(
    support: ScoringSupport,
    predictions: ScoredPredictions,
) -> Mapping[str, np.ndarray]:
    """Join predictions onto fixed benchmark support IDs with exact full coverage."""

    if predictions.case_id != support.case_id or predictions.support_name != support.support_name:
        raise ScoringSupportError("prediction case/support identity does not match ground truth")
    truth_ids = support.support_ids.tolist()
    predicted_ids = predictions.support_ids.tolist()
    truth_set = set(truth_ids)
    predicted_set = set(predicted_ids)
    missing = truth_set - predicted_set
    unknown = predicted_set - truth_set
    if missing or unknown or len(predicted_ids) != len(truth_ids):
        raise ScoringSupportError(
            "predictions must cover exactly the canonical support IDs; "
            f"missing={sorted(missing)[:5]}, unknown={sorted(unknown)[:5]}"
        )
    positions = {support_id: index for index, support_id in enumerate(predicted_ids)}
    order = np.asarray([positions[support_id] for support_id in truth_ids], dtype=np.int64)
    aligned: dict[str, np.ndarray] = {}
    for quantity_id, truth in support.targets.items():
        if quantity_id not in predictions.values:
            raise ScoringSupportError(f"predictions are missing quantity {quantity_id!r}")
        value = np.asarray(predictions.values[quantity_id], dtype=np.float64)[order]
        if value.shape != truth.shape:
            raise ScoringSupportError(
                f"prediction shape for {quantity_id!r} is {value.shape}, expected {truth.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ScoringSupportError(
                f"predictions for {quantity_id!r} contain non-finite values"
            )
        aligned[quantity_id] = value
    return aligned


def normalized_support_sha256(support: ScoringSupport) -> str:
    """Hash normalized support arrays without container-specific metadata."""

    digest = hashlib.sha256()

    def add_label(label: str) -> None:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    add_label("fluidsbench-normalized-support-v1")
    add_label(support.case_id)
    add_label(support.support_name)
    for support_id in support.support_ids.tolist():
        add_label(str(support_id))
    for label, value in [
        ("coordinates", support.coordinates),
        ("weights", support.weights),
    ]:
        add_label(label)
        array = np.asarray(value, dtype="<f8", order="C")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    for quantity_id in sorted(support.targets):
        add_label(quantity_id)
        array = np.asarray(support.targets[quantity_id], dtype="<f8", order="C")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a local scoring-support manifest/index/chunk chain."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-set", required=True)
    args = parser.parse_args()
    try:
        release = load_support_release(args.manifest, args.case_set)
    except ScoringSupportError as error:
        parser.error(str(error))
    print(
        f"PASS {release.manifest['dataset_id']}/{args.case_set}: "
        f"{len(release.cases)} cases, {len(release.supports)} supports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
