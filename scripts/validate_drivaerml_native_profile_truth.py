#!/usr/bin/env python3
"""Validate the retained native-truth export against real run_419 arrays.

This validator reads the generated ``cases/run_419.json`` itself.  It does not
accept a summary receipt as a substitute for the truth arrays.  The comparison
authorities are the retained real Transolver velocity evaluator output, the
current 40-series prediction fixture, and the retained Cp evaluator output.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_drivaerml_native_profile_truth as exporter  # noqa: E402


RUN419 = "run_419"
VELOCITY_EVIDENCE = (
    ROOT
    / "benchmark-specs/drivaerml/evidence/"
    "transolver-run419-relative-velocity-evaluator-smoke-v1.json"
)
CP_EVIDENCE = (
    ROOT
    / "benchmark-specs/drivaerml/evidence/"
    "transolver-run419-relative-cp-evaluator-smoke-v1.json"
)
PREDICTION_FIXTURE = (
    ROOT
    / "benchmark-specs/drivaerml/evidence/"
    "transolver-run419-current-relative-profile-v3/profiles/chunk-000.json"
)
EXPECTED_INPUT_SHA256 = {
    "velocity_evidence": (
        "b6c35ba11938506e5d4897a81ad459db1383e56472207d489d63a843ae4be417"
    ),
    "cp_evidence": (
        "70851f3b1204255773f895a725156cb33d6c3bff7fabe6995c0ab2c64e552e5f"
    ),
    "prediction_fixture": (
        "3a5bca4c5f8730dcae4e29f69f415dc393050c6ebca6cf6e1303d307985bf262"
    ),
}
EXPECTED_SELECTED_VALUES_SHA256 = (
    "a1cd9c5bad71b720e6434fbb821aa480fc2f7555516375329bfd02ced43752d0"
)
EXPECTED_NATIVE_VOLUME = {
    "tuple_count": 121_635_947,
    "declared_payload_bytes": 1_459_631_364,
    "retained_full_payload_sha256": (
        "54d4286234b5dab04f41678b99463ff6a9abe622625dbc34ca6ca73e64c6d580"
    ),
    "selected_unique_raw_cell_id_count": 6_038,
    "selected_values_sha256": EXPECTED_SELECTED_VALUES_SHA256,
}
EXPECTED_NATIVE_BOUNDARY = {
    "logical_path": "run_419/boundary_419.vtp",
    "logical_size_bytes": 543_851_361,
    "source_sha256": (
        "e6ee75027b0dd87d44fc3f49188ecec95a5193ac0de148babe9dca95424da1c5"
    ),
    "tuple_count": 7_284_102,
    "declared_payload_bytes": 29_136_408,
    "referenced_producer_row_count": 6_921,
    "selected_unique_raw_polygon_id_count": 5_855,
    "selected_values_sha256": (
        "83da391858c806fa0b1b092549d6f19ca99a32e3f7275829516f6c0e9c9f4c9d"
    ),
}
HISTORICAL_CP_MEAN_RMSE = 0.01551664229339102
CP_COUNT_BY_STATION = {
    "upperbody_centerline": 1434,
    "underbody_centerline": 1470,
    "sidewall_z_0_15": 1953,
    "front_left_wheelhouse_y_neg_0_6": 595,
}
RUN419_CP_DISPLAY_RANGE = {
    "upperbody_centerline": (-0.8331868945998707, 3.6396731134620603),
    "underbody_centerline": (-0.8331869840450868, 3.639718889923447),
    "sidewall_z_0_15": (-0.8208888126595923, 3.6357615293629575),
    "front_left_wheelhouse_y_neg_0_6": (
        -0.3792471834540514,
        0.4087099569235014,
    ),
    "sidewall_front_wheelhouse_relative": (
        -0.8209005267002477,
        3.6357573498755116,
    ),
    "front_left_wheelhouse_relative": (
        -0.35074854500754504,
        0.39064587559344227,
    ),
}
RUN419_CP_DISPLAY_IDENTITY = {
    "upperbody_centerline": {
        "count": 1434,
        "sha256": "8407666b92d559f699117d56e86c617ff8b085f275443cab601c41d7ac172748",
    },
    "underbody_centerline": {
        "count": 1470,
        "sha256": "53032020bf0b166361432d3d403851a560466713148c09acfc14f620bb28a8e7",
    },
    "sidewall_z_0_15": {
        "count": 1953,
        "sha256": "7056a50589527d5aba7f6369576f49a77a900df5c61ae8639332bb464cf7e31f",
    },
    "front_left_wheelhouse_y_neg_0_6": {
        "count": 595,
        "sha256": "1db6ba671f6c2635cab43e4d837aa008aa7d5003de6c8bd5f9df07edffd8db28",
    },
    "sidewall_front_wheelhouse_relative": {
        "count": 886,
        "sha256": "99a0085d61ef021708ef671d9c088208476c8a788d15ebbff33ee381fbe91e45",
    },
    "front_left_wheelhouse_relative": {
        "count": 583,
        "sha256": "8a1811385f25c403f934344b96ad09a09f1f01b31554f155a2094da030366415",
    },
}
RECEIPT_SCHEMA = "fluidsbench-drivaerml-native-profile-truth-run419-validation-v2"


class TruthValidationError(ValueError):
    """Raised when retained output differs from real run_419 evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TruthValidationError(message)


def _load(path: Path, label: str, expected_sha256: str | None = None) -> tuple[dict, bytes]:
    document, payload = exporter.load_json(path, label)
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise TruthValidationError(f"{label} SHA-256 differs")
    return document, payload


def _finite_array(value: object, label: str) -> list[float]:
    if not isinstance(value, list):
        raise TruthValidationError(f"{label} must be an array")
    result: list[float] = []
    for position, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TruthValidationError(f"{label}[{position}] must be numeric")
        item = float(raw)
        if not math.isfinite(item):
            raise TruthValidationError(f"{label}[{position}] must be finite")
        result.append(item)
    return result


def _strictly_increasing(values: Sequence[float], label: str) -> None:
    if not values or any(right <= left for left, right in zip(values, values[1:])):
        raise TruthValidationError(f"{label} must be non-empty and strictly increasing")


def _series_map(case: Mapping[str, object], label: str) -> dict[tuple[str, str], dict]:
    raw = case.get("series")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise TruthValidationError(f"{label} series must be an object array")
    result = {
        (str(item.get("family_id")), str(item.get("station_id"))): item
        for item in raw
    }
    if len(result) != len(raw):
        raise TruthValidationError(f"{label} contains duplicate family/station series")
    return result


def _validate_run419_cp_display(
    station: str,
    display: Sequence[float],
    supplied_identity: object,
    label: str,
) -> None:
    expected = RUN419_CP_DISPLAY_IDENTITY.get(station)
    _require(expected is not None, f"{label} display identity authority is missing")
    actual_identity = exporter.coordinate_array_identity_sha256(display)
    _require(
        len(display) == expected["count"]
        and supplied_identity == actual_identity == expected["sha256"],
        f"{label} display coordinate differs from the exact retained producer midpoint array",
    )


def _safe_output_file(output_dir: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str):
        raise TruthValidationError(f"{label} path must be a string")
    candidate = (output_dir / relative).resolve()
    try:
        candidate.relative_to(output_dir.resolve())
    except ValueError as error:
        raise TruthValidationError(f"{label} escapes the output directory") from error
    return candidate


def _validate_generated_series(case: Mapping[str, object]) -> None:
    series = case.get("series")
    _require(isinstance(series, list) and len(series) == 40, "run_419 must contain 40 series")
    _require(
        [(item.get("family_id"), item.get("station_id")) for item in series]
        == exporter._expected_series_keys(),
        "run_419 series namespace/order differs",
    )
    expected_representations = ["materialized"] * 36 + [
        "shared_alias",
        "shared_alias",
        "materialized",
        "materialized",
    ]
    materialized_by_key: dict[tuple[str, str], dict] = {}
    aliases: list[dict] = []
    for item, representation in zip(series, expected_representations):
        key = f"{item.get('family_id')}/{item.get('station_id')}"
        _require(item.get("representation") == representation, f"{key} representation differs")
        common_fields = {
            "panel_id",
            "family_id",
            "placement_mode",
            "station_id",
            "quantity_id",
            "quantity",
            "units",
            "scoring_role",
            "representation",
            "placement_receipt_identity_sha256",
            "series_identity_sha256",
        }
        if representation == "shared_alias":
            expected_fields = common_fields | {"shared_support_ref"}
        else:
            expected_fields = common_fields | {
                "support_identity_sha256",
                "coordinate_id",
                "coordinate_unit",
                "coordinate_identity_sha256",
                "value_identity_sha256",
                "sample_index",
                "raw_native_cell_id",
                "coordinate",
                "value",
                "segments",
                "unsupported_samples",
            }
            if item.get("quantity_id") == "cp":
                expected_fields |= {
                    "display_coordinate_id",
                    "display_coordinate_unit",
                    "display_coordinate_identity_sha256",
                    "display_coordinate",
                }
        _require(set(item) == expected_fields, f"{key} native-v3 fields differ")
        supplied = item.get("series_identity_sha256")
        body = dict(item)
        body.pop("series_identity_sha256", None)
        _require(supplied == exporter.series_identity_sha256(body), f"{key} identity differs")
        if representation == "shared_alias":
            aliases.append(item)
            _require(
                not any(
                    field in item
                    for field in (
                        "coordinate",
                        "display_coordinate",
                        "value",
                        "raw_native_cell_id",
                    )
                ),
                f"{key} alias duplicates materialized arrays",
            )
            continue
        coordinates = _finite_array(item.get("coordinate"), f"{key} coordinates")
        values = _finite_array(item.get("value"), f"{key} values")
        samples = item.get("sample_index")
        raw_ids = item.get("raw_native_cell_id")
        _require(
            isinstance(samples, list)
            and isinstance(raw_ids, list)
            and len(samples) == len(raw_ids) == len(coordinates) == len(values) > 0,
            f"{key} materialized arrays are not aligned",
        )
        _require(
            item.get("coordinate_identity_sha256")
            == exporter.coordinate_array_identity_sha256(coordinates),
            f"{key} coordinate identity differs",
        )
        _require(
            item.get("value_identity_sha256")
            == exporter.binary64_array_identity_sha256(values),
            f"{key} value identity differs",
        )
        if item.get("quantity_id") == "cp":
            display = _finite_array(
                item.get("display_coordinate"), f"{key} display coordinates"
            )
            _require(
                item.get("coordinate_id") == "arc_length_m"
                and item.get("coordinate_unit") == "m"
                and item.get("display_coordinate_id")
                == exporter.CP_DISPLAY_COORDINATE_ID
                and item.get("display_coordinate_unit")
                == exporter.CP_DISPLAY_COORDINATE_UNIT
                and len(display) == len(coordinates)
                and item.get("display_coordinate_identity_sha256")
                == exporter.coordinate_array_identity_sha256(display),
                f"{key} display coordinate binding differs",
            )
            expected_range = RUN419_CP_DISPLAY_RANGE.get(str(item.get("station_id")))
            _require(expected_range is not None, f"{key} display range authority is missing")
            _validate_run419_cp_display(
                str(item.get("station_id")),
                display,
                item.get("display_coordinate_identity_sha256"),
                key,
            )
            _require(
                min(display) == expected_range[0] and max(display) == expected_range[1],
                f"{key} physical-x range differs",
            )
        else:
            _require(
                not any(
                    field in item
                    for field in (
                        "display_coordinate_id",
                        "display_coordinate_unit",
                        "display_coordinate_identity_sha256",
                        "display_coordinate",
                    )
                ),
                f"{key} velocity series unexpectedly has display coordinates",
            )
        materialized_by_key[(str(item.get("family_id")), str(item.get("station_id")))] = item
    for alias in aliases:
        reference = alias.get("shared_support_ref")
        _require(isinstance(reference, dict), "Cp alias reference differs")
        canonical_key = (
            str(reference.get("canonical_family_id")),
            str(reference.get("canonical_station_id")),
        )
        canonical = materialized_by_key.get(canonical_key)
        _require(canonical is not None, f"{canonical_key} alias target is missing")
        _require(
            reference.get("canonical_support_identity_sha256")
            == canonical.get("support_identity_sha256"),
            f"{canonical_key} alias support differs",
        )


def _validate_release(output_dir: Path) -> tuple[dict, dict, dict[str, object]]:
    case_path = output_dir / "cases" / f"{RUN419}.json"
    case, case_payload = _load(case_path, "generated run_419 case")
    _require(case_payload == exporter.canonical_json_bytes(case), "run_419 is not canonical JSON")
    _require(case.get("schema") == exporter.CASE_SCHEMA, "run_419 case schema differs")
    _require(case.get("truth_source") == exporter.TRUTH_SOURCE, "run_419 truth source differs")
    _require(case.get("relative_scoring_activated") is False, "run_419 activates scoring")
    exporter.document_identity_sha256(case, "case_identity")
    _validate_generated_series(case)

    index, index_payload = _load(output_dir / "index.json", "native truth index")
    _require(index.get("schema") == exporter.INDEX_SCHEMA, "native truth index schema differs")
    _require(index.get("case_count") == 484, "native truth index must cover 484 cases")
    _require(index.get("truth_source") == exporter.TRUTH_SOURCE, "index truth source differs")
    _require(index.get("relative_scoring_activated") is False, "index activates scoring")
    exporter.document_identity_sha256(index, "index_identity")
    locations = [
        item
        for item in index.get("case_locations", [])
        if isinstance(item, dict) and item.get("case_id") == RUN419
    ]
    _require(len(locations) == 1, "index must locate run_419 exactly once")
    location = locations[0]
    chunk_path = _safe_output_file(output_dir, location.get("chunk_path"), "run_419 chunk")
    chunk, chunk_payload = _load(
        chunk_path, "run_419 chunk", str(location.get("chunk_sha256"))
    )
    _require(chunk.get("truth_source") == exporter.TRUTH_SOURCE, "chunk truth source differs")
    exporter.document_identity_sha256(chunk, "chunk_identity")
    offset = location.get("case_offset")
    cases = chunk.get("cases")
    _require(
        isinstance(offset, int)
        and not isinstance(offset, bool)
        and isinstance(cases, list)
        and 0 <= offset < len(cases)
        and cases[offset] == case,
        "chunked run_419 differs from the generated case artifact",
    )

    provenance, provenance_payload = _load(output_dir / "provenance.json", "provenance")
    release, release_payload = _load(output_dir / "release-receipt.json", "release receipt")
    exporter.document_identity_sha256(provenance, "provenance_identity")
    exporter.document_identity_sha256(release, "release_identity")
    for label, document in (("provenance", provenance), ("release", release)):
        _require(document.get("truth_source") == exporter.TRUTH_SOURCE, f"{label} truth source differs")
        _require(
            document.get("relative_scoring_activated") is False,
            f"{label} activates relative scoring",
        )
    _require(release.get("submissions_opened") is False, "release opens submissions")
    _require(release.get("owner_approval_complete") is False, "release claims owner approval")
    return case, index, {
        "case": {
            "path": f"cases/{RUN419}.json",
            "sha256": hashlib.sha256(case_payload).hexdigest(),
            "size_bytes": len(case_payload),
        },
        "chunk": {
            "path": str(location["chunk_path"]),
            "sha256": hashlib.sha256(chunk_payload).hexdigest(),
            "size_bytes": len(chunk_payload),
        },
        "index": {
            "path": "index.json",
            "sha256": hashlib.sha256(index_payload).hexdigest(),
            "size_bytes": len(index_payload),
        },
        "provenance": {
            "path": "provenance.json",
            "sha256": hashlib.sha256(provenance_payload).hexdigest(),
            "size_bytes": len(provenance_payload),
        },
        "release": {
            "path": "release-receipt.json",
            "sha256": hashlib.sha256(release_payload).hexdigest(),
            "size_bytes": len(release_payload),
        },
    }


def _expected_runs(sample_indices: Sequence[int]) -> list[tuple[int, int, int, int]]:
    if not sample_indices:
        return []
    output: list[tuple[int, int, int, int]] = []
    emitted_start = 0
    for position in range(1, len(sample_indices) + 1):
        if position == len(sample_indices) or sample_indices[position] != sample_indices[position - 1] + 1:
            output.append(
                (
                    emitted_start,
                    position,
                    sample_indices[emitted_start],
                    sample_indices[position - 1] + 1,
                )
            )
            emitted_start = position
    return output


def validate_velocity_families(
    generated_case: Mapping[str, object],
    velocity_evidence: Mapping[str, object],
    prediction_case: Mapping[str, object],
) -> list[dict[str, object]]:
    generated = _series_map(generated_case, "generated run_419")
    predictions = _series_map(prediction_case, "prediction fixture")
    families = velocity_evidence.get("families")
    _require(isinstance(families, list) and len(families) == 2, "velocity evidence families differ")
    expected_families = {
        exporter.CONSTANT_VELOCITY_FAMILY: "constant",
        exporter.RELATIVE_VELOCITY_FAMILY: "relative",
    }
    _require(
        {item.get("family_id"): item.get("placement_mode") for item in families}
        == expected_families,
        "velocity family namespace differs",
    )
    results: list[dict[str, object]] = []
    for family in families:
        family_id = str(family["family_id"])
        placement = str(family["placement_mode"])
        profiles = family.get("profiles")
        _require(
            isinstance(profiles, list)
            and [item.get("profile_id") for item in profiles] == list(exporter.VELOCITY_STATIONS),
            f"{family_id} station set/order differs",
        )
        valid_total = 0
        invalid_total = 0
        for profile in profiles:
            profile_id = str(profile["profile_id"])
            station_id = (
                f"autocfd5_{profile_id.lower()}" if placement == "constant" else profile_id
            )
            key = (family_id, station_id)
            output = generated.get(key)
            prediction = predictions.get(key)
            _require(output is not None and prediction is not None, f"{family_id}/{station_id} is missing")
            valid_series = profile.get("valid_series")
            _require(isinstance(valid_series, dict), f"{family_id}/{station_id} evidence differs")
            samples = valid_series.get("sample_index")
            _require(
                isinstance(samples, list)
                and all(isinstance(item, int) and not isinstance(item, bool) for item in samples),
                f"{family_id}/{station_id} sample indexes differ",
            )
            _require(output.get("sample_index") == samples, f"{family_id}/{station_id} gaps differ")
            _require(
                output.get("value") == valid_series.get("truth_velocity_ratio"),
                f"{family_id}/{station_id} native truth array differs",
            )
            _require(
                prediction.get("prediction") == valid_series.get("prediction_velocity_ratio"),
                f"{family_id}/{station_id} Transolver prediction differs",
            )
            _require(
                prediction.get("coordinate") == output.get("coordinate"),
                f"{family_id}/{station_id} current prediction coordinates differ",
            )
            evidence_distance = _finite_array(
                valid_series.get("distance_m"), f"{family_id}/{station_id} evidence distance"
            )
            _strictly_increasing(evidence_distance, f"{family_id}/{station_id} evidence distance")
            if placement == "constant":
                _require(
                    output.get("coordinate") == valid_series.get("distance_m"),
                    f"{family_id}/{station_id} constant coordinates differ",
                )
            else:
                denominator = int(profile["sample_count"]) - 1
                expected_coordinates = [sample / denominator for sample in samples]
                _require(
                    output.get("coordinate") == expected_coordinates,
                    f"{family_id}/{station_id} normalized coordinates differ",
                )

            sample_count = int(profile["sample_count"])
            complement = sorted(set(range(sample_count)) - set(samples))
            unsupported = output.get("unsupported_samples")
            _require(isinstance(unsupported, list), f"{family_id}/{station_id} gaps are missing")
            _require(
                [item.get("sample_index") for item in unsupported] == complement,
                f"{family_id}/{station_id} unsupported sample set differs",
            )
            observed_reasons = collections.Counter(
                str(item.get("reason")) for item in unsupported
            )
            _require(
                dict(observed_reasons) == profile.get("invalid_reason_counts"),
                f"{family_id}/{station_id} unsupported reasons differ",
            )
            _require(
                len(samples) == profile.get("valid_count")
                and len(complement) == profile.get("invalid_count"),
                f"{family_id}/{station_id} coverage counts differ",
            )
            expected_runs = _expected_runs(samples)
            segments = output.get("segments")
            arcs = profile.get("contiguous_valid_arcs")
            _require(
                isinstance(segments, list)
                and isinstance(arcs, list)
                and len(segments) == len(arcs) == len(expected_runs),
                f"{family_id}/{station_id} contiguous gap topology differs",
            )
            coordinates = output["coordinate"]
            for segment, arc, expected in zip(segments, arcs, expected_runs):
                emitted_start, emitted_stop, sample_start, sample_stop = expected
                _require(
                    segment
                    == {
                        "segment_id": "supported",
                        "emitted_index_start": emitted_start,
                        "emitted_index_stop": emitted_stop,
                        "sample_index_start": sample_start,
                        "sample_index_stop": sample_stop,
                        "coordinate_start": coordinates[emitted_start],
                        "coordinate_stop": coordinates[emitted_stop - 1],
                    },
                    f"{family_id}/{station_id} segment descriptor differs",
                )
                _require(
                    arc.get("sample_index_start") == sample_start
                    and arc.get("sample_index_stop_exclusive") == sample_stop
                    and arc.get("sample_count") == emitted_stop - emitted_start
                    and arc.get("distance_m_start") == evidence_distance[emitted_start]
                    and arc.get("distance_m_stop") == evidence_distance[emitted_stop - 1],
                    f"{family_id}/{station_id} retained arc differs",
                )
            valid_total += len(samples)
            invalid_total += len(complement)
        _require(
            valid_total == family.get("valid_count")
            and invalid_total == family.get("invalid_count")
            and valid_total + invalid_total == family.get("sample_count"),
            f"{family_id} aggregate coverage differs",
        )
        results.append(
            {
                "family_id": family_id,
                "placement_mode": placement,
                "station_count": 16,
                "sample_count": valid_total + invalid_total,
                "valid_count": valid_total,
                "unsupported_count": invalid_total,
                "native_truth_arrays_exact": True,
                "coordinates_and_gap_topology_exact": True,
            }
        )
    return results


def cumulative_segment_weighted_rmse(
    coordinate_end: Sequence[object],
    prediction: Sequence[object],
    truth: Sequence[object],
) -> tuple[float, float]:
    coordinates = _finite_array(list(coordinate_end), "Cp cumulative coordinates")
    predicted = _finite_array(list(prediction), "Cp prediction")
    expected = _finite_array(list(truth), "Cp truth")
    if len(coordinates) < 2 or len(predicted) != len(coordinates) or len(expected) != len(coordinates):
        raise TruthValidationError("Cp arrays must be aligned and contain at least two rows")
    _strictly_increasing(coordinates, "Cp cumulative coordinates")
    if coordinates[0] <= 0.0:
        raise TruthValidationError("first Cp cumulative coordinate must be positive")
    weights = [coordinates[0], *[right - left for left, right in zip(coordinates, coordinates[1:])]]
    total = math.fsum(weights)
    weighted_sse = math.fsum(
        weight * (predicted_value - truth_value) ** 2
        for weight, predicted_value, truth_value in zip(weights, predicted, expected)
    )
    result = math.sqrt(weighted_sse / total)
    if not math.isfinite(result):
        raise TruthValidationError("Cp segment-weighted RMSE is not finite")
    return result, total


def validate_constant_cp(
    generated_case: Mapping[str, object],
    cp_evidence: Mapping[str, object],
    prediction_case: Mapping[str, object],
) -> tuple[list[dict[str, object]], float]:
    generated = _series_map(generated_case, "generated run_419")
    predictions = _series_map(prediction_case, "prediction fixture")
    diagnostic = cp_evidence.get("constant_candidate_diagnostic")
    _require(isinstance(diagnostic, dict), "constant Cp evidence differs")
    historical = {
        item.get("station_id"): item
        for item in diagnostic.get("cuts", [])
        if isinstance(item, dict)
    }
    _require(set(historical) == set(CP_COUNT_BY_STATION), "historical Cp station set differs")
    results: list[dict[str, object]] = []
    rmse_values: list[float] = []
    for station, count in CP_COUNT_BY_STATION.items():
        key = (exporter.CONSTANT_CP_FAMILY, station)
        truth = generated.get(key)
        prediction = predictions.get(key)
        _require(truth is not None and prediction is not None, f"constant Cp {station} is missing")
        _require(
            truth.get("coordinate") == prediction.get("coordinate"),
            f"constant Cp {station} prediction coordinates differ",
        )
        _require(
            truth.get("support_identity_sha256")
            == prediction.get("support_identity_sha256")
            == historical[station].get("support_identity_sha256"),
            f"constant Cp {station} support identity differs",
        )
        _require(
            truth.get("placement_receipt_identity_sha256")
            == prediction.get("placement_receipt_identity_sha256"),
            f"constant Cp {station} placement receipt differs",
        )
        coordinates = truth.get("coordinate")
        values = truth.get("value")
        predicted = prediction.get("prediction")
        _require(
            isinstance(coordinates, list)
            and isinstance(values, list)
            and isinstance(predicted, list)
            and len(coordinates) == len(values) == len(predicted) == count
            and truth.get("sample_index") == list(range(count)),
            f"constant Cp {station} arrays/count differ",
        )
        rmse, total_length = cumulative_segment_weighted_rmse(
            coordinates, predicted, values
        )
        _require(
            historical[station].get("segment_count") == count
            and historical[station].get("total_length_m") == total_length,
            f"constant Cp {station} retained length/count differ",
        )
        historical_rmse = float(historical[station]["segment_length_weighted_rmse"])
        _require(
            math.isclose(rmse, historical_rmse, rel_tol=0.0, abs_tol=2.0e-9),
            f"constant Cp {station} recomputed RMSE differs from evaluator evidence",
        )
        rmse_values.append(rmse)
        results.append(
            {
                "station_id": station,
                "segment_count": count,
                "total_length_m": total_length,
                "segment_length_weighted_rmse": rmse,
                "historical_evaluator_rmse": historical_rmse,
                "absolute_delta_from_historical": abs(rmse - historical_rmse),
                "display_coordinate_id": truth["display_coordinate_id"],
                "display_coordinate_identity_sha256": truth[
                    "display_coordinate_identity_sha256"
                ],
                "physical_x_min_m": min(truth["display_coordinate"]),
                "physical_x_max_m": max(truth["display_coordinate"]),
                "prediction_uses_reference_display_coordinate": True,
                "scoring_arc_coordinate_unchanged": True,
            }
        )
    mean_rmse = math.fsum(rmse_values) / len(rmse_values)
    _require(
        math.isclose(mean_rmse, HISTORICAL_CP_MEAN_RMSE, rel_tol=0.0, abs_tol=2.0e-9),
        "recomputed four-cut Transolver mean RMSE differs",
    )
    return results, mean_rmse


def cross_bind_all_prediction_series(
    generated_case: Mapping[str, object], prediction_case: Mapping[str, object]
) -> None:
    """Bind every current prediction series to the independently generated truth grid."""

    generated_map = _series_map(generated_case, "generated run_419")
    prediction_map = _series_map(prediction_case, "prediction fixture")
    _require(set(generated_map) == set(prediction_map), "40-series fixture namespace differs")
    _require(len(generated_map) == 40, "fixture must contain exactly 40 series")
    for key, generated in generated_map.items():
        predicted = prediction_map[key]
        for field in (
            "panel_id",
            "family_id",
            "placement_mode",
            "station_id",
            "quantity_id",
            "representation",
            "scoring_role",
            "placement_receipt_identity_sha256",
        ):
            _require(
                generated.get(field) == predicted.get(field),
                f"{key} fixture {field} differs",
            )
        if generated.get("representation") == "materialized":
            predicted_coordinates = _finite_array(
                predicted.get("coordinate"), f"{key} prediction coordinates"
            )
            predicted_values = _finite_array(
                predicted.get("prediction"), f"{key} predictions"
            )
            _require(
                generated.get("support_identity_sha256")
                == predicted.get("support_identity_sha256")
                and generated.get("coordinate_id") == predicted.get("coordinate_id")
                and generated.get("coordinate_unit")
                == predicted.get("coordinate_unit")
                and generated.get("coordinate") == predicted_coordinates
                and len(generated.get("value", []))
                == len(predicted_coordinates)
                == len(predicted_values),
                f"{key} fixture support/coordinate/prediction binding differs",
            )
        else:
            _require(
                generated.get("shared_support_ref") == predicted.get("shared_support_ref"),
                f"{key} shared alias differs",
            )


def validate(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    generated_case, index, output_bindings = _validate_release(output_dir)
    velocity, velocity_payload = _load(
        VELOCITY_EVIDENCE,
        "run_419 velocity evidence",
        EXPECTED_INPUT_SHA256["velocity_evidence"],
    )
    cp, cp_payload = _load(
        CP_EVIDENCE, "run_419 Cp evidence", EXPECTED_INPUT_SHA256["cp_evidence"]
    )
    prediction, prediction_payload = _load(
        PREDICTION_FIXTURE,
        "run_419 current prediction fixture",
        EXPECTED_INPUT_SHA256["prediction_fixture"],
    )
    prediction_cases = prediction.get("cases")
    _require(
        isinstance(prediction_cases, list)
        and len(prediction_cases) == 1
        and prediction_cases[0].get("case_id") == RUN419,
        "current prediction fixture must contain exactly run_419",
    )
    prediction_case = prediction_cases[0]
    cross_bind_all_prediction_series(generated_case, prediction_case)

    native_volume = generated_case.get("native_volume")
    _require(isinstance(native_volume, dict), "generated native volume audit is missing")
    for field, expected in EXPECTED_NATIVE_VOLUME.items():
        _require(
            native_volume.get(field) == expected,
            f"generated native volume {field} differs",
        )
    selected = native_volume.get("selected_values_sha256")
    _require(selected == EXPECTED_SELECTED_VALUES_SHA256, "generated selected U identity differs")
    _require(
        selected == velocity.get("native_truth", {}).get("selected_values_sha256"),
        "generated selected U identity differs from real evaluator evidence",
    )
    native_boundary = generated_case.get("native_boundary")
    _require(isinstance(native_boundary, dict), "generated native boundary audit is missing")
    for field, expected in EXPECTED_NATIVE_BOUNDARY.items():
        _require(
            native_boundary.get(field) == expected,
            f"generated native boundary {field} differs",
        )
    _require(
        native_boundary.get("full_file_verification")
        == "sha256_recomputed_against_native_source_pin"
        and native_boundary.get("producer_value_crosscheck")
        == "exact_float_equality_all_referenced_rows",
        "generated native boundary verification declaration differs",
    )
    velocity_results = validate_velocity_families(
        generated_case, velocity, prediction_case
    )
    cp_results, cp_mean = validate_constant_cp(generated_case, cp, prediction_case)

    return exporter.identity_bound_document(
        {
            "schema": RECEIPT_SCHEMA,
            "schema_version": 2,
            "status": "passed_real_run419_array_level_validation",
            "case_id": RUN419,
            "truth_source": dict(exporter.TRUTH_SOURCE),
            "dataset_revision": exporter.DATASET_REVISION,
            "evaluator_git_revision": generated_case["generator"]["evaluator_git_revision"],
            "relative_scoring_activated": False,
            "submissions_opened": False,
            "generated_output_bindings": output_bindings,
            "comparison_inputs": {
                "velocity_evidence": {
                    "path": str(VELOCITY_EVIDENCE.relative_to(ROOT)),
                    "sha256": hashlib.sha256(velocity_payload).hexdigest(),
                    "size_bytes": len(velocity_payload),
                },
                "cp_evidence": {
                    "path": str(CP_EVIDENCE.relative_to(ROOT)),
                    "sha256": hashlib.sha256(cp_payload).hexdigest(),
                    "size_bytes": len(cp_payload),
                },
                "prediction_fixture": {
                    "path": str(PREDICTION_FIXTURE.relative_to(ROOT)),
                    "sha256": hashlib.sha256(prediction_payload).hexdigest(),
                    "size_bytes": len(prediction_payload),
                },
                "validator_source": {
                    "path": "scripts/validate_drivaerml_native_profile_truth.py",
                    "sha256": exporter.sha256_file(Path(__file__).resolve()),
                    "size_bytes": Path(__file__).resolve().stat().st_size,
                },
            },
            "selected_native_velocity_values_sha256": selected,
            "selected_native_boundary_values_sha256": native_boundary[
                "selected_values_sha256"
            ],
            "native_volume_audit": dict(EXPECTED_NATIVE_VOLUME),
            "native_boundary_audit": dict(EXPECTED_NATIVE_BOUNDARY),
            "velocity_family_results": velocity_results,
            "constant_cp_results": cp_results,
            "constant_cp_case_equal_cut_mean_rmse": cp_mean,
            "historical_cp_case_equal_cut_mean_rmse": HISTORICAL_CP_MEAN_RMSE,
            "cp_mean_absolute_delta_from_historical": abs(
                cp_mean - HISTORICAL_CP_MEAN_RMSE
            ),
            "claims": {
                "generated_run419_arrays_loaded": True,
                "stored_summary_substituted_for_arrays": False,
                "all_40_series_cross_bound": True,
                "both_velocity_families_independently_verified": True,
                "cp_rmse_recomputed_from_truth_and_prediction_arrays": True,
                "cp_physical_x_bound_to_retained_cut_segment_midpoints": True,
                "current_prediction_format_requires_no_display_coordinate": True,
                "arc_length_scoring_coordinate_unchanged": True,
                "scientific_activation": False,
            },
        },
        "validation_identity",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate(args.output_dir)
        binding = None
        if args.receipt is not None:
            binding = exporter.write_canonical(
                args.receipt.expanduser().resolve(), result, check=args.check
            )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "case_id": RUN419,
                    "selected_native_velocity_values_sha256": result[
                        "selected_native_velocity_values_sha256"
                    ],
                    "constant_cp_case_equal_cut_mean_rmse": result[
                        "constant_cp_case_equal_cut_mean_rmse"
                    ],
                    "validation_identity_sha256": result["validation_identity"][
                        "sha256"
                    ],
                    "receipt": binding,
                    "relative_scoring_activated": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (TruthValidationError, exporter.ExportError, OSError, ValueError) as error:
        raise SystemExit(f"native profile truth validation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
