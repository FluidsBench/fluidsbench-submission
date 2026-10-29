"""Enforce the immutable WindsorML profile release at evaluation and reduction.

The manifest is a trust anchor, not participant metadata. Both consumers read
the same pinned support, so a plausible alternative set of native sample IDs,
coordinates or truth values cannot silently define a different benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import SOURCE_IDENTITY_SHA256


SPEC_DIRECTORY = Path(__file__).resolve().parents[2] / "benchmark-specs" / "windsorml"
PROFILE_DEFINITION_PATH = SPEC_DIRECTORY / "profile-definition-v2.json"
PROFILE_MANIFEST_PATH = (
    SPEC_DIRECTORY / "profile-support" / "windsorml-profile-support-v3-manifest.json"
)
PROFILE_MANIFEST_SHA256 = (
    "e660311ef82ec19347e91eeb1753c1f0af4c1bd65ba01491fb47ac960da51376"
)
PROFILE_DEFINITION_SHA256 = (
    "d58014ae66d92ea5ffce3c4f3b8e206447873d563d8bc3ffb4cbeee81539e056"
)
PROFILE_SUPPORT_SCHEMA = "windsorml-profile-support-v3"
PROFILE_TRUTH_ATOL = 1.0e-5  # Existing native-stream/support agreement tolerance.
PROFILE_SERIES_SOURCE = "evaluator_derived_from_complete_native_fields"


class WindsorMLProfileError(ValueError):
    """Raised when support or evidence differs from the pinned profile release."""


@dataclass(frozen=True)
class ProfileSupport:
    document: Mapping[str, Any]
    definition: Mapping[str, Any]
    sha256: str

    def evidence_binding(self) -> dict[str, object]:
        return {
            "case_id": self.document["case_id"],
            "support_schema": PROFILE_SUPPORT_SCHEMA,
            "profile_support_manifest_sha256": PROFILE_MANIFEST_SHA256,
            "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
            "source_identity_sha256": SOURCE_IDENTITY_SHA256,
            "case_support_sha256": self.sha256,
            "sample_count": self.document["sample_count"],
            "body_height_m": self.document["body_height_m"],
        }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WindsorMLProfileError(f"{label} must be an object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WindsorMLProfileError(f"duplicate profile JSON key {key!r}")
        result[key] = value
    return result


def _read_pinned(path: Path, digest: str, label: str) -> dict[str, Any]:
    # Hash and parse the same small byte buffer, never reopen after hashing.
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise WindsorMLProfileError(
                f"{label} SHA-256 differs from the frozen release"
            )
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WindsorMLProfileError(f"cannot read {label}: {error}") from error
    return dict(_mapping(document, label))


def _samples(value: object, count: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != count:
        raise WindsorMLProfileError(f"{label} must contain exactly {count} samples")
    try:
        valid = all(
            not isinstance(v, bool)
            and isinstance(v, (int, float))
            and math.isfinite(v)
            for v in value
        )
    except OverflowError:
        valid = False
    if not valid:
        raise WindsorMLProfileError(f"{label} must contain finite numbers")
    return np.asarray(value, dtype=np.float64)


def load_profile_support(
    *, case_id: str, value: Mapping[str, object] | str | Path | None = None
) -> ProfileSupport:
    """Load canonical support, or verify a supplied file/document against it.

    File copies must retain the pinned bytes. The in-memory API remains usable,
    but a document must equal the decoded canonical file, including all IDs and
    metadata; caller-provided hashes are never trusted.
    """

    manifest = _read_pinned(
        PROFILE_MANIFEST_PATH, PROFILE_MANIFEST_SHA256, "profile manifest"
    )
    definition = _read_pinned(
        PROFILE_DEFINITION_PATH, PROFILE_DEFINITION_SHA256, "profile definition"
    )
    if (
        manifest.get("schema") != "windsorml-profile-support-manifest-v1"
        or manifest.get("dataset_id") != "windsorml"
        or manifest.get("support_schema") != PROFILE_SUPPORT_SCHEMA
        or manifest.get("source_identity_sha256") != SOURCE_IDENTITY_SHA256
        or manifest.get("profile_definition_sha256") != PROFILE_DEFINITION_SHA256
        or manifest.get("sample_count") != definition["sampling"]["sample_count"]
    ):
        raise WindsorMLProfileError("profile manifest release bindings differ")
    entries = [entry for entry in manifest["cases"] if entry["case_id"] == case_id]
    if len(entries) != 1:
        raise WindsorMLProfileError(
            f"profile manifest does not bind exactly one {case_id}"
        )
    entry = entries[0]
    if entry["file"] != f"{case_id}.json":
        raise WindsorMLProfileError("profile manifest case filename differs")
    path = (
        Path(value).expanduser()
        if isinstance(value, (str, Path))
        else PROFILE_MANIFEST_PATH.parent / entry["file"]
    )
    document = _read_pinned(path, entry["sha256"], f"{case_id} profile support")
    if isinstance(value, Mapping):
        try:
            supplied = json.dumps(dict(value), sort_keys=True, allow_nan=False)
            canonical = json.dumps(document, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise WindsorMLProfileError(
                "profile support is not a finite JSON document"
            ) from error
        if supplied != canonical:
            raise WindsorMLProfileError(
                "profile support differs from the hash-pinned case document"
            )
    elif value is not None and not isinstance(value, (str, Path)):
        raise WindsorMLProfileError("profile support must be a path or object")

    count = definition["sampling"]["sample_count"]
    if (
        document.get("schema") != PROFILE_SUPPORT_SCHEMA
        or document.get("case_id") != case_id
        or document.get("sample_count") != count
        or document.get("body_height_m") != entry["body_height_m"]
        or document.get("reference_velocity_m_s")
        != definition["velocity_stations"]["reference_velocity_m_s"]
    ):
        raise WindsorMLProfileError(f"{case_id} profile support metadata differs")
    families = _mapping(document.get("families"), "profile support families")
    expected = {family["family_id"]: family for family in definition["families"]}
    if set(families) != set(expected):
        raise WindsorMLProfileError(
            "profile support families differ from the frozen definition"
        )
    for family_id, family in expected.items():
        stations = _mapping(families[family_id], family_id)
        if set(stations) != set(family["station_ids"]):
            raise WindsorMLProfileError(
                f"{family_id} stations differ from the frozen definition"
            )
        velocity = family["quantity_id"] == "ux_over_uinf"
        for station_id, raw in stations.items():
            station = _mapping(raw, station_id)
            label = f"{family_id}/{station_id}"
            coordinate = _samples(
                station.get("coordinate"), count, f"{label} coordinate"
            )
            if np.any(np.diff(coordinate) <= 0):
                raise WindsorMLProfileError(
                    f"{label} coordinate must increase strictly"
                )
            _samples(
                station.get("truth_ux_over_uinf" if velocity else "truth_cp"),
                count,
                f"{label} truth",
            )
            ids = station.get("native_cell_ids" if velocity else "native_point_ids")
            if (
                not isinstance(ids, list)
                or len(ids) != count
                or any(type(v) is not int or v < 0 for v in ids)
            ):
                raise WindsorMLProfileError(
                    f"{label} must contain {count} native integer IDs"
                )
    return ProfileSupport(document, definition, entry["sha256"])


def validate_profile_evidence(
    value: object, *, case_id: str
) -> dict[str, tuple[list[float], list[float]]]:
    """Return complete series in definition order after checking their binding."""

    support = load_profile_support(case_id=case_id)
    evidence = _mapping(value, f"{case_id} profiles")
    for key, expected in support.evidence_binding().items():
        actual = evidence.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise WindsorMLProfileError(
                f"{case_id} profiles.{key} differs from frozen support"
            )
    if evidence.get("participant_profile_payload_accepted") is not False:
        raise WindsorMLProfileError(
            f"{case_id} profiles must be evaluator-derived, never submitted"
        )
    families = _mapping(evidence.get("families"), f"{case_id} profile families")
    expected_families = {
        family["family_id"] for family in support.definition["families"]
    }
    if set(families) != expected_families:
        raise WindsorMLProfileError(
            f"{case_id} profile families differ from frozen support"
        )
    count = support.document["sample_count"]
    result = {}
    for family in support.definition["families"]:
        family_id = family["family_id"]
        series = families[family_id]
        if not isinstance(series, list) or len(series) != len(family["station_ids"]):
            raise WindsorMLProfileError(
                f"{case_id}/{family_id} must contain every required station exactly once"
            )
        by_station = {}
        for raw in series:
            item = _mapping(raw, f"{case_id}/{family_id} station")
            station_id = item.get("station_id")
            if (
                not isinstance(station_id, str)
                or station_id not in family["station_ids"]
                or station_id in by_station
            ):
                raise WindsorMLProfileError(
                    f"{case_id}/{family_id} has an unknown or duplicate station"
                )
            by_station[station_id] = item
        truths: list[float] = []
        predictions: list[float] = []
        for station_id in family["station_ids"]:
            item = by_station[station_id]
            label = f"{case_id}/{family_id}/{station_id}"
            if (
                item.get("quantity_id") != family["quantity_id"]
                or type(item.get("sample_count")) is not int
                or item["sample_count"] != count
                or item.get("source") != PROFILE_SERIES_SOURCE
            ):
                raise WindsorMLProfileError(
                    f"{label} series metadata differs from frozen support"
                )
            coordinate = _samples(item.get("coordinate"), count, f"{label} coordinate")
            truth = _samples(item.get("truth"), count, f"{label} truth")
            prediction = _samples(item.get("prediction"), count, f"{label} prediction")
            frozen = support.document["families"][family_id][station_id]
            truth_key = (
                "truth_ux_over_uinf"
                if family["quantity_id"] == "ux_over_uinf"
                else "truth_cp"
            )
            if not np.array_equal(coordinate, frozen["coordinate"]):
                raise WindsorMLProfileError(
                    f"{label} coordinate differs from frozen support"
                )
            if not np.allclose(
                truth, frozen[truth_key], rtol=0.0, atol=PROFILE_TRUTH_ATOL
            ):
                raise WindsorMLProfileError(
                    f"{label} truth differs from frozen support"
                )
            truths.extend(truth.tolist())
            predictions.extend(prediction.tolist())
        result[family_id] = (truths, predictions)
    return result
