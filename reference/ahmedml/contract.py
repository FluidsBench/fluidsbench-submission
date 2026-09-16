"""Immutable public-source identities for the candidate AhmedML evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


SOURCE_SCHEMA = "ahmedml-public-native-source-identity-v1"
DATASET_ID = "ahmedml"
DATASET_VERSION = "ahmedml-native-v1-candidate"
REPOSITORY_ID = "neashton/ahmedml"
REPOSITORY_REVISION = "02688c727cdb8dc8678e28abc6bbbb7e93c5fa15"
SOURCE_IDENTITY_SHA256 = (
    "56a620a5b335cef6cb0587e186df321eb907bbe46e8b81aa4de302b8fe73cf44"
)
PROFILE_DEFINITION_SHA256 = (
    "1048a0380f70de778e3e51db07d6910579f720d6f2dd1d9ad7179dbfe88d1cae"
)
REGION_DEFINITION_SHA256 = (
    "338db5b806caec2883e2583e491b751546624d3f101c04ae644cf9357b1275d7"
)
CASE_COUNT = 500
FORCE_ABSOLUTE_TOLERANCE = 2.0e-6
SURFACE_FIELDS: Mapping[str, int] = MappingProxyType(
    {"pMean": 1, "wallShearStressMean": 3}
)
VOLUME_FIELDS: Mapping[str, int] = MappingProxyType({"pMean": 1, "UMean": 3})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_RE = re.compile(r"run_([1-9][0-9]*)\Z")


class AhmedMLContractError(ValueError):
    """Raised when a candidate AhmedML contract input fails closed."""


@dataclass(frozen=True)
class ForceReplayException:
    """One checksum-bound source CSV/native-field audit discrepancy."""

    exception_id: str
    case_id: str
    published_cd: float
    published_cl: float
    field_replay_cd: float
    field_replay_cl: float
    observation_tolerance: float
    reason: str


# The run_492 force CSV is internally inconsistent with the complete pinned
# native surface fields. Its approximately 0.8% discrepancy is not used as a
# tolerance relaxation: all observed values are frozen here, the source files
# remain SHA-bound, and force scoring continues to use field integration.
FORCE_REPLAY_EXCEPTIONS: Mapping[str, ForceReplayException] = MappingProxyType(
    {
        "run_492": ForceReplayException(
            exception_id="ahmedml-run-492-force-csv-native-field-mismatch-v1",
            case_id="run_492",
            published_cd=0.31594543729,
            published_cl=0.55676680791,
            field_replay_cd=0.31839300386407177,
            field_replay_cl=0.5609792089607799,
            observation_tolerance=FORCE_ABSOLUTE_TOLERANCE,
            reason=(
                "The pinned force_mom CSV does not replay from the pinned complete "
                "native surface pMean and wallShearStressMean fields. Native-field "
                "integration remains authoritative for evaluator truth."
            ),
        )
    }
)


def classify_force_replay(
    *,
    case_id: str,
    published_cd: float,
    published_cl: float,
    field_replay_cd: float,
    field_replay_cl: float,
) -> ForceReplayException | None:
    """Validate a force replay, returning only a fully pinned source exception."""

    observed = (published_cd, published_cl, field_replay_cd, field_replay_cl)
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in observed
    ):
        raise AhmedMLContractError(
            f"{case_id} force replay contains a non-finite value"
        )
    exception = FORCE_REPLAY_EXCEPTIONS.get(case_id)
    if exception is not None:
        expected = (
            exception.published_cd,
            exception.published_cl,
            exception.field_replay_cd,
            exception.field_replay_cl,
        )
        if any(
            abs(float(actual) - expected_value) > exception.observation_tolerance
            for actual, expected_value in zip(observed, expected, strict=True)
        ):
            raise AhmedMLContractError(
                f"{case_id} does not reproduce its pinned force-replay exception"
            )
        return exception
    if (
        abs(field_replay_cd - published_cd) > FORCE_ABSOLUTE_TOLERANCE
        or abs(field_replay_cl - published_cl) > FORCE_ABSOLUTE_TOLERANCE
    ):
        raise AhmedMLContractError(
            "field-derived force coefficients do not reproduce force_mom CSV: "
            f"derived ({field_replay_cd}, {field_replay_cl}), "
            f"published ({published_cd}, {published_cl})"
        )
    return None


def sha256_file(path: str | Path, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Hash a file without retaining its payload."""

    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
        raise AhmedMLContractError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            while block := source.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise AhmedMLContractError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AhmedMLContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AhmedMLContractError(
                    f"{label} contains forbidden non-finite token {token}"
                )
            ),
        )
    except AhmedMLContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AhmedMLContractError(f"cannot read {label} {source}: {error}") from error
    if not isinstance(value, dict):
        raise AhmedMLContractError(f"{label} must contain a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AhmedMLContractError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AhmedMLContractError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AhmedMLContractError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise AhmedMLContractError(f"{label} must be a lowercase SHA-256")
    return result


def _relative_path(value: object, label: str) -> PurePosixPath:
    text = _string(value, label)
    if "\\" in text:
        raise AhmedMLContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise AhmedMLContractError(f"{label} must be a normalized relative path")
    return path


@dataclass(frozen=True)
class SourceFileIdentity:
    """One file in the immutable public AhmedML revision."""

    path: PurePosixPath
    sha256: str
    size_bytes: int

    def resolve(self, dataset_root: str | Path) -> Path:
        root = Path(dataset_root).expanduser().resolve()
        # Keep the declared pathname rather than resolving its final symlink.
        # The verified local development mirror deliberately deduplicates some
        # tiny CSV files with symlinks; consumers must still hash the resolved
        # regular-file bytes before use.
        result = root.joinpath(*self.path.parts)
        try:
            result.absolute().relative_to(root)
        except ValueError as error:
            raise AhmedMLContractError(
                f"source path {self.path.as_posix()!r} escapes dataset root"
            ) from error
        return result


@dataclass(frozen=True)
class AhmedMLSourceCase:
    """Exact native source identities and tuple counts for one run."""

    case_id: str
    run_id: int
    surface_entity_count: int
    volume_entity_count: int
    boundary: SourceFileIdentity
    surface_cell_area: SourceFileIdentity
    volume: SourceFileIdentity
    geometry_parameters: SourceFileIdentity
    force_coefficients: SourceFileIdentity


@dataclass(frozen=True)
class AhmedMLSourceIdentity:
    """Validated public-source identity indexed by canonical case ID."""

    source_path: Path
    sha256: str
    repository_id: str
    repository_revision: str
    cases: tuple[AhmedMLSourceCase, ...]
    _by_case: Mapping[str, AhmedMLSourceCase] = field(repr=False, compare=False)

    def case(self, case_id: str) -> AhmedMLSourceCase:
        try:
            return self._by_case[case_id]
        except KeyError as error:
            raise AhmedMLContractError(f"unknown AhmedML case {case_id!r}") from error


def _source_file(value: object, label: str, expected_path: str) -> SourceFileIdentity:
    item = _mapping(value, label)
    path = _relative_path(item.get("path"), f"{label}.path")
    if path.as_posix() != expected_path:
        raise AhmedMLContractError(
            f"{label}.path must be {expected_path!r}, found {path.as_posix()!r}"
        )
    return SourceFileIdentity(
        path=path,
        sha256=_sha256(item.get("sha256"), f"{label}.sha256"),
        size_bytes=_integer(item.get("size_bytes"), f"{label}.size_bytes", minimum=1),
    )


def _validate_field_contract(
    value: object,
    expected: Mapping[str, int],
    label: str,
) -> None:
    fields = _mapping(value, label)
    if set(fields) != set(expected):
        raise AhmedMLContractError(f"{label} field names differ from the contract")
    for name, components in expected.items():
        item = _mapping(fields[name], f"{label}.{name}")
        if item.get("components") != components or item.get("dtype") != "Float32":
            raise AhmedMLContractError(
                f"{label}.{name} must be Float32 with {components} component(s)"
            )


def load_source_identity(
    path: str | Path,
    *,
    expected_sha256: str = SOURCE_IDENTITY_SHA256,
) -> AhmedMLSourceIdentity:
    """Load the exact 500-case public source identity.

    The JSON file is hash-pinned independently of every file identity it
    contains.  Materialized dataset files are verified later by the evaluator.
    """

    source_path = Path(path).expanduser().resolve()
    actual_sha = sha256_file(source_path)
    if actual_sha != _sha256(expected_sha256, "expected source identity SHA-256"):
        raise AhmedMLContractError(
            f"source identity SHA-256 is {actual_sha}, expected {expected_sha256}"
        )
    document = read_json(source_path, label="AhmedML source identity")
    if (
        document.get("schema") != SOURCE_SCHEMA
        or document.get("dataset_id") != DATASET_ID
        or document.get("dataset_version") != DATASET_VERSION
        or document.get("association") != "CellData"
    ):
        raise AhmedMLContractError("AhmedML source identity header is inconsistent")
    repository = _mapping(document.get("repository"), "repository")
    if (
        repository.get("id") != REPOSITORY_ID
        or repository.get("revision") != REPOSITORY_REVISION
    ):
        raise AhmedMLContractError("AhmedML repository identity is inconsistent")
    _validate_field_contract(document.get("surface_fields"), SURFACE_FIELDS, "surface_fields")
    _validate_field_contract(document.get("volume_fields"), VOLUME_FIELDS, "volume_fields")
    values = document.get("cases")
    if not isinstance(values, list) or document.get("case_count") != CASE_COUNT:
        raise AhmedMLContractError("AhmedML source identity must declare 500 cases")
    if len(values) != CASE_COUNT:
        raise AhmedMLContractError("AhmedML source identity case array must contain 500 cases")

    cases: list[AhmedMLSourceCase] = []
    for offset, value in enumerate(values, start=1):
        label = f"cases[{offset - 1}]"
        item = _mapping(value, label)
        run_id = _integer(item.get("run_id"), f"{label}.run_id", minimum=1)
        case_id = _string(item.get("case_id"), f"{label}.case_id")
        match = _CASE_RE.fullmatch(case_id)
        if run_id != offset or match is None or int(match.group(1)) != run_id:
            raise AhmedMLContractError(
                "AhmedML cases must be contiguous run_1 through run_500"
            )
        prefix = f"run_{run_id}"
        cases.append(
            AhmedMLSourceCase(
                case_id=case_id,
                run_id=run_id,
                surface_entity_count=_integer(
                    item.get("surface_entity_count"),
                    f"{label}.surface_entity_count",
                    minimum=1,
                ),
                volume_entity_count=_integer(
                    item.get("volume_entity_count"),
                    f"{label}.volume_entity_count",
                    minimum=1,
                ),
                boundary=_source_file(
                    item.get("boundary"),
                    f"{label}.boundary",
                    f"{prefix}/boundary_{run_id}.vtp",
                ),
                surface_cell_area=_source_file(
                    item.get("surface_cell_area"),
                    f"{label}.surface_cell_area",
                    f"{prefix}/boundary_cell_area_{run_id}.npy",
                ),
                volume=_source_file(
                    item.get("volume"),
                    f"{label}.volume",
                    f"{prefix}/volume_{run_id}.vtu",
                ),
                geometry_parameters=_source_file(
                    item.get("geometry_parameters"),
                    f"{label}.geometry_parameters",
                    f"{prefix}/geo_parameters_{run_id}.csv",
                ),
                force_coefficients=_source_file(
                    item.get("force_coefficients"),
                    f"{label}.force_coefficients",
                    f"{prefix}/force_mom_{run_id}.csv",
                ),
            )
        )
    case_tuple = tuple(cases)
    return AhmedMLSourceIdentity(
        source_path=source_path,
        sha256=actual_sha,
        repository_id=REPOSITORY_ID,
        repository_revision=REPOSITORY_REVISION,
        cases=case_tuple,
        _by_case=MappingProxyType({case.case_id: case for case in case_tuple}),
    )


__all__ = [
    "AhmedMLContractError",
    "AhmedMLSourceCase",
    "AhmedMLSourceIdentity",
    "CASE_COUNT",
    "DATASET_ID",
    "DATASET_VERSION",
    "FORCE_ABSOLUTE_TOLERANCE",
    "FORCE_REPLAY_EXCEPTIONS",
    "ForceReplayException",
    "PROFILE_DEFINITION_SHA256",
    "REGION_DEFINITION_SHA256",
    "REPOSITORY_ID",
    "REPOSITORY_REVISION",
    "SOURCE_IDENTITY_SHA256",
    "SourceFileIdentity",
    "classify_force_replay",
    "load_source_identity",
    "read_json",
    "sha256_file",
]
