"""Immutable public-source identities for the candidate WindsorML evaluator.

Structured after :mod:`reference.ahmedml.contract`, with four WindsorML-specific
divergences that are deliberate and load-bearing:

1. **Case IDs are 0-indexed.** WindsorML publishes ``run_0`` through ``run_349``;
   AhmedML publishes ``run_1`` through ``run_500``. The AhmedML case pattern
   ``run_([1-9][0-9]*)`` rejects ``run_0`` outright.
2. **The vertical axis is +y, not +z.** Measured from the published meshes: the
   body sits on a ground plane at ``y = 0`` spanning ``y in [0, 0.343]`` and is
   laterally symmetric about ``z = 0`` over ``z in [-0.1945, 0.1945]``. Lift is
   therefore the y component. Getting this wrong silently produces a lift score
   near zero that still "looks" plausible, so it is asserted here and regression
   tested.
3. **Surface fields are PointData**, not CellData, and the published
   ``boundary_dual_area_N.npy`` sidecar is the canonical per-point quadrature
   weight. AhmedML pins CellData with an evaluator-generated area array.
4. **Forces are scored by integrating both sides, and the published CSV is the
   audit anchor rather than the scoring truth.** Predicted coefficients can only
   come from integrating the predicted fields, so truth must be integrated the
   same way: otherwise the ~0.26% gap between the dual-area point quadrature and
   the solver's exact cell integration becomes an error floor that even a
   perfect prediction cannot reach, and ``cd_r2`` could never be 1. AhmedML can
   afford to fail closed at 2e-6 because its cell-area integration reproduces
   its solver exactly; WindsorML cannot, so the published CSV instead bounds the
   *truth* integration via :func:`classify_force_replay` at
   :data:`FORCE_REPLAY_RELATIVE_TOLERANCE`. That keeps scoring self-consistent
   while still refusing a case whose mesh, weights, or axis mapping has drifted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


SOURCE_SCHEMA = "windsorml-public-native-source-identity-v1"
DATASET_ID = "windsorml"
DATASET_VERSION = "windsorml-native-v1-candidate"
REPOSITORY_ID = "neashton/windsorml"
REPOSITORY_REVISION = "8a6ca32ae22c94f54df2186d1b0ccf9662a294c2"

# Published per-run cases: run_0 .. run_349 inclusive.
CASE_COUNT = 350
FIRST_RUN_ID = 0

# The aggregate tables and the official split manifest both enumerate 355
# design variants, but run_350..run_354 have no per-run payload in the public
# release. They are therefore unscoreable and are excluded from every scored
# case set. The official manifest itself is left untouched.
UNPUBLISHED_CASE_IDS: frozenset[str] = frozenset(
    f"run_{run_id}" for run_id in range(350, 355)
)

SURFACE_ASSOCIATION = "PointData"
VOLUME_ASSOCIATION = "CellData"
SURFACE_FIELDS: Mapping[str, int] = MappingProxyType(
    {"cpavg": 1, "cfxavg": 1, "cfyavg": 1, "cfzavg": 1}
)
VOLUME_FIELDS: Mapping[str, int] = MappingProxyType(
    {"velocityxavg": 1, "velocityyavg": 1, "velocityzavg": 1, "pressureavg": 1}
)

# Force convention, measured from the release rather than assumed.
# A_ref recovered exactly as frontal_area * cd_varref / cd_fixed = 0.112 m^2.
FORCE_REFERENCE_AREA_M2 = 0.112
DRAG_AXIS_INDEX = 0  # +x
LIFT_AXIS_INDEX = 1  # +y  <- not z; see module docstring
SIDE_AXIS_INDEX = 2  # +z
AXIS_CONVENTION: Mapping[str, str] = MappingProxyType(
    {"drag": "+x", "lift": "+y", "side": "+z"}
)

# Coefficients are scored from field integration on both sides; the published
# CSV is the audit anchor. See the module docstring for why the CSV cannot be
# the scoring truth.
FORCE_TRUTH_SOURCE = "native_field_integration"

# Agreement required between the published force CSV and the dual-area truth
# integration, as |replay - published| <= ABSOLUTE + RELATIVE * |published|.
#
# A purely relative bound is the wrong instrument here: side force and lift can
# be legitimately near zero (run_306 publishes cl = +0.000106), so a 0.0003
# absolute difference becomes a 310% relative one while meaning nothing. The
# absolute deltas are tightly bounded across all 350 published runs -- drag
# max 0.0089, lift max 0.0034 -- so the absolute term carries the audit for
# small coefficients and the relative term keeps large ones honest.
#
# Both constants are frozen from the observed distribution over all 350
# published runs (see scripts/build_windsorml_case_support.py):
#
#   cd |absolute|  median 0.000037  p95 0.004562  max 0.008926
#   cl |absolute|  median 0.000006  p95 0.000780  max 0.003369
#
# With RELATIVE fixed at 0.02, the tightest ABSOLUTE that still admits every
# published run is 0.002849. 0.005 keeps roughly 1.75x headroom -- enough to
# absorb legitimate case-to-case variation without blunting the audit.
FORCE_REPLAY_ABSOLUTE_TOLERANCE = 0.005
FORCE_REPLAY_RELATIVE_TOLERANCE = 0.02

# SHA-256 of benchmark-specs/windsorml/public-source-identity/
# windsorml-public-source-identity-v1.json, pinned so the evaluator cannot be
# pointed at a re-generated identity that silently differs.
SOURCE_IDENTITY_SHA256 = (
    "e9bc888931e26220a9c7bddc202a66ab96a043343bf9d70066d8de4ad8ba2cff"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_RE = re.compile(r"run_(0|[1-9][0-9]*)\Z")


class WindsorMLContractError(ValueError):
    """Raised when a candidate WindsorML contract input fails closed."""


def sha256_file(path: str | Path, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Hash a file without retaining its payload."""

    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
        raise WindsorMLContractError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            while block := source.read(chunk_bytes):
                digest.update(block)
    except OSError as error:
        raise WindsorMLContractError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WindsorMLContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                WindsorMLContractError(
                    f"{label} contains forbidden non-finite token {token}"
                )
            ),
        )
    except WindsorMLContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WindsorMLContractError(f"cannot read {label} {source}: {error}") from error
    if not isinstance(value, dict):
        raise WindsorMLContractError(f"{label} must contain a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WindsorMLContractError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WindsorMLContractError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WindsorMLContractError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WindsorMLContractError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise WindsorMLContractError(f"{label} must be finite")
    return result


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise WindsorMLContractError(f"{label} must be a lowercase SHA-256")
    return result


def _relative_path(value: object, label: str) -> PurePosixPath:
    text = _string(value, label)
    if "\\" in text:
        raise WindsorMLContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise WindsorMLContractError(f"{label} must be a normalized relative path")
    return path


@dataclass(frozen=True)
class SourceFileIdentity:
    """One file in the immutable public WindsorML revision."""

    path: PurePosixPath
    sha256: str
    size_bytes: int

    def resolve(self, dataset_root: str | Path) -> Path:
        root = Path(dataset_root).expanduser().resolve()
        result = root.joinpath(*self.path.parts)
        try:
            result.absolute().relative_to(root)
        except ValueError as error:
            raise WindsorMLContractError(
                f"source path {self.path.as_posix()!r} escapes dataset root"
            ) from error
        return result


@dataclass(frozen=True)
class WindsorMLForceTruth:
    """Published coefficients for one case, at the fixed reference area."""

    cd: float
    cl: float
    cs: float
    cmy: float


@dataclass(frozen=True)
class WindsorMLSourceCase:
    """Exact native source identities and tuple counts for one run."""

    case_id: str
    run_id: int
    surface_entity_count: int
    volume_entity_count: int
    boundary: SourceFileIdentity
    surface_dual_area: SourceFileIdentity
    volume: SourceFileIdentity
    geometry_parameters: SourceFileIdentity
    force_coefficients: SourceFileIdentity
    force_truth: WindsorMLForceTruth


@dataclass(frozen=True)
class WindsorMLSourceIdentity:
    """Validated public-source identity indexed by canonical case ID."""

    source_path: Path
    sha256: str
    repository_id: str
    repository_revision: str
    cases: tuple[WindsorMLSourceCase, ...]
    _by_case: Mapping[str, WindsorMLSourceCase] = field(repr=False, compare=False)

    def case(self, case_id: str) -> WindsorMLSourceCase:
        try:
            return self._by_case[case_id]
        except KeyError as error:
            raise WindsorMLContractError(
                f"unknown WindsorML case {case_id!r}"
            ) from error

    def has_case(self, case_id: str) -> bool:
        return case_id in self._by_case


def _source_file(value: object, label: str, expected_path: str) -> SourceFileIdentity:
    item = _mapping(value, label)
    path = _relative_path(item.get("path"), f"{label}.path")
    if path.as_posix() != expected_path:
        raise WindsorMLContractError(
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
        raise WindsorMLContractError(f"{label} field names differ from the contract")
    for name, components in expected.items():
        item = _mapping(fields[name], f"{label}.{name}")
        if item.get("components") != components or item.get("dtype") != "Float32":
            raise WindsorMLContractError(
                f"{label}.{name} must be Float32 with {components} component(s)"
            )


def _force_truth(value: object, label: str) -> WindsorMLForceTruth:
    item = _mapping(value, label)
    if set(item) != {"cd", "cl", "cs", "cmy"}:
        raise WindsorMLContractError(f"{label} must declare cd, cl, cs and cmy")
    return WindsorMLForceTruth(
        cd=_finite_float(item.get("cd"), f"{label}.cd"),
        cl=_finite_float(item.get("cl"), f"{label}.cl"),
        cs=_finite_float(item.get("cs"), f"{label}.cs"),
        cmy=_finite_float(item.get("cmy"), f"{label}.cmy"),
    )


def classify_force_replay(
    *,
    case_id: str,
    published: WindsorMLForceTruth,
    replay_cd: float,
    replay_cl: float,
) -> Mapping[str, float]:
    """Audit the dual-area truth integration against the published coefficients.

    Scoring compares an integrated truth to an integrated prediction, so this
    never enters a metric. It exists to refuse a case whose integrated drag or
    lift drifts further from the published CSV than the frozen tolerance, which
    would indicate a corrupted mesh, a wrong weight array, or a mis-declared
    axis convention -- in particular, mapping lift to z instead of y collapses
    ``cl`` from ~0.49 to ~-0.01 and is caught here immediately.
    """

    for name, value in (("replay_cd", replay_cd), ("replay_cl", replay_cl)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WindsorMLContractError(f"{case_id} {name} must be a number")
        if not math.isfinite(float(value)):
            raise WindsorMLContractError(f"{case_id} {name} is not finite")

    deltas: dict[str, float] = {}
    for name, replayed, truth in (
        ("cd", float(replay_cd), published.cd),
        ("cl", float(replay_cl), published.cl),
    ):
        absolute = abs(replayed - truth)
        allowance = (
            FORCE_REPLAY_ABSOLUTE_TOLERANCE
            + FORCE_REPLAY_RELATIVE_TOLERANCE * abs(truth)
        )
        if absolute > allowance:
            raise WindsorMLContractError(
                f"{case_id} {name} replay differs from the published value by "
                f"{absolute:.6f}, above the allowed {allowance:.6f} "
                f"({FORCE_REPLAY_ABSOLUTE_TOLERANCE} + "
                f"{FORCE_REPLAY_RELATIVE_TOLERANCE} * |{truth:.6f}|)"
            )
        deltas[name] = absolute
    return MappingProxyType(deltas)


def load_source_identity(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> WindsorMLSourceIdentity:
    """Load the exact 350-case public source identity.

    ``expected_sha256`` is optional only so the identity file can be generated
    and self-checked in one pass; evaluator entry points must always pass the
    pinned digest.
    """

    source_path = Path(path).expanduser().resolve()
    actual_sha = sha256_file(source_path)
    if expected_sha256 is not None:
        if actual_sha != _sha256(expected_sha256, "expected source identity SHA-256"):
            raise WindsorMLContractError(
                f"source identity SHA-256 is {actual_sha}, expected {expected_sha256}"
            )
    document = read_json(source_path, label="WindsorML source identity")
    if (
        document.get("schema") != SOURCE_SCHEMA
        or document.get("dataset_id") != DATASET_ID
        or document.get("dataset_version") != DATASET_VERSION
        or document.get("surface_association") != SURFACE_ASSOCIATION
        or document.get("volume_association") != VOLUME_ASSOCIATION
    ):
        raise WindsorMLContractError("WindsorML source identity header is inconsistent")
    repository = _mapping(document.get("repository"), "repository")
    if (
        repository.get("id") != REPOSITORY_ID
        or repository.get("revision") != REPOSITORY_REVISION
    ):
        raise WindsorMLContractError("WindsorML repository identity is inconsistent")

    forces = _mapping(document.get("force_convention"), "force_convention")
    if (
        forces.get("reference_area_m2") != FORCE_REFERENCE_AREA_M2
        or dict(_mapping(forces.get("axes"), "force_convention.axes")) != dict(AXIS_CONVENTION)
        or forces.get("truth_source") != FORCE_TRUTH_SOURCE
        or forces.get("audit_reference") != "published_force_mom_csv"
    ):
        raise WindsorMLContractError("WindsorML force convention is inconsistent")

    _validate_field_contract(document.get("surface_fields"), SURFACE_FIELDS, "surface_fields")
    _validate_field_contract(document.get("volume_fields"), VOLUME_FIELDS, "volume_fields")

    values = document.get("cases")
    if not isinstance(values, list) or document.get("case_count") != CASE_COUNT:
        raise WindsorMLContractError("WindsorML source identity must declare 350 cases")
    if len(values) != CASE_COUNT:
        raise WindsorMLContractError(
            "WindsorML source identity case array must contain 350 cases"
        )

    cases: list[WindsorMLSourceCase] = []
    for offset, value in enumerate(values):
        label = f"cases[{offset}]"
        item = _mapping(value, label)
        expected_run = FIRST_RUN_ID + offset
        run_id = _integer(item.get("run_id"), f"{label}.run_id", minimum=0)
        case_id = _string(item.get("case_id"), f"{label}.case_id")
        match = _CASE_RE.fullmatch(case_id)
        if run_id != expected_run or match is None or int(match.group(1)) != run_id:
            raise WindsorMLContractError(
                "WindsorML cases must be contiguous run_0 through run_349"
            )
        if case_id in UNPUBLISHED_CASE_IDS:
            raise WindsorMLContractError(
                f"{case_id} has no public payload and cannot appear in the source identity"
            )
        prefix = f"run_{run_id}"
        cases.append(
            WindsorMLSourceCase(
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
                    f"{prefix}/boundary_{run_id}.vtu",
                ),
                surface_dual_area=_source_file(
                    item.get("surface_dual_area"),
                    f"{label}.surface_dual_area",
                    f"{prefix}/boundary_dual_area_{run_id}.npy",
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
                force_truth=_force_truth(item.get("force_truth"), f"{label}.force_truth"),
            )
        )
    case_tuple = tuple(cases)
    return WindsorMLSourceIdentity(
        source_path=source_path,
        sha256=actual_sha,
        repository_id=REPOSITORY_ID,
        repository_revision=REPOSITORY_REVISION,
        cases=case_tuple,
        _by_case=MappingProxyType({case.case_id: case for case in case_tuple}),
    )


__all__ = [
    "AXIS_CONVENTION",
    "CASE_COUNT",
    "DATASET_ID",
    "DATASET_VERSION",
    "DRAG_AXIS_INDEX",
    "FORCE_REFERENCE_AREA_M2",
    "FORCE_REPLAY_ABSOLUTE_TOLERANCE",
    "FORCE_REPLAY_RELATIVE_TOLERANCE",
    "FORCE_TRUTH_SOURCE",
    "FIRST_RUN_ID",
    "LIFT_AXIS_INDEX",
    "REPOSITORY_ID",
    "REPOSITORY_REVISION",
    "SIDE_AXIS_INDEX",
    "SOURCE_IDENTITY_SHA256",
    "SOURCE_SCHEMA",
    "SURFACE_ASSOCIATION",
    "SURFACE_FIELDS",
    "UNPUBLISHED_CASE_IDS",
    "VOLUME_ASSOCIATION",
    "VOLUME_FIELDS",
    "SourceFileIdentity",
    "WindsorMLContractError",
    "WindsorMLForceTruth",
    "WindsorMLSourceCase",
    "WindsorMLSourceIdentity",
    "classify_force_replay",
    "load_source_identity",
    "read_json",
    "sha256_file",
]
