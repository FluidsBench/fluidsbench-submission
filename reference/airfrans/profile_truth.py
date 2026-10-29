"""AirfRANS candidate profile-truth release: deterministic packaging and validation.

Packages the ground-truth velocity-profile JSON files written by
``examples/airfrans-profile-extraction/extract.py`` (``ground_truth_reference``
mode) into a release that follows the HiLiftAeroML candidate profile-truth
layout (``reference/hiliftaeroml/native_profile_truth.py``)::

    cases/<case_id>.json          the extracted truth, byte-identical to the scored file
    case-records/<case_id>.json   identity, source hashes and checks for one case
    chunks/chunk-NNN.json         case-record and truth digests, 20 cases per chunk
    case-sets/<case_set_id>.json  thin, ordered index of one official split
    index.json                    case universe, chunks and per-case locations
    provenance.json               dataset, extractor, runtime and truth method
    manifest.json                 entry point; its SHA-256 identifies the release
    release-receipt.json          manifest digest and packaging claims

Each unique case is stored once, so a case shared by two splits (41 cases are
in both the Full and the AoA-extrapolation test splits) is not duplicated; the
case-set indexes only reference it. Unlike HiLiftAeroML, the truth artifacts
stay in the extractor's JSON format, byte for byte: they are exactly the files
``score.py --ground-truth`` consumed. Every document is derived from the inputs
only, with no timestamps, so packaging is deterministic and ``check=True``
compares every byte of an existing release instead of writing.

Packaging and validation never read model predictions or open a release.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DATASET_ID = "airfrans"
RELEASE_ID = "airfrans-native-profile-truth-v1-candidate"
OFFICIAL_RELEASE_ID = "airfrans-native-profile-truth-v1"
TRUTH_FORMAT = "fluidsbench-airfrans-native-profile-truth-v1-candidate"
STATUS = "candidate_owner_review_required"
CASES_PER_CHUNK = 20
TRUTH_MODE = "ground_truth_reference"
PERFECT_COPY_MAX_DIFFERENCE = 0.0
EXTRACTOR_PATH = "examples/airfrans-profile-extraction/extract.py"
SOURCE_DATASET_KEYS = (
    "dataset_release",
    "archive_url",
    "archive_size_bytes",
    "archive_last_modified",
    "manifest_member",
    "manifest_sha256",
)


def _schema(kind: str) -> str:
    return f"airfrans-native-profile-truth-{kind}-v1-candidate"


ACTIVATION = {
    "owner_approval_complete": False,
    "published": False,
    "submissions_opened": False,
}


class ProfileTruthError(ValueError):
    """Raised when profile truth cannot be packaged or does not validate."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise ProfileTruthError(f"cannot encode canonical JSON: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _descriptor(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"file": relative, "sha256": sha256_file(path), "byte_size": path.stat().st_size}


def _write(path: Path, payload: bytes, *, check: bool) -> str:
    """Write ``payload`` once; with ``check`` require the existing bytes to match."""

    if check:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise ProfileTruthError(f"determinism check differs: {path}")
        return sha256_bytes(payload)
    if path.exists() or path.is_symlink():
        raise ProfileTruthError(f"refusing to overwrite release artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_bytes(payload)


def _write_json(path: Path, body: Mapping[str, Any], *, check: bool) -> str:
    return _write(path, canonical_json_bytes(body), check=check)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileTruthError(f"cannot read {label} {path}: {error}") from error


@dataclass(frozen=True)
class ProfileDefinition:
    """The parts of ``velocity-profiles-v1.json`` a truth file must satisfy."""

    profile_definition_id: str
    sha256: str
    station_ids: tuple[str, ...]
    quantity_ids: tuple[str, ...]
    panel_id: str
    sample_count: int
    sampling_runtime: Mapping[str, Any]


def load_profile_definition(path: Path) -> ProfileDefinition:
    body = _load_json(path, "profile definition")
    return ProfileDefinition(
        profile_definition_id=body["profile_definition_id"],
        sha256=sha256_file(path),
        station_ids=tuple(station["id"] for station in body["stations"]),
        quantity_ids=tuple(quantity["id"] for quantity in body["quantities"]),
        panel_id=body["metric_binding"]["panel_id"],
        sample_count=int(body["extraction"]["sample_count"]),
        sampling_runtime=body["sampling_runtime"],
    )


def runtime_versions(definition: ProfileDefinition) -> dict[str, str]:
    """The extractor's ``provenance.versions`` block the definition pins."""

    runtime = definition.sampling_runtime
    return {
        "airfrans_distribution": runtime["airfrans_distribution_version"],
        "airfrans_module": runtime["airfrans_module_version"],
        "numpy": runtime["numpy_version"],
        "pyvista": runtime["pyvista_version"],
        "vtk": runtime["vtk_version"],
    }


def check_truth_document(
    document: Mapping[str, Any], case_id: str, definition: ProfileDefinition
) -> dict[str, Any]:
    """Check one extracted truth file; return the facts its case record binds."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ProfileTruthError(f"{case_id}: {message}")

    provenance = document.get("provenance", {})
    require(provenance.get("mode") == TRUTH_MODE, f"provenance.mode is not {TRUTH_MODE!r}")
    require(
        provenance.get("profile_definition_id") == definition.profile_definition_id,
        "profile_definition_id differs from the pinned profile definition",
    )
    require(
        provenance.get("versions") == runtime_versions(definition),
        "runtime versions differ from the profile definition's sampling_runtime",
    )
    require(
        provenance.get("airfrans_runtime_source_file_sha256")
        == definition.sampling_runtime["airfrans_runtime_source_file_sha256"],
        "airfrans runtime source hash differs from the profile definition",
    )
    copy_check = provenance.get("perfect_copy_validation", {})
    require(
        copy_check.get("maximum_absolute_difference") == PERFECT_COPY_MAX_DIFFERENCE,
        "perfect-copy validation did not report an exact copy",
    )
    input_hashes = provenance.get("input_hashes", {})
    require(
        set(input_hashes) == {"internal_vtu", "aerofoil_vtp"},
        "source-mesh hashes are missing",
    )
    cases = document.get("cases", [])
    require(len(cases) == 1 and cases[0].get("case_id") == case_id, "must hold exactly this case")
    expected = [
        (station_id, quantity_id)
        for station_id in definition.station_ids
        for quantity_id in definition.quantity_ids
    ]
    observed = [(series.get("station_id"), series.get("quantity_id")) for series in cases[0]["series"]]
    require(sorted(observed) == sorted(expected) and len(observed) == len(expected),
            f"series must be exactly {expected}")
    for series in cases[0]["series"]:
        require(series.get("panel_id") == definition.panel_id, "series panel_id differs")
        for key in ("coordinate", "prediction"):
            values = series.get(key)
            require(
                isinstance(values, list)
                and len(values) == definition.sample_count
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values),
                f"{series.get('station_id')}/{series.get('quantity_id')} {key} must hold "
                f"{definition.sample_count} finite samples",
            )
    return {
        "source_mesh_sha256": {
            "internal_vtu": input_hashes["internal_vtu"],
            "aerofoil_vtp": input_hashes["aerofoil_vtp"],
        },
        "extractor_sha256": provenance.get("extractor_sha256"),
        "series_count": len(expected),
        "samples_per_series": definition.sample_count,
    }


@dataclass(frozen=True)
class CaseSetSource:
    """One official split and the directory holding its extracted truth files."""

    split_file: Path
    truth_dir: Path
    scoring_support_manifest: Path | None = None


def _case_universe_sha256(case_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes({"dataset_id": DATASET_ID, "case_ids": list(case_ids)}))


def _scoring_support_binding(manifest_path: Path, case_set_id: str) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "scoring-support manifest")
    matches = [entry for entry in manifest.get("case_sets", []) if entry.get("id") == case_set_id]
    if len(matches) != 1:
        raise ProfileTruthError(
            f"scoring-support manifest {manifest_path} has no single case set {case_set_id!r}"
        )
    index_path = manifest_path.parent / matches[0]["index_file"]
    return {
        "release_id": manifest["release_id"],
        "case_set_index_sha256": sha256_file(index_path),
    }


def build_release(
    *,
    sources: Sequence[CaseSetSource],
    profile_definition_path: Path,
    repository_root: Path,
    extractor_commit: str,
    extractor_commit_file_sha256: str,
    output_root: Path,
    check: bool = False,
) -> dict[str, Any]:
    """Write (or with ``check`` compare) the complete release; return its manifest.

    ``extractor_commit_file_sha256`` is the SHA-256 of ``extract.py`` at
    ``extractor_commit``; it must equal the extractor hash every truth file
    records, so the recorded commit really produced the truth.
    """

    repository_root = repository_root.resolve()
    definition = load_profile_definition(profile_definition_path)
    payloads: dict[str, bytes] = {}
    facts: dict[str, dict[str, Any]] = {}
    case_sets: list[dict[str, Any]] = []
    dataset_sources: list[dict[str, Any]] = []
    for source in sources:
        split = _load_json(source.split_file, "split file")
        dataset_sources.append(
            {key: split["source"][key] for key in SOURCE_DATASET_KEYS}
        )
        case_set_id = split["case_set_id"]
        if any(entry["case_set_id"] == case_set_id for entry in case_sets):
            raise ProfileTruthError(f"case set {case_set_id!r} is given twice")
        for case_id in split["case_ids"]:
            payload = (source.truth_dir / f"{case_id}.json").read_bytes()
            if case_id in payloads:
                if payloads[case_id] != payload:
                    raise ProfileTruthError(
                        f"{case_id} is in several splits but its truth files differ"
                    )
                continue
            payloads[case_id] = payload
            facts[case_id] = check_truth_document(json.loads(payload), case_id, definition)
        case_sets.append(
            {
                "case_set_id": case_set_id,
                "split_id": split["split_id"],
                "case_ids": list(split["case_ids"]),
                "split_file": source.split_file.resolve().relative_to(repository_root).as_posix(),
                "split_file_sha256": sha256_file(source.split_file),
                "scoring_support": (
                    _scoring_support_binding(source.scoring_support_manifest, case_set_id)
                    if source.scoring_support_manifest is not None
                    else None
                ),
            }
        )
    if any(entry != dataset_sources[0] for entry in dataset_sources):
        raise ProfileTruthError("the split files name different source dataset releases")
    extractor_hashes = {fact["extractor_sha256"] for fact in facts.values()}
    if len(extractor_hashes) != 1:
        raise ProfileTruthError(f"truth files come from several extractors: {sorted(extractor_hashes)}")
    extractor_sha256 = extractor_hashes.pop()
    if extractor_sha256 != extractor_commit_file_sha256:
        raise ProfileTruthError(
            f"{EXTRACTOR_PATH} at {extractor_commit} does not match the extractor "
            "recorded in the truth files"
        )

    universe = sorted(payloads)
    records: dict[str, tuple[dict[str, Any], str]] = {}
    for case_id in universe:
        artifact = f"cases/{case_id}.json"
        artifact_sha = _write(output_root / artifact, payloads[case_id], check=check)
        record = {
            "schema": _schema("case"),
            "schema_version": 1,
            "format": TRUTH_FORMAT,
            "status": STATUS,
            "dataset_id": DATASET_ID,
            "release_id": RELEASE_ID,
            "case_id": case_id,
            "profile_definition": {
                "id": definition.profile_definition_id,
                "sha256": definition.sha256,
            },
            "truth_artifact": {
                "file": artifact,
                "sha256": artifact_sha,
                "byte_size": len(payloads[case_id]),
                "format": "airfrans-profile-extraction-json",
                "mode": TRUTH_MODE,
            },
            "source_mesh_sha256": facts[case_id]["source_mesh_sha256"],
            "extractor_sha256": extractor_sha256,
            "series": {
                "station_ids": list(definition.station_ids),
                "quantity_ids": list(definition.quantity_ids),
                "series_count": facts[case_id]["series_count"],
                "samples_per_series": facts[case_id]["samples_per_series"],
            },
            "truth_authority": {
                "kind": "public_dataset_extraction",
                "prediction_bearing_evaluator_outputs_used_as_source": False,
                "perfect_copy_maximum_absolute_difference": PERFECT_COPY_MAX_DIFFERENCE,
            },
        }
        record_sha = _write_json(output_root / "case-records" / f"{case_id}.json", record, check=check)
        records[case_id] = (record, record_sha)

    chunks: list[dict[str, Any]] = []
    case_locations: dict[str, dict[str, Any]] = {}
    for chunk_index, start in enumerate(range(0, len(universe), CASES_PER_CHUNK)):
        selected = universe[start : start + CASES_PER_CHUNK]
        filename = f"chunks/chunk-{chunk_index:03d}.json"
        chunk_body = {
            "schema": _schema("chunk"),
            "schema_version": 1,
            "format": TRUTH_FORMAT,
            "dataset_id": DATASET_ID,
            "release_id": RELEASE_ID,
            "cases": [
                {
                    "case_id": case_id,
                    "case_record_sha256": records[case_id][1],
                    "truth_artifact_sha256": records[case_id][0]["truth_artifact"]["sha256"],
                }
                for case_id in selected
            ],
        }
        chunk_sha = _write_json(output_root / filename, chunk_body, check=check)
        chunks.append(
            {
                "file": filename,
                "sha256": chunk_sha,
                "byte_size": (output_root / filename).stat().st_size,
                "case_count": len(selected),
                "case_ids": list(selected),
            }
        )
        for case_id in selected:
            case_locations[case_id] = {
                "chunk_file": filename,
                "chunk_sha256": chunk_sha,
                "case_record_file": f"case-records/{case_id}.json",
                "case_record_sha256": records[case_id][1],
                "truth_artifact": records[case_id][0]["truth_artifact"],
            }

    index = {
        "schema": _schema("index"),
        "schema_version": 1,
        "format": TRUTH_FORMAT,
        "status": STATUS,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "profile_definition": {"id": definition.profile_definition_id, "sha256": definition.sha256},
        "case_universe_sha256": _case_universe_sha256(universe),
        "case_count": len(universe),
        "case_ids": universe,
        "chunks": chunks,
        "case_locations": case_locations,
    }
    index_sha = _write_json(output_root / "index.json", index, check=check)

    case_set_descriptors: list[dict[str, Any]] = []
    for entry in case_sets:
        filename = f"case-sets/{entry['case_set_id']}.json"
        body = {
            "schema": _schema("case-set-index"),
            "schema_version": 1,
            "format": TRUTH_FORMAT,
            "dataset_id": DATASET_ID,
            "release_id": RELEASE_ID,
            "case_set_id": entry["case_set_id"],
            "split_id": entry["split_id"],
            "official_split": {"file": entry["split_file"], "sha256": entry["split_file_sha256"]},
            "scoring_support": entry["scoring_support"],
            "case_count": len(entry["case_ids"]),
            "case_ids": entry["case_ids"],
            "master_index_sha256": index_sha,
            "resolution": "resolve every case through master index case_locations",
        }
        digest = _write_json(output_root / filename, body, check=check)
        case_set_descriptors.append(
            {
                "case_set_id": entry["case_set_id"],
                "split_id": entry["split_id"],
                "case_count": len(entry["case_ids"]),
                "file": filename,
                "sha256": digest,
                "byte_size": (output_root / filename).stat().st_size,
            }
        )

    provenance = {
        "schema": _schema("provenance"),
        "schema_version": 1,
        "status": STATUS,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "source_dataset": dataset_sources[0],
        "profile_definition": {
            "id": definition.profile_definition_id,
            "file": "benchmark-specs/airfrans/velocity-profiles-v1.json",
            "sha256": definition.sha256,
        },
        "extractor": {
            "file": EXTRACTOR_PATH,
            "sha256": extractor_sha256,
            "code_commit": extractor_commit,
            "command": (
                f"python3 {EXTRACTOR_PATH} --dataset-root <AirfRANS>/Dataset "
                "--case-name <case_id> --output <case_id>.json"
            ),
        },
        "runtime": {
            "versions": runtime_versions(definition),
            "airfrans_runtime_source_file_sha256": definition.sampling_runtime[
                "airfrans_runtime_source_file_sha256"
            ],
            "requirements_file": definition.sampling_runtime["requirements_file"],
        },
        "case_universe_sha256": _case_universe_sha256(universe),
        "truth_method": {
            "source": "U from each case's native _internal.vtu, sampled by airfrans.Simulation.boundary_layer",
            "mode": f"{TRUTH_MODE}: every value is checked as an exact copy of the sampled CFD field",
            "case_deduplication": "one artifact per unique case; case-set indexes are references only",
            "artifact_format": "the extractor's JSON, byte-identical to the files used for scoring",
        },
        "participant_artifacts_remain_prediction_only": True,
    }
    provenance_sha = _write_json(output_root / "provenance.json", provenance, check=check)

    manifest = {
        "schema": _schema("manifest"),
        "schema_version": 1,
        "format": TRUTH_FORMAT,
        "status": STATUS,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "profile_definition": {"id": definition.profile_definition_id, "sha256": definition.sha256},
        "case_count": len(universe),
        "case_set_count": len(case_set_descriptors),
        "master_index": {
            "file": "index.json",
            "sha256": index_sha,
            "byte_size": (output_root / "index.json").stat().st_size,
        },
        "provenance": {
            "file": "provenance.json",
            "sha256": provenance_sha,
            "byte_size": (output_root / "provenance.json").stat().st_size,
        },
        "case_sets": case_set_descriptors,
        "storage": {
            "truth_artifact_bytes": sum(len(payload) for payload in payloads.values()),
            "case_artifacts_duplicated_across_case_sets": 0,
        },
        "activation": dict(ACTIVATION),
    }
    manifest_sha = _write_json(output_root / "manifest.json", manifest, check=check)
    receipt = {
        "schema": _schema("release-receipt"),
        "schema_version": 1,
        "status": "complete_candidate_not_published",
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "manifest_sha256": manifest_sha,
        "case_count": len(universe),
        "case_set_count": len(case_set_descriptors),
        "deterministic_packaging": True,
        **ACTIVATION,
    }
    _write_json(output_root / "release-receipt.json", receipt, check=check)
    _require_exact_tree(output_root, expected_release_files(universe, len(chunks), case_sets))
    return manifest


def expected_release_files(
    case_ids: Sequence[str], chunk_count: int, case_sets: Sequence[Mapping[str, Any]]
) -> set[str]:
    files = {"manifest.json", "index.json", "provenance.json", "release-receipt.json"}
    files |= {f"cases/{case_id}.json" for case_id in case_ids}
    files |= {f"case-records/{case_id}.json" for case_id in case_ids}
    files |= {f"chunks/chunk-{index:03d}.json" for index in range(chunk_count)}
    files |= {f"case-sets/{entry['case_set_id']}.json" for entry in case_sets}
    return files


def _require_exact_tree(output_root: Path, expected_files: set[str]) -> None:
    """Reject missing or extra files, links, and interrupted-write debris."""

    observed: set[str] = set()
    for path in output_root.rglob("*"):
        relative = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise ProfileTruthError(f"profile truth release contains a symlink: {relative}")
        if path.is_file():
            observed.add(relative)
        elif not path.is_dir():
            raise ProfileTruthError(f"profile truth release contains a non-file entry: {relative}")
    if observed != expected_files:
        raise ProfileTruthError(
            "profile truth release inventory differs: "
            f"extra={sorted(observed - expected_files)[:5]}, "
            f"missing={sorted(expected_files - observed)[:5]}"
        )


def validate_release(
    *, release_root: Path, profile_definition_path: Path, repository_root: Path,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Check the whole release graph from the manifest down; return a summary.

    Every file must be listed and match its recorded SHA-256 and size; every
    truth file must pass ``check_truth_document`` against the pinned profile
    definition; each case set must equal its official split file, in order;
    and the case universe must be exactly the union of the case sets.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ProfileTruthError(message)

    def check_descriptor(entry: Mapping[str, Any], label: str) -> Any:
        path = release_root / entry["file"]
        require(path.is_file(), f"{label} is missing: {entry['file']}")
        payload = path.read_bytes()
        require(sha256_bytes(payload) == entry["sha256"], f"{label} SHA-256 differs")
        require(len(payload) == entry["byte_size"], f"{label} byte size differs")
        return json.loads(payload)

    definition = load_profile_definition(profile_definition_path)
    manifest_path = release_root / "manifest.json"
    manifest = _load_json(manifest_path, "release manifest")
    release_id = manifest.get("release_id")
    require(release_id in {RELEASE_ID, OFFICIAL_RELEASE_ID}, "unknown profile-truth release identity")
    official = release_id == OFFICIAL_RELEASE_ID
    activation = (
        {"owner_approval_complete": True, "published": manifest.get("publication_status") != "prepared_dev_only", "submissions_opened": False}
        if official else ACTIVATION
    )
    if official:
        require(manifest.get("status") == "official", "official truth manifest status differs")
        require(manifest.get("format") == "fluidsbench-airfrans-native-profile-truth-v1", "official truth format differs")
        approval = manifest.get("owner_approval", {})
        require(all(isinstance(approval.get(key), str) and approval[key].strip()
                    for key in ("approved_by", "approved_at", "pull_request_url")),
                "official truth requires owner approval")
    manifest_sha = sha256_file(manifest_path)
    receipt = _load_json(release_root / "release-receipt.json", "release receipt")
    require(receipt.get("manifest_sha256") == manifest_sha, "release receipt manifest SHA-256 differs")
    for document, label in ((manifest, "manifest"), (receipt, "release receipt")):
        require(document.get("release_id") == release_id, f"{label} release_id differs")
        require(document.get("dataset_id") == DATASET_ID, f"{label} dataset_id differs")
        for key, value in activation.items():
            flags = document.get("activation", document)
            require(flags.get(key) is value, f"{label} must declare {key}={value}")
    require(
        manifest.get("profile_definition") == {"id": definition.profile_definition_id, "sha256": definition.sha256},
        "manifest profile definition differs from the pinned profile definition",
    )

    index = check_descriptor(manifest["master_index"], "master index")
    provenance = check_descriptor(manifest["provenance"], "provenance")
    universe = index["case_ids"]
    require(universe == sorted(set(universe)), "case universe must be sorted and unique")
    require(index["case_count"] == len(universe) == manifest["case_count"], "case counts differ")
    require(
        index["case_universe_sha256"] == _case_universe_sha256(universe)
        == provenance["case_universe_sha256"],
        "case universe SHA-256 differs",
    )
    require(
        provenance["profile_definition"]["sha256"] == definition.sha256,
        "provenance profile definition differs",
    )
    require(
        provenance["runtime"]["versions"] == runtime_versions(definition),
        "provenance runtime versions differ from the profile definition",
    )

    chunked: list[str] = []
    for chunk in index["chunks"]:
        body = check_descriptor(chunk, chunk["file"])
        require([case["case_id"] for case in body["cases"]] == chunk["case_ids"], f"{chunk['file']} cases differ")
        for case in body["cases"]:
            location = index["case_locations"][case["case_id"]]
            require(location["chunk_file"] == chunk["file"], f"{case['case_id']} chunk location differs")
            require(location["chunk_sha256"] == chunk["sha256"], f"{case['case_id']} chunk SHA-256 differs")
            require(
                case["case_record_sha256"] == location["case_record_sha256"]
                and case["truth_artifact_sha256"] == location["truth_artifact"]["sha256"],
                f"{case['case_id']} chunk digests differ from the master index",
            )
        chunked.extend(chunk["case_ids"])
    require(chunked == universe, "chunks must cover the case universe exactly once, in order")
    require(set(index["case_locations"]) == set(universe), "case locations differ from the universe")

    extractor_hashes: set[str] = set()
    for case_id in universe:
        location = index["case_locations"][case_id]
        record_path = release_root / location["case_record_file"]
        require(sha256_file(record_path) == location["case_record_sha256"], f"{case_id} case record differs")
        record = _load_json(record_path, "case record")
        require(record["case_id"] == case_id and record["release_id"] == release_id, f"{case_id} record identity differs")
        require(record["truth_artifact"] == location["truth_artifact"], f"{case_id} truth descriptor differs")
        if metadata_only:
            extractor_hashes.add(record["extractor_sha256"])
            continue
        truth = check_descriptor(record["truth_artifact"], f"{case_id} truth artifact")
        facts = check_truth_document(truth, case_id, definition)
        require(record["source_mesh_sha256"] == facts["source_mesh_sha256"], f"{case_id} source-mesh hashes differ")
        require(record["extractor_sha256"] == facts["extractor_sha256"], f"{case_id} extractor hash differs")
        extractor_hashes.add(facts["extractor_sha256"])
    require(
        extractor_hashes == {provenance["extractor"]["sha256"]},
        "truth files do not all come from the extractor recorded in provenance",
    )

    covered: set[str] = set()
    case_set_summary: dict[str, int] = {}
    for descriptor in manifest["case_sets"]:
        body = check_descriptor(descriptor, descriptor["file"])
        require(body["master_index_sha256"] == manifest["master_index"]["sha256"], f"{descriptor['file']} master index differs")
        split_path = repository_root / body["official_split"]["file"]
        require(sha256_file(split_path) == body["official_split"]["sha256"], f"{descriptor['file']} official split file differs")
        split = _load_json(split_path, "split file")
        require(
            body["case_ids"] == split["case_ids"] and body["case_set_id"] == split["case_set_id"],
            f"{descriptor['file']} must list the official split's cases in order",
        )
        require(set(body["case_ids"]) <= set(universe), f"{descriptor['file']} references unknown cases")
        covered |= set(body["case_ids"])
        case_set_summary[body["case_set_id"]] = len(body["case_ids"])
    require(covered == set(universe), "case universe must equal the union of the case sets")

    expected = expected_release_files(universe, len(index["chunks"]), [
        {"case_set_id": descriptor["case_set_id"]} for descriptor in manifest["case_sets"]
    ])
    if metadata_only:
        expected -= {f"cases/{case_id}.json" for case_id in universe}
    _require_exact_tree(release_root, expected)
    return {
        "release_id": release_id,
        "manifest_sha256": manifest_sha,
        "case_count": len(universe),
        "case_sets": case_set_summary,
        "file_count": len(expected),
        "truth_arrays_checked": not metadata_only,
    }


def resolve_case_set_truth(release_root: Path, case_set_id: str) -> list[Path]:
    """Truth files of one case set, in official order, resolved via the master index."""

    manifest = _load_json(release_root / "manifest.json", "release manifest")
    matches = [entry for entry in manifest["case_sets"] if entry["case_set_id"] == case_set_id]
    if len(matches) != 1:
        raise ProfileTruthError(f"release has no case set {case_set_id!r}")
    case_set = _load_json(release_root / matches[0]["file"], "case-set index")
    index = _load_json(release_root / manifest["master_index"]["file"], "master index")
    return [
        release_root / index["case_locations"][case_id]["truth_artifact"]["file"]
        for case_id in case_set["case_ids"]
    ]
