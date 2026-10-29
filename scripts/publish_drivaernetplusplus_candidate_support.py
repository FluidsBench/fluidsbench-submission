#!/usr/bin/env python3
"""Publish the candidate DrivAerNet++ pressure scoring support (maintainer tool).

Two resumable stages, both run in the pinned participant environment
(``requirements-drivaernetplusplus-participant.txt``):

``build``
    Read every official_test ``SurfacePressureVTK`` file once and write, per
    case, ``support.npz`` (dual-area weights, reference pressure, frozen
    profile-station point indices) and ``case-support.json``.  Cases already
    built and verified are skipped, so an interrupted run resumes.

``package``
    Group built cases into downloadable archives, write the distribution
    index ``support-distribution-v1.json``, the public source identity, the
    per-case records under ``case-artifacts/``, and the hashed
    manifest -> case-index -> chunk chain, then self-check the result.

Example::

    python scripts/publish_drivaernetplusplus_candidate_support.py build \\
        --vtk-dir /data/DrivAerNetPlusPlus/SurfacePressureVTK --work /scratch/dnpp-build --workers 48
    python scripts/publish_drivaernetplusplus_candidate_support.py package \\
        --work /scratch/dnpp-build --archive-dir /scratch/dnpp-archives
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.drivaernetplusplus.contract import (  # noqa: E402
    CASE_SET_ID,
    DATASET_ID,
    DATASET_VERSION,
    DEFAULT_BASE_URL,
    DISTRIBUTION_PATH,
    DISTRIBUTION_SCHEMA,
    EVALUATION_REFERENCE_VERSION,
    QUANTITY_ID,
    RELEASE_DIR,
    RELEASE_ID,
    RELEASE_TAG,
    SOURCE_IDENTITY_PATH,
    SOURCE_IDENTITY_SCHEMA,
    SUPPORT_ID,
    ContractError,
    official_split,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from reference.drivaernetplusplus.native import SUPPORT_RULES_VERSION  # noqa: E402
from reference.drivaernetplusplus.support import (  # noqa: E402
    build_case_support,
    deterministic_zip_info,
    load_npz_bytes,
    validate_case_support,
)

CASES_PER_CHUNK = 50
DATAVERSE_DOI = "doi:10.7910/DVN/K7PWNJ"
DATAVERSE_VERSION = "1.0"
METRIC_BINDINGS = [
    {
        "metric_id": "surface_pressure_rel_l2",
        "quantity_id": QUANTITY_ID,
        "reduction": "relative_l2_percent",
        "weighting": "support_weights",
        "dataset_weighting": "surface_point_dual_area",
        "aggregation": "per_geometry_then_macro_average",
        "case_evidence": "metric_value",
    },
    {
        "metric_id": "surface_pressure_equal_entity_rel_l2",
        "quantity_id": QUANTITY_ID,
        "reduction": "relative_l2_percent",
        "weighting": "uniform",
        "dataset_weighting": "surface_points_equal",
        "aggregation": "per_geometry_then_macro_average",
        "case_evidence": "metric_value",
    },
    {
        "metric_id": "surface_pressure_rel_l1",
        "quantity_id": QUANTITY_ID,
        "reduction": "relative_l1_percent",
        "weighting": "support_weights",
        "dataset_weighting": "surface_point_dual_area",
        "aggregation": "per_geometry_then_macro_average",
        "case_evidence": "metric_value",
    },
    {
        "metric_id": "surface_pressure_r2",
        "quantity_id": QUANTITY_ID,
        "reduction": "r2",
        "weighting": "uniform",
        "dataset_weighting": "surface_points_equal",
        "aggregation": "flatten_all_aligned_field_values",
        "case_evidence": "aggregate_only",
    },
]


class PublisherError(ValueError):
    """Raised when the public files cannot produce the agreed release."""


def source_path(vtk_dir: Path, case_id: str) -> Path:
    for name in (f"{case_id}.vtk", f"DrivAer_{case_id}.vtk"):
        if (vtk_dir / name).is_file():
            return vtk_dir / name
    raise PublisherError(f"missing public surface-pressure file for {case_id} in {vtk_dir}")


def _built(work: Path, case_id: str) -> bool:
    record_path = work / "cases" / case_id / "case-support.json"
    npz_path = work / "cases" / case_id / "support.npz"
    if not (record_path.is_file() and npz_path.is_file()):
        return False
    record = read_json(record_path)
    payload = npz_path.read_bytes()
    if sha256_bytes(payload) != record["support_npz"]["sha256"]:
        return False
    validate_case_support(record, load_npz_bytes(payload, case_id), case_id)
    return True


def _build_one(case_id: str, vtk_path: str, work: str) -> str:
    destination = Path(work) / "cases" / case_id
    payload, record = build_case_support(case_id, Path(vtk_path))
    validate_case_support(record, load_npz_bytes(payload, case_id), case_id)
    with tempfile.TemporaryDirectory(prefix=f".{case_id}-", dir=Path(work) / "cases") as temporary:
        staging = Path(temporary) / case_id
        staging.mkdir()
        (staging / "support.npz").write_bytes(payload)
        write_json(staging / "case-support.json", record)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    return f"{case_id}: {record['native']['point_count']} points, area {record['native']['total_dual_area_m2']:.4f} m2"


def build(vtk_dir: Path, work: Path, case_ids: Sequence[str], workers: int) -> int:
    (work / "cases").mkdir(parents=True, exist_ok=True)
    pending = [case_id for case_id in case_ids if not _built(work, case_id)]
    print(f"{len(case_ids) - len(pending)} case(s) already built; building {len(pending)}", flush=True)
    paths = {case_id: str(source_path(vtk_dir, case_id)) for case_id in pending}
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, c, paths[c], str(work)): c for c in pending}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            case_id = futures[future]
            try:
                print(f"[{done}/{len(pending)}] {future.result()}", flush=True)
            except Exception as error:  # report every failing case, then fail
                failures.append(case_id)
                print(f"[{done}/{len(pending)}] {case_id}: FAILED {error}", flush=True)
    if failures:
        raise PublisherError(f"{len(failures)} case(s) failed: {failures[:10]}")
    return len(pending)


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[offset : offset + size]) for offset in range(0, len(values), size)]


def package(
    work: Path,
    archive_dir: Path,
    case_ids: Sequence[str],
    *,
    base_url: str,
    published_at: str,
    consume: bool,
) -> None:
    if not base_url.startswith("https://") or not base_url.endswith("/"):
        raise PublisherError("base URL must be an https:// prefix ending in '/'")
    archive_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        # A consumed case keeps its record; its arrays live only in the archive.
        if not (work / "cases" / case_id / "case-support.json").is_file():
            raise PublisherError(f"{case_id} is not built; run the build stage first")
        records[case_id] = read_json(work / "cases" / case_id / "case-support.json")

    archives = []
    archive_of: dict[str, dict[str, Any]] = {}
    for index, members in enumerate(_chunks(case_ids, CASES_PER_CHUNK)):
        name = f"{DATASET_ID}-support-{index:03d}.zip"
        path = archive_dir / name
        temporary = path.with_suffix(".zip.part")
        if _archive_matches(path, members, work, records):
            # Manifest-only changes never re-roll published archive bytes.
            print(f"reusing verified {name}", flush=True)
        else:
            _write_archive(temporary, members, work, records)
            temporary.rename(path)
        if consume:
            for case_id in members:
                (work / "cases" / case_id / "support.npz").unlink(missing_ok=True)
        entry = {
            "file": name,
            "url": base_url + name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "case_ids": members,
        }
        archives.append(entry)
        for case_id in members:
            archive_of[case_id] = entry
        print(f"archived {name}: {len(members)} cases, {entry['size_bytes'] / 1e6:.1f} MB", flush=True)
    _write_release_chain(work, case_ids, records, archives, archive_of, published_at=published_at)


def _archive_matches(path: Path, members: Sequence[str], work: Path, records: dict[str, dict[str, Any]]) -> bool:
    if not path.is_file():
        return False
    expected = [name for case_id in members for name in (f"{case_id}/support.npz", f"{case_id}/case-support.json")]
    with zipfile.ZipFile(path) as archive:
        if [item.filename for item in archive.infolist()] != expected:
            return False
        for case_id in members:
            if archive.read(f"{case_id}/case-support.json") != (work / "cases" / case_id / "case-support.json").read_bytes():
                return False
            if sha256_bytes(archive.read(f"{case_id}/support.npz")) != records[case_id]["support_npz"]["sha256"]:
                return False
    return True


def _write_archive(temporary: Path, members: Sequence[str], work: Path, records: dict[str, dict[str, Any]]) -> None:
    with zipfile.ZipFile(temporary, "w") as archive:
        for case_id in members:
            case_dir = work / "cases" / case_id
            payload = (case_dir / "support.npz").read_bytes()
            if sha256_bytes(payload) != records[case_id]["support_npz"]["sha256"]:
                raise PublisherError(f"{case_id}: built support changed before packaging")
            archive.writestr(deterministic_zip_info(f"{case_id}/support.npz", compress=False), payload)
            archive.writestr(
                deterministic_zip_info(f"{case_id}/case-support.json", compress=False),
                (case_dir / "case-support.json").read_bytes(),
            )


def _write_release_chain(
    work: Path,
    case_ids: Sequence[str],
    records: dict[str, dict[str, Any]],
    archives: list[dict[str, Any]],
    archive_of: dict[str, dict[str, Any]],
    *,
    published_at: str,
) -> None:
    """Per-case records retained in Git, and the manifest chain that binds them."""

    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    case_set_dir = RELEASE_DIR / "case-sets" / CASE_SET_ID
    chunk_descriptors = []
    for chunk_index, members in enumerate(_chunks(case_ids, CASES_PER_CHUNK)):
        cases = []
        for case_id in members:
            record = records[case_id]
            record_path = case_set_dir / "case-artifacts" / case_id / "case-support.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            # Byte-identical to the archived copy, which the evaluator checks.
            record_bytes = (work / "cases" / case_id / "case-support.json").read_bytes()
            record_path.write_bytes(record_bytes)
            record_sha = sha256_bytes(record_bytes)
            archive = archive_of[case_id]
            cases.append(
                {
                    "case_id": case_id,
                    "support_instances": [
                        {
                            "support_id": SUPPORT_ID,
                            "entity_count": record["native"]["point_count"],
                            "artifacts": [
                                {
                                    "role": "candidate-case-support-manifest",
                                    "path": f"case-artifacts/{case_id}/case-support.json",
                                    "sha256": record_sha,
                                    "byte_size": record_path.stat().st_size,
                                    "format": "json",
                                },
                                {
                                    "role": "candidate-support-arrays",
                                    "archive": {
                                        "url": archive["url"],
                                        "sha256": archive["sha256"],
                                        "member": record["support_npz"]["member"],
                                    },
                                    "sha256": record["support_npz"]["sha256"],
                                    "byte_size": record["support_npz"]["size_bytes"],
                                    "format": "npz",
                                },
                            ],
                            "parameters": {
                                "association": "PointData",
                                "selection": "all_native_surface_points",
                                "source_file": record["source"]["file_name"],
                                "source_sha256": record["source"]["sha256"],
                                "source_size_bytes": record["source"]["size_bytes"],
                                "points_float32_sha256": record["native"]["points_float32_sha256"],
                                "total_dual_area_m2": record["native"]["total_dual_area_m2"],
                                "rules_version": SUPPORT_RULES_VERSION,
                            },
                            "expected_support_sha256": record_sha,
                        }
                    ],
                }
            )
        chunk_name = f"chunk-{chunk_index:03d}.json"
        chunk_sha = write_json(
            case_set_dir / chunk_name,
            {
                "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
                "schema_version": "1.0",
                "release_id": RELEASE_ID,
                "dataset_id": DATASET_ID,
                "case_set_id": CASE_SET_ID,
                "cases": cases,
            },
        )
        chunk_descriptors.append(
            {"file": chunk_name, "sha256": chunk_sha, "case_count": len(members), "case_ids": members}
        )
    index_sha = write_json(
        case_set_dir / "index.json",
        {
            "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
            "schema_version": "1.0",
            "release_id": RELEASE_ID,
            "dataset_id": DATASET_ID,
            "case_set_id": CASE_SET_ID,
            "case_count": len(case_ids),
            "chunks": chunk_descriptors,
        },
    )
    manifest_sha = write_json(
        RELEASE_DIR / "manifest.json",
        {
            "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
            "schema_version": "1.0",
            "release_id": RELEASE_ID,
            "status": "candidate",
            "published_at": published_at,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "evaluation_reference_version": EVALUATION_REFERENCE_VERSION,
            "coordinate_frame": {
                "id": "drivaernetplusplus-physical-v1",
                "axis_order": ["x", "y", "z"],
                "handedness": "right",
                "length_unit": "m",
                "normalization": "none for scoring; profile coordinates use x/L and z/L per design",
            },
            "supports": [
                {
                    "id": SUPPORT_ID,
                    "domain": "surface",
                    "location_definition": {
                        "mode": "native_entities",
                        "format": "vtk_legacy",
                        "artifact_role": "candidate-support-arrays",
                        "entity": "point",
                        "selection": {"kind": "all_entities"},
                        "coordinate_rule": "source_point",
                        "ordering": "source_entity_index",
                        "support_id_rule": {"kind": "source_entity_index"},
                        # Mass-lumped fan-triangulated dual areas, published as
                        # an array so scoring never recomputes geometry.
                        "weight_rule": {
                            "kind": "artifact_field",
                            "artifact_role": "candidate-support-arrays",
                            "field": "dual_area_m2",
                        },
                    },
                    "quantities": [
                        {
                            "id": QUANTITY_ID,
                            "unit": "m2/s2",
                            "components": [
                                {
                                    "id": "p",
                                    "target_field": "p",
                                    "target_association": "point_data",
                                    "prediction_field": "p",
                                }
                            ],
                        }
                    ],
                    "metric_bindings": METRIC_BINDINGS,
                    "required_coverage": {
                        "count_fraction": 1.0,
                        "weight_fraction": 1.0,
                        "unmapped_count": 0,
                    },
                    "extrapolation_policy": "forbidden",
                }
            ],
            "case_sets": [
                {
                    "id": CASE_SET_ID,
                    "case_count": len(case_ids),
                    "index_file": f"case-sets/{CASE_SET_ID}/index.json",
                    "index_sha256": index_sha,
                }
            ],
            "notes": (
                "Candidate native DrivAerNet++ surface-pressure support: every PointData point "
                f"of the {len(case_ids)} official_test SurfacePressureVTK files (Harvard Dataverse "
                f"{DATAVERSE_DOI} version {DATAVERSE_VERSION}), reference pressure p as stored, "
                "and mass-lumped fan-triangulated dual-area weights. Large arrays are distributed "
                "through support-distribution-v1.json. Not an official release: no owner approval "
                "or publication-validation record."
            ),
        },
    )
    write_json(
        SOURCE_IDENTITY_PATH,
        {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "dataset_id": DATASET_ID,
            "source": {
                "dataverse_doi": DATAVERSE_DOI,
                "dataverse_version": DATAVERSE_VERSION,
                "access_route": "Harvard Dataverse family ZIP archives (Globus-backed); extract SurfacePressureVTK files locally",
                "licence": "CC BY-NC 4.0",
            },
            "file_name_rule": "<case_id>.vtk or DrivAer_<case_id>.vtk",
            "case_count": len(case_ids),
            "total_bytes": sum(records[c]["source"]["size_bytes"] for c in case_ids),
            "cases": [
                {
                    "case_id": case_id,
                    "sha256": records[case_id]["source"]["sha256"],
                    "size_bytes": records[case_id]["source"]["size_bytes"],
                    "point_count": records[case_id]["native"]["point_count"],
                }
                for case_id in case_ids
            ],
        },
    )
    write_json(
        DISTRIBUTION_PATH,
        {
            "schema": DISTRIBUTION_SCHEMA,
            "schema_version": 1,
            "status": "dev_prerelease",
            "dataset_id": DATASET_ID,
            "scientific_release_id": RELEASE_ID,
            "scientific_manifest_sha256": manifest_sha,
            "rules_version": SUPPORT_RULES_VERSION,
            "release_tag": RELEASE_TAG,
            "licence": "CC BY-NC 4.0 (derived from the DrivAerNet++ public release)",
            "case_count": len(case_ids),
            "total_bytes": sum(item["size_bytes"] for item in archives),
            "archives": archives,
        },
    )
    print(f"wrote manifest {RELEASE_DIR / 'manifest.json'} ({manifest_sha})")
    print(f"wrote distribution index {DISTRIBUTION_PATH}")


def bind() -> str:
    """Point the specification's candidate binding at the freshly written manifest."""

    from reference.drivaernetplusplus.contract import SPEC_PATH

    spec = read_json(SPEC_PATH)
    binding = spec["scoring_support"]["candidate_manifest"]
    if binding.get("release_id") != RELEASE_ID:
        raise PublisherError("specification candidate binding names a different release")
    digest = sha256_file(RELEASE_DIR / "manifest.json")
    binding["manifest_sha256"] = digest
    write_json(SPEC_PATH, spec)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--vtk-dir", type=Path, required=True)
    build_parser.add_argument("--work", type=Path, required=True)
    build_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    build_parser.add_argument("--case-ids", nargs="+")
    package_parser = commands.add_parser("package")
    package_parser.add_argument("--work", type=Path, required=True)
    package_parser.add_argument("--archive-dir", type=Path, required=True)
    package_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    package_parser.add_argument("--published-at", default="2026-10-18T00:00:00Z")
    package_parser.add_argument(
        "--consume", action="store_true", help="delete each built support.npz once archived (saves disk)"
    )
    package_parser.add_argument(
        "--bind", action="store_true", help="update the specification's candidate manifest SHA-256"
    )
    args = parser.parse_args(argv)
    case_ids = list(official_split()["case_ids"])
    try:
        if args.command == "build":
            if args.case_ids:
                unknown = sorted(set(args.case_ids) - set(case_ids))
                if unknown:
                    parser.error(f"case IDs outside official_test: {unknown[:5]}")
                case_ids = list(args.case_ids)
            build(args.vtk_dir.resolve(), args.work.resolve(), case_ids, args.workers)
        else:
            package(
                args.work.resolve(),
                args.archive_dir.resolve(),
                case_ids,
                base_url=args.base_url,
                published_at=args.published_at,
                consume=args.consume,
            )
            if args.bind:
                print(f"bound submission-spec.json to manifest {bind()}")
    except (PublisherError, ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
