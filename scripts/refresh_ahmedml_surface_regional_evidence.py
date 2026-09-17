#!/usr/bin/env python3
"""Upgrade AhmedML fixture evidence with evaluator-owned surface regions.

Only the complete native surface fields are replayed.  Existing volume field,
force, profile, and metric evidence is preserved after exact field-statistic
comparison, so this avoids rereading the much larger native volume files.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.contract import (  # noqa: E402
    PROFILE_DEFINITION_SHA256,
    REGION_DEFINITION_SHA256,
    VOLUME_REGION_DEFINITION_SHA256,
    load_source_identity,
)
from reference.ahmedml.evaluator import (  # noqa: E402
    DEFAULT_ENCODED_CHUNK_BYTES,
    DEFAULT_MAX_PREDICTION_CHUNK_ROWS,
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    SURFACE_REGION_IDS,
    SURFACE_SUPPORT_ID,
    _evaluate_field,
    _field_evidence,
    _index_source,
    _open_verified_source,
    _required_array,
    _validate_manifest,
    surface_region_codes,
)
from reference.ahmedml.prediction_chunks import (  # noqa: E402
    DEFAULT_HASH_CHUNK_BYTES,
    DEFAULT_VALIDATION_BLOCK_ROWS,
    load_prediction_chunk_manifest,
)
from reference.ahmedml.support import (  # noqa: E402
    load_case_support,
    open_support_array,
)


class RefreshError(ValueError):
    """Raised when legacy evidence cannot be upgraded exactly."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RefreshError(f"{path} is not a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _refresh_case(
    case_id: str,
    *,
    dataset_root: Path,
    support_root: Path,
    prediction_root: Path,
    input_evidence_root: Path,
    output_evidence_root: Path,
    source_identity_path: Path,
) -> tuple[str, int]:
    identity = load_source_identity(source_identity_path)
    case = identity.case(case_id)
    support = load_case_support(
        support_root / case_id / "case-support.json", source_identity=identity
    )
    evidence = _read_json(input_evidence_root / f"{case_id}.json")
    if evidence.get("case_id") != case_id:
        raise RefreshError(f"{case_id} legacy evidence identity differs")
    old_regional = evidence.get("report_only_regional_diagnostics")
    if (
        not isinstance(old_regional, dict)
        or old_regional.get("definition_sha256")
        != VOLUME_REGION_DEFINITION_SHA256
        or old_regional.get("ranking_effect") != "none"
    ):
        raise RefreshError(f"{case_id} legacy volume regional evidence differs")

    manifest = load_prediction_chunk_manifest(
        prediction_root / case_id / "surface" / "manifest.json"
    )
    _validate_manifest(
        manifest,
        case_id=case_id,
        support_id=SURFACE_SUPPORT_ID,
        expected_count=case.surface_entity_count,
        maximum_rows=DEFAULT_MAX_PREDICTION_CHUNK_ROWS,
    )
    with (
        _open_verified_source(
            case.boundary, dataset_root, label=f"{case_id} boundary"
        ) as boundary,
        _open_verified_source(
            case.surface_cell_area,
            dataset_root,
            label=f"{case_id} surface area",
        ) as surface_area_source,
        open_support_array(support, "surface_area_vector") as area_vectors,
    ):
        surface_area_source.handle.seek(0)
        surface_areas = np.load(
            surface_area_source.handle, mmap_mode=None, allow_pickle=False
        )
        surface_area_source.assert_unchanged(
            context="while its area array was consumed"
        )
        boundary_index = _index_source(
            boundary.handle,
            expected_type="PolyData",
            expected_cells=case.surface_entity_count,
        )
        codes = surface_region_codes(area_vectors)
        pressure = _evaluate_field(
            stream=boundary.handle,
            vtk_index=boundary_index,
            array=_required_array(boundary_index, "pMean", 1),
            manifest=manifest,
            field_id="surface_pressure",
            field_name="pMean",
            weights=surface_areas,
            region_codes=codes,
            region_ids=SURFACE_REGION_IDS,
            region_domain="surface",
            sample_ids=None,
            sample_component=None,
            sample_scale=1.0,
            area_vectors=None,
            force_sign=None,
            hash_chunk_bytes=DEFAULT_HASH_CHUNK_BYTES,
            validation_block_rows=DEFAULT_VALIDATION_BLOCK_ROWS,
            encoded_chunk_bytes=DEFAULT_ENCODED_CHUNK_BYTES,
        )
        shear = _evaluate_field(
            stream=boundary.handle,
            vtk_index=boundary_index,
            array=_required_array(boundary_index, "wallShearStressMean", 3),
            manifest=manifest,
            field_id="surface_wall_shear",
            field_name="wallShearStressMean",
            weights=surface_areas,
            region_codes=codes,
            region_ids=SURFACE_REGION_IDS,
            region_domain="surface",
            sample_ids=None,
            sample_component=None,
            sample_scale=1.0,
            area_vectors=None,
            force_sign=None,
            hash_chunk_bytes=DEFAULT_HASH_CHUNK_BYTES,
            validation_block_rows=DEFAULT_VALIDATION_BLOCK_ROWS,
            encoded_chunk_bytes=DEFAULT_ENCODED_CHUNK_BYTES,
        )

    fields = evidence.get("field_statistics")
    if not isinstance(fields, dict):
        raise RefreshError(f"{case_id} field statistics are absent")
    for field_id, result in (
        ("surface_pressure", pressure),
        ("surface_wall_shear", shear),
    ):
        if fields.get(field_id) != _field_evidence(result):
            raise RefreshError(
                f"{case_id} {field_id} replay differs from legacy evidence"
            )
    prediction = evidence.get("prediction_inputs", {}).get("surface", {})
    if (
        prediction.get("manifest_sha256") != manifest.sha256
        or prediction.get("chunk_sha256") != list(pressure.chunk_sha256)
    ):
        raise RefreshError(f"{case_id} surface prediction lineage differs")

    evidence["schema"] = EVIDENCE_SCHEMA
    evidence["schema_version"] = EVIDENCE_SCHEMA_VERSION
    evidence["support"] = {
        "case_support_sha256": support.manifest_sha256,
        "profile_definition_sha256": PROFILE_DEFINITION_SHA256,
        "regional_definition_sha256": REGION_DEFINITION_SHA256,
        "volume_region_definition_sha256": VOLUME_REGION_DEFINITION_SHA256,
    }
    evidence["report_only_regional_diagnostics"] = {
        "contract_sha256": REGION_DEFINITION_SHA256,
        "definition_id": "ahmedml-native-regions-v2-candidate",
        "ranking_effect": "none",
        "surface_pressure": dict(pressure.regional_diagnostics or {}),
        "surface_wall_shear": dict(shear.regional_diagnostics or {}),
        "volume_pressure": old_regional.get("volume_pressure"),
        "volume_velocity": old_regional.get("volume_velocity"),
    }
    _write_json(output_evidence_root / f"{case_id}.json", evidence)
    return case_id, case.surface_entity_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--input-evidence-root", type=Path, required=True)
    parser.add_argument("--output-evidence-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--source-identity",
        type=Path,
        default=(
            ROOT
            / "benchmark-specs"
            / "ahmedml"
            / "public-source-identity"
            / "ahmedml-public-source-identity-v1.json"
        ),
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must lie in [1, 16]")
    output = args.output_evidence_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output evidence directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    split = _read_json(ROOT / "benchmark-specs" / "ahmedml" / "splits" / "full.json")
    case_ids = split.get("case_ids")
    if not isinstance(case_ids, list) or len(case_ids) != 50:
        parser.error("AhmedML Full split must contain 50 cases")
    keyword = {
        "dataset_root": args.dataset_root.expanduser().resolve(),
        "support_root": args.support_root.expanduser().resolve(),
        "prediction_root": args.prediction_root.expanduser().resolve(),
        "input_evidence_root": args.input_evidence_root.expanduser().resolve(),
        "output_evidence_root": output,
        "source_identity_path": args.source_identity.expanduser().resolve(),
    }
    completed = 0
    entity_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_refresh_case, case_id, **keyword): case_id
            for case_id in case_ids
        }
        for future in as_completed(futures):
            case_id, count = future.result()
            completed += 1
            entity_count += count
            print(f"[{completed:02d}/50] {case_id}: {count:,} surface cells", flush=True)
    print(
        json.dumps(
            {
                "case_count": completed,
                "surface_entity_count": entity_count,
                "output": str(output),
                "regional_contract_sha256": REGION_DEFINITION_SHA256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
