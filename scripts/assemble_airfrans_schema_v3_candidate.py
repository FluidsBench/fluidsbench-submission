#!/usr/bin/env python3
"""Assemble a schema-v3 AirfRANS candidate submission package (Layer 4).

Mirrors scripts/assemble_hiliftaeroml_schema_v3_candidate.py's config ->
list-blockers -> assemble workflow, adapted to what Layers 1-3 already
produced for AirfRANS:

* Layer 1 (scripts/build_airfrans_scoring_support.py): the ground-truth
  scoring-support release (manifest/index/chunks/tables).
* Layer 2 (scripts/derive_airfrans_predicted_fields.py): per-case predicted
  fields, packaged as a "kind": "scored_predictions" manifest.
* Layer 3 (scripts/finalize_airfrans_metrics.py): metrics/cases.json, already
  schema-v3 compliant, with the profile R2 and composite scores folded in.

This script fills the remaining schema-v3 package: submission.json,
evaluation-evidence.json, discretization.json + discretization/cases.jsonl,
and profiles/index.json + chunks (from the already-extracted boundary-layer
profile predictions). It never writes maintainer-only files (approval,
maintainer-validation.json, prediction-artifact-checks.json).

Every fact only the submitter can supply (architecture, training regime,
checkpoint hashes, measured compute, training-time spatial representation)
comes from a package-config.json copied from
examples/airfrans-v3-candidate/package-config.template.json. Any remaining
``__REPLACE_WITH_...__``/``__UNRESOLVED_...__`` token is a blocker:

    python scripts/assemble_airfrans_schema_v3_candidate.py --config PATH --list-blockers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scoring_support import sha256_file  # noqa: E402

TOKEN_PREFIXES = ("__REPLACE_", "__UNRESOLVED_", "__RESOLVED_LOCALLY__", "__RESOLVED_AT_ASSEMBLY_TIME__")
LITERAL_TOKEN_PREFIXES = ("__REPLACE_", "__UNRESOLVED_")

DEFAULT_SPEC = ROOT / "benchmark-specs" / "airfrans" / "submission-spec.json"
SCHEMA_ROOT = ROOT / "schemas"
DATASET_ID = "airfrans"


class AssembleError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def find_blockers(node: Any, path: str = "config") -> list[str]:
    blockers: list[str] = []
    if isinstance(node, str):
        if node.startswith(LITERAL_TOKEN_PREFIXES):
            blockers.append(f"{path} = {node!r}")
    elif isinstance(node, dict):
        for key, value in node.items():
            blockers.extend(find_blockers(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            blockers.extend(find_blockers(value, f"{path}[{index}]"))
    return blockers


def validate_against_schema(schema_relative_path: str, document: Any, label: str) -> None:
    schema = load_json(SCHEMA_ROOT / schema_relative_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    if errors:
        details = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:5])
        raise AssembleError(f"{label} failed schema validation ({len(errors)} errors): {details}")


# --------------------------------------------------------------------------
# discretization.json / discretization/cases.jsonl
# --------------------------------------------------------------------------


def build_discretization(
    config: dict,
    submission_id: str,
    split_id: str,
    scoring_support_release_id: str,
    scoring_support_manifest_sha256: str,
    case_records: list[dict],
) -> tuple[dict, list[dict]]:
    training_config = config["spatial_discretization"]["training"]
    inference_config = config["spatial_discretization"]["inference"]

    def per_case_count(entity: str, values: list[int]) -> dict:
        return {
            "entity": entity,
            "count": {
                "kind": "per_case",
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
            },
        }

    def per_case_count_bare(values: list[int]) -> dict:
        return {
            "kind": "per_case",
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }

    domain_counts = [c["n_domain"] for c in case_records]
    curve_counts = [c["n_curve"] for c in case_records]

    def native_representation(entity: str, values: list[int], representation: str, case_record_id: str) -> dict:
        return {
            "used": True,
            "case_record_id": case_record_id,
            "representation": representation,
            "entity_counts": [per_case_count(entity, values)],
            "native_comparison": {"status": "not_applicable"},
            "sampling": {"kind": "none"},
            "domain": {"kind": "full_dataset_domain"},
            "connectivity": "native",
        }

    discretization = {
        "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json",
        "schema_version": "1.0",
        "submission_id": submission_id,
        "dataset_id": DATASET_ID,
        "split_id": split_id,
        "scoring_support_release_id": scoring_support_release_id,
        "scoring_support_manifest_sha256": scoring_support_manifest_sha256,
        "training": {
            "surface_input": training_config["curve_input"],
            "surface_supervision": training_config["curve_supervision"],
            "volume_input": training_config["domain_input"],
            "volume_supervision": training_config["domain_supervision"],
        },
        "inference": {
            "geometry_dependency": inference_config["geometry_dependency"],
            "surface_input": native_representation(
                "airfoil_curve_points", curve_counts, "native_airfoil_curve_points", "native-curve-points"
            ),
            "volume_input": native_representation(
                "domain_points", domain_counts, "native_2d_domain_points", "native-domain-points"
            ),
            "direct_outputs": [
                {
                    "id": "native-domain-velocity-pressure",
                    "domain": "volume",
                    "representation": native_representation(
                        "domain_points", domain_counts, "native_2d_domain_points", "native-domain-points"
                    ),
                    "queries_per_forward_pass": per_case_count_bare(domain_counts),
                }
            ],
            # Only the two spatial supports Neil's submission-spec.json declares under
            # public_supports get a discretization mapping. Force/coefficients are a
            # derived scalar quantity (re-integrated by the evaluator from the domain
            # prediction, like HiLift's forces), not a separate declared support, so it
            # is scored via metrics/cases.json but intentionally has no mapping entry
            # here.
            "mappings": [
                {
                    "support_id": "two-dimensional-domain-native",
                    "source_output_id": "native-domain-velocity-pressure",
                    "method": {"kind": "identity"},
                    "implementation": "scripts/derive_airfrans_predicted_fields.py",
                    "extrapolation_policy": "forbidden",
                    "unmapped_fraction": 0.0,
                    "extrapolated_fraction": 0.0,
                    "final_coverage_fraction": 1.0,
                },
                {
                    "support_id": "airfoil-curve-native",
                    "source_output_id": "native-domain-velocity-pressure",
                    "method": {
                        "kind": "reference_rule",
                        "rule_id": "airfrans-surface-subset-reorganize-v1",
                        "rule_version": "1",
                    },
                    "implementation": "scripts/derive_airfrans_predicted_fields.py",
                    "extrapolation_policy": "forbidden",
                    "unmapped_fraction": 0.0,
                    "extrapolated_fraction": 0.0,
                    "final_coverage_fraction": 1.0,
                },
            ],
        },
        "case_manifest": {
            "format": "jsonl",
            "file": "discretization/cases.jsonl",
            "sha256": None,  # filled in by the caller once the jsonl file is written
            "case_count": len(case_records),
        },
    }

    case_rows = []
    for record in case_records:
        case_rows.append(
            {
                "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json",
                "schema_version": "1.0",
                "submission_id": submission_id,
                "dataset_id": DATASET_ID,
                "split_id": split_id,
                "case_id": record["case_id"],
                "inference": {
                    "inputs": [
                        {
                            "id": "native-domain-points",
                            "entity_counts": [{"entity": "domain_points", "count": record["n_domain"]}],
                        },
                        {
                            "id": "native-curve-points",
                            "entity_counts": [{"entity": "airfoil_curve_points", "count": record["n_curve"]}],
                        },
                    ],
                    "direct_outputs": [
                        {
                            "id": "native-domain-velocity-pressure",
                            "entity_counts": [{"entity": "domain_points", "count": record["n_domain"]}],
                        }
                    ],
                    "mappings": [
                        {
                            "support_id": "two-dimensional-domain-native",
                            "source_output_id": "native-domain-velocity-pressure",
                            "support_count": record["n_domain"],
                            "scored_count": record["n_domain"],
                            "unmapped_count": 0,
                            "extrapolated_count": 0,
                            "final_coverage_fraction": 1.0,
                        },
                        {
                            "support_id": "airfoil-curve-native",
                            "source_output_id": "native-domain-velocity-pressure",
                            "support_count": record["n_curve"],
                            "scored_count": record["n_curve"],
                            "unmapped_count": 0,
                            "extrapolated_count": 0,
                            "final_coverage_fraction": 1.0,
                        },
                    ],
                },
            }
        )
    return discretization, case_rows


# --------------------------------------------------------------------------
# profiles/index.json + chunks
# --------------------------------------------------------------------------


def build_profiles(
    predicted_profiles_dir: Path,
    case_ids: list[str],
    output_dir: Path,
    submission_id: str,
    split_id: str,
    case_set_id: str,
) -> tuple[dict, int]:
    chunks_meta = []
    for chunk_index, case_id in enumerate(case_ids):
        source_path = predicted_profiles_dir / f"{case_id}.json"
        if not source_path.is_file():
            raise AssembleError(f"missing predicted profile for {case_id}: {source_path}")
        payload = load_json(source_path)
        cases = payload.get("cases", [])
        if len(cases) != 1 or cases[0].get("case_id") != case_id:
            raise AssembleError(f"{source_path} does not contain exactly one series block for {case_id}")
        chunk_filename = f"chunk-{chunk_index:03d}.json"
        chunk_path = output_dir / "profiles" / chunk_filename
        chunk_sha256 = write_json(chunk_path, {"schema_version": "1.0", "cases": cases})
        chunks_meta.append(
            {
                "file": chunk_filename,
                "case_ids": [case_id],
                "sha256": chunk_sha256,
            }
        )
    index = {
        "schema_version": "1.0",
        "format": "fluidsbench-profile-chunks-v1",
        "submission_id": submission_id,
        "dataset_id": DATASET_ID,
        "split_id": split_id,
        "case_set_id": case_set_id,
        "case_count": len(chunks_meta),
        "case_id_status": "official",
        "chunks": chunks_meta,
    }
    write_json(output_dir / "profiles" / "index.json", index)
    return index, len(chunks_meta)


# --------------------------------------------------------------------------
# main assembly
# --------------------------------------------------------------------------


def assemble(
    *,
    config: dict,
    scoring_support_manifest_path: Path,
    metrics_cases_path: Path,
    predicted_profiles_dir: Path,
    profile_score_path: Path,
    output_dir: Path,
    spec_path: Path,
) -> None:
    if output_dir.exists():
        raise AssembleError(f"output directory already exists, refusing to overwrite: {output_dir}")

    spec = load_json(spec_path)
    support_manifest = load_json(scoring_support_manifest_path)
    support_manifest_sha256 = sha256_file(scoring_support_manifest_path)
    metrics_cases = load_json(metrics_cases_path)

    submission_id = config["submission_id"]
    split_id = config["split_id"]
    case_set_id = config["case_set_id"]

    case_records = []
    for case in metrics_cases["cases"]:
        supports_by_id = {support["support_id"]: support for support in case["supports"]}
        case_records.append(
            {
                "case_id": case["case_id"],
                "n_domain": supports_by_id["two-dimensional-domain-native"]["support_count"],
                "n_curve": supports_by_id["airfoil-curve-native"]["support_count"],
            }
        )
    case_ids = [record["case_id"] for record in case_records]

    output_dir.mkdir(parents=True)

    # metrics/cases.json: already produced by Layer 3, copy verbatim.
    metrics_sha256 = write_json(output_dir / "metrics" / "cases.json", metrics_cases)

    # discretization.json + discretization/cases.jsonl
    discretization, case_rows = build_discretization(
        config, submission_id, split_id, support_manifest["release_id"], support_manifest_sha256, case_records
    )
    case_manifest_sha256 = write_jsonl(output_dir / "discretization" / "cases.jsonl", case_rows)
    discretization["case_manifest"]["sha256"] = case_manifest_sha256
    discretization_sha256 = write_json(output_dir / "discretization.json", discretization)
    validate_against_schema("v3/discretization.schema.json", discretization, "discretization.json")
    for row in case_rows:
        validate_against_schema("v3/discretization-case.schema.json", row, f"discretization row {row['case_id']}")

    # profiles/index.json + chunks
    profile_index, profile_case_count = build_profiles(
        predicted_profiles_dir, case_ids, output_dir, submission_id, split_id, case_set_id
    )
    profile_index_sha256 = sha256_file(output_dir / "profiles" / "index.json")

    profile_score = load_json(profile_score_path)
    profile_definition_sha256 = spec["profile_definition"]["sha256"]
    profile_definition_id = spec["profile_definition"]["id"]

    # evaluation-evidence.json
    evaluation_evidence = {
        "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json",
        "schema_version": "3.0",
        "submission_id": submission_id,
        "dataset_id": DATASET_ID,
        "dataset_version": spec["dataset_version"],
        "split_id": split_id,
        "split_sha256": next(s["sha256"] for s in spec["splits"] if s["id"] == split_id),
        "case_set_id": case_set_id,
        "reference_version": spec["evaluation_reference_version"],
        "command": (
            "python3 -m reference.evaluate_predictions "
            f"--support-manifest {scoring_support_manifest_path} --case-set {case_set_id} "
            f"--prediction-manifest <layer2-output>/manifest.json --submission-id {submission_id} "
            f"--split-id {split_id} --output metrics/cases.json ; "
            "python3 scripts/finalize_airfrans_metrics.py (folds in velocity_profile_r2 + composite scores)"
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "submitted_evaluation",
        "metric_values": metrics_cases["metric_values"],
        "profile_index_sha256": profile_index_sha256,
        "profile_ground_truth_release_id": profile_definition_id,
        "profile_ground_truth_manifest_sha256": profile_definition_sha256,
        "scoring_support_release_id": support_manifest["release_id"],
        "scoring_support_manifest_sha256": support_manifest_sha256,
        "discretization_sha256": discretization_sha256,
        "case_metrics_sha256": metrics_sha256,
        "notes": (
            "AirfRANS remains closed (submission-spec.json scoring_support.status="
            "owner_review_required, submissions_open=false); this is a candidate "
            "package, not an accepted result."
        ),
    }
    evaluation_evidence_sha256 = write_json(output_dir / "evaluation-evidence.json", evaluation_evidence)
    validate_against_schema("v3/evaluation-evidence.schema.json", evaluation_evidence, "evaluation-evidence.json")

    # submission.json
    participant = config["participant"]
    reproducibility_config = dict(config["reproducibility"])
    for optional_key in ("model_artifact", "environment"):
        value = reproducibility_config.get(optional_key)
        if isinstance(value, str) and value.startswith(LITERAL_TOKEN_PREFIXES):
            reproducibility_config.pop(optional_key, None)

    methodology = dict(participant["methodology"])
    methodology["format"] = "fluidsbench-method-v1"

    total_params = methodology["architecture"]["total_parameter_count"]
    submission = {
        "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json",
        "schema_version": "3.0",
        "submission_id": submission_id,
        "result_revision": config["result_revision"],
        "model": participant["model"],
        "model_type": participant["model_type"],
        "model_types": participant["model_types"],
        "training_regime": participant["training_regime"],
        "target_data_used": participant["target_data_used"],
        "external_pretraining": participant["external_pretraining"],
        "pretraining_data": participant.get("pretraining_data", []),
        "dataset": spec["dataset_name"],
        "dataset_id": DATASET_ID,
        "dataset_version": spec["dataset_version"],
        "split": next(s["label"] for s in spec["splits"] if s["id"] == split_id),
        "split_id": split_id,
        "case_set_id": case_set_id,
        "split_sha256": evaluation_evidence["split_sha256"],
        "parameter_count_millions": round(total_params / 1_000_000, 6),
        "methodology": methodology,
        "submitter_name": participant["submitter_name"],
        "institution": participant["institution"],
        "paper_url": participant["paper_url"],
        "submitted_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "evaluation": {
            "reference_version": spec["evaluation_reference_version"],
            "command": evaluation_evidence["command"],
            "evidence_file": "evaluation-evidence.json",
            "evidence_sha256": evaluation_evidence_sha256,
        },
        "reproducibility": reproducibility_config,
        "scoring_support": {
            "status": "candidate",
            "release_id": support_manifest["release_id"],
            "manifest_url": config["release_bindings"]["scoring_support"]["manifest_url"],
            "manifest_sha256": support_manifest_sha256,
        },
        "spatial_discretization": {
            "format": "fluidsbench-discretization-v1",
            "file": "discretization.json",
            "sha256": discretization_sha256,
        },
        "case_metrics": {
            "format": "fluidsbench-case-metrics-v1",
            "file": "metrics/cases.json",
            "sha256": metrics_sha256,
            "case_count": metrics_cases["case_count"],
        },
        "metric_values": metrics_cases["metric_values"],
        "profile_data": {
            "format": "fluidsbench-profile-chunks-v1",
            "index_file": "profiles/index.json",
            "case_count": profile_case_count,
            "case_set_id": case_set_id,
            "profile_ground_truth_release_id": profile_definition_id,
            "profile_ground_truth_manifest_sha256": profile_definition_sha256,
        },
        "note": (
            "Candidate AirfRANS schema-v3 package. AirfRANS submissions remain "
            "closed (owner_review_required) pending the owner_decisions_required "
            "checklist in benchmark-specs/airfrans/submission-spec.json. "
            "scoring_support.manifest_url is a placeholder (not resolvable): no "
            "AirfRANS scoring-support release has been published anywhere yet; "
            "replace it once/if the dataset owner publishes one."
        ),
    }
    validate_against_schema("v3/submission.schema.json", submission, "submission.json")
    write_json(output_dir / "submission.json", submission)

    print(f"Assembled candidate package at {output_dir}")
    print(f"  submission_id       = {submission_id}")
    print(f"  case_count          = {metrics_cases['case_count']}")
    print(f"  overall_score       = {metrics_cases['metric_values'].get('overall_score')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scoring-support-manifest", type=Path)
    parser.add_argument("--metrics-cases", type=Path)
    parser.add_argument("--predicted-profiles-dir", type=Path)
    parser.add_argument("--profile-score", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-blockers", action="store_true")
    args = parser.parse_args(argv)

    config = load_json(args.config)
    blockers = find_blockers(config)
    if args.list_blockers:
        if blockers:
            print(f"{len(blockers)} blocker(s) in {args.config}:")
            for blocker in blockers:
                print(f"  - {blocker}")
        else:
            print(f"No blockers found in {args.config}.")
        return 1 if blockers else 0

    if blockers:
        parser.error(
            f"{len(blockers)} unresolved config token(s); rerun with --list-blockers to see them"
        )
    for required in (
        "scoring_support_manifest",
        "metrics_cases",
        "predicted_profiles_dir",
        "profile_score",
        "output",
    ):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required for assembly")

    try:
        assemble(
            config=config,
            scoring_support_manifest_path=args.scoring_support_manifest,
            metrics_cases_path=args.metrics_cases,
            predicted_profiles_dir=args.predicted_profiles_dir,
            profile_score_path=args.profile_score,
            output_dir=args.output,
            spec_path=args.spec,
        )
    except AssembleError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
