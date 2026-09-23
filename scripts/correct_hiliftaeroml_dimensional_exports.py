#!/usr/bin/env python3
"""Version the SI export of the 23 pinned HiLift previews, preserving old receipts.

Without --apply, validate an existing correction or preflight the native packages.
This changes neither predictions nor benchmark activation. The derived registry
must be reviewed and its digest pinned in reference/hiliftaeroml/dimensional_units.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.dimensional_units import (  # noqa: E402
    CORRECTION_PATH,
    corrected_documents,
    corrected_registration_view,
    digest,
    export_binding,
    validate_contract,
)
from scripts.build_hiliftaeroml_submission_zip import build_deterministic_submission_zip  # noqa: E402
from scripts.validate_submission import HILIFT_REGISTERED_PREVIEW_CONFIGS  # noqa: E402

SOURCE_REVISION = "cb64dacb296e467bc706de19093142c66b4056da"
SOURCE_FEED_SHA256 = "7702b02f01408b26e16d0d687ade3cebce4c4267a395dd803fa3f0d645dea572"
SOURCE_DIRECTORY = CORRECTION_PATH.with_suffix("") / "sources"


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def byte_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def archive_identity(directory: Path, output: Path) -> dict:
    receipt = build_deterministic_submission_zip(directory, output)
    return {"sha256": receipt["archive_sha256"], "size_bytes": output.stat().st_size,
            "member_count": receipt["member_count"]}


def correct(root: Path, *, apply: bool = False) -> dict:
    validate_contract(root)
    registry_path = root / CORRECTION_PATH
    if registry_path.exists():
        # Idempotent: check the recorded bytes, never multiply an SI value again.
        for configuration in HILIFT_REGISTERED_PREVIEW_CONFIGS:
            binding = configuration["binding"]
            path = root / binding["submission_path"]
            corrected_registration_view(root, path, json.loads(path.read_text()), binding)
        return {"status": "already_corrected", "records": len(HILIFT_REGISTERED_PREVIEW_CONFIGS),
                "registry_sha256": digest(registry_path)}
    if digest(root / "leaderboard/all.json") != SOURCE_FEED_SHA256:
        raise ValueError("native feed differs from the audited source release")
    if (root / SOURCE_DIRECTORY).exists():
        raise ValueError("original-source directory already exists without its correction registry")

    registry = {
        "schema": "hiliftaeroml-dimensional-export-correction-v1",
        "export_binding": export_binding(),
        "source_revision": SOURCE_REVISION,
        "source_feed_sha256": SOURCE_FEED_SHA256,
        "activation_effect": "none",
        "preservation": "Same predictions and result IDs; original validation receipts are unchanged. Only eight dimensional field MAE/RMSE metrics and their artifact hashes are corrected. Relative statistics, scores, profiles, regional diagnostics, supports and masks are unchanged.",
        "records": {},
    }
    writes: dict[Path, bytes] = {}
    case_count = 0
    with tempfile.TemporaryDirectory(prefix="hilift-si-correction-") as temporary:
        workspace = Path(temporary)
        for configuration in HILIFT_REGISTERED_PREVIEW_CONFIGS:
            binding = configuration["binding"]
            path = root / binding["submission_path"]
            evidence_path = path.parent / "evaluation-evidence.json"
            cases_path = path.parent / "metrics/cases.json"
            original_submission = path.read_bytes()
            original_evidence = evidence_path.read_bytes()
            submission = json.loads(original_submission)
            evidence = json.loads(original_evidence)
            cases = json.loads(cases_path.read_bytes())
            if byte_digest(original_submission) != binding["submission_json_sha256"]:
                raise ValueError(f"{path}: original submission hash differs")
            if byte_digest(original_evidence) != submission["evaluation"]["evidence_sha256"]:
                raise ValueError(f"{path}: original evidence hash differs")
            if digest(cases_path) != submission["case_metrics"]["sha256"] or digest(cases_path) != evidence["case_metrics_sha256"]:
                raise ValueError(f"{path}: original case hash differs")
            if len(cases["cases"]) != configuration["case_count"]:
                raise ValueError(f"{path}: original case count differs")
            archive = workspace / "package.zip"
            if archive_identity(path.parent, archive) != binding["deterministic_archive"]:
                raise ValueError(f"{path}: original whole-package archive identity differs")
            result, proof, case_document = corrected_documents(submission, evidence, cases)
            case_bytes = json_bytes(case_document)
            case_sha = byte_digest(case_bytes)
            result["case_metrics"]["sha256"] = case_sha
            proof["case_metrics_sha256"] = case_sha
            proof_bytes = json_bytes(proof)
            proof_sha = byte_digest(proof_bytes)
            result["evaluation"]["evidence_sha256"] = proof_sha
            result_bytes = json_bytes(result)

            staged = workspace / "package"
            shutil.copytree(path.parent, staged)
            for relative, content in (("submission.json", result_bytes), ("evaluation-evidence.json", proof_bytes), ("metrics/cases.json", case_bytes)):
                (staged / relative).write_bytes(content)
                writes[path.parent / relative] = content
            corrected_archive = archive_identity(staged, archive)
            shutil.rmtree(staged)

            source = SOURCE_DIRECTORY / binding["submission_id"]
            writes[root / source / "submission.json"] = original_submission
            writes[root / source / "evaluation-evidence.json"] = original_evidence
            registry["records"][binding["submission_id"]] = {
                "source": {
                    "submission_file": (source / "submission.json").as_posix(),
                    "evidence_file": (source / "evaluation-evidence.json").as_posix(),
                    "submission_json_sha256": byte_digest(original_submission),
                    "evaluation_evidence_sha256": byte_digest(original_evidence),
                    "case_metrics_sha256": digest(cases_path),
                    "deterministic_archive": binding["deterministic_archive"],
                },
                "corrected": {
                    "submission_json_sha256": byte_digest(result_bytes),
                    "evaluation_evidence_sha256": proof_sha,
                    "case_metrics_sha256": case_sha,
                    "deterministic_archive": corrected_archive,
                    "case_count": len(cases["cases"]),
                },
            }
            case_count += len(cases["cases"])

    registry_bytes = json_bytes(registry)
    if apply:
        # Every source and derived archive passed before any tracked file changes.
        # Write the registry last so interrupted runs cannot be accepted as complete.
        for path, content in writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        registry_path.write_bytes(registry_bytes)
    return {"status": "corrected" if apply else "preflight_passed", "records": len(registry["records"]),
            "cases": case_count, "registry_sha256": byte_digest(registry_bytes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(correct(ROOT, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
