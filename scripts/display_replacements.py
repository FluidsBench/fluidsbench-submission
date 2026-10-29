"""Maintainer-owned dev display replacements across dataset contracts.

These links never change scientific version history or source packages.
Only a registered prototype's identical retained inference may be replaced.
"""

import hashlib
import json
import re
from copy import deepcopy


REGISTRY = "benchmark-specs/dev-display-replacements.json"


def apply_display_replacements(root, manifest, rows_by_dataset):
    release = manifest.get("data_release", {})
    if release.get("status") != "prototype_dummy_data" or release.get("source_ref") != "dev":
        return
    path = root / REGISTRY
    if not path.exists():
        return
    registry = json.loads(path.read_text())
    if set(registry) != {"schema_version", "replacements"} or registry["schema_version"] != "1.0":
        raise ValueError("invalid dev display replacement registry")
    if not isinstance(registry["replacements"], list):
        raise ValueError("display replacements must be a list")
    rows = {r["submission_id"]: r for group in rows_by_dataset.values() for r in group}
    sources, targets = set(), set()
    pending = []
    for item in registry["replacements"]:
        if not isinstance(item, dict) or set(item) != {
            "source_submission_id", "source_sha256", "replacement_submission_id",
            "replacement_sha256", "reason", "review_url",
        }:
            raise ValueError("invalid dev display replacement fields")
        source_id, target_id = item["source_submission_id"], item["replacement_submission_id"]
        if source_id == target_id or source_id in sources or target_id in targets:
            raise ValueError("duplicate or self-referencing display replacement")
        sources.add(source_id)
        targets.add(target_id)
        if sources & targets:
            raise ValueError("display replacement chains and cycles are not supported")
        if source_id not in rows or target_id not in rows:
            raise ValueError("display replacement requires both published source and approved target")
        source, target = rows[source_id], rows[target_id]
        if source.get("record_type") != "pre_release_reference" or target.get("approval", {}).get("status") != "approved":
            raise ValueError("display replacement requires a registered reference and approved target")
        if source.get("dataset_version") == target.get("dataset_version"):
            raise ValueError("same-contract updates must use result_revision")
        for field in ("dataset_id", "split_id", "split_sha256", "case_set_id", "model", "methodology", "reproducibility", "metric_values"):
            if source.get(field) != target.get(field):
                raise ValueError(f"retained-inference display replacement must preserve {field}")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("display replacement requires a reason")
        if not isinstance(item["review_url"], str) or not re.fullmatch(
            r"https://github\.com/[^/]+/[^/]+/pull/[1-9][0-9]*", item["review_url"]
        ):
            raise ValueError("display replacement requires a review PR URL")
        bindings = {}
        for label, row, digest in (("source", source, item["source_sha256"]), ("replacement", target, item["replacement_sha256"])):
            relative = f"submissions/{row['dataset_id']}/{row['submission_id']}/submission.json"
            file = root / relative
            if not file.resolve().is_relative_to((root / "submissions").resolve()):
                raise ValueError("display replacement path escapes submissions")
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError(f"display replacement {label} package hash mismatch")
            bindings[label] = {"path": relative, "sha256": digest}
        for section, field, content in (("profile_data", "index_file", "chunks"), ("case_metrics", "file", "cases")):
            before = json.loads((root / source[section][field]).read_text())
            after = json.loads((root / target[section][field]).read_text())
            if before[content] != after[content]:
                raise ValueError(f"retained-inference display replacement must preserve {section}")
        pending.append((source, target, {**deepcopy(item), "bindings": bindings}))
    # Apply only after the entire registry passes, so failures cannot hide rows.
    for source, target, metadata in pending:
        source["display_replacement"] = {**metadata, "replaced_by": target["submission_id"]}
        target_series = target["result_revision"]["series_id"]
        for row in rows.values():
            if (row["dataset_id"], row["split_id"], row["result_revision"]["series_id"]) == (
                target["dataset_id"], target["split_id"], target_series
            ):
                row["display_replacement"] = {**deepcopy(metadata), "replaces": source["submission_id"]}
