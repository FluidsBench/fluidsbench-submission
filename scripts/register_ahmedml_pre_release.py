#!/usr/bin/env python3
"""Create an exact dev-only binding for a genuine AhmedML candidate package.

By default the command prints the proposed registry entry and changes nothing.
With ``--register`` it atomically appends that exact entry to the checked-in
closed-candidate registry.  Registration permits a validated result to appear
only in the prototype dev feed; it never creates approval, opens submissions,
or makes a result citable or promotable.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reference.ahmedml.pre_release import (  # noqa: E402
    AhmedMLPreReleaseError,
    REGISTRY_SCHEMA,
    REGISTRY_STATUS,
    build_registry_entry,
)
from scripts.validate_submission import (  # noqa: E402
    load_submission_json,
    validate_submission_file,
)


DEFAULT_REGISTRY = (
    ROOT / "benchmark-specs" / "ahmedml" / "pre-release-reference-registry.json"
)


def _registry(path: Path) -> dict[str, Any]:
    try:
        value = load_submission_json(path)
    except (OSError, UnicodeError, ValueError) as error:
        raise AhmedMLPreReleaseError(f"cannot read registry {path}: {error}") from error
    expected_activation = {
        "owner_approval_complete": False,
        "published": False,
        "submissions_opened": False,
        "registration_changes_activation": False,
    }
    if (
        not isinstance(value, dict)
        or value.get("schema") != REGISTRY_SCHEMA
        or value.get("status") != REGISTRY_STATUS
        or value.get("dataset_id") != "ahmedml"
        or value.get("activation") != expected_activation
        or not isinstance(value.get("entries"), list)
    ):
        raise AhmedMLPreReleaseError(
            "registry is not the closed AhmedML dev-only lifecycle record"
        )
    entries = value["entries"]
    if not all(isinstance(entry, dict) for entry in entries):
        raise AhmedMLPreReleaseError("registry entries must be objects")
    identifiers = [entry.get("submission_id") for entry in entries]
    paths = [entry.get("submission_path") for entry in entries]
    scopes = [entry.get("scope") for entry in entries]
    if (
        any(not isinstance(item, str) or not item for item in identifiers + paths)
        or len(identifiers) != len(set(identifiers))
        or len(paths) != len(set(paths))
        or any(
            not isinstance(scope, dict)
            or not isinstance(scope.get("split_id"), str)
            or not scope["split_id"]
            for scope in scopes
        )
    ):
        raise AhmedMLPreReleaseError(
            "registry entries must have unique non-empty IDs and paths plus "
            "a non-empty scope.split_id"
        )
    return value


def _write_atomic(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def register(
    submission_path: Path,
    *,
    registry_path: Path,
    write: bool,
) -> dict[str, Any]:
    submission_path = submission_path.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    if write and registry_path != DEFAULT_REGISTRY.resolve():
        raise AhmedMLPreReleaseError(
            "registration may write only the repository's AhmedML registry"
        )
    errors, statistics = validate_submission_file(
        submission_path,
        candidate_dry_run=True,
    )
    if errors:
        raise AhmedMLPreReleaseError(
            "candidate dry-run validation failed: " + " | ".join(errors)
        )
    entry = build_registry_entry(submission_path, repository_root=ROOT)
    if statistics != {
        "cases": entry["scope"]["case_count"],
        "series": entry["scope"]["profile_series_count"],
    }:
        raise AhmedMLPreReleaseError(
            "candidate validator coverage differs from the proposed binding"
        )
    registry = _registry(registry_path)
    entries = registry["entries"]
    conflicts = [
        existing
        for existing in entries
        if isinstance(existing, dict)
        and (
            existing.get("submission_id") == entry["submission_id"]
            or existing.get("submission_path") == entry["submission_path"]
        )
    ]
    if conflicts:
        if len(conflicts) == 1 and conflicts[0] == entry:
            return {
                "status": "already_registered_exactly",
                "entry": entry,
                "registry": str(registry_path),
                "written": False,
            }
        raise AhmedMLPreReleaseError(
            "registry already contains a different binding for this ID or path"
        )
    if write:
        entries.append(entry)
        entries.sort(key=lambda item: (item["scope"]["split_id"], item["submission_id"]))
        _write_atomic(registry_path, registry)
    return {
        "status": "registered" if write else "proposed_not_written",
        "entry": entry,
        "registry": str(registry_path),
        "written": write,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_json", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--register",
        action="store_true",
        help="append the exact entry atomically after candidate validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = register(
            args.submission_json,
            registry_path=args.registry,
            write=args.register,
        )
    except (AhmedMLPreReleaseError, OSError, ValueError) as error:
        print(
            json.dumps({"status": "blocked", "error": str(error)}, sort_keys=True),
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
