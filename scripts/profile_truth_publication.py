"""Validate official publication metadata without changing evaluated truth bytes.

This stdlib-only contract is mirrored in the website's bin directory. The
publication binds the existing evaluator references to immutable native CFD
assets; it does not promote closed datasets or redefine any numerical support.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import unquote, urlsplit


def identity(binding: object) -> dict:
    value = binding if isinstance(binding, dict) else {}
    return {key: value.get(key) for key in ("release_id", "manifest_sha256")}


def binding_errors(binding: object) -> list[str]:
    if not isinstance(binding, dict):
        return ["profile truth publication must define a manifest binding"]
    identifier = binding.get("release_id", "")
    digest = binding.get("manifest_sha256", "")
    url = binding.get("manifest_url", "")
    if not isinstance(url, str):
        return ["profile truth publication requires a clean HTTPS URL"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,158}[a-z0-9])?", identifier):
        return ["profile truth publication requires a safe release ID"]
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        return ["profile truth publication requires a lowercase SHA-256 digest"]
    try:
        parts = urlsplit(url)
        parts.port
        valid = (
            parts.scheme == "https" and parts.hostname and not parts.username
            and not parts.password and not parts.query and not parts.fragment
            and identifier in unquote(parts.path).split("/")
            and url == url.strip() and not any(ord(c) < 33 or ord(c) == 127 for c in url)
        )
    except (TypeError, ValueError):
        valid = False
    return [] if valid else ["profile truth publication requires a clean HTTPS URL containing its release ID"]


def publication_errors(payload: bytes, binding: dict, manifest: dict, *, catalog: dict | None = None,
                       dataset_truth_bindings: dict | None = None) -> list[str]:
    errors = binding_errors(binding)
    if errors:
        return errors
    if hashlib.sha256(payload).hexdigest() != binding["manifest_sha256"]:
        return ["profile truth publication manifest SHA-256 differs from its binding"]
    try:
        publication = json.loads(payload)
    except (ValueError, UnicodeError):
        return ["profile truth publication is not valid JSON"]
    if not isinstance(publication, dict):
        return ["profile truth publication must be a JSON object"]
    release = publication.get("data_release", {})
    if not isinstance(release, dict):
        return ["profile truth publication has no release metadata"]
    source = release.get("source_commit", "")
    if (
        publication.get("schema_version") != "1.0" or release.get("id") != binding["release_id"]
        or release.get("status") != "official" or not isinstance(source, str)
        or not re.fullmatch(r"[a-f0-9]{40}", source) or release.get("source_ref") != source
        or release.get("source_repository") != "https://github.com/FluidsBench/fluidsbench"
    ):
        return ["profile truth publication must be official and pinned to an immutable website source commit"]
    source_base = f"https://raw.githubusercontent.com/FluidsBench/fluidsbench/{source}/"
    declared_catalog = publication.get("catalog", {})
    if not isinstance(declared_catalog, dict):
        return ["profile truth publication catalog must be an object"]
    expected_catalog = manifest.get("data_release", {}).get("profile_ground_truth", {})
    if (
        identity(declared_catalog) != identity(expected_catalog)
        or declared_catalog.get("manifest_url") != source_base + "assets/data/profile-ground-truth/manifest.json"
        or declared_catalog.get("asset_base_url") != source_base + "assets/data/profile-ground-truth/"
    ):
        errors.append("profile truth publication catalog differs from the evaluated reference binding")
    datasets = publication.get("datasets")
    if not isinstance(datasets, list) or any(not isinstance(d, dict) for d in datasets):
        return errors + ["profile truth publication datasets must be objects"]
    expected = {d["slug"]: d for d in manifest.get("datasets", [])
                if d.get("scoring_support", {}).get("status") == "official"}
    identifiers = [d.get("id") for d in datasets]
    if any(not isinstance(slug, str) for slug in identifiers):
        return errors + ["profile truth publication dataset IDs must be strings"]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected):
        errors.append("profile truth publication must cover each official dataset exactly once and exclude unapproved datasets")
    for dataset in datasets:
        slug = dataset.get("id")
        if slug not in expected:
            continue
        support = expected[slug]["scoring_support"]
        if dataset.get("status") != "official" or support.get("status") != "official" or identity(dataset.get("scoring_support")) != identity(support):
            errors.append(f"{slug}: profile truth publication scoring support differs")
        browser = dataset.get("browser_profile_truth", {})
        submitted = dataset.get("submission_profile_ground_truth", {})
        if not isinstance(browser, dict) or not isinstance(submitted, dict):
            errors.append(f"{slug}: profile truth publication bindings must be objects")
            continue
        target = (dataset_truth_bindings or {}).get(slug, expected_catalog)
        if slug == "airfrans" and slug not in (dataset_truth_bindings or {}):
            target = browser.get("source_release", {})
        if not submitted.get("release_id") or not re.fullmatch(r"[a-f0-9]{64}", str(submitted.get("manifest_sha256", ""))) or identity(submitted) != identity(target):
            errors.append(f"{slug}: profile truth publication submission binding differs")
        if browser.get("kind") == "catalog_dataset":
            declared = browser.get("dataset", {})
            if not isinstance(declared, dict) or not isinstance(declared.get("native_profile_truth"), dict):
                errors.append(f"{slug}: profile truth publication requires native CFD references")
                continue
            native = declared.get("native_profile_truth", {})
            if declared.get("id") != slug or native.get("source_kind") != "native_cfd" or native.get("analytical_dummy") is not False:
                errors.append(f"{slug}: profile truth publication requires native CFD references")
            if catalog is not None and declared != next((d for d in catalog.get("datasets", []) if d.get("id") == slug), None):
                errors.append(f"{slug}: published native CFD catalog entry differs from the pinned catalog")
        elif browser.get("kind") == "native_manifest" and slug in {"airfrans", "drivaernetplusplus"}:
            if (browser.get("source_kind") != "native_cfd" or browser.get("analytical_dummy") is not False
                or browser.get("manifest_url") != source_base + f"assets/data/{slug}-native-profiles/manifest.json"
                or not re.fullmatch(r"[a-f0-9]{64}", str(browser.get("manifest_sha256", "")))):
                errors.append(f"{slug}: published browser reference must bind an immutable native CFD manifest")
        else:
            errors.append(f"{slug}: unsupported profile truth publication source")
    return errors
