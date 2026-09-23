#!/usr/bin/env python3
"""Render the pressure audit; never modify scoring specifications or results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/pressure-references.json"
OUTPUT = ROOT / "docs/PRESSURE_REFERENCES.md"
DATASETS = {"ahmedml", "airfrans", "drivaerml", "drivaernetplusplus", "hiliftaeroml", "windsorml", "rotor37", "vki-ls59"}


def load_audit() -> dict:
    audit = json.loads(SOURCE.read_text())
    if audit.get("schema_version") != 1 or set(audit["datasets"]) != DATASETS:
        raise ValueError("pressure audit must cover exactly the eight visible datasets")
    for slug, entry in audit["datasets"].items():
        if entry["status"] not in {"confirmed", "partial", "inferred", "not_applicable"}:
            raise ValueError(f"{slug}: invalid evidence status")
        if entry["status"] in {"partial", "inferred"} and not entry["follow_up"]:
            raise ValueError(f"{slug}: unresolved evidence needs a specific follow-up")
        for source in entry["sources"]:
            if not source["url"].startswith("https://") or not source["locator"]:
                raise ValueError(f"{slug}: evidence needs an HTTPS source and locator")
        for evidence in entry["code_evidence"]:
            path = ROOT / evidence["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
                raise ValueError(f"{slug}: audited input changed: {evidence['path']}; re-audit before updating its hash")
    return audit


def render(audit: dict) -> str:
    lines = [
        "# Pressure definitions and references", "",
        f"Audited {audit['audited_on']} against submission revision `{audit['audited_submission_revision']}`.", "",
        audit["scope"], "",
        "Generated from [pressure-references.json](pressure-references.json). Source confirmations describe evidence, not permission to open submissions.", "",
        "## Before evaluating", "",
    ]
    lines.extend(f"{text}\n" for text in audit["common"].values())
    lines.extend(["## Dataset inventory", "", "| Dataset | Evidence status |", "| --- | --- |"])
    for slug, entry in audit["datasets"].items():
        lines.append(f"| [{entry['name']}](#{slug}) | {entry['status_label']} |")
    for slug, entry in audit["datasets"].items():
        lines.extend(["", f'<a id="{slug}"></a>', "", f"## {entry['name']}", "", f"**{entry['status_label']}.** {entry['summary']}", ""])
        for field in entry["fields"]:
            lines.extend([
                f"### {field['domain']}", "",
                f"- **Native field / association:** `{field['raw_field']}`; {field['association']}.",
                f"- **Units:** {field['units']}.",
                f"- **Reference:** {field['reference']}",
                f"- **Current evaluation:** {field['evaluation']}",
                f"- **Conversion:** {field['conversion']}", "",
            ])
        lines.extend(f"{note}\n" for note in entry["notes"])
        if entry["follow_up"]:
            lines.extend(["**Unresolved evidence / required closure**", ""])
            lines.extend(f"- {note}" for note in entry["follow_up"])
            lines.append("")
        lines.extend(["**Sources checked**", ""])
        for source in entry["sources"]:
            lines.append(f"- [{source['label']}]({source['url']}), {source['locator']}: {source['finding']}")
        lines.extend(["", "<details>", "<summary>Evaluator and specification trace</summary>", ""])
        for evidence in entry["code_evidence"]:
            lines.append(f"- [{evidence['path']}](../{evidence['path']}): {evidence['finding']}")
        lines.extend(["", "SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.", "", "</details>"])
    lines.extend([
        "", "## Audit limits and next steps", "",
        "The WindsorML header inspection, Rotor37 sample calculation and HiLiftAeroML exported-metric replay are recorded in [pressure-reference-observations.json](pressure-reference-observations.json). These checks do not rerun full-dataset native inference.", "",
        "Resolve WindsorML’s volume export definition and obtain Rotor37’s explicit owner declaration. HiLiftAeroML’s pressure convention is confirmed and its dimensional export is corrected through `hiliftaeroml-dimensional-export-si-v1`; see [the correction record](../benchmark-specs/hiliftaeroml/DIMENSIONAL_EXPORT_CORRECTION.md). No author messages were sent by this audit.", "",
        "A changed offset or unit conversion needs source evidence, an explicit evaluator or export version, recomputation of affected metrics and a distinct result release. Preserve original artifacts and receipts. The HiLiftAeroML correction follows this process without changing the pressure offset, weights, split, validity mask, threshold or score. The other dataset findings remain documentation only.", "",
        "To check the audit inputs and regenerate this document:", "",
        "```bash", "python3 scripts/render_pressure_reference_docs.py --check", "```", "",
        "Omit `--check` to render. Add `--website-root /path/to/fluidsbench` to synchronize or check the website’s exact JSON copy.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--website-root", type=Path)
    args = parser.parse_args()
    audit = load_audit()
    outputs = {OUTPUT: render(audit).encode()}
    if args.website_root:
        outputs[args.website_root / "_data/pressure_references.json"] = SOURCE.read_bytes()
    for path, expected in outputs.items():
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                raise SystemExit(f"stale pressure documentation: {path}")
        else:
            path.write_bytes(expected)
    print(f"Pressure audit: 8 datasets; audited inputs match; {'checked' if args.check else 'rendered'} {len(outputs)} outputs.")


if __name__ == "__main__":
    main()
