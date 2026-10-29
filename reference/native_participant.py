"""Resumable dev submissions using the AhmedML and WindsorML evaluators.

This module owns orchestration only: native physics, profile sampling, split
reduction and package validation remain with each dataset's existing modules.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import html
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reference.ahmedml.contract import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "surface_and_volume"
RUNTIME = ("Linux", "3.12.13", "2.2.6", "9.5.2")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}\Z")


class WorkflowError(ValueError):
    """The participant inputs cannot be evaluated or resumed reproducibly."""


def validate_package_policy(config):
    allowed = {
        "schema",
        "split_id",
        "profile_cases_per_chunk",
        "inference_attestation",
        "participant",
        "evaluation",
        "spatial_discretization",
        "release_bindings",
        "prediction_scope",
        "force_prediction_source",
    }
    if set(config) - allowed:
        raise WorkflowError(
            "unknown package config keys; custom case lists and direct force values are forbidden"
        )
    if config.get("prediction_scope", SCOPE) != SCOPE:
        raise WorkflowError("this dataset requires surface_and_volume predictions")
    if config.get("force_prediction_source", "field_integrated") != "field_integrated":
        raise WorkflowError("only field_integrated forces are accepted")


def load_json(path):
    return read_json(path, label="workflow JSON")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_tree(directory, *, exclude=()):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise WorkflowError(f"saved outputs must not contain symlinks: {path}")
        if path.is_file() and path.relative_to(directory).as_posix() not in exclude:
            result[path.relative_to(directory).as_posix()] = sha256_file(path)
    return result


def check_runtime():
    import numpy

    try:
        import vtk
    except ImportError as error:
        raise WorkflowError(
            "use the pinned participant requirements or Dockerfile.native-participant"
        ) from error
    actual = (
        platform.system(),
        platform.python_version(),
        numpy.__version__,
        vtk.vtkVersion.GetVTKVersion(),
    )
    if actual != RUNTIME:
        raise WorkflowError(
            "evaluation requires Linux, Python 3.12.13, NumPy 2.2.6 and VTK 9.5.2; use Dockerfile.native-participant"
        )


@contextlib.contextmanager
def campaign_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a+b"
    ) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowError(
                "another evaluation is using this output directory"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class ParticipantWorkflow:
    def __init__(self, dataset):
        if dataset not in ("ahmedml", "windsorml"):
            raise WorkflowError("unsupported participant dataset")
        self.dataset = dataset
        self.spec_root = ROOT / "benchmark-specs" / dataset
        self.spec_path = self.spec_root / "submission-spec.json"
        self.contract = importlib.import_module(f"reference.{dataset}.contract")
        self.predictions = importlib.import_module(
            f"reference.{dataset}.prediction_chunks"
        )

    def official_split(self, split_id):
        spec = load_json(self.spec_path)
        matches = [s for s in spec["splits"] if s["id"] == split_id]
        if len(matches) != 1:
            raise WorkflowError(
                f"split_id must be one of the prescribed splits: {[s['id'] for s in spec['splits']]}"
            )
        declaration = matches[0]
        path = self.spec_root / declaration["index_file"]
        if sha256_file(path) != declaration["sha256"]:
            raise WorkflowError("prescribed split SHA-256 differs")
        result = load_json(path)
        if len(result["case_ids"]) != declaration["case_count"]:
            raise WorkflowError("prescribed split case count differs")
        return result

    def load_entry(self, entry_root):
        entry = load_json(Path(entry_root) / "entry.json")
        required = {"schema", "submission_id", "split_id", "prediction_scope"}
        if not required <= set(entry) or set(entry) - required - {
            "force_prediction_source"
        }:
            raise WorkflowError(
                "entry has missing or unknown keys; custom case lists and direct force values are forbidden"
            )
        if (
            entry["schema"] != f"fluidsbench-{self.dataset}-entry-v1"
            or entry["prediction_scope"] != SCOPE
        ):
            raise WorkflowError(
                "this dataset requires the matching entry schema and surface_and_volume predictions"
            )
        if not isinstance(entry["submission_id"], str) or not SAFE_ID.fullmatch(
            entry["submission_id"]
        ):
            raise WorkflowError("submission_id must be a safe lowercase identifier")
        if (
            entry.get("force_prediction_source", "field_integrated")
            != "field_integrated"
        ):
            raise WorkflowError("only field_integrated forces are accepted")
        self.official_split(entry["split_id"])
        return entry

    def source_identity(self):
        path = (
            self.spec_root
            / "public-source-identity"
            / f"{self.dataset}-public-source-identity-v1.json"
        )
        return self.contract.load_source_identity(
            path, expected_sha256=self.contract.SOURCE_IDENTITY_SHA256
        )

    def native_files(self, case):
        area = (
            case.surface_cell_area
            if self.dataset == "ahmedml"
            else case.surface_dual_area
        )
        return [
            case.boundary,
            area,
            case.volume,
            case.geometry_parameters,
            case.force_coefficients,
        ]

    def fetch_data(self, split_id, destination, *, dry_run=False):
        cases = self.official_split(split_id)["case_ids"]
        identity = self.source_identity()
        files = [f for cid in cases for f in self.native_files(identity.case(cid))]
        plan = {
            "repository": identity.repository_id,
            "revision": identity.repository_revision,
            "case_count": len(cases),
            "total_bytes": sum(f.size_bytes for f in files),
            "files": [f.path.as_posix() for f in files],
        }
        if not dry_run:
            hf = shutil.which("hf")
            if not hf:
                raise WorkflowError(
                    "install requirements-native-participant.txt to obtain the hf downloader"
                )
            subprocess.run(
                [
                    hf,
                    "download",
                    identity.repository_id,
                    *plan["files"],
                    "--type",
                    "dataset",
                    "--revision",
                    identity.repository_revision,
                    "--local-dir",
                    str(Path(destination).expanduser().resolve()),
                ],
                check=True,
            )
        return plan

    def support_root(self, destination=None):
        if self.dataset == "ahmedml":
            from reference.ahmedml.distribution import cache_root

            return cache_root(Path(destination) if destination is not None else None)
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        return (
            Path(destination or base / "fluidsbench/windsorml").expanduser().resolve()
        )

    def profile_manifest(self, destination=None, *, offline=False):
        binding = load_json(ROOT / "leaderboard/manifest.json")["data_release"][
            "profile_ground_truth"
        ]
        path = (
            self.support_root(destination)
            / "profile-ground-truth"
            / binding["manifest_sha256"]
            / "manifest.json"
        )
        if path.exists():
            if sha256_file(path) != binding["manifest_sha256"]:
                raise WorkflowError(
                    "cached profile-ground-truth manifest SHA-256 differs"
                )
        elif offline:
            raise WorkflowError(
                "profile-ground-truth manifest is missing offline; run fetch-support on a connected machine"
            )
        else:
            from scripts.fetch_profile_ground_truth import fetch_manifest

            fetch_manifest(path)
        return path

    def fetch_support(self, split_id, destination=None, *, offline=False):
        cases = self.official_split(split_id)["case_ids"]
        root = self.support_root(destination)
        if self.dataset == "ahmedml":
            from reference.ahmedml.distribution import (
                ensure_case_support,
                load_distribution,
            )

            index = None if offline else load_distribution()
            for cid in cases:
                ensure_case_support(cid, destination=root, offline=offline, index=index)
        else:
            from reference.windsorml.profiles import load_profile_support

            for cid in cases:
                load_profile_support(case_id=cid)
        manifest = self.profile_manifest(root, offline=offline)
        return {
            "case_count": len(cases),
            "root": str(root),
            "profile_ground_truth_manifest": str(manifest),
            "native_support": (
                "verified_download_cache"
                if self.dataset == "ahmedml"
                else "verified_bundled_files"
            ),
        }

    def prediction_manifests(self, entry_root, case):
        root = Path(entry_root) / "cases" / case.case_id
        if not root.is_dir() or {p.name for p in root.iterdir()} != {
            "surface",
            "volume",
        }:
            raise WorkflowError(
                f"{case.case_id}: expected only surface and volume prediction directories; direct force files are forbidden"
            )
        result = {}
        for domain in ("surface", "volume"):
            manifest = self.predictions.load_prediction_chunk_manifest(
                root / domain / "manifest.json"
            )
            association = (
                "PointData"
                if domain == "surface" and self.dataset == "windsorml"
                else "CellData"
            )
            suffix = "points" if association == "PointData" else "cells"
            if (
                manifest.case_id != case.case_id
                or manifest.support_id != f"{self.dataset}_{domain}_native_{suffix}"
                or manifest.association != association
                or manifest.total_row_count != getattr(case, f"{domain}_entity_count")
            ):
                raise WorkflowError(
                    f"{case.case_id}/{domain}: prediction identity, association or native count differs"
                )
            if any(c.row_count > 1_000_000 for c in manifest.chunks):
                raise WorkflowError(
                    "prediction chunks may contain at most 1000000 rows"
                )
            result[domain] = manifest
        return result

    def preflight(self, entry_root, dataset_root, support_root=None, *, offline=False):
        entry_root, dataset_root = [
            Path(p).expanduser().resolve() for p in (entry_root, dataset_root)
        ]
        entry = self.load_entry(entry_root)
        cases = self.official_split(entry["split_id"])["case_ids"]
        case_root = entry_root / "cases"
        if not case_root.is_dir() or {p.name for p in case_root.iterdir()} != set(
            cases
        ):
            raise WorkflowError(
                "prediction cases must exactly match the prescribed test split"
            )
        identity = self.source_identity()
        for cid in cases:
            case = identity.case(cid)
            self.prediction_manifests(entry_root, case)
            for item in self.native_files(case):
                path = dataset_root / item.path
                if not path.is_file() or path.stat().st_size != item.size_bytes:
                    raise WorkflowError(
                        f"missing or wrong-sized native file: {path}; run fetch-data"
                    )
        support = self.fetch_support(entry["split_id"], support_root, offline=offline)
        return {
            "status": "preflight_passed",
            "entry": entry,
            "case_count": len(cases),
            "support": support,
            "prediction_arrays": "hashes, values and native order checked before each case completes",
        }

    def implementation_identity(self):
        paths = list((ROOT / "reference").rglob("*.py")) + list(
            (ROOT / "scripts").glob("*.py")
        )
        paths += list(self.spec_root.rglob("*.json"))
        paths += [ROOT / "requirements-native-participant.txt"]
        return digest(
            {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}
        )

    @staticmethod
    def prediction_binding(manifests):
        for manifest in manifests.values():
            if sha256_file(manifest.path) != manifest.sha256:
                raise WorkflowError("prediction manifest changed")
            for chunk in manifest.chunks:
                if sha256_file(chunk.path) != chunk.sha256:
                    raise WorkflowError(
                        f"prediction chunk SHA-256 differs: {chunk.path}"
                    )
        return {
            domain: {"manifest": m.sha256, "chunks": [c.sha256 for c in m.chunks]}
            for domain, m in manifests.items()
        }

    def run_case(self, cid, manifests, dataset_root, support_root, output):
        args = [
            sys.executable,
            str(ROOT / f"scripts/evaluate_{self.dataset}_candidate_case.py"),
            "--case-id",
            cid,
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(output),
        ]
        if self.dataset == "ahmedml":
            args += ["--support-root", str(support_root), "--offline"]
        for domain in ("surface", "volume"):
            flag = (
                f"--{domain}-prediction-manifest"
                if self.dataset == "ahmedml"
                else f"--{domain}-manifest"
            )
            args += [flag, str(manifests[domain].path)]
        subprocess.run(args, check=True)

    def reduce_cases(self, entry, evidence_root, output):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / f"scripts/score_{self.dataset}_candidate_dataset.py"),
                "--split-id",
                entry["split_id"],
                "--case-evidence-directory",
                str(evidence_root),
                "--output",
                str(output),
            ],
            check=True,
        )

    def evaluate_entry(
        self,
        entry_root,
        dataset_root,
        output,
        support_root=None,
        *,
        resume=False,
        offline=False,
    ):
        check_runtime()
        entry_root, dataset_root, output = [
            Path(p).expanduser().resolve() for p in (entry_root, dataset_root, output)
        ]
        with campaign_lock(output):
            if any(p.name != ".lock" for p in output.iterdir()) and not resume:
                raise WorkflowError(
                    "output is not empty; use --resume or choose a new output directory"
                )
            checked = self.preflight(
                entry_root, dataset_root, support_root, offline=offline
            )
            entry, support = checked["entry"], checked["support"]
            cases = self.official_split(entry["split_id"])["case_ids"]
            identity = self.source_identity()
            implementation = self.implementation_identity()
            campaign = {
                "schema": "fluidsbench-native-campaign-v1",
                "entry": entry,
                "case_ids": cases,
                "implementation": implementation,
                "runtime": list(RUNTIME),
                "profile_manifest": sha256_file(
                    support["profile_ground_truth_manifest"]
                ),
            }
            marker = output / "campaign.json"
            if marker.exists():
                if load_json(marker) != campaign:
                    raise WorkflowError(
                        "campaign inputs or evaluator changed; choose a new output directory"
                    )
            else:
                if (output / "cases").exists() or (output / "evaluation").exists():
                    raise WorkflowError("existing outputs have no campaign identity")
                with tempfile.TemporaryDirectory(
                    prefix=".identity-", dir=output
                ) as temporary:
                    staged = Path(temporary) / "campaign.json"
                    write_json(staged, campaign)
                    staged.rename(marker)
            case_root = output / "cases"
            case_root.mkdir(exist_ok=True)
            if {p.name for p in case_root.iterdir()} - set(cases):
                raise WorkflowError(
                    "checkpoint directory contains cases outside the prescribed split"
                )
            bindings, reused = {}, 0
            for index, cid in enumerate(cases, 1):
                manifests = self.prediction_manifests(entry_root, identity.case(cid))
                inputs = {
                    "campaign": digest(campaign),
                    "predictions": self.prediction_binding(manifests),
                }
                destination = case_root / cid
                if destination.exists():
                    receipt = load_json(destination / "checkpoint.json")
                    if (
                        receipt.get("inputs") != inputs
                        or receipt.get("schema") != "fluidsbench-native-checkpoint-v1"
                        or set(receipt.get("outputs", {})) != {"evidence.json"}
                        or file_tree(destination, exclude=("checkpoint.json",))
                        != receipt["outputs"]
                    ):
                        raise WorkflowError(
                            f"{cid}: checkpoint inputs or output changed; choose a new output directory"
                        )
                    reused += 1
                    print(
                        f"[{index}/{len(cases)}] {cid}: verified checkpoint",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"[{index}/{len(cases)}] {cid}: evaluating",
                        file=sys.stderr,
                        flush=True,
                    )
                    with tempfile.TemporaryDirectory(
                        prefix=".case-", dir=output
                    ) as temporary:
                        staged = Path(temporary) / cid
                        staged.mkdir()
                        self.run_case(
                            cid,
                            manifests,
                            dataset_root,
                            Path(support["root"]),
                            staged / "evidence.json",
                        )
                        if (
                            self.prediction_binding(manifests) != inputs["predictions"]
                            or self.implementation_identity() != implementation
                        ):
                            raise WorkflowError(
                                "inputs or evaluator changed during evaluation"
                            )
                        outputs = file_tree(staged)
                        if set(outputs) != {"evidence.json"}:
                            raise WorkflowError(
                                "case evaluator did not produce exactly one evidence file"
                            )
                        write_json(
                            staged / "checkpoint.json",
                            {
                                "schema": "fluidsbench-native-checkpoint-v1",
                                "inputs": inputs,
                                "outputs": outputs,
                            },
                        )
                        staged.rename(destination)
                bindings[cid] = sha256_file(destination / "evidence.json")
            final = output / "evaluation"
            if final.exists():
                receipt = self.verify_evaluation(final)
                if (
                    receipt["campaign"] != campaign
                    or receipt["case_evidence"] != bindings
                ):
                    raise WorkflowError(
                        "completed evaluation differs from case checkpoints"
                    )
            else:
                with tempfile.TemporaryDirectory(
                    prefix=".reduce-", dir=output
                ) as temporary:
                    staged = Path(temporary) / "evaluation"
                    evidence = staged / "case-evidence"
                    evidence.mkdir(parents=True)
                    for cid in cases:
                        shutil.copyfile(
                            case_root / cid / "evidence.json", evidence / f"{cid}.json"
                        )
                    self.reduce_cases(entry, evidence, staged / "dataset-evidence.json")
                    shutil.copyfile(
                        support["profile_ground_truth_manifest"],
                        staged / "profile-ground-truth.json",
                    )
                    if self.implementation_identity() != implementation:
                        raise WorkflowError("evaluator changed during reduction")
                    write_json(
                        staged / "workflow.json",
                        {
                            "schema": "fluidsbench-native-evaluation-v1",
                            "campaign": campaign,
                            "case_evidence": bindings,
                            "outputs": file_tree(staged),
                        },
                    )
                    self.verify_evaluation(staged)
                    staged.rename(final)
            return {
                "status": "complete_dev_evaluation",
                "case_count": len(cases),
                "reused_cases": reused,
                "output": str(final),
            }

    def verify_evaluation(self, evaluation_root):
        root = Path(evaluation_root).expanduser().resolve()
        receipt = load_json(root / "workflow.json")
        campaign = receipt.get("campaign", {})
        entry = campaign.get("entry", {})
        if (
            receipt.get("schema") != "fluidsbench-native-evaluation-v1"
            or entry.get("schema") != f"fluidsbench-{self.dataset}-entry-v1"
            or campaign.get("implementation") != self.implementation_identity()
            or campaign.get("runtime") != list(RUNTIME)
        ):
            raise WorkflowError("evaluation identity or evaluator differs")
        cases = self.official_split(entry.get("split_id"))["case_ids"]
        expected = {f"case-evidence/{cid}.json" for cid in cases} | {
            "dataset-evidence.json",
            "profile-ground-truth.json",
        }
        actual = file_tree(root, exclude=("workflow.json",))
        if (
            set(actual) != expected
            or actual != receipt.get("outputs")
            or campaign.get("case_ids") != cases
        ):
            raise WorkflowError("evaluation output coverage or hashes changed")
        if receipt.get("case_evidence") != {
            cid: actual[f"case-evidence/{cid}.json"] for cid in cases
        }:
            raise WorkflowError("evaluation case bindings differ")
        binding = load_json(ROOT / "leaderboard/manifest.json")["data_release"][
            "profile_ground_truth"
        ]
        if actual["profile-ground-truth.json"] != binding["manifest_sha256"]:
            raise WorkflowError("evaluation profile manifest differs from this release")
        return receipt

    def verify_package(self, directory):
        directory = Path(directory).expanduser().resolve()
        specification = load_json(self.spec_path)
        if specification.get("status") == "official":
            from scripts.validate_submission import validate_submission_file

            errors, _ = validate_submission_file(
                directory / "submission.json",
                contributor_stage=True,
                require_registered_path=False,
            )
            if errors:
                raise WorkflowError("; ".join(errors))
            return {"status": "valid_official_package", "output": str(directory)}
        pre_release = importlib.import_module(f"reference.{self.dataset}.pre_release")
        pre_release.package_tree_binding(directory)
        submission = load_json(directory / "submission.json")
        sid = submission.get("submission_id")
        if (
            not isinstance(sid, str)
            or not SAFE_ID.fullmatch(sid)
            or submission.get("dataset_id") != self.dataset
        ):
            raise WorkflowError("package submission_id or dataset_id differs")
        with tempfile.TemporaryDirectory(prefix=f"{self.dataset}-verify-") as temporary:
            root = Path(temporary)
            copied = root / "submissions" / self.dataset / sid
            shutil.copytree(directory, copied)
            pre_release.build_registry_entry(
                copied / "submission.json", repository_root=root
            )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_submission.py"),
                "--candidate-dry-run",
                str(directory),
            ],
            check=True,
        )
        return {"status": "valid_dev_package", "output": str(directory)}

    def package(
        self, evaluation_root, config, discretization_cases, output, checkpoints
    ):
        root, output = [
            Path(p).expanduser().resolve() for p in (evaluation_root, output)
        ]
        receipt = self.verify_evaluation(root)
        entry = receipt["campaign"]["entry"]
        value = load_json(config)
        if (
            value.get("split_id") != entry["split_id"]
            or value.get("participant", {}).get("submission_id")
            != entry["submission_id"]
            or value.get("prediction_scope", SCOPE) != SCOPE
        ):
            raise WorkflowError(
                "package config must match the evaluated submission_id, split and scope"
            )
        if output.exists():
            raise WorkflowError("package output already exists")
        assembler = importlib.import_module(
            f"scripts.assemble_{self.dataset}_schema_v3_candidate"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".package-", dir=output.parent
        ) as temporary:
            staged = Path(temporary) / output.name
            assembler.assemble_package(
                config_path=Path(config),
                specification_path=self.spec_path,
                case_evidence_root=root / "case-evidence",
                discretization_cases_path=Path(discretization_cases),
                profile_truth_manifest_path=root / "profile-ground-truth.json",
                checkpoint_paths=checkpoints,
                dataset_evidence_path=root / "dataset-evidence.json",
                output_path=staged,
            )
            if load_json(self.spec_path).get("status") == "official":
                from scripts.finalize_official_native_package import finalize

                finalize(staged, self.spec_path)
            self.verify_package(staged)
            staged.rename(output)
        return {
            "status": (
                "validated_official_package"
                if load_json(self.spec_path).get("status") == "official"
                else "validated_dev_package"
            ),
            "output": str(output),
        }

    def report(self, evaluation_root, case_id, output):
        root = Path(evaluation_root).expanduser().resolve()
        receipt = self.verify_evaluation(root)
        if case_id not in receipt["campaign"]["case_ids"]:
            raise WorkflowError("case is absent from the evaluated split")
        case = load_json(root / "case-evidence" / f"{case_id}.json")
        dataset = load_json(root / "dataset-evidence.json")
        profiles = case["profiles"]
        if self.dataset == "ahmedml":
            series = [(s["panel_id"], s) for s in profiles["series"]]
        else:
            series = [
                (family, s)
                for family, stations in profiles["families"].items()
                for s in stations
            ]
        rows = [
            "<!doctype html><meta charset='utf-8'><title>FluidsBench case report</title>",
            "<style>body{font:16px system-ui;margin:2rem;max-width:1000px}svg{width:100%;max-width:640px;border:1px solid #ccc}td,th{padding:.3rem;text-align:left}</style>",
            f"<h1>{html.escape(self.dataset)} / {html.escape(case_id)}</h1><p>Development result. Blue: truth. Orange: prediction.</p>",
            "<h2>Complete-split metrics</h2><table>",
        ]
        for key, value in dataset["metric_values"].items():
            rows.append(
                f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
            )
        rows.append("</table><h2>Case profiles</h2>")
        for family, s in series:
            x, truth, prediction = s["coordinate"], s["truth"], s["prediction"]
            xmin, xmax, ymin, ymax = (
                min(x),
                max(x),
                min(*truth, *prediction),
                max(*truth, *prediction),
            )
            rows.append(
                f"<section><h3>{html.escape(family)} / {html.escape(s['station_id'])}</h3><svg viewBox='0 0 640 240' role='img' aria-label='Truth and prediction profile'>"
            )
            for values, color in ((truth, "#1671b8"), (prediction, "#d96b00")):
                points = " ".join(
                    f"{20+(xi-xmin)/(xmax-xmin or 1)*600:.3f},{220-(yi-ymin)/(ymax-ymin or 1)*200:.3f}"
                    for xi, yi in zip(x, values, strict=True)
                )
                rows.append(
                    f"<polyline fill='none' stroke='{color}' stroke-width='1.5' points='{points}'/>"
                )
            rows.append(
                f"</svg><p>Coordinate: {xmin:.4g} to {xmax:.4g}; {html.escape(s['quantity_id'])}: {ymin:.4g} to {ymax:.4g}</p></section>"
            )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write("\n".join(rows))
        return {
            "status": "report_written",
            "output": str(output),
            "profile_count": len(series),
        }
