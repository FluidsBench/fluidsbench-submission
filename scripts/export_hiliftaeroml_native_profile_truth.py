#!/usr/bin/env python3
"""Generate the inactive benchmark-owned HiLiftAeroML profile truth release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.hiliftaeroml.native_profile_truth import (  # noqa: E402
    NativeProfileTruthError,
    assemble_release,
    build_case_truth,
    build_source_index,
    load_case_universe,
    load_source_index,
    preflight,
)
from reference.hiliftaeroml.native_profile_truth_materializer import (  # noqa: E402
    build_prerequisite_authority_index,
    load_prerequisite_authority_index,
    materialize_case_truth,
    prerequisite_preflight,
)

DEFAULT_SUPPORT_MANIFEST = (
    ROOT
    / "benchmark-specs"
    / "hiliftaeroml"
    / "scoring-support"
    / "hiliftaeroml-native-all-splits-support-v1-candidate"
    / "manifest.json"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode",
        required=True,
        choices=(
            "prerequisite-index",
            "prerequisite-preflight",
            "prerequisite-case",
            "prerequisite-assemble",
            "oracle-source-index",
            "oracle-preflight",
            "oracle-case",
            "oracle-assemble",
        ),
    )
    result.add_argument(
        "--support-manifest",
        type=Path,
        default=DEFAULT_SUPPORT_MANIFEST,
        help="Frozen eight-case-set scoring-support manifest.",
    )
    result.add_argument(
        "--source-index",
        type=Path,
        help="Oracle-only evaluator-output index (never a production authority).",
    )
    result.add_argument(
        "--authority-index",
        type=Path,
        help="Internal prediction-free prerequisite authority index.",
    )
    result.add_argument(
        "--outputs-root",
        action="append",
        type=Path,
        default=[],
        help="Oracle-only evaluator-native output root; repeat in priority order.",
    )
    result.add_argument("--cp-record-root", action="append", type=Path, default=[])
    result.add_argument(
        "--velocity-record-root", action="append", type=Path, default=[]
    )
    result.add_argument(
        "--validity-record-root", action="append", type=Path, default=[]
    )
    result.add_argument(
        "--oracle-case-root",
        type=Path,
        help="Optional completed case output used only for exact array equality.",
    )
    result.add_argument("--output-dir", type=Path)
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--case-index", type=int)
    selection.add_argument("--case-id")
    result.add_argument("--cases-per-chunk", type=int, default=20)
    result.add_argument("--preflight-receipt", type=Path)
    result.add_argument("--check", action="store_true")
    return result


def _require(value: object, message: str) -> None:
    if value is None:
        raise NativeProfileTruthError(message)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        universe = load_case_universe(args.support_manifest)
        if args.mode == "prerequisite-index":
            _require(
                args.authority_index,
                "--authority-index is required in prerequisite-index mode",
            )
            body = build_prerequisite_authority_index(
                universe=universe,
                cp_record_roots=args.cp_record_root,
                velocity_record_roots=args.velocity_record_root,
                validity_record_roots=args.validity_record_root,
                output_path=args.authority_index,
                check=args.check,
            )
            print(
                json.dumps(
                    {
                        "status": body["status"],
                        "available_case_count": body["available_case_count"],
                        "missing_case_count": body["missing_case_count"],
                        "authority_index": str(args.authority_index.resolve()),
                        "prediction_outputs_used_as_authority": False,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.mode.startswith("prerequisite-"):
            _require(
                args.authority_index,
                f"--authority-index is required in {args.mode} mode",
            )
            authority, authority_sha = load_prerequisite_authority_index(
                args.authority_index,
                universe=universe,
                require_complete=args.mode
                in {"prerequisite-case", "prerequisite-assemble"},
            )
            if args.mode == "prerequisite-preflight":
                receipt = prerequisite_preflight(
                    universe=universe,
                    authority=authority,
                    authority_sha256=authority_sha,
                )
                payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                if args.preflight_receipt is not None:
                    if args.check:
                        if (
                            not args.preflight_receipt.is_file()
                            or args.preflight_receipt.is_symlink()
                            or args.preflight_receipt.read_text(encoding="utf-8")
                            != payload
                        ):
                            raise NativeProfileTruthError(
                                "preflight receipt determinism check differs"
                            )
                    else:
                        if (
                            args.preflight_receipt.exists()
                            or args.preflight_receipt.is_symlink()
                        ):
                            raise NativeProfileTruthError(
                                "refusing to overwrite preflight receipt"
                            )
                        args.preflight_receipt.parent.mkdir(parents=True, exist_ok=True)
                        args.preflight_receipt.write_text(payload, encoding="utf-8")
                print(payload, end="")
                return 0 if receipt.get("status") == "ready" else 2
            _require(args.output_dir, f"--output-dir is required in {args.mode} mode")
            if args.mode == "prerequisite-case":
                if args.case_id is not None:
                    case_id = args.case_id
                    if case_id not in universe.case_ids:
                        raise NativeProfileTruthError(
                            f"case ID is outside the frozen universe: {case_id}"
                        )
                else:
                    if args.case_index is None:
                        raise NativeProfileTruthError(
                            "--case-index or --case-id is required in prerequisite-case mode"
                        )
                    if args.case_index < 0 or args.case_index >= len(universe.case_ids):
                        raise NativeProfileTruthError("--case-index is out of range")
                    case_id = universe.case_ids[args.case_index]
                record, oracle = materialize_case_truth(
                    case_id=case_id,
                    authority=authority,
                    output_root=args.output_dir,
                    oracle_case_root=args.oracle_case_root,
                    check=args.check,
                )
                print(
                    json.dumps(
                        {
                            "status": "pass" if args.check else "complete",
                            "case_id": case_id,
                            "case_identity_sha256": record["case_identity_sha256"],
                            "oracle_audit": oracle,
                            "prediction_outputs_used_as_authority": False,
                        },
                        sort_keys=True,
                    )
                )
                return 0
            manifest = assemble_release(
                universe=universe,
                source_index_sha256=authority_sha,
                output_root=args.output_dir,
                cases_per_chunk=args.cases_per_chunk,
                check=args.check,
            )
            print(
                json.dumps(
                    {
                        "status": "pass"
                        if args.check
                        else "complete_candidate_not_published",
                        "case_count": manifest["case_count"],
                        "case_set_count": manifest["case_set_count"],
                        "truth_artifact_bytes": manifest["storage"][
                            "truth_artifact_bytes"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.mode == "oracle-source-index":
            _require(
                args.source_index, "--source-index is required in source-index mode"
            )
            if not args.outputs_root:
                raise NativeProfileTruthError(
                    "at least one --outputs-root is required in oracle-source-index mode"
                )
            body = build_source_index(
                universe=universe,
                outputs_roots=args.outputs_root,
                output_path=args.source_index,
                check=args.check,
            )
            print(
                json.dumps(
                    {
                        "status": body["status"],
                        "available_case_count": body["available_case_count"],
                        "missing_case_count": body["missing_case_count"],
                        "source_index": str(args.source_index.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return 0

        _require(args.source_index, f"--source-index is required in {args.mode} mode")
        source, source_sha = load_source_index(
            args.source_index,
            universe=universe,
            require_complete=args.mode in {"oracle-case", "oracle-assemble"},
        )
        if args.mode == "oracle-preflight":
            receipt = preflight(
                universe=universe,
                source=source,
                source_index_sha256=source_sha,
            )
            payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            if args.preflight_receipt is not None:
                if args.check:
                    if (
                        not args.preflight_receipt.is_file()
                        or args.preflight_receipt.is_symlink()
                        or args.preflight_receipt.read_text(encoding="utf-8") != payload
                    ):
                        raise NativeProfileTruthError(
                            "preflight receipt determinism check differs"
                        )
                else:
                    if (
                        args.preflight_receipt.exists()
                        or args.preflight_receipt.is_symlink()
                    ):
                        raise NativeProfileTruthError(
                            "refusing to overwrite preflight receipt"
                        )
                    args.preflight_receipt.parent.mkdir(parents=True, exist_ok=True)
                    args.preflight_receipt.write_text(payload, encoding="utf-8")
            print(payload, end="")
            return 0

        _require(args.output_dir, f"--output-dir is required in {args.mode} mode")
        if args.mode == "oracle-case":
            if args.case_id is not None:
                case_id = args.case_id
                if case_id not in universe.case_ids:
                    raise NativeProfileTruthError(
                        f"case ID is outside the frozen universe: {case_id}"
                    )
            else:
                if args.case_index is None:
                    raise NativeProfileTruthError(
                        "--case-index or --case-id is required in case mode"
                    )
                if args.case_index < 0 or args.case_index >= len(universe.case_ids):
                    raise NativeProfileTruthError("--case-index is out of range")
                case_id = universe.case_ids[args.case_index]
            record = build_case_truth(
                case_id=case_id,
                source=source,
                output_root=args.output_dir,
                check=args.check,
            )
            print(
                json.dumps(
                    {
                        "status": "pass" if args.check else "complete",
                        "case_id": case_id,
                        "case_identity_sha256": record["case_identity_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        manifest = assemble_release(
            universe=universe,
            source_index_sha256=source_sha,
            output_root=args.output_dir,
            cases_per_chunk=args.cases_per_chunk,
            check=args.check,
        )
        print(
            json.dumps(
                {
                    "status": "pass"
                    if args.check
                    else "complete_candidate_not_published",
                    "case_count": manifest["case_count"],
                    "case_set_count": manifest["case_set_count"],
                    "truth_artifact_bytes": manifest["storage"]["truth_artifact_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (NativeProfileTruthError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
