#!/usr/bin/env python3
"""Exercise a persistent participant cache across two disposable containers.

The zero predictions here are synthetic plumbing tests, never submission data.
Only the smallest published archive is downloaded (about 21 MB).
"""
import argparse
import tempfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.drivaernetplusplus.distribution import ensure_archive, load_case_support, load_distribution
from reference.drivaernetplusplus.evaluator import score_case
from reference.drivaernetplusplus.participant_workflow import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "check"))
    parser.add_argument("--support-root", required=True, type=Path)
    args = parser.parse_args()
    check_runtime()
    archive = min(load_distribution()["archives"], key=lambda entry: entry["size_bytes"])
    ensure_archive(archive, destination=args.support_root, offline=args.mode == "check")
    if args.mode == "check":
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "prediction.npz"
            for case_id in archive["case_ids"]:
                support = load_case_support(case_id, destination=args.support_root)
                np.savez(prediction, p=np.zeros(support.point_count, dtype="<f8"))
                result = score_case(support, prediction)
                assert result["metric_values"]["surface_pressure_rel_l2"] == 100.0
                assert len(result["profiles"]) == 3
                print(f"Verified persistent cache and synthetic scoring: {case_id}")
    else:
        print(f"Downloaded and verified {archive['file']}")


if __name__ == "__main__":
    main()
