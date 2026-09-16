from __future__ import annotations

from pathlib import Path

import pytest
from reference.hiliftaeroml.native_profile_truth import NativeProfileTruthError
from scripts.hiliftaeroml_native_profile_truth_fast_dispatch import (
    EXPECTED_CASE_COUNT,
    EXPECTED_TASK_COUNT,
    _validate_output_boundary,
    striped_indices,
)

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts/run_hiliftaeroml_native_profile_truth_fast.sbatch"
SUBMITTER = ROOT / "scripts/submit_hiliftaeroml_native_profile_truth_fast.sh"


def test_fast_dispatch_stripes_exact_frozen_universe_once() -> None:
    stripes = [striped_indices(rank) for rank in range(EXPECTED_TASK_COUNT)]
    assert [len(stripe) for stripe in stripes] == [
        170,
        170,
        170,
        169,
        169,
        169,
        169,
        169,
    ]
    assert all(
        tuple(range(rank, EXPECTED_CASE_COUNT, EXPECTED_TASK_COUNT)) == stripe
        for rank, stripe in enumerate(stripes)
    )
    flattened = [index for stripe in stripes for index in stripe]
    assert len(flattened) == EXPECTED_CASE_COUNT
    assert len(set(flattened)) == EXPECTED_CASE_COUNT
    assert sorted(flattened) == list(range(EXPECTED_CASE_COUNT))


@pytest.mark.parametrize("rank", [-1, EXPECTED_TASK_COUNT])
def test_fast_dispatch_rejects_out_of_range_rank(rank: int) -> None:
    with pytest.raises(NativeProfileTruthError, match="rank is out of range"):
        striped_indices(rank)


def test_fast_dispatch_rejects_changed_task_or_case_count() -> None:
    with pytest.raises(NativeProfileTruthError, match="task count"):
        striped_indices(0, task_count=7)
    with pytest.raises(NativeProfileTruthError, match="case count"):
        striped_indices(0, case_count=1354)


def test_fast_dispatch_requires_fresh_isolated_output(tmp_path: Path) -> None:
    output = tmp_path / "release"
    receipt = tmp_path / "validation.json"
    _validate_output_boundary(
        output_dir=output, validation_receipt=receipt, require_fresh=True
    )
    output.mkdir()
    with pytest.raises(NativeProfileTruthError, match="already exists"):
        _validate_output_boundary(
            output_dir=output, validation_receipt=receipt, require_fresh=True
        )
    with pytest.raises(NativeProfileTruthError, match="outside"):
        _validate_output_boundary(
            output_dir=tmp_path / "new-release",
            validation_receipt=tmp_path / "new-release/validation.json",
            require_fresh=True,
        )


def test_fast_batch_resources_and_pipeline_are_exact() -> None:
    text = BATCH.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=cpu",
        "#SBATCH --qos=cpu-short",
        "#SBATCH --nodes=8",
        "#SBATCH --ntasks=8",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=32G",
        "#SBATCH --time=04:00:00",
    ):
        assert directive in text
    assert text.count("--kill-on-bad-exit=1") == 2
    assert text.count("--mode prerequisite-assemble") == 2
    assert text.count("--mode generate") == 1
    assert text.count("--mode check") == 1
    assert "--metadata-only" not in text
    assert '--receipt "${HILIFT_TRUTH_VALIDATION_RECEIPT}"' in text
    assert text.index("--mode generate") < text.index("--mode prerequisite-assemble")
    assert text.index("--mode prerequisite-assemble") < text.index("--mode check")
    assert text.rindex("--mode prerequisite-assemble") < text.index(
        '--receipt "${HILIFT_TRUTH_VALIDATION_RECEIPT}"'
    )
    for forbidden in ("hiliftaeroml-submission", "leaderboard", "git push", "publish"):
        assert forbidden not in text.lower()


def test_fast_submitter_is_dry_run_by_default_and_double_gates_submission() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    assert 'mode="${1:---dry-run}"' in text
    assert "sbatch --test-only" in text
    assert "YES_OWNER_APPROVED_FAST_PROFILE_TRUTH_1355" in text
    assert "YES_OWNER_APPROVED_SUBMIT_FAST_PROFILE_TRUTH_1355" in text
    assert text.index("--test-only") < text.index("--parsable")
