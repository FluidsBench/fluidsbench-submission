#!/usr/bin/env bash
# One task of the wide profile-support fleet. Selects its cases from the
# scored-case list by SLURM_PROCID and works them in reverse order, so this
# fleet converges towards the array fleets rather than racing them.
set -uo pipefail
REPO=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/fluidsbench-submission-windsorml
WORK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench

RUNS=$(awk -v shard="${SLURM_PROCID:?}" -v n="${SLURM_NTASKS:?}" \
    'NR % n == shard % n { a[++c]=$1 } END { for (i=c; i>=1; i--) printf "%s%s", sep, a[i], sep="," }' \
    "$WORK/scored_case_runs.txt")
[ -z "$RUNS" ] && { echo "task $SLURM_PROCID: no cases"; exit 0; }
echo "task $SLURM_PROCID/$SLURM_NTASKS runs: $RUNS"

exec python3 "$REPO/scripts/build_windsorml_profile_support.py" \
    --runs "$RUNS" --out-dir "$WORK/profile_support_v3" --skip-existing
