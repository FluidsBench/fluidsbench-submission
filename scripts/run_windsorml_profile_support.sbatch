#!/usr/bin/env bash
#SBATCH --job-name=wml_profiles
#SBATCH --partition=cpu
# cpu-short caps at 4 submitted jobs, so an array needs cpu-normal. Its
# MaxTRESPU of node=2 plus the 83 GB per-case footprint means about four tasks
# run concurrently; the rest backfill.
#SBATCH --qos=cpu-normal
#SBATCH --account=coreai_modulus_cae
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Measured MaxRSS is 83 GB per case: the 291M-cell volume mesh plus its cell
# centres. 90 GB leaves headroom without dropping below two tasks per node.
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/logs/profiles_%A_%a.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/logs/profiles_%A_%a.err

set -euo pipefail

REPO="${WINDSORML_REPO:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/fluidsbench-submission-windsorml}"
WORK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench
IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/containers/physicsnemo26.03.sqsh
SHARDS=20

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy ftp_proxy no_proxy

# Interleave the scored cases so every shard gets a similar mix of geometries.
RUNS=$(awk -v shard="$SLURM_ARRAY_TASK_ID" -v n="$SHARDS" \
    'NR % n == shard % n { printf "%s%s", sep, $1; sep="," }' \
    "$WORK/scored_case_runs.txt")

if [ -z "$RUNS" ]; then
    echo "shard $SLURM_ARRAY_TASK_ID has no cases"
    exit 0
fi
echo "shard $SLURM_ARRAY_TASK_ID runs: $RUNS"

srun --container-image="$IMAGE" \
     --container-mounts=/lustre/fs1:/lustre/fs1 \
     --container-workdir="$REPO" \
     python3 "$REPO/scripts/build_windsorml_profile_support.py" \
        --runs "$RUNS" \
        --out-dir "$WORK/profile_support_v3" \
        --skip-existing
