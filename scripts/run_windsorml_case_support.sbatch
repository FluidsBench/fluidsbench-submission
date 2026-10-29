#!/usr/bin/env bash
#SBATCH --job-name=wml_support
#SBATCH --partition=cpu
# cpu-short caps at 4 submitted jobs per user (MaxSubmitPU=4), which a 20-task
# array violates. cpu-normal allows 200 submissions and a 1-day wall; its
# node=2 cap is not binding here because twenty 4-CPU tasks pack onto two
# 96-core nodes.
#SBATCH --qos=cpu-normal
#SBATCH --account=coreai_modulus_cae
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/logs/support_%A_%a.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/logs/support_%A_%a.err

set -euo pipefail

REPO="${WINDSORML_REPO:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/fluidsbench-submission-windsorml}"
OUT=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/case_support
IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/containers/physicsnemo26.03.sqsh

# The login process supplies a localhost-only proxy that does not exist on
# compute nodes; this job is offline anyway, but keep the environment clean.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy ftp_proxy no_proxy

srun --container-image="$IMAGE" \
     --container-mounts=/lustre/fs1:/lustre/fs1 \
     --container-workdir="$REPO" \
     python3 "$REPO/scripts/build_windsorml_case_support.py" \
        --task-id "$SLURM_ARRAY_TASK_ID" \
        --num-shards 20 \
        --out-dir "$OUT"
