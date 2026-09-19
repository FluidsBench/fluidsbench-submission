#!/usr/bin/env bash
#SBATCH --job-name=wml_prof_wide
#SBATCH --partition=cpu
# cpu-short is capped at 4 SUBMITTED jobs but allows node=20 per user, so one
# wide job buys far more concurrency than an array under cpu-normal's node=2.
# cpu-free has no node cap at all but does not schedule on this cluster even
# for a trivial job, so it is not usable.
#SBATCH --qos=cpu-short
#SBATCH --account=coreai_modulus_cae
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
# 2 tasks x ~90 GB on a 188 GB node.
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/logs/wide_%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/logs/wide_%j.err

set -uo pipefail
REPO=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench/fluidsbench-submission-windsorml
WORK=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/windsorml/fluidsbench
IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/containers/physicsnemo26.03.sqsh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy ftp_proxy no_proxy

srun --container-image="$IMAGE" \
     --container-mounts=/lustre/fs1:/lustre/fs1 \
     --container-workdir="$REPO" \
     bash "$REPO/scripts/profile_shard_worker.sh"
