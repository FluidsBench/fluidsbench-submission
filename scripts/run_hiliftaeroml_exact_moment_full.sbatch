#!/bin/bash
#SBATCH --job-name=hl_cm_all1800
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=hilift_exact_moment_full_%j.out
#SBATCH --error=hilift_exact_moment_full_%j.err

set -euo pipefail

: "${HILIFT_MOMENT_AUDIT_APPROVED:?full approval sentinel is required}"
: "${HILIFT_MOMENT_PYTHON:?absolute Python interpreter is required}"
: "${HILIFT_MOMENT_REPO:?absolute FluidsBench repository is required}"
: "${HILIFT_MOMENT_RECIPE:?absolute canonical recipe root is required}"
: "${HILIFT_MOMENT_DATASET:?absolute dataset root is required}"
: "${HILIFT_MOMENT_OUTPUT_ROOT:?absolute isolated campaign output root is required}"
: "${HILIFT_MOMENT_ALL1800_INVENTORY:?absolute all-1800 inventory is required}"
: "${HILIFT_MOMENT_AGGREGATE:?absolute aggregate path is required}"
: "${HILIFT_MOMENT_VALIDATION:?absolute validation receipt path is required}"
: "${HILIFT_MOMENT_CELL_CHUNK:?fixed campaign cell chunk is required}"

if [[ "${HILIFT_MOMENT_AUDIT_APPROVED}" != "YES_OWNER_APPROVED_EXACT_MOMENT_AUDIT_ALL1800" ]]; then
  echo "Refusing full audit: approval sentinel differs" >&2
  exit 64
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != "8" || "${SLURM_NTASKS:-}" != "8" || "${SLURM_CPUS_PER_TASK:-}" != "2" ]]; then
  echo "Refusing changed full allocation; require 8 nodes/tasks and 2 CPUs/task" >&2
  exit 64
fi
for path in \
  "${HILIFT_MOMENT_PYTHON}" \
  "${HILIFT_MOMENT_REPO}" \
  "${HILIFT_MOMENT_RECIPE}" \
  "${HILIFT_MOMENT_DATASET}" \
  "${HILIFT_MOMENT_OUTPUT_ROOT}" \
  "${HILIFT_MOMENT_ALL1800_INVENTORY}" \
  "${HILIFT_MOMENT_AGGREGATE}" \
  "${HILIFT_MOMENT_VALIDATION}"; do
  if [[ "${path}" != /* || "${path}" == *,* ]]; then
    echo "Refusing non-absolute or comma-bearing path: ${path}" >&2
    exit 64
  fi
done
if [[ ! "${HILIFT_MOMENT_CELL_CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Cell chunk must be a positive integer" >&2
  exit 64
fi
restart_count="${SLURM_RESTART_COUNT:-0}"
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ || ! "${restart_count}" =~ ^[0-9]+$ ]]; then
  echo "Numeric Slurm job/restart metadata is required" >&2
  exit 64
fi
attempt_id="${SLURM_JOB_ID}-r${restart_count}"

export HILIFT_EXCLUDE_SYSTEM_DIST_PACKAGES=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export HILIFT_NUMBA_CACHE_PARENT="${HILIFT_MOMENT_OUTPUT_ROOT}/numba-cache"

driver="${HILIFT_MOMENT_REPO}/scripts/audit_hiliftaeroml_exact_moments.py"
common=(
  --dataset-root "${HILIFT_MOMENT_DATASET}"
  --recipe-root "${HILIFT_MOMENT_RECIPE}"
  --output-root "${HILIFT_MOMENT_OUTPUT_ROOT}"
  --all1800-inventory "${HILIFT_MOMENT_ALL1800_INVENTORY}"
  --scope full
  --cell-chunk "${HILIFT_MOMENT_CELL_CHUNK}"
)

export HILIFT_NUMBA_CACHE_ROLE=preflight
"${HILIFT_MOMENT_PYTHON}" "${driver}" --mode preflight "${common[@]}"
unset HILIFT_NUMBA_CACHE_ROLE
srun \
  --nodes=8 \
  --ntasks=8 \
  --ntasks-per-node=1 \
  --cpus-per-task=2 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  "${HILIFT_MOMENT_PYTHON}" "${driver}" --mode dispatch "${common[@]}"

if [[ "${HILIFT_MOMENT_AGGREGATE}" == *.json ]]; then
  aggregate_target="${HILIFT_MOMENT_AGGREGATE%.json}.${attempt_id}.json"
else
  aggregate_target="${HILIFT_MOMENT_AGGREGATE}.${attempt_id}.json"
fi
if [[ "${HILIFT_MOMENT_VALIDATION}" == *.json ]]; then
  validation_target="${HILIFT_MOMENT_VALIDATION%.json}.${attempt_id}.json"
else
  validation_target="${HILIFT_MOMENT_VALIDATION}.${attempt_id}.json"
fi
"${HILIFT_MOMENT_PYTHON}" "${driver}" \
  --mode aggregate \
  "${common[@]}" \
  --rank-attempt-id "${attempt_id}" \
  --aggregate-output "${aggregate_target}"
"${HILIFT_MOMENT_PYTHON}" "${driver}" \
  --mode validate \
  "${common[@]}" \
  --rank-attempt-id "${attempt_id}" \
  --aggregate-output "${aggregate_target}" \
  --validation-receipt "${validation_target}"
