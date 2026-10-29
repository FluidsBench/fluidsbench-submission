#!/bin/bash
#SBATCH --job-name=hl_cm_pilot
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=hilift_exact_moment_pilot_%j.out
#SBATCH --error=hilift_exact_moment_pilot_%j.err

set -euo pipefail

: "${HILIFT_MOMENT_AUDIT_APPROVED:?pilot approval sentinel is required}"
: "${HILIFT_MOMENT_PYTHON:?absolute Python interpreter is required}"
: "${HILIFT_MOMENT_REPO:?absolute FluidsBench repository is required}"
: "${HILIFT_MOMENT_RECIPE:?absolute canonical recipe root is required}"
: "${HILIFT_MOMENT_DATASET:?absolute dataset root is required}"
: "${HILIFT_MOMENT_OUTPUT_ROOT:?absolute isolated pilot output root is required}"
: "${HILIFT_MOMENT_ALL1800_INVENTORY:?absolute all-1800 inventory is required}"
: "${HILIFT_MOMENT_CELL_CHUNK:?fixed pilot cell chunk is required}"

if [[ "${HILIFT_MOMENT_AUDIT_APPROVED}" != "YES_OWNER_APPROVED_EXACT_MOMENT_AUDIT_PILOT6" ]]; then
  echo "Refusing pilot: approval sentinel differs" >&2
  exit 64
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != "1" || "${SLURM_NTASKS:-}" != "1" || "${SLURM_CPUS_PER_TASK:-}" != "2" ]]; then
  echo "Refusing changed pilot allocation" >&2
  exit 64
fi
for path in \
  "${HILIFT_MOMENT_PYTHON}" \
  "${HILIFT_MOMENT_REPO}" \
  "${HILIFT_MOMENT_RECIPE}" \
  "${HILIFT_MOMENT_DATASET}" \
  "${HILIFT_MOMENT_OUTPUT_ROOT}" \
  "${HILIFT_MOMENT_ALL1800_INVENTORY}"; do
  if [[ "${path}" != /* || "${path}" == *,* ]]; then
    echo "Refusing non-absolute or comma-bearing path: ${path}" >&2
    exit 64
  fi
done
if [[ ! "${HILIFT_MOMENT_CELL_CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Cell chunk must be a positive integer" >&2
  exit 64
fi

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
  --scope pilot
  --cell-chunk "${HILIFT_MOMENT_CELL_CHUNK}"
)

export HILIFT_NUMBA_CACHE_ROLE=preflight
"${HILIFT_MOMENT_PYTHON}" "${driver}" --mode preflight "${common[@]}"
unset HILIFT_NUMBA_CACHE_ROLE
srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=2 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  "${HILIFT_MOMENT_PYTHON}" "${driver}" --mode dispatch "${common[@]}"
