#!/bin/bash
#SBATCH --job-name=ahmed_support
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=ahmedml_case_support_%A_%a.out
#SBATCH --error=ahmedml_case_support_%A_%a.err

set -euo pipefail

: "${AHMED_SUPPORT_PYTHON:?absolute Python interpreter is required}"
: "${AHMED_SUPPORT_REPO:?absolute fluidsbench-submission checkout is required}"
: "${AHMED_SUPPORT_DATASET:?absolute AhmedML dataset root is required}"
: "${AHMED_SUPPORT_OUTPUT:?absolute isolated support output root is required}"

for value in \
  "${AHMED_SUPPORT_PYTHON}" \
  "${AHMED_SUPPORT_REPO}" \
  "${AHMED_SUPPORT_DATASET}" \
  "${AHMED_SUPPORT_OUTPUT}"; do
  if [[ "${value}" != /* ]]; then
    echo "AhmedML support paths must be absolute: ${value}" >&2
    exit 64
  fi
done

case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${case_index}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
  exit 64
fi
case_id="$(
  "${AHMED_SUPPORT_PYTHON}" \
    "${AHMED_SUPPORT_REPO}/scripts/list_ahmedml_support_cases.py" \
    --index "${case_index}"
)"
if [[ ! "${case_id}" =~ ^run_[1-9][0-9]*$ ]]; then
  echo "No canonical case ID at zero-based position ${case_index}" >&2
  exit 64
fi

if [[ -f "${AHMED_SUPPORT_OUTPUT}/${case_id}/case-support.json" ]]; then
  echo "AhmedML support already exists for ${case_id}; leaving it unchanged."
  exit 0
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  "${AHMED_SUPPORT_PYTHON}" \
  "${AHMED_SUPPORT_REPO}/scripts/build_ahmedml_case_support.py" \
  --case-id "${case_id}" \
  --dataset-root "${AHMED_SUPPORT_DATASET}" \
  --output-root "${AHMED_SUPPORT_OUTPUT}"
