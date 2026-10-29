#!/bin/bash
#SBATCH --job-name=hl_truth_fast
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=hilift_truth_fast_%j.out
#SBATCH --error=hilift_truth_fast_%j.err

set -euo pipefail

: "${HILIFT_TRUTH_FAST_APPROVED:?fast truth approval sentinel is required}"
: "${HILIFT_TRUTH_PYTHON:?absolute Python interpreter is required}"
: "${HILIFT_TRUTH_REPO:?absolute FluidsBench repository path is required}"
: "${HILIFT_TRUTH_AUTHORITY_INDEX:?absolute authority index is required}"
: "${HILIFT_TRUTH_AUTHORITY_SHA256:?pinned authority SHA-256 is required}"
: "${HILIFT_TRUTH_PREFLIGHT_RECEIPT:?absolute preflight receipt is required}"
: "${HILIFT_TRUTH_OUTPUT_DIR:?absolute isolated output directory is required}"
: "${HILIFT_TRUTH_VALIDATION_RECEIPT:?absolute validation receipt is required}"

if [[ "${HILIFT_TRUTH_FAST_APPROVED}" != "YES_OWNER_APPROVED_FAST_PROFILE_TRUTH_1355" ]]; then
  echo "Refusing fast truth campaign: approval sentinel differs" >&2
  exit 64
fi
for path in \
  "${HILIFT_TRUTH_PYTHON}" \
  "${HILIFT_TRUTH_REPO}" \
  "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}" \
  "${HILIFT_TRUTH_OUTPUT_DIR}" \
  "${HILIFT_TRUTH_VALIDATION_RECEIPT}"; do
  if [[ "${path}" != /* ]]; then
    echo "Refusing non-absolute campaign path: ${path}" >&2
    exit 64
  fi
done
if [[ ! -x "${HILIFT_TRUTH_PYTHON}" ]]; then
  echo "Python interpreter is not executable" >&2
  exit 66
fi
if [[ ! -f "${HILIFT_TRUTH_AUTHORITY_INDEX}" || -L "${HILIFT_TRUTH_AUTHORITY_INDEX}" ]]; then
  echo "Authority index is absent or a symlink" >&2
  exit 66
fi
if [[ ! -f "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}" || -L "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}" ]]; then
  echo "Preflight receipt is absent or a symlink" >&2
  exit 66
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != "8" || "${SLURM_NTASKS:-}" != "8" || "${SLURM_CPUS_PER_TASK:-}" != "2" ]]; then
  echo "Refusing changed Slurm allocation; require 8 nodes, 8 tasks, and 2 CPUs/task" >&2
  exit 64
fi

dispatcher="${HILIFT_TRUTH_REPO}/scripts/hiliftaeroml_native_profile_truth_fast_dispatch.py"
exporter="${HILIFT_TRUTH_REPO}/scripts/export_hiliftaeroml_native_profile_truth.py"
validator="${HILIFT_TRUTH_REPO}/scripts/validate_hiliftaeroml_native_profile_truth.py"
for path in "${dispatcher}" "${exporter}" "${validator}"; do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "Required core truth script is absent or a symlink: ${path}" >&2
    exit 66
  fi
done

common_args=(
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}"
  --authority-sha256 "${HILIFT_TRUTH_AUTHORITY_SHA256}"
  --preflight-receipt "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}"
  --output-dir "${HILIFT_TRUTH_OUTPUT_DIR}"
  --validation-receipt "${HILIFT_TRUTH_VALIDATION_RECEIPT}"
)

"${HILIFT_TRUTH_PYTHON}" "${dispatcher}" --mode preflight "${common_args[@]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun \
  --nodes=8 \
  --ntasks=8 \
  --ntasks-per-node=1 \
  --cpus-per-task=2 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  "${HILIFT_TRUTH_PYTHON}" "${dispatcher}" \
  --mode generate "${common_args[@]}"

"${HILIFT_TRUTH_PYTHON}" "${exporter}" \
  --mode prerequisite-assemble \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  --output-dir "${HILIFT_TRUTH_OUTPUT_DIR}"

srun \
  --nodes=8 \
  --ntasks=8 \
  --ntasks-per-node=1 \
  --cpus-per-task=2 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  "${HILIFT_TRUTH_PYTHON}" "${dispatcher}" \
  --mode check "${common_args[@]}"

"${HILIFT_TRUTH_PYTHON}" "${exporter}" \
  --mode prerequisite-assemble \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  --output-dir "${HILIFT_TRUTH_OUTPUT_DIR}" \
  --check

"${HILIFT_TRUTH_PYTHON}" "${validator}" \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  --output-dir "${HILIFT_TRUTH_OUTPUT_DIR}" \
  --receipt "${HILIFT_TRUTH_VALIDATION_RECEIPT}"

"${HILIFT_TRUTH_PYTHON}" "${dispatcher}" --mode receipt "${common_args[@]}"
