#!/bin/bash
#SBATCH --job-name=hl_truth_asm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=hilift_truth_assemble_%j.out
#SBATCH --error=hilift_truth_assemble_%j.err

set -euo pipefail

: "${HILIFT_TRUTH_EXPORT_APPROVED:?owner approval sentinel is required}"
: "${HILIFT_TRUTH_PYTHON:?absolute Python interpreter is required}"
: "${HILIFT_TRUTH_REPO:?absolute FluidsBench repository path is required}"
: "${HILIFT_TRUTH_AUTHORITY_INDEX:?absolute authority index is required}"
: "${HILIFT_TRUTH_AUTHORITY_SHA256:?pinned authority SHA-256 is required}"
: "${HILIFT_TRUTH_OUTPUT_DIR:?absolute candidate output directory is required}"

if [[ "${HILIFT_TRUTH_EXPORT_APPROVED}" != "YES_OWNER_APPROVED_1355" ]]; then
  echo "Refusing truth assembly: owner approval sentinel differs" >&2
  exit 64
fi

for path in \
  "${HILIFT_TRUTH_PYTHON}" \
  "${HILIFT_TRUTH_REPO}" \
  "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  "${HILIFT_TRUTH_OUTPUT_DIR}"; do
  if [[ "${path}" != /* ]]; then
    echo "Refusing non-absolute campaign path: ${path}" >&2
    exit 64
  fi
done

observed_sha256="$(sha256sum "${HILIFT_TRUTH_AUTHORITY_INDEX}" | cut -d ' ' -f 1)"
if [[ "${observed_sha256}" != "${HILIFT_TRUTH_AUTHORITY_SHA256}" ]]; then
  echo "Authority-index SHA-256 differs" >&2
  exit 65
fi

mode_args=()
if [[ "${HILIFT_TRUTH_CHECK:-0}" == "1" ]]; then
  mode_args+=(--check)
elif [[ "${HILIFT_TRUTH_CHECK:-0}" != "0" ]]; then
  echo "HILIFT_TRUTH_CHECK must be 0 or 1" >&2
  exit 64
fi

"${HILIFT_TRUTH_PYTHON}" \
  "${HILIFT_TRUTH_REPO}/scripts/export_hiliftaeroml_native_profile_truth.py" \
  --mode prerequisite-assemble \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  --output-dir "${HILIFT_TRUTH_OUTPUT_DIR}" \
  "${mode_args[@]}"
