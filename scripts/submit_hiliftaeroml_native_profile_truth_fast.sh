#!/bin/bash
set -euo pipefail

mode="${1:---dry-run}"
if [[ "${mode}" != "--dry-run" && "${mode}" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 64
fi

: "${HILIFT_TRUTH_FAST_APPROVED:?fast truth approval sentinel is required}"
: "${HILIFT_TRUTH_ACCOUNT:?Slurm account is required}"
: "${HILIFT_TRUTH_PYTHON:?absolute Python interpreter is required}"
: "${HILIFT_TRUTH_REPO:?absolute FluidsBench repository path is required}"
: "${HILIFT_TRUTH_AUTHORITY_INDEX:?absolute authority index is required}"
: "${HILIFT_TRUTH_AUTHORITY_SHA256:?pinned authority SHA-256 is required}"
: "${HILIFT_TRUTH_PREFLIGHT_RECEIPT:?absolute preflight receipt is required}"
: "${HILIFT_TRUTH_OUTPUT_DIR:?absolute isolated output directory is required}"
: "${HILIFT_TRUTH_VALIDATION_RECEIPT:?absolute validation receipt is required}"
: "${HILIFT_TRUTH_LOG_DIR:?absolute existing log directory is required}"

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
  "${HILIFT_TRUTH_VALIDATION_RECEIPT}" \
  "${HILIFT_TRUTH_LOG_DIR}"; do
  if [[ "${path}" != /* || "${path}" == *,* ]]; then
    echo "Refusing non-absolute or comma-bearing campaign path: ${path}" >&2
    exit 64
  fi
done
if [[ ! -d "${HILIFT_TRUTH_LOG_DIR}" || -L "${HILIFT_TRUTH_LOG_DIR}" ]]; then
  echo "Log directory is absent or a symlink" >&2
  exit 66
fi

dispatcher="${HILIFT_TRUTH_REPO}/scripts/hiliftaeroml_native_profile_truth_fast_dispatch.py"
batch_script="${HILIFT_TRUTH_REPO}/scripts/run_hiliftaeroml_native_profile_truth_fast.sbatch"
"${HILIFT_TRUTH_PYTHON}" "${dispatcher}" \
  --mode preflight \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  --authority-sha256 "${HILIFT_TRUTH_AUTHORITY_SHA256}" \
  --preflight-receipt "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}" \
  --output-dir "${HILIFT_TRUTH_OUTPUT_DIR}" \
  --validation-receipt "${HILIFT_TRUTH_VALIDATION_RECEIPT}"

export_values="ALL,HILIFT_TRUTH_FAST_APPROVED=${HILIFT_TRUTH_FAST_APPROVED},HILIFT_TRUTH_PYTHON=${HILIFT_TRUTH_PYTHON},HILIFT_TRUTH_REPO=${HILIFT_TRUTH_REPO},HILIFT_TRUTH_AUTHORITY_INDEX=${HILIFT_TRUTH_AUTHORITY_INDEX},HILIFT_TRUTH_AUTHORITY_SHA256=${HILIFT_TRUTH_AUTHORITY_SHA256},HILIFT_TRUTH_PREFLIGHT_RECEIPT=${HILIFT_TRUTH_PREFLIGHT_RECEIPT},HILIFT_TRUTH_OUTPUT_DIR=${HILIFT_TRUTH_OUTPUT_DIR},HILIFT_TRUTH_VALIDATION_RECEIPT=${HILIFT_TRUTH_VALIDATION_RECEIPT}"
sbatch_args=(
  --account="${HILIFT_TRUTH_ACCOUNT}"
  --chdir="${HILIFT_TRUTH_LOG_DIR}"
  --export="${export_values}"
)

if [[ "${mode}" == "--dry-run" ]]; then
  echo "DRY RUN: Slurm test-only; no job will be submitted"
  sbatch --test-only "${sbatch_args[@]}" "${batch_script}"
  exit 0
fi

if [[ "${HILIFT_TRUTH_FAST_SUBMIT_APPROVED:-}" != "YES_OWNER_APPROVED_SUBMIT_FAST_PROFILE_TRUTH_1355" ]]; then
  echo "Refusing submission: second submit approval sentinel differs" >&2
  exit 64
fi
sbatch --parsable "${sbatch_args[@]}" "${batch_script}"
