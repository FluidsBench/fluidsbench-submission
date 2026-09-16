#!/bin/bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mode="${1:---dry-run}"
if [[ "${mode}" != "--dry-run" && "${mode}" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 64
fi

: "${HILIFT_MOMENT_ACCOUNT:?Slurm account is required}"
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
: "${HILIFT_MOMENT_LOG_DIR:?absolute existing log directory is required}"

if [[ "${HILIFT_MOMENT_AUDIT_APPROVED}" != "YES_OWNER_APPROVED_EXACT_MOMENT_AUDIT_ALL1800" ]]; then
  echo "Refusing full audit: approval sentinel differs" >&2
  exit 64
fi
if [[ ! -d "${HILIFT_MOMENT_LOG_DIR}" || -L "${HILIFT_MOMENT_LOG_DIR}" ]]; then
  echo "Campaign log directory is absent or a symlink" >&2
  exit 66
fi

driver="${HILIFT_MOMENT_REPO}/scripts/audit_hiliftaeroml_exact_moments.py"
batch="${HILIFT_MOMENT_REPO}/scripts/run_hiliftaeroml_exact_moment_full.sbatch"
if [[ "${HILIFT_MOMENT_OUTPUT_ROOT}" != /* || "${HILIFT_MOMENT_DATASET}" != /* ]]; then
  echo "Output and dataset roots must be absolute before local preflight" >&2
  exit 66
fi
if [[ ! -d "${HILIFT_MOMENT_OUTPUT_ROOT}" || -L "${HILIFT_MOMENT_OUTPUT_ROOT}" ]]; then
  echo "Output root is absent or a symlink" >&2
  exit 66
fi
if [[ ! -d "${HILIFT_MOMENT_DATASET}" || -L "${HILIFT_MOMENT_DATASET}" ]]; then
  echo "Dataset root is absent or a symlink" >&2
  exit 66
fi
output_root_real="$(realpath -e -- "${HILIFT_MOMENT_OUTPUT_ROOT}")"
dataset_root_real="$(realpath -e -- "${HILIFT_MOMENT_DATASET}")"
if [[ "${output_root_real}" != "${HILIFT_MOMENT_OUTPUT_ROOT}" || "${dataset_root_real}" != "${HILIFT_MOMENT_DATASET}" ]]; then
  echo "Output and dataset roots must be canonical paths without symlinked components" >&2
  exit 66
fi
case "${output_root_real}/" in
  "${dataset_root_real}/"*)
    echo "Output root must be outside the source dataset" >&2
    exit 66
    ;;
esac
local_cache_parent="${output_root_real}/numba-cache"
local_preflight_cache="${local_cache_parent}/local-preflight"
for cache_directory in "${local_cache_parent}" "${local_preflight_cache}"; do
  if [[ -L "${cache_directory}" || ( -e "${cache_directory}" && ! -d "${cache_directory}" ) ]]; then
    echo "Local preflight cache path is a symlink or non-directory: ${cache_directory}" >&2
    exit 66
  fi
  if [[ ! -e "${cache_directory}" ]]; then
    mkdir -- "${cache_directory}"
  fi
  if [[ ! -d "${cache_directory}" || -L "${cache_directory}" ]]; then
    echo "Local preflight cache path is unavailable or a symlink: ${cache_directory}" >&2
    exit 66
  fi
done
NUMBA_CACHE_DIR="${local_preflight_cache}" \
HILIFT_EXCLUDE_SYSTEM_DIST_PACKAGES=1 \
"${HILIFT_MOMENT_PYTHON}" "${driver}" \
  --mode preflight \
  --dataset-root "${HILIFT_MOMENT_DATASET}" \
  --recipe-root "${HILIFT_MOMENT_RECIPE}" \
  --output-root "${HILIFT_MOMENT_OUTPUT_ROOT}" \
  --all1800-inventory "${HILIFT_MOMENT_ALL1800_INVENTORY}" \
  --scope full \
  --cell-chunk "${HILIFT_MOMENT_CELL_CHUNK}"

export_values="ALL,HILIFT_MOMENT_ACCOUNT=${HILIFT_MOMENT_ACCOUNT},HILIFT_MOMENT_AUDIT_APPROVED=${HILIFT_MOMENT_AUDIT_APPROVED},HILIFT_MOMENT_PYTHON=${HILIFT_MOMENT_PYTHON},HILIFT_MOMENT_REPO=${HILIFT_MOMENT_REPO},HILIFT_MOMENT_RECIPE=${HILIFT_MOMENT_RECIPE},HILIFT_MOMENT_DATASET=${HILIFT_MOMENT_DATASET},HILIFT_MOMENT_OUTPUT_ROOT=${HILIFT_MOMENT_OUTPUT_ROOT},HILIFT_MOMENT_ALL1800_INVENTORY=${HILIFT_MOMENT_ALL1800_INVENTORY},HILIFT_MOMENT_AGGREGATE=${HILIFT_MOMENT_AGGREGATE},HILIFT_MOMENT_VALIDATION=${HILIFT_MOMENT_VALIDATION},HILIFT_MOMENT_CELL_CHUNK=${HILIFT_MOMENT_CELL_CHUNK}"
sbatch_args=(
  --account="${HILIFT_MOMENT_ACCOUNT}"
  --chdir="${HILIFT_MOMENT_LOG_DIR}"
  --export="${export_values}"
)

if [[ "${mode}" == "--dry-run" ]]; then
  echo "DRY RUN: Slurm test-only; no full campaign will be submitted"
  sbatch --test-only "${sbatch_args[@]}" "${batch}"
  exit 0
fi
if [[ "${HILIFT_MOMENT_FULL_SUBMIT_APPROVED:-}" != "YES_OWNER_APPROVED_SUBMIT_EXACT_MOMENT_ALL1800" ]]; then
  echo "Refusing submission: second full-campaign sentinel differs" >&2
  exit 64
fi
sbatch --parsable "${sbatch_args[@]}" "${batch}"
