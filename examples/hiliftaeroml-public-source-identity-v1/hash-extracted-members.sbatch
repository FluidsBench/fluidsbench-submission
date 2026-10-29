#!/bin/bash
#SBATCH --job-name=hl_srcsha_opt
#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --array=0-1023%2
#SBATCH --output=hilift_source_hash_%A_%a.out
#SBATCH --error=hilift_source_hash_%A_%a.err

set -euo pipefail

: "${HILIFT_SOURCE_HASH_CAMPAIGN_APPROVED:?approval sentinel is required}"
: "${HILIFT_SOURCE_HASH_TOOL:?absolute CLI path is required}"
: "${HILIFT_SOURCE_INVENTORY:?absolute inventory path is required}"
: "${HILIFT_SOURCE_INVENTORY_SHA256:?pinned inventory SHA-256 is required}"
: "${HILIFT_EXTRACTED_DATASET_ROOT:?absolute extracted dataset root is required}"
: "${HILIFT_SOURCE_RECEIPT_DIR:?absolute receipt directory is required}"

if [[ "${HILIFT_SOURCE_HASH_CAMPAIGN_APPROVED}" != "YES_RECOVERY_CONFIRMED_IDLE" ]]; then
  echo "Refusing optional hash campaign: recovery-idle approval is absent" >&2
  exit 64
fi

if [[ "${HILIFT_SOURCE_HASH_TOOL}" != /* ]] || \
   [[ "${HILIFT_SOURCE_INVENTORY}" != /* ]] || \
   [[ "${HILIFT_EXTRACTED_DATASET_ROOT}" != /* ]] || \
   [[ "${HILIFT_SOURCE_RECEIPT_DIR}" != /* ]]; then
  echo "Refusing non-absolute campaign path" >&2
  exit 64
fi

python "${HILIFT_SOURCE_HASH_TOOL}" hash-extracted-case \
  --inventory "${HILIFT_SOURCE_INVENTORY}" \
  --expected-inventory-sha256 "${HILIFT_SOURCE_INVENTORY_SHA256}" \
  --dataset-root "${HILIFT_EXTRACTED_DATASET_ROOT}" \
  --domain surface \
  --receipt-dir "${HILIFT_SOURCE_RECEIPT_DIR}"
