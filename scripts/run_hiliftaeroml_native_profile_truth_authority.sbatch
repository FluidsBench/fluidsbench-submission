#!/bin/bash
#SBATCH --job-name=hl_truth_auth
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=hilift_truth_authority_%j.out
#SBATCH --error=hilift_truth_authority_%j.err

set -euo pipefail

: "${HILIFT_TRUTH_INDEX_APPROVED:?authority-index approval sentinel is required}"
: "${HILIFT_TRUTH_PYTHON:?absolute Python interpreter is required}"
: "${HILIFT_TRUTH_REPO:?absolute FluidsBench repository path is required}"
: "${HILIFT_TRUTH_AUTHORITY_INDEX:?absolute authority-index output is required}"
: "${HILIFT_TRUTH_PREFLIGHT_RECEIPT:?absolute preflight-receipt output is required}"
: "${HILIFT_TRUTH_CP_ROOTS:?colon-separated Cp prerequisite roots are required}"
: "${HILIFT_TRUTH_VELOCITY_ROOTS:?colon-separated velocity prerequisite roots are required}"
: "${HILIFT_TRUTH_VALIDITY_ROOTS:?colon-separated validity prerequisite roots are required}"

if [[ "${HILIFT_TRUTH_INDEX_APPROVED}" != "YES_OWNER_APPROVED_INDEX_ONLY" ]]; then
  echo "Refusing authority inventory: owner approval sentinel differs" >&2
  exit 64
fi

for path in \
  "${HILIFT_TRUTH_PYTHON}" \
  "${HILIFT_TRUTH_REPO}" \
  "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}"; do
  if [[ "${path}" != /* ]]; then
    echo "Refusing non-absolute campaign path: ${path}" >&2
    exit 64
  fi
done

IFS=: read -r -a cp_roots <<< "${HILIFT_TRUTH_CP_ROOTS}"
IFS=: read -r -a velocity_roots <<< "${HILIFT_TRUTH_VELOCITY_ROOTS}"
IFS=: read -r -a validity_roots <<< "${HILIFT_TRUTH_VALIDITY_ROOTS}"
root_args=()
for path in "${cp_roots[@]}"; do
  [[ "${path}" == /* ]] || { echo "Cp root is not absolute" >&2; exit 64; }
  root_args+=(--cp-record-root "${path}")
done
for path in "${velocity_roots[@]}"; do
  [[ "${path}" == /* ]] || { echo "Velocity root is not absolute" >&2; exit 64; }
  root_args+=(--velocity-record-root "${path}")
done
for path in "${validity_roots[@]}"; do
  [[ "${path}" == /* ]] || { echo "Validity root is not absolute" >&2; exit 64; }
  root_args+=(--validity-record-root "${path}")
done

"${HILIFT_TRUTH_PYTHON}" \
  "${HILIFT_TRUTH_REPO}/scripts/export_hiliftaeroml_native_profile_truth.py" \
  --mode prerequisite-index \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  "${root_args[@]}"

"${HILIFT_TRUTH_PYTHON}" \
  "${HILIFT_TRUTH_REPO}/scripts/export_hiliftaeroml_native_profile_truth.py" \
  --mode prerequisite-preflight \
  --authority-index "${HILIFT_TRUTH_AUTHORITY_INDEX}" \
  --preflight-receipt "${HILIFT_TRUTH_PREFLIGHT_RECEIPT}"

sha256sum "${HILIFT_TRUTH_AUTHORITY_INDEX}"
