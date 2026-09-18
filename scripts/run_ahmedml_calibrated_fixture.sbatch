#!/bin/bash
#SBATCH --job-name=ahmed_fixture
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=ahmedml_fixture_%A_%a.out
#SBATCH --error=ahmedml_fixture_%A_%a.err

set -euo pipefail

: "${AHMED_FIXTURE_PYTHON:?absolute Python interpreter is required}"
: "${AHMED_FIXTURE_REPO:?absolute fluidsbench-submission checkout is required}"
: "${AHMED_FIXTURE_DATASET:?absolute AhmedML dataset root is required}"
: "${AHMED_FIXTURE_SUPPORT:?absolute AhmedML case-support root is required}"
: "${AHMED_FIXTURE_PREDICTIONS:?absolute isolated prediction root is required}"
: "${AHMED_FIXTURE_EVIDENCE:?absolute isolated evidence root is required}"
: "${AHMED_FIXTURE_CHECKPOINT:?absolute audited checkpoint path is required}"

for value in \
  "${AHMED_FIXTURE_PYTHON}" \
  "${AHMED_FIXTURE_REPO}" \
  "${AHMED_FIXTURE_DATASET}" \
  "${AHMED_FIXTURE_SUPPORT}" \
  "${AHMED_FIXTURE_PREDICTIONS}" \
  "${AHMED_FIXTURE_EVIDENCE}" \
  "${AHMED_FIXTURE_CHECKPOINT}"; do
  if [[ "${value}" != /* ]]; then
    echo "AhmedML fixture paths must be absolute: ${value}" >&2
    exit 64
  fi
done

case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${case_index}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
  exit 64
fi
case_id="$(
  "${AHMED_FIXTURE_PYTHON}" \
    "${AHMED_FIXTURE_REPO}/scripts/list_ahmedml_support_cases.py" \
    --index "${case_index}"
)"
if [[ ! "${case_id}" =~ ^run_[1-9][0-9]*$ ]]; then
  echo "No canonical case ID at zero-based position ${case_index}" >&2
  exit 64
fi

support_manifest="${AHMED_FIXTURE_SUPPORT}/${case_id}/case-support.json"
prediction_root="${AHMED_FIXTURE_PREDICTIONS}/${case_id}"
evidence_path="${AHMED_FIXTURE_EVIDENCE}/${case_id}.json"
if [[ ! -f "${support_manifest}" ]]; then
  echo "Missing evaluator-owned support for ${case_id}: ${support_manifest}" >&2
  exit 66
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

if [[ ! -f "${prediction_root}/fixture-provenance.json" ]]; then
  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${AHMED_FIXTURE_PYTHON}" \
    "${AHMED_FIXTURE_REPO}/scripts/create_ahmedml_geotransolver_calibrated_fixture.py" \
    --case-id "${case_id}" \
    --dataset-root "${AHMED_FIXTURE_DATASET}" \
    --output-root "${AHMED_FIXTURE_PREDICTIONS}" \
    --checkpoint "${AHMED_FIXTURE_CHECKPOINT}"
else
  echo "Prediction fixture already exists for ${case_id}; leaving it unchanged."
fi

if [[ ! -f "${evidence_path}" ]]; then
  mkdir -p "${AHMED_FIXTURE_EVIDENCE}"
  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${AHMED_FIXTURE_PYTHON}" \
    "${AHMED_FIXTURE_REPO}/scripts/evaluate_ahmedml_candidate_case.py" \
    --case-id "${case_id}" \
    --dataset-root "${AHMED_FIXTURE_DATASET}" \
    --case-support "${support_manifest}" \
    --surface-prediction-manifest "${prediction_root}/surface/manifest.json" \
    --volume-prediction-manifest "${prediction_root}/volume/manifest.json" \
    --output "${evidence_path}"
else
  echo "Evaluator evidence already exists for ${case_id}; leaving it unchanged."
fi
