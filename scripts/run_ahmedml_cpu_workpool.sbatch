#!/bin/bash
#SBATCH --job-name=ahmed_workpool
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=ahmedml_workpool_%A_%a.out
#SBATCH --error=ahmedml_workpool_%A_%a.err

set -euo pipefail

: "${AHMED_WORKPOOL_PYTHON:?absolute Python interpreter is required}"
: "${AHMED_WORKPOOL_REPO:?absolute fluidsbench-submission checkout is required}"
: "${AHMED_WORKPOOL_DATASET:?absolute AhmedML dataset root is required}"
: "${AHMED_WORKPOOL_SUPPORT:?absolute AhmedML case-support root is required}"
: "${AHMED_WORKPOOL_PREDICTIONS:?absolute isolated prediction root is required}"
: "${AHMED_WORKPOOL_EVIDENCE:?absolute isolated evidence root is required}"
: "${AHMED_WORKPOOL_CHECKPOINT:?absolute audited checkpoint path is required}"
: "${AHMED_WORKPOOL_SUPPORT_START:?first zero-based support index is required}"
: "${AHMED_WORKPOOL_SUPPORT_STOP:?last zero-based support index is required}"

for value in \
  "${AHMED_WORKPOOL_PYTHON}" \
  "${AHMED_WORKPOOL_REPO}" \
  "${AHMED_WORKPOOL_DATASET}" \
  "${AHMED_WORKPOOL_SUPPORT}" \
  "${AHMED_WORKPOOL_PREDICTIONS}" \
  "${AHMED_WORKPOOL_EVIDENCE}" \
  "${AHMED_WORKPOOL_CHECKPOINT}"; do
  if [[ "${value}" != /* ]]; then
    echo "AhmedML work-pool paths must be absolute: ${value}" >&2
    exit 64
  fi
done

workers="${AHMED_WORKPOOL_WORKERS:-${SLURM_CPUS_PER_TASK}}"
phases="${AHMED_WORKPOOL_PHASES:-both}"
shard_id="${SLURM_ARRAY_TASK_ID:-0}"
shard_count="${AHMED_WORKPOOL_SHARDS:-1}"
for value in \
  "${workers}" \
  "${shard_id}" \
  "${shard_count}" \
  "${AHMED_WORKPOOL_SUPPORT_START}" \
  "${AHMED_WORKPOOL_SUPPORT_STOP}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "AhmedML work-pool indices and counts must be non-negative integers: ${value}" >&2
    exit 64
  fi
done
if (( workers < 1 || workers > SLURM_CPUS_PER_TASK )); then
  echo "AHMED_WORKPOOL_WORKERS must be in [1, ${SLURM_CPUS_PER_TASK}]" >&2
  exit 64
fi
if (( shard_count < 1 || shard_id >= shard_count )); then
  echo "Invalid shard ${shard_id}/${shard_count}" >&2
  exit 64
fi
if (( AHMED_WORKPOOL_SUPPORT_START > AHMED_WORKPOOL_SUPPORT_STOP )); then
  echo "Support start must not exceed support stop" >&2
  exit 64
fi
if [[ "${phases}" != "both" && "${phases}" != "support" && "${phases}" != "fixture" ]]; then
  echo "AHMED_WORKPOOL_PHASES must be one of: both, support, fixture" >&2
  exit 64
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case_id_for_index() {
  "${AHMED_WORKPOOL_PYTHON}" \
    "${AHMED_WORKPOOL_REPO}/scripts/list_ahmedml_support_cases.py" \
    --index "$1"
}

run_support_case() {
  local case_index="$1"
  local case_id
  case_id="$(case_id_for_index "${case_index}")"
  if [[ -f "${AHMED_WORKPOOL_SUPPORT}/${case_id}/case-support.json" ]]; then
    echo "Support already exists for ${case_id}; leaving it unchanged."
    return 0
  fi
  "${AHMED_WORKPOOL_PYTHON}" \
    "${AHMED_WORKPOOL_REPO}/scripts/build_ahmedml_case_support.py" \
    --case-id "${case_id}" \
    --dataset-root "${AHMED_WORKPOOL_DATASET}" \
    --output-root "${AHMED_WORKPOOL_SUPPORT}"
}

run_fixture_case() {
  local case_index="$1"
  local case_id
  case_id="$(case_id_for_index "${case_index}")"
  local support_manifest="${AHMED_WORKPOOL_SUPPORT}/${case_id}/case-support.json"
  local prediction_root="${AHMED_WORKPOOL_PREDICTIONS}/${case_id}"
  local evidence_path="${AHMED_WORKPOOL_EVIDENCE}/${case_id}.json"
  if [[ ! -f "${support_manifest}" ]]; then
    echo "Missing evaluator-owned support for ${case_id}: ${support_manifest}" >&2
    return 66
  fi
  if [[ ! -f "${prediction_root}/fixture-provenance.json" ]]; then
    "${AHMED_WORKPOOL_PYTHON}" \
      "${AHMED_WORKPOOL_REPO}/scripts/create_ahmedml_geotransolver_calibrated_fixture.py" \
      --case-id "${case_id}" \
      --dataset-root "${AHMED_WORKPOOL_DATASET}" \
      --output-root "${AHMED_WORKPOOL_PREDICTIONS}" \
      --checkpoint "${AHMED_WORKPOOL_CHECKPOINT}"
  else
    echo "Prediction fixture already exists for ${case_id}; leaving it unchanged."
  fi
  if [[ ! -f "${evidence_path}" ]]; then
    mkdir -p "${AHMED_WORKPOOL_EVIDENCE}"
    "${AHMED_WORKPOOL_PYTHON}" \
      "${AHMED_WORKPOOL_REPO}/scripts/evaluate_ahmedml_candidate_case.py" \
      --case-id "${case_id}" \
      --dataset-root "${AHMED_WORKPOOL_DATASET}" \
      --case-support "${support_manifest}" \
      --surface-prediction-manifest "${prediction_root}/surface/manifest.json" \
      --volume-prediction-manifest "${prediction_root}/volume/manifest.json" \
      --output "${evidence_path}"
  else
    echo "Evaluator evidence already exists for ${case_id}; leaving it unchanged."
  fi
}

run_bounded_pool() {
  local phase="$1"
  local first="$2"
  local last="$3"
  local -a pids=()
  local failures=0
  local index
  local pid
  for ((index = first + shard_id; index <= last; index += shard_count)); do
    if [[ "${phase}" == "support" ]]; then
      run_support_case "${index}" &
    else
      run_fixture_case "${index}" &
    fi
    pids+=("$!")
    if (( ${#pids[@]} >= workers )); then
      pid="${pids[0]}"
      if ! wait "${pid}"; then
        failures=1
      fi
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=1
    fi
  done
  return "${failures}"
}

if [[ "${phases}" == "both" || "${phases}" == "support" ]]; then
  run_bounded_pool \
    support \
    "${AHMED_WORKPOOL_SUPPORT_START}" \
    "${AHMED_WORKPOOL_SUPPORT_STOP}"
fi

# The non-ranked full-split development fixture uses canonical indices 0--49.
if [[ "${phases}" == "both" || "${phases}" == "fixture" ]]; then
  run_bounded_pool fixture 0 49
fi
