# AhmedML and WindsorML submission workflow

Use the current release checkout. Both datasets now have the same commands
for preparing inputs, checking a complete entry, evaluating with recovery,
creating a case report and packaging a submission. Both datasets are open for submissions.

## Install and describe your entry

On Linux, use Python **3.12.13** and the pinned participant environment:

```bash
python3.12 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install -r requirements-native-participant.txt
dataset=ahmedml  # Use windsorml for WindsorML.
python scripts/$dataset.py --help
mkdir -p my-entry
cp examples/$dataset-v3-candidate/entry.template.json my-entry/entry.json
```

Choose a new submission ID and one prescribed split in `entry.json`. Both
datasets require `surface_and_volume` predictions. Custom case lists, unofficial
split names, directly predicted force coefficients and separate force files are
rejected. Forces and profiles are derived from the submitted native fields.

Produce native-order prediction manifests and NPZ chunks for **every** test case:

```text
my-entry/
  entry.json
  cases/
    run_N/
      surface/manifest.json
      surface/chunks/...
      volume/manifest.json
      volume/chunks/...
```

Use the exact ordered case list in `benchmark-specs/<dataset>/splits/<split>.json`.
AhmedML Full has 50 cases. WindsorML Full has 35 published cases; do not add a
replacement for the unavailable `run_354`. All eight prescribed split labels are
supported for each dataset. Use the dataset's existing prediction format:
[AhmedML](../examples/ahmedml-v3-candidate/README.md#inputs-and-blockers) or
[WindsorML](../examples/windsorml-v3-candidate/README.md#2-predict-and-evaluate-every-required-case).
AhmedML surface IDs index cells; WindsorML surface IDs index points. Each chunk
may contain at most 1,000,000 native rows. Paths in a manifest are relative to it.

## Prepare data and support

Use the same split as the entry. The offline download plan reports the exact
files and bytes before starting a potentially large native-data download:

```bash
python scripts/$dataset.py fetch-data --split-id full \
  --destination /data/$dataset --dry-run
python scripts/$dataset.py fetch-data --split-id full \
  --destination /data/$dataset
python scripts/$dataset.py fetch-support --split-id full \
  --destination support/$dataset
```

`fetch-data` uses the immutable dataset revision and selects each case's boundary,
published area weights, volume, geometry parameters and force CSV. Existing exact
files can be reused. Model training or inference may need additional files.

AhmedML support comes from the checksum-verified public prepared-cache release.
WindsorML profile support is already bundled and hash-pinned in the checkout;
its published surface dual areas arrive with `fetch-data`. Neither dataset asks
participants to generate evaluator support. Both `fetch-support` commands also
retain the small global profile-truth manifest needed for packaging.

For an offline node, copy the checkout, selected public data and complete support
directory from a connected machine. `fetch-support --offline` verifies reuse.
Add `--offline` to the following validation and evaluation commands to forbid
support downloads. Packaging uses the manifest retained with the evaluation.

## Validate and evaluate

```bash
python scripts/$dataset.py validate-entry my-entry \
  --dataset-root /data/$dataset --support-root support/$dataset
python scripts/$dataset.py evaluate-entry my-entry \
  --dataset-root /data/$dataset --support-root support/$dataset \
  --output work/my-result --resume
```

Preflight checks exact test membership, manifest association and native row
counts, required source-file sizes and all selected support. Per-case evaluation
checks source/prediction hashes, NPZ values and coverage. The evaluator streams
the large native data; no full VTK volume load is needed for participant scoring.

Repeat the same command after interruption. A completed case is saved atomically
with a receipt. Resume verifies prediction bytes, evidence and the evaluator
before reuse; unfinished cases and reductions are retried automatically. Changed
predictions, code or contracts require a new output directory. Concurrent runs
cannot write the same campaign. Do not hand-edit saved evidence or receipts.

The complete split reduction and retained per-case evidence are in
`work/my-result/evaluation/`. To view a standalone report, select an evaluated case:

```bash
python scripts/$dataset.py report --evaluation-root work/my-result/evaluation \
  --case-id run_N --output work/run_N.html
```

The table shows complete-split metrics; plots show the selected case's truth and
predictions. WindsorML relative profiles remain report-only. The commands retain
the existing scientific definitions, pressure conventions, weights and case sets.

## Package and submit

Complete `examples/<dataset>-v3-candidate/package-config.template.json` with the
actual method, compute and checkpoint records, saving it as
`my-entry/package-config.json`. Its split and submission ID must match `entry.json`.
Provide the actual per-case spatial records as `my-entry/discretization-cases.jsonl`,
following the dataset's existing packaging guide. Keep the fixed release bindings.

```bash
python scripts/$dataset.py package --evaluation-root work/my-result/evaluation \
  --config my-entry/package-config.json \
  --discretization-cases my-entry/discretization-cases.jsonl \
  --checkpoint YOUR_CHECKPOINT_ID=/path/to/model.pt \
  --output submissions/$dataset/YOUR_SUBMISSION_ID
python scripts/$dataset.py verify-package submissions/$dataset/YOUR_SUBMISSION_ID
```

Repeat `--checkpoint ID=PATH` for each declared model checkpoint. Packaging checks
the actual checkpoint bytes, re-reduces the complete case evidence and validates
the assembled result package before publishing its output directory. A failed package
attempt can be repeated without cleaning a partial output. `verify-package` also
accepts an external package directory. Neither command registers or approves it.

Commit only `submissions/<dataset>/<new-submission-id>/` and open a PR against
**`dev`**. The normal contributor checks apply. After review, maintainer approval
and feed generation publish the result on the leaderboard.
Those are publication steps; the submitted package needs no manual repairs.

## Pinned Linux container

```bash
docker build -f Dockerfile.native-participant -t fluidsbench-native .
docker run --rm fluidsbench-native scripts/ahmedml.py --help
docker run --rm fluidsbench-native scripts/windsorml.py --help
docker run --rm -v "$PWD:/work" -v /path/to/data:/data:ro -w /work \
  fluidsbench-native scripts/$dataset.py evaluate-entry my-entry \
  --dataset-root /data/$dataset --support-root support/$dataset \
  --output work/my-result --resume --offline
```

The image pins Python 3.12.13, NumPy 2.2.6 and VTK 9.5.2. Use the same image and
mounted checkout for evaluation, resume and packaging. Mount all checkpoint and
output paths used by the commands; mount data writable when downloading. Use a
normal Git clone with history for packaging. On shared machines, Docker's
`--user "$(id -u):$(id -g)"` option preserves output ownership.
