# DrivAerML submission guide

Use the current release checkout. The participant command prepares immutable support,
checks inputs, evaluates the whole official test split with recovery, and produces
a validated result directory. See the [dataset index](../README.md) for availability and
[Submitting a result](../../SUBMITTING.md) for validation and review.

## Install

On Linux, use Python 3.12.13:

```bash
python3.12 -m venv .venv-drivaerml
.venv-drivaerml/bin/pip install -r requirements-drivaerml-participant.txt
source .venv-drivaerml/bin/activate
python scripts/drivaerml.py --help
```

This pins NumPy 2.2.6 and VTK 9.5.2, and installs the `hf` download command.
Owner atlas/rendering dependencies are not needed. A Linux container is also
available; see the final section.

## Describe the entry and produce predictions

Copy `examples/drivaerml-v3-candidate/entry.template.json` to `my-entry/entry.json`.
Choose your submission ID, official split, and either `surface_and_volume` or
`surface_only`. The official splits are `full`, `medium`, `scarce`, `super_scarce`,
`geometry`, `high_drag`, `low_drag`, and `rear_separation`. The command takes the
exact test membership from the published split. Custom splits, custom case lists,
and directly predicted force coefficients are rejected. Forces always come from
integration of the predicted surface pressure and wall shear.

Write the existing [native NPZ prediction format and writer](../../docs/NATIVE_PREDICTIONS.md)
in this layout, for every test case:

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

For `surface_only`, omit the `volume/` directory. There must be no extra case
directories or direct-force files. The manifest paths for each NPZ are relative
to that manifest; each chunk contains at most 1,000,000 raw native cells.

## Prepare inputs

The commands below use `full` and `surface_and_volume`; use the same split and
scope as `entry.json`. Check the download plan first: full native volume data
is large. The plan reports exact files and total bytes without network access.
Existing local dataset files can be reused.

```bash
python scripts/drivaerml.py fetch-data --split-id full \
  --prediction-scope surface_and_volume --destination /data/drivaerml --dry-run
python scripts/drivaerml.py fetch-data --split-id full \
  --prediction-scope surface_and_volume --destination /data/drivaerml
python scripts/drivaerml.py fetch-support --split-id full \
  --destination support/drivaerml-native-v3
```

`fetch-data` uses the immutable dataset revision and downloads only the selected
scope's native inputs. The force-truth CSV is already retained in the repository.
Volume parts are streamed directly; do not concatenate a second large VTU copy.
`fetch-support` downloads SHA-256-verified published profile support and is safe
to rerun. Participants do not generate support or calculate an extra NPZ cache.

For an offline compute node, copy the selected dataset inputs and support
directory from an online machine. Add `--offline` to validation/evaluation;
missing support then fails before native evaluation starts.

## Validate and evaluate

```bash
python scripts/drivaerml.py validate-entry my-entry \
  --dataset-root /data/drivaerml --support-root support/drivaerml-native-v3
python scripts/drivaerml.py evaluate-entry my-entry \
  --dataset-root /data/drivaerml --support-root support/drivaerml-native-v3 \
  --output work/my-result --resume
```

Preflight checks complete split membership, manifest coverage, required file
sizes, and the pinned profile support. The evaluator checks hashes, NPZ arrays,
native row order and scientific evidence as each case runs. Full split reductions
and package-ready metrics/profiles appear in `work/my-result/evaluation/`.

If interrupted, repeat the same command. Completed core/profile pairs are saved
together. Resume verifies prediction bytes and checkpoint outputs, and refuses
reuse if predictions, evaluator code or scoring contracts changed. An interrupted
case or reduction is retried automatically. Changed inputs require a new output
directory. Completed results can also be checked with the same `--resume` command.

Cp and velocity profiles use normalized trapezoidal arc-length weighting, with
equal total weight per case and profile. Unsupported gaps are never bridged.
Relative profiles are report-only and do not contribute to the score.
See the [profile scoring contract](profile-scoring-v2.json) for the exact definitions.

An optional, standalone case report shows truth and predictions:

```bash
python scripts/drivaerml.py report --evaluation-root work/my-result/evaluation \
  --support-root support/drivaerml-native-v3 --case-id run_419 \
  --output work/run_419.html
```

Choose a case in your evaluated split. The table contains complete-split metrics;
the plots contain the selected case's profiles.

## Package and submit

Complete the participant/methodology fields in
`examples/drivaerml-v3-candidate/package-config.template.json`, save it as
`my-entry/package-config.json`, and provide your actual discretization case
records and checkpoint bytes as described in the [package guide](../../examples/drivaerml-v3-candidate/README.md).
The submission ID, scope and split must match the entry. Keep the supplied release
bindings. Evaluation does not invent model, training, inference or compute metadata.

```bash
python scripts/drivaerml.py package --evaluation-root work/my-result/evaluation \
  --config my-entry/package-config.json \
  --discretization-cases my-entry/discretization-cases.jsonl \
  --checkpoint YOUR_CHECKPOINT_ID=/path/to/checkpoint \
  --support-root support/drivaerml-native-v3 \
  --output submissions/drivaerml/YOUR_SUBMISSION_ID
python scripts/drivaerml.py verify-package submissions/drivaerml/YOUR_SUBMISSION_ID \
  --support-root support/drivaerml-native-v3
```

Repeat `--checkpoint ID=PATH` for every declared checkpoint. `package` checks the
checkpoint bytes, assembles the result and runs the contributor validator
plus the genuine-inference checks before publishing its output directory.
`verify-package` also accepts a package directory outside a Git checkout. It does
not register or approve it.

Commit only `submissions/drivaerml/YOUR_SUBMISSION_ID/` and open a PR against
`main`. Maintainers review and approve the result before leaderboard publication. The public JSON directory is the submission
payload; AutoCFD5 confidential ZIPs are not the FluidsBench intake format.

## Container

```bash
docker build -f Dockerfile.drivaerml -t fluidsbench-drivaerml .
docker run --rm fluidsbench-drivaerml --help
docker run --rm -v "$PWD:/work" -v /path/to/drivaerml:/data:ro -w /work \
  fluidsbench-drivaerml evaluate-entry /work/my-entry \
  --dataset-root /data --support-root /work/support/drivaerml-native-v3 \
  --output /work/work/my-result --resume --offline
```

The image supplies the pinned Linux runtime and the same participant command.
Use a normal Git clone (with its history) for the mounted repository when
packaging: the validator verifies that the frozen evaluator commits are real
and reachable. Run all container commands from the same mounted checkout.
Mount all required input/output locations; mount the dataset writable when using
`fetch-data`. On a shared machine, pass `--user "$(id -u):$(id -g)"` to Docker so
new output files belong to you. Use the same image for every resumed run.

Explore the [dataset leaderboard preview](https://fluidsbench.org/leaderboard-preview/?dataset=drivaerml),
which contains no submitted results. See [review and publication](../../SUBMITTING.md#see-the-leaderboard-and-follow-publication)
for how an approved result reaches an official release.
