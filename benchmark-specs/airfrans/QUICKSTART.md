# AirfRANS submission guide

Use the current release checkout. Supported splits are **Full** (`full`, 200 test
cases), **Scarce** (`scarce`, the same 200 test cases but a different training
regime), and **AoA extrapolation** (`aoa_extrapolation`, 196 test cases).
Reynolds extrapolation and custom case lists are rejected. Follow [Submitting a result](../../SUBMITTING.md) for validation and review.

## Environment

The container installs the existing scientific runtime: AirfRANS 0.1.5.1,
NumPy 2.5.2, PyVista 0.48.4 and VTK 9.6.2. Extraction also checks the installed
AirfRANS source hash against the approved profile contract.

```bash
docker build -f Dockerfile.airfrans -t fluidsbench-airfrans .
```

Build from a clean checkout before downloading support or adding predictions.
Docker installations without BuildKit ignore `Dockerfile.airfrans.dockerignore`;
the current ignore file also omits `support/`. Rebuilding a populated checkout
can therefore copy large local artifacts into the image. Keep those artifacts
outside the build checkout and mount them at runtime if you need to rebuild.

For the commands below, either install `requirements-airfrans-participant.txt`
into a Python 3.12 environment, or run them inside the container. For example:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v /path/to/AirfRANS/Dataset:/data:ro \
  -w /work fluidsbench-airfrans scripts/airfrans.py --help
```

`--user "$(id -u):$(id -g)"` runs the container as your own user. Without it
the container runs as root, and on Linux every file it writes into your checkout
(support, evaluation outputs, the final package) would belong to root and could
only be changed or deleted with `sudo`.

Inside the container use `/data` for `--dataset-root` and keep entries, support
and outputs under `/work` so they persist. These working directories
(`support/`, `work/`, `my-entry/`) are ignored by git; commit only the final
`submissions/airfrans/<submission_id>/` directory. Public source/weight sharing
is optional. Checkpoint hashes and truthful method/compute records are required.

## Prepare the reusable reference support

```bash
python scripts/airfrans.py fetch-support --destination support/airfrans --dry-run
python scripts/airfrans.py fetch-support --destination support/airfrans
```

This downloads about **1.53 GB** of approved archives, verifies their exact
sizes and SHA-256 hashes, safely extracts them, and validates all 355 unique
reference cases. Archives remain cached alongside the installation; allow at
least 4 GB of disk space for support setup. The complete installation appears
atomically under `support/airfrans/installed/` after validation succeeds.
Participants never generate reference weights or ground truth.

Repeat the command to verify and reuse the cache. For an offline machine, copy
the support directory and run:

```bash
python scripts/airfrans.py fetch-support --destination support/airfrans --offline
```

Alternatively put the three published files (`support-metadata.tar.gz`,
`native-tables.tar`, `profile-truth.tar.gz`) in one directory and provide
`--archive-dir /path/to/archives --offline` for the first installation.
Archive downloads interrupted before verification are retried automatically.
Installed files are checksum checked before reuse; a changed cache is rejected.
The support release policy in [RELEASES.md](RELEASES.md) still applies.

## Supply meshes and native predictions

Use your existing original preprocessed AirfRANS `Dataset` directory. Each
test case needs `<case>/<case>_internal.vtu` and `<case>/<case>_aerofoil.vtp`.
The workflow verifies their hashes against the published reference extraction.
The prepared support does not replace these meshes: the existing AirfRANS
library needs their connectivity to derive wall shear, forces and profiles.

```bash
mkdir -p my-entry/predictions
cp examples/airfrans-participant/entry.template.json my-entry/entry.json
cp examples/airfrans-participant/package-config.template.json my-entry/package-config.json
```

Choose a new submission ID and the correct split in `entry.json`. Supply exactly
one `my-entry/predictions/<case_id>.npz` for every case in the ordered official
split, with these four arrays only:

| Array | Shape | Meaning |
|---|---|---|
| `velocity` | `(N, 2)` | Cartesian velocity in m/s at native internal points |
| `pressure` | `(N,)` or `(N, 1)` | Native kinematic pressure in m²/s² |
| `internal_points` | `(N, 2)` or `(N, 3)` | Native internal coordinates, in metres |
| `airfoil_points` | `(M, 2)` or `(M, 3)` | Native airfoil coordinates, in metres |

All arrays must be finite, denormalized and in the unchanged native point order.
Any upstream mapping must be disclosed in the spatial records; this workflow
does not infer or silently perform a mapping. A model-specific export example
is [pack_native_predictions.py](../../examples/airfrans-v3-candidate/pack_native_predictions.py).
Direct force arrays, extra cases and incomplete test sets are rejected. Forces
and wall shear are derived from velocity/pressure through the existing AirfRANS
library; profiles use the existing approved extractor.

## Validate, evaluate and resume

```bash
python scripts/airfrans.py validate-entry my-entry \
  --dataset-root /path/to/AirfRANS/Dataset --support-root support/airfrans
python scripts/airfrans.py evaluate-entry my-entry \
  --dataset-root /path/to/AirfRANS/Dataset --support-root support/airfrans \
  --output work/my-result --resume
```

Validation checks the entire split, source mesh hashes, array shapes and finite
values before evaluation. Per-case evaluation also checks coordinate order
against the actual meshes. No support downloads occur during evaluation.
Each case's derived fields and profiles are saved together atomically. Native
meshes are processed in separate workers, one case at a time. The final reduction
uses the existing field/force and balanced profile scorers without changing their
weights or formulas.

If interrupted, repeat the identical `evaluate-entry ... --resume` command.
Completed cases are verified and reused; unfinished cases and final reductions
are retried. Changes to predictions, meshes, runtime, evaluator or saved results
are rejected. Use a new output directory for a changed campaign. Keep the entire
`work/my-result` directory through packaging, including its case checkpoints.
Derived prediction tables are retained for reproducibility; the final reduction
uses hard links within that directory to avoid a second copy of those tables.

An optional standalone report needs no network or original meshes:

```bash
python scripts/airfrans.py report --evaluation-root work/my-result/evaluation \
  --case-id <case_id_from_your_split> --output work/case-report.html
```

## Package and submit

Complete `my-entry/package-config.json` with your actual method, training,
checkpoint digests and measured inference time. Its submission ID and split
must match the evaluated entry. Release IDs and hashes are filled in automatically
from the approved evaluation contract.

Provide the actual `training` and `inference` objects in `spatial_discretization`
and one ordered record per test case in `my-entry/discretization-cases.jsonl`.
These follow [the spatial specification](../../docs/RESULT_FORMAT.md) and
[case schema](../../schemas/v3/discretization-case.schema.json).
The [generic discretization template](../../examples/v3-template/discretization.json)
and its [case-record layout](../../examples/v3-template/discretization/cases.jsonl)
show the required fields. Fill them with your model's actual inference and mapping details.

```bash
python scripts/airfrans.py package --evaluation-root work/my-result/evaluation \
  --config my-entry/package-config.json \
  --discretization-cases my-entry/discretization-cases.jsonl \
  --output submissions/airfrans/my-model-full-v1
python scripts/airfrans.py verify-package submissions/airfrans/my-model-full-v1
```

Optionally add `--checkpoint CHECKPOINT_ID=/path/to/weights` to `package` to
verify the recorded raw checkpoint digest locally. This does not copy or upload
the weights. Packaging verifies retained evaluation files and runs the real
contributor validator before exposing the output directory; failed validation
leaves no partial final package. Use this command for new submissions.

Open a PR to **`main`** containing only the new submission directory. Do not add
approval records. Maintainers review and approve the package before publication. A passing contributor check is not approval.
