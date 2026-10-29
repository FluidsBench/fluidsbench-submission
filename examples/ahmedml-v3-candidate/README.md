# AhmedML schema-v3 submission package

The [participant quickstart](../../benchmark-specs/ahmedml/QUICKSTART.md) provides
the complete resumable command sequence and pinned Linux container. The
per-case commands below remain available for existing pipelines.

This directory is a reusable configuration template, not a submission.
AhmedML is open for submissions.
The participant workflow creates an unapproved result package
for any of the eight official split labels only after all placeholder tokens,
checkpoint hashes, complete evaluator evidence, and spatial records agree.

The package requires an explicit attestation that every prediction was produced by actual model
inference and not generated from evaluation truth, and verifies the local bytes
of every checkpoint named by the methodology record.

<a id="inputs"></a>

Use the shared [hardware and compute fields](../../METHODOLOGY.md#hardware-and-compute)
for each training stage and inference campaign, including GPU model and allocation counts.

## First-time setup

Update your fork from upstream `dev`. Use Linux and Python 3.12 for native
evaluation; the native-array protections use Linux descriptor semantics.
Install the evaluator dependencies from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose the split you will submit and download its prepared evaluator support:

```bash
python scripts/fetch_ahmedml_support.py --split full --dry-run
python scripts/fetch_ahmedml_support.py --split full
```

Full, Medium, Scarce and Super-scarce share 50 test cases: approximately
1.74 GB to download and 6.62 GB installed. Other splits need approximately
3.45–3.48 GB to download and 12.61–13.59 GB installed. Shared cases are reused.
The default location is `~/.cache/fluidsbench/ahmedml/<scientific-manifest-sha256>/`
(or under `XDG_CACHE_HOME`). Each archive and extracted file is checked against
the frozen release. Interrupted runs can be repeated; verified cases are reused.
You do not build geometry support or obtain private maintainer files.

For HPC scratch or shared storage, use `--destination /scratch/ahmedml-support`
and pass `--support-root /scratch/ahmedml-support` to the evaluator, or set
`FLUIDSBENCH_AHMEDML_SUPPORT_ROOT`. Pre-stage on a connected machine and add
`--offline` to evaluation when compute nodes have no network access.

The cache is additional to the public CFD data. Obtain the selected cases from
[`neashton/ahmedml` at the pinned revision](https://huggingface.co/datasets/neashton/ahmedml/tree/02688c727cdb8dc8678e28abc6bbbb7e93c5fa15):
`boundary_N.vtp`, `boundary_cell_area_N.npy`, `volume_N.vtu`,
`geo_parameters_N.csv`, and `force_mom_N.csv` within each `run_N/` directory.
The split files list the required case IDs; the
[public source identity](../../benchmark-specs/ahmedml/public-source-identity/ahmedml-public-source-identity-v1.json)
pins their paths, sizes and hashes. An existing copy of these exact files is reusable.
The support download does not download raw CFD or model predictions.

## Inputs and blockers

Copy `package-config.template.json` outside the repository and replace every
`__REPLACE_...__` value with the actual method record. The fixed release
bindings must continue to equal the published specification; do not invent or
silently update them. Check unresolved values without touching evaluator data:

```bash
python scripts/assemble_ahmedml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --list-unresolved
```

For each case in the selected split, first run the evaluator against
the complete surface and volume native-order prediction manifests:

```bash
python scripts/evaluate_ahmedml_candidate_case.py \
  --case-id run_N \
  --dataset-root /path/to/ahmedml-public-revision-02688c7 \
  --surface-prediction-manifest /path/to/predictions/run_N/surface/manifest.json \
  --volume-prediction-manifest /path/to/predictions/run_N/volume/manifest.json \
  --output /path/to/case-evidence/run_N.json
```

The evaluator finds the prepared cache automatically. If a case is missing, it
downloads and verifies that case before evaluation begins. To use an existing
cache explicitly, add `--case-support /path/to/local-support/run_N/case-support.json`;
an explicit path is never downloaded or replaced. Prepared support also works
with `--support-root` and `--offline` as described above.

The prediction transport may be chunked, but it must cover every native cell
exactly once and preserve native order. The evaluator derives all field scores,
Cd/Cl, three moving-geometry Cp cuts, four moving-geometry wake profiles, and
surface/volume regional diagnostics from those same fields. Participant-authored
profile values are not accepted.

Create one JSONL spatial record per case in exact split order. Each record uses
the schema-v3 discretization-case format and reports the actual model input,
direct-output, and mapping counts. Reusing native meshes, sampling model points,
or mapping predictions back to native cells must be described rather than
silently normalized by the assembler.

<a id="assemble-and-validate"></a>

## Package, validate and submit

Follow the [AhmedML participant quickstart](../../benchmark-specs/ahmedml/QUICKSTART.md)
to evaluate the complete split and run `python scripts/ahmedml.py package`.
That command checks the evaluation outputs, checkpoint identities and release
bindings and creates the result directory. Use its generated package for submission.
Keep the template release bindings unchanged.

```bash
python scripts/ahmedml.py verify-package submissions/ahmedml/YOUR_SUBMISSION_ID
python scripts/validate_submission.py --contributor-stage submissions/ahmedml/YOUR_SUBMISSION_ID
```

Commit only the generated `submissions/ahmedml/YOUR_SUBMISSION_ID/` directory
and open a pull request against `dev`. Follow [Submitting a result](../../SUBMITTING.md).
Maintainers review, approve and publish the result. Do not add approval records,
registration files or generated leaderboard feeds to a participant pull request.
