# AhmedML schema-v3 genuine-inference candidate

This directory is a reusable configuration template, not a submission. The
AhmedML native evaluator remains a closed owner-review candidate and public
submissions are not open. The assembler creates an unapproved candidate package
for any of the eight official split labels only after all placeholder tokens,
checkpoint hashes, complete evaluator evidence, and spatial records agree.

The genuine-inference path is separate from
`assemble_ahmedml_schema_v3_dev_fixture.py`. It does not accept
`prototype_fixture` methodology and never reads fixture provenance. It requires
an explicit attestation that every prediction was produced by actual model
inference and not generated from evaluation truth, and verifies the local bytes
of every checkpoint named by the methodology record.

<a id="inputs"></a>

## Inputs and blockers

Copy `package-config.template.json` outside the repository and replace every
`__REPLACE_...__` value with the actual method record. The fixed release
bindings must continue to equal the repository candidate; do not invent or
silently update them. Check unresolved values without touching evaluator data:

```bash
python scripts/assemble_ahmedml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --list-unresolved
```

For each case in the selected split, first run the candidate evaluator against
the complete surface and volume native-order prediction manifests:

```bash
python scripts/evaluate_ahmedml_candidate_case.py \
  --case-id run_N \
  --dataset-root /path/to/ahmedml-public-revision-02688c7 \
  --case-support /path/to/local-support/run_N/manifest.json \
  --surface-prediction-manifest /path/to/predictions/run_N/surface/manifest.json \
  --volume-prediction-manifest /path/to/predictions/run_N/volume/manifest.json \
  --output /path/to/case-evidence/run_N.json
```

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

## Commands: assemble and validate

```bash
python scripts/assemble_ahmedml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --case-evidence-directory /path/to/case-evidence \
  --dataset-evidence /path/to/optional-retained-split-reduction.json \
  --discretization-cases /path/to/discretization/cases.jsonl \
  --profile-ground-truth-manifest /path/to/profile-ground-truth/manifest.json \
  --checkpoint surface-model=/path/to/surface/model_best.pt \
  --checkpoint volume-model=/path/to/volume/model_best.pt \
  --output submissions/ahmedml/ahmedml-my-model-full-v1

python scripts/validate_submission.py \
  --candidate-dry-run \
  submissions/ahmedml/ahmedml-my-model-full-v1
```

Use exactly one `--checkpoint ID=PATH` for each methodology checkpoint ID. A
single joint model needs one argument; separate surface and volume models need
two. The optional retained dataset evidence is compared byte-for-data with a
fresh reduction; the assembler always recomputes the split from per-case
evidence.

## Outputs

The output contains prediction-only profile chunks with exactly seven series
per case and 128 values per series, complete per-case sufficient statistics,
discretization evidence, and zero-weight regional diagnostics. It contains no
`approval`, maintainer validation, or official claim.

## Dev-only pre-release registration

Maintainer-only registration can expose an exact candidate on the dev feed. Candidate validation alone does not publish it or approve it.

<details>
<summary>Inspect or register an immutable pre-release binding</summary>

After the package passes the candidate dry run and is stored at its conventional
repository path, inspect its proposed exact binding:

```bash
python scripts/register_ahmedml_pre_release.py \
  submissions/ahmedml/ahmedml-my-model-full-v1/submission.json
```

The proposal pins the submission, every declared artifact, all checkpoint
digests, split/case/profile coverage, and a logical hash of every package file.
It changes nothing. A maintainer can append that exact reviewed binding with:

```bash
python scripts/register_ahmedml_pre_release.py \
  --register \
  submissions/ahmedml/ahmedml-my-model-full-v1/submission.json

python scripts/manage_leaderboard.py build
```

Registration affects only the prototype dev feed. The resulting row is labeled
`pre_release_reference`, remains non-citable and non-promotable, and disappears
from an official feed unless it later follows the separate owner approval
workflow. Any changed package byte invalidates the registration.

</details>
