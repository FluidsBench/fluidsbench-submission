# DrivAerML FluidsBench schema-v3 candidate package

This directory is a configuration template, not a submission and not scoring
support. It deliberately contains `__REPLACE_...__` participant fields and
`__UNRESOLVED_DRIVAERML_...__` owner-release fields. Those strings are
machine-detectable blockers; they are not example hashes or release IDs.
The benchmark's separate release hand-off may already hash-bind a local v10
profile registry. That is not an instruction to replace this template's v10
tokens: wait until the active specification and all owner-release bindings are
published as one coherent release.

Copy `package-config.template.json` outside the repository, fill the
participant fields, and use the immutable release values published by the
DrivAerML benchmark owner. Check what remains before attempting assembly:

```bash
python scripts/assemble_drivaerml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --list-unresolved
```

The config-v2 format for schema-v3 packages requires a structured
`participant.methodology` record. Start from the placeholders in the template
and use [`methodology.example.json`](methodology.example.json) only as a shape
example. Replace all illustrative values with the submitted method's actual:

- named architecture components, exact total and submitter-trainable parameter
  counts, scoped hyperparameters and inputs, and the production path for all
  four required fields;
- normalization, preprocessing, and sampling plus every submitter or upstream
  training stage; submitter stages include their fitting procedure, runs,
  random seeds when stochastic, and measured compute;
- a raw-file SHA-256, byte description, component scope, role, and pre-test
  selection rule for every loaded checkpoint file; and
- complete-split inference campaign wall time, aggregate device time, and their
  inclusion boundaries.

`parameter_count_millions` is deliberately absent from the configuration. The
assembler derives it from `methodology.architecture.total_parameter_count`. The
checkpoint hashes bind the local bytes actually loaded for inference; they do
not require public model-weight upload and need not equal the hash of an
optional published model archive.

The template shows one jointly trained component and one gradient-based stage.
Add components, stages, and checkpoint-file entries when surface and volume use
different networks, when training is staged, or when a model is an ensemble.
For an upstream stage use `status=performed_upstream`, a component scope, and
an upstream reference. This provenance choice is independent of the top-level
target-data `training_regime`.

The token report does not validate repository bindings or evaluator outputs;
successful assembly is the authoritative readiness check. Once the
owner-release fields are published and the frozen evaluator produces
a complete `metrics/cases.json` and `profiles/` directory, assemble the
participant-owned files with:

```bash
python scripts/assemble_drivaerml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --case-metrics /path/to/metrics/cases.json \
  --profiles /path/to/profiles \
  --discretization-cases /path/to/discretization/cases.jsonl \
  --output /path/to/drivaerml-my-model-v1
```

The assembler derives dataset and split identities from the checked-in
DrivAerML specification, canonicalizes the participant files, creates the
hash-bound `submission.json`, `evaluation-evidence.json`, and
`discretization.json`, and schema-checks the result. It refuses unresolved
tokens, incomplete or reordered cases, missing metrics or profiles, mismatched
release bindings, and an existing output directory.

Evaluator-produced `metrics/cases.json` and `profiles/index.json` must already
carry the exact submission, split, and case-set identities supplied to the
evaluator. `metrics/cases.json` must additionally carry the exact candidate
support release ID and manifest hash; the profile index has no release-identity
fields. The assembler rejects conflicts instead of rewriting them and verifies
every source profile chunk against the evaluator's input index before
canonicalizing it. Only the participant-authored per-case
`discretization/cases.jsonl` may omit its repeated schema/submission/dataset/
split identity fields; the assembler inserts those fields and rejects any
conflicting value that is present.

The frozen evaluator Git revision is benchmark-owned and is pinned by
`scoring_support.dataset_evaluator_binding` in the benchmark specification; it
is never written to `evaluation.code_revision`. That optional submission field
identifies the participant's public model code. The assembler omits it when
`participant.reproducibility.code` is absent; when that block is supplied, it
copies the participant code commit into both submission and evidence records.

The output contains no `approval`, `maintainer-validation.json`, or
`prediction-artifact-checks.json`; those are not contributor-owned records.
Complete native prediction artifacts are optional and are intentionally not
added by this initial assembler.
