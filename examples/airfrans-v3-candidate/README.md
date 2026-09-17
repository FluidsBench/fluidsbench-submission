# AirfRANS FluidsBench schema-v3 candidate package

This directory holds the config template for assembling an AirfRANS
schema-v3 candidate submission, plus one concrete example
(`transolverpp-full-v1-candidate-config.json`) used to build the
`transolverpp-full-v1` package under `submissions/airfrans/`.

AirfRANS remains closed for real submissions
(`benchmark-specs/airfrans/submission-spec.json` has
`scoring_support.status: owner_review_required` and
`submissions_open: false`), so nothing built with this tooling is an
accepted result. `scripts/validate_submission.py --contributor-stage`
will still report failures tracing back to that closed status (missing
official scoring-support manifest binding, dataset not marked
`"official"`, no leaderboard entry for the profile ground truth) — all
three are on the dataset owner's own pending-approvals list, not a
defect in the package.

## Pipeline

Predictions become a candidate package in four steps. The first two
need the pinned `airfrans`/PyVista/VTK stack
(`requirements-airfrans-evaluator.txt`); the third is pure numpy and
reuses the repository's generic evaluator; the fourth is packaging only.

### 1. Ground-truth scoring support (once per dataset, not per model)

```bash
python scripts/build_airfrans_scoring_support.py \
  --dataset-root /path/to/AirfRANS/Dataset \
  --output /path/to/airfrans-native-ground-truth-v1-candidate
```

For every case in the official split, computes point-dual physical
weights (`reference/airfrans/geometry.py`: mass-lumped dual-cell area on
the 2D domain, dual-segment length on the airfoil curve — the first use
of point-dual weighting in this repo, since every other dataset
publishes cell-associated fields instead) alongside the ground-truth
velocity, pressure, wall shear, and forces, all read directly from the
pinned `airfrans` library. No prediction is involved, so this release
is reusable across every future AirfRANS submission. It is large
(hundreds of MB for the Full split); keep it outside the repository,
the way the produced `transolverpp-full-v1-candidate-config.json`
example does.

### 2. Per-submission field derivation (once per model)

```bash
python scripts/derive_airfrans_predicted_fields.py \
  --dataset-root /path/to/AirfRANS/Dataset \
  --predictions-dir /path/to/packed-model-predictions \
  --scoring-support-manifest /path/to/airfrans-native-ground-truth-v1-candidate/manifest.json \
  --submission-id my-model-full-v1 \
  --output /path/to/my-model-predicted-fields
```

Injects the model's predicted velocity/pressure into the same
`airfrans.Simulation` object used for ground truth and lets the library
compute the same derived quantities (wall shear, forces, curve
pressure) from the prediction. `--predictions-dir` expects one
`<case_id>.npz` per case with `velocity`, `pressure`,
`internal_points`, and `airfoil_points` arrays in unchanged native
point order (see `AirfRANS_Inference_Handover/pack_predictions.py` for
an example packer). Writes a `kind: scored_predictions` manifest
directly consumable by `reference.evaluate_predictions`.

### 3. Metric scoring

```bash
python scripts/finalize_airfrans_metrics.py \
  --support-manifest /path/to/airfrans-native-ground-truth-v1-candidate/manifest.json \
  --case-set standard \
  --prediction-manifest /path/to/my-model-predicted-fields/manifest.json \
  --submission-id my-model-full-v1 \
  --split-id full \
  --profile-score /path/to/velocity-profile-score.json \
  --output /path/to/metrics-cases.json
```

This calls the repository's existing generic evaluator
(`reference/evaluate_predictions.py`), which already implements every
reduction AirfRANS needs (relative-L2/L1, per-geometry-then-macro-average,
and the all-test-cases R²/MAE aggregation the force metrics use) —
nothing AirfRANS-specific was added there. `finalize_airfrans_metrics.py`
only adds what that generic pipeline can't know on its own: it folds in
`velocity_profile_r2` (from the existing
`examples/airfrans-profile-extraction/extract.py` + `score.py`
pipeline) and computes the composite scores
(`overall_score`/`field_score`/`force_score`/`diagnostic_score`) from
`benchmark-specs/airfrans/submission-spec.json`'s own declarations.

### 4. Package assembly

Copy `package-config.template.json` outside the repository and replace
every `__REPLACE_WITH_...__` token with the submitted method's real
methodology (architecture, parameter counts, training/checkpoint
details, measured compute) and submitter identity. List what's still
unresolved without writing output:

```bash
python scripts/assemble_airfrans_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --list-blockers
```

Once resolved, assemble the package and run the repository validator:

```bash
python scripts/assemble_airfrans_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --scoring-support-manifest /path/to/airfrans-native-ground-truth-v1-candidate/manifest.json \
  --metrics-cases /path/to/metrics-cases.json \
  --predicted-profiles-dir /path/to/predicted-profiles \
  --profile-score /path/to/velocity-profile-score.json \
  --output submissions/airfrans/my-model-full-v1

python scripts/validate_submission.py --contributor-stage submissions/airfrans/my-model-full-v1
```

The assembler writes `submission.json`, `evaluation-evidence.json`,
`discretization.json` + `discretization/cases.jsonl`,
`metrics/cases.json`, and `profiles/index.json` + per-case chunks. It
never writes `approval`, `maintainer-validation.json`, or
`prediction-artifact-checks.json` — those are maintainer-only records.
