# DrivAerML FluidsBench schema-v3 package

DrivAerML is open for submissions.
Copy `package-config.template.json` and replace the participant `__REPLACE_...__`
fields. Its benchmark-owned release bindings are already filled in and must
match the checked-out release specification.

Start with the [resumable quickstart](../../benchmark-specs/drivaerml/QUICKSTART.md)
and the [participant guide](../../benchmark-specs/drivaerml/PARTICIPANT_GUIDE.md).
It covers native prediction chunks, fixed and relative profile extraction,
complete-split scoring, packaging, and submission review.

Use the shared [hardware and compute fields](../../METHODOLOGY.md#hardware-and-compute)
for each training stage and inference campaign, including GPU model and allocation counts.

## Participant configuration

Copy `package-config.template.json` outside the repository, fill the
participant fields. Keep the supplied benchmark release values. Check what
participant information remains before attempting assembly:

```bash
python scripts/assemble_drivaerml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --list-unresolved
```

Include the common [methodology record](../../METHODOLOGY.md) and the selected scope's outputs.
`surface_only` requires the two surface fields and eight fixed/relative Cp series; omit volume predictions and velocity series, retain fixed
zero contributions without renormalizing weights, and do not fill missing metrics with dummy values (maximum overall score 60).
`surface_and_volume` requires all four fields and 40 fixed/relative profile series.
Relative diagnostics carry zero score weight. Shared Cp aliases are generated automatically.

<details>
<summary>Required method configuration, checkpoint identity, stages, and parameter counts</summary>

The config-v2 format for schema-v3 packages requires a structured
`participant.methodology` record. Start from the placeholders in the template
and use [`methodology.example.json`](methodology.example.json) only as a shape
example. Replace all illustrative values with the submitted method's actual:

- named architecture components, exact total and submitter-trainable parameter
  counts, scoped hyperparameters and inputs, and the production path for all
  required fields for the selected scope (two surface fields in either scope, plus two volume fields only for `surface_and_volume`);
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

</details>

## Package, validate and submit

Follow the [DrivAerML participant quickstart](../../benchmark-specs/drivaerml/QUICKSTART.md)
to evaluate the complete split and run `python scripts/drivaerml.py package`.
That command checks the evaluation outputs, checkpoint identities and release
bindings and creates the result directory. Use its generated package for submission.
Keep the template release bindings unchanged.

```bash
python scripts/drivaerml.py verify-package submissions/drivaerml/YOUR_SUBMISSION_ID
python scripts/validate_submission.py --contributor-stage submissions/drivaerml/YOUR_SUBMISSION_ID
```

Commit only the generated `submissions/drivaerml/YOUR_SUBMISSION_ID/` directory
and open a pull request against `dev`. Follow [Submitting a result](../../SUBMITTING.md).
Maintainers review, approve and publish the result. Do not add approval records,
registration files or generated leaderboard feeds to a participant pull request.
