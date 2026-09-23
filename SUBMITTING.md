# Submitting a result

**Intake is closed for every dataset.** Use a candidate workflow only for local or dataset-owner-coordinated review until the
dataset's specification declares an official, owner-approved support release with `submissions_open=true`.
Candidate, prototype, and pre-release reference results do not establish an official leaderboard claim.

## 1. Choose the dataset and split

Start in the [dataset index](benchmark-specs/README.md). Read its guide and `submission-spec.json`, then pin the exact dataset
revision, ordered case list, split hash, scoring support, and evaluator release. An `official` split list alone does not open
submissions. Never invent missing release IDs or hashes to complete a template.

Fit models and learned preprocessing on the allowed training cases; use validation cases only as the selected split permits.
Public evaluation fields are for final evaluation only: no fitting, early stopping, checkpoint/model selection, hyperparameter
or manual tuning, normalization, or preprocessing statistics. Disclose prior contact with evaluation data in your PR; it may
make the result ineligible for ranking. See the [eligibility policy](OPEN_REPRODUCIBILITY.md).

## 2. Evaluate every required entity

Use the selected dataset's evaluator and definitions. Internal representations and inference chunks may differ from the native
mesh, but the final mapping must supply exactly one prediction for each required original entity. Mapping error is part of
the result; do not reconstruct benchmark weights, alter masks, omit difficult cases, or average chunk-level metrics.

<details>
<summary>Metric calculation: equations, weighting, chunk reduction, and reference example (required when implementing an evaluator interface)</summary>

Read the selected dataset's page on the [FluidsBench website](https://fluidsbench.org/datasets/) and its machine-readable
specification under [`benchmark-specs/`](benchmark-specs/). The specification lists accepted splits, required metric IDs, units,
directions, profile panels, stations, and quantities. Where profile extraction depends on a dataset library, the optional
`profile_definition` binding also pins a machine-readable extraction file and its SHA-256 digest.

The equations, edge cases, fixed-support joins, and NumPy reference implementations are documented in
[`reference/README.md`](reference/README.md). Each dataset specification identifies the exact files and field associations. It uses
surface and flow-domain terminology for three-dimensional datasets. Lower-dimensional files are described by what they actually
represent: a two-dimensional flow domain, a two-dimensional surface manifold embedded in three dimensions, or a one-dimensional
boundary curve. Participants create keyed predictions using their own inference pipeline. The reference evaluator
joins those predictions to every benchmark-owned support ID and the fixed public ground truth, checks complete coverage, and
produces the metric evidence. Mapping or interpolation to the original public entities is part of the submitted evaluation
pipeline, so its error is included in the result.

For relative L2 field errors, the standard paired reporting policy is:

- on a three-dimensional surface, a two-dimensional surface manifold, or a one-dimensional boundary curve, report the area- or
  length-weighted result as the primary value and the unweighted result as a secondary value;
- in a three-dimensional volume, or the two-dimensional flow domain, report the unweighted result as the primary value and the
  volume- or area-weighted result as a secondary value; and
- calculate each test case separately, then macro-average the case results so every case has equal influence.

Apply this paired policy only where the dataset contract requires both variants. DrivAerML, HiLiftAeroML, and WindsorML do not require a volume-weighted secondary metric.

The weights are face area, curve length, cell volume, or cell area as appropriate. For point- or node-associated fields,
FluidsBench supplies the corresponding per-point or per-node weight from the exact pinned mesh. Submitters use those
benchmark-owned values; they do not independently reconstruct weights from a modified mesh. For vector fields, one entity weight
multiplies the squared vector magnitude, not each component as a separate spatial sample. The selected dataset's machine-readable
specification remains authoritative if its published source metric requires a documented exception.

```python
from reference.metrics import relative_l2

pressure_l2_percent = relative_l2(
    ground_truth_pressure,
    predicted_pressure,
    weights=surface_face_areas,
)
```

Large cases may be evaluated in chunks. For each case and metric, accumulate
`numerator = sum(w * ||prediction - ground_truth||^2)` and
`denominator = sum(w * ||ground_truth||^2)` across all chunks, together with the entity count and total weight. Take
`100 * sqrt(numerator / denominator)` only after the complete case has been accumulated. Never calculate an L2 value per chunk and
average those chunk values. Use `w = 1` for the unweighted variant and the benchmark-supplied area, length, volume, or cell-area
weight for the weighted variant.

Run the executable reference example with:

```bash
python3 -m reference.example_calculation
```

</details>

## 3. Prepare the result package

Follow the [result format](docs/RESULT_FORMAT.md) and [methodology guide](METHODOLOGY.md). Record the actual architecture,
training and selection procedure, loaded checkpoint digests, measured compute, spatial representation, and mapping counts.
Report the GPU/device model and devices per job separately from peak campaign concurrency; see
[hardware and compute](METHODOLOGY.md#hardware-and-compute) for the shared fields and unknown-value rules.
Use the dataset's assembler where provided; preserve its evidence, required profile format, and hash bindings.

Create one globally unique `submissions/<dataset-id>/<series-id>-vN/` directory. A new series starts at v1; revisions increment
by one and preserve previous packages. Follow the [complete version rules](docs/RESULT_FORMAT.md#result-versions), including
legacy IDs. Leave `approval` absent; contributors must not create maintainer validation or optional prediction-check records.

You may generate equivalent JSON in Python, MATLAB, Julia, C++, or another language; the schema and values are authoritative.
Public code, model weights, environments, artifact documentation, and full prediction fields remain optional. When supplied,
their links, revisions, hashes, and licences must pass the [reproducibility contract](OPEN_REPRODUCIBILITY.md).

To request the optional blue **Metrics verified** badge, tick the separate verification request in your PR and link complete,
versioned scored predictions. A maintainer must recompute the metrics across the full split before the badge can appear.
See the [short verification guide](docs/OPTIONAL_VERIFICATION.md); model weights and training code remain optional.

## 4. Validate locally

Run commands from the repository root. Dataset-native evaluators may require additional pinned dependencies or Linux; check the
dataset guide before installing or running a large evaluation.

Create a Python environment and install the validation/reference dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Validate one directory:

```bash
python3 scripts/validate_submission.py --contributor-stage submissions/ahmedml/my-model-v1
```

Dataset owners may exercise a closed schema-v3 candidate before activation
only when the dataset specification pins an exact
`scoring_support.candidate_manifest` (status, release ID, repository file,
public URL, and SHA-256):

```bash
python3 scripts/validate_submission.py --candidate-dry-run submissions/<dataset-id>/<package-id>
```

The package declares `scoring_support.status=candidate`; the dataset specification remains closed (for example, `owner_review_required`). This mode requires
`submissions_open=false` and a candidate support manifest. It rejects approval
and maintainer-owned metadata and reports only non-approving candidate
validity. Normal and `--contributor-stage` validation continue to require an
official, open, owner-approved scoring-support release.

Validate every source submission and verify that generated feeds are synchronized:

```bash
python3 scripts/manage_leaderboard.py check
```

The validator checks JSON schemas, exact public split and complete original-support coverage, unweighted and area-, length-, or
volume-weighted relative-L2 sufficient statistics, case-metric aggregation, spatial counts and domains, declared mappings,
evaluation-evidence
identities and checksums, profile coverage, coordinate ordering, array lengths, finite values, split/chunk hashes, any declared
open-artifact metadata, and approval lifecycle rules. It does not execute the model, download optional remote predictions, or
recompute submitted base metrics.

## 5. Open a result PR when the dataset is accepting submissions

1. Commit exactly one entirely new submission directory. Do not change specifications, schemas, scripts, workflows, generated
   feeds, existing submissions, or maintainer-owned files in that PR.
2. Open the PR against `main` after FluidsBench announces that the dataset is accepting real submissions.
3. Complete the [contributor checklist](.github/pull_request_template.md) and resolve all automated failures.
4. Retain the evaluation evidence: maintainers review scientific provenance, data use, submitted values, licences, and any
   declared artifacts, and may request calculation evidence.

Passing validation does not approve or publish the result. Maintainers first merge the unapproved package, then use a separate
[validation and approval PR](docs/MAINTAINERS.md#review-and-approve-a-result) to publish it. Required approval checks submitted
data only. Optional prediction audits are descriptive and do not change approval, ranking, citation, or promotion eligibility.

Closed candidates stay local or in owner-coordinated review. For DrivAerML and HiLiftAeroML, follow the exact release blockers
and local-support instructions in their guides; a successful dry run does not authorize a public result PR.
