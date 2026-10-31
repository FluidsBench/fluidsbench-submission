# Submitting a result

Start with the [dataset index](benchmark-specs/README.md) and your dataset's quickstart.
The steps below apply to every accepted dataset.

## Set up your submission branch

On GitHub, [fork the submission repository](https://github.com/FluidsBench/fluidsbench-submission/fork)
into your account. Replace `YOUR_GITHUB_USERNAME` below, then run:

```bash
github_name=YOUR_GITHUB_USERNAME
git clone "https://github.com/$github_name/fluidsbench-submission.git"
cd fluidsbench-submission
git remote add upstream https://github.com/FluidsBench/fluidsbench-submission.git
git fetch upstream main
git switch -c submit-my-model upstream/main
```

This creates a submission branch from current upstream `main`, with `origin`
pointing to your writable fork. If you used the website's reference-example
checkout, keep it for that example and use this fork checkout for your result.
Run the dataset quickstart from this repository root. Keep the same evaluation
checkout through evaluation, resume and packaging; the workflow records its code
identity. The PR must contain only the generated result directory.

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
by one and preserve previous packages. Follow the [complete version rules](docs/RESULT_FORMAT.md#result-versions). Leave `approval` absent; contributors must not create maintainer validation or optional prediction-check records.

You may generate equivalent JSON in Python, MATLAB, Julia, C++, or another language; the schema and values are authoritative.
Public code, model weights, environments, artifact documentation, and full prediction fields remain optional. When supplied,
their links, revisions, hashes, and licences must pass the [reproducibility contract](OPEN_REPRODUCIBILITY.md).

To request the optional blue **Metrics verified** badge, tick the separate verification request in your PR and link complete,
versioned scored predictions. A maintainer must recompute the metrics across the full split before the badge can appear.
See the [short verification guide](docs/OPTIONAL_VERIFICATION.md); model weights and training code remain optional.

## 4. Validate locally

Run commands from the repository root. Dataset-native evaluators may require additional pinned dependencies or Linux; check the
dataset guide before installing or running a large evaluation.

Use the pinned participant environment from your dataset guide for evaluation and packaging. For standalone validation,
create a separate environment with the shared validation/reference dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Use the current `main` branch of the FluidsBench submission repository and target your pull request at `main`.
Validate the generated result directory with the current intake flags:

```bash
python3 scripts/validate_submission.py --contributor-stage \
  submissions/<dataset-id>/<submission-id>
```

Open a pull request containing only that new result directory. This route supports AirfRANS Full, Scarce and AoA extrapolation,
plus AhmedML, WindsorML, DrivAerML and DrivAerNet++. Check the [dataset index](benchmark-specs/README.md) for current availability.

CI validates the package. Maintainers then review it and handle publication separately; merging the initial submission does
not itself publish a leaderboard row. See the
[maintainer guide](docs/MAINTAINERS.md) for approval, registration and feed generation.

The validator checks JSON schemas, exact public split and complete original-support coverage, unweighted and area-, length-, or
volume-weighted relative-L2 sufficient statistics, case-metric aggregation, spatial counts and domains, declared mappings,
evaluation-evidence identities and checksums, profile coverage, coordinate ordering, array lengths, finite values, split/chunk
hashes, any declared open-artifact metadata, and approval lifecycle rules. It does not execute the model, download optional
remote predictions, or recompute submitted base metrics.

## 5. Open a result PR when the dataset is accepting submissions

1. Commit exactly one entirely new submission directory. Do not change specifications, schemas, scripts, workflows, generated
   feeds, existing submissions, or maintainer-owned files in that PR.
2. Target the `main` branch using the validation command above.
3. Complete the [contributor checklist](.github/pull_request_template.md) and resolve all automated failures.
4. Retain the evaluation evidence: maintainers review scientific provenance, data use, submitted values, licences, and any
   declared artifacts, and may request calculation evidence.

For example, after creating and validating `submissions/ahmedml/my-model-full-v1/`:

```bash
git add -- submissions/ahmedml/my-model-full-v1/
git diff --cached --stat
git commit -m "Submit AhmedML my-model-full-v1"
git push -u origin submit-my-model
```

Replace the dataset and submission ID with your generated directory. Check that
the staged diff contains only that directory. Open the
[comparison page](https://github.com/FluidsBench/fluidsbench-submission/compare),
select **compare across forks**, and choose base repository
`FluidsBench/fluidsbench-submission`, base branch `main`, your fork as the head
repository, and `submit-my-model` as the compare branch. Create the PR and complete
its contributor checklist. Predictions, local support, work directories and
environments stay outside the PR.

Maintainers review the submitted package and handle approval, registration and publication through the
[maintainer workflow](docs/MAINTAINERS.md). Leave maintainer records and generated feeds out of your submission.
The review checks submitted data; optional prediction audits are recorded separately.

## See the leaderboard and follow publication

The [leaderboard preview](https://fluidsbench.org/leaderboard-preview/) shows the
metrics, methodology and native CFD profile viewer during the submission period.
Choose your dataset directly:
[AirfRANS](https://fluidsbench.org/leaderboard-preview/?dataset=airfrans),
[AhmedML](https://fluidsbench.org/leaderboard-preview/?dataset=ahmedml),
[WindsorML](https://fluidsbench.org/leaderboard-preview/?dataset=windsorml),
[DrivAerML](https://fluidsbench.org/leaderboard-preview/?dataset=drivaerml) or
[DrivAerNet++](https://fluidsbench.org/leaderboard-preview/?dataset=drivaernetplusplus).
The preview contains no submitted results.

CI validates your PR, maintainers review it, and approval is recorded separately.
An approved result appears only when included in an official published release;
merging the contributor PR alone does not add a row. The first-release cutoff
and leaderboard reveal are **5 December 2026, 23:59:59 Anywhere on Earth
(UTC−12)**. See the [website's release dates](https://fluidsbench.org/run/#first-release-dates).
After publication, use the [public leaderboard](https://fluidsbench.org/) and
the result's immutable release link to view its scores, profiles and claim record.
