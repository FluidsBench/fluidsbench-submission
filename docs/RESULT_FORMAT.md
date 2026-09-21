# Result package reference

Use the [submission workflow](../SUBMITTING.md) for the steps and the selected [dataset guide](../benchmark-specs/README.md)
for exact outputs, native support, units, reductions, and profile format. This reference preserves the shared file and version
requirements; the linked JSON schemas and dataset contracts are authoritative.

## Submission directory

Create one directory under the matching dataset:

```text
submissions/<dataset-id>/<submission-id>/
  submission.json
  evaluation-evidence.json
  discretization.json
  discretization/
    cases.jsonl
  metrics/
    cases.json
  profiles/
    index.json
    chunk-000.json
    chunk-001.json
```

Use lowercase letters, numbers, and hyphens for the globally unique `submission-id`. Do not edit generated files under
`leaderboard/` directly.

New submissions use schema v3. The material under [`examples/v3-template/`](../examples/v3-template/) demonstrates the fixed-support
and prediction-manifest formats; dataset-specific submission templates can be completed only after the dataset owner marks its
scoring support `official` and opens submissions. Historical v1 prototype fixtures remain under `submissions/`; registered candidate previews retain their separate ineligible status, and [`examples/v2-template/`](../examples/v2-template/) is retained only to interpret historical v2 packages.

The tree shows the generic JSON profile layout. HiLiftAeroML compact-v2 uses prediction-only per-case NPZ artifacts instead.
Maintainers alone add `maintainer-validation.json` and, optionally, `prediction-artifact-checks.json` after submission.

## Metrics, spatial metadata, and profiles

`submission.json` contains model, submitter, methodology, dataset, split, aggregate metrics, scoring-support, spatial-discretization,
case-metric, profile-index, evaluation, and optional open-artifact metadata. New submissions must match
[`schemas/v3/submission.schema.json`](../schemas/v3/submission.schema.json). Historical approved schema-v2 packages remain readable and
publishable, but the contributor-stage validator rejects new v2 packages because they lack mandatory v3 evidence.

The required `evaluation` object records the FluidsBench reference version, exact command, and SHA-256 checksum of
`evaluation-evidence.json`; when code metadata is shared, it must also record the matching contributor code revision. The evidence
file repeats the command and submitted metric values and binds the dataset version, split digest, case set, scoring-support
manifest, spatial report, per-case metrics, profile-index checksum, and exact public profile-ground-truth release. New submissions
use
[`schemas/v3/evaluation-evidence.schema.json`](../schemas/v3/evaluation-evidence.schema.json). This evidence is not a replacement for
scientific review and does not contain full surface or volume fields.

Every new package also declares `reproducibility.contract_version=open-reproducibility-3.0`, public result-package access, public
evaluation-only data use, and an open licence for submitted result data. Public code, model weights, environment files, and artifact
documentation are optional and do not affect approval, rank, citation, or promotion eligibility. When declared, code must be pinned
to a full commit, model and environment artifacts must be pinned by SHA-256, and code/model licences must use the allowed open SPDX
identifiers. See [`OPEN_REPRODUCIBILITY.md`](../OPEN_REPRODUCIBILITY.md) for the complete eligibility and validation policy.

Every schema-v3 package includes the same structured methodology record covering architecture components, exact total and
submitter-trainable parameter counts, inputs and outputs, data handling, training stages, checkpoint selection, and measured
compute. Each dataset's `methodology-contract.json` names only that benchmark's existing required outputs; it does not change the
prediction or scoring process. A SHA-256 digest is required for every checkpoint file actually loaded by a parameterized method so
the result has an exact model identity; publishing those checkpoint bytes, source code, or a model archive remains optional. See
the repository-wide [`methodology guide`](../METHODOLOGY.md) and, for the native-mesh workflow, the
[`DrivAerML participant guide`](../benchmark-specs/drivaerml/PARTICIPANT_GUIDE.md).

The required `metrics/cases.json` records every test case, canonical support, support/scored counts, complete count and weight
coverage, unmapped/extrapolated counts, per-case metric values, and the additive sufficient statistics required by each relative-L2
metric. Those statistics include numerator, denominator, entity count, and total weight for the unweighted and
area-, length-, or volume-weighted variants.
Its aggregate `metric_values` must exactly match `submission.json`; macro-averaged spatial metrics are recalculated from its case
values by the validator. Global R2 and published dataset-reference RRMSE values are explicitly `aggregate_only`: the generic
evaluator calculates them from complete prediction arrays, while normal package validation checks their identities and rule
bindings without pretending to recompute them.

The required `discretization.json` distinguishes training input, training supervision, inference input, direct model output, and the
mapping to each canonical support. `discretization/cases.jsonl` records actual evaluation-case counts and mapping coverage. A method
may infer in chunks or use sparse points, grids, a participant mesh, or another representation internally. After the declared
mapping, however, it must provide exactly one prediction for every required entity in the original public field-bearing files.
When a native-resolution comparison is reported, each case supplies its model/native counts and fraction; the validator reconciles
those records with the fixed or minimum/median/maximum summary and verifies `fraction = model_count / native_count`.

Contributors leave `approval` absent and must not add `maintainer-validation.json` or
`prediction-artifact-checks.json`. After submitted-data validation, maintainers add
the separately hashed validation record and `approval.status=approved`. Prototype packages use `approval.status=prototype`; the feed
builder publishes only prototype and maintainer-approved rows. The sole exception class is an exact maintainer-registered,
hash-bound HiLiftAeroML `pre_release_reference`; each remains unapproved and explicitly ineligible for citation or promotion.

Sharing full or example prediction fields is optional. When used, `prediction_artifacts` points to a revision-pinned public Hugging
Face dataset manifest. A maintainer may add one `prediction-artifact-checks.json` index recording accessibility, format, or explicit
metric-recomputation checks. Normal CI validates metadata and local hashes only and never downloads multi-gigabyte artifacts.

Profile data is kept out of the scalar leaderboard feed. The generic JSON format below is used where the dataset contract selects it. HiLiftAeroML instead requires its [compact-v2 prediction-only NPZ format](../benchmark-specs/hiliftaeroml/PARTICIPANT_GUIDE.md#5-produce-the-official-compact-v2-profiles); do not convert it to this example. In the generic format, each chunk contains a manageable group of test geometries using compact
parallel arrays:

```json
{
  "case_id": "ahmedml_geometry_test_0001",
  "series": [
    {
      "panel_id": "pressure_profiles",
      "station_id": "upper_body_centerline",
      "quantity_id": "cp",
      "coordinate": [0.0, 0.5, 1.0],
      "prediction": [0.72, -0.31, 0.04]
    }
  ]
}
```

For the generic JSON format, use roughly 20-30 geometries per chunk unless the dataset specifies otherwise. Generic JSON coordinates must be finite, unique, and strictly increasing; dataset-specific native/compact profile formats retain their declared topology, masks, ordering, and gaps instead. Every required case,
panel, station, quantity, and any exact coordinate/sample-count rule is declared by the dataset specification and its bound
profile definition. Prototype fixtures may use an explicitly declared abridged profile only while they remain labelled
`approval.status=prototype`. The profile schemas are:

- [`schemas/v1/profile-index.schema.json`](../schemas/v1/profile-index.schema.json)
- [`schemas/v1/profile-chunk.schema.json`](../schemas/v1/profile-chunk.schema.json)

New maintainer validation records must match
[`schemas/v3/maintainer-validation.schema.json`](../schemas/v3/maintainer-validation.schema.json).

For schema v3, `profile_data.profile_ground_truth_release_id` and
`profile_data.profile_ground_truth_manifest_sha256` must exactly match `data_release.profile_ground_truth` in the leaderboard
manifest. The same values are repeated in the submitter-authored evaluation evidence and the maintainer validation record.

The profile index records the SHA-256 checksum and case IDs for each chunk. The generic format uses normal JSON so pull requests remain readable; hosting can compress it during delivery. HiLiftAeroML uses deterministic compact NPZ chunks under its separate contract.

## Result versions

Published result packages are immutable. Every schema-v3 submission declares a stable result series and an integer version:

```json
"submission_id": "team-model-drivaerml-full-v2",
"result_revision": {
  "series_id": "team-model-drivaerml-full",
  "version": 2,
  "supersedes": "team-model-drivaerml-full-v1",
  "change_summary": "Retrained the model and corrected the mesh-to-support mapping."
}
```

The first package is `<series-id>-v1`, uses `version: 1`, and sets `supersedes` to `null`. An update copies the previous package into
a completely new `<series-id>-vN` directory, replaces every changed prediction, metric, profile, evidence record, and checksum,
increments the version by exactly one, points to the immediately preceding published submission, and explains the material change.
Never edit, rename, or delete an earlier published directory.

For a result published before this versioning contract whose ID does not end in `-v1`, keep that package unchanged. Its first update
uses `<legacy-submission-id>-v2`, retains `<legacy-submission-id>` as the series ID, and sets `supersedes` to the exact legacy ID. For
example, `drivaerml-ab-upt-v2` supersedes `drivaerml-ab-upt`. The validator treats that immutable legacy package as v1.

A revision remains in the same series only when the exact dataset version, split hash, case set, submitter, and institution are
unchanged. A genuinely different benchmark contract, model family, or submitting team starts a new series at v1. The validator
rejects skipped versions, forks, missing or unpublished predecessors, non-chronological dates, and IDs that do not match
`<series-id>-vN`.

The current leaderboard ranks only the latest published version in each series, so repeated updates cannot occupy multiple ranking
positions. Earlier versions, their submitted scores, dates, metadata, and change summaries remain in the hash-bound revision-history
feed and are available from the result details. An immutable release snapshot continues to preserve the exact version and rank that
it originally published. Historical schema-v1/v2 packages are exposed as v1-compatible legacy records; all new schema-v3 packages
must declare `result_revision` explicitly.

## Metric reductions and ranking

Field metrics use the equations and edge-case behaviour in [metric reference](../reference/README.md). The standard relative-L2 policy reports
both unweighted and geometry-weighted views:

- a three-dimensional surface, a two-dimensional surface manifold embedded in three dimensions, or a one-dimensional boundary
  curve uses area or length weighting as the primary value and
  an unweighted result as the secondary value;
- a three-dimensional volume, or a two-dimensional flow domain, uses an unweighted result as the primary value and
  volume or area weighting as the secondary value; and
- each case is calculated separately, then case values are macro-averaged so every test case has equal influence.

Dataset-specific exceptions remain authoritative: DrivAerML, HiLiftAeroML, and WindsorML have no volume-weighted secondary metric.

For point- or node-associated arrays, the weighted value uses the authoritative per-point or per-node area, length, or volume
published in the support,
not weights reconstructed by a submitter. One spatial weight multiplies a scalar squared error or the complete squared vector
magnitude. A dataset specification must explicitly document any additional published source metric or different reduction,
including the VKI-LS59 and Rotor37 RRMSE reductions.

Inference may be performed in memory-safe chunks, but the final mapped result must cover every official entity in every case. Each
relative-L2 entry in `metrics/cases.json` records the additive numerator and denominator, entity count, and total weight. Chunk
numerators and denominators are summed before taking one square root for the complete case; chunk-level L2 values must never be
averaged. Sharing complete prediction fields remains optional.

Each metric's `aggregation` and `weighting` fields make the reduction explicit. `per_geometry_then_macro_average` gives every
test geometry equal influence; `flatten_all_aligned_field_values` evaluates one reduction over all aligned samples;
`all_test_cases` gives scalar cases equal weight; the two `benchmark_*_rrmse_across_cases` values use the exact RRMSE equations in
the reference documentation. Derived metrics list either `derived_score_equation` or `derived_arithmetic_mean`.

Every dataset currently ranks by the higher-is-better `overall_score` at one decimal place. The dataset's
`overall_score_composite` object is the machine-readable source of truth for its component metric IDs, weights, transforms, and
error caps. A `bounded_error` component contributes `clip(100 * (1 - error / cap), 0, 100)`; a `bounded_quality` component
contributes `100 * clip(value, 0, 1)`. A `physics_null_skill` component contributes
`100 * (1 - error / baseline_error)` without clipping, so worse-than-null methods retain negative skill. Its strictly positive
baseline error must be published by the frozen evaluator before the composite status can become `active`; a
`pending_reference_baselines` candidate is not rankable. The declared non-negative component weights sum to one. The validator recomputes this
composite from the submitted component values, so `overall_score` cannot be supplied independently. BlendedNet uses its four
area-weighted surface-field L2 values; DrivAerNet++ currently uses its one active area-weighted pressure L2; Rotor37 uses its six
existing RRMSE quantities; and VKI-LS59 uses its eight existing RRMSE quantities. Supplementary metrics remain visible but do not
silently enter the ranking score.

Each specification also publishes the leaderboard `ranking` contract: metric ID, direction, decimal places, decimal rounding rule,
and competition-ranking method. The decimal places must equal that metric's display digits in the release manifest. FluidsBench
rounds the submitter-supplied ranking value with decimal half-up rounding before both display and comparison, and assigns equal
rounded values the same competition rank (`1, 2, 2, 4`). Rank scope is always one immutable release, dataset, and split; filters or
later submissions do not rewrite a historical rank.

## Licences

Repository code and published leaderboard metadata are covered by the [Apache License 2.0](../LICENSE). Each real submission declares
an open `reproducibility.result_data_license_spdx` for its contributed profile values and result material; submitters must have the
right to provide them under that licence. Upstream datasets and model artifacts retain their own declared licences. Code and model
artifacts are optional. If supplied, they must be publicly accessible and use one of the explicitly allowed open SPDX identifiers
in the v3 submission schema. See the category-specific lists in
[`OPEN_REPRODUCIBILITY.md`](../OPEN_REPRODUCIBILITY.md); proprietary and unrecognised declared licence values fail validation.
