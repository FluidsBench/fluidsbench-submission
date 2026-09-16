# Benchmark submission specifications

Each dataset directory contains a machine-readable `submission-spec.json` and one split index per accepted leaderboard split.
The specification lists the required scalar metrics and profile panel, station, and quantity IDs. It also declares the exact
original public field-bearing files, required entities, arrays, point/node/face/cell associations, area, length, or volume weights,
canonical
scoring-support status, and support identity. The validator consumes these same files, so the written instructions and automated
checks use one contract. Three-dimensional specifications use surface and flow-domain terminology; two-dimensional specifications
use boundary curve and two-dimensional flow-domain terminology.

Each official specification pins `evaluation_reference_version` to an immutable FluidsBench reference release. Submission and
maintainer-validation records must identify that exact release. The reference defines how submitters calculate their values;
FluidsBench does not rerun the submitted model or recompute base metrics from full prediction fields.

Schema-v3 evaluation evidence repeats the dataset version, split SHA-256, case-set ID, scoring-support release and manifest
SHA-256, spatial-report SHA-256, per-case-metric SHA-256, and exact public profile-ground-truth release ID and manifest SHA-256 from
the submission. This binds the submitted package to its declared evaluation and comparison basis.

All official split case IDs and evaluation ground truth are public. The evaluation data may be used only for final evaluation,
never for fitting, model or checkpoint selection, hyperparameter selection, manual tuning, or preprocessing statistics. Real
submissions make the machine-readable `public_test_data_use=\"evaluation_only\"` declaration defined by
[`../OPEN_REPRODUCIBILITY.md`](../OPEN_REPRODUCIBILITY.md).

Split indexes marked `official` contain dataset-owner-approved case IDs. Indexes marked `prototype_generated` preserve the
current dummy leaderboard's declared test count but are not suitable for real submissions. Dataset owners must replace those IDs,
set the status to `official`, and review the resulting SHA-256 value before FluidsBench opens that split for submissions.

The `scoring_support` object uses one of four statuses:

- `official`: the dataset owner approved the manifest and case set and `submissions_open` may be true;
- `owner_review_required`: a concrete proposal exists but the dataset owner has not approved it;
- `prototype`: format or evaluator development only; or
- `retired`: preserved for historical interpretation and closed to new submissions.

Every closed status supplies `closed_reason`. Only an `official`, owner-approved release with `submissions_open=true` can accept a
schema-v3 result. The manifest pins its case-set index; the index pins every chunk; and chunks locate or hash the benchmark-owned
coordinates, weights, ground truth, and stable IDs. For current field benchmarks, the support covers every required entity in the
pinned original public file. The validator requires its cases to match the accepted split exactly and every case to have complete
entity coverage.

Dataset support is approved independently. A dataset collaborator changes only their dataset directory in one pull request:

1. replace prototype split IDs with the ordered official evaluation case IDs, set each `case_id_status` to `official`, and update
   the split counts and SHA-256 values;
2. confirm and pin the original public file names, immutable revisions and hashes, field-array names, entity associations, required
   patches or masks, and exact entity ordering for every case;
3. publish stable support IDs and authoritative area, length, volume, or cell-area weights. Face- or cell-associated data use the
   corresponding face area,
   curve-segment length, cell volume, or cell area. Point- or node-associated data use benchmark-generated deterministic
   mass-lumped dual measures from the exact pinned mesh;
4. add the dataset's scoring-support manifest, case-set indexes, chunks, and any small local support artifacts. The support must
   bind the complete original entity set, public ground truth, both required relative-L2 weightings, and case-macro aggregation;
5. set `scoring_support.status` to `official`, add the immutable release ID, repository-relative manifest path, public HTTPS
   manifest URL, manifest SHA-256, and their name/date/pull-request URL under `owner_approval`; if support artifacts are remote or
   require dataset-specific loading, also add the pinned `publication_validation` receipt produced by that loader;
6. keep `submissions_open=false` while the scientific contract is under review, then set it to `true` only when that dataset is
   ready to receive v3 result pull requests; and
7. run `python scripts/validate_scoring_supports.py`.

That command validates only the releases declared by each specification. For an official dataset it schema-checks the manifest,
follows and hashes the manifest → case-index → chunk chain, checks local artifact hashes, and requires every case set to match its
ordered official split exactly. It also requires the manifest's metric bindings to cover exactly the field and case-scalar metrics
in `submission-spec.json`, including reduction, weighting, aggregation, evidence mode, and evaluator-version pins. A complete local
materialized-table release is smoke-loaded case by case. A remote or dataset-specific release instead requires a committed receipt
pinning the validator file/digest, manifest digest, normalized-support digest, case/support counts, reviewer, timestamp, and pull
request. Other datasets may remain `owner_review_required` and closed. A collaborator's approving pull request is the durable
scientific approval record; FluidsBench's protected-branch maintainer still performs the final merge.

Field metrics use the equations and edge-case behaviour in [`../reference/`](../reference/). The standard relative-L2 policy reports
both unweighted and geometry-weighted views:

- a three-dimensional surface, a two-dimensional surface manifold embedded in three dimensions, or a one-dimensional boundary
  curve uses area or length weighting as the primary value and
  an unweighted result as the secondary value;
- a three-dimensional volume, or a two-dimensional flow domain, uses an unweighted result as the primary value and
  volume or area weighting as the secondary value; and
- each case is calculated separately, then case values are macro-averaged so every test case has equal influence.

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
