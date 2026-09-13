# HiLiftAeroML closed-candidate participant guide

This guide describes how to prepare and locally validate a HiLiftAeroML
FluidsBench schema-v3 candidate. It is intentionally not an invitation to
submit a result. The checked-in specification has `submissions_open: false`,
the profile truth is not public, and final evaluator/owner release gates remain
authoritative.

No private leaderboard is part of this work. The commands below create and
validate local files only. They do not upload a result or make it visible in a
public or private leaderboard.

## 1. Select one declared evaluation label

Choose one `split_id` from [`splits/`](splits/) and preserve its exact ordered
`case_ids`. A package names one label even when another label shares the same
evaluation case set.

| `split_id` | Label | Cases | Exact case-set ID |
| --- | --- | ---: | --- |
| `full` | Full | 360 | `caseset-ac791749e527` |
| `medium` | Medium | 360 | `caseset-ac791749e527` |
| `scarce` | Scarce | 360 | `caseset-ac791749e527` |
| `super_scarce` | Super scarce | 360 | `caseset-ac791749e527` |
| `geometry` | Geometry | 360 | `caseset-53990ea68fa6` |
| `geometry_medium` | Geometry medium | 360 | `caseset-53990ea68fa6` |
| `geometry_scarce` | Geometry scarce | 360 | `caseset-53990ea68fa6` |
| `geometry_super_scarce` | Geometry super scarce | 360 | `caseset-53990ea68fa6` |
| `single_aoa_4` | AoA 4 | 36 | `caseset-7a743a20b3bd` |
| `single_aoa_12` | AoA 12 | 36 | `caseset-02fc12ff3494` |
| `single_aoa_22` | AoA 22 | 36 | `caseset-85ecccd9ccda` |
| `aoa` | AoA extrapolation | 900 | `caseset-29693354ed8a` |
| `deflection` | Deflection | 360 | `caseset-c0ecb14de138` |
| `stall` | Stall | 723 | `caseset-804491c8956e` |

These are 14 labels over eight unique case sets and 1,355 unique cases in the
union. The JSON files bind evaluation membership and order; they do not invent
or replace the dataset owner's training/validation definitions. Use the
owner-published training regime matching the label. Record that regime,
preprocessing statistics, model-selection procedure, and checkpoints in the
methodology record. Treat public evaluation fields as evaluation-only.

Before expensive inference, verify that the chosen label, ordered case list,
and case-set identity agree with [`submission-spec.json`](submission-spec.json).
Never substitute a similarly named local split or sort the cases differently.

## 2. Predict all four native fields

The current HiLiftAeroML candidate has one required scope: complete surface and
volume prediction. It does not define the DrivAerML `surface_only` zero-score
fallback. Every selected case must supply:

- scalar surface pressure at every native boundary VTU `PointData` point;
- three-component surface wall shear at every native boundary VTU `PointData`
  point;
- scalar pressure at every retained valid native volume VTU `PointData` point;
  and
- three-component velocity at every retained valid native volume VTU
  `PointData` point.

Read the exact public boundary and volume archive members bound by the candidate
support. Do not remesh, resample, deduplicate, triangulate into a new point set,
or silently reorder either support. Retain zero-based raw native point IDs so
the evaluator can prove complete, duplicate-free coverage.

The volume validity mask is benchmark-owned. A raw volume point is retained
when raw Float32 `avg(P) != 0.0` before normalization. Do not recompute the mask
from predicted pressure, replace it with a tolerance, or remove difficult
points. The primary volume reduction gives one unit weight to every retained
point.

At the scoring interface the relative-field bases are:

- pressure: `(P-p_inf)/q_inf` for both domains;
- wall shear: `tau_wall/q_inf`; and
- velocity: `U/|U_inf|`.

If a model predicts dimensional values, convert them deterministically and
describe that conversion. If it predicts these nondimensional quantities
directly, do not scale them a second time. The evaluator uses each case's own
freestream data to produce the dimensional MAE/RMSE diagnostics.

## 3. Chunk without changing the metric

Native cases are large and may be processed in bounded-memory chunks. Chunking
is an execution detail, not a change of support or scoring. Across one case,
the chunk raw-ID streams must form one exact partition of the expected native
support with no gap, overlap, duplicate, omission, or extrapolation.

For scalar values, accumulate the weighted error and truth sums over the entire
case. For a vector value, one point weight multiplies
`||prediction-truth||^2` and `||truth||^2`; do not calculate a separate relative
L2 for each vector component and average them. The complete-case relative L2 is

`100 * sqrt(sum_i(w_i * ||prediction_i-truth_i||^2) / sum_i(w_i * ||truth_i||^2))`.

Use the published native nodal dual area for the two primary surface metrics.
Also retain the equal-native-node surface relative-L2 diagnostics. Use `w_i=1`
for retained volume points; no volume dual-volume or cell-volume weighting is
part of this candidate.

Accumulate additive numerators, denominators, counts, and weight sums across
chunks. Calculate ratios, square roots, fractions, MAE, RMSE, and R2 only after
the complete case closes. Never average chunk-local metrics. After every case
is complete, macro-average its derived field metric equally across the selected
case set; a 900-case split does not give larger native meshes more weight than
smaller ones.

Incomplete cases fail closed. Do not insert zeros, NaNs, imputed metrics, or
copied truth to make an aggregate appear complete.

## 4. Derive forces and exact pitching moment

Use the frozen evaluator to integrate the same native surface prediction used
for the field metrics. Do not train or submit a separate force head as a
replacement for integration, and do not hand-edit the evaluator-produced load
values.

The load stream consumes `Cp` and `Cf` on the training-`q_inf` basis. For each
case it:

1. integrates pressure and viscous force exactly to degree one over the native
   polygon's ordered fan;
2. multiplies the integrated numerators by `q_inf/qRef` exactly once;
3. normalizes force by `areaRef` and rotates body force into drag/lift using the
   case angle; and
4. integrates body-y moment about `forcesCoR` with the exact degree-two
   linear-triangle mass matrix, then normalizes by `areaRef * chordRef`.

The prediction and native truth must use the identical convention. A perfect
surface prediction must therefore give perfect load metrics. No selected case
may be excluded, downweighted, or assigned a relaxed tolerance.

`Cd` and `Cl` use equal-case R2 and contribute to the composite. Their MAE and
the exact pitching-moment MAE are required diagnostics. `CM` does not receive a
second composite weight. The released `force_mom` CSVs, including documented
surface-integrated corrections, are release provenance and reference records;
the candidate scoring truth remains direct native-surface reintegration.

The numerical reasoning and retained audit scope are described in
[`EXACT_MOMENT_AUDIT.md`](EXACT_MOMENT_AUDIT.md). Participants should consume
the frozen evaluator product, not reproduce a separate interpretation from the
audit prose.

## 5. Produce the exact Cp and velocity profiles

Profiles are deterministic derivatives of the complete native predictions.
They are not independent model outputs and must not be replaced with values
sampled on a participant-defined mesh.

### Surface Cp rows A-J

Every case includes source rows `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`,
and `J`. A plane cut can contain several disconnected physical graphs. Retain
the evaluator's branch, graph, segment, and vertex identities and physical
segment lengths. Do not join disconnected branches, sort all points into one
polyline, or create an edge across a gap.

Within one case, truth is centered independently in every connected graph.
Graph arc-length-weighted SSE and SST are pooled across rows A-J, producing one
case Cp R2. Complete-case Cp R2 values are then macro-averaged equally.

### Volume velocity stations

Use exactly `B.2`, `B.3`, `C.1`, `C.2`, and `C.3`, in that order. Each station
has 801 requested rows. The evaluator publishes the requested coordinates,
native support mapping, validity mask, physical polyline arc-length weights,
and explicit gap intervals.

Do not change the validity mask, fill invalid values, or bridge an invalid run
with a trapezoidal edge. Invalid rows remain represented, their prediction
entries remain NaN as required by the profile format, and their weights are
zero. Truth is centered within each station; station-weighted SSE and SST are
pooled within the case before complete-case R2 is macro-averaged equally.

The participant profile NPZ chunks are prediction-only. They include the exact
alignment/topology support required by
[`native-profile-format-v1.json`](native-profile-format-v1.json), but no truth
or reference arrays. Hidden truth is joined only by the candidate validator
from the separately supplied, benchmark-owned local release. The evaluator
must emit every selected case and all ten Cp plus five velocity station aliases.

### Inactive compact profile-v2 alternative

The additive compact candidate is a maintainer-local alternative for the
profile payload only. Its evaluator-owned release holds cut geometry,
connected-graph topology, branch boundaries, sample placement, velocity
validity masks, weights, alignment, and hidden truth outside the submission.
Each participant case artifact contains exactly the two prediction arrays
defined by
[`native-profile-format-v2.json`](native-profile-format-v2.json):
`cp_q_delta` and `velocity_speed_over_u_inf`. It contains no evaluator support
or truth arrays. The NPZ uses deterministic level-9 ZIP Deflate. The velocity
member losslessly unsigned-delta-codes and byte-shuffles the exact little-endian
float32 bit patterns before compression; decoding restores every submitted bit
and therefore does not change values or scores.

The selected Cp support uses at most 128 points per physical connected graph,
placed uniformly in native physical arc length. Cp values are fixed-int16
quantized and delta encoded at evaluator-owned retained-branch boundaries.
Velocity values are scalar `float32` speed-over-freestream predictions for all
and only the evaluator-owned valid rows at the same five stations. They are
stored as the `uint8` byte transform defined by the contract so browsers can
decode the ordinary Deflate NPZ without evaluator support or a native LZMA
runtime. The
selection evidence is recorded in
[`compact-cp-representation-decision-v1.json`](compact-cp-representation-decision-v1.json)
and
[`compact-cp-representation-audit-v1.md`](compact-cp-representation-audit-v1.md).

This alternative does not compact or replace the canonical native surface
prediction. Full native Cp and wall-shear field metrics, forces, and pitching
moment remain unchanged. A float32, plot-only projection of truth for all
1,355 unique cases across the eight official case sets is publicly bound for
dashboard comparison by
[`public-compact-profile-truth-binding-v1.json`](public-compact-profile-truth-binding-v1.json),
but it omits scoring weights and is not an evaluator release. The v2 candidate
therefore remains unbound for submission intake and inactive; access to either
the public plot bundle or a local support directory changes none of those
lifecycle facts.

## 6. Include regional reports only when complete

Regional diagnostics answer where a method is performing well or poorly using
the same predictions and sufficient statistics. They require no new model
inference and have weight `0.0`.

The surface partition contains:

- `nacelle_installation_envelope_proxy`;
- `inboard_high_lift_envelope_proxy`;
- `outboard_high_lift_envelope_proxy`; and
- `fuselage_tail_and_remaining`.

The volume partition contains:

- `near_airframe_sdf_band`;
- `aft_airframe_wake_envelope_proxy`;
- `near_aircraft_flow_envelope`; and
- `farfield_and_remaining`.

These are deterministic geometric envelopes. They must not be described as
true CAD-part labels, slat/flap segmentation, boundary-layer detection, vortex
identification, or a flow-topology wake classifier.

Regional statistics use dual area on the surface and equal valid nodes in the
volume. Their global reconstruction must agree with the corresponding official
field sufficient statistics. They never alter the field metrics, component
scores, overall score, acceptance status, or ranking.

`include_regional_diagnostics` in the package configuration should be `true`
only when the evaluator produced the exact complete-selected-set aggregate. If
that report is unavailable, set it to `false` and omit the report rather than
fabricating or partially aggregating it. The authoritative definition is
[`regional-diagnostics-v2.json`](regional-diagnostics-v2.json). The v2 report
uses equal-case whole-support-normalized RMSE as the primary volume-region
comparison while retaining local relative L2 and R2 as zero-weight diagnostics.

## 7. Record method and discretization provenance

Every schema-v3 package includes a `fluidsbench-method-v1` methodology record
covering all four required outputs in
[`methodology-contract.json`](methodology-contract.json). It must describe:

- every architecture component and its role, distinguishing surface and volume
  models when they are separate;
- exact total learned parameters and exact parameters updated by the submitter;
- all input features, direct or deterministically derived outputs, key
  hyperparameters, normalization, preprocessing, and sampling;
- every training or upstream stage, its data/procedure, seeds, and measured
  compute as applicable;
- the raw-file SHA-256 of every checkpoint file actually loaded, including
  shards, adapters, or separate surface/volume checkpoints; and
- measured complete-selected-set inference hardware, maximum concurrency,
  wall time excluding queue delay, aggregate device time, and whether timing
  includes preprocessing and native-support mapping.

The exact checkpoint digests identify local bytes; they do not require model
weights to be uploaded. Sharing complete native prediction fields is optional,
but omission of those large fields does not relax the evaluator receipt,
metric, profile, or methodology requirements.

`discretization.json` and ordered `discretization/cases.jsonl` separately record
the actual training and inference representations: input/support sizes,
sampling, supervision, direct outputs, and mapping back to the canonical native
supports. Methodology and discretization must describe the same pipeline.

Start from
[`../../examples/hiliftaeroml-v3-candidate/package-config.template.json`](../../examples/hiliftaeroml-v3-candidate/package-config.template.json).
Every `__REPLACE_...__` or `__UNRESOLVED_HILIFTAEROML_...__` string is a real
blocker, not a sample value. Do not invent a revision, release hash, or metric
to replace one.

## 8. Close the evaluator-native case set

Before package assembly, the native evaluator must produce every case in the
selected ordered case set. Its generic case-set products bind the case-set ID,
case-set SHA-256, case count, and order. The expected product family includes:

- a `hiliftaeroml-transolver-case-set-aggregate-artifacts-v1` artifact
  manifest;
- `hiliftaeroml-transolver-case-set-results-v1` aggregate results;
- `hiliftaeroml-transolver-case-set-case-metrics-v1` ordered case metrics;
- `hiliftaeroml-transolver-case-set-auxiliary-summary-v1` auxiliary/profile
  summary;
- one `hiliftaeroml-transolver-case-receipt-v1` per case; and
- the complete report-only regional aggregate when requested.

Each case receipt closes the chain from contracts through surface and volume
summaries to the exact artifact descriptors consumed by aggregation. Use the
evaluator's producer and reducer; do not hand-author these records. A swapped
case, stale summary, missing receipt, mismatched size/path/hash, incomplete
load, or wrong case order blocks assembly.

Surface and volume output roots may differ. This permits a corrected surface
replay to be paired with a previously completed, unchanged volume replay while
both remain independently receipt-bound. The assembler still requires both
domains for every selected case.

## 9. Inspect blockers, assemble, and dry-run

Work from the root of `fluidsbench-submission`. Copy and fill the candidate
configuration outside the checked-in example directory. First inspect all
release, configuration, truth, aggregate, and per-case provenance gates without
writing a package:

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/case-set-aggregate \
  --native-surface-outputs /path/to/native/surface-outputs \
  --native-volume-outputs /path/to/native/volume-outputs \
  --native-receipts /path/to/native/case-receipts \
  --candidate-profile-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  --list-blockers
```

Do not bypass reported blockers. In particular, access to a local inactive
truth release does not publish that truth, freeze the evaluator, approve a
method, or open submissions.

When blocker inspection reports that full assembly validation is ready, create
a new local output directory. The path must not already exist:

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/case-set-aggregate \
  --native-surface-outputs /path/to/native/surface-outputs \
  --native-volume-outputs /path/to/native/volume-outputs \
  --native-receipts /path/to/native/case-receipts \
  --candidate-profile-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  --output /path/to/local/hiliftaeroml-my-method-v1
```

If both domains share one parent, `--native-outputs` is the backward-compatible
shorthand for the two domain-specific options. The assembler writes atomically
and creates participant-owned schema-v3 files only. It does not create
`approval`, `maintainer-validation.json`, or
`prediction-artifact-checks.json`.

Validate the package against the same authorized local candidate truth:

```bash
python scripts/validate_submission.py \
  --candidate-dry-run \
  --candidate-profile-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  /path/to/local/hiliftaeroml-my-method-v1
```

A successful command reports candidate dry-run validity only. It does not
grant official acceptance, benchmark-owner approval, contributor-stage
eligibility, or leaderboard visibility. Do not use `--contributor-stage` for
this closed workflow.

### Optional local compact-v2 exercise

Twenty-three retained real-surrogate compact packages comprise eleven Transolver
results and twelve GeoTransolver results. Together they cover Full, Scarce, the
three fixed-AoA splits, Super scarce, Geometry scarce, Geometry super scarce,
Geometry, and all three out-of-distribution splits (AoA, deflection, and stall).
The public plot-only truth separately covers every official case set. A maintainer with the
authorized native-profile truth and complete native outputs for the target case
set can materialize an evaluator-owned scoring support release locally:

```bash
python scripts/materialize_hiliftaeroml_compact_profile_support.py \
  --submission-spec benchmark-specs/hiliftaeroml/submission-spec.json \
  --split benchmark-specs/hiliftaeroml/splits/full.json \
  --surface-outputs-root /path/to/native/surface/per-case/outputs \
  --volume-outputs-root /path/to/native/volume/per-case/outputs \
  --source-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  --output-root /authorized/local/hiliftaeroml-compact-profile-support-v2-candidate
```

The `--split`, `--surface-outputs-root`, and `--volume-outputs-root` groups may
be repeated in matching order. Maintainers can instead reconstruct support for
all eight official case sets without prediction outputs from the exact frozen
authority already bound into the native-profile truth release:

```bash
python scripts/materialize_hiliftaeroml_compact_profile_support.py \
  --submission-spec benchmark-specs/hiliftaeroml/submission-spec.json \
  --split benchmark-specs/hiliftaeroml/splits/full.json \
  --split benchmark-specs/hiliftaeroml/splits/geometry_scarce.json \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_4.json \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_12.json \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_22.json \
  --split benchmark-specs/hiliftaeroml/splits/aoa.json \
  --split benchmark-specs/hiliftaeroml/splits/deflection.json \
  --split benchmark-specs/hiliftaeroml/splits/stall.json \
  --prerequisite-authority-index /authorized/local/hiliftaeroml-native-profile-truth-authority-v1.json \
  --source-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  --output-root /authorized/local/hiliftaeroml-compact-profile-support-v2-candidate
```

That path validates the authority against the truth-release binding, reads no
surrogate prediction output, applies the same limit of 128 Cp points per
physical graph, and stores every overlapping physical case once. Its receipt
records how the authority record/payload hashes occupy the support format's
four legacy source-hash provenance slots. The completed two-build all-case
evidence is
[`compact-profile-all-case-support-validation-v1.json`](compact-profile-all-case-support-validation-v1.json).
It is the selected current local candidate support binding, and its ten-preview
rebind is recorded in
[`compact-profile-all-case-support-rebind-v1.json`](compact-profile-all-case-support-rebind-v1.json).
Both records remain intentionally non-activating.

The output is benchmark/evaluator-owned local support and must remain outside
the participant package. Use it in place of, not alongside,
`--candidate-profile-truth-release` when inspecting blockers or assembling:

Compact assembly also requires a top-level `compact_evaluation` object in the
package configuration with the exact compact assembler `command` and its
RFC3339 `generated_at` time. Keep the native-v1 `evaluation` object unchanged;
the assembler selects between the two records strictly by profile mode.

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/case-set-aggregate \
  --native-surface-outputs /path/to/native/surface/per-case/outputs \
  --native-volume-outputs /path/to/native/volume/per-case/outputs \
  --native-receipts /path/to/native/case-receipts \
  --candidate-compact-profile-support-release /authorized/local/hiliftaeroml-compact-profile-support-v2-candidate \
  --output /path/to/local/hiliftaeroml-my-method-compact-v2

python scripts/validate_submission.py \
  --candidate-dry-run \
  --candidate-compact-profile-support-release /authorized/local/hiliftaeroml-compact-profile-support-v2-candidate \
  /path/to/local/hiliftaeroml-my-method-compact-v2
```

The assembler reports the completed package size but has no aggregate byte
ceiling, because the official splits have materially different case counts.
The bounded per-file/archive checks and the validator remain in force. Passing
them remains only local candidate evidence; it does not publish support,
approve the evaluator, or open submissions. The native profile-v1 commands
above remain the retained workflow.

## 10. Package boundary while submissions are closed

The schema-v3 directory contains the normal participant envelope:

- `submission.json`;
- `metrics/cases.json`;
- `discretization.json` and `discretization/cases.jsonl`;
- `profiles/index.json` and deterministic prediction-only profile chunks;
- `evaluation-evidence.json`; and
- optional complete `regional-diagnostics.json`.

Do not add hidden truth, native truth fields, benchmark-owned approval files,
or fabricated release metadata. Do not edit specifications, schemas, evaluator
code, scoring support, workflows, or an existing result inside a participant
package.

For compact-v2 packages, each per-case NPZ has exactly `cp_q_delta` and
`velocity_speed_over_u_inf`; the latter is the contract-defined lossless
float32-bit transform stored as `uint8`. Evaluator-owned support remains in the
separately bound local release. Full native surface/volume evaluation products
and their receipt chain remain required outside the portable package assembly
inputs.

While `submissions_open` is `false`, retain the package locally or share it
directly with the benchmark owner for coordinated implementation review. Do
not open a leaderboard-result pull request and do not treat a screenshot or
locally rendered score as a leaderboard entry. No private leaderboard is
planned in this workflow.

If HiLiftAeroML is later opened, follow the then-current owner-published guide
and rebuild against the final frozen evaluator and releases. This closed
candidate guide does not pre-authorize that future submission.
