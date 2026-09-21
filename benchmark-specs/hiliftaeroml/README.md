<a id="hiliftaeroml-closed-candidate-submission-contract"></a>

# HiLiftAeroML: prepare a candidate result

**Public submissions are closed.** Follow the [participant guide](PARTICIPANT_GUIDE.md) and
[compact-v2 packaging example](../../examples/hiliftaeroml-v3-candidate/README.md) for a local, owner-coordinated dry run.
You need complete native surface and volume predictions, evaluator aggregates and case receipts, actual methodology/spatial
records, and authorized compact-v2 scoring support. Public plot truth alone is insufficient.

Every case requires four fields: surface pressure and wall shear, volume pressure and velocity. There is no surface-only option.
Use the evaluator's loads and compact-v2 profiles; regional reports are optional and have zero score weight.
Check the [activation checklist](CANDIDATE_ACTIVATION_CHECKLIST.md) before treating any candidate as ready for intake.

<details>
<summary>Current candidate status and the 23 registered real-inference previews</summary>

HiLiftAeroML has a participant-shaped FluidsBench schema-v3 contract, but it is
not open for submissions. The machine-readable
[`submission-spec.json`](submission-spec.json) currently has
`submissions_open: false`. A candidate package may be used for local or
benchmark-owner-coordinated dry runs only; it is not accepted, citable as an
official result, or evidence of benchmark approval. Twenty-three exact
real-inference packages are additionally registered by path and hashes as
public `pre_release_reference` rows: eleven Transolver results and twelve
GeoTransolver results covering every represented Table 5 split, including the
Transolver out-of-distribution deflection and GeoTransolver Scarce results. They allow the evaluator,
plots, and leaderboard integration to be exercised before intake opens. Their
citation and promotion eligibility are all false.

No private leaderboard is created or operated by this candidate workflow.
`--candidate-dry-run` validates files locally and does not register arbitrary
candidates. Each permitted preview has a `compact-profile-*-validation-v1.json`
record in this directory and is bound in code to its exact submission and
deterministic compact-archive identities.

The candidate brings the thoroughly tested DrivAerML packaging pattern to
HiLiftAeroML while retaining HiLiftAeroML's native scientific definitions:

- complete native-point surface and volume coverage with bounded-memory
  chunking;
- area-weighted surface and equal-valid-node volume reductions;
- exact surface-force and pitching-moment integration;
- native pressure cuts and velocity profiles; and
- optional, zero-weight surface and volume regional reports.

The detailed workflow is in [`PARTICIPANT_GUIDE.md`](PARTICIPANT_GUIDE.md).
The activation work that remains is tracked separately in
[`CANDIDATE_ACTIVATION_CHECKLIST.md`](CANDIDATE_ACTIVATION_CHECKLIST.md).

</details>

## Official evaluation labels and case sets

Each package declares exactly one `split_id`. The 14 accepted labels map to
eight exact ordered evaluation case sets:

| `split_id`              | Display label         | Evaluation cases | Case-set ID            |
| ----------------------- | --------------------- | ---------------: | ---------------------- |
| `full`                  | Full                  |              360 | `caseset-ac791749e527` |
| `medium`                | Medium                |              360 | `caseset-ac791749e527` |
| `scarce`                | Scarce                |              360 | `caseset-ac791749e527` |
| `super_scarce`          | Super scarce          |              360 | `caseset-ac791749e527` |
| `geometry`              | Geometry              |              360 | `caseset-53990ea68fa6` |
| `geometry_medium`       | Geometry medium       |              360 | `caseset-53990ea68fa6` |
| `geometry_scarce`       | Geometry scarce       |              360 | `caseset-53990ea68fa6` |
| `geometry_super_scarce` | Geometry super scarce |              360 | `caseset-53990ea68fa6` |
| `single_aoa_4`          | AoA 4                 |               36 | `caseset-7a743a20b3bd` |
| `single_aoa_12`         | AoA 12                |               36 | `caseset-02fc12ff3494` |
| `single_aoa_22`         | AoA 22                |               36 | `caseset-85ecccd9ccda` |
| `aoa`                   | AoA extrapolation     |              900 | `caseset-29693354ed8a` |
| `deflection`            | Deflection            |              360 | `caseset-c0ecb14de138` |
| `stall`                 | Stall                 |              723 | `caseset-804491c8956e` |

The union contains 1,355 unique physical cases. Labels that share a case set
still represent distinct declared training regimes; they are not interchangeable
names. The files in [`splits/`](splits/) bind the ordered evaluation cases.
Participants must use the matching dataset-owner training definition and
describe it in the methodology record. Public evaluation fields are
evaluation-only and must not influence fitting, normalization statistics,
checkpoint selection, or manual model selection.

## Native fields and reductions

<details>
<summary>Read native fields and reductions requirements</summary>

The closed candidate requires all four logical predictions for every case in
the selected evaluation set:

| Support                                   | Required prediction        | Primary spatial weight           |
| ----------------------------------------- | -------------------------- | -------------------------------- |
| all native boundary `PointData` points    | scalar surface pressure    | published native nodal dual area |
| all native boundary `PointData` points    | three-component wall shear | published native nodal dual area |
| retained native volume `PointData` points | scalar volume pressure     | one per valid point              |
| retained native volume `PointData` points | three-component velocity   | one per valid point              |

The volume validity rule is the raw Float32 `avg(P) != 0.0` test before
normalization. There is no dual-volume or cell-volume-weighted secondary
metric. Surface equal-native-node relative L2 is retained as a mandatory
secondary diagnostic for both surface fields.

For one case, relative L2 is

`100 * sqrt(sum_i(w_i * ||prediction_i - truth_i||^2) / sum_i(w_i * ||truth_i||^2))`.

For vector fields one point weight multiplies the complete squared
three-component magnitude. A chunk contributes additive sufficient statistics;
division and the square root happen only after the complete case closes. The
benchmark value is then the equal arithmetic mean of the complete-case values
in the selected split. It is not a pooled all-point ratio and never an average
of chunk-local metrics.

Relative metrics use `(P-p_inf)/q_inf`, `tau_wall/q_inf`, and `U/|U_inf|`.
The companion MAE and RMSE diagnostics are converted back to their dimensional
bases per case before equal-case macro aggregation.

</details>

## Loads and composite score

<details>
<summary>Read loads and composite score requirements</summary>

The evaluator derives loads from the same complete native surface prediction;
participants do not provide an independently fitted force value. Pressure and
viscous force use exact degree-one integration on each native polygon's ordered
fan. Input `Cp` and `Cf` are on the training-`q_inf` basis, and the evaluator
multiplies each integrated numerator by `q_inf/qRef` exactly once before
normalizing by the case `areaRef`.

Pitching moment is the exact degree-two integral of
`(r-forcesCoR) cross traction` on the same ordered fan and is normalized by
`qRef * areaRef * chordRef`. `Cd` and `Cl` are ranked through equal-case R2.
`c_drag_mae`, `c_lift_mae`, and exact `c_pitch_mae` are required diagnostics;
pitching moment has no duplicate composite weight. The released force/moment
tables are reference and provenance records. Scored truth is canonical native
surface reintegration, so participants must not substitute table values for
their prediction-side integration.

The bounded candidate composite is:

| Component                      | Transform                            | Weight |
| ------------------------------ | ------------------------------------ | -----: |
| surface pressure relative L2   | `clip(100 * (1 - error/15), 0, 100)` |   0.15 |
| surface wall-shear relative L2 | `clip(100 * (1 - error/20), 0, 100)` |   0.10 |
| volume velocity relative L2    | `clip(100 * (1 - error/12), 0, 100)` |   0.15 |
| volume pressure relative L2    | `clip(100 * (1 - error/15), 0, 100)` |   0.10 |
| `Cd` R2                        | `100 * clip(R2, 0, 1)`               |   0.15 |
| `Cl` R2                        | `100 * clip(R2, 0, 1)`               |   0.10 |
| velocity-profile R2            | `100 * clip(R2, 0, 1)`               |   0.15 |
| Cp-cut R2                      | `100 * clip(R2, 0, 1)`               |   0.10 |

This preserves 50% field, 25% force, and 25% profile weight. The specification,
not this prose summary, is the numerical source of truth.

</details>

## Profile diagnostics

<details>
<summary>Read profile diagnostics requirements</summary>

Every complete case has ten pressure-cut rows, `A` through `J`. Cp scoring
retains every disconnected physical cut graph, centers truth independently
within each graph, pools graph weighted SSE and SST within the case, and then
macro-averages complete-case R2 equally. Physical connected-cut arc length is
the weight; disconnected graphs must never be joined by a new scoring edge.

Volume velocity uses exactly five stations: `B.2`, `B.3`, `C.1`, `C.2`, and
`C.3`. Each station retains 801 requested rows, its authoritative validity
mask, physical polyline arc-length weights, and explicit gaps. Truth is centered
within each station; station SSE and SST are pooled within the case before the
equal-case macro average. Invalid runs remain serialized with zero weight and
must not be interpolated or bridged in evaluator support; participant arrays
contain only the valid rows in evaluator-defined order.

Compact profile-v2 is the official and sole participant-facing profile
representation. Earlier HiLift profile package formats are not accepted. The current
`submission-spec.json` selects the unchanged compact contract. Its historical
`-candidate` identifiers and contract metadata remain frozen to preserve the
validated support, all 23 package archives, and their recorded scores.
Representation selection does not publish support or attest the implementation.
Participant artifacts contain predictions only; evaluator-owned alignment,
topology, weights, masks, and truth remain outside the package.

The website repository now publishes a separate, checksum-bound float32
projection of the Cp and velocity truth for all 1,355 unique cases in all
eight official case sets. Case artifacts are deduplicated across overlapping
splits, while each case-set index preserves its exact official order. That
derivative contains neither scoring weights nor the evaluator's lossless truth
and cannot be used for metric recomputation. Its exact cross-repository binding
is
[`public-compact-profile-truth-binding-v1.json`](public-compact-profile-truth-binding-v1.json).

### Official compact profile-v2 representation

Compact profile-v2 removes geometry and topology arrays from the participant
profile artifacts. The evaluator owns that immutable support,
joins it outside the submission, and accepts exactly two prediction-value
arrays per case: quantized Cp deltas and scalar Float32 velocity-profile
values. The exact velocity float32 bits use a lossless unsigned-delta and byte
shuffle before browser-compatible level-9 ZIP Deflate. No truth, reference,
geometry, topology, mask, weight, or alignment array is included in the
participant artifact.

The frozen Full360 study selected at most 128 samples per physical connected
Cp graph, placed uniformly in physical graph arc length. The all-split support
replay additionally confirms that a closed native contour is unwrapped by
repeating its first vertex, thereby retaining its existing closing segment
without creating a new edge; this does not change any Full360 support bytes.
Velocity retains the
five stations and submits all and only evaluator-selected valid rows as
scalar `float32` speed-over-freestream values; the on-disk `uint8` transform
round-trips every bit. This compact path changes only
Cp-cut and velocity-profile plotting/scoring payloads. Complete native-surface
Cp scoring, wall-shear scoring, force integration, and pitching-moment
integration continue to use the unchanged full native surface prediction.

The compact assembler does not impose an aggregate package-size ceiling:
package size scales with the official case count, while the strict per-file,
array-shape, archive-member, prediction-only, and release-binding safety checks
remain in force. Package size is reported in the assembly receipt. This does
not make the candidate published or active. The format and retained scientific
evidence are [`native-profile-format-v2.json`](native-profile-format-v2.json),
[`compact-cp-representation-decision-v1.json`](compact-cp-representation-decision-v1.json),
and
[`compact-cp-representation-audit-v1.md`](compact-cp-representation-audit-v1.md).
The prediction-free two-build materialization and compatibility evidence for
all 1,355 cases is
[`compact-profile-all-case-support-validation-v1.json`](compact-profile-all-case-support-validation-v1.json).
It is the selected current local candidate binding for all eight official case
sets; the package rebind and its ten independent A/B package checks are
recorded in
[`compact-profile-all-case-support-rebind-v1.json`](compact-profile-all-case-support-rebind-v1.json).
Neither record activates submissions.
The completed two-build size, identity, metric, and validation receipts are
[`compact-profile-full360-validation-v1.json`](compact-profile-full360-validation-v1.json),
[`compact-profile-aoa4-validation-v1.json`](compact-profile-aoa4-validation-v1.json),
[`compact-profile-aoa12-validation-v1.json`](compact-profile-aoa12-validation-v1.json),
[`compact-profile-aoa22-validation-v1.json`](compact-profile-aoa22-validation-v1.json),
[`compact-profile-super-scarce-validation-v1.json`](compact-profile-super-scarce-validation-v1.json),
[`compact-profile-geometry-scarce-validation-v1.json`](compact-profile-geometry-scarce-validation-v1.json),
and
[`compact-profile-geometry-super-scarce-validation-v1.json`](compact-profile-geometry-super-scarce-validation-v1.json),
[`compact-profile-geometry-validation-v1.json`](compact-profile-geometry-validation-v1.json),
[`compact-profile-ood-aoa-validation-v1.json`](compact-profile-ood-aoa-validation-v1.json),
[`compact-profile-ood-deflection-validation-v1.json`](compact-profile-ood-deflection-validation-v1.json),
and
[`compact-profile-ood-stall-validation-v1.json`](compact-profile-ood-stall-validation-v1.json).
The corresponding GeoTransolver receipts use the
`compact-profile-geotransolver-*-validation-v1.json` names and cover Full,
Scarce, AoA 4/12/22, Geometry, Geometry scarce, Geometry super scarce, Super scarce,
OOD AoA, OOD Deflection, and OOD Stall.
The retained evaluator implementation manifest predates this additive compact
preview and the v2 regional dashboard contract, and does not attest either.
Their implementation provenance therefore remains explicitly
`unbound_worktree_candidate` until a complete additive-source inventory and
immutable evaluator revision are frozen; preview registration does not
activate or implicitly revise that older attestation.

</details>

## Regional reporting

<details>
<summary>Read regional reporting requirements</summary>

[`regional-diagnostics-v2.json`](regional-diagnostics-v2.json) defines four
surface regions and four volume regions. They reuse the native predictions and
the primary field weights; no second inference is required. They report where
error is concentrated, but their weight is `0.0`, they do not change any
official metric, and they do not change the overall score.

The named regions are geometric proxies, not CAD-part or flow-topology labels.
Surface reports cover the nacelle-installation envelope, inboard high-lift
envelope, outboard high-lift envelope, and the remaining surface. Volume
reports cover the near-airframe signed-distance band, an aft wake envelope,
the near-aircraft flow envelope, and the remaining far field.

Regional output is optional while `required_for_new_submissions` remains
`false`. If supplied, it must cover the complete selected case set and
reconstruct the corresponding global sufficient statistics; a partial or
fabricated report is not accepted.

For volume-region comparisons, v2 adds an equal-case mean of each region's
RMSE normalized by that case's whole-volume truth RMS. This is the primary
dashboard view because it remains interpretable when local farfield pressure
is close to zero. Local regional relative L2 and regional R2 remain visible as
diagnostics. All regional values still have weight `0.0`; the official volume
metrics, component scores, weights, and overall score are unchanged.

</details>

## Closed schema-v3 workflow

Use the configuration template and instructions in
[`../../examples/hiliftaeroml-v3-candidate/`](../../examples/hiliftaeroml-v3-candidate/)
and [`PARTICIPANT_GUIDE.md`](PARTICIPANT_GUIDE.md). The assembler consumes the
native evaluator's complete case-set aggregate, separate surface and volume
output roots if necessary, case receipts, and the authorized local compact-v2
evaluator-support release. It writes a normal FluidsBench schema-v3 package and
refuses missing cases, placeholder values, stale hashes, unresolved release
bindings, or an existing output directory.

Validate a resulting directory with `scripts/validate_submission.py` in
`--candidate-dry-run` mode. A pass means only that the directory implements the
closed candidate contract. It does not open submissions, grant owner approval,
or create a leaderboard entry. Only the 23 separately registered, hash-bound
Transolver and GeoTransolver previews are visible as non-citable pre-release
references.

The machine-readable authorities are:

- [`submission-spec.json`](submission-spec.json) for metrics, weights, split
  aliases, and lifecycle status;
- [`methodology-contract.json`](methodology-contract.json) for required model
  outputs;
- [`native-profile-format-v2.json`](native-profile-format-v2.json) for the sole
  accepted prediction-only profile serialization;
- [`public-compact-profile-truth-binding-v1.json`](public-compact-profile-truth-binding-v1.json)
  for the non-scoring all-case browser-plot truth;
- [`regional-diagnostics-v2.json`](regional-diagnostics-v2.json) for optional
  report-only regions; and
- [`candidate-evaluator-release-binding.json`](candidate-evaluator-release-binding.json)
  for the fail-closed release hand-off.

The compact-v2 representation is official. Its evaluator-owned scoring support
is not yet public for every enabled case set, so publishing the plot-only
derivative does not open profile intake.
