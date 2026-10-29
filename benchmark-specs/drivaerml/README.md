# DrivAerML submission contract

Use the public DrivAerML release and eight prescribed splits described below.
See the [dataset index](../README.md) for submission availability and the
[activation checklist](ACTIVATION_CHECKLIST.md) for the support-release review record.

## Prepare a result

Start with the [resumable quickstart](QUICKSTART.md), then consult the [participant guide](PARTICIPANT_GUIDE.md) and
[packaging example](../../examples/drivaerml-v3-candidate/README.md). The complete native evaluator requires Linux and the
published dependency pins. Keep the supplied release bindings and replace every participant placeholder before packaging.

| Scope                | Required predictions and profiles                                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `surface_and_volume` | Native surface pressure/wall shear and volume pressure/velocity; four Cp cuts and 16 velocity profiles.                                                                             |
| `surface_only`       | Native surface pressure/wall shear and four Cp cuts. No volume files or velocity series; omitted components contribute zero with fixed weights retained (maximum overall score 60). |

Both scopes use complete native coverage, evaluator-derived forces, actual methodology/checkpoint records, and the same
immutable release gates. Do not insert dummy values for unavailable metrics or renormalize score weights.
The [specification](submission-spec.json) is authoritative; the [activation checklist](ACTIVATION_CHECKLIST.md) tracks intake gates.

## Public data and split use

The source dataset is `neashton/drivaerml` at immutable revision
`7a5c0948ce27be709b1116a3a190f806e7a8f79f`. Each file in [`splits/`](splits/)
contains the exact ordered training, validation, and test case IDs from the
owner-published manifest. Test fields are public but evaluation-only: they must
not be used for fitting, tuning, checkpoint selection, manual selection, or
preprocessing statistics.

The native surface-area weights are now public in that same pinned Hugging Face
release. The authoritative `surface_cell_areas/manifest.json` has SHA-256
`1401c7e80bd86f3aa2d640289db9b088ce1e0825327e18eeb1ab2852de04323e`
and binds 484 `run_N/boundary_cell_area_N.npy` payloads to 4,159,517,910 raw
boundary polygons. FluidsBench uses those published arrays directly and does
not regenerate areas during evaluation.

<details>
<summary>Native-field requirements, force/profile reductions, masks, methodology, and score components</summary>

For every selected case:

1. Read every native surface polygon from `run_N/boundary_N.vtp` as `CellData`.
   Predict `pMeanTrim` and all three components of `wallShearStressMeanTrim` in
   raw VTK cell order. Use `run_N/boundary_cell_area_N.npy` for the primary
   area-weighted reductions; the equal-polygon values are mandatory secondary
   results. Verify each payload against the public manifest and its case record;
   never apply it to a reordered, remeshed, triangulated, or STL surface.
2. For `surface_and_volume`, reconstruct the actual `run_N/volume_N.vtu` by byte-concatenating the exact
   ordered part list in [`proposal/native-source-pin.json`](proposal/native-source-pin.json).
   Ten cases have a `.02.part`, so code must not assume two parts. Predict
   `UMeanTrim` and `pMeanTrim` for every native `CellData` cell in raw VTK order.
   Score volume fields with one equal weight per native cell; no geometric
   cell-volume array or volume-weighted secondary metric is required.
3. Inference may process native entities in bounded-memory chunks. Each chunk
   must retain the raw native cell IDs and the chunks must form one complete,
   duplicate-free partition of the case. Sum additive numerators, denominators,
   counts, and weights across chunks; calculate one case metric only after the
   complete case is represented. Never average chunk-local L1, L2, MAE, or RMSE
   values.
4. Run the reference evaluator locally to derive per-case `Cd`, `Cl`,
   `CmPitch`, and report-only `Clf`/`Clr` by integrating the predicted surface
   pressure and wall shear with the defined constant-reference convention.
   Submit those coefficients and their required metrics in the result JSON. The
   per-case `metrics/cases.json` entry uses `force_coefficients` with `cd`,
   `cl`, `cm_pitch`, `clf`, and `clr`; the last two remain report-only. The
   authoritative truth is `force_mom_constref_all.csv`; per-case
   `run_N/force_mom_constref_N.csv` files are convenience mirrors that must
   replay the matching aggregate row.
5. Use the same evaluator to extract the diagnostics defined in
   [`drivaerml-diagnostics-v10.json`](drivaerml-diagnostics-v10.json), with the
   [profile scoring contract](profile-scoring-v2.json): 16 AutoCFD5
   velocity lines on the published 10 mm grid and four continuous Cp cuts:
   upperbody centreline (`y=0`), underbody centreline (`y=0`), sidewall
   (`z=0.15 m`), and front-left wheelhouse (`y=-0.6 m`). The current
   velocity metric is global R2 with equal total
   weight for every case and line and normalized trapezoidal arc-length weight
   within each line. Continuous Cp-cut R2 is a separate ranked component with
   equal total weight for every case and cut and normalized trapezoidal
   arc-length weight within each cut. Unsupported gaps are never bridged.
   The corresponding RMSE values are report-only diagnostics. Submit the complete
   evaluator-produced coordinate and prediction arrays for all 16 velocity
   profiles for `surface_and_volume` and all four Cp cuts in either scope, in the normal FluidsBench profile JSON. `surface_only` supplies no velocity-profile series.

The participant runs the frozen reference evaluator locally and submits its
derived JSON values. Sharing the complete native prediction fields is optional
under the repository-wide reproducibility policy. A maintainer may optionally
audit a shared, revision-pinned artifact, but native-field sharing and
maintainer recomputation are neither participant requirements nor activation
gates.

Every new schema-v3 DrivAerML result also includes the mandatory
`fluidsbench-method-v1` methodology record and the required outputs in
[`methodology-contract.json`](methodology-contract.json). It binds a detailed
multi-component architecture and exact total/submitter-trainable parameter
counts, input and output definitions, data handling, every submitter or upstream
training stage, compute for every submitter-performed stage, every exact loaded
checkpoint-file SHA-256, and measured inference compute. The record is required
even when public code and model weights are not shared. This makes the method and
checkpoint identity reviewable without turning optional artifact publication
into a ranking or approval requirement. See
[`PARTICIPANT_GUIDE.md`](PARTICIPANT_GUIDE.md) and the
[`methodology.example.json`](../../examples/drivaerml-v3-candidate/methodology.example.json)
shape example.

Velocity validity is benchmark-owned support. Only coordinates confirmed to be
inside the morphed solid or outside the released fluid domain may be excluded.
Every other coordinate must map deterministically to a native volume cell or
the affected line, and therefore that case's ranked velocity component, is
unavailable. Every row remains explicit: the evaluator never snaps, silently
omits, interpolates, extrapolates, or bridges across an excluded point. The
`1e-3` steradian polyhedron test is an angular numerical tolerance,
separate from the `1e-6 m` boundary tolerance; intermediate solid-angle results
fail closed. The published all-case support, explicit unsupported rows and
segment breaks, 10 mm grid, and those tolerances are scientifically approved.
Velocity truth is exact zeroth-order native
`CellData`, so visible stair steps are expected and must not be smoothed or
interpolated in truth or scoring.

The primary leaderboard definition has nine components: four global fields,
three independently ranked field-integrated coefficients, the velocity
profiles with weight 0.15, and the continuous Cp cuts with weight 0.10. The
four field errors use `clip(100 * (1 - error/cap), 0, 100)` with fixed caps of
15% surface pressure, 20% wall shear, 12% volume velocity, and 15% volume
pressure. Force and profile R2 values use `100 * clip(R2, 0, 1)`. The component
weights preserve a 50% field, 25% force, and 25% profile split. `Clf`, `Clr`,
and all RMSE values for the selected scope remain mandatory report-only diagnostics and receive no
duplicate composite weight.

The complete machine-readable source of truth is
[`submission-spec.json`](submission-spec.json). Implementation evidence is indexed in [`evidence/README.md`](evidence/README.md);
history and paper-parity definitions remain in [`proposal/`](proposal/). The
opening requirements and their resolved status are tracked in
[`ACTIVATION_CHECKLIST.md`](ACTIVATION_CHECKLIST.md).

</details>

## Scientific approval and published support

DrivAerML is open for all eight prescribed splits. Use the [quickstart](QUICKSTART.md)
and [participant guide](PARTICIPANT_GUIDE.md) with the official
`drivaerml-native-support-v3` release bound in `submission-spec.json`.

The [opening decision](OWNER_OPENING_DECISION.md) records the owner approval.
Geometry-relative diagnostics are report-only, with zero scoring weight.

The [scientific approval record](evidence/owner-scientific-approval-2026-08-28.json)
binds the dataset, native-array and force evidence, fixed and relative support,
and real `run_419` validation. Surface arc length remains the Cp scoring
coordinate; native segment-midpoint streamwise x is for display only.

Implementation evidence is indexed in [evidence/README.md](evidence/README.md).
It includes the all-484 native-surface force replay and the equal-cell volume
audit over 978 pinned segments and 68,949,662,110 cells. The specification and
opening decision govern submissions.
