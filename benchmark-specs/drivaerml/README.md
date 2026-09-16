# DrivAerML submission contract

The reviewed proposal is now the participant-facing candidate contract.
It uses the exact public DrivAerML release and the eight owner-published splits.
Neil Ashton, acting as dataset owner, approved the candidate's native-field,
force, velocity-profile, continuous-Cp, sampling, and tolerance decisions on
2026-08-28. Submissions remain closed only while the immutable production
release, multi-model sensitivity study, and independent full-submission dry run
are completed. Existing leaderboard rows remain non-rankable prototype fixtures.

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

For every selected case:

1. Read every native surface polygon from `run_N/boundary_N.vtp` as `CellData`.
   Predict `pMeanTrim` and all three components of `wallShearStressMeanTrim` in
   raw VTK cell order. Use `run_N/boundary_cell_area_N.npy` for the primary
   area-weighted reductions; the equal-polygon values are mandatory secondary
   results. Verify each payload against the public manifest and its case record;
   never apply it to a reordered, remeshed, triangulated, or STL surface.
2. Reconstruct the actual `run_N/volume_N.vtu` by byte-concatenating the exact
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
   [`drivaerml-diagnostics-v9.json`](drivaerml-diagnostics-v9.json): 16 AutoCFD5
   velocity lines on the candidate 10 mm grid and four continuous Cp cuts:
   upperbody centreline (`y=0`), underbody centreline (`y=0`), sidewall
   (`z=0.15 m`), and front-left wheelhouse (`y=-0.6 m`). Once the contract is
   activated, the ranked velocity metric will be global R2 with equal total
   weight for every case and line and normalized trapezoidal arc-length weight
   within each line. Continuous Cp-cut R2 is a separate ranked component with
   equal total weight for every case and cut and normalized native
   cut-intersection segment-length weight within each cut. The corresponding
   RMSE values remain report-only diagnostics. Submit the complete
   evaluator-produced coordinate and prediction arrays for all 16 velocity
   profiles and all four Cp cuts in the normal FluidsBench profile JSON.

The 209 discrete Cp probes are not part of the DrivAerML submission or scoring
contract. Participants do not submit probe outputs or probe-mapping support,
and no discrete-probe metric is calculated. The combined v8 registry and
existing 209-probe mapping artifacts are retained only as inactive,
non-normative research evidence. This exclusion does not apply to the four
continuous Cp cuts, which the participant's local frozen evaluator will derive
from its native surface `pMeanTrim` prediction. Participants provide no extra
Cp field in the VTP, but do include the derived continuous-cut series in their
submitted profile JSON.

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
candidate `1e-3` steradian polyhedron test is an angular numerical tolerance,
separate from the `1e-6 m` boundary tolerance; intermediate solid-angle results
fail closed. The published all-case support, explicit unsupported rows and
segment breaks, 10 mm grid, and those tolerances are scientifically approved.
The earlier proposed resolution and tolerance campaigns are no longer separate
activation gates; this is an owner decision and not a claim that those
superseded experiments ran. Velocity truth is exact zeroth-order native
`CellData`, so visible stair steps are expected and must not be smoothed or
interpolated in truth or scoring.

The primary leaderboard definition has nine components: four global fields,
three independently ranked field-integrated coefficients, the velocity
profiles with weight 0.15, and the continuous Cp cuts with weight 0.10. The
four field errors use `clip(100 * (1 - error/cap), 0, 100)` with fixed caps of
15% surface pressure, 20% wall shear, 12% volume velocity, and 15% volume
pressure. Force and profile R2 values use `100 * clip(R2, 0, 1)`. The component
weights preserve a 50% field, 25% force, and 25% profile split. `Clf`, `Clr`,
and all RMSE values remain mandatory report-only diagnostics and receive no
duplicate composite weight.

The complete machine-readable source of truth is
[`submission-spec.json`](submission-spec.json). Candidate evidence and its
eligibility status are indexed in [`evidence/README.md`](evidence/README.md);
history and paper-parity definitions remain in [`proposal/`](proposal/). The
exact work still needed before opening submissions is tracked in
[`ACTIVATION_CHECKLIST.md`](ACTIVATION_CHECKLIST.md).

## Dataset-owner scientific approval

The machine-readable approval record is
[`evidence/owner-scientific-approval-2026-08-28.json`](evidence/owner-scientific-approval-2026-08-28.json)
(SHA-256
`448f2b852df7066fc311eed75ea94caa756ae6f63573d12a50798804c08929b3`).
It binds the pinned dataset, all-case native array and force evidence, the v10
profile candidate, all-case fixed/relative support, published native-v3 truth,
and the real `run_419` validation. It also records that surface arc length is
the Cp scoring coordinate and native segment-midpoint streamwise x is display
only.

This scientific approval is deliberately distinct from the final
release-specific approval. The latter cannot be issued until it can bind the
frozen evaluator, scoring-support and truth release IDs, three-model
sensitivity evidence, and approving release commit.

## Prototype leaderboard fixtures

The ten checked-in DrivAerML leaderboard rows can be regenerated with:

```bash
python3 scripts/generate_drivaerml_leaderboard_fixtures.py
python3 scripts/manage_leaderboard.py build
```

The fixture generator uses every official test case and writes all 16 AutoCFD5
lines on the exact station-specific 10 mm candidate grids (3,756 velocity
samples per case) plus four continuous Cp curves. Those checked-in submission
predictions remain analytical CFD-like prototype data, not trained-model
predictions. Separately, the leaderboard now loads the published native-v3 CFD
ground truth for all 484 cases and overlays it only when exact support and
coordinate identities match. Diagnostic RMSE and R2 metrics for the prototype
rows are recomputed from their generated curves, the Cp error is dimensionally
tied to the pressure-field fixture through `Cp = 2 pMeanTrim / 38.889^2`, and
force R2 is tied to the exact pinned force table through
[`force-r2-truth-statistics.json`](force-r2-truth-statistics.json). The
displayed bounded scores are therefore internally consistent, while all rows
remain labelled non-rankable prototype data.

The candidate evaluator's all-484 native-surface force replay is recorded in
[`evidence/force-replay-all484.json`](evidence/force-replay-all484.json). This
passing implementation evidence is included in the 2026-08-28 scientific
approval. It does not by itself freeze the evaluator or open submissions.

The path-free native-volume evidence now includes both the original two-case
implementation pilot and the strict
[`all-484 equal-cell primary audit`](evidence/native-volume-equal-cell-primary-all484.json).
The latter verifies all 978 pinned segments and 68,949,662,110 native cells,
including exact field shape/order, finite values, complete coverage, and two
independent chunk partitions. It is not a model result or scoring baseline.
This audit supplies the complete volume-field weighting evidence
because the contract uses equal native cells only.

The earlier all-484 discrete-probe candidate sweep and its review artifacts
remain indexed in [`evidence/README.md`](evidence/README.md) for research
provenance only. They are not participant instructions, submission artifacts,
or scoring support, and their unresolved rows do not block a submission
contract that otherwise becomes eligible. Exact immutable native extraction
support and native truth for the four continuous Cp cuts are now complete for
all 484 cases and scientifically approved. They still need to be bound into the
final immutable scoring-support and profile-ground-truth release identities.

The earlier proposed 1, 2, 5, and 10 mm profile study is retained in
[`proposal/PROFILE_RESOLUTION_CONVERGENCE_INPUT.md`](proposal/PROFILE_RESOLUTION_CONVERGENCE_INPUT.md)
as historical review methodology. The owner has accepted the published
deterministic 10 mm support without requiring that separate campaign for
activation. The remaining sensitivity gate is different: it requires genuine
outputs from at least three distinct trained models on a common official cohort
and the paired 10,000-replicate bootstrap after the evaluator is frozen.
Publishing those models' complete native fields is not required.

FluidsBench contributors can follow the bounded-memory native-mesh and closed
candidate package workflow in
[`PARTICIPANT_GUIDE.md`](PARTICIPANT_GUIDE.md). AutoCFD5 is used there only as
the provenance for the 16 velocity-line definitions; it is not a second
submission target. The candidate release hand-off is recorded in
[`candidate-release-bindings.json`](candidate-release-bindings.json); its
unresolved owner-release tokens are blockers and are never copied into the
active `submission-spec.json`. The hand-off may hash-bind the complete local
`drivaerml-diagnostics-v10.json` file while those owner fields remain
unresolved; that partial hand-off is integrity evidence only and neither
activates v10 nor makes a participant package ready. A one-command synthetic
two-part/three-part dry run is provided under
[`examples/drivaerml-candidate-native-chunks/`](../../examples/drivaerml-candidate-native-chunks/);
it is an ineligible transport and packaging fixture, not an official
DrivAerML submission. Candidate native-volume support generation uses the exact
receipt-compatible runtime Python 3.12.13 and NumPy 2.2.6. Surface and velocity
evidence remains pinned to VTK 9.5.2.

The same hand-off hash-binds the closed report-only
[`drivaerml-relative-diagnostics-v3.json`](drivaerml-relative-diagnostics-v3.json)
contract and its DrivAerML-only namespaced profile-chunk schema. The relative
velocity family is v3, the relative Cp family remains v1, and both have
composite weight zero. Exact copies of the all-484 velocity-placement,
velocity-mapping, and relative-Cp manifests are retained under
[`support/relative-v3/`](support/relative-v3/), byte-verified against their
producer outputs, and checked against the ordered official 484-case registry.
The nested `relative_support` hand-off is therefore ready. A separate immutable
pending release record binds those files, the per-case relative-series identity
index, and evaluator revision `b9db402a3fa0efbb94fe36be2ba1fe6f7b4bc1e1`.
The underlying scientific support is covered by the 2026-08-28 owner approval.
The release record still correctly retains its release-specific approval as
pending because that schema requires the approval to bind the not-yet-produced
genuine-model sensitivity evidence and the final approving commit.
`submission-spec.json` consequently keeps `profile_format_enabled=false`, and
the validator rejects the format unless every activation gate is complete.
Support publication is not scientific activation. The nested status is
independent and does not block promotion of the ranked constant candidate.
This extension does not alter the active constant velocity weight
0.15, constant Cp weight 0.10, or legacy profile package shape.
