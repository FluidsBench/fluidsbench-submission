# DrivAerML activation checklist

This checklist records the current state of the DrivAerML FluidsBench
submission and leaderboard contract. AutoCFD5 is referenced only as provenance
for the velocity-line geometry; this is not an AutoCFD activation checklist.

## Current status

The participant contract, evaluator implementation, all-case native reference
data, and leaderboard visualization are complete as a closed release candidate.
Neil Ashton, acting as the DrivAerML dataset owner, approved the scientific
evidence and decisions on 2026-08-28. The machine-readable approval is
[`evidence/owner-scientific-approval-2026-08-28.json`](evidence/owner-scientific-approval-2026-08-28.json)
(SHA-256
`448f2b852df7066fc311eed75ea94caa756ae6f63573d12a50798804c08929b3`).

Submissions nevertheless remain closed. Scientific approval does not supply
the still-missing immutable production release identities, multi-model
sensitivity evidence, or independent full-submission dry run.

## Completed and scientifically approved

- The 484-case `neashton/drivaerml` release is pinned at revision
  `7a5c0948ce27be709b1116a3a190f806e7a8f79f`, with every native source file,
  multipart volume, and all eight official splits hash-bound.
- Surface predictions use every native VTP `CellData` polygon in raw VTK cell
  order. Primary surface errors use the published same-order area arrays;
  equal-polygon errors remain mandatory secondary results.
- Volume predictions use every reconstructed native VTU `CellData` cell in raw
  VTK cell order and one equal weight per native cell. The all-case audit covers
  all 978 parts and 68,949,662,110 cells. Geometric volume weights are neither
  required nor accepted.
- The native pressure-plus-wall-shear force evaluator replayed all 484 cases
  within the fixed `1e-6` coefficient tolerance. Pressure gauge, wall-shear
  sign, reference quantities, force and moment equations, chunk invariance,
  and `CmPitch=(Clf-Clr)/2` are approved. `Clf` and `Clr` remain report-only.
- The owner-approved v10 profile candidate defines 16 fixed AutoCFD5 velocity
  lines at 10 mm and four continuous Cp cuts: upper-body centreline,
  underbody centreline, sidewall at `z=0.15 m`, and front-left wheelhouse at
  `y=-0.6 m`. The historical 209 discrete Cp probes remain outside submission,
  scoring, and activation.
- Fixed and geometry-relative profile support is published for all 484 cases.
  Geometry-relative velocity and Cp remain separately labelled report-only
  diagnostics with composite weight zero; they are never overlaid as if they
  shared the fixed support.
- Native CFD ground truth is published for all 484 cases: 19,360 profile
  series in 61 chunks. It contains 16 fixed velocity, 16 relative velocity,
  four fixed Cp, and four relative Cp entries per case. The native-v3 index
  SHA-256 is
  `e7cf14f161fc7dbf22794e6f66db4e157329be960d2a4368140e08cf0608a5ae`.
- Velocity truth is exact zeroth-order native `CellData` sampling. Its visible
  stair steps are expected and approved; truth and scoring must not smooth,
  interpolate, extrapolate, snap, or bridge across unsupported gaps. The
  published explicit unsupported rows and segment breaks are part of the
  accepted support.
- The earlier proposed 1/2/5/10 mm convergence campaign and 0.5/1/2 micrometre
  tolerance replay are superseded as activation requirements by the owner's
  acceptance of the published deterministic 10 mm support, `1e-6 m` boundary
  closure tolerance, and `1e-3` steradian fail-closed polyhedron tolerance.
  This is an owner decision, not a claim that the superseded experiments ran.
- Continuous Cp truth is selected directly from native surface `pMeanTrim` and
  uses the retained native intersection segments without interpolation or
  resampling. Surface arc length remains the scoring coordinate. Native
  segment-midpoint streamwise x in metres is an approved display-only
  coordinate and must not be sorted or used to join separate segments.
- The real Transolver `run_419` fixture passed current array-level profile
  validation against both velocity families and the continuous Cp cuts. It is
  useful end-to-end regression evidence but is not itself a full official
  participant submission.
- Schema-v3 participant packaging, methodology disclosure, profile identity
  checks, fail-closed browser matching, and the bounded nine-component score
  are implemented and tested.

## Still required before submissions open

1. **Freeze the production release.** Choose and publish the final evaluator
   Git revision, scoring-support release ID/manifest/URL/SHA-256, and
   profile-ground-truth release ID/SHA-256. Resolve the remaining placeholders
   in [`candidate-release-bindings.json`](candidate-release-bindings.json),
   promote v10 into the participant-facing specification, and bind every
   participant-produced metric/profile identity to those immutable releases.
2. **Complete the current scoring-sensitivity gate.** Run at least three
   genuine trained DrivAerML checkpoints on a common official cohort and the
   prescribed paired 10,000-replicate bootstrap. This validates ranking
   stability; it is separate from the now-complete scientific review of native
   support and profile definitions. Relative profile families stay report-only
   and their submission format stays disabled until this gate and their final
   release-specific approval record are complete.
3. **Run one independent full schema-v3 submission dry run.** It must exercise
   a complete official split and the frozen evaluator/release identities. The
   retained one-case Transolver `run_419` regression does not satisfy this
   full-split independence gate.
4. **Publish the official release.** After the preceding checks pass, create
   the final release-bound owner approval, set scoring support to `official`,
   set `submissions_open` to `true`, replace prototype leaderboard status with
   the official release, and promote the reviewed `dev` branches to the
   production branches/site.

Until those four release gates pass, the current implementation is suitable
for internal review and participant dry runs, but not official ranking or
citation.

## No longer blockers

- dataset-owner scientific review of the native fields, force convention,
  velocity support, Cp support, sampling choices, and numerical tolerances;
- all-case fixed/relative velocity or continuous-Cp support and native truth;
- the proposed profile-resolution and tolerance campaigns superseded by the
  explicit owner decision above;
- the inactive 209-probe Cp research atlas or its historical invalid rows;
- smoothing the native velocity truth for display; and
- publication of participants' complete native prediction fields or mandatory
  maintainer recomputation. Those remain optional reproducibility audits.
