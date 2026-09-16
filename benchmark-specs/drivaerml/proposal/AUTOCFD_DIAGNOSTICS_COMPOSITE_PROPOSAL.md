# AutoCFD velocity, FluidsBench Cp-cut, and composite proposal for DrivAerML

Status: **historical review source; the profile science and all-case support
were owner-approved on 2026-08-28, while production release and ranking gates
remain open**

Prepared: 2026-08-19

> Current-state note (2026-08-28): this file preserves the pre-approval
> rationale and proposed experiments. The published v10/native-v3 profile
> support, 10 mm velocity grid, explicit gaps, numerical tolerances, and four
> continuous Cp cuts are now scientifically approved. The 1/2/5/10 mm
> convergence and 0.5/1/2 micrometre tolerance campaigns described below are
> no longer activation requirements; this is an owner decision, not a claim
> that those experiments ran. Follow the current
> [`../ACTIVATION_CHECKLIST.md`](../ACTIVATION_CHECKLIST.md) and
> [`../evidence/owner-scientific-approval-2026-08-28.json`](../evidence/owner-scientific-approval-2026-08-28.json),
> not the historical “before activation” wording below.

Scoring update (2026-08-21): the earlier physics-null proposal in this review
record is superseded by the active machine-readable definition in
`../submission-spec.json`. DrivAerML now uses the shared FluidsBench fixed caps
for the four field errors and bounded global R2 for `Cd`, `Cl`, `CmPitch`, the
16 velocity lines, and the four continuous Cp cuts. Historical null-baseline
discussion below is retained only as decision provenance and is not a
participant instruction.

Owner scope update: **on 2026-08-21 the benchmark owner removed the 209
discrete Cp probes from DrivAerML submissions and scoring.** This does not
remove the four continuous Cp cuts: upperbody centreline, underbody centreline,
sidewall at `z=0.15 m`, and front-left wheelhouse at `y=-0.6 m`. The
historical submission-facing registry was `drivaerml-diagnostics-v9.json`; the
all-case continuous-cut support has since been published and scientifically
approved in the v10/native-v3 candidate, but still awaits final release
binding. The 209-probe
definitions, mappings, atlases, and replays documented below are retained only
as inactive research evidence. They are not participant inputs or outputs,
evaluator dependencies, composite components, scoring support, or activation
gates.

## Authoritative AutoCFD inputs

- AutoCFD5 Case 2 description, 23 February 2026:
  `https://autocfd5.s3.eu-west-1.amazonaws.com/test-cases/case2/AutoCFD5_Case2_Intro_260223.pdf`
  (`sha256:2308a7e11202eeadc4f4080b4d5703af89207ddf342ede86f16aeaf90e3f67a7`)
- AutoCFD5 DrivAer result template v8:
  `https://autocfd5.s3.eu-west-1.amazonaws.com/test-cases/case2/AutoCFD5_DrivAer_Result_Template_v8.xlsm`
  (`sha256:5c7e04e5ccd7190b4f81173a3e5dcde09b9e25a9d91fdb866d48beab2e1e8960`)
- AutoCFD5 Case 2a baseline surface:
  `https://autocfd5.s3.eu-west-1.amazonaws.com/test-cases/case2/DrivAer_Notchback_baseline_geometry.stl.gz`
  (`sha256:ef04c7964530f32a8c805efd8c26364f8e63ee508084292295ab0b15d828001f`)

The current guide marks these diagnostic definitions as unchanged from
AutoCFD4.

The submission-facing v9 registry incorporates the velocity lines
`sha256:6eb1528034e27a75ab1949551d58c8a161331bf6d343ca7fb4324f0800ca4d12`,
expanded 10 mm velocity samples
`sha256:5f1bcf84a633aa6bfdd776764c3295d5d624ef0b6c3649f67de446281ae5ba97`.
It also names the four retained continuous Cp cuts. Their all-case extraction
support and hashes were subsequently published and scientifically approved;
only immutable production-release binding remains open.
For historical research reproducibility only, the inactive unique Cp taps are
`sha256:2c1e216ef5693b26d43b9b4f55586ca35516224af0ad874987e5902c67f08e6f`,
and Cp panel membership
`sha256:6e288077b8b87e0c5d8d77c1050b07e4b5dd155a8e4fad8815ce4b0581995538`.
The candidate tap-to-component review atlas is
`sha256:b4ef9270f2633f2bd3d271e18de23f93e8da3398d67fbc34e77271fb37cbce66`.
None of those probe hashes defines the retained continuous-cut support.

## What AutoCFD actually defines

AutoCFD asks for static `Cp` at 209 discrete surface-pressure taps. These are
not continuous, uniformly sampled Cp cuts. The exact unique probe IDs and
nominal baseline-geometry XYZ coordinates are recorded in
`autocfd_cp_taps_nominal.csv`. The input sheet stores them in 13 source/PID
families. The AutoCFD plotting sheet rearranges the same 209 probes into 15
named plotting sequences and reuses eight probes between plots; that 217-row
plot membership and ordering is recorded in
`autocfd_cp_panel_membership.csv`.

This describes the AutoCFD source and the earlier discrete-probe research
proposal, not the current DrivAerML submission contract. Participants do not
submit Cp probe values or mappings, and the evaluator does not calculate or
score them. The separately retained continuous Cp cuts must not be implemented
by treating these sparse probe rows as cut samples.

The v8 workbook has one internal inconsistency: its hidden plotting sheet swaps
the `y` coordinates attached to underbody probe IDs 623 and 629 relative to the
mandatory `Surface Data 2a/2b` input tables. The benchmark owner decided on
2026-08-19 that the mandatory input tables are authoritative. The registries
therefore use `623: y=-0.590 m` and `629: y=-0.360 m` while retaining the
plotting order.

AutoCFD also asks for normalized velocity magnitude, `|U|/U_inf`, along all 16
lines `V1`-`V6`, `U1`-`U6`, `L1`, and `R1`-`R3`. Their exact start and end XYZ
coordinates, converted from millimetres to metres without changing the DrivAer
CAD frame, are recorded in `autocfd_velocity_lines_nominal.csv`. Eleven lines
have wind-tunnel correlation data; the other five remain mandatory columns in
the AutoCFD result template and must not be described as experimentally
supported. The benchmark owner decided on 2026-08-19 that all 16 lines enter
the ranked CFD/ML diagnostic, with the experimentally supported 11-line subset
reported separately. That subset label records AutoCFD experimental
availability at those baseline locations; DrivAerML ranking still compares ML
against the released CFD truth for each morph and must not be presented as
per-morph experimental validation.

All coordinates are metres in the released DrivAer CAD frame: the origin is at
the front-wheel axle, `+x` is downstream/freestream and drag, `y=0` is the
longitudinal symmetry plane, `+z` is upward, and the road is
`z=-0.3176 m`. Use `U_inf=38.889 m/s` in `+x`; do not rotate, translate, or
case-normalize the AutoCFD coordinates.

AutoCFD does not define a fixed velocity-profile resolution. It permits each
participant to choose a point count sufficient to represent the profile. The
v8 input sheet provides 1,001 data rows per line, but this is capacity rather
than a prescribed spacing. A fixed grid is therefore a FluidsBench addition
required for reproducible scoring, not an AutoCFD rule.

## Frozen proposed FluidsBench sampling and extraction

Use endpoint-inclusive, equal-arc sampling at a nominal 10 mm spacing:

| Lines | Count per line | Nominal spacing |
|---|---:|---:|
| `V1`-`V6` | 201 | 0.01 m |
| `U1`-`U6` | 301 | 0.01 m |
| `L1` | 651 | 0.01 m |
| `R1`-`R3` | 31 | approximately 0.01 m |

This gives exactly 3,756 samples per case and keeps the longest line (`L1`, 651
samples) within the 1,001-row AutoCFD template capacity. For endpoints `a` and
`b` and count `N`, sample `k` is `a + k(b-a)/(N-1)`. The proposed v1 scoring
grid is 10 mm; it is a FluidsBench reproducibility choice and must not be
described as AutoCFD-prescribed. All 3,756 resulting coordinates and distances
are expanded in `autocfd_velocity_samples_10mm.csv` so evaluator
implementations need not reproduce decimal rounding independently.

Before activation, build a 1 mm reference support (37,416 samples per case) and
audit nested 2, 5, and 10 mm grids. Every candidate grid (2, 5, and 10 mm) must
change aggregate profile loss by no more than 0.5%, every case-macro loss by
no more than 1%, every case-line loss by no more than 2%, and retain a
method-order Kendall tau-b of at least 0.99 relative to 1 mm before the proposed
10 mm grid can be retained. A passing 10 mm block cannot hide a failed 2 or
5 mm block. For candidate spacing `h`, define a relative change as
`|L_h-L_1mm|/L_1mm`. If `L_1mm <= 1e-12`,
require absolute agreement within `1e-12` instead. Apply the 0.5% limit to the
final case/line macro loss, the 1% limit to each case's 16-line macro loss, and
the 2% limit to each case-line loss. Compute Kendall tau-b on the final method
scores, declaring score differences at most `1e-12` tied. Pin the method set
before the study: the physics null, nearest-training-design-vector control,
and at least three distinct trained model/checkpoint submissions. Failure leaves the
diagnostic unranked until a new version globally adopts the coarsest finer grid
that passes; it must not silently change an active contract.

For the Kendall calculation, rank methods by their final `E_profile` values
(equivalently by profile skill with one frozen positive `B_profile`), never by
the nine-component overall composite.

For the proposed native `CellData` track, the benchmark-owned extractor locates
every native volume cell whose closure contains each valid velocity sample and
takes `UMeanTrim` from the smallest raw VTK cell ID before forming
`|U|/38.889`. Use a `1e-6 m` absolute closure tolerance; pre-activation replay
must additionally show that 0.5 and 2 micrometre tolerances do not alter any
non-face assignment or ranked result. This is an explicit zeroth-order
finite-volume reconstruction, applied identically to truth and prediction. The
released meshes contain hexahedra, polyhedra, wedges, pyramids, and tetrahedra,
so the owner extractor must support all five; it must not assume an all-hex
mesh or rely on an implementation-defined `vtkProbeFilter` face tie-break.
The exact candidate-discovery and closure kernel, dependency revision, and
candidate-count semantics are not yet frozen: activation requires a published,
hashed reference evaluator plus golden all-case assignment and tolerance
replays. The rules above are therefore proposed selection semantics, not a
claim that an executable locator is already released.
Only samples confirmed to be inside the morphed solid or outside the released
fluid domain may be excluded, and they must be marked by a hash-bound,
owner-published validity mask. A submitter may not create or alter that mask.
`no_native_cell_within_closure_tolerance`, a failed native-cell evaluation, or
an ambiguous polyhedron solid-angle result is an unresolved mapping failure,
not an automatic exclusion. Every non-excluded sample must select one
deterministic raw VTK cell ID; otherwise its line and the case's complete ranked
velocity component are unavailable pending a support correction. There is no
snapping, interpolation, extrapolation, or silent omission. The scoring support
publishes XYZ, distance from the line start, owner inclusion/reason, mapping
status/reason, source raw cell ID, candidate count, geometric tolerance, value,
and hashes for every case. A separately reported PointData-interpolated trace
is a sensitivity diagnostic only.

For profile reduction, let `q=|U|/U_inf`, `e_k=q_pred,k-q_true,k`, and
`ds_k=s_(k+1)-s_k`. For case `c` and line `l`, let `A_cl` contain only edges
whose two endpoints are valid, and calculate

`R_cl = sqrt(sum_(k in A_cl) ds_k*(e_k^2+e_(k+1)^2)/2 /
             sum_(k in A_cl) ds_k)`.

Only owner-excluded endpoints may be removed from `A_cl`; an unresolved mapping
failure makes the line unavailable rather than silently shortening it. Never
bridge an excluded gap. Every case/line must retain positive contributing
length after the owner mask is applied; otherwise the profile component remains
unranked pending an owner-reviewed support correction. The ranked profile error is
`E_profile = mean_c(mean_l(R_cl))`, with all 16 lines and cases weighted
equally. This is an RMSE of an already freestream-normalized quantity, not an
RMSE divided by a second data-dependent normalizer.

## Retained continuous Cp cuts

The current contract retains exactly these four continuous surface-pressure
cuts as one composite component:

| Cut ID | Geometric plane |
|---|---|
| `upperbody_centerline` | upperbody intersection at `y=0` |
| `underbody_centerline` | underbody intersection at `y=0` |
| `sidewall_z_0_15` | sidewall intersection at `z=0.15 m` |
| `front_left_wheelhouse_y_neg_0_6` | front-left wheelhouse intersection at `y=-0.6 m` |

These are continuous case-specific surface/plane intersections, not the 209
AutoCFD taps and not four selected subsets of those taps. The participant's
local evaluator derives truth and prediction from its native surface-pressure
prediction using

`Cp = 2 * pMeanTrim / (38.889 m/s)^2`.

Before activation, a separate immutable Cp-cut support must freeze the included
anatomical components, connected-curve segmentation, coordinate origin and
orientation, plane-intersection tolerance, tie and degeneracy rules, validity
handling, raw native polygon and local-segment provenance, positive finite
intersection-segment lengths, source hashes, and full-case versus chunked
invariance. Surface `pMeanTrim` is native `CellData`, so its value is piecewise
constant on each intersected polygon segment; there is no resampled Cp grid or
trapezoidal interpolation. For case `c` and cut `r`, the fixed reduction is

`R_cr = sqrt(sum_j(ds_j*(Cp_pred,j-Cp_truth,j)^2) / sum_j(ds_j))`,

where `j` runs over the frozen native polygon-intersection segments and `ds_j`
is the segment length. Then
`E_cp_cut=mean_c(mean_(r=1)^4(R_cr))`. Until that support and its golden replay
are approved, the Cp-cut component remains unranked. The excluded probe
registry, probe atlas, and probe overrides may not be used as a shortcut for
this support.

## Inactive 209-probe research record

Everything in this section is historical discrete-probe research evidence and
has no role in the current submission, evaluator, score, or activation
decision. It is retained so the earlier mapping investigation remains
reproducible; completing the mappings or approving the atlas is not required
before submissions open. This exclusion does not apply to the four continuous
Cp cuts defined above.

The 209 Cp coordinates belong to the baseline AutoCFD geometry and generally do
not lie on a morphed DrivAerML surface. Propagating every probe through the
original ANSA morph would be scientifically preferable, but the benchmark
owner reported on 2026-08-19 that the retained case-specific probe
correspondence is not believed to be available. Plain fixed world XYZ and
unrestricted nearest-VTP projection are therefore invalid surface-sampling
rules for the morphed cases.

The proposed reproducible surrogate is:

1. Derive a candidate primary anatomical solid for every nominal tap by
   projecting it onto the pinned official AutoCFD baseline STL, then require
   owner visual sign-off before freezing it. Map baseline names to the 49-name
   vocabulary observed in the pinned `run_1/drivaer_1.stl` and expected, but
   not yet proven, in every public `run_N/drivaer_N.stl`; the baseline
   `Body_Fascia_front_1_lower_lip` is merged into DrivAerML
   `BodyFasciafront1`. The resulting 209-row owner-review atlas is
   `autocfd_cp_tap_component_registry.csv`.
2. In each case, project the tap only onto that permitted named STL solid.
   Never fall back to an unrestricted surface. Preserve the exact physical cut
   where one exists: `y=0` for upperbody/underbody centreline, `z=0.150 m` for
   sidewall/rear-end 150 mm, `z=0.500 m` for rear-end 500 mm, and the declared
   per-probe `y` or `z` plane for the wheelhouse. Window/pillar/trunk and
   off-centre detailed-underbody points use constrained three-dimensional
   closest projection. The atlas explicitly stores `projection_mode`,
   `cut_axis`, and `cut_value_m` for every probe; do not infer them from labels.
   For a cut, intersect every permitted triangle with the exact plane and
   choose the closest point on the resulting line segments. Otherwise choose
   the closest point on the permitted triangles. Let `d_min` be the minimum
   candidate distance; all
   candidates with `d <= d_min + 1e-6 m` form the tie set, from which select
   the smallest raw STL triangle ID, defined as the zero-based global `facet`
   order in the ASCII file. Classify a vertex as lying on a cut plane when the
   absolute value of its signed distance is at most `1e-9 m`. A crossing yields
   its closed segment or point; a fully coplanar triangle contributes the
   entire closed triangle.
   Choose the closest point on that union before applying the distance tie.
3. Bridge that mapped STL point to the native boundary VTP. Require distance
   at most 2 mm and absolute normal dot product at least
   `cos(30 deg)=0.8660254037844387`;
   discard candidates failing either filter, and flag selected bridge distances
   over 0.5 mm for visual review. Order the remaining candidates by distance,
   then decreasing normal agreement, then smallest raw VTK polygon ID. As
   above, the distance tie set is every candidate within `1e-6 m` of `d_min`;
   normal-agreement values within `1e-12` are tied. Recompute the STL facet unit
   normal from its ordered-vertex cross product. For a VTP polygon with released
   connectivity-order vertices `v_i`, use the unit area vector
   `normalize(sum_i(v_i cross v_(i+1)))`, with cyclic indexing. A facet or
   polygon whose unnormalized area-vector magnitude is at most `1e-15 m^2` is
   degenerate and invalid. The absolute normal dot product deliberately makes
   orientation reversal irrelevant.
4. Publish nominal and mapped XYZ, primary solid, STL triangle, nominal
   displacement, normals, native raw polygon ID, bridge distance/agreement,
   validity/reason, source hashes, and any owner override for every case/tap.

The atlas columns prefixed `component_assignment_` record the unrestricted
three-dimensional baseline nearest-triangle audit used only to select and
review the candidate anatomical component. They are not outputs of the row's
declared cut or closest-point rule and must never be copied into scoring
support. The declared rule is executed afresh on every pinned case STL; its
rule-specific point, triangle, distance, and bridge result are separate
per-case support fields.

The explicit per-case projection-rule families are:

| AutoCFD PID / points | Projection rule |
|---|---|
| 600 and 610 | component-constrained cut at `y=0` |
| 601 and 611 | component-constrained cut at `z=0.150 m` |
| 614 | component-constrained cut at `z=0.500 m` |
| 624 points 1 and 8-17 | front-wheelhouse cut at `y=-0.750 m` |
| 624 points 2, 7, and 18-26 | front-wheelhouse cut at `y=-0.600 m` |
| 624 points 3-6 | front-wheelhouse cut at each probe's nominal `z` |
| 625 points 1-21 | detailed-underbody centreline cut at `y=0` |
| 602-604, 612, 613, 615, and 625 points 22-34 | component-constrained 3-D closest point |

The automated baseline component-assignment audit places all 209 taps on
plausible wetted vehicle components: none maps to a domain, artificial control
surface, tyre, rim, brake, wheel support, plinth, powertrain, or exhaust
component. No first/
second component gap is within 0.1 mm (the minimum is 1.542456 mm). Of the 209,
205 component-assignment nearest points lie within 1 mm of the official STL.
Probes 625, 626, 552, and 553 have component-assignment offsets of 5.000633,
4.902647, 2.754784, and 1.543416 mm respectively; probe 120 is the only other
offset above 0.5 mm, at 0.983322 mm. These flags are recorded in the atlas and
every row remains `pending_visual_signoff`; the audit does not substitute for
owner review or rule-specific projection.

The superseded research proposal would have derived both truth and prediction
at the selected native polygon from the same canonical surface-pressure field:

`Cp = 2 * pMeanTrim / (38.889 m/s)^2`.

It would not have required a separately predicted `CpMeanTrim` array. On all 8,828,095
native polygons in `run_1`, this transform reproduces the released
`CpMeanTrim` with maximum absolute difference `4.694e-7` and RMSE
`1.698e-8`. The previously planned all-484 replay and `1e-6` acceptance test
are no longer activation requirements. The released `CpMeanTrim` remains a
source QA array, not a current model target.

Tyres, rims, brakes, wheel supports, mirrors, door handles, and artificial
`CTRL_SURFACE_*` patches were globally excluded unless the research atlas
explicitly assigned a probe to one of them. The historical mapping gates were:
the declared component must exist; the declared projection must return a
finite, nondegenerate candidate; the 2 mm bridge and 30-degree normal filters
must pass; the selected raw polygon must be unique after the fixed tie rules;
and nominal displacement must not exceed `0.278618 m` (10% of the 2.78618 m
reference wheelbase). Displacement above `0.2089635 m` (7.5% wheelbase) was a
mandatory owner-review flag. Side, symmetry, panel-order, mirrored-pair, and
duplicate/collapse summaries were non-normative owner-review diagnostics in
that proposal.

Under the superseded research proposal, every one of the 209 mappings would
have needed to be valid for every ranked case. A failed mapping would have been
repaired only before activation by an immutable owner-override row
keyed by `(case_id, autocfd_probe_id)` and containing replacement mapping IDs
and coordinates, reason, author, version, and source hashes. No such probe
repair or active probe support is required under the current probe-free
contract; this does not waive the separate continuous-cut support. If the probe
work is revisited, an active support version must never be edited in place.
This construction must be described as a
reproducible AutoCFD-location surrogate on each morph, not as the original
material tap transported through ANSA.

## Diagnostic reductions

- `velocity_profile_uinf_rmse`: dimensionless RMSE of `|U|/U_inf` along each line
  using arc-length trapezoidal weights, then an equal mean over the 16 lines,
  then an equal mean over cases. Also report the 11-line experimental-parity
  subset and every individual line.
- `cp_cut_rmse`: `E_cp_cut=mean_c(mean_(r=1)^4(R_cr))`, where `R_cr` is the
  native intersection-segment-length-weighted continuous-cut RMSE defined
  above. Every cut and case must be represented. The immutable segment support
  remains an activation item; the evaluator may not substitute or average the
  excluded discrete probes.
- R-squared remains a secondary literature diagnostic only. It is not the
  ranked profile statistic because flattened R-squared changes with sampling
  density and is unstable for nearly flat individual profiles.

All profile predictions must come from the same model and checkpoint as the
native predictions evaluated locally by the participant. The benchmark reference evaluator run at
the contributor stage recomputes these diagnostics from benchmark truth rather
than trusting manually supplied scalar summaries; this does not imply that a
FluidsBench maintainer reruns the model or downloads its complete predictions.
AutoCFD supplies the velocity-line locations; FluidsBench defines the four
continuous cut planes. The ranked truth is the released DrivAerML CFD for each
morphology, not the baseline AutoCFD wind-tunnel measurements.

## Owner-approved overall-score structure

The benchmark owner approved on 2026-08-19 the broader FluidsBench utility
allocation: 50% global fields, 25% field-integrated engineering quantities, and
25% local diagnostics. Within the local branch, the 16 velocity profiles are
AutoCFD-defined and the four continuous Cp cuts are FluidsBench-defined. The
2026-08-21 clarification removes only the 209 discrete probes and retains the
0.15/0.10 split between velocity profiles and continuous Cp cuts. The nine
component weights therefore still sum to one. The composite remains inactive
pending the scientific and activation gates.

| Component | Weight |
|---|---:|
| native surface pressure field | 0.15 |
| native surface wall-shear field | 0.10 |
| native volume velocity field | 0.15 |
| native volume pressure field | 0.10 |
| field-integrated `Cd` | 0.15 |
| field-integrated total `Cl` | 0.05 |
| field-integrated axle balance `CmPitch` | 0.05 |
| 16 AutoCFD velocity profiles | 0.15 |
| 4 continuous Cp cuts | 0.10 |

The raw component errors are fixed as follows:

- surface pressure and wall shear: arithmetic mean over cases of the complete
  case area-weighted relative L2 defined in `SCIENTIFIC_CONTRACT.md`;
- volume velocity and pressure: arithmetic mean over cases of the complete
  case equal-native-cell relative L2, preserving the current repository-wide
  volume default; no geometric cell-volume secondary is required;
- field-integrated `Cd`, `Cl`, and `CmPitch`: separate equal-case RMSEs of the
  coefficients integrated locally from each participant's native surface
  predictions using the constant
  AutoCFD convention `A_ref=2.17 m^2`, `L_ref=2.78618 m`,
  `CoR=(1.40009,0,-0.3176) m`, `rho_inf=1 kg/m^3`, and
  `U_inf=38.889 m/s`, with drag in `+x`, lift in `+z`, and pitch about `+y`.
  Always derive and publish `Clf=Cl/2+CmPitch` and
  `Clr=Cl/2-CmPitch`, including their individual RMSEs, but do not assign them
  additional composite weight: `Cl` and `CmPitch` are the two independent total
  and axle-balance modes already containing the same information. Report the
  geometry-specific-reference coefficients separately as an unranked AB-UPT/
  aerodynamic-efficiency view;
- velocity profiles: the equal-case/equal-line arc-weighted RMSE defined above;
  and
- continuous Cp cuts: the four-cut error whose deterministic native
  polygon-intersection segment support must be frozen through the activation
  work above.
  The 209 probes do not enter it.

The pinned constant-reference truth table has the exact header
`run,cd,cl,clf,clr,cs`, 484 finite rows, and one unique integer `run` for each
public case. Join `run=N` to `case_id=run_N` without positional matching; the
ranked dimensionless targets are lowercase `cd`, lowercase `cl`, and
`CmPitch=(clf-clr)/2`. The released `clf` and `clr` columns are mandatory
reporting targets and satisfy `cl=clf+clr` to CSV rounding (maximum absolute
residual `1.0e-7`). A missing,
duplicate, non-integer, nonfinite, unexpected, or manifest-extraneous run is a
hard evaluator error. The 16 known held-back run numbers are absent by design.

For each ranked coefficient `k` in `{Cd, Cl, CmPitch}`, use equal-case global
R2. Retain equal-case RMSE for those coefficients and for `Clf` and `Clr` as
report-only diagnostics. The four field errors use fixed bounded-error caps of
15%, 20%, 12%, and 15%; the five force/profile R2 components use bounded
quality `100*clip(R2,0,1)`. `Cd`, `Cl`, `CmPitch`, `Clf`, and `Clr` are
integrated or derived locally from each participant's native surface
predictions using the constant-reference force contract. If a separately
versioned direct-scalar force task is later activated, report it separately;
it cannot enter this composite. For every native boundary polygon `f`, let
`c_f` be the
arithmetic mean of its vertex coordinates and define the
released-connectivity-order oriented area vector

`A_f = 0.5 * sum_i((v_i-c_f) cross (v_(i+1)-c_f))`.

Let `C_f` be the OpenFOAM v2212
`primitiveMeshTools::makeFaceCentresAndAreas` face centre: the vertex mean for
a triangle and otherwise the triangle-area-magnitude-weighted centroid about
the vertex mean. With `p_f=pMeanTrim_f` and
`tau_f=wallShearStressMeanTrim_f`, use every native polygon exactly once and
compute

`dF_f = p_f*A_f - tau_f*|A_f|`,

`F = rho_inf * sum_f(dF_f)`, and

`M_CoR = rho_inf * sum_f((C_f-CoR) cross dF_f)`.

Both stored fields are kinematic (`m^2/s^2`), so multiplication by `rho_inf`
produces dynamic force and moment. With
`qA=0.5*rho_inf*U_inf^2*A_ref`, calculate
`Cd=F_x/qA`, `Cl=F_z/qA`, `CmPitch=M_CoR,y/(qA*L_ref)`,
`Clf=Cl/2+CmPitch`, and `Clr=Cl/2-CmPitch`. These are equivalent front/rear
axle loads from the whole-surface force and pitch moment, not front/rear surface
integrals. The evaluator follows the
[OpenFOAM v2212 `forceCoeffs` definition](https://api.openfoam.com/2212/classFoam_1_1functionObjects_1_1forceCoeffs.html);
Appendix B of the dataset paper contains apparent pitch/yaw-axis and
extra-`L_ref` typographical inconsistencies and must not be implemented
literally.

A run-1 replay over all 8,828,095 polygons gives
`Cd=0.310924656699`, `Cl=0.069382211528`,
`CmPitch=-0.080062755339`, `Clf=-0.045371649575`, and
`Clr=0.114753861103`. Absolute differences from pinned rounded truth are
`4.33e-8`, `1.53e-9`, `4.25e-10`, and `3.89e-8` for
`Cd/Cl/Clf/Clr`. Before activation, require every public case to reproduce all
four released coefficients within absolute `1e-6`; do not activate if that
all-case replay fails. Each `B_j` must be finite and strictly positive;
otherwise that component and the overall composite remain unranked.

Publish all nine raw errors and component skills, the three 0.50/0.25/0.25
branch scores, the overall score, and equal-case bootstrap confidence intervals.
Compute each branch as the normalized weighted mean of its component skills;
the overall score is equivalently `sum_j(w_j*S_j)` over the nine table rows.
Also publish report-only individual RMSEs for `Clf` and `Clr`, the maximum
predicted per-case closure residual `Cl-(Clf+Clr)`, and the frozen source-truth
closure audit (whose current maximum is `1.0e-7` from CSV rounding); those
diagnostics receive no extra weight.
Bootstrap replicates resample cases and recompute both `E_j` and `B_j` on each
replicate. Use 10,000 paired case-bootstrap replicates with seed `20260819`, one
shared resampled case-index vector across every method and component within a
replicate, and the two-sided 2.5/97.5 percentile interval. Order evaluated
cases by ascending integer `run`, then generate the row-major `(10000,n_cases)`
index matrix with NumPy 2.2.6 `Generator(PCG64(20260819))` and its `integers`
method using `low=0`, `high=n_cases`, `size=(10000,n_cases)`, `dtype=int64`,
and `endpoint=False`. Use NumPy percentile
`method="linear"` (Hyndman-Fan type 7) and publish the generated index matrix or
its canonical serialization and SHA-256 before activation.

The nearest-training control used in sensitivity analysis is the
**nearest-training-design-vector** control; it is benchmark-owned, unranked,
and not a submission method. Read `geo_parameters_all.csv` from the pinned
dataset revision (SHA-256
`2d4e8b3f166d1facab2a38821af2c3cc1dbba47cfbb7bf0814e3e0bf1fe3e829`),
join `run_N` to integer `Run=N`, strip ASCII whitespace from headers, and use
these 16 columns in order: `Vehicle_Length`, `Vehicle_Width`, `Vehicle_Height`,
`Front_Overhang`, `Front_Planview`, `Hood_Angle`, `Approach_Angle`,
`Windscreen_Angle`, `Greenhouse_Tapering`, `Backlight_Angle`,
`Decklid_Height`, `Rearend_tapering`, `Rear_Overhang`,
`Rear_Diffusor_Angle`, `Vehicle_Ride_Height`, and `Vehicle_Pitch`.

For each evaluated split, the donor pool is exactly that split's declared
training list. Compute each descriptor's training-pool mean and population
standard deviation (`ddof=0`); missing/nonfinite values or a nonpositive scale
are hard errors. Select the training case minimizing the sum of squared
descriptor differences divided by those 16 variances, breaking an exact tie by
smallest integer `Run`. Validation/test cases and force or flow statistics never
enter fitting or selection. Publish and hash the resulting
`(split_id,target_case_id,donor_case_id,distance_squared)` map.

That donor map does not yet define a full-composite prediction because native
meshes differ. Raw VTK IDs must never be copied between cases. This control
remains an activation gate until a benchmark-owned donor-to-target surface and
volume field-transfer operator, support-validity policy, and golden replay are
published.

Do not activate the overall score until field-integrated force replay matches
the pinned source force files and weight/cap-free baseline sensitivity is tested
with a null baseline, the nearest-training-design-vector control after its
transfer contract is frozen, and several real model architectures.

## Owner decisions recorded on 2026-08-19 and 2026-08-21

1. The 2026-08-19 research decision treated case-specific ANSA-morphed
   pressure-probe positions as unavailable and selected a benchmark-owned
   constrained-projection investigation. The 2026-08-21 scope decision
   supersedes its implementation and activation requirement.
2. For preservation of that inactive research record, the mandatory v8 input
   tables remain authoritative for underbody probes 623 and 629.
3. Rank all 16 mandatory AutoCFD velocity lines and additionally report the
   experimentally supported 11-line subset.
4. Use the 50/25/25 multi-scale allocation and the 60/40 priorities within each
   multi-component branch. The local-diagnostics branch therefore remains
   split between velocity profiles (0.15 overall) and continuous Cp cuts (0.10
   overall); removing the discrete probes does not remove the Cp-cut component.
5. On 2026-08-21, exclude the 209 discrete Cp probes from participant
   submissions, evaluation, scoring, baselines, bootstrap evidence, and
   activation gates. Retain the existing probe files only as inactive research
   evidence. Keep the four true continuous Cp cuts in the submission and the
   nine-component composite, with velocity profiles weighted 0.15 and the
   continuous Cp-cut component weighted 0.10. Probe mapping and Cp-cut
   extraction are separate scientific problems and must not be conflated.

## Remaining implementation gates

These are reproducibility and validation tasks, not unresolved benchmark-owner
policy questions:

1. Complete the 1/2/5/10 mm profile-resolution convergence study and verify the
   frozen proposed 10 mm grid satisfies the declared activation criteria.
2. Publish and hash the reference containing-cell implementation for all five
   released cell types, then golden-replay every profile assignment and
   tolerance-sensitivity result.
3. Replay the now-explicit native force-and-pitch-moment integration over all
   484 cases and require the declared absolute `1e-6` tolerance for `Cd`, `Cl`,
   `Clf`, and `Clr`, plus chunk-invariance tests.
4. Generate versioned golden truth and scoring support for the 16 velocity
   lines, with checksums, source raw IDs, validity masks, and deterministic
   evaluator replay tests. Do not include the inactive probe artifacts.
5. Generate and validate separate continuous extraction and scoring support for
   the four Cp cuts, including plane-tolerance sensitivity, raw polygon and
   local-segment provenance, positive finite segment-length checks, validity
   handling, deterministic replay, and chunk invariance. Do not use the
   209-probe mappings or atlas as cut support.
6. Run the nearest-training-design-vector (after its transfer contract),
   fixed-cap/bounded-R2 real-model sensitivity checks, and the declared
   bootstrap before enabling official ranking.
