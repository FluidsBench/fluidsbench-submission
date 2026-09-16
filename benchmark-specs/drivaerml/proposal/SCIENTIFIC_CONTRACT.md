# DrivAerML benchmark contract proposed for scientific review

Status: **dataset-owner scientific approval recorded 2026-08-28; production
release, sensitivity, and independent dry-run gates remain; submissions closed**

Prepared: 2026-08-19

Scope: the public `neashton/drivaerml` release pinned in this directory

> Current-state note (2026-08-28): this file preserves the reasoning and the
> proposed pre-approval requirements as a historical review record. The owner
> has since approved the completed all-case native, force, velocity, and
> continuous-Cp evidence and has accepted the published deterministic 10 mm
> velocity support and tolerances. Statements below that owner confirmation,
> all-case profile support, or the proposed resolution/tolerance campaigns are
> still pending are superseded by
> [`../evidence/owner-scientific-approval-2026-08-28.json`](../evidence/owner-scientific-approval-2026-08-28.json)
> and the current [`../ACTIVATION_CHECKLIST.md`](../ACTIVATION_CHECKLIST.md).
> The remaining release gates are not superseded.

## Decision in one paragraph

The proposed canonical FluidsBench task evaluates predictions on a frozen raw
support: all native released surface polygons from VTP `CellData` and a
finite-volume-aligned candidate consisting of all VTU `CellData` volume cells.
The owner must still confirm that the volume arrays are the un-interpolated
solver carrier before activation. Models may resample, interpolate, or use point
representations internally, but their predictions must be returned once for
every frozen canonical entity. Surface fields report physical-area and
equal-polygon errors; volume fields use the repository's equal-native-cell
definition only. All errors are calculated per case and then macro-averaged.
Activation remains conditional on the declared baseline/model sensitivity
study. AB-UPT and GeoTransolver are
preserved as explicitly different literature tracks because they do not use the
same volume support. On 2026-08-21 the benchmark owner excluded the 209
discrete Cp probes from participant submissions and scoring; their existing
files are inactive research records and are not activation gates. Four
continuous Cp cuts remain in scope and require separate extraction support. No
velocity profile, Cp cut, force integration, exclusion mask, or overall
composite becomes official until its complete definition and a golden reference
calculation are approved.

## Why this is the review-grade choice

The raw association is not universal across automotive datasets. It must be
read from each pinned dataset rather than inferred from a model architecture.
Only the DrivAerML, AB-UPT, and GeoTransolver rows below are bound to immutable
sources in this proposal. AhmedML, WindsorML, and HiLiftAeroML are contextual
local raw-file audits from 2026-08-18; they motivate dataset-specific handling
but are not normative evidence until their exact release revisions and audit
manifests are added.

| Evidence | Surface used | Volume used | Consequence |
|---|---|---|---|
| DrivAerML raw CFD release | VTP `CellData` polygons | VTU exposes both `CellData` and `PointData`; the latter is used by one paper pipeline | The two volume associations are different supports and cannot share one score; CellData provenance still needs owner confirmation |
| AB-UPT on DrivAerML | cell centres carrying VTP `CellData` | cell centres carrying VTU cell fields | Cell-based precedent, but published volume metrics use a random subset rather than every cell |
| GeoTransolver on DrivAerML | VTP `CellData` polygons | every raw VTU `PointData` vertex | Valid literature representation, but not the same support as the proposed cell benchmark |
| AhmedML contextual local audit (release pin not included) | VTP `CellData` polygons | VTU `CellData` cells | Example of a cell/cell raw-data contract |
| WindsorML contextual local audit (release pin not included) | VTU `PointData` surface vertices | VTU `CellData` volume cells | Example showing that even related datasets can need different surface and volume associations |
| HiLiftAeroML contextual local audit (release pin not included) | boundary-VTU `PointData` | volume-VTU `PointData`, with an owner-declared validity mask | Example showing that point support and exclusions are appropriate only when the source data and owner contract require them |

DrivAerML was solved with an OpenFOAM finite-volume method and its dataset paper
describes approximately 8.8 million surface cells and 160 million volume cells
per case. This makes a cell-based track the strongest FVM-aligned candidate and
avoids privileging AB-UPT, GeoTransolver, or the model being prepared for
submission. It is still a proposal, not proof that every released CellData array
is an un-interpolated solver field; that provenance is an activation blocker.

## Immutable public release

- Dataset: `neashton/drivaerml`
- Hugging Face Git revision:
  `7a5c0948ce27be709b1116a3a190f806e7a8f79f`
- Public cases: 484 (`run_1` to `run_500`, excluding the 16 IDs recorded in
  `native-source-pin.json`)
- Canonical source pin SHA-256:
  `4fc9077f8f23f4994c98f4d0e7a17aef7b998de4c996638e3a8a616b6d923fdd`
- Owner-published split manifest SHA-256:
  `032a2e9f88926d9218a1943b51e650135cc78683cad6b0a38f3cf4f9dfba647d`
- Surface-area manifest SHA-256:
  `1401c7e80bd86f3aa2d640289db9b088ce1e0825327e18eeb1ab2852de04323e`

Each reconstructed volume is the byte concatenation of its ordered `.00.part`,
`.01.part`, and, for ten cases, `.02.part` files. The ordered part paths, sizes,
and SHA-256 object identities in `native-source-pin.json` define the logical
VTU exactly; similarly named unparted alternatives are not canonical.

## Canonical support and identity

### Surface

- Source: `run_<N>/boundary_<N>.vtp` at the pinned revision.
- Association: VTK `CellData`.
- Entities: every native polygon, in raw VTK cell order.
- Stable ID: `(run_<N>, raw_vtk_cell_id)`.
- Interchange coordinate: polygon cell centre computed by the benchmark loader.
- Physical weight: the owner-published deterministic polygon area. The public
  area file has one little-endian float32 value per raw VTK cell and the same
  tuple order.
- The scoring support must pin the area algorithm and version, numeric tolerance,
  positivity/finiteness checks, and source-to-area tuple-count check.

### Volume

- Source: the reconstructed `run_<N>/volume_<N>.vtu` at the pinned revision.
- Association: VTK `CellData`.
- Entities: every finite-volume cell, in raw VTK cell order.
- Stable ID: `(run_<N>, raw_vtk_cell_id)`.
- Interchange coordinate: cell centre computed by the benchmark loader.
- Metric weight: one equal weight per raw native cell. No geometric cell-volume
  array is required or accepted as a submission input.
- Before activation, the owner must confirm the CellData provenance. The support
  must pin tuple counts, component counts, finiteness, raw order, and exact
  duplicate-free coverage for every case.

Coordinates are audit information, not identity. A submission cannot align
entities by nearest-neighbour coordinate matching. Chunk boundaries are also not
part of the scientific support; they are only a memory-management detail.

## Required field targets

| ID | Raw array | Components | Raw meaning |
|---|---|---:|---|
| `surface_pressure` | surface `pMeanTrim` | 1 | relative kinematic pressure |
| `surface_wall_shear` | surface `wallShearStressMeanTrim` | 3 | wall-shear-stress vector |
| `volume_velocity` | volume `UMeanTrim` | 3 | mean velocity vector |
| `volume_pressure` | volume `pMeanTrim` | 1 | relative kinematic pressure |

The dataset defines `pMeanTrim` as kinematic pressure, `p/rho_inf`, and fixes the
pressure reference to `p_ref=0 Pa` at `(x,y,z)=(80,10,10) m` on the outlet
boundary. That gauge and `rho_inf=1 kg/m^3` are part of the proposed contract.
The scoring-support audit must verify that every case uses this convention.
Kinematic pressure is reported in `m^2/s^2`; a conversion to pascals must be a
separately named, explicitly density-scaled metric.

The raw wall-shear vector is a required prediction target and is the
time-averaged OpenFOAM wall-shear-stress field: kinematic traction in
`m^2/s^2`, defined using the patch normal into the fluid domain. Convert it to
dynamic traction only by multiplying by `rho_inf`. Field comparison uses the
raw stored vector without a sign transformation. The field-derived body-force
sign is fixed separately below by replay against the public force table.

The release also contains `CpMeanTrim` and `CptMeanTrim`. They must not both be
called “pressure” without qualification. `CpMeanTrim` is the static pressure
coefficient; `CptMeanTrim` is the total pressure coefficient used by the AB-UPT
volume-pressure paper result. The canonical target above is static
`pMeanTrim`; a future coefficient target must have a separate metric ID and an
owner-verified conversion. The 209 discrete Cp probes are not part of the
current submission or score. The four retained continuous Cp cuts are derived
by the participant's local evaluator from its native `pMeanTrim` prediction using
`Cp=2*pMeanTrim/(38.889 m/s)^2`; no `CpMeanTrim` prediction or manually
supplied Cp scalar summary is accepted. The probe research record does not add
a submitted field or evaluator requirement.
Kinematic pressure must not be labelled as pascals.

For vector fields, the error at an entity is the norm of the component-wise
error vector. A difference between predicted and true vector magnitudes is not a
substitute because it can conceal direction error.

## Reductions and aggregation

For each case and target, accumulate the following over the complete support:

\[
E_{2,w}=100\sqrt{\frac{\sum_i w_i\lVert\hat y_i-y_i\rVert_2^2}
                              {\sum_i w_i\lVert y_i\rVert_2^2}},
\qquad
E_{2,1}=100\sqrt{\frac{\sum_i \lVert\hat y_i-y_i\rVert_2^2}
                              {\sum_i \lVert y_i\rVert_2^2}}.
\]

- On the surface, `w_i` is the fixed published polygon area; the equal-polygon
  view sets every `w_i=1` and is a mandatory secondary result.
- In the volume, every native cell has `w_i=1`; there is no physical-volume
  secondary result.
- Surface area/equal-polygon and volume equal-cell MAE and RMSE are mandatory
  diagnostics, particularly for pressure fields whose relative denominator can
  be small or gauge-sensitive.
- Every case-level relative-L2 truth denominator must be finite and strictly
  positive. A zero or nonfinite denominator is a benchmark-support error; the
  case may not be silently omitted, pooled, or replaced by zero.
- Any nonfinite prediction on a required valid entity is a hard submission
  error. Any nonfinite truth value, coordinate, or required weight is a hard
  benchmark-support error.

For the applicable surface or volume weight above, the absolute diagnostics are

\[
\operatorname{MAE}_w=\frac{\sum_iw_i\lVert\hat y_i-y_i\rVert_2}{\sum_iw_i},
\qquad
\operatorname{RMSE}_w=\sqrt{\frac{\sum_iw_i\lVert\hat y_i-y_i\rVert_2^2}
                                  {\sum_iw_i}}.
\]

For a scalar, the norm is absolute value; for a vector, it is the Euclidean norm
of the component-wise error. The result has the source target's unit: `m/s` for
velocity and `m^2/s^2` for both kinematic pressure and the raw kinematic
wall-shear vector.

- Compute each complete case first, then take the arithmetic mean of the case
  values. Do not pool large cases with small cases.
- With chunked evaluation, sum the additive numerators, denominators, counts,
  and weights across all chunks before applying a square root or division. Never
  average chunk-local norms.
- The completed candidate numerical evidence uses binary64 accumulation, pins
  the raw-entity/block merge order and acceptance tolerances, and publishes
  full-case-versus-chunked replays across all 484 cases. This completes the
  candidate chunk-invariance implementation evidence; activation still requires
  owner approval and incorporation into the frozen evaluator and immutable
  scoring-support release.

The surface area and equal-polygon results answer different questions and
neither may be omitted. Volume scoring intentionally follows the existing
equal-cell default, which emphasizes refined mesh regions and avoids introducing
an additional geometry-derived support. The candidate composite remains
inactive until reference baselines establish its distributions and the declared
sensitivity review confirms that the ranking is stable enough to publish. Any
near-body or wake-only score would require a separate owner-published geometric
region; no hidden crop is permitted.

## Case splits

Use the owner-published split lists in `owner-published-splits.json`:

| Track | Train | Validation | Test |
|---|---:|---:|---:|
| `full` | 400 | 34 | 50 |
| `medium` | 133 | 34 | 50 |
| `scarce` | 67 | 34 | 50 |
| `super_scarce` | 11 | 34 | 50 |
| `geometry` | 339 | 48 | 97 |
| `high_drag` | 339 | 48 | 97 |
| `low_drag` | 339 | 48 | 97 |
| `rear_separation` | 339 | 48 | 97 |

The `full` 400/34/50 IDs reproduce the AB-UPT seed-42 split. The other public
tracks are also owner-published and should keep their exact IDs and ordering.
Public test fields are evaluation-only: they cannot be used for fitting,
checkpoint selection, hyperparameter selection, manual tuning, or preprocessing
statistics.

GeoTransolver's drag-aware 436/48 split is a separate legacy paper split. It is
not silently substituted for an owner-published FluidsBench split.

## Paper-parity results

Paper-parity scores are useful reference results, but their namespace must state
their support, split, reduction, and coverage.

### AB-UPT literature parity

- Exact public `full` 400/34/50 split.
- Surface: all native VTP cells.
- Volume: VTU cells randomly subsampled to approximately the surface-cell count,
  repeated ten times—not the full native volume.
- Unweighted relative L2 per case, then case macro-average.
- Volume `p_v` means total pressure coefficient, not static `pMeanTrim`.
- Force values integrate dense surface predictions using polygon areas and
  normals and geometry-specific `A_ref`.

Its published field numbers must be labelled “AB-UPT sampled-cell literature
result” and must not be placed in the canonical full-volume column. This is not
yet an executable parity contract: the exact random-subsample generator, seeds,
and aggregation of the ten stochastic evaluations must be pinned first.

### GeoTransolver parity

- Exact legacy 436-train/48-held-out drag-aware split pinned to the published
  PhysicsNeMo-CFD lists.
- Surface: all native VTP `CellData` polygons.
- Volume: all raw VTU `PointData` vertices. For example, the audited
  paper-pipeline run 112 Zarr support has 159,899,850 entries, exactly matching
  raw VTU points rather than its 142,143,167 cells.
- Unweighted relative L1 per case, then case macro-average.
- Full support is inferred in chunks, reassembled, and scored once.

Its published field numbers must be labelled “GeoTransolver PointData parity.”
To claim reproduction of those published vector numbers, the parity evaluator
must use the released implementation's entity error
`abs(norm(prediction)-norm(truth))`. That magnitude-only formula must be named
explicitly and must not replace component-wise vector error in the canonical
benchmark.

## Forces and coefficients

Direct scalar prediction and force reconstructed from predicted fields are two
different tasks and must have different metric IDs.

The pinned release contains two scientifically distinct candidate truth
conventions:

- `force_mom_all.csv`: geometry-specific reference area and length; this measures
  aerodynamic efficiency and is consistent with the AB-UPT force calculation;
  and
- `force_mom_constref_all.csv`: constant `A_ref=2.17 m^2` and
  `L_ref=2.78618 m`; this retains geometry-size effects under one normalization.

No directly predicted scalar-force submission metric is frozen yet. For forces
reconstructed locally from each participant's native surface predictions,
however, the proposed
composite selects the constant AutoCFD convention. Rank `Cd`, total lift `Cl`,
and the independent front/rear balance
`CmPitch=(Clf-Clr)/2` from `force_mom_constref_all.csv`. The engineering branch
assigns overall weights `0.15/0.05/0.05` to `Cd/Cl/CmPitch`; this preserves the
approved 60/40 drag/lift-family priority without counting the dependent
quantities `Cl`, `Clf`, and `Clr` as three independent observations. Always
publish `Clf` and `Clr` themselves, including their individual RMSEs, but do
not give them additional composite weight. The geometry-specific
`force_mom_all.csv` result remains a separately named, unranked AB-UPT/
aerodynamic-efficiency diagnostic. The two reference conventions must never be
mixed between cases because they answer different design questions.

The pinned constant-reference aggregate table has exact columns
`run,cd,cl,clf,clr,cs`. It contains 484 finite rows and exactly one integer
`run` for every public case; join `run=N` to `case_id=run_N`. Truth for `Cd`,
`Cl`, `Clf`, and `Clr` comes from the corresponding lowercase columns and is
dimensionless; derive truth `CmPitch=(clf-clr)/2` case by case. All 484 rows
satisfy `cl=clf+clr` to the released seven-digit precision (maximum absolute
residual `1.0e-7`, RMS `3.18e-8`). Reject missing, duplicate, non-integer,
nonfinite, unexpected, or manifest-extraneous run IDs rather than dropping or
positionally aligning them.

For each ranked coefficient `k` in `{Cd, Cl, CmPitch}`, use equal-case global
R2, `1-sum_c((k_pred,c-k_true,c)^2)/sum_c((k_true,c-mean(k_true))^2)`.
Publish equal-case RMSE for all three ranked coefficients and for the mandatory
report-only `Clf` and `Clr` diagnostics. Publish the maximum predicted
`Cl-(Clf+Clr)` closure residual separately from the frozen source-table closure
audit, whose `1.0e-7` maximum reflects CSV rounding.

The freestream values are `U_inf=38.889 m/s` and `rho_inf=1 kg/m^3`; drag is
`+x`, lift is `+z`, and positive pitch is about `+y`. The constant convention
uses `A_ref=2.17 m^2`, `L_ref=2.78618 m`, and
`CoR=(1.40009,0,-0.3176) m`. Every one of the 484 public
`geo_ref_<run>.csv` files has those same `aRefRef`, `lRefRef`, and
`forcesCoRRef` values. The implied symmetric equivalent axle-load locations
are `x_front=0.007 m` and `x_rear=2.79318 m`; these are load-reaction
locations, not a front/rear surface partition.

For each native polygon, let `c_f` be the arithmetic mean of its vertex
coordinates and form
`A_f=0.5*sum_i((v_i-c_f) cross (v_(i+1)-c_f))` in released connectivity order.
Let `C_f` be the OpenFOAM v2212
`primitiveMeshTools::makeFaceCentresAndAreas` face centre: the vertex mean for
a triangle, otherwise the triangle-area-magnitude-weighted centroid of the
triangles formed from each edge and the vertex mean. Explicitly, for a
non-triangle let `c_bar=(1/n)*sum_i(v_i)`,
`n_i=(v_(i+1)-v_i) cross (c_bar-v_i)`, and `a_i=|n_i|`; then
`C_f=(1/3)*sum_i(a_i*(v_i+v_(i+1)+c_bar))/sum_i(a_i)`. Released connectivity
order is retained. OpenFOAM's `sum_i(a_i)<ROOTVSMALL` zero-area fallback is a
hard benchmark-support error because every scored polygon must have positive
finite area. Define
`dF_f=pMeanTrim_f*A_f-wallShearStressMeanTrim_f*|A_f|` and, using every native
polygon exactly once, calculate

`F=rho_inf*sum_f(dF_f)` and
`M_CoR=rho_inf*sum_f((C_f-CoR) cross dF_f)`.

With `qA=0.5*rho_inf*U_inf^2*A_ref`, calculate
`Cd=F_x/qA`, `Cl=F_z/qA`,
`CmPitch=M_CoR,y/(qA*L_ref)`,
`Clf=Cl/2+CmPitch`, and `Clr=Cl/2-CmPitch`. Thus `Cl=Clf+Clr` exactly before
output rounding. These equations follow the OpenFOAM v2212 `forceCoeffs`
implementation used for the dataset. Appendix B of the dataset paper appears
to swap the pitch/yaw axes in its displayed moment equations and to divide an
already dimensionless `CmPitch` by `L_ref` again in its displayed axle-split
equation; the evaluator must not reproduce those apparent typographical
inconsistencies.

A full native-field replay for run 1 gives
`Cd=0.310924656699`, `Cl=0.069382211528`,
`CmPitch=-0.080062755339`, `Clf=-0.045371649575`, and
`Clr=0.114753861103`. Absolute differences from the pinned rounded truth are
`4.33e-8`, `1.53e-9`, `4.25e-10`, and `3.89e-8` for
`Cd/Cl/Clf/Clr`, respectively. The completed candidate all-484 native-field
replay passed the absolute `1e-6` tolerance for `Cd`, `Cl`, derived `CmPitch`,
`Clf`, and `Clr` in every public case; its largest full-case-versus-chunked
difference was `8.33e-17`. Per-case `geo_ref_<run>.csv`
identities beyond the constant fields above are needed only to activate the
separately reported geometry-specific-reference diagnostic. Do not call a
scale-free force R-squared calculation a fully specified coefficient
calculation. The completed candidate replay is hash-bound in
[`../evidence/force-replay-all484.json`](../evidence/force-replay-all484.json).
The compact `run-1-force-axle-replay-summary.json` and separate
`force-definition-audit-all484.json` remain supporting records. This completes
the candidate numerical force evidence but does not confer owner approval or
replace the pending frozen evaluator and immutable scoring-support release.

## Validity, exclusions, velocity profiles, and continuous Cp cuts

- No submitter may infer an exclusion mask from target values. In particular,
  zero pressure is not by itself invalid.
- Before release, the benchmark owner must audit every required array in all 484
  cases for finite values and tuple-count consistency.
- Any exclusion must be published in the scoring support with `case_id`, raw
  entity ID, coordinates, reason, construction version, and source identity.
  The same frozen mask applies to every submission.
- For AutoCFD5 velocity coordinates, the only permitted owner exclusions are
  `inside_morphed_solid` and `outside_released_fluid_domain`. A failed or
  ambiguous containing-cell query is not an exclusion. Every non-excluded
  coordinate must map to one deterministic raw native cell ID or the affected
  line is unavailable. Owner-excluded points and their adjacent edges contribute
  nothing, and an evaluator must never bridge the resulting gap.
- The former prototype profile and Cp-cut station names did not include a full
  extraction and ground-truth contract and were not supported by AB-UPT or
  GeoTransolver scoring. The submission-facing replacement is the separately
  versioned diagnostic vocabulary in `../drivaerml-diagnostics-v9.json`: 16 AutoCFD5 velocity
  lines plus four continuous Cp cuts. It remains unranked until exact
  case-specific mapping/extraction, truth hashes, resolution studies, and
  validation support are published and activated.
- The exact nominal AutoCFD4/5 velocity lines, the four FluidsBench cut planes,
  fixed proposed sampling, reduction, and nine-component composite semantics
  are recorded in
  `AUTOCFD_DIAGNOSTICS_COMPOSITE_PROPOSAL.md`. The 209 discrete-probe
  registries, atlas, and mapping discussion in that proposal are explicitly
  inactive research evidence. Probe mapping, atlas review, and probe truth
  replay are not activation requirements and must not be substituted for the
  continuous-cut support.

## Historical approval blockers

The following list was the pre-approval review plan. Items 1-6 and the
scientific support portions of item 8 are now satisfied or explicitly
superseded by the 2026-08-28 owner decision. The sensitivity work in item 7 and
the immutable release binding/final release approval in item 8 remain open; see
the current activation checklist.

1. an all-case VTK inventory proving required array association, tuple count,
   component count, finiteness, units, pressure gauge, wall-shear sign, and the
   provenance of CellData versus PointData;
2. stable native surface and volume entity IDs, complete duplicate-free raw-cell
   order and coverage evidence for equal-cell volume scoring, and fixed surface
   coordinates and areas bound to the immutable source files, with the
   surface-area algorithm, tolerances, and QA checks pinned; no geometric
   volume-cell weights or volume-weight algorithm are required;
3. a declared exclusion policy and, if needed, a public stable-ID mask;
4. golden calculations for scalar and vector field reductions, including
   chunk-invariance tests, plus the reference velocity-profile locator and
   all-case AutoCFD velocity-profile support replay, and distinct continuous
   extraction, plane-tolerance sensitivity, segment-length reduction, and truth
   replay for the four Cp cuts;
5. an all-484-case replay of the now-explicit force-and-pitch-moment integration
   formula against public constant-reference `Cd`, `Cl`, `Clf`, and `Clr`, plus
   chunk-invariance tests and,
   if the distinct directly predicted scalar task is ever activated, explicit
   target columns, reductions, and edge behavior;
6. official split index files generated from the pinned owner manifest;
7. the nearest-training-design donor map and golden donor-to-target transfer,
   the frozen bootstrap index artifact, and evidence from the declared
   nine-component baseline/model sensitivity analysis that the owner-approved
   composite is stable enough for a single ranking; and
8. dataset-owner approval of the immutable scoring-support manifest and
   evaluator release.

## Primary sources

- [DrivAerML dataset paper](https://arxiv.org/html/2408.11969)
- [AB-UPT paper](https://openreview.net/forum?id=nwQ8nitlTZ)
- [AB-UPT release used for the audit](https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers/tree/2211068fd58600c52ed38de692f020e2d2b7c987)
- [AB-UPT DrivAerML cell-field loader](https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers/blob/2211068fd58600c52ed38de692f020e2d2b7c987/src/drivaerml_dataset.py)
- [GeoTransolver paper](https://arxiv.org/html/2512.20399)
- [GeoTransolver PhysicsNeMo release used for the audit](https://github.com/NVIDIA/physicsnemo/tree/43de886be3e623758a1798d45fee5da44e21c335/examples/cfd/external_aerodynamics/transformer_models)
- [GeoTransolver full-support inference implementation](https://github.com/NVIDIA/physicsnemo/blob/43de886be3e623758a1798d45fee5da44e21c335/examples/cfd/external_aerodynamics/transformer_models/src/inference_on_zarr.py)
- [GeoTransolver preprocessing release](https://github.com/NVIDIA/physicsnemo-curator/tree/6d6d98e4a427353fbf2933601b398ee998d4314f/examples/external_aerodynamics)
- [GeoTransolver DrivAerML split release](https://github.com/NVIDIA/physicsnemo-cfd/tree/0612ec4ed54484a47bfa134eda7b3b012a607624/workflows/benchmarking/drivaer_ml_files)
- [OpenFOAM wall-shear-stress field definition](https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/wallShearStress/)
- [OpenFOAM v2212 force-coefficient and axle-split definition](https://api.openfoam.com/2212/classFoam_1_1functionObjects_1_1forceCoeffs.html)
- [AhmedML dataset paper](https://arxiv.org/html/2407.20801)
- [WindsorML dataset paper](https://arxiv.org/html/2407.19320)
