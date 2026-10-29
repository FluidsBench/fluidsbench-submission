# DrivAerML activation status

DrivAerML is open for submissions on all eight prescribed splits. The active
contract accepts surface-and-volume or surface-only packages through the normal
contributor validator. Follow the [quickstart](QUICKSTART.md).

## Opening requirements completed

- The official `drivaerml-native-support-v3` manifest is published and hash-bound
  in `submission-spec.json`, together with the owner approval and validation receipt.
- The native evaluator, package assembler and official contributor validator accept
  the same approved scientific contract.
- Daniel's retained CANTO Full package was revalidated against the final release
  identities: 50 official cases and 2,000 profile series. It remains a historical
  trial and is absent from the launch source and leaderboard.
- The [owner opening decision](OWNER_OPENING_DECISION.md) waives the proposed
  multi-model sensitivity study. This does not claim that the study was run.
- Geometry-relative diagnostics have zero weight and remain report-only.
- The leaderboard is empty and unreleased. Receiving or approving submissions
  does not publish an official ranking; that is a separate release step.

The repository ownership move and production website deployment await Neil's
review. Those operational steps do not change the scientific contract below.

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
