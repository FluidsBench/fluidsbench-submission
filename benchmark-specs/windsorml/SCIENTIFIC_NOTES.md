# WindsorML scientific notes

WindsorML is open for submissions. The [quickstart](QUICKSTART.md) gives the participant workflow; [submission-spec.json](submission-spec.json) and [profile-definition-v2.json](profile-definition-v2.json) define the authoritative contract.

## Native inventory

Read with `vtkXMLUnstructuredGridReader.UpdateInformation()` (metadata only — a
full read of a 21 GB volume file needs well over 100 GB of RAM):

| | association | arrays |
|---|---|---|
| `boundary_N.vtu` | **PointData** | `cpavg`, `cfxavg`, `cfyavg`, `cfzavg`, `cpvar`, `yplusavg`, `Normals` (CellData holds only `faceid`) |
| `volume_N.vtu` | **CellData** (PointData empty) | `velocityxavg/yavg/zavg`, `pressureavg`, and six Reynolds stresses |

Scale across the 350 published runs: **2,021,398–2,650,517 surface points** and
**267,431,654–330,440,927 volume cells**.

Unlike AhmedML, arrays are not duplicated across associations, so the "evaluator
must explicitly select CellData" ambiguity does not arise. `faceid` takes a
single value, so the boundary is the body only and no patch selection is needed.

`boundary_dual_area_N.npy` is float32, one positive barycentric dual area per
native point, in point order. It is the canonical surface weight and must not be
recomputed — the dataset README warns that VTK polygon-area recomputation drifts
for non-planar quads. This removes AhmedML's need for a generated surface-area
sidecar entirely.

## The axis convention

**The vertical axis is +y, not +z.** AhmedML and DrivAerML both use +z for lift,
so this is the single easiest thing to get wrong.

Measured bounds for run_0: x ∈ [−0.561, 0.483], y ∈ [0, 0.343] (ground plane
upward), z ∈ [−0.1945, +0.1945] (laterally symmetric). Integrating the
coefficient fields against the published dual areas with `A_ref = 0.112 m²`:

| axis | integrated | published | difference |
|---|---|---|---|
| x → `cd` | 0.28108820 | 0.28181696 | 0.26% |
| **y → `cl`** | **0.48836350** | **0.48822198** | **0.03%** |
| z → `cs` | −0.01127239 | 0.00082344 | small absolute |

Mapping lift to +z instead collapses `cl` from ~0.49 to ~−0.01 — a plausible
looking number that is silently wrong. `reference/windsorml/contract.py` freezes
the mapping and `tests/test_windsorml_evaluator.py` regression-guards it.

## Force convention, and why the published CSV is not the scoring truth

`force_mom_*.csv` uses a constant reference area; recovered exactly as
`frontal_area × cd_varref / cd_fixed` = **0.112 m²**. `force_mom_varref_*.csv`
uses each case's own frontal area. The constant-area table is authoritative here,
matching how the `high_drag`/`low_drag` split families were built.

Predicted coefficients can only come from integrating predicted fields. If truth
came from the CSV instead, the gap between the dual-area point quadrature and the
solver's exact cell integration would become an error floor that no prediction
could reach, and `cd_r2` could never be 1. Across all published runs that gap
reaches **2.94% on drag** — a severe floor.

So **truth and prediction are integrated identically**, and the published CSV is
retained as an audit anchor: the truth integration must reproduce it within a
frozen tolerance, which refuses a case whose mesh, weights, or axis mapping has
drifted. AhmedML can fail closed at 2e-6 because its cell-area integration
reproduces its solver exactly; WindsorML cannot, and pretending otherwise would
have been the wrong contract.

### Replay tolerance

Observed absolute deltas between the published CSV and the truth integration:

| | median | p95 | max |
|---|---|---|---|
| `cd` | 0.000037 | 0.004562 | 0.008926 |
| `cl` | 0.000006 | 0.000780 | 0.003369 |

The audit is `|replay − published| ≤ 0.005 + 0.02 × |published|`. With the
relative term fixed at 0.02, the tightest absolute term that still admits every
published run is 0.002849, so 0.005 keeps about 1.75x headroom.

A purely relative bound would be wrong: lift and side force can be legitimately
near zero. `run_306` publishes `cl = +0.000106`, where a negligible 0.000328
offset is a **310% relative** difference. The absolute term carries small
coefficients; the relative term keeps large ones honest.

## Splits and the scored case set

The dataset ships its own authoritative split manifest at `splits/manifest.json`:
eight families over 355 design variants. But `run_350`–`run_354` have no per-run
payload in the public release, and two of them land in test sets.

FluidsBench therefore scores the **intersection of the official manifest with the
350 published runs**. The official manifest is neither modified nor regenerated;
each split file records the official counts and the excluded IDs.

| family | official test | scored | excluded |
|---|---:|---:|---|
| full / medium / scarce / super_scarce | 36 | 35 | `run_354` |
| geometry | 71 | 71 | — |
| high_drag | 71 | 70 | `run_352` |
| low_drag | 71 | 70 | `run_354` |
| image_wake | 71 | 71 | — |

**235 → 233 unique scored test cases.** WindsorML numbers are therefore not
comparable to any future run against the full official manifest; the spec says so.

The splits README already documents the 355/350 asset gap and imputes geometry and
wake scores for the five, so the *assignment* is deliberate. What it does not
address — and what this reduction fixes — is that a scoring contract needs ground
truth for every test case it scores.


## Profile diagnostics: two placement families

Following the DrivAerML relative-diagnostics contract, every diagnostic kind is
published twice:

| | placement | scoring role | weight |
|---|---|---|---|
| `windsorml_velocity_constant_v1` | fixed absolute coordinates | ranked | 0.15 |
| `windsorml_velocity_relative_v1` | vertical rescaled by body height | report only | 0.0 |
| `windsorml_cp_constant_v1` | fixed absolute coordinates | ranked | 0.10 |
| `windsorml_cp_relative_v1` | vertical rescaled by body height | report only | 0.0 |

WindsorML needs a far smaller relative frame than DrivAerML. Streamwise and
lateral extents are identical in all 350 published runs, so a constant x or z
station is already the same physical location everywhere; **only body height
varies, from 0.316 m to 0.473 m**, so only the vertical coordinate is rescaled
as `eta = y / h_case`.

Both are needed. A constant cut at y = 0.194 m is 61% of the shortest body's
height but 41% of the tallest, so it samples a different part of the roof shear
layer per case and the metric partly measures geometry. A relative cut at fixed
`eta` samples the same position in the shear layer everywhere, but is no longer
a fixed physical probe. Publishing both, with only one weighted, follows the
DrivAerML rule that at most one placement mode per diagnostic kind may carry
nonzero weight.

Relative `eta` values are anchored so each relative station reduces exactly to
its constant counterpart on run_0 (`eta_cut = 0.56491`, `h = 0.34342 m`). That
was verified on real data: the two families produce identical series for all
five velocity stations and the side cut. Only `cp_base_vertical` differs, which
is the intended fix -- the constant line spans y in [0, 0.5] and overshoots the
0.343 m body, while the relative one spans exactly the base.
