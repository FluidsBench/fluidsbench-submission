# WindsorML FluidsBench readiness review

Date: 2026-09-17

## Executive conclusion

The public WindsorML release contains enough native CFD data to support a real
FluidsBench contract, and this branch delivers one: a pinned source identity, the
eight official split bindings, a bounded-memory evaluator, a split scorer, and a
test suite. The prototype material it replaces was dummy data — a single
fabricated case ID (`windsorml_default_test_0001`), four fabricated velocity
stations, and ten dummy submissions — none of which should ever have been scored.

Two things are **not** ready and are held back deliberately: the profile panels
(the velocity stations were invented and need a real physical choice), and the
cell-volume-weighted volume metrics (the sidecars are unaffordable at this mesh
size). Submissions stay closed pending owner review.

WindsorML is in several respects *easier* than AhmedML: its field associations
are unambiguous, and its surface quadrature weights are published rather than
evaluator-generated. It is harder in exactly one respect — force integration
does not reproduce the published coefficients to machine precision — and that
drove the single most consequential design decision here.

## Pinned inputs

- Public dataset: `neashton/windsorml`
- Immutable revision: `8a6ca32ae22c94f54df2186d1b0ccf9662a294c2`
- Local mirror: `/lustre/.../users/nashton/windsorml/data`

The mirror was downloaded and verified on 2026-09-16: **3,178 files, 7.769 TB**,
every file byte-exact against the Hub-reported size, zero missing. Per-run
`images/` were excluded by design (192,851 files, ~45 GB). Six stray upload-temp
files live in the repo (`run_287/volume_287.vtu.CFC7E38A` and similar, 35.7 GB
total); every affected run also carries its correct payload, so they are excluded
and should be deleted from the Hub separately.

As an independent check, the 350 freshly pulled `boundary_*.vtu` were compared
against a previously downloaded copy at
`datasets/native_surface_boundaries/windsorml/`: 350/350 identical in size, and
byte-identical on spot samples.

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

## What was built

- `reference/windsorml/contract.py` — hash-pinned source identity, frozen axis and
  force convention, force-replay audit. Accepts `run_0` (WindsorML is 0-indexed;
  AhmedML's `run_([1-9][0-9]*)` pattern rejects it).
- `reference/windsorml/evaluator.py` — bounded-memory per-case evaluator over
  inline-binary VTU, chunk- and order-invariant.
- `reference/windsorml/dataset_scorer.py` — split reduction; macro-averaged field
  metrics, pooled coefficient R². Score transforms come from `reference/scores.py`
  so WindsorML cannot disagree with the other datasets about `bounded_error`.
- `reference/windsorml/prediction_chunks.py` — WindsorML-owned transport shim.
- `benchmark-specs/windsorml/` — pinned source identity, eight real split files,
  rewritten `submission-spec.json`.
- `tests/test_windsorml_{contract,evaluator,dataset_scorer}.py` — 38 tests.

### Shared-module change

`reference/drivaerml/prediction_chunks.py` needed three backwards-compatible
extensions, because WindsorML is the first dataset with a **PointData** support
and the first that is 0-indexed:

1. the case-ID pattern now admits `run_0`;
2. `association` is looked up per support rather than hard-coded to `CellData`;
3. the WindsorML supports and format are registered.

Verified neutral: the full suite is **9 failed / 679 passed** both with and
without the change, an identical failure set. Those nine failures are
pre-existing and environmental — they require Python 3.12.13 and the container
provides 3.12.3.

Separately, `tests/test_ahmedml_development_fixture.py` writes into the **live
tracked fixture** that `tests/test_validator.py::test_complete_example_is_valid`
then validates, so that test passes or fails depending on execution order. It is
unrelated to WindsorML but worth fixing.

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

## Deliberately deferred

- **Nothing. Profile support is complete.** All 233 scored cases are generated,
  validated and hash-pinned by
  `profile-support/windsorml-profile-support-v3-manifest.json`. Every case
  reports exactly 8 containment fallbacks, which is the expected pattern: the
  eight vertical stations each begin at y = 0 on the ground plane, outside the
  fluid mesh, while the two lateral stations report none.
- **Cell-volume-weighted volume metrics are removed, not stubbed.** At ~291M
  cells × 350 cases the sidecars cost roughly 400 GB and 1,000 CPU-hours. The
  primary volume metric was already equal-cell weighted.

## Release state

Every technical decision is settled and recorded in the spec: the scored case
set, the component weights, the stations and resolution, and the sampling
semantics. `scoring_support.status` remains `owner_review_required` and
`submissions_open` remains false, because the repository validator holds those
to a closed vocabulary and requires a non-empty decision list; the single
remaining entry is `approve_release_and_open_submissions`, which is a release
decision rather than a technical one.

Measured body height across all 233 scored cases is **0.29957 m to 0.48888 m**,
wider than the 0.31574-0.47340 m taken from an eight-run sample during design.
The constant horizontal cut at y = 0.194 m still lies inside every body, and the
spread it motivates is slightly larger than first documented: 65% of the
shortest body's height against 40% of the tallest.

## Owner decisions required

1. Approve the 235 → 233 scored-case reduction.
2. Review the relative placement families and decide whether either should ever
   take the ranked weight in place of the constant family.
3. ~~Approve component weights.~~ Approved 2026-09-17, unchanged at field 0.50 /
   force 0.25 / diagnostic 0.25. That is identical to AhmedML and HiLiftAeroML
   component for component, and to DrivAerML apart from its force group, which
   splits three ways (cd 0.15, cl 0.05, c_pitch 0.05) because it scores pitching
   moment. WindsorML publishes `cmy`, so a `c_pitch_r2` term is supported by the
   data, but it would need surface moment integration and a moment reference
   point the dataset does not publish. Not adopted.
4. Decide whether `geo_parameters_all.csv` or the per-run CSVs are authoritative —
   the aggregate omits `ratio_length_front_rear`, which the splits README
   documents and works around, but participants need to be told which to use.
5. Delete the six stray `.vtu.XXXXXXXX` upload temporaries from the Hub.

## Release acceptance gates

- Source identity, splits, weights, units, axis and force conventions immutable
  and documented.
- Every scored test case has complete source support.
- Full native entity coverage enforced without loading a case into memory.
- Metrics invariant to valid prediction chunking and ordering.
- Synthetic golden tests and a real run_0 smoke test pass.
- No dummy submissions or fabricated stations anywhere in the dataset's material.
