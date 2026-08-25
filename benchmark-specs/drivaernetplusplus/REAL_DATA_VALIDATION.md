# DrivAerNet++ real-data validation record

Interim evidence that the pinned public release reproduces the agreed support. This is not
the formal `publication_validation` receipt — that requires the official immutable release
manifest — but it records the checks already run against a real released file, so they can
be reproduced independently.

## Source file checked

| Item | Value |
|---|---|
| Case ID | `E_S_WW_WM_075` (estateback, smooth underbody, WW wheels) |
| Public file | `SurfacePressureVTK/DrivAer_E_S_WW_WM_075.vtk` |
| Release | Harvard Dataverse `doi:10.7910/DVN/K7PWNJ` version 1.0 |
| Format | legacy VTK **BINARY** PolyData |
| Size | 26,013,034 bytes |
| SHA-256 | `2fca6566b41ae63cc8d4450f728629f3f8cf6b521038bd50ccf12bd2f6c57e58` |
| Point arrays | `p` only |
| Points / cells | 481,363 / 442,114 |

The file is **not stored in git**. It is CC BY-NC 4.0 upstream data (this repository is
Apache 2.0) and is obtained on demand from the pinned release into `samples/`, where it is
verified against the SHA-256 above — see [`samples/README.md`](samples/README.md) for the
procedure, attribution and reuse conditions. Only its checksum and the measurements derived
from it are recorded here.

## Publisher run

`tools/publish_scoring_support.py` materialized the complete native support for this case and
passed its self-checks:

```
materialized E_S_WW_WM_075: 481363 points
SELF-CHECK PASS: 1 case(s), identity/perturbation/chunk checks green
```

Checks covered: every native point represented exactly once; identity prediction scores
exactly zero (and R² = 1); a single-point perturbation reproduces the hand-calculated
relative L2; chunked and single-pass sufficient statistics agree exactly.

Materialized footprint: **~11 MB per case** (NPZ), i.e. roughly **12.5 GB** for the full
1,154-case evaluation split — the release therefore needs external hosting, not repository
storage.

## Confirmed dataset facts

| Question | Finding |
|---|---|
| Pressure convention | Kinematic **gauge** pressure, freestream reference zero. Measured `max(p) = 449.41 m²/s²` at the stagnation point, matching `½U∞² = 450` at U∞ = 30 m/s. `Cp = p / 450` is therefore exact. |
| Axis orientation | `+x` front→rear (stagnation at `x_min`), `y` lateral and symmetric about 0, `+z` up with the ground plane at `z = 0`. Bounding box: length 4.646 m, width 1.985 m, height 1.449 m. |
| Polygon winding | **Not consistent.** On a flat roof patch (10,342 cells) the stored cell normals split exactly 50/50 between `+z` and `−z`. Outward normals require an explicit orientation pass; connectivity order alone is not usable. |
| Surface topology | Exactly **5 connected components**: body (401,931 points) plus 4 wheels (~19,800–19,917 points each). Front wheels centred near `x ≈ 0.00`, rear near `x ≈ 2.70`, radius `R ≈ 0.31 m`, at `|y| ≈ 0.76 m`. |
| Open boundaries | 18 small loops, all at `z = 0` with radius ≈ 0.02–0.03 m — tyre/ground contact patches, not wheel-arch openings. |
| Dual-area weights | Total surface area 29.931 m². Point dual areas: median 6.77e-05 m², max/median ≈ 2.1×, coefficient of variation 0.242; the largest 10% of points hold 13.5% of total area. The mesh is close to uniform. |
| Weighting sensitivity | For error uncorrelated with cell size, area-weighted and equal-point relative L2 differ by ≈ 0.2 percentage points; when error concentrates on large-area points the gap grows to ≈ 3 pp. Equal-point published results are therefore indicative of, but not substitutable for, the area-weighted primary metric. |

## Consequences already applied

* `profiles/STATION_DEFINITIONS.md` now requires an explicit normal-orientation pass before
  side classification, and defines the wheelhouse station from connected components rather
  than from an unspecified "wheel patch".

## Still to validate on real data

* Profile extraction for all four stations on representative cases from every configuration
  family (`E_S_*`, `F_S_*`, `N_S_*`, `F_D_WM_WW_*`).
* A full-split publisher run producing the official manifest, checksums, and the formal
  `publication_validation` receipt.
