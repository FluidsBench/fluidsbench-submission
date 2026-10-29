# DrivAerNet++ real-data validation record

Evidence that the pinned public release reproduces the agreed support. This is not
the formal `publication_validation` receipt, which belongs to an owner-approved
official release, but every check below can be reproduced from the public files.

## Full official_test publisher run (2026-10-18)

`scripts/publish_drivaernetplusplus_candidate_support.py` was run on all 1,154
official_test files, in the pinned participant runtime (Linux, Python 3.12.13,
NumPy 2.2.6, VTK 9.5.2).

| Item | Value |
|---|---|
| Source files | 1,154 `SurfacePressureVTK` files, 30,155,164,093 bytes; per-file SHA-256 in `public-source-identity/` |
| Provenance check | `E_S_WW_WM_075` is byte-identical to the file recorded below (SHA-256 `2fca6566…6c57e58`) |
| Scored points | 516,881,608 (362,900–547,646 per design); every point has a positive dual area |
| Support release | `drivaernetplusplus-native-pressure-support-v1-candidate`, 24 archives, 5,490,953,040 bytes |
| Build | 48 workers, about 4 s per design; one design required the fragment rule below |

Checks:

* **Weights.** On the build machine the vectorized dual-area implementation is
  **bit-identical** to the original reference loop in `tools/publish_scoring_support.py`
  for real designs from all three body types. On other CPUs the two agree to
  floating-point round-off (2e-16 relative on CI runners). This is one reason the
  weights are published rather than recomputed by participants. Total
  surface area of `E_S_WW_WM_075` is 29.9306 m², matching the 29.931 m² below.
* **Truth.** `p` is stored exactly as published (float32). Every design reaches at
  least 449.2 m²/s² (stagnation ½U∞² = 450). 79 designs also contain 1–58 points
  outside the physical range, above 460 or below −4,500 m²/s² (see `README.md`).
* **Topology.** 1,153 designs have exactly 5 connected components (body plus four
  wheels). `F_D_WM_WW_3850` has a sixth, 7-point fragment near the roof edge
  (x ≈ 1.04 m, y ≈ 0.62 m, z ≈ 1.28 m). It is scored, never profiled.
* **Orientation.** After VTK's consistency and auto-orientation pass, every component
  of at least 100 points has outward normals on average, in every design.
* **Profiles.** All three stations extract on every design; see
  `profiles/STATION_DEFINITIONS.md`.
* **Scoring path.** For a real Transolver prediction on `E_S_WWC_WM_008`, the
  evaluator (reading the downloaded archive) and an independent path (original VTK
  file, original loop weights, `reference.metrics`) agree on every printed digit:
  13.09847589542854 % area-weighted relative L2.
* **Chain.** `validate_candidate_manifest_release` passes, and the case order of the
  manifest chain equals `splits/official_test.json`.

Reproduce:

```bash
python scripts/publish_drivaernetplusplus_candidate_support.py build \
  --vtk-dir /data/dnpp/SurfacePressureVTK --work /scratch/dnpp-build --workers 48
python scripts/publish_drivaernetplusplus_candidate_support.py package \
  --work /scratch/dnpp-build --archive-dir /scratch/dnpp-archives
```

`support.npz` members are written with fixed ZIP metadata, so their SHA-256 depends
only on array values and NumPy's `.npy` header format. The deflate stream can differ
across zlib builds, which changes archive bytes but not the arrays.

## Original single-file check (2026-09)

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

## Release preparation (2026-10-19)

The 24 archives are hosted at the URLs in `support-distribution-v1.json`. The
owner's scientific confirmations and release-readiness response are recorded in
`OWNER_APPROVAL.md`. Native profile ground truth is deployed on the dev website.

`tools/validate_publication.py` revalidates all archived NPZ arrays against their
retained records, source identities and the manifest chain, and records the
normalized support digest in
`releases/drivaernetplusplus-v1-preparation/publication-validation.json`. This is
full candidate-artifact validation, without rerunning source-mesh extraction or
model inference. The receipt stays separate from the active candidate specification.
Official activation still requires a receipt bound to the final official manifest;
the current evaluation data, dev ranking status and closed intake are unchanged.
