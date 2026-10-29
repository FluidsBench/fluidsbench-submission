# DrivAerNet++ evaluation guide

Start with the [resumable participant quickstart](QUICKSTART.md) for data
verification, support download, whole-split evaluation, recovery and validated
packaging. See the [dataset index](../README.md) for submission availability.
The [owner approval record](OWNER_APPROVAL.md) describes the scientific definition.
Templates are in the [package example](../../examples/drivaernetplusplus-v3-candidate/README.md).

## Data and scoring contract

| Item | Value |
|---|---|
| Public source | Harvard Dataverse [doi:10.7910/DVN/K7PWNJ](https://doi.org/10.7910/DVN/K7PWNJ) version 1.0, `SurfacePressureVTK` (CC BY-NC 4.0) |
| Source identity | [`public-source-identity/`](public-source-identity/drivaernetplusplus-public-source-identity-v1.json): SHA-256, size and point count of all 1,154 test files |
| Split | [`splits/official_test.json`](splits/official_test.json): the 1,154-design test list of `Mohamedelrefaie/DrivAerNet` commit `8b8c053`, unchanged |
| Predicted field | `p`, kinematic gauge pressure (m²/s²), freestream reference zero; `Cp = p / 450` at U∞ = 30 m/s |
| Scored entities | Every native PointData point, exactly once, in native order (516,881,608 points in total; 362,900–547,646 per design) |
| Primary metric | `surface_pressure_rel_l2`: area-weighted relative L2 (%), per design, macro-averaged over designs |
| Diagnostics | Equal-point relative L2, area-weighted relative L1, equal-point R² pooled over all points |
| Score | `overall_score = clip(100 − surface_pressure_rel_l2, 0, 100)` |

Weights are mass-lumped dual areas: every polygon's fan-triangulated area is
split equally among its vertices. They, the reference pressure (float32,
exactly as stored) and the profile points are the **prepared support**,
produced once by `scripts/publish_drivaernetplusplus_candidate_support.py` from
the public files and distributed through
[`support-distribution-v1.json`](support-distribution-v1.json) as 24 archives
(5.49 GB). The hashed manifest → case index → chunk chain and every per-case
record are in [`scoring-support/`](scoring-support/). Participants download the
support; they never regenerate it.

## Profiles

Three report-only Cp profiles per design are read by the evaluator from the
submitted predictions at frozen native points: `upper_body_centerline`,
`underbody_centerline` and `rear_body_centerline`. Their extraction rules and
full-split validation are in
[`profiles/STATION_DEFINITIONS.md`](profiles/STATION_DEFINITIONS.md). Profiles do not enter the score.

## Known properties of the public data

These are recorded so that results are interpreted correctly; none changes
which points are scored.

* **Localized pressure extremes.** 79 of the 1,154 test designs (49
  detailed-underbody) contain 1–58 points, 768 in total, with `p` beyond the
  physical range: above 460 m²/s² (stagnation is 450, up to 5,735) or below
  −4,500 m²/s² (Cp < −10, down to −27,143 in `F_D_WM_WW_1119`). They cover at most
  0.004 m² per design. The approved definition
  scores them as stored, confirmed by the dataset owner on 2026-10-18.
* **Detached fragment.** `F_D_WM_WW_3850` has a sixth connected component of
  7 points near the roof edge. It is scored like every other point.
* **Unassigned designs.** The release has 8,150 pressure files; 29 designs are
  in no split and absent from the drag-coefficient table because other released
  modalities were incomplete for them. See [`OWNER_APPROVAL.md`](OWNER_APPROVAL.md).

## Status

See [`OWNER_APPROVAL.md`](OWNER_APPROVAL.md) for the approved scientific
definition and owner confirmations, and
[`REAL_DATA_VALIDATION.md`](REAL_DATA_VALIDATION.md) for the full-split
publisher run.

The official `drivaernetplusplus-native-pressure-support-v1` metadata is bound in
`submission-spec.json`; submissions are open. Trial model packages are not part
of the launch repository or leaderboard.
