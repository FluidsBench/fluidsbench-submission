# DrivAerNet++ dataset-owner approval (interim record)

The dataset owner's scientific approval is recorded here while the scoring support is still
`owner_review_required`. It **cannot** yet be written into `submission-spec.json` as the
`owner_approval` field: `scripts/validate_scoring_supports.py` lists `owner_approval` and
`publication_validation` among the fields a non-official specification must not publish. The
values below move into the specification in the same change that sets
`scoring_support.status` to `official`.

## Approval

| Field | Value |
|---|---|
| `approved_by` | Mohamed Elrefaie (DrivAerNet++ dataset author) |
| `approved_at` | 2026-09-04 |
| `pull_request_url` | https://github.com/neilashton/fluidsbench-submission/pull/2 |

## What is approved

The scientific definition of the pressure-only v1 benchmark as it stands on this branch:

- the pinned public source (Harvard Dataverse `doi:10.7910/DVN/K7PWNJ` version 1.0,
  `SurfacePressureVTK/DrivAer_<ID>.vtk`, PointData `p`), and the splits pinned at
  `github.com/Mohamedelrefaie/DrivAerNet` commit `8b8c053`;
- the pressure convention: kinematic **gauge** pressure in m²/s² with the freestream
  reference at zero, so `Cp = p / 450` is exact at U∞ = 30 m/s;
- the official evaluation split: the 1,154-case test list, ordered lexicographically;
- the scoring support: every native surface point exactly once, with mass-lumped dual-area
  weights, and the area-weighted relative L2 as the primary ranking metric;
- `overall_score = 100 − surface_pressure_rel_l2` for v1;
- retaining all four pressure-profile stations, subject to the extraction rules in
  `profiles/STATION_DEFINITIONS.md` being validated on real geometries.

## What is not yet approved

Approval of the definition does **not** by itself open submissions. Still outstanding:

1. the immutable scoring-support release (manifest, case-set indexes, checksums) produced by
   a full-split publisher run, and its `publication_validation` receipt;
2. validated profile extraction across all configuration families, and the resulting public
   profile ground truth in the website repository;
3. retirement of the prototype `default` split and the placeholder leaderboard rows;
4. documentation of why 29 designs (8,150 referenced vs 8,121 in the split files and drag CSV)
   are not assigned to any split. As of 2026-09-04 this is not stated publicly in either the
   DrivAerNet or CarBench repositories; the owner will confirm the reason and it will be
   recorded here and in the CarBench documentation.
