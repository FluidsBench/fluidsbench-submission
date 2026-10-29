> **Historical preparation record.** Status and branch references below describe earlier trials and release preparation. For current submission availability, use the [dataset index](../README.md) and the dataset quickstart.

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
- `overall_score = clip(100 − surface_pressure_rel_l2, 0, 100)` for v1;
- the three report-only pressure-profile stations confirmed below, with extraction
  rules in `profiles/STATION_DEFINITIONS.md`. This supersedes the initial conditional
  proposal to retain four stations.

## Owner confirmations (2026-10-18)

Recorded by Mohamed Elrefaie (DrivAerNet++ dataset author), answering the three
questions raised on pull request #59 before official activation. The confirmations
are recorded in [PR #61](https://github.com/neilashton/fluidsbench-submission/pull/61)
and the [owner's response](https://github.com/neilashton/fluidsbench-submission/pull/59#issuecomment-6256964063).

1. **Profile stations.** `front_wheelhouse` is retired. The three validated
   report-only stations (`upper_body_centerline`, `underbody_centerline`,
   `rear_body_centerline`) are retained as defined in
   `profiles/STATION_DEFINITIONS.md`. Reason: on the detailed-underbody family
   (`F_D_WM_WW`, 595 of the 1,154 test designs) the wheel-centre plane misses the
   arch liner, so the station cannot be defined consistently across the split.
   Profiles do not enter the score.
2. **Localized pressure extremes.** The 768 points in 79 test designs outside the
   physical pressure range (see `README.md`) stay scored exactly as stored, with no
   masking. Reasons: this is the approved definition (every native point of the
   released file); the points are part of the public CFD release, and equivalent
   artefacts occur in 625 of the 8,121 published designs (7.7%), training designs
   included; and their split-level effect is small (0.26 points of mean area-weighted
   relative L2 for the dev Transolver result). Their per-design effect is documented
   so results can be interpreted.
3. **The 29 unassigned designs.** They were removed because one or more of the other
   released modalities (3D meshes, wall shear stress, full CFD fields, or annotations)
   was missing or incomplete for them. DrivAerNet++ splits and the drag table cover
   only designs with every modality. The repository history agrees: 28 of the 29 were
   in the June 2024 splits (20 train, 7 validation, 1 test) and were removed in the
   January 2025 revision that introduced the current splits and
   `DrivAerNetPlusPlus_Cd_8k_Updated.csv`. `F_S_WWS_WM_516` was never assigned. Their
   pressure fields are unremarkable: 5 connected components, finite values, normal
   stagnation pressure, and pressure drag within 2.2 standard deviations of their
   family. The official test list is used unchanged.

## Release preparation and activation

Approval of the definition and the confirmations above do **not** by themselves open
submissions. The following evidence is complete on dev:

1. **Scoring-support release.** `drivaernetplusplus-native-pressure-support-v1-candidate`
   (manifest, case index, chunks, 1,154 per-case records) is complete, and its 24
   archives are hosted on the `drivaernetplusplus-support-v1-dev` release with
   GitHub-reported SHA-256 values matching `support-distribution-v1.json`. Also
   prepared: the [approval record](releases/drivaernetplusplus-v1-preparation/approval-record.json)
   and [full publication-validation evidence](releases/drivaernetplusplus-v1-preparation/publication-validation.json).
   These bind the candidate release without making it official.
2. **Public profile ground truth: deployed.** All 1,154 cases and 3,462 native Cp
   traces are available on the dev review website, merged in
   [website PR #69](https://github.com/neilashton/fluidsbench/pull/69). Frozen native
   point IDs and coordinates match the submitted profiles; no interpolation is used.
3. **Prototype split and placeholder rows: retired.** The genuine Transolver result
   is registered as an unranked dev reference.

Remaining work is the separate official-release transition: select the final release
identities, bind the approval and validation receipt to that exact manifest, update
the evaluator/participant contracts and website together, then explicitly open intake.
Wall shear and volume remain outside this pressure-only v1 release. See the
[release preparation notes](releases/drivaernetplusplus-v1-preparation/README.md).
