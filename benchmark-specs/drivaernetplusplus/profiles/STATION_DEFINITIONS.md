# DrivAerNet++ pressure-profile stations (v1, validated on official_test)

Status: **frozen extraction rules, validated on all 1,154 official_test designs, and
confirmed by the dataset owner on 2026-10-18** (three stations; `front_wheelhouse` retired). The rules are implemented in
[`reference/drivaernetplusplus/native.py`](../../../reference/drivaernetplusplus/native.py)
(`extract_profile_stations`, rules version `drivaernetplusplus-native-support-rules-v1`).
They run once, when the support is published, and freeze one ordered list of native
point indices per station and design in each case's `support.npz`. The evaluator
reads the submitted prediction at those points, so a profile is an exact subset of
the scored field: participants submit no profile artifact, and nothing is
interpolated. Profiles are report-only and do not enter the score.

## Shared conventions

- Source: `SurfacePressureVTK/<ID>.vtk`, PointData `p` (kinematic gauge pressure,
  m²/s²), coordinates in metres. Quantity `cp = p / 450`.
- Per design: `x_min`, `z_min` and `L = x_max − x_min` from that design's own points.
- Connected components by polygon connectivity. The body is the largest. Every design
  has at least 5 (body plus four wheels); `F_D_WM_WW_3850` has one extra 7-point
  fragment. Every component of at least 100 points must have outward normals on
  average (mean of `n · (x − centroid)` positive).
- Normals: VTK `vtkPolyDataNormals` on the triangulated surface with consistency and
  auto-orientation on, splitting off (point order preserved and checked). The stored
  polygon winding is inconsistent, so connectivity order alone cannot be used.
- Centreline band: body points with `|y| ≤ 2e-3 · L`.
- **Outer envelope.** The band is multi-valued wherever the centre plane meets more
  than one wetted surface: spoiler undersides, the cowl step, and, on the detailed
  underbody, engine-bay and tunnel walls. Each station therefore keeps exactly one
  native point per `0.005 · L` bin of its coordinate: the extreme point in the
  station's outward direction. Ties go to the lowest native index. Coordinates are
  strictly increasing by construction.

## Stations

| Station | Candidate points | Envelope (per bin) | Coordinate |
|---|---|---|---|
| `upper_body_centerline` | band, `n_z > 0.2` | highest `z` | `(x − x_min)/L` |
| `underbody_centerline` | band, `n_z < −0.2` | lowest `z` | `(x − x_min)/L` |
| `rear_body_centerline` | band, `n_x > 0.5`, `(x − x_min)/L ≥ 0.9` | rearmost `x` | `(z − z_min)/L` |

The rear-region condition is needed because the cowl step at the windscreen base and
the roof trailing edge also face rearward on the centreline. Without it the trace
jumps forward along the car.

**`front_wheelhouse` is retired.** At the wheel-centre plane `y = c_y`, the smooth
families show a clean arch-liner arc. On the detailed-underbody family `F_D_WM_WW`,
however, the plane misses the arch liner entirely and meets only the front and rear
walls of the wheel well. This is the failure case the draft named, and per the draft
the station is dropped rather than defined ambiguously.

## Validation on all 1,154 official_test designs

Every design passed extraction. Points per station:

| Family | Designs | upper | under | rear |
|---|---:|---:|---:|---:|
| `E_S_WWC_WM` | 118 | 200–201 | 200 | 28–48 |
| `E_S_WW_WM` | 92 | 200–201 | 199–200 | 27–47 |
| `F_D_WM_WW` | 595 | 200–201 | 199–201 | 25–39 |
| `F_S_WWC_WM` | 94 | 200–201 | 200 | 24–36 |
| `F_S_WWS_WM` | 89 | 200–201 | 200 | 25–35 |
| `N_S_WWC_WM` | 49 | 200–201 | 200 | 26–35 |
| `N_S_WWS_WM` | 55 | 200–201 | 200 | 25–36 |
| `N_S_WW_WM` | 62 | 200–201 | 200 | 25–35 |

Checks against the draft's list:

1. Visual review of every station on two or more designs per family, covering every
   wheel variant (WW, WWC, WWS) and both underbodies. Upper-body and rear-base
   traces are clean and contiguous on every family; so is the underbody on the
   smooth families. On `F_D_WM_WW`, the lowest downward-facing centreline surface
   genuinely crosses open cavities between about x/L = 0.35 and 0.65. The trace is
   deterministic, but its Cp is jagged there.
2. Station point counts are stable (table above); the minimum is 24 and the rule
   fails a design with fewer than 20.
3. Strict ordering holds by construction and is checked for every design.
4. Cp is `p / 450` read at the frozen native points, the same values that are scored.
5. Thresholds `2e-3·L`, `n_z ±0.2`, `n_x 0.5` are kept from the draft; `0.005·L` bins
   and `x/L ≥ 0.9` are new and frozen with the rules version.
