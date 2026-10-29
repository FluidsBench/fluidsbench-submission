# WindsorML evaluation guide

Start with the [resumable participant quickstart](QUICKSTART.md) for data/support
preparation, whole-split evaluation, recovery and validated packaging.

See the [dataset index](../README.md) for submission availability and the
[packaging guide](../../examples/windsorml-v3-candidate/README.md).

## Data and scoring contract

Use `neashton/windsorml` at revision
`8a6ca32ae22c94f54df2186d1b0ccf9662a294c2`. The checked-in
[public source identity](public-source-identity/windsorml-public-source-identity-v1.json)
pins every source file's path, size, SHA-256 and native entity counts.
The public payload contains **350 runs, `run_0` through `run_349`**.
Runs 350–354 appear in aggregate metadata but have no published per-run fields.

Use the exact ordered `case_ids` in `splits/<split-id>.json`. These preserve the
official assignments intersected with published runs; do not replace missing
cases. Scores on these reduced sets are not comparable to results using the
unreduced official sets.

| Split | Available test cases | Official test cases | Excluded test IDs |
| --- | ---: | ---: | --- |
| full, medium, scarce, super_scarce | 35 | 36 | run_354 |
| geometry | 71 | 71 | None |
| high_drag | 70 | 71 | run_352 |
| low_drag | 70 | 71 | run_354 |
| image_wake | 71 | 71 | None |

The eight labels share five test sets, covering 233 distinct cases. Training
must still follow the selected label's source assignment and the available
published runs. The four baseline labels share evaluation cases, not training
sets.

| Domain | Native association | Required model predictions | Scoring weights |
| --- | --- | --- | --- |
| Surface | **PointData**, every boundary point | `cpavg`, `cfxavg`, `cfyavg`, `cfzavg` (dimensionless) | Published `boundary_dual_area_N.npy`; equal-point diagnostics also reported |
| Volume | **CellData**, every volume cell | `velocityxavg`, `velocityyavg`, `velocityzavg` (m/s), `pressureavg` (raw published values) | Equal native cell |

The volume pressure units/reference remain unresolved in the
[pressure audit](../../docs/PRESSURE_REFERENCES.md#windsorml). Predict the raw
published `pressureavg` values; do not apply a new offset, density scaling or
surface Cp conversion. This workflow preserves that existing evaluator rule.

Do not regenerate surface dual areas. Volume cell-volume sidecars are not
required. Vector relative L2 sums squared component errors and truth before
one square root; it does not average component L2 values. Chunk statistics
are added before reducing each case, then cases are macro-averaged equally.

Forces use **drag +x, lift +y, side +z**, and reference area **0.112 m²**.
Both truth and predictions are integrated from native fields. Published force
CSVs audit the integration; they are not substituted as scoring truth.

The evaluator derives all **16 profiles × 128 samples per case** from those
same complete fields. Fixed placement supplies five velocity and three Cp
series used in the diagnostic score. The corresponding eight relative series
are reported with zero score weight. Native IDs, coordinates, and truth are
already frozen for all 233 cases in `profile-support/`. Submitters do not write
profile values separately or regenerate the support.

## Download only the required data

For one case, using the Hugging Face CLI:

```bash
hf download neashton/windsorml --repo-type dataset \
  --revision 8a6ca32ae22c94f54df2186d1b0ccf9662a294c2 \
  --include 'run_7/boundary_7.vtu' 'run_7/boundary_dual_area_7.npy' \
            'run_7/volume_7.vtu' 'run_7/geo_parameters_7.csv' \
            'run_7/force_mom_7.csv' \
  --local-dir /path/to/windsorml
```

Use the source identity's exact paths for every selected case; model inputs
may require additional geometry files. The evaluator checks pinned source
bytes and reads native fields in bounded chunks. Each volume still contains
roughly 267–330 million cells, so allow substantial disk, memory and evaluation
time. Do not attempt a full VTK mesh load merely to package the result.

## Maintainer support metadata

The official `windsorml-native-support-v1` manifest is bound in `submission-spec.json`.
The publisher below is a maintainer preparation tool; a new official
release still requires review and publication.
Its compact case metadata is reproducible from the checked-in source identity
and profile support; it does not regenerate quadrature or resample profiles:

```bash
python scripts/publish_windsorml_candidate_support.py --output /tmp/windsor-support
```

The metadata covers every scored native point/cell, derived force support and
all four profile families. WindsorML is open for submissions. Use the participant quickstart to package
results and the shared submission workflow to open a pull request.

See [scientific notes](SCIENTIFIC_NOTES.md) for field associations, coordinate conventions, force integration and split construction.
