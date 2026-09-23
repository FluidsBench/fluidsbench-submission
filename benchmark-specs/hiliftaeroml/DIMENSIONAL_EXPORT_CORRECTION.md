# HiLiftAeroML dimensional export correction v1

The original 23 preview packages exported eight field MAE/RMSE diagnostics in
solver-native units even though the specification labelled them Pa and m/s.
The current ICLR manuscript's `scripts/build_hilift_absolute_error_assets.py`
already applies the correction. An independent replay checked all 8,142 case
records against that manuscript's corrected table.

The FluidsBench assembler now preserves each complete case's native evaluator
inverse scale, converts its dimensional error to SI, then averages cases equally:

| Diagnostics | Native units | SI conversion |
| --- | --- | --- |
| Surface pressure, surface wall shear, volume pressure MAE/RMSE | slug/(in*s²) | multiply by 574.5631077637795 into Pa |
| Volume velocity MAE/RMSE | in/s | multiply by 0.0254 into m/s |

For example, GeoTransolver full-split surface pressure RMSE changes from
0.1519601920160553 to **87.3107201811254 Pa**, and volume velocity RMSE from
104.57217679910636 to **2.6561332906973014 m/s**. These numbers describe the
same predictions. Relative L1/L2, R2, coefficients, composite scores, ordering,
profiles, nondimensional regional diagnostics, weights and validity masks are
unchanged. No pressure offset is fitted or changed. The original per-case
pressure scale is retained; it is not replaced with nominal load `qRef`.

## Version and provenance

- [Export contract](dimensional-export-si-v1.json): exact metric inventory,
  factors and manuscript source identity. The assembler and validator pin its
  digest. Evaluation evidence must identify this export stage exactly once,
  separately from the frozen native evaluator.
- [Correction registry](dimensional-export-correction-v1.json): original and
  corrected submission, evidence, case-data and deterministic ZIP identities.
  Its digest is pinned in the validator's conversion helper.
- Original submission metadata and evidence are retained byte for byte under
  `dimensional-export-correction-v1/sources/`. Historical validation receipts
  and their code-pinned registrations remain unchanged. Original case data
  remain available at source revision `cb64dacb296e467bc706de19093142c66b4056da`.
- The same submission IDs identify the same model runs. The separate export
  version and the regenerated content-addressed feed release distinguish the
  corrected package from the original immutable feed.

The correction does not approve the dataset, open submissions, create a new
model evaluation, or make preview results eligible for official claims.

## Reproduce and check

`scripts/correct_hiliftaeroml_dimensional_exports.py` checks the original whole
ZIPs against their registrations before deriving the corrections. On the
audited native source checkout, run it with `--apply`; review and pin the emitted
registry digest before validating or rebuilding feeds. Without `--apply` it
only checks. Once corrected, either invocation verifies the registered bytes
and never applies the conversion a second time.

```bash
python scripts/correct_hiliftaeroml_dimensional_exports.py
python -m unittest tests.test_hiliftaeroml_dimensional_units
python scripts/manage_leaderboard.py check
```

Future packages must be assembled with the corrected assembler from native
evaluator outputs. Do not multiply values already labelled by this SI export
binding or change the native predictions, relative statistics or profile files.
