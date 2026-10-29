# HiLiftAeroML dimensional exports

HiLiftAeroML is coming soon. Its package assembler converts native dimensional
field errors to SI once, using the [export contract](dimensional-export-si-v1.json).
The native evaluator's per-case inverse scale is applied before conversion;
case errors are then averaged with equal case weights.

| Diagnostics | Native units | SI conversion |
| --- | --- | --- |
| Surface pressure, wall shear and volume pressure MAE/RMSE | slug/(in*s²) | multiply by 574.5631077637795 to obtain Pa |
| Volume velocity MAE/RMSE | in/s | multiply by 0.0254 to obtain m/s |

Evaluation evidence records this stage exactly once as
`hiliftaeroml-dimensional-export-si-v1`. Values carrying this binding are already
in SI and must not be converted again. Relative errors, R², coefficients,
profiles, regional diagnostics and composite scores retain their own definitions.
No pressure offset is fitted or changed. The original field-evaluator pressure
scale must not be replaced by the load-normalization `qRef`.

The assembler and validator pin the contract's SHA-256. Synthetic regression
cases in `tests/test_hiliftaeroml_dimensional_units.py` check unit factors,
equal-case aggregation, unchanged sufficient statistics and duplicate-conversion
rejection. Those checks do not execute a submitted model or open submissions.
