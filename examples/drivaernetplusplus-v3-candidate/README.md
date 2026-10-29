# DrivAerNet++ submission package

Templates for the participant workflow in
[`benchmark-specs/drivaernetplusplus/QUICKSTART.md`](../../benchmark-specs/drivaernetplusplus/QUICKSTART.md).

| File | Use |
|---|---|
| `entry.template.json` | Copy to `my-entry/entry.json`; set `submission_id` (lowercase letters, digits, hyphens). |
| `package-config.template.json` | Copy to `my-entry/package-config.json`; replace every `__REPLACE_` value. Keep `release_bindings` exactly as published: they pin the scoring support, evaluator and profile ground truth, and `package` refuses a mismatch. |

`spatial_discretization` follows the schema-v3 discretization contract
(`schemas/v3/discretization.schema.json`). Volume representations are always
`{"used": false}` for this pressure-only benchmark. The single mapping must
target `drivaernetplusplus-surface-native-points-v1` with full coverage; use
`identity` when your model predicts native points directly and declare any
interpolation (`nearest`, `linear`, ...) otherwise.

Submit the generated package following the [submission workflow](../../SUBMITTING.md).
