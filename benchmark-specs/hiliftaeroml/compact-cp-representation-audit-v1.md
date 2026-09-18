# HiLiftAeroML compact Cp representation decision audit

Date: 2026-09-03
Status: selected candidate, unbound and inactive
Selection: 128 samples per evaluator-owned physical branch graph

## Decision

Freeze the candidate v2 compact Cp representation at a maximum of 128 samples per physical graph. Samples are placed uniformly in native physical arc length on each retained branch. The evaluator owns the cut geometry, branch topology, and graph identity; a submission carries prediction values only. Cp is quantized as `q = int16(round(Cp * 1024))`, with the first value on each retained branch stored absolutely and subsequent values stored as `int16` `cp_q_delta` differences.

This is a scientific selection, not an activation. `native-profile-format-v2.json` remains a candidate format with a 128-point maximum. It is not bound into a release, published, or activated by this audit.

## Why 128

The conservative plot-fidelity gate is a Full360 worst-case prediction visual RMSE of at most 0.02 Cp. The adjacent budgets establish the boundary:

| Samples per graph | Worst visual RMSE (Cp) | Gate | Cp delta-zlib9 bytes | Projected payload bytes* |
| ---: | ---: | :---: | ---: | ---: |
| 64 | 0.0385837256 | fail | 3,953,760 | 9,261,708 |
| 96 | 0.0203768748 | fail | 5,180,276 | 10,488,224 |
| **128** | **0.0153068336** | **pass** | **6,281,236** | **11,589,184** |
| 160 | 0.0135811849 | pass | 7,299,991 | 12,607,939 |
| 192 | 0.0121051945 | pass | 8,249,842 | 13,557,790 |

\* Each projection adds 4,783,039 bytes for Full360 float32 velocity values and 524,909 bytes of existing non-profile compressed payload. It excludes modest JSON and ZIP container overhead.

Thus 96 narrowly misses the gate by 0.0003768748 Cp, while 128 passes with 0.0046931664 Cp of visual-RMSE margin. The larger budgets improve numerical fidelity, but spend another 1,018,755 bytes at 160 or 1,968,606 bytes at 192 on Cp values alone without changing the ranking-stress-test conclusion.

At 128, the current Transolver macro R2 changes from 0.9636931486 on native support to 0.9633831287 before fixed-int16 quantization, a delta of -0.0003100199. Fixed-int16 quantization adds only -6.8597e-7 to macro R2, with a maximum absolute per-case R2 effect of 1.5944e-5.

## Package projection

The selected payload projection is:

```text
 6,281,236  Cp cp_q_delta, zlib level 9
 4,783,039  velocity float32 values, zlib level 9
   524,909  existing non-profile compressed payload
-----------
11,589,184  projected payload bytes before JSON/ZIP overhead
```

This left 3,410,816 bytes before overhead against the original 15,000,000-byte Full360 design target. That target was useful for choosing the representation, but is not an aggregate assembly gate for the larger official case sets. Two deterministic package builds and their actual archive sizes and hashes remain the authoritative final check.

The 524,909-byte non-profile baseline is the sum of compressed member sizes after excluding the 360 surface-Cp and 360 volume-velocity artifacts from the 763-member prior candidate archive. That archive is 2,264,458,600 bytes with SHA-256 `a8dfd6ffbe6d103bc1a3123f6e6f756bacf960f92acc7079237b443bbbb62504`.

## Placement comparison

The 128-point placement study covered five focused difficult cases and six placement policies. Uniform physical-arc placement was chosen because it is deterministic, topology-aware through evaluator-owned branch support, and independent of hidden truth. It also produced the smallest five-case compressed Cp total (91,815 bytes) and a lower mean visual RMSE than native-index placement (0.0108553 versus 0.0112668 Cp).

The slightly lower mean visual RMSE of `hybrid_truth_geometry` (0.0107074 Cp) is not actionable because the policy depends on truth and is incompatible with a prediction-only contract. Native-index placement had a slightly smaller focused-set maximum and score delta, but ties the representation to mesh indexing rather than physical cut distance. Geometry-only and other eligible hybrid variants did not offer a consistent improvement over uniform physical-arc sampling.

## Ranking stress test

Across nine synthetic/current baselines and all 360 cases, 128, 160, and 192 each have 35 concordant pairs and one discordant pair out of 36, for Kendall tau-b 0.9444444444. The sole reversal is a deliberately high-frequency near-tie:

- Native support ranks `half_transolver_error` above `truth_plus_deterministic_noise` by 0.0005581044 macro R2.
- Compact support ranks the deterministic-noise case above the half-error case by 0.0026114911 at 128, 0.0026233525 at 160, and 0.0026225607 at 192.

The deterministic disturbance is `0.05 * sin(native_point_index * sqrt(2) + case_phase)`. It oscillates in native point-index space rather than physical arc length and is outside the plot-bandlimited purpose of this representation. Raising the support budget does not remove its reversal, so 160 and 192 provide no ranking justification for displacing 128.

Every other pair remains concordant. In particular, the perfect-truth, half/scaled-error, smooth-error, current-model, low-frequency, bias, and zero baselines preserve their native ordering. This audit deliberately does **not** claim Kendall tau of at least 0.99 for the nine-method set containing the adversarial native-index oscillation.

## Scoring and lifecycle boundary

Compact profiles are for plotting and profile diagnostics. They do not replace or alter the canonical full-surface inputs, metrics, or ranking path; full-surface scoring remains unchanged.

No assembler, validator, evaluator-core, schema, submission specification, configuration, release binding, publication, activation, or package build is changed by this audit.

## Evidence identity

The source JSON artifacts were generated in `/tmp`; the exact identities used for this retained summary are:

| Evidence | Bytes | SHA-256 |
| --- | ---: | --- |
| `hilift_cp_compaction_full360_uniform_v1.json` (64/96/128) | 1,327,299 | `3c316e5ae1ce4edf08708ec7d756488510f6116cf89915eea3c42addf57e7707` |
| `hilift_cp_compaction_full360_uniform_160_v1.json` | 563,651 | `bdc563b900d0bd00261b4a1f84e40771fde51b981ae8c405a464a347e67d6ea0` |
| `hilift_cp_compaction_full360_uniform_192_v1.json` | 563,710 | `82610c7c2d15b3143c9b8a042b579a24188b23c6f6569ffc30ec620444d3354d` |
| `hilift_cp_adaptive_128_focused_v1.json` (placement) | 56,368 | `713478f1c523e0faef7524d3c39bfa77a56af27901a2cc52f906e0a8bad817bd` |
| `hilift_cp_ranking_stability_full360_v1.json` (128) | 2,471 | `1965222decbec5307b7d927d97619b52d8c24f9ddb2f400556677eb2fc357d29` |
| `hilift_cp_ranking_stability_full360_b160_b192_v1.json` | 23,425 | `a3a71a4422718fe9ecd098624d40c74a7918566c13da373cf23cb5149438781d` |

The companion machine-readable record is `compact-cp-representation-decision-v1.json` in this directory.
