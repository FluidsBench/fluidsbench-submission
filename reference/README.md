# Reference metric implementation

Participants own model inference, prediction dimensionalisation, any mapping from their native output to the canonical scoring
locations, and profile extraction. Benchmark-owned scoring-support files provide the exact ground-truth arrays, coordinates,
weights, and stable IDs used in the final comparison. [`evaluate_predictions.py`](evaluate_predictions.py) implements the bound
field, scalar, cross-case, and dataset-reference reductions after predictions have been joined to that support. It does not
prescribe a model's native input or output format.

## Fixed scoring support

For a schema-v3 result, the selected dataset specification points to one immutable, owner-approved scoring-support release. Its
hash chain defines the public case set and, for each support, the exact stable IDs, coordinates, area, length, or volume weights,
quantities,
components, ground-truth values, and coverage and extrapolation rules used for scoring.

The submitter creates keyed predictions and maps any native model output to those benchmark support IDs. The reference evaluator
uses benchmark-owned coordinates, weights, and ground truth, rejects missing, duplicate, or unknown IDs, and then writes
`metrics/cases.json`. It does not interpolate or resample either side of the comparison. Any interpolation, nearest-neighbour
transfer, projection, or direct query needed to reach the support must therefore happen in the submitter's evaluation code and be
declared in `discretization.json`.

The generic loader and synthetic executable example are in [`scoring_support.py`](scoring_support.py) and
[`evaluate_predictions.py`](evaluate_predictions.py). The loader currently handles safe local `materialized_table` JSON, CSV, and
NPZ artifacts; an official release may provide a reviewed dataset-specific implementation for another declared location mode.

```bash
python3 -m reference.evaluate_predictions \
  --support-manifest examples/v3-template/support/manifest.json \
  --case-set standard \
  --prediction-manifest examples/v3-template/predictions/manifest.json \
  --submission-id synthetic-open-model-v1 \
  --split-id default \
  --output /tmp/fluidsbench-synthetic-case-metrics.json
```

The evaluator follows each binding's published aggregation rule. Ordinary L1, L2, MAE, MSE, and RMSE field metrics produce
per-case evidence and are macro-averaged when the contract says so. Global R2 and the benchmark field/scalar RRMSE rules are
calculated from the complete cross-case arrays and are marked `aggregate_only`, so no misleading per-case surrogate is emitted.
The output also contains support counts, count and weight coverage, unmapped and extrapolated counts, and additive sufficient
statistics for every relative-L2 binding. The repository validator requires exact official case and support coverage, verifies
each relative-L2 value against those statistics, and recalculates macro-averages from submitted per-case evidence. Aggregate-only
values remain submitter-created values bound to the exact rule and support release. Required approval validates the submitted
package but does not execute the model or regenerate its predictions.

Before approving a scoring-support proposal, a dataset owner can validate its local manifest, case-set index, chunks, artifact
paths and hashes, identities, support coverage, and normalized support digests with:

```bash
python3 -m reference.scoring_support \
  --manifest benchmark-specs/<dataset-id>/scoring-support/<release-id>/manifest.json \
  --case-set <case-set-id>
```

The command prints `PASS` only after the complete local chain has loaded successfully. Dataset-owner approval and the
`submission-spec.json` status change to `official` remain explicit review decisions; this command does not make that policy change.
The repository-wide `python3 scripts/validate_scoring_supports.py` gate additionally checks that every official manifest binds
exactly the executable metrics in its dataset specification. Releases using remote artifacts or a dataset-specific loader must
also commit a `publication_validation` receipt that pins the validator file and digest, manifest digest, normalized-support digest,
validated case/support counts, reviewer, timestamp, and approving pull request.

## Common equations

For ground truth \(y_i\), prediction \(\hat y_i\), and non-negative weights \(w_i\):

\[
\operatorname{MSE}_w = \frac{\sum_i w_i(\hat y_i-y_i)^2}{\sum_i w_i}
\]

\[
\operatorname{MAE}_w = \frac{\sum_i w_i |\hat y_i-y_i|}{\sum_i w_i}
\]

\[
\operatorname{RMSE}_w = \sqrt{\frac{\sum_i w_i(\hat y_i-y_i)^2}{\sum_i w_i}}
\]

\[
L_{1,\mathrm{rel}}(\%) = 100\frac{\sum_i w_i|\hat y_i-y_i|}{\sum_i w_i|y_i|}
\]

\[
L_{2,\mathrm{rel}}(\%) = 100\frac{\sqrt{\sum_i w_i(\hat y_i-y_i)^2}}{\sqrt{\sum_i w_i y_i^2}}
\]

\[
R^2_w = 1-\frac{\sum_i w_i(y_i-\hat y_i)^2}{\sum_i w_i(y_i-\bar y_w)^2},\qquad
\bar y_w=\frac{\sum_iw_i y_i}{\sum_iw_i}
\]

An unweighted metric sets \(w_i=1\), so every stored point, face, or cell counts equally. An area-, length-, or volume-weighted
metric uses the corresponding benchmark-supplied measure represented by each scoring point, face, or cell. Supported
dataset-weighting tokens are `surface_face_area`, `cell_volume`, `boundary_line_length`, `interior_cell_area`,
`surface_point_dual_area`, `volume_point_dual_volume`, `boundary_point_dual_length`, and `interior_point_dual_area`. Face- and
cell-associated values can use their face area, boundary-line length, cell area, or cell volume directly. Point-associated values
need the corresponding benchmark-published dual area, dual length, or dual volume; a submitter must not derive a competing set of
weights. The geometric measure appears once in each weighted sum, not squared.

Dataset specifications publish which variant is primary and which is secondary. They also state whether a metric is computed
globally or per geometry followed by a macro-average. Macro-averaging is the default for field metrics so each test geometry has
equal influence regardless of mesh resolution. Coefficient metrics normally use equal case weights.

For per-geometry values \(m_k\), the macro-average over \(K\) test geometries is:

\[
\bar m = \frac{1}{K}\sum_{k=1}^{K}m_k
\]

Vector quantities must include every declared component. Apply the same benchmark-supplied spatial weight to each component and flatten the
component axis before evaluating the norm unless the dataset specification states another reduction.

### Chunk-safe relative L2

Inference may be split into any number of chunks, but a chunk-level L2 value is not itself additive. For each case, support, and
relative-L2 metric, accumulate these four sufficient statistics:

\[
N=\sum_i w_i\lVert\hat{\mathbf y}_i-\mathbf y_i\rVert_2^2,
\qquad
D=\sum_i w_i\lVert\mathbf y_i\rVert_2^2,
\qquad
n=\#\{i\},
\qquad
W=\sum_iw_i.
\]

For chunks \(c\), calculate the case value once from the sums:

\[
L_{2,\mathrm{rel}}(\%)=100\sqrt{\frac{\sum_c N_c}{\sum_c D_c}}.
\]

Never average chunk L2 values. `entity_count` records \(n\), and `total_weight` records \(W\), both over spatial entities rather
than flattened vector components. For uniform weighting, `total_weight` must equal `entity_count`. The case-metrics file stores
these values under each support's `metric_sufficient_statistics`, keyed by metric ID. When the published cross-case rule is
`per_geometry_then_macro_average`, reconstruct each complete case first and only then average the case metrics.

Inputs must be dimensional where the metric unit is dimensional. The functions reject empty arrays, non-finite numbers, negative
weights, zero total weight, zero relative-error denominators, and constant-ground-truth R2 rather than silently inventing a value.

VKI-LS59 and Rotor37 additionally use the following benchmark-specific RRMSE equations. For field case \(k\) with
\(N_k\) values:

\[
\operatorname{RRMSE}_{field} = \sqrt{\frac{1}{K}\sum_{k=1}^{K}
\frac{\lVert\hat{\mathbf y}_k-\mathbf y_k\rVert_2^2/N_k}{\lVert\mathbf y_k\rVert_\infty^2}}
\]

For scalar outputs:

\[
\operatorname{RRMSE}_{scalar} = \sqrt{\frac{1}{K}\sum_{k=1}^{K}
\frac{|\hat y_k-y_k|^2}{|y_k|^2}}
\]

## Benchmark scores

[`scores.py`](scores.py) evaluates the dataset-declared composite used as the ranking metric for every dataset. For each error
component \(e_k\) with published cap \(c_k\), or bounded quality component \(q_k\), the component score is

\[
S_k^{error}=\operatorname{clip}(100(1-e_k/c_k),0,100),\qquad
S_k^{quality}=100\operatorname{clip}(q_k,0,1).
\]

For non-negative declared weights \(\alpha_k\) that sum to one:

\[
S_{overall}=\sum_k \alpha_k S_k.
\]

Some benchmark contracts instead use an explicit physics-null reference error
\(B_k>0\):

\[
S_k^{null}=100(1-e_k/B_k).
\]

This transform is not clipped. A perfect prediction scores 100, the frozen null
prediction scores 0, and a method worse than that null retains negative skill.
The component is executable only after the specification publishes its finite,
strictly positive `baseline_error`; a candidate contract may keep the composite
at `pending_reference_baselines`, in which case submitted prototype score values
are not recomputed or eligible for ranking.

The exact component metric IDs, transforms, caps or baseline errors, and weights live in each dataset's `overall_score_composite` object. The
validator recalculates the composite from those component values. External-aerodynamics specifications additionally partition
those components into field, force, and diagnostic groups through `component_score_groups`. For any declared group \(G\), its
intermediate score is the normalized weighted score

\[
S_G=\frac{\sum_{j\in G}\alpha_jS_j}{\sum_{j\in G}\alpha_j}.
\]

The published metric equations use \(F\), \(C\), and \(D\) for the declared field, coefficient/force, and diagnostic groups.
The group declarations must cover every overall-score component exactly once. This makes the intermediate scores executable for
dataset-specific weightings and component sets, including a diagnostic group containing only a velocity-profile metric. The
validator derives these scores from the same transformations and weights as the overall composite rather than maintaining a
second hard-coded formula.

For the four newly unified rankings, the same generic rule simplifies to `100 - blended_surface_rel_l2` for BlendedNet,
`100 - surface_pressure_rel_l2` for DrivAerNet++, and `100 * (1 - total_error)` for Rotor37 and VKI-LS59, with the result bounded
to `[0, 100]`. The composite is calculated from the underlying four, one, six, or eight declared components respectively rather
than from a rounded display value. These score caps and weights remain prototype policy until the dataset owners approve them.
Dataset-specific aggregate metrics use the unweighted arithmetic mean of the exact source metric IDs listed in
`submission-spec.json`.

Run the array-level example from the repository root with:

```bash
python3 -m reference.example_calculation
```
