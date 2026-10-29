# DrivAerML participant guide

This guide describes the native-mesh workflow for a DrivAerML result submitted
to `fluidsbench-submission`. DrivAerML is open for submissions across all eight prescribed splits, with
full surface-and-volume or surface-only predictions.

Start with the [resumable participant quickstart](QUICKSTART.md): prepare data
and support, validate your entry, evaluate the complete split, then package and
verify the result. It uses the same native field/force evaluator and validation
checks documented below. Profile integration matches AutoCFD5 v1.1.7.
The shared [submission workflow](../../SUBMITTING.md) covers package versions,
optional artifacts and later PR review.

Use the shared [hardware and compute fields](../../METHODOLOGY.md#hardware-and-compute)
for each training stage and inference campaign, including GPU model and allocation counts.

## Before you start

The complete native-VTK evaluator is currently supported on Linux
only, including Linux HPC nodes and Linux containers or WSL. Its retained-file
protections rely on Linux descriptor-filesystem semantics and are not yet
validated on native macOS or Windows. JSON/schema and profile-package checks
may run on other platforms, but official evaluator evidence must be generated
on Linux until cross-platform support is implemented and tested.

The generic repository checks use `requirements.txt`. The exact recommended
stack for the complete VTK-based DrivAerML evidence workflow is defined in
[`requirements-drivaerml-participant.txt`](../../requirements-drivaerml-participant.txt):
Python 3.12.13, NumPy 2.2.6, and VTK 9.5.2. Runtime enforcement differs by
workflow. Velocity-assignment generation rejects a Python, NumPy, or VTK
version mismatch. The native-surface reader rejects a VTK version other than
9.5.2, and its evidence records the dependency identities available to that
path. Force replay records its Python, NumPy, and VTK versions but does not
reject a different Python patch or NumPy version. The bounded XML equal-cell
volume audit does not invoke VTK or compute cell volumes; it records its
Python/NumPy runtime for provenance, while the strict aggregate validates the
v2 schema and source identities. Use the exact recommended stack when
reproducing the complete evaluation workflow.

```bash
python3.12 --version  # must print: Python 3.12.13
python3.12 -m venv .venv-drivaerml
.venv-drivaerml/bin/pip install -r requirements-drivaerml-participant.txt
.venv-drivaerml/bin/python -c \
  'import platform,numpy,vtk; print(platform.python_version(), numpy.__version__, vtk.vtkVersion.GetVTKVersion())'
# must print: 3.12.13 2.2.6 9.5.2
```

## 1. Choose one official split

Choose exactly one JSON file in [`splits/`](splits/). Fit the model and every
learned or data-dependent preprocessing quantity using only that file's
`train_case_ids`. The `validation_case_ids` may be used only as the split
permits for model selection. The `case_ids` are public evaluation cases, but
their fields must not affect fitting, tuning, checkpoint choice, manual
selection, normalization, or other training statistics.

Record the split ID, ordered train/validation/test case lists, source dataset
revision, checkpoint hashes, and preprocessing configuration. Record the model
commit locally as well; publishing that commit through optional code metadata
is encouraged but is not required for approval or ranking.

### Choose the prediction scope before inference

Every submission must predict the complete native surface: `pMeanTrim` and
three-component `wallShearStressMeanTrim` for every native boundary polygon in
unchanged raw VTK order. Select one scope for the entire evaluated split:

- `surface_and_volume` also predicts `pMeanTrim` and `UMeanTrim` on every
  native volume cell and produces the velocity-profile diagnostics.
- `surface_only` supplies no volume prediction, no reconstructed volume VTU,
  and no velocity-profile values. Its volume-velocity, volume-pressure, and
  velocity-profile components score exactly zero with their published weights
  retained; weights are not renormalized, so its maximum overall score is 60.

Do not fill omitted components with dummy values. The evaluator records their
unavailable scope explicitly.

### Submit a reproducible method record

Include the common [methodology record](../../METHODOLOGY.md) with the outputs for your selected scope, exact parameter counts,
all loaded checkpoint-file digests, training/selection details, and measured compute. Keep it consistent with the spatial records.
Public checkpoint bytes are optional. Test splits contain 50 or 97 cases; use the selected split's count, not all 484 cases.

<details>
<summary>Required methodology detail: components, stages, parameter counting, checkpoint files, and timing</summary>

Every schema-v3 DrivAerML package must include
`methodology.format=fluidsbench-method-v1`. It must cover the two mandatory
surface outputs in [`methodology-contract.json`](methodology-contract.json),
and, in `surface_and_volume` scope, the two volume outputs as well. A model
name or broad architecture label is not sufficient. The record must describe:

- the complete architecture as one or more named components, including each
  component's specific family, role, layer/operator description, parameter
  count, and the key hyperparameters needed to distinguish it;
- the exact total number of learned scalar parameters loaded for inference and
  the subset updated by the submitter, plus every input feature and the
  component that consumes it;
- whether every required native field for the selected scope is a direct output
  or is deterministically derived from an output, and which component produces
  it; in `surface_only` scope, state explicitly that the two volume fields are
  not submitted rather than assigning them dummy outputs;
- normalization, preprocessing, and training-time sampling or downsampling;
- every training stage and the components it affected; a submitter-performed
  stage records either its gradient recipe (loss and term weights, optimizer
  groups, schedule, integer batch definition, and duration) or another fully
  described fitting procedure, one seed per stochastic run, and measured
  training compute; an upstream stage instead cites the upstream method;
- one raw-file SHA-256 entry for every checkpoint file actually read to create
  the predictions, including its component scope, byte description, role, and
  checkpoint-selection rule; and
- measured inference hardware, maximum concurrent device count, evaluated case
  count, campaign wall time excluding queue delay, aggregate device time,
  whether preprocessing and native-support mapping are included, and the
  timing scope.

Training provenance and `training_regime` answer different questions. The
regime records target-data use. A submitter may therefore perform external
pretraining for a zero-shot result, while a from-scratch checkpoint may have
been trained upstream. Describe the actual actor independently in each stage.
Use multiple components and stages for separate surface/volume models,
ensembles, adapters, or staged optimization.
For each submitter stage, the campaign wall time and aggregate device-hours
cover all declared runs in that stage and exclude scheduler queue delay; explain
parallel or heterogeneous allocations in `compute.measurement_notes`.

Use
[`methodology.example.json`](../../examples/drivaerml-v3-candidate/methodology.example.json)
as a filled structural example. Its values are illustrative, not a baseline.
The assembler derives the display field `parameter_count_millions` from
`architecture.total_parameter_count`. This is the sum of unique learned scalar
values loaded across the architecture components: count ensemble copies
separately, count shared storage once, and assign shared parameters to one
component so the component counts still sum exactly. The separate
`submitter_trainable_parameter_count` counts unique scalars updated by the
submitter. The assembler rejects broken component references, inconsistent
counts, duplicate IDs, missing required output fields, or an inference case
count that differs from the selected split. Current official test sets contain
50 or 97 cases, not all 484 public cases; copy the count from the selected split.

Calculate each checkpoint entry from the raw file the loader read, for example
with `sha256sum checkpoint.pt`. A sharded, base-plus-adapter, or directory model
therefore has one entry per loaded weight file; do not substitute the hash of a
repacked archive. These declared digests identify the exact local bytes but do
not require them to be uploaded. Public code, model weights, environment
artifacts, and full native predictions remain optional. If an optional public
model artifact is supplied, its archive digest may legitimately differ from the
digest of a checkpoint stored inside it.

The methodology record is the scientific description of the model. The
required `discretization.json` and its case records remain authoritative for
actual representation sizes, direct outputs, sampling counts, and mappings to
the scoring supports. The two disclosures must describe the same pipeline;
maintainers reject contradictions.

</details>

## 2. Predict on the actual native supports

The source dataset is `neashton/drivaerml` at immutable revision
`7a5c0948ce27be709b1116a3a190f806e7a8f79f`. Resolve every case and multipart
file from [`proposal/native-source-pin.json`](proposal/native-source-pin.json),
not from directory discovery or a presumed run-number range.

The same release now includes the authoritative native surface weights. For
each `run_N`, load `run_N/boundary_cell_area_N.npy` together with
`surface_cell_areas/cases/run_N.json`. The complete 484-case manifest is
`surface_cell_areas/manifest.json` (SHA-256
`1401c7e80bd86f3aa2d640289db9b088ce1e0825327e18eeb1ab2852de04323e`).
For example, to fetch only the run-1 support:

```bash
hf download neashton/drivaerml \
  surface_cell_areas/manifest.json \
  surface_cell_areas/cases/run_1.json \
  run_1/boundary_cell_area_1.npy \
  --repo-type dataset \
  --revision 7a5c0948ce27be709b1116a3a190f806e7a8f79f
```

For each evaluated case:

1. Read `run_N/boundary_N.vtp` without remeshing or triangulating it. Predict
   scalar `pMeanTrim` and three-component `wallShearStressMeanTrim` as native
   polygon `CellData` in zero-based raw VTK cell order. Load the corresponding
   public area payload in that identical order; do not calculate a replacement
   from a processed surface.
2. For `surface_and_volume`, byte-concatenate the pinned `.00.part`, `.01.part`, and optional `.02.part`
   files in that exact order, with no delimiter or transformation. Ten cases
   have three parts. A seekable segmented reader may present the same logical
   bytes without writing a roughly 50 GB reconstructed VTU.
3. For `surface_and_volume`, read the reconstructed unstructured grid without remeshing. Predict scalar
   `pMeanTrim` and three-component `UMeanTrim` for every native volume
   `CellData` cell in zero-based raw VTK cell order.

Surface pressure and wall shear are kinematic quantities in `m^2/s^2`.
Volume pressure is in `m^2/s^2`, velocity is in `m/s`, coordinates are in
metres, and fixed surface weights are in `m^2`.

## 3. Use bounded native-cell chunks correctly

Inference chunks are independent of multipart byte-transport parts. A
prediction chunk contains a contiguous half-open raw-cell interval and the
corresponding fields. Across a case, the intervals must exactly partition
`[0, native_cell_count)` with no gap, overlap, duplicate, omission, remapping,
or extrapolation.

The local evaluator accepts one manifest per case and support. Each
manifest identifies its count and SHA-256-bound NPZ chunks. Every NPZ contains
signed-int64 `raw_cell_id` and exactly these Float32 or Float64 arrays:

| Support                                           | Required arrays                                    |
| ------------------------------------------------- | -------------------------------------------------- |
| `surface_native_cells`                            | `pMeanTrim: [N]`, `wallShearStressMeanTrim: [N,3]` |
| `volume_native_cells` (`surface_and_volume` only) | `pMeanTrim: [N]`, `UMeanTrim: [N,3]`               |

For bounded validation, each NPZ chunk may contain at most 512 MiB across the
sum of all archive members' declared uncompressed byte sizes; this is not a
limit on the compressed archive size. Both ZIP `ZIP_STORED` and `ZIP_DEFLATED`
compression are accepted. Its ZIP central directory may be at most 64 KiB and
each NPY header at most 4,096 bytes. Use C-contiguous, non-object arrays; split
larger predictions into more contiguous raw-cell intervals.

Chunking must not change a metric. For each complete case, add the weighted
squared-error numerator, weighted truth-square denominator, weighted absolute
error, entity count, and total weight across chunks. Apply square roots and
division only after the complete case is represented. Never average
chunk-local relative-L2, MAE, or RMSE values.

The primary surface metrics use the fixed published same-order polygon areas;
equal-polygon metrics are mandatory secondary values. The primary volume
metrics weight every native cell equally; no geometric cell-volume array or
volume-weighted secondary metric is required. In `surface_only` mode the
volume metrics are absent, rather than replaced by zeros, and their fixed
score contributions are zero. The evaluator audits, but never
regenerates, the fixed surface-area inputs.

## 4. Run the FluidsBench reference evaluator locally

Run the owner-approved evaluator locally. Do not
substitute hand-calculated force conventions or independently defined profile
extraction. The evaluator integrates the participant's native surface pressure
and wall shear to produce per-case `Cd`, `Cl`, `CmPitch`, and report-only
`Clf`/`Clr`, together with their submitted metrics. It uses the approved
mappings and extraction support to produce the 16 AutoCFD5 velocity profiles
for `surface_and_volume`; a `surface_only` evaluation emits their explicit
unavailable/zero-score status without opening a volume source. It derives the
four FluidsBench continuous Cp cuts from the submitted surface pressure in
either scope when the frozen support is active. Include the evaluator-produced
coefficients, metrics, and applicable profile series in the submission package.
The AutoCFD5 name identifies the source line geometry only; this JSON belongs
to the DrivAerML FluidsBench package and is not submitted to AutoCFD. This
keeps truth and predictions on the same geometry, raw IDs, force convention,
validity masks, and reductions.

For schema v3, each case in `metrics/cases.json` carries the evaluator-produced
prediction object `force_coefficients` with exactly `cd`, `cl`, `cm_pitch`,
`clf`, and `clr`. The first three are ranked independently; `clf` and `clr` are
report-only closure evidence.

`force_mom_constref_all.csv` is the canonical force-truth table. The per-case
`run_N/force_mom_constref_N.csv` files remain supported mirrors only after the
evaluator proves that each row agrees exactly with the aggregate table. This
does not change any target value or score; it provides one hash-bound source of
truth while retaining the per-case files for convenient local processing.

<details>
<summary>Required profile support, global R2 reductions, score caps, and velocity validity rules</summary>

The package binds [`drivaerml-diagnostics-v10.json`](drivaerml-diagnostics-v10.json)
and the immutable native-v3 profile truth/support release. It contains
16 fixed velocity lines and four fixed continuous Cp cuts: upper-body centreline
(`y=0`), underbody centreline (`y=0`), sidewall (`z=0.15 m`), and front-left
wheelhouse (`y=-0.6 m`). The evaluator also produces the corresponding zero-weight
relative diagnostics. Extraction uses retained native raw-cell IDs, supported
segments and exact coordinates; participants do not regenerate supports.

The four continuous Cp cuts remain a ranked component with composite weight
0.10; the velocity profiles retain weight 0.15. The evaluator derives every cut
from native surface `pMeanTrim`. Participants therefore do not add a Cp-cut
field to the VTP or predict a second Cp representation, but they do submit the
evaluator-produced Cp-cut coordinate and prediction arrays in the normal
FluidsBench profile JSON.

The ranked profile metrics are global R2 values, not macro-averaged RMSE. Each
case and each required line or cut receives equal total weight. Within a
velocity line or Cp cut, samples use normalized trapezoidal arc-length weighting
without bridging unsupported gaps, as defined by the
[profile scoring contract](profile-scoring-v2.json). A single global weighted
truth mean is used for each profile family. This
prevents the 651-sample L1 velocity line from outweighing a 31-sample R line.
The corresponding equal-case/equal-line and equal-case/equal-cut RMSE values
remain report-only diagnostics.

The four field components use fixed bounded-error caps of 15% for surface
pressure, 20% for wall shear, 12% for volume velocity, and 15% for volume
pressure. `Cd`, `Cl`, `CmPitch`, velocity-profile, and Cp-cut R2 use bounded
quality scores `100 * clip(R2, 0, 1)`. The nine weights total 50% fields, 25%
forces, and 25% profiles.

Only the 209 discrete Cp probes are excluded. Participants do not submit probe
outputs, probe mappings, or probe-specific support, and no discrete-probe
metric is calculated. The combined v8 registry and existing 209-probe files in
the proposal and evidence directories are retained only as inactive,
non-normative research records.

For `surface_and_volume`, participants run the evaluator over their complete
native `UMeanTrim` prediction and do not create or modify velocity-profile
validity masks. The frozen evaluator will apply the owner-published mask and
containing-cell assignments and emit the JSON profile series to include in the
package. `surface_only` has no velocity-profile series. Only
`inside_morphed_solid` and `outside_released_fluid_domain` are permitted owner
exclusion reasons. An unresolved containing-cell or cell-evaluation failure is
not an automatic exclusion: it makes the required line unavailable. Excluded
or unsupported coordinates remain explicit and remove only their point and
adjacent trapezoidal edges; gaps are never bridged. The owner has approved the
published all-case support and this fail-closed behavior. The final evaluator
must bind and reproduce the identical support rather than infer a new mask.

</details>

## 5. Install and run the evaluator tools

Use the [runtime installed above](#before-you-start). Run these commands from the repository root.

<details>
<summary>Optional synthetic transport demonstration (not a real result)</summary>

Run the small two-part/three-part teaching fixture with one command:

```bash
.venv-drivaerml/bin/python examples/drivaerml-candidate-native-chunks/reference_driver.py \
  --output /tmp/drivaerml-candidate-native-chunk-demo
```

The fixture is synthetic and explicitly ineligible. It proves ordered two- and
three-part byte concatenation, raw-ID chunk coverage, additive reduction, and
full-versus-chunked invariance for the two surface and two volume fields. Its
dummy schema-v3 package mirrors the current participant shape: area-weighted
and equal-polygon surface metrics, equal-cell volume metrics, coherent force
coefficients, all four Cp cuts, all sixteen AutoCFD5 velocity profiles, and all
32 report-only metrics. Its small transport payloads and support
tables are JSON, not VTK, and its force/profile values are synthetic; it does
not claim real DrivAerML extraction, force integration, reconstruction, model
results, or official scoring support.

The command performs the fixture's schema-v3 checks itself. Its deliberately
unregistered `synthetic-drivaerml-shaped` namespace is not accepted by the
repository's official-dataset semantic validator. Use your model's complete
predictions for a submission.

</details>

For a real `surface_and_volume` case, first create or obtain surface
and volume prediction manifests. For `surface_only`, create only the surface
manifest and pass `--prediction-scope surface_only`; do not provide a volume
manifest, reconstructed volume file, or `--multipart`. The zero-field generator
is a transport test only, never a model or published baseline:

```bash
.venv-drivaerml/bin/python scripts/create_drivaerml_dummy_predictions.py \
  --case-id run_1 \
  --surface-count 8828095 \
  --volume-count 147449586 \
  --max-chunk-rows 1000000 \
  --output-root /tmp/drivaerml-run1-zero
```

The `surface_and_volume` real-case interface consumes the actual native surface
and reconstructed volume plus prediction manifests. Volume errors are
accumulated with one equal weight per raw native cell:

```bash
.venv-drivaerml/bin/python scripts/evaluate_drivaerml_candidate_case.py \
  --case-id run_1 \
  --native-source-pin benchmark-specs/drivaerml/proposal/native-source-pin.json \
  --dataset-root /path/to/drivaerml \
  --monolithic-vtu /path/to/drivaerml/run_1/volume_1.vtu \
  --surface-area-npy /path/to/run_1/boundary_cell_area_1.npy \
  --surface-prediction-manifest /path/to/surface/manifest.json \
  --volume-prediction-manifest /path/to/volume/manifest.json \
  --output /path/to/work/core/run_1.json \
  --profile-output /path/to/work/profiles/run_1.json
```

Use `--multipart` instead of `--monolithic-vtu` when the pinned part files are
present below `--dataset-root`. The evaluator verifies raw native-cell coverage
and accumulates equal-cell sufficient statistics directly; participants do not
supply a geometric volume-weight file.

For surface-only evaluation, use the same command with the volume arguments
removed and add `--prediction-scope surface_only`. This path validates and
scores only the native surface; it deliberately does not open a volume VTK or
prediction stream.

Run the case command for **every** test case in the selected split, with core
and profile evidence named `run_N.json` in their respective directories. The
profile stage reads the same complete prediction manifests, verifies chunk
hashes and native coverage, and extracts both fixed and relative profiles.
It downloads SHA-256-verified immutable native-v3 truth/support chunks into
`~/.cache/fluidsbench/drivaerml/`; for offline use set
`FLUIDSBENCH_DRIVAERML_TRUTH_ROOT` to the retained native-v3 directory containing
`index.json` and `chunks/`. The package contains predictions, not copied truth.

Then reduce the complete split in one command:

```bash
.venv-drivaerml/bin/python scripts/score_drivaerml_candidate_dataset.py \
  --split-id full \
  --core-evidence-dir /path/to/work/core \
  --diagnostic-evidence-dir /path/to/work/profiles \
  --output /path/to/work/dataset-evidence.json \
  --regional-diagnostics-output /path/to/work/regional-diagnostics.json \
  --schema-v3-case-metrics-output /path/to/work/metrics/cases.json \
  --schema-v3-profile-output-dir /path/to/work/package-profiles \
  --submission-id <submission-id>
```

Use your selected split ID consistently. `full` has 50 test cases; the OOD
split has 97. The reducer reads the retained canonical force CSV and published
release bindings automatically. It computes force and profile global R²,
all field/diagnostic metrics, component scores, and the bounded overall score.
It refuses missing, extra, duplicated, or reordered evidence where the contract
requires order. Do not calculate or insert missing scores by hand.

Full-field packages contain 40 fixed/relative series per case. Surface-only
packages contain eight Cp series and no velocity series. Two relative Cp cuts
are explicit shared-support aliases. Relative families remain report-only at
zero weight; only fixed families contribute to the composite.

## 6. Package and submit

Use the [participant quickstart](QUICKSTART.md) to run `scripts/drivaerml.py package`
and `verify-package` against the complete evaluation output. This creates the
release-bound result package and validates it before submission.

```bash
python scripts/validate_submission.py --contributor-stage submissions/drivaerml/YOUR_SUBMISSION_ID
```

Open a pull request against `dev` containing only that new result directory.
Follow the [submission workflow](../../SUBMITTING.md) and
[result version rules](../../docs/RESULT_FORMAT.md#result-versions).
Maintainers review and approve the package before leaderboard publication.
Code, model weights and full prediction artifacts remain optional.

## Relative diagnostics

The evaluator derives fixed and relative profiles from the same native predictions.
Relative velocity and Cp families remain report-only with zero composite weight.
The official package preserves their frozen coordinates, support identities and
native gaps. The owner removed sensitivity studies as an opening requirement;
no such study is claimed. See the [owner decision](OWNER_OPENING_DECISION.md)
and [profile contract](drivaerml-relative-diagnostics-v3.json).
