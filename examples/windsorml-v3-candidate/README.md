# WindsorML schema-v3 submission package

The [participant quickstart](../../benchmark-specs/windsorml/QUICKSTART.md) provides
the complete resumable command sequence and pinned Linux container. The
per-case commands below remain available for existing pipelines.

Use the current release checkout. This workflow packages genuine model
inference for submission review. Read the
[data, split and physics contract](../../benchmark-specs/windsorml/README.md)
first. Daniel's end-to-end trial is to run this workflow with his own model and
checkpoint on one complete declared split, then open a PR targeting `dev`.

## 1. Configure the actual method

```bash
python -m pip install -r requirements.txt
cp examples/windsorml-v3-candidate/package-config.template.json /path/to/package-config.json
```

Replace every `__REPLACE_...__` value with actual method, training, compute,
checkpoint, and spatial discretization records. Use a new globally unique
submission ID. Keep the supplied release bindings unchanged. Declare every
checkpoint actually loaded; the assembler checks its local bytes against the
methodology SHA-256. Publishing checkpoint files is optional.

```bash
python scripts/assemble_windsorml_schema_v3_candidate.py \
  --config /path/to/package-config.json --list-unresolved
```

The inference attestation must be accurate: all prediction values must come
from the declared model inference, without generating them from evaluation
truth. Validation verifies artifacts and declarations; it does not prove that
the model was executed.

## 2. Predict and evaluate every required case

Read the ordered list from `benchmark-specs/windsorml/splits/<split-id>.json`.
The `full` test set has **35** available cases, not 36. Produce predictions for
every original surface point and volume cell in native source order.

Use one prediction manifest per case and domain. Each has this structure
(shown for surface; replace symbolic values with actual counts and digests):

```json
{
  "format": "windsorml-native-prediction-chunks-candidate",
  "format_version": 1,
  "artifact_role": "local_evaluator_input_not_official_submission_artifact",
  "case_id": "run_7",
  "support_id": "windsorml_surface_native_points",
  "association": "PointData",
  "total_row_count": "ACTUAL_NATIVE_POINT_COUNT",
  "field_components": {"cpavg": 1, "cfxavg": 1, "cfyavg": 1, "cfzavg": 1},
  "chunks": [{
    "chunk_index": 0, "file": "chunk-00000.npz", "sha256": "ACTUAL_FILE_SHA256",
    "row_count": "ACTUAL_CHUNK_ROWS", "raw_cell_id_start": 0,
    "raw_cell_id_stop": "EXCLUSIVE_STOP"
  }]
}
```

NPZ chunks contain `raw_cell_id` (int64) and each named scalar prediction field
(float32 or float64). `raw_cell_id` indexes
**points for surface** and **cells for volume**. Use consecutive IDs starting
at zero, contiguous non-overlapping chunks, at most 1,000,000 rows per chunk,
and complete coverage. For volume use support `windsorml_volume_native_cells`,
association `CellData`, and scalar fields `velocityxavg`, `velocityyavg`,
`velocityzavg`, `pressureavg`. Prediction manifests and NPZ files stay local;
the package retains their hashes.

Run once per required case:

```bash
python scripts/evaluate_windsorml_candidate_case.py \
  --case-id run_7 --dataset-root /path/to/windsorml \
  --surface-manifest /path/to/predictions/run_7/surface/manifest.json \
  --volume-manifest /path/to/predictions/run_7/volume/manifest.json \
  --output /path/to/evidence/run_7.json
```

The CLI automatically loads the pinned source identity and profile support.
It checks complete native coverage and derives field metrics, forces, and all
fixed and relative profiles. No manually supplied truth manifest or generated
support sidecars are needed for evaluation.

Optionally retain the split reduction:

```bash
python scripts/score_windsorml_candidate_dataset.py \
  --split-id full --case-evidence-directory /path/to/evidence \
  --output /path/to/dataset-evidence.json
```

## Package, validate and submit

Follow the [WindsorML participant quickstart](../../benchmark-specs/windsorml/QUICKSTART.md)
to evaluate the complete split and run `python scripts/windsorml.py package`.
That command checks the evaluation outputs, checkpoint identities and release
bindings and creates the result directory. Use its generated package for submission.
Keep the template release bindings unchanged.

```bash
python scripts/windsorml.py verify-package submissions/windsorml/YOUR_SUBMISSION_ID
python scripts/validate_submission.py --contributor-stage submissions/windsorml/YOUR_SUBMISSION_ID
```

Commit only the generated `submissions/windsorml/YOUR_SUBMISSION_ID/` directory
and open a pull request against `dev`. Follow [Submitting a result](../../SUBMITTING.md).
Maintainers review, approve and publish the result. Do not add approval records,
registration files or generated leaderboard feeds to a participant pull request.
