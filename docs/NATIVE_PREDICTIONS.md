# AhmedML and DrivAerML prediction manifests

Use this example after running your own model. It writes local evaluator inputs;
the dataset's `package` command later creates the result directory for your PR.
Run Python from the repository root in your dataset's pinned environment.

Each case needs a surface manifest and, for `surface_and_volume`, a volume
manifest. All arrays must contain your model's denormalized predictions in the
unchanged native cell order. The field names and component counts are:

| Dataset | Domain | Prediction `support_id` | `field_components` |
| --- | --- | --- | --- |
| AhmedML | Surface | `ahmedml_surface_native_cells` | `{"pMean": 1, "wallShearStressMean": 3}` |
| AhmedML | Volume | `ahmedml_volume_native_cells` | `{"pMean": 1, "UMean": 3}` |
| DrivAerML | Surface | `surface_native_cells` | `{"pMeanTrim": 1, "wallShearStressMeanTrim": 3}` |
| DrivAerML | Volume | `volume_native_cells` | `{"pMeanTrim": 1, "UMeanTrim": 3}` |

Scalar arrays have shape `(N,)`; vector arrays have shape `(N, 3)`. Use float32
or float64. The writer adds signed-int64 `raw_cell_id` values and splits arrays
into at most 1,000,000 rows per NPZ. It preserves the supplied order and calculates
each file's SHA-256. Large `.npy` prediction arrays can be opened with
`np.load(path, mmap_mode="r", allow_pickle=False)`.

## Write the manifests and NPZ chunks

```python
import hashlib
import json
from pathlib import Path

import numpy as np

from reference.drivaerml.prediction_chunks import support_field_components


def write_prediction(dataset, case_id, domain, fields, destination):
    """Write one case/domain from actual native-order model predictions."""
    if dataset not in {"ahmedml", "drivaerml"} or domain not in {"surface", "volume"}:
        raise ValueError("choose ahmedml/drivaerml and surface/volume")
    prefix = "ahmedml_" if dataset == "ahmedml" else ""
    support_id = f"{prefix}{domain}_native_cells"
    components = dict(support_field_components(support_id))
    if set(fields) != set(components):
        raise ValueError(f"required fields: {components}")
    count = len(next(iter(fields.values())))
    if count == 0:
        raise ValueError("predictions must cover the complete native support")
    for name, width in components.items():
        array = fields[name]
        shape = (count,) if width == 1 else (count, width)
        if array.shape != shape or array.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{name}: expected float32/float64 with shape {shape}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)  # use a fresh output directory
    (destination / "chunks").mkdir()
    chunks = []
    for index, start in enumerate(range(0, count, 1_000_000)):
        stop = min(start + 1_000_000, count)
        arrays = {name: np.ascontiguousarray(value[start:stop]) for name, value in fields.items()}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("predictions must be finite; correct the inputs and use a fresh directory")
        relative = f"chunks/chunk-{index:05d}.npz"
        path = destination / relative
        np.savez(path, raw_cell_id=np.arange(start, stop, dtype=np.int64), **arrays)
        chunks.append({
            "chunk_index": index, "file": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": stop - start, "raw_cell_id_start": start, "raw_cell_id_stop": stop,
        })
    manifest = {
        "format": f"{dataset}-native-prediction-chunks-candidate", "format_version": 1,
        "artifact_role": "local_evaluator_input_not_official_submission_artifact",
        "case_id": case_id, "support_id": support_id, "association": "CellData",
        "total_row_count": count, "field_components": components, "chunks": chunks,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
```

For **AhmedML**, after defining the function above, pass the arrays returned by
your inference pipeline (the variable names below represent those actual arrays):

```python
write_prediction("ahmedml", case_id, "surface", {
    "pMean": surface_pressure,
    "wallShearStressMean": surface_wall_shear,
}, f"my-entry/cases/{case_id}/surface")
write_prediction("ahmedml", case_id, "volume", {
    "pMean": volume_pressure,
    "UMean": volume_velocity,
}, f"my-entry/cases/{case_id}/volume")
```

For **DrivAerML**, use the dataset's exact field names:

```python
write_prediction("drivaerml", case_id, "surface", {
    "pMeanTrim": surface_pressure,
    "wallShearStressMeanTrim": surface_wall_shear,
}, f"my-entry/cases/{case_id}/surface")
# Omit this call for a surface_only entry.
write_prediction("drivaerml", case_id, "volume", {
    "pMeanTrim": volume_pressure,
    "UMeanTrim": volume_velocity,
}, f"my-entry/cases/{case_id}/volume")
```

Repeat for every `case_id` in the selected split, using each case's actual native
counts. Any mapping from another representation must happen before writing
these arrays and must be described in the spatial records. The evaluator checks
the predictions against the pinned source counts; the writer does not perform
mapping or establish that inference was run.

The generated manifest is complete: its paths are relative to `manifest.json`,
its chunk intervals cover `[0, total_row_count)` exactly, and its hashes identify
the actual NPZ bytes. Keep the literal format identifiers shown above. Then run
your dataset's `validate-entry` and `evaluate-entry` commands from its quickstart.

## Record your actual spatial representation

Start with the dataset-specific single-case examples:

- [AhmedML run_4](../examples/ahmedml-v3-candidate/discretization-case.example.json)
- [DrivAerML run_11](../examples/drivaerml-v3-candidate/discretization-case.example.json)

Their native counts and scoring-support IDs come from the published support.
They illustrate direct native-cell inference, including evaluator-derived forces
and profiles where required. Replace the submission ID, split, input/output IDs
and representation counts to describe your actual model. For another case, use
that case's support counts. For sparse or alternative outputs, record their
actual counts and the mapping to complete scoring support.

Keep input/output IDs consistent with `spatial_discretization.inference` in your
package configuration. The summary and per-case records must describe the same
pipeline. In DrivAerML `surface_only` scope, omit the volume input, output and
mapping, and set the summary's volume representations to `{"used": false}`.

Write one compact JSON object per line in `my-entry/discretization-cases.jsonl`,
in the exact split order. The [case schema](../schemas/v3/discretization-case.schema.json)
defines these records; the [summary example](../examples/v3-template/discretization.json)
shows the separate training and inference objects needed by the package config.
