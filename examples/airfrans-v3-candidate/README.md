# AirfRANS submission package

AirfRANS is open for Full, Scarce and AoA extrapolation. Start with the
[participant quickstart](../../benchmark-specs/airfrans/QUICKSTART.md) for verified
support downloads, resumable evaluation and validated packaging.

Copy the [participant configuration template](../airfrans-participant/package-config.template.json) outside the repository and fill in your model,
training, checkpoint and spatial-discretization information. Generate the package
with `python scripts/airfrans.py package`, then check it with
`python scripts/airfrans.py verify-package` and follow
[Submitting a result](../../SUBMITTING.md).

`pack_native_predictions.py` illustrates how to write the required per-case NPZ
files. Adapt its prediction loader to your model. Preserve every native point and
its order; the workflow checks coordinates and coverage before evaluation.
