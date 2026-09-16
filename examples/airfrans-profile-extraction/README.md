# AirfRANS profile extraction example

This directory contains the reference extractor for FluidsBench 1D velocity
profiles from the two-dimensional AirfRANS CFD meshes. The four lines start on
the geometric extrados and follow the outward unit surface normal. AirfRANS
stores inward-pointing normals from version 0.1.4 onward, so the extractor
explicitly negates and normalizes them before sampling.

## Environment

Install the exact runtime in [`requirements.txt`](requirements.txt). The script
checks the installed distribution versions, the known stale AirfRANS module
version, the installed AirfRANS source hash, and the requirements-file hash
against `benchmark-specs/airfrans/velocity-profiles-v1.json` before extracting.

The pinned versions are:

- `airfrans == 0.1.5.1`
- `numpy == 2.5.2`
- `pyvista == 0.48.4`
- `vtk == 9.6.2`

PyVista 0.48.4 declares `vtk < 9.7.0`; VTK 9.7.0 is therefore not a compatible
runtime.

## Regenerate the reference fixture

Run the extractor against the official preprocessed AirfRANS `Dataset` folder:

```bash
python3 extract.py --dataset-root /path/to/AirfRANS/Dataset
```

The committed fixture uses an official evaluation case and records hashes for
both source meshes, all four sampling origins and outward normals, VTK validity
counts, package provenance, and a perfect-copy prediction check. Every station
must contain exactly 1,001 valid samples spanning 0.0 to 0.1 m.

## Extract model predictions

Researchers do not need to write predicted unstructured meshes. Supply a NumPy
array with shape `(number_of_native_internal_points, 2)` in the unchanged
AirfRANS internal-point order:

```bash
python3 extract.py \
  --dataset-root /path/to/AirfRANS/Dataset \
  --case-name airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548 \
  --velocity-predictions predicted_velocity.npy \
  --output predicted_profiles.json
```

An `.npz` file is also accepted; its default array key is `velocity` and can be
changed with `--prediction-key`. A PyTorch result can be passed without any VTK
serialization by converting the tensor to a NumPy array:

```python
predicted_velocity = output.detach().cpu().numpy()
```

The reusable `extract_profiles(simulation, spec, reference=False,
velocity=predicted_velocity)` function performs the same in-memory operation for
callers that load `extract.py` as a module. The reference path uses
`reference=True`; the prediction path explicitly uses `reference=False` after
installing the supplied Cartesian velocity array on `Simulation.velocity`.

## Calculate the profile score

The leaderboard retains one overall velocity-profile score. It does not flatten
the Cartesian components or chord stations together. Instead, it calculates one
R2 across the complete selected split for each of the four-station by
two-component combinations, bounds each group R2 to `[0, 1]`, and averages the
eight values equally. This gives streamwise and transverse velocity equal total
weight and prevents the scale of one component from masking another.

Ground truth and predictions may each be supplied as one file or as multiple
profile chunks:

```bash
python3 score.py \
  --ground-truth ground-truth-chunk-*.json \
  --predictions prediction-chunk-*.json \
  --split-id full \
  --output profile-score.json
```

The output contains `velocity_profile_r2` and the eight raw and bounded group
R2 diagnostics. Case coverage and coordinates must match exactly. By default,
the scorer rejects anything except complete coverage of the selected official
split. `--allow-partial` is available only for explicitly labelled calibration
checks such as a single-case extractor fixture; its result is not a leaderboard
score.
