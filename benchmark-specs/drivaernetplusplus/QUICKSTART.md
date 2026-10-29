# DrivAerNet++ submission guide

Use the current release checkout. The participant command checks your copy of the
public data, downloads the prepared scoring support, evaluates every native
surface point of all 1,154 `official_test` designs with recovery, and produces a
validated result directory. The dataset owner's scientific
approval and latest confirmations are recorded in [OWNER_APPROVAL.md](OWNER_APPROVAL.md).
See the [dataset index](../README.md) for availability and [Submitting a result](../../SUBMITTING.md) for validation and review.

The benchmark is **pressure only**: predict the kinematic gauge pressure `p`
(m²/s², freestream reference zero, so `Cp = p / 450`) at every point of each
`SurfacePressureVTK` file. The primary metric is the area-weighted relative L2
error, computed per design and averaged over designs; `overall_score =
clip(100 − surface_pressure_rel_l2, 0, 100)`.

## Install

On Linux, use Python 3.12.13:

```bash
python3.12 -m venv .venv-dnpp
.venv-dnpp/bin/pip install -r requirements-drivaernetplusplus-participant.txt
source .venv-dnpp/bin/activate
python scripts/drivaernetplusplus.py --help
```

This pins NumPy 2.2.6 (the evaluator's only numerical dependency),
jsonschema 4.26.0 and VTK 9.5.2. The prediction example below uses this installed
VTK reader. A container is described at the end.

## 1. Get and verify the public data

Download the 15 family archives of `SurfacePressureVTK` from Harvard Dataverse
[doi:10.7910/DVN/K7PWNJ](https://doi.org/10.7910/DVN/K7PWNJ) version 1.0
(Globus-backed; about 210 GB) and extract them into one directory. The test
designs span every family, so all archives are needed. Files named
`<case_id>.vtk` or `DrivAer_<case_id>.vtk` are both accepted. Then check that
your copy is byte-identical to the pinned release:

```bash
python scripts/drivaernetplusplus.py verify-data --dataset-root /data/dnpp/SurfacePressureVTK
```

Each of the 1,154 test files is checked against the SHA-256 in
`public-source-identity/`. The data are CC BY-NC 4.0.

## 2. Write one prediction per test design

Copy `examples/drivaernetplusplus-v3-candidate/entry.template.json` to
`my-entry/entry.json` and choose a submission ID (lowercase letters, digits and
hyphens). The case list is fixed: the 1,154 IDs of `splits/official_test.json`.

```text
my-entry/
  entry.json
  cases/
    E_S_WWC_WM_008/prediction.npz
    ...
```

Each `prediction.npz` holds exactly one array, `p`: float32 or float64, shape
`(N,)`, finite, **in the native PointData order** of that design's VTK file.
There must be one value for every native point; nothing is interpolated or
filled in by the evaluator. Any resampling your model needs to reach the native
points is allowed, but it is part of your method, carries its own error, and
must be declared in the discretization record (step 5). The example below reads
the native points with VTK; supply your own loaded `model` and run it for each
`case_id` in the test list:

```python
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkPolyDataReader

source = Path("/data/dnpp/SurfacePressureVTK") / f"{case_id}.vtk"
if not source.is_file():
    source = source.with_name(f"DrivAer_{case_id}.vtk")
reader = vtkPolyDataReader()
reader.SetFileName(str(source))
reader.Update()
points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
p = np.asarray(model.predict(points), dtype=np.float32)  # one value per native point
output = Path("my-entry/cases") / case_id / "prediction.npz"
output.parent.mkdir(parents=True, exist_ok=True)
np.savez(output, p=p)
```

Check the entry before evaluating:

```bash
python scripts/drivaernetplusplus.py validate-entry my-entry
```

## 3. Download the prepared support

The reference pressure, dual-area weights and profile points are prepared once
by FluidsBench from the public files and published as 24 hash-pinned archives
(about 5.5 GB). Do not regenerate them.

```bash
python scripts/drivaernetplusplus.py fetch-support --destination support/drivaernetplusplus --dry-run   # plan, no network
python scripts/drivaernetplusplus.py fetch-support --destination support/drivaernetplusplus
```

These commands cache archives in `support/drivaernetplusplus`, which is also mounted
by the container commands below. Always pass the same directory to evaluation
with `--support-root`. Without an explicit directory, the default is
`~/.cache/fluidsbench/drivaernetplusplus/<manifest-sha>`
(or `FLUIDSBENCH_DRIVAERNETPLUSPLUS_SUPPORT_ROOT`).
They are verified on download and every case is re-verified against the records
in Git when it is scored. `--mirror DIR` installs archives from a local copy.

## 4. Evaluate, with recovery

```bash
python scripts/drivaernetplusplus.py evaluate-entry my-entry --output work/my-result \
  --support-root support/drivaernetplusplus --resume
```

Every case is scored once and published atomically with a checkpoint bound to
the evaluator, the support, and the exact bytes of its prediction. If the run
stops, rerun the same command: verified cases are reused. A changed prediction
or evaluator is refused; use a new output directory. The complete split is
reduced into `work/my-result/evaluation/`. Inspect a case with:

```bash
python scripts/drivaernetplusplus.py report --evaluation-root work/my-result/evaluation \
  --case-id E_S_WWC_WM_008 --output work/E_S_WWC_WM_008.html
```

Three report-only Cp profiles (upper-body and underbody centrelines, rear base)
are read by the evaluator from your predictions at frozen native points; see
`profiles/STATION_DEFINITIONS.md`. You do not submit profiles.

<a id="5-package-and-open-a-pull-request-to-dev"></a>

## 5. Package and submit

```bash
python scripts/drivaernetplusplus.py discretization-template --output my-entry/discretization-cases.jsonl
cp examples/drivaernetplusplus-v3-candidate/package-config.template.json my-entry/package-config.json
# fill in every __REPLACE_ value; keep release_bindings unchanged
python scripts/drivaernetplusplus.py package --evaluation-root work/my-result/evaluation \
  --config my-entry/package-config.json --discretization-cases my-entry/discretization-cases.jsonl \
  --checkpoint YOUR_CHECKPOINT_ID=/path/to/model.pt \
  --output submissions/drivaernetplusplus/YOUR_SUBMISSION_ID
python scripts/drivaernetplusplus.py verify-package submissions/drivaernetplusplus/YOUR_SUBMISSION_ID
```

The discretization template describes direct native-point inference (identity
mapping); edit its inputs and outputs if your model used another representation.
`package` refuses unresolved tokens, a config that does not match the evaluated
entry, checkpoint bytes that differ from the methodology SHA-256, and any
evaluation file changed after reduction. The pull request must add exactly that
one directory and target `main`.

<a id="6-review-and-display-on-the-dev-leaderboard"></a>

## 6. Review and publication

Maintainers review the package and handle [approval and publication](../../docs/MAINTAINERS.md).
Your pull request contains only the generated result directory. Merging it does not itself publish a row.
Approved results are published on the [FluidsBench leaderboard](https://fluidsbench.org/), with scores and three native Cp profile comparisons.

## Container

Run from the repository root. The persistent `support/drivaernetplusplus` directory
is shared by download and evaluation, including across interrupted runs. Files are
written as your user so the resulting package can be edited and committed on the host.

```bash
mkdir -p support/drivaernetplusplus work
docker build -f Dockerfile.drivaernetplusplus -t fluidsbench-drivaernetplusplus .
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v "$PWD/support/drivaernetplusplus:/support" -w /work \
  fluidsbench-drivaernetplusplus fetch-support --destination /support
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v "$PWD/support/drivaernetplusplus:/support" -w /work \
  fluidsbench-drivaernetplusplus evaluate-entry my-entry \
  --output work/my-result --support-root /support --resume
```

Use the same mounts and working directory for `validate-entry`, `report`,
`discretization-template`, `package` and `verify-package`. Mount any external data
or model checkpoint directory at the path used in your command, for example
`-v /data:/data:ro` for `verify-data --dataset-root /data/dnpp/SurfacePressureVTK`.
