# DrivAerNet++ sample geometry

One real released surface file, used to validate the DrivAerNet++ publisher and
profile-extraction rules against genuine geometry without a full Dataverse/Globus download.

The geometry itself is **not stored in git**. Upstream data is CC BY-NC 4.0 while this
repository is Apache 2.0, and bulk binaries would stay in the repository history permanently,
so the file is obtained from the pinned public release and verified against its checksum.
`*.vtk` files in this directory are ignored by git.

## Licence and attribution — read before reuse

| Item | Value |
|---|---|
| File | `DrivAer_E_S_WW_WM_075.vtk` |
| Source release | Harvard Dataverse `doi:10.7910/DVN/K7PWNJ` version 1.0, `SurfacePressureVTK/` |
| SHA-256 | `2fca6566b41ae63cc8d4450f728629f3f8cf6b521038bd50ccf12bd2f6c57e58` |
| Size | 26,013,034 bytes (legacy VTK BINARY PolyData) |
| **Licence** | **CC BY-NC 4.0 — non-commercial use only** |
| Attribution | Elrefaie et al., *DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks* — https://arxiv.org/abs/2406.09624 |

> **The file does not carry the repository's Apache-2.0 terms.** It remains CC BY-NC 4.0.
> Commercial use — including training models for commercial tools — is not permitted. Anyone
> redistributing or building on it must observe the upstream licence, not the repository licence.

## Contents

| File | Purpose |
|---|---|
| `sample_split.json` | One-case split index so the publisher can be run directly on the sample. |
| `DrivAer_E_S_WW_WM_075.vtk` *(not in git)* | Estateback, smooth underbody, WW wheels. 481,363 points / 442,114 cells; single PointData array `p` (kinematic gauge pressure, m²/s²). |

## Obtaining the file

Download `SurfacePressureVTK/DrivAer_E_S_WW_WM_075.vtk` from the pinned release into this
directory, then confirm the checksum before use:

```bash
sha256sum benchmark-specs/drivaernetplusplus/samples/DrivAer_E_S_WW_WM_075.vtk
# 2fca6566b41ae63cc8d4450f728629f3f8cf6b521038bd50ccf12bd2f6c57e58
```

## Reproducing the validation

The released files are legacy VTK **BINARY**, which the publisher reads through pyvista. That is
not part of the repository's `requirements.txt`, so install it into the evaluation environment
first:

```bash
python -m pip install pyvista
```

```bash
python benchmark-specs/drivaernetplusplus/tools/publish_scoring_support.py \
  --vtk-dir benchmark-specs/drivaernetplusplus/samples \
  --split benchmark-specs/drivaernetplusplus/samples/sample_split.json \
  --out /tmp/dnpp-sample-release
```

Expected output:

```
materialized E_S_WW_WM_075: 481363 points
SELF-CHECK PASS: 1 case(s), identity/perturbation/chunk checks green
```

The measurements derived from this file — pressure convention, axis orientation, winding
inconsistency, connected-component structure, and dual-area weight statistics — are recorded
in [`../REAL_DATA_VALIDATION.md`](../REAL_DATA_VALIDATION.md).

## Scope

This is a **single geometry for validation only**. The full 1,154-case evaluation split is
pinned by DOI and per-file checksum and is never committed.
