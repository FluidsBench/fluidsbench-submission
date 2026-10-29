# AhmedML prepared support distribution

The `ahmedml-derived-cache-v1-dev` GitHub prerelease distributes an exact copy of
the existing 316-case evaluator cache. It is development support, not official
submission activation or approval. The public CFD dataset remains canonical.

Each ZIP contains the original `case-support.json`, `surface_area_vector_m2.npy`,
`volume_cell_volume_m3.npy`, `volume_region_code.npy`, and
`profile_mapping_and_truth.npz`. Both the archive transport and every extracted
file are checksum-bound. Archives are installed one case at a time under a
per-case lock; partial downloads are never used. Completed cases can be reused
across splits. A corrupt cache is replaced only after a verified replacement is
ready; its old directory is retained as `.run_N-replaced-<id>` for inspection.

## Contributor commands

```bash
python scripts/fetch_ahmedml_support.py --split full --dry-run
python scripts/fetch_ahmedml_support.py --split full
python scripts/fetch_ahmedml_support.py --split full --offline
```

Use `--destination /scratch/ahmedml-support` for another cache root. Evaluation
accepts `--support-root` or `FLUIDSBENCH_AHMEDML_SUPPORT_ROOT`; otherwise it uses
`${XDG_CACHE_HOME:-~/.cache}/fluidsbench/ahmedml/<scientific-manifest-sha256>`.
An explicit evaluator `--case-support .../run_N/case-support.json` remains an
offline-compatible override. `--offline` prevents support network access.

`--case-id` may be repeated; `--all` selects the 316-case union. Full, Medium,
Scarce and Super-scarce reuse the same 50 cases. The downloader reports the
maximum compressed transfer and installed size for the selection. Source CFD
files, predictions and checkpoints require additional storage.

## Maintainer preparation and release verification

Keep the cache builder, source revision, scientific manifest, case manifests,
payload bytes and metric definitions unchanged when updating transport tooling.
Use a new versioned release and update the distribution binding if transport
assets ever change; do not overwrite a published pinned asset.

```bash
python scripts/prepare_ahmedml_support_distribution.py \
  --cache-root /path/to/verified-cache \
  --output /path/to/new-release-directory
```

This command verifies the cache against the Git scientific manifests, preserves
the original files, uses fixed ZIP member timestamps and permissions, and emits
316 archives, `support-distribution-v1.json` and `SHA256SUMS`. The source cache
verification receipt is deliberately not bundled: its original machine paths
are not required by participants. The output directory must be new.

The current archives total 10,962,124,779 bytes; installed files total
41,454,139,431 bytes including case manifests. Upload to a draft prerelease,
check the uploaded asset sizes and SHA-256 digests, then publish the prerelease.
Commit the identical distribution index under `benchmark-specs/ahmedml/`.

Run the complete public-download qualification on Linux:

```bash
python scripts/verify_ahmedml_support_distribution.py \
  --output verification/ahmedml-public-support.json
```

It starts from an empty directory for every case, uses public unauthenticated
downloads, and runs the existing deep cache verifier. It processes all 316
cases and 1,264 payloads with less than 1 GB of cache staging space, rather than
requiring all 41 GB to coexist on the runner. It does not download CFD meshes
or run model inference. Archive data is removed only from its own temporary
directories. The verification receipt lists every checked case.

`Verify AhmedML public support` runs this check for PRs changing the distribution
index and can also be dispatched manually once available on the default branch.
It also follows the contributor commands in a separate empty cache for the Full
split, verifies offline reuse, and deeply verifies all 50 installed cases.
Routine submission CI tests the installer using small fixtures; it does not
download the entire cache. Existing AhmedML intake tests exercise a 50-case,
350-series synthetic package without claiming that it is model inference.
