# HiLiftAeroML public-source identity freeze

This workflow closes the public-source identity gate without treating the
current extracted Lustre files as the authority. It does not submit Slurm jobs,
download archives, or publish a benchmark.

## Authoritative activation tier

The evaluator must bind each surface and volume source to all of the following:

1. public dataset `nvidia/HiLiftAeroML` at immutable revision
   `1c266d3869bc2968ff97d2107c9c3919be03ed32`;
2. exact repository-relative `.vtu.tgz` path, Hugging Face LFS SHA-256, and
   compressed byte size;
3. exactly one regular VTU member with the exact case-local basename and
   declared uncompressed byte size; and
4. streaming extraction from that verified object, rejecting links,
   directories, path components, extra members, truncation, or size drift.

The archive hash and strict extraction rule bind the extracted bytes. A
separate SHA-256 of the extracted VTU is useful audit evidence, but is not
required to activate this tier. A pre-existing extracted file is not assumed
equivalent merely because its filename and size agree: either re-extract it
from the verified archive, or compare it with an optional member hash produced
while streaming that archive.

## Read-only audit result (2026-09-01)

The fourteen split labels resolve to eight unique case sets and a union of
exactly 1,355 cases.

- The published `data_quality/volume_zero_fill_raw_point_ids_all1800_manifest.json`
  already pins every one of the 1,800 public volume archives at the revision
  above. It records each LFS SHA-256, archive size, member name, and member
  size. Selecting the 1,355 evaluation cases is metadata-only. Their compressed
  volume archives total exactly `32,414,584,056,072` bytes.
- The immutable Hugging Face tree API returns the same volume identities and
  the corresponding surface archive LFS SHA-256/size records. Fetching the tree
  metadata is roughly twenty small API pages and reads no archive payload.
- The selected 1,355 compressed surface archives total exactly
  `17,234,771,549,735` bytes. Together with the selected volume archives, the
  evaluator can retrieve `49,649,355,605,807` exact compressed bytes without
  relying on the current extracted Lustre tree.
- The local refined root contains all 1,355 expected surface VTUs and all 1,355
  expected volume VTUs. Their sizes agree with the frozen support evidence and
  published volume manifest with zero mismatches.
- Only the ten LHC180 cases retain both `.vtu.tgz` files locally: ten surface
  archives (`131,418,712,825` bytes) and ten volume archives
  (`246,647,822,249` bytes). The other 1,345 surface and 1,345 volume archives
  are not local.
- The volume velocity-stencil receipt set covers all 1,355 evaluation cases
  with one full source-VTU SHA-256 each (all 1,355 hashes are unique). The
  chain contains 1,319 newly generated records and the exact 36 reused,
  separately audited AoA-4 held-out records. A read-only audit rehashed
  8,368,277 bytes of record JSON and 2,992,533 bytes of audit JSON, stat-bound
  464,493,782 bytes of NPZ payloads, and found zero receipt/record/audit/source
  mismatches. These are full local volume-file identities produced by the
  original stencil generator; every receipt says the finalizer did not reread
  the full source file. They therefore avoid a new 122.318-TB volume hash pass,
  while remaining explicitly distinct from verified-public-archive
  equivalence.

The raw extracted-file footprint is:

| Domain | Bytes | Decimal TB | TiB |
| --- | ---: | ---: | ---: |
| Surface | 29,592,551,727,473 | 29.593 | 26.914 |
| Volume | 122,318,382,366,726 | 122.318 | 111.248 |
| Total | 151,910,934,094,199 | 151.911 | 138.162 |

Because the volume hashes already exist, the only missing optional local
content hashes are the 1,355 surfaces: exactly `29,592,551,727,473` bytes
(29.593 TB / 26.914 TiB). The surface-only per-case read is
19,346,311,960 bytes minimum, 21,706,063,174 bytes median,
23,032,184,114 bytes at p95, and 23,572,978,208 bytes maximum. Aggregate
read-time floors are 32.88, 16.44, 10.96, and 8.22 hours at sustained aggregate
rates of 250, 500, 750, and 1,000 MB/s, respectively. Queueing and metadata
overhead are additional.

## Deterministic metadata freeze

The first command retrieves only immutable repository-tree metadata. The
second cross-checks all public volume records against the already published
1,800-case manifest, obtains declared member sizes from the exact all-split
support inventory, and writes the 1,355-case binding. Existing outputs are
never overwritten with different bytes.

```bash
python scripts/pin_hiliftaeroml_public_sources.py fetch-hf-snapshot \
  --output benchmark-specs/hiliftaeroml/public-source-identity/hf-archive-metadata-rev-1c266d3869bc2968ff97d2107c9c3919be03ed32.json

python scripts/pin_hiliftaeroml_public_sources.py build \
  --volume-manifest ../data_quality/volume_zero_fill_raw_point_ids_all1800_manifest.json \
  --hf-snapshot benchmark-specs/hiliftaeroml/public-source-identity/hf-archive-metadata-rev-1c266d3869bc2968ff97d2107c9c3919be03ed32.json \
  --volume-stencil-receipt-dir /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nashton/neil-highlifttraining/physicsnemo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/eval_runs/hilift_native_prerequisites_table5_v1_20260810/receipts/volume_velocity_profile_stencils \
  --output benchmark-specs/hiliftaeroml/public-source-identity/hiliftaeroml-public-source-identity-v1.json

python scripts/pin_hiliftaeroml_public_sources.py verify \
  --inventory benchmark-specs/hiliftaeroml/public-source-identity/hiliftaeroml-public-source-identity-v1.json \
  --hf-snapshot benchmark-specs/hiliftaeroml/public-source-identity/hf-archive-metadata-rev-1c266d3869bc2968ff97d2107c9c3919be03ed32.json
```

The JSON Schema files are
`schemas/hiliftaeroml/public-source-identity-v1.schema.json` and
`schemas/hiliftaeroml/hf-archive-metadata-snapshot-v1.schema.json`.

The completed metadata snapshot is SHA-256
`fb2d3881ac7b2ddea08f12fe4f202c5ce483436ece3065ee56c46f02777b0a05`.
The completed 1,355-case public-source inventory is SHA-256
`44f62b90d58ab8e070c3df2a37c98435c576519222b1853a511e79b2124dc881`.
It has status `complete_archive_member_binding`, and an independent second
build reproduced byte-identical output. Both documents pass their Draft 2020-12
JSON Schemas and the CLI's cross-document verification.

## Optional storage-heavy local attestation

`hash-extracted-members.sbatch` is deliberately fail closed. It requires a
literal approval sentinel, a pinned inventory digest, and explicit paths. It
uses CPU nodes only, one CPU and 2 GiB per task, hashes one case per task, and
caps the array at two concurrent tasks. It requests `--domain surface`, because
the 1,355 receipt-bound volume hashes already exist. The cluster permits at most 1,024 array
tasks per submitted array, so the plan uses `0-1023%2` followed with `afterok`
by `1024-1354%2`; the total cap remains two.

Do not submit either array while the 157-case sidecar recovery is active. The
optional receipts explicitly say that a hash of a pre-existing local file does
not establish public equivalence. `audit-local-archive` instead hashes the
compressed object and its one member in a single streaming pass, and can be
used for the ten retained archives or a separately approved spot check.

The batch script passes `bash -n`. On 2026-09-01, both `sbatch --test-only`
array ranges reached Slurm but were rejected by the transient
`QOSMaxSubmitJobPerUserLimit` because the user's existing queue was already at
its submit limit. Re-run test-only after that queue and the sidecar recovery
drain; this is an optional campaign precondition, not an activation blocker.
