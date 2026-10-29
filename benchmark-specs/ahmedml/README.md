<a id="ahmedml-candidate-evaluator"></a>

# AhmedML evaluation guide

Start with the [resumable participant quickstart](QUICKSTART.md) for data/support
preparation, whole-split evaluation, recovery and validated packaging.

See the [dataset index](../README.md) for submission availability. The benchmark definition is pinned to `neashton/ahmedml` revision
`02688c727cdb8dc8678e28abc6bbbb7e93c5fa15` and covers all eight published
split families using their real `run_N` case IDs.

The evaluator scores complete native `CellData` fields without
participant-side resampling:

- surface `pMean` and `wallShearStressMean` on every boundary polygon;
- volume `pMean` and `UMean` on every native volume cell;
- drag and lift reconstructed from the complete surface prediction;
- three evaluator-owned surface Cp cuts with exactly 128 samples each;
- four evaluator-owned wake-velocity profiles with exactly 128 samples each;
- zero-weight dominant-normal surface diagnostics for pressure and wall shear;
- zero-weight near-body, wake, and farfield volume diagnostics.

Surface relative L2 is polygon-area weighted, with equal-polygon values
reported secondarily. Volume relative L2 uses equal native cells as the primary
metric, with cell-volume-weighted values reported secondarily. Profiles are
derived from the same complete fields used for spatial scoring; a submission
cannot supply its own profile values.

Start with the [packaging instructions](../../examples/ahmedml-v3-candidate/README.md).
You need the pinned public files, the automatically downloadable prepared cache, complete evaluator outputs for one official split, and actual model
checkpoint/spatial records. Run commands from the repository root. Shared package, version, and optional-artifact rules are in
[SUBMITTING.md](../../SUBMITTING.md).

Use the shared [hardware and compute fields](../../METHODOLOGY.md#hardware-and-compute)
for each training stage and inference campaign, including GPU model and allocation counts.

## Canonical public data and derived cache

The public dataset is canonical. Install the prepared cache for your split with
`python scripts/fetch_ahmedml_support.py --split full` (replace `full` as needed).
The evaluator finds the default cache automatically and downloads a missing
case before starting evaluation. Participants do not need to rebuild support.
The [first-time setup guide](../../examples/ahmedml-v3-candidate/README.md#first-time-setup)
covers alternate storage, offline use and the separately required public CFD files.

<details>
<summary>Source files, cache verification commands, profile geometry, and the run_492 force exception</summary>

The large local evaluator support is not an additional ground-truth release.
The pinned Hugging Face revision remains canonical:

- `boundary_<N>.vtp` contains the native surface geometry, polygon winding,
  and surface fields;
- `boundary_cell_area_<N>.npy` already contains one public scalar area for
  every native boundary polygon; and
- `volume_<N>.vtu` contains the native volume geometry, topology, and fields.

The evaluator deterministically caches four products for the 316-case union of
all official test sets. Oriented area vectors are reconstructed from public VTP
polygon winding and checked against the public scalar areas. Native cell
volumes are calculated from the public VTU with VTK 9.5.2 and are used only by
secondary physical-volume-weighted metrics. Region codes and frozen profile
mappings are also derived from the same public geometry. None of these four
cache products replaces or extends the canonical dataset.

`derived-cache-contract-v1.json` pins the public dataset revision, builder Git
revision and file digest, exact Python/NumPy/VTK environment, generated
artifact rules, and compact per-case hash manifest. The approximately 41.45 GB
payload is available as an optional, byte-identical public mirror in the
[`ahmedml-derived-cache-v1-dev` prerelease](https://github.com/neilashton/fluidsbench-submission/releases/tag/ahmedml-derived-cache-v1-dev).
`support-distribution-v1.json` pins one compressed archive per case. The mirror
does not replace the canonical dataset or change the scientific cache contract.
An evaluator installs or reuses verified support before evaluation; it never
rebuilds geometry during an individual submission. Maintainers can still
reconstruct and verify the identical cache from the public data when needed.

Use the metadata preflight for a quick deployment check and the deep mode when
provisioning the production cache:

```bash
python scripts/verify_ahmedml_derived_cache.py \
  --cache-root /path/to/preinstalled/ahmedml-cache \
  --split full

python scripts/verify_ahmedml_derived_cache.py \
  --cache-root /path/to/preinstalled/ahmedml-cache \
  --split full \
  --deep \
  --output /path/to/ahmedml-cache-verification.json
```

Deep mode hashes every generated NPY/NPZ payload against the exact expected
digest. Normal scoring repeats the relevant retained-file hash verification as
each support array is opened. Omit `--split` to verify all 316 cases, or use
`--case-id run_N` for individual installed cases. Deep native-array verification
runs on Linux; the downloader's checksum verification also works on macOS.

The profile geometry follows each case rather than remaining at fixed physical
coordinates. Cp targets use the case body/slant dimensions and actual surface
bounds; wake targets use that case's `L`, `H`, and `W`. The common body frame
keeps the rear plane at `x=0`, symmetry plane at `y=0`, and ground at `z=0`.
Each resulting native-cell mapping is frozen in evaluator support, so runtime
scoring never repeats a nearest-neighbour search.

The pinned `run_492` force CSV is the sole source-audit exception: its Cd/Cl
values differ from integration of the pinned complete native surface fields by
about 0.8%. The evaluator does not relax a dataset-wide tolerance or substitute
the CSV. It hash-binds and value-binds this one discrepancy and uses native
surface-field integration as force truth, exactly as it does for every other
case.

</details>

## Relevant files

<details>
<summary>Evaluator, support, and maintainer file reference</summary>

- `submission-spec.json`: score, split, and lifecycle contract.
- `profile-definition-v1.json`: frozen 3-Cp/4-velocity 128-point profile rules.
- `derived-cache-contract-v1.json`: canonical-source, reproducible-derivation,
  and production-cache installation contract.
- `regional-diagnostics-v2.json`: evaluator-owned surface and volume regional
  rules.
- `reference/ahmedml/prediction_chunks.py`: AhmedML-owned interface to the
  shared bounded native-order prediction transport.
- `public-source-identity/`: pinned public file identities and entity counts.
- `reference/ahmedml/`: support loading, field evaluation, dataset reduction,
  and regional aggregation.
- `scripts/build_ahmedml_case_support.py`: materializes evaluator-owned support.
- `scripts/fetch_ahmedml_support.py`: installs the frozen prepared cache by split or case.
- `support-distribution-v1.json`: binds the public mirror to the unchanged scientific release.
- [Support distribution maintenance](../../docs/AHMEDML_SUPPORT_DISTRIBUTION.md): packaging and fresh-download verification.
- `scripts/verify_ahmedml_derived_cache.py`: validates a preinstalled cache in
  metadata or full-payload mode and can emit a deployment receipt.
- `requirements-ahmedml-support-builder.txt`: exact NumPy and VTK versions for
  deterministic cache reconstruction.
- `scripts/publish_ahmedml_candidate_support.py`: validates all 316 local cases
  and publishes compact hash-bound metadata while keeping large arrays local.
- `scripts/evaluate_ahmedml_candidate_case.py`: evaluates one complete case.
- `scripts/score_ahmedml_candidate_dataset.py`: reduces one official split.
- `scripts/assemble_ahmedml_schema_v3_candidate.py`: assembles a real
  checkpoint-inference package for any official split, verifies checkpoint
  bytes, and recomputes all evaluator-derived package products.
- `examples/ahmedml-v3-candidate/`: genuine-inference configuration template
  and complete packaging instructions.

The official `ahmedml-native-support-v1` metadata is hash-bound in
`submission-spec.json`. Support publication is a maintainer task. A changed support contract requires a separately reviewed
release; participants use the existing published support. Regeneration is not
required to copy or rebuild an identical production cache.

</details>
