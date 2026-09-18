# AhmedML candidate evaluator

AhmedML remains a closed, owner-review candidate. Public submissions are not
open. The benchmark definition is pinned to `neashton/ahmedml` revision
`02688c727cdb8dc8678e28abc6bbbb7e93c5fa15` and covers all eight published
split families using their real `run_N` case IDs.

The candidate evaluator scores complete native `CellData` fields without
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

## Canonical public data and derived cache

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
artifact rules, and compact per-case hash manifest. The approximately 39 GB
payload therefore does not need a second public upload. A production evaluator
must preinstall or rebuild it once and verify it before accepting submissions;
it must never rebuild support during an individual submission evaluation.

Use the metadata preflight for a quick deployment check and the deep mode when
provisioning the production cache:

```bash
python scripts/verify_ahmedml_derived_cache.py \
  --cache-root /path/to/preinstalled/ahmedml-cache

python scripts/verify_ahmedml_derived_cache.py \
  --cache-root /path/to/preinstalled/ahmedml-cache \
  --deep \
  --output /path/to/ahmedml-cache-verification.json
```

Deep mode hashes every generated NPY/NPZ payload against the exact expected
digest. Normal scoring repeats the relevant retained-file hash verification as
each support array is opened.

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

## Development fixture

`ahmedml-geotransolver-calibrated-dev-fixture-v1` is an explicit non-ranked
integration fixture. It deterministically perturbs public CFD truth on every
native cell. A retained legacy GeoTransolver checkpoint supplies only four
representative error magnitudes and is not executed. The checkpoint is not
known to correspond to any official AhmedML split.

The fixture exists only to exercise package validation and the development
leaderboard. It is permanently ineligible for ranking, citation, promotion, or
interpretation as surrogate inference. Its exact field metrics, forces,
profiles, and regional diagnostics are nevertheless recomputed by the same
dataset-owned evaluator path intended for future real submissions.

## Relevant files

- `submission-spec.json`: score, split, and lifecycle contract.
- `profile-definition-v1.json`: frozen 3-Cp/4-velocity 128-point profile rules.
- `derived-cache-contract-v1.json`: canonical-source, reproducible-derivation,
  and production-cache installation contract.
- `regional-diagnostics-v2.json`: evaluator-owned surface and volume regional
  rules. It retains `regional-diagnostics-v1.json` unchanged as the legacy
  volume partition bound by existing large per-case support.
- `reference/ahmedml/prediction_chunks.py`: AhmedML-owned interface to the
  shared bounded native-order prediction transport.
- `public-source-identity/`: pinned public file identities and entity counts.
- `reference/ahmedml/`: support loading, field evaluation, dataset reduction,
  and regional aggregation.
- `scripts/build_ahmedml_case_support.py`: materializes evaluator-owned support.
- `scripts/verify_ahmedml_derived_cache.py`: validates a preinstalled cache in
  metadata or full-payload mode and can emit a deployment receipt.
- `requirements-ahmedml-support-builder.txt`: exact NumPy and VTK versions for
  deterministic cache reconstruction.
- `scripts/run_ahmedml_cpu_workpool.sbatch`: packs independent support and
  fixture cases efficiently onto CPU-only Slurm nodes.
- `scripts/publish_ahmedml_candidate_support.py`: validates all 316 local cases
  and publishes compact hash-bound metadata while keeping large arrays local.
- `scripts/evaluate_ahmedml_candidate_case.py`: evaluates one complete case.
- `scripts/score_ahmedml_candidate_dataset.py`: reduces one official split.
- `scripts/assemble_ahmedml_schema_v3_candidate.py`: assembles a real
  checkpoint-inference package for any official split, verifies checkpoint
  bytes, and recomputes all evaluator-derived package products.
- `examples/ahmedml-v3-candidate/`: genuine-inference configuration template
  and complete packaging instructions.
- `pre-release-reference-registry.json`: initially empty maintainer registry
  for exact hash-bound, non-citable genuine-inference rows on the dev feed.
- `scripts/register_ahmedml_pre_release.py`: proposes or atomically appends one
  validated dev-only registry binding without creating approval.
- `scripts/assemble_ahmedml_schema_v3_dev_fixture.py`: creates the non-ranked
  schema-v3 development package.

The compact 316-case candidate metadata is already published and hash-bound in
`submission-spec.json`. If it is deliberately regenerated after a contract
change, `scripts/publish_ahmedml_candidate_support.py` validates every case
before atomically recording the new candidate manifest SHA-256. Regeneration
is not required merely to copy or rebuild an identical production cache.
