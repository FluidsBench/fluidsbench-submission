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
- zero-weight near-body, wake, and farfield volume diagnostics.

Surface relative L2 is polygon-area weighted, with equal-polygon values
reported secondarily. Volume relative L2 uses equal native cells as the primary
metric, with cell-volume-weighted values reported secondarily. Profiles are
derived from the same complete fields used for spatial scoring; a submission
cannot supply its own profile values.

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
- `regional-diagnostics-v1.json`: mutually exclusive volume-region rules.
- `public-source-identity/`: pinned public file identities and entity counts.
- `reference/ahmedml/`: support loading, field evaluation, dataset reduction,
  and regional aggregation.
- `scripts/build_ahmedml_case_support.py`: materializes evaluator-owned support.
- `scripts/run_ahmedml_cpu_workpool.sbatch`: packs independent support and
  fixture cases efficiently onto CPU-only Slurm nodes.
- `scripts/publish_ahmedml_candidate_support.py`: validates all 316 local cases
  and publishes compact hash-bound metadata while keeping large arrays local.
- `scripts/evaluate_ahmedml_candidate_case.py`: evaluates one complete case.
- `scripts/score_ahmedml_candidate_dataset.py`: reduces one official split.
- `scripts/assemble_ahmedml_schema_v3_dev_fixture.py`: creates the non-ranked
  schema-v3 development package.

After all 316 local support cases complete, publish and bind the compact
candidate metadata with
`python scripts/publish_ahmedml_candidate_support.py --bind-submission-specification`.
The command validates every case before it
atomically records the candidate manifest SHA-256 in `submission-spec.json`.
