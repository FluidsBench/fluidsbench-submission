# HiLiftAeroML exact pitching-moment audit

This is a maintainer-only, read-only audit. It does not change a surface or
volume VTU, a force CSV, a benchmark score, or an activated release.

## Numerical contract

Native polygons use their ordered fan triangulation `(v0, vj, vj+1)` and the
released nodal fields are piecewise linear. Pressure follows the existing
positive convention

`Cp = (PROJ(AVG(P)) - 176.352) / qRef`, `force_p = +Cp n dA`,

and wall shear uses `Cf = AVG(TAU_WALL) / qRef`. Force is integrated exactly
with the degree-one barycentric rule. Moment about the case `forcesCoR` uses
the exact degree-two triangle mass matrix:

- `integral(lambda_i^2) = A/6`;
- `integral(lambda_i lambda_j) = A/12` for `i != j`.

The receipt reports exact pressure, viscous, and total moments as well as the
old vertex-lumped diagnostic. `CM` is positive body-y and is normalized by
`areaRef * chordRef`. Both numerical-only and published `cm_ci95`-aware
comparisons are reported; a scientific mismatch does not make a structurally
valid audit receipt fail.

Before a published drag/lift pressure+viscous closure residual can contribute
to any force-comparison tolerance, it must independently satisfy the canonical
CSV guard:

`abs(total - pressure - viscous) <= sum(the three CSV half-quanta) + 64 * eps64 * max(1, abs(total), abs(pressure), abs(viscous))`.

The receipt records each axis's values, three half-quanta, absolute residual,
rounding bound, Float64 scale/slack, acceptance limit, and pass status. A
closure beyond that rounding envelope aborts the case receipt; semantic replay
recomputes and validates the complete closure record.

The production Numba kernel iterates native cells directly. It does not
materialize fan triangles or a combined `N x 3` shear array. The NumPy kernel
is retained as an independent, testable reference.

## Frozen scopes and source identity

- All cases: 1,800 cases, numeric geometry/AoA order, newline digest
  `fb20b620338bd7f1671294e050a28adc673897c87e8cb7601ac57e528fca3056`.
- Frozen local inventory:
  `case_inventory_all1800.json`, SHA-256
  `bf31c77ff240764c4772bcebe7d49698e573023500bbad4d4e9591df4c2cca9b`.
- FluidsBench union: 1,355 cases in public-source numeric order, newline digest
  `00ce591c7c6c6c2e29e02f797c8cfd09dcc0d1d3911674ac1b07631c0125d352`.
- The same 1,355-member scoring-support union has legacy lexical-order digest
  `57dcf0b23cee897ed22e28e32a64ae1d1fb9cf03303fdafcc5e91066a36746d2`.
  The aggregate records both rules and fails if their memberships differ.

For every case, the live VTU size/mtime must equal the frozen 1,800-case
inventory. For public cases, the pinned archive member basename and declared
size must also agree; this is explicitly labeled name/size equivalence, not
local extracted-content equivalence. During integration the four required
logical point arrays receive namespace-, dtype-, shape-, and byte-bound
SHA-256s.

## Durable output

An isolated campaign root contains only compact artifacts:

- `cases/geo_LHC...json`: immutable successful case receipts;
- `failures/<case>/attempt-*.json`: preserved failed-attempt provenance;
- `locks/<case>.lock`: inert advisory-lock inodes; OS locks vanish on process
  death, so SIGKILL does not block resumption;
- `ranks/<job>-r<restart>/rank-000.json` through `rank-007.json`: actual
  assignment, completion, runtime, kernel-self-check, and per-case timing
  receipts;
- one aggregate and one aggregate-replay validation receipt outside `cases/`.

Aggregation requires all eight rank receipts from one Slurm attempt, proves
their actual union is exactly indices 0 through 1799 once, validates every
case receipt semantically, and rejects mixed source-code hashes, `p_inf`, cell
chunks, quadrature contracts, or frozen inventories. The attempt identity is
`<SLURM_JOB_ID>-r<SLURM_RESTART_COUNT>`, so requeue receipts never overwrite a
prior attempt. Aggregate and validation filenames receive the same attempt
suffix. A crash replay verifies and accepts an existing derived artifact only
when its bytes are identical; differing content fails closed. Validation
replays the aggregate from receipts byte-for-byte; it deliberately says
`source_reintegration_performed: false`.

## Launch gate

Do not launch all 1,800 cases first. Run two six-case pilots, each containing
the five known force exceptions plus `geo_LHC001_AoA_4` as a normal control.
Use separate output roots and different fixed chunks, for example 100000 and
131071. The CLI `--mode compare-pilots` requires identical algorithm and
backend hashes, source VTU/CSV identities, geometry, topology, logical field
hashes, reference/normalization values, published tokens, and quadrature. Only
the fixed cell chunk and execution provenance may differ. It compares force
and exact/vertex CM coefficients with recorded absolute and scaled-relative
tolerances of `2e-10`.

After both pilots and their comparison pass, the recommended full launch is
the eight-node/eight-rank `cpu-short` job in
`run_hiliftaeroml_exact_moment_full.sbatch`. Each rank owns indices
`rank, rank+8, ...`. The available eight-node CPU QoS has a four-hour wall
limit. (`normal` is not a CPU fallback: scheduler test-only rejects it with
`QOSMinGRES` because that QoS requires four GPUs; `cpu-normal` is capped at two
nodes.) Pilot per-case timings determine whether one attempt can finish. Each
rank uses an isolated Numba cache
and compiles/tests the production Float32, read-only point-field signature;
`PYTHONNOUSERSITE=1`, `PYTHONUNBUFFERED=1`, and sanitized `sys.path` provenance
are recorded. A timeout is resumable: submit the same campaign root again.
Valid cases skip, requeues receive a distinct restart attempt ID, and the
final attempt still emits all eight coherent rank receipts before aggregation.

The submit wrappers default to Slurm `--test-only`. Real submission requires a
second explicit sentinel. These tools do not upload to Hugging Face, publish,
activate, commit, push, touch the standalone submission project, or create a
private leaderboard.
