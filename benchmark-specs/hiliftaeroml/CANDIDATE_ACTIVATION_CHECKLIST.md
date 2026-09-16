# HiLiftAeroML schema-v3 candidate checklist

This is a closed owner-review candidate. It does not open submissions, approve a
public scoring change, or supply missing result values. Its purpose is to make a
complete Transolver dry run fail closed against the native Table-5 evaluator
contract before the benchmark owner considers activation.

## Frozen candidate identities

- Dataset: `hiliftaeroml`, with fourteen split labels bound to eight unique
  ordered evaluation case sets (1,355 unique cases across their union).
- Full / Medium / Scarce / Super scarce: `caseset-ac791749e527`, 360 cases,
  ordered-case digest
  `ac791749e5279ecf6746fcce20e3ec32408fd33b22127d5270de968be7842acf`.
- Geometry / Geometry medium / Geometry scarce / Geometry super scarce:
  `caseset-53990ea68fa6`, 360 cases, ordered-case digest
  `53990ea68fa6043879c9e080542ddeffe9836421dcf29d417c6ab47245c4c231`.
- AoA 4: `caseset-7a743a20b3bd`, 36 cases, ordered-case digest
  `7a743a20b3bd4d4dbdb8b0316da982f0d3b38c6d38f479644c44d052d6539892`.
- AoA 12: `caseset-02fc12ff3494`, 36 cases, ordered-case digest
  `02fc12ff3494cc1d8d7c50cc1ee40bd8c1caaef85b7e97dbb8e5d0b88d28eb53`.
- AoA 22: `caseset-85ecccd9ccda`, 36 cases, ordered-case digest
  `85ecccd9ccda92ca6bb44eaffbb2de583ac81fc12edd24cad1f581d0ab13b5c4`.
- AoA extrapolation: `caseset-29693354ed8a`, 900 cases, ordered-case digest
  `29693354ed8ae4e08abb15efa54a818f78148e4a328dbdd88602ac505f004c44`.
- Deflection: `caseset-c0ecb14de138`, 360 cases, ordered-case digest
  `c0ecb14de138973cb20f64f6a7d1d6f9e4e6ea7442eba2620acdba39dd48d272`.
- Stall: `caseset-804491c8956e`, 723 cases, ordered-case digest
  `804491c8956e7d4c1a3604ffe91d4ceb06bd03931c7d2b3bbcaf094c8d4e4f8e`.
- Candidate scoring-support release:
  `hiliftaeroml-native-all-splits-support-v1-candidate`.
- Candidate scoring-support manifest digest:
  `98a9a8d015e42c80f5993e30da94201011bebf67ff44d5574a7ef68a8b5dfbee`.
- Regional report-only contract digest: `8cf926d06706b8cc7fd58d821f395bf8cb6565ef1f3aabe6be18e0a279101da4`.
- Closed candidate evaluator revision:
  `68899f780d96b70f2badb5658971c87af0b17172`, frozen for maintainer-local
  candidate dry runs only with no activation effect.

The scoring-support generator verifies all 1,355 unique evaluation cases
against the complete internal 1,800-case surface and volume integrity
inventories, then filters and sanitizes one reusable release case set for each
unique ordered set. Those inventories are evidence, not substitutes for
immutable public source-content hashes.

## Scoring contract exercised by the dry run

- Surface pressure and wall shear primary relative L2 use native nodal dual
  area. Equal-native-node relative L2 is retained as a diagnostic.
- Volume pressure and velocity primary relative L2 use one unit per retained
  valid native node. The retained mask is raw Float32 `avg(P) != 0.0` before
  normalization. No dual-volume secondary metric is part of this candidate.
- Each inference chunk emits additive numerator and denominator statistics.
  Chunks are summed over the complete case, then one square root is taken. No
  epsilon is added. All complete-case values in the selected split are
  macro-averaged equally.
- Relative field L1/L2 use the evaluator's nondimensional bases: pressure
  coefficient `(P-p_inf)/q_inf`, wall shear `tau_wall/q_inf`, and velocity
  `U/|U_inf|`.
- Existing paper-compatible field MAE/RMSE remain dimensional. Each complete
  case is inverse-scaled with its own `q_inf` or `|U_inf|` before all cases in
  the selected split are macro-averaged.
- Vector relative L2 uses one nodal weight on the squared three-component
  vector magnitude. Vector MAE/RMSE retain the native evaluator's
  per-component reduction over the three scalar components.
- Force coefficients use all selected cases equally. Prediction-side forces
  are converted from the checkpoint's training-`q_inf` basis and normalized
  with each case's `qRef` and `areaRef`; pitching moment additionally uses
  `chordRef` and exact degree-two ordered-fan integration about `forcesCoR`.
  Truth-load integration uses the same case references. The released
  `force_mom` CSVs at HiLiftAeroML tag `force-mom-overrides-v1.1` remain the
  reference/provenance record, including the six documented surface-integrated
  mean overrides. Cp uses exact rows A--J,
  physical connected-cut arc length, and independently centered connected
  graphs. Velocity uses exactly B.2, B.3, C.1, C.2, and C.3, physical polyline
  arc length, and within-station centering. Profile R2 is formed per complete
  case and then macro-averaged equally.
- Regional diagnostics have weight zero and cannot alter any official metric
  or score. They remain optional until the complete-split aggregate writer has
  passed end-to-end validation.

## Required selected-split dry-run package evidence

The current commissioned Transolver package exercises Full first. The same
requirements apply unchanged to every other split: a completed package must
contain real values derived from one surface checkpoint and one volume
checkpoint over every ordered case in the selected split.

1. `submission.json` with the two model components, exact checkpoint-file
   hashes, methodology, training and inference compute, candidate scoring
   support identity, and aggregate metrics.
2. `metrics/cases.json` with complete support counts, coverage, per-case field
   metrics, relative-L2 sufficient statistics, and per-case load coefficients.
3. `discretization.json` plus ordered `discretization/cases.jsonl`, separating
   training inputs, supervision, inference inputs, direct model output, and
   canonical-support mapping.
4. `profiles/index.json` and its chunks with every selected case and the exact ten Cp
   plus five velocity station aliases. The Cp serializer must retain disconnected
   graph identity without creating scoring connections between graphs.
5. `evaluation-evidence.json` binding all package hashes, aggregate values,
   checkpoint identities, code revision, evaluator revision, and split/support
   identities.
6. Optionally, `regional-diagnostics.json` in
   `hiliftaeroml-regional-diagnostics-aggregate-v2`. It must be omitted rather
   than fabricated if the complete-split aggregate is not available.

Revision-pinned public prediction fields are optional under schema v3. Omitting
`prediction_artifacts` is not a candidate-dry-run blocker and does not relax any
of the evidence above.

## Completed Full360 closed candidate replay

The commissioned Transolver Full replay completed for all 360 ordered cases.
Two independent assemblies produced the same 763-file logical package tree,
`9c02d241eaf2ea6deecd4181816ed690d664b10f6eea0c2db4e5880aa2b1a0ab`,
and both packages passed `--candidate-dry-run` validation. Two deterministic
763-member ZIP builds were byte-identical at 2,264,458,600 bytes with SHA-256
`a8dfd6ffbe6d103bc1a3123f6e6f756bacf960f92acc7079237b443bbbb62504`.
The portable completion-receipt identity is
`d6f65d6ed83b8d480a89b9bb4044c39e46f46987c4dd283f62e20a4d74aa6eb1`.

The replay scored 69.15276784043088 overall, 48.167429082451235 for fields,
99.52105782630258 for forces, and 80.75515537051848 for diagnostics. Its Cp-cut
and velocity-profile R2 values were 0.9636931485730299 and
0.7034571571266213. The case evidence covers 50,766,193,080 surface points and
83,728,136,475 retained valid volume points with no incomplete support record.

This closes only the technical Full360 force-and-overall replay gate. It does
not approve or activate the evaluator, publish hidden truth, open submissions,
upload or publish the package, or create a public or private leaderboard entry.

## Remaining activation gates after the evaluator freeze and successful internal dry run

- Retain the unchanged public surface/volume archive-object inventory at its
  frozen revision and separately bind the force release
  `force-mom-overrides-v1.1` (`bbec30bcfc6103309c1375c5228b3ad0a586bfaf`).
- Pin the extracted surface/volume VTU member
  SHA-256 identities for all 1,355 unique evaluation cases, including source
  arrays, association, ordering, and the retained volume mask.
- Finish and owner-approve the authoritative surface dual-area, Cp cut, load,
  and five-station velocity prerequisite chain for every case.
- Freeze a lossless Cp disconnected-graph profile serialization and publish a
  matching complete-split profile-ground-truth release.
- Exercise and independently audit the same-stream nondimensional relative and
  dimensional absolute field reductions, vector semantics, loads, and profile
  aggregation.
- Obtain separate benchmark-owner approval of evaluator revision
  `68899f780d96b70f2badb5658971c87af0b17172` before any public activation or
  accepted submission.
