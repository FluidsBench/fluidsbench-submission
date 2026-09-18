# HiLiftAeroML compact-v2 candidate packaging

Compact-v2 is the sole selected participant profile representation. Public
submissions remain closed: immutable evaluator support publication, complete
implementation bindings, source pins, and release-specific owner approval are
still required. See the [participant guide](../../benchmark-specs/hiliftaeroml/PARTICIPANT_GUIDE.md)
for the fourteen training/evaluation labels, complete native-field requirements,
metrics, method records, and remaining gates.

The format contract and wire identifiers retain their historical `-candidate`
suffix. They are immutable identities, not the current lifecycle state. The
current [submission specification](../../benchmark-specs/hiliftaeroml/submission-spec.json)
selects that exact compact contract as official and rejects earlier native-v1
participant packages. Existing evaluator support, prediction bytes, validation
receipts, and all 23 registered previews remain unchanged.

## Prepare the configuration and native products

Copy `package-config.template.json` outside this example directory and fill in
all participant and release fields. Preserve the selected split's exact case
order. Every unresolved placeholder is a blocker; never invent a release hash,
checkpoint digest, metric, or compute measurement.

Use the dataset evaluator to produce complete surface and volume predictions,
case-set aggregates, and per-case receipts. Both domains are required for every
case. The assembler verifies the chain from aggregate hashes through case
receipts and surface/volume summaries to each consumed artifact. The native
source-truth exporter remains an internal prerequisite for evaluator support,
not a participant package path.

The top-level `evaluation` object records the exact compact assembler command
and its RFC3339 `generated_at` time. There is one profile path. The retained
Transolver example configurations cover eleven labels; update their method,
checkpoint, timing, command, and provenance fields for your own run.

## Inspect, assemble, and validate locally

Run from the repository root, using authorized evaluator-owned compact support:

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/aggregate \
  --native-surface-outputs /path/to/native/surface/per-case/outputs \
  --native-volume-outputs /path/to/native/volume/per-case/outputs \
  --native-receipts /path/to/native/per-case/receipts \
  --profile-support-release /path/to/hiliftaeroml-compact-profile-support-v2-candidate \
  --list-blockers

python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/aggregate \
  --native-surface-outputs /path/to/native/surface/per-case/outputs \
  --native-volume-outputs /path/to/native/volume/per-case/outputs \
  --native-receipts /path/to/native/per-case/receipts \
  --profile-support-release /path/to/hiliftaeroml-compact-profile-support-v2-candidate \
  --output /path/to/hiliftaeroml-my-method-v2

python scripts/validate_submission.py \
  --candidate-dry-run \
  --profile-support-release /path/to/hiliftaeroml-compact-profile-support-v2-candidate \
  /path/to/hiliftaeroml-my-method-v2
```

Do not bypass blockers. The output must not already exist. Repeating `--output`
exactly twice produces two independent packages while reusing one fully
validated support handle for the same ordered case set. The completed package
size is reported; there is no aggregate 15 MB cap because split sizes differ.
Per-file, shape, inventory, digest, and deterministic archive checks remain in
force.

Each case's NPZ contains exactly `cp_q_delta` and
`velocity_speed_over_u_inf`. Cp uses at most 128 samples per physical connected
graph, with evaluator-owned placement and fixed-int16 delta encoding. Velocity
uses the contract's lossless float32-bit transform stored as `uint8`. Coordinates,
topology, validity masks, weights, and truth remain in separate evaluator support.
Full native fields still determine field errors, forces, and pitching moment.
The public plot-truth release is insufficient for profile metric recomputation.

The assembler creates participant-owned schema-v3 files only. It never creates
approval or maintainer-validation records. Optional regional reports are
complete-split, report-only evidence with zero score weight. Local dry-run
success does not open public intake or create a leaderboard entry.

## Retained evidence and support

Eleven Transolver and twelve GeoTransolver packages are registered as unapproved,
non-citable preview references. Their unchanged validation receipts are listed
in the [dataset README](../../benchmark-specs/hiliftaeroml/README.md).
The [all-case support validation](../../benchmark-specs/hiliftaeroml/compact-profile-all-case-support-validation-v1.json)
covers all 1,355 unique cases and eight case sets. Its
[rebind receipt](../../benchmark-specs/hiliftaeroml/compact-profile-all-case-support-rebind-v1.json)
records compatibility and independent package checks. Historical commands in
these receipts describe the original runs; use the commands above for new work.

The frozen base evaluator revision does not attest the later compact adapter or
regional-v2 implementation. Their provenance remains explicitly
`unbound_worktree_candidate` until an immutable implementation release is bound.
Selecting the representation does not rewrite that evidence.

Maintainers can materialize support from authorized native source truth and
validated evaluator outputs, or directly from the frozen prerequisite authority.
The [participant guide](../../benchmark-specs/hiliftaeroml/PARTICIPANT_GUIDE.md)
describes the materializer and exact split routing. Keep evaluator support
outside participant packages and use the declared manifest digest unchanged.
