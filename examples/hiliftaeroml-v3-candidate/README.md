# HiLiftAeroML FluidsBench schema-v3 candidate package

This directory contains a reusable configuration template, the concrete
Full360 Transolver candidate configuration, six retained-split Transolver
configurations, and three additional finished-split Transolver configurations.
None is itself a submission, scoring-support release, or
accepted leaderboard result.
HiLiftAeroML remains a closed owner-review candidate.  Strings beginning with
`__REPLACE_` and `__UNRESOLVED_HILIFTAEROML_` are machine-detectable blockers
rather than illustrative release identities.

`transolver-full360-candidate-config.json` and the files under
`transolver-sixsplit-candidate-configs/` and
`transolver-finished-split-candidate-configs/` record real Transolver replays.
Their `evaluation.command` values are executable
from the FluidsBench repository root once all referenced campaign products
exist and target the canonical `package-a` directories. The evaluator
`code_revision` is
pinned to `68899f780d96b70f2badb5658971c87af0b17172`, the immutable
implementation commit recorded by the repository binding.  This freeze permits
only controlled maintainer-local candidate dry runs; do not replace it with
another working-tree revision.  It does not open submissions, approve the
candidate, or publish profile truth.

The package adapter consumes evaluator-native per-case products directly.  It
recomputes Cp and velocity profile R2 from prediction-only chunks joined to an
explicitly supplied local hidden-truth candidate; it does not rewrite case
identities, fill missing cases, or synthesize unavailable loads.  It accepts
every official HiLiftAeroML split in `submission-spec.json`; the 14 public
split labels map to the exact eight retained case sets.

The commands and Full360 reproducibility evidence below retain the native
profile-v1 workflow. An additive compact profile-v2 path is documented
separately later in this file; it does not alter or supersede that evidence.

Copy `package-config.template.json` outside the repository and replace the
participant fields with the submitted method's actual record.  The structured
methodology must describe the architecture and exact parameter count, all four
required surface/volume outputs, data handling and training stages, the raw
loaded checkpoint-file hashes, and measured complete-split inference compute.
`parameter_count_millions` is derived from the exact total parameter count and
must not be supplied separately.

Before attempting assembly, inspect all owner, configuration, and native gates
without writing output:

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/aggregate \
  --native-surface-outputs /path/to/native/surface/per-case/outputs \
  --native-volume-outputs /path/to/native/volume/per-case/outputs \
  --native-receipts /path/to/native/per-case/receipts \
  --candidate-profile-truth-release /path/to/hiliftaeroml-native-profile-truth-v1-candidate \
  --list-blockers
```

The completed profile-ground-truth candidate is repository-bound for this
maintainer-local dry run, but remains unpublished, unapproved, and inactive.
Supplying its local release path does not open public profile intake.  The
evaluator revision is frozen for this maintainer-local dry run.  Public
activation remains blocked on benchmark-owner approval, public profile-truth
publication, and the other activation-checklist gates.  Force and overall
scores still require exact complete truth-load coverage for every selected
case.  An `unavailable_validated_exception` is reported as a blocker;
imputation, zero-fill, case omission, and partial force/overall scores are
forbidden.

Once the required local candidate inputs and every evaluator-native case
product are complete, assemble a participant-owned directory:

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/aggregate \
  --native-surface-outputs /path/to/native/surface/per-case/outputs \
  --native-volume-outputs /path/to/native/volume/per-case/outputs \
  --native-receipts /path/to/native/per-case/receipts \
  --candidate-profile-truth-release /path/to/hiliftaeroml-native-profile-truth-v1-candidate \
  --output /path/to/hiliftaeroml-my-model-v1

python scripts/validate_submission.py \
  --candidate-dry-run \
  --candidate-profile-truth-release /path/to/hiliftaeroml-native-profile-truth-v1-candidate \
  /path/to/hiliftaeroml-my-model-v1
```

When both domains genuinely share one parent, `--native-outputs` remains a
backward-compatible shorthand. Do not use it when a corrected surface replay
is paired with retained volume outputs.

Build two independent archives and require byte identity before retaining the
submission ZIP:

```bash
python scripts/build_hiliftaeroml_submission_zip.py \
  /path/to/hiliftaeroml-my-model-v1 \
  /path/to/hiliftaeroml-my-model-v1.build-a.zip

python scripts/build_hiliftaeroml_submission_zip.py \
  /path/to/hiliftaeroml-my-model-v1 \
  /path/to/hiliftaeroml-my-model-v1.build-b.zip

cmp --silent \
  /path/to/hiliftaeroml-my-model-v1.build-a.zip \
  /path/to/hiliftaeroml-my-model-v1.build-b.zip
sha256sum \
  /path/to/hiliftaeroml-my-model-v1.build-a.zip \
  /path/to/hiliftaeroml-my-model-v1.build-b.zip
```

The two SHA-256 values must match. Archive members are sorted and receive
fixed timestamps, permissions, and compression settings; symbolic links and
non-regular source entries are rejected.

The commissioned Full360 Transolver replay completed this procedure over all
360 ordered cases. Both independently assembled 763-file packages passed the
candidate validator and had logical-tree SHA-256
`9c02d241eaf2ea6deecd4181816ed690d664b10f6eea0c2db4e5880aa2b1a0ab`.
Both 2,264,458,600-byte, 763-member ZIP builds had SHA-256
`a8dfd6ffbe6d103bc1a3123f6e6f756bacf960f92acc7079237b443bbbb62504`.
The resulting headline values were 69.15276784043088 overall,
48.167429082451235 for fields, 99.52105782630258 for forces,
80.75515537051848 for diagnostics, 0.9636931485730299 for Cp-cut R2, and
0.7034571571266213 for velocity-profile R2. This is reproducibility evidence
for a closed candidate only: it did not approve or activate the evaluator,
publish hidden truth, open submissions, upload or publish the package, or
create a public or private leaderboard entry.

## Additive compact profile-v2 multi-split exercise

The compact path is an inactive, unpublished maintainer-local candidate for
profile plots and profile R2 payloads only. Full native-surface Cp scoring,
wall-shear scoring, exact force integration, and exact pitching-moment
integration are unchanged and still consume the complete native surface
prediction. The compact representation does not make the native field outputs
optional.

Compact packages state this provenance boundary explicitly: the retained
dataset-evaluator revision applies only to the native-v1 base field, force, and
noncompact scoring path, while the compact-v2 implementation remains an
unbound worktree candidate with no code revision or implementation-manifest
digest. A future freeze must replace those null bindings explicitly.

Evaluator-owned support remains outside the participant package. It owns the
physical cut geometry and connected-graph topology, a frozen maximum of 128 Cp
samples per physical graph placed uniformly in physical arc length, and the
velocity coordinates, validity mask, weights, and ordering. A participant case
artifact contains exactly prediction values: `cp_q_delta` and scalar `float32`
`velocity_speed_over_u_inf`. The latter is stored as a lossless unsigned-delta,
byte-shuffled transform of its exact little-endian float32 bits. Both members
use deterministic level-9 ZIP Deflate so the dashboard can decode them with
browser-native decompression. See the candidate
[`native-profile-format-v2.json`](../../benchmark-specs/hiliftaeroml/native-profile-format-v2.json),
the
[`compact-cp-representation-decision-v1.json`](../../benchmark-specs/hiliftaeroml/compact-cp-representation-decision-v1.json),
and the
[`compact-cp-representation-audit-v1.md`](../../benchmark-specs/hiliftaeroml/compact-cp-representation-audit-v1.md).

With authorized native outputs and native-profile truth, materialize the local
evaluator-support release for the eight official case sets used by the ten
previews:

```bash
python scripts/materialize_hiliftaeroml_compact_profile_support.py \
  --submission-spec benchmark-specs/hiliftaeroml/submission-spec.json \
  --split benchmark-specs/hiliftaeroml/splits/full.json \
  --surface-outputs-root /path/to/full/native/surface/outputs \
  --volume-outputs-root /path/to/full/native/volume/outputs \
  --split benchmark-specs/hiliftaeroml/splits/geometry_scarce.json \
  --surface-outputs-root /path/to/geometry/native/outputs \
  --volume-outputs-root /path/to/geometry/native/outputs \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_4.json \
  --surface-outputs-root /path/to/aoa4/native/outputs \
  --volume-outputs-root /path/to/aoa4/native/outputs \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_12.json \
  --surface-outputs-root /path/to/aoa12/native/outputs \
  --volume-outputs-root /path/to/aoa12/native/outputs \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_22.json \
  --surface-outputs-root /path/to/aoa22/native/outputs \
  --volume-outputs-root /path/to/aoa22/native/outputs \
  --split benchmark-specs/hiliftaeroml/splits/aoa.json \
  --surface-outputs-root /path/to/ood-aoa/native/outputs \
  --volume-outputs-root /path/to/ood-aoa/native/outputs \
  --split benchmark-specs/hiliftaeroml/splits/stall.json \
  --surface-outputs-root /path/to/ood-stall/native/outputs \
  --volume-outputs-root /path/to/ood-stall/native/outputs \
  --source-truth-release /path/to/hiliftaeroml-native-profile-truth-v1-candidate \
  --output-root /path/to/hiliftaeroml-compact-profile-support-v2-candidate
```

The split, surface-output-root, and volume-output-root arguments repeat in
matching positional order. A case present in more than one case set is
materialized once from its first declared route; every case-set index retains
its exact official member order.
The per-case native hashes retained by that release record its materialization
lineage, not one model's authorization. Each surrogate is instead bound to
its own native receipt hashes and must independently match evaluator-owned Cp
topology/coordinates and velocity station coordinates, ordering, validity
mask, and weights. This permits several surrogate predictions for the same
physical case without duplicating truth or evaluator support. A future change
that produces a digest other than the manifest currently pinned by
`submission-spec.json` must additionally pass
`--allow-manifest-rebind-from <currently-pinned-manifest-sha256>`. That flag
permits materialization only; it does not edit the specification, activate the
candidate, publish truth, or open submissions.

That output is evaluator-owned and must not be copied into the submission.
Pass it to the assembler and validator instead of the native-v1 truth flag:

For this path, add a top-level `compact_evaluation` object containing the
exact compact assembler `command` and RFC3339 `generated_at` time. The
assembler uses that record only in compact mode and preserves the existing
native-v1 `evaluation` record. The checked-in Full360 configuration shows both.

```bash
python scripts/assemble_hiliftaeroml_schema_v3_candidate.py \
  --config /path/to/package-config.json \
  --native-aggregate /path/to/native/aggregate \
  --native-surface-outputs /path/to/native/surface/per-case/outputs \
  --native-volume-outputs /path/to/native/volume/per-case/outputs \
  --native-receipts /path/to/native/per-case/receipts \
  --candidate-compact-profile-support-release /path/to/hiliftaeroml-compact-profile-support-v2-candidate \
  --output /path/to/hiliftaeroml-my-model-compact-v2

python scripts/validate_submission.py \
  --candidate-dry-run \
  --candidate-compact-profile-support-release /path/to/hiliftaeroml-compact-profile-support-v2-candidate \
  /path/to/hiliftaeroml-my-model-compact-v2
```

The compact assembler validates the complete package inventory before retaining
output. The benchmark does not impose a HiLift-specific package or ZIP size
cap; normal repository and hosting limits still apply. Deterministic archive
checks and local validation are not publication, activation, owner approval,
or evidence for a case set other than the package's exact declared split. The
native-v1 workflow and its retained package hashes above remain unchanged.

The ten-split compact exercise completed two independent assemblies per split.
The ten canonical package trees passed candidate dry-run validation against the
evaluator-owned all-case support; each corresponding A/B pair compared equal
recursively, and each independent deterministic ZIP pair is byte-identical.
The support selection and rebind receipt are
[`compact-profile-all-case-support-rebind-v1.json`](../../benchmark-specs/hiliftaeroml/compact-profile-all-case-support-rebind-v1.json).

| Split receipt | Cases | Regular package bytes | ZIP bytes |
| --- | ---: | ---: | ---: |
| [Full](../../benchmark-specs/hiliftaeroml/compact-profile-full360-validation-v1.json) | 360 | 11,758,641 | 8,724,067 |
| [AoA 4](../../benchmark-specs/hiliftaeroml/compact-profile-aoa4-validation-v1.json) | 36 | 1,258,072 | 842,039 |
| [AoA 12](../../benchmark-specs/hiliftaeroml/compact-profile-aoa12-validation-v1.json) | 36 | 1,372,495 | 955,829 |
| [AoA 22](../../benchmark-specs/hiliftaeroml/compact-profile-aoa22-validation-v1.json) | 36 | 1,376,137 | 960,142 |
| [Super scarce](../../benchmark-specs/hiliftaeroml/compact-profile-super-scarce-validation-v1.json) | 360 | 13,569,255 | 10,532,488 |
| [Geometry scarce](../../benchmark-specs/hiliftaeroml/compact-profile-geometry-scarce-validation-v1.json) | 360 | 12,365,210 | 9,322,198 |
| [Geometry super scarce](../../benchmark-specs/hiliftaeroml/compact-profile-geometry-super-scarce-validation-v1.json) | 360 | 13,250,900 | 10,205,122 |
| [Geometry](../../benchmark-specs/hiliftaeroml/compact-profile-geometry-validation-v1.json) | 360 | 11,555,256 | 8,517,360 |
| [OoD AoA](../../benchmark-specs/hiliftaeroml/compact-profile-ood-aoa-validation-v1.json) | 900 | 32,105,249 | 24,720,935 |
| [OoD stall](../../benchmark-specs/hiliftaeroml/compact-profile-ood-stall-validation-v1.json) | 723 | 24,979,559 | 19,015,374 |

There is no HiLift-specific aggregate package or ZIP byte ceiling. Each linked
inactive-candidate receipt records the exact package, archive, evaluator
support, profile-contract, case-set, and metric hashes.

The output uses the same FluidsBench schema-v3 submission, evidence,
case-metrics, scoring-support, and discretization envelopes as DrivAerML.  Its
profile payload is necessarily HiLift-specific: Cp is retained as disconnected
physical cut graphs with branch/segment topology, while velocity retains the
exact five stations (`B.2`, `B.3`, `C.1`, `C.2`, `C.3`), 801 rows per station,
and explicit invalid gaps.  Deterministic prediction-only NPZ artifacts exclude
all truth/reference arrays.  Their reported per-case and macro profile scores
are independently recomputed by the candidate validator against the inactive
hidden-truth release.  Optional regional diagnostics are report-only,
zero-weight evidence and cannot change official metrics or the overall score.

The adapter writes atomically and refuses an existing output path.  It creates
no `approval`, `maintainer-validation.json`, or
`prediction-artifact-checks.json`; those are benchmark-maintainer records.
Every consumed per-case artifact is fail-closed against the retained chain
`aggregate artifact manifest -> case metrics input hash -> case receipt ->
surface/volume summary -> artifact descriptor`; a missing receipt, swapped
case file, stale summary, path/size mismatch, or digest mismatch blocks output.
