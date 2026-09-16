# DrivAerML candidate evidence index

This directory contains implementation evidence, diagnostic pilot records, and
the dataset-owner scientific approval for the closed DrivAerML candidate
contract. Presence here does not make an artifact official scoring support or
an accepted submission. The exception is the explicitly scoped approval record
indexed below: it approves the cited scientific evidence and decisions but does
not activate submissions or official ranking. Repository-relative proposal,
profile, and evidence paths in the machine contract resolve from
`benchmark-specs/drivaerml/`; upstream dataset paths instead resolve within the
pinned `neashton/drivaerml` release. Historical evidence keeps the claims it
made when generated and is not rewritten after approval.

The benchmark owner decided on 2026-08-21 that the 209 discrete Cp probes are
excluded from the current participant submission and score. Every discrete-
probe Cp artifact indexed here is retained only as inactive research evidence.
It is not a participant requirement, evaluator dependency, composite component,
scoring support, or activation gate; its presence does not authorize probe
scoring. Four continuous Cp cuts remain in scope, but they require separate
extraction evidence and support that is not supplied by these probe artifacts.

| Artifact | SHA-256 | Scope and eligibility |
| --- | --- | --- |
| [`owner-scientific-approval-2026-08-28.json`](owner-scientific-approval-2026-08-28.json) | `448f2b852df7066fc311eed75ea94caa756ae6f63573d12a50798804c08929b3` | Neil Ashton's dataset-owner approval of the cited native support, force convention, 10 mm velocity support and tolerances, four continuous Cp cuts, native-v3 truth, display/scoring coordinates, and expected zeroth-order velocity stair steps. It explicitly does not freeze a production release, complete multi-model sensitivity, open submissions, or enable ranking. |
| [`force-replay-all484.json`](force-replay-all484.json) | `631cd02c3a4215b254489652c1d93dfd781ecdb9478ff1ab11f24294743e8a17` | Passing candidate evaluator replay for all 484 cases, including fixed-area audit and chunk invariance. It is covered by the 2026-08-28 scientific approval but does not by itself freeze a release or activate scoring. |
| [`native-volume-run1-run44-equal-cell-primary-pilot.json`](native-volume-run1-run44-equal-cell-primary-pilot.json) | `d009b6ac708fa320d21492b4cc44fc846b61e99b445836b5dedc704a934592d7` | Two-case implementation evidence for multipart reconstruction, native fields, raw order, complete coverage, and equal-cell metric invariance. It covers only `run_1` and `run_44` and is not scoring support. |
| [`native-volume-equal-cell-primary-all484.json`](native-volume-equal-cell-primary-all484.json) | `bda42a125ffb4d6484756e77ac7e495974f39d4bce3e663154322e9e560827c7` | Passing all-484 audit of 978 pinned VTU segments, native `CellData`, tuple/component counts, finite values, declared units, raw-cell order and complete duplicate-free coverage. Two chunk partitions agree within `2.22e-16` for additive statistics and `5.69e-14` for metrics. This directly exercises the equal-native-cell volume contract; it is not a physics-null baseline, model result, or scoring support. |
| [`native-volume-equal-cell-primary-all484-provenance.json`](native-volume-equal-cell-primary-all484-provenance.json) | `b5ffe2234bb1597cf041ff5d97458f3d0d6e81db7a30591a7e26f51ebc032fde` | Path-free provenance for the all-484 audit: exact committed generator revision and source hashes, Python/NumPy runtime, launcher hash, scheduler scope, aggregate hash, and a canonical digest over all 484 external case-receipt identities. It does not turn the audit into scoring support or bundle the 10.7 MB receipt set. |
| [`velocity-mapping-run1-run44-candidate-v7-pilot.json`](velocity-mapping-run1-run44-candidate-v7-pilot.json) | `91a7bb4cb7b7c6167c3577f62745021409c93cebc6b2685a57cdfe1debc06d76` | Explicit non-public candidate-v7 containing-cell geometry pilot for `run_1` and `run_44`. The compact aggregate accounts for all 134,768 requested rows at 1, 2, 5, and 10 mm, including 4,489 invalid rows, and hash-binds the complete duplicate-free external row-level mapping files. It is not an all-case tolerance replay, prediction convergence result, model-ordering result, validity mask, or scoring support. |
| [`velocity-run1-run44-candidate-v7-pilot-provenance.json`](velocity-run1-run44-candidate-v7-pilot-provenance.json) | `be785d01618584691a60bba9faa8125376ac788dae1cfeeaf763e97fc7ff767e` | Path-free binding for the two-case velocity pilot: exact committed mapping revision and source hashes, Python/NumPy/VTK runtime, launcher and scheduler identities, external receipt and mapping identities, and a byte-identical separate-process `run_1` repeat. It makes no activation, approval, model, convergence, or participant dry-run claim. |
| [`real-reference-driver-run1-run44-candidate-v7-pilot.json`](real-reference-driver-run1-run44-candidate-v7-pilot.json) | `da80aebb11eb18d3bb66b7a968a94e9a779a6e60f5413ef73d2fbd6485d76572` | Passing candidate-only reference-driver receipt for pinned real `run_1` and `run_44` inputs. It verifies two- and three-part native-volume transport, complete deterministic all-zero surface and volume prediction chunks, equal-native-cell volume weighting, and consistent prediction identities across the core and diagnostic evaluators. The chunks are a transport and reduction fixture, not a model, physics-null baseline, participant dry run, or scoring result. |
| [`real-reference-driver-run1-run44-candidate-v7-pilot-provenance.json`](real-reference-driver-run1-run44-candidate-v7-pilot-provenance.json) | `24fd18d70098e5943959c8f37e2cd0ade211322e0785734d0dd035200db8fff2` | Path-free binding for the reference-driver pilot: exact committed driver revision and source hashes, runtime, launcher and completed scheduler job, execution-config hash, all-zero fixture receipts and manifests, fixed surface-area arrays, historical candidate discrete-probe Cp supports, 10 mm velocity mappings, and external child-evidence identities. The probe inputs record what the historical pilot consumed and are neither current Cp-cut support nor current requirements. The artifact does not make the candidate evaluator frozen or public scoring support. |
| [`volume-weight-vtk-run_1-failure-diagnostic.json`](volume-weight-vtk-run_1-failure-diagnostic.json) | `aa2a209cafbd598930bfbe2dd1a73e8c06108188aff69c30841bfc46bfe7927e` | Superseded historical record for a rejected geometric cell-volume experiment. Geometric volume weights are not part of the equal-native-cell contract; this file has no scoring or activation role. |
| [`volume-weight-vtk96-run_1-wedge-probe.json`](volume-weight-vtk96-run_1-wedge-probe.json) | `2970c507038bc1c3978542cc8e07c6682230db496f646a4887467645304a58a6` | Superseded historical isolated-cell experiment under VTK 9.6.0. Geometric volume weights are not part of the equal-native-cell contract; this file has no scoring or activation role. |
| [`cp-mapping-all484-hardened.json`](cp-mapping-all484-hardened.json) | `634e95279a2fb1078b3547616f29ddfc0a38ffe03f0b487fa0688be95aadbe81` | Inactive research sweep over 484 cases and all 101,156 case/probe rows. It retains 100,281 valid rows and all 875 invalid rows with explicit reasons. It has no submission, scoring, or activation role. |
| [`cp-stl-inventory-all484-hardened.json`](cp-stl-inventory-all484-hardened.json) | `cd4e1788a633196524b11f9d5f8868df05b71b11fed7fdf86892678c9cf41676` | Inactive research named-STL inventory for 484 cases: 364,568,214 raw facets, 23,716 named solids, and 68,902,520,476 source bytes. It is deliberately separate from the native-source pin and is not scoring support or an activation dependency. |
| [`cp-owner-review-atlas-all484-hardened.manifest.json`](cp-owner-review-atlas-all484-hardened.manifest.json) | `6a72536911010bdb741586018d1a13887f19756e135e8d35e6ea2da4b9fc7c4c` | Compact manifest for the external 488-page inactive discrete-probe research atlas. It covers all 484 cases and retains all 101,156 rows, including 875 invalid rows and 999 review-flagged rows. The external PDF is hash-bound as `e133fa14b6150c6f60590d19ba9b590072da0b758a19383976739c0f0fce654e`; visual sign-off is not claimed or required for the current probe-free contract. |
| [`volume-cell-types-run_1-pilot.json`](volume-cell-types-run_1-pilot.json) | `63f4c794807fd4d327d2e5017db585b41047c758e1ca9eeb4d22f2010817092e` | Diagnostic-only native-cell inventory for `run_1`; it is not a volume-weight definition or positivity result. The mesh contains 6,349,119 `vtkPolyhedron` cells. |
| [`volume-cell-types-run_44-pilot.json`](volume-cell-types-run_44-pilot.json) | `4f3a1cc6bf1c4010daf4a94c2bdf56cc9fb4a684e8cb2610dce6f00bcf482f71` | Diagnostic-only native-cell inventory for `run_44`; it is not a volume-weight definition or positivity result. The mesh contains 7,509,644 `vtkPolyhedron` cells. |
| [`cp-mapping-run1-run44-pilot.json`](cp-mapping-run1-run44-pilot.json) | `7200e6a97356b8edffa9e7eb1cd22fe31d79daf8551a7a9405273a6c5591a0ea` | Inactive discovery-only two-case Cp research pilot. The file declares itself incomplete with `public_evidence_eligible=false` and `public_scoring_support_eligible=false`. It records seven invalid mappings and omits 482 official cases. It is not scoring support or an activation dependency. |
| [`cp-stl-inventory-run1-run44-pilot.json`](cp-stl-inventory-run1-run44-pilot.json) | `db2b4031bf0051a51c4709f90e11eaee8b3886b858ce96649aa381778c9c5b0a` | Inactive discovery-only two-case named-STL research inventory. The file declares itself incomplete with `public_evidence_eligible=false` and `public_scoring_support_eligible=false`. It has no role in the current submission or activation. |
| [`transolver-run419-relative-cp-evaluator-smoke-v1.json`](transolver-run419-relative-cp-evaluator-smoke-v1.json) | `70851f3b1204255773f895a725156cb33d6c3bff7fabe6995c0ab2c64e552e5f` | Byte-identical retained Cp-cut evaluator report from real Transolver checkpoint inference for `run_419`. It exhausted and hash-verified all eight surface prediction chunks and records the four real relative Cp series, including their evaluator-produced support identities. It is a one-case positive integration fixture only: report-only weight remains zero, it does not itself freeze the evaluator or bind checkpoint identity, and it is not a submission, quality claim, sensitivity review, approval, or activation artifact. |
| [`transolver-run419-relative-velocity-evaluator-smoke-v1.json`](transolver-run419-relative-velocity-evaluator-smoke-v1.json) | `b6c35ba11938506e5d4897a81ad459db1383e56472207d489d63a843ae4be417` | Byte-identical retained velocity-profile evaluator report from real Transolver checkpoint inference for `run_419`. It exhausted and hash-verified all 122 volume prediction chunks and contains prediction/truth series for all 16 relative velocity stations. This report predates per-station support and placement-receipt identity emission; only family-level mapping and receipt hashes are present, so it cannot by itself exercise the relative submission semantic identity fields. It is a one-case report-only smoke fixture, not all-case sensitivity, scoring support, approval, or activation evidence. |
| [`transolver-run419-current-relative-profile-v3/current-case-evaluation.json`](transolver-run419-current-relative-profile-v3/current-case-evaluation.json) | `6aa349290049ad5498dd3a578865fc14aee5521be1ebc43208b43082a7a24bdf` | Complete current core-evaluator replay for the original real Transolver `run_419` predictions. It verifies the pinned native surface/volume sources, fixed surface areas, all eight surface chunks and all 122 volume chunks, and the native-cell reductions. It is compact regression evidence, not an official submission or scoring result. |
| [`transolver-run419-current-relative-profile-v3/current-case-implementation-receipt.json`](transolver-run419-current-relative-profile-v3/current-case-implementation-receipt.json) | `26b9b67be887fd991c63bbab04c68192768edceac65813815b2d1073a4568be1` | Deterministic path-free receipt binding the preceding evidence bytes and runtime to evaluator commit `b9db402a3fa0efbb94fe36be2ba1fe6f7b4bc1e1`, with preflight and postflight implementation-byte verification. |
| [`transolver-run419-current-relative-profile-v3/profiles/index.json`](transolver-run419-current-relative-profile-v3/profiles/index.json) | `654d5afec080e0f89ba3fa5a50135e72af23eef4af0f9b1ba1bf22a3f757c548` | One-case candidate profile index for the current schema-v3 regression fixture. It remains candidate-only and does not replace the public v1 dummy submission. |
| [`transolver-run419-current-relative-profile-v3/profiles/chunk-000.json`](transolver-run419-current-relative-profile-v3/profiles/chunk-000.json) | `3a5bca4c5f8730dcae4e29f69f415dc393050c6ebca6cf6e1303d307985bf262` | Real evaluator-produced 40-series `run_419` fixture: 16 constant and 16 relative velocity profiles, four constant Cp cuts, two shared centreline Cp aliases, and two materialized moving Cp cuts. Every materialized series is bound to retained support, placement-receipt, coordinate and prediction arrays; the two aliases use `shared_support_ref`. |
| [`transolver-run419-current-relative-profile-v3/provenance.json`](transolver-run419-current-relative-profile-v3/provenance.json) | `62ef43e1f5b8fe30c8b03fc36d315c4ad1b6faa242b0ec56c1ee0cb41cd5576a` | Reproducible path-free provenance for the current fixture. It binds dataset revision `7a5c0948ce27be709b1116a3a190f806e7a8f79f`, model revision `96477aeb86d24c26ccf0797bca1b3851268017d0`, every prediction chunk, all retained relative-support manifests and indices, evaluator implementation, series lineage and non-activation claims. |
| [`drivaerml-native-profile-truth-run419-validation-v1.json`](drivaerml-native-profile-truth-run419-validation-v1.json) | `ea614d5404a5a0c83283d2bc19d5153214297863a009cf7ff469921d86b8edf6` | Passing array-level validation of the first all-484 native profile truth publication against all 40 real Transolver `run_419` series. The companion native-v3 publication adds physical-x display coordinates without changing the approved values, scoring coordinates, support, or segment topology. |

The two historical `run_419` report files above deliberately retain the evaluator's exact bytes,
including its own scope-limit declarations. The Cp report can be checked directly
against the all-case support index for all four relative Cp support identities.
For velocity, only station coverage and real prediction/truth transport can be
checked: adding missing per-station identities after the fact would fabricate
evidence, so the fixture records that limitation instead.

The new current-format directory resolves that regression-fixture limitation by
replaying the producer supports and propagating their identities through the
current assembler. It deliberately emits only producer rows with valid native
support. The constant and relative velocity arrays therefore preserve known
interior gaps and must remain report-only until any future activation review
defines explicit gap semantics. Only one genuine trained checkpoint pair and
one prediction case were available. At least two additional distinct trained
DrivAerML checkpoints, with predictions over a common multi-case official
cohort, are still required for the three-model 10,000-replicate paired-bootstrap
sensitivity gate.

## Current blockers

The dataset-owner scientific review is complete. All-case native field, force,
fixed/relative velocity, and four-cut continuous-Cp evidence are published and
approved. The proposed resolution/tolerance campaigns are no longer separate
activation requirements; the owner accepted the published deterministic 10 mm
support and its explicit gap/tolerance semantics. Historical artifacts retain
their original non-approval claims because immutable evidence is not rewritten
after a later decision.

Only release and ranking-validation work remains:

1. bind the final production evaluator revision, scoring-support manifest and
   profile-ground-truth release identities in `candidate-release-bindings.json`;
2. run the three-genuine-model, paired 10,000-replicate sensitivity/bootstrap
   gate on a common official cohort;
3. run one independent full-split schema-v3 participant dry run against those
   frozen identities; and
4. create the final release-specific approval record, open submissions, replace
   prototype leaderboard status, and promote `dev` to production.

The final release-specific approval remains pending by design: its schema must
bind the not-yet-produced sensitivity evidence and the approving release
commit. That does not undo the scientific approval recorded above.

## Provenance limits

The force aggregate was produced by
`scripts/audit_drivaerml_force_case.py` and
`scripts/aggregate_drivaerml_force_replay.py`. Both native-volume aggregates
were produced by `scripts/audit_drivaerml_native_volume_case.py` and
`scripts/aggregate_drivaerml_native_volume_audits.py`; the complete aggregate
strictly revalidated all 484 case receipts and was reproduced byte-for-byte.
Its companion provenance file binds the exact Git revision, runtime, launcher,
and the ordered hashes of the external case receipts.
Those retained artifacts use the earlier v1 receipt vocabulary, in which the
unit-weight declaration was named `volume_weights`; their bound generating
revision remains the reproducibility source. The current v2 tools accept only
the unambiguous `volume_weighting: one_per_native_cell` schema and do not accept
or generate geometric volume arrays.
The inactive Cp research files were produced by the candidate Cp builder,
strict aggregator, and
`scripts/build_drivaerml_cp_owner_review_atlas.py`; the cell-type files were
produced by `scripts/audit_drivaerml_volume_cell_types.py`. The superseded
physical-volume failure diagnostic records an execution of the hash-bound
`reference/drivaerml/volume_weights.py` snapshot. Its 1,128-byte stderr log is
external and is bound by SHA-256 in the diagnostic; no generated weight array
or success receipt exists. The VTK 9.6.0 record similarly binds a compact
external probe, its script, exact NumPy and VTK wheels, and the upstream VTK
tag/commit. Neither historical experiment has a role in the current contract.

Current candidate source readers include retained-descriptor identity checks,
but compact aggregate JSON cannot by itself demonstrate that runtime mechanism.
The all-484 equal-cell audit now has a companion manifest binding its exact
generating Git commit, runtime, launcher, and ordered receipt identities; the
10.7 MB per-case receipt set remains external. Other candidate aggregates have
only the provenance recorded in their own files. The 20,883,275-byte atlas PDF
is also external; only its SHA-256, page count, and complete compact review
manifest are committed. Those limits prevent this directory alone from serving
as an independently replayable immutable scoring-support release.

The velocity aggregate was produced by
`scripts/generate_drivaerml_velocity_assignments.py` and
`scripts/aggregate_drivaerml_velocity_assignments.py`. Its companion provenance
file binds the exact committed source snapshot, runtime, launchers, scheduler
jobs, and identities of both external case receipts and all eight mapping
files. A second process reproduced the `run_1` receipt and four mappings
byte-for-byte. The external row-level artifacts are not bundled in Git, and the
repeat does not extend the pilot beyond two cases or provide scientific
validation of the candidate kernel choices.

The reference-driver receipt is a byte-identical copy of the external campaign
receipt. Its separate provenance file binds the exact source commit and hashes,
runtime, launcher, scheduler execution, pathful input-config identity, compact
identities for every fixed input, and all four external child evidence files.
The large native sources, fixed arrays, prediction chunks, mappings, and child
evidence remain external; their hashes do not turn this two-case fixture into a
standalone or immutable scoring-support release.
