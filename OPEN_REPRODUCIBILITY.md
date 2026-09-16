# Open reproducibility contract

FluidsBench uses the versioned `open-reproducibility-3.0` contract for new leaderboard results. Evaluation case IDs, original
public field-bearing files, canonical scoring entities, benchmark-supplied area, length, volume, or cell-area weights, and ground
truth are public. The submitter runs the model
and creates every prediction, per-case metric, aggregate metric, profile, and spatial-discretization record. FluidsBench validates
and publishes those files, plots the submitted values, and compares submitted profiles with the pinned public ground truth.

Required approval does **not** include executing the model, regenerating predictions, or recomputing base metrics from full
prediction fields. Approval means that the submitted package passed the published submitted-data checks and maintainer review.
Optional prediction artifacts and optional maintainer checks are recorded separately and do not affect approval, rank, academic
claim eligibility, or promotion eligibility.

Public code, trained-model, environment, and artifact-documentation links are also optional. Omitting them does not affect
approval, rank, academic claim eligibility, or promotion eligibility. When a submitter chooses to provide one, its version, digest,
URL, and licence fields must satisfy the schema so the website does not present a mutable or ambiguous artifact.
The required `reproducibility.access=public` declaration applies to the submitted result package; it does not assert that any
optional external artifact was supplied.

This publication policy is separate from method identity inside required result metadata. In particular, every schema-v3
DrivAerML result records the SHA-256 digest of each raw checkpoint file whose bytes were loaded for inference, along with its role
and pre-evaluation selection rule. The digest identifies the method used to produce the result; it does not require the checkpoint
to be uploaded or made public. A separately declared optional model archive may contain those bytes and therefore may have a
different archive digest.

Every release manifest records the applicable identifier as `data_release.reproducibility_contract_version`.

## Public evaluation-data policy

Public evaluation data may be used only to calculate the final submitted result. It must not be used for model fitting, early
stopping, checkpoint or model selection, hyperparameter selection, architecture selection, manual result tuning, or preprocessing
statistics. A new submission records `reproducibility.public_test_data_use="evaluation_only"` as an explicit declaration.

If any public evaluation data influenced the model or its configuration, the result is not eligible for an official ranking under
this contract. The submitter must disclose that use rather than make the `evaluation_only` declaration.

## Canonical scoring support

An official dataset release publishes a hash-pinned scoring-support chain:

```text
manifest.json
  -> case-set index
     -> case chunks
        -> coordinates, area, length, or volume weights, ground truth, and stable support IDs
```

The dataset specification declares the support status, release ID, manifest URL and repository path, manifest SHA-256, whether
submissions are open, and dataset-owner approval. A v3 result can be submitted only when that exact support release is `official`,
owner-approved, and open. `prototype`, `owner_review_required`, and `retired` releases are closed; the specification gives the
explicit reason.

The scoring-support manifest defines each support's domain, original public file, exact arrays, point/node/face/cell association,
required patches or masks, entity ordering, quantities, components, weights, mapping and extrapolation policy, and metric bindings.
Its case set must cover exactly the official split, and each case support covers every required entity in the pinned original
field-bearing file. Each submitter joins predictions to stable benchmark `support_id` values. Coordinates, benchmark-supplied area,
length, volume, or cell-area weights, and ground truth come from the benchmark-owned support, not from the prediction artifact.
Mapping from a method's native
output to these entities is part of the submitted evaluation pipeline, so mapping error remains in the reported result.

Three-dimensional specifications describe body surfaces and flow volumes. Lower-dimensional specifications distinguish a
two-dimensional flow domain, a two-dimensional surface manifold embedded in three dimensions, and a one-dimensional boundary
curve. The standard relative-L2 policy reports an area- or length-weighted primary value and an unweighted secondary value on a
surface, surface manifold, or boundary curve; in a volume or two-dimensional flow domain, it reports an unweighted
primary value and a volume- or area-weighted secondary value. Face- and cell-associated data use authoritative face area, curve
length, cell volume, or cell area. Point- and node-associated data use deterministic mass-lumped dual measures supplied by
FluidsBench from the exact pinned mesh. Submitters do not create their own weights from a resampled or modified mesh.

## Submitter-supplied package

A new `submitted_evaluation` package uses `schemas/v3/submission.schema.json` and includes:

- `submission.json`, with aggregate metrics and exact scoring-support, spatial, per-case, profile, evaluation, and artifact
  bindings;
- `evaluation-evidence.json`, whose checksum binds the evaluation command, aggregate values, dataset and split, case set, scoring
  support, spatial report, per-case metrics, profile index, and public profile-ground-truth release;
- `metrics/cases.json`, containing every official case and support, the scored and expected counts, count and weight coverage,
  unmapped and extrapolated counts, per-case values, and additive relative-L2 numerator, denominator, entity-count, and total-weight
  evidence for the unweighted and area-, length-, or volume-weighted variants; global and dataset-reference metrics use explicit
  `aggregate_only` bindings
  rather than invented per-case surrogates;
- `discretization.json` and `discretization/cases.jsonl`, distinguishing training inputs, training supervision, inference inputs,
  direct model outputs, and mapping to each canonical scoring support;
- complete required profile chunks and their hash-pinned index; and
- an open result-data licence, plus any optional public, revision-pinned code, model, environment, and documentation metadata the
  submitter chooses to share.

The model's direct output does not have to use the public dataset mesh. A model may infer in memory-safe chunks or use sparse
points, structured grids, participant meshes, query points, or another declared representation. It must report the actual surface,
boundary-curve, volume, or two-dimensional-domain counts as applicable, together with sampling, domain or bounding box,
native-resolution comparison, connectivity, direct output counts, and mapping used. After that mapping, every required entity in
the exact original public support must have exactly one prediction.

If a native-resolution comparison is reported, every case record carries the corresponding model count, native count, and
fraction. Validation checks those case records against the summary and verifies the fraction arithmetically. This makes a method's
internal downsampling inspectable while the mapped evaluation result still covers the complete original public support.

Large supports may be processed in chunks. For each case and relative-L2 variant, the submitter sums the weighted squared-error
numerator and weighted ground-truth-square denominator across chunks, together with entity count and total weight, and takes one
square root only after the complete case is accumulated. Chunk-level L2 values must never be averaged. Cases are then
macro-averaged so each official test case has equal influence. For vector fields, one spatial weight multiplies the complete
squared vector magnitude.

The result-data licence is required. If optional code or model artifacts are declared, the v3 schema accepts only the corresponding
open SPDX identifiers listed in `schemas/v3/submission.schema.json`; `PROPRIETARY` and arbitrary unrecognised strings are rejected.
A missing licence identifier may be proposed in a separate repository-maintenance change; it must not be introduced through an
individual result package.

Optional code, model, environment, and documentation can support inspection and reuse by the community. Their inclusion does not
mean FluidsBench ran them, and their omission is not a validation failure. Any declared URLs must work without credentials, and
declared versions and hashes identify the exact referenced artifacts. Automated contributor validation checks URL syntax and
declared identifiers; maintainers review public accessibility without executing the code or model.

The contributor leaves `approval` absent and does not add `maintainer-validation.json` or
`prediction-artifact-checks.json`. Contributor CI is run with:

```bash
python3 scripts/validate_submission.py --contributor-stage submissions/<dataset-id>/<submission-id>
```

Passing that check means the package is structurally complete and internally consistent. It does not approve the result or include
it in the public leaderboard feed.

## Optional prediction artifacts and checks

A submitter may declare a revision-pinned public Hugging Face dataset containing either the complete scored predictions or a clearly
labelled example subset. Its manifest binds the scoring-support release and split, lists exact case coverage, and hashes its files.
The submission records its licence and manifest digest.

Normal validation checks only metadata and repository-local hashes. It never downloads a large remote artifact. A maintainer may
later perform an accessibility, format, or metric-recomputation check and record it in the maintainer-owned
`prediction-artifact-checks.json`. The public claim states whether predictions were shared and whether a check was recorded, but
neither the presence nor outcome of that optional check changes result eligibility.

## Maintainer validation and approval

Maintainers review the submission and run the repository validator over the submitter-supplied files. Required validation covers:

- schemas and identities;
- exact official split and complete original scoring-support coverage;
- support counts, count and weight coverage, mappings, and extrapolation policy;
- metric IDs, ranges, additive relative-L2 sufficient statistics, case-macro aggregation, and declared derived-score arithmetic;
- spatial summaries and per-case spatial records;
- profile structure, finite values, and the hash chain;
- the required result-data licence, any declared optional artifact metadata, and evaluation-data-use eligibility.

The maintainer records this in `maintainer-validation.json`, using
`schemas/v3/maintainer-validation.schema.json`. It binds the exact submission, evaluation evidence, scoring-support release,
spatial report, per-case metrics, profile index, dataset, split, case set, evaluator version, and public profile-ground-truth
release. Its scope is `submitted_data_only`, with model execution and metric recomputation recorded as `not_performed`.

A maintainer-owned approval change hashes that record under `approval.validation`, records the approver, approval date, and pull
request URL, rebuilds the feeds, and opens a draft pull request. The repository workflow can create this change from the exact
submission path and supplied identities and timestamps. Protected branches keep the final merge manual. The validating and
approving maintainer may be the same person, including when FluidsBench has a sole maintainer.

An approval is specific to the declared dataset and split, scoring-support release, contract version, evaluator version, metric
values, spatial and profile bytes, public ground-truth release, and any optional code commit, model digest, or environment digest
that was declared. Changing any of these requires a new submission ID and new validation and approval.

## Immutable releases and academic claims

An official release is served from an immutable asset base containing its release ID. Its manifest pins the scalar-feed digest,
claim-index digest, submitted profile-index digests, public profile-ground-truth release, and dataset scoring-support status and
identity. If generated claims already exist for an official release, the builder refuses changed feed or claim bytes under the
same release ID and URLs.

Each feed row has a hash-pinned result-claim record. It preserves competition rank within that exact release, dataset, and split,
using the published display precision. It binds the claim to the exact feed row and the available submission, evaluation,
scoring-support, spatial, case-metric, profile, optional prediction-check, and maintainer-validation records. A later release can
assign a different rank without changing the historical claim.

## Contract changes and historical schemas

Schema v3 and `open-reproducibility-3.0` are mandatory for new contributor-stage submissions. Previously approved schema-v2
packages remain readable and publishable under `open-reproducibility-2.0`; schema-v1 prototype fixtures remain interpretable.
Neither v1 nor a newly submitted v2 package can become a new official result.

Clarifications that do not change eligibility or required fields may retain the identifier. Before a contract version has any
official record, pre-launch corrections may also retain its identifier. Once an official record exists, any material change to
artifacts, submitted-data validation, or comparison semantics receives a new contract version. Existing approved records retain the
contract version under which they were validated.
