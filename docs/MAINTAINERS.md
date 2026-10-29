# Maintainer guide

Participant instructions are in [SUBMITTING.md](../SUBMITTING.md). This guide covers dataset activation, approval, and generated
releases. Run commands from the repository root. Keep scientific approval, submitted-data validation, and release publication
as distinct recorded steps.

## Responsibilities

FluidsBench follows the versioned [`open-reproducibility-3.0`](../OPEN_REPRODUCIBILITY.md) track. Evaluation ground truth and case
IDs are public. The submission process deliberately separates four responsibilities:

1. Dataset owners pin the original public field-bearing files and publish an immutable canonical scoring support covering every
   required entity in those files. The support declares exact arrays, point/node/face/cell association, stable IDs, authoritative
   area, length, volume, or cell-area weights, case set, and coverage rules.
2. Participants run their own model, create predictions for that complete support, calculate per-case and aggregate metrics, and
   create the required profiles and spatial-discretization records. Inference may be chunked and a method may use a different
   internal representation, but its final mapped predictions must cover every official entity. Participants may optionally link
   public, versioned source code, model, environment, and artifact documentation.
3. Participants must run the FluidsBench contributor-stage validator before opening a pull request. Their package remains
   unapproved and is not included in the public feed.
4. A maintainer validates the submitted files, hash chain, exact case/support coverage, and metadata, then approves the package for
   publication.

FluidsBench does not execute the submitted code or model, regenerate predictions, or recompute base metrics from full prediction
fields as part of required approval. It plots and tabulates the submitter-created values and compares them with the fixed public
ground truth. Optional prediction sharing and optional maintainer checks are recorded separately and do not affect approval, rank,
citation eligibility, or promotion eligibility.

## Review and approve a result

Use this approval workflow for all five open datasets listed in the [dataset index](../benchmark-specs/README.md).

1. Commit exactly one new submission directory. Do not modify schemas, specifications, validators, workflows, generated feeds, or
   existing submissions in the same pull request. Use a new globally unique `<series-id>-vN` submission ID and the revision rules
   in the [result reference](RESULT_FORMAT.md#result-versions).
2. Target the repository’s submission branch. AirfRANS accepts Full, Scarce and AoA; Reynolds remains closed. The other open datasets accept the scopes listed in the dataset index.
3. Complete the pull request checklist and resolve all automated validation failures.
4. Maintainers review scientific provenance, public-evaluation-use eligibility, metadata, submitted values, result-data licence,
   and any declared artifact metadata, then merge the contributor package while it is still unapproved and absent from public
   feeds.
5. A maintainer can run the **Maintainer approve and regenerate** workflow with the exact submission path, validator/approver
   identities, validation timestamp, and approval date. Select the submission branch; the approval PR targets the dispatched branch.
   The workflow creates the validation record, binds its PR URL and rebuilds the feeds.
   Approval does not publish a result while the leaderboard release is unreleased. After an official release, check the result ID, scores and profiles on the leaderboard.
6. Optional prediction checks are descriptive and never change approval or claim eligibility. The separate **Check optional
   prediction artifact metadata** workflow checks declarations only; it cannot award the blue badge. Follow the
   [full-split metric-verification procedure](OPTIONAL_VERIFICATION.md#maintainer-procedure) for an actual recomputation.

External pull requests are restricted to one entirely new submission directory. Maintainers may deliberately apply the restricted
`trusted-maintenance` label to allow a reviewed external repository-maintenance contribution; validation/approval changes must still
come from a maintainer-owned branch.

## Activate a dataset support release

Dataset support is approved independently. A dataset collaborator changes only their dataset directory in one pull request:

1. supply the ordered official evaluation case IDs, set each `case_id_status` to `official`, and update
   the split counts and SHA-256 values;
2. confirm and pin the original public file names, immutable revisions and hashes, field-array names, entity associations, required
   patches or masks, and exact entity ordering for every case;
3. publish stable support IDs and authoritative area, length, volume, or cell-area weights. Face- or cell-associated data use the
   corresponding face area,
   curve-segment length, cell volume, or cell area. Point- or node-associated data use benchmark-generated deterministic
   mass-lumped dual measures from the exact pinned mesh;
4. add the dataset's scoring-support manifest, case-set indexes, chunks, and any small local support artifacts. The support must
   bind the complete original entity set, public ground truth, the dataset-required relative-L2 weightings and aggregation (including any documented exceptions);
5. set `scoring_support.status` to `official`, add the immutable release ID, repository-relative manifest path, public HTTPS
   manifest URL, manifest SHA-256, and their name/date/pull-request URL under `owner_approval`; if support artifacts are remote or
   require dataset-specific loading, also add the pinned `publication_validation` receipt produced by that loader;
6. keep `submissions_open=false` while the scientific contract is under review, then set it to `true` only when that dataset is
   ready to receive v3 result pull requests; and
7. run `python scripts/validate_scoring_supports.py`.

That command validates only the releases declared by each specification. For an official dataset it schema-checks the manifest,
follows and hashes the manifest → case-index → chunk chain, checks local artifact hashes, and requires every case set to match its
ordered official split exactly. It also requires the manifest's metric bindings to cover exactly the field and case-scalar metrics
in `submission-spec.json`, including reduction, weighting, aggregation, evidence mode, and evaluator-version pins. A complete local
materialized-table release is smoke-loaded case by case. A remote or dataset-specific release instead requires a committed receipt
pinning the validator file/digest, manifest digest, normalized-support digest, case/support counts, reviewer, timestamp, and pull
request. Other datasets may remain `owner_review_required` and closed. A collaborator's approving pull request is the durable
scientific approval record; FluidsBench's protected-branch maintainer still performs the final merge.

## Generated leaderboard feeds

The unreleased leaderboard remains empty. Official releases contain only validated and approved submitted-data rows:

```text
leaderboard/
  manifest.json
  datasets/<dataset-id>.json
  all.json
  revisions.json
  claims/
    index.json
    <dataset-id>/<split-id>/<submission-id>.json
leaderboard.json
```

The website loads the selected scalar dataset lazily. `all.json`, `leaderboard.json`, and the dataset files contain only the latest
version of each result series and therefore define the current rankings. `revisions.json` is a separately hash-pinned, unranked
history containing every published version. Profile arrays are not copied into these feeds; each row contains a relative profile
index path, and the browser fetches only the selected geometry's chunk.

`leaderboard/manifest.json` also publishes the data-release identifier, generation time, source reference, contract version,
licence scope, archive URL, immutable asset base, SHA-256 digest of the complete scalar feed, and the expected release ID and
manifest digest for the public profile ground truth. Every feed row carries the SHA-256 digest of its profile index. An unreleased
leaderboard uses `archive_url: null`; an official release must provide an immutable HTTPS archive URL plus asset-base and interactive
release-view URLs containing the release ID. The separate full `source_commit` records repository provenance without creating a
self-referential commit hash.

HiLiftAeroML selects the compact-v2 prediction format and remains closed for
submissions. Its public comparison truth covers 1,355 unique cases and eight
case sets; plot truth does not replace the authorized evaluator support.

Official `asset_base_url` and `release_view_url` values are clean HTTPS directory bases: their final path segment is exactly the
safe lowercase release ID, they end in `/`, and they contain no query or fragment. Official builds preserve the manifest's explicit
timezone-qualified generation timestamp. If generated claims already exist for that official release ID, the builder refuses any
changed feed digest, ranking contract, claim set, or published claim metadata; a new release requires a new ID and matching URL
bases. This makes result permalinks and relative release-asset URLs identical across Python, JavaScript, archives, and CDNs.

Ranks are release records, not live-page claims. They are calculated within exactly one release, dataset, and split using the
dataset's declared ranking metric and its published decimal precision. Values are rounded in decimal with `decimal_half_up`, then
competition-ranked (`1, 2, 2, 4`); the rounded value is also the displayed value, so two visibly equal results cannot receive
different ranks. The feed publishes the raw value, rounded/display value, rank, ranked-result count, and tie count in each row's
`ranking` object.

Every release also generates one machine-readable result-claim record per feed row. The claims index hashes every record, and the
release manifest hashes the index. Each record identifies the release, result, ranking scope and policy, feed digest, source
submission, evaluation evidence, and profile-index bindings. Schema-v3 claims additionally bind the scoring-support identity and
URL, spatial report, spatial case records, and per-case metric evidence; a recorded optional prediction check is also hashed.
Official records additionally bind maintainer validation and provide an immutable result permalink based on
`data_release.release_view_url`; `archive_url` remains the separate DOI or data-archive landing page. Each claim's exact byte URL is
its repository-relative `leaderboard/claims/...` path under the immutable
`data_release.asset_base_url`. The build chain is deliberately one-way: ranked feed, feed digest, claim records, claims-index
digest, then manifest pin.

`bindings.result` identifies the exact-byte-hashed complete feed and the row's zero-based array index. A verifier hashes the feed
bytes, loads that indexed row, and compares its identity, ranking, eligibility, and artifact bindings with the claim. This avoids
language-dependent JSON number canonicalization while retaining a byte-for-byte release link.

The generated-artifact checker also verifies that `record_count` and `eligible_record_count` equal the claim-index contents; these
cross-field equalities are semantic checks beyond what the portable JSON Schema expresses.

Official records state that validation covered submitted data only. Prediction sharing and
optional check status are published as separate descriptive fields and do not change eligibility. The schemas are
[`schemas/releases/result-claim.schema.json`](../schemas/releases/result-claim.schema.json) and
[`schemas/releases/claim-index.schema.json`](../schemas/releases/claim-index.schema.json).

Maintainers rebuild and verify feeds with:

```bash
python3 scripts/manage_leaderboard.py build
python3 scripts/manage_leaderboard.py check
```

## Official split indexes

Each open dataset uses owner-approved ordered case IDs with `case_id_status=official`.
The submission binds the exact split digest. Review the case order, counts and
SHA-256 before opening a dataset or publishing a new contract.

## HiLiftAeroML local support for a coordinated dry run

HiLiftAeroML is coming soon. A maintainer with authorized native-profile truth
and complete native outputs for the target case set can materialize an evaluator-owned scoring support release locally:

```bash
python scripts/materialize_hiliftaeroml_compact_profile_support.py \
  --submission-spec benchmark-specs/hiliftaeroml/submission-spec.json \
  --split benchmark-specs/hiliftaeroml/splits/full.json \
  --surface-outputs-root /path/to/native/surface/per-case/outputs \
  --volume-outputs-root /path/to/native/volume/per-case/outputs \
  --source-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  --output-root /authorized/local/hiliftaeroml-compact-profile-support-v2-candidate
```

The `--split`, `--surface-outputs-root`, and `--volume-outputs-root` groups may
be repeated in matching order. Maintainers can instead reconstruct support for
all eight official case sets without prediction outputs from the exact frozen
authority already bound into the native-profile truth release:

```bash
python scripts/materialize_hiliftaeroml_compact_profile_support.py \
  --submission-spec benchmark-specs/hiliftaeroml/submission-spec.json \
  --split benchmark-specs/hiliftaeroml/splits/full.json \
  --split benchmark-specs/hiliftaeroml/splits/geometry_scarce.json \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_4.json \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_12.json \
  --split benchmark-specs/hiliftaeroml/splits/single_aoa_22.json \
  --split benchmark-specs/hiliftaeroml/splits/aoa.json \
  --split benchmark-specs/hiliftaeroml/splits/deflection.json \
  --split benchmark-specs/hiliftaeroml/splits/stall.json \
  --prerequisite-authority-index /authorized/local/hiliftaeroml-native-profile-truth-authority-v1.json \
  --source-truth-release /authorized/local/hiliftaeroml-native-profile-truth-v1-candidate \
  --output-root /authorized/local/hiliftaeroml-compact-profile-support-v2-candidate
```

That path validates the authority against the truth-release binding, reads no
surrogate prediction output, applies the same limit of 128 Cp points per
physical graph, and stores every overlapping physical case once. Its receipt
records how the authority record/payload hashes occupy the support format's
four legacy source-hash provenance slots. The completed two-build all-case
evidence is
[`compact-profile-all-case-support-validation-v1.json`](../benchmark-specs/hiliftaeroml/compact-profile-all-case-support-validation-v1.json).
It is the selected local support binding. This evidence does not open submissions.

Keep this benchmark/evaluator-owned support outside participant packages. Creating it does not publish support, approve the evaluator, or open submissions.

## Other closed candidates: owner-coordinated dry runs

Dataset owners may exercise other closed schema-v3 candidates before activation
only when the dataset specification pins an exact
`scoring_support.candidate_manifest` (status, release ID, repository file,
public URL, and SHA-256):

```bash
python3 scripts/validate_submission.py --candidate-dry-run submissions/<dataset-id>/<package-id>
```

The package declares `scoring_support.status=candidate`; the dataset specification remains closed (for example, `owner_review_required`). This mode requires
`submissions_open=false` and a candidate support manifest. It rejects approval
and maintainer-owned metadata and reports only non-approving candidate
validity. A successful dry run does not authorize a result PR for a dataset outside the routes listed in [the dataset index](../benchmark-specs/README.md).
`--candidate-dry-run` and `--contributor-stage` are separate modes and cannot be combined.
