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

1. Commit exactly one new submission directory. Do not modify schemas, specifications, validators, workflows, generated feeds, or
   existing submissions in the same pull request. Use a new globally unique `<series-id>-vN` submission ID and the revision rules
   in the [result reference](RESULT_FORMAT.md#result-versions).
2. Open a pull request against `main` once FluidsBench announces that the dataset is accepting real submissions.
3. Complete the pull request checklist and resolve all automated validation failures.
4. Maintainers review scientific provenance, public-evaluation-use eligibility, metadata, submitted values, result-data licence,
   and any declared artifact metadata, then merge the contributor package while it is still unapproved and absent from public
   feeds.
5. A maintainer can run the **Maintainer approve and regenerate** workflow with the exact submission path, validator/approver
   identities, validation timestamp, and approval date. It creates the validation record, binds the resulting draft PR URL,
   rebuilds and verifies the compact feeds, and opens the protected-branch PR. The final merge remains manual.
6. Optional prediction checks are descriptive and never change approval or claim eligibility. The separate **Check optional
   prediction artifact metadata** workflow checks declarations only; it cannot award the blue badge. Follow the
   [full-split metric-verification procedure](OPTIONAL_VERIFICATION.md#maintainer-procedure) for an actual recomputation.

External pull requests are restricted to one entirely new submission directory. Maintainers may deliberately apply the restricted
`trusted-maintenance` label to allow a reviewed external repository-maintenance contribution; validation/approval changes must still
come from a maintainer-owned branch.

### Maintainer validation follow-up checklist

Complete this checklist in the separate maintainer-owned validation/approval pull request.

- [ ] Scientific provenance and retained evidence have been reviewed.
- [ ] The result-data licence and any declared optional artifact access, digests, and licences have been reviewed.
- [ ] A maintainer validated the submitter-supplied files, hashes, required coverage, and ground-truth comparison basis.
- [ ] `maintainer-validation.json` records `submitted_data_only`, `model_execution=not_performed`, and
      `metric_recomputation=not_performed`.
- [ ] The validation record binds the scoring support, spatial report, per-case metrics, profiles, and evaluation evidence.
- [ ] A maintainer added the validation checksum, `approval.status`, approver, approval date, and this pull-request URL.

## Activate a dataset support release

Dataset support is approved independently.A dataset collaborator changes only their dataset directory in one pull request:

1. replace prototype split IDs with the ordered official evaluation case IDs, set each `case_id_status` to `official`, and update
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

Source submissions are converted into release-specific compact scalar feeds. Prototype releases contain only explicitly marked
prototype rows; official releases contain only validated and approved submitted-data rows:

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
manifest digest for the public profile ground truth. Every feed row carries the SHA-256 digest of its profile index. Prototype
releases use `archive_url: null`; an official release must provide an immutable HTTPS archive URL plus asset-base and interactive
release-view URLs containing the release ID. The separate full `source_commit` records repository provenance without creating a
self-referential commit hash.

HiLiftAeroML's eleven registered Transolver previews and twelve registered
GeoTransolver previews use the sole selected compact-v2 prediction contract.
The historical candidate identifiers remain immutable; the current HiLift
submission specification records representation selection separately from
public release approval. Earlier native-v1 participant packages are not accepted. Their public
Cp and velocity comparison truth lives in
the website repository as a deduplicated 1,355-case release covering all eight
official case sets. It is bound as plot-only metadata; the private lossless
evaluator truth remains the scoring authority and is not copied into this
repository.

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

Prototype claim records remain useful for testing the interface, but explicitly set academic-citation and promotion eligibility to
false and have no immutable permalink. Official records state that validation covered submitted data only. Prediction sharing and
optional check status are published as separate descriptive fields and do not change eligibility. The schemas are
[`schemas/releases/result-claim.schema.json`](../schemas/releases/result-claim.schema.json) and
[`schemas/releases/claim-index.schema.json`](../schemas/releases/claim-index.schema.json).

Maintainers rebuild and verify feeds with:

```bash
python3 scripts/manage_leaderboard.py build
python3 scripts/manage_leaderboard.py check
```

## Prototype split indexes

Inspect each split's `case_id_status`. An `official` index contains owner-approved ordered case IDs; a `prototype_generated`
index preserves a dummy fixture's counts and cannot be used for a real submission. Replace prototype IDs, set `official`, and
review the updated counts and SHA-256 before opening intake. The submission records the exact split digest so later changes
are detectable. Official split IDs alone do not activate a scoring-support release.

## HiLiftAeroML local support for a coordinated dry run

Twenty-three retained real-surrogate compact packages comprise eleven Transolver
results and twelve GeoTransolver results. Together they cover Full, Scarce, the
three fixed-AoA splits, Super scarce, Geometry scarce, Geometry super scarce,
Geometry, and all three out-of-distribution splits (AoA, deflection, and stall).
The public plot-only truth separately covers every official case set. A maintainer with the
authorized native-profile truth and complete native outputs for the target case
set can materialize an evaluator-owned scoring support release locally:

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
It is the selected current local candidate support binding, and its ten-preview
rebind is recorded in
[`compact-profile-all-case-support-rebind-v1.json`](../benchmark-specs/hiliftaeroml/compact-profile-all-case-support-rebind-v1.json).
Both records remain intentionally non-activating.

Keep this benchmark/evaluator-owned support outside participant packages. Creating it does not publish support, approve the evaluator, or open submissions.
