# Optional metric verification

**Metrics verified** means that FluidsBench recomputed the reported metrics from the shared scored predictions across the complete
test split. It does not certify that FluidsBench ran the model or that training never used public test data.

Verification is optional. Requesting it, sharing predictions, or waiting for a check does not change scoring, ranking, approval,
citation, or promotion eligibility. The existing evaluation-data-use policy still applies.

## Request a check

1. In your submission PR, tick **Request optional metric verification**. For an existing result, open an issue identifying its exact
   submission ID and revision. If predictions were not declared in that package, use the normal result-revision process to add them.
2. Declare public, revision-pinned **scored predictions covering the complete split** in `submission.json.prediction_artifacts`.
   Use the existing [artifact contract](../OPEN_REPRODUCIBILITY.md#optional-prediction-artifacts-and-checks): include the manifest,
   file hashes, licence, case coverage, split and scoring-support identity. Example subsets and direct model outputs alone do not
   qualify. Code, model weights and environment artifacts remain optional.
3. Give the artifact ID(s), total download size, and instructions for replaying the dataset evaluator. Keep the files available
   while the maintainer reviews them. Do not create a maintainer check record yourself.

A maintainer checks feasibility and schedules the work; there is no promised turnaround. A matching full-split replay can receive
the blue badge after review. Partial checks or unresolved differences receive no badge. The check details identify the result,
split, coverage, reviewer and date, with a link to the evidence record.

## Maintainer procedure

### 1. Pin the inputs

Use a maintainer-controlled workspace. Review the package with the usual validator and confirm the declared official case list,
complete scoring-support release, prediction manifest and file hashes. Check all declared scored-prediction artifacts; a complete
split is not enough if some required fields, forces or profiles are missing. Verify public access and the declared licence.

Retain the exact submission bytes, artifact revision/manifest digest, support release/digest, evaluator reference version and code
revision. Read and review the replay instructions before executing anything. Run the benchmark evaluator, not the submitted model.
Use benchmark-owned truth and weights; never substitute submitter-supplied truth or change the score definitions.

Normal metadata validation does not download prediction files or recompute metrics:

```bash
python3 scripts/validate_submission.py --prediction-check metadata submissions/<dataset-id>/<submission-id>
```

Closed candidates still use their documented candidate workflow and remain unapproved. A verification request never opens intake.

### 2. Recompute using the dataset's pinned evaluator

Download the declared immutable prediction files separately, after reviewing their size. Check the manifest digest against the
submission declaration before using it. Write the replay to a new directory outside the source submission and retain the command,
runtime/environment, logs and outputs.

For scoring releases supported by the generic materialized-table loader, the existing command is:

```bash
python3 -m reference.evaluate_predictions \
  --support-manifest /path/to/pinned-support/manifest.json \
  --case-set <case-set-id> \
  --prediction-manifest /path/to/pinned-predictions/manifest.json \
  --submission-id <submission-id> --split-id <split-id> \
  --output /path/to/new-replay/cases.json
```

The command hashes prediction/support files, joins exact support IDs and requires every indexed case. It is **not a universal
adapter** for dataset-native or compact formats. For those, use the dataset's reviewed evaluator and retain all of its field,
force, profile and aggregate outputs. Do not convert unsupported formats by guessing, omit metrics, or treat a metadata pass as
recomputation. If a complete replay is not available, leave verification pending.

### 3. Compare all values and retain a report

Compare the independently generated case evidence and every metric in the submitted package, including derived scores. A field-only
replay cannot verify a result that also reports forces or profile metrics. Where a native evaluator emits aggregate results
separately, supply its JSON object containing `metric_values` with `--recomputed-metrics`.

```bash
python3 scripts/compare_prediction_replay.py submissions/<dataset-id>/<submission-id> \
  --recomputed-case-metrics /path/to/new-replay/cases.json \
  --output /path/to/new-replay/comparison.json
```

The helper checks the submitted case-file hash, identities, ordered cases, supports, coverage, sufficient statistics and all metric
values. It records input digests and discrepancies, rejects incomplete output, and never alters the package or creates a badge
record. It ignores only the case report's generation timestamp. Floating metric/statistic comparisons use `rel_tol=1e-12` and
`abs_tol=1e-12`; identities and counts are exact. These are strict review thresholds, not a change to scoring or rounding rules.
Do not widen them merely to obtain a pass; investigate and document any evaluator/version/precision discrepancy.

A successful comparison is not proof that its input was independently recomputed. The reviewer must inspect the replay provenance
and full metric coverage. Never copy submitted values into a replay to fill missing metrics. Retain the evaluator revision,
command/environment, logs, prediction/support digests and comparison report together; record their immutable location and hashes
in the maintainer check notes.

### 4. Record and review the check

Only after the complete replay matches and a maintainer reviews the evidence, create the existing
[`prediction-artifact-checks.json`](../schemas/v3/prediction-artifact-checks.schema.json) in a **maintainer-owned** follow-up PR.
For each scored artifact, record:

- its exact `artifact_id`, declared `repository_revision` and `manifest_sha256`;
- `checked_by` and a timezone-qualified `checked_at`;
- `status="metrics_recomputed"` and `metric_recomputation="performed"`;
- `expected_case_count`, `checked_case_count` and `recomputed_case_count`, all equal to the independently confirmed complete split;
- `notes` identifying the evaluator version/revision, exact replay command, compared metric scope, and retained evidence location/digests.

Use the schema's top-level `submission_id` and version fields. Preserve other valid check records. For DrivAerML, also supply its
required dataset-native recomputation receipt where applicable; a generic field check cannot stand in for the native evaluation.
The existing validator checks the record's bindings and coverage; it does not establish that a reviewer actually ran the replay.

Run the normal package validation and feed build/check in the maintainer PR, following the
[release procedure](MAINTAINERS.md#generated-leaderboard-feeds). Do not rewrite immutable releases. Review the new feed/check digests,
badge details, and unchanged metric values/ranks before publication. Verification does not substitute for dataset activation or
required submitted-data approval.

If only accessibility, format or a subset was checked, use the corresponding honest status and counts. Record failures and resolve
discrepancies through ordinary review; never mark `performed` to hide missing coverage. Subsequent result revisions need their own
check against their exact artifacts and support.

## Local workflow check

The automated tests recompute the small synthetic reference example and exercise missing metrics, mismatches, coverage, identity,
file-hash and output-protection failures. They do not check a real submission, publish a check record, or award a badge:

```bash
python3 -m unittest tests.test_prediction_replay_comparison tests.test_scoring_support_v3
```
