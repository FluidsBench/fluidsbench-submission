# Historical schema v2 package example

This directory is a complete, schema-valid illustration of the historical `open-reproducibility-2.0` file set. It demonstrates every v2
field and the checksum links between `submission.json`, `evaluation-evidence.json`, `profiles/index.json`, and `chunk-000.json`.
It is not benchmark-valid as checked in: values beginning with `replace-` or `replace_`, the one-case split, example URLs, and dummy
digests must be replaced with the selected official benchmark specification and the submitter's real data.

Do not use this directory for a new contribution. New contributor-stage submissions require schema v3 and
`open-reproducibility-3.0`; see [`../v3-template/`](../v3-template/).

The submitter creates:

- `submission.json` using `schemas/v2/submission.schema.json`;
- `evaluation-evidence.json` using `schemas/v2/evaluation-evidence.schema.json`; and
- the complete `profiles/` index and chunks using the v1 profile transport schemas.

Copy the directory to `submissions/<dataset-id>/<submission-id>/`, remove `maintainer-validation.json`, and then:

1. replace the dataset, split, case-set, metric, panel, station, and quantity placeholders from the official specification;
2. replace all scalar values and profile predictions with the submitter-produced results;
3. copy the exact profile ground-truth release ID and manifest SHA-256 from `leaderboard/manifest.json` into both
   `submission.json` and `evaluation-evidence.json`;
4. replace the public code, model, environment, documentation, licence, and digest metadata;
5. regenerate each chunk SHA-256, then the profile-index SHA-256, then the evaluation-evidence SHA-256; and
6. run `python3 scripts/validate_submission.py <submission-directory>` only when inspecting a historical package.

The maintainer—not the submitter—later creates `maintainer-validation.json`. Its example shows the exact validation scope and the
three hashes it binds. Once validated, the maintainer adds this object to `submission.json`:

```json
"approval": {
  "status": "approved",
  "approved_by": "FluidsBench maintainer",
  "approved_at": "2026-07-21",
  "pull_request_url": "https://github.com/neilashton/fluidsbench-submission/pull/123",
  "validation": {
    "evidence_file": "maintainer-validation.json",
    "evidence_sha256": "caf117e59d24aae822358afe03c7b7518a92e512bfc65f8170b5f94743f32696"
  }
}
```

Adding `approval` changes the bytes of `submission.json` but not `reviewed_submission_sha256`: that digest is deliberately computed
from the contributor submission with `approval` omitted.
