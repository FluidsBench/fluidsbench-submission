## Submission

- Dataset:
- Split:
- Submission ID:
- Model:

<a id="checklist"></a>

## Contributor checklist

- [ ] **Dataset and release:** I used the exact dataset/split version recorded in `submission.json`; the exact scoring-support
      release is official, owner-approved, and open for submissions.
- [ ] **Evaluation data:** I used public evaluation fields only for final evaluation, not fitting, selection, tuning, or
      preprocessing statistics. I disclosed any earlier contact below.
- [ ] **Complete evaluation:** I followed the published equations and dataset-specific reductions. `metrics/cases.json` covers
      every required case/support with complete count and weight coverage; profiles cover every required case, station, and
      quantity. Spatial records accurately describe representations, counts, domains, direct outputs, and mappings.
- [ ] **Methodology:** My schema-v3 method record covers every architecture component, exact total and submitter-trainable
      parameter counts, scoped inputs/outputs, data handling, training stages, every loaded checkpoint-file digest, and measured
      training/inference compute as applicable.
- [ ] **Evidence and bindings:** The recorded evaluation run produced `evaluation-evidence.json`. The reference version, command,
      evidence checksum, and any declared matching code revisions are accurate. Dataset version, split hash, case set, support
      release/hash, spatial hash, case-metric hash, and profile-ground-truth release/hash agree throughout the package.
- [ ] **Immutable package:** This PR adds exactly one new directory and does not change earlier results or repository machinery.
      Its ID is `<result_revision.series_id>-v<result_revision.version>`; v1 has `supersedes: null`, and later versions identify
      the immediately preceding published result and explain the change. I left `approval` absent and added no
      `maintainer-validation.json` or `prediction-artifact-checks.json`.
- [ ] **Optional artifacts:** Any supplied code/model/environment/documentation has accurate public URLs, revisions, digests,
      and licences. Any prediction-artifact declaration also has the exact manifest digest, support identity, split, and coverage.
- [ ] **Publication rights:** I have the right to publish the submitted metadata, profiles, and result material under the
      declared open result-data licence.
- [ ] **Validation and review:** I successfully ran
      `python3 scripts/validate_submission.py --contributor-stage <submission-directory>` and retained the calculation evidence
      that maintainers may request before approval.

## Notes

Describe preprocessing, dimensionalisation, external pretraining, unusual metric handling, and reproducibility limitations.
Disclose contact with evaluation data before the final evaluation run; it may make the result ineligible for ranking.

<a id="maintainer-validation-follow-up-pull-request"></a>

Maintainers: use the [separate approval checklist](../docs/MAINTAINERS.md#maintainer-validation-follow-up-checklist)
in the maintainer-owned validation/approval PR.
