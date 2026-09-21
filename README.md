<a id="fluidsbench-submission"></a>

# FluidsBench submissions

Evaluate a physics AI surrogate model and share its results on the [FluidsBench leaderboard](https://fluidsbench.org/).
This repository holds the dataset contracts, evaluators, result packages, and generated data feeds.

**Public submissions are currently closed for every dataset.** Work on `dev` is pre-release: prototype rows are illustrative;
registered HiLiftAeroML Transolver and GeoTransolver previews retain real inference and CFD comparison truth but remain
unapproved and non-citable. A local candidate validation pass does not open submissions or create a leaderboard entry.

## Start here

1. **Choose a dataset:** check the [dataset index](benchmark-specs/README.md), its exact split, and its current release status.
2. **Prepare your run:** follow [Submitting a result](SUBMITTING.md) and the dataset's guide. Public evaluation data is for final
   evaluation only; it must not influence training, tuning, model selection, or preprocessing statistics.
3. **Evaluate and package:** run the dataset evaluator on every required native entity, then collect its metrics, profiles,
   evidence, methodology, and spatial records. See the [result format](docs/RESULT_FORMAT.md).
4. **Validate and submit when open:** pass contributor-stage validation and open a PR containing one new result directory.
   Maintainers review the submitted data before a separate approval step publishes it.

For a coordinated closed-candidate dry run, start with the dataset-specific instructions:

| Dataset      | Start with                                                             | Packaging example                                                 |
| ------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| AhmedML      | [Native evaluator and support](benchmark-specs/ahmedml/README.md)      | [Candidate package](examples/ahmedml-v3-candidate/README.md)      |
| DrivAerML    | [Participant guide](benchmark-specs/drivaerml/PARTICIPANT_GUIDE.md)    | [Candidate package](examples/drivaerml-v3-candidate/README.md)    |
| HiLiftAeroML | [Participant guide](benchmark-specs/hiliftaeroml/PARTICIPANT_GUIDE.md) | [Candidate package](examples/hiliftaeroml-v3-candidate/README.md) |

Other datasets and their machine-readable specifications are in the [dataset index](benchmark-specs/README.md).
The generic v3 example is a synthetic format demonstration, not a ready-to-submit result.

## What you need to provide

Every new result uses schema v3 and the [`open-reproducibility-3.0` contract](OPEN_REPRODUCIBILITY.md): a public result package
with an open result-data licence, complete metrics and profiles, evaluation evidence, spatial records, and a
[methodology record](METHODOLOGY.md). The method record includes exact parameter counts, measured compute, and the SHA-256 of
every checkpoint file actually loaded by a parameterized method. Publishing the checkpoint bytes is optional.

Source code, model weights, environment files, artifact documentation, and full prediction fields are optional. If supplied,
their access, version, hash, and licence declarations must satisfy the contract. Their presence does not change ranking,
approval, citation, or promotion eligibility.

FluidsBench validates submitted files, coverage, identities, and hashes. Required approval does not execute your model,
regenerate predictions, or recompute base metrics from full fields. Optional prediction checks are recorded separately.

## Reference and help

- [Metric equations and edge cases](reference/README.md)
- [Examples: candidate configurations, synthetic demonstrations, and historical formats](examples/README.md)
- [Maintainer approval, dataset activation, and release feeds](docs/MAINTAINERS.md)
- [Questions and issues](https://github.com/neilashton/fluidsbench-submission/issues)

<a id="responsibilities"></a>
<a id="submission-directory"></a>
<a id="1-calculate-metrics"></a>
<a id="2-prepare-metrics-spatial-metadata-and-profiles"></a>
<a id="result-versions"></a>
<a id="3-validate-and-submit"></a>
<a id="generated-leaderboard-feeds"></a>
<a id="prototype-split-indexes"></a>

The former README sections now live in the [submission workflow](SUBMITTING.md),
[package and version reference](docs/RESULT_FORMAT.md), and [maintainer guide](docs/MAINTAINERS.md).

## License

Repository code and leaderboard metadata use [Apache 2.0](LICENSE). Submitters must have the right to publish their result
material under its declared open licence. Upstream datasets and optional artifacts retain their own licences;
[allowed SPDX identifiers](OPEN_REPRODUCIBILITY.md) are enforced by the schema.
