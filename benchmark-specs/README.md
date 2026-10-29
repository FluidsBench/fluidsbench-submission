<a id="benchmark-submission-specifications"></a>

# Dataset index

Choose a dataset and follow its quickstart. Each guide covers the data, prediction format, evaluation and package generation.
For validation commands and the pull-request target, see [Submitting a result](../SUBMITTING.md#4-validate-locally).

| Dataset | Submission availability | Supported scope | Start here |
| --- | --- | --- | --- |
| AirfRANS | Open for submissions | Full, Scarce and AoA extrapolation; Reynolds is closed | [Quickstart](airfrans/QUICKSTART.md) · [Specification](airfrans/submission-spec.json) |
| AhmedML | Open for submissions | Eight prescribed splits; surface and volume | [Quickstart](ahmedml/QUICKSTART.md) · [Dataset guide](ahmedml/README.md) |
| WindsorML | Open for submissions | Eight prescribed splits; surface and volume | [Quickstart](windsorml/QUICKSTART.md) · [Dataset guide](windsorml/README.md) |
| DrivAerML | Open for submissions | Eight prescribed splits; surface and volume, or surface only | [Quickstart](drivaerml/QUICKSTART.md) · [Participant guide](drivaerml/PARTICIPANT_GUIDE.md) |
| DrivAerNet++ | Open for submissions | Pressure at every native surface point of the 1,154 official-test designs | [Quickstart](drivaernetplusplus/QUICKSTART.md) · [Specification](drivaernetplusplus/submission-spec.json) |
| HiLiftAeroML | Coming soon | Owner-coordinated evaluation trials only | [Participant guide](hiliftaeroml/PARTICIPANT_GUIDE.md) |
| BlendedNet | Coming soon | Proposed geometry-holdout split | [Specification](blendednet/submission-spec.json) |
| Rotor37 | Coming soon | Evaluation support is being prepared | [Specification](rotor37/submission-spec.json) |
| VKI-LS59 | Coming soon | Evaluation support is being prepared | [Specification](vki-ls59/submission-spec.json) |

Accepted packages are reviewed before publication on the leaderboard.
The quickstarts link to the package examples and exact scoring specifications; use those definitions for your evaluation.

For pressure units, offsets and coefficient conversions, see the [pressure reference guide](../docs/PRESSURE_REFERENCES.md).
Maintainers can find support-release states and activation requirements in the [maintainer guide](../docs/MAINTAINERS.md#activate-a-dataset-support-release).

## Shared requirements

Every dataset uses the same [hardware and compute fields](../METHODOLOGY.md#hardware-and-compute).
Report them per model result and training stage, not in the dataset's field contract.

All official case IDs and evaluation ground truth are public. Use evaluation data only for final evaluation, never for fitting,
selection, tuning, or preprocessing statistics. New packages declare `public_test_data_use="evaluation_only"` under the
[open reproducibility policy](../OPEN_REPRODUCIBILITY.md).

The support manifest pins the case index and chunks, which locate or hash the benchmark-owned coordinates, IDs, weights, and
truth. Final predictions must cover every required original entity in every selected case. Methods may use another internal
representation or chunk inference; the declared mapping and complete coverage remain required. Field associations distinguish
points, nodes, faces, and cells; domains distinguish 3D surfaces/volumes, 2D flow domains or embedded surfaces, and 1D boundaries.

Use the dataset's exact reductions and exceptions, the [metric equations](../reference/README.md), and the
[package/ranking reference](../docs/RESULT_FORMAT.md). Each official specification pins `evaluation_reference_version`;
submission and maintainer records must bind that same immutable evaluator
release and repeat the exact dataset, split, support, spatial, metric, and profile-truth identities. Required approval validates
submitted data; it does not execute models or recompute base metrics from full fields.

Dataset owners: follow the [support activation procedure](../docs/MAINTAINERS.md#activate-a-dataset-support-release), including
scientific approval, immutable publication bindings, coverage checks, and any loader-specific validation receipt.
