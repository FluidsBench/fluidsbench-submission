<a id="benchmark-submission-specifications"></a>

# Dataset contracts

**All datasets are currently closed to public submissions.** Choose a dataset below, then follow [SUBMITTING.md](../SUBMITTING.md).
Each `submission-spec.json` defines the accepted splits, exact native fields and associations, support identities, metrics,
units, weights, profiles, evaluator version, and lifecycle gates consumed by the validator.

| Dataset      | Instructions / specification                                                                                | Candidate package                                          |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| AhmedML      | [Evaluator guide](ahmedml/README.md) · [Specification](ahmedml/submission-spec.json)                        | [Example](../examples/ahmedml-v3-candidate/README.md)      |
| DrivAerML    | [Participant guide](drivaerml/PARTICIPANT_GUIDE.md) · [Specification](drivaerml/submission-spec.json)       | [Example](../examples/drivaerml-v3-candidate/README.md)    |
| HiLiftAeroML | [Participant guide](hiliftaeroml/PARTICIPANT_GUIDE.md) · [Specification](hiliftaeroml/submission-spec.json) | [Example](../examples/hiliftaeroml-v3-candidate/README.md) |
| AirfRANS     | [Specification](airfrans/submission-spec.json)                                                              | —                                                          |
| BlendedNet   | [Specification](blendednet/submission-spec.json)                                                            | —                                                          |
| DrivAerNet++ | [Specification](drivaernetplusplus/submission-spec.json)                                                    | —                                                          |
| Rotor37      | [Specification](rotor37/submission-spec.json)                                                               | —                                                          |
| VKI-LS59     | [Specification](vki-ls59/submission-spec.json)                                                              | —                                                          |
| WindsorML    | [Specification](windsorml/submission-spec.json)                                                             | —                                                          |

## Check status before starting

The `scoring_support` statuses are:

| Status                  | Meaning                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| `official`              | Owner-approved support; intake still requires `submissions_open=true`. |
| `owner_review_required` | A proposal exists; it is not an activated official release.            |
| `prototype`             | Format or evaluator development only.                                  |
| `retired`               | Historical interpretation; closed to new results.                      |

Read `closed_reason` whenever a dataset is closed. A separate pinned `candidate_manifest` can permit a non-approving local
dry run. It does not substitute for an official, owner-approved, open release. Split indexes marked `official` bind ordered
evaluation cases; `prototype_generated` indexes are dummy case lists and cannot support real submissions.

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
