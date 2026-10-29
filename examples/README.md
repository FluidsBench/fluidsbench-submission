# Submission examples

New results use schema v3. **AirfRANS, AhmedML, WindsorML, DrivAerML and DrivAerNet++ are open for submissions.** Follow the selected [dataset guide](../benchmark-specs/README.md) and
[submission workflow](../SUBMITTING.md); examples do not bypass release or approval gates.

HiLiftAeroML and the remaining datasets are coming soon.

## Dataset configurations

| Example                                             | Inputs and limits                                                                                                                                                                                                                                                            |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AhmedML](ahmedml-v3-candidate/README.md)           | Genuine inference on any of eight official splits; verifies each local checkpoint and re-reduces complete case evidence into metrics, profiles, spatial records, and regional reports.                                      |
| [WindsorML](windsorml-v3-candidate/README.md) | Complete native point/cell inference, all eight split labels, 16 fixed/relative profiles, checkpoint verification and package validation. |
| [DrivAerML](drivaerml-v3-candidate/README.md)       | Requires matching scoring support, evaluator, profile-v10, and profile-truth bindings. Unresolved release tokens block assembly. Config-v2 describes a schema-v3 package with the common methodology record and scope-specific outputs.                                    |
| [DrivAerNet++](drivaernetplusplus-v3-candidate/README.md) | Pressure-only predictions at every native surface point of all 1,154 test designs; prepared support and resumable evaluation. |
| [HiLiftAeroML](hiliftaeroml-v3-candidate/README.md) | Supports all official labels, native fields, disconnected Cp graphs, five velocity stations, and optional zero-weight regions. Requires complete truth-load coverage and authorized local compact-v2 support; evaluator-revision and public-release gates remain unresolved. |

Illustrative values in the DrivAerML [methodology example](drivaerml-v3-candidate/methodology.example.json) show structure only.
Never substitute example values or invented hashes for actual method records or missing release bindings. Maintainers review and approve packages before leaderboard publication.

## Demonstrations

| Example                                             | Use it for                                                                                                                                                                                                                  |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [v3 template](v3-template/)                         | Synthetic demonstration of methodology, support, sufficient statistics, spatial records, profiles, optional artifacts, validation records, and SHA-256 bindings. It is not an official result or a ready-to-submit package. |
| [Profile chunk](profile-chunk.example.json)         | Abridged Cp, Cf, and velocity JSON series; not a complete benchmark split.                                                                                                                                                  |
| [AirfRANS extraction](airfrans-profile-extraction/) | Hash-bound official-data fixture and pinned extrados-velocity extractor, including native point-ordered NumPy/PyTorch predictions without a predicted VTU.                                                                  |

The [result reference](../docs/RESULT_FORMAT.md) explains fixed support, exact coverage, geometry weights, vector reductions,
chunk accumulation, and dataset-specific exceptions. Submitters provide the evaluated result, profile, and spatial data. Full
prediction sharing is optional, as are public code, model weights, environments, and artifact documentation. Supplied artifacts
must meet the [access, revision, digest, and licence rules](../OPEN_REPRODUCIBILITY.md).

Equivalent JSON may be produced in any language. Contributor validation checks the package; maintainer submitted-data validation
and approval are separate, and required approval does not execute a shared model.
