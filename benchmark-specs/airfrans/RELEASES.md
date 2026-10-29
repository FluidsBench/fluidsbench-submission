# AirfRANS support downloads

For evaluation and packaging, follow the [participant quickstart](QUICKSTART.md).
This page describes the support releases and their independent verification.

Submissions are open for **Full, Scarce and AoA extrapolation**. Reynolds
extrapolation is closed. Full and Scarce share the ordered 200-case evaluation
set; AoA uses 196 cases. The native support and lossless profile truth cover
355 unique cases.

## Downloads

- [Native support](https://github.com/neilashton/fluidsbench-submission/releases/tag/airfrans-native-support-v1)
- [Lossless profile truth](https://github.com/neilashton/fluidsbench-submission/releases/tag/airfrans-native-profile-truth-v1)

The [submission specification](submission-spec.json) selects the release IDs,
manifest URLs and SHA-256 digests. The [approval binding](official-release-binding.json)
records the approved artifact hashes. Verify every download before evaluation.

## Install and verify

Use the checkout specified in [Submitting a result](../../SUBMITTING.md) and install `requirements.txt`. Keep the two
downloads in separate directories because they contain files with the same names.
For example, with GitHub CLI:

```bash
mkdir support-download truth-download
gh release download airfrans-native-support-v1 --repo neilashton/fluidsbench-submission --dir support-download
gh release download airfrans-native-profile-truth-v1 --repo neilashton/fluidsbench-submission --dir truth-download
```

In each download directory, run `shasum -a 256 -c SHA256SUMS` (macOS) or
`sha256sum -c SHA256SUMS` (Linux) before extracting. The support download includes
its JSON schemas. Extract `support-metadata.tar.gz` and `native-tables.tar` into
the **same empty directory**. It will contain `manifest.json`, `case-sets/` and
`tables/`. Archive descriptors bind both the archive and each installed member.
The evaluator reads verified local members and does not fetch files itself.

```bash
python scripts/publish_airfrans_official_release.py --verify-native /path/to/installed/support
```

Extract `profile-truth.tar.gz` into a **different empty directory**, then run:

```bash
python scripts/validate_airfrans_profile_truth.py --release /path/to/installed/profile-truth
```

Native verification loads 1,188 support instances across two ordered case sets
(396 case references, 355 unique cases). Profile verification checks all 355
lossless case arrays and their source-mesh, extractor, runtime and hash bindings.
CI checks committed metadata and the complete artifact-validation receipt.

## Evaluate and submit

Use dataset version `airfrans-native-v1` and evaluator version
`airfrans-scoring-v2`. Select `full`, `scarce` or `aoa_extrapolation`.
The participant workflow binds scoring support to `airfrans-native-support-v1`
and profile truth to the dataset-local `airfrans-native-profile-truth-v1`
manifest, using the exact hashes in the specification.

Follow the [quickstart](QUICKSTART.md) to evaluate complete native coverage,
extract the required predicted profiles, calculate aggregate metrics and
create the schema-v3 package. Validate the generated directory:

```bash
python scripts/validate_submission.py --contributor-stage submissions/airfrans/<new-submission-id>
```

Follow [Submitting a result](../../SUBMITTING.md) to open a pull request
containing only the new result directory. Maintainers add approval metadata
after review. Passing contributor validation does not publish or rank a result.
Code, model weights and complete prediction-field sharing are optional.

## Scoring contract

The scientific implementation is pinned to
`b320575e35d6f61f94ac852fdb084dec080e4ecb`.

The overall weights are curve pressure 15%, curve wall shear 10%, domain
velocity 15%, domain pressure 10%, drag R² 15%, lift R² 10%, and boundary-layer
profile R² 25%. Field caps are 15%, 20%, 12%, and 15%, respectively. Curve
primary errors use native dual lengths; domain primary errors use equal native
nodes. Alternate equal-entity and physical-weight errors are diagnostics.
The four profile stations, two Cartesian quantities, and 1,001 samples per
series are defined by `airfrans-boundary-layer-v1`.
