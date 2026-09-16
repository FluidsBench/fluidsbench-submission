# Submission examples

New submissions use schema v3. Start with the synthetic [`v3-template/`](v3-template/) to understand the submission, detailed
methodology record, evaluation
evidence, canonical scoring support, per-case metrics and sufficient statistics, spatial report, optional prediction artifact,
profiles, maintainer validation, and internal SHA-256 bindings. It is a format fixture, not an official dataset result; use the
selected dataset's official support release and specification when submissions open.

For the closed DrivAerML candidate, use the dataset-specific
[`drivaerml-v3-candidate/`](drivaerml-v3-candidate/) configuration and
fail-closed assembler. It contains explicit unresolved release tokens rather
than invented hashes and cannot produce a package until the repository
publishes matching candidate support, evaluator, profile-v10, and profile
ground-truth bindings. Its config-v2 format for schema-v3 packages also
requires the common FluidsBench methodology disclosure with DrivAerML's
dataset-specific required fields; the filled
[`methodology.example.json`](drivaerml-v3-candidate/methodology.example.json)
shows the record's shape but contains illustrative values only.

For the closed HiLiftAeroML candidate, use
[`hiliftaeroml-v3-candidate/`](hiliftaeroml-v3-candidate/). Its direct native
adapter supports all official split labels, preserves disconnected Cp graphs
and the exact five-station velocity support, and reports zero-weight surface
and volume regions. It refuses to build force or overall results when any
case lacks exact complete truth-load coverage. The completed hidden-profile
truth candidate is bound for an explicit maintainer-local dry run, while the
public truth binding remains unpublished and inactive; the template still
retains an unresolved evaluator-revision token until that revision is frozen.

A real dataset specification pins its original public field-bearing files and requires one mapped prediction for every official
entity in every case. It also identifies whether fields are point-, node-, face-, or cell-associated and supplies the authoritative
area, length, volume, or cell-area weights. The guidance distinguishes three-dimensional surfaces and flow domains, two-dimensional
flow domains,
two-dimensional surface manifolds embedded in three dimensions, and one-dimensional boundary curves.

Methods may perform inference in chunks or on another internal representation. Relative-L2 chunks are combined by adding their
numerators and denominators, entity counts, and total weights before calculating the complete-case value; chunk-level L2 values are
never averaged. Surface, two-dimensional surface-manifold, or boundary-curve results report area or length weighting as primary
and an unweighted result as secondary.
Volume or two-dimensional-domain results report an unweighted result as primary and volume or area weighting as secondary. Complete
case values are macro-averaged across the test set.

[`v2-template/`](v2-template/) is retained to interpret previously approved schema-v2 packages. The contributor-stage validator
does not accept it as a new submission.

[`profile-chunk.example.json`](profile-chunk.example.json) demonstrates the compact case/series representation with Cp, Cf, and
velocity profiles. It is intentionally abridged and is not a complete benchmark split.

[`airfrans-profile-extraction/`](airfrans-profile-extraction/) provides the hash-bound official-data reference fixture and the
pinned extractor for AirfRANS extrados velocity profiles. It also demonstrates evaluating native point-ordered NumPy or PyTorch
velocity predictions without serializing a predicted VTU mesh.

For a historical v1 prototype directory that passes the dummy-data validator, inspect the AhmedML
[`transolver`](../submissions/ahmedml/transolver/) submission. It includes all prototype case IDs, chunk checksums, required
pressure stations, optional velocity stations, metadata, and scalar metrics. Do not use it as a real-submission template.

Participants may generate equivalent JSON with Python, MATLAB, Julia, C++, or another language. The schema and final values matter;
the code does not need to follow the example implementation. Public code, a pinned model, a locked environment, and artifact
documentation are optional. If supplied, their URLs, versions, hashes, and licences must satisfy the open reproducibility contract.

The submitter supplies all prediction, result, spatial, and profile data. Sharing the complete prediction fields is optional.
Contributor-stage validation checks the package but does not approve it. Maintainers validate the submitted files and hashes before
approval; FluidsBench does not execute any shared model artifact as part of required approval.
