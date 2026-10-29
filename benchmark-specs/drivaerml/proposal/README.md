# DrivAerML scientific sources

DrivAerML is open for submissions. Start with the [quickstart](../QUICKSTART.md).
The [submission specification](../submission-spec.json) and
[profile scoring contract](../profile-scoring-v2.json) are authoritative.

This directory stores the source identities, geometry and audit inputs used to
construct the published support. Their paths and bytes are retained because
support manifests and scientific evidence bind them by SHA-256.

- [Scientific contract notes](SCIENTIFIC_CONTRACT.md): fields, reductions and forces.
- [Profile and composite scoring](AUTOCFD_DIAGNOSTICS_COMPOSITE_PROPOSAL.md).
- [Native profile truth export](NATIVE_PROFILE_TRUTH_EXPORT.md).
- [Optional convergence tooling](PROFILE_RESOLUTION_CONVERGENCE_INPUT.md).
- `native-source-pin.json`: canonical native-file identities for all 484 cases.
- `owner-published-splits.json` and `split-index-candidates/`: ordered split sources.
- `autocfd_velocity_lines_nominal.csv` and `autocfd_velocity_samples_10mm.csv`: velocity sampling geometry.
- `force-definition-audit-all484.json` and `run-1-force-axle-replay-summary.json`: coefficient closure and native-force evidence.

Discrete Cp probe inputs are unscored research artifacts. Participant scoring
uses the four continuous Cp cuts; probe files are not an intake requirement.
