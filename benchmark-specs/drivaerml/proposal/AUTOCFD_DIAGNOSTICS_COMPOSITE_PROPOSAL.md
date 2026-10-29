# DrivAerML profile and composite scoring

Use [profile-scoring-v2.json](../profile-scoring-v2.json) for the exact scoring
rules and [submission-spec.json](../submission-spec.json) for component weights,
metric bindings and prediction scopes.

Sixteen AutoCFD5 velocity lines use deterministic 10 mm support. The four
continuous Cp cuts cover upper-body centreline, underbody centreline, sidewall
at `z=0.15 m`, and the front-left wheel. Geometry, station identities and native
truth are bound by the published support.

Cp and velocity profiles use normalized trapezoidal arc-length weighting.
Each case/profile has equal total weight; unsupported gaps are never bridged,
and no origin-to-first-point edge is added. Profile R² is bounded to [0, 1]
for scoring. Fixed velocity profiles contribute 0.15 and continuous Cp cuts
0.10 to the overall score. Relative profiles and discrete probes do not
contribute to the score.

The other groups are native-field errors (0.50 total) and field-integrated
force coefficients (0.25 total). Surface-only submissions keep the same
weights and score zero on unavailable volume components, giving a maximum of
60. Participants must use the prescribed splits and full required support.

The [scientific notes](SCIENTIFIC_CONTRACT.md) and [evidence index](../evidence/README.md)
explain native associations, force integration and support validity.
