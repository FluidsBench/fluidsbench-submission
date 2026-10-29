# DrivAerML scientific contract notes

The [submission specification](../submission-spec.json) defines the current
contract. DrivAerML accepts the eight prescribed splits and either
`surface_and_volume` or `surface_only` predictions. Use the [participant guide](../PARTICIPANT_GUIDE.md)
for evaluation and package generation.

## Native fields and identity

Every native surface polygon and reconstructed volume cell retains its original
VTK CellData order. Surface errors use the published same-order polygon areas;
volume errors use one equal weight per native cell. Equal-polygon surface
errors are reported separately. Geometric volume weights are not used.
The source pin identifies all 484 cases, including every volume part.

## Fields, forces and reductions

The scored fields are surface pressure, surface wall shear, volume velocity and
volume pressure. The field group has total weight 0.50, force group 0.25, and
profile group 0.25. The four field caps are 15%, 20%, 12% and 15% respectively.
The bounded field score is `clip(100 * (1 - error/cap), 0, 100)`.
Force and profile R² scores use `100 * clip(R², 0, 1)`.

Predicted forces are integrated from native pressure and wall-shear predictions.
The same integration defines truth, with fixed pressure reference, shear sign,
reference dimensions and moment origin. The force replay covers all 484 cases
at the declared coefficient tolerance. `CmPitch=(Clf-Clr)/2`; axle lift and
required dimensional errors remain report-only. See the [force definition](../submission-spec.json).

## Profiles and validity

The [profile scoring contract](../profile-scoring-v2.json) uses normalized
trapezoidal arc-length weights within each profile and equal total weight per
case/profile. There are 16 fixed velocity lines sampled at 10 mm and four
continuous Cp cuts. Unsupported gaps are never bridged. Velocity values use
native-cell truth; no smoothing or interpolation changes the score.

Geometry-relative profiles have zero scoring weight. Surface-only packages
omit volume outputs and retain zero for unavailable volume/velocity components,
without renormalizing weights; their maximum overall score is 60.

## Scientific evidence

The [evidence index](../evidence/README.md) identifies native-volume audits,
force replays, profile support and owner approval. Published source, geometry,
truth and support hashes remain authoritative. Optional convergence or
sensitivity experiments are not submission-opening requirements; the
[owner opening decision](../OWNER_OPENING_DECISION.md) records that scope.
