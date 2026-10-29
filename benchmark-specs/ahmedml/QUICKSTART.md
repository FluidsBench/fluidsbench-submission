# AhmedML participant quickstart

Follow the [resumable participant workflow](../../docs/NATIVE_PARTICIPANT_WORKFLOW.md)
with `dataset=ahmedml`. It handles data/support preparation, complete-split
evaluation, resume, reports and validated packaging. See [Submitting a result](../../SUBMITTING.md) for validation and review.

- Full, Medium, Scarce and Super-scarce each evaluate the same 50 test cases.
- The other four prescribed splits evaluate 100 cases each.
- Both surface and volume predictions are required, in native cell order.
- The prepared cache downloads automatically; no support reconstruction is needed.
- Force coefficients are integrated from predicted pressure and wall shear.

The regional summation fix retains the existing scoring definitions and validation
tolerances. Existing exact public data and prepared cache remain reusable.
See the [data/scoring contract](README.md) and
[methodology and package guide](../../examples/ahmedml-v3-candidate/README.md).
