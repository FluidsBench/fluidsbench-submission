# WindsorML participant quickstart

Follow the [resumable participant workflow](../../docs/NATIVE_PARTICIPANT_WORKFLOW.md)
with `dataset=windsorml`. It handles data/support preparation, complete-split
evaluation, resume, reports and validated packaging. See [Submitting a result](../../SUBMITTING.md) for validation and review.

- Use the exact published-case intersection in the repository's prescribed split.
  Full has 35 cases; the other split counts are listed in the [contract](README.md).
- Both surface and volume predictions are required. Surface IDs index **points**;
  volume IDs index **cells**.
- Profile support for all 233 distinct test cases is bundled in the checkout.
  Published surface dual areas are downloaded with the public case data.
- Forces are integrated from the predicted native coefficient fields.
- Predict the raw published volume `pressureavg` values. The published pressure
  convention is retained while its physical units/reference remain under audit.

See the [methodology and package guide](../../examples/windsorml-v3-candidate/README.md).
