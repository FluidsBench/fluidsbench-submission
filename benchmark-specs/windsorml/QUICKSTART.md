# WindsorML participant quickstart

Follow the [resumable participant workflow](../../docs/NATIVE_PARTICIPANT_WORKFLOW.md)
with `dataset=windsorml`. It handles data/support preparation, complete-split
evaluation, resume, reports and validated packaging. See [Submitting a result](../../SUBMITTING.md) for validation and review.

- Use the [WindsorML v2 manifest](https://huggingface.co/datasets/neashton/windsorml/blob/ba7e66955650b2ff0482517f256d84c1eda815e7/splits/v2/manifest.json) for model training, validation and evaluation.
  Full has 35 test cases. See the [contract](README.md) for the other split counts, and the
  [erratum](https://huggingface.co/datasets/neashton/windsorml/blob/ba7e66955650b2ff0482517f256d84c1eda815e7/splits/v2/README.md) if you trained or tuned against earlier assignments.
- Both surface and volume predictions are required. Surface IDs index **points**;
  volume IDs index **cells**.
- Profile support for all 230 distinct test cases is bundled in the checkout.
  Published surface dual areas are downloaded with the public case data.
- Forces are integrated from the predicted native coefficient fields.
- Predict the raw published volume `pressureavg` values. The published pressure
  convention is retained while its physical units/reference remain under audit.

See the [methodology and package guide](../../examples/windsorml-v3-candidate/README.md).

Explore the [dataset leaderboard preview](https://fluidsbench.org/leaderboard-preview/?dataset=windsorml),
which contains no submitted results. See [review and publication](../../SUBMITTING.md#see-the-leaderboard-and-follow-publication)
for how an approved result reaches an official release.
