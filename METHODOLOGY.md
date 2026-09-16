# Methodology record

Every new schema-v3 result includes `submission.json.methodology` with
`format=fluidsbench-method-v1`. The record describes the method that produced
the submitted values; it does not change prediction, scoring, profile, split,
or approval formats.

The common record requires:

- named architecture components, their roles, descriptions, and parameter
  counts, with an exact total and exact submitter-trainable count;
- important hyperparameters and every input feature, including its spatial
  domain and component count;
- every benchmark-required predicted field, whether it is direct or derived,
  and the component responsible for it;
- normalization, preprocessing, and sampling;
- every submitter-performed or upstream training stage, including the fitting
  procedure, runs, seeds, and measured training compute where the submitter
  performed training;
- a raw-file SHA-256 and pre-evaluation selection rule for every checkpoint
  file loaded by a parameterized submitted method; and
- measured end-to-end inference hardware and timing over the complete official
  evaluation split.

Each dataset owns
`benchmark-specs/<dataset-id>/methodology-contract.json`. That small contract
lists the outputs already required by the dataset and their domains and
component counts. It does not add scoring targets. A submission may disclose
additional outputs, but it must cover every field in the selected dataset's
contract. Derived forces, profiles, or case scalars should be marked
`derived_from_model_output` rather than presented as direct network outputs.

Use `record_kind=submitter_reported` for a real submission. In that mode,
parameter counts must be exact, training provenance must cover every component,
each parameterized component must be bound to the exact loaded checkpoint
bytes, and inference compute must be measured. Zero-parameter methods are
valid: report zero parameters and no checkpoints, while describing their
fitting or deterministic procedure truthfully.

The checked-in schema-v1 leaderboard rows are dummy fixtures. They use
`record_kind=prototype_fixture`; nominal parameter counts may be reconstructed
from the displayed millions value, and unavailable training, checkpoint, and
timing details are explicitly marked as not recorded. These records exercise
the interface and must not be cited as descriptions supplied or verified by
the named method authors.

The complete machine-readable definition is `$defs.fluidsbench_methodology`
inside [`schemas/v3/submission.schema.json`](schemas/v3/submission.schema.json).
The filled synthetic
[`examples/v3-template/submission.json`](examples/v3-template/submission.json)
shows the common structure, and the
[`DrivAerML example`](examples/drivaerml-v3-candidate/methodology.example.json)
shows a multi-domain CFD method record.
