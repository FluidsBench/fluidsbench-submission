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

## Hardware and compute

Use the same fields for every dataset, in each submitter-performed training
stage's `compute` and in `inference_compute`. Training and inference may use
different hardware. For new packages, include:

| Field | What to report |
| --- | --- |
| `accelerator` | `type`, `vendor`, and exact `model`, e.g. `gpu`, `NVIDIA`, `H200`. Include relevant memory or form-factor variants. Count individual devices, not nodes or racks. |
| `devices_per_job` | Devices allocated to one job. Use `null` if unknown or variable, and explain in the notes. |
| `max_concurrent_device_count` | The campaign's largest simultaneous device allocation, including parallel jobs. It must be at least `devices_per_job` when known. |
| `hardware` / `measurement_notes` | Keep hardware details, the evidence for identity/allocation, what one job processes, and timing boundaries here. |

For example, a campaign running up to ten four-GPU jobs concurrently could
report this fragment (illustrative, not a complete compute record):

```json
{
  "accelerator": {"type": "gpu", "vendor": "NVIDIA", "model": "H200"},
  "devices_per_job": 4,
  "max_concurrent_device_count": 40
}
```

Use `type=gpu|cpu|tpu|other|mixed|unknown`. Vendor and model are nullable:
an unconfirmed GPU is `{"type":"gpu","vendor":null,"model":null}`.
Do not guess from a cluster or node name. For mixed models within a campaign,
use `type=mixed`, `model=null`, and name the models, counts and their device-time
breakdown in the notes; vendor may be set only if common to every device.
For `type=unknown`, both vendor and model must be null. Explain unknowns and
use one consistent device accounting unit across counts and timings, stating
that unit for CPU or other non-GPU work. Different training stages have their
own identities; their concurrent counts must not be added together.

Existing timing requirements still apply: record training campaign elapsed
hours and aggregate device-hours across all reported runs, and inference
campaign elapsed seconds and aggregate device-seconds over the entire selected
official evaluation split. Include the preprocessing/mapping flags. Sum actual
device allocations over time; neither peak concurrency times campaign duration
nor devices per job establishes measured device-time when allocations vary.
State gaps, retries and exclusions. GPU identity alone does not make provisional
allocation estimates verified measurements.

These metadata fields do not affect physics scores or ranking. The new fields
are optional in schema v3 for compatibility with existing packages; missing
identity remains visibly unconfirmed. Do not silently modify accepted records
or immutable releases. Correct hardware through the normal result-revision
and review flow, regenerating affected hashes, validation records and feeds.
Identical training campaigns/checkpoints may reuse training metadata across
submissions; inference measurements must describe each submitted dataset/split.
