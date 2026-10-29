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
  procedure, runs, seeds, and training compute with its measurement basis where the submitter
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

## Complete-case throughput and timing protocol

The inference headline is **complete cases/s = case_count /
campaign_wall_time_seconds**. One case is one benchmark geometry and condition,
with every output and every point required by its declared prediction scope.
All chunk passes and all model components needed for that case count together.
A query batch, subsampled mesh, surface-only result, and full surface-plus-volume
result are not interchangeable workloads. Compare the same dataset, split,
output support, precision, protocol, hardware and allocation.

The reciprocal is campaign-average seconds/case, not single-request latency
when work runs in parallel. Device-seconds/case uses the independently summed
allocation. Neither number changes physics scores or rank.

New measurements can declare `timing_protocol=fluidsbench-complete-case-v1`:

1. Load checkpoints, initialize the runtime and compile before measurement.
   Report that setup separately in `setup_wall_time_seconds` when available.
2. Declare `execution.precision`, batch size and unit per model invocation,
   `warmup_cases` (zero is allowed), and exact framework/runtime versions in
   `execution.software`. Warm up outside the measurement. Do not tune on test
   labels. Explain mixed precision, variable batches and cache policy in notes.
3. Start the campaign timer with the benchmark's unprocessed case inputs in host
   memory, before any model-specific preprocessing or host-to-device transfer.
   Include normalization, features, every chunk/forward pass, inverse transforms,
   mapping to the required benchmark support, and transfer of the complete
   predictions back to host memory. Synchronize accelerators before stopping.
   Input-file reads, output-file writes, metric evaluation and setup are outside
   this boundary. Report those separately in notes if measured; do not omit work
   by caching model-specific preprocessing between repetitions.
4. Measure the complete official split for each repetition. Record every run in
   `campaign_runs` with `wall_time_seconds` and its paired
   `aggregate_device_time_seconds`. Sum allocated device time only within the
   measured interval, including idle allocated devices. Record concurrency and
   explain idle gaps or retries. A failed/incomplete run cannot supply throughput;
   disclose failures separately. `case_count` is unique cases **per run**.
5. Sort runs by wall time, preserving input order for ties, and select index
   `floor((run_count - 1) / 2)` (the lower median). Copy both times from that same
   run into the top-level campaign totals. One repetition is allowed and remains
   visible; never combine the fastest wall time with another run's device time.

The [synthetic v1 example](examples/compute-v1/methodology.example.json) shows a
complete method record with three paired repetitions. Replace every illustrative
value with your own evidence.

The schema requires complete metadata when this protocol is declared. It cannot
prove that a timer or prediction actually followed the declared boundaries;
retain logs, commands and measurements for review. `hardware_identity_basis`
distinguishes `machine_recorded`, `owner_confirmed`, `submitter_reported` and
`unknown`. This is provenance, not independent verification. Report exact device
variants only when known. `peak_device_memory_bytes` is optional peak usage on
one device (state allocated/resident measurement method in notes), not total
cluster memory; unknown remains null. `max_concurrent_jobs` is optional and
separate from devices per job or maximum concurrent devices.

`timing_protocol` is optional. When omitted or set to `legacy_reported`,
throughput can be derived from recorded campaign totals with the reported scope
shown; this does not certify compliance with complete-case-v1. Partial chunk-loop
measurements must not be substituted for full campaign time.

## Training cost and estimates

Add `measurement_basis=measured|estimated|unknown` to each submitter training
compute block. Measured means summed allocation records; a calendar envelope
multiplied by a device count is an estimate, even if hardware is confirmed.
A missing measurement basis is shown as unspecified. The website labels estimates
and unspecified totals and only plots totals explicitly measured for every
submitter stage. Unknown/missing costs are never zero.

Use `cost_scope=final_training|hyperparameter_search|mixed|other|unknown` to
separate training the selected checkpoints from search/tuning. Include all
reported runs once; do not add a final run twice if already covered by search.
Upstream training stays in its own stages and is excluded from submitter totals,
with that exclusion visible. Costs for different GPU models are not equivalent.

The real [GB200 example](tests/fixtures/hilift-gb200-compute.json) is a regression
fixture derived from Neil Ashton's AoA 4 candidate and owner confirmation. It
retains package-reported inference totals, FP32 provenance for the matching
archived predictions, and estimated training totals. It is not a new result,
release, approval, or v1-compliant timing claim. The raw scheduler ledger was
not located in the local audit. Original submissions and releases remain unchanged.
