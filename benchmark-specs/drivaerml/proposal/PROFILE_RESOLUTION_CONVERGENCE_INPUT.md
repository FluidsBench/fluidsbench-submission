# DrivAerML profile-resolution convergence input

This is a candidate evidence format. It does not activate scoring or open
submissions.

> Current-state note (2026-08-28): this checker and the proposed 1/2/5/10 mm
> campaign are retained as optional research tooling, not as current activation
> requirements. The dataset owner accepted the published deterministic 10 mm
> support, explicit gaps, and numerical tolerances without claiming that the
> superseded campaign ran. See
> [`../evidence/owner-scientific-approval-2026-08-28.json`](../evidence/owner-scientific-approval-2026-08-28.json)
> and [`../ACTIVATION_CHECKLIST.md`](../ACTIVATION_CHECKLIST.md). The
> owner-review language below describes the historical checker contract.

Run the checker with:

```bash
.venv/bin/python scripts/check_drivaerml_profile_resolution_convergence.py \
  --input study.json \
  --output evidence.json
```

The input uses schema
`drivaerml-profile-resolution-convergence-input-v1`. Its `losses` array is
indexed as `[method][case][profile][spacing]`; the declared orders must match
those dimensions exactly. `profile_order` is exactly `V1`-`V6`, `U1`-`U6`,
`L1`, `R1`-`R3`, and `spacings_mm` is exactly `[1, 2, 5, 10]`.

`method_set.methods` declares an ordered physics null, nearest-training-design
control, and genuine trained model/checkpoint methods. Each declaration pins
its prediction artifact by SHA-256, and the canonical JSON hash of the whole
ordered method array is supplied as `method_set.sha256`. At least three
distinct trained model/checkpoint pairs with distinct prediction-artifact
hashes are required for eligibility. The checker validates those declarations
structurally; scientific authenticity and approval still require owner review.

The checker writes complete aggregate, case-macro, and case-line comparisons
for 2, 5, and 10 mm against 1 mm, plus Kendall tau-b method-order evidence.
Every one of the 2, 5, and 10 mm comparison blocks must pass both the loss
thresholds and method-order gate before the retained 10 mm grid is eligible for
owner review; a passing 10 mm block cannot hide a failed 2 or 5 mm block. It
returns exit code 0 for an eligible candidate study, 1 for valid but ineligible
evidence, and 2 for malformed or contract-drifted input. Exit code 0 requires
the exact ordered 484-case release and a hash-bound owner validity mask. Because
that mask is not yet published, the present candidate always returns 1 even if
every numerical gate passes. A future version may return 0 after binding the
mask, but that result will still be non-activating pending owner approval. A
reduced case list is a pilot and also returns exit code 1.

Minimal top-level shape:

```json
{
  "schema": "drivaerml-profile-resolution-convergence-input-v1",
  "schema_version": 1,
  "study_id": "immutable-study-id",
  "method_set": {
    "pinned_before_study": true,
    "sha256": "<canonical ordered methods SHA-256>",
    "methods": ["<method objects>"]
  },
  "case_order": ["run_1", "run_44"],
  "profile_order": ["V1", "V2", "V3", "V4", "V5", "V6", "U1", "U2", "U3", "U4", "U5", "U6", "L1", "R1", "R2", "R3"],
  "spacings_mm": [1, 2, 5, 10],
  "losses": ["<method x case x profile x spacing finite nonnegative losses>"]
}
```

Use `reference.drivaerml.profile_convergence.method_set_sha256(methods)` to
derive the method-set pin; do not hand-edit it after losses have been produced.

All four grids must use one owner-published 1 mm master validity mask, with the
2, 5, and 10 mm masks obtained only by strides 2, 5, and 10. Only
`inside_morphed_solid` and `outside_released_fluid_domain` may be owner
exclusions. An unmapped non-excluded coordinate is an unresolved support
failure and cannot contribute to an owner-review-eligible convergence result.
Excluded rows remain explicit, and a loss may use only edges whose two
endpoints are included and mapped; it must never bridge a gap. The candidate
mask binding is still pending, so current invalid pilot rows cannot be treated
as approved exclusions.

## Construct losses from native predictions

Maintainers should normally construct the loss tensor from the actual artifacts
rather than authoring it by hand:

```bash
.venv/bin/python \
  scripts/evaluate_drivaerml_profile_resolution_from_predictions.py \
  --input native-prediction-study.json \
  --mappings-root /path/to/validated/velocity-assignment-receipts \
  --dataset-root /path/to/pinned/drivaerml \
  --output profile-resolution-evidence.json
```

The input schema is
`drivaerml-profile-resolution-native-prediction-study-v1`. It contains the same
ordered `method_set` and `case_order`, plus one ordered native prediction-chunk
manifest for every method and case. Relative manifest paths resolve from the
study file. The command strictly replays the 1, 2, 5, and 10 mm containing-cell
artifacts, verifies every prediction chunk and exact raw-cell coverage, verifies
the pinned multipart VTU bytes, streams the complete native `CellData
UMeanTrim[3]` truth payload, and retains only mapped values in memory. It fails
before opening the study, prediction manifests, mappings, or native source
unless the runtime is exactly Python 3.12.13 and NumPy 2.2.6, then records that
runtime, execution limits, source-file hashes, and Git state in the evidence.
VTK is not required by this command because it streams the inline XML payload
directly.

The exact public case order is the immutable 484-case native-source order.
Pilot studies may use a reduced ordered subset for implementation checks, but
their output is explicitly ineligible for owner review. Declarations do not
prove that trained checkpoints are scientifically genuine; model/checkpoint
provenance and owner attestation remain required inputs.

The current real-data command is bounded in memory but is not yet a resumable
all-case executor: it runs the ordered cases in one process and writes the
aggregate evidence only after the final case. A late preemption therefore
requires a complete rerun. Before the genuine all-484, five-method study is
treated as production-hardened, add immutable per-case worker receipts and a
strict restartable aggregate replay; this remains execution hardening, not
completed scientific evidence.
