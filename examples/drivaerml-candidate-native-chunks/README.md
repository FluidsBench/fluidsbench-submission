# DrivAerML native-chunk reference drivers

These examples cover two different purposes. Neither activates the DrivAerML
contract or represents owner approval.

## One-command synthetic package

`reference_driver.py` is a tiny teaching fixture. It uses no VTP, VTU,
DrivAerML truth, real model, real fixed-area array, or geometric volume array.
It generates its own positive surface areas and equal-weight volume cells.
After installing the candidate requirements, one driver command creates and
schema-validates a complete dummy schema-v3 package, including
`submission.json`:

```bash
python3 -m venv .venv-drivaerml
.venv-drivaerml/bin/pip install -r requirements-drivaerml-evaluator.txt
.venv-drivaerml/bin/python examples/drivaerml-candidate-native-chunks/reference_driver.py \
  --output /tmp/drivaerml-synthetic-native-chunk-demo
```

The generated submission is explicitly synthetic and ineligible. It has no
approval block and belongs to the separate `synthetic-drivaerml-shaped`
namespace. Both its submission declaration and bundled support manifest use
the non-approving `candidate` status. The driver requires the real
`drivaerml` specification to remain exactly
`candidate_scoring_contract` / `owner_review_required` /
`submissions_open: false` before it runs. The driver checks that the generated
submission ID, dataset ID, and scoring-support release ID begin with
`synthetic-` and remain distinct from the real dataset; other identifiers
follow their field-specific schema vocabulary.

The fixture demonstrates ordered two-part and three-part byte transport;
separate native-surface-shaped and native-volume-shaped prediction tables;
complete duplicate-free raw-ID chunk coverage; area-weighted and equal-polygon
surface reductions; equal-cell volume reductions; the DrivAerML per-entity
vector-error rule; coherent five-coefficient force payloads; all four Cp cuts
and all sixteen AutoCFD5 velocity-line payload shapes; and all 27 current
non-score candidate metrics. It also exercises discretization records,
evaluation evidence, and local validation against every applicable repository
JSON Schema. Its transport bytes and support tables are JSON, not VTK, and the
profile and force values are deterministic synthetic examples rather than
validated physical extraction or integration results.

The driver performs those schema checks as part of the one command. The
fictional dataset ID is deliberately not registered with the normal
`scripts/validate_submission.py` dataset registry, so both normal validation
and `--candidate-dry-run` reject this teaching package. Candidate dry-run
validation is reserved for an explicitly registered dataset with an exact
benchmark-owned candidate-manifest binding. This keeps the real closed
DrivAerML contract fail-closed rather than creating a synthetic acceptance
path.

## Real run_1/run_44 pilot

`real_reference_driver.py` is a small fail-closed orchestrator for exactly:

- `run_1`, whose pinned logical VTU has two byte parts; and
- `run_44`, whose pinned logical VTU has three byte parts.

It imports and calls the repository's core and AutoCFD5 candidate evaluator
entry points. Scientific reductions are not copied into the example. Both
cases always use the verified ordered multipart stream. The driver consumes
the immutable source pin, Neil's fixed surface-area arrays, participant
prediction chunks, and the 10 mm velocity mapping/receipt. Volume fields use
one equal weight per native cell; no geometric cell-volume array is required.
It downloads and generates nothing. Run it from a clean Git checkout
so the receipt can bind the entire evaluator to one revision. Copy
`real-case-inputs.example.json`, replace its paths, and run:

```bash
python3 -m venv .venv-drivaerml
.venv-drivaerml/bin/pip install -r requirements-drivaerml-evaluator.txt
.venv-drivaerml/bin/python examples/drivaerml-candidate-native-chunks/real_reference_driver.py \
  --case-inputs /path/to/real-case-inputs.json \
  --native-source-pin benchmark-specs/drivaerml/proposal/native-source-pin.json \
  --dataset-root /path/to/pinned/drivaerml \
  --output /path/to/new/run1-run44-evidence
```

Successful output contains core and diagnostic candidate evidence for both
cases plus a hashed validation receipt; it never contains `submission.json`.

The 209 discrete Cp probes are not part of the DrivAerML submission or scoring
contract. The pilot therefore requires no participant probe values or probe
mapping support; historical probe artifacts elsewhere in the repository are
non-normative research evidence only. Four continuous Cp cuts remain in the
candidate contract and will be derived locally from each participant's native
surface `pMeanTrim` prediction, without a separate participant field. Their
exact immutable native extraction support is pending and is not demonstrated
by this pilot.

Prediction `.npz` files remain bounded inputs: the loader caps an NPY header at
4 KiB, the sum of all archive members' declared uncompressed byte sizes at 512
MiB, and the ZIP central directory at 64 KiB. ZIP `ZIP_STORED` and
`ZIP_DEFLATED` compression are both accepted; the 512 MiB cap is not a
compressed-file-size limit. Split transport parts and inference chunks are
independent concepts.
Never average chunk-local RMSE values; let the evaluator accumulate additive
statistics and reduce once after complete raw-cell coverage.
