# DrivAerNet++ release preparation

Status: **prepared on dev; official intake closed**.

`approval-record.json` records Mohamed Elrefaie's scientific confirmations and
candidate-release readiness statement from PR #61. `publication-validation.json`
records a complete read and validation of all 24 hosted support archives, all
1,154 native cases and their three report-only profile stations. The candidate
manifest, archive bytes, scores and existing unranked dev result are unchanged.

The validator checks full archive hashes, each NPZ and array digest, the retained
record/manifest/source-identity chain, exact split coverage, finite positive
weights and ordered native profile coordinates. Its normalized digest covers the
case ID, native point count, all array names/dtypes/shapes and raw little-endian
array bytes, in official split order. Point IDs are `range(native_point_count)`.
No pressure points are masked. `hosted-assets.json` is a timestamped GitHub API
snapshot; its asset sizes, URLs and GitHub SHA-256 digests match the distribution.

Reproduce from the repository root using an existing participant cache:

```bash
python benchmark-specs/drivaernetplusplus/tools/validate_publication.py \
  --support-root support/drivaernetplusplus --output /tmp/dnpp-publication-validation.json
```

This reads about 5.49 GB of compressed archives in place, processes one case at a
time and needs no GPU, original meshes or model checkpoint. If the cache is empty,
first use the normal participant `fetch-support` command. The validator uses
NumPy to decode and hash existing arrays; it does not regenerate numerical weights.
Its actual Python/NumPy/platform versions are retained in the receipt.

The fast check below validates the committed receipt's input hashes and closed
contract. It does **not** claim to reread the archives:

```bash
python benchmark-specs/drivaernetplusplus/tools/validate_publication.py --check-receipt
```

## Separate official activation

1. Select final release identities and URLs. Preserve the native support arrays;
   no scientific regeneration is required for the approved definition.
2. Update the manifest and the dataset-specific loader, evaluator and packaging
   contracts together; their current identities explicitly require a candidate.
3. Revalidate the final manifest and hosted assets. Record its exact manifest hash,
   owner approval and publication-validation receipt in the official specification.
   This candidate receipt must not be copied over with a different manifest hash.
4. Update the website audit, availability and profile-release bindings, and run
   the participant-to-leaderboard trial before explicitly opening intake. The
   existing dev result remains unranked until separately accepted as an official result.

Pressure is the complete v1 task. Wall shear and volume are outside its scope.
Source-mesh reconstruction, the owner's new analysis across 8,150 pressure files,
and model inference were not rerun by this artifact-validation step.
