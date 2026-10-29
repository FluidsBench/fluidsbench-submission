# AirfRANS participant templates

Follow the [dev quickstart](../../benchmark-specs/airfrans/QUICKSTART.md).
Copy `entry.template.json` and `package-config.template.json` into your entry.
Replace every metadata placeholder with an accurate record of your model and
inference. Do not edit release bindings or reference support; the commands
resolve these from the approved dev contract.

The two `spatial_discretization` objects and the separate per-case JSONL must
describe actual training/inference sampling, direct output counts and mappings.
Their schemas and worked historical examples are linked from the quickstart.
The packager preserves these records instead of assuming that exported native
arrays imply native model inputs or one forward pass per case.
