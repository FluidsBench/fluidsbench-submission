# Launch configuration

The authoritative repositories are `FluidsBench/fluidsbench-submission` and
`FluidsBench/fluidsbench`. Participant pull requests target `main`.

AirfRANS Full, Scarce and AoA extrapolation, AhmedML, WindsorML, DrivAerML and
DrivAerNet++ accept packages through contributor validation and maintainer approval.
AirfRANS Reynolds remains closed. HiLiftAeroML, BlendedNet, Rotor37 and VKI-LS59
are coming soon. The latest HiLiftAeroML scientific work is retained.

The initial source has no inherited Git history or trial submissions. The
leaderboard remains empty and unreleased until an explicit official publication.
The cutoff and planned first release are 5 December 2026, 23:59:59 Anywhere on Earth
(6 December 2026, 11:59:59 UTC); the clock does not publish results automatically.

Scientific payload bytes and their checksums are immutable. `reference/hosting.py`
resolves original archive locations to checksum-identical FluidsBench release
mirrors. Original URLs and revision IDs inside immutable evidence are provenance,
not a requirement to clone the old repository history. Prepared metadata, native
truth, scoring rules and owner approvals retain their scientific identities.

Before publication, run the complete tests, scoring-support checks, contributor
validation, feed checks, participant runtime checks and website compatibility CI.
Review and publication procedures are in [MAINTAINERS.md](MAINTAINERS.md).
