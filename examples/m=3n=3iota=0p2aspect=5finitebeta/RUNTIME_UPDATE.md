# Runtime update: 2026-09-15

The active case uses VMEX main `5c69195ff1045a6c7b9acc0708724c3b04b967ab` (0.9.0), including
upstream `8db9c825`. New runs verify all 78 VMEX Python files.
The optimizer default now accepts only `FSQ / forward_ftol <= 100`.
Only `core/optimize.py` changed relative to the previous active runtime.
Free-boundary solver and adjoint implementations are unchanged.

The prior manifest is preserved byte-for-byte in `runtime_manifest.0bc787b72917.json`.
`runtime_migration.json` records this transition; the earlier migration record
and original d6910b428841 manifest are also retained.

Input hashes, physical targets, case acceptance tolerances, historical results
and frozen source snapshots are unchanged. Historical numerical or physical
qualification does not qualify this runtime. Resume still requires the existing
fresh certification of the exact saved state and restoration checks; a failure
must stop the run. No acceptance threshold was relaxed. Repeat numerical and
physical qualification before treating a new trajectory as validated.
