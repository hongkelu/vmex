# Runtime update: 2026-09-15

The active case now uses VMEX main `0bc787b72917371ab122f76e870e1cf6ab141102`.
The previous runtime manifest is preserved in `runtime_manifest.d6910b428841.json`.
`runtime_migration.json` records the transition. New runs verify all 78
VMEX Python source files before computation.

Input hashes, physical targets, acceptance tolerances, historical results and
frozen source snapshots are unchanged. Historical numerical or physical
qualification does not qualify this runtime. Resuming still requires the
existing fresh certification of the exact saved state and restoration checks;
failed certification must stop the run. Repeat numerical/physical qualification
before treating a new trajectory as validated.
