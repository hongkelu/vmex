# Runtime update: 2026-09-16

The active case uses VMEX main `345f940c56264dc2a66c95b0201330283cc5241d` (0.9.1), including
upstream `ef88081f`. New runs verify all 78 VMEX Python files.
Only `core/implicit.py` (cross-platform callback placement) and
`core/extender.py` (exterior-field source sampling) changed from the previous pin.
Free-boundary solver and adjoint implementations and acceptance thresholds remain intact.

The previous manifest is retained byte-for-byte in `runtime_manifest.5c69195ff104.json`;
`runtime_migration.json` records this update and earlier migrations remain archived.
ESSOS 0.17 supplies the released coil API. Inputs, physical targets, checkpoints
and historical results are preserved. Resuming still requires fresh certification
of the exact saved state and restoration checks. A failure must stop the run;
no prior qualification transfers to this runtime.
