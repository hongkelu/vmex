# Runtime environment of the three-term qualification runs (Perlmutter, 2026-10-22)

The single runtime artifact for this program (brief, package E). Raw data: `raw/env/` (`pip_freeze.txt`,
`sources.json`, `manifest_example.json`).

| component | version / source |
|---|---|
| Python | 3.12, conda env `uwplasma-env` |
| JAX / jaxlib / CUDA plugin / PJRT | 0.12.0.1 / 0.12.0.1 / jax-cuda13-plugin 0.12.0.1 / jax-cuda13-pjrt 0.12.0.1 (wheels) |
| ESSOS | 0.21.0 (wheel) |
| SOLVAX | 0.31.0 (wheel) |
| DKX | 2.11.0 (wheel) |
| Equinox / Lineax | 0.13.8 / 0.1.1 |
| NumPy / SciPy / netCDF4 | 2.5.4 / 1.18.1 / 1.7.5 |
| simsopt | 1.11.1 (reference only) |
| booz_xform_jax | 0.5.0 |
| GPUs | NVIDIA A100-SXM4-80GB, driver 580.178.04 (UUIDs per run in each `manifest.json`) |
| DESC (references) | separate env `desc-latest`, DESC 0.17.3+35.gdbf5fc564 (local checkout `refs/DESC-latest`, commit dbf5fc564), JAX 0.6.2 |

**vmex is never imported from the installed distribution.** The env carries an older editable vmex (0.11.10, from
`vmex-env`); every run sets `PYTHONPATH` to an immutable pinned worktree `vmex-pin-<commit>`, and its
`manifest.json` records the git revision, the dirty-diff hash, the sha256 of the probe/driver/library files,
the versions above, and the GPU inventory (with `CUDA_VISIBLE_DEVICES`).

Runtime settings:
- `XLA_PYTHON_CLIENT_PREALLOCATE=true` with `MEM_FRACTION=0.9`: JAX 0.12 autotuning runs out of memory otherwise.
  The fresh-process checkpoint test runs with preallocation off, so its child process can claim the GPU.
- One process per GPU, pinned with `CUDA_VISIBLE_DEVICES` before JAX is imported (`onnode.sh`).
- Precision: float64 throughout (`JAX_ENABLE_X64=1`).

The DESC GPU runs need an unset `LD_LIBRARY_PATH`: the module cuDNN 9.5 conflicts with the wheel's 9.8.
