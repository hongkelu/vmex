"""Block factorization: probe chunk x row batch, time and difference from the default factors.

    COIL_CASE=qa4-beta python extra/bench_factor.py <run dir with input.run> [vacuum|beta]   (cwd: vmex tree, GPU)
"""
import dataclasses
import functools
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import vmex as vj  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
if len(sys.argv) > 2 and sys.argv[2] == "vacuum":
    inp = dataclasses.replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                              ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None)
cfg = im.make_config(inp, multigrid=True, hot_restart=True)
params = im.params_from_input(inp)
state, mask = im.solve_implicit_with_aux(params, cfg)
state, mask = jax.block_until_ready((state, mask))
fields = im._active_state_fields(cfg)
mn = int(mask.R_cos.shape[1])
print(f"block size {len(fields) * mn}, ns {cfg.resolution.ns}", flush=True)
z = im._dof_projector(cfg, mask)(state)


@functools.partial(jax.jit, static_argnums=(0, 1))
def build(probe, rows, z, params, frozen, mask):
    system = im._raw_block_system(params, cfg, frozen, mask, fields, probe, z_star=z, row_chunk_size=rows)
    return system.factors, system.lower, system.diagonal, system.upper


ref = None
for probe, rows in ((20, 1), (64, 1), (128, 1), (384, 1), (20, 7), (64, 7), (128, 7), (64, 49), (384, 7), (384, 49)):
    try:
        out = jax.block_until_ready(build(probe, rows, z, params, state, mask))  # compile
        t = time.perf_counter()
        for _ in range(2):
            out = jax.block_until_ready(build(probe, rows, z, params, state, mask))
        dt = (time.perf_counter() - t) / 2
        mem = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
        blocks = np.concatenate([np.ravel(np.asarray(a)) for a in out[1:]])
        if ref is None:
            ref = blocks
        diff = np.max(np.abs(blocks - ref)) / np.max(np.abs(ref))
        print(f"probe {probe:4d} rows {rows:3d}: {dt:7.2f} s  peak {mem:5.1f} GiB  max rel diff of blocks {diff:.1e}",
              flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"probe {probe:4d} rows {rows:3d}: failed {str(error)[:200]}", flush=True)
