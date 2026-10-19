"""On-surface virtual casing: JVP of B_plasma along 8 state directions, source x target chunk sizes.

    python extra/bench_vc.py <run dir with input.run>     (cwd: vmex tree, GPU)
"""
import dataclasses
import functools
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
eq = opt.solve_equilibrium(inp)
n = 48
data0 = vc.surface_field_data_from_state(inp, eq.state, runtime=eq.runtime, nphi=n, ntheta=n)
precision = vc.plan_vc_precision(data0, digits=4, quad_nt=4 * int(inp.nfp) * n, quad_np=2 * n)
print(f"quadrature {precision.quad_nt} x {precision.quad_np}", flush=True)
rng = np.random.default_rng(0)
tangents = [jax.tree.map(lambda a: jnp.asarray(1e-4 * rng.normal(size=a.shape)), eq.state) for _ in range(8)]
batch = jax.tree.map(lambda *a: jnp.stack(a), *tangents)


@functools.partial(jax.jit, static_argnums=(0, 1))
def run(src, trg, state, batch):
    def f(s):
        data = vc.surface_field_data_from_state(inp, s, runtime=eq.runtime, nphi=n, ntheta=n)
        cfg = vc.ExteriorFieldConfig(digits=4, chunk_size=src, target_chunk_size=trg, dtype="float64")
        field = vc.VirtualCasingExteriorField(data, cfg)
        return field._vc.compute_internal_B(field.B_total, digits=4, chunk_size=src, target_chunk_size=trg,
                                            quad_nt=precision.quad_nt, quad_np=precision.quad_np,
                                            patch_dim0=precision.patch_dim0, patch_idx=precision.patch_idx)
    _, linear = jax.linearize(f, state)
    return jax.lax.map(linear, batch)


ref = None
for src, trg in ((1024, 256), (1024, 2304), (4096, 256), (4096, 1152), (9216, 256), (9216, 1152), (36864, 64),
                 (36864, 256)):
    try:
        out = jax.block_until_ready(run(src, trg, eq.state, batch))
        t = time.perf_counter()
        for _ in range(2):
            out = jax.block_until_ready(run(src, trg, eq.state, batch))
        dt = (time.perf_counter() - t) / 2
        mem = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
        out = np.asarray(out)
        ref = out if ref is None else ref
        print(f"src {src:6d} trg {trg:5d}: {dt:7.3f} s per 8 directions, peak {mem:5.1f} GiB, "
              f"max rel diff {np.max(np.abs(out - ref)) / np.max(np.abs(ref)):.1e}", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"src {src} trg {trg}: failed {str(error)[:160]}", flush=True)
