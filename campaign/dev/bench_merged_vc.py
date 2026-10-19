"""On-surface virtual casing with J and B.n in one singular pass: value vs the package, primal and JVP time.

    COIL_CASE=qa4-beta python extra/bench_merged_vc.py <seed run dir>     (cwd: vmex tree, GPU)
"""
import dataclasses
import math
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
from virtual_casing_jax.integrals import curl_single_layer_gradient, laplace_fxd_u_eval_vec_singular  # noqa: E402
from virtual_casing_jax.surface_ops import complete_vec_field, cross_prod, dot_prod, resample  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
eq = opt.solve_equilibrium(inp)
n = 48
data0 = vc.surface_field_data_from_state(inp, eq.state, runtime=eq.runtime, nphi=n, ntheta=n)
plan = vc.plan_vc_precision(data0, digits=4, quad_nt=4 * int(inp.nfp) * n, quad_np=2 * n)
surf = lambda s: vc.surface_field_data_from_state(inp, s, runtime=eq.runtime, nphi=n, ntheta=n)  # noqa: E731


def field_of(d):
    return vc.VirtualCasingExteriorField(d, vc.ExteriorFieldConfig(digits=4, chunk_size=1024, target_chunk_size=8,
                                                                    dtype="float64"))


def package(s):
    return vc.plasma_field_on_boundary(surf(s), digits=4, precision=plan)


def merged(s):
    f = field_of(surf(s))
    v = f._vc
    quad_nt, quad_np, patch_dim0, patch_idx = v._apply_precision(plan, None, None, None, None)
    v._ensure_b_setup(quad_nt, quad_np, 4)
    setup = v._b_setup
    B0 = jnp.asarray(f.B_total).reshape((3, v.src_nt, v.src_np))
    dtheta = math.pi * (1.0 / (v.nfp * v.trg_nt * 2) - 1.0 / (v.nfp * v.src_nt * 2)) if v.half_period else 0.0
    B0c = complete_vec_field(B0, False, v.half_period, v.nfp, v.src_nt, v.src_np, dtheta)
    Bq = resample(B0c, v.nfp_eff * v.src_nt, v.src_np, setup.quad_nt, setup.quad_np)
    density = jnp.concatenate([cross_prod(Bq, setup.normal), dot_prod(Bq, setup.normal)[None]], axis=0)
    grad = laplace_fxd_u_eval_vec_singular(setup.quad_coord, setup.dX, density, v.trg_nt, v.trg_np, v.nfp_eff,
                                           digits=4, patch_dim0=patch_dim0, chunk_size=1024, target_chunk_size=None,
                                           patch_idx=patch_idx, orient=setup.orient)
    grad = jnp.asarray(grad).reshape((4, 3, v.trg_nt, v.trg_np))
    B_on = resample(B0c, v.nfp_eff * v.src_nt, v.src_np, v.nfp_eff * v.trg_nt, v.trg_np)[:, : v.trg_nt, :]
    return -(curl_single_layer_gradient(grad[:3]) + grad[3]) + 0.5 * B_on


rng = np.random.default_rng(0)
dirs = jax.tree.map(lambda *a: jnp.stack(a), *[jax.tree.map(lambda a: jnp.asarray(1e-6 * rng.normal(size=a.shape)),
                                                             eq.state) for _ in range(8)])


def timed(label, fn, *args):
    out = jax.block_until_ready(fn(*args))
    t = time.perf_counter()
    for _ in range(3):
        out = jax.block_until_ready(fn(*args))
    print(f"{label:40s} {(time.perf_counter() - t) / 3 * 1e3:8.1f} ms", flush=True)
    return np.asarray(out)


jvp = lambda f: jax.jit(lambda s, ts: jax.lax.map(lambda t: jax.jvp(f, (s,), (t,))[1], ts, batch_size=8))  # noqa: E731
a = timed("package primal", jax.jit(package), eq.state)
b = timed("merged primal", jax.jit(merged), eq.state)
print(f"  max rel diff of B_plasma {np.max(np.abs(a - b)) / np.max(np.abs(a)):.2e}", flush=True)
da = timed("package JVP (8 directions)", jvp(package), eq.state, dirs)
db = timed("merged JVP (8 directions)", jvp(merged), eq.state, dirs)
print(f"  max rel diff of the JVPs {np.max(np.abs(da - db)) / np.max(np.abs(da)):.2e}", flush=True)
