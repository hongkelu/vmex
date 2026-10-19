"""Merged on-surface virtual casing: far-field quadrature vs singular correction, primal and JVP (8 directions).

    COIL_CASE=qa4-beta python extra/bench_vc_split.py <seed run dir>     (cwd: vmex tree, GPU)
"""
import dataclasses
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
from virtual_casing_jax import integrals as I  # noqa: E402
from virtual_casing_jax.surface_ops import complete_vec_field, cross_prod, dot_prod, resample  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
eq = opt.solve_equilibrium(inp)
n = 48
data0 = vc.surface_field_data_from_state(inp, eq.state, runtime=eq.runtime, nphi=n, ntheta=n)
plan = vc.plan_vc_precision(data0, digits=4, quad_nt=4 * int(inp.nfp) * n, quad_np=2 * n)


def pieces(s, which, src=1024):
    d = vc.surface_field_data_from_state(inp, s, runtime=eq.runtime, nphi=n, ntheta=n)
    f = vc.VirtualCasingExteriorField(d, vc.ExteriorFieldConfig(digits=4, chunk_size=1024, target_chunk_size=8,
                                                                 dtype="float64"))
    v = f._vc
    qt, qp, pd, pi = v._apply_precision(plan, None, None, None, None)
    v._ensure_b_setup(qt, qp, 4)
    st = v._b_setup
    B0 = complete_vec_field(jnp.asarray(f.B_total).reshape((3, v.src_nt, v.src_np)), False, v.half_period, v.nfp,
                            v.src_nt, v.src_np, 0.0)
    Bq = resample(B0, v.nfp_eff * v.src_nt, v.src_np, st.quad_nt, st.quad_np)
    dens = jnp.concatenate([cross_prod(Bq, st.normal), dot_prod(Bq, st.normal)[None]], axis=0)
    if which == "far":
        Xt = I.field_period_target_coords(st.quad_coord, v.trg_nt, v.trg_np, v.nfp_eff)
        area = I.surf_normal_area_elem(st.dX, st.quad_coord)[1]
        return jax.vmap(lambda dn: I.laplace_fxd_u_eval(st.quad_coord, Xt, dn, area, chunk_size=src,
                                                        target_chunk_size=256))(dens)
    return I.laplace_fxd_u_eval_vec_singular(st.quad_coord, st.dX, dens, v.trg_nt, v.trg_np, v.nfp_eff, digits=4,
                                             patch_dim0=pd, chunk_size=src, target_chunk_size=256, patch_idx=pi,
                                             orient=st.orient)


rng = np.random.default_rng(0)
dirs = jax.tree.map(lambda *a: jnp.stack(a), *[jax.tree.map(lambda a: jnp.asarray(1e-6 * rng.normal(size=a.shape)),
                                                             eq.state) for _ in range(8)])


def timed(label, fn, *args):
    jax.block_until_ready(fn(*args))
    t = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    print(f"{label:44s} {(time.perf_counter() - t) / 3 * 1e3:8.1f} ms", flush=True)


for which in ("far", "total"):
    f = lambda s, w=which: pieces(s, w)  # noqa: E731
    timed(f"primal {which}", jax.jit(f), eq.state)
    timed(f"JVP {which} (8 directions)", jax.jit(lambda s, ts, f=f: jax.lax.map(
        lambda t: jax.jvp(f, (s,), (t,))[1], ts, batch_size=8)), eq.state, dirs)
