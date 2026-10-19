"""Prototype: DKX's per-surface kinetic <j.B> as one lax.map over stacked surface templates, against dkx's own loop.

    COIL_CASE=qi6-beta python <repo>/campaign/checks/dkx_map_proto.py <seed dir with input.run>     (cwd: vmex tree, GPU)

Prints compile and run times of value and gradient for both, and their largest relative differences.
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt  # noqa: E402
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402


def mapped_kinetic(km, state, rt):
    """``km._kinetic`` with the surface loop as ``lax.map`` (same per-surface arithmetic)."""
    from booz_xform_jax.jax_api import booz_xform_jax_impl
    from vmex.core.boozer_tables import boozer_input_tables
    from dkx.bootstrap import PARALLEL_CURRENT
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve
    from dkx.workflows.geometry_adapters import PSI_A_HAT_LEAVES, kinetic_operator_on_boozer_surface

    setup = rt.setup
    s_full = np.asarray(setup.s_full)
    nfp = int(rt.resolution.nfp)
    rows, s_rows, templates = km._setup(int(s_full.size))
    psi_a = jnp.abs((s_full[1] - s_full[0]) * jnp.sum(jnp.asarray(setup.phipf)[1:]))
    tables0 = boozer_input_tables(state, rt, int(rows[0]))
    xm, xn = np.asarray(tables0["xm"]), np.asarray(tables0["xn"])
    plan = km._plan(xm, xn, nfp)
    stacked = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *templates)

    def one(args):
        row, template = args
        tables = boozer_input_tables(state, rt, row)
        booz = booz_xform_jax_impl(
            **{k: jnp.asarray(tables[k])[None] for k in ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc",
                                                         "iota")},
            xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm), xn_nyq=jnp.asarray(xn),
            constants=plan.constants, grids=plan.grids, plan=plan)
        g, i, iota = booz["bvco_b"][0], booz["buco_b"][0], booz["iota_b"][0]
        operator = kinetic_operator_on_boozer_surface(
            dataclasses.replace(template, **{leaf: getattr(template, leaf) / psi_a for leaf in PSI_A_HAT_LEAVES}),
            bmnc_b=booz["bmnc_b"][0], ixm_b=np.asarray(plan.grids.xm_b), ixn_b=np.asarray(plan.grids.xn_b), nfp=nfp,
            iota=iota, g_hat=g, i_hat=i, signgs=int(np.sign(setup.signgs)))
        solved = solve(operator, operator.rhs(), method="auto", tol=km.tol, differentiable=True,
                       tier1_keep_lowest=operator.n_xi, emit=None)
        moments = profile_moments_from_operator(operator, solved.x.reshape(-1))
        return moments["FSABjHat"].reshape(()) * PARALLEL_CURRENT

    return s_rows, jax.lax.map(one, (jnp.asarray(rows), stacked))


seed = Path(sys.argv[1])
inp = dataclasses.replace(vj.VmecInput.from_file(seed / "input.run"), lfreeb=False)
eq = opt.solve_equilibrium(inp)
km = tt.dkx_kinetic(inp)
rt = tt.on_grid(eq.runtime)
km._kinetic(eq.state, rt)  # host-side templates, as the driver does before tracing
print(f"{len(km.rows(int(np.asarray(rt.setup.s_full).size))[0])} DKX rows", flush=True)
grid = np.asarray(rt.setup.s_full)


def runtime_for(state):
    return rt


variants = dict(loop=lambda st: km._kinetic(st, rt)[1], mapped=lambda st: mapped_kinetic(km, st, rt)[1])
results = {}
for name, fun in variants.items():
    f = jax.jit(fun)
    g = jax.jit(jax.grad(lambda st: jnp.sum(fun(st) ** 2) * 1e-12))
    t = time.perf_counter(); v = np.asarray(f(eq.state)); tc = time.perf_counter() - t
    t = time.perf_counter(); v = np.asarray(f(eq.state)); tr = time.perf_counter() - t
    t = time.perf_counter(); d = jax.block_until_ready(g(eq.state)); gc = time.perf_counter() - t
    t = time.perf_counter(); d = jax.block_until_ready(g(eq.state)); gr = time.perf_counter() - t
    mem = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
    results[name] = (v, np.asarray(d.R_cos))
    print(f"{name:7s}: value compile+run {tc:7.1f} s, run {tr:6.2f} s | gradient compile+run {gc:7.1f} s, run {gr:6.2f} s"
          f" | peak {mem:.1f} GiB", flush=True)
(v0, d0), (v1, d1) = results["loop"], results["mapped"]
print(f"max rel diff: value {np.abs(v1 - v0).max() / np.abs(v0).max():.2e}, gradient(R_cos) "
      f"{np.abs(d1 - d0).max() / np.abs(d0).max():.2e}", flush=True)
