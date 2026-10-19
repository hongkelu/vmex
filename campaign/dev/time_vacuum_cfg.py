"""Vacuum-iota solve variants: time, value and gradient against the default config.

    COIL_CASE=qa4-beta python extra/time_vacuum_cfg.py <run dir with input.run>     (cwd: the vmex tree, GPU)
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
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core.statephysics import _iotas_half  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
eqs = [opt.solve_equilibrium(inp)]
for k in (1, 2):
    rbc = np.array(inp.rbc); rbc[inp.ntor - 1, 1] += k * 1e-3
    eqs.append(opt.solve_equilibrium(dataclasses.replace(inp, rbc=rbc), initial_state=eqs[0].state))
vacuum = dataclasses.replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                             ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None)


def make(**kw):
    cfg = im.make_config(vacuum, hot_restart=True, **kw)
    base = im.params_from_input(vacuum)
    im.runtime_from_params(base, cfg)

    def iota(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)
        rows, cols = np.asarray(runtime.modes.n) + int(inp.ntor), np.asarray(runtime.modes.m)
        p = dataclasses.replace(base, rbc=jnp.zeros_like(base.rbc).at[rows, cols].set(rmnc),
                                zbs=jnp.zeros_like(base.zbs).at[rows, cols].set(zmns))
        vs, vr = im.solve_implicit(p, cfg), im.runtime_from_params(p, cfg)
        half = _iotas_half(vs, vr)[1:]
        return jnp.concatenate([opt.axis_iota(vs, vr)[None], half, 1.5 * half[-1:] - 0.5 * half[-2:-1]])

    f = jax.jit(iota)
    g = jax.jit(jax.grad(lambda s, r: jnp.min(jnp.abs(iota(s, r)))))
    return f, g


results = {}
for name, kw in (("default (multigrid, refine)", dict(multigrid=True)),
                 ("single grid, refine", dict(multigrid=False)),
                 ("multigrid, no refine", dict(multigrid=True, refine_tol=np.inf)),
                 ("single grid, no refine", dict(multigrid=False, refine_tol=np.inf))):
    f, g = make(**kw)
    f(eqs[0].state, eqs[0].runtime)  # warm at the base boundary
    out = []
    for eq in eqs[1:]:
        t = time.perf_counter(); v = np.asarray(f(eq.state, eq.runtime)); tv = time.perf_counter() - t
        t = time.perf_counter(); d = jax.block_until_ready(g(eq.state, eq.runtime)); tg = time.perf_counter() - t
        out.append((tv, tg, v, np.asarray(d.R_cos)))
    results[name] = out
    print(f"{name:30s} value {out[0][0]:6.1f} / {out[1][0]:6.1f} s   value+grad {out[0][1]:6.1f} / {out[1][1]:6.1f} s",
          flush=True)
ref = results["default (multigrid, refine)"]
for name, out in results.items():
    dv = max(np.max(np.abs(o[2] - r[2])) for o, r in zip(out, ref))
    dg = max(np.max(np.abs(o[3] - r[3])) / np.max(np.abs(r[3])) for o, r in zip(out, ref))
    print(f"{name:30s} max |d iota| {dv:.2e}   max rel d grad(R_cos) {dg:.2e}", flush=True)
