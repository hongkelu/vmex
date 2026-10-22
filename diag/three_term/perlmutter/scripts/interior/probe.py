"""Why test_interior_state_does_not_depend_on_its_start fails on Perlmutter: per start, the force residuals of the
tight solve, the largest state difference per field from the cold solve, and the row difference.
usage (cwd: a vmex tree): python probe.py"""
import sys
from dataclasses import replace

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
import os as _os

if _os.environ.get("KEEP_M1"):  # never release the constrained m=1 Z force (solver and implicit residual)
    from vmex.core import solver as _solver

    _solver.zero_m1_z_force = lambda force, enabled: force
    im.zero_m1_z_force = lambda force, enabled: force
    print("KEEP_M1", flush=True)

print("jax", jax.__version__, jax.devices())
inp, field = t._small_vacuum()
import os
if os.environ.get("MPOL"):
    m = int(os.environ["MPOL"])
    inp = replace(t._deck("input.LandremanPaul2021_QA_lowres", m, int(os.environ.get("NS", 7))), phiedge=-0.025,
                  curtor=0.0, am=np.zeros(21))
if os.environ.get("FTOL"):
    inp = replace(inp, ftol_array=np.array([float(os.environ["FTOL"])]), niter_array=np.array([20000]))
model = fvc.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **t.SMALL)
x, params = np.asarray(model.x0), model.params0
print("deck ftol", model.fixed.ftol_array, "niter", model.fixed.niter_array, "cfg ftol", model.cfg.ftol)
near = model.solve(model.with_boundary(params, jnp.asarray(x + 1e-3 * np.asarray(model.x_scale))))[0]
perturbed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                         if jnp.issubdtype(a.dtype, jnp.inexact) else a, model.seed[0])
out = []
for name, seed in (("cold", None), ("near", near), ("perturbed", perturbed)):
    for cfg in {model.cfg, model.cfg_trial}:
        for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED, im._LAST_REFINED):
            cache.pop(cfg, None)
    model._linearization = None
    rows, aux = model.evaluate(x, params, field, seed=seed, tight=True)
    res = im._LAST_SOLVE[model.cfg][1]
    print(f"{name}: fsqr {float(res.fsqr):.3e} fsqz {float(res.fsqz):.3e} fsql {float(res.fsql):.3e} "
          f"converged {bool(res.converged)} iter {getattr(res, 'iterations', getattr(res, 'iter', '?'))}")
    out.append((name, np.asarray(rows), aux[0]))
    model._last_linearize = None
names = [f for f in getattr(out[0][2], "_fields", [])] or [str(i) for i in range(len(jax.tree.leaves(out[0][2])))]
for name, rows, state in out[1:]:
    rel = np.max(np.abs(rows - out[0][1])) / np.max(np.abs(out[0][1]))
    diffs = [(n, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))), float(np.max(np.abs(np.asarray(b)))))
             for n, a, b in zip(names, jax.tree.leaves(state), jax.tree.leaves(out[0][2])) if np.size(b)]
    print(f"{name} vs cold: rows max rel {rel:.2e}; state max |diff| per leaf:",
          ", ".join(f"{n}={d:.1e}/{m:.1e}" for n, d, m in diffs))
print("where the states differ (largest |diff| per radius js, and the mode index at the largest):")
for name, rows, state in out[1:]:
    for leaf, a, b in zip(("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"), jax.tree.leaves(state),
                          jax.tree.leaves(out[0][2])):
        d = np.abs(np.asarray(a) - np.asarray(b))
        if d.max() == 0:
            continue
        per_js = d.max(axis=1)
        js, mode = np.unravel_index(np.argmax(d), d.shape)
        print(f"  {name} {leaf}: max {d.max():.2e} at js {js} mode {mode}; per js "
              + " ".join(f"{v:.0e}" for v in per_js))
mn = getattr(model.cfg, "mnmax", None)
print("ns", np.asarray(out[0][2].R_cos).shape)
