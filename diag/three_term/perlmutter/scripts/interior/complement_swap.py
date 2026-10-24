"""Is the frozen m = 1 complement what makes the interface rows depend on the interior start?

usage: python complement_swap.py OUT.json   (cwd: a vmex tree; a GPU; env MPOL, NS, FTOL as in rows_gauge.py)

The three starts of rows_gauge.py (cold, near, perturbed).  For each: the solved state refined onto the discrete
equilibrium as is, and the same state with its frozen m = 1 complement (the antisymmetric part of the Z_sin
(m = 1, +n)/(m = 1, -n) pair, the direction the dof projector leaves out) replaced by the cold state's, then refined.
If the swapped rows agree with the cold rows, the start dependence is the complement's history-dependent value.
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import implicit as im  # noqa: E402


def main():
    """Rows from three starts, with and without the cold complement; write OUT.json."""
    inp, field = t._small_vacuum()
    if os.environ.get("MPOL") or os.environ.get("NS"):
        inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_lowres", int(os.environ.get("MPOL", 2)),
                                          int(os.environ.get("NS", 7))), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    if os.environ.get("FTOL"):
        inp = dataclasses.replace(inp, ftol_array=np.array([float(os.environ["FTOL"])]), niter_array=np.array([50000]))
    model = fvc.ThreeTermFreeBoundaryModel(dataclasses.replace(inp, lfreeb=False), **t.SMALL)
    x, params = np.asarray(model.x0), model.params0
    near = model.solve(model.with_boundary(params, jnp.asarray(x + 1e-3 * np.asarray(model.x_scale))))[0]
    perturbed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                             if jnp.issubdtype(a.dtype, jnp.inexact) else a, model.seed[0])
    mask = im._fixed_boundary_dof_mask(model.cfg)
    pos, neg = im._m1_pair_columns(model.cfg)
    states, params_x = {}, None
    for name, seed in (("cold", None), ("near", near), ("perturbed", perturbed)):
        for cfg in {model.cfg, model.cfg_trial}:
            for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED, im._LAST_REFINED):
                cache.pop(cfg, None)
        model._linearization = None
        _, aux = model.evaluate(x, params, field, seed=seed, tight=True)
        states[name], params_x = aux[0], aux[2]
        model._last_linearize = None

    def anti(st):
        v = np.asarray(st.Z_sin)
        return 0.5 * (v[:, pos] - v[:, neg])

    def with_complement(st, a):
        v = np.asarray(st.Z_sin).copy()
        sym = 0.5 * (v[:, pos] + v[:, neg])
        v[:, pos], v[:, neg] = sym + a, sym - a
        return dataclasses.replace(st, Z_sin=jnp.asarray(v))

    def rows(st):
        return np.asarray(model._rows(im._refined_state(model.cfg, params_x, st, mask), params_x, field))

    base = rows(states["cold"])
    a_cold = anti(states["cold"])
    out = dict(resolution=model.resolution(), complement_rms=float(np.sqrt(np.mean(a_cold**2))))
    for name in ("near", "perturbed"):
        st = states[name]
        r_plain, r_swap = rows(st), rows(with_complement(st, a_cold))
        rel = lambda r: float(np.max(np.abs(r - base)) / np.max(np.abs(base)))  # noqa: E731
        out[name] = dict(complement_gap=float(np.max(np.abs(anti(st) - a_cold))),
                         rows_max_rel_plain=rel(r_plain), rows_max_rel_swapped=rel(r_swap))
        print(name, out[name], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
