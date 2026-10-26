"""A deterministic gauge for the frozen m = 1 complement: complement(s) = complement(edge) w(s), then refine.

usage: python complement_gauge.py OUT.json   (cwd: a vmex tree; a GPU; env MPOL, NS, FTOL as in rows_gauge.py)

The three starts of rows_gauge.py.  Printed: the cold complement's radial profile relative to its edge value.  For
each profile w (rho, s, rho^3): every start's complement set to complement(edge) w(s), refined (complement held);
reported: the rows' spread across starts, and the change from the cold rows as is
(the gauge's cost).
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
from vmex.core import freeboundary_three_term as fvc  # noqa: E402
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
    ns = a_cold.shape[0]
    s_full = np.linspace(0.0, 1.0, ns)
    edge = a_cold[-1]
    ratio = a_cold / np.where(np.abs(edge) > 1e-14, edge, np.nan)[None, :]
    out = dict(resolution=model.resolution(), edge=edge.tolist(), cold_profile=ratio.tolist(), s=s_full.tolist())
    print("cold complement / edge, by surface (columns n = 1..):", flush=True)
    for j in range(ns):
        print(f"  s={s_full[j]:.3f} rho={np.sqrt(s_full[j]):.3f}", np.array2string(ratio[j], precision=4), flush=True)
    rel = lambda r, b: float(np.max(np.abs(r - b)) / np.max(np.abs(b)))  # noqa: E731
    for label, w in (("rho", np.sqrt(s_full)), ("s", s_full), ("rho3", s_full**1.5)):
        target = w[:, None] * edge[None, :]
        got = {}
        for name, st in states.items():
            refined = im._refined_state(model.cfg, params_x, with_complement(st, target), mask)
            got[name] = np.asarray(model._rows(refined, params_x, field))
        out[label] = dict(spread=max(rel(got[k], got["cold"]) for k in ("near", "perturbed")),
                          cost_vs_cold_plain=rel(got["cold"], base))
        print(label, out[label], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
