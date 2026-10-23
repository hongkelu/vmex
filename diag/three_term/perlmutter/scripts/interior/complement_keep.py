"""Option K for the frozen m = 1 complement: keep its force (never released) and refine it with the evolved dofs.

usage: python complement_keep.py OUT.npz   (cwd: a vmex tree; a GPU; env MPOL, NS, FTOL as rows_gauge.py; KEEP=1)

KEEP=1: residue.f90's constrain_m1 release is disabled in the host solver and the implicit residual, and the dof
projector is the elementwise mask (the m = 1 pair is not symmetrized), so the Newton refinement solves for the
complement too.  KEEP unset: the code as is.  Saved per start (cold, near, perturbed): the refined state's rows,
and for the cold start iota and the surfaces (interior_gauge.py's measures) and the refined residual norm.
Compare two runs with complement_keep_compare.py.
"""
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "tests")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_freeboundary_vc as t  # noqa: E402
from interior_gauge import physical  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core import solver as _solver  # noqa: E402
from vmex.core.solver import SpectralState  # noqa: E402

KEEP = bool(os.environ.get("KEEP"))
if KEEP:
    _solver.zero_m1_z_force = lambda force, enabled: force
    im.zero_m1_z_force = lambda force, enabled: force

    def _elementwise(cfg, dof_mask):
        def P(tree):
            return SpectralState(**{k: getattr(dof_mask, k) * getattr(tree, k) for k in im._STATE_FIELDS})
        return P

    im._dof_projector = _elementwise


def main():
    """Rows from three starts and the cold physics; write OUT.npz."""
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
    P = im._dof_projector(model.cfg, mask)
    out = {}
    for name, seed in (("cold", None), ("near", near), ("perturbed", perturbed)):
        for cfg in {model.cfg, model.cfg_trial}:
            for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED, im._LAST_REFINED):
                cache.pop(cfg, None)
        model._linearization = None
        _, aux = model.evaluate(x, params, field, seed=seed, tight=True)
        state, params_x = aux[0], aux[2]
        refined = im._refined_state(model.cfg, params_x, state, mask)
        out[f"rows_{name}"] = np.asarray(model._rows(refined, params_x, field))
        F = im.residual_fn(model.cfg, refined, mask, formulation="raw")
        out[f"residual_{name}"] = float(im._tree_norm(F(P(refined), params_x)))
        print(name, "residual", out[f"residual_{name}"], flush=True)
        if name == "cold":
            ph = physical(opt.solve_equilibrium(model.fixed, initial_state=refined).wout)
            out["iota"], out["surfaces"] = ph["iota"], np.asarray(ph["surfaces"])
            out["bmin"], out["bmax"] = ph["bmin"], ph["bmax"]
        model._last_linearize = None
    np.savez(sys.argv[1], **out)


if __name__ == "__main__":
    main()
