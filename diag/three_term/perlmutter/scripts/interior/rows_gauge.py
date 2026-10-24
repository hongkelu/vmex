"""The three-term interface rows at one boundary from three interior starts, with and without the canonical m = 0
lambda gauge (canonical_gauge.py): test_interior_state_does_not_depend_on_its_start's comparison.

usage: python rows_gauge.py OUT.json   (cwd: a vmex tree; a GPU)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_freeboundary_vc as t  # noqa: E402
from canonical_gauge import canonical  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core.restart import state_from_wout  # noqa: E402


def main():
    """Compare the rows from the three starts with and without the gauge; write OUT.json."""
    inp, field = t._small_vacuum()
    if os.environ.get("MPOL") or os.environ.get("NS"):  # the same deck at another resolution
        inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_lowres", int(os.environ.get("MPOL", 2)),
                                          int(os.environ.get("NS", 7))), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    if os.environ.get("FTOL"):
        inp = dataclasses.replace(inp, ftol_array=np.array([float(os.environ["FTOL"])]), niter_array=np.array([50000]))
    model = fvc.ThreeTermFreeBoundaryModel(dataclasses.replace(inp, lfreeb=False), **t.SMALL)
    x, params = np.asarray(model.x0), model.params0
    near = model.solve(model.with_boundary(params, jnp.asarray(x + 1e-3 * np.asarray(model.x_scale))))[0]
    perturbed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                             if jnp.issubdtype(a.dtype, jnp.inexact) else a, model.seed[0])
    fixed = model.fixed
    mask = im._fixed_boundary_dof_mask(model.cfg)
    out = dict(plain={}, gauged={})
    rows = dict(plain={}, gauged={})
    for name, seed in (("cold", None), ("near", near), ("perturbed", perturbed)):
        for cfg in {model.cfg, model.cfg_trial}:
            for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED, im._LAST_REFINED):
                cache.pop(cfg, None)
        model._linearization = None
        r, aux = model.evaluate(x, params, field, seed=seed, tight=True)
        rows["plain"][name] = np.asarray(r)
        wout = opt.solve_equilibrium(fixed, initial_state=aux[0]).wout
        wc, info = canonical(wout, 3)
        state = jax.device_put(state_from_wout(wc, inp=fixed), jax.devices()[0])
        refined = im._refined_state(model.cfg, aux[2], state, mask)
        rows["gauged"][name] = np.asarray(model._rows(refined, aux[2], field))
        out["gauged"][name + "_condition_after"] = info["condition_after"]
        model._last_linearize = None
    out["resolution"] = model.resolution()
    for label in ("plain", "gauged"):
        base = rows[label]["cold"]
        for name in ("near", "perturbed"):
            out[label][f"{name}_vs_cold_rows_max_rel"] = float(np.max(np.abs(rows[label][name] - base))
                                                              / np.max(np.abs(base)))
            for block, start, stop in model.row_blocks:  # which conditions carry the start dependence
                d = rows[label][name][start:stop] - base[start:stop]
                out[label][f"{name}_{block}_rms_diff"] = float(np.sqrt(np.mean(d**2)))
                out[label][f"{block}_rms"] = float(np.sqrt(np.mean(base[start:stop] ** 2)))
        print(label, {k: v for k, v in out[label].items() if "max_rel" in k}, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
