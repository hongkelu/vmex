"""Candidate deterministic gauge: VMEC's m = 1 polar constraint imposed exactly (frozen complement = 0), then the
implicit layer's Newton refinement of the evolved dofs (which holds the complement fixed).

usage: python gauge_fix_check.py OUT.json   (cwd: a vmex tree; a GPU; env DECK, FTOL as interior_gauge.py)

For the cold, warm and perturbed plain solutions: the complement before, the refined residual norm with the complement
held at its own value and at zero, and the pairwise gaps of the refined states (coefficients) in both cases.  Equal
refined states from every start with the complement at zero, at the refinement tolerance, make complement = 0 a valid
deterministic gauge for this deck.
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from vmex.core import implicit as im

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interior_gauge import FIELDS, deck, leaves, solve  # noqa: E402


def gap(a, b):
    """Largest coefficient difference between two states."""
    la, lb = leaves(a), leaves(b)
    return max(float(np.max(np.abs(la[f] - lb[f]))) for f in FIELDS)


def main():
    """Run the check and write OUT.json."""
    inp = deck()
    cold = solve(inp)
    near = solve(dataclasses.replace(inp, rbc=np.asarray(inp.rbc) * (1 + 1e-3))).state
    warm = solve(inp, near)
    seed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                        if jnp.issubdtype(a.dtype, jnp.inexact) else a, cold.state)
    pert = solve(inp, seed)
    cfg = im._canonical_config(im.make_config(inp, multigrid=True, hot_restart=True))
    mask = im._fixed_boundary_dof_mask(cfg)
    params = im.params_from_input(inp, device=jax.devices()[0])
    pos, neg = im._m1_pair_columns(cfg)

    def zero_complement(state):
        zs = np.array(state.Z_sin)
        sym = 0.5 * (zs[:, pos] + zs[:, neg])
        zs[:, pos], zs[:, neg] = sym, sym
        return dataclasses.replace(state, Z_sin=jnp.asarray(zs))

    def residual(state):
        F = im.residual_fn(cfg, state, mask, formulation="raw")
        P = im._dof_projector(cfg, mask)
        return float(im._tree_norm(F(P(state), params)))

    out = dict(refine_tol=float(cfg.refine_tol), starts={})
    refined = {}
    for name, eq in dict(cold=cold, warm=warm, perturbed=pert).items():
        state = jax.device_put(eq.solution, jax.devices()[0])
        zs = np.asarray(state.Z_sin)
        row = dict(complement=float(np.max(np.abs(0.5 * (zs[:, pos] - zs[:, neg])))))
        for label, start in (("own", state), ("zero", zero_complement(state))):
            row[f"residual_{label}_before"] = residual(start)
            r = im._refined_state(cfg, params, start, mask)
            row[f"residual_{label}_after"] = residual(r)
            refined[(name, label)] = r
        out["starts"][name] = row
        print(name, json.dumps(row), flush=True)
    for label in ("own", "zero"):
        out[f"refined_gaps_{label}"] = {
            "cold-warm": gap(refined[("cold", label)], refined[("warm", label)]),
            "cold-perturbed": gap(refined[("cold", label)], refined[("perturbed", label)]),
            "warm-perturbed": gap(refined[("warm", label)], refined[("perturbed", label)])}
        print(label, out[f"refined_gaps_{label}"], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
