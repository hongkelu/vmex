"""Candidate deterministic interior gauge: lambda's m = 0 harmonics constant in s (equal to the edge's).

Under theta -> theta + f(s, phi), lambda_0n(s) shifts by -f_n(s) (to first order); f must vanish at the fixed
boundary.  The gauge f_n(s) = lambda_0n(s) - lambda_0n(edge) fixes the m = 0 relabelling uniquely, vanishes at the
boundary and leaves the axis free.  This probe brings each start's solution to that gauge with the exact nonlinear
relabelling (iterated), maps it to the state, Newton-refines it (implicit layer, complement held), and compares the
starts.

usage: python canonical_gauge.py OUT.json   (cwd: a vmex tree; a GPU; env DECK, FTOL as interior_gauge.py;
    ITER (default 3) relabelling iterations)
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from vmex.core import implicit as im
from vmex.core.restart import state_from_wout

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interior_gauge import FIELDS, deck, distance, leaves, physical, solve  # noqa: E402


def gap(a, b):
    """Largest coefficient difference between two states."""
    la, lb = leaves(a), leaves(b)
    return max(float(np.max(np.abs(la[f] - lb[f]))) for f in FIELDS)


def m0_lambda(w, phi):
    """lambda's m = 0 part on the half mesh, (ns, len(phi)) (row 0 unused)."""
    xm, xn = np.asarray(w.xm), np.asarray(w.xn)
    k = np.flatnonzero(xm == 0)
    return np.asarray(w.lmns)[:, k] @ np.sin(-np.outer(xn[k], phi))


def relabel(w, f_full, f_half, phi):
    """theta' = theta + f(s, phi) with f given per surface on a phi grid (full mesh for R, Z; half mesh for lambda),
    evaluated exactly and projected on the wout's modes."""
    xm, xn, nfp = np.asarray(w.xm, float), np.asarray(w.xn, float), int(w.nfp)
    nth = 8 * (int(xm.max()) + 1) + 8
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    TH, PH = np.meshgrid(th, phi, indexing="ij")
    A = xm[None, None, :] * TH[..., None] - xn[None, None, :] * PH[..., None]
    bc, bs = np.cos(A).reshape(-1, xm.size), np.sin(A).reshape(-1, xm.size)
    pc, ps = np.linalg.pinv(bc), np.linalg.pinv(bs)
    out = {k: np.array(getattr(w, k), dtype=float) for k in ("rmnc", "zmns", "lmns")}
    worst = 0.0
    for js in range(int(w.ns)):
        for key, f_row, kind in (("rmnc", f_full[js], "c"), ("zmns", f_full[js], "s"), ("lmns", f_half[js], "s")):
            if key == "lmns" and js == 0:
                continue
            F = np.broadcast_to(f_row[None, :], TH.shape)
            shifted = (TH - F)[..., None] * xm[None, None, :] - xn[None, None, :] * PH[..., None]
            coef = np.asarray(getattr(w, key))[js]
            values = (np.cos(shifted) if kind == "c" else np.sin(shifted)) @ coef
            if key == "lmns":
                values = values - F
            proj = (pc if kind == "c" else ps) @ values.ravel()
            worst = max(worst, float(np.max(np.abs((bc if kind == "c" else bs) @ proj - values.ravel()))))
            out[key][js] = proj
    return dataclasses.replace(w, **out), worst


def canonical(w, iterations):
    """The wout relabelled until lambda_0n(s) = lambda_0n(edge) on every half-mesh surface."""
    nfp, ntor = int(w.nfp), int(np.abs(np.asarray(w.xn)).max() // max(int(w.nfp), 1))
    phi = np.linspace(0, 2 * np.pi / nfp, 8 * (2 * ntor + 1) + 8, endpoint=False)
    history, worst = [], 0.0
    for _ in range(iterations):
        lam = m0_lambda(w, phi)
        f_half = lam - lam[-1][None, :]
        f_half[0] = f_half[1]
        f_full = np.zeros_like(f_half)
        f_full[1:-1] = 0.5 * (f_half[1:-1] + f_half[2:])
        f_full[0] = f_half[1]  # the axis: no theta dependence, any f
        w, err = relabel(w, f_full, f_half, phi)
        worst = max(worst, err)
        history.append(float(np.max(np.abs(f_half[1:]))))
    lam = m0_lambda(w, phi)
    return w, dict(f_max_per_iteration=history, condition_after=float(np.max(np.abs((lam - lam[-1][None, :])[1:]))),
                   projection_error=worst)


def main():
    """Bring every start to the gauge, refine, compare; write OUT.json."""
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
    P = im._dof_projector(cfg, mask)

    def residual(state):
        F = im.residual_fn(cfg, state, mask, formulation="raw")
        return float(im._tree_norm(F(P(state), params)))

    iterations = int(os.environ.get("ITER", 3))
    result, gauged, plain = dict(starts={}), {}, {}
    for name, eq in dict(cold=cold, warm=warm, perturbed=pert).items():
        wc, info = canonical(eq.wout, iterations)
        state = jax.device_put(state_from_wout(wc, inp=inp), jax.devices()[0])
        info["residual_before_refine"] = residual(state)
        refined = im._refined_state(cfg, params, state, mask)
        info["residual_after_refine"] = residual(refined)
        gauged[name] = refined
        plain[name] = im._refined_state(cfg, params, jax.device_put(eq.solution, jax.devices()[0]), mask)
        result["starts"][name] = info
        print(name, json.dumps(info), flush=True)
    for label, states in (("plain_refined", plain), ("gauged_refined", gauged)):
        result[label] = {f"{a}-{b}": gap(states[a], states[b])
                         for a, b in (("cold", "warm"), ("cold", "perturbed"), ("warm", "perturbed"))}
        print(label, result[label], flush=True)
    # physical comparison of the gauged, refined states (re-solved briefly from them for the wout)
    phys = {k: physical(solve(inp, v).wout) for k, v in gauged.items()}
    result["gauged_physical"] = {
        f"cold-{b}": dict(iota=float(np.max(np.abs(phys[b]["iota"] - phys["cold"]["iota"]))),
                          surface=max(distance(x, y) for x, y in zip(phys[b]["surfaces"], phys["cold"]["surfaces"])))
        for b in ("warm", "perturbed")}
    print("gauged physical", result["gauged_physical"], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
