"""Work package 1: is the interior start dependence a coordinate freedom, a weak physical direction, or branches?

usage: python interior_gauge.py OUT.json   (cwd: a vmex tree; a GPU)
    env DECK=small (LP QA test deck; MPOL, NS) | DECK=<input file> (e.g. a fixture's input.run, solved fixed-boundary)
        FTOL (default 1e-13), RELABEL_N (default "1,2"), EPS (default "1e-4,3e-4,1e-3,3e-3,1e-2")

(a) Three starts of one fixed-boundary deck: cold, warm (from the converged state of the deck with its boundary
    perturbed by 1e-3), perturbed (the cold state times 1 + 1e-3 cos(k)).  Each is a fresh solve (no hot restart).
    For each state: fsq; the host force gc and the part of it outside the implicit dof subspace, |gc - P gc|; the frozen
    m = 1 complement (antisymmetric Z_sin pair combination) and its change between starts; coefficient differences;
    physical differences: iota(s), surface distance (point-to-segment), |B| min/max per surface.
(b) The relabelling path theta' = theta + f, f = eps s (1 - s) sin(n Nfp phi):
    R' = R(theta' - f), Z' = Z(theta' - f), lambda' = lambda(theta' - f) - f, evaluated exactly on an oversampled grid
    and projected onto the mode table (projection error reported), mapped to the state by state_from_wout; the force
    residual fsq against eps (quadratic: first-order null; linear: not a null direction).
(c) The infinitesimal generator delta = d/deps at eps = 0 (central difference of the exact map at eps = 1e-7): the raw
    implicit residual's JVP along delta against JVPs along random directions of the same norm (dof subspace only).
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from vmex import optimize as opt
from vmex.core import implicit as im
from vmex.core.restart import state_from_wout
from vmex.core.solver import evaluate_forces

FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def deck():
    """The fixed-boundary deck of the run."""
    import vmex as vj

    ftol = float(os.environ.get("FTOL", 1e-13))
    name = os.environ.get("DECK", "small")
    if name == "small":
        sys.path.insert(0, "tests")
        import test_freeboundary_vc as t

        m, ns = int(os.environ.get("MPOL", 2)), int(os.environ.get("NS", 7))
        inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_lowres", m, ns), phiedge=-0.025, curtor=0.0,
                                  am=np.zeros(21))
    else:
        inp = vj.VmecInput.from_file(name)
    return dataclasses.replace(inp, lfreeb=False, mgrid_file="NONE", ftol_array=np.array([ftol]),
                               niter_array=np.array([50000]))


def leaves(state):
    """The state's fields as numpy arrays (by name)."""
    return {f: np.asarray(getattr(state, f)) for f in FIELDS}


def solve(inp, initial=None):
    """A fresh fixed-boundary solve (no hot restart)."""
    return opt.solve_equilibrium(inp, initial_state=initial)


def physical(w):
    """Gauge-invariant per-surface quantities and the surfaces for distances."""
    xm, xn, nfp = np.asarray(w.xm), np.asarray(w.xn), int(w.nfp)
    th = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, 16, endpoint=False)
    surfaces = []
    for js in range(1, int(w.ns)):
        cuts = []
        for p in ph:
            a = xm[None, :] * th[:, None] - xn[None, :] * p
            cuts.append((np.cos(a) @ np.asarray(w.rmnc)[js], np.sin(a) @ np.asarray(w.zmns)[js]))
        surfaces.append(cuts)
    xmn, xnn = np.asarray(w.xm_nyq), np.asarray(w.xn_nyq)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    a = xmn[None, None, :] * TH[..., None] - xnn[None, None, :] * PH[..., None]
    B = np.einsum("tpk,sk->stp", np.cos(a), np.asarray(w.bmnc)[1:])
    return dict(iota=np.asarray(w.iotaf), bmin=B.min(axis=(1, 2)), bmax=B.max(axis=(1, 2)), surfaces=surfaces)


def distance(c1, c2):
    """Largest point-to-segment distance between two surfaces (per cut)."""
    worst = 0.0
    for (R1, Z1), (R2, Z2) in zip(c1, c2):
        P, Q = np.stack([R1, Z1], 1)[::4], np.stack([R2, Z2], 1)
        seg = np.roll(Q, -1, 0) - Q
        t = np.clip(np.einsum("psk,sk->ps", P[:, None] - Q[None], seg) / np.sum(seg**2, 1)[None], 0, 1)
        d = np.min(np.linalg.norm(P[:, None] - (Q[None] + t[..., None] * seg[None]), axis=2), axis=1)
        worst = max(worst, float(d.max()))
    return worst


def relabelled_wout(w, eps, n):
    """The wout of the relabelled state theta' = theta + eps s(1 - s) sin(n Nfp phi), projected on its own modes."""
    xm, xn, nfp = np.asarray(w.xm, float), np.asarray(w.xn, float), int(w.nfp)
    mmax, nmax = int(xm.max()), int(np.abs(xn).max() // max(nfp, 1))
    nth, nph = 8 * (mmax + 1) + 8, 8 * (2 * nmax + 1) + 8
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    ns = int(w.ns)
    s_full = np.linspace(0, 1, ns)
    s_half = np.r_[0.0, 0.5 * (s_full[1:] + s_full[:-1])]
    basis_c = np.cos(xm[None, None, :] * TH[..., None] - xn[None, None, :] * PH[..., None]).reshape(-1, xm.size)
    basis_s = np.sin(xm[None, None, :] * TH[..., None] - xn[None, None, :] * PH[..., None]).reshape(-1, xm.size)
    lstsq_c, lstsq_s = np.linalg.pinv(basis_c), np.linalg.pinv(basis_s)
    out = {k: np.array(getattr(w, k), dtype=float) for k in ("rmnc", "zmns", "lmns")}
    errors = dict(R=0.0, Z=0.0, L=0.0)
    for js in range(ns):
        for key, s_val, kind in (("rmnc", s_full[js], "c"), ("zmns", s_full[js], "s"), ("lmns", s_half[js], "s")):
            if key == "lmns" and js == 0:
                continue
            f = eps * s_val * (1 - s_val) * np.sin(n * nfp * PH)
            shifted = (TH - f)[..., None] * xm[None, None, :] - xn[None, None, :] * PH[..., None]
            coef = np.asarray(getattr(w, key))[js]
            values = (np.cos(shifted) if kind == "c" else np.sin(shifted)) @ coef
            if key == "lmns":
                values = values - f
            proj = (lstsq_c if kind == "c" else lstsq_s) @ values.ravel()
            recon = (basis_c if kind == "c" else basis_s) @ proj
            errors[{"rmnc": "R", "zmns": "Z", "lmns": "L"}[key]] = max(
                errors[{"rmnc": "R", "zmns": "Z", "lmns": "L"}[key]], float(np.max(np.abs(recon - values.ravel()))))
            out[key][js] = proj
    return dataclasses.replace(w, **out), errors


def main():
    """Run (a), (b) and (c) and write OUT.json."""
    result = {}
    inp = deck()
    result["deck"] = dict(name=os.environ.get("DECK", "small"), mpol=int(inp.mpol), ntor=int(inp.ntor),
                          ns=int(np.atleast_1d(inp.ns_array)[-1]), ftol=float(np.atleast_1d(inp.ftol_array)[-1]))
    cold = solve(inp)
    shifted = dataclasses.replace(inp, rbc=np.asarray(inp.rbc) * (1 + 1e-3))
    near = solve(shifted).state
    warm = solve(inp, near)
    seed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                        if jnp.issubdtype(a.dtype, jnp.inexact) else a, cold.state)
    pert = solve(inp, seed)
    cfg = im._canonical_config(im.make_config(inp, multigrid=True, hot_restart=False))
    mask = im._fixed_boundary_dof_mask(cfg)
    P = im._dof_projector(cfg, mask)
    pos, neg = im._m1_pair_columns(cfg)
    rt = cold.solver_context
    dev = jax.devices()[0]
    states = dict(cold=cold, warm=warm, perturbed=pert)
    phys = {k: physical(v.wout) for k, v in states.items()}
    per = {}
    for name, eq in states.items():
        x = jax.device_put(eq.solution, dev)
        gc, res, _ = evaluate_forces(x, rt)
        Pgc = P(gc)
        out_of_space = float(np.sqrt(sum(np.sum((np.asarray(getattr(gc, f)) - np.asarray(getattr(Pgc, f))) ** 2)
                                         for f in FIELDS)))
        total = float(np.sqrt(sum(np.sum(np.asarray(getattr(gc, f)) ** 2) for f in FIELDS)))
        zs = np.asarray(eq.solution.Z_sin)
        complement = 0.5 * (zs[:, pos] - zs[:, neg])
        per[name] = dict(fsqr=float(res.fsqr), fsqz=float(res.fsqz), fsql=float(res.fsql), gc_norm=total,
                         gc_outside_dof_space=out_of_space, complement=complement.tolist())
    result["starts"] = {k: {kk: vv for kk, vv in v.items() if kk != "complement"} for k, v in per.items()}
    base = leaves(cold.solution)
    comp0 = np.array(per["cold"]["complement"])
    result["differences"] = {}
    for name in ("warm", "perturbed"):
        other = leaves(states[name].solution)
        result["differences"][name] = dict(
            coefficients={f: float(np.max(np.abs(other[f] - base[f]))) for f in FIELDS},
            frozen_complement=float(np.max(np.abs(np.array(per[name]["complement"]) - comp0))),
            iota=float(np.max(np.abs(phys[name]["iota"] - phys["cold"]["iota"]))),
            bmin=float(np.max(np.abs(phys[name]["bmin"] - phys["cold"]["bmin"]))),
            bmax=float(np.max(np.abs(phys[name]["bmax"] - phys["cold"]["bmax"]))),
            surface_distance=[distance(a, b) for a, b in zip(phys[name]["surfaces"], phys["cold"]["surfaces"])])
    # (b) relabelling path from the cold solution
    w = cold.wout
    result["relabel"] = []
    for n in [int(v) for v in os.environ.get("RELABEL_N", "1,2").split(",")]:
        for eps in [float(v) for v in os.environ.get("EPS", "1e-4,3e-4,1e-3,3e-3,1e-2").split(",")]:
            wr, errors = relabelled_wout(w, eps, n)
            x = jax.device_put(state_from_wout(wr, inp=inp), dev)
            _, res, _ = evaluate_forces(x, rt)
            result["relabel"].append(dict(n=n, eps=eps, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
                                          fsql=float(res.fsql), projection_error=errors))
            print(f"relabel n {n} eps {eps:.0e}: fsq {float(res.fsqr):.3e} {float(res.fsqz):.3e} "
                  f"{float(res.fsql):.3e}; projection {errors}", flush=True)
    # (c) generator vs random directions, raw implicit residual JVP on the dof subspace
    params = im.params_from_input(inp, device=dev)
    frozen = jax.device_put(cold.solution, dev)
    F = im.residual_fn(cfg, frozen, mask, formulation="raw")
    z0 = frozen

    def norm(t):
        return float(np.sqrt(sum(np.sum(np.asarray(getattr(t, f)) ** 2) for f in FIELDS)))

    result["generator"] = []
    h = 1e-7
    for n in [int(v) for v in os.environ.get("RELABEL_N", "1,2").split(",")]:
        plus = state_from_wout(relabelled_wout(w, h, n)[0], inp=inp)
        minus = state_from_wout(relabelled_wout(w, -h, n)[0], inp=inp)
        delta = P(jax.tree.map(lambda a, b: (jnp.asarray(a) - jnp.asarray(b)) / (2 * h), plus, minus))
        jv = jax.jvp(lambda z: F(z, params), (z0,), (delta,))[1]
        ratio = norm(jv) / norm(delta)
        rng = np.random.default_rng(n)
        rand = []
        for _ in range(4):
            r = P(jax.tree.map(lambda a: jnp.asarray(rng.normal(size=np.shape(a))), delta))
            r = jax.tree.map(lambda a: a * norm(delta) / norm(r), r)
            rand.append(norm(jax.jvp(lambda z: F(z, params), (z0,), (r,))[1]) / norm(r))
        result["generator"].append(dict(n=n, delta_norm=norm(delta), jvp_over_delta=ratio,
                                        random_jvp_over_norm=rand))
        print(f"generator n {n}: |J delta|/|delta| {ratio:.3e}; random {np.round(rand, 4)}", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: result[k] for k in ("deck", "starts", "differences")}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
