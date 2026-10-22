"""Fixed-boundary solves of the small LP QA deck from a cold and a perturbed start, with VMEC2000's diagonal lambda
preconditioner and with the m = 0 lambda radial block (lambda_m0_floor): do the converged states agree?
usage: MPOL=2 NS=15 FTOL=1e-13 python lam0.py   (cwd: a vmex tree)"""
import os
import sys
from dataclasses import replace

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex import optimize as opt  # noqa: E402

m, ns, ftol = int(os.environ.get("MPOL", 2)), int(os.environ.get("NS", 15)), float(os.environ.get("FTOL", 1e-13))
inp = replace(t._deck("input.LandremanPaul2021_QA_lowres", m, ns), phiedge=-0.025, curtor=0.0, am=np.zeros(21),
              lfreeb=False, ftol_array=np.array([ftol]), niter_array=np.array([50000]))
for floor in ((None,) if os.environ.get("NOFLOOR") else (None, 1e-3)):
    kw = {} if floor is None else dict(lambda_m0_floor=floor)
    cold = opt.solve_equilibrium(inp, **kw)
    seed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                        if jnp.issubdtype(a.dtype, jnp.inexact) else a, cold.state)
    pert = opt.solve_equilibrium(inp, initial_state=seed, **kw)
    names = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    diffs = {n: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
             for n, a, b in zip(names, jax.tree.leaves(pert.state), jax.tree.leaves(cold.state))}
    iota_c, iota_p = np.asarray(cold.wout.iotaf), np.asarray(pert.wout.iotaf)
    print(f"lambda_m0_floor={floor}: " + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items() if v)
          + f"; iota max diff {np.max(np.abs(iota_c - iota_p)):.2e}; fsq cold "
          f"{float(cold.wout.fsqr):.1e}/{float(cold.wout.fsqz):.1e}/{float(cold.wout.fsql):.1e}", flush=True)


def curves(wout, js, nphi=8, nth=2048):
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    rmnc, zmns = np.asarray(wout.rmnc)[js], np.asarray(wout.zmns)[js]
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    out = []
    for ph in np.linspace(0, 2 * np.pi / int(wout.nfp), nphi, endpoint=False):
        a = xm[None, :] * th[:, None] - xn[None, :] * ph
        out.append((np.cos(a) @ rmnc, np.sin(a) @ zmns))
    return out


def curve_distance(c1, c2):
    """Largest distance from points of c1 to the polyline c2 (dense sampling, nearest point)."""
    worst = 0.0
    for (R1, Z1), (R2, Z2) in zip(c1, c2):
        P = np.stack([R1, Z1], 1)[::8]
        Q = np.stack([R2, Z2], 1)
        Q1 = np.roll(Q, -1, axis=0)
        seg = Q1 - Q
        t = np.clip(np.einsum("psk,sk->ps", P[:, None, :] - Q[None, :, :], seg) / np.sum(seg**2, 1)[None], 0, 1)
        proj = Q[None] + t[..., None] * seg[None]
        d = np.min(np.linalg.norm(P[:, None, :] - proj, axis=2), axis=1)  # distance to the polyline's segments
        worst = max(worst, float(d.max()))
    return worst


ns_w = int(cold.wout.ns)
for js in (ns_w // 4, ns_w // 2, 3 * ns_w // 4, ns_w - 2):
    cc, cp = curves(cold.wout, js), curves(pert.wout, js)
    coeff = float(np.max(np.abs(np.asarray(cold.wout.rmnc)[js] - np.asarray(pert.wout.rmnc)[js])))
    print(f"js {js}: surface curve distance {curve_distance(cc, cp):.2e} m (segment chord error ~{(2 * np.pi * 0.1 / 2048) ** 2 / 0.8:.0e});"
          f" largest R_mn coefficient change {coeff:.2e}", flush=True)
