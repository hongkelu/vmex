"""Diagnostics of the three-term start dependence (CTH-like vacuum deck, tricubic mgrid field).  Measurement only.

    python hdiag.py OUT base    # H3: deck and NESTOR starts; stationarity, Jacobian spectrum, longer fits
    python hdiag.py OUT quad    # H2: both starts with the same fixed singular quadrature
    python hdiag.py OUT path    # H4: the interface cost along the straight path between the base solutions
    python hdiag.py OUT starts W [beta]   # six starts at label_weight W (vacuum, or the finite-beta deck)

Every boundary is scored by one common scorer: a fixed-boundary equilibrium on it, the three interface conditions on
a 64 x 64 grid with one singular quadrature (4 nfp 64, 128) planned once on the deck-start solution, and field
lines (the vacuum ground truth: RMS distance of their crossings from the LCFS, coil B.n/|B| on it).
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402
from vmex.core.multigrid import _nestor_boundary  # noqa: E402

sys.path.insert(0, "tests")
from test_freeboundary_vc import _vacuum_flux_surface  # noqa: E402

out, mode = Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
beta = mode == "starts" and sys.argv[-1] == "beta"
inp = replace(vj.VmecInput.from_file("examples/data/input.cth_like_free_bdy"),
              mgrid_file="examples/data/mgrid_cth_like.nc")
if not beta:
    inp = replace(inp, pres_scale=0.0, curtor=0.0)
field = vj.MgridField.from_input(inp, inp.mgrid_file, order=3)
fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
N = 64
_plan = {}


def log(row):
    print(json.dumps(row), flush=True)
    with open(out / f"{mode}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def deck_of(rbc, zbs, raxis, zaxis):
    return replace(fixed, rbc=np.asarray(rbc), zbs=np.asarray(zbs), raxis_c=np.asarray(raxis),
                   zaxis_s=np.asarray(zaxis))


def score(deck, tag, **extra):
    """The common scorer on the fixed-boundary equilibrium of ``deck``."""
    eq = opt.solve_equilibrium(deck)
    data = vc.surface_field_data_from_state(deck, eq.solution, runtime=eq.solver_context, nphi=N, ntheta=N)
    if "plan" not in _plan:
        _plan["plan"] = vc.plan_vc_precision(data, digits=4, quad_nt=4 * int(deck.nfp) * N, quad_np=2 * N)
    rows, weights = fvc.boundary_residual(deck, eq.solution, field, runtime=eq.solver_context, nphi=N, ntheta=N,
                                          precision=_plan["plan"])
    res = fvc.summarize_boundary_residual(rows, weights)
    bn, drift = (np.nan, np.nan) if beta else _vacuum_flux_surface(eq.wout, field, lines=32, transits=30, steps=128)
    row = dict(tag=tag, bn=res.normal, pressure=res.pressure, sheet=res.sheet_current,
               cost=0.5 * float(np.sum(np.asarray(weights) * np.sum(np.asarray(rows) ** 2, axis=0))),
               coil_bn=bn, fieldline_drift_mm=1e3 * drift, iota_axis=float(np.asarray(eq.wout.iotaf)[0]),
               iota_edge=float(np.asarray(eq.wout.iotaf)[-1]), r00=float(np.asarray(eq.wout.rmnc)[-1][0]), **extra)
    log(row)
    return eq.wout


def save(tag, fit):
    d = fit.input
    np.savez(out / f"{tag}.npz", rbc=d.rbc, zbs=d.zbs, raxis=d.raxis_c, zaxis=d.zaxis_s)


def stationarity(tag, fit):
    """How stationary the fit is, and the spectrum of its last Jacobian (in the model's scaled coordinates)."""
    J, r = np.asarray(fit.jac), np.asarray(fit.fun)
    sv = np.linalg.svd(J / np.maximum(np.linalg.norm(J, axis=0), 1e-300), compute_uv=False)
    g = J.T @ r
    log(dict(tag=tag, cost=float(fit.cost), grad_norm=float(np.linalg.norm(g)),
             grad_rel=float(np.linalg.norm(g) / (np.linalg.norm(J) * np.linalg.norm(r))),
             sv_cond=float(sv[0] / sv[-1]), sv_smallest_rel=(sv[-8:] / sv[0]).tolist(), n=int(J.shape[1]),
             nfev=int(fit.nfev), njev=int(fit.njev), converged=bool(fit.converged), message=fit.message))


import os  # noqa: E402
LABELLING = os.environ.get("LABELLING", "start")


def fit(start, **kw):
    kw.setdefault("labelling", LABELLING)
    t = time.perf_counter()
    result = fvc.solve_free_boundary_three_term(inp, external_field=field, initial_boundary=start, **kw)
    return result, round(time.perf_counter() - t)


nestor_start = _nestor_boundary(inp)
starts = {"deck": inp, "nestor": nestor_start}
if mode == "base":
    for name, start in starts.items():
        f, s = fit(start)
        save(name, f)
        stationarity(f"{name}: default fit", f)
        score(f.input, f"{name}: default fit", seconds=s)
        # H3: the same fit continued with fresh Jacobians to a tight tolerance
        g, s = (lambda t0: (fvc.solve_free_boundary_three_term(inp, external_field=field, previous=f, ftol=1e-12,
                                                                jacobian_ftol=1e-12, max_nfev=200),
                            round(time.perf_counter() - t0)))(time.perf_counter())
        save(f"{name}-long", g)
        stationarity(f"{name}: continued to ftol 1e-12", g)
        score(g.input, f"{name}: continued to ftol 1e-12", seconds=s)
elif mode == "quad":
    for name, start in starts.items():
        f, s = fit(start, quadrature=(4 * int(inp.nfp) * 48, 96))
        save(f"{name}-quad", f)
        stationarity(f"{name}: fixed quadrature", f)
        score(f.input, f"{name}: fixed quadrature", seconds=s)
elif mode == "starts":
    W = float(sys.argv[3])
    def scaled(f):
        r, z = np.array(inp.rbc), np.array(inp.zbs); r[:, 1:] *= f; z[:, 1:] *= f
        return replace(inp, rbc=r, zbs=z)
    r = np.array(inp.rbc); r[inp.ntor, 0] += 0.01
    rng = np.random.default_rng(451)
    rp, zp = np.array(inp.rbc), np.array(inp.zbs)
    rp[:, 1:] *= 1 + 0.02 * rng.standard_normal(rp[:, 1:].shape)
    zp[:, 1:] *= 1 + 0.02 * rng.standard_normal(zp[:, 1:].shape)
    six = {"deck": inp, "nestor": nestor_start, "m>=1 x0.97": scaled(0.97), "m>=1 x1.03": scaled(1.03),
           "shifted +1 cm": replace(inp, rbc=r), "random 2%": replace(inp, rbc=rp, zbs=zp)}
    walls = {}
    for name, start in six.items():
        f, s = fit(start, label_weight=W)
        save(f"{name.replace(' ', '_').replace('%', 'pct').replace('>=', 'ge')}-W{W:g}", f)
        stationarity(f"{name}: label {W:g}", f)
        walls[name] = score(f.input, f"{name}: label {W:g}", seconds=s, label_weight=W)
    ref = walls["nestor"]
    from test_freeboundary_vc import _lcfs_distance
    log(dict(tag=f"LCFS distance to the NESTOR start's fit, mm (label {W:g})",
             **{k: 1e3 * _lcfs_distance(w, ref) for k, w in walls.items()}))
elif mode == "path":
    a, b = (np.load(out.parent / "base" / f"{k}.npz") for k in ("deck", "nestor"))
    for s in np.linspace(0, 1, 11):
        mix = {k: (1 - s) * a[k] + s * b[k] for k in ("rbc", "zbs", "raxis", "zaxis")}
        score(deck_of(**mix), f"path s={s:.1f}", s=float(s))
