"""How much do the three-term interface rows see an exact interior relabelling that leaves the boundary fixed?

usage: python relabel_rows.py OUT.json   (cwd: a vmex tree; a GPU; env MPOL, NS (small LP QA deck), FTOL)

At the converged cold state of one boundary: theta' = theta + f with f = eps s (1 - s) sin(m theta - n Nfp phi)
(f = 0 at the boundary and on the axis; stellarator symmetric), applied exactly on an oversampled grid and projected
(projection error reported), mapped back to the state.  Reported per (m, n, eps): the force residual of the
relabelled state and the change of the rows, by block, relative to their RMS.  In the continuum the boundary field
does not change: a change linear in eps that does not fall with ns is the boundary-field evaluation's gauge
sensitivity.
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core.restart import state_from_wout  # noqa: E402
from vmex.core.solver import evaluate_forces  # noqa: E402


def relabel(w, eps, m_f, n_f):
    """The wout relabelled by theta' = theta + eps s(1-s) sin(m_f theta - n_f Nfp phi), projected on its modes."""
    xm, xn, nfp = np.asarray(w.xm, float), np.asarray(w.xn, float), int(w.nfp)
    nth = 8 * (int(xm.max()) + m_f + 1) + 8
    nph = 8 * (2 * int(np.abs(xn).max() // max(nfp, 1)) + 2 * n_f + 1) + 8
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    A = xm[None, None, :] * TH[..., None] - xn[None, None, :] * PH[..., None]
    bc, bs = np.cos(A).reshape(-1, xm.size), np.sin(A).reshape(-1, xm.size)
    pc, ps = np.linalg.pinv(bc), np.linalg.pinv(bs)
    ns = int(w.ns)
    s_full = np.linspace(0, 1, ns)
    s_half = np.r_[0.0, 0.5 * (s_full[1:] + s_full[:-1])]
    out = {k: np.array(getattr(w, k), dtype=float) for k in ("rmnc", "zmns", "lmns")}
    worst = 0.0
    for js in range(ns):
        for key, s_val, kind in (("rmnc", s_full[js], "c"), ("zmns", s_full[js], "s"), ("lmns", s_half[js], "s")):
            if key == "lmns" and js == 0:
                continue
            f = eps * s_val * (1 - s_val) * np.sin(m_f * TH - n_f * nfp * PH)
            shifted = (TH - f)[..., None] * xm[None, None, :] - xn[None, None, :] * PH[..., None]
            coef = np.asarray(getattr(w, key))[js]
            values = (np.cos(shifted) if kind == "c" else np.sin(shifted)) @ coef
            if key == "lmns":
                values = values - f
            proj = (pc if kind == "c" else ps) @ values.ravel()
            worst = max(worst, float(np.max(np.abs((bc if kind == "c" else bs) @ proj - values.ravel()))))
            out[key][js] = proj
    return dataclasses.replace(w, **out), worst


def main():
    """Relabel, recompute the rows, write OUT.json."""
    inp, field = t._small_vacuum()
    m, ns = int(os.environ.get("MPOL", 2)), int(os.environ.get("NS", 7))
    inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_lowres", m, ns), phiedge=-0.025, curtor=0.0,
                              am=np.zeros(21))
    inp = dataclasses.replace(inp, ftol_array=np.array([float(os.environ.get("FTOL", 1e-13))]),
                              niter_array=np.array([50000]))
    model = fvc.ThreeTermFreeBoundaryModel(dataclasses.replace(inp, lfreeb=False), **t.SMALL)
    x, params = np.asarray(model.x0), model.params0
    rows0, aux = model.evaluate(x, params, field, tight=True)
    state, _, params_x, _ = aux
    eq = opt.solve_equilibrium(model.fixed, initial_state=state)
    rt = eq.solver_context
    base = np.asarray(model._rows(jax.device_put(eq.solution, jax.devices()[0]), params_x, field))
    out = dict(resolution=model.resolution(), cases=[])
    for m_f, n_f in ((0, 1), (1, 0), (1, 1), (2, 1)):
        for eps in (1e-4, 1e-3, 1e-2):
            w, proj_err = relabel(eq.wout, eps, m_f, n_f)
            relabelled = jax.device_put(state_from_wout(w, inp=model.fixed), jax.devices()[0])
            _, res, _ = evaluate_forces(relabelled, rt)
            rows = np.asarray(model._rows(relabelled, params_x, field))
            row = dict(m=m_f, n=n_f, eps=eps, projection_error=proj_err, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
                       fsql=float(res.fsql), rows_max_rel=float(np.max(np.abs(rows - base)) / np.max(np.abs(base))))
            for block, start, stop in model.row_blocks:
                d = rows[start:stop] - base[start:stop]
                row[block] = float(np.sqrt(np.mean(d**2)) / max(np.sqrt(np.mean(base[start:stop] ** 2)), 1e-300))
            out["cases"].append(row)
            print(json.dumps({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
