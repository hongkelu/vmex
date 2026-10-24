"""Offline check of the normal-displacement chart (work package 3) on a saved certificate record.

usage: python chart_offline.py RECORD.npz INPUT_FILE   (CPU is enough: linear algebra on the record's Jacobian)

The chart X = X_ref + u(theta, phi) n_RZ (the unit normal of the reference cross-section in the R-Z plane at fixed
phi), u = sum_mn u_mn cos(m theta - n Nfp phi) (stellarator symmetric), with u's modes those of R.  Its linear map C
(u -> boundary coordinates x) comes from a least-squares projection of (dR, dZ) = u (n_R, n_Z) on the boundary's
modes on an oversampled grid; the projection error is reported.  Compared on the record: the singular values and
condition number of the (column-normalized) boundary Jacobian J_b and of J_b C, and the Gauss-Newton gain each
predicts (the current columns are eliminated first in both).
"""
import json
import sys
from dataclasses import replace

import numpy as np

import vmex as vj
from vmex.core.optimize import boundary_arrays_from_x, pack_boundary


def main():
    """Build the chart at the record's boundary and compare the two Jacobians."""
    data = np.load(sys.argv[1], allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    inp = vj.VmecInput.from_file(sys.argv[2])
    res = meta["resolution"]
    max_mode = int(res["max_mode"])
    nb = int(res["n_boundary"])
    inp = replace(inp, rbc=np.asarray(data["rbc"]), zbs=np.asarray(data["zbs"]))
    x_b = np.asarray(pack_boundary(inp, max_mode, vary_major_radius=True))
    assert x_b.size == nb, (x_b.size, nb)
    nfp = int(inp.nfp)
    rbc, zbs = np.asarray(data["rbc"]), np.asarray(data["zbs"])
    mmax, nmax = rbc.shape[1] - 1, (rbc.shape[0] - 1) // 2
    th = np.linspace(0, 2 * np.pi, 6 * (mmax + 1) + 6, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, 6 * (2 * nmax + 1) + 6, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n_idx = np.arange(rbc.shape[0]) - nmax
    A = np.arange(rbc.shape[1])[None, None, None, :] * TH[..., None, None] - (n_idx * nfp)[None, None, :, None] * PH[..., None, None]
    C_, S_ = np.cos(A), np.sin(A)
    R_t = np.einsum("tpnm,nm->tp", -S_ * np.arange(rbc.shape[1]), rbc)
    Z_t = np.einsum("tpnm,nm->tp", C_ * np.arange(rbc.shape[1]), zbs)
    norm = np.hypot(R_t, Z_t)
    nR, nZ = Z_t / norm, -R_t / norm  # unit normal of the cross-section in the R-Z plane (outward for the usual sign)

    # boundary coordinates -> (R, Z) on the grid (linear: the map's columns), to project chart displacements on x
    def surface(x):
        rb, zb = map(np.asarray, boundary_arrays_from_x(inp, x, max_mode, vary_major_radius=True))
        return np.einsum("tpnm,nm->tp", C_, rb).ravel(), np.einsum("tpnm,nm->tp", S_, zb).ravel()

    R0, Z0 = surface(x_b)
    basis = np.zeros((2 * R0.size, nb))
    for k in range(nb):
        e = np.zeros(nb)
        e[k] = 1.0
        R1, Z1 = surface(x_b + e)
        basis[:, k] = np.r_[R1 - R0, Z1 - Z0]
    proj = np.linalg.pinv(basis)
    # u's modes: those of R (rbc's (n, m) with the boundary coordinates' mode limits), plus the constant
    modes = [(i, m) for i in range(rbc.shape[0]) for m in range(rbc.shape[1])
             if (m > 0 or n_idx[i] >= 0) and m <= max_mode and abs(n_idx[i]) <= max_mode]
    C = np.zeros((nb, len(modes)))
    errors = np.zeros(len(modes))
    for j, (i, m) in enumerate(modes):
        u = np.cos(m * TH - n_idx[i] * nfp * PH)
        target = np.r_[(u * nR).ravel(), (u * nZ).ravel()]
        coef = proj @ target
        errors[j] = float(np.sqrt(np.mean((basis @ coef - target) ** 2) / np.mean(target**2)))
        C[:, j] = coef
    worst = float(errors.max())
    import os
    keep = errors <= float(os.environ.get("CHART_MAX_ERROR", 1.0))  # chart modes the boundary can represent
    by_m = {}
    for (i, m), e in zip(modes, errors):
        by_m.setdefault(m, []).append(e)
    print("projection RMS error by m (max over n):", {m: round(max(v), 3) for m, v in sorted(by_m.items())})
    C = C[:, keep]
    modes = [mode for mode, k in zip(modes, keep) if k]
    J, r = np.asarray(data["J"]), np.asarray(data["rows"])
    n = np.asarray(data["x"]).size
    J_b, J_c = J[:, :nb], J[:, nb:n]

    def reduced(Jb):
        if J_c.shape[1]:
            Q, _ = np.linalg.qr(J_c)
            Jb = Jb - Q @ (Q.T @ Jb)
            rr = r - Q @ (Q.T @ r)
        else:
            rr = r
        Jn = Jb / np.maximum(np.linalg.norm(Jb, axis=0), 1e-300)
        s = np.linalg.svd(Jn, compute_uv=False)
        step = np.linalg.lstsq(Jn, rr, rcond=None)[0]
        res = rr - Jn @ step
        return s, float((rr @ rr - res @ res) / (r @ r))

    s_full, gain_full = reduced(J_b)
    s_chart, gain_chart = reduced(J_b @ C)
    out = dict(n_boundary=nb, n_chart=len(modes), chart_projection_error=worst,
               full=dict(condition=float(s_full[0] / s_full[-1]), gain=gain_full,
                         smallest=[float(v) for v in s_full[-5:]]),
               chart=dict(condition=float(s_chart[0] / s_chart[-1]), gain=gain_chart,
                          smallest=[float(v) for v in s_chart[-5:]]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
