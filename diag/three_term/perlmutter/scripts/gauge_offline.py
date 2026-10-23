"""D2: are the weak directions of a fit's reduced Jacobian the boundary's relabelling directions?

usage: python gauge_offline.py RECORD.npz INPUT_FILE [K]   (CPU)

At the record's boundary: the relabelling directions theta -> theta + f, f = cos/sin(m theta - n Nfp phi) over the
boundary's own modes (stellarator-symmetric pairing: f odd, sin(m theta - n Nfp phi)), give dR = -R_theta f,
dZ = -Z_theta f; each is projected on the boundary coordinates (least squares on an oversampled grid; projection
error reported) to columns T (scaled coordinates).  The weakest K right singular vectors V of the column-scaled,
current-reduced Jacobian are compared with span(T): the fraction of each vector's norm inside span(T).
"""
import json
import sys
from dataclasses import replace

import numpy as np

import vmex as vj
from vmex.core.optimize import boundary_arrays_from_x, pack_boundary


def main():
    """Subspace overlap of the weak directions with the relabelling directions; print JSON."""
    data = np.load(sys.argv[1], allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    inp = vj.VmecInput.from_file(sys.argv[2])
    max_mode = int(meta["resolution"]["max_mode"])
    nb = dict((b[0], (b[1], b[2])) for b in meta["column_blocks"])["boundary"][1]
    inp = replace(inp, rbc=np.asarray(data["rbc"]), zbs=np.asarray(data["zbs"]))
    x_b = np.asarray(pack_boundary(inp, max_mode, vary_major_radius=True))
    nfp = int(inp.nfp)
    rbc, zbs = np.asarray(data["rbc"]), np.asarray(data["zbs"])
    mmax, nmax = rbc.shape[1] - 1, (rbc.shape[0] - 1) // 2
    th = np.linspace(0, 2 * np.pi, 6 * (mmax + 1) + 6, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, 6 * (2 * nmax + 1) + 6, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n_idx = np.arange(rbc.shape[0]) - nmax
    m_idx = np.arange(rbc.shape[1])
    A = m_idx[None, None, None, :] * TH[..., None, None] - (n_idx * nfp)[None, None, :, None] * PH[..., None, None]
    C_, S_ = np.cos(A), np.sin(A)
    R_t = np.einsum("tpnm,nm->tp", -S_ * m_idx, rbc)
    Z_t = np.einsum("tpnm,nm->tp", C_ * m_idx, zbs)

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
    T, errors = [], []
    for i in range(rbc.shape[0]):
        for m in range(rbc.shape[1]):
            if (m == 0 and n_idx[i] <= 0) or m > max_mode or abs(n_idx[i]) > max_mode:
                continue
            f = np.sin(m * TH - n_idx[i] * nfp * PH)  # odd: keeps stellarator symmetry
            target = np.r_[(-R_t * f).ravel(), (-Z_t * f).ravel()]
            coef = proj @ target
            errors.append(float(np.linalg.norm(basis @ coef - target) / np.linalg.norm(target)))
            T.append(coef)
    T = np.array(T).T  # raw boundary coordinates
    scale = np.asarray(data["x_scale"], dtype=float)[:nb]
    Ts = T / scale[:, None]  # scaled coordinates
    Qt, _ = np.linalg.qr(Ts)
    J = np.asarray(data["J"])
    Jb = J[:, :nb] * scale
    if J.shape[1] > nb:
        Qc, _ = np.linalg.qr(J[:, nb:])
        Jb = Jb - Qc @ (Qc.T @ Jb)
    norms = np.linalg.norm(Jb, axis=0)
    _, sv, Vt = np.linalg.svd(Jb / norms, full_matrices=False)
    V = (Vt[-K:].T / norms[:, None])
    V = V / np.linalg.norm(V, axis=0)
    inside = np.linalg.norm(Qt.T @ V, axis=0)
    rnd = np.random.default_rng(0).standard_normal((nb, 200))
    rnd /= np.linalg.norm(rnd, axis=0)
    out = dict(n_relabel=int(T.shape[1]), n_boundary=nb, relabel_projection_error_max=float(max(errors)),
               relabel_projection_error_median=float(np.median(errors)),
               weakest_sv=[float(v) for v in sv[-K:]], weak_in_relabel_span=[float(v) for v in inside],
               random_in_relabel_span_median=float(np.median(np.linalg.norm(Qt.T @ rnd, axis=0))))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
