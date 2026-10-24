"""How much of a certificate's remaining Gauss-Newton gain is relabelling, and how much is physical (option 5 offline).

usage: python certificate_split.py RECORD.npz INPUT_FILE [RECORD.npz INPUT_FILE ...]   (CPU; one JSON line each)

The certificate (freeboundary_vc._fit_certificate) is ``|Q^T r|^2 / |r|^2`` for the scaled boundary Jacobian
``J = J[:, :n] * x_scale`` (n: the model coordinates, boundary then bootstrap current) and the record's rows ``r``. This script first recomputes it from the stored arrays and
refuses a record whose recomputation differs from the stored value (relative 1e-8): the split below is of that same
quantity. (gauge_offline.py's ``gain_full`` is not: it removes every non-boundary column from the rows first and
normalizes by the reduced residual.)

Relabelling directions: theta -> theta + f with f = sin(m theta - n nfp phi) over the boundary's modes (odd f keeps
stellarator symmetry), dR = -R_theta f, dZ = -Z_theta f, least-squares projected onto the boundary coordinates
(scaled), as in gauge_offline.py. ``T`` is their span (orthonormal), ``B`` its complement in the scaled boundary
space. Reported:
- ``stationarity``: the certificate (recomputed = stored);
- ``physical``: the gain of the Gauss-Newton step restricted to ``B``, ``|Q_{JB}^T r|^2 / |r|^2``: what a
  certificate that ignores relabelling would measure;
- ``relabel_share``: ``1 - physical / stationarity`` (the share of the remaining gain only relabelling can take);
- ``step`` and its parts in T and B (scaled length of the unconstrained step ``J^+ r``);
- ``relabel_projection_error``: how well the boundary coordinates represent the relabellings (median, max).
"""
import json
import sys

import numpy as np


def relabel_basis(rbc, zbs, nfp, max_mode, surface):
    """Scaled-free (raw) boundary-coordinate directions of the relabellings, and their projection errors."""
    mmax, nmax = rbc.shape[1] - 1, (rbc.shape[0] - 1) // 2
    th = np.linspace(0, 2 * np.pi, 6 * (mmax + 1) + 6, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, 6 * (2 * nmax + 1) + 6, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n_idx, m_idx = np.arange(rbc.shape[0]) - nmax, np.arange(rbc.shape[1])
    A = m_idx[None, None, None, :] * TH[..., None, None] - (n_idx * nfp)[None, None, :, None] * PH[..., None, None]
    C_, S_ = np.cos(A), np.sin(A)
    R_t = np.einsum("tpnm,nm->tp", -S_ * m_idx, rbc)
    Z_t = np.einsum("tpnm,nm->tp", C_ * m_idx, zbs)
    basis = surface(C_, S_)
    proj = np.linalg.pinv(basis)
    T, errors = [], []
    for i in range(rbc.shape[0]):
        for m in range(rbc.shape[1]):
            if (m == 0 and n_idx[i] <= 0) or m > max_mode or abs(n_idx[i]) > max_mode:
                continue
            f = np.sin(m * TH - n_idx[i] * nfp * PH)
            target = np.r_[(-R_t * f).ravel(), (-Z_t * f).ravel()]
            coef = proj @ target
            errors.append(float(np.linalg.norm(basis @ coef - target) / np.linalg.norm(target)))
            T.append(coef)
    return np.array(T).T, errors


def split(path, deck):
    """The split of one record's certificate (``deck``: the run's input.run, for the coordinate map)."""
    from dataclasses import replace

    import vmex as vj
    from vmex.core.optimize import boundary_arrays_from_x, pack_boundary

    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    nb = dict((b[0], (b[1], b[2])) for b in meta["column_blocks"])["boundary"][1]
    # the certificate's columns are the model coordinates: the boundary, then any bootstrap-current values
    nc = int(meta["resolution"].get("n_coordinates", nb))
    J = np.asarray(d["J"])[:, :nc] * np.asarray(d["x_scale"], float)[:nc]
    r = np.asarray(d["rows"])
    Q, _ = np.linalg.qr(J)
    stationarity = float(np.sum((Q.T @ r) ** 2) / (r @ r))
    stored = float(meta["certificate"]["stationarity"])
    if abs(stationarity - stored) > 1e-8 * max(stored, 1e-300):
        raise SystemExit(f"{path}: recomputed certificate {stationarity:.6e} != stored {stored:.6e}; not splitting")
    res = meta["resolution"]
    max_mode, nfp = int(res["max_mode"]), int(res["nfp"])
    rbc, zbs = np.asarray(d["rbc"]), np.asarray(d["zbs"])
    inp = replace(vj.VmecInput.from_file(deck), rbc=rbc, zbs=zbs)  # the run's deck with the record's boundary
    x_b = np.asarray(pack_boundary(inp, max_mode, vary_major_radius=True))
    if x_b.size != nb:
        raise SystemExit(f"{path}: {x_b.size} packed coordinates != {nb} boundary columns")

    def surface(C_, S_):
        def at(x):
            rb, zb = map(np.asarray, boundary_arrays_from_x(inp, x, max_mode, vary_major_radius=True))
            return np.einsum("tpnm,nm->tp", C_, rb).ravel(), np.einsum("tpnm,nm->tp", S_, zb).ravel()
        R0, Z0 = at(x_b)
        cols = []
        for k in range(nb):
            e = np.zeros(nb)
            e[k] = 1.0
            R1, Z1 = at(x_b + e)
            cols.append(np.r_[R1 - R0, Z1 - Z0])
        return np.array(cols).T

    T_raw, errors = relabel_basis(rbc, zbs, nfp, max_mode, surface)
    Ts = T_raw / np.asarray(d["x_scale"], float)[:nb, None]  # scaled coordinates
    Qt, _ = np.linalg.qr(Ts)
    U, sv, _ = np.linalg.svd(np.eye(nb) - Qt @ Qt.T)
    B = U[:, sv > 0.5]  # orthonormal complement of the relabelling span (boundary coordinates)
    # current coordinates stay free in both the full and the physical step
    B = np.block([[B, np.zeros((nb, nc - nb))], [np.zeros((nc - nb, B.shape[1])), np.eye(nc - nb)]])
    Qt = np.vstack([Qt, np.zeros((nc - nb, Qt.shape[1]))])
    Qb, _ = np.linalg.qr(J @ B)
    physical = float(np.sum((Qb.T @ r) ** 2) / (r @ r))
    step = np.linalg.lstsq(J, r, rcond=None)[0]
    return dict(record=path, stationarity=stationarity, physical=physical,
                relabel_share=1 - physical / stationarity if stationarity > 0 else 0.0,
                step=float(np.linalg.norm(step)), step_relabel=float(np.linalg.norm(Qt.T @ step)),
                step_physical=float(np.linalg.norm(B.T @ step)), n_relabel=int(Qt.shape[1]),
                n_physical=int(B.shape[1]), n_current=int(nc - nb), relabel_projection_error=[float(np.median(errors)), float(max(errors))],
                label_rows="label" in [b[0] for b in meta["row_blocks"]])


if __name__ == "__main__":
    args = sys.argv[1:]
    for p, deck in zip(args[::2], args[1::2]):
        print(json.dumps(split(p, deck)), flush=True)
