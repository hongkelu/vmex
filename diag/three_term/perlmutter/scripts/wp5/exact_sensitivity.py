"""C3: exact stationarity-derivative reference for the small design-gradient problem (design_gradient_fd.py).

usage: python exact_sensitivity.py OUT.json   (cwd: a vmex tree; a GPU; env FTOL (default 1e-9))

At the certified fit b*(d) of design d = (0, 0.1, 0.3) of design_gradient_fd.py's problem, with central
differences of tight evaluations at FIXED boundaries (no fits):
    g(b, d) = J_b(b, d)^T r(b, d)              (the fit's stationarity function, from model.linearize)
    H   = dg/db = J_b^T J_b + S_bb             (column j: central difference along b_j)
    F_d = dg/dd = J_b^T J_d + S_bd             (column k: central difference along d_k)
    Q_b, Q_d: the loss at the state of (b, d)  (central differences)
    exact: dQ/dd = Q_d - Q_b H^-1 F_d;   Gauss-Newton: H, F_d without S (what the problem differentiates with).
Compared with the problem's gradient (problem.value_and_grad) and with the re-solved FD of design_grad.json.
Also reported: the relative size of S_bb against J_b^T J_b, and the symmetry of the differenced H.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_gradient_fd import build  # noqa: E402


def main():
    """Build the exact reference and compare; write OUT.json."""
    ftol = float(os.environ.get("FTOL", 1e-9))
    problem = build(ftol)
    model = problem.model
    x = np.array([0.0, 0.1, 0.3])
    _, grad = problem.value_and_grad(x)
    sol = problem._record(x)
    b = np.asarray(sol["x"], dtype=float)
    nb = b.size

    def at(bb, xx):
        xj = jnp.asarray(xx)
        params, field = problem._params_at(xj), problem.field(xj)
        rows, aux = model.evaluate(bb, params, field, tight=True)
        J = np.asarray(model.linearize(bb, aux, field)[0])[:, :nb]
        state, _, params_b, _ = aux
        loss = float(np.asarray(problem._rows(state, params_b, xj))[0])
        return np.asarray(rows), J, loss

    r0, J0, q0 = at(b, x)
    g0 = J0.T @ r0
    scale = np.asarray(model.x_scale, dtype=float)
    hb, hd = 1e-4, 1e-4
    H = np.zeros((nb, nb))
    Q_b = np.zeros(nb)
    for j in range(nb):
        e = np.zeros(nb)
        e[j] = hb * scale[j]
        (rp, Jp, qp), (rm, Jm, qm) = at(b + e, x), at(b - e, x)
        H[:, j] = (Jp.T @ rp - Jm.T @ rm) / (2 * e[j])
        Q_b[j] = (qp - qm) / (2 * e[j])
    F_d = np.zeros((nb, x.size))
    Q_d = np.zeros(x.size)
    J_d = np.zeros((r0.size, x.size))
    for k in range(x.size):
        e = np.zeros(x.size)
        e[k] = hd
        (rp, Jp, qp), (rm, Jm, qm) = at(b, x + e), at(b, x - e)
        F_d[:, k] = (Jp.T @ rp - Jm.T @ rm) / (2 * hd)
        Q_d[k] = (qp - qm) / (2 * hd)
        J_d[:, k] = (rp - rm) / (2 * hd)
    gn_H = J0.T @ J0
    exact = Q_d - Q_b @ np.linalg.solve(H, F_d)
    gauss_newton = Q_d - Q_b @ np.linalg.solve(gn_H, J0.T @ J_d)
    reference = None
    ref_file = Path(os.environ.get("RESOLVED", ""))
    if ref_file.is_file():
        rows = json.loads(ref_file.read_text())
        tight = [r for r in rows if r["boundary_ftol"] == min(rr["boundary_ftol"] for rr in rows)][0]
        reference = tight["differences"]["0.001"]

    def rel(a, b_):
        return (np.abs(np.asarray(a) - np.asarray(b_)) / np.maximum(np.abs(np.asarray(b_)), 1e-12)).tolist()

    out = dict(boundary_ftol=ftol, stationarity_gradient_norm=float(np.linalg.norm(g0)),
               S_bb_relative=float(np.linalg.norm(H - gn_H) / np.linalg.norm(gn_H)),
               H_asymmetry=float(np.linalg.norm(H - H.T) / np.linalg.norm(H)),
               problem_gradient=np.asarray(grad).tolist(), exact=exact.tolist(),
               gauss_newton_by_differences=gauss_newton.tolist(), resolved_fd=reference,
               exact_vs_resolved=None if reference is None else rel(exact, reference),
               problem_vs_resolved=None if reference is None else rel(grad, reference),
               gauss_newton_vs_problem=rel(gauss_newton, grad))
    print(json.dumps(out, indent=1), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
