"""Where a refused certificate's predicted Gauss-Newton gain sits.  usage: python analyze.py cert_XXX.npz [n_current]

J is the scaled joint Jacobian (boundary columns, then n_current bootstrap columns), r the rows: interface
(6 per grid point: B.n, pressure, 3 sheet-current components, labelling), equal-arc phase rows (nphi), net current
(1), bootstrap (n_current).  Reports the cost and the GN step's predicted gain per row block, the current's
conditional gain |Q_c^T r|^2 (at fixed boundary), the reduced (boundary) gain, the GN step size per column block.
"""
import sys

import numpy as np

d = np.load(sys.argv[1])
J, r = d["J"], d["r"]
nc = int(sys.argv[2]) if len(sys.argv) > 2 else 53
m, n = J.shape
nb = n - nc
grid = 48 * 48
blocks = {"interface": slice(0, 6 * grid), "phase": slice(6 * grid, m - 1 - nc), "net current": slice(m - 1 - nc, m - nc),
          "bootstrap": slice(m - nc, m)}
cost = r @ r
step = -np.linalg.lstsq(J, r, rcond=None)[0]
r_new = r + J @ step
print(f"rows {m} cols {n} (boundary {nb}, current {nc}); cost(|r|^2) {cost:.4e}; GN predicted gain "
      f"{1 - r_new @ r_new / cost:.4f}")
for name, s in blocks.items():
    print(f"  {name:12s} rows {s.stop - s.start:6d}: |r|^2 {r[s] @ r[s]:.3e} ({r[s] @ r[s] / cost:.3f})  after GN "
          f"{r_new[s] @ r_new[s]:.3e}")
Qc, _ = np.linalg.qr(J[:, nb:])
gc = Qc.T @ r
print(f"current alone (fixed boundary): conditional gain {gc @ gc / cost:.4f}")
Jb = J[:, :nb] - Qc @ (Qc.T @ J[:, :nb])
rr = r - Qc @ gc
sb = -np.linalg.lstsq(Jb, rr, rcond=None)[0]
rb = rr + Jb @ sb
print(f"reduced (boundary, current re-solved): gain {(rr @ rr - rb @ rb) / cost:.4f}")
print(f"GN step norm: boundary {np.linalg.norm(step[:nb]):.3e}, current {np.linalg.norm(step[nb:]):.3e}")
sv = np.linalg.svd(J, compute_uv=False)
print(f"singular values: max {sv[0]:.3e} min {sv[-1]:.3e}; boundary block cond "
      f"{np.linalg.cond(J[:, :nb]):.3e}; current block cond {np.linalg.cond(J[:, nb:]):.3e}")
U, s, Vt = np.linalg.svd(J, full_matrices=False)
g = U.T @ r
gain = g**2 / cost
order = np.argsort(s)
cum = np.cumsum(gain[order])
for k in (1, 3, 10, 30, 100):
    print(f"gain from the {k:3d} smallest singular directions: {cum[k - 1]:.4f} (s <= {s[order][k - 1]:.2e}); "
          f"their GN step length {np.linalg.norm((g / s)[order][:k]):.3e}")
