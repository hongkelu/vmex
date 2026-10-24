"""Weak-subspace preconditioned stationarity gradient against the full (Gauss-Newton preconditioned) one, same seed
and step: per row |sub - full| / |full|, GN error for scale, CG iterations and evaluations.

usage: python subspace_vs_full.py FULL_DIR SUB_DIR [SUB_DIR ...]   (stationarity_gradient_check.json in each)
"""
import json
import sys
from pathlib import Path

import numpy as np

full = json.loads(Path(sys.argv[1], "stationarity_gradient_check.json").read_text())
F, GN = np.array(full["gradients"]), np.array(full["gauss_newton"])
print(f"{Path(sys.argv[1]).name}: h {full['h']}, CG {full['cg_iterations']}, {full['evaluations']} evaluations, "
      f"{full['seconds']:.0f} s")
for d in sys.argv[2:]:
    sub = json.loads(Path(d, "stationarity_gradient_check.json").read_text())
    S = np.array(sub["gradients"])
    n = np.linalg.norm(F, axis=1)
    err = np.linalg.norm(S - F, axis=1) / n
    gn = np.linalg.norm(GN - F, axis=1) / n
    print(f"  {Path(d).name}: k {sub.get('subspace')}, CG {sub['cg_iterations']}, {sub['evaluations']} evaluations "
          f"({sub.get('subspace_evaluations')} subspace), {sub['seconds']:.0f} s; true residual max "
          f"{max(sub.get('true_residual', [np.nan])):.1e}")
    print("    sub vs full per row: " + " ".join(f"{e:.1e}" for e in err) + f"   (max {err.max():.1e}; GN max "
          f"{gn.max():.1e})")
