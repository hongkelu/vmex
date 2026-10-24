"""stationarity_design_gradient (all rows, reverse-pass products) against exact_design_gradient (one row,
linearization products): two implementations of the same derivative, from the same seed.

usage: python stationarity_vs_exact.py STATIONARITY_DIR EXACT_DIR [EXACT_DIR ...]
Each EXACT_DIR's exact_gradient_check.json names its row (``row``, default 0).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    """Print, per reference row, |stationarity - exact| / |exact| and both against Gauss-Newton."""
    st = json.loads(Path(sys.argv[1], "stationarity_gradient_check.json").read_text())
    grads, gn = np.array(st["gradients"]), np.array(st["gauss_newton"])
    rows = list(st["rows"])
    print(f"{Path(sys.argv[1]).name}: consistency {st['consistency']}, CG {st['cg_iterations']}, "
          f"{st['evaluations']} evaluations, {st['seconds']:.0f} s")
    for d in sys.argv[2:]:
        ex = json.loads(Path(d, "exact_gradient_check.json").read_text())
        row = int(ex.get("row", 0))
        e = np.array(ex["exact"])
        g = grads[rows.index(row)]
        n = np.linalg.norm(e)
        print(f"  row {row} vs {Path(d).name}: stationarity-vs-exact {np.linalg.norm(g - e) / n:.2e}   "
              f"GN-vs-exact {np.linalg.norm(gn[rows.index(row)] - e) / n:.2e}   "
              f"(exact GN copy vs this GN {np.linalg.norm(np.array(ex['gauss_newton']) - gn[rows.index(row)]) / np.linalg.norm(gn[rows.index(row)]):.1e})")


if __name__ == "__main__":
    main()
