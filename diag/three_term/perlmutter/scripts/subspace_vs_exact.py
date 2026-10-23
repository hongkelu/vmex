"""Error of the low-rank (subspace) design gradient against the converged exact stationarity gradient, per k.

usage: python subspace_vs_exact.py SUBSPACE_DIR [SUBSPACE_DIR ...] --exact EXACT_DIR [--row R]
Each SUBSPACE_DIR holds subspace_gradient_check.json (driver --check-subspace-gradient); EXACT_DIR holds
exact_gradient_check.json from the same seed (driver --check-exact-gradient; its row, default 0). Prints, per k,
the relative error |g - exact| / |exact| of the subspace gradient and of Gauss-Newton, and the seconds.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--exact", required=True)
    ap.add_argument("--row", type=int, default=0)
    a = ap.parse_args()
    ex = np.array(json.loads(Path(a.exact, "exact_gradient_check.json").read_text())["exact"])
    n = np.linalg.norm(ex)
    for d in a.dirs:
        f = Path(d, "subspace_gradient_check.json")
        if not f.exists():
            print(f"{d}: (no result yet)")
            continue
        for run in json.loads(f.read_text())["runs"]:
            g, gn = np.array(run["gradients"])[a.row], np.array(run["gauss_newton"])[a.row]
            print(f"{Path(d).name:24s} k={run['k']:3d}: subspace {np.linalg.norm(g - ex) / n:.3e}  "
                  f"GN {np.linalg.norm(gn - ex) / n:.3e}  ({run['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
