"""Same design, different labellings: certified values and stationarity gradients side by side.

usage: python labelling_invariance.py REF_DIR OTHER_DIR [OTHER_DIR ...]
Each DIR has checkpoint.npz (x, values) and stationarity_gradient_check.json.  Prints, against REF: the design
difference (must be 0), per-value relative differences, per-row gradient differences, and both GN errors for scale.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(d):
    c = np.load(Path(d, "checkpoint.npz"))
    g = json.loads(Path(d, "stationarity_gradient_check.json").read_text())
    return c["x"], c["values"], np.array(g["gradients"]), np.array(g["gauss_newton"])


x0, v0, g0, gn0 = load(sys.argv[1])
print(f"reference {Path(sys.argv[1]).name}: values {np.array2string(v0, precision=5)}")
for d in sys.argv[2:]:
    x, v, g, gn = load(d)
    n = np.linalg.norm(g0, axis=1)
    print(f"  {Path(d).name}: design diff {np.abs(x - x0).max():.1e}")
    print("    values rel:   " + " ".join(f"{e:+.1e}" for e in (v - v0) / np.abs(v0)))
    print("    gradient rel: " + " ".join(f"{e:.1e}" for e in np.linalg.norm(g - g0, axis=1) / n))
    print("    (GN vs exact, ref: " + " ".join(f"{e:.1e}" for e in np.linalg.norm(gn0 - g0, axis=1) / n) + ")")
