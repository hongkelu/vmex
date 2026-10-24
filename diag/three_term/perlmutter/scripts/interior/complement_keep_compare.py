"""Compare complement_keep.py runs: usage python complement_keep_compare.py PLAIN.npz KEEP.npz"""
import json
import sys

import numpy as np

from interior_gauge import distance


def rel(a, b):
    """Largest change relative to the largest row."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


plain, keep = np.load(sys.argv[1]), np.load(sys.argv[2])
out = dict(
    keep_spread=max(rel(keep[f"rows_{k}"], keep["rows_cold"]) for k in ("near", "perturbed")),
    plain_spread=max(rel(plain[f"rows_{k}"], plain["rows_cold"]) for k in ("near", "perturbed")),
    keep_vs_plain_rows=rel(keep["rows_cold"], plain["rows_cold"]),
    keep_residuals=[float(keep[f"residual_{k}"]) for k in ("cold", "near", "perturbed")],
    iota=float(np.max(np.abs(keep["iota"] - plain["iota"]))),
    bmin=float(np.max(np.abs(keep["bmin"] - plain["bmin"]))),
    surface=max(distance(a, b) for a, b in zip(keep["surfaces"], plain["surfaces"])),
)
print(json.dumps(out))
