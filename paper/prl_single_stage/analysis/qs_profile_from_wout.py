"""Two-term QS profile (simsopt convention, helicity (1,0)) of wout files, on CPU.

    JAX_PLATFORMS=cpu python qs_profile_from_wout.py <out.json> <label=wout.nc> [<label=wout.nc> ...]

Writes {label: {"s": [...], "profile": [...], "total": x, "iota_axis":..., "iota_edge":..., "aspect":...}}
on the 10 surfaces s = 0.1..1.0 and on Gil's 51 surfaces.
"""
import json
import sys

import numpy as np

out = sys.argv[1]
import vmex as vj
from vmex import optimize as opt

s10 = np.arange(1, 11) / 10
s51 = np.arange(0, 1.01, 1.01 / 50)
qs10 = opt.QuasisymmetryRatioResidual(s10, 1, 0)
qs51 = opt.QuasisymmetryRatioResidual(s51, 1, 0)
res = {}
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    w = vj.read_wout(path)
    res[label] = dict(wout=path, s10=s10.tolist(), profile10=np.asarray(qs10.profile(w)).tolist(), total10=float(qs10.total(w)),
                      s51=s51.tolist(), profile51=np.asarray(qs51.profile(w)).tolist(), total51=float(qs51.total(w)),
                      iota_axis=float(np.asarray(w.iotaf)[0]), iota_edge=float(np.asarray(w.iotaf)[-1]),
                      aspect=float(w.aspect), beta=float(w.betatotal))
    print(f"{label:22s} QS10 {res[label]['total10']:.3e}  QS51 {res[label]['total51']:.3e}  iota {res[label]['iota_axis']:.3f}->{res[label]['iota_edge']:.3f}  A {res[label]['aspect']:.3f}", flush=True)
json.dump(res, open(out, "w"), indent=1)
