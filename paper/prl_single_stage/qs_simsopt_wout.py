"""Gil's exact QS metric code on a wout file: simsopt QuasisymmetryRatioResidual, helicity (1, 0), 51 surfaces.

    python qs_simsopt_wout.py <wout.nc> [<out.json>]

simsopt's Vmec in wout-only mode needs no VMEC2000 extension.  Run with simsopt-env
(MPICH_GPU_SUPPORT_ENABLED=0).
"""
import json
import sys

import numpy as np
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec

wout = sys.argv[1]
v = Vmec(wout)
s51 = np.arange(0, 1.01, 1.01 / 50)
s10 = np.arange(1, 11) / 10
qs51 = QuasisymmetryRatioResidual(v, s51, helicity_m=1, helicity_n=0)
qs10 = QuasisymmetryRatioResidual(v, s10, helicity_m=1, helicity_n=0)
rec = dict(wout=wout, qs_total_51=float(qs51.total()), qs_profile_51=[float(x) for x in qs51.profile()],
           qs_total_10=float(qs10.total()), qs_profile_10=[float(x) for x in qs10.profile()],
           iota_axis=float(v.wout.iotaf[0]), iota_edge=float(v.wout.iotaf[-1]), aspect=float(v.wout.aspect))
print(f"simsopt QS on {wout}: QS(51) {rec['qs_total_51']:.3e}, QS(10) {rec['qs_total_10']:.3e}", flush=True)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(rec, indent=1) + "\n")
