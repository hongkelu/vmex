"""Gil's QS step with vmex instead of VMEC2000: fixed-boundary solve of a QFM surface + 51-surface QS.

    python qfm_qs_vmex.py <qfm_surface.json> <outdir>

Reads the QFM boundary written by gil_postprocess.py, puts it into the deck it
names (truncated to the deck's MPOL/NTOR, as simsopt's Vmec does), solves the
fixed-boundary equilibrium with vmex and evaluates the two-term quasisymmetry
residual (simsopt convention) on Gil's 51 surfaces np.arange(0, 1.01, 1.01/50)
and on our 10 surfaces 0.1..1.0.  Writes <outdir>/wout_qfm.nc and qfm_qs.json.
GPU or CPU; run inside an allocation.
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

q = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

import vmex as vj
from vmex import optimize as opt

inp = vj.VmecInput.from_file(q["input"])
mpol, ntor = int(inp.mpol), int(inp.ntor)
rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
dropped = 0
for m, n, v in q["rc"]:
    if m < mpol and abs(n) <= ntor:
        rbc[n + ntor, m] = v
    else:
        dropped += 1
for m, n, v in q["zs"]:
    if m < mpol and abs(n) <= ntor:
        zbs[n + ntor, m] = v
    else:
        dropped += 1
inp = replace(inp, rbc=rbc, zbs=zbs, lfreeb=False)
t0 = time.time()
eq = opt.solve_equilibrium(inp)
w = eq.wout
vj.write_wout(str(out / "wout_qfm.nc"), w)
s51 = np.arange(0, 1.01, 1.01 / 50)
s10 = np.arange(1, 11) / 10
qs51 = opt.QuasisymmetryRatioResidual(s51, 1, 0)
qs10 = opt.QuasisymmetryRatioResidual(s10, 1, 0)
rec = dict(qfm=sys.argv[1], deck=q["input"], mpol=mpol, ntor=ntor, modes_dropped=dropped, seconds=time.time() - t0,
           qs_total_51=float(qs51.total(w)), qs_profile_51=np.asarray(qs51.profile(w)).tolist(), qs_surfaces_51=s51.tolist(),
           qs_total_10=float(qs10.total(w)), qs_profile_10=np.asarray(qs10.profile(w)).tolist(),
           iota_axis=float(np.asarray(w.iotaf)[0]), iota_edge=float(np.asarray(w.iotaf)[-1]), aspect=float(w.aspect),
           volume=float(w.volume_p))
(out / "qfm_qs.json").write_text(json.dumps(rec, indent=1) + "\n")
print(f"vmex on QFM surface: QS(51) {rec['qs_total_51']:.3e}, QS(10) {rec['qs_total_10']:.3e}, iota {rec['iota_axis']:.4f}->{rec['iota_edge']:.4f}, A {rec['aspect']:.3f}, {dropped} modes dropped, {rec['seconds']:.0f} s", flush=True)
