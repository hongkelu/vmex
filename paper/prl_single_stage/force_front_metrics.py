"""Coil metrics in the Gil-Fig-1 / Hurwitz convention for any ESSOS coil file at R0 = 1 m: max force (SIMSOPT coil_force,
regularization_circ(0.05 m), currents rescaled to mean |B| = 0.2271 T on the LP QA target = total current 3e5 A),
<|B.n|>/<B> on that surface (32x32 half period), length per hfp, max curvature, max MSC, coil-coil and coil-surface distance.

    python force_front_metrics.py <out.json> <name=coils.json> [<name=coils.json> ...]      (uwplasma-env, CPU)

Writes/updates out.json {name: {...}} (existing entries are kept unless recomputed).
"""
import json
import sys
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart, Current, RegularizedCoil, coils_via_symmetries, regularization_circ
from simsopt.geo import (CurveCurveDistance, CurveLength, CurveSurfaceDistance, CurveXYZFourier, LpCurveCurvature,
                         MeanSquaredCurvature, SurfaceRZFourier)

P = Path(__file__).resolve().parent
INPUT = P.parent / "vmex/examples/data/input.LandremanPaul2021_QA_lowres"
B_REF = 0.2271
out_path = Path(sys.argv[1])
res = json.load(open(out_path)) if out_path.exists() else {}
s = SurfaceRZFourier.from_vmec_input(str(INPUT), range="half period", nphi=32, ntheta=32)
s_full = SurfaceRZFourier.from_vmec_input(str(INPUT), range="full torus", nphi=128, ntheta=64)
pts = s.gamma().reshape(-1, 3)
for arg in sys.argv[2:]:
    name, fn = arg.split("=", 1)
    d = json.load(open(fn))
    dofs = np.asarray(d.get("dofs_curves_raw", d.get("dofs_curves")), float)
    cur = np.asarray(d.get("dofs_currents_raw", d.get("dofs_currents")), float)
    order, nfp, nb = int(d["order"]), int(d["nfp"]), len(dofs)
    curves = []
    for b in dofs:
        c = CurveXYZFourier(256, order); c.x = b.reshape(-1); curves.append(c)
    coils = coils_via_symmetries(curves, [Current(float(i)) for i in cur], nfp, bool(d["stellsym"]))
    bs = BiotSavart(coils); bs.set_points(pts)
    scale = B_REF / float(np.mean(bs.AbsB()))
    coils = coils_via_symmetries(curves, [Current(float(i) * scale) for i in cur], nfp, bool(d["stellsym"]))
    bs = BiotSavart(coils); bs.set_points(pts)
    B = bs.B().reshape(32, 32, 3)
    bn = float(np.mean(np.abs(np.sum(B * s.unitnormal(), axis=2)))) / float(np.mean(bs.AbsB()))
    reg = [RegularizedCoil(c.curve, c.current, regularization_circ(0.05)) for c in coils]
    forces = [float(np.max(np.linalg.norm(np.asarray(c.force(reg)), axis=1))) for c in reg[:nb]]
    allc = [c.curve for c in coils]
    res[name] = dict(coils=fn, ncoils=nb, order=order, current_scale=scale,
                     lengths=[float(CurveLength(c).J()) for c in curves], total_length=float(sum(CurveLength(c).J() for c in curves)),
                     max_kappa=float(max(np.max(c.kappa()) for c in curves)), max_msc=float(max(MeanSquaredCurvature(c).J() for c in curves)),
                     min_cc=float(CurveCurveDistance(allc, 100.0, num_basecurves=nb).shortest_distance()),
                     min_cs=float(CurveSurfaceDistance(allc, s_full, 100.0).shortest_distance()),
                     max_force=float(max(forces)), forces=forces, normalized_BdotN=bn)
    r = res[name]
    print(f"{name:22s} L {r['total_length']:6.2f} kappa {r['max_kappa']:5.2f} msc {r['max_msc']:5.2f} cc {r['min_cc']:.3f} cs {r['min_cs']:.3f} "
          f"Fmax {r['max_force']:8.1f} N/m  Bn {bn:.2e}", flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("wrote", out_path)
