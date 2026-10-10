"""Gil et al.'s engineering metrics (their SIMSOPT objective classes) on a SIMSOPT BiotSavart JSON.

    python gil_coil_metrics.py <biot_savart.json> <out.json> [--surface-wout wout.nc --surface-scale 10.1266]
                                                           [--a 0.15] [--turns 200]

Reports, per their Table 2 conventions at reactor scale: total coil length per half period (sum of the
base coils), max curvature, max mean-squared curvature, min coil-coil distance (all coils), min
coil-surface distance, and the max force per unit length (coil_force with regularization_circ(a),
a = 0.15 m as in auglag_qa_reactorscale.py), raw and divided by the turn count.  The surface is the LP QA
reactor-scale target unless --surface-wout gives the equilibrium the coils actually made (its boundary,
scaled to reactor size).  Pure CPU; run with uwplasma-env, JAX_PLATFORMS=cpu.
"""
import argparse
import json

import numpy as np
from simsopt import load
from simsopt.field import RegularizedCoil, regularization_circ
from simsopt.geo import (CurveCurveDistance, CurveLength, CurveSurfaceDistance, LpCurveCurvature, MeanSquaredCurvature,
                         SurfaceRZFourier)

INPUT = "/pscratch/sd/h/hongkelu/finitebeta-single-stage/simsopt-editable/tests/test_files/input.LandremanPaul2021_QA_reactorScale_lowres"

ap = argparse.ArgumentParser()
ap.add_argument("coils")
ap.add_argument("out")
ap.add_argument("--surface-wout", default=None)
ap.add_argument("--surface-scale", type=float, default=10.1266)
ap.add_argument("--a", type=float, default=0.15)
ap.add_argument("--turns", type=float, default=200.0)
args = ap.parse_args()

bs = load(args.coils)
coils_in = bs.coils
# rebuild with the conductor regularization their force metric uses
coils = [RegularizedCoil(c.curve, c.current, regularization_circ(args.a)) for c in coils_in]
nbase = None
for i, c in enumerate(coils):
    if type(c.curve).__name__ != "CurveXYZFourier":
        nbase = i
        break
nbase = nbase or len(coils)
base = coils[:nbase]
curves = [c.curve for c in coils]

if args.surface_wout:
    s = SurfaceRZFourier.from_wout(args.surface_wout, range="full torus", nphi=128, ntheta=64)
    s.x = s.x * args.surface_scale
else:
    s = SurfaceRZFourier.from_vmec_input(INPUT, range="full torus", nphi=128, ntheta=64)

lengths = [CurveLength(c.curve).J() for c in base]
kappa = [float(np.max(np.abs(np.asarray(c.curve.kappa())))) for c in base]
msc = [MeanSquaredCurvature(c.curve).J() for c in base]
cc = CurveCurveDistance(curves, 10.0).shortest_distance()
cs = CurveSurfaceDistance([c.curve for c in base], s, 10.0).shortest_distance()
forces = [float(np.max(np.linalg.norm(np.asarray(c.force(coils)), axis=1))) for c in base]
rec = dict(coils=args.coils, surface=args.surface_wout or INPUT, nbase=nbase, ncoils=len(coils),
           length_per_hfp=float(sum(lengths)), lengths=[float(x) for x in lengths],
           max_kappa=float(max(kappa)), max_msc=float(max(msc)), min_cc=float(cc), min_cs=float(cs),
           max_force_N_per_m=float(max(forces)), max_force_MN_per_m_per_turn=float(max(forces)) / args.turns / 1e6,
           forces_N_per_m=forces, a_regularization=args.a, turns=args.turns)
json.dump(rec, open(args.out, "w"), indent=1)
print(f"L/hfp {rec['length_per_hfp']:.1f} m  kmax {rec['max_kappa']:.3f}  MSC {rec['max_msc']:.4f}  cc {cc:.3f} m  cs {cs:.3f} m  "
      f"Fmax {rec['max_force_N_per_m']:.3e} N/m = {rec['max_force_MN_per_m_per_turn']:.3f} MN/m per turn")
