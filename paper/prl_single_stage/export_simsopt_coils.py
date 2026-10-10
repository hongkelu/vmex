"""ESSOS coil file (vmex scale, R0 = 1 m) -> SIMSOPT BiotSavart JSON at reactor scale.

    python export_simsopt_coils.py <coils.json> <out.json> [--scale 10.1266] [--b-target 5.865]

Geometry is multiplied by --scale (LP QA: R0 10.1266 m, the Gil/Wechsung reactor
scale); currents are multiplied by --scale (keeps B) and then by --b-target
(the field at R0 = 1 m is 1 T in vmex runs, so the result has ~--b-target tesla,
the ARIES-CS <B> everybody scales to).  The symmetry copies are rebuilt with
simsopt's coils_via_symmetries, the convention of the published files.
Run with simsopt-env (MPICH_GPU_SUPPORT_ENABLED=0 on a login node).
"""
import argparse
import json

import numpy as np
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier

ap = argparse.ArgumentParser()
ap.add_argument("coils")
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=10.1266)
ap.add_argument("--b-target", type=float, default=5.865, help="provisional field scale; superseded by --b-surface")
ap.add_argument("--b-surface", type=float, default=5.39, help="rescale currents so the mean |B| on the reactor-scale LP QA target surface is this (Gil archive: 5.39 T); 0 disables")
ap.add_argument("--nquad", type=int, default=256)
args = ap.parse_args()

d = json.load(open(args.coils))
dofs = np.asarray(d.get("dofs_curves_raw", d.get("dofs_curves")), float)          # (nbase, 3, 2o+1)
currents = np.asarray(d.get("dofs_currents_raw", d.get("dofs_currents")), float)   # (nbase,)
order, nfp, stellsym = int(d["order"]), int(d["nfp"]), bool(d["stellsym"])
if d.get("scaling_factor", 0.0) != 0.0 or d.get("scale_fixed", 1.0) != 1.0:
    raise SystemExit("coil file uses ESSOS mode scaling; convert the raw dofs first")

base_curves, base_currents = [], []
for b, I in zip(dofs, currents):
    c = CurveXYZFourier(args.nquad, order)
    c.x = (b * args.scale).reshape(-1)
    base_curves.append(c)
    base_currents.append(Current(float(I) * args.scale * args.b_target))
coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
if args.b_surface > 0:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier
    _s = SurfaceRZFourier.from_vmec_input("/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data/input.LandremanPaul2021_QA_reactorScale_lowres",
                                          range="full torus", quadpoints_phi=np.linspace(0, 1, 128, endpoint=False), quadpoints_theta=np.linspace(0, 1, 64, endpoint=False))
    _bs = BiotSavart(coils); _bs.set_points(_s.gamma().reshape(-1, 3))
    _f = args.b_surface / float(np.mean(np.linalg.norm(_bs.B(), axis=1)))
    for c in base_currents:
        c.x = c.x * _f
    print(f"currents rescaled by {_f:.4f} so that mean |B| on the target surface = {args.b_surface} T")
bs = BiotSavart(coils)
bs.save(args.out)
print(f"wrote {args.out}: {len(coils)} coils, order {order}, base currents {np.round([c.get_value() for c in base_currents], 0)} A")
