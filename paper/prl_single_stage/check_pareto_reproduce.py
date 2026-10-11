"""Reproduce the published Pareto-front coordinates (Hurwitz et al. 2024 / Gil et al. 2026 Fig. 1) from our converted coil
files: max coil force (SIMSOPT coil_force, regularization_circ(0.05 m), archive currents) and <|B.n|>/<B> on the
R0 = 1 m LP QA surface, compared with the archive's results.json.

    python check_pareto_reproduce.py [--n 8]            (SIMSOPT 1.11 in uwplasma-env, CPU)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart, Current, RegularizedCoil, coils_via_symmetries, regularization_circ
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier

P = Path(__file__).resolve().parent
INPUT = P.parent / "vmex/examples/data/input.LandremanPaul2021_QA_lowres"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--gil", action="store_true", help="Gil's 25/30 m Pareto sets instead (no archive values: force vs his threshold)")
args = ap.parse_args()

s = SurfaceRZFourier.from_vmec_input(str(INPUT), range="half period", nphi=32, ntheta=32)


def build(cfile, currents, nquad=256):
    d = json.load(open(cfile))
    dofs = np.asarray(d["dofs_curves_raw"], float)
    order = int(d["order"])
    curves, cur = [], []
    for b, I in zip(dofs, currents):
        c = CurveXYZFourier(nquad, order); c.x = b.reshape(-1); curves.append(c); cur.append(Current(float(I)))
    return coils_via_symmetries(curves, cur, d["nfp"], d["stellsym"])


if args.gil:
    print(f"{'set':18s} {'L/hfp':>6s} {'thr kN/m':>9s} {'Fmax ours':>11s} {'kappa':>7s} {'Bn ours':>9s}")
    for r in json.load(open(P / "coils/gil_pareto_index.json")):
        coils = build(P / "coils" / f"{r['name']}.json", r["archive_currents"])
        reg = [RegularizedCoil(c.curve, c.current, regularization_circ(0.05)) for c in coils]
        fmax = max(np.max(np.linalg.norm(np.asarray(c.force(reg)), axis=1)) for c in reg[:r["ncoils"]])
        bs = BiotSavart(coils); bs.set_points(s.gamma().reshape(-1, 3)); B = bs.B().reshape(32, 32, 3)
        bn = float(np.mean(np.abs(np.sum(B * s.unitnormal(), axis=2)))) / float(np.mean(bs.AbsB()))
        print(f"{r['name']:18s} {r['total_length']:6.2f} {r['force_threshold_kNm']:9.1f} {fmax:11.1f} {r['max_kappa']:7.1f} {bn:9.3e}")
    raise SystemExit

idx = sorted(json.load(open(P / "coils/hurwitz_pareto_index.json")), key=lambda r: r["max_force_archive"])
pick = [idx[int(round(i))] for i in np.linspace(0, len(idx) - 1, args.n)]
print(f"{'set':18s} {'L/hfp':>6s} {'Fmax archive':>13s} {'Fmax ours':>11s} {'ratio':>6s} {'Bn archive':>11s} {'Bn ours':>9s} {'ratio':>6s}")
for r in pick:
    res = json.load(open(Path(r["path"]) / "results.json"))
    coils = build(P / "coils" / f"{r['name']}.json", res["coil_currents"])
    nb = len(res["coil_currents"])
    reg = [RegularizedCoil(c.curve, c.current, regularization_circ(0.05)) for c in coils]   # SIMSOPT 1.11 API of coil_force
    fmax = max(np.max(np.linalg.norm(np.asarray(c.force(reg)), axis=1)) for c in reg[:nb])
    bs = BiotSavart(coils); bs.set_points(s.gamma().reshape(-1, 3))
    B = bs.B().reshape(32, 32, 3)
    bn = float(np.mean(np.abs(np.sum(B * s.unitnormal(), axis=2)))); modb = float(np.mean(bs.AbsB()))
    print(f"{r['name']:18s} {r['total_length']:6.2f} {res['max_max_force']:13.1f} {fmax:11.1f} {fmax / res['max_max_force']:6.3f} "
          f"{res['normalized_BdotN']:11.3e} {bn / modb:9.3e} {(bn / modb) / res['normalized_BdotN']:6.3f}")
