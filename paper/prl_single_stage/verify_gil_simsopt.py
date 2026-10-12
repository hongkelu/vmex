"""Verify Gil et al.'s published Pareto points with SIMSOPT on HIS OWN objects (no conversion).

    python verify_gil_simsopt.py            (uwplasma-env, CPU, login node)

Loads every qa_pareto_{25,30}_comparison/*/biot_savart_same_setup.json natively (the files were written by a SIMSOPT
development version whose Coil carries `regularization`; that key is removed before simsopt.load and re-attached
through RegularizedCoil). For each run: per-coil length, max curvature, max MSC on his quadrature points, min
coil-coil and coil-surface distance (LP QA surface at R0 = 1 m, full torus), max self-consistent Lorentz force with his
regularization, at his stored currents AND rescaled to <|B|> = 0.2271 T on the target (archive convention), and
<|B.n|>/<B>. Also compares with our converted coils/gilpar_*.json metrics. Writes runs/gil_simsopt_verification.json.
"""
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from simsopt._core import load
from simsopt.field import BiotSavart, RegularizedCoil
from simsopt.geo import CurveCurveDistance, CurveLength, CurveSurfaceDistance, MeanSquaredCurvature, SurfaceRZFourier

P = Path(__file__).resolve().parent
SRC = P.parent / "references/gil2026/auglag_extract/zenodo_repository"
INPUT = P.parent / "vmex/examples/data/input.LandremanPaul2021_QA_lowres"
B_REF = 0.2271
s_half = SurfaceRZFourier.from_vmec_input(str(INPUT), range="half period", nphi=32, ntheta=32)
s_full = SurfaceRZFourier.from_vmec_input(str(INPUT), range="full torus", nphi=128, ntheta=64)
pts = s_half.gamma().reshape(-1, 3)
ours = json.load(open(P / "runs/force_front_metrics.json"))


def load_native(fn):
    d = json.load(open(fn))
    regs = {}
    for k, v in d["simsopt_objs"].items():
        if isinstance(v, dict) and v.get("@class") == "Coil" and "regularization" in v:
            regs[k] = float(v.pop("regularization")["data"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as t:
        json.dump(d, t)
    bs = load(t.name)
    return bs, sorted(set(regs.values()))


def forces(coils, reg):
    rc = [RegularizedCoil(c.curve, c.current, reg) for c in coils]
    return [float(np.max(np.linalg.norm(np.asarray(c.force(rc)), axis=1))) for c in rc]


out = {}
for d in sorted(SRC.glob("qa_pareto_*_comparison/*/")):
    fn = d / "biot_savart_same_setup.json"
    if not fn.exists():
        continue
    L = 30 if "_30_" in d.name else 25
    thr = re.search(r"threshold([0-9p.]+)kNm", d.name).group(1).replace("p", ".")
    name = f"gilpar_L{L}_F{thr}"
    bs, regs = load_native(fn)
    coils = bs.coils
    base = [c.curve for c in coils[:5]]
    allc = [c.curve for c in coils]
    bs.set_points(pts)
    absb = float(np.mean(bs.AbsB()))
    B = bs.B().reshape(32, 32, 3)
    bn = float(np.mean(np.abs(np.sum(B * s_half.unitnormal(), axis=2)))) / absb
    reg = regs[0]
    f_raw = forces(coils, reg)
    scale2 = (B_REF / absb) ** 2
    r = dict(regularization=regs, a_equiv=float(np.sqrt(reg * np.sqrt(np.e))), quadpoints=len(base[0].quadpoints),
             order=int(base[0].order), lengths=[float(CurveLength(c).J()) for c in base],
             max_kappa=float(max(np.max(c.kappa()) for c in base)), max_msc=float(max(MeanSquaredCurvature(c).J() for c in base)),
             min_cc=float(CurveCurveDistance(allc, 100.0, num_basecurves=5).shortest_distance()),
             min_cs=float(CurveSurfaceDistance(allc, s_full, 100.0).shortest_distance()),
             mean_absB=absb, Bn=bn, max_force_raw=max(f_raw[:5]), max_force_0p2271=max(f_raw[:5]) * scale2,
             currents=[float(c.current.get_value()) for c in coils[:5]])
    r["total_length"] = sum(r["lengths"])
    out[name] = r
    o = ours.get(name, {})
    print(f"{name:17s} reg={reg:.4e} (a={r['a_equiv']:.3f}) nq={r['quadpoints']} L={r['total_length']:.2f} max_l={max(r['lengths']):.2f} "
          f"kappa={r['max_kappa']:.2f} msc={r['max_msc']:.2f} cc={r['min_cc']:.4f} cs={r['min_cs']:.4f} <|B|>={absb:.4f} "
          f"F_raw={r['max_force_raw']/1e3:.2f} F@0.2271={r['max_force_0p2271']/1e3:.2f} kN/m Bn={bn:.2e}"
          + (f"  | ours: kappa={o['max_kappa']:.2f} cc={o['min_cc']:.4f} F={o['max_force']/1e3:.2f}" if o else ""), flush=True)
json.dump(out, open(P / "runs/gil_simsopt_verification.json", "w"), indent=1)
print("wrote runs/gil_simsopt_verification.json")
