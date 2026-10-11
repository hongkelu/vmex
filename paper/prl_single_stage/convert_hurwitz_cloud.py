"""Hurwitz, Landreman & Kaptanoglu 2024 scan (Zenodo 13913510, LP QA at R0 = 1 m, 5 coils per half period) ->
ESSOS coil files for the sets that satisfy our fixed Pareto thresholds.

    python convert_hurwitz_cloud.py <output_root> [--all] [--limit N]      (pure numpy; login-node safe)

Scans <output_root>/**/results.json (one per optimization), keeps the sets whose max curvature, max MSC, coil-coil and
coil-surface distances satisfy the fixed set at R0 = 1 m (kappa <= 5.063, MSC <= 5.127, cc >= 0.0988, cs >= 0.148;
--all keeps everything), parses the matching biot_savart.json (SIMSOPT graph, no simsopt needed), and writes
coils/hurwitz_<uuid8>.json with the currents scaled to the LP QA deck's RBTOR (as convert_coilsets.py) plus
coils/hurwitz_cloud_index.json (name, UUID, total length per half period, the archive's own metrics).  The force in the
archive uses a = 0.05 m regularization at R0 = 1 m and is NOT comparable to Gil's convention: forces are recomputed by
gil_coil_metrics.py after the 12x12 evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "references/gil2026"))
from dump_coils import Graph  # noqa: E402  (pure numpy SIMSOPT-JSON reader)
from convert_coilsets import fourier_gamma, symmetric_copies, linked_rbphi, RBTOR, NSEG  # noqa: E402

THR = dict(max_kappa=5.063, max_msc=5.127, min_cc=0.0988, min_cs=0.148)   # R0 = 1 m
OUT = HERE / "coils"

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--all", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tol", type=float, default=1.0, help="threshold tolerance factor (1.02 = 2 %% slack)")
args = ap.parse_args()

results = sorted(Path(args.root).rglob("results.json"))
print(f"{len(results)} optimizations under {args.root}")
rows, kept, bad_parse = [], 0, 0
for rf in results:
    r = json.load(open(rf))
    m = dict(max_kappa=float(r["max_max_κ"]), max_msc=float(r["max_MSC"]), min_cc=float(r["coil_coil_distance"]),
             min_cs=float(r["coil_surface_distance"]), total_length=float(sum(r["lengths"])), ncoils=int(r["ncoils"]),
             order=int(r["order"]), nfp=int(r["nfp"]), normalized_BdotN=float(r["normalized_BdotN"]),
             max_force_archive=float(r["max_max_force"]), force_threshold=float(r["force_threshold"]),
             success=bool(r.get("success", True)), UUID=r["UUID"], path=str(rf.parent))
    ok = (m["max_kappa"] <= THR["max_kappa"] * args.tol and m["max_msc"] <= THR["max_msc"] * args.tol
          and m["min_cc"] >= THR["min_cc"] / args.tol and m["min_cs"] >= THR["min_cs"] / args.tol)
    m["feasible"] = bool(ok)
    if not (ok or args.all):
        continue
    bs = rf.parent / "biot_savart.json"
    if not bs.exists():
        continue
    try:
        G = Graph(bs)
        base_dofs, currents = [], []
        for cref in G.root["coils"]:
            coil = G.deref(cref)
            _, dofs, order = G.curve(coil["curve"])
            if dofs is not None:
                base_dofs.append(dofs.reshape(3, 2 * order + 1)); currents.append(G.current(coil["current"]))
    except Exception as e:  # noqa: BLE001
        bad_parse += 1; print(f"  parse failed {rf.parent.name}: {e}"); continue
    base = np.array(base_dofs); rel = np.array(currents)
    coils = []
    for b, I in zip(base, rel):
        g, dg = fourier_gamma(b, order, NSEG)
        coils += symmetric_copies(g, dg, I, m["nfp"], True)
    factor = RBTOR / linked_rbphi(coils)
    name = f"hurwitz_{m['UUID'][:8]}"
    json.dump(dict(nfp=m["nfp"], stellsym=True, order=order, n_segments=NSEG, dofs_curves_raw=base.tolist(), scaling_type=2,
                   scaling_factor=0.0, scale_fixed=1.0, dofs_currents_raw=(rel * factor).tolist(), currents_scale=None),
              open(OUT / f"{name}.json", "w"), indent=1)
    m["name"] = name; m["currents"] = (rel * factor).tolist()
    rows.append(m); kept += 1
    print(f"{name}  L/hfp {m['total_length']:.3f}  kappa {m['max_kappa']:.2f}  msc {m['max_msc']:.2f}  cc {m['min_cc']:.4f}  cs {m['min_cs']:.3f}  "
          f"Bn {m['normalized_BdotN']:.1e}  {'feasible' if ok else 'outside'}")
    if args.limit and kept >= args.limit:
        break
json.dump(rows, open(OUT / "hurwitz_cloud_index.json", "w"), indent=1)
print(f"kept {kept} sets ({sum(r['feasible'] for r in rows)} inside the fixed thresholds), {bad_parse} parse failures; index -> coils/hurwitz_cloud_index.json")
