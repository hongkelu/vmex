"""Gil-realized runs at converged resolution (GRc): start = Gil's AL coils, every limit = the value his point attains
when measured at 1024 points (runs/force_front_metrics_1024.json), so his own coils start exactly feasible in our
converged rows. Tie-or-beat on his own terms. -> jobs_GRc.txt

    python build_grc.py
"""
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
M = json.load(open(P / "runs/force_front_metrics_1024.json"))
EV = P / "runs/lpqa_vacuum_eval_12x12"
tpl = next(l for l in (P / "jobs_batch14.txt").read_text().splitlines() if l.startswith("ff25GA-F10|")).split("|")
base = json.loads(tpl[3])
for k in ("FORCE_LIMIT", "OBJECTIVE_SCALE"):
    base.pop(k, None)
extra = re.sub(r"--coils \S+", "", tpl[4]); extra = re.sub(r"--restart \S+", "", extra)
extra = re.sub(r"--design-step-scale \S+", "--design-step-scale 0.5", extra).strip()
out = []
for g in ("L25_F9", "L25_F10.5", "L25_F12", "L30_F10", "L30_F11", "L30_F12", "L30_F14"):
    m = M["gilpar_" + g]
    qa = json.load(open(EV / f"gilpar_{g}" / "row.json"))["qa_total"]
    o = dict(base, CURVATURE_LIMIT=round(max(12.0, m["max_kappa"] * 1.005), 3), MSC_LIMIT=round(max(6.0, m["max_msc"] * 1.005), 3),
             COIL_DISTANCE_LIMIT=round(m["min_cc"] * 0.995, 5), COIL_SURFACE_DISTANCE_LIMIT=0.166, CONDUCTOR_RADIUS=0.05,
             TOTAL_LENGTH_LIMIT=round(m["total_length"] * 1.0005, 4), LENGTH_LIMIT=round(max(6.5, max(m["lengths"]) * 1.0005), 4),
             FORCE_LIMIT=round(m["max_force"] / m["current_scale"] ** 2 * 1.003, 1), OBJECTIVE_SCALE=float(f"{1 / qa:.3g}"),
             OPTIMIZER_FTOL=1e-12, CURVATURE_MARGIN=0.0, MSC_MARGIN=0.0, DISTANCE_MARGIN=0.0, LENGTH_MARGIN=0.0,
             ASPECT_RANGE=[5.99, 6.01])
    name = f"GRc{g[1:3]}-F{g.split('_F')[1]}"
    out.append("|".join([name, tpl[1], tpl[2], json.dumps(o), f"{extra} --coils coils/gilpar_{g}.json", "-"]))
    print(f"{name:12s} QS0 {qa:.2e}  cc>={o['COIL_DISTANCE_LIMIT']:.4f} kappa<={o['CURVATURE_LIMIT']} msc<={o['MSC_LIMIT']} "
          f"L<={o['TOTAL_LENGTH_LIMIT']} coil<={o['LENGTH_LIMIT']} F<={m['max_force'] * 1.003 / 1e3:.3f} kN/m")
(P / "jobs_GRc.txt").write_text("# Gil-realized constraints at converged resolution (build_grc.py)\n" + "\n".join(out) + "\n")
