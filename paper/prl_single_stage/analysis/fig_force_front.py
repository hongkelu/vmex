"""Force-vs-QS Pareto fronts at 25 m per half period (Gil et al. 2026 Fig. 1 setting, y-axis = QS error of the equilibrium
the coils actually make, three-term free-boundary solve at 12x12 / VC 64).

    python analysis/fig_force_front.py analysis/fig_force_front.pdf

Published: Hurwitz/Kaptanoglu scan sets (front under Gil's filter: grey; cloud sample: light grey) and Gil's augmented-
Lagrangian points (C1), all re-solved by us; x = max coil force in their convention (a = 0.05 m, <B> = 0.2271 T on the
LP QA target, kN/m) from runs/force_front_metrics.json.  Ours: ff25A-* (plasma pinned, C0 filled) and ff25B-* (aspect
ratio free, C0 hollow), latest saved step, re-solved row when present else the run's own value (hollow + listed).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parents[1]
out = sys.argv[1]
EV = P / "runs/lpqa_vacuum_eval_12x12"
M = json.load(open(P / "runs/force_front_metrics.json")) if (P / "runs/force_front_metrics.json").exists() else {}
OURS, GIL, HUR = "C0", "C1", "0.45"
notes = []


def qs_of(name):
    f = EV / name / "row.json"
    return json.load(open(f))["qa_total"] if f.exists() else None


def latest_run(prefix):
    cands = sorted((p for p in (P / "runs/batch").glob(prefix + "*") if p.is_dir() and re.fullmatch(re.escape(prefix) + r"(-c\d+)?", p.name)),
                   key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
    return cands[-1] if cands else None


plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "lines.linewidth": 1.5, "axes.linewidth": 0.8, "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=300)
front = {("hurfront_" + p["uuid"][:8]) for p in json.load(open(P.parent / "references/hurwitz2024/gil_filter_front.json"))}
for name, m in M.items():
    q = qs_of(name)
    if q is None:
        continue
    x = m["max_force"] / 1e3
    if name.startswith("gilpar_L25"):
        ax.plot(x, q, "*", color=GIL, ms=9, zorder=4)
    elif name.startswith("gilpar"):
        continue
    elif name in front:
        ax.plot(x, q, "o", color=HUR, ms=4, zorder=3)
    else:
        ax.plot(x, q, ".", color="0.7", ms=4, zorder=2)
for tag, mfc, lab in (("A", OURS, "pinned"), ("B", "white", "aspect free")):
    xs, ys, hollow = [], [], []
    for thr in (7, 8, 9, 10, 11, 12, 13, 14, 16, 18):
        name = f"ff25{tag}-F{thr}"
        d = latest_run(name)
        if d is None:
            continue
        q = qs_of("ours_" + name)
        kind = "eval"
        if q is None:
            mm = d / "metrics.jsonl"
            if mm.exists() and mm.stat().st_size:
                rows = [json.loads(l) for l in mm.read_text().splitlines() if l.strip()]
                q, kind = rows[-1]["qa"], "run"; notes.append(f"{d.name}: run value at step {rows[-1]['step']} (not re-solved)")
        x = M.get("ours_" + name, {}).get("max_force")
        if q is None or x is None:
            notes.append(f"{name}: missing {'QS' if q is None else 'force metric'}"); continue
        xs.append(x / 1e3); ys.append(q); hollow.append(kind != "eval")
    if xs:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order], "-", color=OURS, lw=1.4, ls="-" if tag == "A" else "--", zorder=4)
        for x, y, h in zip(xs, ys, hollow):
            ax.plot(x, y, "D", color=OURS, ms=6, mfc="white" if (h or tag == "B") else OURS, mew=1.4, zorder=5)
ax.set_yscale("log"); ax.set_xlabel("max coil force (kN/m, $\\langle B\\rangle$ = 0.227 T, $R_0$ = 1 m)")
ax.set_ylabel("QS error of the equilibrium the coils make")
ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
handles = [Line2D([], [], marker="D", color=OURS, ls="-", ms=6, label="this work, plasma pinned ($R_0$, A, $\\iota$)"),
           Line2D([], [], marker="D", color=OURS, ls="--", ms=6, mfc="white", mew=1.4, label="this work, aspect ratio free"),
           Line2D([], [], marker="*", color=GIL, ls="", ms=9, label="Gil et al. 2026, augmented Lagrangian"),
           Line2D([], [], marker="o", color=HUR, ls="", ms=4, label="Hurwitz et al. 2024 scan, Pareto front"),
           Line2D([], [], marker=".", color="0.7", ls="", ms=4, label="Hurwitz et al. 2024 scan, other sets")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.5, columnspacing=1.2)
ax.set_title("LP QA, 25 m per half period, 5 coils, $\\kappa\\leq$5, MSC$\\leq$5, cc$\\geq$0.1 m, cs$\\geq$0.15 m", fontsize=6.5, loc="left")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out); print("\n".join("  " + n for n in notes))
