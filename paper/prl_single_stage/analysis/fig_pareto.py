"""Pareto figure: QS error of the equilibrium the coils make vs the coil-length budget, everything else fixed.

    python analysis/fig_pareto.py analysis/fig_pareto.pdf [--eval 12x12]

Fixed for every point: LP QA target (nfp 2, A 6, vacuum) with the plasma pinned (R0, A, min iota), 4 coils per half
period of Fourier order 16 (3-coil sets drawn as triangles), curvature <= 0.5/m, MSC <= 0.05/m^2, coil-coil >= 1.0 m,
coil-surface >= 1.5 m, max force <= 0.90 MN/m per turn, same free-boundary evaluator at one resolution.
Ours: runs/batch/pareto-L<L>* (latest continuation; the 12x12 evaluator row ours_pareto-L<L> when present, otherwise the
run's own value, listed on stdout).  Before the scan exists, the 18/20/24 m fair twins stand in (hollow markers).
Published: Gil 2026 (circles) and Wechsung 2022 (squares); a marker is hollow when that set violates one of the fixed
thresholds by more than 2% (listed on stdout).  Lengths are quoted at reactor scale (R0 = 10.1266 m) as in Gil's Table 2.
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
tag = sys.argv[sys.argv.index("--eval") + 1] if "--eval" in sys.argv else "12x12"
EV, EV8 = P / f"runs/lpqa_vacuum_eval_{tag}", P / "runs/lpqa_vacuum_eval"
SCALE = 10.1266
OURS, GIL, WECH, TGT = "C0", "C1", "C2", "0.45"
# fixed thresholds at reactor scale
THR = dict(max_kappa=0.5, max_msc=0.05, min_cc=1.0, min_cs=1.5, max_force_MN_per_m_per_turn=0.90)
notes = []


def published_qs(name):
    for d in (EV / name, EV8 / name):
        if (d / "row.json").exists():
            if d.parent == EV8:
                notes.append(f"{name}: 8x8 evaluator")
            return json.load(open(d / "row.json"))["qa_total"]


def published_feasible(name):
    f = P / "runs/published" / name / "gil_score_step0" / "gil_coil_metrics.json"
    if not f.exists():
        return True
    m = json.load(open(f))
    bad = [k for k, v in THR.items() if (m[k] > v * 1.02 if k.startswith("max") else m[k] < v * 0.98)]
    if bad:
        notes.append(f"{name} violates {bad}")
    return not bad


def latest_run(prefix):
    cands = sorted((p for p in (P / "runs/batch").glob(prefix + "*") if p.is_dir() and re.fullmatch(re.escape(prefix) + r"(-c\d+)?", p.name)),
                   key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
    return cands[-1] if cands else None


def ours(prefix):
    d = latest_run(prefix)
    if d is None:
        return None, None
    ev = EV / f"ours_{prefix}" / "row.json"
    if ev.exists():
        return json.load(open(ev))["qa_total"], "eval"
    m = d / "metrics.jsonl"
    if m.exists() and m.stat().st_size:
        rows = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
        notes.append(f"{d.name}: run value at step {rows[-1]['step']} (not yet rescored)")
        return rows[-1]["qa"], "run"
    return None, None


plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "lines.linewidth": 1.5, "axes.linewidth": 0.8, "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=300)
# published points
for name, L, color, marker in (("gil_L18", 182.2, GIL, "o"), ("gil_L20", 203.4, GIL, "o"), ("gil_L24", 239.6, GIL, "o"),
                               ("wechsung_L18", 182.4, WECH, "s"), ("wechsung_L20", 202.7, WECH, "s"), ("wechsung_L24", 243.1, WECH, "s"),
                               ("gil_3coil_L18", 182.2, GIL, "^")):
    q = published_qs(name)
    if q is None:
        continue
    ok = published_feasible(name)
    ax.plot(L, q, marker, color=color, ms=7, mfc=color if ok else "white", mew=1.4, zorder=3)
# our front
xs, ys, kinds = [], [], []
for L in (16, 17, 18, 19, 20, 21, 22, 24):
    q, kind = ours(f"pareto-L{L}")
    if q is None and L in (18, 20, 24):  # stand-ins until the scan exists
        q, kind = ours({18: "vac-L18-fair", 20: "vac-L20-fair", 24: "vac-L24-wech-fair"}[L])
        kind = "standin" if q is not None else None
    if q is not None:
        xs.append(L * SCALE); ys.append(q); kinds.append(kind)
if xs:
    ax.plot(xs, ys, "-", color=OURS, lw=1.6, zorder=2)
    for x, y, k in zip(xs, ys, kinds):
        ax.plot(x, y, "D", color=OURS, ms=6, mfc=OURS if k == "eval" else "white", mew=1.4, zorder=4)
q3, k3 = ours("vac-3coil-fair")
if q3 is not None:
    ax.plot(182.2, q3, "^", color=OURS, ms=7, mfc=OURS if k3 == "eval" else "white", mew=1.4, zorder=4)
t = json.load(open(EV / "target.json" if (EV / "target.json").exists() else EV8 / "target.json"))["qa_total"]
ax.axhline(t, color=TGT, ls="--", lw=1.0)
ax.text(246, t * 1.3, "fixed-boundary target", color=TGT, fontsize=7, ha="right")
ax.set_yscale("log"); ax.set_xlim(158, 248); ax.set_ylim(4e-7, 4e-3)
ax.set_xticks([162, 172, 182, 192, 203, 213, 223, 243])
ax.set_xticklabels(["16", "17", "18", "19", "20", "21", "22", "24"])
ax.set_xlabel("coil length per half period (m at $R_0$ = 1 m; ×10.13 at reactor scale)")
ax.set_ylabel("QS error of the equilibrium the coils make")
ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
handles = [Line2D([], [], marker="D", color=OURS, ls="-", ms=6, label="this work, free-boundary single stage"),
           Line2D([], [], marker="o", color=GIL, ls="", ms=7, label="Gil et al. 2026, stage two"),
           Line2D([], [], marker="s", color=WECH, ls="", ms=7, label="Wechsung et al. 2022, stage two"),
           Line2D([], [], marker="^", color="0.3", ls="", ms=7, label="3 coils per half period"),
           Line2D([], [], marker="o", color="0.3", ls="", ms=7, mfc="white", mew=1.4, label="hollow: outside the fixed thresholds")]
ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95, edgecolor="0.85", fontsize=6.5)
ax.set_title("LP QA, 4 coils/hfp, $\\kappa\\leq$0.5, MSC$\\leq$0.05, cc$\\geq$1.0 m, cs$\\geq$1.5 m, F$\\leq$0.9 MN/m", fontsize=6.5, loc="left")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out); print("\n".join("  " + n for n in notes))
