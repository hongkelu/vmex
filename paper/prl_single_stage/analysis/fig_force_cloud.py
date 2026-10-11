"""Force-vs-field-error figure in the style of Gil et al. 2026 Fig. 1, under ONE constraint set (Gil's PRE thresholds as bounds:
kappa <= 5, MSC <= 5, cc >= 0.1 m, cs >= 0.15 m; 5 coils per half period, LP QA at R0 = 1 m, a = 0.05 m, <B> = 0.227 T):

    python analysis/fig_force_cloud.py analysis/fig_force_cloud.pdf [--box|--scan]

(a) every archive optimization inside the constraint set (grey cloud, their own metric <|B.n|>/<B>), its Pareto front (black),
    Gil's 25 m points (orange stars) and our strict-box runs (blue, B.n of the final coils on the LP QA target);
(b) the same front members and our runs re-solved with the three-term free-boundary evaluator at 12x12 / VC 64:
    QS error of the equilibrium the coils make.  --scan uses Gil's a-posteriori filter instead (kappa < 12, l < 6.5 m, ...),
    which is the set our 'S' family (scan constraints, start's own length) is compared with.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parents[1]
out = sys.argv[1]
mode = "scan" if "--scan" in sys.argv else "box"
cloud = json.load(open(P / "runs/hurwitz_cloud.json"))
M = json.load(open(P / "runs/force_front_metrics.json"))
EV = P / "runs/lpqa_vacuum_eval_12x12"
OURS, GIL = "C0", "C1"


def inside(r):
    if mode == "box":
        return r["kappa"] <= 5 and r["msc"] <= 5 and r["cc"] >= 0.1 and r["cs"] >= 0.15
    return r["kappa"] < 12 and r["maxl"] < 6.5 and r["cs"] > 0.166 and r["cc"] > 0.083 and r["bn"] < 4e-3


def front(pts, key_y="bn"):
    pts = sorted(pts, key=lambda p: (p[key_y], p["force"])); out, best = [], 1e9
    for p in pts:
        if p["force"] < best:
            out.append(p); best = p["force"]
    return sorted(out, key=lambda p: p["force"])


def qs_of(name):
    f = EV / name / "row.json"
    return json.load(open(f))["qa_total"] if f.exists() else None


def latest_run(prefix):
    c = sorted((p for p in (P / "runs/batch").glob(prefix + "*") if p.is_dir() and re.fullmatch(re.escape(prefix) + r"(-c\d+)?", p.name)),
               key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
    return c[-1] if c else None


sel = [r for r in cloud if inside(r) and 6000 <= r["force"] <= 18000 and r["bn"] < 1e-2]
fr = front(sel)
families = (("A", "-", "plasma pinned ($R_0$, A, $\\iota$)"), ("B", "--", "aspect ratio free")) if mode == "box" else \
           (("SA", "-", "plasma pinned ($R_0$, A, $\\iota$)"), ("SB", "--", "aspect ratio free"))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "lines.linewidth": 1.4, "axes.linewidth": 0.8, "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300, gridspec_kw=dict(wspace=0.3))
# (a) their metric
a.scatter([r["force"] / 1e3 for r in sel], [r["bn"] for r in sel], s=3, color="0.75", linewidths=0, zorder=1, rasterized=True)
a.plot([r["force"] / 1e3 for r in fr], [r["bn"] for r in fr], "-", color="0.2", lw=1.2, zorder=3)
a.plot([r["force"] / 1e3 for r in fr], [r["bn"] for r in fr], "o", color="0.2", ms=3, zorder=3)
for n, m in M.items():
    if n.startswith("gilpar_L25"):
        a.plot(m["max_force"] / 1e3, m["normalized_BdotN"], "*", color=GIL, ms=9, zorder=4)
notes = []
for tag, ls, lab in families:
    pts = []
    for thr in (7, 8, 9, 10, 11, 12, 13, 14, 16, 18):
        name = f"ff25{tag}-F{thr}"; m = M.get("ours_" + name)
        d = latest_run(name)
        if m is None or d is None:
            continue
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()] if (d / "metrics.jsonl").exists() else []
        q = qs_of("ours_" + name)
        if q is None and rows:
            q = rows[-1]["qa"]; notes.append(f"{name}: run value at step {rows[-1]['step']}")
        pts.append((m["max_force"] / 1e3, m["normalized_BdotN"], q, rows[-1]["step"] if rows else -1, rows[-1]["bn"] if rows else np.nan))
    pts.sort()
    if tag in ("A", "SA"):   # pinned plasma only: their metric (coil field on the LP QA target) and ours (three-term residual on own LCFS)
        a.plot([p[0] for p in pts], [p[1] for p in pts], "D", color=OURS, ms=5, ls=ls, mfc=OURS, mew=1.2, zorder=5)
        a.plot([p[0] for p in pts], [p[4] for p in pts], "D", color=OURS, ms=5, ls=":", mfc="white", mew=1.2, zorder=5)
    b.plot([p[0] for p in pts], [p[2] for p in pts], "D", color=OURS, ms=5, ls=ls, mfc=OURS if ls == "-" else "white", mew=1.2, zorder=5, label=f"this work, {lab}")
# (b) re-solved front members (and cloud samples) in actual QS
xs, ys, xc, yc = [], [], [], []
names = {("hurfront_" + r["uuid"][:8]): r for r in sel}
for n, m in M.items():
    if not n.startswith("hurfront_") or n not in names:
        continue
    q = qs_of(n)
    if q is None:
        continue
    (xs if names[n] in fr else xc).append(m["max_force"] / 1e3); (ys if names[n] in fr else yc).append(q)
b.plot(xc, yc, ".", color="0.75", ms=4, zorder=1)
b.plot(xs, ys, "o", color="0.2", ms=3.5, zorder=3)
for n, m in M.items():
    if n.startswith("gilpar_L25") and qs_of(n) is not None:
        b.plot(m["max_force"] / 1e3, qs_of(n), "*", color=GIL, ms=9, zorder=4)
for ax in (a, b):
    ax.set_yscale("log"); ax.set_xlim(6.5, 17.5); ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("max coil force (kN/m; $\\langle B\\rangle$ = 0.227 T, $R_0$ = 1 m)")
a.set_ylabel("$\\langle|B\\cdot n|\\rangle/\\langle B\\rangle$ on the target"); b.set_ylabel("QS error of the equilibrium the coils make")
title = ("Gil's thresholds as bounds: $\\kappa\\leq$5, MSC$\\leq$5, cc$\\geq$0.1 m, cs$\\geq$0.15 m" if mode == "box"
         else "Gil's filter: $\\kappa<$12, $\\ell<$6.5 m, cc$>$0.083 m, cs$>$0.166 m")
a.set_title(f"(a)  {len(sel)} archive sets, " + title, loc="left", fontsize=6.5)
b.set_title(f"(b)  re-solved: {len(xs)} front members, {len(xc)} others", loc="left", fontsize=6.5)
handles = [Line2D([], [], marker=".", color="0.75", ls="", ms=6, label="Hurwitz et al. 2024 scan (inside the constraint set)"),
           Line2D([], [], marker="o", color="0.2", ls="-", ms=3.5, label="its Pareto front"),
           Line2D([], [], marker="*", color=GIL, ls="", ms=9, label="Gil et al. 2026, 25 m"),
           Line2D([], [], marker="D", color=OURS, ls="-", ms=5, label=f"this work, {families[0][2]}: (a) coil field on the LP QA target, (b) QS"),
           Line2D([], [], marker="D", color=OURS, ls=":", ms=5, mfc="white", mew=1.2, label="this work: three-term B.n residual on the LCFS the coils make"),
           Line2D([], [], marker="D", color=OURS, ls="--", ms=5, mfc="white", mew=1.2, label=f"this work, {families[1][2]} (b only)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.26)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out, f"| cloud {len(sel)}, front {len(fr)}, re-solved front members {len(xs)}"); print("\n".join("  " + n for n in notes))
