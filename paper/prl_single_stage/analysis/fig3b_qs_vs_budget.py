"""Fig. 3(b): actual QS error vs coil-length budget on the LP QA benchmark (vacuum).

    python fig3b_qs_vs_budget.py analysis/fig3b_qs_vs_budget.pdf [--eval 12x12]

Every point is the two-term QS error of the three-term free-boundary equilibrium the coils make, scored by ONE
evaluator (eval_coilsets_vacuum.py) at one resolution: runs/lpqa_vacuum_eval_<tag>/ (published sets and ours_<run>).
Falls back to runs/lpqa_vacuum_eval (8x8) for published sets and to the runs' own metrics.jsonl for ours when the
common-resolution row is missing (marked with a dot in the legend note).  Orange = Gil 2026, aqua = Wechsung 2022,
blue = ours: filled = margins (stricter than theirs), open = exactly their thresholds.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e5e4e0"
P = Path(__file__).resolve().parents[1]
out = sys.argv[1]
tag = sys.argv[sys.argv.index("--eval") + 1] if "--eval" in sys.argv else "12x12"
EV, EV8 = P / f"runs/lpqa_vacuum_eval_{tag}", P / "runs/lpqa_vacuum_eval"

# budget -> (published sets, ours under margins, ours under exactly their thresholds); run names = latest continuation
ROWS = {18: dict(gil="gil_L18", wech="wechsung_L18", margin="vac-L18-c2", exact="vac-L18-exact"),
        20: dict(gil="gil_L20", wech="wechsung_L20", margin="vac-L20-c3", exact="vac-L20-exact"),
        24: dict(gil="gil_L24", wech="wechsung_L24", margin="vac-L24-wech-c1", exact="vac-L24-wech-feas2")}
THREE = dict(gil="gil_3coil_L18", margin="vac-3coil-c2", exact="vac-3coil-exact")
fallback = []


def published(name):
    for d in (EV / name, EV8 / name):
        if (d / "row.json").exists():
            if d.parent == EV8 and EV.exists():
                fallback.append(name)
            return json.load(open(d / "row.json"))["qa_total"]
    return None


def ours(run):
    d = EV / f"ours_{run}"
    if (d / "row.json").exists():
        return json.load(open(d / "row.json"))["qa_total"]
    m = P / "runs/batch" / run / "metrics.jsonl"
    if m.exists():
        fallback.append(run)
        return json.loads(m.read_text().splitlines()[-1])["qa"]
    return None


plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.linewidth": 0.6, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
ax.set_yscale("log"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
first = True
for x, r in ROWS.items():
    lab = (lambda s: s if first else None)
    ax.plot(x, published(r["gil"]), "o", color=ORANGE, ms=6, label=lab("Gil et al. 2026, stage two"))
    ax.plot(x, published(r["wech"]), "s", color=AQUA, ms=6, label=lab("Wechsung et al. 2022, stage two"))
    if (v := ours(r["margin"])) is not None:
        ax.plot(x, v, "D", color=BLUE, ms=5, label=lab("ours, constraints stricter than theirs"))
    if (v := ours(r["exact"])) is not None:
        ax.plot(x, v, "D", color=BLUE, ms=7, mfc="white", mew=1.4, label=lab("ours, constraints exactly theirs"))
    first = False
ax.plot(18, published(THREE["gil"]), "^", color=ORANGE, ms=7)
if (v := ours(THREE["margin"])) is not None:
    ax.plot(18, v, "^", color=BLUE, ms=6)
if (v := ours(THREE["exact"])) is not None:
    ax.plot(18, v, "^", color=BLUE, ms=8, mfc="white", mew=1.4)
ax.annotate("triangles: 3 coils per half period", xy=(18, published(THREE["gil"])), xytext=(8, 2), textcoords="offset points", fontsize=6, color=MUTED)
tgt = EV / "target.json" if (EV / "target.json").exists() else EV8 / "target.json"
if tgt.exists():
    t = json.load(open(tgt))["qa_total"]
    ax.axhline(t, color=MUTED, ls="--", lw=1)
    ax.annotate("fixed-boundary target", xy=(24, t), xytext=(0, 3), textcoords="offset points", fontsize=6, color=MUTED, ha="right")
ax.set_xticks([18, 20, 24]); ax.set_xlim(17, 25)
ax.set_xlabel("coil length budget per half period (m, $R_0$ = 1 m)")
ax.set_ylabel("QS error of the equilibrium\nthe coils actually make")
ax.set_title(f"LP QA benchmark (nfp 2, A 6, vacuum), evaluated at {tag}", fontsize=7, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=6)
fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.9)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
print("wrote", out, "fallbacks (not at common resolution):", fallback)
