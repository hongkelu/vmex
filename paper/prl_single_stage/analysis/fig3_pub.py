"""Fig. 3 (publication): LP QA vacuum benchmark vs published stage-two coil sets, matplotlib default colours.

    python analysis/fig3_pub.py analysis/fig3.pdf [--eval 12x12]
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
OURS, GIL, WECH, TGT = "C0", "C1", "C2", "0.45"
ROWS = {18: dict(gil="gil_L18", wech="wechsung_L18", margin="vac-L18-c2", exact="vac-L18-exact"),
        20: dict(gil="gil_L20", wech="wechsung_L20", margin="vac-L20-c3", exact="vac-L20-exact"),
        24: dict(gil="gil_L24", wech="wechsung_L24", margin="vac-L24-wech-c1", exact="vac-L24-wech-feas2")}
THREE = dict(gil="gil_3coil_L18", margin="vac-3coil-c2", exact="vac-3coil-exact")
fallback = []


def row(name, ours=False):
    d = EV / (f"ours_{name}" if ours else name)
    if (d / "row.json").exists():
        r = json.load(open(d / "row.json")); return r["qa_total"], r["qa_surfaces"], r["qa_profile"]
    if not ours and (EV8 / name / "row.json").exists():
        r = json.load(open(EV8 / name / "row.json")); fallback.append(name); return r["qa_total"], r["qa_surfaces"], r["qa_profile"]
    if ours:
        cands = sorted((p for p in (P / "runs/batch").glob(name + "*") if p.is_dir() and re.fullmatch(re.escape(name) + r"(-c\d+)?", p.name)),
                       key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
        for c in reversed(cands):
            m = c / "metrics.jsonl"
            if m.exists() and m.stat().st_size:
                fallback.append(name); return json.loads(m.read_text().splitlines()[-1])["qa"], None, None
    return None, None, None


plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.5, "axes.linewidth": 0.8,
                     "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=300, gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.35))
t = json.load(open(EV / "target.json" if (EV / "target.json").exists() else EV8 / "target.json"))

# (a) QS vs coil-length budget
ms = 7
for x, r in ROWS.items():
    a.plot(x - 0.18, row(r["gil"])[0], "o", color=GIL, ms=ms, zorder=3)
    a.plot(x + 0.18, row(r["wech"])[0], "s", color=WECH, ms=ms, zorder=3)
    if (v := row(r["margin"], True)[0]) is not None:
        a.plot(x, v, "D", color=OURS, ms=ms - 1, zorder=4)
    if (v := row(r["exact"], True)[0]) is not None:
        a.plot(x, v, "D", color=OURS, ms=ms + 1, mfc="white", mew=1.6, zorder=4)
x3 = 18.9  # the 3-coil set shares the 18 m budget; drawn beside the 4-coil points
a.plot(x3, row(THREE["gil"])[0], "^", color=GIL, ms=ms + 1, zorder=3)
if (v := row(THREE["margin"], True)[0]) is not None:
    a.plot(x3, v, "^", color=OURS, ms=ms, zorder=4)
if (v := row(THREE["exact"], True)[0]) is not None:
    a.plot(x3, v, "^", color=OURS, ms=ms + 2, mfc="white", mew=1.6, zorder=4)
a.axhline(t["qa_total"], color=TGT, ls="--", lw=1.0)
a.text(24.6, t["qa_total"] * 1.25, "fixed-boundary target", color=TGT, fontsize=7, ha="right")
a.set_ylim(4e-7, 4e-3)
a.set_yscale("log"); a.set_xticks([18, 20, 24]); a.set_xlim(17.2, 24.8)
a.set_xlabel("coil-length budget per half period (m)")
a.set_ylabel("QS error of the equilibrium the coils make")
a.grid(axis="y", color="0.9", lw=0.6); a.set_axisbelow(True)
a.set_title("(a)", loc="left", fontweight="bold")
handles = [Line2D([], [], marker="o", color=GIL, ls="", ms=ms, label="Gil et al. 2026, stage two"),
           Line2D([], [], marker="s", color=WECH, ls="", ms=ms, label="Wechsung et al. 2022, stage two"),
           Line2D([], [], marker="D", color=OURS, ls="", ms=ms - 1, label="this work, constraints tighter than theirs"),
           Line2D([], [], marker="D", color=OURS, ls="", ms=ms + 1, mfc="white", mew=1.6, label="this work, their exact constraints"),
           Line2D([], [], marker="^", color="0.3", ls="", ms=ms + 1, label="3 coils per half period (18 m)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handletextpad=0.4)

# (b) radial profiles of the 18 m rows
for name, ours, color, ls, lab in ((ROWS[18]["gil"], False, GIL, "-", "Gil, 4 coils"),
                                   (ROWS[18]["wech"], False, WECH, "-", "Wechsung, 4 coils"),
                                   (ROWS[18]["exact"], True, OURS, "-", "this work, 4 coils"),
                                   (THREE["gil"], False, GIL, "--", "Gil, 3 coils"),
                                   (THREE["exact"], True, OURS, "--", "this work, 3 coils")):
    q, s, prof = row(name, ours)
    if prof:
        b.plot(s, prof, color=color, ls=ls, label=lab)
b.plot(t["qa_surfaces"], t["qa_profile"], color=TGT, ls=":", lw=1.2, label="fixed-boundary target")
b.set_yscale("log"); b.set_xlim(0.05, 1.02)
b.set_xlabel("normalized toroidal flux $s$"); b.set_ylabel("QS error per flux surface")
b.grid(axis="y", color="0.9", lw=0.6); b.set_axisbelow(True)
b.set_title("(b)  18 m per half period", loc="left", fontweight="bold")
b.legend(loc="center right", frameon=True, framealpha=0.95, edgecolor="0.85", ncol=1)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.3, top=0.91)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out, "| not yet at common resolution:", sorted(set(fallback)))
