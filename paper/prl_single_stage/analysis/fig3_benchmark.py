"""Fig. 3: the LP QA vacuum benchmark (nfp 2, A 6) against published stage-two coil sets.

    python analysis/fig3_benchmark.py analysis/fig3_benchmark.pdf [--eval 12x12]

(a) QS error of the equilibrium the coils actually make vs the coil-length budget: Gil 2026 (orange), Wechsung 2022
    (aqua), ours (blue; filled = margins stricter than theirs, open = exactly their thresholds); triangles = 3 coils.
(b) Radial QS profile of the tightest rows (18 m: 4 coils and 3 coils): published vs ours (exact thresholds).
Every value from ONE evaluator at one resolution (eval_coilsets_vacuum.py, runs/lpqa_vacuum_eval_<tag>); points not
yet rescored fall back to the 8x8 evaluator (published) or the run's own metrics (ours) and are listed on stdout.
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
ROWS = {18: dict(gil="gil_L18", wech="wechsung_L18", margin="vac-L18-c2", exact="vac-L18-exact"),
        20: dict(gil="gil_L20", wech="wechsung_L20", margin="vac-L20-c3", exact="vac-L20-exact"),
        24: dict(gil="gil_L24", wech="wechsung_L24", margin="vac-L24-wech-c1", exact="vac-L24-wech-feas2")}
THREE = dict(gil="gil_3coil_L18", margin="vac-3coil-c2", exact="vac-3coil-exact")
fallback = []


def row(name, ours=False):
    """(qa_total, s surfaces, profile) from the common evaluator; fallbacks noted."""
    d = EV / (f"ours_{name}" if ours else name)
    if (d / "row.json").exists():
        r = json.load(open(d / "row.json"))
        return r["qa_total"], r["qa_surfaces"], r["qa_profile"]
    if not ours and (EV8 / name / "row.json").exists():
        r = json.load(open(EV8 / name / "row.json")); fallback.append(name)
        return r["qa_total"], r["qa_surfaces"], r["qa_profile"]
    if ours:
        import re
        cands = sorted((p for p in (P / "runs/batch").glob(name + "*") if p.is_dir() and re.fullmatch(re.escape(name) + r"(-c\d+)?", p.name)),
                       key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
        for c in reversed(cands):
            m = c / "metrics.jsonl"
            if m.exists() and m.stat().st_size:
                fallback.append(name)
                return json.loads(m.read_text().splitlines()[-1])["qa"], None, None
    return None, None, None


plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "font.family": "sans-serif"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=200, gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
for ax in (a, b):
    ax.set_yscale("log"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# (a) QS vs budget
first = True
for x, r in ROWS.items():
    lab = (lambda s: s if first else None)
    a.plot(x, row(r["gil"])[0], "o", color=ORANGE, ms=6, label=lab("Gil et al. 2026 (stage two)"))
    a.plot(x, row(r["wech"])[0], "s", color=AQUA, ms=6, label=lab("Wechsung et al. 2022 (stage two)"))
    if (v := row(r["margin"], True)[0]) is not None:
        a.plot(x, v, "D", color=BLUE, ms=5, label=lab("ours, constraints stricter than theirs"))
    if (v := row(r["exact"], True)[0]) is not None:
        a.plot(x, v, "D", color=BLUE, ms=7, mfc="white", mew=1.4, label=lab("ours, constraints exactly theirs"))
    first = False
a.plot(18, row(THREE["gil"])[0], "^", color=ORANGE, ms=7)
if (v := row(THREE["margin"], True)[0]) is not None:
    a.plot(18, v, "^", color=BLUE, ms=6)
if (v := row(THREE["exact"], True)[0]) is not None:
    a.plot(18, v, "^", color=BLUE, ms=8, mfc="white", mew=1.4)
a.annotate("triangles: 3 coils per half period", xy=(18, row(THREE["gil"])[0]), xytext=(8, 2), textcoords="offset points", fontsize=6, color=MUTED)
tgt = EV / "target.json" if (EV / "target.json").exists() else EV8 / "target.json"
t = json.load(open(tgt))
a.axhline(t["qa_total"], color=MUTED, ls="--", lw=1)
a.annotate("fixed-boundary target", xy=(24, t["qa_total"]), xytext=(0, 3), textcoords="offset points", fontsize=6, color=MUTED, ha="right")
a.set_xticks([18, 20, 24]); a.set_xlim(17, 25)
a.set_xlabel("coil length budget per half period (m at $R_0$ = 1 m)")
a.set_ylabel("QS error of the equilibrium the coils make")
a.legend(frameon=False, loc="upper right")
a.set_title("(a)", loc="left", fontsize=8)

# (b) radial profiles, 18 m rows
for name, ours, color, ls, lab in ((THREE["gil"], False, ORANGE, "-.", "Gil, 3 coils"), (THREE["exact"], True, BLUE, "-.", "ours, 3 coils"),
                                   (ROWS[18]["gil"], False, ORANGE, "-", "Gil, 4 coils"), (ROWS[18]["wech"], False, AQUA, "-", "Wechsung, 4 coils"),
                                   (ROWS[18]["exact"], True, BLUE, "-", "ours, 4 coils")):
    q, s, prof = row(name, ours)
    if prof:
        b.plot(s, prof, color=color, ls=ls, lw=1.6, label=lab)
b.plot(t["qa_surfaces"], t["qa_profile"], color=MUTED, ls="--", lw=1.0, label="fixed-boundary target")
b.set_xlabel("normalized toroidal flux $s$"); b.set_ylabel("two-term QS error per surface")
b.set_title("(b)  18 m per half period", loc="left", fontsize=8)
b.legend(frameon=False, loc="lower right", ncol=2)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.9)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
print("wrote", out, "| fallbacks (not at common resolution):", sorted(set(fallback)))
