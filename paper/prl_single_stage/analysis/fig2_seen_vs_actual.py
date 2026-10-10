"""Fig. 2: what the optimizer scored vs what the coils actually make, fixed-boundary vs free-boundary single stage.

    python fig2_seen_vs_actual.py runs/fig2/qa4-fixed/steps.jsonl,runs/fig2/qa4-fixed-c1/steps.jsonl:77 runs/fig2/qa4-free-vac029/steps.jsonl analysis/fig2.pdf

Top: two-term QS error vs optimization step; dashed = value the optimizer scored, solid = QS of the
three-term free-boundary equilibrium of that step's coils.  Bottom: the sheet-current and
pressure-balance residuals of the coils on the step's own boundary (what a free-boundary solve must
repair).  Orange = fixed-boundary single stage, blue = free-boundary single stage (ours).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e5e4e0"


def load(spec):
    """spec = 'a.jsonl,b.jsonl:OFFSET,...': continuation segments restart step numbering at 0, OFFSET re-globalizes them."""
    rows = []
    for part in spec.split(","):
        path, _, off = part.partition(":")
        off = int(off or 0)
        seg = [json.loads(l) for l in open(path) if l.strip()]
        for r in seg:
            if "qa_actual" in r:
                r = dict(r); r["step"] += off
                if not any(abs(q["step"] - r["step"]) < 1e-9 for q in rows):
                    rows.append(r)
    rows.sort(key=lambda r: r["step"])
    return rows


fixed, free, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.linewidth": 0.6, "font.family": "sans-serif"})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 4.2), dpi=200, sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12))
for ax in (a1, a2):
    ax.set_yscale("log"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

for rows, color, name in ((fixed, ORANGE, "fixed-boundary single stage"), (free, BLUE, "free-boundary single stage (ours)")):
    s = [r["step"] for r in rows]
    a1.plot(s, [r["qa_seen"] for r in rows], color=color, ls="--", lw=1.4, label=f"{name}: scored")
    a1.plot(s, [r["qa_actual"] for r in rows], color=color, ls="-", lw=1.8, marker="o", ms=3, label=f"{name}: actual")
    a2.plot(s, [r["K0"] for r in rows], color=color, ls="-", lw=1.6, marker="o", ms=3, label=f"{name}: sheet current K")
    a2.plot(s, [r["pb0"] for r in rows], color=color, ls=":", lw=1.4, label=f"{name}: pressure balance")
a1.set_ylabel("two-term QS error")
a1.set_title(r"QA, $\langle\beta\rangle = 2.5\%$, self-consistent bootstrap current", fontsize=7, loc="left")
a1.legend(frameon=False, loc="upper right", ncol=1, fontsize=6)
a2.set_ylabel("interface residual\non the scored boundary")
a2.set_xlabel("optimization step")
a2.legend(frameon=False, loc="lower left", ncol=1, fontsize=6)
# annotate the step-0 gap of the fixed arm
r0 = fixed[0]
a1.annotate(f"×{r0['qa_actual']/r0['qa_seen']:.1f}  (LCFS off by {r0['lcfs_mm']:.0f} mm)", xy=(r0["step"], r0["qa_actual"]),
            xytext=(6, 2), textcoords="offset points", fontsize=6, color=ORANGE)
fig.subplots_adjust(left=0.2, right=0.98, top=0.93, bottom=0.1)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
print("wrote", out, f"fixed rows {len(fixed)}, free rows {len(free)}")
