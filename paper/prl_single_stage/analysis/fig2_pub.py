"""Fig. 2 (publication): what the optimizer scored vs what the coils make, fixed- vs free-boundary single stage.

    python analysis/fig2_pub.py FIXED_SPEC FREE_SPEC analysis/fig2.pdf
    spec = steps.jsonl[,steps.jsonl:OFFSET,...]  (continuations restart at step 0; OFFSET re-globalizes them)
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE, FIXED = "C0", "C1"


def load(spec):
    rows = {}
    for part in spec.split(","):
        path, _, off = part.partition(":")
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                if "qa_actual" in r:
                    r = dict(r); r["step"] += int(off or 0)
                    rows.setdefault(r["step"], r)
    return [rows[k] for k in sorted(rows)]


fixed, free, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "lines.linewidth": 1.5, "axes.linewidth": 0.8, "font.family": "sans-serif",
                     "mathtext.fontset": "dejavusans"})
fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.4), dpi=300, sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.1))
for rows, color, name in ((fixed, FIXED, "fixed-boundary single stage"), (free, FREE, "free-boundary single stage")):
    s = [r["step"] for r in rows]
    a.plot(s, [r["qa_seen"] for r in rows], color=color, ls="--", lw=1.3, label=f"{name}: scored")
    a.plot(s, [r["qa_actual"] for r in rows], color=color, ls="-", marker="o", ms=2.8, label=f"{name}: actual")
    b.plot(s, [r["K0"] for r in rows], color=color, ls="-", marker="o", ms=2.8, label=name)
r0 = fixed[0]
a.text(0.98, 0.47, f"step 0, fixed-boundary arm:\nactual / scored = {r0['qa_actual'] / r0['qa_seen']:.1f}, boundary off by {r0['lcfs_mm']:.0f} mm",
       transform=a.transAxes, fontsize=6.5, color=FIXED, ha="right", va="bottom")
a.set_yscale("log"); b.set_yscale("log")
a.set_ylabel("two-term QS error")
b.set_ylabel("sheet-current residual\non the scored boundary")
b.set_xlabel("optimization step")
a.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.85")
b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.85")
for ax, lab in ((a, "(a)"), (b, "(b)")):
    ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="top")
fig.suptitle(r"QA, $A$ = 4, $\langle\beta\rangle$ = 2.5 %, self-consistent bootstrap current", fontsize=8, y=0.95)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out, f"fixed rows {len(fixed)}, free rows {len(free)}")
