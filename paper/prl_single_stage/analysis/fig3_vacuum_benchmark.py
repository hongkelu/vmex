"""Fig. 3 (QA row, vacuum LP QA benchmark): two-term QS error vs s for published coil sets and ours.

    python fig3_vacuum_benchmark.py analysis/data/qs_profiles_vacuum.json analysis/fig3_vacuum_qs.pdf

Lines: the prescribed target (gray dashed), the published coil sets evaluated as the free-boundary
equilibria they make (orange = Gil et al. 2026, aqua = Wechsung et al. 2022; line style by length
budget), and our free-boundary single-stage results started from them (blue).  Direct labels at the
right edge; log y.  Palette: dataviz reference categorical slots (validated).
"""
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
STYLE = {"L18": "-", "L20": "--", "L24": ":", "3coil": "-."}

data = json.load(open(sys.argv[1]))
out = sys.argv[2]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.direction": "out", "ytick.direction": "out", "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
ax.set_yscale("log")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

labels = []
for name, d in data.items():
    s, p = d["s10"], d["profile10"]
    if name == "target":
        ax.plot(s, p, color=MUTED, ls="--", lw=1.2)
        labels.append((p[-1], "LP QA target", MUTED))
        continue
    m = re.search(r"(L18|L20|L24|3coil)", name)
    budget = m.group(1) if m else "L18"
    if name.startswith("ext_"):
        color = ORANGE if "gil" in name else AQUA
        src = "Gil" if "gil" in name else "Wechsung"
        ax.plot(s, p, color=color, ls=STYLE[budget], lw=1.4)
        labels.append((p[-1], f"{src} {budget}", color))
    elif name.startswith("ours_"):
        step = re.search(r"step(\d+)", name)
        ax.plot(s, p, color=BLUE, ls=STYLE[budget], lw=1.8)
        lab = f"ours from {'Wechsung' if 'wech' in name else 'Gil'} {budget}"
        labels.append((p[-1], lab, BLUE))

# direct labels at the right edge, nudged apart in log space
labels.sort(key=lambda t: t[0])
ylast = None
for y, text, color in labels:
    if ylast is not None and y / ylast < 1.45:
        y = ylast * 1.45
    ax.annotate(text, xy=(1.0, y), xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                fontsize=6, color=color)
    ylast = y
ax.set_xlabel("normalized toroidal flux  $s$")
ax.set_ylabel("two-term QS residual per surface")
ax.set_xlim(0.1, 1.0)
ax.set_title("LP QA benchmark (nfp 2, A 6, vacuum)", fontsize=7, color=INK, loc="left")
fig.subplots_adjust(right=0.64, left=0.17, bottom=0.17, top=0.9)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"))
print("wrote", out)
