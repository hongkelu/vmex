"""Force-vs-QS Pareto figure (the one that matters): every re-solved archive set inside ONE constraint set as a cloud, its lower
envelope, Gil's 25 m points, and our single-stage fronts — all solved with the same three-term free-boundary evaluator
(12x12 / VC 64), y = QS error of the equilibrium the coils actually make.

    python analysis/fig_force_qs.py analysis/fig_force_qs.pdf [--scan]

Default = Gil's PRE thresholds as bounds (kappa <= 5, MSC <= 5, cc >= 0.1 m, cs >= 0.15 m; 3570 archive sets); --scan = Gil's
a-posteriori filter (kappa < 12, l < 6.5 m, cc > 0.083, cs > 0.166).  Ours: box mode A/B (box + 25 m); scan mode
25GA/30GA (Gil-matched families: scan filter + 25 / 30 m, plasma pinned); latest saved step, re-solved row when present (filled) else the run value (hollow, listed).
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
mode = "scan" if "--scan" in sys.argv else "box"
cloud = {("hurfront_" + r["uuid"][:8]): r for r in json.load(open(P / "runs/hurwitz_cloud.json"))}
M = json.load(open(P / "runs/force_front_metrics.json"))
EV = P / "runs/lpqa_vacuum_eval_12x12"
GIL = "C1"


def inside(r):
    if mode == "box":
        return r["kappa"] <= 5 and r["msc"] <= 5 and r["cc"] >= 0.1 and r["cs"] >= 0.15
    return r["kappa"] < 12 and r["maxl"] < 6.5 and r["cs"] > 0.166 and r["cc"] > 0.083


EV8 = P / "runs/lpqa_vacuum_eval_8x8"
invalid = []
import os
HAVE = {d.name for d in os.scandir(EV) if d.is_dir()}            # one directory listing instead of ~16k stat calls
HAVE8 = {d.name for d in os.scandir(EV8) if d.is_dir()} if EV8.exists() else set()


def qs_of(name, allow_8x8=True):
    """QS of the re-solved equilibrium; a solve is valid only if its three-term B.n residual on the LCFS the coils make is
    below the coil field's B.n on the fixed target (bn < bn_initial) — otherwise the free boundary did not converge."""
    for d in ((EV, EV8) if allow_8x8 else (EV,)):
        if name not in (HAVE if d is EV else HAVE8):
            continue
        f = d / name / "row.json"
        if f.exists():
            r = json.load(open(f))
            if r.get("bn_initial") is not None and r["bn"] >= r["bn_initial"]:
                invalid.append(f"{name} ({d.name}): bn {r['bn']:.1e} >= bn_initial {r['bn_initial']:.1e}"); return None
            return r["qa_total"]
    return None


def latest_run(prefix):
    c = sorted((p for p in (P / "runs/batch").glob(prefix + "*") if p.is_dir() and re.fullmatch(re.escape(prefix) + r"(-c\d+)?", p.name)),
               key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
    return c[-1] if c else None


# re-solved archive sets inside the constraint set (12x12 rows only: 8x8 underestimates QS by 10-57 %, it is screening only);
# x = archive max force, which equals our SIMSOPT-convention metric to 0.3 % on the 81 sets with both
pts = []
for n, r in cloud.items():
    if inside(r):
        q = qs_of(n, allow_8x8=False)
        if q is not None:
            pts.append((M[n]["max_force"] / 1e3 if n in M else r["force"] / 1e3, q, n, r["L"]))
pts.sort()


def envelope(points):
    env, best = [], 1e9
    for p in sorted(points, key=lambda p: (p[1], p[0])):
        if p[0] < best:
            env.append(p); best = p[0]
    return sorted(env)


env = envelope(pts)                                   # all lengths (the cloud as Gil draws it)
env25 = envelope([p for p in pts if p[3] <= 25.0])    # same length budget as the 25 m families
env30 = envelope([p for p in pts if p[3] <= 30.0])
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 6.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "lines.linewidth": 1.4, "axes.linewidth": 0.8, "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
fig, (ax, axr) = plt.subplots(2, 1, figsize=(3.6, 4.2), dpi=300, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.15], hspace=0.08))


def env_value(envp, F):
    """scan envelope value at force F (best QS among envelope members with force <= F)"""
    v = [p[1] for p in envp if p[0] <= F + 0.05]
    return min(v) if v else None


ratio_rows = []
ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", color="0.72", ms=4, zorder=1)
ax.plot([p[0] for p in env25], [p[1] for p in env25], "-", color="0.25", lw=1.1, zorder=2)
ax.plot([p[0] for p in env25], [p[1] for p in env25], "o", color="0.25", ms=3.5, zorder=3)
ax.plot([p[0] for p in env30], [p[1] for p in env30], "--", color="0.45", lw=1.0, zorder=2)
for n, m in M.items():
    if n.startswith("gilpar_L25") and qs_of(n) is not None:
        ax.plot(m["max_force"] / 1e3, qs_of(n), "*", color=GIL, ms=9, zorder=4)
    if mode == "scan" and n.startswith("gilpar_L30") and qs_of(n) is not None:
        ax.plot(m["max_force"] / 1e3, qs_of(n), "*", color=GIL, ms=9, mfc="white", mew=1.0, zorder=4)
FAM = {"box": (("BA", "C0", "D", "-", "this work, box, plasma pinned ($R_0$, A, $\\iota$)"),
               ("BB", "C0", "D", "--", "this work, box, aspect ratio free"),
               ("A", "C2", "s", "-", "this work, box + 25 m, plasma pinned"),
               ("B", "C2", "s", "--", "this work, box + 25 m, aspect ratio free")),
       "scan": (("25GA", "C0", "D", "-", "this work, same constraints + 25 m, plasma pinned ($R_0$, A, $\\iota$)"),
                ("25GBf", "C0", "D", "--", "this work, same constraints + 25 m, aspect ratio free ($R_0$, $\\iota$ pinned)"),
                ("30GA", "C2", "s", "-", "this work, same constraints + 30 m, plasma pinned"))}[mode]
notes, handles = [], []
for tag, color, marker, ls, lab in FAM:
    xs, ys, hollow = [], [], []
    for thr in (7, 8, 9, 10, 10.5, 11, 12, 13, 14, 16, 18):
        name = f"ff{tag}-F{thr}" if tag[0].isdigit() else f"ff25{tag}-F{thr}"; d = latest_run(name); m = M.get("ours_" + name)
        if d is None or m is None:
            continue
        q = qs_of("ours_" + name); h = q is None
        if h:
            mm = d / "metrics.jsonl"
            if not (mm.exists() and mm.stat().st_size):
                continue
            rows = [json.loads(l) for l in mm.read_text().splitlines() if l.strip()]
            q = rows[-1]["qa"]; notes.append(f"{name}: run value at step {rows[-1]['step']} ({d.name})")
        xs.append(m["max_force"] / 1e3); ys.append(q); hollow.append(h)
    if xs:
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in o], [ys[i] for i in o], color=color, ls=ls, lw=1.3, zorder=5)
        envp = env30 if tag.startswith("30") else env25
        rs = []
        for x, y, h in zip(xs, ys, hollow):
            ax.plot(x, y, marker, color=color, ms=5.5, mfc="white" if (h or ls == "--") else color, mew=1.2, zorder=6)
            e = env_value(envp, x)
            if e:
                rs.append((x, y / e, h)); ratio_rows.append((lab, x, y, e, y / e))
        if rs:
            rs.sort()
            axr.plot([r[0] for r in rs], [r[1] for r in rs], color=color, ls=ls, lw=1.1, zorder=5)
            for x, r, h in rs:
                axr.plot(x, r, marker, color=color, ms=4.5, mfc="white" if (h or ls == "--") else color, mew=1.0, zorder=6)
        handles.append(Line2D([], [], marker=marker, color=color, ls=ls, ms=5.5, mfc="white" if ls == "--" else color, mew=1.2, label=lab))
ax.set_yscale("log"); ax.set_xlim(6.5, 17.5)
axr.axhline(1.0, color="0.25", lw=0.8); axr.set_yscale("log"); axr.set_ylim(0.2, 5)
axr.set_yticks([0.25, 0.5, 1, 2, 4]); axr.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
axr.set_ylabel("ours / scan envelope\n(same length budget)", fontsize=7)
axr.grid(axis="y", color="0.9", lw=0.6); axr.set_axisbelow(True)
for n, m in M.items():   # Gil's points against the envelope of their own budget
    if n.startswith("gilpar_L") and qs_of(n) is not None:
        e = env_value(env25 if "L25" in n else env30, m["max_force"] / 1e3)
        if e:
            axr.plot(m["max_force"] / 1e3, qs_of(n) / e, "*", color=GIL, ms=8, mfc=(GIL if "L25" in n else "white"), mew=1.0, zorder=6)
axr.set_xlabel("max coil force (kN/m; $\\langle B\\rangle$ = 0.227 T, $R_0$ = 1 m)")
ax.set_ylabel("QS error of the equilibrium the coils make")
ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
ttl = ("$\\kappa\\leq$5, MSC$\\leq$0.05 (5 /m$^2$), cc$\\geq$0.1 m, cs$\\geq$0.15 m" if mode == "box"
       else "$\\kappa<$12, $\\ell<$6.5 m, cc$>$0.083 m, cs$>$0.166 m")
ax.set_title(f"LP QA, 5 coils/hfp, {ttl}; {len(pts)} archive sets re-solved at 12x12/VC64", fontsize=6.3, loc="left")
handles = [Line2D([], [], marker=".", color="0.72", ls="", ms=6, label="Hurwitz et al. 2024 scan, inside the constraint set (re-solved)"),
           Line2D([], [], marker="o", color="0.25", ls="-", ms=3.5, label="its lower envelope, sets with L $\\leq$ 25 m"),
           Line2D([], [], color="0.45", ls="--", label="its lower envelope, sets with L $\\leq$ 30 m"),
           Line2D([], [], marker="*", color=GIL, ls="", ms=9, label="Gil et al. 2026, 25 m" + (" (outside the box)" if mode == "box" else ""))]\
          + ([Line2D([], [], marker="*", color=GIL, ls="", ms=9, mfc="white", mew=1.0, label="Gil et al. 2026, 30 m")] if mode == "scan" else []) + handles
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=5.6)
for lab, x, y, e, r in ratio_rows:
    print(f"  {lab[:38]:38s} F={x:5.2f} ours={y:.2e} envelope={e:.2e} ratio={r:.2f}")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out, f"| re-solved inside set: {len(pts)}, envelope all/L25/L30 {len(env)}/{len(env25)}/{len(env30)}, invalid solves (bn >= bn_initial): {len(invalid)}")
print("\n".join("  " + n for n in notes + invalid[:20]))
