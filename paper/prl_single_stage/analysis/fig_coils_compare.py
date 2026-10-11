"""Coil shapes: this work vs Gil et al. 2026 vs Wechsung et al. 2022 at the same coil-length budget (18 / 20 / 24 m per hfp).

    python analysis/fig_coils_compare.py analysis/fig_coils_compare.pdf

Top row: top view (x, y) of the base coils of one half period over the LP QA target outline (midplane);
bottom row: the same coils projected on the (R, Z) plane with the target cross-sections at phi = 0 and phi = pi/nfp.
Ours = latest saved step of runs/batch/pareto-L<L>* (C0); Gil (C1) and Wechsung (C2) from coils/lpqa_*.json.
Prints per-set coil metrics (length, max curvature, max MSC, coil-coil distance) at R0 = 1 m.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import netCDF4
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parents[1]
out = sys.argv[1]
OURS, GIL, WECH, TGT = "C0", "C1", "C2", "0.3"
NQ = 400


def latest_coils(prefix):
    cands = sorted((p for p in (P / "runs/batch").glob(prefix + "*") if p.is_dir() and re.fullmatch(re.escape(prefix) + r"(-c\d+)?", p.name)),
                   key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
    for d in reversed(cands):
        fs = sorted(d.glob("coils.step*.json"), key=lambda q: int(re.search(r"step(\d+)", q.name).group(1)))
        if fs:
            return fs[-1]
    return None


def load(fn):
    d = json.load(open(fn))
    dofs = np.asarray(d.get("dofs_curves_raw", d.get("dofs_curves")), float)
    order, nfp = int(d["order"]), int(d["nfp"])
    t = np.arange(NQ) / NQ
    g = np.zeros((len(dofs), NQ, 3)); dg = np.zeros_like(g); ddg = np.zeros_like(g)
    for b, c in enumerate(dofs):
        for i in range(3):
            g[b, :, i] = c[i, 0]
            for j in range(1, order + 1):
                w = 2 * np.pi * j
                s, co = np.sin(w * t), np.cos(w * t)
                g[b, :, i] += c[i, 2 * j - 1] * s + c[i, 2 * j] * co
                dg[b, :, i] += w * (c[i, 2 * j - 1] * co - c[i, 2 * j] * s)
                ddg[b, :, i] += -w * w * (c[i, 2 * j - 1] * s + c[i, 2 * j] * co)
    speed = np.linalg.norm(dg, axis=-1)
    L = speed.mean(axis=1)
    kappa = np.linalg.norm(np.cross(dg, ddg), axis=-1) / speed**3
    msc = (kappa**2 * speed).mean(axis=1) / L
    # all symmetry copies for the coil-coil distance
    copies = []
    for k in range(nfp):
        phi = 2 * np.pi * k / nfp
        rot = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]).T
        for gg in g:
            copies.append(gg @ rot); copies.append((gg * np.array([1, -1, -1])) @ rot)
    cc = min(np.linalg.norm(a[:, None] - b[None], axis=-1).min() for i, a in enumerate(copies) for b in copies[i + 1:])
    return dict(g=g, nfp=nfp, L=L, kmax=kappa.max(axis=1), msc=msc, cc=cc, total=L.sum())


# target LCFS (vacuum LP QA, R0 = 1): midplane outline and cross-sections
tw = next(iter((P / "runs/lpqa_vacuum_eval_12x12").glob("*/wout.nc")), None) or next((P / "runs/lpqa_vacuum_eval").glob("*/wout.nc"))
w = netCDF4.Dataset(tw)
xm, xn, rmnc, zmns = np.array(w["xm"]), np.array(w["xn"]), np.array(w["rmnc"])[-1], np.array(w["zmns"])[-1]
nfp = int(np.array(w["nfp"]))


def lcfs(theta, phi):
    ang = xm[:, None] * theta[None] - xn[:, None] * phi[None]
    return rmnc @ np.cos(ang), zmns @ np.sin(ang)


SETS = {18: ("pareto-L18", "lpqa_gil_L18", "lpqa_wechsung_L18"), 20: ("pareto-L20", "lpqa_gil_L20", "lpqa_wechsung_L20"),
        24: ("pareto-L24", "lpqa_gil_L24", "lpqa_wechsung_L24")}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "lines.linewidth": 1.2, "axes.linewidth": 0.8, "font.family": "sans-serif"})
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9), dpi=300, gridspec_kw=dict(wspace=0.25, hspace=0.3))
print(f"{'set':22s} {'L/hfp':>6s}  {'coil lengths':28s} {'max kappa':>9s} {'max MSC':>8s} {'cc':>6s}")
for col, (L, names) in enumerate(SETS.items()):
    a, b = axes[0, col], axes[1, col]
    phi_hp = np.linspace(0, np.pi / nfp, 200)
    Rin = np.array([lcfs(np.array([np.pi]), np.array([p]))[0][0] for p in phi_hp])
    Rout = np.array([lcfs(np.array([0.0]), np.array([p]))[0][0] for p in phi_hp])
    for R in (Rin, Rout):
        a.plot(R * np.cos(phi_hp), R * np.sin(phi_hp), color=TGT, lw=0.8, ls=":")
    th = np.linspace(0, 2 * np.pi, 200)
    for p, ls in ((0.0, "-"), (np.pi / nfp, "--")):
        R, Z = lcfs(th, np.full_like(th, p)); b.plot(R, Z, color=TGT, lw=0.8, ls=ls)
    for name, color, lab in ((latest_coils(names[0]), OURS, "this work"), (P / "coils" / f"{names[1]}.json", GIL, "Gil"), (P / "coils" / f"{names[2]}.json", WECH, "Wechsung")):
        if name is None or not Path(name).exists():
            continue
        c = load(name)
        tag = f"{lab} L{L}" + (f" ({Path(name).parent.name} {Path(name).stem})" if lab == "this work" else "")
        print(f"{tag[:22]:22s} {c['total']:6.2f}  {np.array2string(c['L'], precision=2):28s} {c['kmax'].max():9.2f} {c['msc'].max():8.2f} {c['cc']:6.3f}")
        for g in c["g"]:
            a.plot(g[:, 0], g[:, 1], color=color, lw=1.0, alpha=0.9)
            b.plot(np.hypot(g[:, 0], g[:, 1]), g[:, 2], color=color, lw=1.0, alpha=0.9)
    a.set_aspect("equal"); b.set_aspect("equal")
    a.set_title(f"{'abc'[col]}) {L} m per half period", loc="left", fontweight="bold")
    a.set_xlabel("x (m)"); b.set_xlabel("R (m)")
    if col == 0:
        a.set_ylabel("y (m)"); b.set_ylabel("Z (m)")
    a.set_xlim(0.0, 1.75); a.set_ylim(-0.1, 1.7); b.set_xlim(0.3, 1.8); b.set_ylim(-0.75, 0.75)
handles = [Line2D([], [], color=OURS, label="this work, free-boundary single stage"), Line2D([], [], color=GIL, label="Gil et al. 2026"),
           Line2D([], [], color=WECH, label="Wechsung et al. 2022"), Line2D([], [], color=TGT, ls=":", label="target LCFS (midplane / cross-sections)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.12)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out)
