"""Maximum-J scan: the second adiabatic invariant J(alpha, s) at a range of physical pitches 1/lambda = B*.

    python <repo>/campaign/postprocess/maxj_scan.py <wout> <out.png> [--surfaces 33] [--pitches 24]     (cwd: vmex tree, GPU)

A configuration is maximum-J where dJ/ds < 0 at fixed physical lambda and field line alpha (Rodriguez, Helander &
Goodman 2024). For every B* between the smallest Bmin and the largest Bmax over the surfaces, J is computed on every
surface where that pitch is trapped (vmex's bounce kernel, as in the plot_wout J panel), dJ/ds by differences of
neighbouring surfaces at the same alpha. Writes the figure and <out>.json with the max-J fraction per pitch.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import vmex as vj
from vmex.core import plotting as pl

p = argparse.ArgumentParser()
p.add_argument("wout")
p.add_argument("out", type=Path)
p.add_argument("--surfaces", type=int, default=33)
p.add_argument("--pitches", type=int, default=24)
args = p.parse_args()

wout = vj.read_wout(args.wout)
booz = pl._boozer_summary_data(wout, n_surfaces=args.surfaces)
s = np.asarray(booz["s_b"])
theta = np.linspace(0, 2 * np.pi, 61)
zeta = np.linspace(0, 2 * np.pi / int(booz["nfp"]), 61)
b = np.stack([pl._boozer_surface_modB(booz, k, theta, zeta) for k in range(s.size)])
bmin, bmax = b.min(axis=(1, 2)), b.max(axis=(1, 2))
# deeply trapped (just above the lowest Bmin) to barely trapped (just below the highest Bmax)
bstar = np.linspace(bmin.min(), bmax.max(), args.pitches + 2)[1:-1]

rows, maps = [], {}
for bs in bstar:
    j = pl._j_invariant_map(booz, pitch=1.0 / bs)["j_map"]          # (ns_b, nalpha), NaN where not trapped
    dj = np.diff(j, axis=0) / np.diff(s)[:, None]                    # between neighbouring surfaces
    ok = np.isfinite(dj)
    frac = float((dj[ok] < 0).mean()) if ok.any() else float("nan")
    rows.append(dict(bstar=float(bs), maxj_fraction=frac, points=int(ok.sum()),
                     s_range=[float(s[np.isfinite(j).any(axis=1)].min()), float(s[np.isfinite(j).any(axis=1)].max())]
                     if np.isfinite(j).any() else None))
    maps[float(bs)] = (j, dj)
    print(f"B* = {bs:.4f} T: max-J (dJ/ds < 0) on {100 * frac:5.1f}% of {ok.sum()} (s, alpha) points", flush=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

alpha = np.linspace(0, 2 * np.pi, 96, endpoint=False)
sm = 0.5 * (s[1:] + s[:-1])
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
# (a) fraction of field lines with dJ/ds < 0, per (s, B*)
F = np.full((bstar.size, sm.size), np.nan)
for i, bs in enumerate(bstar):
    dj = maps[float(bs)][1]
    ok = np.isfinite(dj)
    F[i] = np.where(ok.any(axis=1), (np.where(ok, dj < 0, False)).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
m = ax[0, 0].pcolormesh(sm, bstar, F, cmap="RdBu", vmin=0, vmax=1, shading="nearest")
ax[0, 0].plot(s, bmin, "k--", lw=1, label=r"$B_{min}(s)$")
ax[0, 0].plot(s, bmax, "k:", lw=1, label=r"$B_{max}(s)$")
ax[0, 0].set(xlabel="s", ylabel=r"$1/\lambda = B_*$ [T]", title=r"fraction of field lines with $\partial J/\partial s<0$")
ax[0, 0].legend(loc="lower right", fontsize=8)
fig.colorbar(m, ax=ax[0, 0], label="max-J fraction (blue = max-J)")
# (b) overall max-J fraction per pitch
ax[0, 1].plot(bstar, [100 * r["maxj_fraction"] for r in rows], "o-")
ax[0, 1].set(xlabel=r"$B_*$ [T]", ylabel="% of (s, alpha) points with dJ/ds < 0", title="max-J fraction per pitch",
             ylim=(-3, 103))
ax[0, 1].grid(alpha=0.3)
# (c) alpha-averaged J(s) for every pitch, normalized to its inner value
cmap = plt.get_cmap("viridis")
for i, bs in enumerate(bstar):
    j = maps[float(bs)][0]
    jm = np.nanmean(j, axis=1) if np.isfinite(j).any() else None
    if jm is not None and np.isfinite(jm).sum() > 1:
        jm = jm / np.nanmax(jm)
        ax[0, 2].plot(s, jm, color=cmap(i / max(1, bstar.size - 1)), lw=1.2)
ax[0, 2].set(xlabel="s", ylabel=r"$\langle J\rangle_\alpha / \max$", title=r"$J(s)$ averaged over $\alpha$ (colour: $B_*$ low→high)")
ax[0, 2].grid(alpha=0.3)
# (d-f) polar maps at deeply, intermediate and barely trapped pitches
for a, i in zip(ax[1], (1, bstar.size // 2, bstar.size - 2)):
    bs = float(bstar[i]); j = maps[bs][0]
    X, Y = s[:, None] * np.cos(alpha)[None], s[:, None] * np.sin(alpha)[None]
    jj = j / (np.nanmax(j) if np.isfinite(j).any() else 1)
    c = a.contourf(np.c_[X, X[:, :1]], np.c_[Y, Y[:, :1]], np.c_[jj, jj[:, :1]], 20, cmap="viridis")
    fig.colorbar(c, ax=a, label="J / max J")
    a.set_aspect("equal"); a.set(xlabel=r"$s\cos\alpha$", ylabel=r"$s\sin\alpha$",
                                 title=f"$B_*$ = {bs:.3f} T: max-J on {100 * rows[i]['maxj_fraction']:.0f}%")
fig.suptitle(f"Maximum-J scan, {Path(args.wout).parent.parent.name}", fontsize=14)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
args.out.with_suffix(".json").write_text(json.dumps(dict(s=s.tolist(), bmin=bmin.tolist(), bmax=bmax.tolist(), pitches=rows), indent=1))
print("wrote", args.out)
