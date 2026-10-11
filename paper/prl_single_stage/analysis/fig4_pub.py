"""Fig. 4 (publication): the A4 QA showcase at <beta> 2.5 % with self-consistent bootstrap current, free-boundary single stage.

    JAX_PLATFORMS=cpu python analysis/fig4_pub.py analysis/fig4.pdf [--run qa4-free-vac029-s1] [--step N] [--alpha runs/alpha/qa4-free/qa4_free_trace.json]

(a) coils and the LCFS the coils actually make, coloured by |B|; (b) |B| in Boozer coordinates on s = 0.5;
(c) Boozer spectrum vs s (symmetric n = 0 modes grey, the largest non-symmetric modes coloured);
(d) rotational transform and bootstrap current density.  Matplotlib default colours, C0 = this work.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4

P = Path(__file__).resolve().parents[1]
out = sys.argv[1]


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


run = arg("--run", "qa4-free-vac029-s1")
cands = sorted((p for p in (P / "runs/batch").glob(run + "*") if p.is_dir() and re.fullmatch(re.escape(run) + r"(-c\d+)?", p.name)),
               key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0)
d = cands[-1]
step = int(arg("--step", max(int(re.search(r"step(\d+)", q.name).group(1)) for q in d.glob("wout.step*.nc"))))
wout, coils = d / f"wout.step{step}.nc", d / f"coils.step{step}.json"
alpha = P / arg("--alpha", "runs/alpha/qa4-free/qa4_free_trace.json")
print("run", d.name, "step", step)

# ---------------------------------------------------------------- equilibrium
w = netCDF4.Dataset(wout)
V = lambda k: np.array(w[k])
ns, nfp = int(V("ns")), int(V("nfp"))
xm, xn, xm_nyq, xn_nyq = V("xm"), V("xn"), V("xm_nyq"), V("xn_nyq")
rmnc, zmns, bmnc = V("rmnc"), V("zmns"), V("bmnc")
s_full = np.linspace(0, 1, ns)
s_half = (np.arange(ns) - 0.5) / (ns - 1)
iotaf, presf, jdotb, betatotal = V("iotaf"), V("presf"), V("jdotb"), float(V("betatotal"))
metrics = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
mrow = next((m for m in metrics if m["step"] == step), metrics[-1])

# LCFS geometry (full grid, last surface) and |B| there (half grid, last surface)
nt, nz = 64, 128
th = np.linspace(0, 2 * np.pi, nt)
ze = np.linspace(0, 2 * np.pi, nz)
TH, ZE = np.meshgrid(th, ze, indexing="ij")
ang = xm[:, None, None] * TH[None] - xn[:, None, None] * ZE[None]
R = np.einsum("m,mtz->tz", rmnc[-1], np.cos(ang))
Z = np.einsum("m,mtz->tz", zmns[-1], np.sin(ang))
ang_n = xm_nyq[:, None, None] * TH[None] - xn_nyq[:, None, None] * ZE[None]
Bs = np.einsum("m,mtz->tz", bmnc[-1], np.cos(ang_n))
X, Y = R * np.cos(ZE), R * np.sin(ZE)

# ---------------------------------------------------------------- coils (ESSOS raw dofs = simsopt CurveXYZFourier)
cj = json.load(open(coils))
dofs = np.asarray(cj.get("dofs_curves_raw", cj.get("dofs_curves")), float)   # (nbase, 3, 2o+1)
order, stellsym = int(cj["order"]), bool(cj["stellsym"])
t = np.linspace(0, 1, 200)
basis = [np.ones_like(t)]
for j in range(1, order + 1):
    basis += [np.sin(2 * np.pi * j * t), np.cos(2 * np.pi * j * t)]
basis = np.array(basis)                                                        # (2o+1, nt)
base = np.einsum("bcj,jt->bct", dofs, basis)                                    # (nbase, 3, nt)
curves = []
for k in range(nfp):
    for flip in ([False, True] if stellsym else [False]):
        for g in base:
            x, y, z = g
            if flip:
                y, z = -y, -z
            phi = 2 * np.pi * k / nfp
            curves.append(np.array([x * np.cos(phi) - y * np.sin(phi), x * np.sin(phi) + y * np.cos(phi), z]))

# ---------------------------------------------------------------- Boozer transform
import booz_xform_jax as bx

b = bx.Booz_xform()
b.read_wout(str(wout), flux=True)
b.verbose = 0
b.mboz, b.nboz = 24, 12
s_list = np.linspace(0.04, 0.98, 24)
b.register_surfaces(list(s_list) + [0.5])
b.run()
s_b = np.asarray(b.s_b)
xm_b, xn_b, B_b = np.asarray(b.xm_b), np.asarray(b.xn_b), np.asarray(b.bmnc_b)   # (mnboz, nsurf)
B00 = B_b[(xm_b == 0) & (xn_b == 0)][0]
j05 = int(np.argmin(np.abs(s_b - 0.5)))
thb = np.linspace(0, 2 * np.pi, 120)
zeb = np.linspace(0, 2 * np.pi / nfp, 120)
THB, ZEB = np.meshgrid(thb, zeb, indexing="ij")
Bbooz = np.einsum("m,mtz->tz", B_b[:, j05], np.cos(xm_b[:, None, None] * THB[None] - xn_b[:, None, None] * ZEB[None]))
nonsym = xn_b != 0
rms_nonsym = np.sqrt(np.sum((B_b[nonsym] / B00) ** 2, axis=0))
print(f"Boozer non-symmetric rms/B00: s=0.25 {rms_nonsym[np.argmin(abs(s_b-0.25))]:.2e}  s=0.5 {rms_nonsym[j05]:.2e}  s=0.98 {rms_nonsym[np.argmin(abs(s_b-0.98))]:.2e}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.5, "axes.linewidth": 0.8,
                     "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(7.0, 5.4), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.15, 1], wspace=0.32, hspace=0.42, left=0.07, right=0.98, top=0.95, bottom=0.08)
a = fig.add_subplot(gs[0, 0], projection="3d")
bax, c, dax = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# (a) coils + LCFS coloured by |B|
norm = plt.Normalize(Bs.min(), Bs.max())
a.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(norm(Bs)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1)
for q in curves:
    a.plot(*q, color="C0", lw=1.1, alpha=0.95, zorder=5)
lim = 0.78 * np.max(np.abs(np.concatenate([q[:2].ravel() for q in curves])))
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim * 0.4, lim * 0.4)
a.set_box_aspect((1, 1, 0.4), zoom=1.45); a.view_init(elev=35, azim=-55); a.set_axis_off()
sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
cb = fig.colorbar(sm, ax=a, fraction=0.03, pad=0.0, shrink=0.5, aspect=18, location="left")
cb.set_label("$|B|$ on the LCFS (T, $R_0$ = 1 m)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
a.set_title("(a)", loc="left", fontweight="bold", y=0.98)
txt = [f"$\\langle\\beta\\rangle$ = {100 * betatotal:.1f} %, self-consistent bootstrap",
       f"A = {mrow.get('aspect', float(V('aspect'))):.2f}, $\\iota$ = {abs(iotaf[0]):.2f} → {abs(iotaf[-1]):.2f}",
       f"QS error of the equilibrium the coils make: {mrow['qa']:.1e}"]
if alpha.exists():
    al = json.load(open(alpha))
    txt.append(f"{al['particles_lost']}/{al['nparticles']} alphas lost in {al['tmax']} s (s = {al['s']}, reactor scale)")
a.text2D(0.0, -0.02, "\n".join(txt), transform=a.transAxes, fontsize=6.8, va="top", ha="left", linespacing=1.35)

# (b) |B| in Boozer coordinates on s = 0.5
cf = bax.contourf(ZEB * nfp / (2 * np.pi), THB / (2 * np.pi), Bbooz, levels=18, cmap="viridis")
bax.contour(ZEB * nfp / (2 * np.pi), THB / (2 * np.pi), Bbooz, levels=18, colors="k", linewidths=0.3, alpha=0.5)
bax.set_xlabel("Boozer toroidal angle $N_{fp}\\,\\zeta_B / 2\\pi$"); bax.set_ylabel("Boozer poloidal angle $\\theta_B / 2\\pi$")
bax.set_title(f"(b)  $|B|$ in Boozer coordinates, $s$ = {s_b[j05]:.2f}", loc="left", fontweight="bold")
cb2 = fig.colorbar(cf, ax=bax, fraction=0.045, pad=0.03); cb2.ax.tick_params(labelsize=6.5); cb2.set_label("T", fontsize=7)

# (c) Boozer spectrum vs s
order_idx = [i for i in range(len(s_b)) if i != j05] + [j05]
srt = np.argsort(s_b)
for i in np.where(~nonsym & (xm_b > 0))[0]:
    c.plot(s_b[srt], np.abs(B_b[i, srt] / B00[srt]), color="0.75", lw=0.8, zorder=1)
top = np.argsort(-np.max(np.abs(B_b[nonsym] / B00), axis=1))[:6]
idx_nonsym = np.where(nonsym)[0][top]
for k, i in enumerate(idx_nonsym):
    c.plot(s_b[srt], np.abs(B_b[i, srt] / B00[srt]), color=f"C{k}", lw=1.3, zorder=3, label=f"m = {int(xm_b[i])}, n = {int(xn_b[i] / nfp)}")
c.plot(s_b[srt], rms_nonsym[srt], color="k", lw=1.6, ls="--", zorder=4, label="rms of all $n\\neq0$")
c.set_yscale("log"); c.set_xlim(0, 1); c.set_ylim(1e-5, 0.3)
c.set_xlabel("normalized toroidal flux $s$"); c.set_ylabel("$|B_{mn}| / B_{00}$")
c.set_title("(c)  Boozer spectrum (grey: $n$ = 0)", loc="left", fontweight="bold")
c.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95, edgecolor="0.85", fontsize=6.2, handlelength=1.4, columnspacing=0.9)
c.grid(axis="y", color="0.9", lw=0.6); c.set_axisbelow(True)

# (d) iota and bootstrap current
dax.plot(s_full, np.abs(iotaf), color="C0", label="$\\iota$")
dax.set_xlabel("normalized toroidal flux $s$"); dax.set_ylabel("rotational transform $\\iota$", color="C0")
dax.set_xlim(0, 1); dax.tick_params(axis="y", colors="C0")
d2 = dax.twinx()
d2.plot(s_full, jdotb / 1e3, color="C3", label="$\\langle J\\cdot B\\rangle$")
d2.set_ylabel("bootstrap $\\langle J\\cdot B\\rangle$ (kA T m$^{-2}$)", color="C3"); d2.tick_params(axis="y", colors="C3")
dax.set_title("(d)  profiles", loc="left", fontweight="bold")
dax.grid(axis="y", color="0.9", lw=0.6); dax.set_axisbelow(True)

fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out)
