"""Boozer quasisymmetry error of a WOUT: symmetry-breaking |B| modes (n != 0 for QA) against s.

    python extra/qs_error.py <wout> <outdir> [label]     (cwd: the vmex tree, GPU)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from vmex.core.boozer import run_booz_xform  # noqa: E402

wout, outdir = Path(sys.argv[1]), Path(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else wout.stem
outdir.mkdir(parents=True, exist_ok=True)
path = run_booz_xform(wout, mbooz=24, nbooz=24, outdir=outdir)
with Dataset(path) as d:
    bmnc = np.asarray(d["bmnc_b"][:])
    xm, xn = np.asarray(d["ixm_b"][:]), np.asarray(d["ixn_b"][:])
    jlist = np.asarray(d["jlist"][:])
    ns = int(d["ns_b"][:])
    nfp = int(d["nfp_b"][:])
s = (jlist - 1.5) / (ns - 1)
b00 = bmnc[:, (xm == 0) & (xn == 0)][:, 0]
broken = xn != 0
qs = np.sqrt(np.sum(bmnc[:, broken] ** 2, axis=1)) / np.abs(b00)
sym = np.sqrt(np.sum(bmnc[:, ~broken & (xm > 0)] ** 2, axis=1)) / np.abs(b00)
top = np.argsort(-np.max(np.abs(bmnc[:, broken]), axis=0))[:8]
idx = np.flatnonzero(broken)[top]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].semilogy(s, qs, "k-", lw=2, label="symmetry-breaking (n != 0)")
ax[0].semilogy(s, sym, "C0--", label="symmetric (n = 0, m > 0)")
ax[0].set(xlabel="s", ylabel="RMS |B_mn| / B_00", title="Boozer QA error")
ax[0].legend(); ax[0].grid(alpha=0.3)
for k in idx:
    ax[1].semilogy(s, np.abs(bmnc[:, k]) / np.abs(b00), label=f"m={xm[k]}, n={xn[k] // nfp} nfp")
ax[1].set(xlabel="s", ylabel="|B_mn| / B_00", title="largest symmetry-breaking modes")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
j = int(np.argmin(np.abs(s - 0.5)))
th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 128), np.linspace(0, 2 * np.pi / nfp, 128), indexing="ij")
B = sum(bmnc[j, k] * np.cos(xm[k] * th - xn[k] * ph) for k in range(xm.size))
c = ax[2].contourf(ph, th, B, 30, cmap="viridis"); fig.colorbar(c, ax=ax[2], label="|B| [T]")
ax[2].set(xlabel="Boozer phi", ylabel="Boozer theta", title=f"|B| on s = {s[j]:.2f}")
fig.suptitle(label)
fig.tight_layout()
fig.savefig(outdir / "qs_error.png", dpi=140)
np.savez(outdir / "qs_error.npz", s=s, qs=qs, sym=sym, bmnc=bmnc, xm=xm, xn=xn)
print(f"QS error (RMS n!=0 / B00): s=0.1 {np.interp(0.1, s, qs):.2e}, 0.5 {np.interp(0.5, s, qs):.2e}, "
      f"1.0 {qs[-1]:.2e}; max {qs.max():.2e}; wrote {outdir / 'qs_error.png'}")
