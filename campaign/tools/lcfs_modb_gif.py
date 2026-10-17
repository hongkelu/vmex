"""3D GIF of the coils and the LCFS over an optimization's saved steps, the LCFS coloured by |B| on it.

    JAX_PLATFORMS=cpu python extra/lcfs_modb_gif.py <title> <out.gif> <run dir> [--stride N] [--last K]

Each frame reads ``wout.stepK.nc`` (LCFS from rmnc/zmns; |B| from bmnc, extrapolated from the last two half-mesh
surfaces to the edge) and ``coils.stepK.json``; the colour range is shared by every frame.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("title")
p.add_argument("out")
p.add_argument("run", type=Path)
p.add_argument("--stride", type=int, default=5)
p.add_argument("--last", type=int, help="last step to include (default: every saved step)")
p.add_argument("--fps", type=int, default=6)
args = p.parse_args()

from essos.coils import Coils  # noqa: E402  (after argparse: imports JAX)

steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in args.run.glob("wout.step*.nc"))
steps = [s for s in steps if args.last is None or s <= args.last]
steps = sorted(set(steps[::args.stride] + steps[-1:]))
theta = np.linspace(0, 2 * np.pi, 65)
phi = np.linspace(0, 2 * np.pi, 161)
TH, PH = np.meshgrid(theta, phi, indexing="ij")


def frame_data(step):
    with netCDF4.Dataset(args.run / f"wout.step{step}.nc") as w:
        xm, xn = np.asarray(w["xm"][:]), np.asarray(w["xn"][:])
        rmnc, zmns = np.asarray(w["rmnc"][-1]), np.asarray(w["zmns"][-1])
        xm_b, xn_b = np.asarray(w["xm_nyq"][:]), np.asarray(w["xn_nyq"][:])
        bmnc = np.asarray(w["bmnc"][:])
    b_edge = 1.5 * bmnc[-1] - 0.5 * bmnc[-2]
    ang = xm[:, None, None] * TH - xn[:, None, None] * PH
    R, Z = np.tensordot(rmnc, np.cos(ang), 1), np.tensordot(zmns, np.sin(ang), 1)
    B = np.tensordot(b_edge, np.cos(xm_b[:, None, None] * TH - xn_b[:, None, None] * PH), 1)
    coils = np.asarray(Coils.from_json(str(args.run / f"coils.step{step}.json")).gamma)
    return R * np.cos(PH), R * np.sin(PH), Z, B, coils


data = {s: frame_data(s) for s in steps}
lo = min(d[3].min() for d in data.values())
hi = max(d[3].max() for d in data.values())
norm = plt.Normalize(lo, hi)
cmap = plt.get_cmap("viridis")
reach = max(np.abs(np.r_[d[0].ravel(), d[1].ravel(), d[4][..., :2].ravel()]).max() for d in data.values())

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection="3d")
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, label="|B| on the LCFS [T]")


def draw(i):
    ax.clear()
    step = steps[i]
    X, Y, Z, B, coils = data[step]
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(B)), rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)
    for c, curve in enumerate(coils):
        closed = np.vstack([curve, curve[:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="0.25", lw=1.0)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-0.6 * reach, 0.6 * reach)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-60)
    ax.set_title(f"{args.title}  step {step}")


animation = FuncAnimation(fig, draw, frames=len(steps))
animation.save(args.out, writer=PillowWriter(fps=args.fps))
draw(len(steps) - 1)
fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=130)
print(f"wrote {args.out}: {len(steps)} frames, steps {steps[0]}..{steps[-1]}, |B| {lo:.3f}..{hi:.3f} T")
