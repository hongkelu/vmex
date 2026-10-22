"""Poincare plots, surface overlays, drift against s and iota profiles for the free-boundary methods (vacuum).

    python fieldlines.py OUT CONFIG [CONFIG ...]      (CONFIG as in fbbench.py; vacuum configurations only)

For each configuration: NESTOR + VMEX (from the deck), the three-term solve with equal-arc labelling at 1e-3 (C) and
with no labelling rows (NL), both from the deck boundary.  Field lines of the coil (vacuum) field are traced with RK4
in phi from points on a ray from the magnetic axis to 1.15 minor radii beyond the LCFS, and from 32 points on each
method's surfaces s = 0.2 ... 1; crossings of phi = 0 are plotted and their RMS distance from the starting surface
is the drift.  Writes OUT/<config>.png and OUT/<config>.json.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).parent))
from fbbench import config  # noqa: E402

from dataclasses import replace  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core.multigrid import _nestor_boundary  # noqa: E402

S_DRIFT = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
S_OVERLAY = (0.25, 0.5, 0.75, 1.0)
COLORS = {"NESTOR": "tab:blue", "C 1e-3": "tab:red", "no label": "tab:green"}


def surface(w, s, theta, phi=0.0):
    j = int(round(s * (np.asarray(w.rmnc).shape[0] - 1)))
    ang = np.outer(theta, np.asarray(w.xm)) - np.asarray(w.xn) * phi
    return np.cos(ang) @ np.asarray(w.rmnc)[j], np.sin(ang) @ np.asarray(w.zmns)[j]


def tracer(field, nfp, steps=128):
    h = 2 * np.pi / (nfp * steps)

    def rhs(phi, y):
        BR, Bp, BZ = field.b_cyl(y[0], jnp.full_like(y[0], phi), y[1])
        return jnp.stack([y[0] * BR / Bp, y[0] * BZ / Bp])

    def step(y, phi):
        k1 = rhs(phi, y)
        k2 = rhs(phi + h / 2, y + h / 2 * k1)
        k3 = rhs(phi + h / 2, y + h / 2 * k2)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + rhs(phi + h, y + h * k3))

    period = jax.jit(lambda y: jax.lax.fori_loop(0, steps, lambda k, y: step(y, k * h), y))

    def trace(R, Z, periods):
        y, out = jnp.asarray([R, Z]), []
        for _ in range(periods):
            y = period(y)
            out.append(np.asarray(y))
        return np.stack(out, axis=-1)  # (2, lines, periods)
    return trace


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for cname in sys.argv[2:]:
        tag = cname.split("/")[-1].replace(":", "_")
        inp, field, vacuum, _, options = config(cname)
        assert vacuum, "field lines are the ground truth in vacuum only"
        nfp = int(inp.nfp)
        walls, seconds = {}, {}
        t0 = time.perf_counter()
        walls["NESTOR"] = opt.solve_equilibrium(replace(_nestor_boundary(inp, external_field=field), lfreeb=False,
                                                        mgrid_file="NONE")).wout
        seconds["NESTOR"] = time.perf_counter() - t0
        for name, kw in (("C 1e-3", dict(labelling="equal_arc", label_weight=1e-3)),
                         ("no label", dict(labelling="start", label_weight=0.0))):
            t0 = time.perf_counter()
            walls[name] = fvc.solve_free_boundary_three_term(inp, external_field=field, **kw, **options).equilibrium.wout
            seconds[name] = time.perf_counter() - t0
        trace = tracer(field, nfp)
        theta = np.linspace(0, 2 * np.pi, 721)
        dense = np.linspace(0, 2 * np.pi, 2881)
        result = dict(config=cname, seconds=seconds, drift_mm={}, iota={})
        # the Poincare ray: from the (NESTOR) axis outward along theta = 0 to 1.15 x the LCFS's minor radius
        w0 = walls["NESTOR"]
        R_axis, Z_axis = float(np.sum(np.asarray(w0.raxis_cc))), 0.0
        R_lcfs, Z_lcfs = surface(w0, 1.0, np.array([0.0]))
        a = float(R_lcfs[0] - R_axis)
        ray_R = R_axis + np.linspace(0.02, 1.15, 40) * a
        poincare = trace(ray_R, np.full_like(ray_R, Z_axis + float(Z_lcfs[0]) * 0), 200)
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        ax = axes[0]
        ax.plot(poincare[0].ravel(), poincare[1].ravel(), ",", color="k", alpha=0.6)
        for name, w in walls.items():
            for s in S_OVERLAY:
                R, Z = surface(w, s, theta)
                ax.plot(R, Z, color=COLORS[name], lw=1.4 if s == 1.0 else 0.7,
                        ls="-" if s == 1.0 else "--", label=name if s == 1.0 else None)
        ax.set(aspect="equal", xlabel="R [m]", ylabel="Z [m]", title=f"{cname.split('/')[-1]}: field lines (phi = 0)")
        ax.legend(fontsize=8)
        # zoom on the LCFS's outboard edge
        ax2 = axes[1]
        ax2.plot(poincare[0].ravel(), poincare[1].ravel(), ".", ms=0.6, color="k", alpha=0.6)
        for name, w in walls.items():
            R, Z = surface(w, 1.0, theta)
            ax2.plot(R, Z, color=COLORS[name], lw=1.2, label=f"{name} LCFS")
        ax2.set(xlim=(R_lcfs[0] - 0.25 * a, R_lcfs[0] + 0.2 * a), ylim=(-0.3 * a, 0.3 * a), aspect="equal",
                title="outboard edge (zoom)", xlabel="R [m]")
        ax2.legend(fontsize=8)
        ax3 = axes[2]
        for name, w in walls.items():
            drifts = []
            for s in S_DRIFT:
                th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                R, Z = surface(w, s, th)
                pts = trace(R, Z, 30)  # (2, 32, 30)
                Rd, Zd = surface(w, s, dense)
                P = pts.reshape(2, -1).T
                d = np.min(np.hypot(P[:, None, 0] - Rd[None], P[:, None, 1] - Zd[None]), axis=1)
                drifts.append(1e3 * float(np.sqrt(np.mean(d ** 2))))
            result["drift_mm"][name] = dict(zip(map(str, S_DRIFT), drifts))
            iota = np.abs(np.asarray(w.iotaf))
            result["iota"][name] = [float(iota[0]), float(iota[-1])]
            ax3.plot(S_DRIFT, drifts, "o-", color=COLORS[name], label=f"{name} (iota {iota[0]:.4f} -> {iota[-1]:.4f})")
        ax3.set(yscale="log", xlabel="s of the starting surface", ylabel="RMS drift of phi = 0 crossings [mm]",
                title="drift against s (30 transits)")
        ax3.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{tag}.png", dpi=130)
        plt.close(fig)
        # iota profiles with the nearest low-order rationals
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, w in walls.items():
            ax.plot(np.linspace(0, 1, np.asarray(w.iotaf).size), np.abs(np.asarray(w.iotaf)), color=COLORS[name],
                    label=name)
        lo, hi = ax.get_ylim()
        for p in range(1, 8):
            for q in range(1, 20):
                if lo <= p / q <= hi and np.gcd(p, q) == 1 and q <= 12:
                    ax.axhline(p / q, color="gray", lw=0.6, ls=":")
                    ax.text(1.0, p / q, f" {p}/{q}", fontsize=7, va="center")
        ax.set(xlabel="s", ylabel="|iota|", title=f"{cname.split('/')[-1]}: iota and rationals (q <= 12)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{tag}_iota.png", dpi=130)
        plt.close(fig)
        (out / f"{tag}.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
