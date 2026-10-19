#!/usr/bin/env python3
"""Which vacuum CTH-like boundary is a flux surface of the coil field?  Field lines from each code's surfaces.

    python benchmarks/cth_vacuum_poincare.py DECK MGRID OUT name=wout ... [--transits 50] [--lines 32]     (GPU)

In vacuum (PRES_SCALE = CURTOR = 0) the deck's mgrid field at EXTCUR is the whole field, so a boundary is correct
only if field lines started on it stay on it, and its exact condition is the coil field's B.n = 0.  For every
wout and for the trilinear (order 1: the kernel VMEC2000, NESTOR and the three-term solve used) and tricubic
(order 3: smooth) interpolants, lines start from --lines points on the surfaces s = 0.25 ... 1 at phi = 0 and are
traced --transits toroidal transits (RK4 in phi, --steps per field period).  Their crossings of phi = 0 and of half
a field period are compared with the surface they started on.  Writes OUT/poincare.json (per wout and order:
{s: [max, RMS]} distance in mm, the coil B.n/|B| max and RMS on the LCFS) and OUT/<name>-order<k>.npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np

S = (0.25, 0.5, 0.75, 0.9, 1.0)


def wout_surfaces(path, s):
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        rmnc, zmns = np.asarray(nc["rmnc"][:]), np.asarray(nc["zmns"][:])
        j = np.round(np.asarray(s) * (rmnc.shape[0] - 1)).astype(int)
        return np.asarray(nc["xm"][:], float), np.asarray(nc["xn"][:], float), rmnc[j], zmns[j], int(nc["nfp"][:])


def curves(b, theta, phi):
    xm, xn, rmnc, zmns, _ = b
    a = np.outer(theta, xm) - xn * phi
    return np.cos(a) @ rmnc.T, np.sin(a) @ zmns.T  # (theta, surfaces)


def curve_distance(R, Z, Rc, Zc):
    P = np.stack([R, Z], -1)[:, None, :]
    A = np.stack([Rc, Zc], -1)[None]
    AB = np.roll(A, -1, axis=1) - A
    t = np.clip(np.sum((P - A) * AB, -1) / np.maximum(np.sum(AB * AB, -1), 1e-300), 0, 1)
    return np.min(np.linalg.norm(P - (A + t[..., None] * AB), axis=-1), axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("deck", type=Path)
    p.add_argument("mgrid", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("wouts", nargs="+", help="name=path")
    p.add_argument("--lines", type=int, default=32)
    p.add_argument("--transits", type=int, default=50)
    p.add_argument("--steps", type=int, default=144, help="RK4 steps per field period (a multiple of the 36 planes)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import vmex as vj
    from vmex.core.mgrid import MgridField

    inp = vj.VmecInput.from_file(str(a.deck))
    wouts = dict(w.split("=", 1) for w in a.wouts)
    record = {}
    for order in (1, 3):
        field = MgridField.from_input(inp, a.mgrid, order=order)
        nfp = int(field.nfp)
        h = 2 * np.pi / (nfp * a.steps)

        def rhs(phi, y, field=field):
            BR, Bp, BZ = field.b_cyl(y[0], jnp.full_like(y[0], phi), y[1])
            return jnp.stack([y[0] * BR / Bp, y[0] * BZ / Bp])

        def rk4(y, phi, rhs=rhs, h=h):
            k1 = rhs(phi, y)
            k2 = rhs(phi + h / 2, y + h / 2 * k1)
            k3 = rhs(phi + h / 2, y + h / 2 * k2)
            k4 = rhs(phi + h, y + h * k3)
            return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        @jax.jit
        def period(y, phi0, rk4=rk4, h=h):
            def body(carry, k):
                nxt = rk4(carry, phi0 + k * h)
                return nxt, nxt
            return jax.lax.scan(body, y, jnp.arange(a.steps))

        half = a.steps // 2
        theta0 = np.linspace(0, 2 * np.pi, a.lines, endpoint=False)
        dense = np.linspace(0, 2 * np.pi, 2881)
        for name, path in wouts.items():
            b = wout_surfaces(path, S)
            R0, Z0 = curves(b, theta0, 0.0)
            y = jnp.asarray(np.stack([R0.T.ravel(), Z0.T.ravel()]))
            at0, athalf = [], []
            for k in range(a.transits * nfp):
                y, ys = period(y, k * 2 * np.pi / nfp)
                at0.append(np.asarray(y))
                athalf.append(np.asarray(ys[half - 1]))
            crossings = np.stack([np.stack(at0), np.stack(athalf)])  # (planes, crossings, 2, surfaces * lines)
            per_surface = {}
            for i, si in enumerate(S):
                distances = []
                for points, phi in zip(crossings[..., i * a.lines:(i + 1) * a.lines], (0.0, np.pi / nfp)):
                    R, Z = points[:, 0, :].ravel(), points[:, 1, :].ravel()
                    ok = np.isfinite(R) & np.isfinite(Z)
                    Rc, Zc = curves(b, dense, phi)
                    distances.append(curve_distance(R[ok], Z[ok], Rc[:, i], Zc[:, i]))
                dist = np.concatenate(distances)
                per_surface[f"{si:g}"] = [round(1e3 * float(dist.max()), 4), round(1e3 * float(np.sqrt(np.mean(dist**2))), 4)]
            # the exact vacuum condition: the coil field's B.n/|B| on the LCFS
            xm, xn, rmnc, zmns, _ = b
            th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 128, endpoint=False),
                                 np.linspace(0, 2 * np.pi / nfp, 72, endpoint=False), indexing="ij")
            c, s_ = np.cos(th[..., None] * xm - ph[..., None] * xn), np.sin(th[..., None] * xm - ph[..., None] * xn)
            R, Z = c @ rmnc[-1], s_ @ zmns[-1]
            e_theta = np.stack([(-s_ * xm) @ rmnc[-1], 0 * R, (c * xm) @ zmns[-1]])
            e_phi = np.stack([(s_ * xn) @ rmnc[-1], R, (-c * xn) @ zmns[-1]])
            normal = np.cross(e_phi, e_theta, axis=0)
            B = np.stack([np.asarray(x) for x in field.b_cyl(jnp.asarray(R), jnp.asarray(ph), jnp.asarray(Z))])
            f = np.sum(B * normal, 0) / np.linalg.norm(B, axis=0) / np.linalg.norm(normal, axis=0)
            row = dict(surfaces=per_surface, lost=int(np.sum(~np.all(np.isfinite(crossings[:, :, 0, :]), axis=(0, 1)))),
                       coil_bn_max=float(np.abs(f).max()), coil_bn_rms=float(np.sqrt(np.mean(f**2))))
            record[f"{name}/order{order}"] = row
            print(name, f"order {order}", json.dumps(row), flush=True)
            np.savez(a.out / f"{name}-order{order}.npz", crossings=crossings, s=np.asarray(S), lines=a.lines,
                     nfp=nfp, wout=str(path))
            (a.out / "poincare.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
