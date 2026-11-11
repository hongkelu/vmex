#!/usr/bin/env python
"""Poincare plot of a single-stage run: VMEC flux surfaces inside, traced field lines outside the plasma.

    python benchmarks/edge_poincare.py --run runs/qa --xpoint 0.98 0.73 --output runs/qa-poincare

``--run`` holds ``coils.json``, ``wout.nc`` and optionally ``pf.json`` (the layout of
``benchmarks/edge_fixed_points.py``).  Inside the boundary the flux surfaces are the WOUT's; outside, field lines of
the coils' plus the plasma's virtual-casing field (:class:`vmex.core.extender.VmecExtender`) are traced from seeds on
the plane ``phi = 0`` (:func:`vmex.core.periodic_fieldlines.trace_fieldline`) and marked on ``phi = 0`` and on the
half-period plane.  A line ends where it leaves the box, enters the plasma or comes within ``--coil-gap`` of a coil;
it is coloured by the field periods it lasted.  With ``--xpoint`` the closed line through that point is refined by
shooting and its unstable and stable manifolds are drawn.  Writes ``poincare.png`` and ``poincare.npz``.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory with coils.json and wout.nc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xpoint", type=float, nargs=2, metavar=("R", "Z"), help="a closed line's point at phi = 0")
    parser.add_argument("--periods", type=int, default=200, help="field periods traced")
    parser.add_argument("--steps", type=int, default=48, help="Runge-Kutta steps per half field period")
    parser.add_argument("--seeds", type=int, default=24, help="seeds per ray")
    parser.add_argument("--reach", type=float, default=0.22, help="length of the seed rays outside the boundary [m]")
    parser.add_argument("--gap", type=float, default=0.03, help="smallest distance of a seed to the boundary [m]")
    parser.add_argument("--coil-gap", type=float, default=0.03, help="a line ends this close to a coil [m]")
    parser.add_argument("--margin", type=float, default=0.4, help="box around the boundary [m]")
    parser.add_argument("--vc-grid", type=int, default=96, help="virtual-casing grid per field period")
    parser.add_argument("--surfaces", type=int, default=12, help="flux surfaces drawn")
    parser.add_argument("--title", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.path import Path as Polygon
    from essos.coils import Coils
    from essos.fields import BiotSavart

    from vmex.core import periodic_fieldlines as pf
    from vmex.core import virtual_casing as vc
    from vmex.core.extender import VmecExtender
    from vmex.core.wout import read_wout

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wout = read_wout(str(args.run / "wout.nc"))
    coils = Coils.from_json(str(args.run / "coils.json"))
    nfp = int(wout.nfp)
    half = np.pi / nfp
    biot_savart = BiotSavart(coils)
    coil_points = np.asarray(coils.gamma).reshape(-1, 3)
    if (args.run / "pf.json").exists():  # the driver's circular coil pairs
        from vmex.core.coil_design import SummedField, circular_coil_pairs

        saved = json.loads((args.run / "pf.json").read_text())
        pairs = circular_coil_pairs(*(np.asarray(saved[k]) for k in ("radius_m", "height_m", "current_a")))
        biot_savart = SummedField(biot_savart, BiotSavart(pairs))
        coil_points = np.concatenate((coil_points, np.asarray(pairs.gamma).reshape(-1, 3)))
    surface = vc.surface_field_data_from_wout(wout, nphi=args.vc_grid, ntheta=args.vc_grid)
    total = VmecExtender.from_surface_data(surface, external_field=lambda xyz: jax.vmap(biot_savart.B)(xyz),
                                           accuracy_check="off", near_surface="direct")
    field = lambda p, xyz: total.B(xyz)  # noqa: E731

    def section(phi, s_index):  # the WOUT's surface s_index on the plane phi
        theta = np.linspace(0.0, 2.0 * np.pi, 361)[:, None]
        angle = np.asarray(wout.xm)[None] * theta - np.asarray(wout.xn)[None] * phi
        return np.stack((np.cos(angle) @ np.asarray(wout.rmnc)[s_index], np.sin(angle) @ np.asarray(wout.zmns)[s_index]), -1)

    ns = np.asarray(wout.rmnc).shape[0]
    planes = (0.0, half)
    edges = [section(phi, ns - 1) for phi in planes]
    axis0 = section(0.0, 0)[0]
    low, high = edges[0].min(axis=0) - args.margin, edges[0].max(axis=0) + args.margin
    for e in edges[1:]:
        low, high = np.minimum(low, e.min(axis=0) - args.margin), np.maximum(high, e.max(axis=0) + args.margin)

    # seeds on phi = 0: rays from the magnetic axis through the boundary, towards the X-point and along the midplane
    def ray(direction):
        direction = np.asarray(direction) / np.linalg.norm(direction)
        t = np.linspace(0.0, 3.0, 3001)
        inside = Polygon(edges[0]).contains_points(axis0 + t[:, None] * direction)
        t_edge = t[np.argmin(inside)]
        return axis0 + (t_edge + np.linspace(args.gap, args.reach, args.seeds))[:, None] * direction

    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)]
    if args.xpoint:
        directions.append(np.asarray(args.xpoint) - axis0)
    seeds = np.concatenate([ray(d) for d in directions])
    print(f"[seeds] {len(seeds)} on {len(directions)} rays; box R {low[0]:.2f}..{high[0]:.2f} Z {low[1]:.2f}..{high[1]:.2f}",
          flush=True)

    chunk = 10  # field periods per compiled call
    angles = jnp.arange(2 * chunk + 1) * half
    advance = jax.jit(lambda r, z: pf.trace_fieldline(field, None, r, z, angles, substeps=args.steps))

    def alive(r, z, plane):
        ok = np.isfinite(r) & np.isfinite(z)
        r, z = np.where(ok, r, 0.0), np.where(ok, z, 0.0)
        ok &= (r > low[0]) & (r < high[0]) & (z > low[1]) & (z < high[1])
        ok &= ~Polygon(edges[plane]).contains_points(np.stack((r, z), -1))
        phi = planes[plane]
        xyz = np.stack((r * np.cos(phi), r * np.sin(phi), z), -1)
        near = np.abs(np.angle(np.exp(1j * (np.arctan2(coil_points[:, 1], coil_points[:, 0]) - phi)))) < 0.2
        if near.any():
            ok &= np.min(np.linalg.norm(xyz[:, None] - coil_points[near][None], axis=-1), axis=1) > args.coil_gap
        return ok

    r, z = seeds[:, 0].copy(), seeds[:, 1].copy()
    living = np.ones(len(seeds), dtype=bool)
    life = np.zeros(len(seeds))                      # field periods each line lasted
    marks = {0: [], 1: []}                           # (R, Z, seed index) per plane
    clock = time.monotonic()
    for start in range(0, args.periods, chunk):
        path_r, path_z = (np.asarray(v) for v in advance(jnp.asarray(r), jnp.asarray(z)))
        for k in range(1, 2 * chunk + 1):
            plane = k % 2
            living &= alive(path_r[k], path_z[k], plane)
            marks[plane].append(np.stack((path_r[k][living], path_z[k][living], np.flatnonzero(living)), -1))
            life[living] = start + 0.5 * k
        r, z = np.where(living, path_r[-1], seeds[:, 0]), np.where(living, path_z[-1], seeds[:, 1])
        print(f"[trace] {start + chunk} periods, {int(living.sum())} of {len(seeds)} lines left, "
              f"{time.monotonic() - clock:.0f} s", flush=True)
        if not living.any():
            break
    marks = {k: np.concatenate(v) for k, v in marks.items()}

    manifolds, x_point = {}, None
    if args.xpoint:
        r0, z0, M, step = pf.return_map_fixed_point(field, None, np.asarray([args.xpoint[0]]), np.asarray([args.xpoint[1]]),
                                                    nfp=nfp, steps=8 * args.steps, iterations=10)
        M, x_point = np.asarray(M)[0], np.asarray([float(r0[0]), float(z0[0])])
        values, vectors = np.linalg.eig(M)
        print(f"[x-point] ({x_point[0]:.5f}, {x_point[1]:.5f}) Newton step {float(step[0]):.1e} Tr M {np.trace(M):.4f} "
              f"det {np.linalg.det(M):.6f} eigenvalues {values}", flush=True)
        if float(step[0]) < 1e-6 and abs(np.trace(M)) > 2:
            values, vectors = values.real, vectors.real
            for name, k, sign in (("unstable", int(np.argmax(np.abs(values))), 1.0),
                                  ("stable", int(np.argmin(np.abs(values))), -1.0)):
                growth = abs(values[k]) ** sign
                offsets = 1e-4 * growth ** np.linspace(0.0, 1.0, 400)
                start = x_point + np.concatenate((offsets, -offsets))[:, None] * vectors[:, k]
                ends = jnp.asarray([0.0, sign * 2.0 * half])
                step_map = jax.jit(lambda a, b, ends=ends: tuple(
                    v[-1] for v in pf.trace_fieldline(field, None, a, b, ends, substeps=4 * args.steps)))
                pieces, a, b = [start], start[:, 0], start[:, 1]
                for _ in range(int(np.ceil(np.log(0.5 / 1e-4) / np.log(growth)))):
                    a, b = (np.asarray(v) for v in step_map(jnp.asarray(a), jnp.asarray(b)))
                    ok = alive(a, b, 0)
                    pieces.append(np.where(ok[:, None], np.stack((a, b), -1), np.nan))
                    a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
                manifolds[name] = np.stack(pieces)       # (iterations + 1, 2 * 400, 2)

    np.savez(out / "poincare.npz", seeds=seeds, life=life, marks0=marks[0], marks1=marks[1],
             **{f"manifold_{k}": v for k, v in manifolds.items()}, **({} if x_point is None else {"x_point": x_point}))

    figure, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    lost = life < args.periods
    norm = LogNorm(vmin=1.0, vmax=max(args.periods, 2))
    for plane, axis in enumerate(axes):
        phi = planes[plane]
        for s_index in np.unique(np.round(np.linspace(0.0, 1.0, args.surfaces + 1)[1:] ** 0.5 * (ns - 1)).astype(int)):
            axis.plot(*section(phi, s_index).T, color="tab:blue", lw=1.6 if s_index == ns - 1 else 0.6)
        points = marks[plane]
        index = points[:, 2].astype(int)
        for mirror in (1.0, -1.0):  # stellarator symmetry: the mirrored marks are those of the lines traced backwards
            kept = ~lost[index]
            axis.plot(points[kept, 0], mirror * points[kept, 1], ",", color="0.25")
            mesh = axis.scatter(points[~kept, 0], mirror * points[~kept, 1], s=1.5, c=life[index][~kept], cmap="autumn_r",
                                norm=norm, linewidths=0)
        near = np.abs(np.angle(np.exp(1j * (np.arctan2(coil_points[:, 1], coil_points[:, 0]) - phi)))) < 0.03
        axis.plot(np.hypot(coil_points[near, 0], coil_points[near, 1]), coil_points[near, 2], "k.", ms=4)
        if plane == 0:
            for name, color in (("unstable", "tab:green"), ("stable", "tab:purple")):
                if name in manifolds:
                    for sign in (1.0, -1.0):  # the lower X-point's manifolds are the mirrored ones of the other kind
                        other = {"unstable": "stable", "stable": "unstable"}[name] if sign < 0 else name
                        for branch in np.split(manifolds[other], 2, axis=1):
                            axis.plot(branch[..., 0].ravel(), sign * branch[..., 1].ravel(), ".", ms=1.2, color=color,
                                      label=f"{name} manifold" if sign > 0 and branch is not None else None)
            if x_point is not None:
                axis.plot([x_point[0]] * 2, [x_point[1], -x_point[1]], "kx", ms=10, mew=2)
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                axis.legend(unique.values(), unique.keys(), loc="lower right", markerscale=8, fontsize=8)
        axis.set(xlabel="R [m]", ylabel="Z [m]", aspect="equal", xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                 title=f"phi = {'0' if plane == 0 else 'half field period'}")
    figure.colorbar(mesh, ax=axes, label="field periods before a line is lost (grey: kept all "
                    f"{args.periods})", shrink=0.7)
    figure.suptitle(args.title or str(args.run))
    figure.savefig(out / "poincare.png", dpi=170)
    print(f"[done] {int(lost.sum())} of {len(seeds)} lines lost; {out / 'poincare.png'}", flush=True)


if __name__ == "__main__":
    main()
