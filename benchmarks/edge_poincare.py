#!/usr/bin/env python
"""Poincare plot of a single-stage run: VMEC flux surfaces inside, traced field lines outside the plasma.

    python benchmarks/edge_poincare.py --run runs/qa --xpoint 0.98 0.73 --output runs/qa-poincare

``--run`` holds ``coils.json``, ``wout.nc`` and optionally ``pf.json`` (the layout of
``benchmarks/edge_fixed_points.py``).  Inside the boundary the flux surfaces are the WOUT's; outside, field lines of
the coils' plus the plasma's virtual-casing field (:class:`vmex.core.extender.VmecExtender`) are traced from seeds on
the plane ``phi = 0`` (:func:`vmex.core.periodic_fieldlines.trace_fieldline`) and marked on ``--planes`` equally
spaced planes of one field period.  A line ends where it leaves the box, enters the plasma or comes within ``--coil-gap`` of a coil;
it is coloured by the field periods it lasted.  With ``--xpoint`` the closed line through that point is refined by
shooting; the line's point and its unstable and stable manifolds are drawn on every plane.  Writes ``poincare.png`` and ``poincare.npz``;
``--replot`` draws the figure again from that file without tracing.
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
    parser.add_argument("--periods", type=int, default=150, help="field periods traced")
    parser.add_argument("--steps", type=int, default=64, help="Runge-Kutta steps per field period")
    parser.add_argument("--planes", type=int, default=4, help="planes per field period (even)")
    parser.add_argument("--seeds", type=int, default=40, help="seeds per ray")
    parser.add_argument("--reach", type=float, default=0.22, help="length of the seed rays outside the boundary [m]")
    parser.add_argument("--gap", type=float, default=0.03, help="smallest distance of a seed to the boundary [m]")
    parser.add_argument("--coil-gap", type=float, default=0.03, help="a line ends this close to a coil [m]")
    parser.add_argument("--wall", type=float, metavar="D",
                        help="a line ends on a wall D [m] off the plasma boundary (the driver's --edge-wall): its strike "
                             "points are drawn and the lengths to the wall printed")
    parser.add_argument("--margin", type=float, default=0.4, help="box around the boundary [m]")
    parser.add_argument("--vc-grid", type=int, default=64, help="virtual-casing grid per field period")
    parser.add_argument("--near-surface", default="direct", choices=("direct", "routed"),
                        help="plasma field: the direct quadrature (accurate a few grid spacings off the boundary) or "
                             "the extender's near-surface rule where a point is close (slower)")
    parser.add_argument("--surfaces", type=int, default=12, help="flux surfaces drawn")
    parser.add_argument("--title", default="")
    parser.add_argument("--replot", action="store_true",
                        help="draw the figure from the poincare.npz of an earlier call with the same options")
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
    period = 2.0 * np.pi / nfp
    biot_savart = BiotSavart(coils)
    coil_points = np.asarray(coils.gamma).reshape(-1, 3)
    if (args.run / "pf.json").exists():  # the driver's circular coil pairs
        from vmex.core.coil_design import SummedField, coil_pairs_from_dict

        pairs = coil_pairs_from_dict(json.loads((args.run / "pf.json").read_text()))
        biot_savart = SummedField(biot_savart, BiotSavart(pairs))
        coil_points = np.concatenate((coil_points, np.asarray(pairs.gamma).reshape(-1, 3)))
    surface = vc.surface_field_data_from_wout(wout, nphi=args.vc_grid, ntheta=args.vc_grid)
    total = VmecExtender.from_surface_data(surface, external_field=lambda xyz: jax.vmap(biot_savart.B)(xyz),
                                           accuracy_check="off",
                                           **({"near_surface": "direct"} if args.near_surface == "direct" else {}))
    field = lambda p, xyz: total.B(xyz)  # noqa: E731

    def section(phi, s_index):  # the WOUT's surface s_index on the plane phi
        theta = np.linspace(0.0, 2.0 * np.pi, 361)[:, None]
        angle = np.asarray(wout.xm)[None] * theta - np.asarray(wout.xn)[None] * phi
        return np.stack((np.cos(angle) @ np.asarray(wout.rmnc)[s_index], np.sin(angle) @ np.asarray(wout.zmns)[s_index]), -1)

    ns = np.asarray(wout.rmnc).shape[0]
    planes = np.arange(args.planes) * period / args.planes
    substeps = max(4, args.steps // args.planes)
    edges = [section(phi, ns - 1) for phi in planes]
    axis0 = section(0.0, 0)[0]
    low = np.min([e.min(axis=0) for e in edges], axis=0) - args.margin
    high = np.max([e.max(axis=0) for e in edges], axis=0) + args.margin
    coil_angle = np.arctan2(coil_points[:, 1], coil_points[:, 0])

    def coils_near(plane, width):  # coil points within ``width`` [rad] of a plane, on any field period
        return coil_points[np.abs(np.angle(np.exp(1j * nfp * (coil_angle - planes[plane])))) < nfp * width]

    # seeds on phi = 0: rays from the magnetic axis through the boundary (the lower half is the mirror image)
    def ray(direction):
        direction = np.asarray(direction) / np.linalg.norm(direction)
        t = np.linspace(0.0, 3.0, 3001)
        inside = Polygon(edges[0]).contains_points(axis0 + t[:, None] * direction)
        t_edge = t[np.argmin(inside)]
        return axis0 + (t_edge + np.linspace(args.gap, args.reach, args.seeds))[:, None] * direction

    directions = [(np.cos(a), np.sin(a)) for a in np.linspace(0.0, np.pi, 5)]
    if args.xpoint:
        directions.append(np.asarray(args.xpoint) - axis0)
    seeds = np.concatenate([ray(d) for d in directions])
    print(f"[seeds] {len(seeds)} on {len(directions)} rays; box R {low[0]:.2f}..{high[0]:.2f} Z {low[1]:.2f}..{high[1]:.2f}",
          flush=True)

    chunk = 5  # field periods per compiled call
    angles = jnp.arange(chunk * args.planes + 1) * period / args.planes
    advance = jax.jit(lambda r, z: pf.trace_fieldline(field, None, r, z, angles, substeps=substeps))

    def alive(r, z, plane):
        ok = np.isfinite(r) & np.isfinite(z)
        r, z = np.where(ok, r, 0.0), np.where(ok, z, 0.0)
        ok &= (r > low[0]) & (r < high[0]) & (z > low[1]) & (z < high[1])
        ok &= ~Polygon(edges[plane]).contains_points(np.stack((r, z), -1))
        near = coils_near(plane, 0.2)  # rotated to the plane's own field period: only the distance matters
        if len(near):
            near_r, near_phi = np.hypot(near[:, 0], near[:, 1]), np.arctan2(near[:, 1], near[:, 0])
            d_phi = np.angle(np.exp(1j * nfp * (near_phi - planes[plane]))) / nfp
            near = np.stack((near_r * np.cos(d_phi), near_r * np.sin(d_phi), near[:, 2]), -1)
            xyz = np.stack((r, np.zeros_like(r), z), -1)
            ok &= np.min(np.linalg.norm(xyz[:, None] - near[None], axis=-1), axis=1) > args.coil_gap
        if args.wall is not None:
            ok &= to_plasma(r, z, plane) < args.wall
        return ok

    def to_plasma(r, z, plane):  # distance [m] to the plasma boundary
        xyz = np.stack((r * np.cos(planes[plane]), r * np.sin(planes[plane]), z), -1)
        return np.asarray(pf.surface_distances(jnp.asarray(xyz), surface.gamma, nfp))

    if not args.replot:
        r, z = seeds[:, 0].copy(), seeds[:, 1].copy()
        living = np.ones(len(seeds), dtype=bool)
        life = np.zeros(len(seeds))                      # field periods each line lasted
        marks = {k: [] for k in range(args.planes)}      # (R, Z, seed index) per plane
        strikes = {k: [] for k in range(args.planes)}    # with --wall: (R, Z, seed index) of the first point beyond it
        clock = time.monotonic()
        for start in range(0, args.periods, chunk):
            # only the lines still followed are traced, in batches padded to a power of two (few compiled sizes)
            active = np.flatnonzero(living)
            size = max(8, 1 << int(np.ceil(np.log2(len(active)))))
            padded = np.r_[active, np.full(size - len(active), active[0])]
            path_r, path_z = (np.tile(seeds[:, i], (chunk * args.planes + 1, 1)) for i in (0, 1))
            traced = [np.asarray(v) for v in advance(jnp.asarray(r[padded]), jnp.asarray(z[padded]))]
            path_r[:, active], path_z[:, active] = traced[0][:, :len(active)], traced[1][:, :len(active)]
            for k in range(1, chunk * args.planes + 1):
                plane = k % args.planes
                before = living.copy()
                living &= alive(path_r[k], path_z[k], plane)
                if args.wall is not None:
                    ended = before & ~living & np.isfinite(path_r[k]) & np.isfinite(path_z[k])
                    ended[ended] = to_plasma(path_r[k][ended], path_z[k][ended], plane) >= args.wall
                    strikes[plane].append(np.stack((path_r[k][ended], path_z[k][ended], np.flatnonzero(ended)), -1))
                marks[plane].append(np.stack((path_r[k][living], path_z[k][living], np.flatnonzero(living)), -1))
                life[living] = start + k / args.planes
            r, z = np.where(living, path_r[-1], seeds[:, 0]), np.where(living, path_z[-1], seeds[:, 1])
            print(f"[trace] {start + chunk} periods, {int(living.sum())} of {len(seeds)} lines left, "
                  f"{time.monotonic() - clock:.0f} s", flush=True)
            if not living.any():
                break
        marks = {k: np.concatenate(v) for k, v in marks.items()}
        strikes = {k: np.concatenate(v) if v else np.zeros((0, 3)) for k, v in strikes.items()}
        if args.wall is not None:
            # length along the torus to the wall, one way (the lines are followed in +phi only), by starting distance
            start = to_plasma(seeds[:, 0], seeds[:, 1], 0)
            hit = np.zeros(len(seeds), dtype=bool)
            hit[np.concatenate([v[:, 2] for v in strikes.values()]).astype(int)] = True
            length = life * period * float(np.asarray(wout.rmnc)[0, 0])
            print(f"[wall] {args.wall:.3f} m off the boundary: {int(hit.sum())} of {len(seeds)} lines end on it, "
                  f"{int((~hit & (life < args.periods)).sum())} elsewhere, {int((life >= args.periods).sum())} kept", flush=True)
            bins = np.linspace(args.gap, min(args.reach, args.wall), 5)
            for a, b in zip(bins[:-1], bins[1:]):
                inside = hit & (start >= a) & (start < b)
                if inside.any():
                    print(f"[wall] seeds {100 * a:.1f}-{100 * b:.1f} cm off the boundary: {int(inside.sum())} lines, "
                          f"length to the wall median {np.median(length[inside]):.1f} m, longest {length[inside].max():.1f} m "
                          f"({np.median(life[inside]):.1f} field periods)", flush=True)

        # The closed line through --xpoint and its manifolds.  On plane k: the unstable manifold is the one on phi = 0 traced
        # forwards to the plane, the stable one traced backwards to it.
        manifolds, x_line = {}, None
        forward = jnp.asarray(planes)
        to_planes = {1.0: jax.jit(lambda a, b: pf.trace_fieldline(field, None, a, b, forward, substeps=4 * substeps)),
                     -1.0: jax.jit(lambda a, b: pf.trace_fieldline(field, None, a, b, -forward, substeps=4 * substeps))}
        if args.xpoint:
            r0, z0, M, step = pf.return_map_fixed_point(field, None, np.asarray([args.xpoint[0]]), np.asarray([args.xpoint[1]]),
                                                        nfp=nfp, steps=8 * args.steps, iterations=10)
            M = np.asarray(M)[0]
            x_line = np.stack([np.asarray(v)[:, 0] for v in to_planes[1.0](r0, z0)], -1)      # (planes, 2)
            values, vectors = np.linalg.eig(M)
            print(f"[x-point] ({x_line[0, 0]:.5f}, {x_line[0, 1]:.5f}) Newton step {float(step[0]):.1e} Tr M {np.trace(M):.4f} "
                  f"det {np.linalg.det(M):.6f} eigenvalues {values}; on the planes: "
                  + ", ".join(f"({a:.4f}, {b:.4f})" for a, b in x_line), flush=True)
            if float(step[0]) < 1e-6 and abs(np.trace(M)) > 2:
                values, vectors = values.real, vectors.real
                for name, k, sign in (("unstable", int(np.argmax(np.abs(values))), 1.0),
                                      ("stable", int(np.argmin(np.abs(values))), -1.0)):
                    growth = abs(values[k]) ** sign
                    offsets = 1e-4 * growth ** np.linspace(0.0, 1.0, 400)
                    start = x_line[0] + np.concatenate((offsets, -offsets))[:, None] * vectors[:, k]
                    ends = jnp.asarray([0.0, sign * period])
                    step_map = jax.jit(lambda a, b, ends=ends: tuple(
                        v[-1] for v in pf.trace_fieldline(field, None, a, b, ends, substeps=4 * args.steps)))
                    pieces, a, b = [start], start[:, 0], start[:, 1]
                    for _ in range(int(np.ceil(np.log(0.5 / 1e-4) / np.log(growth)))):
                        a, b = (np.asarray(v) for v in step_map(jnp.asarray(a), jnp.asarray(b)))
                        ok = alive(a, b, 0)
                        pieces.append(np.where(ok[:, None], np.stack((a, b), -1), np.nan))
                        a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
                    pieces = np.stack(pieces)                 # (iterations + 1, 2 * 400, 2) on phi = 0
                    flat = np.nan_to_num(pieces.reshape(-1, 2), nan=float(x_line[0, 0]))
                    moved = np.stack([np.asarray(v) for v in to_planes[sign](jnp.asarray(flat[:, 0]), jnp.asarray(flat[:, 1]))], -1)
                    moved = np.where(np.isfinite(pieces.reshape(-1, 2)).all(axis=-1)[None, :, None], moved, np.nan)
                    order = np.arange(args.planes) if sign > 0 else (-np.arange(args.planes)) % args.planes
                    on_planes = np.empty((args.planes,) + pieces.shape)
                    on_planes[order] = moved.reshape((args.planes,) + pieces.shape)
                    manifolds[name] = on_planes               # (planes, iterations + 1, 2 * 400, 2)

        np.savez(out / "poincare.npz", seeds=seeds, life=life, planes=planes, **{f"marks{k}": v for k, v in marks.items()},
                 **{f"strikes{k}": v for k, v in strikes.items()},
                 **{f"manifold_{k}": v for k, v in manifolds.items()}, **({} if x_line is None else {"x_line": x_line}))
    else:
        saved = np.load(out / "poincare.npz")
        life, x_line = saved["life"], saved["x_line"] if "x_line" in saved else None
        marks, strikes = ({k: saved[f"{name}{k}"] for k in range(args.planes)} for name in ("marks", "strikes"))
        manifolds = {k: saved[f"manifold_{k}"] for k in ("unstable", "stable") if f"manifold_{k}" in saved}

    columns = int(np.ceil(args.planes / 2)) if args.planes > 2 else args.planes
    rows = int(np.ceil(args.planes / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns + 1.5, 5.6 * rows), constrained_layout=True,
                                squeeze=False)
    lost = life < args.periods
    norm = LogNorm(vmin=1.0, vmax=max(args.periods, 2))
    other = {"unstable": "stable", "stable": "unstable"}
    for plane, axis in enumerate(axes.ravel()[:args.planes]):
        mirror = (-plane) % args.planes  # stellarator symmetry: (R, Z) at phi is a point of the reversed line at -phi
        for s_index in np.unique(np.round(np.linspace(0.0, 1.0, args.surfaces + 1)[1:] ** 0.5 * (ns - 1)).astype(int)):
            axis.plot(*section(planes[plane], s_index).T, color="tab:blue", lw=1.6 if s_index == ns - 1 else 0.6)
        for points, flip in ((marks[plane], 1.0), (marks[mirror], -1.0)):
            index = points[:, 2].astype(int)
            kept = ~lost[index]
            axis.plot(points[kept, 0], flip * points[kept, 1], ",", color="0.25")
            mesh = axis.scatter(points[~kept, 0], flip * points[~kept, 1], s=1.5, c=life[index][~kept], cmap="autumn_r",
                                norm=norm, linewidths=0)
        near = coils_near(plane, 0.03)
        axis.plot(np.hypot(near[:, 0], near[:, 1]), near[:, 2], "k.", ms=4)
        if args.wall is not None:
            grid_r, grid_z = np.meshgrid(np.linspace(low[0], high[0], 90), np.linspace(low[1], high[1], 110))
            distance = np.array(to_plasma(grid_r.ravel(), grid_z.ravel(), plane)).reshape(grid_r.shape)
            distance[Polygon(edges[plane]).contains_points(np.stack((grid_r.ravel(), grid_z.ravel()), -1)).reshape(grid_r.shape)] = 0.0
            axis.contour(grid_r, grid_z, distance, levels=[args.wall], colors="0.4", linestyles="--", linewidths=1.0)
            for points, flip in ((strikes[plane], 1.0), (strikes[mirror], -1.0)):
                axis.plot(points[:, 0], flip * points[:, 1], "*", color="tab:red", ms=5, mew=0,
                          label="ends on the wall (dashed)" if flip > 0 else None)
        for name, color in (("unstable", "tab:green"), ("stable", "tab:purple")):
            if name not in manifolds:
                continue
            # the lower line's manifolds are the mirrored ones of the other kind
            for source, flip in ((manifolds[name][plane], 1.0), (manifolds[other[name]][mirror], -1.0)):
                for branch in np.split(source, 2, axis=1):  # drawn while it is a resolved curve
                    curve = branch.reshape(-1, 2)
                    jump = np.linalg.norm(np.diff(curve, axis=0), axis=1)
                    end = int(np.argmax(~(jump < 0.03))) if np.any(~(jump < 0.03)) else len(curve)
                    axis.plot(curve[:end, 0], flip * curve[:end, 1], "-", lw=1.4, color=color,
                              label=f"{name} manifold" if flip > 0 else None)
        if x_line is not None:
            axis.plot([x_line[plane, 0], x_line[mirror, 0]], [x_line[plane, 1], -x_line[mirror, 1]], "kx", ms=11, mew=2.5,
                      label="closed line (X-point)", zorder=5)
        if plane == 0:
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                axis.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8)
        fraction = f"{plane}/{args.planes} field period" if plane else "0"
        axis.set(xlabel="R [m]", ylabel="Z [m]", aspect="equal", xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                 title=f"phi = {fraction}")
    for axis in axes.ravel()[args.planes:]:
        axis.set_visible(False)
    figure.colorbar(mesh, ax=axes, label="field periods before a line is lost (grey: kept all "
                    f"{args.periods})", shrink=0.7)
    figure.suptitle(args.title or str(args.run))
    figure.savefig(out / "poincare.png", dpi=170)
    print(f"[done] {int(lost.sum())} of {len(seeds)} lines lost; {out / 'poincare.png'}", flush=True)


if __name__ == "__main__":
    main()
