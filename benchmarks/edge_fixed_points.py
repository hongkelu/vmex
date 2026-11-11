#!/usr/bin/env python
"""Closed field lines outside the plasma of a single-stage run, in the coil and in the total field.

    python benchmarks/edge_fixed_points.py --run runs/qa --output runs/qa-edge

``--run`` holds ``coils.json`` and ``wout.nc`` (the free-boundary solve of those coils).  On the plane
``phi = 0`` a grid outside the boundary is mapped once around ``--periods`` field periods
(:func:`vmex.core.periodic_fieldlines.return_map`) in the coils' field and in the coils' plus the plasma's
virtual-casing field; the grid cells where both components of the displacement change sign are refined to closed lines
(:func:`~vmex.core.periodic_fieldlines.periodic_fieldline`) with their tangent maps.  Writes
``fixed_points.json`` (position, ``Tr M``, distances to the boundary and to the coils) and
``return_map.png``.
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
    parser.add_argument("--coils", type=Path, help="coil file (default: <run>/coils.json)")
    parser.add_argument("--wout", type=Path, help="WOUT (default: <run>/wout.nc)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=1)
    parser.add_argument("--grid", type=int, default=48, help="grid points per direction")
    parser.add_argument("--margin", type=float, default=0.25, help="box around the boundary's cross-section [m]")
    parser.add_argument("--gap", type=float, default=0.06, help="smallest distance of a start to the boundary [m]")
    parser.add_argument("--coil-gap", type=float, default=0.04, help="smallest distance of a start to a coil [m]")
    parser.add_argument("--steps", type=int, default=96, help="Runge-Kutta steps per field period of the grid's map")
    parser.add_argument("--nodes", type=int, default=65, help="collocation nodes per field period (odd)")
    parser.add_argument("--vc-grid", type=int, default=64, help="virtual-casing grid per field period")
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument("--pf", type=float, nargs=3, action="append", metavar=("R", "Z", "I"), default=[],
                        help="add a pair of circular coils of radius R [m] at +-Z [m] with current I [A] (repeatable); "
                             "the plasma field stays the WOUT's")
    parser.add_argument("--candidates", type=int, default=16, help="grid cells refined per field")
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
    from matplotlib.path import Path as Polygon
    from essos.coils import Coils
    from essos.fields import BiotSavart

    from vmex.core import periodic_fieldlines as pf
    from vmex.core import virtual_casing as vc
    from vmex.core.extender import VmecExtender
    from vmex.core.wout import read_wout

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wout = read_wout(str(args.wout or args.run / "wout.nc"))
    coils = Coils.from_json(str(args.coils or args.run / "coils.json"))
    nfp = int(wout.nfp)
    biot_savart = BiotSavart(coils)
    if args.pf:
        from vmex.core.coil_design import SummedField, circular_coil_pairs

        biot_savart = SummedField(biot_savart, BiotSavart(circular_coil_pairs(*np.asarray(args.pf).T)))
    coil_field = lambda xyz: jax.vmap(biot_savart.B)(xyz)  # noqa: E731
    surface = vc.surface_field_data_from_wout(wout, nphi=args.vc_grid, ntheta=args.vc_grid)
    exterior = VmecExtender.from_surface_data(surface, external_field=coil_field, digits=args.digits,
                                              accuracy_check="off")
    # The grid's map uses the direct quadrature (accurate a few source spacings off the boundary, hence --gap);
    # the fixed points are then solved with the near-surface rule where the direct one misses the digits.
    direct = VmecExtender.from_surface_data(surface, external_field=coil_field, digits=args.digits,
                                            accuracy_check="off", near_surface="direct")
    fields = {"coils": (lambda p, xyz: coil_field(xyz),) * 2,
              "total": (lambda p, xyz: direct.B(xyz), lambda p, xyz: exterior.B(xyz))}

    gamma = np.asarray(surface.gamma)                       # (3, nphi, ntheta), one field period
    section = np.stack((np.hypot(gamma[0, 0], gamma[1, 0]), gamma[2, 0]), axis=-1)   # phi = 0
    coil_points = np.asarray(coils.gamma).reshape(-1, 3)
    low, high = section.min(axis=0) - args.margin, section.max(axis=0) + args.margin
    R, Z = np.meshgrid(np.linspace(low[0], high[0], args.grid), np.linspace(low[1], high[1], args.grid), indexing="ij")
    xyz = np.stack((R.ravel(), np.zeros(R.size), Z.ravel()), axis=-1)
    to_boundary = np.min(np.linalg.norm(xyz[:, None, [0, 2]] - section[None], axis=-1), axis=1)
    to_coils = np.min(np.linalg.norm(xyz[:, None] - coil_points[None], axis=-1), axis=1)
    start = (~Polygon(section).contains_points(xyz[:, [0, 2]])) & (to_boundary > args.gap) & (to_coils > args.coil_gap)
    print(f"[grid] {int(start.sum())} of {R.size} starts outside the boundary, R {low[0]:.3f}..{high[0]:.3f} m, "
          f"Z {low[1]:.3f}..{high[1]:.3f} m", flush=True)

    def distances(line):
        points = np.asarray(line.xyz)
        return (float(pf.surface_distance(jnp.asarray(points), surface.gamma, nfp)),
                float(np.min(np.linalg.norm(points[:, None] - coil_points[None], axis=-1))))

    report, figure = {}, plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, (name, (survey, field)) in zip(figure[1], fields.items()):
        clock = time.monotonic()
        image = jax.jit(lambda r, z, field=survey: pf.return_map(field, None, r, z, nfp=nfp, periods=args.periods,
                                                                 steps=args.steps))
        r1, z1 = (np.asarray(v) for v in image(jnp.asarray(R.ravel()[start]), jnp.asarray(Z.ravel()[start])))
        moved = np.full((2, R.size), np.nan)
        moved[:, start] = r1 - R.ravel()[start], z1 - Z.ravel()[start]
        moved = moved.reshape((2,) + R.shape)
        shift = np.hypot(*moved)
        # a fixed point lies in a grid cell where both components of P(x) - x change sign
        corners = np.stack([moved[:, i:moved.shape[1] - 1 + i, j:moved.shape[2] - 1 + j] for i in (0, 1) for j in (0, 1)])
        crossing = np.all(np.isfinite(corners), axis=(0, 1)) & np.all(
            (corners.min(axis=0) < 0) & (corners.max(axis=0) > 0), axis=0)
        cells = np.argwhere(crossing)
        cells = cells[np.argsort(shift[tuple(cells.T)])][:args.candidates]
        minima = np.argwhere(shift == np.nanmin(shift))
        starts = (R[tuple(cells.T)] + 0.5 * (R[1, 0] - R[0, 0]), Z[tuple(cells.T)] + 0.5 * (Z[0, 1] - Z[0, 0]))
        print(f"[{name}] grid mapped in {time.monotonic() - clock:.0f} s; {len(cells)} cells with a sign change in both "
              f"components; smallest displacement {np.nanmin(shift):.1e} m at "
              f"({R[tuple(minima[0])]:.3f}, {Z[tuple(minima[0])]:.3f})", flush=True)
        np.savez(out / f"return_map_{name}.npz", R=R, Z=Z, shift=shift, section=section)
        # Each candidate: its field line over the nodes, made periodic by removing the mismatch linearly, then Newton
        # on the collocated line.  A solve that stalls, leaves the box or ends in the plasma is no fixed point.
        points, n = [], args.nodes * args.periods
        nodes = pf.fieldline_nodes(n, nfp, args.periods)
        if len(cells):
            traced = jax.jit(lambda r, z, field=survey: pf.trace_fieldline(
                field, None, r, z, jnp.asarray(np.r_[nodes, nodes[0] + 2 * np.pi * args.periods / nfp]), substeps=4))
            paths = [np.asarray(v) for v in traced(jnp.asarray(starts[0]), jnp.asarray(starts[1]))]
            ramp = (np.arange(n) / n)[:, None]
            guesses = [v[:-1] - ramp * (v[-1] - v[0]) for v in paths]
            solve = jax.jit(lambda r, z, field=field: pf.periodic_fieldline(field, None, r, z, nfp=nfp,
                                                                            periods=args.periods, max_iterations=30))
            tangent_of = jax.jit(lambda line, field=field: pf.tangent_map(field, None, line))
        for k in range(len(cells)):
            if not np.all(np.isfinite(guesses[0][:, k]) & np.isfinite(guesses[1][:, k])):
                continue
            line = solve(jnp.asarray(guesses[0][:, k]), jnp.asarray(guesses[1][:, k]))
            r, z = float(line.R[0]), float(line.Z[0])
            if not (float(line.step) < 1e-9 and low[0] <= r <= high[0] and low[1] <= z <= high[1]):
                continue
            if Polygon(section).contains_point((r, z)) or any(np.hypot(r - p["R"], z - p["Z"]) < 1e-6 for p in points):
                continue
            tangent = np.asarray(tangent_of(line))
            trace = float(np.trace(tangent))
            boundary_distance, coil_distance = distances(line)
            points.append(dict(
                R=r, Z=z, trace=trace, determinant=float(np.linalg.det(tangent)),
                kind="X" if abs(trace) > 2 else "O", residue=(2.0 - trace) / 4.0,
                collocation_step=float(line.step), collocation_residual=float(line.residual),
                boundary_distance_m=boundary_distance, coil_distance_m=coil_distance,
                R_nodes=np.asarray(line.R).tolist(), Z_nodes=np.asarray(line.Z).tolist()))
            print(f"[{name}] {points[-1]['kind']}-point R={r:.5f} Z={z:.5f} Tr M={trace:.6f} "
                  f"det={points[-1]['determinant']:.8f} boundary {boundary_distance:.4f} m coils {coil_distance:.4f} m",
                  flush=True)
        report[name] = points
        print(f"[{name}] {len(points)} fixed points; {time.monotonic() - clock:.0f} s", flush=True)
        mesh = axis.pcolormesh(R, Z, np.log10(shift), shading="nearest", cmap="viridis")
        figure[0].colorbar(mesh, ax=axis, label="log10 |P(x) - x| [m]")
        axis.plot(*np.r_[section, section[:1]].T, "w-", lw=1.5)
        near = np.abs(np.arctan2(coil_points[:, 1], coil_points[:, 0])) < 0.03
        axis.plot(np.hypot(coil_points[near, 0], coil_points[near, 1]), coil_points[near, 2], "r.", ms=3)
        for p in points:
            axis.plot(p["R"], p["Z"], "rx" if p["kind"] == "X" else "wo", ms=9, mew=2, mfc="none")
        axis.set(title=f"{name} field, {args.periods} field period(s), phi = 0", xlabel="R [m]", ylabel="Z [m]",
                 aspect="equal")
    figure[0].savefig(out / "return_map.png", dpi=150)
    (out / "fixed_points.json").write_text(json.dumps(dict(
        run=str(args.run), pf=args.pf, nfp=nfp, periods=args.periods, grid=args.grid, steps=args.steps, **report), indent=1))
    print(f"[done] {', '.join(f'{len(v)} in the {k} field' for k, v in report.items())}; {out}", flush=True)


if __name__ == "__main__":
    main()
