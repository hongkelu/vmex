#!/usr/bin/env python
"""Closed field lines outside the plasma of a single-stage run, in the coil and in the total field.

    python benchmarks/edge_fixed_points.py --run runs/qa --output runs/qa-edge

``--run`` holds ``coils.json`` and ``wout.nc`` (the free-boundary solve of those coils).  On the plane
``phi = 0`` a grid outside the boundary is mapped once around ``--periods`` field periods
(:func:`vmex.core.periodic_fieldlines.return_map`) in the coils' field and in the coils' plus the plasma's
virtual-casing field; the grid minima of the displacement are refined to fixed points
(:func:`~vmex.core.periodic_fieldlines.return_map_fixed_point`) and each is solved again as a collocated line
(:func:`~vmex.core.periodic_fieldlines.periodic_fieldline`) with its tangent map.  Writes
``fixed_points.json`` (position, ``Tr M``, distances to the boundary and to the coils) and
``return_map.png``.
"""

import argparse
import json
import os
from pathlib import Path

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
    parser.add_argument("--gap", type=float, default=0.01, help="smallest distance of a start to the boundary [m]")
    parser.add_argument("--coil-gap", type=float, default=0.04, help="smallest distance of a start to a coil [m]")
    parser.add_argument("--steps", type=int, default=96, help="Runge-Kutta steps per field period of the grid's map")
    parser.add_argument("--newton-steps", type=int, default=384, help="the same, of the fixed-point solves")
    parser.add_argument("--nodes", type=int, default=65, help="collocation nodes per field period (odd)")
    parser.add_argument("--vc-grid", type=int, default=64, help="virtual-casing grid per field period")
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=24, help="grid minima refined per field")
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
    coil_field = lambda xyz: jax.vmap(biot_savart.B)(xyz)  # noqa: E731
    surface = vc.surface_field_data_from_wout(wout, nphi=args.vc_grid, ntheta=args.vc_grid)
    exterior = VmecExtender.from_surface_data(surface, external_field=coil_field, digits=args.digits,
                                              accuracy_check="off")
    fields = {"coils": lambda p, xyz: coil_field(xyz), "total": lambda p, xyz: exterior.B(xyz)}

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
    for axis, (name, field) in zip(figure[1], fields.items()):
        image = jax.jit(lambda r, z, field=field: pf.return_map(field, None, r, z, nfp=nfp, periods=args.periods,
                                                                steps=args.steps))
        r1, z1 = (np.asarray(v) for v in image(jnp.asarray(R.ravel()[start]), jnp.asarray(Z.ravel()[start])))
        shift = np.full(R.size, np.nan)
        shift[start] = np.hypot(r1 - R.ravel()[start], z1 - Z.ravel()[start])
        shift = shift.reshape(R.shape)
        padded = np.pad(np.where(np.isfinite(shift), shift, np.inf), 1, constant_values=np.inf)
        neighbours = np.min([padded[1 + i:padded.shape[0] - 1 + i, 1 + j:padded.shape[1] - 1 + j]
                             for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], axis=0)
        minima = np.argwhere(np.isfinite(shift) & (np.where(np.isfinite(shift), shift, np.inf) <= neighbours))
        minima = minima[np.argsort(shift[tuple(minima.T)])][:args.candidates]
        points = []
        found = pf.return_map_fixed_point(field, None, R[tuple(minima.T)], Z[tuple(minima.T)], nfp=nfp,
                                          periods=args.periods, steps=args.newton_steps)
        for r, z, M, step in zip(*(np.asarray(v) for v in found)):
            # converged, resolved (the map preserves flux: det M = 1), in the box, outside the plasma, and new
            if not (step < 1e-9 and abs(np.linalg.det(M) - 1.0) < 1e-3 and low[0] <= r <= high[0] and low[1] <= z <= high[1]):
                continue
            if Polygon(section).contains_point((r, z)) or any(np.hypot(r - p["R"], z - p["Z"]) < 1e-6 for p in points):
                continue
            r, z = float(r), float(z)
            nodes = pf.fieldline_nodes(args.nodes * args.periods, nfp, args.periods)
            guess = pf.trace_fieldline(field, None, r, z, nodes, substeps=8)
            line = pf.periodic_fieldline(field, None, *guess, nfp=nfp, periods=args.periods)
            tangent = np.asarray(pf.tangent_map(field, None, line))
            trace = float(np.trace(tangent))
            boundary_distance, coil_distance = distances(line)
            points.append(dict(
                R=r, Z=z, trace_shooting=float(np.trace(np.asarray(M))), trace=trace, determinant=float(np.linalg.det(tangent)),
                kind="X" if abs(trace) > 2 else "O", residue=(2.0 - trace) / 4.0,
                collocation_step=float(line.step), collocation_residual=float(line.residual),
                boundary_distance_m=boundary_distance, coil_distance_m=coil_distance,
                R_nodes=np.asarray(line.R).tolist(), Z_nodes=np.asarray(line.Z).tolist()))
            print(f"[{name}] {points[-1]['kind']}-point R={r:.5f} Z={z:.5f} Tr M={trace:.6f} "
                  f"(shooting {points[-1]['trace_shooting']:.6f}) det={points[-1]['determinant']:.6f} "
                  f"boundary {boundary_distance:.4f} m coils {coil_distance:.4f} m", flush=True)
        report[name] = points
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
        run=str(args.run), nfp=nfp, periods=args.periods, grid=args.grid, steps=args.steps, **report), indent=1))
    print(f"[done] {', '.join(f'{len(v)} in the {k} field' for k, v in report.items())}; {out}", flush=True)


if __name__ == "__main__":
    main()
