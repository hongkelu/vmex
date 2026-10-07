#!/usr/bin/env python
"""Stage-two coil fit that produced the committed ``examples/data/ESSOS_coils_<case>.json`` files.

Starting from ``P.N_COILS`` circular coils of radius ``--radius`` per half
period, centred on R = ``P.RADIUS_TARGET``, L-BFGS-B minimizes

    0.5 (rms B.n/|B| / 1e-3)^2 + 0.5 WEIGHT sum_j min(c_j, 0)^2

on the case's fixed-boundary seed (``_common.seed_input``, no equilibrium
solve), with c_j the coil rows of ``_coil_constraints.py`` and the plasma
clearance. The currents are then scaled so the coils' toroidal flux through
the seed's phi = 0 cross-section is PHIEDGE. The optimization scripts rescale
them again, to an edge R B_phi of B0 R0 (``_common.scale_coil_currents``).

    COIL_CASE=qa4-beta python fit_coils.py --output ESSOS_coils_qa4_beta.json

The winding radii were 0.55 m for ``qa4-beta*``, 0.46 m for ``qi6-beta*`` and
0.42-0.5 m for ``qa3``, ``qh`` and ``qi`` (``--radius``), with 3000 iterations
and weight 1 (100 for ``qa4-beta-tok``, whose coils otherwise exceed the length limit). The fit is fast on a GPU and slow on a loaded CPU:
the coil-coil distance over all symmetry copies dominates.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ["JAX_ENABLE_X64"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402
from _common import normal_field_rms, seed_input  # noqa: E402

RADIUS = {"qa4-beta": 0.55, "qa4-beta-tok": 0.55, "qi6-beta": 0.46, "qi6-beta-tok": 0.46}  # m


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radius", type=float, default=RADIUS.get(P.CASE),
                        help=f"winding radius of the circular start [m] (default for COIL_CASE={P.CASE}: "
                             f"{RADIUS.get(P.CASE)})")
    parser.add_argument("--maxiter", type=int, default=3000, help="L-BFGS-B iterations")
    parser.add_argument("--weight", type=float, default=1.0, help="penalty weight of the coil rows")
    parser.add_argument("--output", type=Path, default=Path(f"coils.{P.CASE}.json"))
    args = parser.parse_args(argv)
    if args.radius is None:
        parser.error(f"COIL_CASE={P.CASE} has no default --radius")
    return args


def main(argv=None):
    args = parse_args(argv)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import _coil_constraints as coil_limits

    inp = seed_input()
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=coil_limits.SURFACE_GRID[0], ntheta=coil_limits.SURFACE_GRID[1])
    curves = CreateEquallySpacedCurves(P.N_COILS, P.COIL_ORDER, P.RADIUS_TARGET, args.radius,
                                       n_segments=P.N_SEGMENTS, nfp=inp.nfp, stellsym=True)
    coils0 = Coils(curves, np.full(P.N_COILS, 1.0e5))
    x0 = jnp.asarray(coils0.curves.dofs).ravel()

    def coils_from_u(u):
        return coils0.with_dofs(jnp.concatenate((x0 + P.COIL_STEP * u, coils0.dofs_currents)))

    def rows(coils):
        return jnp.concatenate([coil_limits.coil_inequalities(coils), jnp.atleast_1d(
            (coil_limits.surface_distance(coils, surface) - P.COIL_SURFACE_DISTANCE_LIMIT - P.DISTANCE_MARGIN)
            / P.COIL_SURFACE_DISTANCE_LIMIT)])

    def objective(u):
        c = coils_from_u(u)
        return 0.5 * (normal_field_rms(c, surface) / 1e-3)**2 + 0.5 * args.weight * jnp.sum(jnp.minimum(rows(c), 0.0)**2)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    print(f"circular: B.n rms {float(normal_field_rms(coils0, surface)):.3e}, "
          f"min row {float(jnp.min(rows(coils0))):.3f}", flush=True)
    fit = minimize(lambda u: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))), np.zeros(x0.size), jac=True,
                   method="L-BFGS-B", options=dict(maxiter=args.maxiter, maxcor=30, ftol=1e-15, gtol=1e-10))
    coils = coils_from_u(jnp.asarray(fit.x))

    # Toroidal flux through the seed's phi = 0 cross-section (as single_stage_optimization.py).
    rho, rw = np.polynomial.legendre.leggauss(24)
    rho, rw = 0.5 * (rho + 1), 0.5 * rw
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    m = np.arange(inp.rbc.shape[1])
    rbc0, zbs0 = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
    cm, sm = np.cos(np.outer(theta, m)), np.sin(np.outer(theta, m))
    re, ze, dre, dze = cm @ rbc0, sm @ zbs0, -sm @ (m * rbc0), cm @ (m * zbs0)
    r, z = rbc0[0] + rho[:, None] * (re - rbc0[0]), rho[:, None] * ze
    area = rho[:, None] * ((re - rbc0[0]) * dze - ze * dre)
    points = jnp.asarray(np.stack([r, 0 * r, z], -1).reshape(-1, 3))
    bphi = np.asarray(jax.vmap(BiotSavart(coils).B)(points))[:, 1].reshape(r.shape)
    flux = float(np.sum(rw[:, None] * bphi * area) * 2 * np.pi / theta.size)
    scale = abs(float(inp.phiedge)) / abs(flux)
    coils = Coils(coils.curves, coils.dofs_currents_raw * scale, currents_scale=coils.currents_scale)

    metrics = coil_limits.coil_metrics(coils)
    report = dict(case=P.CASE, radius=args.radius, nit=int(fit.nit), message=str(fit.message),
                  bn_rms=float(normal_field_rms(coils, surface)), min_row=float(jnp.min(rows(coils))),
                  current_A=float(np.asarray(coils.currents)[0]), flux_scale=scale,
                  length=np.asarray(metrics["length"]).tolist(), peak=np.asarray(metrics["peak"]).tolist(),
                  msc=np.asarray(metrics["msc"]).tolist(), coil_distance=float(metrics["coil_distance"]),
                  clearance=float(coil_limits.surface_distance(coils, surface)))
    print(json.dumps(report), flush=True)
    coils.to_json(str(args.output))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
