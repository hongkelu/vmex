#!/usr/bin/env python
r"""Fixed-boundary single-stage counterpart of the coil-constraint benchmark.

The boundary and the coils form one variable vector, every trial solves the
fixed-boundary equilibrium, and SLSQP minimizes

    J = (1/2) |r_QS|^2 + (1/2) NORMAL_FIELD_WEIGHT rms(B.n/|B|)^2

(r_QS the constructed QI residual for ``COIL_CASE=qi``) subject to the hard
inequalities of ``parameters.py`` (minimum |iota|, aspect band, major radius,
the QI case's mirror ratio), the coil limits of ``_coil_constraints.py`` (length,
curvature, mean squared curvature, separation, plasma clearance) and the
normal-field limit. The B.n term keeps the prescribed boundary close to what
the coils produce, so the fixed-boundary QA stays meaningful for the coils.

In vacuum PHIEDGE only scales the field, so the coils may enclose any flux;
the coils' toroidal flux through the boundary over PHIEDGE is logged as
``flux_ratio`` (``postprocess.py --match-flux`` rescales the currents by it).

    python single_stage_optimization.py --steps 5 --output runs/fixed

``--beta`` is the finite-beta counterpart of the free-boundary arm: the same
fixed pressure p ~ 1 - s and zero net current, and PHIEDGE becomes a design
variable (``vary_phiedge``), so the field-strength band below holds B0 without
fixing the plasma size, as the free arm's fixed currents do. The plasma currents
then carry a field of their own, so the normal-field limit and objective term
apply to the total (B_coils + B_plasma).n/|B|, with B_plasma from virtual
casing on every trial's equilibrium and differentiated through it, and an exact
field-strength band replaces the coil-only flux: outside the plasma R B_phi is
set by the coils alone (Ampere's law), so the equilibrium's edge R B_phi =
bvco(s=1) must equal the coils' linked mu0 I / 2 pi at any beta. With p(1) = 0
and a zero total B.n the exterior field matches the interior one, so pressure
balance is checked at the end, with the total and coil-only B.n, on a 61 x 64
grid where virtual casing is planned afresh.

``--bootstrap`` (with ``--beta``) adds the free arm's self-consistent Redl
bootstrap current: the current-spline values and CURTOR become plasma variables,
and a plasma row holds the Redl mismatch under ``P.REDL_TOLERANCE``.

A rejected equilibrium trial counts the plasma rows as violated.

Continue a run with ``--coils <out>/coils.json --wout <out>/wout.nc``: the
boundary restarts from the WOUT and SLSQP from an identity Hessian.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402
from _common import (NORMAL_FIELD_CONSTRAINT, boundary_diagnostics, boundary_from_wout, bootstrap_input,  # noqa: E402
                     bootstrap_mismatch, coil_field, finite_beta_input, max_abs_iota, normal_field_rms,
                     redl_profiles, resize_coils, restart_input, scale_coil_currents, seed_input,
                     target_residual, weighted_rms)

MAX_MODE = 8                       # boundary modes varied; RBC(0,0) stays fixed
ESS_ALPHA = 1.2
# SLSQP's first step has an identity Hessian: the QH and QI residuals start near 1,
# 50 times the QA seed's, so their boundary coordinates are scaled down.
BOUNDARY_STEP, COIL_STEP = (0.1 if P.SEED is None else 0.02), P.COIL_STEP
NORMAL_FIELD_WEIGHT = 1.0e3
OPTIMIZER_FTOL = 1e-10
FIELD_STRENGTH_TOLERANCE = 0.005   # finite beta: relative band on edge R B_phi around the coils' mu0 I / 2 pi
VC_DIGITS = 4                      # significant digits of the virtual-casing plasma field
NPHI, NTHETA = 37, 32


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="SLSQP iterations")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--coils", type=Path, default=P.COILS_FILE)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="seed beta: on-axis (WOUT betaxis) for COIL_CASE=ellipse5-beta7, else volume-average")
    parser.add_argument("--wout", type=Path, help="restart the boundary from this WOUT's last surface")
    parser.add_argument("--save-every", type=int, default=25, help="save coils and WOUT every N steps")
    parser.add_argument("--bootstrap", action="store_true",
                        help="reactor-like kinetic profiles and a self-consistent Redl bootstrap current")
    parser.add_argument("--restart", type=Path, help="continue a finished run from its input.run, final coils and "
                        "WOUT boundary (no seed calibration); pass the run's --beta/--bootstrap")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils = args.restart / "coils.json"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    if args.device == "cpu":
        # On a GPU host JAX_PLATFORMS would count as a user placement and disable VMEX's CPU implicit default.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    import _coil_constraints as coil_limits
    from vmex.core import virtual_casing as vc
    from vmex.core.fields import surface_currents
    from vmex.core.statephysics import _field_chain  # private: the covariant fields behind the edge R B_phi

    started = time.monotonic()
    inp = seed_input()
    if args.wout is not None:
        inp = boundary_from_wout(inp, vj.read_wout(args.wout))
    redl = None
    if args.restart is not None:
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.bootstrap:
        inp, _, redl = bootstrap_input(inp, args.beta, args.device)
    else:  # also in vacuum: it sets PHIEDGE for B0
        inp, _ = finite_beta_input(inp, args.beta, args.device)
    phiedge = abs(float(inp.phiedge))
    inp.to_indata(out / "input.run")  # the deck as run: resolution, pressure and seed PHIEDGE
    mismatch = None if redl is None else bootstrap_mismatch(inp, redl, args.device)

    qs = target_residual()
    # In vacuum PHIEDGE only scales the plasma field, a null direction; at finite beta the
    # field-strength band ties it to the boundary.
    vary_phiedge = P.FREE_PHIEDGE and args.beta > 0
    plasma_problem = opt.VmecProblem.from_tuples(
        inp, [(qs.residuals_state, 0.0, 1.0)], max_mode=MAX_MODE, use_ess=True, ess_alpha=ESS_ALPHA,
        vary_phiedge=vary_phiedge, current_dofs=P.CURRENT_KNOTS - 1 if redl is not None else None)

    coils = resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS)
    if args.restart is None:  # a restart keeps the run's own currents
        coils = scale_coil_currents(coils, P.B0 * float(inp.rbc[inp.ntor, 0]))
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    x_boundary0 = plasma_problem.x0
    x_coils0 = np.asarray(coils0.curves.dofs).ravel()
    x0 = np.concatenate([x_boundary0, x_coils0])
    scales = np.concatenate([BOUNDARY_STEP * plasma_problem.scales, np.full(x_coils0.size, COIL_STEP)])
    n_boundary = x_boundary0.size  # plasma variables, with the PHIEDGE dof last when it varies
    if redl is not None:  # the current spline values (in units of the largest) and CURTOR/1e6, before PHIEDGE
        block = slice(n_boundary - int(vary_phiedge) - P.CURRENT_KNOTS, n_boundary - int(vary_phiedge))
        assert plasma_problem.names[block.stop - 1].startswith("CURTOR")
        scales[block] = P.CURRENT_STEP * np.r_[np.ones(P.CURRENT_KNOTS - 1), abs(float(inp.curtor)) / 1e6]

    def phiedge_at(x):
        return phiedge * (1.0 + x[n_boundary - 1]) if vary_phiedge else phiedge

    def objects_from_x(x, nphi=NPHI, ntheta=NTHETA):
        rbc, zbs = plasma_problem.boundary_from_x(x[:n_boundary])
        surface = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=nphi, ntheta=ntheta)
        coils = coils0.with_dofs(jnp.concatenate((x[n_boundary:], coils0.dofs_currents)))
        return rbc, zbs, surface, coils

    # Toroidal flux of the coil field through the boundary's phi = 0 cross-section:
    # Gauss-Legendre in radius, uniform in angle, differentiable in boundary and coils.
    rho, rho_weights = np.polynomial.legendre.leggauss(24)
    rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)

    def toroidal_flux(rbc, zbs, coils):
        m = np.arange(rbc.shape[1])
        cos_m, sin_m = np.cos(np.outer(theta, m)), np.sin(np.outer(theta, m))
        rbc0, zbs0 = rbc.sum(axis=0), zbs.sum(axis=0)
        r_edge, z_edge = cos_m @ rbc0, sin_m @ zbs0
        dr_edge, dz_edge = -sin_m @ (m * rbc0), cos_m @ (m * zbs0)
        r = rbc0[0] + rho[:, None] * (r_edge - rbc0[0])
        z = rho[:, None] * z_edge
        area = rho[:, None] * ((r_edge - rbc0[0]) * dz_edge - z_edge * dr_edge)
        points = jnp.stack([r, jnp.zeros_like(r), z], axis=-1).reshape(-1, 3)
        b_phi = jax.vmap(BiotSavart(coils).B)(points)[:, 1].reshape(r.shape)
        return jnp.sum(rho_weights[:, None] * b_phi * area) * (2.0 * np.pi / theta.size)

    if args.beta > 0:
        # Virtual casing picks its quadrature once, on the concrete seed, so the
        # plasma field stays differentiable in the boundary on every trial.
        seed = plasma_problem.equilibrium_from_x(x_boundary0)
        precision = vc.plan_vc_precision(vc.surface_field_data_from_state(
            inp, seed.solution, runtime=seed.solver_context, nphi=NPHI, ntheta=NTHETA), digits=VC_DIGITS)

    # Poloidal current the coils link, mu0 I / 2 pi = (1/2 pi) \oint R B_phi dphi on R = RBC(0,0), Z = 0.
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    loop = float(inp.rbc[inp.ntor, 0]) * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    b_phi = np.sum(np.asarray(coil_field(coils0)(jnp.asarray(loop))) * np.stack(
        [-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1), axis=-1)
    linked_rbtor = abs(float(inp.rbc[inp.ntor, 0]) * float(np.mean(b_phi)))
    print(f"coils link mu0 I / 2 pi = {linked_rbtor:.6f} T m")

    def rbtor_ratio(state, ctx):
        """Equilibrium edge R B_phi over the coils' linked mu0 I / 2 pi."""
        fields = _field_chain(state, ctx)[3]
        currents = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv, trig=ctx.trig,
                                    s=jnp.asarray(ctx.setup.s_full), signgs=ctx.setup.signgs)
        return jnp.abs(currents.rbtor) / linked_rbtor

    def total_normal_field_rms(coils, state, ctx):
        """Area-weighted RMS of (B_coils + B_plasma).n/|B| on the boundary."""
        data = vc.surface_field_data_from_state(inp, state, runtime=ctx, nphi=NPHI, ntheta=NTHETA)
        interface = vc.PlasmaVacuumInterface.from_surface_data(data, digits=VC_DIGITS, precision=precision)
        normal = interface.bnormal_residual(coil_field(coils)) / jnp.linalg.norm(data.B_total, axis=0)
        return weighted_rms(interface.weights, normal)

    aspect_lower, aspect_upper = P.ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN

    def plasma_rows(state, ctx, coils):
        """Rows that need the equilibrium, scaled so c >= 0 is feasible."""
        iota, aspect, radius = opt.min_abs_iota(state, ctx), opt.aspect_ratio(state, ctx), opt.major_radius(state, ctx)
        rows = [(iota - P.IOTA_FLOOR - P.IOTA_MARGIN) / P.IOTA_FLOOR,
                (aspect - aspect_lower) / aspect_scale, (aspect_upper - aspect) / aspect_scale,
                (radius - P.RADIUS_TARGET + width) / P.RADIUS_TOLERANCE,
                (P.RADIUS_TARGET + width - radius) / P.RADIUS_TOLERANCE]
        if P.MIRROR_LIMIT:
            rows.append((P.MIRROR_LIMIT - P.MIRROR_MARGIN - opt.mirror_ratio(state, ctx)) / P.MIRROR_LIMIT)
        if P.IOTA_CEILING:
            rows.append((P.IOTA_CEILING - P.IOTA_MARGIN - max_abs_iota(state, ctx)) / P.IOTA_FLOOR)
        if args.beta > 0:
            strength = rbtor_ratio(state, ctx) - 1.0
            rows += [1.0 - total_normal_field_rms(coils, state, ctx) / NORMAL_FIELD_CONSTRAINT,
                     (FIELD_STRENGTH_TOLERANCE - strength) / FIELD_STRENGTH_TOLERANCE,
                     (FIELD_STRENGTH_TOLERANCE + strength) / FIELD_STRENGTH_TOLERANCE]
        if redl is not None:
            rows.append(1.0 - mismatch(state, ctx) / P.REDL_TOLERANCE)
        return jnp.stack(rows)

    n_plasma = 5 + bool(P.MIRROR_LIMIT) + bool(P.IOTA_CEILING) + 3 * (args.beta > 0) + int(redl is not None)

    def plasma_constraint(u):
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        coils = objects_from_x(x)[3]
        rows, _ = plasma_problem.jax_quantity_from_state(
            x[:n_boundary], lambda state, ctx: plasma_rows(state, ctx, coils))
        return rows

    def coil_rows(u):
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        _, _, surface, coils = objects_from_x(x)
        clearance = coil_limits.surface_distance(coils, surface)
        rows = [coil_limits.coil_inequalities(coils),
                jnp.atleast_1d((clearance - P.COIL_SURFACE_DISTANCE_LIMIT - P.DISTANCE_MARGIN)
                               / P.COIL_SURFACE_DISTANCE_LIMIT),
                # At finite beta the normal-field limit is on the total field: plasma_rows.
                *([] if args.beta > 0 else [
                    jnp.atleast_1d(1.0 - normal_field_rms(coils, surface) / NORMAL_FIELD_CONSTRAINT)])]
        return jnp.concatenate(rows)

    coil_rows_jit = jax.jit(coil_rows)
    plasma_rows_jit = jax.jit(plasma_constraint)
    # One adjoint per plasma row: the gradient of w . rows at a unit w, the forward solve reused.
    plasma_row_grad = jax.jit(jax.grad(lambda u, w: jnp.vdot(w, plasma_constraint(u))))
    coil_rows_jac = jax.jit(jax.jacrev(coil_rows))
    cache = {}
    def half_qa(u):
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        return plasma_problem.jax_objective_from_state(x[:n_boundary], lambda state, ctx: jnp.zeros(1),
                                                       n_extra_terms=1)[0]

    qa_value_and_grad = jax.jit(jax.value_and_grad(half_qa))

    def normal_field_cost(u):
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        _, _, surface, coils = objects_from_x(x)
        if args.beta > 0:  # the total field, through the equilibrium
            return plasma_problem.jax_extra_costs_from_state(x[:n_boundary], lambda state, ctx: (
                0.5 * NORMAL_FIELD_WEIGHT * total_normal_field_rms(coils, state, ctx)**2)[None], n_extra_terms=1)[0]
        return 0.5 * NORMAL_FIELD_WEIGHT * normal_field_rms(coils, surface)**2

    normal_field_value_and_grad = jax.jit(jax.value_and_grad(normal_field_cost))

    def objective(u):
        u = np.asarray(u, dtype=float)
        if cache.get("key") != u.tobytes():
            half_qa, gradient = qa_value_and_grad(jnp.asarray(u))
            extra, extra_gradient = normal_field_value_and_grad(jnp.asarray(u))
            cache.update(key=u.tobytes(), qa=2 * float(half_qa), value=float(half_qa + extra),
                         gradient=np.asarray(gradient + extra_gradient))
        return cache["value"], cache["gradient"].copy()

    def plasma_values(u):
        """Plasma rows; a rejected equilibrium (NaN rows) counts as violated."""
        rows = np.asarray(plasma_rows_jit(jnp.asarray(np.asarray(u, dtype=float))))
        return np.where(np.isfinite(rows), rows, -1.0)

    def plasma_jacobian(u):
        u = jnp.asarray(np.asarray(u, dtype=float))
        jac = np.stack([np.asarray(plasma_row_grad(u, jnp.eye(n_plasma)[i])) for i in range(n_plasma)])
        return np.where(np.isfinite(jac), jac, 0.0)

    last = dict(time=time.monotonic(), step=0)

    def log_step(u):
        u = np.asarray(u, dtype=float)
        value, _ = objective(u)
        x = x0 + scales * u
        equilibrium = plasma_problem.equilibrium_from_x(x[:n_boundary])
        state, ctx = equilibrium.solution, equilibrium.runtime
        rbc, zbs, surface, coils = objects_from_x(jnp.asarray(x))
        rows = np.asarray(coil_limits.coil_inequalities(coils))
        now = time.monotonic()
        row = dict(step=last["step"], qa=cache["qa"], objective=value,
                   min_abs_iota=float(opt.min_abs_iota(state, ctx)), aspect=float(opt.aspect_ratio(state, ctx)),
                   major_radius_m=float(opt.major_radius(state, ctx)),
                   **({"mirror_ratio": float(opt.mirror_ratio(state, ctx))} if P.MIRROR_LIMIT else {}),
                   coil_surface_distance_m=float(coil_limits.surface_distance(coils, surface)),
                   coil_minimum_scaled_slack=float(np.min(rows)),
                   normal_field_rms=float(normal_field_rms(coils, surface)),
                   plasma_minimum_scaled_slack=float(np.min(plasma_values(u))),
                   flux_ratio=float(abs(toroidal_flux(rbc, zbs, coils)) / phiedge_at(x)),
                   phiedge=float(phiedge_at(x)),
                   beta=float(opt.volume_average_beta(state, ctx)),
                   step_seconds=now - last["time"], elapsed_seconds=now - started)
        if args.beta > 0:
            row.update(total_normal_field_rms=float(total_normal_field_rms(coils, state, ctx)),
                       rbtor_ratio=float(rbtor_ratio(state, ctx)))
        if redl is not None:
            row.update(redl_mismatch=float(mismatch(state, ctx)), curtor=float(equilibrium.inp.curtor))
        last.update(time=now, step=last["step"] + 1)
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {P.TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"aspect={row['aspect']:.4f} R={row['major_radius_m']:.5f} flux={row['flux_ratio']:.5f} "
              f"B.n={row.get('total_normal_field_rms', row['normal_field_rms']):.2e} beta={row['beta']:.4%} "
              f"RBphi={row.get('rbtor_ratio', float('nan')):.5f} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              f"plasma_slack={row['plasma_minimum_scaled_slack']:.4f} "
              + (f"{P.BOOTSTRAP_MODEL}={row['redl_mismatch']:.2e} CURTOR={row['curtor']:.0f}A " if redl is not None else "") +
              f"{row['step_seconds']:.1f}s", flush=True)

    def save(tag, u):
        x = x0 + scales * np.asarray(u, dtype=float)
        objects_from_x(jnp.asarray(x))[3].to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), plasma_problem.equilibrium_from_x(x[:n_boundary]).wout)

    def checkpoint(u):
        log_step(u)
        if args.save_every and (last["step"] - 1) % args.save_every == 0:
            save(f".step{last['step'] - 1}", u)

    u0 = np.zeros_like(x0)
    checkpoint(u0)
    constraints = [dict(type="ineq", fun=plasma_values, jac=plasma_jacobian),
                   dict(type="ineq", fun=lambda u: np.asarray(coil_rows_jit(jnp.asarray(u))),
                        jac=lambda u: np.asarray(coil_rows_jac(jnp.asarray(u))))]
    result = minimize(objective, u0, jac=True, method="SLSQP", callback=checkpoint,
                      constraints=constraints, options={"maxiter": args.steps, "ftol": OPTIMIZER_FTOL})
    x = x0 + scales * result.x
    coils = objects_from_x(jnp.asarray(x))[3]
    coils.to_json(str(out / "coils.json"))
    wout = plasma_problem.equilibrium_from_x(x[:n_boundary]).wout
    vj.write_wout(str(out / "wout.nc"), wout)
    summary = dict(iterations=int(result.nit), evaluations=int(result.nfev), success=bool(result.success),
                   message=str(result.message), elapsed_seconds=time.monotonic() - started,
                   beta_target=args.beta, pres_scale=float(inp.pres_scale), endpoint=boundary_diagnostics(wout, coils))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
