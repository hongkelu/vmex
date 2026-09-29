#!/usr/bin/env python
"""Free-boundary single-stage coil optimization with hard coil constraints.

Only the coils vary. Each trial solves the vacuum free-boundary equilibrium of
those coils, predicted from the accepted root and certified before use, and
SLSQP minimizes quasisymmetry subject to hard inequalities: minimum |iota|,
major radius, aspect ratio, coil-to-plasma clearance, and per-coil length,
curvature, mean squared curvature and coil separation (``parameters.py``).

PHIEDGE is a design variable and the coil currents are fixed at B0 = 1 T. In
vacuum only the flux per ampere sets the plasma size, so fixing both the
currents and PHIEDGE pins it: at iota >= 0.41 that holds the aspect ratio at its
4.9 floor (QA ~0.02), while a free PHIEDGE lets SLSQP reach aspect 5.1 (QA
~0.004). FREE_PHIEDGE = False restores the pinned case.

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free

Every accepted step appends one line to ``metrics.jsonl``. Coils and a WOUT
are saved every ``--save-every`` steps and at the end; restart from them with
``--coils <out>/coils.json --wout <out>/wout.nc``.

``--beta 0.005`` runs the same case at finite beta: a fixed pressure p ~ 1 - s,
scaled once so the fixed-boundary seed has that volume-average beta, with zero
net toroidal current. With fixed currents and pressure, a free PHIEDGE changes
the size but hardly B0 or beta. The loss and constraints are unchanged.
The free-boundary solve already makes
the boundary a flux surface of the coil plus plasma field, so the virtual-casing
B.n/|B| and pressure balance are saved diagnostics only, together with beta.
Before the free-boundary seed solve the coils are refitted to the finite-beta
fixed-boundary seed (no equilibrium solves): they must cancel the plasma's own
normal field, (B_coils + B_plasma).n = 0 with B_plasma from virtual casing,
under the coil limits of ``parameters.py`` as penalties.

``--bootstrap`` (with ``--beta``) replaces the zero current with a self-consistent
Redl bootstrap current (``bootstrap_input``): the current-spline values and CURTOR
join PHIEDGE as design variables, and the Redl mismatch is a hard constraint.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402

# Numerical controls of the equilibrium, adjoint and matrix-free solves.
ROOT_TOLERANCE, ROOT_POLISH_TOLERANCE = 2e-6, 1e-12
ROOT_POLISH_STEPS = 10             # damped Newton steps may be needed from a 1e-9 ordinary-solve residual
OPTIMIZER_FTOL = 1e-10
ADJOINT_RESIDUAL_RTOL, ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 1e-9, 32, 20000
MATRIXFREE = dict(rtol=1e-11, restart=100, max_restarts=3, rhs_batch_size=3)
LU_REFRESH_HORIZON = 10
# Finite-beta seed coil refit: iterations, B.n/|B| unit, and penalty weight of the scaled coil rows.
COIL_FIT_MAXITER, COIL_FIT_NORMAL_SCALE, COIL_FIT_WEIGHT = 200, 1.0e-3, 1.0e3
NEWTON_STEPS = 8  # Newton-correct predicted trials on the seed LU before any ordinary solve
PHIEDGE_STEP = 0.05  # coordinate scale of the relative PHIEDGE change
DENSE_DERIVATIVES = True  # every derivative a dense solve whose LU seeds the next step's trials (~2x faster)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="accepted SLSQP steps")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--coils", type=Path, default=P.COILS_FILE)
    parser.add_argument("--no-coil-fit", action="store_true", help="use --coils as given, without the seed refit")
    parser.add_argument("--wout", type=Path, help="restart the initial solve from this WOUT")
    parser.add_argument("--max-seconds", type=float, default=float("inf"), help="optimization wall-time budget")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="seed beta: volume-average, or on-axis (WOUT betaxis) for COIL_CASE=qa6")
    parser.add_argument("--bootstrap", action="store_true",
                        help="reactor-like kinetic profiles and a self-consistent Redl bootstrap current")
    parser.add_argument("--restart", type=Path, help="continue a finished run from its input.run, final coils and "
                        "WOUT (no seed calibration or coil refit); pass the run's --beta/--bootstrap")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils, args.wout = args.restart / "coils.json", args.restart / "wout.nc"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    return args


def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes without changing the curves or currents."""
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    old = coils.curves
    raw = jnp.pad(old.dofs / old.scaling, ((0, 0), (0, 0), (0, 2 * (order - coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor, scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)


def finite_beta_input(inp, beta, device, am=(1.0, -1.0)):
    """Give ``inp`` a pressure p ~ ``am`` (power series in s, default 1 - s) whose fixed-boundary seed has ``beta``.

    ``P.BETA_DEFINITION`` "volume": ``beta`` is <beta>; "axis": WOUT ``betaxis``.
    The pressure is ramped in with hot restarts, then rescaled to ``beta``.
    With ``P.B0`` set, PHIEDGE is rescaled until the edge R B_phi (the coils'
    mu0 I / 2 pi) is B0 R0, so the field strength follows the flux as in
    upstream #426. Returns the input and the last fixed-boundary solve.
    """
    import numpy as np
    from vmex import optimize as opt

    axis, r0 = P.BETA_DEFINITION == "axis", float(inp.rbc[inp.ntor, 0])
    b0 = 1.0 if P.B0 is None else P.B0
    pressure = beta * b0**2 / (8e-7 * np.pi) if axis else beta / (4e-7 * np.pi)
    shape = np.zeros_like(np.asarray(inp.am, dtype=float))
    shape[: len(am)] = am
    inp = replace(inp, am=shape)
    fixed = None
    ramp = (0.25, 0.5, 0.75, 1.0) if beta > 0 else (1.0,)
    corrections = 3 if axis or P.B0 is not None else 1
    for index in range(len(ramp) + corrections):
        if index < len(ramp):
            inp = replace(inp, pres_scale=ramp[index] * pressure)
        else:
            # beta ~ p / PHIEDGE^2 at fixed shape, so a flux rescale carries its pressure along
            measured = float(fixed.wout.betaxis if axis else fixed.wout.betatotal)
            flux = 1.0 if P.B0 is None else b0 * r0 / abs(float(fixed.wout.rbtor))
            inp = replace(inp, phiedge=float(inp.phiedge) * flux,
                          pres_scale=inp.pres_scale * flux**2 * (beta / measured if beta > 0 else 1.0))
        fixed = opt.solve_equilibrium(inp, initial_state=None if fixed is None else fixed.state, device=device,
                                      raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"PRES_SCALE = {inp.pres_scale:.6e} Pa, PHIEDGE = {float(inp.phiedge):.6f} Wb: betaxis = "
          f"{float(w.betaxis):.4%}, <beta> = {float(w.betatotal):.4%}, edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
    return inp, fixed


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch."""
    import numpy as np
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    r0, b0 = float(inp.rbc[inp.ntor, 0]), 1.0 if P.B0 is None else P.B0
    t0 = P.REACTOR_T0 * (b0 / P.REACTOR_B0) ** (2 / 3) * (r0 / P.REACTOR_R0) ** (1 / 3)
    n0 = P.REACTOR_N0 * (b0 / P.REACTOR_B0) ** (4 / 3) * (P.REACTOR_R0 / r0) ** (1 / 3)
    scale = float(inp.pres_scale) / (2 * ELEMENTARY_CHARGE * n0 * t0)
    n0, t0 = n0 * scale ** (2 / 3), t0 * scale ** (1 / 3)
    profiles = KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), t0 * np.array([1.0, -1.0]),
                               t0 * np.array([1.0, -1.0]))
    return profiles, RedlBootstrapMismatch(profiles, 0, np.asarray(P.REDL_SURFACES), n_lambda=P.REDL_N_LAMBDA)


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import numpy as np
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(w.xm, int), np.asarray(w.xn, int) // int(w.nfp),
                          np.asarray(w.rmnc)[-1], np.asarray(w.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=float(w.phi[-1]), lfreeb=False)
    if int(inp.ncurr) == 1:
        spline = "spline" in str(inp.pcurr_type)
        field_name, values = ("ac_aux_f", w.ac_aux_f) if spline else ("ac", w.ac)
        inp = replace(inp, curtor=float(w.ctor),
                      **{field_name: np.asarray(values, dtype=float)[: np.size(getattr(inp, field_name))]})
    return inp


def bootstrap_input(inp, beta, device):
    """``finite_beta_input`` with reactor-like kinetic profiles and their self-consistent Redl current.

    ne = n0 (1 - s^5) and Te = Ti = T0 (1 - s), so p = 2 e ne Te ~ (1 - s)(1 - s^5). n0 and T0
    start at the Helios-like reactor's collisionality nu* ~ n R / T^2 and beta ~ n T / B^2 moved
    to this R0 and B0, and follow the beta calibration as n ~ p^(2/3), T ~ p^(1/3), which keeps
    nu*. A Picard loop then makes the current Redl's, and it is resampled onto
    ``P.CURRENT_KNOTS`` spline knots. Returns the input, the equilibrium and the Redl mismatch.
    """
    import numpy as np
    from vmex import optimize as opt
    from vmex.core.bootstrap import self_consistent_bootstrap

    inp, _ = finite_beta_input(inp, beta, device, am=(1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
    profiles, redl = redl_profiles(inp)
    n0, t0 = float(profiles.ne_coeffs[0]), float(profiles.Te_coeffs[0])
    ac = np.zeros_like(np.asarray(inp.ac, dtype=float))
    ac[0] = 1.0
    inp = replace(inp, ncurr=1, pcurr_type="power_series", ac=ac, curtor=0.0)
    picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=P.PICARD_ITERATIONS, tol=P.PICARD_TOLERANCE,
                                       degree=P.CURRENT_KNOTS - 1, s_eval=np.asarray(P.REDL_SURFACES),
                                       solve_kwargs=dict(device=device))
    inp = opt.resample_current_profile(picard.input, P.CURRENT_KNOTS)
    fixed = opt.solve_equilibrium(inp, initial_state=picard.equilibrium.state, device=device,
                                  raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"Redl seed: n0 = {n0:.4e} 1/m^3, T0 = {t0:.1f} eV, Picard {picard.iterations} iterations "
          f"(converged {picard.converged}), CURTOR = {float(inp.curtor):.1f} A, mismatch = "
          f"{float(redl.total(w)):.3e}; <beta> = {float(w.betatotal):.4%}, iota = {float(np.min(np.abs(w.iotaf))):.4f}"
          f"..{float(np.max(np.abs(w.iotaf))):.4f}, edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
    return inp, fixed, redl


def scale_coil_currents(coils, rbtor):
    """Coils with every current scaled so the linked mu0 I / 2 pi (R B_phi at R = 1 m, Z = 0) is ``rbtor``."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils
    from essos.fields import BiotSavart

    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    points = jnp.asarray(np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1))
    field = np.asarray(jax.vmap(BiotSavart(coils).B)(points))
    linked = abs(float(np.mean(-np.sin(phi) * field[:, 0] + np.cos(phi) * field[:, 1])))
    return Coils(coils.curves, coils.dofs_currents_raw * (rbtor / linked), currents_scale=coils.currents_scale)


def boundary_diagnostics(wout, coils, nphi=61, ntheta=64):
    """Interface diagnostics of a WOUT in its coils' field; virtual casing is planned on this grid.

    Returns beta, the RMS/max of (B_coils + B_plasma).n/|B|, the RMS of the
    coil-only B.n/|B|, and the RMS of the pressure-balance residual
    (|B_out|^2 - |B_in|^2 - 2 mu0 p) / |B_in|^2.
    """
    import jax
    import jax.numpy as jnp
    from essos.fields import BiotSavart
    import vmex as vj

    biot_savart = BiotSavart(coils)

    def field(points):
        return jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)

    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=nphi, ntheta=ntheta)
    weights = interface.weights
    total = interface.bnormal_residual(field) / jnp.linalg.norm(interface.total_B_out(field), axis=0)
    coil_only = interface.external_Bn(field) / jnp.linalg.norm(interface.external_B(field), axis=0)
    balance = interface.pressure_balance_residual(field) / interface.Bin_mag2
    return dict(beta=float(wout.betatotal), normal_field_rms=float(jnp.sqrt(jnp.sum(weights * total**2))),
                normal_field_max=float(jnp.max(jnp.abs(total))),
                coil_normal_field_rms=float(jnp.sqrt(jnp.sum(weights * coil_only**2))),
                pressure_balance_rms=float(jnp.sqrt(jnp.sum(weights * balance**2))))


def fit_coils_to_plasma(coils, wout, inp, *, maxiter=COIL_FIT_MAXITER):
    """Coils whose field with the plasma's own is tangent to a fixed-boundary finite-beta seed."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj
    import _coil_constraints as coil_limits

    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=37, ntheta=32)
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=coil_limits.SURFACE_GRID[0], ntheta=coil_limits.SURFACE_GRID[1])
    x0 = jnp.asarray(coils.curves.dofs).ravel()

    def coils_from_u(u):
        return coils.with_dofs(jnp.concatenate((x0 + P.COIL_STEP * u, coils.dofs_currents)))

    def normal_field_rms(new):
        biot_savart = BiotSavart(new)

        def field(points):
            return jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)

        normal = interface.bnormal_residual(field) / jnp.linalg.norm(interface.total_B_out(field), axis=0)
        return jnp.sqrt(jnp.sum(interface.weights * normal**2))

    def objective(u):
        new = coils_from_u(u)
        rows = jnp.concatenate([coil_limits.coil_inequalities(new), jnp.atleast_1d(
            (coil_limits.surface_distance(new, surface) - P.COIL_SURFACE_DISTANCE_LIMIT - P.DISTANCE_MARGIN)
            / P.COIL_SURFACE_DISTANCE_LIMIT)])
        return (0.5 * (normal_field_rms(new) / COIL_FIT_NORMAL_SCALE)**2
                + 0.5 * COIL_FIT_WEIGHT * jnp.sum(jnp.minimum(rows, 0.0)**2))

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    before = float(normal_field_rms(coils))
    fit = minimize(lambda u: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))), np.zeros(x0.size), jac=True,
                   method="L-BFGS-B", bounds=[(-5.0, 5.0)] * x0.size, options=dict(maxiter=maxiter, maxcor=20))
    fitted = coils_from_u(jnp.asarray(fit.x))
    print(f"[coil fit] {fit.nit} L-BFGS-B iterations: (B_coils + B_plasma).n/|B| RMS "
          f"{before:.3e} -> {float(normal_field_rms(fitted)):.3e} on the fixed-boundary seed", flush=True)
    return fitted


def main(argv=None):
    args = parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if args.device == "gpu" else "cpu")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils
    from essos.surfaces import surfacerzfourier_from_boundary
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im
    import _coil_constraints as coil_limits

    started = time.monotonic()
    mpol, ntor, ns = P.RESOLUTION
    inp = vj.VmecInput.from_file(P.INPUT_FILE)
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=P.GRID[0], nzeta=P.GRID[1])
    inp = replace(inp, ns_array=np.array([ns]), ftol_array=np.array([P.EQUILIBRIUM_FTOL]), lfreeb=False)
    if P.NITER is not None:
        inp = replace(inp, niter_array=np.array([P.NITER]), delt=P.DELT)
    seed = redl = None
    if args.restart is not None:
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.bootstrap:
        inp, fixed, redl = bootstrap_input(inp, args.beta, args.device)
        seed = fixed.state
    elif args.beta > 0 or P.B0 is not None:
        inp, fixed = finite_beta_input(inp, args.beta, args.device)
        seed = fixed.state
    if args.wout is not None:
        seed = vj.state_from_wout(vj.read_wout(args.wout), inp=inp, ns=ns)
    elif seed is None:
        seed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                     polish_force_balance=False).state
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
    inp.to_indata(out / "input.run")  # the deck as run: resolution, pressure and seed PHIEDGE

    # Reload the saved coils so a restart builds a bit-identical coordinate chart.
    coils = resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS)
    if P.B0 is not None and args.restart is None:  # a restart keeps the run's own currents
        coils = scale_coil_currents(coils, P.B0 * float(inp.rbc[inp.ntor, 0]))
    if args.beta > 0 and args.wout is None and not args.no_coil_fit:  # at beta = 0 a uniform scale keeps B.n/|B|
        coils = fit_coils_to_plasma(coils, fixed.wout, replace(inp, lfreeb=False))
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    # --bootstrap: the spline values but the last, then CURTOR, follow PHIEDGE as design variables.
    current = np.r_[np.asarray(inp.ac_aux_f)[: P.CURRENT_KNOTS - 1], inp.curtor] if args.bootstrap else None
    scales = np.r_[[PHIEDGE_STEP] * P.FREE_PHIEDGE, [P.CURRENT_STEP] * (0 if current is None else current.size),
                   P.COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = opt.CoilParameters.from_coils(coils0, current_dofs=(), scales=scales,
                                          phiedge=float(inp.phiedge) if P.FREE_PHIEDGE else None,
                                          plasma_current=current, plasma_current_spline=True)
    qs = opt.QuasisymmetryRatioResidual(np.asarray(P.QA_SURFACES), 1, 0)

    def boundary(state, runtime, grid):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)
        rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
        rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
        zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=grid[0], ntheta=grid[1])

    aspect_lower, aspect_upper = P.ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN

    def clearance(state, runtime, coils):
        return coil_limits.surface_distance(coils, boundary(state, runtime, coil_limits.SURFACE_GRID))

    def qa(state, runtime):
        residuals = qs.residuals_state(state, runtime)
        return jnp.vdot(residuals, residuals)

    def loss(state, runtime, coils):
        return 0.5 * qa(state, runtime)

    def aspect(state, runtime, coils):
        return opt.aspect_ratio(state, runtime)

    seconds = {}

    def record(name, **data):
        if name == "proposal":
            seconds["trials"] = seconds.get("trials", 0) + 1
        if name == "newton_correction":  # keep why a trial fell back to the ordinary solve
            with open(out / "events.jsonl", "a") as stream:
                stream.write(json.dumps(dict(event=name, **{k: v for k, v in data.items()
                                                            if isinstance(v, (bool, int, float, str))})) + "\n")
        if "seconds" in data:
            seconds[name] = seconds.get(name, 0.0) + float(data["seconds"])

    from jax import monitoring
    # Time actually spent tracing and compiling; the cache's "compile_time_saved" is not.
    monitoring.register_event_duration_secs_listener(lambda event, duration, **_: record(
        "compile", seconds=duration) if event.startswith("/jax/core/compile/") else None)

    problem = opt.FreeBoundaryProblem.from_loss(
        inp, loss, quantities=(opt.min_abs_iota, opt.major_radius, *([redl.total_state] if redl is not None else [])),
        coil_quantities=(clearance, aspect),
        parameterization=chart, restart_from=seed, root_residual_atol=ROOT_TOLERANCE, event=record,
        deadline=started + args.max_seconds,
        solver_options=dict(device=args.device, ftol=P.EQUILIBRIUM_FTOL, edge_force_tolerance=P.EQUILIBRIUM_FTOL,
                            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
                            adjoint_dense_max_dofs=ADJOINT_MAX_DOFS, adjoint_residual_rtol=ADJOINT_RESIDUAL_RTOL))
    problem.enable_root_polishing(tolerance=ROOT_POLISH_TOLERANCE, max_steps=ROOT_POLISH_STEPS)
    problem.enable_matrix_free(**MATRIXFREE, refresh_horizon=LU_REFRESH_HORIZON, refresh_max_steps=args.steps,
                               dense_derivatives=DENSE_DERIVATIVES)
    if NEWTON_STEPS:
        problem.enable_newton_correction(max_steps=NEWTON_STEPS)

    coil_rows = coil_limits.constraint(chart.coils_from_x)
    for method in ("fun", "jac"):
        def timed(x, _call=getattr(coil_rows, method), _name=f"coil_constraint_{method}"):
            start = time.monotonic()
            value = _call(x)
            record(_name, seconds=time.monotonic() - start)
            return value
        setattr(coil_rows, method, timed)
    nredl = int(redl is not None)  # the Redl mismatch row sits between the plasma and coil quantities
    constraints = [problem.nonlinear_constraint(
        [P.IOTA_FLOOR + P.IOTA_MARGIN, P.RADIUS_TARGET - width, *[-np.inf] * nredl,
         P.COIL_SURFACE_DISTANCE_LIMIT + P.DISTANCE_MARGIN, aspect_lower],
        [np.inf, P.RADIUS_TARGET + width, *[P.REDL_TOLERANCE] * nredl, np.inf, aspect_upper],
        scales=[P.IOTA_FLOOR, P.RADIUS_TOLERANCE, *[P.REDL_TOLERANCE] * nredl, P.COIL_SURFACE_DISTANCE_LIMIT,
                aspect_scale]), coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        problem.coils_from_x(x).to_json(str(out / f"coils{tag}.json"))
        wout = problem.equilibrium_from_x(x).wout
        vj.write_wout(str(out / f"wout{tag}.nc"), wout)
        # Diagnostics only: none of these enter the loss or the constraints.
        row = dict(step=problem.accepted_step, phiedge=float(wout.phi[-1]), b0=float(wout.b0),
                   rbtor=abs(float(wout.rbtor)), betaxis=float(wout.betaxis))
        if args.beta > 0:
            row.update(boundary_diagnostics(wout, problem.coils_from_x(x)))
        with open(out / "diagnostics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[diagnostics] PHIEDGE={row['phiedge']:.5f} Wb B0={row['b0']:.4f} T R B_phi={row['rbtor']:.4f} T m "
              f"betaxis={row['betaxis']:.4%}", flush=True)
        if args.beta > 0:
            print(f"[diagnostics] beta={row['beta']:.4%} B.n/|B| rms={row['normal_field_rms']:.3e} "
                  f"max={row['normal_field_max']:.3e} coil-only rms={row['coil_normal_field_rms']:.3e} "
                  f"pressure balance rms={row['pressure_balance_rms']:.3e}")

    last = dict(time=time.monotonic(), x=problem.accepted.parameters.copy())
    def log_step():
        x, record = problem.accepted.parameters, problem.accepted
        iota, radius, *mismatch, surface, aspect_value = map(float, problem.constraint_values(x))
        now = time.monotonic()
        objective = problem.fun(x)
        row = dict(step=problem.accepted_step, qa=2 * objective, objective=objective, min_abs_iota=iota, major_radius_m=radius,
                   aspect=aspect_value, coil_surface_distance_m=surface,
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   phiedge_factor=float(chart.phiedge_at(x) / chart.phiedge) if P.FREE_PHIEDGE else 1.0,
                   **(dict(redl_mismatch=mismatch[0], curtor=float(chart.plasma_params_at(
                       problem.cfg.params, x).curtor)) if mismatch else {}),
                   step_u_linf=float(np.max(np.abs((x - last["x"]) / problem.scales))),
                   root_residual=float(record.root_residual_norm), fedge=float(record.result.fedge),
                   step_seconds=now - last["time"], elapsed_seconds=now - started,
                   seconds={k: round(v, 3) for k, v in seconds.items()},
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        seconds.clear()
        last.update(time=now, x=x.copy())
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}) + "\n")
        print(f"[step {row['step']}] QA={row['qa']:.6e} iota={iota:.5f} R={radius:.5f} aspect={aspect_value:.4f} "
              f"clearance={surface:.4f} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              + (f"redl={mismatch[0]:.2e} CURTOR={row['curtor']:.0f}A " if mismatch else "") +
              f"{row['step_seconds']:.1f}s", flush=True)
        if row["step"] % args.save_every == 0:
            save(f".step{row['step']}")

    log_step()
    try:
        while True:  # a rejected equilibrium trial ends an SLSQP call; restart it from the accepted root
            before = problem.accepted_step
            result = opt.minimize(problem, x0=problem.accepted.parameters, method="SLSQP", constraints=constraints,
                                  callback=lambda x: log_step(),
                                  options=dict(maxiter=args.steps - before, ftol=OPTIMIZER_FTOL))
            if (result.stop_reason != "equilibrium_trial_rejected" or problem.accepted_step == before
                    or problem.accepted_step >= args.steps):
                break
        stop = dict(success=bool(result.success), message=str(result.message),
                    stop_reason=getattr(result, "stop_reason", None))
    except TimeoutError:
        stop = dict(success=False, message="wall-time budget reached", stop_reason="walltime")
    save("")
    summary = dict(accepted_steps=problem.accepted_step, **stop, beta_target=args.beta,
                   pres_scale=float(inp.pres_scale), solver=problem.solver_info,
                   failed_trials=problem.metadata["holder"]["failed_trials"],
                   elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str))
    problem.close()


if __name__ == "__main__":
    main()
