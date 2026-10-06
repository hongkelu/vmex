#!/usr/bin/env python
"""Free-boundary single-stage coil optimization with hard coil constraints.

The design variables are the coil shapes, PHIEDGE (``P.FREE_PHIEDGE``) and,
with ``--bootstrap``, the plasma current spline values and CURTOR; the coil
currents are fixed. Every trial solves the free-boundary equilibrium of those
coils at the case's pressure and current, predicted from the accepted root and
certified before use (``vmex.optimize.FreeBoundaryProblem``), so the boundary
is a flux surface of the coil plus plasma field and no B.n term is needed.
SLSQP minimizes

    J = (1/2) |r|^2

with r the quasisymmetry residual of ``P.HELICITY`` (or the constructed QI
residual when it is None), subject to hard inequalities, in this row order:

- min |iota| >= ``IOTA_FLOOR`` (with ``IOTA_AXIS`` on the axis too); major radius within ``RADIUS_TOLERANCE`` of R0;
- the edge mirror ratio <= ``MIRROR_LIMIT`` and max |iota| <= ``IOTA_CEILING``,
  for the cases that set them;
- with ``--bootstrap``, the bootstrap mismatch <= ``REDL_TOLERANCE``;
- coil-to-plasma clearance, aspect ratio in ``ASPECT_RANGE``;
- and the pure-coil rows of ``_coil_constraints.py``: per-coil length,
  curvature and mean squared curvature, coil-coil distance.

Every trial is an ordinary VMEC free-boundary solve from the tangent prediction,
then Newton-polished onto the coupled root, as the fixed arm refines its VMEC
solves; there is no Newton correction in place of the VMEC solve.

The limits are in ``parameters.py``. The coil currents are scaled once so the
edge R B_phi is B0 R0 (B0 = 1 T), or at finite beta the seed's own edge R B_phi;
PHIEDGE then sets the plasma size. Fixing
both pins the size: for ``ellipse5`` in vacuum that holds the aspect ratio at
its 4.9 floor (QA ~0.02), while a free PHIEDGE reaches 5.1 (QA ~0.004).

``--beta`` sets a fixed pressure p ~ 1 - s calibrated on the fixed-boundary seed
(<beta>, or on-axis beta for ``ellipse5-beta7``, ``P.BETA_DEFINITION``) with zero
net current, and refits the seed coils to (B_coils + B_plasma).n = 0 on that
seed (virtual-casing B_plasma, coil limits as penalties, no equilibrium
solves). A converged free-boundary state has B.n = 0 by construction, so no
virtual-casing B.n is evaluated during the run. ``--bootstrap`` (with ``--beta``) uses reactor-like
kinetic profiles and a self-consistent bootstrap current, Redl's or, for
``P.BOOTSTRAP_MODEL = "dkx"`` (``qi6-beta*``, needs the ``dkx`` package), DKX's.

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
    COIL_CASE=qa4-beta python free_boundary_single_stage_optimization.py --beta 0.01 --bootstrap --output runs/free-qa4

Outputs in ``--output``: ``input.run`` (the deck as run), ``coils.initial.json``,
one ``metrics.jsonl`` line per accepted step (``qa``, ``min_abs_iota``,
``redl_mismatch``, ...), ``diagnostics.jsonl``,
``coils.stepN.json`` / ``wout.stepN.nc`` every ``--save-every`` steps, and the
final ``coils.json``, ``wout.nc`` and ``summary.json``. ``--restart <run>``
continues a finished run from its deck, coils and WOUT.

Derivatives use structured factors of the coupled Jacobian (radial block
tridiagonal plus NESTOR's low-rank coupling, ``--factorization structured``):
O(ns) memory, so ``--ns``/``--modes`` can exceed what the dense LU
(``--factorization dense``, O(ns^2)) fits on a GPU, with the same accuracy gate.
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
from _common import (bootstrap_input, bootstrap_mismatch, coil_field,  # noqa: E402
                     finite_beta_input, max_abs_iota, min_abs_iota, redl_profiles, resize_coils, restart_input,
                     scale_coil_currents, seed_input, target_residual, total_normal_field, weighted_rms)

# Numerical controls of the equilibrium, adjoint and matrix-free solves.
ROOT_TOLERANCE, ROOT_POLISH_TOLERANCE = 2e-6, 1e-12
ROOT_POLISH_STEPS = 30             # damped Newton from a 1e-9 ordinary-solve residual (a finite-beta restart needs >10)
ROOT_POLISH_LADDER = (1e-12, 1e-10, 1e-9, 1e-8, 1e-7)  # loosest polish target the start may settle on
OPTIMIZER_FTOL = 1e-10
ADJOINT_RESIDUAL_RTOL, ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 1e-9, 32, 40000  # the cap binds --factorization dense only
MATRIXFREE = dict(rtol=1e-11, restart=100, max_restarts=3, rhs_batch_size=6)  # derivative rows per GMRES batch
LU_REFRESH_HORIZON = 10
# Finite-beta seed coil refit: iterations, B.n/|B| unit, and penalty weight of the scaled coil rows.
COIL_FIT_NORMAL_SCALE, COIL_FIT_WEIGHT = 1.0e-3, 1.0e3
PHIEDGE_STEP = 0.05  # coordinate scale of the relative PHIEDGE change
DENSE_DERIVATIVES = True  # every derivative on fresh factors of its own root, which seed the next step's trials


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
    parser.add_argument("--ns", type=int, help="radial resolution of the optimization solves (default: P.RESOLUTION)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="seed beta: on-axis (WOUT betaxis) for COIL_CASE=ellipse5-beta7, else volume-average")
    parser.add_argument("--bootstrap", action="store_true",
                        help="reactor-like kinetic profiles and a self-consistent bootstrap current (P.BOOTSTRAP_MODEL)")
    parser.add_argument("--restart", type=Path, help="continue a finished run from its input.run, final coils and "
                        "WOUT (no seed calibration or coil refit); pass the run's --beta/--bootstrap")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"),
                        help="poloidal and toroidal modes of the optimization solves (default: P.RESOLUTION)")
    parser.add_argument("--max-iterations", type=int, help="VMEC iteration cap of every free-boundary solve "
                        "(default: the deck's NITER)")
    parser.add_argument("--polish-tolerance", type=float, default=ROOT_POLISH_TOLERANCE,
                        help="Newton root-polish target (the root residual floors near 1e-11 at 12x12 modes)")
    parser.add_argument("--factorization", choices=("structured", "dense"), default="structured",
                        help="derivative factors: O(ns) block-Thomas + Woodbury (default), or the dense LU")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils, args.wout = args.restart / "coils.json", args.restart / "wout.nc"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    return args


def fit_coils_to_plasma(coils, wout, inp):
    """Coils whose field with the plasma's own is tangent to a fixed-boundary finite-beta seed."""
    import jax
    import jax.numpy as jnp
    import numpy as np
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
        return weighted_rms(interface.weights, total_normal_field(interface, coil_field(new)))

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
                   method="L-BFGS-B", bounds=[(-5.0, 5.0)] * x0.size,
                   options=dict(maxiter=P.COIL_FIT_MAXITER, maxcor=20, ftol=1e-15, gtol=1e-12))
    fitted = coils_from_u(jnp.asarray(fit.x))
    print(f"[coil fit] {fit.nit} L-BFGS-B iterations ({fit.message}): (B_coils + B_plasma).n/|B| RMS "
          f"{before:.3e} -> {float(normal_field_rms(fitted)):.3e} on the fixed-boundary seed", flush=True)
    return fitted


def main(argv=None):
    args = parse_args(argv)
    P.RESOLUTION = (*(args.modes or P.RESOLUTION[:2]), args.ns or P.RESOLUTION[2])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    # Unlike the fixed arm, also on GPU: this process-wide placement overrides VMEX's CPU default for the
    # implicit (adjoint) solves, and the free-boundary solves and their derivatives should all run on args.device.
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
    inp = seed_input()
    redl = None
    if args.restart is not None:  # parse_args set --wout to the run's WOUT
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.bootstrap:
        inp, fixed, redl = bootstrap_input(inp, args.beta, args.device)
    else:  # also in vacuum: it sets PHIEDGE for B0
        inp, fixed = finite_beta_input(inp, args.beta, args.device)
    if args.wout is not None:
        seed = vj.state_from_wout(vj.read_wout(args.wout), inp=inp, ns=ns)
    else:
        seed = fixed.state
    if args.max_iterations:
        inp = replace(inp, niter_array=np.array([args.max_iterations]))
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
    inp.to_indata(out / "input.run")  # the deck as run: resolution, pressure and seed PHIEDGE

    # Reload the saved coils so a restart builds a bit-identical coordinate chart.
    coils = resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS)
    if args.restart is None:  # a restart keeps the run's own currents
        # At finite beta the seed's own edge R B_phi (below B0 R0 by its diamagnetism): with B.n = 0
        # alone, a net-current mismatch leaves a toroidal-field jump that breaks pressure balance.
        rbtor = abs(float(fixed.wout.rbtor)) if args.beta > 0 and args.wout is None else P.B0 * float(inp.rbc[inp.ntor, 0])
        coils = scale_coil_currents(coils, rbtor)
    if args.beta > 0 and args.wout is None and not args.no_coil_fit:  # at beta = 0 a uniform scale keeps B.n/|B|
        coils = fit_coils_to_plasma(coils, fixed.wout, replace(inp, lfreeb=False))
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    # --bootstrap: the spline values but the last, then CURTOR, follow PHIEDGE as design variables.
    current = np.r_[np.asarray(inp.ac_aux_f)[: P.CURRENT_KNOTS - 1], inp.curtor] if args.bootstrap else None
    scales = np.r_[[PHIEDGE_STEP] * P.FREE_PHIEDGE, [P.CURRENT_STEP] * (0 if current is None else current.size),
                   P.COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = coil_limits.CoilChart(coils0, current_dofs=(), scales=scales,
                                  phiedge=float(inp.phiedge) if P.FREE_PHIEDGE else None,
                                  plasma_current=current, plasma_current_spline=True)
    chart.check_input(inp)
    qs = target_residual()

    def boundary(state, runtime, grid):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)  # private: no public traced LCFS of a state
        rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
        rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
        zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=grid[0], ntheta=grid[1])

    aspect_lower, aspect_upper = P.ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN

    def clearance(state, runtime, x):
        return coil_limits.surface_distance(chart.coils_from_x(x), boundary(state, runtime, coil_limits.SURFACE_GRID))

    def qa(state, runtime):
        residuals = qs.residuals_state(state, runtime)
        return jnp.vdot(residuals, residuals)

    def loss(state, runtime, x):
        return 0.5 * qa(state, runtime)

    def aspect(state, runtime, x):
        return opt.aspect_ratio(state, runtime)

    seconds = {}

    def record(name, **data):
        if name == "proposal":
            seconds["trials"] = seconds.get("trials", 0) + 1
        if "seconds" in data:
            seconds[name] = seconds.get(name, 0.0) + float(data["seconds"])

    from jax import monitoring
    # Time actually spent tracing and compiling; the cache's "compile_time_saved" is not.
    monitoring.register_event_duration_secs_listener(lambda event, duration, **_: record(
        "compile", seconds=duration) if event.startswith("/jax/core/compile/") else None)

    mirror = (opt.mirror_ratio,) if P.MIRROR_LIMIT else ()
    ceiling = (max_abs_iota,) if P.IOTA_CEILING else ()
    problem = opt.FreeBoundaryProblem.from_loss(
        inp, loss, chart.x0, field_from_parameters=chart, plasma_from_parameters=chart.plasma_from_parameters,
        scales=chart.scales, names=chart.dof_names,
        quantities=(min_abs_iota, opt.major_radius, *mirror, *ceiling,
                    *([bootstrap_mismatch(inp, redl, args.device)] if redl is not None else [])),
        parameter_quantities=(clearance, aspect), restart_from=seed, root_residual_atol=ROOT_TOLERANCE, event=record,
        deadline=started + args.max_seconds,
        solver_options=dict(device=args.device, ftol=P.EQUILIBRIUM_FTOL, edge_force_tolerance=P.EDGE_FORCE_TOLERANCE,
                            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
                            adjoint_dense_max_dofs=ADJOINT_MAX_DOFS, adjoint_residual_rtol=ADJOINT_RESIDUAL_RTOL,
                            adjoint_factorization=args.factorization))
    # The coupled root residual has a resolution-dependent floor (~1e-11 at 12x12 modes): polish to the
    # tightest target the converged start reaches, so that later trials are held to it too.
    for tolerance in [args.polish_tolerance] + [v for v in ROOT_POLISH_LADDER if v > args.polish_tolerance]:
        try:
            problem.enable_root_polishing(tolerance=tolerance, max_steps=ROOT_POLISH_STEPS)
            break
        except vj.VmecError as error:  # transactional: the unpolished root stays usable
            if tolerance >= ROOT_POLISH_LADDER[-1]:
                raise
            print(f"[polish] {tolerance:.0e} not reached ({error}); trying a looser target", flush=True)
    print(f"[polish] Newton root-polish target {tolerance:.0e}", flush=True)
    problem.enable_matrix_free(**MATRIXFREE, refresh_horizon=LU_REFRESH_HORIZON, refresh_max_steps=args.steps,
                               dense_derivatives=DENSE_DERIVATIVES)

    coil_rows = coil_limits.constraint(chart.coils_from_x)
    for method in ("fun", "jac"):
        def timed(x, _call=getattr(coil_rows, method), _name=f"coil_constraint_{method}"):
            start = time.monotonic()
            value = _call(x)
            record(_name, seconds=time.monotonic() - start)
            return value
        setattr(coil_rows, method, timed)
    nredl = int(redl is not None)  # the Redl mismatch row sits between the plasma and coil quantities
    mirror_bounds = [(-np.inf, P.MIRROR_LIMIT - P.MIRROR_MARGIN, P.MIRROR_LIMIT)] if mirror else []
    lower, upper, row_scales = zip(
        (P.IOTA_FLOOR + P.IOTA_MARGIN, np.inf, P.IOTA_FLOOR),
        (P.RADIUS_TARGET - width, P.RADIUS_TARGET + width, P.RADIUS_TOLERANCE), *mirror_bounds,
        *([(-np.inf, P.IOTA_CEILING - P.IOTA_MARGIN, P.IOTA_FLOOR)] if ceiling else []),
        *[(-np.inf, P.REDL_TOLERANCE, P.REDL_TOLERANCE)] * nredl,
        (P.COIL_SURFACE_DISTANCE_LIMIT + P.DISTANCE_MARGIN, np.inf, P.COIL_SURFACE_DISTANCE_LIMIT),
        (aspect_lower, aspect_upper, aspect_scale))
    constraints = [problem.nonlinear_constraint(list(lower), list(upper), scales=list(row_scales)), coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        chart.coils_from_x(x).to_json(str(out / f"coils{tag}.json"))
        wout = problem.equilibrium_from_x(x).wout
        vj.write_wout(str(out / f"wout{tag}.nc"), wout)
        # Diagnostics only: none of these enter the loss or the constraints.
        row = dict(step=problem.accepted_step, phiedge=float(wout.phi[-1]), b0=float(wout.b0),
                   rbtor=abs(float(wout.rbtor)), betaxis=float(wout.betaxis))
        with open(out / "diagnostics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[diagnostics] PHIEDGE={row['phiedge']:.5f} Wb B0={row['b0']:.4f} T R B_phi={row['rbtor']:.4f} T m "
              f"betaxis={row['betaxis']:.4%}", flush=True)

    last = dict(time=time.monotonic(), x=problem.accepted.parameters.copy())
    def log_step():
        x, record = problem.accepted.parameters, problem.accepted
        values = list(map(float, problem.constraint_values(x)))
        iota, radius, surface, aspect_value = values[0], values[1], values[-2], values[-1]
        mismatch = values[2 + len(mirror) + len(ceiling):-2]
        now = time.monotonic()
        objective = problem.fun(x)
        row = dict(step=problem.accepted_step, qa=2 * objective, objective=objective, min_abs_iota=iota, major_radius_m=radius,
                   aspect=aspect_value, **({'mirror_ratio': values[2]} if mirror else {}),
                   **({'max_abs_iota': values[2 + len(mirror)]} if ceiling else {}), coil_surface_distance_m=surface,
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
        print(f"[step {row['step']}] {P.TARGET_NAME}={row['qa']:.6e} iota={iota:.5f} R={radius:.5f} aspect={aspect_value:.4f} "
              f"clearance={surface:.4f} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              + (f"{P.BOOTSTRAP_MODEL}={mismatch[0]:.2e} CURTOR={row['curtor']:.0f}A " if mismatch else "") +
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
