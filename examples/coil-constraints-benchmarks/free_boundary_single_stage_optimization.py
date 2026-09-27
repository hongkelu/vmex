#!/usr/bin/env python
"""Free-boundary single-stage coil optimization with hard coil constraints.

Only the coils vary. Each trial solves the vacuum free-boundary equilibrium of
those coils, predicted from the accepted root and certified before use, and
SLSQP minimizes quasisymmetry subject to hard inequalities: minimum |iota|,
major radius, aspect ratio, coil-to-plasma clearance, and per-coil length,
curvature, mean squared curvature and coil separation (``parameters.py``).
``OPTIMIZER = "L-BFGS-B"`` instead minimizes the fixed-boundary arm's objective,
QA plus the same rows as hinge penalties, so both arms share one algorithm.

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free

Every accepted step appends one line to ``metrics.jsonl``. Coils and a WOUT
are saved every ``--save-every`` steps and at the end; restart from them with
``--coils <out>/coils.json --wout <out>/wout.nc``. This case is vacuum only.
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
OPTIMIZER_FTOL = 1e-10
ADJOINT_RESIDUAL_RTOL, ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 1e-9, 32, 20000
MATRIXFREE = dict(rtol=1e-11, restart=100, max_restarts=3, rhs_batch_size=3)
LU_REFRESH_HORIZON = 10
NEWTON_STEPS = None  # Newton-correct predicted trials on the seed LU before any ordinary solve
IOTA_CONSTRAINT = "min"  # "min": one minimum-|iota| row; "surfaces": one row per radial surface
IOTA_RAMP = None  # e.g. 0.05: raise the iota floor in stages of RAMP_STAGE_STEPS accepted steps
RAMP_STAGE_STEPS = 5
OPTIMIZER = "SLSQP"  # "L-BFGS-B": the hinge penalties of single_stage_optimization.py instead
PENALTY_WEIGHT = 1.0e3  # its CONSTRAINT_WEIGHT
STEP_CAP = None  # e.g. 0.2: SLSQP box around the stage start, in scaled coordinates, per RAMP_STAGE_STEPS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="accepted SLSQP steps")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--coils", type=Path, default=HERE / "coils.initial.json")
    parser.add_argument("--wout", type=Path, help="restart the initial solve from this WOUT")
    parser.add_argument("--max-seconds", type=float, default=float("inf"), help="optimization wall-time budget")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args(argv)


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
    inp = vj.VmecInput.from_file(HERE / "input.rotating_ellipse")
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=P.GRID[0], nzeta=P.GRID[1])
    inp = replace(inp, ns_array=np.array([ns]), ftol_array=np.array([P.EQUILIBRIUM_FTOL]), lfreeb=False)
    if args.wout is not None:
        seed = vj.state_from_wout(vj.read_wout(args.wout), inp=inp, ns=ns)
    else:
        seed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                     polish_force_balance=False).state
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")

    # Reload the saved coils so a restart builds a bit-identical coordinate chart.
    resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS).to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    scales = P.COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()
    chart = opt.CoilParameters.from_coils(coils0, current_dofs=(), scales=scales)
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
        if OPTIMIZER == "SLSQP":
            return 0.5 * qa(state, runtime)
        # The fixed-boundary arm's rows; its normal-field and flux rows hold here by construction.
        aspect_value, radius = opt.aspect_ratio(state, runtime), opt.major_radius(state, runtime)
        rows = jnp.concatenate([
            jnp.stack([(opt.min_abs_iota(state, runtime) - P.IOTA_FLOOR - P.IOTA_MARGIN) / P.IOTA_FLOOR,
                       (aspect_value - aspect_lower) / aspect_scale, (aspect_upper - aspect_value) / aspect_scale,
                       (radius - P.RADIUS_TARGET + width) / P.RADIUS_TOLERANCE,
                       (P.RADIUS_TARGET + width - radius) / P.RADIUS_TOLERANCE]),
            coil_limits.coil_inequalities(coils),
            jnp.atleast_1d((clearance(state, runtime, coils) - P.COIL_SURFACE_DISTANCE_LIMIT - P.DISTANCE_MARGIN)
                           / P.COIL_SURFACE_DISTANCE_LIMIT)])
        return 0.5 * qa(state, runtime) + 0.5 * PENALTY_WEIGHT * jnp.sum(jnp.maximum(-rows, 0.0)**2)

    def aspect(state, runtime, coils):
        return opt.aspect_ratio(state, runtime)

    def iota_rows(state, runtime):
        if IOTA_CONSTRAINT == "surfaces":  # the same half-mesh surfaces as min_abs_iota
            from vmex.core.statephysics import _iotas_half
            return jnp.abs(_iotas_half(state, runtime)[1:])
        return opt.min_abs_iota(state, runtime)

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

    problem = opt.FreeBoundaryProblem.from_loss(
        # L-BFGS-B differentiates only the loss; SLSQP also needs the constraint rows.
        inp, loss, quantities=(iota_rows, opt.major_radius) if OPTIMIZER == "SLSQP" else (),
        coil_quantities=(clearance, aspect) if OPTIMIZER == "SLSQP" else (),
        parameterization=chart, restart_from=seed, root_residual_atol=ROOT_TOLERANCE, event=record,
        deadline=started + args.max_seconds,
        solver_options=dict(device=args.device, ftol=P.EQUILIBRIUM_FTOL, edge_force_tolerance=P.EQUILIBRIUM_FTOL,
                            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
                            adjoint_dense_max_dofs=ADJOINT_MAX_DOFS, adjoint_residual_rtol=ADJOINT_RESIDUAL_RTOL))
    problem.enable_root_polishing(tolerance=ROOT_POLISH_TOLERANCE)
    problem.enable_matrix_free(**MATRIXFREE, refresh_horizon=LU_REFRESH_HORIZON, refresh_max_steps=args.steps)
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
    observe = jax.jit(lambda state, x: jnp.stack([
        jnp.min(jnp.atleast_1d(iota_rows(state, problem.rt))), opt.major_radius(state, problem.rt),
        clearance(state, problem.rt, chart.coils_from_x(x)), opt.aspect_ratio(state, problem.rt)]))
    n_iota = problem.constraint_values(problem.accepted.parameters).size - 3 if OPTIMIZER == "SLSQP" else 0

    def constraints(floor):
        return [problem.nonlinear_constraint(
            [floor + P.IOTA_MARGIN] * n_iota + [P.RADIUS_TARGET - width, P.COIL_SURFACE_DISTANCE_LIMIT + P.DISTANCE_MARGIN,
                                                aspect_lower],
            [np.inf] * n_iota + [P.RADIUS_TARGET + width, np.inf, aspect_upper],
            scales=[P.IOTA_FLOOR] * n_iota + [P.RADIUS_TOLERANCE, P.COIL_SURFACE_DISTANCE_LIMIT, aspect_scale]),
            coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        problem.coils_from_x(x).to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), problem.equilibrium_from_x(x).wout)

    last = dict(time=time.monotonic(), x=problem.accepted.parameters.copy())
    qa_of = jax.jit(lambda state: qa(state, problem.rt))

    def log_step():
        x, record = problem.accepted.parameters, problem.accepted
        iota, radius, surface, aspect_value = map(float, observe(record.state, jnp.asarray(x)))
        now = time.monotonic()
        row = dict(step=problem.accepted_step, qa=float(qa_of(record.state)), objective=problem.fun(x), min_abs_iota=iota, major_radius_m=radius,
                   aspect=aspect_value, coil_surface_distance_m=surface,
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
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
              f"{row['step_seconds']:.1f}s", flush=True)
        if row["step"] and row["step"] % args.save_every == 0:
            save(f".step{row['step']}")

    log_step()
    floors = [P.IOTA_FLOOR]
    if IOTA_RAMP:
        start_iota = float(observe(problem.accepted.state, jnp.asarray(problem.accepted.parameters))[0])
        floors = [*np.arange(start_iota + IOTA_RAMP, P.IOTA_FLOOR, IOTA_RAMP), P.IOTA_FLOOR]
    from scipy.optimize import Bounds
    try:
        index = 0
        while problem.accepted_step < args.steps:
            floor = floors[min(index, len(floors) - 1)]
            staged = index < len(floors) - 1 or bool(STEP_CAP)
            remaining = args.steps - problem.accepted_step
            budget = min(RAMP_STAGE_STEPS, remaining) if staged else remaining
            x, before = problem.accepted.parameters, problem.accepted_step
            print(f"[stage {index}] {OPTIMIZER}, iota floor {floor:.4f}, up to {budget} accepted steps", flush=True)
            if OPTIMIZER == "L-BFGS-B":
                # A rejected line-search trial ends the call; restart from the accepted state.
                result = opt.minimize(problem, x0=x, method="L-BFGS-B", callback=lambda x: log_step(),
                                      options=dict(maxiter=budget, maxcor=20, maxls=20, ftol=1e-12, gtol=1e-8))
            else:
                box = None if not STEP_CAP else Bounds(x - STEP_CAP * problem.scales, x + STEP_CAP * problem.scales)
                result = opt.minimize(problem, x0=x, method="SLSQP", bounds=box,
                                      constraints=constraints(floor), callback=lambda x: log_step(),
                                      options=dict(maxiter=budget, ftol=OPTIMIZER_FTOL))
            index += 1
            restart = staged or result.stop_reason == "equilibrium_trial_rejected"
            if (problem.accepted_step == before and index >= len(floors)) or not restart:
                break
        stop = dict(success=bool(result.success), message=str(result.message),
                    stop_reason=getattr(result, "stop_reason", None))
    except TimeoutError:
        stop = dict(success=False, message="wall-time budget reached", stop_reason="walltime")
    save("")
    summary = dict(accepted_steps=problem.accepted_step, **stop, solver=problem.solver_info,
                   failed_trials=problem.metadata["holder"]["failed_trials"],
                   elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str))
    problem.close()


if __name__ == "__main__":
    main()
