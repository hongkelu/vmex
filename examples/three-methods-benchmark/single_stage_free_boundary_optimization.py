#!/usr/bin/env python
"""L-BFGS-B/AugLag single stage, with the fast free-boundary implicit backend.

The optimizer, QA samples, plasma inequalities and coil regularization follow
single_stage_optimization.py. Only coils are independent variables here. The
normal-field coupling is supplied by the free-boundary equilibrium, so it is
reported rather than added to the augmented Lagrangian. B0 and R are diagnostics.
Start from fitted coils, not circular coils. See README.md for provenance.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import sys
import time

# Same constants as the fixed single-stage example. Keep them easy to edit.
IOTA_FLOOR, ASPECT_LIMIT, NORMAL_FIELD_LIMIT = 0.19, 5.0, 0.01
IOTA_CONSTRAINT, ASPECT_CONSTRAINT = 0.20, 5.0
LENGTH_TARGET, LENGTH_WEIGHT = 4.1, 1.0
CURVATURE_LIMIT, CURVATURE_OBJECTIVE_LIMIT, CURVATURE_WEIGHT = 7.0, 6.9, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.15, 1.0e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1.0e3
PENALTY_START, PENALTY_GROWTH, PENALTY_MAX = 10.0, 10.0, 1.0e6
CONSTRAINT_TOLERANCE = 1.0e-3
MAX_STAGES, STAGE_MAXITER, MAX_TRIALS = 8, 25, 300
PARAMETER_BOUND, COIL_STEP = 3.0, 0.05
NPHI, NTHETA = 37, 32
ADJOINT_BATCH_SIZE, ROOT_TOLERANCE = 32, 2.0e-6


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--coils", type=Path, default=here / "coils.stage2.json")
    parser.add_argument("--output", type=Path, default=here / "runs" / time.strftime("free-%Y%m%d-%H%M%S"))
    parser.add_argument("--max-stages", type=int, default=MAX_STAGES)
    parser.add_argument("--stage-maxiter", type=int, default=STAGE_MAXITER)
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--verify-ns", type=int, default=101)
    parser.add_argument("--verify-ftol", type=float, default=1.0e-12)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--movie", action="store_true", help="make a GIF after optimization")
    args = parser.parse_args()
    if min(args.max_stages, args.stage_maxiter, args.max_trials, args.wall_seconds) < 1:
        parser.error("iteration, evaluation and time budgets must be positive")
    if args.verify_ns < 3 or not 0 < args.verify_ftol < 1:
        parser.error("invalid verification resolution or tolerance")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        path = out / "cache" / key
        path.mkdir(parents=True)
        os.environ[key] = str(path)
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda" if args.device == "gpu" else "cpu",
                      VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false",
                      MPLBACKEND="Agg")
    # Always use the branch containing this example, including on a GPU server.
    repo = here.parents[1]
    sys.path.insert(0, str(repo))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im, statephysics as physics
    from vmex.core import freeboundary_continuation as fc, freeboundary_implicit as fbi
    from vmex.core.freeboundary import _solve_free_boundary_stage
    from vmex.core.freeboundary_problem import FreeBoundaryProblem
    from vmex.core.coil_parameters import CoilParameters
    from vmex.core.residuals import m1_constrained_to_physical
    from vmex.core.transforms import physical_to_internal_scale
    from vmex.core.errors import VmecError
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary

    started = time.perf_counter()
    if jax.default_backend() != args.device or not jax.config.x64_enabled:
        raise RuntimeError("requested device and float64 precision are required")
    def timeout(_signum, _frame):
        raise TimeoutError("run wall-time budget reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.wall_seconds)
    def write_json(name, data):
        (out / name).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    inp = vj.VmecInput.from_file(here / "input.rotating_ellipse")
    coils0 = Coils.from_json(str(args.coils.resolve()))
    if inp.lasym or coils0.nfp != inp.nfp:
        raise ValueError("this example requires a symmetric input and matching coil NFP")
    # Match the fixed example's ESSOS scaled curve coordinates, not the
    # projected optimizer's different mode-dependent step sizes.
    scales = COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()
    chart = CoilParameters.from_coils(coils0, current_dofs=(), scales=scales)
    n_coils = len(chart.currents)
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.05, 1.0, 6), 1, 0)
    write_json("provenance.json", dict(command=sys.argv, device=str(jax.devices()[0]),
        python=sys.version, jax=jax.__version__, vmex=vj.__file__,
        dependencies={name: version(name) for name in ("numpy", "scipy", "essos", "solvax", "equinox")},
        input_sha256=sha(here / "input.rotating_ellipse"), coils_sha256=sha(args.coils),
        script_sha256=sha(Path(__file__)), core_sha256={str(p.relative_to(repo)): sha(p)
            for p in sorted((repo / "vmex/core").glob("*.py"))}, settings=vars(args),
        currents_A=chart.currents.tolist(), stage_two_included_in_timing=False))
    print("Running single_stage_free_boundary_optimization.py", flush=True)
    print(f"Device: {jax.devices()[0]}; {chart.size} coil variables, fixed currents", flush=True)
    print(f"L-BFGS-B + AugLag; QA weight 1; min |iota| >= {IOTA_CONSTRAINT}, aspect <= {ASPECT_CONSTRAINT} during optimization")
    print("B0 and R are diagnostics. Normal-field RMS is checked after the free-boundary solve.")
    print(f"dof_names = {chart.dof_names}")
    print("[seed] solving shared fixed-boundary input...", flush=True)
    fixed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                  polish_force_balance=False)
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
    ftol, niter = float(inp.ftol_array[-1]), int(inp.niter_array[-1])
    solver = fbi.make_free_boundary_config(inp, chart(jnp.asarray(chart.x0)),
        field_from_parameters=chart, device=args.device, ftol=ftol, max_iterations=niter,
        include_edge_in_convergence=True, edge_force_tolerance=ftol,
        adjoint_solver="forward_dense_jax", adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
        adjoint_fail="error", adjoint_residual_rtol=1.0e-6)
    rt = im.runtime_from_params(im.params_from_input(inp), solver.implicit)
    print("[seed] solving and certifying free equilibrium...", flush=True)
    seed_stage = _solve_free_boundary_stage(inp, external_field=chart(jnp.asarray(chart.x0)),
        resolution=solver.resolution, ftol=ftol, max_iterations=niter, initial_state=fixed.state,
        include_edge_in_convergence=True, edge_force_tolerance=ftol, use_fft=False,
        error_on_no_convergence=True, jacobian_retries=0, allow_initial_axis_reguess=False)
    cfg = fc.make_free_boundary_continuation_config_from_state(solver, im.params_from_input(inp), chart.x0,
        state=seed_stage.result.state, rcon0=seed_stage.rcon0, zcon0=seed_stage.zcon0,
        parameter_scales=chart.scales, continuation_step=0.1, root_residual_atol=ROOT_TOLERANCE)
    # Certification does no iterations; retain the preceding solve's count.
    cfg = replace(cfg, _anchor=replace(cfg._anchor,
        result=replace(cfg._anchor.result, iterations=int(seed_stage.result.iterations))))
    # This driver uses explicit roots, never cfg's generic memoized root solver.
    # Only the SciPy callback promotes `accepted`; every prediction/retry below
    # is made from that root. cfg is the immutable numerical/ownership context.
    accepted = cfg._anchor
    initial = accepted
    mode_scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    mode_m, mode_n = np.asarray(rt.modes.m), np.asarray(rt.modes.n)

    def boundary(state):
        rc, zs, _, _ = m1_constrained_to_physical(state.R_cos, state.Z_sin,
            state.R_sin, state.Z_cos, modes=rt.modes, lthreed=bool(solver.resolution.lthreed),
            lasym=False, lconm1=True)
        rbc = jnp.zeros_like(jnp.asarray(inp.rbc)).at[mode_n + inp.ntor, mode_m].set(rc[-1] * mode_scale)
        zbs = jnp.zeros_like(jnp.asarray(inp.zbs)).at[mode_n + inp.ntor, mode_m].set(zs[-1] * mode_scale)
        return rbc, zbs

    def surface(state, nphi=NPHI, ntheta=NTHETA):
        return surfacerzfourier_from_boundary(*boundary(state), inp.nfp, nphi=nphi, ntheta=ntheta)

    def augmented_lagrangian(constraints, multipliers, penalty):
        shifted = jnp.maximum(multipliers - penalty * constraints, 0.0)
        return jnp.sum(shifted**2 - multipliers**2) / (2.0 * penalty)

    def lagrangian(state, x, multipliers, penalty):
        coils = chart.coils_from_x(x)
        surf = surface(state)
        rows = qs.residuals_state(state, rt)
        constraints = jnp.stack([opt.min_abs_iota(state, rt) / IOTA_CONSTRAINT - 1.0,
                                 1.0 - opt.aspect_ratio(state, rt) / ASPECT_CONSTRAINT])
        costs = jnp.stack([
            0.5 * jnp.vdot(rows, rows),
            0.5 * LENGTH_WEIGHT * jnp.sum((coils.length[:n_coils] - LENGTH_TARGET)**2),
            0.5 * CURVATURE_WEIGHT * jnp.sum(jnp.maximum(coils.curvature[:n_coils] - CURVATURE_OBJECTIVE_LIMIT, 0)**2),
            0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(coils, COIL_DISTANCE_LIMIT, block_size=32),
            0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(coils, surf, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
            augmented_lagrangian(constraints, multipliers, penalty)])
        return jnp.sum(costs), (costs, constraints)

    # Differentiate both paths: coils directly, AND their equilibrium response.
    # In particular coil-surface distance differentiates the moving LCFS too.
    local_value_grad = jax.jit(jax.value_and_grad(lagrangian, argnums=(0, 1), has_aux=True))
    term_names = ("quasisymmetry", "coil length", "coil curvature", "coil separation",
                  "coil-surface separation", "iota and aspect constraints")
    monitor = opt.OptimizationMonitor(stream=None)
    history, frames = [], {}
    accepted_linearization = candidate_linearization = None
    candidate = accepted
    multipliers, penalty = np.zeros(2), PENALTY_START
    counts = dict(trials=0, solves=1, failed_trials=0, accepted=0)
    last = {}
    cycle_started = time.perf_counter()
    cycle_gradient_seconds = 0.0
    cycle_solve_seconds = 0.0

    def metrics(record):
        state = record.state
        return {"QA total": float(qs.total_state(state, rt)), "aspect": float(opt.aspect_ratio(state, rt)),
            "mean iota": float(opt.mean_iota(state, rt)), "min |iota|": float(opt.min_abs_iota(state, rt)),
            "B0 [T]": abs(float(physics.on_axis_magnetic_field(state, rt))),
            "R [m]": float(physics.major_radius(state, rt)),
            **{k: float(getattr(record.result, k)) for k in ("fsqr", "fsqz", "fsql", "fedge")}}

    def record_step(value, terms, gradient_seconds=0.0, solve_seconds=0.0, step_seconds=0.0):
        row = dict(step=counts["accepted"], trials=counts["trials"], objective=float(value),
                   **metrics(accepted), gradient_seconds=gradient_seconds, solve_seconds=solve_seconds,
                   step_seconds=step_seconds, elapsed_seconds=time.perf_counter()-started,
                   equilibrium_iterations=int(accepted.result.iterations))
        history.append(row)
        with (out / "free_boundary_steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        monitor.record(accepted.parameters, cost=float(value), iteration=counts["accepted"],
            equilibrium_solves=counts["solves"], rejected_trials=counts["failed_trials"], terms=terms)
        if args.movie:
            frames[accepted.parameters.tobytes()] = tuple(np.asarray(a) for a in boundary(accepted.state))
        # Lossless accepted-state evidence; no rejected state is saved as accepted.
        np.savez_compressed(out / f"accepted_{counts['accepted']:04d}.npz", parameters=accepted.parameters,
            rcon0=accepted.rcon0, zcon0=accepted.zcon0, multipliers=multipliers, penalty=penalty,
            **{k: np.asarray(getattr(accepted.state, k)) for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
        print(f"[step {row['step']:4d}] QA total = {row['QA total']:.6e}, objective = {value:.6e}, "
              f"aspect = {row['aspect']:.4f}, mean iota = {row['mean iota']:.4f}, min |iota| = {row['min |iota|']:.4f}, "
              f"B0 = {row['B0 [T]']:.6f} T, R = {row['R [m]']:.6f} m\n"
              f"            forces = {row['fsqr']:.2e}/{row['fsqz']:.2e}/{row['fsql']:.2e}, edge = {row['fedge']:.2e}; "
              f"gradient {gradient_seconds:.2f} s, solves {solve_seconds:.2f} s, complete step {step_seconds:.2f} s", flush=True)

    class EvaluationBudget(Exception):
        pass

    def value_and_grad(u):
        nonlocal candidate, candidate_linearization, accepted_linearization
        nonlocal cycle_gradient_seconds, cycle_solve_seconds
        x = scales * np.asarray(u)
        key = (x.tobytes(), multipliers.tobytes(), float(penalty))
        if last.get("key") == key:
            return last["value"], last["gradient"].copy()
        if counts["trials"] >= args.max_trials:
            raise EvaluationBudget()
        counts["trials"] += 1
        print(f"[trial {counts['trials']}] equilibrium/prediction...", flush=True)
        t0 = time.perf_counter()
        if np.array_equal(x, accepted.parameters):
            candidate = accepted
        else:
            delta = x - accepted.parameters
            if accepted_linearization is None:
                raise RuntimeError("evaluate the accepted root before proposing a trial")
            tangent = accepted_linearization.tangent(accepted, cfg, jnp.asarray(delta))
            points = max(1, int(np.ceil(np.max(np.abs(delta / scales)) / 0.1)))
            if points > 64:
                raise RuntimeError("trial exceeds the bounded prediction path")
            try:
                for index in range(1, points + 1):
                    fraction = index / points
                    point = x if index == points else accepted.parameters + fraction * delta
                    predicted = jax.tree.map(lambda a, b: a + fraction*b, accepted.state, tangent)
                    counts["solves"] += 1
                    print(f"  correction {index}/{points} from accepted step {counts['accepted']}", flush=True)
                    stage = _solve_free_boundary_stage(inp, external_field=chart(jnp.asarray(point)),
                        resolution=solver.resolution, ftol=ftol, max_iterations=niter,
                        initial_state=predicted, constraint_continuation=(accepted.rcon0, accepted.zcon0),
                        include_edge_in_convergence=True, edge_force_tolerance=ftol, use_fft=False,
                        error_on_no_convergence=True, jacobian_retries=0, allow_initial_axis_reguess=False)
                    candidate = fc.certify_free_boundary_continuation_state(cfg, point, stage.result.state,
                        rcon0=stage.rcon0, zcon0=stage.zcon0, result=stage.result)
                    print(f"  {stage.result.iterations} iterations; edge force {float(stage.result.fedge):.2e}", flush=True)
            except VmecError as exc:
                counts["failed_trials"] += 1
                print(f"  numerical rejection: {exc}", flush=True)
                # L-BFGS-B may backtrack; the callback refuses nonfinite trials.
                cycle_solve_seconds += time.perf_counter()-t0
                return np.inf, np.zeros_like(u)
        solve_seconds = time.perf_counter()-t0
        # Preserve the last finite trial's factors across numerical rejections;
        # SciPy may return that trial. Release them only when replacing it.
        if candidate_linearization is not None and candidate_linearization is not accepted_linearization:
            candidate_linearization.close()
            candidate_linearization = None
        print(f"[trial {counts['trials']}] implicit adjoint...", flush=True)
        t1 = time.perf_counter()
        (value, (costs, constraints)), (state_bar, direct_gradient) = local_value_grad(
            candidate.state, jnp.asarray(x), jnp.asarray(multipliers), jnp.asarray(penalty))
        rhs = jax.tree.map(lambda a: a[None], state_bar)
        linearization = fc.free_boundary_continuation_state_pullback(candidate, cfg, rhs, return_linearization=True)
        gradient = (np.asarray(linearization.field_jacobian)[0] + np.asarray(direct_gradient)) * scales
        if not np.isfinite(float(value)) or not np.all(np.isfinite(gradient)):
            linearization.close()
            raise FloatingPointError("nonfinite certified objective/gradient")
        candidate_linearization = linearization
        if candidate is accepted:
            if accepted_linearization is not None:
                accepted_linearization.close()
            accepted_linearization = linearization
        gradient_seconds = time.perf_counter()-t1
        cycle_gradient_seconds += gradient_seconds
        cycle_solve_seconds += solve_seconds
        last.update(key=key, value=float(value), gradient=gradient.copy(), root=candidate,
                    terms=dict(zip(term_names, map(float, np.asarray(costs)))),
                    updated=np.maximum(multipliers - penalty*np.asarray(constraints), 0.0))
        print(f"  objective {float(value):.6e}, QA total {2*float(costs[0]):.6e}; "
              f"solve/predict {solve_seconds:.2f} s, gradient {gradient_seconds:.2f} s", flush=True)
        return float(value), gradient

    def callback(u):
        nonlocal accepted, accepted_linearization, cycle_started, cycle_gradient_seconds, cycle_solve_seconds
        if last.get("key", (None,))[0] != (scales*np.asarray(u)).tobytes():
            raise RuntimeError("SciPy accepted a trial without a finite certified evaluation")
        if np.array_equal(last["root"].parameters, accepted.parameters):
            return
        if accepted_linearization is not None:
            accepted_linearization.close()
        accepted = last["root"]
        accepted_linearization = candidate_linearization
        counts["accepted"] += 1
        record_step(last["value"], last["terms"], cycle_gradient_seconds, cycle_solve_seconds,
                    time.perf_counter()-cycle_started)
        cycle_started = time.perf_counter()
        cycle_gradient_seconds = cycle_solve_seconds = 0.0

    status, stages, result, failure = "stage_budget_reached", 0, None, None
    initial_value = final_value = None
    try:
        initial_value, _ = value_and_grad(np.zeros(chart.size))
        record_step(initial_value, last["terms"], cycle_gradient_seconds, cycle_solve_seconds)
        # Keep the first gradient in the first complete-step timer as well.
        previous_violation = np.inf
        for stage_index in range(args.max_stages):
            result = minimize(value_and_grad, accepted.parameters/scales, jac=True, method="L-BFGS-B",
                bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * chart.size, callback=callback,
                options={"maxiter": args.stage_maxiter, "maxfun": args.max_trials-counts["trials"],
                         "maxcor": 20, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
            value_and_grad(accepted.parameters/scales)
            final_value = last["value"]
            updated = last["updated"]
            violation = float(np.max(np.abs(updated-multipliers))) / penalty
            stages += 1
            print(f"[stage {stage_index+1}] {result.nit} L-BFGS-B iterations, {counts['trials']} trials, "
                  f"violation = {violation:.3e}, multipliers = {updated}, penalty = {penalty:.1e}", flush=True)
            if violation <= CONSTRAINT_TOLERANCE and result.status == 0:
                status = "converged"
                break
            if result.status not in (0, 1):
                status = "optimizer_failed"
                break
            multipliers = updated
            if violation > 0.25 * previous_violation:
                penalty = min(PENALTY_GROWTH*penalty, PENALTY_MAX)
            previous_violation = violation
    except EvaluationBudget:
        status = "evaluation_budget_reached"
    except Exception as exc:
        status, failure = "failed", f"{type(exc).__name__}: {exc}"
        print(f"Optimization stopped: {failure}", flush=True)
    finally:
        for linearization in (candidate_linearization, accepted_linearization):
            if linearization is not None:
                linearization.close()
        signal.alarm(0)

    optimization_seconds = time.perf_counter()-started
    summary = dict(example=Path(__file__).name, status=status, failure=failure, stages=stages, **counts,
        optimization_seconds=optimization_seconds, seed=metrics(initial), final=metrics(accepted),
        optimizer_success=status == "converged", derivative_qualified=False,
        targets={"min |iota| >=": IOTA_FLOOR, "aspect <=": ASPECT_LIMIT},
        diagnostics_only=["B0", "major radius"], verification="not run")
    write_json("single_stage_free_boundary_optimization_summary.json", summary)
    chart.coils_from_x(jnp.asarray(accepted.parameters)).to_json(str(out / "coils_single_stage_free_boundary_optimized.json"))
    monitor.save(out / "single_stage_free_boundary_objectives.csv")
    if failure:
        raise RuntimeError(failure)

    # All WOUT construction, independent verification, VTK and plots happen
    # after the timed optimizer, never in value_and_grad or its callback.
    signal.alarm(args.wall_seconds)
    print("Saving accepted equilibrium and running independent verification...", flush=True)
    def export(record, name):
        export_cfg = replace(cfg, _anchor=record, parameter_anchor=record.parameters)
        problem = FreeBoundaryProblem.from_tuples(inp, [(qs.residuals_state, 0.0, 1.0)],
            parameterization=chart, continuation=export_cfg, objective_normalization=1.0)
        try:
            wout = problem.equilibrium_from_x(record.parameters).wout
            return vj.write_wout(out / name, wout)
        finally:
            problem.close()
    initial_wout = export(initial, "wout_single_stage_free_boundary_initial.nc")
    accepted_wout = export(accepted, "wout_single_stage_free_boundary_accepted.nc")
    rbc, zbs = boundary(accepted.state)
    final_input = replace(inp, rbc=np.asarray(rbc), zbs=np.asarray(zbs),
        ns_array=np.array([args.verify_ns]), ftol_array=np.array([args.verify_ftol]), niter_array=np.array([8000]))
    final_input.to_indata(out / "input.single_stage_free_boundary_optimized")
    final_coils = chart.coils_from_x(jnp.asarray(accepted.parameters))
    verification = FreeBoundaryProblem.from_tuples(final_input, [(qs.residuals_state, 0.0, 1.0)],
        coils=final_coils, current_dofs=(), restart_from=accepted_wout, objective_normalization=1.0,
        solver_options=dict(device=args.device, ftol=args.verify_ftol, max_iterations=8000,
            edge_force_tolerance=args.verify_ftol, adjoint_dense_batch_size=ADJOINT_BATCH_SIZE))
    try:
        verified = verification.equilibrium_from_x(verification.x0)
        wout_path = vj.write_wout(out / "wout_single_stage_free_boundary_optimized.nc", verified.wout)
        reporter = opt.EquilibriumReporter(("QA total", qs.total, ".6e"),
            ("aspect", opt.aspect_ratio, ".4f"), ("mean iota", opt.mean_iota, ".4f"),
            ("min |iota|", opt.min_abs_iota, ".4f"),
            ("B0 [T]", physics.on_axis_magnetic_field, ".6f"), ("R [m]", physics.major_radius, ".6f"))
        final_values = reporter("final verification", verified)
    finally:
        verification.close()
    from essos.surfaces import SurfaceRZFourier
    surface_initial = SurfaceRZFourier.from_wout_file(initial_wout, nphi=60, ntheta=60)
    surface_final = SurfaceRZFourier.from_wout_file(wout_path, nphi=61, ntheta=64)
    magnetic_field = np.asarray(jax.vmap(BiotSavart(final_coils).B)(surface_final.gamma.reshape(-1, 3))).reshape(surface_final.gamma.shape)
    bn = np.sum(magnetic_field*np.asarray(surface_final.unitnormal), axis=2) / np.linalg.norm(magnetic_field, axis=2)
    weights = np.asarray(surface_final.area_element); weights = weights / weights.sum()
    rms, maximum = float(np.sqrt(np.sum(weights*bn**2))), float(np.max(np.abs(bn)))
    points = np.asarray(final_coils.gamma)
    surface_points = np.asarray(surface_final.gamma).reshape(-1, 3)
    coil_surface_distance = min(float(np.linalg.norm(p[:, None]-surface_points[None], axis=2).min()) for p in points)
    coil_distance = min(float(np.linalg.norm(points[i][:, None]-points[j][None], axis=2).min())
                        for i in range(len(points)) for j in range(i+1, len(points)))
    curvature = float(np.max(np.asarray(final_coils.curvature)))
    checks = [("min |iota|", final_values["min |iota|"], IOTA_FLOOR, "below"),
        ("aspect", final_values["aspect"], ASPECT_LIMIT, "above"), ("B.n/B RMS", rms, NORMAL_FIELD_LIMIT, "above"),
        ("coil-surface distance", coil_surface_distance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
        ("coil-coil distance", coil_distance, COIL_DISTANCE_LIMIT, "below"),
        ("maximum curvature", curvature, CURVATURE_LIMIT, "above")]
    unmet = [f"{name} {value:.6g} {side} {limit}" for name, value, limit, side in checks
             if not np.isfinite(value) or (value < limit if side == "below" else value > limit)]
    print(f"\nObjective initially {initial_value:.6e}; {counts['accepted']} accepted steps, {counts['trials']} trials; {status}")
    print(f"Coil lengths = {np.asarray(final_coils.length[:n_coils])}")
    print(f"B.n/B: area-weighted RMS = {100*rms:.3f}%, max = {100*maximum:.3f}%")
    print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m; coil-coil = {coil_distance:.4f} m")
    print(f"Maximum curvature = {curvature:.4f} 1/m")
    print("This run did NOT meet its stated targets: " + "; ".join(unmet) if unmet else "All stated targets met.")
    summary.update(verification="passed numerical checks", verified=dict(final_values, **{"B.n/B RMS": rms,
        "B.n/B max": maximum, "coil-surface distance": coil_surface_distance,
        "coil-coil distance": coil_distance, "maximum curvature": curvature}), unmet=unmet, met=not unmet)
    write_json("single_stage_free_boundary_optimization_summary.json", summary)
    for label, surf, coils in (("initial", surface_initial, coils0), ("optimized", surface_final, final_coils)):
        surf.to_vtk(str(out / f"surface_single_stage_free_boundary_{label}"), field=BiotSavart(coils))
        coils.to_vtk(str(out / f"coils_single_stage_free_boundary_{label}"))
    if not args.no_plots:
        vj.plot_optimization_objects(out / "single_stage_free_boundary_optimization.png",
            ("Initial", surface_initial, coils0), ("Optimized", surface_final, final_coils))
        monitor.plot(out / "single_stage_free_boundary_objectives.png", title="Free-boundary objective terms")
        for path in vj.plot_wout(wout_path, out).values():
            print(f"Wrote {path}")
    if args.movie:
        def objects(x):
            return surfacerzfourier_from_boundary(*frames[x.tobytes()], inp.nfp, nphi=NPHI, ntheta=NTHETA), chart.coils_from_x(jnp.asarray(x))
        def colors(_x, objects):
            surf, coils = objects
            return np.linalg.norm(np.asarray(jax.vmap(BiotSavart(coils).B)(surf.gamma.reshape(-1, 3))), axis=1).reshape(surf.gamma.shape[:2])
        monitor.movie_surface_coils(out / "single_stage_free_boundary_optimization.gif", objects,
            x0=np.zeros(chart.size), scales=np.ones(chart.size), surface_color=colors)
    signal.alarm(0)
    print(f"Wrote results to {out}", flush=True)
    return 1 if unmet or not summary["optimizer_success"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
