#!/usr/bin/env python
"""Scalar free-boundary plasma/coil optimization without AugLag.

Match the weighted objective in single_stage_optimization_scalar.py, including
its two normal-field terms. Use our implicit scalar adjoint and retained state
prediction. The input rotating ellipse seeds the equilibrium; fitted coils
then determine the free boundary. The default coils are the saved stage-two
fit used by the fixed scalar run completed on 2026-09-18. Radius and B0 are reported, not constrained.
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def gradient_check_passes(checks, objective_rtol, fine_only, constrained):
    """Keep coarse evidence; optionally gate the objective at the finest h."""
    selected = [min(checks, key=lambda row: row["h"])] if fine_only else checks
    objective_ok = (all(math.isfinite(row["relative_error"]) for row in checks)
                    and all(row["relative_error"] < objective_rtol for row in selected))
    constraints_ok = not constrained or all(math.isfinite(error) and error < 1e-3
        for row in checks for error in row["constraint_relative_errors"])
    return objective_ok, constraints_ok


def run(script_path, settings, plasma_costs, coil_costs):
    import _scalar_constraints as limits
    here = script_path.parent
    parser = argparse.ArgumentParser(description="Matched free-boundary scalar optimization; see the Settings in the example.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=settings.device,
                        help="execution device (default: settings.device in the example)")
    parser.add_argument("--coils", type=Path, default=here / "coils_single_stage_scalar_fitted_1789769333906005000.json",
                        help="stage-two coils saved by the fixed scalar run; currents remain fixed")
    parser.add_argument("--output", type=Path, default=here / "runs" / time.strftime("free-scalar-%Y%m%d-%H%M%S"))
    parser.add_argument("--resume-from", type=Path, help="completed free run whose accepted endpoint seeds this segment")
    parser.add_argument("--resume-manifest", type=Path, help="SHA256 download manifest authenticating the prior run")
    parser.add_argument("--maxiter", type=int, default=settings.maxiter)
    parser.add_argument("--method", choices=("BFGS", "L-BFGS-B"), default=settings.method)
    parser.add_argument("--max-trials", type=int, default=settings.max_trials)
    parser.add_argument("--accepted-steps", type=int, help="optional accepted-step budget")
    parser.add_argument("--adjoint", choices=("dense", "matrixfree"), default="dense",
                        help="matrixfree retains a dense seed LU as a preconditioner")
    parser.add_argument("--adjoint-max-dofs", type=int, default=4096,
                        help="explicit memory bound for the dense seed")
    parser.add_argument("--adjoint-batch-size", type=int, default=settings.adjoint_batch_size,
                        help="columns per batch when assembling the initial dense seed")
    parser.add_argument("--resolution", type=int, nargs=3, metavar=("M", "N", "NS"))
    parser.add_argument("--grid", type=int, nargs=2, metavar=("NTHETA", "NPHI"))
    parser.add_argument("--no-boundary-error", action="store_true",
                        help="remove both normal-field objective terms; keep diagnostics")
    parser.add_argument("--fd-ftol", type=float,
                        help="validation-only force/edge tolerance; defaults to --ftol")
    parser.add_argument("--check-gradient", action=argparse.BooleanOptionalAction, default=True,
                        help="check the scalar adjoint with independently corrected centered differences")
    parser.add_argument("--gradient-rtol", type=float, default=1e-3,
                        help="objective directional-check relative tolerance; constraints retain 1e-3")
    parser.add_argument("--gradient-fine-only", action="store_true",
                        help="gate the objective at the smallest h; retain both checks as evidence")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--optimization-seconds", type=int,
                        help="optional separate optimization budget, starting after qualification")
    parser.add_argument("--verification-seconds", type=int, default=1800)
    parser.add_argument("--initial-ftol", type=float,
                        help="fixed-boundary initialization tolerance; otherwise use the input deck")
    parser.add_argument("--ftol", type=float, help="override the equilibrium force tolerance")
    parser.add_argument("--verify-ns", type=int, default=101)
    parser.add_argument("--verify-maxiter", type=int, default=8000)
    parser.add_argument("--verify-ftol", type=float, default=1.0e-14)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--constrained", action="store_true", help="SLSQP with shared iota and physical-radius inequalities")
    args = parser.parse_args()
    settings = replace(settings, adjoint_batch_size=args.adjoint_batch_size)
    if args.constrained:
        args.method = "SLSQP"
    if args.ftol is None:
        args.ftol = limits.FORCE_TOLERANCE if args.constrained else settings.equilibrium_ftol
    if args.adjoint == "matrixfree" and not args.check_gradient:
        parser.error("matrixfree requires starting-point derivative qualification")
    if args.accepted_steps is not None and args.accepted_steps < 1:
        parser.error("--accepted-steps must be positive")
    if min(args.adjoint_max_dofs, args.adjoint_batch_size, args.verification_seconds) < 1:
        parser.error("adjoint sizes and verification budget must be positive")
    if args.optimization_seconds is not None and args.optimization_seconds < 1:
        parser.error("--optimization-seconds must be positive")
    if args.initial_ftol is not None and not 0 < args.initial_ftol < 1:
        parser.error("--initial-ftol must be between zero and one")
    if args.resolution and (args.resolution[0]<1 or args.resolution[1]<0 or args.resolution[2]<3):
        parser.error("invalid M/N/NS resolution")
    if args.grid and min(args.grid)<3:
        parser.error("angular grid sizes must be at least 3")
    if args.fd_ftol is not None and not 0 < args.fd_ftol < 1:
        parser.error("--fd-ftol must be between zero and one")
    if args.resume_from and (args.resolution or args.grid):
        parser.error("resolution overrides cannot be combined with exact-state resume")
    if min(args.maxiter, args.max_trials, args.wall_seconds) < 1:
        parser.error("iteration, evaluation and time budgets must be positive")
    if args.ftol is not None and not 0 < args.ftol < 1:
        parser.error("--ftol must be between zero and one")
    if args.verify_ns < 3 or args.verify_maxiter < 1 or not 0 < args.verify_ftol < 1:
        parser.error("invalid verification resolution or tolerance")
    if not 0 < args.gradient_rtol < 1:
        parser.error("--gradient-rtol must be between zero and one")
    if bool(args.resume_from) != bool(args.resume_manifest):
        parser.error("--resume-from and --resume-manifest must be used together")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    (out / script_path.name).write_text(script_path.read_text())
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    (out / "_scalar_gradient_check.py").write_text((here / "_scalar_gradient_check.py").read_text())
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        path = out / "cache" / key
        path.mkdir(parents=True)
        os.environ[key] = str(path)
    # NESTOR uses an explicit CPU lane; the plasma backend must still be GPU.
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda,cpu" if args.device == "gpu" else "cpu",
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
    from vmex.core.solver import SpectralState
    from _scalar_resume import load_endpoint
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos import objective_functions  # noqa: F401 -- preflight optional dependencies
    import pyevtk  # noqa: F401 -- required by final VTK export
    from essos.surfaces import surfacerzfourier_from_boundary
    from _scalar_diagnostics import StepHistory, check_control_settings

    restored, continuation = None, None
    if args.resume_from:
        restored, continuation = load_endpoint(args.resume_from, args.resume_manifest, repo, asdict(settings), args, script_path)
        (out / "_scalar_resume.py").write_text((here / "_scalar_resume.py").read_text())
    step_offset = continuation["step_offset"] if continuation else 0
    matched_settings = check_control_settings(settings, here / "single_stage_optimization_scalar.py")
    (out / "_scalar_diagnostics.py").write_text((here / "_scalar_diagnostics.py").read_text())
    (out / "_scalar_constraints.py").write_text((here / "_scalar_constraints.py").read_text())

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
    def phase(name, budget):
        write_json("phase.json", dict(name=name, started_monotonic=time.monotonic(), budget_seconds=budget))
        signal.alarm(budget)
    phase("initialization_and_qualification", args.wall_seconds)

    inp = vj.VmecInput.from_file(here / "input.rotating_ellipse")
    if args.resolution or args.grid:
        m, n, ns = args.resolution or (inp.mpol, inp.ntor, int(inp.ns_array[-1]))
        ntheta, nphi = args.grid or (inp.ntheta, inp.nzeta)
        inp = inp.change_resolution(mpol=m, ntor=n, ntheta=ntheta, nzeta=nphi)
        inp = replace(inp, ns_array=np.array([ns]))
    if args.initial_ftol is not None:
        inp = replace(inp, ftol_array=np.array([args.initial_ftol]))
    if inp.lasym:
        raise ValueError("this example requires stellarator symmetry")
    n_coils = settings.n_coils

    # Reuse the exact fitted geometry and currents; do not recalibrate or refit.
    coils0 = Coils.from_json(str(args.coils.resolve()))
    if coils0.nfp != inp.nfp or not coils0.stellsym:
        raise ValueError("fitted coils must match NFP and stellarator symmetry")
    if coils0.n_segments != settings.n_segments or coils0.dofs_curves.shape != (settings.n_coils, 3, 2*settings.coil_order+1):
        raise ValueError("expected three order-5 coils with 64 segments")
    coils_path = out / "coils.stage2.json"
    coils0.to_json(str(coils_path))
    metadata = dict(source=str(args.coils.resolve()), source_sha256=sha(args.coils),
                    coils_sha256=sha(coils_path), currents_A=np.asarray(coils0.dofs_currents_raw).tolist())
    write_json("coils.stage2.metadata.json", metadata)
    # Match the fixed example's ESSOS scaled curve coordinates, not the
    # projected optimizer's different mode-dependent step sizes.
    scales = settings.coil_step / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()
    chart = CoilParameters.from_coils(coils0, current_dofs=(), scales=scales)
    n_coils = len(chart.currents)
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    write_json("provenance.json", dict(command=sys.argv, device=str(jax.devices()[0]),
        python=sys.version, jax=jax.__version__, vmex=vj.__file__,
        dependencies={name: version(name) for name in ("numpy", "scipy", "essos", "solvax", "equinox", "diffrax", "lineax", "optimistix", "pyevtk")},
        input_sha256=sha(here / "input.rotating_ellipse"), coils_sha256=sha(coils_path),
        script_sha256=sha(script_path), driver_sha256=sha(Path(__file__)),
        fixed_script_sha256=sha(here / "single_stage_optimization_scalar.py"),
        matched_constants=matched_settings, example_settings=asdict(settings), core_sha256={str(p.relative_to(repo)): sha(p)
            for p in sorted((repo / "vmex/core").glob("*.py"))}, settings=vars(args),
        currents_A=chart.currents.tolist(), stage_two_included_in_timing=False, stage_two=metadata,
        continuation=continuation))
    write_json("constraint_settings.json", dict(enabled=args.constrained,
        **{k:v for k,v in vars(limits).items() if k.isupper()}))
    seed_surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs),
        inp.nfp, nphi=settings.nphi, ntheta=settings.ntheta)
    write_json("initial_coil_costs.json", np.asarray(coil_costs(coils0, seed_surface)).tolist())
    free_started = time.perf_counter()
    print(f"Running {script_path.name}", flush=True)
    print(f"Device: {jax.devices()[0]}; {chart.size} coil variables, fixed currents", flush=True)
    print(f"{args.method}, no AugLag; QA/aspect/iota weights = 1/{settings.aspect_weight}/{settings.iota_weight}")
    print(f"Aspect target {settings.aspect_target}; min |iota| floor {settings.iota_floor}; both are soft penalties.")
    print("Explicit iota/radius inequalities enabled." if args.constrained else "B0 and R are diagnostics.")
    if restored is None:
        print("[seed] solving shared fixed-boundary input...", flush=True)
        fixed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                      polish_force_balance=False)
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
    ftol, niter = args.ftol or float(inp.ftol_array[-1]), int(inp.niter_array[-1])
    inp = replace(inp, ftol_array=np.array([ftol]))
    solver = fbi.make_free_boundary_config(inp, chart(jnp.asarray(chart.x0)),
        field_from_parameters=chart, device=args.device, ftol=ftol, max_iterations=niter,
        include_edge_in_convergence=True, edge_force_tolerance=ftol,
        adjoint_solver="forward_dense_jax", adjoint_dense_batch_size=settings.adjoint_batch_size,
        adjoint_fail="error", adjoint_residual_rtol=1.0e-6, adjoint_dense_max_dofs=args.adjoint_max_dofs)
    rt = im.runtime_from_params(im.params_from_input(inp), solver.implicit)
    if restored is None:
        print("[seed] solving and certifying free equilibrium...", flush=True)
        seed_stage = _solve_free_boundary_stage(inp, external_field=chart(jnp.asarray(chart.x0)),
            resolution=solver.resolution, ftol=ftol, max_iterations=niter, initial_state=fixed.state,
            include_edge_in_convergence=True, edge_force_tolerance=ftol, use_fft=False,
            error_on_no_convergence=True, jacobian_retries=0, allow_initial_axis_reguess=False)
        seed_x, seed_state = chart.x0, seed_stage.result.state
        rcon0, zcon0 = seed_stage.rcon0, seed_stage.zcon0
        seed_iterations = int(seed_stage.result.iterations)
    else:
        print(f"[resume] certifying saved accepted step {step_offset}; fresh SLSQP state", flush=True)
        seed_x = restored["parameters"]
        seed_state = SpectralState(*(jnp.asarray(restored[k]) for k in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")))
        rcon0, zcon0 = jnp.asarray(restored["rcon0"]), jnp.asarray(restored["zcon0"])
        seed_iterations = 0
    cfg = fc.make_free_boundary_continuation_config_from_state(solver, im.params_from_input(inp), seed_x,
        state=seed_state, rcon0=rcon0, zcon0=zcon0,
        parameter_scales=chart.scales, continuation_step=0.1, root_residual_atol=settings.root_tolerance)
    cfg = replace(cfg, _anchor=replace(cfg._anchor,
        result=replace(cfg._anchor.result, iterations=seed_iterations)))
    # This driver uses explicit roots, never cfg's generic memoized root solver.
    # Only the SciPy callback promotes `accepted`; every prediction/retry below
    # is made from that root. cfg is the immutable numerical/ownership context.
    accepted = cfg._anchor
    initial = accepted
    initial_u = np.asarray(accepted.parameters) / scales
    np.savez_compressed(out / "seed_equilibrium.npz", parameters=accepted.parameters,
        rcon0=accepted.rcon0, zcon0=accepted.zcon0,
        **{k: np.asarray(getattr(accepted.state, k)) for k in
           ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    mode_scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    mode_m, mode_n = np.asarray(rt.modes.m), np.asarray(rt.modes.n)

    def boundary(state):
        rc, zs, _, _ = m1_constrained_to_physical(state.R_cos, state.Z_sin,
            state.R_sin, state.Z_cos, modes=rt.modes, lthreed=bool(solver.resolution.lthreed),
            lasym=False, lconm1=True)
        rbc = jnp.zeros_like(jnp.asarray(inp.rbc)).at[mode_n + inp.ntor, mode_m].set(rc[-1] * mode_scale)
        zbs = jnp.zeros_like(jnp.asarray(inp.zbs)).at[mode_n + inp.ntor, mode_m].set(zs[-1] * mode_scale)
        return rbc, zbs

    def surface(state, nphi=settings.nphi, ntheta=settings.ntheta):
        return surfacerzfourier_from_boundary(*boundary(state), inp.nfp, nphi=nphi, ntheta=ntheta)

    def objective(state, x):
        coils = chart.coils_from_x(x)
        surf = surface(state)
        coil_rows = coil_costs(coils, surf)
        if args.no_boundary_error:
            coil_rows = coil_rows.at[:2].set(0.)
        costs = jnp.concatenate((plasma_costs(state, rt), coil_rows))
        return jnp.sum(costs), costs

    # One scalar adjoint includes the equilibrium response of EVERY term,
    # including surface normals, areas and coil-surface clearance. Add the
    # explicit coil derivative to obtain the total derivative.
    local_value_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
    def constrained_rows(state, x):
        value, costs = objective(state, x)
        rows = jnp.concatenate((jnp.atleast_1d(value), limits.inequalities(limits.physical_values(state, rt))))
        return rows, (rows, costs)
    local_rows_jac = jax.jit(jax.jacrev(constrained_rows, argnums=(0,1), has_aux=True))
    local_rows_value = jax.jit(constrained_rows)
    local_value = jax.jit(objective)
    term_names = ("quasisymmetry", "aspect", "iota floor", "normal field", "normal-field excess",
                  "coil length", "coil curvature", "coil separation", "coil-surface separation")
    monitor = opt.OptimizationMonitor(stream=None)
    history = []
    step_history = StepHistory(out)
    accepted_linearization = candidate_linearization = None
    candidate = accepted
    counts = dict(trials=0, solves=int(restored is None), failed_trials=0, accepted=0)
    last = {}
    preconditioner = None
    cycle_started = time.perf_counter()
    cycle_gradient_seconds = 0.0
    cycle_solve_seconds = 0.0

    def metrics(record):
        state = record.state
        return {"QA total": float(qs.total_state(state, rt)), "aspect": float(opt.aspect_ratio(state, rt)),
            "mean iota": float(opt.mean_iota(state, rt)), "min |iota|": float(opt.min_abs_iota(state, rt)),
            "B0 [T]": abs(float(physics.on_axis_magnetic_field(state, rt))),
            "R [m]": float(physics.major_radius(state, rt)),
            "RBC(0,0) [m]": float(boundary(state)[0][inp.ntor, 0]),
            "root residual": float(record.root_residual_norm),
            **{k: float(getattr(record.result, k)) for k in ("fsqr", "fsqz", "fsql", "fedge")}}

    def record_step(value, terms, gradient_seconds=0.0, solve_seconds=0.0, step_seconds=0.0):
        row = dict(step=counts["accepted"], absolute_step=step_offset+counts["accepted"], trials=counts["trials"], objective=float(value),
                   **metrics(accepted), gradient_seconds=gradient_seconds, solve_seconds=solve_seconds,
                   step_seconds=step_seconds, elapsed_seconds=time.perf_counter()-started,
                   equilibrium_iterations=int(accepted.result.iterations))
        history.append(row)
        step_history.record(accepted.parameters / scales, chart.coils_from_x(jnp.asarray(accepted.parameters)),
            surface(accepted.state), objective=value, qa=row["QA total"], aspect=row["aspect"],
            minimum_iota=row["min |iota|"], aspect_target=settings.aspect_target,
            iota_floor=settings.iota_floor, coil_step=settings.coil_step,
            rbc00_m=row["RBC(0,0) [m]"], absolute_step=row["absolute_step"], **limits.diagnostics(accepted.state, rt),
            **{key: row[key] for key in ("fsqr", "fsqz", "fsql")})
        with (out / "free_boundary_scalar_steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        monitor.record(accepted.parameters, cost=float(value), iteration=counts["accepted"],
            equilibrium_solves=counts["solves"], rejected_trials=counts["failed_trials"], terms=terms)
        # Lossless accepted-state evidence; no rejected state is saved as accepted.
        np.savez_compressed(out / f"accepted_{counts['accepted']:04d}.npz", parameters=accepted.parameters,
            rcon0=accepted.rcon0, zcon0=accepted.zcon0,
            **{k: np.asarray(getattr(accepted.state, k)) for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
        print(f"[step {row['absolute_step']:4d}] QA total = {row['QA total']:.6e}, objective = {value:.6e}, "
              f"aspect = {row['aspect']:.4f}, mean iota = {row['mean iota']:.4f}, min |iota| = {row['min |iota|']:.4f}, "
              f"B0 = {row['B0 [T]']:.6f} T, R = {row['R [m]']:.6f} m\n"
              f"            forces = {row['fsqr']:.2e}/{row['fsqz']:.2e}/{row['fsql']:.2e}, edge = {row['fedge']:.2e}; "
              f"gradient {gradient_seconds:.2f} s, solves {solve_seconds:.2f} s, complete step {step_seconds:.2f} s", flush=True)

    class AcceptedBudget(Exception):
        pass

    class EvaluationBudget(Exception):
        pass

    def value_and_grad(u, *, derivative=True, need_gradient=True):
        nonlocal candidate, candidate_linearization, accepted_linearization
        nonlocal cycle_gradient_seconds, cycle_solve_seconds
        # Preserve the saved root bit-for-bit despite coordinate roundoff.
        x = np.asarray(initial.parameters).copy() if np.array_equal(u, initial_u) else scales * np.asarray(u)
        key = x.tobytes()
        if derivative and last.get("key") == key:
            return last["value"], gradient_at(u) if need_gradient else None
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
            tangent = None
            if derivative:
                diagnostics = []
                try:
                    tangent = accepted_linearization.tangent(accepted, cfg, jnp.asarray(delta), diagnostics=diagnostics)
                finally:
                    write_json(f"predictor_{counts['trials']:04d}.json", dict(
                        accepted_step=counts['accepted'], checks=diagnostics))
            points = max(1, int(np.ceil(np.max(np.abs(delta / scales)) / 0.1)))
            if points > 64:
                counts["failed_trials"] += 1
                cycle_solve_seconds += time.perf_counter()-t0
                if not derivative:
                    raise RuntimeError("finite-difference perturbation exceeds the bounded solve path")
                print("  numerical rejection: trial exceeds the bounded prediction path", flush=True)
                return np.inf, np.zeros_like(u)
            try:
                # Every correction starts from the accepted state, never a rejected root.
                predicted = jax.tree.map(lambda a, b: a+b, accepted.state, tangent) if derivative else accepted.state
                correction_ftol = ftol if derivative else (args.fd_ftol or ftol)
                counts["solves"] += 1
                print(f"  correction from accepted step {counts['accepted']}", flush=True)
                stage = _solve_free_boundary_stage(inp, external_field=chart(jnp.asarray(x)),
                    resolution=solver.resolution, ftol=correction_ftol, max_iterations=niter,
                    initial_state=predicted, constraint_continuation=(accepted.rcon0, accepted.zcon0),
                    include_edge_in_convergence=True, edge_force_tolerance=correction_ftol, use_fft=False,
                    error_on_no_convergence=False, jacobian_retries=0, allow_initial_axis_reguess=False)
                if not stage.result.converged:
                    evidence = dict(trial=counts['trials'], accepted_step=counts['accepted'],
                        finite_difference=not derivative, force_tolerance=correction_ftol,
                        iterations=int(stage.result.iterations), ier_flag=int(stage.result.ier_flag),
                        **{k:float(getattr(stage.result,k)) for k in ('fsqr','fsqz','fsql','fedge')})
                    write_json(f"failed_solve_{counts['trials']:04d}.json", evidence)
                    raise VmecError(f"equilibrium did not converge: {evidence}")
                candidate = fc.certify_free_boundary_continuation_state(cfg, x, stage.result.state,
                    rcon0=stage.rcon0, zcon0=stage.zcon0, result=stage.result)
                print(f"  {stage.result.iterations} iterations; edge force {float(stage.result.fedge):.2e}", flush=True)
            except VmecError as exc:
                if not derivative:
                    raise RuntimeError(f"finite-difference equilibrium did not converge: {exc}") from exc
                counts["failed_trials"] += 1
                print(f"  numerical rejection: {exc}", flush=True)
                # SciPy may backtrack; the callback refuses nonfinite trials.
                cycle_solve_seconds += time.perf_counter()-t0
                return np.inf, np.zeros_like(u)
        solve_seconds = time.perf_counter()-t0
        if not derivative:
            # Independent finite differences start directly from the accepted
            # equilibrium, never from the tangent whose accuracy is being tested.
            value = float(objective(candidate.state, jnp.asarray(x))[0])
            return dict(value=value, iterations=int(candidate.result.iterations), seconds=solve_seconds,
                        constraints=np.asarray(limits.inequalities(limits.physical_values(candidate.state, rt))).tolist(),
                        **metrics(candidate))
        if args.constrained:
            rows, (_, costs) = local_rows_value(candidate.state, jnp.asarray(x))
            value, constraints = rows[0], np.asarray(rows[1:])
        else:
            value, costs = local_value(candidate.state, jnp.asarray(x))
            constraints = None
        if not np.isfinite(float(value)) or (constraints is not None and not np.all(np.isfinite(constraints))):
            raise FloatingPointError("nonfinite certified objective/constraints")
        if candidate_linearization is not None and candidate_linearization is not accepted_linearization:
            candidate_linearization.close()
        candidate_linearization = None
        cycle_solve_seconds += solve_seconds
        last.update(key=key, value=float(value), gradient=None, root=candidate,
                    terms=dict(zip(term_names, map(float, np.asarray(costs)))),
                    constraints=constraints, constraint_jacobian=None, solve_seconds=solve_seconds)
        return last["value"], gradient_at(u) if need_gradient else None

    def gradient_at(u):
        """Differentiate a certified trial only when the optimizer requests it."""
        nonlocal candidate, candidate_linearization, accepted_linearization, cycle_gradient_seconds
        x = np.asarray(initial.parameters).copy() if np.array_equal(u, initial_u) else scales*np.asarray(u)
        if last.get("key") != x.tobytes():
            raise RuntimeError("evaluate and certify trial values before derivatives")
        if last["gradient"] is not None:
            return last["gradient"].copy()
        candidate = last["root"]
        solve_seconds = last["solve_seconds"]
        print(f"[trial {counts['trials']}] implicit adjoint...", flush=True)
        t1 = time.perf_counter()
        if args.constrained:
            (rhs, direct_rows), (rows, costs) = local_rows_jac(candidate.state, jnp.asarray(x))
            value, direct_gradient = rows[0], direct_rows[0]
        else:
            (value, costs), (state_bar, direct_gradient) = local_value_grad(candidate.state, jnp.asarray(x))
            rhs = jax.tree.map(lambda a: a[None], state_bar)
        diagnostics = []
        backend = 'dense' if preconditioner is None else 'matrixfree'
        try:
            linearization = fc.free_boundary_continuation_state_pullback(
                candidate, cfg, rhs, return_linearization=True, preconditioner=preconditioner,
                diagnostics=diagnostics)
        finally:
            write_json(f"adjoint_{counts['trials']:04d}_{backend}.json", dict(
                accepted_step=counts['accepted'], checks=diagnostics))
        gradient = (np.asarray(linearization.field_jacobian)[0] + np.asarray(direct_gradient)) * scales
        if not np.isfinite(float(value)) or not np.all(np.isfinite(gradient)):
            linearization.close()
            raise FloatingPointError("nonfinite certified objective/gradient")
        try:
            np.testing.assert_allclose(float(value), last["value"], rtol=1e-10, atol=1e-12)
            if args.constrained:
                np.testing.assert_allclose(np.asarray(rows[1:]), last["constraints"], rtol=1e-10, atol=1e-12)
        except AssertionError:
            linearization.close()
            raise
        linearization.offload_factors()
        candidate_linearization = linearization
        if candidate is accepted:
            if accepted_linearization is not None:
                accepted_linearization.close()
            accepted_linearization = linearization
        gradient_seconds = time.perf_counter()-t1
        cycle_gradient_seconds += gradient_seconds
        last.update(key=x.tobytes(), value=float(value), gradient=gradient.copy(), root=candidate,
                    terms=dict(zip(term_names, map(float, np.asarray(costs)))))
        if args.constrained:
            last.update(constraints=np.asarray(rows[1:]),
                constraint_jacobian=(np.asarray(linearization.field_jacobian)[1:]+np.asarray(direct_rows)[1:])*scales)
        print(f"  objective {float(value):.6e}, QA total {2*float(costs[0]):.6e}; "
              f"solve/predict {solve_seconds:.2f} s, gradient {gradient_seconds:.2f} s", flush=True)
        return gradient.copy()

    def callback(u):
        nonlocal accepted, accepted_linearization, cycle_started, cycle_gradient_seconds, cycle_solve_seconds
        x = np.asarray(initial.parameters) if np.array_equal(u, initial_u) else scales*np.asarray(u)
        if last.get("key") != x.tobytes():
            raise RuntimeError("SciPy accepted a trial without a finite certified evaluation")
        if last.get("gradient") is None or candidate_linearization is None:
            raise RuntimeError("cannot accept a trial before its certified gradient")
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
        if args.accepted_steps is not None and counts['accepted'] >= args.accepted_steps:
            raise AcceptedBudget()

    status, result, failure = "iteration_budget_reached", None, None
    initial_value = None
    derivative_qualified = False
    optimization_started = None
    try:
        initial_value, _ = value_and_grad(initial_u)
        if continuation:
            measured = metrics(initial)
            for key in ("QA total", "min |iota|", "R [m]", "aspect"):
                if not np.isclose(measured[key], continuation["previous_final"][key], rtol=1e-10, atol=1e-12):
                    raise ValueError(f"restored endpoint metric differs: {key}")
            for key, value in restored.items():
                actual = getattr(initial.state, key) if hasattr(initial.state, key) else getattr(initial, key)
                if not np.array_equal(np.asarray(actual), value):
                    raise ValueError(f"restored checkpoint changed: {key}")
            write_json("resume_check.json", dict(passed=True, arrays_identical=True, metrics=measured, **continuation))
        record_step(initial_value, last["terms"], cycle_gradient_seconds, cycle_solve_seconds)
        if args.check_gradient:
            direction = np.random.default_rng(0).normal(size=chart.size)
            direction /= np.linalg.norm(direction)
            analytic = float(np.dot(last["gradient"], direction))
            constraint_analytic = last["constraint_jacobian"] @ direction if args.constrained else None
            from _scalar_gradient_check import centered_checks
            checks, gradient_endpoints = centered_checks(
                lambda u: value_and_grad(u, derivative=False), initial_u, direction,
                analytic, constraint_analytic,
                save=lambda report: write_json("gradient_check.json", dict(
                    seed=0, predictor=False, force_tolerance=args.fd_ftol or ftol, **report)))
            objective_ok, constraints_ok = gradient_check_passes(
                checks, args.gradient_rtol, args.gradient_fine_only, args.constrained)
            passed = objective_ok and constraints_ok
            write_json("gradient_check.json", dict(seed=0, predictor=False, endpoints=gradient_endpoints,
                checks=checks, passed=passed, force_tolerance=args.fd_ftol or ftol, objective_rtol=args.gradient_rtol,
                objective_gate="finest h" if args.gradient_fine_only else "all h", constraint_rtol=1e-3))
            if not objective_ok:
                raise RuntimeError("independently corrected directional derivative check failed")
            if not passed:
                raise RuntimeError("constraint directional derivative check failed")
            value_and_grad(initial_u)
        if args.adjoint == "matrixfree":
            from jax.flatten_util import ravel_pytree
            dense_rows = np.vstack((last['gradient'], last['constraint_jacobian'])) if args.constrained else last['gradient'][None]
            delta = jnp.asarray(scales*direction*1e-3)
            dense_tangent = accepted_linearization.tangent(accepted, cfg, delta)
            preconditioner = accepted_linearization.preconditioner()
            last['gradient'] = None
            value_and_grad(initial_u)
            rows = np.vstack((last['gradient'], last['constraint_jacobian'])) if args.constrained else last['gradient'][None]
            errors = np.linalg.norm(rows-dense_rows,axis=1)/np.maximum(np.linalg.norm(dense_rows,axis=1),1e-30)
            tangent = accepted_linearization.tangent(accepted, cfg, delta)
            difference = ravel_pytree(jax.tree.map(jnp.subtract,tangent,dense_tangent))[0]
            tangent_error = float(jnp.linalg.norm(difference)/jnp.maximum(jnp.linalg.norm(ravel_pytree(dense_tangent)[0]),1e-30))
            passed = bool(np.all(errors<1e-8) and np.isfinite(tangent_error) and tangent_error<1e-8)
            write_json('matrixfree_check.json',dict(passed=passed,gradient_relative_errors=errors.tolist(),tangent_relative_error=tangent_error))
            if not passed:
                raise RuntimeError('matrix-free seed gradient or predictor differs from dense reference')
        derivative_qualified = args.check_gradient
        write_json("qualification.json", dict(passed=derivative_qualified,
            adjoint=args.adjoint, seconds=time.perf_counter()-started))
        optimization_started = time.perf_counter()
        if args.optimization_seconds is not None:
            phase("optimization", args.optimization_seconds)
        # The first gradient is also included in the first complete-step time.
        options = {"maxiter": args.maxiter, "gtol": 1.0e-8}
        bounds = None
        if args.method == "L-BFGS-B":
            options.update(maxls=20, ftol=1e-12, maxcor=20, maxfun=args.max_trials-counts["trials"])
            # The control's box is centred on the pre-fit coils. Our u=0 is
            # the fitted set, so translate the box along with the coordinates.
            original_coils = Coils.from_json(str(here / "coils.initial.scalar.json"))
            fit_offset = (np.asarray(coils0.curves.dofs)-np.asarray(original_coils.curves.dofs)).ravel() / settings.coil_step
            bounds = [(-settings.parameter_bound-offset, settings.parameter_bound-offset) for offset in fit_offset]
        constraints = ()
        if args.constrained:
            options = {"maxiter": args.maxiter, "ftol": 1e-10}
            def constraint_values(u):
                value, _ = value_and_grad(u, need_gradient=False)
                return last["constraints"].copy() if np.isfinite(value) else np.full(3, -1e6)
            def constraint_jacobian(u):
                value, _ = value_and_grad(u)
                return last["constraint_jacobian"].copy() if np.isfinite(value) else np.zeros((3,chart.size))
            constraints = [dict(type="ineq", fun=constraint_values, jac=constraint_jacobian)]
        def slsqp_gradient(u):
            # Gradient requests, unlike SLSQP callbacks, follow accepted line searches.
            _, gradient = value_and_grad(u)
            callback(u)
            return gradient
        result = minimize((lambda u:value_and_grad(u, need_gradient=False)[0]) if args.constrained else value_and_grad,
                          initial_u, jac=slsqp_gradient if args.constrained else True, method=args.method,
                          bounds=bounds, constraints=constraints, callback=None if args.constrained else callback, options=options)
        if args.constrained:
            value_and_grad(result.x)
            callback(result.x)
        status = "converged" if result.success else ("iteration_budget_reached" if result.status == (9 if args.constrained else 1) else "optimizer_failed")
        print(f"[{args.method}] {result.nit} iterations, {counts['trials']} trials: {result.message}", flush=True)
    except AcceptedBudget:
        status = "accepted_step_budget_reached"
    except EvaluationBudget:
        status = "evaluation_budget_reached"
    except TimeoutError:
        status = ("optimization_time_budget_reached" if optimization_started is not None
                  and args.optimization_seconds is not None else "walltime_budget_reached")
    except Exception as exc:
        status, failure = "failed", f"{type(exc).__name__}: {exc}"
        print(f"Optimization stopped: {failure}", flush=True)
    finally:
        for linearization in (candidate_linearization, accepted_linearization):
            if linearization is not None:
                linearization.close()
        if preconditioner is not None:
            preconditioner.close()
        signal.alarm(0)

    optimization_seconds = 0.0 if optimization_started is None else time.perf_counter()-optimization_started
    summary = dict(example=script_path.name, status=status, failure=failure, method=args.method, **counts,
        absolute_step=step_offset+counts["accepted"], continuation=continuation,
        optimization_seconds=optimization_seconds, stage_two=metadata,
        free_initialization_and_optimization_seconds=time.perf_counter()-free_started,
        seed=metrics(initial), final=metrics(accepted),
        constrained=args.constrained, constraint_diagnostics=limits.diagnostics(accepted.state, rt),
        optimizer_success=status == "converged", derivative_qualified=derivative_qualified, directional_check_requested=args.check_gradient,
        targets={"min |iota| floor": settings.iota_floor, "aspect target (soft)": settings.aspect_target,
                 "radius target [m]": limits.RADIUS_TARGET if args.constrained else None,
                 "radius tolerance [m]": limits.RADIUS_TOLERANCE if args.constrained else None},
        best_feasible_step=min((row for row in step_history.rows if row["constraints_feasible"]), key=lambda row:row["objective"], default={}).get("step"),
        diagnostics_only=["B0"] if args.constrained else ["B0", "major radius"], verification="not run")
    write_json("free_boundary_single_stage_scalar_optimization_summary.json", summary)
    chart.coils_from_x(jnp.asarray(accepted.parameters)).to_json(str(out / "coils_free_boundary_single_stage_scalar_optimized.json"))
    monitor.save(out / "free_boundary_single_stage_scalar_objectives.csv")
    if failure:
        raise RuntimeError(failure)

    # All WOUT construction, independent verification, VTK and plots happen
    # after the timed optimizer, never in value_and_grad or its callback.
    phase("verification", args.verification_seconds)
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
    initial_wout = export(initial, "wout_free_boundary_single_stage_scalar_initial.nc")
    accepted_wout = export(accepted, "wout_free_boundary_single_stage_scalar_accepted.nc")
    rbc, zbs = boundary(accepted.state)
    final_input = replace(inp, rbc=np.asarray(rbc), zbs=np.asarray(zbs),
        ns_array=np.array([args.verify_ns]), ftol_array=np.array([args.verify_ftol]), niter_array=np.array([args.verify_maxiter]))
    final_input.to_indata(out / "input.free_boundary_single_stage_scalar_optimized")
    final_coils = chart.coils_from_x(jnp.asarray(accepted.parameters))
    verification = FreeBoundaryProblem.from_tuples(final_input, [(qs.residuals_state, 0.0, 1.0)],
        coils=final_coils, current_dofs=(), restart_from=accepted_wout, objective_normalization=1.0,
        solver_options=dict(device=args.device, ftol=args.verify_ftol, max_iterations=args.verify_maxiter,
            edge_force_tolerance=args.verify_ftol, adjoint_dense_batch_size=settings.adjoint_batch_size))
    try:
        verified = verification.equilibrium_from_x(verification.x0)
        forces = {k:float(getattr(verified.result,k)) for k in ('fsqr','fsqz','fsql','fedge')}
        if not verified.result.converged or not all(np.isfinite(v) and v<=args.verify_ftol for v in forces.values()):
            raise RuntimeError(f"independent endpoint force checks failed: {forces}")
        wout_path = vj.write_wout(out / "wout_free_boundary_single_stage_scalar_optimized.nc", verified.wout)
        reporter = opt.EquilibriumReporter(("QA total", qs.total, ".6e"),
            ("aspect", opt.aspect_ratio, ".4f"), ("mean iota", opt.mean_iota, ".4f"),
            ("min |iota|", opt.min_abs_iota, ".4f"),
            ("B0 [T]", physics.on_axis_magnetic_field, ".6f"), ("R [m]", physics.major_radius, ".6f"))
        final_values = reporter("final verification", verified)
        endpoint_check = dict(numerically_verified=True, ns=args.verify_ns, mpol=inp.mpol, ntor=inp.ntor,
            grid=[inp.ntheta,inp.nzeta], force_tolerance=args.verify_ftol, forces=forces,
            root_residual=float(verification.accepted.root_residual_norm),
            axis_iota=float(verified.wout.iotaf[0]),
            full_mesh_min_abs_iota=float(np.min(np.abs(verified.wout.iotaf))),
            constraint_scope="Configured half-mesh iota minimum (axis excluded) and physical major radius",
            normal_field_diagnostic_only=args.no_boundary_error, values=final_values)
        endpoint_check['physical_constraints_met'] = bool(final_values['min |iota|']>=limits.IOTA_FLOOR
            and abs(final_values['R [m]']-limits.RADIUS_TARGET)<=limits.RADIUS_TOLERANCE)
        endpoint_check['full_mesh_iota_floor_met'] = endpoint_check['full_mesh_min_abs_iota']>=limits.IOTA_FLOOR
        write_json('verification.json', endpoint_check)
    finally:
        verification.close()
    from essos.surfaces import SurfaceRZFourier
    surface_initial = SurfaceRZFourier.from_wout_file(initial_wout, nphi=60, ntheta=60)
    surface_final = SurfaceRZFourier.from_wout_file(wout_path, nphi=inp.nzeta if args.grid else 61,
                                                  ntheta=inp.ntheta if args.grid else 64)
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
    checks = [("min |iota|", final_values["min |iota|"], limits.IOTA_FLOOR if args.constrained else settings.iota_floor, "below"),
        ("B.n/B RMS", rms, settings.normal_field_limit, "above"),
        ("B.n/B max", maximum, settings.normal_field_limit, "above"),
        ("coil-surface distance", coil_surface_distance, settings.coil_surface_distance_limit, "below"),
        ("coil-coil distance", coil_distance, settings.coil_distance_limit, "below"),
        ("maximum curvature", curvature, settings.curvature_limit, "above")]
    if args.constrained:
        checks.extend([("major radius lower", final_values["R [m]"], limits.RADIUS_TARGET-limits.RADIUS_TOLERANCE, "below"),
                       ("major radius upper", final_values["R [m]"], limits.RADIUS_TARGET+limits.RADIUS_TOLERANCE, "above")])
    unmet = [f"{name} {value:.6g} {side} {limit}" for name, value, limit, side in checks
             if not np.isfinite(value) or (value < limit if side == "below" else value > limit)]
    print(f"\nObjective initially {initial_value:.6e}; {counts['accepted']} accepted steps, {counts['trials']} trials; {status}")
    print(f"Coil lengths = {np.asarray(final_coils.length[:n_coils])}")
    print(f"B.n/B: area-weighted RMS = {100*rms:.3f}%, max = {100*maximum:.3f}%")
    print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m; coil-coil = {coil_distance:.4f} m")
    print(f"Maximum curvature = {curvature:.4f} 1/m")
    print("Unmet inequality limits: " + "; ".join(unmet) if unmet else "Reported inequality limits met; aspect and length remain soft targets.")
    summary.update(verification="passed numerical checks", verified=dict(final_values, **{"B.n/B RMS": rms,
        "B.n/B max": maximum, "coil-surface distance": coil_surface_distance,
        "coil-coil distance": coil_distance, "maximum curvature": curvature, "coil lengths": np.asarray(final_coils.length[:n_coils]).tolist(),
        "aspect target error": final_values["aspect"]-settings.aspect_target}), unmet=unmet, inequalities_met=not unmet)
    write_json("free_boundary_single_stage_scalar_optimization_summary.json", summary)
    initial_coils = chart.coils_from_x(jnp.asarray(initial.parameters))
    for label, surf, coils in (("initial", surface_initial, initial_coils), ("optimized", surface_final, final_coils)):
        surf.to_vtk(str(out / f"surface_free_boundary_single_stage_scalar_{label}"), field=BiotSavart(coils))
        coils.to_vtk(str(out / f"coils_free_boundary_single_stage_scalar_{label}"))
    if not args.no_plots:
        vj.plot_optimization_objects(out / "free_boundary_single_stage_scalar_optimization.png",
            ("Initial", surface_initial, initial_coils), ("Optimized", surface_final, final_coils))
        monitor.plot(out / "free_boundary_single_stage_scalar_objectives.png", title="Free-boundary objective terms")
        for path in vj.plot_wout(wout_path, out).values():
            print(f"Wrote {path}")
    signal.alarm(0)
    print(f"Wrote results to {out}", flush=True)
    return 1 if unmet or not summary["optimizer_success"] else 0
