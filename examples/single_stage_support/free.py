"""Shared free-boundary logging, endpoint checks and plot exports.

The production entry supplies its physical problem and optimizer. Derivative
qualification lives in separate verification programs and is never invoked here.
"""
import csv
from dataclasses import replace
import json
from functools import lru_cache
import signal
import time
from types import SimpleNamespace

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str)+"\n")

def run(args, *, case, method="SLSQP", coil_limits=None):
    """Prepare or restore a start, optimize and independently solve the endpoint."""
    if method not in ("SLSQP", "L-BFGS-B"):
        raise ValueError(f"unsupported optimizer: {method}")
    settings = {name: value for name, value in vars(case).items() if name.isupper() and name not in ("HERE", "P")}
    if args.dry_run:
        print(json.dumps(dict(optimizer=method, parameters=settings, arguments=vars(args)), indent=2, default=str))
        return 0
    out = case.setup_run(args)
    if getattr(case, "COIL_CONSTRAINTS", False):
        import _coil_constraints as coil_limits
    import numpy as np
    from vmex import optimize as opt
    qualified = case.read_qualification(args)

    def timeout(*_):
        raise TimeoutError("phase wall-time budget reached")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(case.INITIALIZATION_SECONDS)
    problem = None
    phase = "initialization"
    started = time.perf_counter()
    try:
        timings = dict(adjoint=0.0, tangent=0.0, correction=0.0, root_polish=0.0)
        trials = 0

        def event(name, **data):
            nonlocal trials
            if name in timings and 'seconds' in data:
                timings[name] += data.get('total_seconds', data['seconds'])
            if name == "proposal":
                trials += 1
                if trials > case.MAX_TRIALS:
                    raise StopIteration("trial budget reached")
            if name in ("adjoint", "tangent", "matrixfree_check", "dense_recovery", "preconditioner_refresh", "preconditioner_refresh_start", "root_polish", "accepted_state_retry"):
                # Keep diagnostic rows and failure reasons without serializing root arrays.
                record = {key: value for key, value in data.items() if key != "candidate"}
                with (out / "solver_events.jsonl").open("a") as stream:
                    stream.write(json.dumps({"event": name, "proposal_count": trials, **record}, default=str) + "\n")

        stage = case.build_problem(args, event=event, qualified=qualified)
        problem, inp = stage.problem, stage.inp
        qs, inequalities = stage.qs, stage.inequalities
        initial_equilibrium = problem.equilibrium_from_x(problem.x0)
        monitor = opt.OptimizationMonitor(stream=None)
        history = []
        cycle_started = time.perf_counter()

        def record_step():
            nonlocal cycle_started
            x = problem.accepted.parameters
            eq = problem.equilibrium_from_x(x)
            physical = problem.constraint_values(x)
            iota, radius = physical[:2]
            row = dict(step=problem.accepted_step, objective=problem.fun(x),
                qa=float(qs.total_state(eq.state, eq.runtime)), aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
                min_abs_iota=float(iota), major_radius_m=float(radius),
                rbc00_m=float(opt.boundary_from_state(eq.state, eq.runtime)[0][inp.ntor, 0]),
                radius_error_m=float(radius-case.RADIUS_TARGET), iota_constraint_slack=float(iota-case.IOTA_FLOOR),
                radius_constraint_slack_m=float(case.RADIUS_TOLERANCE-abs(radius-case.RADIUS_TARGET)),
                optimizer_constraints_feasible=bool(np.min(inequalities(physical)) >= -1e-8),
                constraints_feasible=bool(iota >= case.IOTA_FLOOR and abs(radius-case.RADIUS_TARGET) <= case.RADIUS_TOLERANCE),
                gradient_seconds=timings["adjoint"], predictor_seconds=timings["tangent"], solve_seconds=timings["correction"],
                polish_seconds=timings['root_polish'], root_residual=float(problem.accepted.root_residual_norm),
                step_seconds=time.perf_counter()-cycle_started, elapsed_seconds=time.perf_counter()-started,
                **{key: float(getattr(eq.result, key)) for key in ("fsqr", "fsqz", "fsql", "fedge")})
            if hasattr(stage, "interface_metrics"):
                row.update(stage.interface_metrics(eq.state, eq.runtime, problem.coils_from_x(x)))
            if coil_limits is not None:
                slack = float(np.min(stage.coil_constraint.fun(x)))
                clearance = float(physical[2])
                row.update(coil_minimum_scaled_slack=slack, coil_surface_distance_m=clearance)
                row["optimizer_constraints_feasible"] &= slack >= -1e-8
                row["constraints_feasible"] &= slack >= 0 and clearance >= case.COIL_SURFACE_DISTANCE_LIMIT
            history.append(row)
            write_json(out / "accepted_steps.json", history)
            write_json(out / f"checkpoint_{problem.accepted_step:04d}.json", problem.save_checkpoint(out/f"accepted_{problem.accepted_step:04d}.npz"))
            monitor.record(x, cost=row["objective"], iteration=problem.accepted_step)
            monitor.save(out / "free_boundary_scalar_objectives.csv")
            print(f"[step {problem.accepted_step}] objective {row['objective']:.6e}; QA {row['qa']:.6e}; "
                  f"gradient {row['gradient_seconds']:.2f}s, predictor {row['predictor_seconds']:.2f}s, "
                  f"correction {row['solve_seconds']:.2f}s", flush=True)
            timings.update(dict.fromkeys(timings, 0.0))
            cycle_started = time.perf_counter()

        # Production constructs its own checked LU; parity/FD experiments live
        # exclusively in verify_free_boundary_single_stage.py.
        case.configure_solver(problem, args)
        record_step()
        phase = "optimization"
        optimization_started = time.perf_counter()
        signal.alarm(case.OPTIMIZATION_SECONDS)

        status, result = "iteration_budget_reached", None
        try:
            result = case.run_optimizer(stage, args, record_step, method=method)
            if result.stop_reason:
                status = result.stop_reason
            elif result.success:
                status = "converged"
            else:
                budget_status = 1 if method == "L-BFGS-B" else 9
                status = "iteration_or_evaluation_budget_reached" if result.status == budget_status else "optimizer_failed"
        except StopIteration as error:
            status = str(error).replace(" ", "_")
        except TimeoutError:
            status = "optimization_time_budget_reached"
        except opt.TrialRejected as error:
            # L-BFGS-B needs a gradient for every proposal. Stop cleanly at the
            # last certified accepted state when a proposal cannot supply one.
            status = "equilibrium_trial_rejected"
            write_json(out / "rejected_trial.json", dict(error=str(error)))
        summary = dict(optimizer=method, linear_solver=problem.solver_info, nonlinear_constraints=method == "SLSQP", status=status, optimizer_success=status == "converged", accepted_steps=problem.accepted_step,
            optimization_seconds=time.perf_counter()-optimization_started,
            derivative_qualified=qualified is not None,
            qualification=None if args.qualification is None else str(args.qualification.resolve()),
            initial=history[0], final=history[-1], verification="not run",
            optimizer_message=None if result is None else str(result.message))
        write_json(out / "optimization_summary.json", summary)

        phase = "verification"
        verified = case.verify_endpoint(stage, args, summary, history, initial_equilibrium)
        phase = "postprocessing"
        case.postprocess(stage, args, summary, monitor, history, initial_equilibrium, verified)
        print(f"{status}; numerical verification passed; physical limits met: "
              f"{summary['inequalities_met']}. Results: {out}", flush=True)
        return 0 if summary["optimizer_success"] and summary["inequalities_met"] else 1
    except Exception as error:
        write_json(out / "failure.json", dict(phase=phase, error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        if problem is not None:
            problem.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

def verify_endpoint(case, coil_limits, stage, args, summary, history, initial_equilibrium):
    """Re-solve the final coils at higher resolution and report physical limits."""
    import jax
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    out = args.output.resolve()
    problem, inp, qs = stage.problem, stage.inp, stage.qs
    method = summary["optimizer"]
    verification = None
    try:
        signal.alarm(case.VERIFICATION_SECONDS)
        accepted_eq = problem.equilibrium_from_x(problem.accepted.parameters)
        initial_wout = vj.write_wout(out / "wout_initial.nc", initial_equilibrium.wout)
        accepted_wout = vj.write_wout(out / "wout_accepted.nc", accepted_eq.wout)
        final_coils = problem.coils_from_x(problem.accepted.parameters)
        final_coils.to_json(str(out / "coils_optimized.json"))
        rbc, zbs, _, _ = opt.boundary_from_state(accepted_eq.state, accepted_eq.runtime)
        final_input = replace(inp, rbc=np.asarray(rbc), zbs=np.asarray(zbs), ns_array=np.array([case.VERIFY_NS]),
                              ftol_array=np.array([case.VERIFY_FTOL]), niter_array=np.array([case.VERIFY_MAXITER]))
        final_input.to_indata(out / "input.optimized")
        problem.close()
        verification = opt.FreeBoundaryProblem.from_tuples(final_input, [(qs.residuals_state, 0, 1)],
            coils=final_coils, coil_current_dofs=(), restart_from=accepted_wout, objective_normalization=1.0,
            solver_options=dict(device=args.device, ftol=case.VERIFY_FTOL, edge_force_tolerance=case.VERIFY_FTOL,
                                max_iterations=case.VERIFY_MAXITER))
        verified = verification.equilibrium_from_x(verification.x0)
        forces = {key: float(getattr(verified.result, key)) for key in ("fsqr", "fsqz", "fsql", "fedge")}
        if not verified.result.converged or not all(np.isfinite(v) and v <= case.VERIFY_FTOL for v in forces.values()):
            raise RuntimeError(f"independent endpoint force checks failed: {forces}")
        wout_path = vj.write_wout(out / "wout_optimized.nc", verified.wout)
        surf = SurfaceRZFourier.from_wout_file(wout_path, nphi=coil_limits.VERIFY_SURFACE_GRID[0] if coil_limits else 61,
                                              ntheta=coil_limits.VERIFY_SURFACE_GRID[1] if coil_limits else 64)
        field = np.asarray(jax.vmap(BiotSavart(final_coils).B)(surf.gamma.reshape(-1, 3))).reshape(surf.gamma.shape)
        bn = np.sum(field*np.asarray(surf.unitnormal), axis=2)/np.linalg.norm(field, axis=2)
        weights = np.asarray(surf.area_element)
        rms, maximum = float(np.sqrt(np.sum(weights*bn**2)/weights.sum())), float(np.max(np.abs(bn)))
        interface_metrics = {}
        if hasattr(stage, "interface_metrics"):
            interface_metrics = stage.interface_metrics(verified.state, verified.runtime, final_coils, 61, 64)
            rms, maximum = interface_metrics["normal_field_rms"], interface_metrics["normal_field_max"]
        points, surface_points = np.asarray(final_coils.gamma), np.asarray(surf.gamma).reshape(-1, 3)
        coil_distance = min(float(np.linalg.norm(points[i][:, None]-points[j][None], axis=2).min())
                            for i in range(len(points)) for j in range(i+1, len(points)))
        clearance = min(float(np.linalg.norm(p[:, None]-surface_points[None], axis=2).min()) for p in points)
        iota, radius = float(opt.min_abs_iota(verified.state, verified.runtime)), float(opt.major_radius(verified.state, verified.runtime))
        geometry_report = None
        curvature = float(np.max(np.asarray(final_coils.curvature)))
        if coil_limits is not None:
            geometry_report = coil_limits.verify(final_coils, surf, out / "coil_verification.json")
            curvature = max(r["peak_per_m"] for r in geometry_report["coils"])
            coil_distance = geometry_report["coil_coil_lower_bound_m"]
            clearance = geometry_report["refined_coil_surface_distance_m"]
        reporter = opt.EquilibriumReporter(
            ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
            ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
        reported = reporter("final verification", verified)
        lengths = np.asarray(final_coils.length[:case.N_COILS])
        print(f"\nObjective: {history[0]['objective']:.6e} -> {history[-1]['objective']:.6e} "
              f"in {problem.accepted_step} accepted {method} steps")
        print(f"Coil lengths = {lengths} (length reference {case.LENGTH_TARGET:.4f} m)")
        print(f"B.n/B: RMS = {100*rms:.3f}%, max = {100*maximum:.3f}% "
              f"(target < {100*case.NORMAL_FIELD_LIMIT:.1f}%)")
        print(f"Minimum coil-surface distance = {clearance:.4f} m (target >= {case.COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
        print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {case.COIL_DISTANCE_LIMIT:.4f} m)")
        print(f"Maximum curvature = {curvature:.4f} 1/m (target <= {case.CURVATURE_LIMIT:.4f} 1/m)")
        print(f"Minimum |iota| = {iota:.4f} (target >= {case.IOTA_FLOOR:.4f}); "
              f"aspect = {reported['aspect']:.4f} (soft target {case.ASPECT_TARGET:.4f})")
        print(f"Major radius = {radius:.6f} m (target {case.RADIUS_TARGET:.4f} +/- {case.RADIUS_TOLERANCE:.4f} m)")
        checks = [("minimum |iota|", iota, case.IOTA_FLOOR, "below"),
                  ("radius error [m]", abs(radius-case.RADIUS_TARGET), case.RADIUS_TOLERANCE, "above"),
                  ("B.n/B RMS", rms, case.NORMAL_FIELD_LIMIT, "above"),
                  ("B.n/B maximum", maximum, case.NORMAL_FIELD_LIMIT, "above"),
                  ("coil-surface distance [m]", clearance, case.COIL_SURFACE_DISTANCE_LIMIT, "below"),
                  ("coil-coil distance [m]", coil_distance, case.COIL_DISTANCE_LIMIT, "below"),
                  ("curvature [1/m]", curvature, case.CURVATURE_LIMIT, "above")]
        unmet = [f"{name}: {value:.6g} {side} {limit:.6g}" for name, value, limit, side in checks
                 if not np.isfinite(value) or (value < limit if side == "below" else value > limit)]
        if geometry_report is not None:
            unmet += [name for name,passed in geometry_report["checks"].items() if not passed]
        feasible = not unmet
        if unmet:
            print("This run did NOT meet its stated limits: " + "; ".join(unmet))
        summary.update(verification="passed numerical checks", inequalities_met=feasible, unmet=unmet,
            coil_constraints=geometry_report,
            best_feasible_step=min((row for row in history if row["constraints_feasible"]),
                                   key=lambda row: row["objective"], default={}).get("step"),
            verified=dict(forces=forces, ns=case.VERIFY_NS, qa=reported["QS total"], aspect=reported["aspect"],
                          mean_iota=reported["mean iota"], magnetic_well=reported["magnetic well"], min_abs_iota=iota,
                          major_radius_m=radius, normal_field_rms=rms, normal_field_max=maximum,
                          coil_distance_m=coil_distance, coil_surface_distance_m=clearance, maximum_curvature=curvature,
                          coil_lengths_m=lengths.tolist(), aspect_target_error=reported["aspect"]-case.ASPECT_TARGET,
                          radius_error_m=radius-case.RADIUS_TARGET, iota_constraint_slack=iota-case.IOTA_FLOOR,
                          radius_constraint_slack_m=case.RADIUS_TOLERANCE-abs(radius-case.RADIUS_TARGET),
                          full_mesh_min_abs_iota=float(np.min(np.abs(verified.wout.iotaf)))))
        summary["verified"].update(interface_metrics)
        if getattr(stage, "finite_beta", False):
            summary.update(beta_policy="diagnostic only; fixed pressure and PHIEDGE", plasma_current_A=0.)
        write_json(out / "optimization_summary.json", summary)

        return SimpleNamespace(initial_wout=initial_wout, wout_path=wout_path,
                               final_coils=final_coils, surface=surf)
    finally:
        if verification is not None:
            verification.close()

def postprocess(case, stage, args, summary, monitor, history, initial_equilibrium, verified):
    """Export accepted-state diagnostics, geometry, figures, movie and WOUT plots."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    out = args.output.resolve()
    problem, inp, chart, coils0 = stage.problem, stage.inp, stage.chart, stage.coils
    surface, coil_costs, scales = stage.surface, stage.coil_costs, stage.chart.scales
    initial_wout, wout_path = verified.initial_wout, verified.wout_path
    surf, final_coils = verified.surface, verified.final_coils
    _phase = "postprocessing"
    signal.alarm(case.POSTPROCESSING_SECONDS)
    postprocessing_started = time.perf_counter()
    seed_surface = SurfaceRZFourier.from_wout_file(initial_wout, nphi=60, ntheta=60)
    for label, export_surface, export_coils in (("initial", seed_surface, coils0), ("optimized", surf, final_coils)):
        # A coil-only field is not the total field in a finite-beta plasma.
        export_surface.to_vtk(str(out / f"surface_{label}"),
                              **({} if getattr(stage, "finite_beta", False) else {"field": BiotSavart(export_coils)}))
        export_coils.to_vtk(str(out / f"coils_{label}"))

    # Replay saved accepted roots, never solve previous iterates again.
    points = monitor.x_history
    frame_steps = {tuple(x): step for step, x in enumerate(points)}

    @lru_cache(maxsize=1)
    def accepted_frame(step):
        checkpoint = out / f"accepted_{step:04d}.npz"
        identity = json.loads((out / f"checkpoint_{step:04d}.json").read_text())
        state = problem.state_from_checkpoint(checkpoint, sha256=identity["sha256"], parameters=points[step])
        return state, surface(state, initial_equilibrium.runtime), chart.coils_from_x(jnp.asarray(points[step]))

    post_monitor = opt.OptimizationMonitor(stream=None)
    coil_cost_values = jax.jit(coil_costs)
    coil_diagnostics = opt.CoilDiagnostics(scales, coefficient_step=case.COIL_STEP)
    for step, (x, row) in enumerate(zip(points, history)):
        state, step_surface, step_coils = accepted_frame(step)
        row.update(coil_diagnostics.record(x, step_coils, path=out / f"step_{step:04d}.npz"))
        coil_field = np.asarray(jax.vmap(BiotSavart(step_coils).B)(step_surface.gamma.reshape(-1, 3))).reshape(step_surface.gamma.shape)
        normal_field = np.sum(coil_field*np.asarray(step_surface.unitnormal), axis=-1)/np.linalg.norm(coil_field, axis=-1)
        area = np.asarray(step_surface.area_element)
        row.update(qa_residual_l2=float(np.sqrt(row["qa"])), aspect_error=row["aspect"]-case.ASPECT_TARGET,
            iota_violation=max(case.IOTA_FLOOR-row["min_abs_iota"], 0.0),
            normal_field_rms=float(np.sqrt(np.sum(area*normal_field**2)/area.sum())),
            normal_field_max=float(np.max(np.abs(normal_field))))
        if hasattr(stage, "interface_metrics"):
            row.update(stage.interface_metrics(state, initial_equilibrium.runtime, step_coils))
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError(f"nonfinite accepted-step diagnostics at step {step}")
        terms = dict(quasisymmetry=0.5*row["qa"], aspect=0.5*case.ASPECT_WEIGHT*row["aspect_error"]**2,
                     **{"iota floor": 0.5*case.IOTA_WEIGHT*row["iota_violation"]**2})
        terms.update(zip(("coil length", "coil curvature", "coil separation", "coil-surface separation"),
                         map(float, np.asarray(coil_cost_values(step_coils, step_surface)))))
        if hasattr(stage, "extra_objective_terms"):
            terms.update({key: float(value) for key, value in
                          stage.extra_objective_terms(state, initial_equilibrium.runtime, step_coils).items()})
        np.testing.assert_allclose(sum(terms.values()), row["objective"], rtol=1e-10, atol=1e-12)
        post_monitor.record(x, cost=row["objective"], iteration=step, terms=terms)
    with (out / "accepted_steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(out / "accepted_steps.json", history)
    post_monitor.save(out / "free_boundary_scalar_objectives.csv")

    if not args.no_plots:
        vj.plot_optimization_objects(out / "optimization.png", ("Initial", seed_surface, coils0), ("Optimized", surf, final_coils))
        post_monitor.plot(out / "objectives.png", title="Single-stage objective terms")
        if args.movie:
            def objects_from_x(x):
                return accepted_frame(frame_steps[tuple(x)])[1:]

            def movie_colors(x, objects):
                state = accepted_frame(frame_steps[tuple(x)])[0]
                data = vj.surface_field_data_from_state(inp, state, runtime=initial_equilibrium.runtime,
                                                       nphi=case.NPHI, ntheta=case.NTHETA)
                magnitude = jnp.linalg.norm(data.B_total, axis=0)
                if case.MOVIE_SURFACE_COLOR == "absB":
                    return magnitude
                interface = vj.PlasmaVacuumInterface.from_surface_data(data, digits=4)
                return interface.bnormal_residual(chart(jnp.asarray(x)))/magnitude

            post_monitor.movie(out / "optimization.gif", objects_from_x,
                color_factory=movie_colors if case.MOVIE_SURFACE_COLOR is not None else None,
                color_label=case.MOVIE_SURFACE_COLOR, cmap="jet")
        for path in vj.plot_wout(wout_path, out).values():
            print(f"Wrote {path}")
    summary.update(postprocessing="complete", postprocessing_seconds=time.perf_counter()-postprocessing_started)
    write_json(out / "optimization_summary.json", summary)
