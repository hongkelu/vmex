#!/usr/bin/env python3
"""Dense-JAX QA case with frozen iota/aspect/B0 targets and exact checkpoint resume."""

import argparse
import dataclasses
import json
import os
import signal
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-step", type=int, default=3000)
    parser.add_argument(
        "--initialize-only", action="store_true", help="prepare and certify step zero without optimization"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-wall-hours", type=float, default=1.0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    if bool(args.resume_checkpoint) != bool(args.checkpoint_sha256):
        parser.error("resume checkpoint and expected SHA256 must be supplied together")
    if not 1 <= args.target_step <= 3000 or not 0 < args.max_wall_hours <= 24:
        parser.error("bounded to absolute step 3000 and at most 24 hours")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key, name in [
        ("JAX_COMPILATION_CACHE_DIR", "jax"),
        ("MPLCONFIGDIR", "mpl"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TMPDIR", "tmp"),
        ("CUDA_CACHE_PATH", "cuda"),
    ]:
        p = output / "cache" / name
        p.mkdir(parents=True)
        os.environ[key] = str(p)
    os.environ.update(JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import solvax
    from vmex.core import implicit as im, freeboundary_implicit as fbi, freeboundary_continuation as fc
    from vmex.core.freeboundary import _solve_free_boundary_stage
    from vmex.core.wout import wout_from_state, write_wout
    from case import (
        CASE,
        STATE_NAMES,
        PARAMETER_SCALES,
        CONSTRAINT_SCALES,
        load_case,
        physical_rows,
        resolve_targets,
        make_rows,
        metrics,
        sha256,
    )
    from optimization import Policy, proposal, displacement, motion_bounds, converged, backtrack, TrialRejected
    from vmex.core.errors import VmecError
    from checkpoint import load_checkpoint, verify_restoration
    from diagnostics import Diagnostics
    from vmex.core.solver import SpectralState

    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600
    diagnostics_writer = None
    accepted = rows = targets = last_stage = None
    step = initial_step = 0
    status = "initializing"
    loss_scale = None
    resume = None
    initial_gradient_norm = None
    last_gradient = None
    last_gradient_step = None

    def stop(signum, frame):
        raise TimeoutError(f"signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def write(p, obj):
        temp = p.with_suffix(p.suffix + ".tmp")
        temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        temp.replace(p)

    def event(phase, **values):
        item = dict(phase=phase, elapsed_s=time.monotonic() - started, **values)
        s = json.dumps(item, allow_nan=False)
        with (output / "progress.jsonl").open("a") as f:
            f.write(s + "\n")
        print(s, flush=True)

    def checkpoint():
        p = output / f"checkpoint_step_{step:04d}.npz"
        data = dict(
            schema_version=np.asarray("vmex.iota02-aspect5-accepted/v1"),
            accepted_step=np.asarray(step),
            parameters=np.asarray(accepted.parameters),
            parameter_scales=PARAMETER_SCALES,
            targets=targets,
            constraint_scales=CONSTRAINT_SCALES,
            loss_scale=np.asarray(loss_scale),
            phiedge=np.asarray(inp.phiedge),
            rcon0=np.asarray(accepted.rcon0),
            zcon0=np.asarray(accepted.zcon0),
            contract_sha256=np.asarray(sha256(CASE / "case_contract.json")),
            provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        data["optimizer_policy_sha256"] = np.asarray(sha256(CASE / "optimizer_policy.json"))
        if initial_gradient_norm is not None:
            data["initial_projected_gradient_norm"] = np.asarray(initial_gradient_norm)
        data.update({n: np.asarray(getattr(accepted.state, n)) for n in STATE_NAMES})
        data.update({"mask_" + n: np.asarray(getattr(accepted.dof_mask, n)) for n in STATE_NAMES})
        with p.open("xb") as f:
            np.savez_compressed(f, **data)
        record = dict(path=str(p), sha256=sha256(p), accepted_step=step)
        write(output / "latest_checkpoint.json", record)
        return record

    def export_state(coil_path, wout_path):
        from essos.coils import Coils, Curves

        currents = np.asarray(builder.base_currents_at(jnp.asarray(accepted.parameters)))
        # Re-evaluate the vacuum on this exact fixed plasma/coil geometry for
        # every exported pair. Imported anchors have no attached VacuumOutput;
        # ordinary results can carry cadence caches from a preceding geometry.
        from vmex.core.freeboundary import _vacuum_executables, _vacuum_output, FreeBoundaryState

        export_rt = dataclasses.replace(
            rt,
            rcon0=accepted.rcon0,
            zcon0=accepted.zcon0,
            lfreeb=True,
            jmax=int(solver.resolution.ns),
            presf_ns_scale=fbi._presf_ns_scale_traceable(params, inp, int(solver.resolution.ns)),
        )
        axis_r = jnp.full((solver.resolution.nzeta,), float(np.asarray(inp.rbc)[inp.ntor, 0]))
        basis, program, _ = _vacuum_executables(
            solver.resolution,
            mf=int(inp.mpol) + 1,
            nf=int(inp.ntor),
            signgs=int(rt.setup.signgs),
            wint=np.asarray(rt.trig.wint),
            modes=rt.modes,
            axis_r0=axis_r,
            axis_z0=jnp.zeros_like(axis_r),
            use_fft=False,
            solve_on_plasma_device=True,
        )
        vacuum_values = program.full(accepted.state, export_rt, builder(jnp.asarray(accepted.parameters)))
        vacuum = _vacuum_output(
            FreeBoundaryState(potvac=vacuum_values["potvac"], surface_fields=vacuum_values["surface_fields"]), basis
        )
        if vacuum is None or any(
            not np.all(np.isfinite(np.asarray(x)))
            for x in (vacuum.potsin, vacuum.bsubu, vacuum.bsubv, vacuum.bsupu, vacuum.bsupv)
        ):
            raise ValueError("snapshot requires finite fixed-geometry vacuum output")
        w = wout_from_state(
            inp=inp,
            state=accepted.state,
            niter=accepted.result.iterations,
            fsqr=accepted.result.fsqr,
            fsqz=accepted.result.fsqz,
            fsql=accepted.result.fsql,
            vacuum_output=vacuum,
            nextcur=len(currents),
            extcur=currents,
            curlabel=tuple(f"base_coil_{i}" for i in range(len(currents))),
        )
        write_wout(wout_path, w, overwrite=False)
        Coils(Curves(builder.curve_dofs_at(jnp.asarray(accepted.parameters)), 75, 2, True), currents).to_json(
            str(coil_path)
        )

    def record_diagnostics():
        diagnostics_writer.record(
            step=step,
            elapsed_s=time.monotonic() - started,
            metrics=metrics(values, targets, loss_scale),
            root_residual=accepted.root_residual_norm,
            forces={n: float(getattr(accepted.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
            checkpoint=json.loads((output / "latest_checkpoint.json").read_text()),
            export=export_state,
        )
        if step == 0 or step % 20 == 0:
            event("diagnostic_snapshot", absolute_step=step, path=str(output / "diagnostics" / f"step_{step:04d}"))

    try:
        inp, builder, seed, contract, input_hashes = load_case()
        policy = Policy(**json.loads((CASE / "optimizer_policy.json").read_text()))
        if sha256(CASE / "optimizer_policy.json") != contract["optimizer_policy_sha256"]:
            raise ValueError("optimizer policy hash mismatch")
        inp = dataclasses.replace(
            inp,
            ftol_array=np.asarray([contract["force_tolerance"]]),
            niter_array=np.asarray([contract["max_iterations"]]),
        )
        if args.resume_checkpoint:
            resume = load_checkpoint(
                args.resume_checkpoint,
                args.checkpoint_sha256,
                contract_sha256=sha256(CASE / "case_contract.json"),
                input_hashes=input_hashes,
                parameter_scales=PARAMETER_SCALES,
                constraint_scales=CONSTRAINT_SCALES,
                phiedge=inp.phiedge,
                target_step=args.target_step,
                previous_contract_path=CASE / "resume_previous_contract.json",
                current_contract=contract,
                migration_path=CASE / "contract_migration.json",
            )
            initial_step = step = int(resume["accepted_step"])
            seed = SpectralState(**{n: jnp.asarray(resume[n]) for n in STATE_NAMES})
            targets = resume["targets"].copy()
            loss_scale = float(resume["loss_scale"])
            if (
                str(resume.get("optimizer_policy_sha256", "")) == sha256(CASE / "optimizer_policy.json")
                and "initial_projected_gradient_norm" in resume
            ):
                initial_gradient_norm = float(resume["initial_projected_gradient_norm"])
        p = np.zeros(111) if resume is None else resume["parameters"].copy()
        device = jax.devices("gpu" if args.device.startswith("cuda") else "cpu")[0]
        import inspect
        import sys
        from essos.coils import Coils

        source = CASE.parents[1]
        provenance = dict(
            diagnostics=dict(
                export_interval=20, initial_snapshot=True, final_snapshot=True, history=["metrics.csv", "metrics.jsonl"]
            ),
            contract=contract,
            optimizer_policy=dataclasses.asdict(policy),
            optimizer_policy_sha256=sha256(CASE / "optimizer_policy.json"),
            input_hashes=input_hashes,
            main_commit="0bc787b72917371ab122f76e870e1cf6ab141102",
            jax=jax.__version__,
            solvax=solvax.__version__,
            python=sys.version,
            device=str(device),
            essos_coils_sha256=sha256(inspect.getsourcefile(Coils)),
            initial_step=initial_step,
            target_step=args.target_step,
            max_wall_hours=args.max_wall_hours,
            resume_checkpoint=None
            if resume is None
            else dict(
                path=str(args.resume_checkpoint.resolve()),
                sha256=args.checkpoint_sha256,
                absolute_step=initial_step,
                previous_contract_sha256=str(resume["contract_sha256"]),
                current_contract_sha256=sha256(CASE / "case_contract.json"),
                contract_changed=str(resume["contract_sha256"]) != sha256(CASE / "case_contract.json"),
            ),
            source_sha256={
                str(f.relative_to(source)): sha256(f)
                for folder in (source / "vmex", CASE)
                for f in folder.rglob("*.py")
                if "runs" not in f.parts
            },
        )
        write(output / "manifest.json", provenance)
        inp.to_indata(output / "input.effective")
        solver = fbi.make_free_boundary_config(
            inp,
            builder(jnp.asarray(p)),
            field_from_parameters=builder,
            device=device,
            ftol=contract["force_tolerance"],
            max_iterations=contract["max_iterations"],
            adjoint_solver=contract["adjoint_solver"],
            adjoint_tol=contract["adjoint_tol"],
            adjoint_residual_rtol=contract["adjoint_residual_rtol"],
            adjoint_dense_batch_size=4,
            adjoint_dense_max_dofs=4096,
            adjoint_gcrot_m=30,
            adjoint_gcrot_k=10,
            adjoint_maxiter=100,
            adjoint_fail="error",
            include_edge_in_convergence=True,
            edge_force_tolerance=contract["edge_force_tolerance"],
        )
        params = im.params_from_input(inp)
        rt = im.runtime_from_params(params, solver.implicit)
        if args.initialize_only and resume is not None:
            raise ValueError("initialize-only cannot resume")
        initial_solves = 0
        initial_rcon = jnp.zeros_like(rt.rcon0) if resume is None else jnp.asarray(resume["rcon0"])
        initial_zcon = jnp.zeros_like(rt.zcon0) if resume is None else jnp.asarray(resume["zcon0"])
        if resume is None:
            event(
                "initial_ordinary_solve_start", ntheta=48, nzeta=40, ns=31, root_polishing=False, parameters_zero=True
            )
            initial_stage = _solve_free_boundary_stage(
                inp,
                external_field=builder(jnp.asarray(p)),
                resolution=solver.resolution,
                ftol=contract["initial_ordinary_ftol"],
                max_iterations=contract["max_iterations"],
                initial_state=seed,
                include_edge_in_convergence=True,
                edge_force_tolerance=contract["initial_ordinary_edge_tolerance"],
                error_on_no_convergence=False,
                jacobian_retries=0,
                allow_initial_axis_reguess=False,
                use_fft=False,
            )
            initial_solves = 1
            event(
                "initial_ordinary_solve_complete",
                converged=bool(initial_stage.result.converged),
                iterations=int(initial_stage.result.iterations),
                forces={n: float(getattr(initial_stage.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
            )
            if not initial_stage.result.converged:
                raise RuntimeError("initial 48x40 ordinary equilibrium did not converge")
            # continuation_state is the restart buffer (xstore), not the
            # converged state whose residuals the ordinary solve reports.
            seed = initial_stage.result.state
            initial_rcon, initial_zcon = initial_stage.rcon0, initial_stage.zcon0
            provenance["initial_equilibrium_solves_executed"] = initial_solves
            provenance["initial_ordinary_iterations"] = int(initial_stage.result.iterations)
            write(output / "manifest.json", provenance)
        event(
            "initial_certification_start",
            absolute_step=step,
            parameters_zero=bool(np.all(p == 0)),
            initial_equilibrium_solves=initial_solves,
            resumed=resume is not None,
        )
        cfg = fc.make_free_boundary_continuation_config_from_state(
            solver,
            params,
            p,
            state=seed,
            rcon0=initial_rcon,
            zcon0=initial_zcon,
            parameter_scales=PARAMETER_SCALES,
            continuation_step=0.1,
            max_continuation_steps=contract["max_continuation_steps"],
            root_residual_atol=contract["root_residual_atol"],
        )
        accepted = cfg._anchor
        if resume is None:
            ordinary_edge = float(initial_stage.result.fedge)
            fresh_edge = float(accepted.result.fedge)
            agreement = bool(np.isclose(ordinary_edge, fresh_edge, rtol=1e-5, atol=1e-15))
            event(
                "initial_edge_residual_agreement",
                ordinary_fedge=ordinary_edge,
                fresh_fedge=fresh_edge,
                rtol=1e-5,
                atol=1e-15,
                passed=agreement,
                certified_state="ordinary_result.state",
                result_vs_restart_buffer_max_abs=max(
                    float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                    for a, b in zip(
                        jax.tree.leaves(initial_stage.result.state), jax.tree.leaves(initial_stage.continuation_state)
                    )
                ),
            )
            if not agreement:
                raise RuntimeError("ordinary/fresh edge residual disagreement")
        for n in STATE_NAMES:
            if not np.array_equal(np.asarray(getattr(seed, n)), np.asarray(getattr(accepted.state, n))):
                raise RuntimeError("initial certification changed state")
        start_physical = np.asarray(physical_rows(accepted.state, rt))
        if resume is None:
            targets = resolve_targets(start_physical)
            loss_scale = max(float(make_rows(rt, jnp.asarray(targets), 1.0)(accepted.state)[0]), 0.001)
        else:
            verify_restoration(accepted, resume)
            event(
                "checkpoint_restored_exactly",
                absolute_step=step,
                sha256=args.checkpoint_sha256,
                targets=targets.tolist(),
                loss_scale=loss_scale,
                initial_equilibrium_solves=0,
            )
        rows = make_rows(rt, jnp.asarray(targets), loss_scale)
        values = np.asarray(rows(accepted.state))
        write(
            output / "resolved_targets.json",
            dict(
                mean_iota=float(targets[0]),
                aspect_ratio=float(targets[1]),
                signed_b0_T=float(targets[2]),
                start_step=initial_step,
                start_physical=dict(zip(("mean_iota", "aspect_ratio", "b0"), start_physical.tolist())),
                targets_source="fixed original targets; QA normalization from refined initial equilibrium"
                if resume is None
                else "authenticated checkpoint; original targets and normalization preserved",
                constraint_scales=CONSTRAINT_SCALES.tolist(),
                constraint_tolerances=(policy.constraint_relative_tolerance * np.abs(targets)).tolist(),
                loss_scale=loss_scale,
            ),
        )
        event(
            "initialized",
            absolute_step=step,
            metrics=metrics(values, targets, loss_scale),
            root_residual=accepted.root_residual_norm,
            forces={n: float(getattr(accepted.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
            checkpoint=checkpoint(),
        )
        diagnostics_writer = Diagnostics(
            output,
            targets=targets,
            tolerances=policy.constraint_relative_tolerance * np.abs(targets),
            export_interval=20,
        )
        record_diagnostics()
        if args.initialize_only:
            status = "initialized_only"
            return
        while step < args.target_step:
            if time.monotonic() >= deadline:
                raise TimeoutError("walltime")
            diagnostics = []
            t = time.monotonic()
            event("adjoint_start", absolute_step=step)
            try:
                rhs = jax.jacrev(rows)(accepted.state)
                jac = np.asarray(
                    fc.free_boundary_continuation_state_pullback(accepted, cfg, rhs, diagnostics=diagnostics)
                )
            finally:
                event("adjoint", absolute_step=step, seconds=time.monotonic() - t, rows=diagnostics)
            direction = proposal(values, jac, PARAMETER_SCALES, targets, CONSTRAINT_SCALES, policy)
            if initial_gradient_norm is None:
                initial_gradient_norm = direction.projected_gradient_norm
            last_gradient = direction.projected_gradient_norm
            last_gradient_step = step
            is_converged, gradient_threshold = converged(direction, initial_gradient_norm, policy)
            event(
                "direction",
                absolute_step=step,
                mode=direction.mode,
                projected_gradient_norm=last_gradient,
                gradient_threshold=gradient_threshold,
                constraints=direction.constraint_info,
            )
            if is_converged:
                status = "converged"
                event(
                    "converged",
                    absolute_step=step,
                    criterion="feasible and small scaled equality-tangent QA gradient",
                    projected_gradient_norm=last_gradient,
                    threshold=gradient_threshold,
                )
                break

            def evaluate_trial(delta, trial):
                nonlocal last_stage, point
                last_stage = None
                if time.monotonic() >= deadline:
                    raise TimeoutError("walltime")
                count = max(1, int(np.ceil(np.max(np.abs(delta / PARAMETER_SCALES)) / 0.1)))
                name = f"trial_step_{step + 1:04d}_trial_{trial:02d}"
                np.savez_compressed(
                    output / (name + "_proposal.npz"),
                    parameters=accepted.parameters,
                    delta=delta,
                    rows=values,
                    jacobian=jac,
                )
                motion, current = motion_bounds(delta)
                event(
                    "proposal",
                    absolute_step=step + 1,
                    trial=trial,
                    points=count,
                    maximum_coil_bound_m=motion,
                    maximum_sampled_step_m=float(np.max(np.linalg.norm(displacement(delta), axis=-1))),
                    maximum_current_fraction=current,
                    predicted_row_change=(jac @ delta).tolist(),
                )
                if count > contract["max_continuation_steps"]:
                    raise TrialRejected("continuation budget exceeded")
                try:
                    diagnostics = []
                    t = time.monotonic()
                    try:
                        tangent = fbi.free_boundary_state_tangent(
                            cfg.params,
                            jnp.asarray(accepted.parameters),
                            solver,
                            accepted.state,
                            accepted.dof_mask,
                            jnp.asarray(delta),
                            rcon0=accepted.rcon0,
                            zcon0=accepted.zcon0,
                            diagnostics=diagnostics,
                        )
                    finally:
                        event(
                            "tangent",
                            absolute_step=step + 1,
                            trial=trial,
                            seconds=time.monotonic() - t,
                            rows=diagnostics,
                        )
                    previous = accepted
                    for index in range(1, count + 1):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("walltime")
                        point = accepted.parameters + delta * (index / count)
                        predicted = jax.tree.map(lambda x, dx: x + dx / count, previous.state, tangent)
                        last_stage = _solve_free_boundary_stage(
                            inp,
                            external_field=builder(jnp.asarray(point)),
                            resolution=solver.resolution,
                            ftol=contract["force_tolerance"],
                            max_iterations=contract["max_iterations"],
                            initial_state=predicted,
                            constraint_continuation=(previous.rcon0, previous.zcon0),
                            include_edge_in_convergence=True,
                            edge_force_tolerance=contract["edge_force_tolerance"],
                            error_on_no_convergence=False,
                            jacobian_retries=0,
                            allow_initial_axis_reguess=False,
                            use_fft=False,
                        )
                        event(
                            "ordinary_correction",
                            absolute_step=step + 1,
                            trial=trial,
                            point=index,
                            points=count,
                            converged=bool(last_stage.result.converged),
                            iterations=int(last_stage.result.iterations),
                            forces={n: float(getattr(last_stage.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
                        )
                        if not last_stage.result.converged:
                            raise TrialRejected("ordinary equilibrium did not converge")
                        previous = fc.certify_free_boundary_continuation_state(
                            cfg,
                            point,
                            last_stage.result.state,
                            rcon0=last_stage.rcon0,
                            zcon0=last_stage.zcon0,
                            result=last_stage.result,
                        )
                        event(
                            "certification",
                            absolute_step=step + 1,
                            trial=trial,
                            point=index,
                            root_residual=previous.root_residual_norm,
                            forces={n: float(getattr(previous.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
                        )
                    return previous, np.asarray(rows(previous.state))
                except VmecError as exc:
                    raise TrialRejected(f"{type(exc).__name__}: {exc}") from exc
                finally:
                    if last_stage is not None:
                        np.savez_compressed(
                            output / (name + "_candidate.npz"),
                            parameters=np.asarray(point),
                            rcon0=np.asarray(last_stage.rcon0),
                            zcon0=np.asarray(last_stage.zcon0),
                            eligible_for_resume=np.asarray(False),
                            **{n: np.asarray(getattr(last_stage.result.state, n)) for n in STATE_NAMES},
                        )

            def record_trial(record):
                event("trial_decision", absolute_step=step + 1, **record)

            # point is assigned inside the trial callback; rejected states never reanchor cfg.
            point = None
            result = backtrack(values, direction, targets, CONSTRAINT_SCALES, policy, evaluate_trial, record_trial)
            if result.candidate is None:
                status = "stagnated"
                event(
                    "stagnated",
                    absolute_step=step,
                    reason="no acceptable step within finite trial budget",
                    trials=len(result.trials),
                    projected_gradient_norm=last_gradient,
                )
                break
            cfg = fc.reanchor_free_boundary_continuation_config(cfg, result.candidate)
            accepted = cfg._anchor
            step += 1
            values = result.values
            last_stage = None
            event(
                "promoted",
                absolute_step=step,
                metrics=metrics(values, targets, loss_scale),
                root_residual=accepted.root_residual_norm,
                maximum_coil_bound_m=motion_bounds(result.delta)[0],
                maximum_current_fraction=motion_bounds(result.delta)[1],
                trial_count=len(result.trials),
                acceptance=result.trials[-1],
                forces={n: float(getattr(accepted.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
                checkpoint=checkpoint(),
            )
            record_diagnostics()
        else:
            status = "step_budget_reached"
    except BaseException as exc:
        status = "failed"
        event("failure", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary = dict(
            status=status,
            initial_projected_gradient_norm=initial_gradient_norm,
            last_projected_gradient_norm=last_gradient,
            gradient_evaluated_at_step=last_gradient_step,
            initial_step=initial_step,
            final_step=step,
            target_step=args.target_step,
            promoted_steps=step - initial_step,
            elapsed_s=time.monotonic() - started,
        )
        if last_stage is not None and status in ("failed", "stagnated"):
            p = output / "unaccepted_last_ordinary_state.npz"
            np.savez_compressed(
                p,
                parameters=np.asarray(point),
                rcon0=np.asarray(last_stage.rcon0),
                zcon0=np.asarray(last_stage.zcon0),
                **{n: np.asarray(getattr(last_stage.result.state, n)) for n in STATE_NAMES},
            )
            summary["unaccepted_diagnostic_state"] = dict(path=str(p), sha256=sha256(p), eligible_for_resume=False)
        if accepted is not None and rows is not None:
            summary.update(
                final_metrics=metrics(np.asarray(rows(accepted.state)), targets, loss_scale),
                root_residual=accepted.root_residual_norm,
                targets=targets.tolist(),
                final_checkpoint=json.loads((output / "latest_checkpoint.json").read_text()),
            )
            if diagnostics_writer is not None:
                try:
                    final_export = diagnostics_writer.snapshot(
                        step=step,
                        checkpoint=summary["final_checkpoint"],
                        metrics=summary["final_metrics"],
                        export=export_state,
                    )
                    summary["final_wout"] = final_export["files"]["wout.nc"]
                    summary["final_coils"] = final_export["files"]["coils.json"]
                    summary["metrics_csv"] = "metrics.csv"
                    summary["metrics_jsonl"] = "metrics.jsonl"
                except Exception as export_error:
                    summary["diagnostics_export_error"] = f"{type(export_error).__name__}: {export_error}"
                    summary["status"] = "failed"
        write(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
