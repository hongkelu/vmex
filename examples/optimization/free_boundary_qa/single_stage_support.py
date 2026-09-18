"""Input authentication, checkpointing, diagnostics and run lifecycle.

The public FreeBoundaryProblem owns numerical evaluation and promotion.
This module owns case loading, checkpoint files, diagnostics and run lifetime.
"""

import argparse
import dataclasses
import json
import os
import signal
import time
from pathlib import Path

from adjoint_batch import BATCH_SIZES, tune_adjoint_batch


def source_commit(source):
    """Record the actual checkout revision; source hashes capture local edits."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv=None, *, target_step, device, max_wall_hours, batch_size=32,
               output_dir=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="48x40 free-boundary QA optimization")
    parser.add_argument("--target-step", type=int, default=target_step)
    parser.add_argument(
        "--initialize-only", action="store_true", help="prepare and certify step zero without optimization"
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir, required=output_dir is None)
    parser.add_argument("--device", default=device)
    parser.add_argument("--max-wall-hours", type=float, default=max_wall_hours)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    # Every batch choice retains the maintained fast predictor and its gates.
    parser.add_argument("--equilibrium-predictor", choices=("reused_dense",), default="reused_dense")
    parser.add_argument("--adjoint-dense-batch-size", type=lambda v: v if v == 'auto' else int(v),
                        choices=('auto', *BATCH_SIZES), default=batch_size,
                        help="32 (default) or 64 skips tuning; auto opts into a startup timing comparison")
    args = parser.parse_args(argv)
    if bool(args.resume_checkpoint) != bool(args.checkpoint_sha256):
        parser.error("resume checkpoint and expected SHA256 must be supplied together")
    if not 1 <= args.target_step <= 3000 or not 0 < args.max_wall_hours <= 24:
        parser.error("bounded to absolute step 3000 and at most 24 hours")
    if args.initialize_only and args.resume_checkpoint:
        parser.error("initialize-only cannot resume")
    return args


class FreeBoundaryRun:
    """Own one run and its accepted equilibrium; trial states stay separate."""

    def __init__(self, args):
        if args.equilibrium_predictor != 'reused_dense' or args.adjoint_dense_batch_size not in ('auto', *BATCH_SIZES):
            raise ValueError('this workflow requires reused_dense prediction and adjoint batch auto, 32, or 64')
        self.args = args
        self.adjoint_batch_size = BATCH_SIZES[0] if args.adjoint_dense_batch_size == 'auto' else args.adjoint_dense_batch_size
        self.batch_tuning_done = args.adjoint_dense_batch_size != 'auto'
        self.output = self.args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        for key, name in [
            ("JAX_COMPILATION_CACHE_DIR", "jax"),
            ("MPLCONFIGDIR", "mpl"),
            ("XDG_CACHE_HOME", "xdg"),
            ("TMPDIR", "tmp"),
            ("CUDA_CACHE_PATH", "cuda"),
        ]:
            p = self.output / "cache" / name
            p.mkdir(parents=True)
            os.environ[key] = str(p)
        os.environ.update(JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1")
        self.started = time.monotonic()
        self.deadline = self.started + self.args.max_wall_hours * 3600
        self.diagnostics_writer = None
        self.accepted = self.rows = self.targets = self.last_stage = None
        self.step = self.initial_step = 0
        self.status = "initializing"
        self.loss_scale = None
        self.resume = None
        self.initial_gradient_norm = None
        self.last_gradient = None
        self.last_gradient_step = None
        self.inp = self.builder = self.seed = self.contract = None
        self.input_hashes = self.policy = self.params = self.rt = None
        self.solver = self.cfg = self.values = self.provenance = None
        self.jac = self.point = None
        self.linearization = None

    def __enter__(self):
        self._signal_handlers = {
            sig: signal.signal(sig, self._stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            self.setup_problem()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self.status = "failed"
                self.event("failure", error_type=type(exc).__name__, error=str(exc))
            self.save_results()
        finally:
            self._close_linearization()
            for sig, handler in self._signal_handlers.items():
                signal.signal(sig, handler)
        return False

    @staticmethod
    def _stop(signum, frame):
        raise TimeoutError(f"signal {signum}")

    def write(self, p, obj):
        temp = p.with_suffix(p.suffix + ".tmp")
        temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        temp.replace(p)

    def event(self, phase, **values):
        item = dict(phase=phase, elapsed_s=time.monotonic() - self.started, **values)
        s = json.dumps(item, allow_nan=False)
        with (self.output / "progress.jsonl").open("a") as f:
            f.write(s + "\n")
        print(s, flush=True)

    def checkpoint(self):
        import numpy as np
        from case import CASE, STATE_NAMES, sha256
        p = self.output / f"checkpoint_step_{self.step:04d}.npz"
        data = dict(
            schema_version=np.asarray("vmex.iota02-aspect5-accepted/v1"),
            accepted_step=np.asarray(self.step),
            parameters=np.asarray(self.accepted.parameters),
            parameter_scales=self.parameter_scales,
            targets=self.targets,
            constraint_scales=self.constraint_scales,
            loss_scale=np.asarray(self.loss_scale),
            phiedge=np.asarray(self.inp.phiedge),
            rcon0=np.asarray(self.accepted.rcon0),
            zcon0=np.asarray(self.accepted.zcon0),
            contract_sha256=np.asarray(sha256(CASE / "case_contract.json")),
            provenance_json=np.asarray(json.dumps(self.provenance, sort_keys=True)),
        )
        data["optimizer_policy_sha256"] = np.asarray(sha256(CASE / "optimizer_policy.json"))
        if self.initial_gradient_norm is not None:
            data["initial_projected_gradient_norm"] = np.asarray(self.initial_gradient_norm)
        data.update({n: np.asarray(getattr(self.accepted.state, n)) for n in STATE_NAMES})
        data.update({"mask_" + n: np.asarray(getattr(self.accepted.dof_mask, n)) for n in STATE_NAMES})
        with p.open("xb") as f:
            np.savez_compressed(f, **data)
        record = dict(path=str(p), sha256=sha256(p), accepted_step=self.step)
        self.write(self.output / "latest_checkpoint.json", record)
        return record

    def export_state(self, coil_path, wout_path):
        import jax.numpy as jnp
        import numpy as np
        from vmex.core import freeboundary_implicit as fbi
        from vmex.core.wout import wout_from_state, write_wout
        from essos.coils import Coils, Curves

        currents = np.asarray(self.builder.base_currents_at(jnp.asarray(self.accepted.parameters)))
        # Re-evaluate the vacuum on this exact fixed plasma/coil geometry for
        # every exported pair. Imported anchors have no attached VacuumOutput;
        # ordinary results can carry cadence caches from a preceding geometry.
        from vmex.core.freeboundary import _vacuum_executables, _vacuum_output, FreeBoundaryState

        export_rt = dataclasses.replace(
            self.rt,
            rcon0=self.accepted.rcon0,
            zcon0=self.accepted.zcon0,
            lfreeb=True,
            jmax=int(self.solver.resolution.ns),
            presf_ns_scale=fbi._presf_ns_scale_traceable(self.params, self.inp, int(self.solver.resolution.ns)),
        )
        axis_r = jnp.full((self.solver.resolution.nzeta,), float(np.asarray(self.inp.rbc)[self.inp.ntor, 0]))
        basis, program, _ = _vacuum_executables(
            self.solver.resolution,
            mf=int(self.inp.mpol) + 1,
            nf=int(self.inp.ntor),
            signgs=int(self.rt.setup.signgs),
            wint=np.asarray(self.rt.trig.wint),
            modes=self.rt.modes,
            axis_r0=axis_r,
            axis_z0=jnp.zeros_like(axis_r),
            use_fft=False,
            solve_on_plasma_device=True,
        )
        vacuum_values = program.full(self.accepted.state, export_rt, self.builder(jnp.asarray(self.accepted.parameters)))
        vacuum = _vacuum_output(
            FreeBoundaryState(potvac=vacuum_values["potvac"], surface_fields=vacuum_values["surface_fields"]), basis
        )
        if vacuum is None or any(
            not np.all(np.isfinite(np.asarray(x)))
            for x in (vacuum.potsin, vacuum.bsubu, vacuum.bsubv, vacuum.bsupu, vacuum.bsupv)
        ):
            raise ValueError("snapshot requires finite fixed-geometry vacuum output")
        w = wout_from_state(
            inp=self.inp,
            state=self.accepted.state,
            niter=self.accepted.result.iterations,
            fsqr=self.accepted.result.fsqr,
            fsqz=self.accepted.result.fsqz,
            fsql=self.accepted.result.fsql,
            vacuum_output=vacuum,
            nextcur=len(currents),
            extcur=currents,
            curlabel=tuple(f"base_coil_{i}" for i in range(len(currents))),
        )
        write_wout(wout_path, w, overwrite=False)
        Coils(Curves(self.builder.curve_dofs_at(jnp.asarray(self.accepted.parameters)), 75, 2, True), currents).to_json(
            str(coil_path)
        )

    def record_diagnostics(self):
        from case import metrics
        self.diagnostics_writer.record(
            step=self.step,
            elapsed_s=time.monotonic() - self.started,
            metrics=metrics(self.values, self.targets, self.loss_scale),
            root_residual=self.accepted.root_residual_norm,
            forces={n: float(getattr(self.accepted.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
            checkpoint=json.loads((self.output / "latest_checkpoint.json").read_text()),
            export=self.export_state,
        )
        if self.step == 0 or self.step % 20 == 0:
            self.event("diagnostic_snapshot", absolute_step=self.step, path=str(self.output / "diagnostics" / f"step_{self.step:04d}"))


    def _close_linearization(self):
        """Release factors at the same lifecycle points for every entry point."""
        if self.linearization is not None:
            self.linearization.close()
            self.linearization = None


    def _state_linearization(self, cfg, diagnostics):
        """Use the same certified gradient path for tuning and normal steps."""
        import jax
        from vmex.core import freeboundary_continuation as fc

        # Differentiate each row with respect to the equilibrium state, then
        # pull those derivatives back to the 111 coil/current parameters.
        # Retain the same dense factors for the trial's tangent prediction.
        rhs = jax.jacrev(self.rows)(self.accepted.state)
        return fc.free_boundary_continuation_state_pullback(
            self.accepted, cfg, rhs, diagnostics=diagnostics, return_linearization=True
        )


    def _tune_adjoint_batch(self, diagnostics):
        """Select a batch once, retaining its solver identity and factors."""
        # Residual/JIT caches depend on solver identity. Reuse the
        # existing baseline instead of constructing an equivalent one.
        configs = {batch: self.cfg if batch == self.cfg.solver.adjoint_dense_batch_size
                   else dataclasses.replace(self.cfg, solver=dataclasses.replace(
                       self.cfg.solver, adjoint_dense_batch_size=batch)) for batch in BATCH_SIZES}
        batch_diagnostics = {}

        def evaluate(batch):
            if time.monotonic() >= self.deadline:
                raise TimeoutError('walltime')
            rows = []
            root = self._state_linearization(configs[batch], rows)
            try:
                self.event('adjoint_batch_certification', batch=batch, rows=rows)
                batch_diagnostics[batch] = rows
            except BaseException:
                root.close()
                raise
            return root

        self.adjoint_batch_size, self.linearization, report = tune_adjoint_batch(
            evaluate, deadline=self.deadline, event=self.event)
        self.cfg = configs[self.adjoint_batch_size]
        diagnostics.extend(batch_diagnostics[self.adjoint_batch_size])
        self.solver = self.cfg.solver
        self.batch_tuning_done = True
        self.provenance.update(adjoint_dense_batch_size=self.adjoint_batch_size,
                               adjoint_batch_tuning=report)
        self.write(self.output / 'adjoint_batch_tuning.json', report)
        self.write(self.output / 'manifest.json', self.provenance)


    def has_converged(self, direction):
        from vmex.core.projected_optimization import converged
        if self.initial_gradient_norm is None:
            self.initial_gradient_norm = direction.projected_gradient_norm
        self.last_gradient = direction.projected_gradient_norm
        self.last_gradient_step = self.step
        is_converged, gradient_threshold = converged(direction, self.initial_gradient_norm, self.policy)
        self.event(
            "direction",
            absolute_step=self.step,
            mode=direction.mode,
            projected_gradient_norm=self.last_gradient,
            gradient_threshold=gradient_threshold,
            constraints=direction.constraint_info,
            restoring_weight=direction.restoring_weight,
            combined_cap_factor=direction.combined_cap_factor,
            predicted_constraint_change=direction.predicted_constraint_change,
            objective_directional_derivative=direction.objective_directional_derivative,
        )
        if is_converged:
            self.status = "converged"
            self.event(
                "converged",
                absolute_step=self.step,
                criterion="feasible and small scaled equality-tangent QA gradient",
                projected_gradient_norm=self.last_gradient,
                threshold=gradient_threshold,
            )
        return is_converged

    def record_trial(self, record):
        self.event("trial_decision", absolute_step=self.step + 1, **record)

    def report_stagnation(self, result):
        self.status = "stagnated"
        self.event(
            "stagnated",
            absolute_step=self.step,
            reason="no acceptable step within finite trial budget",
            trials=len(result.trials),
            projected_gradient_norm=self.last_gradient,
        )


    def save_results(self):
        import numpy as np
        from case import STATE_NAMES, metrics, sha256
        summary = dict(
            status=self.status,
            initial_projected_gradient_norm=self.initial_gradient_norm,
            last_projected_gradient_norm=self.last_gradient,
            gradient_evaluated_at_step=self.last_gradient_step,
            initial_step=self.initial_step,
            final_step=self.step,
            target_step=self.args.target_step,
            promoted_steps=self.step - self.initial_step,
            elapsed_s=time.monotonic() - self.started,
        )
        if self.last_stage is not None and self.status in ("failed", "stagnated"):
            p = self.output / "unaccepted_last_ordinary_state.npz"
            np.savez_compressed(
                p,
                parameters=np.asarray(self.point),
                rcon0=np.asarray(self.last_stage.rcon0),
                zcon0=np.asarray(self.last_stage.zcon0),
                **{n: np.asarray(getattr(self.last_stage.result.state, n)) for n in STATE_NAMES},
            )
            summary["unaccepted_diagnostic_state"] = dict(path=str(p), sha256=sha256(p), eligible_for_resume=False)
        if self.accepted is not None and self.rows is not None:
            summary.update(
                final_metrics=metrics(np.asarray(self.rows(self.accepted.state)), self.targets, self.loss_scale),
                root_residual=self.accepted.root_residual_norm,
                targets=self.targets.tolist(),
                final_checkpoint=json.loads((self.output / "latest_checkpoint.json").read_text()),
            )
            if self.diagnostics_writer is not None:
                try:
                    final_export = self.diagnostics_writer.snapshot(
                        step=self.step,
                        checkpoint=summary["final_checkpoint"],
                        metrics=summary["final_metrics"],
                        export=self.export_state,
                    )
                    summary["final_wout"] = final_export["files"]["wout.nc"]
                    summary["final_coils"] = final_export["files"]["coils.json"]
                    summary["metrics_csv"] = "metrics.csv"
                    summary["metrics_jsonl"] = "metrics.jsonl"
                except Exception as export_error:
                    summary["diagnostics_export_error"] = f"{type(export_error).__name__}: {export_error}"
                    summary["status"] = "failed"
        self.write(self.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


    def _load_inputs(self):
        import jax.numpy as jnp
        import numpy as np
        from case import CASE, STATE_NAMES, load_case, sha256
        from vmex.core.projected_optimization import ProjectedOptions
        from checkpoint import load_checkpoint
        from vmex.core.solver import SpectralState
        from case import PARAMETER_SCALES, CONSTRAINT_SCALES
        self.parameter_scales = PARAMETER_SCALES
        self.constraint_scales = CONSTRAINT_SCALES
        self.inp, self.builder, self.seed, self.contract, self.input_hashes = load_case()
        self.policy = ProjectedOptions(**json.loads((CASE / "optimizer_policy.json").read_text()))
        if sha256(CASE / "optimizer_policy.json") != self.contract["optimizer_policy_sha256"]:
            raise ValueError("optimizer policy hash mismatch")
        self.inp = dataclasses.replace(
            self.inp,
            ftol_array=np.asarray([self.contract["force_tolerance"]]),
            niter_array=np.asarray([self.contract["max_iterations"]]),
        )
        if self.args.resume_checkpoint:
            self.resume = load_checkpoint(
                self.args.resume_checkpoint,
                self.args.checkpoint_sha256,
                contract_sha256=sha256(CASE / "case_contract.json"),
                input_hashes=self.input_hashes,
                parameter_scales=self.parameter_scales,
                constraint_scales=self.constraint_scales,
                phiedge=self.inp.phiedge,
                target_step=self.args.target_step,
                previous_contract_path=CASE / "resume_previous_contract.json",
                current_contract=self.contract,
                migration_path=CASE / "contract_migration.json",
            )
            self.initial_step = self.step = int(self.resume["accepted_step"])
            self.seed = SpectralState(**{n: jnp.asarray(self.resume[n]) for n in STATE_NAMES})
            self.targets = self.resume["targets"].copy()
            self.loss_scale = float(self.resume["loss_scale"])
            # This migration retains the projected-gradient definition and its
            # original reference; changing the proposal must not reset the test.
            if "initial_projected_gradient_norm" in self.resume:
                self.initial_gradient_norm = float(self.resume["initial_projected_gradient_norm"])
                if not np.isfinite(self.initial_gradient_norm) or self.initial_gradient_norm <= 0:
                    raise ValueError("invalid saved projected-gradient reference")
        p = np.zeros(111) if self.resume is None else self.resume["parameters"].copy()
        return p

    def _record_provenance(self, device):
        import jax
        import solvax
        from case import CASE, sha256
        import inspect
        import sys
        from essos.coils import Coils

        source = CASE.parents[2]
        self.provenance = dict(
            equilibrium_predictor=self.args.equilibrium_predictor,
            adjoint_dense_batch_size=self.adjoint_batch_size,
            adjoint_dense_batch_requested=self.args.adjoint_dense_batch_size,
            diagnostics=dict(
                export_interval=20, initial_snapshot=True, final_snapshot=True, history=["metrics.csv", "metrics.jsonl"]
            ),
            contract=self.contract,
            optimizer_policy=dataclasses.asdict(self.policy),
            optimizer_policy_sha256=sha256(CASE / "optimizer_policy.json"),
            input_hashes=self.input_hashes,
            source_commit=source_commit(source),
            entry_point="single_stage_free_boundary_optimization.py",
            jax=jax.__version__,
            solvax=solvax.__version__,
            python=sys.version,
            device=str(device),
            essos_coils_sha256=sha256(inspect.getsourcefile(Coils)),
            initial_step=self.initial_step,
            target_step=self.args.target_step,
            max_wall_hours=self.args.max_wall_hours,
            resume_checkpoint=None
            if self.resume is None
            else dict(
                path=str(self.args.resume_checkpoint.resolve()),
                sha256=self.args.checkpoint_sha256,
                absolute_step=self.initial_step,
                previous_contract_sha256=str(self.resume["contract_sha256"]),
                current_contract_sha256=sha256(CASE / "case_contract.json"),
                contract_changed=str(self.resume["contract_sha256"]) != sha256(CASE / "case_contract.json"),
            ),
            source_sha256={
                str(f.relative_to(source)): sha256(f)
                for folder in (source / "vmex", CASE)
                for f in folder.rglob("*.py")
                if "runs" not in f.parts
            },
        )
        self.write(self.output / "manifest.json", self.provenance)
        self.inp.to_indata(self.output / "input.effective")

    def _check_initial_equilibrium(self, initial_stage):
        import jax
        import numpy as np
        from case import STATE_NAMES
        if self.resume is None:
            ordinary_edge = float(initial_stage.result.fedge)
            fresh_edge = float(self.accepted.result.fedge)
            agreement = bool(np.isclose(ordinary_edge, fresh_edge, rtol=1e-5, atol=1e-15))
            self.event(
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
            if not np.array_equal(np.asarray(getattr(self.seed, n)), np.asarray(getattr(self.accepted.state, n))):
                raise RuntimeError("initial certification changed state")

    def _record_initial_state(self, start_physical):
        import numpy as np
        from case import metrics
        from diagnostics import Diagnostics
        self.write(
            self.output / "resolved_targets.json",
            dict(
                mean_iota=float(self.targets[0]),
                aspect_ratio=float(self.targets[1]),
                signed_b0_T=float(self.targets[2]),
                start_step=self.initial_step,
                start_physical=dict(zip(("mean_iota", "aspect_ratio", "b0"), start_physical.tolist())),
                targets_source="fixed original targets; QA normalization from refined initial equilibrium"
                if self.resume is None
                else "authenticated checkpoint; original targets and normalization preserved",
                constraint_scales=self.constraint_scales.tolist(),
                constraint_tolerances=(self.policy.constraint_relative_tolerance * np.abs(self.targets)).tolist(),
                loss_scale=self.loss_scale,
            ),
        )
        self.event(
            "initialized",
            absolute_step=self.step,
            metrics=metrics(self.values, self.targets, self.loss_scale),
            root_residual=self.accepted.root_residual_norm,
            forces={n: float(getattr(self.accepted.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
            checkpoint=self.checkpoint(),
        )
        self.diagnostics_writer = Diagnostics(
            self.output,
            targets=self.targets,
            tolerances=self.policy.constraint_relative_tolerance * np.abs(self.targets),
            export_interval=20,
        )
        self.record_diagnostics()

    def _record_proposal(self, delta, trial, count, name):
        import numpy as np
        from case import sampled_displacement
        np.savez_compressed(
            self.output / (name + "_proposal.npz"),
            parameters=self.accepted.parameters,
            delta=delta,
            rows=self.values,
            jacobian=self.jac,
        )
        motion, current = self.builder.motion_bounds(delta)
        self.event(
            "proposal",
            absolute_step=self.step + 1,
            trial=trial,
            points=count,
            maximum_coil_bound_m=motion,
            maximum_sampled_step_m=float(np.max(np.linalg.norm(sampled_displacement(delta), axis=-1))),
            maximum_current_fraction=current,
            predicted_row_change=(self.jac @ delta).tolist(),
        )

    def _record_ordinary_correction(self, trial, index, count, correction_started):
        self.event(
            "ordinary_correction",
            seconds=time.monotonic() - correction_started,
            equilibrium_predictor=self.args.equilibrium_predictor,
            absolute_step=self.step + 1,
            trial=trial,
            point=index,
            points=count,
            converged=bool(self.last_stage.result.converged),
            iterations=int(self.last_stage.result.iterations),
            forces={n: float(getattr(self.last_stage.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
        )

    def _record_certification(self, trial, index, previous):
        self.event(
            "certification",
            absolute_step=self.step + 1,
            trial=trial,
            point=index,
            root_residual=previous.root_residual_norm,
            forces={n: float(getattr(previous.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
        )

    def _record_candidate(self, name):
        import numpy as np
        from case import STATE_NAMES
        if self.last_stage is not None:
            np.savez_compressed(
                self.output / (name + "_candidate.npz"),
                parameters=np.asarray(self.point),
                rcon0=np.asarray(self.last_stage.rcon0),
                zcon0=np.asarray(self.last_stage.zcon0),
                eligible_for_resume=np.asarray(False),
                **{n: np.asarray(getattr(self.last_stage.result.state, n)) for n in STATE_NAMES},
            )
