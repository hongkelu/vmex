"""Authenticated case inputs and persistence around the public problem API.

The reusable problem owns numerical evaluation. This adapter translates its
observer events into the established checkpoint/diagnostic format.
"""

import time
from single_stage_support import FreeBoundaryRun, parse_args  # noqa: F401


class CaseRun(FreeBoundaryRun):
    """Prepare the original seed/checkpoint and record public-API iterations."""

    def setup_problem(self):
        """Load or restore the case, certify the equilibrium, and define QA rows."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from vmex.core import implicit as im, freeboundary_implicit as fbi, freeboundary_continuation as fc
        from vmex.core.freeboundary import _solve_free_boundary_stage

        p = self._load_inputs()
        platform, _, ordinal = self.args.device.partition(":")
        if platform not in ("cpu", "cuda", "gpu") or (ordinal and not ordinal.isdecimal()):
            raise ValueError("device must be cpu, cpu:N, cuda:N, or gpu:N")
        devices = jax.devices("cpu" if platform == "cpu" else "gpu")
        device = devices[int(ordinal or "0")]
        self._record_provenance(device)
        self.solver = fbi.make_free_boundary_config(
            self.inp,
            self.builder(jnp.asarray(p)),
            field_from_parameters=self.builder,
            device=device,
            ftol=self.contract["force_tolerance"],
            max_iterations=self.contract["max_iterations"],
            adjoint_solver=self.contract["adjoint_solver"],
            adjoint_tol=self.contract["adjoint_tol"],
            adjoint_residual_rtol=self.contract["adjoint_residual_rtol"],
            adjoint_dense_batch_size=self.adjoint_batch_size,
            adjoint_dense_max_dofs=4096,
            adjoint_gcrot_m=30,
            adjoint_gcrot_k=10,
            adjoint_maxiter=100,
            adjoint_fail="error",
            include_edge_in_convergence=True,
            edge_force_tolerance=self.contract["edge_force_tolerance"],
        )
        self.params = im.params_from_input(self.inp)
        self.rt = im.runtime_from_params(self.params, self.solver.implicit)
        initial_solves = 0
        initial_stage = None
        initial_rcon = jnp.zeros_like(self.rt.rcon0) if self.resume is None else jnp.asarray(self.resume["rcon0"])
        initial_zcon = jnp.zeros_like(self.rt.zcon0) if self.resume is None else jnp.asarray(self.resume["zcon0"])
        if self.resume is None:
            self.event(
                "initial_ordinary_solve_start",
                ntheta=48,
                nzeta=40,
                ns=31,
                root_polishing=False,
                parameters_zero=True,
            )
            initial_stage = _solve_free_boundary_stage(
                self.inp,
                external_field=self.builder(jnp.asarray(p)),
                resolution=self.solver.resolution,
                ftol=self.contract["initial_ordinary_ftol"],
                max_iterations=self.contract["max_iterations"],
                initial_state=self.seed,
                include_edge_in_convergence=True,
                edge_force_tolerance=self.contract["initial_ordinary_edge_tolerance"],
                error_on_no_convergence=False,
                jacobian_retries=0,
                allow_initial_axis_reguess=False,
                use_fft=False,
            )
            initial_solves = 1
            self.event(
                "initial_ordinary_solve_complete",
                converged=bool(initial_stage.result.converged),
                iterations=int(initial_stage.result.iterations),
                forces={n: float(getattr(initial_stage.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
            )
            if not initial_stage.result.converged:
                raise RuntimeError("initial 48x40 ordinary equilibrium did not converge")
            # continuation_state is the restart buffer (xstore), not the
            # converged state whose residuals the ordinary solve reports.
            self.seed = initial_stage.result.state
            initial_rcon, initial_zcon = initial_stage.rcon0, initial_stage.zcon0
            self.provenance["initial_equilibrium_solves_executed"] = initial_solves
            self.provenance["initial_ordinary_iterations"] = int(initial_stage.result.iterations)
            self.write(self.output / "manifest.json", self.provenance)
        self.event(
            "initial_certification_start",
            absolute_step=self.step,
            parameters_zero=bool(np.all(p == 0)),
            initial_equilibrium_solves=initial_solves,
            resumed=self.resume is not None,
        )
        self.cfg = fc.make_free_boundary_continuation_config_from_state(
            self.solver,
            self.params,
            p,
            state=self.seed,
            rcon0=initial_rcon,
            zcon0=initial_zcon,
            parameter_scales=self.parameter_scales,
            continuation_step=0.1,
            max_continuation_steps=self.contract["max_continuation_steps"],
            root_residual_atol=self.contract["root_residual_atol"],
        )
        self.accepted = self.cfg._anchor
        self._check_initial_equilibrium(initial_stage)

    @property
    def input(self):
        return self.cfg.solver.implicit.inp

    @property
    def coil_parameters(self):
        return self.builder

    @property
    def continuation(self):
        return self.cfg

    def bind(self, problem):
        import numpy as np
        from checkpoint import verify_restoration

        self.problem = problem
        if problem.cfg is not self.cfg or problem.parameterization is not self.builder:
            raise ValueError("case persistence requires its original certified root and coil chart")
        expected = [self.contract["targets"][name] for name in ("mean_iota", "aspect_ratio", "b0")]
        np.testing.assert_array_equal(problem.targets, expected)
        np.testing.assert_array_equal(problem.constraint_scales, self.constraint_scales)
        np.testing.assert_array_equal(
            problem.constraint_tolerances, self.policy.constraint_relative_tolerance * np.abs(problem.targets)
        )
        if self.resume is not None:
            np.testing.assert_array_equal(problem.targets, self.resume["targets"])
            if problem.loss_scale != float(self.resume["loss_scale"]):
                raise ValueError("resume must retain the saved QA normalization")
            verify_restoration(problem.accepted, self.resume)
            self.event(
                "checkpoint_restored_exactly",
                absolute_step=self.step,
                sha256=self.args.checkpoint_sha256,
                targets=problem.targets.tolist(),
                loss_scale=problem.loss_scale,
                initial_equilibrium_solves=0,
            )
        self.targets = problem.targets.copy()
        self.loss_scale = problem.loss_scale
        self.rows = problem._compact_state
        self.values = problem.optimizer_rows(problem.accepted)
        self._record_initial_state(problem.constraint_values(problem.accepted.parameters))
        if not self.batch_tuning_done and not self.args.initialize_only:
            diagnostics = []
            self._tune_adjoint_batch(diagnostics)
            problem.cfg, problem.solver = self.cfg, self.cfg.solver
            problem._linearization = self.linearization
            problem._linearization_record = self.accepted
            problem._compact_jac = np.asarray(self.linearization.field_jacobian)
            self.linearization = None
        problem._emit = self.solver_event

    def solver_event(self, phase, **data):
        if phase == "proposal":
            self.jac = self.problem._compact_jac
            trial = data["trial"]
            self._trial_name = f"trial_step_{self.step + 1:04d}_trial_{trial:02d}"
            self._record_proposal(data["delta"], trial, data["points"], self._trial_name)
        elif phase == "correction":
            self.last_stage, self.point = data["stage"], data["parameters"]
            self._record_ordinary_correction(
                data["trial"], data["index"], data["points"], time.monotonic() - data["seconds"]
            )
        elif phase == "certification":
            self._record_certification(data["trial"], data["index"], data["candidate"])
        elif phase == "candidate":
            self.last_stage, self.point = data["stage"], data["parameters"]
            self._record_candidate(self._trial_name)
        else:
            self.event(phase, absolute_step=self.step, **data)

    def optimizer_event(self, phase, **data):
        if phase == "direction":
            self.has_converged(data["direction"])
        elif phase == "trial":
            self.record_trial(data["record"])
        elif phase == "stagnated":
            from types import SimpleNamespace

            self.report_stagnation(SimpleNamespace(trials=data["trials"]))
        elif phase == "accepted":
            from case import metrics

            result = data["result"]
            self.cfg, self.accepted = self.problem.cfg, self.problem.accepted
            self.step += 1
            self.values = result.values
            self.last_stage = None
            self.initial_gradient_norm = result.initial_gradient_norm
            motion, current = self.problem.parameterization.motion_bounds(result.trial.delta)
            self.event(
                "promoted",
                absolute_step=self.step,
                metrics=metrics(self.values, self.targets, self.loss_scale),
                root_residual=self.accepted.root_residual_norm,
                trial_count=len(result.trial.trials),
                acceptance=result.trial.trials[-1],
                maximum_coil_bound_m=motion,
                maximum_current_fraction=current,
                forces={name: float(getattr(self.accepted.result, name)) for name in ("fsqr", "fsqz", "fsql", "fedge")},
                checkpoint=self.checkpoint(),
            )
            self.record_diagnostics()

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            if hasattr(self, "problem"):
                self.problem.close()
