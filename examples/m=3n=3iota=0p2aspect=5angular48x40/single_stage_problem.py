"""Numerical implementation of the maintained free-boundary single-stage case.

The example defines its objective and physical constraints and passes them to
FreeBoundaryProblem. This module owns equilibrium initialization, differentiation,
tangent prediction and the projected/restoring optimizer. The exact accepted-state
and rejection rules are shared by direct, legacy and benchmark callers.
"""

import time
from single_stage_support import FreeBoundaryRun


def _objective_functions(problem):
    # Legacy benchmark callers instantiate FreeBoundaryRun directly. Their
    # default callbacks resolve to the same maintained example definitions.
    from case import physical_rows, resolve_targets, make_rows

    return (
        getattr(problem, "physical_value_function", None) or physical_rows,
        getattr(problem, "target_function", None) or resolve_targets,
        getattr(problem, "row_factory", None) or make_rows,
    )


class FreeBoundaryProblem(FreeBoundaryRun):
    """The scientific operations; only accepted equilibria become new anchors."""

    def __init__(self, args, *, rows=None, physical_values=None, targets=None):
        super().__init__(args)
        self.row_factory = rows
        self.physical_value_function = physical_values
        self.target_function = targets

    def setup_problem(self):
        """Load or restore the case, certify the equilibrium, and define QA rows."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from vmex.core import implicit as im, freeboundary_implicit as fbi, freeboundary_continuation as fc
        from vmex.core.freeboundary import _solve_free_boundary_stage
        from checkpoint import verify_restoration

        physical_rows, resolve_targets, make_rows = _objective_functions(self)
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
        initial_rcon = (
            jnp.zeros_like(self.rt.rcon0) if self.resume is None else jnp.asarray(self.resume["rcon0"])
        )
        initial_zcon = (
            jnp.zeros_like(self.rt.zcon0) if self.resume is None else jnp.asarray(self.resume["zcon0"])
        )
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
                forces={
                    n: float(getattr(initial_stage.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")
                },
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
        start_physical = np.asarray(physical_rows(self.accepted.state, self.rt))
        if self.resume is None:
            self.targets = resolve_targets(start_physical)
            self.loss_scale = max(
                float(make_rows(self.rt, jnp.asarray(self.targets), 1.0)(self.accepted.state)[0]), 0.001
            )
        else:
            verify_restoration(self.accepted, self.resume)
            self.event(
                "checkpoint_restored_exactly",
                absolute_step=self.step,
                sha256=self.args.checkpoint_sha256,
                targets=self.targets.tolist(),
                loss_scale=self.loss_scale,
                initial_equilibrium_solves=0,
            )
        self.rows = make_rows(self.rt, jnp.asarray(self.targets), self.loss_scale)
        self.values = np.asarray(self.rows(self.accepted.state))
        self._record_initial_state(start_physical)

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

    def linearize(self):
        """Return QA/constraint rows and their derivatives with respect to coils."""
        import numpy as np

        if time.monotonic() >= self.deadline:
            raise TimeoutError("walltime")
        self.point = None
        self._close_linearization()
        diagnostics = []
        t = time.monotonic()
        self.event("adjoint_start", absolute_step=self.step)
        try:
            if not self.batch_tuning_done:
                self._tune_adjoint_batch(diagnostics)
            else:
                if self.cfg.solver.adjoint_dense_batch_size != self.adjoint_batch_size:
                    raise RuntimeError("selected adjoint batch configuration was replaced")
                self.linearization = self._state_linearization(self.cfg, diagnostics)
            jac = np.asarray(self.linearization.field_jacobian)
        finally:
            self.event("adjoint", absolute_step=self.step, seconds=time.monotonic() - t, rows=diagnostics)
        self.jac = jac
        return self.values, jac

    def evaluate_trial(self, delta, trial):
        """Reused dense tangent, ordinary correction, and strict certification."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from vmex.core import freeboundary_continuation as fc
        from vmex.core.freeboundary import _solve_free_boundary_stage
        from optimization import TrialRejected
        from vmex.core.errors import VmecError

        self.last_stage = None
        if time.monotonic() >= self.deadline:
            raise TimeoutError("walltime")
        count = max(1, int(np.ceil(np.max(np.abs(delta / self.parameter_scales)) / 0.1)))
        name = f"trial_step_{self.step + 1:04d}_trial_{trial:02d}"
        self._record_proposal(delta, trial, count, name)
        if count > self.contract["max_continuation_steps"]:
            raise TrialRejected("continuation budget exceeded")
        try:
            if self.linearization is None:
                raise RuntimeError("trial requires the current accepted-root dense linearization")
            diagnostics = []
            t = time.monotonic()
            try:
                tangent = self.linearization.tangent(
                    self.accepted, self.cfg, jnp.asarray(delta), diagnostics=diagnostics
                )
            finally:
                self.event(
                    "tangent",
                    absolute_step=self.step + 1,
                    trial=trial,
                    seconds=time.monotonic() - t,
                    rows=diagnostics,
                )
            previous = self.accepted
            for index in range(1, count + 1):
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("walltime")
                self.point = self.accepted.parameters + delta * (index / count)
                # Advance along the coil proposal in bounded increments. The
                # previous certified intermediate state seeds the next solve;
                # every new backtracking trial starts again at self.accepted.
                predicted = jax.tree.map(lambda x, dx: x + dx / count, previous.state, tangent)
                correction_started = time.monotonic()
                self.last_stage = _solve_free_boundary_stage(
                    self.inp,
                    external_field=self.builder(jnp.asarray(self.point)),
                    resolution=self.solver.resolution,
                    ftol=self.contract["force_tolerance"],
                    max_iterations=self.contract["max_iterations"],
                    initial_state=predicted,
                    constraint_continuation=(previous.rcon0, previous.zcon0),
                    include_edge_in_convergence=True,
                    edge_force_tolerance=self.contract["edge_force_tolerance"],
                    error_on_no_convergence=False,
                    jacobian_retries=0,
                    allow_initial_axis_reguess=False,
                    use_fft=False,
                )
                self._record_ordinary_correction(trial, index, count, correction_started)
                if not self.last_stage.result.converged:
                    raise TrialRejected("ordinary equilibrium did not converge")
                previous = fc.certify_free_boundary_continuation_state(
                    self.cfg,
                    self.point,
                    self.last_stage.result.state,
                    rcon0=self.last_stage.rcon0,
                    zcon0=self.last_stage.zcon0,
                    result=self.last_stage.result,
                )
                self._record_certification(trial, index, previous)
            return previous, np.asarray(self.rows(previous.state))
        except VmecError as exc:
            raise TrialRejected(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._record_candidate(name)


def proposal(values, jacobian, scales, targets, constraint_scales, policy):
    from vmex.core.projected_optimization import proposal as implementation
    from optimization import motion_bounds
    return implementation(values, jacobian, scales, targets, constraint_scales, policy,
                          motion_bounds=motion_bounds)


def acceptance(before, after, direction, alpha, targets, constraint_scales, policy):
    from vmex.core.projected_optimization import acceptance as implementation
    return implementation(before, after, direction, alpha, targets, constraint_scales, policy)


def backtrack(before, direction, targets, constraint_scales, policy, evaluate, record):
    from vmex.core.projected_optimization import backtrack as implementation
    from optimization import motion_bounds
    from single_stage_support import TrialRejected as LegacyRejected
    from vmex.core.projected_optimization import TrialRejected
    def evaluate_compatible(delta, trial):
        try:
            return evaluate(delta, trial)
        except LegacyRejected as exc:
            raise TrialRejected(str(exc)) from exc
    return implementation(before, direction, targets, constraint_scales, policy, evaluate_compatible, record,
                          motion_bounds=motion_bounds)


def optimize(problem, target_step):
    while problem.step < target_step:
        # Differentiate QA and all three constraints through the equilibrium.
        values, jacobian = problem.linearize()

        # Project QA descent along the constraints, then add target restoration.
        direction = proposal(
            values,
            jacobian,
            problem.parameter_scales,
            problem.targets,
            problem.constraint_scales,
            problem.policy,
        )
        # Stationarity requires feasibility and a sufficiently small projected
        # gradient, using the original run's reference norm after checkpoint resume.
        if problem.has_converged(direction):
            return "converged"

        # Reuse the accepted-root factors and scale its tangent for backtracking,
        # solve the equilibrium, and check QA and physical bounds.
        trial = backtrack(
            values,
            direction,
            problem.targets,
            problem.constraint_scales,
            problem.policy,
            problem.evaluate_trial,
            problem.record_trial,
        )
        if trial.candidate is None:
            problem.report_stagnation(trial)
            return "stagnated"

        # Only an accepted trial becomes the next equilibrium and checkpoint.
        problem.accept(trial)

    return "step_budget_reached"
