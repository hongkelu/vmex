"""Optimizer-neutral free-boundary objectives with explicit coil coordinates.

Objective tuples use FunctionProblem's residual/gradient convention. Physical
bands are separate constraints. Only accept() changes the continuation anchor;
ordinary function evaluation, reporting, and rejected trials never promote it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .problem import FunctionProblem
from .optimize import Equilibrium
from .coil_parameters import CoilParameters
from .projected_optimization import TrialRejected
from . import freeboundary_continuation as fc, freeboundary_implicit as fbi, implicit as im
from .errors import VmecError


@jax.custom_jvp
def _norm(x):
    return jnp.linalg.norm(x)


@_norm.defjvp
def _norm_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    value = _norm(x)
    return value, jnp.where(value > 0, jnp.vdot(x, dx) / jnp.where(value > 0, value, 1.0), 0.0)


@dataclass(frozen=True)
class TargetBand:
    """A scalar physical target; acceptance uses atol + rtol*abs(target).

    scale is a conditioning scale, independent of the physical tolerance.
    function follows the standard function(state, runtime) convention.
    """

    function: Callable
    target: float
    rtol: float = 0.01
    atol: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        numbers = (self.target, self.rtol, self.atol, self.scale)
        if (
            not callable(self.function)
            or not np.all(np.isfinite(numbers))
            or self.rtol < 0
            or self.atol < 0
            or self.scale <= 0
            or self.tolerance <= 0
        ):
            raise ValueError("finite target, positive scale, and positive physical tolerance required")

    @property
    def tolerance(self):
        """Absolute physical tolerance, in the units of the quantity."""
        return self.atol + self.rtol * abs(self.target)


@dataclass(frozen=True)
class _Equilibrium(Equilibrium):
    _wout_factory: Callable = field(kw_only=True, repr=False)

    @cached_property
    def wout(self):
        return self._wout_factory()


class FreeBoundaryProblem(FunctionProblem):
    """Weighted plasma objectives and derivatives with respect to coil variables.

    Build with from_tuples(). The host-side equilibrium/adjoint machinery is
    eager; the state objective functions are JIT compiled. No files, environment
    variables, CLI state, or signal handlers are owned by this class.
    """

    @classmethod
    def from_tuples(
        cls,
        inp,
        objective_terms,
        *,
        coils=None,
        parameterization=None,
        current_dofs=None,
        max_coil_mode=None,
        scales=None,
        constraints=(),
        restart_from=None,
        checkpoint=None,
        checkpoint_sha256=None,
        checkpoint_identity=None,
        solver_options=None,
        objective_normalization="initial",
        continuation=None,
        x0=None,
        continuation_step=0.1,
        max_continuation_steps=64,
        root_residual_atol=2e-6,
        event=None,
        deadline=None,
    ):
        """Construct a coil problem, solve/certify its seed, and fix normalization.

        Pass coils plus current_dofs, or an explicit CoilParameters chart.
        restart_from is a spectral seed for one ordinary solve. continuation
        attaches an already certified config with the identical field chart;
        the caller authenticates checkpoint data before creating that config.
        A numeric objective_normalization preserves an existing normalization.
        checkpoint plus checkpoint_sha256 restores an accepted state without
        solving it again; certification must preserve every stored array.
        checkpoint_identity must describe the objective and optimizer settings.
        Only state/runtime objective terms are supported in this first API.
        """
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary implicit optimization requires JAX_ENABLE_X64=1")
        if (coils is None) == (parameterization is None):
            raise ValueError("provide exactly one of coils or parameterization")
        if parameterization is None:
            if current_dofs is None:
                raise ValueError("select current_dofs explicitly (or () to fix all currents)")
            parameterization = CoilParameters.from_coils(
                coils, current_dofs=current_dofs, max_coil_mode=max_coil_mode, scales=scales
            )
        elif any(x is not None for x in (current_dofs, max_coil_mode, scales)):
            raise ValueError("coordinate settings belong to the supplied parameterization")
        constraints = tuple(constraints)
        if not all(isinstance(c, TargetBand) for c in constraints):
            raise TypeError("constraints must be TargetBand instances")
        from . import _freeboundary_checkpoint as storage

        if checkpoint is not None and (continuation is not None or restart_from is not None or x0 is not None):
            raise ValueError("checkpoint cannot be combined with continuation, seed or x0")
        if checkpoint is None and checkpoint_sha256 is not None:
            raise ValueError("checkpoint_sha256 requires a checkpoint")
        if continuation is not None and checkpoint_identity is not None:
            raise ValueError("checkpoint storage requires construction with solver_options, not continuation")
        saved = None
        identity = None
        if checkpoint_identity is not None:
            identity = storage.identity(inp, parameterization, constraints, solver_options or {},
                                        dict(objectives=checkpoint_identity, continuation_step=continuation_step,
                                             max_continuation_steps=max_continuation_steps,
                                             root_residual_atol=root_residual_atol))
        if checkpoint is not None:
            if identity is None:
                raise ValueError("checkpoint_identity is required to authenticate objective and optimizer settings")
            if objective_normalization != "initial":
                raise ValueError("checkpoint owns the objective normalization")
            saved = storage.read(checkpoint, checkpoint_sha256, identity)
            x0 = saved["parameters"]
            objective_normalization = float(saved["loss_scale"])
        if continuation is not None:
            if any(x is not None for x in (restart_from, solver_options, x0)):
                raise ValueError("continuation already specifies solver, seed and initial parameters")
            if continuation.solver.field_from_parameters is not parameterization:
                raise ValueError("continuation and problem must share the identical field chart")
            if inp is not continuation.solver.implicit.inp:
                raise ValueError("use the input held by the supplied continuation config")
            cfg = continuation
        else:
            if not inp.lfreeb:
                raise ValueError("input must enable free-boundary equilibrium")
            if np.iscomplexobj(x0):
                raise ValueError("initial coil parameters must be real")
            point = parameterization.x0 if x0 is None else np.asarray(x0, dtype=float)
            if point.shape != parameterization.x0.shape or not np.all(np.isfinite(point)):
                raise ValueError("invalid initial coil parameters")
            opts = dict(solver_options or {})
            if opts.get("include_edge_in_convergence", True) is not True:
                raise ValueError("free-boundary optimization requires strict edge convergence")
            opts["include_edge_in_convergence"] = True
            opts.setdefault("adjoint_solver", "forward_dense_jax")
            opts.setdefault("adjoint_fail", "error")
            opts.setdefault("adjoint_dense_batch_size", 32)
            if opts["adjoint_solver"] != "forward_dense_jax":
                raise ValueError("this API currently requires the forward_dense_jax derivative backend")
            solver = fbi.make_free_boundary_config(
                inp, parameterization(jnp.asarray(point)), field_from_parameters=parameterization, **opts
            )
            params = im.params_from_input(inp)
            from .freeboundary import _solve_free_boundary_stage

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("walltime")
            if saved is not None:
                from .solver import SpectralState
                state = SpectralState(*(jnp.asarray(saved[f]) for f in storage.FIELDS))
                cfg = fc.make_free_boundary_continuation_config_from_state(
                    solver, params, point, state=state, rcon0=jnp.asarray(saved["rcon0"]),
                    zcon0=jnp.asarray(saved["zcon0"]), parameter_scales=parameterization.scales,
                    continuation_step=continuation_step, max_continuation_steps=max_continuation_steps,
                    root_residual_atol=root_residual_atol)
                storage.verify(cfg._anchor, saved)
            else:
                if isinstance(restart_from, (str, Path)):
                    from .restart import restart_state
                    restart_from = restart_state(restart_from, inp, ns=solver.resolution.ns)
                stage = _solve_free_boundary_stage(
                    inp,
                    external_field=parameterization(jnp.asarray(point)),
                    resolution=solver.resolution,
                    ftol=solver.implicit.ftol,
                    max_iterations=solver.implicit.max_iterations,
                    initial_state=restart_from,
                    include_edge_in_convergence=True,
                    edge_force_tolerance=solver.edge_force_tolerance,
                    error_on_no_convergence=False,
                    jacobian_retries=0,
                    allow_initial_axis_reguess=False,
                    use_fft=False,
                )
                if not stage.result.converged:
                    raise VmecError("initial ordinary equilibrium did not converge")
                cfg = fc.make_free_boundary_continuation_config_from_state(
                    solver,
                    params,
                    point,
                    state=stage.result.state,
                    rcon0=stage.rcon0,
                    zcon0=stage.zcon0,
                    parameter_scales=parameterization.scales,
                    continuation_step=continuation_step,
                    max_continuation_steps=max_continuation_steps,
                    root_residual_atol=root_residual_atol,
                )
                if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                       zip(jax.tree.leaves(stage.result.state), jax.tree.leaves(cfg._anchor.state))):
                    raise VmecError("initial certification changed the ordinary state")
                if not np.isclose(stage.result.fedge, cfg._anchor.result.fedge, rtol=1e-5, atol=1e-15):
                    raise VmecError("ordinary/fresh edge residual disagreement")
        if not cfg.solver.include_edge_in_convergence:
            raise ValueError("a strict edge-certified continuation config is required")
        if cfg.solver.adjoint_solver != "forward_dense_jax":
            raise ValueError("this API currently requires the forward_dense_jax derivative backend")
        if not np.array_equal(cfg.parameter_scales, parameterization.scales):
            raise ValueError("continuation and coil coordinate scales differ")
        problem = cls(
            inp,
            parameterization,
            cfg,
            tuple(objective_terms),
            constraints,
            objective_normalization,
            event=event,
            deadline=deadline,
        )
        problem.checkpoint_identity = identity
        if saved is not None:
            problem.accepted_step = int(saved["accepted_step"])
            problem.initial_gradient_norm = saved["gradient_reference"]
        return problem

    def __init__(
        self, inp, parameterization, cfg, objective_terms, constraints, normalization, *, event=None, deadline=None
    ):
        from .optimize import residuals_from_tuples

        self.inp, self.parameterization, self.cfg = inp, parameterization, cfg
        self.solver, self.params = cfg.solver, cfg.params
        self.rt = im.runtime_from_params(self.params, self.solver.implicit)
        self.accepted = cfg._anchor
        self.accepted_step = 0
        self.initial_gradient_norm: float | None = None
        self.checkpoint_identity: str | None = None
        self.targets = np.array([c.target for c in constraints])
        self.constraint_scales = np.array([c.scale for c in constraints])
        self.constraint_tolerances = np.array([c.tolerance for c in constraints])
        self.constraints = constraints
        self._emit = event or (lambda *args, **kwargs: None)
        self.deadline = deadline
        self._linearization = self._linearization_record = None
        self._compact_jac = None
        self._records = {self._key(self.accepted.parameters): self.accepted}

        def raw(state):
            return residuals_from_tuples(state, self.rt, objective_terms)

        first = np.asarray(raw(self.accepted.state))
        if first.size == 0 or not np.all(np.isfinite(first)):
            raise ValueError("objective must provide finite nonempty residuals")
        if isinstance(normalization, str):
            if normalization != "initial":
                raise ValueError("normalization must be 'initial' or a positive fixed number")
            normalization = max(float(jnp.linalg.norm(jnp.asarray(first))), 0.001)
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError("normalization must be finite and positive")
        self.loss_scale = float(normalization)
        self._residual_state = jax.jit(lambda state: raw(state) / self.loss_scale)

        def physical(state):
            values = [jnp.asarray(c.function(state, self.rt)) for c in constraints]
            if any(v.shape != () for v in values):
                raise ValueError("TargetBand quantities must be scalar")
            return jnp.stack(values) if values else jnp.empty((0,), dtype=first.dtype)

        self._physical_state = jax.jit(physical)

        def compact(state):
            return jnp.r_[
                _norm(raw(state)) / self.loss_scale,
                (self._physical_state(state) - jnp.asarray(self.targets)) / jnp.asarray(self.constraint_scales),
            ]

        self._compact_state = jax.jit(compact)
        # Validate quantities eagerly; an invalid definition must not start an optimizer.
        initial = np.asarray(self._compact_state(self.accepted.state))
        if not np.all(np.isfinite(initial)):
            raise ValueError("nonfinite objective or physical constraints")
        super().__init__(
            self.accepted.parameters,
            names=parameterization.dof_names,
            scales=parameterization.scales,
            fun=self._value,
            value_and_grad=self._value_gradient,
            residual=self._residual_values,
            residual_and_jac=self._residual_jacobian,
            metadata={"holder": {"failed_trials": 0}},
        )

    def _check_time(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("walltime")

    @staticmethod
    def _x(x):
        if np.iscomplexobj(x):
            raise ValueError("coil parameters must be real")
        return FunctionProblem._x(x)

    def _validate_x(self, x):
        if np.iscomplexobj(x):
            raise ValueError("coil parameters must be real")
        x = np.asarray(x, dtype=float)
        if x.shape != self.x0.shape or not np.all(np.isfinite(x)):
            raise ValueError("invalid coil parameter vector")
        return x

    def _record(self, x):
        x = self._validate_x(x)
        key = self._key(x)
        if key not in self._records:
            self._records[key] = self._trial(x - self.accepted.parameters, 0)
            # Retain at most the anchor and most recently evaluated trial.
            self._records = {self._key(self.accepted.parameters): self.accepted, key: self._records[key]}
        return self._records[key]

    def _close_linearization(self):
        if self._linearization is not None:
            self._linearization.close()
        self._linearization = self._linearization_record = self._compact_jac = None

    def _derivatives(self, record):
        self._check_time()
        if self._linearization_record is not record:
            self._close_linearization()
            diagnostics = []
            self._emit("adjoint_start")
            started = time.monotonic()
            try:
                rhs = jax.jacrev(self._compact_state)(record.state)
                self._linearization = fc.free_boundary_continuation_state_pullback(
                    record, self.cfg, rhs, diagnostics=diagnostics, return_linearization=True
                )
                self._linearization_record = record
                self._compact_jac = np.asarray(self._linearization.field_jacobian)
            finally:
                self._emit("adjoint", seconds=time.monotonic() - started, rows=diagnostics)
        return self._compact_jac

    def optimizer_rows(self, record):
        """Return objective norm and scaled target errors without a derivative solve."""
        return np.asarray(self._compact_state(record.state))

    def linearize(self):
        """Return compact objective/constraint rows and derivatives at the accepted root."""
        return self.optimizer_rows(self.accepted), self._derivatives(self.accepted).copy()

    def _trial(self, delta, trial):
        from .freeboundary import _solve_free_boundary_stage

        self._check_time()
        delta = np.asarray(delta, dtype=float)
        count = max(1, int(np.ceil(np.max(np.abs(delta / self.scales)) / self.cfg.continuation_step)))
        if count > self.cfg.max_continuation_steps:
            raise TrialRejected("continuation budget exceeded")
        self._derivatives(self.accepted)
        self._emit("proposal", delta=delta.copy(), trial=trial, points=count, jacobian=self._compact_jac.copy())
        last_stage = point = None
        try:
            diagnostics = []
            started = time.monotonic()
            try:
                tangent = self._linearization.tangent(
                    self.accepted, self.cfg, jnp.asarray(delta), diagnostics=diagnostics
                )
            finally:
                self._emit("tangent", trial=trial, seconds=time.monotonic() - started, rows=diagnostics)
            previous = self.accepted
            for index in range(1, count + 1):
                self._check_time()
                point = self.accepted.parameters + delta * (index / count)
                predicted = jax.tree.map(lambda x, dx: x + dx / count, previous.state, tangent)
                started = time.monotonic()
                last_stage = _solve_free_boundary_stage(
                    self.inp,
                    external_field=self.parameterization(jnp.asarray(point)),
                    resolution=self.solver.resolution,
                    ftol=self.solver.implicit.ftol,
                    max_iterations=self.solver.implicit.max_iterations,
                    initial_state=predicted,
                    constraint_continuation=(previous.rcon0, previous.zcon0),
                    include_edge_in_convergence=True,
                    edge_force_tolerance=self.solver.edge_force_tolerance,
                    error_on_no_convergence=False,
                    jacobian_retries=0,
                    allow_initial_axis_reguess=False,
                    use_fft=False,
                )
                self._emit(
                    "correction",
                    stage=last_stage,
                    parameters=point,
                    trial=trial,
                    index=index,
                    points=count,
                    seconds=time.monotonic() - started,
                )
                if not last_stage.result.converged:
                    raise TrialRejected("ordinary equilibrium did not converge")
                previous = fc.certify_free_boundary_continuation_state(
                    self.cfg,
                    point,
                    last_stage.result.state,
                    rcon0=last_stage.rcon0,
                    zcon0=last_stage.zcon0,
                    result=last_stage.result,
                )
                self._emit("certification", candidate=previous, trial=trial, index=index)
            return previous
        except (VmecError, TrialRejected) as exc:
            self.metadata["holder"]["failed_trials"] += 1
            if isinstance(exc, TrialRejected):
                raise
            raise TrialRejected(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._emit("candidate", stage=last_stage, parameters=point, trial=trial)

    def evaluate_trial(self, delta, trial):
        """Evaluate a proposal from the accepted anchor without promotion."""
        delta = self._validate_x(delta)
        candidate = self._trial(delta, trial)
        self._records = {self._key(self.accepted.parameters): self.accepted, self._key(candidate.parameters): candidate}
        return candidate, self.optimizer_rows(candidate)

    def accept(self, candidate):
        """Promote a candidate returned by this problem after optimizer acceptance."""
        if self._records.get(self._key(candidate.parameters)) is not candidate:
            raise ValueError("candidate is not a current evaluation of this problem")
        self.cfg = fc.reanchor_free_boundary_continuation_config(self.cfg, candidate)
        self.accepted = self.cfg._anchor
        self.accepted_step += 1
        self._close_linearization()
        self._records = {self._key(self.accepted.parameters): self.accepted}
        self._vg_cache = self._rj_cache = None

    def _value(self, x):
        rows = self.optimizer_rows(self._record(x))
        return 0.5 * float(rows[0]) ** 2

    def _value_gradient(self, x):
        record = self._record(x)
        rows = self.optimizer_rows(record)
        jac = self._derivatives(record)
        return 0.5 * float(rows[0]) ** 2, rows[0] * jac[0]

    def _residual_values(self, x):
        return np.asarray(self._residual_state(self._record(x).state))

    def _residual_jacobian(self, x):
        record = self._record(x)
        rhs = jax.jacrev(self._residual_state)(record.state)
        jac = fc.free_boundary_continuation_state_pullback(record, self.cfg, rhs)
        return np.asarray(self._residual_state(record.state)), np.asarray(jac)

    def constraint_values(self, x):
        """Return unscaled physical quantities, in their original units."""
        return np.asarray(self._physical_state(self._record(x).state))

    def constraint_jac(self, x):
        """Return derivatives of the physical quantities with respect to x."""
        record = self._record(x)
        return self._derivatives(record)[1:].copy() * self.constraint_scales[:, None]

    def coils_from_x(self, x):
        """Reconstruct coils, without solving or changing the accepted equilibrium."""
        return self.parameterization.coils_from_x(self._validate_x(x))

    def equilibrium_from_x(self, x):
        """Return a certified equilibrium; WOUT uses its exact fixed-geometry vacuum."""
        record = self._record(x)
        return _Equilibrium(self.inp, record.state, self.rt, record.result, _wout_factory=lambda: self._wout(record))

    def tune_adjoint_batch(self):
        """Compare batches 32/64 at this root and retain the certified winner's factors."""
        from ._freeboundary_dense import BATCH_SIZES, tune_adjoint_batch

        self._check_time()
        self._close_linearization()
        configs = {batch: self.cfg if batch == self.solver.adjoint_dense_batch_size
                   else replace(self.cfg, solver=replace(self.solver, adjoint_dense_batch_size=batch))
                   for batch in BATCH_SIZES}
        rhs = jax.jacrev(self._compact_state)(self.accepted.state)

        def evaluate(batch):
            self._check_time()
            diagnostics = []
            root = fc.free_boundary_continuation_state_pullback(
                self.accepted, configs[batch], rhs, diagnostics=diagnostics, return_linearization=True)
            try:
                self._emit("adjoint_batch_certification", batch=batch, rows=diagnostics)
            except BaseException:
                root.close()
                raise
            return root

        selected, linearization, report = tune_adjoint_batch(
            evaluate, deadline=self.deadline if self.deadline is not None else float("inf"), event=self._emit)
        self.cfg, self.solver = configs[selected], configs[selected].solver
        self._linearization = linearization
        self._linearization_record = self.accepted
        self._compact_jac = np.asarray(linearization.field_jacobian)
        return report

    def save_checkpoint(self, path):
        """Save the accepted root and optimizer reference for exact authenticated resume."""
        from ._freeboundary_checkpoint import write
        return write(self, path)

    def close(self):
        """Release retained derivative factors without altering accepted results."""
        self._close_linearization()

    def _wout(self, record):
        from .wout import wout_from_state

        currents = np.asarray(self.parameterization.base_currents_at(jnp.asarray(record.parameters)))
        # Re-evaluate the vacuum on this exact fixed plasma/coil geometry for
        # every exported pair. Imported anchors have no attached VacuumOutput;
        # ordinary results can carry cadence caches from a preceding geometry.
        from vmex.core.freeboundary import _vacuum_executables, _vacuum_output, FreeBoundaryState

        export_rt = replace(
            self.rt,
            rcon0=record.rcon0,
            zcon0=record.zcon0,
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
        vacuum_values = program.full(record.state, export_rt, self.parameterization(jnp.asarray(record.parameters)))
        vacuum = _vacuum_output(
            FreeBoundaryState(potvac=vacuum_values["potvac"], surface_fields=vacuum_values["surface_fields"]), basis
        )
        if vacuum is None or any(
            not np.all(np.isfinite(np.asarray(x)))
            for x in (vacuum.potsin, vacuum.bsubu, vacuum.bsubv, vacuum.bsupu, vacuum.bsupv)
        ):
            raise ValueError("snapshot requires finite fixed-geometry vacuum output")
        return wout_from_state(
            inp=self.inp,
            state=record.state,
            niter=record.result.iterations,
            fsqr=record.result.fsqr,
            fsqz=record.result.fsqz,
            fsql=record.result.fsql,
            vacuum_output=vacuum,
            nextcur=len(currents),
            extcur=currents,
            curlabel=tuple(f"base_coil_{i}" for i in range(len(currents))),
        )
