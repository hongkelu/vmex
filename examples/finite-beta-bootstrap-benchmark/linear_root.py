"""Coupled-root Newton and total derivatives, with checked reusable host LU.

No finite differences in production. Device JVPs build bounded column batches;
only the full matrix/factors live on the host, avoiding a second GPU-sized LU.
A stale LU is a preconditioner, never accepted as the current Jacobian.
"""

from dataclasses import dataclass, field
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import time
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning
from scipy.sparse.linalg import LinearOperator, gmres


class RootFailure(RuntimeError):
    pass


class SlowProgress(RootFailure):
    """Request a smaller continuation step without an ordinary seed solve."""


class _RefreshNeeded(RuntimeError):
    pass


@partial(jax.jit, static_argnums=(0,))
def _linearize(residual, y, x):
    """Keep the design dynamic in the tape, so new points reuse executables."""
    return jax.linearize(lambda state: residual(state, x), y)


@jax.jit
def _tangent_apply(tangent, vector):
    return tangent(vector)


@jax.jit
def _tangent_transpose(tangent, vector):
    return jax.linear_transpose(tangent, jnp.zeros_like(vector))(vector)[0]


@jax.jit
def _tangent_columns(tangent, template, indices):
    return jax.vmap(lambda i: tangent(jax.nn.one_hot(i, template.size, dtype=template.dtype)))(indices)


@dataclass
class ReuseCost:
    """Measure forward/transpose costs separately; refresh only if it pays."""

    dense_seconds: float
    horizon: int
    samples: dict = field(default_factory=dict)
    best: dict = field(default_factory=dict)

    def observe(self, seconds, transpose):
        values = self.samples.setdefault(transpose, [])
        values.append(seconds)
        # Ignore the first two calls in each direction for compilation/warmup.
        if len(values) <= 2:
            return False
        self.best[transpose] = min(self.best.get(transpose, float("inf")), seconds)
        if len(values) < 5:
            return False
        recent = float(np.median(values[-3:]))
        baseline = self.best[transpose]
        return recent > max(2.0, 2 * baseline) and (recent - baseline) * self.horizon > self.dense_seconds


@dataclass
class Options:
    atol: float = 1e-12
    newton_steps: int = 12
    rtol: float = 1e-10
    gate: float = 1e-9
    restart: int = 80
    cycles: int = 4
    batch: int = 8
    max_dofs: int = 20000
    extra_newton_steps: int = 4
    refresh_horizon: int = 12
    reuse_budget_fraction: float = 0.2
    adaptive_newton: bool = True
    separate_action_warmup: bool = False
    saved_linearization: bool = False
    # Independently switchable for measured comparisons with direct actions.
    saved_krylov_actions: bool = False
    newton_rtol: float | None = None
    newton_gate: float | None = None
    trial_dense_rebuilds: int | None = None
    stagnation_steps: int = 3


class LinearRoot:
    def __init__(
        self, residual, *, valid=lambda y, x: True, invalid_reason=None, options=None, event=lambda **kw: None
    ):
        self.residual = jax.jit(residual)
        self.valid, self.options, self.event = valid, options or Options(), event
        if self.options.trial_dense_rebuilds is not None and self.options.trial_dense_rebuilds < 1:
            raise ValueError("trial dense rebuild budget must allow at least one recovery")
        if self.options.stagnation_steps < 2:
            raise ValueError("stagnation detection needs at least two Newton corrections")
        self._segment_dense_builds = None
        self.invalid_reason = invalid_reason
        self.factors = None
        self.cost = None
        self.refresh_pending = False
        self.last_solve = {}
        self._warmed_actions = set()
        self._tangent = None
        self._tangent_point = None
        self._factor_point = None
        self._columns = jax.jit(
            lambda y, x, idx: jax.vmap(
                lambda i: jax.jvp(self.residual, (y, x), (jax.nn.one_hot(i, y.size, dtype=y.dtype), jnp.zeros_like(x)))[
                    1
                ]
            )(idx)
        )
        self._jvp = jax.jit(lambda y, x, v: jax.jvp(self.residual, (y, x), (v, jnp.zeros_like(x)))[1])
        self._vjp = jax.jit(lambda y, x, v: jax.vjp(lambda z: self.residual(z, x), y)[1](v)[0])
        self._parameter_jvp = jax.jit(lambda y, x, dx: jax.jvp(self.residual, (y, x), (jnp.zeros_like(y), dx))[1])

    @staticmethod
    def _same_point(point, y, x):
        return point is not None and np.array_equal(point[0], y) and np.array_equal(point[1], x)

    @contextmanager
    def continuation_segment(self):
        """Bound speculative Newton work and roll back to the certified prefix LU.

        Factor arrays are never changed in place. Keep their references, not a
        second dense copy. Linear-work timing is cumulative even on rollback.
        The caller must certify the candidate before leaving this context.
        """
        if self._segment_dense_builds is not None:
            raise RuntimeError("nested continuation segments are not supported")
        previous = self.factors, self._factor_point, deepcopy(self.cost), self.refresh_pending
        self._segment_dense_builds = 0
        try:
            yield
        except BaseException:
            self.factors, self._factor_point, self.cost, self.refresh_pending = previous
            self.event(event="continuation_factor_rollback")
            raise
        finally:
            self._segment_dense_builds = None

    def _before_dense_rebuild(self, *, newton):
        # Initialization and strict tangent/adjoint recovery are unrestricted.
        # This only bounds work inside a speculative continuation segment.
        if newton and self._segment_dense_builds is not None:
            budget = self.options.trial_dense_rebuilds
            if budget is not None and self._segment_dense_builds >= budget:
                self.event(event="continuation_dense_budget", rebuilds=self._segment_dense_builds,
                           budget=budget)
                raise SlowProgress("trial Newton dense rebuild budget; reduce continuation step")
            self._segment_dense_builds += 1

    def _prepare_tangent(self, y, x):
        """Keep one exact-point linearization; an LU may outlive this cache."""
        if not self._same_point(self._tangent_point, y, x):
            self._tangent = self._tangent_point = None
            started = time.monotonic()
            value, tangent = _linearize(self.residual, jnp.asarray(y), jnp.asarray(x))
            value.block_until_ready()
            self._tangent = tangent
            self._tangent_point = (np.array(y, copy=True), np.array(x, copy=True))
            self.event(event="linearization", seconds=time.monotonic() - started)
        return self._tangent

    def predict(self, y, previous_x, trial_x):
        """First-order full-root response, checked against the accepted tangent."""
        y = np.asarray(y, dtype=float)
        previous_x, trial_x = np.asarray(previous_x, dtype=float), np.asarray(trial_x, dtype=float)
        forcing = np.asarray(
            self._parameter_jvp(jnp.asarray(y), jnp.asarray(previous_x), jnp.asarray(trial_x - previous_x))
        )
        return y + self.linear(y, previous_x, -forcing)

    def rebuild(self, y, x):
        started = time.monotonic()
        n = len(y)
        if not 0 < n <= self.options.max_dofs:
            raise RootFailure("coupled active dimension exceeds the explicit dense limit")
        matrix = np.empty((n, n), dtype=np.float64, order="F")
        tangent = self._prepare_tangent(y, x) if self.options.saved_linearization else None
        for start in range(0, n, self.options.batch):
            count = min(self.options.batch, n - start)
            indices = jnp.arange(self.options.batch) + start
            block = (
                _tangent_columns(tangent, jnp.asarray(y), indices)
                if tangent is not None
                else self._columns(jnp.asarray(y), jnp.asarray(x), indices)
            )
            matrix[:, start : start + count] = np.asarray(block)[:count].T
        if not np.all(np.isfinite(matrix)):
            raise RootFailure("nonfinite coupled Jacobian")
        assembly_seconds = time.monotonic() - started
        self.factors = None  # release old host factors before factoring the replacement
        self._factor_point = None
        factor_started = time.monotonic()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                self.factors = lu_factor(matrix, overwrite_a=True, check_finite=False)
        except (LinAlgWarning, ValueError, np.linalg.LinAlgError) as error:
            raise RootFailure("coupled Jacobian factorization failed") from error
        seconds = time.monotonic() - started
        factor_seconds = time.monotonic() - factor_started
        self.cost = ReuseCost(seconds, self.options.refresh_horizon)
        self._factor_point = (np.array(y, copy=True), np.array(x, copy=True))
        self.refresh_pending = False
        self.event(
            event="dense_factor",
            dofs=n,
            seconds=seconds,
            assembly_seconds=assembly_seconds,
            factor_seconds=factor_seconds,
        )

    def linear(self, y, x, rhs, *, transpose=False, newton=False):
        if newton and transpose:
            raise ValueError("Newton forcing accuracy cannot be used for an adjoint")
        rtol = self.options.newton_rtol if newton and self.options.newton_rtol is not None else self.options.rtol
        gate = self.options.newton_gate if newton and self.options.newton_gate is not None else self.options.gate
        if not 0 < rtol <= gate < 1:
            raise ValueError("linear tolerances must satisfy 0 < rtol <= gate < 1")
        rhs = np.asarray(rhs, dtype=float)
        if not np.all(np.isfinite(rhs)):
            raise RootFailure("nonfinite linear right hand side")
        if np.linalg.norm(rhs) == 0:
            return np.zeros_like(rhs)
        action = self._vjp if transpose else self._jvp

        def actual(v):
            return np.asarray(action(jnp.asarray(y), jnp.asarray(x), jnp.asarray(v)))

        for attempt in range(2):
            fresh = self.factors is None or attempt == 1 or self.refresh_pending
            if fresh:
                self._before_dense_rebuild(newton=newton)
                if self.refresh_pending:
                    self.event(event="preconditioner_refresh", reason="measured_reuse_cost")
                self.rebuild(y, x)

            if self.options.saved_krylov_actions:
                tangent = self._prepare_tangent(y, x)
                cached_action = _tangent_transpose if transpose else _tangent_apply

                def apply(v):
                    return np.asarray(cached_action(tangent, jnp.asarray(v)))
            else:
                apply = actual
            direct = fresh or (self.options.saved_linearization and self._same_point(self._factor_point, y, x))
            # First-use JVP/VJP compilation is not evidence of a stale LU.
            # Warm each action signature once, outside its Krylov time budget.
            signature = (transpose, np.shape(y), np.asarray(y).dtype.str, np.shape(x))
            if self.options.separate_action_warmup and signature not in self._warmed_actions:
                warm_started = time.monotonic()
                apply(np.zeros_like(rhs))
                if self.options.saved_krylov_actions:
                    actual(np.zeros_like(rhs))
                self._warmed_actions.add(signature)
                self.event(event="linear_action_warmup", transpose=transpose, seconds=time.monotonic() - warm_started)
            started = time.monotonic()
            iterations = 0

            def iteration(_):
                nonlocal iterations
                iterations += 1
                if not fresh and self.cost is not None and self.options.reuse_budget_fraction > 0:
                    budget = max(10.0, self.options.reuse_budget_fraction * self.cost.dense_seconds)
                    if time.monotonic() - started > budget:
                        raise _RefreshNeeded

            def solve(v):
                return lu_solve(self.factors, v, trans=int(transpose), check_finite=False)

            if direct:
                value = solve(rhs)
            else:
                try:
                    value, _ = gmres(
                        LinearOperator((len(y), len(y)), matvec=apply, dtype=float),
                        rhs,
                        M=LinearOperator((len(y), len(y)), matvec=solve, dtype=float),
                        rtol=rtol,
                        atol=0,
                        restart=self.options.restart,
                        maxiter=self.options.cycles,
                        callback=iteration,
                        callback_type="pr_norm",
                    )
                except _RefreshNeeded:
                    self.event(
                        event="preconditioner_refresh",
                        reason="krylov_time_budget",
                        seconds=time.monotonic() - started,
                        krylov_iterations=iterations,
                    )
                    continue
            defect = rhs - actual(value)
            error = np.linalg.norm(defect) / np.linalg.norm(rhs)
            self.event(
                event="linear_initial",
                transpose=transpose,
                dense=fresh,
                relative_residual=float(error) if np.isfinite(error) else None,
            )
            # Correct against the actual tangent, not only the assembled matrix.
            # Batched assembly and a full JVP can differ at roundoff level.
            for refinement in range(3):
                if not np.isfinite(error) or error <= gate:
                    break
                candidate = value + solve(defect)
                candidate_defect = rhs - actual(candidate)
                candidate_error = np.linalg.norm(candidate_defect) / np.linalg.norm(rhs)
                self.event(
                    event="linear_refinement",
                    transpose=transpose,
                    iteration=refinement + 1,
                    relative_residual=float(candidate_error) if np.isfinite(candidate_error) else None,
                )
                if not np.isfinite(candidate_error) or candidate_error >= error:
                    break
                value, defect, error = candidate, candidate_defect, candidate_error
            if direct and np.isfinite(error) and error > gate:
                correction, _ = gmres(
                    LinearOperator((len(y), len(y)), matvec=actual, dtype=float),
                    defect,
                    M=LinearOperator((len(y), len(y)), matvec=solve, dtype=float),
                    rtol=rtol,
                    atol=0,
                    restart=self.options.restart,
                    maxiter=self.options.cycles,
                )
                candidate = value + correction
                candidate_error = np.linalg.norm(rhs - actual(candidate)) / np.linalg.norm(rhs)
                if np.isfinite(candidate_error) and candidate_error < error:
                    value, error = candidate, candidate_error
            self.event(
                event="linear_solve",
                transpose=transpose,
                dense=fresh,
                direct=direct,
                purpose="newton" if newton else "adjoint" if transpose else "tangent",
                residual_gate=gate,
                relative_residual=float(error) if np.isfinite(error) else None,
                seconds=time.monotonic() - started,
                krylov_iterations=iterations,
            )
            if np.isfinite(error) and error <= gate:
                if not direct and self.cost is not None:
                    self.refresh_pending = self.cost.observe(time.monotonic() - started, transpose)
                return value
        raise RootFailure(f"coupled linear residual failed after dense recovery: {error:.6e}")

    def solve(self, y, x):
        y, x = np.array(y, dtype=float, copy=True), np.asarray(x, dtype=float)
        started = time.monotonic()
        ratios, alphas = [], []
        limit = self.options.newton_steps
        maximum = limit + self.options.extra_newton_steps
        self.last_solve = {}
        for iteration in range(maximum + 1):
            if not self.valid(y, x):
                detail = "" if self.invalid_reason is None else f": {self.invalid_reason(y, x)}"
                raise RootFailure("invalid geometry at root iterate" + detail)
            f = np.asarray(self.residual(jnp.asarray(y), jnp.asarray(x)))
            norm = np.linalg.norm(f)
            if not np.isfinite(norm):
                raise RootFailure("nonfinite coupled residual")
            if norm <= self.options.atol:
                self.last_solve = dict(
                    iterations=iteration,
                    minimum_alpha=min(alphas, default=1.0),
                    seconds=time.monotonic() - started,
                    root_residual=float(norm),
                )
                self.event(event="root_converged", **self.last_solve)
                return y
            if iteration == limit:
                rapid = len(ratios) >= 2 and max(ratios[-2:]) < 0.25 and min(alphas[-2:]) >= 0.5
                if not rapid or iteration == maximum:
                    error = SlowProgress if self.options.adaptive_newton else RootFailure
                    raise error(f"coupled root iteration budget: {norm:g}; smaller continuation required")
                limit += 1
                self.event(event="newton_extension", iterations=limit, root_residual=float(norm))
            delta = self.linear(y, x, -f, newton=True)
            for backtrack in range(12):
                alpha = 0.5**backtrack
                trial = y + alpha * delta
                if not self.valid(trial, x):
                    continue
                defect = np.asarray(self.residual(jnp.asarray(trial), jnp.asarray(x)))
                if np.all(np.isfinite(defect)) and np.linalg.norm(defect) <= (1 - 1e-4 * alpha) * norm:
                    ratios.append(float(np.linalg.norm(defect) / norm))
                    alphas.append(alpha)
                    y = trial
                    self.event(
                        event="root_polish",
                        iteration=iteration + 1,
                        before=float(norm),
                        after=float(np.linalg.norm(defect)),
                        alpha=alpha,
                    )
                    if (
                        self.options.adaptive_newton
                        and len(ratios) >= self.options.stagnation_steps
                        and min(ratios[-self.options.stagnation_steps:]) > 0.8
                        and max(alphas[-self.options.stagnation_steps:]) <= 0.125
                        and np.linalg.norm(defect) > self.options.atol
                    ):
                        raise SlowProgress("coupled Newton needs sustained strong damping; reduce continuation step")
                    break
            else:
                raise RootFailure("coupled Newton backtracking exhausted")
        raise RootFailure(f"coupled root did not reach {self.options.atol:g}: {norm:g}")

    def derivative(self, rows, y, x):
        """dG/dx = G_x - G_y F_y^{-1} F_x, including all current feedback."""
        y, x = jnp.asarray(y), jnp.asarray(x)
        values = np.asarray(rows(y, x))
        gy, gx = jax.jacrev(rows, argnums=(0, 1))(y, x)
        pull = jax.vjp(lambda p: self.residual(y, p), x)[1]
        jac = []
        for a, b in zip(np.atleast_2d(gy), np.atleast_2d(gx)):
            adjoint = self.linear(y, x, a, transpose=True)
            jac.append(b - np.asarray(pull(jnp.asarray(adjoint))[0]))
        return values, np.asarray(jac)
