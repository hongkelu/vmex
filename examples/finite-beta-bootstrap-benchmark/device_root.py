"""Device Krylov solves of the complete equilibrium/Redl root.

The accepted LU is immutable host storage. Device copies live only during a
solve. Candidate recovery factors cannot replace it before optimizer acceptance.
Both finite-beta boundary formulations share this backend; vacuum is unchanged.
"""

from dataclasses import replace
from functools import partial
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from solvax.krylov import gmres

from vmex.core.freeboundary_problem import _LURefresh
from linear_root import LinearRoot, RootFailure, ReuseCost, _tangent_columns


@partial(jax.jit, static_argnames=("batch",))
def _assemble(tangent, template, *, batch):
    n = template.size
    blocks = jax.lax.map(
        lambda i: _tangent_columns(tangent, template, i * batch + jnp.arange(batch)),
        jnp.arange((n + batch - 1) // batch),
    )
    return blocks.reshape((-1, n))[:n].T


@partial(jax.jit, donate_argnums=(0,))
def _factor(matrix):
    return jl.lu_factor(matrix)


@partial(jax.jit, static_argnames=("transpose",))
def _direct(factors, rhs, *, transpose):
    return jl.lu_solve(factors, rhs.T, trans=int(transpose)).T


@partial(jax.jit, static_argnames=("transpose", "rtol", "restart", "cycles"))
def _krylov(tangent, factors, rhs, *, transpose, rtol, restart, cycles):
    def solve(vector):
        action = (lambda v: jax.linear_transpose(tangent, jnp.zeros_like(v))(v)[0]) if transpose else tangent
        result = gmres(
            action, vector, precond=lambda v: jl.lu_solve(factors, v, trans=int(transpose)),
            rtol=rtol, atol=0.0, restart=min(restart, vector.size), max_restarts=cycles,
        )
        return result.x, result.iterations, result.residual_norm, result.converged
    return jax.vmap(solve)(rhs)


class DeviceRoot(LinearRoot):
    """Same nonlinear root and checks, device linear solves and accepted LU policy."""

    def __init__(self, *args, rhs_batch_size=2, dense_assembly="device", **kwargs):
        super().__init__(*args, **kwargs)
        if rhs_batch_size not in (1, 2, 3, 4) or dense_assembly not in ("host", "device"):
            raise ValueError("RHS batch 1..4 and host/device assembly required")
        if not self.options.saved_linearization:
            raise ValueError("device solves require a saved current-root linearization")
        self.rhs_batch_size, self.dense_assembly = rhs_batch_size, dense_assembly
        self._accepted = None
        self._accepted_key = None
        self._trial = {}
        self._step_seconds = 0.0
        self._refresh = None
        self._compiled = set()

    def _snapshot(self):
        return self.factors, self._factor_point, self.cost, self._step_seconds

    def _restore(self, snapshot):
        self.factors, self._factor_point, self.cost, self._step_seconds = snapshot
        self.refresh_pending = False

    def initialize_accepted(self, y, x, *, remaining):
        if self._accepted is not None:
            if self._accepted_key != np.asarray(x).tobytes():
                raise ValueError("accepted factor state belongs to a different optimizer anchor")
            return
        self._accepted_key = np.asarray(x).tobytes()
        self._accepted = self._snapshot()
        dense_seconds = self.cost.dense_seconds if self.cost is not None else 0.0
        self._refresh = _LURefresh(self.options.refresh_horizon, dense_seconds, remaining=remaining)

    def begin_trial(self, x):
        if self._accepted is not None:
            self._restore(self._accepted)
        self._step_seconds = 0.0
        self._trial.clear()

    def retain_trial(self, x):
        key = np.asarray(x).tobytes()
        if key == self._accepted_key:
            if self.factors is not self._accepted[0]:
                self._refresh = _LURefresh(self.options.refresh_horizon, self.cost.dense_seconds,
                                           remaining=self._refresh.remaining)
            self._accepted = self._snapshot()
        else:
            self._trial[key] = self._snapshot()

    def activate_trial(self, x):
        key = np.asarray(x).tobytes()
        snapshot = self._trial.get(key)
        if snapshot is None and key == self._accepted_key:
            snapshot = self._accepted
        if snapshot is not None:
            self._restore(snapshot)

    def discard_trial(self):
        self._trial.clear()
        if self._accepted is not None:
            self._restore(self._accepted)

    def accept_trial(self, y, x, rows, jacobian):
        """Refresh transactionally; caller promotes physical state after success."""
        self.activate_trial(x)
        previous = self._snapshot()
        refresh = replace(self._refresh, costs=list(self._refresh.costs))
        recovered = self.factors is not self._accepted[0]
        elective = False if recovered else refresh.observe(self._step_seconds)
        reason = "recovery" if recovered else "cost" if elective else None
        try:
            if elective:
                self.event(event="preconditioner_refresh_start", reason="accepted_step_cost")
                self.rebuild(y, x)
                _, refreshed = self.derivative(rows, y, x)
                errors = np.linalg.norm(refreshed - jacobian, axis=1) / np.maximum(
                    np.linalg.norm(jacobian, axis=1), 1e-30)
                if not np.all(errors <= 1e-6):
                    raise RootFailure(f"dense refresh changed derivative rows: {errors}")
                jacobian = refreshed
                self.event(event="refresh_gradient_parity", relative_errors=errors.tolist())
            if reason is not None:
                remaining = refresh.remaining
                if recovered and remaining is not None:
                    remaining = max(0, remaining - 1)
                refresh = _LURefresh(refresh.horizon, self.cost.dense_seconds, remaining=remaining)
                self.event(event="preconditioner_refresh", reason=reason, seconds=self.cost.dense_seconds)
        except BaseException:
            self._restore(previous)
            raise
        self._refresh = refresh
        self._accepted_key = np.asarray(x).tobytes()
        self._step_seconds = 0.0
        self._accepted = self._snapshot()
        self._trial.clear()
        return jacobian

    def rebuild(self, y, x):
        if self.dense_assembly == "host":
            super().rebuild(y, x)
        else:
            if not 0 < len(y) <= self.options.max_dofs:
                raise RootFailure("coupled active dimension exceeds the explicit dense limit")
            tick = time.monotonic()
            tangent = self._prepare_tangent(y, x)
            matrix = _assemble(tangent, jnp.asarray(y), batch=self.options.batch)
            matrix.block_until_ready()
            if not bool(jnp.all(jnp.isfinite(matrix))):
                raise RootFailure("nonfinite coupled Jacobian")
            assembly = time.monotonic() - tick
            factor_started = time.monotonic()
            factors = _factor(matrix)
            host = tuple(np.array(v) for v in factors)
            if not np.all(np.isfinite(host[0])):
                raise RootFailure("nonfinite coupled LU factors")
            self.factors = host
            self._factor_point = (np.array(y, copy=True), np.array(x, copy=True))
            self.cost = ReuseCost(time.monotonic() - tick, self.options.refresh_horizon)
            self.event(event="dense_factor", backend="device", dofs=len(y), seconds=self.cost.dense_seconds,
                       assembly_seconds=assembly, factor_seconds=time.monotonic() - factor_started)
        for array in self.factors:
            array.setflags(write=False)
        self.refresh_pending = False

    def _warm(self, function, *args, **kwargs):
        # Compile separately; changed roots pass tape arrays dynamically.
        key = (function, tuple((tuple(v.shape), str(v.dtype)) for v in jax.tree.leaves(args)), tuple(kwargs.items()))
        if key not in self._compiled:
            tick = time.monotonic()
            function.lower(*args, **kwargs).compile()
            self._compiled.add(key)
            self.event(event="device_compile", kernel=function.__name__, seconds=time.monotonic() - tick)

    def _batch(self, y, x, rhs, *, transpose, newton=False):
        rtol = self.options.newton_rtol if newton else self.options.rtol
        gate = self.options.newton_gate if newton else self.options.gate
        rtol = self.options.rtol if rtol is None else rtol
        gate = self.options.gate if gate is None else gate
        if newton and transpose or not 0 < rtol <= gate < 1:
            raise ValueError("invalid Newton/adjoint linear tolerance policy")
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim != 2 or rhs.shape[1] != len(y) or not np.all(np.isfinite(rhs)):
            raise RootFailure("nonfinite or invalid linear right hand sides")
        action = self._vjp if transpose else self._jvp
        tangent = self._prepare_tangent(y, x)
        norms = np.linalg.norm(rhs, axis=1)
        signature = (transpose, np.shape(y), np.asarray(y).dtype.str, np.shape(x))
        if signature not in self._warmed_actions:
            tick = time.monotonic()
            np.asarray(action(y, x, jnp.zeros_like(y)))
            self._warmed_actions.add(signature)
            self.event(event="linear_action_warmup", transpose=transpose, seconds=time.monotonic()-tick)

        def defect(value):
            return rhs - np.stack([np.asarray(action(y, x, v)) for v in value])

        for attempt in range(2):
            if self.factors is None or attempt:
                self._before_dense_rebuild(newton=newton)
                if attempt:
                    self.event(event="dense_recovery", purpose="adjoint" if transpose else "forward")
                self.rebuild(y, x)
            factors = jax.tree.map(jnp.asarray, self.factors)
            right = jnp.asarray(rhs)
            direct = self._same_point(self._factor_point, y, x)
            if direct:
                self._warm(_direct, factors, right, transpose=transpose)
            else:
                self._warm(_krylov, tangent, factors, right, transpose=transpose,
                           rtol=rtol, restart=self.options.restart, cycles=self.options.cycles)
            started = time.monotonic()
            if direct:
                value = np.asarray(_direct(factors, right, transpose=transpose))
                iterations = np.zeros(len(rhs), dtype=int)
            else:
                result = _krylov(tangent, factors, right, transpose=transpose,
                                 rtol=rtol, restart=self.options.restart, cycles=self.options.cycles)
                value, iterations = np.asarray(result[0]), np.asarray(result[1])
            errors = np.linalg.norm(defect(value), axis=1) / np.maximum(norms, 1e-300)
            for _ in range(3):
                if not np.all(np.isfinite(errors)) or np.all(errors <= gate):
                    break
                delta = _direct(factors, jnp.asarray(defect(value)), transpose=transpose)
                candidate = value + np.asarray(delta)
                candidate_errors = np.linalg.norm(defect(candidate), axis=1) / np.maximum(norms, 1e-300)
                improved = np.isfinite(candidate_errors) & (candidate_errors < errors)
                value = np.where(improved[:, None], candidate, value)
                errors = np.where(improved, candidate_errors, errors)
                if not np.any(improved):
                    break
            if direct and np.all(np.isfinite(errors)) and np.any(errors > gate):
                # Dense roundoff can disagree with the separately applied AD
                # operator. Correct that defect with the current operator.
                correction = _krylov(tangent, factors, jnp.asarray(defect(value)), transpose=transpose,
                                     rtol=rtol, restart=self.options.restart, cycles=self.options.cycles)[0]
                candidate = value + np.asarray(correction)
                candidate_errors = np.linalg.norm(defect(candidate), axis=1) / np.maximum(norms, 1e-300)
                improved = np.isfinite(candidate_errors) & (candidate_errors < errors)
                value = np.where(improved[:, None], candidate, value)
                errors = np.where(improved, candidate_errors, errors)
            seconds = time.monotonic() - started
            self._step_seconds += seconds
            self.event(event="linear_batch_solve", backend="device", transpose=transpose, direct=direct,
                       purpose="newton" if newton else "adjoint" if transpose else "tangent",
                       seconds=seconds, rows=len(rhs), krylov_iterations=iterations.tolist(),
                       relative_residuals=[float(e) if np.isfinite(e) else None for e in errors], residual_gate=gate)
            del factors
            if np.all(np.isfinite(errors)) and np.all(errors <= gate):
                return value
        raise RootFailure(f"device linear residual failed after dense recovery: {errors}")

    def linear(self, y, x, rhs, *, transpose=False, newton=False):
        rhs = np.asarray(rhs, dtype=float)
        if np.all(rhs == 0):
            return np.zeros_like(rhs)
        return self._batch(y, x, rhs[None, :], transpose=transpose, newton=newton)[0]

    def derivative(self, rows, y, x):
        y, x = jnp.asarray(y), jnp.asarray(x)
        values = np.asarray(rows(y, x))
        gy, gx = jax.jacrev(rows, argnums=(0, 1))(y, x)
        rhs = np.atleast_2d(np.asarray(gy))
        unique, mapping = [], []
        for vector in rhs:
            if not np.any(vector):
                mapping.append((None, 0))
                continue
            match = next(((i, sign) for i, v in enumerate(unique) for sign in (1, -1)
                          if np.array_equal(vector, sign * v)), None)
            if match is None:
                match = (len(unique), 1)
                unique.append(vector)
            mapping.append(match)
        solutions = []
        for start in range(0, len(unique), self.rhs_batch_size):
            group = unique[start:start + self.rhs_batch_size]
            solutions.extend(self._batch(y, x, np.stack(group), transpose=True))
        adjoints = np.stack([np.zeros_like(rhs[0]) if i is None else sign * solutions[i] for i, sign in mapping])
        pull = jax.vjp(lambda p: self.residual(y, p), x)[1]
        jac = np.atleast_2d(np.asarray(gx)) - np.asarray(jax.vmap(lambda v: pull(v)[0])(jnp.asarray(adjoints)))
        return values, jac
