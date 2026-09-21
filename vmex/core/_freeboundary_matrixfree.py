"""Current-root Krylov adjoints and tangents with an explicitly retained seed LU.

Only the preconditioner is reused. Numerical tapes are rebuilt at every root,
and acceptance checks use the full projected operator, not the old factors.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from jax.flatten_util import ravel_pytree
from solvax.krylov import gmres

from . import implicit as im
from ._freeboundary_dense import DenseRootLinearization, _active_space, _compress, _expand


def _signature(tree):
    return jax.tree.structure(tree), [(x.shape, x.dtype) for x in jax.tree.leaves(tree)]


@dataclass(eq=False)
class SeedLU:
    """An immutable host copy of certified factors, independent of root lifetime."""

    factors: object
    space: object
    signature: object
    field_shape: tuple
    cfg: object
    rtol: float
    restart: int
    max_restarts: int

    @classmethod
    def from_root(cls, root, *, rtol=1e-11, restart=30, max_restarts=10):
        """Snapshot a dense seed without retaining its numerical differentiation tape."""
        if type(root) is not DenseRootLinearization or root.factors is None:
            raise ValueError("a live dense linearization is required to create a seed LU")
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("Krylov rtol must be finite and in (0, 1)")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (restart, max_restarts)):
            raise ValueError("restart and max_restarts must be positive integers")
        factors = tuple(np.array(x, copy=True) for x in root.factors)
        space = jax.tree.map(lambda x: np.array(x, copy=True), root.space)
        if factors[0].dtype != np.float64:
            raise TypeError("seed LU requires float64")
        for value in (*factors, *space):
            value.setflags(write=False)
        return cls(factors, space, _signature(root.z), root.field.shape, root.cfg, rtol, restart, max_restarts)

    def validate(self, z, field, space, cfg):
        """Reject closed seeds and changes to the solver, state layout or active basis."""
        if self.factors is None:
            raise ValueError("seed LU preconditioner is closed")
        if cfg is not self.cfg or _signature(z) != self.signature or field.shape != self.field_shape:
            raise ValueError("seed LU belongs to a different solver or state layout")
        if any(not np.array_equal(a, b) for a, b in zip(space, self.space)):
            raise ValueError("seed LU active space differs from the current root")

    def close(self):
        """Release the seed; already-created root linearizations remain independent."""
        self.factors = self.space = self.signature = self.cfg = None


@jax.jit
def _forward_from_transpose(transpose, template, vector):
    return jax.linear_transpose(lambda value: transpose(value)[0], template)(vector)[0]


def prepare(z, params, field, frozen, rcon, zcon, *, residual):
    """Keep tape arrays dynamic so changed roots do not create new JIT callables."""
    from .freeboundary_implicit import _prepare_reverse_transpose

    transpose = _prepare_reverse_transpose(z, params, field, frozen, rcon, zcon, residual=residual)
    return jax.tree_util.Partial(_forward_from_transpose, transpose, z)


@partial(jax.jit, static_argnames=("transpose", "rtol", "restart", "max_restarts"))
def solve(action, template, space, factors, rhs, *, transpose, rtol, restart, max_restarts):
    """Bounded right-preconditioned FGMRES; the caller certifies the full equation."""

    def operator(vector):
        value = _expand(vector, template, space)
        result = jax.linear_transpose(action, template)(value)[0] if transpose else action(value)
        return _compress(result, space)

    answer = gmres(
        operator,
        rhs,
        precond=lambda value: jl.lu_solve(factors, value, trans=int(transpose)),
        rtol=rtol,
        atol=0.0,
        restart=min(restart, rhs.size),
        max_restarts=max_restarts,
    )
    return answer.x, answer.iterations


@dataclass(eq=False)
class MatrixFreeRootLinearization(DenseRootLinearization):
    """Retain a current-root tape and the seed factors for checked predictor solves."""

    rtol: float = 1e-11
    restart: int = 30
    max_restarts: int = 10
    _tangent_backend = "matrixfree_seed_lu"

    def _solve_tangent(self, rhs):
        solution, iterations = solve(
            self.action,
            self.z,
            self.space,
            self.factors,
            -_compress(rhs, self.space),
            transpose=False,
            rtol=self.rtol,
            restart=self.restart,
            max_restarts=self.max_restarts,
        )
        return _expand(solution, self.z, self.space), int(iterations)


def solve_matrixfree_adjoint(
    residual,
    z,
    params,
    field,
    frozen,
    rcon,
    zcon,
    rhs_batch,
    mask,
    cfg,
    *,
    preconditioner,
    diagnostics=None,
    return_linearization=False,
):
    """Solve arbitrary RHS batches; reuse only exact equal/opposite RHS solutions."""
    if cfg.adjoint_solver != "forward_dense_jax" or cfg.adjoint_fail != "error":
        raise ValueError("seed LU requires forward_dense_jax with adjoint_fail=error")
    space = _active_space(cfg.implicit, mask, cfg.adjoint_dense_max_dofs)
    preconditioner.validate(z, field, space, cfg)
    space = jax.tree.map(jnp.asarray, space)
    factors = jax.tree.map(jnp.asarray, preconditioner.factors)
    action = prepare(z, params, field, frozen, rcon, zcon, residual=residual)
    reduced = jax.vmap(lambda value: _compress(value, space))(rhs_batch)
    options = dict(rtol=preconditioner.rtol, restart=preconditioner.restart, max_restarts=preconditioner.max_restarts)
    adjoints, cache = [], {}
    for row in range(reduced.shape[0]):
        vector = reduced[row]
        host = np.asarray(vector)
        if not np.all(np.isfinite(host)):
            raise ValueError("nonfinite adjoint right-hand side")
        key, opposite = host.tobytes(), (-host).tobytes()
        reused = key in cache or opposite in cache
        if not np.any(host):
            solution, iterations = jnp.zeros_like(vector), 0
        elif reused:
            solution = cache[key] if key in cache else -cache[opposite]
            iterations = 0
        else:
            solution, iterations = solve(action, z, space, factors, vector, transpose=True, **options)
        adjoint = _expand(solution, z, space)
        rhs = jax.tree.map(lambda value: value[row], rhs_batch)
        applied = jax.linear_transpose(action, z)(adjoint)[0]
        defect = jax.tree.map(jnp.subtract, applied, rhs)
        norm = float(jnp.linalg.norm(ravel_pytree(defect)[0]))
        rhs_norm = float(jnp.linalg.norm(ravel_pytree(rhs)[0]))
        rtol = cfg.adjoint_residual_rtol
        tolerance = float(im._adjoint_acceptance(cfg.implicit, rhs_norm)) if rtol is None else rtol * rhs_norm
        passed = bool(jnp.all(jnp.isfinite(solution))) and np.isfinite(norm) and norm <= tolerance
        if diagnostics is not None:
            diagnostics.append(
                dict(
                    row=row,
                    residual_norm=norm,
                    rhs_norm=rhs_norm,
                    relative_residual=norm / rhs_norm if rhs_norm else (0.0 if norm == 0 else float("inf")),
                    tolerance=tolerance,
                    iterations=int(iterations),
                    accepted=passed,
                    backend="matrixfree_seed_lu",
                    reused_rhs=reused,
                )
            )
        if not passed:
            im._raise_adjoint_unconverged(
                cfg.implicit,
                iterations=int(iterations),
                residual_norm=norm,
                tolerance=tolerance,
                method=f"matrixfree_seed_lu row {row}",
            )
        cache[key] = solution
        adjoints.append(adjoint)
    result = jax.tree.map(lambda *values: jnp.stack(values), *adjoints)
    if return_linearization:
        root = MatrixFreeRootLinearization(
            residual, z, params, field, frozen, rcon, zcon, space, preconditioner.factors, action, cfg, **options
        )
        return result, root
    return result
