"""Bounded Newton refinement of the full plasma/vacuum root.

This helper never promotes an optimizer state or changes core defaults.
It retains the input point's inactive coordinates and constraint baselines.
"""
import time
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import implicit as im, freeboundary_implicit as fbi
from . import freeboundary_continuation as fc
from . import _freeboundary_matrixfree as mf
from ._freeboundary_dense import _active_space, _compress, _expand
from .errors import AdjointSolveError, VmecError


class RootPolishError(VmecError):
    """A bounded numerical root-refinement failure."""


def norm(tree):
    return float(jnp.linalg.norm(ravel_pytree(tree)[0]))


def refine(accepted, cfg, preconditioner, *, tolerance=1e-12, max_steps=3, check_time=lambda: None):
    """Return a newly certified root; reject failed refinement explicitly."""
    started = time.perf_counter()
    if not np.isfinite(tolerance) or tolerance <= 0 or isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('positive tolerance and bounded step count required')
    solver = cfg.solver
    seed = preconditioner._seed
    if preconditioner._cfg is not cfg:
        raise ValueError('preconditioner configuration mismatch')
    frozen = accepted.state
    project = im._dof_projector(solver.implicit, accepted.dof_mask)
    residual = fbi._projected_residual(solver, accepted.dof_mask)
    field = jnp.asarray(accepted.parameters)
    z = project(frozen)
    space = _active_space(solver.implicit, accepted.dof_mask, solver.adjoint_dense_max_dofs)
    seed.validate(z, field, space, solver)
    space = jax.tree.map(jnp.asarray, space)
    factors = jax.tree.map(jnp.asarray, seed.factors)

    def evaluate(value):
        return residual(value, cfg.params, field, frozen, accepted.rcon0, accepted.zcon0)

    f = evaluate(z)
    initial = magnitude = norm(f)
    log = []
    for iteration in range(max_steps):
        check_time()
        if magnitude <= tolerance:
            break
        tick = time.perf_counter()
        action = mf.prepare(z, cfg.params, field, frozen, accepted.rcon0, accepted.zcon0,
                            residual=residual)
        correction, its, krylov_norm, converged = mf.solve(
            action, z, space, factors, -_compress(f, space), transpose=False,
            rtol=1e-6, restart=seed.restart, max_restarts=seed.max_restarts, return_info=True)
        delta = _expand(correction, z, space)
        defect = jax.tree.map(jnp.add, action(delta), f)
        relative_linear_error = norm(defect) / magnitude
        if not np.isfinite(relative_linear_error) or relative_linear_error > 1e-5:
            raise RootPolishError(f'Newton linear residual failed: {relative_linear_error}')
        for backtrack in range(8):
            check_time()
            alpha = 0.5**backtrack
            trial = jax.tree.map(lambda a, b: a + alpha*b, z, delta)
            trial_f = evaluate(trial)
            trial_norm = norm(trial_f)
            if np.isfinite(trial_norm) and trial_norm < magnitude:
                break
        else:
            raise RootPolishError(f'Newton refinement did not reduce residual {magnitude}')
        log.append(dict(iteration=iteration+1, before=magnitude, after=trial_norm,
                        alpha=alpha, krylov_iterations=int(its), krylov_converged=bool(converged),
                        krylov_norm=float(krylov_norm), linear_relative_error=relative_linear_error,
                        seconds=time.perf_counter()-tick))
        z, f, magnitude = trial, trial_f, trial_norm
    if not np.isfinite(magnitude) or magnitude > tolerance:
        raise RootPolishError(f'Newton budget exhausted: {magnitude} > {tolerance}; steps={log}')
    displacement = project(jax.tree.map(jnp.subtract, z, frozen))
    state = jax.tree.map(jnp.add, frozen, displacement)
    inactive_change = norm(jax.tree.map(jnp.subtract, displacement, project(displacement)))
    if inactive_change > 1e-12:
        raise RootPolishError(f'inactive coordinate drift: {inactive_change}')
    tick = time.perf_counter()
    # Recompute vacuum, physical force diagnostics, geometry and root residual.
    # Never attach the original host force diagnostics to a changed state.
    refined = fc.certify_free_boundary_continuation_state(
        cfg, accepted.parameters, state, rcon0=accepted.rcon0, zcon0=accepted.zcon0)
    if not np.isfinite(refined.root_residual_norm) or refined.root_residual_norm > tolerance:
        raise RootPolishError('refinement failed independently recomputed root gate')
    refined = replace(refined, result=replace(refined.result, iterations=accepted.result.iterations))
    return refined, dict(initial_residual=initial, final_residual=float(refined.root_residual_norm),
        tolerance=tolerance, steps=log, inactive_change=inactive_change,
        state_change=norm(displacement), certification_seconds=time.perf_counter()-tick,
        seconds=time.perf_counter()-started)


def polish_with_recovery(record, cfg, preconditioner, build_dense, report, **options):
    """One local dense retry; no trial may replace the accepted preconditioner."""
    started = time.perf_counter()
    failure = None
    try:
        polished, evidence = refine(record, cfg, preconditioner, **options)
    except (RootPolishError, AdjointSolveError) as exc:
        failure = str(exc)
        report(dict(event='dense_retry', failure=failure))
        dense = seed = None
        try:
            dense, seed = build_dense(record)
            polished, evidence = refine(record, cfg, seed, **options)
        finally:
            if seed is not None:
                seed.close()
            if dense is not None:
                dense.close()
    report(dict(event='polished', recovered_with_dense=failure is not None,
                first_failure=failure, total_seconds=time.perf_counter()-started, **evidence))
    return polished
