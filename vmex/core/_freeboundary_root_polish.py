"""Bounded Newton refinement of the full plasma/vacuum root.

This helper never promotes an optimizer state or changes core defaults.
It retains the accepted anchor's inactive coordinates and constraint baselines.
"""
import time
from dataclasses import replace
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import implicit as im, freeboundary_implicit as fbi
from . import freeboundary_continuation as fc
from . import _freeboundary_matrixfree as mf
from ._freeboundary_dense import _active_space, _compress, _expand
from .errors import AdjointSolveError, VmecError
from .geometry import real_space_geometry, half_mesh_jacobian


class RootPolishError(VmecError):
    """A bounded numerical root-refinement failure."""


def norm(tree):
    return float(jnp.linalg.norm(ravel_pytree(tree)[0]))


def check_anchor(record, anchor):
    """The nonlinear solve and the adjoint must use the same active space."""
    for name in ('dof_mask', 'rcon0', 'zcon0'):
        a, b = getattr(record, name), getattr(anchor, name)
        if jax.tree.structure(a) != jax.tree.structure(b) or any(
                not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)):
            raise RootPolishError(f'Root-polish accepted {name} changed')


def inactive_drift(record, anchor, solver):
    project = im._dof_projector(solver.implicit, anchor.dof_mask)
    delta = jax.tree.map(jnp.subtract, record.state, anchor.state)
    return norm(jax.tree.map(jnp.subtract, delta, project(delta)))


@partial(jax.jit, static_argnames=('solver',))
def geometry_valid(state, params, *, solver):
    """Reject folded or nonfinite Newton geometries before residual evaluation."""
    rt = im.runtime_from_params(params, solver.implicit)
    coefficients = im._physical_coefficients(state, modes=rt.modes,
        lthreed=rt.setup.lthreed, lasym=rt.setup.lasym, lconm1=rt.setup.lconm1)
    geometry = real_space_geometry(**dict(zip(('R_cos', 'R_sin', 'Z_cos', 'Z_sin'), coefficients)),
        lambda_cos=state.L_cos, lambda_sin=state.L_sin, modes=rt.modes, trig=rt.trig, s=rt.setup.s_full)
    jacobian = half_mesh_jacobian(geometry, s=rt.setup.s_full)
    return (~jacobian.jacobian_sign_changed & jnp.all(jnp.isfinite(jacobian.tau))
            & (jnp.max(jnp.abs(jacobian.tau)) > 0))


def refine(accepted, cfg, preconditioner, *, anchor=None, tolerance=1e-12, max_steps=3, check_time=lambda: None):
    """Return a newly certified root; reject failed refinement explicitly."""
    started = time.perf_counter()
    if not np.isfinite(tolerance) or tolerance <= 0 or isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('positive tolerance and bounded step count required')
    solver = cfg.solver
    seed = preconditioner._seed
    if preconditioner._cfg is not cfg:
        raise ValueError('preconditioner configuration mismatch')
    anchor = accepted if anchor is None else anchor
    check_anchor(accepted, anchor)
    frozen = anchor.state
    project = im._dof_projector(solver.implicit, accepted.dof_mask)
    residual = fbi._projected_residual(solver, accepted.dof_mask)
    field = jnp.asarray(accepted.parameters)
    z = project(accepted.state)
    input_inactive_drift = inactive_drift(accepted, anchor, solver)
    space = _active_space(solver.implicit, accepted.dof_mask, solver.adjoint_dense_max_dofs)
    seed.validate(z, field, space, solver)
    space = jax.tree.map(jnp.asarray, space)
    factors = jax.tree.map(jnp.asarray, seed.factors)

    def evaluate(value):
        return residual(value, cfg.params, field, frozen, accepted.rcon0, accepted.zcon0)

    def assemble(value):
        return jax.tree.map(jnp.add, frozen, project(jax.tree.map(jnp.subtract, value, frozen)))

    if not geometry_valid(assemble(z), cfg.params, solver=solver):
        raise RootPolishError('invalid root-polish starting geometry')
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
            if not geometry_valid(assemble(trial), cfg.params, solver=solver):
                continue
            trial_f = evaluate(trial)
            trial_norm = norm(trial_f)
            if np.isfinite(trial_norm) and trial_norm < (1-1e-4*alpha)*magnitude:
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
    check_anchor(refined, anchor)
    if not np.isfinite(refined.root_residual_norm) or refined.root_residual_norm > tolerance:
        raise RootPolishError('refinement failed independently recomputed root gate')
    refined = replace(refined, result=replace(refined.result, iterations=accepted.result.iterations))
    return refined, dict(initial_residual=initial, final_residual=float(refined.root_residual_norm),
        tolerance=tolerance, steps=log, inactive_change=inactive_change,
        input_inactive_drift=input_inactive_drift,
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
