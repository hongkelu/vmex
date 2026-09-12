"""Accepted-state parameter continuation using the canonical coupled solver.

Each target follows its own bounded straight path from a fixed anchor. Only
converged, residual-checked states may seed the next point. Derivatives are
of the final equilibrium root, not of the continuation iterations.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import freeboundary_implicit as fbi, implicit as im
from .errors import VmecError
from .solver import SolveResult, SpectralState


@dataclass(frozen=True, eq=False)
class FreeBoundaryContinuationResult:
    """One accepted root with the exact data needed for its pullback.

    ``result`` includes the ordinary forward solver's derived output arrays.
    ``parameters`` is a read-only float64 vector. Numerical root acceptance
    does not certify physical constraints or authorize optimizer promotion.
    Use this record directly for a multi-RHS pullback, including after a memo
    refresh, so derivatives always refer to the state that produced the RHS.
    """

    parameters: np.ndarray
    result: SolveResult
    dof_mask: SpectralState
    rcon0: Any
    zcon0: Any
    root_residual_norm: float
    continuation_steps: int
    _owner: Any = field(repr=False)

    @property
    def state(self):
        return self.result.state


@dataclass
class _Runtime:
    lock: Any = field(default_factory=threading.RLock)
    memo: OrderedDict = field(default_factory=OrderedDict)
    stats: dict = field(default_factory=lambda: dict(
        host_solves=0, iterations=0, cache_hits=0, failures=0, last_path=[]))


@dataclass(frozen=True, eq=False)
class FreeBoundaryContinuationConfig:
    """Fixed profiles, field chart, accepted anchor and bounded path controls.

    Build with :func:`make_free_boundary_continuation_config`. Keep the
    solver config and field chart immutable throughout this config's life.
    Target evaluation never moves the anchor; promotion requires explicit
    :func:`reanchor_free_boundary_continuation_config` after caller acceptance.
    """

    solver: fbi.FreeBoundaryImplicitConfig
    params: im.ImplicitParams
    parameter_anchor: np.ndarray
    parameter_scales: np.ndarray
    continuation_step: float
    max_continuation_steps: int
    root_residual_atol: float
    cache_size: int
    validate_parameters: Callable | None = field(repr=False)
    _anchor: FreeBoundaryContinuationResult | None = field(repr=False)
    _owner: Any = field(default_factory=object, repr=False)
    _runtime: _Runtime = field(default_factory=_Runtime, repr=False)


def _vector(value, shape=None):
    source = np.asarray(value)
    if (source.ndim != 1 or source.size == 0 or np.iscomplexobj(source)
            or not np.issubdtype(source.dtype, np.number)):
        raise ValueError("parameters must be a nonempty real vector")
    if shape is not None and source.shape != shape:
        raise ValueError(f"parameter shape must be {shape}, got {source.shape}")
    array = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("parameters must be finite")
    # Immutable backing bytes prevent even setflags(write=True) from changing
    # an anchor, memo key, validator input, or saved parameter provenance.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _validate(cfg, parameters):
    if cfg.validate_parameters is not None:
        decision = cfg.validate_parameters(parameters)
        if decision is not None and not bool(decision):
            raise ValueError("continuation parameter validator rejected a point")


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _solve_point(cfg, parameters, previous, steps):
    solver, params = cfg.solver, cfg.params
    icfg = solver.implicit
    with im._device_context(icfg):
        external = solver.field_from_parameters(im._device_pin(icfg, jnp.asarray(parameters)))
        inp = im.input_with_params(icfg.inp, params)
        cfg._runtime.stats['host_solves'] += 1
        stage = fbi._solve_free_boundary_stage(
            inp, external_field=external, resolution=solver.resolution,
            ftol=icfg.ftol, max_iterations=icfg.max_iterations,
            initial_state=None if previous is None else previous.state,
            constraint_continuation=None if previous is None else (previous.rcon0, previous.zcon0),
            error_on_no_convergence=False, use_fft=False,
        )
        result = stage.result
        cfg._runtime.stats['iterations'] += int(result.iterations)
        forces = np.asarray([result.fsqr, result.fsqz, result.fsql], dtype=float)
        if (not result.converged or not np.all(np.isfinite(forces))
                or np.any(forces < 0) or np.any(forces > icfg.ftol)):
            raise VmecError("continuation point failed forward force convergence")
        state, mask, rcon0, zcon0 = im._device_pin(
            icfg, jax.tree.map(jnp.asarray,
                fbi._linearization_from_stage(solver, params, stage, inp=inp)))
        if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(
                (state, rcon0, zcon0))):
            raise VmecError("continuation point has non-finite state or baselines")
        residual = fbi._projected_residual(solver, mask)
        project = im._dof_projector(icfg, mask)
        defect = residual(project(state), params, jnp.asarray(parameters), state, rcon0, zcon0)
        norm = float(jnp.linalg.norm(ravel_pytree(defect)[0]))
        if not np.isfinite(norm) or norm > cfg.root_residual_atol:
            raise VmecError(f"continuation root residual {norm:.3e} exceeds {cfg.root_residual_atol:.3e}")
        return FreeBoundaryContinuationResult(
            parameters, dataclasses.replace(result, state=state), mask, rcon0, zcon0,
            norm, steps, cfg._owner)


def make_free_boundary_continuation_config(
    solver: fbi.FreeBoundaryImplicitConfig, params: im.ImplicitParams,
    parameter_anchor, *, parameter_scales=None, continuation_step: float,
    root_residual_atol: float, max_continuation_steps: int = 32,
    cache_size: int = 8, validate_parameters: Callable | None = None,
) -> FreeBoundaryContinuationConfig:
    """Solve and certify a cold anchor using the existing solver's tolerances.

    ``parameter_scales`` nondimensionalizes field coordinates; each path step
    is bounded in scaled infinity norm. ``root_residual_atol`` is deliberately
    explicit: select it and the solver's FTOL through observable convergence
    and independent gradient checks. No root polisher is invoked.
    """
    if solver.adjoint_fail != "error":
        raise ValueError("continuation requires adjoint_fail='error'")
    anchor = _vector(parameter_anchor)
    scales = _vector(np.ones_like(anchor) if parameter_scales is None else parameter_scales, anchor.shape)
    if np.any(scales <= 0):
        raise ValueError("parameter_scales must be positive")
    if validate_parameters is not None and not callable(validate_parameters):
        raise TypeError("validate_parameters must be callable")
    with im._device_context(solver.implicit):
        fixed_params = im._device_pin(solver.implicit, jax.tree.map(jnp.array, params))
    cfg = FreeBoundaryContinuationConfig(
        solver, fixed_params, anchor, scales,
        _positive(continuation_step, 'continuation_step'),
        _positive_int(max_continuation_steps, 'max_continuation_steps'),
        _positive(root_residual_atol, 'root_residual_atol'),
        _positive_int(cache_size, 'cache_size'), validate_parameters, None)
    _validate(cfg, anchor)
    root = _solve_point(cfg, anchor, None, 0)
    return dataclasses.replace(cfg, _anchor=root)


def _root(parameters, cfg, *, force_recompute=False):
    target = _vector(parameters, cfg.parameter_anchor.shape)
    runtime = cfg._runtime
    key = target.tobytes()
    with runtime.lock:
        runtime.stats['last_path'] = []
        try:
            _validate(cfg, target)  # Never bypass validation on a memo hit.
            if not force_recompute:
                if key in runtime.memo:
                    runtime.memo.move_to_end(key)
                    runtime.stats['cache_hits'] += 1
                    return runtime.memo[key]
                if key == cfg.parameter_anchor.tobytes():
                    runtime.stats['cache_hits'] += 1
                    return cfg._anchor
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                delta = target - cfg.parameter_anchor
                required = float(np.max(np.abs(delta / cfg.parameter_scales))) / cfg.continuation_step
            if not np.isfinite(required) or required > cfg.max_continuation_steps:
                raise ValueError("target exceeds the continuation step budget")
            count = max(1, int(np.ceil(required)))
            points = [_vector(target if i == count else cfg.parameter_anchor + delta*(i/count))
                      for i in range(1, count+1)]
            for point in points:
                _validate(cfg, point)
            previous = cfg._anchor
            for index, point in enumerate(points, 1):
                accepted = _solve_point(cfg, point, previous, index)
                runtime.stats['last_path'].append(dict(
                    parameters=point.tolist(), iterations=int(accepted.result.iterations),
                    root_residual_norm=accepted.root_residual_norm))
                previous = accepted
            # Commit only the successful endpoint. A failed refresh preserves
            # the old memo; failed trials never seed another target's path.
            runtime.memo[key] = accepted
            runtime.memo.move_to_end(key)
            while len(runtime.memo) > cfg.cache_size:
                runtime.memo.popitem(last=False)
            return accepted
        except Exception:
            runtime.stats['failures'] += 1
            raise


def free_boundary_continuation_result(parameters, cfg, *, force_recompute=False):
    """Return a certified endpoint and its full solver result.

    Exact targets use a bounded memo. ``force_recompute=True`` follows a fresh
    path from the unchanged anchor; a failed refresh leaves the old entry
    intact. Returned output arrays are copies so callers cannot alter memoized
    results. Retain this record for shared pullbacks.
    """
    if type(force_recompute) is not bool:
        raise TypeError("force_recompute must be a bool")
    root = _root(parameters, cfg, force_recompute=force_recompute)
    return dataclasses.replace(root, result=copy.deepcopy(root.result))


def free_boundary_continuation_stats(cfg):
    """Snapshot counters and the last path; failed partial paths remain visible."""
    with cfg._runtime.lock:
        return copy.deepcopy(cfg._runtime.stats)


def reanchor_free_boundary_continuation_config(cfg, accepted):
    """Explicitly start a new continuation context at a caller-accepted root.

    Call only after the optimizer's objective/constraint acceptance checks.
    This neither solves again nor mutates the old anchor, memo or saved VJPs.
    """
    if accepted._owner is not cfg._owner:
        raise ValueError("accepted root belongs to a different continuation config")
    _validate(cfg, accepted.parameters)
    owner = object()
    anchor = dataclasses.replace(accepted, result=copy.deepcopy(accepted.result),
                                 continuation_steps=0, _owner=owner)
    return dataclasses.replace(cfg, parameter_anchor=accepted.parameters,
                               _anchor=anchor, _owner=owner, _runtime=_Runtime())


def free_boundary_continuation_state_pullback(accepted, cfg, state_cotangents):
    """Shared implicit field-parameter derivatives at the supplied exact root.

    Cotangent leaves have a leading RHS axis. This delegates to main's shared
    coupled/reverse GCROT pullback; add explicit objective field derivatives yourself.
    The saved record prevents a later memo refresh from changing the root.
    """
    if accepted._owner is not cfg._owner:
        raise ValueError("accepted root belongs to a different continuation config")
    _, field_bar = fbi.free_boundary_state_pullback_multi_rhs(
        cfg.params, jnp.asarray(accepted.parameters), cfg.solver,
        accepted.state, accepted.dof_mask, state_cotangents,
        rcon0=accepted.rcon0, zcon0=accepted.zcon0,
        root_residual_atol=cfg.root_residual_atol)
    return field_bar


def _host_callback(cfg, parameters):
    accepted = _root(parameters, cfg)
    return jax.tree.map(np.asarray, (accepted.state, accepted.dof_mask,
                                    accepted.rcon0, accepted.zcon0))


def _callback(parameters, cfg):
    icfg = cfg.solver.implicit
    rcon, zcon = fbi._baseline_struct(cfg.solver)
    return jax.pure_callback(
        functools.partial(_host_callback, cfg),
        (im._state_struct(icfg), im._state_struct(icfg), rcon, zcon),
        parameters, sharding=im._callback_sharding(icfg))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _solve(parameters, cfg):
    with im._device_context(cfg.solver.implicit):
        return _callback(im._device_pin(cfg.solver.implicit, parameters), cfg)[0]


def _solve_fwd(parameters, cfg):
    with im._device_context(cfg.solver.implicit):
        parameters = im._device_pin(cfg.solver.implicit, parameters)
        state, mask, rcon0, zcon0 = _callback(parameters, cfg)
    return state, (cfg.params, parameters, state, mask, rcon0, zcon0)


def _solve_bwd(cfg, saved, state_bar):
    # Use the exact forward record, never re-fetch a potentially refreshed
    # memo. Main owns both the eager and traced scalar adjoint algorithms.
    _, field_bar = fbi._solve_bwd(cfg.solver, saved, state_bar)
    return (field_bar,)


_solve.defvjp(_solve_fwd, _solve_bwd)


def solve_free_boundary_continuation(parameters, cfg):
    """Differentiable endpoint; profiles are fixed and field coordinates vary.

    Ordinary JAX composition adds explicit objective dependence. The backward
    pass reuses main's scalar custom-VJP implementation at the saved endpoint;
    no derivative is taken through anchor selection or continuation steps.
    """
    values = jnp.asarray(parameters)
    if values.shape != cfg.parameter_anchor.shape or not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError("parameters must be a floating vector with the anchor shape")
    return _solve(values.astype(jnp.float64), cfg)


__all__ = [
    'FreeBoundaryContinuationConfig', 'FreeBoundaryContinuationResult',
    'make_free_boundary_continuation_config', 'free_boundary_continuation_result',
    'free_boundary_continuation_stats', 'reanchor_free_boundary_continuation_config',
    'free_boundary_continuation_state_pullback', 'solve_free_boundary_continuation',
]
