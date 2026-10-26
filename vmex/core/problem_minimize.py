"""SciPy SLSQP on a :class:`~vmex.core.problem.FunctionProblem`, with an exact resume of its state.

:func:`vmex.core.optimize.minimize` dispatches a ``FunctionProblem`` here (:func:`_minimize_problem`).  SLSQP's
whole state (quasi-Newton Hessian, multipliers, line-search memory) is saved after every accepted iterate
(:func:`save_slsqp_state`) and restored on a resume (:func:`load_slsqp_state`), so a run split into segments
follows the same iterates as an uninterrupted one.  A problem exposing ``accept_x`` promotes only accepted
iterates.
"""

from __future__ import annotations

import contextlib
import inspect

import numpy as np

from .errors import TrialRejected

__all__ = ["load_slsqp_state", "save_slsqp_state"]


#: Distance outside its bounds at which a constraint row is reported for a
#: rejected SLSQP line-search probe, so the probe is refused without a
#: fabricated Jacobian.
_REJECTED_CONSTRAINT_OFFSET = 1e6


_SLSQP_ARRAYS = ("buffer", "indices", "mult", "u", "origin", "scales", "accepted")


def save_slsqp_state(path, state):
    """Write an SLSQP snapshot (``options["state_callback"]`` of :func:`minimize`) to ``path`` (``.npz``)."""
    np.savez(path, **{name: np.asarray(state[name]) for name in _SLSQP_ARRAYS},
             **{f"state.{key}": np.asarray(value) for key, value in state["state"].items()})


def load_slsqp_state(path):
    """An SLSQP snapshot written by :func:`save_slsqp_state`, for ``options["resume"]`` of :func:`minimize`."""
    with np.load(path) as data:
        state = {name: np.array(data[name]) for name in _SLSQP_ARRAYS}
        state["state"] = {key.split(".", 1)[1]: data[key].item() for key in data.files if key.startswith("state.")}
    return state


class _SlsqpHook:
    """Snapshot and restore SciPy SLSQP's whole state around its reverse-communication routine.

    SciPy's loop calls ``slsqp(state, fx, g, C, d, x, mult, xl, xu, buffer, indices)``, and everything the method
    carries between calls (the quasi-Newton factor of the Lagrangian Hessian, the line search, the multipliers) is
    in ``state``, ``buffer``, ``indices`` and ``mult``.  A call that returns asking for a gradient (``mode == -1``)
    has accepted a major iterate: its arrays are a snapshot from which the method continues exactly.  The first call of
    a resumed run loads one instead of starting from an identity Hessian; the objective, constraints and gradients
    at the point are the ones SciPy evaluated there.
    """

    def __init__(self, resume, state_callback, origin, scales, accepted_of):
        self.resume, self.state_callback = resume, state_callback
        self.origin, self.scales, self.accepted_of = origin, scales, accepted_of

    def __enter__(self):
        from scipy.optimize import _slsqp_py

        if not callable(getattr(_slsqp_py, "slsqp", None)):  # the resume hooks SciPy's SLSQP iteration routine
            raise ImportError("SLSQP state resume needs scipy.optimize._slsqp_py.slsqp (SciPy >= 1.16); "
                              f"this SciPy ({__import__('scipy').__version__}) does not provide it")
        self.module, self.real = _slsqp_py, _slsqp_py.slsqp
        _slsqp_py.slsqp = self
        return self

    def __exit__(self, *exc):
        self.module.slsqp = self.real
        return False

    def __call__(self, state, fx, g, C, d, x, mult, xl, xu, buffer, indices):
        if self.resume is not None:
            saved, self.resume = self.resume, None
            for name, array in (("buffer", buffer), ("indices", indices), ("mult", mult)):
                if saved[name].shape != array.shape:
                    raise ValueError(f"the SLSQP state is of another problem: {name} {saved[name].shape} "
                                     f"vs {array.shape}")
                array[...] = saved[name]
            budget = state["itermax"] - state["iter"]
            state.update({key: type(state[key])(value) for key, value in saved["state"].items()})
            state["itermax"] = state["iter"] + budget
        out = self.real(state, fx, g, C, d, x, mult, xl, xu, buffer, indices)
        if state["mode"] == -1 and self.state_callback is not None:
            # a gradient request: x is the accepted iterate (its Jacobian request promotes it), and the next call
            # continues from these arrays with the gradient there
            self.state_callback(dict(state=dict(state), buffer=buffer.copy(), indices=indices.copy(),
                                     mult=mult.copy(), u=np.array(x), origin=self.origin.copy(),
                                     scales=self.scales.copy(), accepted=self.accepted_of(x)))
        return out


def _minimize_problem(problem, *, x0=None, method="SLSQP", bounds=None,
                      constraints=(), callback=None, options=None, tol=None):
    """Run SciPy SLSQP on a :class:`FunctionProblem` in scaled coordinates.

    SciPy iterates on ``u = (x - x0) / problem.scales``; bounds, constraints,
    callbacks and the returned ``x``/``jac`` stay in the problem's units. A
    problem exposing ``accept_x`` (a
    :class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`) must start
    at its accepted point, and each SLSQP major iterate is promoted with
    ``accept_x`` when its Jacobian is requested.

    Parameters
    ----------
    problem:
        The problem; ``problem.bounds`` is used when ``bounds`` is ``None``.
    x0:
        Start point (default ``problem.x0``).
    method:
        Must be ``"SLSQP"``.
    bounds, constraints, tol:
        As for :func:`scipy.optimize.minimize`; constraints are
        :class:`~scipy.optimize.LinearConstraint` or
        :class:`~scipy.optimize.NonlinearConstraint` with an analytic
        Jacobian. A constraint value that raises :class:`~vmex.core.errors.TrialRejected`
        reports a point far outside its bounds.
    callback:
        Called after each accepted step with ``x``, or with an
        :class:`~scipy.optimize.OptimizeResult` if its only parameter is
        named ``intermediate_result``; ``StopIteration`` stops the run.
    options:
        SciPy SLSQP options; ``maxiter`` (default 100) is the budget of
        accepted steps for a stateful problem.  Also ``state_callback``
        (called after each accepted major iterate with a snapshot of the
        method's whole state: :func:`save_slsqp_state`) and ``resume`` (such
        a snapshot of the start point: the run continues with its
        quasi-Newton Hessian, multipliers, coordinates and iteration count
        instead of an identity Hessian; :func:`load_slsqp_state`).  SciPy
        re-evaluates the objective, constraints and gradients at the start
        point, so a resumed run uses the problem's current values there.  A
        snapshot of another point or problem size raises ``ValueError``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        With ``x``, ``fun`` and ``jac`` at the last accepted point for a
        stateful problem (or on success), plus ``accepted_steps`` and
        ``stop_reason``: ``None`` when SciPy stopped on its own,
        ``"accepted_step_budget_reached"``, ``"callback_stopped"`` or
        ``"equilibrium_trial_rejected"``. The last three set
        ``success=False`` and ``status=99``.
    """
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult
    from scipy.optimize import minimize as scipy_minimize
    from scipy.sparse import issparse

    if method != "SLSQP":
        raise ValueError("minimize(FunctionProblem) supports only method='SLSQP'")
    start = np.asarray(problem.x0 if x0 is None else x0, dtype=float).copy()
    scales = np.asarray(problem.scales, dtype=float)
    if (start.shape != scales.shape or start.ndim != 1 or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(scales) & (scales > 0))):
        raise ValueError("x0 must be finite and match finite positive problem scales")
    promote = getattr(problem, "accept_x", None)
    if promote is not None and not np.array_equal(start, problem.accepted.parameters):
        raise ValueError("stateful minimization must start at the accepted equilibrium")
    options = dict(options or {})
    resume = options.pop("resume", None)
    state_callback = options.pop("state_callback", None)
    x_start = start
    if resume is not None:
        # SciPy's coordinates are the snapshot's (its origin and scales: the quasi-Newton factor is in them), and
        # the snapshot must be of this point, to the mapping's round-off
        origin, saved_scales = np.asarray(resume["origin"], dtype=float), np.asarray(resume["scales"], dtype=float)
        if origin.shape != start.shape or saved_scales.shape != start.shape:
            raise ValueError("the SLSQP state is of another problem size")
        u_mapped = (start - origin) / saved_scales
        roundoff = 4 * np.finfo(float).eps * np.maximum(np.abs(start), np.abs(origin)) / saved_scales
        if not np.all(np.abs(u_mapped - np.asarray(resume["u"])) <= roundoff):
            raise ValueError("the SLSQP state is of another point than x0")
        start, scales = origin, saved_scales
    u_start = np.zeros_like(start) if resume is None else np.asarray(resume["u"], dtype=float).copy()

    def x_of(u):
        """The problem's coordinates of SciPy's ``u``: the start point itself at the start's ``u`` (its round-off
        image would be another point, solved again)."""
        u = np.asarray(u)
        return x_start.copy() if np.array_equal(u, u_start) else start + scales * u

    budget = int(options.get("maxiter", 100))
    if budget < 0:
        raise ValueError("maxiter must be nonnegative")
    if promote is not None:
        # The step budget is enforced by accept(); one spare SLSQP iteration
        # lets the last accepted step request its Jacobian.
        options["maxiter"] = budget + 1
    if bounds is None:
        bounds = problem.bounds
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bounds = Bounds(*bounds)
    if bounds is not None:
        if not isinstance(bounds, Bounds):
            pairs = [(float('-inf') if lo is None else lo,
                      float('inf') if hi is None else hi) for lo, hi in bounds]
            bounds = Bounds(*np.asarray(pairs).T)
        bounds = Bounds((np.asarray(bounds.lb)-start)/scales, (np.asarray(bounds.ub)-start)/scales,
                        keep_feasible=bounds.keep_feasible)
    if isinstance(constraints, (LinearConstraint, NonlinearConstraint, dict)):
        constraints = (constraints,)
    scaled_constraints = []
    for constraint in tuple(constraints):
        if isinstance(constraint, (LinearConstraint, NonlinearConstraint)) and (
                np.all(np.isneginf(constraint.lb)) and np.all(np.isposinf(constraint.ub))):
            continue
        if isinstance(constraint, LinearConstraint):
            matrix = constraint.A.multiply(scales) if issparse(constraint.A) else constraint.A * scales
            scaled_constraints.append(LinearConstraint(
                matrix, constraint.lb-constraint.A@start, constraint.ub-constraint.A@start))
        elif isinstance(constraint, NonlinearConstraint):
            if not callable(constraint.jac):
                raise ValueError("constraints require an analytic Jacobian")
            def values(u, c=constraint):
                try:
                    return c.fun(x_of(u))
                except TrialRejected:
                    # Values can reject a line-search probe; never fabricate its Jacobian.
                    lo, hi = np.broadcast_arrays(c.lb, c.ub)
                    offset = _REJECTED_CONSTRAINT_OFFSET
                    return np.where(np.isfinite(lo), lo-offset, np.where(np.isfinite(hi), hi+offset, 0.))
            scaled_constraints.append(NonlinearConstraint(
                values, constraint.lb, constraint.ub,
                jac=lambda u, c=constraint: np.asarray(c.jac(x_of(u)))*scales))
        else:
            raise TypeError("use scipy LinearConstraint or NonlinearConstraint")

    accepted = x_start.copy()
    value, gradient = problem.value_and_grad(accepted)
    accepted_value, accepted_gradient = value, np.asarray(gradient).copy()
    accepted_steps = 0
    nfev, njev = 1, 1
    if promote is not None and budget == 0:
        return OptimizeResult(x=x_start, fun=value, jac=gradient, success=False,
            status=99, message="accepted_step_budget_reached", stop_reason="accepted_step_budget_reached",
            accepted_steps=0, nit=0, nfev=1, njev=1)

    class _Stop(Exception):
        pass

    def accept(u):
        nonlocal accepted, accepted_value, accepted_gradient, accepted_steps
        x = x_of(u)
        if np.array_equal(x, accepted):
            return
        current_value, current_gradient = problem.value_and_grad(x)
        if promote is not None:
            promote(x)
        accepted, accepted_value = x.copy(), current_value
        accepted_gradient = np.asarray(current_gradient).copy()
        accepted_steps += 1
        if callback is not None:
            try:
                if set(inspect.signature(callback).parameters) == {"intermediate_result"}:
                    callback(intermediate_result=OptimizeResult(
                        x=accepted.copy(), fun=accepted_value, nit=accepted_steps))
                else:
                    callback(accepted.copy())
            except StopIteration as error:
                raise _Stop("callback_stopped") from error
        if promote is not None and accepted_steps >= budget:
            raise _Stop("accepted_step_budget_reached")

    def fun(u):
        nonlocal nfev
        nfev += 1
        try:
            return problem.fun(x_of(u))
        except TrialRejected:
            return np.inf

    def jac(u):
        nonlocal nfev, njev
        nfev += 1
        njev += 1
        _, gradient = problem.value_and_grad(x_of(u))
        gradient = np.asarray(gradient)*scales
        # SLSQP's major-iteration callback can precede backtracking. The next
        # Jacobian request identifies its accepted line-search point instead.
        accept(u)
        return gradient

    try:
        hook = _SlsqpHook(resume, state_callback, start, scales, x_of)
        with hook if (resume is not None or state_callback is not None) else contextlib.nullcontext():
            result = scipy_minimize(fun, u_start, jac=jac, method="SLSQP",
                bounds=bounds, constraints=scaled_constraints, options=options, tol=tol)
        if result.success:
            accept(result.x)
        result.x = x_of(result.x)
        if getattr(result, "jac", None) is not None:
            result.jac = np.asarray(result.jac)/scales
        result.stop_reason = None
    except (_Stop, TrialRejected) as error:
        reason = str(error) if isinstance(error, _Stop) else "equilibrium_trial_rejected"
        result = OptimizeResult(success=False, status=99, message=str(error),
            stop_reason=reason, nit=accepted_steps, nfev=nfev, njev=njev)
    if result.success or promote is not None or getattr(result, "stop_reason", None):
        result.x, result.fun, result.jac = accepted.copy(), accepted_value, accepted_gradient.copy()
    result.accepted_steps = accepted_steps
    return result
