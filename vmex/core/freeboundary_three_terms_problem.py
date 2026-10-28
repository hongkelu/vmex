"""Single-stage optimization at the three-term free boundary: a loss and constraint rows of coils and plasma
parameters, with exact design gradients (:class:`ThreeTermsFreeBoundaryProblem`).

A trial is the boundary fit of its design, warm-started from the first-order prediction of the latest linearized
point and accepted when every interface residual is within ``boundary_max_residual``; its gradient re-solves it to
the deck's tolerance, polishes it once with its own Jacobian and differentiates it (the implicit function theorem
of the boundary fit).
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

import jax
import jax.numpy as jnp

from .errors import TrialRejected, VmecError
from .freeboundary_three_terms import BoundaryResidual, ThreeTermsConfig, ThreeTermsModel
from .problem import FunctionProblem

#: The gradient's polish of a trial: its own Jacobian, to this relative cost change in at most this many evaluations,
#: kept when it cuts the cost below this fraction.
GRADIENT_POLISH_FTOL, GRADIENT_POLISH_NFEV, GRADIENT_POLISH_KEEP = 1e-4, 20, 0.9


def _nonlinear_constraint(values, jacobian, lower, upper, scales):
    """A SciPy ``NonlinearConstraint`` of ``lower <= values(x) <= upper``, rows divided by ``scales``."""
    from scipy.optimize import NonlinearConstraint

    lo, hi, scale = np.broadcast_arrays(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float),
                                        np.asarray(scales, dtype=float))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("ordered bounds and finite positive constraint scales required")
    return NonlinearConstraint(lambda x: np.asarray(values(x)) / scale, lo / scale, hi / scale,
                               jac=lambda x: np.asarray(jacobian(x)) / scale[..., None])


class ThreeTermsFreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its three-term free boundary, for SLSQP.

    Build with :meth:`from_loss` and pass it to :func:`vmex.core.optimize.minimize` (SLSQP).  A design ``x``
    sets the external field through ``field_from_parameters(x)`` (coils, currents or any differentiable field;
    coil charts live with the caller, for example in ESSOS) and, optionally, plasma parameters through
    ``plasma_from_parameters(params, x)`` (an :class:`~vmex.core.implicit.ImplicitParams` map: PHIEDGE, the
    current profile, ...).  Row 0 is ``loss(state, runtime, x)``, then the ``quantities(state, runtime)``,
    the ``parameter_quantities(state, runtime, x)`` and the ``boundary_quantities(rbc, zbs, x)``: functions of
    the trial's fitted boundary itself (INDATA arrays, the model's boundary coordinates), such as a
    coil-plasma clearance.  Every trial is the K = 0 boundary
    of the field and plasma parameters of ``x`` (:class:`ThreeTermsModel`), warm-started from the
    first-order prediction of the latest linearized solution; a trial without a solution raises
    :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design gradient of the
    implicit function theorem of the boundary least squares,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  Nothing recompiles
    when ``x`` changes.  ``model`` is the :class:`ThreeTermsModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures and their reasons, boundary evaluations and linearizations) report
    progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_quantities=(), config=None, boundary_ftol=1e-3,
                  boundary_max_nfev=40, boundary_max_residual=None, boundary_start=None):
        """Solve the seed design's free boundary (that of ``x0`` on the deck ``inp``).

        ``boundary_start`` (a deck) starts that fit from its boundary (and, with ``bootstrap``, its current) instead
        of ``inp``'s: a restart keeps its first run's deck, whose boundary is the anchored labelling's reference and
        the quadrature plan's, and starts from its final boundary.

        ``config`` (a :class:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig`) sets the labelling, grid,
        quadrature, trial tolerance and bootstrap current of the model; its fit tolerances, ``start`` and
        ``max_residual`` are the solver's, not the problem's (``boundary_max_residual`` caps the trials).  A trial's boundary fit stops at a relative cost change of
        ``boundary_ftol`` or after ``boundary_max_nfev`` evaluations; each gradient first re-solves its trial to
        the deck's tolerance and polishes it with its own Jacobian.  With ``boundary_max_residual`` a fit that
        ends with any :class:`BoundaryResidual` term above it is no solution: the trial tries its next start, or
        is rejected.
        """
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities, *boundary_quantities)
        if not all(callable(f) for f in functions) or not (plasma_from_parameters is None
                                                           or callable(plasma_from_parameters)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = ThreeTermsModel(inp, **(ThreeTermsConfig() if config is None else config).model_options())
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_quantities=tuple(boundary_quantities), boundary_ftol=boundary_ftol,
                   boundary_max_nfev=boundary_max_nfev, boundary_max_residual=boundary_max_residual,
                   boundary_start=boundary_start)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_quantities=(), boundary_ftol, boundary_max_nfev,
                 boundary_max_residual=None, boundary_start=None):
        im = model._im
        self.model, self.field = model, field_from_parameters
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_max_residual = None if boundary_max_residual is None else float(boundary_max_residual)
        x0 = np.asarray(x0, dtype=float)
        params_at = (lambda params, x: params) if plasma_from_parameters is None else plasma_from_parameters
        self._params_at = jax.jit(lambda x: params_at(model.params0, x))

        def part(f, with_x):
            def value(state, params, x):
                runtime = im.runtime_from_params(params, model.cfg)
                return jnp.ravel(jnp.asarray(f(state, runtime, x) if with_x else f(state, runtime)))
            return value

        def on_boundary(f):
            return lambda state, params, x: jnp.ravel(jnp.asarray(f(params.rbc, params.zbs, x)))

        parts = (part(loss, True), *(part(f, False) for f in quantities),
                 *(part(f, True) for f in parameter_quantities), *(on_boundary(f) for f in boundary_quantities))

        def rows(state, params, x):
            return jnp.concatenate([f(state, params, x) for f in parts])

        self._rows = jax.jit(rows)
        # The coordinates of x that move plasma parameters: their columns of the interface rows enter the gradient.
        # Structural dependence (a NaN tangent reaches every output that depends on the coordinate, even where its
        # derivative is zero, as x^2 at 0), not a numerical Jacobian at x0.
        base = model.params0
        probes = jnp.where(jnp.eye(x0.size, dtype=bool), jnp.nan, 0.0)
        tangent = jax.vmap(lambda e: jax.jvp(lambda x: params_at(base, x), (jnp.asarray(x0),), (e,))[1])(probes)
        moved = sum(np.any(np.asarray(a).reshape(x0.size, -1) != 0, axis=1) for a in jax.tree.leaves(tangent)
                    if jnp.issubdtype(a.dtype, jnp.inexact))
        self._plasma = np.flatnonzero(moved)
        eye = jnp.eye(x0.size)[self._plasma]
        # The derivatives are those of the forward map, params_at(params0, x), at every x.
        self._plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
            lambda z: params_at(base, z), (x,), (e,))[1])(eye))
        def pullback(state, mask, params, x):
            """Every row's ``ImplicitParams`` gradient, each through its own function's reverse pass only."""
            sizes = [jax.eval_shape(f, state, params, x).size for f in parts]
            return model.pullback(parts, state, mask, params, tuple(jnp.eye(k) for k in sizes), x)

        self._pullback = jax.jit(pullback)

        @jax.jit
        def coordinates(params, b, x, params_bar):
            """``ImplicitParams`` cotangents onto the model coordinates ``b`` and onto ``x`` (plasma parameters)."""
            _, b_vjp = jax.vjp(lambda z: model.with_boundary(params, z), b)
            _, x_vjp = jax.vjp(lambda z: params_at(model.params0, z), x)
            return jax.vmap(lambda bar: (b_vjp(bar)[0], x_vjp(bar)[0]))(params_bar)

        self._coordinates = coordinates
        self._direct = jax.jit(jax.jacrev(rows, argnums=2))
        self._field_vjp = jax.jit(lambda state, params, x, lam: jax.vjp(
            lambda z: model.rows(state, params, field_from_parameters(z)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx: jax.jvp(
            lambda z: model.rows(state, params, field_from_parameters(z)), (x,), (dx,))[1])
        self._cache, self._rejections, self._anchor = {}, {}, {}
        self._start = None if boundary_start is None else model.coordinates(boundary_start)
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0, rejection_reasons={})
        if self._solve(x0) is None:
            raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _solve(self, x):
        """The trial at ``x`` (cached): boundary fit, state and values, or ``None`` (the reason in
        ``_rejections``)."""
        key = np.asarray(x, dtype=float).tobytes()
        if key in self._cache:
            return self._cache[key]
        self.stats["trials"] += 1
        x = np.asarray(x, dtype=float)
        params, field = self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x))
        anchor, starts, J = self._anchor, [self._start], None
        if anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        fit = None
        for start in starts:
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, ftol=self.boundary_ftol,
                                                max_nfev=self.boundary_max_nfev,
                                                target_cost=4 * anchor["cost"] if anchor else None)
            except (VmecError, RuntimeError):  # no equilibrium on the way: try the next start
                continue
            if self.boundary_max_residual is not None:
                state, _, params_b, _ = fit["aux"]
                residual = self.model.boundary_residual(state, params_b, field)
                if max(residual.normal, residual.pressure, residual.sheet_current) > self.boundary_max_residual:
                    self.stats["boundary_evaluations"] += fit["nfev"]
                    fit = None  # the fit stalled away from the free boundary: try the next start
                    continue
            break
        if fit is None:
            return self._reject(key, "no three-term free boundary at this design point")
        self.stats["boundary_evaluations"] += fit["nfev"]
        state, mask, params_b, _ = fit["aux"]
        try:  # rows with solves of their own (a vacuum iota, say) can fail where the boundary fit did not
            values = np.asarray(self._rows(state, params_b, jnp.asarray(x)))
        except (VmecError, RuntimeError) as error:
            return self._reject(key, str(error))
        self._cache[key] = dict(fit, state=state, mask=mask, params=params_b, values=values)
        return self._cache[key]

    def _reject(self, key, reason):
        self.stats["failed"] += 1
        kind = re.sub(r"[-+]?\d[\d.]*(e[-+]?\d+)?", "#", reason)[:80]  # the reason without its numbers
        self.stats["rejection_reasons"][kind] = self.stats["rejection_reasons"].get(kind, 0) + 1
        self._cache[key], self._rejections[key] = None, reason
        return None

    def _record(self, x):
        x = np.asarray(self._x(x), dtype=float)
        sol = self._solve(x)
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected(self._rejections[x.tobytes()])
        return sol

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), re-solved tight and polished with its own Jacobian.

        The record's values become those of the tight (or polished) state it differentiates, so a value asked for
        after its gradient is that state's."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        field, nb = self.field(xj), model.x0.size

        def tight():
            extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
            try:
                J, _, _, aux = model.linearize(sol["x"], sol["aux"], field, extra=extra)
            except RuntimeError as error:  # the trial's state does not tighten to the deck's tolerance
                raise TrialRejected(str(error)) from error
            state, mask, params, _ = aux
            sol.update(aux=aux, state=state, mask=mask, params=params,
                       rows=np.asarray(model._rows(state, params, field)),
                       values=np.asarray(self._rows(state, params, xj)))
            self.stats["linearizations"] += 1
            return J

        J = tight()
        try:
            polished = model.solve_boundary(self._params_at(xj), field, x0=sol["x"], jacobian=J[:, :nb],
                                            ftol=GRADIENT_POLISH_FTOL, max_nfev=GRADIENT_POLISH_NFEV)
        except (VmecError, RuntimeError):  # no polish: the tightened fit's own Jacobian stands
            polished = None
        if polished is not None:
            self.stats["boundary_evaluations"] += polished["nfev"]
            if polished["rows"] @ polished["rows"] < GRADIENT_POLISH_KEEP * sol["rows"] @ sol["rows"]:
                sol.update(x=polished["x"], aux=polished["aux"])
                J = tight()
        J_b, J_p = J[:, :nb], J[:, nb:]
        state, params = sol["state"], sol["params"]
        params_bar = self._pullback(state, sol["mask"], params, xj)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        Q, R = np.linalg.qr(J_b)
        lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column))) for column in lam.T])
        gradients[:, self._plasma] -= lam.T @ J_p
        self._anchor = dict(x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=state,
                            params=params, cost=0.5 * sol["rows"] @ sol["rows"])
        sol["gradients"] = gradients
        return gradients

    def _value(self, x):
        return float(self._record(x)["values"][0])

    def _value_gradient(self, x):
        sol = self._record(x)
        gradients = self._linearize(x, sol)
        return float(sol["values"][0]), gradients[0].copy()

    def constraint_values(self, x):
        """The quantities' values, in their units."""
        return self._record(x)["values"][1:].copy()

    def constraint_jac(self, x):
        """The quantities' design gradients."""
        return self._linearize(x, self._record(x))[1:].copy()

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound the quantities supplied to ``from_loss``, in their units."""
        return _nonlinear_constraint(self.constraint_values, self.constraint_jac, lower, upper, scales)

    def accept_x(self, x):
        """Promote an evaluated point after the optimizer accepts it; drop the other trials."""
        key = np.asarray(self._x(x), dtype=float).tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = _AcceptedPoint(np.asarray(x, dtype=float).copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}

    def state(self, x):
        """``(state, params)`` of ``x``'s free boundary (``params`` the ``ImplicitParams`` it was solved at)."""
        sol = self._record(x)
        return sol["state"], sol["params"]

    def boundary_residual(self, x) -> BoundaryResidual:
        """The three interface conditions at ``x``'s free boundary."""
        sol = self._record(x)
        return self.model.boundary_residual(sol["state"], sol["params"], self.field(jnp.asarray(x)))

    def equilibrium_from_x(self, x, *, rtol=1e-6):
        """The free-boundary equilibrium of ``x`` as an ``Equilibrium`` (fixed-boundary solve on its boundary).

        The re-solve starts from the state of ``x`` the optimizer used; if it moves that state by more than ``rtol``
        (largest relative change of any field) a warning reports it, so the returned equilibrium never silently
        differs from the optimizer's.
        """
        from .optimize import solve_equilibrium

        sol = self._record(x)
        eq = solve_equilibrium(self.model._im.input_with_params(self.model.fixed, sol["params"]),
                               initial_state=sol["state"])
        moved = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(np.asarray(b)))), 1e-300))
                    for a, b in zip(jax.tree.leaves(eq.state), jax.tree.leaves(sol["state"])) if np.size(b))
        if not moved <= rtol:
            import warnings

            warnings.warn(f"the reporting re-solve moved the optimizer's state by {moved:.2e} (> {rtol:g})",
                          RuntimeWarning, stacklevel=2)
        return eq

    def close(self):
        """Drop the cached trials but the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray

