"""Single-stage optimization at the three-term free boundary: a loss and constraint rows of coils and plasma
parameters, with exact design gradients (:class:`ThreeTermsFreeBoundaryProblem`).

A trial is the boundary fit of its design, warm-started from the first-order prediction of the latest linearized
point and accepted when every interface residual is within ``boundary_max_residual``; its gradient re-solves it to
the deck's tolerance, polishes it once with its own Jacobian and differentiates it (the implicit function theorem
of the boundary fit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import scipy.linalg

import jax
import jax.numpy as jnp

from .errors import TrialRejected, VmecError
from .freeboundary_three_terms import BoundaryResidual, ThreeTermsConfig, ThreeTermsModel
from .problem import FunctionProblem

#: The gradient's polish of a trial: its own Jacobian, to this relative cost change in at most this many evaluations,
#: kept when it cuts the cost below this fraction.
GRADIENT_POLISH_FTOL, GRADIENT_POLISH_NFEV, GRADIENT_POLISH_KEEP = 1e-4, 20, 0.9
#: The Newton products' (``gradient_rounds``, ``gradient_fit_xtol``) one-sided difference step in the column-scaled
#: model coordinates.
NEWTON_GRADIENT_EPS = 1e-4
#: The Newton gradient's soft directions: eigenvectors of the column-scaled Gauss-Newton matrix with an eigenvalue below
#: this fraction of the largest.  A difference product cannot resolve them (a near-axisymmetric plasma's radial shift has
#: 1e-7): they are solved by Gauss-Newton, whose residual curvature is negligible along them.
SOFT_DIRECTION_TOL = 1e-5
#: ``gradient_fit_xtol``: the fit's Newton steps before its gradient: at most this many trial solves in all; a step
#: of the curvature model that cuts the Gauss-Newton step by less than ``GRADIENT_FIT_CUT`` is followed by one of
#: GMRES on the products, to this relative residual in at most this many products.  A step is halved at most
#: ``GRADIENT_FIT_HALVINGS`` times, then the Gauss-Newton step at most ``GRADIENT_FIT_GN_HALVINGS`` times.
GRADIENT_FIT_STEPS, GRADIENT_FIT_RTOL, GRADIENT_FIT_PRODUCTS, GRADIENT_FIT_CUT = 30, 1e-2, 12, 0.3
GRADIENT_FIT_HALVINGS, GRADIENT_FIT_GN_HALVINGS = 2, 6
#: A full step that lowers the cost but enlarges the Gauss-Newton step (``J_b`` stale) re-linearizes the fit where
#: it lands, at most this many times a fit.
GRADIENT_FIT_REFRESHES = 3
#: The residual-curvature memory: at most this many of the latest measured directions.
CURVATURE_MEMORY = 64


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
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  This Gauss-Newton form
    leaves out the rows' curvature times the fit's residual, which the fit's stationarity ``F = J_b^T r = 0``
    keeps: with ``gradient_rounds`` the gradient is instead the exact

        dh/dx = dh/dx|_b - mu^T dF/dx,   mu = H^-1 dh/db,   H = dF/db,

    every row's ``mu`` in one block Krylov space preconditioned by the Gauss-Newton matrix, a round adding a direction
    per row still above the relative residual ``gradient_rtol``.  ``H v`` and ``dF/dx^T v`` come together from one-sided
    differences of the fit cost's gradient (one tight solve and one reverse pass, no Jacobian).  With
    ``gradient_fit_xtol`` the fit is first converged to its stationarity, by Newton steps until the column-scaled
    Gauss-Newton step is below it, since both forms assume ``F = 0``.  With ``gradient_certify`` a trial other than the
    accepted point is linearized when first evaluated and rejected unless its fit ends below ``gradient_certify`` times
    ``gradient_fit_xtol`` and its Newton rows below that times ``gradient_rtol``: the line search then shortens the step
    until the equilibrium is converged and the gradient exact.  Nothing recompiles when ``x`` changes.  ``model`` is the
    :class:`ThreeTermsModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures and their reasons, boundary evaluations and linearizations) report
    progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_quantities=(), config=None, boundary_ftol=1e-3,
                  boundary_max_nfev=40, boundary_target_cost=4.0, boundary_max_residual=None, boundary_start=None,
                  gauge_refresh=0, gauge_max_share=None, gauge_release=False, gradient_rounds=0, gradient_rtol=1e-3,
                  gradient_fit_xtol=None, gradient_certify=None, promote_trials=False):
        """Solve the seed design's free boundary (that of ``x0`` on the deck ``inp``).

        ``boundary_start`` (a run's final WOUT, or a deck: its fitted boundary and, with ``bootstrap``, current) is
        the seed design's boundary as it is, re-fitted only if it misses ``boundary_max_residual``, and the start of
        the fits before the first gradient; a WOUT's equilibrium also seeds the seed design's solve (the interior
        state is not unique to the force tolerance).  A restart keeps its first run's deck ``inp`` (the anchored
        labelling's reference and the quadrature plan's boundary) and resumes at its final WOUT.

        ``config`` (a :class:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig`) sets the labelling, grid,
        quadrature, trial tolerance and bootstrap current of the model; its fit tolerances, ``start`` and
        ``max_residual`` are the solver's, not the problem's (``boundary_max_residual`` caps the trials).  A trial's
        boundary fit reuses the accepted point's Jacobian, stopping at a relative cost change of ``boundary_ftol`` or
        after ``boundary_max_nfev`` evaluations, and is taken as it ends if its cost is below ``boundary_target_cost``
        times the accepted point's (``None``: always); otherwise it is refitted with fresh Jacobians.  Fits on a
        reused Jacobian end off the trial's own free boundary by an amount that varies from trial to trial, a noise
        of the order of the interface residual on every value a line search compares; ``boundary_target_cost=0``
        refits every trial with its own Jacobians.  Each gradient first re-solves its trial to
        the deck's tolerance and polishes it with its own Jacobian.  With ``boundary_max_residual`` a fit that
        ends with any :class:`BoundaryResidual` term above it is no solution: the trial tries its next start, or
        is rejected.  With a gauge (labelling ``"flat_arc"``), set at the seed's fit, ``gauge_refresh=K`` sets it again
        at the accepted fit every ``K`` accepted steps (:meth:`accept_x`), and ``gauge_max_share`` whenever the gauge
        rows' share of the accepted fit's cost exceeds it, so it follows the design: its rows vanish there, so no value
        at the accepted point changes.  With ``gauge_release`` each of those refreshes first re-fits the accepted
        boundary without the gauge rows (the no-label fit), so the labelling does not accumulate: the accepted point's
        values then become the no-label fit's.  ``gradient_rounds`` (at most that many block Krylov rounds, to the relative
        residual ``gradient_rtol``) gives the exact Newton gradient instead of the Gauss-Newton one (0);
        ``gradient_certify`` (a factor, with both) rejects a trial whose fit or Newton rows end above that factor times
        their tolerances.  With ``promote_trials`` every trial is evaluated as :meth:`accept_x` would leave it (its
        gauge refresh and converged gradient included, undone if another point is asked for next), so the values
        an optimizer compares are those it continues from.
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
                   boundary_max_nfev=boundary_max_nfev, boundary_target_cost=boundary_target_cost,
                   boundary_max_residual=boundary_max_residual, boundary_start=boundary_start,
                   gauge_refresh=gauge_refresh, gauge_max_share=gauge_max_share, gauge_release=gauge_release,
                   gradient_rounds=gradient_rounds, gradient_rtol=gradient_rtol,
                   gradient_fit_xtol=gradient_fit_xtol, gradient_certify=gradient_certify,
                   promote_trials=promote_trials)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_quantities=(), boundary_ftol, boundary_max_nfev,
                 boundary_target_cost=4.0, boundary_max_residual=None, boundary_start=None, gauge_refresh=0,
                 gauge_max_share=None, gauge_release=False, gradient_rounds=0, gradient_rtol=1e-3,
                 gradient_fit_xtol=None, gradient_certify=None, promote_trials=False):
        im = model._im
        self.gauge_refresh, self.gauge_max_share = int(gauge_refresh), gauge_max_share
        self.gauge_release = bool(gauge_release)
        self.gradient_rounds, self.gradient_rtol = int(gradient_rounds), float(gradient_rtol)
        self.gradient_fit_xtol = None if gradient_fit_xtol is None else float(gradient_fit_xtol)
        self.gradient_certify = None if gradient_certify is None else float(gradient_certify)
        if self.gradient_certify is not None and (self.gradient_fit_xtol is None or not self.gradient_rounds):
            raise ValueError("gradient_certify needs gradient_fit_xtol and gradient_rounds")
        self.promote_trials, self._provisional = bool(promote_trials), None  # (key, the state before it)
        self.model = model
        self.field = lambda x: model.gauged(field_from_parameters(x))
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_target_cost = None if boundary_target_cost is None else float(boundary_target_cost)
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
        self._field_vjp = jax.jit(lambda state, params, x, lam, gauge: jax.vjp(
            lambda z: model.rows(state, params, model.gauged(field_from_parameters(z), gauge)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx, gauge: jax.jvp(
            lambda z: model.rows(state, params, model.gauged(field_from_parameters(z), gauge)), (x,), (dx,))[1])
        self._cache, self._rejections, self._anchor = {}, {}, {}
        self._curvature = _CurvatureMemory(CURVATURE_MEMORY)
        self._multipliers = None  # the last Newton gradient's mu (a column per row): the next solve's first directions
        self._start, self._start_state = None, None
        if boundary_start is not None:
            from .input import VmecInput

            deck = boundary_start
            if not isinstance(boundary_start, VmecInput):  # a WOUT: its state seeds the first solve too
                from .restart import state_from_wout
                from .wout import read_wout

                wout = read_wout(boundary_start) if isinstance(boundary_start, (str, Path)) else boundary_start
                deck, self._start_state = VmecInput.from_wout(wout), state_from_wout(wout, inp=model.fixed)
            self._start = model.coordinates(deck)
        self._start_key = x0.tobytes()
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0, rejection_reasons={})
        if self._solve(x0) is None:
            raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
        if model.labelling == "flat_arc" and model._gauge is None:  # the seed fit's flat directions
            self._set_gauge(x0)
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _count(self, name, amount=1):
        self.stats[name] = self.stats.get(name, 0) + amount

    def _release_gauge(self, x):
        """Replace ``x``'s fit (cached) by the no-label fit from it, so that ``_set_gauge`` sets the gauge there.  If
        that fit fails the old gauge and fit stand."""
        model, key, xj = self.model, np.asarray(x, dtype=float).tobytes(), jnp.asarray(x)
        sol, gauge, field = self._cache[key], model._gauge, self.field(xj)
        model._gauge, model._last_linearize, model._radius = (jnp.zeros_like(gauge[0]), gauge[1]), None, None
        try:
            fit = model.solve_boundary(self._params_at(xj), field, x0=sol["x"], ftol=self.boundary_ftol,
                                       max_nfev=self.boundary_max_nfev)
            fitted = self._fits(fit, field)
            state, mask, params, _ = fit["aux"]
            values = np.asarray(self._rows(state, params, xj)) if fitted else None
        except (VmecError, RuntimeError):
            fitted = False
        if not fitted:
            model._gauge = gauge
            self._count("gauge_release_failures")
            return
        self.stats["boundary_evaluations"] += fit["nfev"]
        self._count("gauge_releases")
        self._cache[key] = dict(fit, state=state, mask=mask, params=params, values=values)

    def _set_gauge(self, x):
        """Set the model's gauge at ``x``'s fit (cached) and re-solve ``x`` from that fit under it."""
        model, key, xj = self.model, np.asarray(x, dtype=float).tobytes(), jnp.asarray(x)
        sol = self._cache[key]
        # with the plasma columns: the gradient at x reuses this linearization, its gauge rows swapped
        extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
        J = model.linearize(sol["x"], sol["aux"], self.field(xj), extra=extra)[0]
        self.stats["gauge_directions"] = model.set_flat_arc_gauge(sol["x"], J)
        self._count("gauge_sets")
        # x's fit as it is (flat arc's rows vanish there), under the new rows
        sol.pop("gradients", None)
        sol.pop("J", None)
        self._cache, self._rejections, self._anchor = {key: sol}, {}, {}
        self._start, self._start_state, self._start_key = np.asarray(sol["x"]), sol["state"], key

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
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx), self.model._gauge))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        fit = None
        if self._start is not None and key == self._start_key:  # a restart: the saved boundary as it is
            # the model's own cold solve of this boundary must not answer for the saved state (a converged
            # fixed-boundary state keeps the m=1 angle its path left it with, so the two differ at finite resolution)
            for memo in (self.model._im._LAST_SOLVE, self.model._im._LAST_REFINED):
                memo.pop(self.model.cfg, None)
            evaluated = self.model.evaluate(self._start, params, field, seed=self._start_state, tight=True)
            if evaluated is not None:
                fit = dict(x=self._start, rows=evaluated[0], aux=evaluated[1], nfev=1)
                if not self._fits(fit, field):
                    self.stats["boundary_evaluations"] += 1
                    fit = None
        for start in [] if fit else starts:
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, ftol=self.boundary_ftol,
                                                max_nfev=self.boundary_max_nfev,
                                                target_cost=self.boundary_target_cost * anchor["cost"]
                                                if anchor and self.boundary_target_cost is not None else None)
            except (VmecError, RuntimeError):  # no equilibrium on the way: try the next start
                continue
            if not self._fits(fit, field):
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
        if not np.all(np.isfinite(values)):  # a row whose own solve did not converge reports NaN
            return self._reject(key, "a row is not finite at this design point")
        self._cache[key] = dict(fit, state=state, mask=mask, params=params_b, values=values)
        return self._cache[key]

    def _fits(self, fit, field):
        """Every interface residual of ``fit`` within ``boundary_max_residual`` (when set)."""
        if self.boundary_max_residual is None:
            return True
        state, _, params_b, _ = fit["aux"]
        residual = self.model.boundary_residual(state, params_b, field)
        return max(residual.normal, residual.pressure, residual.sheet_current) <= self.boundary_max_residual

    def _reject(self, key, reason):
        self.stats["failed"] += 1
        kind = re.sub(r"[-+]?\d[\d.]*(e[-+]?\d+)?", "#", reason)[:80]  # the reason without its numbers
        self.stats["rejection_reasons"][kind] = self.stats["rejection_reasons"].get(kind, 0) + 1
        self._cache[key], self._rejections[key] = None, reason
        return None

    def _record(self, x):
        x = np.asarray(self._x(x), dtype=float)
        if self._provisional is not None and x.tobytes() != self._provisional[0]:
            self._restore()  # the optimizer did not take the promoted trial
        sol = self._solve(x)
        accepted = getattr(self, "accepted", None)
        if sol is not None and not (accepted is not None and np.array_equal(x, accepted.parameters)):
            if self.promote_trials and accepted is not None:
                sol = self._promote_trial(x)
            elif self.gradient_certify is not None:  # the accepted point stands
                sol = self._certify(x, sol)
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected(self._rejections[x.tobytes()])
        return sol

    _SAVED = ("_cache", "_rejections", "_anchor", "_start", "_start_state", "_start_key", "_multipliers")
    _SAVED_MODEL = ("_gauge", "_last_linearize", "_linearization", "_radius")

    def _promote_trial(self, x):
        """The trial ``x`` as :meth:`accept_x` leaves it, with its gradient (``None``: rejected); the state before it
        is kept until the trial is accepted or another point is asked for."""
        key = x.tobytes()
        if self._provisional is not None:
            return self._cache[key]
        curvature = self._curvature
        self._provisional = (key, [getattr(self, name) for name in self._SAVED],
                             [getattr(self.model, name) for name in self._SAVED_MODEL], (curvature.D, curvature.ED))
        try:
            self._promote(x)
            sol = self._certify(x, self._cache[key])
            reason = None if sol is not None else self._rejections[key]
        except TrialRejected as error:  # its fit does not tighten
            sol, reason = None, str(error)
        if sol is None:
            self._restore()
            self._cache[key], self._rejections[key] = None, reason
        return sol

    def _restore(self):
        """Undo :meth:`_promote_trial`; its trial is solved again if asked for."""
        key, saved, model, (self._curvature.D, self._curvature.ED) = self._provisional
        for name, value in zip(self._SAVED, saved):
            setattr(self, name, value)
        for name, value in zip(self._SAVED_MODEL, model):
            setattr(self.model, name, value)
        self._cache.pop(key, None)
        self._provisional = None

    def _certify(self, x, sol):
        """``sol`` with its gradient, or ``None`` (rejected) where its fit step or a Newton row's relative residual ends
        above ``gradient_certify`` (when set) times its tolerance, or the Newton terms fall back to Gauss-Newton."""
        if "gradients" in sol:
            return sol
        anchor, fallbacks = self._anchor, self.stats.get("newton_fallbacks", 0)
        for name in ("fit_step", "newton_residual"):
            self.stats.pop(name, None)
        try:
            self._linearize(x, sol)
        except TrialRejected as error:
            reason = str(error)
        else:
            fit, newton = self.stats.get("fit_step", np.inf), self.stats.get("newton_residual", np.inf)
            factor = self.gradient_certify
            if factor is None or (fit <= factor * self.gradient_fit_xtol and newton <= factor * self.gradient_rtol
                                  and self.stats.get("newton_fallbacks", 0) == fallbacks):
                return sol
            reason = f"uncertified gradient: fit step {fit:.1e}, Newton residual {newton:.1e}"
        self._anchor = anchor  # the next trials start from the last certified linearization
        return self._reject(x.tobytes(), reason)

    def _fit_jacobian(self, x, sol):
        """``sol``'s Jacobian (cached until its gradients are) once it is re-solved tight and its fit polished or
        converged.

        The record's values become those of the tight (or polished) state, so a value asked for after its gradient is
        that state's."""
        if "J" in sol:
            if "fit_step" in sol:  # the step its fit converged to, for a certification that follows
                self.stats["fit_step"] = sol["fit_step"]
            return sol["J"]
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
        try:  # the converged fit's Newton steps replace the polish
            polished = None if self.gradient_fit_xtol is not None else model.solve_boundary(
                self._params_at(xj), field, x0=sol["x"], jacobian=J[:, :nb], ftol=GRADIENT_POLISH_FTOL,
                max_nfev=GRADIENT_POLISH_NFEV)
        except (VmecError, RuntimeError):  # no polish: the tightened fit's own Jacobian stands
            polished = None
        if polished is not None:
            self.stats["boundary_evaluations"] += polished["nfev"]
            if polished["rows"] @ polished["rows"] < GRADIENT_POLISH_KEEP * sol["rows"] @ sol["rows"]:
                sol.update(x=polished["x"], aux=polished["aux"])
                J = tight()
        J_fit = None if self.gradient_fit_xtol is None else self._converge_fit(sol, J[:, :nb], xj, field)
        if J_fit is not None:
            J = np.hstack([J_fit, J[:, nb:]])
            if self.gradient_rounds:  # the Newton terms' products are exact: the Jacobian only preconditions them
                state, mask, params, _ = sol["aux"]
                sol.update(state=state, mask=mask, params=params, rows=np.asarray(model._rows(state, params, field)),
                           values=np.asarray(self._rows(state, params, xj)))
            else:
                J = tight()
        sol["J"] = J
        return J

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), at its tight, converged fit (``_fit_jacobian``)."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        field, nb = self.field(xj), model.x0.size
        J = self._fit_jacobian(x, sol)
        J_b, J_p = J[:, :nb], J[:, nb:]
        state, params = sol["state"], sol["params"]
        params_bar = self._pullback(state, sol["mask"], params, xj)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        S = model.gauge_curvature(sol["x"], params)
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        newton = self._newton_terms(sol, J_b, J_p, S, G_b, xj, field) if self.gradient_rounds else None
        if newton is not None:
            gradients -= newton
        else:
            if S is None:
                Q, R = np.linalg.qr(J_b)
                lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
            else:  # nonlinear gauge rows: J^T J + sum g Hess g (their curvature; not the interface rows')
                lam = J_b @ np.linalg.solve(J_b.T @ J_b + S, G_b.T)
            gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column), model._gauge))
                                   for column in lam.T])
            gradients[:, self._plasma] -= lam.T @ J_p
        self._anchor = dict(x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=state,
                            params=params, cost=0.5 * sol["rows"] @ sol["rows"])
        sol["gradients"] = gradients
        sol.pop("J", None)
        return gradients

    def _cost_gradient(self, z, params0, field, xj, *, design=True, near=None):
        """The fit cost's gradient in ``b`` and (``design``) in ``x`` at boundary ``z`` of the design ``xj``, and
        the rows and state there; ``None`` when ``z`` has no tight solution.  ``near`` ``(b, got)``: a solved
        neighbour (an earlier return at ``b``) that seeds the solve."""
        model = self.model
        near = None if near is None else (near[0], *near[1][3][:2])
        got = model.evaluate(z, params0, field, tight=True, near=near)
        if got is None:
            return None
        r, aux = got
        state, mask, params, _ = aux
        bar = jax.tree.map(lambda a: a[None], model.rows_pullback(state, mask, params, field, r))
        g_b, g_p = self._coordinates(params, jnp.asarray(z), xj, bar)
        g_x = None
        if design:
            g_x = np.asarray(self._field_vjp(state, params, xj, jnp.asarray(r), model._gauge)) + np.asarray(g_p[0])
        return np.asarray(g_b[0]), g_x, np.asarray(r), aux

    def _fit_metric(self, J_b, S):
        """The column norms of ``J_b`` and the column-scaled Gauss-Newton matrix ``P = J_b^T J_b + S``."""
        norms = np.linalg.norm(J_b, axis=0)
        norms[norms == 0.0] = 1.0
        P = (J_b / norms).T @ (J_b / norms)
        return norms, P if S is None else P + S / norms[:, None] / norms[None, :]

    def _model_factor(self, P, norms):
        """The Cholesky factor of ``P`` plus the curvature memory's model; of ``P`` alone (the memory dropped) where
        that is not positive definite."""
        W = self._curvature.model(norms)
        if W is not None:
            try:
                return scipy.linalg.cho_factor(P + W)
            except np.linalg.LinAlgError:
                self._curvature.clear()
                self._count("curvature_resets")
        return scipy.linalg.cho_factor(P)

    def _products(self, cost_gradient, b, base, block, norms, P):
        """``H v`` (column-scaled) and ``dF/dx^T v`` (``None`` without the design part) of the columns ``v`` of
        ``block``: differences of the fit cost's gradient at ``b`` (``base`` there), added to the curvature memory.
        ``None`` when a side has no tight solution."""
        eps, HB, FB = NEWTON_GRADIENT_EPS, [], []
        for v in block.T:
            plus = cost_gradient(b + eps * v / norms, near=(b, base))
            if plus is None:
                return None
            HB.append((plus[0] - base[0]) / eps / norms)
            FB.append(None if plus[1] is None else (plus[1] - base[1]) / eps)
        HB = np.stack(HB, axis=1)
        self._curvature.add(block / norms[:, None], (HB - P @ block) * norms[:, None])
        return HB, None if FB[0] is None else np.stack(FB, axis=1)

    def _converge_fit(self, sol, J_b, xj, field):
        """Newton steps on the fit ``sol`` (in place) until the Gauss-Newton step ``P^-1 F`` (column-scaled,
        ``P = J_b^T J_b + S``, ``F`` the fit cost's gradient by one reverse pass) is below ``gradient_fit_xtol``.  A
        step solves with ``P`` plus the curvature memory's model of the residual curvature ``H - P`` and adds its
        secant pair to the memory; after one that cuts the Gauss-Newton step by less than ``GRADIENT_FIT_CUT`` the
        next is GMRES on difference products ``H v``, preconditioned by that model.  Each step is backtracked until it
        lowers the cost without enlarging the Gauss-Newton step (or, at the solves' noise, halves that step), the
        Gauss-Newton step where it does not.  A full step that lowers the cost but enlarges the Gauss-Newton step
        re-linearizes ``J_b`` where the step lands (at most ``GRADIENT_FIT_REFRESHES`` times) and drops the memory.
        The latest ``J_b`` when ``sol`` moved, else ``None``."""
        from scipy.sparse.linalg import LinearOperator, gmres

        b = np.asarray(sol["x"], dtype=float)
        norms, P = self._fit_metric(J_b, self.model.gauge_curvature(b, sol["params"]))
        factor = scipy.linalg.cho_factor(P)
        params0 = self._params_at(xj)
        cost_gradient = lambda z, near=None: self._cost_gradient(z, params0, field, xj, design=False,  # noqa: E731
                                                                 near=near)
        got = cost_gradient(b)
        if got is None:
            return None
        moved, size, steps, products, exact, n, refreshes = False, np.inf, 0, [0], False, b.size, 0

        def hvp(v):
            out = self._products(cost_gradient, b, base, np.reshape(v, (-1, 1)), norms, P)
            if out is None:
                raise RuntimeError("no tight solution beside the fit")
            products[0] += 1
            return out[0][:, 0]

        while steps < GRADIENT_FIT_STEPS:
            g_b, _, r, _ = base = got
            F = g_b / norms
            gn = -scipy.linalg.cho_solve(factor, F)
            last, size = size, float(np.linalg.norm(gn))
            if size < self.gradient_fit_xtol:
                break
            exact = moved and not exact and size > GRADIENT_FIT_CUT * last
            curvature = self._model_factor(P, norms)
            du = -scipy.linalg.cho_solve(curvature, F)
            if exact:
                precondition = LinearOperator((n, n), matvec=lambda v: scipy.linalg.cho_solve(curvature, np.ravel(v)))
                try:
                    du, _ = gmres(LinearOperator((n, n), matvec=hvp), -F, x0=du, rtol=GRADIENT_FIT_RTOL,
                                  M=precondition,
                                  restart=GRADIENT_FIT_PRODUCTS, maxiter=1)
                except RuntimeError:
                    pass
            directions = [(gn, GRADIENT_FIT_GN_HALVINGS)]
            if not np.array_equal(du, gn):
                directions.insert(0, (du, GRADIENT_FIT_HALVINGS))
            s, trial, stale, tries = self._backtrack(cost_gradient, b, base, directions, factor, norms, size)
            steps += tries
            if s is None:
                break
            if np.linalg.norm(s) >= NEWTON_GRADIENT_EPS:  # the step's secant pair: (H - P) s ~ F(b + s) - F - P s
                self._curvature.add((s / norms)[:, None], ((trial[0] / norms - F - P @ s) * norms)[:, None])
            b, got, moved = b + s / norms, trial, True
            if stale and refreshes < GRADIENT_FIT_REFRESHES:
                try:
                    J_b = np.asarray(self.model.linearize(b, got[3], field)[0])[:, :n]
                except RuntimeError:
                    continue
                refreshes += 1
                norms, P = self._fit_metric(J_b, self.model.gauge_curvature(b, got[3][2]))
                factor = scipy.linalg.cho_factor(P)
                size = float(np.linalg.norm(scipy.linalg.cho_solve(factor, got[0] / norms)))
                exact = True  # the next step is the model's (now plain Gauss-Newton), not GMRES
                self._curvature.clear()
        self._count("fit_steps", steps)
        self._count("fit_products", products[0])
        self._count("fit_linearizations", refreshes)
        self.stats["fit_step"] = sol["fit_step"] = size
        if moved:
            sol.update(x=b, aux=got[3])
        return J_b if moved else None

    def _backtrack(self, cost_gradient, b, base, directions, factor, norms, size):
        """The first acceptable step of ``directions`` (``(step, halvings)`` pairs in order) from the fit ``b``: one
        that lowers the cost without enlarging the Gauss-Newton step of size ``size``, or (at the solves' noise) halves that
        step.  Returns ``(step, trial, stale, tries)``, ``step`` ``None`` when none is; ``stale``: a full step lowered
        the cost but enlarged the Gauss-Newton step, so the Jacobian behind ``factor`` no longer describes the fit."""
        cost, stale, tries = base[2] @ base[2], False, 0
        for direction, halvings in directions:
            for t in 0.5 ** np.arange(halvings + 1):
                trial = cost_gradient(b + t * direction / norms, near=(b, base))
                tries += 1
                if trial is None:
                    continue
                new = float(np.linalg.norm(scipy.linalg.cho_solve(factor, trial[0] / norms)))
                lower = trial[2] @ trial[2] <= cost * (1 + 1e-12)
                if new < 0.5 * size or (new <= size and lower):
                    return t * direction, trial, stale, tries
                stale = stale or (t == 1.0 and lower)
        return None, None, stale, tries

    def _newton_terms(self, sol, J_b, J_p, S, G_b, xj, field):
        """``mu^T dF/dx`` of every row (a row each) at the fit ``sol``, ``F = J_b^T r`` and ``mu = H^-1 dh/db``:
        block Krylov rounds (least residual) preconditioned by ``P = J_b^T J_b + S`` (column-scaled) plus the
        curvature memory's model and started from the last solve's ``mu``, until every row's relative residual is below
        ``gradient_rtol``, only the rows above it adding directions; each new direction's ``H v`` and ``dF/dx^T v`` by differences of the fit cost's
        gradient.  The soft directions (:data:`SOFT_DIRECTION_TOL`) are left out of that space and solved by
        Gauss-Newton.  ``None`` (the Gauss-Newton gradient stands) when a perturbed fit has no tight solution."""
        b, nb = np.asarray(sol["x"], dtype=float), J_b.shape[1]
        norms, P = self._fit_metric(J_b, S)
        g = (G_b / norms).T
        eigenvalues, eigenvectors = scipy.linalg.eigh(P)
        soft = eigenvalues < SOFT_DIRECTION_TOL * eigenvalues[-1]
        U = eigenvectors[:, soft]
        self.stats["soft_directions"] = int(np.sum(soft))
        size = np.maximum(np.linalg.norm(g, axis=0), 1e-300)
        lam = (J_b / norms) @ U @ (U.T @ g / eigenvalues[soft][:, None])  # residual-space multipliers, a column per row
        g = g - U @ (U.T @ g)
        params0 = self._params_at(xj)
        cost_gradient = lambda z, near=None: self._cost_gradient(z, params0, field, xj, near=near)  # noqa: E731
        base = cost_gradient(b)
        curvature = self._model_factor(P, norms)
        V, HV, FX = np.zeros((nb, 0)), np.zeros((nb, 0)), np.zeros((xj.size, 0))
        block = scipy.linalg.cho_solve(curvature, g) if self._multipliers is None else self._multipliers * norms[:, None]
        for _ in range(self.gradient_rounds):
            block = block - U @ (U.T @ block)
            block = block - V @ (V.T @ block)
            u, sv, _ = np.linalg.svd(block, full_matrices=False)
            block = u[:, sv > 1e-6 * sv[0]]
            got = None if base is None else self._products(cost_gradient, b, base, block, norms, P)
            if got is None:
                self._count("newton_fallbacks")
                return None
            self._count("newton_products", block.shape[1])
            V, HV, FX = np.hstack([V, block]), np.hstack([HV, got[0] - U @ (U.T @ got[0])]), np.hstack([FX, got[1]])
            # least residual, never worse round to round; directions below the products' accuracy left out
            coefficients = np.linalg.lstsq(HV, g, rcond=1e-6)[0]
            R = HV @ coefficients - g
            residuals = np.linalg.norm(R, axis=0) / size
            if residuals.max() <= self.gradient_rtol:
                break
            open_ = residuals > self.gradient_rtol  # rows still above it add the next directions
            block = scipy.linalg.cho_solve(curvature, R[:, open_])
        self.stats["newton_residual"] = float(residuals.max())
        self._multipliers = V @ coefficients / norms[:, None]
        terms = (FX @ coefficients).T
        if soft.any():
            state, params = sol["state"], sol["params"]
            terms = terms + np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column), self.model._gauge))
                                      for column in lam.T])
            terms[:, self._plasma] += lam.T @ J_p
        return terms

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
        x = np.asarray(self._x(x), dtype=float)
        key = x.tobytes()
        if self._provisional is None or self._provisional[0] != key:
            if self._provisional is not None:
                self._restore()
            if self._cache.get(key) is None:
                raise ValueError("evaluate the candidate before accepting it")
            self._promote(x)
        self._provisional = None
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}

    def _promote(self, x):
        """Refresh the gauge at ``x``'s fit (cached) where the next accepted step is due one."""
        sol = self._cache[x.tobytes()]
        reset = self.model._gauge is not None and bool(
            self.gauge_refresh and (self.accepted_step + 1) % self.gauge_refresh == 0)
        if self.model._gauge is not None and not reset and self.gauge_max_share is not None:
            if "gradients" not in sol:  # the share is the converged fit's; the gradient follows the gauge decision
                self._fit_jacobian(x, sol)
            reset = self.gauge_share(x) > self.gauge_max_share
        if reset:
            if self.gauge_release:
                self._release_gauge(x)
            self._set_gauge(x)

    def gauge_share(self, x):
        """The gauge rows' share of ``x``'s boundary-fit cost (0 without a gauge, and where it was set)."""
        model = self.model
        if model._gauge is None:
            return 0.0
        sol = self._record(x)
        r = np.asarray(model._rows(sol["state"], sol["params"], self.field(jnp.asarray(self._x(x)))))
        g = r[model._n_interface:model._n_interface + model._gauge[0].shape[0]]
        return float(g @ g / (r @ r))

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
        moved = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))
                          / max(float(np.max(np.abs(np.asarray(b)))), 1e-300))
                    for a, b in zip(jax.tree.leaves(eq.state), jax.tree.leaves(sol["state"])) if np.size(b))
        if not moved <= rtol:
            import warnings

            warnings.warn(f"the reporting re-solve moved the optimizer's state by {moved:.2e} (> {rtol:g})",
                          RuntimeWarning, stacklevel=2)
        return eq

    def close(self):
        """Drop the cached trials but the accepted one."""
        if self._provisional is not None:
            self._restore()
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


class _CurvatureMemory:
    """The fit's residual curvature ``E = H - J_b^T J_b - S`` (``H`` the fit cost's Hessian) along the latest measured
    boundary directions, carried from fit to fit: a symmetric model of ``E`` exact on their span."""

    def __init__(self, size):
        self.size, self.D, self.ED = int(size), None, None

    def clear(self):
        self.D, self.ED = None, None

    def add(self, D, ED):
        """Directions ``D`` (columns, model coordinates) and ``E D``."""
        self.D = D if self.D is None else np.hstack([self.D, D])[:, -self.size:]
        self.ED = ED if self.ED is None else np.hstack([self.ED, ED])[:, -self.size:]

    def model(self, norms):
        """``Y Q^T + Q Y^T - Q (Q^T Y) Q^T`` in the coordinates scaled by ``norms``, ``Q`` an orthonormal basis of the
        directions and ``Y = E Q``; ``None`` while empty."""
        if self.D is None:
            return None
        u, sv, vt = np.linalg.svd(self.D * norms[:, None], full_matrices=False)
        keep = sv > 1e-3 * sv[0]
        Q, Y = u[:, keep], (self.ED / norms[:, None]) @ vt[keep].T / sv[keep]
        A = Q.T @ Y
        return Y @ Q.T + Q @ Y.T - Q @ (0.5 * (A + A.T)) @ Q.T


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray

