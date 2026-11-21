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

import jax
import jax.numpy as jnp

from .errors import TrialRejected, VmecError
from .freeboundary_three_terms import BoundaryResidual, ThreeTermsConfig, ThreeTermsModel
from .problem import FunctionProblem

#: The gradient's polish of a trial: its own Jacobian, to this relative cost change in at most this many evaluations,
#: kept when it cuts the cost below this fraction.
GRADIENT_POLISH_FTOL, GRADIENT_POLISH_NFEV, GRADIENT_POLISH_KEEP = 1e-4, 20, 0.9
#: A trial is its state solved to the deck's tolerance: where that solve moves an interface residual of the trial
#: tolerance's state by more than the largest of them and by more than the floor (in units of ``|B|``), the
#: trial-tolerance state was no equilibrium and the trial is rejected.  The floor is what the two tolerances differ
#: by on one equilibrium: the deck-tolerance solve of a fit of 1e-5 has a normal field of 8e-5 (QI, 16 x 16 modes),
#: which the accepted fit's polish removes; a state of another branch moves by eight times the floor.
TRIAL_DEPARTURE, TRIAL_DEPARTURE_FLOOR = 1.0, 1e-4


def _nonlinear_constraint(values, jacobian, lower, upper, scales):
    """A SciPy ``NonlinearConstraint`` of ``lower <= values(x) <= upper``, rows divided by ``scales``."""
    from scipy.optimize import NonlinearConstraint

    lo, hi, scale = np.broadcast_arrays(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float),
                                        np.asarray(scales, dtype=float))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("ordered bounds and finite positive constraint scales required")
    return NonlinearConstraint(lambda x: np.asarray(values(x)) / scale, lo / scale, hi / scale,
                               jac=lambda x: np.asarray(jacobian(x)) / scale[..., None])


#: A gauge release continues from a fresh Jacobian at most this many times, while each round still lowers the fit's
#: cost by more than GAUGE_RELEASE_GAIN: the reused Jacobian is only good near the accepted fit, and the flat
#: directions it releases curve away from it.
GAUGE_RELEASE_ROUNDS = 6
GAUGE_RELEASE_GAIN = 0.02


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
    first-order prediction of the latest linearized solution, and is its state solved to the deck's tolerance: a trial without a
    solution, or whose trial-tolerance state that solve leaves, raises :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design gradient of the
    implicit function theorem of the boundary least squares,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  This Gauss-Newton form
    leaves out the rows' curvature times the fit's residual.  Nothing recompiles when ``x`` changes.  ``model`` is the
    :class:`ThreeTermsModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures and their reasons, boundary evaluations and linearizations) report
    progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_quantities=(), config=None, boundary_ftol=1e-3,
                  boundary_max_nfev=40, boundary_max_residual=None, boundary_start=None, gauge_refresh=0,
                  gauge_max_share=None, gauge_release=False, gauge_release_rounds=GAUGE_RELEASE_ROUNDS):
        """Solve the seed design's free boundary (that of ``x0`` on the deck ``inp``).

        ``boundary_start`` (a run's final WOUT, or a deck: its fitted boundary and, with ``bootstrap``, current) is
        the seed design's boundary as it is, re-fitted only if it misses ``boundary_max_residual``, and the start of
        the fits before the first gradient; a WOUT's equilibrium also seeds the seed design's solve (the interior
        state is not unique to the force tolerance).  A restart keeps its first run's deck ``inp`` (the anchored
        labelling's reference and the quadrature plan's boundary) and resumes at its final WOUT.

        ``config`` (a :class:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig`) sets the labelling, grid,
        quadrature, trial tolerance and bootstrap current of the model; its fit tolerances, ``start`` and
        ``max_residual`` are the solver's, not the problem's (``boundary_max_residual`` caps the trials).  A trial's
        boundary fit stops at a relative cost change of ``boundary_ftol`` or after ``boundary_max_nfev`` evaluations; each gradient first re-solves its trial to
        the deck's tolerance and polishes it with its own Jacobian.  With ``boundary_max_residual`` a fit that
        ends with any :class:`BoundaryResidual` term above it is no solution: the trial tries its next start, or
        is rejected.  With a gauge (labelling ``"flat_arc"``), set at the seed's fit, ``gauge_refresh=K`` sets it again
        at the accepted fit every ``K`` accepted steps (:meth:`accept_x`), and ``gauge_max_share`` whenever the gauge
        rows' share of the accepted fit's cost exceeds it, so it follows the design: its rows vanish there, so no value
        at the accepted point changes.  With ``gauge_release`` each of those refreshes first re-fits the accepted
        boundary without the gauge rows (the no-label fit), so the labelling does not accumulate: the accepted point's
        values then become the no-label fit's; that fit continues from at most ``gauge_release_rounds`` fresh Jacobians
        while a round lowers its cost by more than the fraction :data:`GAUGE_RELEASE_GAIN`.
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
                   boundary_start=boundary_start, gauge_refresh=gauge_refresh,
                   gauge_max_share=gauge_max_share, gauge_release=gauge_release, gauge_release_rounds=gauge_release_rounds)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_quantities=(), boundary_ftol, boundary_max_nfev,
                 boundary_max_residual=None, boundary_start=None, gauge_refresh=0, gauge_max_share=None,
                 gauge_release=False, gauge_release_rounds=GAUGE_RELEASE_ROUNDS):
        im = model._im
        self.gauge_refresh, self.gauge_max_share = int(gauge_refresh), gauge_max_share
        self.gauge_release = bool(gauge_release)
        self.gauge_release_rounds = int(gauge_release_rounds)
        self.model = model
        self.field = lambda x: model.gauged(field_from_parameters(x))
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
        def pullback(state, mask, params, x, bands):
            """Every row's ``ImplicitParams`` gradient, each through its own function's reverse pass only."""
            sizes = [jax.eval_shape(f, state, params, x).size for f in parts]
            return model.pullback(parts, state, mask, params, tuple(jnp.eye(k) for k in sizes), x, bands=bands)

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
        self._start, self._start_state = None, None
        if boundary_start is not None:
            from .input import VmecInput

            deck = boundary_start
            if not isinstance(boundary_start, VmecInput):  # a WOUT: its state seeds the first solve too
                from .restart import state_from_wout
                from .wout import read_wout

                wout = read_wout(boundary_start) if isinstance(boundary_start, (str, Path)) else boundary_start
                deck, self._start_state = VmecInput.from_wout(wout), state_from_wout(wout, inp=model.fixed)
            fixed = model.fixed
            if (int(deck.mpol), int(deck.ntor)) != (int(fixed.mpol), int(fixed.ntor)):  # a start saved at another resolution
                deck = deck.change_resolution(mpol=int(fixed.mpol), ntor=int(fixed.ntor))
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

    def _released(self, x, sol):
        """The no-label fit from ``sol``, ``x``'s fit under the gauge: its record, or ``None`` where that fit fails
        or its state does not tighten.  It reuses the Jacobian ``sol`` has, else the one its fit used (the last
        accepted fit's): steps of one evaluation, then at most ``gauge_release_rounds`` fresh Jacobians.  The model
        is left as it was."""
        model, xj = self.model, jnp.asarray(x)
        gauge, saved = model._gauge, [getattr(model, name) for name in self._SAVED_MODEL]
        J = sol.get("J")
        if J is None:
            J = self._anchor["J_b"] if self._anchor else self._fit_jacobian(x, sol)
        # radius 0: the steps start at 1/20 of the Gauss-Newton step, which without the gauge rows is long along the
        # flat directions (from its full length the reused Jacobian's first steps all fail)
        model._gauge, model._last_linearize, model._radius = (jnp.zeros_like(gauge[0]), gauge[1]), None, 0.0
        field = self.field(xj)  # after the weights are zeroed: the field carries the gauge it is wrapped with
        if J is not None:  # its gauge rows zero like the weights
            J = np.array(J)[:, :model.x0.size]
            J[model._n_interface:model._n_interface + gauge[0].shape[0]] = 0.0
        try:
            fit = model.solve_boundary(self._params_at(xj), field, x0=sol["x"], jacobian=J, ftol=self.boundary_ftol,
                                       max_nfev=self.boundary_max_nfev)
            for _ in range(self.gauge_release_rounds if J is not None else 0):  # a fresh Jacobian where the reused one ended
                J = np.array(model.linearize(fit["x"], fit["aux"], field)[0])[:, :model.x0.size]
                more = model.solve_boundary(self._params_at(xj), field, x0=fit["x"], jacobian=J, ftol=self.boundary_ftol,
                                            max_nfev=self.boundary_max_nfev)
                gained = more["rows"] @ more["rows"] < (1.0 - GAUGE_RELEASE_GAIN) * (fit["rows"] @ fit["rows"])
                fit = dict(more, nfev=fit["nfev"] + more["nfev"], njev=fit["njev"] + 1 + more["njev"])
                if not gained:
                    break
            fitted = self._fits(fit, field)
            if fitted and not fit["aux"][3]:  # as a trial: the released fit's state at the deck's tolerance
                tight = self._tightened(fit, self._params_at(xj), field)
                fitted = not isinstance(tight, str)
                if fitted:
                    fit = dict(fit, rows=tight[0], aux=tight[1])
            state, mask, params, _ = fit["aux"]
            values = np.asarray(self._rows(state, params, xj)) if fitted else None
        except (VmecError, RuntimeError):
            fitted = False
        finally:
            for name, value in zip(self._SAVED_MODEL, saved):
                setattr(model, name, value)
        if not fitted:
            self._count("gauge_release_failures")
            return None
        self.stats["boundary_evaluations"] += fit["nfev"]
        self._count("gauge_releases")
        self._count("gauge_release_warm", int(J is not None))  # how the releases went: evaluations and Jacobians
        self._count("gauge_release_nfev", int(fit["nfev"]))
        self._count("gauge_release_njev", int(fit["njev"]))
        return dict(fit, state=state, mask=mask, params=params, values=values, released=True)

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
                                                target_cost=4 * anchor["cost"] if anchor else None)
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
        if self.gauge_release and self.model._gauge is not None and anchor:
            # a trial is the fit it has once accepted: the no-label fit from its fit under the gauge
            released = self._released(x, dict(fit))
            if released is not None:
                self._cache[key] = released
                return released
        if not fit["aux"][3]:  # the fit's state at the deck's tolerance: the state its values and gradient are of
            tight = self._tightened(fit, params, field)
            if isinstance(tight, str):
                return self._reject(key, tight)
            fit = dict(fit, rows=tight[0], aux=tight[1])
        state, mask, params_b, _ = fit["aux"]
        try:  # rows with solves of their own (a vacuum iota, say) can fail where the boundary fit did not
            values = np.asarray(self._rows(state, params_b, jnp.asarray(x)))
        except (VmecError, RuntimeError) as error:
            return self._reject(key, str(error))
        self._cache[key] = dict(fit, state=state, mask=mask, params=params_b, values=values)
        return self._cache[key]

    def _tightened(self, fit, params, field):
        """``fit``'s rows and state solved to the deck's tolerance from its trial-tolerance state, or the reason it
        is no trial: no such solution, or one whose interface residuals are not the trial state's
        (:data:`TRIAL_DEPARTURE`)."""
        model = self.model
        state, _, params_b, _ = fit["aux"]
        try:
            evaluated = model.evaluate(fit["x"], params, field, seed=state, tight=True)
        except (VmecError, RuntimeError):
            evaluated = None
        if evaluated is None:
            return "the trial state does not converge to the deck's tolerance"
        self.stats["boundary_evaluations"] += 1
        terms = lambda r: np.array([r.normal, r.pressure, r.sheet_current])  # noqa: E731
        before = terms(model.boundary_residual(state, params_b, field))
        after = terms(model.boundary_residual(evaluated[1][0], evaluated[1][2], field))
        moved = float(np.max(np.abs(after - before)))
        self.stats["tight_shift"] = moved / float(np.max(before))
        if moved > max(TRIAL_DEPARTURE * float(np.max(before)), TRIAL_DEPARTURE_FLOOR):
            return (f"the trial state is not its tight solve's: interface residual {np.max(before):.1e} -> "
                    f"{np.max(after):.1e}")
        return evaluated

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
        sol = self._solve(x)
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected(self._rejections[x.tobytes()])
        return sol

    _SAVED_MODEL = ("_gauge", "_last_linearize", "_linearization", "_radius")

    def _fit_jacobian(self, x, sol):
        """``sol``'s Jacobian (cached until its gradients are) once it is re-solved tight and its fit polished.

        The record's values become those of the tight (or polished) state, so a value asked for after its gradient is
        that state's."""
        if "J" in sol:
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
        try:
            polished = model.solve_boundary(
                self._params_at(xj), field, x0=sol["x"], jacobian=J[:, :nb], ftol=GRADIENT_POLISH_FTOL,
                max_nfev=GRADIENT_POLISH_NFEV)
        except (VmecError, RuntimeError):  # no polish: the tightened fit's own Jacobian stands
            polished = None
        if polished is not None:
            self.stats["boundary_evaluations"] += polished["nfev"]
            if polished["rows"] @ polished["rows"] < GRADIENT_POLISH_KEEP * sol["rows"] @ sol["rows"]:
                sol.update(x=polished["x"], aux=polished["aux"])
                J = tight()
        sol["J"] = J
        return J

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), at its tight, polished fit (``_fit_jacobian``)."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        nb = model.x0.size
        J = self._fit_jacobian(x, sol)
        J_b, J_p = J[:, :nb], J[:, nb:]
        state, params = sol["state"], sol["params"]
        params_bar = self._pullback(state, sol["mask"], params, xj, model.stored_bands(params, state, sol["mask"]))
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        S = model.gauge_curvature(sol["x"], params)
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
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
        """Promote an evaluated point after the optimizer accepts it; drop the other trials.  Returns whether the
        promotion re-fitted the point (a gauge release), so that its values moved."""
        x = np.asarray(self._x(x), dtype=float)
        key = x.tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        releases = self.stats.get("gauge_releases", 0)
        self._promote(x)
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}
        return self.stats.get("gauge_releases", 0) > releases

    def _promote(self, x):
        """Refresh the gauge at ``x``'s fit (cached) where the next accepted step is due one or the gauge rows'
        share of its cost is above ``gauge_max_share``."""
        sol = self._cache[x.tobytes()]
        reset = self.model._gauge is not None and bool(
            self.gauge_refresh and (self.accepted_step + 1) % self.gauge_refresh == 0)
        if self.model._gauge is not None and not reset and self.gauge_max_share is not None:
            if "gradients" not in sol:  # the share is the converged fit's; the gradient follows the gauge decision
                self._fit_jacobian(x, sol)
            reset = self.gauge_share(x) > self.gauge_max_share
            self._count("gauge_share_refreshes", int(reset))
        if reset:
            if self.gauge_release and not sol.get("released"):  # a trial is released when it is solved
                released = self._released(x, sol)
                if released is not None:
                    self._cache[x.tobytes()] = released
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
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray

