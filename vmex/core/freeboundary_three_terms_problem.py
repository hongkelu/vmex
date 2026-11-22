"""Single-stage optimization at the three-term free boundary (:class:`ThreeTermsFreeBoundaryProblem`).

The problem gives a loss and constraint rows of coils and plasma parameters, with their design gradients.  A trial is
the boundary fit of its design.  It is a solution when every interface residual is within ``boundary_max_residual``.

Terms:

- Label rows: rows that hold the boundary's poloidal labelling along the fit's nearly flat tangential directions.
  Only the labelling ``flat_arc`` has them.
- Released fit: the trial's fit without the label rows.
- Follow: every trial is its released fit, and the label rows are set again at each accepted step.
- Anchor: the last linearized point, normally the last accepted one.  Its Jacobian starts the next trials' fits.
- Polish: fresh-Jacobian rounds on a fit before its gradient, up to ``GRADIENT_POLISH_ROUNDS`` of them.

One optimization step with label rows:

1. Trial (``_solve``).  Predict the boundary to first order from the anchor.  Fit it under the label rows with the
   anchor's Jacobian.  Continue from there to the released fit (``_released``), which is the trial.  Where the
   released fit fails, the trial is the fit under the label rows.
2. Gradient (``_linearize``).  Polish the fit (``_fit_jacobian``, up to three rounds).  The trial's values become
   those of the polished fit.  Linearize it, differentiate every row by the implicit function theorem of the boundary
   fit, and make the point the anchor.
3. Accept (``accept_x``).  Release the trial if it is not released yet.  Set the label rows again at the accepted
   fit (``_set_label_rows``), where they vanish, and clear the anchor.  The next trial first linearizes the accepted
   point under the new label rows, which sets the anchor again.

Without label rows a trial is its fit from the anchor, and accepting it changes nothing but the cache.  Without an
anchor a fit starts from ``boundary_start``, or from the deck's boundary.
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

#: The gradient's polish of a trial: rounds of its own (fresh) Jacobian and steps to this relative cost change in at
#: most this many evaluations; a round is kept, and followed by another (three at most), when it cuts the cost below
#: this fraction.  A trial's fit reuses an older Jacobian and can stop in a shallow valley 10-20 % above its minimum
#: (QI at 12 x 12 modes: rounds of 9, 9, 3 and 2 %).
GRADIENT_POLISH_FTOL, GRADIENT_POLISH_NFEV, GRADIENT_POLISH_KEEP, GRADIENT_POLISH_ROUNDS = 1e-4, 20, 0.95, 3


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

    Build with :meth:`from_loss` and pass it to :func:`vmex.core.optimize.minimize` (SLSQP).

    A design ``x`` sets:

    - the external field, through ``field_from_parameters(x)``.  It can be coils, currents or any differentiable
      field.  Coil charts live with the caller, for example in ESSOS.
    - optionally plasma parameters such as PHIEDGE or the current profile, through
      ``plasma_from_parameters(params, x)``, an :class:`~vmex.core.implicit.ImplicitParams` map.

    The rows, in order:

    - row 0, ``loss(state, runtime, x)``;
    - the ``quantities(state, runtime)``;
    - the ``parameter_quantities(state, runtime, x)``;
    - the ``boundary_quantities(rbc, zbs, x)``.  These are functions of the trial's fitted boundary itself, given as
      INDATA arrays (the model's boundary coordinates), such as a coil-plasma clearance.

    Every trial is the K = 0 boundary of the field and plasma parameters of ``x`` (:class:`ThreeTermsModel`).  Its fit
    is warm-started from the first-order prediction of the latest linearized solution.  A trial without a solution
    raises :class:`~vmex.core.errors.TrialRejected`.

    A row ``h`` (loss or quantity) has the design gradient of the implicit function theorem of the boundary least
    squares,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db.

    Here ``r`` are the interface rows and ``J_b`` is their Jacobian in the model's coordinates ``b``: the boundary,
    and with ``bootstrap`` the bootstrap current.  ``dr/dx`` goes through the plasma parameters and the field.  Every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  This Gauss-Newton form leaves
    out the rows' curvature times the fit's residual.  Nothing recompiles when ``x`` changes.

    ``model`` is the :class:`ThreeTermsModel`.  :meth:`boundary_residual` and ``stats`` report progress.  The keys of
    ``stats``:

    - ``trials``: designs tried.  ``failed``: those without a solution.  ``rejection_reasons``: their
      reasons without the numbers, each with its count.
    - ``boundary_evaluations``: equilibrium evaluations of the boundary fits.
    - ``linearizations``: Jacobians taken for the gradients, their polish rounds included.
    - ``fit_ftol``, ``fit_stalled``, ``fit_radius``, ``fit_max_nfev``: how the solved trials' fits ended, before any
      release (the ``reason`` of the model's fit).  ``fit_as_saved``: a restart's saved boundary taken as it is.
    - ``label_directions``: directions the label rows hold since they were last set.  ``label_sets``: times they were
      set.  ``releases`` and ``release_failures``: released fits that solved and that failed.
    - ``polish_rounds``: polish rounds kept.  ``polish_gain``: the last polish round's relative cost reduction.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_quantities=(), config=None, boundary_ftol=1e-3,
                  boundary_max_nfev=40, boundary_max_residual=None, boundary_start=None):
        """Solve the seed design's free boundary (that of ``x0`` on the deck ``inp``).

        ``config``
            A :class:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig`.  It sets the labelling, grid, quadrature
            and bootstrap current of the model.  The problem ignores the config's ``ftol``, ``jacobian_ftol``,
            ``max_nfev``, ``max_residual`` and ``start``.  Those are the solver's.
        ``boundary_ftol``, ``boundary_max_nfev``
            A trial's boundary fit stops at a relative cost change of ``boundary_ftol`` or after ``boundary_max_nfev``
            evaluations.  Each gradient first polishes its trial's fit with its own Jacobian.
        ``boundary_max_residual``
            Caps the trials.  When set, a fit that ends with any :class:`BoundaryResidual` term above it is no
            solution: the trial tries its next start, or is rejected.
        ``boundary_start``
            A run's final WOUT, or a deck.  It gives a fitted boundary and, with ``bootstrap``, its current.  This is
            the seed design's boundary as it is, re-fitted only if it misses ``boundary_max_residual``.  It is also
            the start of the fits before the first gradient.  A WOUT's equilibrium also seeds the seed design's solve,
            because the interior state is not unique to the force tolerance.

        A restart keeps its first run's deck ``inp`` and resumes at its final WOUT.  The deck holds the anchored
        labelling's reference and the quadrature plan's boundary.

        With the labelling ``"flat_arc"`` the label rows are set at the seed's fit and the fit follows the design.
        Every trial is its released fit (no label rows), continued from its fit under the label rows.  So the labelling
        does not accumulate, and accepting a trial changes none of its values.  The label rows are set again at every
        accepted fit (:meth:`accept_x`), where they vanish.
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
        self.model = model
        self.field = lambda x: model.labelled(field_from_parameters(x))
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
        self._field_vjp = jax.jit(lambda state, params, x, lam, labels: jax.vjp(
            lambda z: model.rows(state, params, model.labelled(field_from_parameters(z), labels)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx, labels: jax.jvp(
            lambda z: model.rows(state, params, model.labelled(field_from_parameters(z), labels)), (x,), (dx,))[1])
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
            if (int(deck.mpol), int(deck.ntor)) != (int(fixed.mpol), int(fixed.ntor)):  # saved at another resolution
                deck = deck.change_resolution(mpol=int(fixed.mpol), ntor=int(fixed.ntor))
            self._start = model.coordinates(deck)
        self._start_key = x0.tobytes()
        # Every key the class writes (the class docstring describes them).
        self.stats = dict(trials=0, failed=0, rejection_reasons={}, boundary_evaluations=0, linearizations=0,
                          fit_ftol=0, fit_stalled=0, fit_radius=0, fit_max_nfev=0, fit_as_saved=0,
                          label_directions=0, label_sets=0, releases=0, release_failures=0,
                          polish_rounds=0, polish_gain=0.0)
        self.accepted = None  # no accepted point during the seed solve
        if self._solve(x0) is None:
            raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
        if model.labelling == "flat_arc" and model._labels is None:  # the seed fit's flat directions
            self._set_label_rows(x0)
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _count(self, name, amount=1):
        self.stats[name] += amount

    def _released(self, x, sol):
        """The released fit from ``sol``, ``x``'s fit under the label rows: its record, or ``None`` where it fails.

        Its steps reuse the Jacobian ``sol`` has, else the anchor's, else ``sol``'s own after its polish.  The model is
        left as it was."""
        model, xj = self.model, jnp.asarray(x)
        labels, saved = model._labels, [getattr(model, name) for name in self._SAVED_MODEL]
        J = sol.get("J")
        if J is None:
            J = self._anchor["J_b"] if self._anchor else self._fit_jacobian(x, sol)
        # radius 0: the steps start at 1/20 of the Gauss-Newton step, which without the label rows is long along the
        # flat directions (from its full length the reused Jacobian's first steps all fail)
        model._labels, model._last_linearize, model._radius = (jnp.zeros_like(labels[0]), labels[1]), None, 0.0
        field = self.field(xj)  # after the weights are zeroed: the field carries the label rows it is wrapped with
        if J is not None:  # its label rows zero like the weights
            J = np.array(J)[:, :model.x0.size]
            J[model._n_interface:model._n_interface + labels[0].shape[0]] = 0.0
        fit = None
        try:
            fit = model.solve_boundary(self._params_at(xj), field, x0=sol["x"], jacobian=J, ftol=self.boundary_ftol,
                                       max_nfev=self.boundary_max_nfev)
            self.stats["boundary_evaluations"] += fit["nfev"]
            fitted = self._fits(fit, field)
            state, mask, params = fit["aux"]
            values = np.asarray(self._rows(state, params, xj)) if fitted else None
        except (VmecError, RuntimeError):
            fitted = False
        finally:
            for name, value in zip(self._SAVED_MODEL, saved):
                setattr(model, name, value)
        if not fitted:
            self._count("release_failures")
            return None
        self._count("releases")
        return dict(fit, state=state, mask=mask, params=params, values=values, released=True)

    def _set_label_rows(self, x):
        """Set the model's label rows at ``x``'s fit (cached) and re-solve ``x`` from that fit under it."""
        model, key, xj = self.model, np.asarray(x, dtype=float).tobytes(), jnp.asarray(x)
        sol = self._cache[key]
        # with the plasma columns: the gradient at x reuses this linearization, its label rows swapped
        extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
        J = model.linearize(sol["x"], sol["aux"], self.field(xj), extra=extra)[0]
        self.stats["label_directions"] = model.set_label_rows(sol["x"], J)
        self._count("label_sets")
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
        if self.model._labels is not None and not self._anchor and self.accepted is not None:
            # follow: the accepted point's Jacobian starts every trial's released fit, whatever the caller asked first
            at = self.accepted.parameters
            self._linearize(at, self._cache[at.tobytes()])
        anchor, starts, J = self._anchor, [self._start], None
        if anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx), self.model._labels))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        fit = None
        if self._start is not None and key == self._start_key:  # a restart: the saved boundary as it is
            # the model's own cold solve of this boundary must not answer for the saved state (a converged
            # fixed-boundary state keeps the m=1 angle its path left it with, so the two differ at finite resolution)
            for memo in (self.model._im._LAST_SOLVE, self.model._im._LAST_REFINED):
                memo.pop(self.model.cfg, None)
            evaluated = self.model.evaluate(self._start, params, field, seed=self._start_state)
            if evaluated is not None:
                fit = dict(x=self._start, rows=evaluated[0], aux=evaluated[1], nfev=1)
                if not self._fits(fit, field):
                    self.stats["boundary_evaluations"] += 1
                    fit = None
        if fit is None:
            for start in starts:
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
        self._count(f"fit_{fit.get('reason', 'as_saved')}")  # how the fit ended; the saved boundary has no reason
        if self.model._labels is not None and anchor:
            # a trial is the fit it has once accepted: the released fit from its fit under the label rows
            released = self._released(x, dict(fit))
            if released is not None:
                self._cache[key] = released
                return released
        state, mask, params_b = fit["aux"]
        try:  # rows with solves of their own (a vacuum iota, say) can fail where the boundary fit did not
            values = np.asarray(self._rows(state, params_b, jnp.asarray(x)))
        except (VmecError, RuntimeError) as error:
            return self._reject(key, str(error))
        self._cache[key] = dict(fit, state=state, mask=mask, params=params_b, values=values)
        return self._cache[key]

    def _fits(self, fit, field):
        """Every interface residual of ``fit`` within ``boundary_max_residual`` (when set)."""
        if self.boundary_max_residual is None:
            return True
        state, _, params_b = fit["aux"]
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

    _SAVED_MODEL = ("_labels", "_last_linearize", "_linearization", "_radius")

    def _fit_jacobian(self, x, sol):
        """``sol``'s Jacobian (cached until its gradients are) once its fit is polished.

        The record's values become those of the polished state, so a value asked for after its gradient is that
        state's."""
        if "J" in sol:
            return sol["J"]
        model, xj = self.model, jnp.asarray(x)
        field, nb = self.field(xj), model.x0.size

        def linearized():
            extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
            try:
                J, _, _, aux = model.linearize(sol["x"], sol["aux"], field, extra=extra)
            except RuntimeError as error:  # no linearization at this state
                raise TrialRejected(str(error)) from error
            state, mask, params = aux
            sol.update(aux=aux, state=state, mask=mask, params=params,
                       rows=np.asarray(model._rows(state, params, field)),
                       values=np.asarray(self._rows(state, params, xj)))
            self.stats["linearizations"] += 1
            return J

        J = linearized()
        for _ in range(GRADIENT_POLISH_ROUNDS):
            try:
                polished = model.solve_boundary(
                    self._params_at(xj), field, x0=sol["x"], jacobian=J[:, :nb], ftol=GRADIENT_POLISH_FTOL,
                    max_nfev=GRADIENT_POLISH_NFEV)
            except (VmecError, RuntimeError):  # no polish: the fit's own Jacobian stands
                break
            self.stats["boundary_evaluations"] += polished["nfev"]
            gain = 1.0 - (polished["rows"] @ polished["rows"]) / (sol["rows"] @ sol["rows"])
            self.stats["polish_gain"] = float(gain)  # the last round's relative cost reduction
            if gain <= 1.0 - GRADIENT_POLISH_KEEP:
                break
            sol.update(x=polished["x"], aux=polished["aux"])
            J = linearized()
            self._count("polish_rounds")
        sol["J"] = J
        return J

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), at its polished fit (``_fit_jacobian``)."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        nb = model.x0.size
        J = self._fit_jacobian(x, sol)
        J_b, J_p = J[:, :nb], J[:, nb:]
        state, params = sol["state"], sol["params"]
        params_bar = self._pullback(state, sol["mask"], params, xj, model.stored_bands(params, state, sol["mask"]))
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        S = model.label_rows_curvature(sol["x"], params)
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        if S is None:
            Q, R = np.linalg.qr(J_b)
            lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
        else:  # nonlinear label rows: J^T J + sum g Hess g (their curvature; not the interface rows')
            lam = J_b @ np.linalg.solve(J_b.T @ J_b + S, G_b.T)
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column), model._labels))
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
        """Promote an evaluated point after the optimizer accepts it and drop the other trials.

        With label rows the point is released first if it is not released yet, and the label rows are set
        again at its fit.  Returns whether the point was released here, so that its values moved."""
        x = np.asarray(self._x(x), dtype=float)
        key = x.tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        released = self._promote(x)
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}
        return released

    def _promote(self, x):
        """Set the label rows again at ``x``'s fit (cached).  Returns whether ``x`` was released here."""
        sol, released = self._cache[x.tobytes()], None
        if self.model._labels is not None:
            if not sol.get("released"):  # a trial is released when it is solved; one whose release failed, here
                released = self._released(x, sol)
                if released is not None:
                    self._cache[x.tobytes()] = released
            self._set_label_rows(x)
        return released is not None

    def label_share(self, x):
        """The label rows' share of ``x``'s boundary-fit cost (0 without label rows, and where they were set)."""
        model = self.model
        if model._labels is None:
            return 0.0
        sol = self._record(x)
        r = np.asarray(model._rows(sol["state"], sol["params"], self.field(jnp.asarray(self._x(x)))))
        g = r[model._n_interface:model._n_interface + model._labels[0].shape[0]]
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

