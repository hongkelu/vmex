"""Single-stage optimization on the three-term free boundary: a loss and constraint rows of a design at its boundary.

A design ``x`` sets the external field (``field_from_parameters(x)``: coils,
currents, any differentiable field) and optionally plasma parameters
(``plasma_from_parameters(params, x)``: PHIEDGE, the current profile).  Every
design point is the three-term free boundary of
:mod:`vmex.core.freeboundary_three_terms` with the gauge labelling, accepted by
the same rule as the standalone solve and Newton-converged on the gauge, and
its design gradients are the exact implicit derivatives of that converged point
(:mod:`vmex.core.freeboundary_three_terms_implicit`).  Pass the problem to
:func:`vmex.core.optimize.minimize` (SLSQP).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

import jax
import jax.numpy as jnp

from .errors import TrialRejected, VmecConvergenceError, VmecError
from .freeboundary_three_terms import (
    POLISH_FTOL,
    BoundaryResidual,
    ThreeTermsConfig,
    ThreeTermsFreeBoundaryModel,
    _edge_pressure_error,
    _fingerprint,
)
from .freeboundary_three_terms_implicit import (
    _accept,
    _Design,
    _gauge_gradients,
    _newton_refine,
    _solve_point,
)
from .problem import FunctionProblem

__all__ = ["ThreeTermsFreeBoundaryProblem"]

#: Version of :meth:`ThreeTermsFreeBoundaryProblem.save_checkpoint`'s file.
CHECKPOINT_SCHEMA = "vmex-three-term-checkpoint/4"


def _nonlinear_constraint(values, jacobian, lower, upper, scales):
    from scipy.optimize import NonlinearConstraint

    lo, hi, scale = np.broadcast_arrays(np.asarray(lower, dtype=float),
                                        np.asarray(upper, dtype=float), np.asarray(scales, dtype=float))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("ordered bounds and finite positive constraint scales required")
    return NonlinearConstraint(lambda x: np.asarray(values(x)) / scale, lo / scale, hi / scale,
                               jac=lambda x: np.asarray(jacobian(x)) / scale[..., None])


def _constraint_row(spec):
    """``(function, lower, upper, scale)`` of a constraint entry ``(function, lower, upper[, scale])``."""
    if len(spec) not in (3, 4) or not callable(spec[0]):
        raise TypeError("constraints are (function, lower, upper) or (function, lower, upper, scale)")
    function, lower, upper = spec[:3]
    scale = float(spec[3]) if len(spec) == 4 else 1.0
    if not float(lower) <= float(upper) or not scale > 0:
        raise ValueError(f"a constraint needs lower <= upper and a positive scale, got {spec[1:]}")
    return function, float(lower), float(upper), scale


class ThreeTermsFreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its three-term free boundary, for SLSQP.

    Build with :meth:`from_loss` and pass it to :func:`vmex.core.optimize.minimize`.  Row 0 is the loss
    ``loss(state, runtime, x)``, then the ``quantities`` (``f(state, runtime)``) and the
    ``parameter_quantities`` (``f(state, runtime, x)``).  Every trial is the K = 0 boundary of the field and
    plasma parameters of ``x`` (:class:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel`) on the
    exact gauge, started from the accepted point (its boundary straight into the gauge Newton, then the
    first-order prediction of its linearization); a trial without a certified solution raises
    :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` has the exact design gradient of the converged point,
    the implicit function theorem on the gauge,

        dh/dx = dh/dx|_b - F_d^T mu,   mu = S Z (Z^T S H_L S Z)^-1 Z^T S dh/db,

    (:func:`~vmex.core.freeboundary_three_terms_implicit._gauge_gradients`): every row's ``dh/db`` and plasma term
    from one reverse pass through the equilibrium, ``H_L`` and ``F_d`` from exact second-order products.  Nothing
    recompiles when ``x`` changes.  ``model`` is the model; :meth:`boundary_residual` and ``stats`` (counts of
    trials, failures, boundary evaluations and linearizations, seconds per phase) report progress;
    :meth:`save_checkpoint` / ``from_loss(checkpoint=...)`` restart a run from its accepted point.
    """

    @classmethod
    def from_tuples(cls, inp, terms, x0, *, constraints=(), parameter_constraints=(), design_constraints=(),
                    **kwargs):
        """:meth:`from_loss` with upstream's objective tuples and the constraints beside their bounds.

        ``terms`` are ``(function, target, weight)`` as :meth:`~vmex.core.problem.VmecProblem.from_tuples`:
        ``function(state, runtime)`` (a scalar or a vector; e.g. :func:`~vmex.core.optimize.aspect_ratio`,
        :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`), and the loss is
        ``sum weight * (function - target)**2 / 2`` over the terms and their entries.  ``constraints`` are
        ``(function, lower, upper)`` or ``(function, lower, upper, scale)`` with a scalar ``function(state,
        runtime)``; ``parameter_constraints`` the same with ``function(state, runtime, x)`` (e.g. a coil-plasma
        distance); ``design_constraints`` are constraints on ``x`` alone, needing no equilibrium (a
        ``scipy.optimize.NonlinearConstraint`` such as :func:`~vmex.core.single_stage_targets.coil_constraint`).
        :meth:`constraints` then gives all of them for SLSQP, so adding a target or a bound is one entry here
        (:mod:`vmex.core.single_stage_targets` has coil, coil-plasma distance and vacuum-transform functions).
        The other keywords are :meth:`from_loss`'s."""
        terms = tuple(terms)
        if not terms:
            raise ValueError("at least one objective term is required")
        for term in terms:
            if len(term) != 3 or not callable(term[0]):
                raise TypeError("objective terms are (function, target, weight) with a callable function")
        rows = [_constraint_row(c) for c in constraints]
        parameter_rows = [_constraint_row(c) for c in parameter_constraints]

        def loss(state, runtime, x):
            total = 0.0
            for function, target, weight in terms:
                residual = jnp.ravel(jnp.asarray(function(state, runtime))) - jnp.asarray(target)
                total = total + 0.5 * jnp.sum(jnp.asarray(weight) * residual * residual)
            return total

        problem = cls.from_loss(inp, loss, x0, quantities=tuple(f for f, *_ in rows),
                                parameter_quantities=tuple(f for f, *_ in parameter_rows), **kwargs)
        problem._bounds = tuple(zip(*[bound for _, *bound in rows + parameter_rows])) if rows + parameter_rows else None
        problem._design_constraints = list(design_constraints)
        return problem

    def constraints(self):
        """SLSQP's constraints of :meth:`from_tuples`: one ``scipy.optimize.NonlinearConstraint`` of the equilibrium
        rows' bounds (each row divided by its scale), then the design constraints; a list, possibly empty."""
        bounds = getattr(self, "_bounds", None)
        rows = []
        if bounds is not None:
            lower, upper, scales = bounds
            rows.append(self.nonlinear_constraint(np.asarray(lower), np.asarray(upper), scales=np.asarray(scales)))
        return rows + list(getattr(self, "_design_constraints", []))

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_ftol=None, boundary_max_nfev=None,
                  boundary_max_residual=None, boundary_xtol=None, checkpoint=None, semantic_identity=None,
                  seed_boundary=None, config=None, **model_options):
        """Solve the seed design's free boundary (or, with ``checkpoint``, restore a saved accepted point instead:
        :meth:`restore_checkpoint`; ``x0`` then only fixes the design's layout).  ``semantic_identity`` (any
        JSON-serializable value, e.g. a version and the names of the maps and objectives) is the caller's declaration
        of what the callbacks compute: a checkpoint restores only into a problem with the same declaration, since
        callbacks that agree at one design can still differ in their derivatives.  ``seed_boundary`` (model
        coordinates, or a callable of the model returning them, e.g. a perturbed ``model.x0``) fits the seed
        design's boundary from that start instead of the deck's boundary, the deck staying the labelling reference.

        ``config`` (a :class:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig` with ``labelling="gauge"``, the
        default) sets the discrete problem (grid, quadrature, modes, ``m1_complement``, ``trial_ftol``, ``fit``) and
        the gauge Newton (``config.newton``); ``model_options`` are further
        :class:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel` keywords (``bootstrap``,
        ``bootstrap_helicity``, ``device``, ...).  A trial's boundary fit stops at a relative cost change of
        ``boundary_ftol`` or after ``boundary_max_nfev`` evaluations (default: the config's ``ftol`` and
        ``max_nfev``); every trial is then Newton-converged on the gauge and certified before it has a value (one
        record per design: a gradient never moves it).  With ``boundary_max_residual`` (off by default; unlike the
        config's ``max_residual``, the standalone solver's convergence bound, it rejects trials) a fit that ends
        with any :class:`~vmex.core.freeboundary_three_terms.BoundaryResidual` term above it is no solution: the
        trial tries its next start, or is rejected.  With ``boundary_xtol`` a trial's certified fit must also be within that (scaled) length of
        its stationary point (the certificate's ``step``; it is always recorded, ``stats["max_step"]``).  The
        certificate's normal displacement of that step is recorded too (``stats["max_normal_step"]``).
        """
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if not all(callable(f) for f in functions) or not (plasma_from_parameters is None
                                                           or callable(plasma_from_parameters)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        config = ThreeTermsConfig() if config is None else config
        if config.labelling != "gauge":  # before the model's seed solve
            raise ValueError("the single-stage problem needs labelling='gauge': its gradient differentiates the "
                             "converged point on the gauge")
        newton = config.newton.as_dict()
        boundary_ftol = config.ftol if boundary_ftol is None else boundary_ftol
        boundary_max_nfev = config.max_nfev if boundary_max_nfev is None else boundary_max_nfev
        model = ThreeTermsFreeBoundaryModel(inp, **{**config.model_options(), **model_options})
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_ftol=boundary_ftol, boundary_max_nfev=boundary_max_nfev,
                   boundary_max_residual=boundary_max_residual, boundary_xtol=boundary_xtol, checkpoint=checkpoint,
                   semantic_identity=semantic_identity, seed_boundary=seed_boundary, newton=newton)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_ftol, boundary_max_nfev, boundary_max_residual=None,
                 boundary_xtol=None, checkpoint=None, semantic_identity=None, seed_boundary=None, newton=None):
        im = model._im
        self.model, self.field = model, field_from_parameters
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_max_residual = None if boundary_max_residual is None else float(boundary_max_residual)
        self.boundary_xtol = None if boundary_xtol is None else float(boundary_xtol)
        #: The gauge Newton of every trial (:func:`~vmex.core.freeboundary_three_terms_implicit._newton_refine`): its
        #: options (step_tol, max_steps, quasi, verbose).  It makes the fit's stationary point, and so its response
        #: and exact gradient, determined.
        self.newton_polish = ThreeTermsConfig().newton.as_dict() if newton is None else dict(newton)
        if model.gauge is None:
            raise ValueError("the single-stage problem needs a model with labelling='gauge'")
        self.semantic_identity = json.loads(json.dumps(semantic_identity))
        self._n_rows = 1 + len(quantities) + len(parameter_quantities)
        x0 = np.asarray(x0, dtype=float)
        if checkpoint is not None:  # the saved design is the accepted point; no seed solve
            saved = np.asarray(np.load(checkpoint, allow_pickle=False)["x"], dtype=float)
            if saved.shape != x0.shape:
                raise ValueError(f"the checkpoint's design has {saved.size} coordinates, this problem {x0.size}")
            x0 = saved
        params_at = (lambda params, x: params) if plasma_from_parameters is None else plasma_from_parameters
        #: The design maps and the jitted pieces of the fit's cost the gauge Newton and exact derivatives share.
        self._design = design = _Design(model, params_at, field_from_parameters)
        self._params_at = design.params_at

        def part(f, with_x):
            def value(state, params, x):
                runtime = im.runtime_from_params(params, model.cfg)
                return jnp.ravel(jnp.asarray(f(state, runtime, x) if with_x else f(state, runtime)))
            return value

        parts = (part(loss, True), *(part(f, False) for f in quantities),
                 *(part(f, True) for f in parameter_quantities))

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

        self._cost_pullback, self._coordinates = design.cost_pullback, design.coordinates
        self._direct = jax.jit(jax.jacrev(rows, argnums=2))
        self._field_vjp, self._field_jvp = design.field_vjp, design.field_jvp
        self._cache, self._rejections, self._anchor = {}, {}, {}
        # seconds_*: wall time per phase (trial fits; certificates with their polish; design gradients)
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0, fit_reasons={},
                          seconds_fit=0.0, seconds_certify=0.0, seconds_gradient=0.0,
                          max_condition=0.0)
        self._continuation = None
        # The base problem only stores the layout (names, scales), which the checkpoint definition includes.
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})
        if checkpoint is not None:
            self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0  # the layout restore_checkpoint checks
            self.restore_checkpoint(checkpoint)
        else:
            if callable(seed_boundary):
                seed_boundary = seed_boundary(model)
            starts = None if seed_boundary is None else [np.asarray(seed_boundary, dtype=float)]
            seed = self._solve(x0, starts=starts)
            if seed is None:
                raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
            self._promote(seed)
            self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0

    def _checkpoint(self):
        """The continuation a trial starts from: the boundary fit's latest linearization and trust radius, and the
        equilibrium solver's hot-restart state."""
        im, model = self.model._im, self.model
        return dict(linearization=model._linearization, radius=model._radius,
                    hot={cfg: im._HOT_CACHE.get(cfg) for cfg in {model.cfg, model.cfg_trial}})

    def _restore(self, checkpoint):
        im, model = self.model._im, self.model
        model._linearization, model._radius = checkpoint["linearization"], checkpoint["radius"]
        for cfg, state in checkpoint["hot"].items():
            im._PERTURB_SEED.pop(cfg, None)
            if state is None:
                im._HOT_CACHE.pop(cfg, None)
            else:
                im._HOT_CACHE[cfg] = state

    def _solve(self, x, starts=None, polish=True):
        """The trial at ``x`` (cached): its final record (:meth:`_finalize`), or ``None`` (the reason in
        ``_rejections``).

        Every trial starts from the continuation of the accepted point, so neither a rejected trial nor the order
        of trials changes what a later trial starts from.
        """
        key = np.asarray(x, dtype=float).tobytes()
        if key in self._cache:
            return self._cache[key]
        self.stats["trials"] += 1
        x = np.asarray(x, dtype=float)
        params, field = self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x))
        if self._plasma.size:  # the plasma map may move the pressure: the problem needs it zero at the edge
            error = _edge_pressure_error(self.model._im.input_with_params(self.model.fixed, params))
            if error:
                return self._reject(key, error)
        anchor, J = self._anchor, None
        direct = None  # the index of a start that goes straight to the Newton polish
        if starts is not None:  # given starts (a restore: the saved boundary)
            anchor = {}
        elif anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
            if self.newton_polish is not None:  # the gauge Newton needs no fit
                # First: the accepted boundary straight into the gauge Newton polish (quasi-Newton from the accepted
                # point's model, in a trust region on the gauge).
                starts.insert(0, np.asarray(anchor["b"], dtype=float))
                direct = 0
        else:
            starts = [None]

        def screen(i, fit):
            if not (fit["converged"] or fit["reason"] == "stalled"):  # out of budget, or its trust radius collapsed
                return f"the fit stopped ({fit['reason']})"
            if self.boundary_max_residual is not None:  # a screen on the fit's trial state: twice the final bound
                state, _, params_b, _ = fit["aux"]
                residual = self.model.boundary_residual(state, params_b, field)
                if max(residual.normal, residual.pressure, residual.sheet_current) > 2 * self.boundary_max_residual:
                    return "the fit stopped away from the free boundary"
            return None

        def restore():  # every start from the accepted point's continuation
            if self._continuation is not None:
                self._restore(self._continuation)

        sol, why, _ = _solve_point(
            self.model, params, field, starts, accept=lambda fit: self._finalize(x, params, field, fit, polish),
            fit_options=dict(ftol=self.boundary_ftol, max_nfev=self.boundary_max_nfev, refresh_on_stall=True,
                             target_cost=4 * anchor["cost"] if anchor else None),
            jacobian=J, direct=direct, restore=restore, screen=screen, stats=self.stats)
        if sol is None:
            return self._reject(key, "; ".join(why) or "no three-term free boundary at this design point")
        self._cache[key] = sol
        return sol

    def _reject(self, key, reason):
        self.stats["failed"] += 1
        self.stats["last_rejection"] = str(reason)[:400]
        self._cache[key], self._rejections[key] = None, reason
        return None

    def _finalize(self, x, params, field, fit, polish=True, newton=True):
        """A trial's one record: its fit certified by :meth:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.certify` (re-solved to the
        deck's tolerance, polished with its own Jacobian when ``polish``, certified, and only then valued).  Every
        value, gradient, acceptance and checkpoint of ``x`` uses this record; differentiating it adds gradients and
        never moves its boundary.  Raises :class:`~vmex.core.errors.TrialRejected`."""
        xj, nb = jnp.asarray(x), self.model.x0.size
        extra = self._plasma_directions(params, xj) if self._plasma.size else None
        # With the Newton polish, the fit's own certificate does not decide: Newton converges the point and the gate
        # applies to that (the final certify of the polish); a Levenberg-Marquardt stall is only where Newton starts.
        sol = _accept(self._design, x, params, field, fit, gate=max(self.boundary_ftol, POLISH_FTOL),
                      newton=self.newton_polish if newton else None, xtol=self.boundary_xtol, polish=polish,
                      max_residual=self.boundary_max_residual, extra=extra,
                      values=lambda state, params_b: self._rows(state, params_b, xj), stats=self.stats,
                      anchor_H=(self._anchor or {}).get("gauge_H"), keep_hessian=self._keep_hessian)
        J_b, J_p = sol["J"][:, :nb], sol["J"][:, nb:]
        # What later trials start from if the optimizer accepts this one (accept_x): its linearization now.
        sol.update(continuation=self._checkpoint(), anchor=dict(
            x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=sol["state"],
            params=sol["params"], cost=0.5 * sol["rows"] @ sol["rows"]))
        return sol

    def _keep_hessian(self, record):
        """The converged Newton point's Hessian (:func:`_newton_refine`), for its exact gradient."""
        self._hessian = record

    def _newton_refine(self, x, params, field, sol, extra):
        """:func:`_newton_refine` of the record ``sol`` at design ``x``, with this problem's options."""
        xj = jnp.asarray(x)
        return _newton_refine(self._design, x, params, field, extra, sol["x"], self.newton_polish, stats=self.stats,
                              max_residual=self.boundary_max_residual,
                              values=lambda state, params_b: self._rows(state, params_b, xj),
                              anchor_H=(self._anchor or {}).get("gauge_H"), keep_hessian=self._keep_hessian)

    def _record(self, x):
        x = np.asarray(self._x(x), dtype=float)
        sol = self._solve(x)
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected(self._rejections[x.tobytes()])
        return sol

    def _linearize(self, x, sol):
        """Every row's design gradient at the record ``sol`` (cached): its certified Jacobian, one reverse pass."""
        if "gradients" in sol:
            return sol["gradients"]
        t0 = time.perf_counter()
        try:
            return self._linearize_uncached(x, sol)
        finally:
            self.stats["seconds_gradient"] += time.perf_counter() - t0

    def _linearize_uncached(self, x, sol):
        got = self._gauge_gradients(x, sol)
        sol["exact_gradient"] = {k: got[k] for k in ("true_residual", "condition", "seconds_hessian")}
        self.stats["exact_gradient_true_residual"] = float(max(got["true_residual"]))
        gradients = got["gradients"]
        if not np.all(np.isfinite(gradients)):
            raise TrialRejected("nonfinite design gradient")
        sol["gradients"] = gradients
        return gradients

    def _gauge_gradients(self, x, sol, rows=None):
        """:func:`_gauge_gradients` of every requested row at the record ``sol``: the rows' gradients from one
        reverse pass, then the gauge multipliers."""
        xj = jnp.asarray(x)
        state, params_b = sol["state"], sol["params"]
        b0 = np.asarray(sol["x"], dtype=float)
        try:
            params_bar = self._pullback(state, sol["mask"], params_b, xj)
        except (VmecError, RuntimeError) as error:
            raise TrialRejected(str(error)) from error
        G_b, G_x = map(np.asarray, self._coordinates(params_b, jnp.asarray(b0), xj, params_bar))
        direct = np.asarray(self._direct(state, params_b, xj))
        return _gauge_gradients(self._design, x, sol, G_b, G_x, direct, rows, getattr(self, "_hessian", None),
                                self.stats)

    def _promote(self, sol):
        if "anchor" in sol:
            self._anchor, self._continuation = sol["anchor"], sol["continuation"]

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
        """Promote an evaluated point after the optimizer accepts it (its anchor and continuation become what
        later trials start from); drop the other trials."""
        key = np.asarray(self._x(x), dtype=float).tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = _AcceptedPoint(np.asarray(x, dtype=float).copy()), self.accepted_step + 1
        self._promote(self._cache[key])
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
        """The free-boundary equilibrium of ``x`` as an ``Equilibrium``: the certified state the optimizer used.

        A fixed-boundary re-solve on its boundary, started from that state.  The interior root of the model's
        equations (its ``m1_complement``) is unique, so this reproduces the optimizer's state; if it moves any field
        by more than ``rtol`` (largest change relative to the field's size) this raises
        :class:`~vmex.core.errors.VmecConvergenceError` instead of returning a different equilibrium.
        """
        from .optimize import solve_equilibrium

        sol = self._record(x)
        eq = solve_equilibrium(self.model._im.input_with_params(self.model.fixed, sol["params"]),
                               initial_state=sol["state"])
        moved = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(np.asarray(b)))), 1e-300))
                    for a, b in zip(jax.tree.leaves(eq.state), jax.tree.leaves(sol["state"])) if np.size(b))
        if not moved <= rtol:
            raise VmecConvergenceError(
                f"the exported equilibrium differs from the optimizer's certified state by {moved:.2e} (> {rtol:g})",
                hint="the certified state and the re-solve are on different roots: check the m1_complement policy")
        return eq

    def _definition(self):
        """What a checkpoint must match (schema 4): the deck the model was built on (its labelling reference,
        quadrature plan and axis come from it), the model settings (including the m = 1 complement policy), the fit
        settings, the design layout (size, coordinate names and scales, plasma coordinates), the number of value
        rows, and the caller's ``semantic_identity``."""
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            deck = self.model.fixed.to_indata(Path(folder) / "input.checkpoint").read_text()
        return json.loads(json.dumps(dict(
            deck=deck, model=self.model.options, boundary_ftol=self.boundary_ftol,
            boundary_max_nfev=self.boundary_max_nfev, boundary_max_residual=self.boundary_max_residual,
            boundary_xtol=self.boundary_xtol,
            design_size=int(np.size(self.accepted.parameters)), plasma_coordinates=self._plasma.tolist(),
            names=list(self.names), scales=np.asarray(self.scales, dtype=float).tolist(), rows=self._n_rows,
            semantic_identity=self.semantic_identity)))

    def _maps(self, x):
        """Hash of what ``x`` maps to: its coil field and plasma parameters (the design maps of the problem)."""
        xj = jnp.asarray(x, dtype=float)
        return _fingerprint(self.field(xj), self._params_at(xj))

    def save_checkpoint(self, path):
        """Save the accepted point (``.npz``): its design, certified boundary and interior state, values and
        certificate, and the problem's definition.  :meth:`restore_checkpoint` re-solves it; the optimizer's own
        history is not saved, so a restart is a warm restart of the accepted equilibrium."""
        x = self.accepted.parameters
        sol = self._record(x)
        leaves = jax.tree.leaves(sol["state"])
        meta = dict(schema=CHECKPOINT_SCHEMA, accepted_step=self.accepted_step, definition=self._definition(),
                    maps=self._maps(x), certificate=sol.get("certificate"), state_leaves=len(leaves))
        path = Path(path)
        partial = path.with_name(path.name + ".partial.npz")  # atomic: a killed run leaves the last whole file
        np.savez(partial, x=x, b=np.asarray(sol["x"]), values=sol["values"], rows=sol["rows"],
                 meta=json.dumps(meta), **{f"state{i}": np.asarray(a) for i, a in enumerate(leaves)})
        os.replace(partial, path)

    def restore_checkpoint(self, path, *, rtol=1e-6):
        """Make a saved accepted point (:meth:`save_checkpoint`) this problem's accepted point.

        The problem must have been built with the same definition (deck, settings, design layout) or this raises
        :class:`ValueError`.  The saved boundary itself is evaluated (its interior equilibrium re-solved from the
        saved state, no re-fit: a re-fit moves a soft boundary within the fit's tolerance) and certified; every
        value must agree with the saved one to ``rtol`` relative to its own size or this raises
        :class:`~vmex.core.errors.VmecError`.  Returns the largest relative value change.
        """
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {meta.get('schema')!r}")
        definition = self._definition()
        differ = sorted(k for k in set(definition) | set(meta["definition"])
                        if definition.get(k) != meta["definition"].get(k))
        if differ:
            raise ValueError(f"the checkpoint is of a different problem: {', '.join(differ)} differ")
        model, x = self.model, np.asarray(data["x"], dtype=float)
        if meta["maps"] != self._maps(x):
            raise ValueError("the checkpoint is of a different problem: its design's coil field or plasma "
                             "parameters differ")
        state = jax.tree.unflatten(jax.tree.structure(model.seed[0]),
                                   [jnp.asarray(data[f"state{i}"]) for i in range(meta["state_leaves"])])
        previous = self._continuation  # back in place unless the saved point restores and validates
        self._restore(dict(linearization=None, radius=None, hot={cfg: state for cfg in {model.cfg, model.cfg_trial}}))
        try:
            b, xj = np.asarray(data["b"], dtype=float), jnp.asarray(x)
            params, field = self._params_at(xj), self.field(xj)
            got = model.evaluate(b, params, field, seed=state, tight=True)
            if got is None:
                raise VmecError("the checkpoint's boundary has no certified equilibrium from its saved state")
            rows, aux = got
            fit = dict(x=b, rows=rows, aux=aux, nfev=1, njev=0, accepted=0, converged=True, radius=None,
                       reason="restored")
            try:
                # exactly the saved point (no Newton refinement: refine_accepted() is the explicit step)
                sol = self._finalize(x, params, field, fit, polish=False, newton=False)
            except TrialRejected as error:
                raise VmecError(f"the checkpoint's boundary is not certified: {error}") from error
            saved = np.asarray(data["values"])
            change = float(np.max(np.abs(sol["values"] - saved) / np.maximum(np.abs(saved), 1e-300)))
            if not change <= rtol:
                raise VmecError(f"the restored equilibrium changes the saved values by {change:.2e} (> {rtol:g})")
        except Exception:
            if previous is not None:
                self._restore(previous)
            raise
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), int(meta["accepted_step"])
        self._promote(sol)
        self._cache = {x.tobytes(): sol}
        return change

    def refine_accepted(self):
        """Newton-refine the accepted record (:meth:`_newton_refine`), e.g. a restored
        checkpoint certified by its gain only.  Returns the largest relative change of its values, or NaN when the
        polish does not converge: the restored record (the saved one exactly) is then kept, and the next trials are
        polished as always."""
        x = self.accepted.parameters
        key = x.tobytes()
        sol = self._cache[key]
        xj = jnp.asarray(x)
        params, field = self._params_at(xj), self.field(xj)
        extra = self._plasma_directions(params, xj) if self._plasma.size else None
        try:  # the gauge Newton converges (quadratically) from far: its full budget
            new = self._newton_refine(x, params, field, sol, extra)
        except TrialRejected as error:
            self.stats["refine_unconverged"] = str(error)[:200]
            if self._continuation is not None:
                self._restore(self._continuation)
            return float("nan")
        nb = self.model.x0.size
        new.update(continuation=self._checkpoint(), anchor=dict(
            x=np.asarray(x, dtype=float).copy(), b=new["x"], J_b=new["J"][:, :nb], J_p=new["J"][:, nb:],
            state=new["state"], params=new["params"], cost=0.5 * new["rows"] @ new["rows"]))
        change = float(np.max(np.abs(new["values"] - sol["values"]) / np.maximum(np.abs(sol["values"]), 1e-300)))
        self._promote(new)
        self._cache[key] = new
        return change

    def close(self):
        """Drop the cached trials but the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray

