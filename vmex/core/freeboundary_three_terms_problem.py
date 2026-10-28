"""Single-stage optimization at the three-term free boundary: a loss and constraint rows of coils and plasma
parameters, with exact design gradients (:class:`ThreeTermsFreeBoundaryProblem`).

Every design point is solved once: its boundary fit is re-solved tight, polished with its own Jacobian and
certified (:data:`POLISH_FTOL`) before it has a value, and its gradient is that record's implicit derivative, so a
gradient never moves a value.  Trials start from the continuation of the accepted point only, and
:meth:`ThreeTermsFreeBoundaryProblem.accept_x` (called by :func:`vmex.core.optimize.minimize` when SLSQP accepts a
step) advances it; checkpoints restore an accepted point exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

import jax
import jax.numpy as jnp

from .errors import TrialRejected, VmecError
from .freeboundary_three_terms import (BoundaryResidual, ThreeTermsConfig, ThreeTermsModel, _edge_pressure_error,
                                       _fingerprint, _project)
from .problem import FunctionProblem

#: Stationarity (the gain a Gauss-Newton step predicts) a trial's polish rounds aim at.
POLISH_FTOL = 1e-4
#: Most polish rounds (each with a fresh Jacobian) a trial gets to reach its certificate.
POLISH_ROUNDS = 5

#: Version of :meth:`ThreeTermsFreeBoundaryProblem.save_checkpoint`'s file.
CHECKPOINT_SCHEMA = "vmex-three-terms-checkpoint/1"


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
    first-order prediction of the latest linearized solution; a trial without a certified solution raises
    :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design gradient of the
    implicit function theorem of the boundary least squares,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  Nothing recompiles
    when ``x`` changes.  ``model`` is the :class:`ThreeTermsModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures, boundary evaluations and linearizations) report progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_quantities=(), config=None, boundary_ftol=1e-3,
                  boundary_max_nfev=40, boundary_max_residual=None, boundary_xtol=None):
        """Solve the seed design's free boundary (that of ``x0`` on the deck ``inp``).

        ``config`` (a :class:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig`) sets the labelling, grid,
        quadrature, trial tolerance and bootstrap current of the model; its fit tolerances and ``start`` are
        the solver's, not the problem's.  A trial's boundary fit stops at a
        relative cost change of ``boundary_ftol`` or after ``boundary_max_nfev`` evaluations; every trial is then
        re-solved tight, polished with its own Jacobian and certified before it has a value (one record per
        design: a gradient never moves it).  With ``boundary_max_residual`` a fit that ends with
        any :class:`BoundaryResidual` term above it is no solution: the trial tries its next start, or is
        rejected.  With ``boundary_xtol`` a trial's certified fit must also be within that (scaled) length of
        its stationary point (the certificate's ``step``; it is always recorded, ``stats["max_step"]``).
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
                   boundary_quantities=tuple(boundary_quantities), boundary_ftol=boundary_ftol, boundary_max_nfev=boundary_max_nfev,
                   boundary_max_residual=boundary_max_residual, boundary_xtol=boundary_xtol)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_ftol, boundary_max_nfev, boundary_max_residual=None,
                 boundary_xtol=None, boundary_quantities=()):
        im = model._im
        self.model, self.field = model, field_from_parameters
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_max_residual = None if boundary_max_residual is None else float(boundary_max_residual)
        self.boundary_xtol = None if boundary_xtol is None else float(boundary_xtol)
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
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0, fit_reasons={},
                          rejection_reasons={}, max_condition=0.0)
        self._continuation = None
        seed = self._solve(x0)
        if seed is None:
            raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
        self._promote(seed)
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

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
        if starts is not None:  # given starts (a restore: the saved boundary)
            anchor = {}
        elif anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        else:
            starts = [None]
        fit = None
        for start in starts:
            if self._continuation is not None:  # every start from the accepted point's continuation
                self._restore(self._continuation)
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, ftol=self.boundary_ftol,
                                                max_nfev=self.boundary_max_nfev, refresh_on_stall=True,
                                                target_cost=4 * anchor["cost"] if anchor else None)
            except (VmecError, RuntimeError):  # an uncertified equilibrium: try the next start
                continue
            reasons = self.stats["fit_reasons"]
            reasons[fit["reason"]] = reasons.get(fit["reason"], 0) + 1
            if not fit["converged"]:  # a finite fit is no solution: radius collapse or budget exhausted
                self.stats["boundary_evaluations"] += fit["nfev"]
                fit = None
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
        try:
            sol = self._finalize(x, params, field, fit, polish)
        except TrialRejected as error:
            return self._reject(key, str(error))
        self._cache[key] = sol
        return sol

    def _reject(self, key, reason):
        self.stats["failed"] += 1
        kind = re.sub(r"[-+]?\d[\d.]*(e[-+]?\d+)?", "#", reason)[:80]  # the reason without its numbers
        self.stats["rejection_reasons"][kind] = self.stats["rejection_reasons"].get(kind, 0) + 1
        self._cache[key], self._rejections[key] = None, reason
        return None

    def _finalize(self, x, params, field, fit, polish=True):
        """A trial's one record: its fit re-solved to the deck's tolerance, polished with its own Jacobian
        (``polish``), certified, and only then valued.  Every value, gradient, acceptance and checkpoint of ``x``
        uses this record; differentiating it adds gradients and never moves its boundary.  Raises
        :class:`~vmex.core.errors.TrialRejected`."""
        model, xj, nb = self.model, jnp.asarray(x), self.model.x0.size
        extra = self._plasma_directions(params, xj) if self._plasma.size else None

        def tight(b, aux, project=True):
            try:  # the trial's state may not tighten to the deck's tolerance, or a row's own solve may fail
                J, _, _, aux = model.linearize(b, aux, field, extra=extra)
                state, mask, params_b, _ = aux
                rows, values = np.asarray(model._rows(state, params_b, field)), np.asarray(self._rows(state, params_b, xj))
            except (VmecError, RuntimeError) as error:  # RuntimeError: the relayed sentinel of a VmecError
                raise TrialRejected(str(error)) from error
            J = np.asarray(J)
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(rows)) and np.all(np.isfinite(values))):
                raise TrialRejected("nonfinite rows, values or boundary Jacobian at the tightened state")
            self.stats["linearizations"] += 1
            index = model._projected
            if project and index is not None:
                # The fit closed the bootstrap rows on its trial-tolerance state; tightened, they reopen (QH 2.5%:
                # 9e-11 -> 1.9e-4, 92% of the cost).  Re-solve the current knots on this state, then linearize there.
                try:
                    b_p, got, _ = _project(lambda z: model.evaluate(z, params, field, tight=True), np.asarray(b),
                                           (rows, aux), J[:, index], index, 2)
                except (VmecError, RuntimeError) as error:
                    raise TrialRejected(str(error)) from error
                if not np.array_equal(b_p, b):
                    return tight(b_p, got[1], project=False)
            return dict(x=np.asarray(b), aux=aux, state=state, mask=mask, params=params_b, rows=rows, values=values,
                        J=J)

        sol = dict(fit, **tight(fit["x"], fit["aux"]))
        gate = max(self.boundary_ftol, POLISH_FTOL)  # beyond it a fit is rejected; the polish aims at POLISH_FTOL
        for round_ in range(POLISH_ROUNDS + 1):
            J_b, J_p = sol["J"][:, :nb], sol["J"][:, nb:]
            sol["certificate"] = certificate = _fit_certificate(J_b * np.asarray(model.x_scale)[None], sol["rows"])
            self.stats["max_condition"] = max(self.stats["max_condition"], certificate["condition"])
            if certificate["rank_deficient"]:
                raise TrialRejected(f"rank-deficient boundary Jacobian (condition {certificate['condition']:.2e})")
            done = certificate["stationarity"] <= POLISH_FTOL and (self.boundary_xtol is None
                                                                   or certificate["step"] <= self.boundary_xtol)
            if done or not polish or round_ == POLISH_ROUNDS:
                break
            # Short of the polish target: polish with this (fresh, tight) Jacobian and certify again.
            try:
                polished = model.solve_boundary(params, field, x0=sol["x"], jacobian=J_b, ftol=0.1 * POLISH_FTOL,
                                                max_nfev=20)
            except (VmecError, RuntimeError):
                self.stats["failed_polishes"] = self.stats.get("failed_polishes", 0) + 1
                break
            self.stats["boundary_evaluations"] += polished["nfev"]
            if not polished["rows"] @ polished["rows"] < sol["rows"] @ sol["rows"]:
                break  # the polish gains nothing: the certificate below decides
            sol.update(tight(polished["x"], polished["aux"]))  # the same fit, nearer its stationary point
            self.stats["polish_rounds"] = self.stats.get("polish_rounds", 0) + 1
        if certificate["stationarity"] > gate:
            raise TrialRejected(f"boundary fit not stationary: a Gauss-Newton step with its own Jacobian gains "
                                f"{certificate['stationarity']:.2e} of the cost")
        self.stats["max_step"] = max(self.stats.get("max_step", 0.0), certificate["step"])
        if self.boundary_xtol is not None and certificate["step"] > self.boundary_xtol:
            raise TrialRejected(f"boundary fit not stationary: its Gauss-Newton step is {certificate['step']:.2e} "
                                f"(scaled) long, above boundary_xtol")
        if self.boundary_max_residual is not None:  # on the final state, not the fit's trial one
            residual = model.boundary_residual(sol["state"], sol["params"], field)
            if max(residual.normal, residual.pressure, residual.sheet_current) > self.boundary_max_residual:
                raise TrialRejected("the certified fit is not a free boundary: an interface residual is above "
                                    "boundary_max_residual")
        # What later trials start from if the optimizer accepts this one (accept_x): its linearization now.
        sol.update(continuation=self._checkpoint(), anchor=dict(
            x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=sol["state"],
            params=sol["params"], cost=0.5 * sol["rows"] @ sol["rows"]))
        return sol

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
        xj, nb = jnp.asarray(x), self.model.x0.size
        J_b, J_p = sol["J"][:, :nb], sol["J"][:, nb:]
        state, params = sol["state"], sol["params"]
        try:
            params_bar = self._pullback(state, sol["mask"], params, xj)
        except (VmecError, RuntimeError) as error:  # an adjoint solve that does not certify
            raise TrialRejected(str(error)) from error
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        Q, R = np.linalg.qr(J_b)
        lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column))) for column in lam.T])
        gradients[:, self._plasma] -= lam.T @ J_p
        if not np.all(np.isfinite(gradients)):
            raise TrialRejected("nonfinite design gradient")
        sol["gradients"] = gradients
        return gradients

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

    def _definition(self):
        """What a checkpoint must match: the deck the model was built on (its labelling reference, quadrature plan
        and axis come from it), the model and fit settings, and the design layout."""
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            deck = self.model.fixed.to_indata(Path(folder) / "input.checkpoint").read_text()
        return json.loads(json.dumps(dict(
            deck=deck, model=self.model.options, boundary_ftol=self.boundary_ftol,
            boundary_max_nfev=self.boundary_max_nfev, boundary_max_residual=self.boundary_max_residual,
            boundary_xtol=self.boundary_xtol,
            design_size=int(np.size(self.accepted.parameters)), plasma_coordinates=self._plasma.tolist())))

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
        np.savez(path, x=x, b=np.asarray(sol["x"]), values=sol["values"], rows=sol["rows"],
                 meta=json.dumps(meta), **{f"state{i}": np.asarray(a) for i, a in enumerate(leaves)})

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
                sol = self._finalize(x, params, field, fit, polish=False)
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

    def close(self):
        """Drop the cached trials but the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


def _fit_certificate(J, r):
    """Stationarity and conditioning of a boundary fit with rows ``r`` and (scaled) Jacobian ``J`` at its point.

    ``stationarity`` is the relative cost reduction a Gauss-Newton step with this Jacobian predicts,
    ``|Q^T r|^2 / |r|^2``, and ``step`` that step's length, ``|J^+ r|`` in the (scaled) coordinates: in a flat
    direction a fit can be far from its stationary point while the cost barely changes, which only ``step`` sees.
    ``condition`` is ``J``'s condition number, and ``rank_deficient`` marks a smallest singular value at the
    rounding level, ``max(J.shape) * eps`` of the largest (NumPy's rank rule).
    """
    sv = np.linalg.svd(J, compute_uv=False)
    Q, R = np.linalg.qr(J)
    cost = r @ r
    step = float(np.linalg.norm(np.linalg.solve(R, Q.T @ r))) if sv[-1] > 0 else np.inf
    return dict(stationarity=float(np.sum((Q.T @ r) ** 2) / cost) if cost > 0 else 0.0, step=step,
                condition=float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
                rank_deficient=bool(sv[-1] <= sv[0] * max(J.shape) * np.finfo(float).eps))


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray

