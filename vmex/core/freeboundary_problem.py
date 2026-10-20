"""Scalar free-boundary optimization of the external field: a loss and constraint rows at the free boundary.

:class:`FreeBoundaryProblem` varies a design vector ``x`` that sets the
external field through ``field_from_parameters(x)`` (coils, currents, or any
differentiable field; coil charts live with the caller, for example in ESSOS)
and, optionally, plasma parameters such as PHIEDGE through
``plasma_from_parameters(params, x)``. Every design point is the VMEC + NESTOR
free boundary of :func:`~vmex.core.freeboundary_implicit.solve_free_boundary_implicit`
(an ordinary solve, Newton-anchored on its coupled root) and its derivatives
are that root's implicit adjoint. ``boundary_condition="three_term"`` builds
:class:`~vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem` instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import freeboundary_implicit as fbi, implicit as im
from .errors import TrialRejected, VmecError
from .optimize import Equilibrium
from .problem import FunctionProblem

__all__ = ["FreeBoundaryProblem"]


def _nonlinear_constraint(values, jacobian, lower, upper, scales):
    from scipy.optimize import NonlinearConstraint

    lo, hi, scale = np.broadcast_arrays(np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float), np.asarray(scales, dtype=float))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("ordered bounds and finite positive constraint scales required")
    return NonlinearConstraint(lambda x: np.asarray(values(x))/scale, lo/scale, hi/scale,
                               jac=lambda x: np.asarray(jacobian(x))/scale[..., None])


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
    # Immutable backing bytes keep accepted parameters and cache keys fixed.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


class _Point:
    """One evaluated design point: its rows, and their derivatives once asked for."""

    def __init__(self, parameters, rows):
        self.parameters, self.rows, self.jac = parameters, rows, None


class FreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its free boundary, for SLSQP.

    Build with :meth:`from_loss` and pass it to :func:`vmex.core.optimize.minimize`. Row 0 is the loss, then the
    ``quantities`` and the ``parameter_quantities``. Each design point is solved once and cached; its gradient
    (:meth:`value_and_grad`) and constraint Jacobian (:meth:`constraint_jac`) are reverse-mode derivatives
    through the implicit adjoint of the solve, one adjoint per row. A point whose free boundary does not solve
    raises :class:`~vmex.core.errors.TrialRejected`. :meth:`accept_x` records the point the optimizer accepted
    (``accepted``, ``accepted_step``).
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None,
                  scales=None, names=None, solver_options=None, quantities=(), parameter_quantities=(),
                  boundary_condition="nestor", three_term_options=None):
        """The problem of the scalar ``loss(state, runtime, x)`` from the design point ``x0``.

        Parameters
        ----------
        inp:
            Free-boundary input (``lfreeb``); it fixes the pressure and plasma
            current profiles, except where ``plasma_from_parameters`` sets them.
        loss:
            Scalar ``loss(state, runtime, x)``, used without normalization.
            Its gradient includes the equilibrium response and the explicit
            dependence on ``x``.
        x0:
            Initial design vector.
        field_from_parameters:
            Differentiable ``x -> external field`` (anything with ``b_cyl``,
            for example a Biot-Savart field of coils built from ``x``), as in
            :func:`~vmex.core.freeboundary_implicit.make_free_boundary_config`.
        plasma_from_parameters:
            Optional differentiable ``(params, x) -> params`` setting plasma
            parameters (for example PHIEDGE or the prescribed current profile)
            from ``x``; ``params`` are the input's
            :class:`~vmex.core.implicit.ImplicitParams`.
        scales, names:
            Coordinate scales (default ones) and names of ``x``.
        solver_options:
            Keyword arguments of
            :func:`~vmex.core.freeboundary_implicit.make_free_boundary_config`
            (for example ``ns``, ``ftol``, ``max_iterations`` or ``adjoint_tol``).
        quantities:
            Scalar or 1-D ``function(state, runtime)`` observables, the rows
            of :meth:`constraint_values`; the optimizer defines their bounds.
        parameter_quantities:
            Scalar or 1-D ``function(state, runtime, x)`` observables,
            appended after ``quantities``; their derivatives include the
            explicit and the equilibrium terms. Constraints on ``x`` alone
            (coil geometry, say) belong in ordinary optimizer constraints,
            which need no equilibrium adjoint.
        boundary_condition:
            ``"nestor"`` (VMEX + NESTOR) or ``"three_term"``
            (:class:`~vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem`, with ``three_term_options``).

        Returns
        -------
        FreeBoundaryProblem
            A problem whose accepted point is ``x0``, solved.
        """
        if boundary_condition == "three_term":
            from .freeboundary_vc import ThreeTermFreeBoundaryProblem

            return ThreeTermFreeBoundaryProblem.from_loss(
                inp, loss, x0, field_from_parameters=field_from_parameters,
                plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                quantities=quantities, parameter_quantities=parameter_quantities,
                **dict(three_term_options or {}))
        if boundary_condition != "nestor":
            raise ValueError(f"boundary_condition must be 'nestor' or 'three_term', got {boundary_condition!r}")
        quantities, parameter_quantities = tuple(quantities), tuple(parameter_quantities)
        if not all(callable(f) for f in (loss, field_from_parameters, *quantities, *parameter_quantities)) or (
                plasma_from_parameters is not None and not callable(plasma_from_parameters)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary implicit optimization requires JAX_ENABLE_X64=1")
        if not inp.lfreeb:
            raise ValueError("input must enable free-boundary equilibrium")
        point = _vector(x0)
        cfg = fbi.make_free_boundary_config(inp, field_from_parameters(jnp.asarray(point)),
                                            field_from_parameters=field_from_parameters, **dict(solver_options or {}))
        return cls(inp, cfg, loss, point, plasma_from_parameters=plasma_from_parameters, scales=scales,
                   names=names, quantities=quantities, parameter_quantities=parameter_quantities)

    def __init__(self, inp, cfg, loss, x0, *, plasma_from_parameters=None, scales=None, names=None,
                 quantities=(), parameter_quantities=()):
        self.inp, self.cfg = inp, cfg
        self._plasma_from_parameters = plasma_from_parameters
        params0 = im.params_from_input(inp)
        # Fill implicit's static tables (_boundary_pack_tables) outside any trace: filled while tracing rows they would
        # hold a tracer, which the host solve then meets as a leaked one.
        im.runtime_from_params(params0, cfg.implicit)

        def params_at(x):
            return params0 if plasma_from_parameters is None else plasma_from_parameters(params0, x)

        def solved(x):
            params = params_at(x)
            return fbi.solve_free_boundary_implicit(params, x, cfg), im.runtime_from_params(params, cfg.implicit)

        def rows(x):
            state, runtime = solved(x)
            value = jnp.asarray(loss(state, runtime, x))
            if value.shape != ():
                raise ValueError("loss must return a scalar")
            values = [jnp.asarray(f(state, runtime)) for f in quantities]
            values += [jnp.asarray(f(state, runtime, x)) for f in parameter_quantities]
            if any(v.ndim > 1 for v in values):
                raise ValueError("quantities must be scalars or one-dimensional arrays")
            return jnp.concatenate([value[None], *(jnp.ravel(v) for v in values)])

        self._params_at, self._solved = params_at, jax.jit(solved)
        self._rows = jax.jit(rows)
        self._jac = jax.jit(jax.jacrev(rows))  # one implicit adjoint per row
        self._points = {}
        x0 = _vector(x0)
        self._shape = x0.shape
        self.accepted, self.accepted_step = self._point(x0), 0
        super().__init__(x0, names=names, scales=np.ones_like(x0) if scales is None else scales,
                         fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _point(self, x):
        x = _vector(x, self._shape)
        key = x.tobytes()
        if key not in self._points:
            try:
                rows = np.asarray(self._rows(jnp.asarray(x)))
            except (VmecError, RuntimeError) as error:
                raise TrialRejected(f"no free boundary at this design point: {error}") from error
            if not np.all(np.isfinite(rows)):
                raise TrialRejected("nonfinite objective or constraint rows at this design point")
            self._points[key] = _Point(x, rows)
        return self._points[key]

    def _derivatives(self, point):
        if point.jac is None:
            try:
                point.jac = np.asarray(self._jac(jnp.asarray(point.parameters)))
            except (VmecError, RuntimeError) as error:
                raise TrialRejected(f"no adjoint at this design point: {error}") from error
        return point.jac

    def _value(self, x):
        return float(self._point(x).rows[0])

    def _value_gradient(self, x):
        point = self._point(x)
        return float(point.rows[0]), self._derivatives(point)[0].copy()

    def accept_x(self, x):
        """Record the evaluated point ``x`` as the one the optimizer accepted."""
        point = self._points.get(_vector(x, self._shape).tobytes())
        if point is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = point, self.accepted_step + 1

    def constraint_values(self, x):
        """Return unscaled physical quantities, in their original units."""
        return self._point(x).rows[1:].copy()

    def constraint_jac(self, x):
        """Return derivatives of the physical quantities with respect to x."""
        return self._derivatives(self._point(x))[1:].copy()

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound the physical quantities supplied to from_loss, in their units."""
        return _nonlinear_constraint(self.constraint_values, self.constraint_jac, lower, upper, scales)

    def equilibrium_from_x(self, x):
        """The free-boundary equilibrium of design point ``x``."""
        x = jnp.asarray(self._point(x).parameters)
        state, runtime = self._solved(x)
        inp = self.inp if self._plasma_from_parameters is None else im.input_with_params(self.inp, self._params_at(x))
        return Equilibrium(inp, state, runtime, fbi._FREE_LAST_RESULT.get(self.cfg))

    def close(self):
        """Drop every cached point but the accepted one."""
        self._points = {self.accepted.parameters.tobytes(): self.accepted}
