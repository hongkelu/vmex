"""Coil designs and coil constraint rows for single-stage free-boundary optimization.

:class:`CoilDesign` maps one design vector to ESSOS coils (and optionally the
plasma's PHIEDGE and current profile): it is the ``field_from_parameters`` and
``plasma_from_parameters`` of a free-boundary problem such as
:class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`,
and its field is ESSOS' differentiable ``BiotSavart``.  :class:`CoilLimits`
turns ESSOS' coil lengths, curvatures and distances into scaled inequality
rows, and :func:`coil_constraint` makes the pure-coil rows a SciPy
``NonlinearConstraint`` that needs no equilibrium solve.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

from .._compat import require_optional


class CoilDesign:
    """Design vector -> ESSOS coils and plasma parameters.

    ``x`` holds, in order: with ``phiedge`` (the deck's PHIEDGE in Wb) a
    relative PHIEDGE change, PHIEDGE = phiedge * (1 + x[0]); with
    ``plasma_current`` (the deck's leading ``k`` AC coefficients, or AC_AUX_F
    values when ``plasma_current_spline``, then CURTOR in A) ``k + 1`` additive
    changes of those values in units of the largest nominal shape value and of
    ``abs(CURTOR)``; relative changes of the ``current_dofs`` base-coil currents
    (``current[i] = nominal[i] * (1 + x[k])``); and additive changes of every
    base coil's unscaled Cartesian Fourier coefficients in metres.  ``x0 = 0`` is
    the nominal design.  Calling the design gives the ``BiotSavart`` field of
    :meth:`coils_from_x`; :attr:`plasma_from_parameters` (``None`` without plasma
    coordinates) maps solver parameters to the design's PHIEDGE and current.
    ``scales`` default to 0.06 per relative current, 0.002 m per constant
    Fourier coefficient, ``0.002 / k**2`` m per coefficient of order ``k``, and
    ``phiedge_scale`` / ``plasma_current_scale`` for the plasma coordinates.
    """

    def __init__(self, coils, *, current_dofs=(), scales=None, phiedge=None, phiedge_scale=0.05,
                 plasma_current=None, plasma_current_spline=False, plasma_current_scale=0.05):
        require_optional("essos", "coil designs")
        self.coefficients = np.asarray(coils.dofs_curves) / np.asarray(coils.curves.scaling)[None, None, :]
        self.currents = np.asarray(coils.dofs_currents_raw, dtype=float)
        self.current_dofs = tuple(int(i) for i in current_dofs)
        if len(set(self.current_dofs)) != len(self.current_dofs) or any(
                not 0 <= i < len(self.currents) or self.currents[i] == 0 for i in self.current_dofs):
            raise ValueError("current_dofs must be unique indices of base coils with nonzero currents")
        self.nfp, self.stellsym, self.n_segments = int(coils.nfp), bool(coils.stellsym), int(coils.n_segments)
        if phiedge is not None and not (np.isfinite(phiedge) and phiedge != 0 and phiedge_scale > 0):
            raise ValueError("phiedge must be finite and nonzero with a positive phiedge_scale")
        self.phiedge = None if phiedge is None else float(phiedge)
        self.nphiedge = int(phiedge is not None)
        self.plasma_current = None if plasma_current is None else np.asarray(plasma_current, dtype=float)
        self.plasma_current_spline = bool(plasma_current_spline)
        self.nplasma = 0 if self.plasma_current is None else self.plasma_current.size
        if self.plasma_current is not None:
            nominal = self.plasma_current
            if nominal.ndim != 1 or nominal.size < 2 or not plasma_current_scale > 0:
                raise ValueError("plasma_current is [k >= 1 shape values..., CURTOR], with a positive scale")
            self.plasma_current_units = np.r_[np.full(nominal.size - 1, np.max(np.abs(nominal[:-1]))),
                                              abs(nominal[-1])]
            if not np.all(np.isfinite(self.plasma_current_units)) or np.any(self.plasma_current_units == 0):
                raise ValueError("plasma_current needs finite values, a nonzero shape value and a nonzero CURTOR")
        self._coil0 = self.nphiedge + self.nplasma
        self.size = self._coil0 + len(self.current_dofs) + self.coefficients.size
        self.x0 = np.zeros(self.size)
        order = (self.coefficients.shape[2] - 1) // 2
        modes = ["constant"] + [f"{kind}({k})" for k in range(1, order + 1) for kind in ("sin", "cos")]
        self.dof_names = tuple(["phiedge/nominal"] * self.nphiedge
                               + [f"plasma_current[{j}]/unit" for j in range(self.nplasma - 1)]
                               + ["curtor/nominal"] * bool(self.nplasma)
                               + [f"current[{i}]/nominal" for i in self.current_dofs]
                               + [f"coil[{i}].{axis}.{mode}" for i in range(len(self.currents))
                                  for axis in "xyz" for mode in modes])
        if scales is None:
            mode_scales = [0.002] + [0.002 / k**2 for k in range(1, order + 1) for _ in range(2)]
            scales = np.r_[np.full(self.nphiedge, phiedge_scale), np.full(self.nplasma, plasma_current_scale),
                           np.full(len(self.current_dofs), 0.06), np.tile(mode_scales, 3 * len(self.currents))]
        self.scales = np.asarray(scales, dtype=float)
        if self.scales.shape != self.x0.shape or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("one positive finite scale per coordinate required")
        self.plasma_from_parameters = self.plasma_params_at if self._coil0 else None

    def phiedge_at(self, x):
        """PHIEDGE [Wb] at ``x``; only for a design built with ``phiedge``."""
        if self.phiedge is None:
            raise ValueError("this design does not vary PHIEDGE")
        return self.phiedge * (1.0 + jnp.asarray(x)[0])

    def plasma_params_at(self, params, x):
        """``params`` (``ImplicitParams``) with the PHIEDGE and current profile of ``x``."""
        x = jnp.asarray(x)
        if self.phiedge is not None:
            params = dataclasses.replace(params, phiedge=self.phiedge_at(x))
        if self.plasma_current is not None:
            values = jnp.asarray(self.plasma_current) + x[self.nphiedge:self._coil0] * self.plasma_current_units
            name = "ac_aux_f" if self.plasma_current_spline else "ac"
            profile = getattr(params, name).at[: self.nplasma - 1].set(values[:-1])
            params = dataclasses.replace(params, curtor=values[-1], **{name: profile})
        return params

    def base_currents_at(self, x):
        """Base-coil currents [A] at ``x``."""
        x = jnp.asarray(x)
        currents = jnp.asarray(self.currents)
        for local, base in enumerate(self.current_dofs):
            currents = currents.at[base].add(x[self._coil0 + local] * self.currents[base])
        return currents

    def coils_from_x(self, x):
        """ESSOS coils at ``x``, without changing the nominal coils."""
        from essos.coils import Coils, Curves

        x = jnp.asarray(x)
        raw = jnp.asarray(self.coefficients) + x[self._coil0 + len(self.current_dofs):].reshape(
            self.coefficients.shape)
        return Coils(Curves(raw, self.n_segments, self.nfp, self.stellsym), self.base_currents_at(x))

    def __call__(self, x):
        """The differentiable filament field (``essos.fields.BiotSavart``) of :meth:`coils_from_x`."""
        from essos.fields import BiotSavart

        return BiotSavart(self.coils_from_x(x))


def resampled(coils, points: int, *, symmetric: bool = True):
    """ESSOS curves of ``coils`` on ``points`` quadrature points; only the base curves unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


@dataclass(frozen=True)
class CoilLimits:
    """Engineering limits of a coil set, as scaled inequality rows (feasible where every row is >= 0).

    Each row is ``(limit - margin - value) / limit`` (or ``(value - limit - margin) / limit`` for a
    distance): the margins keep a design that SLSQP holds only to its tolerance strictly inside the
    limits.  Curvature, its arc-length mean square and the length are sampled on ``curvature_points``
    points of each base coil, distances on ``distance_points`` points of every coil (ESSOS'
    segment-exact ``coil_coil_distances`` / ``coil_self_distances`` / ``coil_surface_distances``).
    ``self_clearance`` (m) guards non-intersection, ``min_speed`` (m per unit angle) a degenerate
    parametrization.
    """

    length: float
    curvature: float
    mean_squared_curvature: float
    coil_distance: float
    surface_distance: float | None = None
    length_margin: float = 1e-5
    curvature_margin: float = 0.10
    mean_squared_curvature_margin: float = 0.02
    distance_margin: float = 0.001
    self_clearance: float = 1e-6
    min_speed: float = 1e-4
    curvature_points: int = 512
    distance_points: int = 128

    def rows(self, coils, *, per_point: bool = False):
        """The coil rows: per base coil its length, peak curvature, mean squared curvature and minimum speed,
        and the coil-coil and self distances.  ``per_point`` gives the curvature and distances one row per
        sample point instead of their extremes: smooth where the extreme switches points, the smallest of
        them equal to the extreme row."""
        require_optional("essos", "coil constraint rows")
        from essos.objective_functions import (coil_coil_distances, coil_mean_squared_curvature,
                                               coil_self_distances)

        base = resampled(coils, self.curvature_points, symmetric=False)
        speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
        curves = resampled(coils, self.distance_points)
        coil, own = coil_coil_distances(curves), coil_self_distances(curves)
        curvature = base.curvature if per_point else jnp.max(base.curvature, axis=1)
        if not per_point:
            coil, own = jnp.min(coil), jnp.min(own)
        return jnp.concatenate((
            (self.length - self.length_margin - base.length) / self.length,
            jnp.ravel(self.curvature - self.curvature_margin - curvature) / self.curvature,
            (self.mean_squared_curvature - self.mean_squared_curvature_margin - coil_mean_squared_curvature(base))
            / self.mean_squared_curvature,
            jnp.ravel(coil - self.coil_distance - self.distance_margin) / self.coil_distance,
            jnp.ravel(own - self.self_clearance) / self.coil_distance,
            (jnp.min(speed, axis=1) - self.min_speed) / self.length))

    def surface_clearance(self, coils, surface):
        """Smallest distance [m] from the coils (every symmetry copy) to ``surface.gamma``."""
        require_optional("essos", "coil clearances")
        from essos.objective_functions import coil_surface_distances

        return jnp.min(coil_surface_distances(resampled(coils, self.distance_points), surface))


def chunked_jacfwd(fun, chunk: int = 32):
    """Forward-mode Jacobian of ``fun`` pushing ``chunk`` input directions at a time: memory of many-row
    constraints (per-point coil distances) grows with the directions pushed at once."""

    def jac(x):
        n = x.size
        basis = jnp.eye(-(-n // chunk) * chunk, n, dtype=x.dtype).reshape(-1, chunk, n)
        columns = jax.lax.map(jax.vmap(lambda v: jax.jvp(fun, (x,), (v,))[1]), basis)
        return columns.reshape(-1, columns.shape[-1])[:n].T

    return jac


def coil_constraint(rows_from_x, *, forward_chunk: int | None = None):
    """SciPy ``NonlinearConstraint`` (rows >= 0) of pure-coil rows ``rows_from_x(x)``.

    Needs no equilibrium solve or adjoint right-hand side.  The Jacobian is reverse mode, or with
    ``forward_chunk`` forward mode in chunks of that many directions (:func:`chunked_jacfwd`), for
    thousands of per-point rows.
    """
    from scipy.optimize import NonlinearConstraint

    fun = jax.jit(rows_from_x)
    jac = jax.jit(jax.jacrev(fun) if forward_chunk is None else chunked_jacfwd(fun, forward_chunk))
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))
