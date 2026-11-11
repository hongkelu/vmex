"""Coil designs with plasma coordinates for single-stage free-boundary optimization.

The coils live in ESSOS (``essos.coil_design``): its ``CoilDesign`` maps a design vector to coils and their
differentiable ``BiotSavart`` field, ``CoilLimits`` turns coil lengths, curvatures and distances into scaled
inequality rows, and ``coil_constraint`` makes pure-coil rows a SciPy ``NonlinearConstraint`` that needs no
equilibrium solve.  :class:`CoilDesign` prepends the plasma's PHIEDGE and
current profile to ESSOS' coil coordinates: it is the ``field_from_parameters`` and ``plasma_from_parameters``
of a free-boundary problem such as
:class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`.
"""

from __future__ import annotations

import dataclasses

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
    ``abs(CURTOR)``; then the coordinates of ESSOS' ``CoilDesign`` (relative
    changes of the ``current_dofs`` base-coil currents and additive changes of
    every base coil's Fourier coefficients in metres).  ``x0 = 0`` is the nominal
    design.  Calling the design gives the ``BiotSavart`` field of
    :meth:`coils_from_x`; :attr:`plasma_from_parameters` (``None`` without plasma
    coordinates) maps solver parameters to the design's PHIEDGE and current.
    ``scales`` default to ESSOS' for the coil coordinates and ``phiedge_scale`` /
    ``plasma_current_scale`` for the plasma coordinates.
    """

    def __init__(self, coils, *, current_dofs=(), scales=None, phiedge=None, phiedge_scale=0.05,
                 plasma_current=None, plasma_current_spline=False, plasma_current_scale=0.05):
        require_optional("essos", "coil designs")
        from essos.coil_design import CoilDesign as Coils

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
        self.coils = Coils(coils, current_dofs=current_dofs,
                            scales=None if scales is None else np.asarray(scales, dtype=float)[self._coil0:])
        self.current_dofs = self.coils.current_dofs
        self.size = self._coil0 + self.coils.size
        self.x0 = np.zeros(self.size)
        self.dof_names = tuple(["phiedge/nominal"] * self.nphiedge
                               + [f"plasma_current[{j}]/unit" for j in range(self.nplasma - 1)]
                               + ["curtor/nominal"] * bool(self.nplasma) + list(self.coils.dof_names))
        if scales is None:
            scales = np.r_[np.full(self.nphiedge, phiedge_scale), np.full(self.nplasma, plasma_current_scale),
                           self.coils.scales]
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
        return self.coils.base_currents_at(jnp.asarray(x)[self._coil0:])

    def coils_from_x(self, x):
        """ESSOS coils at ``x``, without changing the nominal coils."""
        return self.coils.coils_from_x(jnp.asarray(x)[self._coil0:])

    def __call__(self, x):
        """The differentiable filament field (``essos.fields.BiotSavart``) of :meth:`coils_from_x`."""
        return self.coils(jnp.asarray(x)[self._coil0:])


@jax.tree_util.register_pytree_node_class
class SummedField:
    """The sum of fields with ``B(point)`` and ``b_cyl(R, phi, Z)``, itself such a field and a pytree of them."""

    def __init__(self, *fields):
        self.fields = fields

    def B(self, point):
        return sum(field.B(point) for field in self.fields)

    def b_cyl(self, R, phi, Z):
        return tuple(sum(parts) for parts in zip(*(field.b_cyl(R, phi, Z) for field in self.fields)))

    def tree_flatten(self):
        return (self.fields,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children[0])


def circular_coil_pairs(radius, height, current, *, n_segments: int = 128):
    """ESSOS coils: for each entry, circles of ``radius`` about the Z axis at ``+height`` and ``-height`` [m].

    Both circles of a pair carry ``current`` [A] in the +phi direction, which keeps stellarator symmetry.
    Differentiable in all three arrays.
    """
    require_optional("essos", "circular coils")
    from essos.coils import Coils, Curves

    radius, height, current = (jnp.atleast_1d(jnp.asarray(v, dtype=float)) for v in (radius, height, current))
    radius, height = jnp.tile(radius, 2), jnp.concatenate((height, -height))
    # ESSOS' Fourier order 1: [constant, sin, cos]; x = radius cos t, y = radius sin t, z = height
    dofs = jnp.zeros((radius.size, 3, 3)).at[:, 0, 2].set(radius).at[:, 1, 1].set(radius).at[:, 2, 0].set(height)
    return Coils(Curves(dofs, n_segments=int(n_segments), nfp=1, stellsym=False), jnp.tile(current, 2))


class PoloidalCoilDesign:
    """A design followed by up-down pairs of circular poloidal-field coils (:func:`circular_coil_pairs`).

    ``x`` is the design's vector, then each pair's current change in units of ``current_unit`` [A] and, with
    ``geometry_scale``, each pair's radius and height changes [m].  Calling it gives the sum of the design's field
    and the pairs'; the plasma coordinates and the coils of the design are the design's own (``design.<name>(x)``
    through the matching method here), so the pairs carry no coil-limit rows.
    """

    def __init__(self, design, radius, height, current, *, current_unit=1.0e5, current_scale=0.05,
                 geometry_scale=None, n_segments=128):
        require_optional("essos", "circular coils")
        self.design, self.n = design, int(design.size)
        self.nominal = tuple(np.atleast_1d(np.asarray(v, dtype=float)) for v in (radius, height, current))
        self.npairs = self.nominal[0].size
        if any(v.shape != (self.npairs,) for v in self.nominal):
            raise ValueError("radius, height and current need one value per pair")
        self.current_unit, self.free_geometry, self.n_segments = float(current_unit), geometry_scale is not None, int(n_segments)
        added = self.npairs * (3 if self.free_geometry else 1)
        self.x0 = np.r_[design.x0, np.zeros(added)]
        self.scales = np.r_[design.scales, np.full(self.npairs, current_scale),
                            np.full(added - self.npairs, geometry_scale if self.free_geometry else 0.0)]
        self.dof_names = (*design.dof_names, *(f"pf_current[{i}]/{self.current_unit:g}A" for i in range(self.npairs)),
                          *(f"pf_{name}[{i}]/m" for name in ("radius", "height") for i in range(self.npairs)
                            if self.free_geometry))
        self.size = self.x0.size
        self.phiedge = getattr(design, "phiedge", None)
        self.plasma_from_parameters = None if getattr(design, "plasma_from_parameters", None) is None else self.plasma_params_at

    def pairs_at(self, x):
        """``(radius, height, current)`` of the pairs at ``x``."""
        radius, height, current = (jnp.asarray(v) for v in self.nominal)
        extra, k = jnp.asarray(x)[self.n:], self.npairs
        current = current + self.current_unit * extra[:k]
        if self.free_geometry:
            radius, height = radius + extra[k:2 * k], height + extra[2 * k:]
        return radius, height, current

    def poloidal_coils_from_x(self, x):
        """The pairs' ESSOS coils at ``x``."""
        return circular_coil_pairs(*self.pairs_at(x), n_segments=self.n_segments)

    def __call__(self, x):
        from essos.fields import BiotSavart

        return SummedField(self.design(jnp.asarray(x)[:self.n]), BiotSavart(self.poloidal_coils_from_x(x)))

    def coils_from_x(self, x):
        return self.design.coils_from_x(jnp.asarray(x)[:self.n])

    def base_currents_at(self, x):
        return self.design.base_currents_at(jnp.asarray(x)[:self.n])

    def phiedge_at(self, x):
        return self.design.phiedge_at(jnp.asarray(x)[:self.n])

    def plasma_params_at(self, params, x):
        return self.design.plasma_params_at(params, jnp.asarray(x)[:self.n])

