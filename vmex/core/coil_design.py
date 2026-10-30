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
