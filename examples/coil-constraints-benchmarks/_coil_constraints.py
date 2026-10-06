"""Coil chart, coil inequalities and independent endpoint checks of the benchmarks.

:class:`CoilChart` maps the design vector to ESSOS coils and to the
differentiable filament field that ``FreeBoundaryProblem`` solves in. Coil
length, curvature and curve points come from ESSOS ``Curves``, resampled at
the constraint resolution independently of the field quadrature. Only what
ESSOS lacks is computed here: mean squared curvature, the minimum coil-coil
and nonadjacent self-segment distances (ESSOS has hinge penalties, not hard
rows) and the coil-surface clearance. All inequality rows are dimensionless
and favourable-positive. Distance constraints are on polygonal curves / a
sampled moving surface; verification refines both and reports that scope,
never a winding-pack claim.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import parameters as P

CURVATURE_POINTS = max(512, 64*P.COIL_ORDER)
DISTANCE_POINTS = max(128, 16*P.COIL_ORDER)
SURFACE_GRID = (61, 64)
SELF_CLEARANCE = 1e-6  # numerical nonintersection guard, metres


@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Biot-Savart field of filament coils, as a differentiable pytree.

    ESSOS' ``BiotSavart`` jit-compiles its methods with the coil object as a
    static ``self``, so every new coil geometry recompiles and no derivative
    flows back to the coil arrays. This pytree carries the arrays as leaves
    instead: one compiled program serves every trial and JAX differentiates
    the field with respect to the coil parameters. The field is the mean over
    each coil's quadrature points of the filament Biot-Savart integrand.
    ``gamma``/``gamma_dash`` are ESSOS' points and tangents, shape
    ``(coils, points, 3)`` in metres; ``currents`` are in amperes.
    """

    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r, phi, z):
        """Evaluate the filament field in cylindrical coordinates, in tesla."""
        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        displacement = xyz[..., None, None, :] - jnp.asarray(self.gamma)
        radius2 = jnp.sum(displacement * displacement, axis=-1)
        inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
        differential = jnp.cross(jnp.asarray(self.gamma_dash), displacement)
        differential = differential * inv_radius3[..., None]
        current_shape = (1,) * (xyz.ndim - 1) + (-1, 1, 1)
        weighted = differential * jnp.reshape(jnp.asarray(self.currents), current_shape)
        bxyz = 1.0e-7 * jnp.mean(jnp.sum(weighted, axis=-3), axis=-2)
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])


class CoilChart:
    """Design vector -> ESSOS coils and plasma parameters: the free arm's ``field_from_parameters``.

    ``x`` holds, in order: with ``phiedge`` (the deck's PHIEDGE in Wb) a relative
    PHIEDGE change, PHIEDGE = phiedge * (1 + x[0]); with ``plasma_current`` (the
    deck's leading ``k`` AC coefficients, or AC_AUX_F spline values when
    ``plasma_current_spline``, then CURTOR in A; ``ncurr = 1``) ``k + 1``
    additive changes of those values in units of the largest nominal shape value
    and of ``abs(CURTOR)``; relative changes of the selected base-coil currents
    (``current[i] = nominal[i] * (1 + x[k])``); and additive changes of every
    base coil's Cartesian Fourier coefficients in metres. ``x0 = 0`` is the
    nominal state. Calling the chart gives the :class:`DirectCoilField` of
    ``coils_from_x(x)``; ``plasma_from_parameters`` (``None`` without plasma
    coordinates) is ``FreeBoundaryProblem``'s map of the plasma parameters.
    ``scales`` default to 0.06 per relative current, 0.002 m per constant Fourier
    coefficient, ``0.002 / k**2`` m per coefficient of order ``k``, and
    ``phiedge_scale`` / ``plasma_current_scale`` for the plasma coordinates.
    """

    def __init__(self, coils, *, current_dofs, scales=None, phiedge=None, phiedge_scale=0.05, plasma_current=None,
                 plasma_current_spline=False, plasma_current_scale=0.05):
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

    @classmethod
    def from_coils(cls, coils, **kwargs):
        """Same as the constructor (the name of VMEX's former ``CoilParameters.from_coils``)."""
        return cls(coils, **kwargs)

    def check_input(self, inp):
        """Raise unless the chart's nominal PHIEDGE and plasma current are the deck's."""
        if self.phiedge is not None and not np.isclose(self.phiedge, float(inp.phiedge), rtol=1e-12, atol=0.0):
            raise ValueError("the chart's nominal PHIEDGE must equal the input's")
        if self.plasma_current is not None:
            source = inp.ac_aux_f if self.plasma_current_spline else inp.ac
            deck = np.r_[np.asarray(source, dtype=float)[: self.nplasma - 1], float(inp.curtor)]
            if int(inp.ncurr) != 1 or not np.allclose(deck, self.plasma_current, rtol=1e-12, atol=0.0):
                raise ValueError("the chart's nominal plasma current must equal the input's (ncurr = 1)")

    def phiedge_at(self, x):
        """PHIEDGE [Wb] at ``x``; only for a chart built with ``phiedge``."""
        if self.phiedge is None:
            raise ValueError("this chart does not vary PHIEDGE")
        return self.phiedge * (1.0 + jnp.asarray(x)[0])

    def plasma_params_at(self, params, x):
        """``params`` (``ImplicitParams``) with the PHIEDGE and current profile of ``x``."""
        from dataclasses import replace

        x = jnp.asarray(x)
        if self.phiedge is not None:
            params = replace(params, phiedge=self.phiedge_at(x))
        if self.plasma_current is not None:
            values = jnp.asarray(self.plasma_current) + x[self.nphiedge:self._coil0] * self.plasma_current_units
            name = "ac_aux_f" if self.plasma_current_spline else "ac"
            profile = getattr(params, name).at[: self.nplasma - 1].set(values[:-1])
            params = replace(params, curtor=values[-1], **{name: profile})
        return params

    def base_currents_at(self, x):
        """Physical base-coil currents in amperes."""
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
        """The differentiable filament field of ``coils_from_x(x)``."""
        coils = self.coils_from_x(x)
        return DirectCoilField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))


def resampled(coils, points, *, symmetric=True):
    """ESSOS curves of ``coils`` on ``points`` quadrature points; base curves only unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


def segment_distances(a, b):
    """All distances between two closed polygons' segments, including interiors."""
    u, v = jnp.roll(a, -1, axis=0)-a, jnp.roll(b, -1, axis=0)-b
    w = a[:, None]-b[None, :]
    aa, bb = jnp.sum(u*u, axis=-1)[:, None], jnp.sum(v*v, axis=-1)[None, :]
    uv = jnp.einsum('ik,jk->ij', u, v)
    uw, vw = jnp.sum(u[:, None]*w, axis=-1), jnp.sum(v[None, :]*w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa*bb-uv*uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv*vw-bb*uw)/safe, (aa*vw-uv*uw)/safe
    def distance(s, t):
        q = w+s[..., None]*u[:, None]-t[..., None]*v[None, :]
        return jnp.sum(q*q, axis=-1)
    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw/bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw+uv)/bb, 0, 1)),
                  distance(jnp.clip(-uw/aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv-uw)/aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1), distance(s,t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0)+1e-30)


def separations(points):
    """Minimum intercoil and nonadjacent self-segment distance, all symmetry copies."""
    pairs = jnp.asarray([(i,j) for i in range(len(points)) for j in range(i+1,len(points))])
    inter = jax.lax.map(lambda ij: jnp.min(segment_distances(points[ij[0]], points[ij[1]])), pairs)
    n = points.shape[1]
    diff = jnp.abs(jnp.arange(n)[:,None]-jnp.arange(n)[None,:])
    nonadjacent = jnp.minimum(diff, n-diff)>1
    own = jax.lax.map(lambda p: jnp.min(jnp.where(nonadjacent, segment_distances(p,p), jnp.inf)), points)
    return jnp.min(inter), jnp.min(own)


def coil_metrics(coils, *, curvature_points=CURVATURE_POINTS, distance_points=DISTANCE_POINTS):
    """ESSOS length, peak curvature and speed of each base coil, plus what ESSOS lacks."""
    base = resampled(coils, curvature_points, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    cc, own = separations(resampled(coils, distance_points).gamma)
    return dict(length=base.length, peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2*speed, axis=1)/jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed, axis=1))


def coil_inequalities(coils):
    m = coil_metrics(coils)
    return jnp.concatenate(((P.LENGTH_LIMIT-P.LENGTH_MARGIN-m['length'])/P.LENGTH_LIMIT,
        (P.CURVATURE_LIMIT-P.CURVATURE_MARGIN-m['peak'])/P.CURVATURE_LIMIT,
        (P.MSC_LIMIT-P.MSC_MARGIN-m['msc'])/P.MSC_LIMIT,
        jnp.atleast_1d((m['coil_distance']-P.COIL_DISTANCE_LIMIT-P.DISTANCE_MARGIN)/P.COIL_DISTANCE_LIMIT),
        jnp.atleast_1d((m['self_distance']-SELF_CLEARANCE)/P.COIL_DISTANCE_LIMIT),
        (m['min_speed']-1e-4)/P.LENGTH_LIMIT))


def surface_distance(coils, surface):
    """Sampled coil-to-moving-surface clearance; JAX differentiates both sides."""
    points = resampled(coils, DISTANCE_POINTS).gamma
    targets = surface.gamma.reshape(-1,3)
    # Mapping bounds memory and preserves exact differentiation of the active min.
    return jnp.min(jax.lax.map(lambda p: jnp.sqrt(jnp.min(jnp.sum((p-targets)**2,axis=1))+1e-30),
                               points.reshape(-1,3)))


def constraint(coils_from_x):
    """Pure-coil rows require no equilibrium solves or adjoint right-hand sides."""
    from scipy.optimize import NonlinearConstraint
    fun = jax.jit(lambda x: coil_inequalities(coils_from_x(x)))
    jac = jax.jit(jax.jacrev(fun))
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))

