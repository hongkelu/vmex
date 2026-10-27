"""Coil and vacuum targets of a single-stage optimization, ready for ``from_tuples``.

Each function returns a callable in the shape the optimization problems take:

- :func:`coil_constraint` — engineering limits of ESSOS coils (length, curvature, mean-squared curvature, coil-coil
  and self distance), a ``scipy.optimize.NonlinearConstraint`` on the design vector alone (no equilibrium solve),
  for ``design_constraints`` of
  :meth:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem.from_tuples`;
  :func:`coil_slacks` are its rows and :func:`coil_metrics` the raw values.
- :func:`coil_plasma_distance` — the distance between the continuous coils and the continuous plasma boundary,
  ``f(state, runtime, x)``, a ``parameter_constraints`` entry.
- :func:`vacuum_iota` / :func:`min_vacuum_iota` — the rotational transform of the plasma boundary without pressure
  or current, ``f(state, runtime)``, a ``constraints`` entry or an objective term.

Coil functions need ``vmex[coils]`` (ESSOS); the coil-plasma distance and the vacuum transform are differentiable
in both the coils and the equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

__all__ = [
    "CoilLimits",
    "coil_constraint",
    "coil_metrics",
    "coil_plasma_distance",
    "coil_slacks",
    "continuous_surface_distance",
    "min_vacuum_iota",
    "vacuum_iota",
]


@dataclass(frozen=True)
class CoilLimits:
    """Engineering limits of every base coil and its symmetry copies.

    ``length`` [m] of each coil, peak ``curvature`` [1/m] along it, its mean-squared curvature ``msc`` [1/m^2], and
    the smallest ``coil_distance`` [m] between coils (symmetry copies included); each is held with its margin
    inside the limit.  ``self_clearance`` [m] keeps a coil from crossing itself and ``min_speed`` keeps its
    parametrization regular.  ``curvature_points`` and ``distance_points`` are the quadrature points per coil.
    """

    length: float
    curvature: float
    msc: float
    coil_distance: float
    length_margin: float = 1e-5
    curvature_margin: float = 0.10
    msc_margin: float = 0.02
    distance_margin: float = 1e-3
    self_clearance: float = 1e-6
    min_speed: float = 1e-4
    curvature_points: int = 512
    distance_points: int = 128


def resampled_curves(coils, points: int, *, symmetric: bool = True):
    """ESSOS curves of ``coils`` on ``points`` quadrature points; the base curves only unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


def _segment_distances(a, b):
    """All distances between the segments of two closed polygons ``a``, ``b`` (points, 3), interiors included."""
    import jax.numpy as jnp

    u, v = jnp.roll(a, -1, axis=0) - a, jnp.roll(b, -1, axis=0) - b
    w = a[:, None] - b[None, :]
    aa, bb = jnp.sum(u * u, axis=-1)[:, None], jnp.sum(v * v, axis=-1)[None, :]
    uv = jnp.einsum("ik,jk->ij", u, v)
    uw, vw = jnp.sum(u[:, None] * w, axis=-1), jnp.sum(v[None, :] * w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa * bb - uv * uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv * vw - bb * uw) / safe, (aa * vw - uv * uw) / safe

    def distance(s, t):
        q = w + s[..., None] * u[:, None] - t[..., None] * v[None, :]
        return jnp.sum(q * q, axis=-1)

    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw / bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw + uv) / bb, 0, 1)),
                  distance(jnp.clip(-uw / aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv - uw) / aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), distance(s, t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0) + 1e-30)


def _nonadjacent(n: int):
    import jax.numpy as jnp

    diff = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    return jnp.minimum(diff, n - diff) > 1


def _separations(points):
    """Smallest coil-coil and non-adjacent self-segment distance of ``points`` (coils, samples, 3)."""
    import jax
    import jax.numpy as jnp

    pairs = jnp.asarray([(i, j) for i in range(len(points)) for j in range(i + 1, len(points))])
    inter = jax.lax.map(lambda ij: jnp.min(_segment_distances(points[ij[0]], points[ij[1]])), pairs)
    nonadjacent = _nonadjacent(points.shape[1])
    own = jax.lax.map(lambda p: jnp.min(jnp.where(nonadjacent, _segment_distances(p, p), jnp.inf)), points)
    return jnp.min(inter), jnp.min(own)


def coil_metrics(coils, limits: CoilLimits | None = None) -> dict:
    """Length, peak curvature, mean-squared curvature and smallest speed of each base coil; the smallest
    coil-coil and self distance over all symmetry copies."""
    import jax.numpy as jnp

    limits = limits or CoilLimits(1.0, 1.0, 1.0, 1.0)
    base = resampled_curves(coils, limits.curvature_points, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    cc, own = _separations(resampled_curves(coils, limits.distance_points).gamma)
    return dict(length=base.length, peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2 * speed, axis=1) / jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed, axis=1))


def coil_slacks(coils, limits: CoilLimits, *, per_point: bool = False):
    """The rows of :func:`coil_constraint`: each limit minus its value and margin, relative to the limit (>= 0 is
    feasible).  ``per_point`` gives the curvature and the distances a row per sample point instead of each coil's
    extreme: each row is then smooth where the extreme switches points, and their minimum is the extreme's row."""
    import jax.numpy as jnp

    L = limits
    if not per_point:
        m = coil_metrics(coils, L)
        return jnp.concatenate(((L.length - L.length_margin - m["length"]) / L.length,
                                (L.curvature - L.curvature_margin - m["peak"]) / L.curvature,
                                (L.msc - L.msc_margin - m["msc"]) / L.msc,
                                jnp.atleast_1d((m["coil_distance"] - L.coil_distance - L.distance_margin)
                                               / L.coil_distance),
                                jnp.atleast_1d((m["self_distance"] - L.self_clearance) / L.coil_distance),
                                (m["min_speed"] - L.min_speed) / L.length))
    base = resampled_curves(coils, L.curvature_points, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    points = resampled_curves(coils, L.distance_points).gamma  # base coils first, then their symmetry copies
    nonadjacent = _nonadjacent(points.shape[1])

    def per_coil(b):
        others = jnp.stack([jnp.min(_segment_distances(points[b], points[j]), axis=1)
                            for j in range(points.shape[0]) if j != b])
        own = jnp.min(jnp.where(nonadjacent, _segment_distances(points[b], points[b]), jnp.inf), axis=1)
        return jnp.min(others, axis=0), own

    coil_distance, self_distance = map(jnp.concatenate, zip(*[per_coil(b) for b in range(curvature.shape[0])]))
    msc = jnp.sum(curvature**2 * speed, axis=1) / jnp.sum(speed, axis=1)
    return jnp.concatenate(((L.length - L.length_margin - base.length) / L.length,
                            jnp.ravel(L.curvature - L.curvature_margin - curvature) / L.curvature,
                            (L.msc - L.msc_margin - msc) / L.msc,
                            (coil_distance - L.coil_distance - L.distance_margin) / L.coil_distance,
                            (self_distance - L.self_clearance) / L.coil_distance,
                            (jnp.min(speed, axis=1) - L.min_speed) / L.length))


def _chunked_jacfwd(fun, chunk: int):
    """Forward-mode Jacobian of ``fun`` with ``chunk`` input directions at a time (the per-point distance kernels'
    memory grows with the directions pushed at once)."""
    import jax
    import jax.numpy as jnp

    def jac(x):
        n = x.size
        basis = jnp.eye(-(-n // chunk) * chunk, n, dtype=x.dtype).reshape(-1, chunk, n)
        columns = jax.lax.map(jax.vmap(lambda v: jax.jvp(fun, (x,), (v,))[1]), basis)
        return columns.reshape(-1, columns.shape[-1])[:n].T

    return jac


def coil_constraint(coils_from_x: Callable[[Any], Any], limits: CoilLimits, *, per_point: bool = False,
                    chunk: int = 32):
    """``scipy.optimize.NonlinearConstraint`` ``coil_slacks(coils_from_x(x)) >= 0`` on the design vector, with its
    JAX Jacobian (reverse mode for the per-coil rows, forward mode in ``chunk`` directions for the per-point ones)."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import NonlinearConstraint

    fun = jax.jit(lambda x: coil_slacks(coils_from_x(x), limits, per_point=per_point))
    jac = jax.jit(_chunked_jacfwd(fun, chunk) if per_point else jax.jacrev(fun))
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))


def continuous_surface_distance(points, rbc, zbs, nfp: int, *, grid=(48, 48), iterations: int = 10):
    """Distance between continuous coils and the continuous surface ``(rbc, zbs)``, and the runner-up gap.

    ``points`` (coils, samples, 3) are samples uniform in each closed curve's parameter; the curve is their periodic
    (trigonometric) interpolant, exact for a Fourier curve sampled above its Nyquist rate.  ``rbc``/``zbs``
    ``(2 ntor + 1, mpol)`` give ``R + i Z = sum (rbc cos + i zbs sin)(m theta - n nfp phi)``.  A grid search over the
    torus finds each sample's nearest surface point; from the seven best samples Newton on ``(s, theta, phi)``
    refines the closest pair of curve and surface, held fixed for differentiation (the envelope theorem).  The value
    is continuous in the coils and the boundary, with a kink where the closest pair switches.  Returns ``(distance,
    gap)``: ``gap`` is how much farther the nearest sample more than two samples away is (a small ``gap`` flags a
    switch nearby)."""
    import jax
    import jax.numpy as jnp

    ntor = (rbc.shape[0] - 1) // 2
    m = jnp.arange(rbc.shape[1])
    n = (jnp.arange(rbc.shape[0]) - ntor) * nfp

    def surface(theta, phi):
        angle = m[None, :] * theta - n[:, None] * phi
        R = jnp.sum(rbc * jnp.cos(angle))
        Z = jnp.sum(zbs * jnp.sin(angle))
        return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z])

    th = jnp.linspace(0, 2 * jnp.pi, grid[0], endpoint=False)
    ph = jnp.linspace(0, 2 * jnp.pi, grid[1] * nfp, endpoint=False)
    TH, PH = jnp.meshgrid(th, ph, indexing="ij")
    S = jax.vmap(jax.vmap(surface))(TH, PH).reshape(-1, 3)
    pts = jnp.asarray(points)
    ncurve, nsample = pts.shape[0], pts.shape[1]
    flat = pts.reshape(-1, 3)
    d2 = jax.lax.map(lambda q: jnp.min(jnp.sum((q - S) ** 2, axis=1)), flat)
    nearest = jax.lax.map(lambda q: jnp.argmin(jnp.sum((q - S) ** 2, axis=1)), jax.lax.stop_gradient(flat))
    order = jnp.argsort(jax.lax.stop_gradient(d2))[:7]  # the coarse search's best candidates, refined below
    coef = jnp.fft.fft(pts, axis=1) / nsample  # (curves, wavenumbers, 3): the trigonometric interpolant
    wave = jnp.fft.fftfreq(nsample, 1.0 / nsample)
    if nsample % 2 == 0:  # the Nyquist term split evenly between +-N/2 keeps the interpolant real and smooth
        coef = coef.at[:, nsample // 2].multiply(0.5)
        coef = jnp.concatenate([coef, coef[:, nsample // 2:nsample // 2 + 1]], axis=1)
        wave = jnp.concatenate([wave, -wave[nsample // 2:nsample // 2 + 1]])

    def curve(c, u, coef=coef):
        """Point of curve ``c`` at parameter ``u`` (sample units: u = i is sample i)."""
        return jnp.real(jnp.sum(coef[c] * jnp.exp(2j * jnp.pi * wave * u / nsample)[:, None], axis=0))

    def pair(j):
        """The coil-surface distance refined by Newton from sample ``j`` and its nearest grid point."""
        c = j // nsample
        z0 = jnp.array([(j % nsample).astype(float), TH.ravel()[nearest[j]], PH.ravel()[nearest[j]]])
        fixed = jax.lax.stop_gradient(coef)
        half = lambda z: 0.5 * jnp.sum((curve(c, z[0], fixed) - surface(z[1], z[2])) ** 2)  # noqa: E731
        z = jax.lax.stop_gradient(z0)
        for _ in range(iterations):
            # Newton on |C - S|^2 / 2 (Gauss-Newton fails when the distance is comparable to a radius of
            # curvature); a gradient step where the Hessian is not positive definite
            g, H = jax.grad(half)(z), jax.hessian(half)(z)
            newton = -jnp.linalg.solve(H + 1e-14 * jnp.eye(3), g)
            ok = (jnp.linalg.eigvalsh(H)[0] > 0) & jnp.all(jnp.isfinite(newton))
            z = z + jnp.where(ok, newton, -g / jnp.maximum(jnp.linalg.norm(H), 1e-300))
        z = jax.lax.stop_gradient(z)
        return jnp.sqrt(jnp.sum((curve(c, z[0]) - surface(z[1], z[2])) ** 2) + 1e-30)

    cand = jnp.stack([pair(j) for j in order])
    best = jnp.argmin(jax.lax.stop_gradient(cand))
    distance = cand[best]
    k = order[best]
    c, i = k // nsample, k % nsample
    sample = jnp.arange(ncurve * nsample)
    near = (sample // nsample == c) & (jnp.abs(((sample % nsample) - i + nsample // 2) % nsample - nsample // 2) <= 2)
    second = jnp.sqrt(jnp.min(jnp.where(near, jnp.inf, jax.lax.stop_gradient(d2))))
    return distance, second - jax.lax.stop_gradient(distance)


def _lcfs_arrays(state, runtime):
    """``(rbc, zbs)`` ``(2 ntor + 1, mpol)`` of a state's plasma boundary (traceable)."""
    import jax.numpy as jnp

    from . import implicit as im

    rmnc, _, _, zmns = im._edge_physical(state, runtime)
    ntor = int(np.max(np.abs(np.asarray(runtime.modes.n))))
    mpol = int(np.max(np.asarray(runtime.modes.m))) + 1
    rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
    rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
    zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
    return rbc, zbs


def coil_plasma_distance(coils_from_x: Callable[[Any], Any], nfp: int, *, distance_points: int = 128,
                         grid=(48, 48)):
    """``f(state, runtime, x)``: the distance between the coils of ``x`` and the plasma boundary of the state
    (:func:`continuous_surface_distance`), a ``parameter_constraints`` entry."""

    def distance(state, runtime, x):
        points = resampled_curves(coils_from_x(x), distance_points).gamma
        return continuous_surface_distance(points, *_lcfs_arrays(state, runtime), int(nfp), grid=grid)[0]

    return distance


def vacuum_iota(inp, *, device=None, m1_complement="gauged"):
    """``f(state, runtime)``: the vacuum ``|iota|`` of the state's plasma boundary on the axis, the half-mesh
    surfaces and the edge.

    The fixed-boundary equilibrium of that boundary without pressure or current (``inp`` otherwise, hot-restarted
    from its previous solve), differentiable through its implicit adjoint; the axis value is
    :func:`~vmex.core.optimize.axis_iota`, the edge one extrapolated from the last two half-mesh surfaces.
    ``m1_complement`` as :attr:`~vmex.core.freeboundary_three_terms.ThreeTermsConfig.m1_complement`: with upstream's
    ``"frozen"`` the vacuum root depends on the hot-restart state (a restored checkpoint's vacuum iota moved by
    5e-7)."""
    import jax
    import jax.numpy as jnp

    from . import implicit as im
    from . import optimize as opt
    from .statephysics import _iotas_half

    vacuum = replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                     ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None, lfreeb=False)
    device = jax.devices()[0] if device is None else device
    cfg = im.make_config(vacuum, multigrid=True, hot_restart=True, device=device, m1_complement=m1_complement)
    base = im.params_from_input(vacuum, device=device)
    im.runtime_from_params(base, cfg)  # its setup, built here: run_setup cannot be traced inside a jit

    def iota(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)
        rmnc, zmns = jax.device_put((rmnc, zmns), device)  # a fixed-boundary state may live on the host
        rows, cols = np.asarray(runtime.modes.n) + int(inp.ntor), np.asarray(runtime.modes.m)
        params = replace(base, rbc=jnp.zeros_like(base.rbc).at[rows, cols].set(rmnc),
                         zbs=jnp.zeros_like(base.zbs).at[rows, cols].set(zmns))
        vstate = im.solve_implicit(params, cfg)
        vrt = im.runtime_from_params(params, cfg)
        half = _iotas_half(vstate, vrt)[1:]
        return jnp.abs(jnp.concatenate([opt.axis_iota(vstate, vrt)[None], half, 1.5 * half[-1:] - 0.5 * half[-2:-1]]))

    return iota


def min_vacuum_iota(inp, *, device=None):
    """``f(state, runtime)``: the smallest of :func:`vacuum_iota` (a vacuum-transform floor)."""
    import jax.numpy as jnp

    profile = vacuum_iota(inp, device=device)
    return lambda state, runtime: jnp.min(profile(state, runtime))
