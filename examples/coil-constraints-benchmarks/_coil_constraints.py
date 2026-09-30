"""Differentiable coil inequalities shared by both coil-constraint benchmarks.

The coil geometry is evaluated analytically from the raw, metre-valued Fourier
coefficients of the ESSOS curves, independently of the Biot-Savart quadrature.
``coil_inequalities`` returns one dimensionless row per limit of
``parameters.py``, feasible when >= 0: per-coil length, peak curvature and mean
squared curvature, the minimum coil-coil distance over all symmetry copies, a
nonlocal self-intersection guard and a regular-parametrization guard.
``surface_distance`` is the coil-to-plasma clearance to a sampled boundary.
Distances are between polygonal curves and a sampled surface, not a
finite-build winding pack.
"""
import jax
import jax.numpy as jnp
import numpy as np

import parameters as P

CURVATURE_POINTS = max(512, 64*P.COIL_ORDER)
DISTANCE_POINTS = max(128, 16*P.COIL_ORDER)
SURFACE_GRID = (61, 64)
SELF_CLEARANCE = 1e-6  # numerical nonintersection guard, metres


def geometry(raw, points):
    """Analytic Fourier coordinates, speed and curvature; raw shape (coil,xyz,mode)."""
    t = jnp.arange(points) / points
    w = 2*jnp.pi*jnp.arange(1, (raw.shape[-1]+1)//2)
    sn, cs = jnp.sin(t[:, None]*w), jnp.cos(t[:, None]*w)
    a, b = raw[:, :, 1::2], raw[:, :, 2::2]
    xyz = raw[:, None, :, 0] + jnp.einsum('tk,ijk->itj', sn, a) + jnp.einsum('tk,ijk->itj', cs, b)
    v = jnp.einsum('tk,ijk->itj', cs*w, a) - jnp.einsum('tk,ijk->itj', sn*w, b)
    acc = -jnp.einsum('tk,ijk->itj', sn*w*w, a) - jnp.einsum('tk,ijk->itj', cs*w*w, b)
    speed = jnp.sqrt(jnp.sum(v*v, axis=-1)+1e-30)
    cross = jnp.cross(v, acc)
    curvature = jnp.sqrt(jnp.sum(cross*cross, axis=-1)+1e-30)/speed**3
    return xyz, speed, curvature


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
    raw = coils.curves.curves
    _, speed, curvature = geometry(raw[:P.N_COILS], curvature_points)
    points, _, _ = geometry(raw, distance_points)
    cc, own = separations(points)
    return dict(length=jnp.mean(speed, axis=1), peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2*speed, axis=1)/jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed,axis=1))


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
    points, _, _ = geometry(coils.curves.curves, DISTANCE_POINTS)
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

