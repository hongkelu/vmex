"""Periodic field lines of a magnetic field and their return maps.

A field line that closes after ``periods`` field periods is a fixed point of
the Poincare return map; outside the last closed flux surface such lines are
the X- and O-points of the edge (divertor X-points, island chains).  Following
Giuliani et al., "Divertor topology and vacuum vessel design for stellarators"
(arXiv:2607.27127), the line is a collocation solve and its type is read from
the tangent map ``M`` of the return map (``det M = 1``):

    ``|Tr M| > 2`` hyperbolic (X-point),  ``|Tr M| < 2`` elliptic (O-point),
    ``Tr M = 2`` parabolic.

The paper's field is the coils' vacuum field.  Here the field is any batched
``field(parameters, xyz (n, 3)) -> B (n, 3)``, for example the coils plus the
plasma's virtual-casing field (:func:`exterior_field`), so the same targets
apply at finite beta and at a free boundary.

The line is parameterized by the toroidal angle, ``R(phi), Z(phi)`` with

    ``dR/dphi = R B_R / B_phi``,  ``dZ/dphi = R B_Z / B_phi``,

on equispaced nodes over ``periods`` field periods (Fourier collocation), which
needs ``B_phi != 0`` along the line.  :func:`periodic_fieldline` is traceable
and differentiable in ``parameters`` by the implicit function theorem of the
collocation residual; nothing differentiates through the Newton iterations.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

__all__ = [
    "PeriodicFieldline",
    "exterior_field",
    "fieldline_nodes",
    "manifold_crossing",
    "periodic_fieldline",
    "return_map",
    "return_map_fixed_point",
    "surface_distance",
    "surface_distances",
    "tangent_map",
    "vessel_distances",
    "trace_fieldline",
]

Field = Callable[[Any, jax.Array], jax.Array]


class PeriodicFieldline(NamedTuple):
    """A closed field line on its collocation nodes.

    ``R``, ``Z`` [m] at the toroidal angles ``phi``; ``residual`` is the largest
    collocation residual [m/rad] and ``step`` the last Newton step [m].
    """

    R: jax.Array
    Z: jax.Array
    phi: jax.Array
    residual: jax.Array
    step: jax.Array

    @property
    def xyz(self) -> jax.Array:
        """Cartesian node positions, shape ``(n, 3)``."""
        return jnp.stack((self.R * jnp.cos(self.phi), self.R * jnp.sin(self.phi), self.Z), axis=-1)


def fieldline_nodes(n: int, nfp: int, periods: int = 1, phi0: float = 0.0) -> np.ndarray:
    """The ``n`` (odd) equispaced toroidal angles of a line closing after ``periods`` field periods."""
    if n < 3 or n % 2 == 0:
        raise ValueError("n must be odd and at least 3")
    return phi0 + 2.0 * np.pi * periods / nfp * np.arange(n) / n


def _derivative_matrix(n: int, length: float) -> np.ndarray:
    """Fourier differentiation on ``n`` (odd) equispaced nodes of a period ``length``."""
    k = 2.0 * np.pi / length * np.fft.fftfreq(n, 1.0 / n)
    return np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0))


def _rhs(field: Field, parameters: Any, R, Z, phi):
    """``(dR/dphi, dZ/dphi)`` of the field line through each ``(R, phi, Z)``."""
    cos, sin = jnp.cos(phi), jnp.sin(phi)
    B = field(parameters, jnp.stack((R * cos, R * sin, Z), axis=-1))
    B_phi = cos * B[:, 1] - sin * B[:, 0]
    return R * (cos * B[:, 0] + sin * B[:, 1]) / B_phi, R * B[:, 2] / B_phi


def _rhs_jacobian(field: Field, parameters: Any, R, Z, phi):
    """The right-hand side and its 2 x 2 derivative in ``(R, Z)`` at each point.

    The field acts point by point, so one tangent along every ``R`` and one
    along every ``Z`` give all the entries.
    """
    f = lambda r, z: _rhs(field, parameters, r, z, phi)  # noqa: E731
    one, zero = jnp.ones_like(R), jnp.zeros_like(R)
    value, (a_rr, a_zr) = jax.jvp(f, (R, Z), (one, zero))
    _, (a_rz, a_zz) = jax.jvp(f, (R, Z), (zero, one))
    return value, (a_rr, a_rz, a_zr, a_zz)


def _newton_matrix(D, a):
    a_rr, a_rz, a_zr, a_zz = a
    return jnp.block([[D - jnp.diag(a_rr), -jnp.diag(a_rz)], [-jnp.diag(a_zr), D - jnp.diag(a_zz)]])


def periodic_fieldline(field: Field, parameters: Any, R, Z, *, nfp: int, periods: int = 1, phi0: float = 0.0,
                       max_iterations: int = 20, tolerance: float = 1e-11) -> PeriodicFieldline:
    """The closed field line near the guess ``R``, ``Z`` (node values, :func:`fieldline_nodes`).

    Newton's method on the collocation residual ``D u - f(u)``, at most
    ``max_iterations`` steps, until the step is below ``tolerance`` [m].  The
    result is differentiable in ``parameters``: the iterations run on frozen
    parameters and one more Newton step at the solution carries the implicit
    derivative ``du = -J^-1 dr``.  ``step`` and ``residual`` say whether the
    solve converged; the system is singular at a parabolic line (``Tr M = 2``).
    """
    R, Z = jnp.asarray(R, dtype=float), jnp.asarray(Z, dtype=float)
    n = R.size
    phi = jnp.asarray(fieldline_nodes(n, nfp, periods, phi0))
    D = jnp.asarray(_derivative_matrix(n, 2.0 * np.pi * periods / nfp))
    frozen = jax.tree.map(jax.lax.stop_gradient, parameters)

    def correction(p, u):
        (f_r, f_z), a = _rhs_jacobian(field, p, u[:n], u[n:], phi)
        r = jnp.concatenate((D @ u[:n] - f_r, D @ u[n:] - f_z))
        return r, jnp.linalg.solve(_newton_matrix(D, a), r)

    def body(carry):
        u, _, k = carry
        _, du = correction(frozen, u)
        return u - du, jnp.max(jnp.abs(du)), k + 1

    u0 = jax.lax.stop_gradient(jnp.concatenate((R, Z)))
    u, step, _ = jax.lax.while_loop(lambda c: (c[1] > tolerance) & (c[2] < max_iterations), body,
                                    (u0, jnp.asarray(jnp.inf, dtype=u0.dtype), 0))
    u = jax.lax.stop_gradient(u)
    # at the solution: the residual depends on the parameters, the matrix is frozen
    _, a = _rhs_jacobian(field, frozen, u[:n], u[n:], phi)
    f_r, f_z = _rhs(field, parameters, u[:n], u[n:], phi)
    r = jnp.concatenate((D @ u[:n] - f_r, D @ u[n:] - f_z))
    u = u - jnp.linalg.solve(jax.lax.stop_gradient(_newton_matrix(D, a)), r)
    return PeriodicFieldline(u[:n], u[n:], phi, jnp.max(jnp.abs(jax.lax.stop_gradient(r))), step)


def tangent_map(field: Field, parameters: Any, line: PeriodicFieldline, *, refine: int = 16) -> jax.Array:
    """The 2 x 2 tangent map ``M`` of the return map at ``line.phi[0]``, in ``(R, Z)``.

    ``dM/dphi = A(phi) M`` with ``A`` the derivative of the field-line
    right-hand side along the line.  ``A`` is evaluated on the nodes,
    interpolated by its Fourier series to ``refine`` (even) times as many
    points and integrated with fourth-order Runge-Kutta steps over them.
    ``Tr M`` does not depend on the coordinates or on the starting angle;
    ``det M = 1`` measures the accuracy.  Differentiable in ``parameters`` and
    in the line.
    """
    if refine < 2 or refine % 2:
        raise ValueError("refine must be even and at least 2")
    n = line.R.size
    _, a = _rhs_jacobian(field, parameters, line.R, line.Z, line.phi)
    A = jnp.stack(a, axis=-1)
    m = refine * n
    spectrum = jnp.fft.fft(A, axis=0)
    half = (n + 1) // 2
    padded = jnp.zeros((m, 4), dtype=spectrum.dtype).at[:half].set(spectrum[:half]).at[m - (n - half):].set(
        spectrum[half:])
    fine = (jnp.real(jnp.fft.ifft(padded, axis=0)) * refine).reshape(m, 2, 2)
    h = 2.0 * (line.phi[1] - line.phi[0]) * n / m
    # one Runge-Kutta step spans two fine points: (start, middle, end)
    stages = jnp.stack((fine[0::2], fine[1::2], jnp.roll(fine, -2, axis=0)[0::2]), axis=1)

    def step(M, A3):
        k1 = A3[0] @ M
        k2 = A3[1] @ (M + 0.5 * h * k1)
        k3 = A3[1] @ (M + 0.5 * h * k2)
        k4 = A3[2] @ (M + h * k3)
        return M + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    M, _ = jax.lax.scan(step, jnp.eye(2, dtype=fine.dtype), stages)
    return M


def trace_fieldline(field: Field, parameters: Any, R, Z, phi, *, substeps: int = 8):
    """Field lines from ``(R, Z)`` at ``phi[0]`` on the increasing angles ``phi``: arrays ``(len(phi), ...)``.

    Fixed fourth-order Runge-Kutta steps, ``substeps`` between consecutive
    angles; ``R`` and ``Z`` are batches of starting points (any one shape).
    """
    R, Z = jnp.asarray(R, dtype=float), jnp.asarray(Z, dtype=float)
    shape = R.shape
    phi = jnp.asarray(phi, dtype=float)

    def f(y, angle):
        d_r, d_z = _rhs(field, parameters, y[0], y[1], jnp.full(y[0].shape, angle))
        return jnp.stack((d_r, d_z))

    def interval(y, ends):
        h = (ends[1] - ends[0]) / substeps

        def rk4(y, i):
            t = ends[0] + i * h
            k1 = f(y, t)
            k2 = f(y + 0.5 * h * k1, t + 0.5 * h)
            k3 = f(y + 0.5 * h * k2, t + 0.5 * h)
            k4 = f(y + h * k3, t + h)
            return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

        y, _ = jax.lax.scan(rk4, y, jnp.arange(substeps))
        return y, y

    y0 = jnp.stack((R.ravel(), Z.ravel()))
    # checkpointed: a reverse pass keeps the points at the angles only and retraces each interval
    _, path = jax.lax.scan(jax.checkpoint(interval), y0, jnp.stack((phi[:-1], phi[1:]), axis=-1))
    path = jnp.concatenate((y0[None], path))
    return path[:, 0].reshape((phi.size,) + shape), path[:, 1].reshape((phi.size,) + shape)


def return_map(field: Field, parameters: Any, R, Z, *, nfp: int, periods: int = 1, phi0: float = 0.0,
               steps: int = 256):
    """``(R, Z)`` after ``periods`` field periods along the field lines from ``(R, Z)`` at ``phi0``."""
    ends = jnp.asarray([phi0, phi0 + 2.0 * np.pi * periods / nfp])
    R1, Z1 = trace_fieldline(field, parameters, R, Z, ends, substeps=steps)
    return R1[-1], Z1[-1]


def return_map_fixed_point(field: Field, parameters: Any, R, Z, *, nfp: int, periods: int = 1, phi0: float = 0.0,
                           steps: int = 256, iterations: int = 20):
    """Fixed points of the return map by Newton's method from each ``(R, Z)``: ``(R, Z, M, step)``.

    A shooting solve, independent of the collocation of
    :func:`periodic_fieldline`: ``M (..., 2, 2)`` is the derivative of the
    integrated map and ``step`` the last of the ``iterations`` Newton steps
    [m], large or not finite where a start did not converge.  ``det M = 1``
    where the integration resolved the line.  Traceable; the starts are a batch
    of any one shape.
    """
    R, Z = jnp.asarray(R, dtype=float), jnp.asarray(Z, dtype=float)

    def image(x):
        r, z = return_map(field, parameters, x[:1], x[1:], nfp=nfp, periods=periods, phi0=phi0, steps=steps)
        return jnp.concatenate((r, z))

    def solve(x):
        def newton(carry, _):
            x, _ = carry
            dx = jnp.linalg.solve(jax.jacfwd(image)(x) - jnp.eye(2), image(x) - x)
            return (x - dx, jnp.max(jnp.abs(dx))), None

        (x, step), _ = jax.lax.scan(newton, (x, jnp.asarray(jnp.inf, dtype=x.dtype)), None, length=iterations)
        return x, jax.jacfwd(image)(x), step

    x, M, step = jax.vmap(solve)(jnp.stack((R.ravel(), Z.ravel()), axis=-1))
    return x[:, 0].reshape(R.shape), x[:, 1].reshape(R.shape), M.reshape(R.shape + (2, 2)), step.reshape(R.shape)


def _manifold_direction(M, x0, stable: bool):
    """Unstable multiplier of the tangent map ``M`` (``Tr M > 2``) and the unit eigenvector, towards ``Z = 0``, of
    its unstable (``stable``: stable) direction at ``x0``."""
    trace = jnp.trace(M)
    unstable = 0.5 * (trace + jnp.sqrt(trace**2 - 4.0))
    multiplier = 1.0 / unstable if stable else unstable
    a, b = jnp.stack((M[0, 1], multiplier - M[0, 0])), jnp.stack((multiplier - M[1, 1], M[1, 0]))
    v = jnp.where(jnp.sum(a * a) > jnp.sum(b * b), a, b)
    return unstable, -v / jnp.linalg.norm(v) * jnp.sign(v[1]) * jnp.sign(x0[1])


def manifold_outer_leg(field: Field, parameters: Any, line: PeriodicFieldline, M, *, nfp: int, periods: int = 1,
                       iterations: int = 4, stable: bool = False, offset: float = 1e-4, steps: int = 128,
                       samples: int = 32, index=None, path: bool = False, per_map: int = 16):
    """Points ``(R, Z)``, each ``(samples + 1,)``, along a leg of an X-line that leaves it away from ``Z = 0``.

    The other branch of the manifold of :func:`manifold_crossing`: on the
    line's first plane, the images under ``iterations`` return maps
    (``stable``: backwards) of ``x0 - s v`` with ``s`` from ``offset`` to
    ``lambda^2 offset``, two images of every point of the leg.  A divertor leg
    runs on from the X-line to the wall; where the points turn back towards the
    plasma, the field outside the X-line folds the leg.  Differentiable in
    ``parameters``, the line and ``M``; ``index`` traces that one point only
    (a scalar pair), e.g. the one picked from a first call without gradients.
    ``path`` returns the whole traces instead, ``(R, Z, phi)`` with ``R`` and
    ``Z`` of shape ``(len(phi),) + s.shape``: the leg's surface from next to the
    X-line on, swept by its field lines, at ``per_map`` angles per map.
    """
    x0, phi0 = jnp.stack((line.R[0], line.Z[0])), line.phi[0]
    unstable, v = _manifold_direction(M, x0, stable)
    # per_map angles per map with steps / per_map Runge-Kutta steps between them: a reverse pass stores that much
    angles = phi0 + (-1.0 if stable else 1.0) * 2.0 * np.pi * periods / nfp * (
        jnp.arange(per_map * iterations + 1) / per_map)
    s = offset * unstable ** (2.0 * jnp.arange(samples + 1) / samples)
    s = s if index is None else s[index]
    R, Z = trace_fieldline(field, parameters, x0[0] - s * v[0], x0[1] - s * v[1], angles,
                           substeps=max(1, steps // per_map))
    return (R, Z, angles) if path else (R[-1], Z[-1])


def manifold_crossing(field: Field, parameters: Any, line: PeriodicFieldline, M, *, nfp: int, periods: int = 1,
                      iterations: int = 2, stable: bool = False, offset: float = 1e-4, steps: int = 128,
                      samples: int = 32, newton: int = 4):
    """Where a leg of an X-line crosses ``Z = 0`` on the line's first plane: ``(R, sine, miss)``.

    The leg is the branch towards ``Z = 0`` of the unstable manifold of the
    line's first node (``stable``: of the stable one, followed backwards).  It
    is the image under ``iterations`` return maps of the points
    ``x0 + s v``, ``s`` in ``[offset, lambda^2 offset]`` [m], with ``v`` the
    eigenvector of the tangent map ``M`` (``Tr M > 2``) and ``lambda`` its
    unstable multiplier; ``s`` is solved for the crossing (the first sign change
    among ``samples`` starts, then ``newton`` steps).  ``sine`` is the ``R``
    component of the leg's unit tangent there and ``miss`` the remaining
    ``|Z|`` [m]: not zero where ``iterations`` maps do not bring that range to
    the plane.  The range holds two images of every point of the leg, so a
    crossing found with ``iterations`` maps stays in range while the design
    moves it by less than a factor ``lambda`` in ``s``.

    In a stellarator-symmetric field on a symmetry plane (``line.phi[0] = 0``)
    the mirror line's stable manifold is this leg's mirror image, and the two
    meet at the crossing at the angle ``2 asin(sine)``: ``sine = 0`` joins the
    legs of the upper and lower X-lines into one separatrix, as in an
    axisymmetric double null, and its size sets the width of the stochastic
    layer around them.  Differentiable in ``parameters``, the line and ``M``;
    ``steps`` Runge-Kutta steps per field period.
    """
    x0, phi0 = jnp.stack((line.R[0], line.Z[0])), line.phi[0]
    unstable, v = _manifold_direction(M, x0, stable)
    ends = jnp.stack((phi0, phi0 + (-1.0 if stable else 1.0) * 2.0 * np.pi * periods / nfp * iterations))

    def image(p, x0, v, s):
        R, Z = trace_fieldline(field, p, x0[0] + s * v[0], x0[1] + s * v[1], ends, substeps=steps * iterations)
        return R[-1], Z[-1]

    frozen = jax.tree.map(jax.lax.stop_gradient, (parameters, x0, v))
    height = lambda s: image(*frozen, s)[1]  # noqa: E731
    starts = offset * jax.lax.stop_gradient(unstable) ** (2.0 * jnp.arange(samples + 1) / samples)
    z = height(starts)
    change = z[:-1] * z[1:] <= 0.0
    k = jnp.where(jnp.any(change), jnp.argmax(change), jnp.argmin(jnp.abs(z[:-1])))
    s = starts[k] - z[k] * (starts[k + 1] - starts[k]) / (z[k + 1] - z[k])

    def correct(s, _):
        value, slope = jax.jvp(height, (s,), (jnp.ones_like(s),))
        return jnp.clip(s - value / slope, starts[0], starts[-1]), None   # outside: the leg is not on x0 + s v

    s, _ = jax.lax.scan(correct, s, None, length=newton)
    value, slope = jax.jvp(height, (s,), (jnp.ones_like(s),))
    # the crossing moves with the parameters: one more step on the traced height, with the frozen slope
    s = jax.lax.stop_gradient(s) - image(parameters, x0, v, jax.lax.stop_gradient(s))[1] / jax.lax.stop_gradient(slope)
    (R, _), (t_r, t_z) = jax.jvp(lambda s: image(parameters, x0, v, s), (s,), (jnp.ones_like(s),))
    return R, t_r / jnp.hypot(t_r, t_z), jnp.abs(jax.lax.stop_gradient(value))


def surface_distances(xyz, gamma, nfp: int) -> jax.Array:
    """Distance [m] from each of the points ``xyz (n, 3)`` to a toroidal surface.

    ``gamma (3, nphi, ntheta)`` samples one field period of the surface (the
    layout of ``VmecSurfaceFieldData.gamma``); the points are compared with all
    ``nfp`` rotations of it.
    """
    surface = jnp.moveaxis(jnp.asarray(gamma), 0, -1).reshape(-1, 3)
    angles = 2.0 * np.pi * np.arange(nfp) / nfp
    cos, sin = jnp.cos(angles)[:, None], jnp.sin(angles)[:, None]
    # rotating the points backwards is the same as rotating the surface forwards
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rotated = jnp.stack((cos * x + sin * y, -sin * x + cos * y, jnp.broadcast_to(z, (nfp, z.size))), axis=-1)
    squared = jnp.sum((rotated[:, :, None] - surface[None, None]) ** 2, axis=-1)   # (nfp, n, surface points)
    return jnp.sqrt(jnp.min(squared, axis=(0, 2)))


def vessel_distances(xyz, outlines, nfp: int) -> jax.Array:
    """Signed distance [m] from each of the points ``xyz (n, 3)`` to a vessel in the point's poloidal plane,
    positive outside.

    ``outlines (P, m, 2)`` are the vessel's closed ``(R, Z)`` outlines on ``P`` planes equally spaced over one
    field period from ``phi = 0``, the same number of points on each in matching order; between planes the outline
    is interpolated linearly.  The in-plane distance stands for the 3-D one where the vessel varies slowly in
    ``phi``.
    """
    outlines = jnp.asarray(outlines)
    planes = outlines.shape[0]
    u = jnp.mod(jnp.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi / nfp) * nfp * planes / (2.0 * np.pi)
    k = jnp.floor(u).astype(int)
    w = (u - k)[:, None, None]
    a = (1.0 - w) * outlines[k % planes] + w * outlines[(k + 1) % planes]           # (n, m, 2)
    b = jnp.roll(a, -1, axis=1)
    p = jnp.stack((jnp.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]), axis=-1)[:, None]
    t = jnp.clip(jnp.sum((p - a) * (b - a), -1) / jnp.sum((b - a) ** 2, -1), 0.0, 1.0)
    distance = jnp.sqrt(jnp.min(jnp.sum((p - a - t[..., None] * (b - a)) ** 2, -1), axis=1))
    above = (a[..., 1] > p[..., 1]) != (b[..., 1] > p[..., 1])  # even-odd rule along +R
    dz = jnp.where(above, b[..., 1] - a[..., 1], 1.0)
    crossing = above & (p[..., 0] < a[..., 0] + (p[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) / dz)
    return jnp.where(jnp.sum(crossing, axis=1) % 2 == 1, -distance, distance)


def surface_distance(xyz, gamma, nfp: int) -> jax.Array:
    """Smallest distance [m] from the points ``xyz (n, 3)`` to a toroidal surface (:func:`surface_distances`)."""
    return jnp.min(surface_distances(xyz, gamma, nfp))


def exterior_field(external_field_from_parameters: Callable[[Any], Any], **options):
    """The total field outside a plasma boundary as ``field(parameters, xyz)``, and its parameter map.

    Returns ``(parameters_of, field)``.  ``parameters_of(surface_data, x)`` packs
    the virtual-casing sources of a (traced) ``VmecSurfaceFieldData`` -- the nodes
    of the finest quadrature level of :class:`~vmex.core.extender.VmecExtender`
    (built with ``options``: ``digits``, ``levels``) with their weighted sheet
    current and charge -- together with the external-field parameters ``x``.
    ``field(parameters, xyz)`` is ``external_field_from_parameters(x)`` plus the
    sum over those sources, the extender's ``near_surface="direct"`` field.

    The sources are built once per equilibrium, not once per field evaluation:
    building them costs as much as hundreds of evaluations at a few points, and
    a field-line trace evaluates the field thousands of times.

    The sum is the trapezoid rule on twice the grid of ``surface_data``, so it
    is accurate only away from the boundary.  Measured on a compact QA with
    minor radius 0.22 m against the extender's graded rule (relative error,
    median / largest of 64 points):

    ============  ===============  ===============  ===============
    distance      grid 64          grid 128         grid 256
    ============  ===============  ===============  ===============
    10 mm         1e-1 / 5e-1      8e-3 / 5e-2      6e-5 / 9e-4
    20 mm         8e-3 / 5e-2      6e-5 / 9e-4      4e-9 / 4e-7
    30 mm         7e-4 / 7e-3      5e-7 / 2e-5      2e-13 / 2e-10
    50 mm         5e-6 / 2e-4      3e-11 / 1e-8     round-off
    100 mm        2e-11 / 2e-8     round-off        round-off
    ============  ===============  ===============  ===============

    Closer than that, use the extender's ``near_surface="graded"`` field.
    """
    from virtual_casing_jax.derivative_kernels import layer_derivatives

    from .extender import VmecExtender, _field_cartesian

    def parameters_of(data, x):
        return VmecExtender.from_surface_data(data, **options)._direct_sources(), x

    def field(parameters, xyz):
        sources, x = parameters
        return _field_cartesian(external_field_from_parameters(x), xyz) + layer_derivatives(xyz, *sources)[0]

    return parameters_of, field
