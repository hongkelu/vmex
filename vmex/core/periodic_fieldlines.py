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
    "tabulated_field",
    "tangent_map",
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
    _, path = jax.lax.scan(interval, y0, jnp.stack((phi[:-1], phi[1:]), axis=-1))
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
    trace = jnp.trace(M)
    unstable = 0.5 * (trace + jnp.sqrt(trace**2 - 4.0))
    multiplier = 1.0 / unstable if stable else unstable
    a, b = jnp.stack((M[0, 1], multiplier - M[0, 0])), jnp.stack((multiplier - M[1, 1], M[1, 0]))
    v = jnp.where(jnp.sum(a * a) > jnp.sum(b * b), a, b)
    v = -v / jnp.linalg.norm(v) * jnp.sign(v[1]) * jnp.sign(x0[1])  # towards Z = 0
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


def surface_distance(xyz, gamma, nfp: int) -> jax.Array:
    """Smallest distance [m] from the points ``xyz (n, 3)`` to a toroidal surface (:func:`surface_distances`)."""
    return jnp.min(surface_distances(xyz, gamma, nfp))


def tabulated_field(field: Field, parameters: Any, *, nfp: int, r_range, z_max: float, shape=(128, 64, 160),
                    symmetric: bool = True, batch: int = 8192, mask=None) -> Field:
    """``field`` sampled once on a grid of one field period, as a field that interpolates in it.

    For tracing many lines for many turns (Poincare plots), where evaluating
    ``field`` at every step is the cost: the cylindrical components are stored on
    ``shape = (nR, nphi, nZ)`` equispaced nodes of ``r_range`` x one field period
    x ``[-z_max, z_max]`` and interpolated by tensor-product Catmull-Rom cubics
    (continuous first derivatives, periodic in ``phi``, the edge value outside the
    box).  ``symmetric``: the field is stellarator symmetric, so only half the
    period is sampled (``nphi`` even).  ``field`` is called on ``batch`` points at
    a time.  ``mask(xyz (n, 3)) -> bool (n,)`` marks nodes where ``field`` is not
    the field wanted (inside a plasma boundary, where a virtual-casing field
    vanishes): they take the value of the nearest other node of their plane, so
    the interpolation next to the boundary sees no jump.  The interpolated field
    is not exactly divergence free and is first-order accurate within a node of
    the mask: compare some lines with ``field`` itself.
    """
    nr, nphi, nz = shape
    period = 2.0 * np.pi / nfp
    r, z = np.linspace(r_range[0], r_range[1], nr), np.linspace(-z_max, z_max, nz)
    phi = np.arange(nphi) * period / nphi
    half = nphi // 2 + 1 if symmetric else nphi
    R, P, Z = np.meshgrid(r, phi[:half], z, indexing="ij")
    xyz = np.stack((R * np.cos(P), R * np.sin(P), Z), axis=-1).reshape(-1, 3)
    evaluate = jax.jit(lambda points: field(parameters, points))
    padded = np.concatenate((xyz, np.tile(xyz[-1:], ((-len(xyz)) % batch, 1))))
    B = np.concatenate([np.asarray(evaluate(jnp.asarray(chunk))) for chunk in padded.reshape(-1, batch, 3)])[:len(xyz)]
    B = B.reshape(R.shape + (3,))
    cos, sin = np.cos(P), np.sin(P)
    table = np.empty((3, nr, nphi, nz))
    table[:, :, :half] = np.stack((cos * B[..., 0] + sin * B[..., 1], cos * B[..., 1] - sin * B[..., 0], B[..., 2]))
    if mask is not None:
        from scipy.ndimage import distance_transform_edt

        masked = np.asarray(mask(xyz)).reshape(R.shape)
        for j in np.flatnonzero(masked.any(axis=(0, 2))):
            i_r, i_z = distance_transform_edt(masked[:, j], return_indices=True)[1]
            table[:, :, j] = table[:, i_r, j, i_z]
    if symmetric:  # (B_R, B_phi, B_Z)(R, -phi, -Z) = (-B_R, B_phi, B_Z)(R, phi, Z)
        mirror = table[:, :, nphi - np.arange(half, nphi), ::-1]
        table[:, :, half:] = np.array([-1.0, 1.0, 1.0])[:, None, None, None] * mirror
    table = jnp.asarray(table)
    origin = jnp.asarray([r[0], 0.0, -z_max])
    spacing = jnp.asarray([r[1] - r[0], period / nphi, z[1] - z[0]])

    def weights(t):  # Catmull-Rom: the cubic through nodes 0 and 1 with centred slopes there
        return jnp.stack((-0.5 * t + t**2 - 0.5 * t**3, 1.0 - 2.5 * t**2 + 1.5 * t**3,
                          0.5 * t + 2.0 * t**2 - 1.5 * t**3, -0.5 * t**2 + 0.5 * t**3), axis=-1)

    def interpolated(_, xyz):
        R, angle = jnp.hypot(xyz[:, 0], xyz[:, 1]), jnp.arctan2(xyz[:, 1], xyz[:, 0])
        u = (jnp.stack((R, jnp.mod(angle, period), xyz[:, 2]), axis=-1) - origin) / spacing
        cell = jnp.floor(jax.lax.stop_gradient(u))
        w = weights(u - cell)                                           # (n, 3, 4)
        index = cell.astype(int)[:, :, None] + jnp.arange(-1, 3)        # (n, 3, 4)
        i_r, i_phi, i_z = jnp.clip(index[:, 0], 0, nr - 1), jnp.mod(index[:, 1], nphi), jnp.clip(index[:, 2], 0, nz - 1)
        nodes = table[:, i_r[:, :, None, None], i_phi[:, None, :, None], i_z[:, None, None, :]]   # (3, n, 4, 4, 4)
        b_r, b_phi, b_z = jnp.einsum("knabc,na,nb,nc->kn", nodes, w[:, 0], w[:, 1], w[:, 2])
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        return jnp.stack((cos * b_r - sin * b_phi, sin * b_r + cos * b_phi, b_z), axis=-1)

    return interpolated


def exterior_field(surface_data, external_field_from_parameters: Callable[[Any], Any], **options):
    """The total field outside a plasma boundary as ``field(parameters, xyz)``, and its parameter map.

    Returns ``(parameters_of, field)``: ``parameters_of(surface_data, x)`` packs
    the arrays of a (traced) ``VmecSurfaceFieldData`` with the external-field
    parameters ``x`` into a pytree, and ``field(parameters, xyz)`` is
    ``external_field_from_parameters(x)`` plus the plasma's virtual-casing field
    (:class:`~vmex.core.extender.VmecExtender`, built with ``options``:
    ``digits``, ``near_surface``, ``graded_nodes``, ...).  ``surface_data`` is a
    template: it fixes the grid, field periods and symmetry.
    """
    from dataclasses import replace

    from .extender import VmecExtender

    names = ("gamma", "B_total", "normal", "area_vector")
    options = {"accuracy_check": "off", **options}

    def parameters_of(data, x):
        return tuple(getattr(data, name) for name in names), x

    def field(parameters, xyz):
        arrays, x = parameters
        data = replace(surface_data, **dict(zip(names, arrays)))
        return VmecExtender.from_surface_data(
            data, external_field=external_field_from_parameters(x), **options).B(xyz)

    return parameters_of, field
