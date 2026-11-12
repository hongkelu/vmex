"""Periodic field lines and their tangent maps on analytic divergence-free fields.

``B = (B0 R0 / R) e_phi + curl(psi e_phi / R)`` with ``psi(R, Z, phi)``: where
``psi`` has a saddle (extremum) in ``(R, Z)`` for all ``phi`` the field has a
closed X-line (O-line) whose tangent map is known in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from vmex.core import periodic_fieldlines as pf  # noqa: E402

NFP, R0, B0, RX, ZX, C = 2, 1.0, 1.0, 1.2, -0.3, 0.5
PERIOD = 2.0 * np.pi / NFP


def _field(sign):
    """psi = C/2 (x^2 + sign y^2) + eps (x cos + y sin + x y cos / 2)(nfp phi), x = R - RX, y = Z - ZX."""
    def psi(point, eps):
        x, y, angle = point[0] - RX, point[1] - ZX, NFP * point[2]
        return 0.5 * C * (x**2 + sign * y**2) + eps * (
            x * jnp.cos(angle) + y * jnp.sin(angle) + 0.5 * x * y * jnp.cos(angle))

    def one(eps, xyz):
        R, phi = jnp.hypot(xyz[0], xyz[1]), jnp.arctan2(xyz[1], xyz[0])
        d_r, d_z, _ = jax.grad(psi)(jnp.stack((R, xyz[2], phi)), eps)
        b_r, b_phi, b_z = -d_z / R, B0 * R0 / R, d_r / R
        return jnp.stack((b_r * jnp.cos(phi) - b_phi * jnp.sin(phi), b_r * jnp.sin(phi) + b_phi * jnp.cos(phi), b_z))

    return lambda eps, xyz: jax.vmap(lambda p: one(eps, p))(xyz)


def _guess(n, shift=0.0):
    return np.full(n, RX + shift), np.full(n, ZX - shift)


@pytest.mark.parametrize("sign, trace", [
    (-1.0, 2.0 * np.cosh(C * RX / (B0 * R0) * PERIOD)),   # saddle: X-line
    (1.0, 2.0 * np.cos(C * RX / (B0 * R0) * PERIOD)),     # minimum: O-line
])
def test_axisymmetric_line_and_trace_are_the_closed_forms(sign, trace):
    field = _field(sign)
    line = pf.periodic_fieldline(field, 0.0, *_guess(9, 0.02), nfp=NFP)
    assert float(line.step) < 1e-11 and float(line.residual) < 1e-11
    np.testing.assert_allclose(line.R, RX, atol=1e-11)
    np.testing.assert_allclose(line.Z, ZX, atol=1e-11)
    M = pf.tangent_map(field, 0.0, line, refine=64)
    assert float(jnp.trace(M)) == pytest.approx(trace, rel=1e-8)
    assert float(jnp.linalg.det(M)) == pytest.approx(1.0, abs=1e-8)


def test_collocation_agrees_with_shooting_on_a_three_dimensional_field():
    field, eps = _field(-1.0), 0.02
    line = pf.periodic_fieldline(field, eps, *_guess(33), nfp=NFP)
    assert float(line.step) < 1e-11 and float(line.residual) < 1e-10
    assert float(np.ptp(np.asarray(line.R))) > 1e-3  # the line is not a circle
    M = pf.tangent_map(field, eps, line)
    r, z, M_shot, step = pf.return_map_fixed_point(field, eps, RX, ZX, nfp=NFP, steps=400)
    assert float(step) < 1e-12
    np.testing.assert_allclose([line.R[0], line.Z[0]], [r, z], atol=1e-9)
    assert float(jnp.trace(M)) == pytest.approx(float(jnp.trace(M_shot)), rel=1e-7)
    assert float(jnp.linalg.det(M)) == pytest.approx(1.0, abs=1e-7)
    # the line traced from its first node returns to the nodes
    R, Z = pf.trace_fieldline(field, eps, line.R[0], line.Z[0], np.r_[np.asarray(line.phi), line.phi[0] + PERIOD],
                              substeps=40)
    np.testing.assert_allclose(R[:-1], line.R, atol=1e-8)
    np.testing.assert_allclose([R[-1], Z[-1]], [line.R[0], line.Z[0]], atol=1e-8)
    # twice around is the same line with the squared map
    R2, Z2 = np.tile(np.asarray(line.R), 2)[:-1], np.tile(np.asarray(line.Z), 2)[:-1]
    twice = pf.periodic_fieldline(field, eps, np.interp(pf.fieldline_nodes(65, NFP, 2), np.r_[line.phi, line.phi + PERIOD][:-1], R2),
                                  np.interp(pf.fieldline_nodes(65, NFP, 2), np.r_[line.phi, line.phi + PERIOD][:-1], Z2),
                                  nfp=NFP, periods=2)
    trace = float(jnp.trace(M))
    assert float(jnp.trace(pf.tangent_map(field, eps, twice))) == pytest.approx(trace**2 - 2.0, rel=1e-6)


def test_parameter_derivatives_are_the_finite_differences_in_both_modes():
    field = _field(-1.0)
    guess = _guess(21)

    def outputs(eps):
        line = pf.periodic_fieldline(field, eps, *guess, nfp=NFP)
        return jnp.stack((jnp.trace(pf.tangent_map(field, eps, line)), line.R[0], line.Z[3]))

    eps, h = 0.02, 1e-5
    fd = (outputs(eps + h) - outputs(eps - h)) / (2.0 * h)
    assert np.all(np.abs(np.asarray(fd)) > 1e-3)
    np.testing.assert_allclose(jax.jacfwd(outputs)(eps), fd, rtol=1e-6)
    np.testing.assert_allclose(jax.jit(jax.jacrev(outputs))(eps), fd, rtol=1e-6)


def test_surface_distance_to_a_torus_uses_every_field_period():
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                             np.linspace(0, PERIOD, 64, endpoint=False), indexing="ij")
    R = 1.0 + 0.2 * np.cos(theta)
    gamma = np.stack((R * np.cos(phi), R * np.sin(phi), 0.2 * np.sin(theta))).transpose(0, 2, 1)
    angle = 0.5 * PERIOD + PERIOD  # in the second field period
    points = jnp.asarray([[1.5 * np.cos(angle), 1.5 * np.sin(angle), 0.0], [3.0, 0.0, 0.0]])
    assert float(pf.surface_distance(points, gamma, NFP)) == pytest.approx(0.3, abs=2e-3)
    np.testing.assert_allclose(pf.surface_distances(points, gamma, NFP), [0.3, 1.8], atol=2e-3)
    gradient = jax.grad(lambda p: pf.surface_distance(p, gamma, NFP))(points)
    np.testing.assert_allclose(gradient[0], [np.cos(angle), np.sin(angle), 0.0], atol=0.05)


def _double_null(eps, xyz):
    """psi = a x^2 / 2 + b Z^2 / 2 - c Z^4 / 4 + eps (Z sin + x cos)(nfp phi): X-lines at Z = +-sqrt(b / c), symmetric."""
    def psi(point, eps):
        x, z, angle = point[0] - RX, point[1], NFP * point[2]
        return 0.25 * x**2 + 0.15 * z**2 - 0.15 * z**4 + eps * (z * jnp.sin(angle) + x * jnp.cos(angle))

    def one(p):
        R, phi = jnp.hypot(p[0], p[1]), jnp.arctan2(p[1], p[0])
        d_r, d_z, _ = jax.grad(psi)(jnp.stack((R, p[2], phi)), eps)
        b_r, b_phi, b_z = -d_z / R, B0 * R0 / R, d_r / R
        return jnp.stack((b_r * jnp.cos(phi) - b_phi * jnp.sin(phi), b_r * jnp.sin(phi) + b_phi * jnp.cos(phi), b_z))

    return jax.vmap(one)(xyz)


def test_legs_of_a_double_null_join_only_without_the_three_dimensional_field():
    guess = np.full(21, RX), np.full(21, np.sqrt(0.5))

    def legs(eps, more=0):  # the unstable leg reaches Z = 0 after 4 or 5 maps, the stable one after 3 or 4
        line = pf.periodic_fieldline(_double_null, eps, *guess, nfp=NFP)
        M = pf.tangent_map(_double_null, eps, line)
        return jnp.stack([jnp.stack(pf.manifold_crossing(_double_null, eps, line, M, nfp=NFP, stable=stable,
                                                         iterations=(3 if stable else 4) + more))
                          for stable in (False, True)])

    joined = np.asarray(legs(0.0))
    np.testing.assert_allclose(joined[:, 2], 0.0, atol=1e-10)                    # both reach Z = 0
    np.testing.assert_allclose(joined[:, 1], 0.0, atol=1e-6)                     # and cross it vertically
    np.testing.assert_allclose(joined[:, 0], RX + np.array([-1.0, 1.0]) * np.sqrt(0.15), atol=1e-7, rtol=0)  # psi = psi_X
    eps, h = 0.01, 1e-5
    split = np.asarray(legs(eps))
    assert abs(split[1, 1]) > 1e-2 and np.all(split[:, 2] < 1e-10)
    np.testing.assert_allclose(legs(eps, 1)[:, :2], split[:, :2], atol=1e-6)    # one more map: the same crossing
    fd = (legs(eps + h) - legs(eps - h))[:, :2] / (2.0 * h)
    np.testing.assert_allclose(jax.jacfwd(legs)(eps)[:, :2], fd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(jax.jit(jax.jacrev(legs))(eps)[:, :2], fd, rtol=1e-5, atol=1e-6)


def test_tabulated_field_reproduces_the_field_and_its_x_line():
    eps, options = 0.01, dict(nfp=NFP, r_range=(0.7, 1.7), z_max=1.1, shape=(64, 32, 72))
    table = pf.tabulated_field(_double_null, eps, **options)
    full = pf.tabulated_field(_double_null, eps, symmetric=False, **options)   # the field is stellarator symmetric
    rng = np.random.default_rng(0)
    R, phi, Z = rng.uniform(0.8, 1.6, 200), rng.uniform(-3.0, 3.0, 200), rng.uniform(-1.0, 1.0, 200)
    xyz = jnp.asarray(np.stack((R * np.cos(phi), R * np.sin(phi), Z), -1))
    exact = np.asarray(_double_null(eps, xyz))
    np.testing.assert_allclose(table(None, xyz), full(None, xyz), atol=1e-12)
    np.testing.assert_allclose(table(None, xyz), exact, atol=2e-4 * np.abs(exact).max())
    start = RX, np.sqrt(0.5)
    r, z, M, step = pf.return_map_fixed_point(_double_null, eps, *start, nfp=NFP)
    r_t, z_t, M_t, step_t = pf.return_map_fixed_point(table, None, *start, nfp=NFP)
    assert float(step) < 1e-10 and float(step_t) < 1e-10
    np.testing.assert_allclose([r_t, z_t], [r, z], atol=2e-4)
    assert float(jnp.trace(M_t)) == pytest.approx(float(jnp.trace(M)), rel=5e-3)
