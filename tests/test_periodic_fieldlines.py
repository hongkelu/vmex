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
    assert step < 1e-12
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
    gradient = jax.grad(lambda p: pf.surface_distance(p, gamma, NFP))(points)
    np.testing.assert_allclose(gradient[0], [np.cos(angle), np.sin(angle), 0.0], atol=0.05)
