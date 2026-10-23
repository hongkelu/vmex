"""The three-term driver's filament field (DirectCoilField): the chunked sum is the plain Biot-Savart mean (CPU)."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def driver():
    os.environ.setdefault("COIL_CASE", "qi6-beta-A8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "optimization"))
    return importlib.import_module("single_stage_free_boundary_optimization_three_term")


def _plain(gamma, gamma_dash, currents, r, phi, z):
    xyz = jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
    d = xyz[:, None, None, :] - gamma
    w = jnp.cross(gamma_dash, d) * (jnp.sum(d * d, axis=-1) ** -1.5)[..., None] * currents[None, :, None, None]
    b = 1e-7 * jnp.mean(jnp.sum(w, axis=1), axis=1)
    return (jnp.cos(phi) * b[:, 0] + jnp.sin(phi) * b[:, 1], -jnp.sin(phi) * b[:, 0] + jnp.cos(phi) * b[:, 1],
            b[:, 2])


@pytest.mark.parametrize("samples", [100, 1024, 1536])
def test_chunked_field_is_the_plain_sum(driver, samples):
    """One chunk (100 points) and several (1024 = 2 x 512, 1536 = 3 x 512): value and gradient equal the
    one-shot sum to round-off."""
    t = jnp.linspace(0, 2 * jnp.pi, samples, endpoint=False)
    rng = np.random.default_rng(0)
    coils = []
    for k in range(3):  # tilted, offset circles of radius 0.4 around the torus
        a = 2 * np.pi * k / 3
        R, Z = 1.0 + 0.4 * jnp.cos(t), 0.4 * jnp.sin(t)
        coils.append(jnp.stack([R * np.cos(a), R * np.sin(a), Z], -1))
    gamma = jnp.stack(coils)
    gamma_dash = jnp.gradient(gamma, axis=1) * samples / (2 * np.pi) * 2 * np.pi
    currents = jnp.asarray(rng.normal(size=3) * 1e5)
    field = driver.DirectCoilField(gamma, gamma_dash, currents)
    r = jnp.asarray(1.0 + 0.1 * rng.random(50))
    phi = jnp.asarray(2 * np.pi * rng.random(50))
    z = jnp.asarray(0.1 * rng.standard_normal(50))
    got, want = field.b_cyl(r, phi, z), _plain(gamma, gamma_dash, currents, r, phi, z)
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-15)
    g_got = jax.grad(lambda rr: jnp.sum(field.b_cyl(rr, phi, z)[1] ** 2))(r)
    g_want = jax.grad(lambda rr: jnp.sum(_plain(gamma, gamma_dash, currents, rr, phi, z)[1] ** 2))(r)
    np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_want), rtol=1e-11, atol=1e-15)
