"""The three-term driver's continuous coil-plasma clearance (CPU; analytic torus and circular coil)."""
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


def _torus(a, nfp=2, ntor=2, mpol=3):
    rbc = jnp.zeros((2 * ntor + 1, mpol)).at[ntor, 0].set(1.0).at[ntor, 1].set(a)
    zbs = jnp.zeros((2 * ntor + 1, mpol)).at[ntor, 1].set(a)
    return rbc, zbs, nfp


def _coil(samples, shift, phi=0.3):
    """A circle of radius 0.45 in the plane phi, centred 0.05 outboard of the torus' minor axis."""
    t = jnp.linspace(0, 2 * jnp.pi, samples, endpoint=False) + shift
    R, Z = 1.05 + 0.45 * jnp.cos(t), 0.45 * jnp.sin(t)
    return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], -1)[None]


@pytest.mark.parametrize("samples", [16, 32, 128])
def test_continuous_clearance_is_exact(driver, samples):
    """Exact clearance 0.45 - 0.05 - 0.2 = 0.2 and d/da = -1 (a: the minor radius), for samples off the closest
    point: the circle is a Fourier curve, so its interpolant and the Newton pair are exact at any sampling."""
    for shift in (0.05, 0.11):
        d, gap = driver.continuous_surface_distance(_coil(samples, shift), *_torus(0.2))
        assert abs(float(d) - 0.2) < 1e-10
        grad = jax.grad(lambda a: driver.continuous_surface_distance(_coil(samples, shift), *_torus(a))[0])(0.2)
        assert float(grad) == pytest.approx(-1.0, abs=1e-8)
        assert float(gap) > 0


def _ellipse(samples, shift, tilt=0.4, phi=0.3):
    """A tilted ellipse in the plane phi (semi-axes 0.5, 0.42), off-centre: its closest point to the torus is at one
    parameter, and ``shift`` slides the samples past it."""
    t = jnp.linspace(0, 2 * jnp.pi, samples, endpoint=False) + shift
    u, v = 0.5 * jnp.cos(t), 0.42 * jnp.sin(t)
    R = 1.08 + jnp.cos(tilt) * u - jnp.sin(tilt) * v
    Z = 0.03 + jnp.sin(tilt) * u + jnp.cos(tilt) * v
    return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], -1)[None]


def test_continuous_clearance_does_not_jump_when_the_closest_sample_changes(driver):
    """Sliding the samples along the same curve across a full sample spacing (the closest sample changes) leaves
    the clearance unchanged, and its design derivative matches central differences across the switch."""
    samples = 24
    spacing = 2 * np.pi / samples
    values = [float(driver.continuous_surface_distance(_ellipse(samples, f * spacing), *_torus(0.2))[0])
              for f in np.linspace(0, 1, 9)]
    assert np.ptp(values) < 1e-10 and 0.05 < values[0] < 0.4

    def f(a, shift):
        return driver.continuous_surface_distance(_ellipse(samples, shift), *_torus(a))[0]

    for shift in (0.0, 0.5 * spacing):
        grad = float(jax.grad(f)(0.2, shift))
        for h in (1e-4, 1e-5, 1e-6):
            fd = (float(f(0.2 + h, shift)) - float(f(0.2 - h, shift))) / (2 * h)
            assert fd == pytest.approx(grad, abs=1e-6)


def test_continuous_clearance_is_below_the_point_cloud(driver):
    """The point cloud overestimates the continuous clearance: the minimum over sampled surface points is >= it."""
    d, _ = driver.continuous_surface_distance(_coil(64, 0.05), *_torus(0.2))
    pts = np.asarray(_coil(64, 0.05)[0])
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 48, endpoint=False), np.linspace(0, 2 * np.pi, 96, endpoint=False),
                         indexing="ij")
    R = 1 + 0.2 * np.cos(th)
    S = np.stack([R * np.cos(ph), R * np.sin(ph), 0.2 * np.sin(th)], -1).reshape(-1, 3)
    cloud = np.min(np.sqrt(np.min(np.sum((pts[:, None] - S[None]) ** 2, -1), 1)))
    assert float(d) < cloud
