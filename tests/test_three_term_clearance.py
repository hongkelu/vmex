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


@pytest.mark.parametrize("samples, tol", [(32, 3e-6), (64, 5e-7), (128, 1e-8)])
def test_continuous_clearance_is_exact_and_converges_with_sampling(driver, samples, tol):
    """Exact clearance 0.45 - 0.05 - 0.2 = 0.2 and d/da = -1 (a: the minor radius), for samples off the closest
    point: the parabola along the coil converges as the cube of the spacing, the surface point is exact (Newton)."""
    for shift in (0.05, 0.11):
        d, gap = driver.continuous_surface_distance(_coil(samples, shift), *_torus(0.2))
        assert abs(float(d) - 0.2) < tol
        grad = jax.grad(lambda a: driver.continuous_surface_distance(_coil(samples, shift), *_torus(a))[0])(0.2)
        assert float(grad) == pytest.approx(-1.0, abs=1e-6)
        assert float(gap) > 0


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
