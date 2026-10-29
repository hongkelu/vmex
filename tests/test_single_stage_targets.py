"""Coil and coil-plasma targets of :mod:`vmex.core.single_stage_targets` (CPU, ESSOS coils of the LP QA case)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("essos")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from essos.coils import Coils  # noqa: E402

from vmex.core import single_stage_targets as targets  # noqa: E402

COILS = Path(__file__).resolve().parents[1] / "examples" / "data" / "ESSOS_biot_savart_LandremanPaulQA.json"
LIMITS = targets.CoilLimits(length=6.0, curvature=6.0, msc=6.0, coil_distance=0.1, curvature_points=256,
                            distance_points=64)


@pytest.fixture(scope="module")
def coils():
    return Coils.from_json(str(COILS))


def test_coil_metrics_are_essos_values(coils):
    """Length, peak curvature and mean-squared curvature are ESSOS's on the base coils; the distances are
    positive and the smallest coil-coil distance is below every coil's length."""
    m = targets.coil_metrics(coils, LIMITS)
    base = targets.resampled_curves(coils, LIMITS.curvature_points, symmetric=False)
    np.testing.assert_allclose(m["length"], base.length, rtol=1e-12)
    np.testing.assert_allclose(m["peak"], np.max(base.curvature, axis=1), rtol=1e-12)
    assert np.all(np.asarray(m["msc"]) <= np.asarray(m["peak"]) ** 2 + 1e-12)
    assert 0 < float(m["coil_distance"]) < float(np.min(m["length"]))
    assert float(m["self_distance"]) > 0 and np.all(np.asarray(m["min_speed"]) > 0)


def test_per_point_rows_have_the_extreme_rows_minimum(coils):
    """Each per-point row family's minimum is the matching per-coil extreme row."""
    extreme = np.asarray(targets.coil_slacks(coils, LIMITS))
    points = np.asarray(targets.coil_slacks(coils, LIMITS, per_point=True))
    n = targets.resampled_curves(coils, 8, symmetric=False).gamma.shape[0]  # base coils
    assert np.isfinite(extreme).all() and np.isfinite(points).all()
    assert min(points) == pytest.approx(min(extreme), rel=1e-9, abs=1e-12)
    assert extreme.size == 4 * n + 2  # length, peak curvature, msc, min speed per coil; coil and self distance


def test_coil_constraint_has_the_jacobian_of_its_rows(coils):
    """The constraint's Jacobian is the derivative of its rows (central differences in two directions)."""
    dofs = jnp.asarray(coils.dofs)

    def coils_from_x(x):
        return coils.with_dofs(dofs.at[:x.size].add(x))

    constraint = targets.coil_constraint(coils_from_x, LIMITS)
    x0 = np.zeros(3)
    J = constraint.jac(x0)
    assert J.shape == (np.size(constraint.fun(x0)), 3)
    for k in range(2):
        e = np.zeros(3)
        e[k] = 1e-6
        fd = (constraint.fun(x0 + e) - constraint.fun(x0 - e)) / 2e-6
        np.testing.assert_allclose(J[:, k], fd, rtol=1e-4, atol=1e-6)


def test_continuous_surface_distance_of_a_coil_around_a_circular_torus():
    """A circular coil of radius 0.5 in the poloidal plane around a circular torus (R0 1, a 0.3) is 0.2 from it."""
    s = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    points = np.stack([1.0 + 0.5 * np.cos(s), np.zeros_like(s), 0.5 * np.sin(s)], axis=-1)[None]
    rbc = np.array([[1.0, 0.3]])
    zbs = np.array([[0.0, 0.3]])
    distance, gap = targets.continuous_surface_distance(jnp.asarray(points), jnp.asarray(rbc), jnp.asarray(zbs), 1)
    assert float(distance) == pytest.approx(0.2, abs=1e-8)
    assert float(gap) >= 0
