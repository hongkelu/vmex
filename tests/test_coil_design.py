"""Coil designs and coil constraint rows (vmex.core.coil_design) on the bundled Landreman-Paul QA coils."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("essos")
objective_functions = pytest.importorskip("essos.objective_functions")
if not hasattr(objective_functions, "coil_coil_distances"):
    pytest.skip("needs ESSOS' coil distance values (coil_coil_distances)", allow_module_level=True)

from essos.coils import Coils  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402

import vmex as vj  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
LIMITS = vj.CoilLimits(length=5.0, curvature=5.0, mean_squared_curvature=5.0, coil_distance=0.15,
                       curvature_points=64, distance_points=32)


@pytest.fixture(scope="module")
def coils():
    return Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))


def test_nominal_design_is_the_coils_and_their_field(coils):
    design = vj.CoilDesign(coils, current_dofs=(0, 1))
    np.testing.assert_allclose(design.coils_from_x(design.x0).gamma, coils.gamma, atol=1e-14)
    r, phi, z = jnp.asarray([1.0, 1.1]), jnp.asarray([0.0, 0.3]), jnp.asarray([0.0, 0.05])
    np.testing.assert_allclose(design(design.x0).b_cyl(r, phi, z), BiotSavart(coils).b_cyl(r, phi, z), rtol=1e-13)
    x = design.x0.copy()
    x[1] = 0.1  # the second current coordinate: base coil 1 up 10%
    np.testing.assert_allclose(design.base_currents_at(x)[1], 1.1 * coils.dofs_currents_raw[1], rtol=1e-14)
    assert design.plasma_from_parameters is None and len(design.dof_names) == design.size


def test_plasma_coordinates_move_phiedge_and_the_current_profile(coils):
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    inp = replace(inp, ac=np.r_[1.0, -1.0, np.zeros(19)], curtor=1.0e4)
    design = vj.CoilDesign(coils, phiedge=float(inp.phiedge), plasma_current=np.r_[inp.ac[:2], inp.curtor])
    params = im.params_from_input(inp)
    x = design.x0.copy()
    x[:4] = 0.1, 0.5, 0.0, -0.2
    moved = design.plasma_from_parameters(params, x)
    np.testing.assert_allclose(moved.phiedge, 1.1 * inp.phiedge, rtol=1e-14)
    np.testing.assert_allclose(moved.ac[:2], [1.5, -1.0], rtol=1e-14)
    np.testing.assert_allclose(moved.curtor, 0.8e4, rtol=1e-14)


def test_extreme_rows_are_the_minima_of_the_point_rows(coils):
    extreme, point = np.asarray(LIMITS.rows(coils)), np.asarray(LIMITS.rows(coils, per_point=True))
    n = coils.dofs_curves.shape[0]
    curvature_points, distance_points = LIMITS.curvature_points, LIMITS.distance_points
    assert extreme.size == 4 * n + 2
    assert point.size == 3 * n + n * curvature_points + 2 * n * distance_points
    np.testing.assert_allclose(point[:n], extreme[:n])                                   # length
    np.testing.assert_allclose(point[n:n + n * curvature_points].reshape(n, -1).min(axis=1), extreme[n:2 * n])
    rest = point[n + n * curvature_points:]
    np.testing.assert_allclose(rest[:n], extreme[2 * n:3 * n])                         # mean squared curvature
    np.testing.assert_allclose(rest[n:n + n * distance_points].min(), extreme[3 * n], rtol=1e-12)
    np.testing.assert_allclose(rest[n + n * distance_points:-n].min(), extreme[3 * n + 1], rtol=1e-12)
    np.testing.assert_allclose(rest[-n:], extreme[-n:])                                # minimum speed


def test_coil_constraint_jacobians_match_differences(coils):
    design = vj.CoilDesign(coils, current_dofs=(0,))
    rows = lambda x: LIMITS.rows(design.coils_from_x(x))  # noqa: E731
    reverse = vj.coil_constraint(rows)
    forward = vj.coil_constraint(rows, forward_chunk=16)
    rng = np.random.default_rng(0)
    x = 0.1 * design.scales * rng.standard_normal(design.size)
    J = reverse.jac(x)
    np.testing.assert_allclose(forward.jac(x), J, rtol=1e-10, atol=1e-12)
    v = design.scales * rng.standard_normal(design.size)
    h = 1e-6
    fd = (reverse.fun(x + h * v) - reverse.fun(x - h * v)) / (2 * h)
    np.testing.assert_allclose(J @ v, fd, rtol=1e-5, atol=1e-8)
