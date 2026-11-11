"""Coil designs with plasma coordinates (vmex.core.coil_design) on the bundled Landreman-Paul QA coils.

The coil-only design, limits and constraint are ESSOS' (essos.coil_design) and tested there."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

try:
    import essos.coil_design  # noqa: F401
    from essos.coils import Coils
    from essos.fields import BiotSavart
except ImportError:
    Coils = BiotSavart = None

# A marker rather than importorskip, so the module still collects (as skips) without ESSOS.
pytestmark = pytest.mark.skipif(Coils is None, reason="needs ESSOS' coil designs (essos.coil_design)")

import vmex as vj  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def coils():
    return Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))


def test_nominal_design_is_the_coils_and_their_field(coils):
    """Without plasma coordinates the design is ESSOS' coil design, through ``vmex.CoilDesign``."""
    design = vj.CoilDesign(coils, current_dofs=(0, 1))
    np.testing.assert_allclose(design.coils_from_x(design.x0).gamma, coils.gamma, atol=1e-14)
    r, phi, z = jnp.asarray([1.0, 1.1]), jnp.asarray([0.0, 0.3]), jnp.asarray([0.0, 0.05])
    np.testing.assert_allclose(design(design.x0).b_cyl(r, phi, z), BiotSavart(coils).b_cyl(r, phi, z), rtol=1e-13)
    x = design.x0.copy()
    x[1] = 0.1  # the second current coordinate: base coil 1 up 10%
    np.testing.assert_allclose(design.base_currents_at(x)[1], 1.1 * coils.dofs_currents_raw[1], rtol=1e-14)
    assert design.plasma_from_parameters is None and len(design.dof_names) == design.size
    # the field of a traced design passes into a jitted function (its pytree metadata stays hashable)
    inner = jax.jit(lambda field: field.b_cyl(r, phi, z)[0])
    np.testing.assert_allclose(jax.jit(lambda x: inner(design(x)))(x), inner(design(x)), rtol=1e-13)


def test_plasma_coordinates_move_phiedge_and_the_current_profile(coils):
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    inp = replace(inp, ac=np.r_[1.0, -1.0, np.zeros(19)], curtor=1.0e4)
    design = vj.CoilDesign(coils, phiedge=float(inp.phiedge), plasma_current=np.r_[inp.ac[:2], inp.curtor])
    params = im.params_from_input(inp)
    x = design.x0.copy()
    x[:4] = 0.1, 0.5, 0.0, -0.2
    x[4] = 0.01  # the first coil coordinate: the plasma coordinates come first
    np.testing.assert_allclose(design.coils_from_x(x).gamma, vj.CoilDesign(coils).coils_from_x(x[4:]).gamma)
    moved = design.plasma_from_parameters(params, x)
    np.testing.assert_allclose(moved.phiedge, 1.1 * inp.phiedge, rtol=1e-14)
    np.testing.assert_allclose(moved.ac[:2], [1.5, -1.0], rtol=1e-14)
    np.testing.assert_allclose(moved.curtor, 0.8e4, rtol=1e-14)


def test_circular_pairs_are_the_loop_formula_and_join_a_design(coils):
    """On the Z axis a pair's field is the two loops' closed form; the joined design adds it to the coils' field."""
    from vmex.core.coil_design import PoloidalCoilDesign, circular_coil_pairs

    radius, height, current = 1.2, 0.8, 2.0e5
    z = np.array([-0.3, 0.0, 0.5])
    loops = BiotSavart(circular_coil_pairs(radius, height, current, n_segments=64))
    exact = sum(2e-7 * np.pi * current * radius**2 / (radius**2 + (z - h) ** 2) ** 1.5 for h in (height, -height))
    B = np.asarray(jax.vmap(loops.B)(jnp.asarray(np.stack((0 * z, 0 * z, z), axis=-1))))
    np.testing.assert_allclose(B[:, 2], exact, rtol=1e-12)
    np.testing.assert_allclose(B[:, :2], 0.0, atol=1e-14)

    base = vj.CoilDesign(coils, current_dofs=(0,))
    design = PoloidalCoilDesign(base, radius, height, current, geometry_scale=0.01)
    assert design.size == base.size + 3 == len(design.dof_names) == design.scales.size
    r, phi, zz = jnp.asarray([1.0, 1.1]), jnp.asarray([0.0, 0.3]), jnp.asarray([0.0, 0.05])
    x = design.x0.copy()
    x[base.size:] = 0.5, 0.01, -0.02   # +50 kA, radius +1 cm, height -2 cm
    moved = BiotSavart(circular_coil_pairs(radius + 0.01, height - 0.02, current + 0.5e5, n_segments=128))
    expected = [a + b for a, b in zip(BiotSavart(coils).b_cyl(r, phi, zz), moved.b_cyl(r, phi, zz))]
    np.testing.assert_allclose(design(x).b_cyl(r, phi, zz), expected, rtol=1e-12)
    # stellarator symmetric: B_R odd, B_phi and B_Z even under (phi, Z) -> (-phi, -Z)
    b_r, b_phi, b_z = moved.b_cyl(r, phi, zz)
    m_r, m_phi, m_z = moved.b_cyl(r, -phi, -zz)
    np.testing.assert_allclose([m_r, m_phi, m_z], [-b_r, b_phi, b_z], atol=1e-13)
    # a traced design's field is a pytree: it passes into a jitted function and differentiates
    inner = jax.jit(lambda field: field.b_cyl(r, phi, zz)[2])
    np.testing.assert_allclose(jax.jit(lambda x: inner(design(x)))(x), inner(design(x)), rtol=1e-13)
    gradient = jax.grad(lambda x: jnp.sum(design(x).b_cyl(r, phi, zz)[2]))(jnp.asarray(x))
    assert np.all(np.asarray(gradient[base.size:]) != 0)
