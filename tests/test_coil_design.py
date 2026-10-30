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
