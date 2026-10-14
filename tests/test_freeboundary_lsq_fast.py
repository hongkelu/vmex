"""Exact-Newton least squares of the three-terms free boundary, fast lane (vmex.core.freeboundary_vc).

The labelling gauges (relabelling map, restoration) and, with vacuum ``B_coil . n`` rows (no virtual casing), the
exact Hessian against central differences of the exact gradient and a converged Newton solve, at a resolution that
runs in minutes.  The virtual-casing rows, start independence and design gradients are in
``test_freeboundary_lsq.py`` (full lane).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_lsq as lsq  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

from tests.test_freeboundary_vc import _coil_field, _deck  # noqa: E402

pytestmark = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")


def test_relabelling_map_and_gauge_restoration():
    """The relabelling map is the identity at omega = 0; the restoration lands on c = 0, and a relabelled
    boundary restores to the same boundary (the gauge is a property of the surface, not of its labelling)."""
    inp = _deck("input.LandremanPaul2021_QA_lowres", 5, 9)
    for gauge in ("canonical",):
        model = lsq.LeastSquaresFreeBoundaryModel.__new__(lsq.LeastSquaresFreeBoundaryModel)
        model.fixed, model.fixed_boundary, model.max_mode = replace(inp, lfreeb=False), False, 4
        model.x0 = opt.pack_boundary(inp, 4, vary_major_radius=True)
        model.n_boundary = model.x0.size
        model.x_scale = np.ones_like(model.x0)
        model._gauge_setup(gauge)
        x0 = jnp.asarray(model.x0)
        np.testing.assert_allclose(model._relabel(x0, jnp.zeros(model.n_gauge)), model.x0, atol=1e-14)
        x1 = model.gauge_restore(model.x0)
        assert np.max(np.abs(np.asarray(model.gauge_constraints(jnp.asarray(x1))))) < 1e-12
        if gauge == "canonical":  # a relabelled start restores to the same boundary, up to the truncation
            shifted = np.asarray(model._relabel(jnp.asarray(x1), 1e-4 * jnp.ones(model.n_gauge)))
            np.testing.assert_allclose(model.gauge_restore(shifted), x1, atol=1e-6)
    with pytest.raises(ValueError, match="unknown gauge"):
        model._gauge_setup("anchor")



@pytest.mark.usefixtures("_module_jit_enabled")
def test_vacuum_rows_newton_with_exact_hessian():
    """Vacuum B_coil.n rows: the exact Hessian matches central differences of the exact gradient, and the solve
    converges in the canonical gauge with a positive-definite reduced Hessian."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 9), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    model = lsq.LeastSquaresFreeBoundaryModel(replace(inp, lfreeb=False), nphi=16, ntheta=16, quadrature=(64, 32))
    assert model.vacuum
    x = model.gauge_restore(model.x0)
    _, aux = model.evaluate(x, model.params0, field, tight=True)
    so = model.second_order(x, aux, field)
    np.testing.assert_allclose(so["g_adjoint"], so["g"], rtol=1e-6, atol=1e-12 * np.max(np.abs(so["g"])))
    S = np.asarray(model.x_scale)
    v = np.random.default_rng(0).standard_normal(x.size)
    v /= np.linalg.norm(v)
    eps, grads = 1e-4, []
    for sign in (1, -1):
        xs = x + sign * eps * S * v
        _, aux_s = model.evaluate(xs, model.params0, field, tight=True)
        grads.append(S * model.gradient(xs, aux_s, field))
    fd = (grads[0] - grads[1]) / (2 * eps)
    np.testing.assert_allclose(S * (so["H"] @ (S * v)), fd, rtol=1e-4, atol=1e-4 * np.max(np.abs(fd)))
    out = model.solve_boundary(model.params0, field, max_iter=60)
    assert out["converged"], out["status"]
    assert out["history"][-1][2] > 0
