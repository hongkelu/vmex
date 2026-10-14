"""Exact-Newton least squares of the three-terms free boundary, fast lane (vmex.core.freeboundary_vc).

With vacuum ``B_coil . n`` rows (no virtual casing), the
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


from vmex.core import freeboundary_lsq as lsq  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

from tests.test_freeboundary_vc import _coil_field, _deck  # noqa: E402

pytestmark = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")


@pytest.mark.usefixtures("_module_jit_enabled")
def test_vacuum_rows_newton_with_exact_hessian():
    """Vacuum B_coil.n rows: the exact Hessian matches central differences of the exact gradient, and the solve
    converges with a positive-definite reduced Hessian."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 9), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    model = lsq.LeastSquaresFreeBoundaryModel(replace(inp, lfreeb=False), nphi=16, ntheta=16, quadrature=(64, 32))
    assert model.vacuum
    x = np.array(model.x0)
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
    # Hessian-vector blocks: equal to the columns, and a block that exhausts device memory is split in halves
    so_p = model.second_order(x, aux, field, columns=False)
    V = np.eye(x.size)[:3]
    calls = []
    columns = model._hessian_columns

    def short_of_memory(*args):
        calls.append(jax.tree.leaves(args[6])[0].shape[0])  # the block's width (dz_cols)
        if len(calls) == 1:
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: out of memory (test)")
        return columns(*args)

    model.hessian_chunk, model._hessian_columns = 4, short_of_memory
    try:
        np.testing.assert_allclose(so_p["hvp"](V), so["H"][:3], rtol=1e-8, atol=1e-10 * np.max(np.abs(so["H"])))
    finally:
        model._hessian_columns = columns
    assert model.hessian_chunk == 1 and calls[0] == 3 and len(calls) == 4
    out = model.solve_boundary(model.params0, field, max_iter=60)
    assert out["converged"], out["status"]
    assert out["history"][-1][2] > 0
