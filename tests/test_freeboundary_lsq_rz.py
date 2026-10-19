"""The RZ least-squares free boundary (vmex.core.freeboundary_lsq_rz), full lane (GPU).

The PR lane runs ``test_freeboundary_lsq_rz_fast.py``.  At finite beta (the three interface conditions by virtual
casing, the whole interior edge field): the exact Hessian against central differences of the exact gradient, one
certified boundary from the deck, a shrunk deck and a relabelled deck; and the exact design gradients against
re-solved central differences.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_lsq_rz as rz  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

from tests.test_freeboundary_lsq_rz_fast import relabelled  # noqa: E402
from tests.test_freeboundary_vc import _coil_field, _deck  # noqa: E402

pytestmark = [pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax"),
              pytest.mark.full]

GRID = 32  # >= 4 x the boundary modes
OPTIONS = dict(nphi=GRID, ntheta=GRID, quadrature=(4 * 2 * GRID, 2 * GRID), hessian_chunk=2)


@pytest.fixture(scope="module")
def model_and_field():
    """LP QA at 0.5 % beta with its fitted coils: the three interface conditions by virtual casing."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 7, 13)  # boundary modes 4 (interior - 2)
    model = rz.LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), **OPTIONS)
    assert not model.vacuum
    return model, _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")


def test_rz_finite_beta_hessian_and_start_independence(model_and_field):
    """The rows are the three interface conditions (five a point) and the gauge; the exact Hessian matches central
    differences of the exact gradient; the deck, the deck shrunk by 1 % and the relabelled deck reach one boundary
    (surface and labelling) with a positive-definite reduced Hessian."""
    model, field = model_and_field
    x = np.array(model.x0)
    rows, aux = model.evaluate(x, model.params0, field, tight=True)
    assert rows.size == 6 * model.nphi * model.ntheta + 1  # B.n, pressure, 3 sheet-current components, gauge; G
    so = model.second_order(x, aux, field)
    assert so["asymmetry"] < 1e-8
    np.testing.assert_allclose(so["g_adjoint"], so["g"], rtol=1e-6, atol=1e-12 * np.max(np.abs(so["g"])))
    S = np.asarray(model.x_scale)
    v = np.random.default_rng(0).standard_normal(x.size)
    v /= np.linalg.norm(v)
    eps, grads = 1e-4, []
    for sign in (1, -1):
        xs = x + sign * eps * S * v
        _, aux_s = model.evaluate(xs, model.params0, field, tight=True)
        grads.append(S * model.gradient(xs, aux_s, field, shift=so["shift"]))
    fd = (grads[0] - grads[1]) / (2 * eps)
    np.testing.assert_allclose(S * (so["H"] @ (S * v)), fd, rtol=1e-4, atol=1e-4 * np.max(np.abs(fd)))
    deck = model.fixed
    m = np.arange(np.asarray(deck.rbc).shape[1])[None, :] >= 1
    shrunk = replace(deck, rbc=np.where(m, 0.99, 1.0) * np.asarray(deck.rbc),
                     zbs=np.where(m, 0.99, 1.0) * np.asarray(deck.zbs))
    ends = []
    for x0 in (None, model.displacement_of(shrunk), model.displacement_of(relabelled(deck))):
        out = model.solve_boundary(model.params0, field, x0=x0, max_iter=60)
        assert out["converged"], out["status"]
        assert out["history"][-1][2] > 0
        ends.append(out["x"])
    for end in ends[1:]:
        assert model._boundary_rms(end - ends[0]) < 1e-5 * model.R0


def test_rz_exact_design_gradients_match_resolved_differences():
    """Design gradients (exact reduced Hessian) against re-solved central differences, along a plasma coordinate
    (PHIEDGE) and a field coordinate (the coil currents), at finite beta."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 7, 13)
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,), method="least_squares_rz",
        three_terms_options=dict(boundary_max_nfev=60, **OPTIONS))
    assert type(problem.model) is rz.LeastSquaresRZFreeBoundaryModel
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = []
        for s in (1, -1):
            problem._anchor = {}  # cold trials: the differences do not lean on the gradient's own linearization
            values.append(np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)])
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=2e-3, atol=1e-9 * max(1.0, abs(value)))
