"""The least-squares free boundary (vmex.core.freeboundary_lsq), full lane.

The PR lane runs ``test_freeboundary_lsq_fast.py``.

The second-order adjoint (exact Hessian of 1/2 |rows|^2 through the equilibrium), the labelling gauges
(relabelling map, restoration onto the gauge), the trust-region Newton solve and the exact design gradients
on the gauge manifold, each against an independent oracle: central differences of the exact gradient, a
relabelled start, and re-solved central differences.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_lsq as lsq  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

from tests.test_freeboundary_vc import _coil_field, _deck  # noqa: E402

pytestmark = [pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax"),
              pytest.mark.full]  # ~30 min on a GPU: the PR lane runs test_freeboundary_lsq_fast.py

GRID = 32  # >= 4 x the boundary modes
OPTIONS = dict(nphi=GRID, ntheta=GRID, quadrature=(4 * 2 * GRID, 2 * GRID), hessian_chunk=2)


def _vacuum_deck():
    return replace(_deck("input.LandremanPaul2021_QA_lowres", 6, 13), phiedge=-0.025, curtor=0.0, am=np.zeros(21))


@pytest.fixture(scope="module")
def model_and_field():
    """LP QA at 0.5 % beta with its fitted coils: the three-terms rows (virtual casing, band-limited edge)."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 7, 13)  # boundary modes 4 (interior - 2)
    model = lsq.LeastSquaresFreeBoundaryModel(replace(inp, lfreeb=False), **OPTIONS)
    assert not model.vacuum
    return model, _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")


@pytest.mark.usefixtures("_module_jit_enabled")
def test_exact_hessian_matches_gradient_differences(model_and_field):
    """H v from the second-order adjoint against central differences of the exact gradient J^T r, and the
    gradient from the multiplier against J^T r."""
    model, field = model_and_field
    x = np.array(model.x0)
    _, aux = model.evaluate(x, model.params0, field, tight=True)
    so = model.second_order(x, aux, field)
    assert so["asymmetry"] < 1e-8
    np.testing.assert_allclose(so["g_adjoint"], so["g"], rtol=1e-6, atol=1e-12 * np.max(np.abs(so["g"])))
    S = np.asarray(model.x_scale)
    v = np.random.default_rng(0).standard_normal(x.size)
    v /= np.linalg.norm(v)
    eps = 1e-4
    grads = []
    for sign in (1, -1):
        xs = x + sign * eps * S * v
        _, aux_s = model.evaluate(xs, model.params0, field, tight=True)
        grads.append(S * model.gradient(xs, aux_s, field))
    fd = (grads[0] - grads[1]) / (2 * eps)
    np.testing.assert_allclose(S * (so["H"] @ (S * v)), fd, rtol=1e-4, atol=1e-4 * np.max(np.abs(fd)))


@pytest.mark.usefixtures("_module_jit_enabled")
def test_solve_is_start_independent(model_and_field):
    """Two starts (the deck, and a random displacement of it) reach the same boundary at finite beta,
    with a positive-definite reduced Hessian."""
    model, field = model_and_field
    nb = model.n_boundary
    start = model.x0.copy()
    start[:nb] += 0.3 * model.x_scale[:nb] * np.random.default_rng(1).standard_normal(nb)
    ends = []
    for x0 in (None, start):
        out = model.solve_boundary(model.params0, field, x0=x0, max_iter=40)
        assert out["converged"], out["status"]
        assert out["history"][-1][2] > 0  # positive-definite reduced Hessian
        ends.append(out["x"])
    np.testing.assert_allclose(ends[0], ends[1], atol=1e-6)


@pytest.mark.usefixtures("_module_jit_enabled")
def test_solve_entry_least_squares():
    """``solve_free_boundary_three_terms`` (vacuum: B_coil.n rows) returns a converged result with its history;
    an unknown method is rejected."""
    inp = _vacuum_deck()
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    entry = {k: v for k, v in OPTIONS.items() if k != "hessian_chunk"}
    fit = fvc.solve_free_boundary_three_terms(inp, external_field=field, method="least_squares", **entry)
    assert fit.model.vacuum and isinstance(fit.model, lsq.LeastSquaresFreeBoundaryModel)
    assert fit.success and fit.status.startswith("converged")
    assert fit.history.shape[1] == 10
    assert fit.boundary_residual.normal < 1e-2
    with pytest.raises(ValueError, match="method must be"):
        fvc.solve_free_boundary_three_terms(inp, external_field=field, method="lsq")


@pytest.mark.usefixtures("_module_jit_enabled")
def test_exact_design_gradients_match_resolved_differences():
    """Design gradients with the exact reduced Hessian (``exact_gradient=True``) against re-solved central
    differences, along a plasma coordinate (PHIEDGE) and a field coordinate (the coil currents)."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = FreeBoundaryProblem.from_loss(
        _vacuum_deck(), lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,), boundary_condition="three_terms",
        three_terms_options=dict(method="least_squares", boundary_max_nfev=40, **OPTIONS))
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = [np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=2e-3, atol=1e-9 * max(1.0, abs(value)))


@pytest.mark.usefixtures("_module_jit_enabled")
def test_bootstrap_rows_are_constraints_with_exact_design_gradients():
    """LP QA at 0.5 % beta with the Redl current in the solve: the current's rows hold as equality constraints at
    the solution, and the design gradients (a plasma coordinate, PHIEDGE, which moves the constraint rows too, and
    the coil currents) match re-solved central differences."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    from tests.test_freeboundary_vc import _lp_beta0p5_profiles

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    problem = FreeBoundaryProblem.from_loss(
        replace(inp, lfreeb=False), lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,), boundary_condition="three_terms",
        three_terms_options=dict(method="least_squares", boundary_max_nfev=40, bootstrap=_lp_beta0p5_profiles(inp),
                                 **OPTIONS))
    model = problem.model
    m = model.n_constraint_rows
    assert m == model.x0.size - model.n_boundary > 0
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    sol = problem._record(x0)
    rows = np.asarray(sol["rows"])
    assert 0.5 * rows[-m:] @ rows[-m:] <= 1e-10 * 0.5 * rows[:-m] @ rows[:-m]
    jac = problem.constraint_jac(x0)
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = [np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=5e-3, atol=1e-9 * max(1.0, abs(value)))
