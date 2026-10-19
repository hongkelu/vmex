"""The RZ least-squares free boundary (vmex.core.freeboundary_lsq_rz), fast lane.

Vacuum ``B . n`` rows (no virtual casing) at a resolution that runs in minutes: the coordinates (every packed
coefficient, any labelling), the deck gauge's rows, the exact Hessian against central differences of the exact
gradient, and a certified solve that reaches the same surface and labelling from a relabelled start.  The
finite-beta rows (three interface conditions) and the design gradients are in ``test_freeboundary_lsq_rz.py``
(full lane).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


from vmex.core import freeboundary_lsq_rz as rz  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

from tests.test_freeboundary_vc import _coil_field, _deck  # noqa: E402

pytestmark = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")

OPTIONS = dict(nphi=16, ntheta=16, quadrature=(64, 32))


def _vacuum_deck():
    return replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 9), phiedge=-0.025, curtor=0.0, am=np.zeros(21))


def relabelled(inp, amplitude=0.1, nt=64, nz=32):
    """``inp``'s surface in the labelling ``theta -> theta + amplitude sin(theta)``, fitted on its own modes."""
    rbc, zbs, nfp = np.asarray(inp.rbc, float), np.asarray(inp.zbs, float), int(inp.nfp)
    ntor, mpol = (rbc.shape[0] - 1) // 2, rbc.shape[1]
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    ze = np.linspace(0, 2 * np.pi / nfp, nz, endpoint=False)
    T, P = np.meshgrid(th, ze, indexing="ij")
    ns, ms = np.arange(-ntor, ntor + 1), np.arange(mpol)
    old = ms[None, :, None, None] * (T + amplitude * np.sin(T))[None, None] - ns[:, None, None, None] * nfp * P
    R, Z = np.einsum("nm,nmij->ij", rbc, np.cos(old)), np.einsum("nm,nmij->ij", zbs, np.sin(old))
    new = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * P
    cos, sin = np.cos(new).reshape(rbc.size, -1).T, np.sin(new).reshape(rbc.size, -1).T
    return replace(inp, rbc=np.linalg.lstsq(cos, R.ravel(), rcond=None)[0].reshape(rbc.shape),
                   zbs=np.linalg.lstsq(sin, Z.ravel(), rcond=None)[0].reshape(zbs.shape))


def test_rz_vacuum_hessian_gauge_and_relabelled_start():
    """Every coefficient is a coordinate (a relabelled deck is a coordinate vector of its own), the gauge rows
    vanish at the deck and price a relabelling, the exact Hessian matches central differences of the exact
    gradient, and the solve from the deck and from the relabelled deck certifies one boundary."""
    inp = _vacuum_deck()
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    model = rz.LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), **OPTIONS)
    assert model.vacuum and model.gauge == "deck"
    modes = len(model.boundary_modes) - 1  # (0, 0) is RBC(0, 0), the last coordinate
    assert model.n_boundary == 2 * modes + 1 == model.x0.size
    np.testing.assert_allclose(model.displacement_of(model.fixed), 0.0, atol=1e-15)
    start = model.displacement_of(relabelled(model.fixed))
    assert model._boundary_rms(start) > 1e-3  # the relabelling moves the coefficients (the points, by mm)
    x = np.array(model.x0)
    rows, aux = model.evaluate(x, model.params0, field, tight=True)
    grid = model.nphi * model.ntheta
    np.testing.assert_array_equal(rows[grid:2 * grid], 0.0)  # the gauge rows: zero at the deck
    rows_relab, _ = model.evaluate(start, model.params0, field, tight=True)
    # a relabelling by 0.1 sin(theta) costs about gauge_weight 0.1 / sqrt(2) (area-weighted RMS)
    assert np.linalg.norm(rows_relab[grid:2 * grid]) > 0.03 * model.gauge_weight
    so = model.second_order(x, aux, field)
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
    ends = []
    for x0 in (None, start):
        out = model.solve_boundary(model.params0, field, x0=x0, max_iter=60)
        assert out["converged"], out["status"]
        assert out["history"][-1][2] > 0  # positive-definite reduced Hessian: the gauge removes the flat directions
        ends.append(out["x"])
    assert model._boundary_rms(ends[0] - ends[1]) < 1e-5 * model.R0  # the same points: surface and labelling


def test_rz_entry_and_options():
    """``method="least_squares_rz"`` through ``solve_free_boundary_three_terms`` (its ``method_options``) and
    ``FreeBoundaryProblem``; unknown gauges and options are rejected; the default methods are unchanged."""
    from vmex.core.freeboundary_lsq import LeastSquaresFreeBoundaryProblem
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = _vacuum_deck()
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    with pytest.raises(ValueError, match="gauge must be"):
        rz.LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), gauge="equal_area", **OPTIONS)
    with pytest.raises(ValueError, match="takes no method_options"):
        fvc.solve_free_boundary_three_terms(inp, external_field=field, method_options=dict(gauge="none"))
    fit = fvc.solve_free_boundary_three_terms(inp, external_field=field, method="least_squares_rz",
                                              method_options=dict(gauge="none"), **OPTIONS)
    assert isinstance(fit.model, rz.LeastSquaresRZFreeBoundaryModel) and fit.model.gauge == "none"
    assert fit.success and fit.boundary_residual.normal < 1e-2
    other = fit.model.rereferenced(fit.x)  # the solved boundary as the deck: same class and options, zero offset
    assert type(other) is rz.LeastSquaresRZFreeBoundaryModel and other.gauge == "none"
    np.testing.assert_allclose(np.asarray(other._x_ref), np.asarray(fit.model._x_ref) + fit.x, atol=1e-14)
    assert issubclass(rz.LeastSquaresRZFreeBoundaryProblem, LeastSquaresFreeBoundaryProblem)
    assert rz.LeastSquaresRZFreeBoundaryProblem.model_class is rz.LeastSquaresRZFreeBoundaryModel
    calls = []

    class Recorder(rz.LeastSquaresRZFreeBoundaryProblem):
        @classmethod
        def from_loss(cls, *args, **kw):
            calls.append(kw)
            return cls

    original = rz.LeastSquaresRZFreeBoundaryProblem
    rz.LeastSquaresRZFreeBoundaryProblem = Recorder
    try:
        built = FreeBoundaryProblem.from_loss(inp, lambda *a: 0.0, np.zeros(1), field_from_parameters=lambda x: field,
                                              method="least_squares_rz", three_terms_options=dict(gauge_weight=1e-3))
    finally:
        rz.LeastSquaresRZFreeBoundaryProblem = original
    assert built is Recorder and calls[0]["gauge_weight"] == 1e-3


def test_rz_normal_tangential_basis_same_answer():
    """The basis ``"normal_tangential"`` is a change of coordinates only: it maps onto every packed coefficient
    (square, invertible), round-trips a relabelled surface, and the solve certifies the same boundary as the basis
    ``"rz"``."""
    inp = _vacuum_deck()
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    rz_model = rz.LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), **OPTIONS)
    nt_model = rz.LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), basis="normal_tangential", **OPTIONS)
    assert nt_model._basis.shape == (rz_model.n_boundary, rz_model.n_boundary) == (nt_model.n_boundary,) * 2
    surface = relabelled(nt_model.fixed)
    y = nt_model.displacement_of(surface)
    np.testing.assert_allclose(nt_model._basis @ y, rz_model.displacement_of(surface), atol=1e-12)
    ends = []
    for model in (rz_model, nt_model):
        out = model.solve_boundary(model.params0, field, max_iter=60)
        assert out["converged"], out["status"]
        params = model.with_boundary(model.params0, out["x"])
        ends.append(np.r_[np.ravel(params.rbc), np.ravel(params.zbs)])
    assert np.sqrt(np.mean((ends[0] - ends[1]) ** 2)) < 1e-6 * rz_model.R0
    with pytest.raises(ValueError, match="basis must be"):
        rz.LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), basis="xyz", **OPTIONS)


def test_rz_labelling_presolve():
    """The labelling pre-solve returns the deck with a weak-gauge solve's boundary (and nothing else changed), and
    ``labelling_presolve`` solves from it to a certified boundary (the strong gauge's)."""
    inp = _vacuum_deck()
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    deck = rz.presolve_labelling(replace(inp, lfreeb=False), field, max_iter=10, **OPTIONS)
    assert not np.allclose(deck.rbc, inp.rbc) and deck.phiedge == inp.phiedge and np.array_equal(deck.ns_array, inp.ns_array)
    fit = fvc.solve_free_boundary_three_terms(
        inp, external_field=field, method="least_squares_rz",
        method_options=dict(gauge_weight=1e-1, labelling_presolve=dict(max_iter=10)), **OPTIONS)
    assert fit.success and np.allclose(fit.model.fixed.rbc, deck.rbc)  # the pre-solved deck is the model's


def test_fresh_exact_model_takes_its_whole_newton_step():
    """With ``exact_takes_newton_step`` (the RZ least squares' setting), a fresh exact (positive-definite) model's
    first step is its whole Newton step, whatever radius the cheaper models' failures left: on the sphere rows of
    ``test_freeboundary_lsq_fast`` the Gauss-Newton and BFGS steps overshoot and the radius shrinks below their Newton
    steps; without it the exact model reached its own Newton step only over radius doublings, each a further exact
    model in a trial."""
    from vmex.core import freeboundary_lsq as lsq
    from tests.test_freeboundary_lsq_fast import _SphereModel

    assert rz.LeastSquaresRZFreeBoundaryModel.exact_takes_newton_step
    assert not lsq.LeastSquaresFreeBoundaryModel.exact_takes_newton_step
    calls, current = [], {"exact": False, "first": False}
    solution = lsq._trust_region_solution

    def recorded(w, gq, radius, newton):  # (exact model, its first step, radius, Newton step)
        calls.append((current["exact"], current["first"], radius, newton))
        current["first"] = False
        return solution(w, gq, radius, newton)

    rng = np.random.default_rng(0)
    lsq._trust_region_solution = recorded
    try:
        for n, weight in ((3, 0.01), (6, 0.03), (6, 0.1)):
            start = np.r_[0.3, rng.standard_normal(n - 1)]
            model = _SphereModel(weight, n)
            model.exact_takes_newton_step = True
            model_of = model._derivative_model

            def tagged(exact, *args, model_of=model_of, **kwargs):
                current.update(exact=bool(exact), first=True)
                return model_of(exact, *args, **kwargs)

            model._derivative_model = tagged
            out = model._newton(None, None, x0=start / np.linalg.norm(start), max_iter=200)
            assert out["converged"], (n, weight, out["status"])
            np.testing.assert_allclose(out["x"], np.r_[1.0, np.zeros(n - 1)], atol=1e-5)
    finally:
        lsq._trust_region_solution = solution
    finite = [c for c in calls if np.isfinite(c[3])]
    assert any(radius < newton for exact, _, radius, newton in finite if not exact)  # cheap steps cut by the radius
    firsts = [(radius, newton) for exact, first, radius, newton in finite if exact and first]
    assert firsts and all(radius >= newton for radius, newton in firsts)


def test_rz_design_gradient_with_krylov_trial_certificates():
    """The RZ problem's trials carry the Krylov certificate (``trial_certificate = "krylov"``) and an accepted
    point's design gradient builds the full columns itself: on the vacuum deck the gradient along PHIEDGE and the
    coil currents matches central differences of re-solved trials (FD truncation at h = 5e-4: ~1e-7)."""
    from dataclasses import replace as _replace

    from vmex import optimize as opt
    from vmex.core.freeboundary_lsq import LeastSquaresFreeBoundaryProblem
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    assert rz.LeastSquaresRZFreeBoundaryProblem.trial_certificate == "krylov"
    assert LeastSquaresFreeBoundaryProblem.trial_certificate == "full"
    inp = _vacuum_deck()
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: _replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: _replace(params, phiedge=params.phiedge * (1 + x[0])),
        method="least_squares_rz", three_terms_options=dict(boundary_max_nfev=80, **OPTIONS))
    value, grad = problem.value_and_grad(problem.x0)
    h = 5e-4
    for k in range(2):
        values = []
        for s in (1, -1):
            problem._anchor = {}
            values.append(float(problem.fun(problem.x0 + s * h * np.eye(2)[k])))
        central = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(grad[k], central, rtol=1e-5)
