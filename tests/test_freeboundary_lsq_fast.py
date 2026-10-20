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


import jax.numpy as jnp  # noqa: E402

from vmex.core import freeboundary_lsq as lsq  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

from tests.test_freeboundary_vc import _coil_field, _deck  # noqa: E402

pytestmark = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")


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
        grads.append(S * model.gradient(xs, aux_s, field, shift=so["shift"]))
    fd = (grads[0] - grads[1]) / (2 * eps)
    np.testing.assert_allclose(S * (so["H"] @ (S * v)), fd, rtol=1e-4, atol=1e-4 * np.max(np.abs(fd)))
    # Hessian-vector blocks: equal to the columns, and a block that exhausts device memory is split in halves
    so_p = model.second_order(x, aux, field, columns=False)
    V = np.eye(x.size)[:3]
    calls = []
    # the checkpointed program, as after a block of stored scans ran out of memory (the fallback order itself is
    # test_hessian_falls_back_to_checkpointed_scans_then_narrower_blocks_on_device_memory)
    model.hessian_remat = True
    columns = model._hessian_columns_remat

    def short_of_memory(*args):
        calls.append(jax.tree.leaves(args[6])[0].shape[0])  # the block's width (dz_cols)
        if len(calls) == 1:
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: out of memory (test)")
        return columns(*args)

    model.hessian_chunk, model._hessian_columns_remat = 4, short_of_memory
    try:
        np.testing.assert_allclose(so_p["hvp"](V), so["H"][:3], rtol=1e-8, atol=1e-10 * np.max(np.abs(so["H"])))
    finally:
        model._hessian_columns_remat = columns
    assert model.hessian_chunk == 2 and calls[0] == 3 and len(calls) == 3  # blocks of two, never one column
    accepted = []
    out = model.solve_boundary(model.params0, field, max_iter=60, verbose=2, callback=accepted.append)
    assert out["converged"], out["status"]
    assert out["history"][-1][2] > 0
    assert len(accepted) >= 1 and np.array_equal(accepted[-1], out["x"])
    # inside an optimizer an unconverged solve is an error, not an answer
    model.require_converged = True
    with pytest.raises(RuntimeError, match="did not converge"):
        model.solve_boundary(model.params0, field, max_iter=1)


def test_entry_rejects_what_it_does_not_support():
    """Stellarator-asymmetric decks and non-callable losses are refused before any solve."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 9), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    with pytest.raises(NotImplementedError, match="lasym"):
        lsq.solve_free_boundary_least_squares(replace(inp, lasym=True), external_field=object())
    with pytest.raises(TypeError, match="callable"):
        lsq.LeastSquaresFreeBoundaryProblem.from_loss(inp, None, np.zeros(1), field_from_parameters=lambda x: x)


class _ToyModel(lsq.LeastSquaresFreeBoundaryModel):
    """The trust-region Newton of :class:`LeastSquaresFreeBoundaryModel` on a cheap nonlinear least squares:
    extended Rosenbrock rows and one nonlinear equality row (last), no equilibrium.  ``fail_at`` makes the
    evaluation at boundaries farther than it from the start fail (no equilibrium), as trials can."""

    def __init__(self, n=4, fail_at=np.inf):
        self.x0, self.x_scale, self.n_boundary, self.n_constraint_rows = np.zeros(n), np.ones(n), n, 1
        self.cfg = type("cfg", (), {"ftol": 1e-24})()
        self.R0, self.boundary_modes = 1.0, [(1, 0)] * n
        self.hessian_chunk, self.certificate_extra = 2, None
        self.require_converged, self.fail_at = False, fail_at

        def rows(x):
            body = [10.0 * (x[i + 1] - x[i] ** 2) for i in range(n - 1)] + [1.0 - x[i] for i in range(n - 1)]
            return jnp.stack(body + [x[0] ** 2 + x[-1] - 0.5])

        self.rows_fn = jax.jit(rows)
        self.jac_fn = jax.jit(jax.jacfwd(rows))
        self.hess_fn = jax.jit(lambda x, w: jax.hessian(lambda z: 0.5 * jnp.sum((rows(z) + w) ** 2))(x))

    def evaluate(self, x, params, field, seed=None, tight=False):
        if np.max(np.abs(x - self.x0)) > self.fail_at:
            return None
        return np.asarray(self.rows_fn(jnp.asarray(x))), (np.array(x), None, np.array(x), None)

    def repeat(self, x, params, field, aux):  # deterministic rows: no noise
        return np.asarray(self.rows_fn(jnp.asarray(x)))

    def _rows(self, state, params, field):
        return np.asarray(self.rows_fn(jnp.asarray(state)))

    def linearize(self, x, aux, field, extra=None):
        return np.asarray(self.jac_fn(jnp.asarray(x))), None, None, aux

    def gradient(self, x, aux, field, shift=None):
        r = np.asarray(self.rows_fn(jnp.asarray(x)))
        return np.asarray(self.jac_fn(jnp.asarray(x))).T @ (r if shift is None else r + shift)

    def second_order(self, x, aux, field, extra=None, columns=True):
        J, r = self.linearize(x, aux, field)[0], self._rows(x, None, None)
        k, S = r.size - 1, np.asarray(self.x_scale)
        nu = self._reduced(S * (J[:k].T @ r[:k]), J[k:] * S[None])[2]
        shift = np.r_[np.zeros(k), nu - r[k:]]
        H = np.asarray(self.hess_fn(jnp.asarray(x), jnp.asarray(shift)))
        out = dict(J=J, r=r, g=J.T @ (r + shift), g_adjoint=J.T @ (r + shift), aux=aux, shift=shift,
                   asymmetry=0.0, adjoint=0.0, seconds={})
        if columns:
            out.update(H=H, C=np.zeros((x.size, 0)))
        else:
            out["hvp"] = lambda V: np.atleast_2d(V) @ H
        return out


def test_trust_region_newton_on_a_nonlinear_constrained_least_squares():
    """The solve's step logic (trust region with bisection, second-order corrections, failed trials, the
    restoration of the constraint row, Krylov and full-column exact models) on extended Rosenbrock rows with a
    nonlinear equality row: it converges to the constrained minimum (stationary on the constraint)."""
    start = np.array([-1.2, 1.0, -0.5, 0.3])
    nearby = np.asarray(_ToyModel().jac_fn(jnp.asarray(start + 0.1)))  # a warm start's (stale) Jacobian
    for model, kw in ((_ToyModel(), {}), (_ToyModel(), dict(jacobian=nearby)), (_ToyModel(fail_at=2.0), {})):
        out = model._newton(None, None, x0=start, max_iter=200, verbose=1, **kw)
        assert out["converged"], out["status"]
        x = out["x"]
        r = np.asarray(model.rows_fn(jnp.asarray(x)))
        J = np.asarray(model.jac_fn(jnp.asarray(x)))
        # the constraint row restored until its Newton step is below the coordinates' round-off step tolerance
        assert abs(r[-1]) / np.linalg.norm(J[-1]) <= np.sqrt(np.finfo(float).eps) * model.R0
        g = J[:-1].T @ r[:-1]  # stationary on the constraint: g in the span of the constraint's gradient
        c = J[-1] / np.linalg.norm(J[-1])
        # to the resolution floor: the solve stops once the remaining gain 1/2 g^T H^-1 g is below 2 sqrt(FTOL) cost,
        # which leaves a gradient of at most sqrt(2 floor |H|) (|H| ~ the Gauss-Newton matrix's norm)
        floor = 2.0 * np.sqrt(model.cfg.ftol) * 0.5 * float(r[:-1] @ r[:-1])
        resolved = np.sqrt(2.0 * floor * np.linalg.norm(J[:-1].T @ J[:-1], 2))
        assert np.linalg.norm(g - (g @ c) * c) <= max(resolved, 1e-12 * max(1.0, np.linalg.norm(g)))
    model = _ToyModel()
    model.certificate_extra = lambda params: None  # full-column certificates
    out = model._newton(None, None, x0=start, max_iter=200)
    assert out["converged"], out["status"]
    with pytest.raises(RuntimeError, match="dependent"):
        model._reduced(np.zeros(4), np.ones((2, 4)))
    assert model._reduced(np.ones(4), None)[0].shape == (4, 4)


def test_trust_region_step_reaches_the_radius_on_near_null_models():
    """The trust-region subproblem's step lies on the radius whenever the Newton step is longer, also on models whose
    smallest eigenvalues are below 1e-14 of the largest (M 16 finite beta: 3e-15 .. 53): a bisection bracket
    relative to the largest eigenvalue returned a step far inside the radius there (|z| 1.3 for radius 13), so the
    radius never grew and the solve crawled along the near-null directions until max_iter."""
    rng = np.random.default_rng(1)
    w = np.sort(np.r_[7e-15, np.logspace(-12, np.log10(52.0), 300)])
    radius = 13.0
    for scale in (1e-12, 1e-10, 1e-6):
        gq = rng.standard_normal(w.size) * scale
        newton = float(np.linalg.norm(gq / w))
        assert newton > radius
        z = lsq._trust_region_solution(w, gq, radius, newton)
        np.testing.assert_allclose(np.linalg.norm(z), radius, rtol=1e-10)
        mu = np.median(-gq / z - w)  # the step is -(w + mu)^-1 gq with one mu >= 0 (its optimality condition)
        assert mu >= 0.0
        np.testing.assert_allclose(z, -gq / (w + mu), rtol=1e-6)
    gq = rng.standard_normal(w.size) * 1e-16  # a Newton step inside the radius is returned as it is
    newton = float(np.linalg.norm(gq / w))
    assert newton < radius
    np.testing.assert_allclose(lsq._trust_region_solution(w, gq, radius, newton), -gq / w)
    w_indefinite = np.r_[-1e-3, w]  # indefinite: on the radius, shifted past the negative eigenvalue
    gq = rng.standard_normal(w_indefinite.size)
    np.testing.assert_allclose(np.linalg.norm(lsq._trust_region_solution(w_indefinite, gq, 2.0, np.inf)), 2.0,
                               rtol=1e-10)


class _CircleModel(_ToyModel):
    """Rows ``(x0 - 2, w x1)`` on the unit circle (the constraint row): the minimum is ``(1, 0)``, and a step along
    the circle's tangent violates the constraint at second order (the Maratos effect of the l1 merit)."""

    def __init__(self, weight):
        super().__init__(n=2)

        def rows(x):
            return jnp.stack([x[0] - 2.0, weight * x[1], x[0] ** 2 + x[1] ** 2 - 1.0])

        self.rows_fn = jax.jit(rows)
        self.jac_fn = jax.jit(jax.jacfwd(rows))
        self.hess_fn = jax.jit(lambda x, w: jax.hessian(lambda z: 0.5 * jnp.sum((rows(z) + w) ** 2))(x))


def test_constraint_restoration_lets_tangential_steps_pass_the_merit_test():
    """From far along the circle the model's tangential steps leave the circle at second order and the l1 merit
    rejects them; restoring the constraint row from the trial lets them pass, so the solve takes far fewer models
    (26 against 36 without the restoration at ``w = 3``) and still converges where the steps are short (``w = 0.1``)."""
    start = np.array([np.cos(1.2), np.sin(1.2)])
    for weight, most in ((3.0, 30), (0.1, 40)):
        out = _CircleModel(weight)._newton(None, None, x0=start, max_iter=200)
        assert out["converged"], out["status"]
        np.testing.assert_allclose(out["x"], [1.0, 0.0], atol=1e-6)
        assert out["njev"] <= most, out["njev"]


class _SphereModel(_ToyModel):
    """Rows ``(x0 - 2, w x1, ..., w x_{n-1})`` on the unit sphere (the constraint row): along the sphere the cheap
    models' curvature is ``w^2`` (J^T J), the Lagrangian's ``w^2 + 1`` (the multiplier 1/2 times the sphere's
    curvature 2), so for small ``w`` Gauss-Newton and BFGS steps overshoot and pass only once restored."""

    def __init__(self, weight, n):
        super().__init__(n=n)

        def rows(x):
            return jnp.concatenate([jnp.stack([x[0] - 2.0]), weight * x[1:], jnp.stack([jnp.sum(x * x) - 1.0])])

        self.rows_fn = jax.jit(rows)
        self.jac_fn = jax.jit(jax.jacfwd(rows))
        self.hess_fn = jax.jit(lambda x, w: jax.hessian(lambda z: 0.5 * jnp.sum((rows(z) + w) ** 2))(x))
        self.exact_models = 0
        model = self._derivative_model

        def counted(exact, *args, **kwargs):
            self.exact_models += bool(exact)
            return model(exact, *args, **kwargs)

        self._derivative_model = counted


def test_cheap_models_missing_the_constraint_curvature_hand_over_to_the_exact_model():
    """Where the constraint rows' curvature dominates, a cheap model whose step passes only once restored, or that
    fails again after a fresh Gauss-Newton model, hands over to the exact (Lagrangian) model.  Before, a restored
    step's ratio (crediting the violation it removed) kept such models (151 models over these starts against 162),
    and escalating on the ratio before the restoration alone cycled on one start (300 models without converging)."""
    rng = np.random.default_rng(0)
    total = 0
    for n in (3, 6):
        for weight in (0.01, 0.03, 0.1):
            start = np.r_[0.3, rng.standard_normal(n - 1)]
            model = _SphereModel(weight, n)
            out = model._newton(None, None, x0=start / np.linalg.norm(start), max_iter=200)
            assert out["converged"], (n, weight, out["status"])
            np.testing.assert_allclose(out["x"], np.r_[1.0, np.zeros(n - 1)], atol=1e-5)
            assert model.exact_models >= 1
            total += out["njev"]
    assert total <= 160, total


def test_trial_step_tolerance_and_previous_exact_certificate():
    """An optimization trial (``trial_step_tolerance``) stops once the Newton step moves the boundary by less than
    the tolerance, certified by the previous exact Hessian with the exact gradient, within that tolerance of the full
    rule's answer and in no more models; a full-precision solve keeps the full rule exactly."""
    start = np.array([-1.2, 1.0, -0.5, 0.3])

    def solve(tolerance=None, full_precision=False):
        model = _ToyModel()
        model.certificate_extra = lambda params: None  # full-column exact models, as an optimization's trials
        model.trial_step_tolerance, model.full_precision = tolerance, full_precision
        return model._newton(None, None, x0=start, max_iter=200)

    full = solve()
    trial = solve(1e-3)
    assert full["converged"] and trial["converged"]
    assert trial["njev"] <= full["njev"]
    assert np.max(np.abs(trial["x"] - full["x"])) <= 1e-3
    # the tolerance's current-value part (relative to the values, over R0): a step of 1e-4 of them passes at 1e-3, a
    # trust-region step never does, nor any step of a full-precision solve
    model = _ToyModel()
    model.n_boundary, model.boundary_modes, model.trial_step_tolerance = 2, [(1, 0)] * 2, 1e-3
    x, step = np.ones(4), np.r_[1e-5, 1e-5, 1e-4, 1e-4]
    assert model._step_converged(step, None, x, newton=True)
    assert not model._step_converged(step, None, x)
    assert not model._step_converged(np.r_[1e-5, 1e-5, 1e-2, 1e-2], None, x, newton=True)
    model.full_precision = True
    assert not model._step_converged(step, None, x, newton=True)
    certified, reference = solve(1e-3, full_precision=True), solve(None, full_precision=True)
    np.testing.assert_array_equal(certified["x"], reference["x"])
    assert certified["njev"] == reference["njev"]


def test_problem_entry_and_finite_beta_rows():
    """Cheap end-to-end coverage: the least-squares single-stage problem in vacuum (design gradient along the coil
    currents against central differences of cold trials), the solve entry through ``method="least_squares"``
    with an ``initial_boundary``, and the finite-beta jump rows (band-limited edge field, virtual casing) at a
    small grid."""
    from vmex import optimize as opt
    from vmex.core import freeboundary_vc as fvc
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 9), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    grid = dict(nphi=16, ntheta=16, quadrature=(64, 32))
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        boundary_condition="three_terms", three_terms_options=dict(method="least_squares", **grid))
    assert isinstance(problem, lsq.LeastSquaresFreeBoundaryProblem)
    assert problem.model.a_priori_floor is False  # an optimization converges to the measured repeatability
    value, grad = problem.value_and_grad(problem.x0)
    h = 2e-3
    values = []
    for s in (1, -1):
        problem._anchor = {}
        values.append(problem.fun(problem.x0 + s * h * np.eye(2)[1]))
    np.testing.assert_allclose(grad[1], (values[0] - values[1]) / (2 * h), rtol=1e-2, atol=1e-8)
    fit = fvc.solve_free_boundary_three_terms(inp, external_field=coils, method="least_squares", initial_boundary=inp,
                                              **grid)
    assert fit.success and isinstance(fit.model, lsq.LeastSquaresFreeBoundaryModel)
    beta = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 13)
    model = lsq.LeastSquaresFreeBoundaryModel(replace(beta, lfreeb=False), **grid)
    assert not model.vacuum
    # the rows' virtual-casing plan is the base model's (planned once for every band): the plan of the deck's
    # surface with the model's digits and quadrature
    surface = fvc._surface_data(model.fixed, model.cfg, model.seed[0], model.params0, grid["nphi"], grid["ntheta"])
    planned = vc.plan_vc_precision(surface, digits=model.digits, quad_nt=grid["quadrature"][0],
                                   quad_np=grid["quadrature"][1])
    for name in ("quad_nt", "quad_np", "patch_dim0", "patch_idx"):
        assert np.array_equal(np.asarray(getattr(model.precision, name)), np.asarray(getattr(planned, name))), name
    rows, aux = model.evaluate(np.array(model.x0), model.params0,
                               _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"), tight=True)
    assert np.all(np.isfinite(rows)) and rows.size == 3 * 16 * 16 + model.n_constraint_rows


def test_jump_rows_use_the_whole_interior_field():
    """The finite-beta rows are the jump of the whole interior edge field (``interior_band=False``, the default):
    equal to :func:`_jump_rows` without the band limit, different from the band-limited rows (``interior_band``),
    and a re-referenced model keeps the setting."""
    grid = dict(nphi=16, ntheta=16, quadrature=(64, 32))
    beta = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 13), lfreeb=False)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    rows = {}
    for band in (False, True):
        model = lsq.LeastSquaresFreeBoundaryModel(beta, interior_band=band, **grid)
        r, aux = model.evaluate(np.array(model.x0), model.params0, field, tight=True)
        rows[band] = np.asarray(r)[:-model.n_constraint_rows]
        state, _, params, _ = aux
        direct = lsq._jump_rows(model.fixed, state, field, runtime=model._im.runtime_from_params(params, model.cfg),
                                nphi=grid["nphi"], ntheta=grid["ntheta"], digits=model.digits,
                                precision=model.precision, B0=model._consts["B0"], band_limited=band)
        np.testing.assert_allclose(rows[band], np.asarray(direct), rtol=1e-10, atol=1e-14)
        assert model.rereferenced(np.array(model.x0)).interior_band is band
    # the band limit removes the interior field's out-of-band part: the rows differ
    assert np.max(np.abs(rows[False] - rows[True])) > 1e-6 * np.max(np.abs(rows[False]))


def test_hessian_falls_back_to_checkpointed_scans_then_narrower_blocks_on_device_memory():
    """The exact Hessian's blocks run with the singular quadratures' scans stored; a block that runs out of device
    memory is repeated with them checkpointed (the same columns), and only then with narrower blocks."""
    n = 6
    model = lsq.LeastSquaresFreeBoundaryModel.__new__(lsq.LeastSquaresFreeBoundaryModel)
    model.x0, model.n_constraint_rows, model.hessian_chunk, model.hessian_remat = np.zeros(n), 0, 4, False
    model._last_second_order = None
    H = np.arange(n * n, dtype=float).reshape(n, n)
    H = H + H.T
    tangents = {"z": jnp.eye(n)}
    model.linearize = lambda x, aux, field, extra=None: (np.eye(n), tangents, tangents, (None, None, None, None))
    model._rows = lambda state, params, field: np.ones(n)
    model._multiplier = lambda *a: (None, None, 0.0)
    model._project = lambda dp, grad: np.zeros(n)
    calls = []

    def program(remat):
        def run(state, mask, params, field, shift, lam, dz_cols, dp_cols, dz_b, dp_b):
            width = dz_cols["z"].shape[0]
            calls.append((remat, width))
            if len(calls) <= 2:  # the stored scans, then the first checkpointed width, run out of memory
                raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: out of memory")
            return np.asarray(dz_cols["z"]) @ H
        return run

    model._hessian_columns, model._hessian_columns_remat = program(False), program(True)
    so = model.second_order(np.zeros(n), (None, None, None, None), None)
    assert calls[:3] == [(False, 4), (True, 4), (True, 2)]
    assert model.hessian_remat and model.hessian_chunk == 2
    assert np.allclose(so["H"], H)
    model.hessian_chunk, calls[:] = 2, [(None, 0)] * 2  # at the narrowest checkpointed block an error is raised
    model._last_second_order = None
    model._hessian_columns_remat = lambda *a: (_ for _ in ()).throw(jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED"))
    with pytest.raises(jax.errors.JaxRuntimeError):
        model.second_order(np.ones(n), (None, None, None, None), None)


def test_trials_stop_the_host_solve_at_host_ftol_and_full_precision_at_the_deck():
    """Tight trial solves stop the host solve at ``host_ftol`` and refine to the residual floor; full-precision solves
    (a certified answer) use the deck's FTOL; both land on the same root (rows within the equilibria's repeatability);
    a model without a looser host config solves its trials with the deck's."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 9), phiedge=-0.025, curtor=0.0, am=np.zeros(21),
                  ftol_array=np.array([1e-14]))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    model = lsq.LeastSquaresFreeBoundaryModel(replace(inp, lfreeb=False), nphi=16, ntheta=16, quadrature=(64, 32),
                                              host_ftol=1e-12)
    assert model.host_ftol == 1e-12 and model.cfg_host is not None and model.cfg_host.ftol == 1e-12
    assert model.cfg.ftol == 1e-14 and model.converged_ftol == 1e-14  # the floor stays the deck's
    assert model._tight_config() is model.cfg_host
    x = np.array(model.x0)
    x[0] += 1e-3 * model.x_scale[0]
    rows_host, aux = model.evaluate(x, model.params0, field, tight=True)
    model.full_precision = True
    assert model._tight_config() is model.cfg
    rows_deck = model.repeat(x, model.params0, field, aux)
    np.testing.assert_allclose(rows_host, rows_deck, rtol=0, atol=1e-8 * np.max(np.abs(rows_deck)))
    other = lsq.LeastSquaresFreeBoundaryModel.__new__(lsq.LeastSquaresFreeBoundaryModel)
    other.cfg, other.full_precision, other.cfg_host = model.cfg, False, None  # host_ftol at or under the deck's
    assert other._tight_config() is model.cfg
    assert lsq.HOST_FTOL == 1e-13


class _HumpModel(_ToyModel):
    """Rows ``(A - x0^2, 2 (x1 - x0 / 10), x2 - 0.3)`` and the constraint row ``x2 + x1^2 - 0.3``: the minimum is
    near ``x0 = sqrt(A)``, and from ``x0 ~ 0`` the exact (Lagrangian) model is indefinite (``6 x0^2 < 2 A``) for a
    long way.  Boundaries with ``|x0| > fail_at`` have no equilibrium (the evaluation fails), as a trial whose free
    boundary lies past a region without equilibria (f97-qa-sl-c3, 10-19)."""

    def __init__(self, A, fail_at=np.inf):
        super().__init__(n=3)
        self.fail_at = fail_at

        def rows(x):
            return jnp.stack([A - x[0] ** 2, 2.0 * (x[1] - 0.1 * x[0]), x[2] - 0.3, x[2] + x[1] ** 2 - 0.3])

        self.rows_fn = jax.jit(rows)
        self.jac_fn = jax.jit(jax.jacfwd(rows))
        self.hess_fn = jax.jit(lambda x, w: jax.hessian(lambda z: 0.5 * jnp.sum((rows(z) + w) ** 2))(x))

    def evaluate(self, x, params, field, seed=None, tight=False):
        if abs(x[0]) > self.fail_at:
            return None
        return np.asarray(self.rows_fn(jnp.asarray(x))), (np.array(x), None, np.array(x), None)


def test_trust_radius_collapse_is_no_convergence():
    """A trust radius that collapses on trials without equilibrium is no convergence: trial steps below the rows'
    noise certify only once the model's whole Newton step was evaluated.  Here the minimum (x0 = 10) lies past the
    region with equilibria (|x0| <= 5); the solve used to report "converged: trial steps below the rows' noise" at
    its edge, a point whose reduced gradient is far from zero (a non-root)."""
    start = np.array([1e-3, 0.0, 0.3])
    model = _HumpModel(100.0, fail_at=5.0)
    out = model._newton(None, None, x0=start, max_iter=200)
    assert not out["converged"], out["status"]
    assert out["status"].startswith("stalled: the trust radius collapsed"), out["status"]
    x = out["x"]
    r, J = np.asarray(model.rows_fn(jnp.asarray(x))), np.asarray(model.jac_fn(jnp.asarray(x)))
    c = J[-1] / np.linalg.norm(J[-1])
    g = J[:-1].T @ r[:-1]
    assert np.linalg.norm(g - (g @ c) * c) > 10.0  # not stationary on the constraint
    # with every equilibrium available, the same start converges to the minimum
    out = _HumpModel(100.0)._newton(None, None, x0=start, max_iter=200)
    assert out["converged"], out["status"]
    np.testing.assert_allclose(out["x"][0], 10.0, rtol=1e-3)


def test_warm_start_outside_the_newton_basin_hands_over():
    """A warm start (``_newton_boundary(undamped=True)``, an optimizer's trial) stops, not converged, at the second
    exact model in a row without its full Newton step (indefinite, or cut by the radius) instead of crawling: 5
    models against 41 before the radius collapsed, the trial's next start then takes over.  Inside the basin the
    warm solve is the solve without the test, to the last bit; a cold solve (``solve_boundary``) never stops so."""
    start = np.array([1e-3, 0.0, 0.3])
    model = _HumpModel(100.0, fail_at=5.0)
    J = np.asarray(model.jac_fn(jnp.asarray(start)))
    crawl = model._newton(None, None, x0=start, max_iter=200, jacobian=J)
    warm = _HumpModel(100.0, fail_at=5.0)._newton(None, None, x0=start, max_iter=200, jacobian=J, basin=True)
    assert not warm["converged"] and warm["status"].startswith("stalled: outside the Newton basin"), warm["status"]
    assert warm["njev"] <= 6 < crawl["njev"], (warm["njev"], crawl["njev"])
    # far from the minimum (x0 = 100) with every equilibrium available: positive-definite exact models whose steps
    # the radius cuts short, two in a row
    far = _HumpModel(1e4)
    far = far._newton(None, None, x0=start, max_iter=300, jacobian=np.asarray(far.jac_fn(jnp.asarray(start))),
                      basin=True)
    assert not far["converged"] and "radius cut short" in far["status"], far["status"]
    assert far["outside_basin"]
    answer = _HumpModel(100.0)._newton(None, None, x0=start, max_iter=200)["x"]
    for shift in (1e-3, 1e-2, 1e-1):  # near the answer: in the basin, the same solve with or without the test
        near = answer + shift * np.array([1.0, -1.0, 0.5])
        J = np.asarray(model.jac_fn(jnp.asarray(answer)))  # the anchor's Jacobian
        a = _HumpModel(100.0)._newton(None, None, x0=near, max_iter=200, jacobian=J)
        b = _HumpModel(100.0)._newton(None, None, x0=near, max_iter=200, jacobian=J, basin=True)
        assert a["converged"] and b["converged"], (a["status"], b["status"])
        np.testing.assert_array_equal(a["x"], b["x"])
        assert a["njev"] == b["njev"]
    calls = []
    model = _HumpModel(100.0)
    model._newton = lambda params, field, **kw: calls.append(kw["basin"]) or dict(converged=True, status="")
    model._newton_boundary(None, None, x0=start, jacobian=J, undamped=True)
    model._newton_boundary(None, None, x0=start, jacobian=J)
    model.solve_boundary(None, None, x0=start, continuation=False)
    assert calls == [True, False, False]


def test_warm_start_at_its_answer_stays_in_the_newton_basin():
    """A warm start reaching its answer is not stopped as outside the Newton basin.  The basin test counts exact
    models without their full Newton step.  A cheap model's rejected step cuts the radius the next exact model starts
    from, and a full step can fail the merit test at second order in the constraint rows (Maratos); neither is the
    exact model failing.  An exact model after such a cut starts from its Newton step, and one whose Newton step has
    an equilibrium and lowers the merit is in the basin.  Before, these sphere solves (Gauss-Newton and BFGS missing
    the constraint curvature) stopped "outside the Newton basin" on 18 of 24 starts, most at the answer itself
    (f98-qh-al-c4, 10-19: both warm starts of a QH trial stopped so, at rho 0.993 for a step cut to 1/10 of its
    Newton step by the radius of the quasi-Newton step before it)."""
    rng = np.random.default_rng(0)
    for n in (3, 6):
        for weight in (0.01, 0.03, 0.1, 0.3):
            for radius in (None, 1e-3, 1e-5):
                start = np.r_[0.3, rng.standard_normal(n - 1)]
                start = start / np.linalg.norm(start)
                model = _SphereModel(weight, n)
                J = np.asarray(model.jac_fn(jnp.asarray(start)))
                out = model._newton(None, None, x0=start, max_iter=200, jacobian=J, basin=True, radius=radius)
                assert out["converged"], (n, weight, radius, out["status"])
                np.testing.assert_allclose(out["x"], np.r_[1.0, np.zeros(n - 1)], atol=1e-5)
                assert out["njev"] <= 40, (n, weight, radius, out["njev"])


class _InflectionModel(_ToyModel):
    """A large constant row and ``x0^2 - 1`` (minimum at ``x0 = 1``, an inflection of the cost at ``1/sqrt 3``), the
    constraint row ``x1 - 1/2``, at FTOL 1e-6: from ``x0 = 0.6`` the exact model's whole predicted gain (1.8) is below
    the a-priori floor 2 sqrt(FTOL) cost (10), though the cost still falls by that much on the way to the minimum."""

    def __init__(self):
        super().__init__(n=2)
        self.cfg = type("cfg", (), {"ftol": 1e-6})()

        def rows(x):
            return jnp.stack([100.0 + 0.0 * x[0], x[0] ** 2 - 1.0, x[1] - 0.5])

        self.rows_fn = jax.jit(rows)
        self.jac_fn = jax.jit(jax.jacfwd(rows))
        self.hess_fn = jax.jit(lambda x, w: jax.hessian(lambda z: 0.5 * jnp.sum((rows(z) + w) ** 2))(x))


def test_optimization_solves_converge_to_the_measured_repeatability():
    """The a-priori floor 2 sqrt(FTOL) cost accepts a point that is no minimum where the model is nearly flat; an
    optimization's model (``a_priori_floor = False``, set by LeastSquaresFreeBoundaryProblem) converges to the measured
    repeatability, to the full-precision answer: an accepted trial is the root its certification finds (f97-qa-al c1
    step 10: trial QA 0.135246 against 0.13334 certified)."""
    start = np.array([0.6, 0.5])
    floor = _InflectionModel()._newton(None, None, x0=start, max_iter=100)
    assert floor["status"] == "converged: gain below the cost's repeatability" and floor["x"][0] == 0.6
    model = _InflectionModel()
    model.a_priori_floor = False
    trial = model._newton(None, None, x0=start, max_iter=100)
    model = _InflectionModel()
    model.full_precision = True
    certified = model._newton(None, None, x0=start, max_iter=100)
    assert trial["converged"] and certified["converged"]
    np.testing.assert_allclose(trial["x"][0], 1.0, atol=1e-5)
    np.testing.assert_array_equal(trial["x"], certified["x"])


def test_trial_fails_at_once_when_every_warm_start_is_outside_the_newton_basin():
    """``ThreeTermsFreeBoundaryProblem._solve``: when both warm starts of a trial stop outside the Newton basin, the
    cold start and the re-referenced deck (which would descend the same soft valley from farther, 45 min in
    f97-qa-sl-c3) are skipped and the trial fails at once (the optimizer rejects it); a warm start that stalls for
    another reason still hands over to them."""
    calls = []

    class Model:
        deck = None

        def __init__(self, outside):
            self.outside = outside

        def _newton_boundary(self, params, field, **kw):
            calls.append("warm")
            return dict(converged=False, nfev=3, outside_basin=self.outside, status="")

        def solve_boundary(self, params, field, **kw):
            calls.append("cold")
            return dict(converged=False, nfev=1, status="")

    for outside, expected in ((True, ["warm", "warm"]), (False, ["warm", "warm", "cold", "reference"])):
        problem = object.__new__(lsq.LeastSquaresFreeBoundaryProblem)
        problem._cache, problem.stats, problem._seed_start = {}, dict(trials=0, boundary_evaluations=0, failed=0), None
        problem._params_at, problem.field = (lambda x: None), (lambda x: None)
        problem._plasma, problem.boundary_max_nfev = np.zeros(0, dtype=int), 10
        problem._field_jvp = lambda *args: np.zeros(2)
        problem._anchor = dict(x=np.zeros(3), b=np.zeros(2), J_b=np.eye(2), J_p=np.zeros((2, 0)), state=None,
                               params=None)
        problem._rereference = lambda x, anchor: calls.append("reference")
        problem.model = Model(outside)
        calls.clear()
        assert lsq.ThreeTermsFreeBoundaryProblem._solve(problem, np.ones(3)) is None
        assert calls == expected, calls
        assert problem.stats.get("outside_basin", 0) == int(outside)


def test_trial_cold_start_crawling_outside_the_newton_basin_stops():
    """An optimization trial's cold start (``basin=COLD_TRIAL_BASIN``) stops once that many exact models in a row lack
    their Newton step, instead of crawling at minutes a model (f98-qh-sl-c5, 10-20: an hour on one trial's cold start,
    5 such models in a row and still infeasible); a cold start that converges is unchanged."""
    from vmex.core.freeboundary_vc import COLD_TRIAL_BASIN

    start = np.array([1e-3, 0.0, 0.3])
    crawl = _HumpModel(100.0, fail_at=5.0)._newton(None, None, x0=start, max_iter=300)
    stop = _HumpModel(100.0, fail_at=5.0)._newton(None, None, x0=start, max_iter=300, basin=COLD_TRIAL_BASIN)
    assert not crawl["converged"] and not stop["converged"]
    assert stop["outside_basin"] and stop["njev"] <= 10 < crawl["njev"], (stop["njev"], crawl["njev"])
    a = _HumpModel(100.0)._newton(None, None, x0=start, max_iter=300)
    b = _HumpModel(100.0)._newton(None, None, x0=start, max_iter=300, basin=COLD_TRIAL_BASIN)
    assert a["converged"] and b["converged"]
    np.testing.assert_array_equal(a["x"], b["x"])
    rng = np.random.default_rng(0)
    for n in (3, 6):
        for weight in (0.01, 0.03, 0.1, 0.3):
            x0 = np.r_[0.3, rng.standard_normal(n - 1)]
            out = _SphereModel(weight, n)._newton(None, None, x0=x0 / np.linalg.norm(x0), max_iter=200,
                                                  basin=COLD_TRIAL_BASIN)
            assert out["converged"], (n, weight, out["status"])
