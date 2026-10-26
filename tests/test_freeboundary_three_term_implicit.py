"""The gauge Newton and the exact derivatives of the three-term free boundary.

:mod:`vmex.core.freeboundary_three_term_implicit`: the implicit-function-theorem
sensitivities of a least-squares stationary point on analytic problems (exact
against Gauss-Newton), and (``full``) the gauge Newton on the vacuum
Landreman--Paul QA case: a cold standalone gauge solve and the single-stage
seed certify on the gauge, and the exact design gradient matches re-solved
differences through the coil field and a nonlinear plasma map.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_three_term as fbt  # noqa: E402
from vmex.core import freeboundary_three_term_implicit as fbi  # noqa: E402

from tests.test_freeboundary_three_term import SMALL, _coil_field, _deck, _small_vacuum, needs_vc  # noqa: E402


def _exact_and_gauss_newton_sensitivity(residual, b, d):
    """``db/dd`` of ``min_b |residual(b, d)|^2`` at a stationary ``b``: exact (curvature terms by AD) and
    Gauss-Newton, both through :func:`fbi._least_squares_sensitivity`."""
    r = np.asarray(residual(b, d))
    J_b = np.asarray(jax.jacfwd(residual, 0)(b, d))
    J_d = np.asarray(jax.jacfwd(residual, 1)(b, d))
    S_bb = np.einsum("i,ijk->jk", r, np.asarray(jax.jacfwd(jax.jacfwd(residual, 0), 0)(b, d)))
    S_bd = np.einsum("i,ijk->jk", r, np.asarray(jax.jacfwd(jax.jacfwd(residual, 0), 1)(b, d)))
    return (fbi._least_squares_sensitivity(J_b, J_d, S_bb, S_bd), fbi._least_squares_sensitivity(J_b, J_d),
            J_b, J_d)


@pytest.mark.parametrize("which", ["state_curvature", "mixed"])
def test_least_squares_sensitivity_needs_the_curvature_terms(which):
    """Two analytic least squares at b = 0, d = 0 where the residual does not vanish: exact db/dd = 1.25, Gauss-Newton
    1.  r = (b - d, sqrt(0.1)(b^2 - 1)) needs S_bb; r = (b - d, (b d - 1)/2) needs only the mixed S_bd.  The exact
    sensitivity matches central differences of the re-solved minimum; the residual multipliers are Gauss-Newton."""
    import scipy.optimize

    if which == "state_curvature":
        def residual(b, d):
            return jnp.stack([b[0] - d[0], jnp.sqrt(0.1) * (b[0] ** 2 - 1.0)])
    else:
        def residual(b, d):
            return jnp.stack([b[0] - d[0], 0.5 * (b[0] * d[0] - 1.0)])
    zero = jnp.zeros(1)
    exact, gauss_newton, J_b, J_d = _exact_and_gauss_newton_sensitivity(residual, zero, zero)
    assert exact[0, 0] == pytest.approx(1.25, rel=1e-12) and gauss_newton[0, 0] == pytest.approx(1.0, rel=1e-12)

    def solved(d):
        return scipy.optimize.least_squares(lambda b: np.asarray(residual(jnp.asarray(b), jnp.asarray([d]))),
                                            np.zeros(1), xtol=1e-15, ftol=1e-15, gtol=1e-15).x[0]

    for h in (1e-2, 1e-3, 1e-4):  # a plateau
        assert (solved(h) - solved(-h)) / (2 * h) == pytest.approx(1.25, rel=1e-3)
    G_b = np.array([[1.0]])  # h = b: the design gradient through the multipliers is Gauss-Newton's
    lam = fbi._residual_multipliers(J_b, G_b)
    assert float(-(lam.T @ J_d)[0, 0]) == pytest.approx(float((G_b @ gauss_newton)[0, 0]), rel=1e-12)


def test_residual_multipliers_equal_the_forward_gauss_newton_sensitivity():
    """The adjoint form of the gradient (multipliers, one per row) equals the forward Gauss-Newton sensitivity."""
    rng = np.random.default_rng(0)
    J_b, J_d, G_b = rng.normal(size=(30, 7)), rng.normal(size=(30, 4)), rng.normal(size=(3, 7))
    lam = fbi._residual_multipliers(J_b, G_b)
    np.testing.assert_allclose(-lam.T @ J_d, G_b @ fbi._least_squares_sensitivity(J_b, J_d), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("residual", [
    lambda b, d: jnp.stack([b - d, jnp.sqrt(0.1) * (b * b - 1.0)]),  # state curvature
    lambda b, d: jnp.stack([b - d, 0.5 * (b * d - 1.0)]),  # mixed term only
])
def test_least_squares_sensitivity_exact_vs_gauss_newton(residual):
    """At the stationary point b = 0 of d = 0, exact db/dd = 1.25; Gauss-Newton (no second-order terms) gives
    1, the error a Gauss-Newton gradient carries at a nonzero residual.  The exact value matches re-solved finite
    differences."""
    import scipy.optimize

    b, d = 0.0, 0.0
    r = np.asarray(residual(b, d))
    J_y = np.asarray(jax.jacfwd(residual, 0)(b, d))[:, None]
    J_d = np.asarray(jax.jacfwd(residual, 1)(b, d))
    hess = jax.hessian(residual, (0, 1))(b, d)  # ((d2/db2, d2/db dd), ...), each per residual component
    S_yy, S_yd = r @ np.asarray(hess[0][0]), r @ np.asarray(hess[0][1])
    assert np.asarray(jax.grad(lambda y: 0.5 * jnp.sum(residual(y, d) ** 2))(b)) == 0.0
    exact = fbi.least_squares_sensitivity(J_y, J_d, S_yy, S_yd)
    np.testing.assert_allclose(exact, [[1.25]], rtol=1e-13)
    np.testing.assert_allclose(fbi.least_squares_sensitivity(J_y, J_d), [[1.0]], rtol=1e-13)
    h = 1e-5
    stationarity = jax.grad(lambda y, dd: 0.5 * jnp.sum(residual(y, dd) ** 2))
    solved = [scipy.optimize.brentq(lambda y, dd=dd: float(stationarity(y, dd)), -0.1, 0.1, xtol=1e-15)
              for dd in (-h, h)]
    assert (solved[1] - solved[0]) / (2 * h) == pytest.approx(1.25, rel=1e-6)


@needs_vc
@pytest.mark.full
def test_exact_gauge_cold_solve_certifies_on_the_gauge(monkeypatch):
    """The production path (gauge labelling, m1_complement="gauged", the gauge Newton, exact gradient) from a cold
    start on the vacuum QA deck at mpol 4: the single-stage seed record is the gauge Newton's, converged and
    certified, its boundary satisfies the gauge to GAUGE_TOL, and its exact design gradient is finite with a small
    true residual; the standalone gauge solve with the same settings lands on the same boundary.  A problem without
    the gauge is refused."""
    monkeypatch.setattr(fbt, "POLISH_FTOL", 1e-5)
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    base = field.currents
    nfp = int(inp.nfp)
    config = vj.ThreeTermConfig(nphi=24, ntheta=24, quadrature=(4 * nfp * 24, 48), trial_ftol=1e-11)

    def build(config):
        return vj.ThreeTermFreeBoundaryProblem.from_loss(
            inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
            field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.05 * x[0])),
            quantities=(opt.mean_iota,), boundary_ftol=1e-5, config=config)

    problem = build(config)
    record = problem._record(problem.x0)
    assert record["newton"]["converged"] and record["newton"]["gauge"] == "condensation"
    assert record["reason"] == "newton-gauge"
    gauge = np.asarray(problem.model.gauge_rows(jnp.asarray(record["x"])))
    assert gauge.size and float(np.max(np.abs(gauge))) <= 10 * fbi.GAUGE_TOL
    assert max(vars(problem.boundary_residual(problem.x0)).values()) < 1e-2
    value, grad = problem.value_and_grad(problem.x0)
    assert np.isfinite(value) and np.all(np.isfinite(grad))
    assert problem.stats["exact_gradient_true_residual"] <= 1e-6
    standalone = vj.solve_free_boundary_three_term(inp, external_field=field, config=config)
    tt = standalone.three_term
    assert standalone.converged and tt.certified and tt.newton["converged"] and tt.hessian is not None
    scale = np.asarray(problem.model.x_scale)
    assert np.linalg.norm((tt.x - record["x"]) / scale) < 1e-6  # one stationary point on the gauge
    problem.close()
    with pytest.raises(ValueError, match="labelling='gauge'"):
        build(replace(config, labelling="none"))


@needs_vc
@pytest.mark.full
def test_plasma_map_derivatives_at_nonzero_offsets(monkeypatch):
    """The exact design gradient differentiates the plasma map itself, phiedge0 (1 + x1/2) + x2^2 / 100: at a
    nonzero offset, not with the updated PHIEDGE as its base, and through x2 although its derivative is zero at x0;
    it agrees with central differences of Newton-converged gauge points to 1e-3."""
    monkeypatch.setattr(fbt, "POLISH_FTOL", 1e-9)
    monkeypatch.setattr(fbt, "POLISH_ROUNDS", 20)
    inp, field = _small_vacuum()
    base = field.currents
    problem = vj.ThreeTermFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(3),
        field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.05 * x[0])),
        plasma_from_parameters=lambda params, x: replace(
            params, phiedge=params.phiedge * (1 + 0.5 * x[1]) + 0.01 * x[2] ** 2),
        boundary_ftol=1e-9, boundary_max_nfev=400, config=vj.ThreeTermConfig(**SMALL))
    assert list(problem._plasma) == [1, 2]
    x = np.array([0.0, 0.1, 0.3])
    _, exact = problem.value_and_grad(x)
    h = 1e-3
    for k in (0, 1, 2):
        e = np.eye(3)[k] * h
        fd = (problem.fun(x + e) - problem.fun(x - e)) / (2 * h)
        assert exact[k] == pytest.approx(fd, rel=1e-3), k
    problem.close()
